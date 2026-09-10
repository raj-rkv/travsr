//! Per-language toolchain access for Phase B sandboxes.
//!
//! Build-tool analyzers (`scip-go`, `scip-java`, …) drive the language's real
//! build to resolve symbols. To do that they must read their language's module /
//! build caches and see the toolchain's environment variables. The default
//! `Standard` sandbox (ADR-017 Rule 1) `env_clear`s everything and denies reads
//! outside the repo + scratch, so the analyzer resolves **zero** packages and
//! emits an empty index (observed: scip-go indexes `[0/0]` packages under the
//! sandbox vs 244 implementations unsandboxed).
//!
//! This module computes the **minimal** extra read/write paths + env each
//! language's analyzer needs, so the sandbox can grant exactly those and nothing
//! more. Languages with no out-of-repo toolchain needs (e.g. builtins) return an
//! empty grant set — a no-op.
//!
//! Extending to a new language = add a match arm with that toolchain's caches +
//! env (e.g. java → `~/.gradle`, `~/.m2`, `JAVA_HOME`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Extra sandbox grants a language's Phase B analyzer needs beyond the repo +
/// scratch. All paths are host paths; the sandbox layer canonicalizes and maps
/// them per-platform (sandbox-exec subpaths on macOS, bwrap binds on Linux).
#[derive(Debug, Clone, Default)]
pub struct ToolchainAccess {
    /// Directories the analyzer must be able to **read** (toolchain, module cache).
    pub read_paths: Vec<PathBuf>,
    /// Directories the analyzer must be able to **write** (build cache).
    pub write_paths: Vec<PathBuf>,
    /// Directories the analyzer must be able to **execute** binaries from — the
    /// genuine toolchain *bin* dirs it shells out to (GOROOT/bin, GOPATH/bin,
    /// JAVA_HOME/bin, the sbt/dotnet bin dirs). These are ADDITIVE over
    /// `read_paths`: an exec dir is also listed in `read_paths`, so platforms that
    /// already grant reads-with-execute (Linux bwrap binds, the macOS profile) are
    /// unchanged, while the Windows AppContainer — which grants read WITHOUT
    /// execute by default — adds the execute right only here.
    ///
    /// Deliberately EXCLUDES module / package caches (GOMODCACHE, ~/.m2, ~/.ivy2,
    /// ~/.nuget): those are populated over the network during dependency
    /// resolution, so granting execute there would let a planted binary run.
    /// Least-privilege: execute exactly the toolchain, read the caches. The
    /// residual risk equals the existing `~/.travsr/bin` execute grant — both are
    /// user-write-only trusted install dirs.
    pub exec_paths: Vec<PathBuf>,
    /// Env vars to pass through into the otherwise-cleared sandbox env.
    pub env: Vec<(String, String)>,
}

/// Strip the Windows extended-length verbatim prefix (`\\?\`, `\\?\UNC\`) from a
/// path. `std::fs::canonicalize` and the daemon's `repo_root` both carry it on
/// Windows, but the analyzer sidecars (scip-dotnet's `System.Uri`, sbt, …) and
/// env vars like `DOTNET_ROOT` cannot parse it. No-op on unix and on any
/// already-clean path.
pub(crate) fn strip_windows_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s
        .strip_prefix(r"\\?\UNC\")
        .map(|r| format!(r"\\{r}"))
        .or_else(|| s.strip_prefix(r"\\?\").map(|r| r.to_string()))
    {
        Some(stripped) => PathBuf::from(stripped),
        None => p,
    }
}

/// One repo-relative path a Phase B analyzer must be able to write, with the
/// rest of the repo root kept read-only (ADR-017 Rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoWrite {
    /// A directory subtree the analyzer writes into (created on the host if
    /// absent so the writable bind has a mount source).
    Dir(&'static str),
    /// A single file the analyzer generates (created empty on the host if absent).
    File(&'static str),
}

impl RepoWrite {
    pub fn subpath(&self) -> &'static str {
        match self {
            RepoWrite::Dir(p) | RepoWrite::File(p) => p,
        }
    }
}

/// The exact repo-relative subpaths `language`'s Phase B analyzer must write,
/// keeping the rest of the repo root read-only. Empty for every language whose
/// analyzer writes only outside the repo; non-empty for the ones that drive the
/// project's own build tool (scala, java, kotlin, csharp) and for php, whose
/// analyzer hardcodes its output path.
///
/// scala: `sbt compile` writes build outputs to `target/` inside the project
/// (sbt's layout has no out-of-tree build option): at the root, under
/// `project/`, and once per platform in an sbt-crossproject tree. And
/// SemanticDB is enabled via a settings file (`.travsr-semanticdb.sbt`) the
/// wrapper drops alongside `build.sbt`. Narrowing to these subpaths — rather than
/// the whole repo root — means a hostile `build.sbt` executed by sbt during
/// indexing cannot rewrite arbitrary repo files. Every other language's build
/// tool writes only to its own cache dir (`~/.gradle`, `~/go/pkg`, …), already
/// covered by `ToolchainAccess::write_paths`.
pub fn repo_write_subpaths(language: &str) -> &'static [RepoWrite] {
    match language {
        "scala" => &[
            RepoWrite::Dir("target"),
            RepoWrite::Dir("project/target"),
            // sbt's meta-build of the meta-build. Present in a real crossproject
            // tree and written by the same `sbt compile`.
            RepoWrite::Dir("project/project/target"),
            // #832 moved the scala sidecar to sbt-crossproject support on the
            // read side (`find_semanticdb_files` walks `target/` at any depth,
            // and its own test asserts `jvm/target` and `native/target`), but
            // this grant stayed on the single-module layout. A crossproject
            // build writes SemanticDB per platform: on the pinned
            // scala-parser-combinators fixture all 150 `.semanticdb` files land
            // under js/jvm/native and NONE under the granted `target/`. On macOS
            // scala runs under the `Elevated` policy, which skips sandbox-exec
            // entirely, so the mismatch is invisible there; on Linux the repo
            // root is a `--ro-bind` and those writes take EROFS.
            RepoWrite::Dir("js/target"),
            RepoWrite::Dir("jvm/target"),
            RepoWrite::Dir("native/target"),
            RepoWrite::File(".travsr-semanticdb.sbt"),
        ],
        // scip-php has no `--output`: it hardcodes `index.scip` relative to its
        // working directory, which has to be the repo for it to find
        // composer.json at all. Without this grant the sidecar's write is denied
        // and `file_put_contents` returns false with a zero exit status, i.e. a
        // silent empty index rather than a reported failure. The sidecar moves
        // the file into scratch and removes it, so nothing survives the run.
        "php" => &[RepoWrite::File("index.scip")],
        // These three drive the project's own build tool, which writes its
        // output into the project. Same trade scala already makes, same shape
        // of grant, and without it their Phase B is dead on Linux: the repo root
        // is a `--ro-bind`, so javac and the compiler plugin take EROFS and the
        // language indexes to nothing. All three are `RequiresElevated`, which
        // on macOS skips the sandbox entirely, which is why this was invisible.
        //
        // Verified on Linux for java/maven (bwrap, arm64): with `target/` bound
        // over a read-only root, writing new files, deleting files inside it and
        // deleting its CONTENTS all succeed, and a write outside the grant is
        // still denied. Only removing the `target` directory itself fails, and
        // the java sidecar passes `-Dmaven.clean.failOnError=false` so `clean`
        // degrades that to a warning: contents are still cleared, javac still
        // runs, BUILD SUCCESS. Gradle needs no equivalent flag because scip-java
        // drives it through `scipCompileAll` and never runs `clean`.
        //
        // kotlin and csharp were measured the same way, on the same host.
        // csharp needed nothing beyond `obj/` and `bin/`: `dotnet build` writes
        // its assembly and succeeds. kotlin needed one directory that guessing
        // would have missed, and did miss: the Kotlin Gradle Plugin opens a
        // build session under `<project>/.kotlin/sessions/`, so without it
        // `compileKotlin` dies with
        // `FileSystemException: .kotlin/sessions/....salive: Read-only file
        // system` while every other grant is in place. With it, the same build
        // compiles. `.kotlin` is deliberately NOT on java's list: a pure java
        // gradle build never loads that plugin, and a mixed java/kotlin repo
        // that turns out to need it is one line, added when it is observed
        // rather than guessed at now.
        "java" => &[
            RepoWrite::Dir("target"),
            RepoWrite::Dir("build"),
            RepoWrite::Dir(".gradle"),
        ],
        "kotlin" => &[
            RepoWrite::Dir("build"),
            RepoWrite::Dir(".gradle"),
            RepoWrite::Dir(".kotlin"),
        ],
        "csharp" => &[RepoWrite::Dir("obj"), RepoWrite::Dir("bin")],
        _ => &[],
    }
}

/// Whether any existing component of `root`/`subpath` is a symlink.
///
/// Checked component by component, not just at the leaf: a link anywhere on the
/// path (`target` -> `/`, then `target/x`) escapes the repo just as well. A
/// component that does not exist yet is fine, since it is created as a real
/// dir/file immediately after and the result is re-stat'd before it is granted.
///
/// Shared by every platform that materialises a repo-write grant: the host
/// creates the path as the user, UNSANDBOXED, so a repo shipping php's
/// `index.scip` as a link to `~/.ssh/authorized_keys` would otherwise get that
/// target created and then handed to the sandbox writable.
pub fn grant_path_has_symlink(root: &std::path::Path, subpath: &str) -> bool {
    let mut p = root.to_path_buf();
    for component in std::path::Path::new(subpath).components() {
        p.push(component);
        match std::fs::symlink_metadata(&p) {
            Ok(md) => {
                if md.file_type().is_symlink() {
                    return true;
                }
            }
            // Does not exist yet: nothing to follow.
            Err(_) => return false,
        }
    }
    false
}

/// Compute the toolchain grants for a language's Phase B analyzer.
/// Empty for languages with no out-of-repo toolchain needs.
pub fn toolchain_access(language: &str) -> ToolchainAccess {
    match language {
        "go" => go_access(),
        "dart" => dart_access(),
        "java" => java_access(),
        "kotlin" => kotlin_access(),
        "scala" => scala_access(),
        "php" => php_access(),
        "csharp" => csharp_access(),
        "ruby" => ruby_access(),
        "objectivec" => objc_access(),
        "swift" => swift_access(),
        "rust" => rust_access(),
        "typescript" | "javascript" => typescript_access(),
        "python" => python_access(),
        // scip-clang reads compile_commands.json + source tree only; no module
        // caches, no network. System headers (/usr, /Library/Developer, /opt/homebrew)
        // are already readable via the base macOS sandbox profile and equivalent
        // bwrap binds on Linux. The only sandbox requirement is a writable scratch
        // dir, which is now injected via InvokeRequest::scratch.
        "c" | "cpp" => ToolchainAccess::default(),
        _ => ToolchainAccess::default(),
    }
}

/// `dart run emit.dart` needs:
///   - `~/.pub-cache/` (read) — package:analyzer and its transitive deps
///   - The emitter script's package root (read) — emit.dart + .dart_tool/
///
/// The Dart SDK itself is under /opt/homebrew (macOS homebrew) which the
/// Standard sandbox already allows via `(subpath "/opt/homebrew")`.
///
/// Emitter package root is discovered by finding `travsr-lang-dart` on PATH
/// and computing `<binary-dir>/../share/travsr-lang-dart/` (installed layout).
/// Falls back to the dev path `<binary-dir>/../../packages/dart-scip-emitter/`.
fn dart_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();

    // Dart pub package cache — required by package:analyzer at runtime.
    // travsr-dart-index-emitter (AOT binary) resolves the Dart SDK root as
    // ~/.travsr/ (one level above its own ~/.travsr/bin/). Grant read on
    // ~/.travsr/lib (symlinked to the real SDK lib dir) and ~/.travsr/version
    // so FolderBasedDartSdk can initialise inside the sandbox.
    if let Some(home) = home() {
        let pub_cache = home.join(".pub-cache");
        tracing::debug!(path = %pub_cache.display(), exists = pub_cache.exists(), "dart_access: pub-cache grant");
        read_paths.push(pub_cache);

        let travsr_lib = home.join(".travsr").join("lib");
        if travsr_lib.exists() {
            tracing::debug!(path = %travsr_lib.display(), "dart_access: granting read on ~/.travsr/lib (SDK lib symlink for AOT emitter)");
            read_paths.push(travsr_lib);
        }
        let travsr_version = home.join(".travsr").join("version");
        if travsr_version.exists() {
            tracing::debug!(path = %travsr_version.display(), "dart_access: granting read on ~/.travsr/version (SDK version for AOT emitter)");
            read_paths.push(travsr_version);
        }
    }

    // The dart binary needs to read its own SDK snapshot files at startup
    // (e.g. dartdev_aot.dart.snapshot in <sdk>/bin/snapshots/).
    // `dart --version` works without these (version is baked into the VM) but
    // `dart run` fails if the dartdev snapshot directory is sandbox-denied.
    // Resolve symlinks so we grant the REAL bin/ dir, not just the symlink dir.
    let dart_sdk_bin = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|dir| dir.join("dart"))
        .find(|p| p.is_file())
        .and_then(|dart_bin| {
            let real = std::fs::canonicalize(&dart_bin)
                .ok()
                .and_then(|r| r.parent().map(|p| p.to_path_buf()));
            tracing::debug!(
                symlink = %dart_bin.display(),
                real_bin_dir = ?real,
                "dart_access: dart SDK bin dir"
            );
            real.or_else(|| dart_bin.parent().map(|p| p.to_path_buf()))
        });
    if let Some(sdk_bin) = dart_sdk_bin {
        tracing::debug!(path = %sdk_bin.display(), "dart_access: granting read on dart SDK bin dir (snapshots)");
        read_paths.push(sdk_bin);
    }

    // Emitter script directory: discover by resolving travsr-lang-dart on PATH.
    // The sidecar's emitter_path() checks the same relative locations.
    let dart_bin = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|dir| dir.join("travsr-lang-dart"))
        .find(|p| p.is_file());

    tracing::debug!(found = dart_bin.is_some(), path = ?dart_bin, "dart_access: travsr-lang-dart binary on PATH");

    let emitter_dir = dart_bin
    .and_then(|bin| bin.parent().map(|p| p.to_path_buf()))
    .and_then(|bin_dir| {
        // Installed: <prefix>/share/travsr-lang-dart/
        let installed = bin_dir.parent().map(|prefix| {
            prefix.join("share").join("travsr-lang-dart")
        });
        if let Some(ref p) = installed {
            tracing::debug!(path = %p.display(), exists = p.exists(), "dart_access: checking installed emitter dir");
            if p.exists() {
                return installed;
            }
        }
        // Dev: <bin_dir>/../../packages/dart-scip-emitter/
        let dev = bin_dir
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("packages").join("dart-scip-emitter"));
        if let Some(ref p) = dev {
            tracing::debug!(path = %p.display(), exists = p.exists(), "dart_access: checking dev emitter dir");
        }
        dev
    });

    tracing::debug!(emitter_dir = ?emitter_dir, "dart_access: resolved emitter dir");

    if let Some(dir) = emitter_dir {
        if dir.exists() {
            tracing::debug!(path = %dir.display(), "dart_access: granting read on emitter dir");
            read_paths.push(dir);
        } else {
            tracing::debug!(path = %dir.display(), "dart_access: emitter dir does not exist, not granted");
        }
    }

    // TRAVSR_DART_EMITTER: the sidecar binary (travsr-lang-dart) finds the AOT
    // emitter via current_exe() → sibling lookup. When the sidecar is the
    // npm-installed binary (~/.nvm/.../bin/travsr-lang-dart), the sibling path
    // is ~/.nvm/.../bin/travsr-dart-index-emitter which does not exist — only
    // ~/.travsr/bin/travsr-dart-index-emitter exists (put there by `travsr lang
    // install dart`). Injecting this env var hits emitter_path() check 1,
    // bypassing the broken relative-path lookup regardless of which travsr-lang-dart
    // variant is on PATH.
    let mut env = Vec::new();
    if let Some(home) = home() {
        let emitter_bin = home
            .join(".travsr")
            .join("bin")
            .join("travsr-dart-index-emitter");
        if emitter_bin.exists() {
            tracing::debug!(
                path = %emitter_bin.display(),
                "dart_access: setting TRAVSR_DART_EMITTER"
            );
            env.push((
                "TRAVSR_DART_EMITTER".to_string(),
                emitter_bin.to_string_lossy().into_owned(),
            ));
        } else {
            tracing::debug!(
                path = %emitter_bin.display(),
                "dart_access: travsr-dart-index-emitter not found, TRAVSR_DART_EMITTER not set"
            );
        }
    }

    if let Some(h) = home() {
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    tracing::debug!(read_paths = ?read_paths, env_keys = ?env.iter().map(|(k,_)| k).collect::<Vec<_>>(), "dart_access: final grants");

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// The user's home directory, on every platform this ships to.
///
/// `HOME` alone is not enough. Windows sets `USERPROFILE` and normally leaves
/// `HOME` unset, so this returned `None` there and every caller below silently
/// dropped its grant: the NuGet package cache, `~/.gradle`, `~/.m2`, the
/// Coursier and Composer caches, `~/.dotnet`. Each toolchain that depends on one
/// was therefore missing a read path it needs on Windows, which reads as "the
/// analyzer produced nothing" rather than as a configuration error.
///
/// This is not a widening of the sandbox policy: every one of those paths is
/// already granted by deliberate design on macOS and Linux, and each is a
/// well-known subdirectory of the invoking user's own home. Honouring the
/// variable Windows actually sets makes the platforms agree rather than giving
/// Windows anything the others do not have. `phase_b_dart` in `travsr-analysis`
/// already reads the pair this way.
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Resolve the JDK installation dir (`JAVA_HOME`): the env var if set, else
/// parsed from `java -XshowSettings:properties`. Shared by the JVM-driven
/// analyzers (scip-java, sbt) so the sandbox can read the JDK and execute
/// `JAVA_HOME/bin/java`.
fn java_home() -> Option<PathBuf> {
    std::env::var("JAVA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            // -XshowSettings prints to stderr for the `-version` combination.
            let output = Command::new("java")
                .args(["-XshowSettings:properties", "-version"])
                .output()
                .ok()?;
            let text = String::from_utf8_lossy(&output.stderr);
            text.lines()
                .find(|l| l.contains("java.home"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| PathBuf::from(v.trim()))
        })
}

/// The directory a PATH-resolvable tool lives in, symlinks resolved so the REAL
/// dir is granted. PATHEXT-aware (finds `sbt.bat` on Windows, not just `sbt`).
/// `None` when the tool is not on PATH.
fn tool_bin_dir(tool: &str) -> Option<PathBuf> {
    let exe = travsr_core::exec::resolve_executable(tool)?;
    std::fs::canonicalize(&exe)
        .ok()
        .and_then(|r| r.parent().map(|p| p.to_path_buf()))
        .or_else(|| exe.parent().map(|p| p.to_path_buf()))
}

/// `scip-java index` drives Gradle (and Maven) to resolve dependencies. Needs:
///   - `JAVA_HOME`       (read) — JDK installation dir
///   - `~/.gradle`       (read+write) — Gradle daemon, caches, wrapper downloads
///   - `~/.m2`           (read) — Maven local repository
///   - `HOME` + `GRADLE_USER_HOME` env vars so Gradle finds its home
///
/// Also the base grant set for `kotlin_access`: kotlin-language-server drives
/// the same Gradle/Maven classpath resolution under the hood, even though it
/// isn't scip-java itself (see `kotlin_access` for what it additionally needs).
fn java_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut write_paths = Vec::new();
    let mut env = Vec::new();

    // JAVA_HOME: env var takes priority; fall back to `java -XshowSettings:properties`.
    let java_home: Option<PathBuf> = java_home();
    if let Some(ref p) = java_home {
        tracing::debug!(path = %p.display(), "java_access: JAVA_HOME grant");
        read_paths.push(p.clone());
        env.push(("JAVA_HOME".to_string(), p.to_string_lossy().into_owned()));
    }

    // Gradle home: env var, then ~/.gradle.
    let gradle_home: Option<PathBuf> = std::env::var("GRADLE_USER_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join(".gradle")));
    if let Some(ref p) = gradle_home {
        tracing::debug!(path = %p.display(), "java_access: GRADLE_USER_HOME grant (read+write)");
        read_paths.push(p.clone());
        write_paths.push(p.clone());
        env.push((
            "GRADLE_USER_HOME".to_string(),
            p.to_string_lossy().into_owned(),
        ));
    }

    // Maven local repository (~/.m2) — read-only.
    if let Some(h) = home() {
        let m2 = h.join(".m2");
        tracing::debug!(path = %m2.display(), exists = m2.exists(), "java_access: ~/.m2 grant (read)");
        read_paths.push(m2);
    }

    if let Some(h) = home() {
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    // G5: scip-java (and kotlin-language-server) drive Gradle/Maven, which invoke
    // `java`. Grant execute on JAVA_HOME/bin so the sandbox can run it; the JDK
    // root is already in read_paths. Gradle/Maven caches stay read-only.
    let mut exec_paths = Vec::new();
    if let Some(p) = &java_home {
        exec_paths.push(p.join("bin"));
    }

    ToolchainAccess {
        read_paths,
        write_paths,
        exec_paths,
        env,
    }
}

/// kotlin-language-server (KLS) needs everything `java_access` grants
/// (JAVA_HOME, `~/.gradle`, `~/.m2`) plus read+execute on its OWN installation
/// dir. Unlike scip-java, whose runnable file sits directly in `~/.travsr/bin`
/// (already execute-granted for every language, see `sandbox/windows.rs`),
/// `travsr lang install kotlin` extracts the server into `~/.travsr/kls/` (see
/// `ZipBinarySpec::extract_dir`) and the `~/.travsr/bin/kotlin-language-server`
/// wrapper launches the real binary/`.bat` INSIDE that dir. Neither blanket
/// grant covers `~/.travsr/kls`, so a sandboxed run used to fail silently at
/// image load: the wrapper spawned (it lives in the granted `~/.travsr/bin`),
/// but the launcher it execs, and the jars under `server/lib` it reads, were
/// both unreachable — 0 nodes, 0 edges, no error surfaced.
fn kotlin_access() -> ToolchainAccess {
    let mut access = java_access();
    if let Some(h) = home() {
        let kls = h.join(".travsr").join("kls");
        tracing::debug!(path = %kls.display(), exists = kls.exists(), "kotlin_access: KLS install dir grant (read+execute)");
        access.read_paths.push(kls.clone());
        access.exec_paths.push(kls);
    }
    access
}

/// Coursier's on-disk artifact cache. `lm-coursier` (sbt's library-management
/// backend since sbt 1.3, still used in 2.x) resolves ordinary project
/// dependencies — including the `semanticdb-scalac` compiler plugin this
/// wrapper's settings file pulls in — through Coursier, separate from
/// `sbt_global_base_dir` below (sbt's own self-bootstrap). Location matches
/// Coursier's own default `cache-dir`: `%LOCALAPPDATA%\Coursier\cache` on
/// Windows, `~/Library/Caches/Coursier` on macOS, `$XDG_CACHE_HOME/coursier`
/// (else `~/.cache/coursier`) on Linux.
fn coursier_cache_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Coursier").join("cache"))
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library").join("Caches").join("Coursier"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| home().map(|h| h.join(".cache")))
            .map(|c| c.join("coursier"))
    }
}

/// sbt 2.x's own "global base" — where the `sbt` launcher self-fetches and
/// caches the actual sbt implementation jar it's about to run
/// ("[launcher] getting org.scala-sbt sbt <ver>"). This is NOT `~/.sbt`: sbt
/// 2.x's launcher script (`sbt.bat`/`sbt`) defaults `-Dsbt.global.base` to
/// `%LOCALAPPDATA%\sbt` on Windows (confirmed by reading the shipped
/// `sbt.bat`: `else if defined LOCALAPPDATA set _SBT_OPTS=-Dsbt.global.base=
/// !LOCALAPPDATA!\sbt`) and to `${XDG_CONFIG_HOME:-$HOME/.config}/sbt` on
/// unix (same default in the `sbt` shell launcher). `boot/`, `cache/`, and the
/// content-addressed `v2/` store all live directly under this dir. Without
/// this grant the self-fetch fails with a raw `java.io.IOException: Access is
/// denied` even though `~/.ivy2`/`~/.sbt`/the Coursier cache above are all
/// granted — sbt never gets far enough to touch any of those.
fn sbt_global_base_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("sbt"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home().map(|h| h.join(".config")))
            .map(|c| c.join("sbt"))
    }
}

/// `scip-scala` drives sbt to resolve dependencies. Needs:
///   - `~/.ivy2` (read)             — Ivy2 artifact cache (sbt's primary resolver)
///   - `~/.sbt`  (read+write)       — sbt home: launchers, plugins, boot dir
///   - Coursier cache (read+write) — dependency resolution (see above)
///   - sbt global base (read+write) — sbt's own self-bootstrap (see above)
///   - `HOME` and `SBT_OPTS` env vars
fn scala_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut write_paths = Vec::new();
    let mut env = Vec::new();

    if let Some(h) = home() {
        let ivy2 = h.join(".ivy2");
        tracing::debug!(path = %ivy2.display(), exists = ivy2.exists(), "scala_access: ~/.ivy2 grant (read)");
        read_paths.push(ivy2);

        let sbt = h.join(".sbt");
        tracing::debug!(path = %sbt.display(), exists = sbt.exists(), "scala_access: ~/.sbt grant (read+write)");
        read_paths.push(sbt.clone());
        write_paths.push(sbt);

        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    if let Some(cc) = coursier_cache_dir() {
        tracing::debug!(path = %cc.display(), exists = cc.exists(), "scala_access: coursier cache grant (read+write)");
        read_paths.push(cc.clone());
        write_paths.push(cc);
    }

    if let Some(gb) = sbt_global_base_dir() {
        tracing::debug!(path = %gb.display(), exists = gb.exists(), "scala_access: sbt global base grant (read+write)");
        read_paths.push(gb.clone());
        write_paths.push(gb);
    }

    if let Ok(sbt_opts) = std::env::var("SBT_OPTS") {
        env.push(("SBT_OPTS".to_string(), sbt_opts));
    }

    // G5: sbt drives the build through the JVM. Grant execute on the sbt
    // launcher's bin dir and on JAVA_HOME/bin (java); their roots are also
    // read-granted so the launcher and JDK can be read. Ivy/sbt caches above stay
    // read-only.
    let mut exec_paths = Vec::new();
    if let Some(jh) = java_home() {
        exec_paths.push(jh.join("bin"));
        read_paths.push(jh.clone());
        env.push(("JAVA_HOME".to_string(), jh.to_string_lossy().into_owned()));
    }
    if let Some(sbt_bin) = tool_bin_dir("sbt") {
        tracing::debug!(path = %sbt_bin.display(), "scala_access: sbt bin dir grant (read+execute)");
        read_paths.push(sbt_bin.clone());
        exec_paths.push(sbt_bin);
    }

    ToolchainAccess {
        read_paths,
        write_paths,
        exec_paths,
        env,
    }
}

/// `scip-php` resolves Composer packages. Needs the global Composer cache:
///   - Composer home dir (read) — `composer config --global home`, then
///     `COMPOSER_HOME` env, then `~/.composer`, then `~/.config/composer`
fn php_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut env = Vec::new();

    let composer_home: Option<PathBuf> =
        run_cmd_stdout("composer", &["config", "--global", "home"])
            .map(PathBuf::from)
            .or_else(|| std::env::var("COMPOSER_HOME").ok().map(PathBuf::from))
            .or_else(|| home().map(|h| h.join(".composer")))
            .or_else(|| home().map(|h| h.join(".config").join("composer")));

    if let Some(ref p) = composer_home {
        tracing::debug!(path = %p.display(), exists = p.exists(), "php_access: COMPOSER_HOME grant (read)");
        read_paths.push(p.clone());
        env.push((
            "COMPOSER_HOME".to_string(),
            p.to_string_lossy().into_owned(),
        ));
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// Resolve the dotnet install root that holds `host/`, `sdk/`, `shared/` — the
/// value `DOTNET_ROOT` must point at, and the dir the sandbox must grant read +
/// execute so scip-dotnet's apphost can load the runtime and shell out to
/// `dotnet`.
///
/// Tries a `dotnet` launcher — `tool_path` first (PATHEXT-aware, so `dotnet.exe`
/// resolves on Windows), then the well-known installs a minimal-PATH daemon
/// omits — and maps the first one carrying a real SDK to its root. **The
/// non-PATH launchers are the load-bearing part**: a daemon launched from a GUI
/// or a login shell without `/opt/homebrew/bin` on PATH left `tool_path` finding
/// nothing, so no `DOTNET_ROOT` was injected and no exec grant issued, and
/// scip-dotnet failed to launch — the C# lane then produced no reference edges.
/// When no launcher is reachable at all, the well-known SDK roots are probed
/// directly (this also covers the per-user `~/.dotnet` when `dotnet` on PATH is a
/// runtime-only host).
fn dotnet_sdk_root() -> Option<PathBuf> {
    if let Some(root) = dotnet_launcher_candidates()
        .into_iter()
        .filter(|p| p.is_file())
        .find_map(|exe| dotnet_sdk_root_from_binary(&exe))
    {
        return Some(root);
    }
    well_known_dotnet_roots()
        .into_iter()
        .find(|d| d.join("sdk").is_dir())
}

/// `dotnet` launcher locations to try, PATH-resolved first, then the well-known
/// installs a sandboxed or GUI-launched daemon's PATH omits: Homebrew's
/// version-independent `opt/` symlink (Apple silicon and Intel), the official
/// installer directory (where `dotnet` sits directly in the SDK root), the
/// Windows machine-wide install, and a user-local `~/.dotnet`.
fn dotnet_launcher_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(p) = travsr_core::exec::tool_path("dotnet") {
        out.push(p);
    }
    out.extend(
        [
            "/opt/homebrew/opt/dotnet/bin/dotnet",
            "/usr/local/opt/dotnet/bin/dotnet",
            "/usr/local/share/dotnet/dotnet",
            "/usr/share/dotnet/dotnet",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    out.extend(
        windows_dotnet_roots()
            .into_iter()
            .map(|r| r.join("dotnet.exe")),
    );
    if let Some(h) = home() {
        out.push(h.join(".dotnet").join("dotnet"));
    }
    out
}

/// The machine-wide dotnet install directories on Windows.
///
/// Without these the minimal-PATH case this whole fallback targets was covered
/// on Windows only when the user happened to have a per-user `~/.dotnet`, even
/// though [`dotnet_sdk_root_from_binary`] already reasons about
/// `C:\Program Files\dotnet` carrying an SDK-less `host/`. `ProgramFiles` /
/// `ProgramFiles(x86)` are honoured so a non-default system drive or a 32-bit
/// install still resolves; the literal paths are the fallback for when the
/// variables are unset. The `sdk/` requirement at both call sites is what keeps
/// a runtime-only root from being chosen.
///
/// Empty on non-Windows, where these paths do not exist and probing them would
/// only cost a stat.
fn windows_dotnet_roots() -> Vec<PathBuf> {
    if !cfg!(windows) {
        return Vec::new();
    }
    let mut out: Vec<PathBuf> = Vec::new();
    for var in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(base) = std::env::var(var) {
            out.push(PathBuf::from(base).join("dotnet"));
        }
    }
    for literal in [r"C:\Program Files\dotnet", r"C:\Program Files (x86)\dotnet"] {
        let p = PathBuf::from(literal);
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// The SDK-bearing root a `dotnet` launcher belongs to, or `None` if it carries
/// no `sdk/`.
///
/// Requires an actual `sdk/` (not just `host/`): scip-dotnet runs `dotnet
/// restore`/build, so a runtime-only host root is useless (`C:\Program
/// Files\dotnet` carries `host/` even when SDK-less, so a `host/` check would
/// wrongly pick it over a real SDK elsewhere). Two layouts:
///   - **Standard** (Windows / dotnet-install / Linux tarball / Program Files):
///     `dotnet(.exe)` sits directly in the root holding `host/ sdk/ shared/`.
///   - **Homebrew macOS:** `…/Cellar/dotnet/<ver>/bin/dotnet` → `…/libexec`.
fn dotnet_sdk_root_from_binary(exe: &std::path::Path) -> Option<PathBuf> {
    // canonicalize resolves symlinks (Homebrew's `bin/dotnet` shim) but adds the
    // `\\?\` verbatim prefix on Windows — strip it, or dotnet chokes on a
    // `\\?\`-prefixed DOTNET_ROOT / exec-grant path.
    let real =
        strip_windows_verbatim(std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf()));
    let dir = real.parent()?;
    if dir.join("sdk").is_dir() {
        return Some(dir.to_path_buf());
    }
    if let Some(libexec) = dir.parent().map(|p| p.join("libexec")) {
        if libexec.join("sdk").is_dir() {
            return Some(libexec);
        }
    }
    None
}

/// SDK roots to probe directly when no `dotnet` launcher is reachable, plus the
/// per-user dotnet-install default. Filtered by an actual `sdk/` at the call site.
fn well_known_dotnet_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = [
        "/opt/homebrew/opt/dotnet/libexec",
        "/usr/local/opt/dotnet/libexec",
        "/usr/local/share/dotnet",
        "/usr/share/dotnet",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();
    out.extend(windows_dotnet_roots());
    if let Some(h) = home() {
        out.push(h.join(".dotnet"));
    }
    out
}

/// `scip-dotnet` resolves NuGet packages. Needs:
///   - NuGet global-packages dir (read+write) — `dotnet restore` downloads new packages here
///   - `~/.dotnet` (read) — dotnet global tools dir; scip-dotnet binary lives here
///   - dotnet runtime root (read) — non-standard for Homebrew installs on macOS
///   - `DOTNET_ROOT` env — scip-dotnet needs it when dotnet is not in /usr/local
///   - `HOME` env var — dotnet uses it for config resolution
fn csharp_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut write_paths = Vec::new();
    let mut env = Vec::new();

    // NuGet global-packages dir — restore reads cached packages AND writes new ones.
    // `dotnet nuget locals global-packages --list` prints:
    // "global-packages: /home/user/.nuget/packages"
    let nuget_packages: Option<PathBuf> =
        run_cmd_stdout("dotnet", &["nuget", "locals", "global-packages", "--list"])
            .and_then(|s| s.split_once(':').map(|(_, p)| PathBuf::from(p.trim())))
            .or_else(|| std::env::var("NUGET_PACKAGES").ok().map(PathBuf::from))
            .or_else(|| home().map(|h| h.join(".nuget").join("packages")));

    if let Some(ref p) = nuget_packages {
        tracing::debug!(path = %p.display(), exists = p.exists(), "csharp_access: NuGet packages grant (read+write)");
        read_paths.push(p.clone());
        write_paths.push(p.clone());
        env.push((
            "NUGET_PACKAGES".to_string(),
            p.to_string_lossy().into_owned(),
        ));
    }

    // ~/.dotnet — dotnet global tools live here (scip-dotnet binary + runtime shim).
    if let Some(h) = home() {
        let dotnet_dir = h.join(".dotnet");
        if dotnet_dir.exists() {
            tracing::debug!(path = %dotnet_dir.display(), "csharp_access: ~/.dotnet grant (read)");
            read_paths.push(dotnet_dir);
        }
    }

    // dotnet runtime root (holds host/, sdk/, shared/): honour DOTNET_ROOT if set,
    // else resolve it via `dotnet_sdk_root` (PATHEXT-aware, so it works on Windows
    // where scip-dotnet's apphost needs both the exec grant below and the injected
    // DOTNET_ROOT to locate the runtime; also covers Homebrew's libexec layout).
    let dotnet_root: Option<PathBuf> = std::env::var("DOTNET_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(dotnet_sdk_root);
    if let Some(ref p) = dotnet_root {
        tracing::debug!(path = %p.display(), "csharp_access: DOTNET_ROOT grant (read)");
        read_paths.push(p.clone());
        env.push(("DOTNET_ROOT".to_string(), p.to_string_lossy().into_owned()));
    }

    if let Some(h) = home() {
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    // G5: scip-dotnet shells out to `dotnet` (in the dotnet root) and is itself a
    // global tool under ~/.dotnet/tools. Grant execute on both; their parents are
    // already read-granted. The NuGet package cache stays read/write only — never
    // executable, since it is filled over the network during restore.
    let mut exec_paths = Vec::new();
    if let Some(p) = &dotnet_root {
        exec_paths.push(p.clone());
    }
    if let Some(h) = home() {
        exec_paths.push(h.join(".dotnet").join("tools"));
    }

    ToolchainAccess {
        read_paths,
        write_paths,
        exec_paths,
        env,
    }
}

/// `scip-ruby` resolves gem dependencies. Needs the RubyGems install dirs:
///   - `GEM_HOME` (read) — primary gem dir from `gem environment gemdir`
///   - all `GEM_PATH` entries (read) — from `gem environment gempath`
///   - `HOME` env var
fn ruby_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut env = Vec::new();

    let gem_home: Option<PathBuf> = run_cmd_stdout("gem", &["environment", "gemdir"])
        .map(PathBuf::from)
        .or_else(|| std::env::var("GEM_HOME").ok().map(PathBuf::from));
    if let Some(ref p) = gem_home {
        tracing::debug!(path = %p.display(), exists = p.exists(), "ruby_access: GEM_HOME grant (read)");
        read_paths.push(p.clone());
        env.push(("GEM_HOME".to_string(), p.to_string_lossy().into_owned()));
    }

    let gem_path_str: Option<String> = run_cmd_stdout("gem", &["environment", "gempath"])
        .or_else(|| std::env::var("GEM_PATH").ok());
    if let Some(ref gp) = gem_path_str {
        for dir in gp.split(':') {
            let p = PathBuf::from(dir.trim());
            if !p.as_os_str().is_empty() {
                tracing::debug!(path = %p.display(), "ruby_access: GEM_PATH entry grant (read)");
                read_paths.push(p);
            }
        }
        env.push(("GEM_PATH".to_string(), gp.clone()));
    }

    if let Some(h) = home() {
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// `travsr-lang-objectivec` parses `.m`/`.mm` in-process via libclang. Beyond the
/// toolchain under `/Library/Developer` (already granted by the base macOS
/// profile), libclang reads additional system support files under `/Library`
/// *outside* `/Library/Developer` while resolving Objective-C system frameworks.
/// Denying those makes every translation unit fail to parse, so the emitter
/// returns an empty index and the whole language silently yields zero symbols
/// (observed: 0 nodes sandboxed vs 13 unsandboxed on the same corpus; bisecting
/// the profile showed a `/Library` read grant is the difference, and
/// `/Library/Developer` alone is insufficient).
///
/// Grant read on `/Library` and the active SDK (via `xcrun --show-sdk-path`,
/// which points into `/Applications/Xcode.app` when a full Xcode is selected).
/// Read-only, in the same class as the base profile's existing `/System` and
/// `/usr` read grants — ADR-017's threat model targets writes, network, and
/// exfiltration, not reads of shared system directories.
fn objc_access() -> ToolchainAccess {
    let mut read_paths = vec![PathBuf::from("/Library")];
    if let Some(sdk) = run_cmd_stdout("xcrun", &["--show-sdk-path"]).map(PathBuf::from) {
        tracing::debug!(path = %sdk.display(), exists = sdk.exists(), "objc_access: Xcode SDK grant (read)");
        read_paths.push(sdk);
    }
    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env: vec![],
    }
}

/// `travsr-swift-index-emitter` uses SwiftSyntax for pure parse-based indexing
/// (no compilation). Defensively grant the Xcode SDK path (via `xcrun
/// --show-sdk-path`) and the Swift toolchain bin dir in case the emitter is
/// dynamically linked against Swift stdlib dylibs. If `xcrun` is absent
/// (Linux, non-Xcode environments) we return empty grants — the emitter is
/// expected to be self-contained there.
fn swift_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();

    if let Some(sdk) = run_cmd_stdout("xcrun", &["--show-sdk-path"]).map(PathBuf::from) {
        tracing::debug!(path = %sdk.display(), exists = sdk.exists(), "swift_access: Xcode SDK grant (read, defensive)");
        read_paths.push(sdk);
    } else {
        tracing::debug!("swift_access: xcrun not found, no SDK path granted");
    }

    // Swift toolchain bin dir — for dynamically loaded stdlib components.
    let swift_bin_dir = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|dir| dir.join("swift"))
        .find(|p| p.is_file())
        .and_then(|swift_bin| {
            std::fs::canonicalize(&swift_bin)
                .ok()
                .and_then(|r| r.parent().map(|p| p.to_path_buf()))
                .or_else(|| swift_bin.parent().map(|p| p.to_path_buf()))
        });
    if let Some(ref p) = swift_bin_dir {
        tracing::debug!(path = %p.display(), "swift_access: Swift toolchain bin dir grant (read)");
        read_paths.push(p.clone());
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env: vec![],
    }
}

/// `rust-analyzer lsif` needs read access to:
///   - `CARGO_HOME` (~/.cargo) — rust-analyzer binary + crates registry
///   - `RUSTUP_HOME` (~/.rustup) — active toolchain (used by rust-analyzer for stdlib analysis)
///   - `HOME` env so rustup/cargo can locate their homes at runtime
fn rust_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut env = Vec::new();

    let cargo_home = std::env::var("CARGO_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join(".cargo")));
    if let Some(ref p) = cargo_home {
        tracing::debug!(path = %p.display(), exists = p.exists(), "rust_access: CARGO_HOME grant (read)");
        read_paths.push(p.clone());
        env.push(("CARGO_HOME".to_string(), p.to_string_lossy().into_owned()));
    }

    let rustup_home = std::env::var("RUSTUP_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join(".rustup")));
    if let Some(ref p) = rustup_home {
        tracing::debug!(path = %p.display(), exists = p.exists(), "rust_access: RUSTUP_HOME grant (read)");
        read_paths.push(p.clone());
        env.push(("RUSTUP_HOME".to_string(), p.to_string_lossy().into_owned()));
    }

    if let Some(h) = home() {
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// `travsr-lsif-ts` is a Node.js binary. Node may be managed by nvm (~/.nvm) or
/// installed to a custom npm prefix (~/.npm-global). Grant both when present so
/// the sidecar can resolve `node` and its module tree inside the sandbox.
/// Homebrew node (/opt/homebrew) and system node (/usr/local) are already covered
/// by the macOS sandbox profile's base rules.
fn typescript_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut env = Vec::new();

    if let Some(home) = home() {
        let nvm_dir = std::env::var("NVM_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".nvm"));
        if nvm_dir.exists() {
            tracing::debug!(path = %nvm_dir.display(), "typescript_access: NVM_DIR grant (read)");
            read_paths.push(nvm_dir.clone());
            env.push((
                "NVM_DIR".to_string(),
                nvm_dir.to_string_lossy().into_owned(),
            ));
        }

        // Global npm prefix, e.g. ~/.npm-global — covers globally installed travsr-lsif-ts.
        let npm_global = std::env::var("NPM_CONFIG_PREFIX")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".npm-global"));
        if npm_global.exists() {
            tracing::debug!(path = %npm_global.display(), "typescript_access: npm global prefix grant (read)");
            read_paths.push(npm_global);
        }

        env.push(("HOME".to_string(), home.to_string_lossy().into_owned()));
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// `scip-python` uses the Python runtime to resolve imports. Grant:
///   - `PYENV_ROOT` (~/.pyenv) if pyenv is in use
///   - `~/.local` — pip user-scheme install prefix (pip install --user)
fn python_access() -> ToolchainAccess {
    let mut read_paths = Vec::new();
    let mut env = Vec::new();

    if let Some(home) = home() {
        let pyenv_root = std::env::var("PYENV_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".pyenv"));
        if pyenv_root.exists() {
            tracing::debug!(path = %pyenv_root.display(), "python_access: PYENV_ROOT grant (read)");
            read_paths.push(pyenv_root.clone());
            env.push((
                "PYENV_ROOT".to_string(),
                pyenv_root.to_string_lossy().into_owned(),
            ));
        }

        // pip --user installs land in ~/.local/lib/pythonX.Y/site-packages and
        // ~/.local/bin — grant the whole prefix so the analyzer can import them.
        let local = home.join(".local");
        if local.exists() {
            tracing::debug!(path = %local.display(), "python_access: ~/.local grant (read)");
            read_paths.push(local);
        }

        env.push(("HOME".to_string(), home.to_string_lossy().into_owned()));
    }

    ToolchainAccess {
        read_paths,
        write_paths: vec![],
        exec_paths: vec![],
        env,
    }
}

/// Run `go env <keys>` and return key→value. Empty if `go` is not on PATH or fails.
/// `go env KEY1 KEY2` prints one value per line in argument order.
fn go_env(keys: &[&str]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(output) = Command::new("go").arg("env").args(keys).output() else {
        return out;
    };
    if !output.status.success() {
        return out;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for (k, v) in keys.iter().zip(text.lines()) {
        let v = v.trim();
        if !v.is_empty() {
            out.insert((*k).to_string(), v.to_string());
        }
    }
    out
}

/// Run an arbitrary command and return its trimmed stdout, or `None` if the
/// command is absent, fails, or produces empty output.
fn run_cmd_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// scip-go uses `go/packages` (i.e. the real `go` toolchain) to load packages, so
/// it needs: the module cache (`GOMODCACHE`, read), the build cache (`GOCACHE`,
/// read+write), `GOPATH` (covers `~/go/bin/scip-go` itself + `pkg/mod`), and the
/// `GO*`/`HOME` env so the toolchain finds them. Paths resolved via `go env`, with
/// `$HOME`-based fallbacks if `go` is not reachable at sandbox-build time.
fn go_access() -> ToolchainAccess {
    let env_map = go_env(&["GOPATH", "GOCACHE", "GOMODCACHE"]);

    let gopath = env_map
        .get("GOPATH")
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join("go")));
    let gomodcache = env_map
        .get("GOMODCACHE")
        .map(PathBuf::from)
        .or_else(|| gopath.as_ref().map(|p| p.join("pkg/mod")));
    let gocache = env_map.get("GOCACHE").map(PathBuf::from).or_else(|| {
        // Default GOCACHE: macOS → $HOME/Library/Caches/go-build, Linux → $HOME/.cache/go-build.
        home().map(|h| {
            if cfg!(target_os = "macos") {
                h.join("Library/Caches/go-build")
            } else {
                h.join(".cache/go-build")
            }
        })
    });

    let mut read_paths = Vec::new();
    let mut write_paths = Vec::new();
    let mut env = Vec::new();

    if let Some(p) = &gopath {
        read_paths.push(p.clone()); // covers ~/go/bin/scip-go + pkg/mod
        env.push(("GOPATH".to_string(), p.to_string_lossy().into_owned()));
    }
    if let Some(p) = &gomodcache {
        read_paths.push(p.clone()); // modules are read-only
        env.push(("GOMODCACHE".to_string(), p.to_string_lossy().into_owned()));
    }
    if let Some(p) = &gocache {
        read_paths.push(p.clone());
        write_paths.push(p.clone()); // `go build` writes compiled artifacts here
        env.push(("GOCACHE".to_string(), p.to_string_lossy().into_owned()));
    }
    // GOROOT: the Go standard library. scip-go's type checker reads stdlib
    // source for cross-package type resolution. Missing on large repos (e.g.
    // Kubernetes 2255 packages) causes the sandbox to block stdlib reads and
    // scip-go to exit silently with 0 edges despite a successful handshake.
    let goroot = std::env::var("GOROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| run_cmd_stdout("go", &["env", "GOROOT"]).map(|s| PathBuf::from(s.trim())));
    if let Some(ref p) = goroot {
        tracing::debug!(path = %p.display(), "go_access: GOROOT grant (stdlib for type checker)");
        read_paths.push(p.clone());
        env.push(("GOROOT".to_string(), p.to_string_lossy().into_owned()));
    }

    if let Some(h) = home() {
        // Go tooling consults $HOME for defaults; pass it through.
        env.push(("HOME".to_string(), h.to_string_lossy().into_owned()));
    }

    // G5: the sandbox must EXECUTE the toolchain, not just read it. GOPATH/bin
    // holds scip-go itself; GOROOT/bin holds go, which scip-go shells out to for
    // cross-package type resolution. Both are already in read_paths (via their
    // roots) — listing the bin dirs here adds the execute right the Windows
    // AppContainer withholds by default. Module/build caches stay read-only.
    let mut exec_paths = Vec::new();
    if let Some(p) = &gopath {
        exec_paths.push(p.join("bin"));
    }
    if let Some(p) = &goroot {
        exec_paths.push(p.join("bin"));
    }

    ToolchainAccess {
        read_paths,
        write_paths,
        exec_paths,
        env,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dotnet_launcher_candidates, dotnet_sdk_root_from_binary, strip_windows_verbatim,
        well_known_dotnet_roots,
    };
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "travsr-toolchain-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// The repo-write grant guard: every platform that materialises a grant
    /// calls this before creating the path as the user, so a link anywhere on
    /// it must be refused, not just at the leaf.
    #[test]
    fn grant_path_symlink_guard_rejects_a_link_at_any_component() {
        use super::grant_path_has_symlink;
        let root = scratch("grantlink");

        // A path that does not exist yet is fine: it gets created as a real
        // file or directory immediately after, then re-stat'd.
        assert!(!grant_path_has_symlink(&root, "index.scip"));

        // A real file and a real nested directory are both fine.
        std::fs::write(root.join("real.scip"), b"").expect("write file");
        std::fs::create_dir_all(root.join("jvm/target")).expect("create dirs");
        assert!(!grant_path_has_symlink(&root, "real.scip"));
        assert!(!grant_path_has_symlink(&root, "jvm/target"));

        #[cfg(unix)]
        {
            let outside = root.join("outside.txt");
            std::fs::write(&outside, b"secret").expect("write outside");

            // Leaf is a link.
            std::os::unix::fs::symlink(&outside, root.join("linked.scip")).expect("symlink");
            assert!(grant_path_has_symlink(&root, "linked.scip"));

            // An intermediate component is a link: the leaf below it is a real
            // directory, so a leaf-only check would pass this.
            let elsewhere = root.join("elsewhere");
            std::fs::create_dir_all(elsewhere.join("target")).expect("create elsewhere");
            std::os::unix::fs::symlink(&elsewhere, root.join("js")).expect("symlink dir");
            assert!(grant_path_has_symlink(&root, "js/target"));
        }
    }

    /// Homebrew layout: `<root>/bin/dotnet` with the SDK under `<root>/libexec`.
    /// This is the install a minimal-PATH daemon could not resolve before the fix.
    #[test]
    fn dotnet_root_resolves_homebrew_layout() {
        let root = scratch("dn-brew");
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::create_dir_all(root.join("libexec/sdk")).unwrap();
        std::fs::write(root.join("bin/dotnet"), b"#!/bin/sh\n").unwrap();

        let got = dotnet_sdk_root_from_binary(&root.join("bin/dotnet")).unwrap();
        // `dotnet_sdk_root_from_binary` strips the Windows `\\?\` verbatim prefix
        // `canonicalize` adds; strip `want` the same way (a no-op off Windows).
        assert_eq!(
            got,
            strip_windows_verbatim(std::fs::canonicalize(root.join("libexec")).unwrap())
        );
    }

    /// Standard layout: `dotnet` sits directly in the SDK root beside `sdk/`.
    #[test]
    fn dotnet_root_resolves_standard_layout() {
        let root = scratch("dn-std");
        std::fs::create_dir_all(root.join("sdk")).unwrap();
        std::fs::write(root.join("dotnet"), b"#!/bin/sh\n").unwrap();

        let got = dotnet_sdk_root_from_binary(&root.join("dotnet")).unwrap();
        // Strip the `\\?\` prefix from `want` to match the function (see above).
        assert_eq!(
            got,
            strip_windows_verbatim(std::fs::canonicalize(&root).unwrap())
        );
    }

    /// A runtime-only host (no `sdk/`) is rejected: scip-dotnet needs the SDK to
    /// restore/build, so a runtime root is worse than falling through.
    #[test]
    fn dotnet_root_rejects_runtime_only_host() {
        let root = scratch("dn-runtime");
        std::fs::create_dir_all(root.join("host")).unwrap();
        std::fs::write(root.join("dotnet"), b"#!/bin/sh\n").unwrap();
        assert!(dotnet_sdk_root_from_binary(&root.join("dotnet")).is_none());
    }

    /// The Homebrew fallbacks a sandboxed / GUI-launched PATH omits are in the
    /// search, both as launcher candidates and as direct-root candidates.
    #[test]
    fn dotnet_candidates_cover_the_sandbox_path_gap() {
        assert!(dotnet_launcher_candidates()
            .iter()
            .any(|p| p.ends_with("opt/homebrew/opt/dotnet/bin/dotnet")));
        assert!(well_known_dotnet_roots()
            .iter()
            .any(|p| p.ends_with("opt/homebrew/opt/dotnet/libexec")));
    }

    #[test]
    fn strips_drive_verbatim_prefix() {
        assert_eq!(
            strip_windows_verbatim(PathBuf::from(r"\\?\C:\Users\home\.dotnet")),
            PathBuf::from(r"C:\Users\home\.dotnet")
        );
    }

    #[test]
    fn strips_unc_verbatim_prefix_back_to_unc() {
        assert_eq!(
            strip_windows_verbatim(PathBuf::from(r"\\?\UNC\server\share\repo")),
            PathBuf::from(r"\\server\share\repo")
        );
    }

    #[test]
    fn no_op_on_clean_paths() {
        assert_eq!(
            strip_windows_verbatim(PathBuf::from(r"C:\repo")),
            PathBuf::from(r"C:\repo")
        );
        assert_eq!(
            strip_windows_verbatim(PathBuf::from("/home/user/repo")),
            PathBuf::from("/home/user/repo")
        );
    }
}
