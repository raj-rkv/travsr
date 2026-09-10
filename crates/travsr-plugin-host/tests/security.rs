//! ADR-017 security tests — merge-gating for P5-S1 onward.
//! These tests verify the sandbox policy enforcements defined in ADR-017.
//! Tests that require bwrap (Linux) are skipped on other platforms.

use travsr_plugin_host::phase_b::catalog::{lookup, SandboxRequirement};
use travsr_plugin_host::sandbox::policy::SandboxPolicy;
#[cfg(target_os = "linux")]
use travsr_plugin_host::sandbox::policy::SandboxUnavailable;

// 1. Egress allowed — Standard sandbox permits network (build tools need it)
//
// ADR-017 intentionally allows network for Standard/NativeIpc policies so that
// build tools (go mod, npm, pip, …) can fetch dependencies. FS confinement via
// bwrap still applies; only network namespace isolation is skipped.
#[test]
#[cfg(target_os = "linux")]
fn sandbox_standard_allows_network() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    // tempdir() reads TMPDIR; serialise against the env-mutating provider tests.
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let scratch = tempfile::tempdir().expect("scratch");

    let cmd = build_sandboxed_command(
        "sh",
        &[
            "-c",
            "curl -s --max-time 2 http://1.1.1.1/ > /dev/null 2>&1; echo $?",
        ],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "",
    );

    match cmd {
        Err(SandboxUnavailable(ref msg)) => {
            if std::env::var("CI").is_ok() {
                panic!("sandbox unavailable in CI: {msg}");
            }
            eprintln!("SKIP: {msg}");
        }
        Ok(spawner) => {
            let output = spawner.output().expect("spawn");
            let exit_code = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // The shell command is `sh -c "curl ...; echo $?"` — sh always exits 0
            // when bwrap runs successfully, so output.status.success() is always true
            // for a live sandbox. We inspect exit_code directly:
            //   "0"   → curl ran and network succeeded (expected: network is allowed)
            //   "127" → curl not found inside bwrap (acceptable: no curl in namespace)
            //   "126" → curl not executable (acceptable)
            //   other → curl found but network blocked → test FAILS
            let bwrap_failed = !output.status.success();
            let curl_succeeded = exit_code == "0";
            let curl_absent = exit_code == "127" || exit_code == "126";
            assert!(
                bwrap_failed || curl_absent || curl_succeeded,
                "Standard sandbox blocked network but ADR-017 Amendment A1 intentionally \
                 allows it for build tools; exit_code={exit_code}"
            );
        }
    }
}

// 2. Fail-closed — no sandbox → Err(SandboxUnavailable), never an unsandboxed command
#[test]
fn sandbox_unavailable_returns_err_not_fallback_command() {
    // tempdir() reads TMPDIR; serialise against the env-mutating provider tests.
    let _env = env_guard();
    // On any platform, if sandbox is unavailable we get Err, not a fallback Command.
    // We verify this by calling the wrong-platform builder and checking the type.
    //
    // The sandbox builders return Result<Command, SandboxUnavailable>.
    // SandboxUnavailable must propagate upward — callers must not run the subprocess.
    // This test verifies the type contract is correct (no silent fallback).

    #[cfg(target_os = "linux")]
    {
        use travsr_plugin_host::sandbox::macos::build_sandboxed_command;
        let repo = tempfile::tempdir().expect("tempdir");
        let scratch = tempfile::tempdir().expect("scratch");
        let result = build_sandboxed_command(
            "sh",
            &["-c", "true"],
            repo.path(),
            scratch.path(),
            &SandboxPolicy::Standard,
        );
        // On Linux, the macOS builder always returns Err
        assert!(
            result.is_err(),
            "macOS sandbox builder should return Err on Linux"
        );
    }

    #[cfg(target_os = "macos")]
    {
        use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
        let repo = tempfile::tempdir().expect("tempdir");
        let scratch = tempfile::tempdir().expect("scratch");
        let result = build_sandboxed_command(
            "sh",
            &["-c", "true"],
            repo.path(),
            scratch.path(),
            &SandboxPolicy::Standard,
            "",
        );
        assert!(
            result.is_err(),
            "Linux sandbox builder should return Err on macOS"
        );
    }

    // On Windows or other platforms, both builders return Err — validate both.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        use travsr_plugin_host::sandbox::linux::build_sandboxed_command as linux_build;
        use travsr_plugin_host::sandbox::macos::build_sandboxed_command as macos_build;
        let repo = tempfile::tempdir().expect("tempdir");
        let scratch = tempfile::tempdir().expect("scratch");
        assert!(linux_build(
            "sh",
            &[],
            repo.path(),
            scratch.path(),
            &SandboxPolicy::Standard,
            "",
        )
        .is_err());
        assert!(macos_build(
            "sh",
            &[],
            repo.path(),
            scratch.path(),
            &SandboxPolicy::Standard
        )
        .is_err());
    }
}

// 3. Trust gate — ADR-017 Rule 3, driven through the REAL spawn path (#414).
//
// These tests call `invoke_phase_b_all` (the production Phase B entry point)
// against a lang.toml that registers an external language whose provider
// binary is present on PATH, so the ONLY thing separating "spawned" from
// "skipped" is the per-corpus trust grant. The earlier versions of these
// tests only unit-tested `TrustConfig::is_trusted`, which production never
// called — they passed while the gate was dead code.

/// Serialises every test that reads or writes process-global env.
///
/// The provider tests `set_var`/`remove_var` on PATH/TRAVSR_BIN_DIR/
/// TRAVSR_LANG_TOML; in a multithreaded process that is a data race against any
/// concurrent `getenv` — including the `env::temp_dir()` lookup inside every
/// `tempfile::tempdir()` call the sandbox tests make. `setenv` reallocates the
/// C `environ` array, so a concurrent read can see a corrupted PATH and the
/// analyzer resolution silently misses the on-PATH stub (`skipped_no_analyzer`).
/// Env is per-process, so holding this lock across every env-touching test in
/// this binary removes the race; the pure struct-validation tests take no env
/// and need no guard. (Other test binaries are separate processes, unaffected.)
static TRUST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TRUST_ENV_LOCK`], recovering from a poisoned guard left by an
/// unrelated test panic (the data being guarded is process env, not test state).
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    TRUST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// How the fake `travsr-lang-go` provider is laid out on the temp PATH dir.
enum ProviderLayout {
    /// A directly spawnable stub: `.exe` on Windows (any bytes — a spawn
    /// failure still lands in `crashed`, proving the spawn path ran), an
    /// executable shell script elsewhere.
    NativeStub,
    /// Windows npm layout: only a `.cmd` shim, no PE anywhere (#573).
    #[cfg(windows)]
    CmdShimOnly,
    /// Windows npm layout: a `.cmd` shim whose packaged native binary sits at
    /// the conventional `node_modules/<pkg>/bin/<name>.exe` (#573).
    #[cfg(windows)]
    CmdShimWithPackagedExe,
}

/// Write a directly spawnable executable stub named `name` into `bin_dir`:
/// an `MZ`-prefixed `.exe` on Windows (a spawn failure still lands in `crashed`,
/// proving the spawn path ran), an executable `#!/bin/sh` script elsewhere.
fn write_exec_stub(bin_dir: &std::path::Path, name: &str) {
    use std::io::Write as _;
    #[cfg(windows)]
    let path = bin_dir.join(format!("{name}.exe"));
    #[cfg(not(windows))]
    let path = bin_dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create exec stub");
    #[cfg(windows)]
    f.write_all(b"MZ not a real PE").expect("write");
    #[cfg(not(windows))]
    f.write_all(b"#!/bin/sh\nexit 0\n").expect("write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("chmod exec stub");
    }
}

/// Stub go's UNDERLYING analyzer (`scip-go`, go's `catalog.command`) onto the
/// test PATH. Since #743 the resolver's G2 gate skips a language whose wrapper
/// is present but whose analyzer tool is absent (`tool_available` false), so
/// without this the fake `travsr-lang-go` wrapper resolves but go is dropped as
/// `missing_tool` before it ever reaches the trust gate or spawn stage these
/// tests exercise. Only layouts that are meant to reach spawn need it.
fn write_fake_analyzer_tool(bin_dir: &std::path::Path) {
    write_exec_stub(bin_dir, "scip-go");
}

/// Materialise `layout` inside `bin_dir`.
fn write_fake_provider(bin_dir: &std::path::Path, layout: &ProviderLayout) {
    match layout {
        ProviderLayout::NativeStub => {
            write_exec_stub(bin_dir, "travsr-lang-go");
            write_fake_analyzer_tool(bin_dir);
        }
        #[cfg(windows)]
        ProviderLayout::CmdShimOnly => {
            std::fs::write(
                bin_dir.join("travsr-lang-go.cmd"),
                "@ECHO off\r\nnode \"%~dp0\\cli.js\" %*\r\n",
            )
            .expect("write shim");
        }
        #[cfg(windows)]
        ProviderLayout::CmdShimWithPackagedExe => {
            std::fs::write(
                bin_dir.join("travsr-lang-go.cmd"),
                "@ECHO off\r\nrem opaque shim body\r\n",
            )
            .expect("write shim");
            let exe = bin_dir
                .join("node_modules")
                .join("@travsr-plugin")
                .join("go")
                .join("bin")
                .join("travsr-lang-go.exe");
            std::fs::create_dir_all(exe.parent().unwrap()).expect("mk packaged bin dir");
            std::fs::write(&exe, b"MZ not a real PE").expect("write packaged exe");
            write_fake_analyzer_tool(bin_dir);
        }
    }
}

/// Drive `invoke_phase_b_all` for a registered external language ("go", fake
/// `travsr-lang-go` on PATH per `layout`) under the given `trusted_corpora`
/// list, and return the outcome. Restores env before returning.
fn run_phase_b_with_provider(
    corpus: &str,
    trusted_corpora: &[&str],
    layout: ProviderLayout,
) -> travsr_plugin_host::PhaseBOutcome {
    let _guard = env_guard();

    // lang.toml: "go" registered; trust list as given.
    let cfg_dir = tempfile::tempdir().expect("cfg tempdir");
    let lang_toml = cfg_dir.path().join("lang.toml");
    let trusted = trusted_corpora
        .iter()
        .map(|c| format!("{c:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    std::fs::write(
        &lang_toml,
        format!("registered = [\"go\"]\ntrusted_corpora = [{trusted}]\n"),
    )
    .expect("write lang.toml");

    // Fake provider binary so the CatalogResolver surfaces "go" — without one
    // the language never reaches the spawn loop and the gates would be
    // untestable. The stub is never a working plugin: if it is ever spawned
    // the handshake fails and the language lands in `crashed`, which is
    // itself proof the spawn path was reached.
    let bin_dir = tempfile::tempdir().expect("bin tempdir");
    write_fake_provider(bin_dir.path(), &layout);

    // `which_binary` checks ~/.travsr/bin before PATH — on the real machine
    // running this test suite that dir can (and, on a dev box that has ever run
    // `travsr lang install go`, does) already contain a real travsr-lang-go, which
    // would win over this test's fake PATH-injected provider and silently test
    // against the wrong binary. Point it at an empty dir so only `layout` is
    // ever visible, matching a machine with nothing installed globally.
    let empty_travsr_bin = tempfile::tempdir().expect("empty travsr bin tempdir");

    let old_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths: Vec<std::path::PathBuf> = vec![bin_dir.path().to_path_buf()];
    paths.extend(std::env::split_paths(&old_path));
    let new_path = std::env::join_paths(paths).expect("join PATH");
    let old_lang_toml = std::env::var_os("TRAVSR_LANG_TOML");
    let old_travsr_bin_dir = std::env::var_os("TRAVSR_BIN_DIR");
    std::env::set_var("PATH", &new_path);
    std::env::set_var("TRAVSR_LANG_TOML", &lang_toml);
    std::env::set_var("TRAVSR_BIN_DIR", empty_travsr_bin.path());

    let repo = tempfile::tempdir().expect("repo tempdir");
    std::fs::write(repo.path().join("main.go"), "package main\n").expect("write go file");

    let indexer = travsr_plugin_host::PluginIndexer::new(corpus);
    let inputs = travsr_plugin_host::PhaseBInputs {
        repo_root: repo.path(),
        present_languages: ["go".to_string()].into_iter().collect(),
        indexable_paths: &[],
        liveness: None,
    };
    let (.., outcome) = indexer.invoke_phase_b_all(&inputs);

    // Restore process env before any assertion can panic in the caller.
    std::env::set_var("PATH", old_path);
    match old_lang_toml {
        Some(v) => std::env::set_var("TRAVSR_LANG_TOML", v),
        None => std::env::remove_var("TRAVSR_LANG_TOML"),
    }
    match old_travsr_bin_dir {
        Some(v) => std::env::set_var("TRAVSR_BIN_DIR", v),
        None => std::env::remove_var("TRAVSR_BIN_DIR"),
    }

    outcome
}

#[test]
fn trust_gate_blocks_untrusted_corpus() {
    // A trust grant exists — but for a DIFFERENT corpus. The gate must be
    // per-corpus, not "any grant anywhere unlocks Phase B".
    let outcome = run_phase_b_with_provider(
        "github.com/acme/untrusted-repo",
        &["github.com/acme/some-other-repo"],
        ProviderLayout::NativeStub,
    );
    assert_eq!(
        outcome.skipped_untrusted_corpus,
        vec!["go".to_string()],
        "registered external language must be skipped before spawn when the \
         corpus has no trust grant"
    );
    assert!(
        outcome.ran.is_empty(),
        "nothing may run for an untrusted corpus: {:?}",
        outcome.ran
    );
    assert!(
        outcome.crashed.is_empty(),
        "no spawn may be attempted for an untrusted corpus (a crash entry \
         means the sidecar was spawned): {:?}",
        outcome.crashed
    );
}

#[test]
fn trust_gate_allows_explicitly_trusted_corpus() {
    let outcome = run_phase_b_with_provider(
        "github.com/acme/my-repo",
        &["github.com/acme/my-repo"],
        ProviderLayout::NativeStub,
    );
    assert!(
        outcome.skipped_untrusted_corpus.is_empty(),
        "trusted corpus must pass the gate: {:?}",
        outcome.skipped_untrusted_corpus
    );
    // Past the gate, "go" must reach the spawn stage. The stub provider is not
    // a real plugin, so the language lands in `ran`, `crashed`, or
    // `version_mismatch` depending on how far the handshake gets — any of
    // these proves the spawn path executed, which the old assertion-only test
    // never did.
    let reached_spawn = outcome.ran.iter().any(|l| l == "go")
        || outcome.crashed.iter().any(|l| l == "go")
        || outcome.version_mismatch.iter().any(|(l, ..)| l == "go");
    assert!(
        reached_spawn,
        "trusted corpus must reach the spawn stage; outcome: ran={:?} crashed={:?} \
         version_mismatch={:?} skipped_no_analyzer={:?}",
        outcome.ran, outcome.crashed, outcome.version_mismatch, outcome.skipped_no_analyzer
    );
}

// 3b. npm .cmd shims — the AppContainer spawn runs PE images only (#573).

/// A provider installed only as an npm `.cmd` shim must be skipped at
/// resolution (with the actionable hint logged by the resolver — the
/// classification itself is unit-tested in resolver.rs) and never handed to
/// the sandbox spawn, where it used to surface as a bare CreateProcessW crash
/// recorded against the language.
#[cfg(windows)]
#[test]
fn cmd_shim_only_provider_is_skipped_not_crashed() {
    let outcome = run_phase_b_with_provider(
        "github.com/acme/my-repo",
        &["github.com/acme/my-repo"], // trusted, so the shim is the only gate
        ProviderLayout::CmdShimOnly,
    );
    assert!(
        outcome.crashed.is_empty(),
        "a .cmd-only provider must never reach the spawn (a crash entry means \
         the doomed shim was spawned): {:?}",
        outcome.crashed
    );
    assert!(outcome.ran.is_empty(), "nothing can run: {:?}", outcome.ran);
    assert!(
        outcome.version_mismatch.is_empty(),
        "no handshake can happen: {:?}",
        outcome.version_mismatch
    );
    // The skip must be user-visible, not silent: the resolver returns None for
    // the unresolvable shim, so the language lands in skipped_no_analyzer and
    // `travsr init`/`status` surface the `travsr lang install go` hint (#573's
    // "actionable message, not a bare CreateProcessW error" criterion).
    assert!(
        outcome.skipped_no_analyzer.iter().any(|l| l == "go"),
        "a .cmd-only provider must surface as skipped_no_analyzer so the user \
         gets the install hint: {outcome:?}"
    );
}

/// When npm's packaged native binary ships next to the shim
/// (`node_modules/<pkg>/bin/<name>.exe`, the travsr packaging convention),
/// the resolver must adopt it and the language must reach the spawn stage.
#[cfg(windows)]
#[test]
fn cmd_shim_with_packaged_exe_reaches_spawn() {
    let outcome = run_phase_b_with_provider(
        "github.com/acme/my-repo",
        &["github.com/acme/my-repo"],
        ProviderLayout::CmdShimWithPackagedExe,
    );
    let reached_spawn = outcome.ran.iter().any(|l| l == "go")
        || outcome.crashed.iter().any(|l| l == "go")
        || outcome.version_mismatch.iter().any(|(l, ..)| l == "go");
    assert!(
        reached_spawn,
        "the packaged exe behind the shim must be spawned; outcome: ran={:?} crashed={:?} \
         version_mismatch={:?} skipped_no_analyzer={:?}",
        outcome.ran, outcome.crashed, outcome.version_mismatch, outcome.skipped_no_analyzer
    );
    assert!(
        !outcome.skipped_no_analyzer.iter().any(|l| l == "go"),
        "must not be reported as analyzer-less when the packaged exe exists: {outcome:?}"
    );
}

// 4. FS confinement — repo root is read-only in sandbox
#[test]
#[cfg(target_os = "linux")]
fn sandbox_repo_root_is_read_only() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    // tempdir() reads TMPDIR; serialise against the env-mutating provider tests.
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let scratch = tempfile::tempdir().expect("scratch");
    let breach_path = repo.path().join("sandbox_write_test.txt");

    let cmd = build_sandboxed_command(
        "sh",
        &[
            "-c",
            &format!("echo test > {} 2>&1; echo $?", breach_path.display()),
        ],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "",
    );

    match cmd {
        Err(SandboxUnavailable(ref msg)) => {
            if std::env::var("CI").is_ok() {
                panic!("sandbox unavailable in CI: {msg}");
            }
            eprintln!("SKIP: {msg}");
        }
        Ok(spawner) => {
            let _ = spawner.output();
            assert!(
                !breach_path.exists(),
                "sandbox allowed write to repo root — FS confinement broken"
            );
        }
    }
}

// 4b. FS confinement — scala's narrowed repo-write grant opens only its build
// subpaths (target/), never the whole repo root. A write to a repo-root file
// outside those subpaths must still be denied; a write into target/ must succeed.
#[test]
#[cfg(target_os = "linux")]
fn sandbox_scala_repo_write_is_narrowed_to_build_subpaths() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let scratch = tempfile::tempdir().expect("scratch");
    let breach_path = repo.path().join("build.sbt"); // repo root, not a write subpath
    let allowed_path = repo.path().join("target").join("write_test.txt");

    // Breach: writing a repo-root file outside the narrowed subpaths must fail.
    let breach = build_sandboxed_command(
        "sh",
        &[
            "-c",
            &format!("echo hostile > {} 2>&1; true", breach_path.display()),
        ],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "scala",
    );
    match breach {
        Err(SandboxUnavailable(ref msg)) => {
            if std::env::var("CI").is_ok() {
                panic!("sandbox unavailable in CI: {msg}");
            }
            eprintln!("SKIP: {msg}");
            return;
        }
        Ok(spawner) => {
            let _ = spawner.output();
            assert!(
                !breach_path.exists() || std::fs::read_to_string(&breach_path).unwrap_or_default().trim() != "hostile",
                "scala sandbox allowed a write to a repo-root file outside target/ — narrowing broken"
            );
        }
    }

    // Allowed: writing into target/ (a granted subpath) must succeed.
    let allowed = build_sandboxed_command(
        "sh",
        &["-c", &format!("echo ok > {}", allowed_path.display())],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "scala",
    );
    if let Ok(spawner) = allowed {
        let _ = spawner.output();
        assert!(
            allowed_path.exists(),
            "scala sandbox blocked a write into target/ — the narrowed grant must still allow sbt's build outputs"
        );
    }
}

// 4c. The repo-write grant set is a signed enumeration (ADR-017 Amendment A6),
// not a knob. Pinned exactly, so widening it has to be a deliberate test edit:
// the previous suite asserted only "build.sbt denied, target/ allowed", which
// would have passed unchanged if a language had been granted the whole root.
#[test]
fn repo_write_grants_are_exactly_the_authorised_set() {
    use travsr_plugin_host::sandbox::toolchain::{repo_write_subpaths, RepoWrite};

    assert_eq!(
        repo_write_subpaths("scala").to_vec(),
        vec![
            RepoWrite::Dir("target"),
            RepoWrite::Dir("project/target"),
            RepoWrite::Dir("project/project/target"),
            RepoWrite::Dir("js/target"),
            RepoWrite::Dir("jvm/target"),
            RepoWrite::Dir("native/target"),
            RepoWrite::File(".travsr-semanticdb.sbt"),
        ],
        "scala's repo-write grant must match ADR-017 Amendment A6 exactly"
    );
    assert_eq!(
        repo_write_subpaths("php").to_vec(),
        vec![RepoWrite::File("index.scip")],
        "php's repo-write grant must be the single output file, never a directory"
    );
    // These three drive the project's own build tool, which writes into the
    // project. Verified on Linux for java/maven: `target/` bound over a
    // read-only root lets clean clear the contents and javac run, while a write
    // outside the grant is still denied.
    assert_eq!(
        repo_write_subpaths("java").to_vec(),
        vec![
            RepoWrite::Dir("target"),
            RepoWrite::Dir("build"),
            RepoWrite::Dir(".gradle"),
        ],
        "java's repo-write grant must match ADR-017 Amendment A8 exactly"
    );
    assert_eq!(
        repo_write_subpaths("kotlin").to_vec(),
        vec![
            RepoWrite::Dir("build"),
            RepoWrite::Dir(".gradle"),
            // The Kotlin Gradle Plugin's build session dir. Measured: without
            // it compileKotlin dies on a read-only .kotlin/sessions/ even with
            // every other grant present.
            RepoWrite::Dir(".kotlin"),
        ],
        "kotlin's repo-write grant must match ADR-017 Amendment A8 exactly"
    );
    assert_eq!(
        repo_write_subpaths("csharp").to_vec(),
        vec![RepoWrite::Dir("obj"), RepoWrite::Dir("bin")],
        "csharp's repo-write grant must match ADR-017 Amendment A8 exactly"
    );

    // Everything else keeps a fully read-only repo root.
    for lang in [
        "go",
        "rust",
        "python",
        "typescript",
        "javascript",
        "ruby",
        "dart",
        "swift",
        "objectivec",
        "c",
        "cpp",
        "",
    ] {
        assert!(
            repo_write_subpaths(lang).is_empty(),
            "{lang} must not hold any repo-write grant (ADR-017 Rule 1)"
        );
    }

    // No grant may name the repo root itself or escape it.
    for lang in ["scala", "php", "java", "kotlin", "csharp"] {
        for entry in repo_write_subpaths(lang) {
            let sub = entry.subpath();
            assert!(
                !sub.is_empty(),
                "{lang}: an empty subpath grants the repo root"
            );
            assert!(
                !sub.starts_with('/') && !sub.split('/').any(|c| c == ".."),
                "{lang}: grant `{sub}` escapes the repo root"
            );
        }
    }
}

// 4d. php's grant is one output file, not the repo. A write to any other
// repo-root path must still be denied; the write to index.scip must succeed.
#[test]
#[cfg(target_os = "linux")]
fn sandbox_php_repo_write_is_narrowed_to_index_scip() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let scratch = tempfile::tempdir().expect("scratch");
    let breach_path = repo.path().join("composer.json"); // repo root, not granted
    let allowed_path = repo.path().join("index.scip");

    let breach = build_sandboxed_command(
        "sh",
        &[
            "-c",
            &format!("echo hostile > {} 2>&1; true", breach_path.display()),
        ],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "php",
    );
    match breach {
        Err(SandboxUnavailable(ref msg)) => {
            if std::env::var("CI").is_ok() {
                panic!("sandbox unavailable in CI: {msg}");
            }
            eprintln!("SKIP: {msg}");
            return;
        }
        Ok(spawner) => {
            let _ = spawner.output();
            assert!(
                !breach_path.exists()
                    || std::fs::read_to_string(&breach_path)
                        .unwrap_or_default()
                        .trim()
                        != "hostile",
                "php sandbox allowed a write to a repo-root file outside index.scip"
            );
        }
    }

    let allowed = build_sandboxed_command(
        "sh",
        &["-c", &format!("echo ok > {}", allowed_path.display())],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "php",
    );
    if let Ok(spawner) = allowed {
        let _ = spawner.output();
        assert_eq!(
            std::fs::read_to_string(&allowed_path)
                .unwrap_or_default()
                .trim(),
            "ok",
            "php sandbox blocked the write to index.scip — scip-php has no --output"
        );
    }
}

// 4e. A grant path that is a symlink must never be created or bound. The host
// creates the grant path UNSANDBOXED as the user, and create/open/`--bind` all
// follow links, so a repo shipping `index.scip` as a link to a file in the
// user's home would otherwise get that target created and bound WRITABLE.
#[test]
#[cfg(target_os = "linux")]
fn sandbox_symlinked_repo_write_grant_is_refused() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let outside = tempfile::tempdir().expect("outside");
    let scratch = tempfile::tempdir().expect("scratch");

    // Stands in for ~/.ssh/authorized_keys: outside the repo, does not exist.
    let victim = outside.path().join("authorized_keys");
    std::os::unix::fs::symlink(&victim, repo.path().join("index.scip")).expect("symlink");

    let cmd = build_sandboxed_command(
        "sh",
        &[
            "-c",
            &format!(
                "echo pwned > {}/index.scip 2>/dev/null; true",
                repo.path().display()
            ),
        ],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "php",
    );
    match cmd {
        Err(SandboxUnavailable(ref msg)) => {
            if std::env::var("CI").is_ok() {
                panic!("sandbox unavailable in CI: {msg}");
            }
            eprintln!("SKIP: {msg}");
            return;
        }
        Ok(spawner) => {
            let _ = spawner.output();
        }
    }

    assert!(
        !victim.exists(),
        "a symlinked repo-write grant reached outside the repo — the host followed the link"
    );
}

// 5. Scratch dir is writable
#[test]
#[cfg(target_os = "linux")]
fn sandbox_scratch_dir_is_writable() {
    use travsr_plugin_host::sandbox::linux::build_sandboxed_command;
    // tempdir() reads TMPDIR; serialise against the env-mutating provider tests.
    let _env = env_guard();

    let repo = tempfile::tempdir().expect("tempdir");
    let scratch = tempfile::tempdir().expect("scratch");
    // Inside the sandbox the scratch area is a tmpfs at /travsr-scratch.
    // Verify the sandbox process can write there (status must be 0).
    let cmd = build_sandboxed_command(
        "sh",
        &["-c", "echo ok > /travsr-scratch/write_test.txt"],
        repo.path(),
        scratch.path(),
        &SandboxPolicy::Standard,
        "",
    );

    match cmd {
        Err(_) => {
            eprintln!("SKIP: bwrap not available");
        }
        Ok(spawner) => {
            let status = spawner.status().expect("spawn");
            assert!(
                status.success(),
                "sandbox blocked write to /travsr-scratch — should be writable"
            );
        }
    }
}

// 6. Cache integrity — daemon-computed key; version bump causes miss
#[test]
fn cache_key_is_daemon_computed() {
    use travsr_plugin_host::cache::{CacheKey, ParseCache};
    use travsr_plugin_protocol::ParseResponse;

    let mut cache = ParseCache::new();
    let hash = [1u8; 32];
    let version = "0.5.1";

    let key = CacheKey {
        plugin_version: version.to_string(),
        file_hash: hash,
    };
    cache.insert(key, ParseResponse::default());

    // Same key hits
    assert!(cache.get(version, hash).is_some(), "cache miss on same key");
    // Different version misses
    assert!(
        cache.get("0.5.2", hash).is_none(),
        "version bump should cause cache miss"
    );
    // Different hash misses
    let other_hash = [2u8; 32];
    assert!(
        cache.get(version, other_hash).is_none(),
        "different hash should miss"
    );
}

// 7. RequiresElevated catalog entries are correctly classified. Elevated
// network access is auto-granted for local use (ADR-017 amendment); the runtime
// Elevated sandbox requirement is retained, only the human approval gate is gone.
#[test]
fn elevated_languages_keep_elevated_sandbox_requirement() {
    for lang in &["java", "kotlin", "csharp", "scala"] {
        let entry = lookup(lang).unwrap_or_else(|| panic!("{lang} not in catalog"));
        assert_eq!(
            entry.sandbox,
            SandboxRequirement::RequiresElevated,
            "{lang} should be RequiresElevated"
        );
    }
}

#[test]
fn standard_languages_do_not_require_approval() {
    for lang in &["rust", "go", "python", "php", "ruby", "typescript"] {
        let entry = lookup(lang).unwrap_or_else(|| panic!("{lang} not in catalog"));
        assert_eq!(
            entry.sandbox,
            SandboxRequirement::Standard,
            "{lang} should be Standard sandbox"
        );
    }
}

// 8. SandboxPolicy::Elevated validation rejects empty fields
#[test]
fn elevated_policy_rejects_empty_reason() {
    let policy = SandboxPolicy::Elevated {
        permitted_hosts: vec!["example.com".to_string()],
        reason: String::new(), // empty — should fail
        approved_by: "pse-handle".to_string(),
        approved_date: "2026-05-31".to_string(),
    };
    assert!(
        policy.validate().is_err(),
        "empty reason should fail validation"
    );
}

#[test]
fn elevated_policy_rejects_empty_approved_by() {
    let policy = SandboxPolicy::Elevated {
        permitted_hosts: vec!["example.com".to_string()],
        reason: "test reason".to_string(),
        approved_by: String::new(), // empty
        approved_date: "2026-05-31".to_string(),
    };
    assert!(
        policy.validate().is_err(),
        "empty approved_by should fail validation"
    );
}

#[test]
fn elevated_policy_accepts_valid_fields() {
    let policy = SandboxPolicy::Elevated {
        permitted_hosts: vec!["repo1.maven.org".to_string()],
        reason: "Maven dependency resolution".to_string(),
        approved_by: "pse-handle".to_string(),
        approved_date: "2026-05-31".to_string(),
    };
    assert!(
        policy.validate().is_ok(),
        "valid Elevated policy should pass validation"
    );
}

#[test]
fn elevated_policy_rejects_empty_approved_date() {
    let policy = SandboxPolicy::Elevated {
        permitted_hosts: vec!["repo1.maven.org".to_string()],
        reason: "Maven dependency resolution".to_string(),
        approved_by: "pse-handle".to_string(),
        approved_date: String::new(), // empty — re-review date is mandatory
    };
    assert!(
        policy.validate().is_err(),
        "empty approved_date should fail validation"
    );
}

#[test]
fn elevated_policy_rejects_empty_permitted_hosts() {
    let policy = SandboxPolicy::Elevated {
        permitted_hosts: vec![], // no allowlist — cannot mean "all hosts"
        reason: "Maven dependency resolution".to_string(),
        approved_by: "pse-handle".to_string(),
        approved_date: "2026-05-31".to_string(),
    };
    assert!(
        policy.validate().is_err(),
        "empty permitted_hosts should fail validation"
    );
}

// ADR-017 Rule 1: explicit allowlist only — no wildcards, no CIDR.
#[test]
fn elevated_policy_rejects_wildcard_and_cidr_hosts() {
    for bad in &[
        "*.maven.org",                   // wildcard
        "*",                             // full wildcard
        "10.0.0.0/8",                    // CIDR range
        "https://repo.maven.apache.org", // scheme
        "repo.maven.apache.org:443",     // port
        "repo .maven.org",               // whitespace
    ] {
        let policy = SandboxPolicy::Elevated {
            permitted_hosts: vec![bad.to_string()],
            reason: "test".to_string(),
            approved_by: "pse-handle".to_string(),
            approved_date: "2026-05-31".to_string(),
        };
        assert!(
            policy.validate().is_err(),
            "permitted host '{bad}' should be rejected (no wildcards/CIDR/scheme/port)"
        );
    }
}

#[test]
fn validate_permitted_host_accepts_bare_hostnames() {
    use travsr_plugin_host::sandbox::policy::validate_permitted_host;
    for ok in &[
        "repo1.maven.org",
        "repo.maven.apache.org",
        "plugins.gradle.org",
    ] {
        assert!(
            validate_permitted_host(ok).is_ok(),
            "bare hostname '{ok}' should be accepted"
        );
    }
}

// ── Crash isolation ───────────────────────────────────────────────────────────

#[test]
fn sidecar_stub_disabled_health_is_isolated() {
    use std::path::PathBuf;
    use travsr_plugin_host::transport::{PluginHealth, Sidecar, Transport};
    use travsr_plugin_protocol::{InvokeRequest, ParseRequest};

    // A disabled Sidecar (stub) returns Err without panicking — daemon continues
    let sidecar = Sidecar::stub("java");
    assert!(
        matches!(sidecar.health(), PluginHealth::Disabled(_)),
        "stub sidecar should report Disabled health"
    );

    let req = ParseRequest {
        path: PathBuf::from("Test.java"),
        vname_path: "Test.java".into(),
        corpus: "test".into(),
        package: "".into(),
        source: None,
    };
    // parse() returns Err, not panic — daemon can handle the error and continue
    let result = sidecar.parse(req);
    assert!(
        result.is_err(),
        "disabled sidecar parse should return Err, not panic"
    );

    // invoke_phase_b also returns Err gracefully
    let invoke_req = InvokeRequest {
        corpus: String::new(),
        root: PathBuf::from("."),
        scratch: PathBuf::default(),
        files: None,
    };
    let invoke_result = sidecar.invoke_phase_b(invoke_req);
    assert!(
        invoke_result.is_err(),
        "disabled sidecar invoke_phase_b should return Err"
    );
}

#[test]
fn supervisor_disables_language_after_repeated_crashes() {
    use travsr_plugin_host::supervisor::Supervisor;

    let mut sup = Supervisor::new();
    assert!(!sup.is_disabled("rust"), "rust should start enabled");

    // Record crashes up to the limit (MAX_CRASHES = 3)
    sup.record_crash("rust", "segfault in grammar");
    assert!(!sup.is_disabled("rust"), "not disabled after 1 crash");

    sup.record_crash("rust", "segfault again");
    assert!(!sup.is_disabled("rust"), "not disabled after 2 crashes");

    sup.record_crash("rust", "segfault again");
    assert!(
        sup.is_disabled("rust"),
        "should be disabled after 3 crashes"
    );

    // Other languages unaffected
    assert!(
        !sup.is_disabled("go"),
        "unrelated language should not be disabled"
    );
}
