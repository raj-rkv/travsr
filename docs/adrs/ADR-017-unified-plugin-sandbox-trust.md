# ADR-017: Unified Plugin Sandbox & Trust Model

**Date:** 2026-05-30
**Status:** Proposed
**Phase:** 5 (P5-S1 — structural prerequisite for P5-S2–P5-S5)
**Author:** Principal Security Engineer
**Supersedes:** ADR-006 (rust-analyzer subprocess trust model) — generalised here
**Obviates:** the planned per-tool ADR chain referenced by RFC-008 §7 — ADR-011 (`scip-java` trust), ADR-012 (`scip-typescript` trust), ADR-013 (`scip-kotlin` trust), ADR-015 (bridge plugin panic-isolation), ADR-016 (TOML descriptor trust). These are **not written**; their concerns fold into this single ADR.
**Related:** RFC-011 (two-transport plugin architecture — companion; merges same sprint), RFC-008, RFC-009, ADR-005 (per-language corpus naming — trust keyed per canonical corpus), ADR-006 (the precedent this generalises), CLAUDE.md §Non-Negotiable Principles

---

## Context

RFC-011 replaces the hardcoded indexer + per-tool-trust-ADR model with one `Plugin` contract behind two transports (in-process for first-party Phase A; sidecar + sandbox for Phase B and community plugins). That refactor moves the trust boundary in one direction (every untrusted invoker now sits behind a single transport) and opens a new surface in another (community `--command` binaries; an internal IPC channel; a content-addressed parse cache).

RFC-008 §7 would have required a fresh subprocess-trust ADR per language (ADR-006, then ADR-011/012/013, …). The threat each would describe is identical: **a SCIP/LSIF toolchain executes the indexed repository's `build.rs`, proc-macros, Gradle/Maven scripts, or `setup.py` — arbitrary code the indexer did not author and cannot audit.** Re-arguing that per language is process cost without a security benefit.

This ADR defines **one** sandbox policy and **one** trust-gating model for **every** plugin subprocess — Phase B invokers and community Phase A plugins alike — and the compensating controls that make the in-process transport safe for first-party grammars.

This ADR decides:

1. What sandbox every plugin subprocess runs under, and what happens when the sandbox is unavailable.
2. How trust is granted before a subprocess that executes repo-related code is spawned.
3. Why and under what conditions in-process (un-sandboxed) execution is acceptable for first-party Phase A grammars.
4. How the parse cache and the internal IPC channel are protected.

---

## Decision

### Rule 1 — One sandbox policy for every plugin subprocess

Every Sidecar-transport spawn (RFC-011 §2) — whether a Phase B SCIP/LSIF invoker or a community Phase A plugin — runs under a single `SandboxPolicy::Standard`:

```
SandboxPolicy::Standard
  network:    ALLOW             (intentional — see Amendment A1 below)
  filesystem: repo root         → READ-ONLY (narrow build-output exception for
                                   scala and php only — the authoritative
                                   enumeration is Amendment A6 below, which
                                   supersedes the A5 list)
              scratch tmpdir     → READ-WRITE (per-invocation, removed after)
              everything else    → DENY
  resources:  CPU / RAM / wall-clock caps enforced
  env:        scrubbed allowlist — permitted set:
                PATH    (toolchain discovery)
                LANG    (locale)
                LC_ALL  (locale)
                TMPDIR  (set to the per-invocation scratch dir; NOT the user's $TMPDIR)
              Any variable beyond this list requires explicit justification recorded here.
              No HOME, no CARGO_HOME, no GIT_*, no SSH_*, no AWS_* / GCP_* / AZURE_*,
              no CI, no GITHUB_TOKEN, no NPM_TOKEN, and no other credential-carrying vars.
```

> **Amendment A1 — Network allow for Standard/NativeIpc (2026-06-11)**
>
> Network egress is **intentionally allowed** in `Standard` and `NativeIpc` sandbox policies.
>
> **Rationale:** Every language build tool that Phase B invokes (`go mod download`, `npm install`,
> `pip install`, `pub get`, Maven/Gradle for scip-java) requires outbound network access to
> resolve and fetch dependencies at analysis time. Blocking network caused 0 edges on
> multi-package repos (e.g. Kubernetes: 2255 packages, 0 LSIF edges → 426,636 after fix).
>
> **Compensating controls:**
> - Filesystem confinement via bwrap/sandbox-exec still fully applies — the plugin cannot
>   write outside its scratch dir or read outside the repo root.
> - `Elevated` policy continues to exist for documenting host-allowlists; host-level egress
>   controls (firewall / egress proxy) remain the network boundary.
> - The sentence in Rule 1 below ("disable the network-deny rule entirely, it cannot be
>   granted under Elevated — escalate to CTO") referred to ad-hoc Elevated exceptions that
>   bypass documented allowlists. Standard's unconditional network-allow is not an Elevated
>   exception; it is an explicit policy change recorded here and enforced uniformly.
>
> **Approved by:** Principal Security Engineer (PSE review 2026-06-11, branch
> `feature/travsr-daemon-init-at-scale-295`)

> **Amendment A2 — Windows sandbox FFI unsafe-code sanction (2026-08-05)**
>
> The Windows sandbox mechanism (AppContainer + Job Object, the Windows row of the
> table below) is implemented against raw Win32 APIs — `CreateAppContainerProfile`,
> `CreateProcessW` with `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES`,
> `SetNamedSecurityInfoW`/`GetAce`, `CreateJobObjectW`, `OpenProcess` — none of which
> have safe standard-library equivalents. This amendment sanctions `unsafe` code in
> the `travsr-plugin-host` workspace under the following invariants:
>
> 1. **Confinement:** every `unsafe` block lives in exactly one file,
>    `crates/travsr-plugin-host/src/sandbox/windows/ffi.rs`. The crate root carries
>    `#![deny(unsafe_code)]`; only that file carries the `#![allow(unsafe_code)]`
>    override. Any second override site re-opens this amendment.
> 2. **Encapsulation:** the file exposes only safe wrappers; OS handles, SIDs, ACLs,
>    and attribute lists are owned by RAII types (`OwnedHandle`, `AppContainerSid`,
>    `AttrList`, `OwnedSecurityCapabilities`) so no raw resource outlives its owner.
> 3. **Verification:** invariants with a history of violation are pinned by unit
>    tests running against real OS objects (capability-pointer stability after moves,
>    Job Object limit read-back, DACL ACE round-trips, process liveness).
>
> This corrects a stale citation: the override previously cited RFC-014, which covers
> Phase B symbol unification and says nothing about unsafe or FFI. ADR-017 is the
> governing document for the sandbox mechanism, so the sanction is recorded here.
>
> **Approved by:** _pending Tech Lead sign-off (raised in PR #577 review; drafted
> 2026-08-05)_

> **Amendment A3 — per-language toolchain env forwarding (2026-08-05)**
>
> Rule 1's scrubbed env allowlist (PATH / LANG / LC_ALL / TMPDIR) is extended by the
> **daemon-computed** per-language toolchain variables in
> `crates/travsr-plugin-host/src/sandbox/toolchain.rs` (`ToolchainAccess::env`):
> `HOME`, `GOPATH`, `GOCACHE`, `GOMODCACHE`, `GOROOT`, `JAVA_HOME`,
> `GRADLE_USER_HOME`, `SBT_OPTS`, `COMPOSER_HOME`, `NUGET_PACKAGES`, `DOTNET_ROOT`,
> `GEM_HOME`, `GEM_PATH`, `CARGO_HOME`, `RUSTUP_HOME`, `NVM_DIR`, `PYENV_ROOT`,
> and `TRAVSR_DART_EMITTER`.
>
> **Rationale:** Phase B analyzers drive the language's real build tool, which
> resolves module/build caches through these variables; without them the analyzer
> resolves zero packages and emits an empty index (the scip-go 0-of-244 case that
> created `toolchain.rs`). These are location pointers computed by the daemon from
> the same paths that receive the sandbox's filesystem grants — they are **not** an
> ambient-environment passthrough, and the filesystem confinement (not the env)
> remains the enforcement boundary: a forwarded `HOME` value does not grant access
> to anything the FS rules deny. The Rule 1 exclusion of credential-carrying
> variables (`GIT_*`, `SSH_*`, `AWS_*`/`GCP_*`/`AZURE_*`, `CI`, `GITHUB_TOKEN`,
> `NPM_TOKEN`, …) is unchanged and still absolute; `HOME` and `CARGO_HOME` move
> from the "never" list to this justified set.
>
> Linux (`sandbox/linux.rs`) and macOS (`sandbox/macos.rs`) have forwarded this set
> since `toolchain.rs` was introduced; Windows (`sandbox/windows/ffi.rs`) matches as
> of #501. This amendment makes that shipped behavior the recorded policy instead of
> an undocumented divergence (raised in PR #577 review).
>
> **Approved by:** _pending Tech Lead sign-off (raised in PR #577 review; drafted
> 2026-08-05)_

> **Amendment A4 — Windows unsandboxed-by-consent Phase B path (2026-08-20)**
>
> Rule 2 forbids any path that runs a plugin subprocess un-sandboxed as a
> *fallback for a missing sandbox*. This amendment records a **narrow,
> consent-gated exception** on **Windows only**, where the AppContainer sandbox
> (Amendment A2) cannot host the JVM/.NET build tools some analyzers must run
> (Gradle for scip-java, sbt for scala, scip-dotnet for C#). On such hosts the
> catalog marks the analyzer `WindowsSandbox::Unsupported`, and unamended Rule 2
> would disable the language entirely.
>
> **Scope (all four must hold — the exception is otherwise never taken):**
> 1. **Windows only.** Linux and macOS keep Rule 2 fail-closed unchanged; there
>    is no unsandboxed path on those platforms. `decide_windows_sandbox` returns
>    `Sandboxed` for `!is_windows`.
> 2. **Unsupported analyzer only.** Taken only when `decide_windows_sandbox` sees
>    `WindowsSandbox::Unsupported`. Where the Windows sandbox *is* available
>    (`WindowsSandbox::Supported`), it remains the default and this path is not
>    reached.
> 3. **Explicit consent on record.** Off by default. The child spawns only after
>    a persistent per-language grant (`travsr lang allow-unsandboxed <lang>`,
>    stored in `~/.travsr/lang.toml`) or a session opt-in (`--allow-unsandboxed`
>    / `TRAVSR_ALLOW_UNSANDBOXED=1`). With no grant the decision is `NeedsConsent`
>    and the language is skipped with an honest "needs your permission" status.
> 4. **Consent is never repo-resident.** The grant lives in the user's
>    `~/.travsr/` config, never in the indexed repo — a repo cannot opt *itself*
>    into unsandboxed execution (the Rule 3 invariant is preserved).
>
> **Consent semantics.** The grant is the user's informed decision to run that
> language's own build tool with their own privileges — the same trade-off the
> project already accepts for the rust `--allow-unsandboxed` LSIF path (the
> ADR-006 lineage this mirrors). It is **per-language**, not per-corpus; tighter
> per-corpus granularity is tracked as follow-up UX hardening. `travsr lang
> allow-unsandboxed` explains the trade-off and requires interactive `[y/N]`
> confirmation before recording the grant (or an explicit `--yes` to grant it
> non-interactively; with no terminal and no `--yes` the grant is refused, never
> recorded silently). `travsr lang list` surfaces the language as running by
> unsandboxed consent.
>
> **Env policy.** The unsandboxed child does **not** inherit the daemon's ambient
> environment. `build_unsandboxed_command` (`sandbox/mod.rs`) calls `env_clear()`
> and forwards only: (a) the daemon-computed toolchain variables of Amendment A3
> (`ToolchainAccess::env`); (b) `HOME`, `GRADLE_USER_HOME` and `PATH`, set
> explicitly; and (c) a fixed allowlist of OS-essential, non-credential variables
> (`is_allowed_passthrough_env`: `SYSTEMROOT`, `PATHEXT`, `TEMP`, `USERPROFILE`,
> `APPDATA`, `PROGRAMFILES`, … — matched case-insensitively). The Rule 1
> credential exclusions (`GITHUB_TOKEN`, `AWS_*`/`GCP_*`/`AZURE_*`, `SSH_*`,
> `NPM_TOKEN`, `GIT_*`, …) remain **absolute**: they are outside the allowlist and
> so are dropped, and no daemon secret reaches Gradle/sbt/scip-dotnet.
>
> **Residual risk (accepted).** On a consented Windows host the analyzer and the
> repo build it drives run with the user's privileges, and the Rule 1 filesystem
> confinement does not apply — the same residual risk already accepted for the
> rust `--allow-unsandboxed` LSIF path, now extended to the JVM/.NET sidecars on
> hosts where the sandbox cannot host them. The exception stays bounded by the
> four scope conditions above and the secret-scrubbed environment.
>
> **Approved by:** Principal Security Engineer (PSE review 2026-08-21, PR #743).
> Scope, consent semantics, and env policy reviewed against the implementation
> (`decide_windows_sandbox`, `build_unsandboxed_command`, the env allowlist tests)
> and found to match. Windows-only, consent-gated, secret-scrubbed; the residual
> risk is the same one already accepted for the rust `--allow-unsandboxed` LSIF
> path (ADR-006 lineage), bounded by the four scope conditions.

> **Amendment A5 — Narrow build-output write for scala (2026-08-21)**
>
> Rule 1 pins the repo root READ-ONLY. sbt (the scala Phase B driver) cannot
> compile without writing its build outputs into the repo tree, so a strict
> read-only root leaves scala with no working Phase B. Rather than flip the whole
> root writable, the sandbox grants write to a **typed, fixed allowlist of
> build-output subpaths only** (`toolchain::repo_write_subpaths("scala")`):
>
> ```
> target/                    (sbt compile output)
> project/target/            (sbt meta-build output)
> .travsr-semanticdb.sbt     (the generated SemanticDB-enable settings file)
> ```
>
> Everything else under the repo root stays READ-ONLY on both bwrap (Linux) and
> Seatbelt (macOS); every other language keeps a fully read-only root
> (`repo_write_subpaths` returns empty for them). The grant is source-defined and
> compile-time, so a repo cannot widen it (Rule 3 invariant preserved).
>
> **Residual risk (accepted).** A hostile scala repo's own `build.sbt` runs during
> indexing (that is inherent to compiling it) and can now write under `target/`,
> `project/target/`, and the one settings file — build directories a compile would
> write anyway. It still cannot modify source, `.git`, or any path outside those
> subpaths. Pinned by `sandbox_scala_repo_write_is_narrowed_to_build_subpaths`
> (Linux, runs bwrap on CI): a repo-root write outside the allowlist is denied, an
> allowed `target/` write succeeds.
>
> **Approved by:** Principal Security Engineer (PSE review 2026-08-21, PR #743).
> The narrowed subpath grant and its enforcement test were reviewed and found to
> confine writes to build outputs only, with source and VCS metadata protected.

> **Amendment A6 — Revised repo-write enumeration: scala crossproject, php, and
> a symlink guard (2026-09-10)**
>
> Amendment A5 authorised exactly three repo-relative write subpaths, all for
> scala. Two facts observed since then put real Phase B runs outside that list,
> so the authorised enumeration is revised here rather than drifting in code.
>
> **1. scala is a crossproject build, not a single-module one.** An sbt
> crossproject writes its SemanticDB output per platform. On the pinned
> scala-parser-combinators fixture all 150 `.semanticdb` files land under
> `js/target`, `jvm/target` and `native/target` and NONE under the granted
> `target/`, and sbt's meta-build of the meta-build writes `project/project/target`.
> Under A5 those writes take EROFS on Linux (the repo root is a `--ro-bind`);
> macOS runs scala under `Elevated`, which skips Seatbelt, so the mismatch was
> invisible there.
>
> **2. php has a repo-relative output with no redirect flag.** `scip-php` has no
> `--output`: it hardcodes `index.scip` relative to its working directory, which
> must be the repo for it to find `composer.json` at all. Denied, its
> `file_put_contents` returns false with a zero exit status, i.e. a silent empty
> index rather than a reported failure. The sidecar moves the file into scratch
> and removes it, so nothing survives the run.
>
> The authorised set is therefore (`toolchain::repo_write_subpaths`):
>
> ```
> scala:  target/                    (sbt compile output)
>         project/target/            (sbt meta-build output)
>         project/project/target/    (sbt meta-build of the meta-build)
>         js/target/                 (crossproject, JS platform)
>         jvm/target/                (crossproject, JVM platform)
>         native/target/             (crossproject, Native platform)
>         .travsr-semanticdb.sbt     (the generated SemanticDB-enable settings file)
>
> php:    index.scip                 (scip-php's only output path)
> ```
>
> Every other language keeps a fully read-only repo root
> (`repo_write_subpaths` returns empty for them), and everything under the root
> outside this list stays READ-ONLY on both bwrap (Linux) and Seatbelt (macOS).
> The grants are compile-time `&'static str` matched on `language` alone, with no
> `..` and no repo-controlled input, so a repo still cannot widen its own grant
> (Rule 3 invariant preserved).
>
> **Symlink guard (new requirement).** The host creates each grant path before
> binding it, and it does so UNSANDBOXED as the user. `create_dir_all`,
> `OpenOptions::open` and bwrap's `--bind` source resolution all follow symlinks,
> so a repo shipping `index.scip` (or `target/`) as a link to
> `~/.ssh/authorized_keys`, `~/.bashrc` or `~/.travsr/lang.toml` would have that
> target created and then bind-mounted WRITABLE into the sandbox, defeating the
> A1 compensating control and the A5 residual-risk statement. The Linux path now
> rejects a grant whose path has a symlink at any component, and re-stats the leaf
> after creating it, skipping the bind with a warning rather than binding it. It
> fails closed: a skipped grant costs that language its build output. macOS was
> never exposed, since a Seatbelt `(literal ...)` rule matches the resolved path.
>
> **Residual risk (accepted).** A hostile scala repo's own `build.sbt`, and a
> hostile php repo's composer configuration, run during indexing (inherent to
> analysing them) and can write under the listed build directories and the two
> named files. They still cannot modify source, `.git`, or any path outside the
> list. php's single grant is a file, not a directory. Pinned by
> `repo_write_grants_are_exactly_the_authorised_set` (all platforms, enumerates
> the list so a future widening must be a deliberate test edit),
> `sandbox_scala_repo_write_is_narrowed_to_build_subpaths` and
> `sandbox_php_repo_write_is_narrowed_to_index_scip` (Linux, run bwrap on CI), and
> `sandbox_symlinked_repo_write_grant_is_refused` (Linux).
>
> **Scope of the symlink guard (Security review, 2026-09-11).** It closes the
> threat this ADR models, which is hostile repo *content*: a link committed to
> the repository, and so present before the run starts, is refused at every
> component of the grant path. It does NOT close a concurrent local attacker who
> swaps an intermediate component between the component walk and the
> `create_dir_all`, because the post-create re-stat checks the leaf only and
> resolves the components above it. That attacker already has code execution as
> the user on the machine being indexed, which is outside this ADR's model, so it
> is accepted rather than mitigated. Closing it properly needs
> `openat2(RESOLVE_NO_SYMLINKS)` on Linux and the equivalent elsewhere. The
> earlier wording here claimed the guard meant a grant path "can no longer reach
> outside the repo", which overstates it; that sentence is removed above.
>
> **The host writes into the checkout to establish these grants (Security review,
> 2026-09-11).** bwrap needs its bind source to exist and a Windows ACL can only
> be set on an existing object, so the host `create_dir_all`s every `Dir` grant
> and touches every `File` grant, as the user and UNSANDBOXED, on every run of
> that language. Indexing a scala repo therefore creates up to six directories
> and one file in the working tree, and nothing removes them. Not an escape, but
> it is unrequested mutation of the user's tree by the mechanism that exists to
> prevent mutation, and it dirties `git status`. Accepted for now, tracked for
> whoever next touches this path.
>
> **`.travsr-semanticdb.sbt` is already vestigial (Security review,
> 2026-09-11).** travsr-lang#31 (commit `e00ccfc`) replaced the injected settings
> file with an `sbt` command line, so no current sidecar writes that name. The
> grant is retained deliberately: already-released sidecars still inject the file
> and would take EROFS on Linux without it. Drop it once a minimum sidecar
> version is enforced. The enumeration above therefore lists one grant that is
> live for old sidecars only.
>
> **Approved by:** _pending Principal Security Engineer sign-off (drafted
> 2026-09-10 with the fix for the symlinked-grant escape)._

> **Amendment A7 — Windows honours the A6 enumeration per subpath, not as a
> boolean (2026-09-10)**
>
> A6 authorises specific repo-relative subpaths. Linux binds each one and macOS
> emits a Seatbelt rule for each one, but the Windows AppContainer path read the
> enumeration through a `needs_repo_write(language) -> bool` helper and, when it
> was true, granted `ACCESS_GENERIC_ALL` on the **repo root**. That is Rule 1
> inverted on one platform: a hostile `composer.json` or composer plugin
> executed during indexing could rewrite any file in the repo, including source
> and `.git`, where the same language on Linux and macOS may write one file.
>
> The helper's own comment justified the coarse bool on the grounds that "scala
> is `WindowsSandbox::Unsupported` there and never reaches it". That stopped
> being true when A6 added php, which is `WindowsSandbox::Supported`.
>
> Windows now grants `ACCESS_GENERIC_READ` on the repo root unconditionally and
> a separate `ACCESS_GENERIC_ALL` per authorised subpath, inheritable for a
> directory grant and this-object-only for a file grant. As on Linux, the host
> materialises each path first (an ACL can only be set on an object that
> exists), which is also what lets the root stay read-only: the analyzer opens
> an existing file rather than needing `FILE_ADD_FILE` on the directory. The
> symlink guard A6 introduced is now shared by both platforms
> (`toolchain::grant_path_has_symlink`) instead of living inside the Linux
> builder, since Windows creates the same paths as the same unsandboxed user.
> `needs_repo_write` is deleted: with no caller left, keeping it would preserve
> the shape that caused this.
>
> Not verified by execution. The AppContainer tests live in
> `sandbox-windows.yml`, which is `workflow_dispatch` only and does not run on
> pull requests, so this change is covered by a Windows-target type-check and by
> the portable guard unit test, not by a spawn on Windows. Running that workflow
> before merge is the remaining verification.
>
> **A `File` grant is wider than it needs to be (Security review, 2026-09-11).**
> `ACCESS_GENERIC_ALL` on a file maps to `FILE_ALL_ACCESS`, which includes
> `DELETE`. A sidecar has no legitimate reason to unlink a file in the user's
> repository, and granting it opens a one-way door: the delete succeeds, and the
> recreate then needs `FILE_ADD_FILE` on the repo root, which is deliberately
> only `ACCESS_GENERIC_READ`. travsr-lang#31 hit exactly that and destroyed the
> user's `index.scip` on Windows before it was corrected on the sidecar side.
> The durable fix belongs here, not there: a `RepoWrite::File` should be granted
> read plus write WITHOUT `DELETE`, so the class is unreachable whatever a
> sidecar does. Recorded as SEV-4 (defence in depth) rather than a merge blocker,
> because the sidecar-side fix removes the live exploit path and narrowing the
> mask is a change to the trust-boundary crate that deserves its own review.
>
> **Type-check confirmed (Security review, 2026-09-11).**
> `cargo check --target x86_64-pc-windows-gnu -p travsr-plugin-host --all-targets`
> is clean, so the claim above is now evidenced rather than asserted. It is still
> not a spawn: running `sandbox-windows.yml` before merge remains required.
>
> **Approved by:** _pending Principal Security Engineer sign-off (drafted
> 2026-09-10)._

> **Amendment A8 — java, kotlin and csharp get the same build-output grant scala
> already has (2026-09-11)**
>
> These three drive the project's own build tool (maven, gradle, dotnet), which
> writes its output into the project. They held no repo-write grant at all, so on
> Linux the `--ro-bind` repo root made javac and the SemanticDB compiler plugin
> take EROFS and the language indexed to nothing. All three are
> `RequiresElevated`, and macOS skips `sandbox-exec` entirely for that policy, so
> the gap never showed on the development platform.
>
> The authorised set gains:
>
> ```
> java:   target/ build/ .gradle/
> kotlin: build/ .gradle/ .kotlin/
> csharp: obj/ bin/
> ```
>
> **Measured, not reasoned.** On Linux under bwrap (arm64, Debian, maven 3.9),
> with `target/` bound over a read-only repo root: writing new files inside it,
> deleting files inside it, and deleting its whole CONTENTS all succeed, while a
> write outside the grant is still denied (EROFS). The single operation that
> fails is removing the `target` DIRECTORY itself, because that is a write to the
> read-only parent. By default maven-clean-plugin treats that as fatal:
> `Failed to delete /repo/target`, BUILD FAILURE, no index at all.
>
> The sidecar therefore passes `-Dmaven.clean.failOnError=false`
> (travsr-lang#31). Re-measured with it: clean still clears the contents, the
> undeletable directory degrades to a WARNING, javac runs, BUILD SUCCESS. The
> contents are all `clean` was needed for, since scip-java passes
> `-Dmaven.compiler.useIncrementalCompilation=false` and clearing the classes is
> what makes every source stale again. Gradle needs no equivalent, because
> scip-java drives it through `scipCompileAll` and never runs `clean`.
>
> **Residual risk (accepted).** A hostile `pom.xml`, `build.gradle` or `.csproj`
> executes during indexing, which is inherent to analysing those projects, and
> can now write under the listed build directories. It still cannot touch source,
> `.git`, or anything outside the list. This is strictly LESS exposure than the
> status quo on macOS, where all three run with no filesystem confinement at all.
> The grants are compile-time `&'static str` matched on `language`, with no `..`
> and no repo-controlled input, so Rule 3 holds.
>
> **All three verified on Linux (2026-09-11).** kotlin and csharp were measured
> the same way, on the same host, rather than inferred from java.
>
> csharp needed nothing beyond `obj/` and `bin/`: `dotnet build --no-restore`
> inside the sandbox emits its assembly and reports `Build succeeded, 0 Errors`.
>
> kotlin needed a directory that inference had missed, and this is the reason
> the list is measured rather than reasoned. With `build/` and `.gradle/` granted
> and outputs wiped to force a real compile, `compileKotlin` FAILED on
> `java.nio.file.FileSystemException:
> /repo/.kotlin/sessions/kotlin-compiler-*.salive: Read-only file system`. The
> Kotlin Gradle Plugin opens a build session under `<project>/.kotlin/`. Adding
> that one directory turns the same build into `BUILD SUCCESSFUL` with the class
> file written. An earlier run that reported success without it proved nothing:
> every task was `UP-TO-DATE`, which is why the outputs are wiped first.
>
> `.kotlin/` is deliberately NOT on java's list. A pure java gradle build never
> loads that plugin. A mixed java/kotlin repo that turns out to need it is one
> line, added when it is observed rather than guessed at now.
>
> **Supersedes** the earlier review position that this needed an RFC on the grant
> mechanism before anything could be granted. That position rested on an
> unexecuted reading of the bwrap source and, once run, cost four lines and one
> maven flag.
>
> Pinned by `repo_write_grants_are_exactly_the_authorised_set`.
>
> **Approved by:** _pending Principal Security Engineer sign-off (drafted
> 2026-09-11)._

Mechanism by platform (DevOps owns the implementation, Security owns the policy):

| Platform | Primary mechanism | Fallback |
|---|---|---|
| Linux (incl. OCI A1 / aarch64) | **bubblewrap** (outer namespace + seccomp-bpf container) + **Landlock** FS rules (additive, if kernel ≥ 5.13) | bubblewrap without Landlock if kernel < 5.13; plugin **disabled** (fail-closed) if bubblewrap is absent |
| macOS | `sandbox-exec` (Seatbelt) profile | **plugin disabled (fail-closed per Rule 2)** — if `sandbox-exec` is absent or returns non-zero at spawn time, treat identically to a missing sandbox: disable the plugin, emit `tracing::warn!`, surface as `disabled (sandbox unavailable)` in `travsr language list` |

The policy is defined **once**, reviewed **once**, and applied at every spawn. Adding a language does not re-open the policy. A language whose toolchain needs an exception (e.g. legitimate network access to fetch a toolchain component) does not get a new ADR — it gets a reviewed, named exception recorded in `travsr.toml` and surfaced to the user; the **default** is always `Standard`.

`SandboxPolicy` is defined normatively as an enum; `Elevated` is the exception variant:

```rust
pub enum SandboxPolicy {
    /// The default — applied to every Sidecar spawn unless an explicit exception is approved.
    Standard,
    /// Exception variant — requires PSE sign-off before any implementation PR merges.
    Elevated {
        /// Explicit allowlist of hosts the plugin may reach. No wildcards. No CIDR ranges.
        /// Example: vec!["repo1.maven.org".to_string(), "plugins.gradle.org".to_string()]
        permitted_hosts: Vec<String>,
        /// One-sentence human-readable justification recorded in travsr.toml and shown in
        /// `travsr language list`. Required; empty string is rejected at parse time.
        reason: String,
        /// GitHub username/handle of the Security reviewer who approved this exception.
        approved_by: String,
        /// ISO-8601 date the approval was recorded (e.g. "2026-06-01"). Approvals older
        /// than 12 months require re-review.
        approved_date: String,
    },
}
```

Approval requirement: any use of `SandboxPolicy::Elevated` must be reviewed and signed off by the Principal Security Engineer before the implementation PR merges. Self-approval is forbidden. If an exception would require a wildcard host (e.g. `*.gradle.org`) or disable the network-deny rule entirely, it cannot be granted under `Elevated` — escalate to CTO.

> **Amendment A5 (local auto-grant of the elevated approval).** The per-user
> approval gate above (the `travsr lang approve` step and the interactive/extension
> consent form for java/kotlin/scala/csharp) is **auto-granted for local use**. The
> resolver synthesizes the `Elevated` policy from the catalog's default hosts with
> sentinel audit fields (`approved_by: "auto"`, `reason: "auto-approved"`), so
> installing or indexing these four languages is frictionless on every surface. The
> **runtime** `Elevated` sandbox policy is unchanged — only the human approval moment
> is removed. This is a design-time PSE sign-off for the four first-party languages,
> not a per-user one; the class of languages and their host allowlists still live in
> the catalog and are reviewed here, not chosen by the repo.
>
> Two honest caveats, recorded rather than glossed:
> - The `permitted_hosts` allowlist has never been enforced by any shipped sandbox
>   backend (each logs that it relies on an external firewall/egress proxy Travsr does
>   not ship), so removing the consent moment loses no traffic filtering that existed.
> - On **macOS**, `Elevated` already skips `sandbox-exec` and runs the analyzer with
>   ulimit caps only (the JVM/sbt filesystem needs cannot be expressed in a Seatbelt
>   profile). So these four already ran with reduced isolation on macOS, and now they
>   do so with **no consent moment**. That is the one genuine security delta; it is
>   consistent with a local-first tool but is not cosmetic. On Linux, `Elevated` and
>   `Standard` are already behaviorally identical (bwrap FS confinement retained).

### Rule 2 — Fail-closed (non-negotiable)

If the sandbox mechanism is unavailable on the host (missing `bwrap`, kernel without seccomp, `sandbox-exec` failure), the affected plugin is **disabled** and its files are **not indexed**. There is **no path** that runs a plugin subprocess un-sandboxed as a fallback for a missing sandbox. (The one narrow, Windows-only, consent-gated exception — for analyzers the AppContainer sandbox cannot host — is recorded in **Amendment A4** above; it is an explicit user grant, not a silent fallback.)

> The "trust the user's local toolchain because the sandbox tool is missing" fallback **is** the vulnerability. It is forbidden. (Security hard rule; CLAUDE.md #3 local-first.)

The daemon emits a `tracing::warn!` naming the plugin and the missing mechanism, and `travsr language list` shows the language as `disabled (sandbox unavailable)`. Phase A for *first-party* languages is unaffected because it runs in-process (Rule 4), not in a sandbox — the structural graph is always available.

### Rule 3 — Trust is granted per canonical corpus, before spawn

A subprocess that executes repo-related code is spawned **only** after an explicit, persistent trust grant, keyed per canonical corpus (ARCH-102), reusing the ADR-006 primitive:

```
travsr config set plugins.trust.<canonical-corpus> true     # Phase B invokers
travsr language add <lang> --command <binary>               # community plugin: explicit opt-in
```

- **The daemon never enables a code-executing subprocess based on content found inside the repo itself.** A repo cannot opt *itself* into Phase B or into a community plugin. (Direct inheritance of ADR-006 Rule 1.)
- Trust is **per corpus**, not global: trusting `github.com/me/my-repo` does not trust a dependency checkout.
- Community `--command` binaries are **never** auto-discovered. The user names the binary explicitly; the daemon records the grant; the binary then runs under `SandboxPolicy::Standard` like any Phase B invoker.

### Rule 4 — In-process is permitted only for first-party, fuzzed grammars

The in-process transport (RFC-011 §10) runs first-party Tree-sitter grammars (C, via FFI) in the daemon's address space with **no** sandbox. This is accepted because Phase A executes **no untrusted code** — it parses bytes — and because the residual risk (a memory bug in a C grammar triggered by crafted source) is contained by mandatory compensating controls:

In-process eligibility requires **all** of:

1. **First-party** — the grammar crate is a workspace dependency carried in the monorepo (never a `--command` plugin).
2. **Pinned** — exact patch version (e.g. `=0.23.4`; `x` is not valid Cargo semver after `=`), `cargo-deny` advisory gate in CI (Tree-sitter grammar CVEs have shipped historically — RFC-003 §7). The pinned version must be updated in `deny.toml` on every grammar bump and reviewed in the PR that changes it.
3. **Fuzzed** — a `cargo-fuzz` target exists for the grammar under `fuzz/` and runs in the nightly fuzz workflow.
4. **Fixture-gated** — golden fixtures (RFC-003 §6) gate every change.

Any grammar that cannot meet all four runs under the **Sidecar** transport instead (fault-isolated, §10 RFC-011), trading IPC cost for crash isolation. The choice is per-grammar and reversible at the transport boundary.

### Rule 5 — Cache and IPC integrity

- **Cache keys are daemon-computed.** The parse cache (RFC-011 §6) is keyed by `(plugin_version, sha256(file))` where the `sha256` is computed by the **daemon**, never reported by the plugin. A plugin cannot select or forge a cache slot. `plugin_version` (from the handshake) is a hard invalidation component; a plugin-logic change that fails to bump it is a freshness bug, not a security bypass, but the daemon additionally records the running `plugin_version` in graph metadata so drift is detectable (`travsr status`).

  **CI enforcement gate (normative):** The CI pipeline MUST compute a content hash of each plugin crate's `src/` tree (e.g. `sha256sum $(find crates/travsr-plugin-<lang>/src -type f | sort)`) and store it alongside `plugin_version` in a `plugin-hashes.lock` file committed to the repo. On every CI run the pipeline re-hashes the source tree and asserts it matches the recorded hash if and only if `plugin_version` is unchanged. A source-tree change with no `plugin_version` bump fails CI with a mandatory error message naming the affected plugin. This makes the "forgot to bump" failure mode a build error rather than a silent stale-cache bug.
- **Protocol version is fail-fast.** A plugin whose `protocol_version` the daemon does not support is refused at registration (RFC-011 §4) — never driven with a mismatched contract that could mis-decode into forged nodes/edges.
- **The IPC channel is not externally reachable.** stdin/stdout to a spawned child; no listening socket; carries no client data. It does not widen the network attack surface and does not breach MCP-only (RFC-011 §9).

---

## Threat model — rows touched / added (T-table)

| Row | Asset | Threat | Likelihood | Impact | Mitigation | Δ |
|---|---|---|---|---|---|---|
| **T4** | Developer machine | Untrusted code execution via indexed repo's build scripts/proc-macros (Phase B) | Med | High | `SandboxPolicy::Standard` (Rule 1), fail-closed (Rule 2), per-corpus trust before spawn (Rule 3) | **improved** — now structural, not per-tool |
| **T11 (new)** | Developer machine | Malicious community plugin binary (`--command`) running in the daemon's authority | Med (supply chain) | High | Same sandbox as Phase B (Rule 1); explicit per-corpus opt-in (Rule 3); never in-process (Rule 4); fail-closed (Rule 2) | new |
| **T12 (new)** | Graph integrity | Parse-cache poisoning — a forged/stale `(plugin_version, sha256)` entry serving fabricated nodes/edges | Low | Med | Daemon-computed hash, plugin never supplies keys; mandatory version bump on logic change (Rule 5) | new |
| **T13 (new)** | Graph integrity / availability | Malformed source triggers a memory bug in an in-process C grammar → daemon RCE/crash | Low | High | In-process restricted to first-party + pinned + fuzzed + fixture-gated (Rule 4); else run sidecar | new |

---

## Findings carried from the RFC-011 design review

| # | Severity | Finding | Required mitigation | Blocks merge |
|---|---|---|---|---|
| 1 | SEV-1 | Any "sandbox missing ⇒ run plugin anyway" path | Fail-closed; disable the plugin (Rule 2) | **yes** |
| 2 | SEV-2 | In-process grammar runs attacker-controlled source in-daemon (C/FFI) | Restrict in-process to first-party fuzzed grammars (Rule 4); document crash domain (RFC-011 §10) | **yes** |
| 3 | SEV-2 | `--command` community binary is arbitrary local code | Per-corpus trust opt-in + `SandboxPolicy::Standard` (Rules 1, 3) | **yes** |

---

## Tests Security requires QA to add

- **Egress allowed (Standard/NativeIpc):** a plugin running under `Standard` or `NativeIpc` policy CAN reach the network — this is intentional per Amendment A1. The test `sandbox_standard_allows_network` verifies this. `Elevated` policy follows the same allow rule; host-level egress controls enforce the permitted-hosts list.
- **Fail-closed:** with the sandbox mechanism forcibly unavailable, the plugin is disabled and indexes **zero** of its files — assert no fallback path runs the subprocess.
- **Trust gate:** a `--command` plugin with no `plugins.trust.<corpus>` / `--command` opt-in is **refused**; a repo cannot opt itself into Phase B via committed config.
- **FS confinement:** a plugin attempting to write inside the repo root (outside the scratch tmpdir) fails; repo is read-only.
- **Resource caps:** a plugin exceeding the wall-clock cap is killed and the file marked failed without hanging the index run.
- **Cache integrity:** a plugin-supplied cache key is ignored; only the daemon-computed `(plugin_version, sha256)` selects a slot.

(These are merge-gating for any sprint landing a Sidecar plugin — P5-S1 onward.)

---

## Supply-chain check (applies per added language/plugin)

- [ ] `cargo-deny` passes (advisories, licenses, bans) — including each pinned `tree-sitter-<lang>` grammar.
- [ ] New grammar/invoker deps fit the MIT/Apache-2.0/BSD-3-Clause/ISC allowlist.
- [ ] `cargo-fuzz` target present for any in-process grammar (Rule 4.3).
- [ ] Release artifact sigstore-signed + SLSA provenance (unchanged release bar).

---

## Consequences

**Positive:**
- One sandbox review instead of N. Adding a language is no longer gated on a bespoke trust ADR — it inherits `SandboxPolicy::Standard`, and only a genuine exception (network, elevated FS) triggers a (named, recorded) review. This unblocks RFC-011's P5-S3–P5-S5 language additions without per-language Security re-litigation.
- Net posture **improvement** over RFC-008: the sandbox boundary is structural (enforced at the transport layer for *every* untrusted spawn) rather than per-tool and easy to forget.
- The fail-closed rule removes the most common real-world sandbox bypass (the "tool missing, run anyway" fallback).

**Negative / accepted:**
- The in-process transport keeps a shared crash domain for first-party grammars (T13). Accepted under Rule 4's compensating controls; the escape hatch (run that grammar sidecar) is always available at the transport boundary.
- A genuinely new threat surface (community `--command` plugins) exists, but is gated (Rule 3) and deferred to Phase 5 for the *ecosystem* (publishing/registry); Phase 4 ships first-party only, so the surface is dormant until the trust root is designed.

---

## Escalations

- **Authenticity vs. execution.** This ADR sandboxes *execution*. Plugin *authenticity* (signing, a registry trust root for community plugins) is **out of scope** and deferred to a Phase 5 ADR, gated by the CTO's decision (2026-05-30) to defer the community SDK/registry. Until then, `--command` plugins are a local, explicit, per-corpus opt-in only.
- **Elevated-sandbox exceptions.** Any language requiring `SandboxPolicy::Elevated` returns to Security for the *exception* (not a full per-language ADR). If an exception would force off the OCI free tier or slip a public commitment > 1 sprint, escalate to CTO + PM.

---

## Verdict

**APPROVED_WITH_MITIGATIONS** — the three SEV-1/SEV-2 findings above are blocking and must land with the S12 transport work. With them in place, the two-transport architecture (RFC-011) is a net security improvement over the per-tool model it replaces.

---

## References

- RFC-011 — Two-Transport Language Plugin Architecture (companion)
- ADR-006 — rust-analyzer Subprocess Trust Model (generalised here)
- RFC-008 — Multi-Language Extension Architecture (per-tool ADR chain obviated)
- RFC-009 — Cross-Language Bridge Plugin System (bridge panic-isolation folded into Rule 4 / RFC-011 §10)
- ADR-005 — Per-Language Corpus Naming (trust keyed per canonical corpus)
- ARCH-102 — Kythe Corpus Naming Convention (canonical-corpus identity for trust grants)
- CLAUDE.md — Non-Negotiable Principles (#2 always-fresh, #3 local-first, #7 no-unsafe)
