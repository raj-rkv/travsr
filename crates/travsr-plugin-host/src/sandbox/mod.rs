pub mod linux;
pub mod macos;
pub mod policy;
pub mod toolchain;
#[cfg(target_os = "windows")]
pub mod windows;

pub use policy::{SandboxPolicy, SandboxUnavailable};

/// The scratch directory as the SIDECAR will see it, given the host path the
/// daemon created.
///
/// The two differ on Linux, and only on Linux. bwrap binds the host directory
/// at the fixed [`linux::SCRATCH_MOUNT`] rather than at its own path, and
/// nothing maps the host path into the namespace: `/tmp` is not bound and the
/// root is a fresh tmpfs. A sidecar handed the host path therefore finds
/// nothing there and every write through it fails, silently, as a zero-node
/// index. macOS grants the canonical host path in its Seatbelt profile and
/// Windows ACLs that same path, so on both the host path is the real one and is
/// returned unchanged.
///
/// The Linux remap is unconditional rather than conditional on having
/// sandboxed: `Sidecar::build_cmd` routes every Linux spawn through
/// `linux::build_sandboxed_command`, which fails closed when bwrap is missing,
/// and the unsandboxed path is Windows-only. So a Linux sidecar that is running
/// at all is inside the namespace where this name resolves.
///
/// This reports the truth about an existing bind; it grants nothing new. The
/// mount and the child's `TMPDIR` already pointed here.
pub fn sidecar_scratch_path(host: &std::path::Path) -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    {
        let _ = host;
        std::path::PathBuf::from(linux::SCRATCH_MOUNT)
    }
    #[cfg(not(target_os = "linux"))]
    {
        host.to_path_buf()
    }
}

/// How a single stdio stream is configured for a sandboxed process.
#[derive(Clone, Copy)]
pub enum StdioCfg {
    /// Inherit the parent process's handle.
    Inherit,
    /// Redirect to the null device (discard).
    Null,
    /// Create an anonymous pipe; the parent end is accessible on the child.
    Pipe,
}

impl StdioCfg {
    fn into_stdio(self) -> std::process::Stdio {
        match self {
            StdioCfg::Inherit => std::process::Stdio::inherit(),
            StdioCfg::Null => std::process::Stdio::null(),
            StdioCfg::Pipe => std::process::Stdio::piped(),
        }
    }
}

/// Completed sandboxed child process.
pub enum SandboxedChild {
    Standard(std::process::Child),
    #[cfg(target_os = "windows")]
    AppContainer(windows::AppContainerChild),
}

impl SandboxedChild {
    pub fn id(&self) -> u32 {
        match self {
            SandboxedChild::Standard(c) => c.id(),
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(c) => c.id(),
        }
    }

    pub fn kill(&mut self) -> std::io::Result<()> {
        match self {
            SandboxedChild::Standard(c) => c.kill(),
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(c) => c.kill(),
        }
    }

    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        match self {
            SandboxedChild::Standard(c) => c.wait(),
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(c) => c.wait(),
        }
    }

    pub fn wait_with_output(self) -> std::io::Result<std::process::Output> {
        match self {
            SandboxedChild::Standard(c) => c.wait_with_output(),
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(c) => c.wait_with_output(),
        }
    }

    /// Extract IPC streams (stdin write, stdout read) for protocol use.
    /// Returns `None` if the child was not spawned with `StdioCfg::Pipe` on both.
    pub fn take_ipc_streams(
        &mut self,
    ) -> Option<(
        Box<dyn std::io::Write + Send>,
        Box<dyn std::io::Read + Send>,
    )> {
        match self {
            SandboxedChild::Standard(child) => {
                let stdin = child.stdin.take()?;
                let stdout = child.stdout.take()?;
                Some((
                    Box::new(stdin) as Box<dyn std::io::Write + Send>,
                    Box::new(stdout) as Box<dyn std::io::Read + Send>,
                ))
            }
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(ac) => ac.take_ipc_streams(),
        }
    }

    /// Take the child's piped stderr, if it was spawned with `StdioCfg::Pipe`.
    /// Used to drain a Phase B sidecar's own diagnostics into a bounded ring so
    /// a libclang/parse failure that yields a silent empty index is surfaced via
    /// tracing instead of being swallowed. `None` for the Windows AppContainer
    /// path (stderr handled by the container) and for stub children.
    pub fn take_stderr(&mut self) -> Option<std::process::ChildStderr> {
        match self {
            SandboxedChild::Standard(child) => child.stderr.take(),
            #[cfg(target_os = "windows")]
            SandboxedChild::AppContainer(_) => None,
        }
    }
}

/// Build a plain (unsandboxed) child command for a Phase B analyzer.
///
/// Used ONLY on Windows, and ONLY for analyzers whose build tools cannot run
/// inside the isolation layer (`WindowsSandbox::Unsupported`), and ONLY after the
/// user has granted explicit permission (see `resolver`). The child runs with the
/// user's own privileges — the same trade-off the project already accepts for the
/// rust `--allow-unsandboxed` LSIF path.
///
/// The toolchain environment (JAVA_HOME, GRADLE_USER_HOME, HOME, …) is forwarded
/// and `~/.travsr/bin` is prepended to PATH, mirroring what the isolated path sets
/// up, so the analyzer and the build tool it drives resolve. `scratch` is the cwd
/// (matching the isolated spawn); the repo root reaches the sidecar via the
/// InvokeRequest, not the cwd.
pub fn build_unsandboxed_command(
    program: &str,
    args: &[&str],
    scratch: &std::path::Path,
    language: &str,
) -> SandboxedSpawn {
    let mut cmd = std::process::Command::new(program);
    cmd.args(args);
    cmd.current_dir(scratch);

    // Start from an empty environment and forward only an allowlist. The daemon's
    // own environment can hold credentials (GITHUB_TOKEN, AWS_*, SSH_*, …); an
    // unsandboxed analyzer drives the repo's own build (Gradle/sbt), so no daemon
    // secret should reach it. The toolchain env, HOME, GRADLE_USER_HOME and PATH
    // are re-applied explicitly below from values the toolchain helpers already
    // captured, so clearing here does not break analyzer/build-tool resolution.
    cmd.env_clear();
    for (k, v) in std::env::vars_os() {
        if k.to_str().is_some_and(is_allowed_passthrough_env) {
            cmd.env(&k, &v);
        }
    }

    let access = toolchain::toolchain_access(language);
    for (k, v) in &access.env {
        cmd.env(k, v);
    }

    // PATH is (re)applied only inside this block. If `home_dir()` returns None —
    // an environment with neither HOME nor USERPROFILE — the child inherits no
    // PATH at all (PATH is deliberately not in the passthrough allowlist). That
    // fails closed: the build tool simply will not resolve and Phase B produces
    // nothing, rather than the child running with the daemon's unscrubbed PATH.
    if let Some(home) = dirs::home_dir() {
        // The toolchain env helpers key their HOME/cache paths off the `HOME`
        // variable, which is unset on Windows (it uses `USERPROFILE`), so JVM build
        // tools like Gradle end up with nowhere to place their home/temp dir. An
        // unsandboxed child runs as the user, so give it the real home explicitly —
        // set `HOME` and Gradle's home unless the toolchain env already provided
        // them. Harmless off Windows, where `HOME` is already correct.
        let home_str = home.to_string_lossy().into_owned();
        if !access.env.iter().any(|(k, _)| k == "HOME") {
            cmd.env("HOME", &home_str);
        }
        if !access.env.iter().any(|(k, _)| k == "GRADLE_USER_HOME") {
            cmd.env("GRADLE_USER_HOME", home.join(".gradle"));
        }

        // Prepend ~/.travsr/bin so the wrapper's installed siblings (and analyzers
        // the wrapper shells out to) resolve, matching the isolated path's PATH
        // handling.
        let travsr_bin = home.join(".travsr").join("bin");
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let mut dirs_vec = vec![travsr_bin];
        dirs_vec.extend(std::env::split_paths(&existing));
        if let Ok(joined) = std::env::join_paths(dirs_vec) {
            cmd.env("PATH", joined);
        }
    }

    SandboxedSpawn::Wrapped(cmd)
}

/// System environment variables a build tool legitimately needs to start,
/// forwarded from the daemon's environment to the unsandboxed child. Everything
/// else — notably any credential the daemon holds (GITHUB_TOKEN, AWS_*, SSH_*, …)
/// — is dropped. HOME, GRADLE_USER_HOME and PATH are not listed here because
/// `build_unsandboxed_command` sets them explicitly. Matched case-insensitively
/// because Windows environment names are case-insensitive.
fn is_allowed_passthrough_env(name: &str) -> bool {
    const ALLOW: &[&str] = &[
        // Windows OS essentials for spawning a process and starting a JVM.
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "USERNAME",
        "COMPUTERNAME",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "ALLUSERSPROFILE",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "OS",
        // Locale and temp dir (deterministic tool output; Unix parity).
        "LANG",
        "LC_ALL",
        "TMPDIR",
    ];
    ALLOW.iter().any(|a| a.eq_ignore_ascii_case(name))
}

/// Result of `build_sandboxed_command`. Configure stdio, then call `spawn`.
pub enum SandboxedSpawn {
    /// Linux (bwrap) or macOS (sandbox-exec): wraps the outer wrapper process.
    Wrapped(std::process::Command),
    /// Windows: AppContainer + Job Object (ADR-017 Rules 1-2).
    #[cfg(target_os = "windows")]
    AppContainer(windows::AppContainerSpawn),
}

impl SandboxedSpawn {
    pub fn stdin(&mut self, cfg: StdioCfg) -> &mut Self {
        match self {
            SandboxedSpawn::Wrapped(cmd) => {
                cmd.stdin(cfg.into_stdio());
            }
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(ac) => ac.set_stdin(cfg),
        }
        self
    }

    pub fn stdout(&mut self, cfg: StdioCfg) -> &mut Self {
        match self {
            SandboxedSpawn::Wrapped(cmd) => {
                cmd.stdout(cfg.into_stdio());
            }
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(ac) => ac.set_stdout(cfg),
        }
        self
    }

    pub fn stderr(&mut self, cfg: StdioCfg) -> &mut Self {
        match self {
            SandboxedSpawn::Wrapped(cmd) => {
                cmd.stderr(cfg.into_stdio());
            }
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(ac) => ac.set_stderr(cfg),
        }
        self
    }

    /// Spawn the sandboxed process.
    pub fn spawn(self) -> std::io::Result<SandboxedChild> {
        match self {
            SandboxedSpawn::Wrapped(mut cmd) => Ok(SandboxedChild::Standard(cmd.spawn()?)),
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(ac) => Ok(SandboxedChild::AppContainer(ac.spawn()?)),
        }
    }

    /// Convenience: spawn with inherited stdio, wait, return exit status.
    pub fn status(self) -> std::io::Result<std::process::ExitStatus> {
        match self {
            SandboxedSpawn::Wrapped(mut cmd) => cmd.status(),
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(mut ac) => {
                ac.set_stdin(StdioCfg::Inherit);
                ac.set_stdout(StdioCfg::Inherit);
                ac.set_stderr(StdioCfg::Inherit);
                ac.spawn()?.wait()
            }
        }
    }

    /// Convenience: stdin=null, capture stdout+stderr, wait, return Output.
    pub fn output(self) -> std::io::Result<std::process::Output> {
        match self {
            SandboxedSpawn::Wrapped(mut cmd) => cmd.output(),
            #[cfg(target_os = "windows")]
            SandboxedSpawn::AppContainer(mut ac) => {
                ac.set_stdin(StdioCfg::Null);
                ac.set_stdout(StdioCfg::Pipe);
                ac.set_stderr(StdioCfg::Pipe);
                ac.spawn()?.wait_with_output()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value a sidecar is handed has to be a name IT can open. On Linux the
    /// host path is not one: bwrap binds the scratch directory at
    /// [`linux::SCRATCH_MOUNT`] and nothing maps the host path into the
    /// namespace, so sending it made every write through `InvokeRequest::scratch`
    /// fail and the language report zero nodes with no error. ruby, c and cpp
    /// all consumed the field directly and were all affected.
    #[test]
    fn sidecar_scratch_path_is_the_mount_on_linux_and_the_host_path_elsewhere() {
        let host = std::path::Path::new("/tmp/.tmpHostOnly123");
        let got = sidecar_scratch_path(host);
        if cfg!(target_os = "linux") {
            assert_eq!(
                got,
                std::path::Path::new(linux::SCRATCH_MOUNT),
                "a sandboxed Linux sidecar can only reach scratch by its mount point"
            );
            assert_ne!(
                got, host,
                "the host path resolves to nothing inside the namespace"
            );
        } else {
            assert_eq!(
                got, host,
                "macOS grants the canonical host path and Windows ACLs it, so it is the real one"
            );
        }
    }

    /// The mount point is shared by three places that must agree: the `--bind`
    /// target, the child's `TMPDIR`, and what the sidecar is told. Pinned as a
    /// literal so changing it stays a deliberate edit rather than something that
    /// silently desynchronises the three.
    #[test]
    fn scratch_mount_is_the_documented_path() {
        assert_eq!(linux::SCRATCH_MOUNT, "/travsr-scratch");
    }

    /// Process env is shared, so a test that sets/removes vars must not run
    /// concurrently with another that reads them. Hold this across every
    /// env-touching test in this binary (recovering from a poisoned guard, since
    /// the data guarded is process env, not test state).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The unsandboxed passthrough allowlist forwards OS essentials (matched
    /// case-insensitively) but drops credentials the daemon may hold.
    #[test]
    fn passthrough_env_allowlist_drops_secrets() {
        assert!(is_allowed_passthrough_env("SYSTEMROOT"));
        assert!(is_allowed_passthrough_env("SystemRoot"));
        assert!(is_allowed_passthrough_env("PATHEXT"));
        assert!(is_allowed_passthrough_env("TEMP"));

        assert!(!is_allowed_passthrough_env("GITHUB_TOKEN"));
        assert!(!is_allowed_passthrough_env("AWS_ACCESS_KEY_ID"));
        assert!(!is_allowed_passthrough_env("AWS_SECRET_ACCESS_KEY"));
        assert!(!is_allowed_passthrough_env("SSH_AUTH_SOCK"));
        assert!(!is_allowed_passthrough_env("NPM_TOKEN"));
    }

    /// `build_unsandboxed_command` must not leak a daemon secret into the child's
    /// environment, while still forwarding an allowlisted OS variable.
    #[test]
    fn unsandboxed_command_excludes_daemon_secrets() {
        let _env = env_guard();
        std::env::set_var("TRAVSR_TEST_FAKE_SECRET", "s3cr3t");
        std::env::set_var("SYSTEMROOT", "C:\\Windows");
        let scratch = std::env::temp_dir();
        let spawn = build_unsandboxed_command("java", &["-version"], &scratch, "java");
        // Off Windows, `SandboxedSpawn` has only the `Wrapped` variant
        // (AppContainer is windows-only), so this pattern is irrefutable and rustc
        // flags the `else` as unreachable. It is refutable on Windows, so allow the
        // lint only where the pattern cannot fail.
        #[cfg_attr(not(target_os = "windows"), allow(irrefutable_let_patterns))]
        let SandboxedSpawn::Wrapped(cmd) = spawn
        else {
            panic!("unsandboxed command must be a plain wrapped command");
        };
        // A daemon secret is dropped; an allowlisted OS var survives.
        let mut saw_secret = false;
        let mut saw_systemroot = false;
        for (k, _) in cmd.get_envs() {
            match k.to_str() {
                Some("TRAVSR_TEST_FAKE_SECRET") => saw_secret = true,
                Some(s) if s.eq_ignore_ascii_case("SYSTEMROOT") => saw_systemroot = true,
                _ => {}
            }
        }
        std::env::remove_var("TRAVSR_TEST_FAKE_SECRET");
        assert!(
            !saw_secret,
            "daemon secret must not reach the unsandboxed child"
        );
        assert!(saw_systemroot, "allowlisted OS var must be forwarded");
    }
}
