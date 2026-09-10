//! Windows AppContainer + Job Object sandbox (ADR-017 Rules 1-2).
//! Safe wrapper — no `unsafe` code in this file; all unsafe is in `ffi.rs`.

mod ffi;

/// #500: process-liveness probe for the embed sidecar's shutdown grace poll.
/// ADR-017 A2 Invariant 1: the sizing probes live here too, so every unsafe
/// block in this crate stays confined to ffi.rs.
pub(crate) use ffi::{available_physical_memory_mb, pid_alive, windows_p_core_count};

/// #572: spawn a long-lived detached child (the daemon) with handle
/// inheritance restricted to an explicit allowlist of its own NUL stdio, so
/// no inheritable handle in this process — a shell pipe on our stdout, or a
/// pipe a grandparent passed down — can leak into it and hold a pipeline
/// open. Public because the CLI's daemonizing re-exec (`travsr-cli`,
/// `forbid(unsafe_code)`) is the caller; the unsafe stays confined to ffi.rs
/// per ADR-017 A2.
pub use ffi::spawn_detached_with_inherit_allowlist;

use crate::sandbox::policy::{SandboxPolicy, SandboxUnavailable};
use crate::sandbox::StdioCfg;
use std::io;
use std::path::PathBuf;

fn profile_name(repo_root: &std::path::Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    repo_root.hash(&mut h);
    format!("travsr-{:016x}", h.finish())
}

fn to_mode(cfg: StdioCfg) -> ffi::StdioMode {
    match cfg {
        StdioCfg::Pipe => ffi::StdioMode::Pipe,
        StdioCfg::Null => ffi::StdioMode::Null,
        StdioCfg::Inherit => ffi::StdioMode::Inherit,
    }
}

// ── AppContainerChild ─────────────────────────────────────────────────────────

/// Live AppContainer child process. All handles owned here; `_job` keeps the
/// Job Object alive so `KILL_ON_JOB_CLOSE` fires on drop.
pub struct AppContainerChild {
    process: ffi::OwnedHandle,
    _job: ffi::OwnedJobHandle,
    pid: u32,
    stdin_write: Option<ffi::OwnedHandle>,
    stdout_read: Option<ffi::OwnedHandle>,
    stderr_read: Option<ffi::OwnedHandle>,
}

impl AppContainerChild {
    pub fn id(&self) -> u32 {
        self.pid
    }

    pub fn kill(&mut self) -> io::Result<()> {
        ffi::terminate_process(self.process.as_handle())
    }

    pub fn wait(&mut self) -> io::Result<std::process::ExitStatus> {
        let code = ffi::wait_for_process(self.process.as_handle())?;
        // ExitStatusExt::from_raw is stable since Rust 1.72; workspace requires 1.75+.
        use std::os::windows::process::ExitStatusExt;
        Ok(std::process::ExitStatus::from_raw(code))
    }

    pub fn wait_with_output(mut self) -> io::Result<std::process::Output> {
        // Close parent's stdin write end so the child sees EOF.
        drop(self.stdin_write.take());

        // Read stdout and stderr concurrently to prevent deadlock on full pipe buffers.
        let stdout_handle = self.stdout_read.take();
        let stderr_handle = self.stderr_read.take();

        let stdout_thread = stdout_handle.map(|h| {
            std::thread::spawn(move || -> io::Result<Vec<u8>> {
                use std::io::Read;
                let mut f = ffi::handle_into_read_file(h);
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                Ok(buf)
            })
        });

        let stderr_thread = stderr_handle.map(|h| {
            std::thread::spawn(move || -> io::Result<Vec<u8>> {
                use std::io::Read;
                let mut f = ffi::handle_into_read_file(h);
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                Ok(buf)
            })
        });

        let status = self.wait()?;

        let stdout = stdout_thread
            .map(|t| t.join().unwrap_or_else(|_| Ok(Vec::new())))
            .transpose()?
            .unwrap_or_default();

        let stderr = stderr_thread
            .map(|t| t.join().unwrap_or_else(|_| Ok(Vec::new())))
            .transpose()?
            .unwrap_or_default();

        Ok(std::process::Output {
            status,
            stdout,
            stderr,
        })
    }

    /// Extract IPC streams (stdin write, stdout read) for protocol communication.
    /// Returns `None` if the child was not spawned with `StdioCfg::Pipe` on both.
    pub fn take_ipc_streams(
        &mut self,
    ) -> Option<(Box<dyn io::Write + Send>, Box<dyn io::Read + Send>)> {
        let stdin_h = self.stdin_write.take()?;
        let stdout_h = self.stdout_read.take()?;
        let stdin_file = ffi::handle_into_write_file(stdin_h);
        let stdout_file = ffi::handle_into_read_file(stdout_h);
        Some((Box::new(stdin_file), Box::new(stdout_file)))
    }
}

// ── AppContainerSpawn ─────────────────────────────────────────────────────────

/// AppContainer spawn builder. Created by `build_sandboxed_command`; configure
/// stdio via `set_stdin/stdout/stderr`, then launch with `spawn()`.
pub struct AppContainerSpawn {
    program: String,
    args: Vec<String>,
    repo_root: PathBuf,
    scratch_dir: PathBuf,
    policy: SandboxPolicy,
    /// Per-language toolchain cache grants (read/write paths) and env.
    /// #501: the child env is an explicit allowlist block (`build_env_block`),
    /// NOT inherited — `toolchain.env` (JAVA_HOME/GOPATH/…) is forwarded into
    /// it and `~/.travsr/bin` is prepended to PATH, mirroring linux.rs/macos.rs.
    toolchain: crate::sandbox::toolchain::ToolchainAccess,
    /// The exact repo-relative subpaths this language's analyzer must write
    /// (php's `index.scip`, scala's `target/`). The repo root itself always
    /// stays read-only per ADR-017 Rule 1; each of these gets its own grant.
    /// Empty for every other language.
    repo_write: &'static [crate::sandbox::toolchain::RepoWrite],
    /// The language being analysed, for the grant-skipped diagnostics below.
    language: String,
    stdin: StdioCfg,
    stdout: StdioCfg,
    stderr: StdioCfg,
}

impl AppContainerSpawn {
    pub(super) fn set_stdin(&mut self, cfg: StdioCfg) {
        self.stdin = cfg;
    }
    pub(super) fn set_stdout(&mut self, cfg: StdioCfg) {
        self.stdout = cfg;
    }
    pub(super) fn set_stderr(&mut self, cfg: StdioCfg) {
        self.stderr = cfg;
    }

    /// Materialise and grant this language's repo-write subpaths, leaving the
    /// repo root itself read-only.
    ///
    /// The host creates each path first because an ACL can only be set on an
    /// object that already exists. Pre-creating is also what keeps the root
    /// read-only: the analyzer then only needs to open an existing file, not
    /// FILE_ADD_FILE on the directory holding it.
    ///
    /// Best-effort per entry, matching linux.rs: a grant that cannot be
    /// established costs that language its build output, never the whole spawn.
    fn grant_repo_write_subpaths(&self, sid: &ffi::AppContainerSid) {
        for entry in self.repo_write {
            // A grant path must be a real path inside the repo, never a link out
            // of it. This code runs UNSANDBOXED as the user, and both the create
            // below and the ACL write follow reparse points, so a repo shipping
            // `index.scip` as a link to a file outside the repo would get that
            // target created and then handed to the sandbox writable.
            if crate::sandbox::toolchain::grant_path_has_symlink(&self.repo_root, entry.subpath()) {
                tracing::warn!(
                    language = %self.language,
                    subpath = %entry.subpath(),
                    "repo-write grant skipped: a component of the path is a symlink, \
                     which would grant its target writable to the sandbox (ADR-017 Rule 1)"
                );
                continue;
            }
            let host = self.repo_root.join(entry.subpath());
            let is_dir = match entry {
                crate::sandbox::toolchain::RepoWrite::Dir(_) => {
                    let _ = std::fs::create_dir_all(&host);
                    true
                }
                crate::sandbox::toolchain::RepoWrite::File(_) => {
                    if let Some(parent) = host.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&host);
                    false
                }
            };
            // Re-stat what the create produced: fail closed if it is a symlink
            // after all (the check above raced) or if the create failed and left
            // nothing safe to grant.
            let is_real = std::fs::symlink_metadata(&host)
                .map(|m| !m.file_type().is_symlink())
                .unwrap_or(false);
            if !is_real {
                tracing::warn!(
                    language = %self.language,
                    subpath = %entry.subpath(),
                    "repo-write grant skipped: the path is not a real file or directory after \
                     creating it, so there is nothing safe to grant"
                );
                continue;
            }
            // A directory grant must inherit so files the analyzer creates under
            // it are writable too; a file grant must not, or it would leak to
            // every sibling under the directory holding it.
            let granted = if is_dir {
                ffi::grant_path_access(&host, sid.as_psid(), ffi::ACCESS_GENERIC_ALL)
            } else {
                ffi::grant_path_access_this_only(&host, sid.as_psid(), ffi::ACCESS_GENERIC_ALL)
            };
            if let Err(e) = granted {
                tracing::warn!(
                    language = %self.language,
                    subpath = %entry.subpath(),
                    err = %e,
                    "repo-write grant failed, the analyzer may report an empty index"
                );
            }
        }
    }

    /// Spawn the plugin binary inside an AppContainer with a Job Object.
    /// Fail-closed (ADR-017 Rule 2): any setup failure returns `Err`.
    pub(super) fn spawn(self) -> io::Result<AppContainerChild> {
        let profile = profile_name(&self.repo_root);
        let elevated = matches!(self.policy, SandboxPolicy::Elevated { .. });

        // ── 1–4. AppContainer SID, profile, DACL grants, SECURITY_CAPABILITIES ─
        let sid = ffi::derive_appcontainer_sid(&profile)?;
        ffi::ensure_appcontainer_profile(&profile)?;
        // ADR-017 Rule 1: the repo root is read-only, on every platform. A
        // language that must write into its own project tree gets a grant on
        // exactly those subpaths, layered over the read-only root — never
        // GENERIC_ALL on the root itself, which would let a hostile
        // `composer.json` or `build.sbt` rewrite any file in the repo during
        // indexing. This mirrors the per-subpath `--bind` linux.rs builds and
        // the per-subpath Seatbelt rules macos.rs emits.
        ffi::grant_path_access(&self.repo_root, sid.as_psid(), ffi::ACCESS_GENERIC_READ)?;
        self.grant_repo_write_subpaths(&sid);
        // A direct CreateFile open of repo_root works from the grant above
        // alone, but resolving its REAL/canonical path (GetFinalPathNameByHandleW —
        // what std::fs::canonicalize and Java NIO's Path.toRealPath() both call)
        // additionally needs FILE_TRAVERSE on every ancestor up to the volume
        // root, which an AppContainer token doesn't get for free. See
        // ffi::grant_ancestor_traverse's doc comment (found via sbt 2.x).
        ffi::grant_ancestor_traverse(&self.repo_root, sid.as_psid());
        ffi::grant_path_access(&self.scratch_dir, sid.as_psid(), ffi::ACCESS_GENERIC_ALL)?;
        // Per-language toolchain caches (best-effort: a missing cache dir is not fatal).
        for path in &self.toolchain.read_paths {
            let _ = ffi::grant_path_access(path, sid.as_psid(), ffi::ACCESS_GENERIC_READ);
        }
        for path in &self.toolchain.write_paths {
            let _ = ffi::grant_path_access(path, sid.as_psid(), ffi::ACCESS_GENERIC_ALL);
        }
        // G5: toolchain *bin* dirs (GOPATH/bin, GOROOT/bin, JAVA_HOME/bin,
        // sbt/dotnet bin) need read+EXECUTE, not just read — otherwise the
        // AppContainer token finds the toolchain by name (PATH is forwarded) but
        // cannot map its image, and the analyzer spawns nothing and reports zero
        // symbols with no error. These are a strict subset of executable
        // directories (never module/package caches), so the extra right is scoped
        // to exactly the binaries the analyzer legitimately runs. Best-effort like
        // the other toolchain grants: a bin dir under a machine-wide tree (Program
        // Files) may not be re-ACL-able without admin, in which case that binary
        // stays unrunnable and only the user-writable dirs (e.g. GOPATH/bin) gain
        // execute — which is still enough for the analyzer itself to run.
        for path in &self.toolchain.exec_paths {
            let _ = ffi::grant_path_access(path, sid.as_psid(), ffi::ACCESS_GENERIC_READ_EXECUTE);
        }
        // PR #577 review: an AppContainer token cannot map an image whose DACL
        // carries no AppContainer ACE — user-profile trees don't carry ALL
        // APPLICATION PACKAGES, and the owner-only hardening of ~/.travsr
        // (#507, travsr-store restrict_to_owner_windows) strips inherited ACEs
        // for anything created under it afterwards. The resolver hands us
        // plugin binaries from exactly ~/.travsr/bin, so grant read+execute on
        // that dir and on the program's own directory; without this the
        // sandbox spawn fails ERROR_ACCESS_DENIED at image load and the PATH
        // prepend (#501) points at files the child could not execute anyway.
        // Best-effort like the toolchain grants: the dir may not exist yet,
        // and a program under a machine-wide tree (Program Files) already
        // carries the needed ACEs. Idempotent per #505 — a repeat is one read.
        if let Some(home) = dirs::home_dir() {
            let travsr_bin = home.join(".travsr").join("bin");
            if travsr_bin.is_dir() {
                let _ = ffi::grant_path_access(
                    &travsr_bin,
                    sid.as_psid(),
                    ffi::ACCESS_GENERIC_READ_EXECUTE,
                );
            }
        }
        if let Some(program_dir) = std::path::Path::new(&self.program).parent() {
            if program_dir.is_dir() {
                let _ = ffi::grant_path_access(
                    program_dir,
                    sid.as_psid(),
                    ffi::ACCESS_GENERIC_READ_EXECUTE,
                );
            }
        }

        // PSE R5 (#499): capability storage is heap-pinned inside the owner;
        // the binding only needs to stay alive until CreateProcessW returns.
        let security_caps = ffi::build_security_capabilities(sid.as_psid(), elevated)?;

        if let SandboxPolicy::Elevated {
            permitted_hosts, ..
        } = &self.policy
        {
            tracing::warn!(
                permitted_hosts = ?permitted_hosts,
                "network-permitted policy active: Windows AppContainer allows the \
                 internet-client capability with no per-host filtering; enforce the \
                 permitted hosts with an egress proxy"
            );
        }

        // ── 5. Job Object ──────────────────────────────────────────────────────
        let job = ffi::create_job_with_limits()?;

        // ── 6. CreateProcessW inside AppContainer (P5-S3) ─────────────────────
        let handles = ffi::spawn_in_appcontainer(
            &self.program,
            &self.args,
            &self.scratch_dir,
            &self.toolchain.env,  // #501: forwarded into the child env block
            security_caps.caps(), // PSE R5: owner `security_caps` still live here
            job,
            to_mode(self.stdin),
            to_mode(self.stdout),
            to_mode(self.stderr),
        )?;

        Ok(AppContainerChild {
            process: handles.process,
            _job: handles._job,
            pid: handles.pid,
            stdin_write: handles.stdin_write,
            stdout_read: handles.stdout_read,
            stderr_read: handles.stderr_read,
        })
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Returns a `SandboxedSpawn::AppContainer` for the given arguments.
/// Fail-closed: validates Elevated policy before returning.
pub fn build_sandboxed_command(
    program: &str,
    args: &[&str],
    repo_root: &std::path::Path,
    scratch_dir: &std::path::Path,
    policy: &SandboxPolicy,
    language: &str,
) -> Result<super::SandboxedSpawn, SandboxUnavailable> {
    if let SandboxPolicy::Elevated { .. } = policy {
        policy.validate()?;
    }
    Ok(super::SandboxedSpawn::AppContainer(AppContainerSpawn {
        program: program.to_string(),
        args: args.iter().map(|s| s.to_string()).collect(),
        repo_root: repo_root.to_path_buf(),
        scratch_dir: scratch_dir.to_path_buf(),
        policy: policy.clone(),
        toolchain: crate::sandbox::toolchain::toolchain_access(language),
        repo_write: crate::sandbox::toolchain::repo_write_subpaths(language),
        language: language.to_string(),
        stdin: StdioCfg::Inherit,
        stdout: StdioCfg::Inherit,
        stderr: StdioCfg::Inherit,
    }))
}
