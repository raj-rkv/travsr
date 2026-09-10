//! Linux sandbox (bubblewrap + optional Landlock). Fail-closed per ADR-017 Rule 2.
use crate::sandbox::policy::{SandboxPolicy, SandboxUnavailable};
use crate::sandbox::SandboxedSpawn;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::process::Command;

/// Permitted env vars (ADR-017 Rule 1). TMPDIR set by caller to scratch dir.
pub const ENV_ALLOWLIST: &[&str] = &["PATH", "LANG", "LC_ALL"];

/// Where the host's scratch directory is mounted INSIDE the sandbox.
///
/// bwrap remaps it rather than binding it at its own host path, so this is the
/// only name a sandboxed sidecar can reach it by: `/tmp` is not bound and the
/// namespace root is a fresh tmpfs, so the host path resolves to nothing. Both
/// the `--bind` below and the `TMPDIR` handed to the child use this constant,
/// and so does [`crate::sandbox::sidecar_scratch_path`], which is what tells a
/// sidecar the name it can actually open.
pub const SCRATCH_MOUNT: &str = "/travsr-scratch";

#[cfg(target_os = "linux")]
pub fn build_sandboxed_command(
    program: &str,
    args: &[&str],
    repo_root: &Path,
    scratch_dir: &Path,
    policy: &SandboxPolicy,
    language: &str,
) -> Result<SandboxedSpawn, SandboxUnavailable> {
    if !bwrap_available() {
        return Err(SandboxUnavailable(
            "bubblewrap (bwrap) is not available or cannot create sandboxes on this host \
             (not on PATH, or kernel namespace support is restricted); \
             install bubblewrap and verify unprivileged user namespaces are enabled"
                .into(),
        ));
    }

    // For Elevated policy, validate fields first (fail-closed per ADR-017 Rule 2).
    if let SandboxPolicy::Elevated { .. } = policy {
        policy.validate()?;
    }

    let repo = repo_root.to_string_lossy();
    let scratch = scratch_dir.to_string_lossy();

    // Per-language toolchain grants (e.g. go module/build caches + GO*/HOME env).
    // Empty for languages with no out-of-repo needs.
    let tc = crate::sandbox::toolchain::toolchain_access(language);

    let mut cmd = Command::new("bwrap");

    match policy {
        // Standard / NativeIpc: network access allowed — build tools (go, npm,
        // pip, …) may need to fetch missing dependencies. File-system confinement
        // via bwrap still applies; no --unshare-net.
        SandboxPolicy::Standard | SandboxPolicy::NativeIpc => {}
        SandboxPolicy::Elevated {
            permitted_hosts, ..
        } => {
            // Elevated: FS confinement via bwrap still applies, but --unshare-net
            // is intentionally skipped so the plugin can reach its permitted hosts.
            // Host-level filtering (firewall / egress proxy) must enforce the
            // permitted_hosts list — bwrap has no per-host network rule support.
            tracing::info!(
                permitted_hosts = ?permitted_hosts,
                "network-permitted policy active: this plugin's network isolation is \
                 intentionally off so it can reach its permitted hosts; enforce the host \
                 allowlist with egress controls (a firewall or proxy), since bubblewrap has \
                 no per-host network rules"
            );
        }
    }
    // NativeIpc policy: tool uses POSIX IPC queues/shm (e.g. scip-clang parallel
    // workers). Skip --unshare-ipc so mq_open/shm_open work inside the namespace.
    let ipc_unshare: &[&str] = if matches!(policy, SandboxPolicy::NativeIpc) {
        &["--unshare-pid", "--unshare-uts"]
    } else {
        &["--unshare-pid", "--unshare-uts", "--unshare-ipc"]
    };
    cmd.args(ipc_unshare);
    for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64"] {
        cmd.args(["--ro-bind-try", path, path]);
    }
    for path in [
        "/etc/alternatives",
        "/etc/localtime",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
    ] {
        cmd.args(["--ro-bind-try", path, path]);
    }
    cmd.args(["--proc", "/proc", "--dev", "/dev"]);
    // Provide a writable scratch area inside the sandbox at /travsr-scratch by
    // bind-mounting the host scratch dir there. The host dir is owned by the
    // real UID (created via tempfile by the daemon), so the sandboxed process —
    // which runs as that same UID since we don't remap with --unshare-user — can
    // write to it. bwrap creates the /travsr-scratch mount point in its root
    // tmpfs automatically. A plain --tmpfs would be root-owned and unwritable.
    cmd.args(["--bind", scratch.as_ref(), SCRATCH_MOUNT]); // writable scratch
                                                           // ADR-017 Rule 1: the repo root is read-only. A language that must write
                                                           // build outputs into its own project tree (scala: sbt has no out-of-tree
                                                           // build) gets writable binds for exactly those subpaths, layered over the
                                                           // read-only root — never the whole repo. bwrap needs the mount source to
                                                           // exist, so the host subpath is created first (build-artifact dirs / the
                                                           // generated settings file — the same paths the analyzer writes anyway).
    cmd.args(["--ro-bind", repo.as_ref(), repo.as_ref()]); // repo: ro
    for entry in crate::sandbox::toolchain::repo_write_subpaths(language) {
        let host = std::path::Path::new(repo.as_ref()).join(entry.subpath());
        // A grant path must be a real path inside the repo, never a link out of
        // it. `create_dir_all`, `OpenOptions::open` and bwrap's own `--bind`
        // source resolution all FOLLOW symlinks, and this code runs UNSANDBOXED
        // as the user: a repo shipping php's `index.scip` as a link to
        // `~/.ssh/authorized_keys` would get that target created and then
        // bind-mounted WRITABLE into the sandbox. Skipping the grant costs that
        // language its build output; binding it costs the user's home directory.
        if crate::sandbox::toolchain::grant_path_has_symlink(
            Path::new(repo.as_ref()),
            entry.subpath(),
        ) {
            tracing::warn!(
                language = %language,
                subpath = %entry.subpath(),
                "repo-write grant skipped: a component of the path is a symlink, \
                 which would bind its target writable into the sandbox (ADR-017 Rule 1)"
            );
            continue;
        }
        match entry {
            crate::sandbox::toolchain::RepoWrite::Dir(_) => {
                let _ = std::fs::create_dir_all(&host);
            }
            crate::sandbox::toolchain::RepoWrite::File(_) => {
                if let Some(parent) = host.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&host);
            }
        }
        // Re-stat the leaf the create just produced: fail closed if it is a
        // symlink after all (the check above raced) or if the create failed and
        // left no bind source at all.
        let is_real = std::fs::symlink_metadata(&host)
            .map(|m| !m.file_type().is_symlink())
            .unwrap_or(false);
        if !is_real {
            tracing::warn!(
                language = %language,
                subpath = %entry.subpath(),
                "repo-write grant skipped: the path is not a real file or directory after \
                 creating it, so there is nothing safe to bind"
            );
            continue;
        }
        let host = host.to_string_lossy();
        cmd.args(["--bind", host.as_ref(), host.as_ref()]); // writable subpath
    }
    // Per-language toolchain caches: read-only module/toolchain dirs, writable
    // build cache. Bound at their host paths so the GO*/HOME env (set below) resolve.
    for path in &tc.read_paths {
        let p = path.to_string_lossy();
        cmd.args(["--ro-bind-try", p.as_ref(), p.as_ref()]);
    }
    for path in &tc.write_paths {
        let p = path.to_string_lossy();
        cmd.args(["--bind-try", p.as_ref(), p.as_ref()]);
    }
    // Bind ~/.travsr/bin so tools installed by `travsr lang install` (e.g. scip-java,
    // scip-go) are accessible inside the bwrap namespace.
    if let Ok(home) = std::env::var("HOME") {
        let travsr_bin = format!("{home}/.travsr/bin");
        cmd.args(["--ro-bind-try", &travsr_bin, &travsr_bin]);
    }
    cmd.args(["--die-with-parent", "--"]);
    // Resource caps (ADR-017 Rule 1): 4 GiB virtual memory + 300s CPU via ulimit.
    let quoted_args = args
        .iter()
        .map(|a| format!("'{}'", a.replace('\'', r"'\''")))
        .collect::<Vec<_>>()
        .join(" ");
    let inner = format!(
        "ulimit -v 4194304 2>/dev/null; ulimit -t 300 2>/dev/null; exec '{}' {}",
        program.replace('\'', r"'\''"),
        quoted_args
    );
    cmd.args(["sh", "-c", &inner]);
    cmd.env_clear();
    for key in ENV_ALLOWLIST {
        if let Ok(val) = std::env::var(key) {
            cmd.env(key, val);
        }
    }
    cmd.env("TMPDIR", SCRATCH_MOUNT);
    // Per-language toolchain env (e.g. GOPATH/GOCACHE/GOMODCACHE/HOME) so the
    // analyzer's build tool locates its caches inside the cleared sandbox env.
    for (key, val) in &tc.env {
        cmd.env(key, val);
    }
    // Prepend ~/.travsr/bin so tools installed by `travsr lang install` (e.g. scip-java,
    // scip-go) are on PATH inside the sandbox.
    if let Ok(home) = std::env::var("HOME") {
        let travsr_bin = format!("{home}/.travsr/bin");
        let base = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{travsr_bin}:{base}"));
    }
    Ok(SandboxedSpawn::Wrapped(cmd))
}

/// Returns true if `bwrap` is on PATH (installed).
/// Use this to distinguish "not installed" from "installed but cannot namespace" —
/// the CI panic in sandbox tests should only fire for the former.
#[cfg(target_os = "linux")]
pub fn bwrap_is_on_path() -> bool {
    Command::new("bwrap")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// Cached probe: can bwrap actually create a sandbox on this host?
// On some CI runners (Docker-in-Docker, AppArmor-restricted kernels) bwrap is
// installed but cannot create user namespaces. We detect this once and treat such
// hosts as sandbox-unavailable so callers receive Err(SandboxUnavailable) and skip
// gracefully instead of spawning a bwrap that exits non-zero.
#[cfg(target_os = "linux")]
static BWRAP_FUNCTIONAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(target_os = "linux")]
fn bwrap_available() -> bool {
    *BWRAP_FUNCTIONAL.get_or_init(|| {
        // Mount /lib and /lib64 alongside /usr so the dynamic linker
        // (/lib64/ld-linux-x86-64.so.2) is reachable inside the probe sandbox.
        // On merged-usr systems (Ubuntu 22.04+) /lib and /lib64 are host
        // symlinks → /usr/lib{,64}; they don't appear automatically inside
        // bwrap's fresh tmpfs root, causing execvp("true") to fail with ENOENT.
        Command::new("bwrap")
            .args([
                "--ro-bind-try",
                "/usr",
                "/usr",
                "--ro-bind-try",
                "/lib",
                "/lib",
                "--ro-bind-try",
                "/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--",
                "true",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

#[cfg(not(target_os = "linux"))]
pub fn build_sandboxed_command(
    _p: &str,
    _a: &[&str],
    _r: &Path,
    _s: &Path,
    _policy: &SandboxPolicy,
    _lang: &str,
) -> Result<SandboxedSpawn, SandboxUnavailable> {
    Err(SandboxUnavailable(
        "Linux sandbox not available on this platform".into(),
    ))
}
