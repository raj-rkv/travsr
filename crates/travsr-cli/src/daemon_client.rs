//! Daemon-routed CLI queries (#318 O1).
//!
//! `ask` / `graph` / `status` first try the daemon's control socket — the
//! daemon holds the store open and warm, so a routed query skips the
//! per-command SQLite open that dominates CLI latency. Any failure (no daemon,
//! protocol version skew, transport error, malformed payload) falls back to
//! [`open_read_store`], which itself prefers the read-only fast path.

use std::path::Path;

use serde::de::DeserializeOwned;
use travsr_store::SqliteStore;

/// Outcome of an attempt to bring up the background daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpawnOutcome {
    /// The daemon is up and answering on its control socket.
    Started,
    /// The daemon is alive and holding the lock but has not finished its
    /// initial scan / socket bind yet (large repo). Not a failure.
    Starting,
    /// A daemon already holds the lock; nothing was spawned.
    AlreadyRunning,
    /// We spawned a child but it never came up (spawn error, or it died during
    /// startup — e.g. a control-socket bind failure, see [`crate::daemon_start_error`]).
    Failed,
}

/// True iff a **live** daemon currently holds this repo's exclusive lock.
///
/// This is the race-free singleton authority. It tries the very same
/// `.travsr/daemon.lock` flock the daemon holds for its lifetime, rather than
/// probing the control socket (unbound during the 10–30 s initial watcher scan)
/// or reading the lock-file PID (momentarily empty during the daemon's own
/// open→write window). Because the OS releases an flock when its holder dies,
/// "held" inherently means "a live process holds it" — no PID-liveness check
/// needed. If we can take the lock, no daemon holds it and we release immediately.
///
/// Opens read-only and NEVER creates the file: an absent lock means no daemon
/// can possibly hold it, so `false` is the answer without writing anything.
/// The previous `.create(true)` made this liveness probe create the very file
/// it was probing, so every non-interactive `travsr init` (CI, pipes,
/// scripts) left a stale zero-byte `.travsr/daemon.lock` behind, a file the
/// singleton protocol treats as meaningful, dropped there by something that
/// only wanted to read it (#636 round-2 review).
pub(crate) fn daemon_lock_held(repo_root: &Path) -> bool {
    use fs2::FileExt as _;
    let lock_path = repo_root.join(".travsr").join("daemon.lock");
    let Ok(file) = std::fs::OpenOptions::new().read(true).open(&lock_path) else {
        return false; // absent / .travsr missing / unopenable → no daemon possible
    };
    match file.try_lock_exclusive() {
        Ok(()) => {
            let _ = fs2::FileExt::unlock(&file);
            false
        }
        Err(_) => true,
    }
}

/// Spawn the background daemon **only if** none currently holds the repo lock,
/// then confirm the outcome. Race-free front door used by `daemon start`,
/// `daemon restart`, and `travsr init` so no entry point ever spawns a doomed
/// child against a running daemon.
pub(crate) fn spawn_background_daemon(repo_root: &Path, exe: &Path, verbose: bool) -> SpawnOutcome {
    if daemon_lock_held(repo_root) {
        return SpawnOutcome::AlreadyRunning;
    }
    // #688: `--verbose` reaches the child as RUST_LOG, and an explicit RUST_LOG
    // from the caller wins because it is more specific than a boolean can be.
    // Set on this process rather than on the command: the spawn below is split
    // per platform and the Windows path goes through a helper that takes no
    // environment, while a child inherits ours on both. This process exits as
    // soon as the daemon is up, so nothing else observes the change.
    if verbose && std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "debug");
    }
    // #592: clear any breadcrumb from a previous failed start so we only observe
    // THIS start's outcome.
    let err_path = repo_root.join(".travsr").join("daemon-start.err");
    let _ = std::fs::remove_file(&err_path);

    // Re-exec ourselves as the long-lived foreground worker (which re-acquires the
    // lock — the last-line-of-defense guard for the tight spawn race).
    #[cfg(unix)]
    let spawned: std::io::Result<()> = std::process::Command::new(exe)
        .args(["daemon", "start", "--foreground"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ());
    // #503/#572: on Windows the daemon must be fully detached from the
    // launching console (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, or
    // Ctrl+C / closing the terminal kills it and the Task Scheduler autostart
    // pops a console window) AND spawned with an explicit handle-inheritance
    // allowlist. `std::process::Command` forces bInheritHandles=TRUE whenever
    // stdio is configured, so EVERY inheritable handle in this process leaked
    // into the long-lived daemon even though its own stdio is null — the
    // write end of a pipe the shell attached to our stdout, or one a
    // grandparent (cargo → test harness → travsr) created inheritably and
    // passed down. Either pins the pipe open so its reader never sees EOF
    // until the daemon exits. The raw spawn below lists exactly the daemon's
    // three NUL stdio handles as inheritable and nothing else.
    #[cfg(windows)]
    let spawned = travsr_plugin_host::sandbox::windows::spawn_detached_with_inherit_allowlist(
        exe,
        &["daemon", "start", "--foreground"],
    );

    if spawned.is_err() {
        return SpawnOutcome::Failed;
    }
    // Confirm the child actually came up. Polling only the lock reports a false
    // "Started" (travsr #592): the daemon takes the lock early, then opens its
    // stores, scans the tree, and only then binds its control socket — a bind
    // failure (a path over SUN_LEN, a hostile fallback dir) happens well after
    // the lock is held. So we wait for a *terminal* state instead of a timer:
    //   • the daemon answers on its socket            → Started (truly up)
    //   • it wrote a startup-error breadcrumb          → Failed
    //   • the lock was held then released (child died) → Failed
    //   • still holding the lock at the deadline       → Starting (big-repo scan)
    let mut saw_lock = false;
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        // Positive terminal state: a round-trip to the control socket succeeds.
        if send_daemon_command(repo_root, &travsr_ipc::ControlMessage::Status).is_ok() {
            return SpawnOutcome::Started;
        }
        if err_path.exists() {
            return SpawnOutcome::Failed;
        }
        if daemon_lock_held(repo_root) {
            saw_lock = true;
        } else if saw_lock {
            // Lock was held then released → the child died during startup.
            return SpawnOutcome::Failed;
        }
    }
    // Deadline reached: alive and scanning is fine; never having taken the lock
    // means the child never really started.
    if saw_lock {
        SpawnOutcome::Starting
    } else {
        SpawnOutcome::Failed
    }
}

/// Send one control message to the daemon for `repo_root`.
///
/// Dispatches to the platform-appropriate transport:
/// Unix → `UnixTransport` (domain socket), Windows → `NamedPipeTransport`.
pub fn send_daemon_command(
    repo_root: &Path,
    msg: &travsr_ipc::ControlMessage,
) -> anyhow::Result<travsr_ipc::ControlResponse> {
    let addr = travsr_ipc::ControlAddr::for_repo(repo_root);

    #[cfg(unix)]
    {
        let travsr_dir = repo_root.join(".travsr");
        let mut t = travsr_ipc::unix::UnixTransport::connect(&addr, &travsr_dir)?;
        travsr_ipc::ControlTransport::send_request(&mut t, msg)
    }

    #[cfg(windows)]
    {
        let mut t = travsr_ipc::windows::NamedPipeTransport::connect(&addr)?;
        travsr_ipc::ControlTransport::send_request(&mut t, msg)
    }

    #[cfg(all(not(unix), not(windows)))]
    {
        let _ = (addr, msg);
        anyhow::bail!("daemon control socket not supported on this platform")
    }
}

/// Route one query to a running daemon. `None` means "use the direct path" —
/// no daemon listening, version skew (old/new daemon), or a payload the CLI
/// cannot parse. Never errors: daemon routing is best-effort by design.
pub fn try_query<T: DeserializeOwned>(
    repo_root: &Path,
    tool: &str,
    args: serde_json::Value,
) -> Option<T> {
    let msg = travsr_ipc::ControlMessage::Query {
        protocol: travsr_ipc::QUERY_PROTOCOL_VERSION,
        tool: tool.to_string(),
        args,
    };
    let resp = send_daemon_command(repo_root, &msg).ok()?;
    // Handshake: the daemon must speak exactly our query protocol version.
    // Older daemons answer `ok:false` (parse error) with no protocol field.
    if !resp.ok || resp.protocol != Some(travsr_ipc::QUERY_PROTOCOL_VERSION) {
        return None;
    }
    serde_json::from_value(resp.result?).ok()
}

/// Direct-open fallback: read-only fast open (skips migrations and FTS/vocab/
/// synonym backfills), degrading to a full writable open when the store has
/// pending migrations or the read-only open is not possible.
pub fn open_read_store(db_path: &Path) -> anyhow::Result<SqliteStore> {
    match SqliteStore::open_read_only(db_path) {
        Ok(s) => Ok(s),
        Err(_) => Ok(SqliteStore::open(db_path)?),
    }
}

/// Warn on stderr when the call-graph index cannot answer authoritatively.
///
/// The MCP tools have carried this note since #617; the CLI never did. So a
/// terminal user running `travsr references X` while Phase B was still building
/// got an empty result and no indication that empty meant "not indexed yet"
/// rather than "no callers". The agent was told; the human was not.
///
/// Deliberately a warning rather than terminal progress output. Once `daemon
/// start` returns, the daemon is detached and owns no terminal, so nothing can
/// be pushed while indexing runs. What a user actually needs is not "work is
/// happening" but "the answer you are reading may be incomplete" — which is
/// only worth saying at the moment they ask.
///
/// Opens its own read-only handle because `travsr ask` answers from the daemon
/// on the warm path and never opens a store locally at all. Three metadata
/// reads, so the cost does not justify threading a store through every caller.
///
/// Silent on any error: a freshness note is not worth failing a query over.
///
/// A linked worktree served by another checkout's index takes precedence: its
/// note names both trees and states plainly that waiting or re-indexing will
/// never make the served index describe this worktree, which is exactly the
/// advice the freshness note ("has not caught up with the current commit") gives
/// and gets wrong here. The one true half of the freshness note — that empty
/// results from a degraded index are not authoritative — is folded into the
/// cross-checkout note instead (see `repo::cross_checkout_note`), so it is not
/// lost, and the misleading "it will catch up to your tree" framing is not said.
///
/// Returns whether the cross-checkout note was emitted, so a caller can suppress
/// its own hedged drift note keyed to the note that was actually printed rather
/// than to a fresh recomputation of the same predicate.
pub fn warn_if_call_graph_degraded(db_path: &Path) -> bool {
    // Every caller of this entry (`ask`, `references`, `graph --all`, and the
    // call-edge directions of `graph`) rides call edges, so the folded-in
    // degraded caveat applies to all of them.
    let cross = warn_if_cross_checkout(db_path, true);
    if !cross {
        warn_if_phase_b_degraded(db_path);
    }
    cross
}

/// The Phase B completeness half of [`warn_if_call_graph_degraded`], on its own.
/// Split out so a caller that has already decided the cross-checkout question
/// (`graph`, which emits the cross-checkout note for every direction but the
/// completeness note only for the call-edge directions) can reach it without
/// re-running the cross-checkout classification.
pub(crate) fn warn_if_phase_b_degraded(db_path: &Path) {
    if let Ok(store) = open_read_store(db_path) {
        if let Some(note) = travsr_mcp::phase_b_degraded_note(&store) {
            eprintln!("warning: {note}");
        }
    }
}

/// Warn on stderr when this command is answering out of a **different
/// checkout's** index: the caller stands in a linked worktree whose reads are
/// redirected to the main worktree (`repo::find_git_root`). Returns whether the
/// note was emitted.
///
/// Separate from [`warn_if_call_graph_degraded`] because the two conditions are
/// unrelated. That one is about how complete the call graph is, so commands
/// riding only Phase A edges skip it; this one is about which tree the answer
/// describes, which is wrong for `deps`, `pattern` and `status` just as much as
/// for `callers`. A complete answer about the wrong tree is still wrong.
///
/// `reads_call_edges` says whether *this command's* answer rides call edges,
/// and so whether the folded-in degraded caveat ("an empty or short result from
/// it is not authoritative either") applies to what it is about to print. It is
/// the same question the Phase B split above answers, asked by the same call
/// sites: `pattern` greps a file set and `graph --direction deps` rides Phase A
/// import edges, so both are complete whether or not Phase B has run, and
/// telling their users to distrust a complete result is the exact failure those
/// exemptions exist to prevent. Standing in a worktree must not bring the claim
/// back through a different door.
///
/// Classifies from the filesystem first and opens the store only once the note
/// is certain, so the ordinary case costs no store open at all. When it does
/// open, it reads both the commit to name and whether the served index is
/// Phase B degraded from the *same* handle, so the degraded caveat costs no
/// second open.
pub fn warn_if_cross_checkout(db_path: &Path, reads_call_edges: bool) -> bool {
    let Ok(cwd) = std::env::current_dir() else {
        return false;
    };
    let Some(here) = crate::repo::served_by_other_checkout(&cwd, db_path) else {
        return false;
    };
    let Some(served) = db_path.parent().and_then(Path::parent) else {
        return false;
    };
    // Classify first, mute second. `TRAVSR_NO_WORKTREE_NOTE` silences this note;
    // it must not also report "not a cross-checkout", because callers key the
    // suppression of the freshness and drift notes to this return value and
    // both of those are wrong here whether or not the user wants to read about
    // it. Opting out of one accurate note would otherwise hand back two
    // misleading ones. Returning before the store open keeps the hatch as cheap
    // as it was.
    if crate::repo::worktree_note_suppressed() {
        return true;
    }
    let (commit, degraded) = match open_read_store(db_path) {
        Ok(s) => (
            s.get_meta("last_commit").ok().flatten(),
            reads_call_edges && travsr_mcp::phase_b_degraded_note(&s).is_some(),
        ),
        Err(_) => (None, false),
    };
    eprintln!(
        "warning: {}",
        crate::repo::cross_checkout_note(&here, served, commit.as_deref(), degraded)
    );
    true
}

#[cfg(test)]
mod lock_tests {
    use super::*;

    #[test]
    fn lock_not_held_on_empty_repo() {
        let tmp = tempfile::tempdir().unwrap();
        // No .travsr dir → cannot open lock → not held (no daemon possible).
        assert!(!daemon_lock_held(tmp.path()));
        // With .travsr present but nobody holding the flock → still not held.
        std::fs::create_dir_all(tmp.path().join(".travsr")).unwrap();
        assert!(!daemon_lock_held(tmp.path()));
    }

    /// #636 round-2 review: a liveness probe must not create the file it
    /// probes. The old `.create(true)` open left a zero-byte
    /// `.travsr/daemon.lock` behind on every non-interactive `travsr init`.
    #[test]
    fn probing_never_creates_the_lock_file() {
        let tmp = tempfile::tempdir().unwrap();
        let travsr = tmp.path().join(".travsr");
        std::fs::create_dir_all(&travsr).unwrap();

        assert!(!daemon_lock_held(tmp.path()));
        assert!(
            !travsr.join("daemon.lock").exists(),
            "probing must not create .travsr/daemon.lock"
        );
    }

    /// The singleton semantics the probe exists for are unchanged: an
    /// exclusively locked, already-present lock file still reads as held.
    #[test]
    fn lock_held_for_an_existing_exclusively_locked_file() {
        use fs2::FileExt as _;
        let tmp = tempfile::tempdir().unwrap();
        let travsr = tmp.path().join(".travsr");
        std::fs::create_dir_all(&travsr).unwrap();
        std::fs::write(travsr.join("daemon.lock"), b"12345").unwrap();

        let held = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(travsr.join("daemon.lock"))
            .unwrap();
        held.lock_exclusive().unwrap();
        assert!(daemon_lock_held(tmp.path()));

        fs2::FileExt::unlock(&held).unwrap();
        drop(held);
        assert!(!daemon_lock_held(tmp.path()));
        // Still there, still holding the daemon's PID: the probe never
        // truncated or clobbered it.
        assert_eq!(
            std::fs::read_to_string(travsr.join("daemon.lock")).unwrap(),
            "12345"
        );
    }

    #[test]
    fn lock_held_when_another_fd_holds_flock() {
        use fs2::FileExt as _;
        let tmp = tempfile::tempdir().unwrap();
        let travsr = tmp.path().join(".travsr");
        std::fs::create_dir_all(&travsr).unwrap();
        // Simulate a running daemon: hold an exclusive flock on daemon.lock.
        let held = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(travsr.join("daemon.lock"))
            .unwrap();
        held.lock_exclusive().unwrap();
        assert!(
            daemon_lock_held(tmp.path()),
            "must report held while flock is taken"
        );
        fs2::FileExt::unlock(&held).unwrap();
        drop(held);
        assert!(
            !daemon_lock_held(tmp.path()),
            "must report free once released"
        );
    }

    /// The note is shared with the MCP tools rather than reimplemented, so what
    /// is worth pinning here is that the CLI classifies the same three states
    /// the same way — and, just as importantly, stays silent on the fourth.
    #[test]
    fn the_cli_and_the_mcp_tools_agree_on_call_graph_completeness() {
        use travsr_store::SqliteStore;

        let set = |s: &mut SqliteStore, pairs: &[(&str, &str)]| {
            for (k, v) in pairs {
                s.set_meta(k, v).unwrap();
            }
        };

        // Phase B never ran: init happened, no phase_b marker.
        let mut store = SqliteStore::open_in_memory().unwrap();
        set(&mut store, &[("last_commit", "abc1234")]);
        assert!(travsr_mcp::phase_b_degraded_note(&store).is_some());

        // Phase B ran, but HEAD has moved past it.
        let mut store = SqliteStore::open_in_memory().unwrap();
        set(
            &mut store,
            &[("last_commit", "def5678"), ("phase_b_commit", "abc1234")],
        );
        assert!(travsr_mcp::phase_b_degraded_note(&store).is_some());

        // Markers agree, but a watcher reindex dropped call edges (#583).
        let mut store = SqliteStore::open_in_memory().unwrap();
        set(
            &mut store,
            &[
                ("last_commit", "abc1234"),
                ("phase_b_commit", "abc1234"),
                ("phase_b_dirty", "1"),
            ],
        );
        assert!(travsr_mcp::phase_b_degraded_note(&store).is_some());

        // Complete and clean: silent. A note that always fires is noise, and
        // noise is how the ADR-017 warning made the daemon log unreadable.
        let mut store = SqliteStore::open_in_memory().unwrap();
        set(
            &mut store,
            &[("last_commit", "abc1234"), ("phase_b_commit", "abc1234")],
        );
        assert_eq!(travsr_mcp::phase_b_degraded_note(&store), None);
    }
}
