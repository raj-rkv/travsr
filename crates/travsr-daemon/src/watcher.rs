//! File-system watcher — spawns a notify-backed producer thread.
//!
//! Uses `notify` v8 (already a workspace dependency) directly rather than a
//! debouncer wrapper, to avoid pulling in a conflicting version of the notify
//! types crate. A 500 ms coalescing window is implemented with a HashMap of
//! pending events flushed by a background tick.
//!
//! The watcher runs on a plain OS thread so that kqueue/inotify descriptors
//! have a proper thread to live on. A `WatcherHandle` provides a shutdown
//! signal so that the watcher exits cleanly when the daemon stops.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::event::{AccessKind, ModifyKind, RenameMode};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

/// Events the daemon's indexer worker receives from the watcher.
#[derive(Debug)]
pub enum WatchEvent {
    Upsert(PathBuf),
    Remove(PathBuf),
}

/// A handle to the running watcher threads. Drop to signal shutdown.
pub struct WatcherHandle {
    stop: Arc<AtomicBool>,
    /// #801: raw events accepted from the OS, counted BEFORE any userspace
    /// filtering.
    ///
    /// Observable because the userspace filter makes the interesting failure
    /// invisible from the outside: `should_skip_all` discards `target/` events
    /// either way, so "no reindex happened" is equally true whether the tree is
    /// unwatched or whether every one of its 251,662 files is being watched,
    /// queued and stat'd first. That indistinguishability is precisely why the
    /// existing filter tests passed while the daemon grew 13 MB/s.
    raw_events: Arc<AtomicU64>,
    /// Raw events dropped because the bounded queue was full (#801).
    raw_dropped: Arc<AtomicU64>,
    _event_thread: std::thread::JoinHandle<()>,
    _flush_thread: std::thread::JoinHandle<()>,
}

impl WatcherHandle {
    /// Raw OS events accepted so far, before userspace filtering.
    pub fn raw_events(&self) -> u64 {
        self.raw_events.load(Ordering::Relaxed)
    }

    /// Raw OS events shed because the bounded queue was full.
    pub fn raw_dropped(&self) -> u64 {
        self.raw_dropped.load(Ordering::Relaxed)
    }
}

impl Drop for WatcherHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// Debounce window — coalesce rapid saves into a single reindex call.
const DEBOUNCE_MS: u64 = 500;

/// Hard cap on the debounce table size.
///
/// At ~150 bytes per entry (PathBuf + PendingKind + Instant + HashMap overhead)
/// this limits the table to ~15 MB. During a pathological event flood (e.g. a
/// `git checkout` touching 100k+ files while the single indexer worker is
/// saturated), the bounded tokio channel back-pressures the flush thread which
/// lets the debounce table grow unboundedly without this cap.
///
/// Policy: updates to *already-pending* paths coalesce as normal. Brand-new
/// paths beyond the cap are dropped with a throttled warning. The post-commit
/// hook reconciles via `git diff-tree` on the next commit; `travsr init`
/// always fully recovers the graph.
const MAX_PENDING: usize = 100_000;

/// #801: capacity of the raw OS-event queue between the notify callback and the
/// event loop.
///
/// Sized well above any burst a human edit or a `git checkout` produces, and far
/// below the point where holding that many `notify::Event` values costs real
/// memory, so it only engages on a genuine flood. Deliberately smaller than
/// `MAX_PENDING`: this queue holds unfiltered events, most of which the very
/// next step discards, whereas the debounce table holds paths already known to
/// be relevant and worth keeping.
const RAW_EVENT_CAP: usize = 8_192;

/// Mirrored by `travsr-mcp`'s own `SKIP_DIRS` so `find_pattern` searches the
/// same file universe the graph is built from (#448). The dependency rules run
/// `travsr-daemon → travsr-mcp`, so the constant cannot be shared; keep the two
/// lists identical.
pub(crate) const SKIP_DIRS: &[&str] = &[
    ".claude",
    ".git",
    ".travsr",
    "target",
    "node_modules",
    "dist",
    ".next",
    ".vscode",
    ".vscode-test",
];

/// Internal pending-event entry for the debounce table.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PendingKind {
    Upsert,
    Remove,
}

/// Spawn the file-system watcher. Returns a `WatcherHandle`; drop it to stop.
///
/// Blocks until the underlying kqueue/inotify watch is fully established and
/// skip directories have been unwatched. This guarantees that when `spawn`
/// returns, the caller can safely create files in `.travsr/` (e.g. the daemon
/// control socket) without triggering kqueue ENOTSUP errors.
///
/// Events that originate from files whose mtime predates `start_time`
/// (FSEvents on macOS can replay old events on startup) are silently dropped.
pub fn spawn(
    repo_root: &Path,
    tx: mpsc::Sender<WatchEvent>,
    start_time: Instant,
) -> anyhow::Result<WatcherHandle> {
    let repo_root = repo_root.to_path_buf();
    let gitignore = build_ignore_matcher(&repo_root);
    let stop = Arc::new(AtomicBool::new(false));

    // Pending debounce table shared between the notify callback and the flush thread.
    let pending: Arc<Mutex<HashMap<PathBuf, (PendingKind, Instant)>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // #801: bounded. This channel sits UPSTREAM of the debounce table's
    // `MAX_PENDING` cap, so that cap never applied to it: a flood grew this
    // queue without limit, at ~13 MB/s on a repo with a populated `target/`,
    // until the machine thrashed. Shedding is the right policy for a
    // reindex trigger, exactly as it already is for the debounce table: the
    // next event for a path supersedes the one dropped, and the periodic
    // reconcile catches anything missed entirely.
    let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<notify::Event>(RAW_EVENT_CAP);
    let raw_events = Arc::new(AtomicU64::new(0));
    let raw_dropped = Arc::new(AtomicU64::new(0));

    // ── Flush thread ─────────────────────────────────────────────────────────
    // Every DEBOUNCE_MS, emit entries whose deadline has passed. Runs on an OS
    // thread so it doesn't interact with Tokio's blocking thread pool.
    let pending_flush = Arc::clone(&pending);
    let tx_flush = tx.clone();
    let stop_flush = Arc::clone(&stop);
    let flush_thread = std::thread::Builder::new()
        .name("travsr-watcher-flush".into())
        .spawn(move || {
            while !stop_flush.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(DEBOUNCE_MS));
                if stop_flush.load(Ordering::Acquire) {
                    break;
                }
                let now = Instant::now();
                let ready: Vec<(PathBuf, PendingKind)> = {
                    let mut guard = pending_flush.lock().unwrap_or_else(|e| e.into_inner());
                    let ready: Vec<(PathBuf, PendingKind)> = guard
                        .iter()
                        .filter(|(_, (_, deadline))| *deadline <= now)
                        .map(|(p, (k, _))| (p.clone(), k.clone()))
                        .collect();
                    for (path, _) in &ready {
                        guard.remove(path);
                    }
                    ready
                };
                for (path, kind) in ready {
                    let ev = match kind {
                        PendingKind::Remove => WatchEvent::Remove(path),
                        PendingKind::Upsert => WatchEvent::Upsert(path),
                    };
                    if tx_flush.blocking_send(ev).is_err() {
                        return; // channel closed — daemon shutting down
                    }
                }
            }
        })
        .expect("failed to spawn watcher flush thread");

    // Ready channel: the event thread signals Ok(()) once the watch is fully
    // set up and skip dirs are unwatched. spawn() blocks on this before
    // returning so the caller cannot create the socket before kqueue is done
    // scanning — preventing the ENOTSUP race on .travsr/daemon.sock.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

    // ── Event-processing thread ───────────────────────────────────────────────
    // Owns the `RecommendedWatcher` so it lives on a stable OS thread.
    let raw_events_cb = Arc::clone(&raw_events);
    let raw_dropped_cb = Arc::clone(&raw_dropped);
    let stop_event = Arc::clone(&stop);
    let event_thread = std::thread::Builder::new()
        .name("travsr-watcher-event".into())
        .spawn(move || {
            // #403: the ignore matcher is rebuilt in-place when a
            // .gitignore/.travsrignore changes, so it must be mutable. It lives
            // only on this event thread, so no synchronisation is needed.
            let mut gitignore = gitignore;
            let mut watcher = match RecommendedWatcher::new(
                move |res: notify::Result<notify::Event>| {
                    if let Ok(ev) = res {
                        raw_events_cb.fetch_add(1, Ordering::Relaxed);
                        // try_send, never send: this runs on the notify thread,
                        // and blocking it stalls the OS event source itself.
                        if let Err(std::sync::mpsc::TrySendError::Full(_)) = raw_tx.try_send(ev) {
                            let n = raw_dropped_cb.fetch_add(1, Ordering::Relaxed) + 1;
                            if n == 1 || n % 10_000 == 0 {
                                tracing::warn!(
                                    dropped = n,
                                    cap = RAW_EVENT_CAP,
                                    "watcher event queue full; shedding events. \
                                     The periodic reconcile still catches missed paths."
                                );
                            }
                        }
                    }
                },
                notify::Config::default(),
            ) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!("failed to create file watcher: {e}");
                    let _ = ready_tx.send(Err(anyhow::anyhow!("failed to create watcher: {e}")));
                    return;
                }
            };

            if let Err(e) = watcher.watch(&repo_root, RecursiveMode::Recursive) {
                tracing::error!("failed to watch {}: {e}", repo_root.display());
                let _ = ready_tx.send(Err(anyhow::anyhow!("failed to watch repo: {e}")));
                return;
            }

            // #801: drop the watches this module has always claimed to drop.
            // Two doc comments promised it and no code did it, so a recursive
            // watch covered every SKIP_DIRS tree: on a built Rust repo that is
            // 15,664 directories and 251,662 files under `target/` alone,
            // producing events that were queued and stat'd before being
            // discarded by `should_skip_all`.
            //
            // This is not only a cost fix. `spawn`'s contract, which
            // `lib.rs` relies on before binding the control socket, is that
            // `.travsr/` is unwatched by the time it returns; otherwise the
            // kqueue backend opens `daemon.sock` and gets ENOTSUP, killing the
            // whole watch. That guarantee has never actually held.
            //
            // Failures are logged and not fatal: a skip dir that does not exist
            // in this repo is the common case, and losing the optimisation is
            // never worth refusing to watch the repo at all.
            unwatch_skipped_subtrees(&mut watcher, &repo_root, &gitignore);

            // Signal ready, watch is established and skip dirs are unwatched,
            // so the caller can proceed.
            let _ = ready_tx.send(Ok(()));

            // Block on raw events until the stop flag is set.
            // `raw_rx` unblocks when `raw_tx` is dropped (watcher is dropped above
            // when this scope exits, but we need to exit the loop first).
            // Use a timeout-based recv to periodically check the stop flag.
            let mut dropped_total: u64 = 0;
            loop {
                if stop_event.load(Ordering::Acquire) {
                    break;
                }
                match raw_rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(event) => {
                        // #801: a READ is not a change, and on inotify it is
                        // reported as one. `notify` subscribes to `IN_OPEN`
                        // (8.2.0 `inotify.rs:427`), so every file the daemon
                        // opens comes back to it as an event, and the two reads
                        // it does on the hot path close a cycle:
                        //
                        //   read .gitignore -> Access(Open) -> is_ignore_file ->
                        //   build_ignore_matcher reads every .gitignore -> ...
                        //
                        //   edit a file -> Upsert -> reindex_files reads it ->
                        //   Access(Open) -> newer than daemon start, so the
                        //   staleness guard passes it -> Upsert -> ...
                        //
                        // The first spins at ~150k events/sec from a single
                        // `cat .gitignore` by any process; the second reindexes
                        // every edited file once per debounce interval until the
                        // daemon restarts. Neither is reachable on fsevent,
                        // windows or kqueue: none of them emit `Access` at all,
                        // which is why this reproduces only on Linux.
                        //
                        // Placed above the matcher rebuild because that rebuild
                        // runs before any per-path filtering, so nothing below
                        // can break the cycle. `Access(Close(Write))` is kept: it
                        // is a genuine write signal.
                        if matches!(
                            event.kind,
                            EventKind::Access(AccessKind::Open(_) | AccessKind::Read)
                        ) {
                            continue;
                        }

                        // #403: a change to any .gitignore/.travsrignore
                        // invalidates the cached matcher — rebuild it so newly
                        // ignored paths stop being indexed without a daemon
                        // restart. Done once per event before its paths are
                        // filtered so the new rules apply to this batch too.
                        if event.paths.iter().any(|p| is_ignore_file(p)) {
                            gitignore = build_ignore_matcher(&repo_root);
                        }
                        for path in &event.paths {
                            // #801 review: the startup unwatch is a one-shot
                            // snapshot, and notify re-adds a recursive watch for
                            // any directory created later under a watched parent.
                            // So `cargo clean && cargo build`, or `npm install`
                            // on a fresh clone, put the whole tree back under
                            // watch and bring the original failure with it. That
                            // is the common case for the repo shape this issue
                            // was reported on, not an edge case.
                            //
                            // Re-drop it the moment it appears. Gated on Create
                            // first, which is a cheap enum match and rare next to
                            // Modify, so the `should_skip_dir` and `is_dir` cost
                            // is paid only when a directory is actually born.
                            //
                            // `should_skip_dir`, not `should_skip_all`: a
                            // `build/` gitignore rule is directory only, so
                            // asking the file-shaped question about a directory
                            // answers "not ignored" and the tree stays watched.
                            if matches!(event.kind, EventKind::Create(_))
                                && should_skip_dir(path, &repo_root, &gitignore)
                                && path.is_dir()
                            {
                                if let Err(e) = watcher.unwatch(path) {
                                    tracing::debug!(
                                        "unwatch {} not applied (non-fatal): {e}",
                                        path.display()
                                    );
                                }
                                continue;
                            }

                            // #801: the cheap prefix test runs FIRST. It used to
                            // run after the `metadata` call below, so every event
                            // from a skipped tree paid a stat syscall before being
                            // rejected on a string compare. Ordering only: a
                            // skipped path is discarded either way, so nothing
                            // downstream sees a different set of paths.
                            if should_skip_all(path, &repo_root, &gitignore) {
                                continue;
                            }

                            // Drop events predating daemon start (FSEvents replay guard).
                            if let Ok(meta) = std::fs::metadata(path) {
                                if let Ok(modified) = meta.modified() {
                                    if modified
                                        .elapsed()
                                        .map(|age| age > start_time.elapsed())
                                        .unwrap_or(false)
                                    {
                                        continue;
                                    }
                                }
                            }

                            // Name(From) is the "old path" half of a rename — treat
                            // it as Remove so deleted nodes are tombstoned.
                            // Name(To) and all other events are Upsert.
                            // For Upsert: also apply the upsert-specific filter
                            // (is_file + language ext). NOT applied to Remove: the
                            // file no longer exists so is_file() returns false.
                            let kind = match event.kind {
                                EventKind::Remove(_)
                                | EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
                                    PendingKind::Remove
                                }
                                _ => {
                                    if should_skip_upsert(path) {
                                        continue;
                                    }
                                    PendingKind::Upsert
                                }
                            };

                            let deadline = Instant::now() + Duration::from_millis(DEBOUNCE_MS);
                            let mut guard = pending.lock().unwrap_or_else(|e| e.into_inner());
                            // Coalesce rules:
                            // - Upsert → Remove: Remove wins (file deleted, pending write discarded).
                            // - Remove → Upsert: Upsert wins (file recreated; must re-index).
                            // - Upsert → Upsert / Remove → Remove: update deadline, keep kind.
                            // Updates to existing pending paths always coalesce — only
                            // brand-new paths are gated by MAX_PENDING.
                            match guard.get(path) {
                                Some((PendingKind::Upsert, _)) if kind == PendingKind::Remove => {
                                    guard.insert(path.clone(), (PendingKind::Remove, deadline));
                                }
                                Some(_) => {
                                    guard.insert(path.clone(), (kind, deadline));
                                }
                                None => {
                                    if guard.len() < MAX_PENDING {
                                        guard.insert(path.clone(), (kind, deadline));
                                    } else {
                                        dropped_total += 1;
                                        if dropped_total == 1 || dropped_total % 10_000 == 0 {
                                            tracing::warn!(
                                                dropped = dropped_total,
                                                cap = MAX_PENDING,
                                                "watcher debounce table full, new paths \
                                                 dropped (run `travsr init` to reconcile)"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Normal — poll the stop flag and continue.
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        break;
                    }
                }
            }
            // `watcher` is dropped here, which stops the kernel watch.
        })
        .expect("failed to spawn watcher event thread");

    // Block until the event thread confirms the watch is established.
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => anyhow::bail!("watcher event thread exited before signalling ready"),
    }

    Ok(WatcherHandle {
        stop,
        raw_events,
        raw_dropped,
        _event_thread: event_thread,
        _flush_thread: flush_thread,
    })
}

/// Build the combined ignore matcher used to filter watcher events.
///
/// Collects every `.gitignore` and `.travsrignore` under `repo_root` into one
/// matcher (#403). `.travsrignore` files are added after `.gitignore` so their
/// rules take precedence, mirroring `init_repo`'s
/// `SKIP_DIRS < .gitignore < .travsrignore` ordering. The walk skips `SKIP_DIRS`
/// so it never descends into `node_modules/`, `target/`, etc. looking for nested
/// ignore files.
pub(crate) fn build_ignore_matcher(repo_root: &Path) -> Gitignore {
    let mut builder = GitignoreBuilder::new(repo_root);
    // Two passes so all `.gitignore` rules are added before any `.travsrignore`
    // rule: later-added patterns win in gitignore semantics, so `.travsrignore`
    // (the user's Travsr-specific overrides) takes precedence.
    for target in [".gitignore", ".travsrignore"] {
        for entry in walkdir::WalkDir::new(repo_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                // Do not descend into SKIP_DIRS — matches init and avoids a slow
                // first walk into node_modules/ on large repos.
                !(e.file_type().is_dir()
                    && e.file_name()
                        .to_str()
                        .is_some_and(|n| SKIP_DIRS.contains(&n)))
            })
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() && entry.file_name() == target {
                let _ = builder.add(entry.into_path());
            }
        }
    }
    builder.build().unwrap_or(Gitignore::empty())
}

/// True when `path` is a `.gitignore` / `.travsrignore` file whose creation,
/// edit, or removal should trigger a matcher rebuild (#403).
pub(crate) fn is_ignore_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some(".gitignore" | ".travsrignore")
    )
}

/// Filter applied to ALL events (Upsert and Remove).
/// Skips SKIP_DIRS components and gitignored/travsrignored paths.
pub(crate) fn should_skip_all(path: &Path, repo_root: &Path, gitignore: &Gitignore) -> bool {
    skip_check(path, repo_root, gitignore, false)
}

/// [`should_skip_all`] for a path known to be a directory.
///
/// #801: the two cannot share one `is_dir` value, and the difference is not
/// cosmetic. A `build/` rule is directory-only, so matching the directory
/// `build` itself requires `is_dir = true`; asking with `false` says "not
/// ignored" and a caller walking the tree then descends into the very subtree it
/// meant to prune. That is exactly what `skipped_subtree_roots` did before this
/// split: it returned `build/obj` instead of `build`, having already walked all
/// of `build`.
///
/// Event filtering keeps `is_dir = false`, because a watcher event names a path
/// that is usually a file and often already deleted, so `is_dir` cannot be
/// probed reliably there. Files under an ignored directory are still caught by
/// the `_or_any_parents` half.
pub(crate) fn should_skip_dir(path: &Path, repo_root: &Path, gitignore: &Gitignore) -> bool {
    skip_check(path, repo_root, gitignore, true)
}

/// Shared body of [`should_skip_all`] and [`should_skip_dir`], so the SKIP_DIRS
/// half can never drift between them.
fn skip_check(path: &Path, repo_root: &Path, gitignore: &Gitignore, is_dir: bool) -> bool {
    let rel = path.strip_prefix(repo_root).unwrap_or(path);
    if rel
        .components()
        .any(|c| SKIP_DIRS.iter().any(|skip| c.as_os_str() == *skip))
    {
        return true;
    }
    // `matched_path_or_any_parents` (not `matched`) so a directory rule like
    // `vendor/` or `**/generated/` excludes the files *under* it (#403). Plain
    // `matched` only tests the exact path and never applies the parent-dir rule.
    gitignore
        .matched_path_or_any_parents(rel, is_dir)
        .is_ignore()
}

/// #801: drop the OS watches for every subtree `should_skip_all` would discard.
///
/// Derived from the filter rather than restating it. An earlier version of this
/// walked `SKIP_DIRS` alone, which left the *other* half of the filter
/// unaddressed: `should_skip_all` also honours `.gitignore` / `.travsrignore`,
/// so a repo whose ignored `build/`, `vendor/` or `.venv/` holds tens of
/// thousands of files stayed fully watched and paid exactly the cost this fix
/// exists to remove. Asking the filter means the watched set and the discarded
/// set cannot disagree, including for a `.travsrignore` re-include (`!vendored/`),
/// which keeps its watch because the filter says to keep it.
///
/// Prunes rather than walking the whole repo: a directory that is skipped is
/// never descended into. The traversal therefore costs work proportional to the
/// tree that stays watched, which is the tree the daemon has to enumerate
/// anyway, not to the `target/` being excluded.
///
/// `unwatch` failures are logged and never fatal. `WatchNotFound` is the
/// expected result on FSEvents (the macOS default backend), which streams one
/// recursive watch from the root and has no per-directory watch to remove;
/// verified against notify 8.2.0 rather than assumed. Backends with
/// per-directory watches (inotify, the platform the 13 MB/s failure was
/// measured on) genuinely drop the subtree here. On FSEvents the cost is bounded
/// instead by the reordered filter and the bounded queue, which is why those are
/// not optional extras to this fix.
fn unwatch_skipped_subtrees(
    watcher: &mut RecommendedWatcher,
    repo_root: &Path,
    gitignore: &Gitignore,
) {
    for path in skipped_subtree_roots(repo_root, gitignore) {
        if let Err(e) = watcher.unwatch(&path) {
            tracing::debug!("unwatch {} not applied (non-fatal): {e}", path.display());
        }
    }
}

/// The shallowest directories `should_skip_all` discards, for [`unwatch_skipped_subtrees`].
///
/// Split from the unwatching so the decision is testable without a live watcher,
/// and therefore on every platform rather than only the ones where `unwatch`
/// does something. The end-to-end watch behaviour is covered separately.
///
/// Prunes: a skipped directory is returned and never descended into, so the walk
/// costs work proportional to the tree that stays watched, which the daemon has
/// to enumerate anyway, not to the `target/` being excluded.
fn skipped_subtree_roots(repo_root: &Path, gitignore: &Gitignore) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut stack = vec![repo_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // Unreadable directory: leave its watch in place. Failing to prune is
            // a lost optimisation; refusing to watch would be a lost edit.
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if should_skip_dir(&path, repo_root, gitignore) {
                roots.push(path);
            } else {
                stack.push(path);
            }
        }
    }
    roots
}

/// Additional filter applied only to Upsert events.
/// Skips non-regular files and files with no indexable language extension.
/// NOT applied to Remove events: the file no longer exists on disk when the
/// event fires, so `is_file()` would return false for a just-deleted source file.
fn should_skip_upsert(path: &Path) -> bool {
    if !path.is_file() {
        return true;
    }
    // Skip files that are neither a recognized source/data-format extension nor
    // a name-recognized manifest (go.mod, *.csproj). Single source of truth so
    // the watcher and the init walk admit exactly the same files.
    !travsr_core::is_indexable_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #801: a SKIP_DIRS tree must not be WATCHED, not merely filtered.
    ///
    /// The distinction is the entire bug. `should_skip_all` works and always
    /// did, so every `target/` event was correctly discarded, but only AFTER
    /// being delivered, queued and stat'd. On a built repo (15,664 dirs,
    /// 251,662 files) that grew the daemon 13 MB/s until the machine thrashed.
    ///
    /// Asserting "no reindex happened" would therefore pass on the broken code
    /// too: the filter guarantees that either way. This asserts on `raw_events`,
    /// counted in the notify callback before any filtering, which is the only
    /// place the difference is observable from outside.
    ///
    /// Linux only, and not for convenience. `unwatch` needs per-directory
    /// watches, which inotify has and FSEvents does not: FSEvents streams one
    /// recursive watch from the root, and `unwatch` on a subdirectory returns
    /// `WatchNotFound` (verified against notify 8.2.0, not assumed). So this
    /// property is unachievable on the macOS default backend, and asserting it
    /// there would pin a behaviour no code can deliver. macOS is covered by
    /// `raw_event_queue_is_bounded_under_flood` below, which holds everywhere.
    /// Linux is also the platform the reported failure was measured on.
    #[cfg(target_os = "linux")]
    #[test]
    fn skip_dirs_are_not_watched_only_filtered() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();

        let (tx, _rx) = mpsc::channel::<WatchEvent>(1024);
        let handle = spawn(root, tx, Instant::now()).expect("watcher spawns");

        // Control arm first: a watched path must move the counter, otherwise the
        // assertion below is vacuous. Without this the whole test passes if the
        // counter simply never increments, which is the shape of guard that let
        // this ship in the first place.
        for i in 0..40 {
            std::fs::write(root.join(format!("src/f{i}.ts")), "export const x = 1;").unwrap();
        }
        wait_until(|| handle.raw_events() > 0, Duration::from_secs(5));
        let after_watched = settled(|| handle.raw_events(), Duration::from_secs(10));
        assert!(
            after_watched > 0,
            "a write inside the repo must produce raw events, or this test proves nothing"
        );

        for i in 0..400 {
            std::fs::write(root.join(format!("target/debug/deps/x{i}.rlib")), "junk").unwrap();
        }
        let from_skip = settled(|| handle.raw_events(), Duration::from_secs(10)) - after_watched;
        assert!(
            from_skip < 40,
            "400 writes under target/ produced {from_skip} raw events: the tree is \
             still watched, so every file is queued and stat'd before the skip \
             filter discards it (#801). Expected the tree to be unwatched."
        );
    }

    /// #801: the unwatch set is derived from the filter, on every platform.
    ///
    /// Runs everywhere because it asserts the DECISION, not the `unwatch` call:
    /// `unwatch` is a no-op on FSEvents, so an end-to-end test can only run on
    /// Linux, and that would leave this logic unverified on the machine most of
    /// this repo is developed on.
    ///
    /// The case that motivated splitting it out: the first version of the fix
    /// walked `SKIP_DIRS` alone and missed the other half of `should_skip_all`,
    /// so an ignored `build/` with tens of thousands of files stayed watched.
    #[test]
    fn skipped_subtree_roots_follow_the_filter_not_a_second_list() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".gitignore"), "build/\nvendored/\n").unwrap();
        std::fs::write(root.join(".travsrignore"), "!vendored/\n").unwrap();
        for d in [
            "target/debug",   // SKIP_DIRS
            "node_modules/a", // SKIP_DIRS
            "build/obj",      // gitignored: the case the first fix missed
            "vendored/pkg",   // gitignored then RE-INCLUDED: must be kept
            "src/inner",      // ordinary
        ] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }

        let gi = build_ignore_matcher(root);
        let roots = skipped_subtree_roots(root, &gi);
        let rel: std::collections::HashSet<String> = roots
            .iter()
            .map(|p| {
                p.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();

        for want in ["target", "node_modules", "build"] {
            assert!(rel.contains(want), "{want} must be unwatched; got {rel:?}");
        }
        for keep in ["vendored", "src"] {
            assert!(
                !rel.contains(keep),
                "{keep} is kept by the filter, so unwatching it would silently stop \
                 indexing a tree the user asked to index; got {rel:?}"
            );
        }
        // Pruned, not walked: nothing BELOW a skipped root is returned, which is
        // what keeps this proportional to the kept tree.
        assert!(
            !rel.iter().any(|p| p.contains('/')),
            "only the shallowest skipped directory should be returned, not its \
             children; got {rel:?}"
        );
    }

    /// #801 follow-up: a gitignored tree must be unwatched too, not just
    /// SKIP_DIRS.
    ///
    /// The first version of this fix walked `SKIP_DIRS` alone, which left half
    /// of `should_skip_all` unaddressed: it also honours `.gitignore` and
    /// `.travsrignore`. A repo whose ignored `build/` holds tens of thousands of
    /// files stayed fully watched and paid the whole cost the fix exists to
    /// remove, and no test would have noticed, since the userspace filter
    /// discards those events either way.
    ///
    /// Also pins the re-include: `!vendored/` in `.travsrignore` means the
    /// filter keeps that tree, so the watch must be kept too. Deriving the
    /// unwatch set from the filter is what makes that true for free rather than
    /// by remembering to special-case it.
    #[cfg(target_os = "linux")]
    #[test]
    fn gitignored_trees_are_unwatched_and_reincludes_are_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".gitignore"), "build/\nvendored/\n").unwrap();
        // Re-include: the filter keeps this, so the watch must survive.
        std::fs::write(root.join(".travsrignore"), "!vendored/\n").unwrap();
        std::fs::create_dir_all(root.join("build/obj")).unwrap();
        std::fs::create_dir_all(root.join("vendored")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();

        let (tx, _rx) = mpsc::channel::<WatchEvent>(1024);
        let handle = spawn(root, tx, Instant::now()).expect("watcher spawns");

        // Control: a kept tree moves the counter.
        for i in 0..40 {
            std::fs::write(root.join(format!("src/f{i}.ts")), "x").unwrap();
        }
        wait_until(|| handle.raw_events() > 0, Duration::from_secs(5));
        let base = settled(|| handle.raw_events(), Duration::from_secs(10));
        assert!(
            base > 0,
            "a watched dir must produce events, or this proves nothing"
        );

        // The gitignored tree must be invisible to the OS watch.
        for i in 0..400 {
            std::fs::write(root.join(format!("build/obj/o{i}.o")), "junk").unwrap();
        }
        let from_ignored = settled(|| handle.raw_events(), Duration::from_secs(10)) - base;
        assert!(
            from_ignored < 40,
            "400 writes under a gitignored build/ produced {from_ignored} raw \
             events: the tree is still watched (#801)"
        );

        // The re-included tree must still be watched.
        let before_reinclude = settled(|| handle.raw_events(), Duration::from_secs(10));
        for i in 0..20 {
            std::fs::write(root.join(format!("vendored/v{i}.ts")), "x").unwrap();
        }
        wait_until(
            || handle.raw_events() > before_reinclude,
            Duration::from_secs(5),
        );
        assert!(
            handle.raw_events() > before_reinclude,
            "`!vendored/` is re-included by .travsrignore, so the filter keeps it \
             and the watch must be kept: unwatching it would silently stop \
             indexing a tree the user asked to index"
        );
    }

    /// #801 review: a skip dir created AFTER spawn must be dropped too.
    ///
    /// The startup unwatch is a snapshot, and notify re-adds a recursive watch
    /// for any directory born later under a watched parent. Without the
    /// re-drop in the event loop, `cargo clean && cargo build` on a fresh clone
    /// puts the whole of `target/` back under watch and reproduces #801 in
    /// full. The reviewer measured 1604 raw events from that path against the
    /// under-40 the startup case produces.
    ///
    /// That is the common case for the repo shape this issue was reported on:
    /// a Rust repo whose `target/` does not exist until the first build.
    ///
    /// Written as two batches. The first one cannot avoid racing the unwatch, so
    /// its count says more about the machine than about the code; the second one
    /// runs after the unwatch has had time to apply, and is what the assertion
    /// rests on.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_skip_dir_created_after_spawn_is_dropped_too() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        // Deliberately NOT created before spawn: that is the whole point.
        let (tx, _rx) = mpsc::channel::<WatchEvent>(4096);
        let handle = spawn(root, tx, Instant::now()).expect("watcher spawns");

        for i in 0..20 {
            std::fs::write(root.join(format!("src/f{i}.ts")), "x").unwrap();
        }
        wait_until(|| handle.raw_events() > 0, Duration::from_secs(5));
        let base = settled(|| handle.raw_events(), Duration::from_secs(10));
        assert!(base > 0, "control: a watched dir must move the counter");

        // What `cargo build` does on a fresh clone. This first batch RACES the
        // unwatch by construction: the Create event has to be delivered and
        // processed before any later write stops being reported, so some number
        // always gets through, and that number is whatever the box was fast
        // enough to deliver in the meantime. It is therefore measured, not
        // asserted on. A bound here reads as an assertion about the machine: it
        // was 219 on a 2 core CI runner against under 40 on a developer laptop.
        std::fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        for i in 0..400 {
            std::fs::write(root.join(format!("target/debug/deps/x{i}.rlib")), "junk").unwrap();
        }
        let after_race = settled(|| handle.raw_events(), Duration::from_secs(20));
        // The producer going quiet does not prove the consumer has caught up,
        // and it is the consumer that calls `unwatch`. Give it room.
        std::thread::sleep(Duration::from_secs(1));
        let before_second = handle.raw_events();

        // The assertion, and the reason for the second batch: by now the unwatch
        // has certainly been applied, so these 400 writes are not racing
        // anything. Either the tree is unwatched and they are silent, or it was
        // re-watched and they arrive in full. That is a property of the code
        // rather than of the scheduler, so it holds on any machine.
        for i in 400..800 {
            std::fs::write(root.join(format!("target/debug/deps/x{i}.rlib")), "junk").unwrap();
        }
        let from_second = settled(|| handle.raw_events(), Duration::from_secs(20)) - before_second;

        assert!(
            from_second < 40,
            "400 writes into a target/ created after spawn produced {from_second} \
             raw events once the unwatch had had time to apply ({} during the \
             race itself): the startup unwatch is a snapshot and notify \
             re-watched the new tree, so `cargo clean && cargo build` reproduces \
             #801 (#801 review)",
            after_race - base
        );
    }

    /// #801: a full raw queue sheds rather than blocking the producer.
    ///
    /// Scope, stated because the obvious test here does not work. I first wrote
    /// this as "flood the watcher and assert it keeps accepting", and
    /// mutation-checked it by reverting to the unbounded channel: it still
    /// passed. An unbounded channel also keeps accepting, so that assertion
    /// discriminated nothing. Forcing a real overflow through the watcher is not
    /// reliable either, because the consumer drains roughly as fast as a test
    /// loop can create files, so the queue empties as fast as it fills and drop
    /// counts are timing dependent.
    ///
    /// So this pins the property that actually matters and can be asserted
    /// deterministically: when the queue is full, the producer sheds and keeps
    /// going. `send` would block here, and the producer is the notify callback,
    /// so blocking it stalls the OS event source itself: the watcher would stop
    /// noticing edits rather than merely fall behind.
    ///
    /// What this does NOT cover is the wiring, that the watcher's own channel is
    /// the bounded kind. `RAW_EVENT_CAP`'s use at the `sync_channel` call is the
    /// only place that is decided, and it is one line under review.
    #[test]
    fn a_full_raw_queue_sheds_instead_of_blocking() {
        let (tx, _rx) = std::sync::mpsc::sync_channel::<u32>(2);
        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        match tx.try_send(3) {
            Err(std::sync::mpsc::TrySendError::Full(v)) => assert_eq!(v, 3),
            other => panic!("a full bounded queue must report Full, got {other:?}"),
        }
        // And the producer is still usable afterwards: shedding is not fatal.
        let _ = _rx.recv().unwrap();
        assert!(
            tx.try_send(4).is_ok(),
            "after a slot frees the producer must resume, or a single burst would \
             permanently deafen the watcher"
        );
    }

    /// Poll `cond` until true or `budget` elapses. Watchers are inherently
    /// asynchronous, so a fixed sleep either flakes or wastes wall clock.
    ///
    /// Only the Linux watch tests need this, and `-D warnings` rejects it as
    /// dead code elsewhere, so it carries the same gate as its callers.
    #[cfg(target_os = "linux")]
    fn wait_until(cond: impl Fn() -> bool, budget: Duration) {
        let deadline = Instant::now() + budget;
        while Instant::now() < deadline {
            if cond() {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// Wait until `counter` stops moving, then return its settled value.
    ///
    /// Waiting for "some event arrived" is not enough before taking a baseline,
    /// and getting that wrong is what made this suite fail in CI: the first
    /// event of a 40-file burst satisfies `> 0` while the other 39 are still in
    /// flight, so the stragglers land after the baseline is taken and are
    /// attributed to whatever is measured next. That reported 59 events from a
    /// directory the OS was not watching at all.
    #[cfg(target_os = "linux")]
    fn settled(counter: impl Fn() -> u64, budget: Duration) -> u64 {
        let deadline = Instant::now() + budget;
        let mut last = counter();
        let mut stable = 0;
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            let now = counter();
            // Three consecutive quiet samples: one is not enough, because inotify
            // delivers in bursts with gaps between them.
            stable = if now == last { stable + 1 } else { 0 };
            last = now;
            if stable >= 3 {
                break;
            }
        }
        last
    }

    /// #801: indexing a file must not schedule indexing it again.
    ///
    /// The second cycle with the same cause as
    /// `reading_an_ignore_file_does_not_feed_the_watcher`, and the one that
    /// keeps running after every ordinary edit. `handle_watch_event` reindexes
    /// on `Upsert`, `reindex_files` reads the file, inotify reports that read as
    /// `Access(Open)`, the staleness guard passes it because the file really is
    /// newer than daemon start and stays that way, and `Access(_)` falls into
    /// the `_` arm of the kind match, which means `Upsert`. One edit, then a
    /// reindex every `DEBOUNCE_MS` until the daemon restarts, once per file
    /// edited in the session.
    ///
    /// Observed over a fixed window rather than with `settled`, deliberately.
    /// The broken behaviour ticks about once a second (the flush thread runs
    /// every `DEBOUNCE_MS` and each re-entry sets a fresh `DEBOUNCE_MS`
    /// deadline), so three quiet 50 ms samples land in the gap between two ticks
    /// and `settled` would report a stopped counter on code that has not
    /// stopped. Only a window several ticks wide separates "indexed once" from
    /// "indexing forever".
    #[cfg(target_os = "linux")]
    #[test]
    fn reindexing_a_file_does_not_schedule_itself_again() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let file = root.join("app.ts");
        std::fs::write(&file, "export const x = 1;").unwrap();

        let (tx, mut rx) = mpsc::channel::<WatchEvent>(1024);
        let handle = spawn(root, tx, Instant::now()).expect("watcher spawns");

        // Stand in for `handle_watch_event`: the only part that matters here is
        // that indexing READS the file, which is what closes the cycle.
        let reindexes = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&reindexes);
        std::thread::spawn(move || {
            while let Some(ev) = rx.blocking_recv() {
                if let WatchEvent::Upsert(path) = ev {
                    let _ = std::fs::read_to_string(&path);
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        // Edit only after `start_time` is unambiguously in the past. The
        // staleness guard compares the file's mtime against it, and a filesystem
        // with coarse mtime granularity can stamp a write made microseconds
        // after `spawn` as predating it, at which point the edit is dropped and
        // the control arm below fails for a reason that has nothing to do with
        // what this test is about. The other watch tests never see it because
        // they assert on `raw_events`, which is counted before that guard.
        std::thread::sleep(Duration::from_millis(1_100));
        std::fs::write(&file, "export const x = 2;").unwrap();

        // Control arm: the edit must reach the indexer at all, or the assertion
        // below passes on a watcher that delivers nothing. Budgeted generously
        // because it crosses a 500 ms debounce and a 500 ms flush tick before it
        // can arrive, and a loaded 2 core CI box stretches both.
        wait_until(
            || reindexes.load(Ordering::Relaxed) > 0,
            Duration::from_secs(30),
        );
        assert!(
            reindexes.load(Ordering::Relaxed) > 0,
            "one edit must produce one reindex, or this test proves nothing"
        );

        // Let the edit's own events drain before taking the baseline. A single
        // write arrives as `Modify(Data)` and again as `Access(Close(Write))`,
        // which coalesce into one reindex when they share a debounce window and
        // into two when they do not, so the absolute count is not the thing to
        // assert on. The growth after everything has drained is.
        //
        // `settled` rather than a fixed sleep: on a slow box the second of those
        // two events can land after any wall clock deadline this test picks, and
        // it would be counted as growth. It cannot mask the failure, because a
        // running cycle ticks about once a second and the window below is six.
        let after_edit = settled(
            || reindexes.load(Ordering::Relaxed),
            Duration::from_secs(15),
        );

        // Nothing is touched in this window. The cycle ticks about once a second
        // when it is running, so anything above zero here is the file
        // rescheduling itself, and it is load independent: a slower machine
        // changes the rate, not the fact that it never stops.
        std::thread::sleep(Duration::from_secs(6));
        let growth = reindexes.load(Ordering::Relaxed) - after_edit;
        assert_eq!(
            growth, 0,
            "{growth} further reindexes with nothing touched: indexing the file \
             reads it, and the read comes back as another Upsert, so every edited \
             file is reindexed once per debounce interval until the daemon \
             restarts (#801)."
        );

        drop(handle);
    }

    /// #801: reading a watched file must not count as changing it.
    ///
    /// The failure this pins is a cycle, not a cost. `notify`'s inotify backend
    /// subscribes to `IN_OPEN` (8.2.0 `inotify.rs:427`), so a read produces an
    /// event; `build_ignore_matcher` reads every `.gitignore` in the repo; and
    /// the rebuild is triggered by an event naming an ignore file. Read,
    /// rebuild, read. One `cat .gitignore` by an unrelated process, with nothing
    /// ever written, held the watcher at ~150k events/sec indefinitely.
    ///
    /// Asserting on `raw_events` rather than on reindexes for the same reason
    /// the tests above do: the userspace filter discards these events either
    /// way, so every downstream observation is identical whether the cycle is
    /// running or not.
    ///
    /// Linux only, like the other watch tests here: fsevent, windows and kqueue
    /// never construct an `Access` event, so there is nothing to observe there.
    #[cfg(target_os = "linux")]
    #[test]
    fn reading_an_ignore_file_does_not_feed_the_watcher() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(root.join("app.ts"), "export const x = 1;").unwrap();

        let (tx, _rx) = mpsc::channel::<WatchEvent>(1024);
        let handle = spawn(root, tx, Instant::now()).expect("watcher spawns");

        // Control arm, as above: a real write must move the counter, or the
        // assertion below passes on a watcher that sees nothing at all.
        std::fs::write(root.join("app.ts"), "export const x = 2;").unwrap();
        wait_until(|| handle.raw_events() > 0, Duration::from_secs(5));
        let after_write = settled(|| handle.raw_events(), Duration::from_secs(10));
        assert!(
            after_write > 0,
            "a write inside the repo must produce raw events, or this test proves nothing"
        );

        // A separate process reads the file once and writes nothing, the way
        // `git status`, ripgrep or an editor does.
        let status = std::process::Command::new("cat")
            .arg(root.join(".gitignore"))
            .stdout(std::process::Stdio::null())
            .status()
            .expect("cat runs");
        assert!(status.success(), "cat .gitignore failed");

        let from_read = settled(|| handle.raw_events(), Duration::from_secs(10)) - after_write;
        assert!(
            from_read < 100,
            "one read of .gitignore produced {from_read} raw events: the read is \
             being treated as a change, so rebuilding the matcher reads the file \
             again and the watcher feeds itself (#801)."
        );
    }

    #[test]
    fn ignore_matcher_honors_travsrignore_and_skip_dirs() {
        // #403: the incremental matcher must read .travsrignore (not just
        // .gitignore) so files init excluded stay excluded on edit, and must
        // still hard-skip SKIP_DIRS.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".travsrignore"), "vendor/\n**/generated/\n").unwrap();
        std::fs::create_dir_all(root.join("vendor")).unwrap();
        std::fs::create_dir_all(root.join("src/generated")).unwrap();

        let m = build_ignore_matcher(root);

        assert!(
            should_skip_all(&root.join("vendor/lib.ts"), root, &m),
            "vendor/ excluded by .travsrignore must be skipped"
        );
        assert!(
            should_skip_all(&root.join("src/generated/api.ts"), root, &m),
            "**/generated/ excluded by .travsrignore must be skipped"
        );
        assert!(
            !should_skip_all(&root.join("src/app.ts"), root, &m),
            "a normal source file must not be skipped"
        );
        assert!(
            should_skip_all(&root.join("node_modules/x/index.js"), root, &m),
            "node_modules is always skipped (SKIP_DIRS)"
        );
    }

    #[test]
    fn travsrignore_takes_precedence_over_gitignore() {
        // .travsrignore is added after .gitignore, so its rules win: a negation
        // there re-includes a file .gitignore excluded (no parent dir excluded,
        // so gitignore re-inclusion semantics allow it).
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".gitignore"), "secret.ts\n").unwrap();
        std::fs::write(root.join(".travsrignore"), "!secret.ts\n").unwrap();

        let m = build_ignore_matcher(root);
        assert!(
            !should_skip_all(&root.join("secret.ts"), root, &m),
            ".travsrignore negation must override the .gitignore exclusion"
        );
    }

    #[test]
    fn travsrignore_reincludes_gitignored_directory() {
        // #599: `.gitignore` excludes a whole subtree that `.travsrignore`
        // re-includes with `!vendored/`. init indexes it, so the watcher must
        // NOT skip edits to files under it — otherwise those files go
        // permanently stale (the state frozen at the last full init).
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".gitignore"), "vendored/\n").unwrap();
        std::fs::write(root.join(".travsrignore"), "!vendored/\n").unwrap();
        std::fs::create_dir_all(root.join("vendored")).unwrap();

        let m = build_ignore_matcher(root);
        assert!(
            !should_skip_all(&root.join("vendored/dep.ts"), root, &m),
            "#599: .travsrignore `!vendored/` must re-include a gitignored dir so \
             the watcher keeps it fresh"
        );
    }

    /// #448: `travsr-mcp` keeps its own copy of this list so `find_pattern`
    /// searches the same file universe the walker builds the graph from. The
    /// crate dependency edge runs daemon -> mcp, so this side is the only one
    /// that can assert the two never drift.
    ///
    /// If this fails, a directory was added to or removed from one list only.
    /// Update `SKIP_DIRS` in `travsr-mcp/src/tools.rs` to match, or
    /// `find_pattern` will report matches from files the graph does not
    /// contain (or miss files it does).
    #[test]
    fn skip_dirs_matches_the_mcp_copy() {
        assert_eq!(
            SKIP_DIRS,
            travsr_mcp::SKIP_DIRS,
            "travsr-daemon and travsr-mcp SKIP_DIRS have drifted apart"
        );
    }
}
