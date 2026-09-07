use std::path::{Path, PathBuf};

use travsr_store::registry;

/// How a linked worktree's gitlink `.git` file is resolved. This is the only
/// behavioral difference between the read and write repo-root resolvers, split
/// out so the two intents cannot silently drift back together (issues #302/#586).
#[derive(Clone, Copy)]
enum WorktreeMode {
    /// Read path: prefer the worktree's own `.travsr/` if it has one, otherwise
    /// redirect to the main worktree so read commands find the parent repo's
    /// index instead of erroring `not initialized` (issue #302).
    PreferLocalElseMain,
    /// Write path: always stay in the linked worktree so `init`/`fsck`/`embed`/
    /// `daemon`/`hook-run` create and mutate the worktree's own `.travsr/`,
    /// never the main worktree's — silently writing a different checkout's index
    /// is the bug (issue #586).
    StayInWorktree,
}

/// Resolve the Travsr repo root for a **read** command (`ask`, `graph`,
/// `references`, `pattern`, `status`, `explain`, `config`, MCP `serve`, …) — the
/// directory whose `.travsr/` holds the index to query.
///
/// Resolution order (issue #302 — repo resolution must be git-aware):
/// 1. Walk up to the nearest `.git` entry. For a standard repo (`.git` is a
///    directory) that directory is the root.
/// 2. For a **linked worktree** (`.git` is a gitlink *file*): if the worktree
///    has its own `.travsr/graph.db` (created by a write command per issue #586)
///    use it; otherwise resolve the main worktree via
///    `git rev-parse --git-common-dir` so reads find the parent repo's index
///    instead of erroring `not initialized` (issue #302).
/// 3. If no `.git` is found at all, fall back to the global registry: if `start`
///    lives under a registered repo root, return that root.
pub fn find_git_root(start: &Path) -> anyhow::Result<PathBuf> {
    resolve_repo_root(start, WorktreeMode::PreferLocalElseMain)
}

/// Resolve the repo root for a **write** command (`init`, `fsck`, `embed`
/// reindex/switch/init/reconfigure/gc/calibrate, `daemon`, `hook-run`).
///
/// Identical to [`find_git_root`] except a linked worktree always resolves to
/// the worktree's **own** directory, never the main worktree. Write commands
/// mutate the resolved `.travsr/`; redirecting them to the main worktree
/// silently mutates a different checkout at a different commit (issue #586).
pub fn find_git_root_for_write(start: &Path) -> anyhow::Result<PathBuf> {
    resolve_repo_root(start, WorktreeMode::StayInWorktree)
}

fn resolve_repo_root(start: &Path, mode: WorktreeMode) -> anyhow::Result<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        let git_entry = current.join(".git");
        if git_entry.is_dir() {
            return Ok(current);
        }
        if git_entry.is_file() {
            // Linked worktree: `.git` is a gitlink file pointing at the main
            // repo's git dir.
            return Ok(match mode {
                WorktreeMode::PreferLocalElseMain => {
                    if current.join(".travsr/graph.db").is_file() {
                        current
                    } else {
                        main_worktree_root(&current).unwrap_or(current)
                    }
                }
                WorktreeMode::StayInWorktree => current,
            });
        }
        if !current.pop() {
            // Not inside a git repository — last resort, consult the registry
            // in case `start` sits under a repo that was initialized elsewhere.
            if let Some(root) = registered_ancestor(start) {
                return Ok(root);
            }
            anyhow::bail!("not inside a git repository; run `git init` first, then `travsr init`");
        }
    }
}

/// Resolve the main-worktree root from inside a linked worktree.
///
/// Returns `None` (caller falls back to the worktree dir) when git is
/// unavailable, too old for `--path-format`, or the common dir does not point
/// at a real main worktree (e.g. a submodule gitlink under `.git/modules`).
fn main_worktree_root(worktree_dir: &Path) -> Option<PathBuf> {
    // Bounded (#717 triage): this runs while resolving the repo root, before
    // anything is printed, so a git call that never returns here is a CLI that
    // hangs with no output at all. `None` already means "fall back to the
    // worktree dir", which is the right answer for a query that did not land.
    // `worktree_dir` is passed as the working directory, not as a `-C <string>`
    // argument: a path with bytes that are not valid UTF-8 is legal, and
    // converting it to a string first would mangle it into U+FFFD. Git would then
    // fail to resolve a worktree that exists, this would return `None`, and
    // `resolve_repo_root` would fall back to treating the linked worktree as the
    // repo root, which has no index. The user sees "not initialized". That is the
    // #302 regression this function exists to prevent.
    //
    // The answer is a path, so it is decoded as one rather than as text. A
    // lossy decode here would corrupt a common dir that contains non-UTF-8
    // bytes into U+FFFD, `root.join(".git").is_dir()` would then miss against
    // the real filesystem, and this would return `None` for exactly the same
    // "not initialized" outcome by a different route.
    let common = crate::git_bounded::git_path_bounded(
        Some(worktree_dir),
        ["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?;
    // `--git-common-dir` is `<main>/.git`; its parent is the main worktree.
    let root = common.parent()?.to_path_buf();
    // Guard against submodule gitlinks (`.git/modules/<name>`): only accept a
    // directory that actually looks like a main worktree.
    if root.join(".git").is_dir() {
        Some(root)
    } else {
        None
    }
}

/// Whether two resolved repo roots name the **same directory**.
///
/// Deliberately not `a == b`. The two sides reach this comparison by different
/// routes and are not spelled alike even when they are one directory: the
/// caller's own root is walked up from `cwd` and is therefore native, while a
/// served root is derived from `git rev-parse --git-common-dir`, which on
/// Windows renders forward slashes (`C:/repo/.git`) where the native path uses
/// backslashes (`C:\repo\.git`). Windows and the default macOS filesystem are
/// also case-insensitive, so `C:\Repo` and `c:\repo` are the same directory; a
/// path may arrive as an 8.3 short name or as a UNC share
/// (`\\server\share\repo`); and on macOS `/tmp` is a symlink to `/private/tmp`.
/// A raw string compare reports "different tree" for every one of those.
///
/// `canonicalize` is the operating system's own answer to "which directory is
/// this": a verbatim `\\?\`-prefixed path on Windows (long-path, case- and
/// short-name-resolved, UNC included) and a fully resolved absolute path
/// elsewhere. Only used for the comparison, never for display, so the user
/// still sees the paths they typed.
///
/// A path that cannot be canonicalized (it does not exist, or is unreadable)
/// counts as the **same** directory. The one caller uses this to decide whether
/// to tell a user their answer came from another checkout, and that claim must
/// never rest on a path this process could not resolve.
fn same_directory(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => true,
    }
}

/// `Some(worktree_root)` when `cwd` stands in a **linked worktree** whose read
/// queries are answered out of `served_root`, a different checkout. That is the
/// `PreferLocalElseMain` redirect at the top of this file, seen from the far
/// end: the answer is correct about the main worktree and wrong about the tree
/// the user is editing.
///
/// `None` for a standard repo, for a linked worktree that has its own
/// `.travsr/` (no redirect happened), and whenever the two roots cannot be told
/// apart.
///
/// Gated on `.git` being a gitlink *file*, which is true only inside a linked
/// worktree. That keeps the signal scoped to the redirect it describes: an
/// explicit `--db <other repo>` or a registry-resolved root also serves a
/// different directory, but by the user's own instruction, and must not be
/// narrated as a worktree mix-up.
fn worktree_served_elsewhere(cwd: &Path, served_root: &Path) -> Option<PathBuf> {
    let own = find_git_root_for_write(cwd).ok()?;
    if !own.join(".git").is_file() {
        return None;
    }
    (!same_directory(&own, served_root)).then_some(own)
}

/// State plainly that this answer describes another checkout.
///
/// The condition this replaces was reported as index staleness ("semantic
/// analysis has not caught up with the current commit"), which describes a race
/// inside one tree and is repaired by waiting or re-indexing. Neither helps
/// here: the served index is a faithful description of a *different* checkout,
/// and no amount of re-indexing it will make it describe this worktree. Naming
/// both roots is the whole point, so a reader can see which tree the
/// `path:line` they are about to trust belongs to.
///
/// `served_degraded` folds in the one true half of the freshness note this
/// replaces: when the served index is itself Phase B degraded, an empty or
/// short result from it is not authoritative, and a #302 user who decides the
/// main index is what they wanted and keeps querying it needs to know that. The
/// "wait / re-index / run `travsr status`" advice from the freshness note is
/// dropped, because it reads as "this will catch up to your tree", which it
/// never will.
pub(crate) fn cross_checkout_note(
    here: &Path,
    served: &Path,
    served_commit: Option<&str>,
    served_degraded: bool,
) -> String {
    let commit = match served_commit.map(str::trim).filter(|c| !c.is_empty()) {
        Some(c) => format!(" (indexed at commit {c})"),
        None => String::new(),
    };
    let degraded = if served_degraded {
        " That index's own call graph is also incomplete right now, so an empty \
         or short result from it is not authoritative either."
    } else {
        ""
    };
    format!(
        "[note: you are standing in the linked worktree {}, but travsr is \
         answering from the index at {}{}, which is a different checkout. \
         Everything it reports, including any path, line number or symbol, \
         describes that tree, not this one. This is not index staleness: \
         re-indexing that checkout will never make it describe this worktree. \
         Run `travsr init` here to give this worktree its own index.{} \
         (set TRAVSR_NO_WORKTREE_NOTE=1 to silence this note)]",
        here.display(),
        served.display(),
        commit,
        degraded
    )
}

/// Production entry: `Some(worktree_root)` when the index at `db_path`
/// (`<root>/.travsr/graph.db`) belongs to a checkout other than the linked
/// worktree at `cwd`.
///
/// Filesystem only, so a caller can classify before deciding whether it is
/// worth opening a store. `cwd` is a parameter rather than the process's own
/// working directory so this stays a pure predicate a caller can unit-test, and
/// so it compares the tree the caller actually resolved its root from rather
/// than whatever the process cwd happens to be.
pub(crate) fn served_by_other_checkout(cwd: &Path, db_path: &Path) -> Option<PathBuf> {
    // `<root>/.travsr/graph.db` -> `<root>`.
    let served = db_path.parent()?.parent()?;
    worktree_served_elsewhere(cwd, served)
}

/// The note for [`served_by_other_checkout`], or `None` when the answer really
/// does describe the tree the caller is standing in.
///
/// Classification only: `TRAVSR_NO_WORKTREE_NOTE` is deliberately **not** read
/// here. `Some` means "this answer is about another checkout", which is also
/// what tells a caller to drop its own hedged drift note — a claim that stays
/// true when the user has merely asked not to read about it. Collapsing the two
/// into one `None` would mean muting this note handed the user back the two
/// misleading notes it replaces. The caller mutes the *print* via
/// [`worktree_note_suppressed`]; see `travsr status`.
///
/// `served_commit` is the index's `last_commit` when the caller already has a
/// store open, otherwise `None`; this resolver deliberately does not open one,
/// so it passes `served_degraded = false`. The one caller is `travsr status`,
/// whose stdout is counts and a commit, never a `path:line`, so the degraded
/// caveat about empty results would not apply to anything it prints.
pub(crate) fn cross_checkout_note_for_db(
    cwd: &Path,
    db_path: &Path,
    served_commit: Option<&str>,
) -> Option<String> {
    let served = db_path.parent()?.parent()?;
    let here = served_by_other_checkout(cwd, db_path)?;
    Some(cross_checkout_note(&here, served, served_commit, false))
}

/// Escape hatch (`TRAVSR_NO_WORKTREE_NOTE`): serving a linked worktree from the
/// main index is the intended #302 behavior, and an agent setup that spawns
/// worktrees under `.claude/worktrees/` sees the note on every read call. Read
/// at the emit entry points, never in [`served_by_other_checkout`] or
/// [`cross_checkout_note_for_db`], so the classification a caller keys its own
/// note suppression to stays a pure predicate. Follows the crate's existing
/// `TRAVSR_NO_RERANK` / `TRAVSR_ABSTAIN_GUESSES` convention.
///
/// Silences the print only. It never turns a cross-checkout answer back into an
/// ordinary one: the freshness and drift notes stay suppressed, because they
/// are wrong here whether or not the user wants to read about it. The note
/// names this variable so the people it is aimed at can find it.
pub(crate) fn worktree_note_suppressed() -> bool {
    std::env::var_os("TRAVSR_NO_WORKTREE_NOTE").is_some()
}

/// Find the deepest registered repo root that is an ancestor of (or equal to)
/// `start`. Registry values are `<root>/.travsr/graph.db` paths.
fn registered_ancestor(start: &Path) -> Option<PathBuf> {
    let repos = registry::all_repos().ok()?;
    let mut best: Option<PathBuf> = None;
    for db_path in repos.values() {
        // Strip `graph.db` then `.travsr` to recover the repo root.
        let Some(root) = db_path.parent().and_then(|p| p.parent()) else {
            continue;
        };
        if start.starts_with(root) {
            // Prefer the deepest (most specific) matching root.
            let deeper = best
                .as_ref()
                .map(|b| root.as_os_str().len() > b.as_os_str().len())
                .unwrap_or(true);
            if deeper {
                best = Some(root.to_path_buf());
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn standard_repo_resolves_to_repo_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        let root = find_git_root(tmp.path()).unwrap();
        assert_eq!(
            root.canonicalize().unwrap(),
            tmp.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn subdir_walks_up_to_repo_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        let sub = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&sub).unwrap();
        let root = find_git_root(&sub).unwrap();
        assert_eq!(
            root.canonicalize().unwrap(),
            tmp.path().canonicalize().unwrap()
        );
    }

    /// Issue #302 acceptance: a linked worktree must resolve to the main
    /// worktree (whose `.travsr/` holds the index), not the worktree dir.
    #[test]
    fn worktree_resolves_to_main_worktree() {
        if !git_available() {
            eprintln!("skipping: git not available");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("main");
        std::fs::create_dir(&main).unwrap();
        assert!(git(&main, &["init", "-q"]));
        assert!(git(&main, &["config", "user.email", "t@t.t"]));
        assert!(git(&main, &["config", "user.name", "t"]));
        std::fs::write(main.join("f.txt"), "x").unwrap();
        assert!(git(&main, &["add", "."]));
        assert!(git(&main, &["commit", "-qm", "init"]));

        let wt = tmp.path().join("wt");
        assert!(git(&main, &["worktree", "add", "-q", wt.to_str().unwrap()]));
        // `.git` in a worktree is a gitlink file, not a directory.
        assert!(wt.join(".git").is_file());

        let resolved = find_git_root(&wt).unwrap();
        assert_eq!(
            resolved.canonicalize().unwrap(),
            main.canonicalize().unwrap(),
            "worktree must resolve to the main repo root"
        );

        // Also from a subdirectory of the worktree.
        let wt_sub = wt.join("sub");
        std::fs::create_dir(&wt_sub).unwrap();
        let resolved_sub = find_git_root(&wt_sub).unwrap();
        assert_eq!(
            resolved_sub.canonicalize().unwrap(),
            main.canonicalize().unwrap()
        );
    }

    /// Build `<tmp>/main` (a committed repo) and a linked worktree `<tmp>/wt`.
    /// Returns `(tmpdir, main, wt)`. Skips (returns `None`) when git is absent.
    fn main_and_worktree() -> Option<(tempfile::TempDir, PathBuf, PathBuf)> {
        if !git_available() {
            eprintln!("skipping: git not available");
            return None;
        }
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("main");
        std::fs::create_dir(&main).unwrap();
        assert!(git(&main, &["init", "-q"]));
        assert!(git(&main, &["config", "user.email", "t@t.t"]));
        assert!(git(&main, &["config", "user.name", "t"]));
        std::fs::write(main.join("f.txt"), "x").unwrap();
        assert!(git(&main, &["add", "."]));
        assert!(git(&main, &["commit", "-qm", "init"]));
        let wt = tmp.path().join("wt");
        assert!(git(&main, &["worktree", "add", "-q", wt.to_str().unwrap()]));
        assert!(wt.join(".git").is_file());
        Some((tmp, main, wt))
    }

    /// Issue #586 acceptance: the **write** resolver must stay in the worktree,
    /// never redirect to the main worktree, so `init`/`fsck`/`embed` create and
    /// mutate the worktree's own `.travsr/`. Companion to
    /// [`worktree_resolves_to_main_worktree`] — the read and write intents must
    /// not silently collapse back into one behavior.
    #[test]
    fn worktree_write_resolver_stays_in_worktree() {
        let Some((_tmp, main, wt)) = main_and_worktree() else {
            return;
        };

        // Read resolver (no local index yet) redirects to main — issue #302.
        let read = find_git_root(&wt).unwrap();
        assert_eq!(read.canonicalize().unwrap(), main.canonicalize().unwrap());

        // Write resolver must resolve to the worktree itself — issue #586.
        let write = find_git_root_for_write(&wt).unwrap();
        assert_eq!(
            write.canonicalize().unwrap(),
            wt.canonicalize().unwrap(),
            "write resolver must not redirect a worktree to the main repo root"
        );

        // And from a subdirectory of the worktree.
        let wt_sub = wt.join("sub");
        std::fs::create_dir(&wt_sub).unwrap();
        let write_sub = find_git_root_for_write(&wt_sub).unwrap();
        assert_eq!(
            write_sub.canonicalize().unwrap(),
            wt.canonicalize().unwrap()
        );
    }

    /// `same_directory` must answer about directories, not spellings: a path
    /// that walks through `sub/..` is the same directory as its parent, and two
    /// genuinely distinct directories are not. The `..` case is the portable
    /// stand-in for the platform spellings this guards against on Windows
    /// (forward slashes out of `git rev-parse`, case differences, 8.3 short
    /// names, UNC shares), which cannot be constructed on a POSIX test runner.
    #[test]
    fn same_directory_compares_directories_not_strings() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        assert!(same_directory(tmp.path(), &sub.join("..")));

        let other = tmp.path().join("other");
        std::fs::create_dir(&other).unwrap();
        assert!(!same_directory(&sub, &other));
    }

    /// Fail-safe: an unresolvable path counts as the same directory, so a
    /// cross-checkout claim is never made on the strength of a path this
    /// process could not resolve.
    #[test]
    fn same_directory_treats_unresolvable_paths_as_same() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(same_directory(
            tmp.path(),
            &tmp.path().join("does-not-exist")
        ));
    }

    /// The core acceptance, built without invoking git: a linked worktree's
    /// `.git` is a *file* holding `gitdir: <path>`, and that shape alone is what
    /// distinguishes the redirect from a normal repo. Constructing it by hand
    /// keeps the classification independent of a git binary and of the
    /// platform's worktree bookkeeping.
    #[test]
    fn hand_built_gitlink_served_by_another_root_is_cross_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("main");
        std::fs::create_dir_all(main.join(".git/worktrees/wt")).unwrap();

        let wt = tmp.path().join("wt");
        std::fs::create_dir(&wt).unwrap();
        // The pointer file git writes into a linked worktree. Its payload is a
        // platform-native absolute path, so it is built from the real `main`
        // path rather than a hand-spelled string.
        std::fs::write(
            wt.join(".git"),
            format!("gitdir: {}\n", main.join(".git/worktrees/wt").display()),
        )
        .unwrap();

        let here = worktree_served_elsewhere(&wt, &main).expect("must flag the redirect");
        assert!(
            same_directory(&here, &wt),
            "must report the worktree it is standing in"
        );

        let note = cross_checkout_note(&here, &main, Some("abc1234"), false);
        assert!(note.contains(&wt.display().to_string()), "got: {note}");
        assert!(note.contains(&main.display().to_string()), "got: {note}");
        assert!(note.contains("abc1234"), "got: {note}");
        assert!(
            note.contains("not index staleness"),
            "the note must not be mistaken for the staleness warning it replaces: {note}"
        );
        // The hatch is reachable only from the note: it is in no `--help` and no
        // doc, so the people seeing this on every read call have nowhere else to
        // find it.
        assert!(note.contains("TRAVSR_NO_WORKTREE_NOTE=1"), "got: {note}");
    }

    /// A linked worktree answered from its own root is not a cross-checkout
    /// answer, even though `.git` is still a gitlink file. Guards the
    /// `PreferLocalElseMain` case where the worktree was `travsr init`-ed in
    /// place (issue #586) against a spurious note.
    #[test]
    fn gitlink_served_by_its_own_root_is_not_cross_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let wt = tmp.path().join("wt");
        std::fs::create_dir(&wt).unwrap();
        std::fs::write(wt.join(".git"), "gitdir: /nonexistent/.git/worktrees/wt\n").unwrap();
        assert!(worktree_served_elsewhere(&wt, &wt).is_none());
    }

    /// A standard repo (`.git` is a directory) never produces the note, even
    /// when it is served from somewhere else entirely (an explicit `--db`, or a
    /// registry-resolved root). Those are the user's own instruction, not the
    /// worktree redirect this note describes.
    #[test]
    fn standard_repo_is_never_reported_as_cross_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir(&elsewhere).unwrap();
        assert!(worktree_served_elsewhere(&repo, &elsewhere).is_none());
    }

    /// The commit clause is omitted entirely when the index's commit is
    /// unknown or blank, rather than rendered as an empty parenthetical.
    #[test]
    fn cross_checkout_note_omits_an_unknown_commit() {
        let here = Path::new("/here");
        let served = Path::new("/there");
        assert!(!cross_checkout_note(here, served, None, false).contains("indexed at commit"));
        assert!(!cross_checkout_note(here, served, Some("  "), false).contains("indexed at commit"));
    }

    /// Serializes the tests that mutate `TRAVSR_NO_WORKTREE_NOTE`, since env is
    /// process-global and the suite runs in parallel.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Holds [`ENV_LOCK`] and restores `TRAVSR_NO_WORKTREE_NOTE` on drop.
    ///
    /// Both halves matter only on a red run, which is the run whose output has
    /// to be readable. Without the restore, a failing assert leaves the var set
    /// for every later test in the process; without `into_inner`, the panic that
    /// caused it also poisons the mutex, so the next test to take the lock dies
    /// on the poison and reports as a second, unrelated failure that buries the
    /// first. There is no state behind this lock to be corrupted, so ignoring
    /// the poison is sound.
    struct EnvGuard {
        // Held for its `Drop`, never read.
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn acquire() -> Self {
            Self {
                _lock: ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("TRAVSR_NO_WORKTREE_NOTE");
        }
    }

    /// End-to-end over the production predicate against a **real** `git
    /// worktree` (not a hand-built pointer file): exercises both the gitlink
    /// shape git actually writes and the `<root>/.travsr/graph.db ->
    /// parent().parent()` derivation that only the public entry does. `cwd` is a
    /// parameter, so this needs neither a chdir nor a git-served cwd.
    #[test]
    fn served_by_other_checkout_flags_a_real_worktree() {
        let Some((_tmp, main, wt)) = main_and_worktree() else {
            return;
        };
        let db = main.join(".travsr").join("graph.db");

        // Standing in the worktree, served by the main index -> cross-checkout.
        let here = served_by_other_checkout(&wt, &db).expect("worktree served by main");
        assert!(same_directory(&here, &wt));

        // From a subdirectory of the worktree, same answer.
        let wt_sub = wt.join("sub");
        std::fs::create_dir(&wt_sub).unwrap();
        assert!(served_by_other_checkout(&wt_sub, &db).is_some());

        // Standing in the main checkout itself, no redirect -> no note.
        assert!(served_by_other_checkout(&main, &db).is_none());
    }

    /// `cross_checkout_note_for_db` (the `travsr status` entry) names both roots
    /// for a real worktree and stays silent in the main checkout. Takes no env
    /// lock: this entry is pure classification and no longer reads
    /// `TRAVSR_NO_WORKTREE_NOTE`.
    #[test]
    fn cross_checkout_note_for_db_names_the_worktree() {
        let Some((_tmp, main, wt)) = main_and_worktree() else {
            return;
        };
        let db = main.join(".travsr").join("graph.db");

        let note = cross_checkout_note_for_db(&wt, &db, Some("abc1234"))
            .expect("worktree served by main must produce a note");
        assert!(note.contains(&wt.display().to_string()), "got: {note}");
        assert!(note.contains(&main.display().to_string()), "got: {note}");

        assert!(cross_checkout_note_for_db(&main, &db, Some("abc1234")).is_none());
    }

    /// The degraded caveat (finding #1) rides inside the single note: the
    /// `false` variant never claims empty results are unauthoritative, the
    /// `true` variant does, and neither carries the "wait / run `travsr status`"
    /// advice from the freshness note it replaces.
    #[test]
    fn cross_checkout_note_folds_in_a_degraded_served_index() {
        let here = Path::new("/wt");
        let served = Path::new("/main");
        let clean = cross_checkout_note(here, served, Some("abc1234"), false);
        assert!(!clean.contains("not authoritative"), "got: {clean}");

        let degraded = cross_checkout_note(here, served, Some("abc1234"), true);
        assert!(degraded.contains("not authoritative"), "got: {degraded}");
        assert!(degraded.contains("also incomplete"), "got: {degraded}");
        assert!(
            !degraded.contains("travsr status") && !degraded.contains("has not caught up"),
            "the degraded caveat must not resurrect the freshness advice: {degraded}"
        );
    }

    /// `TRAVSR_NO_WORKTREE_NOTE` silences the *print*, and must not also erase
    /// the *classification*.
    ///
    /// The two are what a caller keys two different decisions to: whether to
    /// print this note, and whether the freshness and drift notes it replaces
    /// still apply. Folding the hatch into the classifier made muting one
    /// accurate note hand the user back two misleading ones, so this pins that
    /// `cross_checkout_note_for_db` keeps answering "yes, another checkout"
    /// regardless of the variable, and that the variable is visible on its own
    /// through [`worktree_note_suppressed`] for the emitter to act on.
    #[test]
    fn worktree_note_hatch_is_separate_from_the_classification() {
        let _guard = EnvGuard::acquire();
        let Some((_tmp, main, wt)) = main_and_worktree() else {
            return;
        };
        let db = main.join(".travsr").join("graph.db");

        std::env::remove_var("TRAVSR_NO_WORKTREE_NOTE");
        assert!(!worktree_note_suppressed());
        assert!(cross_checkout_note_for_db(&wt, &db, Some("abc1234")).is_some());

        std::env::set_var("TRAVSR_NO_WORKTREE_NOTE", "1");
        assert!(worktree_note_suppressed());
        assert!(
            cross_checkout_note_for_db(&wt, &db, Some("abc1234")).is_some(),
            "the hatch must silence the note, not reclassify the answer as \
             belonging to this tree"
        );
    }

    /// Once a worktree has its own index (a write command created
    /// `<wt>/.travsr/graph.db`), the **read** resolver must prefer it over the
    /// main worktree — otherwise the read/write split would leave reads pointed
    /// at a different checkout than the one `init` just indexed (issue #586).
    #[test]
    fn worktree_read_resolver_prefers_local_index() {
        let Some((_tmp, _main, wt)) = main_and_worktree() else {
            return;
        };

        // Simulate the worktree having been `travsr init`-ed under the fix.
        std::fs::create_dir_all(wt.join(".travsr")).unwrap();
        std::fs::write(wt.join(".travsr/graph.db"), b"").unwrap();

        let read = find_git_root(&wt).unwrap();
        assert_eq!(
            read.canonicalize().unwrap(),
            wt.canonicalize().unwrap(),
            "read resolver must use the worktree's own index once it has one"
        );
    }
}
