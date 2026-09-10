//! `travsr status` — index and graph health summary.
//!
//! Data acquisition is shared with the daemon via `travsr_mcp::query`
//! (#318 O1): a running daemon answers from its warm store; otherwise the
//! store is opened directly (read-only fast path).

use anyhow::Context as _;
use travsr_mcp::query::{self, StatusPayload};

use crate::daemon_client;
use crate::repo::find_git_root;

/// M7: compare `last_commit` vs `phase_b_commit` to describe Phase B freshness.
///
/// #583: equal markers are not sufficient evidence of freshness. A watcher
/// reindex rewrites a file's Phase A nodes and drops that file's `ref/call`
/// edges without moving HEAD, so both markers still agree while `get_callers`
/// RFC-027 section 12: render the cumulative live-lane precision reading.
///
/// Returns `None` when nothing has been measured, so the line never appears on a
/// repo that has not exercised the lane.
///
/// Coverage is always shown beside precision. A precision figure alone invites
/// the reading "1.00 means it is perfect", when it may mean "two of four hundred
/// claims were checkable and both happened to be right". The gate is on
/// precision; coverage is what tells you whether the gate saw anything.
fn live_precision_line(store: &travsr_store::SqliteStore) -> Option<String> {
    let raw = store.get_meta("live_precision").ok().flatten()?;
    let parts: Vec<u64> = raw
        .split(',')
        .filter_map(|p| p.trim().parse::<u64>().ok())
        .collect();
    let [agree, disagree, unverifiable] = parts.as_slice() else {
        return None;
    };
    let total = agree + disagree + unverifiable;
    if total == 0 {
        return None;
    }
    let verified = agree + disagree;
    let precision = if verified > 0 {
        format!("{:.4}", *agree as f64 / verified as f64)
    } else {
        // Not "1.0000": nothing was checked, and saying so is the point.
        "n/a".to_string()
    };
    Some(format!(
        "live lane: precision {precision} over {verified}/{total} verifiable claims \
         ({disagree} disagreed with semantic analysis)"
    ))
}

/// and `get_blast_radius` answer from a graph degraded below the committed
/// snapshot. Reporting `complete` there is the actual harm; the edges
/// themselves return on the next commit's Phase B run.
///
/// The dirty flag therefore only changes the verdict inside that one window.
/// Once the markers diverge, `pending` already tells the user a run is coming.
///
/// The wording names the condition, not a remedy, because there is no single
/// correct remedy. The motivating cases (branch switch, `git stash pop`,
/// revert) all restore the file to its committed content, so the working tree
/// ends up equal to HEAD with the flag still set and the `ref/call` edge still
/// missing. Telling the user to commit is a dead end there: there is nothing
/// to stage. Recovery is `travsr init --semantic`, or any later commit that
/// fires the hook.
///
/// The `--semantic` is load-bearing rather than decorative. Plain `travsr init`
/// defers Phase B to the daemon (`run_phase_b_inline = semantic || !has_commit`),
/// so on an already-committed repo it re-runs Phase A and returns without
/// rebuilding the `ref/call` edges this flag is reporting as missing. The flag
/// therefore survives, and the message would be naming a command that cannot
/// clear it. Nor can the daemon rescue it here: `arm_phase_b_if_pending` only
/// arms when `last_commit != phase_b_commit`, and in this state they are equal.
///
/// Clearing the flag on the deferred path instead would be the wrong fix: the
/// edges genuinely are missing until Phase B re-runs, so the flag is honest and
/// it is the remedy that was wrong.
fn phase_b_state(payload: &StatusPayload) -> String {
    match payload.phase_b_commit.as_deref() {
        Some(pb) if !pb.is_empty() && Some(pb) == payload.last_commit.as_deref() => {
            if payload.phase_b_dirty {
                // A mid-edit reindex dropped the changed region's committed
                // edges. Whether that is a real degradation depends on the live
                // overlay, in three cases:
                //   - live lane inactive (no ref_resolution rows at all: a
                //     headless daemon with no editor, or a generic-detector
                //     language with no lexical floor): nothing recovered the
                //     edit, so it is genuinely stale until a refresh.
                //   - active with references still pending: name how many are
                //     unknown until commit.
                //   - active with nothing pending: the overlay resolved every
                //     reference it detected, so "stale, re-run init" would be
                //     wrong advice.
                // The counts are repo-wide and phase_b_dirty is a single flag,
                // so this last case cannot prove every dropped edge came back:
                // an editor-resolved file and a headless generic-language edit
                // (which leaves no rows at all) both feed one flag, and the
                // resolved rows may belong only to the first. So it reports the
                // recovery it can see without claiming a full refresh, which the
                // commit-gated path is what actually delivers.
                let live_active = payload.live_refs_resolved > 0 || payload.live_refs_pending > 0;
                if !live_active {
                    "stale (run travsr init --semantic to refresh)".to_string()
                } else if payload.live_refs_pending > 0 {
                    format!(
                        "{} reference(s) in uncommitted edits not yet resolved",
                        payload.live_refs_pending
                    )
                } else {
                    "uncommitted edits resolved where detected; commit for a full refresh"
                        .to_string()
                }
            } else {
                // #712: the marker now advances even when a language crashed, so
                // the healthy languages are complete and queryable at HEAD. Name
                // any crashed language rather than reporting a flat "complete"
                // that contradicts the per-language reality and the crash
                // warning printed below.
                // Downgrade a flat "complete" when a language that is turned on for
                // this repo did not run to a completed analysis: it crashed, or it
                // never ran at all (its analyzer is missing, or a pre-upgrade index
                // skipped it pending the now-removed elevated approval). A run that
                // DID complete and found no symbols is not counted — 0 nodes is a
                // valid result, not a failure — and languages the user has not turned
                // on (not trusted / not registered) are their own separate notice,
                // not a downgrade of the ones that did run.
                let crashed = crashed_langs(payload);
                let not_run: Vec<String> = warned_langs(payload, "skipped_no_analyzer")
                    .into_iter()
                    .chain(warned_langs(payload, "needs_consent"))
                    // needs_approval is vestigial (elevated access is auto-granted
                    // now, ADR-017 A5), but a pre-upgrade index can still have it in
                    // stored meta; keep honouring it so status stays honest rather
                    // than reporting a flat "complete" for a language that never ran.
                    .chain(warned_langs(payload, "needs_approval"))
                    .collect();
                if crashed.is_empty() && not_run.is_empty() {
                    "complete".to_string()
                } else {
                    let mut parts = Vec::new();
                    if !crashed.is_empty() {
                        parts.push(format!("crashed: {}", crashed.join(", ")));
                    }
                    if !not_run.is_empty() {
                        parts.push(format!("not run: {}", not_run.join(", ")));
                    }
                    format!("partial ({})", parts.join("; "))
                }
            }
        }
        Some(pb) if !pb.is_empty() => "pending".to_string(),
        _ => "not run".to_string(),
    }
}

/// #712: languages whose Phase B sidecar crashed on the last run, parsed from the
/// `phase_b_warnings` meta (`crashed:<lang>` entries). Used to downgrade the
/// `semantic:` field from `complete` to `partial (crashed: …)` so it agrees with
/// the crash warning and the per-language outcome.
fn crashed_langs(payload: &StatusPayload) -> Vec<String> {
    warned_langs(payload, "crashed")
}

/// Languages named by a `<kind>:<lang>` entry in the `phase_b_warnings` meta, for
/// the given `kind`. Used to reconcile the `semantic:` field with the per-language
/// warnings printed below it, so the summary line never contradicts them.
fn warned_langs(payload: &StatusPayload, kind: &str) -> Vec<String> {
    let prefix = format!("{kind}:");
    payload
        .phase_b_warnings
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter_map(|w| w.strip_prefix(&prefix))
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// #825: render the unreconciled SCIP definitions behind the E6 miss warning.
///
/// The miss set is deterministic, so the actionable information is *which*
/// definitions miss (which language/construct is failing), not a bare count.
/// Each stored row is `lang\tkind\tsymbol\tpath:line`; a malformed row is passed
/// through verbatim rather than dropped. Pure (no stderr) so it is unit-testable.
///
/// Display is capped at `SHOW`. `missed` is the true total from the warning's
/// `missed/attempted` rate, which is larger than `list` whenever the daemon hit
/// its own storage cap (`MAX_MISS_ROWS`) — the overflow line must count from it,
/// not from the stored rows, or the tail contradicts the warning above it.
fn unification_miss_lines(list: Option<&str>, missed: Option<usize>) -> Vec<String> {
    const SHOW: usize = 20;
    let Some(list) = list.filter(|s| !s.is_empty()) else {
        return Vec::new();
    };
    let rows: Vec<&str> = list.lines().collect();
    let mut out: Vec<String> = rows
        .iter()
        .take(SHOW)
        .map(|row| {
            let mut cols = row.split('\t');
            match (cols.next(), cols.next(), cols.next(), cols.next()) {
                (Some(lang), Some(kind), Some(symbol), Some(loc)) => {
                    format!("  {lang} {kind} {symbol}  {loc}")
                }
                _ => format!("  {row}"),
            }
        })
        .collect();
    // `max(rows.len())` keeps the tail honest if the rate and the list ever
    // disagree the other way (daemon/CLI skew): never under-report the rest.
    let total = missed.unwrap_or(0).max(rows.len());
    if total > out.len() {
        out.push(format!("  … and {} more", total - out.len()));
    }
    out
}

/// #645 WS-B: the caller's live short HEAD, read at `cwd` (before the worktree
/// redirect in `find_git_root`, so a linked worktree reports its own commit,
/// not the main worktree's). `None` when git is unavailable or the dir is not a
/// repo — the mismatch note then correctly never fires.
fn head_at(cwd: &std::path::Path) -> Option<String> {
    // Bounded: an unbounded `output()` here can never return on Windows when a
    // git child or grandchild inherits the pipe (#717 triage, same mechanism as
    // #503 / #572). A HEAD that does not arrive is the same as no HEAD, which
    // this function already handles.
    // `cwd` goes through as a real path rather than `-C <string>`: a path with
    // bytes that are not valid UTF-8 is legal, and converting it to a string
    // first would mangle it into U+FFFD and lose a repo that exists.
    crate::git_bounded::git_stdout_bounded(Some(cwd), ["rev-parse", "--short", "HEAD"])
}

pub fn run() -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("getting current directory")?;
    // `head_at` and `find_git_root` are independent, bounded git queries on the
    // same `cwd` (the latter only shells out in the linked-worktree branch, via
    // `main_worktree_root`). Run concurrently rather than sequentially: with a
    // wedged git, sequential calls each pay their own `GIT_QUERY_TIMEOUT`, so
    // this command could stall for up to 2x the bound instead of 1x.
    let head_handle = {
        let cwd = cwd.clone();
        std::thread::spawn(move || head_at(&cwd))
    };
    let repo_root = find_git_root(&cwd)?;
    let head = head_handle.join().ok().flatten();

    let db_path = repo_root.join(".travsr").join("graph.db");

    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    let payload: StatusPayload =
        match daemon_client::try_query(&repo_root, "status", serde_json::json!({})) {
            Some(p) => p,
            None => {
                let store = daemon_client::open_read_store(&db_path)
                    .with_context(|| format!("opening graph database at {}", db_path.display()))?;
                query::status_query(&store)?
            }
        };

    let last_commit = payload.last_commit.as_deref().unwrap_or("(none)");
    let phase_b_state = phase_b_state(&payload);
    // RFC-021 P5: reranker state. Old daemons omit the field (serde default
    // empty) — suppress the segment then so mixed CLI/daemon versions stay clean.
    let rerank_segment = if payload.rerank.is_empty() {
        String::new()
    } else {
        format!(" | rerank: {}", payload.rerank)
    };
    println!(
        "nodes: {} | edges: {} | schema: v{} | last_commit: {} | semantic: {}{}",
        payload.nodes, payload.edges, payload.schema, last_commit, phase_b_state, rerank_segment
    );

    // RFC-027 section 12: the live lane's measured precision, so the per-language
    // shipping gate has a number a human can read rather than a log line that
    // scrolled away.
    //
    // Read straight from the store rather than added to `StatusPayload`: this is
    // a diagnostic, and threading it through the query payload would mean a
    // protocol bump that every mixed CLI/daemon pair then has to survive.
    //
    // Silent when the lane has never claimed anything, which is every repo that
    // has not used it — a counter of zero is not news.
    if let Ok(store) = daemon_client::open_read_store(&db_path) {
        if let Some(line) = live_precision_line(&store) {
            println!("{line}");
        }
    }

    // #645 WS-B: the freshness markers only ever compare against each other,
    // never against the repository. Compare the caller's live HEAD (read at cwd,
    // above) to the index's last_commit so a checkout at a different revision —
    // a linked worktree, or a HEAD move the daemon has not yet reconciled — is
    // never answered for silently. cwd-local, so it holds for both the
    // daemon-answered and cold-store payloads.
    if let Some(head) = head.as_deref() {
        let stored = payload.last_commit.as_deref().unwrap_or("");
        if let Some(note) = travsr_mcp::head_index_mismatch_note(head, stored) {
            eprintln!("{note}");
        }
    }

    // RFC-014 #317 re-index policy: surface signature-format skew so the user
    // knows the graph was built with an older format and a re-index is due.
    let sig_v = payload.signature_format_version;
    if sig_v != travsr_core::SIGNATURE_FORMAT_VERSION {
        eprintln!(
            "warning: this index was built with an older version of travsr (format v{sig_v}, current v{}); run `travsr init` to rebuild it",
            travsr_core::SIGNATURE_FORMAT_VERSION
        );
    }

    // L11: detect FTS/nodes skew — indicates a partial write or corrupt FTS index.
    let fts = payload.fts_nodes;
    if fts > 0 && fts != payload.nodes {
        eprintln!(
            "warning: text search index has {fts} rows but the graph has {} nodes; run `travsr init` to rebuild",
            payload.nodes
        );
    }

    // H3: surface Phase B warnings so the user knows about crashed/mismatched
    // analyzers without having to re-read the init output.
    if let Some(warnings) = &payload.phase_b_warnings {
        if !warnings.is_empty() {
            // Trust is per-repo, not per-language: a single `install` enables
            // every language at once, so collapse the "not enabled here" notices
            // into one line rather than repeating it per language (matches init).
            let untrusted: Vec<&str> = warnings
                .split(',')
                .filter_map(|w| w.strip_prefix("untrusted_corpus:"))
                .collect();
            if !untrusted.is_empty() {
                eprintln!(
                    "warning: semantic analysis is not enabled for this repository yet ({}); run `travsr lang install <lang>` here to enable",
                    untrusted.join(", ")
                );
            }
            for warn in warnings.split(',') {
                let parts: Vec<&str> = warn.splitn(2, ':').collect();
                match parts.as_slice() {
                    // #712: point at the force path. A plain `travsr init
                    // --semantic` re-runs on top of the existing graph, which a
                    // no-op Phase A can make look like it did nothing; `--force`
                    // purges and rebuilds so the retry is unambiguous.
                    ["crashed", lang] => eprintln!(
                        "warning: semantic analyzer for '{lang}' crashed, fix the tool (e.g. `travsr lang install {lang}`), then re-run `travsr init --semantic --force` to rebuild"
                    ),
                    ["version_mismatch", rest] => {
                        let v: Vec<&str> = rest.splitn(3, ':').collect();
                        if let [lang, expected, got] = v.as_slice() {
                            eprintln!(
                                "warning: the '{lang}' analyzer is out of date (protocol v{got}, expected v{expected}); run `travsr lang install {lang}`"
                            );
                        }
                    }
                    // Windows only: an analyzer that cannot run inside Travsr's
                    // isolation and has no permission on record. The one-time
                    // permission is the only thing standing between it and full
                    // analysis here.
                    ["needs_consent", lang] => eprintln!(
                        "warning: full '{lang}' analysis needs your permission to run; run `travsr lang allow-unsandboxed {lang}`"
                    ),
                    // #712: analyzer ran but produced no nodes over the repo's
                    // source files of this language — a silent zero-node result,
                    // not a crash. Point at the tool and a rebuild.
                    // UX-3: the analyzer is installed and active (the zero-node
                    // warning only fires for a language that ran), so telling the
                    // user to reinstall misdirects — reinstalling changes nothing.
                    // The real causes are the sidecar failing to parse/resolve this
                    // repo's sources (e.g. a sandbox-denied read, a missing SDK, or
                    // no buildable project). Point at the sidecar's own diagnostics,
                    // which the host now forwards on stderr.
                    // #724: definitions arrived, occurrences did not, so no call
                    // edge can come from this language. The analyzer reported
                    // success, which is what makes it worth saying out loud.
                    ["no_references", lang] => {
                        eprintln!(
                            "warning: '{lang}' analysis produced definitions but no references, so no call edges came from it. The analyzer reported success, so this is its output being incomplete rather than a crash. Re-run `RUST_LOG=travsr_plugin_host=debug travsr init --semantic --force` to see its own diagnostics"
                        );
                    }
                    ["zero_nodes", lang] => {
                        eprintln!(
                            // `=warn`, not `=debug`: this PR promoted the
                            // zero-node stderr echo in `travsr-plugin-host`'s
                            // transport from `debug!` to `warn!` precisely so
                            // the cause is reachable without turning on
                            // everything else, and `observability.rs` tells an
                            // agent `=warn` for this same condition. Two
                            // surfaces naming two levels for one problem is how
                            // the advice starts drifting.
                            "warning: '{lang}' analysis ran but found no symbols, though the repo has '{lang}' sources. The analyzer is installed, so reinstalling will not help. The cause is in the analyzer's own output: re-run `RUST_LOG=travsr_plugin_host=warn travsr init --semantic --force` to see it. It may be this project (a missing SDK, an unbuildable project, or a build that skips part of its sources), or it may be how travsr invoked the analyzer"
                        );
                        // Name the concrete thing to check rather than leaving
                        // the possible causes as the only clue: the catalog
                        // already knows what this language's analyzer needs
                        // from the project.
                        if let Some(entry) = travsr_plugin_host::phase_b::catalog::lookup(lang) {
                            let prereq = entry.effective_prerequisites();
                            if !prereq.is_empty() && prereq != "none" {
                                eprintln!("  needs: {prereq}");
                            }
                        }
                        // #724 Finding 4: the most common cause of a zero-node
                        // Java run on macOS is scip-java's javac shim crashing
                        // under the stock bash 3.2. Surface the actionable fix.
                        if *lang == "java" {
                            if let Some(hint) = crate::progress::macos_java_bash_hint() {
                                eprintln!("  {hint}");
                            }
                        }
                    }
                    // #449: languages present in the repo whose Phase B sidecar
                    // never ran, previously a silent skip that left the user
                    // with "0 references" and no explanation.
                    // A language whose analyzer has no build for this OS can never
                    // reach full analysis here, so pointing at `travsr lang install`
                    // (which just dead-ends) is misleading — state the honest
                    // "not available on this platform" instead.
                    ["skipped_unregistered", lang]
                        if crate::lang::full_analysis_unavailable_here(lang) =>
                    {
                        eprintln!(
                            "note: full '{lang}' analysis is not available on this platform, structural analysis still works"
                        )
                    }
                    ["skipped_unregistered", lang] => eprintln!(
                        "warning: '{lang}' sources found but full analysis is not set up. Run `travsr lang install {lang}`"
                    ),
                    ["skipped_no_analyzer", lang]
                        if crate::lang::full_analysis_unavailable_here(lang) =>
                    {
                        eprintln!(
                            "note: full '{lang}' analysis is not available on this platform, structural analysis still works"
                        )
                    }
                    // #414 (ADR-017 Rule 3): registered globally but this repo was
                    // never enabled. Collapsed into one combined line above the
                    // loop (trust is per-repo, so one install fixes all of them).
                    ["untrusted_corpus", _] => {}
                    ["skipped_no_analyzer", lang] => eprintln!(
                        "warning: '{lang}' is registered but its analyzer binary is missing. Run `travsr lang install {lang}`"
                    ),
                    // L5a: scip-clang (c/cpp) needs a compile_commands.json at the
                    // repo root — without one it hangs, so it is skipped up front.
                    ["skipped_no_compdb", lang] => eprintln!(
                        "warning: full '{lang}' analysis needs a compile database (compile_commands.json) at the repo root. Generate one (e.g. `bear -- make`, or CMake's CMAKE_EXPORT_COMPILE_COMMANDS)"
                    ),
                    // E6: SCIP definitions that did not unify onto their Phase A
                    // tree-sitter node — their references attribute to an orphaned
                    // duplicate node instead. `rate` is missed/attempted.
                    //
                    // #825: the miss set is deterministic (same sources + emitter
                    // output => the identical misses every run), so the old
                    // "Re-run `travsr init --semantic` if it persists" advice was a
                    // no-op that could never move the count. Name the symbols
                    // instead — that is what a dev can act on.
                    //
                    // The list itself can legitimately be absent: an index whose
                    // last Phase B run predates #825 has the warning key but not
                    // the list key, as does an old daemon serving a new CLI. Only
                    // then does re-running do something — it writes the list — so
                    // say that instead of promising rows that never print.
                    ["scip_unification_misses", rate] => {
                        let missed = rate
                            .split_once('/')
                            .and_then(|(m, _)| m.parse::<usize>().ok());
                        let rows = unification_miss_lines(
                            payload.scip_unification_miss_list.as_deref(),
                            missed,
                        );
                        let tail = if rows.is_empty() {
                            "run `travsr init --semantic` once to record which definitions they are"
                        } else {
                            "the unreconciled definitions are listed below"
                        };
                        eprintln!(
                            "warning: {rate} semantic definitions did not match their parsed symbol, so some references may resolve to a duplicate. This is deterministic (re-running the index will not change the count); {tail}."
                        );
                        for row in rows {
                            eprintln!("{row}");
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // M1 / #738: warn when Rust's full cross-file edges are degraded. The
    // sandbox remedy is per-OS: only Linux has a sandbox the user can install
    // (bubblewrap); Windows and macOS have none to add here, so the only path
    // to full edges is the trusted-repo opt-in. `cfg!` (not `#[cfg]`) keeps
    // every platform's wording compiled and checked.
    if let Some(reason) = &payload.rust_lsif_degraded {
        match reason.as_str() {
            "sandbox_unavailable" => {
                let remedy = if cfg!(target_os = "linux") {
                    "Install bubblewrap, or re-run `travsr init --allow-unsandboxed` if you trust this repo."
                } else {
                    "Re-run `travsr init --allow-unsandboxed` if you trust this repo."
                };
                eprintln!(
                    "warning: Rust is on basic analysis, full cross-file edges (from rust-analyzer) were skipped because they need a security sandbox that is not available here. {remedy}"
                );
            }
            // #738: rust-analyzer ran and produced references, but every one was
            // dropped during resolution (none matched a parsed symbol). On Windows
            // this was the path-normalization bug; if it persists after upgrading,
            // it points at a repo-root/URI mismatch worth reporting.
            "all_refs_dropped" => eprintln!(
                "warning: Rust is on basic analysis, rust-analyzer produced \
                 references but none could be matched to indexed symbols, so no \
                 type-resolved call edges were added (structural call edges are \
                 unaffected). Re-run `travsr init --force --allow-unsandboxed \
                 --semantic`; if it persists, please report it."
            ),
            _ => {}
        }
    }

    // WS-2: warn when Dart Phase B ran without resolved dependencies, so a
    // partial cross-package index is never mistaken for a complete one.
    if let Some(pkgs) = payload.dart_deps_unresolved.as_deref() {
        if !pkgs.is_empty() {
            eprintln!(
                "warning: Dart cross-package references are incomplete, these \
                 package(s) were indexed without resolved dependencies: {pkgs}. \
                 Run `dart pub get` in each to enable cross-package references \
                 (intra-package references are unaffected)."
            );
        }
    }

    // RFC-025 §8: sidecar version health (installed vs required vs latest), with
    // the exact remedy. Computed offline; the `latest` note is present only when
    // the local cache is warm. Prints nothing when no sidecar is installed.
    crate::sidecar_health::print_block();

    // #712 F: the embed sidecar can be installed while no backend is active, so
    // the semantic path silently runs without embeddings. Nudge to enable it.
    crate::embed::hint_activate_if_installed(&repo_root);

    Ok(())
}

#[cfg(test)]
mod tests {

    /// #717: `head_at` runs on every `travsr status`, and it used an unbounded
    /// `Command::output()`. These pin the two answers it must give without
    /// hanging for either: a real repo reports a short SHA, a directory that is
    /// not a repo reports nothing.
    /// The line never appears on a repo that has not used the live lane, and
    /// never reports a precision it did not measure.
    #[test]
    fn live_precision_line_is_silent_until_something_is_measured() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        assert_eq!(live_precision_line(&store), None, "no reading, no line");

        store.set_meta("live_precision", "0,0,0").unwrap();
        assert_eq!(
            live_precision_line(&store),
            None,
            "an empty tally is not news"
        );

        store.set_meta("live_precision", "garbage").unwrap();
        assert_eq!(
            live_precision_line(&store),
            None,
            "a corrupt value is not a reading"
        );
    }

    /// Coverage is always shown, and an unverified sample says so instead of
    /// rendering a perfect score it did not earn.
    #[test]
    fn live_precision_line_reports_coverage_beside_precision() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();

        store.set_meta("live_precision", "99,1,100").unwrap();
        let line = live_precision_line(&store).unwrap();
        assert!(line.contains("0.9900"), "precision: {line}");
        assert!(line.contains("100/200"), "coverage must be visible: {line}");
        assert!(
            line.contains("1 disagreed"),
            "false positives named: {line}"
        );

        // Nothing checkable: must not read as perfect.
        store.set_meta("live_precision", "0,0,40").unwrap();
        let line = live_precision_line(&store).unwrap();
        assert!(
            line.contains("n/a"),
            "forty unchecked claims must not render as 1.0000: {line}"
        );
    }

    #[test]
    fn head_at_reports_a_sha_inside_a_repo_and_none_outside() {
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let sha = head_at(here).expect("this crate lives in a git repo");
        assert!(!sha.is_empty(), "a short SHA, not an empty string");
        assert!(!sha.contains('\n'), "arrives trimmed: {sha:?}");
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "a short SHA is hex: {sha:?}"
        );

        let tmp = tempfile::tempdir().unwrap();
        assert!(
            head_at(tmp.path()).is_none(),
            "outside a repo there is no HEAD, and asking must not hang"
        );
    }

    /// The bound is what stops a wedged git holding the CLI forever, so the
    /// happy path must not be anywhere near it.
    #[test]
    fn head_at_is_far_inside_its_deadline() {
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let started = std::time::Instant::now();
        let _ = head_at(here);
        assert!(
            started.elapsed() * 4 < crate::git_bounded::GIT_QUERY_TIMEOUT,
            "a warm rev-parse should finish in a small fraction of the deadline, took {:?}",
            started.elapsed()
        );
    }
    use super::*;

    fn payload(last: &str, phase_b: &str, dirty: bool) -> StatusPayload {
        StatusPayload {
            nodes: 1,
            fts_nodes: 1,
            edges: 0,
            schema: 21,
            journal: "wal".into(),
            last_commit: Some(last.to_string()),
            signature_format_version: travsr_core::SIGNATURE_FORMAT_VERSION,
            phase_b_commit: Some(phase_b.to_string()),
            phase_b_warnings: None,
            rust_lsif_degraded: None,
            rerank: String::new(),
            phase_b_dirty: dirty,
            live_refs_resolved: 0,
            live_refs_pending: 0,
            dart_deps_unresolved: None,
            scip_unification_miss_list: None,
        }
    }

    #[test]
    fn phase_b_reports_complete_when_markers_agree_and_nothing_is_dirty() {
        assert_eq!(phase_b_state(&payload("abc", "abc", false)), "complete");
    }

    #[test]
    fn phase_b_reports_stale_when_a_watcher_reindex_degraded_the_graph() {
        // #583: the exact window this PR exists for. Both markers agree, so the
        // old logic said `complete`, but the file's `ref/call` edges are gone.
        assert_eq!(
            phase_b_state(&payload("abc", "abc", true)),
            "stale (run travsr init --semantic to refresh)"
        );
    }

    #[test]
    fn phase_b_reports_recovery_without_claiming_a_full_refresh() {
        // The live lane resolved its references with nothing pending, so the
        // dirty marker must not read as "stale, re-run init". But the counts are
        // repo-wide, so this cannot prove a separate headless edit (which leaves
        // no rows) also came back: the message reports the recovery it can see
        // and names the commit as the full refresh, never a bare "up to date".
        let mut p = payload("abc", "abc", true);
        p.live_refs_resolved = 12;
        p.live_refs_pending = 0;
        assert_eq!(
            phase_b_state(&p),
            "uncommitted edits resolved where detected; commit for a full refresh"
        );
    }

    #[test]
    fn phase_b_names_pending_references_instead_of_a_blanket_stale() {
        // Live lane active with some references still unresolved; report how many
        // are unknown until commit rather than a flat "stale".
        let mut p = payload("abc", "abc", true);
        p.live_refs_resolved = 5;
        p.live_refs_pending = 3;
        assert_eq!(
            phase_b_state(&p),
            "3 reference(s) in uncommitted edits not yet resolved"
        );
    }

    #[test]
    fn phase_b_names_pending_even_when_nothing_resolved_yet() {
        // The lane ran (rows exist) but settled nothing so far: still "active
        // with pending", not the inactive "stale", so the count is honest.
        let mut p = payload("abc", "abc", true);
        p.live_refs_resolved = 0;
        p.live_refs_pending = 4;
        assert_eq!(
            phase_b_state(&p),
            "4 reference(s) in uncommitted edits not yet resolved"
        );
    }

    #[test]
    fn phase_b_still_reports_pending_when_markers_diverge() {
        // A run is already coming, so "commit to refresh" would be wrong
        // advice. The dirty flag must not override this.
        assert_eq!(phase_b_state(&payload("def", "abc", false)), "pending");
        assert_eq!(phase_b_state(&payload("def", "abc", true)), "pending");
    }

    #[test]
    fn phase_b_reports_not_run_before_the_first_run() {
        assert_eq!(phase_b_state(&payload("abc", "", true)), "not run");
    }

    #[test]
    fn phase_b_reports_partial_when_a_language_crashed_but_markers_agree() {
        // #712: the marker advances on a partial run (healthy languages are
        // queryable at HEAD), so the field must name the failed language rather
        // than claiming a flat "complete". A crash and a never-ran (analyzer
        // missing) both downgrade: neither ran to a completed analysis.
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("crashed:objectivec,skipped_no_analyzer:php".into());
        assert_eq!(
            phase_b_state(&p),
            "partial (crashed: objectivec; not run: php)"
        );
    }

    #[test]
    fn phase_b_partial_names_every_crashed_language() {
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("crashed:objectivec,crashed:swift".into());
        assert_eq!(phase_b_state(&p), "partial (crashed: objectivec, swift)");
    }

    #[test]
    fn phase_b_downgrades_when_an_enabled_language_never_ran() {
        // A language turned on for this repo whose analyzer is missing or is
        // waiting on the user's unsandboxed-run permission never ran, so
        // "complete" would contradict the warning printed below. Both are named
        // under "not run".
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("skipped_no_analyzer:php,needs_consent:go".into());
        assert_eq!(phase_b_state(&p), "partial (not run: php, go)");
    }

    #[test]
    fn phase_b_still_honours_a_pre_upgrade_needs_approval_warning() {
        // #756 review: elevated access is auto-granted now, so this build never
        // writes `needs_approval`. But a pre-upgrade index persisted it in store
        // meta, and that survives the upgrade until the next reindex. The
        // language genuinely never ran, so status must stay honest with
        // "not run", not collapse to a flat "complete".
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("needs_approval:java".into());
        assert_eq!(phase_b_state(&p), "partial (not run: java)");
    }

    #[test]
    fn unification_misses_render_named_rows() {
        // #825: the diagnostic names each unreconciled def (lang, kind, symbol,
        // path:line) instead of only counting them.
        let list = "swift\tclass\tAdHandler\tAd.swift:12\nruby\tfunction\tApp.missing\tapp.rb:99";
        let lines = unification_miss_lines(Some(list), Some(2));
        assert_eq!(
            lines,
            vec![
                "  swift class AdHandler  Ad.swift:12".to_string(),
                "  ruby function App.missing  app.rb:99".to_string(),
            ]
        );
    }

    #[test]
    fn unification_misses_cap_display_and_count_the_rest() {
        let list = (0..25)
            .map(|i| format!("go\tfunction\tf{i}\tx.go:{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let lines = unification_miss_lines(Some(&list), Some(25));
        assert_eq!(lines.len(), 21, "20 rows + one overflow line");
        assert_eq!(lines.last().unwrap(), "  … and 5 more");
    }

    #[test]
    fn unification_misses_overflow_counts_from_the_true_total() {
        // #825 review: `write_phase_b_results` caps the STORED list at 100 rows
        // while display caps at 20, so for the issue's 152/2632 case counting the
        // remainder from the stored rows printed "… and 80 more" directly under a
        // warning that said 152. The overflow must count from the rate.
        let list = (0..100)
            .map(|i| format!("swift\tclass\tT{i}\tA.swift:{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let lines = unification_miss_lines(Some(&list), Some(152));
        assert_eq!(lines.len(), 21);
        assert_eq!(lines.last().unwrap(), "  … and 132 more");
        // An unparseable rate must never under-report: fall back to the rows.
        let lines = unification_miss_lines(Some(&list), None);
        assert_eq!(lines.last().unwrap(), "  … and 80 more");
    }

    #[test]
    fn unification_misses_empty_or_absent_render_nothing() {
        assert!(unification_miss_lines(None, Some(152)).is_empty());
        assert!(unification_miss_lines(Some(""), Some(152)).is_empty());
    }

    #[test]
    fn phase_b_zero_nodes_still_reports_complete() {
        // A run that COMPLETED and produced no symbols still completed — 0 nodes is
        // a valid result, not a failure — so it must not downgrade "complete".
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("zero_nodes:go".into());
        assert_eq!(phase_b_state(&p), "complete");
    }

    #[test]
    fn phase_b_opt_out_languages_do_not_downgrade_complete() {
        // Languages the user has not turned on for this repo (not trusted / not
        // registered) are not a failure of the ones that did run — they have their
        // own separate notice and must not turn "complete" into "partial".
        let mut p = payload("abc", "abc", false);
        p.phase_b_warnings = Some("untrusted_corpus:go,skipped_unregistered:php".into());
        assert_eq!(phase_b_state(&p), "complete");
    }
}
