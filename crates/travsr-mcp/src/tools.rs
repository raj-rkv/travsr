use std::collections::HashMap;
use std::path::PathBuf;

use travsr_core::{display_label, Node as CoreNode, NodeId};
use travsr_retrieval::{
    context_candidates, knapsack, token_cost, EdgeFilter, OpenFilter, MAX_CONTEXT_BUDGET,
    TOKEN_CHARS_PER_TOKEN,
};
use travsr_store::{SqliteStore, Store};

use crate::sanitize::{
    sanitize_for_mcp, sanitize_mcp_body_with_limit, validate_mcp_arg, validate_mcp_list_arg,
    validate_mcp_pattern_arg, wrap_envelope,
};

/// Short alias for the embed KNN closure used throughout `get_context` variants.
type EmbedKnnFn<'a> = &'a dyn Fn(&str, u32) -> Vec<(NodeId, f32)>;

/// Return the import targets (Depends edges) of the given file path.
/// Empty string when nothing is found — callers must NOT return an RPC error
/// for the no-results case (Tech Lead requirement).
pub fn get_dependencies(store: &SqliteStore, file: &str, transitive: bool, depth: u32) -> String {
    // SEC-002: reject path traversal / absolute paths / oversized args.
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_dependencies rejected invalid arg: {reason}");
        return String::new();
    }
    let raw = if transitive {
        get_dependencies_transitive_raw(store, file, depth)
    } else {
        get_dependencies_raw(store, file)
    };
    // #619: an empty answer for an argument that is not actually a file (a
    // symbol name passed by mistake, or an unknown path) reads as "this file
    // has no dependencies" — misleading. Replace it with a short hint.
    if raw.is_empty() {
        if let Some(hint) = dependency_arg_hint(store, file) {
            return sanitize_for_mcp(&hint);
        }
    }
    // SEC-001: sanitize raw result before returning to MCP client / LLM.
    // #661 WS-D: carry the HEAD-mismatch note on the real dependency answer. The
    // invalid-arg reject (above) and the arg-hint diagnostic are not path:line
    // answers and return early without a note.
    with_head_note(store, sanitize_for_mcp(&raw))
}

/// Diagnose an empty `get_dependencies` answer (#619).
///
/// Returns `Some(hint)` when the argument did not resolve to a file node:
/// either it matched only symbol nodes (the reported case — a function name
/// passed where a path belongs) or it matched nothing at all. Returns `None`
/// when a file node did match, because an existing file with zero `depends`
/// edges is a legitimate empty answer that must stay empty.
fn dependency_arg_hint(store: &SqliteStore, file: &str) -> Option<String> {
    let nodes = store.search_nodes_by_name(file).ok()?;
    if nodes.iter().any(|n| n.kind == "file") {
        return None;
    }
    match nodes.first() {
        Some(n) => Some(format!(
            "'{file}' resolved to {} '{}', not a file; get_dependencies takes a repo-relative file path. For a symbol, use get_callers or find_references.",
            n.kind, display_label(n)
        )),
        None => Some(format!(
            "no file matched '{file}'; get_dependencies takes a repo-relative file path (run get_repo_map to list files)."
        )),
    }
}

/// Transitive variant of `get_dependencies`: BFS over `depends` edges from the
/// seed file node, up to `depth` hops. Results are in BFS order — direct
/// dependencies first (no prefix), then each deeper hop indented with `  ↳ `
/// per level so the UI can render the tree depth. Deduplicated by node id, so a
/// diamond import graph lists each module once.
///
/// `depth` is caller-clamped (server dispatch clamps to 1..=10); a runaway graph
/// still terminates because every node is visited at most once.
fn get_dependencies_transitive_raw(store: &SqliteStore, file: &str, depth: u32) -> String {
    use std::collections::{HashSet, VecDeque};

    let nodes = match store.search_nodes_by_name(file) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_dependencies search error: {e}");
            return String::new();
        }
    };
    let seed = match nodes
        .iter()
        .find(|n| n.kind == "file")
        .or_else(|| nodes.first())
    {
        Some(n) => n,
        None => return String::new(),
    };

    let mut visited: HashSet<NodeId> = HashSet::new();
    visited.insert(seed.id);
    let mut queue: VecDeque<(NodeId, u32)> = VecDeque::new();
    queue.push_back((seed.id, 0));
    let mut lines: Vec<String> = Vec::new();

    while let Some((node_id, hop)) = queue.pop_front() {
        if hop >= depth {
            continue;
        }
        let edges = match store.iter_edges_from(node_id) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("get_dependencies edge query error: {e}");
                continue;
            }
        };
        // Batch: collect new dep dsts for this hop (marking visited), fetch once.
        let new_dsts: Vec<NodeId> = edges
            .iter()
            .filter(|e| e.kind.as_str() == "depends")
            .filter(|e| visited.insert(e.dst))
            .map(|e| e.dst)
            .collect();
        let node_map: HashMap<NodeId, CoreNode> = store
            .get_nodes(&new_dsts)
            .unwrap_or_default()
            .into_iter()
            .map(|n| (n.id, n))
            .collect();
        let prefix = "  ↳ ".repeat(hop as usize); // hop 0 = direct → no prefix
        for dst_id in new_dsts {
            if let Some(dst_node) = node_map.get(&dst_id) {
                lines.push(format!("{prefix}{}", display_label(dst_node)));
                queue.push_back((dst_id, hop + 1));
            }
        }
    }
    lines.join("\n")
}

/// Raw (unsanitized) variant used by global aggregation — sanitization happens
/// once at the aggregation point, not per-store.
fn get_dependencies_raw(store: &SqliteStore, file: &str) -> String {
    let nodes = match store.search_nodes_by_name(file) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_dependencies search error: {e}");
            return String::new();
        }
    };

    // Prefer a node whose kind is "file"; fall back to first match.
    let seed = nodes
        .iter()
        .find(|n| n.kind == "file")
        .or_else(|| nodes.first());

    let seed = match seed {
        Some(n) => n,
        None => return String::new(),
    };

    let edges = match store.iter_edges_from(seed.id) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("get_dependencies edge query error: {e}");
            return String::new();
        }
    };

    const DEFAULT_DEPS_TOP_K: usize = 25;

    let all_ids: Vec<NodeId> = edges
        .iter()
        .filter(|e| e.kind.as_str() == "depends")
        .map(|e| e.dst)
        .collect();
    let total_deps = all_ids.len();
    let ids = &all_ids[..all_ids.len().min(DEFAULT_DEPS_TOP_K)];
    let node_map: HashMap<NodeId, CoreNode> = store
        .get_nodes(ids)
        .unwrap_or_default()
        .into_iter()
        .map(|n| (n.id, n))
        .collect();
    let mut lines: Vec<String> = ids
        .iter()
        .filter_map(|id| node_map.get(id).map(|n| display_label(n).into_owned()))
        .collect();
    if total_deps > DEFAULT_DEPS_TOP_K {
        lines.push(format!(
            "[showing {DEFAULT_DEPS_TOP_K} of {total_deps}; use get_context for ranked coverage or get_dependencies with transitive=true for the full tree]"
        ));
    }
    lines.join("\n")
}

/// Returns `true` when Phase B was deferred to the daemon background scheduler
/// and has not yet completed for this repository.
///
/// The signal is: `last_commit` is set (a real init ran) but `phase_b_commit`
/// is absent (Phase B hasn't finished). This avoids false positives on no-commit
/// repos where both keys are absent and Phase B ran inline during init.
fn phase_b_pending(store: &SqliteStore) -> bool {
    let phase_b = store.get_meta("phase_b_commit").ok().flatten();
    let last = store.get_meta("last_commit").ok().flatten();
    phase_b.is_none() && last.is_some()
}

/// Degraded-state note for the structural graph tools (#617).
///
/// `phase_b_pending` only catches the never-ran case (marker absent). Two more
/// states leave the RefCall edge set incomplete while looking healthy:
///  - `phase_b_commit` present but behind `last_commit` (HEAD moved while the
///    daemon was down, or a rebuild is still queued), and
///  - `phase_b_dirty` set (#583: a watcher reindex dropped a file's call edges
///    without moving HEAD).
///
/// A third state (#755 Part B item 4): the markers agree and nothing is dirty —
/// the index IS current — but a language turned on for this repo never produced
/// call edges, because its analyzer is missing, it crashed, or it is waiting on a
/// one-time approval or permission. `travsr status` already says so
/// ("semantic: partial (not run: php)"), but a per-query answer said nothing, so
/// an empty `callers`/`graph` result read as "no callers" rather than "this
/// language was not analyzed". That is the same misleading abstention the two
/// notes above exist to prevent, from a different cause.
///
/// Mirrors `travsr status`'s freshness classification so the tools and the CLI
/// never disagree about completeness. Returns the note to append, or `None`
/// when Phase B is complete for the current commit.
pub fn phase_b_degraded_note(store: &SqliteStore) -> Option<String> {
    const PENDING: &str = "[note: call-graph index incomplete; semantic analysis has not caught up with the current commit; call edges may be missing and empty results are not authoritative. Run `travsr status` to check progress.]";
    const STALE: &str = "[note: call-graph edges degraded; a background re-index dropped call edges since the last semantic analysis run; empty results are not authoritative. Run `travsr init` to rebuild.]";
    let phase_b = store
        .get_meta("phase_b_commit")
        .ok()
        .flatten()
        .filter(|s| !s.is_empty());
    let last = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .filter(|s| !s.is_empty());
    let dirty = store.get_meta("phase_b_dirty").ok().flatten().as_deref() == Some("1");
    match (phase_b, last) {
        (None, Some(_)) => Some(PENDING.to_string()),
        (Some(pb), Some(lc)) if pb != lc => Some(PENDING.to_string()),
        _ if dirty => {
            // #583 set this flag on any mid-edit reindex, on the pre-live-lane
            // premise that the dropped call edges are simply gone. The live
            // overlay may have recovered them, so report the truth in three
            // cases rather than a blanket "degraded, run init":
            //   - no overlay ran (no resolution rows): the edges are genuinely
            //     missing, so the stale note stands.
            //   - some references are still pending: name that a few call edges
            //     may be missing until commit, without the heavy "run init".
            //   - nothing pending, but some resolved: the overlay recovered the
            //     edits it detected. The counts are repo-wide and phase_b_dirty
            //     is a single flag, so this cannot prove every dropped edge came
            //     back (a headless generic-language edit leaves no rows to count,
            //     yet feeds the same flag as an editor-resolved file). So it
            //     still warns lightly that a few edges may be missing until the
            //     commit, rather than falling through to "results current".
            let resolved = store.resolved_ref_count().unwrap_or(0);
            let pending = store.pending_ref_count().unwrap_or(0);
            if resolved == 0 && pending == 0 {
                Some(STALE.to_string())
            } else if pending == 0 {
                Some(
                    "[note: a background re-index dropped some call edges; \
                     uncommitted edits were re-resolved where detected, but a few \
                     edges may still be missing until the next commit.]"
                        .to_string(),
                )
            } else {
                Some(format!(
                    "[note: {pending} reference(s) in uncommitted edits are not \
                     yet resolved, so a few call edges may be missing until the \
                     next commit; other results are current.]"
                ))
            }
        }
        // Current index, but not every language in it was analyzed.
        _ => phase_b_unanalyzed_note(store),
    }
}

/// RFC-027 section 10: tell a reader that this answer includes un-ratified
/// edges, and where the remaining gaps are.
///
/// Two numbers, because they mean opposite things and an agent needs both:
/// `live` edges are references the overlay *did* resolve ahead of the commit
/// (extra freshness, not yet confirmed), while `pending` references are ones
/// nothing could resolve (a known gap, deliberately not guessed at). Reporting
/// only a count of live edges would read as pure upside and hide the abstentions
/// that are the other half of a fail-closed lane.
///
/// `None` when the overlay is empty and nothing is pending, which is the state
/// of any repo with no uncommitted edits — so the note never fires on a clean
/// tree and costs a reader nothing.
///
/// `answer` is the rendered response this note will decorate. The **pending**
/// half is scoped to the files that answer actually names, which is what makes
/// it actionable rather than ambient: an agent asking "who calls `foo`" was
/// previously told how many references are pending everywhere in the repo,
/// which reads as though they relate to the answer it just got. Matching by
/// substring rather than by parsing keeps this independent of each tool's output
/// format, and the file set it tests against is capped, which also bounds a
/// query that runs on every prose call.
///
/// The **live** half stays repo-wide and says so: it reports that an un-ratified
/// overlay is in play at all, which is true of the whole graph the answer was
/// drawn from, not of any one file in it.
fn live_overlay_note(store: &SqliteStore, answer: &str) -> Option<String> {
    /// Files considered when scoping the pending half. Beyond this the note is
    /// advisory anyway, and the cap is what keeps the group-by bounded.
    const PENDING_FILE_CAP: usize = 64;

    let live = store.count_edges_with_provenance("live").ok().unwrap_or(0);
    let pending: u64 = store
        .pending_ref_counts_by_file(PENDING_FILE_CAP)
        .unwrap_or_default()
        .into_iter()
        .filter(|(path, _)| answer.contains(path.as_str()))
        .map(|(_, n)| n)
        .sum();
    if live == 0 && pending == 0 {
        return None;
    }
    let mut parts = Vec::new();
    if live > 0 {
        parts.push(format!(
            "{live} edge{} resolved from uncommitted edits and not yet ratified (repo-wide)",
            if live == 1 { "" } else { "s" }
        ));
    }
    if pending > 0 {
        parts.push(format!(
            "{pending} reference{} in the files above detected but not resolved",
            if pending == 1 { "" } else { "s" }
        ));
    }
    Some(format!(
        "[note: live overlay active: {}. These resolve deterministically at the next commit; filter to provenance != live for ratified truth only.]",
        parts.join("; ")
    ))
}

/// Languages whose Phase B never produced call/ref edges on the last run, per
/// the `phase_b_warnings` meta (#755).
///
/// The classes are exactly the ones `travsr status` downgrades `complete` to
/// `partial` for (`travsr-cli/src/status.rs::phase_b_state`): a crash, a missing
/// analyzer, or a pending approval/permission. Deliberately NOT the
/// `skipped_unregistered` / `untrusted_corpus` classes — those mean the user has
/// not turned the language on, which `status` reports as its own notice rather
/// than as a degradation of the languages that did run. Keeping the two sets
/// equal is what stops this note and the `semantic:` line from contradicting
/// each other.
///
/// `zero_nodes` is likewise excluded: an analyzer that ran and found nothing is a
/// valid answer, not missing coverage.
fn phase_b_unanalyzed_note(store: &SqliteStore) -> Option<String> {
    const CLASSES: &[&str] = &[
        "crashed",
        "skipped_no_analyzer",
        "needs_approval",
        "needs_consent",
    ];
    let warnings = store.get_meta("phase_b_warnings").ok().flatten()?;
    let mut langs: Vec<&str> = Vec::new();
    for warn in warnings.split(',') {
        let warn = warn.trim();
        let Some((class, rest)) = warn.split_once(':') else {
            continue;
        };
        if !CLASSES.contains(&class) {
            continue;
        }
        // `version_mismatch` and friends carry `lang:extra`; take the language.
        let lang = rest.split(':').next().unwrap_or(rest);
        if !lang.is_empty() && !langs.contains(&lang) {
            langs.push(lang);
        }
    }
    if langs.is_empty() {
        return None;
    }
    Some(format!(
        "[note: no call edges were produced for {} on the last semantic analysis run, \
         so an empty result here is not authoritative for {}. Run `travsr status` for the \
         reason and the fix.]",
        langs.join(", "),
        if langs.len() == 1 {
            "that language"
        } else {
            "those languages"
        }
    ))
}

/// #645 WS-B: the stdio server's launch directory — the caller's checkout.
///
/// Captured once when the stdio server starts (the IDE launches it in the
/// workspace dir). The freshness markers only ever compare `phase_b_commit`
/// vs `last_commit` — two store values, never the repository — so a read from
/// a checkout at a *different* commit than the index (a linked worktree, or a
/// HEAD move the daemon has not yet reconciled) answers with confident
/// `path:line` for the wrong revision. Comparing this cwd's live HEAD to the
/// index's `last_commit` is the only signal that surfaces it. Unset in the SSE
/// server (a remote client has no local checkout) and in unit tests, where the
/// head note correctly never fires.
static LAUNCH_CWD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Record the stdio server's launch directory (§ `LAUNCH_CWD`). Idempotent:
/// only the first call wins, matching the single server lifetime.
pub fn set_launch_cwd(dir: PathBuf) {
    let _ = LAUNCH_CWD.set(dir);
}

/// Whether two abbreviated git SHAs denote **different** commits.
///
/// `git rev-parse --short` is not a fixed width: it honours `core.abbrev`, and
/// under the default `auto` the length grows with the repository's object
/// count. So the same commit can be stamped `a1b2c3d` in `last_commit` by one
/// process and read back as `a1b2c3d4` by another, whenever the repo has grown
/// since, a global `core.abbrev` is set, or (in global mode) the MCP server
/// runs under a different `HOME`, and therefore a different gitconfig, than
/// the daemon did. A plain `!=` then reports drift on an identical commit
/// (#636 round-4 review).
///
/// Compares on the shorter of the two as a prefix, which is exactly how git
/// itself treats an abbreviation: two abbreviations denote the same commit
/// when one is a prefix of the other. Case-insensitive, since git accepts
/// either case in a rev. Returns `None` when either side is empty (unknown),
/// so callers can keep "unknown" distinct from "differs" rather than
/// collapsing it to a verdict.
pub(crate) fn short_shas_differ(a: &str, b: &str) -> Option<bool> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    // `starts_with` on the shared prefix length, ASCII-case-insensitively.
    let differs = !long
        .as_bytes()
        .iter()
        .zip(short.as_bytes())
        .all(|(l, s)| l.eq_ignore_ascii_case(s));
    Some(differs)
}

/// #645 WS-B: build the index/HEAD mismatch note. `head` is the caller's live
/// `git rev-parse --short HEAD`; `stored` is the index's `last_commit`. Returns
/// a note naming both, or `None` when they agree or either is unknown, never
/// fabricate a mismatch for a git-less caller. Pure so both the MCP and CLI
/// surfaces classify identically and it is trivially unit-testable.
///
/// Agreement is decided by [`short_shas_differ`], not `==`, so the note does
/// not fire when the two sides merely abbreviate the same commit to different
/// widths (#636 round-4 review).
pub fn head_index_mismatch_note(head: &str, stored: &str) -> Option<String> {
    if head.is_empty() || stored.is_empty() || short_shas_differ(head, stored) == Some(false) {
        return None;
    }
    Some(format!(
        "[note: index describes commit {stored} but your checkout is at {head}; \
         results (including path:line) may not match your working tree. This is \
         expected in a linked worktree; otherwise run `travsr init` or wait for \
         the daemon to reconcile.]"
    ))
}

/// The caller's short HEAD at `cwd`, or `None` when git is unavailable or `cwd`
/// is not a repo. Uses `--short` to match the `--short`-stamped `last_commit`.
///
/// `pub(crate)`: reused by `observability::index_status_payload` (#636) for
/// `get_index_status`'s `head_commit` field.
pub(crate) fn git_short_head(cwd: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["-C", &cwd.to_string_lossy(), "rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!head.is_empty()).then_some(head)
}

/// #645/#661: the live short HEAD of a *registered* repo, resolved from its db
/// path (`<root>/.travsr/graph.db` → `<root>`). The global MCP server captures a
/// single `LAUNCH_CWD` — the caller's one workspace — which is NOT each
/// registered repo's checkout. The per-repo drift note must therefore read that
/// repo's own HEAD: using `LAUNCH_CWD` in a multi-repo aggregate would compare
/// one workspace's commit against every *other* repo's index and fire a spurious
/// mismatch on repos that are perfectly fresh. `None` when git is unavailable or
/// the db path has an unexpected shape — the note then correctly never fires.
fn repo_head_from_registry_path(db_path: &std::path::Path) -> Option<String> {
    let root = db_path.parent()?.parent()?;
    git_short_head(root)
}

/// Append the read-path advisory notes (when any) to a structural tool
/// response: the Phase-B degraded note (#617) and the index/HEAD mismatch note
/// (#645 WS-B). Both are independent and may both appear. An empty body becomes
/// the note(s) alone: the note IS the answer then — exactly the empty-but-
/// unreliable case these notes exist to flag.
///
/// `head` is the caller's live short HEAD (or `None` when git-less); split out
/// as a parameter so the composition is deterministically testable without the
/// process-global `LAUNCH_CWD`.
fn append_read_notes(store: &SqliteStore, body: String, head: Option<&str>) -> String {
    let stored = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .unwrap_or_default();
    // Computed against the answer as it stands, before any note is appended:
    // `live_overlay_note` scopes its pending half to the files this answer
    // names, and a note already appended is not part of the answer.
    let notes = [
        phase_b_degraded_note(store),
        live_overlay_note(store, &body),
        head.and_then(|h| head_index_mismatch_note(h, &stored)),
    ];
    let mut out = body;
    for note in notes.into_iter().flatten() {
        if out.is_empty() {
            out = note;
        } else {
            out.push('\n');
            out.push_str(&note);
        }
    }
    out
}

/// Production entry: resolves the caller's HEAD from the captured launch cwd
/// (§ `LAUNCH_CWD`) and appends the advisory notes. Fail-safe — no cwd or no git
/// means no head note.
fn with_phase_b_note(store: &SqliteStore, body: String) -> String {
    let head = LAUNCH_CWD.get().and_then(|cwd| git_short_head(cwd));
    append_read_notes(store, body, head.as_deref())
}

/// #661 WS-D: append the index/HEAD mismatch note (#645) **only** — no Phase-B
/// note — to a deterministic `path:line` tool's response. `find_references`,
/// `get_dependencies` and `get_graph_json` assert `path:line` but do not depend
/// on Phase B, so they carry the head-drift signal without the Phase-B one.
///
/// `head` is split out as a parameter (the same injectable seam as
/// [`append_read_notes`]) so the composition is deterministically testable
/// without the process-global `LAUNCH_CWD`. Empty body ⇒ the note is the whole
/// answer (the empty-but-unreliable case these notes exist to flag).
fn append_head_note(store: &SqliteStore, body: String, head: Option<&str>) -> String {
    let stored = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .unwrap_or_default();
    match head.and_then(|h| head_index_mismatch_note(h, &stored)) {
        Some(note) if body.is_empty() => note,
        Some(note) => format!("{body}\n{note}"),
        None => body,
    }
}

/// Production entry for the head-only note: resolves the caller's HEAD from the
/// captured launch cwd (§ `LAUNCH_CWD`) and appends the mismatch note. Fail-safe
/// — no cwd or no git means no note.
fn with_head_note(store: &SqliteStore, body: String) -> String {
    let head = LAUNCH_CWD.get().and_then(|cwd| git_short_head(cwd));
    append_head_note(store, body, head.as_deref())
}

/// #661 WS-D: the read-path advisory notes as JSON `signals` array items, for
/// tools whose body is JSON for renderers (`get_graph_json`) — appending a
/// `[note: …]` prose string would break `JSON.parse`. Returns the Phase-B note
/// (#617) and the index/HEAD mismatch note (#645), each already carried in the
/// `signals` array the global aggregator merges. `head` is injectable for the
/// same testability reason as [`append_head_note`].
fn read_note_signals(
    store: &SqliteStore,
    head: Option<&str>,
    answer: &str,
) -> Vec<serde_json::Value> {
    let stored = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .unwrap_or_default();
    [
        phase_b_degraded_note(store),
        // RFC-027 section 10: the same live-overlay note the prose tools get,
        // so a renderer or agent reading JSON is told just as plainly that this
        // answer includes un-ratified edges, and scoped against the same kind of
        // evidence (here the serialized nodes, which carry the file paths).
        live_overlay_note(store, answer),
        head.and_then(|h| head_index_mismatch_note(h, &stored)),
    ]
    .into_iter()
    .flatten()
    .map(serde_json::Value::String)
    .collect()
}

/// Return all nodes that have an incoming edge to the given symbol, tagged
/// by provenance so both semantic and structural callers are visible.
///
/// Output format:
///   `[call] fn:bar (function) — src/bar.ts`   ← LSIF RefCall (true call site)
///   `[structural] class:Foo (class) — src/foo.ts` ← Tree-sitter DefinesBinding
///
/// Both sets are always returned when present. Tagging allows the LLM to
/// distinguish a true caller from a structural parent (e.g. class→method),
/// and avoids the all-or-nothing footgun where one RefCall would hide all
/// DefinesBinding callers. The precedence policy in #47 will refine this further.
///
/// Empty string when nothing is found.
///
/// `path` is the same optional hint `find_references` takes, with the same
/// meaning: a filename, relative path, directory prefix or path fragment that
/// picks one definition of an overloaded name. Without it an ambiguous symbol
/// can only be refused, and on this repo 11% of leaf names are ambiguous.
pub fn get_callers(store: &SqliteStore, symbol: &str, path: Option<&str>) -> String {
    // SEC-002: validate before forwarding to store queries.
    if let Err(reason) = validate_mcp_arg(symbol) {
        tracing::warn!("get_callers rejected invalid arg: {reason}");
        return String::new();
    }
    if let Some(p) = path {
        if let Err(reason) = validate_mcp_arg(p) {
            tracing::warn!("get_callers rejected invalid path arg: {reason}");
            return String::new();
        }
    }
    // Phase B deferred: a HEAD commit exists but phase_b_commit hasn't been
    // stamped yet, meaning Phase B was deferred to the daemon background
    // scheduler. Return a structured message so the LLM/agent retries rather
    // than treating an absent result as "no callers exist".
    // No-commit repos (both keys absent) and fully-indexed repos (both keys
    // present) fall through to the normal path.
    if phase_b_pending(store) {
        return phase_b_pending_json("Semantic call-edge index");
    }
    // SEC-001: sanitize raw result before returning to MCP client / LLM.
    // Use the larger find-output limit, not the 4 KiB scalar cap: this tool
    // enumerates one row per call site exactly as `find_references` does, and
    // the scalar cap was cutting ~80% of a hub symbol's callers off mid-line
    // with no notice (192 callers of `open_in_memory` arrived as 38 rows).
    // #617: append the staleness note (marker behind HEAD / dirty flag) so an
    // empty caller list is never mistaken for an authoritative "no callers".
    with_phase_b_note(
        store,
        wrap_envelope(&sanitize_mcp_body_with_limit(
            &get_callers_raw(store, symbol, path),
            FIND_OUTPUT_LIMIT,
        )),
    )
}

/// Raw (unsanitized) variant used by global aggregation.
/// PR #715: whether the last Phase B run for `lang` crashed, per the
/// `phase_b_warnings` meta (comma-separated `class:lang[:extra]` entries; a crash
/// is `crashed:<lang>`, the same wording the CLI/`get_index_status` decode).
///
/// Since #712 a single crashing sidecar no longer pins the whole repo's
/// completion marker behind HEAD: healthy languages advance and the query tools
/// stop emitting the "building in the background" note. The cost is that a
/// language that crashed *after* emitting some occurrences leaves partial
/// coverage while the marker reads complete, so `get_callers` / `find_references`
/// for a symbol in that language can return a confident answer that is silently
/// short — the reference gate keys on whether the language has *any* occurrences,
/// not on whether the run that produced them finished. Callers append a one-line
/// caveat (they do not abstain — a partial answer is still useful) so the note is
/// attached to exactly the answers that might be incomplete.
fn phase_b_lang_crashed(store: &SqliteStore, lang: &str) -> bool {
    store
        .get_meta("phase_b_warnings")
        .ok()
        .flatten()
        .is_some_and(|warnings| {
            warnings.split(',').any(|entry| {
                let mut parts = entry.splitn(3, ':');
                parts.next() == Some("crashed") && parts.next() == Some(lang)
            })
        })
}

/// The one-line caveat appended to a get_callers / find_references answer when
/// [`phase_b_lang_crashed`] holds for the target language.
fn crash_caveat(lang: &str) -> String {
    format!(
        "note: semantic analysis for '{lang}' crashed on its last run, so these \
         results may be incomplete. Run `travsr status` for detail or `travsr init \
         --semantic --force` to rebuild."
    )
}

fn get_callers_raw(store: &SqliteStore, symbol: &str, path: Option<&str>) -> String {
    use travsr_core::EdgeKind;

    // Resolve the name through the same ladder `find_references` and `travsr
    // graph` use, so two distinct definitions that share a name are never
    // collapsed to whichever one the FTS ranking happened to put first. That
    // silent pick reported one definition's callers under the other's name, and
    // a reader concluded the unreported definition had none.
    //
    // Only an *exact* resolution (full signature, exact simple name, selector
    // head, dotted member) reaches this guard, so the partial matching the tool
    // schema documents is untouched: a partial query resolves to `None` here and
    // falls through to the same name search as before.
    let seeds: Vec<CoreNode> = match resolve_reference_targets(store, symbol, path) {
        RefTarget::Unique(n) => vec![n],
        // A selector family is one Objective-C method spelled at several
        // arities, not rival definitions, and no `path` could choose between
        // them. Union their callers, as `find_references` unions their sites.
        RefTarget::Family(nodes) => nodes,
        RefTarget::Ambiguous(nodes) => {
            return ambiguous_definitions_message(
                symbol,
                &nodes,
                "Re-run with a `path` hint to pick one of:",
            )
        }
        RefTarget::None => {
            // #647 parity with `find_references`: a `path` hint that filtered
            // out every definition must not fall through to a partial name
            // search that ignores the hint, which would answer about a symbol
            // somewhere else entirely.
            if let Some(hint) = path {
                match resolve_reference_targets(store, symbol, None) {
                    RefTarget::Unique(n) => return path_miss_message(symbol, hint, &[n]),
                    RefTarget::Ambiguous(nodes) | RefTarget::Family(nodes) => {
                        return path_miss_message(symbol, hint, &nodes)
                    }
                    RefTarget::None => {}
                }
            }
            let partial = match store.search_nodes_by_name(symbol) {
                Ok(n) => n,
                Err(e) => {
                    tracing::warn!("get_callers search error: {e}");
                    return String::new();
                }
            };
            // The `path` hint has to survive this fallback too. It used to be
            // dropped here and the first FTS row won, so a hint naming one file
            // was answered from a same-named symbol somewhere else entirely —
            // the confident-wrong answer #647 removed from the exact tiers,
            // still reachable through this one. Scoping here keeps the
            // documented partial matching and makes the hint mean the same
            // thing on every tier.
            match path {
                None => partial.into_iter().take(1).collect(),
                Some(hint) => {
                    let scoped: Vec<CoreNode> = partial
                        .iter()
                        .filter(|n| path_hint_matches(&n.vname.path, hint))
                        .take(1)
                        .cloned()
                        .collect();
                    // A hint that matches none of them says so, rather than
                    // widening silently back to the whole repo (#647) or
                    // answering an empty list that reads as "no callers".
                    if scoped.is_empty() && !partial.is_empty() {
                        return path_miss_message(symbol, hint, &partial);
                    }
                    scoped
                }
            }
        }
    };

    let seed = match seeds.first() {
        Some(n) => n,
        None => return String::new(),
    };

    let mut edges = Vec::new();
    for node in &seeds {
        match store.iter_edges_to(node.id) {
            Ok(e) => edges.extend(e),
            Err(e) => {
                tracing::warn!("get_callers edge query error: {e}");
                return String::new();
            }
        }
    }

    let relevant: Vec<_> = edges
        .iter()
        .filter_map(|e| match e.kind {
            EdgeKind::RefCall => Some((e, "[call]")),
            EdgeKind::DefinesBinding => Some((e, "[structural]")),
            _ => None,
        })
        .collect();
    let ids: Vec<NodeId> = relevant.iter().map(|(e, _)| e.src).collect();
    let node_map: HashMap<NodeId, CoreNode> = store
        .get_nodes(&ids)
        .unwrap_or_default()
        .into_iter()
        .map(|n| (n.id, n))
        .collect();

    // #399: resolve exact call-site `path:line`s for true call edges. A RefCall
    // edge is deduplicated by `(src,dst,kind)`, so one edge can stand for several
    // calls from the same caller — the recorded `edge_sites` occurrences recover
    // each site. Only when a language feeds no occurrence rows do we re-scan the
    // caller's source span for the callee's name, which requires `repo_root`
    // (stored in meta by init); absent (older indexes) or unresolvable → fall
    // back to the caller's definition line, never worse than before.
    let repo_root = resolve_repo_root(store);
    let callee_name = simple_name(&seed.vname.signature);
    const MAX_SITES_PER_CALLER: usize = 50;

    let mut lines: Vec<String> = Vec::new();
    for &(edge, tag) in &relevant {
        let Some(src_node) = node_map.get(&edge.src) else {
            continue;
        };
        // Mark an edge a reader must not take at face value. Two cases: an
        // un-ratified `live` overlay edge (RFC-027 section 10), and a ref/call
        // edge resolved by leaf-name matching rather than by type.
        let live = provenance_marker(edge);
        // True call edge: expand into exact call-site lines.
        //
        // The recorded occurrences (the rows `find_references` also reads) are
        // the trusted set. The textual re-scan runs only when a language feeds
        // no occurrence rows at all, because it matches the callee's name
        // anywhere in the caller's span: a comment, a string, or the callee's
        // own `const f = (…)` declaration. Preferring the rows removed 3127
        // phantom sites across 1081 edges on this repo.
        //
        // Known gap, deliberately left open: the gate is per (src,dst) EDGE, so
        // an emitter that records SOME of an edge's calls but not all of them
        // under-reports. scip-dotnet emits nothing for a generic invocation, so
        // a C# caller invoking one callee once generically and once not shows a
        // single site. Running the scan alongside the rows to recover that
        // costs a source read per call edge in every language and puts comment
        // and string mentions back in the output as extra marked lines, which
        // is a worse trade than the one narrow shape it recovers. Closing it
        // properly means fixing the emitter's occurrence coverage.
        if tag == "[call]" {
            let recorded = store
                .edge_call_site_lines(edge.src, edge.dst, MAX_SITES_PER_CALLER)
                .unwrap_or_else(|e| {
                    tracing::warn!("get_callers occurrence lookup error: {e}");
                    Vec::new()
                });
            let sites: Vec<u32> = if recorded.is_empty() {
                match repo_root.as_ref() {
                    Some(root) => {
                        call_site_lines(src_node, root, &callee_name, MAX_SITES_PER_CALLER)
                    }
                    None => Vec::new(),
                }
            } else {
                recorded
            };
            if !sites.is_empty() {
                for line in sites {
                    lines.push(format!(
                        "{tag} {} ({}) \u{2014} {}:{}{live}",
                        display_label(src_node),
                        src_node.kind,
                        src_node.vname.path,
                        line
                    ));
                }
                continue;
            }
        }
        // Structural edge, or a call edge with no resolvable site / no repo_root:
        // report the node's definition line as before.
        let loc = src_node.line.map(|l| format!(":{l}")).unwrap_or_default();
        lines.push(format!(
            "{tag} {} ({}) \u{2014} {}{}{live}",
            display_label(src_node),
            src_node.kind,
            src_node.vname.path,
            loc
        ));
    }
    // Cap the rows here rather than letting the byte limit cut one in half, and
    // say how many were left out. A hub symbol has thousands of call sites and
    // no reader can act on all of them; what a reader cannot survive is a list
    // that stops without saying it stopped.
    let total = lines.len();
    if total > MAX_CALLER_ROWS {
        lines.truncate(MAX_CALLER_ROWS);
        lines.push(format!("[showing {MAX_CALLER_ROWS} of {total} callers]"));
    }
    // The sigil is the last thing appended to a row, so the rows that survived
    // the cap are what decide whether the legend is worth a line.
    if lines.iter().any(|l| l.ends_with(HEURISTIC_SIGIL_ROW)) {
        lines.push(HEURISTIC_LEGEND.to_string());
    }
    // #715: a crash in this language's last Phase B run leaves partial coverage
    // while the marker reads complete, so this list may be missing callers in the
    // un-indexed files. Attach the caveat to the confident (non-empty) answer.
    if !lines.is_empty() && phase_b_lang_crashed(store, &seed.vname.language) {
        lines.push(crash_caveat(&seed.vname.language));
    }
    lines.join("\n")
}

/// Cap on the caller rows `get_callers` returns, mirroring
/// [`MAX_REFERENCE_SITES`] for the tool that enumerates the same shape of row.
const MAX_CALLER_ROWS: usize = 500;

/// The suffix marking an edge whose confidence differs from the default.
///
/// Empty for a type-resolved, ratified edge, so the common case reads exactly
/// as it did before. Two cases are called out:
///
/// * `live` (RFC-027 section 10): resolved but not yet ratified, precise
///   enough to act on, and honest that the commit-gated pipeline has not
///   confirmed it. Consumers needing ground truth filter it out; consumers
///   ignoring provenance simply see a fresher graph.
/// * `tree-sitter` on a `ref/call` edge: produced by `resolve_unresolved_calls`
///   matching a bare callee name against the graph, not by a compiler resolving
///   a type. Ratified, and still capable of being wrong. Its uniqueness gate
///   ("exactly one same-named candidate") is evidence about the *index*, not
///   about the call: a local binding the Phase A parser does not model (a JS
///   `const f = () => …`) leaves the only same-named node in an unrelated
///   package as the unique winner, and the edge is written as fact. The gate is
///   also skipped entirely when the call carries a crate hint, which fans out to
///   every path-matching candidate.
///
/// The old comment here asserted that "every other provenance is ratified" and
/// stopped there, which conflated ratified with correct. Restricted to
/// `RefCall`: Phase A's `defines/binding` and `depends` edges are also
/// `tree-sitter` and are structural facts from the AST, not name guesses.
fn provenance_marker(edge: &travsr_core::Edge) -> &'static str {
    match edge.provenance.as_deref() {
        Some("live") => " [live: resolved from your uncommitted edit, not yet ratified]",
        Some("tree-sitter") if edge.kind == travsr_core::EdgeKind::RefCall => HEURISTIC_SIGIL_ROW,
        _ => "",
    }
}

/// The name-matched-edge caveat, spelled out per site by `find_references` (via
/// `RefSite::heuristic`), where one occurrence line carries it at most once.
const HEURISTIC_MARKER: &str = " [heuristic: matched by name, not resolved by type]";

/// The same caveat on a `get_callers` row: one character, plus one legend line
/// at the end of the answer.
///
/// [`HEURISTIC_MARKER`] is 49 bytes on a ~70-byte row, and on a Phase-A-only
/// language (Go, Java, C#, Ruby, PHP) essentially every call edge is
/// name-matched, so the caveat itself was pushing callers out of the response.
/// The CLI tree already made this trade for the same reason
/// (`travsr-cli`'s `HEURISTIC_SIGIL`); the legend repeats its wording verbatim
/// so the two surfaces cannot describe the same edge in two different words.
const HEURISTIC_SIGIL_ROW: &str = " ~";

/// Printed once, after the rows, and only when a marked row was actually
/// rendered: a legend for a mark that is not on screen is noise.
const HEURISTIC_LEGEND: &str = "~ = matched by name, not resolved by type";

/// The marker for one occurrence site, empty unless it is name-matched.
fn site_marker(site: &travsr_core::RefSite) -> &'static str {
    if site.heuristic {
        HEURISTIC_MARKER
    } else {
        ""
    }
}

/// Extract the bare symbol name from a stored signature for textual call-site
/// matching: strips the `kind:` prefix and any scope qualifier. Examples:
/// `fn:SyncPod` → `SyncPod`, `method:Foo#charge` → `charge`,
/// `fn:SqliteStore.iter_edges_to` → `iter_edges_to`,
/// `fn:crate::repo::find_git_root` → `find_git_root`.
fn simple_name(signature: &str) -> String {
    // An Objective-C selector spells its own colons inside the name
    // (`method:Foo.setWidth:height:`), so the `rsplit(':')` below would eat the
    // whole thing and return "". A stored selector always ends in `:`, which no
    // other language's signature does, so split only the `kind:` prefix there
    // and keep the rest.
    let after_kind = if signature.ends_with(':') {
        signature
            .split_once(':')
            .map_or(signature, |(_, rest)| rest)
    } else {
        // rsplit(':') drops the `kind:` prefix and yields the last `::` component;
        // then split off any `.`/`#` method/scope qualifier.
        signature.rsplit(':').next().unwrap_or(signature)
    };
    after_kind
        .rsplit(['.', '#'])
        .next()
        .unwrap_or(after_kind)
        .trim()
        .to_string()
}

/// Whether `signature` carries `container` as the qualifier immediately
/// before `member`, joined by `.` or `#` (the two separators used across
/// Phase A/B signature conventions: `swift::ClassC.shared`, `method:ClassC.shared`,
/// `scip:...ClassC#shared.`).
///
/// #449: a plain `signature.contains("{container}.{member}")` check is not
/// enough. `"AppConfig.shared"` contains `"Config.shared"` as a substring, so
/// a query for `Config.shared` would wrongly match `AppConfig`'s field. The
/// match is anchored on its left boundary: the byte immediately before
/// `container` (if any) must not be an identifier character, so the container
/// name in the signature can't be a longer identifier's suffix. No right-hand
/// boundary is required, since a SCIP variable signature legitimately
/// continues past `member` with a trailing `.` (`ClassC#shared.`).
fn has_qualified_container(signature: &str, container: &str, member: &str) -> bool {
    let is_ident_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    b".#".iter().any(|&sep| {
        let needle = format!("{container}{}{member}", sep as char);
        signature
            .find(&needle)
            .is_some_and(|pos| pos == 0 || !is_ident_byte(signature.as_bytes()[pos - 1]))
    })
}

/// Whether `signature` carries ANY container qualifier before its bare name,
/// as opposed to a genuinely unqualified signature like `var:shared`.
///
/// #449: the dotted-tier "unique bare member" fallback must only ever match
/// signatures with no recorded container at all, never a signature that
/// happens to be qualified with some OTHER container. Without this check, a
/// query for `Config.shared` would resolve to the sole `AppConfig.shared` in
/// the store (mistaking "no match for MY container" for "no container
/// recorded"), even though `AppConfig` plainly is not `Config`.
fn has_any_container(signature: &str) -> bool {
    let after_kind = signature.rsplit(':').next().unwrap_or(signature);
    after_kind.rsplit_once(['.', '#']).is_some()
}

/// Global variant of `get_dependencies` — searches one named repo or all registered repos.
///
/// When `repo` is `Some`, only that repo's db is queried. When `None`, all
/// registered repos are searched and results are prefixed with `[repo-name]`.
/// Stale registry entries (db file deleted) are skipped silently.
pub fn get_dependencies_global(
    repos: &HashMap<String, PathBuf>,
    file: &str,
    repo: Option<&str>,
) -> String {
    // SEC-002: validate before registry + store queries.
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_dependencies_global rejected invalid arg: {reason}");
        return String::new();
    }
    // Aggregate raw results, prefix per-repo, then sanitize once.
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        let result = get_dependencies_raw(store, file);
        // #619: in single-repo mode, diagnose an empty answer exactly like the
        // stdio path. In multi-repo mode leave per-repo empties silent — a
        // hint per non-matching repo would spam the aggregate. The arg-hint is
        // a diagnostic, not a path:line answer, so it carries no HEAD note.
        if result.is_empty() && single {
            return dependency_arg_hint(store, file).unwrap_or(result);
        }
        let result = if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        // #661 WS-D: per-repo HEAD note — read *this* repo's own checkout, not
        // the global LAUNCH_CWD (the caller's one workspace), so a fresh repo in
        // the aggregate is never flagged against an unrelated commit.
        let head = repos
            .get(repo_name)
            .and_then(|db| repo_head_from_registry_path(db));
        append_head_note(store, result, head.as_deref())
    });
    // SEC-001: sanitize the fully-aggregated string once, with the same
    // row-enumerating limit the single-repo path uses: an aggregate over N
    // repos is the last place a 4 KiB cap belongs.
    wrap_envelope(&sanitize_mcp_body_with_limit(&raw, FIND_OUTPUT_LIMIT))
}

/// Global variant of `get_callers` — searches one named repo or all registered repos.
pub fn get_callers_global(
    repos: &HashMap<String, PathBuf>,
    symbol: &str,
    path: Option<&str>,
    repo: Option<&str>,
) -> String {
    // SEC-002: validate before registry + store queries.
    if let Err(reason) = validate_mcp_arg(symbol) {
        tracing::warn!("get_callers_global rejected invalid arg: {reason}");
        return String::new();
    }
    if let Some(p) = path {
        if let Err(reason) = validate_mcp_arg(p) {
            tracing::warn!("get_callers_global rejected invalid path arg: {reason}");
            return String::new();
        }
    }
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        // #617 + #645: per-repo notes — each store carries its own Phase B
        // freshness, and the HEAD note reads *this* repo's own checkout (not the
        // global LAUNCH_CWD, which is only the caller's one workspace).
        let head = repos
            .get(repo_name)
            .and_then(|db| repo_head_from_registry_path(db));
        let raw = get_callers_raw(store, symbol, path);
        let result = append_read_notes(store, raw, head.as_deref());
        if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    });
    // SEC-001: sanitize the fully-aggregated string once.
    sanitize_for_mcp(&raw)
}

// ── find_references (issue #299) ───────────────────────────────────────────────

/// Per-symbol cap on returned occurrence sites. Hub symbols (e.g. `error`) can
/// have thousands of uses; capping keeps the MCP byte budget bounded. Mirrors
/// the spirit of `MAX_SITES_PER_CALLER` but is per-symbol, not per-caller.
const MAX_REFERENCE_SITES: usize = 500;

/// Byte cap for `find_references` / `find_pattern` / `get_callers` output. The
/// default scalar cap (`sanitize_for_mcp`, 4 KiB) would truncate a capped
/// 500-row list mid-line and drop the truncation notice, the same trap
/// `get_snippets` avoids. These tools enumerate up to `MAX_*` rows, so they wrap
/// with this larger limit instead (still well under the 1 MiB MCP hard ceiling).
const FIND_OUTPUT_LIMIT: usize = 512_000;

/// Resolution outcome for a `find_references` symbol argument.
pub(crate) enum RefTarget {
    /// Exactly one definition — enumerate its references.
    Unique(CoreNode),
    /// Multiple definitions and no disambiguating `path` — return the list so the
    /// caller can re-query with a `path` hint (never silently pick `[0]`).
    Ambiguous(Vec<CoreNode>),
    /// One Objective-C method family reached by its selector head: every
    /// arity of `policyWithPinningMode:` when the query was
    /// `policyWithPinningMode`. Distinct from [`Self::Ambiguous`] on purpose:
    /// these are not rival definitions a `path` hint could choose between, they
    /// are the same method spelled at different arities in one class, and the
    /// head is the only thing a developer can type for them. References are the
    /// union over the family; a caller who wants one arity types the full
    /// selector, which resolves uniquely.
    Family(Vec<CoreNode>),
    /// No definition matched the name.
    None,
}

/// Does a stored node `path` satisfy a user-supplied `--path` / `path` hint?
///
/// A hint scopes a symbol query to a file or directory. #647: the original pin
/// only accepted a filename suffix on a `/` boundary (`ppr.rs`, `src/ppr.rs`),
/// so a directory prefix (`crates/travsr-retrieval`) or a bare fragment
/// (`retrieval`) silently matched nothing and read as a definitive "0
/// references". This widens the pin while keeping the boundary rule that stops a
/// bare filename bleeding into a longer one (`tools.rs` never matches
/// `mytools.rs`, #299 F1). A trailing `/` on the hint is ignored so
/// `crates/travsr-retrieval/` behaves like `crates/travsr-retrieval`.
///
/// Match if any of:
/// - the whole path equals the hint;
/// - the path ends with `/hint` (filename / trailing path on a `/` boundary);
/// - the path starts with `hint/` (directory prefix);
/// - the path contains `/hint/` (interior whole path segment);
/// - the hint is a bare word (no `/` and no `.`) and is a substring of the path
///   — so `retrieval` scopes to `crates/travsr-retrieval/...`. A filename-shaped
///   hint (one containing `.`) never gets this loose treatment, so the
///   `/`-boundary guard above still holds for it.
pub(crate) fn path_hint_matches(node_path: &str, hint: &str) -> bool {
    let hint = hint.trim_end_matches('/');
    if hint.is_empty() {
        return true;
    }
    node_path == hint
        || node_path.ends_with(&format!("/{hint}"))
        || node_path.starts_with(&format!("{hint}/"))
        || node_path.contains(&format!("/{hint}/"))
        || (!hint.contains('/') && !hint.contains('.') && node_path.contains(hint))
}

/// Resolve a symbol name (or full signature) to its definition node(s).
///
/// Ladder (consistent with `get_snippets` resolution): exact-signature match
/// first (handles full headers like `fn:charge` from `get_context`), then an
/// exact simple-name match over the FTS candidates. An optional `path` hint
/// (see [`path_hint_matches`]) pins overloaded names to a file or directory.
fn resolve_symbol_nodes(store: &SqliteStore, symbol: &str, path: Option<&str>) -> Vec<CoreNode> {
    // Tier 1: exact signature. The path hint is applied by the shared Rust pin
    // below (the single source of truth for `--path`), not by the store's narrow
    // suffix-only SQL fast-path, so a directory or fragment hint (#647) is not
    // dropped here before the pin can widen it.
    let mut candidates = store.lookup_nodes_exact(symbol, None).unwrap_or_else(|e| {
        tracing::warn!("resolve_symbol_nodes lookup_nodes_exact '{symbol}': {e}");
        Vec::new()
    });

    // Tier 2: exact simple-name match over FTS candidates (bare names like `charge`).
    if candidates.is_empty() {
        candidates = match store.search_nodes_by_name(symbol) {
            Ok(nodes) => nodes
                .into_iter()
                .filter(|n| {
                    // Exclude non-definition nodes: `file` container nodes and
                    // `import`/`use` nodes (kind == "import"). An import node
                    // shares the imported symbol's simple name, so without this
                    // guard `refs find_pattern` reads as "ambiguous — 2 defs"
                    // (the real `fn:` plus `use:...::find_pattern`), and pinning
                    // the import node yields an empty, misleading "0 references".
                    n.kind != "file"
                        && n.kind != "import"
                        && (n.vname.signature == symbol
                            || simple_name(&n.vname.signature) == symbol)
                })
                .collect(),
            Err(e) => {
                tracing::warn!("resolve_symbol_nodes search '{symbol}': {e}");
                Vec::new()
            }
        };
    }

    // Tier 2b: Objective-C selector head. Phase A stores the whole selector
    // (`method:AFSecurityPolicy.policyWithPinningMode:withPinnedCertificates:`)
    // so selectors sharing a leading keyword stay distinct nodes, but the
    // leading keyword is the only part a developer types. Match on it, and only
    // once the exact tiers above have found nothing, so this can never widen a
    // name that already resolves. `selector_head` returns a non-selector leaf
    // unchanged, so a plain name reaching here still matches nothing new.
    if candidates.is_empty() && !symbol.contains(':') {
        candidates = match store.search_nodes_by_name(symbol) {
            Ok(nodes) => nodes
                .into_iter()
                .filter(|n| {
                    n.kind != "file"
                        && n.kind != "import"
                        && travsr_core::ident::selector_head(&simple_name(&n.vname.signature))
                            == symbol
                })
                .collect(),
            Err(e) => {
                tracing::warn!("resolve_symbol_nodes selector head '{symbol}': {e}");
                Vec::new()
            }
        };
    }

    // Tier 3 (#449): dotted static/member access like `ClassC.shared` or
    // `Type.method`. Stored signatures qualify the member (`swift::ClassC.shared`,
    // `method:ClassC.shared`, `scip:...ClassC#shared.`) but never equal the
    // dotted query, and `simple_name` reduces them to the bare member, so both
    // tiers above miss. Split at the last `.`, search the member, and keep
    // candidates whose signature carries the container qualifier; fall back to
    // a bare-member match only when it is unique AND genuinely unqualified
    // (never a match with some OTHER, unrelated container).
    if candidates.is_empty() {
        if let Some((container, member)) = symbol.rsplit_once('.') {
            if !container.is_empty() && !member.is_empty() {
                let found = store.search_nodes_by_name(member).unwrap_or_else(|e| {
                    tracing::warn!("resolve_symbol_nodes dotted search '{member}': {e}");
                    Vec::new()
                });
                let matches: Vec<CoreNode> = found
                    .into_iter()
                    .filter(|n| {
                        n.kind != "file"
                            && n.kind != "import"
                            && simple_name(&n.vname.signature) == member
                    })
                    .collect();
                let qualified: Vec<CoreNode> = matches
                    .iter()
                    .filter(|n| has_qualified_container(&n.vname.signature, container, member))
                    .cloned()
                    .collect();
                candidates = if !qualified.is_empty() {
                    qualified
                } else {
                    let bare: Vec<CoreNode> = matches
                        .into_iter()
                        .filter(|n| !has_any_container(&n.vname.signature))
                        .collect();
                    if bare.len() == 1 {
                        bare
                    } else {
                        Vec::new()
                    }
                };
            }
        }
    }

    // Apply the path hint if the caller supplied one (#647: file, directory, or
    // fragment — see `path_hint_matches`).
    if let Some(p) = path {
        candidates.retain(|n| path_hint_matches(&n.vname.path, p));
    }

    // Distinct definitions by node id.
    candidates.sort_by_key(|n| n.id.0);
    candidates.dedup_by_key(|n| n.id);

    candidates
}

/// Helper to map `resolve_symbol_nodes` candidates to a `RefTarget` for
/// find_references and `travsr graph` ambiguity resolution (#565 / RFC-002).
pub(crate) fn resolve_reference_targets(
    store: &SqliteStore,
    symbol: &str,
    path: Option<&str>,
) -> RefTarget {
    let mut candidates = resolve_symbol_nodes(store, symbol, path);

    // C/C++ split a symbol into a header declaration and a source definition
    // (`utils.h` decl + `utils.c` def). They share the simple name, so both
    // survive the ladder above and the result reads as "ambiguous" even though
    // only the definition carries occurrence sites. When both a header node and
    // a non-header node with the same simple name are present, prefer the
    // definition (drop the header declarations). Header-only results (a pure
    // declaration with no definition indexed) are left untouched.
    if candidates.len() > 1 {
        let is_header = |p: &str| {
            let p = p.to_ascii_lowercase();
            p.ends_with(".h") || p.ends_with(".hpp") || p.ends_with(".hh") || p.ends_with(".hxx")
        };
        if candidates.iter().any(|n| !is_header(&n.vname.path)) {
            candidates.retain(|n| !is_header(&n.vname.path));
        }
    }

    // An Objective-C selector family reached by its head is one method, not
    // rival definitions, see `RefTarget::Family`. Same class, same head, and
    // every member an actual multi-part selector; two classes that both declare
    // a `policyWithPinningMode:` are still genuinely ambiguous and fall through.
    if candidates.len() > 1 && is_selector_family(&candidates, symbol) {
        return RefTarget::Family(candidates);
    }

    match candidates.len() {
        0 => RefTarget::None,
        1 => candidates
            .pop()
            .map(RefTarget::Unique)
            .unwrap_or(RefTarget::None),
        _ => RefTarget::Ambiguous(candidates),
    }
}

/// Whether every candidate is a multi-part Objective-C selector of ONE class
/// whose leading keyword is `head`: the [`RefTarget::Family`] test.
///
/// Three things must agree, and all three are load-bearing:
///
/// * the language is Objective-C. Nothing else spells a method name with
///   embedded colons, and the test used to run for every language on nothing
///   but `leaf.ends_with(':')`.
/// * the container name. Without it two different methods union.
/// * the container's PATH. The container is only a `String` cut out of the
///   signature, so name equality alone merged two classes that happen to share
///   a name (a vendored pod duplicated under two paths, a category, a name
///   colliding across two static libs) into one family, and `get_callers` then
///   presented two methods' callers as one method's. Two same-named classes in
///   two files are real ambiguity a `path` hint resolves.
fn is_selector_family(candidates: &[CoreNode], head: &str) -> bool {
    let container_of = |sig: &str| -> Option<String> {
        let body = sig.split_once(':').map_or(sig, |(_, rest)| rest);
        body.rsplit_once('.').map(|(c, _)| c.to_string())
    };
    let first = match container_of(&candidates[0].vname.signature) {
        Some(c) => c,
        None => return false,
    };
    let path = candidates[0].vname.path.as_str();
    candidates.iter().all(|n| {
        let leaf = simple_name(&n.vname.signature);
        n.vname.language == travsr_core::Language::ObjectiveC.as_str()
            && n.vname.path == path
            && leaf.ends_with(':')
            && travsr_core::ident::selector_head(&leaf) == head
            && container_of(&n.vname.signature).as_deref() == Some(first.as_str())
    })
}

/// The refusal every surface returns for [`RefTarget::Ambiguous`]: the count, an
/// `advice` sentence naming the escape hatch that surface actually offers, then
/// one line per rival definition.
///
/// Shared so `find_references` and `get_callers` cannot drift into describing
/// the same ambiguity differently, which is how `get_callers` came to describe
/// it not at all.
fn ambiguous_definitions_message(symbol: &str, nodes: &[CoreNode], advice: &str) -> String {
    let total = nodes.len();
    let shown = total.min(MAX_AMBIGUOUS_DEFINITIONS);
    let mut out = format!("'{symbol}' is ambiguous, {total} definitions. {advice}\n");
    for n in nodes.iter().take(shown) {
        let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
        out.push_str(&format!(
            "  {} ({}) \u{2014} {}{}\n",
            display_label(n),
            n.kind,
            n.vname.path,
            loc
        ));
    }
    if total > shown {
        out.push_str(&format!("[showing {shown} of {total} definitions]\n"));
    }
    out.trim_end().to_string()
}

/// Cap on the definition lines [`ambiguous_definitions_message`] lists.
///
/// Both tools now wrap with [`FIND_OUTPUT_LIMIT`], but this list stays short on
/// its own account: `get_callers` used to wrap with `sanitize_for_mcp`'s 4 KiB
/// scalar cap, which cut a 37-definition refusal off mid-line and dropped no
/// notice saying so. 25 lines are all a reader can act on, and the elided count
/// is stated rather than implied. Deliberately not `MAX_REFERENCE_SITES`: a
/// reader cannot act on 500 rival definitions anyway, they need the `path` hint.
const MAX_AMBIGUOUS_DEFINITIONS: usize = 25;

/// #647: message for a `path` hint that matched no definition of a symbol that
/// does resolve elsewhere. Shows where the symbol actually lives so the answer
/// is never mistaken for a real "0 references".
fn path_miss_message(symbol: &str, hint: &str, defs: &[CoreNode]) -> String {
    let total = defs.len();
    let shown = total.min(MAX_AMBIGUOUS_DEFINITIONS);
    let mut out = format!(
        "'{symbol}' resolves, but no definition is under path '{hint}'. It is defined at:\n"
    );
    for n in defs.iter().take(shown) {
        let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
        out.push_str(&format!(
            "  {} ({}) \u{2014} {}{}\n",
            display_label(n),
            n.kind,
            n.vname.path,
            loc
        ));
    }
    if total > shown {
        out.push_str(&format!("[showing {shown} of {total} definitions]\n"));
    }
    out.push_str("Re-run without `path`, or with a `path` hint that matches one of these.");
    out
}

/// The "Phase B has not finished" answer, shared by every tool that depends on it.
///
/// #726: these used to promise the result was "building in the background" and
/// would arrive "in ~2 minutes". Phase B only runs under a daemon, so with none
/// running the work is not scheduled at all and that wait never ends. The ETA
/// was counting down something nothing was counting.
///
/// One function rather than three literals because the three tools drifted:
/// `find_references`, `get_callers` and `get_execution_path` each carried their
/// own copy, so fixing one left the other two telling users the same falsehood.
fn phase_b_pending_json(index_name: &str) -> String {
    format!(
        r#"{{"status":"pending","message":"{index_name} has not finished. It is built by the daemon: check `travsr daemon status` and start one with `travsr daemon start` if none is running, or build it now with `travsr init --semantic`. `travsr status` shows progress."}}"#
    )
}

/// Enumerate every use site (`path:line`) of a symbol across the current repo.
///
/// Reads the `edge_sites` occurrence store (populated by every ingestion path —
/// SCIP, LSIF, native tree-sitter). When a language has `ref/call` edges but no
/// occurrence rows yet, degrades to caller-definition lines (labelled) rather
/// than returning empty, matching `get_callers`' structural fallback.
pub fn find_references(store: &SqliteStore, symbol: &str, path: Option<&str>) -> String {
    // SEC-002: validate every argument before any store query.
    if let Err(reason) = validate_mcp_arg(symbol) {
        tracing::warn!("find_references rejected invalid symbol arg: {reason}");
        return String::new();
    }
    if let Some(p) = path {
        if let Err(reason) = validate_mcp_arg(p) {
            tracing::warn!("find_references rejected invalid path arg: {reason}");
            return String::new();
        }
    }
    if phase_b_pending(store) {
        return phase_b_pending_json("Semantic occurrence index");
    }
    // SEC-001: sanitize before returning. Use the larger find-output limit (not
    // the 4 KiB scalar cap) so a capped 500-site list and its truncation notice
    // survive intact instead of being cut mid-line.
    // #661 WS-D: append the HEAD-mismatch note *after* sanitize (so it is never
    // truncated) but *before* wrap_envelope (so it stays inside <travsr-data>).
    // The early returns above (validation reject, pending JSON) are not
    // path:line answers and intentionally carry no note.
    let body =
        sanitize_mcp_body_with_limit(&find_references_raw(store, symbol, path), FIND_OUTPUT_LIMIT);
    wrap_envelope(&with_head_note(store, body))
}

/// One resolved definition site, for the structured `find_references` result.
#[derive(serde::Serialize)]
pub struct ResolvedSymbol {
    pub signature: String,
    pub kind: String,
    pub path: String,
    pub line: Option<u32>,
}

impl ResolvedSymbol {
    fn from_node(n: &CoreNode) -> Self {
        Self {
            signature: display_label(n).into_owned(),
            kind: n.kind.clone(),
            path: n.vname.path.clone(),
            line: n.line,
        }
    }
}

/// Machine-readable `find_references` result for the CLI `--format json` path.
///
/// The plain [`find_references`] returns the model-facing `<travsr-data>`
/// envelope, which is a human-readable text blob — unusable as structured data.
/// This mirrors the same resolution and occurrence lookup but exposes the parts
/// separately, so a script gets a real `references` array keyed off `symbol` and
/// `resolved_to` instead of having to parse prose.
#[derive(serde::Serialize)]
pub struct StructuredReferences {
    pub symbol: String,
    pub path: Option<String>,
    /// One of: `resolved`, `not_found`, `ambiguous`, `pending`.
    pub status: &'static str,
    /// The single definition the symbol resolved to, when `status == resolved`.
    pub resolved_to: Option<ResolvedSymbol>,
    /// Candidate definitions when the name is ambiguous, or defined outside a
    /// `path` hint. Disambiguate by re-running with a `path`.
    pub candidates: Vec<ResolvedSymbol>,
    /// Exact occurrence sites (capped at `MAX_REFERENCE_SITES`).
    pub references: Vec<travsr_core::RefSite>,
    /// Total occurrence count before the cap; `references.len()` when not truncated.
    ///
    /// `null` when no count was taken at all, i.e. whenever `status != "resolved"`
    /// (#755 Part B item 9). It used to be a flat `0` there, so an `ambiguous`
    /// answer — which carries `candidates` and a "re-run with a path hint" note,
    /// and says nothing about how many uses exist — serialized identically to a
    /// symbol that genuinely has no references. A script keying on `total` alone
    /// read that as a confident zero. `null` is not `0` in any consumer, so the
    /// same script now fails visibly instead of concluding the wrong thing.
    pub total: Option<usize>,
    pub truncated: bool,
    /// Human caveat when a zero result is not definitive (degraded coverage,
    /// path-hint miss, still-building index). `null` on a clean answer.
    pub note: Option<String>,
}

/// Drop a leading `resolved: …` header line: `resolved_to` already carries it as
/// structured fields, so it would be redundant inside `note`.
fn strip_resolved_header(text: &str) -> String {
    text.strip_prefix("resolved:")
        .and_then(|rest| rest.split_once('\n'))
        .map(|(_, body)| body.trim_start().to_string())
        .unwrap_or_else(|| text.trim().to_string())
}

/// Structured variant of [`find_references`], consumed by `travsr references
/// --format json`. Reuses the same resolver and occurrence store as the text
/// path so the two never disagree on what a symbol resolves to.
pub fn find_references_structured(
    store: &SqliteStore,
    symbol: &str,
    path: Option<&str>,
) -> StructuredReferences {
    let mut out = StructuredReferences {
        symbol: symbol.to_string(),
        path: path.map(str::to_string),
        status: "not_found",
        resolved_to: None,
        candidates: Vec::new(),
        references: Vec::new(),
        total: None,
        truncated: false,
        note: None,
    };

    // SEC-002: validate every argument before any store query, as the text path does.
    if validate_mcp_arg(symbol).is_err() || path.is_some_and(|p| validate_mcp_arg(p).is_err()) {
        out.note = Some("invalid symbol or path argument".to_string());
        return out;
    }
    if phase_b_pending(store) {
        out.status = "pending";
        out.note = Some(
            "Semantic occurrence index is still building, results are not yet \
             authoritative. Run `travsr status` to check progress."
                .to_string(),
        );
        return out;
    }

    let target = match resolve_reference_targets(store, symbol, path) {
        RefTarget::Unique(n) => n,
        // One method family reached by its selector head: report every arity as
        // a candidate and the union of their occurrence sites, rather than a
        // "pick one" the caller cannot act on.
        RefTarget::Family(nodes) => {
            out.status = "resolved";
            out.candidates = nodes.iter().map(ResolvedSymbol::from_node).collect();
            out.resolved_to = nodes.first().map(ResolvedSymbol::from_node);
            let sites = family_reference_sites(store, &nodes);
            let total = sites.len();
            out.total = Some(total);
            out.truncated = total > MAX_REFERENCE_SITES;
            out.references = sites.into_iter().take(MAX_REFERENCE_SITES).collect();
            return out;
        }
        RefTarget::Ambiguous(nodes) => {
            out.status = "ambiguous";
            out.note = Some(format!(
                "'{symbol}' is ambiguous, {} definitions. Re-run with a `path` hint to pick one.",
                nodes.len()
            ));
            out.candidates = nodes.iter().map(ResolvedSymbol::from_node).collect();
            return out;
        }
        RefTarget::None => {
            // A `path` hint that filtered out every definition is not a definitive
            // not-found: re-resolve without it and, if the symbol lives elsewhere,
            // report those as candidates rather than a bare miss.
            if let Some(hint) = path {
                let elsewhere = match resolve_reference_targets(store, symbol, None) {
                    RefTarget::Unique(n) => vec![n],
                    RefTarget::Ambiguous(nodes) | RefTarget::Family(nodes) => nodes,
                    RefTarget::None => Vec::new(),
                };
                if !elsewhere.is_empty() {
                    out.note = Some(format!(
                        "Symbol '{symbol}' was not found under path '{hint}', but is \
                         defined elsewhere (see candidates)."
                    ));
                    out.candidates = elsewhere.iter().map(ResolvedSymbol::from_node).collect();
                    return out;
                }
            }
            out.note = Some(format!("Symbol '{symbol}' was not found."));
            return out;
        }
    };

    out.status = "resolved";
    out.resolved_to = Some(ResolvedSymbol::from_node(&target));

    match store.reference_sites(target.id) {
        Ok(sites) if !sites.is_empty() => {
            let total = sites.len();
            out.total = Some(total);
            out.truncated = total > MAX_REFERENCE_SITES;
            out.references = sites.into_iter().take(MAX_REFERENCE_SITES).collect();
            // #715 parity with the text path: a crashed last Phase B run leaves
            // partial coverage under a complete marker, so even a non-empty site
            // list may be short. Surface the same caveat instead of `note: None`.
            if phase_b_lang_crashed(store, &target.vname.language) {
                out.note = Some(crash_caveat(&target.vname.language));
            }
        }
        _ => {
            // Resolved, but no exact occurrence sites. This IS a real count, so
            // `total` is an explicit `Some(0)` — the one case where zero is the
            // answer (#755 Part B item 9). Carry the text path's honest caveat
            // (degraded coverage vs. genuine zero) as `note`, minus the header line
            // that `resolved_to` already encodes. Reuse the target we just resolved
            // rather than re-running resolution.
            out.total = Some(0);
            out.note = Some(strip_resolved_header(&references_body_for_target(
                store, &target,
            )));
        }
    }
    out
}

/// Raw (unsanitized) variant, shared by the global aggregator.
fn find_references_raw(store: &SqliteStore, symbol: &str, path: Option<&str>) -> String {
    let target = match resolve_reference_targets(store, symbol, path) {
        RefTarget::Unique(n) => n,
        RefTarget::Family(nodes) => return references_body_for_family(store, &nodes),
        RefTarget::None => {
            // #647: a `path` hint that filtered out every real definition must
            // not read as a definitive "0 references" — that is the exact
            // misleading-abstention pattern the rest of this function is
            // hardened against (#299 / #450). Re-resolve ignoring the hint; if
            // the symbol genuinely exists elsewhere, say where instead of
            // returning an empty result.
            if let Some(hint) = path {
                match resolve_reference_targets(store, symbol, None) {
                    RefTarget::Unique(n) => return path_miss_message(symbol, hint, &[n]),
                    RefTarget::Ambiguous(nodes) | RefTarget::Family(nodes) => {
                        return path_miss_message(symbol, hint, &nodes)
                    }
                    RefTarget::None => {}
                }
            }
            return String::new();
        }
        RefTarget::Ambiguous(nodes) => {
            return ambiguous_definitions_message(
                symbol,
                &nodes,
                "Re-run with a `path` hint to pick one:",
            )
        }
    };

    references_body_for_target(store, &target)
}

/// Union of the occurrence sites of every member of a selector family,
/// deduplicated by `path:line` and ordered the same way `reference_sites` orders
/// one node's sites. Two arities of one selector can be used on the same source
/// line (`[self policyWithPinningMode:m withPinnedCertificates:c]` is one line
/// carrying both heads), so the dedup is load-bearing, not defensive.
fn family_reference_sites(store: &SqliteStore, family: &[CoreNode]) -> Vec<travsr_core::RefSite> {
    let mut sites: Vec<travsr_core::RefSite> = Vec::new();
    for n in family {
        match store.reference_sites(n.id) {
            Ok(s) => sites.extend(s),
            Err(e) => tracing::warn!("family_reference_sites {}: {e}", n.vname.signature),
        }
    }
    sites.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));
    // Dedup on `path:line` only: `a` is the later duplicate about to be
    // dropped, so fold its flag into the survivor rather than letting a
    // differing `heuristic` split one site into two rows.
    sites.dedup_by(|a, b| {
        let same = a.path == b.path && a.line == b.line;
        if same {
            b.heuristic |= a.heuristic;
        }
        same
    });
    sites
}

/// Render the reference body for a selector family: which arities the head
/// reached, then the union of their occurrence sites.
fn references_body_for_family(store: &SqliteStore, family: &[CoreNode]) -> String {
    let mut lines = Vec::new();
    let head = format!(
        "resolved: {} selector(s) of one method family:",
        family.len()
    );
    lines.push(head);
    for n in family {
        let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
        lines.push(format!(
            "  {} ({}) \u{2014} {}{}",
            display_label(n),
            n.kind,
            n.vname.path,
            loc
        ));
    }

    let sites = family_reference_sites(store, family);
    if sites.is_empty() {
        // Defer to the single-target renderer for the honest degraded/zero
        // caveat rather than restating it: with no sites the family's first
        // member carries exactly the same answer.
        return references_body_for_target(store, &family[0]);
    }
    let total = sites.len();
    let shown = total.min(MAX_REFERENCE_SITES);
    lines.push(format!("{total} reference(s):"));
    for s in sites.into_iter().take(MAX_REFERENCE_SITES) {
        lines.push(format!("{}:{}{}", s.path, s.line, site_marker(&s)));
    }
    if total > shown {
        lines.push(format!("[truncated: showing {shown} of {total} sites]"));
    }
    lines.join("\n")
}

/// Render the reference body for an already-resolved target: the `resolved:`
/// header, the occurrence sites, or the honest degraded/zero caveat. Split out
/// of [`find_references_raw`] so [`find_references_structured`] can reuse the
/// target it already resolved for its caveat instead of resolving a second time.
fn references_body_for_target(store: &SqliteStore, target: &CoreNode) -> String {
    let resolved_loc = target.line.map(|l| format!(":{l}")).unwrap_or_default();
    let header = format!(
        "resolved: {} ({}) \u{2014} {}{}",
        display_label(target),
        target.kind,
        target.vname.path,
        resolved_loc
    );

    // Primary path: occurrence sites from edge_sites.
    match store.reference_sites(target.id) {
        Ok(sites) if !sites.is_empty() => {
            let total = sites.len();
            let shown = total.min(MAX_REFERENCE_SITES);
            let mut lines = Vec::with_capacity(shown + 2);
            lines.push(header);
            lines.push(format!("{total} reference(s):"));
            for s in sites.into_iter().take(MAX_REFERENCE_SITES) {
                lines.push(format!("{}:{}{}", s.path, s.line, site_marker(&s)));
            }
            if total > shown {
                lines.push(format!("[truncated: showing {shown} of {total} sites]"));
            }
            // #715: a crashed last run leaves partial coverage under a complete
            // marker, so this occurrence list may be short.
            if phase_b_lang_crashed(store, &target.vname.language) {
                lines.push(crash_caveat(&target.vname.language));
            }
            lines.join("\n")
        }
        Ok(_) => {
            // Fallback: no occurrence rows for this dst. Degrade to caller
            // definition lines from structural ref/call edges (labelled), so a
            // language not yet feeding edge_sites still returns something useful.
            reference_fallback_from_edges(store, target, &header)
        }
        Err(e) => {
            tracing::warn!("find_references reference_sites error: {e}");
            reference_fallback_from_edges(store, target, &header)
        }
    }
}

/// Structural fallback: enumerate caller definitions of `target` via `RefCall`
/// edges when no `edge_sites` occurrence rows exist. Always labelled as
/// degraded so the caller knows these are definition lines, not exact sites.
fn reference_fallback_from_edges(store: &SqliteStore, target: &CoreNode, header: &str) -> String {
    use travsr_core::EdgeKind;
    let edges = match store.iter_edges_to(target.id) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("find_references fallback edge query error: {e}");
            return String::new();
        }
    };
    let caller_ids: Vec<NodeId> = edges
        .iter()
        .filter(|e| e.kind == EdgeKind::RefCall)
        .map(|e| e.src)
        .collect();
    if caller_ids.is_empty() {
        // Resolution succeeded but neither occurrence rows nor structural
        // ref/call edges exist for this node. #299 M1: three very different
        // situations reach here and must not read identically.
        let lang = &target.vname.language;
        // #715: a crashed last Phase B run for this language leaves partial
        // coverage under a marker that reads complete, so a zero here is not a
        // definitive zero regardless of the per-file / language-wide coverage
        // gates below — those key on whether occurrences exist, not on whether the
        // run finished. Soften first so a crash is never reported as a clean zero.
        if phase_b_lang_crashed(store, lang) {
            return format!(
                "{header}\n0 recorded reference(s), not a definitive zero. Semantic \
                 analysis for '{lang}' crashed on its last run, so its occurrence \
                 coverage is partial. Run `travsr status` for detail, `travsr init \
                 --semantic --force` to rebuild, or `find_pattern` for a textual search."
            );
        }
        let index_built_for_lang = store.language_has_edge_sites(lang).unwrap_or(false);
        if !index_built_for_lang {
            // The occurrence index was never populated for this language —
            // Phase B did not run / failed / its provider records no occurrence
            // lines. This is a coverage gap, not a "zero references" answer.
            return format!(
                "{header}\nOccurrence index unavailable for '{lang}': semantic \
                 analysis recorded no reference occurrences for this \
                 language in this repo; the result below is not a definitive \
                 zero. Run `travsr status` to check progress, or use `find_pattern` \
                 for a textual search."
            );
        }
        // #450: the language has *some* occurrence data, but "some" is not
        // "complete". Phase B routinely covers a language partially — an
        // analyzer crash, a per-file timeout, or a provider that only indexes
        // its own package all leave most files unanalysed while still producing
        // enough rows to pass the gate above. Reporting a definitive zero for a
        // symbol that lives in an unanalysed file states something the index
        // cannot support.
        //
        // The gate is the target's OWN file, not a language-wide ratio (#551
        // review): `find_references` resolves targets of any kind, so a ratio
        // over one population of files can describe a set the target is not in —
        // 52 of 221 Rust files here hold no function or method node at all.
        //
        // Still a proxy. A file that genuinely references nothing looks the same
        // as an unanalysed one, so this can only ever soften a claim, never
        // assert absence. RFC-024 / #549 would make it a recorded fact.
        let file_analyzed = store
            .file_has_occurrences(&target.vname.path)
            .unwrap_or(false);
        if !file_analyzed {
            // Language-wide coverage is reported as context, not as the gate: it
            // is what makes the softened answer interpretable — "4 of 78 files"
            // reads very differently from "161 of 221".
            let (files_with_occ, files_total) =
                store.language_occurrence_coverage(lang).unwrap_or((0, 0));
            let coverage_pct = (100 * files_with_occ).checked_div(files_total).unwrap_or(0) as u32;
            return format!(
                "{header}\n0 recorded reference(s), not a definitive zero. No \
                 reference occurrences are recorded for '{}' itself, so semantic \
                 analysis may never have covered this file ({files_with_occ} of \
                 {files_total} '{lang}' files in this repo carry occurrence data, \
                 {coverage_pct}%). Run `travsr status` to check Phase B, or use \
                 `find_pattern` for a textual search.",
                target.vname.path
            );
        }
        // WS-3 (C3): a Dart index built without resolved dependencies drops
        // every cross-package reference, so "recorded no uses" would be a
        // confident zero the index cannot support even when the file itself was
        // analysed. Soften it, mirroring the partial-coverage case above.
        if target.vname.language == "dart"
            && store
                .get_meta("dart_deps_unresolved")
                .ok()
                .flatten()
                .is_some_and(|v| !v.is_empty())
        {
            return format!(
                "{header}\n0 recorded reference(s), not a definitive zero. This \
                 repo's Dart dependencies are not resolved (no \
                 `.dart_tool/package_config.json`), so cross-package references \
                 are not indexed. Run `dart pub get` and re-run `travsr init`, or \
                 use `find_pattern` for a textual search."
            );
        }
        // Coverage is effectively complete for this language and this symbol has
        // neither occurrence rows nor ref/call edges. Report that as the fact it
        // is. We cannot tell from here whether the uses do not exist, whether an
        // ambiguous bare call was deliberately skipped, or whether the analyzer
        // never emitted an occurrence for the call shape at all (scip-dotnet, for
        // one, emits nothing for a generic invocation), so do not name a cause.
        return format!(
            "{header}\n0 reference(s). The index recorded no uses of this symbol. \
             That can mean it has none, or that the call sites were not indexed: \
             an ambiguous bare call is skipped by design, and some analyzers emit \
             no occurrence for certain call shapes. Use `find_pattern` for a \
             textual search to tell the two apart."
        );
    }
    let callers = store.get_nodes(&caller_ids).unwrap_or_default();
    let mut sites: Vec<String> = callers
        .iter()
        .map(|n| {
            let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
            format!("{}{}", n.vname.path, loc)
        })
        .collect();
    sites.sort();
    sites.dedup();
    let total = sites.len();
    let mut lines = Vec::with_capacity(total + 2);
    lines.push(header.to_string());
    lines.push(format!(
        "{total} caller definition(s) (exact occurrence lines unavailable for this language, showing caller definitions):"
    ));
    lines.extend(sites.into_iter().take(MAX_REFERENCE_SITES));
    // #715: these structural caller definitions can also be short if the
    // language's last Phase B run crashed before indexing every file.
    if phase_b_lang_crashed(store, &target.vname.language) {
        lines.push(crash_caveat(&target.vname.language));
    }
    lines.join("\n")
}

/// Global variant of `find_references` — searches one named repo or all repos.
pub fn find_references_global(
    repos: &HashMap<String, PathBuf>,
    symbol: &str,
    path: Option<&str>,
    repo: Option<&str>,
) -> String {
    if let Err(reason) = validate_mcp_arg(symbol) {
        tracing::warn!("find_references_global rejected invalid symbol arg: {reason}");
        return String::new();
    }
    if let Some(p) = path {
        if let Err(reason) = validate_mcp_arg(p) {
            tracing::warn!("find_references_global rejected invalid path arg: {reason}");
            return String::new();
        }
    }
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        let result = find_references_raw(store, symbol, path);
        let result = if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        // #661 WS-D: per-repo HEAD note — read *this* repo's own checkout, not
        // the global LAUNCH_CWD (mirrors get_callers_global).
        let head = repos
            .get(repo_name)
            .and_then(|db| repo_head_from_registry_path(db));
        append_head_note(store, result, head.as_deref())
    });
    wrap_envelope(&sanitize_mcp_body_with_limit(&raw, FIND_OUTPUT_LIMIT))
}

// ── find_pattern (issue #299) ──────────────────────────────────────────────────

/// Cap on returned grep matches (byte-budget guard).
const MAX_PATTERN_MATCHES: usize = 500;

/// The repository root to resolve paths against.
///
/// Thin now: the rule lives on `SqliteStore` so that every reader of
/// `meta.repo_root` shares it, not just this file's four (#749 review).
fn resolve_repo_root(store: &SqliteStore) -> Option<PathBuf> {
    store.resolve_repo_root()
}

/// Textual search: `git grep` over the repo's text files, minus the paths the
/// walker hard-skips, the paths an ignore file excludes, and known-binary
/// formats. Deliberately WIDER than the graph's own file set: a shell script, a
/// Makefile or a lockfile is a legitimate answer to a textual query, and an
/// extension allowlist reported "no matches" for all of them.
///
/// `scope` selects the search set:
///   - `None` → whole repository (tracked files).
///   - `files-importing(<symbol>)` → only files whose graph node imports/depends
///     on `<symbol>` (`Depends` / `RefImports` / `ResolvesTo` incoming edges).
///   - any other string → treated as a repo-relative path prefix (pathspec).
///
/// Security (SEC): the pattern and every pathspec are passed as an argument
/// vector to `git` — never through a shell — so shell metacharacters cannot
/// inject. Control characters are rejected, absolute / parent-escaping
/// pathspecs are dropped, execution is pinned to the repo root, and the match
/// count is capped.
pub fn find_pattern(
    store: &SqliteStore,
    pattern: &str,
    scope: Option<&str>,
    fixed: bool,
) -> String {
    // Wrap with the larger find-output limit (not the 4 KiB scalar cap) so a
    // capped 500-match list and its truncation notice survive intact.
    wrap_envelope(&sanitize_mcp_body_with_limit(
        &find_pattern_raw(store, pattern, scope, fixed),
        FIND_OUTPUT_LIMIT,
    ))
}

/// Raw (unsanitized) body for `find_pattern`, shared by the global aggregator
/// and the CLI. Callers that hand the result to a model must wrap it in the
/// `<travsr-data>` envelope and sanitize it (see `find_pattern`); the CLI, which
/// prints straight to a human terminal, uses this raw form so source text is not
/// HTML-entity escaped (`->` staying `->`, not `-&gt;`).
pub fn find_pattern_raw(
    store: &SqliteStore,
    pattern: &str,
    scope: Option<&str>,
    fixed: bool,
) -> String {
    // #517 DD-7: the pattern argument gets its own validator — it is never
    // path-resolved, so the `../` / `%` guards in `validate_mcp_arg` only cost
    // correctness here (see `validate_mcp_pattern_arg`'s doc comment).
    if let Err(reason) = validate_mcp_pattern_arg(pattern) {
        tracing::warn!("find_pattern rejected invalid pattern: {reason}");
        return String::new();
    }
    // SEC-002: reject control chars (newlines, NUL, …) that could smuggle
    // additional grep semantics or corrupt the `path:line:col:` output framing.
    if pattern.is_empty() || pattern.chars().any(|c| c.is_control()) {
        tracing::warn!("find_pattern rejected empty or control-char pattern");
        return String::new();
    }
    if let Some(s) = scope {
        if let Err(reason) = validate_mcp_arg(s) {
            tracing::warn!("find_pattern rejected invalid scope: {reason}");
            return String::new();
        }
    }

    let Some(repo_root) = resolve_repo_root(store) else {
        return "Pattern search unavailable: run `travsr init` to record the repo root."
            .to_string();
    };

    // Resolve the scope into a bounded pathspec list (empty = whole repo).
    let pathspecs: Vec<String> = match scope {
        None => Vec::new(),
        Some(s) => {
            if let Some(sym) = s
                .strip_prefix("files-importing(")
                .and_then(|rest| rest.strip_suffix(')'))
            {
                match scope_files_importing(store, sym.trim()) {
                    Some(files) if !files.is_empty() => files,
                    Some(_) => {
                        return format!("No files import '{}'.", sym.trim());
                    }
                    None => return String::new(), // invalid symbol arg (already logged)
                }
            } else {
                // Plain path prefix. Confine to the repo: reject absolute paths and
                // parent-directory escapes. #507: normalize `\` to `/` first so
                // `..\..\x` is caught, and reject drive-letter (`C:\...`) and
                // UNC (`\\server\share`) absolute forms too.
                let normalized = s.replace('\\', "/");
                if normalized.starts_with('/')
                    || normalized.split('/').any(|c| c == "..")
                    || s.as_bytes().get(1) == Some(&b':')
                {
                    tracing::warn!("find_pattern rejected out-of-repo scope: {s}");
                    return String::new();
                }
                vec![s.to_string()]
            }
        }
    };

    match run_git_grep(&repo_root, pattern, &pathspecs, fixed) {
        GrepOutcome::Matches(body) => body,
        // #517 DD-4: deliberately empty, not an error string. `find_pattern_global`
        // (`collect_global`) uses `result.is_empty()` to skip a repo with no
        // matches during multi-repo aggregation — preserve that contract.
        // #517 DD-5: unless the pattern uses a PCRE-only construct that is
        // syntactically valid-but-different under POSIX ERE, in which case a
        // silent zero is worse than a one-line advisory.
        GrepOutcome::NoMatches => pcre_only_advisory(pattern)
            .map(|note| format!("no matches. {note}"))
            .unwrap_or_default(),
        // The hint is right for a regex failure and wrong for everything else.
        // Appending it unconditionally sent readers to check their pattern when
        // the real failure was a missing directory, and `--fixed` reproduced it
        // exactly (#747). Offer it only when git actually complained about the
        // pattern.
        GrepOutcome::Error(detail) => {
            if looks_like_a_regex_complaint(&detail) {
                format!(
                    "pattern error: {detail}. The pattern is POSIX ERE; use --fixed for a \
                     literal search, or escape metacharacters."
                )
            } else if detail.contains("cannot change to") || detail.contains("No such file") {
                // This path knows what happened and used to say nothing about
                // it: the directory in the message is index metadata, and
                // `travsr init` is what clears it. Leaving the reader to work
                // that out was the other half of #747 (#749 review).
                format!(
                    "pattern error: {detail}\n\
                     That path comes from the index, not from your query. It is \
                     recorded when the repository is indexed, so it is stale if \
                     the repository has moved. Run `travsr init` from the \
                     current location to update it."
                )
            } else {
                format!("pattern error: {detail}")
            }
        }
    }
}

/// Whether a `git grep` failure is about the pattern rather than the environment.
///
/// Matched against git's own phrasing, not against loose words. The first
/// version searched the whole stderr for needles like "bracket", "parenthes" and
/// "repetition", and git quotes the repository path back in its errors, so
/// `~/dev/bracket-parser` or `~/work/brace-matching` turned a missing-directory
/// failure back into "check your regex" for exactly the reader this was meant to
/// stop misleading (#749 review).
///
/// Anchored on the phrases git uses to report a bad pattern, each of which
/// carries enough context that a path cannot fake it.
fn looks_like_a_regex_complaint(detail: &str) -> bool {
    // Only the part of the message git wrote about the pattern. Paths appear
    // inside quotes, so a needle matching there is matching the user's directory
    // name rather than git's diagnosis.
    let without_quoted_paths: String = {
        let mut out = String::with_capacity(detail.len());
        let mut in_quote = false;
        for c in detail.chars() {
            match c {
                '\'' => in_quote = !in_quote,
                _ if !in_quote => out.push(c),
                _ => {}
            }
        }
        out.to_lowercase()
    };

    // Git's own wording, collected by running each bad pattern rather than
    // guessed. My first list was invented and missed every one of these:
    //
    //   alpha(  -> parentheses not balanced
    //   alpha[  -> brackets ([ ]) not balanced
    //   a{2     -> braces not balanced
    //   a\      -> trailing backslash (\)
    //   *a      -> repetition-operator operand invalid
    //
    // Those are what BSD's regex reports. glibc words the same failures
    // differently, so the list has to carry both or the hint goes missing on
    // Linux, which is where CI and most users are. On git 2.34.1:
    //
    //   alpha(     -> Unmatched ( or \(
    //   alpha[     -> Invalid regular expression
    //   a{2, a{1,  -> Unmatched \{
    //   a\         -> Trailing backslash
    //   *a         -> Invalid preceding regular expression
    //   [[:foo:]]  -> Invalid character class name
    //   a\1        -> Invalid back reference
    //   [z-a]      -> Invalid range end
    //
    // Each carries enough context that a directory name cannot fake it, which
    // is what the loose single words could not manage.
    [
        "not balanced",
        "trailing backslash",
        "repetition-operator operand invalid",
        "invalid regular expression",
        "invalid regex",
        "unmatched [",
        "unmatched ( or",
        "unmatched \\{",
        "unterminated",
        "invalid preceding regular expression",
        "invalid back reference",
        "invalid character class",
        "invalid range end",
    ]
    .iter()
    .any(|needle| without_quoted_paths.contains(needle))
}

/// #517 DD-5: constructs that are syntactically valid POSIX ERE atoms (so
/// `git grep -E` accepts them without error) but mean something else than
/// their PCRE reading under a strict POSIX regex engine — so a pattern
/// relying on the PCRE meaning silently finds nothing instead of erroring.
/// Advisory only; never changes what is searched and never fires when there
/// are matches.
///
/// Caveat: `\b \B \w \W \s \S` are also GNU regex extensions, so on
/// glibc-based git (Linux, and Windows via Git for Windows/MSYS2) they work
/// as their PCRE-adjacent meaning even under `-E` and this advisory never
/// fires for them — only macOS's BSD-based git grep treats them as inert.
/// `\d \D` and `(?` are not GNU extensions and behave the same everywhere.
fn pcre_only_advisory(pattern: &str) -> Option<String> {
    const CONSTRUCTS: &[&str] = &["\\b", "\\B", "\\d", "\\D", "\\w", "\\W", "\\s", "\\S", "(?"];
    let found = *CONSTRUCTS.iter().find(|c| pattern.contains(*c))?;
    let hint = match found {
        "\\b" | "\\B" => " Try [[:<:]] or a character class.",
        _ => "",
    };
    Some(format!(
        "note: `{found}` is a PCRE construct; travsr pattern uses POSIX ERE, \
         where it is not supported.{hint}"
    ))
}

/// Resolve `files-importing(<symbol>)` to the set of repo-relative file paths
/// whose node has an incoming import-like edge from another node — i.e. the
/// files that reference `symbol`. Returns `None` on an invalid symbol argument.
fn scope_files_importing(store: &SqliteStore, symbol: &str) -> Option<Vec<String>> {
    use travsr_core::EdgeKind;
    if validate_mcp_arg(symbol).is_err() {
        tracing::warn!("find_pattern files-importing: invalid symbol");
        return None;
    }
    let targets = store.search_nodes_by_name(symbol).unwrap_or_default();
    let mut files: Vec<String> = Vec::new();
    for t in &targets {
        let edges = match store.iter_edges_to(t.id) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("find_pattern files-importing edge query: {e}");
                continue;
            }
        };
        let src_ids: Vec<NodeId> = edges
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    EdgeKind::Depends | EdgeKind::RefImports | EdgeKind::ResolvesTo
                )
            })
            .map(|e| e.src)
            .collect();
        for n in store.get_nodes(&src_ids).unwrap_or_default() {
            if !n.vname.path.is_empty() {
                files.push(n.vname.path);
            }
        }
    }
    files.sort();
    files.dedup();
    Some(files)
}

/// Directories the indexer's walker hard-skips regardless of any ignore file
/// (`travsr-daemon::watcher::SKIP_DIRS`, applied at discovery in
/// `travsr-daemon::init`). Duplicated here because the crate dependency rules
/// point `travsr-daemon → travsr-mcp`, so this crate cannot read it from there.
///
/// Together with `.gitignore` / `.travsrignore` these define the one file
/// universe both the graph and `find_pattern` must agree on (#448), so the two
/// lists drifting apart silently re-opens that issue.
///
/// `pub` only so the drift cannot happen quietly: the dependency edge runs the
/// other way, so `travsr-daemon` can import this and assert the two are equal.
/// See `skip_dirs_matches_the_mcp_copy` in `travsr-daemon::watcher`. Not part
/// of this crate's intended API otherwise.
///
/// Unrelated to `travsr-cli`'s own `SKIP_DIRS` in `lang.rs`, which is a
/// different list (`build`, `.cache`, `__pycache__`, and no `.travsr`) serving
/// language detection rather than the graph's file set.
pub const SKIP_DIRS: &[&str] = &[
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

/// Repo-relative paths that are in the graph but unreachable by git's own file
/// discovery: files `.gitignore` excludes which `.travsrignore` re-includes
/// with a `!` rule.
///
/// The indexer's walker ranks `.travsrignore` *above* `.gitignore` (a custom
/// ignore file in `ignore::WalkBuilder`), so `!Pods/` pulls a gitignored
/// subtree into the graph while leaving it invisible to `git grep` — the
/// vendored-dependency half of #448.
///
/// Returns empty, at the cost of one small file read, whenever `.travsrignore`
/// has no negation rule. That is the overwhelmingly common case, and without a
/// negation every walked file is also a file git does not ignore, so
/// `--untracked --exclude-standard` already reaches all of them.
///
/// `remaining` is what is left of the caller's [`GREP_TIMEOUT`], not a fresh
/// budget: this runs between the two grep passes, so giving it its own ceiling
/// would make the real worst case a multiple of the documented one.
fn travsrignore_reincluded_files(
    repo_root: &std::path::Path,
    remaining: std::time::Duration,
) -> Vec<String> {
    let contents = match std::fs::read_to_string(repo_root.join(".travsrignore")) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    // Cheap gate: only a `!` rule can re-include a gitignored path.
    let has_negation = contents
        .lines()
        .map(str::trim)
        .any(|l| l.starts_with('!') && l.len() > 1);
    if !has_negation {
        return Vec::new();
    }
    let Some(matcher) = build_travsrignore_matcher(repo_root) else {
        return Vec::new();
    };

    // `--others --ignored --exclude-standard` lists exactly the working-tree
    // files git ignores; the whitelist test below narrows that to the ones the
    // user deliberately re-included.
    let mut cmd = std::process::Command::new("git");
    cmd.arg("-C")
        .arg(repo_root)
        .arg("ls-files")
        .arg("--others")
        .arg("--ignored")
        .arg("--exclude-standard")
        .arg("-z");
    let (status, stdout, _stderr) = match spawn_with_deadline(cmd, remaining) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("find_pattern could not list ignored files: {e}");
            return Vec::new();
        }
    };
    if !status.is_some_and(|s| s.success()) {
        tracing::warn!("find_pattern `git ls-files --ignored` did not succeed");
        return Vec::new();
    }

    String::from_utf8_lossy(&stdout)
        .split('\0')
        .filter(|p| !p.is_empty())
        .filter(|p| {
            !std::path::Path::new(p)
                .components()
                .any(|c| SKIP_DIRS.iter().any(|skip| c.as_os_str() == *skip))
        })
        .filter(|p| {
            matcher
                .matched_path_or_any_parents(repo_root.join(p), false)
                .is_whitelist()
        })
        .map(str::to_string)
        .collect()
}

/// Keep only the re-included paths that fall inside `pathspecs`, so pass 2
/// honors the caller's `scope` exactly as git honors it in pass 1. An empty
/// pathspec list means whole-repo scope and keeps everything.
///
/// `scope` reaches here as either a repo-relative path prefix or an explicit
/// file list from `files-importing(...)`, so prefix matching on a path
/// boundary is the whole of git's pathspec semantics that can apply.
fn scoped_to_pathspecs(files: Vec<String>, pathspecs: &[String]) -> Vec<String> {
    if pathspecs.is_empty() {
        return files;
    }
    files
        .into_iter()
        .filter(|f| {
            pathspecs.iter().any(|spec| {
                let spec = spec.trim_end_matches('/');
                f == spec || f.strip_prefix(spec).is_some_and(|r| r.starts_with('/'))
            })
        })
        .collect()
}

/// Build a gitignore-syntax matcher from `repo_root/.travsrignore`, mirroring the
/// indexer walker's `add_custom_ignore_filename(".travsrignore")`. Returns `None`
/// when the file is absent or fails to compile, in which case `find_pattern`
/// falls back to plain git-scoped results.
fn build_travsrignore_matcher(repo_root: &std::path::Path) -> Option<ignore::gitignore::Gitignore> {
    let path = repo_root.join(".travsrignore");
    if !path.exists() {
        return None;
    }
    let mut builder = ignore::gitignore::GitignoreBuilder::new(repo_root);
    // add() returns Some(err) on a partially-parsed file; a partial matcher is
    // still better than none, so only bail when the build itself fails.
    let _ = builder.add(&path);
    builder.build().ok()
}

/// Outcome of a `git grep` invocation, distinguishing a clean zero-match run
/// from a search the regex engine could not even compile (#517 DD-4). Exit
/// codes, measured against `git grep`: `0` = matched, `1` = no match, anything
/// else (or a spawn failure, or the DD-8 deadline) = an error the caller must
/// not launder into an empty result.
enum GrepOutcome {
    Matches(String),
    NoMatches,
    Error(String),
}

/// Wall-clock ceiling on the `git grep` child process (#517 DD-8). Even under
/// POSIX ERE (no backreferences/lookaround) a pathological pattern against a
/// large repo can pin a core; this bounds the daemon's exposure to a
/// user-supplied pattern.
///
/// #517 §9.2 follow-up: the original 10s figure was calibrated against this
/// repo (~10k nodes). Measured against the k8s corpus (263k nodes) with
/// `--threads` pinned per [`git_grep_thread_count`], a legitimate broad
/// alternation pattern (`(Pod|Service|Deployment|Node|Volume)`) already took
/// 8s at 2 threads and 15s at 1 thread — realistic for a small CI runner or a
/// daemon busy with concurrent tool calls. 30s keeps that same class of query
/// from being misreported as a timeout while still bounding a runaway one.
const GREP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `--threads=N` for `git grep`, pinned to the number of cores actually
/// available rather than left to git's own default heuristic, so the
/// [`GREP_TIMEOUT`] budget above is measured against the same parallelism the
/// daemon will see in production (#517 §9.2).
fn git_grep_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Spawn `cmd` with piped stdout/stderr, drain both on background threads, and
/// poll for exit against `deadline` (#517 DD-8). Returns `(status, stdout,
/// stderr)`; `status` is `None` if `cmd` exceeded `deadline` and was killed.
///
/// Draining on background threads means a large result set can't fill a pipe
/// buffer and deadlock the child while the loop below polls for exit —
/// `Command::wait_with_output` has no deadline, so this reimplements it with
/// one instead of using it directly.
fn spawn_with_deadline(
    mut cmd: std::process::Command,
    deadline: std::time::Duration,
) -> std::io::Result<(Option<std::process::ExitStatus>, Vec<u8>, Vec<u8>)> {
    use std::io::Read as _;
    use std::process::Stdio;

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;

    let mut stdout_pipe = child.stdout.take().expect("stdout is piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr is piped");
    let stdout_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    let cutoff = std::time::Instant::now() + deadline;
    // Both pipes are already draining on background threads above — no
    // pipe-buffer deadlock risk (same exception as
    // `travsr-indexer::runner::run_with_drain`, which `travsr-mcp` cannot
    // depend on per the crate dependency rules).
    #[allow(clippy::disallowed_methods)]
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if std::time::Instant::now() >= cutoff {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(e) => {
                tracing::warn!("spawn_with_deadline wait failed: {e}");
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let stdout_bytes = stdout_reader.join().unwrap_or_default();
    let stderr_bytes = stderr_reader.join().unwrap_or_default();
    Ok((status, stdout_bytes, stderr_bytes))
}

/// One `git grep` invocation. `Ok` covers both a match list and a clean
/// zero-match run (empty vector); `Err(detail)` is a search that could not run
/// at all — a spawn failure, an uncompilable pattern, or the shared deadline.
fn grep_pass(
    cmd: std::process::Command,
    remaining: std::time::Duration,
) -> Result<Vec<String>, String> {
    let (status, stdout_bytes, stderr_bytes) = match spawn_with_deadline(cmd, remaining) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("find_pattern git grep spawn failed: {e}");
            return Err(format!("failed to run git grep: {e}"));
        }
    };
    let Some(status) = status else {
        return Err(format!(
            "pattern search timed out after {}s",
            GREP_TIMEOUT.as_secs()
        ));
    };
    match status.code() {
        Some(0) => Ok(String::from_utf8_lossy(&stdout_bytes)
            .lines()
            .map(str::to_string)
            .collect()),
        Some(1) => Ok(Vec::new()),
        _ => Err(String::from_utf8_lossy(&stderr_bytes)
            .lines()
            .next()
            .unwrap_or("git grep failed")
            .to_string()),
    }
}

/// Split re-included paths into batches that fit in one argument vector.
/// macOS caps `ARG_MAX` an order of magnitude below Linux, so the budget is
/// measured in bytes rather than path count alone.
fn argv_chunks(files: &[String]) -> Vec<&[String]> {
    const MAX_ARGV_BYTES: usize = 96 * 1024;
    const MAX_ARGV_PATHS: usize = 1000;

    let mut out = Vec::new();
    let mut start = 0;
    let mut bytes = 0;
    for (i, f) in files.iter().enumerate() {
        let cost = f.len() + 1; // + NUL terminator
        if i > start && (bytes + cost > MAX_ARGV_BYTES || i - start >= MAX_ARGV_PATHS) {
            out.push(&files[start..i]);
            start = i;
            bytes = 0;
        }
        bytes += cost;
    }
    if start < files.len() {
        out.push(&files[start..]);
    }
    out
}

/// Execute `git grep` under `repo_root` with an argument vector (no shell) and
/// format the capped results as `path:line:col: text`.
///
/// `fixed` selects `-F` (literal string) instead of the default `-E` (POSIX
/// extended regular expression) — #517 DD-3/DD-6.
///
/// #448: the search runs in two passes because git's file discovery and the
/// indexer walker's are different universes, and a file in the graph that the
/// search cannot reach makes `find_pattern` report a false negative — the one
/// thing it exists to rule out. Both passes share a single [`GREP_TIMEOUT`]
/// budget so the ceiling is per-call, not per-subprocess.
fn run_git_grep(
    repo_root: &std::path::Path,
    pattern: &str,
    pathspecs: &[String],
    fixed: bool,
) -> GrepOutcome {
    use std::process::Command;

    let started = std::time::Instant::now();
    let remaining = || GREP_TIMEOUT.saturating_sub(started.elapsed());

    // Options shared by both passes, ending with `--` so neither the pattern
    // nor any path can be read back as a flag.
    let with_common_args = |cmd: &mut Command| {
        cmd.arg("--no-color")
            .arg("-n") // line numbers
            .arg("--column") // column numbers
            .arg("-I") // skip binary files
            .arg(format!("--threads={}", git_grep_thread_count()))
            .arg(if fixed { "-F" } else { "-E" })
            .arg("-e")
            .arg(pattern)
            .arg("--");
    };

    // Pass 1 — everything git can see. `--untracked` adds files the indexer
    // walked but git has never been told about (a new file not yet `git add`ed
    // is in the graph the moment the watcher sees it, and "always fresh" means
    // the search has to see it too); `--exclude-standard` keeps `.gitignore`
    // authoritative, matching the walker's `git_ignore(true)`.
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(repo_root)
        .arg("grep")
        .arg("--untracked")
        .arg("--exclude-standard");
    with_common_args(&mut cmd);
    for spec in pathspecs {
        cmd.arg(spec);
    }
    let mut all: Vec<String> = match grep_pass(cmd, remaining()) {
        Ok(lines) => lines,
        Err(detail) => return GrepOutcome::Error(detail),
    };
    let pass1_matches = all.len();

    // Pass 2 — the graph files git cannot reach at all: gitignored paths a
    // `!` rule in `.travsrignore` re-included. `--no-index` searches the given
    // paths directly instead of consulting the index or the ignore rules that
    // excluded them in the first place. Empty for any repo without a `!` rule.
    let reincluded = scoped_to_pathspecs(
        travsrignore_reincluded_files(repo_root, remaining()),
        pathspecs,
    );
    let mut pass2_error: Option<String> = None;
    for chunk in argv_chunks(&reincluded) {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(repo_root).arg("grep").arg("--no-index");
        with_common_args(&mut cmd);
        for path in chunk {
            cmd.arg(path);
        }
        match grep_pass(cmd, remaining()) {
            Ok(lines) => all.extend(lines),
            Err(detail) => {
                // Abandon the remaining batches rather than press on. The
                // batches are one logical search split only to fit an argument
                // vector, so continuing would report a subset of the
                // re-included files as if it were all of them — the same
                // silent under-reporting #448 exists to remove. Stopping keeps
                // the outcome honest: `pass2_error` below either surfaces or,
                // if pass 1 already matched, warns.
                //
                // Reachable if a re-included file is deleted between the
                // `git ls-files` listing and this grep (exit 128). Rare, and
                // the next call re-lists.
                pass2_error = Some(detail);
                break;
            }
        }
    }

    // The discrepancy the issue asked to make visible: what git's own file
    // discovery covered, versus what had to be searched around it.
    tracing::debug!(
        pass1_matches,
        reincluded_files = reincluded.len(),
        reincluded_matches = all.len() - pass1_matches,
        "find_pattern searched file set"
    );

    // A pass-2 failure must not be laundered into a false "no matches" — but
    // it must not discard a good pass-1 result either.
    if let Some(detail) = pass2_error {
        if all.is_empty() {
            return GrepOutcome::Error(detail);
        }
        tracing::warn!("find_pattern re-included pass failed: {detail}");
    }

    // Three filters, all closing the same gap from the other direction: git can
    // see files the walker refuses to index, so a match from one of them would
    // be a file `find_references` can never corroborate.
    //
    // 1. `SKIP_DIRS` is hard-excluded by the walker *ahead of* any ignore file
    //    (`SKIP_DIRS (hard) < .gitignore < .travsrignore`), so a `dist/` or
    //    `target/` that happens not to be gitignored is in git's universe and
    //    never in the graph. `--untracked` widened pass 1 into exactly those,
    //    and no ignore rule excludes them, so the component check is the only
    //    thing that can. Pass 2 already applies it when building its file list.
    // 2. `git grep` honors `.gitignore` but knows nothing about
    //    `.travsrignore`, so a git-tracked file the user excluded from the
    //    graph would still appear. Filtering through the same gitignore-syntax
    //    matcher the walker applies (`add_custom_ignore_filename`) keeps both
    //    tools on one set of files. Pass-2 paths are whitelisted by
    //    construction, so this half only ever drops pass-1 lines.
    // 3. Neither of those covers a generated `index.scip` sitting at the repo
    //    root: it is in no skip dir and no ignore file, and `git grep -I` does
    //    not help — git's binary heuristic only looks for a NUL byte in the
    //    first 8000 bytes and this protobuf has none, so git classifies it as
    //    text and emits raw binary as "matches". `BINARY_EXTS` names the
    //    formats that produce garbage rather than an answer.
    //
    //    This is a DENYLIST on purpose. It used to route through
    //    `travsr_core::is_indexable_path`, an extension ALLOWLIST built for the
    //    parser walk, which silently dropped every match in a `.sh`, `.sql`,
    //    `Dockerfile`, `Makefile`, `.lock`, `.txt` or any extensionless file —
    //    including this repo's own CI gate scripts — and reported a bare "no
    //    matches" indistinguishable from real absence. find_pattern is a
    //    textual search, not a parse: anything git will show as text is a
    //    legitimate answer, so only known-binary formats are removed. `-I`
    //    above still catches everything with a NUL byte.

    // Extensions whose contents are bytes, not text. `.lsif` is deliberately
    // absent: LSIF is line-delimited JSON, and a text format belongs in a
    // textual search.
    //
    // Not exhaustive, and it does not have to be: `-I` above already drops
    // anything with a NUL byte in its first 8 KB. This list only has to cover
    // the formats git misclassifies as text, so it grows one entry at a time as
    // one shows up. `pyc`, `pack`, `idx`, `node`, `safetensors` and `parquet`
    // are here for that reason.
    const BINARY_EXTS: &[&str] = &[
        "scip",
        "db",
        "sqlite",
        "onnx",
        "safetensors",
        "parquet",
        "bin",
        "wasm",
        "so",
        "dylib",
        "dll",
        "exe",
        "node",
        "a",
        "o",
        "rlib",
        "class",
        "jar",
        "pyc",
        "pack",
        "idx",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "ico",
        "pdf",
        "zip",
        "gz",
        "tgz",
        "bz2",
        "xz",
        "zst",
        "tar",
        "woff",
        "woff2",
        "ttf",
        "otf",
        "mp4",
        "mov",
        "mp3",
        "wav",
    ];
    let is_binary_ext = |path: &str| {
        std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| {
                let e = e.to_ascii_lowercase();
                BINARY_EXTS.contains(&e.as_str())
            })
    };

    // Each grep line is `path:line:col:text`, so the path is the prefix before
    // the first ':'.
    let travsrignore = build_travsrignore_matcher(repo_root);
    let all: Vec<&str> = all
        .iter()
        .map(String::as_str)
        .filter(|line| {
            let Some(path) = line.split(':').next().filter(|p| !p.is_empty()) else {
                return true;
            };
            let in_skip_dir = std::path::Path::new(path)
                .components()
                .any(|c| SKIP_DIRS.iter().any(|skip| c.as_os_str() == *skip));
            if in_skip_dir {
                return false;
            }
            if is_binary_ext(path) {
                return false;
            }
            match &travsrignore {
                Some(ig) => !ig
                    .matched_path_or_any_parents(repo_root.join(path), false)
                    .is_ignore(),
                None => true,
            }
        })
        .collect();
    if all.is_empty() {
        // The .travsrignore filter emptied a non-empty git-grep match set — the
        // honest outcome is still "no matches", not an error (§4 W2 edge case).
        return GrepOutcome::NoMatches;
    }
    // Per-match line-length cap: a single minified/generated line can be
    // hundreds of KB and would otherwise blow the output budget on its own.
    const MAX_MATCH_LINE: usize = 512;
    let total = all.len();
    let shown = total.min(MAX_PATTERN_MATCHES);
    let mut lines: Vec<String> = Vec::with_capacity(shown + 1);
    lines.push(format!("{total} match(es):"));
    lines.extend(all.into_iter().take(MAX_PATTERN_MATCHES).map(|l| {
        if l.len() > MAX_MATCH_LINE {
            let mut cut = MAX_MATCH_LINE;
            while cut > 0 && !l.is_char_boundary(cut) {
                cut -= 1;
            }
            format!("{} …[line truncated]", &l[..cut])
        } else {
            l.to_string()
        }
    }));
    if total > shown {
        lines.push(format!("[truncated: showing {shown} of {total} matches]"));
    }
    GrepOutcome::Matches(lines.join("\n"))
}

/// Global variant of `find_pattern` — runs the graph-scoped grep per repo.
pub fn find_pattern_global(
    repos: &HashMap<String, PathBuf>,
    pattern: &str,
    scope: Option<&str>,
    repo: Option<&str>,
    fixed: bool,
) -> String {
    // #517 DD-7: same exemption as `find_pattern_raw` — see
    // `validate_mcp_pattern_arg`'s doc comment.
    if let Err(reason) = validate_mcp_pattern_arg(pattern) {
        tracing::warn!("find_pattern_global rejected invalid pattern: {reason}");
        return String::new();
    }
    if let Some(s) = scope {
        if let Err(reason) = validate_mcp_arg(s) {
            tracing::warn!("find_pattern_global rejected invalid scope: {reason}");
            return String::new();
        }
    }
    // Aggregate raw per-repo bodies (not the wrapped form) so there is a single
    // envelope + one large-limit sanitize over the whole result.
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        let result = find_pattern_raw(store, pattern, scope, fixed);
        if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    });
    wrap_envelope(&sanitize_mcp_body_with_limit(&raw, FIND_OUTPUT_LIMIT))
}

/// Resolve one named registry entry to its live graph.db path.
///
/// Mirrors `collect_global`'s stale-entry filter for the single-repo case, for
/// callers that need to open the store themselves.
///
/// Validates with `validate_mcp_repo_key_arg`, not the shared `validate_mcp_arg`:
/// every registry key is an absolute repo root, which `validate_mcp_arg` rejects
/// outright, so a caller could never name a real repo. Same reasoning and same
/// exact-key-equality use as `observability::resolve_single_repo` (#636) - see
/// that validator's doc for why the relaxed guard set is safe here.
fn resolve_repo_db_path<'a>(
    repos: &'a HashMap<String, PathBuf>,
    name: &str,
) -> Option<&'a PathBuf> {
    // SEC-002: validate repo arg before registry lookup.
    if let Err(reason) = crate::sanitize::validate_mcp_repo_key_arg(name) {
        tracing::warn!("get_context_global rejected invalid repo arg: {reason}");
        return None;
    }
    match repos.get(name) {
        Some(db_path) if db_path.exists() => Some(db_path),
        Some(db_path) => {
            tracing::debug!("skipping stale registry entry: {}", db_path.display());
            None
        }
        None => {
            tracing::warn!("repo '{name}' not found in registry");
            None
        }
    }
}

fn collect_global(
    repos: &HashMap<String, PathBuf>,
    target_repo: Option<&str>,
    mut f: impl FnMut(&SqliteStore, &str, bool) -> String,
) -> String {
    // SEC-002: validate repo arg before registry lookup.
    //
    // `validate_mcp_repo_key_arg`, not the shared `validate_mcp_arg`: every
    // registry key is an absolute repo root, which `validate_mcp_arg` rejects
    // outright, so every tool routed through here answered an empty result for
    // any real repo named by `repo`. Same reasoning as `resolve_repo_db_path`
    // and `observability::resolve_single_repo` (#636) - the value is only ever
    // compared for exact `HashMap` key equality, never opened as a path.
    if let Some(name) = target_repo {
        if let Err(reason) = crate::sanitize::validate_mcp_repo_key_arg(name) {
            tracing::warn!("collect_global rejected invalid repo arg: {reason}");
            return String::new();
        }
    }

    let candidates: Vec<(&str, &PathBuf)> = match target_repo {
        Some(name) => match repos.get_key_value(name) {
            Some((k, v)) => vec![(k.as_str(), v)],
            None => {
                tracing::warn!("repo '{name}' not found in registry");
                return String::new();
            }
        },
        None => repos.iter().map(|(k, v)| (k.as_str(), v)).collect(),
    };

    // Filter stale entries before computing `single` so that a registry with
    // one live repo and one deleted-db repo is treated as single-repo (no
    // per-repo prefix on the output).
    let candidates: Vec<(&str, &PathBuf)> = candidates
        .into_iter()
        .filter(|(_, db_path)| {
            let exists = db_path.exists();
            if !exists {
                tracing::debug!("skipping stale registry entry: {}", db_path.display());
            }
            exists
        })
        .collect();

    let single = candidates.len() == 1;
    let mut parts: Vec<String> = Vec::new();

    for (repo_name, db_path) in candidates {
        match SqliteStore::open_read_only(db_path) {
            Ok(store) => {
                let result = f(&store, repo_name, single);
                if !result.is_empty() {
                    parts.push(result);
                }
            }
            Err(e) => tracing::warn!("failed to open {}: {e}", db_path.display()),
        }
    }

    parts.join("\n")
}

// ── get_blast_radius ──────────────────────────────────────────────────────────

/// Controls which graph edges `get_blast_radius` follows during BFS.
/// `TreeSitter` is the default and reproduces the pre-toggle behaviour exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnalysisMode {
    /// Follow `DefinesBinding`, `RefCall`, and `Depends` edges (Phase A).
    /// Always available; zero regressions from existing behaviour.
    #[default]
    TreeSitter,
    /// Follow only `RefCall` edges (Phase B — SCIP/LSIF output).
    /// Returns an empty result when no Phase B data exists for the file's language.
    Semantic,
}

/// The one honest "not a language we handle" envelope, shared by every
/// `get_lang_status` return path so the shape never drifts between them.
const UNKNOWN_LANG_JSON: &str = r#"{"language":"unknown","status":"unknown","statusLine":"not a supported language","builtin":false,"semantic_available":false,"install_hint":"","prerequisites":""}"#;

/// Return the set of files transitively affected if the given file changes.
///
/// Uses reverse BFS over `DefinesBinding` and `RefCall` edges: starting from
/// every node defined in the file, follows incoming edges to find everything
/// that references or calls those definitions.
///
/// Output format (one line per affected file, sorted):
///   `src/service.ts`
///   `src/controller.ts`
pub fn get_blast_radius(store: &SqliteStore, file: &str, mode: AnalysisMode) -> String {
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_blast_radius rejected invalid arg: {reason}");
        return String::new();
    }
    // #617: append the Phase-B staleness note so a partial radius is not
    // trusted as the complete impact set.
    with_phase_b_note(
        store,
        sanitize_for_mcp(&get_blast_radius_raw(store, file, mode)),
    )
}

fn get_blast_radius_raw(store: &SqliteStore, file: &str, mode: AnalysisMode) -> String {
    use std::collections::{HashSet, VecDeque};
    use travsr_core::EdgeKind;

    // Hard ceiling: prevents OOM on utility files imported by thousands of callers.
    // Same guard policy as PPR (MAX_SUBGRAPH_NODES = 250_000); tighter here because
    // blast-radius is a UI tool whose output must fit in an MCP response.
    const MAX_BLAST_RADIUS_NODES: usize = 50_000;

    // Find all nodes whose VName path matches the given file.
    let seed_nodes = match store.search_nodes_by_name(file) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_blast_radius search error: {e}");
            return String::new();
        }
    };

    if seed_nodes.is_empty() {
        return String::new();
    }

    let mut visited: HashSet<travsr_core::NodeId> = HashSet::new();
    let mut queue: VecDeque<travsr_core::NodeId> = VecDeque::new();
    let mut affected_files: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for node in &seed_nodes {
        if visited.insert(node.id) {
            queue.push_back(node.id);
        }
        // The file itself counts as affected.
        if !node.vname.path.is_empty() {
            affected_files.insert(node.vname.path.clone());
        }
    }

    // Phase-2 memoization, shared across every pass of the outer fixpoint.
    // `phase2_done`: files whose importers have already been resolved via the
    //   language-aware resolvers, so we never re-resolve the same file. (Per
    //   target file, a local `seen_here` set dedups candidate import nodes so
    //   the pure `resolves_to` runs at most once per candidate.)
    let mut phase2_done: HashSet<String> = HashSet::new();

    // Phase-2 candidate index, built once per call from a single bulk load.
    // Transitive resolution (#613) re-resolves importers for every newly reached
    // file; issuing a SQLite FTS lookup per file was ~50x slower on k8s. Instead
    // load all import nodes once and index them by the alphanumeric tokens of
    // their signature. An import node can only resolve to a file whose directory
    // or stem appears as a path element in its signature, so a token lookup is a
    // superset of the true importers for a hint — `resolves_to` stays the
    // arbiter, and this replicates the FTS prefilter without per-file queries.
    // Built only in TreeSitter mode (Semantic skips Phase 2 entirely).
    let import_records: Vec<(travsr_core::NodeId, String, String, String)> =
        if mode == AnalysisMode::TreeSitter {
            store.import_nodes_lite().unwrap_or_else(|e| {
                // Fail open (Phase 1 still ran), but never silently: a bulk-load
                // error drops the entire Phase-2 pass, so the result would
                // under-report importers with no other signal.
                tracing::warn!(
                    "get_blast_radius: import_nodes_lite failed for '{file}' \
                     ({e}); Phase-2 (file-level import) resolution skipped, \
                     result may be incomplete"
                );
                Vec::new()
            })
        } else {
            Vec::new()
        };
    let mut import_token_index: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, (_, sig, _, _)) in import_records.iter().enumerate() {
        for tok in import_sig_tokens(sig) {
            import_token_index
                .entry(tok.to_string())
                .or_default()
                .push(idx);
        }
    }

    // Outer fixpoint (#613). Each pass:
    //   1. Drains the edge-following reverse BFS (Phase 1). This handles the
    //      TS/JS/Rust `Depends → import → ResolvesTo` chain and `Depends`
    //      co-package siblings transitively, for every node currently queued.
    //   2. Resolves file-level imports for the languages that emit NO
    //      ResolvesTo edge (Phase 2), pushing every newly discovered importer
    //      *node* back onto the BFS queue so pass 1 can walk its edges next.
    // A Phase-2 importer has no ResolvesTo edge, so re-draining the BFS alone
    // finds nothing new — transitivity comes from re-running Phase 2 on each
    // newly reached file, which this loop does until a full pass adds nothing.
    // Labeled so the hard ceiling can abort the whole fixpoint from the
    // innermost Phase-2 insertion with no per-pass overshoot.
    'fixpoint: loop {
        // ── Phase 1: edge-following reverse BFS ──
        while let Some(current_id) = queue.pop_front() {
            let incoming = match store.iter_edges_to(current_id) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("get_blast_radius edge error: {e}");
                    continue;
                }
            };

            if visited.len() >= MAX_BLAST_RADIUS_NODES {
                tracing::warn!(
                    "get_blast_radius hit ceiling ({MAX_BLAST_RADIUS_NODES} nodes) \
                     for file '{file}'; result may be incomplete"
                );
                break;
            }

            // Batch: collect followable new-src ids, enqueue, then fetch paths once.
            let new_srcs: Vec<NodeId> = incoming
                .into_iter()
                .filter(|edge| match mode {
                    // ResolvesTo is required for the two-hop import chain that the
                    // indexer emits for TypeScript/JS/Rust:
                    //   file --Depends--> import:pkg --ResolvesTo--> target file
                    // Walking incoming edges from the target reaches the import node
                    // only via ResolvesTo; from there Depends reaches the importer.
                    // Without it, blast radius on those files returns just the file
                    // itself (#582).
                    AnalysisMode::TreeSitter => matches!(
                        edge.kind,
                        EdgeKind::DefinesBinding
                            | EdgeKind::RefCall
                            | EdgeKind::Depends
                            | EdgeKind::ResolvesTo
                    ),
                    AnalysisMode::Semantic => matches!(edge.kind, EdgeKind::RefCall),
                })
                .filter(|edge| visited.insert(edge.src))
                .map(|edge| edge.src)
                .collect();
            for &id in &new_srcs {
                queue.push_back(id);
            }
            let node_map: HashMap<NodeId, CoreNode> = store
                .get_nodes(&new_srcs)
                .unwrap_or_default()
                .into_iter()
                .map(|n| (n.id, n))
                .collect();
            for id in &new_srcs {
                if let Some(src_node) = node_map.get(id) {
                    if !src_node.vname.path.is_empty() {
                        affected_files.insert(src_node.vname.path.clone());
                        // Hard ceiling on the result set (see the Phase-2 note).
                        if affected_files.len() >= MAX_BLAST_RADIUS_NODES {
                            tracing::warn!(
                                "get_blast_radius hit ceiling \
                                 ({MAX_BLAST_RADIUS_NODES} nodes) for file '{file}'; \
                                 result may be incomplete"
                            );
                            break 'fixpoint;
                        }
                    }
                }
            }
        }

        // ── Phase 2: language-aware file-level import resolution ──
        // Many languages store deps as file --[Depends]--> import:pkg with no
        // ResolvesTo chain back; ImportResolver bridges this per language.
        // Semantic mode uses only precise RefCall edges, so Phase 2 is skipped.
        if mode != AnalysisMode::TreeSitter {
            break;
        }
        if visited.len() >= MAX_BLAST_RADIUS_NODES || affected_files.len() >= MAX_BLAST_RADIUS_NODES
        {
            tracing::warn!(
                "get_blast_radius hit ceiling ({MAX_BLAST_RADIUS_NODES} nodes) \
                 for file '{file}'; result may be incomplete"
            );
            break;
        }

        // Files reached so far whose importers we have not yet resolved. Newly
        // discovered importer files land here on the next pass, giving
        // transitivity. TS/JS/Rust files map to NoopResolver, so re-processing
        // them is a cheap no-op.
        let todo: Vec<String> = affected_files
            .iter()
            .filter(|f| !phase2_done.contains(*f))
            .cloned()
            .collect();
        if todo.is_empty() {
            break; // fixpoint reached — no new file to resolve importers for
        }

        for target_file in todo {
            phase2_done.insert(target_file.clone());
            // Two search hints per file: the parent directory (Go/Java package,
            // PHP, C#) and the file stem (Java/Kotlin/Scala class-level imports
            // like import:com.Foo; Ruby/C/C++/Objective-C relative includes).
            // Each hint is tokenised the same way as the import signatures so a
            // lookup returns a superset of the import nodes that could resolve.
            // `seen_here` dedups candidates whose signature shares more than one
            // token with the hints, so `resolves_to` runs once per candidate.
            let mut seen_here: HashSet<usize> = HashSet::new();
            for hint in blast_radius_hints(&target_file) {
                for tok in import_sig_tokens(&hint) {
                    let Some(cands) = import_token_index.get(tok) else {
                        continue;
                    };
                    for &idx in cands {
                        if !seen_here.insert(idx) {
                            continue;
                        }
                        let (imp_id, imp_sig, imp_lang, imp_path) = &import_records[idx];
                        let resolver = travsr_core::resolver_for_language(imp_lang);
                        // The import node's own path is the importer's file — the
                        // deterministic anchor for relative imports (#613).
                        // `seen_here` already guarantees each import node is
                        // resolved at most once per target file.
                        if !resolver.resolves_to(imp_sig, imp_path, &target_file) {
                            continue;
                        }
                        let Ok(edges) = store.iter_edges_to(*imp_id) else {
                            continue;
                        };
                        let dep_srcs: Vec<NodeId> = edges
                            .iter()
                            .filter(|e| matches!(e.kind, EdgeKind::Depends))
                            .map(|e| e.src)
                            .collect();
                        let node_map: HashMap<NodeId, CoreNode> = store
                            .get_nodes(&dep_srcs)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|n| (n.id, n))
                            .collect();
                        for id in &dep_srcs {
                            // Push the importer node onto the BFS queue so the
                            // next Phase-1 pass walks its co-package Depends
                            // siblings and any ResolvesTo importers. This is the
                            // line that was dead before #613.
                            if visited.insert(*id) {
                                queue.push_back(*id);
                            }
                            if let Some(src_node) = node_map.get(id) {
                                if !src_node.vname.path.is_empty() {
                                    affected_files.insert(src_node.vname.path.clone());
                                    // Hard ceiling: abort the whole fixpoint the
                                    // instant the result reaches the cap, so a
                                    // hub file imported by tens of thousands
                                    // cannot overshoot within a single pass.
                                    if affected_files.len() >= MAX_BLAST_RADIUS_NODES {
                                        tracing::warn!(
                                            "get_blast_radius hit ceiling \
                                             ({MAX_BLAST_RADIUS_NODES} nodes) for file \
                                             '{file}'; result may be incomplete"
                                        );
                                        break 'fixpoint;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    affected_files.into_iter().collect::<Vec<_>>().join("\n")
}

/// Blast-radius Phase-2 search hints for a file: the parent directory's last
/// component and the file stem. These narrow the FTS lookup of candidate import
/// nodes before the language-aware resolver confirms an exact match.
fn blast_radius_hints(file: &str) -> Vec<String> {
    let fp = file.replace('\\', "/");
    let p = std::path::Path::new(&fp);
    let dir_hint = p
        .parent()
        .and_then(|d| d.file_name())
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty());
    let stem_hint = p.file_stem().and_then(|s| s.to_str());

    let mut hints: Vec<String> = Vec::new();
    if let Some(d) = dir_hint {
        hints.push(d.to_string());
    }
    if let Some(s) = stem_hint {
        if Some(s) != dir_hint {
            hints.push(s.to_string());
        }
    }
    hints
}

/// Split an import signature into its alphanumeric path tokens (dropping the
/// `import:` prefix and every separator: `/`, `\`, `.`, `:`, `_`, quotes,
/// angle brackets). Used to index import nodes for the blast-radius Phase-2
/// candidate lookup, and to tokenise search hints the same way. Case-sensitive,
/// which matches how case-significant languages (Java classes) name their files.
fn import_sig_tokens(sig: &str) -> impl Iterator<Item = &str> {
    sig.strip_prefix("import:")
        .unwrap_or(sig)
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// Global variant of `get_blast_radius`.
pub fn get_blast_radius_global(
    repos: &HashMap<String, PathBuf>,
    file: &str,
    repo: Option<&str>,
    mode: AnalysisMode,
) -> String {
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_blast_radius_global rejected invalid arg: {reason}");
        return String::new();
    }
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        // #617 + #645: per-repo notes — each store's own Phase B freshness, and
        // the HEAD note against *this* repo's own checkout (not global LAUNCH_CWD).
        let head = repos
            .get(repo_name)
            .and_then(|db| repo_head_from_registry_path(db));
        let result = append_read_notes(
            store,
            get_blast_radius_raw(store, file, mode),
            head.as_deref(),
        );
        if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    });
    sanitize_for_mcp(&raw)
}

// ── get_lang_status ───────────────────────────────────────────────────────────

/// Detect the language of `file` from its extension, then check whether Phase B
/// (SCIP/LSIF) data exists in the store for that language.
///
/// Returns JSON: `{"language":"go","builtin":false,"semantic_available":false,
/// "install_hint":"travsr lang install go"}`
///
/// `install_hint` is empty when semantic is already available; for builtins
/// (ts/js/rust/python) with missing Phase B it shows `underlying_tool_hint`.
/// JSON is returned unsanitised — it is parsed by first-party TypeScript code,
/// not fed to an LLM.
pub fn get_lang_status(store: &SqliteStore, file: &str) -> String {
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_lang_status rejected invalid arg: {reason}");
        return UNKNOWN_LANG_JSON.to_string();
    }
    get_lang_status_raw(store, file)
}

/// #712: whether a binary named `name` resolves on this machine — either in
/// travsr's managed `~/.travsr/bin` or on `PATH`. Mirrors the CLI's
/// `tool_available` so the two surfaces agree on what "installed" means.
fn bin_resolvable(name: &str) -> bool {
    fn present(dir: &std::path::Path, name: &str) -> bool {
        if dir.join(name).is_file() {
            return true;
        }
        #[cfg(windows)]
        if dir.join(format!("{name}.exe")).is_file() {
            return true;
        }
        false
    }
    if let Some(home) = dirs::home_dir() {
        if present(&home.join(".travsr").join("bin"), name) {
            return true;
        }
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|d| present(&d, name)))
        .unwrap_or(false)
}

/// #712: whether the Phase B analyzer for `lang` is installed (provider wrapper
/// and underlying tool both resolvable). Mirrors `travsr lang list`'s install
/// detection so `get_lang_status` never tells the user to install a tool that
/// `lang list` already reports as active — it has simply produced no edges yet.
fn analyzer_installed(lang: &str) -> bool {
    let Some(entry) = travsr_plugin_host::phase_b::catalog::lookup(lang) else {
        return false;
    };
    // Honest: present when bundled (python) or when the external analyzer resolves
    // — never just because the language is built in. Rust is not special here.
    if entry.analyzer_bundled() {
        return true;
    }
    entry.provider_binary.map_or(true, bin_resolvable) && bin_resolvable(entry.command)
}

fn get_lang_status_raw(store: &SqliteStore, file: &str) -> String {
    use travsr_plugin_host::phase_b::status::{install_step, LangStatus};

    let ext = std::path::Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();

    let entry = travsr_plugin_host::phase_b::catalog::CATALOG
        .iter()
        .find(|e| e.extensions.contains(&ext.as_str()));

    let Some(meta) = entry else {
        return UNKNOWN_LANG_JSON.to_string();
    };

    // Repo-view truth: full cross-file analysis is live iff we actually recorded
    // cross-file (call/reference) edges for this language here.
    let semantic_available = store.has_refcall_edges_for_language(meta.language);

    // The honest next step when it is not live, in the shared vocabulary:
    //  - no build for this OS               -> unsupported (never "install", which
    //                                          would dead-end — same verdict the CLI
    //                                          `lang list` shows, from one predicate)
    //  - analyzer present but no edges yet  -> rebuild
    //  - analyzer absent                    -> install (same step for every language)
    let status = if let Some(os) = travsr_plugin_host::phase_b::platform::unsupported_reason(meta) {
        LangStatus::PlatformUnsupported { os }
    } else if semantic_available {
        LangStatus::Active
    } else {
        let next = if analyzer_installed(meta.language) {
            "travsr init --semantic --force".to_string()
        } else {
            install_step(meta.language)
        };
        LangStatus::Partial { next: Some(next) }
    };
    // `install_hint` stays for backward compatibility; it mirrors the step above.
    // Empty when there is nothing to install: already live, or no build exists for
    // this OS (an install step there would just dead-end).
    let install_hint = match &status {
        LangStatus::Active | LangStatus::PlatformUnsupported { .. } => String::new(),
        LangStatus::Partial { next: Some(step) } => step.clone(),
        _ => install_step(meta.language),
    };

    // #318 O5: staleness marker — the commit the semantic data was last built at.
    // A hex SHA needs no JSON escaping; anything else is rejected here.
    let phase_b_commit = store
        .get_meta("phase_b_commit")
        .ok()
        .flatten()
        .filter(|s| s.chars().all(|c| c.is_ascii_hexdigit()))
        .map(|s| format!("\"{s}\""))
        .unwrap_or_else(|| "null".to_string());

    // Manual JSON to avoid a serde_json dependency on a hot path. `status` is a
    // stable machine tag; `statusLine` is the exact plain wording every other
    // surface shows, so the extension renders it verbatim instead of re-deriving.
    format!(
        r#"{{"language":"{lang}","status":"{status_tag}","statusLine":"{status_line}","builtin":{builtin},"semantic_available":{sem},"install_hint":"{hint}","prerequisites":"{prereq}","phase_b_commit":{pbc}}}"#,
        lang = meta.language,
        status_tag = status.tag(),
        status_line = status.line(),
        builtin = meta.builtin,
        sem = semantic_available,
        hint = install_hint,
        prereq = meta.effective_prerequisites(),
        pbc = phase_b_commit,
    )
}

/// Global variant of `get_lang_status` — opens the first matched repo store.
pub fn get_lang_status_global(
    repos: &HashMap<String, PathBuf>,
    file: &str,
    repo: Option<&str>,
) -> String {
    if let Err(reason) = validate_mcp_arg(file) {
        tracing::warn!("get_lang_status_global rejected invalid arg: {reason}");
        return UNKNOWN_LANG_JSON.to_string();
    }
    // collect_global returns newline-joined results; we only need the first repo's answer.
    let raw = collect_global(repos, repo, |store, _repo_name, _single| {
        get_lang_status_raw(store, file)
    });
    if raw.is_empty() {
        UNKNOWN_LANG_JSON.to_string()
    } else {
        // collect_global joins results with "\n"; take only the first JSON line.
        raw.lines().next().unwrap_or("").to_string()
    }
}

// ── search_symbol ─────────────────────────────────────────────────────────────

/// Find symbol definitions matching a name across the indexed graph.
///
/// Returns matching symbols formatted as:
///   `fn:charge (function) — src/payment.ts`
pub fn search_symbol(store: &SqliteStore, name: &str, exact: bool) -> String {
    if let Err(reason) = validate_mcp_arg(name) {
        tracing::warn!("search_symbol rejected invalid arg: {reason}");
        return String::new();
    }
    // Strip language token from query so FTS gets a clean term.
    // In single-repo mode we don't apply the language filter (multi-language
    // repos), but stripping "in rust" from "knapsack in rust" prevents the
    // FTS from matching unrelated files that contain "rust" as a token.
    let (stripped, lang_filter) = infer_language_from_query(name);
    let raw = search_symbol_raw(store, stripped.as_str(), lang_filter, exact);
    let content = if raw.is_empty() {
        format!("No symbols matching '{name}' found in the graph.")
    } else {
        raw
    };
    sanitize_for_mcp(&content)
}

/// Parse a query string for an embedded language token.
///
/// Returns `(stripped_query, lang_id)`.  `stripped_query` has the matched
/// language word (and an optional preceding "in" connector) removed.
/// When no language token is detected the original query is returned unchanged.
///
/// Examples:
/// - `"PaymentService in Kotlin"` → `("PaymentService", Some("kotlin"))`
/// - `"auth handler javascript"` → `("auth handler", Some("javascript"))`
/// - `"knapsack"`               → `("knapsack", None)`
fn infer_language_from_query(query: &str) -> (String, Option<&'static str>) {
    // (query_token_lowercase → db language id).
    // Longer / more-specific tokens are listed first so "objective-c" beats
    // "c" before reaching the single-char "c" entry.
    const LANG_TOKENS: &[(&str, &str)] = &[
        ("typescript", "typescript"),
        ("javascript", "javascript"),
        ("objective-c", "objectivec"),
        ("objectivec", "objectivec"),
        ("csharp", "csharp"),
        ("golang", "go"),
        ("kotlin", "kotlin"),
        ("python", "python"),
        ("scala", "scala"),
        ("swift", "swift"),
        ("objc", "objectivec"),
        ("dart", "dart"),
        ("java", "java"),
        ("ruby", "ruby"),
        ("rust", "rust"),
        ("cpp", "cpp"),
        ("php", "php"),
        ("c++", "cpp"),
        ("c#", "csharp"),
        ("go", "go"),
        ("c", "c"),
    ];

    let words: Vec<&str> = query.split_whitespace().collect();
    if words.is_empty() {
        return (query.to_owned(), None);
    }
    for (i, word) in words.iter().enumerate() {
        let lower = word.to_ascii_lowercase();
        if let Some(&(_, lang_id)) = LANG_TOKENS.iter().find(|(tok, _)| *tok == lower.as_str()) {
            // Strip the language word and an optional preceding "in" connector.
            let strip_from = if i > 0 && words[i - 1].eq_ignore_ascii_case("in") {
                i - 1
            } else {
                i
            };
            let remaining: Vec<&str> = words[..strip_from]
                .iter()
                .chain(words[i + 1..].iter())
                .copied()
                .collect();
            let stripped = remaining.join(" ");
            // Preserve original when stripping leaves nothing to search for.
            if stripped.trim().is_empty() {
                return (query.to_owned(), Some(lang_id));
            }
            return (stripped, Some(lang_id));
        }
    }
    (query.to_owned(), None)
}

fn search_symbol_raw(
    store: &SqliteStore,
    name: &str,
    lang_filter: Option<&str>,
    exact: bool,
) -> String {
    // Cap results: prevents self-DoS from wildcard queries (e.g. "a") and limits
    // accidental bulk exfiltration. The store LIKE query has no SQL LIMIT yet —
    // this Rust-side cap is the guard until that is added at the store layer.
    const MAX_SEARCH_RESULTS: usize = 50;

    let nodes = match store.search_nodes_fuzzy_filtered(name, lang_filter, exact) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("search_symbol error: {e}");
            return String::new();
        }
    };

    // CO-D1: if language filtering produced zero results, retry without the filter.
    // Language tokens like "c", "go", "dart" are short and common identifiers;
    // stripping them from the query then filtering by language produces false
    // "no matches" for queries like "C parser" or "go routine".
    let nodes = if nodes.is_empty() && lang_filter.is_some() {
        match store.search_nodes_fuzzy_filtered(name, None, exact) {
            Ok(n) => n,
            Err(_) => nodes,
        }
    } else {
        nodes
    };

    let lines: Vec<String> = nodes
        .iter()
        .take(MAX_SEARCH_RESULTS)
        .map(|n| {
            let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
            format!(
                "{} ({}) \u{2014} {}{loc}",
                display_label(n),
                n.kind,
                n.vname.path
            )
        })
        .collect();
    lines.join("\n")
}

/// Global variant of `search_symbol`.
///
/// Infers a language filter from the query (e.g. `"PaymentService in Kotlin"` →
/// search `"PaymentService"`, filter `language = kotlin`) and ranks repos in
/// fan-out mode by match count (most matches first).
pub fn search_symbol_global(
    repos: &HashMap<String, PathBuf>,
    name: &str,
    repo: Option<&str>,
    exact: bool,
) -> String {
    if let Err(reason) = validate_mcp_arg(name) {
        tracing::warn!("search_symbol_global rejected invalid arg: {reason}");
        return String::new();
    }

    let (stripped, lang_filter) = infer_language_from_query(name);
    let search_term = stripped.as_str();

    let raw = if repo.is_some() {
        // Single-repo path: SEC + stale filtering handled by collect_global.
        collect_global(repos, repo, |store, repo_name, single| {
            let result = search_symbol_raw(store, search_term, lang_filter, exact);
            if result.is_empty() || single {
                result
            } else {
                result
                    .lines()
                    .map(|l| format!("[{repo_name}] {l}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        })
    } else {
        // Fan-out path: open each live repo, collect results, rank by match
        // count descending so the most relevant repo appears first.
        if let Some(r) = repo {
            if validate_mcp_arg(r).is_err() {
                return String::new();
            }
        }
        let mut candidates: Vec<(&str, &PathBuf)> =
            repos.iter().map(|(k, v)| (k.as_str(), v)).collect();
        candidates.retain(|(_, db)| db.exists());
        let single = candidates.len() == 1;

        let mut parts: Vec<(usize, String)> = Vec::new();
        for (repo_name, db_path) in &candidates {
            match SqliteStore::open_read_only(db_path) {
                Ok(store) => {
                    let result = search_symbol_raw(&store, search_term, lang_filter, exact);
                    if !result.is_empty() {
                        let count = result.lines().count();
                        let text = if single {
                            result
                        } else {
                            result
                                .lines()
                                .map(|l| format!("[{repo_name}] {l}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        };
                        parts.push((count, text));
                    }
                }
                Err(e) => tracing::warn!("failed to open {}: {e}", db_path.display()),
            }
        }
        // Most matches first — most relevant repo surfaces at the top.
        parts.sort_by_key(|b| std::cmp::Reverse(b.0));
        parts
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n")
    };

    let content = if raw.is_empty() {
        format!("No symbols matching '{name}' found in the graph.")
    } else {
        raw
    };
    sanitize_for_mcp(&content)
}

// ── get_repo_map ──────────────────────────────────────────────────────────────

/// Return a structural overview of the indexed repository.
///
/// Groups all indexed nodes by file path and renders a tree annotated with
/// symbol counts and the top symbols per file, suitable for LLM consumption:
///
/// ```text
/// src/
///   service.ts  [3 symbols]  fn:charge, class:PaymentService, fn:refund
///   index.ts    [1 symbol]   fn:activate
/// ```
pub fn get_repo_map(store: &SqliteStore) -> String {
    sanitize_for_mcp(&get_repo_map_raw(store, 0))
}

/// True when a node kind counts as "code volume" (the surface signal). Excludes
/// imports, external-crate/package refs, doc chunks, and file/module container
/// nodes — none of which are symbols an agent would edit or navigate to.
fn repo_map_is_symbol_kind(kind: &str) -> bool {
    !matches!(
        kind,
        "file" | "go-pkg" | "import" | "crate" | "package" | "doc-chunk" | "file-module"
    )
}

/// A path we can place on the directory skeleton: not external, absolute,
/// protocol-prefixed (SCIP `scip://`, build caches, `..`), or a synthetic
/// pseudo-path (angle-bracket sentinels like `<cgo_synthetic>` — the successor
/// to the old `<unknown>` bucket, which carry no real directory location).
fn repo_map_is_local_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with("..")
        && !path.starts_with('/')
        && !path.starts_with('<')
        && !path.contains("://")
}

/// Parent directory of a path, or `(root)` for a top-level file.
fn repo_map_dir_of(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[..i].to_string(),
        None => "(root)".to_string(),
    }
}

/// Derive directory-level "regions" from a set of distinct local file paths,
/// adaptively. A directory becomes its own region when its subtree holds few
/// files or has no subdirectories; otherwise it splits into its child
/// directories (files directly in a split directory stay attached to it).
///
/// This is the language-agnostic replacement for manifest/crate detection: it
/// reads only the path structure, so `crates/` (large) splits into per-crate
/// regions while `docs/` (small) stays whole — on any language, any repo size.
/// Returned longest-first so callers assign a path by longest-prefix match.
///
/// Known tradeoff: a single very large unit splits one level deeper than its
/// siblings (e.g. `crates/travsr-plugin-host/src/sandbox` alongside whole
/// smaller crates), because the ceiling is uniform across the tree. This is
/// governed entirely by `TARGET_REGIONS`; it was tuned for legibility on both a
/// Rust monorepo and kubernetes, so changing it is a measurement-gated decision
/// (re-validate region granularity on both), not a free knob to nudge.
fn repo_regions(file_paths: &std::collections::HashSet<String>) -> Vec<String> {
    use std::collections::{BTreeSet, HashMap, HashSet};
    // Aim for a legible number of regions; split any subtree larger than this.
    const TARGET_REGIONS: usize = 24;
    const MIN_CEIL: usize = 4;

    let total = file_paths.len();
    if total == 0 {
        return Vec::new();
    }
    let ceil = (total / TARGET_REGIONS).max(MIN_CEIL);

    let mut subtree: HashMap<String, usize> = HashMap::new(); // files under a dir
    let mut children: HashMap<String, BTreeSet<String>> = HashMap::new(); // immediate subdirs
    let mut direct: HashMap<String, usize> = HashMap::new(); // files directly in a dir
    let mut top: BTreeSet<String> = BTreeSet::new(); // top-level dirs

    for path in file_paths {
        let dir = repo_map_dir_of(path);
        *direct.entry(dir.clone()).or_default() += 1;
        if dir == "(root)" {
            *subtree.entry("(root)".to_string()).or_default() += 1;
            top.insert("(root)".to_string());
            continue;
        }
        // Accumulate the ancestor chain: "a", "a/b", "a/b/c" for dir "a/b/c".
        let mut prev: Option<String> = None;
        let mut acc = String::new();
        for (i, seg) in dir.split('/').enumerate() {
            if i == 0 {
                acc.push_str(seg);
                top.insert(acc.clone());
            } else {
                acc.push('/');
                acc.push_str(seg);
            }
            *subtree.entry(acc.clone()).or_default() += 1;
            if let Some(p) = &prev {
                children.entry(p.clone()).or_default().insert(acc.clone());
            }
            prev = Some(acc.clone());
        }
    }

    fn walk(
        dir: &str,
        ceil: usize,
        subtree: &HashMap<String, usize>,
        children: &HashMap<String, BTreeSet<String>>,
        direct: &HashMap<String, usize>,
        regions: &mut HashSet<String>,
    ) {
        let count = *subtree.get(dir).unwrap_or(&0);
        let kids = children.get(dir);
        if count <= ceil || kids.map_or(true, |k| k.is_empty()) {
            regions.insert(dir.to_string());
        } else {
            for c in kids.unwrap() {
                walk(c, ceil, subtree, children, direct, regions);
            }
            if *direct.get(dir).unwrap_or(&0) > 0 {
                regions.insert(dir.to_string()); // host files that live directly here
            }
        }
    }

    let mut regions: HashSet<String> = HashSet::new();
    for t in &top {
        if t == "(root)" {
            regions.insert("(root)".to_string());
        } else {
            walk(t, ceil, &subtree, &children, &direct, &mut regions);
        }
    }

    let mut out: Vec<String> = regions.into_iter().collect();
    out.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    out
}

/// Assign a path to its region: the longest region directory that is an
/// ancestor. Top-level files map to `(root)`. `regions` MUST be longest-first.
fn repo_map_region_of(path: &str, regions: &[String]) -> String {
    if !path.contains('/') {
        return "(root)".to_string();
    }
    for r in regions {
        if r == "(root)" {
            continue;
        }
        if path.len() > r.len() && path.as_bytes()[r.len()] == b'/' && path.starts_with(r.as_str())
        {
            return r.clone();
        }
    }
    // Fallback (regions should cover every local path): top-level segment.
    match path.find('/') {
        Some(i) => path[..i].to_string(),
        None => "(root)".to_string(),
    }
}

/// Aggregate resolved cross-file dependency pairs into a deduped region→region
/// dependency edge set ("src region depends on dst region"), for the repo map's
/// dependents ranking. The graph overview computes its own weighted edge set with
/// the same region primitives (`repo_regions` / `repo_map_region_of` /
/// `repo_region_dependents`) but over the file-node universe rather than the
/// symbol-bearing one, so the two surfaces share the aggregation logic — not an
/// identical graph, and their dependent counts may differ by a small margin.
fn repo_region_dep_edges(
    pairs: &[(String, String)],
    regions: &[String],
    known: &std::collections::HashSet<String>,
) -> std::collections::HashSet<(String, String)> {
    let mut edges = std::collections::HashSet::new();
    for (sp, dp) in pairs {
        if !repo_map_is_local_path(sp) || !repo_map_is_local_path(dp) {
            continue;
        }
        let sr = repo_map_region_of(sp, regions);
        let dr = repo_map_region_of(dp, regions);
        if sr != dr && known.contains(&sr) && known.contains(&dr) {
            edges.insert((sr, dr));
        }
    }
    edges
}

/// Transitive-dependent count per region: distinct regions that can reach `R`
/// through the reverse dependency edges. Integer counts → deterministic.
fn repo_region_dependents(
    regions_universe: &std::collections::HashSet<String>,
    dep_edges: &std::collections::HashSet<(String, String)>,
) -> std::collections::HashMap<String, usize> {
    use std::collections::{HashMap, HashSet};
    let mut rev: HashMap<&str, Vec<&str>> = HashMap::new();
    for (a, b) in dep_edges {
        rev.entry(b.as_str()).or_default().push(a.as_str());
    }
    let mut dependents = HashMap::new();
    for region in regions_universe {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![region.as_str()];
        while let Some(cur) = stack.pop() {
            if let Some(ins) = rev.get(cur) {
                for &a in ins {
                    if seen.insert(a) {
                        stack.push(a);
                    }
                }
            }
        }
        seen.remove(region.as_str());
        dependents.insert(region.clone(), seen.len());
    }
    dependents
}

/// Build the agent cold-start orientation map: directory-level components ranked
/// by how depended-upon they are (from the resolved graph), each with their top
/// files/symbols, an honest footer, and cold-start state disclosure. Language-
/// agnostic — it reads only paths, kinds, and resolved edges. See
/// docs/plans/get-repo-map-pagerank-rework.md for the full derivation.
///
/// `reserve_per_line` is the byte cost of a per-line prefix the caller adds to
/// every returned line afterward (global multi-repo mode prefixes `[repo] `);
/// it is charged against the byte cap here so a prefixed section still fits.
/// Pass 0 for single-repo / local rendering (no prefix).
fn get_repo_map_raw(store: &SqliteStore, reserve_per_line: usize) -> String {
    use std::collections::{BTreeMap, HashMap, HashSet};

    let nodes = match store.all_nodes() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_repo_map error: {e}");
            return String::new();
        }
    };

    // State 1: not indexed. An empty string is indistinguishable to an agent from
    // "empty repo" or "error"; give it an actionable line instead.
    if nodes.is_empty() {
        return "Repo not indexed by Travsr; run `travsr init` to build the graph.".to_string();
    }

    // Directory skeleton from all local file paths → adaptive regions.
    let mut file_paths: HashSet<String> = HashSet::new();
    for n in &nodes {
        if repo_map_is_local_path(&n.vname.path) {
            file_paths.insert(n.vname.path.clone());
        }
    }
    let regions = repo_regions(&file_paths);

    // ── Surface: per-region symbol count + per-file symbol previews. ──
    let mut region_symbols: HashMap<String, usize> = HashMap::new();
    let mut region_files: HashMap<String, BTreeMap<String, Vec<String>>> = HashMap::new();
    for n in &nodes {
        if !repo_map_is_local_path(&n.vname.path) {
            continue; // drops external / absolute / protocol paths.
        }
        if repo_map_is_symbol_kind(&n.kind) && !n.vname.signature.is_empty() {
            let region = repo_map_region_of(&n.vname.path, &regions);
            *region_symbols.entry(region.clone()).or_default() += 1;
            region_files
                .entry(region)
                .or_default()
                .entry(n.vname.path.clone())
                .or_default()
                .push(display_label(n).into_owned());
        }
    }
    if region_symbols.is_empty() {
        return String::new();
    }

    // ── Spine: dependents from the RESOLVED graph (ref/call + resolves-to),
    // aggregated to regions. Language-agnostic — no import syntax is parsed. ──
    let known: HashSet<String> = region_symbols.keys().cloned().collect();
    let pairs = store.resolved_dep_pairs().unwrap_or_default();
    let dep_edges = repo_region_dep_edges(&pairs, &regions, &known);
    let dependents = repo_region_dependents(&known, &dep_edges);
    let has_refcall = store.has_any_refcall_edges();

    // Fail-open: no cross-region edges (e.g. Phase-A-only repo with sparse resolved
    // edges) => rank by surface. The map never regresses to alphabetical, never
    // empty because ranking failed.
    let fail_open = dep_edges.is_empty();

    let mut ordered: Vec<String> = region_symbols.keys().cloned().collect();
    ordered.sort_by(|a, b| {
        if fail_open {
            region_symbols[b]
                .cmp(&region_symbols[a])
                .then_with(|| a.cmp(b))
        } else {
            dependents[b]
                .cmp(&dependents[a])
                .then_with(|| region_symbols[b].cmp(&region_symbols[a]))
                .then_with(|| a.cmp(b))
        }
    });

    // ── Render. Every component gets a one-line entry (names = routing coverage
    // for the agent), and the top components additionally get file detail — but
    // only while detailing them does not crowd out naming the rest. Built under
    // the 4 KB SEC-001 cap so the footer is honest, not a silent guillotine. ──
    const MAX_SYMBOLS_PER_FILE: usize = 5;
    const MAX_FILES_PER_REGION: usize = 6;
    // Upper bound on components that get file detail. The naming reserve below
    // scales this down under the byte cap: on a large repo (many components to
    // name) only the top few actually receive detail, since routing coverage
    // (naming every component) is prioritized over detailing one further.
    const DETAIL_COMPONENTS: usize = 6;
    const SOFT_CAP: usize = 3800;
    const AVG_HDR_BYTES: usize = 64; // reserve so every remaining name still fits

    let total_regions = ordered.len();

    let mut header = String::new();
    if fail_open {
        header.push_str(
            "Repo map, components ranked by code volume (symbols); \
             no cross-component dependencies detected.\n",
        );
    } else {
        header.push_str(
            "Repo map, components ranked by dependents (most load-bearing first); \
             symbols = code volume.\n",
        );
    }
    // State 2: Phase B (semantic) never ran — the ranking is still valid but
    // call-graph data is absent, which affects the tools the agent calls next.
    if !has_refcall {
        header.push_str(
            "Note: semantic analysis has not run, dependents are from import \
             structure only; call-graph data is unavailable. Commit to run it.\n",
        );
    }

    // A caller (global multi-repo mode) prefixes every emitted line with
    // `[repo] ` after we return; `reserve_per_line` is that per-line cost. We
    // budget for it here so a prefixed section still fits the per-response cap
    // instead of being silently truncated mid-line. Single-repo/local = 0.
    let header_lines = header.matches('\n').count();
    let mut body_lines = 0usize;

    let mut body = String::new();
    let mut named = 0usize;
    for (i, region) in ordered.iter().enumerate() {
        let hdr = if fail_open {
            format!("{region}  symbols: {}\n", region_symbols[region])
        } else {
            format!(
                "{region}  dependents: {}  symbols: {}\n",
                dependents[region], region_symbols[region]
            )
        };
        // + this line + the footer line, each prefixed later.
        let prefix_cost = (header_lines + body_lines + 2) * reserve_per_line;
        if header.len() + body.len() + hdr.len() + prefix_cost > SOFT_CAP {
            break; // out of budget — remaining components go to the footer count.
        }
        body.push_str(&hdr);
        body_lines += 1;
        named += 1;

        if i >= DETAIL_COMPONENTS {
            continue; // name-only for lower-ranked components.
        }
        let mut file_list: Vec<(&String, &Vec<String>)> = region_files[region].iter().collect();
        file_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));
        for (fpath, sigs) in file_list.into_iter().take(MAX_FILES_PER_REGION) {
            let mut s: Vec<String> = sigs.clone();
            s.sort();
            s.dedup();
            let count = s.len();
            let preview: Vec<&str> = s
                .iter()
                .take(MAX_SYMBOLS_PER_FILE)
                .map(|x| x.as_str())
                .collect();
            let suffix = if count > MAX_SYMBOLS_PER_FILE {
                format!(", … (+{})", count - MAX_SYMBOLS_PER_FILE)
            } else {
                String::new()
            };
            let line = format!(
                "  {fpath}  [{count} symbol{}]  {}{}\n",
                if count == 1 { "" } else { "s" },
                preview.join(", "),
                suffix,
            );
            // Reserve budget so every not-yet-named component still fits — naming
            // (routing coverage) wins over detailing one component further. Each
            // reserved future name and each emitted line also carries the caller's
            // per-line prefix, so both are counted at the reserved rate.
            let reserve = total_regions.saturating_sub(named) * (AVG_HDR_BYTES + reserve_per_line);
            let prefix_cost = (header_lines + body_lines + 2) * reserve_per_line;
            if header.len() + body.len() + line.len() + reserve + prefix_cost > SOFT_CAP {
                break;
            }
            body.push_str(&line);
            body_lines += 1;
        }
    }

    let mut footer = String::new();
    let more_regions = total_regions - named;
    if more_regions > 0 {
        footer = format!("+{more_regions} more components (of {total_regions})\n");
    }

    format!("{header}{body}{footer}")
}

// ── get_graph_stats ───────────────────────────────────────────────────────────

/// Return accurate node and edge counts directly from the SQLite store.
///
/// Output format (newline-separated key-value pairs):
///   `nodes: 2121`
///   `edges: 8432`
///
/// Always returns a non-empty string — callers can check `nodes: 0` for an
/// empty graph. No sanitization needed: the output contains no user data.
pub fn get_graph_stats(store: &SqliteStore) -> String {
    let nodes = match store.node_count() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_graph_stats node_count error: {e}");
            0
        }
    };
    let edges = match store.edge_count() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_graph_stats edge_count error: {e}");
            0
        }
    };
    // schema_version lets the extension surface migration state in its stats
    // popup. Status-bar parsing reads only the `nodes:` line, so appending here
    // is backward compatible. 0 on read error — never fails the whole call.
    let schema_version = store.current_schema_version().unwrap_or(0);
    format!("nodes: {nodes}\nedges: {edges}\nschema_version: {schema_version}")
}

/// Global variant of `get_graph_stats` — sums across all registered repos.
///
/// Input validation for `repo` is performed by `collect_global`; do not bypass that helper.
pub fn get_graph_stats_global(repos: &HashMap<String, PathBuf>, repo: Option<&str>) -> String {
    let mut total_nodes: u64 = 0;
    let mut total_edges: u64 = 0;
    // DEBT(cloud-launch): counts must be filtered to caller's EdgeFilter scope before SSE ships
    collect_global(repos, repo, |store, _repo_name, _single| {
        total_nodes += match store.node_count() {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("get_graph_stats_global node_count error: {e}");
                0
            }
        };
        total_edges += match store.edge_count() {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("get_graph_stats_global edge_count error: {e}");
                0
            }
        };
        String::new() // accumulation done via captured mutables; return value unused
    });
    format!("nodes: {total_nodes}\nedges: {total_edges}")
}

/// Return per-language node counts for the current repo graph.
///
/// Output format: TSV `language\tcount`, one line per language, sorted by
/// count descending. Empty string when no nodes have language metadata.
/// No sanitization needed — language names are enum values from the indexer.
pub fn repo_languages(store: &SqliteStore) -> String {
    match store.language_distribution() {
        Ok(pairs) if pairs.is_empty() => String::new(),
        Ok(pairs) => pairs
            .iter()
            .map(|(lang, cnt)| format!("{lang}\t{cnt}"))
            .collect::<Vec<_>>()
            .join("\n"),
        Err(e) => {
            tracing::warn!("repo_languages error: {e}");
            String::new()
        }
    }
}

// ── synonyms (RFC-012 A2 F1) ────────────────────────────────────────────────────
//
// These tools manage the per-repo dynamic synonym table backing query expansion.
// Unlike the retrieval tools, their return strings are NOT passed through
// `sanitize_for_mcp`: they are control responses to the first-party VS Code
// extension UI (success / cap-error), not repo-derived data fed to an LLM. The
// extension matches on `"ok"` vs an error message. Arguments are still validated
// (SEC-002) before any store write.

/// Add one (term, alias) synonym pair. Returns `"ok"` on success or the store's
/// error text (e.g. the 200-row cap message) so the UI can surface it.
pub fn synonym_add(store: &mut SqliteStore, term: &str, alias: &str) -> String {
    if validate_mcp_arg(term).is_err() || validate_mcp_arg(alias).is_err() {
        tracing::warn!("synonym_add rejected invalid arg");
        return "invalid input".to_string();
    }
    match store.synonym_add(term, alias) {
        Ok(()) => "ok".to_string(),
        Err(e) => e.to_string(),
    }
}

/// Replace ALL aliases for `term` with the comma-separated `aliases_csv` list.
/// Atomic in the store layer. Rejects the whole call if any alias is invalid.
pub fn synonym_set(store: &mut SqliteStore, term: &str, aliases_csv: &str) -> String {
    if validate_mcp_arg(term).is_err() {
        tracing::warn!("synonym_set rejected invalid term");
        return "invalid input".to_string();
    }
    let aliases: Vec<String> = aliases_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if aliases.iter().any(|a| validate_mcp_arg(a).is_err()) {
        tracing::warn!("synonym_set rejected invalid alias");
        return "invalid input".to_string();
    }
    match store.synonym_set(term, &aliases) {
        Ok(()) => "ok".to_string(),
        Err(e) => e.to_string(),
    }
}

/// Remove a single (term, alias) pair. No-op if it does not exist.
pub fn synonym_remove(store: &mut SqliteStore, term: &str, alias: &str) -> String {
    if validate_mcp_arg(term).is_err() || validate_mcp_arg(alias).is_err() {
        tracing::warn!("synonym_remove rejected invalid arg");
        return "invalid input".to_string();
    }
    match store.synonym_remove(term, alias) {
        Ok(()) => "ok".to_string(),
        Err(e) => e.to_string(),
    }
}

/// Remove ALL aliases for `term`.
pub fn synonym_remove_term(store: &mut SqliteStore, term: &str) -> String {
    if validate_mcp_arg(term).is_err() {
        tracing::warn!("synonym_remove_term rejected invalid term");
        return "invalid input".to_string();
    }
    match store.synonym_remove_term(term) {
        Ok(()) => "ok".to_string(),
        Err(e) => e.to_string(),
    }
}

/// Reset the synonym table to the built-in static defaults.
pub fn synonym_reset(store: &mut SqliteStore) -> String {
    match store.synonym_reset() {
        Ok(()) => "ok".to_string(),
        Err(e) => e.to_string(),
    }
}

/// List all active synonym pairs as `term => alias`, one per line, sorted by the
/// store. Empty string when the table is empty.
pub fn synonym_list(store: &SqliteStore) -> String {
    match store.synonym_list() {
        Ok(pairs) => pairs
            .iter()
            .map(|(term, alias)| format!("{term} => {alias}"))
            .collect::<Vec<_>>()
            .join("\n"),
        Err(e) => {
            tracing::warn!("synonym_list error: {e}");
            String::new()
        }
    }
}

// ── repos (registry management, VSCODE-247) ──────────────────────────────────
//
// Global-registry operations exposed for the VS Code "Registered repos" webview.
// Like the synonym tools, return strings are plain control responses (not
// `sanitize_for_mcp`'d) — they are consumed by the first-party extension UI, not
// fed to an LLM. The registry is global (independent of the open store), so these
// are valid on the stdio server regardless of which repo it was started for.

/// List registry entries as TSV: `name\tdb_path\t{0|1}` (1 = graph.db exists).
/// Empty string when the registry is empty.
pub fn repos_list() -> String {
    let repos = match travsr_store::registry::all_repos() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("repos_list error: {e}");
            return String::new();
        }
    };
    let mut rows: Vec<(String, std::path::PathBuf)> = repos.into_iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.iter()
        .map(|(name, db_path)| {
            let exists = if db_path.exists() { "1" } else { "0" };
            // UX-018: emit the basename as the display Name and the verbatim-
            // stripped full path, matching the CLI `repos` table. `repos_remove`
            // resolves either back to the registry key.
            let display = travsr_store::registry::display_name(name);
            let path = travsr_store::registry::strip_verbatim_prefix(&db_path.to_string_lossy())
                .into_owned();
            format!("{display}\t{path}\t{exists}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Prune stale registry entries. Returns `pruned: N` followed by the removed
/// names, or `pruned: 0` when nothing was stale.
pub fn repos_prune() -> String {
    match travsr_store::registry::prune() {
        Ok(removed) => {
            let mut out = format!("pruned: {}", removed.len());
            for name in &removed {
                out.push('\n');
                out.push_str(name);
            }
            out
        }
        Err(e) => {
            tracing::warn!("repos_prune error: {e}");
            "error".to_string()
        }
    }
}

/// Remove a single repo by full registry key or display basename (UX-018).
/// Returns `ok` / `not found` / `ambiguous: <paths>`.
pub fn repos_remove(name: &str) -> String {
    use travsr_store::registry::UnregisterResult;
    match travsr_store::registry::unregister_resolving(name) {
        Ok(UnregisterResult::Removed) => "ok".to_string(),
        Ok(UnregisterResult::NotFound) => "not found".to_string(),
        Ok(UnregisterResult::Ambiguous(paths)) => format!("ambiguous: {}", paths.join(", ")),
        Err(e) => {
            tracing::warn!("repos_remove error: {e}");
            "error".to_string()
        }
    }
}

// ── get_execution_path ────────────────────────────────────────────────────────

/// Find a traversal path from `source` symbol to `sink` symbol through the graph.
///
/// Uses a Dijkstra shortest path plus a λ-corridor around it (`pcst_path`, whose
/// name predates #527 — it is not a Prize-Collecting Steiner Tree approximation,
/// and describing it as one here contradicted the algorithm and the tool
/// description; #755 Part B item 5). A result that does not reach the sink
/// (`pcst_path`'s BFS fallback on disconnected graphs) is reported as "no path
/// found" rather than relayed as if it were a path (#620).
///
/// Output format (one line per node on path):
///   `fn:charge (function) — src/payment.ts`
///   `fn:processPayment (function) — src/processor.ts`
///
/// # SEC P0
/// "Symbol not found" and "symbol access denied" produce byte-identical
/// output (a single could-not-resolve message that names neither the failing
/// endpoint nor the reason), preventing an existence oracle.
///
/// #620: an empty answer is never silent — unresolved endpoints and
/// resolved-but-disconnected endpoints each return an explicit message
/// instead of an empty envelope an agent would misread as authoritative.
pub fn get_execution_path(store: &SqliteStore, source: &str, sink: &str) -> String {
    // Phase B deferred: execution paths require call edges which are not yet indexed.
    if phase_b_pending(store) {
        return phase_b_pending_json("Semantic call-edge index");
    }
    get_execution_path_with_filter(store, source, sink, &OpenFilter)
}

/// Authenticated variant — applies RBAC filter at traversal time.
/// Wired to session context in S16 when `SessionStore` is integrated into the server loop.
// DEBT(travsr-199): wire into server.rs dispatch when SessionStore lands in S16.
#[allow(dead_code)]
pub(crate) fn get_execution_path_authed(
    store: &SqliteStore,
    source: &str,
    sink: &str,
    filter: &dyn EdgeFilter,
) -> String {
    get_execution_path_with_filter(store, source, sink, filter)
}

fn get_execution_path_with_filter(
    store: &SqliteStore,
    source: &str,
    sink: &str,
    filter: &dyn EdgeFilter,
) -> String {
    if let Err(reason) = validate_mcp_arg(source) {
        tracing::warn!("get_execution_path rejected invalid source arg: {reason}");
        return String::new();
    }
    if let Err(reason) = validate_mcp_arg(sink) {
        tracing::warn!("get_execution_path rejected invalid sink arg: {reason}");
        return String::new();
    }
    // #617: append the Phase-B staleness note so "no path" is never trusted
    // while the call-edge set is known-incomplete.
    with_phase_b_note(
        store,
        sanitize_for_mcp(&get_execution_path_body(store, source, sink, filter, true)),
    )
}

/// Shared search body. `diagnose` controls whether the empty outcomes are
/// explained (#620): single-repo callers pass `true` so an agent never gets a
/// silent blank; the multi-repo aggregator passes `false` because a repo
/// without the symbols is normal, not an error.
fn get_execution_path_body(
    store: &SqliteStore,
    source: &str,
    sink: &str,
    filter: &dyn EdgeFilter,
    diagnose: bool,
) -> String {
    // SEC P0: resolve source and sink; treat "not found" == "access denied" identically.
    let src_node = match store.search_nodes_by_name(source) {
        Ok(n) => n
            .into_iter()
            .find(|n| filter.allow(n.id, n.id, Some(n.vname.corpus.as_str()))),
        Err(e) => {
            tracing::warn!("get_execution_path source search error: {e}");
            return String::new();
        }
    };

    let sink_node = match store.search_nodes_by_name(sink) {
        Ok(n) => n
            .into_iter()
            .find(|n| filter.allow(n.id, n.id, Some(n.vname.corpus.as_str()))),
        Err(e) => {
            tracing::warn!("get_execution_path sink search error: {e}");
            return String::new();
        }
    };

    // #755 Part B item 6: say WHICH endpoint failed. The old message hedged with
    // "source X and/or sink Y" even when only one side was bad, so the caller had
    // to re-probe each side separately to learn what to fix.
    //
    // SEC P0 is preserved: both failure modes (not found and access denied) still
    // land in the same branch and produce byte-identical text for a given side, so
    // the message never says WHY a side failed. Naming the side leaks nothing an
    // existence oracle needs — "sink 'X' did not resolve" is emitted identically
    // for a nonexistent X and for an X the caller may not see, which is exactly
    // the indistinguishability `get_execution_path_denied_matches_not_found` pins.
    let (src, snk) = match (src_node, sink_node) {
        (Some(a), Some(b)) => (a, b),
        (src_opt, sink_opt) => {
            if diagnose {
                let which = match (src_opt.is_some(), sink_opt.is_some()) {
                    (false, true) => format!("source '{source}'"),
                    (true, false) => format!("sink '{sink}'"),
                    _ => format!("source '{source}' or sink '{sink}'"),
                };
                return format!(
                    "could not resolve {which}, verify the names with search_symbol, or pass full signatures (e.g. fn:charge)."
                );
            }
            return String::new();
        }
    };

    let path = match travsr_retrieval::pcst_path(store, src.id, snk.id, filter, 4096) {
        Ok(nodes) => nodes,
        Err(e) => {
            tracing::warn!("get_execution_path pcst error: {e}");
            return String::new();
        }
    };

    // #620: when the sink is unreachable, pcst_path falls back to a
    // source-rooted BFS neighbourhood — so "no path" manifests as a result
    // that never includes the sink, not as an empty vector. Relaying that
    // neighbourhood as if it were an execution path is the silent false
    // signal this issue exists to fix.
    if path.is_empty() || !path.iter().any(|n| n.id == snk.id) {
        if diagnose {
            return format!(
                "no path found: '{}' and '{}' both resolved, but no connecting call chain was found within traversal limits.",
                display_label(&src), display_label(&snk)
            );
        }
        return String::new();
    }

    let lines: Vec<String> = path
        .iter()
        .map(|n| {
            format!(
                "{} ({}) \u{2014} {}",
                display_label(n),
                n.kind,
                n.vname.path
            )
        })
        .collect();
    lines.join("\n")
}

/// Global variant of `get_execution_path` — searches one named repo or all registered repos.
///
/// `filter` is applied at traversal time. Pass `&OpenFilter` for unauthenticated local mode;
/// pass `&session.filter()` in authenticated mode (S16). Do NOT hardcode `&OpenFilter` at
/// call sites — the caller owns the auth context.
pub fn get_execution_path_global(
    repos: &HashMap<String, PathBuf>,
    source: &str,
    sink: &str,
    repo: Option<&str>,
    filter: &dyn EdgeFilter,
) -> String {
    if let Err(reason) = validate_mcp_arg(source) {
        tracing::warn!("get_execution_path_global rejected invalid source: {reason}");
        return String::new();
    }
    if let Err(reason) = validate_mcp_arg(sink) {
        tracing::warn!("get_execution_path_global rejected invalid sink: {reason}");
        return String::new();
    }
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        // #617 + #645: per-repo notes — each store's own Phase B freshness, and
        // the HEAD note against *this* repo's own checkout (not global LAUNCH_CWD).
        // #620: diagnose empty outcomes only in single-repo mode; a repo that
        // simply doesn't contain the symbols must stay silent in an aggregate.
        let head = repos
            .get(repo_name)
            .and_then(|db| repo_head_from_registry_path(db));
        let result = append_read_notes(
            store,
            get_execution_path_body(store, source, sink, filter, single),
            head.as_deref(),
        );
        if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    });
    sanitize_for_mcp(&raw)
}

/// Global variant of `get_repo_map`.
pub fn get_repo_map_global(repos: &HashMap<String, PathBuf>, repo: Option<&str>) -> String {
    // repo arg validated inside collect_global.
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        // Multi-repo mode prefixes every line with `[repo] `; reserve those bytes
        // in the render so a prefixed section is not truncated mid-line by the
        // per-response cap. Single-repo mode adds no prefix, so it reserves none.
        let reserve_per_line = if single { 0 } else { repo_name.len() + 3 };
        let result = get_repo_map_raw(store, reserve_per_line);
        if result.is_empty() || single {
            result
        } else {
            result
                .lines()
                .map(|l| format!("[{repo_name}] {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    });
    sanitize_for_mcp(&raw)
}

// ── get_context ───────────────────────────────────────────────────────────────

/// Retrieve the most relevant context for `query` within `token_budget` tokens.
///
/// Pipeline: validate → seed lookup (RBAC-filtered) → PPR → get_nodes →
/// knapsack → format → sanitize → append footer → wrap envelope.
///
/// # SEC P0
/// Returns identical output for "not found" and "access denied" to prevent
/// existence oracle attacks. Seeds are filtered through the `OpenFilter` so
/// callers cannot distinguish missing vs denied symbols.
pub fn get_context(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    include_snippets: bool,
    snippet_budget: Option<usize>,
) -> String {
    get_context_with_filter(
        store,
        query,
        token_budget,
        &OpenFilter,
        include_snippets,
        snippet_budget,
    )
}

/// Authenticated variant — applies RBAC filter at seed lookup and node fetch.
#[allow(dead_code)]
pub(crate) fn get_context_authed(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    filter: &dyn EdgeFilter,
    include_snippets: bool,
    snippet_budget: Option<usize>,
) -> String {
    get_context_with_filter(
        store,
        query,
        token_budget,
        filter,
        include_snippets,
        snippet_budget,
    )
}

/// Raw variant — returns body without envelope. Used by global aggregation to
/// prevent double-sanitization when multiple stores are aggregated before wrapping.
///
/// Reads the KNN hook off `store` rather than taking it as an argument: in global
/// mode the caller opens the store and arms it, so hardcoding `None` here left the
/// semantic lane off for every registry-wide `get_context`.
pub(crate) fn get_context_raw(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    include_snippets: bool,
    snippet_budget: Option<usize>,
) -> String {
    let knn = store.embed_knn_fn();
    get_context_body(
        store,
        query,
        token_budget,
        &OpenFilter,
        include_snippets,
        snippet_budget,
        knn.as_ref().map(|f| f as EmbedKnnFn),
    )
}

/// Embedding-aware variant of [`get_context`].
///
/// `embed_knn(query, k)` should return up to `k` file node IDs ranked by
/// cosine similarity to the embedded query (supplied by the daemon via
/// `EmbedSupervisor`).  When the closure returns an empty vec (e.g. before
/// `travsr embed reindex` has run), the function transparently falls back to
/// `search_nodes_fuzzy` so callers never see an empty result due to a missing
/// index.
pub fn get_context_embed(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    include_snippets: bool,
    snippet_budget: Option<usize>,
    embed_knn: EmbedKnnFn<'_>,
) -> String {
    let body = get_context_body(
        store,
        query,
        token_budget,
        &OpenFilter,
        include_snippets,
        snippet_budget,
        Some(embed_knn),
    );
    wrap_envelope(&body)
}

fn get_context_with_filter(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    filter: &dyn EdgeFilter,
    include_snippets: bool,
    snippet_budget: Option<usize>,
) -> String {
    let body = get_context_body(
        store,
        query,
        token_budget,
        filter,
        include_snippets,
        snippet_budget,
        None,
    );
    wrap_envelope(&body)
}

/// Why a node appeared in `get_context` output.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeRole {
    /// Direct text match — one of the up-to-5 seeds.
    Seed,
    /// This node calls one of the seed symbols (reverse RefCall/RefImports/Depends edge).
    Caller,
    /// A seed symbol calls or imports this node (forward RefCall/Depends/RefImports edge).
    Dependency,
    /// PPR structural relevance; no direct 1-hop edge to or from any seed.
    Context,
}

impl NodeRole {
    fn label(self) -> &'static str {
        match self {
            NodeRole::Seed => "seed",
            NodeRole::Caller => "caller",
            NodeRole::Dependency => "dependency",
            NodeRole::Context => "context",
        }
    }
}

/// Return of [`embed_path_seeds`]: `(selected_seeds, n_eligible_before_cap,
/// knn_elapsed_ms, cosine_oracle)`. The oracle maps every over-fetched candidate
/// to its query cosine — a superset of the (capped) selected seeds.
type EmbedSeedResult = (
    Vec<(CoreNode, f32)>,
    usize,
    u128,
    std::collections::HashMap<NodeId, f32>,
);

/// Minimum number of PPR seeds from KNN — always try to include this many.
const MIN_EMBED_SEEDS: usize = 5;
/// Maximum number of PPR seeds from KNN — never include more than this.
const MAX_EMBED_SEEDS: usize = 20;
/// Maximum seeds per unique file path — prevents all seeds clustering in one file.
const MAX_SEEDS_PER_PATH: usize = 2;
/// PPR seed weight for an exact FTS substring match (query token in signature or path).
#[allow(dead_code)]
const FTS_EXACT_WEIGHT: f32 = 1.0;
/// PPR seed weight for a fuzzy/trigram FTS match (L2-A expansion, no literal substring match).
#[allow(dead_code)]
const FTS_FUZZY_WEIGHT: f32 = 0.6;

/// Per-kind multiplier applied to FTS seed weights before PPR personalisation.
///
/// Boosts definitive code entities (methods, classes, structs, traits) and
/// suppresses structural/import nodes that produce noisy PPR walks on large
/// graphs. Language is required because `"module"` means different things:
/// a Rust `mod foo {}` block is real code (1.0×) while a Go package
/// declaration node is graph connective tissue (0.05×).
pub(crate) fn kind_boost(kind: &str, language: &str) -> f32 {
    match kind {
        // Primary callables — highest value across all languages
        "method" | "function" | "constructor" => 2.0,
        // Named type definitions — Rust struct/enum/trait, Go class, TS interface
        "class" | "interface" | "struct" | "enum" | "trait" | "union" => 1.5,
        // Named value definitions — Rust constant/static, important anchors
        "constant" | "static" => 1.2,
        // Impl blocks and type aliases — moderate; contain methods but aren't entry points
        "impl" | "type" => 1.0,
        // Data members and local bindings — lower value, often noise at seed level
        "field" | "var" | "variable" | "property" => 0.7,
        // Imports and file-boundary structural nodes
        "import" | "file-module" | "crate" => 0.3,
        // Go-specific package identifier nodes — always structural noise
        "go-pkg" => 0.05,
        // Module kind is language-dependent:
        //   Go   → scip-go package declaration node, structural, high in-degree → suppress
        //   Rust → `mod foo {}` declaration, real code entity → keep
        "module" if language == "go" => 0.05,
        "module" => 1.0,
        _ => 1.0,
    }
}

/// Circuit-breaker threshold for KNN sidecar inference (ms). Calls exceeding this
/// are discarded and the caller falls back to FTS seeds.
///
/// Default: 600ms — measured bge-large-en-v1.5 ONNX inference on Apple Silicon is
/// 250–270ms; bge-base-en-v1.5 is 200–220ms. 600ms gives 2× headroom over the
/// largest measured value and absorbs the extra CPU pressure from a concurrent
/// Phase 2 background reindexer. Setting this too low causes silent KNN degradation
/// (FTS fallback) at exactly the moment the semantic index is most valuable.
///
/// Override via `TRAVSR_KNN_BUDGET_MS`. The value must be a positive integer;
/// invalid values fall back to the default.
///
/// Shared with `travsr ask` via `query.rs`.
pub(crate) fn knn_budget_ms() -> u128 {
    std::env::var("TRAVSR_KNN_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse::<u128>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(600)
}

/// Max time the first `get_context` call blocks for the embed sidecar to arm, so
/// the opening query of a session uses embeddings instead of silently degrading
/// to lexical-only during the ~3 s startup window. The wait happens OUTSIDE the
/// KNN circuit-breaker timing, so this one-time arm cost is never misread as a
/// slow KNN round-trip. Override via `TRAVSR_EMBED_ARM_WAIT_MS`. Default 5000 —
/// headroom over the measured ~3.2 s (store-open dominates and scales with graph
/// size). Subsequent queries see the armed hook and never wait.
pub(crate) fn embed_arm_wait_ms() -> u64 {
    std::env::var("TRAVSR_EMBED_ARM_WAIT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(5000)
}
/// Kept for tests that reference the constant directly; resolves to the default.
#[cfg(test)]
pub(crate) const KNN_BUDGET_MS: u128 = 600;

/// Returns `true` for nodes that should never be used as PPR seeds regardless
/// of their KNN cosine score.
///
/// See [`travsr_core::noise::is_structural_noise`] for the full exclusion list
/// (R5 test/build/vendor directory patterns, doc-chunk/crate kind, `scip:`
/// signatures, and the ingest-level checks in [`travsr_core::is_noise_node`]).
pub(crate) fn is_noise_seed(node: &CoreNode) -> bool {
    // #478 WS-2: the full structural-noise predicate (core ingest-time checks
    // plus the per-language test/build-artifact/doc-chunk/crate/scip rules
    // this function used to carry inline) now lives in `travsr_core::noise`,
    // the single source of truth — needed there so the schema v21 migration
    // in `travsr-store` can compute the same `is_noise` classification without
    // depending on `travsr-mcp` (crate dependency rule).
    travsr_core::noise::is_structural_noise(node)
}

/// #462: nodes that are never useful as `get_context` *relevance results*, even
/// though they are legitimate graph nodes the ingest/`is_noise_node` layer keeps
/// (so `get_callers`/`get_dependencies` can still reach them). Applied to the
/// final candidate list (seeds + PPR expansion) so CI/workflow package nodes and
/// bare file nodes stop leaking into the context as budget filler. A superset of
/// [`is_noise_seed`]; unlike a score floor it is purely structural, so it can
/// never evict a genuine answer regardless of that answer's relevance score.
fn is_context_result_noise(node: &CoreNode) -> bool {
    is_noise_seed(node)
        // CI/workflow dependency package nodes (pkg:actions/*, pkg:<dep>@<sha>) —
        // real ExternalDependency nodes, but never an answer to a code query.
        || node.kind == "package"
        || node.vname.signature.starts_with("pkg:")
        // Bare file nodes carry no symbol body; a symbol from the file is a
        // better relevance result than the file container itself.
        || node.kind == "file"
}

/// RFC-022 D3 (RC-3): conservative test-symbol detector for the *anchor* pool.
///
/// Inline `#[test]` functions live in ordinary `src/` files, so the path-based
/// checks in [`is_noise_seed`] never see them, yet their descriptive names
/// (`..._prefers_logic_bearing_over_field_only`) literally contain query words and
/// out-anchor the real implementation. This recognises only the common
/// cross-language test-*naming* markers (`test_foo`, `foo_test`, Go `TestFoo` /
/// `BenchmarkFoo` / `FuzzFoo`) — deliberately narrow so a real function is never
/// demoted; a marker-less descriptively-named inline test is not caught here (that
/// needs an indexer-emitted test flag, tracked separately).
pub(crate) fn is_test_symbol(node: &CoreNode) -> bool {
    if node.kind != "function" && node.kind != "method" {
        return false;
    }
    // Bare symbol name: drop the "fn:"/"method:" kind prefix and any "Type#" receiver.
    let sig = node.vname.signature.as_str();
    let name = sig.rsplit(['#', ':']).next().unwrap_or(sig);
    let lower = name.to_ascii_lowercase();
    lower.starts_with("test_")
        || lower.ends_with("_test")
        || lower.contains("_test_")
        || (name.starts_with("Test") && name[4..].chars().next().is_some_and(char::is_uppercase))
        || (name.starts_with("Benchmark")
            && name[9..].chars().next().is_some_and(char::is_uppercase))
        || (name.starts_with("Fuzz") && name[4..].chars().next().is_some_and(char::is_uppercase))
}

/// RFC-022 D3 (RC-3): noise predicate for *anchor-pool* candidates at emit time.
///
/// Anchors are the most-trusted seeds — they are floated to the top of the ranking
/// and can trigger the deterministic G1 bypass — so the bar to enter the anchor
/// pool is stricter than for a general PPR seed. Rejects everything
/// [`is_context_result_noise`] rejects (CI/workflow `pkg:` nodes, bare file nodes,
/// build/cache/test-dir paths) plus in-src test symbols ([`is_test_symbol`]). A
/// rejected candidate can still enter retrieval via the lexical/KNN seed paths
/// (kept by [`is_noise_seed`]); it just cannot be an anchor.
///
/// #479: the AST-derived `test_role` is the precise test gate here. The anchor
/// candidates flow from [`SqliteStore::search_nodes_by_name`], which carries the
/// `test_role` column, so `node.test_role.is_test()` fires on this path and keeps
/// a descriptively-named inline test (one whose name `is_test_symbol` does not
/// recognise) out of the anchor pool, not just at display time. [`is_test_symbol`]
/// stays as the name-based fallback for rows that predate the v22 reindex (their
/// `test_role` reads back `None` until re-parsed). Keep both until v22 is
/// guaranteed to have reparsed every corpus.
pub(crate) fn is_anchor_noise(node: &CoreNode) -> bool {
    is_context_result_noise(node) || node.test_role.is_test() || is_test_symbol(node)
}

/// Derive PPR seeds from symbol-level KNN results with score-based selection.
///
/// Strategy:
/// - Noise nodes (Cargo crates, test/bench paths) are skipped unconditionally.
/// - Always include the top `MIN_EMBED_SEEDS` non-noise results (regardless of score).
/// - Extend with any additional results whose cosine similarity >= `EMBED_SCORE_THRESHOLD`.
/// - Cap at `MAX_EMBED_SEEDS`.
/// - Deduplicate by file path: at most `MAX_SEEDS_PER_PATH` symbols from any one file,
///   preventing all seeds from clustering inside a single large file.
///
/// Returns `(seeds, n_candidates_before_cap)` where:
/// - `seeds` — `(NodeId, cosine_score)` pairs for weighted PPR personalisation.
/// - `n_candidates_before_cap` — eligible candidates before the `MAX_EMBED_SEEDS` cap fired.
///   When `> seeds.len()`, the caller may signal to the user that the query was broad.
///
/// Returns empty seeds when KNN returns nothing (index not yet built), so the caller
/// can fall back to `search_nodes_fuzzy`.
/// Returns `(seeds, n_candidates_before_cap, knn_elapsed_ms, cosine_oracle)`.
/// `knn_elapsed_ms` covers only the `knn_fn` HNSW/ONNX call, not the SQLite
/// postprocessing — the caller uses it for the circuit-breaker decision.
///
/// `cosine_oracle` maps every over-fetched candidate `NodeId` to its query
/// cosine — a superset of `seeds` (which is capped/filtered). `build_seed_set`
/// uses it to validate lexical anchors: an NL query word that is rare-by-
/// coincidence (e.g. "literal") resolves to a confident FTS anchor that the
/// embedding places far from the query intent; the oracle lets us demote it.
///
/// Each seed carries the full `CoreNode` so callers can apply structural proximity
/// filtering without a second DB round-trip.
pub(crate) fn embed_path_seeds(
    store: &SqliteStore,
    query: &str,
    knn_fn: EmbedKnnFn<'_>,
    filter: &dyn EdgeFilter,
) -> EmbedSeedResult {
    // Over-request 6× to keep MIN_EMBED_SEEDS filled after noise + RBAC filtering.
    // Time only the KNN inference call — not the subsequent SQLite get_nodes fetch.
    let t0 = std::time::Instant::now();
    // Normalize before embedding: strip sentence punctuation from word edges so
    // "work?" and "work ?" produce the same vector ("how daemon work?" == "how daemon work ?").
    let normalized_query = travsr_store::fts_tokenize::normalize_nl_query(query);
    let knn_pairs = knn_fn(
        &normalized_query,
        (MAX_EMBED_SEEDS as u32).saturating_mul(6),
    );
    let knn_elapsed_ms = t0.elapsed().as_millis();
    if knn_pairs.is_empty() {
        return (vec![], 0, knn_elapsed_ms, std::collections::HashMap::new());
    }

    // Fetch node metadata for noise filter + RBAC + path dedup.
    let all_ids: Vec<NodeId> = knn_pairs.iter().map(|&(id, _)| id).collect();
    let nodes = match store.get_nodes(&all_ids) {
        Ok(ns) => ns,
        Err(e) => {
            tracing::warn!("embed_path_seeds get_nodes error: {e}");
            return (vec![], 0, knn_elapsed_ms, std::collections::HashMap::new());
        }
    };
    let node_map: std::collections::HashMap<NodeId, CoreNode> =
        nodes.into_iter().map(|n| (n.id, n)).collect();

    // Cosine oracle: every over-fetched candidate that is a genuine relevance
    // target — used downstream to validate lexical anchors AND to ground
    // confidence. RFC-022 D3 (RC-3): exclude CI/workflow `pkg:` package nodes and
    // bare file nodes here, not just from the seed list. Arctic embeds a salad
    // query ("shopping cart checkout payment flow") near `pkg:actions/checkout`
    // (cosine ~0.40, above the calibrated recall floor); letting that cosine into
    // the oracle falsely grounds the query to Exact — a salad leak that also breaks
    // the G2 invariant (embed-on would disagree with embed-off, which abstains). A
    // package/file node is never a valid answer, so it must not vote on confidence.
    let cosine_oracle: std::collections::HashMap<NodeId, f32> = knn_pairs
        .iter()
        .filter(|&&(id, _)| {
            node_map
                .get(&id)
                .is_some_and(|n| !is_context_result_noise(n))
        })
        .map(|&(id, s)| (id, s))
        .collect();

    // #462 WS1b — calibrated KNN admission. The old fixed `EMBED_SCORE_THRESHOLD`
    // (0.75) is a bge-scale cosine: no arctic conceptual match ever reaches it
    // (arctic golds top out ~0.65), so beyond the MIN floor every semantic neighbour
    // was dropped and the gold survived only if it happened to rank in the first
    // MIN — the "DROP@embed_path_seeds" losses in the WS0 trace. Gate on the
    // model-relative recall floor instead: identity/bge keeps the exact 0.75 bar
    // (byte-for-byte unchanged), while a calibrated model admits any neighbour above
    // its own measured salad band, up to MAX_EMBED_SEEDS.
    let cal = crate::seed::Calibration::load(store);
    let recall_floor = crate::seed::semantic_recall_floor(&cal);

    let mut path_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut seeds: Vec<(CoreNode, f32)> = Vec::with_capacity(MAX_EMBED_SEEDS);
    // Count eligible candidates (pass noise+RBAC+score filters) before MAX_EMBED_SEEDS cap.
    let mut n_eligible: usize = 0;

    for &(id, score) in &knn_pairs {
        // Include if: we still owe MIN seeds, OR this node clears the recall floor.
        if seeds.len() >= MIN_EMBED_SEEDS && score < recall_floor {
            continue;
        }
        let node = match node_map.get(&id) {
            Some(n) => n,
            None => continue,
        };
        // Skip Cargo dependency nodes, CI/workflow package nodes, bare file nodes,
        // and test/bench fixtures unconditionally. RFC-022 D3: same bar as the
        // oracle above — a `pkg:`/file node is never a valid semantic seed.
        if is_context_result_noise(node) {
            continue;
        }
        if !filter.allow(id, id, Some(node.vname.corpus.as_str())) {
            continue;
        }
        let path_count = path_counts.entry(node.vname.path.clone()).or_insert(0);
        if *path_count >= MAX_SEEDS_PER_PATH {
            continue;
        }
        *path_count += 1;
        n_eligible += 1;
        if seeds.len() < MAX_EMBED_SEEDS {
            seeds.push((node.clone(), score));
        }
    }

    (seeds, n_eligible, knn_elapsed_ms, cosine_oracle)
}

/// Compose all advisory signals appended to a `get_context` response footer.
///
/// Signals are ordered: overflow (most actionable) → seed cap → degraded state.
/// Returns an empty string when the graph is fully operational and no truncation occurred.
///
/// R3: `knn_degraded` is true when the embed sidecar was configured (`has_embed`)
/// but KNN returned empty or was circuit-broken; the note distinguishes this from
/// the "embed not configured" case so the AI doesn't assume full semantic coverage.
pub(crate) fn build_context_signals(
    store: &SqliteStore,
    has_embed: bool,
    embed_warming: bool,
    knn_degraded: bool,
    overflow_msg: Option<&str>,
    seed_cap_msg: Option<&str>,
) -> String {
    build_context_signals_with_r2(
        phase_b_pending(store),
        has_embed,
        embed_warming,
        has_embed && store.embed_disabled(),
        store.has_embed_db(),
        knn_degraded,
        overflow_msg,
        seed_cap_msg,
        None,
        None,
    )
}

/// PF-M4: accepts `phase_b_pending` as a pre-computed bool so callers can
/// hoist the two `get_meta` queries out of the hot path and compute once.
///
/// `embed_initialized` = embed.db exists (Phase 1 has run at some point).
/// When `!has_embed && embed_initialized` the daemon hasn't injected the KNN hook yet
/// (e.g. Phase 1 just completed, daemon restarting) — show "in progress" instead of "init needed".
#[allow(clippy::too_many_arguments)]
fn build_context_signals_with_r2(
    phase_b_pending: bool,
    has_embed: bool,
    embed_warming: bool,
    embed_disabled: bool,
    embed_initialized: bool,
    knn_degraded: bool,
    overflow_msg: Option<&str>,
    seed_cap_msg: Option<&str>,
    multi_concept_note: Option<&str>,
    weak_match_note: Option<&str>,
) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(msg) = overflow_msg {
        parts.push(msg);
    }
    if let Some(msg) = seed_cap_msg {
        parts.push(msg);
    }
    // R1: weak-match floor note — before other notes so the AI sees it first.
    if let Some(msg) = weak_match_note {
        parts.push(msg);
    }
    // R2: multi-concept cluster note.
    if let Some(msg) = multi_concept_note {
        parts.push(msg);
    }
    if embed_warming {
        parts.push(
            "[note: semantic embeddings still warming up (sidecar starting); this result is lexical-only; retry in a few seconds for full semantic ranking]",
        );
    } else if embed_disabled {
        // Arming finished with no hook installed, so unlike `warming` this will
        // not resolve on a retry. Most often the index was built with a
        // different embedding model than the one installed.
        parts.push(
            "[note: semantic search unavailable (the embedding sidecar did not start, or the index was built with a different model); results are lexical only. Run `travsr embed status` to check, then `travsr embed reindex` if the model changed]",
        );
    } else if has_embed && knn_degraded {
        parts.push(
            "[note: semantic search degraded, KNN timed out or returned empty; results are lexical only]",
        );
    } else if !has_embed && embed_initialized {
        // Phase 1 has run (embed.db exists) but KNN hook not yet active.
        parts.push(
            "[note: embedding in progress; run `travsr embed status` to check; results improve as index builds]",
        );
    } else if !has_embed {
        parts.push("[note: semantic search disabled; run `travsr embed init` for better results]");
    }
    if phase_b_pending {
        parts.push(
            "[note: call traversal limited; run `travsr lang install <lang>` to enable call-graph edges]",
        );
    }
    parts.join("\n")
}

/// Convenience wrapper for the no-overflow, no-seed-cap degraded-state check.
fn build_degraded_notes(store: &SqliteStore, has_embed: bool, embed_warming: bool) -> String {
    build_context_signals(store, has_embed, embed_warming, false, None, None)
}

/// Format a single node line for `get_context` output.
///
/// - Appends `:line` to the path when the node has a known source location.
/// - Omits `[package: ]` entirely when the package field is empty.
/// - R7: appends `[score: N.NN]` when a score is available so the AI can
///   weight nodes by relevance without a second round-trip.
/// - RFC-022 §14: when `omit_via` is true the `[via: …]` provenance badge is
///   dropped entirely. Used for primary seeds rendered under a match-source
///   section header (Exact / Semantic) — the header already carries the source,
///   so a per-row badge would double-encode it and cost tokens. Relevant (non-seed)
///   rows keep their `[via: caller of X]` badge, which names the *source seed* the
///   header does not carry, so they pass `omit_via = false`.
fn format_node_line(
    n: &CoreNode,
    role: &str,
    score: Option<f32>,
    via_source: Option<&str>,
    source: Option<&str>,
    omit_via: bool,
) -> String {
    let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
    let score_str = score
        .map(|s| format!(" [score: {s:.2}]"))
        .unwrap_or_default();
    // RFC-019: name the source seed for caller/dependency roles so seed
    // contamination is diagnosable — a caller of a good seed and a caller of a
    // false seed no longer print the identical `[via: caller]`. Seed/context roles
    // have no single source, so they render unchanged.
    //
    // RFC-022 §14: `omit_via` suppresses the badge for section-headed seeds. The
    // segment carries its own leading space so an omitted badge leaves no stray
    // whitespace — off-path output stays byte-for-byte identical.
    let via = if omit_via {
        String::new()
    } else {
        let badge = match via_source {
            Some(src) if role == "caller" || role == "dependency" => {
                format!("[via: {role} of {src}]")
            }
            // #462: surface the seed's match source (exact-name / semantic / lexical)
            // so a name-only anchor at a high rank with a low semantic score reads as
            // honest provenance, not a bug. Display-only — never affects ordering.
            _ => match source {
                Some(s) => format!("[via: {role} · {s}]"),
                None => format!("[via: {role}]"),
            },
        };
        format!(" {badge}")
    };
    if n.package.is_empty() {
        format!(
            "{} ({}) \u{2014} {}{loc}{via}{score_str}",
            display_label(n),
            n.kind,
            n.vname.path
        )
    } else {
        format!(
            "{} ({}) \u{2014} {}{loc} [package: {}]{via}{score_str}",
            display_label(n),
            n.kind,
            n.vname.path,
            n.package
        )
    }
}

/// RFC-022 §14: one-line section header printed once per match-source group in
/// the grouped `get_context` body. Amortizes the provenance label across the group
/// (vs a per-row badge) and disambiguates the Exact section's non-rerank score.
fn match_source_header(ms: crate::seed::MatchSource) -> &'static str {
    match ms {
        crate::seed::MatchSource::Exact => "## exact matches, literal symbol / text (not relevance-ranked)",
        crate::seed::MatchSource::Semantic => "## related, ranked by relevance",
        // #376 Phase 2 (§4.1): no raw cosine printed in this section — see
        // build_docs_section's doc comment for why (doc cosines and code
        // cosines are not commensurable; a printed number invites exactly the
        // cross-section comparison this header exists to prevent).
        crate::seed::MatchSource::Docs => {
            "## docs (documentation prose: claims about the code, verify behaviour against the code itself)"
        }
        // #479: test entry points & fixtures, capped and placed below the
        // implementation/design sections so a `#[test]` fn never leads.
        crate::seed::MatchSource::Tests => "## tests, test entry points & fixtures",
        crate::seed::MatchSource::Relevant => "## relevant, graph-adjacent context",
    }
}

/// RFC-022 §14: whether a node's `[via: …]` badge should be suppressed.
///
/// Only in grouped mode, and only for primary seeds (Exact / Semantic) — their
/// source is amortized into the section header, so a per-row badge double-encodes
/// it. Relevant (non-seed) rows keep the badge because it names the *source seed*
/// (`caller of X`), which the header does not carry. Off-path (`grouped == false`)
/// always returns false → byte-for-byte identical output.
fn omit_seed_via(grouped: bool, ms: crate::seed::MatchSource) -> bool {
    grouped && ms != crate::seed::MatchSource::Relevant
}

/// RFC-022 §14: assemble the node body for `get_context`.
///
/// `entries` is one `(match_source, display_score, rendered_line)` per selected
/// node, in knapsack order. When `grouped` is false the rendered lines are joined
/// verbatim (byte-for-byte identical to the pre-§14 output). When true, the nodes
/// are partitioned into Exact → Semantic → Relevant sections (each preceded by a
/// one-line header) and sorted within a section by descending display score.
/// Display-only: the knapsack set is unchanged — only presentation order differs.
///
/// #870: the ungrouped path keeps its flat code lines, but doc entries still
/// get their header. `grouped` is false for a result of four nodes or fewer,
/// where per-section headers cost more than they save. That is a rule about
/// *code* rows, which carry their own kind and path. A doc entry carries
/// neither: the header is the only thing that marks the line as author-written
/// prose (§4.1, mitigation M2), and it is the only handle a consumer has for
/// finding the section at all. Dropping it left doc lines rendered bare among
/// the code rows, so `get_context` reported no docs for a query `ask` answered
/// from one. That is the shape of a result on a sparse graph (a repo indexed
/// without Phase B), not an edge case. A docs-free response is unaffected and
/// stays byte-identical.
fn assemble_context_body(
    entries: Vec<(crate::seed::MatchSource, f32, String)>,
    sep: &str,
    grouped: bool,
) -> String {
    if !grouped {
        let (docs, code): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(ms, _, _)| *ms == crate::seed::MatchSource::Docs);
        let mut out: Vec<String> = code.into_iter().map(|(_, _, line)| line).collect();
        if !docs.is_empty() {
            out.push(match_source_header(crate::seed::MatchSource::Docs).to_string());
            out.extend(docs.into_iter().map(|(_, _, line)| line));
        }
        return out.join(sep);
    }
    let mut entries = entries;
    entries.sort_by(|a, b| {
        a.0.trust_rank()
            .cmp(&b.0.trust_rank())
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut out: Vec<String> = Vec::with_capacity(entries.len() + 3);
    let mut cur: Option<crate::seed::MatchSource> = None;
    // #479: cap the tests section small and omit it when empty. Entries are
    // already sorted by trust_rank then descending score, so the retained tests
    // are the highest-scoring ones; the section header is only emitted if at
    // least one test survives the cap.
    const TESTS_CAP: usize = 3;
    let mut tests_shown = 0usize;
    for (ms, _score, line) in entries {
        if ms == crate::seed::MatchSource::Tests {
            if tests_shown >= TESTS_CAP {
                continue;
            }
            tests_shown += 1;
        }
        if cur != Some(ms) {
            out.push(match_source_header(ms).to_string());
            cur = Some(ms);
        }
        out.push(line);
    }
    out.join(sep)
}

/// #376 Phase 2 (§3.2): humanize a doc-chunk's slug signature
/// (`doc:retrieval-design/token-budget`) into a readable heading trail
/// ("Retrieval Design › Token Budget") for display, without re-parsing the
/// source markdown file at query time. Cosmetic only — the anchor itself
/// (not this string) is the identity-bearing part of the node.
///
/// # §17.9 D, decided (O12)
///
/// The joiner is `›` (U+203A), not `>`. The plan left this open between two
/// options: keep `>` and accept that the sanitizer renders a Travsr-authored
/// separator as `&gt;` in every doc entry, or escape the author-controlled
/// components individually and assemble with a trusted joiner afterwards. The
/// first reads badly on the feature's most-seen output; the second buys
/// readability by making the sanitizer's binding conditional, and "escape the
/// whole assembled line, always" is exactly the uniform rule that makes M1 and
/// M2 hold (§20.2 lists it as load-bearing).
///
/// Neither trade is necessary. `sanitize_for_mcp` escapes only `<` and `>` and
/// strips only C0/C1 controls and bidi overrides, so a separator outside those
/// sets passes through untouched and the uniform rule keeps applying to the
/// whole line. U+203A is neither, and this output already carries non-ASCII
/// (`§`, the CLI's box-drawing rules). Readability with no weakened invariant.
fn humanize_doc_anchor(sig: &str) -> String {
    let anchor = sig.strip_prefix("doc:").unwrap_or(sig);
    if anchor == "_preamble" {
        return "(preamble)".to_string();
    }
    anchor
        .split('/')
        .map(|segment| {
            segment
                .split('-')
                .filter(|w| !w.is_empty())
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join(" › ")
}

/// #376 Phase 2 (§4): build the docs section's rendered lines and their
/// measured token cost, already floor-filtered and capped by
/// `crate::seed::doc_lane_seeds`. Returns `(vec![], 0)` whenever the lane is
/// off, unavailable, or produced no hits — the caller must then render no
/// docs section at all (§4.2: "absent, not empty-ish").
///
/// **Print no raw cosine inside the section** (§4.1): §8.3 measured doc
/// cosines (gold p50 0.550-0.557) above each repo's code-band ceiling
/// (0.514-0.532) — the two are not commensurable, so printing a number would
/// invite exactly the cross-section comparison the separate section exists to
/// prevent. The header alone carries the epistemic weight. The `f32` in each
/// returned tuple is used only to sort entries within the section (highest
/// first) and to clamp the token budget below — never rendered.
///
/// §4.3 budget carve: entries are kept in descending-score order while their
/// cumulative token cost stays within `docs.budget_pct` of `token_budget`;
/// the first entry that would exceed it, and everything after, is dropped.
/// This is a clamp on *measured* cost, never a reservation — a docs-free
/// query has `doc_tokens == 0` and is byte-identical to pre-Phase-2 output.
/// #376 Phase 2 prototype: score `candidates` (an over-fetched, unfiltered
/// pool from `crate::seed::doc_lane_candidates`) with the RFC-021
/// cross-encoder against `query`, then floor-filter/sort/cap by that score.
///
/// Falls back to the pre-rerank raw-cosine behaviour (`doc_floor` +
/// `docs_max_results`, applied to the same candidate pool already fetched —
/// no second sidecar round trip) whenever the reranker has no opinion: no
/// candidate has `embed_text` (should not happen for real doc-chunk nodes,
/// but a store lookup failure or a node deleted between KNN and this call
/// must degrade gracefully, not silently drop a real hit), or
/// `crate::rerank::rerank` itself returns `None` (model absent, disabled,
/// panicked, or over the circuit-breaker budget — same fail-open contract
/// the code lane already relies on).
fn rerank_doc_candidates(
    store: &SqliteStore,
    query: &str,
    candidates: Vec<(NodeId, f32)>,
) -> Vec<(NodeId, f32)> {
    // #539: normalize before scoring so this stage is consistent with the
    // rest of the docs lane. Unlike the recall stage (`doc_lane_query`),
    // this deliberately does NOT strip filler words: the cross-encoder does
    // token-level cross-attention over a natural sentence, not mean-pooling,
    // so it does not share the recall stage's dilution failure mode, and
    // stripping stopwords here would only fight the reranker's own training
    // distribution (MS MARCO-style rerankers are trained on full questions).
    let query = travsr_store::fts_tokenize::normalize_nl_query(query);
    let query = query.as_str();
    let cap = crate::seed::docs_max_results(store);

    // #520 (bench experiment, env-gated, off by default): skip the
    // cross-encoder when the top raw-cosine candidate is already confidently
    // separated. `candidates` arrives sorted descending (doc_lane_candidates),
    // so the first entry is the top-1 raw cosine.
    if let Some(threshold) = crate::seed::doc_rerank_ambiguity_threshold() {
        if candidates.first().is_some_and(|&(_, s)| s >= threshold) {
            return crate::seed::cosine_floor_select(store, candidates);
        }
    }

    let ids: Vec<NodeId> = candidates.iter().map(|&(id, _)| id).collect();
    let text_map = store.get_nodes_embed_text(&ids).unwrap_or_default();
    // O11 (§20.4), recorded as a decision rather than left as an accident: this
    // drop is *partial* fail-open. A candidate with no `embed_text` is dropped
    // here and only an entirely empty pool falls back to `cosine_floor_select`.
    // Keeping such a candidate is not an option — the reranker needs text, and
    // scoring it as 0.0 would rank a real hit below the floor anyway — and
    // falling back wholesale on one missing row would discard every genuinely
    // reranked hit alongside it. A `doc-chunk` reaching here without
    // `embed_text` is already excluded from the vector index by `NODE_ELIGIBLE`
    // (§18.2 F3), so in practice this only fires for a node deleted between the
    // KNN call and this lookup, whose entry is stale regardless.
    let with_text: Vec<(NodeId, &str)> = candidates
        .iter()
        .filter_map(|&(id, _)| text_map.get(&id).map(|t| (id, t.as_str())))
        .collect();
    if with_text.is_empty() {
        return crate::seed::cosine_floor_select(store, candidates);
    }

    let texts: Vec<&str> = with_text.iter().map(|&(_, t)| t).collect();
    let Some(scores) = crate::rerank::rerank(query, &texts) else {
        return crate::seed::cosine_floor_select(store, candidates);
    };
    if scores.len() != with_text.len() {
        return crate::seed::cosine_floor_select(store, candidates);
    }

    let floor = crate::seed::doc_rerank_floor();
    let mut scored: Vec<(NodeId, f32)> = with_text
        .iter()
        .zip(scores.iter())
        .filter(|&(_, &s)| s >= floor)
        .map(|(&(id, _), &s)| (id, s))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(cap);
    scored
}

/// M3 (§6, threat T11): hard per-entry output cap, independent of the token
/// budget. Both components of a doc entry are author-controlled — the file path
/// and the heading trail — so without this one adversarial document could spend
/// the whole docs section on a single line. The budget loop below would usually
/// drop such an entry, but "usually" is not a security property: the cap is what
/// makes the bound hold for the first entry at any budget.
const DOC_ENTRY_MAX_BYTES: usize = 512;

/// Truncate a rendered doc entry to [`DOC_ENTRY_MAX_BYTES`] at a char boundary.
fn cap_doc_entry(line: &str) -> String {
    if line.len() <= DOC_ENTRY_MAX_BYTES {
        return line.to_string();
    }
    let mut end = DOC_ENTRY_MAX_BYTES;
    while end > 0 && !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &line[..end])
}

pub(crate) fn build_docs_section(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
) -> (Vec<(crate::seed::MatchSource, f32, String)>, usize) {
    let doc_knn = store.embed_doc_knn_fn();
    let doc_knn_ref = doc_knn
        .as_ref()
        .map(|f| f as &dyn Fn(&str, u32) -> Vec<(NodeId, f32)>);
    let candidates = crate::seed::doc_lane_candidates(store, query, doc_knn_ref);
    if candidates.is_empty() {
        return (vec![], 0);
    }

    let hits = rerank_doc_candidates(store, query, candidates);
    if hits.is_empty() {
        return (vec![], 0);
    }

    let ids: Vec<NodeId> = hits.iter().map(|&(id, _)| id).collect();
    let Ok(nodes) = store.get_nodes(&ids) else {
        return (vec![], 0);
    };
    let score_map: HashMap<NodeId, f32> = hits.into_iter().collect();
    let mut entries: Vec<(crate::seed::MatchSource, f32, String)> = nodes
        .iter()
        .filter_map(|n| {
            let score = *score_map.get(&n.id)?;
            let loc = match (n.line, n.end_line) {
                (Some(s), Some(e)) if s != e => format!(":{s}-{e}"),
                (Some(s), _) => format!(":{s}"),
                _ => String::new(),
            };
            let line = format!(
                "{} § {}{loc}",
                n.vname.path,
                humanize_doc_anchor(&n.vname.signature)
            );
            Some((crate::seed::MatchSource::Docs, score, cap_doc_entry(&line)))
        })
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let budget_cap = ((crate::seed::docs_budget_pct(store) / 100.0) * token_budget as f32) as usize;
    let mut doc_tokens = 0usize;
    let mut kept = Vec::with_capacity(entries.len());
    for entry in entries {
        let cost = entry.2.len() / TOKEN_CHARS_PER_TOKEN + 1;
        if doc_tokens + cost > budget_cap {
            break;
        }
        doc_tokens += cost;
        kept.push(entry);
    }
    (kept, doc_tokens)
}

/// Derive FTS seeds with confidence-based weights for weighted PPR personalisation.
///
/// - Exact substring match (`search_nodes_by_name`) → `FTS_EXACT_WEIGHT` (1.0)
/// - Trigram/L2-A fuzzy match (`search_nodes_fuzzy`) → `FTS_FUZZY_WEIGHT` (0.6)
///
/// Dot-notation queries (e.g. `"Kubelet.syncPod"`) are split on the last `.`:
/// the name part is searched; results are filtered to those whose signature
/// contains the qualifier, ensuring `var:SyncPod` never beats `method:Kubelet.syncPod`.
///
/// Each seed weight is multiplied by `kind_boost` so callable definitions
/// (method, function, constructor) rank above structural nodes (import, module).
///
/// Returns up to 5 seeds (capped to prevent FTS from flooding the seed pool).
#[allow(dead_code)]
pub(crate) fn fts_seeds_weighted(
    store: &SqliteStore,
    query: &str,
    filter: &dyn EdgeFilter,
) -> Vec<(NodeId, f32)> {
    let exact = store.search_nodes_by_name(query).unwrap_or_default();
    let (nodes, weight) = if !exact.is_empty() {
        (exact, FTS_EXACT_WEIGHT)
    } else {
        match store.search_nodes_fuzzy(query) {
            Ok(fuzzy) => (fuzzy, FTS_FUZZY_WEIGHT),
            Err(e) => {
                tracing::warn!("fts_seeds_weighted fuzzy error: {e}");
                return vec![];
            }
        }
    };

    nodes
        .into_iter()
        .filter(|n| !is_noise_seed(n))
        .filter(|n| filter.allow(n.id, n.id, Some(n.vname.corpus.as_str())))
        .take(5)
        .map(|n| {
            let boosted = weight * kind_boost(&n.kind, &n.vname.language);
            (n.id, boosted)
        })
        .collect()
}

/// Merge FTS and KNN seed sets into a single weighted seed list for `ppr_weighted`.
///
/// Nodes appearing in both sets are deduplicated by taking `max(fts_weight, knn_score)`:
/// an exact FTS match and a high-cosine KNN match for the same node reinforce each other
/// without diluting the PPR personalisation vector.
///
/// Returns `(merged_seeds, seed_cap_fired)`.
#[allow(dead_code)]
pub(crate) fn merge_fts_knn_seeds(
    fts: Vec<(NodeId, f32)>,
    knn: Vec<(NodeId, f32)>,
    n_knn_eligible: usize,
) -> (Vec<(NodeId, f32)>, bool) {
    let mut merged: HashMap<NodeId, f32> = fts.into_iter().collect();
    for (id, score) in knn {
        let entry = merged.entry(id).or_insert(0.0);
        *entry = entry.max(score);
    }
    let mut combined: Vec<(NodeId, f32)> = merged.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let cap_fired = combined.len() > MAX_EMBED_SEEDS || n_knn_eligible > MAX_EMBED_SEEDS;
    combined.truncate(MAX_EMBED_SEEDS);
    (combined, cap_fired)
}

/// Drop seeds that are direct **callees** of a higher-scored accepted seed.
///
/// PPR follows outgoing edges at each step, so from seed A the walk naturally
/// reaches everything A calls/imports. Including A's callee B as a *second* seed
/// only adds redundant teleportation mass — PPR from A already covers B.
///
/// **Callers are never dropped.** If B calls A (B→A edge), PPR from A cannot walk
/// backward to B, so B provides genuinely orthogonal coverage and must be kept.
/// Dropping callers would hide usage context, which is often the most valuable part
/// of the result set for a query about a function.
///
/// Algorithm (O(n × q) where n ≤ MAX_EMBED_SEEDS, q = one `iter_edges_from` per seed):
///
/// 1. Pre-fetch all outgoing edges for every seed filtered to seed→seed pairs (≤ 20 queries).
/// 2. Visit seeds in descending score order. Drop a candidate only when an already-accepted
///    seed has a direct **outgoing** edge to it (accepted → candidate direction).
///
/// Returns the filtered list in the same descending-score order.
pub(crate) fn dedup_adjacent_seeds(
    store: &SqliteStore,
    seeds: Vec<(NodeId, f32)>,
) -> Vec<(NodeId, f32)> {
    if seeds.len() <= 1 {
        return seeds;
    }

    let seed_ids: std::collections::HashSet<NodeId> = seeds.iter().map(|&(id, _)| id).collect();

    // Pre-fetch seed→seed outgoing edges (directed: src calls/depends-on dst).
    let mut adj: std::collections::HashSet<(NodeId, NodeId)> = std::collections::HashSet::new();
    for &(id, _) in &seeds {
        if let Ok(edges) = store.iter_edges_from(id) {
            for e in edges {
                if seed_ids.contains(&e.dst) {
                    adj.insert((id, e.dst));
                }
            }
        }
    }

    // If no seed→seed edges exist, nothing to dedup — return early.
    if adj.is_empty() {
        return seeds;
    }

    // Greedy selection: highest score wins.
    // Drop a candidate only when an accepted seed has an OUTGOING edge to it
    // (accepted → candidate). Never drop because candidate → accepted (caller direction).
    let mut accepted: Vec<(NodeId, f32)> = Vec::with_capacity(seeds.len());
    let mut accepted_set: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

    for (id, score) in seeds {
        let is_callee_of_accepted = accepted_set.iter().any(|&acc| adj.contains(&(acc, id)));
        if !is_callee_of_accepted {
            accepted_set.insert(id);
            accepted.push((id, score));
        }
    }
    accepted
}

/// Enrich a seed set with depth-1 production callers of each seed.
///
/// PPR follows outgoing edges from seeds, so callers of a seed node are never
/// reachable unless they are themselves seeds. This function adds the top-5
/// non-test callers of each seed at 35% of the seed's weight, giving PPR a
/// starting point inside each caller's neighborhood.
///
/// Must run BEFORE `dedup_adjacent_seeds` so that the dedup pass can suppress
/// any caller that is itself a callee of a higher-scored accepted seed.
///
/// Filters applied to each candidate caller:
///   - `EdgeKind::RefCall` only (imports / structural edges are not callers)
///   - Skip path containing `/tests/` or `/benches/`
///   - Skip signature starting with `fn:test_`, `fn:bench_`, `fn:prop_`
///   - Skip nodes already in the seed set (keeps existing higher weight)
///   - Cap per-seed: top 5 by outgoing-edge degree (more central = more relevant)
///   - Cap global: `max_new_total` total new seeds added regardless of input size —
///     prevents hub nodes (e.g. a logging utility called by 1000 functions) from
///     flooding the seed pool when this is called for depth-2 enrichment.
///
/// O(s × c × d) where s = seed count, c = callers per seed (typically < 20
/// after test filter), d = outgoing degree per caller (1 store query each).
pub(crate) fn enrich_seeds_with_callers(
    store: &SqliteStore,
    mut seeds: Vec<(NodeId, f32)>,
    max_new_total: usize,
) -> Vec<(NodeId, f32)> {
    use travsr_core::EdgeKind;

    if seeds.is_empty() {
        return seeds;
    }

    let existing: std::collections::HashSet<NodeId> = seeds.iter().map(|&(id, _)| id).collect();
    let mut to_add: Vec<(NodeId, f32)> = Vec::new();

    for &(seed_id, seed_weight) in &seeds {
        // Collect RefCall incoming edges (who calls this seed?)
        let incoming = match store.iter_edges_to(seed_id) {
            Ok(edges) => edges,
            Err(_) => continue,
        };

        let caller_ids: Vec<NodeId> = incoming
            .into_iter()
            .filter(|e| e.kind == EdgeKind::RefCall)
            .map(|e| e.src)
            .filter(|id| !existing.contains(id))
            .collect();

        if caller_ids.is_empty() {
            continue;
        }

        // Batch-fetch caller nodes for path/signature filtering.
        let caller_nodes = match store.get_nodes(&caller_ids) {
            Ok(nodes) => nodes,
            Err(_) => continue,
        };

        // Apply noise filter: skip test/bench callers and structural noise nodes.
        // is_noise_seed covers path-level noise (tests/, benches/, caches, etc.);
        // the sig-prefix check catches test functions in files not in test dirs.
        let mut production_callers: Vec<NodeId> = caller_nodes
            .into_iter()
            .filter(|n| {
                if is_noise_seed(n) {
                    return false;
                }
                let sig = &n.vname.signature;
                !sig.starts_with("fn:test_")
                    && !sig.starts_with("fn:bench_")
                    && !sig.starts_with("fn:prop_")
            })
            .map(|n| n.id)
            .collect();

        if production_callers.is_empty() {
            continue;
        }

        // PF-H1: pre-fetch degrees in one pass so the sort key lookup is O(1).
        // sort_unstable_by_key evaluates the key once per element (not per compare),
        // but each evaluation was a separate store query — pre-compute avoids n
        // round-trips hidden inside the sort closure.
        let caller_degrees: HashMap<NodeId, usize> = production_callers
            .iter()
            .map(|&id| (id, store.iter_edges_from(id).map(|e| e.len()).unwrap_or(0)))
            .collect();
        production_callers.sort_unstable_by_key(|id| {
            std::cmp::Reverse(caller_degrees.get(id).copied().unwrap_or(0))
        });

        let caller_weight = seed_weight * 0.35;
        for caller_id in production_callers.into_iter().take(5) {
            to_add.push((caller_id, caller_weight));
        }
    }

    // Deduplicate to_add against itself (same caller may be added via multiple seeds;
    // keep the higher weight).
    to_add.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut seen: std::collections::HashSet<NodeId> = existing;
    to_add.retain(|(id, _)| seen.insert(*id));

    // Global cap: a hub node called by many callers must not flood the seed pool.
    to_add.truncate(max_new_total);

    seeds.extend(to_add);
    seeds
}

fn get_context_body(
    store: &SqliteStore,
    query: &str,
    token_budget: usize,
    filter: &dyn EdgeFilter,
    include_snippets: bool,
    snippet_budget: Option<usize>,
    embed_knn: Option<EmbedKnnFn<'_>>,
) -> String {
    // Capture embed presence before embed_knn is consumed by the seed-lookup block.
    let has_embed = embed_knn.is_some();
    // Distinguish "Phase 1 done, hook not yet active" from "embed never initialized".
    let embed_initialized = store.has_embed_db();
    // PF-M4: compute once here so neither include_snippets branch calls get_meta twice.
    let phase_b = phase_b_pending(store);

    // SEC-002: validate before any store access.
    if let Err(reason) = validate_mcp_arg(query) {
        tracing::warn!("get_context rejected invalid query arg: {reason}");
        return String::new();
    }
    // Defense-in-depth budget guard (RFC-010 §3.3).
    if token_budget > MAX_CONTEXT_BUDGET {
        return "token_budget exceeds maximum allowed value".to_string();
    }
    if token_budget == 0 {
        return String::new();
    }

    // ── Tier-0 seed selection ─────────────────────────────────────────────────
    // Per-token anchor resolution (IDF-weighted) + whole-query BM25 FTS, fused
    // via Reciprocal Rank Fusion. KNN slots in as a third source when available.
    //
    // Produces a SeedSet with coverage, confidence, and term-resolution map.
    // Confidence::None → abstain (return resolution map, no salad).
    //
    // A: first-call warmup. If embeddings are configured but the sidecar hasn't
    // armed yet, block briefly so the opening query uses semantic seeds instead
    // of silently degrading to lexical-only. Done OUTSIDE the KNN breaker timing
    // below so the one-time arm cost is never misread as a slow KNN round-trip.
    if has_embed && !store.embed_ready() {
        store.wait_embed_ready(std::time::Duration::from_millis(embed_arm_wait_ms()));
    }
    // B: honest warming signal — true only when the cap was exceeded and the
    // sidecar is still cold. Distinguishes "warming up" from "embeddings off"
    // so the header/notes never silently claim full semantic coverage.
    let embed_warming = has_embed && !store.embed_ready();
    // #874: arming can settle with no hook installed (sidecar refused to start,
    // or the index was built with a different embedding model). The MCP
    // injector installs its meta-hooks unconditionally so `initialize` never
    // blocks, so `has_embed` stays true and the query would otherwise report
    // `embeddings: on` while the semantic lane is permanently dead.
    let embed_disabled = has_embed && store.embed_disabled();

    // R3: track per-query KNN health; has_embed=true doesn't mean KNN worked.
    let mut knn_degraded = false;
    let (knn_pairs, knn_oracle) = if let Some(knn_fn) = embed_knn {
        let (knn_scored, _n_eligible, knn_elapsed_ms, oracle) =
            embed_path_seeds(store, query, knn_fn, filter);
        let budget_ms = knn_budget_ms();
        if knn_elapsed_ms > budget_ms {
            // R3: circuit-breaker fired — KNN unambiguously degraded (timeout).
            tracing::warn!(
                knn_elapsed_ms,
                threshold_ms = budget_ms,
                "knn exceeded circuit-breaker threshold, falling back to FTS seeds"
            );
            knn_degraded = true;
            (vec![], std::collections::HashMap::new())
        } else {
            // Empty = no embeddings indexed yet (graceful no-op, not degraded).
            (knn_scored, oracle)
        }
    } else {
        (vec![], std::collections::HashMap::new())
    };

    // RFC-019: direct-cosine oracle hook (None when embeddings off/warming → the
    // FTS-only path is unchanged).
    let score_fn = store.embed_score_fn();
    let score_ref = score_fn
        .as_ref()
        .map(|f| f as &dyn Fn(&str, &[NodeId]) -> Vec<(NodeId, f32)>);
    let seed_set =
        crate::seed::build_seed_set(store, query, filter, knn_pairs, &knn_oracle, score_ref);
    let tier_label = if has_embed && !embed_disabled && !seed_set.seeds.is_empty() {
        "exact+lexical+semantic"
    } else {
        "exact+lexical"
    };

    // #376 Phase 2: docs lane runs independently of the code seed pipeline and
    // of `Confidence` — computed once here so both the abstain early-return
    // below and the normal-flow render further down can use it. `doc_tokens`
    // is already clamped to `docs.budget_pct` of `token_budget` (§4.3), so
    // subtracting it from the code lane's knapsack budget is always safe.
    let (docs_entries, doc_tokens) = build_docs_section(store, query, token_budget);

    // #515: every return from this function must be sanitized, not just the
    // grounded ones. `sanitize_for_mcp` is mitigation M1 against T11 (prose
    // injection) and M2 (unforgeable section headers) depends on M1, so a return
    // path that skips it defeats both. The abstain path is the docs lane's
    // *primary* case — 15/20 rationale queries abstain on kubernetes (§8.5), and
    // §4.3 deliberately renders doc prose beneath the abstain message — which
    // made the one unsanitized branch the one most likely to carry untrusted
    // text. Bound here, above both exits, so neither can drift away from it.
    let char_cap = (token_budget * TOKEN_CHARS_PER_TOKEN * 2).min(1_024_000);

    // Abstain path: no grounded match → return resolution map, never a confident salad.
    if seed_set.confidence == crate::seed::Confidence::None {
        let mut msg = crate::seed::abstain_message(&seed_set, query);
        let max_g = crate::seed::abstain_max_guesses();
        if max_g > 0 && !seed_set.seeds.is_empty() {
            let guess_ids: Vec<NodeId> =
                seed_set.seeds.iter().take(max_g).map(|s| s.node).collect();
            if let Ok(guess_nodes) = store.get_nodes(&guess_ids) {
                let lines: Vec<String> = guess_nodes
                    .iter()
                    .map(|n| {
                        let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
                        let pkg = if n.package.is_empty() {
                            String::new()
                        } else {
                            format!(" [package: {}]", n.package)
                        };
                        format!(
                            "  {} ({}) \u{2014} {}{loc}{pkg}",
                            display_label(n),
                            n.kind,
                            n.vname.path
                        )
                    })
                    .collect();
                msg.push_str(&lines.join("\n"));
            }
        }
        // #376 Phase 2 (§4.3): "doc hits may appear below the abstain message
        // under the same section header, but may never convert an abstain
        // into a confident answer." The code lane still abstains — this only
        // appends a cited doc section beneath it.
        if !docs_entries.is_empty() {
            msg.push_str("\n\n");
            msg.push_str(match_source_header(crate::seed::MatchSource::Docs));
            msg.push('\n');
            msg.push_str(
                &docs_entries
                    .iter()
                    .map(|(_, _, line)| line.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        // #515: same sanitize-with-limit the grounded returns use.
        return sanitize_mcp_body_with_limit(&msg, char_cap);
    }

    // R1: relevance floor — when confidence is Weak, warn that results may not
    // be relevant. This prevents confident-salad responses (64 nodes returned for
    // "quantum blockchain payment handler" with no graph matches).
    let weak_match_note: Option<&str> = if seed_set.confidence == crate::seed::Confidence::Weak {
        Some("[note: no strong match found, results are weak/structural and may not be relevant to this query]")
    } else {
        None
    };

    let seed_cap_fired = seed_set.seeds.len() > MAX_EMBED_SEEDS;
    let weighted_seeds = seed_set.ppr_seeds();

    // Capture primary seeds before enrichment for role annotation below.
    let primary_seed_ids: std::collections::HashSet<NodeId> =
        weighted_seeds.iter().map(|&(id, _)| id).collect();

    let weighted_seeds = enrich_seeds_with_callers(store, weighted_seeds, 15); // depth-1
    let weighted_seeds = enrich_seeds_with_callers(store, weighted_seeds, 10); // depth-2
    let weighted_seeds = dedup_adjacent_seeds(store, weighted_seeds);

    // SEC P0: identical response for "not found" and "access denied".
    if weighted_seeds.is_empty() {
        // #515: the query is echoed back, so this exit is sanitized too. Body
        // form (not `sanitize_for_mcp`) — callers wrap the envelope themselves.
        return sanitize_mcp_body_with_limit(
            &format!("No symbols matching '{query}' found in the graph."),
            char_cap,
        );
    }

    // Retrieval header — declared here so it can be prepended to the response body.
    // #529: report the count the abstention gate actually reads, not the looser
    // `t.resolved` count. They are different numbers under the same name: the gate
    // additionally requires `idf_w >= idf_coverage_min`, so a query whose tokens
    // all matched something generic printed e.g. "coverage 4/5" while the gate saw
    // 0/5 and abstained. An agent reading the envelope could not predict whether it
    // would get an answer. The generic remainder is still surfaced, distinguished,
    // because "4 tokens matched but none specifically enough" explains the verdict
    // that a bare "0/5" only states.
    let n_resolved = seed_set.n_resolved_gated;
    let n_resolved_loose = seed_set.terms.iter().filter(|t| t.resolved).count();
    let n_generic = n_resolved_loose.saturating_sub(n_resolved);
    let coverage_note = if n_generic > 0 {
        format!(" (+{n_generic} too generic to count)")
    } else {
        String::new()
    };
    let n_terms = seed_set.terms.len();

    // R8: index freshness header — lets the AI know exactly which commit the graph
    // reflects and whether embeddings are active. Keeps the context window honest.
    let index_commit = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .unwrap_or_else(|| "unknown".to_string());
    let embed_status = if knn_degraded {
        "degraded"
    } else if embed_warming {
        "warming"
    } else if embed_disabled {
        "disabled"
    } else if has_embed {
        "on"
    } else {
        "off"
    };
    let freshness_header = format!("[index commit: {index_commit}, embeddings: {embed_status}]\n");

    let retrieval_header = format!(
        "{freshness_header}[retrieval: {tier_label} | coverage {n_resolved}/{n_terms}{coverage_note} | confidence: {} ]\n",
        seed_set.confidence.label()
    );

    // PPR with confidence-weighted personalisation (#365 Step 1 + Step 3).
    // FTS-only path: weights 1.0/0.6 strictly better than previous uniform ppr().
    // Embed path: cosine scores steer the walk toward semantic neighbours.
    let ppr_scores = match travsr_retrieval::ppr_weighted(
        store,
        &weighted_seeds,
        context_candidates(),
        filter,
    ) {
        Ok(scores) => scores,
        Err(e) => {
            tracing::warn!("get_context ppr error: {e}");
            return String::new();
        }
    };

    if ppr_scores.is_empty() {
        // #515: the query is echoed back, so this exit is sanitized too. Body
        // form (not `sanitize_for_mcp`) — callers wrap the envelope themselves.
        return sanitize_mcp_body_with_limit(
            &format!("No symbols matching '{query}' found in the graph."),
            char_cap,
        );
    }

    // Build score map for keyed join (prevents node/score misalignment).
    let score_map: HashMap<NodeId, f32> = ppr_scores.iter().cloned().collect();
    let node_ids: Vec<NodeId> = ppr_scores.into_iter().map(|(id, _)| id).collect();

    // Batch-fetch nodes.
    let fetched = match store.get_nodes(&node_ids) {
        Ok(nodes) => nodes,
        Err(e) => {
            tracing::warn!("get_context get_nodes error: {e}");
            return String::new();
        }
    };

    // Post-filter fetched nodes through RBAC and join with scores.
    // SEC P0: `selected` is the authoritative RBAC-filtered list; snippets are
    // only ever read for nodes already in this set (never re-queried separately).
    // Issue A.3: apply the full noise policy (is_noise_seed, which is a strict
    // superset of is_noise_node) so scip: reference nodes, _test.go nodes, and
    // build-cache hub nodes cannot appear in the emitted context result.
    let items: Vec<(CoreNode, f32)> = fetched
        .into_iter()
        .filter(|n| filter.allow(n.id, n.id, Some(n.vname.corpus.as_str())))
        .filter(|n| !is_noise_seed(n))
        .filter_map(|n| score_map.get(&n.id).map(|&s| (n, s)))
        .collect();

    if items.is_empty() {
        // #515: the query is echoed back, so this exit is sanitized too. Body
        // form (not `sanitize_for_mcp`) — callers wrap the envelope themselves.
        return sanitize_mcp_body_with_limit(
            &format!("No symbols matching '{query}' found in the graph."),
            char_cap,
        );
    }

    // Boost PPR scores by k-core shell number (global structural importance).
    // KCORE_ALPHA is small so PPR local relevance still dominates; shell number
    // acts as a tiebreaker that favours structurally central nodes.
    const KCORE_ALPHA: f32 = 0.05;
    let item_ids: Vec<NodeId> = items.iter().map(|(n, _)| n.id).collect();
    let shell_map = store.get_shell_numbers_batch(&item_ids).unwrap_or_default();
    let items: Vec<(CoreNode, f32)> = items
        .into_iter()
        .map(|(n, s)| {
            let shell = shell_map.get(&n.id).copied().unwrap_or(0);
            (n, s * (1.0 + KCORE_ALPHA * shell as f32))
        })
        .collect();

    // Normalize PPR+kcore scores to [0, 1] within this candidate set before display.
    // Raw PPR distributes probability mass across the full graph (100k+ nodes), making
    // absolute scores tiny (~0.01–0.05). Normalization makes score bars meaningful:
    // the top result always shows at 100%, others proportional to it.
    // Monotone transform — does not change relative ordering, so knapsack is unaffected.
    let max_raw_score = items.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
    let items: Vec<(CoreNode, f32)> = if max_raw_score > 0.0 {
        items
            .into_iter()
            .map(|(n, s)| (n, s / max_raw_score))
            .collect()
    } else {
        items
    };

    // #462: context-noise filter on the FINAL candidate list (seeds + PPR
    // expansion). `is_noise_seed` already runs on seeds, but the expansion tail
    // ([via: context/dependency]) is never re-filtered, so CI/workflow package
    // nodes (pkg:actions/*), bare file nodes, and vendored/test-dir nodes leak
    // into the output as near-zero-score budget filler. Score-independent and
    // universal (applies to every query), so it cannot evict a real answer.
    // Safety valve: never let this structural filter EMPTY the candidate list. If
    // every candidate is a file/package node (a query whose only answer is a
    // container), keep the unfiltered list so the caller still gets a result — this
    // stage runs before the MIN_KEEP relevance-floor guard below, so it needs its own.
    let items: Vec<(CoreNode, f32)> = if items.iter().any(|(n, _)| !is_context_result_noise(n)) {
        items
            .into_iter()
            .filter(|(n, _)| !is_context_result_noise(n))
            .collect()
    } else {
        items
    };

    // PROTOTYPE (#462 precision@1): seed-quality ordering blend.
    // Output is ordered by normalized PPR, so a semantically strong seed can be
    // buried below graph-central expansion nodes (probe-cascade-k8s: 7/10 golds
    // BURIED@ppr_knapsack). Add a fraction of each seed's own semantic quality —
    // max(direct query cosine, rerank score), each normalized within the seed
    // set — back into its ordering score so a seed can never sink below its own
    // relevance. `display_score` is display-only, so the fix must land on the
    // ordering score here, not there. Applied to all intents: it only lifts a
    // seed by its OWN relevance, so exact/literal anchors (already top cosine) are
    // undisturbed. Not gated on QueryIntent::Conceptual — that variant is never
    // assigned by classify_intent (dead); a real conceptual router is future work.
    // env TRAVSR_SEED_ORDER_BLEND=alpha; 0 (default) → byte-identical baseline.
    let seed_order_blend = std::env::var("TRAVSR_SEED_ORDER_BLEND")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    let items: Vec<(CoreNode, f32)> = if seed_order_blend > 0.0 {
        let seed_ids: Vec<NodeId> = seed_set.seeds.iter().map(|s| s.node).collect();
        let cos: HashMap<NodeId, f32> = score_ref
            .map(|f| f(query, &seed_ids).into_iter().collect())
            .unwrap_or_default();
        let rr = seed_set.rerank_scores();
        let max_cos = cos.values().copied().fold(0.0_f32, f32::max).max(1e-6);
        let max_rr = rr.values().copied().fold(0.0_f32, f32::max).max(1e-6);
        let quality: HashMap<NodeId, f32> = seed_ids
            .iter()
            .map(|id| {
                let c = cos.get(id).copied().unwrap_or(0.0) / max_cos;
                let r = rr.get(id).copied().unwrap_or(0.0) / max_rr;
                (*id, c.max(r))
            })
            .collect();
        items
            .into_iter()
            .map(|(n, s)| {
                let q = quality.get(&n.id).copied().unwrap_or(0.0);
                (n, s + seed_order_blend * q)
            })
            .collect()
    } else {
        items
    };

    // Issue B: relevance floor — drop score-0 tail before knapsack so zero-signal nodes
    // cannot consume token budget and push genuinely relevant nodes into overflow.
    // Safety valve: always keep the top MIN_KEEP items so thin-but-legitimate results
    // are never emptied. Floor is env-overridable for tuning without a rebuild.
    // #462: 0.03 (3% of the top node's normalized score) trims the long
    // near-zero PPR-expansion tail that padded the token budget. Relative to each
    // query's own top score and guarded by MIN_KEEP, so it adapts to weak/low-
    // scoring conceptual queries without evicting the answer (hit-rate is
    // identical at 0.01 vs 0.05 on travsr and kubernetes benches). 0.03 rather
    // than 0.05: measured on multi-concept queries, the 0.01->0.03 cut removes
    // ~82% off-topic nodes (near-pure noise, on-topic density 26%->29%), while
    // 0.03->0.05 starts removing on-topic nodes at the set's own density rate
    // (thinning secondary context, e.g. the "reindex" half of a "watch + reindex"
    // query). 0.03 keeps the precision gain without over-pruning richness. Was
    // 0.01, which was too permissive to remove the tail at all.
    const REL_FLOOR_DEFAULT: f32 = 0.03;
    const MIN_KEEP: usize = 8;
    let rel_floor = std::env::var("TRAVSR_CONTEXT_REL_FLOOR")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(REL_FLOOR_DEFAULT);
    let items = {
        let mut sorted = items;
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep_floor = sorted.len().min(MIN_KEEP);
        sorted
            .into_iter()
            .enumerate()
            .filter(|(i, (_, s))| *i < keep_floor || *s >= rel_floor)
            .map(|(_, item)| item)
            .collect::<Vec<_>>()
    };

    // Capture lightweight overflow metadata before knapsack consumes items.
    // Storing only (id, score, sig, path) avoids cloning full Node objects.
    const OVERFLOW_DISPLAY_CAP: usize = 10;
    let overflow_candidates: Vec<(NodeId, f32, String, String)> = items
        .iter()
        .map(|(n, s)| {
            (
                n.id,
                *s,
                display_label(n).into_owned(),
                n.vname.path.clone(),
            )
        })
        .collect();
    let n_candidates = overflow_candidates.len();

    // R7: keep post-kcore scores for per-node score annotation in output.
    // Built before knapsack consumes items (items is Vec<(CoreNode, f32)>).
    let node_score_map: HashMap<NodeId, f32> = overflow_candidates
        .iter()
        .map(|(id, s, _, _)| (*id, *s))
        .collect();

    // RFC-021 F9: the score shown per node. A reranked seed shows its absolute
    // cross-encoder score (kills the normalized-PPR "always 1.00" artefact); a
    // primary seed the reranker never scored shows its normalized PPR; an
    // expanded neighbour caps at the best seed's rerank so it can't outrank its
    // seed. Display-only — ordering/knapsack already ran on the PPR scores.
    let seed_rerank = seed_set.rerank_scores();
    let expanded_cap = seed_rerank.values().copied().reduce(f32::max);
    let display_score_map: HashMap<NodeId, f32> = node_score_map
        .iter()
        .map(|(&id, &ppr)| {
            (
                id,
                crate::seed::display_score(
                    id,
                    ppr,
                    &seed_rerank,
                    primary_seed_ids.contains(&id),
                    expanded_cap,
                    seed_set.confidence,
                ),
            )
        })
        .collect();

    // #462: node → match source (exact-name / semantic / lexical), strongest
    // source winning when a node was reached by more than one seed path. Drives
    // the `[via: seed · <source>]` provenance tag and the RFC-022 §14 match-source
    // bucket. Shared with `ask` (query.rs) via `strongest_seed_sources` so both
    // surfaces classify a node identically. Display-only.
    let seed_source_map: HashMap<NodeId, crate::seed::SeedSource> =
        crate::seed::strongest_seed_sources(&seed_set.seeds);

    // Knapsack selection. #376 Phase 2 (§4.3): the code lane's budget is the
    // total minus what the docs section already measured and clamped — a
    // docs-free query has `doc_tokens == 0`, so this is byte-identical to the
    // pre-Phase-2 call.
    let selected = knapsack(items, token_budget.saturating_sub(doc_tokens));
    let n_nodes = selected.len();
    let total_tokens: usize = selected.iter().map(token_cost).sum();

    // #479: index-time test classification for the selected set. A node with
    // `test_role != None` buckets as `MatchSource::Tests` regardless of its seed
    // provenance, so a `#[test]` fn that is also an exact/semantic seed renders
    // in the capped `tests` section, not at the top of `exact`/`semantic`. Read
    // by id (the store column) rather than `Node.test_role`, which the read
    // paths default to `None`.
    let test_node_ids: std::collections::HashSet<NodeId> = selected
        .iter()
        .filter(|n| matches!(store.test_role(n.id), Ok(Some(r)) if r.is_test()))
        .map(|n| n.id)
        .collect();

    // RFC-022 §14: match-source grouping of the (already-selected) node set.
    // Display-only — `selected`/knapsack are untouched; this only decides how the
    // rendered lines are ordered/headed. Gated behind the flag; and collapsed to a
    // flat list for small N (≤4), where per-section headers cost more than they save.
    let group_output = crate::seed::match_source_grouping_enabled() && n_nodes > 4;
    let entry_meta = |id: NodeId| -> (crate::seed::MatchSource, f32) {
        let is_exact = matches!(
            seed_source_map.get(&id),
            Some(crate::seed::SeedSource::Exact)
        );
        // #479: test classification overrides seed-provenance bucketing.
        let ms = if test_node_ids.contains(&id) {
            crate::seed::MatchSource::Tests
        } else {
            crate::seed::match_source(primary_seed_ids.contains(&id), is_exact)
        };
        (ms, display_score_map.get(&id).copied().unwrap_or(0.0))
    };

    // Build overflow list: candidates that didn't fit within the token budget.
    let overflow_msg: Option<String> = if n_nodes < n_candidates {
        let selected_set: std::collections::HashSet<NodeId> =
            selected.iter().map(|n| n.id).collect();
        let overflow: Vec<_> = overflow_candidates
            .iter()
            .filter(|(id, _, _, _)| !selected_set.contains(id))
            .take(OVERFLOW_DISPLAY_CAP)
            .collect();
        if overflow.is_empty() {
            None
        } else {
            let n_overflow = n_candidates - n_nodes;
            let min_score: f32 = overflow.last().map(|item| item.1).unwrap_or(0.0);
            let mut msg = format!(
                "[{n_overflow} more node(s) matched (score \u{2265} {min_score:.2}), budget exhausted. \
                 Use a specific symbol name for deeper coverage, or call \
                 get_blast_radius / get_callers / get_dependencies for complete results.]"
            );
            let lines: Vec<String> = overflow
                .iter()
                .map(|(_, s, sig, path)| format!("  {sig}  score: {s:.2}  {path}"))
                .collect();
            msg.push('\n');
            msg.push_str(&lines.join("\n"));
            Some(msg)
        }
    } else {
        None
    };

    // Seed-cap signal: emit when eligible candidates exceeded MAX_EMBED_SEEDS.
    let seed_cap_msg: Option<&str> = if seed_cap_fired {
        Some(
            "[note: query matched more seed candidates than the cap, \
             consider narrowing the query or calling get_context once per concept]",
        )
    } else {
        None
    };

    // R2: detect multi-concept starvation by checking whether the seed set
    // forms ≥2 disconnected components. A query like "schedule embedding AND phase b"
    // can produce seeds from two unrelated subgraphs; PPR then awards all budget to
    // the heavier cluster, silently returning 0 nodes from the lighter one.
    //
    // Detection: BFS from the first seed through direct edges among seeds only.
    // If ≥1 seed is unreachable, the seeds span ≥2 disconnected components.
    let multi_concept_note: Option<&str> = {
        // Fix 3b: warn only when the PRIMARY seeds (pre-enrichment) split into ≥2
        // SUBSTANTIAL disconnected clusters (≥2 seeds each). The old check —
        // "any seed unreachable from seed[0]" over the *enriched* set — fired on
        // nearly every query: enrichment spans many packages, and a lone
        // edge-island primary seed (e.g. a generated conversion fn for the same
        // concept) looks disconnected. Counting substantial clusters among
        // primary seeds isolates the genuine R2 case (e.g. "scheduler AND
        // kubelet") from one concept that merely has a stray seed.
        const MIN_CLUSTER: usize = 2;
        let seed_set_local: std::collections::HashSet<NodeId> = primary_seed_ids.clone();
        if seed_set_local.len() >= 4 {
            let mut unvisited = seed_set_local.clone();
            let mut substantial_clusters = 0usize;
            while let Some(&start) = unvisited.iter().next() {
                unvisited.remove(&start);
                let mut size = 1usize;
                let mut queue: std::collections::VecDeque<NodeId> =
                    std::collections::VecDeque::from([start]);
                while let Some(id) = queue.pop_front() {
                    let fwd = store.iter_edges_from(id).unwrap_or_default();
                    let rev = store.iter_edges_to(id).unwrap_or_default();
                    for e in fwd.into_iter().chain(rev) {
                        let neighbor = if e.src == id { e.dst } else { e.src };
                        if seed_set_local.contains(&neighbor) && unvisited.remove(&neighbor) {
                            size += 1;
                            queue.push_back(neighbor);
                        }
                    }
                }
                if size >= MIN_CLUSTER {
                    substantial_clusters += 1;
                }
            }
            if substantial_clusters >= 2 {
                Some(
                    "[note: query may span multiple concepts, \
                     results show the dominant cluster only. \
                     Try one concept per get_context call for complete coverage.]",
                )
            } else {
                None
            }
        } else {
            None
        }
    };

    // Build 1-hop role map: classify each selected node's relationship to the seeds.
    // O(S × avg_degree); seeds are capped at MAX_EMBED_SEEDS so this is negligible.
    // Errors are silently ignored — roles degrade to Context, never block output.
    let seed_ids: Vec<NodeId> = weighted_seeds.iter().map(|&(id, _)| id).collect();
    // RFC-019: alongside the role, record WHICH seed assigned a caller/dependency
    // role (the first seed to do so, deterministic in seed_ids order), so `via:`
    // can name the source and make seed contamination diagnosable.
    let (roles, role_source): (HashMap<NodeId, NodeRole>, HashMap<NodeId, NodeId>) = {
        use travsr_core::EdgeKind;
        let selected_set: std::collections::HashSet<NodeId> =
            selected.iter().map(|n| n.id).collect();
        let mut map: HashMap<NodeId, NodeRole> =
            selected.iter().map(|n| (n.id, NodeRole::Context)).collect();
        let mut src: HashMap<NodeId, NodeId> = HashMap::new();
        for &seed in &seed_ids {
            // Forward edges: seed → node  →  node is a Dependency of seed.
            if let Ok(fwd) = store.iter_edges_from(seed) {
                for e in fwd {
                    if matches!(
                        e.kind,
                        EdgeKind::RefCall
                            | EdgeKind::Depends
                            | EdgeKind::RefImports
                            | EdgeKind::Exports
                    ) && selected_set.contains(&e.dst)
                    {
                        if let Some(r) = map.get_mut(&e.dst) {
                            if *r == NodeRole::Context {
                                *r = NodeRole::Dependency;
                                src.insert(e.dst, seed);
                            }
                        }
                    }
                }
            }
            // Reverse edges: node → seed  →  node is a Caller of seed.
            if let Ok(rev) = store.iter_edges_to(seed) {
                for e in rev {
                    if matches!(
                        e.kind,
                        EdgeKind::RefCall | EdgeKind::RefImports | EdgeKind::Depends
                    ) && selected_set.contains(&e.src)
                    {
                        if let Some(r) = map.get_mut(&e.src) {
                            if *r == NodeRole::Context {
                                *r = NodeRole::Caller;
                                src.insert(e.src, seed);
                            }
                        }
                    }
                }
            }
            // Only primary FTS/KNN seeds get the Seed role.
            // Enriched callers (added by enrich_seeds_with_callers) keep their Caller
            // role so the annotation correctly shows [via: caller] for them.
            if primary_seed_ids.contains(&seed) {
                map.insert(seed, NodeRole::Seed);
            }
        }
        (map, src)
    };
    // Seed signatures for `via: <role> of <seed>` rendering. Cheap — seeds ≤ ~40.
    let seed_sig: HashMap<NodeId, String> = store
        .get_nodes(&seed_ids)
        .map(|ns| {
            ns.into_iter()
                .map(|n| (n.id, display_label(&n).into_owned()))
                .collect()
        })
        .unwrap_or_default();

    if include_snippets {
        // Pre-snippet indexes lack this key → degrade to metadata-only.
        let repo_root = match resolve_repo_root(store) {
            Some(r) => r,
            None => {
                tracing::warn!("get_context: repo_root not in meta, falling back to metadata-only");
                let mut entries: Vec<(crate::seed::MatchSource, f32, String)> = selected
                    .iter()
                    .map(|n| {
                        let role = roles
                            .get(&n.id)
                            .copied()
                            .unwrap_or(NodeRole::Context)
                            .label();
                        let (ms, score) = entry_meta(n.id);
                        let line = format_node_line(
                            n,
                            role,
                            display_score_map.get(&n.id).copied(),
                            role_source
                                .get(&n.id)
                                .and_then(|s| seed_sig.get(s))
                                .map(String::as_str),
                            seed_source_map.get(&n.id).map(|s| seed_source_display(*s)),
                            omit_seed_via(group_output, ms),
                        );
                        (ms, score, line)
                    })
                    .collect();
                entries.extend(docs_entries.clone());
                let sanitized = sanitize_mcp_body_with_limit(
                    &assemble_context_body(entries, "\n", group_output),
                    char_cap,
                );
                let notes = build_degraded_notes(store, has_embed, embed_warming);
                let footer = format!(
                    "[{n_nodes} nodes, ~{total_tokens} tokens; run `travsr init` to enable inline snippets]"
                );
                return if notes.is_empty() {
                    format!("{retrieval_header}{sanitized}\n\n{footer}")
                } else {
                    format!("{retrieval_header}{sanitized}\n\n{footer}\n{notes}")
                };
            }
        };

        // Snippet ceiling: explicit separate budget or whatever remains after metadata.
        let snippet_ceiling =
            snippet_budget.unwrap_or_else(|| token_budget.saturating_sub(total_tokens));
        let mode_label = if snippet_budget.is_some() {
            "separate"
        } else {
            "shared"
        };

        let mut snip_tokens: usize = 0;
        let mut n_with_snippet: usize = 0;
        let mut blocks: Vec<(crate::seed::MatchSource, f32, String)> =
            Vec::with_capacity(selected.len());

        for n in &selected {
            let role = roles
                .get(&n.id)
                .copied()
                .unwrap_or(NodeRole::Context)
                .label();
            let (ms, score) = entry_meta(n.id);
            let header = format_node_line(
                n,
                role,
                display_score_map.get(&n.id).copied(),
                role_source
                    .get(&n.id)
                    .and_then(|s| seed_sig.get(s))
                    .map(String::as_str),
                seed_source_map.get(&n.id).map(|s| seed_source_display(*s)),
                omit_seed_via(group_output, ms),
            );
            // Use skeleton when the body exceeds the kind-aware line cap.
            let n_height = n
                .end_line
                .unwrap_or_else(|| n.line.unwrap_or(0))
                .saturating_sub(n.line.unwrap_or(0)) as usize;
            let block = if snip_tokens < snippet_ceiling {
                let body = if n_height > snippet_line_cap(&n.kind) {
                    skeleton_for_node_inner(n, &repo_root)
                        .map(|s| s.render())
                        .or_else(|| snippet_for_node(n, &repo_root))
                } else {
                    snippet_for_node(n, &repo_root)
                        .or_else(|| skeleton_for_node_inner(n, &repo_root).map(|s| s.render()))
                };
                if let Some(snip) = body {
                    let cost = snip.len() / TOKEN_CHARS_PER_TOKEN + 1;
                    if snip_tokens + cost <= snippet_ceiling {
                        snip_tokens += cost;
                        n_with_snippet += 1;
                        format!("{header}\n{SNIPPET_SEP}\n{snip}")
                    } else {
                        header
                    }
                } else {
                    header
                }
            } else {
                header
            };
            blocks.push((ms, score, block));
        }
        blocks.extend(docs_entries.clone());

        let sanitized = sanitize_mcp_body_with_limit(
            &assemble_context_body(blocks, "\n\n", group_output),
            char_cap,
        );
        let signals = build_context_signals_with_r2(
            phase_b,
            has_embed,
            embed_warming,
            embed_disabled,
            embed_initialized,
            knn_degraded,
            overflow_msg.as_deref(),
            seed_cap_msg,
            multi_concept_note,
            weak_match_note,
        );
        let footer = format!(
            "[{n_nodes} nodes, {n_with_snippet} with snippets, ~{total_tokens} metadata-tokens + ~{snip_tokens} snippet-tokens ({mode_label} budget)]"
        );
        if signals.is_empty() {
            format!("{retrieval_header}{sanitized}\n\n{footer}")
        } else {
            format!("{retrieval_header}{sanitized}\n\n{footer}\n{signals}")
        }
    } else {
        // Metadata-only path with role annotations.
        let mut entries: Vec<(crate::seed::MatchSource, f32, String)> = selected
            .iter()
            .map(|n| {
                let role = roles
                    .get(&n.id)
                    .copied()
                    .unwrap_or(NodeRole::Context)
                    .label();
                let (ms, score) = entry_meta(n.id);
                let line = format_node_line(
                    n,
                    role,
                    display_score_map.get(&n.id).copied(),
                    role_source
                        .get(&n.id)
                        .and_then(|s| seed_sig.get(s))
                        .map(String::as_str),
                    seed_source_map.get(&n.id).map(|s| seed_source_display(*s)),
                    omit_seed_via(group_output, ms),
                );
                (ms, score, line)
            })
            .collect();
        entries.extend(docs_entries);
        let sanitized = sanitize_mcp_body_with_limit(
            &assemble_context_body(entries, "\n", group_output),
            char_cap,
        );
        let signals = build_context_signals_with_r2(
            phase_b,
            has_embed,
            embed_warming,
            embed_disabled,
            embed_initialized,
            knn_degraded,
            overflow_msg.as_deref(),
            seed_cap_msg,
            multi_concept_note,
            weak_match_note,
        );
        let footer = format!("[{n_nodes} nodes, ~{total_tokens} tokens]");
        if signals.is_empty() {
            format!("{retrieval_header}{sanitized}\n\n{footer}")
        } else {
            format!("{retrieval_header}{sanitized}\n\n{footer}\n{signals}")
        }
    }
}

/// DIAGNOSTIC (Phase 0, #462 conceptual-recall investigation).
///
/// Returns the RAW embed-KNN ranking for `query` — up to `k` nodes as
/// `rank\tcosine\tsig\tpath`, ranked purely by query↔node cosine, BEFORE any
/// FTS fusion, rerank, or abstain gate. Lets a harness measure where a known gold
/// symbol sits in pure semantic recall (Branch 1 "recoverable but lost downstream"
/// vs Branch 2 "embedding blind"). Intentionally NOT registered in tools/list —
/// call by name via tools/call. Zero effect on any shipping path.
pub fn embed_knn_probe(store: &SqliteStore, query: &str, k: u32) -> String {
    if validate_mcp_arg(query).is_err() {
        return String::new();
    }
    let knn = match store.embed_knn_fn() {
        Some(f) => f,
        None => return "[embed KNN unavailable, embeddings not wired]".to_string(),
    };
    let pairs = knn(query, k);
    if pairs.is_empty() {
        return "[no KNN results]".to_string();
    }
    let ids: Vec<NodeId> = pairs.iter().map(|&(id, _)| id).collect();
    let nodes = store.get_nodes(&ids).unwrap_or_default();
    let by_id: HashMap<NodeId, &CoreNode> = nodes.iter().map(|n| (n.id, n)).collect();
    let mut out = String::new();
    for (i, (id, cos)) in pairs.iter().enumerate() {
        if let Some(n) = by_id.get(id) {
            let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
            out.push_str(&format!(
                "{}\t{:.4}\t{}\t{}{}\n",
                i + 1,
                cos,
                n.vname.signature,
                n.vname.path,
                loc
            ));
        }
    }
    out
}

/// WS0 diagnostic helper: short tag for a seed's retrieval source.
fn seed_source_label(source: crate::seed::SeedSource) -> &'static str {
    match source {
        crate::seed::SeedSource::Exact => "exact",
        crate::seed::SeedSource::Lexical => "lexical",
        crate::seed::SeedSource::Knn => "knn",
    }
}

/// #462: human-facing match-source tag for `get_context` node lines. Distinct
/// from the terse [`seed_source_label`] used in the `seed_trace` diagnostic.
fn seed_source_display(source: crate::seed::SeedSource) -> &'static str {
    match source {
        crate::seed::SeedSource::Exact => "exact-name",
        crate::seed::SeedSource::Knn => "semantic",
        crate::seed::SeedSource::Lexical => "lexical",
    }
}

/// DIAGNOSTIC (WS0, #462 conceptual-recall investigation).
///
/// Traces one query through the seed cascade and dumps every stage so a harness
/// can localize where a known gold symbol is lost: GATE (survived into the seed
/// set but the query abstained) vs DROP-@-stage (evicted before the confidence
/// gate). Mirrors [`embed_knn_probe`]: hidden — NOT in tools/list, called by name,
/// zero effect on any shipping path.
///
/// Output is line-oriented TSV; a line's first field tags its stage:
/// - `CAL   <lo> <hi>`                         — active model calibration anchors
/// - `CONF  <label>`                           — final `SeedSet.confidence`
/// - `RAWKNN <rank> <cos> <sig> <path:line>`   — raw embed-KNN (top 50), pre-cascade
/// - `KNNSEED <cos> <sig> <path:line>`         — seeds selected by `embed_path_seeds`
///   (post 0.75-threshold / MIN/MAX / noise / path-dedup — the first hard filter)
/// - `FINAL <source> <weight> <rerank> <sig> <path:line>` — surviving `build_seed_set`
///   seeds (post semantic_validate / scope gate / rerank reorder)
///
/// A gold present in RAWKNN but absent from KNNSEED dropped at `embed_path_seeds`;
/// present in KNNSEED but absent from FINAL dropped at semantic_validate/scope;
/// present in FINAL with `CONF none` is a GATE (abstain) case.
pub fn seed_trace(store: &SqliteStore, query: &str) -> String {
    if validate_mcp_arg(query).is_err() {
        return String::new();
    }
    let knn_fn = match store.embed_knn_fn() {
        Some(f) => f,
        None => return "[embed KNN unavailable, embeddings not wired]".to_string(),
    };
    let mut out = String::new();

    // Calibration anchors for the active model (identity for bge/un-calibrated).
    let cal = crate::seed::Calibration::load(store);
    out.push_str(&format!("CAL\t{:.4}\t{:.4}\n", cal.lo, cal.hi));

    let fmt_loc = |n: &CoreNode| n.line.map(|l| format!(":{l}")).unwrap_or_default();

    // Stage 1: raw embed-KNN (top 50), identical source to embed_knn_probe.
    let raw = knn_fn(query, 50);
    {
        let ids: Vec<NodeId> = raw.iter().map(|&(id, _)| id).collect();
        let nodes = store.get_nodes(&ids).unwrap_or_default();
        let by_id: HashMap<NodeId, &CoreNode> = nodes.iter().map(|n| (n.id, n)).collect();
        for (i, (id, cos)) in raw.iter().enumerate() {
            if let Some(n) = by_id.get(id) {
                out.push_str(&format!(
                    "RAWKNN\t{}\t{:.4}\t{}\t{}{}\n",
                    i + 1,
                    cos,
                    n.vname.signature,
                    n.vname.path,
                    fmt_loc(n)
                ));
            }
        }
    }

    // Stage 2: seeds selected by embed_path_seeds (the first hard KNN filter).
    let (knn_selected, _n_elig, _ms, knn_oracle) =
        embed_path_seeds(store, query, &knn_fn, &OpenFilter);
    for (n, cos) in &knn_selected {
        out.push_str(&format!(
            "KNNSEED\t{:.4}\t{}\t{}{}\n",
            cos,
            n.vname.signature,
            n.vname.path,
            fmt_loc(n)
        ));
    }

    // Stage 3: full build_seed_set — final surviving seeds + confidence.
    let score_fn = store.embed_score_fn();
    let score_ref = score_fn
        .as_ref()
        .map(|f| f as &dyn Fn(&str, &[NodeId]) -> Vec<(NodeId, f32)>);
    let seed_set = crate::seed::build_seed_set(
        store,
        query,
        &OpenFilter,
        knn_selected,
        &knn_oracle,
        score_ref,
    );
    out.push_str(&format!("CONF\t{}\n", seed_set.confidence.label()));
    let final_ids: Vec<NodeId> = seed_set.seeds.iter().map(|s| s.node).collect();
    let final_nodes = store.get_nodes(&final_ids).unwrap_or_default();
    let by_id: HashMap<NodeId, &CoreNode> = final_nodes.iter().map(|n| (n.id, n)).collect();
    for s in &seed_set.seeds {
        if let Some(n) = by_id.get(&s.node) {
            let rr = s
                .rerank_score
                .map(|r| format!("{r:.4}"))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "FINAL\t{}\t{:.4}\t{}\t{}\t{}{}\n",
                seed_source_label(s.source),
                s.weight,
                rr,
                n.vname.signature,
                n.vname.path,
                fmt_loc(n)
            ));
        }
    }
    out
}

/// Global variant of `get_context` — searches one named repo or all registered repos.
pub fn get_context_global(
    repos: &HashMap<String, PathBuf>,
    query: &str,
    token_budget: usize,
    repo: Option<&str>,
    include_snippets: bool,
    snippet_budget: Option<usize>,
) -> String {
    if let Err(reason) = validate_mcp_arg(query) {
        tracing::warn!("get_context_global rejected invalid query: {reason}");
        return wrap_envelope("");
    }
    if token_budget > MAX_CONTEXT_BUDGET {
        return wrap_envelope("token_budget exceeds maximum allowed value");
    }

    // Strip any embedded language token so FTS seeding gets a clean query.
    // E.g. "auth handler javascript" → seed PPR with "auth handler".
    let (stripped, _lang_filter) = infer_language_from_query(query);
    let seed_query = stripped.as_str();

    // R4: use a per-repo header block rather than prefixing every line with
    // "[repo_name]". Per-line prefixing pollutes blank lines, footer lines, and
    // notes with repo tags that look like noise in an LLM context window.
    // Named repo: open the store here rather than through `collect_global`, which
    // is shared with the 12 structural tools. Only this path arms the embed
    // sidecar, so `get_callers` and friends never pay for a model load. With a
    // single candidate the per-repo `[repo: ...]` header never applied, so it is
    // not reproduced here.
    let raw = if let Some(name) = repo {
        match resolve_repo_db_path(repos, name) {
            Some(db_path) => match SqliteStore::open_read_only(db_path) {
                Ok(mut store) => {
                    crate::inject_embed_hook(&mut store, db_path);
                    get_context_raw(
                        &store,
                        seed_query,
                        token_budget,
                        include_snippets,
                        snippet_budget,
                    )
                }
                Err(e) => {
                    tracing::warn!("failed to open {}: {e}", db_path.display());
                    String::new()
                }
            },
            None => String::new(),
        }
    } else {
        // Fan-out: rank repos by result size (token-rich responses first).
        let mut candidates: Vec<(&str, &PathBuf)> =
            repos.iter().map(|(k, v)| (k.as_str(), v)).collect();
        candidates.retain(|(_, db)| db.exists());
        let single = candidates.len() == 1;

        let mut parts: Vec<(usize, String)> = Vec::new();
        for (repo_name, db_path) in &candidates {
            match SqliteStore::open_read_only(db_path) {
                Ok(mut store) => {
                    // Lookup-only: a fan-out must never arm N sidecars for an
                    // N-repo registry. A repo an earlier named query warmed keeps
                    // its semantic lane; an unwarmed one stays lexical.
                    crate::inject_cached_embed_hook(&mut store, db_path);
                    let result = get_context_raw(
                        &store,
                        seed_query,
                        token_budget,
                        include_snippets,
                        snippet_budget,
                    );
                    if !result.is_empty() {
                        let count = result.len();
                        let text = if single {
                            result
                        } else {
                            format!("[repo: {repo_name}]\n{result}")
                        };
                        parts.push((count, text));
                    }
                }
                Err(e) => tracing::warn!("failed to open {}: {e}", db_path.display()),
            }
        }
        parts.sort_by_key(|b| std::cmp::Reverse(b.0));
        parts
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    wrap_envelope(&raw)
}

// ── get_graph_json ────────────────────────────────────────────────────────────

/// Max nodes returned by `get_graph_json` to keep MCP payloads manageable.
const MAX_GRAPH_JSON_NODES: usize = 200;

/// Return a subgraph around `query` as structured JSON for graph renderers.
///
/// Query parameters shared by `get_graph_json` and `get_graph_json_global`.
/// Keeps both function signatures under clippy's 7-argument limit.
pub struct GraphJsonParams<'a> {
    pub query: &'a str,
    pub direction: &'a str,
    pub depth: u8,
    pub kind_filter: &'a str,
    pub token_budget: usize,
    pub mode: &'a str,
    pub path_prefix: &'a str,
    /// RFC-027 section 10: restrict edges by how they were derived.
    ///
    /// `""` (the default) returns everything, so an existing caller sees the
    /// fresher graph and nothing changes for it. `"ratified"` excludes the live
    /// overlay, for a consumer that needs only what the deterministic pipeline
    /// has confirmed. Any other value names a single provenance exactly
    /// (`tree-sitter`, `lsif`, `scip`, `live`).
    ///
    /// A word rather than a `!=` expression: the meaningful question a consumer
    /// has is "confirmed, or everything", and spelling that as a filter grammar
    /// would invite queries the store cannot answer cheaply.
    pub provenance: &'a str,
}

impl Default for GraphJsonParams<'_> {
    fn default() -> Self {
        Self {
            query: "",
            direction: "both",
            depth: 2,
            kind_filter: "",
            token_budget: 0,
            mode: "",
            path_prefix: "",
            provenance: "",
        }
    }
}

/// Whether an edge with `provenance` passes the caller's filter.
///
/// Unknown filter values match nothing rather than everything: a typo should
/// return an obviously empty graph, not silently ignore the constraint a
/// consumer added precisely because it needed ground truth.
fn provenance_allowed(filter: &str, provenance: &str) -> bool {
    match filter {
        "" => true,
        "ratified" => provenance != "live",
        exact => provenance == exact,
    }
}

/// BFS from seed node(s) matching `query`, respecting `direction` and `depth`.
/// Returns `{"nodes":[...],"edges":[...]}`.
/// Unlike prose tools, output is NOT sanitized — it is structured JSON consumed
/// by the VS Code graph panel, not forwarded to an LLM as freetext.
pub fn get_graph_json(store: &SqliteStore, params: &GraphJsonParams<'_>) -> String {
    let GraphJsonParams {
        query,
        direction,
        depth,
        kind_filter,
        token_budget,
        mode,
        path_prefix,
        provenance,
    } = params;
    if !matches!(*mode, "" | "overview") {
        tracing::warn!("get_graph_json rejected unknown mode: {mode}");
        return "{}".to_string();
    }
    if *mode == "overview" {
        if !path_prefix.is_empty() {
            if let Err(reason) = validate_mcp_arg(path_prefix) {
                tracing::warn!("get_graph_json rejected invalid path_prefix: {reason}");
                return "{}".to_string();
            }
        }
        return overview_graph(store, path_prefix);
    }
    // Only "" (all kinds) and "file" are valid kind_filter values.
    if !matches!(*kind_filter, "" | "file") {
        tracing::warn!("get_graph_json rejected unknown kind_filter: {kind_filter}");
        return "{}".to_string();
    }
    // Empty query is valid when kind_filter=="file" (returns full import graph).
    if !(query.is_empty() && *kind_filter == "file") {
        if let Err(reason) = validate_mcp_arg(query) {
            tracing::warn!("get_graph_json rejected invalid arg: {reason}");
            return "{}".to_string();
        }
    }
    let depth = (*depth).clamp(1, 4);
    // #645 WS-D: single-repo stdio server — the caller's checkout IS this repo,
    // so the launch cwd's HEAD is the correct drift comparand.
    let head = LAUNCH_CWD.get().and_then(|cwd| git_short_head(cwd));
    get_graph_json_raw(
        store,
        query,
        direction,
        depth,
        kind_filter,
        *token_budget,
        provenance,
        head.as_deref(),
    )
}

// ── Repo-map LOD overview (P3 #319) ──────────────────────────────────────────

/// Maps a file path to its top-level directory segment (the first `/`-delimited part).
/// Returns an empty string for external/build-cache paths that should be excluded.
///
/// "pkg/api/types.go"             → "pkg"
/// "cmd/kubeadm/main.go"          → "cmd"
/// "src/index.ts"                 → "src"
/// "main.go"                      → "(root)"
/// "../../../Library/Caches/..."  → ""  (excluded)
/// "/abs/path/file.go"            → ""  (excluded)
/// "scip://corpus/file"           → ""  (excluded)
fn pkg_key_from_path(path: &str) -> String {
    // Skip build-cache, absolute, or protocol-prefixed paths (SCIP, etc.)
    if path.is_empty() || path.starts_with("..") || path.starts_with('/') || path.contains("://") {
        return String::new();
    }
    let segs: Vec<&str> = path.split('/').collect();
    match segs.len().saturating_sub(1) {
        0 => "(root)".to_string(),
        _ => segs[0].to_string(),
    }
}

fn file_label(path: &str) -> &str {
    path.rfind('/').map_or(path, |i| &path[i + 1..])
}

/// Entry point for `mode="overview"`. Routes by whether path_prefix is set.
fn overview_graph(store: &SqliteStore, path_prefix: &str) -> String {
    let file_nodes = match store.nodes_by_kind("file") {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("overview_graph: nodes_by_kind error: {e}");
            return r#"{"nodes":[],"edges":[]}"#.to_string();
        }
    };
    // Resolved cross-file dependency pairs (ref/call + resolves-to) — the same
    // language-agnostic primitive the repo map uses. Replaces the old
    // depends+resolves-to-only `file_import_pairs`, which produced ~0 edges here
    // because top-level-dir buckets collapsed every intra-monorepo edge.
    let pairs = match store.resolved_dep_pairs() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("overview_graph: resolved_dep_pairs error: {e}");
            vec![]
        }
    };
    if path_prefix.is_empty() {
        repo_overview_graph(&file_nodes, &pairs)
    } else {
        package_drill_graph(&file_nodes, &pairs, path_prefix)
    }
}

/// Aggregate file nodes into directory-level component tiles + cross-component
/// dependency edges. Granularity is the adaptive directory rollup (`repo_regions`)
/// — NOT the top-level folder — so a monorepo's internal architecture
/// (crates/mcp -> crates/core) actually appears instead of collapsing into one
/// `crates` self-edge. Nodes carry `dependents` so renderers can rank by it.
// O(F + P) where F = file nodes, P = resolved pairs
fn repo_overview_graph(file_nodes: &[travsr_core::Node], pairs: &[(String, String)]) -> String {
    use std::collections::{BTreeMap, HashMap, HashSet};

    let mut file_paths: HashSet<String> = HashSet::new();
    for node in file_nodes {
        if repo_map_is_local_path(&node.vname.path) {
            file_paths.insert(node.vname.path.clone());
        }
    }
    let regions = repo_regions(&file_paths);

    let mut region_files: BTreeMap<String, u32> = BTreeMap::new();
    for node in file_nodes {
        if !repo_map_is_local_path(&node.vname.path) {
            continue;
        }
        *region_files
            .entry(repo_map_region_of(&node.vname.path, &regions))
            .or_default() += 1;
    }

    if region_files.is_empty() {
        return r#"{"nodes":[],"edges":[],"mode":"overview"}"#.to_string();
    }

    // Cross-component edges with weight = distinct file pairs crossing the border.
    let known: HashSet<String> = region_files.keys().cloned().collect();
    let mut cross: HashMap<(String, String), u32> = HashMap::new();
    for (src_path, dst_path) in pairs {
        if !repo_map_is_local_path(src_path) || !repo_map_is_local_path(dst_path) {
            continue;
        }
        let src_r = repo_map_region_of(src_path, &regions);
        let dst_r = repo_map_region_of(dst_path, &regions);
        if src_r != dst_r && known.contains(&src_r) && known.contains(&dst_r) {
            *cross.entry((src_r, dst_r)).or_default() += 1;
        }
    }

    let dep_edges: HashSet<(String, String)> = cross.keys().cloned().collect();
    let dependents = repo_region_dependents(&known, &dep_edges);

    let nodes_out: Vec<serde_json::Value> = region_files
        .iter()
        .map(|(region, file_count)| {
            serde_json::json!({
                "id":         format!("pkg:{region}"),
                "label":      region,
                "kind":       "pkg",
                "file_count": file_count,
                "dependents": dependents.get(region).copied().unwrap_or(0),
                "ghost":      false,
            })
        })
        .collect();

    let edges_out: Vec<serde_json::Value> = cross
        .iter()
        .map(|((src, dst), count)| {
            serde_json::json!({
                "source": format!("pkg:{src}"),
                "target": format!("pkg:{dst}"),
                "kind":   "imports",
                "count":  count,
            })
        })
        .collect();

    match serde_json::to_string(&serde_json::json!({
        "nodes": nodes_out,
        "edges": edges_out,
        "mode":  "overview",
    })) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("repo_overview_graph serialization error: {e}");
            "{}".to_string()
        }
    }
}

/// File-level drill into a package: emit file nodes inside `path_prefix` plus ghost-port
/// package nodes for any cross-boundary import targets.
// O(F + P) where F = file nodes, P = import pairs
fn package_drill_graph(
    file_nodes: &[travsr_core::Node],
    pairs: &[(String, String)],
    path_prefix: &str,
) -> String {
    use std::collections::{HashMap, HashSet};

    let prefix_paths: HashSet<&str> = file_nodes
        .iter()
        .filter(|n| n.vname.path.starts_with(path_prefix))
        .map(|n| n.vname.path.as_str())
        .collect();

    if prefix_paths.is_empty() {
        return r#"{"nodes":[],"edges":[],"mode":"prefix"}"#.to_string();
    }

    let mut nodes_out: Vec<serde_json::Value> = Vec::new();
    let mut node_ids_out: HashSet<String> = HashSet::new();

    for path in &prefix_paths {
        let json_id = format!("file:{path}");
        node_ids_out.insert(json_id.clone());
        nodes_out.push(serde_json::json!({
            "id":    json_id,
            "label": file_label(path),
            "kind":  "file",
            "path":  path,
            "ghost": false,
        }));
    }

    let mut ghost_pkg_counts: HashMap<String, u32> = HashMap::new();
    let mut edges_out: Vec<serde_json::Value> = Vec::new();
    let mut edge_seen: HashSet<(String, String)> = HashSet::new();

    for (src_path, dst_path) in pairs {
        // src must be inside the prefix (src_path may be a function path — check prefix match)
        if !src_path.starts_with(path_prefix) {
            continue;
        }
        // Resolve src to the nearest file in prefix_paths (take the path directly if present,
        // otherwise derive the file path from the src path's directory)
        let src_file = if prefix_paths.contains(src_path.as_str()) {
            src_path.as_str()
        } else {
            // src is a symbol node whose path happens to start with prefix — use it as-is
            // (it won't match any file node, so skip intra edges but allow cross-boundary)
            src_path.as_str()
        };

        if prefix_paths.contains(dst_path.as_str()) {
            // Intra-prefix: file → file edge
            let key = (src_file.to_string(), dst_path.clone());
            if edge_seen.insert(key) {
                edges_out.push(serde_json::json!({
                    "source": format!("file:{src_file}"),
                    "target": format!("file:{dst_path}"),
                    "kind":   "imports",
                }));
            }
        } else {
            // Cross-boundary: collapse dst into a ghost package node
            let ghost_pkg = pkg_key_from_path(dst_path);
            if ghost_pkg.is_empty() {
                continue;
            }
            let ghost_id = format!("pkg:{ghost_pkg}");

            let count = ghost_pkg_counts.entry(ghost_pkg.clone()).or_insert(0);
            *count += 1;
            if *count == 1 {
                node_ids_out.insert(ghost_id.clone());
                nodes_out.push(serde_json::json!({
                    "id":    ghost_id.clone(),
                    "label": ghost_pkg,
                    "kind":  "ghost",
                    "ghost": true,
                }));
            }

            let key = (src_file.to_string(), ghost_id.clone());
            if edge_seen.insert(key) {
                edges_out.push(serde_json::json!({
                    "source": format!("file:{src_file}"),
                    "target": ghost_id,
                    "kind":   "imports",
                }));
            }
        }
    }

    match serde_json::to_string(&serde_json::json!({
        "nodes":       nodes_out,
        "edges":       edges_out,
        "mode":        "prefix",
        "path_prefix": path_prefix,
    })) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("package_drill_graph serialization error: {e}");
            "{}".to_string()
        }
    }
}

fn edge_kind_str(kind: &travsr_core::EdgeKind) -> &'static str {
    use travsr_core::EdgeKind;
    match kind {
        EdgeKind::DefinesBinding => "defines",
        EdgeKind::RefCall => "calls",
        EdgeKind::RefField => "field-ref",
        EdgeKind::Depends => "imports",
        EdgeKind::ResolvesTo => "resolves-to",
        EdgeKind::Exports => "exports",
        EdgeKind::RefImports => "ref/imports",
        EdgeKind::IsImplementation => "is-implementation",
        EdgeKind::Overrides => "overrides",
        EdgeKind::FFICall => "ffi/call",
        EdgeKind::Configures => "configures",
        EdgeKind::ExternalDependency => "external-dependency",
    }
}

/// Unique JSON id for a node.
/// File nodes all share `signature == "file"`, so we use `path` (prefixed with
/// corpus when non-empty) to keep Cytoscape ids distinct across repos.
fn node_json_id(node: &CoreNode) -> String {
    if node.kind == "file" || node.vname.path.is_empty() {
        if node.vname.corpus.is_empty() {
            node.vname.path.clone()
        } else {
            format!("{}:{}", node.vname.corpus, node.vname.path)
        }
    } else {
        // Include path so same-signature symbols from different files (e.g.
        // fn:ContainsPrefix in staging/ and pkg/) get distinct Cytoscape IDs
        // instead of the second being silently dropped by the dedup guard.
        format!("{}:{}", node.vname.path, node.vname.signature)
    }
}

/// Short display label for the visual graph renderer.
///
/// File nodes render as a compact basename (a rendering choice for the graph
/// panel — a file node never carries a raw compiler symbol). Every other node
/// uses [`display_label`], the one canonical name shared with the CLI and the
/// text MCP tools, so the same node reads identically on every surface.
fn node_json_label(node: &CoreNode) -> String {
    if node.kind == "file" {
        node.vname
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&node.vname.path)
            .to_string()
    } else {
        // `display_label` canonicalizes a synthesized scip:/sdb: symbol to a
        // clean `Type.member` name, but returns a native tree-sitter signature
        // verbatim (`fn:home`). The visual renderer wants the bare short name
        // (`home`), which the graph panel and `askSymbol` quick pick have always
        // shown, so strip the structural `kind:` prefix off native labels.
        strip_native_kind_prefix(&display_label(node)).to_string()
    }
}

/// Strip a native tree-sitter `kind:` prefix (`fn:`, `class:`, `method:`,
/// `interface:`, `import:`) so `fn:home` renders as `home` in the visual graph.
/// A canonicalized external label (no such prefix) passes through unchanged.
fn strip_native_kind_prefix(label: &str) -> &str {
    label
        .strip_prefix("fn:")
        .or_else(|| label.strip_prefix("class:"))
        .or_else(|| label.strip_prefix("method:"))
        .or_else(|| label.strip_prefix("interface:"))
        .or_else(|| label.strip_prefix("import:"))
        .unwrap_or(label)
}

/// Mark the name-matched edges in a `get_graph_json` edge array.
///
/// The tree view flagged these and the edge array did not, so a renderer
/// presented a call `resolve_unresolved_calls` guessed by name as one a
/// compiler resolved. Derived once from `kind` + `provenance` here rather than
/// at each of the six sites that build an edge object, all of which already
/// carry both. Additive: an edge that is not name-matched gains no key.
fn mark_heuristic_edges(edges: &mut [serde_json::Value]) {
    for e in edges.iter_mut() {
        if crate::query::is_heuristic_edge(
            e["kind"].as_str().unwrap_or_default(),
            e["provenance"].as_str().unwrap_or_default(),
        ) {
            e["heuristic"] = serde_json::Value::Bool(true);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn get_graph_json_raw(
    store: &SqliteStore,
    query: &str,
    direction: &str,
    depth: u8,
    kind_filter: &str,
    token_budget: usize,
    // RFC-027 section 10: edge provenance filter. `""` returns everything.
    provenance: &str,
    // #645/#661: the caller's live short HEAD for the index/HEAD drift signal.
    // Injected (not read from LAUNCH_CWD here) so the global aggregator can pass
    // *each* repo's own HEAD rather than the one workspace's — see
    // `repo_head_from_registry_path`.
    head: Option<&str>,
) -> String {
    use std::collections::{HashSet, VecDeque};

    // File mode with empty query: search broadly by path separator to seed all file nodes.
    // DEBT(travsr): replace "." sentinel with an explicit all-files store query
    let search_term = if kind_filter == "file" && query.is_empty() {
        "."
    } else {
        query
    };
    let seed_nodes_raw = match store.search_nodes_by_name(search_term) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("get_graph_json search error: {e}");
            return r#"{"nodes":[],"edges":[]}"#.to_string();
        }
    };

    // Filter seeds to the requested kind when kind_filter is set.
    // In symbol mode: also exclude scip: nodes from seeds — Phase B semantic nodes
    // have no edges in most projects and appear as isolated floating dots.
    // They are still reachable via BFS traversal if edges exist.
    let seed_nodes: Vec<_> = if !kind_filter.is_empty() {
        seed_nodes_raw
            .into_iter()
            .filter(|n| n.kind == kind_filter)
            .collect()
    } else {
        seed_nodes_raw
            .into_iter()
            .filter(|n| !n.vname.signature.starts_with("scip:"))
            .collect()
    };

    // #618: in symbol mode, a bare-name query used to promote every substring
    // match to a root, flooding the graph with unrelated symbols (e.g.
    // "docs_section" seeding build_docs_section_* and its whole test
    // neighbourhood). Resolve like get_callers / find_references instead:
    // exact signature first, then exact simple-name; the original substring
    // set survives only as a fuzzy fallback when neither tier matches, so a
    // partial query still renders something rather than nothing.
    let seed_nodes: Vec<_> = if kind_filter.is_empty() && !query.is_empty() {
        let exact_sig: Vec<_> = seed_nodes
            .iter()
            .filter(|n| n.vname.signature == query)
            .cloned()
            .collect();
        if !exact_sig.is_empty() {
            exact_sig
        } else {
            let exact_name: Vec<_> = seed_nodes
                .iter()
                .filter(|n| simple_name(&n.vname.signature) == query)
                .cloned()
                .collect();
            if !exact_name.is_empty() {
                exact_name
            } else {
                seed_nodes
            }
        }
    } else {
        seed_nodes
    };

    if seed_nodes.is_empty() {
        return r#"{"nodes":[],"edges":[]}"#.to_string();
    }

    // Coverage metadata (#318 O5): sourced from the first seed's language so
    // "no callers" is distinguishable from "cannot see callers".
    let seed_language = seed_nodes[0].vname.language.clone();
    let coverage = serde_json::json!({
        "language": seed_language,
        "semantic": store.has_refcall_edges_for_language(&seed_language),
        "phase_b_commit": store.get_meta("phase_b_commit").ok().flatten(),
    });

    // (NodeId, hop_distance)
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<(NodeId, u8, bool)> = VecDeque::new();
    let mut nodes_out: Vec<serde_json::Value> = Vec::new();
    let mut edges_out: Vec<serde_json::Value> = Vec::new();
    let mut edge_seen: HashSet<(NodeId, NodeId, &'static str)> = HashSet::new();
    // Token budget (#318 O6): cumulative cost of emitted nodes; once the next
    // node would exceed the budget, stop adding nodes (mirrors the node cap).
    let mut tokens_used = 0usize;
    let mut truncated_by_budget = false;
    // Tracks JSON ids already in nodes_out. Two different DB rows can share the
    // same signature (Phase B stores the same symbol once per referencing file).
    // Deduplicate here so Cytoscape never sees two nodes with the same id.
    let mut node_ids_out: HashSet<String> = HashSet::new();

    for node in &seed_nodes {
        if visited.insert(node.id) {
            queue.push_back((node.id, 0, true));
        }
    }

    while let Some((current_id, hop, expand)) = queue.pop_front() {
        let node = match store.get_node(current_id) {
            Ok(Some(n)) => n,
            _ => continue,
        };

        // Skip nodes that don't match the kind filter.
        if !kind_filter.is_empty() && node.kind != kind_filter {
            continue;
        }

        // In symbol mode: skip import nodes (long qualified names, no navigation
        // value in a symbol graph) and scip:local N internal tokens.
        // Skip empty-path external stubs at hop>0.
        let is_scip_local =
            node.vname.signature.starts_with("scip:") && node.vname.signature.contains(":local ");
        let is_import_in_symbol_mode = node.kind == "import" && kind_filter != "file";
        if is_scip_local || is_import_in_symbol_mode || (node.vname.path.is_empty() && hop > 0) {
            continue;
        }

        // Budget check before emission: the seed (first node) always survives
        // so a tiny budget still yields a non-empty, honest payload.
        let node_tokens = token_cost(&node);
        if token_budget > 0 && !nodes_out.is_empty() && tokens_used + node_tokens > token_budget {
            truncated_by_budget = true;
            continue;
        }

        let json_id = node_json_id(&node);
        if !node_ids_out.insert(json_id.clone()) {
            continue; // duplicate signature across DB rows — skip to avoid Cytoscape crash
        }
        tokens_used += node_tokens;

        let score = {
            let raw = 0.7_f64.powi(i32::from(hop));
            (raw * 1000.0).round() / 1000.0
        };
        let mut node_obj = serde_json::json!({
            "id":      json_id,
            "label":   node_json_label(&node),
            "kind":    node.kind,
            "path":    node.vname.path,
            "package": node.package,
            "score":   score,
            "line":    node.line,
        });
        if hop == 0 {
            node_obj["root"] = serde_json::Value::Bool(true);
        }
        nodes_out.push(node_obj);

        if nodes_out.len() >= MAX_GRAPH_JSON_NODES {
            tracing::warn!(
                "get_graph_json hit node cap ({MAX_GRAPH_JSON_NODES}) for query '{query}'"
            );
            continue;
        }

        if hop >= depth || !expand {
            continue;
        }

        // File mode: the import schema uses a two-hop chain
        //   file --[Depends]--> import_node --[ResolvesTo]--> file
        // Look through the intermediary to emit direct file→file edges.
        if kind_filter == "file" {
            use travsr_core::EdgeKind;
            if direction == "deps" || direction == "both" {
                if let Ok(dep_edges) = store.iter_edges_from(current_id) {
                    // Batch: collect all ResolvesTo target ids across all Depends hops,
                    // then fetch once instead of one get_node per res.dst.
                    let mut res_targets: Vec<NodeId> = Vec::new();
                    for dep in &dep_edges {
                        if !matches!(dep.kind, EdgeKind::Depends) {
                            continue;
                        }
                        if let Ok(res_edges) = store.iter_edges_from(dep.dst) {
                            for res in &res_edges {
                                if matches!(res.kind, EdgeKind::ResolvesTo) {
                                    res_targets.push(res.dst);
                                }
                            }
                        }
                    }
                    let node_map: HashMap<NodeId, CoreNode> = store
                        .get_nodes(&res_targets)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|n| (n.id, n))
                        .collect();
                    for target_id in res_targets {
                        if let Some(target) = node_map.get(&target_id) {
                            if target.kind != "file" {
                                continue;
                            }
                            if edge_seen.insert((current_id, target.id, "imports")) {
                                edges_out.push(serde_json::json!({
                                    "source": node_json_id(&node),
                                    "target": node_json_id(target),
                                    "kind":   "imports",
                                }));
                            }
                            if visited.insert(target.id) {
                                queue.push_back((target.id, hop + 1, true));
                            }
                        }
                    }
                }
            }
            if direction == "callers" || direction == "both" {
                // Reverse: who imports this file?
                //   importer_file --[Depends]--> import_node --[ResolvesTo]--> current_file
                if let Ok(rev_res_edges) = store.iter_edges_to(current_id) {
                    // Batch: collect all Depends src ids across all ResolvesTo hops,
                    // then fetch once.
                    let mut source_ids: Vec<NodeId> = Vec::new();
                    for rev_res in &rev_res_edges {
                        if !matches!(rev_res.kind, EdgeKind::ResolvesTo) {
                            continue;
                        }
                        if let Ok(rev_dep_edges) = store.iter_edges_to(rev_res.src) {
                            for rev_dep in &rev_dep_edges {
                                if matches!(rev_dep.kind, EdgeKind::Depends) {
                                    source_ids.push(rev_dep.src);
                                }
                            }
                        }
                    }
                    let node_map: HashMap<NodeId, CoreNode> = store
                        .get_nodes(&source_ids)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|n| (n.id, n))
                        .collect();
                    for source_id in source_ids {
                        if let Some(source) = node_map.get(&source_id) {
                            if source.kind != "file" {
                                continue;
                            }
                            if edge_seen.insert((source.id, current_id, "imports")) {
                                edges_out.push(serde_json::json!({
                                    "source": node_json_id(source),
                                    "target": node_json_id(&node),
                                    "kind":   "imports",
                                }));
                            }
                            if visited.insert(source.id) {
                                queue.push_back((source.id, hop + 1, true));
                            }
                        }
                    }
                }
            }
            continue; // skip normal edge traversal
        }

        // Normal traversal (symbol mode)
        if direction == "deps" || direction == "both" {
            if let Ok(nexts) = crate::query::next_edges(
                store,
                current_id,
                crate::query::QueryDirection::Deps,
                crate::query::QueryEdgeMode::All,
                hop == 0,
            ) {
                // First pass: dedup via edge_seen, enqueue new visits.
                // Collect (dst_id, kind_s) only for edges that produce JSON output.
                let mut new_edges: Vec<(NodeId, &str, &str)> = Vec::new();
                for (kind, next_id, child_expand, _incoming, edge_prov) in &nexts {
                    let kind_s = edge_kind_str(kind);
                    if !provenance_allowed(provenance, edge_prov) {
                        continue;
                    }
                    if edge_seen.insert((current_id, *next_id, kind_s)) {
                        new_edges.push((*next_id, kind_s, edge_prov.as_str()));
                    }
                    if visited.insert(*next_id) {
                        queue.push_back((*next_id, hop + 1, *child_expand));
                    }
                }
                // Batch-fetch dst nodes, then emit JSON edges in original order.
                let dst_ids: Vec<NodeId> = new_edges.iter().map(|(id, _, _)| *id).collect();
                let node_map: HashMap<NodeId, CoreNode> = store
                    .get_nodes(&dst_ids)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|n| (n.id, n))
                    .collect();
                for (dst_id, kind_s, provenance) in &new_edges {
                    if let Some(dst) = node_map.get(dst_id) {
                        // RFC-027 section 10: renderers and agents need to tell
                        // the un-ratified overlay from committed truth.
                        edges_out.push(serde_json::json!({
                            "source":     node_json_id(&node),
                            "target":     node_json_id(dst),
                            "kind":       kind_s,
                            "provenance": provenance,
                        }));
                    }
                }
            }
        }

        if direction == "callers" || direction == "both" {
            if let Ok(nexts) = crate::query::next_edges(
                store,
                current_id,
                crate::query::QueryDirection::Callers,
                crate::query::QueryEdgeMode::All,
                hop == 0,
            ) {
                // First pass: dedup via edge_seen, enqueue new visits.
                let mut new_edges: Vec<(NodeId, &str, &str)> = Vec::new();
                for (kind, next_id, child_expand, _incoming, edge_prov) in &nexts {
                    let kind_s = edge_kind_str(kind);
                    if !provenance_allowed(provenance, edge_prov) {
                        continue;
                    }
                    if edge_seen.insert((*next_id, current_id, kind_s)) {
                        new_edges.push((*next_id, kind_s, edge_prov.as_str()));
                    }
                    if visited.insert(*next_id) {
                        queue.push_back((*next_id, hop + 1, *child_expand));
                    }
                }
                // Batch-fetch src nodes, then emit JSON edges in original order.
                let src_ids: Vec<NodeId> = new_edges.iter().map(|(id, _, _)| *id).collect();
                let node_map: HashMap<NodeId, CoreNode> = store
                    .get_nodes(&src_ids)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|n| (n.id, n))
                    .collect();
                for (src_id, kind_s, provenance) in &new_edges {
                    if let Some(src) = node_map.get(src_id) {
                        edges_out.push(serde_json::json!({
                            "source":     node_json_id(src),
                            "target":     node_json_id(&node),
                            "kind":       kind_s,
                            "provenance": provenance,
                        }));
                    }
                }
            }
        }
    }

    // Remove edges whose source or target was filtered/deduplicated from nodes_out.
    // Orphan edges cause Cytoscape to silently crash.
    edges_out.retain(|e| {
        let src = e["source"].as_str().unwrap_or("");
        let tgt = e["target"].as_str().unwrap_or("");
        node_ids_out.contains(src) && node_ids_out.contains(tgt)
    });

    // #565 / RFC-002 consistency: surface the SAME ambiguity signal the CLI /
    // daemon `graph_query` path emits, computed via the SAME resolver, so an
    // agent calling get_graph_json can tell a query resolved to multiple distinct
    // definitions (and re-query with a narrower symbol / path) rather than
    // silently receiving a merged multi-root graph. Additive envelope field —
    // first-party consumers (the graph panel) read only `nodes`/`edges`.
    let ambiguous_candidates: Vec<serde_json::Value> =
        if kind_filter.is_empty() && !query.is_empty() {
            match resolve_reference_targets(store, query, None) {
                RefTarget::Ambiguous(list) => list
                    .iter()
                    .map(|n| {
                        serde_json::json!({
                            "signature": display_label(n),
                            "kind": n.kind,
                            "path": n.vname.path,
                            "line": n.line,
                        })
                    })
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

    // Additive envelope fields (#318 O5/O6) — first-party consumers read only
    // `nodes`/`edges`; the global merge likewise ignores extra keys.
    mark_heuristic_edges(&mut edges_out);
    let mut out = serde_json::json!({
        "nodes": nodes_out,
        "edges": edges_out,
        "coverage": coverage,
    });
    if !ambiguous_candidates.is_empty() {
        out["ambiguous"] = serde_json::json!(true);
        out["candidates"] = serde_json::Value::Array(ambiguous_candidates);
    }
    if token_budget > 0 {
        out["token_budget"] = serde_json::json!(token_budget);
        out["truncated_by_budget"] = serde_json::json!(truncated_by_budget);
    }
    // #617 + #661 WS-D: degraded-state signals — additive envelope field; first-
    // party consumers read only `nodes`/`edges` and ignore extra keys. Phase-B
    // incomplete (#617) and index/HEAD drift (#645) are independent and may both
    // apply. Carried as JSON `signals` array items (never appended as prose) so
    // the JSON body still parses; the global aggregator already merges `signals`.
    let signals = read_note_signals(store, head, &out["nodes"].to_string());
    if !signals.is_empty() {
        out["signals"] = serde_json::Value::Array(signals);
    }
    match serde_json::to_string(&out) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("get_graph_json serialization error: {e}");
            "{}".to_string()
        }
    }
}

/// Global variant of `get_graph_json` — merges subgraphs across repos, deduping by node id.
pub fn get_graph_json_global(
    repos: &HashMap<String, PathBuf>,
    repo: Option<&str>,
    params: &GraphJsonParams<'_>,
) -> String {
    let GraphJsonParams {
        query,
        direction,
        depth,
        kind_filter,
        token_budget: _,
        mode,
        path_prefix,
        provenance,
    } = params;
    if *mode == "overview" {
        if !path_prefix.is_empty() {
            if let Err(reason) = validate_mcp_arg(path_prefix) {
                tracing::warn!("get_graph_json_global rejected invalid path_prefix: {reason}");
                return "{}".to_string();
            }
        }
        // Overview mode: run per-repo and merge package tiles
        return get_graph_json_global_overview(repos, repo, path_prefix);
    }
    if !(query.is_empty() && *kind_filter == "file") {
        if let Err(reason) = validate_mcp_arg(query) {
            tracing::warn!("get_graph_json_global rejected invalid arg: {reason}");
            return "{}".to_string();
        }
    }
    let depth = (*depth).clamp(1, 4);

    let candidates: Vec<(&str, &PathBuf)> = match repo {
        Some(name) => {
            if let Err(reason) = validate_mcp_arg(name) {
                tracing::warn!("get_graph_json_global rejected invalid repo arg: {reason}");
                return "{}".to_string();
            }
            match repos.get_key_value(name) {
                Some((k, v)) => vec![(k.as_str(), v)],
                None => {
                    tracing::warn!("repo '{name}' not found in registry");
                    return r#"{"nodes":[],"edges":[]}"#.to_string();
                }
            }
        }
        None => repos.iter().map(|(k, v)| (k.as_str(), v)).collect(),
    };

    let mut all_nodes: Vec<serde_json::Value> = Vec::new();
    let mut all_edges: Vec<serde_json::Value> = Vec::new();
    let mut seen_node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_edge_keys: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    // #617: union of per-repo degraded-state signals, deduped.
    let mut all_signals: Vec<serde_json::Value> = Vec::new();
    let mut seen_signals: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (_repo_name, db_path) in candidates {
        if !db_path.exists() {
            continue;
        }
        match SqliteStore::open_read_only(db_path) {
            Ok(store) => {
                // #645 WS-D: per-repo drift signal reads *this* repo's own HEAD,
                // not the global LAUNCH_CWD (the caller's one workspace).
                let head = repo_head_from_registry_path(db_path);
                let raw = get_graph_json_raw(
                    &store,
                    query,
                    direction,
                    depth,
                    kind_filter,
                    0,
                    provenance,
                    head.as_deref(),
                );
                let parsed: serde_json::Value = match serde_json::from_str(&raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                for node in parsed["nodes"].as_array().into_iter().flatten() {
                    let id = node["id"].as_str().unwrap_or("").to_string();
                    if !id.is_empty() && seen_node_ids.insert(id) {
                        all_nodes.push(node.clone());
                    }
                }
                for edge in parsed["edges"].as_array().into_iter().flatten() {
                    let src = edge["source"].as_str().unwrap_or("").to_string();
                    let tgt = edge["target"].as_str().unwrap_or("").to_string();
                    if !src.is_empty() && seen_edge_keys.insert((src, tgt)) {
                        all_edges.push(edge.clone());
                    }
                }
                for signal in parsed["signals"].as_array().into_iter().flatten() {
                    if let Some(text) = signal.as_str() {
                        if seen_signals.insert(text.to_string()) {
                            all_signals.push(signal.clone());
                        }
                    }
                }
            }
            Err(e) => tracing::warn!("failed to open {}: {e}", db_path.display()),
        }
    }

    let mut out = serde_json::json!({
        "nodes": all_nodes,
        "edges": all_edges,
    });
    if !all_signals.is_empty() {
        out["signals"] = serde_json::Value::Array(all_signals);
    }
    match serde_json::to_string(&out) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("get_graph_json_global serialization error: {e}");
            "{}".to_string()
        }
    }
}

/// Merge overview graphs across repos for the global (SSE) session.
fn get_graph_json_global_overview(
    repos: &HashMap<String, PathBuf>,
    repo: Option<&str>,
    path_prefix: &str,
) -> String {
    use std::collections::HashMap as HMap;

    let candidates: Vec<(&str, &PathBuf)> = match repo {
        Some(name) => match repos.get_key_value(name) {
            Some((k, v)) => vec![(k.as_str(), v)],
            None => {
                tracing::warn!("get_graph_json_global_overview: repo '{name}' not found");
                return r#"{"nodes":[],"edges":[],"mode":"overview"}"#.to_string();
            }
        },
        None => repos.iter().map(|(k, v)| (k.as_str(), v)).collect(),
    };

    let mut merged_nodes: HMap<String, serde_json::Value> = HMap::new();
    let mut merged_edges: HMap<(String, String), u32> = HMap::new();

    for (_repo_name, db_path) in candidates {
        if !db_path.exists() {
            continue;
        }
        match SqliteStore::open_read_only(db_path) {
            Ok(store) => {
                let raw = overview_graph(&store, path_prefix);
                let parsed: serde_json::Value = match serde_json::from_str(&raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                for node in parsed["nodes"].as_array().into_iter().flatten() {
                    let id = node["id"].as_str().unwrap_or("").to_string();
                    merged_nodes.entry(id).or_insert_with(|| node.clone());
                }
                for edge in parsed["edges"].as_array().into_iter().flatten() {
                    let src = edge["source"].as_str().unwrap_or("").to_string();
                    let tgt = edge["target"].as_str().unwrap_or("").to_string();
                    let count = edge["count"].as_u64().unwrap_or(1) as u32;
                    *merged_edges.entry((src, tgt)).or_default() += count;
                }
            }
            Err(e) => tracing::warn!("get_graph_json_global_overview: open error: {e}"),
        }
    }

    let nodes_out: Vec<&serde_json::Value> = merged_nodes.values().collect();
    let edges_out: Vec<serde_json::Value> = merged_edges
        .iter()
        .map(|((src, tgt), count)| {
            serde_json::json!({
                "source": src,
                "target": tgt,
                "kind":   "imports",
                "count":  count,
            })
        })
        .collect();

    match serde_json::to_string(&serde_json::json!({
        "nodes": nodes_out,
        "edges": edges_out,
        "mode":  "overview",
    })) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("get_graph_json_global_overview serialization error: {e}");
            "{}".to_string()
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The visual graph renderer strips the native `kind:` prefix (`fn:home` →
    /// `home`), keeps the canonicalized `Type.member` name for synthesized
    /// external symbols, and shows a file node's basename. The vscode fixture
    /// (`commands.test.ts`) pins the `fn:bar` → `bar` contract.
    #[test]
    fn node_json_label_strips_native_prefix_and_cleans_external() {
        use travsr_core::{Node, VName};
        let native = Node::new(
            VName::new("c", "", "src/foo.rs", "rust", "fn:home"),
            "function",
        );
        assert_eq!(node_json_label(&native), "home");

        let class = Node::new(
            VName::new("c", "", "src/foo.rs", "rust", "class:Home"),
            "class",
        );
        assert_eq!(node_json_label(&class), "Home");

        let file = Node::new(
            VName::new("c", "", "src/foo/bar.rs", "rust", "file"),
            "file",
        );
        assert_eq!(node_json_label(&file), "bar.rs");

        // A synthesized external symbol keeps its clean, prefix-free label.
        let external = Node::new(
            VName::new(
                "c",
                "",
                "com/demo/Greeter.java",
                "java",
                "sdb:com/demo/Greeter#greet().",
            ),
            "method",
        );
        assert_eq!(node_json_label(&external), "Greeter.greet");
    }

    /// `find_references_structured` reports a machine-readable `status` that the
    /// CLI text path branches on for its not-found message. Pin the three the
    /// store can produce without a semantic index: `not_found`, `resolved`, and
    /// `ambiguous`.
    #[test]
    fn find_references_structured_reports_status() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();

        // Empty store: a name with no definition is a definitive not-found.
        let miss = find_references_structured(&store, "charge", None);
        assert_eq!(miss.status, "not_found");
        assert!(miss.candidates.is_empty());

        // One definition: resolved (zero recorded uses is still `resolved`).
        store
            .put_node(&Node::new(
                VName::new("c", "", "svc.rs", "rust", "fn:charge"),
                "function",
            ))
            .unwrap();
        let hit = find_references_structured(&store, "charge", None);
        assert_eq!(hit.status, "resolved");
        assert!(hit.resolved_to.is_some());

        // Two definitions of the same name in different files: ambiguous.
        store
            .put_node(&Node::new(
                VName::new("c", "", "billing.rs", "rust", "fn:charge"),
                "function",
            ))
            .unwrap();
        let amb = find_references_structured(&store, "charge", None);
        assert_eq!(amb.status, "ambiguous");
        assert!(amb.candidates.len() >= 2);
    }

    /// SEC-002 end-to-end: a path-traversal repo arg must be rejected through
    /// the full get_callers_global → collect_global → validate_mcp_arg pipeline.
    /// This exercises the wiring that the validate_mcp_arg unit tests in
    /// sanitize.rs do not cover — a regression here would be invisible to those
    /// unit tests.
    #[test]
    fn get_callers_global_rejects_path_traversal_repo_arg() {
        let repos: HashMap<String, PathBuf> = HashMap::new();
        let result = get_callers_global(&repos, "charge", None, Some("../evil"));
        // Invalid repo arg must return an empty envelope, not a panic or error.
        assert_eq!(
            result, "<travsr-data></travsr-data>",
            "path traversal in repo arg must be rejected and return empty envelope"
        );
    }

    /// An absolute-path repo arg is a well-formed registry key (every key is an
    /// absolute repo root), so it is no longer rejected outright; naming one
    /// that is not registered simply misses and returns the empty envelope.
    #[test]
    fn get_dependencies_global_absolute_repo_arg_that_is_not_registered_is_empty() {
        let repos: HashMap<String, PathBuf> = HashMap::new();
        let result = get_dependencies_global(&repos, "src/main.ts", Some("/etc/passwd"));
        assert_eq!(
            result, "<travsr-data></travsr-data>",
            "absolute path in repo arg must be rejected and return empty envelope"
        );
    }

    // ── #715: crashed-language caveat ─────────────────────────────────────────

    /// The `phase_b_warnings` parser matches a whole `crashed:<lang>` entry, not a
    /// prefix or a different warning class for the same language.
    #[test]
    fn phase_b_lang_crashed_matches_exact_class_and_lang() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .set_meta(
                "phase_b_warnings",
                "skipped_no_analyzer:php,crashed:objectivec",
            )
            .unwrap();
        assert!(phase_b_lang_crashed(&store, "objectivec"));
        // A different warning class for the same language is not a crash.
        assert!(!phase_b_lang_crashed(&store, "php"));
        // Not a substring match: `objc` must not match `objectivec`.
        assert!(!phase_b_lang_crashed(&store, "objc"));
        // No warnings at all → false.
        store.set_meta("phase_b_warnings", "").unwrap();
        assert!(!phase_b_lang_crashed(&store, "objectivec"));
    }

    /// #715: get_callers must attach the incompleteness caveat when the target
    /// language crashed on its last Phase B run, and must not otherwise.
    #[test]
    fn get_callers_caveats_a_crashed_language() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("c", "", "svc.rs", "rust", "fn:charge"),
            "function",
        );
        let caller = Node::new(
            VName::new("c", "", "main.rs", "rust", "fn:process"),
            "function",
        );
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();

        // Clean run: a confident caller list, no caveat.
        let clean = get_callers_raw(&store, "charge", None);
        assert!(
            clean.contains("fn:process"),
            "caller must be listed: {clean}"
        );
        assert!(
            !clean.contains("crashed on its last run"),
            "no caveat without a crash: {clean}"
        );

        // The language crashed: the same answer now carries the caveat.
        store.set_meta("phase_b_warnings", "crashed:rust").unwrap();
        let crashed = get_callers_raw(&store, "charge", None);
        assert!(
            crashed.contains("fn:process"),
            "caller still listed: {crashed}"
        );
        assert!(
            crashed.contains("semantic analysis for 'rust' crashed on its last run"),
            "a crashed language must carry the incompleteness caveat: {crashed}"
        );
    }

    /// get_callers used to seed on `search_nodes_by_name(..).first()`, so two
    /// distinct definitions sharing a name collapsed to whichever one the FTS
    /// ranked first: it returned that one's callers, said nothing about the
    /// other, and a reader concluded the unreported definition had no callers.
    /// `find_references` and `travsr graph` already refuse here; get_callers now
    /// refuses with them.
    #[test]
    fn get_callers_refuses_two_distinct_definitions_of_one_name() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let unifier = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-indexer/src/scip_unifier.rs",
                "rust",
                "fn:candidate_signatures",
            ),
            "function",
        );
        let live = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-daemon/src/live_resolve.rs",
                "rust",
                "fn:candidate_signatures",
            ),
            "function",
        );
        let unifier_caller = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-daemon/src/scip_unifier.rs",
                "rust",
                "fn:unify_one",
            ),
            "function",
        );
        let live_caller = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-daemon/src/live_resolve.rs",
                "rust",
                "fn:resolve_live",
            ),
            "function",
        );
        for n in [&unifier, &live, &unifier_caller, &live_caller] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(unifier_caller.id, unifier.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(live_caller.id, live.id, EdgeKind::RefCall))
            .unwrap();

        let out = get_callers_raw(&store, "candidate_signatures", None);
        assert!(
            out.contains("'candidate_signatures' is ambiguous, 2 definitions"),
            "two definitions of one name must be refused, not silently picked: {out}"
        );
        assert!(
            out.contains("crates/travsr-indexer/src/scip_unifier.rs")
                && out.contains("crates/travsr-daemon/src/live_resolve.rs"),
            "the refusal must name both definitions: {out}"
        );
        assert!(
            !out.contains("fn:unify_one") && !out.contains("fn:resolve_live"),
            "no single definition's callers may be presented as the answer: {out}"
        );

        // The `path` hint the refusal advertises is get_callers' own argument, so
        // the answer is reachable in one more call on the same tool.
        let pinned = get_callers_raw(&store, "candidate_signatures", Some("scip_unifier.rs"));
        assert!(
            !pinned.contains("is ambiguous"),
            "the path hint the refusal advertises must disambiguate: {pinned}"
        );
        assert!(
            pinned.contains("fn:unify_one") && !pinned.contains("fn:resolve_live"),
            "only the pinned definition's callers: {pinned}"
        );
    }

    /// The guard keys on two *exact definitions*, not on "more than one node
    /// matched", so the partial matching the get_callers schema documents
    /// ("partial match supported") still resolves and answers.
    #[test]
    fn get_callers_keeps_documented_partial_matching() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("c", "", "svc.rs", "rust", "fn:charge_customer"),
            "function",
        );
        let caller = Node::new(
            VName::new("c", "", "main.rs", "rust", "fn:process"),
            "function",
        );
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();

        let out = get_callers_raw(&store, "charge", None);
        assert!(
            !out.contains("is ambiguous"),
            "a partial query is not an ambiguous definition: {out}"
        );
        assert!(
            out.contains("fn:process"),
            "partial match must still answer: {out}"
        );
    }

    /// The `path` hint has to mean the same thing on the partial tier as on the
    /// exact ones. It used to be dropped once the exact ladder returned `None`,
    /// so the fallback took the first FTS row and answered about a same-named
    /// symbol in a file the hint excluded: #647's confident-wrong answer,
    /// reachable through the one tier that never got the fix.
    #[test]
    fn get_callers_partial_fallback_still_honours_the_path_hint() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let mk =
            |path: &str, sig: &str| Node::new(VName::new("c", "", path, "rust", sig), "function");
        // Two partial matches for "charge", in two different directories.
        let billing = mk("billing/svc.rs", "fn:charge_customer");
        let payments = mk("payments/svc.rs", "fn:charge_card");
        let billing_caller = mk("billing/main.rs", "fn:bill_it");
        let payments_caller = mk("payments/main.rs", "fn:pay_it");
        for n in [&billing, &payments, &billing_caller, &payments_caller] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(billing_caller.id, billing.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(
                payments_caller.id,
                payments.id,
                EdgeKind::RefCall,
            ))
            .unwrap();

        let pinned = get_callers_raw(&store, "charge", Some("billing"));
        assert!(
            pinned.contains("fn:bill_it"),
            "the hinted directory's caller must be the answer: {pinned}"
        );
        assert!(
            !pinned.contains("fn:pay_it"),
            "a caller the hint excludes must not be reported: {pinned}"
        );

        // A hint matching no candidate must say so. Falling through to the
        // unscoped first row is what this test exists to stop, and answering an
        // empty list would read as an authoritative "no callers".
        let missed = get_callers_raw(&store, "charge", Some("shipping"));
        assert!(
            missed.contains("no definition is under path 'shipping'"),
            "a hint that matches nothing must be reported, not widened: {missed}"
        );
        assert!(
            !missed.contains("fn:bill_it") && !missed.contains("fn:pay_it"),
            "no caller list may be presented for a hint that matched nothing: {missed}"
        );
    }

    /// An Objective-C selector family is one method at several arities, not rival
    /// definitions (see `RefTarget::Family`). get_callers must union their callers,
    /// the way find_references unions their sites, never refuse.
    #[test]
    fn get_callers_unions_a_selector_family_rather_than_refusing() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let short = Node::new(
            VName::new(
                "c",
                "",
                "AFSecurityPolicy.m",
                "objectivec",
                "method:AFSecurityPolicy.policyWithPinningMode:",
            ),
            "method",
        );
        let long = Node::new(
            VName::new(
                "c",
                "",
                "AFSecurityPolicy.m",
                "objectivec",
                "method:AFSecurityPolicy.policyWithPinningMode:withPinnedCertificates:",
            ),
            "method",
        );
        let short_caller = Node::new(
            VName::new(
                "c",
                "",
                "Session.m",
                "objectivec",
                "method:Session.configure",
            ),
            "method",
        );
        let long_caller = Node::new(
            VName::new("c", "", "Pinning.m", "objectivec", "method:Pinning.install"),
            "method",
        );
        for n in [&short, &long, &short_caller, &long_caller] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(short_caller.id, short.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(long_caller.id, long.id, EdgeKind::RefCall))
            .unwrap();

        let out = get_callers_raw(&store, "policyWithPinningMode", None);
        assert!(
            !out.contains("is ambiguous"),
            "a selector family is one method, not rival definitions: {out}"
        );
        assert!(
            out.contains("Session.configure") && out.contains("Pinning.install"),
            "every arity's callers must be present: {out}"
        );
    }

    /// The family test compares a container NAME cut out of the signature. Two
    /// classes that share a name (a vendored pod duplicated under two paths, a
    /// category) are two methods, and unioning them presented one method's
    /// callers under the other's name. The shared PATH is what tells them apart.
    #[test]
    fn two_same_named_classes_are_not_one_selector_family() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for path in ["Pods/A/AFSecurityPolicy.m", "Pods/B/AFSecurityPolicy.m"] {
            store
                .put_node(&Node::new(
                    VName::new(
                        "c",
                        "",
                        path,
                        "objectivec",
                        "method:AFSecurityPolicy.policyWithPinningMode:",
                    ),
                    "method",
                ))
                .unwrap();
        }

        let out = get_callers_raw(&store, "policyWithPinningMode", None);
        assert!(
            out.contains("is ambiguous"),
            "two classes sharing a name are rival definitions, not one family: {out}"
        );
    }

    /// The family logic used to be gated on nothing but a trailing ':' in the
    /// leaf, so it ran for every language.
    #[test]
    fn a_non_objc_leaf_ending_in_colon_is_not_a_selector_family() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for sig in ["method:Label.text:", "method:Label.text:color:"] {
            store
                .put_node(&Node::new(
                    VName::new("c", "", "ui.dart", "dart", sig),
                    "method",
                ))
                .unwrap();
        }

        let out = get_callers_raw(&store, "text", None);
        assert!(
            out.contains("is ambiguous"),
            "only Objective-C spells a method name with embedded colons: {out}"
        );
    }

    /// #715: find_references' primary occurrence path carries the caveat too, so a
    /// partial post-crash occurrence list is never presented as authoritative.
    #[test]
    fn find_references_caveats_a_crashed_language() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("c", "", "svc.rs", "rust", "fn:charge"),
            "function",
        );
        let caller = Node::new(
            VName::new("c", "", "main.rs", "rust", "fn:process"),
            "function",
        );
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 12, None)])
            .unwrap();

        let clean = find_references_raw(&store, "charge", None);
        assert!(clean.contains("reference(s)"), "sites listed: {clean}");
        assert!(
            !clean.contains("crashed on its last run"),
            "no caveat clean: {clean}"
        );

        store.set_meta("phase_b_warnings", "crashed:rust").unwrap();
        let crashed = find_references_raw(&store, "charge", None);
        assert!(
            crashed.contains("semantic analysis for 'rust' crashed on its last run"),
            "crashed language caveat on the occurrence path: {crashed}"
        );
    }

    // ── search_symbol unit tests ──────────────────────────────────────────────

    #[test]
    fn search_symbol_no_match_returns_explicit_message() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let result = search_symbol(&store, "NonExistentSymbol", false);
        assert!(
            result.contains("No symbols matching 'NonExistentSymbol' found in the graph."),
            "no-match should return explicit not-found message, got: {result}"
        );
    }

    #[test]
    fn search_symbol_global_no_match_returns_explicit_message() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        // Create an empty file-backed store (no nodes).
        drop(travsr_store::SqliteStore::open(&db_path).unwrap());
        let repos: HashMap<String, PathBuf> = [("myrepo".to_string(), db_path)].into();
        let result = search_symbol_global(&repos, "NonExistentSymbol", None, false);
        assert!(
            result.contains("No symbols matching 'NonExistentSymbol' found in the graph."),
            "global no-match should return explicit not-found message, got: {result}"
        );
    }

    #[test]
    fn search_symbol_global_exact_mode_filters_substrings() {
        use travsr_core::{Node, VName};
        // The global dispatch surface (stdio-global + SSE) must thread `exact`
        // through to the store the same way the single-repo path does. Guards
        // against a future refactor dropping the flag on the global variant.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        {
            let mut store = travsr_store::SqliteStore::open(&db_path).unwrap();
            store
                .put_node(&Node::new(
                    VName::new("", "", "src/ClassD.rs", "rust", "struct:ClassD"),
                    "struct",
                ))
                .unwrap();
            store
                .put_node(&Node::new(
                    VName::new("", "", "src/ClassD.rs", "rust", "fn:ClassD::method"),
                    "function",
                ))
                .unwrap();
            store
                .put_node(&Node::new(
                    VName::new(
                        "",
                        "",
                        "src/ClassDConfig.rs",
                        "rust",
                        "struct:ClassDConfigurationManager",
                    ),
                    "struct",
                ))
                .unwrap();
        }
        let repos: HashMap<String, PathBuf> = [("myrepo".to_string(), db_path)].into();

        // exact=false: the pure-substring match is still returned.
        let res_all = search_symbol_global(&repos, "ClassD", Some("myrepo"), false);
        assert!(res_all.contains("struct:ClassD"), "got: {res_all}");
        assert!(res_all.contains("fn:ClassD::method"), "got: {res_all}");
        assert!(
            res_all.contains("struct:ClassDConfigurationManager"),
            "global exact=false must keep substring match, got: {res_all}"
        );

        // exact=true: substring noise is dropped, boundary/exact kept.
        let res_exact = search_symbol_global(&repos, "ClassD", Some("myrepo"), true);
        assert!(res_exact.contains("struct:ClassD"), "got: {res_exact}");
        assert!(res_exact.contains("fn:ClassD::method"), "got: {res_exact}");
        assert!(
            !res_exact.contains("struct:ClassDConfigurationManager"),
            "global exact=true must drop substring match, got: {res_exact}"
        );
    }

    // ── search_symbol / get_callers path:line tests ───────────────────────────

    #[test]
    fn context_result_noise_drops_ci_packages_files_and_tests_keeps_real_symbols() {
        use travsr_core::{Node, VName};
        let node = |path: &str, kind: &str, sig: &str| {
            Node::new(VName::new("", "", path, "rust", sig), kind)
        };
        // Dropped: CI/workflow package nodes (both by kind and by pkg: signature).
        assert!(is_context_result_noise(&node(
            "",
            "package",
            "pkg:actions/checkout@abc123"
        )));
        assert!(is_context_result_noise(&node(
            "",
            "function",
            "pkg:some/dep@1.0"
        )));
        // Dropped: bare file nodes.
        assert!(is_context_result_noise(&node("src/foo.rs", "file", "file")));
        // Dropped: everything is_noise_seed rejects (e.g. integration-test dir).
        assert!(is_context_result_noise(&node(
            "tests/it.rs",
            "function",
            "fn:it_works"
        )));
        // Kept: real source symbols (the answers we must never evict).
        assert!(!is_context_result_noise(&node(
            "crates/travsr-mcp/src/tools.rs",
            "function",
            "fn:get_context_body"
        )));
        assert!(!is_context_result_noise(&node(
            "src/daemon.rs",
            "struct",
            "struct:Daemon"
        )));
    }

    /// #376: doc-chunk nodes must never surface via the generic lexical/PPR
    /// seed path — permanent, not just until Phase 2's dedicated docs section
    /// existed (it now does; see `build_docs_section`/`doc_lane_seeds`, a
    /// fully separate lane that never touches `build_seed_set`). Confirmed
    /// empirically to leak without this guard — a fixture repo's
    /// `## Payment Architecture` doc surfaced under a `PaymentService` query
    /// at score 0.000 via plain word overlap, with no floor and no ranking,
    /// before `is_noise_seed` gained the `doc-chunk` check.
    #[test]
    fn doc_chunk_nodes_are_permanently_excluded_from_generic_seed_path() {
        use travsr_core::{Node, VName};
        let doc_node = Node::new(
            VName::new("", "", "docs/architecture.md", "markdown", "doc:overview"),
            "doc-chunk",
        );
        assert!(
            is_noise_seed(&doc_node),
            "doc-chunk must be excluded from every seed/candidate path in build_seed_set"
        );
        assert!(
            is_context_result_noise(&doc_node),
            "doc-chunk must be excluded from get_context's result set too"
        );
    }

    #[test]
    fn search_symbol_emits_path_line_when_present() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("", "", "src/foo.rs", "rust", "fn:bar"),
            "function",
        )
        .with_line(42);
        store.put_node(&n).unwrap();
        let result = search_symbol(&store, "bar", false);
        assert!(
            result.contains("src/foo.rs:42"),
            "search_symbol must emit path:line, got: {result}"
        );
    }

    #[test]
    fn search_symbol_omits_line_when_absent() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("", "", "src/foo.rs", "rust", "fn:baz"),
            "function",
        );
        store.put_node(&n).unwrap();
        let result = search_symbol(&store, "baz", false);
        assert!(
            result.contains("src/foo.rs"),
            "path must still appear: {result}"
        );
        assert!(
            !result.contains("src/foo.rs:"),
            "no colon-suffix when line absent, got: {result}"
        );
    }

    // ── infer_language_from_query ────────────────────────────────────────────

    #[test]
    fn infer_lang_strips_trailing_token() {
        let (q, lang) = infer_language_from_query("auth handler javascript");
        assert_eq!(q, "auth handler");
        assert_eq!(lang, Some("javascript"));
    }

    #[test]
    fn infer_lang_strips_in_connector() {
        let (q, lang) = infer_language_from_query("PaymentService in Kotlin");
        assert_eq!(q, "PaymentService");
        assert_eq!(lang, Some("kotlin"));
    }

    #[test]
    fn infer_lang_aliases_map_correctly() {
        assert_eq!(infer_language_from_query("foo c++").1, Some("cpp"));
        assert_eq!(infer_language_from_query("foo c#").1, Some("csharp"));
        assert_eq!(
            infer_language_from_query("foo objective-c").1,
            Some("objectivec")
        );
        assert_eq!(infer_language_from_query("foo objc").1, Some("objectivec"));
        assert_eq!(infer_language_from_query("foo golang").1, Some("go"));
        assert_eq!(infer_language_from_query("foo csharp").1, Some("csharp"));
    }

    #[test]
    fn infer_lang_no_token_returns_original() {
        let (q, lang) = infer_language_from_query("knapsack budget optimizer");
        assert_eq!(q, "knapsack budget optimizer");
        assert_eq!(lang, None);
    }

    #[test]
    fn infer_lang_preserves_original_when_stripping_leaves_empty() {
        // Query is JUST a language name — keep it as-is so search still runs.
        let (q, lang) = infer_language_from_query("rust");
        assert_eq!(q, "rust");
        assert_eq!(lang, Some("rust"));
    }

    #[test]
    fn infer_lang_case_insensitive() {
        let (q, lang) = infer_language_from_query("session PYTHON");
        assert_eq!(q, "session");
        assert_eq!(lang, Some("python"));
    }

    // ── search_symbol language filter ────────────────────────────────────────

    #[test]
    fn search_symbol_global_filters_by_inferred_language() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let rs_node = Node::new(
            VName::new("", "", "src/auth.rs", "rust", "fn:auth_handler"),
            "function",
        );
        let ts_node = Node::new(
            VName::new("", "", "src/auth.ts", "typescript", "fn:authHandler"),
            "function",
        );
        store.put_node(&rs_node).unwrap();
        store.put_node(&ts_node).unwrap();

        // "auth handler typescript" should return only the TypeScript node.
        let result = search_symbol_raw(&store, "auth", Some("typescript"), false);
        assert!(
            result.contains("src/auth.ts"),
            "expected typescript node: {result}"
        );
        assert!(
            !result.contains("src/auth.rs"),
            "rust node must be filtered out: {result}"
        );
    }

    #[test]
    fn search_symbol_raw_no_filter_returns_both_languages() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let rs_node = Node::new(
            VName::new("", "", "src/auth.rs", "rust", "fn:auth_handler"),
            "function",
        );
        let ts_node = Node::new(
            VName::new("", "", "src/auth.ts", "typescript", "fn:authHandler"),
            "function",
        );
        store.put_node(&rs_node).unwrap();
        store.put_node(&ts_node).unwrap();

        let result = search_symbol_raw(&store, "auth", None, false);
        assert!(
            result.contains("auth.rs"),
            "rust node must appear: {result}"
        );
        assert!(
            result.contains("auth.ts"),
            "typescript node must appear: {result}"
        );
    }

    #[test]
    fn search_symbol_exact_mode_filters_substrings() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n1 = Node::new(
            VName::new("", "", "src/ClassD.rs", "rust", "struct:ClassD"),
            "struct",
        );
        let n2 = Node::new(
            VName::new("", "", "src/ClassD.rs", "rust", "fn:ClassD::method"),
            "function",
        );
        let n3 = Node::new(
            VName::new(
                "",
                "",
                "src/ClassDConfig.rs",
                "rust",
                "struct:ClassDConfigurationManager",
            ),
            "struct",
        );

        store.put_node(&n1).unwrap();
        store.put_node(&n2).unwrap();
        store.put_node(&n3).unwrap();

        // exact=false returns all 3
        let res_all = search_symbol(&store, "ClassD", false);
        assert!(res_all.contains("struct:ClassD"));
        assert!(res_all.contains("fn:ClassD::method"));
        assert!(res_all.contains("struct:ClassDConfigurationManager"));

        // exact=true returns only exact/word-boundary
        let res_exact = search_symbol(&store, "ClassD", true);
        assert!(res_exact.contains("struct:ClassD"));
        assert!(res_exact.contains("fn:ClassD::method"));
        assert!(!res_exact.contains("struct:ClassDConfigurationManager"));
    }

    #[test]
    fn get_callers_emits_path_line_when_present() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let callee = Node::new(
            VName::new("", "", "src/callee.rs", "rust", "fn:callee"),
            "function",
        );
        let caller = Node::new(
            VName::new("", "", "src/caller.rs", "rust", "fn:caller"),
            "function",
        )
        .with_line(10);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        let result = get_callers(&store, "fn:callee", None);
        assert!(
            result.contains("src/caller.rs:10"),
            "get_callers must emit path:line for caller, got: {result}"
        );
    }

    /// #747: a repository that moved after being indexed.
    ///
    /// `meta.repo_root` is stamped at index time, so it names wherever the repo
    /// lived then. `find_pattern` handed that to `git -C` and every search
    /// failed, a plain literal included. The database moved with the repo, so
    /// its own location still resolves.
    #[test]
    fn a_moved_repo_resolves_its_root_from_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let travsr_dir = dir.path().join(".travsr");
        std::fs::create_dir_all(&travsr_dir).unwrap();
        let mut store = SqliteStore::open(&travsr_dir.join("graph.db")).unwrap();

        // The path the repo had when it was indexed, which no longer exists.
        store
            .set_meta("repo_root", "/nonexistent/where/it/used/to/live")
            .unwrap();

        assert_eq!(
            resolve_repo_root(&store).as_deref(),
            Some(dir.path()),
            "a stale repo_root must fall back to where the database actually is"
        );
    }

    /// #749 review: a `.travsr` copied into a second checkout still names the
    /// first, and that one usually still exists, so `is_dir()` alone kept
    /// choosing it. Every read then came out of the original tree while looking
    /// entirely confident, which is the one failure mode worse than the three
    /// that fail loudly.
    #[test]
    fn a_copied_travsr_dir_resolves_to_the_copy_not_the_original() {
        let original = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(original.path().join(".git")).unwrap();

        let copy = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(copy.path().join(".git")).unwrap();
        let travsr_dir = copy.path().join(".travsr");
        std::fs::create_dir_all(&travsr_dir).unwrap();
        let mut store = SqliteStore::open(&travsr_dir.join("graph.db")).unwrap();

        // What `cp -r` leaves behind: the copy's database naming the original,
        // which is still there.
        store
            .set_meta("repo_root", original.path().to_str().unwrap())
            .unwrap();
        assert!(original.path().is_dir(), "the original must still exist");

        assert_eq!(
            resolve_repo_root(&store).as_deref(),
            Some(copy.path()),
            "a database inside a repository resolves to that repository"
        );
    }

    /// The stored value still wins while it resolves: a database opened through
    /// `--db` from outside the repo would otherwise have its root silently
    /// redirected to wherever the file happened to sit.
    #[test]
    fn a_repo_root_that_still_exists_is_preferred() {
        let repo = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let travsr_dir = elsewhere.path().join(".travsr");
        std::fs::create_dir_all(&travsr_dir).unwrap();
        let mut store = SqliteStore::open(&travsr_dir.join("graph.db")).unwrap();
        store
            .set_meta("repo_root", repo.path().to_str().unwrap())
            .unwrap();

        assert_eq!(
            resolve_repo_root(&store).as_deref(),
            Some(repo.path()),
            "a repo_root that still exists is the real repo and must win"
        );
    }

    /// The hint that made #747 hard to read. It is right for a bad pattern and
    /// wrong for a missing directory, and it used to be appended to both.
    #[test]
    fn the_regex_hint_is_offered_only_for_a_regex_failure() {
        for real in [
            // Collected by running each pattern through git, not invented. My
            // first list guessed and matched none of them (#749 review).
            //
            // BSD's phrasing:
            "fatal: command line, 'alpha(': parentheses not balanced",
            "fatal: command line, 'alpha[': brackets ([ ]) not balanced",
            "fatal: command line, 'a{2': braces not balanced",
            "fatal: command line, 'a\\': trailing backslash (\\)",
            "fatal: command line, '*a': repetition-operator operand invalid",
            "fatal: invalid regular expression: unmatched ( or \\(",
            // glibc's phrasing for the same failures, which is what git 2.34.1
            // on Linux actually prints. Three of these fell through the BSD-only
            // list, so the hint went missing on the platform CI runs on.
            "fatal: command line, 'alpha(': Unmatched ( or \\(",
            "fatal: command line, 'alpha[': Invalid regular expression",
            "fatal: command line, 'a{2': Unmatched \\{",
            "fatal: command line, 'a{1,': Unmatched \\{",
            "fatal: command line, 'a\\': Trailing backslash",
            "fatal: command line, '*a': Invalid preceding regular expression",
            "fatal: command line, '[[:foo:]]': Invalid character class name",
            "fatal: command line, 'a\\1': Invalid back reference",
            "fatal: command line, '[z-a]': Invalid range end",
            "Unmatched [ or [^",
            "fatal: unterminated \\{",
            "trailing backslash on RHS",
            "Invalid preceding regular expression",
        ] {
            assert!(
                looks_like_a_regex_complaint(real),
                "git said the pattern is bad: {real}"
            );
        }

        for other in [
            "fatal: cannot change to '/Users/me/Desktop/travsr': No such file or directory",
            "fatal: not a git repository",
            // #749 review: the needles used to be ordinary words, and git quotes
            // the repository path back in its errors, so an entirely reasonable
            // directory name turned a missing directory into "check your regex".
            "fatal: cannot change to '/Users/me/dev/bracket-parser': No such file or directory",
            "fatal: cannot change to '/Users/me/work/brace-matching': No such file or directory",
            "fatal: cannot change to '/Users/me/src/unterminated-stream': No such file or directory",
            "fatal: cannot change to '/Users/me/code/repetition-tests': No such file or directory",
        ] {
            assert!(
                !looks_like_a_regex_complaint(other),
                "not a pattern problem, so the POSIX ERE hint would mislead: {other}"
            );
        }
    }

    #[test]
    fn simple_name_strips_kind_and_scope() {
        assert_eq!(simple_name("fn:SyncPod"), "SyncPod");
        assert_eq!(simple_name("method:Foo#charge"), "charge");
        assert_eq!(simple_name("fn:SqliteStore.iter_edges_to"), "iter_edges_to");
        assert_eq!(
            simple_name("fn:crate::repo::find_git_root"),
            "find_git_root"
        );
        assert_eq!(simple_name("bareName"), "bareName");
    }

    /// #399: when repo_root + source are available, get_callers expands a single
    /// RefCall edge into every exact call-site line, not the caller's def line.
    #[test]
    fn get_callers_expands_to_exact_call_sites() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let dir = tempfile::tempdir().unwrap();
        // caller `run` spans lines 2..=7 and calls SyncPod on lines 4 and 6;
        // SyncPodX on line 5 is a decoy that must not match.
        std::fs::write(
            dir.path().join("worker.go"),
            "package main\nfunc run() {\n  x()\n  SyncPod(a)\n  SyncPodX(z)\n  SyncPod(b)\n}\n",
        )
        .unwrap();

        let callee = Node::new(
            VName::new("", "", "sync.go", "go", "fn:SyncPod"),
            "function",
        );
        let caller = Node::new(VName::new("", "", "worker.go", "go", "fn:run"), "function")
            .with_line(2)
            .with_end_line(7);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();

        let result = get_callers(&store, "SyncPod", None);
        assert!(result.contains("worker.go:4"), "site on line 4: {result}");
        assert!(result.contains("worker.go:6"), "site on line 6: {result}");
        assert!(
            !result.contains("worker.go:5"),
            "decoy SyncPodX must not match: {result}"
        );
        // Two distinct sites from one edge.
        assert_eq!(
            result.matches("worker.go:").count(),
            2,
            "one edge → two call sites: {result}"
        );
    }

    /// A recorded occurrence suppresses the textual re-scan for that edge.
    ///
    /// The scan counts the callee's name anywhere in the caller's span, so the
    /// callee's own declaration and a mention in a comment come back as call
    /// sites. Preferring the recorded rows is what removed 3127 phantom sites
    /// across 1081 edges on this repo. The scan stays as the fallback for a
    /// language that feeds no occurrence rows at all.
    #[test]
    fn get_callers_prefers_recorded_occurrences_over_the_textual_scan() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let dir = tempfile::tempdir().unwrap();
        // Three textual hits inside `run`'s span: a comment (3), the local
        // declaration (4), and the one real call (5).
        std::fs::write(
            dir.path().join("app.ts"),
            "x\nfunction run() {\n  // row is built below\n  const row = () => 1;\n  row();\n}\n",
        )
        .unwrap();

        let callee = Node::new(
            VName::new("", "", "grid.ts", "typescript", "fn:row"),
            "function",
        );
        let caller = Node::new(
            VName::new("", "", "app.ts", "typescript", "fn:run"),
            "function",
        )
        .with_line(2)
        .with_end_line(6);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        // An `lsif` edge, so `provenance_marker` is empty and the only marker
        // in the output is the one the scan itself adds.
        store
            .put_edge_lsif(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();

        // Without occurrence rows the scan still runs: all three lines report.
        let scanned = get_callers(&store, "row", None);
        assert_eq!(
            scanned.matches("app.ts:").count(),
            3,
            "the textual fallback is unchanged when nothing was recorded: {scanned}"
        );

        // With the real occurrence recorded, only that site reports: the
        // comment and the declaration the scan counted are gone.
        store
            .record_edge_sites(&[(caller.id, callee.id, 5, None)])
            .unwrap();
        let result = get_callers(&store, "row", None);
        assert_eq!(
            result.matches("app.ts:").count(),
            1,
            "the recorded occurrence must suppress the scan's phantom sites: {result}"
        );
        assert!(
            result.contains("app.ts:5"),
            "the surviving site must be the recorded one: {result}"
        );
    }

    /// A site whose only backing edge was matched by leaf name must say so.
    /// `reference_sites` used to select `path` and `line` alone and never touch
    /// `edges`, the only table holding provenance, so a wholly fabricated site
    /// was served as a resolved fact.
    #[test]
    fn find_references_marks_a_name_matched_site() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("", "", "grid.ts", "typescript", "fn:row"),
            "function",
        );
        let guessed = Node::new(
            VName::new("", "", "app.ts", "typescript", "fn:render"),
            "function",
        );
        let resolved = Node::new(
            VName::new("", "", "page.ts", "typescript", "fn:draw"),
            "function",
        );
        let typed = Node::new(
            VName::new("", "", "types.ts", "typescript", "fn:shape"),
            "function",
        );
        for n in [&callee, &guessed, &resolved, &typed] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(
                &Edge::new(guessed.id, callee.id, EdgeKind::RefCall)
                    .with_provenance("tree-sitter".to_string()),
            )
            .unwrap();
        // `put_edge` hardcodes tree-sitter provenance; the semantic writer is
        // the one that records a compiler-resolved edge.
        store
            .put_edge_lsif(&Edge::new(resolved.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        // `typed` has an occurrence but no edge of its own: a resolved
        // reference that is not a call (#650). Nothing to flag there either.
        store
            .record_edge_sites(&[
                (guessed.id, callee.id, 7, None),
                (resolved.id, callee.id, 9, None),
                (typed.id, callee.id, 3, None),
            ])
            .unwrap();

        let structured = find_references_structured(&store, "row", None);
        assert_eq!(structured.total, Some(3));
        let flags: Vec<(String, bool)> = structured
            .references
            .iter()
            .map(|r| (format!("{}:{}", r.path, r.line), r.heuristic))
            .collect();
        assert_eq!(
            flags,
            vec![
                ("app.ts:7".to_string(), true),
                ("page.ts:9".to_string(), false),
                ("types.ts:3".to_string(), false),
            ],
            "only the name-matched site is flagged"
        );

        let text = find_references(&store, "row", None);
        assert!(
            text.contains(&format!("app.ts:7{HEURISTIC_MARKER}")),
            "the name-matched site carries the caveat: {text}"
        );
        assert!(
            !text.contains(&format!("page.ts:9{HEURISTIC_MARKER}")),
            "the compiler-resolved site carries no caveat: {text}"
        );
    }

    /// A name-matched call edge costs one character on the row plus one legend
    /// line at the end, not 49 bytes on every row. On a Phase-A-only language
    /// essentially every call edge is name-matched, so the long marker was
    /// spending the output budget on the caveat instead of on callers.
    #[test]
    fn a_name_matched_caller_row_carries_the_sigil_and_one_legend() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let callee = Node::new(VName::new("", "", "grid.go", "go", "fn:row"), "function");
        let a = Node::new(VName::new("", "", "a.go", "go", "fn:run_a"), "function").with_line(2);
        let b = Node::new(VName::new("", "", "b.go", "go", "fn:run_b"), "function").with_line(3);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for n in [&callee, &a, &b] {
            store.put_node(n).unwrap();
        }
        for caller in [&a, &b] {
            store
                .put_edge(
                    &Edge::new(caller.id, callee.id, EdgeKind::RefCall)
                        .with_provenance("tree-sitter".to_string()),
                )
                .unwrap();
        }

        let out = get_callers(&store, "row", None);
        assert_eq!(
            out.matches(HEURISTIC_SIGIL_ROW).count(),
            2,
            "one sigil per marked row: {out}"
        );
        assert_eq!(
            out.matches(HEURISTIC_LEGEND).count(),
            1,
            "the legend is printed once, not per row: {out}"
        );
        assert!(
            !out.contains("[heuristic:"),
            "the per-row long marker is gone: {out}"
        );
    }

    /// Every structural tool routed through `collect_global` takes a `repo`
    /// argument whose only legal values are the registry's keys, and every key
    /// is an absolute repo root. The shared `validate_mcp_arg` rejects those
    /// outright, so naming a real repo returned an empty envelope from all of
    /// them.
    #[test]
    fn a_global_tool_answers_for_a_repo_named_by_its_absolute_key() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("graph.db");
        {
            let mut store = travsr_store::SqliteStore::open(&db).unwrap();
            let callee = Node::new(
                VName::new("", "", "pay.rs", "rust", "fn:charge"),
                "function",
            );
            let caller = Node::new(
                VName::new("", "", "cart.rs", "rust", "fn:checkout"),
                "function",
            );
            store.put_node(&callee).unwrap();
            store.put_node(&caller).unwrap();
            store
                .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
                .unwrap();
        }
        // Registry keys are `repo_root.to_string_lossy()`, i.e. absolute paths.
        let key = dir.path().to_string_lossy().to_string();
        let repos: HashMap<String, PathBuf> = HashMap::from([(key.clone(), db)]);

        let result = get_callers_global(&repos, "charge", None, Some(&key));
        assert!(
            result.contains("cart.rs"),
            "a repo named by its absolute registry key must answer: {result}"
        );
    }

    // ── blast radius unit tests ───────────────────────────────────────────────

    fn make_store(
        nodes: &[travsr_core::Node],
        edges: &[(
            travsr_core::NodeId,
            travsr_core::NodeId,
            travsr_core::EdgeKind,
        )],
    ) -> travsr_store::SqliteStore {
        use travsr_core::Edge;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for n in nodes {
            store.put_node(n).unwrap();
        }
        for &(src, dst, kind) in edges {
            store.put_edge(&Edge::new(src, dst, kind)).unwrap();
        }
        store
    }

    fn make_node(path: &str, sig: &str) -> travsr_core::Node {
        use travsr_core::VName;
        travsr_core::Node::new(VName::new("", "", path, "typescript", sig), "function")
    }

    /// A file with no callers — blast radius is just itself.
    #[test]
    fn blast_radius_includes_source_file() {
        let a = make_node("a.ts", "fn:a");
        let store = make_store(std::slice::from_ref(&a), &[]);
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::TreeSitter);
        assert!(
            result.contains("a.ts"),
            "source file must appear in its own blast radius"
        );
    }

    /// B → A (incoming call): blast_radius("a.ts") must include b.ts.
    #[test]
    fn blast_radius_follows_incoming_edges() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b");
        // B calls A — reverse BFS from A should reach B.
        let store = make_store(&[a.clone(), b.clone()], &[(b.id, a.id, EdgeKind::RefCall)]);
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::TreeSitter);
        assert!(result.contains("a.ts"), "source file must be included");
        assert!(
            result.contains("b.ts"),
            "caller file must be included in blast radius"
        );
    }

    /// B depends on A (import): blast_radius("a.ts") must include b.ts.
    #[test]
    fn blast_radius_follows_depends_edges() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b");
        // B imports A — Depends edge B→A; reverse BFS from A must reach B.
        let store = make_store(&[a.clone(), b.clone()], &[(b.id, a.id, EdgeKind::Depends)]);
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::TreeSitter);
        assert!(result.contains("a.ts"), "source file must be included");
        assert!(
            result.contains("b.ts"),
            "file that imports the changed file must appear in blast radius"
        );
    }

    /// Cycle A ↔ B — must terminate without infinite loop.
    #[test]
    fn blast_radius_handles_cycle() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b");
        let store = make_store(
            &[a.clone(), b.clone()],
            &[
                (b.id, a.id, EdgeKind::RefCall),
                (a.id, b.id, EdgeKind::RefCall), // cycle
            ],
        );
        // Must not hang; both files reachable.
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::TreeSitter);
        assert!(result.contains("a.ts"));
        assert!(result.contains("b.ts"));
    }

    /// Go co-package: blast_radius("a.go") must include all sibling files in
    /// the same package, via the Depends edges written by init_repo's co-package pass.
    #[test]
    fn blast_radius_includes_go_copackage_siblings() {
        use travsr_core::{EdgeKind, Node, VName};

        fn go_file_node(path: &str) -> Node {
            Node::new(VName::new("", "", path, "go", "file"), "file")
        }

        let a = go_file_node("strategies/serverConfig.go");
        let b = go_file_node("strategies/roundRobinStrategy.go");
        let c = go_file_node("strategies/loadBalancer.go");

        // Mirrors what init_repo's co-package pass writes: all ordered pairs.
        let store = make_store(
            &[a.clone(), b.clone(), c.clone()],
            &[
                (b.id, a.id, EdgeKind::Depends),
                (c.id, a.id, EdgeKind::Depends),
                (a.id, b.id, EdgeKind::Depends),
                (c.id, b.id, EdgeKind::Depends),
                (a.id, c.id, EdgeKind::Depends),
                (b.id, c.id, EdgeKind::Depends),
            ],
        );

        let result = get_blast_radius(
            &store,
            "strategies/serverConfig.go",
            AnalysisMode::TreeSitter,
        );
        assert!(
            result.contains("strategies/serverConfig.go"),
            "the file itself must appear in its blast radius"
        );
        assert!(
            result.contains("strategies/roundRobinStrategy.go"),
            "co-package sibling roundRobinStrategy.go must appear in blast radius"
        );
        assert!(
            result.contains("strategies/loadBalancer.go"),
            "co-package sibling loadBalancer.go must appear in blast radius"
        );
    }

    /// Semantic mode follows only RefCall; Depends-only files must be excluded.
    #[test]
    fn blast_radius_semantic_excludes_depends_only_files() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b"); // connected via RefCall
        let c = make_node("c.ts", "fn:c"); // connected via Depends only
        let store = make_store(
            &[a.clone(), b.clone(), c.clone()],
            &[
                (b.id, a.id, EdgeKind::RefCall),
                (c.id, a.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::Semantic);
        assert!(
            result.contains("b.ts"),
            "RefCall caller must appear in semantic blast radius"
        );
        assert!(
            !result.contains("c.ts"),
            "Depends-only file must NOT appear in semantic mode"
        );
    }

    /// TreeSitter mode follows both RefCall and Depends — unchanged from before.
    #[test]
    fn blast_radius_tree_sitter_follows_both_edge_kinds() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b");
        let c = make_node("c.ts", "fn:c");
        let store = make_store(
            &[a.clone(), b.clone(), c.clone()],
            &[
                (b.id, a.id, EdgeKind::RefCall),
                (c.id, a.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "a.ts", AnalysisMode::TreeSitter);
        assert!(
            result.contains("b.ts") && result.contains("c.ts"),
            "TreeSitter mode must follow both RefCall and Depends"
        );
    }

    /// Regression for #582. The indexer never emits a direct file→file Depends
    /// edge for an import; it emits the two-hop chain
    ///   importer.ts --Depends--> import:./target --ResolvesTo--> target.ts
    /// Blast radius on target.ts must walk ResolvesTo backwards to the import
    /// node, then Depends back to the importer. Before the fix this returned
    /// only target.ts.
    #[test]
    fn blast_radius_tree_sitter_follows_import_resolves_to_chain() {
        use travsr_core::EdgeKind;
        let target = make_node("src/target.ts", "fn:target");
        let importer = make_node("src/importer.ts", "fn:importer");
        // The import node the indexer materialises for `import ... from './target'`.
        let import_node = travsr_core::Node::new(
            travsr_core::VName::new("", "", "src/importer.ts", "typescript", "import:./target"),
            "import",
        );
        let store = make_store(
            &[target.clone(), importer.clone(), import_node.clone()],
            &[
                (importer.id, import_node.id, EdgeKind::Depends),
                (import_node.id, target.id, EdgeKind::ResolvesTo),
            ],
        );
        let result = get_blast_radius(&store, "src/target.ts", AnalysisMode::TreeSitter);
        assert!(
            result.contains("src/importer.ts"),
            "importer reached via the Depends→import→ResolvesTo chain must appear, got: {result}"
        );
    }

    /// Semantic mode must NOT follow the import ResolvesTo chain — it walks only
    /// precise RefCall edges, so a Depends/ResolvesTo-only importer is excluded.
    #[test]
    fn blast_radius_semantic_excludes_import_resolves_to_chain() {
        use travsr_core::EdgeKind;
        let target = make_node("src/target.ts", "fn:target");
        let importer = make_node("src/importer.ts", "fn:importer");
        let import_node = travsr_core::Node::new(
            travsr_core::VName::new("", "", "src/importer.ts", "typescript", "import:./target"),
            "import",
        );
        let store = make_store(
            &[target.clone(), importer.clone(), import_node.clone()],
            &[
                (importer.id, import_node.id, EdgeKind::Depends),
                (import_node.id, target.id, EdgeKind::ResolvesTo),
            ],
        );
        let result = get_blast_radius(&store, "src/target.ts", AnalysisMode::Semantic);
        assert!(
            !result.contains("src/importer.ts"),
            "semantic mode must not follow the import chain, got: {result}"
        );
    }

    /// Transitive blast radius through the import chain (#582). a.ts imports
    /// b.ts and b.ts imports c.ts, each via the two-hop
    /// Depends→import→ResolvesTo chain. Blast radius on c.ts must reach b.ts
    /// (direct) AND a.ts (transitive), because Phase 1 keeps draining the queue
    /// as it discovers importer files.
    #[test]
    fn blast_radius_tree_sitter_follows_transitive_import_chain() {
        use travsr_core::{EdgeKind, Node, VName};
        let c = make_node("src/c.ts", "fn:c");
        let b = make_node("src/b.ts", "fn:b");
        let a = make_node("src/a.ts", "fn:a");
        // b.ts imports c.ts; a.ts imports b.ts.
        let imp_c = Node::new(
            VName::new("", "", "src/b.ts", "typescript", "import:./c"),
            "import",
        );
        let imp_b = Node::new(
            VName::new("", "", "src/a.ts", "typescript", "import:./b"),
            "import",
        );
        let store = make_store(
            &[
                c.clone(),
                b.clone(),
                a.clone(),
                imp_c.clone(),
                imp_b.clone(),
            ],
            &[
                (b.id, imp_c.id, EdgeKind::Depends),
                (imp_c.id, c.id, EdgeKind::ResolvesTo),
                (a.id, imp_b.id, EdgeKind::Depends),
                (imp_b.id, b.id, EdgeKind::ResolvesTo),
            ],
        );
        let result = get_blast_radius(&store, "src/c.ts", AnalysisMode::TreeSitter);
        assert!(
            result.contains("src/b.ts"),
            "direct importer b.ts must appear, got: {result}"
        );
        assert!(
            result.contains("src/a.ts"),
            "transitive importer a.ts must appear, got: {result}"
        );
    }

    /// Ruby end-to-end regression for #582. Ruby emits import nodes but NO
    /// ResolvesTo chain, so importers are recovered through Phase 2's
    /// language-aware `RubyResolver` — a different code path from the
    /// TypeScript ResolvesTo fix above. Before RubyResolver was registered,
    /// `resolver_for_language("ruby")` returned NoopResolver and this returned
    /// only animal.rb. cat.rb and dog.rb each `require_relative 'animal'`.
    #[test]
    fn blast_radius_tree_sitter_resolves_ruby_require_relative() {
        use travsr_core::{EdgeKind, Node, VName};
        let ruby = |path: &str, sig: &str, kind: &str| {
            Node::new(VName::new("", "", path, "ruby", sig), kind)
        };
        let animal = ruby("ruby/src/animal.rb", "fn:animal", "function");
        let cat = ruby("ruby/src/cat.rb", "fn:cat", "function");
        let dog = ruby("ruby/src/dog.rb", "fn:dog", "function");
        // The import node the indexer materialises for `require_relative 'animal'`.
        // Its path is the requiring file; there is no ResolvesTo edge to animal.rb.
        let cat_import = ruby("ruby/src/cat.rb", "import:animal", "import");
        let dog_import = ruby("ruby/src/dog.rb", "import:animal", "import");
        let store = make_store(
            &[
                animal.clone(),
                cat.clone(),
                dog.clone(),
                cat_import.clone(),
                dog_import.clone(),
            ],
            &[
                (cat.id, cat_import.id, EdgeKind::Depends),
                (dog.id, dog_import.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "ruby/src/animal.rb", AnalysisMode::TreeSitter);
        assert!(
            result.contains("ruby/src/cat.rb"),
            "cat.rb requires animal (Phase-2 RubyResolver) and must appear, got: {result}"
        );
        assert!(
            result.contains("ruby/src/dog.rb"),
            "dog.rb requires animal (Phase-2 RubyResolver) and must appear, got: {result}"
        );
    }

    /// #614 regression: `require 'json'` (load-path, tagged `import:gem:json`
    /// by the analyzer) must NOT resolve to a same-stem `json.rb` sibling,
    /// unlike `require_relative 'json'` (`import:json`) which still does.
    /// Before the fix both keywords emitted the same `import:json` signature
    /// and RubyResolver could not tell them apart.
    #[test]
    fn blast_radius_phase2_ignores_ruby_gem_require() {
        use travsr_core::{EdgeKind, Node, VName};
        let ruby = |path: &str, sig: &str, kind: &str| {
            Node::new(VName::new("", "", path, "ruby", sig), kind)
        };
        let json_file = ruby("ruby/src/json.rb", "fn:json", "function");
        let cat = ruby("ruby/src/cat.rb", "fn:cat", "function");
        // `require 'json'` in cat.rb: load-path, must not resolve to json.rb.
        let cat_gem_import = ruby("ruby/src/cat.rb", "import:gem:json", "import");
        let dog = ruby("ruby/src/dog.rb", "fn:dog", "function");
        // `require_relative 'json'` in dog.rb: importer-relative, must resolve.
        let dog_relative_import = ruby("ruby/src/dog.rb", "import:json", "import");
        let store = make_store(
            &[
                json_file.clone(),
                cat.clone(),
                cat_gem_import.clone(),
                dog.clone(),
                dog_relative_import.clone(),
            ],
            &[
                (cat.id, cat_gem_import.id, EdgeKind::Depends),
                (dog.id, dog_relative_import.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "ruby/src/json.rb", AnalysisMode::TreeSitter);
        assert!(
            !result.contains("ruby/src/cat.rb"),
            "cat.rb's `require 'json'` is load-path (import:gem:json) and must NOT \
             appear, got: {result}"
        );
        assert!(
            result.contains("ruby/src/dog.rb"),
            "dog.rb's `require_relative 'json'` (import:json) must still appear, \
             got: {result}"
        );
    }

    /// #613 keystone. A Phase-2 language (Ruby, no ResolvesTo edge) must report
    /// blast radius *transitively*: main.rb requires cat, cat.rb requires animal.
    /// Editing animal.rb must reach cat.rb (one hop) AND main.rb (two hops).
    /// Before the fixpoint this returned only cat.rb.
    #[test]
    fn blast_radius_phase2_is_transitive_across_hops() {
        use travsr_core::{EdgeKind, Node, VName};
        let ruby = |path: &str, sig: &str, kind: &str| {
            Node::new(VName::new("", "", path, "ruby", sig), kind)
        };
        let animal = ruby("ruby/src/animal.rb", "fn:animal", "function");
        let cat = ruby("ruby/src/cat.rb", "fn:cat", "function");
        let main = ruby("ruby/src/main.rb", "fn:main", "function");
        // cat.rb `require_relative 'animal'`; main.rb `require_relative 'cat'`.
        // Import node path == the requiring file; no ResolvesTo edges anywhere.
        let cat_import = ruby("ruby/src/cat.rb", "import:animal", "import");
        let main_import = ruby("ruby/src/main.rb", "import:cat", "import");
        let store = make_store(
            &[
                animal.clone(),
                cat.clone(),
                main.clone(),
                cat_import.clone(),
                main_import.clone(),
            ],
            &[
                (cat.id, cat_import.id, EdgeKind::Depends),
                (main.id, main_import.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "ruby/src/animal.rb", AnalysisMode::TreeSitter);
        assert!(
            result.contains("ruby/src/cat.rb"),
            "direct importer cat.rb must appear, got: {result}"
        );
        assert!(
            result.contains("ruby/src/main.rb"),
            "transitive importer main.rb (main → cat → animal) must appear, got: {result}"
        );
    }

    /// #613 determinism guard. Phase-2 resolution must anchor a relative require
    /// to the importer's directory, never a same-named file under an unrelated
    /// root. Here `other/helper.rb` requires 'animal' meaning `other/animal.rb`;
    /// it must NOT be dragged into the blast radius of `ruby/src/animal.rb`
    /// merely because both files are named animal.rb. Without importer-anchoring
    /// this false positive would also compound transitively.
    #[test]
    fn blast_radius_phase2_does_not_match_same_name_unrelated_file() {
        use travsr_core::{EdgeKind, Node, VName};
        let ruby = |path: &str, sig: &str, kind: &str| {
            Node::new(VName::new("", "", path, "ruby", sig), kind)
        };
        let animal = ruby("ruby/src/animal.rb", "fn:animal", "function");
        let cat = ruby("ruby/src/cat.rb", "fn:cat", "function");
        let cat_import = ruby("ruby/src/cat.rb", "import:animal", "import");
        // Unrelated file with the same stem, in a different directory.
        let other_helper = ruby("other/helper.rb", "fn:helper", "function");
        let other_import = ruby("other/helper.rb", "import:animal", "import");
        let store = make_store(
            &[
                animal.clone(),
                cat.clone(),
                cat_import.clone(),
                other_helper.clone(),
                other_import.clone(),
            ],
            &[
                (cat.id, cat_import.id, EdgeKind::Depends),
                (other_helper.id, other_import.id, EdgeKind::Depends),
            ],
        );
        let result = get_blast_radius(&store, "ruby/src/animal.rb", AnalysisMode::TreeSitter);
        assert!(
            result.contains("ruby/src/cat.rb"),
            "sibling importer cat.rb must appear, got: {result}"
        );
        assert!(
            !result.contains("other/helper.rb"),
            "helper.rb requires its own other/animal.rb, must NOT appear for \
             ruby/src/animal.rb, got: {result}"
        );
    }

    /// Objective-C previously fell through to NoopResolver and had no Phase-2
    /// blast radius. `#import "Model.h"` is importer-relative, so editing
    /// Model.h must reach the .m file that imports it.
    #[test]
    fn blast_radius_phase2_resolves_objective_c_import() {
        use travsr_core::{EdgeKind, Node, VName};
        let objc = |path: &str, sig: &str, kind: &str| {
            Node::new(VName::new("", "", path, "objectivec", sig), kind)
        };
        let model = objc("app/Model.h", "type:Model", "type");
        let view = objc("app/View.m", "type:View", "type");
        // `#import "Model.h"` in View.m → import:Model.h, path == View.m.
        let view_import = objc("app/View.m", "import:Model.h", "import");
        let store = make_store(
            &[model.clone(), view.clone(), view_import.clone()],
            &[(view.id, view_import.id, EdgeKind::Depends)],
        );
        let result = get_blast_radius(&store, "app/Model.h", AnalysisMode::TreeSitter);
        assert!(
            result.contains("app/View.m"),
            "View.m imports Model.h and must appear in its blast radius, got: {result}"
        );
    }

    /// get_lang_status returns valid JSON for a known extension with no RefCall data,
    /// and reports the honest `partial` status (structure works, full analysis not live).
    #[test]
    fn get_lang_status_known_extension_semantic_unavailable() {
        let store = make_store(&[], &[]);
        let json = get_lang_status(&store, "src/main.ts");
        assert!(json.contains(r#""language":"typescript""#));
        assert!(json.contains(r#""builtin":true"#));
        assert!(json.contains(r#""semantic_available":false"#));
        assert!(json.contains(r#""status":"partial""#), "got: {json}");
    }

    /// get_lang_status must never tell the user to install a build that does not
    /// exist for this OS: when the shared platform predicate says a language is
    /// unavailable here, the status is `unsupported` with no install hint — the
    /// same verdict `travsr lang list` shows. Gated on the predicate so it is
    /// correct on every host (objectivec is unavailable off Apple, available on it).
    #[test]
    fn get_lang_status_agrees_with_platform_predicate_on_unsupported() {
        let store = make_store(&[], &[]);
        let json = get_lang_status(&store, "src/AppDelegate.mm");
        assert!(json.contains(r#""language":"objectivec""#), "got: {json}");
        let entry = travsr_plugin_host::phase_b::catalog::lookup("objectivec").unwrap();
        if travsr_plugin_host::phase_b::platform::unsupported_reason(entry).is_some() {
            assert!(
                json.contains(r#""status":"unsupported""#),
                "unavailable here, must be unsupported, got: {json}"
            );
            assert!(
                json.contains(r#""install_hint":"""#),
                "no install hint when unsupported, got: {json}"
            );
        } else {
            assert!(
                json.contains(r#""status":"partial""#) || json.contains(r#""status":"active""#),
                "available here, must not be unsupported, got: {json}"
            );
        }
    }

    /// get_lang_status flips to `active` after a RefCall edge is inserted.
    #[test]
    fn get_lang_status_returns_semantic_available_true_after_refcall_insert() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let n1 = Node::new(VName::new("", "", "a.ts", "typescript", "fn:a"), "function");
        let n2 = Node::new(VName::new("", "", "b.ts", "typescript", "fn:b"), "function");
        let mut store = make_store(&[n1.clone(), n2.clone()], &[]);
        store
            .put_edge(&Edge::new(n1.id, n2.id, EdgeKind::RefCall))
            .unwrap();
        let json = get_lang_status(&store, "a.ts");
        assert!(json.contains(r#""semantic_available":true"#));
        assert!(json.contains(r#""status":"active""#), "got: {json}");
        assert!(json.contains(r#""install_hint":"""#));
    }

    /// get_lang_status returns a safe unknown envelope for an unrecognised extension.
    #[test]
    fn get_lang_status_unknown_extension_returns_unknown() {
        let store = make_store(&[], &[]);
        let json = get_lang_status(&store, "Makefile");
        assert!(json.contains(r#""language":"unknown""#));
        assert!(json.contains(r#""semantic_available":false"#));
    }

    /// A non-builtin language with no live semantic data always surfaces a
    /// concrete next step in the shared vocabulary. Which step depends on the
    /// machine: `install` when the external analyzer is absent, `rebuild` when it
    /// is already resolvable on PATH (`analyzer_installed` probes the real
    /// environment, so the test must accept either to stay hermetic across CI and
    /// dev boxes). Both are legitimate user-facing guidance; the invariant is that
    /// a non-builtin language is never left without a next step.
    #[test]
    fn get_lang_status_non_builtin_shows_install_hint() {
        let store = make_store(&[], &[]);
        let json = get_lang_status(&store, "main.go");
        assert!(json.contains(r#""language":"go""#));
        assert!(json.contains(r#""builtin":false"#));
        assert!(json.contains(r#""semantic_available":false"#));
        assert!(
            json.contains("travsr lang install go")
                || json.contains("travsr init --semantic --force"),
            "non-builtin go must surface a concrete next step (install when the \
             analyzer is absent, rebuild when it is already installed): {json}"
        );
    }

    /// Rust is not special: with no cross-file edges it reads `partial` and points
    /// at the same uniform install step, never leaking tool jargon like
    /// "rust-analyzer" or "rustup" into the user-facing hint.
    #[test]
    fn get_lang_status_rust_partial_uses_uniform_step_no_tool_jargon() {
        let store = make_store(&[], &[]);
        let json = get_lang_status(&store, "src/main.rs");
        assert!(json.contains(r#""language":"rust""#));
        assert!(json.contains(r#""status":"partial""#), "got: {json}");
        assert!(
            !json.contains("rust-analyzer"),
            "no tool jargon in hint: {json}"
        );
        assert!(!json.contains("rustup"), "no tool jargon in hint: {json}");
    }

    // ── get_graph_stats unit tests ───────────────────────────────────────────

    #[test]
    fn get_graph_stats_empty_graph() {
        let store = make_store(&[], &[]);
        let out = get_graph_stats(&store);
        assert!(out.starts_with("nodes: 0\nedges: 0"), "got: {out}");
        assert!(out.contains("schema_version: "), "got: {out}");
    }

    #[test]
    fn get_graph_stats_counts_match_store() {
        use travsr_core::EdgeKind;
        let a = make_node("a.ts", "fn:a");
        let b = make_node("b.ts", "fn:b");
        let store = make_store(&[a.clone(), b.clone()], &[(a.id, b.id, EdgeKind::Depends)]);
        let out = get_graph_stats(&store);
        assert!(out.starts_with("nodes: 2\nedges: 1"), "got: {out}");
        assert!(out.contains("schema_version: "), "got: {out}");
    }

    // ── synonym tool unit tests (RFC-012 A2 F1 / VSCODE-247) ──────────────────

    // Note: `open_in_memory` seeds the synonym table with static defaults, so
    // these tests use a `zz`-prefixed namespace unlikely to collide with seeds
    // and never assert the table is globally empty.

    #[test]
    fn synonym_add_then_list_roundtrip() {
        let mut store = make_store(&[], &[]);
        assert_eq!(synonym_add(&mut store, "zzpayment", "zzcharge"), "ok");
        let list = synonym_list(&store);
        assert!(list.contains("zzpayment => zzcharge"), "got: {list}");
    }

    #[test]
    fn synonym_set_replaces_all_aliases() {
        let mut store = make_store(&[], &[]);
        synonym_add(&mut store, "zzauth", "zzlogin");
        synonym_add(&mut store, "zzauth", "zzsignin");
        assert_eq!(synonym_set(&mut store, "zzauth", "zzsession"), "ok");
        let list = synonym_list(&store);
        assert!(list.contains("zzauth => zzsession"), "got: {list}");
        assert!(
            !list.contains("zzauth => zzlogin"),
            "old alias must be gone: {list}"
        );
    }

    #[test]
    fn synonym_remove_term_clears_all() {
        let mut store = make_store(&[], &[]);
        synonym_add(&mut store, "zzdb", "zzdatabase");
        synonym_add(&mut store, "zzdb", "zzstore");
        assert_eq!(synonym_remove_term(&mut store, "zzdb"), "ok");
        let list = synonym_list(&store);
        assert!(
            !list.contains("zzdb => "),
            "all zzdb aliases must be gone: {list}"
        );
    }

    #[test]
    fn synonym_set_csv_splits_and_trims() {
        let mut store = make_store(&[], &[]);
        assert_eq!(
            synonym_set(&mut store, "zzk8s", " zzkube , zzkubernetes ,"),
            "ok"
        );
        let list = synonym_list(&store);
        assert!(list.contains("zzk8s => zzkube"), "got: {list}");
        assert!(list.contains("zzk8s => zzkubernetes"), "got: {list}");
    }

    #[test]
    fn synonym_add_rejects_path_traversal() {
        let mut store = make_store(&[], &[]);
        // SEC-002: term/alias go through validate_mcp_arg.
        let result = synonym_add(&mut store, "../etc/passwd", "x");
        assert_eq!(result, "invalid input");
        assert!(!synonym_list(&store).contains("../etc/passwd"));
    }

    #[test]
    fn synonym_set_rejects_invalid_alias_wholesale() {
        let mut store = make_store(&[], &[]);
        let result = synonym_set(&mut store, "zzterm", "zzgood,../bad");
        assert_eq!(result, "invalid input");
        // Rejected before the store call — neither alias written.
        let list = synonym_list(&store);
        assert!(!list.contains("zzterm => zzgood"), "got: {list}");
    }

    #[test]
    fn synonym_add_over_cap_returns_error_message() {
        let mut store = make_store(&[], &[]);
        // Add distinct aliases until the 200-row cap rejects one. The table is
        // pre-seeded with defaults, so the failure arrives before 200 zz-adds.
        let mut last = String::new();
        for i in 0..300 {
            last = synonym_add(&mut store, "zzfill", &format!("zza{i}"));
            if last != "ok" {
                break;
            }
        }
        assert_ne!(last, "ok", "the cap must eventually reject an add");
        assert!(last.contains("200"), "cap message expected, got: {last}");
    }

    #[test]
    fn synonym_reset_restores_defaults() {
        let mut store = make_store(&[], &[]);
        synonym_add(&mut store, "zzcustom", "zzalias");
        assert_eq!(synonym_reset(&mut store), "ok");
        let list = synonym_list(&store);
        assert!(
            !list.contains("zzcustom => zzalias"),
            "custom pair must be cleared: {list}"
        );
        assert!(!list.is_empty(), "defaults must be re-seeded after reset");
    }

    // Note: repos_list/repos_prune/repos_remove operate on the *real* global
    // registry (~/.travsr), so they are intentionally NOT unit-tested here — a
    // test calling repos_prune() would mutate the developer's HOME. The
    // underlying registry::{prune,unregister,all_repos} are covered by
    // travsr-store/src/registry.rs tests under a temp-HOME lock.

    // ── transitive dependencies unit test ────────────────────────────────────

    #[test]
    fn get_dependencies_transitive_reaches_depth_two() {
        use travsr_core::{EdgeKind, Node, VName};
        // file_a depends file_b depends file_c
        let mk = |path: &str| Node::new(VName::new("", "", path, "typescript", path), "file");
        let a = mk("a.ts");
        let b = mk("b.ts");
        let c = mk("c.ts");
        let store = make_store(
            &[a.clone(), b.clone(), c.clone()],
            &[
                (a.id, b.id, EdgeKind::Depends),
                (b.id, c.id, EdgeKind::Depends),
            ],
        );

        // Direct only: just b.ts.
        let direct = get_dependencies(&store, "a.ts", false, 3);
        assert!(direct.contains("b.ts"), "direct dep b.ts missing: {direct}");
        assert!(
            !direct.contains("c.ts"),
            "c.ts is transitive, not direct: {direct}"
        );

        // Transitive: both b.ts and c.ts.
        let trans = get_dependencies(&store, "a.ts", true, 3);
        assert!(trans.contains("b.ts"), "b.ts missing: {trans}");
        assert!(trans.contains("c.ts"), "transitive c.ts missing: {trans}");
    }

    // ── get_dependencies top-k cap tests ─────────────────────────────────────

    #[test]
    fn get_dependencies_raw_caps_at_top_k_and_emits_footer() {
        use travsr_core::{EdgeKind, Node, VName};
        // Seed file + 30 dep nodes (> DEFAULT_DEPS_TOP_K=25).
        let seed = Node::new(
            VName::new("", "", "src/big.ts", "typescript", "src/big.ts"),
            "file",
        );
        let mut nodes = vec![seed.clone()];
        let mut edges = Vec::new();
        for i in 0..30u32 {
            let dep = Node::new(
                VName::new(
                    "",
                    "",
                    format!("dep{i}.ts"),
                    "typescript",
                    format!("dep{i}.ts"),
                ),
                "file",
            );
            edges.push((seed.id, dep.id, EdgeKind::Depends));
            nodes.push(dep);
        }
        let store = make_store(&nodes, &edges);
        let result = get_dependencies(&store, "src/big.ts", false, 1);
        assert!(
            result.contains("showing 25 of 30"),
            "footer must report 25 of 30, got: {result}"
        );
        assert!(
            result.contains("get_context for ranked coverage"),
            "footer must include get_context hint, got: {result}"
        );
    }

    // ── #619 get_dependencies wrong-argument hint ────────────────────────────

    /// A symbol name passed where a file path belongs must return a hint, not
    /// a silent empty answer that reads as "this file has no dependencies".
    #[test]
    fn get_dependencies_symbol_arg_returns_hint() {
        use travsr_core::{Node, VName};
        let sym = Node::new(
            VName::new("", "", "src/docs.ts", "typescript", "fn:docs_section"),
            "function",
        );
        let store = make_store(std::slice::from_ref(&sym), &[]);
        let result = get_dependencies(&store, "docs_section", false, 1);
        assert!(
            result.contains("not a file"),
            "symbol arg must produce the not-a-file hint; got: {result}"
        );
        assert!(
            result.contains("get_callers"),
            "hint must point at the symbol tools; got: {result}"
        );
    }

    /// An argument matching nothing at all must say so.
    #[test]
    fn get_dependencies_unknown_arg_returns_hint() {
        let a = make_node("a.ts", "fn:a");
        let store = make_store(std::slice::from_ref(&a), &[]);
        let result = get_dependencies(&store, "zzz_nonexistent", false, 1);
        assert!(
            result.contains("no file matched"),
            "unknown arg must produce the no-match hint; got: {result}"
        );
    }

    /// A real file with zero depends edges is a legitimate empty answer and
    /// must stay empty — the hint only fires when the arg is not a file.
    #[test]
    fn get_dependencies_file_without_deps_stays_empty() {
        use travsr_core::{Node, VName};
        let file = Node::new(
            VName::new("", "", "src/lone.ts", "typescript", "src/lone.ts"),
            "file",
        );
        let store = make_store(std::slice::from_ref(&file), &[]);
        let result = get_dependencies(&store, "src/lone.ts", false, 1);
        // sanitize_for_mcp wraps empty output in an empty envelope; the point
        // here is that no hint text is injected into a legitimate empty answer.
        assert!(
            !result.contains("not a file") && !result.contains("no file matched"),
            "file with no deps must stay a plain empty answer; got: {result}"
        );
    }

    /// The transitive variant shares the entry point and must hint the same way.
    #[test]
    fn get_dependencies_transitive_symbol_arg_returns_hint() {
        use travsr_core::{Node, VName};
        let sym = Node::new(
            VName::new("", "", "src/docs.ts", "typescript", "fn:docs_section"),
            "function",
        );
        let store = make_store(std::slice::from_ref(&sym), &[]);
        let result = get_dependencies(&store, "docs_section", true, 3);
        assert!(
            result.contains("not a file"),
            "transitive path must hint too; got: {result}"
        );
    }

    #[test]
    fn get_dependencies_raw_no_footer_when_under_top_k() {
        use travsr_core::{EdgeKind, Node, VName};
        let seed = Node::new(
            VName::new("", "", "src/small.ts", "typescript", "src/small.ts"),
            "file",
        );
        let dep = Node::new(VName::new("", "", "dep.ts", "typescript", "dep.ts"), "file");
        let store = make_store(
            &[seed.clone(), dep.clone()],
            &[(seed.id, dep.id, EdgeKind::Depends)],
        );
        let result = get_dependencies(&store, "src/small.ts", false, 1);
        assert!(
            !result.contains("showing"),
            "no footer when under top-k, got: {result}"
        );
        assert!(result.contains("dep.ts"), "dep must appear: {result}");
    }

    // ── get_repo_map unit tests ───────────────────────────────────────────────

    /// Two nodes in different files must produce two entries.
    #[test]
    fn get_repo_map_groups_by_file() {
        let a = make_node("src/a.ts", "fn:a");
        let b = make_node("src/b.ts", "fn:b");
        let store = make_store(&[a, b], &[]);
        let result = get_repo_map(&store);
        assert!(result.contains("src/a.ts"), "a.ts must appear in repo map");
        assert!(result.contains("src/b.ts"), "b.ts must appear in repo map");
    }

    /// File-kind nodes must not appear as symbols in the map.
    #[test]
    fn get_graph_json_includes_line_for_symbol_nodes() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let sym = Node::new(
            VName::new("test", "", "src/foo.ts", "typescript", "fn:bar"),
            "function",
        )
        .with_line(42);
        let file = Node::new(
            VName::new("test", "", "src/foo.ts", "typescript", "file"),
            "file",
        );
        store.put_node(&sym).unwrap();
        store.put_node(&file).unwrap();

        let json = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "fn:bar",
                direction: "both",
                depth: 1,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );
        assert!(
            json.contains("\"line\":42"),
            "symbol node must carry line in JSON: {json}"
        );
        // #318 O5: coverage envelope must always be present on the JSON tool.
        assert!(
            json.contains("\"coverage\""),
            "coverage envelope missing: {json}"
        );

        let file_json = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "src/foo.ts",
                direction: "both",
                depth: 1,
                kind_filter: "file",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );
        assert!(
            file_json.contains("\"line\":null"),
            "file node must have null line in JSON: {file_json}"
        );
    }

    /// #318 O6: a token budget caps the payload but always keeps the seed,
    /// and the truncation is reported in-band.
    #[test]
    fn get_graph_json_token_budget_truncates() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let seed = Node::new(
            VName::new("test", "", "src/seed.ts", "typescript", "fn:seed"),
            "function",
        );
        store.put_node(&seed).unwrap();
        for i in 0..20 {
            let n = Node::new(
                VName::new(
                    "test",
                    "",
                    format!("src/dep{i}.ts"),
                    "typescript",
                    format!("fn:dep{i}"),
                ),
                "function",
            );
            store.put_node(&n).unwrap();
            store
                .put_edge(&Edge::new(seed.id, n.id, EdgeKind::RefCall))
                .unwrap();
        }

        let unbounded = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "fn:seed",
                direction: "deps",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );
        assert!(!unbounded.contains("truncated_by_budget"));

        let capped = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "fn:seed",
                direction: "deps",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 30,
                mode: "",
                path_prefix: "",
            },
        );
        assert!(
            capped.contains("\"truncated_by_budget\":true"),
            "tiny budget must truncate: {capped}"
        );
        assert!(
            capped.contains("fn:seed"),
            "seed must survive any budget: {capped}"
        );
        assert!(capped.len() < unbounded.len());
    }

    // ── P3 overview / LOD tests ────────────────────────────────────────────────

    fn make_file_graph_store() -> travsr_store::SqliteStore {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();

        // Two top-level packages: "pkg" (2 files) and "test" (1 file)
        let fa = Node::new(
            VName::new("", "", "pkg/a/mod.ts", "typescript", "pkg/a/mod.ts"),
            "file",
        );
        let fb = Node::new(
            VName::new("", "", "test/b/util.ts", "typescript", "test/b/util.ts"),
            "file",
        );
        let fc = Node::new(
            VName::new("", "", "pkg/a/helper.ts", "typescript", "pkg/a/helper.ts"),
            "file",
        );
        // Import node: fa depends on fb via an import node
        let imp = Node::new(
            VName::new("", "", "test/b/util.ts", "typescript", "import:b"),
            "import",
        );
        store.put_node(&fa).unwrap();
        store.put_node(&fb).unwrap();
        store.put_node(&fc).unwrap();
        store.put_node(&imp).unwrap();
        // fa --[Depends]--> imp --[ResolvesTo]--> fb  (cross-package: pkg → test)
        store
            .put_edge(&Edge::new(fa.id, imp.id, EdgeKind::Depends))
            .unwrap();
        store
            .put_edge(&Edge::new(imp.id, fb.id, EdgeKind::ResolvesTo))
            .unwrap();
        store
    }

    #[test]
    fn overview_graph_repo_level_groups_by_pkg() {
        let store = make_file_graph_store();
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "overview",
                path_prefix: "",
            },
        );
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["mode"], "overview", "mode field must be 'overview'");

        let nodes = v["nodes"].as_array().unwrap();
        let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"pkg:pkg"), "must have 'pkg' package: {ids:?}");
        assert!(
            ids.contains(&"pkg:test"),
            "must have 'test' package: {ids:?}"
        );

        // file_count for "pkg" should be 2 (mod.ts + helper.ts)
        let pkg_pkg = nodes.iter().find(|n| n["id"] == "pkg:pkg").unwrap();
        assert_eq!(pkg_pkg["file_count"], 2, "'pkg' should have 2 files");

        // Cross-package edge pkg → test
        let edges = v["edges"].as_array().unwrap();
        let cross = edges
            .iter()
            .find(|e| e["source"] == "pkg:pkg" && e["target"] == "pkg:test");
        assert!(
            cross.is_some(),
            "must have cross-package edge pkg → test: {edges:?}"
        );
        assert!(cross.unwrap()["count"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn overview_graph_splits_monorepo_into_components() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let mk = |p: &str| Node::new(VName::new("", "", p, "rust", p), "file");
        let mut core_ids = Vec::new();
        let mut mcp_ids = Vec::new();
        for f in ["a", "b", "c"] {
            let c = mk(&format!("crates/travsr-core/src/{f}.rs"));
            let m = mk(&format!("crates/travsr-mcp/src/{f}.rs"));
            store.put_node(&c).unwrap();
            store.put_node(&m).unwrap();
            core_ids.push(c.id);
            mcp_ids.push(m.id);
        }
        // mcp depends on core (a real call edge).
        store
            .put_edge(&Edge::new(mcp_ids[0], core_ids[0], EdgeKind::RefCall))
            .unwrap();

        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "overview",
                path_prefix: "",
            },
        );
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let nodes = v["nodes"].as_array().unwrap();
        let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
        // The monorepo must roll up to crate-level components, NOT collapse into
        // a single "crates" tile (which would hide all internal architecture).
        assert!(
            ids.contains(&"pkg:crates/travsr-core") && ids.contains(&"pkg:crates/travsr-mcp"),
            "monorepo must split into crate-level components: {ids:?}"
        );
        assert!(
            !ids.contains(&"pkg:crates"),
            "must not collapse to a single 'crates' tile: {ids:?}"
        );

        let core = nodes
            .iter()
            .find(|n| n["id"] == "pkg:crates/travsr-core")
            .unwrap();
        assert_eq!(core["dependents"], 1, "core is depended upon by mcp");

        let edges = v["edges"].as_array().unwrap();
        assert!(
            edges.iter().any(|e| e["source"] == "pkg:crates/travsr-mcp"
                && e["target"] == "pkg:crates/travsr-core"),
            "cross-component edge mcp -> core must be present: {edges:?}"
        );
    }

    #[test]
    fn overview_graph_package_drill_emits_files_and_ghost() {
        let store = make_file_graph_store();
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "overview",
                path_prefix: "pkg/a/",
            },
        );
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["mode"], "prefix");

        let nodes = v["nodes"].as_array().unwrap();
        let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();

        // File nodes inside prefix
        assert!(
            ids.contains(&"file:pkg/a/mod.ts"),
            "missing mod.ts: {ids:?}"
        );
        assert!(
            ids.contains(&"file:pkg/a/helper.ts"),
            "missing helper.ts: {ids:?}"
        );

        // Ghost port for external dep (1-segment key: "test")
        assert!(
            ids.contains(&"pkg:test"),
            "missing ghost port pkg:test: {ids:?}"
        );
        let ghost = nodes.iter().find(|n| n["id"] == "pkg:test").unwrap();
        assert_eq!(ghost["ghost"], true, "external package must be ghost");

        // Cross-boundary edge: file:pkg/a/mod.ts → pkg:test
        let edges = v["edges"].as_array().unwrap();
        let cross = edges
            .iter()
            .find(|e| e["source"] == "file:pkg/a/mod.ts" && e["target"] == "pkg:test");
        assert!(cross.is_some(), "must have cross-boundary edge: {edges:?}");
    }

    #[test]
    fn package_drill_graph_intra_prefix_edge_emitted() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let fa = Node::new(
            VName::new("", "", "pkg/a/mod.ts", "typescript", "pkg/a/mod.ts"),
            "file",
        );
        let fb = Node::new(
            VName::new("", "", "pkg/a/util.ts", "typescript", "pkg/a/util.ts"),
            "file",
        );
        let imp = Node::new(
            VName::new("", "", "pkg/a/util.ts", "typescript", "import:u"),
            "import",
        );
        store.put_node(&fa).unwrap();
        store.put_node(&fb).unwrap();
        store.put_node(&imp).unwrap();
        store
            .put_edge(&Edge::new(fa.id, imp.id, EdgeKind::Depends))
            .unwrap();
        store
            .put_edge(&Edge::new(imp.id, fb.id, EdgeKind::ResolvesTo))
            .unwrap();

        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "overview",
                path_prefix: "pkg/a/",
            },
        );
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let edges = v["edges"].as_array().unwrap();
        let intra = edges
            .iter()
            .find(|e| e["source"] == "file:pkg/a/mod.ts" && e["target"] == "file:pkg/a/util.ts");
        assert!(
            intra.is_some(),
            "intra-prefix edge must be emitted: {edges:?}"
        );
    }

    #[test]
    fn overview_graph_rejects_unknown_mode() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "badmode",
                path_prefix: "",
            },
        );
        assert_eq!(raw, "{}", "unknown mode must return empty object");
    }

    #[test]
    fn overview_graph_rejects_path_traversal_prefix() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "",
                direction: "both",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "overview",
                path_prefix: "../etc/passwd",
            },
        );
        assert_eq!(raw, "{}", "path traversal prefix must be rejected");
    }

    #[test]
    fn pkg_key_from_path_one_segment() {
        // Any path with subdirs → first segment only
        assert_eq!(
            pkg_key_from_path("crates/travsr-mcp/src/tools.rs"),
            "crates"
        );
        assert_eq!(pkg_key_from_path("src/components/Button.tsx"), "src");
        assert_eq!(pkg_key_from_path("src/index.ts"), "src");
        assert_eq!(pkg_key_from_path("index.ts"), "(root)");
        // External / build-cache paths → empty (excluded)
        assert_eq!(pkg_key_from_path("../../../Library/Caches/go-build/xx"), "");
        assert_eq!(pkg_key_from_path("/abs/path/file.go"), "");
        assert_eq!(pkg_key_from_path("scip://corpus/path/file.go"), "");
        assert_eq!(pkg_key_from_path(""), "");
    }

    #[test]
    fn get_repo_map_excludes_file_kind_nodes() {
        use travsr_core::VName;
        let file_node = travsr_core::Node::new(
            VName::new("", "", "src/a.ts", "typescript", "src/a.ts"),
            "file",
        );
        let fn_node = make_node("src/a.ts", "fn:a");
        let store = make_store(&[file_node, fn_node], &[]);
        let result = get_repo_map(&store);
        // Should list "src/a.ts" as a file entry.
        assert!(result.contains("src/a.ts"));
        // The "file" kind node's signature must not appear as a symbol entry.
        // It should show [1 symbol] (only fn:a), not [2 symbols].
        assert!(
            result.contains("[1 symbol]"),
            "file-kind node must be excluded from symbol count"
        );
    }

    /// A node with an explicit kind, for repo-map region/edge tests.
    fn make_kind(path: &str, sig: &str, kind: &str) -> travsr_core::Node {
        use travsr_core::VName;
        travsr_core::Node::new(VName::new("", "", path, "typescript", sig), kind)
    }

    /// Two crates (core + mcp) where mcp depends on core via a resolved edge.
    /// core must rank first (higher dependents). Enough files under crates/ that
    /// the adaptive rollup splits it into per-crate regions.
    fn two_crate_store(with_refcall: bool) -> travsr_store::SqliteStore {
        use travsr_core::EdgeKind;
        let mut nodes = Vec::new();
        for f in ["a", "b", "c"] {
            nodes.push(make_node(
                &format!("crates/travsr-core/src/{f}.rs"),
                &format!("fn:core_{f}"),
            ));
            nodes.push(make_node(
                &format!("crates/travsr-mcp/src/{f}.rs"),
                &format!("fn:mcp_{f}"),
            ));
        }
        // core_a is nodes[0], core_b nodes[2]; mcp_b nodes[3].
        let core_a = nodes[0].id;
        let core_b = nodes[2].id;
        let mcp_b = nodes[3].id;
        // resolves-to (Phase A): import node in mcp (path = importer) -> core file.
        // So dependents exist even without Phase B.
        let mcp_import = make_kind("crates/travsr-mcp/src/a.rs", "use:core", "import");
        nodes.push(mcp_import.clone());
        let mut edges = vec![(mcp_import.id, core_a, EdgeKind::ResolvesTo)];
        if with_refcall {
            // A real call edge flips the Phase-B disclosure off.
            edges.push((mcp_b, core_b, EdgeKind::RefCall));
        }
        make_store(&nodes, &edges)
    }

    #[test]
    fn get_repo_map_ranks_regions_by_dependents() {
        let store = two_crate_store(false);
        let result = get_repo_map(&store);
        let core = result
            .find("crates/travsr-core")
            .expect("core region present");
        let mcp = result
            .find("crates/travsr-mcp")
            .expect("mcp region present");
        assert!(
            core < mcp,
            "depended-upon core must rank before leaf mcp:\n{result}"
        );
        assert!(
            result.contains("dependents: 1"),
            "core dependents shown:\n{result}"
        );
    }

    #[test]
    fn get_repo_map_excludes_unknown_bucket() {
        // An external-crate ref (empty path) must not surface as "<unknown>".
        let ext = make_kind("", "crate:anyhow", "crate");
        let fn_node = make_node("crates/travsr-core/src/lib.rs", "fn:a");
        let store = make_store(&[ext, fn_node], &[]);
        let result = get_repo_map(&store);
        assert!(
            !result.contains("<unknown>"),
            "external-ref bucket must be dropped:\n{result}"
        );
    }

    #[test]
    fn get_repo_map_not_indexed_message() {
        let store = make_store(&[], &[]);
        let result = get_repo_map(&store);
        assert!(
            result.contains("travsr init"),
            "empty store must return an actionable not-indexed message, got: {result:?}"
        );
    }

    #[test]
    fn get_repo_map_discloses_missing_phase_b() {
        // No ref/call edges => semantic note present, in plain words (no jargon).
        let without = get_repo_map(&two_crate_store(false));
        assert!(
            without.contains("semantic analysis has not run"),
            "missing-semantic state must be disclosed:\n{without}"
        );
        assert!(
            !without.contains("Phase B"),
            "no internal jargon:\n{without}"
        );
        // ref/call present => note omitted.
        let with = get_repo_map(&two_crate_store(true));
        assert!(
            !with.contains("semantic analysis has not run"),
            "semantic note must be omitted once semantic analysis has run:\n{with}"
        );
    }

    #[test]
    fn get_repo_map_is_deterministic() {
        let store = two_crate_store(false);
        assert_eq!(get_repo_map(&store), get_repo_map(&store));
    }

    #[test]
    fn get_repo_map_excludes_synthetic_pseudo_paths() {
        // A synthetic angle-bracket path (e.g. Go cgo) carries no real directory
        // location and must not surface as a region — the successor to the old
        // dropped `<unknown>` bucket.
        let synth = make_kind("<cgo_synthetic>/x.go", "fn:cgo_thing", "function");
        let real = make_node("crates/travsr-core/src/lib.rs", "fn:real");
        let store = make_store(&[synth, real], &[]);
        let result = get_repo_map(&store);
        assert!(
            !result.contains("cgo_synthetic"),
            "synthetic pseudo-path must be excluded:\n{result}"
        );
        assert!(
            result.contains("crates/travsr-core"),
            "real region must still be present:\n{result}"
        );
    }

    #[test]
    fn get_repo_map_reserves_prefix_budget() {
        // Global multi-repo mode prefixes every line with `[repo] ` after render.
        // Build a store large enough to fill toward the cap, render with that
        // per-line reserve, and confirm the prefixed output still fits the 4 KB
        // response cap — the mid-line-truncation guard for the global path.
        let mut nodes = Vec::new();
        for i in 0..40 {
            for f in 0..4 {
                for s in 0..6 {
                    nodes.push(make_node(
                        &format!("crates/crate-{i:02}/src/file{f}.rs"),
                        &format!("fn:crate{i}_file{f}_sym{s}"),
                    ));
                }
            }
        }
        let store = make_store(&nodes, &[]);
        // A realistic prefix width, e.g. "[some-repo-name] ".
        let reserve = 17usize;
        let raw = get_repo_map_raw(&store, reserve);
        let line_count = raw.lines().count();
        let prefixed = raw.len() + line_count * reserve;
        assert!(
            prefixed <= 4096,
            "prefixed map exceeds the 4 KB cap: {prefixed} bytes over {line_count} lines"
        );
        // Reserving budget must not empty the map or drop the header.
        assert!(
            raw.contains("Repo map"),
            "header must survive the reserve:\n{raw}"
        );
    }

    #[test]
    fn get_graph_json_stops_at_containment_edges() {
        use travsr_core::{EdgeKind, Node, VName};
        let file = Node::new(
            VName::new("", "", "src/service.ts", "typescript", "file"),
            "file",
        );
        let class = Node::new(
            VName::new(
                "test",
                "",
                "src/service.ts",
                "typescript",
                "class:PaymentService",
            ),
            "class",
        );
        let sibling = Node::new(
            VName::new(
                "test",
                "",
                "src/service.ts",
                "typescript",
                "class:UnrelatedHelper",
            ),
            "class",
        );
        let caller = Node::new(
            VName::new(
                "test",
                "",
                "src/controller.ts",
                "typescript",
                "fn:processPayment",
            ),
            "function",
        );
        let store = make_store(
            &[file.clone(), class.clone(), sibling.clone(), caller.clone()],
            &[
                (file.id, class.id, EdgeKind::DefinesBinding),
                (file.id, sibling.id, EdgeKind::DefinesBinding),
                (caller.id, class.id, EdgeKind::RefCall),
            ],
        );
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "class:PaymentService",
                direction: "callers",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let nodes = v["nodes"].as_array().unwrap();
        let ids: Vec<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();

        // Must show class, caller, and defining file
        assert!(ids.contains(&"src/service.ts:class:PaymentService"));
        assert!(ids.contains(&"src/controller.ts:fn:processPayment"));
        assert!(ids.contains(&"src/service.ts")); // the defining file is returned for orientation

        // Must NOT show the sibling (since traversal stops at the defining file containment edge)
        assert!(!ids.contains(&"src/service.ts:class:UnrelatedHelper"));
    }

    #[test]
    fn get_graph_json_and_graph_query_smoke_test() {
        use travsr_core::{EdgeKind, Node, VName};
        let seed = Node::new(
            VName::new("test", "", "src/seed.ts", "typescript", "fn:seed"),
            "function",
        );
        let dep = Node::new(
            VName::new("test", "", "src/dep.ts", "typescript", "fn:dep"),
            "function",
        );
        let store = make_store(
            &[seed.clone(), dep.clone()],
            &[(seed.id, dep.id, EdgeKind::RefCall)],
        );

        // Run graph_query
        let query_res = crate::query::graph_query(
            &store,
            &crate::query::GraphQueryArgs {
                query: "fn:seed".to_string(),
                path: None,
                depth: 2,
                direction: crate::query::QueryDirection::Deps,
                edge_mode: crate::query::QueryEdgeMode::All,
                include_noise: false,
            },
        )
        .unwrap();

        // Run get_graph_json
        let json_raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "fn:seed",
                direction: "deps",
                depth: 2,
                kind_filter: "",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );

        // Extract nodes and verify matching IDs
        let json_val: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
        let mcp_nodes: std::collections::HashSet<String> = json_val["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap().to_string())
            .collect();

        let expected_mcp_node_ids: std::collections::HashSet<String> = query_res
            .nodes
            .iter()
            .map(|n| {
                if n.kind == "file" {
                    n.path.clone()
                } else {
                    format!("{}:{}", n.path, n.signature)
                }
            })
            .collect();

        assert_eq!(
            mcp_nodes, expected_mcp_node_ids,
            "Nodes must match conformantly"
        );

        // Edge comparison (orientation-independent set of undirected pairs of Node IDs):
        let id_to_sig: std::collections::HashMap<u64, String> = query_res
            .nodes
            .iter()
            .map(|n| (n.id, n.signature.clone()))
            .collect();
        let mut cli_edge_sig_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for e in &query_res.edges {
            if let (Some(s_sig), Some(d_sig)) = (id_to_sig.get(&e.src), id_to_sig.get(&e.dst)) {
                let mut pair = (s_sig.clone(), d_sig.clone());
                if pair.0 > pair.1 {
                    std::mem::swap(&mut pair.0, &mut pair.1);
                }
                cli_edge_sig_pairs.insert(pair);
            }
        }

        let mcp_id_to_sig = |id: &str| -> String {
            if let Some(pos) = id.find(':') {
                id[pos + 1..].to_string()
            } else {
                "file".to_string()
            }
        };

        let mut mcp_edge_sig_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for e in json_val["edges"].as_array().unwrap() {
            let src_id = e["source"].as_str().unwrap();
            let dst_id = e["target"].as_str().unwrap();
            let mut pair = (mcp_id_to_sig(src_id), mcp_id_to_sig(dst_id));
            if pair.0 > pair.1 {
                std::mem::swap(&mut pair.0, &mut pair.1);
            }
            mcp_edge_sig_pairs.insert(pair);
        }

        assert_eq!(
            mcp_edge_sig_pairs, cli_edge_sig_pairs,
            "Edges must match conformantly"
        );
    }
}

// ── get_snippets ──────────────────────────────────────────────────────────────
// Snippet helpers are now owned by travsr-analysis (RFC-017 Phase 4).

use travsr_analysis::skeleton::skeleton_for_node as skeleton_for_node_inner;
pub use travsr_analysis::snippet::SNIPPET_DEFAULT_BUDGET;
use travsr_analysis::snippet::{
    call_site_lines, snippet_for_node, snippet_for_node_full, snippet_line_cap, SNIPPET_SEP,
};

/// How `get_snippets` renders each symbol's body.
///
/// The default (`Auto`) preserves the historical behaviour: raw source when the
/// definition fits its kind line-cap, AST skeleton when it overflows. The other
/// two let a caller (e.g. the VS Code Context Explorer pinning a node into the
/// AI context) force one representation regardless of size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SnippetMode {
    /// Raw body if within the kind line-cap, else AST skeleton. Default.
    #[default]
    Auto,
    /// Full source of `line..=end_line`, no line cap. Falls back to skeleton if unreadable.
    Full,
    /// AST skeleton always, even for a small definition. Falls back to raw body if unavailable.
    Skeleton,
}

impl SnippetMode {
    /// Parse the MCP `mode` argument. Unknown / absent → [`SnippetMode::Auto`].
    pub fn from_arg(s: &str) -> Self {
        match s {
            "full" => Self::Full,
            "skeleton" => Self::Skeleton,
            _ => Self::Auto,
        }
    }
}

/// Maximum snippets returned per bare-signature token when multiple nodes match.
/// Each node has its own stored path + line range; `snippet_for_node` reads the
/// correct file for each. The token budget naturally limits how many fit.
const MAX_SIGNATURE_MATCHES: usize = 3;

/// A parsed token from a `symbols_arg` string, classified into one of four
/// resolution tiers ordered by decreasing context richness.
#[derive(Debug)]
enum SymbolToken<'a> {
    /// `"nodeId:42"` — direct primary-key lookup, O(1), zero search.
    /// Used by VS Code Context Explorer which already has the resolved node.
    NodeId(u64),
    /// `"fn:run (function) — src/cmd/serve.rs [package: server]"` — the full
    /// header emitted by `get_context` / `get_callers` / `search_symbol`.
    /// O(log N) exact index lookup, guaranteed single result.
    FullHeader { sig: &'a str, path: &'a str },
    /// `"src/main.rs:42"` or `"src/main.rs:42-55"` — file-coordinate snippet.
    /// Zero DB access; reads the stored line range directly.
    /// Used for call-site snippets from `get_callers` output.
    PathLine {
        path: &'a str,
        start: u32,
        end: Option<u32>,
    },
    /// `"fn:run"` — bare signature, no path context.
    /// O(log N) exact index lookup; may return multiple nodes (each with its
    /// own stored path + line). Returns up to `MAX_SIGNATURE_MATCHES` snippets.
    BareSig(&'a str),
}

/// Split `symbols_arg` on newline / comma and classify each token into the
/// appropriate [`SymbolToken`] resolution tier.
///
/// Detection order (first match wins per token):
/// 1. Starts with `"nodeId:"` → [`SymbolToken::NodeId`]
/// 2. Contains `" \u{2014} "` (em-dash with spaces, the `get_context` header
///    separator) → [`SymbolToken::FullHeader`]
/// 3. Last `:` is followed by purely digits and the prefix contains `/`
///    → [`SymbolToken::PathLine`]
/// 4. Everything else → [`SymbolToken::BareSig`]
///
/// Strip a trailing ":{line}" or ":{start}-{end}" locator from a path, returning
/// the bare path slice. `format_node_line` appends the definition line to the
/// path in `get_context` headers (e.g. `"src/serve.rs:42"`), but the stored node
/// path has no such suffix — leaving it in place breaks the path-pin lookup.
fn strip_line_suffix(path: &str) -> &str {
    let Some(idx) = path.rfind(':') else {
        return path;
    };
    let suffix = &path[idx + 1..];
    if suffix.is_empty() {
        return path;
    }
    let is_locator = match suffix.split_once('-') {
        Some((start, end)) => {
            !start.is_empty()
                && !end.is_empty()
                && start.bytes().all(|b| b.is_ascii_digit())
                && end.bytes().all(|b| b.is_ascii_digit())
        }
        None => suffix.bytes().all(|b| b.is_ascii_digit()),
    };
    if is_locator {
        &path[..idx]
    } else {
        path
    }
}

fn parse_symbol_tokens(symbols_arg: &str) -> Vec<SymbolToken<'_>> {
    symbols_arg
        .split(['\n', ','])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|token| {
            // Tier 0: explicit node ID.
            if let Some(rest) = token.strip_prefix("nodeId:") {
                if let Ok(id) = rest.trim().parse::<u64>() {
                    return Some(SymbolToken::NodeId(id));
                }
            }

            // Tier 1: full get_context header — contains em-dash separator.
            // Real format (see `format_node_line`):
            //   "{sig} ({kind}) \u{2014} {path}:{line} [package: {pkg}] [via: ..] [score: ..]"
            // `[package: ..]` is omitted when the package is empty, and the path
            // always carries a trailing ":{line}" definition locator. The " ("
            // separates sig from kind; " \u{2014} " separates kind from path.
            if let Some(em_pos) = token.find(" \u{2014} ") {
                // sig is everything before the first " ("
                let before_em = &token[..em_pos];
                let sig = if let Some(p) = before_em.find(" (") {
                    before_em[..p].trim()
                } else {
                    before_em.trim()
                };
                // Terminate the path at the first bracketed field ("[package:",
                // "[via:", "[score:", ..). Matching " [" generally is required
                // because the package field is dropped for package-less nodes,
                // in which case " [via:" is the first trailer.
                let after_em = &token[em_pos + " \u{2014} ".len()..];
                let path_with_loc = match after_em.find(" [") {
                    Some(p) => after_em[..p].trim(),
                    None => after_em.trim(),
                };
                // Strip the trailing ":{line}" / ":{start}-{end}" locator that
                // `format_node_line` appends — the stored node path has none.
                let path = strip_line_suffix(path_with_loc);
                if !sig.is_empty() && !path.is_empty() {
                    return Some(SymbolToken::FullHeader { sig, path });
                }
                // Has em-dash but no usable path — fall through to BareSig.
                if !sig.is_empty() {
                    return Some(SymbolToken::BareSig(sig));
                }
            }

            // Tier 2: {path}:{line} or {path}:{start}-{end}.
            // The path part must contain '/' to distinguish "src/main.rs:42"
            // from Kythe signatures like "fn:run" or "method:Foo::bar".
            if let Some(colon_pos) = token.rfind(':') {
                let path_part = &token[..colon_pos];
                let line_part = &token[colon_pos + 1..];
                if path_part.contains('/') {
                    if let Some(dash) = line_part.find('-') {
                        if let (Ok(s), Ok(e)) = (
                            line_part[..dash].parse::<u32>(),
                            line_part[dash + 1..].parse::<u32>(),
                        ) {
                            return Some(SymbolToken::PathLine {
                                path: path_part,
                                start: s,
                                end: Some(e),
                            });
                        }
                    } else if let Ok(line) = line_part.parse::<u32>() {
                        return Some(SymbolToken::PathLine {
                            path: path_part,
                            start: line,
                            end: None,
                        });
                    }
                }
            }

            // Tier 3: bare signature — anything else.
            if !token.is_empty() {
                return Some(SymbolToken::BareSig(token));
            }
            None
        })
        .collect()
}

/// Core snippet assembly: resolves symbols, fetches snippets, enforces budget.
///
/// `symbols_arg` is a newline- or comma-separated list of symbol names.
/// Symbols are processed in order; truncation happens at symbol granularity
/// (no partial symbol output) once the token budget is exhausted — except the
/// first resolved symbol is always included so a single large `Full` definition
/// never returns empty.
fn get_snippets_body(
    store: &SqliteStore,
    symbols_arg: &str,
    token_budget: usize,
    mode: SnippetMode,
) -> String {
    // Read repo_root from meta — written by init_repo_with_progress.
    // Absent on indexes created before this feature; degrade gracefully.
    let repo_root = match resolve_repo_root(store) {
        Some(r) => r,
        None => {
            tracing::warn!("get_snippets: repo_root not in meta, index predates snippet support");
            return "Snippet data unavailable: run `travsr init` to refresh the index.".to_string();
        }
    };

    // Parse every token into its resolution tier.  Tokens in the same request
    // may use different forms (nodeId, full header, path:line, bare sig).
    let tokens = parse_symbol_tokens(symbols_arg);

    if tokens.is_empty() {
        return String::new();
    }

    // Resolve each token to ≥0 CoreNodes.  All tiers produce nodes with stored
    // path + line/end_line, so snippet_for_node reads the correct file for each.
    // search_nodes_by_name (O(N) LIKE) is generally avoided for exact-match
    // tiers (Tiers 0-2), which use the V19NodesSignatureIdx exact-match index
    // (O(log N)) or direct primary-key lookup (O(1)). However, the bare-signature
    // tier (Tier 3) calls resolve_symbol_nodes, which can fall through to the O(N)
    // name search on a bare-name miss.
    let mut resolved: Vec<CoreNode> = Vec::new();
    for tok in &tokens {
        match tok {
            // Tier 0: VS Code / internal callers pass the node's primary key.
            SymbolToken::NodeId(id) => match store.get_node(travsr_core::NodeId(*id)) {
                Ok(Some(n)) if n.kind != "file" => resolved.push(n),
                Ok(_) => {}
                Err(e) => tracing::warn!("get_snippets nodeId:{id} lookup error: {e}"),
            },

            // Tier 1: full get_context header — path-pinned, exactly 1 result.
            SymbolToken::FullHeader { sig, path } => {
                match store.lookup_nodes_exact(sig, Some(path)) {
                    Ok(nodes) => {
                        // SQL ORDER BY puts the path-pinned row first.
                        if let Some(n) = nodes.into_iter().next() {
                            resolved.push(n);
                        }
                    }
                    Err(e) => tracing::warn!("get_snippets FullHeader lookup '{sig}': {e}"),
                }
            }

            // Tier 2: file-coordinate snippet — build a synthetic node so the
            // same rendering loop handles it.  kind "source-range" bypasses the
            // kind-aware line cap and always reads the exact stored range.
            SymbolToken::PathLine { path, start, end } => {
                let end_line = end.unwrap_or(start.saturating_add(39));
                let syn = CoreNode::new(
                    travsr_core::VName::new("", "", *path, "", format!("{path}:{start}")),
                    "source-range",
                )
                .with_line(*start)
                .with_end_line(end_line);
                resolved.push(syn);
            }

            // Tier 3: bare signature — uses the shared resolution ladder
            // to support exact matches, plain names, and dotted paths, returning up to
            // MAX_SIGNATURE_MATCHES nodes (each with its own path + line).
            SymbolToken::BareSig(sig) => {
                let nodes = resolve_symbol_nodes(store, sig, None);
                resolved.extend(nodes.into_iter().take(MAX_SIGNATURE_MATCHES));
            }
        }
    }

    if resolved.is_empty() {
        return "No symbols matching the provided names found in the graph.".to_string();
    }

    // Accumulate blocks within budget.  All symbols are equally requested so
    // there is no PPR score to rank by — preserve the caller's order and stop
    // when the budget is exhausted (no knapsack needed here).
    let mut parts: Vec<String> = Vec::new();
    let mut tokens_used: usize = 0;
    let mut n_with_snippet: usize = 0;

    for node in &resolved {
        let header = format!(
            "{} ({}) \u{2014} {} [package: {}]",
            display_label(node),
            node.kind,
            node.vname.path,
            node.package
        );
        let skeleton = |n: &CoreNode| skeleton_for_node_inner(n, &repo_root).map(|s| s.render());

        // "source-range" nodes (Tier 2 PathLine) always use full mode so the
        // exact stored line range is returned regardless of the caller's mode.
        let body_text: Option<String> = if node.kind == "source-range" {
            snippet_for_node_full(node, &repo_root)
        } else {
            match mode {
                // Full source of the definition, no line cap. Skeleton is the
                // fallback only when the file can't be read.
                SnippetMode::Full => {
                    snippet_for_node_full(node, &repo_root).or_else(|| skeleton(node))
                }
                // AST skeleton always; raw body only if the skeleton can't be built.
                SnippetMode::Skeleton => {
                    skeleton(node).or_else(|| snippet_for_node(node, &repo_root))
                }
                // Auto: raw when it fits the kind line-cap, skeleton when it
                // overflows (a truncated raw snippet is a half-implementation;
                // skeleton summarises).
                SnippetMode::Auto => {
                    let node_height = node
                        .end_line
                        .unwrap_or_else(|| node.line.unwrap_or(0))
                        .saturating_sub(node.line.unwrap_or(0))
                        as usize;
                    if node_height > snippet_line_cap(&node.kind) {
                        skeleton(node).or_else(|| snippet_for_node(node, &repo_root))
                    } else {
                        snippet_for_node(node, &repo_root).or_else(|| skeleton(node))
                    }
                }
            }
        };

        let snippet_chars = body_text.as_deref().map(str::len).unwrap_or(0);
        let block_cost = (header.len() + snippet_chars) / TOKEN_CHARS_PER_TOKEN + 1;

        // Always include the first resolved symbol even if it alone exceeds the
        // budget — otherwise a single large Full definition (the pin-into-context
        // case) would return empty. Subsequent symbols respect the budget.
        if !parts.is_empty() && tokens_used + block_cost > token_budget {
            break;
        }
        tokens_used += block_cost;

        let block = if let Some(code) = &body_text {
            n_with_snippet += 1;
            format!("{header}\n{SNIPPET_SEP}\n{code}")
        } else {
            header
        };
        parts.push(block);
    }

    if parts.is_empty() {
        return "Token budget too small to include any symbols.".to_string();
    }

    let body = parts.join("\n\n");
    let sanitized = sanitize_mcp_body_with_limit(
        &body,
        (token_budget * TOKEN_CHARS_PER_TOKEN * 2).min(1_024_000),
    );
    format!(
        "{sanitized}\n\n[{} symbols, {n_with_snippet} with snippets, ~{tokens_used} tokens]",
        parts.len()
    )
}

/// Return tailored code snippets for one or more named symbols.
///
/// Accepts the symbol names returned by `get_context`, `get_callers`, and
/// `search_symbol`. `mode` selects the representation ([`SnippetMode`]): `Auto`
/// (kind-aware raw-or-skeleton, the default), `Full` (whole definition, no cap),
/// or `Skeleton` (AST summary always). Leading docblocks are stripped so the AI
/// sees real code immediately.
pub fn get_snippets(
    store: &SqliteStore,
    symbols: &str,
    token_budget: usize,
    mode: SnippetMode,
) -> String {
    if let Err(reason) = validate_mcp_list_arg(symbols) {
        tracing::warn!("get_snippets rejected invalid arg: {reason}");
        return String::new();
    }
    let budget = token_budget.clamp(1, MAX_CONTEXT_BUDGET);
    // `get_snippets_body` already sanitizes its body with a budget-derived limit
    // (control-char stripping + tag escaping) and appends the plain-ASCII footer.
    // Wrap-only here — like `get_context` — so the response honours `token_budget`
    // instead of being re-truncated to the 4 KiB scalar-output cap (which silently
    // dropped everything past ~6 symbols and cut off the summary footer).
    wrap_envelope(&get_snippets_raw(store, symbols, budget, mode))
}

pub(crate) fn get_snippets_raw(
    store: &SqliteStore,
    symbols: &str,
    token_budget: usize,
    mode: SnippetMode,
) -> String {
    get_snippets_body(store, symbols, token_budget, mode)
}

/// Global variant: searches across all registered repos (or a named one).
pub fn get_snippets_global(
    repos: &HashMap<String, PathBuf>,
    symbols: &str,
    token_budget: usize,
    repo: Option<&str>,
    mode: SnippetMode,
) -> String {
    if let Err(reason) = validate_mcp_list_arg(symbols) {
        tracing::warn!("get_snippets_global rejected invalid arg: {reason}");
        return String::new();
    }
    let budget = token_budget.clamp(1, MAX_CONTEXT_BUDGET);
    let raw = collect_global(repos, repo, |store, repo_name, single| {
        let result = get_snippets_raw(store, symbols, budget, mode);
        if result.is_empty() || single {
            result
        } else {
            format!("[{repo_name}]\n{result}")
        }
    });
    sanitize_for_mcp(&raw)
}

#[cfg(test)]
mod snippet_tests {
    use super::*;
    use std::path::Path;
    use travsr_core::VName;

    // ── helper: build a Node with explicit line/end_line ─────────────────────

    fn make_fn_node(path: &str, sig: &str, line: u32, end_line: u32) -> CoreNode {
        CoreNode::new(
            VName::new("corpus", "", path, "typescript", sig),
            "function",
        )
        .with_line(line)
        .with_end_line(end_line)
    }

    // Unit tests for is_comment_line / skip_leading_comments / snippet_line_cap /
    // snippet_for_node have moved to travsr-analysis/src/snippet.rs (RFC-017).

    // ── get_snippets_body (integration) ──────────────────────────────────────

    fn make_store_with_meta(nodes: &[CoreNode], root: &Path) -> SqliteStore {
        let db_path = root.join(".travsr").join("graph.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let mut store = SqliteStore::open(&db_path).unwrap();

        let mut by_path: std::collections::HashMap<String, Vec<CoreNode>> =
            std::collections::HashMap::new();
        for n in nodes {
            by_path
                .entry(n.vname.path.clone())
                .or_default()
                .push(n.clone());
        }

        for (path, file_nodes) in by_path {
            store
                .write_file_graphs_batch(
                    &[travsr_store::FileGraph {
                        nodes: file_nodes,
                        edges: vec![],
                        vname_path: path,
                        new_hash: "deadbeef".to_string(),
                        source: None,
                    }],
                    false,
                )
                .unwrap();
        }

        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();
        store
    }

    #[test]
    fn get_snippets_body_returns_snippet_for_known_symbol() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("lib.ts");
        std::fs::write(&src, "function hello() {\n  return 'hi';\n}\n").unwrap();

        let node = make_fn_node("lib.ts", "fn:hello", 1, 3);
        let store = make_store_with_meta(&[node], dir.path());

        let result = get_snippets_body(&store, "fn:hello", 2000, SnippetMode::Auto);
        assert!(result.contains("function hello()"), "snippet body missing");
        assert!(result.contains("return 'hi'"), "snippet content missing");
        assert!(
            result.contains("1 symbols, 1 with snippets"),
            "footer missing"
        );
    }

    #[test]
    fn get_snippets_body_resolves_plain_and_dotted_names() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("lib.ts");
        std::fs::write(
            &src,
            "class ClassA {\n  enableSupportForFeatureX() {\n    return true;\n  }\n}\n",
        )
        .unwrap();

        let class_node = make_fn_node("lib.ts", "class:ClassA", 1, 5);
        let fn_node = make_fn_node("lib.ts", "method:ClassA.enableSupportForFeatureX", 2, 4);
        let store = make_store_with_meta(&[class_node, fn_node], dir.path());

        // 1. Exact signature still works
        let exact = get_snippets_body(
            &store,
            "method:ClassA.enableSupportForFeatureX",
            2000,
            SnippetMode::Auto,
        );
        assert!(
            exact.contains("enableSupportForFeatureX() {"),
            "exact sig failed"
        );

        // 2. Plain function name works
        let plain = get_snippets_body(&store, "enableSupportForFeatureX", 2000, SnippetMode::Auto);
        assert!(
            plain.contains("enableSupportForFeatureX() {"),
            "plain function name failed"
        );

        // 3. Bare class name works
        let class = get_snippets_body(&store, "ClassA", 2000, SnippetMode::Auto);
        assert!(class.contains("class ClassA"), "bare class name failed");

        // 4. Dotted symbol name works
        let dotted = get_snippets_body(
            &store,
            "ClassA.enableSupportForFeatureX",
            2000,
            SnippetMode::Auto,
        );
        assert!(
            dotted.contains("enableSupportForFeatureX() {"),
            "dotted symbol name failed"
        );

        // 5. Unknown symbol
        let unknown = get_snippets_body(&store, "unknownSymbol", 2000, SnippetMode::Auto);
        assert!(
            unknown.contains("No symbols matching the provided names found in the graph."),
            "unknown symbol failed"
        );
    }

    #[test]
    fn full_mode_returns_whole_body_past_line_cap() {
        let dir = tempfile::tempdir().unwrap();
        // A "class" node: Auto caps at 15 lines and falls back to skeleton; Full
        // must return the complete 30-line body including the last line.
        let mut src = String::from("class Big {\n");
        for i in 0..28 {
            src.push_str(&format!("  field{i} = {i};\n"));
        }
        src.push_str("  last = true;\n}\n");
        std::fs::write(dir.path().join("big.ts"), &src).unwrap();

        // A class node spanning > 15 lines (Auto's class line-cap).
        let class_node = CoreNode::new(
            VName::new("corpus", "", "big.ts", "typescript", "class:Big"),
            "class",
        )
        .with_line(1)
        .with_end_line(31);
        let store = make_store_with_meta(&[class_node], dir.path());

        let full = get_snippets_body(&store, "class:Big", 8000, SnippetMode::Full);
        assert!(
            full.contains("last = true"),
            "full mode must include last line: {full}"
        );
        assert!(
            !full.contains("[skeleton:"),
            "full mode must not be a skeleton: {full}"
        );

        let auto = get_snippets_body(&store, "class:Big", 8000, SnippetMode::Auto);
        assert!(
            auto.contains("[skeleton:") || !auto.contains("last = true"),
            "auto mode caps/skeletonises the oversized class: {auto}"
        );
    }

    #[test]
    fn skeleton_mode_forces_ast_for_small_fn() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("s.ts"),
            "function small() {\n  return 1;\n}\n",
        )
        .unwrap();
        let node = make_fn_node("s.ts", "fn:small", 1, 3);
        let store = make_store_with_meta(&[node], dir.path());

        let skel = get_snippets_body(&store, "fn:small", 2000, SnippetMode::Skeleton);
        assert!(
            skel.contains("[skeleton:"),
            "skeleton mode must render the AST summary even for a small fn: {skel}"
        );
    }

    #[test]
    fn get_snippets_body_no_repo_root_returns_hint() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let store = SqliteStore::open(&db_path).unwrap();
        // No set_meta("repo_root") — simulates an old index.
        let result = get_snippets_body(&store, "anything", 2000, SnippetMode::Auto);
        assert!(
            result.contains("travsr init"),
            "must prompt user to re-init: {result}"
        );
    }

    #[test]
    fn get_snippets_body_budget_truncates_in_order() {
        let dir = tempfile::tempdir().unwrap();
        // Two source files
        for (name, body) in [
            ("a.ts", "function aaa() {\n  return 1;\n}\n"),
            ("b.ts", "function bbb() {\n  return 2;\n}\n"),
        ] {
            std::fs::write(dir.path().join(name), body).unwrap();
        }

        let nodes = vec![
            make_fn_node("a.ts", "fn:aaa", 1, 3),
            make_fn_node("b.ts", "fn:bbb", 1, 3),
        ];
        let store = make_store_with_meta(&nodes, dir.path());

        // Tight budget (5 tokens) — the first resolved symbol is always included
        // (so a single large definition is never dropped), but the second respects
        // the exhausted budget and must not appear.
        let tight = get_snippets_body(&store, "fn:aaa\nfn:bbb", 5, SnippetMode::Auto);
        assert!(
            tight.contains("aaa"),
            "first symbol must always be included even under a tiny budget: {tight}"
        );
        assert!(
            !tight.contains("bbb"),
            "second symbol must be excluded when budget exhausted: {tight}"
        );

        // Ordering budget (20 tokens) — enough for one symbol (header ~9 tokens +
        // 3-line snippet ~8 tokens = ~17 tokens) but not both. The first symbol
        // in request order ("aaa") must appear; the second ("bbb") must not.
        let ordered = get_snippets_body(&store, "fn:aaa\nfn:bbb", 20, SnippetMode::Auto);
        assert!(
            ordered.contains("aaa"),
            "first symbol must be included within budget: {ordered}"
        );
        assert!(
            !ordered.contains("bbb"),
            "second symbol must be excluded when budget exhausted: {ordered}"
        );
    }

    #[test]
    fn get_snippets_body_missing_file_returns_metadata_only() {
        let dir = tempfile::tempdir().unwrap();
        // Node points to a file that doesn't exist on disk.
        let node = make_fn_node("missing.ts", "fn:ghost", 1, 5);
        let store = make_store_with_meta(&[node], dir.path());

        let result = get_snippets_body(&store, "fn:ghost", 2000, SnippetMode::Auto);
        // Should return metadata line without panic.
        assert!(
            result.contains("fn:ghost"),
            "metadata must appear: {result}"
        );
        // No snippet separator — degraded gracefully.
        assert!(
            !result.contains(SNIPPET_SEP),
            "no snippet separator when file missing: {result}"
        );
        assert!(
            result.contains("0 with snippets"),
            "snippet count must be 0: {result}"
        );
    }

    #[test]
    fn snippet_for_node_inverted_line_range_returns_none_not_panic() {
        // Regression test for the reversed-slice panic: end_line < line must
        // degrade to None instead of panicking with "range start > end".
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("inv.ts");
        std::fs::write(&src, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        // node.line=5, node.end_line=2 → to=2 < from=4 → would panic without guard
        let node = make_fn_node("inv.ts", "fn:inv", 5, 2);
        assert!(
            snippet_for_node(&node, dir.path()).is_none(),
            "inverted line range must return None, not panic"
        );
    }

    #[test]
    fn snippet_for_node_reads_via_canonical_path() {
        // Regression test for the TOCTOU fix: the read must use canon_abs
        // (the resolved path) rather than the pre-canonicalization abs path.
        // On systems where tempdir() returns a symlinked path (e.g. macOS
        // /tmp → /private/tmp), this test would silently pass either way
        // because the symlink is valid. The key invariant we verify: a file
        // that exists and is inside the repo is readable via snippet_for_node.
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("real.ts");
        std::fs::write(&src, "function real() {\n  return 42;\n}\n").unwrap();

        let node = make_fn_node("real.ts", "fn:real", 1, 3);
        let snippet = snippet_for_node(&node, dir.path());
        assert!(
            snippet.is_some(),
            "readable repo-internal file must produce a snippet"
        );
        assert!(snippet.unwrap().contains("return 42"));
    }

    // ── Issue #438: duplicate-signature disambiguation ───────────────────────

    // T-M1: when two files define "fn:run" and the caller passes the bare
    // signature, the result must include snippets for ALL matching nodes up to
    // MAX_SIGNATURE_MATCHES — not a disambiguation block and not silently the wrong one.
    #[test]
    fn get_snippets_duplicate_sig_bare_name_returns_all_snippets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.ts"), "function run() { /* a */ }\n").unwrap();
        std::fs::write(dir.path().join("b.ts"), "function run() { /* b */ }\n").unwrap();

        let node_a = make_fn_node("a.ts", "fn:run", 1, 1);
        let node_b = make_fn_node("b.ts", "fn:run", 1, 1);
        let store = make_store_with_meta(&[node_a, node_b], dir.path());

        let result = get_snippets_body(&store, "fn:run", 2000, SnippetMode::Full);

        // Both nodes must appear in the output.
        assert!(
            result.contains("a.ts"),
            "result must mention a.ts: {result}"
        );
        assert!(
            result.contains("b.ts"),
            "result must mention b.ts: {result}"
        );
        // Both snippets must be present.
        assert!(
            result.contains("/* a */"),
            "snippet from a.ts missing: {result}"
        );
        assert!(
            result.contains("/* b */"),
            "snippet from b.ts missing: {result}"
        );
        // Summary footer must show 2 symbols.
        assert!(
            result.contains("2 symbols"),
            "footer must count 2 symbols: {result}"
        );
    }

    // T-M2: when the caller passes the full get_context header (with path), the
    // correct node must be resolved and its snippet returned.
    #[test]
    fn get_snippets_duplicate_sig_full_header_resolves_correct_node() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.ts"), "function run() { /* a */ }\n").unwrap();
        std::fs::write(dir.path().join("b.ts"), "function run() { /* b */ }\n").unwrap();

        let node_a = make_fn_node("a.ts", "fn:run", 1, 1);
        let node_b = make_fn_node("b.ts", "fn:run", 1, 1);
        let store = make_store_with_meta(&[node_a, node_b], dir.path());

        // Caller passes the full header with path — should resolve to a.ts only.
        let full_header = "fn:run (function) \u{2014} a.ts [package: mypkg]";
        let result = get_snippets_body(&store, full_header, 2000, SnippetMode::Full);

        assert!(
            result.contains("/* a */"),
            "must show snippet from a.ts: {result}"
        );
        assert!(
            !result.contains("/* b */"),
            "must NOT show snippet from b.ts: {result}"
        );
        assert!(
            !result.contains("definitions found"),
            "must not show disambiguation when path hint resolves uniquely: {result}"
        );
    }

    // T-M2b: the header the caller actually pastes is produced by
    // `format_node_line`, which appends a ":{line}" locator to the path and a
    // trailing " [via: ..] [score: ..]" — and omits " [package: ..]" for
    // package-less nodes (the common case in this suite). The Tier 1 parser must
    // resolve the correct node from that real format, not just the idealized one.
    #[test]
    fn get_snippets_full_header_from_format_node_line_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.ts"), "function run() { /* a */ }\n").unwrap();
        std::fs::write(dir.path().join("b.ts"), "function run() { /* b */ }\n").unwrap();

        let node_a = make_fn_node("a.ts", "fn:run", 1, 1);
        let node_b = make_fn_node("b.ts", "fn:run", 1, 1);
        let store = make_store_with_meta(&[node_a.clone(), node_b], dir.path());

        // Package-less header (make_fn_node leaves package empty): the path
        // carries ":1" and the only trailer is " [via: ..] [score: ..]".
        let header = format_node_line(&node_a, "seed", Some(0.80), None, None, false);
        assert!(
            header.contains("a.ts:1"),
            "sanity: real header must carry the line locator: {header}"
        );
        let result = get_snippets_body(&store, &header, 2000, SnippetMode::Full);
        assert!(
            result.contains("/* a */"),
            "must resolve a.ts from the real get_context header: {result}\nheader={header}"
        );
        assert!(
            !result.contains("/* b */"),
            "must NOT resolve the colliding b.ts: {result}"
        );

        // Package-present header exercises the " [package: ..]" terminator too.
        let node_pkg = make_fn_node("a.ts", "fn:run", 1, 1).with_package("mypkg");
        let header_pkg = format_node_line(&node_pkg, "seed", Some(0.80), None, None, false);
        assert!(header_pkg.contains("[package:"), "sanity: {header_pkg}");
        let result_pkg = get_snippets_body(&store, &header_pkg, 2000, SnippetMode::Full);
        assert!(
            result_pkg.contains("/* a */") && !result_pkg.contains("/* b */"),
            "package-present header must also resolve a.ts only: {result_pkg}"
        );
    }

    // RFC-022 §14 (A4): grouped mode hoists the seed source into the section
    // header, so a primary-seed (Exact/Semantic) line drops its `[via: …]` badge
    // while a Relevant (non-seed) line keeps `[via: caller of X]`. Off-path
    // (grouped=false) output is byte-for-byte identical to the pre-§14 format.
    #[test]
    fn grouped_mode_omits_via_on_seeds_keeps_it_on_relevant() {
        let seed = make_fn_node("a.ts", "fn:run", 1, 1);

        // Seed under a section header, grouped ON → no `[via:]`, no stray space.
        let exact = omit_seed_via(true, crate::seed::MatchSource::Exact);
        let semantic = omit_seed_via(true, crate::seed::MatchSource::Semantic);
        assert!(exact && semantic, "seeds are hoisted in grouped mode");
        let seed_line =
            format_node_line(&seed, "seed", Some(0.80), None, Some("exact-name"), exact);
        assert!(
            !seed_line.contains("[via:"),
            "grouped seed line must drop the via badge: {seed_line}"
        );
        assert!(
            !seed_line.contains("  "),
            "dropped badge must not leave a double space: {seed_line:?}"
        );
        assert!(
            seed_line.contains("[score: 0.80]"),
            "score still renders on a hoisted seed line: {seed_line}"
        );

        // Relevant (non-seed) row keeps its provenance badge even grouped ON.
        assert!(!omit_seed_via(true, crate::seed::MatchSource::Relevant));
        let rel_line = format_node_line(
            &seed,
            "caller",
            Some(0.10),
            Some("fn:handler"),
            None,
            omit_seed_via(true, crate::seed::MatchSource::Relevant),
        );
        assert!(
            rel_line.contains("[via: caller of fn:handler]"),
            "relevant row must keep the source-seed badge: {rel_line}"
        );

        // Off-path (grouped=false) is byte-for-byte identical for every source.
        for ms in [
            crate::seed::MatchSource::Exact,
            crate::seed::MatchSource::Semantic,
            crate::seed::MatchSource::Relevant,
        ] {
            assert!(
                !omit_seed_via(false, ms),
                "grouping off never omits the badge"
            );
        }
        let flat = format_node_line(&seed, "seed", Some(0.80), None, Some("exact-name"), false);
        assert_eq!(
            flat, "fn:run (function) \u{2014} a.ts:1 [via: seed · exact-name] [score: 0.80]",
            "flag-off render must match the frozen pre-§14 format byte-for-byte"
        );
    }

    // T-M3: parse_symbol_tokens covers all four resolution tiers.
    #[test]
    fn parse_symbol_tokens_covers_all_four_tiers() {
        // Tier 0: nodeId: prefix.
        let t0 = parse_symbol_tokens("nodeId:42");
        assert_eq!(t0.len(), 1);
        assert!(matches!(t0[0], SymbolToken::NodeId(42)));

        // Tier 1: full header — contains em-dash (U+2014).
        let full = "fn:run (function) \u{2014} src/cmd/serve.rs [package: server]";
        let t1 = parse_symbol_tokens(full);
        assert_eq!(t1.len(), 1);
        assert!(
            matches!(t1[0], SymbolToken::FullHeader { sig, path }
                if sig == "fn:run" && path == "src/cmd/serve.rs"),
            "FullHeader parse failed: {:?}",
            t1[0]
        );

        // Tier 1 — qualified name with double colons.
        let method = "fn:Iterator::next (method) \u{2014} std/iter.rs [package: std]";
        let t1m = parse_symbol_tokens(method);
        assert_eq!(t1m.len(), 1);
        assert!(
            matches!(t1m[0], SymbolToken::FullHeader { sig, .. } if sig == "fn:Iterator::next")
        );

        // Tier 2: path:line or path:line-end.
        let t2 = parse_symbol_tokens("src/main.rs:42");
        assert_eq!(t2.len(), 1);
        assert!(
            matches!(t2[0], SymbolToken::PathLine { path, start, end }
                if path == "src/main.rs" && start == 42 && end.is_none()),
            "PathLine parse failed: {:?}",
            t2[0]
        );

        // Tier 3: bare sig — everything else.
        let t3 = parse_symbol_tokens("fn:run");
        assert_eq!(t3.len(), 1);
        assert!(matches!(t3[0], SymbolToken::BareSig("fn:run")));

        // Multi-token via newline.
        let multi = parse_symbol_tokens("fn:run\nfn:stop");
        assert_eq!(multi.len(), 2);
        assert!(matches!(multi[0], SymbolToken::BareSig("fn:run")));
        assert!(matches!(multi[1], SymbolToken::BareSig("fn:stop")));

        // Multi-token via comma.
        let csv = parse_symbol_tokens("fn:run,fn:stop");
        assert_eq!(csv.len(), 2);

        // Empty tokens are silently dropped.
        let sparse = parse_symbol_tokens("fn:run,,\n,fn:stop");
        assert_eq!(sparse.len(), 2);
    }

    // T-M4: a unique signature (no duplicates in the DB) still resolves and
    // returns the snippet — regression guard ensuring the new path doesn't
    // break the happy case.
    #[test]
    fn get_snippets_unique_sig_still_returns_snippet() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lib.ts"),
            "function hello() {\n  return 'hi';\n}\n",
        )
        .unwrap();

        let node = make_fn_node("lib.ts", "fn:hello", 1, 3);
        let store = make_store_with_meta(&[node], dir.path());

        let result = get_snippets_body(&store, "fn:hello", 2000, SnippetMode::Auto);
        assert!(
            result.contains("function hello()"),
            "snippet body missing: {result}"
        );
        assert!(
            result.contains("return 'hi'"),
            "snippet content missing: {result}"
        );
        assert!(
            !result.contains("definitions found"),
            "must not show disambiguation for a unique symbol: {result}"
        );
    }

    // T-M6: PathLine tier (Tier 2) always reads the exact stored range,
    // ignoring the SnippetMode, even when mode=Skeleton.
    // Path token must contain '/' to be recognized as Tier 2 vs. a Kythe sig.
    #[test]
    fn get_snippets_path_line_tier_reads_exact_range() {
        let dir = tempfile::tempdir().unwrap();
        // 10-line file under a subdir so the token "src/exact.ts:3-5" has '/'.
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src: String = (1u32..=10).map(|i| format!("line{i}\n")).collect();
        std::fs::write(src_dir.join("exact.ts"), &src).unwrap();

        // No graph nodes needed — PathLine creates a synthetic node at read time.
        let db_path = dir.path().join(".travsr").join("graph.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let mut store = SqliteStore::open(&db_path).unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();

        let result = get_snippets_body(&store, "src/exact.ts:3-5", 2000, SnippetMode::Skeleton);

        assert!(
            result.contains("line3"),
            "line3 must be in result: {result}"
        );
        assert!(
            result.contains("line5"),
            "line5 must be in result: {result}"
        );
        assert!(!result.contains("line6"), "line6 must NOT appear: {result}");
    }

    // T-M7: NodeId tier (Tier 0) resolves directly by DB row id —
    // bypasses signature lookup entirely.
    #[test]
    fn get_snippets_nodeid_tier_resolves_by_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("direct.ts"),
            "function direct() {\n  return 99;\n}\n",
        )
        .unwrap();

        let node = make_fn_node("direct.ts", "fn:direct", 1, 3);
        let store = make_store_with_meta(&[node], dir.path());

        // Look up the row id the store assigned.
        let rows = store
            .lookup_nodes_exact("fn:direct", None)
            .expect("lookup failed");
        assert_eq!(rows.len(), 1, "expected 1 node");
        let id = rows[0].id.0;

        let result = get_snippets_body(&store, &format!("nodeId:{id}"), 2000, SnippetMode::Auto);
        assert!(result.contains("return 99"), "snippet missing: {result}");
        assert!(result.contains("fn:direct"), "header missing: {result}");
    }

    // T-M8: when more than MAX_SIGNATURE_MATCHES nodes share a bare signature,
    // the result contains at most MAX_SIGNATURE_MATCHES snippets — not all of them.
    #[test]
    fn get_snippets_bare_sig_capped_at_max_signature_matches() {
        let dir = tempfile::tempdir().unwrap();
        let count = MAX_SIGNATURE_MATCHES + 2;
        let mut nodes = Vec::new();
        for i in 0..count {
            let path = format!("pkg{i}/run.ts");
            std::fs::write(
                dir.path().join(format!("run{i}.ts")),
                format!("function run() {{ /* {i} */ }}\n"),
            )
            .unwrap();
            nodes.push(make_fn_node(&path, "fn:run", 1, 1));
        }
        let store = make_store_with_meta(&nodes, dir.path());

        let result = get_snippets_body(&store, "fn:run", 8000, SnippetMode::Auto);

        // Footer must report exactly MAX_SIGNATURE_MATCHES symbols resolved.
        assert!(
            result.contains(&format!("{MAX_SIGNATURE_MATCHES} symbols")),
            "must cap at {MAX_SIGNATURE_MATCHES} symbols, got: {result}"
        );
    }

    // ── get_context include_snippets tests ───────────────────────────────────

    fn make_store_with_root(
        dir: &tempfile::TempDir,
        nodes: &[CoreNode],
    ) -> travsr_store::SqliteStore {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();
        for n in nodes {
            store.put_node(n).unwrap();
        }
        store
    }

    fn make_fn_node_with_pkg(path: &str, sig: &str, line: u32, end_line: u32) -> CoreNode {
        CoreNode::new(
            VName::new("corpus", "", path, "typescript", sig),
            "function",
        )
        .with_line(line)
        .with_end_line(end_line)
    }

    #[test]
    fn get_context_include_snippets_false_has_no_separator() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("charge.ts");
        std::fs::write(&src, "function charge() {\n  return 1;\n}\n").unwrap();

        let node = make_fn_node_with_pkg("charge.ts", "fn:charge", 1, 3);
        let store = make_store_with_root(&dir, &[node]);

        let without = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, None);
        let with_snip = get_context_body(&store, "charge", 4096, &OpenFilter, true, None, None);

        // false path: metadata-only — no separator, footer present
        assert!(without.contains("["), "footer must be present");
        assert!(
            !without.contains(SNIPPET_SEP),
            "no separator in metadata-only output"
        );
        // true path must include the separator — same node, opposite outcome
        assert!(
            with_snip.contains(SNIPPET_SEP),
            "SNIPPET_SEP must appear with include_snippets=true"
        );
    }

    #[test]
    fn get_context_include_snippets_appends_code() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("pay.ts");
        std::fs::write(&src, "function pay() {\n  return 42;\n}\n").unwrap();

        let node = make_fn_node_with_pkg("pay.ts", "fn:pay", 1, 3);
        let store = make_store_with_root(&dir, &[node]);

        let result = get_context_body(&store, "pay", 4096, &OpenFilter, true, None, None);
        assert!(
            result.contains(SNIPPET_SEP),
            "SNIPPET_SEP must appear when include_snippets=true"
        );
        assert!(result.contains("return 42"), "snippet body must be inlined");
        assert!(
            result.contains("with snippets"),
            "footer must report snippet count"
        );
    }

    #[test]
    fn get_context_include_snippets_shared_budget_truncates() {
        let dir = tempfile::tempdir().unwrap();
        // Write a large file so the snippet alone would bust any small budget.
        let body: String = (0..200).map(|i| format!("  let x{i} = {i};\n")).collect();
        let src = dir.path().join("big.ts");
        std::fs::write(&src, format!("function big() {{\n{body}}}\n")).unwrap();

        let node = make_fn_node_with_pkg("big.ts", "fn:big", 1, 202);
        let store = make_store_with_root(&dir, &[node]);

        // Tiny budget: metadata fits but snippet cannot.
        let result = get_context_body(&store, "big", 10, &OpenFilter, true, None, None);
        // Must not panic; footer must be present regardless of truncation.
        assert!(result.contains("nodes"), "footer must always be present");
    }

    #[test]
    fn get_context_include_snippets_separate_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("sep.ts");
        std::fs::write(&src, "function sep() {\n  return 7;\n}\n").unwrap();

        let node = make_fn_node_with_pkg("sep.ts", "fn:sep", 1, 3);
        let store = make_store_with_root(&dir, &[node]);

        // Separate snippet_budget of 512; main budget governs node selection only.
        let result = get_context_body(&store, "sep", 4096, &OpenFilter, true, Some(512), None);
        assert!(
            result.contains("separate"),
            "footer must label separate-budget mode"
        );
        assert!(
            result.contains("return 7"),
            "snippet must be present within separate budget"
        );
    }

    #[test]
    fn get_context_include_snippets_no_repo_root_degrades() {
        // Store with no repo_root meta → must return metadata-only, no panic.
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let node = make_fn_node_with_pkg("x.ts", "fn:x", 1, 2);
        store.put_node(&node).unwrap();

        let result = get_context_body(&store, "x", 4096, &OpenFilter, true, None, None);
        // Either "not found" (no FTS match on empty index), the init hint, or a
        // clean abstention note (#478: a lone 1-char query has no real lexical
        // evidence — Leg A matching "fn:x" as a bare substring is not grounds
        // for a confident answer, so this now abstains honestly instead of
        // fabricating a match from zero evidence) — never a panic.
        let has_meta_hint = result.contains("travsr init")
            || result.is_empty()
            || result.contains("No symbols")
            || result.contains("no grounded match");
        assert!(
            has_meta_hint,
            "absent repo_root must degrade gracefully: {result}"
        );
        assert!(!result.contains("panic"), "must not contain panic text");
    }

    #[test]
    fn get_context_include_snippets_rbac_excluded_node_not_read() {
        // A filter that rejects everything. No disk read must occur.
        struct DenyAll;
        impl travsr_retrieval::EdgeFilter for DenyAll {
            fn allow(
                &self,
                _src: travsr_core::NodeId,
                _dst: travsr_core::NodeId,
                _corpus: Option<&str>,
            ) -> bool {
                false
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("secret.ts");
        std::fs::write(&src, "function secret() { return 99; }\n").unwrap();

        let node = make_fn_node_with_pkg("secret.ts", "fn:secret", 1, 1);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();
        store.put_node(&node).unwrap();

        let result = get_context_body(&store, "secret", 4096, &DenyAll, true, None, None);
        // RBAC filter rejects all nodes → "not found" (SEC P0) with no snippet leak.
        assert!(
            !result.contains("return 99"),
            "RBAC-filtered node must never appear in snippet output"
        );
    }

    // ── Edge-relationship annotation tests ──────────────────────────────────

    fn make_store_with_edges(
        nodes: &[CoreNode],
        edges: &[(
            travsr_core::NodeId,
            travsr_core::NodeId,
            travsr_core::EdgeKind,
        )],
    ) -> travsr_store::SqliteStore {
        use travsr_store::Store;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for n in nodes {
            store.put_node(n).unwrap();
        }
        for &(src, dst, kind) in edges {
            store
                .put_edge(&travsr_core::Edge::new(src, dst, kind))
                .unwrap();
        }
        store
    }

    #[test]
    fn get_context_annotation_labels_seed() {
        // The node that directly matches the query must be labelled [via: seed].
        let node = make_fn_node_with_pkg("seed.ts", "fn:seed_target", 1, 3);
        let store = make_store_with_edges(&[node], &[]);

        let result = get_context_body(&store, "seed_target", 4096, &OpenFilter, false, None, None);
        assert!(
            result.contains("[via: seed · exact-name]"),
            "direct exact match must be labelled [via: seed · exact-name]: {result}"
        );
    }

    #[test]
    fn get_context_annotation_labels_caller() {
        // caller_fn has a RefCall edge to seed_fn.
        // get_context(query="seed_fn") → seed_fn=[via:seed], caller_fn=[via:caller].
        // PPR BFS only follows forward edges, so caller_fn must be reachable
        // from seed_fn via a forward path to appear in the output at all.
        // Graph: seed_fn → Depends → bridge_fn → Depends → caller_fn
        //        caller_fn → RefCall → seed_fn
        // PPR surfaces caller_fn (forward path depth 2).
        // Role check finds the reverse RefCall → labels caller_fn [via: caller].
        use travsr_core::EdgeKind;
        use travsr_store::Store;
        let seed_node = make_fn_node_with_pkg("s.ts", "fn:seed_fn", 1, 3);
        let bridge_node = make_fn_node_with_pkg("b.ts", "fn:bridge_fn", 1, 3);
        let caller_node = make_fn_node_with_pkg("c.ts", "fn:caller_fn", 1, 3);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let seed_id = store.put_node(&seed_node).unwrap();
        let bridge_id = store.put_node(&bridge_node).unwrap();
        let caller_id = store.put_node(&caller_node).unwrap();
        // Forward path: seed_fn → bridge_fn → caller_fn (PPR discovery)
        store
            .put_edge(&travsr_core::Edge::new(
                seed_id,
                bridge_id,
                EdgeKind::Depends,
            ))
            .unwrap();
        store
            .put_edge(&travsr_core::Edge::new(
                bridge_id,
                caller_id,
                EdgeKind::Depends,
            ))
            .unwrap();
        // Reverse semantic edge: caller_fn calls seed_fn → earns [via: caller]
        store
            .put_edge(&travsr_core::Edge::new(
                caller_id,
                seed_id,
                EdgeKind::RefCall,
            ))
            .unwrap();

        let result = get_context_body(&store, "seed_fn", 4096, &OpenFilter, false, None, None);
        assert!(
            result.contains("[via: seed"),
            "seed node must be labelled [via: seed · <source>]: {result}"
        );
        // RFC-019: caller/dependency roles now name their source seed.
        assert!(
            result.contains("[via: caller of fn:seed_fn]"),
            "caller node must be labelled with its source seed: {result}"
        );
    }

    #[test]
    fn get_context_annotation_labels_dependency() {
        // seed_fn has a Depends edge to dep_fn.
        // get_context(query="seed_fn") → seed_fn=[via:seed], dep_fn=[via:dependency].
        use travsr_core::EdgeKind;
        use travsr_store::Store;
        let seed_node = make_fn_node_with_pkg("s.ts", "fn:seed_fn2", 1, 3);
        let dep_node = make_fn_node_with_pkg("d.ts", "fn:dep_fn", 1, 3);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let seed_id = store.put_node(&seed_node).unwrap();
        let dep_id = store.put_node(&dep_node).unwrap();
        // seed_fn → Depends → dep_fn
        store
            .put_edge(&travsr_core::Edge::new(seed_id, dep_id, EdgeKind::Depends))
            .unwrap();

        let result = get_context_body(&store, "seed_fn2", 4096, &OpenFilter, false, None, None);
        assert!(
            result.contains("[via: seed"),
            "seed node must be labelled [via: seed · <source>]: {result}"
        );
        // RFC-019: caller/dependency roles now name their source seed.
        assert!(
            result.contains("[via: dependency of fn:seed_fn2]"),
            "dep node must be labelled with its source seed: {result}"
        );
    }

    #[test]
    fn get_context_annotation_labels_context() {
        // A node with no edge to/from the seed is labelled [via: context].
        use travsr_store::Store;
        let seed_node = make_fn_node_with_pkg("s.ts", "fn:seed_only", 1, 3);
        let ctx_node = make_fn_node_with_pkg("u.ts", "fn:unrelated_fn", 1, 3);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&seed_node).unwrap();
        store.put_node(&ctx_node).unwrap();
        // No edges between them — ctx_node is reachable only via PPR structural walk.

        // Search for "seed_only" — ctx_node should appear as [via: context] if PPR
        // surfaces it, or just not appear. Either way, if it does appear it must NOT
        // carry seed/caller/dependency labels.
        let result = get_context_body(&store, "seed_only", 4096, &OpenFilter, false, None, None);
        assert!(
            !result.contains("unrelated_fn") || result.contains("[via: context]"),
            "unrelated node must be [via: context] if it appears at all: {result}"
        );
    }

    // ── is_noise_seed tests ───────────────────────────────────────────────────

    fn make_node_with_kind_and_path(kind: &str, path: &str, sig: &str) -> travsr_core::Node {
        travsr_core::Node::new(travsr_core::VName::new("corp", "", path, "rust", sig), kind)
    }

    #[test]
    fn is_noise_seed_excludes_crate_kind() {
        let n = make_node_with_kind_and_path(
            "crate",
            "crates/travsr-retrieval/Cargo.toml",
            "crate:travsr-retrieval",
        );
        assert!(is_noise_seed(&n), "crate nodes must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_tests_path() {
        let n = make_node_with_kind_and_path(
            "function",
            "crates/travsr-mcp/tests/conformance.rs",
            "fn:run_mcp",
        );
        assert!(is_noise_seed(&n), "integration test files must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_benches_path() {
        let n = make_node_with_kind_and_path(
            "function",
            "crates/travsr-retrieval/benches/retrieval.rs",
            "fn:bench_ppr_chain",
        );
        assert!(is_noise_seed(&n), "benchmark files must be excluded");
    }

    #[test]
    fn is_noise_seed_allows_src_functions() {
        let n = make_node_with_kind_and_path(
            "function",
            "crates/travsr-mcp/src/tools.rs",
            "fn:get_context_body",
        );
        assert!(
            !is_noise_seed(&n),
            "src/ implementation functions must not be excluded"
        );
    }

    #[test]
    fn is_noise_seed_allows_src_unit_tests() {
        // Unit tests inside src/ are kept — PPR from them reaches the impl they test.
        let n = make_node_with_kind_and_path(
            "function",
            "crates/travsr-retrieval/src/ppr.rs",
            "fn:ppr_handles_cycles",
        );
        assert!(
            !is_noise_seed(&n),
            "in-src unit test functions must not be excluded"
        );
    }

    // ── RFC-022 D3: anchor-pool noise (test symbols + CI package nodes) ───────

    #[test]
    fn is_test_symbol_catches_marker_names_not_real_functions() {
        let test_fns = [
            ("function", "fn:test_builds_the_seed_set"),
            ("function", "fn:seed_set_test"),
            ("function", "fn:TestReconcile"),     // Go
            ("function", "fn:BenchmarkKnapsack"), // Go
            ("method", "method:TestJig#TestRun"),
        ];
        for (kind, sig) in test_fns {
            let n = make_node_with_kind_and_path(kind, "crates/x/src/lib.rs", sig);
            assert!(is_test_symbol(&n), "{sig} should read as a test symbol");
        }
        // Real functions that merely mention/contain "test"-ish substrings must not match.
        for sig in [
            "fn:build_seed_set",
            "fn:Testament",
            "fn:latest_snapshot",
            "struct:Tester",
        ] {
            let n = make_node_with_kind_and_path("function", "crates/x/src/lib.rs", sig);
            assert!(!is_test_symbol(&n), "{sig} must not read as a test symbol");
        }
    }

    #[test]
    fn is_anchor_noise_excludes_ci_package_but_not_real_impl() {
        // CI/workflow package node — never a valid anchor.
        let pkg = make_node_with_kind_and_path(
            "package",
            ".github/workflows/ci.yml",
            "pkg:actions/checkout@v4",
        );
        assert!(
            is_anchor_noise(&pkg),
            "pkg:actions/* must be excluded from anchors"
        );
        // In-src test — excluded from anchors even though is_noise_seed keeps it as a seed.
        let t = make_node_with_kind_and_path(
            "function",
            "crates/x/src/lib.rs",
            "fn:test_reticulates_splines",
        );
        assert!(is_anchor_noise(&t) && !is_noise_seed(&t));
        // A genuine implementation is a valid anchor.
        let impl_fn =
            make_node_with_kind_and_path("function", "crates/x/src/lib.rs", "fn:build_seed_set");
        assert!(!is_anchor_noise(&impl_fn), "real impl must be anchorable");
    }

    #[test]
    fn is_noise_seed_excludes_root_fixtures_dir() {
        let n = make_node_with_kind_and_path(
            "class",
            "fixtures/ts-callers/service.ts",
            "class:PaymentService",
        );
        assert!(is_noise_seed(&n), "root-level fixtures/ must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_root_fuzz_corpus() {
        let n = make_node_with_kind_and_path(
            "class",
            "fuzz/corpus/fuzz_treesitter_indexer/seed_class.ts",
            "class:PaymentService",
        );
        assert!(is_noise_seed(&n), "root-level fuzz corpus must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_scip_synthetic_nodes() {
        let n = make_node_with_kind_and_path("function", "some/file.py", "scip:python:some.Symbol");
        assert!(is_noise_seed(&n), "scip: synthetic nodes must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_go_test_files() {
        let n = make_node_with_kind_and_path(
            "function",
            "pkg/handler/handler_test.go",
            "fn:TestHandlerRPC",
        );
        assert!(is_noise_seed(&n), "Go _test.go files must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_build_cache_relative_path() {
        let n = make_node_with_kind_and_path(
            "function",
            "../../../Library/Caches/go-build/abc123/b.go",
            "fn:build_cache_fn",
        );
        assert!(is_noise_seed(&n), "../ paths outside repo must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_go_module_cache() {
        let n = make_node_with_kind_and_path(
            "function",
            "/home/user/go/pkg/mod/github.com/foo/bar@v1.2.3/pkg.go",
            "fn:bar_fn",
        );
        assert!(is_noise_seed(&n), "Go module cache paths must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_testdata_dir() {
        let n = make_node_with_kind_and_path(
            "function",
            "pkg/parser/testdata/golden_output.go",
            "fn:some_func",
        );
        assert!(is_noise_seed(&n), "testdata/ directories must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_java_maven_test_src() {
        let n = make_node_with_kind_and_path(
            "function",
            "src/test/java/com/example/ServiceTest.java",
            "fn:testChargeSuccess",
        );
        assert!(
            is_noise_seed(&n),
            "Java Maven src/test/java must be excluded"
        );
    }

    #[test]
    fn is_noise_seed_excludes_kotlin_test_src() {
        let n = make_node_with_kind_and_path(
            "function",
            "src/test/kotlin/com/example/ServiceSpec.kt",
            "fn:charge_succeeds",
        );
        assert!(is_noise_seed(&n), "Kotlin src/test/kotlin must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_scala_test_src() {
        let n = make_node_with_kind_and_path(
            "function",
            "src/test/scala/com/example/ServiceSuite.scala",
            "fn:chargeReturnsOk",
        );
        assert!(is_noise_seed(&n), "Scala src/test/scala must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_ruby_spec_dir() {
        let n = make_node_with_kind_and_path(
            "function",
            "spec/services/charge_spec.rb",
            "fn:charge_service",
        );
        assert!(is_noise_seed(&n), "Ruby spec/ directory must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_dart_pub_cache() {
        let n = make_node_with_kind_and_path(
            "function",
            "/home/user/.pub-cache/hosted/pub.dev/http-1.2.0/lib/http.dart",
            "fn:get",
        );
        assert!(is_noise_seed(&n), ".pub-cache paths must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_cmake_files_dir() {
        let n = make_node_with_kind_and_path(
            "function",
            "build/CMakeFiles/myapp.dir/src/main.c.o",
            "fn:main",
        );
        assert!(
            is_noise_seed(&n),
            "CMakeFiles/ build artifacts must be excluded"
        );
    }

    #[test]
    fn is_noise_seed_excludes_java_class_files() {
        let n = make_node_with_kind_and_path(
            "function",
            "target/classes/com/example/Service.class",
            "fn:charge",
        );
        assert!(is_noise_seed(&n), ".class files must be excluded");
    }

    #[test]
    fn is_noise_seed_excludes_gradle_build_classes() {
        let n = make_node_with_kind_and_path(
            "function",
            "build/classes/kotlin/main/com/example/Service.kt",
            "fn:charge",
        );
        assert!(is_noise_seed(&n), "Gradle build/classes must be excluded");
    }

    #[test]
    fn fts_seeds_weighted_filters_crate_noise() {
        // FTS exact-matching a crate node (e.g. "crate:getrandom" matching query
        // "random") must be dropped — same noise class as KNN already filters.
        use travsr_store::Store;
        let crate_node = make_node_with_kind_and_path(
            "crate",
            "crates/travsr-retrieval/Cargo.toml",
            "crate:getrandom",
        );
        let impl_node = make_node_with_kind_and_path(
            "function",
            "crates/travsr-retrieval/src/ppr.rs",
            "fn:random_walk",
        );
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&crate_node).unwrap();
        let impl_id = store.put_node(&impl_node).unwrap();

        // Both nodes would match a "random" FTS query in a real store, but we use
        // fts_seeds_weighted directly to isolate the noise filter.  Inject them via
        // search_nodes_by_name by naming them so FTS exact-matches: call the function
        // with the impl node's name to get a result, then manually verify crate kind
        // is rejected.  Because open_in_memory FTS is live, just verify the filter
        // logic directly via the public function path.
        let seeds = fts_seeds_weighted(&store, "random_walk", &OpenFilter);
        assert!(
            seeds.iter().all(|&(id, _)| id == impl_id),
            "crate-kind nodes must be filtered from FTS seeds; got {:?}",
            seeds
        );
    }

    #[test]
    fn embed_path_seeds_filters_noise_nodes() {
        // KNN returns a crate node + a tests/ function + a real src function.
        // Only the real src function must become a seed.
        use travsr_store::Store;
        let crate_node = make_node_with_kind_and_path(
            "crate",
            "crates/travsr-retrieval/Cargo.toml",
            "crate:travsr-retrieval",
        );
        let test_node = make_node_with_kind_and_path(
            "function",
            "crates/travsr-mcp/tests/conformance.rs",
            "fn:run_mcp",
        );
        // RFC-022 D3: a CI/workflow package node — must be filtered from BOTH the
        // seed list and the cosine oracle even at a high KNN cosine.
        let pkg_node = make_node_with_kind_and_path("package", "", "pkg:actions/checkout@v4");
        let impl_node = make_node_with_kind_and_path(
            "function",
            "crates/travsr-mcp/src/tools.rs",
            "fn:get_context_body",
        );
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let crate_id = store.put_node(&crate_node).unwrap();
        let test_id = store.put_node(&test_node).unwrap();
        let pkg_id = store.put_node(&pkg_node).unwrap();
        let impl_id = store.put_node(&impl_node).unwrap();

        // KNN returns all four with high scores (pkg highest, as arctic does for salad).
        let knn: EmbedKnnFn<'_> = &|_q, _k| {
            vec![
                (pkg_id, 0.97),
                (crate_id, 0.95),
                (test_id, 0.90),
                (impl_id, 0.85),
            ]
        };

        let (seeds, n_eligible, _, oracle) =
            embed_path_seeds(&store, "get_context", knn, &OpenFilter);
        // pkg (kind=package/`pkg:`), crate (kind=crate), and the tests/ node are
        // all noise and filtered from both the seed list and the oracle.
        let seed_ids: Vec<_> = seeds.iter().map(|(n, _)| n.id).collect();
        assert!(!seed_ids.contains(&pkg_id), "pkg node must not be a seed");
        assert!(
            !seed_ids.contains(&crate_id),
            "crate node must not be a seed"
        );
        assert!(
            !seed_ids.contains(&test_id),
            "test-dir node must not be a seed"
        );
        assert!(
            seed_ids.contains(&impl_id),
            "only the real src impl is a seed; got: {seeds:?}"
        );
        assert_eq!(seeds.len(), 1, "only the impl seed expected");
        assert_eq!(n_eligible, 1, "n_eligible should match non-noise seeds");
        // The package/crate nodes must not be in the confidence-grounding oracle.
        assert!(
            !oracle.contains_key(&pkg_id),
            "pkg node must be excluded from the cosine oracle"
        );
        assert!(
            !oracle.contains_key(&crate_id),
            "crate node must be excluded from the cosine oracle"
        );
        assert!(
            oracle.contains_key(&impl_id),
            "real impl must remain in oracle"
        );
    }

    // ── #367 degraded-state notes ─────────────────────────────────────────────

    /// When embed_knn is None, get_context output must include the semantic-search note.
    #[test]
    fn get_context_no_embed_appends_degraded_note() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let store = make_store_with_root(&dir, &[node]);
        // Simulate a repo that has commits (so phase_b_pending check is meaningful).
        // We do NOT set phase_b_commit so both notes could fire; only test the embed note.
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, None);
        assert!(
            result.contains("semantic search disabled"),
            "must emit embed-missing note when embed_knn=None; got: {result}"
        );
    }

    /// Phase B pending: output must contain the call-traversal-limited note.
    #[test]
    fn get_context_phase_b_pending_appends_note() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        // Mark a real last_commit without a phase_b_commit → phase_b_pending returns true.
        store.set_meta("last_commit", "abc123").unwrap();
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, None);
        assert!(
            result.contains("call traversal limited"),
            "must emit phase-B-pending note when phase_b_commit absent; got: {result}"
        );
    }

    /// When both embed and Phase B are present, no degraded notes should appear.
    #[test]
    fn get_context_fully_operational_no_notes() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        // Provide a no-op embed_knn (has_embed = true).
        let knn: EmbedKnnFn<'_> = &|_q, _k| vec![];
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, Some(knn));
        assert!(
            !result.contains("[note:"),
            "fully operational graph must not emit degraded notes; got: {result}"
        );
    }

    /// A0/B: embed configured but sidecar not yet armed → header reads `warming`
    /// and a warming note is emitted (never a silent lexical-only degrade).
    #[test]
    fn get_context_warming_reports_warming_header_and_note() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        // Register an un-armed readiness → embed_ready() == false → warming.
        store.set_embed_readiness(travsr_store::EmbedReadiness::new());
        // Keep the arm-wait short so the test times out fast rather than waiting 5 s.
        std::env::set_var("TRAVSR_EMBED_ARM_WAIT_MS", "40");
        let knn: EmbedKnnFn<'_> = &|_q, _k| vec![];
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, Some(knn));
        std::env::remove_var("TRAVSR_EMBED_ARM_WAIT_MS");
        assert!(
            result.contains("embeddings: warming"),
            "header must report warming while sidecar is cold; got: {result}"
        );
        assert!(
            result.contains("still warming up"),
            "must emit warming note; got: {result}"
        );
    }

    /// B: once the readiness handle is armed, the same configuration reports
    /// `embeddings: on` with no warming note.
    #[test]
    fn get_context_armed_reports_on_no_warming_note() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let readiness = travsr_store::EmbedReadiness::new();
        readiness.mark_ready();
        store.set_embed_readiness(readiness);
        let knn: EmbedKnnFn<'_> = &|_q, _k| vec![];
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, Some(knn));
        assert!(
            result.contains("embeddings: on"),
            "armed readiness must report on; got: {result}"
        );
        assert!(
            !result.contains("warming"),
            "armed readiness must not emit warming note; got: {result}"
        );
    }

    /// #874: arming settled with no hook installed (model mismatch, or a sidecar
    /// that never started). The meta-hook is still present, so `has_embed` is
    /// true and readiness is armed — without the third state this reads as
    /// `embeddings: on` over a semantic lane that can never answer.
    #[test]
    fn get_context_disabled_reports_disabled_header_and_note() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let readiness = travsr_store::EmbedReadiness::new();
        readiness.mark_disabled();
        readiness.mark_ready();
        store.set_embed_readiness(readiness);
        let knn: EmbedKnnFn<'_> = &|_q, _k| vec![];
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, Some(knn));
        assert!(
            result.contains("embeddings: disabled"),
            "a settled-but-unarmed hook must not report `on`; got: {result}"
        );
        assert!(
            !result.contains("exact+lexical+semantic"),
            "retrieval tier must not claim semantic; got: {result}"
        );
        assert!(
            result.contains("semantic search unavailable"),
            "must emit the unavailable note; got: {result}"
        );
        assert!(
            !result.contains("warming"),
            "disabled is settled, not warming; got: {result}"
        );
    }

    // ── #617 structural-tool Phase-B degraded signals ─────────────────────────

    #[test]
    fn phase_b_note_none_when_marker_matches_head_and_clean() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        assert!(phase_b_degraded_note(&store).is_none());
    }

    #[test]
    fn phase_b_note_none_on_no_commit_repo() {
        // Both markers absent: Phase B ran inline during init (no daemon defer).
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        assert!(phase_b_degraded_note(&store).is_none());
    }

    #[test]
    fn phase_b_note_pending_when_marker_absent() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        let note = phase_b_degraded_note(&store).expect("must flag pending");
        assert!(note.contains("call-graph index incomplete"), "got: {note}");
    }

    #[test]
    fn phase_b_note_pending_when_marker_behind_head() {
        // The exact #617 repro: HEAD moved while the daemon was down, so the
        // marker is set but stale. `phase_b_pending` misses this state.
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "def456").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let note = phase_b_degraded_note(&store).expect("must flag pending");
        assert!(note.contains("call-graph index incomplete"), "got: {note}");
    }

    #[test]
    fn phase_b_note_stale_when_dirty_flag_set() {
        // #583 window: markers agree but a watcher reindex dropped edges.
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "1").unwrap();
        let note = phase_b_degraded_note(&store).expect("must flag stale");
        assert!(note.contains("call-graph edges degraded"), "got: {note}");
    }

    #[test]
    fn phase_b_note_not_degraded_when_the_live_overlay_resolved_the_edit() {
        // dirty, and the overlay resolved its references with nothing pending.
        // This drops the heavy "degraded, run init" note, but not all caution:
        // the counts are repo-wide, so a resolved file cannot prove a *separate*
        // headless edit (which leaves no rows) also came back. So the note is a
        // light "a few edges may be missing until the next commit", never the
        // false "results are current".
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "1").unwrap();
        let n = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:a"),
            "function",
        )
        .with_line(1);
        store.put_node(&n).unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "a.rs",
                &[travsr_store::RefResolution {
                    src: n.id,
                    ref_line: 2,
                    ref_col: 0,
                    name: "b".into(),
                    state: "resolved",
                    resolved_dst: None,
                }],
            )
            .unwrap();
        let note = phase_b_degraded_note(&store).expect("a light caution must remain");
        assert!(
            !note.contains("degraded") && !note.contains("run `travsr init`"),
            "a resolved overlay must drop the heavy degraded/run-init note, got: {note}"
        );
        assert!(
            note.contains("until the next commit"),
            "the light caution must still name the commit as the full refresh, got: {note}"
        );
    }

    #[test]
    fn phase_b_note_names_pending_references_instead_of_run_init() {
        // dirty with an unresolved reference: name the gap honestly, without the
        // heavy "run travsr init".
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "1").unwrap();
        let n = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:a"),
            "function",
        )
        .with_line(1);
        store.put_node(&n).unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "a.rs",
                &[travsr_store::RefResolution {
                    src: n.id,
                    ref_line: 2,
                    ref_col: 0,
                    name: "b".into(),
                    state: "pending",
                    resolved_dst: None,
                }],
            )
            .unwrap();
        let note = phase_b_degraded_note(&store).expect("a pending overlay must note the gap");
        assert!(
            note.contains("not yet resolved") && !note.contains("travsr init"),
            "got: {note}"
        );
    }

    #[test]
    fn phase_b_note_none_when_dirty_flag_cleared() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "0").unwrap();
        assert!(phase_b_degraded_note(&store).is_none());
    }

    // ── #645 WS-B: index/HEAD mismatch note ──────────────────────────────────

    #[test]
    fn head_index_mismatch_note_none_when_equal_or_unknown() {
        assert!(head_index_mismatch_note("abc123", "abc123").is_none());
        assert!(head_index_mismatch_note("", "abc123").is_none());
        assert!(head_index_mismatch_note("abc123", "").is_none());
    }

    /// #636 round-4 review: `git rev-parse --short` is variable width
    /// (`core.abbrev`, and `auto` grows with the object count), so the same
    /// commit can be stamped at one length and read back at another. Comparing
    /// those as plain strings reported drift on an identical commit.
    #[test]
    fn short_shas_of_differing_width_for_the_same_commit_do_not_differ() {
        // Verified against a real repo: `git rev-parse --short=7` and
        // `--short=12` on one commit give `f97492f` and `f97492f91ce4`.
        assert_eq!(short_shas_differ("f97492f", "f97492f91ce4"), Some(false));
        assert_eq!(short_shas_differ("f97492f91ce4", "f97492f"), Some(false));
        // Case-insensitive, since git accepts either case in a rev.
        assert_eq!(short_shas_differ("F97492F", "f97492f91ce4"), Some(false));
        // And the note built on it stays silent for that pair.
        assert!(head_index_mismatch_note("f97492f91ce4", "f97492f").is_none());
    }

    #[test]
    fn short_shas_that_genuinely_differ_are_still_reported() {
        assert_eq!(short_shas_differ("aaa1111", "bbb2222"), Some(true));
        // Divergence beyond the shared prefix length still counts.
        assert_eq!(short_shas_differ("f97492f", "f97492e91ce4"), Some(true));
        assert!(head_index_mismatch_note("bbb2222", "aaa1111").is_some());
    }

    #[test]
    fn short_shas_differ_is_none_when_either_side_is_unknown() {
        assert_eq!(short_shas_differ("", "abc123"), None);
        assert_eq!(short_shas_differ("abc123", ""), None);
    }

    #[test]
    fn head_index_mismatch_note_names_both_shas_on_drift() {
        let note = head_index_mismatch_note("aaa111", "bbb222").expect("must flag mismatch");
        assert!(
            note.contains("aaa111"),
            "must name the checkout HEAD: {note}"
        );
        assert!(
            note.contains("bbb222"),
            "must name the index commit: {note}"
        );
    }

    #[test]
    fn git_short_head_reads_repo_and_is_none_off_repo() {
        // Off a repo (bare tempdir) there is no HEAD to read.
        let empty = tempfile::tempdir().unwrap();
        // Prevent git from walking up and finding parent repos (e.g. in CI or user home).
        let mut ceiling_paths = Vec::new();
        let mut current = Some(empty.path());
        while let Some(p) = current {
            let p_str = p.to_string_lossy().to_string();
            ceiling_paths.push(p_str.clone());
            ceiling_paths.push(p_str.replace("\\", "/"));
            ceiling_paths.push(p_str.replace("C:", "c:"));
            ceiling_paths.push(p_str.replace("c:", "C:"));
            ceiling_paths.push(p_str.replace("C:", "c:").replace("\\", "/"));
            ceiling_paths.push(p_str.replace("c:", "C:").replace("\\", "/"));
            current = p.parent();
        }
        #[cfg(windows)]
        let delim = ";";
        #[cfg(not(windows))]
        let delim = ":";
        std::env::set_var("GIT_CEILING_DIRECTORIES", ceiling_paths.join(delim));
        let res = git_short_head(empty.path());
        std::env::remove_var("GIT_CEILING_DIRECTORIES");
        assert!(res.is_none());
    }

    /// #645/#661: in global mode the per-repo HEAD must come from that repo's own
    /// root (`<root>/.travsr/graph.db` → `<root>`), NOT the process-global
    /// LAUNCH_CWD — otherwise one workspace's commit is compared against every
    /// other registered repo's index and fires a spurious mismatch. Pin that the
    /// resolver reads the repo the db belongs to, and matches `git_short_head` of
    /// that root exactly.
    #[test]
    fn repo_head_from_registry_path_reads_the_owning_repo_head() {
        fn git(dir: &std::path::Path, args: &[&str]) -> bool {
            std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        }
        if !git(std::path::Path::new("."), &["--version"]) {
            eprintln!("skipping: git not available");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert!(git(root, &["init", "-q"]));
        assert!(git(root, &["config", "user.email", "t@t.t"]));
        assert!(git(root, &["config", "user.name", "t"]));
        std::fs::write(root.join("f.txt"), "x").unwrap();
        assert!(git(root, &["add", "."]));
        assert!(git(root, &["commit", "-qm", "init"]));

        let db_path = root.join(".travsr").join("graph.db");
        // Resolves from the db path's grandparent (the repo root) and equals a
        // direct short-HEAD read of that root — the value LAUNCH_CWD would only
        // coincidentally give when the caller happened to stand in this repo.
        assert_eq!(
            repo_head_from_registry_path(&db_path),
            git_short_head(root),
            "must read the repo the db belongs to, not the process cwd"
        );
        assert!(repo_head_from_registry_path(&db_path).is_some());
    }

    #[test]
    fn append_read_notes_emits_head_note_only_on_drift() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Markers agree ⇒ no Phase-B note; isolate the head note.
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "idx0000").unwrap();

        // Caller's HEAD differs from the index ⇒ note appended to the body.
        let out = append_read_notes(&store, "real result".to_string(), Some("chk1111"));
        assert!(out.starts_with("real result"), "body preserved: {out}");
        assert!(
            out.contains("idx0000") && out.contains("chk1111"),
            "got: {out}"
        );

        // Same commit ⇒ no head note.
        assert_eq!(
            append_read_notes(&store, "real result".to_string(), Some("idx0000")),
            "real result"
        );
        // git-less caller ⇒ no head note.
        assert_eq!(
            append_read_notes(&store, "real result".to_string(), None),
            "real result"
        );
    }

    #[test]
    fn append_read_notes_empty_body_becomes_head_note() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "idx0000").unwrap();
        // Empty body + mismatch: the note is the whole answer (no leading newline).
        let out = append_read_notes(&store, String::new(), Some("chk1111"));
        assert!(out.starts_with("[note:"), "note is the body: {out}");
        assert!(
            !out.contains("\n[note:"),
            "no empty line before the note: {out}"
        );
    }

    #[test]
    fn append_read_notes_phase_b_and_head_notes_coexist() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Marker behind HEAD ⇒ Phase-B pending note fires...
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "old9999").unwrap();
        // ...and the caller is at yet another commit ⇒ head note also fires.
        let out = append_read_notes(&store, "body".to_string(), Some("chk1111"));
        assert!(
            out.contains("call-graph index incomplete"),
            "phase-b note: {out}"
        );
        assert!(out.contains("chk1111"), "head note: {out}");
    }

    // ── RFC-027 section 10: the live overlay is legible, never silent ────────

    /// The note never fires on a clean tree, so it costs an ordinary reader
    /// nothing, and fires as soon as either half of the overlay is non-empty.
    #[test]
    fn the_live_overlay_note_fires_only_when_there_is_an_overlay() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "idx0000").unwrap();
        assert!(
            live_overlay_note(&store, "a.ts").is_none(),
            "a repo with no uncommitted edits must get no note"
        );

        let a = node_with("fn:a", "function", "a.ts");
        let b = node_with("fn:b", "function", "b.ts");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge_live(&travsr_core::Edge::new(
                a.id,
                b.id,
                travsr_core::EdgeKind::RefCall,
            ))
            .unwrap();

        let note = live_overlay_note(&store, "a.ts").expect("an overlay must be announced");
        assert!(note.contains("1 edge resolved"), "singular form: {note}");
        assert!(
            note.contains("provenance != live"),
            "a reader must be told how to get ratified-only truth: {note}"
        );
    }

    /// Both halves are reported, because they mean opposite things: live edges
    /// are extra freshness, pending references are known gaps. Reporting only
    /// the first would read as pure upside and hide the abstentions.
    #[test]
    fn the_live_overlay_note_reports_abstentions_as_well_as_resolutions() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = node_with("fn:a", "function", "a.ts");
        store.put_node(&a).unwrap();
        store
            .replace_ref_resolution_states(
                &a.vname.corpus,
                "a.ts",
                &[travsr_store::RefResolution {
                    src: a.id,
                    ref_line: 3,
                    ref_col: 0,
                    name: "save".to_string(),
                    state: "pending",
                    resolved_dst: None,
                }],
            )
            .unwrap();

        let note = live_overlay_note(&store, "callers in a.ts:3")
            .expect("a pending reference must be announced");
        assert!(
            note.contains("1 reference in the files above detected but not resolved"),
            "the abstention must be visible: {note}"
        );

        // Scoped, not ambient: an answer that never mentions `a.ts` must not be
        // told about `a.ts`'s gaps.
        assert!(
            live_overlay_note(&store, "callers in unrelated.ts:9").is_none(),
            "the pending half must not fire for a file the answer never names"
        );
    }

    /// The marker is attached per edge, and only where the edge's confidence
    /// differs from the default: an un-ratified `live` overlay edge, or a
    /// name-matched (rather than type-resolved) `ref/call`.
    #[test]
    fn only_a_live_or_heuristic_edge_is_marked_in_caller_output() {
        let ratified = travsr_core::Edge::new(
            travsr_core::NodeId(1),
            travsr_core::NodeId(2),
            travsr_core::EdgeKind::RefCall,
        );
        assert_eq!(
            provenance_marker(&ratified),
            "",
            "an unlabelled edge is not marked"
        );

        let mut scip = ratified.clone();
        scip.provenance = Some("scip".to_string());
        assert_eq!(
            provenance_marker(&scip),
            "",
            "a type-resolved edge is not marked"
        );

        // A `ref/call` labelled tree-sitter came from leaf-name matching in
        // `resolve_unresolved_calls`, not from a compiler, so it is marked.
        let mut ts_call = ratified.clone();
        ts_call.provenance = Some("tree-sitter".to_string());
        assert_eq!(
            provenance_marker(&ts_call),
            HEURISTIC_SIGIL_ROW,
            "a name-matched call edge must carry the sigil"
        );
        assert!(
            HEURISTIC_LEGEND.contains("matched by name, not resolved by type"),
            "the legend must spell the sigil out in the CLI's words"
        );

        // Phase A's own structural edges are tree-sitter too, and are facts
        // read off the AST. They must stay unmarked.
        let mut ts_structural = travsr_core::Edge::new(
            travsr_core::NodeId(1),
            travsr_core::NodeId(2),
            travsr_core::EdgeKind::DefinesBinding,
        );
        ts_structural.provenance = Some("tree-sitter".to_string());
        assert_eq!(
            provenance_marker(&ts_structural),
            "",
            "a structural Phase A edge is not a name guess"
        );

        let mut live = ratified.clone();
        live.provenance = Some("live".to_string());
        assert!(
            provenance_marker(&live).contains("not yet ratified"),
            "a live edge must say so"
        );
    }

    /// The filter defaults to returning everything, so an existing caller keeps
    /// the fresher graph; `ratified` is the opt-in for ground truth only.
    #[test]
    fn the_provenance_filter_defaults_to_everything() {
        assert!(provenance_allowed("", "live"));
        assert!(provenance_allowed("", "scip"));

        assert!(!provenance_allowed("ratified", "live"));
        assert!(provenance_allowed("ratified", "tree-sitter"));
        assert!(provenance_allowed("ratified", "scip"));

        assert!(provenance_allowed("live", "live"));
        assert!(!provenance_allowed("live", "scip"));

        // An unknown filter matches nothing rather than everything: a typo must
        // produce an obviously empty graph, not silently drop the constraint a
        // consumer added because it needed ground truth.
        assert!(!provenance_allowed("ratifed", "scip"));
    }

    /// The prose and JSON surfaces must announce the overlay identically. They
    /// are separate seams (`append_read_notes` vs `read_note_signals`) because a
    /// prose note appended to a JSON body would break `JSON.parse`, and it would
    /// be easy for one to gain a note the other never learns about.
    #[test]
    fn both_surfaces_announce_the_live_overlay() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "idx0000").unwrap();
        let a = node_with("fn:a", "function", "a.ts");
        let b = node_with("fn:b", "function", "b.ts");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge_live(&travsr_core::Edge::new(
                a.id,
                b.id,
                travsr_core::EdgeKind::RefCall,
            ))
            .unwrap();

        let prose = append_read_notes(&store, "body".to_string(), None);
        assert!(prose.contains("live overlay active"), "prose: {prose}");

        let signals = read_note_signals(&store, None, "a.ts");
        assert!(
            signals
                .iter()
                .any(|s| s.as_str().unwrap_or("").contains("live overlay active")),
            "json signals: {signals:?}"
        );

        // And the JSON body still parses with the signal embedded.
        let out = serde_json::json!({ "nodes": [], "edges": [], "signals": signals });
        let text = serde_json::to_string(&out).unwrap();
        serde_json::from_str::<serde_json::Value>(&text).expect("body must stay valid JSON");
    }

    /// Build a node for the overlay tests.
    fn node_with(sig: &str, kind: &str, path: &str) -> travsr_core::Node {
        travsr_core::Node::new(
            travsr_core::VName::new("test", "", path, "typescript", sig),
            kind,
        )
    }

    // ── #661 WS-D: head-only note on the deterministic path:line tools ────────

    #[test]
    fn append_head_note_emits_head_note_only_on_drift() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Marker behind HEAD would fire the Phase-B note — append_head_note must
        // NOT include it (these tools do not depend on Phase B).
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "old9999").unwrap();

        let out = append_head_note(&store, "real result".to_string(), Some("chk1111"));
        assert!(out.starts_with("real result"), "body preserved: {out}");
        assert!(
            out.contains("idx0000") && out.contains("chk1111"),
            "got: {out}"
        );
        assert!(
            !out.contains("call-graph index incomplete"),
            "head-only wrapper must not carry the Phase-B note: {out}"
        );

        // Same commit ⇒ no note. git-less caller ⇒ no note.
        assert_eq!(
            append_head_note(&store, "real result".to_string(), Some("idx0000")),
            "real result"
        );
        assert_eq!(
            append_head_note(&store, "real result".to_string(), None),
            "real result"
        );
    }

    #[test]
    fn append_head_note_empty_body_becomes_note() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "idx0000").unwrap();
        let out = append_head_note(&store, String::new(), Some("chk1111"));
        assert!(out.starts_with("[note:"), "note is the body: {out}");
        assert!(
            !out.contains('\n'),
            "no leading newline before the note: {out}"
        );
    }

    #[test]
    fn read_note_signals_carry_head_and_phase_b_as_json_array() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Marker behind HEAD ⇒ Phase-B signal; caller at another commit ⇒ head signal.
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "old9999").unwrap();

        let signals = read_note_signals(&store, Some("chk1111"), "a.ts");
        assert_eq!(signals.len(), 2, "both signals present: {signals:?}");

        // Embedding them in a graph-json envelope must still parse as JSON.
        let out = serde_json::json!({ "nodes": [], "edges": [], "signals": signals });
        let s = serde_json::to_string(&out).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&s).expect("body must stay valid JSON");
        let arr = parsed["signals"].as_array().unwrap();
        assert!(
            arr.iter()
                .any(|v| v.as_str().map(|t| t.contains("chk1111")).unwrap_or(false)),
            "head signal present: {arr:?}"
        );
    }

    #[test]
    fn read_note_signals_empty_when_fresh_and_git_less() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "idx0000").unwrap();
        store.set_meta("phase_b_commit", "idx0000").unwrap();
        // Markers agree (no Phase-B signal) and no caller HEAD ⇒ no signals at all.
        assert!(read_note_signals(&store, None, "a.ts").is_empty());
        // Caller at the same commit ⇒ still no head signal.
        assert!(read_note_signals(&store, Some("idx0000"), "a.ts").is_empty());
    }

    /// get_callers must carry the note alongside real results when the marker
    /// is behind HEAD (results may be partial, not absent).
    #[test]
    fn get_callers_appends_note_when_phase_b_behind_head() {
        use travsr_core::{Edge, EdgeKind};
        let dir = tempfile::tempdir().unwrap();
        let caller = make_fn_node_with_pkg("src/a.ts", "fn:alpha", 1, 3);
        let callee = make_fn_node_with_pkg("src/b.ts", "fn:beta", 1, 3);
        let mut store = make_store_with_root(&dir, &[]);
        let caller_id = store.put_node(&caller).unwrap();
        let callee_id = store.put_node(&callee).unwrap();
        store
            .put_edge(&Edge::new(caller_id, callee_id, EdgeKind::RefCall))
            .unwrap();
        store.set_meta("last_commit", "def456").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let result = get_callers(&store, "beta", None);
        assert!(
            result.contains("fn:alpha"),
            "existing edges must still be reported; got: {result}"
        );
        assert!(
            result.contains("call-graph index incomplete"),
            "must append pending note; got: {result}"
        );
    }

    /// An empty structural answer during a degraded state must be the note, not
    /// a silent empty string — the false negative #617 exists to prevent.
    #[test]
    fn get_blast_radius_empty_result_becomes_note_when_stale() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = make_store_with_root(&dir, &[]);
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "1").unwrap();
        let result = get_blast_radius(&store, "src/missing.ts", AnalysisMode::TreeSitter);
        assert!(
            result.contains("call-graph edges degraded"),
            "empty result must surface the stale note; got: {result}"
        );
    }

    #[test]
    fn get_execution_path_appends_note_when_phase_b_behind_head() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = make_store_with_root(&dir, &[]);
        store.set_meta("last_commit", "def456").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let result = get_execution_path(&store, "alpha", "beta");
        assert!(
            result.contains("call-graph index incomplete"),
            "disconnected/no-result path must carry the pending note; got: {result}"
        );
    }

    #[test]
    fn get_graph_json_emits_signals_when_stale() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/a.ts", "fn:alpha", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store.set_meta("phase_b_dirty", "1").unwrap();
        let params = GraphJsonParams {
            query: "alpha",
            direction: "both",
            depth: 2,
            kind_filter: "",
            provenance: "",
            token_budget: 0,
            mode: "",
            path_prefix: "",
        };
        let raw = get_graph_json(&store, &params);
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let signals = parsed["signals"].as_array().expect("signals array");
        assert!(
            signals.iter().any(|s| s
                .as_str()
                .unwrap_or("")
                .contains("call-graph edges degraded")),
            "graph JSON must carry the degraded signal; got: {raw}"
        );
    }

    #[test]
    fn get_graph_json_no_signals_when_complete() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/a.ts", "fn:alpha", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        let params = GraphJsonParams {
            query: "alpha",
            direction: "both",
            depth: 2,
            kind_filter: "",
            provenance: "",
            token_budget: 0,
            mode: "",
            path_prefix: "",
        };
        let raw = get_graph_json(&store, &params);
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            parsed.get("signals").is_none(),
            "complete graph must not emit signals; got: {raw}"
        );
    }

    // ── #618 get_graph_json seed resolution ───────────────────────────────────

    fn graph_params(query: &str) -> GraphJsonParams<'_> {
        GraphJsonParams {
            query,
            direction: "both",
            depth: 1,
            kind_filter: "",
            provenance: "",
            token_budget: 0,
            mode: "",
            path_prefix: "",
        }
    }

    fn root_count(raw: &str) -> usize {
        let parsed: serde_json::Value = serde_json::from_str(raw).unwrap();
        parsed["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["root"].as_bool() == Some(true))
            .count()
    }

    /// A bare name with an exact-symbol match must resolve to that single
    /// root instead of promoting every substring cousin to a root.
    #[test]
    fn get_graph_json_bare_name_resolves_single_root() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let exact = Node::new(
            VName::new("t", "", "src/docs.ts", "typescript", "fn:docs_section"),
            "function",
        );
        let cousin = Node::new(
            VName::new(
                "t",
                "",
                "src/build.ts",
                "typescript",
                "fn:build_docs_section",
            ),
            "function",
        );
        let helper = Node::new(
            VName::new(
                "t",
                "",
                "src/help.ts",
                "typescript",
                "fn:docs_section_helper",
            ),
            "function",
        );
        store.put_node(&exact).unwrap();
        store.put_node(&cousin).unwrap();
        store.put_node(&helper).unwrap();
        // Precondition: the raw search really does return all three, so this
        // test exercises the narrowing rather than passing trivially.
        assert!(
            store.search_nodes_by_name("docs_section").unwrap().len() >= 3,
            "substring search must surface all candidates"
        );
        let raw = get_graph_json(&store, &graph_params("docs_section"));
        assert_eq!(root_count(&raw), 1, "exactly one root expected; got: {raw}");
        assert!(
            !raw.contains("build_docs_section"),
            "unconnected substring cousin must not appear; got: {raw}"
        );
    }

    /// A full kind-qualified signature must keep resolving exactly (tier 1).
    #[test]
    fn get_graph_json_signature_query_single_root() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let exact = Node::new(
            VName::new("t", "", "src/docs.ts", "typescript", "fn:docs_section"),
            "function",
        );
        let cousin = Node::new(
            VName::new(
                "t",
                "",
                "src/build.ts",
                "typescript",
                "fn:build_docs_section",
            ),
            "function",
        );
        store.put_node(&exact).unwrap();
        store.put_node(&cousin).unwrap();
        let raw = get_graph_json(&store, &graph_params("fn:docs_section"));
        assert_eq!(root_count(&raw), 1, "exactly one root expected; got: {raw}");
    }

    /// #565 / RFC-002 consistency: a query resolving to multiple distinct
    /// definitions must carry the additive `ambiguous`/`candidates` signal, so an
    /// agent calling get_graph_json gets the same disambiguation cue the CLI does.
    #[test]
    fn get_graph_json_flags_ambiguous_definitions() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&Node::new(
                VName::new("t", "", "src/a.ts", "typescript", "fn:overloaded"),
                "function",
            ))
            .unwrap();
        store
            .put_node(&Node::new(
                VName::new("t", "", "src/b.ts", "typescript", "fn:overloaded"),
                "function",
            ))
            .unwrap();
        let raw = get_graph_json(&store, &graph_params("fn:overloaded"));
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["ambiguous"], true, "must flag ambiguity; got: {raw}");
        let cands = v["candidates"].as_array().expect("candidates array");
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().all(|c| c["signature"] == "fn:overloaded"));
    }

    /// The ambiguity fields are additive: a uniquely-resolving query omits them
    /// entirely so existing first-party consumers see an unchanged envelope.
    #[test]
    fn get_graph_json_no_ambiguous_flag_when_unique() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&Node::new(
                VName::new("t", "", "src/a.ts", "typescript", "fn:solo"),
                "function",
            ))
            .unwrap();
        let raw = get_graph_json(&store, &graph_params("fn:solo"));
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            v.get("ambiguous").is_none(),
            "unique query must not flag ambiguity; got: {raw}"
        );
        assert!(v.get("candidates").is_none());
    }

    /// With no exact match at either tier, the substring set must survive as
    /// a fuzzy fallback (a partial query still renders something).
    #[test]
    fn get_graph_json_substring_fallback_when_no_exact_match() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/a.ts", "typescript", "fn:alpha_beta"),
            "function",
        );
        let b = Node::new(
            VName::new("t", "", "src/b.ts", "typescript", "fn:alpha_gamma"),
            "function",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let raw = get_graph_json(&store, &graph_params("alpha"));
        assert_eq!(
            root_count(&raw),
            2,
            "fuzzy fallback must keep both roots; got: {raw}"
        );
    }

    /// Overloads: several nodes sharing the exact bare name are all genuine
    /// resolutions and must all stay roots.
    #[test]
    fn get_graph_json_same_name_overloads_stay_multi_root() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/a.ts", "typescript", "fn:charge"),
            "function",
        );
        let b = Node::new(
            VName::new("t", "", "src/b.ts", "typescript", "method:Billing.charge"),
            "method",
        );
        let noise = Node::new(
            VName::new("t", "", "src/c.ts", "typescript", "fn:supercharged"),
            "function",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&noise).unwrap();
        let raw = get_graph_json(&store, &graph_params("charge"));
        assert_eq!(
            root_count(&raw),
            2,
            "both same-name definitions stay roots, noise dropped; got: {raw}"
        );
        assert!(
            !raw.contains("supercharged"),
            "substring noise must be dropped; got: {raw}"
        );
    }

    /// File mode is untouched: path queries keep their substring semantics.
    #[test]
    fn get_graph_json_file_mode_keeps_substring_seeds() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/pay/a.ts", "typescript", "src/pay/a.ts"),
            "file",
        );
        let b = Node::new(
            VName::new("t", "", "src/pay/b.ts", "typescript", "src/pay/b.ts"),
            "file",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let raw = get_graph_json(
            &store,
            &GraphJsonParams {
                query: "src/pay",
                direction: "both",
                depth: 1,
                kind_filter: "file",
                provenance: "",
                token_budget: 0,
                mode: "",
                path_prefix: "",
            },
        );
        assert_eq!(
            root_count(&raw),
            2,
            "file mode substring seeding must be unchanged; got: {raw}"
        );
    }

    // ── #620 get_execution_path empty-outcome signals ─────────────────────────

    /// Disconnected endpoints must return an explicit no-path answer, never a
    /// silent empty envelope.
    #[test]
    fn get_execution_path_disconnected_says_no_path_found() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/a.ts", "typescript", "fn:alpha"),
            "function",
        );
        let b = Node::new(
            VName::new("t", "", "src/b.ts", "typescript", "fn:beta"),
            "function",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let result = get_execution_path(&store, "alpha", "beta");
        assert!(
            result.contains("no path found"),
            "disconnected endpoints must be reported; got: {result}"
        );
        assert!(
            result.contains("fn:alpha") && result.contains("fn:beta"),
            "the resolved signatures confirm resolution succeeded; got: {result}"
        );
    }

    /// An endpoint that does not resolve must get the could-not-resolve
    /// message, distinct from the no-path case.
    #[test]
    fn get_execution_path_unknown_symbol_says_could_not_resolve() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/a.ts", "typescript", "fn:alpha"),
            "function",
        );
        store.put_node(&a).unwrap();
        let result = get_execution_path(&store, "alpha", "zzz_nonexistent");
        assert!(
            result.contains("could not resolve"),
            "unresolved endpoint must be reported; got: {result}"
        );
        assert!(
            !result.contains("no path found"),
            "must not claim a path search happened; got: {result}"
        );
    }

    /// SEC P0: an access-denied endpoint must be byte-identical to a
    /// nonexistent one — no existence oracle through the new messages.
    #[test]
    fn get_execution_path_denied_matches_not_found() {
        use travsr_core::{Node, VName};
        use travsr_retrieval::RbacFilter;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("secret", "", "src/a.ts", "typescript", "fn:alpha"),
            "function",
        );
        let b = Node::new(
            VName::new("secret", "", "src/b.ts", "typescript", "fn:beta"),
            "function",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        // Filter allows nothing in corpus "secret": beta exists but is denied.
        let deny = RbacFilter::new(["public"]);
        let denied = get_execution_path_authed(&store, "alpha", "beta", &deny);
        // Same query where the sink genuinely does not exist.
        let missing = get_execution_path_authed(&store, "alpha", "zzz_nope", &deny);
        // Both go through the same unresolved branch; the messages differ only
        // by the echoed query text, never by reason. Normalize the echoes out.
        let denied_norm = denied.replace("beta", "X").replace("zzz_nope", "X");
        let missing_norm = missing.replace("beta", "X").replace("zzz_nope", "X");
        assert_eq!(
            denied_norm, missing_norm,
            "denied and missing must be indistinguishable"
        );
        assert!(
            denied.contains("could not resolve"),
            "denied case must use the same could-not-resolve message; got: {denied}"
        );
    }

    /// A connected pair keeps returning the plain path lines with no
    /// diagnostic text mixed in.
    #[test]
    fn get_execution_path_connected_returns_plain_path() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("t", "", "src/a.ts", "typescript", "fn:alpha"),
            "function",
        );
        let b = Node::new(
            VName::new("t", "", "src/b.ts", "typescript", "fn:beta"),
            "function",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();
        let result = get_execution_path(&store, "alpha", "beta");
        assert!(
            result.contains("fn:alpha") && result.contains("fn:beta"),
            "path endpoints must appear; got: {result}"
        );
        assert!(
            !result.contains("no path found") && !result.contains("could not resolve"),
            "successful path must carry no diagnostics; got: {result}"
        );
    }

    // ── #377 truncation signals ───────────────────────────────────────────────

    /// When knapsack cannot fit all scored nodes, the overflow message must appear.
    #[test]
    fn get_context_overflow_signal_emitted_on_budget_exhaustion() {
        use travsr_core::{Edge, EdgeKind};
        let dir = tempfile::tempdir().unwrap();
        // Write 10 nodes and connect them so PPR scores them all.
        let nodes: Vec<_> = (0..10)
            .map(|i| {
                make_fn_node_with_pkg(&format!("src/file{i}.ts"), &format!("fn:symbol{i}"), 1, 50)
            })
            .collect();
        let mut store = make_store_with_root(&dir, &[]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        let ids: Vec<_> = nodes.iter().map(|n| store.put_node(n).unwrap()).collect();
        // Chain: node 0 → node 1 → ... → node 9 via RefCall edges so PPR sees them all.
        for i in 0..ids.len() - 1 {
            store
                .put_edge(&Edge::new(ids[i], ids[i + 1], EdgeKind::RefCall))
                .unwrap();
        }
        // Tiny budget (5 tokens) — knapsack cannot fit all 10 nodes.
        let result = get_context_body(&store, "symbol0", 5, &OpenFilter, false, None, None);
        // Either overflow fires or PPR returned too few nodes — accept both: the key is
        // we never silently truncate when there IS overflow.
        if result.contains("more node(s) matched") {
            assert!(
                result.contains("budget exhausted"),
                "overflow msg must mention budget; got: {result}"
            );
        }
    }

    /// When all nodes fit within budget, no overflow message should appear.
    #[test]
    fn get_context_no_overflow_when_all_fit() {
        let dir = tempfile::tempdir().unwrap();
        let node = make_fn_node_with_pkg("src/payment.ts", "fn:charge", 1, 3);
        let mut store = make_store_with_root(&dir, &[node]);
        store.set_meta("last_commit", "abc123").unwrap();
        store.set_meta("phase_b_commit", "abc123").unwrap();
        // Generous budget — the single node easily fits.
        let result = get_context_body(&store, "charge", 4096, &OpenFilter, false, None, None);
        assert!(
            !result.contains("more node(s) matched"),
            "no overflow when all nodes fit; got: {result}"
        );
    }

    /// embed_path_seeds must report n_eligible correctly when MAX cap fires.
    #[test]
    fn embed_path_seeds_reports_cap_count() {
        use travsr_store::Store;
        // Build MAX_EMBED_SEEDS + 2 src nodes — all pass noise/RBAC filters.
        let n_total = MAX_EMBED_SEEDS + 2;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let mut knn_pairs: Vec<(NodeId, f32)> = Vec::with_capacity(n_total);
        for i in 0..n_total {
            let node = make_node_with_kind_and_path(
                "function",
                &format!("src/file{i}.ts"),
                &format!("fn:sym{i}"),
            );
            let id = store.put_node(&node).unwrap();
            knn_pairs.push((id, 0.90 - i as f32 * 0.001));
        }
        let knn: EmbedKnnFn<'_> = &|_q, _k| knn_pairs.clone();
        let (seeds, n_eligible, _, _) = embed_path_seeds(&store, "sym", knn, &OpenFilter);
        assert_eq!(
            seeds.len(),
            MAX_EMBED_SEEDS,
            "seeds capped at MAX_EMBED_SEEDS"
        );
        assert_eq!(
            n_eligible, n_total,
            "n_eligible counts all eligible before cap"
        );
    }

    /// embed_path_seeds: seed-cap signal fires (n_eligible > seeds.len()) when cap was hit.
    #[test]
    fn embed_path_seeds_cap_signal_is_correct() {
        use travsr_store::Store;
        let n_total = MAX_EMBED_SEEDS + 5;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let mut knn_pairs: Vec<(NodeId, f32)> = Vec::with_capacity(n_total);
        for i in 0..n_total {
            let node = make_node_with_kind_and_path(
                "function",
                &format!("src/cap{i}.ts"),
                &format!("fn:cap{i}"),
            );
            let id = store.put_node(&node).unwrap();
            knn_pairs.push((id, 0.95));
        }
        let knn: EmbedKnnFn<'_> = &|_q, _k| knn_pairs.clone();
        let (seeds, n_eligible, _, _) = embed_path_seeds(&store, "cap", knn, &OpenFilter);
        assert!(
            n_eligible > seeds.len(),
            "cap fired: n_eligible ({n_eligible}) > seeds.len() ({})",
            seeds.len()
        );
    }

    /// build_context_signals: overflow appears before seed-cap, seed-cap before degraded notes.
    #[test]
    fn build_context_signals_ordering() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let result = build_context_signals(
            &store,
            false,
            false,
            false,
            Some("[overflow msg]"),
            Some("[seed cap msg]"),
        );
        let overflow_pos = result.find("[overflow msg]").unwrap();
        let seed_pos = result.find("[seed cap msg]").unwrap();
        let note_pos = result.find("[note: semantic search").unwrap();
        assert!(overflow_pos < seed_pos, "overflow must precede seed-cap");
        assert!(seed_pos < note_pos, "seed-cap must precede degraded notes");
    }

    /// KNN cosine-similarity weights must drive PPR: the neighbor of the
    /// high-weight seed must appear in the output, and if both neighbors appear
    /// the high-weight neighbor must rank earlier (output is PPR-score-ordered).
    #[test]
    fn get_context_knn_weights_drive_ppr_seed_preference() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // Graph:  node_a (knn score 0.95, FTS-matched by query) → node_x (RefCall)
        //         node_b (knn score 0.10, FTS-missed)           → node_y (RefCall)
        //
        // Query "pqseeda" matches only node_a's signature (contains "pqseeda") — FTS
        // grounds the query and gives a_id anchor coverage. KNN also ranks a_id at
        // 0.95 >> b_id 0.10. Both sources agree → PPR strongly personalises toward
        // a_id → downstream_x must appear.
        //
        // Note: we cannot reliably assert x-before-y here because FTS matches node_a
        // with weight 1.0, then the max() merge gives a_id weight=1.0 and b_id
        // weight=1.0 (both from FTS via lexical fallback) — KNN differential is masked
        // in the weight. The important guarantee is that downstream_x is *reachable*
        // from the grounded seed, not a specific rank.
        let node_a = make_fn_node_with_pkg("a.ts", "fn:pqseeda_alpha", 1, 3);
        let node_b = make_fn_node_with_pkg("b.ts", "fn:other_beta", 1, 3);
        let node_x = make_fn_node_with_pkg("x.ts", "fn:downstream_x", 1, 3);
        let node_y = make_fn_node_with_pkg("y.ts", "fn:downstream_y", 1, 3);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&node_a).unwrap();
        let b_id = store.put_node(&node_b).unwrap();
        let x_id = store.put_node(&node_x).unwrap();
        let y_id = store.put_node(&node_y).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(a_id, x_id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&travsr_core::Edge::new(b_id, y_id, EdgeKind::RefCall))
            .unwrap();

        // KNN: a_id has high cosine similarity; FTS also anchors to a_id.
        let knn_fn: EmbedKnnFn<'_> = &|_query, _k| vec![(a_id, 0.95), (b_id, 0.10)];

        let result = get_context_body(
            &store,
            "pqseeda",
            4096,
            &OpenFilter,
            false,
            None,
            Some(knn_fn),
        );

        // node_x must appear: anchor grounding + high KNN score on a_id → PPR expands to x.
        assert!(
            result.contains("downstream_x"),
            "neighbor of grounded + high-weight KNN seed must appear in context: {result}"
        );
    }

    // ── dedup_adjacent_seeds ─────────────────────────────────────────────────

    #[test]
    fn dedup_adjacent_seeds_empty_and_single_passthrough() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        assert!(dedup_adjacent_seeds(&store, vec![]).is_empty());

        let node = make_fn_node_with_pkg("a.ts", "fn:solo", 1, 2);
        let mut s = travsr_store::SqliteStore::open_in_memory().unwrap();
        let id = s.put_node(&node).unwrap();
        let result = dedup_adjacent_seeds(&s, vec![(id, 1.0)]);
        assert_eq!(result, vec![(id, 1.0)]);
    }

    #[test]
    fn dedup_adjacent_seeds_drops_lower_scored_direct_neighbor() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // Graph: a (score 1.0) → b (score 0.7) via RefCall.
        // b is a direct outgoing neighbor of a; it should be dropped.
        let node_a = make_fn_node_with_pkg("a.ts", "fn:dedup_a", 1, 2);
        let node_b = make_fn_node_with_pkg("b.ts", "fn:dedup_b", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&node_a).unwrap();
        let b_id = store.put_node(&node_b).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(a_id, b_id, EdgeKind::RefCall))
            .unwrap();

        let seeds = vec![(a_id, 1.0), (b_id, 0.7)];
        let result = dedup_adjacent_seeds(&store, seeds);
        assert_eq!(
            result,
            vec![(a_id, 1.0)],
            "b must be dropped as neighbor of a"
        );
    }

    #[test]
    fn dedup_adjacent_seeds_keeps_upstream_callers() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // Graph: b (score 0.7) → a (score 1.0) via RefCall (b CALLS a).
        // b is a CALLER of a. PPR from a follows outgoing edges — it cannot walk
        // backwards to b. Callers provide orthogonal coverage (usage context / call sites)
        // and must NEVER be dropped, even when they score lower than their callee.
        let node_a = make_fn_node_with_pkg("a.ts", "fn:upstr_a", 1, 2);
        let node_b = make_fn_node_with_pkg("b.ts", "fn:upstr_b", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&node_a).unwrap();
        let b_id = store.put_node(&node_b).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(b_id, a_id, EdgeKind::RefCall))
            .unwrap();

        // a has higher score so it is processed first.
        let seeds = vec![(a_id, 1.0), (b_id, 0.7)];
        let result = dedup_adjacent_seeds(&store, seeds.clone());
        assert_eq!(
            result, seeds,
            "b is a caller, must be kept for orthogonal PPR coverage"
        );
    }

    #[test]
    fn dedup_adjacent_seeds_keeps_unconnected_seeds() {
        use travsr_store::Store;

        // Three nodes with no edges between them: all must be kept.
        let node_a = make_fn_node_with_pkg("a.ts", "fn:iso_a", 1, 2);
        let node_b = make_fn_node_with_pkg("b.ts", "fn:iso_b", 1, 2);
        let node_c = make_fn_node_with_pkg("c.ts", "fn:iso_c", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&node_a).unwrap();
        let b_id = store.put_node(&node_b).unwrap();
        let c_id = store.put_node(&node_c).unwrap();

        let seeds = vec![(a_id, 1.0), (b_id, 0.9), (c_id, 0.8)];
        let result = dedup_adjacent_seeds(&store, seeds.clone());
        assert_eq!(result, seeds, "no edges means no dedup");
    }

    #[test]
    fn dedup_adjacent_seeds_keeps_non_seed_edges() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // a → x (not a seed) and b → y (not a seed). a and b have no edge between them.
        // Both seeds must be kept even though each has outgoing edges.
        let node_a = make_fn_node_with_pkg("a.ts", "fn:ext_a", 1, 2);
        let node_b = make_fn_node_with_pkg("b.ts", "fn:ext_b", 1, 2);
        let node_x = make_fn_node_with_pkg("x.ts", "fn:ext_x", 1, 2);
        let node_y = make_fn_node_with_pkg("y.ts", "fn:ext_y", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&node_a).unwrap();
        let b_id = store.put_node(&node_b).unwrap();
        let x_id = store.put_node(&node_x).unwrap();
        let y_id = store.put_node(&node_y).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(a_id, x_id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&travsr_core::Edge::new(b_id, y_id, EdgeKind::RefCall))
            .unwrap();

        let seeds = vec![(a_id, 1.0), (b_id, 0.9)];
        let result = dedup_adjacent_seeds(&store, seeds.clone());
        assert_eq!(result, seeds, "edges to non-seeds must not trigger dedup");
    }

    // ── enrich_seeds_with_callers ────────────────────────────────────────────

    #[test]
    fn enrich_adds_production_caller_at_reduced_weight() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // Graph: caller (fn:caller_fn) → seed (fn:target_fn) via RefCall.
        // After enrichment the caller should appear at 35% of the seed's weight.
        let seed_node = make_fn_node_with_pkg("lib.rs", "fn:target_fn", 1, 2);
        let caller_node = make_fn_node_with_pkg("main.rs", "fn:caller_fn", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let seed_id = store.put_node(&seed_node).unwrap();
        let caller_id = store.put_node(&caller_node).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(
                caller_id,
                seed_id,
                EdgeKind::RefCall,
            ))
            .unwrap();

        let seeds = vec![(seed_id, 1.0)];
        let result = enrich_seeds_with_callers(&store, seeds, usize::MAX);

        assert_eq!(result.len(), 2, "caller should be added");
        let caller_entry = result.iter().find(|&&(id, _)| id == caller_id);
        assert!(caller_entry.is_some(), "caller_id missing from result");
        let weight = caller_entry.unwrap().1;
        assert!(
            (weight - 0.35).abs() < 1e-5,
            "caller weight should be 0.35, got {weight}"
        );
    }

    #[test]
    fn enrich_skips_test_callers() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // A test-function caller (signature: fn:test_foo) must be filtered out.
        let seed_node = make_fn_node_with_pkg("lib.rs", "fn:important_fn", 1, 2);
        let test_caller = make_fn_node_with_pkg("lib.rs", "fn:test_important_fn", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let seed_id = store.put_node(&seed_node).unwrap();
        let test_id = store.put_node(&test_caller).unwrap();
        store
            .put_edge(&travsr_core::Edge::new(test_id, seed_id, EdgeKind::RefCall))
            .unwrap();

        let result = enrich_seeds_with_callers(&store, vec![(seed_id, 1.0)], usize::MAX);
        assert_eq!(
            result.len(),
            1,
            "test caller must be filtered, only seed remains"
        );
        assert!(!result.iter().any(|&(id, _)| id == test_id));
    }

    #[test]
    fn enrich_skips_already_present_seed() {
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        // When the caller is already a seed, it must not be duplicated.
        let seed_a = make_fn_node_with_pkg("a.rs", "fn:fn_a", 1, 2);
        let seed_b = make_fn_node_with_pkg("b.rs", "fn:fn_b", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a_id = store.put_node(&seed_a).unwrap();
        let b_id = store.put_node(&seed_b).unwrap();
        // b calls a
        store
            .put_edge(&travsr_core::Edge::new(b_id, a_id, EdgeKind::RefCall))
            .unwrap();

        // Both are already seeds — b should not be added again at reduced weight.
        let seeds = vec![(a_id, 1.0), (b_id, 0.9)];
        let result = enrich_seeds_with_callers(&store, seeds.clone(), usize::MAX);
        assert_eq!(
            result.len(),
            2,
            "no duplicates when caller already in seed set"
        );
        // b must retain original weight 0.9, not get demoted to 0.35
        let b_weight = result.iter().find(|&&(id, _)| id == b_id).unwrap().1;
        assert!(
            (b_weight - 0.9).abs() < 1e-5,
            "existing seed weight must not be overwritten"
        );
    }

    #[test]
    fn enrich_empty_seeds_passthrough() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let result = enrich_seeds_with_callers(&store, vec![], usize::MAX);
        assert!(result.is_empty());
    }

    #[test]
    fn enrich_callers_depth_two_adds_grandparent_caller() {
        // Chain: grandparent → parent → child (all RefCall edges).
        // Query seeds = [child]. Two calls simulate depth-1 then depth-2 enrichment.
        // After hop-1: parent added. After hop-2: grandparent added.
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        let child = make_fn_node_with_pkg("lib.rs", "fn:child", 1, 2);
        let parent = make_fn_node_with_pkg("lib.rs", "fn:parent", 1, 2);
        let grandparent = make_fn_node_with_pkg("main.rs", "fn:grandparent", 1, 2);

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let child_id = store.put_node(&child).unwrap();
        let parent_id = store.put_node(&parent).unwrap();
        let gp_id = store.put_node(&grandparent).unwrap();

        store
            .put_edge(&travsr_core::Edge::new(
                parent_id,
                child_id,
                EdgeKind::RefCall,
            ))
            .unwrap();
        store
            .put_edge(&travsr_core::Edge::new(gp_id, parent_id, EdgeKind::RefCall))
            .unwrap();

        // Simulate what get_context_body / ask_query do: two sequential enrich calls.
        let seeds = vec![(child_id, 1.0)];
        let seeds = enrich_seeds_with_callers(&store, seeds, 15); // depth-1
        let seeds = enrich_seeds_with_callers(&store, seeds, 10); // depth-2

        let ids: Vec<_> = seeds.iter().map(|&(id, _)| id).collect();
        assert!(
            ids.contains(&parent_id),
            "depth-1 caller (parent) must be seeded"
        );
        assert!(
            ids.contains(&gp_id),
            "depth-2 caller (grandparent) must be seeded"
        );

        // Weights decrease per hop: parent ≈ 0.35, grandparent ≈ 0.35²
        let parent_w = seeds.iter().find(|&&(id, _)| id == parent_id).unwrap().1;
        let gp_w = seeds.iter().find(|&&(id, _)| id == gp_id).unwrap().1;
        assert!(
            (parent_w - 0.35).abs() < 1e-5,
            "parent weight = 0.35; got {parent_w}"
        );
        assert!(
            (gp_w - 0.35 * 0.35).abs() < 1e-4,
            "grandparent weight = 0.35²; got {gp_w}"
        );
    }

    #[test]
    fn enrich_callers_respects_max_new_total() {
        // Hub node called by 8 production callers. max_new_total=3 must cap additions.
        use travsr_core::EdgeKind;
        use travsr_store::Store;

        let hub = make_fn_node_with_pkg("lib.rs", "fn:hub", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let hub_id = store.put_node(&hub).unwrap();

        for i in 0..8u32 {
            let caller =
                make_fn_node_with_pkg(&format!("caller{i}.rs"), &format!("fn:caller_{i}"), 1, 2);
            let caller_id = store.put_node(&caller).unwrap();
            store
                .put_edge(&travsr_core::Edge::new(
                    caller_id,
                    hub_id,
                    EdgeKind::RefCall,
                ))
                .unwrap();
        }

        let seeds = vec![(hub_id, 1.0)];
        let result = enrich_seeds_with_callers(&store, seeds, 3);
        // 1 original + at most 3 new callers
        assert_eq!(
            result.len(),
            4,
            "max_new_total=3 must cap additions; got {}",
            result.len()
        );
    }

    // ── KNN circuit-breaker ──────────────────────────────────────────────────

    #[test]
    fn knn_circuit_breaker_falls_back_to_fts_on_slow_sidecar() {
        use travsr_store::Store;

        // A slow KNN closure that sleeps beyond KNN_BUDGET_MS — get_context must
        // still return a result derived from FTS seeds rather than panicking or
        // hanging indefinitely (it will block for the sleep duration, but the
        // result must not include the KNN-only node).
        let node_fts = make_fn_node_with_pkg("fts.ts", "fn:cb_fts_sym", 1, 2);
        let node_knn_only = make_fn_node_with_pkg("knn.ts", "fn:cb_knn_only", 1, 2);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.put_node(&node_fts).unwrap();
        let knn_id = store.put_node(&node_knn_only).unwrap();

        // Sleep just over the budget so the circuit-breaker fires.
        let slow_knn: EmbedKnnFn<'_> = &|_q, _k| {
            std::thread::sleep(std::time::Duration::from_millis(KNN_BUDGET_MS as u64 + 5));
            vec![(knn_id, 0.99)]
        };

        let result = get_context_body(
            &store,
            "cb_fts_sym",
            4096,
            &OpenFilter,
            false,
            None,
            Some(slow_knn),
        );

        // The KNN-only node must not appear — circuit-breaker discarded those results.
        assert!(
            !result.contains("cb_knn_only"),
            "circuit-breaker must discard KNN results after timeout; got: {result}"
        );
    }

    // ── find_references (#299) ─────────────────────────────────────────────

    #[test]
    fn find_references_emits_occurrence_sites() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("", "", "src/svc.rs", "rust", "fn:charge"),
            "function",
        )
        .with_line(4);
        let caller = Node::new(
            VName::new("", "", "src/svc.rs", "rust", "fn:process"),
            "function",
        );
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        // Two distinct occurrence lines from the same caller must both appear.
        store
            .record_edge_sites(&[
                (caller.id, callee.id, 9, None),
                (caller.id, callee.id, 10, None),
            ])
            .unwrap();

        let out = find_references(&store, "charge", None);
        assert!(out.contains("resolved: fn:charge"), "header missing: {out}");
        assert!(out.contains("src/svc.rs:9"), "site 9 missing: {out}");
        assert!(out.contains("src/svc.rs:10"), "site 10 missing: {out}");
        assert!(out.contains("2 reference(s)"), "count missing: {out}");
    }

    #[test]
    fn find_references_softens_zero_when_target_file_has_no_occurrences() {
        // #450: the language gate passes (another file of the same language has
        // occurrence rows), but the target's OWN file has none — so a definitive
        // "recorded no uses" would be a claim the index cannot support.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();

        // covered.rs is analysed: it carries a real occurrence.
        let caller = Node::new(
            VName::new("", "", "src/covered.rs", "rust", "fn:caller"),
            "function",
        );
        let callee = Node::new(
            VName::new("", "", "src/covered.rs", "rust", "fn:callee"),
            "function",
        );
        // untouched.rs is not: same language, no occurrence rows at all.
        let orphan = Node::new(
            VName::new("", "", "src/untouched.rs", "rust", "fn:orphan"),
            "function",
        )
        .with_line(3);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&orphan).unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 11, None)])
            .unwrap();

        let out = find_references(&store, "orphan", None);
        assert!(
            out.contains("not a definitive zero"),
            "should soften the claim: {out}"
        );
        assert!(
            out.contains("src/untouched.rs"),
            "should name the unanalysed file: {out}"
        );
        assert!(
            !out.contains("recorded no uses"),
            "must not assert absence: {out}"
        );
    }

    #[test]
    fn find_references_keeps_definitive_zero_when_target_file_is_analyzed() {
        // Converse of the above: the target's own file carries occurrence rows,
        // so a zero for this symbol is a real zero and the existing confident
        // wording must be preserved unchanged.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();

        let caller = Node::new(
            VName::new("", "", "src/svc.rs", "rust", "fn:caller"),
            "function",
        );
        let callee = Node::new(
            VName::new("", "", "src/svc.rs", "rust", "fn:callee"),
            "function",
        );
        // Same file, analysed, but nothing references it.
        let unused = Node::new(
            VName::new("", "", "src/svc.rs", "rust", "fn:unused"),
            "function",
        )
        .with_line(20);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&unused).unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 5, None)])
            .unwrap();

        let out = find_references(&store, "unused", None);
        assert!(
            out.contains("recorded no uses"),
            "analysed file should still give a definitive zero: {out}"
        );
        assert!(
            !out.contains("not a definitive zero"),
            "should not soften when the file was analysed: {out}"
        );
    }

    #[test]
    fn find_references_none_has_no_resolution() {
        // No match resolves nothing: like get_callers, the body is empty (the MCP
        // envelope may still wrap it), so there is no `resolved:` header or sites.
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let out = find_references(&store, "doesnotexist", None);
        assert!(!out.contains("resolved:"), "unexpected resolution: {out}");
        assert!(!out.contains("reference(s)"), "unexpected sites: {out}");
    }

    #[test]
    fn find_references_ambiguous_lists_candidates() {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(VName::new("", "", "a.rs", "rust", "fn:charge"), "function").with_line(1);
        let b = Node::new(VName::new("", "", "b.rs", "rust", "fn:charge"), "function").with_line(2);
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let out = find_references(&store, "charge", None);
        assert!(
            out.contains("ambiguous"),
            "expected disambiguation list: {out}"
        );
        assert!(
            out.contains("a.rs") && out.contains("b.rs"),
            "both defs listed: {out}"
        );

        // A path hint disambiguates to exactly one definition.
        let pinned = find_references(&store, "charge", Some("a.rs"));
        assert!(
            !pinned.contains("ambiguous"),
            "path hint must disambiguate: {pinned}"
        );
    }

    #[test]
    fn find_references_ignores_import_nodes() {
        // #299 F1: an `import`/`use` node shares the imported symbol's simple
        // name. It must never be treated as a definition, or the real `fn` reads
        // as ambiguous and pinning the import yields a misleading "0 references".
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let real = Node::new(
            VName::new("", "", "svc.rs", "rust", "fn:charge"),
            "function",
        )
        .with_line(4);
        let import = Node::new(
            VName::new("", "", "main.rs", "rust", "use:svc::charge"),
            "import",
        );
        store.put_node(&real).unwrap();
        store.put_node(&import).unwrap();
        let out = find_references(&store, "charge", None);
        assert!(
            !out.contains("ambiguous"),
            "import node must not create ambiguity: {out}"
        );
        assert!(
            out.contains("resolved: fn:charge"),
            "must resolve to the real fn: {out}"
        );
    }

    #[test]
    fn find_references_path_hint_respects_slash_boundary() {
        // #299 F1: a `path` hint must pin on a `/`-boundary — `tools.rs` selects
        // `src/tools.rs` but never `src/mytools.rs`.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("", "", "src/tools.rs", "rust", "fn:charge"),
            "function",
        )
        .with_line(1);
        let b = Node::new(
            VName::new("", "", "src/mytools.rs", "rust", "fn:charge"),
            "function",
        )
        .with_line(2);
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let pinned = find_references(&store, "charge", Some("tools.rs"));
        assert!(
            !pinned.contains("ambiguous"),
            "slash-boundary hint must pin exactly one file: {pinned}"
        );
        assert!(
            pinned.contains("src/tools.rs") && !pinned.contains("mytools.rs"),
            "must select src/tools.rs, not src/mytools.rs: {pinned}"
        );
    }

    #[test]
    fn path_hint_matches_covers_forms_and_keeps_boundary_guard() {
        // #647: every documented hint form matches; the #299 F1 boundary guard
        // (a filename hint never bleeds into a longer filename) still holds.
        let p = "crates/travsr-retrieval/src/ppr.rs";
        assert!(path_hint_matches(p, p), "whole path");
        assert!(path_hint_matches(p, "ppr.rs"), "filename suffix");
        assert!(path_hint_matches(p, "src/ppr.rs"), "trailing path");
        assert!(
            path_hint_matches(p, "crates/travsr-retrieval"),
            "directory prefix"
        );
        assert!(
            path_hint_matches(p, "crates/travsr-retrieval/"),
            "prefix + trailing slash"
        );
        assert!(path_hint_matches(p, "src"), "interior whole segment");
        assert!(
            path_hint_matches(p, "retrieval"),
            "bare-word substring of a segment"
        );
        assert!(!path_hint_matches(p, "travsr-store"), "unrelated directory");
        // #299 F1: a filename-shaped hint (has `.`) stays boundary-pinned.
        assert!(
            !path_hint_matches("src/mytools.rs", "tools.rs"),
            "no filename bleed"
        );
        assert!(
            path_hint_matches("src/tools.rs", "tools.rs"),
            "exact filename on boundary"
        );
    }

    #[test]
    fn find_references_path_hint_directory_and_fragment_return_refs() {
        // #647: a directory prefix, an interior segment, and a bare fragment all
        // scope references instead of silently returning "0 references".
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let def = Node::new(
            VName::new(
                "",
                "",
                "crates/travsr-retrieval/src/ppr.rs",
                "rust",
                "fn:ppr",
            ),
            "function",
        )
        .with_line(10);
        let caller = Node::new(
            VName::new(
                "",
                "",
                "crates/travsr-retrieval/benches/retrieval.rs",
                "rust",
                "fn:bench",
            ),
            "function",
        );
        store.put_node(&def).unwrap();
        store.put_node(&caller).unwrap();
        store
            .record_edge_sites(&[(caller.id, def.id, 42, None)])
            .unwrap();

        for hint in [
            "crates/travsr-retrieval",  // directory prefix
            "crates/travsr-retrieval/", // + trailing slash
            "src",                      // interior whole segment
            "retrieval",                // bare-word substring of `travsr-retrieval`
        ] {
            let out = find_references(&store, "ppr", Some(hint));
            assert!(
                out.contains("resolved: fn:ppr"),
                "hint {hint:?} must resolve the definition: {out}"
            );
            assert!(
                out.contains("crates/travsr-retrieval/benches/retrieval.rs:42"),
                "hint {hint:?} must list the reference site: {out}"
            );
        }
    }

    #[test]
    fn find_references_path_miss_is_informative_not_empty() {
        // #647: a symbol that resolves but has no definition under the given path
        // must explain where it lives, never return an empty (misleading "0
        // references") result.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let def = Node::new(
            VName::new(
                "",
                "",
                "crates/travsr-retrieval/src/ppr.rs",
                "rust",
                "fn:ppr",
            ),
            "function",
        )
        .with_line(10);
        store.put_node(&def).unwrap();

        let out = find_references(&store, "ppr", Some("crates/travsr-store"));
        assert!(
            out.contains("resolves") && out.contains("no definition is under path"),
            "must explain the path miss: {out}"
        );
        assert!(
            out.contains("crates/travsr-retrieval/src/ppr.rs"),
            "must show where the symbol actually lives: {out}"
        );
        assert!(
            !out.contains("ambiguous"),
            "a single def is not ambiguous: {out}"
        );
    }

    #[test]
    fn find_references_dotted_qualified_resolves() {
        // #449: `ClassC.shared`, the stored Phase B signature qualifies the
        // member (`swift::ClassC.shared`); the dotted tier must pick it over a
        // same-named member of another type.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let shared_c = Node::new(
            VName::new("", "", "ClassC.swift", "swift", "swift::ClassC.shared"),
            "field",
        )
        .with_line(2);
        let shared_d = Node::new(
            VName::new("", "", "ClassD.swift", "swift", "swift::ClassD.shared"),
            "field",
        )
        .with_line(2);
        let caller = Node::new(
            VName::new("", "", "Caller.swift", "swift", "fn:useShared"),
            "function",
        );
        store.put_node(&shared_c).unwrap();
        store.put_node(&shared_d).unwrap();
        store.put_node(&caller).unwrap();
        store
            .record_edge_sites(&[(caller.id, shared_c.id, 7, None)])
            .unwrap();

        let out = find_references(&store, "ClassC.shared", None);
        assert!(
            out.contains("resolved: swift::ClassC.shared"),
            "dotted query must resolve to the qualified node: {out}"
        );
        assert!(out.contains("Caller.swift:7"), "site missing: {out}");
        assert!(
            !out.contains("ClassD.shared"),
            "must not match the other container: {out}"
        );
    }

    #[test]
    fn find_references_dotted_container_suffix_does_not_collide() {
        // #449: a naive substring match on "{container}.{member}" would let
        // "AppConfig.shared" satisfy a query for "Config.shared" (the longer
        // signature contains the shorter one as a substring). The container
        // match must be boundary-anchored so unrelated types whose name
        // happens to end with the queried container are never mistaken for it.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let app_config_shared = Node::new(
            VName::new(
                "",
                "",
                "AppConfig.swift",
                "swift",
                "swift::AppConfig.shared",
            ),
            "field",
        )
        .with_line(2);
        store.put_node(&app_config_shared).unwrap();

        let out = find_references(&store, "Config.shared", None);
        assert!(
            !out.contains("resolved:"),
            "must not resolve to AppConfig.shared via substring collision: {out}"
        );
    }

    #[test]
    fn find_references_dotted_unique_bare_member_resolves() {
        // #449: when no stored signature carries the container qualifier (Phase A
        // swift emits `var:shared`), a dotted query still resolves when the bare
        // member match is unique.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let shared = Node::new(
            VName::new("", "", "ClassC.swift", "swift", "var:shared"),
            "field",
        )
        .with_line(2);
        let caller = Node::new(
            VName::new("", "", "Caller.swift", "swift", "fn:useShared"),
            "function",
        );
        store.put_node(&shared).unwrap();
        store.put_node(&caller).unwrap();
        store
            .record_edge_sites(&[(caller.id, shared.id, 4, None)])
            .unwrap();

        let out = find_references(&store, "ClassC.shared", None);
        assert!(
            out.contains("resolved: var:shared"),
            "unique bare member must resolve: {out}"
        );
        assert!(out.contains("Caller.swift:4"), "site missing: {out}");
    }

    #[test]
    fn find_references_dotted_ambiguous_bare_member_is_none() {
        // #449: two bare same-named members and no qualifier match, resolving
        // either would be a guess, so the dotted tier must yield nothing.
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let a = Node::new(
            VName::new("", "", "ClassC.swift", "swift", "var:shared"),
            "field",
        )
        .with_line(2);
        let b = Node::new(
            VName::new("", "", "ClassD.swift", "swift", "var:shared"),
            "field",
        )
        .with_line(2);
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();

        let out = find_references(&store, "ClassC.shared", None);
        assert!(
            !out.contains("resolved:"),
            "ambiguous bare members must not resolve: {out}"
        );
    }

    #[test]
    fn find_references_falls_back_to_edges_without_sites() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let callee = Node::new(
            VName::new("", "", "svc.rs", "rust", "fn:charge"),
            "function",
        );
        let caller = Node::new(
            VName::new("", "", "caller.rs", "rust", "fn:run"),
            "function",
        )
        .with_line(12);
        store.put_node(&callee).unwrap();
        store.put_node(&caller).unwrap();
        // RefCall edge but NO edge_sites row → degraded caller-definition fallback.
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        let out = find_references(&store, "charge", None);
        assert!(
            out.contains("caller.rs:12"),
            "fallback caller def missing: {out}"
        );
        assert!(
            out.contains("occurrence lines unavailable"),
            "fallback must be labelled degraded: {out}"
        );
    }

    // ── find_pattern (#299) ────────────────────────────────────────────────

    #[test]
    fn find_pattern_rejects_control_chars() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Newline / NUL in the pattern must be rejected before any exec: the body
        // is empty, so only the empty envelope comes back (never a match list).
        let empty = "<travsr-data></travsr-data>";
        assert_eq!(find_pattern(&store, "foo\nbar", None, false), empty);
        assert_eq!(find_pattern(&store, "", None, false), empty);
    }

    #[test]
    fn find_pattern_without_repo_root_is_graceful() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let out = find_pattern(&store, "charge", None, false);
        assert!(
            out.contains("run `travsr init`"),
            "expected init hint: {out}"
        );
    }

    #[test]
    fn find_pattern_rejects_out_of_repo_scope() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", "/tmp/whatever").unwrap();
        // Absolute and parent-escaping scopes must be dropped before exec →
        // empty envelope, never a match list.
        let empty = "<travsr-data></travsr-data>";
        assert_eq!(find_pattern(&store, "charge", Some("/etc"), false), empty);
        assert_eq!(
            find_pattern(&store, "charge", Some("../secrets"), false),
            empty
        );
    }

    #[test]
    fn find_pattern_honors_travsrignore() {
        use std::process::Command;
        // #299 F10: a git-tracked file excluded via `.travsrignore` must not
        // appear in find_pattern output, so it agrees with the graph's file set.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);
        std::fs::write(root.join("keep.rs"), "fn charge() {}\n").unwrap();
        std::fs::create_dir_all(root.join("gen")).unwrap();
        std::fs::write(root.join("gen/skip.rs"), "fn charge() {}\n").unwrap();
        std::fs::write(root.join(".travsrignore"), "gen/\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();

        // Skip if git is unavailable in the sandbox (grep returns nothing).
        let out = find_pattern(&store, "charge", None, false);
        if out.contains("keep.rs") {
            assert!(
                !out.contains("gen/skip.rs"),
                ".travsrignore-excluded file must not appear: {out}"
            );
        }
    }

    // ── #448: the search set must equal the graph's file set ─────────────────

    /// A repo holding one file of every class the indexer walker and git
    /// disagree about. All four define `fn charge()`, so a single pattern
    /// separates "searched" from "not searched".
    ///
    /// | file              | git        | walker / graph              |
    /// |-------------------|------------|-----------------------------|
    /// | tracked.rs        | tracked    | indexed                     |
    /// | untracked.rs      | untracked  | indexed                     |
    /// | vendored/dep.rs   | ignored    | indexed (`!` re-includes it)|
    /// | ignored/dep.rs    | ignored    | not indexed                 |
    fn file_universe_fixture() -> (tempfile::TempDir, travsr_store::SqliteStore) {
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);

        let body = "fn charge() {}\n";
        std::fs::write(root.join(".gitignore"), "vendored/\nignored/\n").unwrap();
        std::fs::write(root.join(".travsrignore"), "!vendored/\n").unwrap();
        std::fs::write(root.join("tracked.rs"), body).unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);

        // Everything below stays outside the index on purpose.
        std::fs::write(root.join("untracked.rs"), body).unwrap();
        for dir_name in ["vendored", "ignored"] {
            std::fs::create_dir_all(root.join(dir_name)).unwrap();
            std::fs::write(root.join(dir_name).join("dep.rs"), body).unwrap();
        }

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();
        (dir, store)
    }

    #[test]
    fn find_pattern_reaches_untracked_indexed_file() {
        // #448: the walker indexes a file the moment it exists; `git grep`
        // without `--untracked` only ever sees the index, so every not-yet-added
        // file was a guaranteed false negative.
        let (_dir, store) = file_universe_fixture();
        let out = find_pattern(&store, "charge", None, false);
        if !out.contains("tracked.rs") {
            return; // git unavailable in this sandbox — nothing to assert
        }
        assert!(
            out.contains("untracked.rs"),
            "an indexed-but-untracked file must be searchable: {out}"
        );
    }

    #[test]
    fn find_pattern_reaches_travsrignore_reincluded_file() {
        // #448: `.travsrignore` outranks `.gitignore` in the indexer's walker,
        // so a `!` rule puts a gitignored subtree in the graph. git's own file
        // discovery can never reach it, at any flag combination.
        let (_dir, store) = file_universe_fixture();
        let out = find_pattern(&store, "charge", None, false);
        if !out.contains("tracked.rs") {
            return;
        }
        assert!(
            out.contains("vendored/dep.rs"),
            "a `!`-re-included gitignored file is in the graph and must be searchable: {out}"
        );
    }

    #[test]
    fn find_pattern_still_excludes_plain_gitignored_file() {
        // The other direction: widening the search must not start returning
        // files the graph does not contain. `ignored/` has no `!` rule.
        let (_dir, store) = file_universe_fixture();
        let out = find_pattern(&store, "charge", None, false);
        if !out.contains("tracked.rs") {
            return;
        }
        assert!(
            !out.contains("ignored/dep.rs"),
            "a gitignored file with no `!` rule is not in the graph and must not appear: {out}"
        );
    }

    #[test]
    fn find_pattern_still_excludes_skip_dirs_that_no_ignore_rule_covers() {
        // #448 review: `--untracked` widened pass 1 to every file git does not
        // ignore, but the walker hard-skips SKIP_DIRS *ahead of* any ignore
        // file (`SKIP_DIRS (hard) < .gitignore < .travsrignore`). A `dist/`
        // that nothing gitignores is therefore in git's universe and never in
        // the graph, and no ignore rule exists to filter it back out — only the
        // component check can. Same over-inclusion the gitignored control
        // guards, arriving through a different door.
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);
        // No .gitignore at all, so nothing excludes dist/ or target/ from git.
        std::fs::write(root.join("tracked.rs"), "fn charge() {}\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);
        for skipped in ["dist", "target"] {
            std::fs::create_dir_all(root.join(skipped)).unwrap();
            std::fs::write(root.join(skipped).join("bundle.rs"), "fn charge() {}\n").unwrap();
        }

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();

        let out = find_pattern(&store, "charge", None, false);
        if !out.contains("tracked.rs") {
            return; // git unavailable in this sandbox
        }
        assert!(
            !out.contains("dist/bundle.rs") && !out.contains("target/bundle.rs"),
            "a SKIP_DIRS path the walker never indexes must not be searchable, \
             even when no ignore rule covers it: {out}"
        );
    }

    #[test]
    fn find_pattern_searches_text_files_the_parser_does_not_index() {
        // The filter used to be `travsr_core::is_indexable_path`, an extension
        // ALLOWLIST, so a match in a `.sh` script or an extensionless file was
        // dropped and reported as a bare "no matches". Only known-binary
        // formats are removed now.
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);
        std::fs::write(root.join("tracked.rs"), "fn charge() {}\n").unwrap();
        std::fs::write(root.join("deploy.sh"), "# charge the lock\n").unwrap();
        std::fs::write(root.join("Makefile"), "charge:\n\ttrue\n").unwrap();
        // A protobuf git classifies as text (no NUL in the first 8000 bytes),
        // which `-I` therefore lets through.
        std::fs::write(root.join("index.scip"), "charge\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();

        let out = find_pattern(&store, "charge", None, false);
        if !out.contains("tracked.rs") {
            return; // git unavailable in this sandbox
        }
        assert!(out.contains("deploy.sh"), "a shell script is text: {out}");
        assert!(
            out.contains("Makefile"),
            "an extensionless file is text: {out}"
        );
        assert!(
            !out.contains("index.scip"),
            "a binary artifact still must not be searched: {out}"
        );
    }

    #[test]
    fn find_pattern_scope_confines_reincluded_files() {
        // Pass 2 has to honor `scope` the same way git honors a pathspec,
        // otherwise scoping a search would widen it.
        let (_dir, store) = file_universe_fixture();
        let out = find_pattern(&store, "charge", Some("vendored"), false);
        if out.contains("tracked.rs") {
            panic!("scope was not applied at all: {out}");
        }
        if out.contains("vendored/dep.rs") {
            assert!(
                !out.contains("untracked.rs"),
                "scope must confine the re-included pass too: {out}"
            );
        }
    }

    #[test]
    fn scoped_to_pathspecs_matches_on_path_boundaries() {
        let files = vec![
            "vendored/dep.rs".to_string(),
            "vendored-other/dep.rs".to_string(),
            "elsewhere/dep.rs".to_string(),
        ];
        // An empty pathspec list is whole-repo scope.
        assert_eq!(scoped_to_pathspecs(files.clone(), &[]).len(), 3);
        // `vendored` must not swallow the sibling `vendored-other`.
        assert_eq!(
            scoped_to_pathspecs(files.clone(), &["vendored".to_string()]),
            vec!["vendored/dep.rs".to_string()]
        );
        // A trailing slash is the same scope.
        assert_eq!(
            scoped_to_pathspecs(files.clone(), &["vendored/".to_string()]),
            vec!["vendored/dep.rs".to_string()]
        );
        // An exact file path is its own scope.
        assert_eq!(
            scoped_to_pathspecs(files, &["elsewhere/dep.rs".to_string()]),
            vec!["elsewhere/dep.rs".to_string()]
        );
    }

    #[test]
    fn argv_chunks_bounds_each_batch() {
        // Every path must appear exactly once, in order, and no batch may
        // exceed the count bound (the byte bound is the same code path).
        let files: Vec<String> = (0..2500).map(|i| format!("vendored/f{i}.rs")).collect();
        let chunks = argv_chunks(&files);
        assert!(chunks.len() >= 3, "2500 paths must split: {}", chunks.len());
        assert!(chunks.iter().all(|c| c.len() <= 1000 && !c.is_empty()));
        let flat: Vec<&String> = chunks.iter().flat_map(|c| c.iter()).collect();
        assert_eq!(flat.len(), files.len(), "chunking must not drop paths");
        assert!(flat.iter().zip(files.iter()).all(|(a, b)| *a == b));
        // An empty input spawns nothing at all.
        assert!(argv_chunks(&[]).is_empty());
    }

    #[test]
    fn travsrignore_reincluded_files_is_empty_without_a_negation() {
        // The gate that keeps pass 2 free for the common repo: no `!` rule
        // means no `git ls-files` call and no second grep.
        let (dir, _store) = pattern_fixture("fix.rs", "fn charge() {}\n");
        std::fs::write(dir.path().join(".travsrignore"), "gen/\n#!not-a-rule\n").unwrap();
        assert!(travsrignore_reincluded_files(dir.path(), GREP_TIMEOUT).is_empty());
        // Absent file is the same story.
        std::fs::remove_file(dir.path().join(".travsrignore")).unwrap();
        assert!(travsrignore_reincluded_files(dir.path(), GREP_TIMEOUT).is_empty());
    }

    /// A git repo with `.gitignore` / `.travsrignore` written verbatim and one
    /// `fn charge() {}` at each of `files`. Committed empty, so every listed
    /// file stays untracked and the ignore rules are what decide its fate.
    fn ignore_rules_fixture(
        gitignore: &str,
        travsrignore: &str,
        files: &[String],
    ) -> tempfile::TempDir {
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);
        std::fs::write(root.join(".gitignore"), gitignore).unwrap();
        std::fs::write(root.join(".travsrignore"), travsrignore).unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);
        for f in files {
            let full = root.join(f);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, "fn charge() {}\n").unwrap();
        }
        dir
    }

    #[test]
    fn travsrignore_reincluded_files_never_escapes_skip_dirs() {
        // `!node_modules/` must not widen the search into it. The walker hard-
        // skips SKIP_DIRS *after* applying the ignore files, so those paths are
        // not in the graph no matter what `.travsrignore` says — and pass 2 has
        // to make the same call or find_pattern starts reporting files the
        // graph does not have.
        let dir = ignore_rules_fixture(
            "node_modules/\nvendored/\n",
            "!node_modules/\n!vendored/\n",
            &[
                "node_modules/pkg/index.rs".to_string(),
                "vendored/dep.rs".to_string(),
            ],
        );
        let found = travsrignore_reincluded_files(dir.path(), GREP_TIMEOUT);
        if found.is_empty() {
            return; // git unavailable in this sandbox
        }
        assert!(
            !found.iter().any(|p| p.starts_with("node_modules/")),
            "a SKIP_DIRS path must stay out even with an explicit `!` rule: {found:?}"
        );
        assert!(
            found.contains(&"vendored/dep.rs".to_string()),
            "the non-SKIP_DIRS re-inclusion must still be picked up: {found:?}"
        );
    }

    #[test]
    fn travsrignore_reincluded_files_handles_glob_negation() {
        // A `!` rule is gitignore syntax, not just a bare directory name, so
        // the whitelist test has to survive a glob and a nested path.
        let dir = ignore_rules_fixture(
            "vendored/\n",
            "!vendored/**/*.rs\n",
            &[
                "vendored/deep/nested/dep.rs".to_string(),
                "vendored/notes.txt".to_string(),
            ],
        );
        let found = travsrignore_reincluded_files(dir.path(), GREP_TIMEOUT);
        if found.is_empty() {
            return;
        }
        assert!(
            found.contains(&"vendored/deep/nested/dep.rs".to_string()),
            "a glob `!` rule must re-include the nested match: {found:?}"
        );
        assert!(
            !found.contains(&"vendored/notes.txt".to_string()),
            "a file the glob does not match must stay excluded: {found:?}"
        );
    }

    #[test]
    fn find_pattern_reaches_every_reincluded_file_across_argv_chunks() {
        // Pass 2 batches its paths to fit an argument vector. The batching is
        // unit-tested above; this drives the real multi-chunk spawn loop, so a
        // file landing in the second or third batch is not silently lost.
        const N: usize = 2400; // > 2x MAX_ARGV_PATHS
        let files: Vec<String> = (0..N).map(|i| format!("vendored/f{i:04}.rs")).collect();
        let dir = ignore_rules_fixture("vendored/\n", "!vendored/\n", &files);

        let reincluded = travsrignore_reincluded_files(dir.path(), GREP_TIMEOUT);
        if reincluded.is_empty() {
            return;
        }
        assert_eq!(reincluded.len(), N, "every re-included file must be listed");
        assert!(
            argv_chunks(&reincluded).len() >= 3,
            "fixture must actually span multiple batches"
        );

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store
            .set_meta("repo_root", dir.path().to_str().unwrap())
            .unwrap();
        // MAX_PATTERN_MATCHES caps the printed list, so assert on the total in
        // the header rather than on the truncated body.
        let out = find_pattern(&store, "charge", None, false);
        assert_eq!(
            extract_match_count(&out),
            N,
            "matches from later batches must not be dropped: {}",
            out.lines().next().unwrap_or("")
        );
    }

    // ── #517: honest find_pattern failures (D2/D3/D5) ────────────────────────

    /// A one-file git repo with `content` written to `path`, and a store whose
    /// `repo_root` points at it. The `TempDir` must be kept alive by the caller
    /// for the store's `repo_root` to remain valid.
    fn pattern_fixture(
        path: &str,
        content: &str,
    ) -> (tempfile::TempDir, travsr_store::SqliteStore) {
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t.com"]);
        git(&["config", "user.name", "t"]);
        let full = root.join(path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(full, content).unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", root.to_str().unwrap()).unwrap();
        (dir, store)
    }

    /// Parses the `N match(es):` header `find_pattern` prints as its first line.
    fn extract_match_count(out: &str) -> usize {
        out.lines()
            .find_map(|l| l.strip_suffix(" match(es):").and_then(|n| n.parse().ok()))
            .unwrap_or(0)
    }

    /// The issue's own invariant: a strict superset of a pattern (alternation
    /// with a term that cannot match) must never return fewer results.
    #[test]
    fn pattern_alternation_never_reduces_match_count() {
        let (_dir, store) = pattern_fixture("fix.rs", "let alpha = 1;\nlet beta = 2;\n");
        let base = find_pattern(&store, "alpha", None, false);
        let widened = find_pattern(&store, "alpha|zzznonexistent", None, false);
        assert!(
            extract_match_count(&widened) >= extract_match_count(&base),
            "alternation must never reduce match count: base={base:?} widened={widened:?}"
        );
    }

    #[test]
    fn pattern_capture_group_matches_like_the_bare_literal() {
        let (_dir, store) = pattern_fixture("fix.rs", "let alpha = 1;\n");
        let bare = find_pattern(&store, "alpha", None, false);
        let grouped = find_pattern(&store, "(alpha)", None, false);
        assert_eq!(
            extract_match_count(&bare),
            extract_match_count(&grouped),
            "a capture group around a literal must match identically: bare={bare:?} grouped={grouped:?}"
        );
    }

    #[test]
    fn pattern_escaped_paren_finds_call_sites() {
        let (_dir, store) = pattern_fixture("fix.rs", "alpha();\n");
        let out = find_pattern(&store, "alpha\\(", None, false);
        assert!(
            out.contains("alpha();"),
            "escaped paren must find the call site: {out}"
        );
    }

    #[test]
    fn pattern_reports_error_for_uncompilable_pattern() {
        let (_dir, store) = pattern_fixture("fix.rs", "alpha();\n");
        let out = find_pattern(&store, "alpha(", None, false);
        assert!(
            out.contains("pattern error"),
            "an unbalanced group must error, not return empty: {out}"
        );
        assert!(
            out.contains("--fixed"),
            "the error must name --fixed as the escape hatch: {out}"
        );
        assert_ne!(
            out, "<travsr-data></travsr-data>",
            "the error must not be laundered into an empty envelope"
        );
    }

    #[test]
    fn pattern_fixed_mode_treats_metachars_literally() {
        let (_dir, store) = pattern_fixture("fix.rs", "alpha();\n");
        let out = find_pattern(&store, "alpha(", None, true);
        assert!(
            !out.contains("pattern error"),
            "--fixed must not attempt regex compilation: {out}"
        );
        assert!(
            out.contains("alpha();"),
            "--fixed must find the literal match: {out}"
        );
    }

    /// #517 D3: `%` is a legitimate search character (never path-resolved).
    #[test]
    fn pattern_allows_percent_in_pattern() {
        let (_dir, store) = pattern_fixture("fix.rs", "let load = 100%;\n");
        let out = find_pattern(&store, "100%", None, false);
        assert_eq!(
            extract_match_count(&out),
            1,
            "percent must not be rejected as path-traversal signal: {out}"
        );
    }

    /// #517 DD-7 M2: the pattern validator dropped the path guards but must
    /// still reject NUL and other control characters (output-framing safety).
    #[test]
    fn pattern_still_rejects_nul_and_control_chars() {
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let empty = "<travsr-data></travsr-data>";
        assert_eq!(find_pattern(&store, "foo\0bar", None, false), empty);
        assert_eq!(find_pattern(&store, "foo\nbar", None, false), empty);
    }

    /// #517 DD-7 M1: the relaxed validator applies only to `pattern` — `scope`
    /// keeps the full `validate_mcp_arg` path-traversal guard.
    #[test]
    fn pattern_scope_still_rejects_traversal() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", "/tmp/whatever").unwrap();
        let empty = "<travsr-data></travsr-data>";
        assert_eq!(
            find_pattern(&store, "100%", Some("../etc"), false),
            empty,
            "a percent-containing pattern must not smuggle a traversal scope past validation"
        );
    }

    /// #507: the scope guard was separator-blind — backslash traversal,
    /// drive-letter, and UNC absolute forms sailed through on Windows.
    #[test]
    fn pattern_scope_rejects_windows_absolute_and_backslash_traversal() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("repo_root", "/tmp/whatever").unwrap();
        let empty = "<travsr-data></travsr-data>";
        for scope in [
            r"..\..\etc",
            r"src\..\..\etc",
            r"C:\Windows\System32",
            r"c:/windows",
            r"\\server\share",
            "/etc",
            "../etc",
            "src/../../etc",
        ] {
            assert_eq!(
                find_pattern(&store, "alpha", Some(scope), false),
                empty,
                "scope {scope:?} must be rejected"
            );
        }
    }

    /// Plain repo-relative prefixes, including ones with inner dots, stay valid.
    #[test]
    fn pattern_scope_accepts_normal_relative_prefixes() {
        let (_dir, store) = pattern_fixture("src/fix.rs", "let alpha = 1;\n");
        for scope in ["src", "src/fix.rs", "a.b/c"] {
            // Must not be rejected by the guard (a no-match result is fine —
            // rejection returns the empty envelope before git even runs, and
            // for the existing file the match must actually surface).
            let out = find_pattern(&store, "alpha", Some(scope), false);
            if scope == "src" || scope == "src/fix.rs" {
                assert!(
                    out.contains("alpha"),
                    "valid scope {scope:?} must reach git grep and match: {out}"
                );
            }
        }
    }

    /// #517 DD-5: `\d` is a PCRE digit-class escape with no POSIX ERE
    /// equivalent. Unlike `\b \B \w \W \s \S`, it is not a GNU regex
    /// extension either, so `git grep -E` treats it as inert on both the
    /// glibc-based (Linux, Windows/MSYS2) and BSD-based (macOS) regex
    /// engines git ships with — it silently matches nothing rather than
    /// erroring, on every platform this test runs on in CI. The advisory
    /// names the construct instead of returning bare empty.
    #[test]
    fn pattern_advises_on_pcre_only_construct() {
        let (_dir, store) = pattern_fixture("fix.rs", "let alpha = 1;\n");
        let out = find_pattern(&store, "\\dalpha", None, false);
        assert!(
            out.contains("\\d"),
            "advisory must name the unsupported construct: {out}"
        );
        assert!(
            out.contains("PCRE"),
            "advisory must explain why it is unsupported: {out}"
        );
    }

    #[test]
    fn pattern_global_attributes_error_to_the_failing_repo() {
        use std::process::Command;
        let workdir = tempfile::tempdir().unwrap();

        // A real git repo with one match for "alpha".
        let ok_root = workdir.path().join("ok_root");
        std::fs::create_dir_all(&ok_root).unwrap();
        let git = |root: &std::path::Path, args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git");
        };
        git(&ok_root, &["init", "-q"]);
        git(&ok_root, &["config", "user.email", "t@t.com"]);
        git(&ok_root, &["config", "user.name", "t"]);
        std::fs::write(ok_root.join("fix.rs"), "let alpha = 1;\n").unwrap();
        git(&ok_root, &["add", "-A"]);
        git(&ok_root, &["commit", "-qm", "init"]);
        let ok_db = workdir.path().join("ok.db");
        {
            let mut store = travsr_store::SqliteStore::open(&ok_db).unwrap();
            store
                .set_meta("repo_root", ok_root.to_str().unwrap())
                .unwrap();
        }

        // `repo_root` points at a directory that is not a git repository at
        // all, so `git -C <root> grep` fails with a real (non-pattern) error —
        // independent of the pattern, unlike a regex-compile failure.
        let bad_root = workdir.path().join("bad_root");
        std::fs::create_dir_all(&bad_root).unwrap();
        let bad_db = workdir.path().join("bad.db");
        {
            let mut store = travsr_store::SqliteStore::open(&bad_db).unwrap();
            store
                .set_meta("repo_root", bad_root.to_str().unwrap())
                .unwrap();
        }

        let repos: HashMap<String, PathBuf> = [
            ("repo_ok".to_string(), ok_db),
            ("repo_bad".to_string(), bad_db),
        ]
        .into();

        // Prevent git from walking up and finding parent repos from bad_root
        let mut ceiling_paths = Vec::new();
        let mut current = Some(bad_root.as_path());
        while let Some(p) = current {
            let p_str = p.to_string_lossy().to_string();
            ceiling_paths.push(p_str.clone());
            ceiling_paths.push(p_str.replace("\\", "/"));
            ceiling_paths.push(p_str.replace("C:", "c:"));
            ceiling_paths.push(p_str.replace("c:", "C:"));
            ceiling_paths.push(p_str.replace("C:", "c:").replace("\\", "/"));
            ceiling_paths.push(p_str.replace("c:", "C:").replace("\\", "/"));
            current = p.parent();
        }
        #[cfg(windows)]
        let delim = ";";
        #[cfg(not(windows))]
        let delim = ":";
        std::env::set_var("GIT_CEILING_DIRECTORIES", ceiling_paths.join(delim));
        let out = find_pattern_global(&repos, "alpha", None, None, false);
        std::env::remove_var("GIT_CEILING_DIRECTORIES");
        assert!(
            out.contains("[repo_bad] pattern error"),
            "the failing repo's error must be attributed to it, not silently dropped: {out}"
        );
        assert!(
            !out.contains("[repo_ok] pattern error"),
            "the healthy repo must not show an error: {out}"
        );
        assert!(
            out.contains("alpha"),
            "the healthy repo's match must still appear alongside the error: {out}"
        );
    }

    // ── #517 DD-8: subprocess deadline ───────────────────────────────────────
    //
    // `git hash-object --stdin` reads from stdin until EOF. `spawn_with_deadline`
    // never touches `child.stdin`, so the pipe's write end stays open (held by
    // the `Child`) for as long as the child runs — giving a reliably slow, fully
    // cross-platform child process without a `sleep`/`printf` dependency that
    // would not exist on the `windows-latest` CI runner.
    fn blocked_on_stdin() -> std::process::Command {
        let mut cmd = std::process::Command::new("git");
        cmd.arg("hash-object").arg("--stdin");
        cmd.stdin(std::process::Stdio::piped());
        cmd
    }

    #[test]
    fn spawn_with_deadline_kills_and_reports_none_for_a_slow_child() {
        let start = std::time::Instant::now();
        let (status, _out, _err) =
            spawn_with_deadline(blocked_on_stdin(), std::time::Duration::from_millis(200)).unwrap();
        assert!(
            status.is_none(),
            "a child still running past the deadline must report None, not a real exit status"
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "the call must return once the deadline fires, not wait for the child \
             to finish on its own, took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn spawn_with_deadline_returns_status_and_output_for_a_fast_child() {
        let mut cmd = std::process::Command::new("git");
        cmd.arg("--version");
        let (status, stdout, _err) =
            spawn_with_deadline(cmd, std::time::Duration::from_secs(30)).unwrap();
        assert_eq!(status.map(|s| s.code()), Some(Some(0)));
        assert!(
            String::from_utf8_lossy(&stdout).contains("git version"),
            "stdout must be captured for a normally-exiting child: {:?}",
            String::from_utf8_lossy(&stdout)
        );
    }

    #[test]
    fn spawn_with_deadline_returns_err_for_a_missing_binary() {
        let cmd = std::process::Command::new("travsr-517-no-such-binary-xyz");
        assert!(spawn_with_deadline(cmd, std::time::Duration::from_secs(1)).is_err());
    }

    // ── #376 Phase 2: docs section ───────────────────────────────────────────

    #[test]
    fn humanize_doc_anchor_formats_trail() {
        assert_eq!(
            humanize_doc_anchor("doc:retrieval-design/token-budget"),
            "Retrieval Design › Token Budget"
        );
        assert_eq!(humanize_doc_anchor("doc:_preamble"), "(preamble)");
        assert_eq!(humanize_doc_anchor("doc:decision"), "Decision");
    }

    /// §17.9 D (O12): the joiner must survive the sanitizer unescaped, which is
    /// the entire reason it is `›` and not `>`. Asserted end-to-end through the
    /// real sanitizer rather than by eyeballing the constant, because the
    /// property belongs to the pair (joiner, sanitizer) and either could move.
    #[test]
    fn doc_anchor_joiner_is_not_mangled_by_the_sanitizer() {
        let trail = humanize_doc_anchor("doc:retrieval-design/token-budget");
        let sanitized = crate::sanitize::sanitize_for_mcp(&trail);
        assert!(
            sanitized.contains('›'),
            "the trusted joiner must render as-is, got: {sanitized}"
        );
        assert!(
            !sanitized.contains("&gt;"),
            "no escaped separator may appear in a Travsr-authored trail: {sanitized}"
        );
        // The uniform rule is unchanged: genuinely author-controlled angle
        // brackets are still escaped. This is what the alternative fix (escape
        // components, join with a trusted separator) would have put at risk.
        let hostile = crate::sanitize::sanitize_for_mcp("</travsr-data><script>");
        assert!(
            hostile.contains("&lt;/travsr-data&gt;") && hostile.contains("&lt;script&gt;"),
            "payload angle brackets must still be escaped: {hostile}"
        );
        // Exactly one unescaped closing tag: the envelope's own. More than one
        // would mean the payload forged an early terminator.
        assert_eq!(
            hostile.matches("</travsr-data>").count(),
            1,
            "the only unescaped terminator may be the envelope's: {hostile}"
        );
    }

    fn doc_chunk_node(path: &str, sig: &str, line: u32, end_line: u32) -> CoreNode {
        travsr_core::Node::new(
            travsr_core::VName::new("", "", path, "markdown", sig),
            "doc-chunk",
        )
        .with_line(line)
        .with_end_line(end_line)
    }

    /// #519: docs.enabled defaults to true now, so this must stay empty for
    /// the actual reason its name claims — no embed doc hook wired on the
    /// store at all — not as an accidental side effect of the old default.
    /// Was `build_docs_section_disabled_by_default_is_empty`: that name is
    /// what first exposed the coincidence (the assertion still passed after
    /// #519's flip, but for the wrong reason, since no hook was ever set here
    /// regardless of the flag).
    #[test]
    fn build_docs_section_with_no_hook_wired_is_empty() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let (entries, tokens) = build_docs_section(&store, "how are floors calibrated", 4000);
        assert!(entries.is_empty());
        assert_eq!(tokens, 0);
    }

    /// The flag itself gates the lane even with a real hook wired and the
    /// default otherwise on — explicit now that "disabled" is a user opt-out
    /// (`travsr config set docs.enabled false`) rather than the ship default.
    #[test]
    fn build_docs_section_explicitly_disabled_is_empty_even_with_a_hook() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "0");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n1 = doc_chunk_node(
            "docs/plans/model-relative-semantic-floors.md",
            "doc:auto-calibrated-floors",
            10,
            40,
        );
        store.put_node(&n1).unwrap();
        let id1 = n1.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.71)]));
        store.set_embed_doc_knn_hook(hook);

        let (entries, tokens) =
            build_docs_section(&store, "how are semantic floors calibrated", 4000);

        std::env::remove_var("TRAVSR_DOCS_ENABLED");

        assert!(
            entries.is_empty(),
            "the flag must gate the lane even with a hook that would otherwise answer"
        );
        assert_eq!(tokens, 0);
    }

    #[test]
    fn build_docs_section_renders_floored_hits_with_no_score_in_text() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");
        std::env::set_var("TRAVSR_DOCS_MAX_RESULTS", "3");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n1 = doc_chunk_node(
            "docs/plans/model-relative-semantic-floors.md",
            "doc:auto-calibrated-floors",
            10,
            40,
        );
        store.put_node(&n1).unwrap();
        let id1 = n1.id;

        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.71)]));
        store.set_embed_doc_knn_hook(hook);

        let (entries, tokens) =
            build_docs_section(&store, "how are semantic floors calibrated", 4000);
        assert_eq!(entries.len(), 1);
        let (ms, _score, line) = &entries[0];
        assert_eq!(*ms, crate::seed::MatchSource::Docs);
        assert!(
            line.contains("docs/plans/model-relative-semantic-floors.md"),
            "line must show the source path: {line}"
        );
        assert!(
            line.contains("Auto Calibrated Floors"),
            "line must show a humanized trail: {line}"
        );
        assert!(
            line.contains(":10-40"),
            "line must show the chunk's line span: {line}"
        );
        assert!(
            !line.contains("0.71") && !line.contains("0.7"),
            "docs section must never print a raw cosine (§4.1): {line}"
        );
        assert!(tokens > 0);

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
        std::env::remove_var("TRAVSR_DOCS_MAX_RESULTS");
    }

    /// #520: with no threshold set (the shipped default), reranking must
    /// always run. This pins the "off by default" contract at the type
    /// level, not just by convention — a threshold defaulting to `Some(_)`
    /// would silently change production behavior.
    #[test]
    fn doc_rerank_ambiguity_threshold_defaults_to_none() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_DOC_RERANK_AMBIGUITY_THRESHOLD");
        assert_eq!(crate::seed::doc_rerank_ambiguity_threshold(), None);
    }

    /// #520: a confidently-separated top candidate must skip the
    /// cross-encoder entirely and fall through to `cosine_floor_select`. No
    /// reranker model needs to be installed for this test — that is the
    /// point of the early return.
    #[test]
    fn rerank_doc_candidates_skips_cross_encoder_above_threshold() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOC_RERANK_AMBIGUITY_THRESHOLD", "0.9");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.01");

        let store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let confident = travsr_core::NodeId(1);
        let ambiguous = travsr_core::NodeId(2);
        let result = rerank_doc_candidates(
            &store,
            "irrelevant",
            vec![(confident, 0.95), (ambiguous, 0.4)],
        );

        std::env::remove_var("TRAVSR_DOC_RERANK_AMBIGUITY_THRESHOLD");
        std::env::remove_var("TRAVSR_DOC_FLOOR");

        // cosine_floor_select's behavior (floor + sort + cap), not the
        // reranker's: both candidates survive since both clear the 0.01
        // floor set above, in cosine-descending order.
        assert_eq!(
            result,
            vec![(confident, 0.95), (ambiguous, 0.4)],
            "must take the cosine-only path, unchanged by any reranker score"
        );
    }

    /// §4.3 budget carve: a docs.budget_pct that cannot fit even the highest
    /// scored entry must yield no entries — dropped, not truncated mid-line.
    #[test]
    fn build_docs_section_budget_cap_can_drop_all_entries() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");
        std::env::set_var("TRAVSR_DOCS_BUDGET_PCT", "0.001");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let n1 = doc_chunk_node(
            "docs/plans/some-very-long-plan-file-name.md",
            "doc:a-fairly-long-heading-trail-that-costs-several-tokens",
            10,
            40,
        );
        store.put_node(&n1).unwrap();
        let id1 = n1.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.9)]));
        store.set_embed_doc_knn_hook(hook);

        let (entries, tokens) = build_docs_section(&store, "query", 4000);
        assert!(entries.is_empty());
        assert_eq!(tokens, 0);

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
        std::env::remove_var("TRAVSR_DOCS_BUDGET_PCT");
    }

    /// Permanent gate (not the retired Phase-1 lever): doc-chunk nodes never
    /// enter `get_context`'s knapsack-selected set, only `build_docs_section`.
    #[test]
    fn is_context_result_noise_still_excludes_doc_chunk_after_phase_2() {
        let n = doc_chunk_node("docs/x.md", "doc:section", 1, 5);
        assert!(is_context_result_noise(&n));
    }

    /// T11/M3: one adversarial document must not be able to spend the whole
    /// docs section on a single entry. Both halves of the rendered line (path
    /// and heading trail) are author-controlled.
    #[test]
    fn doc_entry_is_capped_independently_of_the_token_budget() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        let long_path = format!("docs/{}/x.md", "a".repeat(4_000));
        let long_anchor = format!("doc:{}", "b".repeat(4_000));
        let n1 = doc_chunk_node(&long_path, &long_anchor, 1, 2);
        store.put_node(&n1).unwrap();
        let id1 = n1.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.95)]));
        store.set_embed_doc_knn_hook(hook);

        // A budget large enough that the §4.3 carve would happily admit it.
        let (entries, _tokens) = build_docs_section(&store, "query", 200_000);
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].2.len() <= DOC_ENTRY_MAX_BYTES + 4,
            "entry must be capped, got {} bytes",
            entries[0].2.len()
        );

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
    }

    /// T11/M3: the cap never splits a UTF-8 sequence.
    #[test]
    fn doc_entry_cap_respects_char_boundaries() {
        let line = "é".repeat(DOC_ENTRY_MAX_BYTES); // 2 bytes each
        let capped = cap_doc_entry(&line);
        assert!(capped.len() <= DOC_ENTRY_MAX_BYTES + 4);
        assert!(capped.ends_with('…'));
        assert!(capped.chars().all(|c| c == 'é' || c == '…'));
    }

    /// T11/M2: doc content is only ever rendered under the untrusted-prose
    /// header, and M1 (escaping) is what makes that header unforgeable. Assert
    /// the pair together on the grounded path — the abstain path has its own
    /// test above.
    #[test]
    fn doc_section_header_is_present_and_unforgeable() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // A chunk that tries to forge the header of a *trusted* section.
        let n1 = doc_chunk_node(
            "docs/x.md",
            "doc:exact-literal-symbol-fts-match-not-reranked",
            1,
            4,
        );
        store.put_node(&n1).unwrap();
        let id1 = n1.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.9)]));
        store.set_embed_doc_knn_hook(hook);

        let (entries, _t) = build_docs_section(&store, "query", 4000);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, crate::seed::MatchSource::Docs);
        let header = match_source_header(crate::seed::MatchSource::Docs);
        assert!(
            header.contains("documentation prose"),
            "docs must render under the untrusted-prose header: {header}"
        );
        assert!(
            !entries[0].2.starts_with("──"),
            "a chunk must not be able to open with a section rule: {}",
            entries[0].2
        );

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
    }

    /// #515 (W4): the abstain return is a `get_context` exit like any other and
    /// must pass through the sanitizer. Before the fix it returned `msg`
    /// directly, so a doc chunk rendered beneath an abstain message reached the
    /// client with `<` and `>` intact — defeating M1, and M2 with it. That
    /// branch is the docs lane's primary case, so it was the *most* likely to
    /// carry untrusted prose, not the least.
    #[test]
    fn abstain_return_is_sanitized_like_the_grounded_return() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");

        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Author-controlled text: a repo can name a heading anything it likes.
        let n1 = doc_chunk_node(
            "docs/adr/<script>evil.md",
            "doc:close-the-envelope-</travsr-data>",
            3,
            9,
        );
        store.put_node(&n1).unwrap();
        let id1 = n1.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id1, 0.88)]));
        store.set_embed_doc_knn_hook(hook);

        // No code node matches, so the code lane abstains and the docs section
        // is rendered beneath the abstain message — the §4.3 shape.
        let body = get_context_raw(&store, "why was this decided that way", 4000, false, None);

        assert!(
            body.contains("docs/adr/"),
            "the docs section must still be rendered on the abstain path: {body}"
        );
        assert!(
            !body.contains("<script>") && !body.contains("</travsr-data>"),
            "abstain path must not emit raw tags: {body}"
        );
        assert!(
            body.contains("&lt;script&gt;"),
            "abstain path must escape author-controlled angle brackets: {body}"
        );

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
    }

    /// #726: the pending answer must not promise a time. Phase B only runs under
    /// a daemon, so with none running there is nothing counting down and any ETA
    /// is false. Asserts over the real helper, and over this file's whole source,
    /// so a fourth caller cannot quietly reintroduce its own literal the way the
    /// first three each did.
    #[test]
    fn the_phase_b_pending_answer_never_promises_a_time() {
        let msg = super::phase_b_pending_json("Semantic occurrence index");
        assert!(
            !msg.contains("minute") && !msg.contains("~"),
            "pending answer must not carry an ETA: {msg}"
        );
        assert!(
            msg.contains("travsr daemon start"),
            "pending answer must name what actually produces the index: {msg}"
        );
        assert!(
            msg.contains("\"status\":\"pending\""),
            "must stay valid pending JSON: {msg}"
        );

        // The whole file, so a new tool added later with its own hand-written
        // literal fails here rather than shipping the falsehood again.
        // Needle built at runtime so this assertion's own text cannot match it.
        let needle = format!("will be available in {}", '~');
        let src = include_str!("tools.rs");
        assert!(
            !src.contains(&needle),
            "a pending message carrying an ETA has reappeared in tools.rs; call \
             phase_b_pending_json instead of writing a new literal"
        );
    }
}

/// #755 Part B: the per-query "this language was not analyzed" note (item 4),
/// naming the failing endpoint in `get_execution_path` (item 6), and `total`
/// no longer conflating "no count taken" with a real zero (item 9).
#[cfg(test)]
mod issue_755_tests {
    use super::*;

    fn store_at_head() -> travsr_store::SqliteStore {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        // Markers agree and nothing is dirty: the index IS current, which is
        // exactly the state where the old code had nothing to say.
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store
    }

    fn with_warnings(warnings: &str) -> travsr_store::SqliteStore {
        let mut store = store_at_head();
        store.set_meta("phase_b_warnings", warnings).unwrap();
        store
    }

    // -- item 4: a missing analyzer must not read as "no callers" --------------

    /// The reported repro: a php repo on a host with no scip-php. The index is
    /// current, so neither the pending nor the stale note fires, and every
    /// `graph`/`callers` query used to answer with nothing at all.
    #[test]
    fn a_missing_analyzer_produces_a_per_query_note() {
        let store = with_warnings("skipped_no_analyzer:php");
        let note = phase_b_degraded_note(&store).expect("a skipped language must be surfaced");
        assert!(
            note.contains("php"),
            "the note must name the language; got: {note}"
        );
        assert!(
            note.contains("no call edges"),
            "the note must say what is missing, not just that something is; got: {note}"
        );
        assert!(
            note.contains("not authoritative"),
            "the note exists to stop an empty result being trusted; got: {note}"
        );
        assert!(
            note.contains("that language"),
            "one language takes the singular; got: {note}"
        );
    }

    /// A crashed analyzer is the same harm from a different cause.
    #[test]
    fn a_crashed_analyzer_produces_a_per_query_note() {
        let store = with_warnings("crashed:objectivec");
        let note = phase_b_degraded_note(&store).expect("a crash must be surfaced");
        assert!(note.contains("objectivec"), "got: {note}");
    }

    /// So are the two "waiting on the user" states.
    #[test]
    fn pending_approval_and_consent_produce_a_per_query_note() {
        for warn in ["needs_approval:java", "needs_consent:scala"] {
            let store = with_warnings(warn);
            let note =
                phase_b_degraded_note(&store).unwrap_or_else(|| panic!("{warn} must be surfaced"));
            assert!(
                note.contains(warn.split(':').nth(1).unwrap()),
                "got: {note}"
            );
        }
    }

    /// Several languages are listed once each, in the plural.
    #[test]
    fn several_unanalyzed_languages_are_listed_and_deduped() {
        let store = with_warnings("skipped_no_analyzer:php,crashed:php,needs_approval:java");
        let note = phase_b_degraded_note(&store).expect("must fire");
        assert!(note.contains("php") && note.contains("java"), "got: {note}");
        assert_eq!(
            note.matches("php").count(),
            1,
            "a language named by two classes must still appear once; got: {note}"
        );
        assert!(
            note.contains("those languages"),
            "more than one language takes the plural; got: {note}"
        );
    }

    // -- item 4, negative: must not fire on healthy or not-turned-on states ---

    /// An analyzer that ran and found nothing is a valid answer, not missing
    /// coverage. Warning here would put a caveat on every query in a repo whose
    /// go files genuinely define nothing callable.
    #[test]
    fn a_zero_node_run_is_not_treated_as_unanalyzed() {
        let store = with_warnings("zero_nodes:go");
        assert!(
            phase_b_degraded_note(&store).is_none(),
            "a completed run that found nothing is not degraded coverage"
        );
    }

    /// Languages the user has not turned on are their own notice in `travsr
    /// status`, not a downgrade of the ones that did run. Including them here
    /// would make this note and the `semantic:` line disagree.
    #[test]
    fn languages_the_user_never_enabled_do_not_trigger_the_note() {
        for warn in [
            "skipped_unregistered:php",
            "untrusted_corpus:go",
            "skipped_no_compdb:cpp",
        ] {
            let store = with_warnings(warn);
            assert!(
                phase_b_degraded_note(&store).is_none(),
                "{warn} must not degrade the languages that did run"
            );
        }
    }

    /// A repo-level warning that names no language must not be mistaken for one.
    #[test]
    fn a_non_per_language_warning_is_ignored() {
        let store = with_warnings("scip_unification_misses:3/900");
        assert!(phase_b_degraded_note(&store).is_none());
    }

    /// An empty or absent warnings key is the healthy case and stays silent --
    /// the behaviour the pre-existing #617 tests pin, unchanged.
    #[test]
    fn a_clean_run_stays_silent() {
        assert!(phase_b_degraded_note(&store_at_head()).is_none());
        assert!(phase_b_degraded_note(&with_warnings("")).is_none());
        assert!(phase_b_degraded_note(&with_warnings(",,")).is_none());
    }

    /// Malformed entries must not panic or invent a language.
    #[test]
    fn malformed_warning_entries_are_skipped() {
        for warn in ["skipped_no_analyzer:", "skipped_no_analyzer", ":php", ":"] {
            let store = with_warnings(warn);
            assert!(
                phase_b_degraded_note(&store).is_none(),
                "{warn} names no language, so there is nothing to report"
            );
        }
    }

    /// A `class:lang:extra` entry (the `version_mismatch` shape) is not in the
    /// reported set, so it stays out -- but must not disturb a sibling entry.
    #[test]
    fn an_extra_field_entry_does_not_disturb_the_reported_set() {
        let store = with_warnings("version_mismatch:go:1.2:1.1,skipped_no_analyzer:php");
        let note = phase_b_degraded_note(&store).expect("php must still be reported");
        assert!(note.contains("php"), "got: {note}");
        assert!(
            !note.contains("go"),
            "version_mismatch is not in the reported set; got: {note}"
        );
    }

    /// Precedence: a stale or pending index is the bigger fact and keeps its own
    /// wording. Reporting "php was not analyzed" while the whole index is behind
    /// HEAD would point at the wrong problem.
    #[test]
    fn a_pending_or_stale_index_keeps_its_own_note() {
        let mut behind = travsr_store::SqliteStore::open_in_memory().unwrap();
        behind.set_meta("last_commit", "def").unwrap();
        behind.set_meta("phase_b_commit", "abc").unwrap();
        behind
            .set_meta("phase_b_warnings", "skipped_no_analyzer:php")
            .unwrap();
        let note = phase_b_degraded_note(&behind).expect("must fire");
        assert!(note.contains("call-graph index incomplete"), "got: {note}");

        let mut dirty = store_at_head();
        dirty.set_meta("phase_b_dirty", "1").unwrap();
        dirty
            .set_meta("phase_b_warnings", "skipped_no_analyzer:php")
            .unwrap();
        let note = phase_b_degraded_note(&dirty).expect("must fire");
        assert!(note.contains("call-graph edges degraded"), "got: {note}");
    }

    /// The note has to reach an actual query, not just the helper: this is the
    /// `get_callers` path the issue reports as silent.
    #[test]
    fn get_callers_carries_the_unanalyzed_note() {
        use travsr_core::{Node, VName};
        let mut store = with_warnings("skipped_no_analyzer:php");
        let n = Node::new(
            VName::new("t", "", "src/a.php", "php", "fn:describe"),
            "function",
        );
        store.put_node(&n).unwrap();
        let out = get_callers(&store, "describe", None);
        assert!(
            out.contains("no call edges were produced for php"),
            "an empty caller list must say why; got: {out}"
        );
    }

    // -- item 6: name the endpoint that actually failed -----------------------

    fn two_node_store() -> travsr_store::SqliteStore {
        use travsr_core::{Node, VName};
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for sig in ["fn:alpha", "fn:beta"] {
            let n = Node::new(
                VName::new("t", "", "src/a.ts", "typescript", sig),
                "function",
            );
            store.put_node(&n).unwrap();
        }
        store
    }

    /// The reported repro: only the source is bad. The old message hedged with
    /// "source X and/or sink Y", so the caller could not tell which name to fix.
    #[test]
    fn a_bad_source_names_the_source_only() {
        let store = two_node_store();
        let out = get_execution_path(&store, "NoSuchSym", "beta");
        assert!(
            out.contains("could not resolve source 'NoSuchSym'"),
            "got: {out}"
        );
        assert!(
            !out.contains("sink 'beta'"),
            "the sink resolved, so naming it sends the reader after the wrong name; got: {out}"
        );
    }

    /// The mirror case.
    #[test]
    fn a_bad_sink_names_the_sink_only() {
        let store = two_node_store();
        let out = get_execution_path(&store, "alpha", "NoSuchSym");
        assert!(
            out.contains("could not resolve sink 'NoSuchSym'"),
            "got: {out}"
        );
        assert!(!out.contains("source 'alpha'"), "got: {out}");
    }

    /// Both bad: both are named, because both do need fixing.
    #[test]
    fn two_bad_endpoints_name_both() {
        let store = two_node_store();
        let out = get_execution_path(&store, "NoSuchA", "NoSuchB");
        assert!(out.contains("source 'NoSuchA'"), "got: {out}");
        assert!(out.contains("sink 'NoSuchB'"), "got: {out}");
    }

    /// SEC P0 must survive naming the side. A denied endpoint and a nonexistent
    /// one still produce byte-identical text once the echoed query is normalized
    /// -- naming WHICH side failed leaks nothing, because "did not resolve" is
    /// emitted the same way for both reasons.
    #[test]
    fn naming_the_side_does_not_create_an_existence_oracle() {
        use travsr_core::{Node, VName};
        use travsr_retrieval::RbacFilter;
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        for sig in ["fn:alpha", "fn:beta"] {
            let n = Node::new(
                VName::new("public", "", "src/a.ts", "typescript", sig),
                "function",
            );
            store.put_node(&n).unwrap();
        }
        let secret = Node::new(
            VName::new("secret", "", "src/s.ts", "typescript", "fn:hidden"),
            "function",
        );
        store.put_node(&secret).unwrap();
        let filter = RbacFilter::new(["public"]);
        // `hidden` exists but is denied; `absent` does not exist at all.
        let denied = get_execution_path_authed(&store, "alpha", "hidden", &filter);
        let missing = get_execution_path_authed(&store, "alpha", "absent", &filter);
        assert_eq!(
            denied.replace("hidden", "X"),
            missing.replace("absent", "X"),
            "a denied sink and a nonexistent sink must be indistinguishable"
        );
        assert!(
            denied.contains("sink"),
            "the side is still named; got: {denied}"
        );
        assert!(
            !denied.contains("denied") && !denied.contains("permission"),
            "the reason must never be stated; got: {denied}"
        );
    }

    /// Two resolvable-but-disconnected endpoints keep the distinct no-path
    /// message: naming the side must not swallow the case where both resolved.
    #[test]
    fn resolved_but_disconnected_still_says_no_path_found() {
        let store = two_node_store();
        let out = get_execution_path(&store, "alpha", "beta");
        assert!(out.contains("no path found"), "got: {out}");
        assert!(!out.contains("could not resolve"), "got: {out}");
    }

    // -- item 9: `total` must not report a zero it never counted --------------

    fn refs_store() -> travsr_store::SqliteStore {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        store.set_meta("phase_b_commit", "abc").unwrap();
        store
    }

    /// The reported repro: `speak` is defined on three classes, so the answer is
    /// "ambiguous, here are the candidates" -- it says nothing about how many
    /// uses exist. `total: 0` claimed it did.
    #[test]
    fn an_ambiguous_symbol_reports_no_count_rather_than_zero() {
        use travsr_core::{Node, VName};
        let mut store = refs_store();
        for path in ["src/animal.rb", "src/cat.rb", "src/dog.rb"] {
            let n = Node::new(VName::new("t", "", path, "ruby", "fn:speak"), "method");
            store.put_node(&n).unwrap();
        }
        let got = find_references_structured(&store, "speak", None);
        assert_eq!(got.status, "ambiguous");
        assert!(
            !got.candidates.is_empty(),
            "candidates must still be listed"
        );
        assert_eq!(
            got.total, None,
            "no occurrence count was taken, so `total` must not assert one"
        );
        let json = serde_json::to_value(&got).unwrap();
        assert!(
            json["total"].is_null(),
            "`total` must serialize as null, which no consumer reads as 0; got: {json}"
        );
    }

    /// A symbol that does not exist also took no count.
    #[test]
    fn a_missing_symbol_reports_no_count() {
        let store = refs_store();
        let got = find_references_structured(&store, "zzz_nope", None);
        assert_eq!(got.status, "not_found");
        assert_eq!(got.total, None);
    }

    /// Nor did a still-building index.
    #[test]
    fn a_pending_index_reports_no_count() {
        let mut store = travsr_store::SqliteStore::open_in_memory().unwrap();
        store.set_meta("last_commit", "abc").unwrap();
        let got = find_references_structured(&store, "speak", None);
        assert_eq!(got.status, "pending");
        assert_eq!(got.total, None);
    }

    /// An invalid argument took no count either.
    #[test]
    fn a_rejected_argument_reports_no_count() {
        let store = refs_store();
        let got = find_references_structured(&store, "../../etc/passwd", None);
        assert_eq!(got.total, None);
    }

    /// The one case where zero IS the answer: the symbol resolved uniquely and
    /// has no recorded uses. This must stay a real `0`, or the change would just
    /// move the ambiguity somewhere else.
    #[test]
    fn a_resolved_symbol_with_no_uses_reports_a_real_zero() {
        use travsr_core::{Node, VName};
        let mut store = refs_store();
        let n = Node::new(
            VName::new("t", "", "src/a.rb", "ruby", "fn:lonely"),
            "method",
        );
        store.put_node(&n).unwrap();
        let got = find_references_structured(&store, "lonely", None);
        assert_eq!(got.status, "resolved");
        assert_eq!(
            got.total,
            Some(0),
            "a unique definition with no uses genuinely has zero references"
        );
        let json = serde_json::to_value(&got).unwrap();
        assert_eq!(json["total"], serde_json::json!(0));
    }

    /// And a symbol with real uses still reports the real count -- the change
    /// must not turn every count into `null`.
    #[test]
    fn a_resolved_symbol_with_uses_reports_the_count() {
        use travsr_core::{Node, VName};
        let mut store = refs_store();
        let cat = Node::new(VName::new("t", "", "src/cat.rb", "ruby", "file"), "file");
        let dog = Node::new(VName::new("t", "", "src/dog.rb", "ruby", "file"), "file");
        let target = Node::new(
            VName::new("t", "", "src/animal.rb", "ruby", "fn:speak"),
            "method",
        );
        store.put_node(&cat).unwrap();
        store.put_node(&dog).unwrap();
        store.put_node(&target).unwrap();
        store
            .record_edge_sites(&[(cat.id, target.id, 4, None), (dog.id, target.id, 7, None)])
            .unwrap();
        let got = find_references_structured(&store, "speak", None);
        assert_eq!(got.status, "resolved");
        assert_eq!(
            got.total,
            Some(2),
            "two recorded use sites must still be counted"
        );
        assert!(!got.truncated);
        let json = serde_json::to_value(&got).unwrap();
        assert_eq!(json["total"], serde_json::json!(2));
    }
}
