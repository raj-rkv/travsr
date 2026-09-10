//! Shared CLI query execution (#318 O1).
//!
//! `travsr ask` / `travsr graph` / `travsr status` produce their data through
//! the functions in this module, on two routes:
//!
//! 1. **Daemon route** — the daemon's control socket receives a
//!    `ControlMessage::Query`, runs these functions against its warm store,
//!    and ships the payload back as JSON. No per-command store open.
//! 2. **Direct route** — no daemon running: the CLI opens the store
//!    (read-only fast path) and calls the same functions in-process.
//!
//! Because both routes execute identical code, CLI output is byte-identical
//! regardless of routing. The serialized payload shapes are versioned by
//! `travsr_ipc::QUERY_PROTOCOL_VERSION` — bump it on any breaking change here.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use travsr_core::{display_label, is_noise_node, NodeId};

type KnnFn<'a> = &'a dyn Fn(&str, u32) -> Vec<(NodeId, f32)>;
use travsr_retrieval::{
    context_candidates, knapsack, ppr_weighted, token_cost, EdgeFilter, OpenFilter,
};
use travsr_store::{SqliteStore, Store, StoreMigratable};

// #478 RFC-023 §6.1/WS-8: `explain_query`'s report type lives in `seed` (an
// internal, crate-private module) alongside the other seed-building internals
// it's built from; re-exported here since `query` is this crate's one public
// surface for CLI-facing payload types.
pub use crate::seed::{
    ExplainDisposition, ExplainLeg, ExplainLegMatches, ExplainReport, ExplainThresholds,
    ExplainToken,
};

/// Token budget for `travsr ask` and the default for `travsr graph --budget`.
/// Matches the MCP `get_context` default.
pub const DEFAULT_TOKEN_BUDGET: usize = 4096;

// ── Payload types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueryDirection {
    Deps,
    Callers,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueryEdgeMode {
    Semantic,
    All,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphQueryArgs {
    pub query: String,
    pub path: Option<String>,
    pub depth: u8,
    pub direction: QueryDirection,
    pub edge_mode: QueryEdgeMode,
    pub include_noise: bool,
}

/// The maximum number of ambiguous candidate definitions to display to the user.
pub const AMBIGUOUS_DISPLAY_LIMIT: usize = 20;

/// The exact-signature store lookup ([`travsr_store::NODE_EXACT_LOOKUP_LIMIT`])
/// must be able to return at least one more candidate than we display, so the
/// CLI can tell "exactly the display limit" apart from "more than the display
/// limit" on the Tier-1 (exact-signature) path and fire the truncation notice
/// (#565 / RFC-002). Guarded at compile time so the two caps can never silently
/// re-coincide.
const _: () = assert!(travsr_store::NODE_EXACT_LOOKUP_LIMIT > AMBIGUOUS_DISPLAY_LIMIT);

/// One graph node, with everything the CLI renderers need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntry {
    pub id: u64,
    pub signature: String,
    pub kind: String,
    pub path: String,
    pub language: String,
    pub depth: u8,
    pub label: String,
    /// Estimated token cost (same formula as `travsr ask` / `get_context`).
    pub tokens: usize,
    pub line: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeEntry {
    pub src: u64,
    pub dst: u64,
    pub kind: String,
    pub provenance: String,
    /// Endpoint signatures resolved at build time so renderers never need
    /// store access (noise endpoints are not in `nodes` but still render).
    pub src_sig: String,
    pub dst_sig: String,
    /// Same flag as [`TreeStep::heuristic`], on the edge list.
    ///
    /// The tree view marked these edges and the `edges` array did not, so
    /// `--format dot` and `get_graph_json` presented a name-matched call as
    /// resolved. `serde(default)` keeps older daemon payloads deserializable.
    #[serde(default)]
    pub heuristic: bool,
}

/// Whether an edge was matched by bare callee name rather than resolved by
/// type: a `ref/call` that `resolve_unresolved_calls` wrote, not a compiler.
/// One predicate for both the tree and the edge list so the two views of a
/// single traversal cannot disagree.
pub(crate) fn is_heuristic_edge(kind: &str, provenance: &str) -> bool {
    kind == travsr_core::EdgeKind::RefCall.as_str() && provenance == "tree-sitter"
}

/// One BFS spanning-tree expansion step, in discovery order — drives the
/// `--format tree` renderer without store access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStep {
    pub parent: u64,
    pub edge_kind: String,
    pub child: u64,
    /// `true` when the stored edge points `child -> parent` (the child is a
    /// caller / container reached by walking an incoming edge), so the tree
    /// renderer can draw the true orientation (#564). `serde(default)` keeps
    /// payloads from daemons predating this field deserializable.
    #[serde(default)]
    pub incoming: bool,
    /// `true` for a `ref/call` edge whose provenance is `tree-sitter`: one
    /// `resolve_unresolved_calls` matched by bare callee name, not one a
    /// compiler resolved by type. Mirrors the condition `provenance_marker` in
    /// `tools.rs` uses for its `get_callers` sigil, so the tree view and
    /// `find_references` cannot describe the same edge differently.
    /// `serde(default)` keeps older daemon payloads deserializable.
    #[serde(default)]
    pub heuristic: bool,
}

/// Coverage / completeness metadata (#318 O5) — distinguishes "no callers"
/// from "cannot see callers".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coverage {
    pub language: String,
    pub semantic: bool,
    pub phase_b_commit: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphPayload {
    /// `None` when no symbol matched the query.
    pub seed: Option<NodeEntry>,
    /// BFS discovery order — index 0 is the seed. Budget truncation keeps a
    /// prefix of this order, so ancestors always survive before descendants.
    pub nodes: Vec<NodeEntry>,
    pub edges: Vec<EdgeEntry>,
    pub tree: Vec<TreeStep>,
    pub coverage: Option<Coverage>,
    pub last_commit: Option<String>,
    /// Ambiguous candidates, if the query resolves to multiple definitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<NodeEntry>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AskRow {
    pub kind: String,
    pub signature: String,
    pub path: String,
    #[serde(default)]
    pub line: Option<u32>,
    pub score: f32,
    /// RFC-022 §14: match-source bucket ("exact"/"semantic"/"relevant") for
    /// provenance grouping. Populated only under `TRAVSR_MATCH_SOURCE=1`; `None`
    /// (skipped in JSON) otherwise, so default output is byte-for-byte unchanged.
    /// Never reorders `rows` — grouping is a text-surface concern.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_source: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AskPayload {
    /// False when no symbol matched the query at all.
    pub matched: bool,
    /// True when a seed matched but PPR produced nothing.
    pub no_results: bool,
    pub rows: Vec<AskRow>,
    pub total_tokens: usize,
    /// True when KNN embed seeds drove PPR instead of FTS. Absent in old JSON → false.
    #[serde(default)]
    pub embed_used: bool,
    /// RFC-021 F9: honest confidence label ("exact"/"strong"/"weak"/"none") for
    /// cross-surface parity with `get_context`. Absent in old JSON → empty.
    #[serde(default)]
    pub confidence: String,
    /// #376 §4: the docs lane's rendered lines (`path § Heading Trail:lines`),
    /// already floored, capped, budget-clamped and sanitized.
    ///
    /// A separate field rather than `AskRow`s on purpose. `AskRow` carries a
    /// mandatory `score` that the CLI prints beside max-normalized PPR scores,
    /// and §4.1 forbids printing a doc score at all: doc and code scores are
    /// not commensurable, so a shared column would invite exactly the
    /// cross-section comparison the separate section exists to prevent. Docs
    /// therefore never enter `rows`, never enter the knapsack, and cannot
    /// displace a code result.
    ///
    /// `skip_serializing_if` keeps the JSON byte-identical to pre-#376 output
    /// whenever the lane is off or produced nothing (§4.2: absent, not empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub docs: Vec<String>,
    /// #478: honest FTS-only / degraded-embed note, reusing
    /// `build_context_signals` so `ask` and `get_context` stay at parity.
    /// Empty when embeddings are fully warm and contributing.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub degraded_note: String,
    /// UX-022: whether the doc index was available to this query path. True only
    /// when docs are enabled *and* the doc-space KNN hook is armed — which the
    /// daemon does at startup but the read-only cold path (a bare `travsr ask`
    /// with no daemon) cannot. The stderr note (UX-010) only fires on grounded
    /// cold-path results, so on a conceptual/abstained query the degradation was
    /// otherwise invisible in every format. This field exposes it structurally
    /// in all formats so a caller knows the answer may be missing a docs section.
    /// Always serialized (no `skip_serializing_if`) so its absence never has to
    /// be disambiguated from `false`.
    #[serde(default)]
    pub docs_available: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusPayload {
    pub nodes: u64,
    /// L11: number of FTS rows — should equal `nodes`. A mismatch indicates a
    /// partial write or corrupt FTS index; user should re-run `travsr init`.
    #[serde(default)]
    pub fts_nodes: u64,
    pub edges: u64,
    pub schema: u32,
    pub journal: String,
    pub last_commit: Option<String>,
    pub signature_format_version: u8,
    /// M7: commit at which Phase B last completed successfully. Compare with
    /// `last_commit` to know if Phase B is pending, complete, or running.
    pub phase_b_commit: Option<String>,
    /// H3: warnings from the last Phase B run (crashed/version_mismatch/needs_approval).
    /// Empty string = no warnings.
    pub phase_b_warnings: Option<String>,
    /// M1 / #738: rust-analyzer LSIF degradation for the last semantic pass.
    /// "sandbox_unavailable" = ra was skipped (OS sandbox missing); "all_refs_dropped"
    /// = ra ran but every reference failed resolution (0 edges landed). Empty = healthy.
    #[serde(default)]
    pub rust_lsif_degraded: Option<String>,
    /// RFC-021 P5: reranker state ("off"/"not installed"/"installed"/"ready"/
    /// "load failed"). Reflects the answering process — the warm daemon reports
    /// the loaded model; the cold CLI path reports on-disk presence. Absent in
    /// old JSON → empty.
    #[serde(default)]
    pub rerank: String,
    /// #583: a watcher reindex rewrote a file's Phase A nodes, dropping that
    /// file's `ref/call` edges, without moving HEAD. `last_commit` and
    /// `phase_b_commit` therefore still agree even though the semantic layer
    /// is degraded below the committed snapshot. Old daemons omit the field
    /// (serde default false), which reads as the pre-#583 behaviour.
    #[serde(default)]
    pub phase_b_dirty: bool,
    /// References the live overlay resolved mid-edit (`ref_resolution_state`
    /// state='resolved'). With `phase_b_dirty` set, a non-zero count is positive
    /// evidence the editor/lexical lane recovered the edited region, so the
    /// semantic surface is not degraded despite the reindex. Old daemons omit the
    /// field (serde default 0).
    #[serde(default)]
    pub live_refs_resolved: u64,
    /// References the live overlay recorded as still unresolved
    /// (`ref_resolution_state` state='pending'): the honest count of edges that
    /// are unknown until the next commit. Zero (with `live_refs_resolved` > 0)
    /// means the live lane settled everything it detected. Old daemons omit the
    /// field (serde default 0).
    #[serde(default)]
    pub live_refs_pending: u64,
    /// WS-2: comma-separated Dart package directories that were indexed without
    /// resolved dependencies (no `.dart_tool/package_config.json`), so their
    /// cross-package references are incomplete. Empty = resolved or no Dart.
    /// Old daemons omit the field (serde default None).
    #[serde(default)]
    pub dart_deps_unresolved: Option<String>,
    /// #825: the actual SCIP definitions behind the `scip_unification_misses`
    /// warning, one per line (`lang\tkind\tsymbol\tpath:line`), capped. Lets
    /// `travsr status` name the unreconciled symbols instead of only counting
    /// them. Empty/None = no misses. Old daemons omit it (serde default None).
    #[serde(default)]
    pub scip_unification_miss_list: Option<String>,
}

// ── status ────────────────────────────────────────────────────────────────────

pub fn status_query(store: &SqliteStore) -> anyhow::Result<StatusPayload> {
    let nodes = store.node_count()?;
    // L11: detect FTS/nodes skew — indicates a partial write or a bad migration.
    // fts_count is the number of rows in nodes_fts (virtual FTS table).
    let fts_count = store.fts_node_count().unwrap_or(nodes);
    // Live-overlay tallies, so a phase_b_dirty edit the editor lane already
    // recovered reads as live-fresh rather than a blanket "stale". A read error
    // degrades to zero, which keeps the conservative signal.
    let resolved_refs = store.resolved_ref_count().unwrap_or(0);
    let pending_refs = store.pending_ref_count().unwrap_or(0);
    Ok(StatusPayload {
        nodes,
        fts_nodes: fts_count,
        edges: store.edge_count()?,
        schema: store.schema_version()?,
        journal: store.journal_mode()?,
        last_commit: store.get_meta("last_commit")?,
        signature_format_version: store.get_signature_format_version()?,
        phase_b_commit: store.get_meta("phase_b_commit")?,
        phase_b_warnings: store.get_meta("phase_b_warnings")?,
        rust_lsif_degraded: store.get_meta("rust_lsif_degraded")?,
        rerank: crate::rerank::rerank_status().to_string(),
        phase_b_dirty: store.get_meta("phase_b_dirty")?.as_deref() == Some("1"),
        live_refs_resolved: resolved_refs,
        live_refs_pending: pending_refs,
        dart_deps_unresolved: store.get_meta("dart_deps_unresolved")?,
        scip_unification_miss_list: store.get_meta("scip_unification_miss_list")?,
    })
}

// ── ask ───────────────────────────────────────────────────────────────────────

/// Maximum byte length of one rendered docs-lane line on the `ask` surface.
///
/// Applied per entry and independently of the token budget — plan §6 mitigation
/// M3, so a single adversarial doc cannot consume the response even if the
/// budget carve would have allowed it.
const DOC_LINE_MAX_BYTES: usize = 512;

/// Run the #376 docs lane for `ask` and sanitize its rendered lines.
///
/// **Why this sanitizes when the rest of `ask` does not.** `sanitize_for_mcp`
/// is applied throughout `tools.rs` (the MCP surface) and nowhere in this
/// module: every other value `ask` returns is graph-derived — symbol names,
/// kinds, paths — whereas a doc chunk originates in author-controlled prose,
/// which is threat row T11. So the docs block is sanitized here specifically,
/// rather than retrofitting the whole `ask` path, which would change existing
/// output bytes for results that are not part of this feature.
///
/// [`crate::sanitize::sanitize_mcp_body_with_limit`] strips C0/C1/DEL controls
/// and Unicode bidi overrides, entity-escapes `<`/`>`, and hard-caps the byte
/// length. No `<travsr-data>` envelope: this is terminal output, and the CLI
/// marks the block with its own untrusted-prose header instead.
///
/// **Scoped honestly:** this is defense in depth on this surface, not the
/// load-bearing control. The rendered line is `path § Heading Trail:lines` with
/// no prose body, and the heading trail is derived from the node's anchor,
/// which `travsr_analysis::markdown::slugify_segment` has already reduced to
/// alphanumerics and `-`. Controls, bidi marks and angle brackets cannot reach
/// this string through the anchor in the first place. It is applied anyway
/// because that invariant lives in another crate and nothing here enforces it,
/// and because the file path — the one component this argument does not cover —
/// is equally author-controlled.
///
/// One thing escaping deliberately does *not* buy on this surface: the CLI
/// delimits sections with `──` box-drawing runs, not tags, so tag escaping does
/// not make the CLI's section header unforgeable the way it makes the MCP
/// envelope unforgeable. Slugification is what prevents an anchor from carrying
/// such a run; a crafted *path* still could, which is a pre-existing property of
/// every row `ask` already prints in its Path column, not something docs add.
fn docs_section(store: &SqliteStore, query: &str, token_budget: usize) -> (Vec<String>, usize) {
    let (entries, doc_tokens) = crate::tools::build_docs_section(store, query, token_budget);
    let lines = entries
        .into_iter()
        .map(|(_, _, line)| {
            crate::sanitize::sanitize_mcp_body_with_limit(&line, DOC_LINE_MAX_BYTES)
        })
        .collect();
    (lines, doc_tokens)
}

/// Run an ask query with unified FTS+KNN seed selection.
///
/// Both FTS and KNN always run when embed is available; results are merged with
/// `max(fts_weight, knn_score)` deduplication so the two sources reinforce shared
/// nodes rather than competing. Seeds structurally adjacent to a higher-scored seed
/// are then dropped (`dedup_adjacent_seeds`) to avoid diluting PPR teleportation mass.
///
/// KNN is subject to a `KNN_BUDGET_MS` circuit-breaker: if the sidecar takes too long
/// its results are discarded and only FTS seeds steer PPR.
pub fn ask_query(
    store: &SqliteStore,
    query: &str,
    knn_fn: Option<KnnFn<'_>>,
) -> anyhow::Result<AskPayload> {
    // Local single-repo mode: process isolation is the auth boundary
    // (RFC-006 §3.1), so every corpus in this store is already the caller's.
    ask_query_with_filter(store, query, knn_fn, &OpenFilter)
}

/// `ask_query` with an explicit traversal filter.
///
/// #413: PPR powers `get_context` and `ask`, and it used to expand its
/// subgraph with no RBAC gate at all — so once RBAC is switched on for the
/// cloud path, ranking would cross corpus boundaries freely and return nodes
/// the caller may not see. PCST was already gated; this was the one primary
/// retrieval path where the wire was missing.
///
/// Pass `&OpenFilter` for unauthenticated local mode; pass the session's
/// filter in authenticated mode. Do not hardcode `&OpenFilter` here once
/// `SessionStore` reaches this call path.
pub fn ask_query_with_filter(
    store: &SqliteStore,
    query: &str,
    knn_fn: Option<KnnFn<'_>>,
    filter: &dyn EdgeFilter,
) -> anyhow::Result<AskPayload> {
    // Strip a leading `:` so VS Code graph-panel queries pass through cleanly.
    let query = query.strip_prefix(':').unwrap_or(query).trim();
    // CO-C1: normalize once at the top so both FTS and embed see the same
    // cleaned query — prevents punctuation divergence on the cold/CLI path.
    let normalized = travsr_store::fts_tokenize::normalize_nl_query(query);
    let query = normalized.as_str();

    // ── Tier-0 seed selection (per-token anchor + BM25 FTS + optional KNN) ──
    let has_embed = knn_fn.is_some();
    let embed_warming = has_embed && !store.embed_ready();
    let knn_pairs;
    let knn_oracle;
    let embed_contributed;
    if let Some(knn) = knn_fn {
        let (knn_scored, _n_eligible, knn_elapsed_ms, oracle) =
            crate::tools::embed_path_seeds(store, query, knn, &OpenFilter);
        let budget_ms = crate::tools::knn_budget_ms();
        if knn_elapsed_ms > budget_ms {
            tracing::warn!(
                knn_elapsed_ms,
                threshold_ms = budget_ms,
                "ask_query knn exceeded circuit-breaker, falling back to FTS seeds"
            );
            knn_pairs = vec![];
            knn_oracle = std::collections::HashMap::new();
            embed_contributed = false;
        } else {
            embed_contributed = !knn_scored.is_empty();
            knn_pairs = knn_scored;
            knn_oracle = oracle;
        }
    } else {
        knn_pairs = vec![];
        knn_oracle = std::collections::HashMap::new();
        embed_contributed = false;
    }

    // #478: honest degraded note — mirrors get_context's build_context_signals
    // rather than a second formatter. knn_degraded is true only when embeddings
    // were configured but did not end up contributing (empty KNN or circuit-broken).
    let knn_degraded = has_embed && !embed_contributed;
    let degraded_note = crate::tools::build_context_signals(
        store,
        has_embed,
        embed_warming,
        knn_degraded,
        None,
        None,
    );

    // RFC-019: direct-cosine oracle hook (None when embeddings off → FTS-only path).
    let score_fn = store.embed_score_fn();
    let score_ref = score_fn
        .as_ref()
        .map(|f| f as &dyn Fn(&str, &[travsr_core::NodeId]) -> Vec<(travsr_core::NodeId, f32)>);
    let seed_set =
        crate::seed::build_seed_set(store, query, &OpenFilter, knn_pairs, &knn_oracle, score_ref);
    // F9: honest confidence label, surfaced on every return for `ask`/CLI parity
    // with get_context (which already prints it in its retrieval header).
    let confidence = seed_set.confidence.label().to_string();

    // #376 §4: docs lane. Placed here, mirroring `get_context_body`, so it runs
    // once for every return path below — including the abstain return, which is
    // where a cited doc section is worth the most (§8.5 measured 15/20 k8s
    // rationale queries as hard abstentions). It runs *after* the code lane's
    // KNN so the code lane's circuit-breaker timing is unchanged, and it reuses
    // the same normalized `query` string, which keeps the sidecar's exact-text
    // query-embedding memo hitting (§4.4: one inference, two searches).
    //
    // Docs are computed independently of `Confidence` and never feed it: they
    // cannot convert an abstention into a confident answer (§4.3).
    let (docs, doc_tokens) = docs_section(store, query, DEFAULT_TOKEN_BUDGET);

    // UX-022: a docs section can only appear when docs are enabled AND the
    // doc-space KNN hook is armed. The cold path arms the code embed hook but
    // never the doc hook, so this is false there regardless of the config flag —
    // exactly the "answer is partial" signal callers need in every format.
    let docs_available = crate::seed::docs_enabled(store) && store.embed_doc_knn_fn().is_some();

    // Abstain when confidence is None — prevents "confident salad" on no-match queries (R1).
    if seed_set.confidence == crate::seed::Confidence::None {
        return Ok(AskPayload {
            matched: false,
            no_results: false,
            rows: Vec::new(),
            total_tokens: doc_tokens,
            embed_used: embed_contributed,
            confidence,
            docs,
            degraded_note: degraded_note.clone(),
            docs_available,
        });
    }

    // F9: seeds' absolute rerank scores + the primary-seed set (pre-enrichment),
    // so display can show the cross-encoder judgment and cap expanded neighbours.
    let seed_rerank = seed_set.rerank_scores();
    let raw_seeds = seed_set.ppr_seeds();
    let primary_seed_ids: HashSet<NodeId> = raw_seeds.iter().map(|&(id, _)| id).collect();

    // Caller enrichment: depth-1 then depth-2 production callers at 0.35× weight per hop.
    let raw_seeds = crate::tools::enrich_seeds_with_callers(store, raw_seeds, 15); // depth-1
    let raw_seeds = crate::tools::enrich_seeds_with_callers(store, raw_seeds, 10); // depth-2
                                                                                   // Drop seeds whose 1-hop PPR expansion would overlap a higher-scored accepted seed.
    let seeds = crate::tools::dedup_adjacent_seeds(store, raw_seeds);

    if seeds.is_empty() {
        return Ok(AskPayload {
            matched: false,
            no_results: false,
            rows: Vec::new(),
            total_tokens: doc_tokens,
            embed_used: embed_contributed,
            confidence: confidence.clone(),
            docs: docs.clone(),
            degraded_note: degraded_note.clone(),
            docs_available,
        });
    }

    // ── PPR with confidence-weighted personalisation ───────────────────────────
    let ppr_scores = ppr_weighted(store, &seeds, context_candidates(), filter)?;
    if ppr_scores.is_empty() {
        return Ok(AskPayload {
            matched: true,
            no_results: true,
            rows: Vec::new(),
            total_tokens: doc_tokens,
            embed_used: embed_contributed,
            confidence: confidence.clone(),
            docs: docs.clone(),
            degraded_note: degraded_note.clone(),
            docs_available,
        });
    }

    let node_ids: Vec<_> = ppr_scores.iter().map(|(id, _)| *id).collect();
    let raw_score_map: HashMap<_, f32> = ppr_scores.into_iter().collect();
    let nodes = store.get_nodes(&node_ids)?;

    // Filter structural package nodes — never valid context regardless of language.
    // Go `go-pkg` nodes are scip-go package identifiers (often 1000+ in-edges).
    // Go `module` nodes are package declarations; Rust `module` nodes (`mod foo`)
    // are real code entities and must NOT be filtered.
    let nodes: Vec<_> = nodes
        .into_iter()
        .filter(|n| !(n.kind == "go-pkg" || (n.kind == "module" && n.vname.language == "go")))
        .collect();

    // Degree damping: penalise hub nodes proportional to their in-degree.
    // adjusted = ppr_score × 1/(1 + ln(max(1, in_degree)))
    // A node with 1000 in-edges is damped to ~14% of its raw PPR score;
    // a node with 5 in-edges retains ~38% — hub nodes are suppressed, not zeroed.
    let filtered_ids: Vec<_> = nodes.iter().map(|n| n.id).collect();
    let in_degrees = store.in_degrees(&filtered_ids)?;
    let mut score_map: HashMap<NodeId, f32> = raw_score_map
        .into_iter()
        .map(|(id, s)| {
            let degree = in_degrees.get(&id).copied().unwrap_or(0);
            let damped = s * (1.0 / (1.0 + (degree as f32).max(1.0).ln()));
            (id, damped)
        })
        .collect();

    // F9: max-normalize damped PPR to [0,1] so `ask` and `get_context` display
    // scores on the same scale (cross-surface parity) and expanded rows are
    // comparable to the seeds' absolute rerank scores. Monotone → knapsack
    // ordering is unaffected.
    let max_score = score_map.values().copied().fold(0.0_f32, f32::max);
    if max_score > 0.0 {
        for v in score_map.values_mut() {
            *v /= max_score;
        }
    }

    let items: Vec<_> = nodes
        .into_iter()
        .filter_map(|n| score_map.get(&n.id).map(|&s| (n, s)))
        .collect();

    // #376 §4.3: the docs section's *measured* cost is subtracted from the code
    // lane's budget, never reserved up front — with the lane off (or no doc hit)
    // `doc_tokens` is 0 and the knapsack call is byte-identical to pre-#376.
    // The reported total then covers both lanes, so the budget stays a hard
    // ceiling on the whole response rather than on the code lane alone.
    let selected = knapsack(items, DEFAULT_TOKEN_BUDGET.saturating_sub(doc_tokens));
    let total_tokens: usize = selected.iter().map(token_cost).sum::<usize>() + doc_tokens;
    // F9: expanded neighbours cap at the best seed's rerank so none outranks its seed.
    let expanded_cap = seed_rerank.values().copied().reduce(f32::max);
    // RFC-022 §14: emit the match-source tag only under the flag. `rows` order is
    // never changed here — grouping is a text-surface concern (CLI table), so the
    // JSON `rows` array stays in knapsack order and hit@k is unaffected.
    let emit_match_source = crate::seed::match_source_grouping_enabled();
    // Nodes whose strongest seed source is an exact literal-name / FTS match →
    // the "exact" match-source bucket (RFC-022 §14). Everything else that is a
    // primary seed is "semantic"; non-seeds are "relevant". Uses the same
    // node→strongest-source classifier as get_context (`strongest_seed_sources`)
    // so a node buckets identically on both surfaces.
    let strongest_source: HashMap<NodeId, crate::seed::SeedSource> = if emit_match_source {
        crate::seed::strongest_seed_sources(&seed_set.seeds)
    } else {
        HashMap::new()
    };
    // #479: index-time test nodes bucket as the "tests" match source regardless
    // of seed provenance, so the CLI groups them into the capped tests section
    // instead of letting a `#[test]` fn lead the exact/semantic group. Read by id
    // (the store column) since `Node.test_role` defaults to `None` on read paths.
    let test_node_ids: std::collections::HashSet<NodeId> = if emit_match_source {
        selected
            .iter()
            .filter(|n| matches!(store.test_role(n.id), Ok(Some(r)) if r.is_test()))
            .map(|n| n.id)
            .collect()
    } else {
        std::collections::HashSet::new()
    };
    let rows = selected
        .into_iter()
        .map(|n| {
            let is_primary = primary_seed_ids.contains(&n.id);
            let signature = display_label(&n).into_owned();
            AskRow {
                score: crate::seed::display_score(
                    n.id,
                    score_map.get(&n.id).copied().unwrap_or(0.0),
                    &seed_rerank,
                    is_primary,
                    expanded_cap,
                    seed_set.confidence,
                ),
                match_source: emit_match_source.then(|| {
                    if test_node_ids.contains(&n.id) {
                        return crate::seed::MatchSource::Tests.label().to_string();
                    }
                    let is_exact = matches!(
                        strongest_source.get(&n.id),
                        Some(crate::seed::SeedSource::Exact)
                    );
                    crate::seed::match_source(is_primary, is_exact)
                        .label()
                        .to_string()
                }),
                kind: n.kind,
                signature,
                path: n.vname.path,
                line: n.line,
            }
        })
        .collect();

    Ok(AskPayload {
        matched: true,
        no_results: false,
        rows,
        total_tokens,
        embed_used: embed_contributed,
        confidence,
        docs,
        degraded_note,
        docs_available,
    })
}

// ── explain (#478 RFC-023 §6.1, WS-8) ────────────────────────────────────────

/// Diagnostic seed-building trace for one query/symbol pair: per-token IDF,
/// per-leg match status, every threshold, and final disposition (live vs an
/// FTS-only counterfactual). Local CLI diagnostic only (RFC-023 §4) — never
/// routed through the daemon control socket like `ask`/`graph`/`status`, so
/// it always uses the direct (cold) store-open path.
pub fn explain_query(
    store: &SqliteStore,
    query: &str,
    symbol: &str,
    knn_fn: Option<KnnFn<'_>>,
) -> crate::seed::ExplainReport {
    let query = query.strip_prefix(':').unwrap_or(query).trim();
    let normalized = travsr_store::fts_tokenize::normalize_nl_query(query);
    let query = normalized.as_str();

    let (knn_pairs, knn_oracle) = match knn_fn {
        Some(knn) => {
            let (knn_scored, _n_eligible, knn_elapsed_ms, oracle) =
                crate::tools::embed_path_seeds(store, query, knn, &OpenFilter);
            let budget_ms = crate::tools::knn_budget_ms();
            if knn_elapsed_ms > budget_ms {
                (vec![], std::collections::HashMap::new())
            } else {
                (knn_scored, oracle)
            }
        }
        None => (vec![], std::collections::HashMap::new()),
    };

    let score_fn = store.embed_score_fn();
    let score_ref = score_fn
        .as_ref()
        .map(|f| f as &dyn Fn(&str, &[travsr_core::NodeId]) -> Vec<(travsr_core::NodeId, f32)>);
    crate::seed::explain_seed_set(
        store,
        query,
        symbol,
        &OpenFilter,
        knn_pairs,
        &knn_oracle,
        score_ref,
    )
}

// ── graph ─────────────────────────────────────────────────────────────────────

fn node_entry(node: &travsr_core::Node, depth: u8) -> NodeEntry {
    NodeEntry {
        id: node.id.0,
        signature: node.vname.signature.clone(),
        kind: node.kind.clone(),
        path: node.vname.path.clone(),
        language: node.vname.language.clone(),
        depth,
        label: display_label(node).to_string(),
        tokens: token_cost(node),
        line: node.line,
    }
}

fn is_semantic_edge(kind: &travsr_core::EdgeKind) -> bool {
    use travsr_core::EdgeKind;
    matches!(
        kind,
        EdgeKind::RefCall
            | EdgeKind::FFICall
            | EdgeKind::Overrides
            | EdgeKind::IsImplementation
            | EdgeKind::RefImports
            | EdgeKind::ResolvesTo
    )
}

/// Containment edges answer "where does this live", not "who calls this".
/// In caller traversal they are shown as orientation but never expanded —
/// expanding one pulls in every sibling definition of the containing file
/// (#517).
fn is_containment_edge(kind: &travsr_core::EdgeKind) -> bool {
    matches!(kind, travsr_core::EdgeKind::DefinesBinding)
}

/// Outgoing/incoming expansion for one node, mirroring `travsr graph`'s edge
/// semantics (RFC-014 file-node caller splice, semantic-preferred mode with
/// per-node structural fallback for display). Whether to surface the
/// language-level "semantic unavailable" note is decided once in `graph_query`
/// from coverage — NOT here — so a single caller-less leaf cannot trip it.
///
/// Returns `(edge_kind, next_id, expand, incoming)`. `expand` is `false` for a
/// containment edge reached in `Callers`/`Both` direction (#517 DD-1): the
/// node is still recorded and displayed, but the traversal does not walk
/// further from it, so a file's other definitions never enter the BFS queue.
/// The read-side provenance of an edge that came out of a store reader, with a
/// `tree-sitter` fallback for a constructed edge that never carried one.
fn prov_of(e: &travsr_core::Edge) -> String {
    e.provenance
        .clone()
        .unwrap_or_else(|| "tree-sitter".to_string())
}

/// One expansion step out of [`next_edges`]:
/// `(edge_kind, next_id, expand, incoming, provenance)`. The 5th element is the
/// edge's true `edges.provenance` (DEBT-75).
pub type NextEdge = (travsr_core::EdgeKind, NodeId, bool, bool, String);

pub fn next_edges(
    store: &SqliteStore,
    node_id: NodeId,
    direction: QueryDirection,
    edge_mode: QueryEdgeMode,
    is_seed: bool,
) -> anyhow::Result<Vec<NextEdge>> {
    // DEBT-75: the 5th element is the edge's true `edges.provenance`, carried
    // through from the store readers so callers no longer have to assume
    // "tree-sitter". `unwrap_or` only fires on a constructed (never-read) edge,
    // which cannot reach here.
    let mut out = Vec::new();
    if matches!(direction, QueryDirection::Deps | QueryDirection::Both) {
        for e in store.iter_edges_from(node_id)? {
            let prov = e.provenance.unwrap_or_else(|| "tree-sitter".to_string());
            out.push((e.kind, e.dst, true, false, prov));
        }
    }
    if matches!(direction, QueryDirection::Callers | QueryDirection::Both) {
        let mut incoming = store.iter_edges_to(node_id)?;
        // RFC-014 #317: a file's callers are the callers of the definitions it
        // contains — splice them in so function-level callers surface when a
        // query resolves to a file node. Edges sourced from the file itself
        // (defines/binding) are skipped to avoid self-loops.
        //
        // #517 DD-2: gated on `is_seed` — this splice is only correct when the
        // *user's query* resolved to a file. A file node reached incidentally
        // by walking a containment edge up from a symbol must not re-trigger
        // it, or every sibling definition in that file floods the traversal.
        if is_seed {
            if let Some(node) = store.get_node(node_id)? {
                if node.kind == "file" {
                    // Cap the splice at 200 definitions: pathological generated
                    // files can hold thousands, each costing an iter_edges_to
                    // round-trip. Full fidelity remains available via per-symbol
                    // queries.
                    for def_id in store
                        .definition_node_ids_in_file(&node.vname.corpus, &node.vname.path)?
                        .into_iter()
                        .take(200)
                    {
                        incoming.extend(
                            store
                                .iter_edges_to(def_id)?
                                .into_iter()
                                .filter(|e| e.src != node_id),
                        );
                    }
                }
            }
        }
        if matches!(edge_mode, QueryEdgeMode::Semantic) {
            let has_semantic = incoming.iter().any(|e| is_semantic_edge(&e.kind));
            if has_semantic {
                for e in &incoming {
                    let s = &e.kind;
                    if is_semantic_edge(s) || matches!(s, travsr_core::EdgeKind::DefinesBinding) {
                        out.push((*s, e.src, !is_containment_edge(s), true, prov_of(e)));
                    }
                }
            } else {
                // This node has no semantic callers — show its structural
                // incoming edges so the caller view is never empty. Whether the
                // *language* lacks semantic data (and thus warrants the note) is
                // judged from coverage in graph_query, not from this one node.
                for e in &incoming {
                    let s = &e.kind;
                    out.push((*s, e.src, !is_containment_edge(s), true, prov_of(e)));
                }
            }
        } else {
            for e in &incoming {
                let s = &e.kind;
                out.push((*s, e.src, !is_containment_edge(s), true, prov_of(e)));
            }
        }
    }
    // Multiple call sites (and the file-node definition splice) can yield the
    // same (kind, src, orientation) triple — collapse them for display.
    let mut seen = HashSet::new();
    out.retain(|(kind, id, _, incoming, _)| seen.insert((*kind, *id, *incoming)));
    // #517 DD-1: non-containment edges (the answer) precede containment edges
    // (orientation) from the same parent. Stable sort preserves DB order
    // within each group, so output stays deterministic.
    out.sort_by_key(|(kind, _, _, _, _)| is_containment_edge(kind));
    Ok(out)
}

fn coverage_for(store: &SqliteStore, language: &str) -> Coverage {
    Coverage {
        language: language.to_string(),
        semantic: store.has_refcall_edges_for_language(language),
        phase_b_commit: store.get_meta("phase_b_commit").ok().flatten(),
    }
}

/// BFS subgraph around the best seed for `args.query`.
///
/// `payload.seed == None` means no symbol matched. Nodes are emitted in BFS
/// discovery order; `tree` holds the spanning-tree expansion steps.
pub fn graph_query(store: &SqliteStore, args: &GraphQueryArgs) -> anyhow::Result<GraphPayload> {
    let mut candidates: Option<Vec<NodeEntry>> = None;
    let seed =
        match crate::tools::resolve_reference_targets(store, &args.query, args.path.as_deref()) {
            crate::tools::RefTarget::Unique(n) => Some(n),
            // A selector family is listed like an ambiguity here rather than
            // merged: a graph has one root, and the arities of a selector are
            // distinct nodes with distinct neighbourhoods. Naming the full
            // selector picks one. (`find_references` unions them instead, since
            // a reference list has no root to conflict over.)
            crate::tools::RefTarget::Ambiguous(list) | crate::tools::RefTarget::Family(list) => {
                let candidates_entries: Vec<NodeEntry> =
                    list.iter().map(|n| node_entry(n, 0)).collect();
                candidates = Some(candidates_entries);
                None
            }
            crate::tools::RefTarget::None => {
                // `resolve_reference_targets` filters `kind == "file"`, so a bare
                // file-name query (`service.ts`) never reaches the symbol ladder.
                // Detect same-basename file ambiguity here so `travsr graph <file>`
                // disambiguates consistently with symbols (#565 / RFC-002), while
                // still honouring an explicit `--path` pin and preserving the
                // legacy fuzzy-recall fallback for non-file queries like `Payment`.
                let matches = store.search_nodes_by_name(&args.query)?;
                let mut file_hits: Vec<travsr_core::Node> = matches
                    .iter()
                    .filter(|n| {
                        n.kind == "file"
                            && (n.vname.path == args.query
                                || n.vname.path.ends_with(&format!("/{}", args.query)))
                    })
                    .cloned()
                    .collect();
                if let Some(p) = args.path.as_deref() {
                    // #647: same widened path hint as the symbol ladder — a
                    // directory or fragment must scope a file-name query too.
                    file_hits.retain(|n| crate::tools::path_hint_matches(&n.vname.path, p));
                }
                file_hits.sort_by_key(|n| n.id.0);
                file_hits.dedup_by_key(|n| n.id);

                match file_hits.len() {
                    1 => file_hits.pop(),
                    n if n >= 2 => {
                        candidates = Some(file_hits.iter().map(|n| node_entry(n, 0)).collect());
                        None
                    }
                    // No file-name match: keep legacy fuzzy recall, but only when
                    // no `--path` was given — an explicit path that matched nothing
                    // is a precise miss, not an invitation to guess a first hit.
                    _ => {
                        if args.path.is_none() {
                            matches
                                .iter()
                                .find(|n| n.kind == "file")
                                .or_else(|| matches.first())
                                .cloned()
                        } else {
                            None
                        }
                    }
                }
            }
        };

    let Some(seed) = seed else {
        return Ok(GraphPayload {
            seed: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            tree: Vec::new(),
            coverage: None,
            last_commit: store.get_meta("last_commit").ok().flatten(),
            candidates,
        });
    };

    let mut nodes: Vec<NodeEntry> = Vec::new();
    let mut node_index: HashSet<NodeId> = HashSet::new();
    let mut edges_raw: Vec<(NodeId, NodeId, String, String)> = Vec::new();
    let mut tree: Vec<TreeStep> = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<(NodeId, u8, bool)> = VecDeque::new();

    visited.insert(seed.id);
    queue.push_back((seed.id, 0, true));

    while let Some((current_id, depth, expand)) = queue.pop_front() {
        if let Some(node) = store.get_node(current_id)? {
            if node_index.insert(current_id) {
                nodes.push(node_entry(&node, depth));
            }
        }

        // #517 DD-1: the depth guard and the terminal-node guard (`!expand`)
        // sit after the node has already been pushed above, so a containment
        // leaf is still emitted and rendered at its true depth — it is simply
        // not walked any further.
        if depth >= args.depth || !expand {
            continue;
        }

        for (edge_kind, next_id, child_expand, edge_incoming, edge_provenance) in next_edges(
            store,
            current_id,
            args.direction,
            args.edge_mode,
            depth == 0,
        )? {
            // #564: orient from the edge itself, not the direction flag — in
            // `Both` mode a single expansion mixes incoming and outgoing edges.
            let (src, dst) = if edge_incoming {
                (next_id, current_id)
            } else {
                (current_id, next_id)
            };
            let heuristic = is_heuristic_edge(edge_kind.as_str(), &edge_provenance);
            edges_raw.push((src, dst, edge_kind.as_str().to_string(), edge_provenance));

            if !visited.contains(&next_id) {
                if let Some(next_node) = store.get_node(next_id)? {
                    if !args.include_noise && is_noise_node(&next_node) {
                        visited.insert(next_id);
                        continue;
                    }
                }
                visited.insert(next_id);
                tree.push(TreeStep {
                    parent: current_id.0,
                    edge_kind: edge_kind.as_str().to_string(),
                    child: next_id.0,
                    incoming: edge_incoming,
                    heuristic,
                });
                queue.push_back((next_id, depth + 1, child_expand));
            }
        }
    }

    let edges = resolve_edge_sigs(store, &nodes, edges_raw)?;
    let coverage = coverage_for(store, &seed.vname.language);
    let seed_entry = nodes.first().cloned();

    Ok(GraphPayload {
        seed: seed_entry,
        nodes,
        edges,
        tree,
        coverage: Some(coverage),
        last_commit: store.get_meta("last_commit")?,
        candidates,
    })
}

/// Whole-graph payload for `travsr graph --all`. No traversal: every node at
/// depth 0, edges verbatim from the store (with true provenance).
/// L7: threshold above which `graph --all` truncates and warns the user.
pub const GRAPH_ALL_NODE_LIMIT: usize = 50_000;

pub fn graph_all_payload(store: &SqliteStore) -> anyhow::Result<GraphPayload> {
    let mut all = store.all_nodes()?;
    // L7: cap at GRAPH_ALL_NODE_LIMIT to prevent OOM on very large repos.
    if all.len() > GRAPH_ALL_NODE_LIMIT {
        eprintln!(
            "warning: graph has {} nodes, showing first {} only. \
             Use `travsr graph <symbol>` for focused traversal.",
            all.len(),
            GRAPH_ALL_NODE_LIMIT
        );
        all.truncate(GRAPH_ALL_NODE_LIMIT);
    }
    let nodes: Vec<NodeEntry> = all.iter().map(|n| node_entry(n, 0)).collect();
    let edges_raw = store.all_edges()?;
    let edges = resolve_edge_sigs(store, &nodes, edges_raw)?;
    Ok(GraphPayload {
        seed: None,
        nodes,
        edges,
        tree: Vec::new(),
        coverage: None,
        last_commit: store.get_meta("last_commit")?,
        candidates: None,
    })
}

fn resolve_edge_sigs(
    store: &SqliteStore,
    nodes: &[NodeEntry],
    edges_raw: Vec<(NodeId, NodeId, String, String)>,
) -> anyhow::Result<Vec<EdgeEntry>> {
    let mut sig_lookup: HashMap<u64, String> =
        nodes.iter().map(|n| (n.id, n.signature.clone())).collect();
    let mut edges = Vec::with_capacity(edges_raw.len());
    for (src, dst, kind, provenance) in edges_raw {
        for nid in [src, dst] {
            if let std::collections::hash_map::Entry::Vacant(e) = sig_lookup.entry(nid.0) {
                if let Some(node) = store.get_node(nid)? {
                    e.insert(node.vname.signature.clone());
                }
            }
        }
        edges.push(EdgeEntry {
            src: src.0,
            dst: dst.0,
            heuristic: is_heuristic_edge(&kind, &provenance),
            kind,
            provenance,
            src_sig: sig_lookup.get(&src.0).cloned().unwrap_or_default(),
            dst_sig: sig_lookup.get(&dst.0).cloned().unwrap_or_default(),
        });
    }
    Ok(edges)
}

// ── token budget (#318 O6) ────────────────────────────────────────────────────

/// Truncate `payload` to `budget` tokens. Returns the number of nodes cut.
///
/// Selection is a prefix of BFS discovery order — the seed always survives
/// (even if it alone exceeds the budget) and a node's parent is always kept
/// before the node, so tree rendering stays connected. `budget == 0` means
/// unlimited.
pub fn apply_token_budget(payload: &mut GraphPayload, budget: usize) -> usize {
    if budget == 0 || payload.nodes.is_empty() {
        return 0;
    }
    let mut cum = 0usize;
    let mut keep = 0usize;
    for (i, n) in payload.nodes.iter().enumerate() {
        if i > 0 && cum + n.tokens > budget {
            break;
        }
        cum += n.tokens;
        keep = i + 1;
    }
    let truncated = payload.nodes.len() - keep;
    if truncated == 0 {
        return 0;
    }
    let cut_ids: HashSet<u64> = payload.nodes[keep..].iter().map(|n| n.id).collect();
    payload.nodes.truncate(keep);
    payload
        .edges
        .retain(|e| !cut_ids.contains(&e.src) && !cut_ids.contains(&e.dst));
    payload
        .tree
        .retain(|t| !cut_ids.contains(&t.parent) && !cut_ids.contains(&t.child));
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use travsr_core::{Edge, EdgeKind, Node, VName};

    fn node(sig: &str, kind: &str, path: &str) -> Node {
        let vname = VName {
            signature: sig.to_string(),
            corpus: "test".to_string(),
            root: String::new(),
            path: path.to_string(),
            language: "typescript".to_string(),
        };
        let id = vname.id();
        Node {
            id,
            vname,
            kind: kind.to_string(),
            package: String::new(),
            line: None,
            end_line: None,
            test_role: travsr_core::TestRole::None,
        }
    }

    fn seeded_store() -> (SqliteStore, Node, Node, Node) {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let file = node("file", "file", "src/service.ts");
        let class = node("class:PaymentService", "class", "src/service.ts");
        let caller = node("fn:processPayment", "function", "src/controller.ts");
        for n in [&file, &class, &caller] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(file.id, class.id, EdgeKind::DefinesBinding))
            .unwrap();
        store
            .put_edge(&Edge::new(caller.id, class.id, EdgeKind::RefCall))
            .unwrap();
        (store, file, class, caller)
    }

    #[test]
    fn status_query_reports_counts_and_schema() {
        let (store, ..) = seeded_store();
        let s = status_query(&store).unwrap();
        assert_eq!(s.nodes, 3);
        assert_eq!(s.edges, 2);
        assert!(s.schema > 0);
    }

    #[test]
    fn graph_query_finds_callers_with_tree_steps() {
        let (store, _, class, caller) = seeded_store();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        let seed = payload.seed.as_ref().expect("seed must match");
        assert_eq!(seed.signature, "class:PaymentService");
        assert!(payload.nodes.iter().any(|n| n.id == caller.id.0));
        assert!(payload
            .tree
            .iter()
            .any(|t| t.parent == class.id.0 && t.child == caller.id.0));
        let cov = payload.coverage.expect("coverage present");
        assert_eq!(cov.language, "typescript");
        assert!(cov.semantic, "ref/call edge exists for typescript");
    }

    #[test]
    fn graph_query_no_match_returns_empty_payload() {
        let (store, ..) = seeded_store();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "DoesNotExist".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Both,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(payload.seed.is_none());
        assert!(payload.nodes.is_empty());
    }

    #[test]
    fn token_budget_keeps_bfs_prefix_and_reports_cut() {
        let (store, ..) = seeded_store();
        let mut payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Both,
                edge_mode: QueryEdgeMode::All,
                include_noise: true,
            },
        )
        .unwrap();
        let total = payload.nodes.len();
        assert!(total >= 2, "need at least seed + one neighbour");
        let seed_tokens = payload.nodes[0].tokens;
        // Budget exactly the seed's cost: everything after it is cut.
        let cut = apply_token_budget(&mut payload, seed_tokens);
        assert_eq!(payload.nodes.len(), 1);
        assert_eq!(cut, total - 1);
        // All surviving edges/tree steps reference surviving nodes only.
        let kept: HashSet<u64> = payload.nodes.iter().map(|n| n.id).collect();
        assert!(payload.tree.iter().all(|t| kept.contains(&t.parent)));
    }

    #[test]
    fn zero_budget_means_unlimited() {
        let (store, ..) = seeded_store();
        let mut payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Both,
                edge_mode: QueryEdgeMode::All,
                include_noise: true,
            },
        )
        .unwrap();
        let before = payload.nodes.len();
        assert_eq!(apply_token_budget(&mut payload, 0), 0);
        assert_eq!(payload.nodes.len(), before);
    }

    // ── #564: every direction mode must preserve true edge orientation ───────

    #[test]
    fn both_direction_preserves_edge_orientation() {
        let (store, _, class, caller) = seeded_store();
        let args = |direction| GraphQueryArgs {
            query: "PaymentService".to_string(),
            path: None,
            depth: 3,
            direction,
            edge_mode: QueryEdgeMode::Semantic,
            include_noise: false,
        };

        // The store holds exactly one call edge: caller --ref/call--> class.
        // Every mode that surfaces it must report that stored orientation.
        for direction in [QueryDirection::Callers, QueryDirection::Both] {
            let payload = graph_query(&store, &args(direction)).unwrap();
            assert!(
                payload
                    .edges
                    .iter()
                    .any(|e| e.kind == "ref/call" && e.src == caller.id.0 && e.dst == class.id.0),
                "{direction:?}: true edge caller -> class missing from payload"
            );
            assert!(
                !payload
                    .edges
                    .iter()
                    .any(|e| e.kind == "ref/call" && e.src == class.id.0 && e.dst == caller.id.0),
                "{direction:?}: reversed edge class -> caller reported"
            );
            // The tree (default `--format tree` output) must carry the same
            // orientation: the caller is reached over an incoming edge, so the
            // step must be tagged `incoming` for the renderer to draw `←`.
            let caller_step = payload
                .tree
                .iter()
                .find(|t| t.parent == class.id.0 && t.child == caller.id.0)
                .unwrap_or_else(|| panic!("{direction:?}: tree step to the caller missing"));
            assert!(
                caller_step.incoming,
                "{direction:?}: caller tree step not tagged incoming, the \
                 tree renderer would draw the in-edge as outgoing"
            );
        }

        // Deps mode reads only outgoing edges — its steps must not be tagged.
        // Seeded from the caller, whose one outgoing edge is the call itself.
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "processPayment".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        let call_step = payload
            .tree
            .iter()
            .find(|t| t.parent == caller.id.0 && t.child == class.id.0)
            .expect("Deps: tree step for the outgoing call missing");
        assert!(
            !call_step.incoming,
            "Deps: outgoing call step wrongly tagged incoming"
        );
    }

    // ── #517: containment edges terminal in caller traversal ──────────────────

    /// `seeded_store` plus a sibling definition in the same file and a second
    /// caller, so containment-vs-answer ordering and expansion are observable.
    fn seeded_store_with_sibling() -> (SqliteStore, Node, Node, Node, Node, Node) {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let file = node("file", "file", "src/service.ts");
        let class = node("class:PaymentService", "class", "src/service.ts");
        let sibling = node("class:UnrelatedHelper", "class", "src/service.ts");
        let caller = node("fn:processPayment", "function", "src/controller.ts");
        let caller2 = node("fn:refundPayment", "function", "src/controller.ts");
        for n in [&file, &class, &sibling, &caller, &caller2] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(file.id, class.id, EdgeKind::DefinesBinding))
            .unwrap();
        store
            .put_edge(&Edge::new(file.id, sibling.id, EdgeKind::DefinesBinding))
            .unwrap();
        store
            .put_edge(&Edge::new(caller.id, class.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(caller2.id, class.id, EdgeKind::RefCall))
            .unwrap();
        (store, file, class, sibling, caller, caller2)
    }

    #[test]
    fn callers_do_not_expand_the_defining_file() {
        let (store, .., sibling, _caller, _caller2) = seeded_store_with_sibling();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(
            !payload.nodes.iter().any(|n| n.id == sibling.id.0),
            "sibling definition in the same file must not be pulled in through \
             the incidentally-reached file's containment edge"
        );
    }

    #[test]
    fn callers_rank_call_edges_before_containment() {
        let (store, file, class, _sibling, caller, caller2) = seeded_store_with_sibling();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        let steps_from_seed: Vec<&TreeStep> = payload
            .tree
            .iter()
            .filter(|t| t.parent == class.id.0)
            .collect();
        let containment_pos = steps_from_seed
            .iter()
            .position(|t| t.child == file.id.0)
            .expect("containment step to the defining file is present");
        let call_positions: Vec<usize> = steps_from_seed
            .iter()
            .enumerate()
            .filter(|(_, t)| t.child == caller.id.0 || t.child == caller2.id.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(call_positions.len(), 2, "both callers must be present");
        assert!(
            call_positions.iter().all(|&p| p < containment_pos),
            "ref/call steps must precede the defines/binding step"
        );
    }

    #[test]
    fn callers_still_show_the_defining_file() {
        let (store, file, ..) = seeded_store_with_sibling();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        let file_node = payload
            .nodes
            .iter()
            .find(|n| n.id == file.id.0)
            .expect("the defining file is still shown for orientation");
        assert_eq!(file_node.depth, 1);
    }

    /// #317 regression guard: querying a *file* directly still splices in the
    /// callers of the definitions it holds. This is the test that fails if
    /// DD-2 were implemented as an unconditional removal of the splice.
    #[test]
    fn file_seed_still_splices_definition_callers() {
        let (store, file, _class, _sibling, caller, caller2) = seeded_store_with_sibling();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "service.ts".to_string(),
                path: None,
                depth: 2,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        let seed = payload.seed.as_ref().expect("seed must match");
        assert_eq!(
            seed.id, file.id.0,
            "the file itself must be the traversal seed"
        );
        assert!(payload.nodes.iter().any(|n| n.id == caller.id.0));
        assert!(payload.nodes.iter().any(|n| n.id == caller2.id.0));
    }

    #[test]
    fn deps_direction_still_expands_the_file() {
        let (store, _file, class, sibling, ..) = seeded_store_with_sibling();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "service.ts".to_string(),
                path: None,
                depth: 2,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(payload.nodes.iter().any(|n| n.id == class.id.0));
        assert!(payload.nodes.iter().any(|n| n.id == sibling.id.0));
    }

    #[test]
    fn budget_keeps_callers_over_containment() {
        let (store, ..) = seeded_store_with_sibling();
        let mut payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Callers,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(
            payload.nodes.len() >= 4,
            "need seed + 2 callers + the containing file for this test to be meaningful"
        );
        // Budget that admits exactly the seed plus the two callers (which sort
        // before the containment file in BFS discovery order).
        let budget: usize = payload.nodes[..3].iter().map(|n| n.tokens).sum();
        apply_token_budget(&mut payload, budget);
        assert_eq!(payload.nodes.len(), 3);
        assert!(
            payload.nodes[1..].iter().all(|n| n.kind != "file"),
            "the two survivors beyond the seed must be ref/call parents, not the containment file"
        );
    }

    #[test]
    fn ask_query_returns_rows_within_budget() {
        let (store, ..) = seeded_store();
        let payload = ask_query(&store, "PaymentService", None).unwrap();
        assert!(payload.matched);
        assert!(payload.total_tokens <= DEFAULT_TOKEN_BUDGET);
    }

    /// RFC-021 F9 (Phase 4): `ask` and `get_context` must agree on the confidence
    /// label and the abstain decision for the same query — both read the same
    /// `build_seed_set` confidence, so a divergence means a plumbing regression on
    /// one surface. get_context prints it in its `| confidence: X ]` header;
    /// abstaining queries emit no such header on either surface.
    #[test]
    fn ask_and_context_agree_on_confidence_and_abstain() {
        let (store, ..) = seeded_store();
        // "PaymentService?" carries trailing punctuation: both surfaces must
        // normalize identically before build_seed_set (RFC-021 F9 parity fix), or
        // the confidence label / abstain decision can diverge on punctuated NL.
        for q in ["PaymentService", "PaymentService?", "xyzzyplughfrobnicate"] {
            let ask = ask_query(&store, q, None).unwrap();
            let ctx = crate::tools::get_context_raw(&store, q, 4000, false, None);
            // Abstain parity: get_context only prints the confidence header when it
            // did NOT abstain; ask sets matched=false on the same None decision.
            let ctx_grounded = ctx.contains("| confidence:");
            assert_eq!(
                ask.matched, ctx_grounded,
                "abstain decision diverged for {q:?}: ask.matched={} ctx=\n{ctx}",
                ask.matched
            );
            // Label parity when both produced a grounded result.
            if ask.matched {
                let label = ctx
                    .split("| confidence:")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .expect("grounded context has a confidence label");
                assert_eq!(ask.confidence, label, "confidence label diverged for {q:?}");
            }
        }
    }

    // ── #376: docs lane on the `ask` surface ─────────────────────────────────

    /// Install one doc-chunk node plus a doc-space KNN hook that always returns
    /// it, so the lane is exercised without a sidecar. The node deliberately has
    /// no `embed_text`, which routes `rerank_doc_candidates` through its
    /// fail-open cosine fallback — deterministic regardless of whether a
    /// reranker model happens to be installed on the machine running the test.
    fn with_doc_chunk(store: &mut travsr_store::SqliteStore, path: &str, sig: &str) {
        let n = travsr_core::Node::new(
            travsr_core::VName::new("", "", path, "markdown", sig),
            "doc-chunk",
        )
        .with_line(61)
        .with_end_line(68);
        store.put_node(&n).unwrap();
        let id = n.id;
        let hook: travsr_store::EmbedKnnHook =
            std::sync::Arc::new(move |_q, _k| Ok(vec![(id, 0.71)]));
        store.set_embed_doc_knn_hook(hook);
    }

    fn docs_env_on() {
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");
    }

    fn docs_env_off() {
        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        std::env::remove_var("TRAVSR_DOC_FLOOR");
    }

    /// §4.2 "absent, not empty-ish", at the wire level: with the lane off, the
    /// serialized payload must not even carry a `docs` key, so existing JSON
    /// consumers see byte-identical output.
    #[test]
    fn ask_docs_absent_from_json_when_lane_is_off() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_off();
        let (store, ..) = seeded_store();
        let payload = ask_query(&store, "PaymentService", None).unwrap();
        assert!(payload.docs.is_empty());
        // UX-022: with the lane off (and no doc hook) the doc index is not
        // available, and the field is always present so a consumer never has to
        // tell `false` from "absent".
        assert!(!payload.docs_available);
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            !json.contains("\"docs\":"),
            "docs key must be omitted when the lane produced nothing: {json}"
        );
        assert!(
            json.contains("\"docs_available\":false"),
            "docs_available must always serialize as the degradation signal: {json}"
        );
    }

    /// §4.1: the docs section renders `path § Heading Trail:lines` and never a
    /// score — docs are not `AskRow`s precisely so no score column exists.
    #[test]
    fn ask_renders_docs_section_without_a_score() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        let (mut store, ..) = seeded_store();
        with_doc_chunk(
            &mut store,
            "docs/adrs/ADR-001-coding-standards.md",
            "doc:coding-standards/consequences",
        );

        let payload = ask_query(&store, "PaymentService", None).unwrap();
        docs_env_off();

        assert_eq!(payload.docs.len(), 1, "docs: {:?}", payload.docs);
        let line = &payload.docs[0];
        assert!(
            line.contains("docs/adrs/ADR-001-coding-standards.md"),
            "{line}"
        );
        assert!(line.contains("Coding Standards"), "{line}");
        assert!(line.contains(":61-68"), "{line}");
        assert!(
            !line.contains("0.71") && !line.contains("0.7"),
            "docs must never print a raw cosine (§4.1): {line}"
        );
        // Docs never enter `rows`, so they cannot displace a code result.
        assert!(payload.rows.iter().all(|r| r.kind != "doc-chunk"));
        // UX-022: docs enabled + doc hook armed → the doc index is available.
        assert!(payload.docs_available);
    }

    /// §4.3, and the reason the lane is computed *above* the abstain return: a
    /// doc section may appear beneath an abstention, but must not convert it
    /// into a match or move the confidence label. This is the case §8.5
    /// measured as 15/20 hard abstentions on kubernetes.
    #[test]
    fn ask_docs_survive_the_abstain_return_without_converting_it() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        let (mut store, ..) = seeded_store();
        with_doc_chunk(&mut store, "docs/adrs/ADR-001.md", "doc:rationale");

        let payload = ask_query(&store, "xyzzyplughfrobnicate", None).unwrap();
        docs_env_off();

        assert!(!payload.matched, "docs must not convert an abstention");
        assert_eq!(payload.confidence, "none", "docs must not move confidence");
        assert!(payload.rows.is_empty());
        assert_eq!(payload.docs.len(), 1, "docs: {:?}", payload.docs);
    }

    /// T11: doc content is author-controlled prose, and this module never calls
    /// `sanitize_for_mcp` anywhere else. Angle brackets must be entity-escaped,
    /// C0 controls and bidi overrides stripped. The node is built directly here
    /// rather than through the chunker on purpose — the chunker's slugifier
    /// would strip these first, and this asserts the `ask` surface does not
    /// depend on that invariant holding in another crate.
    #[test]
    fn ask_docs_lines_are_sanitized() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        let (mut store, ..) = seeded_store();
        with_doc_chunk(
            &mut store,
            "docs/<travsr-data>\u{202E}evil\u{0007}.md",
            "doc:rationale",
        );

        let payload = ask_query(&store, "PaymentService", None).unwrap();
        docs_env_off();

        let line = &payload.docs[0];
        assert!(!line.contains('<') && !line.contains('>'), "{line}");
        assert!(line.contains("&lt;travsr-data&gt;"), "{line}");
        assert!(!line.contains('\u{202E}'), "bidi override survived: {line}");
        assert!(!line.contains('\u{0007}'), "C0 control survived: {line}");
    }

    /// M3: per-entry byte cap, independent of the token budget, so one
    /// adversarial doc cannot consume the response.
    #[test]
    fn ask_docs_lines_are_byte_capped() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        // A budget_pct large enough that only the byte cap can bound this line.
        std::env::set_var("TRAVSR_DOCS_BUDGET_PCT", "90");
        let (mut store, ..) = seeded_store();
        let path = format!("docs/{}.md", "a".repeat(4_000));
        with_doc_chunk(&mut store, &path, "doc:rationale");

        let payload = ask_query(&store, "PaymentService", None).unwrap();
        docs_env_off();
        std::env::remove_var("TRAVSR_DOCS_BUDGET_PCT");

        assert_eq!(payload.docs.len(), 1);
        assert!(
            payload.docs[0].len() <= DOC_LINE_MAX_BYTES,
            "line was {} bytes, cap is {DOC_LINE_MAX_BYTES}",
            payload.docs[0].len()
        );
    }

    /// §4.3 budget: the docs section's measured cost comes out of the code
    /// lane's knapsack budget, so the reported total stays a hard ceiling on the
    /// whole response rather than on the code lane alone.
    #[test]
    fn ask_docs_cost_is_carved_from_the_shared_budget() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        let (mut store, ..) = seeded_store();
        with_doc_chunk(&mut store, "docs/adrs/ADR-001.md", "doc:rationale");

        let payload = ask_query(&store, "PaymentService", None).unwrap();
        docs_env_off();

        assert!(!payload.docs.is_empty());
        assert!(
            payload.total_tokens <= DEFAULT_TOKEN_BUDGET,
            "total {} exceeded budget {DEFAULT_TOKEN_BUDGET}",
            payload.total_tokens
        );
    }

    /// #870: the two surfaces must agree on doc *presence*. `ask` returns
    /// `docs` as its own field on every return path, so a doc entry is always
    /// findable; `get_context` renders the same entries into its body, where
    /// the section header is the only thing that marks them as prose. That
    /// header used to be dropped whenever the code lane selected four nodes or
    /// fewer (`group_output`), which is the normal shape of a result on a
    /// sparse graph: a repo indexed without Phase B, where PPR has almost no
    /// edges to expand along. The doc lines then reached the model as bare
    /// lines among the code rows, and every consumer that finds the section by
    /// its header (the docs-lane gate, the VS Code Context Explorer) read the
    /// response as carrying no docs at all.
    #[test]
    fn ask_and_context_agree_on_doc_presence_when_few_nodes_are_selected() {
        let _guard = crate::seed::DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        docs_env_on();
        let (mut store, ..) = seeded_store();
        with_doc_chunk(
            &mut store,
            "docs/adrs/ADR-001-coding-standards.md",
            "doc:coding-standards/consequences",
        );

        let ask = ask_query(&store, "PaymentService", None).unwrap();
        let ctx = crate::tools::get_context_raw(&store, "PaymentService", 4000, false, None);
        docs_env_off();

        assert_eq!(ask.docs.len(), 1, "docs: {:?}", ask.docs);
        assert!(
            ctx.contains("docs/adrs/ADR-001-coding-standards.md"),
            "get_context must render the same doc entry: {ctx}"
        );
        assert!(
            ctx.contains("## docs"),
            "the doc entry must carry its section header on both surfaces: {ctx}"
        );
    }

    // ── Ambiguity resolution tests (issue #565 / RFC-002) ──────────────────

    #[test]
    fn test_graph_query_unique_candidate() {
        let (store, _, _, _) = seeded_store();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "class:PaymentService".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();

        assert!(payload.seed.is_some());
        assert_eq!(
            payload.seed.as_ref().unwrap().signature,
            "class:PaymentService"
        );
        assert!(payload.candidates.is_none());
        assert!(!payload.nodes.is_empty());
    }

    #[test]
    fn test_graph_query_ambiguous_symbol() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();
        let node_a = Node {
            id: travsr_core::NodeId(101),
            vname: VName {
                signature: "fn:overloaded".to_string(),
                corpus: "test".to_string(),
                root: String::new(),
                path: "src/file_a.rs".to_string(),
                language: "rust".to_string(),
            },
            kind: "function".to_string(),
            package: String::new(),
            line: Some(57),
            end_line: Some(80),
            test_role: travsr_core::TestRole::None,
        };
        let node_b = Node {
            id: travsr_core::NodeId(102),
            vname: VName {
                signature: "fn:overloaded".to_string(),
                corpus: "test".to_string(),
                root: String::new(),
                path: "src/file_b.rs".to_string(),
                language: "rust".to_string(),
            },
            kind: "function".to_string(),
            package: String::new(),
            line: Some(43),
            end_line: Some(70),
            test_role: travsr_core::TestRole::None,
        };
        store.put_node(&node_a).unwrap();
        store.put_node(&node_b).unwrap();

        // 1. Without path hint: remains ambiguous, seed is None, candidates is Some(2)
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "fn:overloaded".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();

        assert!(payload.seed.is_none());
        assert!(payload.nodes.is_empty());
        assert_eq!(payload.candidates.as_ref().unwrap().len(), 2);

        // 2. Exact path hint resolves uniquely: seed is Some, candidates is None
        let payload_unique = graph_query(
            &store,
            &GraphQueryArgs {
                query: "fn:overloaded".to_string(),
                path: Some("file_a.rs".to_string()),
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(payload_unique.seed.is_some());
        assert_eq!(payload_unique.seed.as_ref().unwrap().path, "src/file_a.rs");
        assert!(payload_unique.candidates.is_none());

        // 3. Invalid path hint: resolves to 0 candidates, seed is None, candidates is None
        let payload_invalid = graph_query(
            &store,
            &GraphQueryArgs {
                query: "fn:overloaded".to_string(),
                path: Some("nonexistent.rs".to_string()),
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(payload_invalid.seed.is_none());
        assert!(payload_invalid.candidates.is_none());
    }

    /// Tier-1 blocker guard: a full signature with more definitions than the
    /// exact-lookup store cap must still surface as ambiguous with a candidate
    /// count that exceeds `AMBIGUOUS_DISPLAY_LIMIT` (so the CLI fires its
    /// truncation notice), not a set silently trimmed to the display limit
    /// (#565 / RFC-002).
    #[test]
    fn test_graph_query_ambiguous_full_signature_exceeds_display_limit() {
        use travsr_core::{Node, NodeId, TestRole, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();
        for i in 0..25u64 {
            store
                .put_node(&Node {
                    id: NodeId(200 + i),
                    vname: VName {
                        signature: "fn:overloaded".to_string(),
                        corpus: "test".to_string(),
                        root: String::new(),
                        path: format!("src/file_{i}.rs"),
                        language: "rust".to_string(),
                    },
                    kind: "function".to_string(),
                    package: String::new(),
                    line: Some(1),
                    end_line: Some(2),
                    test_role: TestRole::None,
                })
                .unwrap();
        }

        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "fn:overloaded".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();

        assert!(payload.seed.is_none());
        let candidates = payload.candidates.as_ref().unwrap();
        // Capped at the store's exact-lookup limit, and that cap is strictly
        // above the display limit so "> display limit" is always detectable.
        assert_eq!(candidates.len(), travsr_store::NODE_EXACT_LOOKUP_LIMIT);
        assert!(candidates.len() > AMBIGUOUS_DISPLAY_LIMIT);
    }

    /// File-name queries: two files sharing a basename must disambiguate like
    /// symbols (candidates listed, no arbitrary first-pick), and a `--path` pin
    /// resolves to the unique file (#565 / RFC-002).
    #[test]
    fn test_graph_query_ambiguous_file_name() {
        use travsr_core::{Node, NodeId, TestRole, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();
        for (i, dir) in ["src/a", "src/b"].iter().enumerate() {
            store
                .put_node(&Node {
                    id: NodeId(300 + i as u64),
                    vname: VName {
                        signature: format!("{dir}/service.ts"),
                        corpus: "test".to_string(),
                        root: String::new(),
                        path: format!("{dir}/service.ts"),
                        language: "typescript".to_string(),
                    },
                    kind: "file".to_string(),
                    package: String::new(),
                    line: None,
                    end_line: None,
                    test_role: TestRole::None,
                })
                .unwrap();
        }

        // Bare basename with two matching files: ambiguous, no seed.
        let ambiguous = graph_query(
            &store,
            &GraphQueryArgs {
                query: "service.ts".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(ambiguous.seed.is_none());
        assert_eq!(ambiguous.candidates.as_ref().unwrap().len(), 2);

        // A `--path` pin resolves to the unique file.
        let pinned = graph_query(
            &store,
            &GraphQueryArgs {
                query: "service.ts".to_string(),
                path: Some("src/a/service.ts".to_string()),
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();
        assert!(pinned.candidates.is_none());
        assert_eq!(pinned.seed.as_ref().unwrap().path, "src/a/service.ts");
    }

    #[test]
    fn test_graph_query_still_ambiguous_path() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();
        let node_a = Node {
            id: travsr_core::NodeId(101),
            vname: VName {
                signature: "fn:overloaded".to_string(),
                corpus: "test".to_string(),
                root: String::new(),
                path: "src/subdir1/file.rs".to_string(),
                language: "rust".to_string(),
            },
            kind: "function".to_string(),
            package: String::new(),
            line: Some(57),
            end_line: Some(80),
            test_role: travsr_core::TestRole::None,
        };
        let node_b = Node {
            id: travsr_core::NodeId(102),
            vname: VName {
                signature: "fn:overloaded".to_string(),
                corpus: "test".to_string(),
                root: String::new(),
                path: "src/subdir2/file.rs".to_string(),
                language: "rust".to_string(),
            },
            kind: "function".to_string(),
            package: String::new(),
            line: Some(43),
            end_line: Some(70),
            test_role: travsr_core::TestRole::None,
        };
        store.put_node(&node_a).unwrap();
        store.put_node(&node_b).unwrap();

        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "fn:overloaded".to_string(),
                path: Some("file.rs".to_string()),
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();

        assert!(payload.seed.is_none());
        assert_eq!(payload.candidates.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_graph_query_legacy_recall_fallback() {
        let (store, _, _, _) = seeded_store();
        let payload = graph_query(
            &store,
            &GraphQueryArgs {
                query: "Payment".to_string(),
                path: None,
                depth: 3,
                direction: QueryDirection::Deps,
                edge_mode: QueryEdgeMode::Semantic,
                include_noise: false,
            },
        )
        .unwrap();

        let seed = payload
            .seed
            .as_ref()
            .expect("seed must match via recall fallback");
        assert!(seed.signature == "class:PaymentService" || seed.signature == "fn:processPayment");
    }
}
