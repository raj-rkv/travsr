/// Tier-0 seed quality: model-free seed selection, coverage, confidence, and abstention.
///
/// Pipeline (FTS-only path — zero embeddings required):
///   tokenize_query → per-token anchor resolution (IDF-weighted) → whole-query lexical FTS
///   → RRF fusion → coverage + confidence classification → optional abstention
///
/// KNN (Tier 1) slots into the same rrf_fuse call when the embed sidecar is present.
use std::collections::HashMap;

use travsr_core::{EdgeKind, Node as CoreNode, NodeId};
use travsr_retrieval::EdgeFilter;
use travsr_store::{SqliteStore, Store};

use crate::tools::{is_anchor_noise, is_noise_seed, kind_boost};

// ── Tunable constants (all env-overridable) ──────────────────────────────────

/// Maximum symbol-frequency for an anchor to count as "rare/trusted" (near-IDF weight 1.0).
fn rare_anchor_max() -> usize {
    std::env::var("TRAVSR_RARE_ANCHOR_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// RFC-022 D4 (RC-4): minimum share of the query's total IDF mass a rare exact
/// anchor's term must carry for the deterministic G1 bypass to fire — i.e. for the
/// term to be the query's *subject* rather than an incidental rare word. A literal
/// symbol lookup (`GetWarningsForPod`, `SqliteStore search_nodes_by_name`)
/// concentrates its IDF in the named symbol(s); a conceptual query that merely
/// contains a rare word (`how are results **trimmed** to fit a token budget`,
/// `shopping cart **checkout** payment flow`) spreads its IDF across several tokens,
/// so no single term clears the share. Default 0.55; override via
/// `TRAVSR_G1_SUBJECT_IDF_SHARE`.
fn g1_subject_idf_share() -> f32 {
    std::env::var("TRAVSR_G1_SUBJECT_IDF_SHARE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=1.0).contains(&x))
        .unwrap_or(0.55)
}

/// Coverage threshold for Exact / Strong confidence.
fn coverage_strong() -> f32 {
    std::env::var("TRAVSR_COVERAGE_STRONG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6)
}

/// Coverage threshold for Weak confidence.
fn coverage_weak() -> f32 {
    std::env::var("TRAVSR_COVERAGE_WEAK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25)
}

/// Raw (positive) BM25 score floor for Strong confidence.
/// SQLite FTS5 `-bm25()` on this corpus: ~0.1 (weak) to ~3+ (strong). Default 0.5.
fn bm25_strong_floor() -> f32 {
    std::env::var("TRAVSR_BM25_STRONG_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| x > 0.0)
        .unwrap_or(0.5)
}

/// Maximum guesses to include when confidence == None (abstention path).
pub(crate) fn abstain_max_guesses() -> usize {
    std::env::var("TRAVSR_ABSTAIN_GUESSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// Minimum IDF weight for a resolved token to count toward coverage and the anchor escape hatch.
///
/// Tokens below this threshold (e.g. "map", "get", "list", "run") match too many
/// nodes to be meaningful query signals, so they don't count as "covered" even when
/// they resolve.  Default 0.55 ≈ symbol appears in ≤ ~52 nodes out of 7 k (~0.75%).
/// For a corpus of n nodes, freq_max ≈ n / e^(threshold * ln(n)).
fn idf_coverage_min() -> f32 {
    std::env::var("TRAVSR_IDF_COVERAGE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| x > 0.0 && x <= 1.0)
        .unwrap_or(0.55)
}

/// Anchor-emit IDF cut (#478 RFC-023 §9): a token below this is too generic to
/// emit as an anchor at all — stricter than [`idf_coverage_min`], which only
/// controls whether an already-emitted anchor counts toward coverage.
fn anchor_emit_cut() -> f32 {
    std::env::var("TRAVSR_ANCHOR_EMIT_CUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| x > 0.0 && x <= 1.0)
        .unwrap_or(0.15)
}

/// #709: byte-trigram Jaccard floor for the per-token typo correction. A query
/// token that resolves to nothing exactly is corrected to a real symbol only
/// when a UNIQUE leaf name clears this similarity, `htpresponse` → `HttpResponse`
/// is 0.727, a comfortable margin above the default. Kept strict (and env-tunable
/// for bench sweeps) so near-miss salad words do not ground; the cross-encoder
/// reranker is the downstream precision backstop.
fn fuzzy_correct_jaccard() -> f64 {
    std::env::var("TRAVSR_FUZZY_CORRECT_JACCARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f64| x > 0.0 && x <= 1.0)
        .unwrap_or(0.7)
}

/// RRF k constant — controls how sharply the top ranks dominate.
fn rrf_k() -> f32 {
    std::env::var("TRAVSR_RRF_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| x > 0.0)
        .unwrap_or(60.0)
}

// ── #463 seed diversity: logic-bearing exact-anchor ordering ─────────────────
//
// Per-token exact-anchor resolution emits the top FTS name-matches as anchors that
// reach the cross-encoder. `search_nodes_by_name`'s rank is dominated by name-match
// POSITION (exact > `:suffix` > prefix > substring), with kind only a minor
// tiebreaker WITHIN a band — so a field-only/container node (`struct:Daemon`, fields
// only) outranks the logic-bearing definition of the same symbol (`impl:Daemon`,
// `Daemon::run`) whenever the struct's signature is the closer literal match, and the
// reranker then judges the bare struct, not the code that answers the query.
//
// #463 reorders a SMALL name-match head so a body-bearing definition is preferred
// before the bounded emit, WITHOUT widening the pool to path-substring collisions.
// Measured net-positive with zero salad-abstention regression on travsr + kubernetes
// (the RFC-021 §9.8 guardrail): hit@5 +0.04 (travsr) / +0.17 (k8s), MRR +0.06 / +0.02,
// salads still 100% abstained on both. On by default; `TRAVSR_ANCHOR_KIND_PRIORITY=0`
// restores the original FTS-rank order (rollback escape hatch).

/// Whether the per-token anchor emit reorders its name-match head by logic-bearing
/// kind first (#463). On by default; disabled only by an explicit falsey env value
/// (`0` / `false` / `off`), which restores the pre-#463 FTS-rank order byte-for-byte.
fn anchor_kind_priority() -> bool {
    !matches!(
        std::env::var("TRAVSR_ANCHOR_KIND_PRIORITY").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Size of the name-match head reconsidered by [`order_anchor_candidates`] before the
/// bounded emit. Kept small so name-match quality still gates the pool — a body-bearing
/// def only jumps ahead of a field/struct that was ALSO a top name-match, never a
/// path-substring collision far down the FTS list. Default 6; override via
/// `TRAVSR_ANCHOR_REORDER_WINDOW` (clamped to ≥ 3 so the emitted top-3 always has a
/// reorderable window).
fn anchor_reorder_window() -> usize {
    std::env::var("TRAVSR_ANCHOR_REORDER_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 3)
        .unwrap_or(6)
}

/// Logic-bearing rank for #463 anchor reordering: 0 = has a body (impl/method/fn),
/// 1 = container/type, 2 = leaf (field/var/const/import). Lower sorts first.
fn kind_logic_rank(kind: &str) -> u8 {
    match kind {
        "method" | "function" | "constructor" | "impl" => 0,
        "field" | "var" | "variable" | "property" | "constant" | "static" | "import"
        | "file-module" => 2,
        _ => 1,
    }
}

/// Line-span of a node (0 when unknown). Secondary #463 ordering key — among equally
/// logic-bearing candidates a larger body is the better "answer" for a vague query.
fn span_size(node: &CoreNode) -> u32 {
    match (node.line, node.end_line) {
        (Some(a), Some(b)) if b >= a => b - a,
        _ => 0,
    }
}

/// #463: order a token's FTS name-match candidates so logic-bearing definitions
/// (impl/method/fn) precede field-only/container nodes, tie-broken by larger span,
/// operating only on the top-`window` name-matches. The sort is **stable**, so within
/// an equal `(kind_logic_rank, span)` key the original FTS rank is preserved. Returns
/// borrows into the head of `exact_nodes` (never more than `window`).
///
/// `O(window log window)`.
fn order_anchor_candidates<'a>(exact_nodes: &[&'a CoreNode], window: usize) -> Vec<&'a CoreNode> {
    let mut head: Vec<&CoreNode> = exact_nodes.iter().take(window).copied().collect();
    head.sort_by(|a, b| {
        kind_logic_rank(&a.kind)
            .cmp(&kind_logic_rank(&b.kind))
            .then_with(|| span_size(b).cmp(&span_size(a)))
    });
    head
}

/// Oracle top-cosine at/above which a confident embedding cluster *alone* (no
/// lexical anchor) is a Strong match. Sits in the measured gap between
/// answerable (oracle_top ≥ 0.77) and nonsense (≤ 0.64) queries on bge-small.
/// Override via `TRAVSR_SEMANTIC_PROMOTE_STRONG`.
fn semantic_promote_strong() -> f32 {
    std::env::var("TRAVSR_SEMANTIC_PROMOTE_STRONG")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| x > 0.0 && x <= 1.0)
        .unwrap_or(0.72)
}

/// Minimum over-fetched candidates within the cosine floor band required for
/// semantic promotion — guards against a single fluke neighbour grounding a
/// query. Override via `TRAVSR_SEMANTIC_PROMOTE_MIN_NEAR`.
fn semantic_promote_min_near() -> usize {
    std::env::var("TRAVSR_SEMANTIC_PROMOTE_MIN_NEAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// Minimum anchor cosine for the low-coverage `embed_confirmed_specific` rescue
/// (RFC-019). A specific anchor confirms a diluted-coverage query to Strong only
/// when it sits in the measured *answerable* band, not merely above the weak
/// `SEMANTIC_ABS_FLOOR` (0.55). Separates a genuine low-coverage query ("Load
/// mutationg manifests" → "manifests" ≈0.71) from a salad whose lone resolved word
/// is also literally in the query ("find code by semantic meaning" → "semantic"
/// ≈0.615). Override via `TRAVSR_CONFIRM_ANCHOR_FLOOR`.
fn confirm_anchor_floor() -> f32 {
    std::env::var("TRAVSR_CONFIRM_ANCHOR_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=1.0).contains(&x))
        .unwrap_or(0.66)
}

/// Oracle top-cosine below which the embedding is confident *nothing* in the
/// corpus is near the query. A Weak resting only on coincidental lexical
/// coverage is then vetoed to None (unless a rare exact anchor was named).
/// Default 0.55 sits above the measured nonsense leak (0.49) and far below the
/// answerable floor (0.77). Override via `TRAVSR_SEMANTIC_VETO_FLOOR`.
fn semantic_veto_floor() -> f32 {
    std::env::var("TRAVSR_SEMANTIC_VETO_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=1.0).contains(&x))
        .unwrap_or(0.55)
}

/// Reference-scale (bge) recall floor (#462): the cosine just above the
/// off-domain/salad band, below which a KNN neighbour is treated as noise. Above
/// it, a neighbour is a genuine (weak) semantic match that must NOT be
/// truncated by `semantic_validate` (WS1) nor thrown away as `Confidence::None`
/// (WS2). Authored in the bge reference scale like every other floor here and
/// mapped into the active model's band via [`Calibration`]. Kept just above
/// `REF_LO` (the nonsense/salad p95 anchor) so it clears salads by a small,
/// per-model margin — the k8s/arctic band separates conceptual golds (≥0.44)
/// from salads (≤0.41) at ~0.42, exactly `Calibration::map(0.65)` there.
const SEMANTIC_RECALL_FLOOR_REF: f32 = 0.65;

/// Un-calibrated / bge-reference fallback for [`semantic_recall_floor`]. bge's
/// salad neighbours reach ~0.75 (RFC-019's 0.751 collision), so on an
/// un-calibrated index — where `map` is the identity and cannot place the floor
/// inside a measured band — the recall floor must stay at the conservative
/// high-confidence bar (the pre-#462 `EMBED_SCORE_THRESHOLD`). This preserves
/// bge / un-calibrated behaviour byte-for-byte; only a genuinely calibrated model
/// (arctic et al.) gets the aggressive mapped floor.
const RECALL_FLOOR_IDENTITY: f32 = 0.75;

/// Absolute cosine recall floor in the ACTIVE model's scale (#462). Single source
/// of truth shared by `embed_path_seeds` (KNN seed admission, WS1b),
/// `semantic_validate` (anti-truncation, WS1) and `classify_confidence` (abstain
/// rescue, WS2). `TRAVSR_SEMANTIC_RECALL_FLOOR` overrides with an absolute value.
pub(crate) fn semantic_recall_floor(cal: &Calibration) -> f32 {
    if let Some(x) = floor_env("TRAVSR_SEMANTIC_RECALL_FLOOR") {
        return x;
    }
    if *cal == Calibration::IDENTITY {
        RECALL_FLOOR_IDENTITY
    } else {
        cal.map(SEMANTIC_RECALL_FLOOR_REF)
    }
}

/// Per-(model, corpus) semantic-floor calibration.
///
/// Every absolute cosine floor in this module (`SEMANTIC_ORACLE_MIN`,
/// `SEMANTIC_ABS_FLOOR`, `semantic_veto_floor`, `confirm_anchor_floor`,
/// `semantic_promote_strong`, …) was tuned to **bge-small**'s query↔passage cosine
/// scale — its answerable queries sit ≳0.77 and its nonsense leak ≲0.64. A model
/// with a different scale (arctic-embed-256's Matryoshka-truncated vectors and
/// asymmetric query prefix land materially lower) would over- or under-abstain
/// against those fixed numbers: the k8s bench showed literal symbol queries whose
/// correct node was retrieved but demoted to "speculative guesses".
///
/// `Calibration` carries the active model's two measured anchors and maps each
/// reference-model floor into the model's own cosine space, preserving the floor's
/// *position within the answerable band*. It is derived automatically, label-free,
/// at reindex (see `travsr_plugin_host::calibrate_semantic_floors`) and stored in
/// graph.db meta. When absent/degenerate the mapping is the identity, so bge-small
/// and every un-calibrated repo behave byte-for-byte as before.
///
/// Generic by construction: any model — bundled, future-release, or a user's own
/// `embed_catalog.toml` entry — self-calibrates on its first reindex with no manual
/// numbers. The floors are authored once (below), in the reference scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Calibration {
    /// "Confident but unrelated" cosine scale (nonsense-leak p95).
    pub lo: f32,
    /// "Query matches its answer" cosine scale (self-match p50).
    pub hi: f32,
}

impl Calibration {
    /// bge-small reference anchors the floors in this module were tuned to, in the
    /// same measurement space the auto-calibration probe uses:
    ///   • `REF_LO` = bge's "confident but unrelated" (nonsense) cosine scale — the
    ///     level the veto / abstain floors sit just above (`semantic_veto_floor`,
    ///     `SEMANTIC_ABS_FLOOR` = 0.55).
    ///   • `REF_HI` = bge's "a query matches its answer" (answerable) cosine scale —
    ///     the level the promote floor targets (`semantic_promote_strong` ≈ 0.72,
    ///     answerable band ≈ 0.77).
    /// The probe measures each model's equivalents (nonsense p95, self-match p50); a
    /// floor authored as a bge cosine then maps by the ratio of the two bands. They
    /// MUST bracket the floors so no floor extrapolates below `lo` — that is what made
    /// the veto vanish on a compressed model (the k8s arctic-256 abstain regression).
    const REF_LO: f32 = 0.55;
    const REF_HI: f32 = 0.77;

    /// Identity mapping — the reference model and un-calibrated repos.
    pub(crate) const IDENTITY: Calibration = Calibration {
        lo: Self::REF_LO,
        hi: Self::REF_HI,
    };

    /// Load `{lo, hi}` from graph.db meta (`embed_cos_lo` / `embed_cos_hi`).
    /// Falls back to [`Self::IDENTITY`] when absent, unparseable, non-finite, or
    /// degenerate (`hi <= lo`) — so a missing/corrupt calibration can only ever
    /// reproduce today's behaviour, never break it.
    pub(crate) fn load(store: &SqliteStore) -> Calibration {
        let g = |k: &str| -> Option<f32> {
            store
                .get_meta(k)
                .ok()
                .flatten()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .filter(|x| x.is_finite())
        };
        match (g("embed_cos_lo"), g("embed_cos_hi")) {
            (Some(lo), Some(hi)) if hi > lo => Calibration { lo, hi },
            _ => Self::IDENTITY,
        }
    }

    /// Map a reference-model cosine *threshold* into this model's cosine space,
    /// preserving its fractional position within the `[REF_LO, REF_HI]` band.
    /// Identity when `self == IDENTITY`. Clamped to a valid cosine `[-1, 1]`.
    ///
    /// `O(1)`.
    #[inline]
    pub(crate) fn map(&self, ref_cos: f32) -> f32 {
        let frac = (ref_cos - Self::REF_LO) / (Self::REF_HI - Self::REF_LO);
        (self.lo + frac * (self.hi - self.lo)).clamp(-1.0, 1.0)
    }

    /// Scale a band-relative *width* (e.g. `SEMANTIC_REL_DELTA`) into this model's
    /// band, so a "keep within Δ of the top" rule stays proportional to the model's
    /// answerable-band width rather than an absolute cosine gap. `O(1)`.
    #[inline]
    pub(crate) fn map_delta(&self, ref_delta: f32) -> f32 {
        ref_delta * (self.hi - self.lo) / (Self::REF_HI - Self::REF_LO)
    }
}

// ── Core types ────────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryIntent {
    Lookup,
    Callers,
    Deps,
    Explain,
    Relation,
    /// Query with no named symbol — all tokens are NL words.
    Conceptual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeedSource {
    Exact,
    Lexical,
    Knn,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedTerm {
    pub token: String,
    /// True if ≥1 exact anchor hit was found.
    pub resolved: bool,
    /// Number of graph symbols whose signature contains this token.
    pub symbol_freq: usize,
    /// IDF weight of this token: high = rare/specific, low = generic ("map", "get").
    pub idf_w: f32,
    /// The top-ranked node ID for this token (if resolved). Ties G1 rarity to
    /// the specific term that produced the exact anchor (RFC-021 F3).
    pub top_node: Option<NodeId>,
    /// #540: anchors this token actually emitted.
    ///
    /// Measured, not derived. `min(symbol_freq, MAX_ANCHORS_PER_TOKEN)` is an
    /// upper bound that the loop then reduces further — `is_anchor_noise`,
    /// the RBAC filter, the `contains_token` boundary check and the per-path
    /// cap all drop candidates — so reporting the bound would attribute
    /// relevance and quality drops to capacity. A token suppressed by the IDF
    /// cut never enters the loop at all and stays 0.
    pub anchors_emitted: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Seed {
    pub node: NodeId,
    /// PPR personalisation weight derived from IDF × kind_boost or BM25 × kind_boost.
    pub weight: f32,
    /// Which retrieval source contributed this seed. Reserved for response annotation.
    #[allow(dead_code)]
    pub source: SeedSource,
    /// Raw score before normalisation. Reserved for response annotation.
    #[allow(dead_code)]
    pub score: f32,
    /// RFC-021 Phase 2: absolute cross-encoder relevance score in `[0, 1]`,
    /// `None` when the seed wasn't in the reranked top-K or the reranker is
    /// absent/disabled/degraded for this query. Does not affect
    /// `classify_confidence` yet (Phase 3) — attach + reorder only.
    pub rerank_score: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Confidence {
    Exact,
    Strong,
    Weak,
    None,
}

impl Confidence {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Strong => "strong",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }
}

#[derive(Debug)]
pub(crate) struct SeedSet {
    /// Classified intent — reserved for future intent-routing.
    #[allow(dead_code)]
    pub intent: QueryIntent,
    pub seeds: Vec<Seed>,
    pub terms: Vec<ResolvedTerm>,
    /// Fraction of content tokens that resolved to at least one anchor: [0.0, 1.0].
    /// Reserved for future response annotation.
    #[allow(dead_code)]
    pub coverage: f32,
    pub confidence: Confidence,
    /// Top raw BM25 score (positive) from the lexical FTS path; 0.0 if no FTS results.
    /// Reserved for future response annotation.
    #[allow(dead_code)]
    pub top_bm25: f32,
}

impl SeedSet {
    /// Extract `(NodeId, weight)` pairs for `ppr_weighted`.
    pub(crate) fn ppr_seeds(&self) -> Vec<(NodeId, f32)> {
        self.seeds.iter().map(|s| (s.node, s.weight)).collect()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    /// RFC-021 F9: absolute cross-encoder scores for the seeds the reranker judged.
    ///
    /// Seeds absent from the map had no `rerank_score` — G1-bypassed, beyond the
    /// reranked top-K, or the reranker was off/degraded — and fall back to their
    /// normalized PPR score at display time (see [`display_score`]).
    pub(crate) fn rerank_scores(&self) -> HashMap<NodeId, f32> {
        self.seeds
            .iter()
            .filter_map(|s| s.rerank_score.map(|r| (s.node, r)))
            .collect()
    }
}

/// RFC-021 F9: the score shown next to a node in `ask` / `get_context` output.
///
/// Unifies both surfaces on a single `[0,1]` scale (cross-surface parity):
/// - A reranked seed shows its absolute cross-encoder score, replacing the
///   normalized-PPR "always 1.00" artefact.
/// - A primary seed the reranker never scored (G1 bypass / reranker off) shows
///   its own normalized PPR score, uncapped — it is still a genuine anchor.
/// - An expanded (non-seed) neighbour shows its normalized PPR score capped at
///   `expanded_cap` (the best seed's rerank score) so a neighbour can never
///   outrank the seed it was pulled in by.
pub(crate) fn display_score(
    node: NodeId,
    normalized_ppr: f32,
    seed_rerank: &HashMap<NodeId, f32>,
    is_primary_seed: bool,
    expanded_cap: Option<f32>,
    confidence: Confidence,
) -> f32 {
    if let Some(&r) = seed_rerank.get(&node) {
        r
    } else if is_primary_seed {
        normalized_ppr.min(confidence_display_ceiling(confidence))
    } else {
        match expanded_cap {
            Some(cap) => normalized_ppr.min(cap),
            None => normalized_ppr,
        }
    }
}

/// RFC-022 §14: the provenance bucket a returned node is grouped under.
///
/// A clean per-node partition keyed on how the node entered the result — its seed
/// provenance — so the section header carries a trustworthy "how sure are we"
/// signal:
/// - [`Exact`](MatchSource::Exact): a primary seed whose strongest [`SeedSource`]
///   is [`Exact`](SeedSource::Exact) — a literal symbol / FTS name match. Highest
///   structural certainty.
/// - [`Semantic`](MatchSource::Semantic): a primary seed reached by embedding-KNN
///   or fuzzy-lexical retrieval (not an exact name). Sorted by the shown score,
///   which for a reranked seed is its cross-encoder score (RC-2 transparency).
/// - [`Relevant`](MatchSource::Relevant): a non-seed node PPR/traversal reached —
///   supporting structure.
///
/// Display-only: this classifies the *already-selected* knapsack set for grouped
/// presentation and never influences which nodes are selected or their score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchSource {
    Exact,
    Semantic,
    /// #376 Phase 2: a doc-chunk hit from the dedicated, floored doc-space
    /// lane (§4). Never a member of the code seed set — it does not go
    /// through `match_source()` below, only through `doc_lane_seeds()`.
    Docs,
    /// #479: a node classified as test code at index time (`nodes.test_role !=
    /// None`). Overrides whatever seed-provenance bucket the node would have had,
    /// so a `#[test]` fn that is also an exact/semantic seed still renders in the
    /// capped `tests` section below the implementation sections, never at the top
    /// of `exact`/`semantic`.
    Tests,
    Relevant,
}

impl MatchSource {
    /// Lowercase tag used in the `--format json` `match_source` field and the CLI
    /// section headers. Stable — bench parsers and the VS Code Context Explorer
    /// key off these exact strings.
    pub(crate) fn label(self) -> &'static str {
        match self {
            MatchSource::Exact => "exact",
            MatchSource::Semantic => "semantic",
            MatchSource::Docs => "docs",
            MatchSource::Tests => "tests",
            MatchSource::Relevant => "relevant",
        }
    }

    /// Section order: certainties → strong candidates → design intent → tests →
    /// context (plan §4.1). Docs sits below code sections so code always
    /// leads when it has an answer, and above Relevant because graph-adjacent
    /// filler is less informative than a floored, on-topic doc section. Tests
    /// sits below all of those (#479): a test entry point is never the answer to
    /// "what does X do", so it renders after implementation and design intent but
    /// above unfocused Relevant filler.
    pub(crate) fn trust_rank(self) -> u8 {
        match self {
            MatchSource::Exact => 0,
            MatchSource::Semantic => 1,
            MatchSource::Docs => 2,
            MatchSource::Tests => 3,
            MatchSource::Relevant => 4,
        }
    }
}

/// RFC-022 §14: classify a selected node into its match-source bucket by seed
/// provenance. `is_primary_seed` is membership in the pre-enrichment primary-seed
/// set; `is_exact_source` is whether that seed's strongest [`SeedSource`] is
/// [`Exact`](SeedSource::Exact) (a literal-name / FTS match). A seed is therefore
/// either Exact or Semantic; a non-seed is always Relevant.
pub(crate) fn match_source(is_primary_seed: bool, is_exact_source: bool) -> MatchSource {
    if !is_primary_seed {
        MatchSource::Relevant
    } else if is_exact_source {
        MatchSource::Exact
    } else {
        MatchSource::Semantic
    }
}

/// Precedence of a [`SeedSource`] when a node is reached by several seeds. Exact
/// (literal-name / FTS) is the most trustworthy, then embedding-KNN, then
/// fuzzy-lexical. The strongest source decides both the node's displayed
/// `[via: · source]` provenance badge and its RFC-022 §14 match-source bucket.
pub(crate) fn seed_source_rank(source: SeedSource) -> u8 {
    match source {
        SeedSource::Exact => 3,
        SeedSource::Knn => 2,
        SeedSource::Lexical => 1,
    }
}

/// RFC-022 §14: collapse a seed set to `node -> strongest [`SeedSource`]`.
///
/// This is the single classifier both `get_context` (`tools.rs`) and `ask`
/// (`query.rs`) key their match-source bucket off, so a node lands in the same
/// section on either surface. Because [`Exact`](SeedSource::Exact) outranks every
/// other source, a node reached by an exact anchor *and* a weaker source still
/// buckets as Exact — the two surfaces cannot drift even if the ranking changes.
pub(crate) fn strongest_seed_sources(seeds: &[Seed]) -> HashMap<NodeId, SeedSource> {
    let mut m: HashMap<NodeId, SeedSource> = HashMap::new();
    for s in seeds {
        m.entry(s.node)
            .and_modify(|cur| {
                if seed_source_rank(s.source) > seed_source_rank(*cur) {
                    *cur = s.source;
                }
            })
            .or_insert(s.source);
    }
    m
}

/// RFC-022 §14 gate — **default on**. Match-source grouping (Exact → Semantic →
/// Relevant sections, with the seed source hoisted into the header) is the format
/// every consumer now parses: the VS Code Context Explorer, the `ask` CLI table,
/// and the k8s bench harness. `TRAVSR_MATCH_SOURCE=0` is the kill-switch that
/// restores the flat, byte-for-byte pre-§14 output for a release cycle while the
/// grouped format is validated in the field; any other value (or unset) keeps it on.
pub(crate) fn match_source_grouping_enabled() -> bool {
    std::env::var("TRAVSR_MATCH_SOURCE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

// ── #376 Phase 2: docs lane ───────────────────────────────────────────────────
//
// A dedicated, dense-only, floored retrieval path for `kind == "doc-chunk"`
// nodes (plan §4). Deliberately NOT part of `build_seed_set`/`match_source`:
// docs never compete with code for a seed slot, are never PPR-expanded, and
// never influence `Confidence`/abstention — see §4.3. `is_noise_seed`
// (`tools.rs`) permanently excludes `kind == "doc-chunk"` from the generic
// seed pipeline for exactly this reason; this module is the *only* path a
// doc-chunk node can reach a `get_context`/`ask` response through.

/// Plan §4.2: env-overridable like every other retrieval constant in this
/// module. 0.42 measured in the plan's offline prototype (§8.3): across both
/// bench repos no nonsense/salad query produced a doc cosine above 0.382,
/// while gold p10 was 0.439 (k8s) / 0.469 (travsr) — 0.42 sits inside that
/// separation gap and is reported to transfer across repos (§8.3, tentative
/// pending a third corpus).
pub(crate) fn doc_floor() -> f32 {
    std::env::var("TRAVSR_DOC_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&x| (0.0..=1.0).contains(&x))
        .unwrap_or(0.42)
}

/// The repo root this process is serving, as recorded in `graph.db` meta at
/// index time — the same lookup `tools.rs` uses to resolve snippet paths. It is
/// how a retrieval-side config read finds the *repo's* `.travsr/config.toml`
/// without threading a path through every call site (the `Calibration::load`
/// precedent).
///
/// `None` degrades to global-config-and-env only, which is correct: an
/// in-memory or path-less store has no repo layer to read.
fn config_repo_root(store: &SqliteStore) -> Option<std::path::PathBuf> {
    // A stale root means `<repo>/.travsr/config.toml` is not found, so the repo
    // layer silently drops out of the layered config and retrieval runs on
    // global-and-env defaults with no error and no log line (#749 review).
    store.resolve_repo_root()
}

/// Plan §4.2: "cap the section at 3 entries."
///
/// #376 O1: resolved through the layered config (`env > repo > global >
/// default`), not the environment alone. See [`docs_enabled`] for why that
/// distinction is load-bearing rather than cosmetic.
pub(crate) fn docs_max_results(store: &SqliteStore) -> usize {
    travsr_config::effective("docs.max_results", config_repo_root(store).as_deref())
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(3)
}

/// Plan §7: the docs lane's feature flag. Distinct from hook *availability*
/// (`doc_knn` being `Some`): this is the flag, that is the capability gate (old
/// sidecar, or a repo with no doc-chunk nodes at all).
///
/// # Why this is a config key and not an env var (#376 O1 / G1)
///
/// Retrieval happens in whichever process holds the index — the **daemon** for
/// `travsr ask`, the MCP server for `get_context`. `TRAVSR_DOCS_ENABLED=1`
/// exported in a user's shell therefore reaches the CLI, which is not the
/// process that reads this, and does nothing at all with no error and no
/// warning (plan §18.7, and §20.3 F-D which added the CLI-side note). A config
/// key is read by whichever process performs the retrieval, from a file both
/// processes can see, which is the only form of this switch that works.
///
/// Env still wins when set (it is the highest stored layer), so every existing
/// script, test and bench harness that exports the variable in the *daemon's*
/// environment keeps its current behaviour byte-for-byte.
pub(crate) fn docs_enabled(store: &SqliteStore) -> bool {
    travsr_config::effective_bool("docs.enabled", config_repo_root(store).as_deref())
        .unwrap_or(true)
}

/// The single lock serializing every test that mutates the process-global
/// `docs.*` env knobs (they are all env-var-driven — see the #376 Phase 1
/// completion notes on why travsr-config is not wired into travsr-mcp).
///
/// Crate-visible and defined here, next to the knobs it guards, because
/// `seed.rs`, `tools.rs` and `query.rs` all exercise the docs lane and all run
/// in the *same* test binary. Two of them previously held two independent
/// module-local locks, which serialized each module against itself while still
/// racing the other — a latent flake that only showed up once a third module's
/// tests widened the window.
#[cfg(test)]
pub(crate) static DOCS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Plan §4.3: percentage of `token_budget` the docs section may claim — a
/// clamp applied to the *measured* doc-block token cost, never a reservation
/// (a docs-free repo/query must stay byte-identical to pre-Phase-2 output).
///
/// #376 O1: layered like the other two user-facing docs knobs.
pub(crate) fn docs_budget_pct(store: &SqliteStore) -> f32 {
    travsr_config::effective("docs.budget_pct", config_repo_root(store).as_deref())
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|&x| x > 0.0 && x <= 100.0)
        .unwrap_or(20.0)
}

/// Plan §4.2: "Dense-only. No BM25 leg, no RRF" (§8.1 measured BM25 worse on
/// both repos and RRF actively hurting on travsr).
///
/// Pure floor+sort+cap: filters `candidates` to those at or above
/// [`doc_floor`], sorts by descending score, and caps at [`docs_max_results`].
/// Used both as the pre-reranker filter shape and, unchanged, as
/// `tools::rerank_doc_candidates`'s cosine fallback when the reranker has no
/// opinion (§4.2: below the floor this returns an empty `Vec`, and the caller
/// must render no docs section at all — "absent, not empty-ish").
pub(crate) fn cosine_floor_select(
    store: &SqliteStore,
    candidates: Vec<(NodeId, f32)>,
) -> Vec<(NodeId, f32)> {
    let floor = doc_floor();
    let mut hits: Vec<(NodeId, f32)> = candidates
        .into_iter()
        .filter(|&(_, score)| score >= floor)
        .collect();
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(docs_max_results(store));
    hits
}

pub(crate) type DocKnnFn<'a> = &'a dyn Fn(&str, u32) -> Vec<(NodeId, f32)>;

// ── #376 Phase 2 prototype: cross-encoder reranking of the docs lane ─────────
//
// `doc_lane_seeds` above gates confidence on raw embedding cosine, which is
// tied to whichever embedding backend is active (§8.3's DOC_FLOOR was
// measured against arctic-embed-m-v1.5 only — a different backend's cosine
// scale is unvalidated, the same failure class the code lane's `Calibration`
// exists to fix). The code lane's cross-encoder (RFC-021) sidesteps this
// entirely: its floors are pinned to the reranker MODEL, not the embedding
// backend, and there is exactly one reranker shared across every embedding
// backend — so a reranker-gated docs lane needs no per-embedding-model
// calibration. It also plausibly improves precision on genuinely prose-heavy
// text (an MS-MARCO passage ranker's home turf), unlike the code lane where
// query-prose-vs-code-body token mismatch is a known collapse mode.
//
// `doc_lane_candidates` fetches a larger, unfiltered pool; the caller
// (`tools::build_docs_section`) reranks it and falls back to `doc_floor`-style
// cosine filtering when the reranker is unavailable/disabled/breaker-open —
// same fail-open contract as the code lane's `crate::rerank::rerank`.

/// Candidate pool size for doc-lane reranking — the doc-corpus analogue of
/// `crate::rerank::rerank_topk()` (code lane, default 30). Doc corpora are
/// 2-4 orders of magnitude smaller than code corpora (plan §4.4), so a
/// smaller default keeps rerank cost trivial while still giving the
/// cross-encoder enough candidates to recover a hit that ranked below the
/// raw-cosine top 3.
pub(crate) fn doc_rerank_overfetch() -> usize {
    std::env::var("TRAVSR_DOC_RERANK_OVERFETCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(20)
}

/// Rerank-score floor for the docs lane (plan §14.1).
///
/// Deliberately **not** the code lane's WEAK floor. An earlier revision reused
/// `manifest_weak_floor()` / `DEFAULT_WEAK_FLOOR` (0.5) on the reasoning that a
/// floor is a property of the reranker model rather than of the content being
/// scored. The floor sweep in §14.1 disproved that: 0.5 was measured against
/// *code* candidates, and applied to doc prose it discarded gold hits for no
/// benefit. Measured negative-arm ceilings (every `nonsense`/`salad` query from
/// `queries-seeded-{repo}.json` run through this lane) were 0.0023 on travsr and
/// 0.0003 on k8s — the cross-encoder already separates unrelated prose from real
/// doc content almost perfectly on its own, so the floor only needs to clear
/// that noise ceiling, not to do the ranking.
///
/// [`DOC_RERANK_FLOOR`] gives ~20x margin over the measured ceiling. hit@k is
/// flat across roughly `[0.01, 0.26]` and decreases monotonically above it (a
/// floor gates presence, never rank, so raising it past the noise ceiling can
/// only ever remove gold hits). Live-verified at this value: travsr 1.0/1.0,
/// k8s 0.70/0.90 (§14.3).
///
/// Resolution is env → this default. It deliberately does **not** consult
/// `crate::rerank::manifest_weak_floor()`, which an earlier revision did:
/// `RerankManifest` (`travsr-rerank/src/manifest.rs`) carries only
/// `strong_floor`/`weak_floor`, both authored against *code* candidates, and
/// has no doc-specific field. Reading `weak_floor` here therefore resolved to
/// 0.5 on every machine with the reranker installed — i.e. every real user —
/// silently reinstating the borrowed floor regardless of this constant, and
/// making §14.3's live-verified numbers reachable only via the env override.
/// Add a `doc_weak_floor` field to the manifest before reintroducing a
/// manifest leg here.
///
/// **Model scope:** validated on `ms-marco-MiniLM-L-6-v2` only. A swapped
/// reranker can shift the negative-arm ceiling this value is calibrated
/// against (§13 item 2 measured `bge-reranker-base` at nonsense p95 0.0094 /
/// distractor p95 0.0393, against 0.0000 for both MS MARCO models), so a
/// non-default reranker must re-measure the negative arm before relying on it.
pub(crate) fn doc_rerank_floor() -> f32 {
    floor_env("TRAVSR_DOC_RERANK_FLOOR").unwrap_or(DOC_RERANK_FLOOR)
}

/// #520: ambiguity threshold for *selective* doc-lane reranking (bench
/// experiment, not a shipped default). `None` (the default — no env set)
/// means "always rerank", identical to shipped behavior today.
///
/// When `Some(t)`, `tools::rerank_doc_candidates` skips the cross-encoder
/// pass entirely whenever the top raw-cosine candidate already clears `t` —
/// the reasoning being that a confidently-separated top candidate doesn't
/// need the reranker's judgment, so the (measured 1.3-2.6x, #520) latency
/// cost of the second cross-encoder pass can be skipped for the "easy" case.
///
/// This is env-only on purpose: #520's own gate is that a threshold must
/// clear a strict non-regression bar on hit@1/hit@3 with zero per-query
/// hit-to-miss flips on *both* bench repos before it ships as a default —
/// see `bench/sweep-selective-rerank.mjs`. An earlier naive attempt at this
/// exact lever measurably lost accuracy (travsr 1.0/1.0 -> 0.9/0.9), so this
/// stays opt-in until a threshold is proven, not assumed, safe.
pub(crate) fn doc_rerank_ambiguity_threshold() -> Option<f32> {
    std::env::var("TRAVSR_DOC_RERANK_AMBIGUITY_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|&x| (0.0..=1.0).contains(&x))
}

/// Compiled default for [`doc_rerank_floor`]. See that function for the
/// measurement behind the value.
pub(crate) const DOC_RERANK_FLOOR: f32 = 0.05;

/// Raw candidate pool for the docs lane: same gating as [`doc_lane_seeds`]
/// (`docs_enabled` + hook presence) but **no cosine floor** and a larger `k`
/// — the confidence call belongs to the reranker downstream, not to raw
/// cosine, so filtering here would only throw away recall the reranker could
/// have recovered. Sorted by cosine descending purely so a reranker-unavailable
/// fallback still has a sane order to filter/cap from.
///
/// The query is normalized through [`doc_lane_query`] first — see that
/// function for why that is load-bearing rather than cosmetic.
pub(crate) fn doc_lane_candidates(
    store: &SqliteStore,
    query: &str,
    doc_knn: Option<DocKnnFn<'_>>,
) -> Vec<(NodeId, f32)> {
    if !docs_enabled(store) {
        return vec![];
    }
    let Some(knn) = doc_knn else {
        return vec![];
    };
    let k = doc_rerank_overfetch() as u32;
    let mut hits = knn(&doc_lane_query(query), k);
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

/// The text the docs lane sends to `knn` for **recall**.
///
/// Historically this was byte-identical to what `tools::embed_path_seeds`
/// sends for the code lane's own KNN call (both called `normalize_nl_query`),
/// to hit the sidecar's single-slot exact-text memo (`embed_query_cached`)
/// and save a second ~200-270ms inference — see #376 §4.4, "one inference,
/// two searches".
///
/// #539: that shared full-sentence text is also the root cause of doc-lane
/// recall failures on natural-language queries. A pooled/mean query
/// embedding drifts toward whatever generic prose region the corpus's
/// longer documents occupy as the ratio of filler words ("why", "was",
/// "did", "the") to content words rises, and the correct short technical doc
/// chunk can fall entirely outside the `doc_rerank_overfetch` candidate
/// window before the reranker ever sees it — a recall failure, not merely a
/// rerank-precision one. Embedding content tokens only fixes this because it
/// is exactly what `tokenize_query` already does for the code lane's
/// per-token anchor resolution (`build_seed_set`) — reused here rather than
/// duplicated.
///
/// This intentionally **breaks** the exact-text memo contract with the code
/// lane's `embed_path_seeds` call: the two lanes now send different text
/// (content tokens vs full normalized sentence) whenever the query has any
/// filler words, so a docs-enabled query pays the second inference on
/// purpose. Correctness (citing the right doc) outranks that latency cost —
/// the docs lane exists specifically for the low-confidence/abstention case
/// where a wrong or missing citation is the worse failure. Gate 4 in
/// `bench/run-phase2-gate.mjs` expects this specific divergence rather than
/// requiring single-text identity; `TRAVSR_EMBED_QUERY_CACHE_DEBUG` on the
/// sidecar still reports hit/miss per KNN if the memo rate needs rechecking.
///
/// Falls back to `normalize_nl_query` when `tokenize_query` strips every
/// token (an all-stopword query), then to the trimmed raw query itself when
/// even that is empty (an all-punctuation query — `normalize_nl_query` also
/// strips bare punctuation tokens, e.g. `"???"` normalizes to `""`), so the
/// KNN call is never sent an empty string for a non-blank input.
pub(crate) fn doc_lane_query(query: &str) -> String {
    let tokens = tokenize_query(query);
    if !tokens.is_empty() {
        return tokens.join(" ");
    }
    let normalized = travsr_store::fts_tokenize::normalize_nl_query(query);
    if !normalized.is_empty() {
        return normalized;
    }
    query.trim().to_string()
}

// ── Stop-word list (code-aware: omit "get", "set", "run", "use") ─────────────

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "of", "in", "to", "is", "for", "and", "or", "with", "that", "this", "how",
    "what", "where", "when", "which", "does", "are", "by", "my", "from", "at", "it", "its", "be",
    "as", "on", "so", "all", "me", "i", "you", "we", "they", "he", "she", "was", "were", "had",
    "has", "have", "been", "do", "did", "would", "could", "should",
];

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Extract content tokens from a query for per-token anchor resolution.
/// Also reused by [`doc_lane_query`] (#539) to build the doc-lane's KNN
/// embedding text — `STOP_WORDS` is now a shared dependency of both callers,
/// so retuning it for one lane's ranking affects the other's embeddings too.
///
/// - Strips leading/trailing ASCII punctuation (preserving `_` and `.`)
/// - Removes stop-words and tokens shorter than 2 characters (stop-word/length
///   checks are case-insensitive; the returned token keeps its original case)
///
/// Case is preserved deliberately (#478 fix): `symbol_frequency` and
/// `ident::contains_token` both call `ident::segments` on this token to find
/// its own internal word boundaries (e.g. `"isPrime"` → `["is","prime"]`).
/// That split relies on lower→upper transitions in the token itself — if this
/// function lowercased `"isPrime"` to `"isprime"` first, the boundary is gone
/// permanently and `segments("isprime")` returns one fused, unsplittable
/// segment that can never match the index's real `"prime"` vocabulary entry.
/// `symbol_frequency` then falls back to "absent from vocabulary" and the
/// token is wrongly treated as maximally generic, which can suppress a
/// perfectly unambiguous exact-name query down to total abstention on the
/// FTS-only path. `search_nodes_by_name`'s `LIKE` scan and
/// `ident::contains_token`'s own token-normalisation are already
/// case-insensitive, so returning the original case here is free for them.
pub(crate) fn tokenize_query(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter_map(|w| {
            let stripped =
                w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '_' && c != '.');
            if stripped.is_empty() {
                return None;
            }
            let lower = stripped.to_ascii_lowercase();
            if lower.len() < 2 || STOP_WORDS.contains(&lower.as_str()) {
                return None;
            }
            Some(stripped.to_string())
        })
        .collect()
}

// ── Intent classifier ─────────────────────────────────────────────────────────

/// Deterministic keyword-based intent classification. No LLM involved.
pub(crate) fn classify_intent(query: &str) -> QueryIntent {
    let lower = query.to_ascii_lowercase();
    if lower.contains("calls ")
        || lower.contains("callers")
        || lower.contains("who calls")
        || lower.contains("called by")
    {
        return QueryIntent::Callers;
    }
    if lower.contains("import")
        || lower.contains("depends")
        || lower.contains("dependency")
        || lower.contains("dependencies")
    {
        return QueryIntent::Deps;
    }
    let first_word = lower.split_whitespace().next().unwrap_or("");
    if first_word == "how" || lower.contains("explain") || lower.contains("describe") {
        return QueryIntent::Explain;
    }
    if lower.contains("between") || lower.contains("relationship") || lower.contains("relation") {
        return QueryIntent::Relation;
    }
    QueryIntent::Lookup
}

// ── IDF weight ────────────────────────────────────────────────────────────────

/// IDF-inspired weight for a token that matches `freq` out of `n_total` symbols.
///
/// Returns values in [0.05, 1.0]:
/// - freq=1,   n=10000 → ~0.98  (very specific anchor, near-max weight)
/// - freq=100, n=10000 → ~0.79
/// - freq=1000,n=10000 → ~0.50
/// - freq≥n   (stop-symbol) → floor 0.05
///
/// Formula: clamp( ln((N+1)/(f+1)) / ln(N+1), 0.05, 1.0 )
pub(crate) fn idf_weight(freq: usize, n_total: usize) -> f32 {
    if n_total == 0 {
        return 0.05;
    }
    let num = (n_total + 1) as f32;
    let den = (freq + 1) as f32;
    ((num / den).ln() / num.ln()).clamp(0.05, 1.0)
}

// ── RRF fusion ────────────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion — combines multiple ranked lists into a single score.
///
/// `score(node) = Σ_source  1 / (k + rank_in_source(node))`
///
/// - Scale-free: works regardless of score magnitude per source.
/// - Handles missing sources gracefully (no term = no contribution, not a penalty).
/// - Deterministic: ties broken by NodeId ascending.
///
/// `sources` must each be sorted descending by score (position 0 = best).
pub(crate) fn rrf_fuse(sources: &[&[(NodeId, f32)]], k: f32) -> Vec<(NodeId, f32)> {
    let mut scores: HashMap<NodeId, f32> = HashMap::new();

    for source in sources {
        for (rank, &(node_id, _score)) in source.iter().enumerate() {
            // rank 0 = best → contribution 1/(k+1); rank 1 → 1/(k+2); etc.
            *scores.entry(node_id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }

    let mut result: Vec<(NodeId, f32)> = scores.into_iter().collect();
    // Deterministic: sort desc by RRF score, break ties by NodeId ascending.
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    result
}

/// RFC-022 D1.3/D1.4: weighted Reciprocal Rank Fusion. Identical to [`rrf_fuse`] but
/// each source carries a weight scaling its per-rank contribution, so the sparse
/// BM25 and dense embed legs stay first-class (D1.3) while low-IDF per-token exact
/// anchors — a precision source that otherwise floods the fused top-N with short
/// single-token matches (`var:query`, `fn:set`) and buries the compound symbol
/// (`build_seed_set`) — are capped below them (D1.4). Multi-source reward is
/// preserved: a candidate in several sources still accumulates their summed
/// contributions. `sources` = `(weight, ranked_list)`; each list sorted desc by score.
pub(crate) fn rrf_fuse_weighted(sources: &[(f32, &[(NodeId, f32)])], k: f32) -> Vec<(NodeId, f32)> {
    let mut scores: HashMap<NodeId, f32> = HashMap::new();
    for (weight, source) in sources {
        for (rank, &(node_id, _score)) in source.iter().enumerate() {
            *scores.entry(node_id).or_insert(0.0) += weight / (k + rank as f32 + 1.0);
        }
    }
    let mut result: Vec<(NodeId, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    result
}

/// Whether RFC-022 D1.3/D1.4 weighted seed fusion is active. **Default OFF** (part of
/// the D1 recall phase, pending calibration); enable with `TRAVSR_RRF_WEIGHTED=1`.
fn rrf_weighted_enabled() -> bool {
    matches!(
        std::env::var("TRAVSR_RRF_WEIGHTED").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Per-source RRF weights for [`rrf_fuse_weighted`] (RFC-022 D1.3/D1.4). Specific
/// (high-IDF) exact anchors lead; BM25 + embed are first-class; generic (mid-IDF)
/// per-token anchors are capped so they cannot crowd out the recall legs.
fn rrf_source_weights() -> (f32, f32, f32, f32) {
    let g = |name: &str, d: f32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|x: &f32| *x >= 0.0)
            .unwrap_or(d)
    };
    (
        g("TRAVSR_RRF_W_SPECIFIC_ANCHOR", 2.0),
        g("TRAVSR_RRF_W_BM25", 1.0),
        g("TRAVSR_RRF_W_EMBED", 1.0),
        g("TRAVSR_RRF_W_GENERIC_ANCHOR", 0.4),
    )
}

// ── Confidence classifier ─────────────────────────────────────────────────────

/// Today's cosine/coverage lattice, extracted verbatim (RFC-021 Phase 3). Used
/// as the confidence path when the reranker has no opinion for this query
/// (`rerank_score: None` — model absent, disabled, panicked, or degraded) so
/// the reranker-off surface stays byte-for-byte identical to pre-RFC-021
/// behaviour. Also the path for exact-symbol queries (G1): a literal named
/// symbol is judged deterministically here, never by the NL-trained reranker.
#[allow(clippy::too_many_arguments)]
fn classify_confidence_lexical_fallback(
    terms: &[ResolvedTerm],
    coverage: f32,
    top_bm25: f32,
    has_any_seeds: bool,
    #[allow(unused_variables)] has_knn_seeds: bool,
    // Cluster oracle = the KNN nearest-neighbour cosines ONLY. Drives the
    // cluster-level signals (oracle_top, semantic promote/veto) that were
    // calibrated on neighbour density — must NOT be polluted with anchor
    // self-cosines, which would spuriously inflate oracle_top on symbol queries.
    knn_oracle: &HashMap<NodeId, f32>,
    // RFC-019: per-anchor cosines (KNN neighbours ∪ directly-scored anchors). Used
    // only for the per-anchor `embed_agrees` / `embed_confirmed_specific` lookups.
    anchor_oracle: &HashMap<NodeId, f32>,
    scored_ids: &std::collections::HashSet<NodeId>,
    // Model-relative floor calibration. All cosine thresholds below are authored in
    // the bge-small reference scale and mapped into the active model's scale via
    // `cal`; `Calibration::IDENTITY` reproduces the original absolute floors.
    cal: &Calibration,
) -> Confidence {
    let rare_max = rare_anchor_max();
    let cov_strong = coverage_strong();
    let cov_weak = coverage_weak();
    let bm25_floor = bm25_strong_floor();
    let idf_min = idf_coverage_min();

    // Embed oracle as a confidence check: when the embedding is confident, a
    // lexical anchor only counts toward confidence if the model agrees it is near
    // the query. This stops NL query words that are rare-by-coincidence ("literal"
    // → 1 node, "matching" → 3) from claiming Exact/Strong over a semantic salad.
    // When the oracle is absent or weak we have no embedding opinion and trust
    // lexical evidence unchanged — the FTS-only path is identical to before.
    let oracle_top = knn_oracle.values().copied().fold(0.0_f32, f32::max);
    let oracle_confident = oracle_top >= cal.map(SEMANTIC_ORACLE_MIN);
    let floor = (oracle_top - cal.map_delta(SEMANTIC_REL_DELTA)).max(cal.map(SEMANTIC_ABS_FLOOR));
    // RFC-019: absence is no longer disagreement. With the direct-cosine oracle we
    // MEASURE every specific anchor; an anchor's node that is absent from the oracle
    // *after* we tried to score it (`scored_ids`) is genuinely unscoreable
    // (no stored vector / degraded) → "unknown", not "the model rejects it" → we
    // fall back to lexical evidence (agrees). Only an anchor that is present and
    // below floor counts as disagreement. When no score hook ran (`scored_ids`
    // empty — the FTS-only path), the old `unwrap_or(0.0)` semantics are preserved
    // exactly, so behaviour is byte-for-byte identical without embeddings.
    let embed_agrees = |t: &ResolvedTerm| -> bool {
        if !oracle_confident {
            return true;
        }
        match t.top_node {
            Some(n) => match anchor_oracle.get(&n).copied() {
                Some(c) => c >= floor,                   // measured → trust the model
                None if scored_ids.contains(&n) => true, // scored but unscoreable → unknown
                None => false,                           // never scored → old absent⇒disagree
            },
            None => false,
        }
    };

    // Only count anchors whose IDF weight is high enough to be meaningful signals
    // (generic tokens like "map"/"get" match hundreds of nodes), AND — when the
    // oracle is confident — that the embedding agrees are relevant. This drops
    // the rare-by-coincidence NL anchor ("literal" → c1_raw_literal) that drove
    // the false Exact, while keeping a genuinely named symbol ("build_seed_set").
    let has_rare_anchor = terms
        .iter()
        .any(|t| t.resolved && t.symbol_freq <= rare_max && embed_agrees(t));
    let has_specific_anchor = terms
        .iter()
        .any(|t| t.resolved && t.idf_w >= idf_min && embed_agrees(t));

    // Strict embedding confirmation: the oracle is confident AND actively places a
    // *specific* anchor near the query (present in the over-fetch with cos ≥ floor).
    // Unlike `embed_agrees`, this is NOT vacuously true when the oracle is cold/absent
    // — it is positive evidence used to rescue a query whose lexical coverage is
    // diluted by typos or generic filler but whose anchor the model confirms.
    //
    // RFC-019: with the direct-cosine oracle every specific anchor is now MEASURED, so
    // a single query-word that is also a symbol name ("semantic" in "find code by
    // semantic meaning") scores a real, lexical-overlap-inflated cosine (~0.615) that
    // clears the weak `floor` (bottoms out at ABS_FLOOR 0.55) and would falsely confirm
    // a salad. A LOW-coverage query is therefore only confirmed when its anchor sits in
    // the *answerable* band (`confirm_anchor_floor`, ~0.66) — which a genuine
    // low-coverage query clears ("Load mutationg manifests" → "manifests" ≈0.71) but
    // the salad word does not. High-coverage exact queries are unaffected: they pass
    // through `has_specific_anchor` (coverage ≥ cov_strong) regardless.
    let confirm_floor = floor.max(cal.map(confirm_anchor_floor()));
    let embed_confirmed_specific = oracle_confident
        && terms.iter().any(|t| {
            t.resolved
                && t.idf_w >= idf_min
                && t.top_node
                    .and_then(|n| anchor_oracle.get(&n).copied())
                    .is_some_and(|c| c >= confirm_floor)
        });

    // Fix 1+3a: semantic grounding. A confident oracle whose neighbours form a
    // coherent near cluster (≥ N candidates within the floor band) is a Strong
    // match even with NO lexical anchor. This is the conceptual-query path —
    // "how does the kubelet sync pod status", every token too common to be a
    // specific anchor on a 261 k-node corpus — that the lexical gates can't reach.
    // `oracle_top` separates answerable (≥ 0.77) from nonsense (≤ 0.64); the
    // promote threshold sits in that gap, so salads never reach Strong here.
    let oracle_present = !knn_oracle.is_empty();
    let n_oracle_near = knn_oracle.values().filter(|&&c| c >= floor).count();
    let semantic_strong = has_any_seeds
        && oracle_top >= cal.map(semantic_promote_strong())
        && n_oracle_near >= semantic_promote_min_near();

    // Fix 1c: semantic veto. Embeddings ran and are confident nothing in the
    // corpus is near the query (oracle_top below the veto floor). A Weak that
    // rests on coincidental lexical coverage ("the cat sat on the warm
    // windowsill" → 0.49) is then a false positive → abstain. A rare exact anchor
    // is exempt: a precise symbol the user literally typed is honoured even when
    // the model is weak on it.
    let semantic_veto =
        oracle_present && oracle_top < cal.map(semantic_veto_floor()) && !has_rare_anchor;

    if has_rare_anchor && coverage >= cov_strong {
        Confidence::Exact
    } else if has_specific_anchor && coverage >= cov_strong {
        // A specific (high-IDF) anchor the embedding AGREES with — or that has no
        // embedding opinion to contradict it — at high coverage is a Strong match.
        // e.g. "type Tweak" → type:Tweak (idf ~0.81, cosine ~1.0): not rare enough
        // for Exact (the symbol recurs across packages) but unmistakably the right
        // anchor. `embed_agrees` is baked into `has_specific_anchor`, so a confident
        // oracle that DISAGREES (the NL-salad case — "literal"/"semantic" matching a
        // symbol by coincidence) never reaches here and correctly stays Weak below.
        Confidence::Strong
    } else if embed_confirmed_specific && coverage >= cov_weak {
        // Embedding-confirmed specific anchor at low token-coverage. The query's other
        // tokens are typos or generic filler (e.g. "Load mutationg manifests" → coverage
        // 1/3: "mutationg" is a typo, "load" is generic), but the confident oracle places
        // the specific anchor ("manifests", cos 0.71 ≥ floor 0.643) near the query — a
        // real match, Strong not Weak. Requires a CONFIDENT oracle that AGREES, so salad
        // queries whose anchor is absent from the over-fetch (cos < floor) never reach here.
        Confidence::Strong
    } else if semantic_strong {
        // Fix 1+3a: oracle-grounded Strong (no lexical anchor required). See the
        // `semantic_strong` derivation above — gated on oracle_top in the
        // answerable band plus a coherent near cluster, so nonsense (low
        // oracle_top) and flukes (sparse cluster) never reach here.
        Confidence::Strong
    } else if !oracle_confident && top_bm25 >= bm25_floor && coverage >= cov_strong {
        // Pure-lexical Strong is only honest when there is NO embedding opinion.
        // When the oracle IS confident yet no embed-agreed *specific* anchor exists,
        // the high BM25/coverage came from NL words matching symbols by
        // coincidence — the result is embedding-driven, which is Weak-grade.
        Confidence::Strong
    } else if semantic_veto {
        // Fix 1c: embeddings are confident nothing matches — abstain rather than
        // surface coincidental lexical coverage. Reaches here only after the
        // lexical-Strong path above declined, so genuinely strong lexical evidence
        // (high coverage + BM25) still survives a weak oracle.
        Confidence::None
    } else if has_any_seeds && (coverage >= cov_weak || has_specific_anchor) {
        // KNN seeds enhance grounded queries but never substitute for lexical evidence.
        // has_knn_seeds alone (zero coverage, no anchor) → abstain; KNN always returns K
        // neighbours regardless of semantic distance, so it cannot signal "no match".
        Confidence::Weak
    } else {
        Confidence::None
    }
}

/// Parse a floor env var into a valid `[0, 1]` probability, or `None` if unset
/// or out of range. Shared by both floor accessors.
fn floor_env(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=1.0).contains(&x))
}

/// Absolute floor at/above which the reranker's opinion is a Strong match.
///
/// Resolution order (RFC-021 F8): explicit env override → the model's
/// `model.toml` manifest → the compiled-in `travsr_rerank::DEFAULT_STRONG_FLOOR`.
/// The manifest ships next to `model_fp16.onnx` so model + floors bump
/// atomically; `TRAVSR_RERANK_STRONG_FLOOR` stays the top-priority tuning
/// override (re-runnable per model via `rerank_floor_sweep`, per the RFC).
/// Floors were measured 2026-07-17 (travsr) and re-validated 2026-07-22 on
/// kubernetes — see RFC-021 §13.2 / §9.8.
fn rerank_strong_floor() -> f32 {
    floor_env("TRAVSR_RERANK_STRONG_FLOOR")
        .or_else(|| crate::rerank::manifest_strong_floor().filter(|&x| (0.0..=1.0).contains(&x)))
        .unwrap_or(travsr_rerank::DEFAULT_STRONG_FLOOR)
}

/// Absolute floor at/above which the reranker's opinion is at least a Weak
/// match; below it the query abstains (`Confidence::None`) rather than
/// surfacing a confident salad. Same resolution order as
/// [`rerank_strong_floor`]; override via `TRAVSR_RERANK_WEAK_FLOOR`.
fn rerank_weak_floor() -> f32 {
    floor_env("TRAVSR_RERANK_WEAK_FLOOR")
        .or_else(|| crate::rerank::manifest_weak_floor().filter(|&x| (0.0..=1.0).contains(&x)))
        .unwrap_or(travsr_rerank::DEFAULT_WEAK_FLOOR)
}

// ── #462 WS4: anchor-gated rerank recall-floor rescue ─────────────────────────
//
// The RFC-021 cross-encoder (an MS MARCO web-passage ranker) systematically
// COLLAPSES on natural "how/what is X …" questions about a symbol: the prose
// scaffolding shares no tokens with X's code body, so the model scores the correct
// definition near zero even when a strong LEXICAL anchor names X. Measured on this
// repo (arctic + ms-marco): `"daemon"` scores struct:Daemon 0.92, but
// `"how daemon is implemented"` / `"what does the daemon do"` score every daemon
// definition 0.09–0.35 — below the 0.5 Weak floor — so the query abstains despite a
// genuine exact anchor. Neither the reranker nor the WS2 embedding-oracle rescue
// recovers it (both under-score prose→code); the lexical anchor is the only signal
// that stays correct.
//
// WS4 is the LEXICAL sibling of the WS2 embedding-oracle rescue: when an otherwise
// `None` query is grounded by a genuine exact anchor AND its best rerank score sits
// far above the salad-noise band (even though below the Weak floor), surface it as
// Weak instead of discarding it. Salad-safe by measurement: salad/nonsense queries
// rerank at ≤ ~1e-3 (`"delete all user accounts and drop the database"` = 0.0012,
// `"what is the meaning of life"` = 3e-5), ~100x below the recall floor, while the
// grounded prose queries sit ~2x above it. Embeddings-independent (reads only the
// rerank score + lexical anchors, never a cosine), so the G2 invariant holds and no
// calibration gating is needed — unlike WS2. On by default; `TRAVSR_ANCHOR_RESCUE=0`
// restores the pre-WS4 abstention byte-for-byte.

/// Whether the #462 WS4 anchor-gated rerank recall-floor rescue is active. On by
/// default; disabled only by an explicit falsey env value (`0` / `false` / `off`).
fn anchor_rescue_enabled() -> bool {
    !matches!(
        std::env::var("TRAVSR_ANCHOR_RESCUE").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Rerank recall floor for the #462 WS4 rescue: a cross-encoder score at/above this
/// — even though below the Weak floor — is treated as a real (if approximate) hit
/// when a genuine exact anchor grounds the query. Default 0.15 sits ~100x above the
/// measured salad/nonsense rerank ceiling (~1e-3) and ~2x below the measured
/// grounded-prose band (~0.30). Override via `TRAVSR_RERANK_RECALL_FLOOR`.
fn rerank_recall_floor() -> f32 {
    floor_env("TRAVSR_RERANK_RECALL_FLOOR").unwrap_or(0.15)
}

/// #462 WS4: apply the anchor-gated rerank recall-floor rescue to a base verdict.
///
/// Rescues `None → Weak` iff **all** hold: the base verdict abstained; the query is
/// not a `g1_bypass` deterministic-path query; the query has non-zero grounded
/// coverage (`coverage_ok` — at least one resolved token clears the IDF-coverage
/// bar); a genuine exact lexical anchor grounds the query (`exact_anchor_present`);
/// and the best rerank score clears `recall_floor`. Never promotes to Strong, and
/// never touches a non-`None` verdict. Pure, so it is unit-testable without a store.
///
/// RFC-022 D5 (RC-5): `coverage_ok` closes the WS4 over-rescue. Generic tokens
/// (`get`/`map`/`handle`) still emit an exact anchor (`idf_w >= 0.15`) yet count
/// zero toward coverage (`idf_w < idf_coverage_min`), so `exact_anchor_present`
/// alone let a `coverage=0.000` query be rescued. Requiring `coverage_ok`
/// (`n_resolved >= 1`) means `get map handle` abstains while W1/W2
/// (`coverage` 0.5–1.0) still rescue.
fn anchor_rescued_confidence(
    base: Confidence,
    g1_bypass: bool,
    max_rerank: Option<f32>,
    exact_anchor_present: bool,
    coverage_ok: bool,
    recall_floor: f32,
) -> Confidence {
    if base == Confidence::None
        && !g1_bypass
        && coverage_ok
        && exact_anchor_present
        && max_rerank.is_some_and(|r| r >= recall_floor)
    {
        Confidence::Weak
    } else {
        base
    }
}

// ── RFC-022 D2: rerank-weighted PPR personalisation (RC-2, keystone) ──────────
//
// The cross-encoder scores the fused seeds but PPR personalisation used only the
// idf/bm25/cosine `weight`, so a correct #1 seed the reranker loved (`impl:Daemon`,
// rr 0.666 for "how does the daemon work") was out-voted in the walk by a cluster of
// near-zero-rerank "work" junk (`worktree_resolves_to_main_worktree`, `LangWork`,
// `WorkItem`) and vanished from the rows. Folding the reranker's judgment into each
// NON-exact seed's teleportation weight makes the walk concentrate mass where the
// model saw relevance. Exact anchors keep their structural float untouched (they are
// the literally-named symbol); only the non-exact cluster is scaled. Embeddings-
// independent (reads the cross-encoder score, never a cosine) so the G2 invariant
// holds; inert on g1_bypass queries (rerank is skipped there) and when the reranker
// is absent (`rerank_score: None` → weight unchanged). `TRAVSR_RERANK_PPR_WEIGHT=0`
// restores the pre-D2 weighting byte-for-byte.

/// Whether RFC-022 D2 rerank-weighted PPR is active. On by default; disabled only by
/// an explicit falsey env value (`0` / `false` / `off`).
fn rerank_ppr_weight_enabled() -> bool {
    !matches!(
        std::env::var("TRAVSR_RERANK_PPR_WEIGHT").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Map a cross-encoder rerank score to a PPR teleportation-weight multiplier
/// (RFC-022 D2). Piecewise-linear and monotone: a near-zero score (model judged the
/// seed junk) crushes its mass toward `FLOOR`; a score at the weak floor is neutral
/// (×1.0); a strong score up-weights toward `CEIL`. Bounded so it re-weights the
/// walk without letting a single seed dominate. Pure — unit-testable.
fn rerank_ppr_multiplier(rerank: f32, weak_floor: f32) -> f32 {
    const FLOOR: f32 = 0.1;
    const CEIL: f32 = 2.0;
    let wf = weak_floor.clamp(1e-3, 0.999);
    let m = if rerank <= wf {
        FLOOR + (1.0 - FLOOR) * (rerank / wf).clamp(0.0, 1.0)
    } else {
        1.0 + (CEIL - 1.0) * ((rerank - wf) / (1.0 - wf)).clamp(0.0, 1.0)
    };
    m.clamp(FLOOR, CEIL)
}

// ── RFC-022 (prototype): doc-aware reranker candidate text ────────────────────
//
// The cross-encoder candidate is `signature + docblock-stripped body` — the leading
// doc comment (the natural-language description a conceptual query actually matches,
// e.g. "Personalized PageRank …") is the one thing NOT fed to the model, which is a
// direct contributor to its prose→code collapse. This prototype prepends the node's
// doc prose (from `embed_text`) so the model sees that NL surface. Default OFF
// (`TRAVSR_RERANK_DOC=1`); a no-op when `embed_text` is unpopulated, so the reference
// path is byte-for-byte unchanged.

/// Whether the doc-aware reranker candidate prototype is active. Default OFF.
fn rerank_doc_enabled() -> bool {
    matches!(
        std::env::var("TRAVSR_RERANK_DOC").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Extract the natural-language doc prose from an `embed_text` skeleton
/// (`… | doc: <prose>`). Returns the prose (trimmed), or empty when there is no
/// `doc:` segment (the node has no doc comment).
fn rerank_doc_text(embed_text: &str) -> String {
    match embed_text.split_once("| doc:") {
        Some((_, doc)) => doc.trim().to_string(),
        None => String::new(),
    }
}

/// G1 bypass decision (RFC-021 F2/F3): true when the user named a genuinely
/// *rare* symbol — an exact anchor exists **and** the specific term that
/// produced that anchor (`ResolvedTerm.top_node`) is one of `exact_anchor_ids`
/// **and** is rare enough (`symbol_freq <= rare_anchor_max()`) to be a real,
/// specific identifier rather than a coincidence. Tying rarity to the
/// anchor-producing term (not just any resolved term in the query) stops a
/// salad query with one incidental rare resolved term plus an exact anchor on
/// an unrelated common word from also bypassing. Single source of truth: the
/// caller (`build_seed_set`) uses this both to decide whether to run rerank
/// inference at all (unused when bypassing) and to gate
/// [`classify_confidence`].
fn compute_g1_bypass(terms: &[ResolvedTerm], anchors: &TrustedAnchors<'_>) -> bool {
    g1_bypass_decision(terms, anchors, rare_anchor_max(), g1_subject_idf_share())
}

/// The anchor-provenance sets the G1 decision trusts, grouped rather than
/// threaded as separate positional parameters.
///
/// Each provenance is a named field, so adding a third one (alias or synonym
/// resolution, say) is a field rather than another positional `HashSet` to
/// re-thread through every call site. There is deliberately no `Default` and no
/// constructor that fills a set in for you: a new field then fails to compile at
/// every construction site until it has been considered, where positional
/// parameters would happily let a site keep passing an empty set and silently
/// stop trusting a whole class of anchor.
struct TrustedAnchors<'a> {
    /// Nodes emitted as exact lexical anchors.
    exact: &'a std::collections::HashSet<NodeId>,
    /// Nodes emitted by a #709 unique strict typo correction.
    corrected: &'a std::collections::HashSet<NodeId>,
}

/// Pure core of [`compute_g1_bypass`] (RFC-021 G1 + RFC-022 D4). Split out so the
/// subject-dominance gate is unit-testable without env or a store.
///
/// A term is a *rare exact anchor* when it resolved to a genuinely rare symbol
/// (`symbol_freq <= rare_max`) whose top node is one of the emitted exact anchors.
/// The deterministic bypass fires only when (a) at least one rare exact anchor
/// exists **and** (b) the rare exact anchors *collectively* carry at least
/// `subject_share` of the query's total IDF mass — i.e. the named rare symbol(s)
/// are the query's subject, not incidental rare words.
///
/// This keeps genuine literal lookups intact — a single rare token, or an
/// all-symbol query like `SqliteStore search_nodes_by_name`, puts ~all IDF in the
/// anchors (share ≈ 1.0) — while a conceptual query that merely *contains* a rare
/// word (`how are results trimmed to fit a token budget`) spreads IDF across many
/// non-anchor tokens, dropping the anchor share below the bar so the query flows to
/// the reranker/recall path instead of a false deterministic Exact (RC-4).
fn g1_bypass_decision(
    terms: &[ResolvedTerm],
    anchors: &TrustedAnchors<'_>,
    rare_max: usize,
    subject_share: f32,
) -> bool {
    // #709: a unique strict typo correction is deterministic lexical evidence, so
    // its emitted anchor counts as a rare (subject) anchor regardless of the
    // corrected symbol's own frequency, the query literally names that one
    // symbol, just misspelt. Still gated by the same subject-share test below, so
    // a correction that is only an incidental fragment of a larger query does not
    // trigger the deterministic bypass.
    let is_rare_anchor = |t: &ResolvedTerm| {
        t.resolved
            && t.top_node.is_some_and(|n| {
                anchors.exact.contains(&n)
                    && (t.symbol_freq <= rare_max || anchors.corrected.contains(&n))
            })
    };
    if !terms.iter().any(&is_rare_anchor) {
        return false;
    }
    let total_idf: f32 = terms.iter().map(|t| t.idf_w).sum();
    if total_idf <= 0.0 {
        return false;
    }
    let anchor_idf: f32 = terms
        .iter()
        .map(|t| if is_rare_anchor(t) { t.idf_w } else { 0.0 })
        .sum();
    anchor_idf >= subject_share * total_idf
}

/// RFC-021 F4: four-band seed ordering after rerank. A rank-31+ seed that was
/// never judged by the reranker must not sink below a reranked seed the model
/// scored near zero (judged garbage) — only below seeds it approved.
///
/// Band 0: exact anchors (structural priority, as before RFC-021).
/// Band 1: reranked, at/above the weak floor (model-approved, best first).
/// Band 2: unreranked tail (never judged): keeps RRF order via sort stability.
/// Band 3: reranked, below the weak floor (model-rejected: judged garbage
///         must not outrank seeds that were never judged).
///
/// `sort_by` is stable, so within band 2 every key is equal and the pre-sort
/// (RRF) relative order is preserved for free.
fn sort_seeds_post_rerank(seeds: &mut [Seed], weak_floor: f32) {
    let band = |s: &Seed| -> u8 {
        if s.source == SeedSource::Exact {
            0
        } else {
            match s.rerank_score {
                Some(r) if r >= weak_floor => 1,
                None => 2,
                Some(_) => 3,
            }
        }
    };
    seeds.sort_by(|a, b| {
        band(a).cmp(&band(b)).then_with(|| {
            b.rerank_score
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.rerank_score.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}

/// RFC-021 Phase 3: the reranker's absolute score is the sole seed-selection /
/// abstention signal for NL queries. Two guards preserve the deterministic
/// paths the reranker must never arbitrate:
///
/// - **G1 (exact-symbol bypass):** when the user named a genuinely *rare*
///   symbol — an exact anchor exists **and** the specific term that produced
///   that anchor is rare enough — `symbol_freq <= rare_anchor_max()` — to be
///   a real, specific identifier, not a coincidence — confidence is judged
///   deterministically by [`classify_confidence_lexical_fallback`] regardless
///   of the rerank score. The cross-encoder is NL-trained and would
///   false-abstain on a literal identifier query (e.g. `GetWarningsForPod`
///   scores low against its own name, and is rare: `symbol_freq` 1) —
///   trading salad for missed precise lookups.
///
///   Rarity, not mere coverage, is the load-bearing check: `"delete all user
///   accounts and drop the database"` (the RFC's own reproduction query)
///   resolves an exact anchor too — `drop`/`delete` coincidentally name
///   `SqliteStore.drop`/`delete_file` — and coverage alone measured 3/5 =
///   0.6, clearing the same `coverage_strong()` bar a genuine hit would (a
///   coverage-only gate was tried and still let this exact query bypass to
///   `Confidence::Strong` — Defect A reproduced through the G1 escape
///   hatch). What actually separates the two cases is that `drop` is
///   anything but rare — dozens of types implement `Drop` in this repo, so
///   its `symbol_freq` is large — while `GetWarningsForPod` is unique.
///   Requiring rarity is also embeddings-independent (G2): it reads only
///   `ResolvedTerm.symbol_freq`, a structural fact from the graph, never an
///   oracle/cosine.
///
///   Rarity is tied to the anchor-producing term itself, not just any resolved
///   term in the query, via [`compute_g1_bypass`] — see its doc for why. The
///   caller (`build_seed_set`) computes this once and passes it in as
///   `g1_bypass`, also using it to skip rerank inference entirely on bypassed
///   queries (RFC-021 F2), since the model's opinion is unused either way.
/// - **Reranker unavailable (`rerank_score: None`):** model absent, disabled
///   (`TRAVSR_NO_RERANK`), load failed, panicked, or skipped because the circuit
///   breaker is open — fall back to the identical pre-RFC-021 gate. No
///   regression, no partial state. An over-budget call is no longer in this list:
///   it keeps its scores and only feeds the breaker (see `rerank::rerank`).
///
/// Otherwise the four-arm absolute-floor gate is embeddings-independent by
/// construction (embeddings remain a candidate *source* into RRF fusion, never
/// a confidence input here) — G2: confidence is identical embed-on/off for
/// the same query. One deliberate, bounded exception (#462 WS2): on a
/// *calibrated* model a non-`g1_bypass` query that would abstain is rescued to
/// Weak when the embed-KNN top clears the model-relative recall floor. G2 still
/// holds exactly on identity/bge and on every `g1_bypass` query, where that
/// rescue is inert by construction.
#[allow(clippy::too_many_arguments)]
fn classify_confidence(
    terms: &[ResolvedTerm],
    coverage: f32,
    top_bm25: f32,
    has_any_seeds: bool,
    has_knn_seeds: bool,
    knn_oracle: &HashMap<NodeId, f32>,
    anchor_oracle: &HashMap<NodeId, f32>,
    scored_ids: &std::collections::HashSet<NodeId>,
    cal: &Calibration,
    g1_bypass: bool,
    rerank_score: Option<f32>,
) -> Confidence {
    let base = if g1_bypass {
        classify_confidence_lexical_fallback(
            terms,
            coverage,
            top_bm25,
            has_any_seeds,
            has_knn_seeds,
            knn_oracle,
            anchor_oracle,
            scored_ids,
            cal,
        )
    } else {
        // Guard against inverted floors (strong < weak) from a hand-broken manifest
        // or env override: a Strong verdict must never fire below the Weak/abstain
        // floor, so clamp strong up to at least weak. With correctly-ordered floors
        // this is a no-op.
        let weak = rerank_weak_floor();
        let strong = rerank_strong_floor().max(weak);
        match rerank_score {
            Some(r) if r >= strong => Confidence::Strong,
            Some(r) if r >= weak => Confidence::Weak,
            Some(_) => Confidence::None,
            None => classify_confidence_lexical_fallback(
                terms,
                coverage,
                top_bm25,
                has_any_seeds,
                has_knn_seeds,
                knn_oracle,
                anchor_oracle,
                scored_ids,
                cal,
            ),
        }
    };

    // #462 WS2 — calibrated abstain rescue. The NL-trained cross-encoder
    // systematically UNDER-scores conceptual CODE matches (a query described in
    // prose vs. an identifier-dense function body), so a query whose gold sits at
    // embed-KNN rank #1 with a healthy cosine can still fall below the rerank weak
    // floor and abstain (`Confidence::None`). When the embedding itself is confident
    // — its top neighbour clears the model-relative recall floor — do not throw the
    // result away; surface it as Weak (labelled "no strong match" downstream) so the
    // AI still gets the semantic hit, flagged as approximate. Salad-safe by
    // construction: off-domain top cosines sit BELOW the recall floor (the floor is
    // anchored just above the measured nonsense/salad p95), so this never rescues a
    // salad. Calibrated-only and never on a `g1_bypass` query: the rescue reads an
    // embedding signal, so gating it to calibrated models keeps identity/bge
    // byte-for-byte (G2 holds there exactly), and skipping `g1_bypass` preserves G1's
    // deterministic, embeddings-independent lexical verdict — an embedding must never
    // override the rare-named-symbol decision G1 exists to protect.
    if base == Confidence::None && !g1_bypass && *cal != Calibration::IDENTITY {
        let oracle_top = knn_oracle.values().copied().fold(0.0_f32, f32::max);
        if oracle_top >= semantic_recall_floor(cal) {
            return Confidence::Weak;
        }
    }
    base
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// First two slash-delimited path segments, e.g. "crates/travsr-retrieval" from
/// "crates/travsr-retrieval/src/ppr.rs".  Returns `None` for flat or single-segment paths.
pub(crate) fn package_root(path: &str) -> Option<&str> {
    let mut slashes = 0u8;
    for (i, c) in path.char_indices() {
        if c == '/' {
            slashes += 1;
            if slashes == 2 {
                return Some(&path[..i]);
            }
        }
    }
    None
}

// ── Tier-1 semantic validation thresholds ─────────────────────────────────────

/// Minimum top cosine for the embed oracle to be trusted at all.  Below this the
/// embedding found nothing strongly relevant, so lexical seeds are left untouched
/// (avoids cutting good FTS results on queries the model is weak on).
const SEMANTIC_ORACLE_MIN: f32 = 0.58;
/// Multiplier applied to the strongest non-exact seed weight to set the floor for
/// exact-anchor seeds — guarantees a literal symbol match anchors the PPR walk above
/// any KNN neighbour whose cosine×kind_boost would otherwise dominate.
const EXACT_SEED_PRIORITY: f32 = 1.5;
/// Keep seeds whose cosine is within this band of the top oracle score …
const SEMANTIC_REL_DELTA: f32 = 0.13;
/// … but never below this absolute cosine floor.
const SEMANTIC_ABS_FLOOR: f32 = 0.55;
/// Never cut below this many seeds — a grounded result must not be emptied by
/// validation even when the oracle is harsh.
const SEMANTIC_MIN_KEEP: usize = 5;

/// #393 score-aware scope gate. A lexical seed whose package root is outside the
/// anchor crate scope is admitted when its per-batch normalised score reaches
/// this floor. Rationale: strong lexical evidence (the query term directly names
/// a symbol or a path segment in another crate) is real relevance, not cross-
/// domain drift, so it overrides the structural scope. Weak FTS trigram drift
/// (e.g. "works" → "workspace") scores far below the floor and stays gated,
/// preserving the anti-"confident salad" protection. Generic: match *strength*
/// decides — no per-term string, plural, or stemming heuristics.
///
/// Default 0.3; override via `TRAVSR_SCOPE_STRONG_FLOOR` for bench tuning. Because
/// `norm` is clamped to `<= 1.0`, a value `> 1.0` (e.g. 2.0) restores the pre-#393
/// hard scope gate — every out-of-scope node is dropped, nothing overrides scope.
const SCOPE_STRONG_FLOOR_DEFAULT: f32 = 0.3;

fn scope_strong_floor() -> f32 {
    std::env::var("TRAVSR_SCOPE_STRONG_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=2.0).contains(&x))
        .unwrap_or(SCOPE_STRONG_FLOOR_DEFAULT)
}

/// #478: ceiling on the displayed score of an unscored primary seed (the
/// reranker never scored it — G1 bypass, or reranker off/degraded), so a
/// PPR-normalised 1.0 (true by construction for the top item in any batch,
/// see the PPR normalisation in `tools.rs`) never reads as an absolute
/// judgement next to a `weak`/`none` confidence label. The reranked branch in
/// `display_score` is untouched — an absolute cross-encoder score is already
/// honest.
///
/// Values are display-only proposals (RFC-023 §14.4, not bench-measured).
/// Gate behind `TRAVSR_DISPLAY_TIER_CAP`; set to `0` to restore pre-#478
/// output byte-for-byte.
fn confidence_display_ceiling(confidence: Confidence) -> f32 {
    if std::env::var("TRAVSR_DISPLAY_TIER_CAP").as_deref() == Ok("0") {
        return 1.0;
    }
    match confidence {
        Confidence::Exact | Confidence::Strong => 1.0,
        Confidence::Weak => 0.60,
        Confidence::None => 0.40,
    }
}

/// #393 score-aware scope gate decision (pure, so the admit/gate policy is
/// unit-testable without a store fixture). Returns `true` = DROP the seed.
///
/// An out-of-scope lexical seed is dropped only when anchors established a scope
/// AND the seed's normalised score is below `floor`. In-scope seeds, seeds with
/// no anchor scope, and strong out-of-scope seeds (`norm >= floor`) are kept.
fn scope_gate_drops(has_anchor_scope: bool, in_scope: bool, norm: f32, floor: f32) -> bool {
    has_anchor_scope && !in_scope && norm < floor
}

/// Demote seeds the embed oracle places far from the query, then reshuffle the
/// survivors by cosine.  A seed absent from the oracle (not among the over-fetched
/// nearest neighbours) is treated as semantically distant (cosine 0.0) — this is
/// what cuts a spurious lexical anchor like `c1_raw_literal` (FTS-matched the NL
/// word "literal", cosine far below the genuine semantic neighbours).
///
/// No-op when the oracle is empty (embeddings off/degraded) or not confident
/// (top cosine < `SEMANTIC_ORACLE_MIN`) — lexical evidence then stands unchanged.
fn semantic_validate(
    seeds: Vec<Seed>,
    oracle: &HashMap<NodeId, f32>,
    cal: &Calibration,
) -> Vec<Seed> {
    if oracle.is_empty() || seeds.is_empty() {
        return seeds;
    }
    let top = oracle.values().copied().fold(0.0_f32, f32::max);
    if top < cal.map(SEMANTIC_ORACLE_MIN) {
        return seeds; // oracle not confident — trust lexical evidence
    }
    // The relative floor is anchored on the ORACLE top, which may be a non-seed
    // over-fetch neighbour (or a noise node) far above the best actual seed. When
    // it is, a genuine top-KNN seed can fall below `floor` and, with ≥MIN other
    // seeds above it, be truncated out entirely (#462 WS1: the rank-1/2 conceptual
    // gold "DROP@semantic_validate"). Guard by never cutting a seed that clears the
    // model-relative recall floor — the same "this is a real semantic match" bar
    // `embed_path_seeds` admits KNN on. `keep_floor` only ever LOWERS the cut.
    // Calibrated-only: skipped on identity/bge so truncation stays byte-for-byte —
    // the 0.75 identity recall floor does NOT coincide with the relative `floor` for
    // strong-oracle queries (top > 0.88, where floor > 0.75), so applying it there
    // would silently change the un-validated reference path.
    let floor = (top - cal.map_delta(SEMANTIC_REL_DELTA)).max(cal.map(SEMANTIC_ABS_FLOOR));
    let keep_floor = if *cal == Calibration::IDENTITY {
        floor
    } else {
        floor.min(semantic_recall_floor(cal))
    };

    let mut scored: Vec<(Seed, f32)> = seeds
        .into_iter()
        .map(|s| {
            let cos = oracle.get(&s.node).copied().unwrap_or(0.0);
            (s, cos)
        })
        .collect();

    let n_above = scored.iter().filter(|(_, c)| *c >= keep_floor).count();
    if n_above < SEMANTIC_MIN_KEEP {
        // Graceful: too few clear the floor — keep the top-N by cosine so a
        // grounded result is never emptied by an over-harsh oracle.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(SEMANTIC_MIN_KEEP);
    } else {
        scored.retain(|(_, c)| *c >= keep_floor);
        // Reshuffle: most semantically-relevant first.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    scored.into_iter().map(|(s, _)| s).collect()
}

/// RFC-021 / Issue D — apply `semantic_validate`'s whole-query cosine filter to
/// the lexical/KNN recall tail **only**. `SeedSource::Exact` anchors (deterministic
/// FTS name matches) bypass it entirely and always survive to the reranker, which
/// RFC-021 §15 ratified as the *sole* relevance arbiter.
///
/// Why: `semantic_validate` scores an oracle-absent seed as cosine `0.0` and cuts
/// it. For a conversational query like `"give me data about daemon"`, the *diluted*
/// whole-query embedding scores the genuine `Daemon` exact anchors below the floor,
/// so they were dropped before the cross-encoder ever judged them → false abstain.
/// Letting exact anchors through fixes that without re-introducing the veto: salad
/// (`"...drop the database"`) still abstains because the reranker scores the
/// surviving exact `drop` seed low downstream, not because this stage dropped it.
/// Lexical/KNN seeds (coincidental token-overlap like `"literal"`/`"semantic"`, and
/// embedding neighbours) remain subject to the cosine cut, as before.
fn semantic_validate_preserving_exact(
    seeds: Vec<Seed>,
    oracle: &HashMap<NodeId, f32>,
    cal: &Calibration,
) -> Vec<Seed> {
    let (exact, tail): (Vec<Seed>, Vec<Seed>) = seeds
        .into_iter()
        .partition(|s| s.source == SeedSource::Exact);
    let validated = semantic_validate(tail, oracle, cal);
    // Prepend (not append) the untouched exact anchors. The caller reranks only
    // the top-K front slice (`seeds[..min(len, TRAVSR_RERANK_TOPK)]`); on a dense
    // repo where the surviving lexical/KNN tail is ≥ K, appending would push the
    // exact anchors past that window so they never get a `rerank_score` and drop
    // out of `max_rerank_score` — silently re-opening Issue D (a moderately-common
    // named symbol falsely abstains) by a different mechanism than the veto this
    // helper removed. Exact anchors are the literally-named symbol and must always
    // reach the reranker; `sort_seeds_post_rerank` re-imposes final output order
    // (exact → band 0) afterwards regardless of this input order.
    let mut out = exact;
    out.extend(validated);
    out
}

// ── RFC-019 seed re-rank helpers ──────────────────────────────────────────────

/// Cosine at/above which a **structurally disjoint** non-exact seed is rescued from
/// the contamination gate — i.e. treated as a genuine cross-package semantic match
/// the graph missed rather than a token-collision false positive. Set near-duplicate
/// high: measured token-collision seeds (`RemovePod` ≈0.751 for `GetWarningsForPod`)
/// sit below it, so they are dropped, while a true near-duplicate (≈0.9) survives.
/// Override via `TRAVSR_DISJOINT_RESCUE_COS`.
fn disjoint_rescue_cos() -> f32 {
    std::env::var("TRAVSR_DISJOINT_RESCUE_COS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &f32| (0.0..=1.0).contains(&x))
        .unwrap_or(0.85)
}

/// Max node-visits when expanding the exact anchors' structural neighbourhood —
/// guards against a pathological high-degree hub blowing up the BFS.
const ANCHOR_NEIGHBORHOOD_CAP: usize = 2_000;
/// Hop radius for the structural-disjointness test in the contamination gate.
const ANCHOR_NEIGHBORHOOD_HOPS: usize = 2;
/// Max forward callees seeded per query from the exact anchors (ranked by
/// `kind_boost`), so anchor-callee seeding cannot flood the personalisation vector.
/// Anchors emitted per query token.
///
/// #540: named rather than repeated inline so `travsr explain` can report the
/// shortfall against the value actually used, instead of restating a literal
/// that could drift away from it.
pub(crate) const MAX_ANCHORS_PER_TOKEN: usize = 3;

const MAX_ANCHOR_CALLEE_SEEDS: usize = 8;

#[cfg(test)]
mod anchor_capacity_tests {
    use super::MAX_ANCHORS_PER_TOKEN;

    /// #540: `explain` reports the shortfall as
    /// `symbol_freq - min(symbol_freq, MAX_ANCHORS_PER_TOKEN)`. Pinning the
    /// arithmetic here keeps that report honest if the cap ever moves — the
    /// whole point of naming the constant was that the note must not restate a
    /// literal that has drifted away from the code.
    fn considered(symbol_freq: usize) -> usize {
        symbol_freq.min(MAX_ANCHORS_PER_TOKEN)
    }

    #[test]
    fn a_token_naming_more_symbols_than_the_cap_reports_the_shortfall() {
        // The issue's own case: "sqlite" naming ~168 symbols.
        assert_eq!(considered(168), MAX_ANCHORS_PER_TOKEN);
        assert_eq!(168 - considered(168), 168 - MAX_ANCHORS_PER_TOKEN);
    }

    #[test]
    fn a_token_within_the_cap_reports_no_shortfall() {
        // "journal" named 2 symbols in the same query and carried it — no note
        // should fire for a token that lost nothing.
        assert_eq!(considered(2), 2);
        assert_eq!(2 - considered(2), 0);
        assert_eq!(
            MAX_ANCHORS_PER_TOKEN - considered(MAX_ANCHORS_PER_TOKEN),
            0,
            "exactly at the cap is not a shortfall"
        );
    }
}
/// Base weight for an anchor-callee seed before `kind_boost`. High enough that the
/// delegated function body ranks as a real dependency, below an exact anchor's
/// floated weight so it never outranks the named symbol itself.
const ANCHOR_CALLEE_WEIGHT_BASE: f32 = 0.8;

/// Bounded ≤`ANCHOR_NEIGHBORHOOD_HOPS`-hop neighbourhood of `anchors` over
/// structural edges in **both** directions. Used only to distinguish a
/// contaminated seed (no path to any exact anchor) from a genuine dependency.
/// O(visited·avg_degree); capped at `ANCHOR_NEIGHBORHOOD_CAP` visits. Edge-read
/// errors degrade gracefully (that frontier simply doesn't expand).
fn anchor_neighborhood(
    store: &SqliteStore,
    anchors: &std::collections::HashSet<NodeId>,
) -> std::collections::HashSet<NodeId> {
    let mut visited: std::collections::HashSet<NodeId> = anchors.iter().copied().collect();
    let mut frontier: Vec<NodeId> = anchors.iter().copied().collect();
    for _ in 0..ANCHOR_NEIGHBORHOOD_HOPS {
        if frontier.is_empty() || visited.len() >= ANCHOR_NEIGHBORHOOD_CAP {
            break;
        }
        let mut next: Vec<NodeId> = Vec::new();
        for node in frontier.drain(..) {
            if let Ok(fwd) = store.iter_edges_from(node) {
                for e in fwd {
                    if visited.insert(e.dst) {
                        next.push(e.dst);
                    }
                }
            }
            if let Ok(rev) = store.iter_edges_to(node) {
                for e in rev {
                    if visited.insert(e.src) {
                        next.push(e.src);
                    }
                }
            }
            if visited.len() >= ANCHOR_NEIGHBORHOOD_CAP {
                break;
            }
        }
        frontier = next;
    }
    visited
}

/// Forward `RefCall`/`Depends` callees of the exact `anchors`, as new PPR seeds.
/// Filters build/test noise and RBAC-denied nodes, skips ids already seeded, and
/// keeps the top `MAX_ANCHOR_CALLEE_SEEDS` by `kind_boost` (deterministic tie-break
/// on NodeId). Weight = `ANCHOR_CALLEE_WEIGHT_BASE × kind_boost`.
fn anchor_callee_seeds(
    store: &SqliteStore,
    filter: &dyn EdgeFilter,
    anchors: &std::collections::HashSet<NodeId>,
    existing: &std::collections::HashSet<NodeId>,
) -> Vec<Seed> {
    let mut callee_ids: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for &anchor in anchors {
        if let Ok(fwd) = store.iter_edges_from(anchor) {
            for e in fwd {
                if matches!(e.kind, EdgeKind::RefCall | EdgeKind::Depends)
                    && !existing.contains(&e.dst)
                    && !anchors.contains(&e.dst)
                {
                    callee_ids.insert(e.dst);
                }
            }
        }
    }
    if callee_ids.is_empty() {
        return Vec::new();
    }
    let ids: Vec<NodeId> = callee_ids.into_iter().collect();
    let nodes = match store.get_nodes(&ids) {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    let mut scored: Vec<(NodeId, f32)> = nodes
        .into_iter()
        .filter(|n| !is_noise_seed(n))
        .filter(|n| filter.allow(n.id, n.id, Some(n.vname.corpus.as_str())))
        .map(|n| {
            let w = ANCHOR_CALLEE_WEIGHT_BASE * kind_boost(&n.kind, &n.vname.language);
            (n.id, w)
        })
        .collect();
    // Highest kind_boost first; deterministic tie-break on NodeId ascending.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(MAX_ANCHOR_CALLEE_SEEDS);
    scored
        .into_iter()
        .map(|(node, weight)| Seed {
            node,
            weight,
            source: SeedSource::Lexical,
            score: weight,
            rerank_score: None,
        })
        .collect()
}

/// Build a `SeedSet` for `query` using per-token anchor resolution + whole-query
/// lexical FTS, fused via RRF.
///
/// When `knn_pairs` is non-empty (Tier 1 — embed sidecar contributed), those pairs
/// are included as a third source in `rrf_fuse`, weighted `cosine × kind_boost`.
/// KNN is score-gated (in `embed_path_seeds`), not structure-gated, so it can
/// surface semantically relevant nodes in different crates than the FTS/anchor
/// scope.  After fusion, `semantic_validate` consults the `knn_oracle` (query
/// cosine for every over-fetched candidate) to demote lexical anchors the
/// embedding disagrees with — fixing the "confident salad" on NL queries whose
/// words resolve to rare-by-coincidence symbols.
///
/// Filters applied before RRF:
/// - `is_noise_seed` — excludes crates, test/bench fixtures, build artefacts
/// - RBAC via `filter`
/// - `MAX_SEEDS_PER_PATH` dedup (2 per file) to prevent all seeds clustering in one file
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn build_seed_set(
    store: &SqliteStore,
    query: &str,
    filter: &dyn EdgeFilter,
    knn_pairs: Vec<(CoreNode, f32)>,
    knn_oracle: &HashMap<NodeId, f32>,
    score_fn: Option<&dyn Fn(&str, &[NodeId]) -> Vec<(NodeId, f32)>>,
) -> SeedSet {
    const MAX_SEEDS_PER_PATH: usize = 2;

    // Cross-surface parity (RFC-021 F9): normalize once here so BOTH callers feed
    // the identical cleaned query downstream. `ask_query` normalized upstream but
    // `get_context` did not, so a punctuated query ("how does the daemon reindex?")
    // reached the direct-cosine oracle (and FTS + reranker) as a different string
    // on the two surfaces, diverging `max_rerank_score` and hence the confidence
    // label. `normalize_nl_query` is idempotent, so the already-normalized ask path
    // is unaffected.
    let normalized_query = travsr_store::fts_tokenize::normalize_nl_query(query);
    let query = normalized_query.as_str();

    let intent = classify_intent(query);
    let content_tokens = tokenize_query(query);
    let n_content = content_tokens.len();

    // Model-relative floor calibration for this repo's active embedding model.
    // Identity (no-op) when the repo is un-calibrated or embedded with the bge-small
    // reference model — so the FTS-only and bge-small paths are byte-for-byte unchanged.
    let cal = Calibration::load(store);

    // Corpus size for IDF.  Clamp to ≥ 1 000 so the formula remains meaningful in
    // small repos / in-memory test stores: idf_weight(1, 1) collapses to the 0.05 floor,
    // making every token look "generic" and causing spurious abstention in tiny corpora.
    // For real production corpora (≥ 1 000 nodes) the value is unchanged.
    let n_total = store.total_node_count().unwrap_or(10_000).max(1_000);

    // ── Per-token anchor resolution ───────────────────────────────────────────
    let mut terms: Vec<ResolvedTerm> = Vec::with_capacity(n_content);
    // anchor_pairs: (NodeId, weight) where weight = idf_weight * kind_boost
    let mut anchor_raw: Vec<(NodeId, f32)> = Vec::new();
    // Track per-path counts across anchor resolution too.
    let mut anchor_path_counts: HashMap<String, usize> = HashMap::new();

    // #463 anchor-ordering config is read once here, not per token: both are
    // `std::env::var` reads (global lock + String alloc) and cannot change within a
    // single query, so hoisting them out of the loop keeps the seed hot path free of
    // per-token env lookups.
    let anchor_kind_priority = anchor_kind_priority();
    let anchor_reorder_window = anchor_reorder_window();
    // IDF-coverage bar, hoisted (reused for n_resolved below). RFC-022 D1.4: an
    // anchor from a token clearing this bar is "specific"; one from a mid-IDF token
    // (`idf_w ∈ [0.15, idf_min)`) is "generic" and is down-weighted in weighted RRF.
    let idf_min = idf_coverage_min();
    let mut specific_token_anchor_ids: std::collections::HashSet<NodeId> =
        std::collections::HashSet::new();
    // #709: anchor nodes emitted from a unique strict typo correction. These are
    // deterministic lexical evidence (a UNIQUE, high-Jaccard match to one real
    // symbol), so, like a rare exact anchor, they may drive the g1 bypass and
    // are shielded from the NL cross-encoder, which does not rate a misspelt
    // query against a code symbol. Precision is bounded upstream by the strict,
    // uniqueness-gated correction and the g1 subject-share test, not the reranker.
    let mut corrected_anchor_ids: std::collections::HashSet<NodeId> =
        std::collections::HashSet::new();

    // Boundary predicate reused for both the direct token and a #709 correction:
    // a token counts as resolved only when it appears as a whole word segment (or
    // contiguous run of segments) in the candidate's signature or path, not just
    // as a substring of a longer word (#478 RFC-023 §6.2: an unguarded substring
    // hit read "wal" as resolved via "walker.ts").
    let boundary = |tok: &str, n: &CoreNode| -> bool {
        travsr_core::ident::contains_token(tok, &n.vname.signature)
            || travsr_core::ident::contains_token(tok, &n.vname.path)
    };
    // Resolve every token by name first, then correct all the misses in one
    // call. `fuzzy_correct_symbols` costs one distinct-signature scan whatever
    // the token count, so batching keeps a query carrying several unresolved
    // tokens at a single scan rather than one scan per token on this hot path.
    let by_token: Vec<(String, Vec<CoreNode>, bool)> = content_tokens
        .iter()
        .map(|token| {
            let nodes = store.search_nodes_by_name(token).unwrap_or_default();
            let has_direct = nodes.iter().any(|n| boundary(token, n));
            (token.clone(), nodes, has_direct)
        })
        .collect();
    let missed: Vec<&str> = by_token
        .iter()
        .filter(|(_, _, has_direct)| !has_direct)
        .map(|(token, _, _)| token.as_str())
        .collect();
    let corrections = if missed.is_empty() {
        std::collections::HashMap::new()
    } else {
        store
            .fuzzy_correct_symbols(&missed, fuzzy_correct_jaccard())
            .unwrap_or_default()
    };

    for (token, direct_nodes, has_direct) in by_token {
        let token = &token;

        // #709: when a token resolves to nothing by name, accept a UNIQUE strong
        // trigram correction to a real symbol so a typo (`htpresponse`) grounds
        // through the anchor path instead of abstaining. The correction reuses the
        // corrected symbol's own name for frequency/IDF (so `HttpResponse` is
        // measured, not the unindexed typo) and, being deliberate and unique, is
        // always emitted as an anchor below, the cross-encoder reranker is the
        // precision backstop. Only fires on the exact-miss cold branch.
        let (exact_nodes, resolve_token, corrected): (Vec<CoreNode>, String, bool) = if has_direct {
            (direct_nodes, token.clone(), false)
        } else if let Some(fix) = corrections.get(token) {
            let fixed_nodes = store.search_nodes_by_name(fix).unwrap_or_default();
            if fixed_nodes.iter().any(|n| boundary(fix, n)) {
                (fixed_nodes, fix.clone(), true)
            } else {
                (direct_nodes, token.clone(), false)
            }
        } else {
            (direct_nodes, token.clone(), false)
        };

        // #478: absent from the vocabulary (unknown token, or < 3 bytes so
        // never indexed) is treated as maximally generic, matching the
        // previous `unwrap_or(n_total)` failure behaviour. A token we cannot
        // measure must never be *promoted* to a rare anchor.
        let freq = store
            .symbol_frequency(&resolve_token)
            .ok()
            .flatten()
            .unwrap_or(n_total);
        let boundary_matched: Vec<&CoreNode> = exact_nodes
            .iter()
            .filter(|n| boundary(&resolve_token, n))
            .collect();
        let resolved = !boundary_matched.is_empty();

        // #778: the anchor candidate POOL — nodes that carry the (possibly
        // corrected) token as an actual NAME segment, not a mere substring.
        // `search_nodes_by_name` is a SUBSTRING match, so for a short token it
        // also returns path/substring noise ("ui" inside "building" ->
        // `method:Gym.building_*`). The per-node `contains_token` gate in the
        // emit loop already rejects that noise, but leaving it in the pool let
        // the #463 kind-reorder promote logic-bearing noise ahead of the true
        // match and then consume the `MAX_ANCHORS_PER_TOKEN` budget via
        // take-then-filter, starving the exact `class:UI` anchor (issue #778).
        //
        // This is a RECALL INCREASE, not a no-op: pre-filtering changes what the
        // `anchor_reorder_window` head covers, so for a starved token the emitted
        // anchors can go from 0 to `MAX_ANCHORS_PER_TOKEN` — which shifts RRF
        // ordering, the per-path budget and the g1 input. It only ever admits
        // nodes the emit loop's own `contains_token` gate would have kept anyway;
        // it never admits a node that could not have emitted. Uses the
        // signature-only predicate to match that gate exactly (a path-only match
        // is not an anchor — #478). Borrows from `exact_nodes` (no clone): the
        // pool lives only within this loop iteration.
        let anchor_pool: Vec<&CoreNode> = exact_nodes
            .iter()
            .filter(|n| travsr_core::ident::contains_token(&resolve_token, &n.vname.signature))
            .collect();

        // #709: the corrected anchor deliberately bypasses the *budget* gates
        // below (the IDF emit cut and the per-path cap) because a unique strict
        // correction is deliberate lexical evidence that must seed and must
        // become this term's `top_node` for the g1 rare-anchor linkage. It must
        // still clear the same *correctness* gates the ordinary loop applies,
        // which is what this shares with it: anchor-noise, RBAC, and the
        // signature-only `contains_token`. That last one matters most. The
        // `boundary` predicate above accepts a signature OR path match, so
        // taking the first boundary match unfiltered could pin a protected,
        // gate-bypassing anchor on a symbol that merely lives in a file named
        // after the corrected token rather than being that symbol. Candidates
        // are kind-ordered first for the same reason as the ordinary loop, so
        // the correction anchors to a definition rather than to whichever
        // import FTS happened to rank first.
        let corrected_pick: Option<&CoreNode> = if corrected {
            let ordered: Vec<&CoreNode> = if anchor_kind_priority {
                order_anchor_candidates(&anchor_pool, anchor_reorder_window)
            } else {
                anchor_pool
                    .iter()
                    .copied()
                    .take(MAX_ANCHORS_PER_TOKEN)
                    .collect()
            };
            ordered.into_iter().find(|node| {
                !is_anchor_noise(node)
                    && filter.allow(node.id, node.id, Some(node.vname.corpus.as_str()))
                    && travsr_core::ident::contains_token(&resolve_token, &node.vname.signature)
            })
        } else {
            None
        };

        // A corrected term's `top_node` is the node it will actually emit, not
        // the raw first boundary match: `g1_bypass_decision` requires the two to
        // agree, so a correction whose candidates all fail the gates above must
        // report no top node rather than one it never anchored.
        let top_node = if corrected {
            corrected_pick.map(|n| n.id)
        } else {
            boundary_matched.first().map(|n| n.id)
        };

        let idf_w = idf_weight(freq, n_total);

        // #540: filled in after the anchor loop below with what it actually
        // emitted. Stays 0 for a token suppressed by the IDF cut, which never
        // reaches the loop.
        let term_idx = terms.len();
        terms.push(ResolvedTerm {
            token: token.clone(),
            resolved,
            symbol_freq: freq,
            idf_w,
            top_node,
            anchors_emitted: 0,
        });

        // #709: a unique strict typo correction emits exactly one protected
        // anchor, the correction target, bypassing the generic IDF/path-cap
        // gates so the corrected symbol is guaranteed to seed and to be the term's
        // `top_node` (which the g1 rare-anchor linkage requires). Recorded in
        // `corrected_anchor_ids` so the g1 bypass shields it from the NL reranker.
        if corrected {
            if let Some(node) = corrected_pick {
                let w = idf_w * kind_boost(&node.kind, &node.vname.language);
                anchor_raw.push((node.id, w));
                specific_token_anchor_ids.insert(node.id);
                corrected_anchor_ids.insert(node.id);
                *anchor_path_counts
                    .entry(node.vname.path.clone())
                    .or_insert(0) += 1;
                terms[term_idx].anchors_emitted = 1;
            }
            continue;
        }

        // Only emit high-IDF tokens as anchors (suppresses generic "queue", "run" etc.)
        if idf_w < anchor_emit_cut() {
            continue;
        }
        // Emit top-3 matches per token to represent the anchor fully.
        // `added_for_this_token` guarantees this token at least one anchor slot even
        // when the shared per-path budget is already exhausted by an EARLIER token —
        // otherwise a common word processed first (e.g. "daemon") can fully starve a
        // later, more specific token's match in the same file (e.g. "watch" ->
        // handle_watch_event in crates/travsr-daemon/src/lib.rs, dropped purely
        // because "daemon" already filled that path's 2-slot cap). Every candidate
        // still has to pass noise/filter/token_is_sig_component first, so the bypass
        // only ever fires for a genuine match — it just can't be starved by token
        // processing order. Beyond its first slot, a token is capped exactly as
        // before.
        let mut added_for_this_token = 0usize;
        // #463: prefer logic-bearing definitions among the top name-matches before the
        // bounded emit (on by default; `TRAVSR_ANCHOR_KIND_PRIORITY=0` restores the
        // original FTS-rank order). See [`order_anchor_candidates`].
        let ordered: Vec<&CoreNode> = if anchor_kind_priority {
            order_anchor_candidates(&anchor_pool, anchor_reorder_window)
        } else {
            anchor_pool
                .iter()
                .copied()
                .take(MAX_ANCHORS_PER_TOKEN)
                .collect()
        };
        for node in ordered.into_iter().take(MAX_ANCHORS_PER_TOKEN) {
            // RFC-022 D3: stricter anchor-pool gate than the general `is_noise_seed`
            // — also drops CI/workflow `pkg:` nodes and in-src test symbols so they
            // cannot out-anchor the real implementation or trigger a false G1 bypass.
            if is_anchor_noise(node) {
                continue;
            }
            if !filter.allow(node.id, node.id, Some(node.vname.corpus.as_str())) {
                continue;
            }
            // Require the token to appear as a whole word segment (or a contiguous
            // run of segments) in the signature, not just as a substring of a
            // longer word. #478 WS-1: `ident::contains_token` is camelCase/
            // PascalCase-aware (unlike the old punctuation-only boundary check),
            // so "sqlite" now correctly anchors to `fn:SqliteStore.exec_ddl` while
            // "works" still does not anchor to "provideWorkspaceChatContext".
            // #709: use the (possibly corrected) resolve_token so a typo's
            // corrected symbol name gates its own anchors.
            if !travsr_core::ident::contains_token(&resolve_token, &node.vname.signature) {
                continue;
            }
            let path_count = anchor_path_counts
                .entry(node.vname.path.clone())
                .or_insert(0);
            if *path_count >= MAX_SEEDS_PER_PATH && added_for_this_token > 0 {
                continue;
            }
            *path_count += 1;
            added_for_this_token += 1;
            let w = idf_w * kind_boost(&node.kind, &node.vname.language);
            anchor_raw.push((node.id, w));
            // RFC-022 D1.4: a node anchored by ANY specific (high-IDF) token is a
            // specific anchor; the weighted-RRF split below caps only the remainder.
            if idf_w >= idf_min {
                specific_token_anchor_ids.insert(node.id);
            }
        }

        // #540: the measured count, after every drop the loop applies.
        terms[term_idx].anchors_emitted = added_for_this_token;
    }

    // `rrf_fuse` (below) requires every source sorted descending by score —
    // "position 0 = best" is its documented precondition, since it scores by
    // list position, not by the weight itself. `anchor_raw` is built in
    // token-processing order (all of one token's matches, then the next), not
    // relevance order, so without this sort a token processed later in the
    // query (e.g. "daemon") gets a worse RRF rank than an earlier token's
    // weaker matches (e.g. "data") purely because it was appended later —
    // regardless of which one is actually the better match.
    anchor_raw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // G1: computed here, before semantic_validate, not after it. `anchor_raw`
    // is exactly what will become the Exact-sourced seeds once RRF fusion
    // runs, so its own node ids are the right (and only available) exact-id
    // set at this point. This matters: `semantic_validate` (below) can drop
    // an Exact seed whose cosine to the whole-query embedding is low even
    // when the seed is a genuine, rare, literal match — e.g. "give me data
    // about daemon" scored struct:Daemon's cosine below several unrelated
    // "data"-flavoured seeds', and semantic_validate silently removed it
    // before G1 (which exists precisely to protect rare exact matches from
    // NL-trained signals) ever got a chance to see it, since the old
    // g1_bypass was computed from the POST-semantic_validate seed list. G1's
    // deterministic path must not depend on the NL embedding oracle agreeing
    // first — that is exactly the failure mode it exists to prevent.
    let early_exact_ids: std::collections::HashSet<NodeId> =
        anchor_raw.iter().map(|&(id, _)| id).collect();
    let g1_bypass = compute_g1_bypass(
        &terms,
        &TrustedAnchors {
            exact: &early_exact_ids,
            corrected: &corrected_anchor_ids,
        },
    );

    // Count only tokens that resolve to specific-enough anchors (IDF ≥ threshold).
    // Generic tokens like "map" or "get" match hundreds of unrelated nodes and must
    // not inflate coverage — they are not evidence the query is grounded in this repo.
    let n_resolved = terms
        .iter()
        .filter(|t| t.resolved && t.idf_w >= idf_min)
        .count();
    // When there are no content tokens (query is all stop-words or very short chars),
    // use a neutral coverage of 0.5 — confidence falls back to BM25 score and seed count.
    let coverage = if n_content == 0 {
        0.5
    } else {
        n_resolved as f32 / n_content as f32
    };

    // ── Structural scope from anchor seeds ───────────────────────────────────
    // Anchor seeds (direct symbol resolution) are the most trusted source.
    // Their package roots (first two path segments) define the structural scope:
    // any FTS or KNN result from outside this scope is cross-domain noise and is dropped.
    //
    // Example: "how ppr works" → anchors in crates/travsr-retrieval → scope = that crate.
    //          FTS trigram-matches "works" → "workspace" in packages/travsr-vscode
    //          → filtered out even though FTS returned it with a high BM25 score.
    //
    // When no anchor seeds exist (no token resolved to a specific symbol), scope is
    // unrestricted and all FTS / KNN results are admitted as the only available signal.
    let anchor_roots: std::collections::HashSet<&str> = anchor_path_counts
        .keys()
        .filter_map(|p| package_root(p))
        .collect();

    // ── Whole-query lexical FTS (scored BM25) ────────────────────────────────
    let lexical_scored = store.search_nodes_fuzzy_scored(query).unwrap_or_default();

    // #393: `search_nodes_fuzzy_scored` now returns RRF-ordered + kind-diversified
    // results, so the strongest lexical match is NO LONGER at position 0. Take the
    // explicit max over the batch — using `.first()` here would understate the
    // normalisation denominator and make the score-aware scope gate below (and the
    // `top_bm25` abstention signal) silently more permissive than intended.
    //
    // #478 RFC-023 §5.4/Evidence E: fold over `bm25_natural`, not the old
    // conflated `natural` field. `natural` mixes a real BM25 score (Leg B/C)
    // with a position-derived synthetic score (Leg A) and even an L2-A/embed
    // score on a miss — reading it as "the BM25 batch max" is exactly how
    // `fn:walk` reached norm 1.0 with no lexical evidence at all. `bm25_natural`
    // is `None` unless Leg B or Leg C actually matched, so this fold is now a
    // true BM25-scale max, not a mixed-scale one.
    let top_bm25 = lexical_scored
        .iter()
        .filter_map(|hit| hit.bm25_natural)
        .fold(0.0_f32, f32::max);
    // Normalise BM25 scores per-batch for PPR weight (max = 1.0); floor at 0.05.
    let max_bm25 = top_bm25.max(0.001);

    let mut lex_path_counts: HashMap<String, usize> = HashMap::new();
    let mut lexical_raw: Vec<(NodeId, f32)> = Vec::new();
    for hit in &lexical_scored {
        let node = &hit.node;
        // #478: no real BM25-scale evidence (Leg B/C) for this node — it only
        // reached the fused result via Leg A (exact/name) or, on a miss,
        // L2-A/embed. Those are not lexical evidence; the anchor loop above
        // and the KNN loop below are the correct paths for them. Admitting a
        // node here with no lexical backing at all is exactly how a
        // substring-only match used to escape crate scope at norm 1.0.
        let Some(bm25) = hit.bm25_natural else {
            continue;
        };
        if is_noise_seed(node) {
            continue;
        }
        if !filter.allow(node.id, node.id, Some(node.vname.corpus.as_str())) {
            continue;
        }
        let norm = (bm25 / max_bm25).clamp(0.05, 1.0);
        // Score-aware structural scope gate (#393). If anchors established a crate
        // scope, a node outside it is dropped ONLY when its lexical match is weak
        // (normalised score < scope_strong_floor()). A strong lexical match is trusted
        // regardless of crate, so a legitimately multi-domain query (a term that
        // names both a symbol and a directory in different crates) is not collapsed
        // to a single anchor's scope. Weak cross-domain FTS drift stays gated.
        // Nodes with no parseable root are allowed through as before.
        //
        // Safety note: on a warm embed oracle, `semantic_validate` / the RFC-019
        // contamination gate further prune anything admitted here that the query
        // embedding disagrees with. On the FTS-only / cold-oracle path there is no
        // such backstop, so the `scope_strong_floor()` (relative BM25) IS the gate.
        // The floor is deliberately the sole knob (env-tunable), and it depends on
        // `max_bm25` being the true batch max above — hence the explicit fold-max.
        let in_scope = match package_root(&node.vname.path) {
            Some(root) => anchor_roots.contains(root),
            None => true, // no parseable root — allowed through as before
        };
        if scope_gate_drops(
            !anchor_roots.is_empty(),
            in_scope,
            norm,
            scope_strong_floor(),
        ) {
            continue;
        }
        let path_count = lex_path_counts.entry(node.vname.path.clone()).or_insert(0);
        if *path_count >= MAX_SEEDS_PER_PATH {
            continue;
        }
        *path_count += 1;
        let w = norm * kind_boost(&node.kind, &node.vname.language);
        lexical_raw.push((node.id, w));
    }

    // ── KNN seeds — score-gated, not structure-gated ─────────────────────────
    // KNN's purpose is to surface semantically relevant nodes that lexical
    // search misses — including nodes in different crates or packages from the
    // FTS/anchor scope.  Applying an anchor-root scope gate here defeats that:
    // it degrades KNN to a reinforcement-only signal and blocks it from
    // correcting FTS mis-scoping (e.g. surfacing retrieval code when FTS
    // anchored to security test functions, or surfacing VS Code extension code
    // when FTS anchored to Rust crates for a "typescript extension" query).
    //
    // Nodes that appear in both KNN and anchor/FTS naturally rank higher via
    // RRF multi-source fusion, so semantically-correct nodes beat noise even
    // when noise-adjacent neighbours slip through.
    //
    // Weight = cosine × kind_boost, mirroring the anchor/lexical weighting so a
    // KNN seed competes fairly in the PPR personalisation vector. Raw cosine
    // (~0.65–0.80) would otherwise lose to a normalised FTS match × kind_boost
    // (up to ~3.0), systematically under-weighting semantic seeds.
    // Filter KNN results through is_noise_seed before weighting: build-cache nodes,
    // test paths, and OS caches must not enter the PPR personalisation vector.
    let knn_raw: Vec<(NodeId, f32)> = knn_pairs
        .iter()
        .filter(|(n, _)| !is_noise_seed(n))
        .map(|(n, s)| {
            (
                n.id,
                *s * crate::tools::kind_boost(&n.kind, &n.vname.language),
            )
        })
        .collect();

    // ── RRF fusion ────────────────────────────────────────────────────────────
    let k = rrf_k();
    let rrf_result = if rrf_weighted_enabled() {
        // RFC-022 D1.3/D1.4: split the exact-anchor source into specific (high-IDF)
        // vs generic (mid-IDF), then fuse with per-source weights so BM25/embed stay
        // first-class and generic per-token anchors cannot flood the fused top-N.
        // `anchor_raw` is already sorted desc by weight, so each split preserves that
        // order (rrf_fuse_weighted's position precondition).
        let (specific_anchor_raw, generic_anchor_raw): (Vec<_>, Vec<_>) = anchor_raw
            .iter()
            .cloned()
            .partition(|(id, _)| specific_token_anchor_ids.contains(id));
        let (w_spec, w_bm25, w_embed, w_generic) = rrf_source_weights();
        rrf_fuse_weighted(
            &[
                (w_spec, &specific_anchor_raw),
                (w_bm25, &lexical_raw),
                (w_embed, &knn_raw),
                (w_generic, &generic_anchor_raw),
            ],
            k,
        )
    } else {
        rrf_fuse(&[&anchor_raw, &lexical_raw, &knn_raw], k)
    };

    // Convert RRF result back to Seed structs, resolving weights from the
    // anchor/lexical maps (RRF score used as final ordering, original weights for PPR).
    // Build a weight lookup: node → max(anchor_weight, lexical_weight, knn_weight).
    let anchor_map: HashMap<NodeId, f32> = anchor_raw.iter().cloned().collect();
    let lexical_map: HashMap<NodeId, f32> = lexical_raw.iter().cloned().collect();
    let knn_map: HashMap<NodeId, f32> = knn_raw.iter().cloned().collect();

    // Source classification for metadata: prefer Exact > Lexical > Knn
    let seeds: Vec<Seed> = rrf_result
        .into_iter()
        .map(|(node_id, rrf_score)| {
            let anchor_w = anchor_map.get(&node_id).copied();
            let lex_w = lexical_map.get(&node_id).copied();
            let knn_w = knn_map.get(&node_id).copied();
            // Weight fed to PPR: take the max across sources so any strong signal wins.
            let weight = anchor_w
                .unwrap_or(0.0)
                .max(lex_w.unwrap_or(0.0))
                .max(knn_w.unwrap_or(0.0))
                .max(rrf_score * 0.5); // fallback: 50% of RRF score when weight unknown

            let source = if anchor_w.is_some() {
                SeedSource::Exact
            } else if knn_w.is_some() {
                SeedSource::Knn
            } else {
                SeedSource::Lexical
            };

            Seed {
                node: node_id,
                weight,
                source,
                score: rrf_score,
                rerank_score: None,
            }
        })
        .collect();

    let has_knn = !knn_raw.is_empty();

    // ── RFC-019: direct-cosine oracle augmentation ─────────────────────────────
    // The KNN oracle only holds cosines for nodes the KNN over-fetch happened to
    // surface. MEASURE the specific anchors (and exact-anchor nodes) it missed, so
    // the classifier/ranker use a true query↔candidate cosine instead of inferring
    // "relevance by membership". `scored_ids` records every id we submitted so the
    // classifier can tell "scored but unscoreable → unknown" from "never scored".
    // No-op when `score_fn` is None: `augmented_oracle == *knn_oracle`, `scored_ids`
    // empty → every downstream path is byte-for-byte identical to the FTS-only path.
    let mut augmented_oracle: HashMap<NodeId, f32> = knn_oracle.clone();
    let mut scored_ids: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    if let Some(score) = score_fn {
        let mut to_score: Vec<NodeId> = Vec::new();
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for t in &terms {
            if let Some(n) = t.top_node {
                if !knn_oracle.contains_key(&n) && seen.insert(n) {
                    to_score.push(n);
                }
            }
        }
        for &(id, _) in &anchor_raw {
            if !knn_oracle.contains_key(&id) && seen.insert(id) {
                to_score.push(id);
            }
        }
        if !to_score.is_empty() {
            scored_ids.extend(to_score.iter().copied());
            for (id, cos) in score(query, &to_score) {
                augmented_oracle.insert(id, cos);
            }
        }
    }

    // ── Tier-1 semantic validation ────────────────────────────────────────────
    // When the embed oracle is confident, use the query cosine to demote LEXICAL
    // anchors the embedding disagrees with. NL query words that are rare-by-
    // coincidence ("literal", "semantic") resolve to lexical anchors that are
    // semantically unrelated to intent; the whole-query embedding detects this.
    // No-op when embeddings are absent/degraded.
    //
    // RFC-021 / Issue D: exact anchors are EXEMPT (see
    // `semantic_validate_preserving_exact`). A deterministic FTS name match must
    // reach the reranker — the ratified relevance arbiter — and never be vetoed
    // by a diluted whole-query embedding ("give me data about daemon" abstained
    // because the `Daemon` exact anchors were cosine-cut before the cross-encoder
    // saw them). Salad still abstains: the reranker judges the surviving exact
    // seed low downstream.
    //
    // Skipped entirely when G1 already bypassed: the user named a genuinely
    // rare, specific symbol, and G1's whole point is that a deterministic
    // exact match must not be second-guessed by an NL-trained embedding signal.
    let mut seeds = if g1_bypass {
        seeds
    } else {
        semantic_validate_preserving_exact(seeds, &augmented_oracle, &cal)
    };

    // ── RFC-019: seed re-rank (contamination gate + anchor-callee seeding) ─────
    // Only when the query named an exact symbol (an exact anchor exists) — never on
    // pure-NL queries, which have no anchor and must be left to the semantic path.
    let exact_anchor_ids: std::collections::HashSet<NodeId> = seeds
        .iter()
        .filter(|s| s.source == SeedSource::Exact)
        .map(|s| s.node)
        .collect();
    if !exact_anchor_ids.is_empty() {
        // Bounded ≤2-hop structural neighbourhood of the exact anchors (both
        // directions). Computed once; a non-exact seed OUTSIDE it is structurally
        // disjoint from the named symbol.
        let neighborhood = anchor_neighborhood(store, &exact_anchor_ids);

        // Contamination gate. When the user names an exact symbol, its relevant
        // context is its STRUCTURAL neighbourhood, and a token-collision embedding
        // false positive is not distinguishable by cosine: measured on the live
        // k8s index, `InterPodAffinity.RemovePod` scores cosine **0.751** to query
        // `GetWarningsForPod` purely on the shared "Pod"/method tokens (BGE is
        // NL-trained and cannot disambiguate code identifiers) — well above any
        // sane floor. The reliable signal is structure: drop a non-exact seed that
        // is structurally DISJOINT from every exact anchor, UNLESS its cosine is
        // near-duplicate-high (a genuine cross-package semantic match the graph
        // missed). Structurally-connected seeds and un-scoreable seeds are always
        // kept, and the gate never fires on pure-NL queries (no exact anchor), so
        // conceptual retrieval is untouched.
        let rescue = cal.map(disjoint_rescue_cos());
        let recall = semantic_recall_floor(&cal);
        seeds.retain(|s| {
            if s.source == SeedSource::Exact {
                return true;
            }
            // #462 WS1: a conceptual query's exact anchor is COINCIDENTAL — a common
            // word ("pod", "pending", "scale") that happens to name a symbol — so the
            // anchor's structural neighbourhood need not contain the semantically
            // correct answer. A KNN seed that clears the model recall floor is the
            // embedding's genuine opinion and must not be dropped as "contamination"
            // just for being structurally disjoint from that coincidental anchor.
            // Guarded by `!g1_bypass`: when the user named a genuinely RARE symbol the
            // strict RFC-019 structural gate still governs, so token-collision false
            // positives (RemovePod ≈0.751 for `GetWarningsForPod`) stay dropped.
            // Salad-safe by the same floor: off-domain KNN neighbours sit below the
            // recall floor, so this never re-admits a salad seed. Calibrated-only:
            // on identity/bge the pre-existing `rescue` bar (0.85) still governs, so
            // the strict RFC-019 gate stays byte-for-byte and the [recall, rescue)
            // (0.75–0.85) token-collision band is NOT re-admitted on the reference path.
            if cal != Calibration::IDENTITY
                && !g1_bypass
                && s.source == SeedSource::Knn
                && augmented_oracle
                    .get(&s.node)
                    .map(|&c| c >= recall)
                    .unwrap_or(false)
            {
                return true;
            }
            let disjoint = !neighborhood.contains(&s.node);
            // Drop only a disjoint seed whose (known) cosine is below the
            // near-duplicate rescue bar. Connected or un-scoreable → keep.
            !disjoint
                || augmented_oracle
                    .get(&s.node)
                    .map(|&c| c >= rescue)
                    .unwrap_or(true)
        });

        // Anchor-callee seeding: add the exact anchors' own 1-hop RefCall/Depends
        // callees as seeds, so the function body it delegates to
        // (`warningsForPodSpecAndMeta` for `GetWarningsForPod` — private/lowercase,
        // never lexically seeded) is scored as the dependency it is instead of
        // being buried below a tangential seed. `enrich_seeds_with_callers` adds
        // reverse (caller) edges; this adds the complementary forward callees.
        let existing: std::collections::HashSet<NodeId> = seeds.iter().map(|s| s.node).collect();
        let callees = anchor_callee_seeds(store, filter, &exact_anchor_ids, &existing);
        seeds.extend(callees);
    }

    // Exact-name priority: a node reached via an exact anchor hit is the symbol the
    // user literally named. It must anchor the PPR walk above any semantically-near
    // KNN neighbour — otherwise a high cosine×kind_boost weight (e.g. method:TestJig.Run
    // for query "type:Tweak") outweighs the exact match in the personalisation vector
    // and PPR ranks the neighbour first. Float every Exact seed's weight just above the
    // strongest non-exact seed so PPR concentrates mass on the named symbol and pulls
    // ITS neighbours in as context. No-op when there is no exact match (pure-NL queries).
    // RFC-019 cosine-scaled teleportation: when the oracle is confident, scale each
    // non-exact seed's PPR weight by its true cosine so teleportation mass tracks
    // semantic proximity. A 0.46-cosine seed can no longer hand itself a PPR restart
    // floor above a strong structural dependency of the 1.00 exact anchor (see the
    // `GetWarningsForPod` evidence). Absent-from-oracle → cosine unknown → weight
    // left unchanged (structural seeds like anchor callees keep their justification).
    // No-op when the oracle is cold/absent.
    let oracle_top = augmented_oracle.values().copied().fold(0.0_f32, f32::max);
    let oracle_confident = oracle_top >= cal.map(SEMANTIC_ORACLE_MIN);
    if oracle_confident {
        for s in seeds.iter_mut() {
            if s.source != SeedSource::Exact {
                if let Some(&cos) = augmented_oracle.get(&s.node) {
                    s.weight *= cos.clamp(0.0, 1.0);
                }
            }
        }
    }

    // #462 WS3 — the set of exact anchors that are genuinely SPECIFIC: the top
    // node of a resolved, high-IDF term. A conceptual query ("how are pending pods
    // assigned to nodes") resolves exact anchors on COINCIDENTAL common words
    // ("pod", "node") whose IDF is low; those must NOT be floated above the
    // semantically-correct KNN gold, or they sink the real answer down the ranking
    // (the "gold surfaces but ranks low" residual). A literal query on a high-IDF
    // named symbol stays specific and keeps its float. Two literal edges ARE narrowed
    // (ranking only — the anchor is never dropped, just not floated): a mid-IDF anchor
    // (idf_w in [0.15, idf_min)) and the secondary (non-`top_node`) exact anchors of a
    // high-IDF term. `g1_bypass` (rare named symbol) is always specific.
    let specific_anchor_ids: std::collections::HashSet<NodeId> = terms
        .iter()
        .filter(|t| t.resolved && t.idf_w >= idf_min)
        .filter_map(|t| t.top_node)
        .collect();

    let max_non_exact = seeds
        .iter()
        .filter(|s| s.source != SeedSource::Exact)
        .map(|s| s.weight)
        .fold(0.0_f32, f32::max);
    if max_non_exact > 0.0 {
        let floor = max_non_exact * EXACT_SEED_PRIORITY;
        // Fix 2: when the oracle is confident, only float an exact anchor the
        // embedding ALSO places near the query. A generic English word that
        // happens to be an exact symbol name (e.g. the kubelet `RECONCILE`
        // constant matched for "HorizontalPodAutoscaler reconcile loop") must not
        // outrank the semantically-correct neighbour (`reconcileAutoscaler`).
        // No-op when the oracle is cold/absent — `agrees` is vacuously true.
        let cos_floor =
            (oracle_top - cal.map_delta(SEMANTIC_REL_DELTA)).max(cal.map(SEMANTIC_ABS_FLOOR));
        for s in seeds.iter_mut() {
            if s.source == SeedSource::Exact {
                // WS3: coincidental low-IDF anchors don't float on conceptual queries.
                let specific = g1_bypass || specific_anchor_ids.contains(&s.node);
                let agrees = !oracle_confident
                    || augmented_oracle.get(&s.node).copied().unwrap_or(0.0) >= cos_floor;
                if specific && agrees {
                    s.weight = s.weight.max(floor);
                }
            }
        }
    }

    // G1: confidence will be decided by the deterministic lexical path
    // regardless of what the reranker says, so the (NL-trained) reranker's
    // opinion is unused: skip inference (RFC-021 F2). `g1_bypass` was already
    // computed above, before semantic_validate ran, and is reused here as-is
    // — recomputing it from the post-semantic_validate `exact_anchor_ids`
    // would reintroduce the bug this fix closes (semantic_validate silently
    // removing the seed G1 was supposed to protect, before G1 ever saw it).

    // ── RFC-021: cross-encoder rerank ────────────────────────────────────────
    // Reranks the top-K fused candidates against the query text, attaches
    // Seed.rerank_score, and reorders (exact anchors keep structural priority
    // — they are the literally-named symbol; rerank orders everything else).
    // `max_rerank_score` (None = reranker had no opinion for this query — see
    // classify_confidence's fallback arm) feeds Phase 3's confidence gate. A
    // no-op — every seed keeps `rerank_score: None`, original ordering, and
    // `max_rerank_score` stays `None` — when the reranker is absent, disabled,
    // degraded for this query, or bypassed by G1, so behaviour is byte-for-byte
    // unchanged without it.
    let mut max_rerank_score: Option<f32> = None;
    if !seeds.is_empty() && !g1_bypass {
        let topk = crate::rerank::rerank_topk();
        // A stale root resolves to `Some(gone)` rather than `None` here, so the
        // snippet reads fail and the cross-encoder scores candidates on empty
        // text: search quality degrades with nothing surfaced (#749 review).
        let repo_root = store.resolve_repo_root();
        let take_n = seeds.len().min(topk);
        let candidate_ids: Vec<NodeId> = seeds[..take_n].iter().map(|s| s.node).collect();
        if let Ok(candidate_nodes) = store.get_nodes(&candidate_ids) {
            let node_by_id: HashMap<NodeId, &CoreNode> =
                candidate_nodes.iter().map(|n| (n.id, n)).collect();
            // Signature-first: truncation (travsr-rerank, OnlySecond) drops from
            // the skeleton tail, never the signature. RFC-021 F5: cap the read at
            // ~10 lines instead of the kind-default (up to 40) — travsr-rerank
            // truncates candidate text to MAX_CANDIDATE_CHARS (480, ~10 lines at
            // ~48 chars/line) before tokenization, so lines past that are always
            // discarded; reading fewer lines here saves the per-candidate
            // collection/format work without changing what the model sees.
            const RERANK_SNIPPET_LINES: usize = 10;
            // RFC-022 prototype: fetch the candidates' doc/skeleton prose so it can be
            // prepended to the cross-encoder text (see `rerank_doc_enabled`). Only
            // queried when the flag is on, so the default path is unchanged.
            let doc_texts: HashMap<NodeId, String> = if rerank_doc_enabled() {
                store.get_embed_texts(&candidate_ids).unwrap_or_default()
            } else {
                HashMap::new()
            };
            let texts: Vec<String> = candidate_ids
                .iter()
                .map(|id| match node_by_id.get(id) {
                    Some(n) => {
                        let sig = &n.vname.signature;
                        let body = repo_root.as_deref().and_then(|root| {
                            travsr_analysis::snippet::snippet_for_node_capped(
                                n,
                                root,
                                RERANK_SNIPPET_LINES,
                            )
                        });
                        // Doc prose (prototype): placed AFTER the signature and BEFORE
                        // the body so `OnlySecond` tail-truncation drops the body first
                        // and preserves the NL doc the model needs for conceptual queries.
                        let doc = doc_texts
                            .get(id)
                            .map(|t| rerank_doc_text(t))
                            .filter(|d| !d.is_empty());
                        match (doc, body) {
                            (Some(d), Some(b)) => format!("{sig}\n{d}\n{b}"),
                            (Some(d), None) => format!("{sig}\n{d}"),
                            (None, Some(b)) => format!("{sig}\n{b}"),
                            (None, None) => sig.clone(),
                        }
                    }
                    None => String::new(),
                })
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

            if let Some(scores) = crate::rerank::rerank(query, &text_refs) {
                for (seed, score) in seeds[..take_n].iter_mut().zip(scores) {
                    seed.rerank_score = Some(score);
                }
                sort_seeds_post_rerank(&mut seeds, rerank_weak_floor());
                max_rerank_score = Some(
                    seeds
                        .iter()
                        .filter_map(|s| s.rerank_score)
                        .fold(f32::NEG_INFINITY, f32::max),
                );
            }
        }
    }

    // RFC-022 D2 (keystone): fold the cross-encoder's judgment into the PPR
    // personalisation weight so the reranker DRIVES node selection, not just seed
    // ordering (RC-2). Only non-exact seeds are scaled — exact anchors keep the
    // structural float applied above; un-reranked seeds (`rerank_score: None`, e.g.
    // beyond the reranked top-K) are untouched. Runs after the rerank block so every
    // scored seed already carries its `rerank_score`. Affects ONLY PPR ordering, not
    // the confidence gate below (G2 preserved). Inert without a reranker or on
    // g1_bypass (no `rerank_score` set), and behind `TRAVSR_RERANK_PPR_WEIGHT`.
    if rerank_ppr_weight_enabled() {
        let weak_floor = rerank_weak_floor();
        for s in seeds.iter_mut() {
            if s.source != SeedSource::Exact {
                if let Some(r) = s.rerank_score {
                    s.weight *= rerank_ppr_multiplier(r, weak_floor);
                }
            }
        }
    }

    let confidence = classify_confidence(
        &terms,
        coverage,
        top_bm25,
        !seeds.is_empty(),
        has_knn,
        knn_oracle,        // cluster signals: KNN neighbours only
        &augmented_oracle, // per-anchor agreement: neighbours ∪ scored anchors
        &scored_ids,
        &cal,
        g1_bypass, // G1: rare exact-symbol queries stay deterministic
        max_rerank_score,
    );

    // #462 WS4: anchor-gated rerank recall-floor rescue. `early_exact_ids` is the
    // exact-anchor set that fed RRF — every id in it came from a token that cleared the
    // `idf_w >= 0.15` + `token_is_sig_component` anchor-emit gates, so a non-empty set
    // means the query is grounded by at least one genuine, specific lexical anchor.
    // (Checking a term's *raw* `top_node` instead is wrong: #463 reordering and the
    // noise/component filters mean the emitted anchor for "daemon" — impl/struct:Daemon
    // — is a different node than the raw FTS #1 match, so that check reads false even
    // when the query is clearly anchored.) Rescues an otherwise-`None` prose query so
    // grounded when the cross-encoder still sees moderate relevance. See
    // [`anchor_rescued_confidence`].
    let confidence = if anchor_rescue_enabled() {
        let exact_anchor_present = !early_exact_ids.is_empty();
        // RFC-022 D5: gate the rescue on non-zero grounded coverage. `n_resolved`
        // counts only tokens clearing the IDF-coverage bar, so `coverage_ok` is
        // false for an all-generic query (`get map handle`) that emits anchors but
        // resolves no specific token — those must stay abstained.
        let coverage_ok = n_resolved >= 1;
        anchor_rescued_confidence(
            confidence,
            g1_bypass,
            max_rerank_score,
            exact_anchor_present,
            coverage_ok,
            rerank_recall_floor(),
        )
    } else {
        confidence
    };

    // RFC-022 Phase 0: seed-pipeline diagnostic behind `tracing::debug!`
    // (target `travsr::seed`), replacing the temporary `TRAVSR_DEBUG_SEED`
    // `eprintln` RCA block. The per-seed dump needs a `get_nodes` round-trip, so
    // the whole block is gated on `DEBUG` being enabled to keep the hot path free
    // of that cost when tracing is off.
    if tracing::enabled!(target: "travsr::seed", tracing::Level::DEBUG) {
        tracing::debug!(
            target: "travsr::seed",
            ?query, ?content_tokens, coverage, n_resolved, g1_bypass,
            ?max_rerank_score, ?intent, ?confidence,
            "seed pipeline state",
        );
        for t in &terms {
            let anchored = t.top_node.is_some_and(|n| early_exact_ids.contains(&n));
            tracing::debug!(
                target: "travsr::seed",
                token = %t.token, resolved = t.resolved, idf = t.idf_w,
                freq = t.symbol_freq, anchored, "term",
            );
        }
        let top_ids: Vec<NodeId> = seeds.iter().take(10).map(|s| s.node).collect();
        if let Ok(ns) = store.get_nodes(&top_ids) {
            let by: HashMap<NodeId, &CoreNode> = ns.iter().map(|n| (n.id, n)).collect();
            for s in seeds.iter().take(10) {
                if let Some(n) = by.get(&s.node) {
                    tracing::debug!(
                        target: "travsr::seed",
                        kind = %n.kind, sig = %n.vname.signature,
                        rerank = ?s.rerank_score, source = ?s.source,
                        path = %n.vname.path, "seed",
                    );
                }
            }
        }
    }

    SeedSet {
        intent,
        seeds,
        terms,
        coverage,
        confidence,
        top_bm25,
    }
}

// ── `travsr explain` diagnostic (#478 RFC-023 §6.1, WS-8) ────────────────────

impl SeedSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Lexical => "lexical",
            Self::Knn => "knn",
        }
    }
}

/// Per-query-token report: `travsr explain`'s view of the anchor loop.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExplainToken {
    pub token: String,
    pub symbol_freq: usize,
    pub idf_w: f32,
    pub resolved: bool,
    pub is_anchor_emit: bool,
    pub top_node_signature: Option<String>,
    /// #540: anchors this token actually emitted, measured rather than
    /// derived. `min(symbol_freq, MAX_ANCHORS_PER_TOKEN)` was an upper bound
    /// the loop reduces further, so reporting it attributed noise, RBAC and
    /// boundary drops to capacity.
    pub anchors_emitted: usize,
}

/// One leg's raw (pre-fusion) rank + score for the explained node, from
/// [`travsr_store::ExplainLegs`]. Absent when that leg did not match at all.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExplainLeg {
    pub rank: usize,
    pub raw_score: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ExplainLegMatches {
    pub exact: Option<ExplainLeg>,
    pub word: Option<ExplainLeg>,
    pub trigram: Option<ExplainLeg>,
    pub l2a: Option<ExplainLeg>,
    pub embed: Option<ExplainLeg>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExplainThresholds {
    pub idf_coverage_min: f32,
    pub anchor_emit_cut: f32,
    pub bm25_strong_floor: f32,
    pub scope_strong_floor: f32,
}

/// The explained node's outcome for one `build_seed_set` run (live, or the
/// FTS-only counterfactual).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExplainDisposition {
    pub in_seed_set: bool,
    pub seed_rank: Option<usize>,
    pub weight: Option<f32>,
    pub source: Option<&'static str>,
    pub rerank_score: Option<f32>,
    pub confidence: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExplainReport {
    pub query: String,
    pub symbol: String,
    /// Whether `symbol` resolved to a node in the graph at all — distinct
    /// from whether that node made it into either seed set.
    pub node_found: bool,
    pub target_signature: Option<String>,
    pub target_path: Option<String>,
    pub tokens: Vec<ExplainToken>,
    pub thresholds: ExplainThresholds,
    /// `None` when `symbol` did not resolve to a node.
    pub legs: Option<ExplainLegMatches>,
    pub is_noise: bool,
    pub oracle_cosine: Option<f32>,
    pub live: ExplainDisposition,
    pub fts_only: ExplainDisposition,
}

/// Per-query-token IDF, per-leg match, every gate + threshold, and final
/// disposition — including the FTS-only counterfactual — for one
/// query/symbol pair. Built on top of [`build_seed_set`] and
/// [`SqliteStore::explain_leg_scores`] rather than threading a collector
/// through either: both already compute (or make available) everything this
/// needs, so a diagnostic-only caller does not add any cost or risk to the
/// hot query path (RFC-023 §6.1/§10 — zero cost when not invoked, since this
/// function is only ever called by `travsr explain` itself).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn explain_seed_set(
    store: &SqliteStore,
    query: &str,
    symbol: &str,
    filter: &dyn EdgeFilter,
    knn_pairs: Vec<(CoreNode, f32)>,
    knn_oracle: &HashMap<NodeId, f32>,
    score_fn: Option<&dyn Fn(&str, &[NodeId]) -> Vec<(NodeId, f32)>>,
) -> ExplainReport {
    let normalized_query = travsr_store::fts_tokenize::normalize_nl_query(query);
    let query = normalized_query.as_str();

    // Resolve the target symbol to a node: prefer an exact signature match
    // (mirrors the G1 fast-path's own equality check) over the first
    // substring hit, since `search_nodes_by_name` is substring-based.
    let candidates = store.search_nodes_by_name(symbol).unwrap_or_default();
    let target = candidates
        .iter()
        .find(|n| n.vname.signature == symbol)
        .or_else(|| candidates.first())
        .cloned();
    let target_id = target.as_ref().map(|n| n.id);

    let live_set = build_seed_set(store, query, filter, knn_pairs, knn_oracle, score_fn);
    // FTS-only counterfactual (RFC-023 §11 AC #1, §6.1): the same query with
    // embeddings entirely absent, so `explain` can show whether the live
    // result depends on the embed backstop the RFC exists to stop relying on.
    let empty_oracle: HashMap<NodeId, f32> = HashMap::new();
    let fts_only_set = build_seed_set(store, query, filter, Vec::new(), &empty_oracle, None);

    let tokens: Vec<ExplainToken> = live_set
        .terms
        .iter()
        .map(|t| ExplainToken {
            token: t.token.clone(),
            symbol_freq: t.symbol_freq,
            idf_w: t.idf_w,
            resolved: t.resolved,
            is_anchor_emit: t.idf_w >= anchor_emit_cut(),
            anchors_emitted: t.anchors_emitted,
            top_node_signature: t
                .top_node
                .and_then(|id| store.get_node(id).ok().flatten())
                .map(|n| n.vname.signature),
        })
        .collect();

    let thresholds = ExplainThresholds {
        idf_coverage_min: idf_coverage_min(),
        anchor_emit_cut: anchor_emit_cut(),
        bm25_strong_floor: bm25_strong_floor(),
        scope_strong_floor: scope_strong_floor(),
    };

    let (legs, is_noise, oracle_cosine) = match &target {
        Some(node) => {
            let empty_legs = || travsr_store::ExplainLegs {
                exact: Vec::new(),
                word: Vec::new(),
                trigram: Vec::new(),
                l2a: Vec::new(),
                embed: Vec::new(),
            };
            let leg_scores = store
                .explain_leg_scores(query, None)
                .unwrap_or_else(|_| empty_legs());
            let find = |leg: &[(CoreNode, f32)]| -> Option<ExplainLeg> {
                leg.iter()
                    .position(|(n, _)| n.id == node.id)
                    .map(|rank| ExplainLeg {
                        rank,
                        raw_score: leg[rank].1,
                    })
            };
            let matches = ExplainLegMatches {
                exact: find(&leg_scores.exact),
                word: find(&leg_scores.word),
                trigram: find(&leg_scores.trigram),
                l2a: find(&leg_scores.l2a),
                embed: find(&leg_scores.embed),
            };
            (
                Some(matches),
                is_noise_seed(node),
                knn_oracle.get(&node.id).copied(),
            )
        }
        None => (None, false, None),
    };

    let disposition_for = |set: &SeedSet| -> ExplainDisposition {
        let seed =
            target_id.and_then(|id| set.seeds.iter().enumerate().find(|(_, s)| s.node == id));
        ExplainDisposition {
            in_seed_set: seed.is_some(),
            seed_rank: seed.as_ref().map(|(rank, _)| *rank),
            weight: seed.as_ref().map(|(_, s)| s.weight),
            source: seed.as_ref().map(|(_, s)| s.source.label()),
            rerank_score: seed.as_ref().and_then(|(_, s)| s.rerank_score),
            confidence: set.confidence.label(),
        }
    };

    ExplainReport {
        query: query.to_string(),
        symbol: symbol.to_string(),
        node_found: target.is_some(),
        target_signature: target.as_ref().map(|n| n.vname.signature.clone()),
        target_path: target.as_ref().map(|n| n.vname.path.clone()),
        tokens,
        thresholds,
        legs,
        is_noise,
        oracle_cosine,
        live: disposition_for(&live_set),
        fts_only: disposition_for(&fts_only_set),
    }
}

// ── Abstention message builder ────────────────────────────────────────────────

/// Build the abstention response for `Confidence::None` queries.
///
/// Returns a clear signal to the AI: no grounded match found, here's what we tried,
/// here are closest guesses (if any).
pub(crate) fn abstain_message(seed_set: &SeedSet, query: &str) -> String {
    let max_guesses = abstain_max_guesses();

    let mut resolved_lines: Vec<String> = Vec::new();
    let mut unresolved: Vec<&str> = Vec::new();
    for term in &seed_set.terms {
        if term.resolved {
            resolved_lines.push(format!(
                "  \"{}\" → {} candidates",
                term.token, term.symbol_freq
            ));
        } else {
            unresolved.push(&term.token);
        }
    }

    let mut msg = format!(
        "[note: no grounded match for this query in this repo]\n\
         query: \"{query}\"\n"
    );

    if !resolved_lines.is_empty() {
        msg.push_str("resolved:\n");
        for line in &resolved_lines {
            msg.push_str(line);
            msg.push('\n');
        }
    }
    if !unresolved.is_empty() {
        msg.push_str(&format!("not found: {}\n", unresolved.join(", ")));
    }

    if max_guesses > 0 && !seed_set.seeds.is_empty() {
        msg.push_str(&format!(
            "\n[The core of your query isn't in this repo. \
             Showing closest {} guess(es) only, treat these as speculative.]\n",
            seed_set.seeds.len().min(max_guesses)
        ));
    }

    msg
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── #463 logic-bearing exact-anchor ordering ─────────────────────────────

    /// Build an exact-anchor candidate node with a known kind and span.
    fn anchor_node(kind: &str, sig: &str, line: u32, end_line: u32) -> CoreNode {
        CoreNode::new(
            travsr_core::VName::new("corp", "", "crates/x/src/lib.rs", "rust", sig),
            kind,
        )
        .with_line(line)
        .with_end_line(end_line)
    }

    #[test]
    fn kind_logic_rank_orders_body_then_container_then_leaf() {
        for k in ["method", "function", "constructor", "impl"] {
            assert_eq!(kind_logic_rank(k), 0, "{k} should be body-bearing");
        }
        for k in [
            "struct",
            "class",
            "enum",
            "trait",
            "interface",
            "type",
            "module",
        ] {
            assert_eq!(kind_logic_rank(k), 1, "{k} should be a container/type");
        }
        for k in ["field", "var", "constant", "static", "import"] {
            assert_eq!(kind_logic_rank(k), 2, "{k} should be a leaf");
        }
        // The strict ordering the reorder relies on.
        assert!(kind_logic_rank("method") < kind_logic_rank("struct"));
        assert!(kind_logic_rank("struct") < kind_logic_rank("field"));
    }

    #[test]
    fn span_size_is_line_count_or_zero() {
        assert_eq!(span_size(&anchor_node("impl", "impl:Daemon", 10, 40)), 30);
        // Missing lines and inverted spans collapse to 0 — never panic / underflow.
        let no_lines = CoreNode::new(
            travsr_core::VName::new("corp", "", "p", "rust", "fn:x"),
            "function",
        );
        assert_eq!(span_size(&no_lines), 0);
        assert_eq!(span_size(&anchor_node("function", "fn:x", 40, 10)), 0);
    }

    #[test]
    fn order_anchor_candidates_prefers_logic_bearing_over_field_only() {
        // FTS returns the field-only struct FIRST (closer literal name match) and the
        // logic-bearing impl/method after — exactly the `Daemon` case from #463.
        let nodes = [
            anchor_node("struct", "struct:Daemon", 1, 3),
            anchor_node("impl", "impl:Daemon", 5, 60),
            anchor_node("method", "fn:Daemon.run", 20, 55),
        ];
        let ordered = order_anchor_candidates(&nodes.iter().collect::<Vec<_>>(), 6);
        // Body-bearing definitions now lead (largest-span first); the struct sinks last.
        assert_eq!(ordered[0].kind, "impl");
        assert_eq!(ordered[1].kind, "method");
        assert_eq!(ordered[2].kind, "struct");
    }

    #[test]
    fn order_anchor_candidates_breaks_kind_ties_by_larger_span() {
        let nodes = [
            anchor_node("function", "fn:small", 10, 15), // span 5
            anchor_node("function", "fn:big", 10, 90),   // span 80
        ];
        let ordered = order_anchor_candidates(&nodes.iter().collect::<Vec<_>>(), 6);
        assert_eq!(ordered[0].vname.signature, "fn:big");
        assert_eq!(ordered[1].vname.signature, "fn:small");
    }

    #[test]
    fn order_anchor_candidates_is_stable_within_equal_keys() {
        // Two leaves with identical (rank, span): FTS order must be preserved.
        let nodes = [
            anchor_node("field", "field:a", 1, 1),
            anchor_node("field", "field:b", 1, 1),
        ];
        let ordered = order_anchor_candidates(&nodes.iter().collect::<Vec<_>>(), 6);
        assert_eq!(ordered[0].vname.signature, "field:a");
        assert_eq!(ordered[1].vname.signature, "field:b");
    }

    #[test]
    fn order_anchor_candidates_respects_window_bound() {
        // A logic-bearing node BELOW the window must not be pulled up — name-match
        // quality (FTS rank) still gates the pool; only the head is reordered.
        let nodes = [
            anchor_node("struct", "struct:Foo", 1, 2),
            anchor_node("field", "field:Foo", 3, 4),
            anchor_node("field", "field:Bar", 5, 6),
            anchor_node("method", "fn:Foo.run", 7, 40), // past the window-3 head
        ];
        let ordered = order_anchor_candidates(&nodes.iter().collect::<Vec<_>>(), 3);
        assert_eq!(ordered.len(), 3, "window bounds the reordered set");
        assert!(
            ordered.iter().all(|n| n.kind != "method"),
            "the out-of-window method must not be pulled into the head"
        );
        // Within the window the container still leads the two fields.
        assert_eq!(ordered[0].kind, "struct");
    }

    #[test]
    fn anchor_kind_priority_defaults_on_when_unset() {
        // The knob is a rollback escape hatch: default (unset) is ON. The test harness
        // does not set TRAVSR_ANCHOR_KIND_PRIORITY, so this reads the shipped default.
        assert!(anchor_kind_priority());
    }

    // ── #462 WS4 anchor-gated rerank recall-floor rescue ─────────────────────

    #[test]
    fn ws4_rescues_grounded_prose_query_below_weak_floor() {
        // "how daemon is implemented": genuine exact anchor + rerank 0.346 (below the
        // 0.5 weak floor but far above salad noise) → surface as Weak, not abstain.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.346), true, true, 0.15),
            Confidence::Weak,
        );
    }

    #[test]
    fn ws4_does_not_rescue_salad_below_recall_floor() {
        // "delete all user accounts and drop the database": exact anchors DO exist
        // (drop/delete/user), but the reranker scores its best seed 0.0012 — far below
        // the recall floor. The rerank floor, not the anchor, is what keeps salad None.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.0012), true, true, 0.15),
            Confidence::None,
        );
    }

    #[test]
    fn ws4_requires_a_genuine_exact_anchor() {
        // Moderate rerank but no lexical grounding → stay None (that path is WS2's job,
        // via the embedding oracle, not WS4's).
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.4), false, true, 0.15),
            Confidence::None,
        );
    }

    #[test]
    fn ws4_requires_nonzero_coverage() {
        // RFC-022 D5 (RC-5): `get map handle` emits exact anchors on generic tokens but
        // resolves no IDF-coverage-clearing token (`coverage_ok = false`), so it must
        // stay None even with a moderate rerank score above the recall floor.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.4), true, false, 0.15),
            Confidence::None,
        );
        // With coverage restored (a specific token resolved), the same inputs rescue.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.4), true, true, 0.15),
            Confidence::Weak,
        );
    }

    #[test]
    fn ws4_never_fires_on_g1_bypass() {
        // g1_bypass owns the deterministic lexical verdict; WS4 must not second-guess it.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, true, Some(0.9), true, true, 0.15),
            Confidence::None,
        );
    }

    #[test]
    fn ws4_inert_without_a_rerank_score() {
        // No cross-encoder signal (model absent / g1_bypass skip / breaker-skipped) → the
        // rescue reads no relevance evidence and leaves the verdict untouched.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, None, true, true, 0.15),
            Confidence::None,
        );
    }

    #[test]
    fn ws4_never_downgrades_or_promotes_a_decided_verdict() {
        // Only `None` is rescued; every already-decided verdict passes through unchanged
        // (WS4 can never promote to Strong nor demote a real hit).
        for base in [Confidence::Weak, Confidence::Strong, Confidence::Exact] {
            assert_eq!(
                anchor_rescued_confidence(base, false, Some(0.05), true, true, 0.15),
                base,
                "{base:?} must pass through WS4 unchanged",
            );
        }
    }

    #[test]
    fn ws4_boundary_is_inclusive_at_the_recall_floor() {
        // A score exactly at the floor rescues; a hair below stays None.
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.15), true, true, 0.15),
            Confidence::Weak,
        );
        assert_eq!(
            anchor_rescued_confidence(Confidence::None, false, Some(0.1499), true, true, 0.15),
            Confidence::None,
        );
    }

    #[test]
    fn ws4_defaults_on_with_a_sane_recall_floor() {
        // The knob is a rollback escape hatch: default (unset) is ON, and the floor sits
        // in the measured gap between salad noise (~1e-3) and the grounded band (~0.30).
        assert!(anchor_rescue_enabled());
        let f = rerank_recall_floor();
        assert!(
            f > 0.01 && f < rerank_weak_floor(),
            "recall floor {f} out of band"
        );
    }

    // ── RFC-022 D4: g1_bypass subject-dominance ──────────────────────────────

    fn rt(token: &str, freq: usize, idf: f32, node: Option<NodeId>) -> ResolvedTerm {
        ResolvedTerm {
            token: token.to_string(),
            resolved: node.is_some(),
            symbol_freq: freq,
            idf_w: idf,
            top_node: node,
            anchors_emitted: 0,
        }
    }

    #[test]
    fn g1_bypass_fires_on_single_rare_subject() {
        // A lone rare exact anchor owns 100% of the query IDF → bypass (literal lookup).
        let anchors: std::collections::HashSet<NodeId> = [NodeId(7)].into_iter().collect();
        let terms = vec![rt("getwarningsforpod", 1, 0.95, Some(NodeId(7)))];
        assert!(g1_bypass_decision(
            &terms,
            &TrustedAnchors {
                exact: &anchors,
                corrected: &std::collections::HashSet::new()
            },
            3,
            0.55
        ));
    }

    #[test]
    fn g1_bypass_fires_on_all_symbol_literal_query() {
        // Two rare exact anchors, no filler: anchors carry all the IDF → bypass.
        // (Protects "SqliteStore search_nodes_by_name"-style literal lookups.)
        let anchors: std::collections::HashSet<NodeId> =
            [NodeId(1), NodeId(2)].into_iter().collect();
        let terms = vec![
            rt("sqlitestore", 2, 0.9, Some(NodeId(1))),
            rt("search_nodes_by_name", 1, 0.92, Some(NodeId(2))),
        ];
        assert!(g1_bypass_decision(
            &terms,
            &TrustedAnchors {
                exact: &anchors,
                corrected: &std::collections::HashSet::new()
            },
            3,
            0.55
        ));
    }

    #[test]
    fn g1_bypass_suppressed_for_incidental_rare_word() {
        // "how are results trimmed to fit a token budget": only "trimmed" is a rare
        // anchor; the surrounding conceptual tokens carry enough IDF that the anchor
        // share falls below the subject bar → no bypass (RC-4), so the query reaches
        // the reranker/recall path where `knapsack` can surface.
        let anchors: std::collections::HashSet<NodeId> = [NodeId(9)].into_iter().collect();
        let terms = vec![
            rt("results", 4000, 0.05, None),
            rt("trimmed", 3, 0.85, Some(NodeId(9))),
            rt("fit", 4000, 0.05, None),
            rt("token", 400, 0.35, None),
            rt("budget", 300, 0.4, None),
        ];
        assert!(!g1_bypass_decision(
            &terms,
            &TrustedAnchors {
                exact: &anchors,
                corrected: &std::collections::HashSet::new()
            },
            3,
            0.55
        ));
    }

    #[test]
    fn g1_bypass_requires_the_anchor_to_be_emitted() {
        // A rare resolved term whose top_node was NOT emitted as an exact anchor
        // (filtered as noise / not a signature component) can never bypass.
        let anchors: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let terms = vec![rt("checkout", 2, 0.9, Some(NodeId(5)))];
        assert!(!g1_bypass_decision(
            &terms,
            &TrustedAnchors {
                exact: &anchors,
                corrected: &std::collections::HashSet::new()
            },
            3,
            0.55
        ));
    }

    #[test]
    fn g1_bypass_fires_on_corrected_anchor_despite_non_rare_frequency() {
        // #709: a unique strict typo correction owns the whole query IDF and is
        // the query's subject even though the corrected symbol is not "rare"
        // (freq 5 > rare_max 3). Its emitted anchor is in corrected_anchor_ids, so
        // g1 fires and the NL reranker cannot veto the misspelt-but-deterministic
        // lookup. Mirrors the `htpresponse` → `HttpResponse` grounding.
        let anchors: std::collections::HashSet<NodeId> = [NodeId(7)].into_iter().collect();
        let corrected: std::collections::HashSet<NodeId> = [NodeId(7)].into_iter().collect();
        let terms = vec![rt("htpresponse", 5, 0.74, Some(NodeId(7)))];
        assert!(
            g1_bypass_decision(
                &terms,
                &TrustedAnchors {
                    exact: &anchors,
                    corrected: &corrected
                },
                3,
                0.55
            ),
            "corrected anchor must drive the g1 bypass"
        );
        // Without the correction flag the same non-rare anchor must NOT bypass.
        assert!(
            !g1_bypass_decision(
                &terms,
                &TrustedAnchors {
                    exact: &anchors,
                    corrected: &std::collections::HashSet::new()
                },
                3,
                0.55
            ),
            "a non-rare, uncorrected anchor must not bypass"
        );
    }

    #[test]
    fn g1_bypass_corrected_still_needs_subject_share() {
        // A correction that is only an incidental fragment of a larger query does
        // NOT trigger the bypass, the subject-share test still applies, so a
        // corrected word buried in conceptual filler flows to the reranker path.
        let anchors: std::collections::HashSet<NodeId> = [NodeId(9)].into_iter().collect();
        let corrected: std::collections::HashSet<NodeId> = [NodeId(9)].into_iter().collect();
        let terms = vec![
            rt("how", 4000, 0.05, None),
            rt("does", 4000, 0.05, None),
            rt("the", 4000, 0.05, None),
            rt("htpresponse", 5, 0.30, Some(NodeId(9))),
            rt("renderer", 300, 0.40, None),
            rt("stream", 300, 0.40, None),
        ];
        assert!(!g1_bypass_decision(
            &terms,
            &TrustedAnchors {
                exact: &anchors,
                corrected: &corrected
            },
            3,
            0.55
        ));
    }

    // ── RFC-022 D2: rerank-weighted PPR multiplier ───────────────────────────

    #[test]
    fn rerank_ppr_multiplier_anchors_at_floor_neutral_and_ceil() {
        let wf = 0.5;
        // Junk (rr 0) → crushed to the floor.
        assert!((rerank_ppr_multiplier(0.0, wf) - 0.1).abs() < 1e-6);
        // At the weak floor → neutral (×1.0).
        assert!((rerank_ppr_multiplier(wf, wf) - 1.0).abs() < 1e-6);
        // Perfect (rr 1) → ceiling.
        assert!((rerank_ppr_multiplier(1.0, wf) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rerank_ppr_multiplier_is_monotone_and_clamped() {
        let wf = 0.5;
        let mut prev = rerank_ppr_multiplier(0.0, wf);
        let mut r = 0.05;
        while r <= 1.0 {
            let m = rerank_ppr_multiplier(r, wf);
            assert!(
                m + 1e-6 >= prev,
                "multiplier must be monotone non-decreasing"
            );
            assert!(
                (0.1..=2.0).contains(&m),
                "multiplier must stay in [FLOOR, CEIL]"
            );
            prev = m;
            r += 0.05;
        }
        // Out-of-range inputs clamp rather than explode.
        assert!((rerank_ppr_multiplier(-1.0, wf) - 0.1).abs() < 1e-6);
        assert!((rerank_ppr_multiplier(5.0, wf) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rerank_ppr_weight_defaults_on() {
        // Rollback escape hatch: default (unset) is ON.
        assert!(rerank_ppr_weight_enabled());
    }

    // ── RFC-022 prototype: doc-aware reranker candidate ──────────────────────

    #[test]
    fn rerank_doc_text_extracts_prose_or_empty() {
        assert_eq!(
            rerank_doc_text(
                "function: fn:ppr_weighted | calls: ppr | doc: Personalized PageRank on seeds"
            ),
            "Personalized PageRank on seeds",
        );
        // No doc segment → empty (node has no doc comment).
        assert_eq!(rerank_doc_text("function: fn:x | calls: y"), "");
    }

    #[test]
    fn rerank_doc_defaults_off() {
        // Prototype ships gated; default path is unchanged.
        assert!(!rerank_doc_enabled());
    }

    // ── RFC-022 §14: shared node→strongest-source classifier ─────────────────

    fn seed_with(node: NodeId, source: SeedSource) -> Seed {
        Seed {
            node,
            weight: 1.0,
            source,
            score: 0.0,
            rerank_score: None,
        }
    }

    #[test]
    fn strongest_seed_sources_exact_wins_over_weaker_sources() {
        // A node reached by an exact anchor AND a weaker source must classify as
        // Exact — this is the invariant get_context and ask both key off, so the
        // two surfaces cannot disagree on a node's match-source bucket.
        let seeds = vec![
            seed_with(NodeId(1), SeedSource::Knn),
            seed_with(NodeId(1), SeedSource::Exact), // same node, stronger source
            seed_with(NodeId(1), SeedSource::Lexical),
            seed_with(NodeId(2), SeedSource::Lexical),
            seed_with(NodeId(3), SeedSource::Knn),
        ];
        let m = strongest_seed_sources(&seeds);
        assert_eq!(m.get(&NodeId(1)), Some(&SeedSource::Exact), "exact wins");
        assert_eq!(m.get(&NodeId(2)), Some(&SeedSource::Lexical));
        assert_eq!(m.get(&NodeId(3)), Some(&SeedSource::Knn));
        // Order-independent: reversing the input yields the same strongest map.
        let mut rev = seeds.clone();
        rev.reverse();
        assert_eq!(strongest_seed_sources(&rev), m);

        // The classifier both surfaces then apply: strongest==Exact ⇒ Exact bucket.
        let is_exact = |id: NodeId| matches!(m.get(&id), Some(SeedSource::Exact));
        assert_eq!(match_source(true, is_exact(NodeId(1))), MatchSource::Exact);
        assert_eq!(
            match_source(true, is_exact(NodeId(3))),
            MatchSource::Semantic
        );
        assert_eq!(
            match_source(false, is_exact(NodeId(1))),
            MatchSource::Relevant
        );
    }

    // ── #393 score-aware scope gate ──────────────────────────────────────────

    #[test]
    fn scope_gate_admits_strong_out_of_scope() {
        // Out of the anchor crate scope, but a strong lexical match (norm >= floor)
        // overrides the structural scope — the multi-domain case (#393).
        assert!(!scope_gate_drops(true, false, 0.50, 0.3));
        assert!(!scope_gate_drops(true, false, 0.30, 0.3)); // exactly at floor => kept
    }

    #[test]
    fn scope_gate_drops_weak_out_of_scope() {
        // Weak cross-domain FTS drift stays gated.
        assert!(scope_gate_drops(true, false, 0.10, 0.3));
        assert!(scope_gate_drops(true, false, 0.29, 0.3));
    }

    #[test]
    fn scope_gate_keeps_in_scope_regardless_of_score() {
        assert!(!scope_gate_drops(true, true, 0.01, 0.3));
    }

    #[test]
    fn scope_gate_noop_without_anchor_scope() {
        // No anchors established a scope => nothing is gated (unrestricted).
        assert!(!scope_gate_drops(false, false, 0.01, 0.3));
    }

    #[test]
    fn scope_gate_floor_above_one_restores_hard_gate() {
        // `norm` is clamped to <= 1.0, so a floor > 1.0 (TRAVSR_SCOPE_STRONG_FLOOR=2.0)
        // drops every out-of-scope node — the pre-#393 hard gate.
        for norm in [0.05f32, 0.5, 1.0] {
            assert!(scope_gate_drops(true, false, norm, 2.0));
        }
    }

    #[test]
    fn tokenize_strips_stop_words() {
        let toks = tokenize_query("what calls the payment handler");
        assert!(!toks.contains(&"what".to_string()));
        assert!(!toks.contains(&"the".to_string()));
        assert!(toks.contains(&"calls".to_string()));
        assert!(toks.contains(&"payment".to_string()));
        assert!(toks.contains(&"handler".to_string()));
    }

    /// #478 fix: `tokenize_query` must preserve the token's original case.
    /// Stop-word/length filtering is still case-insensitive ("What" is still
    /// dropped), but a content token like "isPrime" must come back exactly as
    /// written — lowercasing it here would permanently destroy the
    /// lower→upper boundary `ident::segments` needs to split it into
    /// `["is", "prime"]`. See `symbol_frequency_finds_camel_case_query_token`
    /// in `travsr-store` for the downstream consequence this was causing
    /// (a real, unambiguous exact-name query collapsing to total abstention
    /// on the FTS-only path).
    #[test]
    fn tokenize_preserves_original_case() {
        let toks = tokenize_query("What calls isPrime");
        assert!(!toks.contains(&"what".to_string()));
        assert!(
            !toks.contains(&"What".to_string()),
            "stop-word check is case-insensitive"
        );
        assert!(toks.contains(&"isPrime".to_string()), "got: {toks:?}");
        assert!(
            !toks.contains(&"isprime".to_string()),
            "must not be lowercased"
        );
    }

    #[test]
    fn tokenize_strips_punctuation_preserves_underscores() {
        let toks = tokenize_query("search_nodes_fuzzy?");
        assert_eq!(toks, vec!["search_nodes_fuzzy"]);
    }

    #[test]
    fn tokenize_filters_short_tokens() {
        let toks = tokenize_query("a b do it");
        // "a", "b", "do" filtered; "it" filtered (stop-word)
        assert!(toks.is_empty(), "got: {toks:?}");
    }

    #[test]
    fn classify_intent_callers() {
        assert_eq!(
            classify_intent("what calls search_nodes_fuzzy"),
            QueryIntent::Callers
        );
        assert_eq!(
            classify_intent("who calls fts_seeds_weighted"),
            QueryIntent::Callers
        );
    }

    #[test]
    fn classify_intent_deps() {
        assert_eq!(
            classify_intent("what does SqliteStore import"),
            QueryIntent::Deps
        );
    }

    #[test]
    fn classify_intent_explain() {
        assert_eq!(
            classify_intent("how does PPR traversal work"),
            QueryIntent::Explain
        );
    }

    #[test]
    fn classify_intent_lookup_default() {
        assert_eq!(classify_intent("get_context"), QueryIntent::Lookup);
    }

    #[test]
    fn idf_weight_specific_token() {
        // Frequency 1 out of 10000 → near 1.0
        let w = idf_weight(1, 10_000);
        assert!(w > 0.9, "specific token should score near 1.0, got {w}");
    }

    #[test]
    fn idf_weight_common_token() {
        // Frequency 5000 out of 10000 (half of corpus) → low weight
        let w = idf_weight(5_000, 10_000);
        assert!(w < 0.3, "common token should score below 0.3, got {w}");
    }

    #[test]
    fn idf_weight_clamped_to_floor() {
        // Frequency >= n_total → clamped to 0.05
        let w = idf_weight(10_000, 10_000);
        assert!((w - 0.05).abs() < 0.01, "should be at floor, got {w}");
    }

    #[test]
    fn rrf_fuse_deterministic() {
        let a: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0), (NodeId(2), 0.8)];
        let b: Vec<(NodeId, f32)> = vec![(NodeId(2), 1.0), (NodeId(3), 0.5)];
        let result1 = rrf_fuse(&[&a, &b], 60.0);
        let result2 = rrf_fuse(&[&a, &b], 60.0);
        assert_eq!(result1, result2, "RRF must be deterministic");
    }

    #[test]
    fn rrf_fuse_weighted_caps_generic_anchor_flood() {
        // RFC-022 D1.4: a generic per-token anchor at rank 0 must not outrank a BM25
        // match at rank 0 once the generic source is down-weighted. `gold` (the
        // compound BM25 match) beats `junk` (a generic single-token anchor).
        let generic_anchor: Vec<(NodeId, f32)> = vec![(NodeId(99), 1.0)]; // junk, rank 0
        let bm25: Vec<(NodeId, f32)> = vec![(NodeId(7), 1.0)]; // gold, rank 0
        let out = rrf_fuse_weighted(&[(0.4, &generic_anchor), (1.0, &bm25)], 60.0);
        assert_eq!(
            out[0].0,
            NodeId(7),
            "down-weighted generic anchor must not lead"
        );
        assert!(out[0].1 > out[1].1);
    }

    #[test]
    fn rrf_fuse_weighted_multi_source_still_rewarded() {
        // A candidate in two weighted sources accumulates both contributions and beats
        // a single-source candidate of equal per-source rank.
        let s1: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0), (NodeId(2), 0.9)];
        let s2: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0)];
        let out = rrf_fuse_weighted(&[(1.0, &s1), (1.0, &s2)], 60.0);
        assert_eq!(out[0].0, NodeId(1), "multi-source candidate must lead");
    }

    #[test]
    fn rrf_weighted_defaults_off() {
        // D1 recall phase ships gated: weighted fusion is opt-in until calibrated.
        assert!(!rrf_weighted_enabled());
    }

    #[test]
    fn rrf_fuse_tie_break_by_node_id() {
        // Two single-source lists with disjoint nodes all at rank 0 → same RRF score;
        // tie must break by NodeId ascending.
        let a: Vec<(NodeId, f32)> = vec![(NodeId(5), 1.0)];
        let b: Vec<(NodeId, f32)> = vec![(NodeId(3), 1.0)];
        let result = rrf_fuse(&[&a, &b], 60.0);
        assert_eq!(
            result[0].0,
            NodeId(3),
            "lower NodeId should come first on equal RRF score"
        );
        assert_eq!(result[1].0, NodeId(5));
    }

    #[test]
    fn rrf_fuse_shared_node_scores_higher() {
        // Node 1 appears in both sources; node 2 and 3 appear in one each.
        let a: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0), (NodeId(2), 0.8)];
        let b: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0), (NodeId(3), 0.8)];
        let result = rrf_fuse(&[&a, &b], 60.0);
        assert_eq!(
            result[0].0,
            NodeId(1),
            "node present in both sources should rank first"
        );
    }

    #[test]
    fn rrf_fuse_empty_source_ignored() {
        let a: Vec<(NodeId, f32)> = vec![(NodeId(1), 1.0)];
        let empty: Vec<(NodeId, f32)> = vec![];
        let result = rrf_fuse(&[&a, &empty], 60.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, NodeId(1));
    }

    #[test]
    fn display_score_covers_seed_fallback_and_expanded_cap() {
        // One reranked seed at 0.85 → that is also the expanded cap.
        let mut rr: HashMap<NodeId, f32> = HashMap::new();
        rr.insert(NodeId(1), 0.85);
        let cap = rr.values().copied().reduce(f32::max);

        // Reranked seed shows its absolute cross-encoder score, NOT the
        // normalized-PPR 1.0 (this is the "always 1.00" artefact F9 kills).
        assert_eq!(
            display_score(NodeId(1), 1.0, &rr, true, cap, Confidence::Strong),
            0.85
        );
        // Primary seed the reranker never scored → its own normalized PPR, uncapped
        // at Strong/Exact confidence.
        assert_eq!(
            display_score(NodeId(2), 0.9, &rr, true, cap, Confidence::Strong),
            0.9
        );
        // Expanded neighbour above the cap is pulled down so it can't outrank its seed.
        assert_eq!(
            display_score(NodeId(3), 1.0, &rr, false, cap, Confidence::Strong),
            0.85
        );
        // Expanded neighbour already below the cap keeps its own score.
        assert_eq!(
            display_score(NodeId(4), 0.3, &rr, false, cap, Confidence::Strong),
            0.3
        );
        // No reranked seeds at all (ships-dark / reranker off) → cap is a no-op.
        let empty: HashMap<NodeId, f32> = HashMap::new();
        assert_eq!(
            display_score(NodeId(5), 0.7, &empty, false, None, Confidence::Exact),
            0.7
        );
    }

    #[test]
    fn display_score_tier_ceiling_caps_unscored_primary_seed_only() {
        let empty: HashMap<NodeId, f32> = HashMap::new();
        // #478: an unscored primary seed (reranker never scored it) at
        // normalized-PPR 1.0 must not display as 1.0 next to weak/none
        // confidence — that is the "1.000 next to confidence: weak" defect.
        assert_eq!(
            display_score(NodeId(1), 1.0, &empty, true, None, Confidence::Weak),
            0.60
        );
        assert_eq!(
            display_score(NodeId(1), 1.0, &empty, true, None, Confidence::None),
            0.40
        );
        // Exact/Strong stay uncapped.
        assert_eq!(
            display_score(NodeId(1), 1.0, &empty, true, None, Confidence::Exact),
            1.0
        );
        // A score already below the tier ceiling is untouched (min() is a no-op).
        assert_eq!(
            display_score(NodeId(1), 0.5, &empty, true, None, Confidence::Weak),
            0.5
        );
        // The reranked branch is never touched by the ceiling, regardless of confidence.
        let mut rr: HashMap<NodeId, f32> = HashMap::new();
        rr.insert(NodeId(2), 0.95);
        assert_eq!(
            display_score(NodeId(2), 1.0, &rr, true, None, Confidence::None),
            0.95
        );
        // TRAVSR_DISPLAY_TIER_CAP=0 restores pre-#478 output byte-for-byte.
        std::env::set_var("TRAVSR_DISPLAY_TIER_CAP", "0");
        assert_eq!(
            display_score(NodeId(1), 1.0, &empty, true, None, Confidence::None),
            1.0
        );
        std::env::remove_var("TRAVSR_DISPLAY_TIER_CAP");
    }

    #[test]
    fn confidence_none_on_empty() {
        let terms: Vec<ResolvedTerm> = vec![];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.0,
            0.0,
            false,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(c, Confidence::None);
    }

    #[test]
    fn calibration_map_is_identity_at_reference() {
        let id = Calibration::IDENTITY;
        for &x in &[0.55_f32, 0.58, 0.64, 0.66, 0.72, 0.77] {
            assert!((id.map(x) - x).abs() < 1e-6, "map({x}) must be identity");
        }
        assert!((id.map_delta(0.13) - 0.13).abs() < 1e-6);
    }

    #[test]
    fn calibration_map_scales_compressed_model() {
        // A model whose answerable band sits lower/narrower than bge-small's.
        let cal = Calibration { lo: 0.45, hi: 0.62 };
        assert!((cal.map(Calibration::REF_LO) - 0.45).abs() < 1e-6);
        assert!((cal.map(Calibration::REF_HI) - 0.62).abs() < 1e-6);
        // Interior floor stays strictly inside the mapped band, monotonic.
        let m = cal.map(0.66);
        assert!(m > 0.45 && m < 0.62);
        // A band-relative width scales by the band ratio: 0.13 * (0.17 / 0.22).
        let expected = 0.13 * (0.62 - 0.45) / (Calibration::REF_HI - Calibration::REF_LO);
        assert!((cal.map_delta(0.13) - expected).abs() < 1e-5);
    }

    #[test]
    fn calibration_recovers_low_cosine_answerable_query() {
        // Regression for the k8s arctic-256 over-abstention (bench L4/C4): a genuinely
        // answerable conceptual query whose whole cosine scale is lower than bge-small's.
        // Top cluster ~0.62 with a coherent near set, no lexical anchor.
        let mut oracle = HashMap::new();
        oracle.insert(NodeId(1), 0.62);
        oracle.insert(NodeId(2), 0.60);
        oracle.insert(NodeId(3), 0.58);
        let scored = std::collections::HashSet::new();

        // Reference (bge-small) floors demand oracle_top >= 0.72 to promote → abstains
        // even though this is the model's *answerable* band. This is the bug.
        let ref_c = classify_confidence_lexical_fallback(
            &[],
            0.0,
            0.0,
            true,
            true,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
        );
        assert_eq!(
            ref_c,
            Confidence::None,
            "absolute bge-small floors over-abstain on a lower-scale model"
        );

        // Auto-calibrated to this model's band → the same cosines clear the mapped
        // promote floor and the query is correctly Strong (not demoted to guesses).
        let cal = Calibration { lo: 0.45, hi: 0.62 };
        let got = classify_confidence_lexical_fallback(
            &[],
            0.0,
            0.0,
            true,
            true,
            &oracle,
            &oracle,
            &scored,
            &cal,
        );
        assert_eq!(
            got,
            Confidence::Strong,
            "model-relative floors recover the answerable query"
        );
    }

    #[test]
    fn confidence_exact_on_rare_anchor_high_coverage() {
        let terms = vec![ResolvedTerm {
            token: "get_context".into(),
            resolved: true,
            symbol_freq: 1,
            idf_w: 0.98, // very rare → specific anchor
            top_node: Some(NodeId(42)),
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.0,
            true,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(c, Confidence::Exact);
    }

    #[test]
    fn confidence_strong_on_good_bm25_high_coverage() {
        let terms = vec![ResolvedTerm {
            token: "ppr".into(),
            resolved: false,
            symbol_freq: 50,
            idf_w: 0.0, // not resolved — idf_w irrelevant
            top_node: None,
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.8,
            2.0,
            true,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(c, Confidence::Strong);
    }

    #[test]
    fn confidence_strong_on_embed_agreed_specific_anchor() {
        // Regression: "type Tweak" → type:Tweak. The anchor recurs across packages
        // (symbol_freq > rare_max, so NOT Exact) but is high-IDF and the confident
        // embed oracle places it right at the query. A confident oracle must PROMOTE
        // an agreed specific anchor to Strong, not demote it to Weak (the pre-fix bug
        // where the only Strong path was gated on `!oracle_confident`).
        let tweak = NodeId(100);
        let terms = vec![
            ResolvedTerm {
                token: "tweak".into(),
                resolved: true,
                symbol_freq: 8, // > rare_max(3) → not a rare anchor
                idf_w: 0.81,    // >= idf_coverage_min(0.55) → specific anchor
                top_node: Some(tweak),
                anchors_emitted: 0,
            },
            ResolvedTerm {
                token: "type".into(),
                resolved: true,
                symbol_freq: 5000,
                idf_w: 0.2,
                top_node: Some(NodeId(200)),
                anchors_emitted: 0,
            },
        ];
        // Confident oracle that AGREES the anchor is near the query (cosine 0.95).
        let oracle = HashMap::from([(tweak, 0.95_f32)]);
        let c = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.4,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Strong,
            "embed-agreed specific anchor at high coverage must be Strong, not Weak"
        );
    }

    #[test]
    fn confidence_weak_on_embed_disagreed_specific_anchor() {
        // Salad guard: a high-IDF anchor the confident oracle places FAR from the
        // query (cosine 0.0 → absent from oracle) must NOT be promoted to Strong.
        // This is the "find code by semantic meaning" → is_semantic_edge case.
        let anchor = NodeId(100);
        let terms = vec![ResolvedTerm {
            token: "semantic".into(),
            resolved: true,
            symbol_freq: 2,
            idf_w: 0.9,
            top_node: Some(anchor),
            anchors_emitted: 0,
        }];
        // Confident oracle whose top hit is a DIFFERENT node — anchor absent (cosine 0).
        let oracle = HashMap::from([(NodeId(999), 0.88_f32)]);
        let c = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.4,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "embed-disagreed anchor must stay Weak even at high coverage"
        );
    }

    #[test]
    fn confidence_rfc019_absent_but_scored_anchor_is_strong() {
        // RFC-019 core case: `PodSpec` resolves `class:PodSpec` (specific, high-IDF)
        // but that node lands OUTSIDE the KNN over-fetch → absent from the oracle.
        // The direct-cosine oracle DID score it (it's in `scored_ids`) and found no
        // stored vector / it's just absent — "unknown", NOT disagreement. Absence
        // must no longer read as cosine 0 → weak; a measured specific anchor at high
        // coverage is Strong.
        let anchor = NodeId(100);
        let terms = vec![ResolvedTerm {
            token: "podspec".into(),
            resolved: true,
            symbol_freq: 8, // recurs across packages → not rare (not Exact)
            idf_w: 0.58,    // >= idf_coverage_min → specific anchor
            top_node: Some(anchor),
            anchors_emitted: 0,
        }];
        // Confident oracle (top 0.79) whose near cluster does NOT contain the anchor,
        // and only one node clears the floor (so semantic_strong can't fire — this
        // Strong must come from the anchor path).
        let oracle = HashMap::from([(NodeId(999), 0.79_f32)]);
        let scored: std::collections::HashSet<NodeId> = [anchor].into_iter().collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.4,
            true,
            true,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Strong,
            "scored-but-absent specific anchor (unknown) must not be penalised → Strong"
        );
    }

    #[test]
    fn confidence_rfc019_absent_and_unscored_preserves_old_weak() {
        // Load-bearing safety property: with NO score hook (`scored_ids` empty — the
        // FTS-only path), an anchor absent from the confident oracle keeps the OLD
        // absent⇒disagree semantics → Weak, byte-for-byte identical to pre-RFC-019.
        // Identical inputs to the test above, only `scored_ids` differs.
        let anchor = NodeId(100);
        let terms = vec![ResolvedTerm {
            token: "podspec".into(),
            resolved: true,
            symbol_freq: 8,
            idf_w: 0.58,
            top_node: Some(anchor),
            anchors_emitted: 0,
        }];
        let oracle = HashMap::from([(NodeId(999), 0.79_f32)]);
        let c = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.4,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "without a score hook, absent anchor keeps old absent⇒disagree → Weak"
        );
    }

    #[test]
    fn confidence_rfc019_low_coverage_answerable_band_gates_confirm() {
        // Measured "find code by semantic meaning" shape: the lone resolved word
        // "semantic" is also a symbol name, so the direct-cosine oracle MEASURES it
        // at ≈0.615 — a lexical-overlap-inflated cosine that clears the weak floor
        // (0.55) but sits in the nonsense band, below the answerable confirm floor
        // (0.66). At low coverage (0.25) it must NOT confirm to Strong; the same
        // shape with a genuine answerable-band anchor (0.71, "Load mutationg
        // manifests") MUST. This is the salad-regression guard for the absent⇒
        // unknown change.
        let sem = NodeId(60);
        let terms = vec![ResolvedTerm {
            token: "semantic".into(),
            resolved: true,
            symbol_freq: 100,
            idf_w: 0.63,
            top_node: Some(sem),
            anchors_emitted: 0,
        }];
        // Cluster oracle_top 0.659 (confident, but below promote 0.72 → no
        // semantic_strong). Anchor oracle scores "semantic" at 0.615 (present, but
        // below confirm floor 0.66). scored → we measured it.
        let cluster = HashMap::from([(NodeId(999), 0.659_f32)]);
        let anchor_oracle = HashMap::from([(sem, 0.615_f32)]);
        let scored: std::collections::HashSet<NodeId> = [sem].into_iter().collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.25,
            15.0,
            true,
            true,
            &cluster,
            &anchor_oracle,
            &scored,
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "answerable-band gate: a below-0.66 anchor must not confirm a low-coverage salad"
        );

        // Same shape, anchor now in the answerable band (0.71) → Strong.
        let anchor_ok = HashMap::from([(sem, 0.71_f32)]);
        let c2 = classify_confidence_lexical_fallback(
            &terms,
            0.25,
            15.0,
            true,
            true,
            &cluster,
            &anchor_ok,
            &scored,
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c2,
            Confidence::Strong,
            "answerable-band anchor (≥0.66) confirms a genuine low-coverage query"
        );
    }

    #[test]
    fn confidence_strong_on_embed_confirmed_low_coverage() {
        // Measured "Load mutationg manifests" shape: a typo ("mutationg") and a generic
        // token ("load", idf 0.42) drag classifier coverage to 1/3, but the confident
        // oracle places the specific anchor ("manifests", idf 0.66, cos 0.71 ≥ floor) near
        // the query. A real, embedding-confirmed match must be Strong, not Weak.
        let manifests = NodeId(50);
        let terms = vec![
            ResolvedTerm {
                token: "load".into(),
                resolved: true,
                symbol_freq: 1361,
                idf_w: 0.42, // generic — does not count toward coverage/specific anchor
                top_node: Some(NodeId(1)),
                anchors_emitted: 0,
            },
            ResolvedTerm {
                token: "mutationg".into(),
                resolved: false, // typo — unresolved
                symbol_freq: 0,
                idf_w: 1.0,
                top_node: None,
                anchors_emitted: 0,
            },
            ResolvedTerm {
                token: "manifests".into(),
                resolved: true,
                symbol_freq: 71,
                idf_w: 0.66,
                top_node: Some(manifests),
                anchors_emitted: 0,
            },
        ];
        // Confident oracle (top 0.773) that AGREES the anchor is near (cos 0.71 ≥ floor 0.643).
        let oracle = HashMap::from([(manifests, 0.71_f32), (NodeId(99), 0.773_f32)]);
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.333,
            19.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Strong,
            "embed-confirmed specific anchor must be Strong even at low token coverage"
        );
    }

    #[test]
    fn confidence_weak_on_embed_unconfirmed_low_coverage_salad() {
        // Measured salad shape ("find code by semantic meaning"): a specific-ish anchor
        // ("semantic", idf 0.63) but the confident oracle does NOT place it near the query
        // (absent from the over-fetch → cos treated as 0). Must stay Weak — the embedding
        // confirmation gate is what separates this from "Load mutationg manifests".
        let semantic = NodeId(60);
        let terms = vec![
            ResolvedTerm {
                token: "semantic".into(),
                resolved: true,
                symbol_freq: 100,
                idf_w: 0.63,
                top_node: Some(semantic), // NOT in the oracle below → unconfirmed,
                anchors_emitted: 0,
            },
            ResolvedTerm {
                token: "code".into(),
                resolved: true,
                symbol_freq: 2260,
                idf_w: 0.38,
                top_node: Some(NodeId(2)),
                anchors_emitted: 0,
            },
        ];
        // Confident oracle (top 0.659) whose neighbours do NOT include the anchor.
        let oracle = HashMap::from([(NodeId(99), 0.659_f32)]);
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.25,
            15.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "embed-unconfirmed anchor (salad) must stay Weak at low coverage"
        );
    }

    #[test]
    fn confidence_weak_on_coverage_above_floor() {
        // Generic resolved token (low idf) + coverage ≥ 0.25 → Weak via coverage path.
        let terms = vec![ResolvedTerm {
            token: "auth".into(),
            resolved: true,
            symbol_freq: 200,
            idf_w: 0.40, // too generic to count as specific anchor
            top_node: Some(NodeId(7)),
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.3,
            0.1,
            true,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(c, Confidence::Weak);
    }

    #[test]
    fn confidence_weak_on_specific_anchor_low_coverage() {
        // Specific resolved token (high idf) + coverage below floor → Weak via anchor escape.
        // Validates: "google map" abstains, but "PaymentService foo bar baz" doesn't.
        let terms = vec![ResolvedTerm {
            token: "PaymentService".into(),
            resolved: true,
            symbol_freq: 2,
            idf_w: 0.93, // rare → counts as specific anchor
            top_node: Some(NodeId(9)),
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.1,
            0.0,
            true,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "specific rare anchor must trigger Weak even when coverage is below floor"
        );
    }

    #[test]
    fn confidence_none_on_generic_anchor_low_coverage() {
        // Generic resolved token (low idf) + coverage below floor → None → abstain.
        // This is the "what is google map" case: "map" resolves (freq=65, idf≈0.526)
        // but falls below idf_coverage_min (0.55), so it doesn't count as coverage.
        let terms = vec![ResolvedTerm {
            token: "map".into(),
            resolved: true,
            symbol_freq: 65,
            idf_w: 0.526, // real idf for "map" in a 6940-node corpus
            top_node: Some(NodeId(5)),
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.0,
            0.1,
            true,
            false,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::None,
            "generic resolved token with zero coverage must abstain"
        );
    }

    #[test]
    fn confidence_none_on_knn_only_no_fts() {
        // KNN always returns K neighbours regardless of semantic distance — it cannot
        // signal "no match". Zero coverage + no anchor → abstain even when KNN seeds exist.
        let terms = vec![ResolvedTerm {
            token: "xqz_no_match".into(),
            resolved: false,
            symbol_freq: 0,
            idf_w: 0.0,
            top_node: None,
            anchors_emitted: 0,
        }];
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.0,
            0.0,
            true,
            true,
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::None,
            "KNN-only (zero coverage, no anchor) must abstain"
        );
    }

    // ── Fix 1+3a / 1c: semantic promotion + veto ──────────────────────────────

    /// A confident oracle with a coherent near cluster grounds a conceptual query
    /// to Strong even with zero lexical coverage / no anchor — the "how does the
    /// kubelet sync pod status" case (oracle_top 0.83, every word too common).
    #[test]
    fn confidence_semantic_strong_promotes_conceptual_query() {
        let terms = vec![ResolvedTerm {
            token: "pod".into(),
            resolved: true,
            symbol_freq: 14_000, // generic → not rare, not specific
            idf_w: 0.30,
            top_node: None,
            anchors_emitted: 0,
        }];
        // oracle_top 0.83 → floor 0.70; 5 candidates ≥ floor → coherent cluster.
        let oracle: HashMap<NodeId, f32> = [
            (NodeId(1), 0.83),
            (NodeId(2), 0.80),
            (NodeId(3), 0.78),
            (NodeId(4), 0.74),
            (NodeId(5), 0.71),
        ]
        .into_iter()
        .collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.0,
            0.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Strong,
            "confident oracle + near cluster must promote to Strong without a lexical anchor"
        );
    }

    /// Nonsense whose oracle_top sits below the promote threshold (≤0.64 measured)
    /// must NOT be promoted — it abstains.
    #[test]
    fn confidence_semantic_strong_not_fired_for_nonsense() {
        let terms = vec![ResolvedTerm {
            token: "quux".into(),
            resolved: false,
            symbol_freq: 0,
            idf_w: 0.0,
            top_node: None,
            anchors_emitted: 0,
        }];
        // oracle_top 0.64 < promote(0.72), and > veto(0.55) → neither promote nor veto.
        let oracle: HashMap<NodeId, f32> =
            [(NodeId(1), 0.64), (NodeId(2), 0.62), (NodeId(3), 0.60)]
                .into_iter()
                .collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.0,
            0.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::None,
            "sub-threshold oracle (nonsense) must not reach Strong"
        );
    }

    /// Embeddings confident NOTHING matches (oracle_top below veto floor) + only
    /// coincidental lexical coverage → abstain, not Weak. The "cat sat on the warm
    /// windowsill" leak (oracle_top 0.49).
    #[test]
    fn confidence_semantic_veto_abstains_on_coincidental_coverage() {
        let anchor = NodeId(7);
        let terms = vec![ResolvedTerm {
            token: "windowsill".into(),
            resolved: true,
            symbol_freq: 40, // specific (high idf) but not rare
            idf_w: 0.62,
            top_node: Some(anchor),
            anchors_emitted: 0,
        }];
        // oracle present but far (top 0.49 < veto 0.55); anchor absent from oracle.
        let oracle: HashMap<NodeId, f32> = [(NodeId(99), 0.49)].into_iter().collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.5,
            12.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::None,
            "confidently-far oracle must veto coincidental lexical coverage"
        );
    }

    /// The veto exempts a rare exact anchor: a precise symbol the user literally
    /// typed is honoured (Weak) even when the model is weak on it.
    #[test]
    fn confidence_semantic_veto_exempts_rare_anchor() {
        let anchor = NodeId(7);
        let terms = vec![ResolvedTerm {
            token: "PaymentService".into(),
            resolved: true,
            symbol_freq: 2, // rare → exact anchor the user named
            idf_w: 0.93,
            top_node: Some(anchor),
            anchors_emitted: 0,
        }];
        let oracle: HashMap<NodeId, f32> = [(NodeId(99), 0.49)].into_iter().collect();
        // Low coverage so it doesn't hit the Exact branch; must land on Weak, not None.
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.3,
            0.1,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "rare exact anchor must be exempt from the semantic veto"
        );
    }

    /// Strong lexical evidence (high coverage + BM25) survives a weak oracle —
    /// the veto only suppresses Weak-grade coincidences, never a real lexical hit.
    #[test]
    fn confidence_lexical_strong_survives_weak_oracle() {
        let terms = vec![ResolvedTerm {
            token: "ppr".into(),
            resolved: false, // no anchor — pure whole-query BM25 evidence
            symbol_freq: 50,
            idf_w: 0.0,
            top_node: None,
            anchors_emitted: 0,
        }];
        // Oracle present but weak (0.50 < veto 0.55) — would veto if lexical were weak.
        let oracle: HashMap<NodeId, f32> = [(NodeId(99), 0.50)].into_iter().collect();
        let c = classify_confidence_lexical_fallback(
            &terms,
            0.8,
            2.0,
            true,
            true,
            &oracle,
            &oracle,
            &std::collections::HashSet::new(),
            &Calibration::IDENTITY,
        );
        assert_eq!(
            c,
            Confidence::Strong,
            "strong lexical evidence must beat the veto (lexical-Strong path is checked first)"
        );
    }

    // ── semantic_validate (Tier-1 cosine oracle) ──────────────────────────────

    fn seed(id: u64, source: SeedSource) -> Seed {
        Seed {
            node: NodeId(id),
            weight: 1.0,
            source,
            score: 1.0,
            rerank_score: None,
        }
    }

    #[test]
    fn semantic_validate_noop_when_oracle_empty() {
        // Embeddings off/degraded → oracle empty → lexical seeds untouched.
        let seeds = vec![seed(1, SeedSource::Lexical), seed(2, SeedSource::Exact)];
        let out = semantic_validate(seeds.clone(), &HashMap::new(), &Calibration::IDENTITY);
        assert_eq!(out.len(), 2, "empty oracle must be a no-op");
    }

    #[test]
    fn semantic_validate_noop_when_oracle_not_confident() {
        // Top cosine below ORACLE_MIN → embedding found nothing strong → keep all.
        let seeds = vec![seed(1, SeedSource::Lexical), seed(2, SeedSource::Knn)];
        let oracle: HashMap<NodeId, f32> = [(NodeId(2), 0.50)].into_iter().collect();
        let out = semantic_validate(seeds, &oracle, &Calibration::IDENTITY);
        assert_eq!(out.len(), 2, "weak oracle must not cut lexical seeds");
    }

    #[test]
    fn semantic_validate_cuts_salad_keeps_relevant() {
        // The confident-oracle case: NodeId(10..13) are genuine semantic seeds
        // (cosine ~0.68); NodeId(1..6) are spurious FTS anchors absent from the
        // oracle (cosine 0.0). Validation must drop the salad and keep the seeds.
        let mut seeds: Vec<Seed> = (1..=6).map(|i| seed(i, SeedSource::Exact)).collect();
        seeds.extend((10..=15).map(|i| seed(i, SeedSource::Knn)));
        let oracle: HashMap<NodeId, f32> = (10..=15)
            .map(|i| (NodeId(i), 0.68 - (i as f32 - 10.0) * 0.01))
            .collect();
        let out = semantic_validate(seeds, &oracle, &Calibration::IDENTITY);
        let kept: std::collections::HashSet<u64> = out.iter().map(|s| s.node.0).collect();
        assert!(
            kept.iter().all(|&id| id >= 10),
            "spurious FTS anchors (cosine 0) must be cut; got {kept:?}"
        );
        assert_eq!(out.len(), 6, "all 6 genuine semantic seeds must survive");
        // Reshuffled by cosine: highest-cosine seed first.
        assert_eq!(
            out[0].node,
            NodeId(10),
            "survivors reshuffled by cosine desc"
        );
    }

    #[test]
    fn semantic_validate_preserving_exact_keeps_exact_below_floor() {
        // RFC-021 / Issue D repro (hermetic). A moderately-common on-topic exact
        // anchor ("daemon"-style: resolves to real Daemon symbols but is not rare
        // enough for G1) that the DILUTED whole-query embedding leaves out of the
        // oracle (cosine 0.0) must NOT be dropped here — it has to reach the
        // reranker. Contrast `semantic_validate_cuts_salad_keeps_relevant`, which
        // calls the raw `semantic_validate` and drops the exact-sourced seeds.
        let mut seeds: Vec<Seed> = (1..=3).map(|i| seed(i, SeedSource::Exact)).collect();
        // KNN tail the diluted query DOES score (off-topic neighbours it drifted to):
        seeds.extend((10..=15).map(|i| seed(i, SeedSource::Knn)));
        let oracle: HashMap<NodeId, f32> = (10..=15)
            .map(|i| (NodeId(i), 0.68 - (i as f32 - 10.0) * 0.01))
            .collect();

        let out = semantic_validate_preserving_exact(seeds, &oracle, &Calibration::IDENTITY);
        let kept: std::collections::HashSet<u64> = out.iter().map(|s| s.node.0).collect();
        assert!(
            kept.contains(&1) && kept.contains(&2) && kept.contains(&3),
            "exact anchors (cosine 0, absent from oracle) must bypass the veto and \
             reach the reranker; got {kept:?}"
        );
    }

    #[test]
    fn semantic_validate_preserving_exact_still_cuts_lexical_tail() {
        // The exemption is exact-only: a coincidental LEXICAL anchor the confident
        // oracle disagrees with (cosine 0.0, absent) is still cut, so the reranker
        // isn't handed the whole spurious-collision tail.
        let mut seeds: Vec<Seed> = (1..=3).map(|i| seed(i, SeedSource::Lexical)).collect();
        seeds.extend((10..=15).map(|i| seed(i, SeedSource::Knn)));
        let oracle: HashMap<NodeId, f32> = (10..=15)
            .map(|i| (NodeId(i), 0.68 - (i as f32 - 10.0) * 0.01))
            .collect();

        let out = semantic_validate_preserving_exact(seeds, &oracle, &Calibration::IDENTITY);
        let kept: std::collections::HashSet<u64> = out.iter().map(|s| s.node.0).collect();
        assert!(
            kept.iter().all(|&id| id >= 10),
            "lexical anchors the embedding rejects must still be cut; got {kept:?}"
        );
        assert_eq!(out.len(), 6, "only the 6 KNN seeds survive");
    }

    #[test]
    fn semantic_validate_preserving_exact_keeps_exact_inside_rerank_window() {
        // BLOCKER regression (dense-repo Issue D re-opener): `build_seed_set`
        // reranks only the top-K front slice (`seeds[..min(len, TRAVSR_RERANK_TOPK
        // = 30)]`). Build a surviving tail well past K so that appending the exact
        // anchors (the pre-fix behaviour) would push them outside the window — they
        // would then never get a `rerank_score` and would drop out of
        // `max_rerank_score`, silently abstaining on exactly the moderately-common
        // named-symbol / dense-repo case. Prepending must keep them at the front.
        let mut seeds: Vec<Seed> = (1..=3).map(|i| seed(i, SeedSource::Exact)).collect();
        // 40 KNN tail seeds the confident oracle all keeps (well over K = 30).
        seeds.extend((100..140).map(|i| seed(i, SeedSource::Knn)));
        let oracle: HashMap<NodeId, f32> = (100..140).map(|i| (NodeId(i), 0.68)).collect();

        let out = semantic_validate_preserving_exact(seeds, &oracle, &Calibration::IDENTITY);

        let window = out.len().min(30);
        let exact_in_window = out[..window]
            .iter()
            .filter(|s| s.source == SeedSource::Exact)
            .count();
        assert_eq!(
            exact_in_window,
            3,
            "all 3 exact anchors must sit within the top-{window} rerank window \
             regardless of tail size; front sources: {:?}",
            out.iter().take(6).map(|s| s.source).collect::<Vec<_>>()
        );
    }

    #[test]
    fn semantic_validate_never_empties_grounded_result() {
        // Confident oracle but only 2 seeds clear the floor — MIN_KEEP guarantees
        // we still return SEMANTIC_MIN_KEEP rather than emptying the result.
        let mut seeds: Vec<Seed> = (1..=6).map(|i| seed(i, SeedSource::Lexical)).collect();
        seeds.push(seed(10, SeedSource::Knn));
        seeds.push(seed(11, SeedSource::Knn));
        let oracle: HashMap<NodeId, f32> = [
            (NodeId(10), 0.70),
            (NodeId(11), 0.69),
            (NodeId(1), 0.40),
            (NodeId(2), 0.35),
        ]
        .into_iter()
        .collect();
        let out = semantic_validate(seeds, &oracle, &Calibration::IDENTITY);
        assert_eq!(
            out.len(),
            SEMANTIC_MIN_KEEP,
            "must keep top-{SEMANTIC_MIN_KEEP} by cosine, never empty"
        );
    }

    // ── #462: calibrated recall floor + WS1 anti-truncation + WS2 rescue ─────

    /// Arctic-embed-m band measured on kubernetes (lo = nonsense p95, hi =
    /// self-match p50). Conceptual golds sit ~0.44–0.65, salads ≤0.41.
    const ARCTIC: Calibration = Calibration {
        lo: 0.337,
        hi: 0.532,
    };

    #[test]
    fn recall_floor_identity_is_conservative_calibrated_is_lower() {
        // Un-calibrated / bge: keep the pre-#462 high-confidence bar (bge salad
        // neighbours reach ~0.75, so the floor must stay high there).
        assert_eq!(semantic_recall_floor(&Calibration::IDENTITY), 0.75);
        // A calibrated model maps the floor into its own (compressed) band, landing
        // just above the salad p95 — well below the identity bar.
        let arctic = semantic_recall_floor(&ARCTIC);
        assert!(
            (0.40..=0.45).contains(&arctic),
            "arctic recall floor should sit ~0.42 (above salads ≤0.41, below golds ≥0.44); got {arctic}"
        );
    }

    #[test]
    fn semantic_validate_keeps_genuine_seed_below_relative_floor() {
        // WS1: the relative floor is anchored on the ORACLE top, which here is a
        // non-seed over-fetch neighbour (NodeId 99, cos 0.65). That pushes the
        // relative floor to ~0.535, above the genuine rank-low gold (NodeId 1, cos
        // 0.45) — the pre-#462 code truncated it out because 5 other seeds cleared
        // the relative floor. The recall-floor guard (0.42 on arctic) must keep it.
        let mut seeds: Vec<Seed> = vec![seed(1, SeedSource::Knn)]; // the gold
        seeds.extend((2..=6).map(|i| seed(i, SeedSource::Knn))); // 5 stronger seeds
        let mut oracle: HashMap<NodeId, f32> = (2..=6).map(|i| (NodeId(i), 0.60)).collect();
        oracle.insert(NodeId(1), 0.45); // gold: genuine match, below relative floor
        oracle.insert(NodeId(99), 0.65); // non-seed neighbour drives the oracle top
        let out = semantic_validate(seeds, &oracle, &ARCTIC);
        let kept: std::collections::HashSet<u64> = out.iter().map(|s| s.node.0).collect();
        assert!(
            kept.contains(&1),
            "a genuine semantic seed clearing the recall floor must not be truncated \
             just because the oracle top is a stronger non-seed neighbour; got {kept:?}"
        );
    }

    #[test]
    fn semantic_validate_still_cuts_below_recall_floor() {
        // The guard only protects seeds ABOVE the recall floor — a genuine-noise
        // seed (cos 0.30 < 0.42 arctic) is still cut, so the anti-truncation change
        // doesn't reopen the salad it was designed to drop.
        let mut seeds: Vec<Seed> = vec![seed(1, SeedSource::Lexical)]; // noise
        seeds.extend((2..=8).map(|i| seed(i, SeedSource::Knn)));
        let mut oracle: HashMap<NodeId, f32> = (2..=8).map(|i| (NodeId(i), 0.60)).collect();
        oracle.insert(NodeId(1), 0.30); // below arctic recall floor → must be cut
        let out = semantic_validate(seeds, &oracle, &ARCTIC);
        let kept: std::collections::HashSet<u64> = out.iter().map(|s| s.node.0).collect();
        assert!(
            !kept.contains(&1),
            "a below-recall-floor noise seed must still be cut; got {kept:?}"
        );
    }

    #[test]
    fn confidence_ws2_rescues_abstain_when_embedding_confident() {
        // WS2: the reranker under-scored the candidate (0.1, below the weak floor)
        // so the base verdict is None — but the embedding's top neighbour (0.50)
        // clears the arctic recall floor (~0.42), so the result is rescued to Weak
        // rather than thrown away.
        let knn: HashMap<NodeId, f32> = [(NodeId(1), 0.50)].into_iter().collect();
        let empty = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            true,
            &knn,
            &empty,
            &scored,
            &ARCTIC,
            false,
            Some(0.1),
        );
        assert_eq!(
            c,
            Confidence::Weak,
            "a confident embedding must rescue a reranker abstain to Weak"
        );
    }

    #[test]
    fn confidence_ws2_does_not_rescue_off_domain() {
        // Salad-safety: the reranker abstains AND the top embed neighbour (0.30) is
        // below the recall floor — no rescue, stays None. This is the guardrail that
        // keeps salad false-positives at zero.
        let knn: HashMap<NodeId, f32> = [(NodeId(1), 0.30)].into_iter().collect();
        let empty = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            true,
            &knn,
            &empty,
            &scored,
            &ARCTIC,
            false,
            Some(0.1),
        );
        assert_eq!(
            c,
            Confidence::None,
            "an off-domain query below the recall floor must still abstain"
        );
    }

    #[test]
    fn confidence_ws2_no_rescue_on_identity() {
        // WS2 is calibrated-only. On an un-calibrated / bge (identity) index a
        // reranker abstain (0.1) is NOT rescued even when the embed top (0.80) clears
        // the identity recall floor (0.75). This keeps the reference path byte-for-byte
        // and preserves the documented G2 invariant (confidence identical embed-on/off)
        // exactly there — the embedding must not become a confidence input on bge.
        let knn: HashMap<NodeId, f32> = [(NodeId(1), 0.80)].into_iter().collect();
        let empty = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            true,
            &knn,
            &empty,
            &scored,
            &Calibration::IDENTITY,
            false,
            Some(0.1),
        );
        assert_eq!(
            c,
            Confidence::None,
            "WS2 must not rescue on an un-calibrated/identity index (G2 holds there)"
        );
    }

    #[test]
    fn confidence_ws2_no_rescue_on_g1_bypass() {
        // WS2 never rescues a g1_bypass (rare-named-symbol) query, even on a calibrated
        // model with a confident embed top (0.50 ≥ arctic recall ~0.42): G1's
        // deterministic lexical verdict must never be overridden by an embedding signal.
        let knn: HashMap<NodeId, f32> = [(NodeId(1), 0.50)].into_iter().collect();
        let empty = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            true,
            &knn,
            &empty,
            &scored,
            &ARCTIC,
            true,
            Some(0.1),
        );
        assert_eq!(
            c,
            Confidence::None,
            "WS2 must not override the deterministic G1 verdict on a g1_bypass query"
        );
    }

    // ── RFC-021 Phase 3: absolute-floor classify_confidence gate ─────────────

    #[test]
    fn confidence_rerank_above_strong_floor_is_strong() {
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            false, // no exact anchor
            Some(0.9),
        );
        assert_eq!(c, Confidence::Strong);
    }

    #[test]
    fn confidence_rerank_between_floors_is_weak() {
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            false,
            Some(0.65),
        );
        assert_eq!(c, Confidence::Weak);
    }

    /// The RFC-021 reproduction bug, at the unit level: the reranker ran and
    /// scored the best candidate low — must abstain, not surface a confident
    /// salad, regardless of how much lexical coverage the query happened to hit.
    #[test]
    fn confidence_rerank_below_weak_floor_abstains() {
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            false,
            Some(0.1),
        );
        assert_eq!(
            c,
            Confidence::None,
            "a low rerank score must abstain even with seeds present"
        );
    }

    #[test]
    fn confidence_rerank_none_falls_back_to_lexical_gate() {
        // Reranker unavailable for this query (None) — must produce the exact
        // same decision as calling the fallback directly with identical inputs.
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let terms = [ResolvedTerm {
            token: "build_seed_set".into(),
            resolved: true,
            symbol_freq: 1,
            idf_w: 0.9,
            top_node: Some(NodeId(1)),
            anchors_emitted: 0,
        }];
        let via_gate = classify_confidence(
            &terms,
            1.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            false,
            None,
        );
        let via_fallback = classify_confidence_lexical_fallback(
            &terms,
            1.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
        );
        assert_eq!(via_gate, via_fallback);
        assert_eq!(via_gate, Confidence::Exact);
    }

    /// G1 (blocker): an exact-symbol query stays on the deterministic path even
    /// when the (NL-trained) reranker scores the exact match's own name low —
    /// the failure mode the RFC names explicitly (`GetWarningsForPod` scoring
    /// low against its own literal query). Trading salad for a false-abstain on
    /// a precise lookup is exactly what G1 forbids.
    #[test]
    fn confidence_g1_exact_anchor_bypasses_low_rerank_score() {
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let terms = [ResolvedTerm {
            token: "getwarningsforpod".into(),
            resolved: true,
            symbol_freq: 1,
            idf_w: 0.9,
            top_node: Some(NodeId(1)),
            anchors_emitted: 0,
        }];
        let exact_anchor_ids: std::collections::HashSet<NodeId> = [NodeId(1)].into_iter().collect();
        let g1_bypass = compute_g1_bypass(
            &terms,
            &TrustedAnchors {
                exact: &exact_anchor_ids,
                corrected: &std::collections::HashSet::new(),
            },
        );
        assert!(
            g1_bypass,
            "rare term tied to its own exact anchor must bypass"
        );
        let c = classify_confidence(
            &terms,
            1.0,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            g1_bypass,
            Some(0.02), // reranker scored the exact match low — must be ignored
        );
        assert_eq!(
            c,
            Confidence::Exact,
            "exact-symbol queries must not be governed by the rerank floor"
        );
    }

    /// The fix for the RFC-021 reproduction bug at the G1-boundary level: an
    /// exact anchor resolves ("drop" really is `SqliteStore.drop`), and even
    /// measures decent coverage (3/5 in the real repro — dozens of types
    /// implement `Drop`, so "drop" and "delete" both resolve). A
    /// coverage-only gate was tried first and still let this exact query
    /// bypass to `Confidence::Strong` (Defect A reproduced through the G1
    /// escape hatch) — coverage alone can't tell "the user named a symbol"
    /// from "a common word coincidentally IS one." Rarity can: `drop` is
    /// anything but rare (`symbol_freq` far above `rare_anchor_max`), so
    /// `compute_g1_bypass` must return false here; the query falls through
    /// to the rerank floor, where a low score correctly abstains.
    #[test]
    fn confidence_g1_requires_a_rare_anchor_not_just_any_exact_anchor() {
        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let terms = [ResolvedTerm {
            token: "drop".into(),
            resolved: true,
            symbol_freq: 50, // common — dozens of Drop impls, nowhere near rare
            idf_w: 0.6,
            top_node: Some(NodeId(1)),
            anchors_emitted: 0,
        }];
        let exact_anchor_ids: std::collections::HashSet<NodeId> = [NodeId(1)].into_iter().collect();
        let g1_bypass = compute_g1_bypass(
            &terms,
            &TrustedAnchors {
                exact: &exact_anchor_ids,
                corrected: &std::collections::HashSet::new(),
            },
        );
        assert!(!g1_bypass, "a common exact anchor must not bypass");
        let c = classify_confidence(
            &terms,
            0.6, // coverage alone clears coverage_strong — must not be enough on its own
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            g1_bypass,
            Some(0.02), // reranker correctly scores the sentence irrelevant
        );
        assert_eq!(
            c,
            Confidence::None,
            "a coincidental exact-anchor hit on a common word must not bypass the rerank floor"
        );
    }

    /// RFC-021 F3: a salad query that resolves an exact anchor on a *common*
    /// term but ALSO happens to resolve an unrelated *rare* term (whose own
    /// top match is not among the exact anchors) must not bypass G1 either —
    /// rarity has to be tied to the term that produced the anchor, not to any
    /// resolved term in the query.
    #[test]
    fn confidence_g1_rarity_must_be_tied_to_the_anchor_term_not_any_resolved_term() {
        let terms = [
            ResolvedTerm {
                token: "drop".into(),
                resolved: true,
                symbol_freq: 50, // common — this is the term that produced the exact anchor
                idf_w: 0.6,
                top_node: Some(NodeId(1)),
                anchors_emitted: 0,
            },
            ResolvedTerm {
                token: "incidental".into(),
                resolved: true,
                symbol_freq: 1, // rare, but resolves to an unrelated node — not an exact anchor
                idf_w: 0.9,
                top_node: Some(NodeId(2)),
                anchors_emitted: 0,
            },
        ];
        let exact_anchor_ids: std::collections::HashSet<NodeId> = [NodeId(1)].into_iter().collect();
        let g1_bypass = compute_g1_bypass(
            &terms,
            &TrustedAnchors {
                exact: &exact_anchor_ids,
                corrected: &std::collections::HashSet::new(),
            },
        );
        assert!(
            !g1_bypass,
            "rarity on an unrelated resolved term must not launder a common anchor into a bypass"
        );

        let oracle: HashMap<NodeId, f32> = HashMap::new();
        let scored = std::collections::HashSet::new();
        let c = classify_confidence(
            &terms,
            0.6,
            0.0,
            true,
            false,
            &oracle,
            &oracle,
            &scored,
            &Calibration::IDENTITY,
            g1_bypass,
            Some(0.02),
        );
        assert_eq!(
            c,
            Confidence::None,
            "salad with an incidental rare term plus a common anchor must abstain, not bypass"
        );
    }

    /// G2: confidence must be identical whether embeddings ran or not, for the
    /// same query — the reranker's floor gate never looks at knn_oracle /
    /// anchor_oracle at all (only classify_confidence_lexical_fallback does,
    /// and it's reached identically in both cases here since rerank_score is
    /// held constant).
    #[test]
    fn confidence_g2_identical_embed_on_vs_off_for_same_rerank_score() {
        let empty: HashMap<NodeId, f32> = HashMap::new();
        let warm: HashMap<NodeId, f32> = [(NodeId(1), 0.95)].into_iter().collect();
        let scored = std::collections::HashSet::new();
        let with_embed_off = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            false,
            &empty,
            &empty,
            &scored,
            &Calibration::IDENTITY,
            false,
            Some(0.85),
        );
        let with_embed_on = classify_confidence(
            &[],
            0.0,
            0.0,
            true,
            true,
            &warm,
            &warm,
            &scored,
            &Calibration::IDENTITY,
            false,
            Some(0.85),
        );
        assert_eq!(
            with_embed_off, with_embed_on,
            "confidence must not depend on embedding presence when the reranker has an opinion"
        );
    }

    /// RFC-021 F4: a reranked-but-rejected seed (model scored it near zero —
    /// judged garbage) must not outrank a seed the model never saw (rank 31+,
    /// still carries whatever RRF standing it had). Only model-approved seeds
    /// (>= weak floor) may outrank the unjudged tail.
    #[test]
    fn sort_seeds_post_rerank_unjudged_tail_outranks_judged_garbage() {
        let weak = rerank_weak_floor();
        let mut approved = seed(1, SeedSource::Lexical);
        approved.rerank_score = Some(0.9);
        let mut rejected = seed(2, SeedSource::Lexical);
        rejected.rerank_score = Some(0.001);
        let unjudged = seed(3, SeedSource::Lexical); // rerank_score: None (tail)

        let mut seeds = vec![rejected.clone(), unjudged.clone(), approved.clone()];
        sort_seeds_post_rerank(&mut seeds, weak);

        let order: Vec<u64> = seeds.iter().map(|s| s.node.0).collect();
        assert_eq!(
            order,
            vec![1, 3, 2],
            "approved (band 1) > unjudged tail (band 2) > judged garbage (band 3)"
        );
    }

    /// Regression test for a real repro found live on the travsr repo: "how
    /// does the daemon watch for file changes" abstained because
    /// `handle_watch_event` never made it into the anchor set. Root cause:
    /// `anchor_path_counts` is one counter shared across every token in the
    /// query — the earlier-processed "daemon" token filled
    /// `crates/travsr-daemon/src/lib.rs`'s 2-slot budget before "watch" was
    /// processed, so its perfectly valid match (`handle_watch_event`) was
    /// dropped purely by token order, not relevance. A token that resolves a
    /// genuine, all-checks-passed match must always get at least one anchor
    /// slot, even when an earlier token already exhausted the shared
    /// per-path budget.
    #[test]
    fn build_seed_set_later_token_not_starved_by_earlier_tokens_path_cap() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // Two "daemon" matches in the same file exhaust MAX_SEEDS_PER_PATH (2)
        // before the "watch" token is processed.
        let path = "crates/travsr-daemon/src/lib.rs";
        let daemon_new = Node::new(
            VName::new("corpus", "", path, "rust", "fn:daemon_new"),
            "function",
        );
        let daemon_run = Node::new(
            VName::new("corpus", "", path, "rust", "fn:daemon_run"),
            "function",
        );
        let handle_watch = Node::new(
            VName::new("corpus", "", path, "rust", "fn:handle_watch_event"),
            "function",
        );
        store.put_node(&daemon_new).unwrap();
        store.put_node(&daemon_run).unwrap();
        store.put_node(&handle_watch).unwrap();

        let seed_set = build_seed_set(
            &store,
            "daemon watch",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        let seed_ids: std::collections::HashSet<NodeId> =
            seed_set.seeds.iter().map(|s| s.node).collect();
        assert!(
            seed_ids.contains(&handle_watch.id),
            "later token's genuine match must not be starved by an earlier \
             token filling the shared path budget; seeds present: {:?}",
            seed_set.seeds.iter().map(|s| s.node).collect::<Vec<_>>()
        );
    }

    /// A corrected token's protected anchor bypasses the IDF emit cut and the
    /// per-path cap by design, so whatever it emits skips the gates that would
    /// normally drop it. That makes the signature check load-bearing: the
    /// `boundary` predicate accepts a signature OR a path match, so the
    /// correction's candidate list can contain a node that merely lives in a
    /// file named after the corrected symbol. Emitting that node would hand a
    /// path-only coincidence the full protection of a deliberate correction.
    #[test]
    fn build_seed_set_corrected_anchor_requires_a_signature_match() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // The correction target's signature is QUALIFIED (`method:Views.HttpResponse`),
        // so `search_nodes_by_name` scores it in the loose `ELSE 40` tier rather
        // than the `%:Name` tier, exactly tying it with the path-only match. The
        // tie is then broken by the hashed `NodeId` rowid, so which of the two
        // leads the candidate list is effectively arbitrary per symbol. This
        // fixture picks a decoy name that wins that tie, which is what makes the
        // signature filter load-bearing here rather than merely defensive.
        let decoy = Node::new(
            VName::new("corpus", "", "HttpResponse.py", "python", "fn:helper0"),
            "function",
        );
        let real = Node::new(
            VName::new(
                "corpus",
                "",
                "app/api/views/renderers.py",
                "python",
                "method:Views.HttpResponse",
            ),
            "method",
        );
        store.put_node(&decoy).unwrap();
        store.put_node(&real).unwrap();

        // "htpresponse" resolves to nothing by name, so it takes the #709
        // correction branch and grounds to `HttpResponse`.
        let seed_set = build_seed_set(
            &store,
            "htpresponse",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        let seed_ids: std::collections::HashSet<NodeId> =
            seed_set.seeds.iter().map(|s| s.node).collect();
        assert!(
            !seed_ids.contains(&decoy.id),
            "a path-only match must not be emitted as the protected corrected \
             anchor; seeds present: {seed_ids:?}"
        );
        assert!(
            seed_ids.contains(&real.id),
            "the signature match is the correction target and must seed; \
             seeds present: {seed_ids:?}"
        );
    }

    /// #778 regression: a short symbol (`UI`) whose exact definition is crowded
    /// out of the anchor set by SUBSTRING noise. `search_nodes_by_name` is a
    /// substring match, so "ui" also matches methods named `*build**ui**ng*`;
    /// the #463 kind-reorder then promotes those logic-bearing methods (rank 0)
    /// ahead of `class:UI` (rank 1), and the emit loop's take-then-filter spent
    /// all `MAX_ANCHORS_PER_TOKEN` slots on them before the `contains_token`
    /// gate could reject them, so `class:UI` never seeded and `ask "UI"`
    /// abstained on an exact rank-0 match. The fix pre-filters the anchor pool
    /// to signature-segment matches, so the noise never enters the window.
    #[test]
    fn build_seed_set_short_exact_symbol_not_starved_by_substring_noise() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // The real target: a 2-char class name, absent from the word vocab.
        let ui = Node::new(
            VName::new("corpus", "", "core/lib/ui/ui.rb", "ruby", "class:UI"),
            "class",
        );
        store.put_node(&ui).unwrap();
        // Three substring-noise methods: "ui" is a substring of "building"/
        // "buildup"/"builtui", never a name segment. As `method` (logic rank 0)
        // they reorder ahead of the class and, pre-fix, filled the 3 anchor
        // slots via take-then-filter.
        for sig in [
            "method:Gym.building_for_ios",
            "method:Gym.buildup_guid",
            "method:Gym.rebuild_circuit",
        ] {
            store
                .put_node(&Node::new(
                    VName::new("corpus", "", "gym/lib/gym/module.rb", "ruby", sig),
                    "method",
                ))
                .unwrap();
        }

        let seed_set = build_seed_set(
            &store,
            "UI",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        let seed_ids: std::collections::HashSet<NodeId> =
            seed_set.seeds.iter().map(|s| s.node).collect();
        assert!(
            seed_ids.contains(&ui.id),
            "the exact `class:UI` match must seed and not be starved by \
             substring-noise methods; seeds present: {seed_ids:?}"
        );
    }

    /// Regression test for a second, related repro found live: "give me data
    /// about daemon" abstained even though "daemon" alone resolves cleanly
    /// to real, well-connected content. Root cause: `anchor_raw` fed
    /// `rrf_fuse` in raw token-processing order, not weight order —
    /// `rrf_fuse`'s own documented precondition is that each source is
    /// sorted descending by score (position 0 = best), since it scores by
    /// list position. "data" (common, many matches) is processed before
    /// "daemon" (rare, one specific match) purely because it appears earlier
    /// in the query, so daemon's higher-weight match landed at a worse RRF
    /// rank than data's lower-weight ones despite being more relevant.
    #[test]
    fn build_seed_set_anchor_raw_sorted_by_weight_before_rrf_fusion() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // "data": common, several lower-weight matches, each in its own path.
        for i in 0..3 {
            let n = Node::new(
                VName::new(
                    "corpus",
                    "",
                    format!("crates/x/data_{i}.rs"),
                    "rust",
                    format!("fn:data_helper_{i}"),
                ),
                "function",
            );
            store.put_node(&n).unwrap();
        }
        // "daemon": rare, one specific, higher-weight match (lower symbol_freq
        // means higher idf_w; same kind as the data matches so kind_boost is
        // not a confounding variable).
        let daemon = Node::new(
            VName::new(
                "corpus",
                "",
                "crates/travsr-daemon/src/lib.rs",
                "rust",
                "fn:daemon_run",
            ),
            "function",
        );
        store.put_node(&daemon).unwrap();

        // "data" appears before "daemon" in the query, so it is processed
        // first in the per-token anchor loop.
        let seed_set = build_seed_set(
            &store,
            "data daemon",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        assert!(!seed_set.seeds.is_empty(), "expected at least one seed");
        assert_eq!(
            seed_set.seeds[0].node,
            daemon.id,
            "the rarer, higher-weight match must rank first regardless of \
             which token resolved it earlier in query order; seeds: {:?}",
            seed_set.seeds.iter().map(|s| s.node).collect::<Vec<_>>()
        );
    }

    /// Regression test for a third repro found live: "give me data about
    /// daemon" abstained even after the RRF-sort fix. Root cause:
    /// `semantic_validate` ran BEFORE `g1_bypass` was computed and dropped
    /// the exact, rare "daemon" match because its cosine to the noisy,
    /// filler-word-heavy whole-query embedding was lower than several
    /// unrelated seeds' — exactly the failure mode G1 exists to prevent, but
    /// G1 was only ever consulted afterward, too late to help. Fixed by
    /// computing `g1_bypass` from `anchor_raw` right after it is built, and
    /// skipping `semantic_validate` entirely when it is true.
    #[test]
    fn build_seed_set_g1_bypass_protects_exact_match_from_semantic_validate() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // Five unrelated exact matches, all scored high by the (simulated)
        // embed oracle - enough to make it "confident" and, pre-fix, to push
        // the rare daemon match out of semantic_validate's top-N cosine cut.
        let fillers = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let mut oracle: HashMap<NodeId, f32> = HashMap::new();
        for (i, f) in fillers.iter().enumerate() {
            let n = Node::new(
                VName::new(
                    "corpus",
                    "",
                    format!("crates/x/{f}_{i}.rs"),
                    "rust",
                    format!("fn:{f}_helper"),
                ),
                "function",
            );
            store.put_node(&n).unwrap();
            oracle.insert(n.id, 0.85);
        }
        // "daemon": rare, genuine match, deliberately absent from the oracle
        // (cosine treated as 0.0) - simulates the live repro where the exact
        // match's own cosine to the whole conversational query scored low.
        let daemon = Node::new(
            VName::new(
                "corpus",
                "",
                "crates/travsr-daemon/src/lib.rs",
                "rust",
                "fn:daemon_run",
            ),
            "function",
        );
        store.put_node(&daemon).unwrap();

        let seed_set = build_seed_set(
            &store,
            "alpha beta gamma delta epsilon daemon",
            &travsr_retrieval::OpenFilter,
            vec![],
            &oracle,
            None,
        );

        assert!(
            seed_set.seeds.iter().any(|s| s.node == daemon.id),
            "G1's rare exact match must survive semantic_validate, not be cut \
             by a low cosine to the noisy whole-query embedding; seeds: {:?}",
            seed_set
                .seeds
                .iter()
                .map(|s| (s.node, s.source))
                .collect::<Vec<_>>()
        );
    }

    // ── #478 RFC-023: FTS-only lexical precision regression tests ────────────
    //
    // Every test below passes `None` as `score_fn` (the RFC-019 direct-cosine
    // oracle). The original bug was invisible on a warm embed oracle — the
    // oracle vetoed the bad candidate and the query correctly abstained —
    // which is exactly why the pre-#478 suite never caught it. These tests
    // run the FTS-only path deliberately, per RFC-023 AC #10.

    /// The #478 repro itself. `fn:walk` in `walker.ts` is a pure trigram
    /// substring match on the query token "wal" with no word-boundary
    /// evidence, and must not anchor — before this fix it reached norm 1.000
    /// and outranked everything, including escaping crate scope.
    #[test]
    fn wal_does_not_anchor_to_walk_on_the_fts_only_path() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        let walk = Node::new(
            VName::new(
                "corpus",
                "",
                "packages/travsr-lsif-ts/src/walker.ts",
                "typescript",
                "fn:walk",
            ),
            "function",
        );
        store.put_node(&walk).unwrap();

        // A genuine SQLite/WAL-adjacent node so the query resolves to
        // *something* real rather than testing pure abstention.
        let journal_mode = Node::new(
            VName::new(
                "corpus",
                "",
                "crates/travsr-store/src/lib.rs",
                "rust",
                "fn:SqliteStore.journal_mode",
            ),
            "method",
        );
        store.put_node(&journal_mode).unwrap();

        let seed_set = build_seed_set(
            &store,
            "what is the rationale for SQLite WAL",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None, // FTS-only: embed oracle disabled.
        );

        assert!(
            !seed_set.seeds.iter().any(|s| s.node == walk.id),
            "fn:walk must not anchor to the 'wal' token on the FTS-only path; \
             seeds: {:?}",
            seed_set
                .seeds
                .iter()
                .map(|s| (s.node, s.source))
                .collect::<Vec<_>>()
        );
    }

    /// #540 review: the note claimed "cut on capacity, not relevance" for
    /// tokens that never entered the anchor loop at all.
    ///
    /// A token below `anchor_emit_cut` is suppressed by the IDF gate — a
    /// relevance signal — and emits zero anchors. The old field reported
    /// `min(symbol_freq, 3)`, so it said 3 reached when 0 did, and attributed
    /// an explicitly relevance-based drop to capacity.
    ///
    /// Drives the real `explain_seed_set` rather than re-deriving the formula.
    /// The previous test defined its own `min(freq, cap)` inside the test
    /// module and asserted against that, so it mirrored the production line
    /// instead of exercising it and could not have caught this.
    #[test]
    fn a_suppressed_token_reports_zero_anchors_not_the_cap() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        // Many symbols sharing one generic token, so its IDF lands low enough
        // to be suppressed, plus one specific token that does anchor.
        for i in 0..40 {
            let n = Node::new(
                VName::new(
                    "corpus",
                    "",
                    format!("src/mod{i}.rs"),
                    "rust",
                    format!("fn:get_thing{i}"),
                ),
                "function",
            );
            store.put_node(&n).unwrap();
        }
        let special = Node::new(
            VName::new("corpus", "", "src/special.rs", "rust", "fn:zarquon"),
            "function",
        );
        store.put_node(&special).unwrap();

        // No env knob: `get` appears in 40 of 41 nodes, so
        // `idf_weight(40, 41) = ln(42/41)/ln(42)` clamps to 0.05, below the
        // 0.15 default cut. An earlier draft set `TRAVSR_ANCHOR_EMIT_CUT`
        // instead, which is process-global — Rust runs tests as threads in one
        // process, so it changed the cut underneath whatever else was running
        // and broke an unrelated seed test on macOS CI. A test that has to
        // mutate global state to reach its case is testing the knob, not the
        // behaviour.
        let empty: HashMap<NodeId, f32> = HashMap::new();
        let report = explain_seed_set(
            &store,
            "get zarquon",
            "fn:zarquon",
            &travsr_retrieval::OpenFilter,
            Vec::new(),
            &empty,
            None,
        );

        for t in &report.tokens {
            if !t.is_anchor_emit {
                assert_eq!(
                    t.anchors_emitted, 0,
                    "token {:?} was suppressed before the anchor loop, so it emitted nothing, \
                     reporting a non-zero count would attribute a relevance drop to capacity",
                    t.token
                );
            }
            assert!(
                t.anchors_emitted <= t.symbol_freq,
                "token {:?}: emitted {} anchors but only {} symbols name it",
                t.token,
                t.anchors_emitted,
                t.symbol_freq
            );
        }
    }

    /// #478 RFC-023 §6.1 (WS-8): `travsr explain`'s own report on the exact
    /// #478 repro fixture must show the leg separation that explains *why*
    /// `fn:walk` doesn't anchor — a real trigram (substring) match and no
    /// word/exact match — not just the fact that it doesn't. This is the
    /// diagnostic's core value proposition, pinned as a regression.
    #[test]
    fn explain_seed_set_shows_trigram_only_match_for_walk_on_wal_query() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        let walk = Node::new(
            VName::new(
                "corpus",
                "",
                "packages/travsr-lsif-ts/src/walker.ts",
                "typescript",
                "fn:walk",
            ),
            "function",
        );
        store.put_node(&walk).unwrap();

        let report = explain_seed_set(
            &store,
            "what is the rationale for SQLite WAL",
            "fn:walk",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None, // FTS-only: embed oracle disabled.
        );

        assert!(report.node_found, "fn:walk must resolve to a node");
        let legs = report.legs.expect("legs must be Some when node_found");
        assert!(
            legs.trigram.is_some(),
            "fn:walk must show a trigram (substring) match on 'wal', that's \
             the actual mechanism, and explain exists to surface it"
        );
        assert!(
            legs.word.is_none(),
            "fn:walk must NOT show a word-leg match, 'wal' is not a word \
             segment of 'walk', only a substring of it"
        );
        assert!(
            legs.exact.is_none(),
            "fn:walk has no genuine exact/name match for this query"
        );
    }

    /// RFC-023 AC #2 (the W1 fix): `sqlite` must now anchor to
    /// `SqliteStore.*` methods — rejected pre-#478 because the old
    /// punctuation-only boundary check saw `s` immediately after `sqlite` in
    /// `SqliteStore` and refused the match.
    #[test]
    fn sqlite_anchors_to_sqlitestore_method_on_the_fts_only_path() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        let exec_ddl = Node::new(
            VName::new(
                "corpus",
                "",
                "crates/travsr-store/src/lib.rs",
                "rust",
                "fn:SqliteStore.exec_ddl",
            ),
            "method",
        );
        store.put_node(&exec_ddl).unwrap();

        let seed_set = build_seed_set(
            &store,
            "sqlite exec_ddl",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        assert!(
            seed_set.seeds.iter().any(|s| s.node == exec_ddl.id),
            "SqliteStore.exec_ddl must anchor for the token 'sqlite' \
             (camelCase/PascalCase-aware boundary check); seeds: {:?}",
            seed_set
                .seeds
                .iter()
                .map(|s| (s.node, s.source))
                .collect::<Vec<_>>()
        );
    }

    /// RFC-023 AC #6: word-level `symbol_frequency` — "works" must not anchor
    /// via a substring hit inside "workspace" (the regression guard for the
    /// anchor loop's `resolved`/boundary check, not just `contains_token`
    /// in isolation).
    #[test]
    fn works_does_not_anchor_to_workspace_on_the_fts_only_path() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        let workspace_fn = Node::new(
            VName::new(
                "corpus",
                "",
                "packages/travsr-vscode/src/extension.ts",
                "typescript",
                "fn:provideWorkspaceChatContext",
            ),
            "function",
        );
        store.put_node(&workspace_fn).unwrap();

        let seed_set = build_seed_set(
            &store,
            "does this works correctly",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None,
        );

        assert!(
            !seed_set.seeds.iter().any(|s| s.node == workspace_fn.id),
            "provideWorkspaceChatContext must not anchor for the NL verb \
             'works'; seeds: {:?}",
            seed_set
                .seeds
                .iter()
                .map(|s| (s.node, s.source))
                .collect::<Vec<_>>()
        );
    }

    /// Found via CI (not local dev, where an installed embed backend masked
    /// it): an exact camelCase symbol name typed verbatim as a query — e.g.
    /// `travsr ask "isPrime"` against a fresh repo with no embed model
    /// installed, exactly `travsr-daemon`'s
    /// `query_cache_invalidated_by_out_of_band_delete` test's own setup —
    /// must reach at least `Confidence::Strong`, not collapse to `None`.
    ///
    /// Root cause: `tokenize_query` used to lowercase every content token
    /// before `symbol_frequency`/`contains_token` ever saw it. Lowercasing
    /// `"isPrime"` to `"isprime"` destroys the lower→upper transition
    /// `ident::segments` needs to split it into `["is", "prime"]`, so
    /// `segments("isprime")` returns one fused, unsplittable segment that
    /// can never match the index's real `"prime"` vocabulary entry.
    /// `symbol_frequency` then falls back to "absent from vocabulary" (the
    /// corpus-size floor, effectively "maximally generic"), the anchor-emit
    /// IDF gate rejects it, and the query's only real anchor is silently
    /// dropped even though Leg A (exact/name) found it at rank 0. Fixed by
    /// having `tokenize_query` preserve the token's original case.
    #[test]
    fn camel_case_exact_name_query_reaches_strong_confidence_on_the_fts_only_path() {
        use travsr_core::{Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();

        store
            .put_node(&Node::new(
                VName::new("corpus", "", "prime.ts", "typescript", "fn:isPrime"),
                "function",
            ))
            .unwrap();
        store
            .put_node(&Node::new(
                VName::new("corpus", "", "a.ts", "typescript", "class:Alpha"),
                "class",
            ))
            .unwrap();
        store
            .put_node(&Node::new(
                VName::new("corpus", "", "b.ts", "typescript", "class:Beta"),
                "class",
            ))
            .unwrap();

        let seed_set = build_seed_set(
            &store,
            "isPrime",
            &travsr_retrieval::OpenFilter,
            vec![],
            &HashMap::new(),
            None, // FTS-only: embed oracle disabled, matching the CI regression.
        );

        assert!(
            matches!(seed_set.confidence, Confidence::Strong | Confidence::Exact),
            "an exact camelCase symbol-name query must not collapse to a weak \
             confidence, let alone abstain, got {:?}, seeds: {:?}",
            seed_set.confidence,
            seed_set
                .seeds
                .iter()
                .map(|s| (s.node, s.source))
                .collect::<Vec<_>>()
        );
    }

    // ── RFC-021 Phase 0: rerank floor-sweep (dev tool, not a CI test) ─────────
    //
    // Not run in CI: needs both a real cross-encoder model AND a real indexed
    // graph.db. By default it self-dogfoods against travsr's own codebase (the
    // labeled queries compiled in from bench/queries.json). Gated on two env
    // vars, skips gracefully when either is absent.
    //
    // Run (travsr self-corpus):
    //   TRAVSR_RERANK_MODEL_DIR=<dir> TRAVSR_FLOOR_SWEEP_DB=<path/to/.travsr/graph.db> \
    //     cargo test -p travsr-mcp --lib rerank_floor_sweep -- --nocapture --ignored
    //
    // Run (cross-repo, e.g. kubernetes — RFC-021 §9.3 Task 4 floor-generalization
    // gate): additionally point the query set at that repo's labeled file:
    //   TRAVSR_FLOOR_SWEEP_QUERIES=bench/queries-k8s.json \
    //   TRAVSR_FLOOR_SWEEP_DB=/path/to/kubernetes/.travsr/graph.db  (+ model dir)
    #[test]
    #[ignore]
    fn rerank_floor_sweep() {
        let Some(db_path) = std::env::var_os("TRAVSR_FLOOR_SWEEP_DB") else {
            eprintln!("skipping: TRAVSR_FLOOR_SWEEP_DB not set");
            return;
        };
        if std::env::var_os("TRAVSR_RERANK_MODEL_DIR").is_none() {
            eprintln!("skipping: TRAVSR_RERANK_MODEL_DIR not set");
            return;
        }

        #[derive(serde::Deserialize)]
        struct LabeledQuery {
            id: String,
            category: String,
            query: String,
        }
        #[derive(serde::Deserialize)]
        struct QueryFile {
            queries: Vec<LabeledQuery>,
        }
        // Query set: compiled-in travsr self-corpus by default; override with
        // TRAVSR_FLOOR_SWEEP_QUERIES=<path> to calibrate against another repo's
        // labeled set (e.g. bench/queries-k8s.json) — RFC-021 §9.3 Task 3, the
        // cross-repo floor-generalization gate. The file must have the same
        // {queries:[{id,category,query}]} shape and the same positive/negative
        // category vocabulary (literal|conceptual|cross vs nonsense|salad).
        let raw: String = match std::env::var_os("TRAVSR_FLOOR_SWEEP_QUERIES") {
            Some(path) => std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read TRAVSR_FLOOR_SWEEP_QUERIES {path:?}: {e}")),
            None => include_str!("../../../bench/queries.json").to_string(),
        };
        let parsed: QueryFile =
            serde_json::from_str(&raw).expect("floor-sweep query file must parse");

        let store = SqliteStore::open(std::path::Path::new(&db_path)).expect("open graph.db");

        // No KNN, no score_fn: the floor is measured on the reranker's own
        // signal, independent of embeddings — matching G2 (confidence must be
        // embed-on/off identical), so the calibration shouldn't secretly
        // depend on whether embeddings happen to be indexed for this repo.
        let mut scored: Vec<(String, String, f32)> = Vec::new(); // (id, category, max_rerank_score)
        for q in &parsed.queries {
            let seed_set = build_seed_set(
                &store,
                &q.query,
                &travsr_retrieval::OpenFilter,
                vec![],
                &HashMap::new(),
                None,
            );
            let max_score = seed_set
                .seeds
                .iter()
                .filter_map(|s| s.rerank_score)
                .fold(0.0_f32, f32::max);
            scored.push((q.id.clone(), q.category.clone(), max_score));
            eprintln!(
                "{:<4} {:<10} max_rerank_score={:.4}",
                q.id, q.category, max_score
            );
        }

        let is_positive = |cat: &str| matches!(cat, "literal" | "conceptual" | "cross");
        let is_negative = |cat: &str| matches!(cat, "nonsense" | "salad");
        let n_pos = scored.iter().filter(|(_, c, _)| is_positive(c)).count();
        let n_neg = scored.iter().filter(|(_, c, _)| is_negative(c)).count();

        eprintln!(
            "\nfloor sweep (N={}: {n_pos} positive, {n_neg} negative)",
            scored.len()
        );
        eprintln!("floor | negative_abstention | positive_recall");
        let mut best_floor: f32 = 0.0;
        let mut steps = 0;
        while steps <= 20 {
            let floor = steps as f32 * 0.05;
            steps += 1;
            let neg_abstained = scored
                .iter()
                .filter(|(_, c, s)| is_negative(c) && *s < floor)
                .count();
            let pos_retained = scored
                .iter()
                .filter(|(_, c, s)| is_positive(c) && *s >= floor)
                .count();
            let neg_abstention = neg_abstained as f32 / n_neg.max(1) as f32;
            let pos_recall = pos_retained as f32 / n_pos.max(1) as f32;
            eprintln!("{floor:.2}  | {neg_abstention:.3}                | {pos_recall:.3}");
            // Largest floor that still honors the recall gate — negative_abstention
            // is monotonically non-decreasing in floor, so this is the ROC-optimal pick.
            if pos_recall >= 0.90 {
                best_floor = floor;
            }
        }
        eprintln!("\nrecommended WEAK_FLOOR (>=90% positive recall): {best_floor:.2}");

        // STRONG_FLOOR heuristic: median score among positives, NOT hit-verified
        // (a coarser anchor than WEAK_FLOOR — see travsr-qa-engineer note in the
        // RFC-021 Phase 0 discussion: this dataset only labels answerable/nonsense,
        // not graded strong/weak, so this half of the pair is a starting heuristic).
        let mut pos_scores: Vec<f32> = scored
            .iter()
            .filter(|(_, c, _)| is_positive(c))
            .map(|(_, _, s)| *s)
            .collect();
        pos_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = pos_scores.get(pos_scores.len() / 2).copied().unwrap_or(0.0);
        eprintln!("recommended STRONG_FLOOR (heuristic, median of positive scores): {median:.2}");
    }

    // ── #376 Phase 2: docs lane ──────────────────────────────────────────────

    use crate::seed::DOCS_ENV_LOCK;

    #[test]
    fn doc_floor_default_is_0_42() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_DOC_FLOOR");
        assert_eq!(doc_floor(), 0.42);
    }

    /// A store plus a hermetic config environment for the docs knobs (#376 O1).
    ///
    /// The knobs now resolve through `travsr_config` (`env > repo > global >
    /// default`), so a test that merely cleared the env var would still read the
    /// developer's real `~/.travsr/config.toml` — a machine with
    /// `docs.enabled = true` set globally would flip these assertions. Both file
    /// layers are redirected into a fresh tempdir: `HOME` for the global layer,
    /// and the store's `repo_root` meta for the repo layer.
    ///
    /// Callers **must** hold [`DOCS_ENV_LOCK`]: `HOME` is process-global, and
    /// that is the same lock the env knobs are already serialized on.
    struct DocsConfigEnv {
        dir: tempfile::TempDir,
        prev_home: Option<std::ffi::OsString>,
        store: SqliteStore,
    }

    impl DocsConfigEnv {
        fn new() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            let prev_home = std::env::var_os("HOME");
            std::env::set_var("HOME", dir.path().join("home"));

            let repo = dir.path().join("repo");
            std::fs::create_dir_all(repo.join(".travsr")).expect("mk repo/.travsr");
            let mut store = SqliteStore::open_in_memory().expect("store");
            store
                .set_meta("repo_root", &repo.to_string_lossy())
                .expect("set repo_root");

            for k in [
                "TRAVSR_DOCS_ENABLED",
                "TRAVSR_DOCS_MAX_RESULTS",
                "TRAVSR_DOCS_BUDGET_PCT",
            ] {
                std::env::remove_var(k);
            }
            Self {
                dir,
                prev_home,
                store,
            }
        }

        fn repo_root(&self) -> std::path::PathBuf {
            self.dir.path().join("repo")
        }

        /// Write a key into the *repo* layer, the one a user gets from
        /// `travsr config set <key> <value>` inside a repo.
        fn set_repo(&self, key: &str, value: &str) {
            travsr_config::set(key, value, travsr_config::Scope::Repo(self.repo_root()))
                .expect("config set");
        }
    }

    impl Drop for DocsConfigEnv {
        fn drop(&mut self) {
            match self.prev_home.take() {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn docs_max_results_default_is_3() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        assert_eq!(docs_max_results(&env.store), 3);
    }

    /// #519: both bench repos cleared #376 §7's bar (all five docs-lane gates
    /// green, travsr and kubernetes, against merged code), earning the
    /// default flip. Was `docs_enabled_defaults_to_false` — renamed rather
    /// than edited in place, so a reader never trusts a name the assertions
    /// no longer match.
    #[test]
    fn docs_enabled_defaults_to_true() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        assert!(docs_enabled(&env.store));
        std::env::set_var("TRAVSR_DOCS_ENABLED", "0");
        assert!(!docs_enabled(&env.store));
        std::env::remove_var("TRAVSR_DOCS_ENABLED");
    }

    #[test]
    fn docs_budget_pct_default_is_20() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        assert_eq!(docs_budget_pct(&env.store), 20.0);
    }

    /// #376 O1 / G1, the item this plumbing exists for: the switch must be
    /// operable from `travsr config set` with **no environment variable at
    /// all**, because the process that reads it (the daemon for `ask`, the MCP
    /// server for `get_context`) does not inherit the user's shell env.
    #[test]
    fn docs_knobs_are_settable_from_repo_config_without_any_env_var() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();

        // #519: default flipped on once both bench repos cleared #376 §7's
        // bar. The property this test actually pins - repo config controls
        // the switch with no env var involved - is unchanged; only which
        // direction the flip demonstrates is.
        assert!(docs_enabled(&env.store), "default on");
        env.set_repo("docs.enabled", "false");
        assert!(
            !docs_enabled(&env.store),
            "repo config must turn the lane off"
        );

        env.set_repo("docs.max_results", "7");
        assert_eq!(docs_max_results(&env.store), 7);
        env.set_repo("docs.budget_pct", "35");
        assert_eq!(docs_budget_pct(&env.store), 35.0);
    }

    /// Env stays the highest stored layer, so every existing script and bench
    /// harness that exports the variable in the *daemon's* environment keeps
    /// working byte-for-byte after O1.
    #[test]
    fn env_still_overrides_repo_config() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        env.set_repo("docs.enabled", "true");
        std::env::set_var("TRAVSR_DOCS_ENABLED", "0");
        assert!(
            !docs_enabled(&env.store),
            "env must win over the repo config layer"
        );
        std::env::remove_var("TRAVSR_DOCS_ENABLED");
        assert!(docs_enabled(&env.store), "and the repo layer applies again");
    }

    /// A malformed value must fall through to the built-in default rather than
    /// guessing: a typo in `config.toml` cannot silently flip the lane on, and
    /// cannot brick retrieval either.
    #[test]
    fn garbage_config_value_falls_back_to_default() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        // `set` would reject this, so write the file directly — the case is a
        // hand-edited config.toml, which is exactly how it reaches us.
        std::fs::write(
            env.repo_root().join(".travsr").join("config.toml"),
            "[docs]\nenabled = \"ture\"\nmax_results = \"lots\"\n",
        )
        .expect("write config");
        // #519: the built-in default is now `true`.
        assert!(docs_enabled(&env.store));
        assert_eq!(docs_max_results(&env.store), 3);
    }

    /// #519: docs.enabled=true is now the ship default, so a real hook must
    /// produce real candidates with no config at all. Was
    /// `doc_lane_candidates_disabled_by_default_returns_empty` - renamed
    /// rather than edited in place; see `doc_lane_candidates_explicitly_disabled_returns_empty`
    /// for the flag-gates-the-lane property that test used to cover.
    #[test]
    fn doc_lane_candidates_enabled_by_default_returns_candidates() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        let hook: DocKnnFn<'_> = &|_q, _k| vec![(NodeId(1), 0.9)];
        assert!(!doc_lane_candidates(&env.store, "query", Some(hook)).is_empty());
    }

    /// docs.enabled=false must return no doc results regardless of what the
    /// hook would otherwise say — the feature flag, not hook availability,
    /// gates the lane. Explicit now that disabled is no longer the default.
    #[test]
    fn doc_lane_candidates_explicitly_disabled_returns_empty() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        env.set_repo("docs.enabled", "false");
        let hook: DocKnnFn<'_> = &|_q, _k| vec![(NodeId(1), 0.9)];
        assert!(doc_lane_candidates(&env.store, "query", Some(hook)).is_empty());
    }

    #[test]
    fn doc_lane_candidates_none_hook_returns_empty() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");
        assert!(doc_lane_candidates(&env.store, "query", None).is_empty());
        std::env::remove_var("TRAVSR_DOCS_ENABLED");
    }

    /// Unlike the old cosine-floor lane, the candidate pool applies **no
    /// floor** — the reranker downstream makes the confidence call — but is
    /// still sorted by cosine descending so a reranker-unavailable fallback
    /// has a sane order.
    #[test]
    fn doc_lane_candidates_returns_unfiltered_sorted_pool() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");

        let hook: DocKnnFn<'_> = &|_q, _k| {
            vec![
                (NodeId(1), 0.55),
                (NodeId(2), 0.90),
                (NodeId(3), 0.10), // would be below the old DOC_FLOOR — kept here
                (NodeId(4), 0.70),
            ]
        };
        let hits = doc_lane_candidates(&env.store, "query", Some(hook));
        assert_eq!(
            hits,
            vec![
                (NodeId(2), 0.90),
                (NodeId(4), 0.70),
                (NodeId(1), 0.55),
                (NodeId(3), 0.10),
            ],
            "must keep every candidate, sorted descending, no floor applied"
        );

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
    }

    #[test]
    fn doc_rerank_overfetch_default_is_20() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_DOC_RERANK_OVERFETCH");
        assert_eq!(doc_rerank_overfetch(), 20);
    }

    /// The compiled default is the §14.1 sweep result (0.05), **not** the code
    /// lane's WEAK floor (0.5) it originally borrowed. Guards against a revert
    /// to the borrowed value, which cost travsr a gold hit (1.0 → 0.9 hit@1)
    /// and k8s a hit@3 point for no measured benefit.
    #[test]
    fn doc_rerank_floor_default_is_the_swept_value_not_the_code_lane_weak_floor() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_DOC_RERANK_FLOOR");
        // Must hold whether or not a reranker model is installed on this
        // machine: the manifest's `weak_floor` is a code-lane number and must
        // never leak into this lane (that leak made the shipped default 0.5
        // for every real user while the tests, which have no model dir, saw
        // 0.05 and passed).
        let resolved = doc_rerank_floor();
        assert_eq!(resolved, 0.05, "§14.1 swept value");
        assert!(
            resolved < travsr_rerank::DEFAULT_WEAK_FLOOR,
            "the docs lane floor is deliberately below the code lane's WEAK floor \
             ({resolved} vs {}); §14.1 measured the doc negative-arm ceiling at \
             ~0.002, not ~0.5; borrowing the code floor cost travsr a gold hit",
            travsr_rerank::DEFAULT_WEAK_FLOOR
        );
    }

    /// #539: the docs lane's recall stage embeds content tokens only (the
    /// same extraction `tokenize_query` already does for the code lane's
    /// per-token anchor resolution), not the full normalized sentence — this
    /// is what fixes the NL-query dilution bug. This deliberately breaks the
    /// old §4.4 byte-identical-text memo contract with the code lane's
    /// `embed_path_seeds` call whenever the query has filler words; see the
    /// docstring on `doc_lane_query` for why that tradeoff is intentional.
    #[test]
    fn doc_lane_query_strips_filler_words_for_recall() {
        for raw in [
            "how does the knapsack enforce the token budget?",
            "why did we choose PPR over BFS ?",
            "what are the rules about unwrap and error handling in library code",
        ] {
            assert_eq!(
                doc_lane_query(raw),
                tokenize_query(raw).join(" "),
                "doc_lane_query must send tokenize_query's content tokens for {raw:?}"
            );
            assert_ne!(
                doc_lane_query(raw),
                travsr_store::fts_tokenize::normalize_nl_query(raw),
                "test query must actually carry filler words the fix strips, for {raw:?}"
            );
        }
    }

    /// Falls back to the full normalized sentence, never an empty string,
    /// when every token is a stopword.
    #[test]
    fn doc_lane_query_falls_back_to_normalized_sentence_when_all_stopwords() {
        let raw = "  what is the  ";
        assert!(
            tokenize_query(raw).is_empty(),
            "test query must be all-stopword"
        );
        assert_eq!(
            doc_lane_query(raw),
            travsr_store::fts_tokenize::normalize_nl_query(raw),
        );
        assert!(!doc_lane_query(raw).is_empty());
    }

    /// An all-punctuation query strips to zero tokens *and* normalizes to an
    /// empty string (`normalize_nl_query` also drops bare-punctuation
    /// tokens), so the first fallback isn't enough on its own — this asserts
    /// the second fallback (trimmed raw query) keeps the "never empty"
    /// contract for that case too.
    #[test]
    fn doc_lane_query_falls_back_to_raw_text_when_all_punctuation() {
        let raw = "???";
        assert!(
            tokenize_query(raw).is_empty(),
            "test query must strip to zero tokens"
        );
        assert!(
            travsr_store::fts_tokenize::normalize_nl_query(raw).is_empty(),
            "test query must also normalize to empty, exercising the second fallback"
        );
        assert_eq!(doc_lane_query(raw), "???");
    }

    /// The content-token normalization is actually applied at the KNN
    /// boundary, not merely available as a helper — captures the text the
    /// hook receives.
    #[test]
    fn doc_lane_candidates_sends_the_content_token_query_to_knn() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        std::env::set_var("TRAVSR_DOCS_ENABLED", "1");

        let seen = std::sync::Mutex::new(String::new());
        let hook: DocKnnFn<'_> = &|q, _k| {
            *seen.lock().unwrap() = q.to_string();
            vec![]
        };
        let raw = "how does the knapsack enforce the token budget?";
        let _ = doc_lane_candidates(&env.store, raw, Some(hook));

        let sent = seen.lock().unwrap().clone();
        assert_eq!(
            sent,
            doc_lane_query(raw),
            "the hook must receive doc_lane_query's content-token text"
        );
        assert_ne!(
            sent,
            travsr_store::fts_tokenize::normalize_nl_query(raw),
            "test query must exercise the filler-word-stripping change"
        );
        assert_ne!(sent, raw, "test query must exercise a normalization change");

        std::env::remove_var("TRAVSR_DOCS_ENABLED");
    }

    /// Below-floor candidates are dropped entirely (§4.2: absent, not
    /// empty-ish), above-floor candidates are sorted descending and capped.
    /// This is the reranker-unavailable fallback path
    /// (`tools::rerank_doc_candidates`), so it must reproduce the pre-rerank
    /// Phase 2 behaviour exactly.
    #[test]
    fn cosine_floor_select_applies_floor_sort_and_cap() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.5");
        std::env::set_var("TRAVSR_DOCS_MAX_RESULTS", "2");

        let candidates = vec![
            (NodeId(1), 0.55),
            (NodeId(2), 0.90),
            (NodeId(3), 0.40), // below floor
            (NodeId(4), 0.70),
        ];
        let hits = cosine_floor_select(&env.store, candidates);
        assert_eq!(
            hits,
            vec![(NodeId(2), 0.90), (NodeId(4), 0.70)],
            "must drop below-floor node 3, sort descending, cap at 2"
        );

        std::env::remove_var("TRAVSR_DOC_FLOOR");
        std::env::remove_var("TRAVSR_DOCS_MAX_RESULTS");
    }

    #[test]
    fn cosine_floor_select_below_floor_everywhere_is_empty() {
        let _guard = DOCS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = DocsConfigEnv::new();
        std::env::set_var("TRAVSR_DOC_FLOOR", "0.42");

        let candidates = vec![(NodeId(1), 0.30), (NodeId(2), 0.38)];
        assert!(cosine_floor_select(&env.store, candidates).is_empty());

        std::env::remove_var("TRAVSR_DOC_FLOOR");
    }

    /// MatchSource::Docs must sit between Semantic and Relevant (plan §4.1
    /// section order: exact, semantic, docs, relevant).
    #[test]
    fn match_source_docs_trust_rank_between_semantic_and_relevant() {
        assert!(MatchSource::Semantic.trust_rank() < MatchSource::Docs.trust_rank());
        assert!(MatchSource::Docs.trust_rank() < MatchSource::Relevant.trust_rank());
        assert_eq!(MatchSource::Docs.label(), "docs");
    }
}
