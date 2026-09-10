//! travsr-store — pluggable graph storage backend.
//!
//! The MVP uses SQLite (WAL) via `rusqlite`; hyperscale later moves to RocksDB.
//! All backends implement the `Store` trait below.

#![forbid(unsafe_code)]

pub mod fts_tokenize;
pub mod migration;
pub mod registry;
mod seed_lexicon;

pub use migration::{Migration, MigrationRunner, StoreMigratable};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// RFC-027 #813: SHA-256 hex of the source text spanned by a definition, keyed
/// by NodeId (`i64`). It is the byte-identity witness the live overlay uses to
/// tell a definition the edit left untouched (its committed edges stay valid and
/// are preserved) from one whose body changed (its edges are re-resolved).
///
/// `spans` are `(node_id, start_line, end_line)` with 1-based line numbers. A
/// node is omitted (its body treated as changed, so the caller re-resolves — the
/// fail-safe direction) when either its span falls outside the file (a stale
/// span) or its `end_line` is unknown. The `end_line` guard matters: without a
/// known end we could only hash the definition's first line, and an edit to any
/// later line of a multi-line body would then hash identical and be wrongly
/// preserved with its now-stale edges. Line-start offsets are computed once, so
/// this is O(file + nodes).
fn body_hashes_for_spans(
    content: &str,
    spans: impl IntoIterator<Item = (i64, u32, Option<u32>)>,
) -> HashMap<i64, String> {
    // `line_starts[L - 1]` is the byte offset where 1-based line L begins.
    let mut line_starts: Vec<usize> = Vec::with_capacity(content.len() / 24 + 1);
    line_starts.push(0);
    for (i, b) in content.bytes().enumerate() {
        if b == b'\n' {
            line_starts.push(i + 1);
        }
    }
    let total_lines = line_starts.len();
    let mut out = HashMap::new();
    for (id, start, end) in spans {
        let start = start as usize;
        if start == 0 || start > total_lines {
            continue;
        }
        // No known end line: hashing only the first line would miss edits to the
        // rest of a multi-line body. Skip, so the caller re-resolves.
        let Some(end) = end else { continue };
        let end = (end as usize).max(start).min(total_lines);
        let begin = line_starts[start - 1];
        // `end` is 1-based inclusive: take bytes up to the start of line end + 1
        // (or EOF for the final line), so the definition's full text is hashed.
        let finish = if end >= total_lines {
            content.len()
        } else {
            line_starts[end]
        };
        let mut hasher = Sha256::new();
        hasher.update(&content.as_bytes()[begin..finish]);
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            let _ = write!(hex, "{byte:02x}");
        }
        out.insert(id, hex);
    }
    out
}

/// True when `name` occurs as a whole word starting at byte column `col` of
/// `line` (RFC-027 #813 P2, issue #816 defect 1). The column is a 0-based byte
/// offset, matching the occurrence column persisted from SCIP. A word boundary
/// on both sides keeps a substring of a longer identifier from matching.
fn occurrence_at_col(line: &str, name: &str, col: usize) -> bool {
    let bytes = line.as_bytes();
    let nb = name.as_bytes();
    if nb.is_empty() || col + nb.len() > bytes.len() || !line.is_char_boundary(col) {
        return false;
    }
    if &bytes[col..col + nb.len()] != nb {
        return false;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let before_ok = col == 0 || !is_word(bytes[col - 1]);
    let after = col + nb.len();
    let after_ok = after >= bytes.len() || !is_word(bytes[after]);
    before_ok && after_ok
}

/// True when a whole identifier token *starts* at byte column `col` of `line`.
///
/// The weaker companion to [`occurrence_at_col`]: it asks whether the column is
/// a real token boundary, not which token sits there. Used only as the
/// second-tier check in [`snap_changed_occurrence_line`], where the committed
/// callee's name is known not to be the token the source spells (RFC-027 #813
/// P2). Word characters are ASCII, so slicing at `col` is safe once `col` is a
/// char boundary.
fn token_starts_at(line: &str, col: usize) -> bool {
    let bytes = line.as_bytes();
    if col >= bytes.len() || !line.is_char_boundary(col) {
        return false;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    // A word character that is not the middle of a longer identifier.
    is_word(bytes[col]) && (col == 0 || !is_word(bytes[col - 1]))
}

/// How far [`snap_changed_occurrence_line`] will fan out from its start-delta
/// estimate before abstaining.
///
/// The estimate is off by the net lines the edit inserted or deleted above the
/// occurrence, so this caps the line shift a single save can chase. Well past
/// any hand edit, and it keeps the cost of an occurrence that cannot be
/// re-anchored at this cap instead of the definition's span, which is the whole
/// buffer when the definition has no known `end_line`. Beyond it the occurrence
/// abstains and heals at the next commit.
const MAX_SNAP_RADIUS: i64 = 512;

/// The current line of a changed definition's committed occurrence, or `None`
/// to abstain (issue #816 defect 1).
///
/// The committed line is first remapped by the definition's start delta, which
/// is correct for an occurrence above the edit point but stale by the net
/// inserted or deleted line count for one below it: a body edit does not move
/// the definition's start, so the start delta alone leaves later occurrences
/// off. A pure line insertion or deletion preserves each occurrence's byte
/// column, so the occurrence is re-anchored by that column: the line nearest to
/// the start-delta `estimate`, within the definition's current span
/// `[span_start, span_end]`, that carries `name` at byte column `col` as a whole
/// word. Ties prefer the line at or after the estimate, since an insert (which
/// pushes occurrences down to a larger line number) is the common edit.
///
/// The committed callee's name is not always the token the source spells at the
/// occurrence. `Self { .. }` and `-> Self` reference the impl's type, an alias
/// or re-export is written under its local name, and a tuple field is spelled
/// `0`; in every one of those the stored column is right and only the name to
/// compare against is wrong. Measured on this repository, 94.5% of the
/// occurrences the name search alone rejects have a real identifier token
/// starting at exactly the stored column. So a failed search falls back to a
/// second tier: keep the occurrence at the unshifted `estimate` when a whole
/// token starts at `col` there ([`token_starts_at`]).
///
/// That tier deliberately does NOT fan out. The name is what makes a search over
/// candidate lines safe; "some token starts here" would match almost any line at
/// the same indentation, so it is only ever applied at radius 0, where the
/// start-delta estimate is already believed correct. It is strictly stronger
/// than the column-less path, which serves the bare estimate with no check at
/// all, and the daemon still bounds the result to the definition's current span,
/// the editor still resolves the live buffer, and `names_match` still gates the
/// answer against the callee's leaf, so a stale position stays fail-closed.
///
/// Returns `None` when no such line exists in the span and no token starts at
/// the column on the estimate line (the occurrence's own line was reflowed, or
/// the column holds an operator or a lifetime rather than an identifier), so a
/// body edit that rewrites a reference abstains and heals at commit rather than
/// serving a stale position. A mis-snap to another occurrence of the same
/// `(name, col)` inside the same definition is harmless: the editor resolves
/// the same callee, so the daemon derives the same `(enclosing def -> callee)`
/// edge either way.
///
/// The search fans OUT from the estimate and returns the first hit, so its cost
/// is the shift distance (near zero for a typical one-line edit), not the
/// definition's length, and it stops at [`MAX_SNAP_RADIUS`], so an occurrence
/// that ends up abstaining costs that cap rather than the whole span: a huge
/// definition with many occurrences never turns into an occurrence-times-span
/// scan on the save path.
fn snap_changed_occurrence_line(
    lines: &[&str],
    name: &str,
    col: usize,
    estimate: i64,
    span_start: i64,
    span_end: i64,
) -> Option<u32> {
    let lo = span_start.max(1);
    let hi = span_end.max(lo);
    let hit = |l: i64| -> bool {
        l >= lo
            && l <= hi
            && lines
                .get((l - 1) as usize)
                .is_some_and(|line| occurrence_at_col(line, name, col))
    };
    // Radius 0 first, then each ring outward. A tie at the same distance prefers
    // the line after the estimate (a larger line number): an insert, the common
    // edit, pushes occurrences down, so the true line is at or after the
    // estimate. Bounded by how far either end of the span is from the estimate
    // AND by MAX_SNAP_RADIUS, so the loop always terminates and a miss costs the
    // cap rather than the span.
    let max_radius = (estimate - lo)
        .abs()
        .max((hi - estimate).abs())
        .min(MAX_SNAP_RADIUS);
    for r in 0..=max_radius {
        if hit(estimate + r) {
            return Some((estimate + r) as u32);
        }
        if r > 0 && hit(estimate - r) {
            return Some((estimate - r) as u32);
        }
    }
    // Second tier: the name is not the source's spelling for this occurrence.
    // Accept the estimate alone, and only when the column is a real token
    // boundary there, so the editor is pointed at a token rather than into
    // whitespace or the middle of an identifier.
    if estimate >= lo
        && estimate <= hi
        && lines
            .get((estimate - 1) as usize)
            .is_some_and(|line| token_starts_at(line, col))
    {
        return Some(estimate as u32);
    }
    None
}

/// Row cap on [`SqliteStore::lookup_nodes_exact`] (exact-signature Tier 1).
///
/// This is intentionally one greater than `travsr-mcp`'s `AMBIGUOUS_DISPLAY_LIMIT`
/// so a caller can always tell "exactly the display limit" apart from "more than
/// the display limit" on the exact-lookup path: when this many rows come back, at
/// least one definition was withheld. `travsr-mcp` guards the `>` relationship at
/// compile time. If you change this, keep it strictly above that display limit,
/// or the `travsr graph` ambiguity truncation notice (#565 / RFC-002) silently
/// stops firing on the Tier-1 path.
pub const NODE_EXACT_LOOKUP_LIMIT: usize = 21;

/// Header window for [`SqliteStore::node_starting_at_or_below`] (issue #816
/// defect 2 backstop): how many lines above a definition's declaration a
/// provider-reported position may sit and still map to it. Covers a realistic
/// stack of attributes and doc-comment lines while staying bounded, so a stray
/// position in a large gap between definitions never attaches to a distant node.
/// The primary fix (the extension preferring `targetSelectionRange`) already
/// puts the reported line on the declaration for providers that supply it; this
/// window only backstops providers that report the item's full range or a bare
/// location, whose headers are typically short.
const LIVE_DECL_HEADER_LINES: u32 = 64;

/// Row cap on [`SqliteStore::search_nodes_by_name`] (fuzzy simple-name Tier 2).
pub const NODE_NAME_SEARCH_LIMIT: usize = 100;

/// Definition kinds [`SqliteStore::enclosing_definition_at`] recognises as an
/// enclosing scope. Mirrors `definition_node_ids_in_file`; deliberately excludes
/// `field` so a field read never resolves to an enclosing "definition" (#757).
const ENCLOSING_DEFINITION_KINDS: &[&str] = &[
    "function",
    "method",
    "fn",
    "class",
    "interface",
    "struct",
    "trait",
    "enum",
    "type",
    "typedef",
    "union",
    "object",
    "protocol",
    "mixin",
    "extension",
    "namespace",
    "init",
];

/// Node kinds whose whole span is type-level declaration text, so an edit
/// anywhere inside one can re-point how *other* definitions in the file resolve
/// (RFC-027 #813 preservation gate).
///
/// There is no type-reference `EdgeKind`, so a definition that resolves
/// *through* one of these has no edge to it and the edge-based demotion in
/// `reindex_replace` cannot see the dependency. A change to one therefore
/// invalidates the file's whole preserved set.
///
/// Container kinds (`class`, `impl`, `module`, `namespace`, `object`) are
/// deliberately absent: their span covers their members' bodies, so any
/// method-body edit would change them and preservation would never fire. Their
/// members are separate nodes whose own hashes carry the signal instead.
const TYPE_LEVEL_KINDS: &[&str] = &[
    "type",
    "typedef",
    "interface",
    "struct",
    "enum",
    "union",
    "trait",
    "protocol",
];

/// Type alias for the RFC-018 Step 4 semantic-ANN callback injected by the daemon.
/// Returns `(NodeId, cosine_similarity_score)` pairs in descending score order.
pub type EmbedKnnHook =
    Arc<dyn Fn(&str, u32) -> Result<Vec<(NodeId, f32)>, StoreError> + Send + Sync>;

/// Type alias for the RFC-019 direct-cosine oracle callback injected beside
/// [`EmbedKnnHook`]. Given the query text and a set of candidate node ids, it
/// returns the true query↔candidate cosine for **every candidate it can score**.
///
/// Contract (load-bearing): ids the embedding layer cannot score (no stored
/// vector, degraded sidecar) are **omitted** from the result — the caller must
/// treat omission as "unknown", never as cosine 0. A `None` hook is byte-for-byte
/// identical to `EmbedKnnHook = None`: the whole re-rank path collapses to the
/// FTS-only behaviour.
pub type EmbedScoreHook =
    Arc<dyn Fn(&str, &[NodeId]) -> Result<Vec<(NodeId, f32)>, StoreError> + Send + Sync>;

/// Shared arm-state for the embed KNN hook. The injector (MCP/daemon) marks it
/// ready once the sidecar is warm; the query path reads it to decide whether to
/// briefly wait (so the first query uses embeddings) and whether to report
/// `embeddings: warming` honestly instead of a silent lexical-only degrade.
///
/// Without this, the lazily-injected hook returns an empty seed set during the
/// ~3 s sidecar startup window, which the query path can't distinguish from
/// "embeddings off" — so the opening query of a session silently runs lexical-only.
pub struct EmbedReadiness {
    armed: std::sync::atomic::AtomicBool,
    disabled: std::sync::atomic::AtomicBool,
    waiters: (std::sync::Mutex<()>, std::sync::Condvar),
}

impl EmbedReadiness {
    /// Create a new, un-armed readiness handle.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            armed: std::sync::atomic::AtomicBool::new(false),
            disabled: std::sync::atomic::AtomicBool::new(false),
            waiters: (std::sync::Mutex::new(()), std::sync::Condvar::new()),
        })
    }

    /// Mark the hook armed and wake every waiter. Called by the injector's
    /// background init thread the instant the sidecar is warm.
    pub fn mark_ready(&self) {
        self.armed.store(true, std::sync::atomic::Ordering::Release);
        let _g = self.waiters.0.lock().unwrap();
        self.waiters.1.notify_all();
    }

    /// Record that arming finished with no hook installed, so semantic search
    /// is off for the life of this process (sidecar failed to start, or the
    /// index was built with a different embedding model).
    ///
    /// Distinct from "not ready": the injector still calls `mark_ready` after
    /// this, because readiness means "arming has settled" and a readiness that
    /// never flips makes every query block for the full arm-wait. Without this
    /// third state the query path sees a ready handle behind an installed
    /// meta-hook and reports `embeddings: on` while nothing semantic can run.
    /// Call before `mark_ready` so a woken waiter observes both.
    pub fn mark_disabled(&self) {
        self.disabled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// True once `mark_ready` has fired.
    pub fn is_ready(&self) -> bool {
        self.armed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// True once `mark_disabled` has fired.
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Block up to `timeout` for arming; returns the final ready state.
    pub fn wait(&self, timeout: std::time::Duration) -> bool {
        if self.is_ready() {
            return true;
        }
        let deadline = std::time::Instant::now() + timeout;
        let mut g = self.waiters.0.lock().unwrap();
        while !self.is_ready() {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (ng, _) = self.waiters.1.wait_timeout(g, remaining).unwrap();
            g = ng;
        }
        self.is_ready()
    }
}

use anyhow::{Context, Result as AnyResult};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use travsr_core::{
    DirtySet, Edge, EdgeKind, GcReport, Node, NodeId, ReplaceReport, SafetyPolicy, TestRole, VName,
};
use travsr_error::StoreError;

use crate::fts_tokenize::{build_fuzzy_match_expr_db, tokenize_identifier};

// ── RFC-019 direct-cosine oracle helpers ──────────────────────────────────────

/// Decode a stored embedding BLOB into an `f32` vector.
///
/// All shipping backends write dense little-endian `f32` (BGE 384/768/1024-dim,
/// pre-normalised at index time). Returns `None` for an empty blob or a length
/// that is not a whole number of `f32` lanes — the caller then treats the id as
/// unscoreable (omitted), never as cosine 0.
pub fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Cosine similarity between two equal-length vectors. Returns 0.0 when either
/// side is degenerate (zero norm) or the lengths differ — a defensive, never-NaN
/// result. Vectors are pre-normalised at index time so this is ≈ a dot product,
/// but we normalise anyway so a non-normalising future backend stays correct.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// RFC-019 Option A: score `ids` against a pre-embedded `query_vec` by reading the
/// stored candidate vectors from `embed.db` and computing the true cosine.
///
/// Opens `embed_db_path` read-only (the sidecar owns writes) and reads the blobs
/// for `model_id`. Ids with no stored row or an undecodable blob are **omitted**
/// from the result (contract: omission = "unknown"). Chunked at 500 ids to stay
/// within SQLite's bound-parameter limit.
///
/// O(N·d), N = #ids (≤ ~40 in practice), d = vector dim → microseconds after the
/// local read. Errors from opening/reading `embed.db` surface as `StoreError` so
/// the caller (via `embed_score_fn`) degrades to the FTS-only path.
pub fn score_candidates(
    query_vec: &[f32],
    embed_db_path: &Path,
    model_id: &str,
    ids: &[NodeId],
) -> Result<Vec<(NodeId, f32)>, StoreError> {
    if ids.is_empty() || query_vec.is_empty() || !embed_db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(
        embed_db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| StoreError::Database(e.to_string()))?;

    let mut out: Vec<(NodeId, f32)> = Vec::with_capacity(ids.len());
    for chunk in ids.chunks(500) {
        let placeholders = std::iter::repeat("?")
            .take(chunk.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT node_id, embedding FROM node_embeddings \
             WHERE model_id = ? AND node_id IN ({placeholders})"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| StoreError::Database(e.to_string()))?;
        // Bind model_id first, then the chunk's node ids (i64, matching storage).
        let mut binds: Vec<i64> = Vec::with_capacity(chunk.len());
        for id in chunk {
            binds.push(id.0 as i64);
        }
        let rows = stmt
            .query_map(
                params_from_iter(
                    std::iter::once(rusqlite::types::Value::Text(model_id.to_string()))
                        .chain(binds.into_iter().map(rusqlite::types::Value::Integer)),
                ),
                |r| {
                    let id: i64 = r.get(0)?;
                    let blob: Vec<u8> = r.get(1)?;
                    Ok((id, blob))
                },
            )
            .map_err(|e| StoreError::Database(e.to_string()))?;
        for row in rows {
            let (id, blob) = row.map_err(|e| StoreError::Database(e.to_string()))?;
            if let Some(v) = decode_embedding(&blob) {
                out.push((NodeId(id as u64), cosine(query_vec, &v)));
            }
        }
    }
    Ok(out)
}

// ── SQLite migration structs (T2) ─────────────────────────────────────────────
// Each SQL file becomes a concrete Migration so the runner can apply them
// independently and track progress per-version rather than all-or-nothing.

struct V1Initial;
impl Migration for V1Initial {
    fn version(&self) -> u32 {
        1
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        store.exec_ddl(include_str!("migrations/v1_initial.sql"))
    }
}

struct V2EdgeProvenance;
impl Migration for V2EdgeProvenance {
    fn version(&self) -> u32 {
        2
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // SQLite does not support `ALTER TABLE … ADD COLUMN IF NOT EXISTS`.
        // Guard manually so re-running after a crash (atomicity gap) is safe.
        if !store.column_exists("edges", "provenance")? {
            store.exec_ddl(include_str!("migrations/v2_edge_provenance.sql"))?;
        }
        Ok(())
    }
}

struct V3SignatureFormatVersion;
impl Migration for V3SignatureFormatVersion {
    fn version(&self) -> u32 {
        3
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // INSERT OR IGNORE is idempotent — safe to re-run after a crash.
        store.exec_ddl(include_str!("migrations/v3_signature_format_version.sql"))
    }
}

struct V4EdgesSrcKindIdx;
impl Migration for V4EdgesSrcKindIdx {
    fn version(&self) -> u32 {
        4
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // CREATE INDEX IF NOT EXISTS is idempotent — safe to re-run after a crash.
        store.exec_ddl(include_str!("migrations/v4_edges_src_kind_idx.sql"))
    }
}

struct V5LanguagePackage;
impl Migration for V5LanguagePackage {
    fn version(&self) -> u32 {
        5
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // ALTER TABLE … ADD COLUMN has no IF NOT EXISTS in SQLite.
        // Guard manually so re-running after a crash (atomicity gap) is safe.
        if !store.column_exists("nodes", "package")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN package TEXT NOT NULL DEFAULT ''")?;
        }
        if !store.column_exists("edges", "language")? {
            store.exec_ddl(
                "ALTER TABLE edges ADD COLUMN language TEXT NOT NULL DEFAULT 'typescript'",
            )?;
        }
        // CREATE INDEX IF NOT EXISTS is idempotent.
        store.exec_ddl(include_str!("migrations/v5_language_package.sql"))
    }
}

struct V6EdgeConfidence;
impl Migration for V6EdgeConfidence {
    fn version(&self) -> u32 {
        6
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // ALTER TABLE … ADD COLUMN has no IF NOT EXISTS in SQLite.
        // Guard manually so re-running after a crash (atomicity gap) is safe.
        if !store.column_exists("edges", "confidence")? {
            store.exec_ddl(include_str!("migrations/v6_edge_confidence.sql"))?;
        }
        Ok(())
    }
}

struct V7RbacColumns;
impl Migration for V7RbacColumns {
    fn version(&self) -> u32 {
        7
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // ALTER TABLE … ADD COLUMN has no IF NOT EXISTS in SQLite.
        // Guard manually so re-running after a crash (atomicity gap) is safe.
        if !store.column_exists("nodes", "access_corpus")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN access_corpus TEXT")?;
        }
        // CREATE TABLE IF NOT EXISTS is idempotent — safe to re-run.
        store.exec_ddl(
            "CREATE TABLE IF NOT EXISTS sessions (\
               id      TEXT PRIMARY KEY,\
               corpus  TEXT NOT NULL,\
               created INTEGER NOT NULL DEFAULT (unixepoch()),\
               expires INTEGER NOT NULL\
             )",
        )?;
        Ok(())
    }
}

struct V8NodeLine;
impl Migration for V8NodeLine {
    fn version(&self) -> u32 {
        8
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        if !store.column_exists("nodes", "line")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN line INTEGER")?;
        }
        Ok(())
    }
}

struct V9NodesFts;
impl Migration for V9NodesFts {
    fn version(&self) -> u32 {
        9
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // Both statements use IF NOT EXISTS — idempotent on re-run after a
        // crash between up() and set_schema_version (the atomicity gap).
        store.exec_ddl(include_str!("migrations/v9_nodes_fts.sql"))
    }
}

struct V10FtsVocab;
impl Migration for V10FtsVocab {
    fn version(&self) -> u32 {
        10
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // Both CREATE statements use IF NOT EXISTS — idempotent on re-run.
        store.exec_ddl(include_str!("migrations/v10_fts_vocab.sql"))
    }
}

struct V11FtsSynonyms;
impl Migration for V11FtsSynonyms {
    fn version(&self) -> u32 {
        11
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // Both CREATE statements use IF NOT EXISTS — idempotent on re-run.
        store.exec_ddl(include_str!("migrations/v11_fts_synonyms.sql"))
    }
}

/// RFC-018: plain blob embedding table — no sqlite-vec extension required in main process.
/// EmbedPlugin sidecar opens its own DB connection for ANN queries.
/// Numbered v16 so it runs on DBs already at v15 (kcore-shells branch).
struct V16NodeEmbeddings;
impl Migration for V16NodeEmbeddings {
    fn version(&self) -> u32 {
        16
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // CREATE TABLE / INDEX IF NOT EXISTS — idempotent on re-run.
        store.exec_ddl(include_str!("migrations/v16_node_embeddings.sql"))
    }
}

/// RFC-019: move node_embeddings to embed.db; add CDC tombstone trigger.
///
/// graph.db drops node_embeddings (now owned by the embed sidecar in embed.db)
/// and gains node_tombstones + capture_node_delete trigger so the sidecar can
/// prune stale embeddings between reindex passes without a full-table scan.
struct V17NodeTombstones;
impl Migration for V17NodeTombstones {
    fn version(&self) -> u32 {
        17
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        store.exec_ddl(include_str!("migrations/v17_node_tombstones.sql"))
    }
}

/// RFC-014 WS2: end_line column, symbol_aliases table, edge_sites table, G3 cleanup.
struct V13PhaseBUnification;
impl Migration for V13PhaseBUnification {
    fn version(&self) -> u32 {
        13
    }
    // VACUUM cannot run inside an explicit transaction; skip the SC-H2
    // BEGIN/COMMIT wrapping for this migration (see MigrationRunner::run).
    fn is_transactional(&self) -> bool {
        false
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // ALTER TABLE … ADD COLUMN has no IF NOT EXISTS in SQLite — guard manually.
        if !store.column_exists("nodes", "end_line")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN end_line INTEGER")?;
        }
        // symbol_aliases + edge_sites: CREATE TABLE IF NOT EXISTS is idempotent.
        store.exec_ddl(include_str!("migrations/v13_phase_b_unification.sql"))?;
        // G3: delete SCIP anonymous-local nodes and their incident edges.
        // SCIP ingest stores these with signature `scip:<path>:local <N>`
        // (see `travsr_core::is_scip_anonymous_local`, whose semantics this
        // mirrors: a `:local ` separator followed by digits only, to end of
        // string). The pattern is anchored to the `:local [0-9]` suffix and
        // rejects any non-digit after it, so a *path* containing "local N"
        // (e.g. `src/local 9/util.go`) can never false-positive — its
        // signature has `/local`, not `:local`, before the digits. (The
        // ingest path in scip-reader also drops locals; this cleans up DBs
        // written before that filter existed.)
        const G3_LOCAL_FILTER: &str = "signature GLOB 'scip:*:local [0-9]*' \
             AND signature NOT GLOB 'scip:*:local [0-9]*[^0-9]*'";
        store.exec_ddl(&format!(
            "DELETE FROM edges WHERE src IN (\
               SELECT id FROM nodes WHERE {G3_LOCAL_FILTER}\
             ) OR dst IN (\
               SELECT id FROM nodes WHERE {G3_LOCAL_FILTER}\
             )"
        ))?;
        store.exec_ddl(&format!("DELETE FROM nodes WHERE {G3_LOCAL_FILTER}"))?;
        // O7: reclaim freed pages from G3 deletions.  WAL checkpoint first so the
        // VACUUM can compact the database file (VACUUM is a no-op while WAL frames
        // are not yet flushed to the main file).
        store.exec_ddl("PRAGMA wal_checkpoint(TRUNCATE)")?;
        store.exec_ddl("VACUUM")?;
        Ok(())
    }
}

/// #323 R2: covering reverse index on edges(dst, kind, src).
///
/// Symmetric counterpart to v4 `idx_edges_src_kind_cov`. Eliminates the
/// main-table random I/O on every `get_callers` / `get_blast_radius` reverse
/// traversal (SELECT src FROM edges WHERE dst=? AND kind=?).
struct V14CoveringReverseIdx;
impl Migration for V14CoveringReverseIdx {
    fn version(&self) -> u32 {
        14
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        store.exec_ddl(include_str!("migrations/v14_covering_reverse_idx.sql"))
    }
}

/// V15: k-core shell numbers on nodes — recomputed by travsr-retrieval after every index.
struct V15KcoreShells;
impl Migration for V15KcoreShells {
    fn version(&self) -> u32 {
        15
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        if !store.column_exists("nodes", "shell_number")? {
            store.exec_ddl(include_str!("migrations/v15_kcore_shells.sql"))?;
        }
        Ok(())
    }
}

/// V18: embed_text column — pre-computed AST-skeleton embed text per node.
struct V18EmbedText;
impl Migration for V18EmbedText {
    fn version(&self) -> u32 {
        18
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        if !store.column_exists("nodes", "embed_text")? {
            store.exec_ddl(include_str!("migrations/v18_embed_text.sql"))?;
        }
        Ok(())
    }
}

/// V19: index on nodes(signature) — eliminates full-table scan in
/// `nodes_by_signatures` which is called per UnresolvedCall batch every commit.
struct V19NodesSignatureIdx;
impl Migration for V19NodesSignatureIdx {
    fn version(&self) -> u32 {
        19
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        store.exec_ddl("CREATE INDEX IF NOT EXISTS idx_nodes_signature ON nodes(signature)")
    }
}

/// V20 (#299 F7): one-time purge of orphaned `edge_sites` rows.
///
/// `edge_sites` has no FK cascade, and every ingestion path predating this
/// migration deleted `edges` on re-index/G3 without touching `edge_sites`, so a
/// row whose `src` or `dst` node was deleted or re-id'd lingered forever and
/// surfaced in `find_references` as a phantom occurrence. Newer writes purge
/// these owned/both-direction rows at the source (delete_file / reindex_replace),
/// so this migration only needs to clean the historical backlog once.
struct V20PurgeOrphanEdgeSites;
impl Migration for V20PurgeOrphanEdgeSites {
    fn version(&self) -> u32 {
        20
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        store.exec_ddl(
            "DELETE FROM edge_sites \
             WHERE src NOT IN (SELECT id FROM nodes) \
                OR dst NOT IN (SELECT id FROM nodes)",
        )
    }
}

/// #478 RFC-023: word-segmented lexical precision leg (`nodes_fts_words`) +
/// `nodes.is_noise` structural-noise column. See
/// docs/rfcs/RFC-023-lexical-retrieval-architecture.md §5.1.
///
/// Schema only — no data backfill here. `nodes_fts_words`/`is_noise` values
/// for existing rows are populated by `backfill_fts_words_if_needed`, called
/// after the migration runner at `open()`/`open_in_memory()` (same pattern as
/// `backfill_fts_if_needed`/`backfill_vocab_if_needed`), since `Migration::up`
/// only has DDL access (`exec_ddl`), not the per-row Rust logic
/// (`travsr_core::ident::segments`/`travsr_core::noise::is_structural_noise`)
/// needed to compute them.
struct V21LexicalSplit;
impl Migration for V21LexicalSplit {
    fn version(&self) -> u32 {
        21
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        if !store.column_exists("nodes", "is_noise")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN is_noise INTEGER NOT NULL DEFAULT 0")?;
        }
        store.exec_ddl(include_str!("migrations/v21_lexical_split.sql"))
    }
}

/// #479: `nodes.test_role` index-time test classification column.
///
/// Column-only DDL on the same `ALTER TABLE … in up()` template as
/// [`V21LexicalSplit`]'s `is_noise` (Migration::up has DDL access only). AST-
/// derived roles are written by `travsr-analysis`/`travsr-store` on the next
/// reindex of each file, so existing rows read back as `0` ([`TestRole::None`])
/// until then — the `INTEGER NOT NULL DEFAULT 0` default and the serde default
/// agree, so no read path ever sees a NULL.
struct V22TestRole;
impl Migration for V22TestRole {
    fn version(&self) -> u32 {
        22
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        if !store.column_exists("nodes", "test_role")? {
            store.exec_ddl("ALTER TABLE nodes ADD COLUMN test_role INTEGER NOT NULL DEFAULT 0")?;
        }
        Ok(())
    }
}

/// RFC-027: `ref_resolution_state` (so an unresolved reference can be reported
/// as pending instead of silently vanishing or being guessed at) plus the
/// read-path indexes `Edge.provenance` needs (DEBT-75).
///
/// One migration rather than three. The table and its `resolved_dst` column
/// started as two, which meant creating a table and immediately altering it
/// within the same change, and the index work started as a third. None of them
/// ever shipped — released code is at v22 — so they are collapsed here rather
/// than spending three schema versions on one feature.
///
/// The collapse strands one population: a dev database an earlier revision of
/// this branch already stamped at 24 or 25 sits above this max version and the
/// runner skips it, so it keeps the pre-RFC-027 narrow indexes. That is healed
/// outside the runner, at open, by
/// [`SqliteStore::reconcile_provenance_indexes_if_needed`], which keeps the max
/// schema version at 23. Shipped databases are at v22 and receive this v23 in
/// full, so they never depend on the reconcile.
struct V23RefResolutionState;
impl Migration for V23RefResolutionState {
    fn version(&self) -> u32 {
        23
    }
    fn up(&self, store: &mut dyn StoreMigratable) -> anyhow::Result<()> {
        // CREATE TABLE / INDEX IF NOT EXISTS is idempotent, and the two edge
        // indexes are DROP-then-CREATE because their names already exist.
        store.exec_ddl(include_str!("migrations/v23_ref_resolution_state.sql"))?;
        // `CREATE TABLE IF NOT EXISTS` cannot widen a table an earlier build of
        // this same change already created without `resolved_dst`. Guarded
        // rather than assumed, matching V21/V22, because a missing column
        // surfaces as a confusing runtime error rather than a clean failure.
        if !store.column_exists("ref_resolution_state", "resolved_dst")? {
            store.exec_ddl("ALTER TABLE ref_resolution_state ADD COLUMN resolved_dst INTEGER")?;
        }
        Ok(())
    }
}

/// Build the ordered migration runner for the SQLite backend.
/// Register new SQLite migrations here; version order is enforced by the runner.
fn sqlite_migration_runner() -> MigrationRunner {
    let mut r = MigrationRunner::new();
    r.register(V1Initial);
    r.register(V2EdgeProvenance);
    r.register(V3SignatureFormatVersion);
    r.register(V4EdgesSrcKindIdx);
    r.register(V5LanguagePackage);
    r.register(V6EdgeConfidence);
    r.register(V7RbacColumns);
    r.register(V8NodeLine);
    r.register(V9NodesFts);
    r.register(V10FtsVocab);
    r.register(V11FtsSynonyms);
    r.register(V13PhaseBUnification);
    r.register(V14CoveringReverseIdx);
    r.register(V15KcoreShells);
    r.register(V16NodeEmbeddings);
    r.register(V17NodeTombstones);
    r.register(V18EmbedText);
    r.register(V19NodesSignatureIdx);
    r.register(V20PurgeOrphanEdgeSites);
    r.register(V21LexicalSplit);
    r.register(V22TestRole);
    r.register(V23RefResolutionState);
    r
}

/// RFC-027 section 9.2: one reference the live lane examined, and what became
/// of it.
///
/// `resolved_dst` is `None` for a `pending` row — an abstention resolved to
/// nothing, which is the whole point of it.
#[derive(Debug, Clone)]
pub struct RefResolution {
    pub src: NodeId,
    pub ref_line: u32,
    pub ref_col: u32,
    pub name: String,
    /// `"resolved"` or `"pending"`.
    pub state: &'static str,
    pub resolved_dst: Option<NodeId>,
}

/// #811: what one [`SqliteStore::reconcile_ref_resolution_states`] pass removed.
///
/// Two counters rather than one because they answer different questions: a
/// large `cleared_resolved` after a full rebuild is the stale-marker backlog
/// #811 is about, while `purged_orphans` tracks renames. Both zero means the
/// table was already consistent with the graph, which is what a second run must
/// always report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefReconcileReport {
    /// `pending` rows deleted because `edge_sites` now holds a call site at the
    /// same `(src, line)`.
    pub cleared_resolved: usize,
    /// Rows deleted because their `src` node no longer exists.
    pub purged_orphans: usize,
}

impl RefReconcileReport {
    /// Rows removed in total.
    pub fn total(&self) -> usize {
        self.cleared_resolved + self.purged_orphans
    }
}

/// RFC-027 section 8.3: the `DELETE` behind
/// [`SqliteStore::clear_resolved_pending_refs`], on any connection or open
/// transaction, so the standalone method and the transactional
/// [`SqliteStore::reconcile_ref_resolution_states`] share one statement.
fn clear_resolved_pending_refs_on(conn: &rusqlite::Connection) -> rusqlite::Result<usize> {
    conn.execute(
        "DELETE FROM ref_resolution_state \
         WHERE state = 'pending' \
           AND EXISTS (SELECT 1 FROM edge_sites s \
                       WHERE s.src = ref_resolution_state.src \
                         AND s.line = ref_resolution_state.ref_line)",
        [],
    )
}

/// RFC-027 section 9.2: the `DELETE` behind
/// [`SqliteStore::purge_orphan_ref_resolution_states`]; see
/// [`clear_resolved_pending_refs_on`] for why it is factored out.
fn purge_orphan_ref_resolution_states_on(conn: &rusqlite::Connection) -> rusqlite::Result<usize> {
    conn.execute(
        "DELETE FROM ref_resolution_state \
         WHERE NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = ref_resolution_state.src)",
        [],
    )
}

/// RFC-027 section 12: how the live lane scored against Phase B.
///
/// Deliberately three buckets, not two. `unverifiable` is the honest home for a
/// live claim Phase B left no call-site evidence for — Phase B has its own
/// recall gaps, and the code can change between the edit and the commit, so
/// "Phase B did not produce this" is not the same statement as "the live lane
/// was wrong". Folding those into `disagree` would make the meter pessimistic
/// and unactionable, and a meter nobody believes does not gate anything.
///
/// Precision is therefore reported over the verified subset only, with coverage
/// beside it. Precision without coverage would let 1.0 over two of five hundred
/// claims read as a passing grade.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LivePrecision {
    /// Live claims Phase B resolved to the same target at the same call site.
    pub agree: u64,
    /// Live claims Phase B resolved to a *different* target at that site. These
    /// are true false positives, and the number the shipping gate exists for.
    pub disagree: u64,
    /// Live claims with no ratified call-site evidence either way.
    pub unverifiable: u64,
}

impl LivePrecision {
    /// Agreement over the verified subset, or `None` when nothing was verifiable.
    ///
    /// `None` rather than `1.0`: a lane that resolved nothing verifiable has not
    /// earned a perfect score, and returning one would let an empty sample clear
    /// the shipping gate.
    pub fn precision(&self) -> Option<f64> {
        let verified = self.agree + self.disagree;
        (verified > 0).then(|| self.agree as f64 / verified as f64)
    }

    /// Fraction of live claims that could be checked at all.
    pub fn coverage(&self) -> f64 {
        let total = self.agree + self.disagree + self.unverifiable;
        if total == 0 {
            return 0.0;
        }
        (self.agree + self.disagree) as f64 / total as f64
    }

    /// Total live claims this sample covers.
    pub fn claims(&self) -> u64 {
        self.agree + self.disagree + self.unverifiable
    }
}

/// The storage interface every Travsr backend must satisfy.
///
/// All methods return `Result<T, StoreError>` (from `travsr-error`).
/// Callers that use `?` in an `anyhow::Result` context get automatic
/// conversion via `anyhow::Error: From<StoreError>`.
pub trait Store {
    /// Persist a node, returning its assigned id.
    fn put_node(&mut self, node: &Node) -> Result<NodeId, StoreError>;
    /// Persist an edge.
    fn put_edge(&mut self, edge: &Edge) -> Result<(), StoreError>;
    /// Look up a node by id.
    fn get_node(&self, id: NodeId) -> Result<Option<Node>, StoreError>;
    /// Return every outgoing edge from `src`.
    fn iter_edges_from(&self, src: NodeId) -> Result<Vec<Edge>, StoreError>;
    /// Return outgoing edges from `src` filtered to a single [`EdgeKind`].
    ///
    /// Implementations **must** use an indexed `WHERE kind = ?` clause so that
    /// PPR traversal (which filters by kind at every step) meets the p95 < 50ms
    /// budget on graphs up to the MVP ceiling (75M nodes / 2.5B edges).
    ///
    /// The default implementation delegates to [`Store::iter_edges_from`] and
    /// filters in Rust — backends without a kind index should override this.
    fn iter_edges_from_kind(&self, src: NodeId, kind: EdgeKind) -> Result<Vec<Edge>, StoreError> {
        Ok(self
            .iter_edges_from(src)?
            .into_iter()
            .filter(|e| e.kind == kind)
            .collect())
    }
    /// Return every incoming edge to `dst`.
    fn iter_edges_to(&self, dst: NodeId) -> Result<Vec<Edge>, StoreError>;
    /// Batch-fetch nodes by id. Unknown ids are silently skipped.
    /// Output order is not guaranteed to match input order.
    fn get_nodes(&self, ids: &[NodeId]) -> Result<Vec<Node>, StoreError>;
    /// Return all outgoing edges for every node in `srcs` in a single operation.
    ///
    /// Output order is undefined — edges are not grouped by source.
    /// The default implementation loops over [`Store::iter_edges_from`]; backends
    /// with an indexed `src` column should override with a batch `IN (…)` query.
    fn iter_edges_from_batch(&self, srcs: &[NodeId]) -> Result<Vec<Edge>, StoreError> {
        let mut out = Vec::new();
        for &src in srcs {
            out.extend(self.iter_edges_from(src)?);
        }
        Ok(out)
    }
}

/// One file's worth of parsed graph data, ready to write in a batch transaction.
///
/// Produced by the parallel parse workers in `travsr-daemon` and consumed by
/// `SqliteStore::write_file_graphs_batch`.  Keeping parse and write decoupled
/// lets N CPU threads produce data while a single writer thread owns the store.
#[derive(Debug)]
pub struct FileGraph {
    /// Repo-relative path string, e.g. `"src/lib.rs"`.
    pub vname_path: String,
    /// SHA-256 hex of the file, used to update the `files` hash table.
    pub new_hash: String,
    pub nodes: Vec<travsr_core::Node>,
    pub edges: Vec<travsr_core::Edge>,
    /// RFC-027 #813: the file's current source text, so the write path can stamp
    /// each definition's `body_hash` from the same content it is deriving edges
    /// from. That co-write is what makes the hash a sound preservation witness:
    /// a later save preserves a definition only when its body is byte-identical
    /// to what these edges were built from. `None` (e.g. an unchanged file with
    /// no nodes, or a caller that does not supply it) leaves the hashes NULL,
    /// which forgoes preservation for those definitions until their next
    /// re-resolution — the fail-safe direction, never a wrong preservation.
    pub source: Option<String>,
}

/// Aggregate counts returned by [`SqliteStore::write_file_graphs_batch`].
#[derive(Debug, Default)]
pub struct BatchWriteCounts {
    pub nodes_upserted: u64,
    pub edges_upserted: u64,
    pub files_written: u64,
}

/// #478 RFC-023 §5.4: a fused-search result carrying the channel-separated
/// scores that let the abstention gate distinguish "a real BM25-scale leg
/// matched" from "only the position-derived exact leg (or L2-A/embed)
/// contributed" — the distinction Evidence E's collapsed single float
/// couldn't make.
#[derive(Debug, Clone)]
pub struct FusedHit {
    pub node: Node,
    /// Existing pre-#478 semantics, unchanged: the max natural (BM25-scale or
    /// synthetic-position) score across every stage the node appeared in.
    /// `search_nodes_fuzzy`/`search_nodes_fuzzy_filtered` read only this field
    /// (via `.node`), so their output is byte-for-byte unchanged by #478.
    pub natural: f32,
    /// `Some` only when Leg B (word) or Leg C (trigram) — a real BM25-scale
    /// leg — matched this node. `None` for a Leg-A(exact)-only, L2-A-only, or
    /// embed-only hit. This is what makes the abstention gate read a live
    /// signal instead of the position-derived Stage-1 float (Evidence E).
    pub bm25_natural: Option<f32>,
    /// `Some(rank)` (0-based, best-first) only when Leg A (exact/name) matched.
    pub exact_rank: Option<usize>,
}

/// Stages 2/B/3/4 of [`SqliteStore::fused_search_scored`], shared with
/// [`SqliteStore::explain_leg_scores`]. See [`SqliteStore::compute_lexical_stages`].
struct LexicalStages {
    stage2: Vec<(Node, f32)>,
    stage_b: Vec<(Node, f32)>,
    stage3: Vec<(Node, f32)>,
    stage4: Vec<(Node, f32)>,
}

/// #478 RFC-023 §6.1: per-leg raw scores (best-first) for `travsr explain`,
/// from [`SqliteStore::explain_leg_scores`]. Each `Vec` is exactly one leg's
/// own ranked output, before RRF fusion or kind diversification — a node's
/// position in a leg's `Vec` is that leg's rank, and the paired `f32` is that
/// leg's raw (not RRF) score.
pub struct ExplainLegs {
    pub exact: Vec<(Node, f32)>,
    pub word: Vec<(Node, f32)>,
    pub trigram: Vec<(Node, f32)>,
    pub l2a: Vec<(Node, f32)>,
    pub embed: Vec<(Node, f32)>,
}

/// Split a node-count vs map-count difference into two non-negative figures.
///
/// `saturating_sub` on a signed type saturates at `i64::MIN`, not at zero — a
/// detail easy to read past, because on an unsigned type it does clamp at zero,
/// which is plainly what was intended here. Both backfill gates used it that
/// way, so whenever the map held more rows than `nodes` (stale entries left by
/// a delete that ran outside this store instance) the log reported a negative
/// count of missing rows:
///
/// ```text
/// #478: backfilling nodes_fts_words + is_noise  missing=-299 stale=299
/// ```
///
/// Harmless to the backfill itself, which is driven by a `NOT IN` query rather
/// than by these numbers, but it points a reader diagnosing an index problem in
/// exactly the wrong direction.
fn backfill_counts(node_count: i64, map_count: i64) -> (i64, i64) {
    // `saturating_sub` still earns its place — it handles overflow — but the
    // zero clamp has to be explicit on a signed type.
    (
        node_count.saturating_sub(map_count).max(0),
        map_count.saturating_sub(node_count).max(0),
    )
}

/// #509: what makes a file at a path *that* file, rather than merely a file
/// with that name. Compared for equality only; the members have no meaning
/// beyond "these two stats came from the same file or they did not". One shape
/// for both platforms so the rest of the code needs no `cfg`: unix fills two
/// members and leaves the third zero, windows fills all three.
type FileIdentity = (u64, u64, u64);

/// Identity of the file currently at `path`, or `None` when it cannot be
/// stat-ed (absent, or unreadable), which callers treat the same as gone.
///
/// The repository has no portable file-identity helper to reuse: the only
/// prior stat-based identity work is `travsr-ipc`'s `owner_uid`, which is
/// `#[cfg(unix)]` with no Windows counterpart. So this is gated per platform.
///
/// unix: `(st_dev, st_ino)` is the kernel's own identity for a file and is
/// exact here. A file that is unlinked while a descriptor is still open on it
/// keeps its inode allocated until the last descriptor closes, so as long as
/// [`SqliteStore::embed_meta_conn`] holds the old embed.db open, its inode
/// number cannot be handed to the replacement. Different file, different
/// identity, always.
///
/// windows: `std::os::windows::fs::MetadataExt::file_index`, the direct
/// analogue named in #509, is still unstable behind `windows_by_handle`
/// (rust-lang/rust#63010) and cannot be called on stable Rust. This uses the
/// stable triple `(creation_time, last_write_time, file_size)` instead. That
/// is a strong heuristic rather than an identity: NTFS tunneling can carry a
/// deleted file's creation time onto a same-named replacement created within
/// ~15 s, so in that window the other two members carry the comparison. It
/// errs toward *extra* reopens (any write moves `last_write_time`), never
/// toward keeping a dead handle, which is the safe direction: a spurious
/// reopen costs one cache miss, a missed one serves stale answers forever.
/// Note too that SQLite opens database files on Windows without
/// `FILE_SHARE_DELETE`, so unlinking embed.db out from under a live connection
/// fails there outright; this branch covers the rename-over case and the case
/// where the connection was already dropped and reopened.
fn file_identity(path: &Path) -> Option<FileIdentity> {
    let md = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        Some((md.dev(), md.ino(), 0))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        Some((md.creation_time(), md.last_write_time(), md.file_size()))
    }
}

/// #509: fold a file identity and a `PRAGMA data_version` reading into the one
/// opaque `u64` that [`SqliteStore::embed_data_version`] hands back.
///
/// Hashing rather than, say, packing the two into halves of the `u64`, because
/// neither input has a bounded range to pack into. The consumer is the daemon's
/// query cache, which only ever compares this for equality (`CacheKey` derives
/// `PartialEq`/`Hash` and nothing orders it), so an unordered token is all the
/// contract needs. Determinism is only required within a single process: the
/// cache is in memory and does not outlive the daemon.
fn embed_version_token(identity: FileIdentity, version: u64) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    identity.hash(&mut h);
    version.hash(&mut h);
    h.finish()
}

/// SQLite-backed store. The MVP target — zero setup, single file on disk.
pub struct SqliteStore {
    conn: Connection,
    staging_active: bool,
    /// RFC-018 Step 4: optional semantic ANN hook injected by the daemon at
    /// store-open time. `None` by default — zero cost when no embed plugin is
    /// running. The store crate has zero knowledge of the plugin system; only the
    /// daemon injects this via `set_embed_knn_hook`.
    embed_knn_hook: Option<EmbedKnnHook>,
    /// RFC-019: optional direct-cosine oracle hook injected beside `embed_knn_hook`
    /// by the same injector (daemon / MCP). `None` by default — the FTS-only path
    /// is identical when absent.
    embed_score_hook: Option<EmbedScoreHook>,
    /// Arm-state for the embed hook (set alongside `embed_knn_hook` by injectors
    /// that support warm-up signalling). `None` for legacy/test injectors and
    /// in-memory stores — in that case `embed_ready` falls back to hook presence.
    embed_readiness: Option<Arc<EmbedReadiness>>,
    /// RFC-019: sibling embed.db path (e.g. `.travsr/embed.db`). `None` for
    /// in-memory stores. Used by `embed_progress` to ATTACH embed.db and count
    /// embedded nodes without polluting the graph.db WAL with embedding BLOBs.
    embed_db_path: Option<std::path::PathBuf>,
    /// This store's own graph.db path. `None` for in-memory stores.
    ///
    /// Added because `repo_root_from_db_path` was deriving the repository root
    /// from `embed_db_path`, which happens to be graph.db's sibling and is
    /// declared for something else entirely. Relocating embed.db (models already
    /// live outside the repo) would have silently brought #747 back, and no
    /// fixture would have caught it (#749 review).
    db_path: Option<std::path::PathBuf>,
    /// #464 follow-up: persistent read-only connection to the sibling embed.db,
    /// lazily opened on first [`Self::embed_data_version`] call. Persistent
    /// because `PRAGMA data_version` only moves relative to prior reads on the
    /// SAME connection — a fresh connection per call would never observe a
    /// change. `RefCell` keeps the lazy open behind `&self`; the store is not
    /// `Sync` anyway (callers wrap it in a `Mutex`).
    ///
    /// #509: the [`FileIdentity`] the connection was opened against is stored
    /// alongside it, in the same slot so the two can never drift apart. A
    /// connection outlives the file it was opened on when embed.db is unlinked,
    /// so the identity is the only way to notice that the path has come to name
    /// a different file.
    embed_meta_conn: std::cell::RefCell<Option<(Connection, FileIdentity)>>,
    /// #376 Phase 2: optional doc-space semantic hook, injected beside
    /// `embed_knn_hook` by the same injector. Same shape as `EmbedKnnHook`
    /// (reused, not a distinct type — both are `Fn(&str, u32) -> Result<Vec<(NodeId, f32)>, StoreError>`)
    /// but a separate field/slot since the two spaces are independent round
    /// trips (see `travsr-plugin-host::EmbedSupervisor::doc_knn_hook`).
    /// `None` by default — zero cost when docs are disabled or unsupported.
    embed_doc_knn_hook: Option<EmbedKnnHook>,
}

impl Drop for SqliteStore {
    fn drop(&mut self) {
        // Best-effort WAL truncation on clean shutdown — shrinks graph.db-wal
        // to zero bytes so the next open starts with no WAL replay overhead.
        // TRUNCATE mode requires an exclusive lock; it silently skips if any
        // reader holds a shared lock, which is always a safe WAL state.
        let _ = self.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
    }
}

impl SqliteStore {
    /// The repository root to resolve paths against, live rather than remembered.
    ///
    /// `meta.repo_root` is stamped by `init` and by `reindex_files`, so it names
    /// wherever the repository lived at the last index. Move the checkout, clone
    /// it to a second path, mount it at a different point in CI, or rename a home
    /// directory, and every consumer of that key is handed a path that no longer
    /// exists until something reindexes (#747). The database travels with the
    /// repository, so its own location covers that window.
    ///
    /// `None` only when there is neither a recorded value nor a derivable one:
    /// an in-memory store, or a database outside a `.travsr` directory, with no
    /// `repo_root` in meta. A store with nothing to derive from still returns a
    /// recorded value, whether or not that path still resolves, since there is
    /// nothing to contradict it.
    ///
    /// Lives here rather than in a caller because there are six readers of
    /// `meta.repo_root` across `travsr-mcp` alone, and the first version of this
    /// rule was private to `tools.rs`, so `observability.rs` and `seed.rs` could
    /// not use it even had they wanted to. A seventh reader added later now gets
    /// the right behaviour by default instead of by remembering (#749 review).
    ///
    /// Precedence, in order:
    ///
    /// 1. The derived root, when it is a repository in its own right. A `.travsr`
    ///    copied into a second checkout still names the first, and that one
    ///    usually still exists, so preferring the stored value there reads files
    ///    out of the wrong tree while looking entirely confident.
    /// 2. The stored value, while it resolves. This is the `--db` case: a
    ///    database opened from outside any repository is still about the real
    ///    one, and only the stored value knows where that is.
    /// 3. Whichever of the two is left.
    pub fn resolve_repo_root(&self) -> Option<std::path::PathBuf> {
        let stored = self
            .get_meta("repo_root")
            .ok()
            .flatten()
            .filter(|r| !r.is_empty())
            .map(std::path::PathBuf::from);
        let derived = self.repo_root_from_db_path();

        // `.git` is a directory in a normal checkout and a file in a worktree or
        // submodule, so `exists` covers both.
        if derived.as_ref().is_some_and(|d| d.join(".git").exists()) {
            return derived;
        }
        match stored {
            Some(p) if p.is_dir() => Some(p),
            stored => derived.or(stored),
        }
    }

    /// The repository this store's database sits inside, derived from the
    /// database's own location rather than from anything recorded at index time.
    ///
    /// `None` for an in-memory store, which has no location to derive from, and
    /// for any database not sitting at `<repo>/.travsr/graph.db`.
    pub fn repo_root_from_db_path(&self) -> Option<std::path::PathBuf> {
        // `<repo>/.travsr/graph.db` -> `<repo>`, and only that shape.
        //
        // The directory name is checked rather than assumed. A database opened
        // from somewhere else (a test fixture, an explicit `--db`) would
        // otherwise yield whatever happens to sit two levels up, and a confident
        // wrong root is worse than none: callers would resolve `vname.path`
        // against an unrelated directory instead of degrading to metadata-only.
        let dir = self.db_path.as_deref()?.parent()?;
        if dir.file_name()? != ".travsr" {
            return None;
        }
        dir.parent().map(std::path::Path::to_path_buf)
    }

    /// Open (or create) a SQLite-backed store at `path`, enabling WAL and
    /// running any pending migrations via [`MigrationRunner`].
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        (|| -> AnyResult<Self> {
            let conn = Connection::open(path)
                .with_context(|| format!("opening sqlite database at {}", path.display()))?;
            Self::configure(&conn)?;
            // Bootstrap the meta table before the runner reads the schema version.
            Self::bootstrap_meta(&conn)?;
            let mut store = Self {
                conn,
                staging_active: false,
                embed_knn_hook: None,
                embed_score_hook: None,
                embed_readiness: None,
                embed_db_path: Some(path.with_file_name("embed.db")),
                db_path: Some(path.to_path_buf()),
                embed_meta_conn: std::cell::RefCell::new(None),
                embed_doc_knn_hook: None,
            };
            sqlite_migration_runner()
                .run(&mut store)
                .context("running SQLite migrations")?;
            store
                .reconcile_provenance_indexes_if_needed()
                .context("reconciling RFC-027 provenance indexes (issue B)")?;
            store
                .ensure_body_hash_column()
                .context("adding RFC-027 #813 node body_hash column")?;
            store
                .ensure_edge_sites_col_column()
                .context("adding RFC-027 #813 P2 edge_sites col column")?;
            store
                .backfill_fts_if_needed()
                .context("backfilling FTS index")?;
            store
                .backfill_fts_words_if_needed()
                .context("backfilling FTS word index (#478)")?;
            store
                .backfill_vocab_if_needed()
                .context("backfilling fts_vocab index (L2-A)")?;
            store
                .seed_synonyms_if_empty()
                .context("seeding fts_synonyms (RFC-012 A2 F1)")?;
            store
                .run_pragma_optimize()
                .context("PRAGMA optimize after open")?;
            Ok(store)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Open an existing store read-only for query commands (#318 O1).
    ///
    /// Skips everything [`SqliteStore::open`] does that is a write-path
    /// concern — the migration runner, FTS backfill, vocab backfill, and
    /// synonym seeding — which makes the cold CLI path substantially cheaper.
    /// The schema version is still verified: a store pending migrations
    /// returns an error so the caller can fall back to a full `open()`.
    pub fn open_read_only(path: &Path) -> Result<Self, StoreError> {
        (|| -> AnyResult<Self> {
            anyhow::ensure!(
                path.exists(),
                "no graph database at {}; run `travsr init`",
                path.display()
            );
            let conn = Connection::open_with_flags(
                path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .with_context(|| format!("opening sqlite database read-only at {}", path.display()))?;
            // Read-path pragmas only. journal_mode=WAL is a persisted property
            // of the database file; setting pragmas that write (WAL switch,
            // autocheckpoint) is neither possible nor needed here.
            // SC-C1: busy timeout prevents SQLITE_BUSY hard-fail when the daemon
            // write lock is held briefly (e.g. post-commit reindex flush).
            conn.busy_timeout(std::time::Duration::from_secs(5))
                .context("setting busy_timeout (read-only)")?;
            conn.pragma_update(None, "cache_size", -Self::cache_size_kib())
                .context("setting cache_size (read-only)")?;
            conn.pragma_update(None, "temp_store", "MEMORY")
                .context("setting temp_store=MEMORY (read-only)")?;
            conn.pragma_update(None, "mmap_size", Self::mmap_size_bytes())
                .context("setting mmap_size (read-only)")?;
            conn.pragma_update(None, "query_only", "ON")
                .context("setting query_only=ON")?;
            let store = Self {
                conn,
                staging_active: false,
                embed_knn_hook: None,
                embed_score_hook: None,
                embed_readiness: None,
                embed_db_path: Some(path.with_file_name("embed.db")),
                db_path: Some(path.to_path_buf()),
                embed_meta_conn: std::cell::RefCell::new(None),
                embed_doc_knn_hook: None,
            };
            let current = store
                .schema_version()
                .context("reading schema version (read-only)")?;
            let latest = sqlite_migration_runner().latest_version();
            anyhow::ensure!(
                current == latest,
                "schema v{current} ≠ expected v{latest}, pending migrations; reopen writable"
            );
            Ok(store)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Open an in-memory SQLite store. Used in tests; WAL is not available
    /// on `:memory:` connections, so journal mode falls back to MEMORY.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        (|| -> AnyResult<Self> {
            let conn = Connection::open_in_memory().context("opening in-memory sqlite database")?;
            Self::bootstrap_meta(&conn)?;
            let mut store = Self {
                conn,
                staging_active: false,
                embed_knn_hook: None,
                embed_score_hook: None,
                embed_readiness: None,
                embed_db_path: None,
                db_path: None,
                embed_meta_conn: std::cell::RefCell::new(None),
                embed_doc_knn_hook: None,
            };
            sqlite_migration_runner()
                .run(&mut store)
                .context("running SQLite migrations (in-memory)")?;
            store
                .reconcile_provenance_indexes_if_needed()
                .context("reconciling RFC-027 provenance indexes in-memory (issue B)")?;
            store
                .ensure_body_hash_column()
                .context("adding RFC-027 #813 body_hash column (in-memory)")?;
            store
                .ensure_edge_sites_col_column()
                .context("adding RFC-027 #813 P2 edge_sites col column (in-memory)")?;
            store
                .backfill_fts_if_needed()
                .context("backfilling FTS index (in-memory)")?;
            store
                .backfill_fts_words_if_needed()
                .context("backfilling FTS word index in-memory (#478)")?;
            store
                .backfill_vocab_if_needed()
                .context("backfilling fts_vocab index in-memory (L2-A)")?;
            store
                .seed_synonyms_if_empty()
                .context("seeding fts_synonyms in-memory (RFC-012 A2 F1)")?;
            Ok(store)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Inject the RFC-018 Step 4 semantic-ANN hook.
    ///
    /// Called by the daemon after opening the store, when the embed supervisor
    /// is active and the stored model_id matches the plugin's model_id. The hook
    /// fires at the end of `search_nodes_fuzzy` (only when Steps 1–3 all miss).
    pub fn set_embed_knn_hook(&mut self, hook: EmbedKnnHook) {
        self.embed_knn_hook = Some(hook);
    }

    /// Register the arm-state handle shared with the injector's background init
    /// thread. Enables `embed_ready` / `wait_embed_ready` to reflect true warm-up
    /// state rather than mere hook presence.
    pub fn set_embed_readiness(&mut self, readiness: Arc<EmbedReadiness>) {
        self.embed_readiness = Some(readiness);
    }

    /// Whether the embed hook is armed and ready to answer KNN queries.
    ///
    /// When an injector registered readiness (the MCP stdio path), reflects its
    /// armed state. When no readiness was registered (legacy daemon path, the
    /// `travsr ask` path, tests), returns `true` — those callers opt out of
    /// warm-up tracking, so the "warming" state never applies and behaviour is
    /// unchanged. Only consulted when `has_embed` is true at the call site.
    pub fn embed_ready(&self) -> bool {
        match &self.embed_readiness {
            Some(r) => r.is_ready(),
            None => true,
        }
    }

    /// Whether arming finished without installing a KNN hook, so the injected
    /// meta-hook can only ever return an empty result.
    ///
    /// `has_embed` is true whenever a hook is present, and the MCP injector
    /// installs its meta-hooks unconditionally so `initialize` never blocks on
    /// the sidecar. That makes hook presence a poor proxy for "semantic search
    /// works": this reports the difference. `false` for callers that registered
    /// no readiness (daemon, `travsr ask`, tests), which is correct there -
    /// those paths install a hook only once it is real.
    pub fn embed_disabled(&self) -> bool {
        match &self.embed_readiness {
            Some(r) => r.is_disabled(),
            None => false,
        }
    }

    /// Block up to `timeout` for the embed hook to arm, returning the final ready
    /// state. Lets the opening query of a session use embeddings instead of
    /// silently degrading to lexical-only during sidecar startup. Returns `true`
    /// immediately when no readiness was registered (same opt-out as `embed_ready`).
    pub fn wait_embed_ready(&self, timeout: std::time::Duration) -> bool {
        match &self.embed_readiness {
            Some(r) => r.wait(timeout),
            None => true,
        }
    }

    /// Return a callable wrapper around the embed KNN hook, or `None` when no
    /// hook has been injected (embed plugin not installed or index not built).
    ///
    /// The closure owns an `Arc` clone so it can outlive the `SqliteStore`
    /// borrow. Errors from the underlying hook are swallowed into an empty vec.
    pub fn embed_knn_fn(&self) -> Option<impl Fn(&str, u32) -> Vec<(NodeId, f32)>> {
        let hook = self.embed_knn_hook.clone()?;
        Some(move |query: &str, k: u32| -> Vec<(NodeId, f32)> {
            hook(query, k).unwrap_or_default()
        })
    }

    /// #376 Phase 2: inject the doc-space semantic hook beside `set_embed_knn_hook`.
    /// Called by the same injector, only when `EmbedSupervisor::doc_knn_hook`
    /// returned `Some` (sidecar supports it and repo has a doc-space index).
    pub fn set_embed_doc_knn_hook(&mut self, hook: EmbedKnnHook) {
        self.embed_doc_knn_hook = Some(hook);
    }

    /// Return a callable wrapper around the doc-space hook, or `None` when
    /// absent (docs disabled, old sidecar, or repo has no doc-chunk nodes).
    /// Mirrors [`Self::embed_knn_fn`] exactly — same swallow-errors-to-empty
    /// contract.
    pub fn embed_doc_knn_fn(&self) -> Option<impl Fn(&str, u32) -> Vec<(NodeId, f32)>> {
        let hook = self.embed_doc_knn_hook.clone()?;
        Some(move |query: &str, k: u32| -> Vec<(NodeId, f32)> {
            hook(query, k).unwrap_or_default()
        })
    }

    /// #376 Phase 2 cross-encoder prototype: fetch `embed_text` (the
    /// anchor-prefixed doc prose, plan §3.2) for a batch of node ids, so the
    /// docs lane can rerank KNN candidates against the query text instead of
    /// gating purely on raw embedding cosine. Chunked like [`Self::get_nodes`]
    /// to stay under `SQLITE_MAX_VARIABLE_NUMBER`. An id with no row, or a
    /// NULL `embed_text` (deleted between KNN and this call, or a non-doc
    /// node), is silently omitted rather than erroring — the caller treats a
    /// missing entry as "no text to rerank against."
    pub fn get_nodes_embed_text(
        &self,
        ids: &[NodeId],
    ) -> Result<HashMap<NodeId, String>, StoreError> {
        const CHUNK: usize = 999;
        let mut out = HashMap::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, embed_text FROM nodes \
                 WHERE id IN ({placeholders}) AND embed_text IS NOT NULL"
            );
            (|| -> AnyResult<()> {
                let mut stmt = self
                    .conn
                    .prepare(&sql)
                    .context("preparing get_nodes_embed_text query")?;
                let id_params: Vec<i64> = chunk.iter().map(|id| node_id_to_i64(*id)).collect();
                let rows = stmt
                    .query_map(params_from_iter(id_params.iter()), |row| {
                        let id = i64_to_node_id(row.get::<_, i64>(0)?);
                        let text: String = row.get(1)?;
                        Ok((id, text))
                    })
                    .context("executing get_nodes_embed_text query")?;
                for r in rows {
                    let (id, text) = r.context("decoding get_nodes_embed_text row")?;
                    out.insert(id, text);
                }
                Ok(())
            })()
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }
        Ok(out)
    }

    /// Inject the RFC-019 direct-cosine oracle hook. Injected beside
    /// `set_embed_knn_hook` by the same injector (daemon / MCP) once the sidecar
    /// is warm.
    pub fn set_embed_score_hook(&mut self, hook: EmbedScoreHook) {
        self.embed_score_hook = Some(hook);
    }

    /// Return a callable wrapper around the RFC-019 score hook, or `None` when no
    /// hook has been injected. Mirrors [`Self::embed_knn_fn`]: the closure owns an
    /// `Arc` clone so it outlives the borrow, and a hook error is swallowed into an
    /// empty vec — a degraded score reads as "no candidates scored" (unknown),
    /// never as a hard failure or a false cosine 0.
    #[allow(clippy::type_complexity)]
    pub fn embed_score_fn(&self) -> Option<impl Fn(&str, &[NodeId]) -> Vec<(NodeId, f32)>> {
        let hook = self.embed_score_hook.clone()?;
        Some(move |query: &str, ids: &[NodeId]| -> Vec<(NodeId, f32)> {
            hook(query, ids).unwrap_or_default()
        })
    }

    /// Returns `true` when an embed.db sibling file exists for this store.
    /// Used to differentiate "embedding in progress / Phase 1 done but hook not active yet"
    /// from "embedding never initialized — user must run `travsr embed init`".
    pub fn has_embed_db(&self) -> bool {
        self.embed_db_path
            .as_deref()
            .map(|p| p.exists())
            .unwrap_or(false)
    }

    /// Batch in-degree counts (number of incoming edges) for the given node IDs.
    ///
    /// Nodes with no incoming edges are included with a count of 0.
    /// Chunked at 500 IDs per query to stay within SQLite's parameter limit.
    pub fn in_degrees(&self, ids: &[NodeId]) -> AnyResult<std::collections::HashMap<NodeId, u32>> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let mut result: std::collections::HashMap<NodeId, u32> =
            ids.iter().map(|&id| (id, 0u32)).collect();
        for chunk in ids.chunks(500) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT dst, COUNT(*) as cnt FROM edges WHERE dst IN ({placeholders}) GROUP BY dst"
            );
            let mut stmt = self.conn.prepare_cached(&sql)?;
            let params: Vec<rusqlite::types::Value> = chunk
                .iter()
                .map(|id| rusqlite::types::Value::Integer(id.0 as i64))
                .collect();
            let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
            while let Some(row) = rows.next()? {
                let dst: i64 = row.get(0)?;
                let cnt: u32 = row.get(1)?;
                result.insert(NodeId(dst as u64), cnt);
            }
        }
        Ok(result)
    }

    /// Create the `meta` table if it does not already exist.
    /// Must run before the migration runner, which uses meta to read the version.
    fn bootstrap_meta(conn: &Connection) -> AnyResult<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )
        .context("bootstrapping meta table")
    }

    /// Page-cache size in kibibytes, read from `TRAVSR_STORE_CACHE_MB` (default 128).
    /// Negative value as required by SQLite's cache_size pragma convention.
    fn cache_size_kib() -> i64 {
        std::env::var("TRAVSR_STORE_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(128)
            * 1024
    }

    /// Memory-mapped I/O size in bytes, read from `TRAVSR_STORE_MMAP_GB` (default 0 — disabled).
    /// Off by default to avoid per-process RSS bloat in multi-process local deployments.
    /// Enable only on a dedicated single-instance OCI A1 deployment.
    fn mmap_size_bytes() -> i64 {
        std::env::var("TRAVSR_STORE_MMAP_GB")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|&gb| gb > 0)
            .map(|gb| gb * 1024 * 1024 * 1024)
            .unwrap_or(0)
    }

    fn configure(conn: &Connection) -> AnyResult<()> {
        // SC-C1: 5-second retry on SQLITE_BUSY so a second concurrent writer
        // (daemon + CLI, or two Phase B workers) retries instead of hard-failing.
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .context("setting busy_timeout")?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("enabling WAL journal mode")?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .context("setting synchronous=NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .context("enabling foreign_keys pragma")?;
        // Cap page cache (negative value = kibibytes). Default 128 MB; override
        // via TRAVSR_STORE_CACHE_MB. Without a cap the default grows with graph
        // size and was the primary cause of ~700 MB RSS per daemon process.
        conn.pragma_update(None, "cache_size", -Self::cache_size_kib())
            .context("setting cache_size")?;
        // Checkpoint the WAL every 500 pages (~2 MB) instead of the SQLite
        // default of 1000 pages, so the WAL file stays small at rest.
        conn.pragma_update(None, "wal_autocheckpoint", 500i32)
            .context("setting wal_autocheckpoint")?;
        // Keep temp tables in memory instead of writing to /tmp files.
        conn.pragma_update(None, "temp_store", "MEMORY")
            .context("setting temp_store=MEMORY")?;
        // Memory-mapped I/O. Off by default (TRAVSR_STORE_MMAP_GB=0) to avoid
        // per-process RSS bloat; enable only on a dedicated OCI A1 instance.
        conn.pragma_update(None, "mmap_size", Self::mmap_size_bytes())
            .context("setting mmap_size")?;
        Ok(())
    }

    /// Enable or disable bulk-init mode pragmas.
    ///
    /// `enable=true` skips WAL fsyncs (`synchronous=OFF`) and expands the page
    /// cache to 256 MB for the duration of a full `init_repo` run.  Safe because
    /// init is re-runnable: an OS crash may leave a partially-written WAL, but
    /// the next `travsr init` heals it.  NEVER use on the commit-hook path.
    ///
    /// `enable=false` restores the values set by [`configure`].
    pub fn set_bulk_init_mode(&mut self, enable: bool) -> anyhow::Result<()> {
        if enable {
            self.conn
                .pragma_update(None, "synchronous", "OFF")
                .context("bulk init: setting synchronous=OFF")?;
            self.conn
                .pragma_update(None, "cache_size", -262144i64)
                .context("bulk init: setting cache_size=256MB")?;
        } else {
            self.conn
                .pragma_update(None, "synchronous", "NORMAL")
                .context("bulk init: restoring synchronous=NORMAL")?;
            self.conn
                .pragma_update(None, "cache_size", -Self::cache_size_kib())
                .context("bulk init: restoring cache_size")?;
        }
        Ok(())
    }

    /// Run a bounded `PRAGMA optimize` so the query planner has real cardinality
    /// estimates for `nodes` and `edges`.
    ///
    /// Without `ANALYZE`, SQLite uses hardcoded defaults and may pick a full table
    /// scan over an index seek on a fresh database. `analysis_limit = 1000` bounds
    /// the scan cost; `PRAGMA optimize` skips tables whose stats are already fresh.
    /// Safe to call from `&self` — the pragma writes to `sqlite_stat1` internally
    /// but does not require a write transaction from the caller.
    ///
    /// Call sites: end of [`SqliteStore::open`] and end of a full `travsr init`.
    pub fn run_pragma_optimize(&self) -> Result<(), StoreError> {
        self.conn
            .execute_batch("PRAGMA analysis_limit = 1000; PRAGMA optimize;")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return SQLite's `PRAGMA data_version` as seen by this connection.
    ///
    /// The value increments whenever the database file is modified by *another*
    /// connection — including a separate process such as `travsr fsck --fix` or
    /// a manual `sqlite3` edit. It does NOT change for writes made through this
    /// connection itself, so callers holding a read-only connection observe
    /// every out-of-band mutation (issue #464: the daemon's query cache keys on
    /// this so direct `graph.db` writers structurally invalidate cached results).
    pub fn data_version(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .context("querying data_version")
            .map(|v| v as u64)
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return `PRAGMA data_version` of the sibling `embed.db`, or `Ok(None)`
    /// when no embed.db exists (FTS-only mode — there is no embed state to
    /// version).
    ///
    /// `ask` results depend on embed.db (KNN + RFC-019 cosine oracle), which
    /// the embed sidecar rewrites without ever touching graph.db — so
    /// graph.db's [`Self::data_version`] alone cannot invalidate cached `ask`
    /// answers after a `travsr embed reindex`. The daemon's query cache keys
    /// on this value alongside the graph one (#464 follow-up).
    ///
    /// The pragma is read on a persistent, lazily-opened read-only connection:
    /// `data_version` only moves relative to prior reads on the SAME
    /// connection. If embed.db disappears after the connection was opened, the
    /// connection is dropped and `Ok(None)` is returned.
    ///
    /// # What the returned number is (#509)
    ///
    /// It is a freshness token, not the raw pragma. Callers must compare it for
    /// equality only: it carries no ordering, and a larger value does not mean
    /// newer. The token mixes the pragma with the [`file_identity`] of the file
    /// the connection is open on, because the pragma alone cannot see a
    /// delete-and-recreate at the same path, in two separate ways:
    ///
    /// 1. The old code kept the cached connection whenever `path.exists()` was
    ///    true. Deleting embed.db and letting the sidecar rebuild it (a natural
    ///    user recovery from a bad embedding state) leaves the path present at
    ///    every poll, so the connection stayed pinned to the unlinked inode and
    ///    its `data_version` was frozen for the daemon's lifetime. Identity, not
    ///    existence, is what decides whether to reopen.
    /// 2. Reopening alone is still not enough. `data_version` counts writes
    ///    observed by one connection, so a fresh connection to the rebuilt file
    ///    begins its own count from a low value rather than continuing the old
    ///    one, and the two readings routinely collide: the regression test's
    ///    swap reads 2 on both sides. Mixing the identity in makes the token
    ///    move across the swap even when both pragmas read the same.
    ///
    /// So the token changes when embed.db's contents change *or* when the path
    /// comes to name a different file, which is exactly the condition under
    /// which a cached `ask` answer stops being valid.
    pub fn embed_data_version(&self) -> Result<Option<u64>, StoreError> {
        let Some(path) = self.embed_db_path.as_deref() else {
            return Ok(None); // in-memory store — no embed sidecar
        };
        let mut slot = self.embed_meta_conn.borrow_mut();
        let Some(identity) = file_identity(path) else {
            *slot = None;
            return Ok(None);
        };
        // #509: the path resolving to a different file than the one the cached
        // connection was opened on means that connection is reading a corpse.
        if slot
            .as_ref()
            .is_some_and(|(_, opened_on)| *opened_on != identity)
        {
            *slot = None;
        }
        if slot.is_none() {
            let conn = Connection::open_with_flags(
                path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .with_context(|| format!("opening embed.db read-only at {}", path.display()))
            .map_err(|e| StoreError::Database(e.to_string()))?;
            *slot = Some((conn, identity));
        }
        let version = slot
            .as_ref()
            .expect("embed_meta_conn initialized above")
            .0
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .context("querying embed.db data_version")
            .map_err(|e| StoreError::Database(e.to_string()))? as u64;
        Ok(Some(embed_version_token(identity, version)))
    }

    /// Return the live journal mode reported by SQLite. Useful in tests.
    pub fn journal_mode(&self) -> Result<String, StoreError> {
        self.conn
            .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
            .context("querying journal_mode")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn node_count(&self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let n: i64 = self
                .conn
                .query_row("SELECT count(*) FROM nodes", [], |row| row.get(0))
                .context("counting nodes")?;
            Ok(n as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn edge_count(&self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let n: i64 = self
                .conn
                .query_row("SELECT count(*) FROM edges", [], |row| row.get(0))
                .context("counting edges")?;
            Ok(n as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// L11: row count in the FTS index. Should match `node_count()`.
    /// A mismatch indicates partial FTS write — user should re-run `travsr init`.
    pub fn fts_node_count(&self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let n: i64 = self
                .conn
                .query_row("SELECT count(*) FROM nodes_fts", [], |row| row.get(0))
                .context("counting nodes_fts")?;
            Ok(n as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #478: row count in `nodes_fts_words_map` (Leg B retraction memory).
    /// Should match `node_count()`. A mismatch indicates a partial write from
    /// before the v21 migration's backfill completed, or a bug in one of the
    /// write paths — `fsck` reports this, report-only (see RFC-023 §8).
    pub fn fts_words_node_count(&self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let n: i64 = self
                .conn
                .query_row("SELECT count(*) FROM nodes_fts_words_map", [], |row| {
                    row.get(0)
                })
                .context("counting nodes_fts_words_map")?;
            Ok(n as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #478: the `nodes.is_noise` flag computed by `travsr_core::noise::is_structural_noise`
    /// at write time. `None` when `id` is absent. Test/diagnostic accessor —
    /// production filtering reads the column directly in SQL (WS-5).
    pub fn is_noise_flag(&self, id: NodeId) -> Result<Option<bool>, StoreError> {
        (|| -> AnyResult<Option<bool>> {
            let v: Option<i64> = self
                .conn
                .query_row(
                    "SELECT is_noise FROM nodes WHERE id = ?1",
                    params![node_id_to_i64(id)],
                    |row| row.get(0),
                )
                .optional()
                .context("reading is_noise flag")?;
            Ok(v.map(|n| n != 0))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #479: the `nodes.test_role` classification computed by `travsr-analysis`
    /// at index time (AST captures), read back by the mcp `Tests`-bucketer. `None`
    /// (the [`Option`], i.e. absent id) is distinct from `TestRole::None` (the
    /// row exists but is not test code). Mirrors [`Self::is_noise_flag`]: the
    /// bucketer reads it by id rather than relying on `Node.test_role`, which the
    /// generic read paths default to `None` (only the write path carries it).
    pub fn test_role(&self, id: NodeId) -> Result<Option<TestRole>, StoreError> {
        (|| -> AnyResult<Option<TestRole>> {
            let v: Option<i64> = self
                .conn
                .query_row(
                    "SELECT test_role FROM nodes WHERE id = ?1",
                    params![node_id_to_i64(id)],
                    |row| row.get(0),
                )
                .optional()
                .context("reading test_role")?;
            Ok(v.map(TestRole::from_i64))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #478: word-segmented `(sig_words, path_words)` currently indexed for `id`
    /// (`nodes_fts_words_map`). `None` when absent. Test/diagnostic accessor.
    pub fn fts_words_entry(&self, id: NodeId) -> Result<Option<(String, String)>, StoreError> {
        self.conn
            .query_row(
                "SELECT sig_words, path_words FROM nodes_fts_words_map WHERE node_id = ?1",
                params![node_id_to_i64(id)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("reading fts_words_entry")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Embedding coverage counts for a given model, split by k-core shell threshold.
    ///
    /// Returns `(total_symbols, embedded, phase1_total, phase1_done)` where
    /// "embeddable" matches the sidecar's kind filter exactly:
    ///   NOT IN ('file', 'file-module', 'import', 'module', 'field', 'variable')
    /// phase1 = embeddable nodes with `shell_number >= threshold` (high-centrality core).
    /// Phase2 totals can be derived: `phase2_total = total_symbols - phase1_total`.
    ///
    /// Every count is over embeddable nodes, `embedded` included (#862), so
    /// `embedded <= total_symbols` and `phase1_done <= phase1_total` hold for
    /// one model and one graph snapshot, and `total_symbols - embedded` is the
    /// same pending set the sidecar computes for itself.
    ///
    /// RFC-019: embedded counts are read from embed.db (sibling of graph.db) via
    /// ATTACH. Returns (total, 0, phase1_total, 0) when embed.db does not yet
    /// exist (first run before any reindex completes).
    pub fn embed_progress(
        &self,
        model_id: &str,
        shell_threshold: u32,
    ) -> Result<(u64, u64, u64, u64), StoreError> {
        // #391: must match travsr-embed sidecar's NODE_ELIGIBLE predicate exactly —
        // a node is embeddable if it's a normal symbol kind OR the daemon opted it
        // in via embed_text (admits data-format file nodes: yaml/toml/json/xml).
        //
        // #862: all four counts apply this predicate, `embedded` included. A
        // vector's presence is not evidence that its node is embeddable *now*:
        // eligibility is a property of the node's current row, and it moves.
        // `clear_all_embed_texts` on a model switch drops every `embed_text`
        // and the regeneration at the new tier does not necessarily restore it,
        // so a `field` node embedded while it had text becomes ineligible while
        // its vector stays. Counting that vector made `embedded` exceed
        // `total_symbols` (10,009 / 9,996 in the report) and, through the
        // daemon's `phase2_remaining` arithmetic, masked exactly that many
        // genuinely pending nodes. The vectors themselves are left alone: they
        // are valid data for the sidecar's own use, they are simply not
        // progress against the denominator they were being compared to.
        //
        // #376 W1: the trailing clause mirrors the sidecar's exclusion of
        // doc-chunks with no prose. Without it those nodes count as pending
        // forever, so the daemon's tick would re-spawn a sidecar on every pass
        // that can never make progress on them.
        //
        // Two forms: bare columns for `FROM nodes`, and `n.`-qualified for the JOIN.
        const KIND_FILTER: &str = "((kind NOT IN \
             ('file', 'file-module', 'import', 'module', 'field', 'variable') \
             OR embed_text IS NOT NULL) \
             AND NOT (kind = 'doc-chunk' AND embed_text IS NULL))";
        const KIND_FILTER_N: &str = "((n.kind NOT IN \
             ('file', 'file-module', 'import', 'module', 'field', 'variable') \
             OR n.embed_text IS NOT NULL) \
             AND NOT (n.kind = 'doc-chunk' AND n.embed_text IS NULL))";

        (|| -> AnyResult<(u64, u64, u64, u64)> {
            let total_symbols: i64 = self
                .conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM nodes WHERE {KIND_FILTER}"),
                    [],
                    |r| r.get(0),
                )
                .context("counting embeddable nodes")?;

            let phase1_total: i64 = self
                .conn
                .query_row(
                    &format!(
                        "SELECT COUNT(*) FROM nodes WHERE {KIND_FILTER} AND shell_number >= ?1"
                    ),
                    rusqlite::params![shell_threshold],
                    |r| r.get(0),
                )
                .context("counting phase1 embeddable")?;

            // RFC-019: node_embeddings lives in embed.db. ATTACH it to count
            // embedded nodes. Return zeros when embed.db does not exist yet
            // (before first sidecar reindex pass).
            //
            // SC-H3: EdbGuard ensures DETACH runs even on query failure, preventing
            // "database edb already in use" from bricking the connection on next call.
            struct EdbGuard<'g>(&'g rusqlite::Connection);
            impl Drop for EdbGuard<'_> {
                fn drop(&mut self) {
                    let _ = self.0.execute_batch("DETACH DATABASE edb");
                }
            }

            let (embedded, phase1_done) = self
                .embed_db_path
                .as_deref()
                .filter(|p| p.exists())
                .map(|embed_path| -> AnyResult<(i64, i64)> {
                    let embed_path_str = embed_path
                        .to_str()
                        .context("embed.db path is not valid UTF-8")?;
                    self.conn
                        .execute_batch(&format!("ATTACH DATABASE '{embed_path_str}' AS edb"))
                        .context("attaching embed.db")?;
                    let _guard = EdbGuard(&self.conn); // DETACH on any early return
                                                       // #862: same join and predicate as `phase1_done` below, minus
                                                       // the shell clause. The join also drops an orphan row whose
                                                       // node is gone: embed.db is a separate file, so the schema's
                                                       // `ON DELETE CASCADE` cannot reach across to it.
                    let embedded: i64 = self
                        .conn
                        .query_row(
                            &format!(
                                "SELECT COUNT(*) FROM edb.node_embeddings ne \
                                 JOIN nodes n ON ne.node_id = n.id \
                                 WHERE ne.model_id = ?1 AND {KIND_FILTER_N}"
                            ),
                            rusqlite::params![model_id],
                            |r| r.get(0),
                        )
                        .context("counting embedded nodes")?;
                    let phase1_done: i64 = self
                        .conn
                        .query_row(
                            &format!(
                                "SELECT COUNT(*) FROM edb.node_embeddings ne \
                                 JOIN nodes n ON ne.node_id = n.id \
                                 WHERE ne.model_id = ?1 AND {KIND_FILTER_N} \
                                 AND n.shell_number >= ?2"
                            ),
                            rusqlite::params![model_id, shell_threshold],
                            |r| r.get(0),
                        )
                        .context("counting phase1 embedded")?;
                    Ok((embedded, phase1_done))
                    // _guard drops here → DETACH DATABASE edb
                })
                .transpose()?
                .unwrap_or((0, 0));

            Ok((
                total_symbols as u64,
                embedded as u64,
                phase1_total as u64,
                phase1_done as u64,
            ))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return per-language node counts, ordered by count descending.
    /// Returns an empty Vec when no nodes carry language metadata.
    pub fn language_distribution(&self) -> Result<Vec<(String, u64)>, StoreError> {
        (|| -> AnyResult<Vec<(String, u64)>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT language, COUNT(*) as cnt \
                     FROM nodes \
                     WHERE language IS NOT NULL AND language != '' \
                     GROUP BY language \
                     ORDER BY cnt DESC",
                )
                .context("preparing language_distribution")?;
            let pairs = stmt
                .query_map([], |row| {
                    let lang: String = row.get(0)?;
                    let cnt: i64 = row.get(1)?;
                    Ok((lang, cnt as u64))
                })
                .context("querying language distribution")?
                .collect::<rusqlite::Result<Vec<_>>>()
                .context("collecting language distribution")?;
            Ok(pairs)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Returns `true` when at least one `ref/call` edge exists whose source node
    /// has the given `language`. Used by `get_lang_status` to detect whether
    /// Phase B (SCIP/LSIF) has run for a language without a separate metadata
    /// table. Returns `false` on any query error (safe default: show Tree-sitter).
    pub fn has_refcall_edges_for_language(&self, language: &str) -> bool {
        let result: rusqlite::Result<bool> = self.conn.query_row(
            "SELECT EXISTS( \
                SELECT 1 FROM edges e \
                INNER JOIN nodes n ON e.src = n.id \
                WHERE e.kind = 'ref/call' AND n.language = ?1 \
                LIMIT 1 \
             )",
            params![language],
            |row| row.get::<_, bool>(0),
        );
        match result {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("has_refcall_edges_for_language error: {e}");
                false
            }
        }
    }

    /// Load every row from the `files` table into a path → sha256 map.
    ///
    /// Used by `init_repo`'s parallel indexing pipeline to pre-populate an
    /// in-memory skip set so worker threads can compare hashes without hitting
    /// SQLite on every file.
    pub fn get_all_file_hashes(
        &self,
    ) -> Result<std::collections::HashMap<String, String>, StoreError> {
        (|| -> AnyResult<std::collections::HashMap<String, String>> {
            let mut stmt = self
                .conn
                .prepare("SELECT path, sha256 FROM files")
                .context("preparing get_all_file_hashes")?;
            let mapped = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .context("executing get_all_file_hashes")?;
            let owned: rusqlite::Result<std::collections::HashMap<String, String>> =
                mapped.collect();
            owned.context("collecting file hashes")
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn get_file_hash(&self, path: &str) -> Result<Option<String>, StoreError> {
        self.conn
            .query_row(
                "SELECT sha256 FROM files WHERE path = ?1",
                params![path],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("reading file hash")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn put_file_hash(&mut self, path: &str, hex: &str) -> Result<(), StoreError> {
        self.conn
            .execute(
                "INSERT INTO files(path, sha256, last_indexed_at) \
                 VALUES(?1, ?2, unixepoch()) \
                 ON CONFLICT(path) DO UPDATE SET \
                   sha256 = excluded.sha256, \
                   last_indexed_at = excluded.last_indexed_at",
                params![path, hex],
            )
            .context("writing file hash")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    /// Clear every stored per-file content hash (the `files` table).
    ///
    /// #757 audit: `travsr init --force` relies on the incremental hash-delta
    /// re-parsing every file, but the delta skips files whose content hash is
    /// unchanged — and the `--force` graph purge does not touch this cache. An
    /// index built by an older binary therefore reported "up to date" and was
    /// never re-parsed by the current analyzer (so, e.g., newly-added `field`
    /// nodes never appeared). Clearing the cache makes every file look new, so
    /// `--force` genuinely re-parses the whole repo. Returns rows cleared.
    pub fn clear_file_hashes(&mut self) -> Result<usize, StoreError> {
        self.conn
            .execute("DELETE FROM files", [])
            .context("clearing file hashes")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Write a batch of parsed file graphs in a single SQLite transaction.
    ///
    /// For each `FileGraph` in `batch`:
    /// 1. Retract old FTS rows + decrement vocab refcounts for the path.
    /// 2. Delete old edges, their occurrence rows, and nodes for the (corpus, path).
    /// 3. Insert new nodes.
    /// 4. Upsert the file hash.
    ///
    /// Edges are inserted once, after every file in the batch has written its
    /// nodes, so the endpoint-existence guard cannot drop a cross-file edge
    /// whose target file happens to be written later in the same batch.
    ///
    /// When `bulk=true` (init path), step 3 writes only `nodes_fts_map` rows
    /// instead of the full FTS5 + vocab update per node.  Call
    /// [`rebuild_fts_from_map`] after all batches to build the index in one pass.
    ///
    /// When `bulk=false` (incremental / reindex path), each node gets the full
    /// `put_node_fts` treatment — FTS5 insert + vocab increment — exactly as before.
    ///
    /// All files in the batch commit atomically — a failure rolls back the
    /// entire batch, leaving the DB consistent.
    pub fn write_file_graphs_batch(
        &mut self,
        batch: &[FileGraph],
        bulk: bool,
    ) -> Result<BatchWriteCounts, StoreError> {
        let staging = self.staging_active;
        (|| -> AnyResult<BatchWriteCounts> {
            // _bulk_fts_pending is a temp table populated by put_node_fts_map_only
            // and consumed by rebuild_fts_from_map. Create it here so the table
            // exists for the duration of this call even when begin_bulk_fts_tracking
            // was not called explicitly by the caller.
            if bulk {
                self.conn
                    .execute_batch(
                        "CREATE TEMP TABLE IF NOT EXISTS _bulk_fts_pending \
                         (node_id INTEGER PRIMARY KEY);",
                    )
                    .context("creating _bulk_fts_pending temp table")?;
            }
            let tx = self
                .conn
                .transaction()
                .context("starting batch write transaction")?;
            let mut counts = BatchWriteCounts::default();
            // Incremental-path edges are held until every file in the batch has
            // written its nodes. The endpoint guard below drops an edge whose
            // `dst` node does not exist yet, and a cross-file edge names a node
            // another file in this same batch owns (a TypeScript
            // `import:./billing --resolves-to--> file:src/billing.ts`, whose
            // target `file:` node only `src/billing.ts` writes). Inserting per
            // file made that depend on the order the parallel parse workers
            // happened to deliver the two files in. The staging path is already
            // immune, since it promotes all nodes before filtering edges.
            let mut pending_edges: Vec<&travsr_core::Edge> = Vec::new();

            for file in batch {
                // RFC-027 #813: body hash per definition, from this file's own
                // source, stamped alongside the edges the same parse produced.
                // Empty when the caller supplied no source (hashes stay NULL).
                let body_hashes: HashMap<i64, String> = match &file.source {
                    Some(text) => body_hashes_for_spans(
                        text,
                        file.nodes
                            .iter()
                            .filter_map(|n| n.line.map(|l| (node_id_to_i64(n.id), l, n.end_line))),
                    ),
                    None => HashMap::new(),
                };
                if staging {
                    // ── staging path (bulk init only) ─────────────────────────
                    // No delete pass — the DB is empty on a fresh init.
                    // Plain INSERT into constraint-free TEMP tables: no B-tree
                    // read, no index maintenance. flush_staging_to_production
                    // deduplicates in one GROUP BY pass after all files are done.
                    for node in &file.nodes {
                        tx.execute(
                            "INSERT INTO nodes_stage(id,corpus,root,path,language,\
                             signature,kind,package,line,end_line,is_noise,test_role,body_hash) \
                             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
                            params![
                                node_id_to_i64(node.id),
                                node.vname.corpus,
                                node.vname.root,
                                node.vname.path,
                                node.vname.language,
                                node.vname.signature,
                                node.kind,
                                node.package,
                                node.line.map(|l| l as i64),
                                node.end_line.map(|l| l as i64),
                                travsr_core::noise::is_structural_noise(node),
                                node.test_role.as_i64(),
                                body_hashes.get(&node_id_to_i64(node.id)).map(String::as_str),
                            ],
                        )
                        .context("staging: inserting node")?;
                        Self::put_node_fts_map_only(&tx, node)
                            .context("staging: put_node_fts_map_only")?;
                        Self::put_node_fts_words_map_only(&tx, node)
                            .context("staging: put_node_fts_words_map_only")?;
                        counts.nodes_upserted += 1;
                    }
                    for edge in &file.edges {
                        tx.execute(
                            "INSERT INTO edges_stage(src,dst,kind,provenance,confidence) \
                             VALUES(?1,?2,?3,'tree-sitter',?4)",
                            params![
                                node_id_to_i64(edge.src),
                                node_id_to_i64(edge.dst),
                                edge.kind.as_str(),
                                edge.confidence.map(|c| c as i64),
                            ],
                        )
                        .context("staging: inserting edge")?;
                        counts.edges_upserted += 1;
                    }
                } else {
                    // ── incremental path: delete existing rows, then upsert ───
                    // Every delete below is scoped by (corpus, path), the key
                    // [`Self::reindex_replace`] uses: a path is unique within a
                    // corpus, not across them (ARCH-102: one corpus per repo,
                    // derived from its git remote, plus the external-package
                    // corpora). This API takes no `corpus` argument, so it
                    // comes from the file's own nodes; a parse
                    // that produced none leaves it NULL, which matches the whole
                    // path exactly as these statements did before.
                    let corpus: Option<&str> =
                        file.nodes.first().map(|n| n.vname.corpus.as_str());
                    // Load old FTS tokens BEFORE removing map rows so vocab can
                    // be decremented correctly.
                    let old_token_strings: Vec<String> = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT m.tokens FROM nodes_fts_map m \
                                 JOIN nodes n ON n.id = m.node_id \
                                 WHERE n.path = ?1 AND (?2 IS NULL OR n.corpus = ?2)",
                            )
                            .context("preparing old-token load")?;
                        let mapped = stmt
                            .query_map(params![file.vname_path, corpus], |row| {
                                row.get::<_, String>(0)
                            })
                            .context("querying old tokens")?;
                        let owned: rusqlite::Result<Vec<String>> = mapped.collect();
                        owned.context("collecting old tokens")?
                    };
                    tx.execute(
                        "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) \
                         SELECT 'delete', m.node_id, m.tokens \
                         FROM nodes_fts_map m JOIN nodes n ON n.id = m.node_id \
                         WHERE n.path = ?1 AND (?2 IS NULL OR n.corpus = ?2)",
                        params![file.vname_path, corpus],
                    )
                    .context("retracting FTS rows")?;
                    tx.execute(
                        "DELETE FROM nodes_fts_map \
                         WHERE node_id IN (SELECT id FROM nodes \
                           WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2))",
                        params![file.vname_path, corpus],
                    )
                    .context("removing FTS map rows")?;
                    for ts in &old_token_strings {
                        Self::vocab_decrement(&tx, ts)?;
                    }
                    // #478: retract nodes_fts_words the same way (own retraction
                    // memory, no vocab decrement needed — fts5vocab has zero drift).
                    tx.execute(
                        "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                         SELECT 'delete', m.node_id, m.sig_words, m.path_words \
                         FROM nodes_fts_words_map m JOIN nodes n ON n.id = m.node_id \
                         WHERE n.path = ?1 AND (?2 IS NULL OR n.corpus = ?2)",
                        params![file.vname_path, corpus],
                    )
                    .context("retracting FTS word rows")?;
                    tx.execute(
                        "DELETE FROM nodes_fts_words_map \
                         WHERE node_id IN (SELECT id FROM nodes \
                           WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2))",
                        params![file.vname_path, corpus],
                    )
                    .context("removing FTS word map rows")?;
                    // Owned-edge-only delete, the same ownership rule
                    // [`Self::reindex_replace`] states: this file owns its
                    // outbound edges, and inbound edges belong to whoever wrote
                    // them. The blunt `OR dst IN (…)` this replaced also deleted
                    // inbound edges to symbols that *survive* the re-parse, and
                    // nothing re-derives those, because the file that owns them
                    // is not being re-parsed here. A relative TypeScript import is
                    // exactly that shape (`import:./billing --resolves-to-->
                    // file:src/billing.ts`: `src` in the importer, `dst` in the
                    // target), so re-indexing the target erased the importer's
                    // edge. Whether it did depended on the order the parallel
                    // parse workers happened to deliver the two files in, which
                    // is why `travsr init --force` dropped a random subset of
                    // `resolves-to` edges with no error while a fresh index (the
                    // staging path, which has no delete pass) stayed stable.
                    let removed_ids: Vec<i64> = {
                        let new_ids: std::collections::HashSet<i64> =
                            file.nodes.iter().map(|n| node_id_to_i64(n.id)).collect();
                        let mut stmt = tx
                            .prepare(
                                "SELECT id FROM nodes \
                                 WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2)",
                            )
                            .context("preparing old-id snapshot")?;
                        let rows = stmt
                            .query_map(params![file.vname_path, corpus], |r| r.get::<_, i64>(0))
                            .context("querying old ids")?
                            .collect::<rusqlite::Result<Vec<i64>>>()
                            .context("collecting old ids")?;
                        rows.into_iter()
                            .filter(|id| !new_ids.contains(id))
                            .collect()
                    };
                    tx.execute(
                        "DELETE FROM edges WHERE src IN (SELECT id FROM nodes \
                           WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2))",
                        params![file.vname_path, corpus],
                    )
                    .context("deleting owned edges for path")?;
                    // The occurrence rows follow the edges they describe.
                    // `edge_sites` has no FK cascade (#299 F7), so a row used to
                    // survive the delete above and then find no `edges` row in
                    // `reference_sites`'s LEFT JOIN, which reads a missing join as
                    // `heuristic = false` and so serves a stale line as resolved
                    // fact. Owned rows only, the same ownership rule the edge
                    // delete above uses and the one [`Self::reindex_replace`]
                    // states: an occurrence's `src` is the enclosing node in this
                    // same file, and inbound sites belong to the files that wrote
                    // them. An inbound site can only have gone stale if the file
                    // holding it changed, and that file's own re-parse purges it
                    // here. Deleting them from this side would also be a full
                    // scan per id: `edge_sites` is WITHOUT ROWID on
                    // (src, dst, kind, line), so unlike `edges` it has no index
                    // a `dst =` probe can seek on.
                    tx.execute(
                        "DELETE FROM edge_sites WHERE src IN (SELECT id FROM nodes \
                           WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2))",
                        params![file.vname_path, corpus],
                    )
                    .context("deleting owned edge_sites for path")?;
                    // A symbol the re-parse dropped still takes its inbound
                    // edges with it, so the narrower delete above never trades a
                    // lost edge for a dangling one.
                    // `prepare_cached` once, not `tx.execute` per id: on a
                    // `--force` re-parse `removed_ids` is the whole repo's node
                    // set, and an uncached execute re-prepares the statement for
                    // every one of them.
                    if !removed_ids.is_empty() {
                        let mut del_inbound = tx
                            .prepare_cached("DELETE FROM edges WHERE dst = ?1")
                            .context("preparing inbound-edge delete")?;
                        for removed_id in &removed_ids {
                            del_inbound
                                .execute(params![removed_id])
                                .context("deleting inbound edges for a removed symbol")?;
                        }
                    }
                    tx.execute(
                        "DELETE FROM nodes \
                         WHERE path = ?1 AND (?2 IS NULL OR corpus = ?2)",
                        params![file.vname_path, corpus],
                    )
                    .context("deleting nodes for path")?;

                    for node in &file.nodes {
                        let id_i64 = node_id_to_i64(node.id);
                        tx.execute(
                            "INSERT INTO nodes(id,corpus,root,path,language,signature,kind,package,line,end_line,is_noise,test_role,body_hash) \
                             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13) \
                             ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, \
                               package = excluded.package, \
                               line = COALESCE(excluded.line, nodes.line), \
                               end_line = COALESCE(excluded.end_line, nodes.end_line), \
                               is_noise = excluded.is_noise, \
                               test_role = excluded.test_role, \
                               body_hash = excluded.body_hash",
                            params![
                                id_i64,
                                node.vname.corpus,
                                node.vname.root,
                                node.vname.path,
                                node.vname.language,
                                node.vname.signature,
                                node.kind,
                                node.package,
                                node.line.map(|l| l as i64),
                                node.end_line.map(|l| l as i64),
                                travsr_core::noise::is_structural_noise(node),
                                node.test_role.as_i64(),
                                body_hashes.get(&id_i64).map(String::as_str),
                            ],
                        )
                        .context("inserting node in batch")?;
                        if bulk {
                            Self::put_node_fts_map_only(&tx, node)
                                .context("batch put_node_fts_map_only")?;
                            Self::put_node_fts_words_map_only(&tx, node)
                                .context("batch put_node_fts_words_map_only")?;
                        } else {
                            Self::put_node_fts(&tx, node).context("batch put_node_fts")?;
                            Self::put_node_fts_words(&tx, node)
                                .context("batch put_node_fts_words")?;
                        }
                        counts.nodes_upserted += 1;
                    }
                    pending_edges.extend(&file.edges);
                }

                // File hash goes to production in both paths — one row per
                // source file, non-conflicting, needed for SHA256 delta detection.
                tx.execute(
                    "INSERT INTO files(path, sha256, last_indexed_at) \
                     VALUES(?1, ?2, unixepoch()) \
                     ON CONFLICT(path) DO UPDATE SET \
                       sha256 = excluded.sha256, \
                       last_indexed_at = excluded.last_indexed_at",
                    params![file.vname_path, file.new_hash],
                )
                .context("writing file hash in batch")?;
                counts.files_written += 1;
            }

            // UX-9: every batch node now exists, and every other file's nodes
            // already live in `nodes` (the incremental path runs against a
            // populated store), so guard both endpoints here: a parser edge to
            // an un-emitted node (e.g. Ruby `class ::Hash` reopening, or a
            // speculative import candidate whose file does not exist) must not
            // persist a dangling half-edge. `execute` returns 0 when the guard
            // rejects it, keeping `edges_upserted` honest.
            if !pending_edges.is_empty() {
                let mut ins_edge = tx
                    .prepare_cached(
                        "INSERT INTO edges(src,dst,kind,provenance,confidence) \
                         SELECT ?1,?2,?3,'tree-sitter',?4 \
                         WHERE EXISTS(SELECT 1 FROM nodes n WHERE n.id = ?1) \
                           AND EXISTS(SELECT 1 FROM nodes n WHERE n.id = ?2) \
                         ON CONFLICT(src,dst,kind) DO NOTHING",
                    )
                    .context("preparing batch edge insert")?;
                for edge in pending_edges {
                    let inserted = ins_edge
                        .execute(params![
                            node_id_to_i64(edge.src),
                            node_id_to_i64(edge.dst),
                            edge.kind.as_str(),
                            edge.confidence.map(|c| c as i64),
                        ])
                        .context("inserting edge in batch")?;
                    counts.edges_upserted += inserted as u64;
                }
            }

            tx.commit().context("committing batch write transaction")?;
            Ok(counts)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Delete all nodes (and their edges) whose VName path equals `path`.
    /// Edges are deleted first to avoid orphan references; both operations
    /// run inside a single transaction.  Returns the number of nodes removed.
    pub fn delete_nodes_for_path(&mut self, path: &str) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let tx = self
                .conn
                .transaction()
                .context("starting delete transaction")?;

            let count: i64 = tx
                .query_row(
                    "SELECT count(*) FROM nodes WHERE path = ?1",
                    params![path],
                    |row| row.get(0),
                )
                .context("counting nodes to delete")?;

            // Edges have no FK cascade — delete them explicitly before removing nodes.
            tx.execute(
                "DELETE FROM edges \
                 WHERE src IN (SELECT id FROM nodes WHERE path = ?1) \
                    OR dst IN (SELECT id FROM nodes WHERE path = ?1)",
                params![path],
            )
            .context("deleting edges for path")?;

            // Load token strings for vocab decrement BEFORE removing map rows.
            let path_token_strings: Vec<String> = {
                let mut stmt = tx
                    .prepare(
                        "SELECT m.tokens FROM nodes_fts_map m \
                         JOIN nodes n ON n.id = m.node_id WHERE n.path = ?1",
                    )
                    .context("preparing token load for path delete")?;
                let collected: Vec<String> = stmt
                    .query_map(params![path], |row| row.get(0))
                    .context("executing token load for path delete")?
                    .collect::<Result<_, _>>()
                    .context("collecting token strings for vocab decrement")?;
                collected
            };

            // Retract FTS entries before deleting nodes (tokens stored in map).
            tx.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) \
                 SELECT 'delete', m.node_id, m.tokens \
                 FROM nodes_fts_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.path = ?1",
                params![path],
            )
            .context("retracting FTS rows for path")?;
            tx.execute(
                "DELETE FROM nodes_fts_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE path = ?1)",
                params![path],
            )
            .context("removing nodes_fts_map rows for path")?;

            // Decrement fts_vocab refcounts for all retracted tokens (v10 L2-A).
            for ts in &path_token_strings {
                Self::vocab_decrement(&tx, ts)?;
            }

            // #478: retract nodes_fts_words the same way (no vocab decrement needed).
            tx.execute(
                "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                 SELECT 'delete', m.node_id, m.sig_words, m.path_words \
                 FROM nodes_fts_words_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.path = ?1",
                params![path],
            )
            .context("retracting FTS word rows for path")?;
            tx.execute(
                "DELETE FROM nodes_fts_words_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE path = ?1)",
                params![path],
            )
            .context("removing nodes_fts_words_map rows for path")?;

            tx.execute("DELETE FROM nodes WHERE path = ?1", params![path])
                .context("deleting nodes for path")?;

            tx.commit().context("committing delete transaction")?;
            Ok(count as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// True file-deletion GC: removes all nodes, both-direction edges, FTS rows,
    /// and the `files` hash row for `path`. Returns the set of caller file paths
    /// that had inbound edges to the deleted nodes so the daemon can enqueue them
    /// for Tier-0 re-resolution.
    ///
    /// Corpus-scoped so every lookup hits the v13 `(corpus, path)` index.
    /// All operations run in one transaction (atomicity + WAL crash safety).
    pub fn delete_file(&mut self, corpus: &str, path: &str) -> Result<DirtySet, StoreError> {
        (|| -> AnyResult<DirtySet> {
            let tx = self
                .conn
                .transaction()
                .context("starting delete_file transaction")?;

            // Collect callers BEFORE any delete.
            let mut callers = DirtySet::default();
            {
                let mut stmt = tx
                    .prepare(
                        "SELECT DISTINCT n.path \
                         FROM edges e JOIN nodes n ON n.id = e.src \
                         WHERE e.dst IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2) \
                           AND n.path != ?2",
                    )
                    .context("preparing caller query for delete_file")?;
                for row in stmt
                    .query_map(params![corpus, path], |r| r.get::<_, String>(0))
                    .context("executing caller query")?
                {
                    callers.insert(row.context("reading caller path")?);
                }
            }

            // Load token strings for vocab decrement BEFORE removing map rows.
            let token_strings: Vec<String> = {
                let mut stmt = tx
                    .prepare(
                        "SELECT m.tokens FROM nodes_fts_map m \
                         JOIN nodes n ON n.id = m.node_id \
                         WHERE n.corpus=?1 AND n.path=?2",
                    )
                    .context("preparing token load for delete_file")?;
                let rows: Vec<String> = stmt
                    .query_map(params![corpus, path], |r| r.get::<_, String>(0))
                    .context("executing token load")?
                    .collect::<rusqlite::Result<_>>()
                    .context("collecting token strings")?;
                rows
            };

            // Delete ALL edges (both directions) — every symbol in this path is dead.
            tx.execute(
                "DELETE FROM edges \
                 WHERE src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2) \
                    OR dst IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("deleting both-direction edges for delete_file")?;

            // #299 F7: purge the matching occurrence rows. `edge_sites` has no FK
            // cascade (a cascade would over-delete inbound sites on unrelated
            // re-indexes — see reindex_replace), so mirror the both-direction edge
            // delete here: every occurrence into or out of this dead file is gone.
            tx.execute(
                "DELETE FROM edge_sites \
                 WHERE src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2) \
                    OR dst IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("deleting both-direction edge_sites for delete_file")?;

            // Retract FTS entries before deleting nodes.
            tx.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) \
                 SELECT 'delete', m.node_id, m.tokens \
                 FROM nodes_fts_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.corpus=?1 AND n.path=?2",
                params![corpus, path],
            )
            .context("retracting FTS rows for delete_file")?;
            tx.execute(
                "DELETE FROM nodes_fts_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("removing nodes_fts_map rows for delete_file")?;
            for ts in &token_strings {
                Self::vocab_decrement(&tx, ts)?;
            }

            // #478: retract nodes_fts_words the same way (no vocab decrement needed).
            tx.execute(
                "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                 SELECT 'delete', m.node_id, m.sig_words, m.path_words \
                 FROM nodes_fts_words_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.corpus=?1 AND n.path=?2",
                params![corpus, path],
            )
            .context("retracting FTS word rows for delete_file")?;
            tx.execute(
                "DELETE FROM nodes_fts_words_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("removing nodes_fts_words_map rows for delete_file")?;

            // Delete nodes — v17 capture_node_delete trigger auto-writes tombstones.
            tx.execute(
                "DELETE FROM nodes WHERE corpus=?1 AND path=?2",
                params![corpus, path],
            )
            .context("deleting nodes for delete_file")?;

            // Clear the file hash row so Tier-2 reconcile never sees this as a ghost.
            tx.execute("DELETE FROM files WHERE path=?1", params![path])
                .context("removing file hash row for delete_file")?;

            tx.commit().context("committing delete_file transaction")?;
            Ok(callers)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// In-place file update with owned-edge-only semantics (§2 of the GC doc).
    ///
    /// The caller MUST parse first; call this only on parse success so a syntax
    /// error never erases the existing graph. All operations run in one transaction.
    ///
    /// Ownership rule:
    /// - Deletes only `src ∈ nodes(corpus, path)` edges (outbound, owned by this file).
    /// - Inbound edges to surviving symbol IDs are never touched (blank-line/body
    ///   edits are lossless).
    /// - Symbols that vanished have their inbound edges deleted eagerly so PPR/BFS
    ///   never traverses into a missing node.
    ///
    /// RFC-027 #813 (Mechanism A): when `content` is the file's current text and
    /// the edit is a *pure body edit* — the file's set of NodeIds is unchanged,
    /// so no symbol or import was added or removed and the whole file's
    /// resolution context is fixed — the committed owned edges (and their
    /// occurrence rows) of every definition whose body is byte-identical are
    /// **preserved** instead of purged, and this parse's tree-sitter edges for
    /// them are skipped. Only the definitions the edit actually changed lose
    /// their edges and get re-resolved by the live lane. This is what raises
    /// mid-edit recovery from ~71% to ~99% with no language server. Preserved
    /// edges keep their `scip`/`lsif` provenance (they are truth, not overlay),
    /// so the commit sweep and Invariant #4 are unaffected.
    ///
    /// Precision-first: a definition is preserved only when it is *provably*
    /// unchanged (its NodeId survived AND its body hash matches the stored one).
    /// Any doubt — `content` is `None`, a symbol was added or removed, or a body
    /// hash is missing or differs — falls back to the whole-file purge, because a
    /// stale preserved edge is a wrong edge.
    pub fn reindex_replace(
        &mut self,
        corpus: &str,
        path: &str,
        nodes: &[Node],
        edges: &[Edge],
        new_hash: &str,
        content: Option<&str>,
    ) -> Result<ReplaceReport, StoreError> {
        (|| -> AnyResult<ReplaceReport> {
            let tx = self
                .conn
                .transaction()
                .context("starting reindex_replace transaction")?;

            // Snapshot old NodeIds (i64) before any delete.
            let old_ids: std::collections::HashSet<i64> = {
                let mut stmt = tx
                    .prepare("SELECT id FROM nodes WHERE corpus=?1 AND path=?2")
                    .context("preparing old_ids snapshot")?;
                let rows: std::collections::HashSet<i64> = stmt
                    .query_map(params![corpus, path], |r| r.get::<_, i64>(0))
                    .context("executing old_ids snapshot")?
                    .collect::<rusqlite::Result<_>>()
                    .context("collecting old_ids")?;
                rows
            };

            // The NodeIds this parse produced. Computed up front so the
            // pure-body-edit test and the preserved set are known before any
            // delete.
            let new_ids: std::collections::HashSet<i64> =
                nodes.iter().map(|n| node_id_to_i64(n.id)).collect();

            // RFC-027 #813 Mechanism A: the set of definitions whose committed
            // owned edges/sites survive this reparse untouched.
            //
            // Sound only when the edit is a *pure body edit*: the file's NodeId
            // set is unchanged (`old_ids == new_ids`), so no symbol or import was
            // added or removed and the whole file's resolution context is fixed.
            // Under that condition a definition whose body bytes are identical
            // resolves to exactly the same targets, so its committed edges are
            // still correct and re-deriving them is pure waste. Any doubt keeps
            // the set empty and the code below purges the whole file, exactly as
            // before this change.
            let mut preserved: std::collections::HashSet<i64> = match content {
                Some(text) if old_ids == new_ids => {
                    // Body hashes stored for this file's definitions at their last
                    // (re)resolution, and the hashes the current text produces.
                    let stored: HashMap<i64, String> = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT id, body_hash FROM nodes \
                                 WHERE corpus=?1 AND path=?2 AND body_hash IS NOT NULL",
                            )
                            .context("preparing stored body_hash load")?;
                        let rows: HashMap<i64, String> = stmt
                            .query_map(params![corpus, path], |r| {
                                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                            })
                            .context("executing stored body_hash load")?
                            .collect::<rusqlite::Result<_>>()
                            .context("collecting stored body_hash")?;
                        rows
                    };
                    let current = body_hashes_for_spans(
                        text,
                        nodes
                            .iter()
                            .filter_map(|n| n.line.map(|l| (node_id_to_i64(n.id), l, n.end_line))),
                    );
                    stored
                        .iter()
                        .filter(|(id, hash)| current.get(*id) == Some(*hash))
                        .map(|(id, _)| *id)
                        .collect()
                }
                _ => std::collections::HashSet::new(),
            };

            // RFC-027 #813 (finding 1): a byte-identical body proves the
            // definition's own text is unchanged, NOT that the *types* its
            // references resolve through are. NodeIds are name-based
            // (`fn:{name}`, `method:{Type}.{name}`, `field:{Owner}.{f}`), so an
            // edit to a parameter, return, or field type leaves
            // `old_ids == new_ids` intact and changes only the edited
            // definition's hash, yet every same-file definition whose committed
            // receiver-typed method/field edge went through that type now points
            // at the wrong `dst`.
            //
            // Two rules, because a type dependency is only sometimes an edge:
            //
            // 1. A change to a TYPE_LEVEL_KINDS definition invalidates the whole
            //    file. There is no type-reference `EdgeKind` (see `EdgeKind`), so
            //    a `type A = Foo` -> `type A = Bar` edit produces no edge into
            //    `type:A` for rule 2 to follow, and every definition resolving
            //    through `A` would keep a stale committed `dst`.
            // 2. Otherwise demote any preserved definition that references a
            //    changed definition. The reference relationships are the file's
            //    COMMITTED owned edges still in the store (the resolved
            //    `RefCall`/reference edges Phase A tree-sitter never emits — it
            //    emits only `DefinesBinding`/`Depends`/`ResolvesTo`, so this
            //    parse's `edges` alone would demote nothing in production),
            //    unioned with this parse's structural edges. Iterated to a
            //    fixpoint, so demoting B also demotes a preserved A that
            //    references B; each round removes at least one id from a finite
            //    set, so it terminates.
            //
            // Residual, accepted: a container's own declaration (a `class`'s
            // `extends` clause) is not covered, because a container's span holds
            // its members' bodies and treating it as changed would disable
            // preservation on every method-body edit.
            if !preserved.is_empty()
                && nodes.iter().any(|n| {
                    TYPE_LEVEL_KINDS.contains(&n.kind.as_str())
                        && !preserved.contains(&node_id_to_i64(n.id))
                })
            {
                preserved.clear();
            }
            if !preserved.is_empty() {
                // The file's committed owned edges (src ∈ this file), loaded once
                // before the owned-edge purge below. These carry the resolved
                // reference edges that make a caller depend on a changed callee's
                // type; unioned with this parse's structural edges for the
                // fixpoint. Over-demotion (a body change that does not move the
                // callee's resolved type) only costs a re-resolve, never a wrong
                // edge.
                let committed_ref_pairs: Vec<(i64, i64)> = {
                    let mut stmt = tx
                        .prepare(
                            "SELECT src, dst FROM edges \
                             WHERE src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                        )
                        .context("preparing committed owned edge load for demotion")?;
                    let rows: Vec<(i64, i64)> = stmt
                        .query_map(params![corpus, path], |r| {
                            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
                        })
                        .context("executing committed owned edge load")?
                        .collect::<rusqlite::Result<_>>()
                        .context("collecting committed owned edges")?;
                    rows
                };
                let ref_pairs: Vec<(i64, i64)> = committed_ref_pairs
                    .into_iter()
                    .chain(
                        edges
                            .iter()
                            .map(|e| (node_id_to_i64(e.src), node_id_to_i64(e.dst))),
                    )
                    .collect();
                loop {
                    let demoted: Vec<i64> = ref_pairs
                        .iter()
                        .filter_map(|(src, dst)| {
                            (preserved.contains(src)
                                && new_ids.contains(dst)
                                && !preserved.contains(dst))
                            .then_some(*src)
                        })
                        .collect();
                    if demoted.is_empty() {
                        break;
                    }
                    for id in demoted {
                        preserved.remove(&id);
                    }
                }
            }

            // RFC-027 #813 P2: current and committed start line of every
            // definition, so both a preserved definition's occurrences (below)
            // and a changed definition's committed occurrences (captured for the
            // live lane) can be remapped onto the current buffer by their block
            // delta `new_start - old_start`. Only loaded when preservation
            // happened, which is the only case that remaps rather than purges.
            let (new_lines, old_lines): (HashMap<i64, i64>, HashMap<i64, i64>) =
                if preserved.is_empty() {
                    (HashMap::new(), HashMap::new())
                } else {
                    let new_lines: HashMap<i64, i64> = nodes
                        .iter()
                        .filter_map(|n| n.line.map(|l| (node_id_to_i64(n.id), l as i64)))
                        .collect();
                    let old_lines: HashMap<i64, i64> = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT id, line FROM nodes \
                                 WHERE corpus=?1 AND path=?2 AND line IS NOT NULL",
                            )
                            .context("preparing old line load")?;
                        let rows: HashMap<i64, i64> = stmt
                            .query_map(params![corpus, path], |r| {
                                Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
                            })
                            .context("executing old line load")?
                            .collect::<rusqlite::Result<_>>()
                            .context("collecting old lines")?;
                        rows
                    };
                    (new_lines, old_lines)
                };
            // RFC-027 #813 P2 (issue #816 defect 1): current end line of every
            // definition, so a changed definition's occurrences can be re-anchored
            // within its current span. Only needed when preservation happened.
            let new_ends: HashMap<i64, i64> = if preserved.is_empty() {
                HashMap::new()
            } else {
                nodes
                    .iter()
                    .filter_map(|n| {
                        n.end_line.map(|e| (node_id_to_i64(n.id), e as i64))
                    })
                    .collect()
            };
            // Per preserved definition, the line delta from its committed position
            // to its current one. A preserved body is byte-identical, so the whole
            // definition shifted as a block by `new_start - old_start`, and every
            // occurrence inside it shifted by the same amount. This remaps its
            // occurrence rows onto their current lines (below) instead of dropping
            // them, so `find_references` stays correct mid-edit and the remapped
            // line matches what the next commit records (no phantom row, Invariant
            // #4). A definition whose old or new line is unknown is left out and
            // its sites are simply purged.
            let preserved_line_delta: HashMap<i64, i64> = preserved
                .iter()
                .filter_map(|id| {
                    let old = old_lines.get(id)?;
                    let new = new_lines.get(id)?;
                    Some((*id, new - old))
                })
                .collect();
            // Body hashes to stamp on every node this parse writes, so the next
            // save can decide preservation against the text this one committed.
            let new_body_hashes: HashMap<i64, String> = match content {
                Some(text) => body_hashes_for_spans(
                    text,
                    nodes
                        .iter()
                        .filter_map(|n| n.line.map(|l| (node_id_to_i64(n.id), l, n.end_line))),
                ),
                None => HashMap::new(),
            };

            // Load token strings for vocab decrement BEFORE removing map rows.
            let token_strings: Vec<String> = {
                let mut stmt = tx
                    .prepare(
                        "SELECT m.tokens FROM nodes_fts_map m \
                         JOIN nodes n ON n.id = m.node_id \
                         WHERE n.corpus=?1 AND n.path=?2",
                    )
                    .context("preparing token load for reindex_replace")?;
                let rows: Vec<String> = stmt
                    .query_map(params![corpus, path], |r| r.get::<_, String>(0))
                    .context("executing token load")?
                    .collect::<rusqlite::Result<_>>()
                    .context("collecting token strings")?;
                rows
            };

            // RFC-027 #813: hold the preserved definitions in a temp table so the
            // owned-edge and owned-site deletes can exclude them without a bound
            // parameter per id (a hot utility can have thousands). Always present
            // (and empty on the common purge path, where `NOT IN` an empty set
            // matches every owned row — exactly the pre-#813 whole-file purge).
            tx.execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS _preserved_src(id INTEGER PRIMARY KEY); \
                 DELETE FROM _preserved_src;",
            )
            .context("creating _preserved_src temp table")?;
            if !preserved.is_empty() {
                let mut ins = tx
                    .prepare("INSERT OR IGNORE INTO _preserved_src(id) VALUES(?1)")
                    .context("preparing _preserved_src insert")?;
                for id in &preserved {
                    ins.execute(params![id])
                        .context("inserting preserved src id")?;
                }
            }

            // Delete only OWNED (outbound) edges — inbound edges from other files
            // retained. RFC-027 #813: an owned edge whose src is a preserved
            // definition is committed truth still valid after a pure body edit, so
            // it is left in place rather than purged and re-derived.
            tx.execute(
                "DELETE FROM edges \
                 WHERE src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2) \
                   AND src NOT IN (SELECT id FROM _preserved_src)",
                params![corpus, path],
            )
            .context("deleting owned edges for reindex_replace")?;

            // RFC-027 #813 P2: capture the preserved definitions' occurrence rows
            // before the purge, so they can be re-recorded on their current lines
            // rather than dropped. A preserved definition shifted as a block, so
            // each occurrence's new line is its old line plus the definition's
            // delta; that keeps `find_references` correct mid-edit and matches the
            // line the next commit records, so the site store never diverges from
            // a full reindex (Invariant #4). Captured only when a delta is known.
            let preserved_sites: Vec<(i64, i64, String, i64, Option<i64>)> =
                if preserved_line_delta.is_empty() {
                    Vec::new()
                } else {
                    let mut stmt = tx
                        .prepare(
                            "SELECT src, dst, kind, line, col FROM edge_sites \
                             WHERE src IN (SELECT id FROM _preserved_src)",
                        )
                        .context("preparing preserved edge_sites capture")?;
                    let rows: Vec<(i64, i64, String, i64, Option<i64>)> = stmt
                        .query_map([], |r| {
                            Ok((
                                r.get::<_, i64>(0)?,
                                r.get::<_, i64>(1)?,
                                r.get::<_, String>(2)?,
                                r.get::<_, i64>(3)?,
                                r.get::<_, Option<i64>>(4)?,
                            ))
                        })
                        .context("executing preserved edge_sites capture")?
                        .collect::<rusqlite::Result<_>>()
                        .context("collecting preserved edge_sites")?;
                    rows
                };

            // RFC-027 #813 P2: capture the CHANGED definitions' committed
            // occurrences before the purge, so the live lane can enumerate them
            // as editor-resolution targets and reach references the tree-sitter
            // live extractor never detects (macro, desugared, trait-dispatched).
            // Only when preservation happened: a scoped body edit is exactly the
            // case where the changed region is small and the rest is committed
            // truth. On a whole-file re-derive (nothing preserved: first index,
            // added/removed symbol, or no `content`) the tree-sitter reparse
            // already re-emits the file's references, so this capture is skipped
            // to avoid enumerating the whole file on every bulk reindex.
            //
            // Issue #816 defect 1: the current buffer text, split once, so each
            // changed occurrence can be re-anchored by its preserved byte column
            // (below). Present here because preservation implies `content` is
            // `Some` (a body hash needs the text).
            let content_lines: Option<Vec<&str>> = content.map(|t| t.lines().collect());
            let changed_occurrences: Vec<travsr_core::ChangedOccurrence> = if preserved.is_empty() {
                Vec::new()
            } else {
                // Join the callee node for the reference's leaf name; a dst with
                // no node (external synthetic) cannot be named, so it is skipped.
                let mut stmt = tx
                    .prepare(
                        "SELECT es.src, es.line, es.col, es.kind, n.signature \
                         FROM edge_sites es JOIN nodes n ON n.id = es.dst \
                         WHERE es.src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2) \
                           AND es.src NOT IN (SELECT id FROM _preserved_src)",
                    )
                    .context("preparing changed-def occurrence capture")?;
                let rows: Vec<travsr_core::ChangedOccurrence> = stmt
                    .query_map(params![corpus, path], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, Option<i64>>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, String>(4)?,
                        ))
                    })
                    .context("executing changed-def occurrence capture")?
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .context("collecting changed-def occurrences")?
                    .into_iter()
                    .filter_map(|(src, line, col, kind, signature)| {
                        let name = travsr_core::ident::leaf_of(&signature).to_string();
                        // Remap the committed occurrence line onto the current
                        // buffer. The definition's start delta is the first
                        // estimate; it is correct above the edit point but stale
                        // below it, because a body edit does not move the start
                        // (issue #816 defect 1).
                        let start = *new_lines.get(&src)?;
                        let delta = start - old_lines.get(&src)?;
                        let estimate = line + delta;
                        // With a committed column and the current text, re-anchor
                        // the occurrence by that column to its true current line
                        // inside the definition's current span; abstain (heal at
                        // commit) when it cannot be found, rather than serving a
                        // stale position. Without a column or text, keep the
                        // start-delta estimate: the daemon fences it against the
                        // definition's current span and the editor resolves the
                        // live position, so a stale estimate is fail-closed.
                        let new_line = match (col, content_lines.as_deref()) {
                            (Some(c), Some(lines)) => {
                                // An unknown span end falls back to the buffer end,
                                // so the search can still move down to an insert;
                                // MAX_SNAP_RADIUS keeps that bound cheap.
                                let end = new_ends
                                    .get(&src)
                                    .copied()
                                    .unwrap_or(lines.len() as i64);
                                snap_changed_occurrence_line(
                                    lines, &name, c as usize, estimate, start, end,
                                )? as i64
                            }
                            _ => estimate,
                        };
                        if new_line < 1 {
                            return None;
                        }
                        Some(travsr_core::ChangedOccurrence {
                            src: i64_to_node_id(src),
                            line: new_line as u32,
                            col: col.map(|c| c as u32),
                            kind,
                            name,
                        })
                    })
                    .collect();
                rows
            };

            // #299 F7: purge OWNED occurrence rows only — an occurrence's `src` is
            // the enclosing node in the *same* file, so `src ∈ this file` selects
            // exactly the sites that live in this file and will be re-derived by
            // the next Phase B pass. Inbound sites (references from other files to
            // this file's symbols) are preserved, mirroring the owned-edge rule so
            // a blank-line/body edit stays lossless. (A blanket FK cascade would
            // wrongly nuke those inbound sites when the node is deleted+reinserted.)
            //
            // The whole owned set is deleted, preserved definitions included,
            // because a preserved definition may have drifted lines; its sites are
            // then re-inserted below on their current lines (RFC-027 #813 P2), so
            // the row a preserved definition keeps carries the same line the next
            // commit would write, never a stale duplicate.
            tx.execute(
                "DELETE FROM edge_sites \
                 WHERE src IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("deleting owned edge_sites for reindex_replace")?;

            // RFC-027 #813 P2: re-record the preserved definitions' occurrences on
            // their current lines.
            if !preserved_sites.is_empty() {
                let mut ins = tx
                    .prepare(
                        "INSERT OR IGNORE INTO edge_sites(src, dst, kind, line, col) \
                         VALUES(?1, ?2, ?3, ?4, ?5)",
                    )
                    .context("preparing preserved edge_sites reinsert")?;
                // RFC-027 #813 P2: the body is byte-identical so the occurrence
                // column is unchanged; only its line shifted by the definition's
                // block delta. Carry the stored col through verbatim.
                for (src, dst, kind, line, col) in &preserved_sites {
                    let Some(delta) = preserved_line_delta.get(src) else {
                        continue;
                    };
                    let new_line = line + delta;
                    if new_line < 1 {
                        continue; // a nonsensical remap is dropped, healed at commit
                    }
                    ins.execute(params![src, dst, kind, new_line, col])
                        .context("re-inserting preserved edge_site")?;
                }
            }

            // Retract FTS for old nodes.
            tx.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) \
                 SELECT 'delete', m.node_id, m.tokens \
                 FROM nodes_fts_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.corpus=?1 AND n.path=?2",
                params![corpus, path],
            )
            .context("retracting FTS rows for reindex_replace")?;
            tx.execute(
                "DELETE FROM nodes_fts_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("removing nodes_fts_map rows for reindex_replace")?;
            for ts in &token_strings {
                Self::vocab_decrement(&tx, ts)?;
            }

            // #478: retract nodes_fts_words the same way (no vocab decrement needed).
            tx.execute(
                "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                 SELECT 'delete', m.node_id, m.sig_words, m.path_words \
                 FROM nodes_fts_words_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.corpus=?1 AND n.path=?2",
                params![corpus, path],
            )
            .context("retracting FTS word rows for reindex_replace")?;
            tx.execute(
                "DELETE FROM nodes_fts_words_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE corpus=?1 AND path=?2)",
                params![corpus, path],
            )
            .context("removing nodes_fts_words_map rows for reindex_replace")?;

            // Delete old nodes — v17 trigger auto-writes tombstones for changed IDs.
            tx.execute(
                "DELETE FROM nodes WHERE corpus=?1 AND path=?2",
                params![corpus, path],
            )
            .context("deleting old nodes for reindex_replace")?;

            // Write new nodes + FTS within this transaction.
            // put_node_fts accepts &Connection; Transaction derefs to Connection.
            // `new_ids` (this parse's NodeId set) is already computed above for the
            // pure-body-edit test; reuse it for the removed-symbol diff below.
            for node in nodes {
                let id_i64 = node_id_to_i64(node.id);
                // RFC-027 #813: stamp the body hash so the next save can tell
                // whether this definition was left untouched. `NULL` when no
                // `content` was supplied (the pre-#813 behaviour) or the node has
                // no line span, in which case that definition is never preserved.
                let body_hash = new_body_hashes.get(&id_i64);
                tx.execute(
                    "INSERT INTO nodes(id, corpus, root, path, language, signature, kind, package, line, end_line, is_noise, test_role, body_hash) \
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13) \
                     ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, \
                     package = excluded.package, \
                     line = COALESCE(excluded.line, nodes.line), \
                     end_line = COALESCE(excluded.end_line, nodes.end_line), \
                     is_noise = excluded.is_noise, \
                     test_role = excluded.test_role, \
                     body_hash = excluded.body_hash",
                    params![
                        id_i64,
                        node.vname.corpus,
                        node.vname.root,
                        node.vname.path,
                        node.vname.language,
                        node.vname.signature,
                        node.kind,
                        node.package,
                        node.line.map(|l| l as i64),
                        node.end_line.map(|l| l as i64),
                        travsr_core::noise::is_structural_noise(node),
                        node.test_role.as_i64(),
                        body_hash,
                    ],
                )
                .context("inserting node in reindex_replace")?;
                Self::put_node_fts(&tx, node).context("put_node_fts in reindex_replace")?;
                Self::put_node_fts_words(&tx, node)
                    .context("put_node_fts_words in reindex_replace")?;
            }

            // Write new edges. RFC-027 #813: skip this parse's tree-sitter edges
            // for a preserved definition — its committed (scip/lsif) edges were
            // left in place above and already dominate the tree-sitter row, so
            // re-inserting it is at best a no-op and at worst reintroduces a
            // by-name edge the committed graph had resolved away.
            for edge in edges {
                let src_i64 = node_id_to_i64(edge.src);
                if preserved.contains(&src_i64) {
                    continue;
                }
                tx.execute(
                    "INSERT INTO edges(src, dst, kind, provenance, confidence) \
                     VALUES(?1, ?2, ?3, 'tree-sitter', ?4) \
                     ON CONFLICT(src, dst, kind) DO NOTHING",
                    params![
                        src_i64,
                        node_id_to_i64(edge.dst),
                        edge.kind.as_str(),
                        edge.confidence.map(|c| c as i64),
                    ],
                )
                .context("inserting edge in reindex_replace")?;
            }

            // Detect symbols that vanished (old id absent from new parse).
            let removed_ids: Vec<i64> = old_ids
                .iter()
                .filter(|id| !new_ids.contains(id))
                .copied()
                .collect();

            let mut callers = DirtySet::default();
            if !removed_ids.is_empty() {
                // Prepared once for the whole loop, same reason as the batch
                // writer's loop: an uncached execute re-prepares once per
                // removed id, and on a `--force` re-parse that is every node.
                let mut del_inbound = tx
                    .prepare_cached("DELETE FROM edges WHERE dst = ?1")
                    .context("preparing inbound orphan delete")?;
                for &removed_id in &removed_ids {
                    // Find callers before deleting their inbound edge.
                    let mut stmt = tx
                        .prepare(
                            "SELECT DISTINCT n.path FROM edges e JOIN nodes n ON n.id = e.src \
                             WHERE e.dst = ?1 AND n.path != ?2",
                        )
                        .context("preparing per-symbol caller query")?;
                    for row in stmt
                        .query_map(params![removed_id, path], |r| r.get::<_, String>(0))
                        .context("executing per-symbol caller query")?
                    {
                        callers.insert(row.context("reading caller path")?);
                    }
                    // Eagerly delete inbound orphan edges for this removed symbol.
                    del_inbound
                        .execute(params![removed_id])
                        .context("deleting inbound orphan edges for removed symbol")?;
                }
            }

            // Upsert file hash.
            tx.execute(
                "INSERT INTO files(path, sha256, last_indexed_at) \
                 VALUES(?1, ?2, unixepoch()) \
                 ON CONFLICT(path) DO UPDATE SET \
                   sha256 = excluded.sha256, \
                   last_indexed_at = excluded.last_indexed_at",
                params![path, new_hash],
            )
            .context("upserting file hash in reindex_replace")?;

            tx.commit()
                .context("committing reindex_replace transaction")?;

            // RFC-027 #813: the definitions this edit changed are every node the
            // parse produced whose committed edges were not preserved. On a pure
            // body edit that is just the edited definitions; otherwise it is the
            // whole file. This is the changed region both live lanes scope to
            // (see `ReplaceReport::changed_defs`).
            let changed_defs: Vec<NodeId> = nodes
                .iter()
                .filter(|n| !preserved.contains(&node_id_to_i64(n.id)))
                .map(|n| n.id)
                .collect();

            Ok(ReplaceReport {
                removed_count: removed_ids.len(),
                callers,
                changed_defs,
                changed_occurrences,
                preserved_any: !preserved.is_empty(),
            })
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Tier-2 disk-vs-graph reconciliation with §6.5 safety guards.
    ///
    /// `walked_paths` is the VName-normalised path set found on disk (caller
    /// is responsible for the walk + normalisation). `db_paths − walked` = ghosts.
    ///
    /// Safety (§6.5): mass-delete circuit breaker + TOCTOU re-check + batched
    /// deletes ≤500/txn so a large branch switch cannot hold the write lock for
    /// seconds or starve queries.
    pub fn reconcile(
        &mut self,
        walked_paths: &std::collections::HashSet<String>,
        policy: &SafetyPolicy,
        repo_root: &std::path::Path,
        corpus: &str,
    ) -> Result<GcReport, StoreError> {
        (|| -> AnyResult<GcReport> {
            let node_count = self
                .conn
                .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get::<_, i64>(0))
                .context("counting nodes for GcReport")? as u64;
            let edge_count = self
                .conn
                .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get::<_, i64>(0))
                .context("counting edges for GcReport")? as u64;
            let mut report = GcReport {
                node_count,
                edge_count,
                ..GcReport::default()
            };

            let db_hashes = self
                .get_all_file_hashes()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let db_paths: std::collections::HashSet<String> = db_hashes.into_keys().collect();

            let ghosts: Vec<String> = db_paths.difference(walked_paths).cloned().collect();

            // §6.5 S2 — mass-delete circuit breaker.
            let ceiling = std::cmp::max(
                policy.mass_delete_ceiling_min,
                (db_paths.len() as f64 * policy.mass_delete_ceiling_pct) as usize,
            );
            if ghosts.len() > ceiling {
                let reason = format!(
                    "ghost count {} exceeds ceiling {} ({}% of {} tracked paths); \
                     re-run with --force to override",
                    ghosts.len(),
                    ceiling,
                    (policy.mass_delete_ceiling_pct * 100.0) as u32,
                    db_paths.len(),
                );
                tracing::error!(
                    ghost_count = ghosts.len(),
                    ceiling,
                    "reconcile: circuit breaker tripped, deleting nothing"
                );
                report.aborted = true;
                report.abort_reason = Some(reason);
                return Ok(report);
            }

            const BATCH: usize = 500;
            for chunk in ghosts.chunks(BATCH) {
                for ghost_path in chunk {
                    // §6.5 S3 — TOCTOU re-check.
                    if policy.toctou_recheck && repo_root.join(ghost_path).exists() {
                        tracing::debug!(
                            path = %ghost_path,
                            "reconcile: TOCTOU, file reappeared, skipping"
                        );
                        continue;
                    }
                    self.delete_file(corpus, ghost_path)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    report.ghost_paths.push(ghost_path.clone());
                }
                // Yield between batches so queries are not starved.
                std::thread::yield_now();
            }

            Ok(report)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Read-only orphan-edge count: how many edges have a `src` or `dst` absent
    /// from `nodes`, without deleting anything. This is the detection half of
    /// [`sweep_orphans`], so `fsck` can report orphans in the default (no-`--fix`)
    /// mode instead of only ever removing them (issue #580).
    pub fn count_orphans(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM edges \
                 WHERE src NOT IN (SELECT id FROM nodes) \
                    OR dst NOT IN (SELECT id FROM nodes)",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting orphan edges")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Stable content fingerprint of every node, for equality assertions.
    ///
    /// Ordered by id so two graphs built by different routes compare directly.
    pub fn node_fingerprint(&self) -> Result<Vec<String>, StoreError> {
        (|| -> AnyResult<Vec<String>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT corpus || '|' || path || '|' || signature || '|' || kind \
                     FROM nodes ORDER BY id",
                )
                .context("node_fingerprint: prepare")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .context("node_fingerprint: query")?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.context("node_fingerprint: decode")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Stable content fingerprint of every edge **including its provenance**.
    ///
    /// Provenance is in the fingerprint on purpose. The RFC-027 convergence
    /// property is not "the same number of edges" but "the same graph", and an
    /// un-ratified `live` row sitting where a `scip` row belongs has the same
    /// count and the wrong meaning. Counting alone would let exactly the
    /// failure this property exists to rule out slip through.
    pub fn edge_fingerprint(&self) -> Result<Vec<String>, StoreError> {
        (|| -> AnyResult<Vec<String>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT src || '|' || dst || '|' || kind || '|' || provenance \
                     FROM edges ORDER BY src, dst, kind",
                )
                .context("edge_fingerprint: prepare")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .context("edge_fingerprint: query")?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.context("edge_fingerprint: decode")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 8.3: retire the live overlay for the languages that just
    /// completed a Phase B run.
    ///
    /// Most live edges never reach this. When Phase B re-derives an edge the
    /// live lane had already found, the ratification write upserts the same
    /// `(src, dst, kind)` row and relabels its provenance, so the edge is
    /// ratified *in place*. What is still marked `live` afterwards is exactly
    /// the set Phase B did **not** re-derive, which is why deleting it cannot
    /// lose a real edge. That is also why the ratification writes must run
    /// first, and must not be prevented from overwriting a `live` row.
    ///
    /// **Scoped, not blanket.** The Phase B completion marker advances whenever
    /// *any* language produced results, even when another language's sidecar
    /// crashed (#712). A blanket delete would therefore discard the overlay for
    /// a language whose truth was never re-derived, taking away precision the
    /// developer had a moment earlier and not giving it back until that sidecar
    /// is fixed and a later commit runs. Live edges for a crashed language
    /// survive, still labeled `live`, which is honest.
    ///
    /// Keyed on the **src node's** language rather than `edges.language`, which
    /// is a derived label reconciled after the fact and defaulted at insert
    /// time, so it cannot be trusted as the scoping key.
    ///
    /// Invariant #4 is unaffected: a clean run has nothing crashed, so every
    /// language that has nodes is in `languages` and the sweep is total.
    pub fn sweep_live_edges_for_languages(
        &mut self,
        languages: &[String],
    ) -> Result<u64, StoreError> {
        if languages.is_empty() {
            return Ok(0);
        }
        let placeholders = std::iter::repeat("?")
            .take(languages.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "DELETE FROM edges WHERE provenance = 'live' \
             AND src IN (SELECT id FROM nodes WHERE language IN ({placeholders}))"
        );
        self.conn
            .execute(&sql, params_from_iter(languages.iter()))
            .map(|n| n as u64)
            .context("sweeping live edges for ratified languages")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Scoped orphan sweep for the incremental path: drop edges *owned by*
    /// `paths` whose destination is absent from `nodes`.
    ///
    /// The init path already guarantees this. `flush_staging_to_production`
    /// promotes every node in the batch and then drops any staged edge with a
    /// missing endpoint, which it calls making "no orphan edges" a store
    /// invariant at the staging boundary. `reindex_replace` had no equivalent,
    /// so the two paths disagreed and Invariant #4 (incremental == full) failed
    /// in the orphan dimension.
    ///
    /// The edges this matters for are speculative by construction. TypeScript
    /// import resolution emits a `resolves-to` candidate per plausible
    /// extension (`./user` becomes `user.ts`, `user.tsx`, `user.js`) because it
    /// cannot know which one exists without touching the store. On a full index
    /// the losers are dropped at the staging boundary; incrementally they
    /// survived, so every ordinary TypeScript edit left two dead edges behind
    /// and `fsck` reported orphans on a healthy repo.
    ///
    /// **Call this after the whole batch is written, never per file.** That is
    /// the same position the staging flush occupies: an edge from an
    /// already-processed file to one later in the same batch is legitimately
    /// dangling until that file's nodes land, and sweeping mid-batch would
    /// delete it.
    ///
    /// Scoped to `paths` rather than the whole table because the unscoped
    /// [`sweep_orphans`] is a full edge scan, which is a fine cost for `fsck`
    /// and not one to pay on every save.
    pub fn sweep_orphan_edges_for_paths(
        &mut self,
        corpus: &str,
        paths: &[String],
    ) -> Result<u64, StoreError> {
        if paths.is_empty() {
            return Ok(0);
        }
        (|| -> AnyResult<u64> {
            let tx = self
                .conn
                .transaction()
                .context("sweep_orphan_edges_for_paths: begin")?;
            let mut swept = 0u64;
            for path in paths {
                // `src IN (…)` hits the edges primary key prefix, so this is a
                // bounded probe per path rather than a scan of the edge table.
                swept += tx
                    .execute(
                        "DELETE FROM edges \
                         WHERE src IN (SELECT id FROM nodes WHERE corpus = ?1 AND path = ?2) \
                           AND dst NOT IN (SELECT id FROM nodes)",
                        params![corpus, path],
                    )
                    .context("sweeping orphan edges for path")? as u64;
            }
            tx.commit()
                .context("sweep_orphan_edges_for_paths: commit")?;
            Ok(swept)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Orphan-edge sweep: deletes every edge whose src or dst is absent from
    /// `nodes`. Should return 0 in correct operation after Tiers 0–2; a non-zero
    /// count indicates a write-path invariant violation.
    pub fn sweep_orphans(&mut self) -> Result<u64, StoreError> {
        self.conn
            .execute(
                "DELETE FROM edges \
                 WHERE src NOT IN (SELECT id FROM nodes) \
                    OR dst NOT IN (SELECT id FROM nodes)",
                [],
            )
            .map(|n| n as u64)
            .context("sweeping orphan edges")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Read-only graph integrity report: the report-only half of `travsr fsck`.
    ///
    /// NEVER mutates: no reconcile, no sweep, no writes of any kind. Safe to
    /// call against a store opened with [`Self::open_read_only`] (§ `query_only`
    /// PRAGMA would hard-fail any write attempt).
    ///
    /// Computes: `node_count`, `edge_count`, the ghost-path set (DB paths
    /// absent on disk), orphan-edge count, self-referential `ref/call` edge
    /// count, and the lexical (FTS words) index parity check. Extracted from
    /// `travsr_daemon::fsck_repo`'s report path (#636) so travsr-mcp (which
    /// must never depend on travsr-daemon) can answer `get_graph_health`
    /// without opening the store read-write.
    ///
    /// O(F) where F = tracked file count (one `exists()` stat per DB path).
    pub fn integrity_report(&self, repo_root: &std::path::Path) -> Result<GcReport, StoreError> {
        let node_count = self.node_count()?;
        let fts_words_count = self.fts_words_node_count()?;
        let lexical_index_parity_issue = if node_count == fts_words_count {
            None
        } else {
            Some(format!(
                "nodes ({node_count}) != nodes_fts_words_map ({fts_words_count}), \
                 run `travsr init` to re-backfill the lexical word index (#478)"
            ))
        };

        let mut report = GcReport {
            node_count,
            edge_count: self.edge_count()?,
            lexical_index_parity_issue,
            ..GcReport::default()
        };

        // Ghost detection stats each DB path directly rather than re-walking the
        // disk (see `travsr_daemon::fsck_repo`'s doc comment for why, #580):
        // statting the DB paths is symmetric by construction, with no walk
        // config or filters to drift from the write path.
        let db_paths: std::collections::HashSet<String> =
            self.get_all_file_hashes()?.into_keys().collect();
        let mut ghosts: Vec<String> = Vec::new();
        for path in &db_paths {
            if !repo_root.join(path).exists() {
                ghosts.push(path.clone());
            }
        }
        report.ghost_paths = ghosts;

        report.orphan_edges_detected = self.count_orphans()?;
        report.self_ref_call_edges_detected = self.count_self_ref_call_edges()?;

        Ok(report)
    }

    /// Delete all nodes (and their edges) whose VName path starts with `prefix`.
    ///
    /// Used by `init_repo` to purge ghost nodes from directories that were added
    /// to `SKIP_DIRS` after a previous index run (e.g. `.claude/worktrees/`).
    /// Edges are deleted first; both run in a single transaction.
    /// Returns the number of nodes removed.
    ///
    /// Deliberately NOT on the [`Store`] trait — this is a SqliteStore-specific
    /// cleanup operation for the daemon init path.
    pub fn delete_nodes_for_path_prefix(&mut self, prefix: &str) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let pattern = format!("{prefix}%");
            let tx = self
                .conn
                .transaction()
                .context("starting prefix-delete transaction")?;

            let count: i64 = tx
                .query_row(
                    "SELECT count(*) FROM nodes WHERE path LIKE ?1",
                    params![pattern],
                    |row| row.get(0),
                )
                .context("counting nodes to prefix-delete")?;

            // Edges have no FK cascade — delete them explicitly before removing nodes.
            tx.execute(
                "DELETE FROM edges \
                 WHERE src IN (SELECT id FROM nodes WHERE path LIKE ?1) \
                    OR dst IN (SELECT id FROM nodes WHERE path LIKE ?1)",
                params![pattern],
            )
            .context("deleting edges for path prefix")?;

            // Load token strings for vocab decrement BEFORE removing map rows.
            let prefix_token_strings: Vec<String> = {
                let mut stmt = tx
                    .prepare(
                        "SELECT m.tokens FROM nodes_fts_map m \
                         JOIN nodes n ON n.id = m.node_id WHERE n.path LIKE ?1",
                    )
                    .context("preparing token load for prefix delete")?;
                let collected: Vec<String> = stmt
                    .query_map(params![pattern], |row| row.get(0))
                    .context("executing token load for prefix delete")?
                    .collect::<Result<_, _>>()
                    .context("collecting token strings for vocab decrement (prefix)")?;
                collected
            };

            // Retract FTS entries before deleting nodes (tokens stored in map).
            tx.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) \
                 SELECT 'delete', m.node_id, m.tokens \
                 FROM nodes_fts_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.path LIKE ?1",
                params![pattern],
            )
            .context("retracting FTS rows for path prefix")?;
            tx.execute(
                "DELETE FROM nodes_fts_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE path LIKE ?1)",
                params![pattern],
            )
            .context("removing nodes_fts_map rows for path prefix")?;

            // Decrement fts_vocab refcounts for all retracted tokens (v10 L2-A).
            for ts in &prefix_token_strings {
                Self::vocab_decrement(&tx, ts)?;
            }

            // #478: retract nodes_fts_words the same way (no vocab decrement needed).
            tx.execute(
                "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                 SELECT 'delete', m.node_id, m.sig_words, m.path_words \
                 FROM nodes_fts_words_map m JOIN nodes n ON n.id = m.node_id \
                 WHERE n.path LIKE ?1",
                params![pattern],
            )
            .context("retracting FTS word rows for path prefix")?;
            tx.execute(
                "DELETE FROM nodes_fts_words_map \
                 WHERE node_id IN (SELECT id FROM nodes WHERE path LIKE ?1)",
                params![pattern],
            )
            .context("removing nodes_fts_words_map rows for path prefix")?;

            tx.execute("DELETE FROM nodes WHERE path LIKE ?1", params![pattern])
                .context("deleting nodes for path prefix")?;

            tx.commit()
                .context("committing prefix-delete transaction")?;
            Ok(count as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    // ── Name-search family (#453) ─────────────────────────────────────────────
    // `search_nodes_by_name{,_exact}{,_with_lang}` share one SELECT + rank CASE
    // and one row decoder; only the WHERE-match fragment and the optional
    // language filter differ. Keeping the rank ladder in a single place stops the
    // exact and non-exact paths from silently diverging on the next tweak.

    /// SELECT column list plus the shared relevance `rank` expression, up to and
    /// including `FROM nodes`. Callers append `WHERE <match> ...`. `?1` is the
    /// query term; `?2` (when present) is the language filter.
    const NAME_SEARCH_SELECT: &'static str = "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line, test_role,
  (
    CASE
      WHEN signature = ?1 THEN 0
      WHEN signature LIKE ?1 || ':%' OR signature LIKE '%:' || ?1 OR signature LIKE '%:' || ?1 || ':%' THEN 10
      WHEN signature LIKE ?1 || '%' THEN 20
      WHEN path = ?1 THEN 5
      WHEN path LIKE '%/' || ?1 THEN 15
      ELSE 40
    END
    + CASE WHEN path LIKE '%_test.go' OR path LIKE '%test/%'
           OR path LIKE '%.test.%' OR path LIKE '%_test.%'
           OR path LIKE 'tests/%' OR path LIKE '%/tests/%' THEN 25 ELSE 0 END
    + CASE WHEN path LIKE 'third_party/%' OR path LIKE 'vendor/%'
           OR path LIKE '%/node_modules/%' THEN 50 ELSE 0 END
    + CASE WHEN path LIKE '%zz_generated%' OR path LIKE '%.pb.go'
           OR path LIKE '%_pb2.py' OR path LIKE '%.pb.cc' OR path LIKE '%.pb.h'
           OR path LIKE '%.generated.%' OR path LIKE '%.g.dart'
           OR path LIKE '%mock%' OR path LIKE '%fake%' THEN 20 ELSE 0 END
    + (LENGTH(path) / 32)
    + CASE kind
        WHEN 'method'      THEN 0 WHEN 'function'    THEN 0 WHEN 'constructor' THEN 0
        WHEN 'class'       THEN 2 WHEN 'interface'   THEN 2 WHEN 'struct'      THEN 2
        WHEN 'enum'        THEN 2 WHEN 'trait'       THEN 2 WHEN 'union'       THEN 2
        WHEN 'constant'    THEN 3 WHEN 'static'      THEN 3
        WHEN 'impl'        THEN 4 WHEN 'type'        THEN 4 WHEN 'module'      THEN 4
        WHEN 'field'       THEN 6 WHEN 'var'         THEN 6 WHEN 'variable'    THEN 6
        WHEN 'property'    THEN 6
        WHEN 'import'      THEN 8 WHEN 'file-module' THEN 8 WHEN 'crate'       THEN 8
        WHEN 'go-pkg'      THEN 10
        ELSE 4
      END
  ) AS rank
FROM nodes";

    /// Loose default match: any node whose signature or path *contains* `?1`.
    const NAME_MATCH_SUBSTRING: &'static str = "(signature LIKE '%' || ?1 || '%'
   OR path LIKE '%' || ?1 || '%')";

    /// #778: upper bound on [`Self::exact_leaf_name_count`]. Only the
    /// rare-vs-generic distinction is used downstream (a small count grounds a
    /// short exact symbol; anything past a few hundred is already "generic" for
    /// any realistic corpus via IDF), so counting past this adds nothing and the
    /// `LIMIT` caps the worst-case scan for a pathologically common short name.
    /// Hitting the cap means "at least this many"; the caller saturates the
    /// count to the corpus size so IDF floors to generic (a truncated 4096 would
    /// still read as specific — `idf_weight` does not saturate near it).
    const LEAF_NAME_COUNT_CAP: usize = 4096;

    /// Exact/word-boundary/prefix match: drops the loose `ELSE 40` substring
    /// tier so short, common names don't drag in unrelated symbols (#453).
    const NAME_MATCH_BOUNDARY: &'static str = "(signature = ?1
   OR signature LIKE ?1 || ':%'
   OR signature LIKE '%:' || ?1
   OR signature LIKE '%:' || ?1 || ':%'
   OR signature LIKE ?1 || '%'
   OR path = ?1
   OR path LIKE '%/' || ?1)";

    /// Decode one row of [`Self::NAME_SEARCH_SELECT`] into a [`Node`].
    fn map_search_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Node> {
        let id = i64_to_node_id(row.get::<_, i64>(0)?);
        let vname = VName::new(
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
        );
        let kind: String = row.get(6)?;
        let package: String = row.get(7)?;
        let line: Option<i64> = row.get(8)?;
        let end_line: Option<i64> = row.get(9)?;
        // #479: carry the real test_role so `is_anchor_noise` (the only reader
        // of the `Node.test_role` field) can keep AST-detected inline tests out
        // of the anchor pool, not just the ones whose name `is_test_symbol`
        // recognises.
        let test_role: i64 = row.get(10)?;
        Ok(Node {
            id,
            vname,
            kind,
            package,
            line: line.and_then(|l| u32::try_from(l).ok()),
            end_line: end_line.and_then(|l| u32::try_from(l).ok()),
            test_role: TestRole::from_i64(test_role),
        })
    }

    /// Shared body behind the four name-search wrappers: assemble the query from
    /// [`Self::NAME_SEARCH_SELECT`] + `match_clause` (one of the `NAME_MATCH_*`
    /// consts) + the optional language filter, then decode.
    fn run_name_search(
        &self,
        match_clause: &str,
        name: &str,
        lang: Option<&str>,
    ) -> Result<Vec<Node>, StoreError> {
        (|| -> AnyResult<Vec<Node>> {
            let mut sql =
                String::with_capacity(Self::NAME_SEARCH_SELECT.len() + match_clause.len() + 96);
            sql.push_str(Self::NAME_SEARCH_SELECT);
            sql.push_str("\nWHERE ");
            sql.push_str(match_clause);
            sql.push_str("\n  AND kind != 'doc-chunk'");
            if lang.is_some() {
                sql.push_str("\n  AND language = ?2");
            }
            sql.push_str(&format!(
                "\nORDER BY rank ASC, id ASC\nLIMIT {NODE_NAME_SEARCH_LIMIT}"
            ));

            let mut stmt = self.conn.prepare(&sql).context("preparing search query")?;
            let rows = match lang {
                Some(l) => stmt.query_map(params![name, l], Self::map_search_row),
                None => stmt.query_map(params![name], Self::map_search_row),
            }
            .context("executing search query")?;

            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding search row")?);
            }
            tracing::debug!(nodes_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Search nodes by name substring, ranked best-first.
    ///
    /// Results are ordered by match quality (exact > prefix > substring), then
    /// production-file bias (test and vendor paths rank lower), then path length.
    /// At most 100 results are returned. Excludes `doc-chunk` nodes: a doc's
    /// markdown path (e.g. `docs/adrs/ADR-018-drop-kuzu-backend.md`) trivially
    /// substring-matches on stray query words, which was leaking a doc-only
    /// anchor into the code lane's exact-anchor confidence classification
    /// (`Confidence::Exact` on unrelated code results) — docs have their own
    /// dedicated, separately-floored KNN lane (`doc_lane_candidates`).
    pub fn search_nodes_by_name(&self, name: &str) -> Result<Vec<Node>, StoreError> {
        // Log the query name (symbol/path, not file contents — SEC log-redaction rule).
        let _span = tracing::debug_span!("store.search_nodes_by_name", query = name).entered();
        self.run_name_search(Self::NAME_MATCH_SUBSTRING, name, None)
    }

    /// Search nodes by name exact/prefix/word-boundary only, filtering out pure substring matches.
    pub fn search_nodes_by_name_exact(&self, name: &str) -> Result<Vec<Node>, StoreError> {
        let _span =
            tracing::debug_span!("store.search_nodes_by_name_exact", query = name).entered();
        self.run_name_search(Self::NAME_MATCH_BOUNDARY, name, None)
    }

    /// Exact-signature lookup with optional path-suffix pin.
    ///
    /// Uses the `nodes_signature_idx` index for O(log N) access. When
    /// `path_hint` is `Some`, only rows whose `path` equals or ends with the
    /// hint are returned, and those rows sort first. Returns at most
    /// [`NODE_EXACT_LOOKUP_LIMIT`] non-file nodes.
    ///
    /// Callers interpret the result length:
    /// - 0  → not found; fall back to fuzzy search
    /// - 1  → unambiguous match
    /// - >1 + `path_hint` Some → path-pinned rows sort first; take index 0
    /// - >1 + `path_hint` None → genuinely ambiguous; caller must disambiguate
    pub fn lookup_nodes_exact(
        &self,
        signature: &str,
        path_hint: Option<&str>,
    ) -> Result<Vec<Node>, StoreError> {
        // Escape SQL LIKE metacharacters so a path hint is matched literally.
        // Paired with `ESCAPE '\'` in the query. `\` itself is escaped first.
        fn escape_like(s: &str) -> String {
            let mut out = String::with_capacity(s.len());
            for c in s.chars() {
                if matches!(c, '\\' | '%' | '_') {
                    out.push('\\');
                }
                out.push(c);
            }
            out
        }
        let _span = tracing::debug_span!(
            "store.lookup_nodes_exact",
            signature,
            path_hint = path_hint.unwrap_or("(none)")
        )
        .entered();
        // `?2` is the raw hint (exact match + NULL guard); `?3` is the same hint
        // with LIKE metacharacters (`%`, `_`, `\`) escaped so a suffix pin like
        // "my_file.rs" is matched literally and never treated as a wildcard.
        let like_hint: Option<String> = path_hint.map(escape_like);
        (|| -> AnyResult<Vec<Node>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line
FROM nodes
WHERE signature = ?1
  AND kind NOT IN ('file', 'import')
  AND (
        ?2 IS NULL
     OR path = ?2
     OR path LIKE '%/' || ?3 ESCAPE '\\'
  )
ORDER BY
  CASE
    WHEN ?2 IS NOT NULL AND (path = ?2 OR path LIKE '%/' || ?3 ESCAPE '\\') THEN 0
    ELSE 1
  END ASC,
  id ASC
LIMIT ?4",
                )
                .context("preparing lookup_nodes_exact query")?;

            let rows = stmt
                .query_map(
                    params![
                        signature,
                        path_hint,
                        like_hint,
                        NODE_EXACT_LOOKUP_LIMIT as i64
                    ],
                    |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing lookup_nodes_exact query")?;

            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding lookup_nodes_exact row")?);
            }
            tracing::debug!(nodes_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn all_nodes(&self) -> Result<Vec<Node>, StoreError> {
        (|| -> AnyResult<Vec<Node>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line FROM nodes",
                )
                .context("preparing all_nodes query")?;
            let rows = stmt
                .query_map([], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing all_nodes query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding all_nodes row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Fetch all nodes of a given `kind` (full-table scan — use for cheap kinds like "file").
    pub fn nodes_by_kind(&self, kind: &str) -> Result<Vec<Node>, StoreError> {
        (|| -> AnyResult<Vec<Node>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line
                     FROM nodes WHERE kind = ?1",
                )
                .context("preparing nodes_by_kind query")?;
            let rows = stmt
                .query_map(params![kind], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing nodes_by_kind query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding nodes_by_kind row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Bulk-lookup nodes by signature strings. Returns `(id, signature, path)` triples.
    /// Used by the daemon to resolve `UnresolvedCall`s emitted by Phase B.
    /// Returns `(id, signature, path, language)`. Language is carried so the
    /// daemon resolver can scope candidates to the caller's own language — a
    /// call in one language must never resolve to a same-signature definition
    /// in another (E4).
    pub fn nodes_by_signatures(
        &self,
        sigs: &[String],
    ) -> Result<Vec<(NodeId, String, String, String)>, StoreError> {
        if sigs.is_empty() {
            return Ok(Vec::new());
        }
        (|| -> AnyResult<Vec<(NodeId, String, String, String)>> {
            let placeholders = sigs
                .iter()
                .enumerate()
                .map(|(i, _)| format!("?{}", i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT id, signature, path, language FROM nodes WHERE signature IN ({placeholders})"
            );
            let mut stmt = self
                .conn
                .prepare(&sql)
                .context("preparing nodes_by_signatures")?;
            let params_vec: Vec<&dyn rusqlite::ToSql> =
                sigs.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt
                .query_map(params_vec.as_slice(), |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let sig: String = row.get(1)?;
                    let path: String = row.get(2)?;
                    let lang: String = row.get(3)?;
                    Ok((id, sig, path, lang))
                })
                .context("executing nodes_by_signatures")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding nodes_by_signatures row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Function/method definition nodes whose *leaf identifier* (the segment
    /// after the last `.`, or the whole name for an unqualified `fn:name`)
    /// matches one of `names`.
    ///
    /// #299 R1: a Rust method call `recv.method()` cannot be resolved to the
    /// receiver's type syntactically, so the native extractor emits a bare
    /// `fn:method` `UnresolvedCall`. When the definition is a qualified
    /// `fn:Type.method` / `method:Type.method` node, the exact-signature pass
    /// misses it. This precise, index-time fallback recovers the qualified
    /// node by leaf name; the caller resolves only when the match is unique so
    /// no false edge is created. LIKE metacharacters (`_`, `%`) in identifiers
    /// are escaped so `announce_all` is matched literally.
    pub fn fn_nodes_by_leaf_name(
        &self,
        names: &[String],
    ) -> Result<Vec<(NodeId, String, String, String)>, StoreError> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        (|| -> AnyResult<Vec<(NodeId, String, String, String)>> {
            // Each name contributes 3 params (exact + 2 LIKE). Chunk so a large
            // `names` slice never exceeds SQLite's SQLITE_MAX_VARIABLE_NUMBER
            // (default 999 on older builds) or the expression-tree depth limit:
            // 300 names → 900 params / 900 OR-terms per statement.
            const NAMES_PER_CHUNK: usize = 300;
            let mut out = Vec::new();
            for chunk in names.chunks(NAMES_PER_CHUNK) {
                let mut clauses: Vec<&str> = Vec::with_capacity(chunk.len() * 3);
                let mut params: Vec<String> = Vec::with_capacity(chunk.len() * 3);
                for name in chunk {
                    let esc = name
                        .replace('\\', "\\\\")
                        .replace('%', "\\%")
                        .replace('_', "\\_");
                    clauses.push("signature = ?");
                    params.push(format!("fn:{name}"));
                    clauses.push("signature LIKE ? ESCAPE '\\'");
                    params.push(format!("fn:%.{esc}"));
                    clauses.push("signature LIKE ? ESCAPE '\\'");
                    params.push(format!("method:%.{esc}"));
                }
                // Kind set matches `fetch_all_fn_spans` (incl. the `fn` kind used
                // by some Phase A parsers) so leaf-name resolution and span
                // attribution consider the same node population.
                let sql = format!(
                    "SELECT id, signature, path, language FROM nodes \
                     WHERE kind IN ('function','method','fn') AND ({})",
                    clauses.join(" OR ")
                );
                let mut stmt = self
                    .conn
                    .prepare(&sql)
                    .context("preparing fn_nodes_by_leaf_name")?;
                let params_vec: Vec<&dyn rusqlite::ToSql> =
                    params.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
                let rows = stmt
                    .query_map(params_vec.as_slice(), |row| {
                        let id = i64_to_node_id(row.get::<_, i64>(0)?);
                        let sig: String = row.get(1)?;
                        let path: String = row.get(2)?;
                        let lang: String = row.get(3)?;
                        Ok((id, sig, path, lang))
                    })
                    .context("executing fn_nodes_by_leaf_name")?;
                for row in rows {
                    out.push(row.context("decoding fn_nodes_by_leaf_name row")?);
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return every import node as `(id, signature, language, path)`.
    /// `path` is the importing file (the import node's own VName path), which
    /// the query-time `ImportResolver`s use to anchor relative imports.
    /// Used by `get_blast_radius` (Phase 2) to build an in-memory index once per
    /// call, so transitive resolution never issues a per-file FTS lookup (#613).
    pub fn import_nodes_lite(&self) -> Result<Vec<(NodeId, String, String, String)>, StoreError> {
        (|| -> AnyResult<Vec<(NodeId, String, String, String)>> {
            let mut stmt = self
                .conn
                .prepare("SELECT id, signature, language, path FROM nodes WHERE kind = 'import'")
                .context("preparing import_nodes_lite query")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        i64_to_node_id(row.get::<_, i64>(0)?),
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .context("executing import_nodes_lite query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding import_nodes_lite row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return all (src_path, dst_path) pairs via the two-hop import chain:
    /// any_node --[depends]--> import_node --[resolves-to]--> file.
    /// src_path is the path of the node making the import (any kind: file, function, etc.).
    /// dst_path is the path of the resolved target file.
    /// Used by the graph-overview aggregation to compute cross-package edges.
    pub fn file_import_pairs(&self) -> Result<Vec<(String, String)>, StoreError> {
        (|| -> AnyResult<Vec<(String, String)>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT n1.path, n2.path
                     FROM edges e1
                     JOIN nodes n1 ON n1.id = e1.src
                     JOIN edges e2 ON e2.src = e1.dst AND e2.kind = 'resolves-to'
                     JOIN nodes n2 ON n2.id = e2.dst AND n2.kind = 'file'
                     WHERE e1.kind = 'depends'",
                )
                .context("preparing file_import_pairs query")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .context("executing file_import_pairs query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding file_import_pairs row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return distinct resolved cross-file dependency pairs `(src_path, dst_path)`
    /// meaning "the file at src_path depends on the file at dst_path".
    ///
    /// Language-agnostic by construction — it reads only already-resolved graph
    /// edges and node paths, never import syntax. Three complementary sources:
    ///   - two-hop import chain `x --depends--> import --resolves-to--> target`,
    ///     taking the depends *source* as the importer (Phase A; robust to how a
    ///     language models the import node's own path).
    ///   - direct `resolves-to` (import-node path = importer file, #613). Phase A.
    ///   - direct `ref/call` (caller symbol -> callee symbol). Phase B call graph.
    ///
    /// Both endpoints must carry a real path and differ. Used by the repo-map and
    /// graph-overview aggregations to build a directory-level dependency graph
    /// without any per-language logic at query time.
    pub fn resolved_dep_pairs(&self) -> Result<Vec<(String, String)>, StoreError> {
        (|| -> AnyResult<Vec<(String, String)>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT DISTINCT src, dst FROM (
                         SELECT n1.path AS src, n2.path AS dst
                         FROM edges e1
                         JOIN nodes n1 ON n1.id = e1.src
                         JOIN edges e2 ON e2.src = e1.dst AND e2.kind = 'resolves-to'
                         JOIN nodes n2 ON n2.id = e2.dst
                         WHERE e1.kind = 'depends'
                         UNION ALL
                         SELECT ns.path AS src, nd.path AS dst
                         FROM edges e
                         JOIN nodes ns ON ns.id = e.src
                         JOIN nodes nd ON nd.id = e.dst
                         WHERE e.kind IN ('resolves-to', 'ref/call')
                     )
                     WHERE src <> '' AND dst <> '' AND src <> dst",
                )
                .context("preparing resolved_dep_pairs query")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .context("executing resolved_dep_pairs query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding resolved_dep_pairs row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// True when at least one `ref/call` edge exists anywhere in the graph — i.e.
    /// Phase B (SCIP/LSIF semantic analysis) has produced call-graph data. Used to
    /// disclose degraded state in the repo map. `false` on query error (safe:
    /// surfaces the "semantic not available" note rather than hiding it).
    pub fn has_any_refcall_edges(&self) -> bool {
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM edges WHERE kind = 'ref/call' LIMIT 1)",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap_or(false)
    }

    pub fn all_edges(&self) -> Result<Vec<(NodeId, NodeId, String, String)>, StoreError> {
        (|| -> AnyResult<Vec<(NodeId, NodeId, String, String)>> {
            let mut stmt = self
                .conn
                .prepare("SELECT src, dst, kind, provenance FROM edges")
                .context("preparing all_edges query")?;
            let rows = stmt
                .query_map([], |row| {
                    let src = i64_to_node_id(row.get::<_, i64>(0)?);
                    let dst = i64_to_node_id(row.get::<_, i64>(1)?);
                    let kind: String = row.get(2)?;
                    let provenance: String = row.get(3)?;
                    Ok((src, dst, kind, provenance))
                })
                .context("executing all_edges query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding all_edges row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return all `(NodeId, kind)` pairs — used by k-core for shell propagation.
    ///
    /// Cheaper than `all_nodes()` because it projects only `id` and `kind`.
    pub fn all_node_ids_and_kinds(&self) -> Result<Vec<(NodeId, String)>, StoreError> {
        (|| -> AnyResult<Vec<(NodeId, String)>> {
            let mut stmt = self
                .conn
                .prepare("SELECT id, kind FROM nodes")
                .context("preparing all_node_ids_and_kinds query")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        i64_to_node_id(row.get::<_, i64>(0)?),
                        row.get::<_, String>(1)?,
                    ))
                })
                .context("executing all_node_ids_and_kinds query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding all_node_ids_and_kinds row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Batch-write shell numbers produced by k-core decomposition.
    ///
    /// Uses a single prepared statement executed inside one transaction for speed.
    /// O(|shells|) writes — safe to call after every index pass.
    /// Set `embed_text = NULL` for all nodes.
    ///
    /// Called before regenerating embed_text with a new richness tier (model switch).
    pub fn clear_all_embed_texts(&mut self) -> Result<(), StoreError> {
        self.conn
            .execute_batch("UPDATE nodes SET embed_text = NULL")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-022 (reranker-doc prototype): fetch stored `embed_text` for a set of
    /// nodes, skipping rows whose `embed_text` is NULL or empty. Returns
    /// `id -> embed_text`. Used to surface a node's doc/skeleton prose to the
    /// cross-encoder, which otherwise sees a docblock-stripped code body only.
    pub fn get_embed_texts(
        &self,
        ids: &[NodeId],
    ) -> Result<std::collections::HashMap<NodeId, String>, StoreError> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        (|| -> AnyResult<std::collections::HashMap<NodeId, String>> {
            let mut stmt = self
                .conn
                .prepare("SELECT embed_text FROM nodes WHERE id = ?1")
                .context("preparing get_embed_texts stmt")?;
            let mut out = std::collections::HashMap::with_capacity(ids.len());
            for &id in ids {
                let text: Option<String> = stmt
                    .query_row(params![node_id_to_i64(id)], |r| {
                        r.get::<_, Option<String>>(0)
                    })
                    .optional()
                    .context("reading embed_text")?
                    .flatten();
                if let Some(t) = text {
                    if !t.is_empty() {
                        out.insert(id, t);
                    }
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Batch-update the `embed_text` column for a set of nodes.
    ///
    /// Called by the daemon after Phase A indexing to store pre-computed
    /// AST-skeleton text so the embed sidecar stays a pure embedding machine.
    pub fn write_embed_texts_batch(
        &mut self,
        pairs: &[(NodeId, String)],
    ) -> Result<(), StoreError> {
        if pairs.is_empty() {
            return Ok(());
        }
        (|| -> AnyResult<()> {
            let tx = self
                .conn
                .transaction()
                .context("begin write_embed_texts_batch tx")?;
            {
                let mut stmt = tx
                    .prepare("UPDATE nodes SET embed_text = ?2 WHERE id = ?1")
                    .context("preparing write_embed_texts_batch stmt")?;
                for (id, text) in pairs {
                    stmt.execute(params![node_id_to_i64(*id), text])
                        .context("executing write_embed_texts_batch update")?;
                }
            }
            // RFC-022 D1 (RC-1): fold the freshly-written `embed_text` into each
            // node's FTS content so a conceptual query can match the doc/skeleton's
            // natural-language terms (the compound symbol the bare signature hides).
            // Rebuilds the FTS row in the same tx (retract-old + insert-widened),
            // preserving the contentless-FTS5 + vocab invariants. Keeps the
            // always-fresh guarantee: the FTS index tracks embed_text as it lands.
            // Gated OFF by default (pending Phase-4 calibration); when disabled the
            // FTS row is left signature-only (byte-for-byte the pre-D1 behaviour).
            if Self::fts_embed_widen_enabled() {
                let mut sel = tx
                    .prepare("SELECT path, language, signature, kind FROM nodes WHERE id = ?1")
                    .context("preparing embed-fts node lookup")?;
                for (id, text) in pairs {
                    let parts: Option<(String, String, String, String)> = sel
                        .query_row(params![node_id_to_i64(*id)], |row| {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                        })
                        .optional()
                        .context("looking up node parts for embed-fts")?;
                    if let Some((path, language, signature, kind)) = parts {
                        let toks = Self::fts_tokens_with_embed_from_parts(
                            &signature,
                            &path,
                            &kind,
                            &language,
                            Some(text),
                        );
                        Self::put_node_fts_tokens(&tx, *id, &toks)?;
                    }
                }
            }
            tx.commit().context("committing write_embed_texts_batch tx")
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-022 D1 (RC-1): reconcile the FTS content of every already-`embed_text`-
    /// populated node with the current [`fts_embed_widen_enabled`] setting. When the
    /// widening is ON, rebuilds each node's `nodes_fts` row to fold in the doc/
    /// skeleton tokens; when OFF, rebuilds them signature-only (the downgrade path,
    /// so toggling the flag back off fully restores the pre-D1 index). Pure FTS work
    /// — never recomputes embeddings or skeletons. A `fts_embed_text_version` meta
    /// key records the applied state (`"1"` = widened, `"0"` = signature-only) so the
    /// reconcile runs only when the state actually changes (idempotent,
    /// always-fresh-preserving). Returns the number of nodes rebuilt; 0 when current.
    pub fn backfill_fts_embed_text(&mut self) -> Result<usize, StoreError> {
        let widen = Self::fts_embed_widen_enabled();
        let target_version = if widen { "1" } else { "0" };
        (|| -> AnyResult<usize> {
            let current: Option<String> = self
                .conn
                .query_row(
                    "SELECT value FROM meta WHERE key = 'fts_embed_text_version'",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .context("reading fts_embed_text_version")?;
            // No-op when already in the desired state. Treat a never-set version as
            // signature-only ("0"): if the widening is off and was never applied,
            // there is nothing to reconcile.
            let effective = current.as_deref().unwrap_or("0");
            if effective == target_version {
                return Ok(0);
            }
            let rows: Vec<(i64, String, String, String, String, String)> = {
                let mut stmt = self
                    .conn
                    .prepare(
                        "SELECT id, path, language, signature, kind, embed_text \
                         FROM nodes WHERE embed_text IS NOT NULL AND embed_text != ''",
                    )
                    .context("preparing backfill_fts_embed_text scan")?;
                let mapped = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, String>(5)?,
                        ))
                    })
                    .context("scanning nodes for embed-fts backfill")?;
                mapped.collect::<Result<Vec<_>, _>>()?
            };
            let tx = self
                .conn
                .transaction()
                .context("begin backfill_fts_embed_text tx")?;
            for (id_i64, path, language, signature, kind, embed_text) in &rows {
                // widen → fold in embed_text; downgrade → signature-only (None).
                let embed = if widen {
                    Some(embed_text.as_str())
                } else {
                    None
                };
                let toks =
                    Self::fts_tokens_with_embed_from_parts(signature, path, kind, language, embed);
                Self::put_node_fts_tokens(&tx, i64_to_node_id(*id_i64), &toks)?;
            }
            tx.execute(
                "INSERT INTO meta(key, value) VALUES('fts_embed_text_version', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![target_version],
            )
            .context("recording fts_embed_text_version")?;
            tx.commit()
                .context("committing backfill_fts_embed_text tx")?;
            Ok(rows.len())
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Return all nodes where `embed_text IS NULL` and the kind is embeddable
    /// (i.e. excludes file/import/module/variable structural noise).
    ///
    /// Called by the daemon after Phase A to find nodes that need embed_text
    /// populated via `skeleton_for_node`.
    pub fn nodes_missing_embed_text(&self) -> Result<Vec<Node>, StoreError> {
        (|| -> AnyResult<Vec<Node>> {
            let mut stmt = self.conn.prepare(
                "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line \
                 FROM nodes \
                 WHERE embed_text IS NULL \
                   AND kind NOT IN ('file-module', 'import', 'module', 'variable')",
            ).context("preparing nodes_missing_embed_text query")?;
            let rows = stmt
                .query_map([], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing nodes_missing_embed_text query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding nodes_missing_embed_text row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn write_shell_numbers(&mut self, shells: &[(NodeId, u32)]) -> Result<(), StoreError> {
        if shells.is_empty() {
            return Ok(());
        }
        (|| -> AnyResult<()> {
            let tx = self
                .conn
                .transaction()
                .context("begin write_shell_numbers tx")?;
            {
                let mut stmt = tx
                    .prepare("UPDATE nodes SET shell_number = ?2 WHERE id = ?1")
                    .context("preparing write_shell_numbers stmt")?;
                for &(id, shell) in shells {
                    stmt.execute(params![node_id_to_i64(id), shell as i64])
                        .context("executing write_shell_numbers update")?;
                }
            }
            tx.commit().context("committing write_shell_numbers tx")
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Batch-fetch shell numbers for a set of node IDs.
    ///
    /// Unknown IDs are silently skipped (shell defaults to 0 at the call site).
    /// Splits into 500-ID chunks to stay under SQLite's variable limit (SQLITE_MAX_VARIABLE_NUMBER).
    pub fn get_shell_numbers_batch(
        &self,
        ids: &[NodeId],
    ) -> Result<std::collections::HashMap<NodeId, u32>, StoreError> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        (|| -> AnyResult<std::collections::HashMap<NodeId, u32>> {
            let mut out = std::collections::HashMap::with_capacity(ids.len());
            for chunk in ids.chunks(500) {
                let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql =
                    format!("SELECT id, shell_number FROM nodes WHERE id IN ({placeholders})");
                let mut stmt = self
                    .conn
                    .prepare(&sql)
                    .context("preparing get_shell_numbers_batch")?;
                let id_params: Vec<i64> = chunk.iter().map(|id| node_id_to_i64(*id)).collect();
                let rows = stmt
                    .query_map(params_from_iter(id_params.iter()), |row| {
                        let id = i64_to_node_id(row.get::<_, i64>(0)?);
                        let shell: i64 = row.get(1)?;
                        Ok((id, shell as u32))
                    })
                    .context("executing get_shell_numbers_batch")?;
                for row in rows {
                    let (id, shell) = row.context("decoding get_shell_numbers_batch row")?;
                    out.insert(id, shell);
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>, StoreError> {
        self.conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("reading meta key")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    pub fn set_meta(&mut self, key: &str, value: &str) -> Result<(), StoreError> {
        self.conn
            .execute(
                "INSERT INTO meta(key, value) VALUES(?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
            .context("writing meta key")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    /// Return the VName signature format version recorded in this database.
    ///
    /// Returns `0` for legacy databases (pre-RFC-002) that have no such row,
    /// or databases migrated from schema v2 where the row was just added with
    /// the default value `'0'`.
    pub fn get_signature_format_version(&self) -> Result<u8, StoreError> {
        (|| -> AnyResult<u8> {
            let raw = self
                .get_meta("signature_format_version")
                .context("reading signature_format_version")?
                .unwrap_or_else(|| "0".to_string());
            raw.parse::<u8>()
                .with_context(|| format!("invalid signature_format_version in meta: {raw}"))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Write the VName signature format version. Called by the daemon after a
    /// successful full re-index to stamp the active format version.
    pub fn set_signature_format_version(&mut self, v: u8) -> Result<(), StoreError> {
        self.set_meta("signature_format_version", &v.to_string())
            .context("writing signature_format_version")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Persist an edge with LSIF (semantic) provenance.
    ///
    /// LSIF always wins: if an identical (src, dst, kind) row already exists
    /// with `provenance='tree-sitter'`, it is upgraded to `'lsif'`. This
    /// implements the LSIF-beats-tree-sitter precedence policy (ADR-002).
    pub fn put_edge_lsif(&mut self, edge: &Edge) -> Result<(), StoreError> {
        self.conn
            .execute(
                "INSERT INTO edges(src, dst, kind, provenance, confidence) VALUES(?1, ?2, ?3, 'lsif', ?4)
                 ON CONFLICT(src, dst, kind) DO UPDATE SET provenance = 'lsif', confidence = excluded.confidence",
                params![
                    node_id_to_i64(edge.src),
                    node_id_to_i64(edge.dst),
                    edge.kind.as_str(),
                    edge.confidence.map(|c| c as i64),
                ],
            )
            .context("inserting lsif edge")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    /// Persist an edge resolved by the RFC-027 live lane (`provenance='live'`).
    ///
    /// Live edges are an ephemeral overlay over the commit-gated semantic graph:
    /// Tree-sitter detected the reference, the live engine resolved it precisely
    /// (unambiguous-lexical or LSP-disambiguated), and the next Phase B run
    /// ratifies it.
    ///
    /// **The overlay is purely additive: this never relabels an edge that
    /// already exists.** That is what makes retiring it safe. The sweep in
    /// [`sweep_live_edges_for_languages`] deletes rows, so anything it can
    /// reach must be something the live lane itself created — otherwise
    /// retiring the overlay would destroy pre-existing truth rather than
    /// returning the graph to what it was.
    ///
    /// The hazard is concrete, not theoretical. An interface edit re-resolves
    /// the files that reference the edited one (section 6.3), and those files
    /// were *not* re-parsed, so their `tree-sitter` edges are still in place. An
    /// upsert that relabelled them `live` would hand them to the sweep, and any
    /// one Phase B did not happen to re-derive would be deleted outright. The
    /// convergence property test is what surfaced this.
    ///
    /// So the `ON CONFLICT` only refreshes a row that is *already* `live`, which
    /// keeps re-emission idempotent — `reindex_replace` deletes every outbound
    /// edge of a file on each save, so the engine re-emits whole-file rather
    /// than only for references it believes are new. Every other provenance is
    /// left exactly as it was.
    ///
    /// This never mints identity (RFC-027 section 8.2): both endpoints must
    /// already exist as nodes, so the fencing rule and VName uniqueness hold.
    pub fn put_edge_live(&mut self, edge: &Edge) -> Result<(), StoreError> {
        self.conn
            .execute(
                "INSERT INTO edges(src, dst, kind, provenance, confidence) VALUES(?1, ?2, ?3, 'live', ?4) \
                 ON CONFLICT(src, dst, kind) DO UPDATE SET \
                   confidence = excluded.confidence \
                 WHERE edges.provenance = 'live'",
                params![
                    node_id_to_i64(edge.src),
                    node_id_to_i64(edge.dst),
                    edge.kind.as_str(),
                    edge.confidence.map(|c| c as i64),
                ],
            )
            .context("inserting live edge")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    /// PR #715: batch-check which of `ids` already exist in `nodes`, one query per
    /// chunk rather than a `SELECT 1` per id, for the Phase B half-edge guard.
    ///
    /// Returns `(present, prefetch_ok)`: the subset of `ids` found in the store,
    /// and whether every chunk query succeeded. On any query error the caller
    /// treats all endpoints as present (fail-open) so a transient read never drops
    /// a legitimate edge — the same contract the previous per-row `unwrap_or(true)`
    /// held. Chunked under SQLite's default 999 bound-variable ceiling.
    fn prefetch_existing_node_ids(
        conn: &rusqlite::Connection,
        ids: &std::collections::HashSet<i64>,
    ) -> (std::collections::HashSet<i64>, bool) {
        let mut present = std::collections::HashSet::with_capacity(ids.len());
        if ids.is_empty() {
            return (present, true);
        }
        const CHUNK: usize = 900;
        let all: Vec<i64> = ids.iter().copied().collect();
        for chunk in all.chunks(CHUNK) {
            let placeholders = std::iter::repeat("?")
                .take(chunk.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("SELECT id FROM nodes WHERE id IN ({placeholders})");
            let found = (|| -> anyhow::Result<Vec<i64>> {
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(chunk.iter()), |r| {
                        r.get::<_, i64>(0)
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })();
            match found {
                Ok(rows) => present.extend(rows),
                // Fail open: a query error means "cannot prove absence", so the
                // caller keeps every edge rather than dropping a real one.
                Err(_) => return (present, false),
            }
        }
        (present, true)
    }

    /// Write Phase B (SCIP/LSIF semantic) nodes and edges in a single transaction.
    ///
    /// Semantics match repeated `put_node` + `put_edge_lsif` calls but in one
    /// round-trip instead of O(N) transactions. Call this after `invoke_phase_b_all`
    /// returns; the caller must not hold an open transaction.
    pub fn write_phase_b_batch(
        &mut self,
        nodes: &[travsr_core::Node],
        edges: &[travsr_core::Edge],
        provenance: &str,
    ) -> anyhow::Result<()> {
        if nodes.is_empty() && edges.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("write_phase_b_batch: begin")?;
        // #479: Phase B nodes carry no `test_role` (it is Phase-A/AST-derived), so
        // this INSERT deliberately omits the column — a fresh Phase-B-only node
        // takes the schema default (`None`) and the ON CONFLICT leaves any
        // existing Phase-A role intact rather than clobbering it to `None`.
        for node in nodes {
            let id_i64 = node_id_to_i64(node.id);
            tx.execute(
                "INSERT INTO nodes(id, corpus, root, path, language, signature, kind, package, line, end_line, is_noise) \
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) \
                 ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, \
                 package = excluded.package, \
                 line = COALESCE(excluded.line, nodes.line), \
                 end_line = COALESCE(excluded.end_line, nodes.end_line), \
                 is_noise = excluded.is_noise",
                params![
                    id_i64,
                    node.vname.corpus,
                    node.vname.root,
                    node.vname.path,
                    node.vname.language,
                    node.vname.signature,
                    node.kind,
                    node.package,
                    node.line.map(|l| l as i64),
                    node.end_line.map(|l| l as i64),
                    travsr_core::noise::is_structural_noise(node),
                ],
            )
            .context("write_phase_b_batch: insert node")?;
            Self::put_node_fts(&tx, node).context("write_phase_b_batch: put_node_fts")?;
            Self::put_node_fts_words(&tx, node)
                .context("write_phase_b_batch: put_node_fts_words")?;
        }
        // #712: never write a half-edge. An incomplete or partial sidecar result
        // can reference a node it never emitted (or one that a crashed language
        // owns); inserting an edge to a non-existent endpoint creates an orphan
        // that `travsr fsck` later flags. Validate both endpoints against this
        // batch's freshly written nodes and the existing store, dropping any edge
        // that would dangle. Fail open on a query error so a transient read
        // problem never silently discards a legitimate edge.
        {
            let batch_ids: std::collections::HashSet<i64> =
                nodes.iter().map(|n| node_id_to_i64(n.id)).collect();
            // PR #715 perf: validate endpoints against the existing store with a
            // single batched prefetch (one query per ~900 ids) instead of a
            // `SELECT 1` per edge inside the write transaction — Phase B batches on
            // a large repo make the per-row round trip the dominant cost. Only
            // endpoints NOT already in this batch need a lookup. The half-edge
            // guard and its fail-open contract are unchanged: `prefetch_ok == false`
            // (a query error) treats every endpoint as present, so a transient read
            // problem never silently discards a legitimate edge.
            let mut needed: std::collections::HashSet<i64> = std::collections::HashSet::new();
            for edge in edges {
                let src = node_id_to_i64(edge.src);
                let dst = node_id_to_i64(edge.dst);
                if !batch_ids.contains(&src) {
                    needed.insert(src);
                }
                if !batch_ids.contains(&dst) {
                    needed.insert(dst);
                }
            }
            let (present, prefetch_ok) = Self::prefetch_existing_node_ids(&tx, &needed);
            let endpoint_present = |id: i64| -> bool {
                batch_ids.contains(&id) || !prefetch_ok || present.contains(&id)
            };
            let mut dropped = 0usize;
            for edge in edges {
                let src = node_id_to_i64(edge.src);
                let dst = node_id_to_i64(edge.dst);
                if !endpoint_present(src) || !endpoint_present(dst) {
                    dropped += 1;
                    continue;
                }
                // E1: label edges with their true provenance (SCIP relationships
                // vs native tree-sitter leaf-name resolution) instead of
                // hardcoding 'lsif'. Precedence-preserving: a compiler provenance
                // ('lsif'/'scip') already on the row is never demoted by a later
                // write (ADR-002), so a heuristic 'tree-sitter' write cannot
                // overwrite it.
                //
                // RFC-027: the ELSE arm demoting a 'live' row to 'tree-sitter'
                // is deliberate, not a leak. Both callers that pass a
                // 'tree-sitter' provenance (`init_repo_with_progress` and
                // `run_background_phase_b_inner`) are Phase B runs, so reaching
                // here means Phase B just re-derived this edge and it is no
                // longer a live guess. Demotion IS ratification for a
                // co-located edge, and it is what leaves the section 8.3 sweep
                // holding only the live edges Phase B did not re-derive.
                tx.execute(
                    "INSERT INTO edges(src, dst, kind, provenance, confidence) \
                     VALUES(?1, ?2, ?3, ?4, ?5) \
                     ON CONFLICT(src, dst, kind) DO UPDATE SET \
                     provenance = CASE \
                       WHEN excluded.provenance IN ('lsif','scip') THEN excluded.provenance \
                       WHEN edges.provenance IN ('lsif','scip') THEN edges.provenance \
                       ELSE excluded.provenance END, \
                     confidence = excluded.confidence",
                    params![
                        src,
                        dst,
                        edge.kind.as_str(),
                        provenance,
                        edge.confidence.map(|c| c as i64),
                    ],
                )
                .context("write_phase_b_batch: insert edge")?;
            }
            if dropped > 0 {
                tracing::warn!(
                    dropped,
                    "write_phase_b_batch: skipped edges with a missing endpoint (incomplete sidecar result)"
                );
            }
        }
        tx.commit().context("write_phase_b_batch: commit")
    }

    /// E1: reconcile every edge's `language` label to its endpoints' language.
    ///
    /// Edge language is a derived label, not authored data. Edge INSERTs across
    /// the store omit it and rely on the schema default (`'typescript'`), which
    /// mislabels every edge in a non-TypeScript graph. This single idempotent
    /// pass sets each edge's language from its src node (falling back to dst,
    /// then `'unknown'` for a dangling endpoint). Run at the end of indexing
    /// (full and incremental); it is O(edges) and label-only (no edge is added
    /// or removed). Returns the number of rows updated.
    pub fn reconcile_edge_languages(&mut self) -> Result<usize, StoreError> {
        self.conn
            .execute(
                "UPDATE edges SET language = COALESCE( \
                   (SELECT n.language FROM nodes n WHERE n.id = edges.src), \
                   (SELECT n.language FROM nodes n WHERE n.id = edges.dst), \
                   'unknown')",
                [],
            )
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// G2 attribution-aware Phase B write.
    ///
    /// Writes SCIP definition nodes, then for each [`travsr_core::ScipRef`]
    /// finds the innermost function/method node whose span contains
    /// `caller_line` and emits a `ref/call` edge from it (falls back to the
    /// file node if no enclosing function is found).  Also records a row in
    /// `edge_sites` for every attribution (O4 evidence).
    ///
    /// Must be called **after** the Phase A tree-sitter pass so that
    /// `end_line` spans are already in the DB.
    pub fn write_scip_attributed_batch(
        &mut self,
        corpus: &str,
        nodes: &[travsr_core::Node],
        refs: &[travsr_core::ScipRef],
    ) -> anyhow::Result<()> {
        if nodes.is_empty() && refs.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("write_scip_attributed_batch: begin")?;

        // #479: see write_phase_b_batch — this Phase-B INSERT omits `test_role`
        // on purpose so it never clobbers a Phase-A role to `None`.
        for node in nodes {
            let id_i64 = node_id_to_i64(node.id);
            tx.execute(
                "INSERT INTO nodes(id, corpus, root, path, language, signature, kind, package, line, end_line, is_noise) \
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) \
                 ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, \
                 package = excluded.package, \
                 line = COALESCE(excluded.line, nodes.line), \
                 end_line = COALESCE(excluded.end_line, nodes.end_line), \
                 is_noise = excluded.is_noise",
                params![
                    id_i64,
                    node.vname.corpus,
                    node.vname.root,
                    node.vname.path,
                    node.vname.language,
                    node.vname.signature,
                    node.kind,
                    node.package,
                    node.line.map(|l| l as i64),
                    node.end_line.map(|l| l as i64),
                    travsr_core::noise::is_structural_noise(node),
                ],
            )
            .context("write_scip_attributed_batch: insert node")?;
            Self::put_node_fts(&tx, node).context("write_scip_attributed_batch: put_node_fts")?;
            Self::put_node_fts_words(&tx, node)
                .context("write_scip_attributed_batch: put_node_fts_words")?;
        }

        // P4: build span cache — one SELECT per unique caller_path instead of one per ref.
        // 10k refs / 200 files: 10,000 queries → 200 queries.
        let unique_paths: std::collections::HashSet<&str> =
            refs.iter().map(|r| r.caller_path.as_str()).collect();
        let mut span_cache: std::collections::HashMap<&str, Vec<FnSpan>> =
            std::collections::HashMap::with_capacity(unique_paths.len());
        for path in unique_paths {
            span_cache.insert(path, fetch_all_fn_spans(&tx, corpus, path)?);
        }

        // #650: count `src == dst` refs dropped by the guard below, so the
        // positional-collapse rate is observable rather than silent.
        let mut self_loops_rejected: u64 = 0;

        for scip_ref in refs {
            // G2: find the innermost enclosing function/method span in memory.
            let occ_line = scip_ref.caller_line as i64;
            let src_id: Option<i64> = span_cache
                .get(scip_ref.caller_path.as_str())
                .and_then(|spans| find_narrowest_enclosing(spans, occ_line));

            let caller_id = match src_id {
                Some(id) => id,
                None => file_node_for_attribution(&tx, corpus, &scip_ref.caller_path)?,
            };

            let callee_id = node_id_to_i64(scip_ref.callee_id);

            // #650: reject self-referential `ref/call` edges. This is the single
            // write choke point every language's ScipRef producer converges on
            // (rust rust-analyzer LSIF, Dart native, SCIP-protobuf sidecars), and
            // was the only edge-emission path in the pipeline missing the
            // `src == dst` guard that the UnresolvedCall path (daemon: `u.src !=
            // dst`) and the alias-remap paths already enforce. A self-loop arises
            // from positional collapse — the occurrence line and the callee's
            // definition line resolve to the *same* enclosing-function node — and
            // carries zero reachability. Recursion is not represented as a
            // self-edge anywhere in the graph (the tree-sitter path emits zero on
            // the same corpus), so dropping these here keeps the semantic
            // `ref/call` graph consistent with the structural one.
            if caller_id == callee_id {
                self_loops_rejected += 1;
                continue;
            }

            // E3 (W3a) — fail closed (Invariant #4): never write a `ref/call`
            // edge whose callee resolves to no node. The full node table is
            // present here (Phase A nodes already persisted, this batch's
            // `pb_nodes` inserted above in the same tx), so a missing callee is a
            // genuine unresolved symbol (external/stdlib, or a positional lookup
            // that found nothing) — not a not-yet-written node. A dangling
            // `ref/call` pollutes `get_callers`/blast-radius, so drop it here
            // rather than persist it. This gate is what lets the positional
            // rust-analyzer LSIF path (W3b) and any external-symbol reference
            // fail closed instead of leaving the 34% dangling this plan removes.
            // Fetch the callee's kind (not just existence): #757 records a
            // reference onto a `field` node under `ref/field` instead of
            // `ref/call`, matching the native Rust field-ref path, so a field
            // read never appears as a caller in `get_callers`/blast-radius.
            let callee_kind: Option<String> = tx
                .query_row(
                    "SELECT kind FROM nodes WHERE id = ?1",
                    params![callee_id],
                    |row| row.get(0),
                )
                .optional()
                .context("write_scip_attributed_batch: callee existence check")?;
            let Some(callee_kind) = callee_kind else {
                // E3 (W3a): callee resolves to no node — a genuine unresolved
                // symbol (external/stdlib, or a positional miss). Fail closed:
                // a dangling ref would pollute get_callers/blast-radius.
                continue;
            };
            let is_field = callee_kind == "field";

            // #650/#757: only genuine calls create a call-graph `ref/call` edge, so
            // `get_callers` / blast-radius / PageRank stay a call graph and never
            // gain a `src == dst` self-loop or a spurious non-call edge. A field
            // read is not a call: emit a `ref/field` edge (also excluded from the
            // call graph) so it is symmetric with the native Rust path. Other
            // non-call references (type annotations, `self`/`Self`, path segments)
            // still record only their occurrence below so `find_references`
            // enumerates every use site (issue #299). `is_call` defaults to true
            // for call-scoped producers (native tree-sitter, bundled emitters).
            if is_field {
                tx.execute(
                    "INSERT INTO edges(src, dst, kind, provenance, confidence) \
                     VALUES(?1, ?2, 'ref/field', 'scip', NULL) \
                     ON CONFLICT(src, dst, kind) DO UPDATE SET provenance = 'scip'",
                    params![caller_id, callee_id],
                )
                .context("write_scip_attributed_batch: insert field edge")?;
            } else if scip_ref.is_call {
                tx.execute(
                    "INSERT INTO edges(src, dst, kind, provenance, confidence) \
                     VALUES(?1, ?2, 'ref/call', 'scip', NULL) \
                     ON CONFLICT(src, dst, kind) DO UPDATE SET provenance = 'scip'",
                    params![caller_id, callee_id],
                )
                .context("write_scip_attributed_batch: insert edge")?;
            }

            // O4: record the occurrence line for every reference (call, field, or
            // other non-call) so `find_references` covers all use sites. Field
            // occurrences go under `ref/field`; everything else under `ref/call`.
            let site_kind = if is_field { "ref/field" } else { "ref/call" };
            // RFC-027 #813 P2: carry the occurrence byte column when the source
            // supplied one, filling a NULL from a later real position and never
            // duplicating the row (col is not in the PK).
            tx.execute(
                "INSERT INTO edge_sites(src, dst, kind, line, col) VALUES(?1, ?2, ?3, ?4, ?5) \
                 ON CONFLICT(src, dst, kind, line) DO UPDATE SET col = COALESCE(edge_sites.col, excluded.col)",
                params![
                    caller_id,
                    callee_id,
                    site_kind,
                    occ_line,
                    scip_ref.caller_col.map(|c| c as i64)
                ],
            )
            .context("write_scip_attributed_batch: insert edge_site")?;
        }

        if self_loops_rejected > 0 {
            tracing::debug!(
                self_loops_rejected,
                total_refs = refs.len(),
                "write_scip_attributed_batch: dropped self-referential ref/call edges (#650)"
            );
        }

        tx.commit().context("write_scip_attributed_batch: commit")
    }

    /// Count `ref/call` edges whose `src == dst` (#650). Zero in correct
    /// operation once [`Self::write_scip_attributed_batch`] guards them at write
    /// time; a non-zero count means the DB predates that guard (or a producer
    /// bypassed the choke point), so `fsck` surfaces and `--fix` sweeps them.
    pub fn count_self_ref_call_edges(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM edges WHERE kind = 'ref/call' AND src = dst",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting self-referential ref/call edges")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Delete every self-referential (`src == dst`) `ref/call` edge and its
    /// occurrence sites (#650). Returns the number of edges removed. Both the
    /// edge and its `edge_sites` rows are dropped in one transaction so the
    /// occurrence store cannot outlive the edge it described.
    pub fn sweep_self_ref_call_edges(&mut self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let tx = self
                .conn
                .transaction()
                .context("begin self-loop sweep transaction")?;
            tx.execute(
                "DELETE FROM edge_sites WHERE kind = 'ref/call' AND src = dst",
                [],
            )
            .context("sweeping self-referential edge_sites")?;
            let edges = tx
                .execute(
                    "DELETE FROM edges WHERE kind = 'ref/call' AND src = dst",
                    [],
                )
                .context("sweeping self-referential ref/call edges")?;
            tx.commit().context("commit self-loop sweep")?;
            Ok(edges as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Resolve rust-analyzer LSIF positional references (E3 W3b) into `ScipRef`s.
    ///
    /// rust-analyzer LSIF identifies a callee only by its definition location,
    /// not a `travsr_vname`. This maps each `callee_def_(path,line)` to the
    /// **narrowest Phase A node whose span contains that line** — the same
    /// positional rule `write_scip_attributed_batch` uses for the caller — and
    /// emits a `ScipRef` whose `callee_id` is that real node. **Fails closed**:
    /// a reference whose callee def resolves to no node (external symbol, or a
    /// def line outside every span) is dropped, so no dangling edge is produced.
    ///
    /// Read-only. Caller attribution (`caller_line` → enclosing function) is left
    /// to `write_scip_attributed_batch`, through which the returned refs flow.
    ///
    /// O(unique callee paths) span queries + O(refs) span scans.
    pub fn resolve_lsif_positional_refs(
        &self,
        corpus: &str,
        positional: &[travsr_core::LsifPositionalRef],
    ) -> anyhow::Result<Vec<travsr_core::ScipRef>> {
        if positional.is_empty() {
            return Ok(Vec::new());
        }
        // One span query per unique callee-definition path. Any node kind is a
        // valid callee (method/fn/struct/enum/const/…); `.rs` files carry no
        // span-bearing noise nodes (doc-chunks are markdown-only), so no kind
        // filter is needed — the narrowest containing span is the symbol.
        let unique_paths: std::collections::HashSet<&str> = positional
            .iter()
            .map(|p| p.callee_def_path.as_str())
            .collect();
        let mut span_cache: std::collections::HashMap<&str, Vec<FnSpan>> =
            std::collections::HashMap::with_capacity(unique_paths.len());
        for path in unique_paths {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "SELECT id, line, end_line FROM nodes \
                     WHERE corpus = ?1 AND path = ?2 \
                       AND line IS NOT NULL AND end_line IS NOT NULL \
                     ORDER BY (end_line - line) ASC, id ASC",
                )
                .context("resolve_lsif_positional_refs: prepare")?;
            let spans = stmt
                .query_map(params![corpus, path], |row| {
                    Ok(FnSpan {
                        id: row.get(0)?,
                        line: row.get(1)?,
                        end_line: row.get(2)?,
                    })
                })
                .context("resolve_lsif_positional_refs: query")?
                .collect::<rusqlite::Result<Vec<_>>>()
                .context("resolve_lsif_positional_refs: collect")?;
            span_cache.insert(path, spans);
        }

        let mut out = Vec::with_capacity(positional.len());
        for p in positional {
            let def_line = p.callee_def_line as i64;
            let Some(callee_i64) = span_cache
                .get(p.callee_def_path.as_str())
                .and_then(|spans| find_narrowest_enclosing(spans, def_line))
            else {
                continue; // fail closed: callee def resolves to no node
            };
            out.push(travsr_core::ScipRef {
                caller_path: p.caller_path.clone(),
                caller_line: p.caller_line,
                callee_id: i64_to_node_id(callee_i64),
                is_call: p.is_call,
                // RFC-027 #813 P2: the positional LSIF ref already carries the
                // occurrence byte column; pass it through unchanged.
                caller_col: p.caller_col,
            });
        }
        Ok(out)
    }

    /// Occurrence sites (`path:line`) of every `ref/call` to `dst`, read from the
    /// `edge_sites` occurrence store (issue #299 `find_references`).
    ///
    /// `edge_sites.src` is the enclosing-function (or file-fallback) node, which
    /// by construction lives in the **same file** as the occurrence
    /// (`find_narrowest_enclosing` only searches the reference file's spans, and
    /// the fallback is that file's node). Therefore `nodes.path[src]` is exactly
    /// the occurrence file. Rows are deduplicated by the `edge_sites` PK
    /// `(src, dst, kind, line)` and returned in deterministic `(path, line)`
    /// order for stable snapshots.
    ///
    /// An empty vec means no occurrence rows exist for `dst`; callers may then
    /// fall back to structural `ref/call` edge enumeration (as `get_callers`
    /// does) so a language not yet feeding `edge_sites` still degrades gracefully.
    ///
    /// O(k log k) where k = occurrence count (SQLite index scan + sort).
    pub fn reference_sites(&self, dst: NodeId) -> anyhow::Result<Vec<travsr_core::RefSite>> {
        let _span = tracing::debug_span!("store.reference_sites", dst = dst.0).entered();
        let mut stmt = self
            .conn
            .prepare(
                // DISTINCT: two different enclosing `src` nodes can reference
                // `dst` on the same `path:line` (e.g. overlapping macro spans),
                // which the `(src, dst, kind, line)` PK does not dedup. Collapse
                // to unique occurrence locations before returning.
                // #757: field-access use-sites are recorded with kind
                // 'ref/field' (a field read is not a call), so a field node's
                // references live under that kind. Include both so
                // `find_references <field>` returns real occurrences while
                // `get_callers` (which reads the ref/call *edge* set, not
                // edge_sites) stays free of field reads.
                // The occurrence row itself carries no provenance - only
                // `edges` does - so LEFT JOIN it back to recover how the edge
                // behind each site was resolved. LEFT, not inner: a resolved
                // SCIP occurrence that is not a call (a type reference, #650)
                // has no edge row at all and must still be returned.
                // `MAX(...)` over the group keeps the old `DISTINCT (path,
                // line)` row set exactly while answering "was ANY contributing
                // edge name-matched", so two edges landing on one line cannot
                // hide a heuristic one behind a resolved one.
                //
                // The two disjuncts mirror `travsr-mcp::provenance_marker`
                // exactly, so `find_references` and `get_callers` cannot
                // describe the same edge differently: 'live' is an un-ratified
                // overlay edge whatever its kind, and 'tree-sitter' is a name
                // guess only for a call ('tree-sitter' is also the ordinary
                // provenance of a structural field reference, which is why the
                // second disjunct stays restricted to 'ref/call').
                //
                // Coverage limit, measured on this repo's own index
                // (16158 nodes): 11703 of 29566 'ref/call' occurrence rows,
                // 39.6%, find no edge row in the LEFT JOIN at all, and 11637 of
                // those have both endpoints alive in `nodes`, so they are real
                // servable sites rather than dangling rows. Every one comes back
                // `heuristic = NULL -> false` and is therefore presented as
                // resolved fact. The MAX only decides the 60% that do join;
                // the fail-open default for the rest is documented on
                // `travsr_core::RefSite::heuristic`.
                // `live` is deliberately NOT flagged here. `RefSite::heuristic`
                // is a bool and renders as "matched by name, not resolved by
                // type", which is false of a live edge: that edge IS resolved,
                // it is only unratified, and `get_callers` says so in its own
                // words. Flagging it here would caveat it with the wrong cause.
                // The clause could not fire today in any case, since
                // `put_edge_live` writes `edges` and never `edge_sites`, so a
                // live edge has no occurrence row to join. Telling the two
                // apart needs a variant on `RefSite` rather than a bool.
                "SELECT n.path AS path, es.line AS line, \
                 MAX(es.kind = 'ref/call' AND e.provenance = 'tree-sitter') AS heuristic \
                 FROM edge_sites es JOIN nodes n ON n.id = es.src \
                 LEFT JOIN edges e \
                   ON e.src = es.src AND e.dst = es.dst AND e.kind = es.kind \
                 WHERE es.dst = ?1 AND es.kind IN ('ref/call', 'ref/field') \
                 GROUP BY n.path, es.line \
                 ORDER BY n.path, es.line",
            )
            .context("preparing reference_sites query")?;
        let rows = stmt
            .query_map(params![node_id_to_i64(dst)], |row| {
                let path: String = row.get(0)?;
                let line: i64 = row.get(1)?;
                // NULL when no contributing row had an edge to read.
                let heuristic: Option<i64> = row.get(2)?;
                Ok((path, line, heuristic))
            })
            .context("executing reference_sites query")?;
        let mut out = Vec::new();
        for row in rows {
            let (path, line, heuristic) = row.context("decoding reference_sites row")?;
            out.push(travsr_core::RefSite {
                path,
                // Clamp defensively — stored lines are already 1-based u32.
                line: u32::try_from(line).unwrap_or(0),
                heuristic: heuristic.unwrap_or(0) != 0,
            });
        }
        tracing::debug!(sites_returned = out.len());
        Ok(out)
    }

    /// Recorded `ref/call` occurrence lines of ONE edge (`src` -> `dst`), in
    /// ascending order, capped at `max`.
    ///
    /// `get_callers` expands a `(src, dst, kind)`-deduplicated call edge into
    /// its individual sites. It used to do that by re-scanning the caller's
    /// source for the callee's name, which counts anything spelled the same:
    /// a comment, a string, and the callee's own declaration. This reads the
    /// occurrence store instead - the same rows `reference_sites` serves - so
    /// the two tools cannot disagree about where a symbol is called. An empty
    /// result means no occurrence rows exist for this edge (a language not yet
    /// feeding `edge_sites`), and the caller falls back to the textual scan.
    pub fn edge_call_site_lines(
        &self,
        src: NodeId,
        dst: NodeId,
        max: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT line FROM edge_sites \
                 WHERE src = ?1 AND dst = ?2 AND kind = 'ref/call' \
                 ORDER BY line LIMIT ?3",
            )
            .context("preparing edge_call_site_lines query")?;
        let rows = stmt
            .query_map(
                params![
                    node_id_to_i64(src),
                    node_id_to_i64(dst),
                    i64::try_from(max).unwrap_or(i64::MAX)
                ],
                |row| row.get::<_, i64>(0),
            )
            .context("executing edge_call_site_lines query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(u32::try_from(row.context("decoding edge_call_site_lines row")?).unwrap_or(0));
        }
        Ok(out)
    }

    /// RFC-027 #813 P2: current `(line, end_line)` span of each given node, so the
    /// live lane can bound a changed definition's enumerated occurrences to that
    /// definition's current extent (an occurrence whose remapped line fell outside
    /// it after the body changed is not served). A node absent from the graph or
    /// carrying no line is simply absent from the returned map.
    pub fn current_spans(
        &self,
        ids: &[NodeId],
    ) -> anyhow::Result<std::collections::HashMap<NodeId, (u32, Option<u32>)>> {
        let mut out = std::collections::HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let mut stmt = self
            .conn
            .prepare_cached("SELECT line, end_line FROM nodes WHERE id = ?1 AND line IS NOT NULL")
            .context("preparing current_spans query")?;
        for &id in ids {
            let span = stmt
                .query_row(params![node_id_to_i64(id)], |r| {
                    Ok((
                        r.get::<_, i64>(0)? as u32,
                        r.get::<_, Option<i64>>(1)?.map(|e| e as u32),
                    ))
                })
                .optional()
                .context("executing current_spans query")?;
            if let Some(span) = span {
                out.insert(id, span);
            }
        }
        Ok(out)
    }

    /// Whether the occurrence index was actually built for `language` in this
    /// repo, i.e. at least one `ref/call` occurrence row has an endpoint in a
    /// file of that language.
    ///
    /// #299 M1: lets `find_references` distinguish "the occurrence index was
    /// never built for this language" (Phase B absent, its analyzer tool not
    /// installed, or the provider emits no occurrence lines) from "this symbol
    /// genuinely has no references" — the two must not render identically.
    ///
    /// This is deliberately **evidence-based** (an occurrence row must exist)
    /// rather than trusting a per-language "Phase B ran" marker: a sidecar can
    /// run to completion yet produce nothing when its underlying analyzer is
    /// missing (e.g. `scip-php` not installed), and a marker keyed on "ran"
    /// would then falsely report a confident "0 references" for a language whose
    /// references were never analyzed. Checking **either** endpoint (`src` OR
    /// `dst`) also fixes the empty-`language` false-negative: an occurrence's
    /// enclosing `src` node reliably carries the calling file's language even
    /// when the `dst` definition node was stored with an empty `language`.
    /// Whether `path` carries any recorded `ref/call` occurrence — i.e. whether
    /// Phase B appears to have analysed this file at all.
    ///
    /// #450. The first cut of this check used a per-*language* coverage ratio,
    /// which review (#551) correctly rejected: `find_references` resolves
    /// targets of any kind, so a ratio computed over one population of files can
    /// describe a set the target's own file is not in. On this repo's graph that
    /// is 52 of 221 Rust files and 13 of 78 TypeScript files — files holding only
    /// type definitions, with no function or method node in them.
    ///
    /// Asking about the target's own file avoids the mismatch entirely and is a
    /// stronger signal besides: a language-wide ratio says how much of the
    /// language was covered, this says whether *this* file was.
    ///
    /// **Still a proxy.** A file that genuinely references nothing and is
    /// referenced by nothing looks identical to an unanalysed one, so a `false`
    /// means "cannot claim coverage", not "definitely unanalysed". Callers must
    /// treat it as a reason to soften a claim, never as evidence of absence.
    /// RFC-024 / #549 would replace this with a recorded per-file coverage map.
    ///
    /// Empty path returns `false` — the file-attribution fallback's default is
    /// not a real file.
    pub fn file_has_occurrences(&self, path: &str) -> anyhow::Result<bool> {
        if path.is_empty() {
            return Ok(false);
        }
        let present: i64 = self
            .conn
            .query_row(
                "SELECT EXISTS( \
                   SELECT 1 FROM nodes n \
                    WHERE n.path = ?1 \
                      AND n.id IN ( \
                        SELECT src FROM edge_sites WHERE kind = 'ref/call' \
                        UNION SELECT dst FROM edge_sites WHERE kind = 'ref/call'))",
                params![path],
                |row| row.get(0),
            )
            .context("querying file_has_occurrences")?;
        Ok(present != 0)
    }

    /// Files of `language` carrying at least one `ref/call` occurrence, as
    /// `(files_with_occurrences, files_total)`.
    ///
    /// #450. Reported alongside [`Self::file_has_occurrences`] purely as
    /// *context* — it tells a reader how much of the language was analysed, which
    /// makes a softened result interpretable ("4 of 78 files" reads very
    /// differently from "161 of 221"). It is deliberately **not** the gate: per
    /// review on #551, a language-wide ratio can describe a population the target
    /// is not part of.
    ///
    /// Counts every node kind, not just functions and methods. Restricting to
    /// callables made the ratio look healthier than the index is — 161/169 (95%)
    /// against 161/221 (72%) for Rust here — by excluding files the target may
    /// well live in.
    ///
    /// Returns `(0, 0)` for the empty language, matching
    /// [`Self::language_has_edge_sites`]'s treatment of the attribution fallback.
    pub fn language_occurrence_coverage(&self, language: &str) -> anyhow::Result<(u64, u64)> {
        if language.is_empty() {
            return Ok((0, 0));
        }
        // Two scalar sub-selects rather than a `JOIN … ON es.src = n.id OR
        // es.dst = n.id`: an OR-join defeats both `edge_sites` PK prefixes and
        // forces a scan. The UNION of src/dst probes each index separately.
        let (with_occ, total): (i64, i64) = self
            .conn
            .query_row(
                "SELECT \
                   (SELECT COUNT(DISTINCT path) FROM nodes \
                      WHERE id IN ( \
                        SELECT src FROM edge_sites WHERE kind = 'ref/call' \
                        UNION SELECT dst FROM edge_sites WHERE kind = 'ref/call') \
                      AND language = ?1), \
                   (SELECT COUNT(DISTINCT path) FROM nodes WHERE language = ?1)",
                params![language],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("querying language_occurrence_coverage")?;
        Ok((with_occ.max(0) as u64, total.max(0) as u64))
    }

    pub fn language_has_edge_sites(&self, language: &str) -> anyhow::Result<bool> {
        if language.is_empty() {
            // No occurrence index is ever "built for" the empty language; the
            // empty string is the file-attribution fallback's default, not a
            // real language the user can query.
            return Ok(false);
        }
        let present: i64 = self
            .conn
            .query_row(
                "SELECT EXISTS( \
                   SELECT 1 FROM edge_sites es \
                   JOIN nodes n ON n.id IN (es.src, es.dst) \
                   WHERE es.kind = 'ref/call' AND n.language = ?1)",
                params![language],
                |row| row.get(0),
            )
            .context("querying language_has_edge_sites")?;
        Ok(present != 0)
    }

    /// Record `ref/call` occurrence lines directly (issue #299 WS-4).
    ///
    /// Used for producers that already know the enclosing-caller node id and the
    /// occurrence line — currently the daemon's cross-crate `UnresolvedCall`
    /// resolution, where `src` is the caller function node and `line` is the
    /// call-site line. Rows with `line == 0` (unknown) are skipped. Idempotent
    /// via the `edge_sites` PK; safe to re-run on every reindex.
    pub fn record_edge_sites(
        &mut self,
        sites: &[(NodeId, NodeId, u32, Option<u32>)],
    ) -> anyhow::Result<()> {
        if sites.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("record_edge_sites: begin")?;
        for &(src, dst, line, col) in sites {
            if line == 0 {
                continue;
            }
            // Skip self-loops so the occurrence store never diverges from the
            // `ref/call` edge set: `write_phase_b_results` drops any edge that
            // collapses to `src == dst` after alias remapping, so a site with the
            // same collapse would otherwise be an occurrence with no backing edge
            // (find_references would show it; get_callers / blast_radius would not).
            if src == dst {
                continue;
            }
            // RFC-027 #813 P2: the PK is (src, dst, kind, line), so `col` is
            // metadata on that row. INSERT OR IGNORE keeps the first-written row;
            // the UPSERT below fills a NULL `col` from a later real position
            // without ever creating a duplicate row (col is not in the PK) or
            // clobbering a non-NULL col.
            tx.execute(
                "INSERT INTO edge_sites(src, dst, kind, line, col) VALUES(?1, ?2, 'ref/call', ?3, ?4) \
                 ON CONFLICT(src, dst, kind, line) DO UPDATE SET col = COALESCE(edge_sites.col, excluded.col)",
                params![node_id_to_i64(src), node_id_to_i64(dst), line as i64, col.map(|c| c as i64)],
            )
            .context("record_edge_sites: insert")?;
        }
        tx.commit().context("record_edge_sites: commit")
    }

    /// Record field-access occurrence lines under the `ref/field` kind (#757).
    ///
    /// A field read (`x.foo`) is a genuine use-site but not a call, so it is
    /// stored distinctly from `ref/call`: `find_references` returns it (its
    /// query spans both kinds) while `get_callers` / `get_blast_radius`, which
    /// traverse the `ref/call` *edge* set, never surface it as a caller. Skips
    /// unknown (`line == 0`) and self-loop rows for the same reasons
    /// [`Self::record_edge_sites`] does.
    pub fn record_field_sites(
        &mut self,
        sites: &[(NodeId, NodeId, u32, Option<u32>)],
    ) -> anyhow::Result<()> {
        if sites.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("record_field_sites: begin")?;
        for &(src, dst, line, col) in sites {
            if line == 0 || src == dst {
                continue;
            }
            // RFC-027 #813 P2: carry the occurrence byte column, filling a NULL
            // from a later real position and never duplicating the row (see
            // [`Self::record_edge_sites`]).
            tx.execute(
                "INSERT INTO edge_sites(src, dst, kind, line, col) VALUES(?1, ?2, 'ref/field', ?3, ?4) \
                 ON CONFLICT(src, dst, kind, line) DO UPDATE SET col = COALESCE(edge_sites.col, excluded.col)",
                params![node_id_to_i64(src), node_id_to_i64(dst), line as i64, col.map(|c| c as i64)],
            )
            .context("record_field_sites: insert")?;
        }
        tx.commit().context("record_field_sites: commit")
    }

    /// G1: Register a mapping from a raw SCIP symbol string to a unified tree-sitter `NodeId`.
    ///
    /// Idempotent: if the alias already exists with the same `node_id`, this is a no-op.
    pub fn register_symbol_alias(
        &mut self,
        scip_symbol: &str,
        node_id: NodeId,
    ) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO symbol_aliases(scip_symbol, node_id) VALUES(?1, ?2) \
                 ON CONFLICT(scip_symbol) DO UPDATE SET node_id = excluded.node_id",
                params![scip_symbol, node_id_to_i64(node_id)],
            )
            .context("register_symbol_alias")?;
        Ok(())
    }

    /// G1: Batch variant of [`Self::register_symbol_alias`].
    ///
    /// Wraps all inserts in a single transaction with a cached statement so
    /// the unification pass pays one fsync for the whole alias set instead of
    /// one autocommit transaction per alias.
    pub fn register_symbol_aliases(&mut self, aliases: &[(String, NodeId)]) -> anyhow::Result<()> {
        if aliases.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction()
            .context("register_symbol_aliases: begin")?;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO symbol_aliases(scip_symbol, node_id) VALUES(?1, ?2) \
                     ON CONFLICT(scip_symbol) DO UPDATE SET node_id = excluded.node_id",
                )
                .context("register_symbol_aliases: prepare")?;
            for (scip_symbol, node_id) in aliases {
                stmt.execute(params![scip_symbol, node_id_to_i64(*node_id)])
                    .context("register_symbol_aliases: insert")?;
            }
        }
        tx.commit().context("register_symbol_aliases: commit")
    }

    /// G1: Find the tree-sitter node at `(corpus, path)` whose signature
    /// matches one of `signatures`, within `max_delta` lines of `scip_line`.
    ///
    /// Candidate signatures come from
    /// `travsr_indexer::scip_unifier::candidate_signatures` and already encode
    /// the node kind via their prefix (`fn:` / `class:` / `var:` / ...), so no
    /// kind filter is needed.  The `(corpus, path)` index keeps the candidate
    /// row set per-file small.  Returns `None` when no candidate is within the
    /// proximity threshold.
    ///
    /// `signatures` is ordered most → least specific by the caller (e.g.
    /// `method:Server.Serve`, `fn:Server.Serve`, `fn:Serve`). Ties break by
    /// span containment first, then narrowest containing span, then candidate
    /// priority, then line distance (see the E6 note on the SQL below), so a
    /// less-specific same-named node one line closer cannot shadow a more
    /// specific match unless the SCIP line actually falls inside its span.
    ///
    /// The SQL string varies only by candidate count, so `prepare_cached`
    /// hits its statement cache across the millions of calls a monorepo
    /// unification pass makes.
    pub fn find_ts_node_for_unification(
        &self,
        corpus: &str,
        path: &str,
        signatures: &[String],
        scip_line: i64,
        max_delta: i64,
    ) -> anyhow::Result<Option<NodeId>> {
        if signatures.is_empty() {
            return Ok(None);
        }
        // Numbered params: ?1=corpus ?2=path ?3..?(n+2)=signatures
        // ?(n+3)=scip_line ?(n+4)=max_delta. The CASE re-uses the signature
        // params to rank rows by candidate priority (0 = most specific).
        let n = signatures.len();
        let placeholders = (3..n + 3)
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let priority_case = (3..n + 3)
            .map(|i| format!("WHEN ?{i} THEN {}", i - 3))
            .collect::<Vec<_>>()
            .join(" ");
        let line_p = n + 3;
        let delta_p = n + 4;
        // E6: positional-first def-unification. A SCIP definition occurrence's
        // `line` is the selector (name) line, so it falls inside the Phase A
        // node's full `[line, end_line]` span (N2 spans). Prefer the candidate
        // whose span *contains* the SCIP line — this unifies correctly across
        // wide annotation/decorator/doc-comment gaps that the old ±max_delta
        // proximity gate silently dropped (orphaned SCIP twin stealing edges).
        // The ±max_delta window is kept only as a fallback for degenerate spans
        // (NULL/zero-width `end_line`), so this change can only ADD unifications,
        // never remove one that already matched. Ties: narrowest containing span,
        // then candidate-signature priority, then proximity.
        let contains = format!("?{line_p} BETWEEN line AND COALESCE(end_line, line)");
        let sql = format!(
            "SELECT id FROM nodes \
             WHERE corpus = ?1 AND path = ?2 \
               AND signature IN ({placeholders}) \
               AND line IS NOT NULL \
               AND ( ({contains}) OR ABS(line - ?{line_p}) <= ?{delta_p} ) \
             ORDER BY CASE WHEN {contains} THEN 0 ELSE 1 END ASC, \
                      (COALESCE(end_line, line) - line) ASC, \
                      CASE signature {priority_case} END ASC, \
                      ABS(line - ?{line_p}) ASC \
             LIMIT 1"
        );

        let mut bind: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(n + 4);
        bind.push(&corpus);
        bind.push(&path);
        for sig in signatures {
            bind.push(sig);
        }
        bind.push(&scip_line);
        bind.push(&max_delta);

        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .context("find_ts_node_for_unification: prepare")?;
        let id: Option<i64> = stmt
            .query_row(bind.as_slice(), |row| row.get(0))
            .optional()
            .context("find_ts_node_for_unification")?;
        Ok(id.map(i64_to_node_id))
    }

    /// G1 overload-collapse rung: the tree-sitter node at `(corpus, path)`
    /// matching one of `signatures`, but ONLY when the whole candidate set
    /// matches exactly one node in that file.
    ///
    /// Unlike [`Self::find_ts_node_for_unification`] this ignores the SCIP
    /// definition line entirely. It exists for the case where line proximity is
    /// the wrong question: Phase A signatures carry no parameter list, so an
    /// overload set `C.n(...)` collapses into one node anchored at the first
    /// overload, while SCIP emits a separate def per overload at its own line.
    /// The later overloads are the same method group, just not near that anchor.
    ///
    /// The uniqueness requirement is the safety property, and the caller must
    /// pass container-qualified candidates only (see
    /// `travsr_indexer::scip_unifier::overload_collapse_signatures`). With those,
    /// "the only match in this file" and "the right match" are the same thing.
    /// Two or more matches mean the assumption does not hold here, so it returns
    /// `None` rather than guessing.
    pub fn find_unique_ts_node_in_file(
        &self,
        corpus: &str,
        path: &str,
        signatures: &[String],
    ) -> anyhow::Result<Option<NodeId>> {
        if signatures.is_empty() {
            return Ok(None);
        }
        let placeholders = (3..signatures.len() + 3)
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        // LIMIT 2 so "exactly one" is decidable without counting the whole file.
        let sql = format!(
            "SELECT id FROM nodes \
             WHERE corpus = ?1 AND path = ?2 \
               AND signature IN ({placeholders}) \
               AND line IS NOT NULL \
             LIMIT 2"
        );
        let mut bind: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(signatures.len() + 2);
        bind.push(&corpus);
        bind.push(&path);
        for sig in signatures {
            bind.push(sig);
        }
        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .context("find_unique_ts_node_in_file: prepare")?;
        let ids: Vec<i64> = stmt
            .query_map(bind.as_slice(), |row| row.get(0))
            .context("find_unique_ts_node_in_file: query")?
            .collect::<Result<Vec<_>, _>>()
            .context("find_unique_ts_node_in_file: rows")?;
        Ok(match ids.as_slice() {
            [only] => Some(i64_to_node_id(*only)),
            _ => None,
        })
    }

    /// The set of paths in `corpus` that carry at least one Phase A definition —
    /// a node whose signature is not a `scip:` descriptor and whose kind is not
    /// the synthetic `file` node. #780: SCIP tools (scip-ruby) index gitignored
    /// vendored code that the tree-sitter parser deliberately skips, so those
    /// files hold only SCIP `definition` nodes (plus a `file` stub) and their
    /// defs can never reconcile. G1 uses this to tell a file the structural
    /// parser actually indexed from one only the SCIP tool saw, so the latter's
    /// unreconcilable defs are neither counted as misses nor kept as orphans.
    pub fn phase_a_indexed_paths(
        &self,
        corpus: &str,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT DISTINCT path FROM nodes \
                 WHERE corpus = ?1 AND kind != 'file' AND signature NOT LIKE 'scip:%'",
            )
            .context("phase_a_indexed_paths: prepare")?;
        let paths = stmt
            .query_map([corpus], |row| row.get::<_, String>(0))
            .context("phase_a_indexed_paths: query")?
            .collect::<Result<std::collections::HashSet<String>, _>>()
            .context("phase_a_indexed_paths: collect")?;
        Ok(paths)
    }

    /// The single tree-sitter node anywhere in `corpus` matching one of
    /// `signatures`, or `None` when there is not exactly one.
    ///
    /// Last-resort partner to [`find_ts_node_for_unification`], for the case
    /// where a definition and its declaration live in *different files*. The
    /// same-file matcher cannot see across that split, and the cross-file
    /// rescue in `scip_unifier` only helps when the same symbol already
    /// unified somewhere, which never happens when the declaration is the only
    /// Phase A node: C and C++ out-of-line member definitions
    /// (`widget.h` declares `Widget::draw`, `widget.cpp` defines it) are the
    /// motivating shape.
    ///
    /// Deliberately has **no line constraint**, since a declaration's line says
    /// nothing about its definition's, and instead guards on uniqueness:
    /// more than one candidate means the name is ambiguous in this repo, and
    /// merging would attribute calls to the wrong function. Two same-named
    /// `static` functions in different translation units are different
    /// functions in C, and they produce two rows here, so this refuses rather
    /// than guessing. Refusing leaves the pre-existing orphan behaviour
    /// untouched, so this can only ever add correct unifications.
    pub fn find_unique_ts_node_across_files(
        &self,
        corpus: &str,
        signatures: &[String],
    ) -> anyhow::Result<Option<NodeId>> {
        if signatures.is_empty() {
            return Ok(None);
        }
        let placeholders = (2..signatures.len() + 2)
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        // LIMIT 2: enough to tell "exactly one" from "more than one" without
        // reading a whole repo's worth of homonyms.
        let sql = format!(
            "SELECT id FROM nodes \
             WHERE corpus = ?1 AND signature IN ({placeholders}) AND line IS NOT NULL \
             LIMIT 2"
        );

        let mut bind: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(signatures.len() + 1);
        bind.push(&corpus);
        for sig in signatures {
            bind.push(sig);
        }

        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .context("find_unique_ts_node_across_files: prepare")?;
        let ids: Vec<i64> = stmt
            .query_map(bind.as_slice(), |row| row.get(0))
            .context("find_unique_ts_node_across_files: query")?
            .collect::<Result<Vec<_>, _>>()
            .context("find_unique_ts_node_across_files: collect")?;

        Ok(match ids.as_slice() {
            [only] => Some(i64_to_node_id(*only)),
            _ => None,
        })
    }

    /// All definition node ids at `(corpus, path)` — tree-sitter and SCIP —
    /// ordered by line.  Used by the CLI to surface function-level callers
    /// when a query resolves to a file node (RFC-014 #317: a file's callers
    /// are the callers of the definitions it contains).  Container nodes
    /// (file, import, packages, modules) are excluded: their incoming edges
    /// are structural, not call sites.
    pub fn definition_node_ids_in_file(
        &self,
        corpus: &str,
        path: &str,
    ) -> anyhow::Result<Vec<NodeId>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM nodes \
             WHERE corpus = ?1 AND path = ?2 \
               AND kind IN ('function','method','fn','class','interface','struct', \
                            'trait','enum','type','typedef','union','object', \
                            'protocol','mixin','extension','namespace','init', \
                            'var','variable','const','constant','static') \
             ORDER BY line ASC",
        )?;
        let ids = stmt
            .query_map(params![corpus, path], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<i64>, _>>()
            .context("definition_node_ids_in_file")?;
        Ok(ids.into_iter().map(i64_to_node_id).collect())
    }

    /// Count edges carrying `provenance`.
    ///
    /// RFC-027 leans on this twice: the Invariant #4 convergence check asserts
    /// zero `live` rows in a committed graph, and the precision meter needs the
    /// overlay's size. Cheap enough to call per assertion at fixture scale.
    pub fn count_edges_with_provenance(&self, provenance: &str) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM edges WHERE provenance = ?1",
                params![provenance],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting edges by provenance")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 #813: how many *ratified* `ref/call` edges the definitions in one
    /// file own — the "ratified in-repo call edges" the mid-edit recovery metric
    /// counts. Ratified means committed provenance (`scip`/`lsif`, or the
    /// `tree-sitter` a native Phase B leaf-name resolution writes), excluding the
    /// `live` overlay: recovering a ratified edge is preserving it in place, which
    /// is the win, whereas a `live` re-emission is the fail-open floor that
    /// already existed and cannot reach an ambiguous callee. A save that
    /// preserves an unchanged definition keeps its ratified edge in this count;
    /// the pre-#813 whole-file purge dropped it to zero until the next commit.
    pub fn owned_ratified_ref_call_edges(
        &self,
        corpus: &str,
        path: &str,
    ) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM edges e JOIN nodes n ON n.id = e.src \
                 WHERE e.kind = 'ref/call' AND e.provenance != 'live' \
                   AND n.corpus = ?1 AND n.path = ?2",
                params![corpus, path],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting owned ratified ref/call edges")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 12: upsert reference-resolution rows without clearing the
    /// file's others.
    ///
    /// The editor lane arrives *after* the save-path pass has already recorded
    /// every reference in the file, and it answers a subset of them. It must
    /// therefore upgrade the rows it resolved rather than replace the set:
    /// [`Self::replace_ref_resolution_states`] would delete the references the
    /// editor did not answer, and those pending rows are the honest record of
    /// what is still unresolved.
    ///
    /// Rows collide with the save-path pass on the `(src, ref_line, ref_col,
    /// name)` primary key, so an editor answer flips that reference's existing
    /// `pending` row to `resolved` in place instead of forking one reference
    /// into two rows. That is what lets the precision meter score the editor
    /// lane at all: without a claim row a resolution is invisible to
    /// [`Self::live_precision_sample_by_language`], and a language that runs the
    /// editor lane alone could never earn the per-language gate a reading.
    pub fn upsert_ref_resolution_states(
        &mut self,
        rows: &[RefResolution],
    ) -> Result<(), StoreError> {
        (|| -> AnyResult<()> {
            let tx = self
                .conn
                .transaction()
                .context("upsert_ref_resolution_states: begin")?;
            for r in rows {
                tx.execute(
                    "INSERT INTO ref_resolution_state(src, ref_line, ref_col, name, state, resolved_dst) \
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6) \
                     ON CONFLICT(src, ref_line, ref_col, name) DO UPDATE SET \
                       state = excluded.state, resolved_dst = excluded.resolved_dst",
                    params![
                        node_id_to_i64(r.src),
                        r.ref_line as i64,
                        r.ref_col as i64,
                        r.name,
                        r.state,
                        r.resolved_dst.map(node_id_to_i64),
                    ],
                )
                .context("upserting ref_resolution_state row")?;
            }
            tx.commit().context("upsert_ref_resolution_states: commit")?;
            Ok(())
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 9.2: replace a file's reference-resolution rows.
    ///
    /// Called once per live pass over a file, with every reference that pass
    /// saw and what became of it. Whole-file replacement rather than
    /// incremental patching, for the same reason the live engine re-emits
    /// whole-file: `reindex_replace` has just rewritten the file's nodes, so
    /// any row keyed on a node that no longer exists is stale by construction.
    ///
    /// `rows` are `(src, ref_line, ref_col, name, state)` where `state` is
    /// `"pending"` or `"resolved"`. Owned by `src`'s file, mirroring how
    /// `edge_sites` scopes ownership, so the delete below can be keyed on the
    /// same `(corpus, path)` predicate.
    pub fn replace_ref_resolution_states(
        &mut self,
        corpus: &str,
        path: &str,
        rows: &[RefResolution],
    ) -> Result<(), StoreError> {
        (|| -> AnyResult<()> {
            let tx = self
                .conn
                .transaction()
                .context("replace_ref_resolution_states: begin")?;
            tx.execute(
                "DELETE FROM ref_resolution_state \
                 WHERE src IN (SELECT id FROM nodes WHERE corpus = ?1 AND path = ?2)",
                params![corpus, path],
            )
            .context("clearing this file's ref_resolution_state rows")?;
            for r in rows {
                tx.execute(
                    "INSERT INTO ref_resolution_state(src, ref_line, ref_col, name, state, resolved_dst) \
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6) \
                     ON CONFLICT(src, ref_line, ref_col, name) DO UPDATE SET \
                       state = excluded.state, resolved_dst = excluded.resolved_dst",
                    params![
                        node_id_to_i64(r.src),
                        r.ref_line as i64,
                        r.ref_col as i64,
                        r.name,
                        r.state,
                        r.resolved_dst.map(node_id_to_i64),
                    ],
                )
                .context("inserting ref_resolution_state row")?;
            }
            tx.commit()
                .context("replace_ref_resolution_states: commit")?;
            Ok(())
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 12: score the live lane's claims against Phase B's truth.
    ///
    /// Run at ratification, **after** the Phase B writes and **before** the
    /// sweep. After, because it compares against what Phase B derived; before,
    /// because the sweep is about to discard the evidence.
    ///
    /// Verification is at **call-site line** granularity, joining each live
    /// claim in `ref_resolution_state` to the `edge_sites` rows Phase B recorded
    /// for the same `(src, line)`. Anything coarser is not safe to gate on: a
    /// function that calls several things would let a mis-targeted claim match
    /// some *other* call's correct answer and score as agreement. An optimistic
    /// meter is worse than none, because it clears a bar the lane has not met.
    ///
    /// A claim whose site Phase B recorded nothing for is `unverifiable`, not
    /// wrong — Phase B has recall gaps of its own, and the code can change
    /// between the edit and the commit.
    ///
    /// `SCIP wins all ties` (section 12) falls out of the ordering rather than
    /// needing a rule here: by the time this runs, Phase B has already written
    /// its answer over any co-located live row.
    pub fn live_precision_sample(&self) -> Result<LivePrecision, StoreError> {
        (|| -> AnyResult<LivePrecision> {
            let mut stmt = self
                .conn
                .prepare(
                    // Per claim: does Phase B have any site at this line, and does
                    // one of them name the same target?
                    "SELECT \
                       EXISTS (SELECT 1 FROM edge_sites s \
                               WHERE s.src = r.src AND s.line = r.ref_line) AS has_site, \
                       EXISTS (SELECT 1 FROM edge_sites s \
                               WHERE s.src = r.src AND s.line = r.ref_line \
                                 AND s.dst = r.resolved_dst) AS matches \
                     FROM ref_resolution_state r \
                     WHERE r.state = 'resolved' AND r.resolved_dst IS NOT NULL",
                )
                .context("live_precision_sample: prepare")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)? == 1, row.get::<_, i64>(1)? == 1))
                })
                .context("live_precision_sample: query")?;

            let mut out = LivePrecision::default();
            for row in rows {
                match row.context("live_precision_sample: decode")? {
                    (false, _) => out.unverifiable += 1,
                    (true, true) => out.agree += 1,
                    (true, false) => out.disagree += 1,
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 12: [`live_precision_sample`], split by language.
    ///
    /// The shipping gate is **per-language** ("if it cannot hold that bar for a
    /// language, the lane is disabled for that language"), so the meter has to
    /// attribute each claim to one. Attribution is by the **source** node's
    /// language — the file that was edited, which is the file the live lane ran
    /// on — the same key the ratification sweep scopes its delete on, so the
    /// meter and the sweep never disagree about which language a row belongs to.
    ///
    /// Bucketing is in Rust rather than a SQL `GROUP BY` so the two per-claim
    /// `EXISTS` sub-selects stay identical to [`live_precision_sample`]; the
    /// only change is carrying `n.language` alongside them.
    pub fn live_precision_sample_by_language(
        &self,
    ) -> Result<std::collections::BTreeMap<String, LivePrecision>, StoreError> {
        (|| -> AnyResult<std::collections::BTreeMap<String, LivePrecision>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT n.language, \
                       EXISTS (SELECT 1 FROM edge_sites s \
                               WHERE s.src = r.src AND s.line = r.ref_line) AS has_site, \
                       EXISTS (SELECT 1 FROM edge_sites s \
                               WHERE s.src = r.src AND s.line = r.ref_line \
                                 AND s.dst = r.resolved_dst) AS matches \
                     FROM ref_resolution_state r \
                     JOIN nodes n ON n.id = r.src \
                     WHERE r.state = 'resolved' AND r.resolved_dst IS NOT NULL",
                )
                .context("live_precision_sample_by_language: prepare")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)? == 1,
                        row.get::<_, i64>(2)? == 1,
                    ))
                })
                .context("live_precision_sample_by_language: query")?;

            let mut out: std::collections::BTreeMap<String, LivePrecision> =
                std::collections::BTreeMap::new();
            for row in rows {
                let (lang, has_site, matches) =
                    row.context("live_precision_sample_by_language: decode")?;
                let bucket = out.entry(lang).or_default();
                match (has_site, matches) {
                    (false, _) => bucket.unverifiable += 1,
                    (true, true) => bucket.agree += 1,
                    (true, false) => bucket.disagree += 1,
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 12: consume the claims the precision meter just scored.
    ///
    /// Called at ratification immediately after
    /// [`Self::live_precision_sample_by_language`], because the meter is
    /// **cumulative**: it adds each run's sample onto the counters in `meta`.
    /// Nothing else retires a resolved row —
    /// [`Self::replace_ref_resolution_states`] only fires on that file's next
    /// save and [`Self::clear_resolved_pending_refs`] only touches `pending` —
    /// so without this a claim recorded once is re-scored at every subsequent
    /// commit. The *ratio* survives (both buckets inflate together), but the
    /// sample size does not, and the sample size is what
    /// `LIVE_PRECISION_MIN_SAMPLE` gates on: four real claims would cross a bar
    /// meant to need twenty. A re-score is also not stable, because a claim
    /// whose `ref_line` now belongs to a different call after later edits can
    /// flip from agree to disagree and penalise a resolution that was right when
    /// it was made.
    ///
    /// Safe to run here because the claim has no reader left: the sweep needs
    /// nothing from it, and the freshness note reports only `pending` rows.
    ///
    /// **Scoped to `languages`**, the same set the ratification sweep uses and
    /// for the same #712 reason: only a language whose Phase B just completed had
    /// its claims scored by [`Self::live_precision_sample_by_language`], so only
    /// those may be retired. A crashed sidecar's claims must survive — deleting
    /// them here would score them against evidence that was never refreshed
    /// (landing in `unverifiable`) and then discard them before the language's
    /// real ratification could score them against the truth. Keyed on the
    /// **src** node's language, matching the meter's attribution key exactly.
    ///
    /// Returns the number of rows consumed.
    pub fn consume_measured_ref_resolutions(
        &mut self,
        languages: &[String],
    ) -> Result<usize, StoreError> {
        if languages.is_empty() {
            return Ok(0);
        }
        let placeholders = std::iter::repeat("?")
            .take(languages.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "DELETE FROM ref_resolution_state \
             WHERE state = 'resolved' AND resolved_dst IS NOT NULL \
             AND src IN (SELECT id FROM nodes WHERE language IN ({placeholders}))"
        );
        self.conn
            .execute(&sql, params_from_iter(languages.iter()))
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 10: the pending references in `path`, for the
    /// `live_overlay` freshness note.
    ///
    /// Returns `(name, ref_line)` per unresolved reference, so a consumer can
    /// say *which* call sites are un-targeted rather than only how many.
    pub fn pending_refs_in_file(
        &self,
        corpus: &str,
        path: &str,
    ) -> Result<Vec<(String, u32)>, StoreError> {
        (|| -> AnyResult<Vec<(String, u32)>> {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "SELECT r.name, r.ref_line FROM ref_resolution_state r \
                     JOIN nodes n ON n.id = r.src \
                     WHERE r.state = 'pending' AND n.corpus = ?1 AND n.path = ?2 \
                     ORDER BY r.ref_line ASC",
                )
                .context("pending_refs_in_file: prepare")?;
            let rows = stmt
                .query_map(params![corpus, path], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u32))
                })
                .context("pending_refs_in_file: query")?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.context("decoding pending ref row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 sections 6.3 and 8.7.5: the files whose stranded live edges the
    /// save of `path` can now restore — a dependent holding a `pending`
    /// reference that names a symbol `path` defines.
    ///
    /// This is the reverse-closure that the interface-edit fix (§8.7.5) hands to
    /// the editor. The editor only ever publishes the document that was saved, so
    /// a dependent whose edge into `path` was invalidated is never re-resolved on
    /// its own; naming it here lets the target request pull it in beside the
    /// saved file's own references, keeping the editor the initiator.
    ///
    /// Matched on the pending reference's leaf name against this file's
    /// definition signatures — `kind:Leaf` (unqualified) or `kind:Qual.Leaf`
    /// (qualified), the two shapes `signature` takes. Scoped to `language`: a
    /// live edge stays within one language, both because §8.2 keeps it
    /// intra-corpus and because a cross-language definition would not resolve
    /// through the dependent's own provider anyway.
    ///
    /// Matching is an exact comparison against the saved file's definition leaf
    /// names, resolved **once** with [`travsr_core::ident::leaf_of`] — the same
    /// function the daemon's resolver uses, so the two agree on what a leaf is.
    ///
    /// It is deliberately not a `LIKE` pattern built from `r.name`: `_` is
    /// LIKE's single-character wildcard and identifiers are full of underscores,
    /// so `LIKE '%:' || r.name` matched a pending `do_thing` against a definition
    /// named `doXthing`. That is not recall-neutral either, because `limit` is a
    /// hard cap and spurious matches crowd out the genuine dependents.
    ///
    /// Nor is it a correlated sub-select. Comparing in SQL, per pending row,
    /// against every node in the saved file is quadratic, and `DISTINCT` plus
    /// `ORDER BY` force the whole product to be evaluated before `LIMIT` can
    /// apply. Measured on this repo (15.7k nodes, 4.7k pending rows, a 597-node
    /// saved file) that shape cost **925ms**, at request time, on every save.
    /// Resolving the leaf set first turns it into one hash lookup per pending
    /// row and lets the scan stop as soon as `limit` distinct files are found.
    ///
    /// Capped at `limit` files (`LIVE_CLOSURE_FILE_CAP` at the call site): a
    /// symbol thousands of files are pending on is a hot utility, and the
    /// freshness is not worth re-resolving every one on each save. Truncation
    /// costs recall, which the commit-gated path repairs. Over-inclusion (a leaf
    /// name shared by an unrelated symbol) is recall-neutral: the editor resolves
    /// the real position and the daemon maps it fail-closed, so a spurious file
    /// costs a round trip, never a wrong edge.
    pub fn dependents_pending_on_file(
        &self,
        corpus: &str,
        path: &str,
        language: &str,
        limit: usize,
    ) -> Result<Vec<String>, StoreError> {
        (|| -> AnyResult<Vec<String>> {
            // The saved file's definition leaves, resolved once. Bounded by the
            // node count of one file, and served by the (corpus, path) index.
            // A signature with no `kind:` prefix is not a definition signature
            // and is skipped, matching what the previous `':' || name` /
            // `'.' || name` shape test admitted.
            let mut defs = self
                .conn
                .prepare_cached(
                    "SELECT signature FROM nodes \
                     WHERE corpus = ?1 AND path = ?2 AND language = ?3",
                )
                .context("dependents_pending_on_file: prepare defs")?;
            let leaves: std::collections::HashSet<String> = defs
                .query_map(params![corpus, path, language], |row| {
                    row.get::<_, String>(0)
                })
                .context("dependents_pending_on_file: query defs")?
                .filter_map(|r| r.ok())
                .filter(|sig| sig.contains(':'))
                .map(|sig| travsr_core::ident::leaf_of(&sig).to_string())
                .collect();
            if leaves.is_empty() {
                return Ok(Vec::new());
            }

            // Pending references in other files of the same language, in path
            // order so truncation stays deterministic. Filtering in Rust rather
            // than through an `IN (...)` list keeps this clear of SQLite's
            // 999-variable ceiling, which a large file's leaf set would exceed.
            let mut stmt = self
                .conn
                .prepare_cached(
                    "SELECT dep.path, r.name \
                     FROM ref_resolution_state r \
                     JOIN nodes dep ON dep.id = r.src \
                     WHERE r.state = 'pending' \
                       AND dep.corpus = ?1 AND dep.path <> ?2 AND dep.language = ?3 \
                     ORDER BY dep.path ASC",
                )
                .context("dependents_pending_on_file: prepare")?;
            let rows = stmt
                .query_map(params![corpus, path, language], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .context("dependents_pending_on_file: query")?;

            let mut out: Vec<String> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for row in rows {
                let (dep_path, name) = row.context("decoding dependents_pending_on_file row")?;
                if !leaves.contains(&name) {
                    continue;
                }
                if seen.insert(dep_path.clone()) {
                    out.push(dep_path);
                    // Rows arrive in path order, so the cap can stop the scan
                    // instead of being applied after a full evaluation.
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 10: how many references are currently unresolved.
    ///
    /// The count half of [`pending_refs_in_file`], for the envelope note where
    /// only the magnitude is wanted.
    ///
    /// Joins `nodes` for the same reason [`pending_refs_in_file`] does: a row
    /// whose `src` node no longer exists describes a reference that no longer
    /// exists either, and counting it inflates the number the MCP freshness note
    /// shows an agent. [`Self::purge_orphan_ref_resolution_states`] removes such
    /// rows at ratification; the join is what keeps the two readers agreeing in
    /// between.
    pub fn pending_ref_count(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM ref_resolution_state r \
                 JOIN nodes n ON n.id = r.src \
                 WHERE r.state = 'pending'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting pending references")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// References the live overlay resolved mid-edit (`state = 'resolved'`), the
    /// positive counterpart of [`Self::pending_ref_count`].
    ///
    /// With a `phase_b_dirty` marker set, a non-zero count is positive evidence
    /// the editor/lexical lane recovered the edited region, so the status surface
    /// can say the overlay is fresh rather than a blanket "stale". Joins `nodes`
    /// for the same orphan-row reason `pending_ref_count` does.
    pub fn resolved_ref_count(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row(
                "SELECT count(*) FROM ref_resolution_state r \
                 JOIN nodes n ON n.id = r.src \
                 WHERE r.state = 'resolved'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n as u64)
            .context("counting resolved references")
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 10: unresolved references grouped by the file they sit
    /// in, for a freshness note that can scope itself to the answer it decorates.
    ///
    /// [`Self::pending_ref_count`] answers "how many, repo-wide", which is
    /// ambient: an agent asking who calls one symbol was told how many
    /// references are pending everywhere, which reads as though they relate to
    /// the answer. This returns the per-file split so a caller can keep only the
    /// files its response actually mentions.
    ///
    /// Capped at `limit` files, ordered by descending count then path so the cap
    /// keeps the largest gaps and stays deterministic. The cap is what bounds
    /// the work: this runs on every prose query, against a table that grows one
    /// row per detected reference per saved file.
    pub fn pending_ref_counts_by_file(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, u64)>, StoreError> {
        (|| -> AnyResult<Vec<(String, u64)>> {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "SELECT n.path, count(*) AS c FROM ref_resolution_state r \
                     JOIN nodes n ON n.id = r.src \
                     WHERE r.state = 'pending' \
                     GROUP BY n.path \
                     ORDER BY c DESC, n.path ASC LIMIT ?1",
                )
                .context("pending_ref_counts_by_file: prepare")?;
            let rows = stmt
                .query_map(params![limit as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
                })
                .context("pending_ref_counts_by_file: query")?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.context("decoding pending_ref_counts_by_file row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 8.3: clear pending rows that Phase B has since resolved.
    ///
    /// Run at ratification. A reference Phase B recorded a call site for is no
    /// longer pending, whatever resolved it, so this is keyed on site existence
    /// rather than on which lane won.
    ///
    /// Granularity is `(src, ref_line)`, matching [`Self::live_precision_sample`]
    /// exactly. Anything coarser is not safe: keying on `src` alone asks "does
    /// this *enclosing definition* have any outgoing edge at all", so one
    /// resolved call cleared every other pending reference in the same function
    /// on no evidence. Section 9.2's honest abstention is the deliverable the
    /// whole precision-first argument rests on, and silently under-reporting it
    /// is the one thing it cannot afford.
    ///
    /// Returns the number of rows cleared.
    pub fn clear_resolved_pending_refs(&mut self) -> Result<usize, StoreError> {
        clear_resolved_pending_refs_on(&self.conn).map_err(|e| StoreError::Database(e.to_string()))
    }

    /// RFC-027 section 9.2: drop reference rows whose `src` node is gone.
    ///
    /// [`Self::replace_ref_resolution_states`] scopes its delete through
    /// `src IN (SELECT id FROM nodes WHERE corpus = ? AND path = ?)`, which
    /// cannot reach a row whose node id no longer exists. A `NodeId` hashes the
    /// VName, signature included, so renaming a symbol retires its id and takes
    /// that delete's only handle with it: the row outlives every path that could
    /// remove it (the next save cannot see it, and
    /// [`Self::clear_resolved_pending_refs`] cannot either, because a deleted
    /// node has no sites). Left alone the count the freshness note reports climbs
    /// monotonically with every rename in the repo.
    ///
    /// Run once per ratification rather than per save: the readers already join
    /// `nodes` and so stay correct in the meantime, and this is a table scan.
    ///
    /// Returns the number of rows purged.
    pub fn purge_orphan_ref_resolution_states(&mut self) -> Result<usize, StoreError> {
        purge_orphan_ref_resolution_states_on(&self.conn)
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #811: reconcile `ref_resolution_state` against the graph as it stands.
    ///
    /// The two hygiene deletes above, [`Self::clear_resolved_pending_refs`] and
    /// [`Self::purge_orphan_ref_resolution_states`], used to run only from the
    /// daemon's live-overlay ratification, so a `pending` row that a full CLI
    /// Phase B rebuild (`travsr init --semantic --force`) had since resolved
    /// survived it, and a daemon restarted on an already-current index never
    /// reached ratification at all. `pending_ref_count` then over-reported
    /// references the rebuilt graph resolves, which is the misleading abstention
    /// the freshness note exists to prevent.
    ///
    /// This is the one entry point every Phase B completion path calls. It is
    /// keyed purely on evidence in the store, so it is safe to run whenever the
    /// graph is at rest and needs no knowledge of *which* Phase B produced it:
    ///
    /// - a `pending` row with a matching `edge_sites(src, line)` is treated as
    ///   resolved and goes;
    /// - a row whose `src` node no longer exists describes a reference that no
    ///   longer exists and goes;
    /// - everything else stays. Wiping the table would be the easy fix and the
    ///   wrong one: the surviving rows are the honest record of what is still
    ///   unresolved.
    ///
    /// Granularity is `(src, ref_line)`, not `(src, ref_line, name)`, because
    /// `edge_sites` records where a call was resolved and not which name it
    /// resolved. A line holding several references, `a(b(), c())` or a chained
    /// call, therefore has every `pending` row on it cleared as soon as Phase B
    /// records any site there, including abstentions on the other names. That is
    /// the pre-existing [`Self::clear_resolved_pending_refs`] contract (RFC-027
    /// section 8.3, finding 3 chose this over `src` alone), not something #811
    /// introduced, and it errs towards under-reporting on multi-reference lines:
    /// on this repository roughly a third of pending rows share a line with a
    /// differently named reference. Scoping by name needs `edge_sites` to carry
    /// one, which is a schema change and out of scope here. A row on a line with
    /// no site at all, and a live `src` node, is never touched.
    ///
    /// Both deletes run in one transaction so a failure leaves the table exactly
    /// as it was rather than half reconciled, and both are idempotent: a second
    /// run over the same graph finds nothing to delete and returns zeros.
    pub fn reconcile_ref_resolution_states(&mut self) -> Result<RefReconcileReport, StoreError> {
        (|| -> AnyResult<RefReconcileReport> {
            let tx = self
                .conn
                .transaction()
                .context("reconcile_ref_resolution_states: begin")?;
            let cleared_resolved =
                clear_resolved_pending_refs_on(&tx).context("clearing resolved pending refs")?;
            let purged_orphans = purge_orphan_ref_resolution_states_on(&tx)
                .context("purging orphan ref_resolution_state rows")?;
            tx.commit()
                .context("reconcile_ref_resolution_states: commit")?;
            Ok(RefReconcileReport {
                cleared_resolved,
                purged_orphans,
            })
        })()
        // `{:#}` keeps the SQLite cause behind the context, so a daemon log line
        // says *why* the reconcile failed rather than only which step did.
        .map_err(|e| StoreError::Database(format!("{e:#}")))
    }

    /// RFC-027 section 7.5: map a `(path, line)` position to the graph node
    /// that owns it, returning the **narrowest** enclosing definition.
    ///
    /// This is the `location_to_node` primitive the live lane needs at both
    /// ends: the call site's enclosing function becomes the edge's `src`, and
    /// the definition the editor pointed at becomes its `dst`.
    ///
    /// The RFC describes this as range-source-aware (current Tree-sitter spans
    /// for dirty files, SCIP ranges for clean ones). In this store one lookup
    /// covers both, because a node's `line`/`end_line` always come from the most
    /// recent parse of its file and Phase A re-parses a file on save before the
    /// live engine runs. A position that lands in no definition span returns
    /// `None`, which the caller must treat as an abstention rather than
    /// guessing: fail-closed is the whole precision argument (section 8.1).
    ///
    /// Ordering matches [`find_narrowest_enclosing`]: widest-last, so the first
    /// containing span is the tightest one.
    pub fn enclosing_definition_at(
        &self,
        corpus: &str,
        path: &str,
        line: u32,
    ) -> Result<Option<NodeId>, StoreError> {
        self.enclosing_node_at(corpus, path, line, ENCLOSING_DEFINITION_KINDS)
    }

    /// The tightest node whose span contains `line`, restricted to `kinds`.
    ///
    /// Generalizes [`Self::enclosing_definition_at`] so a caller can supply its
    /// own valid endpoint kinds. RFC-027's live lane needs this: mapping the
    /// target of a `ref/field` edge must find the `field` node the editor's
    /// definition provider pointed at, which the definition-only kind set of
    /// `enclosing_definition_at` deliberately excludes (`get_callers` must never
    /// see a field read as a caller, #757). Each live edge kind therefore passes
    /// the kinds that are valid *for it*, and the gate is the kind set itself.
    ///
    /// `kinds` must contain only internal constant strings; they are bound as
    /// parameters, never interpolated, so no user input can reach the SQL text.
    /// An empty `kinds` matches nothing (`Ok(None)`), never everything.
    pub fn enclosing_node_at(
        &self,
        corpus: &str,
        path: &str,
        line: u32,
        kinds: &[&str],
    ) -> Result<Option<NodeId>, StoreError> {
        if kinds.is_empty() {
            return Ok(None);
        }
        (|| -> AnyResult<Option<NodeId>> {
            // Placeholders start at ?4: ?1 corpus, ?2 path, ?3 line.
            let placeholders = (0..kinds.len())
                .map(|i| format!("?{}", i + 4))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT id FROM nodes \
                 WHERE corpus = ?1 AND path = ?2 \
                   AND line IS NOT NULL AND end_line IS NOT NULL \
                   AND line <= ?3 AND end_line >= ?3 \
                   AND kind IN ({placeholders}) \
                 ORDER BY (end_line - line) ASC, id ASC LIMIT 1"
            );
            let mut stmt = self
                .conn
                .prepare_cached(&sql)
                .context("enclosing_node_at: prepare")?;
            use rusqlite::types::Value;
            let mut vals: Vec<Value> = vec![
                Value::Text(corpus.to_string()),
                Value::Text(path.to_string()),
                Value::Integer(line as i64),
            ];
            vals.extend(kinds.iter().map(|k| Value::Text((*k).to_string())));
            let id: Option<i64> = stmt
                .query_row(params_from_iter(vals), |row| row.get(0))
                .optional()
                .context("enclosing_node_at: query")?;
            Ok(id.map(i64_to_node_id))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// The nearest node of one of `kinds` whose declaration begins at or just
    /// below `line`, for a definition position reported ABOVE the node's span
    /// (issue #816 defect 2 backstop).
    ///
    /// A definition provider may report the item's full range or a bare
    /// `Location` whose start sits on a leading doc comment, attribute, or
    /// decorator line, above the daemon node's declaration line, so
    /// [`Self::enclosing_node_at`] (exact span containment) finds nothing. This
    /// maps such a position to the definition it heads: the node of a valid kind
    /// whose start line is the smallest value at or after `line`, bounded to a
    /// header window (`LIVE_DECL_HEADER_LINES`) so a stray position in a large
    /// gap cannot attach to a distant node. The caller still gates on the node's
    /// name, so a wrong header match abstains rather than minting an edge.
    ///
    /// `kinds` are internal constant strings, bound as parameters, never
    /// interpolated. An empty `kinds` matches nothing (`Ok(None)`).
    pub fn node_starting_at_or_below(
        &self,
        corpus: &str,
        path: &str,
        line: u32,
        kinds: &[&str],
    ) -> Result<Option<NodeId>, StoreError> {
        if kinds.is_empty() {
            return Ok(None);
        }
        (|| -> AnyResult<Option<NodeId>> {
            // Placeholders start at ?4: ?1 corpus, ?2 path, ?3 line.
            let placeholders = (0..kinds.len())
                .map(|i| format!("?{}", i + 4))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT id FROM nodes \
                 WHERE corpus = ?1 AND path = ?2 \
                   AND line IS NOT NULL AND end_line IS NOT NULL \
                   AND line >= ?3 AND line <= ?3 + {LIVE_DECL_HEADER_LINES} \
                   AND kind IN ({placeholders}) \
                 ORDER BY line ASC, (end_line - line) ASC, id ASC LIMIT 1"
            );
            let mut stmt = self
                .conn
                .prepare_cached(&sql)
                .context("node_starting_at_or_below: prepare")?;
            use rusqlite::types::Value;
            let mut vals: Vec<Value> = vec![
                Value::Text(corpus.to_string()),
                Value::Text(path.to_string()),
                Value::Integer(line as i64),
            ];
            vals.extend(kinds.iter().map(|k| Value::Text((*k).to_string())));
            let id: Option<i64> = stmt
                .query_row(params_from_iter(vals), |row| row.get(0))
                .optional()
                .context("node_starting_at_or_below: query")?;
            Ok(id.map(i64_to_node_id))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// G1: Look up the unified `NodeId` for a raw SCIP symbol string.
    ///
    /// Returns `None` if the symbol has not been aliased (i.e. no tree-sitter
    /// node matched it during the unification pass).
    pub fn resolve_scip_symbol(&self, scip_symbol: &str) -> anyhow::Result<Option<NodeId>> {
        let id: Option<i64> = self
            .conn
            .query_row(
                "SELECT node_id FROM symbol_aliases WHERE scip_symbol = ?1",
                params![scip_symbol],
                |row| row.get(0),
            )
            .optional()
            .context("resolve_scip_symbol")?;
        Ok(id.map(i64_to_node_id))
    }

    /// Emit `Depends` edges between all ordered file-node pairs that co-define
    /// the same package/module node, using a single SQL self-join.
    ///
    /// Languages whose files share a namespace without explicit imports (Go,
    /// Swift, Kotlin, Java, Dart) emit a package node during Phase A that every
    /// file in the package points at via `DefinesBinding`. This pass finds all
    /// such groups and writes `file_B --Depends--> file_A` for every ordered
    /// pair (B ≠ A), giving blast-radius BFS structural co-package coupling.
    ///
    /// `pkg_kinds` must contain only internal kind strings (e.g. `"go-pkg"`,
    /// `"swift-module"`) — never values derived from user input.
    ///
    /// Returns the number of new edges inserted (`INSERT OR IGNORE` skips
    /// existing rows).
    pub fn emit_copackage_depends(&mut self, pkg_kinds: &[&str]) -> anyhow::Result<usize> {
        if pkg_kinds.is_empty() {
            return Ok(0);
        }
        // All values come from internal constants — no user input reaches here.
        let in_clause = pkg_kinds
            .iter()
            .map(|s| format!("'{}'", s.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT OR IGNORE INTO edges(src, dst, kind, provenance, confidence) \
             SELECT DISTINCT a.src, b.src, 'depends', 'tree-sitter', NULL \
             FROM edges a \
             JOIN edges b   ON a.dst = b.dst AND a.src != b.src \
             JOIN nodes pkg ON pkg.id = a.dst AND pkg.kind IN ({in_clause}) \
             JOIN nodes fa  ON fa.id = a.src  AND fa.kind = 'file' \
             JOIN nodes fb  ON fb.id = b.src  AND fb.kind = 'file' \
             WHERE a.kind = 'defines/binding' \
               AND b.kind = 'defines/binding'"
        );
        self.conn
            .execute(&sql, [])
            .context("emit_copackage_depends")?;
        Ok(self.conn.changes() as usize)
    }
}

// ── StoreMigratable (T3) ──────────────────────────────────────────────────────

impl StoreMigratable for SqliteStore {
    fn exec_ddl(&mut self, ddl: &str) -> anyhow::Result<()> {
        self.conn
            .execute_batch(ddl)
            .context("SqliteStore::exec_ddl")
    }

    fn schema_version(&self) -> anyhow::Result<u32> {
        let raw = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("reading schema_version from meta")?;
        Ok(raw.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    fn set_schema_version(&mut self, v: u32) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![v.to_string()],
            )
            .context("writing schema_version to meta")?;
        Ok(())
    }

    fn column_exists(&self, table: &str, column: &str) -> anyhow::Result<bool> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = ?2",
                params![table, column],
                |row| row.get(0),
            )
            .context("checking column existence via pragma_table_info")?;
        Ok(count > 0)
    }
}

/// Jaccard(byte-trigrams) threshold for L2-A vocabulary-grounded expansion.
/// A vocab token must share at least this fraction of trigrams with a T0 seed
/// to be included as a candidate OR-arm.  0.4 is empirically conservative:
/// tight enough to avoid noise, loose enough to catch plurals and short variants.
const L2A_JACCARD_THRESHOLD: f64 = 0.4;

// ── FTS helpers + fuzzy search (RFC-012 L1) ───────────────────────────────────

impl SqliteStore {
    /// Public accessor for the current schema/migration version recorded in the
    /// `meta` table. Wraps the private `StoreMigratable::schema_version` so the
    /// MCP `get_graph_stats` tool can surface migration state in the VS Code
    /// extension's graph-stats popup (VSCODE-247). Returns 0 if absent.
    pub fn current_schema_version(&self) -> anyhow::Result<u32> {
        StoreMigratable::schema_version(self)
    }

    // ── FTS helpers (RFC-012 L1) ──────────────────────────────────────────────

    /// Build the token string for a node: tokenized signature + path + kind + language.
    fn node_fts_tokens(node: &Node) -> String {
        format!(
            "{} {} {} {}",
            tokenize_identifier(&node.vname.signature),
            tokenize_identifier(&node.vname.path),
            node.kind.to_ascii_lowercase(),
            node.vname.language.to_ascii_lowercase(),
        )
    }

    /// Maximum number of `embed_text` tokens folded into the FTS content (RFC-022
    /// D1). Bounds index growth on long bodies while capturing the doc/skeleton's
    /// leading natural-language terms, which is where the conceptual signal lives.
    const MAX_EMBED_FTS_TOKENS: usize = 48;

    /// RFC-022 D1 (RC-1) recall widening: fold `embed_text` (doc + skeleton) tokens
    /// into the FTS content. **Default OFF** — measured net-neutral-to-negative on
    /// the seeded fixture on its own (the confidence gate, not recall, is the binding
    /// constraint for conceptual queries, and the extra recall adds precision noise
    /// that needs the companion RRF rebalance/generic-cap before it pays off). Ships
    /// behind this flag per the RFC's phased rollout (§8: each phase default-off
    /// until its bench passes; Phase 4 flips defaults after calibration). Enable with
    /// `TRAVSR_FTS_EMBED_WIDEN=1`; the [`backfill_fts_embed_text`] downgrade path
    /// rebuilds signature-only FTS when it is turned back off.
    fn fts_embed_widen_enabled() -> bool {
        matches!(
            std::env::var("TRAVSR_FTS_EMBED_WIDEN").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    }

    /// RFC-022 D1 (RC-1): FTS token string widened with a node's `embed_text`
    /// skeleton, built from raw columns so callers holding a row (e.g.
    /// [`write_embed_texts_batch`]) need not reconstruct a `Node`. The bare
    /// signature names a compound symbol (`ppr_weighted`) with tokens a conceptual
    /// query never uses ("personalized pagerank"); the doc / AST-skeleton
    /// `embed_text` carries exactly those words, so folding a bounded slice of it
    /// into the FTS content lets the sparse leg reward multi-term conceptual matches.
    /// Falls back to signature-only tokens when `embed_text` is absent, so
    /// un-embedded repos and the reference path are unchanged.
    fn fts_tokens_with_embed_from_parts(
        signature: &str,
        path: &str,
        kind: &str,
        language: &str,
        embed_text: Option<&str>,
    ) -> String {
        let base = format!(
            "{} {} {} {}",
            tokenize_identifier(signature),
            tokenize_identifier(path),
            kind.to_ascii_lowercase(),
            language.to_ascii_lowercase(),
        );
        match embed_text {
            Some(t) if !t.trim().is_empty() => {
                let bounded = Self::embed_fts_tokens(t);
                if bounded.is_empty() {
                    base
                } else {
                    format!("{base} {bounded}")
                }
            }
            _ => base,
        }
    }

    /// Extract a bounded, doc-first FTS token slice from an `embed_text` skeleton
    /// (RFC-022 D1). The skeleton layout is
    /// `function: … | module: … | params: … | returns: … | calls: … | doc: <prose>`;
    /// the `doc:` prose carries the natural-language terms a conceptual query uses
    /// ("Personalized PageRank" for `ppr_weighted`), but it trails a potentially long
    /// `calls:` list, so a naive head-slice would drop it. Emit the doc tokens FIRST,
    /// then the structural remainder, capped at [`MAX_EMBED_FTS_TOKENS`] — so the
    /// conceptual signal is always captured while index growth stays bounded.
    fn embed_fts_tokens(embed_text: &str) -> String {
        let (rest, doc) = match embed_text.split_once("| doc:") {
            Some((r, d)) => (r, d),
            None => (embed_text, ""),
        };
        let doc_toks = tokenize_identifier(doc);
        let rest_toks = tokenize_identifier(rest);
        doc_toks
            .split_whitespace()
            .chain(rest_toks.split_whitespace())
            .take(Self::MAX_EMBED_FTS_TOKENS)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Upsert a node's FTS entry.  Handles the contentless-FTS5 invariant:
    /// a row must be explicitly deleted (with its original tokens) before
    /// re-inserting, otherwise the index silently accumulates stale entries.
    fn put_node_fts(conn: &Connection, node: &Node) -> AnyResult<()> {
        Self::put_node_fts_tokens(conn, node.id, &Self::node_fts_tokens(node))
    }

    /// Core of [`put_node_fts`] parameterised on the final token string, so callers
    /// that widen the content (e.g. [`write_embed_texts_batch`] folding in
    /// `embed_text`, RFC-022 D1) reuse the identical retract/insert/vocab dance.
    fn put_node_fts_tokens(conn: &Connection, id: NodeId, new_tokens: &str) -> AnyResult<()> {
        let id_i64 = node_id_to_i64(id);

        // Fetch old tokens (if any) so we can retract the stale FTS row.
        let old_tokens: Option<String> = conn
            .query_row(
                "SELECT tokens FROM nodes_fts_map WHERE node_id = ?1",
                params![id_i64],
                |row| row.get(0),
            )
            .optional()
            .context("reading old FTS tokens")?;

        if let Some(ref old) = old_tokens {
            // Retract the stale contentless-FTS5 row with its original tokens.
            conn.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) VALUES('delete', ?1, ?2)",
                params![id_i64, old],
            )
            .context("retracting stale FTS row")?;
            // Decrement vocab refcounts for retracted tokens (v10 L2-A).
            Self::vocab_decrement(conn, old)?;
        }

        // Insert new FTS row.
        conn.execute(
            "INSERT INTO nodes_fts(rowid, tokens) VALUES(?1, ?2)",
            params![id_i64, new_tokens],
        )
        .context("inserting FTS row")?;

        // Upsert the map so future retractions can supply the exact token string.
        conn.execute(
            "INSERT INTO nodes_fts_map(node_id, tokens) VALUES(?1, ?2)
             ON CONFLICT(node_id) DO UPDATE SET tokens = excluded.tokens",
            params![id_i64, new_tokens],
        )
        .context("upserting nodes_fts_map")?;

        // Increment vocab refcounts for new tokens (v10 L2-A).
        Self::vocab_increment(conn, new_tokens)?;

        Ok(())
    }

    /// Bulk-init variant of [`put_node_fts`]: writes only the `nodes_fts_map` row
    /// and records the node_id in `_bulk_fts_pending` (a temp table created by
    /// [`begin_bulk_fts_tracking`]), skipping the FTS5 inverted-index insert and
    /// `fts_vocab` increments.
    ///
    /// Called by [`write_file_graphs_batch`] when `bulk=true`.
    /// [`rebuild_fts_from_map`] must be called after all batches to populate
    /// `nodes_fts` and `fts_vocab` for only the written nodes.
    fn put_node_fts_map_only(conn: &Connection, node: &Node) -> AnyResult<()> {
        let id_i64 = node_id_to_i64(node.id);
        let tokens = Self::node_fts_tokens(node);
        conn.execute(
            "INSERT INTO nodes_fts_map(node_id, tokens) VALUES(?1, ?2) \
             ON CONFLICT(node_id) DO UPDATE SET tokens = excluded.tokens",
            params![id_i64, tokens],
        )
        .context("bulk: writing nodes_fts_map")?;
        // Track this node_id so rebuild_fts_from_map inserts FTS entries only
        // for written nodes — not for unchanged nodes whose entries already exist.
        conn.execute(
            "INSERT OR IGNORE INTO _bulk_fts_pending(node_id) VALUES(?1)",
            params![id_i64],
        )
        .context("bulk: recording node_id in _bulk_fts_pending")?;
        Ok(())
    }

    /// #478 WS-1/WS-3: word-segmented `(sig, path)` pair for `nodes_fts_words`
    /// (Leg B, RFC-023 §5.1). Segmentation goes through
    /// `travsr_core::ident::segments` — the same segmenter the boundary
    /// predicate (`ident::contains_token`) uses — so Leg B matches exactly the
    /// vocabulary the anchor guard reasons about.
    ///
    /// Drops segments `< 3` bytes, matching `tokenize_identifier`'s existing
    /// filter for `nodes_fts`/`fts_vocab`. Without this, near-universal short
    /// segments (`"fn"` from every `fn:`-prefixed signature, `"id"`, `"ok"`)
    /// leak into Leg B in a way the trigram index's `len>=3` filter always
    /// prevented, turning them into accidental near-universal anchors.
    fn node_fts_words(node: &Node) -> (String, String) {
        let sig = travsr_core::ident::segments(&node.vname.signature)
            .into_iter()
            .filter(|t| t.len() >= 3)
            .collect::<Vec<_>>()
            .join(" ");
        let path = travsr_core::ident::segments(&node.vname.path)
            .into_iter()
            .filter(|t| t.len() >= 3)
            .collect::<Vec<_>>()
            .join(" ");
        (sig, path)
    }

    /// Upsert a node's word-index entry (Leg B). Mirrors [`put_node_fts_tokens`]'s
    /// contentless-FTS5 retract/insert dance for the two-column `nodes_fts_words`
    /// table, using `nodes_fts_words_map` as the retraction memory (parallel to
    /// `nodes_fts_map`). No vocab-refcount step: `nodes_words_vocab` is an
    /// `fts5vocab` view, FTS5-maintained with zero drift (RFC-023 §5.5).
    fn put_node_fts_words(conn: &Connection, node: &Node) -> AnyResult<()> {
        let id_i64 = node_id_to_i64(node.id);
        let (sig_words, path_words) = Self::node_fts_words(node);

        let old: Option<(String, String)> = conn
            .query_row(
                "SELECT sig_words, path_words FROM nodes_fts_words_map WHERE node_id = ?1",
                params![id_i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("reading old FTS word entry")?;

        if let Some((old_sig, old_path)) = old {
            conn.execute(
                "INSERT INTO nodes_fts_words(nodes_fts_words, rowid, sig, path) \
                 VALUES('delete', ?1, ?2, ?3)",
                params![id_i64, old_sig, old_path],
            )
            .context("retracting stale FTS word row")?;
        }

        conn.execute(
            "INSERT INTO nodes_fts_words(rowid, sig, path) VALUES(?1, ?2, ?3)",
            params![id_i64, sig_words, path_words],
        )
        .context("inserting FTS word row")?;

        conn.execute(
            "INSERT INTO nodes_fts_words_map(node_id, sig_words, path_words) VALUES(?1, ?2, ?3)
             ON CONFLICT(node_id) DO UPDATE SET \
               sig_words = excluded.sig_words, path_words = excluded.path_words",
            params![id_i64, sig_words, path_words],
        )
        .context("upserting nodes_fts_words_map")?;

        Ok(())
    }

    /// Bulk-init variant of [`put_node_fts_words`]: writes only the
    /// `nodes_fts_words_map` row and records the node_id in the same
    /// `_bulk_fts_pending` table [`put_node_fts_map_only`] uses (`INSERT OR
    /// IGNORE`, so recording from both is idempotent). [`rebuild_fts_from_map`]
    /// finalizes both the trigram and word indexes for pending nodes in one pass.
    fn put_node_fts_words_map_only(conn: &Connection, node: &Node) -> AnyResult<()> {
        let id_i64 = node_id_to_i64(node.id);
        let (sig_words, path_words) = Self::node_fts_words(node);
        conn.execute(
            "INSERT INTO nodes_fts_words_map(node_id, sig_words, path_words) VALUES(?1, ?2, ?3)
             ON CONFLICT(node_id) DO UPDATE SET \
               sig_words = excluded.sig_words, path_words = excluded.path_words",
            params![id_i64, sig_words, path_words],
        )
        .context("bulk: writing nodes_fts_words_map")?;
        conn.execute(
            "INSERT OR IGNORE INTO _bulk_fts_pending(node_id) VALUES(?1)",
            params![id_i64],
        )
        .context("bulk: recording node_id in _bulk_fts_pending (words)")?;
        Ok(())
    }

    /// Create the per-connection temp table used to track which node IDs had
    /// their `nodes_fts_map` row written during a bulk init run.
    ///
    /// Must be called once before the first [`write_file_graphs_batch`] call
    /// with `bulk=true`.  The table is dropped by [`rebuild_fts_from_map`].
    pub fn begin_bulk_fts_tracking(&mut self) -> anyhow::Result<()> {
        self.conn
            .execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS _bulk_fts_pending \
                 (node_id INTEGER PRIMARY KEY);",
            )
            .context("creating _bulk_fts_pending temp table")?;
        Ok(())
    }

    /// Create constraint-free TEMP tables for staging bulk init writes.
    ///
    /// During `travsr init`, nodes and edges are written here with plain INSERT
    /// (no ON CONFLICT clause, no index maintenance).  After all files are
    /// processed, call [`flush_staging_to_production`] which deduplicates in a
    /// single GROUP BY pass and writes the clean result to the production tables.
    ///
    /// This eliminates the B-tree read (ON CONFLICT uniqueness check) from every
    /// node and edge insert on the hot init path — converting N×(read+write) to
    /// N×write + 1×sort, where the sort uses sequential I/O via SQLite's external
    /// merge sort.
    ///
    /// Safe to call multiple times (TEMP tables use IF NOT EXISTS).  Crash-safe:
    /// TEMP tables are dropped automatically on connection close, leaving the
    /// production tables untouched.
    pub fn begin_staging_tables(&mut self) -> anyhow::Result<()> {
        self.conn
            .execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS nodes_stage( \
                   id INTEGER, corpus TEXT, root TEXT, path TEXT, \
                   language TEXT, signature TEXT, kind TEXT, \
                   package TEXT, line INTEGER, end_line INTEGER, \
                   is_noise INTEGER, test_role INTEGER, body_hash TEXT \
                 ); \
                 CREATE INDEX IF NOT EXISTS nodes_stage_id ON nodes_stage(id); \
                 CREATE TEMP TABLE IF NOT EXISTS edges_stage( \
                   src INTEGER, dst INTEGER, kind TEXT, \
                   provenance TEXT, confidence INTEGER \
                 ); \
                 CREATE INDEX IF NOT EXISTS edges_stage_pk \
                   ON edges_stage(src, dst, kind);",
            )
            .context("creating staging temp tables")?;
        self.staging_active = true;
        Ok(())
    }

    /// Flush staging tables to production in one deduplicating GROUP BY pass.
    ///
    /// Deduplication semantics match the incremental ON CONFLICT path:
    /// - nodes: GROUP BY id deduplicates; MAX(line)/MAX(end_line) picks a value
    ///   when at most one source row has a non-NULL value (by construction,
    ///   same NodeId ⟹ same VName ⟹ same parse output, so all are equal).
    ///   ON CONFLICT(id) DO UPDATE handles re-init where nodes already exist.
    /// - edges: GROUP BY (src,dst,kind) deduplicates; ON CONFLICT DO NOTHING
    ///   is a no-op for any edge already in production.
    ///
    /// Bare non-grouped columns in the nodes SELECT (corpus, root, path, …) are
    /// safe: same id ⟹ same VName (NodeId is a deterministic hash of the VName),
    /// so SQLite's arbitrary-value pick for the group is always correct.
    ///
    /// Must be called after all [`write_file_graphs_batch`] calls with
    /// `staging_active=true` and before [`rebuild_fts_from_map`].
    ///
    /// Returns `(nodes_written, edges_written)` — rows inserted or updated
    /// into production (post-deduplication).
    pub fn flush_staging_to_production(&mut self) -> anyhow::Result<(u64, u64)> {
        if !self.staging_active {
            return Ok((0, 0));
        }
        let result = (|| -> anyhow::Result<(u64, u64)> {
            let tx = self
                .conn
                .transaction()
                .context("starting staging flush transaction")?;

            let nodes_written = tx
                .execute(
                    "INSERT INTO nodes(id,corpus,root,path,language,signature,kind,package,line,end_line,is_noise,test_role,body_hash) \
                       SELECT id,corpus,root,path,language,signature,kind,package, \
                              MAX(line),MAX(end_line),MAX(is_noise),MAX(test_role),MAX(body_hash) \
                       FROM nodes_stage GROUP BY id \
                       ON CONFLICT(id) DO UPDATE SET \
                         kind     = excluded.kind, \
                         package  = excluded.package, \
                         line     = COALESCE(excluded.line,     nodes.line), \
                         end_line = COALESCE(excluded.end_line, nodes.end_line), \
                         is_noise = excluded.is_noise, \
                         test_role = excluded.test_role, \
                         body_hash = excluded.body_hash",
                    [],
                )
                .context("inserting nodes from staging")?;

            // UX-9: nodes are promoted above in this same transaction, so at this
            // point every legitimate endpoint (this batch's nodes + all pre-existing
            // nodes) is present in `nodes`. Drop any staged edge whose endpoint is
            // absent so a parser that emits a `defines` edge from a container node it
            // never emitted (observed: Ruby `class ::Hash` reopening, whose container
            // signature did not round-trip) cannot promote a dangling half-edge.
            // This makes "no orphan edges" a store invariant at the staging boundary
            // rather than a per-writer promise; cross-file `ref/call` edges are
            // unaffected because their targets are already in `nodes` by promotion
            // time (the reason the bulk path stages edges in the first place).
            let orphan_edges: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM edges_stage e \
                       WHERE NOT EXISTS(SELECT 1 FROM nodes n WHERE n.id = e.src) \
                          OR NOT EXISTS(SELECT 1 FROM nodes n WHERE n.id = e.dst)",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if orphan_edges > 0 {
                tracing::warn!(
                    orphan_edges,
                    "staging flush: dropped tree-sitter edges with a missing endpoint (parser emitted an edge to an un-emitted node)"
                );
            }
            let edges_written = tx
                .execute(
                    "INSERT INTO edges(src,dst,kind,provenance,confidence) \
                       SELECT src,dst,kind,provenance,MAX(confidence) \
                       FROM edges_stage e \
                       WHERE EXISTS(SELECT 1 FROM nodes n WHERE n.id = e.src) \
                         AND EXISTS(SELECT 1 FROM nodes n WHERE n.id = e.dst) \
                       GROUP BY src,dst,kind \
                       ON CONFLICT(src,dst,kind) DO NOTHING",
                    [],
                )
                .context("inserting edges from staging")?;

            tx.execute_batch("DROP TABLE nodes_stage; DROP TABLE edges_stage;")
                .context("dropping staging tables")?;

            tx.commit().context("committing staging flush")?;
            Ok((nodes_written as u64, edges_written as u64))
        })();
        // Always clear the flag whether success or failure so callers that
        // catch and continue don't write into a poisoned staging state.
        self.staging_active = false;
        result
    }

    /// Retract a single node from the FTS index.  No-op if the node has no FTS entry.
    #[allow(dead_code)]
    fn delete_node_fts(conn: &Connection, node_id_i64: i64) -> AnyResult<()> {
        let old_tokens: Option<String> = conn
            .query_row(
                "SELECT tokens FROM nodes_fts_map WHERE node_id = ?1",
                params![node_id_i64],
                |row| row.get(0),
            )
            .optional()
            .context("reading FTS tokens for deletion")?;

        if let Some(tokens) = old_tokens {
            conn.execute(
                "INSERT INTO nodes_fts(nodes_fts, rowid, tokens) VALUES('delete', ?1, ?2)",
                params![node_id_i64, tokens],
            )
            .context("retracting FTS row on node delete")?;
            conn.execute(
                "DELETE FROM nodes_fts_map WHERE node_id = ?1",
                params![node_id_i64],
            )
            .context("removing nodes_fts_map row")?;
            // Decrement vocab refcounts for retracted tokens (v10 L2-A).
            Self::vocab_decrement(conn, &tokens)?;
        }

        Ok(())
    }

    /// Reconcile the RFC-027 `Edge.provenance` read-path indexes on open,
    /// independent of `schema_version` (review issue B).
    ///
    /// The three RFC-027 migrations were collapsed into a single v23 while
    /// released code was still at v22, so shipped databases receive the complete
    /// v23 and never see this. But an earlier revision of this branch numbered
    /// the same work v23/v24/v25, so a **dev** database it stamped at 24 or 25
    /// now sits *above* the runner's max version (23): the runner applies only
    /// migrations with `version() > current`, finds none, and the database
    /// silently keeps the pre-RFC-027 narrow edge indexes forever — finding 6
    /// (a lost covering index, invisible until an EXPLAIN) one level up, in the
    /// runner instead of a query.
    ///
    /// The runner cannot reach it and bumping the max version would re-spend the
    /// numbers the collapse reclaimed, so the reconcile lives here, beside
    /// [`Self::backfill_fts_if_needed`] and for the same reason: an idempotent
    /// convergence step gated on a cheap data check, never on the stamp. A
    /// writable open always runs it (the daemon on startup, `init`, and the
    /// CLI's read-only-then-writable fallback for a database newer than the
    /// binary), which is where a stranded database heals.
    ///
    /// Gate: the covering index carries `provenance` iff the rework is already
    /// applied, so a fresh v23 database and an already-healed one both return
    /// here doing no work.
    fn reconcile_provenance_indexes_if_needed(&mut self) -> AnyResult<()> {
        let already_wide: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master \
                 WHERE type = 'index' AND name = 'idx_edges_dst_kind_cov' \
                   AND sql LIKE '%provenance%')",
                [],
                |r| r.get(0),
            )
            .context("checking whether the covering edge index carries provenance")?;
        if already_wide {
            return Ok(());
        }

        tracing::info!(
            event = "store.reconcile.provenance_indexes",
            "widening the edge covering indexes a collapsed-migration database missed"
        );
        // The same idempotent DDL `V23RefResolutionState::up` runs: every
        // statement is `IF NOT EXISTS` or DROP-then-CREATE, so it converges a
        // stranded database without depending on its stamp, and re-stamps
        // nothing (the collapse deliberately keeps the max at v23).
        self.conn
            .execute_batch(include_str!("migrations/v23_ref_resolution_state.sql"))
            .context("reconciling RFC-027 provenance indexes")?;
        if !self.column_exists("ref_resolution_state", "resolved_dst")? {
            self.conn
                .execute(
                    "ALTER TABLE ref_resolution_state ADD COLUMN resolved_dst INTEGER",
                    [],
                )
                .context("adding resolved_dst to a stranded ref_resolution_state")?;
        }
        Ok(())
    }

    /// RFC-027 #813: add the `nodes.body_hash` column when a database predates
    /// it. Follows the collapsed-migration convention documented on
    /// [`Self::reconcile_provenance_indexes_if_needed`]: an idempotent open-time
    /// step gated on a cheap data check, never a schema-version bump (the RFC-027
    /// migrations were deliberately collapsed to keep the max at v23). The column
    /// is nullable, so existing rows read as "unknown body" and are never
    /// preserved until a reparse stamps them, which is the fail-safe direction.
    ///
    /// The column is populated only at write time, by the paths that also derive
    /// the edges from the same source ([`Self::write_file_graphs_batch`] and its
    /// staging flush, and [`Self::reindex_replace`]). There is deliberately no
    /// disk-reading backfill: hashing the current working tree at open() could
    /// stamp a definition whose committed edges were built from different content
    /// (an edit made while the daemon was down leaves no trace to detect), which
    /// is exactly the stale preservation this hash exists to prevent.
    fn ensure_body_hash_column(&mut self) -> AnyResult<()> {
        if !self.column_exists("nodes", "body_hash")? {
            self.conn
                .execute("ALTER TABLE nodes ADD COLUMN body_hash TEXT", [])
                .context("adding body_hash column to nodes (RFC-027 #813)")?;
        }
        Ok(())
    }

    /// RFC-027 #813 P2: add the nullable `edge_sites.col` column via an open-time
    /// backfill, so an occurrence carries the exact column of its reference and
    /// the live overlay can resolve it at the precise editor position instead of
    /// name-searching the line. Column is a 0-based UTF-8 BYTE offset within the
    /// occurrence line (tree-sitter and the SCIP/LSIF sources all normalise to
    /// bytes before recording; the daemon converts to the editor's UTF-16 column
    /// at use). `NULL` for rows written before this shipped or by a source that
    /// cannot give a reliable position, in which case the daemon falls back to
    /// its word-boundary search (no regression). Not a schema bump: the PK stays
    /// `(src, dst, kind, line)` and `col` is metadata on that row.
    fn ensure_edge_sites_col_column(&mut self) -> AnyResult<()> {
        if !self.column_exists("edge_sites", "col")? {
            self.conn
                .execute("ALTER TABLE edge_sites ADD COLUMN col INTEGER", [])
                .context("adding col column to edge_sites (RFC-027 #813 P2)")?;
        }
        Ok(())
    }

    /// Idempotent FTS backfill called once after migrations at `open()` /
    /// `open_in_memory()`.  Cheap gate: if `COUNT(nodes) == COUNT(nodes_fts_map)`
    /// the index is up to date and we return immediately.  On first open after
    /// the v9 migration ships, indexes any nodes not yet in the map.
    fn backfill_fts_if_needed(&mut self) -> AnyResult<()> {
        let node_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .context("counting nodes for FTS backfill gate")?;
        let map_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes_fts_map", [], |r| r.get(0))
            .context("counting nodes_fts_map for FTS backfill gate")?;

        if node_count == map_count {
            return Ok(());
        }

        // map_count > node_count is possible when stale FTS entries exist
        // (nodes deleted via a path that ran outside this store instance).
        // In that case the JOIN in search_nodes_fuzzy silently skips them;
        // the count-inequality still triggers a no-op backfill pass.
        tracing::info!(
            missing = backfill_counts(node_count, map_count).0,
            stale = backfill_counts(node_count, map_count).1,
            "building the text search index for new symbols"
        );

        // Fetch unindexed nodes into a Vec first so the statement is dropped
        // before we open the write transaction (borrow-checker: immutable stmt
        // borrow must not overlap the mutable conn borrow for transaction()).
        let nodes: Vec<Node> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line \
                     FROM nodes WHERE id NOT IN (SELECT node_id FROM nodes_fts_map)",
                )
                .context("preparing FTS backfill query")?;
            let collected = stmt
                .query_map([], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing FTS backfill query")?
                .collect::<Result<_, _>>()
                .context("collecting FTS backfill rows")?;
            collected
        }; // stmt dropped here — conn is free for the write transaction

        // Index in one transaction to minimise WAL pressure.
        let tx = self
            .conn
            .transaction()
            .context("starting FTS backfill transaction")?;
        for node in &nodes {
            Self::put_node_fts(&tx, node).context("put_node_fts during backfill")?;
        }
        tx.commit().context("committing FTS backfill transaction")?;

        tracing::info!(indexed = nodes.len(), "text search index updated");
        Ok(())
    }

    /// #478 WS-3: idempotent backfill for `nodes_fts_words` (Leg B) + `is_noise`,
    /// called once after migrations at `open()` / `open_in_memory()`, mirroring
    /// [`backfill_fts_if_needed`]. Same cheap gate: if
    /// `COUNT(nodes) == COUNT(nodes_fts_words_map)` both are already up to date.
    ///
    /// Cannot share `backfill_fts_if_needed`'s gate or node set: right after the
    /// v21 migration ships, `nodes_fts_map` is already fully populated (no rows
    /// missing) while `nodes_fts_words_map` is completely empty, so a shared
    /// gate would wrongly skip this backfill forever. `is_noise` is recomputed
    /// here too — the ALTER TABLE default (`0`) is correct only for brand-new
    /// rows written after this PR; every pre-existing row needs the real value.
    fn backfill_fts_words_if_needed(&mut self) -> AnyResult<()> {
        let node_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .context("counting nodes for FTS word backfill gate")?;
        let words_map_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes_fts_words_map", [], |r| r.get(0))
            .context("counting nodes_fts_words_map for FTS word backfill gate")?;

        if node_count == words_map_count {
            return Ok(());
        }

        // DEBUG, not INFO. The gate above is a count comparison, which is
        // deliberately conservative: stale map rows left by a delete make the
        // counts differ forever, so this pass runs on every startup and indexes
        // nothing. It cannot be gated on `missing == 0` instead — with stale
        // rows present that figure can read zero while nodes really are
        // unindexed, and the `NOT IN` query below is the only reliable answer.
        // So the pass stays, and only its *outcome* is announced.
        let (missing, stale) = backfill_counts(node_count, words_map_count);
        tracing::debug!(missing, stale, "checking the word index for new symbols");

        let nodes: Vec<Node> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line \
                     FROM nodes WHERE id NOT IN (SELECT node_id FROM nodes_fts_words_map)",
                )
                .context("preparing FTS word backfill query")?;
            let collected = stmt
                .query_map([], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing FTS word backfill query")?
                .collect::<Result<_, _>>()
                .context("collecting FTS word backfill rows")?;
            collected
        };

        let tx = self
            .conn
            .transaction()
            .context("starting FTS word backfill transaction")?;
        for node in &nodes {
            Self::put_node_fts_words(&tx, node).context("put_node_fts_words during backfill")?;
            tx.execute(
                "UPDATE nodes SET is_noise = ?2 WHERE id = ?1",
                params![
                    node_id_to_i64(node.id),
                    travsr_core::noise::is_structural_noise(node),
                ],
            )
            .context("backfilling is_noise")?;
        }
        tx.commit()
            .context("committing FTS word backfill transaction")?;

        // Only a pass that did something is worth a line in the log.
        if nodes.is_empty() {
            tracing::debug!("word index already current");
        } else {
            tracing::info!(
                event = "store.fts_words.backfill",
                indexed = nodes.len(),
                "word index updated"
            );
        }
        Ok(())
    }

    // ── #393: RRF fusion cascade (replaces the first-nonempty short-circuit) ──
    //
    // The legacy cascade returned the first non-empty stage and stopped, which
    // let a partial substring match hide better FTS/embed results and made
    // singular/plural queries return disjoint sets. `fused_search_scored` unifies
    // all three public entry points onto one core that (G1) fast-paths a
    // confident exact-signature hit, (G2) always fuses the two cheap SQLite
    // stages and gates the costly L2-A/embed stages to a combined miss, and
    // (G3) round-robin-interleaves by kind so one kind can't starve out another.

    /// RRF constant `k` (Cormack et al.); dampens the contribution of low ranks.
    const RRF_K: f32 = 60.0;
    /// Per-stage RRF weights. Exact-substring is the most trusted signal.
    const RRF_W_EXACT: f32 = 2.0;
    /// #478 RFC-023 §3: Leg B (word BM25, `nodes_fts_words`) — the new precision
    /// leg. Weighted above the trigram leg since a word-boundary match is
    /// stronger evidence than a substring match. Starting value from the RFC's
    /// own probe; not yet bench-swept (WS-9).
    const RRF_W_WORD: f32 = 1.2;
    /// #478 RFC-023 §3: Leg C (trigram BM25, `nodes_fts`) — down-weighted from
    /// the pre-#478 `1.0` so a pure-substring match (no word-boundary evidence)
    /// can no longer take rank 1 corpus-wide on BM25 length normalisation alone
    /// (Evidence A). Trigram recall/typo-tolerance is unchanged; only its
    /// fusion weight drops. Starting value, not yet bench-swept (WS-9).
    const RRF_W_TRIGRAM: f32 = 0.4;
    const RRF_W_L2A: f32 = 0.7;
    const RRF_W_EMBED: f32 = 1.0;
    /// #478 RFC-023 §5.1: `bm25(nodes_fts_words, W_SIG, W_PATH)` per-column
    /// weights — a signature match outranks a path match for the same term
    /// (Evidence B). From the RFC's §1 probe; not yet bench-swept (WS-9).
    const FTS_WORDS_W_SIG: f64 = 3.0;
    const FTS_WORDS_W_PATH: f64 = 1.0;

    /// Stages 2/B/3/4 of the fused core (everything except Stage 1's G1 exact
    /// fast-path, which only [`Self::fused_search_scored`] special-cases).
    /// Factored out so [`Self::explain_leg_scores`] (#478 RFC-023 §6.1) can
    /// get the same raw per-leg data `travsr explain` needs, without
    /// duplicating the stage-construction logic or threading a collector
    /// through the hot query path.
    fn compute_lexical_stages(
        &self,
        query: &str,
        lang_filter: Option<&str>,
        include_embed: bool,
    ) -> Result<LexicalStages, StoreError> {
        // Stage 2 (Leg C, trigram, down-weighted) — FTS5 BM25 (scored), best-first.
        let stage2 = match build_fuzzy_match_expr_db(query, &self.conn)
            .map_err(|e| StoreError::Database(e.to_string()))?
        {
            Some(expr) => match lang_filter {
                Some(l) => self
                    .fts_query_nodes_scored_with_lang(&expr, l)
                    .map_err(|e| StoreError::Database(e.to_string()))?,
                None => self
                    .fts_query_nodes_scored(&expr)
                    .map_err(|e| StoreError::Database(e.to_string()))?,
            },
            // All tokens < 3 chars — nothing to MATCH.
            None => Vec::new(),
        };

        // Stage B (Leg B, word, RFC-023 §5.1) — word-segmented BM25, best-first.
        // `nodes_fts_words` is `unicode61`, so the query must be pre-segmented
        // the same way the index was populated (one segmenter, by construction),
        // including the `< 3` byte filter `node_fts_words` applies at write time —
        // a MATCH clause for a term the index never contains is dead weight.
        let stage_b_segs: Vec<String> = travsr_core::ident::segments(query)
            .into_iter()
            .filter(|t| t.len() >= 3)
            .collect();
        let stage_b = if stage_b_segs.is_empty() {
            Vec::new()
        } else {
            let expr = stage_b_segs
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR ");
            match lang_filter {
                Some(l) => self
                    .fts_query_words_scored_with_lang(&expr, l)
                    .map_err(|e| StoreError::Database(e.to_string()))?,
                None => self
                    .fts_query_words_scored(&expr)
                    .map_err(|e| StoreError::Database(e.to_string()))?,
            }
        };

        // G2 (#478 RFC-023 §6 item 6): L2-A and embed now always contribute
        // rather than firing only on a combined Stage 1 + Stage 2 + Stage B
        // miss — this closes the #393 cascade short-circuit (a correct L2-A/
        // embed candidate could never surface at all once any cheap stage
        // returned even one weak hit). Weighted RRF fusion means a confident
        // cheap-stage hit is not displaced by weak L2-A/embed noise; it just
        // gets the chance to reinforce or be reinforced.
        let stage3 = self.l2a_scored(query, lang_filter)?;
        let stage4 = if include_embed {
            self.embed_scored(query, lang_filter)?
        } else {
            Vec::new()
        };
        Ok(LexicalStages {
            stage2,
            stage_b,
            stage3,
            stage4,
        })
    }

    /// #478 RFC-023 §6.1: per-leg raw scores for `travsr explain`, uncollapsed
    /// by the G1 exact fast-path (unlike [`Self::fused_search_scored`], whose
    /// whole point is to short-circuit on a crisp signature match) — the
    /// diagnostic's job is showing what every leg found, including when G1
    /// would otherwise have hidden them from the fused result.
    pub fn explain_leg_scores(
        &self,
        query: &str,
        lang_filter: Option<&str>,
    ) -> Result<ExplainLegs, StoreError> {
        let stage1_nodes = match lang_filter {
            Some(l) => self.search_nodes_by_name_with_lang(query, l)?,
            None => self.search_nodes_by_name(query)?,
        };
        let exact = Self::synthetic_desc_scores(stage1_nodes);
        let LexicalStages {
            stage2,
            stage_b,
            stage3,
            stage4,
        } = self.compute_lexical_stages(query, lang_filter, true)?;
        Ok(ExplainLegs {
            exact,
            word: stage_b,
            trigram: stage2,
            l2a: stage3,
            embed: stage4,
        })
    }

    /// Unified fuzzy-search core (#393, #478). `lang_filter = Some(l)` scopes
    /// every stage to language `l`; `include_embed = false` skips the
    /// semantic-ANN stage (used by the get_context seed path, which has its
    /// own KNN channel and must not double-count the embed signal — see plan
    /// §5.1).
    ///
    /// Returns [`FusedHit`]s: `natural` is a BM25-scale relevance float exactly
    /// as before #478 (FTS rows carry `-bm25()`, exact/substring rows a
    /// synthetic descending score, L2-A rows a 0.25 floor, embed rows the
    /// cosine) — every existing consumer of the *ordering* and *natural* score
    /// is unaffected. `bm25_natural`/`exact_rank` are the RFC-023 §5.4 channel
    /// split: `bm25_natural` is `Some` only when Leg B (word) or Leg C
    /// (trigram) matched the node — never for a Leg A (exact)-only or L2-A/
    /// embed-only hit — so the abstention gate that reads it is live rather
    /// than inert (Evidence E). `exact_rank` is `Some` only for a Leg A match.
    fn fused_search_scored(
        &self,
        query: &str,
        lang_filter: Option<&str>,
        include_embed: bool,
        exact_only: bool,
    ) -> Result<Vec<FusedHit>, StoreError> {
        let _span = tracing::debug_span!(
            "store.fused_search_scored",
            query,
            lang = lang_filter.unwrap_or(""),
            include_embed,
            exact_only
        )
        .entered();

        // Stage 1 (Leg A) — exact/word-boundary/prefix, ranked best-first.
        let stage1_nodes = if exact_only {
            match lang_filter {
                Some(l) => self.search_nodes_by_name_with_lang_exact(query, l)?,
                None => self.search_nodes_by_name_exact(query)?,
            }
        } else {
            match lang_filter {
                Some(l) => self.search_nodes_by_name_with_lang(query, l)?,
                None => self.search_nodes_by_name(query)?,
            }
        };

        if exact_only {
            return Ok(Self::synthetic_desc_scores(stage1_nodes)
                .into_iter()
                .enumerate()
                .map(|(rank, (node, natural))| FusedHit {
                    node,
                    natural,
                    bm25_natural: None,
                    exact_rank: Some(rank),
                })
                .collect());
        }

        // G1 — confident-exact fast path: the top row is an exact signature match
        // (SQL rank 0). Return Stage 1 only, so crisp symbol lookups stay precise
        // and can't be diluted by fuzzy neighbours (preserves TL3).
        if let Some(first) = stage1_nodes.first() {
            if first.vname.signature == query {
                tracing::debug!(
                    layer = "exact_fastpath",
                    nodes_returned = stage1_nodes.len()
                );
                return Ok(Self::synthetic_desc_scores(stage1_nodes)
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (node, natural))| FusedHit {
                        node,
                        natural,
                        bm25_natural: None,
                        exact_rank: Some(rank),
                    })
                    .collect());
            }
        }

        let stage1 = Self::synthetic_desc_scores(stage1_nodes);

        let LexicalStages {
            stage2,
            stage_b,
            stage3,
            stage4,
        } = self.compute_lexical_stages(query, lang_filter, include_embed)?;
        let fused = Self::rrf_fuse(&[
            (Self::RRF_W_EXACT, &stage1),
            (Self::RRF_W_WORD, &stage_b),
            (Self::RRF_W_TRIGRAM, &stage2),
            (Self::RRF_W_L2A, &stage3),
            (Self::RRF_W_EMBED, &stage4),
        ]);

        // G3 — kind diversity so one kind can't saturate the visible top-K.
        let diversified = Self::diversify_topk(fused);

        // #478 RFC-023 §5.4: channel split, computed post-fusion by cross-
        // referencing which pre-fusion stage(s) a node appeared in — no change
        // to `rrf_fuse`'s fusion algorithm itself.
        let exact_rank_map: HashMap<NodeId, usize> = stage1
            .iter()
            .enumerate()
            .map(|(rank, (n, _))| (n.id, rank))
            .collect();
        let mut bm25_scale_map: HashMap<NodeId, f32> = HashMap::new();
        for (n, natural) in stage_b.iter().chain(stage2.iter()) {
            bm25_scale_map
                .entry(n.id)
                .and_modify(|s| *s = s.max(*natural))
                .or_insert(*natural);
        }

        Ok(diversified
            .into_iter()
            .map(|(node, natural)| {
                let exact_rank = exact_rank_map.get(&node.id).copied();
                let bm25_natural = bm25_scale_map.get(&node.id).copied();
                FusedHit {
                    node,
                    natural,
                    bm25_natural,
                    exact_rank,
                }
            })
            .collect())
    }

    /// Assign descending synthetic relevance scores [1.0 … 0.1] to a ranked node
    /// list (Stage 1 order is already best-first). Matches the legacy exact-branch
    /// scoring so the exact fast path stays score-compatible.
    fn synthetic_desc_scores(nodes: Vec<Node>) -> Vec<(Node, f32)> {
        let len = nodes.len() as f32;
        nodes
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let score = (1.0f32 - (i as f32 / (len + 1.0))).max(0.1);
                (n, score)
            })
            .collect()
    }

    /// Reciprocal Rank Fusion over pre-ordered stages (each best-first).
    ///
    /// `RRF(d) = Σ_i w_i / (k + rank_i(d))` with 1-based ranks. Nodes are ordered
    /// by descending fused score, tie-broken on node id for determinism. Each
    /// returned node carries its accumulated RRF score (used by
    /// [`Self::diversify_topk`] for score-band tie-breaking) and the **max
    /// natural (BM25-scale) score** across the stages it appeared in, so
    /// downstream BM25 floors stay calibrated (#393 §5.1).
    fn rrf_fuse(stages: &[(f32, &[(Node, f32)])]) -> Vec<(Node, f32, f32)> {
        use std::collections::HashMap;
        // id → (node, accumulated rrf, best natural score)
        let mut acc: HashMap<NodeId, (Node, f32, f32)> = HashMap::new();
        for (weight, stage) in stages {
            for (rank, (node, natural)) in stage.iter().enumerate() {
                let contrib = weight / (Self::RRF_K + (rank as f32) + 1.0);
                let entry = acc
                    .entry(node.id)
                    // Seed the natural score with the first observed value (not
                    // f32::MIN): naturals are always finite here, and MIN.max(NaN)
                    // would silently carry a garbage score if that ever regressed.
                    .or_insert_with(|| (node.clone(), 0.0, *natural));
                entry.1 += contrib;
                entry.2 = entry.2.max(*natural);
            }
        }
        let mut fused: Vec<(Node, f32, f32)> = acc.into_values().collect();
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        fused
    }

    /// Relative score-band width for [`Self::diversify_topk`] (#478 RFC-023 §6
    /// item 7), read from `TRAVSR_DIVERSIFY_BAND_EPSILON` (default 0.05 = 5%).
    /// A node only shares a band with the node(s) ranked above it when its RRF
    /// score is within this fraction of the band leader's score, so the
    /// kind-diversity round-robin can only reorder near-ties, never displace a
    /// clearly-ahead node. 0 disables banding (every node is its own band, so
    /// global RRF order always wins and diversification never fires).
    fn diversify_band_epsilon() -> f32 {
        std::env::var("TRAVSR_DIVERSIFY_BAND_EPSILON")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|&x| (0.0..=1.0).contains(&x))
            .unwrap_or(0.05)
    }

    /// G3 — round-robin interleave by node `kind`, but only as a tie-break
    /// *within* a score band, not across the whole ranked list (#478 RFC-023
    /// §6 item 7). `fused` is already RRF-sorted best-first; we walk it in
    /// contiguous runs ("bands") of nodes whose RRF score is within
    /// [`Self::diversify_band_epsilon`] of the band's leading score, and only
    /// diversify by kind inside each band. Global rank order is preserved
    /// *across* bands, so a clear top-5 same-kind run (Evidence A) is never
    /// perturbed by a minority kind several bands below it — diversity only
    /// kicks in once the fused scores are already close enough that kind is a
    /// legitimate tie-breaker. Still guarantees minority kinds (e.g.
    /// data-format `file` nodes) reach the visible top-K when they are
    /// genuinely competitive with the dominant kind (#393).
    fn diversify_topk(mut fused: Vec<(Node, f32, f32)>) -> Vec<(Node, f32)> {
        if fused.len() <= 2 {
            return fused.into_iter().map(|(n, _rrf, nat)| (n, nat)).collect();
        }
        let epsilon = Self::diversify_band_epsilon();
        let mut out = Vec::with_capacity(fused.len());
        while !fused.is_empty() {
            let band_floor = fused[0].1 * (1.0 - epsilon);
            let band_len = fused
                .iter()
                .take_while(|(_, rrf, _)| *rrf >= band_floor)
                .count()
                .max(1);
            let band: Vec<(Node, f32)> = fused
                .drain(0..band_len)
                .map(|(n, _rrf, nat)| (n, nat))
                .collect();
            out.extend(Self::diversify_band(band));
        }
        out
    }

    /// Round-robin interleave by node `kind` within a single score band,
    /// preserving RRF order within each kind and leading with the band's best
    /// node. A single-kind band passes through unchanged.
    fn diversify_band(band: Vec<(Node, f32)>) -> Vec<(Node, f32)> {
        if band.len() <= 1 {
            return band;
        }
        // Insertion-ordered kind buckets; `band` is already RRF-sorted, so the
        // first bucket created holds the band's best node.
        let mut buckets: Vec<(String, std::collections::VecDeque<(Node, f32)>)> = Vec::new();
        for (node, score) in band {
            match buckets.iter_mut().find(|(k, _)| *k == node.kind) {
                Some((_, q)) => q.push_back((node, score)),
                None => {
                    let kind = node.kind.clone();
                    let mut q = std::collections::VecDeque::new();
                    q.push_back((node, score));
                    buckets.push((kind, q));
                }
            }
        }
        if buckets.len() <= 1 {
            return buckets.into_iter().flat_map(|(_, q)| q).collect();
        }
        let mut out = Vec::new();
        let mut progressed = true;
        while progressed {
            progressed = false;
            for (_, q) in &mut buckets {
                if let Some(item) = q.pop_front() {
                    out.push(item);
                    progressed = true;
                }
            }
        }
        out
    }

    /// L2-A vocabulary-grounded expansion as a scored stage (0.25 floor). Shared
    /// by the fused core; `lang_filter` scopes the FTS lookup. Returns `[]` when
    /// the query yields no tokens or no in-vocabulary expansion candidates.
    fn l2a_scored(
        &self,
        query: &str,
        lang_filter: Option<&str>,
    ) -> Result<Vec<(Node, f32)>, StoreError> {
        (|| -> AnyResult<Vec<(Node, f32)>> {
            let raw_str = tokenize_identifier(query);
            if raw_str.is_empty() {
                return Ok(Vec::new());
            }
            let raw: Vec<String> = raw_str.split_whitespace().map(str::to_string).collect();
            let t0_tokens = crate::seed_lexicon::expand_tokens(&raw);
            let l2a_extra = self.expand_query(&t0_tokens)?;
            if l2a_extra.is_empty() {
                return Ok(Vec::new());
            }
            // #478 RFC-023 §6 item 8: raw T0 query tokens are kept
            // unconditionally — a long query must never lose one of its own
            // words to the arm cap. Only the L2-A expansion candidates are
            // capped, and by IDF (rarest first via `symbol_frequency`), not
            // alphabetically — the old `arms.sort(); arms.truncate(16)` could
            // silently drop a raw token that happened to sort late whenever
            // `t0_tokens` alone already reached 16.
            let mut arms: Vec<String> = t0_tokens.clone();
            let mut extra_ranked: Vec<(String, i64)> = l2a_extra
                .into_iter()
                .filter(|c| !arms.iter().any(|t| t == c))
                .map(|c| {
                    // Absent from vocabulary treated as maximally generic
                    // (pushed to the back), matching the seed.rs fallback contract.
                    let freq = self
                        .symbol_frequency(&c)
                        .ok()
                        .flatten()
                        .map(|f| f as i64)
                        .unwrap_or(i64::MAX);
                    (c, freq)
                })
                .collect();
            extra_ranked.sort_by_key(|(_, freq)| *freq);
            for (c, _) in extra_ranked {
                if arms.len() >= 16 {
                    break;
                }
                arms.push(c);
            }
            let match_expr = arms
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR ");
            let nodes = match lang_filter {
                Some(l) => self.fts_query_nodes_with_lang(&match_expr, l)?,
                None => self.fts_query_nodes(&match_expr)?,
            };
            Ok(nodes.into_iter().map(|n| (n, 0.25f32)).collect())
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Semantic-ANN (RFC-018) as a scored stage carrying the cosine similarity.
    /// Returns `[]` when no embed hook is installed or nothing is scored;
    /// `lang_filter` post-filters KNN results by language.
    fn embed_scored(
        &self,
        query: &str,
        lang_filter: Option<&str>,
    ) -> Result<Vec<(Node, f32)>, StoreError> {
        let Some(knn_fn) = self.embed_knn_hook.as_ref() else {
            return Ok(Vec::new());
        };
        let pairs = knn_fn(query, 20)?;
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<NodeId> = pairs.iter().map(|(id, _)| *id).collect();
        let scores: std::collections::HashMap<NodeId, f32> = pairs.into_iter().collect();
        let nodes = self.get_nodes(&ids)?;
        // get_nodes may reorder; re-attach cosine and restore descending order.
        let mut out: Vec<(Node, f32)> = nodes
            .into_iter()
            .filter(|n| match lang_filter {
                Some(l) => n.vname.language == l,
                None => true,
            })
            .map(|n| {
                let s = scores.get(&n.id).copied().unwrap_or(0.0);
                (n, s)
            })
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        Ok(out)
    }

    /// Language-scoped, scored variant of [`fts_query_nodes_scored`].
    fn fts_query_nodes_scored_with_lang(
        &self,
        match_expr: &str,
        lang: &str,
    ) -> AnyResult<Vec<(Node, f32)>> {
        // #478: is_noise = 0 filters test/vendor/build-artifact noise before the
        // LIMIT (RFC-023 §6 item 5) — the top-K starvation half of #393. Scoped
        // to this FTS-scored leg only, not `search_nodes_by_name` (Leg A), which
        // legitimately needs to find noise-classified nodes by exact name for
        // get_dependencies/get_blast_radius/find_references.
        let sql = "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                          n.kind, n.package, n.line, n.end_line, -bm25(nodes_fts) \
                   FROM nodes_fts \
                   JOIN nodes_fts_map m ON nodes_fts.rowid = m.node_id \
                   JOIN nodes n ON n.id = m.node_id \
                   WHERE nodes_fts MATCH ?1 \
                     AND n.language = ?2 \
                     AND n.is_noise = 0 \
                   ORDER BY bm25(nodes_fts) \
                   LIMIT 50";
        let mut stmt = self
            .conn
            .prepare(sql)
            .context("preparing lang-filtered scored FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr, lang], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                let bm25: f64 = row.get(10).unwrap_or(0.0);
                Ok((
                    Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    },
                    bm25 as f32,
                ))
            })
            .context("executing lang-filtered scored FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding lang-filtered scored FTS5 row")?);
        }
        Ok(out)
    }

    /// Layered seed selection (RFC-012 L1 + A1).
    ///
    /// 1. Exact substring via `search_nodes_by_name` — if ≥1 result, return it.
    /// 2. FTS5 trigram MATCH on the T0 heuristic-normalised token union
    ///    (`build_fuzzy_match_expr`: stopwords stripped, synonym aliases added,
    ///    PA C1 fallback).  If ≥1 result, return it.
    /// 3. L2-A vocabulary-grounded expansion (`expand_query`): Jaccard-similarity
    ///    candidates from `fts_vocab` merged with T0 tokens, capped at 16 OR arms.
    ///    Only fires on combined Step 1 + Step 2 miss.
    ///
    /// The exact-substring step ensures existing benchmarks never regress (TL3).
    /// All synonym/stem output passes through the same double-quote FTS5 escaper
    /// as `build_match_expr` (TL4).  L2-A is vocabulary-grounded: it can only
    /// propose tokens that exist in `fts_vocab` (populated by `put_node_fts`),
    /// so no hallucinated token ever reaches the FTS index — enforcing PA C4.
    ///
    /// #393: thin wrapper over [`fused_search_scored`] — RRF fusion of the cheap
    /// SQLite stages, gated L2-A/embed, and kind diversity replace the legacy
    /// first-nonempty short-circuit. `include_embed = true` (the `ask` path).
    pub fn search_nodes_fuzzy(&self, query: &str) -> Result<Vec<Node>, StoreError> {
        Ok(self
            .fused_search_scored(query, None, true, false)?
            .into_iter()
            .map(|hit| hit.node)
            .collect())
    }

    /// Execute a raw FTS5 MATCH expression against the nodes index.
    /// Shared by Step 2 (T0) and Step 3 (L2-A) to avoid duplicated query logic.
    fn fts_query_nodes(&self, match_expr: &str) -> AnyResult<Vec<Node>> {
        let sql = "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                          n.kind, n.package, n.line, n.end_line \
                   FROM nodes_fts \
                   JOIN nodes_fts_map m ON nodes_fts.rowid = m.node_id \
                   JOIN nodes n ON n.id = m.node_id \
                   WHERE nodes_fts MATCH ?1 \
                   ORDER BY bm25(nodes_fts) \
                   LIMIT 50";
        let mut stmt = self.conn.prepare(sql).context("preparing FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                Ok(Node {
                    id,
                    vname,
                    kind,
                    package,
                    line: line.and_then(|l| u32::try_from(l).ok()),
                    end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                    test_role: TestRole::None,
                })
            })
            .context("executing FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding FTS5 row")?);
        }
        Ok(out)
    }

    /// Language-scoped variant of [`search_nodes_by_name`].
    /// Appends `AND language = ?2` so only nodes from the requested language
    /// are returned, before the 100-row LIMIT.
    fn search_nodes_by_name_with_lang(
        &self,
        name: &str,
        lang: &str,
    ) -> Result<Vec<Node>, StoreError> {
        let _span =
            tracing::debug_span!("store.search_nodes_by_name_with_lang", query = name, lang)
                .entered();
        self.run_name_search(Self::NAME_MATCH_SUBSTRING, name, Some(lang))
    }

    /// Language-scoped variant of [`search_nodes_by_name_exact`].
    fn search_nodes_by_name_with_lang_exact(
        &self,
        name: &str,
        lang: &str,
    ) -> Result<Vec<Node>, StoreError> {
        let _span = tracing::debug_span!(
            "store.search_nodes_by_name_with_lang_exact",
            query = name,
            lang
        )
        .entered();
        self.run_name_search(Self::NAME_MATCH_BOUNDARY, name, Some(lang))
    }

    /// Leg B (RFC-023 §5.1): word-segmented BM25 query over `nodes_fts_words`.
    ///
    /// `match_expr` must already be built from `travsr_core::ident::segments()`
    /// output — `nodes_fts_words` is `unicode61`, which splits on punctuation
    /// but does not split camelCase/PascalCase on its own, so the caller
    /// (`fused_search_scored`) must pre-segment the query the same way the
    /// index was populated (one segmenter, by construction).
    ///
    /// `bm25(nodes_fts_words, W_SIG, W_PATH)`: a signature match outranks a
    /// path match for the same term (Evidence B). Negated like the trigram
    /// leg so higher scores mean better matches.
    fn fts_query_words_scored(&self, match_expr: &str) -> AnyResult<Vec<(Node, f32)>> {
        let sql = format!(
            "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                    n.kind, n.package, n.line, n.end_line, \
                    -bm25(nodes_fts_words, {sig_w}, {path_w}) \
             FROM nodes_fts_words \
             JOIN nodes_fts_words_map m ON nodes_fts_words.rowid = m.node_id \
             JOIN nodes n ON n.id = m.node_id \
             WHERE nodes_fts_words MATCH ?1 \
               AND n.is_noise = 0 \
             ORDER BY bm25(nodes_fts_words, {sig_w}, {path_w}) \
             LIMIT 50",
            sig_w = Self::FTS_WORDS_W_SIG,
            path_w = Self::FTS_WORDS_W_PATH,
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("preparing Leg B word FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                let bm25: f64 = row.get(10).unwrap_or(0.0);
                Ok((
                    Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    },
                    bm25 as f32,
                ))
            })
            .context("executing Leg B word FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding Leg B word FTS5 row")?);
        }
        Ok(out)
    }

    /// Language-filtered variant of [`fts_query_words_scored`].
    fn fts_query_words_scored_with_lang(
        &self,
        match_expr: &str,
        lang: &str,
    ) -> AnyResult<Vec<(Node, f32)>> {
        let sql = format!(
            "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                    n.kind, n.package, n.line, n.end_line, \
                    -bm25(nodes_fts_words, {sig_w}, {path_w}) \
             FROM nodes_fts_words \
             JOIN nodes_fts_words_map m ON nodes_fts_words.rowid = m.node_id \
             JOIN nodes n ON n.id = m.node_id \
             WHERE nodes_fts_words MATCH ?1 \
               AND n.language = ?2 \
               AND n.is_noise = 0 \
             ORDER BY bm25(nodes_fts_words, {sig_w}, {path_w}) \
             LIMIT 50",
            sig_w = Self::FTS_WORDS_W_SIG,
            path_w = Self::FTS_WORDS_W_PATH,
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("preparing lang-filtered Leg B word FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr, lang], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                let bm25: f64 = row.get(10).unwrap_or(0.0);
                Ok((
                    Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    },
                    bm25 as f32,
                ))
            })
            .context("executing lang-filtered Leg B word FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding lang-filtered Leg B word FTS5 row")?);
        }
        Ok(out)
    }

    /// FTS5 query that returns `(Node, bm25_score)` pairs.
    ///
    /// SQLite `bm25()` is negated so higher scores mean better matches (positive = good).
    /// Typical range: 0.05 (weak) to 5.0+ (strong match on a small corpus).
    fn fts_query_nodes_scored(&self, match_expr: &str) -> AnyResult<Vec<(Node, f32)>> {
        // #478: is_noise = 0, see the lang-filtered sibling's comment.
        let sql = "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                          n.kind, n.package, n.line, n.end_line, -bm25(nodes_fts) \
                   FROM nodes_fts \
                   JOIN nodes_fts_map m ON nodes_fts.rowid = m.node_id \
                   JOIN nodes n ON n.id = m.node_id \
                   WHERE nodes_fts MATCH ?1 \
                     AND n.is_noise = 0 \
                   ORDER BY bm25(nodes_fts) \
                   LIMIT 50";
        let mut stmt = self
            .conn
            .prepare(sql)
            .context("preparing scored FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                let bm25: f64 = row.get(10).unwrap_or(0.0);
                Ok((
                    Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    },
                    bm25 as f32,
                ))
            })
            .context("executing scored FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding scored FTS5 row")?);
        }
        Ok(out)
    }

    /// Scored variant of [`search_nodes_fuzzy`]: returns `(Node, score)` pairs where
    /// score is a positive, BM25-scale relevance float (higher = better).
    ///
    /// #393: this is the `get_context`/`ask` seed path. It runs through
    /// [`fused_search_scored`] with `include_embed = false` — get_context has its
    /// own KNN seed channel, so folding the embed stage in here would double-count
    /// the semantic signal (plan §5.1). Scores stay BM25-scale (RRF only reorders,
    /// it does not rescore) so the downstream relevance/abstention floor that reads
    /// the top score remains calibrated.
    ///
    /// #478: returns [`FusedHit`] (not a bare tuple) — this is the one caller
    /// (`travsr-mcp`'s lexical loop) that reads `bm25_natural`/`exact_rank` to
    /// fix the boundary-evidence bug by topology rather than by a discount.
    pub fn search_nodes_fuzzy_scored(&self, query: &str) -> Result<Vec<FusedHit>, StoreError> {
        self.fused_search_scored(query, None, false, false)
    }

    /// Returns the number of nodes whose indexed word set contains `token`
    /// (#478 RFC-023 §5.5).
    ///
    /// Reads `nodes_words_vocab` (`fts5vocab` 'row' mode over `nodes_fts_words`)
    /// — FTS5-maintained document frequency over the same word segmentation
    /// (`travsr_core::ident::segments`) that built the index and that the
    /// boundary predicate (`ident::contains_token`) reasons about. Three
    /// reasons this replaced a `signature LIKE '%tok%'` scan:
    ///   1. Correct: the old scan counted substrings (`wal` matched 22 nodes
    ///      containing "walk"/"walker"/etc. for a token appearing in 1).
    ///   2. Consistent: same segmentation that built the index, not a fourth
    ///      independent notion of "this token matches this node".
    ///   3. Fast: O(1) indexed lookup, replacing a full `nodes` table scan.
    ///
    /// Used as IDF denominator for per-token anchor specificity: high `freq`
    /// means the token is generic ("get", "run") and should contribute low
    /// weight even when it matches a real symbol.
    ///
    /// Returns `None` only when the token occurs in zero nodes. A token whose
    /// identifier segments are all shorter than 3 bytes (e.g. the Ruby class
    /// `UI`) is never in the word vocab (min segment len 3), so it cannot be
    /// measured here; rather than conflate "unindexed" with "occurs nowhere"
    /// (which floored a unique short symbol to the generic IDF and abstained on
    /// an exact rank-0 match — #778), it falls back to the exact-leaf-name
    /// document frequency ([`Self::exact_leaf_name_count`]): how many definition
    /// nodes are named *exactly* this token. That is the same scale the exact
    /// leg (`search_nodes_by_name_exact`, raw score 1.0) treats as a match, so a
    /// genuinely unique short symbol reads as rare and a common member name
    /// (borne by many `*.name` members) reads as generic — the segment vocab
    /// above cannot distinguish them because it never indexed the short token.
    /// A genuine zero still returns `None`, so nonsense short tokens abstain.
    ///
    /// `token` may itself be a compound identifier the caller has not
    /// pre-segmented (`tokenize_query`'s content tokens preserve `_`, e.g.
    /// `"dispatch_tool_call"` stays one token, unlike `ident::segments`'
    /// index-time output). Segmenting here and taking the *minimum*
    /// document frequency across the token's own segments (dropping any
    /// sub-segment `< 3` bytes, same as `node_fts_words` at write time — those
    /// are never indexed and carry no signal either way) approximates the
    /// compound's specificity from its rarest indexed part — the same
    /// reasoning `ident::contains_token`'s contiguous-run match uses — and
    /// keeps this function's contract single-word-token-compatible (a token
    /// with no internal delimiters segments to itself, so plain-word
    /// behaviour is unchanged). Absent if *every* indexable segment is absent
    /// from the vocabulary.
    pub fn symbol_frequency(&self, token: &str) -> Result<Option<usize>, StoreError> {
        (|| -> AnyResult<Option<usize>> {
            if token.is_empty() {
                return Ok(None);
            }
            let segs: Vec<String> = travsr_core::ident::segments(token)
                .into_iter()
                .filter(|s| s.len() >= 3)
                .collect();
            if segs.is_empty() {
                // #778: a token whose identifier segments are all < 3 chars (e.g.
                // the Ruby class `UI`) never enters `nodes_words_vocab` (min segment
                // len 3), so the word-vocab path below cannot measure it. Returning
                // `None` here let the caller fabricate `freq = n_total`, scoring a
                // unique 2-char symbol as maximally generic (the idf floor) — the
                // exact inverse of the truth, so `ask "UI"` abstained on an exact
                // rank-0 match. Fall back to the exact-leaf-name document frequency:
                // how many definition nodes are named *exactly* this token. A real,
                // small count (`UI` -> 1) makes the token rare and lets it ground; a
                // common member name borne by many `*.name` members counts them all
                // and stays generic; a genuine zero (a nonsense short token) stays
                // `None` so it is still treated as generic and abstains.
                let count = self.exact_leaf_name_count(token)?;
                return Ok((count > 0).then_some(count));
            }
            let mut min_doc: Option<i64> = None;
            for seg in &segs {
                let doc: Option<i64> = self
                    .conn
                    .query_row(
                        "SELECT doc FROM nodes_words_vocab WHERE term = ?1",
                        params![seg],
                        |r| r.get(0),
                    )
                    .optional()
                    .context("reading nodes_words_vocab for symbol_frequency")?;
                let Some(d) = doc else {
                    return Ok(None);
                };
                min_doc = Some(min_doc.map_or(d, |m: i64| m.min(d)));
            }
            Ok(min_doc.map(|d| d.max(0) as usize))
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// #778: how many definition nodes are named *exactly* `token` — i.e.
    /// `token` is the leaf of their `kind:Qualified.leaf` signature
    /// (`class:UI` -> `UI`, `method:Widget.id` -> `id`), case-insensitively.
    /// This matches the exact leg's raw-score-1.0 name semantics
    /// (`search_nodes_by_name_exact`), so it is the same scale the anchor path
    /// treats as an exact match — unlike a substring or a segment count.
    ///
    /// The distinction that matters for the [`Self::symbol_frequency`] fallback:
    /// `NAME_MATCH_BOUNDARY`'s unqualified tail forms cannot see a qualified
    /// member (`method:Widget.id` is invisible to a bare-`id` boundary match)
    /// yet its prefix form over-counts (`id%` catches `Identifier`), so it both
    /// fabricates rarity for a common member name and inflates a unique one.
    /// The two leaf clauses below (`'%:' || tok` for an unqualified body,
    /// `'%.' || tok` for a qualified leaf) count exactly the nodes whose own
    /// name is the token, and nothing else.
    ///
    /// Bounded by [`Self::LEAF_NAME_COUNT_CAP`]: only the rare/generic
    /// distinction (via the resulting IDF band) is used downstream, so the true
    /// count above the cap is irrelevant, and the `LIMIT` early-stops the scan
    /// for a pathologically common short name. At the cap the count is
    /// saturated to the corpus size (see the return below) rather than reported
    /// as the truncated `LIMIT` value, so a name common enough to hit the cap
    /// floors IDF to "generic" instead of reading as specific. Returns 0 when no
    /// node bears the exact name — the caller keeps `None` on that genuine zero.
    ///
    /// `_` and `%` in `token` are escaped so a token like `a_b` is matched
    /// literally, not as a LIKE wildcard. Used only as the `symbol_frequency`
    /// fallback for tokens too short (< 3-char segments) to appear in
    /// `nodes_words_vocab`; there is no index for the trailing-leaf match, so it
    /// is a bounded scan on that cold, short-token path only.
    fn exact_leaf_name_count(&self, token: &str) -> AnyResult<usize> {
        // Escape SQL LIKE metacharacters so the token is matched literally,
        // paired with `ESCAPE '\'` below. `\` itself is escaped first.
        fn escape_like(s: &str) -> String {
            let mut out = String::with_capacity(s.len());
            for c in s.chars() {
                if matches!(c, '\\' | '%' | '_') {
                    out.push('\\');
                }
                out.push(c);
            }
            out
        }
        let escaped = escape_like(token);
        // `'%:' || ?1` matches an unqualified body (`class:UI`); `'%.' || ?1` a
        // qualified leaf (`method:Widget.id`). Wrapped in a `LIMIT`-ed subquery
        // so the scan stops after the cap is reached.
        let sql = "SELECT count(*) FROM (\
                     SELECT 1 FROM nodes \
                     WHERE (signature LIKE '%:' || ?1 ESCAPE '\\' \
                         OR signature LIKE '%.' || ?1 ESCAPE '\\') \
                       AND kind != 'doc-chunk' \
                     LIMIT ?2)";
        let n: i64 = self
            .conn
            .query_row(
                sql,
                params![escaped, Self::LEAF_NAME_COUNT_CAP as i64],
                |r| r.get(0),
            )
            .context("counting exact leaf-name matches for symbol_frequency")?;
        // At the cap the true count is unknown but is >= the cap: the `LIMIT`
        // truncated it. A truncated count reads as *specific* through
        // `idf_weight`, which does not saturate anywhere near the cap, so a
        // short leaf name borne by tens of thousands of nodes would clear the
        // anchor-emit cut — the exact inversion this fallback exists to prevent.
        // Saturate to the corpus size N so IDF floors to "generic" instead;
        // below the cap the count is exact. Only the capped branch touches the
        // store again.
        Ok(if n >= Self::LEAF_NAME_COUNT_CAP as i64 {
            self.total_node_count()?
        } else {
            n.max(0) as usize
        })
    }

    /// Returns the total number of nodes in the graph.
    ///
    /// Used as the IDF corpus size N in `idf_weight(freq, N)`.
    pub fn total_node_count(&self) -> Result<usize, StoreError> {
        (|| -> AnyResult<usize> {
            let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM nodes")?;
            let count: i64 = stmt.query_row([], |r| r.get(0))?;
            Ok(count.max(0) as usize)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Language-scoped variant of [`fts_query_nodes`].
    /// Appends `AND n.language = ?2` before the 50-row FTS LIMIT.
    fn fts_query_nodes_with_lang(&self, match_expr: &str, lang: &str) -> AnyResult<Vec<Node>> {
        let sql = "SELECT n.id, n.corpus, n.root, n.path, n.language, n.signature, \
                          n.kind, n.package, n.line, n.end_line \
                   FROM nodes_fts \
                   JOIN nodes_fts_map m ON nodes_fts.rowid = m.node_id \
                   JOIN nodes n ON n.id = m.node_id \
                   WHERE nodes_fts MATCH ?1 \
                     AND n.language = ?2 \
                   ORDER BY bm25(nodes_fts) \
                   LIMIT 50";
        let mut stmt = self
            .conn
            .prepare(sql)
            .context("preparing lang-filtered FTS5 query")?;
        let rows = stmt
            .query_map(params![match_expr, lang], |row| {
                let id = i64_to_node_id(row.get::<_, i64>(0)?);
                let vname = VName::new(
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                );
                let kind: String = row.get(6)?;
                let package: String = row.get(7)?;
                let line: Option<i64> = row.get(8)?;
                let end_line: Option<i64> = row.get(9)?;
                Ok(Node {
                    id,
                    vname,
                    kind,
                    package,
                    line: line.and_then(|l| u32::try_from(l).ok()),
                    end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                    test_role: TestRole::None,
                })
            })
            .context("executing lang-filtered FTS5 query")?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.context("decoding lang-filtered FTS5 row")?);
        }
        Ok(out)
    }

    /// Language-filtered variant of [`search_nodes_fuzzy`].
    ///
    /// When `lang_filter` is `None`, delegates to [`search_nodes_fuzzy`] unchanged.
    /// When `Some(lang)`, each of the 4 search steps applies `AND language = ?`
    /// at the SQL level — before the 50-result FTS cap — guaranteeing results
    /// are from the requested language only.
    pub fn search_nodes_fuzzy_filtered(
        &self,
        query: &str,
        lang_filter: Option<&str>,
        exact_only: bool,
    ) -> Result<Vec<Node>, StoreError> {
        // #393: thin wrapper over the fused core. `lang_filter = None` behaves
        // identically to [`search_nodes_fuzzy`]; `Some(l)` scopes every stage to
        // language `l`. `include_embed = true` (the `search_symbol` path).
        Ok(self
            .fused_search_scored(query, lang_filter, true, exact_only)?
            .into_iter()
            .map(|hit| hit.node)
            .collect())
    }

    /// L2-A: vocabulary-grounded token expansion (RFC-012 A1 Step 3).
    ///
    /// Scans `fts_vocab` for tokens with Jaccard(byte-trigrams) ≥ `L2A_JACCARD_THRESHOLD`
    /// similarity to ANY token in `t0_tokens`.  Returns only tokens that exist in the live
    /// vocabulary (refcount > 0), so no hallucinated token can reach the FTS index
    /// (PA C4 bright line).  The caller caps the merged arm list at 16.
    ///
    /// Cost: O(V · |t0_tokens|) where V = vocabulary size.  Acceptable at MVP
    /// scale; SymSpell/FST can replace for scale-out without changing the API.
    fn expand_query(&self, t0_tokens: &[String]) -> AnyResult<Vec<String>> {
        if t0_tokens.is_empty() {
            return Ok(vec![]);
        }

        // Load vocab tokens with refcount > 0.  Cap at 100 k to remain practical
        // on large graphs; L2-A is only called on a combined miss so frequency is low.
        let vocab: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT token FROM fts_vocab WHERE refcount > 0 LIMIT 100000")
                .context("preparing fts_vocab scan")?;
            let collected: Vec<String> = stmt
                .query_map([], |row| row.get(0))
                .context("executing fts_vocab scan")?
                .collect::<Result<_, _>>()
                .context("collecting fts_vocab tokens")?;
            collected
        };

        // Pre-build a set of T0 tokens for O(1) membership check.
        let t0_set: std::collections::HashSet<&str> =
            t0_tokens.iter().map(String::as_str).collect();

        // Single pass over vocab: a token is a candidate if any T0 token exceeds
        // the Jaccard threshold.  This avoids re-scanning vocab once per T0 token.
        let mut candidates: Vec<String> = Vec::new();
        for v_tok in &vocab {
            if v_tok.len() < 3 {
                continue;
            }
            if t0_set.contains(v_tok.as_str()) {
                continue; // already in T0 union
            }
            if candidates.iter().any(|c| c == v_tok) {
                continue;
            }
            if t0_tokens
                .iter()
                .any(|q| byte_trigram_jaccard(q, v_tok) >= L2A_JACCARD_THRESHOLD)
            {
                candidates.push(v_tok.clone());
            }
        }

        candidates.sort();
        Ok(candidates)
    }

    /// #709: strict single-token typo correction over symbol *leaf names*.
    ///
    /// When a query token matched no symbol by name/substring, look for the one
    /// symbol whose leaf name (the segment after the last `.`/`:` of a signature)
    /// is a very close byte-trigram match, so a typo (`htpresponse`) can still
    /// ground to the real symbol (`HttpResponse`) through the normal anchor path
    /// instead of abstaining. Returns the matched symbol's leaf name in its
    /// original case (so the caller's segmenter and `symbol_frequency` see the
    /// real camelCase identifier), or `None` when nothing clears `min_jaccard` or
    /// the best match is ambiguous, a second *distinct* name sits within
    /// `AMBIGUITY_MARGIN` of the leader. Deliberately conservative: this is the
    /// exact-miss cold path and a false correction is worse than no correction;
    /// the cross-encoder reranker is the downstream precision backstop.
    ///
    /// Cost: O(V) distinct-signature scan (bounded by `LIMIT`), mirroring
    /// [`Self::expand_query`]. SymSpell / an FST index can replace this for
    /// scale-out without changing the signature.
    ///
    /// Single-token convenience wrapper over [`Self::fuzzy_correct_symbols`].
    /// Correcting more than one token in a request should call the batch form
    /// directly, so the scan runs once rather than once per token.
    pub fn fuzzy_correct_symbol(
        &self,
        token: &str,
        min_jaccard: f64,
    ) -> Result<Option<String>, StoreError> {
        Ok(self
            .fuzzy_correct_symbols(&[token], min_jaccard)?
            .remove(token))
    }

    /// Batch form of [`Self::fuzzy_correct_symbol`]: corrects every token in
    /// `tokens` against a single distinct-signature scan, returning only the
    /// tokens that produced an unambiguous correction.
    ///
    /// The scan, not the per-token trigram comparison, is what this call costs:
    /// one pass over up to `LIMIT` distinct signatures. Correcting per token
    /// therefore multiplied full-table scans by the number of unresolved tokens
    /// on the MCP query-serving path, where a single query can carry several.
    /// Scoring all tokens inside one pass is the same shape [`Self::expand_query`]
    /// already uses for vocabulary expansion.
    ///
    /// Each token is scored independently and answers exactly as the single-token
    /// form did, so batching changes cost, not results.
    pub fn fuzzy_correct_symbols(
        &self,
        tokens: &[&str],
        min_jaccard: f64,
    ) -> Result<std::collections::HashMap<String, String>, StoreError> {
        /// A correction only stands when the runner-up distinct name is at least
        /// this far behind, otherwise the typo is ambiguous and we ground nothing.
        const AMBIGUITY_MARGIN: f64 = 0.05;

        (|| -> AnyResult<std::collections::HashMap<String, String>> {
            // Trigram Jaccard is unstable on very short tokens; skip them so a
            // 3-4 char query fragment cannot ground to an unrelated symbol.
            // Deduplicated so a token repeated in one query is scored once.
            let mut lowered: Vec<(&str, String)> = Vec::new();
            for tok in tokens {
                if tok.len() < 5 {
                    continue;
                }
                let lower = tok.to_ascii_lowercase();
                if !lowered.iter().any(|(_, l)| *l == lower) {
                    lowered.push((*tok, lower));
                }
            }
            if lowered.is_empty() {
                return Ok(std::collections::HashMap::new());
            }

            // Per-token accumulator. `exact` records that some indexed leaf name
            // equals the token, which means it would have resolved exactly and so
            // must never be "corrected" to anything.
            struct Acc {
                exact: bool,
                by_name: std::collections::HashMap<String, (f64, String)>,
            }
            let mut accs: Vec<Acc> = lowered
                .iter()
                .map(|_| Acc {
                    exact: false,
                    by_name: std::collections::HashMap::new(),
                })
                .collect();

            let mut stmt = self
                .conn
                .prepare("SELECT DISTINCT signature FROM nodes LIMIT 200000")
                .context("preparing fuzzy_correct_symbol scan")?;
            let mut rows = stmt
                .query([])
                .context("executing fuzzy_correct_symbol scan")?;
            // Streamed: the previous form collected every signature into a `Vec`
            // before scoring, holding the whole scan in memory at once.
            while let Some(row) = rows
                .next()
                .context("collecting fuzzy_correct_symbol signatures")?
            {
                let sig: String = row.get(0)?;
                let leaf = travsr_core::ident::leaf_of(&sig);
                let leaf_lower = leaf.to_ascii_lowercase();
                for (i, (_, tok_lower)) in lowered.iter().enumerate() {
                    if accs[i].exact {
                        continue;
                    }
                    if leaf_lower == *tok_lower {
                        accs[i].exact = true;
                        continue;
                    }
                    let j = byte_trigram_jaccard(tok_lower, &leaf_lower);
                    if j < min_jaccard {
                        continue;
                    }
                    // Best Jaccard per distinct lowercased leaf name; keep an
                    // original-case exemplar to return.
                    accs[i]
                        .by_name
                        .entry(leaf_lower.clone())
                        .and_modify(|e| {
                            if j > e.0 {
                                *e = (j, leaf.to_string());
                            }
                        })
                        .or_insert_with(|| (j, leaf.to_string()));
                }
            }

            let mut out = std::collections::HashMap::new();
            for (i, (tok, _)) in lowered.iter().enumerate() {
                if accs[i].exact {
                    continue;
                }
                // Only the leader and the runner-up decide the outcome, so track
                // them in one pass instead of sorting the whole candidate set.
                // Ordering matches the previous sort exactly: descending Jaccard,
                // ties broken by ascending name.
                let better = |a: &(f64, String), b: &(f64, String)| -> bool {
                    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
                };
                let mut best: Option<(f64, String)> = None;
                let mut second: Option<(f64, String)> = None;
                for cand in accs[i].by_name.values() {
                    if best.as_ref().map_or(true, |b| better(cand, b)) {
                        second = best.take();
                        best = Some(cand.clone());
                    } else if second.as_ref().map_or(true, |s| better(cand, s)) {
                        second = Some(cand.clone());
                    }
                }
                let Some((best_j, best_leaf)) = best else {
                    continue;
                };
                let ambiguous =
                    second.is_some_and(|(second_j, _)| best_j - second_j < AMBIGUITY_MARGIN);
                if !ambiguous {
                    out.insert((*tok).to_string(), best_leaf);
                }
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Increment `fts_vocab` refcounts for every token in `tokens_str` (space-separated).
    /// Skips tokens shorter than 3 bytes (FTS5 trigram minimum).
    fn vocab_increment(conn: &Connection, tokens_str: &str) -> AnyResult<()> {
        for tok in tokens_str.split_whitespace() {
            if tok.len() < 3 {
                continue;
            }
            conn.execute(
                "INSERT INTO fts_vocab(token, refcount) VALUES(?1, 1) \
                 ON CONFLICT(token) DO UPDATE SET refcount = refcount + 1",
                params![tok],
            )
            .context("incrementing fts_vocab refcount")?;
        }
        Ok(())
    }

    /// Return the `fts_vocab` refcount for `token`, or `None` if the token is absent.
    /// Used by integration tests to verify L2-A refcount maintenance (AC e).
    #[doc(hidden)]
    pub fn fts_vocab_refcount(&self, token: &str) -> AnyResult<Option<i64>> {
        self.conn
            .query_row(
                "SELECT refcount FROM fts_vocab WHERE token = ?1",
                params![token],
                |row| row.get(0),
            )
            .optional()
            .context("querying fts_vocab refcount")
    }

    /// Decrement `fts_vocab` refcounts for every token in `tokens_str` (space-separated).
    /// Uses MAX(0, refcount - 1) to guard against underflow on out-of-order deletes.
    /// Skips tokens shorter than 3 bytes — matching the `vocab_increment` guard so
    /// short tokens never trigger a no-op UPDATE round-trip.
    fn vocab_decrement(conn: &Connection, tokens_str: &str) -> AnyResult<()> {
        for tok in tokens_str.split_whitespace() {
            if tok.len() < 3 {
                continue;
            }
            conn.execute(
                "UPDATE fts_vocab SET refcount = MAX(0, refcount - 1) WHERE token = ?1",
                params![tok],
            )
            .context("decrementing fts_vocab refcount")?;
        }
        Ok(())
    }

    /// Idempotent vocab backfill called once after migrations at `open()` /
    /// `open_in_memory()`.  Gate: if `fts_vocab` is non-empty the table is
    /// assumed up-to-date and we return immediately.  On first open after the
    /// v10 migration ships, reconstructs refcounts from the existing
    /// `nodes_fts_map` so pre-v10 databases get a correct vocabulary.
    fn backfill_vocab_if_needed(&mut self) -> AnyResult<()> {
        let vocab_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM fts_vocab", [], |r| r.get(0))
            .context("counting fts_vocab for backfill gate")?;

        if vocab_count > 0 {
            return Ok(()); // already populated
        }

        let token_strings: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT tokens FROM nodes_fts_map")
                .context("preparing nodes_fts_map scan for vocab backfill")?;
            let collected: Vec<String> = stmt
                .query_map([], |row| row.get(0))
                .context("executing nodes_fts_map scan")?
                .collect::<Result<_, _>>()
                .context("collecting token strings for vocab backfill")?;
            collected
        };

        if token_strings.is_empty() {
            return Ok(());
        }

        let tx = self
            .conn
            .transaction()
            .context("starting vocab backfill transaction")?;
        for ts in &token_strings {
            Self::vocab_increment(&tx, ts).context("vocab_increment during backfill")?;
        }
        tx.commit()
            .context("committing vocab backfill transaction")?;

        tracing::info!(nodes = token_strings.len(), "search vocabulary updated");
        Ok(())
    }

    /// Rebuild `nodes_fts` and `fts_vocab` from `nodes_fts_map` in one pass.
    ///
    /// Called by `init_repo_with_progress` after all bulk batches are flushed.
    /// Produces an identical end-state to per-node `put_node_fts` calls but
    /// avoids per-node FTS5 inverted-index writes — the dominant bottleneck on
    /// large repos (kubernetes: ~12M FTS ops → ~10 min, bulk rebuild: ~1 pass).
    ///
    /// Safe to call on an empty store (no-op) or a partially-written store
    /// (idempotent: clears and repopulates both tables).
    pub fn rebuild_fts_from_map(&mut self) -> anyhow::Result<()> {
        // Phase 1: insert FTS entries only for nodes written in this bulk run.
        // _bulk_fts_pending (created by begin_bulk_fts_tracking, populated by
        // put_node_fts_map_only) holds exactly those node IDs. Unchanged files
        // are excluded because they were hash-skipped and never reached
        // put_node_fts_map_only, so their existing FTS entries stay untouched.
        //
        // Contentless FTS5 tables forbid DELETE FROM and the 'rebuild' command.
        // Retracting non-existent entries corrupts the index, so we filter by
        // _bulk_fts_pending rather than clear-and-rebuild.
        let t_fts5 = std::time::Instant::now();
        {
            let tx = self
                .conn
                .transaction()
                .context("starting FTS rebuild transaction")?;
            tx.execute_batch(
                "INSERT INTO nodes_fts(rowid, tokens) \
                   SELECT m.node_id, m.tokens FROM nodes_fts_map m \
                   WHERE EXISTS ( \
                     SELECT 1 FROM _bulk_fts_pending p WHERE p.node_id = m.node_id \
                   ); \
                 INSERT INTO nodes_fts_words(rowid, sig, path) \
                   SELECT m.node_id, m.sig_words, m.path_words FROM nodes_fts_words_map m \
                   WHERE EXISTS ( \
                     SELECT 1 FROM _bulk_fts_pending p WHERE p.node_id = m.node_id \
                   ); \
                 DROP TABLE IF EXISTS _bulk_fts_pending;",
            )
            .context("bulk-inserting nodes_fts + nodes_fts_words from pending nodes")?;
            tx.commit().context("committing FTS rebuild")?;
        }
        tracing::info!(
            elapsed_ms = t_fts5.elapsed().as_millis(),
            "TIMING: FTS5 bulk INSERT done"
        );

        // Phase 2: rebuild fts_vocab by counting tokens in Rust, then
        // bulk-inserting unique token counts. This replaces ~6–10M per-token
        // SQL upserts with ~50k–200k unique-token inserts (kubernetes estimate).
        let t_vocab_scan = std::time::Instant::now();
        let token_strings: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT tokens FROM nodes_fts_map")
                .context("preparing nodes_fts_map scan for vocab rebuild")?;
            let collected: Vec<String> = stmt
                .query_map([], |row| row.get(0))
                .context("executing nodes_fts_map scan")?
                .collect::<rusqlite::Result<_>>()
                .context("collecting token strings for vocab rebuild")?;
            collected
        };
        tracing::info!(
            elapsed_ms = t_vocab_scan.elapsed().as_millis(),
            rows = token_strings.len(),
            "TIMING: nodes_fts_map scan done"
        );

        // Count token frequencies in Rust — O(nodes × tokens_per_node).
        let t_count = std::time::Instant::now();
        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for ts in &token_strings {
            for tok in ts.split_whitespace() {
                if tok.len() >= 3 {
                    *counts.entry(tok.to_string()).or_insert(0) += 1;
                }
            }
        }
        tracing::info!(
            elapsed_ms = t_count.elapsed().as_millis(),
            unique_tokens = counts.len(),
            "TIMING: token count done"
        );

        if counts.is_empty() {
            return Ok(());
        }

        let t_vocab_write = std::time::Instant::now();
        let tx = self
            .conn
            .transaction()
            .context("starting fts_vocab rebuild transaction")?;
        tx.execute("DELETE FROM fts_vocab", [])
            .context("clearing fts_vocab for rebuild")?;
        for (token, refcount) in &counts {
            tx.execute(
                "INSERT INTO fts_vocab(token, refcount) VALUES(?1, ?2)",
                params![token, *refcount as i64],
            )
            .context("inserting rebuilt fts_vocab row")?;
        }
        tx.commit().context("committing fts_vocab rebuild")?;
        tracing::info!(
            elapsed_ms = t_vocab_write.elapsed().as_millis(),
            rows = counts.len(),
            "TIMING: fts_vocab write done"
        );

        tracing::info!(
            nodes = token_strings.len(),
            unique_tokens = counts.len(),
            "bulk init: FTS + vocab rebuild complete"
        );
        Ok(())
    }

    // ── Dynamic synonym table (RFC-012 A2 F1) ────────────────────────────────

    /// Seed `fts_synonyms` from the compile-time static defaults if the table is empty.
    /// Called once at `open()` / `open_in_memory()` after migrations.
    /// Idempotent: if any rows exist, returns immediately without touching the table.
    fn seed_synonyms_if_empty(&mut self) -> AnyResult<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM fts_synonyms", [], |r| r.get(0))
            .context("counting fts_synonyms")?;
        if count > 0 {
            return Ok(());
        }
        let tx = self.conn.transaction()?;
        for (term, aliases) in crate::seed_lexicon::SYNONYMS {
            for alias in *aliases {
                tx.execute(
                    "INSERT OR IGNORE INTO fts_synonyms(term, alias) VALUES(?1, ?2)",
                    params![term, alias],
                )?;
            }
        }
        tx.commit()?;
        tracing::info!("loaded default search synonyms");
        Ok(())
    }

    /// Add a synonym pair. Rejects if the table already has ≥200 rows.
    pub fn synonym_add(&mut self, term: &str, alias: &str) -> AnyResult<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM fts_synonyms", [], |r| r.get(0))
            .context("counting fts_synonyms before add")?;
        anyhow::ensure!(
            count < 200,
            "fts_synonyms is full (200 rows). Use `travsr synonym remove` to make space."
        );
        self.conn.execute(
            "INSERT OR IGNORE INTO fts_synonyms(term, alias) VALUES(?1, ?2)",
            params![term, alias],
        )?;
        Ok(())
    }

    /// Remove a synonym pair. No-op if the pair does not exist.
    pub fn synonym_remove(&mut self, term: &str, alias: &str) -> AnyResult<()> {
        self.conn.execute(
            "DELETE FROM fts_synonyms WHERE term = ?1 AND alias = ?2",
            params![term, alias],
        )?;
        Ok(())
    }

    /// Remove ALL aliases for `term`. No-op if the term has no aliases.
    pub fn synonym_remove_term(&mut self, term: &str) -> AnyResult<()> {
        self.conn
            .execute("DELETE FROM fts_synonyms WHERE term = ?1", params![term])?;
        Ok(())
    }

    /// Declaratively replace ALL aliases for `term` with exactly `aliases`.
    ///
    /// Atomic: the DELETE and every INSERT run inside a single transaction, so a
    /// crash mid-operation can never leave the term with its old aliases removed
    /// and only a partial new set (the failure mode of a separate remove + N adds).
    /// Rejects — and rolls back, leaving the table untouched — if the resulting
    /// row count would exceed the 200-row cap. The cap is evaluated once against
    /// the post-delete count, not per-insert.
    pub fn synonym_set(&mut self, term: &str, aliases: &[String]) -> AnyResult<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM fts_synonyms WHERE term = ?1", params![term])?;
        let count: i64 = tx
            .query_row("SELECT COUNT(*) FROM fts_synonyms", [], |r| r.get(0))
            .context("counting fts_synonyms before set")?;
        anyhow::ensure!(
            count + aliases.len() as i64 <= 200,
            "fts_synonyms would exceed 200 rows. Use `travsr synonym remove` to make space."
        );
        for alias in aliases {
            tx.execute(
                "INSERT OR IGNORE INTO fts_synonyms(term, alias) VALUES(?1, ?2)",
                params![term, alias],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// List all active synonym pairs as (term, alias) tuples.
    pub fn synonym_list(&self) -> AnyResult<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT term, alias FROM fts_synonyms ORDER BY term, alias")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    /// Reset `fts_synonyms` to the static defaults: delete all rows and re-seed.
    pub fn synonym_reset(&mut self) -> AnyResult<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch("DELETE FROM fts_synonyms")?;
        for (term, aliases) in crate::seed_lexicon::SYNONYMS {
            for alias in *aliases {
                tx.execute(
                    "INSERT OR IGNORE INTO fts_synonyms(term, alias) VALUES(?1, ?2)",
                    params![term, alias],
                )?;
            }
        }
        tx.commit()?;
        tracing::info!("reset search synonyms to the built-in defaults");
        Ok(())
    }

    /// Number of node deletions the embed sidecar has not consumed yet (#376 W2).
    ///
    /// The daemon's embed tick decides there is work to do from `embed_progress`,
    /// which is presence-only: a node whose content changed still *has* an
    /// embedding row, so coverage reads 100 % and no pass is ever spawned. The
    /// tombstone log is the only record that those vectors no longer match their
    /// nodes, and until a pass runs it is also the only thing keeping them from
    /// being deleted. Measured on this repo before the fix: 154 pending
    /// tombstones, 152 of them still holding a live vector, with coverage at
    /// 100 % and therefore no spawn scheduled — ever.
    pub fn pending_tombstones(&self) -> Result<u64, StoreError> {
        (|| -> AnyResult<u64> {
            let n: i64 = self
                .conn
                .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
                .context("counting node_tombstones")?;
            Ok(n as u64)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Doc-chunk nodes that have no `embed_text` yet (#376 W1).
    ///
    /// Kept separate from [`nodes_missing_embed_text`] because the daemon runs
    /// this one on the auto-embed path, where the full regeneration is
    /// deliberately avoided (it parses every source file in the repo and blocks
    /// for minutes). Markdown chunks are re-derived by a pure function of the
    /// file's bytes, so the doc-only subset is cheap enough to run before every
    /// spawn — and without it a doc chunk indexed by `travsr init` stays
    /// ineligible, i.e. silently absent from doc retrieval.
    pub fn doc_nodes_missing_embed_text(&self) -> Result<Vec<Node>, StoreError> {
        (|| -> AnyResult<Vec<Node>> {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line \
                     FROM nodes WHERE embed_text IS NULL AND kind = 'doc-chunk'",
                )
                .context("preparing doc_nodes_missing_embed_text query")?;
            let rows = stmt
                .query_map([], |row| {
                    let id = i64_to_node_id(row.get::<_, i64>(0)?);
                    let vname = VName::new(
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    );
                    let kind: String = row.get(6)?;
                    let package: String = row.get(7)?;
                    let line: Option<i64> = row.get(8)?;
                    let end_line: Option<i64> = row.get(9)?;
                    Ok(Node {
                        id,
                        vname,
                        kind,
                        package,
                        line: line.and_then(|l| u32::try_from(l).ok()),
                        end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                        test_role: TestRole::None,
                    })
                })
                .context("executing doc_nodes_missing_embed_text query")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("decoding doc_nodes_missing_embed_text row")?);
            }
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// SC-H1: cap the `node_tombstones` table so it cannot grow unbounded.
    ///
    /// Tombstones are at-least-once: the embed sidecar consumes and acks them
    /// on each reindex pass. If the sidecar is offline for a long time (or the
    /// table is never consumed) the table accumulates indefinitely. This GC
    /// trims in two passes:
    ///
    /// 1. Delete rows older than `max_age_secs` (default 7 days). After a
    ///    full init_repo the sidecar rebuilds all embeddings from scratch, so
    ///    any tombstone older than one full cycle is redundant.
    /// 2. If the table still has > `max_rows` rows after the age trim, keep
    ///    only the newest `max_rows` (the embed sidecar re-derives stale
    ///    entries on the next full reindex).
    ///
    /// Consistency window: tombstones between the GC cut-off and sidecar ack
    /// may be missed. Acceptable because a full init_repo rebuild covers any
    /// gap. Document this as "eventual" rather than "guaranteed once".
    ///
    /// L3: since ack *is* deletion, every tombstone this prunes before the
    /// embed sidecar ever consumed it is a potentially-missed invalidation —
    /// but most pruned tombstones are harmless: the node they name may already
    /// be gone (covered by the orphan sweep already) or may never have had a
    /// vector at all. The second return value is the honest subset that
    /// actually represents risk: pruned tombstones whose node **still exists**
    /// and **has an embedding row** — i.e. a vector that may now be stale with
    /// nothing left to invalidate it.
    pub fn prune_tombstones(
        &mut self,
        max_age_secs: u64,
        max_rows: u64,
    ) -> anyhow::Result<(u64, u64)> {
        // ATTACH/DETACH must bracket the transaction from *outside* it: SQLite
        // refuses `DETACH` while a transaction is open ("database edb is
        // locked"), and a swallowed DETACH leaves the alias bound, so the next
        // `ATTACH edb` on this connection fails "already in use" — which would
        // also break `embed_progress`, since it uses the same alias, and so
        // silently stop the daemon from ever spawning an embed pass again.
        // `embed_progress` gets this right; mirror it here.
        let embed_db_path = self.embed_db_path.clone();
        let edb_attached = embed_db_path
            .as_deref()
            .filter(|p| p.exists())
            .and_then(|p| p.to_str())
            // A repo path may legally contain `'`; doubling it keeps the
            // literal well-formed (and `execute_batch` runs *every* statement
            // in the string, so an unescaped quote is a live injection sink).
            .map(|p| {
                let escaped = p.replace('\'', "''");
                self.conn
                    .execute_batch(&format!("ATTACH DATABASE '{escaped}' AS edb"))
            })
            .transpose()
            .map_err(|e| {
                tracing::warn!("attaching embed.db for the tombstone at-risk count failed: {e}");
            })
            .unwrap_or(None)
            .is_some();

        let out = self.prune_tombstones_locked(max_age_secs, max_rows, edb_attached);

        if edb_attached {
            if let Err(e) = self.conn.execute_batch("DETACH DATABASE edb") {
                tracing::warn!("detaching embed.db after tombstone prune failed: {e}");
            }
        }
        out
    }

    /// Body of [`Self::prune_tombstones`], run with `edb` already attached (or
    /// known absent). Split out so the caller can DETACH on every exit path
    /// without holding a borrow across the transaction.
    fn prune_tombstones_locked(
        &mut self,
        max_age_secs: u64,
        max_rows: u64,
        edb_attached: bool,
    ) -> anyhow::Result<(u64, u64)> {
        let tx = self.conn.transaction()?;
        let cutoff = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64)
            - max_age_secs as i64;

        // COUNT(DISTINCT t.rowid): a node embedded under more than one model
        // has one `node_embeddings` row per model, and multi-model rows are
        // exactly what this release makes normal (`travsr embed gc`). Counting
        // raw join rows would let `at_risk` exceed the tombstones pruned.
        const AT_RISK_JOIN: &str = "JOIN nodes n ON n.id = t.node_id \
             JOIN edb.node_embeddings e ON e.node_id = t.node_id";

        let aged_at_risk: u64 = if edb_attached {
            tx.query_row(
                &format!(
                    "SELECT COUNT(DISTINCT t.rowid) FROM node_tombstones t {AT_RISK_JOIN} \
                     WHERE t.deleted_at < ?1"
                ),
                rusqlite::params![cutoff],
                |r| r.get::<_, i64>(0),
            )? as u64
        } else {
            0
        };
        let aged = tx.execute(
            "DELETE FROM node_tombstones WHERE deleted_at < ?1",
            rusqlite::params![cutoff],
        )? as u64;

        // Count remaining rows.
        let remaining: i64 =
            tx.query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))?;
        let (size_pruned, size_at_risk) = if remaining as u64 > max_rows {
            let size_at_risk: u64 = if edb_attached {
                tx.query_row(
                    &format!(
                        "SELECT COUNT(DISTINCT t.rowid) FROM node_tombstones t {AT_RISK_JOIN} \
                         WHERE t.rowid NOT IN \
                         (SELECT rowid FROM node_tombstones ORDER BY deleted_at DESC LIMIT ?1)"
                    ),
                    rusqlite::params![max_rows as i64],
                    |r| r.get::<_, i64>(0),
                )? as u64
            } else {
                0
            };
            let deleted = tx.execute(
                "DELETE FROM node_tombstones WHERE rowid NOT IN \
                 (SELECT rowid FROM node_tombstones ORDER BY deleted_at DESC LIMIT ?1)",
                rusqlite::params![max_rows as i64],
            )? as u64;
            (deleted, size_at_risk)
        } else {
            (0, 0)
        };
        tx.commit()?;
        let total = aged + size_pruned;
        let at_risk = aged_at_risk + size_at_risk;
        if total > 0 {
            tracing::debug!(aged, size_pruned, at_risk, "pruned node_tombstones");
        }
        Ok((total, at_risk))
    }
}

// ── Store ─────────────────────────────────────────────────────────────────────

impl Store for SqliteStore {
    fn put_node(&mut self, node: &Node) -> Result<NodeId, StoreError> {
        let id_i64 = node_id_to_i64(node.id);
        // Wrap node upsert + FTS write in one transaction so a crash between
        // the two writes cannot leave nodes and nodes_fts_map out of sync.
        // (The backfill gate in open() self-heals any gap, but atomicity is
        // preferable.)  Callers use bare autocommit loops — no nesting conflict.
        (|| -> AnyResult<()> {
            let tx = self
                .conn
                .transaction()
                .context("starting put_node transaction")?;
            tx.execute(
                "INSERT INTO nodes(id, corpus, root, path, language, signature, kind, package, line, end_line, is_noise, test_role) \
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) \
                 ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, \
                 package = excluded.package, \
                 line = COALESCE(excluded.line, nodes.line), \
                 end_line = COALESCE(excluded.end_line, nodes.end_line), \
                 is_noise = excluded.is_noise, \
                 test_role = excluded.test_role",
                params![
                    id_i64,
                    node.vname.corpus,
                    node.vname.root,
                    node.vname.path,
                    node.vname.language,
                    node.vname.signature,
                    node.kind,
                    node.package,
                    node.line.map(|l| l as i64),
                    node.end_line.map(|l| l as i64),
                    travsr_core::noise::is_structural_noise(node),
                    node.test_role.as_i64(),
                ],
            )
            .context("inserting node")?;
            Self::put_node_fts(&tx, node).context("put_node_fts")?;
            Self::put_node_fts_words(&tx, node).context("put_node_fts_words")?;
            tx.commit().context("committing put_node transaction")?;
            Ok(())
        })()
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(node.id)
    }

    fn put_edge(&mut self, edge: &Edge) -> Result<(), StoreError> {
        // Tree-sitter edges use DO NOTHING on conflict: they must never demote
        // an existing 'lsif' row (ADR-002). DO NOTHING is equivalent to the
        // verbose CASE expression but is explicit about intent.
        self.conn
            .execute(
                "INSERT INTO edges(src, dst, kind, provenance, confidence) VALUES(?1, ?2, ?3, 'tree-sitter', ?4)
                 ON CONFLICT(src, dst, kind) DO NOTHING",
                params![
                    node_id_to_i64(edge.src),
                    node_id_to_i64(edge.dst),
                    edge.kind.as_str(),
                    edge.confidence.map(|c| c as i64),
                ],
            )
            .context("inserting tree-sitter edge")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    fn get_node(&self, id: NodeId) -> Result<Option<Node>, StoreError> {
        (|| -> AnyResult<Option<Node>> {
            let row = self
                .conn
                .query_row(
                    "SELECT corpus, root, path, language, signature, kind, package, line, end_line \
                     FROM nodes WHERE id = ?1",
                    params![node_id_to_i64(id)],
                    |row| {
                        let vname = VName::new(
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                        );
                        let kind: String = row.get(5)?;
                        let package: String = row.get(6)?;
                        let line: Option<i64> = row.get(7)?;
                        let end_line: Option<i64> = row.get(8)?;
                        Ok(Node {
                            id,
                            vname,
                            kind,
                            package,
                            line: line.and_then(|l| u32::try_from(l).ok()),
                            end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                            test_role: TestRole::None,
                        })
                    },
                )
                .optional()
                .context("querying node by id")?;
            Ok(row)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    fn iter_edges_from(&self, src: NodeId) -> Result<Vec<Edge>, StoreError> {
        let _span = tracing::debug_span!("store.iter_edges_from", src = src.0).entered();
        (|| -> AnyResult<Vec<Edge>> {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "SELECT dst, kind, confidence, provenance FROM edges WHERE src = ?1",
                )
                .context("preparing iter_edges_from query")?;
            let rows = stmt
                .query_map(params![node_id_to_i64(src)], |row| {
                    let dst_i64: i64 = row.get(0)?;
                    let kind_str: String = row.get(1)?;
                    let confidence: Option<i64> = row.get(2)?;
                    let provenance: String = row.get(3)?;
                    Ok((dst_i64, kind_str, confidence, provenance))
                })
                .context("executing iter_edges_from query")?;

            let mut out = Vec::new();
            for row in rows {
                let (dst_i64, kind_str, confidence, provenance) =
                    row.context("decoding edge row")?;
                let kind = EdgeKind::from_str(&kind_str)
                    .with_context(|| format!("unknown edge kind in storage: {kind_str}"))?;
                out.push(Edge {
                    src,
                    dst: i64_to_node_id(dst_i64),
                    kind,
                    confidence: confidence.map(|c| c as u8),
                    provenance: Some(provenance),
                });
            }
            tracing::debug!(edges_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    /// Indexed variant — uses `WHERE src = ?1 AND kind = ?2` so SQLite can
    /// satisfy the query from `idx_edges_src_kind_cov` (src, kind, dst) as an
    /// index-only scan, without a main-table row fetch. Overrides the trait
    /// default.
    ///
    /// Carries `provenance` (DEBT-75), like every other edge reader. No current
    /// caller consults it here — `travsr-retrieval` never reads the field, and
    /// `query.rs::next_edges` takes the deps direction through
    /// `iter_edges_from` — but returning `None` would not be a neutral omission:
    /// `query.rs::prov_of` maps `None` to `"tree-sitter"`, so an unlabelled edge
    /// is reported as *ratified truth* rather than as unknown. For a lane whose
    /// whole point is never presenting a `live` edge as ratified, a reader that
    /// silently mislabels the moment someone consults it is the wrong trade for
    /// one `String` per edge. The index tail carries the column instead, so the
    /// query stays index-only.
    fn iter_edges_from_kind(&self, src: NodeId, kind: EdgeKind) -> Result<Vec<Edge>, StoreError> {
        let _span =
            tracing::debug_span!("store.iter_edges_from_kind", src = src.0, kind = ?kind).entered();
        (|| -> AnyResult<Vec<Edge>> {
            let mut stmt = self
                .conn
                .prepare("SELECT dst, provenance FROM edges WHERE src = ?1 AND kind = ?2")
                .context("preparing iter_edges_from_kind query")?;
            let rows = stmt
                .query_map(params![node_id_to_i64(src), kind.as_str()], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .context("executing iter_edges_from_kind query")?;

            let mut out = Vec::new();
            for row in rows {
                let (dst_i64, provenance) = row.context("decoding iter_edges_from_kind row")?;
                out.push(Edge::new(src, i64_to_node_id(dst_i64), kind).with_provenance(provenance));
            }
            tracing::debug!(edges_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    fn iter_edges_from_batch(&self, srcs: &[NodeId]) -> Result<Vec<Edge>, StoreError> {
        if srcs.is_empty() {
            return Ok(Vec::new());
        }
        let _span =
            tracing::debug_span!("store.iter_edges_from_batch", count = srcs.len()).entered();
        (|| -> AnyResult<Vec<Edge>> {
            let placeholders = srcs.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT src, dst, kind, confidence, provenance FROM edges WHERE src IN ({placeholders})"
            );
            // prepare() not prepare_cached(): the SQL string varies per chunk length
            // (different number of '?' placeholders), so prepare_cached would create a
            // new cache entry for every distinct chunk size and never actually hit the
            // cache — defeating its purpose and polluting the LRU with O(EDGE_BATCH_SIZE)
            // distinct compiled statements.
            let mut stmt = self
                .conn
                .prepare(&sql)
                .context("preparing iter_edges_from_batch")?;
            let params: Vec<i64> = srcs.iter().map(|&id| node_id_to_i64(id)).collect();
            let rows = stmt
                .query_map(params_from_iter(params.iter()), |row| {
                    let src_i64: i64 = row.get(0)?;
                    let dst_i64: i64 = row.get(1)?;
                    let kind_str: String = row.get(2)?;
                    let confidence: Option<i64> = row.get(3)?;
                    let provenance: String = row.get(4)?;
                    Ok((src_i64, dst_i64, kind_str, confidence, provenance))
                })
                .context("executing iter_edges_from_batch")?;
            let mut out = Vec::new();
            for row in rows {
                let (src_i64, dst_i64, kind_str, confidence, provenance) =
                    row.context("decoding edge row")?;
                let kind = EdgeKind::from_str(&kind_str)
                    .with_context(|| format!("unknown edge kind in storage: {kind_str}"))?;
                out.push(Edge {
                    src: i64_to_node_id(src_i64),
                    dst: i64_to_node_id(dst_i64),
                    kind,
                    confidence: confidence.map(|c| c as u8),
                    provenance: Some(provenance),
                });
            }
            tracing::debug!(edges_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    fn iter_edges_to(&self, dst: NodeId) -> Result<Vec<Edge>, StoreError> {
        let _span = tracing::debug_span!("store.iter_edges_to", dst = dst.0).entered();
        (|| -> AnyResult<Vec<Edge>> {
            let mut stmt = self
                .conn
                .prepare("SELECT src, kind, provenance FROM edges WHERE dst = ?1")
                .context("preparing iter_edges_to query")?;
            let rows = stmt
                .query_map(params![node_id_to_i64(dst)], |row| {
                    let src_i64: i64 = row.get(0)?;
                    let kind_str: String = row.get(1)?;
                    let provenance: String = row.get(2)?;
                    Ok((src_i64, kind_str, provenance))
                })
                .context("executing iter_edges_to query")?;

            let mut out = Vec::new();
            for row in rows {
                let (src_i64, kind_str, provenance) = row.context("decoding edge row")?;
                let kind = EdgeKind::from_str(&kind_str)
                    .with_context(|| format!("unknown edge kind in storage: {kind_str}"))?;
                out.push(Edge::new(i64_to_node_id(src_i64), dst, kind).with_provenance(provenance));
            }
            tracing::debug!(edges_returned = out.len());
            Ok(out)
        })()
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    fn get_nodes(&self, ids: &[NodeId]) -> Result<Vec<Node>, StoreError> {
        // SQLite SQLITE_MAX_VARIABLE_NUMBER is 999 by default; chunk to stay within it.
        const CHUNK: usize = 999;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, corpus, root, path, language, signature, kind, package, line, end_line \
                 FROM nodes WHERE id IN ({placeholders})"
            );
            (|| -> AnyResult<()> {
                let mut stmt = self
                    .conn
                    .prepare(&sql)
                    .context("preparing get_nodes query")?;
                let id_params: Vec<i64> = chunk.iter().map(|id| node_id_to_i64(*id)).collect();
                let rows = stmt
                    .query_map(params_from_iter(id_params.iter()), |row| {
                        let id = i64_to_node_id(row.get::<_, i64>(0)?);
                        let vname = VName::new(
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, String>(5)?,
                        );
                        let kind: String = row.get(6)?;
                        let package: String = row.get(7)?;
                        let line: Option<i64> = row.get(8)?;
                        let end_line: Option<i64> = row.get(9)?;
                        Ok(Node {
                            id,
                            vname,
                            kind,
                            package,
                            line: line.and_then(|l| u32::try_from(l).ok()),
                            end_line: end_line.and_then(|l| u32::try_from(l).ok()),
                            test_role: TestRole::None,
                        })
                    })
                    .context("executing get_nodes query")?;
                for r in rows {
                    out.push(r.context("decoding get_nodes row")?);
                }
                Ok(())
            })()
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }
        Ok(out)
    }
}

/// G2 fallback: resolve the file node id for a SCIP ref whose enclosing
/// function could not be found.
///
/// Looks up the real Phase A file node by `(corpus, path, kind='file')` —
/// never reconstructs the VName hash, which silently diverges when fields
/// like `language` differ (the cause of 385K dangling edges on kubernetes).
/// If the path was never Phase-A-indexed (`.travsrignore` etc.), synthesizes
/// a file node matching the Phase A VName convention so the edge has a real
/// Minimal span descriptor fetched once per unique path for G2 attribution.
struct FnSpan {
    id: i64,
    line: i64,
    end_line: i64,
}

/// Fetch all function/method spans for a single `path` in one query.
///
/// Results are ordered by `(end_line - line) ASC, id ASC` so that
/// [`find_narrowest_enclosing`] can short-circuit on the first match — exactly
/// replicating the `ORDER BY … LIMIT 1` the per-ref SELECT used to do.
fn fetch_all_fn_spans(
    tx: &rusqlite::Transaction<'_>,
    corpus: &str,
    path: &str,
) -> anyhow::Result<Vec<FnSpan>> {
    let mut stmt = tx
        .prepare_cached(
            "SELECT id, line, end_line FROM nodes \
             WHERE corpus = ?1 AND path = ?2 \
               AND kind IN ('function', 'method', 'fn') \
               AND line IS NOT NULL AND end_line IS NOT NULL \
             ORDER BY (end_line - line) ASC, id ASC",
        )
        .context("fetch_all_fn_spans: prepare")?;
    let spans = stmt
        .query_map(params![corpus, path], |row| {
            Ok(FnSpan {
                id: row.get(0)?,
                line: row.get(1)?,
                end_line: row.get(2)?,
            })
        })
        .context("fetch_all_fn_spans: query")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("fetch_all_fn_spans: collect")?;
    Ok(spans)
}

/// Return the `id` of the narrowest function span containing `occ_line`.
///
/// `spans` must be pre-sorted `(end_line - line) ASC, id ASC` (i.e. the order
/// returned by [`fetch_all_fn_spans`]).  The first match is therefore the
/// narrowest enclosing span — identical to `ORDER BY … LIMIT 1`.
fn find_narrowest_enclosing(spans: &[FnSpan], occ_line: i64) -> Option<i64> {
    spans
        .iter()
        .find(|s| s.line <= occ_line && s.end_line >= occ_line)
        .map(|s| s.id)
}

/// src; the language is taken from any sibling node at the same path (SCIP
/// definition nodes for the path are inserted earlier in this transaction).
fn file_node_for_attribution(
    tx: &rusqlite::Transaction<'_>,
    corpus: &str,
    path: &str,
) -> anyhow::Result<i64> {
    if let Some(id) = tx
        .query_row(
            "SELECT id FROM nodes WHERE corpus = ?1 AND path = ?2 AND kind = 'file' LIMIT 1",
            params![corpus, path],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .context("file_node_for_attribution: lookup")?
    {
        return Ok(id);
    }
    let language: String = tx
        .query_row(
            "SELECT language FROM nodes WHERE corpus = ?1 AND path = ?2 LIMIT 1",
            params![corpus, path],
            |row| row.get(0),
        )
        .optional()
        .context("file_node_for_attribution: sibling language")?
        .unwrap_or_default();
    let vname = travsr_core::VName::new(corpus, "", path, &language, "file");
    let id = node_id_to_i64(vname.id());
    tx.execute(
        "INSERT OR IGNORE INTO nodes(id, corpus, root, path, language, signature, kind, package) \
         VALUES(?1, ?2, '', ?3, ?4, 'file', 'file', '')",
        params![id, corpus, path, language],
    )
    .context("file_node_for_attribution: insert file node")?;
    Ok(id)
}

fn node_id_to_i64(id: NodeId) -> i64 {
    id.0 as i64
}

fn i64_to_node_id(v: i64) -> NodeId {
    NodeId(v as u64)
}

/// Jaccard similarity on byte-level trigrams of two strings.
/// Used by `expand_query` (L2-A) to find vocabulary-grounded candidates.
/// Returns 0.0 for strings shorter than 3 bytes.
fn byte_trigram_jaccard(a: &str, b: &str) -> f64 {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() < 3 || bb.len() < 3 {
        return 0.0;
    }
    let ta: std::collections::HashSet<[u8; 3]> =
        ab.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    let tb: std::collections::HashSet<[u8; 3]> =
        bb.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    let intersection = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    /// #749 review: the `.travsr` name check had no direct test in the crate
    /// that owns it. Its only coverage was incidental, in a `travsr-mcp`
    /// snippets test whose fixture happens to keep a non-`.travsr` layout, so
    /// normalising that fixture for consistency would have silently deleted it.
    #[test]
    fn a_database_outside_a_travsr_dir_yields_no_root() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&dir.path().join("graph.db")).unwrap();
        assert_eq!(
            store.repo_root_from_db_path(),
            None,
            "deriving a root from any database location invents a confident \
             wrong answer, which is worse than none"
        );
    }

    #[test]
    fn a_database_inside_a_travsr_dir_yields_its_parent() {
        let dir = tempfile::tempdir().unwrap();
        let travsr_dir = dir.path().join(".travsr");
        std::fs::create_dir_all(&travsr_dir).unwrap();
        let store = SqliteStore::open(&travsr_dir.join("graph.db")).unwrap();
        assert_eq!(
            store.repo_root_from_db_path().as_deref(),
            Some(dir.path()),
            "`<repo>/.travsr/graph.db` derives `<repo>`"
        );
    }

    #[test]
    fn an_in_memory_store_has_no_root_to_derive() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert_eq!(store.repo_root_from_db_path(), None);
    }

    use super::*;
    use travsr_core::VName;

    fn sample_node(sig: &str) -> Node {
        Node::new(
            VName::new("test-corpus", "main", "src/foo.ts", "typescript", sig),
            "function",
        )
    }

    fn doc_chunk(path: &str, anchor: &str) -> Node {
        Node::new(
            VName::new("test-corpus", "", path, "markdown", anchor),
            "doc-chunk",
        )
    }

    // ── #636: SqliteStore::integrity_report ──────────────────────────────────

    #[test]
    fn integrity_report_clean_graph_is_all_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        store.put_node(&a).unwrap();
        store.put_file_hash("src/foo.ts", "deadbeef").unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/foo.ts"), b"export function a() {}").unwrap();

        let report = store.integrity_report(tmp.path()).unwrap();
        assert_eq!(report.node_count, 1);
        assert_eq!(report.edge_count, 0);
        assert!(
            report.ghost_paths.is_empty(),
            "no file on disk yet: {:?}",
            report.ghost_paths
        );
        assert_eq!(report.orphan_edges_detected, 0);
        assert_eq!(report.self_ref_call_edges_detected, 0);
        assert!(report.lexical_index_parity_issue.is_none());
    }

    #[test]
    fn integrity_report_detects_ghost_when_tracked_file_missing_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.put_file_hash("src/deleted.ts", "deadbeef").unwrap();
        // src/deleted.ts is tracked in the DB but never created on disk.

        let report = store.integrity_report(tmp.path()).unwrap();
        assert_eq!(report.ghost_paths, vec!["src/deleted.ts".to_string()]);
    }

    #[test]
    fn integrity_report_does_not_flag_file_present_on_disk_as_ghost() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/present.ts"), b"ok").unwrap();
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.put_file_hash("src/present.ts", "deadbeef").unwrap();

        let report = store.integrity_report(tmp.path()).unwrap();
        assert!(report.ghost_paths.is_empty());
    }

    #[test]
    fn integrity_report_counts_orphan_edges_read_only() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let a_id = store.put_node(&a).unwrap();
        // dst NodeId 999999 has no corresponding node row (an orphan edge).
        store
            .put_edge(&Edge::new(a_id, NodeId(999_999), EdgeKind::RefCall))
            .unwrap();

        let before_nodes = store.node_count().unwrap();
        let report = store.integrity_report(tmp.path()).unwrap();
        assert_eq!(report.orphan_edges_detected, 1);
        // Read-only: never mutates.
        assert_eq!(store.node_count().unwrap(), before_nodes);
        assert_eq!(store.edge_count().unwrap(), 1);
    }

    /// #376 W1: this filter must stay identical to the sidecar's NODE_ELIGIBLE.
    /// A doc-chunk with no prose is not embeddable (its vector would be built
    /// from the heading trail alone), so counting it as pending would make the
    /// daemon spawn a sidecar on every tick that can never close the gap.
    #[test]
    fn embed_progress_excludes_doc_chunks_without_prose() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let with_prose = doc_chunk("docs/a.md", "doc:intro");
        let without_prose = doc_chunk("docs/b.md", "doc:intro");
        let code = sample_node("fn:handler");
        store.put_node(&with_prose).unwrap();
        store.put_node(&without_prose).unwrap();
        store.put_node(&code).unwrap();
        store
            .write_embed_texts_batch(&[(with_prose.id, "doc: a > intro | body".to_string())])
            .unwrap();

        let (total, _embedded, _p1_total, _p1_done) = store.embed_progress("m", 0).unwrap();
        assert_eq!(
            total, 2,
            "embeddable = {{doc-chunk with prose, function}}, not the prose-less chunk"
        );
    }

    // ── #862: `embedded` is over embeddable nodes, like the other three ────────
    //
    // File-backed stores throughout, because `open_in_memory` has no sibling
    // `embed.db` and `embed_progress` reads `embedded` from there. The embed.db
    // is written with the sidecar's own DDL (`travsr-embed::freshness::
    // ensure_schema`), rows and all, so what these tests count is what the
    // daemon and `travsr embed status` count in the field.

    const M: &str = "arctic-embed-m-v1.5";

    fn node_kind(kind: &str, sig: &str) -> Node {
        Node::new(
            VName::new("test-corpus", "main", "src/foo.ts", "typescript", sig),
            kind,
        )
    }

    /// Create `<dir>/embed.db` as the sidecar does and insert one vector per
    /// `(node_id, model_id)`. `node_id` is raw so a test can point a row at a
    /// node that does not exist.
    fn seed_embed_db(dir: &std::path::Path, rows: &[(i64, &str)]) -> std::path::PathBuf {
        let path = dir.join("embed.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS node_embeddings (
                 node_id   INTEGER NOT NULL,
                 model_id  TEXT    NOT NULL,
                 embedding BLOB    NOT NULL,
                 text_hash TEXT,
                 PRIMARY KEY (node_id, model_id)
             ) WITHOUT ROWID;
             CREATE INDEX IF NOT EXISTS idx_node_embeddings_model
                 ON node_embeddings(model_id);",
        )
        .unwrap();
        for (node_id, model_id) in rows {
            conn.execute(
                "INSERT OR REPLACE INTO node_embeddings (node_id, model_id, embedding, text_hash) \
                 VALUES (?1, ?2, X'00', 'h')",
                params![node_id, model_id],
            )
            .unwrap();
        }
        path
    }

    fn raw_vector_count(embed_db: &std::path::Path, model_id: &str) -> i64 {
        Connection::open(embed_db)
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
                params![model_id],
                |r| r.get(0),
            )
            .unwrap()
    }

    fn ids(nodes: &[Node]) -> Vec<i64> {
        nodes.iter().map(|n| node_id_to_i64(n.id)).collect()
    }

    /// `n` eligible function nodes, written to the store.
    fn put_functions(store: &mut SqliteStore, n: usize) -> Vec<Node> {
        (0..n)
            .map(|i| {
                let node = node_kind("function", &format!("fn:f{i}"));
                store.put_node(&node).unwrap();
                node
            })
            .collect()
    }

    /// The daemon's Phase 2 arithmetic (`crates/travsr-daemon`, `phase2_remaining`),
    /// replicated so the store tests can state what the tick would conclude.
    fn daemon_phase2_remaining(p: (u64, u64, u64, u64)) -> u64 {
        let (total, embedded, p1_total, p1_done) = p;
        total
            .saturating_sub(p1_total)
            .saturating_sub(embedded.saturating_sub(p1_done))
    }

    /// Positive 1: every eligible node has a vector, and nothing else does.
    #[test]
    fn embed_progress_is_complete_when_every_eligible_node_has_a_vector() {
        let (mut store, dir) = file_backed_store();
        let nodes = put_functions(&mut store, 12);
        let rows: Vec<(i64, &str)> = ids(&nodes).into_iter().map(|id| (id, M)).collect();
        seed_embed_db(dir.path(), &rows);

        let p = store.embed_progress(M, 1000).unwrap();
        assert_eq!(p, (12, 12, 0, 0));
        assert_eq!(daemon_phase2_remaining(p), 0, "nothing left to embed");
    }

    /// Positive 2: 100 eligible, 75 embedded. The 25 missing vectors are the
    /// pending work, and Phase 2 must see exactly them.
    #[test]
    fn embed_progress_reports_missing_eligible_vectors_as_pending() {
        let (mut store, dir) = file_backed_store();
        let nodes = put_functions(&mut store, 100);
        let rows: Vec<(i64, &str)> = ids(&nodes[..75]).into_iter().map(|id| (id, M)).collect();
        seed_embed_db(dir.path(), &rows);

        let p = store.embed_progress(M, 1000).unwrap();
        assert_eq!(p, (100, 75, 0, 0));
        assert_eq!(daemon_phase2_remaining(p), 25);
    }

    /// Positive 3, the core #862 regression: one ineligible node with a vector
    /// must not lift `embedded` above the eligible count.
    #[test]
    fn embed_progress_does_not_count_a_vector_on_an_ineligible_node() {
        let (mut store, dir) = file_backed_store();
        let eligible = put_functions(&mut store, 10);
        let field = node_kind("field", "field:Foo.bar");
        store.put_node(&field).unwrap();
        let mut rows: Vec<(i64, &str)> = ids(&eligible).into_iter().map(|id| (id, M)).collect();
        rows.push((node_id_to_i64(field.id), M));
        let embed_db = seed_embed_db(dir.path(), &rows);
        assert_eq!(
            raw_vector_count(&embed_db, M),
            11,
            "precondition: the unfiltered count is the inflated 11"
        );

        let (total, embedded, _, _) = store.embed_progress(M, 1000).unwrap();
        assert_eq!(total, 10);
        assert_eq!(embedded, 10, "not 11: the field node is not embeddable");
        assert!(embedded <= total);
        assert_eq!(
            raw_vector_count(&embed_db, M),
            11,
            "the ineligible node's vector is left in place, only not counted"
        );
    }

    /// Positive 4: many ineligible vectors, none of them count.
    #[test]
    fn embed_progress_ignores_every_ineligible_vector() {
        let (mut store, dir) = file_backed_store();
        let eligible = put_functions(&mut store, 5);
        let mut rows: Vec<(i64, &str)> = ids(&eligible).into_iter().map(|id| (id, M)).collect();
        for i in 0..40 {
            let n = node_kind("field", &format!("field:T.f{i}"));
            store.put_node(&n).unwrap();
            rows.push((node_id_to_i64(n.id), M));
        }
        let embed_db = seed_embed_db(dir.path(), &rows);
        assert_eq!(raw_vector_count(&embed_db, M), 45, "precondition");

        assert_eq!(store.embed_progress(M, 1000).unwrap(), (5, 5, 0, 0));
    }

    /// Positive 5: the realistic mixture. Only eligible nodes that have a vector
    /// count towards `embedded`; only eligible nodes count towards `total`.
    #[test]
    fn embed_progress_counts_only_eligible_nodes_with_vectors_in_a_mixed_graph() {
        let (mut store, dir) = file_backed_store();
        let eligible_embedded = put_functions(&mut store, 30);
        let eligible_pending: Vec<Node> = (0..20)
            .map(|i| {
                let n = node_kind("method", &format!("method:C.m{i}"));
                store.put_node(&n).unwrap();
                n
            })
            .collect();
        let mut rows: Vec<(i64, &str)> = ids(&eligible_embedded)
            .into_iter()
            .map(|id| (id, M))
            .collect();
        for i in 0..7 {
            let n = node_kind("variable", &format!("var:v{i}"));
            store.put_node(&n).unwrap();
            rows.push((node_id_to_i64(n.id), M)); // ineligible + embedded
        }
        for i in 0..9 {
            let n = node_kind("import", &format!("import:./i{i}"));
            store.put_node(&n).unwrap(); // ineligible + not embedded
        }
        seed_embed_db(dir.path(), &rows);

        let p = store.embed_progress(M, 1000).unwrap();
        assert_eq!(
            p,
            (50, 30, 0, 0),
            "total = 30 embedded + 20 pending functions and methods; embedded = 30"
        );
        assert_eq!(daemon_phase2_remaining(p), eligible_pending.len() as u64);
    }

    /// Positive 6: vectors for another model are not this model's progress.
    #[test]
    fn embed_progress_is_scoped_to_the_requested_model() {
        let (mut store, dir) = file_backed_store();
        let nodes = put_functions(&mut store, 10);
        let all = ids(&nodes);
        let mut rows: Vec<(i64, &str)> = all[..4].iter().map(|&id| (id, M)).collect();
        rows.extend(all.iter().map(|&id| (id, "bge-large-en-v1.5")));
        seed_embed_db(dir.path(), &rows);

        assert_eq!(store.embed_progress(M, 1000).unwrap(), (10, 4, 0, 0));
        assert_eq!(
            store.embed_progress("bge-large-en-v1.5", 1000).unwrap(),
            (10, 10, 0, 0)
        );
        assert_eq!(
            store.embed_progress("never-installed", 1000).unwrap(),
            (10, 0, 0, 0)
        );
    }

    /// Positive 7: eligible nodes, no vectors at all: everything is pending.
    #[test]
    fn embed_progress_reports_zero_embedded_when_nothing_is_embedded_yet() {
        let (mut store, dir) = file_backed_store();
        put_functions(&mut store, 8);
        // embed.db exists (the sidecar created the schema) but holds no rows.
        seed_embed_db(dir.path(), &[]);

        let p = store.embed_progress(M, 1000).unwrap();
        assert_eq!(p, (8, 0, 0, 0));
        assert_eq!(daemon_phase2_remaining(p), 8);
    }

    /// Positive 8: an empty index, with and without an embed.db, and with an
    /// embed.db that only holds rows for nodes that do not exist.
    #[test]
    fn embed_progress_on_an_empty_index_is_all_zero() {
        let (store, dir) = file_backed_store();
        assert_eq!(
            store.embed_progress(M, 3).unwrap(),
            (0, 0, 0, 0),
            "no embed.db"
        );

        seed_embed_db(dir.path(), &[]);
        assert_eq!(
            store.embed_progress(M, 3).unwrap(),
            (0, 0, 0, 0),
            "empty embed.db"
        );

        seed_embed_db(dir.path(), &[(1, M), (2, M), (3, M)]);
        assert_eq!(
            store.embed_progress(M, 3).unwrap(),
            (0, 0, 0, 0),
            "vectors with no nodes are not progress on an empty graph"
        );
    }

    /// Negative 9: an orphan row (its node was deleted; embed.db is a separate
    /// file, so no cascade reaches it) does not count and is not deleted.
    #[test]
    fn embed_progress_excludes_an_orphan_vector_without_deleting_it() {
        let (mut store, dir) = file_backed_store();
        let nodes = put_functions(&mut store, 3);
        let mut rows: Vec<(i64, &str)> = ids(&nodes).into_iter().map(|id| (id, M)).collect();
        rows.push((987_654_321, M)); // no such node
        let embed_db = seed_embed_db(dir.path(), &rows);

        assert_eq!(store.embed_progress(M, 1000).unwrap(), (3, 3, 0, 0));
        assert_eq!(
            raw_vector_count(&embed_db, M),
            4,
            "the orphan is still there: progress reporting is read-only"
        );

        // The realistic route to an orphan: the node's file is deleted.
        store.delete_nodes_for_path("src/foo.ts").unwrap();
        assert_eq!(store.embed_progress(M, 1000).unwrap(), (0, 0, 0, 0));
        assert_eq!(raw_vector_count(&embed_db, M), 4);
    }

    /// Negative 10: the exact #862 row. `kind = 'field'`, `embed_text IS NULL`,
    /// vector present: excluded. The same kind *with* text is opted in by the
    /// daemon and counts, which is what shows the predicate decides, not the kind.
    #[test]
    fn embed_progress_excludes_a_field_whose_embed_text_is_null_but_counts_one_with_text() {
        let (mut store, dir) = file_backed_store();
        let without_text = node_kind("field", "field:Config.timeout");
        let with_text = node_kind("field", "field:Config.retries");
        store.put_node(&without_text).unwrap();
        store.put_node(&with_text).unwrap();
        store
            .write_embed_texts_batch(&[(with_text.id, "field retries: u32".to_string())])
            .unwrap();
        seed_embed_db(
            dir.path(),
            &[
                (node_id_to_i64(without_text.id), M),
                (node_id_to_i64(with_text.id), M),
            ],
        );

        assert_eq!(
            store.embed_progress(M, 1000).unwrap(),
            (1, 1, 0, 0),
            "one embeddable field (it has text), one vector that counts"
        );

        // The model-switch path that produced the report: every text is cleared
        // and the regeneration never restores this one. The vector stays, the
        // node is no longer embeddable, and the count must follow the node.
        store.clear_all_embed_texts().unwrap();
        assert_eq!(store.embed_progress(M, 1000).unwrap(), (0, 0, 0, 0));
    }

    /// Negative 11: every kind the predicate excludes, each with a vector, plus a
    /// prose-less doc-chunk. One eligible node beside them so the counts are 1/1.
    #[test]
    fn embed_progress_excludes_each_ineligible_kind_even_with_a_vector() {
        let (mut store, dir) = file_backed_store();
        let ok = node_kind("function", "fn:ok");
        store.put_node(&ok).unwrap();
        let mut rows = vec![(node_id_to_i64(ok.id), M)];
        for kind in [
            "file",
            "file-module",
            "import",
            "module",
            "field",
            "variable",
        ] {
            let n = node_kind(kind, &format!("{kind}:x"));
            store.put_node(&n).unwrap();
            rows.push((node_id_to_i64(n.id), M));
        }
        let chunk = doc_chunk("docs/a.md", "doc:intro"); // no prose
        store.put_node(&chunk).unwrap();
        rows.push((node_id_to_i64(chunk.id), M));
        let embed_db = seed_embed_db(dir.path(), &rows);
        assert_eq!(raw_vector_count(&embed_db, M), 8, "precondition");

        assert_eq!(store.embed_progress(M, 1000).unwrap(), (1, 1, 0, 0));
    }

    /// Negative 12: one node, vectors for several models. Only the requested
    /// model's row counts, and it counts once.
    #[test]
    fn embed_progress_counts_a_node_once_per_requested_model() {
        let (mut store, dir) = file_backed_store();
        let n = node_kind("function", "fn:one");
        store.put_node(&n).unwrap();
        let id = node_id_to_i64(n.id);
        seed_embed_db(
            dir.path(),
            &[(id, M), (id, "bge-small-en-v1.5"), (id, "third")],
        );

        assert_eq!(store.embed_progress(M, 1000).unwrap(), (1, 1, 0, 0));
        assert_eq!(store.embed_progress("third", 1000).unwrap(), (1, 1, 0, 0));
        assert_eq!(store.embed_progress("fourth", 1000).unwrap(), (1, 0, 0, 0));
    }

    /// Negative 13: repeated calls are deterministic and change nothing, in
    /// either database. The ATTACH/DETACH cycle must also survive being repeated.
    #[test]
    fn embed_progress_is_deterministic_and_read_only_across_repeated_calls() {
        let (mut store, dir) = file_backed_store();
        let eligible = put_functions(&mut store, 6);
        let field = node_kind("field", "field:T.f");
        store.put_node(&field).unwrap();
        let mut rows: Vec<(i64, &str)> =
            ids(&eligible[..4]).into_iter().map(|id| (id, M)).collect();
        rows.push((node_id_to_i64(field.id), M));
        rows.push((42, M)); // orphan
        let embed_db = seed_embed_db(dir.path(), &rows);
        let nodes_before = store.node_count().unwrap();
        let vectors_before = raw_vector_count(&embed_db, M);

        let first = store.embed_progress(M, 1000).unwrap();
        assert_eq!(first, (6, 4, 0, 0));
        for _ in 0..5 {
            assert_eq!(store.embed_progress(M, 1000).unwrap(), first);
        }
        assert_eq!(store.node_count().unwrap(), nodes_before);
        assert_eq!(raw_vector_count(&embed_db, M), vectors_before);
    }

    /// Negative 14: Phase 1 partially done, with ineligible vectors present.
    /// `phase1_done` was already filtered; the fix must leave it alone and make
    /// `embedded` agree with it, so the derived Phase 2 figure is the truth.
    #[test]
    fn embed_progress_with_partial_phase1_is_not_skewed_by_ineligible_vectors() {
        let (mut store, dir) = file_backed_store();
        let core = put_functions(&mut store, 20);
        let rest: Vec<Node> = (0..80)
            .map(|i| {
                let n = node_kind("method", &format!("method:R.m{i}"));
                store.put_node(&n).unwrap();
                n
            })
            .collect();
        let mut shells: Vec<(NodeId, u32)> = core.iter().map(|n| (n.id, 5)).collect();
        shells.extend(rest.iter().map(|n| (n.id, 0)));
        store.write_shell_numbers(&shells).unwrap();
        // Half the core is embedded, none of the rest, plus five ineligible vectors.
        let mut rows: Vec<(i64, &str)> = ids(&core[..10]).into_iter().map(|id| (id, M)).collect();
        for i in 0..5 {
            let n = node_kind("variable", &format!("var:v{i}"));
            store.put_node(&n).unwrap();
            rows.push((node_id_to_i64(n.id), M));
        }
        seed_embed_db(dir.path(), &rows);

        let p = store.embed_progress(M, 5).unwrap();
        assert_eq!(p, (100, 10, 20, 10));
        let (total, embedded, p1_total, p1_done) = p;
        assert!(p1_done < p1_total, "Phase 1 is genuinely incomplete");
        assert!(p1_done <= p1_total);
        assert!(embedded <= total);
        assert_eq!(
            daemon_phase2_remaining(p),
            80,
            "all 80 Phase 2 nodes are pending; with the unfiltered 15 this read 75"
        );
    }

    /// The invariant the report states, over a graph that has every shape at
    /// once: eligible embedded, eligible pending, ineligible embedded, ineligible
    /// pending, an orphan vector, and another model's vectors.
    #[test]
    fn embed_progress_embedded_never_exceeds_total_symbols() {
        let (mut store, dir) = file_backed_store();
        let eligible = put_functions(&mut store, 9);
        let mut rows: Vec<(i64, &str)> =
            ids(&eligible[..6]).into_iter().map(|id| (id, M)).collect();
        for kind in [
            "field",
            "variable",
            "module",
            "import",
            "file",
            "file-module",
        ] {
            for i in 0..3 {
                let n = node_kind(kind, &format!("{kind}:{i}"));
                store.put_node(&n).unwrap();
                if i < 2 {
                    rows.push((node_id_to_i64(n.id), M));
                }
            }
        }
        rows.push((7_777, M));
        rows.extend(ids(&eligible).into_iter().map(|id| (id, "other-model")));
        let embed_db = seed_embed_db(dir.path(), &rows);
        assert!(
            raw_vector_count(&embed_db, M) > 9,
            "precondition: unfiltered, this model has more vectors than eligible nodes"
        );

        let (total, embedded, p1_total, p1_done) = store.embed_progress(M, 1000).unwrap();
        assert_eq!((total, embedded), (9, 6));
        assert!(embedded <= total);
        assert!(p1_done <= p1_total);
    }

    /// #376 W2: the count the daemon's embed tick uses to notice that content
    /// changed. Coverage alone cannot see this — the changed nodes still have
    /// their (now wrong) embedding rows.
    #[test]
    fn pending_tombstones_counts_unconsumed_deletions() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        assert_eq!(store.pending_tombstones().unwrap(), 0);
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        assert_eq!(
            store.pending_tombstones().unwrap(),
            0,
            "writes do not count"
        );

        store.delete_nodes_for_path("src/foo.ts").unwrap();
        assert_eq!(
            store.pending_tombstones().unwrap(),
            2,
            "one tombstone per deleted node, unconsumed until a sidecar pass runs"
        );
    }

    /// #376 W1: the daemon fills these in before spawning; code nodes are
    /// deliberately out of scope (their regeneration parses the whole repo).
    #[test]
    fn doc_nodes_missing_embed_text_returns_only_prose_less_doc_chunks() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let filled = doc_chunk("docs/a.md", "doc:intro");
        let empty = doc_chunk("docs/b.md", "doc:setup");
        let code = sample_node("fn:handler");
        store.put_node(&filled).unwrap();
        store.put_node(&empty).unwrap();
        store.put_node(&code).unwrap();
        store
            .write_embed_texts_batch(&[(filled.id, "doc: a > intro | body".to_string())])
            .unwrap();

        let missing = store.doc_nodes_missing_embed_text().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].id, empty.id);
        assert_eq!(missing[0].vname.path, "docs/b.md");
    }

    // ── RFC-019 direct-cosine oracle helpers ──────────────────────────────

    fn blob_of(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn decode_embedding_round_trips_f32_le() {
        let v = vec![0.1_f32, -0.25, 0.5, 1.0];
        let decoded = decode_embedding(&blob_of(&v)).expect("valid blob");
        assert_eq!(decoded, v);
    }

    #[test]
    fn decode_embedding_rejects_ragged_and_empty() {
        assert!(decode_embedding(&[]).is_none(), "empty blob → None");
        assert!(decode_embedding(&[1, 2, 3]).is_none(), "len%4≠0 → None");
    }

    #[test]
    fn cosine_is_defensive_on_degenerate_input() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0, "zero norm → 0");
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0, "length mismatch → 0");
        // Identical unit vectors → 1.0.
        let a = [0.6_f32, 0.8];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    /// Build a minimal embed.db with a `node_embeddings` table and score against it.
    #[test]
    fn score_candidates_reads_and_omits_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let embed_db = tmp.path().join("embed.db");
        let conn = Connection::open(&embed_db).unwrap();
        conn.execute_batch(
            "CREATE TABLE node_embeddings (node_id INTEGER NOT NULL, model_id TEXT NOT NULL, \
             embedding BLOB NOT NULL, PRIMARY KEY (node_id, model_id));",
        )
        .unwrap();
        // Node 1: identical to query → cosine 1. Node 2: orthogonal → cosine 0.
        // Node 3: wrong model_id → not scored. Node 4: no row → omitted.
        let q = [1.0_f32, 0.0];
        conn.execute(
            "INSERT INTO node_embeddings VALUES (?1, 'm', ?2)",
            params![1_i64, blob_of(&[1.0, 0.0])],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO node_embeddings VALUES (?1, 'm', ?2)",
            params![2_i64, blob_of(&[0.0, 1.0])],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO node_embeddings VALUES (?1, 'other', ?2)",
            params![3_i64, blob_of(&[1.0, 0.0])],
        )
        .unwrap();
        drop(conn);

        let ids = [NodeId(1), NodeId(2), NodeId(3), NodeId(4)];
        let scored = score_candidates(&q, &embed_db, "m", &ids).unwrap();
        let map: std::collections::HashMap<NodeId, f32> = scored.into_iter().collect();
        assert!((map[&NodeId(1)] - 1.0).abs() < 1e-6, "identical → cosine 1");
        assert!(map[&NodeId(2)].abs() < 1e-6, "orthogonal → cosine 0");
        assert!(!map.contains_key(&NodeId(3)), "wrong model_id omitted");
        assert!(
            !map.contains_key(&NodeId(4)),
            "missing row omitted (unknown)"
        );
    }

    #[test]
    fn score_candidates_missing_db_is_empty_not_error() {
        let tmp = tempfile::tempdir().unwrap();
        let embed_db = tmp.path().join("nope.db");
        let out = score_candidates(&[1.0, 0.0], &embed_db, "m", &[NodeId(1)]).unwrap();
        assert!(out.is_empty(), "absent embed.db → empty, degrades to FTS");
    }

    /// #464 follow-up: embed.db writes must be observable through the store's
    /// persistent embed metadata connection so the daemon's query cache can
    /// key `ask` results on embed state, not just graph.db.
    #[test]
    fn embed_data_version_tracks_out_of_band_embed_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&tmp.path().join("graph.db")).unwrap();

        // No embed.db yet (FTS-only) → None, not an error.
        assert_eq!(store.embed_data_version().unwrap(), None);

        // The sidecar creates embed.db out-of-band.
        let embed_db = tmp.path().join("embed.db");
        let writer = Connection::open(&embed_db).unwrap();
        writer
            .execute_batch("CREATE TABLE node_embeddings (node_id INTEGER PRIMARY KEY);")
            .unwrap();
        let v1 = store
            .embed_data_version()
            .unwrap()
            .expect("embed.db now exists");

        // A write from another connection (the sidecar) must bump the version
        // seen by the store's persistent read connection.
        writer
            .execute("INSERT INTO node_embeddings VALUES (1)", [])
            .unwrap();
        let v2 = store
            .embed_data_version()
            .unwrap()
            .expect("embed.db still exists");
        assert_ne!(v1, v2, "sidecar write must bump embed data_version");

        // No intervening write → the version is stable, so cached entries
        // keyed on it keep hitting.
        assert_eq!(
            store.embed_data_version().unwrap(),
            Some(v2),
            "stable when unchanged"
        );
    }

    /// #509: embed.db deleted and recreated at the same path between two polls
    /// is a *different file*, and the cached read-only connection is left
    /// reading the unlinked inode with a permanently frozen `data_version`.
    /// `path.exists()` is true on both sides of the swap, so existence cannot
    /// see it. The reported version must move, or every `ask` answer cached
    /// before the swap keeps hitting until the daemon restarts.
    /// #509: `file_identity` must change when a path is replaced by a different
    /// file, on every platform.
    ///
    /// Runs on Windows too, unlike the store-level test below, because it never
    /// holds the file open: it swaps the file first and reads identity after, so
    /// nothing is blocking the unlink. That makes this the coverage for the
    /// Windows `(creation_time, last_write_time, file_size)` arm, which is
    /// otherwise reachable by no test on this platform.
    ///
    /// Deliberately asserts inequality of two identities rather than any
    /// particular member. On unix the inode carries it; on Windows creation time
    /// can be tunneled onto a same-named replacement within ~15s (NTFS
    /// tunneling), and the other two members carry it instead. Asserting the
    /// tuple differs is the property both platforms actually promise.
    #[test]
    fn file_identity_changes_when_the_file_is_replaced() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("swapme.db");

        std::fs::write(&p, b"first").unwrap();

        // The open handle is load-bearing, not incidental setup. It mirrors the
        // cached read-only `Connection` production holds across exactly this
        // swap, and it is what makes the assertion below true on every platform:
        // POSIX cannot free an unlinked inode while a descriptor still refers to
        // it, so its number cannot be reallocated to the replacement.
        //
        // Without it this failed on Linux CI and passed on macOS: ext4 handed the
        // recreated file the inode just released, so `(dev, ino)` was unchanged
        // and the replacement looked like the original. That is a real property
        // of the filesystem. The reason it cannot bite production is the pin, so
        // the test has to hold one too, or it is asserting against a situation
        // the code never meets.
        let pin = std::fs::File::open(&p).expect("pin the inode as the cached connection does");
        let first = super::file_identity(&p).expect("identity of an existing file");

        std::fs::remove_file(&p).unwrap();
        std::fs::write(&p, b"second file, different length").unwrap();
        let second = super::file_identity(&p).expect("identity of the replacement");
        drop(pin);

        assert_ne!(
            first, second,
            "a replaced file must not keep the identity of the one it replaced,              or a cached connection to the old file reads as current"
        );

        // Same file, read twice: identity is stable, so it does not spuriously
        // invalidate the cache on every poll.
        assert_eq!(
            second,
            super::file_identity(&p).unwrap(),
            "identity must be stable for an unchanged file"
        );

        // A missing path has no identity, which is what makes the caller fall
        // back to "no embed.db" rather than to a stale reading.
        std::fs::remove_file(&p).unwrap();
        assert!(super::file_identity(&p).is_none());
    }

    ///
    /// Unix only, and not for convenience: the failure mode cannot occur on
    /// Windows. SQLite opens db files there without `FILE_SHARE_DELETE`, so
    /// unlinking embed.db under the live read-only connection is refused
    /// outright (`os error 32`, the file is in use) rather than succeeding and
    /// leaving the connection on an unlinked inode. A `#[cfg(windows)]` variant
    /// would have to fake the swap in a way the platform cannot actually
    /// produce, which would assert against a scenario no user can reach. The
    /// Windows arm of `file_identity` is defensive, for the rename-over and
    /// already-reopened paths, and is covered by
    /// `file_identity_changes_when_the_file_is_replaced` above, which runs on
    /// every platform because it never holds the file open.
    #[cfg(unix)]
    #[test]
    fn embed_data_version_detects_delete_and_recreate() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&tmp.path().join("graph.db")).unwrap();
        let embed_db = tmp.path().join("embed.db");

        // The sidecar creates embed.db; reading it pins the store's persistent
        // read-only connection to that file.
        write_embed_db(&embed_db, 1);
        let before = store
            .embed_data_version()
            .unwrap()
            .expect("embed.db exists");

        // The user deletes .travsr/embed.db to force a rebuild and the sidecar
        // recreates it, both between two polls. WAL sidecars are deleted too,
        // which is what a real `rm .travsr/embed.db*` does.
        for sidecar in ["", "-wal", "-shm"] {
            let p = tmp.path().join(format!("embed.db{sidecar}"));
            if p.exists() {
                std::fs::remove_file(&p).unwrap();
            }
        }
        write_embed_db(&embed_db, 2);
        assert!(
            embed_db.exists(),
            "the path is present on both sides of the swap, which is why \
             exists() cannot detect it"
        );

        let after = store
            .embed_data_version()
            .unwrap()
            .expect("recreated embed.db");
        assert_ne!(
            before, after,
            "delete-and-recreate must move the embed version, otherwise warm \
             ask entries keep hitting against a rebuilt embed.db"
        );

        // And the reopened connection is live, not another corpse: a further
        // out-of-band write to the new file is still observed.
        let writer = Connection::open(&embed_db).unwrap();
        writer
            .execute("INSERT INTO node_embeddings VALUES (99)", [])
            .unwrap();
        assert_ne!(
            after,
            store.embed_data_version().unwrap().unwrap(),
            "the reopened connection must track writes to the new file"
        );
    }

    /// Create (or recreate) an embed.db at `path` holding a single row, the way
    /// the embed sidecar would. Used by the #509 regression test.
    fn write_embed_db(path: &Path, row: i64) {
        let writer = Connection::open(path).unwrap();
        writer
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 CREATE TABLE node_embeddings (node_id INTEGER PRIMARY KEY);",
            )
            .unwrap();
        writer
            .execute("INSERT INTO node_embeddings VALUES (?1)", [row])
            .unwrap();
    }

    #[test]
    fn embed_readiness_wait_returns_on_mark() {
        let r = EmbedReadiness::new();
        assert!(!r.is_ready());
        let r_bg = Arc::clone(&r);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(80));
            r_bg.mark_ready();
        });
        // Should wake promptly once marked, well within the 2 s cap.
        let start = std::time::Instant::now();
        assert!(r.wait(std::time::Duration::from_secs(2)));
        assert!(r.is_ready());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "wait must wake promptly on mark, not spin to the cap"
        );
    }

    #[test]
    fn embed_readiness_wait_times_out_when_never_armed() {
        let r = EmbedReadiness::new();
        assert!(!r.wait(std::time::Duration::from_millis(50)));
        assert!(!r.is_ready());
    }

    #[test]
    fn embed_ready_opt_out_when_no_readiness_registered() {
        // Callers that don't register readiness (legacy daemon path, `travsr ask`,
        // tests) opt out of warm-up tracking → always ready, so the "warming"
        // state never applies and pre-change behaviour is preserved.
        let mut store = SqliteStore::open_in_memory().unwrap();
        assert!(
            store.embed_ready(),
            "no readiness registered → ready (opt-out)"
        );
        store.set_embed_knn_hook(Arc::new(|_q, _k| Ok(vec![])));
        assert!(
            store.embed_ready(),
            "hook present, no readiness → still ready"
        );
        assert!(store.wait_embed_ready(std::time::Duration::from_millis(0)));
    }

    #[test]
    fn embed_ready_reflects_registered_readiness() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let r = EmbedReadiness::new();
        store.set_embed_knn_hook(Arc::new(|_q, _k| Ok(vec![])));
        store.set_embed_readiness(Arc::clone(&r));
        assert!(
            !store.embed_ready(),
            "readiness registered but un-armed → not ready even with hook present"
        );
        r.mark_ready();
        assert!(store.embed_ready(), "armed readiness → ready");
    }

    #[test]
    fn migration_is_idempotent_in_memory() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Re-running the migration runner on an already-migrated store must be a no-op.
        let runner = sqlite_migration_runner();
        runner.run(&mut store).unwrap();
        runner.run(&mut store).unwrap();
        let n = sample_node("fn:roundtrip");
        store.put_node(&n).unwrap();
    }

    #[test]
    fn v14_migration_creates_covering_reverse_index() {
        let store = SqliteStore::open_in_memory().unwrap();
        let cov_exists: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='index' AND name='idx_edges_dst_kind_cov'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(cov_exists, 1, "idx_edges_dst_kind_cov must exist after v14");

        let old_exists: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='index' AND name='idx_edges_dst_kind'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(old_exists, 0, "idx_edges_dst_kind must be dropped by v14");
    }

    /// The plan text for `sql`, as SQLite's EXPLAIN QUERY PLAN reports it
    /// (column 3 is the "detail" string in SQLite 3.36+).
    fn query_plan(store: &SqliteStore, sql: &str) -> String {
        store
            .conn
            .query_row(&format!("EXPLAIN QUERY PLAN {sql}"), [], |row| row.get(3))
            .unwrap()
    }

    /// The traversal hot paths must stay index-only.
    ///
    /// Asserts on the **exact SQL `iter_edges_from_kind` and `iter_edges_to`
    /// issue**, not a hand-written stand-in. The earlier version of this test
    /// hardcoded `SELECT src FROM edges WHERE dst=? AND kind=?` while the code
    /// had moved to `SELECT src, kind, provenance ...`, and it only checked that
    /// the plan *named* the index — which it does whether or not the index
    /// covers. RFC-027's `provenance` column therefore took both queries off
    /// their covering indexes with the guard test still green. Asserting
    /// "COVERING INDEX" against the real query is what makes that lapse visible.
    #[test]
    fn edge_traversal_queries_stay_index_only() {
        let store = SqliteStore::open_in_memory().unwrap();

        let forward = query_plan(
            &store,
            "SELECT dst, provenance FROM edges WHERE src = 1 AND kind = 'ref/call'",
        );
        assert!(
            forward.contains("COVERING INDEX idx_edges_src_kind_cov"),
            "iter_edges_from_kind must be index-only; EXPLAIN detail: {forward}",
        );

        let reverse = query_plan(
            &store,
            "SELECT src, kind, provenance FROM edges WHERE dst = 1",
        );
        assert!(
            reverse.contains("COVERING INDEX idx_edges_dst_kind_cov"),
            "iter_edges_to must be index-only; EXPLAIN detail: {reverse}",
        );
    }

    /// Review issue B: a dev database an earlier revision of this branch stamped
    /// above the collapsed v23 (at 24 or 25) must regain the covering edge
    /// indexes on open, and it must do so **without** bumping the schema version
    /// — the collapse keeps the max at v23 on purpose.
    ///
    /// The migration runner cannot reach such a database (`version() > current`
    /// is never true for a stamp already ahead of the max), so the heal is the
    /// ungated `reconcile_provenance_indexes_if_needed` step that runs at every
    /// writable open. `edge_traversal_queries_stay_index_only` cannot catch this
    /// regression: it builds a fresh store that lands on the collapsed schema
    /// correctly.
    #[test]
    fn a_database_stranded_above_v23_regains_covering_indexes_on_open() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("graph.db");

        // A fresh open lands on the collapsed v23 with the wide indexes; roll it
        // back to the pre-RFC-027 narrow ones and stamp 24 to mimic the earlier
        // branch revision that numbered the index rework v25.
        {
            let store = SqliteStore::open(&db_path).unwrap();
            store
                .conn
                .execute_batch(
                    "DROP INDEX IF EXISTS idx_edges_src_kind_cov; \
                     CREATE INDEX idx_edges_src_kind_cov ON edges(src, kind, dst); \
                     DROP INDEX IF EXISTS idx_edges_dst_kind_cov; \
                     CREATE INDEX idx_edges_dst_kind_cov ON edges(dst, kind, src); \
                     DROP INDEX IF EXISTS idx_edges_live_provenance;",
                )
                .unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', '24') \
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [],
                )
                .unwrap();

            let reverse = query_plan(
                &store,
                "SELECT src, kind, provenance FROM edges WHERE dst = 1",
            );
            assert!(
                !reverse.contains("COVERING INDEX"),
                "precondition: the stranded db must not already cover this: {reverse}"
            );
        }

        // Reopen writable: the runner applies nothing (24 > 23 is false), and the
        // ungated reconcile widens the indexes.
        let store = SqliteStore::open(&db_path).unwrap();

        let forward = query_plan(
            &store,
            "SELECT dst, provenance FROM edges WHERE src = 1 AND kind = 'ref/call'",
        );
        assert!(
            forward.contains("COVERING INDEX idx_edges_src_kind_cov"),
            "the reconcile must restore the forward covering index: {forward}"
        );
        let reverse = query_plan(
            &store,
            "SELECT src, kind, provenance FROM edges WHERE dst = 1",
        );
        assert!(
            reverse.contains("COVERING INDEX idx_edges_dst_kind_cov"),
            "the reconcile must restore the reverse covering index: {reverse}"
        );
        let live_count = query_plan(
            &store,
            "SELECT count(*) FROM edges WHERE provenance = 'live'",
        );
        assert!(
            live_count.contains("idx_edges_live_provenance"),
            "the reconcile must restore the partial live-provenance index: {live_count}"
        );

        // The stamp is deliberately left untouched: the collapse keeps the max
        // schema version at 23, and re-stamping is neither needed (the runner is
        // a no-op here) nor safe once a future migration reuses 24/25.
        let version: String = store
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            version, "24",
            "the reconcile heals indexes without bumping the schema version"
        );
    }

    #[test]
    fn put_and_get_node_round_trip() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = sample_node("fn:rt");
        let id = store.put_node(&n).unwrap();
        let back = store.get_node(id).unwrap().expect("node missing");
        assert_eq!(back, n);
    }

    #[test]
    fn put_and_iter_edges() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        let c = sample_node("fn:c");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&c).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(a.id, c.id, EdgeKind::Depends))
            .unwrap();
        // Insert the same edge twice — should be a no-op.
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();

        let mut edges = store.iter_edges_from(a.id).unwrap();
        edges.sort_by_key(|e| (e.dst.0, e.kind.as_str()));
        assert_eq!(edges.len(), 2);
        assert!(edges
            .iter()
            .any(|e| e.dst == b.id && e.kind == EdgeKind::RefCall));
        assert!(edges
            .iter()
            .any(|e| e.dst == c.id && e.kind == EdgeKind::Depends));
    }

    #[test]
    fn get_missing_node_returns_none() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(store.get_node(NodeId(123_456_789)).unwrap().is_none());
    }

    #[test]
    fn wal_journal_mode_enabled_on_file_backed_store() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("graph.db");
        let store = SqliteStore::open(&db_path).unwrap();
        let mode = store.journal_mode().unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }

    #[test]
    fn reopening_existing_db_keeps_data() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("graph.db");
        let n = sample_node("fn:persist");
        let id = {
            let mut store = SqliteStore::open(&db_path).unwrap();
            store.put_node(&n).unwrap()
        };
        let store = SqliteStore::open(&db_path).unwrap();
        assert_eq!(store.get_node(id).unwrap().as_ref(), Some(&n));
    }

    #[test]
    fn unification_prefers_higher_priority_candidate_over_closer_line() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // SCIP definition at line 10. Candidates ordered most → least
        // specific: method:Server.Serve (priority 0), fn:Serve (priority 1).
        // The lower-priority node is CLOSER (line 11, delta 1) than the
        // higher-priority node (line 13, delta 3) — priority must still win.
        let hi = Node::new(
            VName::new("c", "main", "src/s.go", "go", "method:Server.Serve"),
            "method",
        )
        .with_line(13);
        let lo = Node::new(
            VName::new("c", "main", "src/s.go", "go", "fn:Serve"),
            "function",
        )
        .with_line(11);
        store.put_node(&hi).unwrap();
        store.put_node(&lo).unwrap();

        let candidates = vec!["method:Server.Serve".to_string(), "fn:Serve".to_string()];
        let found = store
            .find_ts_node_for_unification("c", "src/s.go", &candidates, 10, 5)
            .unwrap();
        assert_eq!(found, Some(hi.id), "higher-priority candidate must win");

        // When the higher-priority signature matches nothing, the closer
        // lower-priority node wins on line distance as before.
        let candidates = vec!["method:Other.Serve".to_string(), "fn:Serve".to_string()];
        let found = store
            .find_ts_node_for_unification("c", "src/s.go", &candidates, 10, 5)
            .unwrap();
        assert_eq!(found, Some(lo.id), "falls back to line proximity");
    }

    #[test]
    fn unification_matches_by_span_containment_across_wide_annotation_gap() {
        // E6: the SCIP def line sits inside the Phase A node's [line, end_line]
        // span but farther than max_delta below the name line (heavy
        // annotation / decorator / doc-comment block). The old ±max_delta
        // proximity gate silently dropped this, leaving an orphaned SCIP twin
        // that stole the node's ref/call edges. Span-containment must unify it.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let ts = Node::new(
            VName::new("c", "main", "src/S.java", "java", "fn:handle"),
            "function",
        )
        .with_line(5)
        .with_end_line(30);
        store.put_node(&ts).unwrap();

        let candidates = vec!["fn:handle".to_string()];
        // SCIP def anchored at line 12 — delta 7 from the name line 5, beyond
        // the max_delta of 5, but well inside the [5, 30] body span.
        let found = store
            .find_ts_node_for_unification("c", "src/S.java", &candidates, 12, 5)
            .unwrap();
        assert_eq!(
            found,
            Some(ts.id),
            "span-containment must unify across the annotation gap"
        );
    }

    #[test]
    fn unification_prefers_narrowest_containing_span() {
        // E6: when two candidate spans both contain the SCIP def line (a method
        // nested inside an outer definition that share a same-leaf `fn:` sig
        // candidate), the narrowest containing span is the correct target.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let outer = Node::new(
            VName::new("c", "main", "src/n.go", "go", "fn:run"),
            "function",
        )
        .with_line(1)
        .with_end_line(40);
        let inner = Node::new(
            VName::new("c", "main", "src/n.go", "go", "method:Job.run"),
            "method",
        )
        .with_line(10)
        .with_end_line(20);
        store.put_node(&outer).unwrap();
        store.put_node(&inner).unwrap();

        // Candidates cover both signatures; SCIP def at line 15 is inside both
        // [1,40] and [10,20] — the narrower [10,20] must win.
        let candidates = vec!["method:Job.run".to_string(), "fn:run".to_string()];
        let found = store
            .find_ts_node_for_unification("c", "src/n.go", &candidates, 15, 5)
            .unwrap();
        assert_eq!(found, Some(inner.id), "narrowest containing span wins");
    }

    #[test]
    fn phase_a_indexed_paths_excludes_scip_only_and_file_stub() {
        // #780: a file the tree-sitter parser indexed has a real def node; a file
        // only the SCIP tool saw (gitignored vendored code) has just SCIP defs
        // (plus a `file` stub). Only the former counts as Phase-A-indexed.
        let mut store = SqliteStore::open_in_memory().unwrap();
        // App file: a real tree-sitter method node.
        store
            .put_node(&Node::new(
                VName::new("c", "main", "lib/app.rb", "ruby", "method:App.run"),
                "method",
            ))
            .unwrap();
        // Vendored file: only a SCIP definition node and a `file` stub.
        store
            .put_node(&Node::new(
                VName::new("c", "main", "vendor/gem.rb", "ruby", "scip:vendor/gem.rb:x"),
                "definition",
            ))
            .unwrap();
        store
            .put_node(&Node::new(
                VName::new("c", "main", "vendor/gem.rb", "ruby", "file"),
                "file",
            ))
            .unwrap();

        let paths = store.phase_a_indexed_paths("c").unwrap();
        assert!(paths.contains("lib/app.rb"), "app file is indexed");
        assert!(
            !paths.contains("vendor/gem.rb"),
            "scip-only vendored file must not count as indexed"
        );
    }

    #[test]
    fn edge_sites_dedup_on_reinsert() {
        let store = SqliteStore::open_in_memory().unwrap();
        // The composite PK makes INSERT OR IGNORE an actual dedup: the same
        // (src, dst, kind, line) row inserted twice must yield one row.
        store
            .conn
            .execute(
                "INSERT OR IGNORE INTO edge_sites(src, dst, kind, line) VALUES(1, 2, 'ref/call', 7)",
                [],
            )
            .unwrap();
        store
            .conn
            .execute(
                "INSERT OR IGNORE INTO edge_sites(src, dst, kind, line) VALUES(1, 2, 'ref/call', 7)",
                [],
            )
            .unwrap();
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM edge_sites", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn reference_sites_returns_ordered_deduped_path_line() {
        // #299: reference_sites joins edge_sites.src → nodes.path and returns
        // deterministic (path, line) order, deduped by the edge_sites PK.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "file"),
            "file",
        );
        let b = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "file"),
            "file",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "svc.rs", "rust", "fn:charge"),
            "fn",
        );
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&callee).unwrap();

        // Out-of-order + a duplicate (a.rs:10 twice) to exercise ORDER BY + PK dedup.
        store
            .record_edge_sites(&[
                (a.id, callee.id, 10, None),
                (b.id, callee.id, 3, None),
                (a.id, callee.id, 2, None),
                (a.id, callee.id, 10, None),
            ])
            .unwrap();

        let sites = store.reference_sites(callee.id).unwrap();
        assert_eq!(
            sites,
            vec![
                travsr_core::RefSite {
                    path: "a.rs".into(),
                    line: 2,
                    heuristic: false
                },
                travsr_core::RefSite {
                    path: "a.rs".into(),
                    line: 10,
                    heuristic: false
                },
                travsr_core::RefSite {
                    path: "b.rs".into(),
                    line: 3,
                    heuristic: false
                },
            ]
        );
    }

    #[test]
    fn record_edge_sites_persists_and_backfills_occurrence_column() {
        // RFC-027 #813 P2: a recorded occurrence round-trips its byte column;
        // a later NULL-col re-record keeps the stored col (no duplicate row, no
        // clobber); and a NULL-col row is filled when a real col arrives later.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:caller"),
            "fn",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:callee"),
            "fn",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();

        // First write carries a real byte column.
        store
            .record_edge_sites(&[(caller.id, callee.id, 7, Some(12))])
            .unwrap();
        let col: Option<i64> = store
            .conn
            .query_row(
                "SELECT col FROM edge_sites WHERE src=?1 AND dst=?2 AND line=7",
                params![node_id_to_i64(caller.id), node_id_to_i64(callee.id)],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(col, Some(12), "the byte column must round-trip");

        // A re-record with no column must not duplicate the row nor clobber col.
        store
            .record_edge_sites(&[(caller.id, callee.id, 7, None)])
            .unwrap();
        let (count, col): (i64, Option<i64>) = store
            .conn
            .query_row(
                "SELECT COUNT(*), MAX(col) FROM edge_sites WHERE src=?1 AND dst=?2 AND line=7",
                params![node_id_to_i64(caller.id), node_id_to_i64(callee.id)],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 1, "re-recording must not create a duplicate row");
        assert_eq!(
            col,
            Some(12),
            "a NULL-col re-record must not clobber the col"
        );

        // A NULL-col row is filled when a real column arrives later.
        store
            .record_edge_sites(&[(caller.id, callee.id, 9, None)])
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 9, Some(4))])
            .unwrap();
        let col: Option<i64> = store
            .conn
            .query_row(
                "SELECT col FROM edge_sites WHERE src=?1 AND dst=?2 AND line=9",
                params![node_id_to_i64(caller.id), node_id_to_i64(callee.id)],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(col, Some(4), "a later real col must fill a NULL col");
    }

    #[test]
    fn reference_sites_includes_ref_field_occurrences() {
        // #757: a field node's use-sites are recorded under kind 'ref/field'
        // (a read is not a call). find_references must surface them, so
        // reference_sites spans both 'ref/call' and 'ref/field'.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "user.rs", "rust", "method:User.run"),
            "method",
        );
        let field = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "store.rs", "rust", "field:Store.conn"),
            "field",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&field).unwrap();

        store
            .record_field_sites(&[(caller.id, field.id, 14, None)])
            .unwrap();

        // Stored under 'ref/field', not 'ref/call'.
        let call_rows: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM edge_sites WHERE kind = 'ref/call'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(call_rows, 0, "field ref must not be a ref/call row");

        let sites = store.reference_sites(field.id).unwrap();
        assert_eq!(
            sites,
            vec![travsr_core::RefSite {
                path: "user.rs".into(),
                line: 14,
                heuristic: false
            }]
        );
    }

    #[test]
    fn record_field_sites_skips_zero_line_and_self_loop() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "field:A.x"),
            "field",
        );
        let m = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "method:B.f"),
            "method",
        );
        store.put_node(&n).unwrap();
        store.put_node(&m).unwrap();
        store
            .record_field_sites(&[(m.id, n.id, 0, None), (n.id, n.id, 5, None)])
            .unwrap();
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM edge_sites", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0, "zero-line and self-loop rows are skipped");
    }

    #[test]
    fn reference_sites_empty_when_no_occurrences() {
        let store = SqliteStore::open_in_memory().unwrap();
        let unknown = travsr_core::VName::new("c", "", "x.rs", "rust", "fn:nope").id();
        assert!(store.reference_sites(unknown).unwrap().is_empty());
    }

    #[test]
    fn language_has_edge_sites_distinguishes_missing_index_from_zero_refs() {
        // #299 M1: a language with a recorded ref/call occurrence reads true; a
        // language whose def node exists but has no occurrence row reads false,
        // so the reader can say "index unavailable" instead of "0 references".
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "main.rs", "rust", "file"),
            "file",
        );
        let rust_fn = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "svc.rs", "rust", "fn:charge"),
            "function",
        );
        // Java def node exists but never receives an occurrence row.
        let java_fn = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "Svc.java", "java", "fn:charge"),
            "function",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&rust_fn).unwrap();
        store.put_node(&java_fn).unwrap();

        store
            .record_edge_sites(&[(caller.id, rust_fn.id, 12, None)])
            .unwrap();

        assert!(store.language_has_edge_sites("rust").unwrap());
        // Def node present, but no ref/call site -> index not built for java.
        assert!(!store.language_has_edge_sites("java").unwrap());
        // A language with no nodes at all is also "unavailable", not zero.
        assert!(!store.language_has_edge_sites("go").unwrap());
    }

    #[test]
    fn file_has_occurrences_is_scoped_to_the_file_not_the_language() {
        // #450 / #551 review: the gate must answer "was THIS file analysed",
        // not "was this language analysed". A language can be partially covered
        // — one analysed file and twenty untouched ones — and a symbol in an
        // untouched file must not inherit the analysed file's confidence.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "covered.rs", "rust", "fn:caller"),
            "function",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "covered.rs", "rust", "fn:callee"),
            "function",
        );
        // Same language, different file, never receives an occurrence row.
        let untouched = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "untouched.rs", "rust", "fn:orphan"),
            "function",
        );
        // A non-callable kind: the first cut of this check filtered on
        // kind IN ('function','method') and would have made this file invisible.
        let type_only = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "types.rs", "rust", "struct:Config"),
            "struct",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&untouched).unwrap();
        store.put_node(&type_only).unwrap();

        store
            .record_edge_sites(&[(caller.id, callee.id, 7, None)])
            .unwrap();

        assert!(store.file_has_occurrences("covered.rs").unwrap());
        // Same language, and the language gate would pass — but this file has
        // no occurrence row, so no definitive claim can be made about it.
        assert!(!store.file_has_occurrences("untouched.rs").unwrap());
        assert!(!store.file_has_occurrences("types.rs").unwrap());
        // The attribution fallback's empty default is not a real file.
        assert!(!store.file_has_occurrences("").unwrap());
        // Contrast: the language-wide gate says "covered", which is exactly the
        // over-confidence #450 is about.
        assert!(store.language_has_edge_sites("rust").unwrap());
    }

    #[test]
    fn language_occurrence_coverage_counts_distinct_files_across_all_kinds() {
        // Context metric, not the gate. Must count every kind: restricting to
        // callables inflated the ratio by excluding type-only files, which is
        // what #551 review caught (161/169 = 95% vs 161/221 = 72% on the real
        // graph).
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:a"),
            "function",
        );
        let b = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:b"),
            "function",
        );
        // Two symbols in one file must not count that file twice.
        let b2 = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:b2"),
            "function",
        );
        // Type-only file: no callable, still part of the language's surface.
        let t = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "t.rs", "rust", "struct:T"),
            "struct",
        );
        for n in [&a, &b, &b2, &t] {
            store.put_node(n).unwrap();
        }
        store.record_edge_sites(&[(a.id, b.id, 3, None)]).unwrap();

        // a.rs and b.rs carry occurrences; t.rs does not. Three distinct files.
        let (with_occ, total) = store.language_occurrence_coverage("rust").unwrap();
        assert_eq!(with_occ, 2, "a.rs and b.rs, each counted once");
        assert_eq!(
            total, 3,
            "type-only t.rs must be visible in the denominator"
        );

        // A language with no nodes yields (0, 0) rather than dividing by zero.
        assert_eq!(store.language_occurrence_coverage("go").unwrap(), (0, 0));
        // Empty language matches language_has_edge_sites' treatment.
        assert_eq!(store.language_occurrence_coverage("").unwrap(), (0, 0));
    }

    #[test]
    fn fn_nodes_by_leaf_name_matches_bare_and_qualified_leaves() {
        // #299 R1: the daemon leaf-name fallback recovers a qualified
        // fn:Type.method / method:Type.method node from a bare leaf, restricts
        // to callable kinds, and escapes LIKE metacharacters so `_` is literal.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let nodes = [
            ("a.rs", "fn:describe", "function"),
            ("a.rs", "fn:Animal.describe", "function"),
            ("b.rs", "method:Zoo.add", "method"),
            ("c.rs", "fn:Zoo.announce_all", "method"),
            // `_` must be escaped: an `X` in the same slot must NOT be matched
            // when searching the underscore name.
            ("c.rs", "fn:Zoo.announceXall", "method"),
            // Non-callable kind with a matching leaf must be excluded.
            ("d.rs", "class:describe", "class"),
        ];
        for (path, sig, kind) in nodes {
            let n =
                travsr_core::Node::new(travsr_core::VName::new("c", "", path, "rust", sig), kind);
            store.put_node(&n).unwrap();
        }

        let mut got: Vec<String> = store
            .fn_nodes_by_leaf_name(&["describe".to_string(), "add".to_string()])
            .unwrap()
            .into_iter()
            .map(|(_, sig, _, _)| sig)
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                "fn:Animal.describe".to_string(),
                "fn:describe".to_string(),
                "method:Zoo.add".to_string(),
            ]
        );

        // `announce_all` matches only the literal underscore node, never the
        // `announceXall` wildcard trap -> proves the LIKE `_` is escaped.
        let hit: Vec<String> = store
            .fn_nodes_by_leaf_name(&["announce_all".to_string()])
            .unwrap()
            .into_iter()
            .map(|(_, sig, _, _)| sig)
            .collect();
        assert_eq!(hit, vec!["fn:Zoo.announce_all".to_string()]);

        // Empty input short-circuits with no query.
        assert!(store.fn_nodes_by_leaf_name(&[]).unwrap().is_empty());
    }

    #[test]
    fn record_edge_sites_skips_zero_line() {
        // line == 0 means "unknown occurrence line" and must not be stored.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "file"),
            "file",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:f"),
            "fn",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store
            .record_edge_sites(&[
                (caller.id, callee.id, 0, None),
                (caller.id, callee.id, 5, None),
            ])
            .unwrap();
        let sites = store.reference_sites(callee.id).unwrap();
        assert_eq!(
            sites,
            vec![travsr_core::RefSite {
                path: "a.rs".into(),
                line: 5,
                heuristic: false
            }]
        );
    }

    #[test]
    fn record_edge_sites_skips_self_loop() {
        // #299 F8: a site whose src == dst has no backing ref/call edge (edges
        // drop self-loops), so recording it would diverge find_references from
        // get_callers. It must be skipped.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:f"),
            "function",
        );
        store.put_node(&n).unwrap();
        store.record_edge_sites(&[(n.id, n.id, 5, None)]).unwrap();
        assert!(store.reference_sites(n.id).unwrap().is_empty());
    }

    #[test]
    fn language_has_edge_sites_is_evidence_based_not_marker_based() {
        // #299 F6: "index built for a language" must require an actual occurrence
        // row, not merely that Phase B was invoked. A sidecar can run to
        // completion yet produce nothing when its analyzer tool is missing
        // (e.g. scip-php absent) — that language must read false so
        // find_references says "unavailable", not a confident "0 references".
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "main.go", "go", "fn:main"),
            "function",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "svc.go", "go", "fn:Charge"),
            "function",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 7, None)])
            .unwrap();
        // go has a real occurrence row -> built.
        assert!(store.language_has_edge_sites("go").unwrap());
        // php produced no occurrence rows (analyzer absent) -> not built,
        // regardless of whether its Phase B sidecar "ran".
        assert!(!store.language_has_edge_sites("php").unwrap());
    }

    #[test]
    fn language_has_edge_sites_detects_via_src_when_dst_language_empty() {
        // #299 F6: an occurrence whose `dst` definition node carries an empty
        // language must still count for the calling file's language, since the
        // enclosing `src` node reliably carries it (fixes the empty-dst
        // false-negative). And querying the empty language itself is always false.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:caller"),
            "function",
        );
        // Def node with an empty language (file_node_for_attribution style).
        let blank = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "", "fn:f"),
            "function",
        );
        store.put_node(&caller).unwrap();
        store.put_node(&blank).unwrap();
        store
            .record_edge_sites(&[(caller.id, blank.id, 4, None)])
            .unwrap();
        // rust is detected via the src endpoint even though dst language is empty.
        assert!(store.language_has_edge_sites("rust").unwrap());
        // The empty language itself is never "built".
        assert!(!store.language_has_edge_sites("").unwrap());
    }

    #[test]
    fn fn_nodes_by_leaf_name_includes_fn_kind_and_chunks() {
        // #299 F11: the `fn` kind (some Phase A parsers) must be matched, and a
        // names slice larger than one chunk must still resolve every name.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let fnkind = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:Zoo.feed"),
            "fn",
        );
        store.put_node(&fnkind).unwrap();
        // 350 distinct leaf names (> NAMES_PER_CHUNK) including the real one.
        let mut names: Vec<String> = (0..350).map(|i| format!("leaf{i}")).collect();
        names.push("feed".to_string());
        let got: Vec<String> = store
            .fn_nodes_by_leaf_name(&names)
            .unwrap()
            .into_iter()
            .map(|(_, sig, _, _)| sig)
            .collect();
        assert_eq!(got, vec!["fn:Zoo.feed".to_string()]);
    }

    #[test]
    fn reindex_replace_purges_owned_edge_sites() {
        // #299 F7: re-indexing a file clears its OWNED occurrence rows (src in the
        // file) but preserves inbound sites (references from other files).
        let mut store = SqliteStore::open_in_memory().unwrap();
        let owned_src = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:caller"),
            "function",
        );
        let callee = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:target"),
            "function",
        );
        let external = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:ext"),
            "function",
        );
        store.put_node(&owned_src).unwrap();
        store.put_node(&callee).unwrap();
        store.put_node(&external).unwrap();
        // Owned site (src in a.rs) + inbound site (src in b.rs → dst in a.rs).
        store
            .record_edge_sites(&[
                (owned_src.id, callee.id, 3, None),
                (external.id, callee.id, 9, None),
            ])
            .unwrap();

        // Re-index a.rs with the same nodes.
        store
            .reindex_replace(
                "c",
                "a.rs",
                &[owned_src.clone(), callee.clone()],
                &[],
                "hash1",
                None,
            )
            .unwrap();

        let sites = store.reference_sites(callee.id).unwrap();
        // The owned a.rs:3 site is gone; the inbound b.rs:9 site survives.
        assert_eq!(
            sites,
            vec![travsr_core::RefSite {
                path: "b.rs".into(),
                line: 9,
                heuristic: false
            }]
        );
    }

    #[test]
    fn write_file_graphs_batch_purges_owned_edge_sites() {
        // The incremental write path deletes a re-parsed file's owned edges but
        // used to leave its occurrence rows behind. `reference_sites` LEFT JOINs
        // `edges`, so an orphaned row came back `heuristic = false`, i.e. a stale
        // line served as resolved fact. Same ownership rule as
        // `reindex_replace_purges_owned_edge_sites`: owned rows go, inbound rows
        // stay.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |path: &str, sig: &str| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", path, "rust", sig),
                "function",
            )
        };
        let owned_src = mk("a.rs", "fn:caller");
        let callee = mk("a.rs", "fn:target");
        let external = mk("b.rs", "fn:ext");
        for node in [&owned_src, &callee, &external] {
            store.put_node(node).unwrap();
        }
        store
            .record_edge_sites(&[
                // Owned: src lives in the file being re-parsed.
                (owned_src.id, callee.id, 3, None),
                // Inbound: src lives in another file, dst in this one.
                (external.id, callee.id, 9, None),
            ])
            .unwrap();

        let batch = vec![FileGraph {
            vname_path: "a.rs".into(),
            new_hash: "hash1".into(),
            nodes: vec![owned_src.clone(), callee.clone()],
            edges: vec![],
            source: None,
        }];
        store.write_file_graphs_batch(&batch, false).unwrap();

        assert_eq!(
            store.reference_sites(callee.id).unwrap(),
            vec![travsr_core::RefSite {
                path: "b.rs".into(),
                line: 9,
                heuristic: false
            }],
            "owned a.rs:3 site must be purged, inbound b.rs:9 site must survive"
        );
    }

    // ── RFC-027 #813: scope-aware preservation (Mechanism A) ──────────────────

    /// Provenance of a specific edge, or `None` if the edge is absent. Tests
    /// read `store.conn` directly, as the reconcile/index tests above do.
    fn edge_provenance(store: &SqliteStore, src: NodeId, dst: NodeId) -> Option<String> {
        store
            .conn
            .query_row(
                "SELECT provenance FROM edges WHERE src=?1 AND dst=?2",
                params![node_id_to_i64(src), node_id_to_i64(dst)],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .unwrap()
    }

    /// Two callers (`a`, `b`) in one file, each with a committed `lsif` edge to a
    /// target. A pure body edit that changes only `a`'s body must leave `b`'s
    /// committed edge exactly as it was — same row, same `lsif` provenance,
    /// occurrence site intact — while `a`'s edge is purged so the live lane can
    /// re-resolve it. This is the ~71% -> ~99% win, and it is proven by `b`'s
    /// edge keeping `lsif` rather than being re-derived as `tree-sitter`.
    #[test]
    fn reindex_replace_preserves_unchanged_definition_committed_edges() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        // a: lines 1-2, b: lines 4-5, plus two callee targets.
        let a = mk("fn:a", 1, 2);
        let b = mk("fn:b", 4, 5);
        let x = mk("fn:x", 7, 8);
        let y = mk("fn:y", 10, 11);
        let nodes = vec![a.clone(), b.clone(), x.clone(), y.clone()];
        let ts_edges = vec![
            Edge::new(a.id, x.id, EdgeKind::RefCall),
            Edge::new(b.id, y.id, EdgeKind::RefCall),
        ];

        let content_v1 =
            "fn a() {\n  x();\n}\n\nfn b() {\n  y();\n}\n\nfn x() {\n}\n\nfn y() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h1", Some(content_v1))
            .unwrap();

        // Commit ratification would relabel these as semantic truth; simulate it.
        store
            .put_edge_lsif(&Edge::new(a.id, x.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge_lsif(&Edge::new(b.id, y.id, EdgeKind::RefCall))
            .unwrap();
        store
            .record_edge_sites(&[(a.id, x.id, 2, None), (b.id, y.id, 5, None)])
            .unwrap();
        assert_eq!(edge_provenance(&store, a.id, x.id).as_deref(), Some("lsif"));
        assert_eq!(edge_provenance(&store, b.id, y.id).as_deref(), Some("lsif"));

        // v2: edit only a's body (line 2). b, x, y are byte-identical. The new
        // parse no longer sees a's call (a's body changed), but still sees b's.
        let content_v2 =
            "fn a() {\n  z();\n}\n\nfn b() {\n  y();\n}\n\nfn x() {\n}\n\nfn y() {\n}\n";
        let ts_edges_v2 = vec![Edge::new(b.id, y.id, EdgeKind::RefCall)];
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges_v2, "h2", Some(content_v2))
            .unwrap();

        // b unchanged: its committed edge is preserved, not re-derived — proven
        // by the provenance staying `lsif` rather than being rebuilt as
        // `tree-sitter`.
        assert_eq!(
            edge_provenance(&store, b.id, y.id).as_deref(),
            Some("lsif"),
            "b's committed edge must be preserved with its lsif provenance"
        );
        // RFC-027 #813 P2: b did not move (delta 0), so its occurrence row is
        // re-recorded on its current line rather than dropped, keeping
        // `find_references` correct mid-edit.
        assert_eq!(
            store.reference_sites(y.id).unwrap(),
            vec![travsr_core::RefSite {
                path: "a.rs".into(),
                line: 5,
                heuristic: false
            }],
            "a preserved definition's occurrence is remapped onto its current line"
        );
        // a changed: its committed edge is purged so the live lane re-resolves it.
        assert_eq!(
            edge_provenance(&store, a.id, x.id),
            None,
            "the edited definition's stale edge must be purged"
        );
    }

    /// RFC-027 #813 P2: on a body edit, `reindex_replace` captures the CHANGED
    /// definition's committed occurrences (for the live lane to enumerate as
    /// editor targets) and NOT the preserved definitions', carrying each
    /// occurrence's column and the reference's leaf name, never the stale dst.
    #[test]
    fn reindex_replace_captures_changed_def_occurrences_for_the_live_lane() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        let a = mk("fn:a", 1, 2);
        let b = mk("fn:b", 4, 5);
        let x = mk("fn:x", 7, 8);
        let y = mk("fn:y", 10, 11);
        let nodes = vec![a.clone(), b.clone(), x.clone(), y.clone()];
        let ts_edges = vec![
            Edge::new(a.id, x.id, EdgeKind::RefCall),
            Edge::new(b.id, y.id, EdgeKind::RefCall),
        ];
        let content_v1 =
            "fn a() {\n  x();\n}\n\nfn b() {\n  y();\n}\n\nfn x() {\n}\n\nfn y() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h1", Some(content_v1))
            .unwrap();
        // Committed occurrences, each with a byte column.
        store
            .record_edge_sites(&[(a.id, x.id, 2, Some(2)), (b.id, y.id, 5, Some(2))])
            .unwrap();

        // Edit only a's body; b, x, y stay byte-identical (preserved). The x
        // call itself stays on line 2 at col 2 (a trailing comment changes the
        // body hash without moving the reference), so the occurrence is still a
        // live target to enumerate.
        let content_v2 =
            "fn a() {\n  x(); // edit\n}\n\nfn b() {\n  y();\n}\n\nfn x() {\n}\n\nfn y() {\n}\n";
        let report = store
            .reindex_replace("c", "a.rs", &nodes, &[], "h2", Some(content_v2))
            .unwrap();

        // Only a's occurrence is captured (b, x, y are preserved). a did not
        // move (delta 0) and the x call is still on line 2, so it is captured on
        // line 2 with its column and the callee's leaf name, not the stale dst.
        assert_eq!(
            report.changed_occurrences,
            vec![travsr_core::ChangedOccurrence {
                src: a.id,
                line: 2,
                col: Some(2),
                kind: "ref/call".into(),
                name: "x".into(),
            }],
            "the changed definition's occurrence must be captured with col and name"
        );
    }

    /// Issue #816 defect 1: an edit that inserts a line INSIDE the changed
    /// definition's body shifts every occurrence below the edit point, but the
    /// definition's start does not move (start delta 0). The occurrence must be
    /// captured on its CURRENT line (re-anchored by its preserved byte column),
    /// not the stale committed line the start delta alone would leave.
    #[test]
    fn reindex_replace_remaps_a_changed_defs_occurrence_below_an_intra_body_insert() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        // v1: a is [1,3] and calls x at line 2, col 2. b and x are preserved
        // across the edit so the changed-occurrence capture runs.
        let a1 = mk("fn:a", 1, 3);
        let b1 = mk("fn:b", 5, 6);
        let x1 = mk("fn:x", 8, 9);
        let nodes_v1 = vec![a1.clone(), b1.clone(), x1.clone()];
        let content_v1 = "fn a() {\n  x();\n}\n\nfn b() {\n}\n\nfn x() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes_v1, &[], "h1", Some(content_v1))
            .unwrap();
        store
            .record_edge_sites(&[(a1.id, x1.id, 2, Some(2))])
            .unwrap();

        // v2: insert `  w;` at the top of a's body. a grows to [1,4]; its start
        // is unchanged, so the start delta is 0, but the x call moved to line 3.
        // b and x shift down by one and stay byte-identical (preserved).
        let a2 = mk("fn:a", 1, 4);
        let b2 = mk("fn:b", 6, 7);
        let x2 = mk("fn:x", 9, 10);
        let nodes_v2 = vec![a2.clone(), b2.clone(), x2.clone()];
        let content_v2 = "fn a() {\n  w;\n  x();\n}\n\nfn b() {\n}\n\nfn x() {\n}\n";
        let report = store
            .reindex_replace("c", "a.rs", &nodes_v2, &[], "h2", Some(content_v2))
            .unwrap();

        // The occurrence is captured on its current line 3, not the stale 2 the
        // start delta alone would give.
        assert_eq!(
            report.changed_occurrences,
            vec![travsr_core::ChangedOccurrence {
                src: a2.id,
                line: 3,
                col: Some(2),
                kind: "ref/call".into(),
                name: "x".into(),
            }],
            "a changed def's occurrence below an intra-body insert must remap to its current line"
        );
    }

    /// RFC-027 #813 P2: when an edit above a preserved definition shifts it down,
    /// the definition's occurrence rows are re-recorded on their current lines
    /// (old line plus the definition's shift), not dropped, so `find_references`
    /// stays correct mid-edit and the line matches what the next commit records.
    #[test]
    fn reindex_replace_remaps_a_shifted_preserved_definitions_occurrences() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        // The occurrence targets are external, so they need not be nodes here.
        let x = travsr_core::VName::new("c", "", "z.rs", "rust", "fn:x").id();
        let y = travsr_core::VName::new("c", "", "z.rs", "rust", "fn:y").id();
        // v1: a occupies lines 1-3, b occupies 5-7. b calls y at line 6.
        let a1 = mk("fn:a", 1, 3);
        let b1 = mk("fn:b", 5, 7);
        let content_v1 = "fn a() {\n  x();\n}\n\nfn b() {\n  y();\n}\n";
        store
            .reindex_replace(
                "c",
                "a.rs",
                &[a1.clone(), b1.clone()],
                &[],
                "h1",
                Some(content_v1),
            )
            .unwrap();
        // b's y-call carries a byte column; the block shift must keep it (the
        // body is byte-identical, so only the line moved).
        store
            .record_edge_sites(&[(a1.id, x, 2, None), (b1.id, y, 6, Some(2))])
            .unwrap();

        // v2: a grows by two lines (body changed), pushing b down to 7-9. b's
        // body is byte-identical, so it is preserved and shifts by +2.
        let a2 = mk("fn:a", 1, 5);
        let b2 = mk("fn:b", 7, 9);
        let content_v2 = "fn a() {\n  x();\n  x();\n  x();\n}\n\nfn b() {\n  y();\n}\n";
        store
            .reindex_replace(
                "c",
                "a.rs",
                &[a2.clone(), b2.clone()],
                &[],
                "h2",
                Some(content_v2),
            )
            .unwrap();

        // b's occurrence of y moved from line 6 to line 8 (delta +2).
        assert_eq!(
            store.reference_sites(y).unwrap(),
            vec![travsr_core::RefSite {
                path: "a.rs".into(),
                line: 8,
                heuristic: false
            }],
            "the preserved definition's occurrence must be remapped to its current line"
        );
        // RFC-027 #813 P2: the remap keeps the occurrence's byte column intact.
        let col: Option<i64> = store
            .conn
            .query_row(
                "SELECT col FROM edge_sites WHERE dst=?1 AND line=8",
                params![node_id_to_i64(y)],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(col, Some(2), "the remapped occurrence must keep its column");
        // a changed, so its occurrence was purged and not restored.
        assert_eq!(
            store.reference_sites(x).unwrap(),
            vec![],
            "the edited definition's occurrence is purged, to be re-derived at commit"
        );
    }

    /// Soundness gate: preservation is allowed only when the file's NodeId set is
    /// unchanged. If any symbol is added (the intra-file analogue of adding an
    /// import that could re-point an untouched call), no definition is preserved
    /// even with a byte-identical body — the whole file is re-derived. Proven by
    /// the unchanged definition's edge losing its `lsif` provenance.
    #[test]
    fn reindex_replace_does_not_preserve_when_a_symbol_is_added() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        let a = mk("fn:a", 1, 2);
        let y = mk("fn:y", 4, 5);
        let content_v1 = "fn a() {\n  y();\n}\n\nfn y() {\n}\n";
        store
            .reindex_replace(
                "c",
                "a.rs",
                &[a.clone(), y.clone()],
                &[Edge::new(a.id, y.id, EdgeKind::RefCall)],
                "h1",
                Some(content_v1),
            )
            .unwrap();
        store
            .put_edge_lsif(&Edge::new(a.id, y.id, EdgeKind::RefCall))
            .unwrap();
        assert_eq!(edge_provenance(&store, a.id, y.id).as_deref(), Some("lsif"));

        // v2: a's body is byte-identical, but a new function c is added. The
        // NodeId set changed, so this is not a pure body edit and nothing is
        // preserved — a's edge is re-derived as tree-sitter.
        let c = mk("fn:c", 7, 8);
        let content_v2 = "fn a() {\n  y();\n}\n\nfn y() {\n}\n\nfn c() {\n}\n";
        store
            .reindex_replace(
                "c",
                "a.rs",
                &[a.clone(), y.clone(), c.clone()],
                &[Edge::new(a.id, y.id, EdgeKind::RefCall)],
                "h2",
                Some(content_v2),
            )
            .unwrap();
        assert_eq!(
            edge_provenance(&store, a.id, y.id).as_deref(),
            Some("tree-sitter"),
            "an added symbol must disable preservation for the whole file"
        );
    }

    /// RFC-027 #813 (finding 1): a byte-identical body does not prove the types
    /// its references resolve through are unchanged. A caller `h` whose body is
    /// untouched but which references a definition `make` whose return type
    /// changed (`-> A` to `-> B`) must NOT keep its committed `h -> A.run` edge:
    /// truth is now `B.run`. The NodeId set is unchanged (names are stable), so
    /// only the depth-one caller demotion catches it. An unrelated preserved
    /// definition that references nothing changed still keeps its committed edge.
    #[test]
    fn reindex_replace_demotes_a_caller_of_a_type_changed_definition() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        // make: 1-3 (returns A), h: 5-7 (calls make(), then make().run()),
        // g: 9-11 (unrelated, calls free()), and the two committed targets.
        let make = mk("fn:make", 1, 3);
        let h = mk("fn:h", 5, 7);
        let g = mk("fn:g", 9, 11);
        let run = mk("method:A.run", 13, 14);
        let free = mk("fn:free", 16, 17);
        let nodes = vec![
            make.clone(),
            h.clone(),
            g.clone(),
            run.clone(),
            free.clone(),
        ];
        // Phase A tree-sitter emits only structural edges (DefinesBinding /
        // Depends / ResolvesTo), never RefCall, so the fresh parse carries no
        // h -> make edge. The demotion must therefore read the committed
        // reference edges, which Phase B ratifies below (put_edge_lsif), not
        // this parse's edges.
        let ts_edges: Vec<Edge> = Vec::new();
        let content_v1 = "fn make() -> A {\n  A\n}\n\nfn h() {\n  make().run();\n}\n\nfn g() {\n  free();\n}\n\nfn run() {\n}\n\nfn free() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h1", Some(content_v1))
            .unwrap();
        // Commit ratification: the resolved reference edges become committed
        // truth. h calls make (the receiver source) and make().run() (the
        // receiver-typed edge that goes stale when make's return type changes);
        // g calls free.
        for edge in [
            Edge::new(h.id, make.id, EdgeKind::RefCall),
            Edge::new(h.id, run.id, EdgeKind::RefCall),
            Edge::new(g.id, free.id, EdgeKind::RefCall),
        ] {
            store.put_edge_lsif(&edge).unwrap();
        }
        assert_eq!(
            edge_provenance(&store, h.id, run.id).as_deref(),
            Some("lsif")
        );
        assert_eq!(
            edge_provenance(&store, g.id, free.id).as_deref(),
            Some("lsif")
        );

        // v2: change only make's return type A -> B. h's and g's bodies are
        // byte-identical; the NodeId set is unchanged.
        let content_v2 = "fn make() -> B {\n  B\n}\n\nfn h() {\n  make().run();\n}\n\nfn g() {\n  free();\n}\n\nfn run() {\n}\n\nfn free() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h2", Some(content_v2))
            .unwrap();

        // h references the changed `make`, so it is demoted from preserved and
        // its stale committed edge is purged, to be re-resolved by the live lane.
        assert_eq!(
            edge_provenance(&store, h.id, run.id),
            None,
            "a caller of a type-changed definition must not keep its stale edge"
        );
        // g references only unchanged definitions, so it stays preserved.
        assert_eq!(
            edge_provenance(&store, g.id, free.id).as_deref(),
            Some("lsif"),
            "an unrelated preserved definition keeps its committed edge"
        );
    }

    /// A `type A = Foo` -> `type A = Bar` edit re-points every definition that
    /// resolves through `A`, but there is no type-reference `EdgeKind`, so no
    /// edge into `type:A` exists for the caller demotion to follow. A change to
    /// a TYPE_LEVEL_KINDS definition must therefore invalidate the whole file's
    /// preserved set, or `caller` keeps a committed edge to `Foo.go` when truth
    /// is now `Bar.go`.
    #[test]
    fn reindex_replace_invalidates_the_file_when_a_type_alias_changes() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, kind: &str, line: u32, end: u32| {
            travsr_core::Node::new(travsr_core::VName::new("c", "", "a.rs", "rust", sig), kind)
                .with_line(line)
                .with_end_line(end)
        };
        let alias = mk("type:A", "type", 1, 1);
        let caller = mk("fn:caller", "function", 3, 5);
        let go = mk("method:Foo.go", "method", 7, 8);
        let nodes = vec![alias.clone(), caller.clone(), go.clone()];
        // Tree-sitter emits nothing into `type:A`: the alias is used as a
        // parameter type, which is not an edge kind.
        let ts_edges: Vec<Edge> = Vec::new();
        let v1 = "type A = Foo;\n\nfn caller(a: A) {\n  a.go();\n}\n\nfn go() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h1", Some(v1))
            .unwrap();
        store
            .put_edge_lsif(&Edge::new(caller.id, go.id, EdgeKind::RefCall))
            .unwrap();
        assert_eq!(
            edge_provenance(&store, caller.id, go.id).as_deref(),
            Some("lsif")
        );

        // Only the alias RHS changes. `caller` is byte-identical and the NodeId
        // set is unchanged, so nothing but the type-level rule catches this.
        let v2 = "type A = Bar;\n\nfn caller(a: A) {\n  a.go();\n}\n\nfn go() {\n}\n";
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h2", Some(v2))
            .unwrap();

        assert_eq!(
            edge_provenance(&store, caller.id, go.id),
            None,
            "a type-level change must purge the file's committed edges, not preserve them"
        );
    }

    /// The demotion iterates to a fixpoint: demoting `b` (it references the
    /// changed `c`) must then demote `a`, which references `b`. A single pass
    /// over a snapshot of the changed set leaves `a` preserved with a stale edge.
    #[test]
    fn reindex_replace_demotion_reaches_a_transitive_caller() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let mk = |sig: &str, line: u32, end: u32| {
            travsr_core::Node::new(
                travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                "function",
            )
            .with_line(line)
            .with_end_line(end)
        };
        // c: 1-3 (edited), b: 5-7 (calls c), a: 9-11 (calls b), d: 13-15
        // (unrelated, calls free), free: 17-18, x: 20-21 (the committed target).
        let c = mk("fn:c", 1, 3);
        let b = mk("fn:b", 5, 7);
        let a = mk("fn:a", 9, 11);
        let d = mk("fn:d", 13, 15);
        let free = mk("fn:free", 17, 18);
        let x = mk("method:X.run", 20, 21);
        let nodes = vec![
            c.clone(),
            b.clone(),
            a.clone(),
            d.clone(),
            free.clone(),
            x.clone(),
        ];
        // The fresh parse carries no RefCall (Phase A never emits it); the call
        // chain and the committed targets are ratified through the committed
        // reference edges below, the way Phase B supplies them in production.
        let ts_edges: Vec<Edge> = Vec::new();
        let body = |c_body: &str| {
            format!(
                "fn c() {{\n  {c_body}\n}}\n\nfn b() {{\n  c().run();\n}}\n\nfn a() {{\n  b().run();\n}}\n\nfn d() {{\n  free();\n}}\n\nfn free() {{\n}}\n\nfn run() {{\n}}\n"
            )
        };
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h1", Some(&body("A")))
            .unwrap();
        // Commit ratification: the call chain (b -> c, a -> b) and each caller's
        // committed target (b/a -> X.run), plus d's off-chain call.
        for edge in [
            Edge::new(b.id, c.id, EdgeKind::RefCall),
            Edge::new(a.id, b.id, EdgeKind::RefCall),
            Edge::new(b.id, x.id, EdgeKind::RefCall),
            Edge::new(a.id, x.id, EdgeKind::RefCall),
            Edge::new(d.id, free.id, EdgeKind::RefCall),
        ] {
            store.put_edge_lsif(&edge).unwrap();
        }

        // Only c's body changes.
        store
            .reindex_replace("c", "a.rs", &nodes, &ts_edges, "h2", Some(&body("B")))
            .unwrap();

        assert_eq!(
            edge_provenance(&store, b.id, x.id),
            None,
            "the direct caller of the changed definition is demoted"
        );
        assert_eq!(
            edge_provenance(&store, a.id, x.id),
            None,
            "the transitive caller must be demoted too, not left with a stale edge"
        );
        assert_eq!(
            edge_provenance(&store, d.id, free.id).as_deref(),
            Some("lsif"),
            "a definition off the changed chain keeps its committed edge"
        );
    }

    /// The re-anchoring search stops at [`MAX_SNAP_RADIUS`], so an occurrence
    /// that cannot be found costs the cap rather than the definition's span,
    /// which is the whole buffer when the definition has no known `end_line`.
    #[test]
    fn snap_changed_occurrence_line_is_bounded_by_the_max_radius() {
        // Filler indented past the occurrence column, so the second tier (which
        // accepts the estimate when any token starts at the column) cannot fire
        // and this stays a test of the search cap alone.
        let mut lines: Vec<&str> = vec!["    other();"; 2000];
        let hit = "  foo();";
        // Within the cap: found. Estimate 100, true line 600 (radius 500).
        lines[599] = hit;
        assert_eq!(
            snap_changed_occurrence_line(&lines, "foo", 2, 100, 1, lines.len() as i64),
            Some(600)
        );
        // Beyond the cap: abstains and heals at commit rather than scanning the
        // rest of the buffer.
        lines[599] = "    other();";
        lines[1499] = hit;
        assert_eq!(
            snap_changed_occurrence_line(&lines, "foo", 2, 100, 1, lines.len() as i64),
            None
        );
    }

    /// RFC-027 #813 P2: an occurrence whose source spelling is not the committed
    /// callee's name is kept, not dropped. `Self { .. }` references the impl's
    /// type but is spelled `Self`, so the name search finds nothing while the
    /// stored column points at a perfectly good token. Measured on this
    /// repository this class was 94.5% of everything the name search rejected.
    #[test]
    fn an_occurrence_spelled_differently_from_its_callee_is_still_anchored() {
        let lines = vec![
            "impl Guard {",
            "    fn new() -> Self {",
            "        Self { a: 1 }",
            "    }",
            "}",
        ];
        // Callee leaf is `Guard`; column 8 of line 3 holds `Self`.
        assert_eq!(
            snap_changed_occurrence_line(&lines, "Guard", 8, 3, 1, 5),
            Some(3)
        );
        // A tuple field is spelled `0` while the callee node carries the field
        // name, and the same fallback anchors it.
        let tuple = vec!["fn f(s: S) -> u32 {", "    s.node.0", "}"];
        assert_eq!(
            snap_changed_occurrence_line(&tuple, "first", 11, 2, 1, 3),
            Some(2)
        );
    }

    /// The fallback requires a real token at the column, so a desugared
    /// occurrence whose column lands on an operator still abstains rather than
    /// pointing a language server at punctuation.
    #[test]
    fn an_occurrence_on_a_non_identifier_column_still_abstains() {
        let lines = vec!["fn f(x: A, y: A) -> i32 {", "    x + y", "}"];
        // `x + y` desugars to `Add::add`; column 6 is the `+`.
        assert_eq!(
            snap_changed_occurrence_line(&lines, "add", 6, 2, 1, 3),
            None
        );
        // Column 7 is the space after it, which is not a token start either.
        assert_eq!(
            snap_changed_occurrence_line(&lines, "add", 7, 2, 1, 3),
            None
        );
        // And the middle of an identifier is not a token start.
        let mid = vec!["fn f() {", "    other();", "}"];
        assert_eq!(snap_changed_occurrence_line(&mid, "zzz", 6, 2, 1, 3), None);
    }

    /// The name search still wins when the name IS findable: the fallback is a
    /// last resort at radius 0, never a replacement for re-anchoring a moved
    /// occurrence.
    #[test]
    fn the_name_search_beats_the_token_fallback() {
        let lines = vec!["fn f() {", "    other();", "    foo();", "}"];
        // Column 4 of the estimate line 2 holds `other`, so the fallback would
        // accept line 2; the name search must still find `foo` on line 3.
        assert_eq!(
            snap_changed_occurrence_line(&lines, "foo", 4, 2, 1, 4),
            Some(3)
        );
    }

    /// Property (acceptance criterion): under random body and interface edits, a
    /// preserved edge is never stale-wrong — a definition's edges survive a save
    /// iff its body is byte-identical AND the file's symbol set is unchanged;
    /// otherwise they are re-derived from the new parse. Fuzzes which
    /// definitions are edited and whether a symbol is added/removed, and checks
    /// the graph equals the intended post-edit graph every time.
    #[test]
    fn reindex_replace_preservation_never_produces_a_stale_edge() {
        // Deterministic LCG so a failure reproduces from the seed in the message.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };

        for iter in 0..200u32 {
            let mut store = SqliteStore::open_in_memory().unwrap();
            let mk = |sig: &str, line: u32, end: u32| {
                travsr_core::Node::new(
                    travsr_core::VName::new("c", "", "a.rs", "rust", sig),
                    "function",
                )
                .with_line(line)
                .with_end_line(end)
            };
            // Four callers f0..f3 (2 lines each) that each call target `t`.
            let callers: Vec<_> = (0..4)
                .map(|i| mk(&format!("fn:f{i}"), 1 + i * 3, 2 + i * 3))
                .collect();
            let t = mk("fn:t", 13, 14);
            let body = |bodies: &[u32]| {
                let mut s = String::new();
                for (i, b) in bodies.iter().enumerate() {
                    // 3 lines per caller (2 body lines + 1 blank), body byte varies.
                    s.push_str(&format!("fn f{i}() {{\n  call_{b}();\n}}\n"));
                }
                s.push_str("fn t() {\n}\n");
                s
            };

            let v1_bodies = [0u32, 0, 0, 0];
            let mut nodes: Vec<_> = callers.clone();
            nodes.push(t.clone());
            let v1_edges: Vec<_> = callers
                .iter()
                .map(|c| Edge::new(c.id, t.id, EdgeKind::RefCall))
                .collect();
            store
                .reindex_replace(
                    "c",
                    "a.rs",
                    &nodes,
                    &v1_edges,
                    "h1",
                    Some(&body(&v1_bodies)),
                )
                .unwrap();
            // Ratify: every caller->t edge becomes committed lsif truth.
            for c in &callers {
                store
                    .put_edge_lsif(&Edge::new(c.id, t.id, EdgeKind::RefCall))
                    .unwrap();
            }

            // Random edit: flip each caller's body byte with prob 1/2, and with
            // prob 1/3 also add a fifth function (interface edit).
            let mut v2_bodies = v1_bodies;
            let mut edited = [false; 4];
            for (i, edited_i) in edited.iter_mut().enumerate() {
                if next() % 2 == 0 {
                    v2_bodies[i] = 1;
                    *edited_i = true;
                }
            }
            let add_symbol = next() % 3 == 0;

            let mut v2_src = body(&v2_bodies);
            let mut v2_nodes = callers.clone();
            v2_nodes.push(t.clone());
            if add_symbol {
                let extra = mk("fn:extra", 15, 16);
                v2_src.push_str("fn extra() {\n}\n");
                v2_nodes.push(extra);
            }
            // The new parse re-emits each caller->t call (bodies still call
            // something), as tree-sitter edges.
            let v2_edges: Vec<_> = callers
                .iter()
                .map(|c| Edge::new(c.id, t.id, EdgeKind::RefCall))
                .collect();
            store
                .reindex_replace("c", "a.rs", &v2_nodes, &v2_edges, "h2", Some(&v2_src))
                .unwrap();

            for (i, c) in callers.iter().enumerate() {
                let prov = edge_provenance(&store, c.id, t.id);
                let preserved = !add_symbol && !edited[i];
                let expected = if preserved { "lsif" } else { "tree-sitter" };
                assert_eq!(
                    prov.as_deref(),
                    Some(expected),
                    "iter {iter}: f{i} preserved={preserved} add_symbol={add_symbol} \
                     edited={}: edge provenance wrong (a stale preserved edge is a breach)",
                    edited[i]
                );
            }
        }
    }

    /// Property (finding 1): a preserved caller never keeps a committed edge
    /// that resolved through a definition whose signature changed. Fuzzes
    /// whether the shared callee's return type or a parameter type is mutated
    /// and which callers are body-edited, and asserts every caller of the
    /// mutated callee is demoted (its committed receiver-typed edge purged)
    /// while an off-chain caller keeps its edge. The reference relationships
    /// come from the committed edges, never this parse's (Phase A emits no
    /// RefCall), so this is the production shape.
    #[test]
    fn reindex_replace_demotion_never_keeps_a_stale_typed_edge() {
        // Deterministic LCG so a failure reproduces from the seed in the message.
        let mut state: u64 = 0xD1B54A32D192ED03;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };

        for iter in 0..200u32 {
            let mut store = SqliteStore::open_in_memory().unwrap();
            let mk = |sig: &str, kind: &str, line: u32, end: u32| {
                travsr_core::Node::new(travsr_core::VName::new("c", "", "a.rs", "rust", sig), kind)
                    .with_line(line)
                    .with_end_line(end)
            };
            // make: 1-3 (shared callee), h0..h3: 4-15 (each calls make().run()),
            // ctrl: 16-18 (off-chain, calls free), then the committed targets.
            let make = mk("fn:make", "function", 1, 3);
            let hs: Vec<_> = (0..4)
                .map(|i| mk(&format!("fn:h{i}"), "function", 4 + i * 3, 6 + i * 3))
                .collect();
            let ctrl = mk("fn:ctrl", "function", 16, 18);
            let xrun = mk("method:X.run", "method", 19, 19);
            let free = mk("fn:free", "function", 20, 20);
            let mut nodes = vec![make.clone()];
            nodes.extend(hs.iter().cloned());
            nodes.push(ctrl.clone());
            nodes.push(xrun.clone());
            nodes.push(free.clone());

            let body = |ret: &str, param: &str, edited: &[bool; 4]| {
                let mut s = format!("fn make({param}) -> {ret} {{\n  d()\n}}\n");
                for (i, e) in edited.iter().enumerate() {
                    let call = if *e {
                        "make().run(); log()"
                    } else {
                        "make().run()"
                    };
                    s.push_str(&format!("fn h{i}() {{\n  {call};\n}}\n"));
                }
                s.push_str("fn ctrl() {\n  free();\n}\n");
                s.push_str("fn run() {}\n");
                s.push_str("fn free() {}\n");
                s
            };

            let no_edit = [false; 4];
            // The fresh parse carries no RefCall (Phase A emits only structural
            // edges), so the demotion input is the committed edges alone.
            let ts_edges: Vec<Edge> = Vec::new();
            store
                .reindex_replace(
                    "c",
                    "a.rs",
                    &nodes,
                    &ts_edges,
                    "h1",
                    Some(&body("A", "", &no_edit)),
                )
                .unwrap();
            // Ratify the committed reference edges Phase B would supply: each
            // caller calls make (the receiver source) and make().run() (the
            // stale-able receiver-typed edge); ctrl calls free.
            for h in &hs {
                store
                    .put_edge_lsif(&Edge::new(h.id, make.id, EdgeKind::RefCall))
                    .unwrap();
                store
                    .put_edge_lsif(&Edge::new(h.id, xrun.id, EdgeKind::RefCall))
                    .unwrap();
            }
            store
                .put_edge_lsif(&Edge::new(ctrl.id, free.id, EdgeKind::RefCall))
                .unwrap();

            // Random edit: mutate make's signature (return type, parameter type,
            // or neither) and independently body-edit each caller.
            let sig_kind = next() % 3; // 0 = none, 1 = return type, 2 = param type
            let (ret, param) = match sig_kind {
                1 => ("B", ""),
                2 => ("A", "x: i32"),
                _ => ("A", ""),
            };
            let make_mutated = sig_kind != 0;
            let mut edited = [false; 4];
            for e in edited.iter_mut() {
                *e = next() % 2 == 0;
            }

            store
                .reindex_replace(
                    "c",
                    "a.rs",
                    &nodes,
                    &ts_edges,
                    "h2",
                    Some(&body(ret, param, &edited)),
                )
                .unwrap();

            for (i, h) in hs.iter().enumerate() {
                // A caller keeps its committed edge iff its own body is unchanged
                // AND the callee it resolves through is unchanged.
                let preserved = !make_mutated && !edited[i];
                let expected = if preserved { Some("lsif") } else { None };
                assert_eq!(
                    edge_provenance(&store, h.id, xrun.id).as_deref(),
                    expected,
                    "iter {iter}: h{i} make_mutated={make_mutated} edited={}: \
                     a caller of a signature-changed callee kept a stale edge",
                    edited[i]
                );
            }
            // The off-chain caller is never demoted.
            assert_eq!(
                edge_provenance(&store, ctrl.id, free.id).as_deref(),
                Some("lsif"),
                "iter {iter}: an off-chain caller must keep its committed edge"
            );
        }
    }

    #[test]
    fn delete_file_purges_both_direction_edge_sites() {
        // #299 F7: deleting a file removes every occurrence into or out of it.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a_fn = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:a"),
            "function",
        );
        let b_fn = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "b.rs", "rust", "fn:b"),
            "function",
        );
        store.put_node(&a_fn).unwrap();
        store.put_node(&b_fn).unwrap();
        // a.rs references b.rs (dst in a-file-to-delete would be inbound; here we
        // also record b→a so both directions exist w.r.t. a.rs).
        store
            .record_edge_sites(&[(a_fn.id, b_fn.id, 2, None), (b_fn.id, a_fn.id, 7, None)])
            .unwrap();
        store.delete_file("c", "a.rs").unwrap();
        // Every edge_sites row touching a.rs is gone; b→a (inbound) also removed.
        assert!(store.reference_sites(a_fn.id).unwrap().is_empty());
        assert!(store.reference_sites(b_fn.id).unwrap().is_empty());
    }

    #[test]
    fn register_symbol_aliases_batch_roundtrip() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // symbol_aliases.node_id is a FK → nodes(id); nodes must exist first.
        let n1 = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a/b.go", "go", "method:X.m"),
            "method",
        );
        let n2 = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a/b.go", "go", "method:Y.n"),
            "method",
        );
        store.put_node(&n1).unwrap();
        store.put_node(&n2).unwrap();
        let aliases = vec![
            ("scip . a/b 1.0 X#m().".to_string(), n1.id),
            ("scip . a/b 1.0 Y#n().".to_string(), n2.id),
        ];
        store.register_symbol_aliases(&aliases).unwrap();
        assert_eq!(
            store.resolve_scip_symbol("scip . a/b 1.0 X#m().").unwrap(),
            Some(n1.id)
        );
        assert_eq!(
            store.resolve_scip_symbol("scip . a/b 1.0 Y#n().").unwrap(),
            Some(n2.id)
        );
        // Upsert: re-registering the same scip_symbol with a different node id
        // overwrites; use a distinct node so the FK is satisfied.
        let n3 = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a/b.go", "go", "method:Z.p"),
            "method",
        );
        store.put_node(&n3).unwrap();
        store
            .register_symbol_aliases(&[("scip . a/b 1.0 X#m().".to_string(), n3.id)])
            .unwrap();
        assert_eq!(
            store.resolve_scip_symbol("scip . a/b 1.0 X#m().").unwrap(),
            Some(n3.id)
        );
    }

    // ── RFC-027 ref_resolution_state (review findings 3, 5 and 7) ────────────

    /// A node in `corpus` at `path` with a signature and a span.
    fn live_node(corpus: &str, path: &str, sig: &str, kind: &str, line: u32) -> Node {
        let mut n = Node::new(VName::new(corpus, "", path, "typescript", sig), kind);
        n.line = Some(line);
        n.end_line = Some(line + 20);
        n
    }

    fn pending(src: NodeId, line: u32, name: &str) -> RefResolution {
        RefResolution {
            src,
            ref_line: line,
            ref_col: 0,
            name: name.to_string(),
            state: "pending",
            resolved_dst: None,
        }
    }

    /// Finding 3: the predicate must be `(src, line)`, not `src`.
    ///
    /// Three pending references in one function and Phase B resolves one of
    /// them. Keyed on `src` alone, any single outgoing edge on the enclosing
    /// definition cleared all three, so section 9.2's honest abstention silently
    /// under-reported the other two.
    #[test]
    fn clearing_pending_refs_is_line_wise_not_node_wide() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = live_node("c", "src/a.ts", "fn:caller", "function", 1);
        let callee = live_node("c", "src/b.ts", "fn:callee", "function", 1);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[
                    pending(caller.id, 3, "one"),
                    pending(caller.id, 4, "two"),
                    pending(caller.id, 5, "three"),
                ],
            )
            .unwrap();

        // Phase B resolved only the reference on line 4.
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 4, None)])
            .unwrap();

        assert_eq!(store.clear_resolved_pending_refs().unwrap(), 1);
        assert_eq!(
            store.pending_ref_count().unwrap(),
            2,
            "the two references Phase B said nothing about must stay pending"
        );
    }

    /// `resolved_ref_count` tallies the live overlay's settled references, the
    /// positive evidence the status surface needs to downgrade a `phase_b_dirty`
    /// edit from "stale" to "fresh". It counts only `resolved` rows, and (like
    /// `pending_ref_count`) only those whose `src` node still exists.
    #[test]
    fn resolved_ref_count_tallies_only_resolved_rows() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = live_node("c", "src/a.ts", "fn:caller", "function", 1);
        let callee = live_node("c", "src/b.ts", "fn:callee", "function", 1);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        let resolved = |line: u32, name: &str| RefResolution {
            src: caller.id,
            ref_line: line,
            ref_col: 0,
            name: name.to_string(),
            state: "resolved",
            resolved_dst: Some(callee.id),
        };
        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[
                    resolved(3, "one"),
                    resolved(4, "two"),
                    pending(caller.id, 5, "three"),
                ],
            )
            .unwrap();
        assert_eq!(store.resolved_ref_count().unwrap(), 2);
        assert_eq!(store.pending_ref_count().unwrap(), 1);
    }

    /// Finding 5: a renamed symbol retires its `NodeId`, which is the only
    /// handle `replace_ref_resolution_states` has, so its rows outlive every
    /// delete path. `pending_ref_count` must not count them, and the ratification
    /// purge must remove them.
    #[test]
    fn pending_rows_do_not_survive_the_rename_of_their_symbol() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let before = live_node("c", "src/a.ts", "fn:target", "function", 1);
        store.put_node(&before).unwrap();
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(before.id, 4, "target")])
            .unwrap();
        assert_eq!(store.pending_ref_count().unwrap(), 1);

        // The rename: the old node is gone, a new id takes its place, and the
        // file's rows are rewritten under the new id.
        store
            .reindex_replace("c", "src/a.ts", &[], &[], "h", None)
            .unwrap();
        let after = live_node("c", "src/a.ts", "fn:renamed", "function", 1);
        store.put_node(&after).unwrap();
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(after.id, 4, "renamed")])
            .unwrap();

        assert_eq!(
            store.pending_ref_count().unwrap(),
            1,
            "the count the MCP freshness note reports must not inflate on a rename"
        );
        assert_eq!(store.purge_orphan_ref_resolution_states().unwrap(), 1);
        assert_eq!(store.pending_ref_count().unwrap(), 1);
    }

    // ── #811 reconcile_ref_resolution_states ─────────────────────────────────
    //
    // The store half of #811. Every test below inspects the table directly with
    // the SQL the issue measured with, `SELECT COUNT(*) FROM ref_resolution_state
    // WHERE state = 'pending'`, rather than through `pending_ref_count`, whose
    // `JOIN nodes` would hide an orphan row and so pass over a table that still
    // held it.

    /// The issue's own measurement, verbatim.
    fn raw_pending_count(store: &SqliteStore) -> i64 {
        store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM ref_resolution_state WHERE state = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap()
    }

    fn raw_row_count(store: &SqliteStore) -> i64 {
        store
            .conn
            .query_row("SELECT COUNT(*) FROM ref_resolution_state", [], |r| {
                r.get(0)
            })
            .unwrap()
    }

    fn raw_edge_sites_count(store: &SqliteStore) -> i64 {
        store
            .conn
            .query_row("SELECT COUNT(*) FROM edge_sites", [], |r| r.get(0))
            .unwrap()
    }

    /// Whether a `(src, ref_line, name)` row is still present, in any state.
    fn has_row(store: &SqliteStore, src: NodeId, line: u32, name: &str) -> bool {
        store
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM ref_resolution_state \
                 WHERE src = ?1 AND ref_line = ?2 AND name = ?3)",
                params![node_id_to_i64(src), line as i64, name],
                |r| r.get::<_, i64>(0),
            )
            .unwrap()
            != 0
    }

    fn resolved(src: NodeId, line: u32, name: &str, dst: NodeId) -> RefResolution {
        RefResolution {
            src,
            ref_line: line,
            ref_col: 0,
            name: name.to_string(),
            state: "resolved",
            resolved_dst: Some(dst),
        }
    }

    /// A caller and a callee in two files, with a `ref/call` edge between them
    /// and nothing in `ref_resolution_state` yet.
    fn caller_and_callee(store: &mut SqliteStore) -> (Node, Node) {
        let caller = live_node("c", "src/a.ts", "fn:caller", "function", 1);
        let callee = live_node("c", "src/b.ts", "fn:callee", "function", 1);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store
            .put_edge(&Edge::new(caller.id, callee.id, EdgeKind::RefCall))
            .unwrap();
        (caller, callee)
    }

    /// Positive 1: a `pending` row whose `(src, line)` Phase B has since recorded
    /// a call site for is resolved by definition and must go.
    #[test]
    fn reconcile_removes_a_pending_row_phase_b_resolved() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(caller.id, 3, "callee")])
            .unwrap();
        assert_eq!(raw_pending_count(&store), 1, "precondition");

        // The rebuild's Phase B: the site now exists.
        store
            .record_edge_sites(&[(caller.id, callee.id, 3, None)])
            .unwrap();
        let edges_before = store.edge_count().unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(
            report,
            RefReconcileReport {
                cleared_resolved: 1,
                purged_orphans: 0,
            }
        );
        assert_eq!(raw_pending_count(&store), 0);
        assert!(!has_row(&store, caller.id, 3, "callee"));
        assert_eq!(
            raw_edge_sites_count(&store),
            1,
            "the evidence the reconcile keyed on must itself be untouched"
        );
        assert_eq!(
            store.edge_count().unwrap(),
            edges_before,
            "reconciling the state table must not touch the graph"
        );
    }

    /// Positive 2: the #811 shape is thousands of rows across many files. Every
    /// one with a site goes, in one pass, whatever file or node it belongs to.
    #[test]
    fn reconcile_removes_every_resolved_pending_row_across_files() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let callee = live_node("c", "src/lib.ts", "fn:target", "function", 1);
        store.put_node(&callee).unwrap();

        let mut callers = Vec::new();
        let mut sites = Vec::new();
        for f in 0..6 {
            let path = format!("src/f{f}.ts");
            let a = live_node("c", &path, &format!("fn:a{f}"), "function", 1);
            let b = live_node("c", &path, &format!("fn:b{f}"), "function", 40);
            store.put_node(&a).unwrap();
            store.put_node(&b).unwrap();
            store
                .replace_ref_resolution_states(
                    "c",
                    &path,
                    &[
                        pending(a.id, 3, "target"),
                        pending(a.id, 7, "target"),
                        pending(b.id, 42, "target"),
                    ],
                )
                .unwrap();
            sites.push((a.id, callee.id, 3, None));
            sites.push((a.id, callee.id, 7, None));
            sites.push((b.id, callee.id, 42, None));
            callers.push((a, b));
        }
        assert_eq!(raw_pending_count(&store), 18, "precondition");
        store.record_edge_sites(&sites).unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report.cleared_resolved, 18);
        assert_eq!(report.purged_orphans, 0);
        assert_eq!(raw_pending_count(&store), 0);
        assert_eq!(raw_row_count(&store), 0);
        for (a, b) in &callers {
            assert!(
                store.iter_edges_from(a.id).is_ok() && store.iter_edges_from(b.id).is_ok(),
                "graph reads must still work; the nodes are untouched"
            );
        }
    }

    /// Positive 3: a row whose `src` node no longer exists describes a reference
    /// that no longer exists. `replace_ref_resolution_states` cannot reach it
    /// (it resolves `src` through `nodes`), so the reconcile must.
    #[test]
    fn reconcile_purges_a_row_whose_source_node_is_gone() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let old = live_node("c", "src/a.ts", "fn:old", "function", 1);
        store.put_node(&old).unwrap();
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(old.id, 4, "x")])
            .unwrap();
        // The file is rewritten with the symbol gone: its id is retired.
        store
            .reindex_replace("c", "src/a.ts", &[], &[], "h", None)
            .unwrap();
        assert!(
            store.get_node(old.id).unwrap().is_none(),
            "precondition: the source node must be gone"
        );
        assert_eq!(
            raw_pending_count(&store),
            1,
            "precondition: the row outlived it"
        );

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(
            report,
            RefReconcileReport {
                cleared_resolved: 0,
                purged_orphans: 1,
            }
        );
        assert_eq!(raw_pending_count(&store), 0);
        assert_eq!(raw_row_count(&store), 0);
    }

    /// Positive 4: a realistic mixture. Only the rows the graph disproves go;
    /// the honest abstentions and the unrelated valid rows all stay, and the
    /// graph itself is untouched.
    #[test]
    fn reconcile_over_a_mixed_table_removes_only_what_the_graph_disproves() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        let other = live_node("c", "src/c.ts", "fn:other", "function", 1);
        store.put_node(&other).unwrap();
        // A symbol that will be renamed away, taking its id with it.
        let doomed = live_node("c", "src/d.ts", "fn:doomed", "function", 1);
        store.put_node(&doomed).unwrap();

        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[
                    // Resolved by the rebuild: a site lands on line 3.
                    pending(caller.id, 3, "callee"),
                    // Genuine abstention: nothing ever resolves line 9.
                    pending(caller.id, 9, "nothing"),
                    // Unrelated valid row: the live lane's own answer, which
                    // `clear` never touches (it is keyed on `pending`).
                    resolved(caller.id, 5, "callee", callee.id),
                ],
            )
            .unwrap();
        store
            .replace_ref_resolution_states("c", "src/c.ts", &[pending(other.id, 2, "ghost")])
            .unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "src/d.ts",
                &[
                    pending(doomed.id, 4, "a"),
                    resolved(doomed.id, 6, "b", callee.id),
                ],
            )
            .unwrap();
        // The rename: `src/d.ts` no longer defines `doomed`.
        store
            .reindex_replace("c", "src/d.ts", &[], &[], "h", None)
            .unwrap();
        // Phase B's evidence.
        store
            .record_edge_sites(&[
                (caller.id, callee.id, 3, None),
                (caller.id, callee.id, 5, None),
            ])
            .unwrap();

        let rows_before = raw_row_count(&store);
        let pending_before = raw_pending_count(&store);
        let sites_before = raw_edge_sites_count(&store);
        let edges_before = store.edge_count().unwrap();
        let nodes_before = store.node_count().unwrap();
        assert_eq!((rows_before, pending_before), (6, 4), "precondition");

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(
            report,
            RefReconcileReport {
                cleared_resolved: 1,
                purged_orphans: 2,
            },
            "one resolved pending row, and both rows of the renamed symbol"
        );
        // Gone.
        assert!(!has_row(&store, caller.id, 3, "callee"), "resolved pending");
        assert!(!has_row(&store, doomed.id, 4, "a"), "orphan pending");
        assert!(!has_row(&store, doomed.id, 6, "b"), "orphan resolved");
        // Kept.
        assert!(
            has_row(&store, caller.id, 9, "nothing"),
            "genuine abstention"
        );
        assert!(
            has_row(&store, other.id, 2, "ghost"),
            "genuine abstention, other file"
        );
        assert!(
            has_row(&store, caller.id, 5, "callee"),
            "valid resolved row"
        );
        assert_eq!(raw_pending_count(&store), 2);
        assert_eq!(raw_row_count(&store), rows_before - 3);
        // Before/after: the graph and its evidence are exactly as they were.
        assert_eq!(raw_edge_sites_count(&store), sites_before);
        assert_eq!(store.edge_count().unwrap(), edges_before);
        assert_eq!(store.node_count().unwrap(), nodes_before);
        // The two readers agree with the raw count once orphans are gone.
        assert_eq!(store.pending_ref_count().unwrap(), 2);
    }

    /// Negative 9: a reference nothing resolved has no site and a live node. It
    /// is the honest abstention RFC-027 exists to preserve, and must stay.
    #[test]
    fn reconcile_keeps_a_genuine_unresolved_reference() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, _callee) = caller_and_callee(&mut store);
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(caller.id, 3, "mystery")])
            .unwrap();
        assert_eq!(raw_edge_sites_count(&store), 0, "precondition: no evidence");

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report, RefReconcileReport::default());
        assert_eq!(raw_pending_count(&store), 1);
        assert!(has_row(&store, caller.id, 3, "mystery"));
        assert_eq!(
            store.pending_refs_in_file("c", "src/a.ts").unwrap(),
            vec![("mystery".to_string(), 3)],
            "the freshness note must still be able to report it"
        );
    }

    /// Negative 10: a site on the neighbouring line is not evidence for this
    /// one. `(src, line)` is the predicate, and an off-by-one must not match.
    #[test]
    fn reconcile_ignores_a_site_on_the_neighbouring_line() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(caller.id, 10, "callee")])
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 11, None)])
            .unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report, RefReconcileReport::default());
        assert!(has_row(&store, caller.id, 10, "callee"));
        assert_eq!(raw_pending_count(&store), 1);
    }

    /// Negative 11: the same line in a *different* enclosing definition is a
    /// different call site. A row for `A` must not be cleared by `B`'s site.
    #[test]
    fn reconcile_ignores_a_matching_line_on_a_different_source() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (a, callee) = caller_and_callee(&mut store);
        let b = live_node("c", "src/a.ts", "fn:b", "function", 30);
        store.put_node(&b).unwrap();
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(a.id, 10, "callee")])
            .unwrap();
        // B resolved something on its own line 10, which has nothing to do with A.
        store
            .record_edge_sites(&[(b.id, callee.id, 10, None)])
            .unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report, RefReconcileReport::default());
        assert!(has_row(&store, a.id, 10, "callee"));
        assert_eq!(raw_pending_count(&store), 1);
    }

    /// Negative 12: several sites on *other* lines of the same source are not
    /// evidence either. Keyed on `src` alone this row would go (the finding-3
    /// regression); keyed on `(src, line)` it stays.
    #[test]
    fn reconcile_ignores_sites_on_other_lines_of_the_same_source() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(caller.id, 10, "callee")])
            .unwrap();
        store
            .record_edge_sites(&[
                (caller.id, callee.id, 9, None),
                (caller.id, callee.id, 11, None),
                (caller.id, callee.id, 12, None),
            ])
            .unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report, RefReconcileReport::default());
        assert!(has_row(&store, caller.id, 10, "callee"));
        assert_eq!(raw_pending_count(&store), 1);
        assert_eq!(raw_edge_sites_count(&store), 3);
    }

    /// Negative 13: a table that is already consistent with the graph, holding
    /// both a genuine abstention and a valid resolved row, is left exactly as it
    /// is, and the pass reports that it did nothing.
    #[test]
    fn reconcile_on_a_consistent_table_changes_nothing() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[
                    pending(caller.id, 3, "nothing"),
                    resolved(caller.id, 5, "callee", callee.id),
                ],
            )
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 5, None)])
            .unwrap();
        let rows_before = raw_row_count(&store);

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report, RefReconcileReport::default());
        assert_eq!(report.total(), 0);
        assert_eq!(raw_row_count(&store), rows_before);
        assert_eq!(raw_pending_count(&store), 1);

        // And on a completely empty table.
        let mut empty = SqliteStore::open_in_memory().unwrap();
        assert_eq!(
            empty.reconcile_ref_resolution_states().unwrap(),
            RefReconcileReport::default()
        );
    }

    /// Negative 15: idempotent. The first pass does the work; a second pass over
    /// the same graph finds nothing, errors on nothing, and changes nothing.
    #[test]
    fn reconcile_is_idempotent() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        let doomed = live_node("c", "src/d.ts", "fn:doomed", "function", 1);
        store.put_node(&doomed).unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[
                    pending(caller.id, 3, "callee"),
                    pending(caller.id, 9, "nothing"),
                ],
            )
            .unwrap();
        store
            .replace_ref_resolution_states("c", "src/d.ts", &[pending(doomed.id, 4, "a")])
            .unwrap();
        store
            .reindex_replace("c", "src/d.ts", &[], &[], "h", None)
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 3, None)])
            .unwrap();

        let first = store.reconcile_ref_resolution_states().unwrap();
        assert_eq!(
            first,
            RefReconcileReport {
                cleared_resolved: 1,
                purged_orphans: 1,
            }
        );
        let rows_after_first = raw_row_count(&store);
        let pending_after_first = raw_pending_count(&store);
        assert_eq!(pending_after_first, 1);

        for _ in 0..3 {
            let again = store.reconcile_ref_resolution_states().unwrap();
            assert_eq!(
                again,
                RefReconcileReport::default(),
                "a repeat must be a no-op"
            );
            assert_eq!(raw_row_count(&store), rows_after_first);
            assert_eq!(raw_pending_count(&store), pending_after_first);
        }
        assert!(has_row(&store, caller.id, 9, "nothing"));
    }

    /// Negative 16: the two deletes are one transaction. Inject a failure into
    /// the second (a trigger that aborts the orphan purge) and the first must be
    /// rolled back with it: the table is either fully reconciled or untouched,
    /// never half-way.
    #[test]
    fn a_failed_reconcile_leaves_the_table_untouched() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let (caller, callee) = caller_and_callee(&mut store);
        let doomed = live_node("c", "src/d.ts", "fn:doomed", "function", 1);
        store.put_node(&doomed).unwrap();
        store
            .replace_ref_resolution_states("c", "src/a.ts", &[pending(caller.id, 3, "callee")])
            .unwrap();
        store
            .replace_ref_resolution_states("c", "src/d.ts", &[pending(doomed.id, 4, "a")])
            .unwrap();
        store
            .reindex_replace("c", "src/d.ts", &[], &[], "h", None)
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 3, None)])
            .unwrap();
        assert_eq!(raw_pending_count(&store), 2, "precondition");

        // The orphan purge is the second statement. Make it fail.
        store
            .conn
            .execute_batch(
                "CREATE TRIGGER injected_failure BEFORE DELETE ON ref_resolution_state \
                 WHEN NOT EXISTS (SELECT 1 FROM nodes WHERE id = OLD.src) \
                 BEGIN SELECT RAISE(ABORT, 'injected'); END;",
            )
            .unwrap();

        let err = store
            .reconcile_ref_resolution_states()
            .expect_err("the injected failure must surface");
        assert!(
            err.to_string().contains("injected"),
            "the error must be the injected one, got: {err}"
        );
        assert_eq!(
            raw_pending_count(&store),
            2,
            "the first delete (the resolved pending row) must have been rolled back"
        );
        assert!(has_row(&store, caller.id, 3, "callee"));
        assert!(has_row(&store, doomed.id, 4, "a"));

        // Remove the fault and the same pass completes in full.
        store
            .conn
            .execute_batch("DROP TRIGGER injected_failure;")
            .unwrap();
        assert_eq!(
            store.reconcile_ref_resolution_states().unwrap(),
            RefReconcileReport {
                cleared_resolved: 1,
                purged_orphans: 1,
            }
        );
        assert_eq!(raw_row_count(&store), 0);
    }

    /// Negative 17: a backlog the size #811 measured (thousands of rows) is
    /// reconciled in one pass to exactly the right remainder.
    #[test]
    fn reconcile_handles_a_large_backlog() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let callee = live_node("c", "src/lib.ts", "fn:target", "function", 1);
        store.put_node(&callee).unwrap();

        const FILES: u32 = 40;
        const LINES_PER_FILE: u32 = 100;
        let mut sites = Vec::new();
        for f in 0..FILES {
            let path = format!("src/f{f}.ts");
            let caller = live_node("c", &path, &format!("fn:caller{f}"), "function", 1);
            store.put_node(&caller).unwrap();
            let rows: Vec<RefResolution> = (1..=LINES_PER_FILE)
                .map(|line| pending(caller.id, line, "target"))
                .collect();
            store
                .replace_ref_resolution_states("c", &path, &rows)
                .unwrap();
            // Phase B resolves the even lines only.
            for line in (2..=LINES_PER_FILE).step_by(2) {
                sites.push((caller.id, callee.id, line, None));
            }
        }
        let total = (FILES * LINES_PER_FILE) as i64;
        assert_eq!(raw_pending_count(&store), total, "precondition");
        store.record_edge_sites(&sites).unwrap();

        let report = store.reconcile_ref_resolution_states().unwrap();

        assert_eq!(report.cleared_resolved as i64, total / 2);
        assert_eq!(report.purged_orphans, 0);
        assert_eq!(raw_pending_count(&store), total / 2);
        assert_eq!(store.pending_ref_count().unwrap() as i64, total / 2);
        assert_eq!(
            store.reconcile_ref_resolution_states().unwrap(),
            RefReconcileReport::default(),
            "and the remainder is stable"
        );
    }

    /// Finding 7: `_` is LIKE's single-character wildcard and identifiers are
    /// full of underscores, so a pattern built from `r.name` matched `do_thing`
    /// against a definition named `doXthing`. Spurious matches are not
    /// recall-neutral — they consume the `limit` cap under `ORDER BY path`.
    #[test]
    fn a_dependent_match_is_exact_not_a_like_pattern() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let dep = live_node("c", "dep.ts", "fn:consumer", "function", 1);
        // The only definition in def.ts differs from the pending name by one
        // character in the position `_` would wildcard.
        let decoy = live_node("c", "def.ts", "fn:doXthing", "function", 1);
        store.put_node(&dep).unwrap();
        store.put_node(&decoy).unwrap();
        store
            .replace_ref_resolution_states("c", "dep.ts", &[pending(dep.id, 4, "do_thing")])
            .unwrap();

        assert!(
            store
                .dependents_pending_on_file("c", "def.ts", "typescript", 32)
                .unwrap()
                .is_empty(),
            "an underscore in an identifier must not act as a LIKE wildcard"
        );

        // The genuine match still resolves, in both signature shapes.
        let unqualified = live_node("c", "def.ts", "fn:do_thing", "function", 5);
        store.put_node(&unqualified).unwrap();
        assert_eq!(
            store
                .dependents_pending_on_file("c", "def.ts", "typescript", 32)
                .unwrap(),
            vec!["dep.ts".to_string()],
            "`kind:Leaf` must match"
        );

        let mut qualified_only = SqliteStore::open_in_memory().unwrap();
        let dep2 = live_node("c", "dep.ts", "fn:consumer", "function", 1);
        let method = live_node("c", "def.ts", "method:Thing.do_thing", "method", 5);
        qualified_only.put_node(&dep2).unwrap();
        qualified_only.put_node(&method).unwrap();
        qualified_only
            .replace_ref_resolution_states("c", "dep.ts", &[pending(dep2.id, 4, "do_thing")])
            .unwrap();
        assert_eq!(
            qualified_only
                .dependents_pending_on_file("c", "def.ts", "typescript", 32)
                .unwrap(),
            vec!["dep.ts".to_string()],
            "`kind:Qual.Leaf` must match too"
        );
    }

    /// Finding 4: the meter is cumulative, so a claim it has scored has to be
    /// retired or every later commit counts it again.
    #[test]
    fn a_scored_claim_is_consumed_so_the_sample_size_stays_honest() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let caller = live_node("c", "src/a.ts", "fn:caller", "function", 1);
        let callee = live_node("c", "src/b.ts", "fn:callee", "function", 1);
        store.put_node(&caller).unwrap();
        store.put_node(&callee).unwrap();
        store
            .replace_ref_resolution_states(
                "c",
                "src/a.ts",
                &[RefResolution {
                    src: caller.id,
                    ref_line: 4,
                    ref_col: 0,
                    name: "callee".to_string(),
                    state: "resolved",
                    resolved_dst: Some(callee.id),
                }],
            )
            .unwrap();
        store
            .record_edge_sites(&[(caller.id, callee.id, 4, None)])
            .unwrap();

        assert_eq!(store.live_precision_sample().unwrap().agree, 1);
        // A language that did not complete must not consume the claim: the row
        // survives to be scored at its own ratification (issue A).
        assert_eq!(
            store
                .consume_measured_ref_resolutions(&["rust".to_string()])
                .unwrap(),
            0,
            "a crashed language's claim must not be retired by another's ratification"
        );
        assert_eq!(store.live_precision_sample().unwrap().claims(), 1);
        assert_eq!(
            store
                .consume_measured_ref_resolutions(&["typescript".to_string()])
                .unwrap(),
            1
        );
        assert_eq!(
            store.live_precision_sample().unwrap().claims(),
            0,
            "a second ratification must not re-score a claim already counted"
        );
    }

    fn node_with_path(path: &str, sig: &str) -> Node {
        Node::new(VName::new("", "", path, "typescript", sig), "function")
    }

    // Critical: an existing v1 database (no provenance column) must gain the
    // column with DEFAULT 'tree-sitter' when opened by the v2 code, and all
    // pre-existing edges must read back with provenance='tree-sitter'.
    #[test]
    fn v1_to_v2_migration_adds_provenance_with_default() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("v1.db");

        // Simulate a v1 DB: create schema manually without the provenance column.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO meta VALUES('schema_version', '1');
                 CREATE TABLE nodes (
                   id INTEGER PRIMARY KEY, corpus TEXT NOT NULL, root TEXT NOT NULL,
                   path TEXT NOT NULL, language TEXT NOT NULL,
                   signature TEXT NOT NULL, kind TEXT NOT NULL
                 );
                 CREATE TABLE edges (
                   src INTEGER NOT NULL, dst INTEGER NOT NULL, kind TEXT NOT NULL,
                   PRIMARY KEY (src, dst, kind)
                 );
                 CREATE TABLE files (
                   path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
                   last_indexed_at INTEGER NOT NULL
                 );
                 INSERT INTO nodes VALUES(1,'','','a.ts','typescript','fn:a','function');
                 INSERT INTO nodes VALUES(2,'','','b.ts','typescript','fn:b','function');
                 INSERT INTO edges VALUES(1, 2, 'ref-call');",
            )
            .unwrap();
        }

        // Open with the v2 store — migration must succeed.
        let store = SqliteStore::open(&db_path).expect("v1→v2 migration must succeed");

        // Pre-existing edge must have provenance='tree-sitter' (the column DEFAULT).
        let edges = store.all_edges().unwrap();
        assert_eq!(edges.len(), 1, "pre-existing edge must survive migration");
        assert_eq!(
            edges[0].3, "tree-sitter",
            "migrated edge must default to tree-sitter provenance"
        );
    }

    #[test]
    fn file_hash_round_trips() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.put_file_hash("src/foo.ts", "abc123").unwrap();
        let got = store.get_file_hash("src/foo.ts").unwrap();
        assert_eq!(got, Some("abc123".to_string()));
        assert!(store.get_file_hash("nonexistent.ts").unwrap().is_none());
    }

    #[test]
    fn clear_file_hashes_empties_the_cache() {
        // #757 audit: `--force` clears the per-file hash cache so every file
        // re-parses. After clearing, no stored hash remains.
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.put_file_hash("a.rs", "h1").unwrap();
        store.put_file_hash("b.rs", "h2").unwrap();
        assert_eq!(store.get_all_file_hashes().unwrap().len(), 2);
        let cleared = store.clear_file_hashes().unwrap();
        assert_eq!(cleared, 2);
        assert!(store.get_all_file_hashes().unwrap().is_empty());
        assert!(store.get_file_hash("a.rs").unwrap().is_none());
    }

    #[test]
    fn delete_nodes_for_path_removes_nodes_and_edges() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:foo");
        let b = node_with_path("a.ts", "fn:bar");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::DefinesBinding))
            .unwrap();
        assert_eq!(store.node_count().unwrap(), 2);
        assert_eq!(store.edge_count().unwrap(), 1);

        let deleted = store.delete_nodes_for_path("a.ts").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.edge_count().unwrap(), 0);
    }

    #[test]
    fn delete_nodes_for_path_leaves_other_paths() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:foo");
        let b = node_with_path("b.ts", "fn:bar");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::Depends))
            .unwrap();

        store.delete_nodes_for_path("a.ts").unwrap();

        assert_eq!(store.node_count().unwrap(), 1, "b.ts node must survive");
        assert_eq!(
            store.edge_count().unwrap(),
            0,
            "edge referencing a.ts must be removed"
        );
        assert!(store.get_node(b.id).unwrap().is_some());
    }

    #[test]
    fn delete_nodes_for_path_prefix_removes_matching_nodes_and_edges() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path(".claude/worktrees/session-1/agent.ts", "fn:agent");
        let b = node_with_path(".claude/worktrees/session-2/helper.ts", "fn:helper");
        let c = node_with_path("src/real.ts", "fn:real");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&c).unwrap();
        store
            .put_edge(&Edge::new(a.id, c.id, EdgeKind::Depends))
            .unwrap();

        let deleted = store.delete_nodes_for_path_prefix(".claude/").unwrap();
        assert_eq!(deleted, 2, "two .claude/ nodes must be removed");
        assert_eq!(
            store.node_count().unwrap(),
            1,
            "src/real.ts node must survive"
        );
        assert_eq!(
            store.edge_count().unwrap(),
            0,
            "edge referencing .claude/ node must be removed"
        );
        assert!(
            store.get_node(c.id).unwrap().is_some(),
            "real node must be intact"
        );
    }

    #[test]
    fn delete_nodes_for_path_prefix_returns_zero_for_no_match() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = node_with_path("src/real.ts", "fn:real");
        store.put_node(&n).unwrap();
        let deleted = store.delete_nodes_for_path_prefix(".claude/").unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.node_count().unwrap(), 1);
    }

    #[test]
    fn search_nodes_by_name_finds_partial_match() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = node_with_path("svc.ts", "fn:charge");
        store.put_node(&n).unwrap();
        let results = store.search_nodes_by_name("charge").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vname.signature, "fn:charge");
    }

    #[test]
    fn search_ranks_exact_signature_first() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&node_with_path("pkg/eviction/handler.go", "fn:eviction"))
            .unwrap();
        store
            .put_node(&node_with_path(
                "pkg/eviction/mock_test.go",
                "fn:eviction_test",
            ))
            .unwrap();
        let results = store.search_nodes_by_name("eviction").unwrap();
        // exact signature match must rank first
        assert_eq!(results[0].vname.signature, "fn:eviction");
    }

    #[test]
    fn search_ranks_production_over_test_path() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Both signatures are identical substrings so match-quality tier is equal;
        // only the test-path penalty (+25) should differentiate them.
        store
            .put_node(&node_with_path("pkg/eviction/mock_test.go", "fn:handleX"))
            .unwrap();
        store
            .put_node(&node_with_path("pkg/eviction/handler.go", "fn:handleY"))
            .unwrap();
        let results = store.search_nodes_by_name("handle").unwrap();
        // production file must rank ahead of test file
        assert_eq!(results[0].vname.path, "pkg/eviction/handler.go");
    }

    #[test]
    fn search_ranks_vendor_last() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&node_with_path("vendor/lib/util.go", "fn:util"))
            .unwrap();
        store
            .put_node(&node_with_path("src/util.go", "fn:util2"))
            .unwrap();
        let results = store.search_nodes_by_name("util").unwrap();
        // vendor path must rank below production
        assert_eq!(results[0].vname.path, "src/util.go");
    }

    #[test]
    fn search_ranks_generated_and_mock_below_production() {
        // Issue #297: generated (`zz_generated`, `.pb.go`) and mock/fake files
        // must not outrank the real decl. Both signatures are substring-equal
        // matches (same match tier) so only the generated/mock penalty (+20)
        // differentiates them.
        for gen_path in [
            "pkg/api/zz_generated.deepcopy.go",
            "pkg/api/types.pb.go",
            "pkg/api/mock_handler.go",
            "pkg/api/fake_handler.go",
        ] {
            let mut store = SqliteStore::open_in_memory().unwrap();
            store
                .put_node(&node_with_path(gen_path, "fn:handleGen"))
                .unwrap();
            store
                .put_node(&node_with_path("pkg/api/handler.go", "fn:handleProd"))
                .unwrap();
            let results = store.search_nodes_by_name("handle").unwrap();
            assert_eq!(
                results[0].vname.path, "pkg/api/handler.go",
                "production file must outrank generated/mock file {gen_path}"
            );
        }
    }

    #[test]
    fn search_limit_caps_results() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        for i in 0..150u32 {
            store
                .put_node(&node_with_path(
                    &format!("src/file{i}.go"),
                    &format!("fn:target_{i}"),
                ))
                .unwrap();
        }
        let results = store.search_nodes_by_name("target").unwrap();
        assert!(results.len() <= 100);
    }

    #[test]
    fn search_rank_deterministic_tiebreak() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Two nodes with identical rank (same path pattern, same sig pattern)
        store
            .put_node(&node_with_path("src/a.go", "fn:foo"))
            .unwrap();
        store
            .put_node(&node_with_path("src/b.go", "fn:foo"))
            .unwrap();
        let r1 = store.search_nodes_by_name("foo").unwrap();
        let r2 = store.search_nodes_by_name("foo").unwrap();
        // Order must be deterministic across two calls
        assert_eq!(r1[0].vname.path, r2[0].vname.path);
    }

    /// #478 RFC-023 §6 item 7, Evidence A: a `file` node scoring far below a
    /// clean top-5 run of `function` nodes must stay at the tail, not get
    /// promoted to rank 2 by an unconditional kind round-robin. This is the
    /// exact fixture shape RFC-023 §1's dump showed (`var:m`/`var:fn` in the
    /// top 8 despite scoring far below the genuine top-5 methods).
    #[test]
    fn diversify_topk_does_not_perturb_clear_top5_same_kind_run() {
        let f1 = sample_node("fn:top_1");
        let f2 = sample_node("fn:top_2");
        let f3 = sample_node("fn:top_3");
        let f4 = sample_node("fn:top_4");
        let f5 = sample_node("fn:top_5");
        let file = Node::new(
            VName::new(
                "test-corpus",
                "main",
                "docs/readme.md",
                "markdown",
                "docs/readme.md",
            ),
            "file",
        );
        // RRF scores mirror a real fused list: a decisive same-kind top-5 run
        // (each within TRAVSR_DIVERSIFY_BAND_EPSILON's default 5% of the one
        // above), then a minority-kind node scoring far below any of them.
        let fused = vec![
            (f1.clone(), 0.069, 1.0),
            (f2.clone(), 0.0678, 0.9),
            (f3.clone(), 0.0667, 0.8),
            (f4.clone(), 0.0657, 0.7),
            (f5.clone(), 0.0648, 0.6),
            (file.clone(), 0.010, 0.3),
        ];
        let out = SqliteStore::diversify_topk(fused);
        let sigs: Vec<&str> = out
            .iter()
            .map(|(n, _)| n.vname.signature.as_str())
            .collect();
        assert_eq!(
            sigs,
            vec![
                "fn:top_1",
                "fn:top_2",
                "fn:top_3",
                "fn:top_4",
                "fn:top_5",
                "docs/readme.md",
            ],
            "a low-scoring minority kind must not be promoted ahead of a clear same-kind top-5 run"
        );
    }

    /// #478 RFC-023 §6 item 7 / #393: when scores genuinely are close (within
    /// `TRAVSR_DIVERSIFY_BAND_EPSILON`), kind diversity still applies as a
    /// tie-break — a competitive minority kind must not be starved out of the
    /// visible top-K just because it shares a band with a dominant kind.
    #[test]
    fn diversify_topk_interleaves_within_a_close_score_band() {
        let f1 = sample_node("fn:a");
        let f2 = sample_node("fn:b");
        let file = Node::new(
            VName::new(
                "test-corpus",
                "main",
                "docs/readme.md",
                "markdown",
                "docs/readme.md",
            ),
            "file",
        );
        // All three within 5% of the band leader (0.069 * 0.95 = 0.06555).
        let fused = vec![
            (f1.clone(), 0.069, 1.0),
            (f2.clone(), 0.067, 0.9),
            (file.clone(), 0.066, 0.8),
        ];
        let out = SqliteStore::diversify_topk(fused);
        let sigs: Vec<&str> = out
            .iter()
            .map(|(n, _)| n.vname.signature.as_str())
            .collect();
        assert_eq!(
            sigs,
            vec!["fn:a", "docs/readme.md", "fn:b"],
            "within a close score band, kind round-robin must still promote a competitive minority kind"
        );
    }

    /// #478 RFC-023 §6.1: `explain_leg_scores` must show the leg separation
    /// `travsr explain` exists to surface — the same distinction Evidence A
    /// documents. "wal" is a substring of both `walk` and `walker`, so the
    /// trigram leg matches both; neither node has "wal" as its own word
    /// segment, so the word leg matches neither.
    #[test]
    fn explain_leg_scores_separates_word_from_trigram() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&node_with_path("src/walker.rs", "fn:walk"))
            .unwrap();
        store
            .put_node(&node_with_path("src/walker.rs", "fn:walker_helper"))
            .unwrap();
        let legs = store.explain_leg_scores("wal", None).unwrap();
        assert!(
            legs.trigram.len() >= 2,
            "trigram (substring) leg must match both walk and walker_helper"
        );
        assert!(
            legs.word.is_empty(),
            "word leg must not match, \"wal\" is not a word segment of either node"
        );
    }

    /// #478 RFC-023 §11 AC #3 / Evidence B: `bm25(nodes_fts_words, W_SIG,
    /// W_PATH)` weights the signature column higher than the path column
    /// (3.0 vs 1.0), so a node matching the term in its signature must
    /// outrank one matching only in its path.
    #[test]
    fn fts_query_words_scored_signature_match_outranks_path_match() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store
            .put_node(&node_with_path("src/a.rs", "fn:widget_factory"))
            .unwrap();
        store
            .put_node(&node_with_path("src/widget/other.rs", "fn:make_thing"))
            .unwrap();

        let results = store.fts_query_words_scored("\"widget\"").unwrap();
        let rank_of = |sig: &str| results.iter().position(|(n, _)| n.vname.signature == sig);
        let sig_rank = rank_of("fn:widget_factory").expect("signature match must appear");
        let path_rank = rank_of("fn:make_thing").expect("path match must appear");
        assert!(
            sig_rank < path_rank,
            "a signature match must outrank a path match for the same term \
             (W_SIG=3.0 > W_PATH=1.0); results: {:?}",
            results
                .iter()
                .map(|(n, s)| (n.vname.signature.clone(), *s))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn meta_get_set_round_trips() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        assert!(store.get_meta("foo").unwrap().is_none());
        store.set_meta("foo", "bar").unwrap();
        assert_eq!(store.get_meta("foo").unwrap(), Some("bar".to_string()));
        store.set_meta("foo", "baz").unwrap();
        assert_eq!(store.get_meta("foo").unwrap(), Some("baz".to_string()));
    }

    #[test]
    fn lsif_provenance_wins_over_tree_sitter() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let edge = Edge::new(a.id, b.id, EdgeKind::RefCall);

        // Tree-sitter inserts first
        store.put_edge(&edge).unwrap();
        // LSIF inserts same edge — must win
        store.put_edge_lsif(&edge).unwrap();

        assert_eq!(store.edge_count().unwrap(), 1, "must be exactly one row");
        let edges = store.all_edges().unwrap();
        assert_eq!(edges[0].3, "lsif", "provenance must be upgraded to lsif");
    }

    #[test]
    fn tree_sitter_cannot_demote_lsif_provenance() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let edge = Edge::new(a.id, b.id, EdgeKind::RefCall);

        // LSIF inserts first
        store.put_edge_lsif(&edge).unwrap();
        // Tree-sitter tries to insert same edge — must NOT demote
        store.put_edge(&edge).unwrap();

        let edges = store.all_edges().unwrap();
        assert_eq!(
            edges[0].3, "lsif",
            "tree-sitter must not demote lsif provenance"
        );
    }

    #[test]
    fn tree_sitter_only_edge_has_tree_sitter_provenance() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::Depends))
            .unwrap();

        let edges = store.all_edges().unwrap();
        assert_eq!(edges[0].3, "tree-sitter");
    }

    #[test]
    fn e1_reconcile_edge_languages_derives_from_endpoints() {
        // E1: an edge's language is derived from its src node, not the schema
        // default 'typescript'. write_phase_b_batch labels provenance truthfully.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "fn:a"),
            "function",
        );
        let b = travsr_core::Node::new(
            travsr_core::VName::new("c", "", "a.rs", "rust", "method:T.b"),
            "method",
        );
        let edge = Edge::new(a.id, b.id, EdgeKind::RefCall);
        store
            .write_phase_b_batch(&[a.clone(), b.clone()], &[edge], "tree-sitter")
            .unwrap();

        // Before reconcile: schema default mislabels the edge 'typescript'.
        let lang_before: String = store
            .conn
            .query_row("SELECT language FROM edges LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(lang_before, "typescript", "reproduces the mislabel default");

        let n = store.reconcile_edge_languages().unwrap();
        assert_eq!(n, 1);
        let (lang, prov): (String, String) = store
            .conn
            .query_row("SELECT language, provenance FROM edges LIMIT 1", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(lang, "rust", "edge language tracks its src node");
        assert_eq!(prov, "tree-sitter", "native resolved edges are not 'lsif'");
    }

    #[test]
    fn e1_write_phase_b_batch_does_not_demote_compiler_provenance() {
        // E1 precedence: a 'tree-sitter' batch write must not overwrite an
        // existing compiler ('lsif'/'scip') provenance (ADR-002).
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let edge = Edge::new(a.id, b.id, EdgeKind::RefCall);
        store.put_edge_lsif(&edge).unwrap();
        store
            .write_phase_b_batch(&[], &[edge], "tree-sitter")
            .unwrap();
        let edges = store.all_edges().unwrap();
        assert_eq!(edges[0].3, "lsif", "tree-sitter batch must not demote lsif");
    }

    #[test]
    fn write_phase_b_batch_drops_half_edges_and_keeps_resolvable_ones() {
        // #712 half-edge guard + PR #715 batched-endpoint prefetch. An edge whose
        // endpoint the sidecar never emitted (nor exists in the store) must be
        // dropped; edges resolvable within the batch OR against a pre-existing
        // store node must survive. The pre-existing node exercises the prefetch
        // path (an endpoint not present in this batch).
        let mut store = SqliteStore::open_in_memory().unwrap();
        let existing = sample_node("fn:existing");
        store.put_node(&existing).unwrap();

        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        let ghost = sample_node("fn:ghost"); // referenced but never emitted

        let good_intra = Edge::new(a.id, b.id, EdgeKind::RefCall); // both in batch
        let good_to_existing = Edge::new(a.id, existing.id, EdgeKind::RefCall); // dst in store
        let half = Edge::new(a.id, ghost.id, EdgeKind::RefCall); // dst nowhere

        store
            .write_phase_b_batch(
                &[a.clone(), b.clone()],
                &[good_intra, good_to_existing, half],
                "scip",
            )
            .unwrap();

        let edges = store.all_edges().unwrap();
        assert_eq!(
            edges.len(),
            2,
            "the half-edge to a missing endpoint must be dropped: {edges:?}"
        );
        let dsts: std::collections::HashSet<NodeId> = edges.iter().map(|e| e.1).collect();
        assert!(dsts.contains(&b.id), "intra-batch edge kept");
        assert!(
            dsts.contains(&existing.id),
            "edge to pre-existing store node kept"
        );
        assert!(!dsts.contains(&ghost.id), "ghost-endpoint edge dropped");
    }

    #[test]
    fn lsif_only_edge_has_lsif_provenance() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store
            .put_edge_lsif(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();

        let edges = store.all_edges().unwrap();
        assert_eq!(edges[0].3, "lsif");
    }

    #[test]
    fn iter_edges_from_kind_filters_correctly() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        let c = sample_node("fn:c");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&c).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(a.id, c.id, EdgeKind::Depends))
            .unwrap();

        // Only RefCall edges
        let ref_call = store.iter_edges_from_kind(a.id, EdgeKind::RefCall).unwrap();
        assert_eq!(ref_call.len(), 1);
        assert_eq!(ref_call[0].dst, b.id);
        assert_eq!(ref_call[0].kind, EdgeKind::RefCall);

        // Only Depends edges
        let depends = store.iter_edges_from_kind(a.id, EdgeKind::Depends).unwrap();
        assert_eq!(depends.len(), 1);
        assert_eq!(depends[0].dst, c.id);

        // Kind with no edges returns empty
        let exports = store.iter_edges_from_kind(a.id, EdgeKind::Exports).unwrap();
        assert!(exports.is_empty());
    }

    #[test]
    fn iter_edges_from_kind_consistent_with_iter_edges_from() {
        // The kind-filtered result must be a subset of the full result for the
        // same src, and every returned edge must have the requested kind.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = sample_node("fn:a");
        let b = sample_node("fn:b");
        let c = sample_node("fn:c");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&c).unwrap();
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(a.id, c.id, EdgeKind::DefinesBinding))
            .unwrap();

        let all = store.iter_edges_from(a.id).unwrap();
        let filtered = store.iter_edges_from_kind(a.id, EdgeKind::RefCall).unwrap();
        assert!(filtered.len() <= all.len());
        assert!(filtered.iter().all(|e| e.kind == EdgeKind::RefCall));
    }

    #[test]
    fn iter_edges_to_returns_incoming_edges() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:a");
        let b = node_with_path("b.ts", "fn:b");
        let c = node_with_path("c.ts", "fn:c");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        store.put_node(&c).unwrap();
        store
            .put_edge(&Edge::new(a.id, c.id, EdgeKind::DefinesBinding))
            .unwrap();
        store
            .put_edge(&Edge::new(b.id, c.id, EdgeKind::DefinesBinding))
            .unwrap();

        let edges = store.iter_edges_to(c.id).unwrap();
        let srcs: Vec<NodeId> = edges.iter().map(|e| e.src).collect();
        assert_eq!(edges.len(), 2, "both A→C and B→C must be returned");
        assert!(srcs.contains(&a.id), "A must appear as a caller");
        assert!(srcs.contains(&b.id), "B must appear as a caller");
        assert!(
            edges.iter().all(|e| e.dst == c.id),
            "dst must be C for all edges"
        );
    }

    /// DEBT-75: every `iter_edges_*` reader must carry the row's true
    /// `edges.provenance`, so a consumer that traverses the graph (the MCP
    /// surface, `travsr graph --format json`) reports how an edge was actually
    /// derived instead of assuming `tree-sitter`. Before this, an `lsif` edge
    /// read back through BFS was indistinguishable from a heuristic one.
    #[test]
    fn edge_readers_carry_true_provenance() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:a");
        let b = node_with_path("b.ts", "fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();

        // One tree-sitter edge and one lsif edge out of the same source.
        let ts_edge = Edge::new(a.id, b.id, EdgeKind::DefinesBinding);
        let lsif_edge = Edge::new(a.id, b.id, EdgeKind::RefCall);
        store.put_edge(&ts_edge).unwrap();
        store.put_edge_lsif(&lsif_edge).unwrap();

        let prov = |edges: &[Edge], kind: EdgeKind| -> String {
            edges
                .iter()
                .find(|e| e.kind == kind)
                .and_then(|e| e.provenance.clone())
                .unwrap_or_else(|| panic!("no {kind:?} edge returned"))
        };

        let from = store.iter_edges_from(a.id).unwrap();
        assert_eq!(prov(&from, EdgeKind::RefCall), "lsif");
        assert_eq!(prov(&from, EdgeKind::DefinesBinding), "tree-sitter");

        let to = store.iter_edges_to(b.id).unwrap();
        assert_eq!(prov(&to, EdgeKind::RefCall), "lsif");
        assert_eq!(prov(&to, EdgeKind::DefinesBinding), "tree-sitter");

        let batch = store.iter_edges_from_batch(&[a.id]).unwrap();
        assert_eq!(prov(&batch, EdgeKind::RefCall), "lsif");
        assert_eq!(prov(&batch, EdgeKind::DefinesBinding), "tree-sitter");

        let by_kind = store.iter_edges_from_kind(a.id, EdgeKind::RefCall).unwrap();
        assert_eq!(prov(&by_kind, EdgeKind::RefCall), "lsif");
    }

    /// RFC-027: the live overlay is purely additive.
    ///
    /// It creates edges that were absent and refreshes its own, but never
    /// relabels a row another lane wrote. That is the property that makes the
    /// ratification sweep safe: everything the sweep can delete is something
    /// the live lane created, so retiring the overlay returns the graph to what
    /// it was instead of destroying pre-existing truth.
    #[test]
    fn the_live_overlay_is_additive_and_never_relabels_another_lane() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:a");
        let b = node_with_path("b.ts", "fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let e = Edge::new(a.id, b.id, EdgeKind::RefCall);

        let prov = |st: &SqliteStore| -> String {
            st.iter_edges_from(a.id).unwrap()[0]
                .provenance
                .clone()
                .expect("reader must carry provenance")
        };

        // Absent -> live: the overlay's actual job.
        store.put_edge_live(&e).unwrap();
        assert_eq!(prov(&store), "live", "a missing edge is created as live");

        // live -> live: idempotent. reindex_replace wipes a file's outbound
        // edges on every save, so the engine re-emits whole-file (plan R5).
        store.put_edge_live(&e).unwrap();
        assert_eq!(prov(&store), "live");

        // live -> lsif: ratification wins.
        store.put_edge_lsif(&e).unwrap();
        assert_eq!(prov(&store), "lsif", "lsif must overwrite live");

        // lsif -> live: must NOT demote ratified truth.
        store.put_edge_live(&e).unwrap();
        assert_eq!(
            prov(&store),
            "lsif",
            "a live write must never demote lsif/scip"
        );

        // tree-sitter -> live: must NOT relabel. This is the one the sweep
        // depends on. An interface edit re-resolves files that were never
        // re-parsed and still hold their tree-sitter edges; relabelling those
        // would hand pre-existing truth to the sweep to delete.
        let e2 = Edge::new(a.id, b.id, EdgeKind::DefinesBinding);
        store.put_edge(&e2).unwrap();
        store.put_edge_live(&e2).unwrap();
        let ts = store
            .iter_edges_from(a.id)
            .unwrap()
            .into_iter()
            .find(|x| x.kind == EdgeKind::DefinesBinding)
            .and_then(|x| x.provenance)
            .unwrap();
        assert_eq!(
            ts, "tree-sitter",
            "a live write must leave an existing tree-sitter row untouched"
        );
    }

    /// RFC-027: a Phase B write over a `live` row ratifies it.
    ///
    /// Both callers that pass a 'tree-sitter' provenance are Phase B runs
    /// (`init_repo_with_progress`, `run_background_phase_b_inner`), so such a
    /// write reaching an existing `live` row means Phase B just re-derived that
    /// edge by native leaf-name resolution. Relabelling it is correct: it is no
    /// longer a live guess. This is what leaves the section 8.3 sweep holding
    /// only the live edges Phase B did *not* re-derive, so deleting those
    /// cannot lose a real edge.
    #[test]
    fn write_phase_b_batch_ratifies_a_live_edge() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:a");
        let b = node_with_path("b.ts", "fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let e = Edge::new(a.id, b.id, EdgeKind::RefCall);
        store.put_edge_live(&e).unwrap();

        store
            .write_phase_b_batch(&[], std::slice::from_ref(&e), "tree-sitter")
            .unwrap();
        assert_eq!(
            store.iter_edges_from(a.id).unwrap()[0]
                .provenance
                .as_deref(),
            Some("tree-sitter"),
            "Phase B re-derived this edge natively, so it is ratified, not live"
        );

        // And a SCIP-provenance batch ratifies it as compiler truth.
        store
            .write_phase_b_batch(&[], std::slice::from_ref(&e), "scip")
            .unwrap();
        assert_eq!(
            store.iter_edges_from(a.id).unwrap()[0]
                .provenance
                .as_deref(),
            Some("scip"),
            "scip must overwrite live"
        );
    }

    /// Provenance is metadata about how an edge was derived, not part of its
    /// identity: the store's primary key is `(src, dst, kind)`. A read-back edge
    /// must therefore still compare equal to the constructed one it came from.
    #[test]
    fn provenance_is_not_part_of_edge_identity() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let a = node_with_path("a.ts", "fn:a");
        let b = node_with_path("b.ts", "fn:b");
        store.put_node(&a).unwrap();
        store.put_node(&b).unwrap();
        let built = Edge::new(a.id, b.id, EdgeKind::RefCall);
        store.put_edge_lsif(&built).unwrap();

        let read_back = store.iter_edges_from(a.id).unwrap();
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].provenance.as_deref(), Some("lsif"));
        assert_eq!(
            read_back[0], built,
            "an edge differing only in provenance is the same edge"
        );
    }

    // V4 (no package column) → V5 migration must add package with empty default,
    // add language column to edges, and existing nodes must read back correctly.
    #[test]
    fn v4_to_v5_migration_adds_package_column() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("v4.db");

        // Simulate a v4 DB: schema version 4, nodes table without package column.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO meta VALUES('schema_version', '4');
                 CREATE TABLE nodes (
                   id INTEGER PRIMARY KEY,
                   corpus TEXT NOT NULL, root TEXT NOT NULL, path TEXT NOT NULL,
                   language TEXT NOT NULL, signature TEXT NOT NULL, kind TEXT NOT NULL
                 );
                 CREATE TABLE edges (
                   src INTEGER NOT NULL, dst INTEGER NOT NULL, kind TEXT NOT NULL,
                   provenance TEXT NOT NULL DEFAULT 'tree-sitter',
                   PRIMARY KEY (src, dst, kind)
                 );
                 CREATE TABLE files (
                   path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
                   last_indexed_at INTEGER NOT NULL
                 );
                 INSERT INTO nodes VALUES(10,'','','src/main.rs','rust','fn:main','function');",
            )
            .unwrap();
        }

        // Open with the v5 store — migration must add package + edges.language.
        let store = SqliteStore::open(&db_path).expect("v4→v5 migration must succeed");

        // Pre-existing node must read back with package = '' (the column default).
        let nodes = store.all_nodes().unwrap();
        assert_eq!(nodes.len(), 1, "pre-existing node must survive migration");
        assert_eq!(
            nodes[0].package, "",
            "migrated node must default to empty package"
        );

        // Column must exist and be queryable.
        let has_pkg = store.column_exists("nodes", "package").unwrap();
        assert!(
            has_pkg,
            "nodes.package column must exist after v5 migration"
        );
        let has_lang = store.column_exists("edges", "language").unwrap();
        assert!(
            has_lang,
            "edges.language column must exist after v5 migration"
        );
    }

    // package field round-trips through put_node / get_node.
    #[test]
    fn node_package_round_trips() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new(
                "github.com/a/b",
                "",
                "crates/foo/src/lib.rs",
                "rust",
                "fn:open",
            ),
            "function",
        )
        .with_package("foo-crate");
        let id = store.put_node(&n).unwrap();
        let back = store.get_node(id).unwrap().expect("node must exist");
        assert_eq!(back.package, "foo-crate");
    }

    #[test]
    fn node_line_round_trips() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("github.com/a/b", "", "src/lib.rs", "rust", "fn:open"),
            "function",
        )
        .with_line(42);
        let id = store.put_node(&n).unwrap();
        let back = store.get_node(id).unwrap().expect("node must exist");
        assert_eq!(back.line, Some(42));
    }

    #[test]
    fn node_line_none_for_file_nodes() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("github.com/a/b", "", "src/lib.rs", "rust", "file"),
            "file",
        );
        let id = store.put_node(&n).unwrap();
        let back = store.get_node(id).unwrap().expect("node must exist");
        assert_eq!(back.line, None);
    }

    #[test]
    fn node_line_coalesce_preserves_existing_on_upsert() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let vname = VName::new("g/a/b", "", "src/lib.rs", "rust", "fn:foo");
        // First insert with line = Some(10).
        let n1 = Node::new(vname.clone(), "function").with_line(10);
        store.put_node(&n1).unwrap();
        // Upsert without line — COALESCE must preserve the existing value.
        let n2 = Node::new(vname.clone(), "function");
        store.put_node(&n2).unwrap();
        let back = store.get_node(n1.id).unwrap().expect("node must exist");
        assert_eq!(
            back.line,
            Some(10),
            "COALESCE must preserve existing line on upsert"
        );
    }

    #[test]
    fn migration_v8_adds_line_column_to_existing_db() {
        // Simulate a pre-v8 database: open at v7, insert a node, then upgrade.
        // Since SqliteStore::open always runs all pending migrations, we verify
        // that a node written before the column existed reads back as line = None.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("g/a/b", "", "src/foo.ts", "typescript", "fn:bar"),
            "function",
        );
        store.put_node(&n).unwrap();
        // Re-open (migrations are idempotent — column already exists, guard applies).
        // The important assertion: existing rows have line = None after migration.
        let back = store.get_node(n.id).unwrap().expect("node must exist");
        assert_eq!(
            back.line, None,
            "pre-v8 nodes must read back as line = None"
        );
    }

    // search_nodes_by_name returns package field.
    #[test]
    fn search_nodes_by_name_returns_package() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new(
                "github.com/a/b",
                "",
                "crates/bar/src/lib.rs",
                "rust",
                "fn:bar",
            ),
            "function",
        )
        .with_package("bar-crate");
        store.put_node(&n).unwrap();
        let results = store.search_nodes_by_name("fn:bar").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].package, "bar-crate");
    }

    // T-S1: unique exact-signature match, no path hint.
    #[test]
    fn lookup_nodes_exact_unique_match_no_path_hint() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("corpus", "", "src/main.rs", "rust", "fn:run"),
            "function",
        );
        store.put_node(&n).unwrap();

        let results = store.lookup_nodes_exact("fn:run", None).unwrap();
        assert_eq!(results.len(), 1, "must find the single matching node");
        assert_eq!(results[0].vname.path, "src/main.rs");
        assert_eq!(results[0].vname.signature, "fn:run");
    }

    // T-S2: two nodes with the same signature in different files — path hint
    // must pin to the correct one and return it first.
    #[test]
    fn lookup_nodes_exact_path_pin_disambiguates() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let serve = Node::new(
            VName::new("corpus", "", "src/cmd/serve.rs", "rust", "fn:run"),
            "function",
        );
        let worker = Node::new(
            VName::new("corpus", "", "src/worker/run.rs", "rust", "fn:run"),
            "function",
        );
        store.put_node(&serve).unwrap();
        store.put_node(&worker).unwrap();

        // Pin to serve.rs via exact path.
        let pinned = store
            .lookup_nodes_exact("fn:run", Some("src/cmd/serve.rs"))
            .unwrap();
        assert!(!pinned.is_empty(), "must return the pinned node");
        assert_eq!(
            pinned[0].vname.path, "src/cmd/serve.rs",
            "path-pinned row must sort first"
        );

        // Pin via suffix only (filename).
        let by_suffix = store
            .lookup_nodes_exact("fn:run", Some("serve.rs"))
            .unwrap();
        assert!(!by_suffix.is_empty());
        assert_eq!(by_suffix[0].vname.path, "src/cmd/serve.rs");

        // Pin to worker.
        let worker_pin = store
            .lookup_nodes_exact("fn:run", Some("worker/run.rs"))
            .unwrap();
        assert!(!worker_pin.is_empty());
        assert_eq!(worker_pin[0].vname.path, "src/worker/run.rs");

        // No hint → both rows returned.
        let both = store.lookup_nodes_exact("fn:run", None).unwrap();
        assert_eq!(both.len(), 2, "without a path hint both nodes are returned");
    }

    // T-S3: file-kind nodes are excluded even when signature matches exactly.
    #[test]
    fn lookup_nodes_exact_excludes_file_kind_nodes() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let file_node = Node::new(
            VName::new("corpus", "", "src/run.rs", "rust", "fn:run"),
            "file",
        );
        store.put_node(&file_node).unwrap();

        let results = store.lookup_nodes_exact("fn:run", None).unwrap();
        assert!(
            results.is_empty(),
            "file-kind nodes must be excluded from lookup_nodes_exact"
        );
    }

    // T-S4: signature that does not exist returns an empty vec, not an error.
    #[test]
    fn lookup_nodes_exact_missing_signature_returns_empty() {
        let store = SqliteStore::open_in_memory().unwrap();
        let results = store.lookup_nodes_exact("fn:does_not_exist", None).unwrap();
        assert!(results.is_empty());
    }

    // T-S5: LIKE wildcard characters in the path hint must not be interpreted
    // as SQL wildcards (path_hint is bound as a parameter, not interpolated).
    #[test]
    fn lookup_nodes_exact_path_hint_is_not_sql_wildcard() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n = Node::new(
            VName::new("corpus", "", "src/main.rs", "rust", "fn:run"),
            "function",
        );
        store.put_node(&n).unwrap();

        // The hint "src/%" must be matched literally: since no stored path ends
        // with the literal three chars "src/%", nothing matches. Before the
        // ESCAPE fix, '%' was a wildcard and this would have matched src/main.rs.
        let results = store.lookup_nodes_exact("fn:run", Some("src/%")).unwrap();
        assert!(
            results.is_empty(),
            "wildcard characters in path_hint must not be interpolated as SQL wildcards"
        );
    }

    // T-S6: an underscore in a path hint is a literal character, not the LIKE
    // single-character wildcard. "my_file.rs" must pin only src/my_file.rs and
    // must NOT match the decoy src/myXfile.rs.
    #[test]
    fn lookup_nodes_exact_underscore_hint_is_literal() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let real = Node::new(
            VName::new("corpus", "", "src/my_file.rs", "rust", "fn:run"),
            "function",
        );
        let decoy = Node::new(
            VName::new("corpus", "", "src/myXfile.rs", "rust", "fn:run"),
            "function",
        );
        store.put_node(&real).unwrap();
        store.put_node(&decoy).unwrap();

        // Suffix pin via filename only — exercises the LIKE branch.
        let hits = store
            .lookup_nodes_exact("fn:run", Some("my_file.rs"))
            .unwrap();
        let paths: Vec<&str> = hits.iter().map(|n| n.vname.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["src/my_file.rs"],
            "underscore must be literal, decoy myXfile.rs must not match: {paths:?}"
        );
    }

    #[test]
    fn v7_migration_adds_access_corpus_and_sessions_table() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("v6.db");

        // Simulate a v6 DB (no access_corpus column, no sessions table).
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO meta VALUES('schema_version', '6');
                 CREATE TABLE nodes (
                   id INTEGER PRIMARY KEY, corpus TEXT NOT NULL, root TEXT NOT NULL,
                   path TEXT NOT NULL, language TEXT NOT NULL,
                   signature TEXT NOT NULL, kind TEXT NOT NULL,
                   format_version INTEGER NOT NULL DEFAULT 1,
                   package TEXT NOT NULL DEFAULT ''
                 );
                 CREATE TABLE edges (
                   src INTEGER NOT NULL, dst INTEGER NOT NULL, kind TEXT NOT NULL,
                   provenance TEXT NOT NULL DEFAULT 'tree-sitter',
                   confidence INTEGER,
                   PRIMARY KEY (src, dst, kind)
                 );
                 CREATE TABLE files (
                   path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
                   last_indexed_at INTEGER NOT NULL
                 );
                 INSERT INTO nodes VALUES(1,'corp','','a.rs','rust','fn:a','function',1,'');",
            )
            .unwrap();
        }

        // Open with the v7 store — migration must succeed.
        let store = SqliteStore::open(&db_path).expect("v6→v7 migration must succeed");

        // Pre-existing node must survive migration with access_corpus defaulting to NULL.
        let node = store.get_node(travsr_core::NodeId(1)).unwrap();
        assert!(
            node.is_some(),
            "pre-existing node must survive v7 migration"
        );

        // sessions table must now exist — insert + select a row.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "INSERT INTO sessions (id, corpus, expires) VALUES ('tok1', 'corp', 9999999999)",
            [],
        )
        .unwrap();
        let corpus: String = conn
            .query_row("SELECT corpus FROM sessions WHERE id = 'tok1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(
            corpus, "corp",
            "sessions table round-trip must work after v7"
        );

        // access_corpus column must exist and be nullable.
        conn.execute("UPDATE nodes SET access_corpus = 'corp' WHERE id = 1", [])
            .unwrap();
        let ac: Option<String> = conn
            .query_row("SELECT access_corpus FROM nodes WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(ac, Some("corp".to_string()));
    }

    #[test]
    fn synonym_add_multi_then_list() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.synonym_add("payment", "billing").unwrap();
        store.synonym_add("payment", "invoice").unwrap();
        let pairs = store.synonym_list().unwrap();
        let aliases: Vec<&str> = pairs
            .iter()
            .filter(|(t, _)| t == "payment")
            .map(|(_, a)| a.as_str())
            .collect();
        assert!(aliases.contains(&"billing"));
        assert!(aliases.contains(&"invoice"));
    }

    #[test]
    fn synonym_remove_term_clears_all_aliases() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.synonym_add("payment", "billing").unwrap();
        store.synonym_add("payment", "invoice").unwrap();
        store.synonym_add("payment2", "wire").unwrap();
        store.synonym_remove_term("payment").unwrap();
        let pairs = store.synonym_list().unwrap();
        assert!(
            pairs.iter().all(|(t, _)| t != "payment"),
            "all payment aliases must be removed"
        );
        assert!(
            pairs.iter().any(|(t, a)| t == "payment2" && a == "wire"),
            "payment2 alias must survive"
        );
    }

    #[test]
    fn synonym_remove_term_noop_on_missing_term() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.synonym_add("payment2", "wire").unwrap();
        store.synonym_remove_term("nonexistent").unwrap();
        let pairs = store.synonym_list().unwrap();
        assert!(pairs.iter().any(|(t, a)| t == "payment2" && a == "wire"));
    }

    #[test]
    fn synonym_set_replaces_existing_aliases() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.synonym_add("payment", "billing").unwrap();
        store.synonym_add("payment", "invoice").unwrap();
        // Exercise the real atomic synonym_set, not a hand-rolled remove+add.
        store
            .synonym_set(
                "payment",
                &["charge".to_string(), "transaction".to_string()],
            )
            .unwrap();
        let pairs = store.synonym_list().unwrap();
        let aliases: Vec<&str> = pairs
            .iter()
            .filter(|(t, _)| t == "payment")
            .map(|(_, a)| a.as_str())
            .collect();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"charge"));
        assert!(aliases.contains(&"transaction"));
        assert!(!aliases.contains(&"billing"));
        assert!(!aliases.contains(&"invoice"));
    }

    #[test]
    fn synonym_add_enforces_200_row_cap() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Fill the table up to exactly 200 rows (open() pre-seeds the defaults,
        // so this also pins that the cap counts seeded rows).
        let mut i = 0;
        while store.synonym_list().unwrap().len() < 200 {
            store.synonym_add("filler", &format!("alias{i}")).unwrap();
            i += 1;
        }
        assert_eq!(store.synonym_list().unwrap().len(), 200);
        // The 201st distinct add must be rejected.
        assert!(
            store.synonym_add("filler", "one_too_many").is_err(),
            "add beyond 200 rows must error"
        );
        // The rejected add must not have grown the table.
        assert_eq!(
            store.synonym_list().unwrap().len(),
            200,
            "rejected add must leave the table at exactly 200 rows"
        );
    }

    #[test]
    fn synonym_set_over_cap_rolls_back_entirely() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Fill to 198 rows.
        let mut i = 0;
        while store.synonym_list().unwrap().len() < 198 {
            store.synonym_add("filler", &format!("a{i}")).unwrap();
            i += 1;
        }
        let before = store.synonym_list().unwrap();
        // Setting 5 aliases on a brand-new term would push 198 + 5 = 203 > 200.
        let aliases: Vec<String> = (0..5).map(|n| format!("x{n}")).collect();
        assert!(
            store.synonym_set("newterm", &aliases).is_err(),
            "synonym_set that exceeds the cap must error"
        );
        assert_eq!(
            store.synonym_list().unwrap(),
            before,
            "a failed synonym_set must roll back the DELETE and all INSERTs"
        );
    }

    #[test]
    fn seed_synonyms_is_idempotent() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        assert!(
            !store.synonym_list().unwrap().is_empty(),
            "open must seed static defaults"
        );
        store.synonym_add("payment", "billing").unwrap();
        let after_add = store.synonym_list().unwrap().len();
        // Re-running the seeder on a non-empty table must be a no-op.
        store.seed_synonyms_if_empty().unwrap();
        assert_eq!(
            store.synonym_list().unwrap().len(),
            after_add,
            "re-seeding a non-empty table must not re-add defaults"
        );
    }

    #[test]
    fn synonyms_persist_and_dont_reseed_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("graph.db");
        {
            let mut store = SqliteStore::open(&db_path).unwrap();
            store.synonym_remove_term("auth").unwrap(); // drop a seeded default
            store.synonym_add("payment", "billing").unwrap(); // add a user row
        }
        // Reopen: the seeder must NOT re-add the removed default (table is
        // non-empty), and the user row must persist (v11 idempotency).
        let store = SqliteStore::open(&db_path).unwrap();
        let pairs = store.synonym_list().unwrap();
        assert!(
            pairs.iter().any(|(t, a)| t == "payment" && a == "billing"),
            "user-added synonym must persist across reopen"
        );
        assert!(
            !pairs.iter().any(|(t, _)| t == "auth"),
            "a removed default must not be re-seeded on reopen"
        );
    }

    #[test]
    fn synonym_remove_one_alias_leaves_others() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.synonym_add("payment", "billing").unwrap();
        store.synonym_add("payment", "invoice").unwrap();
        store.synonym_remove("payment", "billing").unwrap();
        let pairs = store.synonym_list().unwrap();
        let aliases: Vec<&str> = pairs
            .iter()
            .filter(|(t, _)| t == "payment")
            .map(|(_, a)| a.as_str())
            .collect();
        assert!(!aliases.contains(&"billing"), "removed alias must be gone");
        assert!(aliases.contains(&"invoice"), "sibling alias must survive");
    }

    #[test]
    fn has_refcall_edges_for_language_returns_false_when_none() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(!store.has_refcall_edges_for_language("typescript"));
    }

    #[test]
    fn has_refcall_edges_for_language_returns_true_after_insert() {
        use travsr_core::{Edge, EdgeKind, Node, VName};
        let mut store = SqliteStore::open_in_memory().unwrap();
        let n1 = Node::new(VName::new("", "", "a.ts", "typescript", "fn:a"), "function");
        let n2 = Node::new(VName::new("", "", "b.ts", "typescript", "fn:b"), "function");
        store.put_node(&n1).unwrap();
        store.put_node(&n2).unwrap();
        store
            .put_edge(&Edge::new(n1.id, n2.id, EdgeKind::RefCall))
            .unwrap();
        assert!(store.has_refcall_edges_for_language("typescript"));
        assert!(
            !store.has_refcall_edges_for_language("rust"),
            "different lang must return false"
        );
    }

    #[test]
    fn embed_fts_tokens_puts_doc_prose_first() {
        // RFC-022 D1: the doc prose trails a long `calls:` list in the skeleton, so
        // a naive head-slice would drop it. `embed_fts_tokens` must emit doc tokens
        // first so the conceptual signal survives the per-node cap.
        let skeleton = "function: fn:ppr_weighted | module: crates/travsr-retrieval/src/ppr.rs \
             | calls: is_empty ok vec sum map iter collect ppr hashmap len entry \
             | doc: reticulation of splines across seeds";
        let toks = SqliteStore::embed_fts_tokens(skeleton);
        let first = toks.split_whitespace().next().unwrap_or("");
        assert_eq!(
            first, "reticulation",
            "doc prose must lead the embed FTS tokens; got: {toks}"
        );
    }

    #[test]
    fn embed_text_widening_matches_doc_only_terms() {
        // RFC-022 D1 (RC-1): a conceptual query term that appears ONLY in the doc
        // ("reticulation"), never in the signature ("ppr_weighted"), must resolve the
        // symbol once its embed_text is folded into FTS via write_embed_texts_batch.
        std::env::set_var("TRAVSR_FTS_EMBED_WIDEN", "1");
        let mut store = SqliteStore::open_in_memory().unwrap();
        let node = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-retrieval/src/ppr.rs",
                "rust",
                "fn:ppr_weighted",
            ),
            "function",
        );
        let id = store.put_node(&node).unwrap();
        // Before embed_text: "reticulation" (doc-only) does not resolve the symbol.
        let before = store.search_nodes_fuzzy("reticulation").unwrap();
        assert!(
            !before
                .iter()
                .any(|n| n.vname.signature == "fn:ppr_weighted"),
            "doc-only term must not match on signature-only FTS"
        );
        // Populate embed_text (as the daemon does) — this widens the FTS content.
        store
            .write_embed_texts_batch(&[(
                id,
                "function: fn:ppr_weighted | doc: reticulation of splines across seeds".to_string(),
            )])
            .unwrap();
        let after = store.search_nodes_fuzzy("reticulation").unwrap();
        assert!(
            after.iter().any(|n| n.vname.signature == "fn:ppr_weighted"),
            "after embed_text widening, the doc term must resolve the symbol"
        );
    }

    #[test]
    fn backfill_fts_embed_text_is_idempotent_and_widens() {
        // Backfill an index whose embed_text was set directly (bypassing the FTS
        // widening), then assert the doc term resolves and a second run is a no-op.
        std::env::set_var("TRAVSR_FTS_EMBED_WIDEN", "1");
        let mut store = SqliteStore::open_in_memory().unwrap();
        let node = Node::new(
            VName::new(
                "c",
                "",
                "crates/travsr-retrieval/src/ppr.rs",
                "rust",
                "fn:ppr_weighted",
            ),
            "function",
        );
        let id = store.put_node(&node).unwrap();
        // Set embed_text WITHOUT the FTS widening path (simulates a pre-D1 index).
        store
            .conn
            .execute(
                "UPDATE nodes SET embed_text = ?2 WHERE id = ?1",
                params![
                    node_id_to_i64(id),
                    "function: fn:ppr_weighted | doc: reticulation of splines"
                ],
            )
            .unwrap();
        assert!(
            !store
                .search_nodes_fuzzy("reticulation")
                .unwrap()
                .iter()
                .any(|n| n.vname.signature == "fn:ppr_weighted"),
            "pre-backfill: doc term must not resolve"
        );
        let n1 = store.backfill_fts_embed_text().unwrap();
        assert_eq!(n1, 1, "backfill should rebuild the one embed_text node");
        assert!(
            store
                .search_nodes_fuzzy("reticulation")
                .unwrap()
                .iter()
                .any(|n| n.vname.signature == "fn:ppr_weighted"),
            "post-backfill: doc term must resolve"
        );
        let n2 = store.backfill_fts_embed_text().unwrap();
        assert_eq!(n2, 0, "second backfill is a meta-gated no-op");
    }

    #[test]
    fn bulk_init_fts_matches_row_by_row() {
        // Index the same nodes via bulk path and row-by-row path; verify
        // that FTS search returns identical results in both cases.
        let nodes = vec![
            Node::new(
                VName::new("c", "", "src/auth.ts", "typescript", "fn:AuthService"),
                "function",
            ),
            Node::new(
                VName::new("c", "", "src/pay.ts", "typescript", "fn:PaymentHandler"),
                "function",
            ),
        ];

        // ── row-by-row (reference) ──────────────────────────────────────────
        let mut ref_store = SqliteStore::open_in_memory().unwrap();
        for n in &nodes {
            ref_store.put_node(n).unwrap();
        }
        let ref_results = ref_store.search_nodes_fuzzy("auth").unwrap();

        // ── bulk path ───────────────────────────────────────────────────────
        let mut bulk_store = SqliteStore::open_in_memory().unwrap();
        let batch: Vec<FileGraph> = nodes
            .iter()
            .map(|n| FileGraph {
                vname_path: n.vname.path.clone(),
                new_hash: "deadbeef".to_string(),
                nodes: vec![n.clone()],
                edges: vec![],
                source: None,
            })
            .collect();
        bulk_store.write_file_graphs_batch(&batch, true).unwrap();
        bulk_store.rebuild_fts_from_map().unwrap();
        let bulk_results = bulk_store.search_nodes_fuzzy("auth").unwrap();

        assert_eq!(
            ref_results.len(),
            bulk_results.len(),
            "bulk FTS must return the same number of results as row-by-row"
        );
        assert!(
            bulk_results
                .iter()
                .any(|n| n.vname.signature.contains("Auth")),
            "bulk FTS must find AuthService"
        );
    }

    #[test]
    fn staging_nodes_and_edges_reach_production_after_flush() {
        let corpus = "staging-test";
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.begin_bulk_fts_tracking().unwrap();
        store.begin_staging_tables().unwrap();

        let node_a = Node::new(
            VName::new(corpus, "", "src/a.rs", "rust", "fn:foo"),
            "function",
        );
        let node_b = Node::new(
            VName::new(corpus, "", "src/b.rs", "rust", "fn:bar"),
            "function",
        );
        let edge = travsr_core::Edge {
            src: node_a.id,
            dst: node_b.id,
            kind: travsr_core::EdgeKind::RefCall,
            confidence: None,
            provenance: None,
        };

        let batch = vec![
            FileGraph {
                vname_path: "src/a.rs".into(),
                new_hash: "aaa".into(),
                nodes: vec![node_a.clone()],
                edges: vec![edge.clone()],
                source: None,
            },
            FileGraph {
                vname_path: "src/b.rs".into(),
                new_hash: "bbb".into(),
                nodes: vec![node_b.clone()],
                edges: vec![],
                source: None,
            },
        ];
        store.write_file_graphs_batch(&batch, true).unwrap();

        // Before flush: production tables must be empty.
        assert_eq!(
            store.node_count().unwrap(),
            0,
            "nodes must be in staging, not production yet"
        );
        assert_eq!(
            store.edge_count().unwrap(),
            0,
            "edges must be in staging, not production yet"
        );

        let (nodes_written, edges_written) = store.flush_staging_to_production().unwrap();

        assert_eq!(nodes_written, 2, "both nodes must reach production");
        assert_eq!(edges_written, 1, "the ref/call edge must reach production");
        assert!(
            !store.staging_active,
            "staging_active must be cleared after flush"
        );

        // Verify graph connectivity.
        let callers = store.iter_edges_to(node_b.id).unwrap();
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].src, node_a.id);
    }

    #[test]
    fn staging_deduplicates_duplicate_nodes_and_edges() {
        let corpus = "dup-test";
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.begin_bulk_fts_tracking().unwrap();
        store.begin_staging_tables().unwrap();

        let node = Node::new(
            VName::new(corpus, "", "src/lib.rs", "rust", "fn:init"),
            "function",
        )
        .with_line(1)
        .with_end_line(10);
        // Same node emitted twice (e.g. two overlapping indexing passes).
        let node_dup = node.clone();
        let edge = travsr_core::Edge {
            src: node.id,
            dst: node.id,
            kind: travsr_core::EdgeKind::RefCall,
            confidence: None,
            provenance: None,
        };

        let batch = vec![
            FileGraph {
                vname_path: "src/lib.rs".into(),
                new_hash: "aaa".into(),
                nodes: vec![node.clone()],
                edges: vec![edge.clone()],
                source: None,
            },
            FileGraph {
                vname_path: "src/lib.rs".into(),
                new_hash: "aaa".into(),
                nodes: vec![node_dup],
                edges: vec![edge],
                source: None,
            },
        ];
        store.write_file_graphs_batch(&batch, true).unwrap();
        let (nodes_written, edges_written) = store.flush_staging_to_production().unwrap();

        // GROUP BY must collapse duplicates.
        assert_eq!(nodes_written, 1, "duplicate node must be deduplicated");
        assert_eq!(edges_written, 1, "duplicate edge must be deduplicated");
    }

    #[test]
    fn staging_flush_is_idempotent_on_existing_db() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.begin_bulk_fts_tracking().unwrap();
        store.begin_staging_tables().unwrap();

        let node = Node::new(
            VName::new("c", "", "src/lib.rs", "rust", "fn:foo"),
            "function",
        )
        .with_line(1)
        .with_end_line(10);
        let batch = vec![FileGraph {
            vname_path: "src/lib.rs".into(),
            new_hash: "aaa".into(),
            nodes: vec![node.clone()],
            edges: vec![],
            source: None,
        }];
        store.write_file_graphs_batch(&batch, true).unwrap();
        store.flush_staging_to_production().unwrap();
        assert_eq!(
            store.node_count().unwrap(),
            1,
            "first flush must produce one node"
        );

        // Second init — same node already in production; ON CONFLICT must upsert cleanly.
        store.begin_staging_tables().unwrap();
        store.write_file_graphs_batch(&batch, true).unwrap();
        let result = store.flush_staging_to_production();
        assert!(result.is_ok(), "re-init must not fail: {:?}", result.err());
        assert_eq!(
            store.node_count().unwrap(),
            1,
            "re-init must not duplicate nodes"
        );
    }

    #[test]
    fn write_path_stamps_body_hash_from_source() {
        // RFC-027 #813: the bulk write path stamps body_hash from the file's own
        // source, co-written with the edges the same parse produced, so it is a
        // sound preservation witness. A caller that supplies no source leaves the
        // hash NULL — preservation forgone for that definition, never wrong.
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.begin_bulk_fts_tracking().unwrap();
        store.begin_staging_tables().unwrap();

        let src = "fn foo() {\n    bar();\n}\n";
        let stamped = Node::new(
            VName::new("c", "", "src/lib.rs", "rust", "fn:foo"),
            "function",
        )
        .with_line(1)
        .with_end_line(3);
        let unstamped = Node::new(
            VName::new("c", "", "src/nul.rs", "rust", "fn:baz"),
            "function",
        )
        .with_line(1)
        .with_end_line(3);
        let batch = vec![
            FileGraph {
                vname_path: "src/lib.rs".into(),
                new_hash: "aaa".into(),
                nodes: vec![stamped.clone()],
                edges: vec![],
                source: Some(src.to_string()),
            },
            FileGraph {
                vname_path: "src/nul.rs".into(),
                new_hash: "bbb".into(),
                nodes: vec![unstamped.clone()],
                edges: vec![],
                source: None,
            },
        ];
        store.write_file_graphs_batch(&batch, true).unwrap();
        store.flush_staging_to_production().unwrap();

        let body_hash = |id: i64| -> Option<String> {
            store
                .conn
                .query_row(
                    "SELECT body_hash FROM nodes WHERE id=?1",
                    params![id],
                    |r| r.get(0),
                )
                .unwrap()
        };
        let id = node_id_to_i64(stamped.id);
        let expect = body_hashes_for_spans(src, [(id, 1u32, Some(3u32))]);
        assert_eq!(
            body_hash(id).as_deref(),
            expect.get(&id).map(String::as_str),
            "source-bearing file must stamp the hash of its definition's body"
        );
        assert!(
            body_hash(id).is_some(),
            "the write path must stamp from source"
        );
        assert_eq!(
            body_hash(node_id_to_i64(unstamped.id)),
            None,
            "a file with no source must leave body_hash NULL (preservation forgone)"
        );
    }

    #[test]
    fn staging_fts_matches_bulk_path_after_flush() {
        let nodes = vec![
            Node::new(
                VName::new("c", "", "src/auth.rs", "rust", "fn:AuthService"),
                "function",
            ),
            Node::new(
                VName::new("c", "", "src/pay.rs", "rust", "fn:PaymentHandler"),
                "function",
            ),
        ];

        // Reference: row-by-row path.
        let mut ref_store = SqliteStore::open_in_memory().unwrap();
        for n in &nodes {
            ref_store.put_node(n).unwrap();
        }
        let ref_results = ref_store.search_nodes_fuzzy("auth").unwrap();

        // Staging path.
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.begin_bulk_fts_tracking().unwrap();
        store.begin_staging_tables().unwrap();
        let batch: Vec<FileGraph> = nodes
            .iter()
            .map(|n| FileGraph {
                vname_path: n.vname.path.clone(),
                new_hash: "hash".into(),
                nodes: vec![n.clone()],
                edges: vec![],
                source: None,
            })
            .collect();
        store.write_file_graphs_batch(&batch, true).unwrap();
        store.flush_staging_to_production().unwrap();
        store.rebuild_fts_from_map().unwrap();
        let stage_results = store.search_nodes_fuzzy("auth").unwrap();

        assert_eq!(
            ref_results.len(),
            stage_results.len(),
            "staging+flush FTS must return same results as row-by-row"
        );
    }

    #[test]
    fn import_nodes_lite_returns_only_import_rows_with_fields() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // One import node (importer path carried in VName::path) and one non-import.
        let import = Node::new(
            VName::new("c", "", "ruby/src/cat.rb", "ruby", "import:animal"),
            "import",
        );
        let func = Node::new(
            VName::new("c", "", "ruby/src/cat.rb", "ruby", "fn:cat"),
            "function",
        );
        store.put_node(&import).unwrap();
        store.put_node(&func).unwrap();

        let rows = store.import_nodes_lite().unwrap();
        assert_eq!(rows.len(), 1, "only the import node must be returned");
        let (id, sig, lang, path) = &rows[0];
        assert_eq!(*id, import.id);
        assert_eq!(sig, "import:animal");
        assert_eq!(lang, "ruby");
        assert_eq!(path, "ruby/src/cat.rb", "path must be the importing file");
    }

    #[test]
    fn bulk_init_vocab_counts_match_row_by_row() {
        let node = Node::new(
            VName::new("c", "", "src/session.ts", "typescript", "fn:SessionStore"),
            "function",
        );

        let mut ref_store = SqliteStore::open_in_memory().unwrap();
        ref_store.put_node(&node).unwrap();

        let mut bulk_store = SqliteStore::open_in_memory().unwrap();
        let batch = vec![FileGraph {
            vname_path: node.vname.path.clone(),
            new_hash: "deadbeef".to_string(),
            nodes: vec![node.clone()],
            edges: vec![],
            source: None,
        }];
        bulk_store.write_file_graphs_batch(&batch, true).unwrap();
        bulk_store.rebuild_fts_from_map().unwrap();

        // "session" and "store" are the primary tokens — both must have refcount 1.
        for tok in &["session", "store"] {
            let ref_rc = ref_store.fts_vocab_refcount(tok).unwrap();
            let bulk_rc = bulk_store.fts_vocab_refcount(tok).unwrap();
            assert_eq!(
                ref_rc, bulk_rc,
                "fts_vocab refcount for '{tok}' must match row-by-row"
            );
            assert!(
                bulk_rc.is_some(),
                "token '{tok}' must be present in fts_vocab after bulk rebuild"
            );
        }
    }

    #[test]
    fn set_bulk_init_mode_restores_synchronous() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("bulk.db");
        let mut store = SqliteStore::open(&db_path).unwrap();
        store.set_bulk_init_mode(true).unwrap();
        store.set_bulk_init_mode(false).unwrap();
        // After restoring, journal_mode must still be WAL (pragma was not changed).
        assert_eq!(store.journal_mode().unwrap().to_lowercase(), "wal");
    }

    // ── file_import_pairs ─────────────────────────────────────────────────────

    #[test]
    fn file_import_pairs_empty_store_returns_empty() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(store.file_import_pairs().unwrap().is_empty());
    }

    #[test]
    fn file_import_pairs_depends_without_resolves_to_excluded() {
        // A Depends edge exists but no ResolvesTo follows it — must return empty.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let src = Node::new(
            VName::new("", "", "pkg/a/mod.ts", "typescript", "fn:a"),
            "function",
        );
        let imp = Node::new(
            VName::new("", "", "pkg/a/mod.ts", "typescript", "import:x"),
            "import",
        );
        store.put_node(&src).unwrap();
        store.put_node(&imp).unwrap();
        store
            .put_edge(&Edge::new(src.id, imp.id, EdgeKind::Depends))
            .unwrap();
        assert!(store.file_import_pairs().unwrap().is_empty());
    }

    #[test]
    fn file_import_pairs_two_hop_chain_returned() {
        // full chain: file --Depends--> import_node --ResolvesTo--> file
        let mut store = SqliteStore::open_in_memory().unwrap();
        let src = Node::new(
            VName::new("", "", "pkg/a/mod.ts", "typescript", "pkg/a/mod.ts"),
            "file",
        );
        let imp = Node::new(
            VName::new("", "", "test/b/util.ts", "typescript", "import:b"),
            "import",
        );
        let dst = Node::new(
            VName::new("", "", "test/b/util.ts", "typescript", "test/b/util.ts"),
            "file",
        );
        store.put_node(&src).unwrap();
        store.put_node(&imp).unwrap();
        store.put_node(&dst).unwrap();
        store
            .put_edge(&Edge::new(src.id, imp.id, EdgeKind::Depends))
            .unwrap();
        store
            .put_edge(&Edge::new(imp.id, dst.id, EdgeKind::ResolvesTo))
            .unwrap();
        let pairs = store.file_import_pairs().unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "pkg/a/mod.ts");
        assert_eq!(pairs[0].1, "test/b/util.ts");
    }

    // ── resolved_dep_pairs ────────────────────────────────────────────────────

    #[test]
    fn resolved_dep_pairs_from_refcall_and_resolvesto() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        // ref/call: caller (a) -> callee (b), both real files.
        let a = Node::new(
            VName::new("", "", "crates/x/a.rs", "rust", "fn:a"),
            "function",
        );
        let b = Node::new(
            VName::new("", "", "crates/y/b.rs", "rust", "fn:b"),
            "function",
        );
        // resolves-to: import node (path = importer) -> target file.
        let imp = Node::new(
            VName::new("", "", "crates/x/a.rs", "rust", "use:z"),
            "import",
        );
        let z = Node::new(
            VName::new("", "", "crates/z/z.rs", "rust", "crates/z/z.rs"),
            "file",
        );
        // same-file ref/call must be excluded.
        let a2 = Node::new(
            VName::new("", "", "crates/x/a.rs", "rust", "fn:a2"),
            "function",
        );
        for n in [&a, &b, &imp, &z, &a2] {
            store.put_node(n).unwrap();
        }
        store
            .put_edge(&Edge::new(a.id, b.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(imp.id, z.id, EdgeKind::ResolvesTo))
            .unwrap();
        store
            .put_edge(&Edge::new(a.id, a2.id, EdgeKind::RefCall))
            .unwrap();

        let mut pairs = store.resolved_dep_pairs().unwrap();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("crates/x/a.rs".to_string(), "crates/y/b.rs".to_string()),
                ("crates/x/a.rs".to_string(), "crates/z/z.rs".to_string()),
            ],
            "must return cross-file ref/call + resolves-to pairs, excluding same-file"
        );
    }

    #[test]
    fn scip_reference_onto_field_node_is_ref_field_not_ref_call() {
        // #757: a SCIP reference whose callee is a `field` node (go/java field
        // access, e.g. `z.animals`) must be recorded as `ref/field`, matching the
        // native Rust path, so it surfaces in find_references but never in the
        // ref/call caller set.
        let corpus = "c";
        let path = "src/zoo.go";
        let mut store = SqliteStore::open_in_memory().unwrap();

        let caller = Node::new(
            VName::new(corpus, "", path, "go", "method:Zoo.Add"),
            "method",
        )
        .with_line(1)
        .with_end_line(10);
        let field = Node::new(
            VName::new(corpus, "", path, "go", "field:Zoo.animals"),
            "field",
        );
        store
            .write_scip_attributed_batch(corpus, &[caller.clone(), field.clone()], &[])
            .unwrap();

        let refs = vec![travsr_core::ScipRef {
            caller_path: path.to_string(),
            caller_line: 5,
            callee_id: field.id,
            // Even if a producer marks it a call, a field callee is never a call.
            is_call: true,
            caller_col: None,
        }];
        store
            .write_scip_attributed_batch(corpus, &[], &refs)
            .unwrap();

        // Edge is ref/field, and there is NO ref/call edge onto the field.
        let all_edges = store.all_edges().unwrap();
        assert!(
            all_edges
                .iter()
                .any(|(_, d, k, _)| *d == field.id && k == "ref/field"),
            "expected a ref/field edge onto the field: {all_edges:?}"
        );
        assert!(
            !all_edges
                .iter()
                .any(|(_, d, k, _)| *d == field.id && k == "ref/call"),
            "field must never get a ref/call edge: {all_edges:?}"
        );
        // find_references still returns the occurrence.
        let sites = store.reference_sites(field.id).unwrap();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].line, 5);
    }

    // P4 golden test: grouped span fetch produces identical (ref → enclosing_id) mapping
    // as the old per-ref SELECT, including nested/overlapping functions and the fallback
    // to file node when no function encloses the reference line.
    #[test]
    fn scip_attributed_batch_p4_attribution_golden() {
        let corpus = "test-corpus";
        let path = "src/foo.rs";
        let mut store = SqliteStore::open_in_memory().unwrap();

        // Outer function: lines 1–20.
        let outer = Node::new(VName::new(corpus, "", path, "rust", "fn:outer"), "function")
            .with_line(1)
            .with_end_line(20);

        // Inner (narrower) function: lines 5–10, nested inside outer.
        let inner = Node::new(VName::new(corpus, "", path, "rust", "fn:inner"), "function")
            .with_line(5)
            .with_end_line(10);

        // A callee symbol (definition node).
        let callee = Node::new(
            VName::new(corpus, "", "src/bar.rs", "rust", "fn:callee"),
            "function",
        );

        // Phase A: write the Phase A tree-sitter nodes so spans are in the DB.
        store
            .write_scip_attributed_batch(
                corpus,
                &[outer.clone(), inner.clone(), callee.clone()],
                &[],
            )
            .unwrap();

        // Three refs on the same file, exercising three cases:
        //   line 7  → inside inner (5–10): narrowest = inner
        //   line 15 → inside outer but NOT inner (11–20): narrowest = outer
        //   line 25 → outside every function: falls back to file node
        let refs = vec![
            travsr_core::ScipRef {
                caller_path: path.to_string(),
                caller_line: 7,
                callee_id: callee.id,
                is_call: true,
                caller_col: None,
            },
            travsr_core::ScipRef {
                caller_path: path.to_string(),
                caller_line: 15,
                callee_id: callee.id,
                is_call: true,
                caller_col: None,
            },
            travsr_core::ScipRef {
                caller_path: path.to_string(),
                caller_line: 25,
                callee_id: callee.id,
                is_call: true,
                caller_col: None,
            },
        ];

        store
            .write_scip_attributed_batch(corpus, &[], &refs)
            .unwrap();

        // Read back the edges and verify attribution.
        // all_edges returns (src: NodeId, dst: NodeId, kind: String, provenance: String).
        let all_edges = store.all_edges().unwrap();
        let ref_call_edges: Vec<_> = all_edges
            .iter()
            .filter(|(_, _, kind, _)| kind == "ref/call")
            .collect();

        // line 7 → inner
        assert!(
            ref_call_edges
                .iter()
                .any(|(src, dst, _, _)| *src == inner.id && *dst == callee.id),
            "line 7 ref must be attributed to inner function"
        );
        // line 15 → outer (not inner)
        assert!(
            ref_call_edges
                .iter()
                .any(|(src, dst, _, _)| *src == outer.id && *dst == callee.id),
            "line 15 ref must be attributed to outer function"
        );
        // line 25 → file node (attribution falls back; language = "rust" from sibling node)
        let file_vname = VName::new(corpus, "", path, "rust", "file");
        let file_id = file_vname.id();
        assert!(
            ref_call_edges
                .iter()
                .any(|(src, dst, _, _)| *src == file_id && *dst == callee.id),
            "line 25 ref must fall back to file node attribution"
        );
    }

    // ── #650: self-referential ref/call guard at the write choke point ──────────

    /// `write_scip_attributed_batch` is the single write path every language's
    /// ScipRef producer converges on (rust rust-analyzer LSIF, Dart native,
    /// SCIP-protobuf sidecars). It must drop `src == dst` `ref/call` edges for
    /// all of them — the guard is language-agnostic by construction, so this
    /// asserts the invariant across a representative language set to catch any
    /// future per-language regression. Positional collapse produced 3,115 false
    /// self-loops (17.7% of the semantic call graph) before this guard.
    #[test]
    fn attributed_batch_drops_self_referential_edges_all_languages() {
        for lang in ["rust", "typescript", "python", "go", "dart", "java"] {
            let corpus = "c";
            let caller_path = "src/a";
            let callee_path = "src/b";
            let mut store = SqliteStore::open_in_memory().unwrap();

            // `selfish` (lines 1–10): a body occurrence resolves back to itself.
            let selfish = Node::new(
                VName::new(corpus, "", caller_path, lang, "fn:selfish"),
                "function",
            )
            .with_line(1)
            .with_end_line(10);
            // `user` (lines 20–30) genuinely calls `helper` in another file.
            let user = Node::new(
                VName::new(corpus, "", caller_path, lang, "fn:user"),
                "function",
            )
            .with_line(20)
            .with_end_line(30);
            let helper = Node::new(
                VName::new(corpus, "", callee_path, lang, "fn:helper"),
                "function",
            )
            .with_line(1)
            .with_end_line(5);

            store
                .write_scip_attributed_batch(
                    corpus,
                    &[selfish.clone(), user.clone(), helper.clone()],
                    &[],
                )
                .unwrap();

            let refs = vec![
                // Occurrence at line 5 (inside `selfish`) attributed to `selfish`:
                // positional collapse → self-loop, must be dropped.
                travsr_core::ScipRef {
                    caller_path: caller_path.to_string(),
                    caller_line: 5,
                    callee_id: selfish.id,
                    is_call: true,
                    caller_col: None,
                },
                // Genuine recursion has the same shape and is also dropped:
                // recursion is not represented as a self-edge anywhere in the graph.
                travsr_core::ScipRef {
                    caller_path: caller_path.to_string(),
                    caller_line: 8,
                    callee_id: selfish.id,
                    is_call: true,
                    caller_col: None,
                },
                // Genuine cross-function call at line 25 (inside `user`) → `helper`.
                travsr_core::ScipRef {
                    caller_path: caller_path.to_string(),
                    caller_line: 25,
                    callee_id: helper.id,
                    is_call: true,
                    caller_col: None,
                },
            ];
            store
                .write_scip_attributed_batch(corpus, &[], &refs)
                .unwrap();

            // Corpus-level invariant: zero self-referential ref/call edges.
            assert_eq!(
                store.count_self_ref_call_edges().unwrap(),
                0,
                "lang={lang}: a self-referential ref/call edge leaked"
            );

            let edges = store.all_edges().unwrap();
            // The genuine cross-function edge survives.
            assert!(
                edges
                    .iter()
                    .any(|(s, d, k, _)| *s == user.id && *d == helper.id && k == "ref/call"),
                "lang={lang}: genuine cross-function edge must survive the guard"
            );
            // No self ref/call edge of any kind exists.
            assert!(
                !edges.iter().any(|(s, d, k, _)| s == d && k == "ref/call"),
                "lang={lang}: no self ref/call edge may exist"
            );
            // The dropped edge also left no occurrence site behind: `selfish` was
            // the callee only of the two self-loops, so it has zero sites.
            assert!(
                store.reference_sites(selfish.id).unwrap().is_empty(),
                "lang={lang}: a self-loop occurrence site leaked into edge_sites"
            );
        }
    }

    /// #650 clean split: a non-call reference (`is_call = false`) records a
    /// `find_references` occurrence (`edge_sites`) but must NOT create a
    /// call-graph `ref/call` edge — so `get_callers` / blast-radius stay
    /// call-only while `find_references` still covers type / `self` / path use
    /// sites. This is the invariant that keeps the WS-A cause fix from
    /// regressing `find_references` on types.
    #[test]
    fn non_call_reference_records_occurrence_but_no_edge() {
        let corpus = "c";
        let mut store = SqliteStore::open_in_memory().unwrap();
        // caller fn `user` at a.rs:1–10; callee TYPE `Widget` at b.rs:1–3.
        let user = Node::new(
            VName::new(corpus, "", "a.rs", "rust", "fn:user"),
            "function",
        )
        .with_line(1)
        .with_end_line(10);
        let widget = Node::new(
            VName::new(corpus, "", "b.rs", "rust", "struct:Widget"),
            "struct",
        )
        .with_line(1)
        .with_end_line(3);
        store
            .write_scip_attributed_batch(corpus, &[user.clone(), widget.clone()], &[])
            .unwrap();

        // A type-annotation reference to `Widget` inside `user` (line 5): the
        // occurrence is real, but it is not a call.
        let refs = vec![travsr_core::ScipRef {
            caller_path: "a.rs".to_string(),
            caller_line: 5,
            callee_id: widget.id,
            is_call: false,
            caller_col: None,
        }];
        store
            .write_scip_attributed_batch(corpus, &[], &refs)
            .unwrap();

        // No `ref/call` edge into Widget → get_callers stays call-only.
        let edges = store.all_edges().unwrap();
        assert!(
            !edges
                .iter()
                .any(|(_, d, k, _)| *d == widget.id && k == "ref/call"),
            "a non-call reference must not create a ref/call edge"
        );
        // But the occurrence site IS recorded → find_references covers it.
        let sites = store.reference_sites(widget.id).unwrap();
        assert_eq!(
            sites.len(),
            1,
            "non-call reference must record a find_references occurrence"
        );
        assert_eq!(sites[0].line, 5);
    }

    /// Reproduces issue #650's exact producer path: the rust-analyzer LSIF
    /// positional resolver maps a callee's *definition line* to its enclosing
    /// node, and `write_scip_attributed_batch` maps the *occurrence line* to its
    /// enclosing node. When both land in the same function (an occurrence inside
    /// the function whose def-line the callee resolves to), the two positional
    /// lookups collapse to one node. The write guard must catch it.
    #[test]
    fn lsif_positional_collapse_produces_no_self_loop() {
        let corpus = "c";
        let mut store = SqliteStore::open_in_memory().unwrap();

        // `f` spans lines 1–10 in a.rs; `g` spans 20–30; `h` spans 1–5 in b.rs.
        let f = Node::new(VName::new(corpus, "", "a.rs", "rust", "fn:f"), "function")
            .with_line(1)
            .with_end_line(10);
        let g = Node::new(VName::new(corpus, "", "a.rs", "rust", "fn:g"), "function")
            .with_line(20)
            .with_end_line(30);
        let h = Node::new(VName::new(corpus, "", "b.rs", "rust", "fn:h"), "function")
            .with_line(1)
            .with_end_line(5);
        store
            .write_scip_attributed_batch(corpus, &[f.clone(), g.clone(), h.clone()], &[])
            .unwrap();

        let positional = vec![
            // Occurrence at a.rs:5 (inside f) whose callee *definition* is a.rs:1
            // (f's own def line, inside f's span) → callee resolves to f →
            // positional collapse → self-loop candidate.
            travsr_core::LsifPositionalRef {
                caller_path: "a.rs".to_string(),
                caller_line: 5,
                callee_def_path: "a.rs".to_string(),
                callee_def_line: 1,
                is_call: true,
                caller_col: None,
            },
            // Genuine call: occurrence at a.rs:25 (inside g) → callee def b.rs:1 (h).
            travsr_core::LsifPositionalRef {
                caller_path: "a.rs".to_string(),
                caller_line: 25,
                callee_def_path: "b.rs".to_string(),
                callee_def_line: 1,
                is_call: true,
                caller_col: None,
            },
        ];

        // Resolve callee def-lines → ScipRefs (the E3 W3b path), then write.
        let refs = store
            .resolve_lsif_positional_refs(corpus, &positional)
            .unwrap();
        assert_eq!(refs.len(), 2, "both positional refs must resolve a callee");
        store
            .write_scip_attributed_batch(corpus, &[], &refs)
            .unwrap();

        assert_eq!(
            store.count_self_ref_call_edges().unwrap(),
            0,
            "positional collapse must not persist a self-loop"
        );
        let edges = store.all_edges().unwrap();
        assert!(
            edges
                .iter()
                .any(|(s, d, k, _)| *s == g.id && *d == h.id && k == "ref/call"),
            "the genuine g → h call must survive"
        );
    }

    /// `--fix` remediation for DBs written before the guard: `fsck` counts and
    /// sweeps self-referential ref/call edges, leaving legitimate cross edges —
    /// and their occurrence sites — intact. (A self-loop *site* can no longer be
    /// created through any public write path: `record_edge_sites` and the guarded
    /// `write_scip_attributed_batch` both refuse `src == dst`; the sweep's
    /// `edge_sites` DELETE remains as defensive cleanup for pre-guard DBs, and
    /// this test proves its `src == dst` scoping does not touch cross sites.)
    #[test]
    fn sweep_self_ref_call_edges_removes_edge_preserves_cross() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let f = Node::new(VName::new("c", "", "a.rs", "rust", "fn:f"), "function");
        let g = Node::new(VName::new("c", "", "a.rs", "rust", "fn:g"), "function");
        store.put_node(&f).unwrap();
        store.put_node(&g).unwrap();

        // A pre-guard self-loop edge, plus a legitimate cross edge and its site.
        store
            .put_edge(&Edge::new(f.id, f.id, EdgeKind::RefCall))
            .unwrap();
        store
            .put_edge(&Edge::new(g.id, f.id, EdgeKind::RefCall))
            .unwrap();
        store.record_edge_sites(&[(g.id, f.id, 5, None)]).unwrap();

        assert_eq!(store.count_self_ref_call_edges().unwrap(), 1);
        assert_eq!(
            store.reference_sites(f.id).unwrap().len(),
            1,
            "cross occurrence site present before sweep"
        );

        assert_eq!(
            store.sweep_self_ref_call_edges().unwrap(),
            1,
            "exactly one self edge swept"
        );
        assert_eq!(store.count_self_ref_call_edges().unwrap(), 0);

        // The legitimate cross edge and its occurrence site are untouched.
        assert!(
            store
                .all_edges()
                .unwrap()
                .iter()
                .any(|(s, d, k, _)| *s == g.id && *d == f.id && k == "ref/call"),
            "legitimate cross ref/call edge must be preserved"
        );
        assert_eq!(
            store.reference_sites(f.id).unwrap().len(),
            1,
            "cross occurrence site must survive the self-loop sweep"
        );
    }

    // ── L3: tombstone at-risk count ────────────────────────────────────────────

    /// A file-backed store (not `open_in_memory`) so `embed_db_path` is set and
    /// `prune_tombstones` can ATTACH a real `embed.db` sibling for the at-risk
    /// JOIN. Returns the store and the tempdir (kept alive for the store's life).
    fn file_backed_store() -> (SqliteStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&dir.path().join("graph.db")).unwrap();
        (store, dir)
    }

    fn write_embedding_row(embed_db_path: &std::path::Path, node_id: i64) {
        let conn = rusqlite::Connection::open(embed_db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS node_embeddings (
                 node_id   INTEGER NOT NULL,
                 model_id  TEXT    NOT NULL,
                 embedding BLOB    NOT NULL,
                 text_hash TEXT,
                 PRIMARY KEY (node_id, model_id)
             ) WITHOUT ROWID;",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO node_embeddings (node_id, model_id, embedding) VALUES (?1, ?2, ?3)",
            rusqlite::params![node_id, "arctic-embed-m-v1.5", vec![0u8; 8]],
        )
        .unwrap();
    }

    #[test]
    fn prune_reports_zero_at_risk_when_every_pruned_node_is_gone() {
        let (mut store, dir) = file_backed_store();
        let n = sample_node("fn:a");
        let id = node_id_to_i64(n.id);
        store.put_node(&n).unwrap();
        // The v17 trigger inserts a node_tombstones row on this delete.
        store
            .conn
            .execute("DELETE FROM nodes WHERE id = ?1", rusqlite::params![id])
            .unwrap();
        // The node still has an embedding row on disk (a stale one, since the
        // node itself is gone) — this must NOT count as at-risk: nothing
        // reads that vector once the node is gone, it is the orphan sweep's
        // job (freshness.rs), not this one's.
        write_embedding_row(&dir.path().join("embed.db"), id);
        // Age the tombstone past the cutoff so it is eligible for pruning.
        store
            .conn
            .execute("UPDATE node_tombstones SET deleted_at = 0", [])
            .unwrap();

        let (total, at_risk) = store.prune_tombstones(3600, 100).unwrap();
        assert_eq!(total, 1, "the aged tombstone must be pruned");
        assert_eq!(
            at_risk, 0,
            "the node is gone, so this must not count as at-risk"
        );
    }

    #[test]
    fn prune_reports_at_risk_when_a_pruned_node_still_has_a_vector() {
        let (mut store, dir) = file_backed_store();
        let n = sample_node("fn:a");
        let id = node_id_to_i64(n.id);
        store.put_node(&n).unwrap();
        store
            .conn
            .execute("DELETE FROM nodes WHERE id = ?1", rusqlite::params![id])
            .unwrap();
        // Node identity is a deterministic VName hash (Kythe-style), not an
        // autoincrement — re-inserting the same VName (e.g. a revert, or the
        // daemon's hash-delta loop reconciling back to prior content) yields
        // the exact same id. The old tombstone from the delete above is now
        // stale: the node it named is back, and has a vector.
        store.put_node(&n).unwrap();
        write_embedding_row(&dir.path().join("embed.db"), id);
        store
            .conn
            .execute("UPDATE node_tombstones SET deleted_at = 0", [])
            .unwrap();

        let (total, at_risk) = store.prune_tombstones(3600, 100).unwrap();
        assert_eq!(total, 1, "the aged tombstone must be pruned");
        assert_eq!(
            at_risk, 1,
            "the node still exists and still has an embedding; this is the real risk case"
        );
    }

    /// Regression: `prune_tombstones` used to `ATTACH`/`DETACH` inside its own
    /// transaction. SQLite refuses `DETACH` while a transaction is open, the
    /// error was swallowed, and the alias stayed bound — so the *second* call
    /// on the same connection failed `ATTACH ... already in use`, silently
    /// reporting `at_risk = 0` forever. `embed_progress` shares the `edb`
    /// alias, so the same connection would also stop reporting embed progress,
    /// which is what the daemon's auto-spawn decision reads.
    #[test]
    fn prune_measures_at_risk_on_every_call_not_just_the_first() {
        let (mut store, dir) = file_backed_store();
        let embed_db = dir.path().join("embed.db");

        for name in ["fn:a", "fn:b"] {
            let n = sample_node(name);
            let id = node_id_to_i64(n.id);
            store.put_node(&n).unwrap();
            store
                .conn
                .execute("DELETE FROM nodes WHERE id = ?1", rusqlite::params![id])
                .unwrap();
            store.put_node(&n).unwrap();
            write_embedding_row(&embed_db, id);
            store
                .conn
                .execute("UPDATE node_tombstones SET deleted_at = 0", [])
                .unwrap();

            let (total, at_risk) = store.prune_tombstones(3600, 100).unwrap();
            assert_eq!(total, 1, "{name}: the aged tombstone must be pruned");
            assert_eq!(
                at_risk, 1,
                "{name}: at-risk must be measured on this call too; a leaked `edb` \
                 attachment silently degrades it to 0"
            );
        }

        // The alias must be free for the next consumer on this connection.
        store
            .embed_progress("arctic-embed-m-v1.5", 0)
            .expect("embed_progress must still be able to ATTACH edb after a prune");
    }

    /// A node embedded under several models has one `node_embeddings` row per
    /// model, and multi-model rows are exactly what `travsr embed gc` makes
    /// normal. Counting raw join rows let `at_risk` exceed the tombstones
    /// actually pruned.
    #[test]
    fn prune_at_risk_counts_nodes_not_embedding_rows() {
        let (mut store, dir) = file_backed_store();
        let n = sample_node("fn:a");
        let id = node_id_to_i64(n.id);
        store.put_node(&n).unwrap();
        store
            .conn
            .execute("DELETE FROM nodes WHERE id = ?1", rusqlite::params![id])
            .unwrap();
        store.put_node(&n).unwrap();

        let embed_db = dir.path().join("embed.db");
        write_embedding_row(&embed_db, id);
        let conn = rusqlite::Connection::open(&embed_db).unwrap();
        conn.execute(
            "INSERT INTO node_embeddings (node_id, model_id, embedding) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, "bge-large-en-v1.5", vec![0u8; 8]],
        )
        .unwrap();
        drop(conn);

        store
            .conn
            .execute("UPDATE node_tombstones SET deleted_at = 0", [])
            .unwrap();

        let (total, at_risk) = store.prune_tombstones(3600, 100).unwrap();
        assert_eq!(total, 1);
        assert_eq!(
            at_risk, 1,
            "one tombstone, one node, two models; at_risk counts nodes, and must never \
             exceed the {total} tombstone(s) pruned"
        );
    }

    /// The exact shape observed on this repo's own index: 10,857 nodes against
    /// 11,156 map rows, which reported `missing=-299`.
    #[test]
    fn backfill_counts_never_report_a_negative() {
        // Stale map rows outnumber nodes.
        assert_eq!(super::backfill_counts(10_857, 11_156), (0, 299));
        // The ordinary direction: rows still to index.
        assert_eq!(super::backfill_counts(11_156, 10_857), (299, 0));
        // In sync.
        assert_eq!(super::backfill_counts(500, 500), (0, 0));
        // Signed saturating_sub saturates at i64::MIN rather than zero, which
        // is what produced the negative in the first place.
        assert_eq!(super::backfill_counts(0, i64::MAX).0, 0);
    }

    /// A cross-file Phase A edge must not depend on the order the batch writer
    /// happened to receive the two files in.
    ///
    /// `import:./billing --resolves-to--> file:src/billing.ts` has its `src` in
    /// the importer and its `dst` in the target. Re-writing the target used to
    /// delete every edge pointing *into* its nodes, so the importer's edge
    /// survived only when the importer came last in the batch. Batch order on
    /// the init path is whatever the parallel parse workers deliver, so
    /// `travsr init --force` dropped a random subset of `resolves-to` edges
    /// with no error.
    ///
    /// Deterministic by construction: it does not race two threads, it pins
    /// both orders of the same batch and requires them to agree.
    #[test]
    fn a_rewritten_batch_keeps_cross_file_edges_in_either_order() {
        let file_node =
            |path: &str| Node::new(VName::new("c", "", path, "typescript", "file"), "file");
        let import_node = Node::new(
            VName::new("c", "", "src/caller.ts", "typescript", "import:./billing"),
            "import",
        );
        let billing = file_node("src/billing.ts");
        let caller = file_node("src/caller.ts");

        let caller_graph = || FileGraph {
            vname_path: "src/caller.ts".into(),
            new_hash: "h-caller".into(),
            nodes: vec![caller.clone(), import_node.clone()],
            edges: vec![
                Edge::new(caller.id, import_node.id, EdgeKind::Depends),
                Edge::new(import_node.id, billing.id, EdgeKind::ResolvesTo),
            ],
            source: None,
        };
        let billing_graph = || FileGraph {
            vname_path: "src/billing.ts".into(),
            new_hash: "h-billing".into(),
            nodes: vec![billing.clone()],
            edges: vec![],
            source: None,
        };
        let resolves_to = |store: &SqliteStore| -> i64 {
            store
                .conn
                .query_row(
                    "SELECT count(*) FROM edges WHERE kind = 'resolves-to'",
                    [],
                    |r| r.get(0),
                )
                .unwrap()
        };

        // Both orders re-write an index that already holds both files, which is
        // what `--force` does: its graph purge leaves the nodes of on-disk
        // files in place, so every file takes the incremental branch below.
        //
        // The seed uses the SAME order as the rewrite, so `importer_first` also
        // covers the INSERT half: the import target is a file this batch has
        // not written yet when the importer is processed, which is what a newly
        // added file looks like. Seeding target-first would have satisfied its
        // own precondition either way and never exercised that.
        for importer_first in [false, true] {
            let batch = || {
                if importer_first {
                    vec![caller_graph(), billing_graph()]
                } else {
                    vec![billing_graph(), caller_graph()]
                }
            };
            let mut store = SqliteStore::open_in_memory().unwrap();
            store.write_file_graphs_batch(&batch(), false).unwrap();
            assert_eq!(
                resolves_to(&store),
                1,
                "the first index must resolve the import \
                 (importer_first = {importer_first})"
            );

            store.write_file_graphs_batch(&batch(), false).unwrap();
            assert_eq!(
                resolves_to(&store),
                1,
                "re-indexing both files must keep the import edge \
                 (importer_first = {importer_first})"
            );
        }
    }

    /// The other half of the ownership rule: a symbol the re-parse dropped must
    /// still take its inbound edges with it, or the narrower delete above would
    /// trade a lost edge for a dangling one.
    #[test]
    fn a_rewritten_batch_drops_inbound_edges_of_a_removed_symbol() {
        let caller = Node::new(
            VName::new("c", "", "src/caller.ts", "typescript", "fn:checkout"),
            "function",
        );
        let charge = Node::new(
            VName::new("c", "", "src/billing.ts", "typescript", "fn:charge"),
            "function",
        );

        let mut store = SqliteStore::open_in_memory().unwrap();
        store
            .write_file_graphs_batch(
                &[
                    FileGraph {
                        vname_path: "src/billing.ts".into(),
                        new_hash: "h1".into(),
                        nodes: vec![charge.clone()],
                        edges: vec![],
                        source: None,
                    },
                    FileGraph {
                        vname_path: "src/caller.ts".into(),
                        new_hash: "h-caller".into(),
                        nodes: vec![caller.clone()],
                        edges: vec![Edge::new(caller.id, charge.id, EdgeKind::RefCall)],
                        source: None,
                    },
                ],
                false,
            )
            .unwrap();
        assert_eq!(store.count_orphans().unwrap(), 0);

        // `charge` is renamed away; nothing re-derives the caller's edge.
        store
            .write_file_graphs_batch(
                &[FileGraph {
                    vname_path: "src/billing.ts".into(),
                    new_hash: "h2".into(),
                    nodes: vec![Node::new(
                        VName::new("c", "", "src/billing.ts", "typescript", "fn:bill"),
                        "function",
                    )],
                    edges: vec![],
                    source: None,
                }],
                false,
            )
            .unwrap();

        assert_eq!(
            store.count_orphans().unwrap(),
            0,
            "a removed symbol must take its inbound edges with it"
        );
    }
}
