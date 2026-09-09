//! RFC-021 Phase 2: safe call-site wrapper around `travsr-rerank`.
//!
//! Everything here degrades to `None` ("no opinion") rather than propagating
//! an error: a missing model, a load failure, an inference panic, or a
//! breaker-open skip must never crash the daemon or block a query — the caller
//! (`seed::build_seed_set`) treats `None` as "leave ordering/confidence
//! untouched", which is what makes Phase 2 safe to ship dark. A slow pass that
//! actually completes keeps its scores and only feeds [`Breaker`]; it is the
//! *next* calls that skip, never the one that was measured.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use travsr_rerank::{RerankManifest, Reranker, TractReranker};

/// Escape hatch: forces the lexical-only fallback regardless of whether a
/// model is configured. Distinct from "no model configured" so operators can
/// disable the reranker on an otherwise-bundled install (minimal installs,
/// Phase 5).
fn rerank_disabled() -> bool {
    std::env::var("TRAVSR_NO_RERANK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Candidates reranked per query. Bounds the forward-pass cost — the plan's
/// K ≈ 20-40 window; default matches the RFC.
pub(crate) fn rerank_topk() -> usize {
    std::env::var("TRAVSR_RERANK_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &usize| x > 0)
        .unwrap_or(30)
}

/// Per-call latency budget. A CPU-bound forward pass can't be safely aborted
/// mid-flight without unsafe thread termination, so this is necessarily
/// measured *after* the call completes — which means it can never make the
/// call that exceeded it any cheaper. It therefore drives [`Breaker`], which
/// skips inference on *subsequent* calls, and never invalidates the scores of
/// the call it measured.
///
/// Read as a UX policy, not a hardware calibration: "a single interactive
/// rerank may take up to this long". That reading is what makes one number
/// portable across an M-series laptop, an OCI A1 core and a Windows CI box —
/// the alternative (a number predicting how fast the model runs) is wrong on
/// every machine but the one it was measured on. A slow machine trips the
/// breaker and stops paying the cost; a fast one never trips it.
///
/// Default 1200, unchanged since RFC-021 Phase 3 E2E (2026-07-18), and still a
/// backstop rather than the primary defense against repo size. The primary
/// defense is `travsr-rerank::MAX_CANDIDATE_CHARS`, which bounds each
/// candidate's tokenizer input so real cost stays roughly repo-agnostic:
/// before that fix, K=30 on kubernetes/kubernetes measured 2.5-9s (vs ~353ms
/// on travsr's own compact Rust fns) because 40-line Go snippets routinely
/// blew past the tokenizer's truncation ceiling, and
/// `PaddingStrategy::BatchLongest` then padded every candidate in a batch up
/// to the longest one. After the input-size fix, the same K=30 on k8s measured
/// 700-950ms, and both candidate pools are hard-capped (30 code lane, 20 doc
/// lane), so per-call cost no longer scales with the repo.
fn rerank_budget_ms() -> u128 {
    std::env::var("TRAVSR_RERANK_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&x: &u128| x > 0)
        .unwrap_or(1200)
}

/// Default for [`BreakerLimits::trip`]: over-budget calls within
/// [`BreakerLimits::window`] recent completed calls that open [`Breaker`]. More
/// than one, because a single slow call is almost always contention from
/// unrelated work on the developer's machine (a build, a test run) rather than
/// a statement about the hardware — measured on this repo:
/// rerank costs 91-671ms per call on an idle 8-core M3 and 1505-3319ms with the
/// CPU saturated by concurrent builds. Reacting to one sample would disable the
/// reranker every time the user compiles something.
const BREAKER_TRIP_COUNT: u32 = 3;

/// Override for [`BREAKER_TRIP_COUNT`]. Clamped to the effective window: a trip
/// count above the window can never be reached, which would silently disable
/// the breaker.
const BREAKER_TRIP_ENV: &str = "TRAVSR_RERANK_BREAKER_TRIP";

/// Override for [`BREAKER_WINDOW`]. Rejected outside `1..=BREAKER_WINDOW_MAX`.
const BREAKER_WINDOW_ENV: &str = "TRAVSR_RERANK_BREAKER_WINDOW";

/// Default for [`BreakerLimits::window`]: how many recent completed calls the
/// trip count is measured over. A *window*, not a strict consecutive streak,
/// because every `get_context` reranks two lanes against one budget — the
/// heavier code lane (30 candidates, `seed.rs`) then the lighter doc lane (20,
/// `tools.rs`) — through one shared
/// [`BREAKER`]. On a machine where only the code lane runs over budget the
/// per-query sequence is `over, under`, so a consecutive-streak counter that
/// any in-budget call reset would never open, even though the reranker is over
/// budget on every single query. `BREAKER_TRIP_COUNT` of the last
/// `BREAKER_WINDOW` calls trips on that pattern while still ignoring an isolated
/// spike from a background build (at most two of the window). A window is more
/// trigger-happy than a streak by construction: at a 5% rate of isolated slow
/// calls about 1% of calls lose reranking, rising to ~6% at 10% and ~17% at
/// 15%, by which point the machine is struggling for real. Kept at or below
/// [`BREAKER_WINDOW_MAX`] so the window fits one atomic word.
const BREAKER_WINDOW: u32 = 5;

/// Hard ceiling on the window, because the window lives in one `u32` and the
/// mask is `(1 << window) - 1`. At 32 that shift overflows: debug builds panic,
/// and release builds compute a mask of `0`, which makes `is_open` permanently
/// false and silently deletes the breaker. A rejected override is far better
/// than either, so [`BreakerLimits::from_env`] refuses anything above this.
const BREAKER_WINDOW_MAX: u32 = 31;

/// The two trip parameters, resolved together because they constrain each
/// other. Overridable so the trip policy can be reproduced and calibrated on a
/// shipped binary: it has a *shape* now ("N of the last M") rather than a single
/// number, the sensible values are machine-dependent, and this crate's own
/// measurements are from one machine. Without these, checking a claim about the
/// policy or trading latency protection for keeping RFC-021 on would mean a
/// custom build, or faking `TRAVSR_RERANK_BUDGET_MS`, which also moves the thing
/// being measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BreakerLimits {
    /// Over-budget calls within `window` that open the breaker.
    trip: u32,
    /// How many recent completed calls `trip` is measured over.
    window: u32,
}

impl BreakerLimits {
    /// Low-bit mask selecting the `window` most recent calls out of
    /// [`Breaker::recent_over_budget`].
    fn mask(self) -> u32 {
        // `window <= BREAKER_WINDOW_MAX` (31) is enforced on construction, so
        // this shift cannot overflow.
        (1u32 << self.window) - 1
    }

    /// Resolve both from the environment, falling back to the defaults. Any
    /// unparseable or out-of-range value is ignored rather than rejected loudly:
    /// this is a diagnostic knob, and a typo in it must not take the reranker
    /// down a path the defaults would not have taken.
    fn from_env() -> Self {
        Self::resolve(
            std::env::var(BREAKER_WINDOW_ENV).ok().as_deref(),
            std::env::var(BREAKER_TRIP_ENV).ok().as_deref(),
        )
    }

    /// The parsing and clamping rules, taking the raw strings so they are
    /// unit-testable without mutating process-global environment or pinning the
    /// `OnceLock` that caches the resolved value.
    fn resolve(window_raw: Option<&str>, trip_raw: Option<&str>) -> Self {
        let window = window_raw
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&w| w > 0 && w <= BREAKER_WINDOW_MAX)
            .unwrap_or(BREAKER_WINDOW);
        // Clamped to the window in both directions, the default included: with a
        // window of 2, a trip of 3 (the default) could never be reached and the
        // breaker would never open at all.
        let trip = trip_raw
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&t| t > 0)
            .unwrap_or(BREAKER_TRIP_COUNT)
            .min(window);
        Self { trip, window }
    }
}

/// While the breaker is open, one call in this many is let through as a probe,
/// so a machine that was merely busy gets the reranker back without restarting
/// the daemon. Deliberately counted in calls rather than wall-clock time: no
/// clock source to disagree about across platforms, and it costs at most one
/// slow query per interval.
const BREAKER_PROBE_INTERVAL: u32 = 16;

/// A real circuit breaker for the reranker: it opens once [`BreakerLimits::trip`]
/// of the last [`BreakerLimits::window`] completed calls ran over budget and
/// then *skips* inference, which is the only point at which skipping saves
/// anything.
///
/// It deliberately does NOT judge the scores of a call that already ran. The
/// cross-encoder is deterministic (`travsr-rerank`: same input, identical
/// output), so how long a call took carries no information about whether its
/// scores are good — discarding them spent the latency and threw away the
/// ranking, which is strictly worse than either running or not running.
///
/// Owns its counters, and can be handed its trip policy up front
/// ([`Breaker::with_limits`]), so the decision logic is unit-testable without
/// reading process-global state or hitting the `OnceLock` pinning caveat that
/// applies to [`RERANKER`]. The [`BREAKER`] static resolves that policy from the
/// environment on first use instead.
struct Breaker {
    /// Sliding window over recent completed calls: bit *i* (from the LSB) is
    /// whether the *i*-th most recent call was over budget, keeping only the low
    /// [`BreakerLimits::window`] bits. Bounded by construction, so it cannot
    /// overflow however long the process runs. The breaker is open when at least
    /// [`BreakerLimits::trip`] of those bits are set.
    recent_over_budget: std::sync::atomic::AtomicU32,
    /// Calls skipped since the last probe; drives [`BREAKER_PROBE_INTERVAL`].
    skipped_since_probe: std::sync::atomic::AtomicU32,
    /// Trip policy, resolved from the environment on first use. A `OnceLock`
    /// rather than a resolve-per-call so `is_open` and `record` cannot disagree
    /// about the mask within one call, and so [`new`](Self::new) stays `const`
    /// and can build the [`BREAKER`] static.
    limits: OnceLock<BreakerLimits>,
}

impl Breaker {
    const fn new() -> Self {
        Self {
            recent_over_budget: std::sync::atomic::AtomicU32::new(0),
            skipped_since_probe: std::sync::atomic::AtomicU32::new(0),
            limits: OnceLock::new(),
        }
    }

    /// A breaker with its trip policy fixed up front, so the decision logic can
    /// be unit-tested at any policy without touching process-global env or
    /// pinning the shared `OnceLock`.
    #[cfg(test)]
    fn with_limits(trip: u32, window: u32) -> Self {
        let breaker = Self::new();
        let _ = breaker.limits.set(BreakerLimits { trip, window });
        breaker
    }

    fn limits(&self) -> BreakerLimits {
        *self.limits.get_or_init(BreakerLimits::from_env)
    }

    /// Open when at least [`BreakerLimits::trip`] of the last
    /// [`BreakerLimits::window`] completed calls were over budget.
    fn is_open(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let limits = self.limits();
        (self.recent_over_budget.load(Relaxed) & limits.mask()).count_ones() >= limits.trip
    }

    /// `true` when this call should skip inference entirely and fail open to
    /// the lexical gate. Lets one call in [`BREAKER_PROBE_INTERVAL`] through
    /// while open so the breaker can close again on its own.
    fn should_skip(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.is_open() {
            return false;
        }
        if self.skipped_since_probe.fetch_add(1, Relaxed) + 1 >= BREAKER_PROBE_INTERVAL {
            self.skipped_since_probe.store(0, Relaxed);
            return false;
        }
        true
    }

    /// Record a completed call. Only calls that actually ran reach here: a
    /// skipped call never runs, and the error/panic arms of [`rerank`] return
    /// before recording, so a model that fails fast or blows up (a different
    /// failure mode from a slow one) does not feed the window. Logs only on the
    /// open/close transitions, so a sustained slow machine warns once rather
    /// than once per query.
    fn record(&self, over_budget: bool, elapsed_ms: u128, budget_ms: u128) {
        use std::sync::atomic::Ordering::Relaxed;
        // An in-budget call reaching here while the breaker is open is usually a
        // probe, and one probe back within budget means the machine has
        // recovered, so close immediately rather than ageing the window out one
        // probe at a time (three probes at `BREAKER_PROBE_INTERVAL` apiece is a
        // far worse recovery for a machine that was merely busy).
        //
        // "Usually", not "only". Two other kinds of call land here, and the
        // close is taken for them too:
        //
        //   * a call that passed `should_skip` while the breaker was still
        //     closed and was overtaken mid-flight by other threads opening it.
        //     Only possible concurrently, which is also the only place the
        //     breaker opens at all: none observed at 1-2 in-flight queries,
        //     tens per 8000 calls at 4-8.
        //   * a genuine probe that landed on the doc lane rather than the code
        //     lane that is actually slow. Both lanes are conditional (`seed.rs`
        //     gates the code lane on non-empty seeds, `tools.rs` returns early
        //     with no doc candidates), so a probe's lane is not fixed, and a
        //     doc-lane probe is in budget by construction and says nothing
        //     about the code lane.
        //
        // Both make the breaker more forgiving than the probe interval alone
        // suggests: measurably more over-budget calls get through than a
        // single-threaded reading predicts. That is the fail-open direction, and
        // it is the price of one shared breaker across two lanes (see
        // [`BREAKER`]) rather than a defect in this branch.
        if !over_budget && self.is_open() {
            self.recent_over_budget.store(0, Relaxed);
            self.skipped_since_probe.store(0, Relaxed);
            tracing::info!(elapsed_ms, "rerank back within budget, resuming");
            return;
        }
        let bit = u32::from(over_budget);
        // Read once for the whole call so the mask that shifts the window and
        // the trip count that reads it cannot come from different policies.
        let limits = self.limits();
        let mask = limits.mask();
        // The closure never returns `None`, so `fetch_update` always yields the
        // previous window; the `else` is unreachable but keeps this panic-free.
        let Ok(prev) = self
            .recent_over_budget
            .fetch_update(Relaxed, Relaxed, |w| Some(((w << 1) | bit) & mask))
        else {
            return;
        };
        let now = ((prev << 1) | bit) & mask;
        // Exactly one warn per opening, and `fetch_update` is what makes that
        // true under concurrency: it is a CAS loop, so the window's values are
        // linearized and each recorder's `prev` is the true predecessor of its
        // own `now`. A second warn would need a `prev` below the trip count that
        // is also some earlier recorder's at-or-above-trip `now`, which cannot
        // both hold. Verified exhaustively over every window state and by
        // stressing concurrent recorders.
        if now.count_ones() >= limits.trip && prev.count_ones() < limits.trip {
            tracing::warn!(
                elapsed_ms,
                threshold_ms = budget_ms,
                trip_count = limits.trip,
                window = limits.window,
                "rerank is consistently over budget on this machine, skipping inference \
                 on further queries and falling back to the lexical gate; it is retried \
                 periodically"
            );
        }
    }
}

/// Process-wide breaker. Shared by both call sites (code lane in `seed.rs`,
/// doc lane in `tools.rs`) on purpose: what it measures is this machine's
/// capacity to run the model interactively, not a property of either lane.
static BREAKER: Breaker = Breaker::new();

/// The canonical per-user install location for the reranker model, mirroring
/// the embed backends' `~/.travsr/models/<id>` layout (see `embed.rs`). P5 ships
/// the model here via `travsr rerank install` / the daemon's background
/// auto-fetch. `None` only when the home directory can't be resolved.
pub(crate) fn default_rerank_model_dir() -> Option<PathBuf> {
    Some(
        dirs::home_dir()?
            .join(".travsr")
            .join("models")
            .join("rerank"),
    )
}

/// Directory containing `model_fp16.onnx` + `tokenizer.json`.
///
/// Resolution order:
/// 1. `TRAVSR_RERANK_MODEL_DIR` — explicit override (dev / CI / offline / a
///    custom model). Returned verbatim; existence is `TractReranker::load`'s job.
/// 2. The default install dir `~/.travsr/models/rerank`, but **only when the
///    model file is actually present** — so a fresh machine that has not fetched
///    the model yet stays `None` (fail-open / ships-dark) rather than pointing
///    the loader at an empty directory.
fn rerank_model_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("TRAVSR_RERANK_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let default = default_rerank_model_dir()?;
    default
        .join(travsr_rerank::MODEL_FILE)
        .exists()
        .then_some(default)
}

/// `None` once load has been attempted (missing config, missing files, or a
/// load-time error) — cached permanently so a broken install doesn't retry an
/// expensive failed load on every query.
///
/// Test-process caveat (RFC-021 F6): first initialization wins for the whole
/// test binary — whichever test calls `reranker()` first (today, always with
/// no model dir configured) permanently pins this to `None` for every test in
/// this module, run in any order. A model-backed wrapper test would flake
/// under the default parallel test harness; the right shape for that is
/// extracting a `fn load_from(dir: Option<PathBuf>) -> Option<TractReranker>`
/// out of the `get_or_init` closure and testing it directly, leaving this
/// static untouched.
static RERANKER: OnceLock<Option<TractReranker>> = OnceLock::new();
static WARM_STARTED: std::sync::Once = std::sync::Once::new();

/// The per-model floor manifest (`model.toml`) next to the model, cached like
/// [`RERANKER`]. `None` when no model dir is configured or the manifest is
/// absent/malformed — the floor read (`seed.rs`) then falls back to the
/// compiled-in `travsr_rerank::DEFAULT_*` floors. Same test-process caveat as
/// `RERANKER`: first init wins for the whole test binary.
static MANIFEST: OnceLock<Option<RerankManifest>> = OnceLock::new();

/// Load (once) the reranker floor manifest for the configured model dir. Runs
/// the self-heal migration (writes `model.toml` for a manual/dev install that
/// predates F8) and, once, verifies the recorded sha256 against the on-disk
/// model so stale floors surface as a warning. Independent of `rerank_disabled`
/// / model *load* success: floors are cheap to resolve and harmless to read
/// even when inference itself is off.
fn manifest() -> Option<&'static RerankManifest> {
    MANIFEST
        .get_or_init(|| {
            let dir = rerank_model_dir()?;
            travsr_rerank::ensure_manifest(&dir);
            let manifest = RerankManifest::read(&dir)?;
            travsr_rerank::warn_on_sha_mismatch(&manifest, &dir);
            Some(manifest)
        })
        .as_ref()
}

/// STRONG floor from the model manifest, if one is present. `seed.rs` prefers
/// an explicit env override, then this, then the compiled-in default.
pub(crate) fn manifest_strong_floor() -> Option<f32> {
    manifest().map(|m| m.strong_floor)
}

/// WEAK floor from the model manifest, if one is present. See
/// [`manifest_strong_floor`].
pub(crate) fn manifest_weak_floor() -> Option<f32> {
    manifest().map(|m| m.weak_floor)
}

/// Human-readable reranker state for `travsr status`. Honest in both contexts:
/// answered by the warm daemon it reflects the loaded model; on the cold CLI
/// fast-path (no daemon) it reflects on-disk presence, since that process never
/// warms the model itself.
///
/// - `off` — `TRAVSR_NO_RERANK` set.
/// - `not installed` — no model configured and none at the default location.
/// - `installed` — model present on disk but not loaded in *this* process.
/// - `ready` — model loaded and serving.
/// - `load failed` — a load was attempted and failed (fail-open to lexical).
pub(crate) fn rerank_status() -> &'static str {
    if rerank_disabled() {
        return "off";
    }
    if rerank_model_dir().is_none() {
        return "not installed";
    }
    match RERANKER.get() {
        Some(Some(_)) => "ready",
        Some(None) => "load failed",
        None => "installed",
    }
}

// ── Model distribution (RFC-021 P5) ──────────────────────────────────────────
//
// The cross-encoder model is byte-identical across every travsr version, so it
// does NOT ride the per-version binary tarball (which also has a 25 MB gate the
// 45.9 MB model would blow). It lives on its own immutable `rerank-model-v1`
// GitHub release and is fetched into `~/.travsr/models/rerank` on demand.
// Integrity is pinned by a committed sha256 per file (tamper-evident,
// independent of the mutable release listing). When the model changes (#462),
// cut a new tag and bump the tag + both hashes together.

/// Immutable, version-independent model release tag.
const MODEL_TAG: &str = "rerank-model-v1";
/// sha256 of `model_fp16.onnx` (ms-marco-MiniLM-L-6-v2, fp16, ~45.9 MB).
const MODEL_SHA256: &str = "a59cbeb92524ab794e3bf3c7a93f5de73bd59b74241b65e81de46dc5bbf4c3ab";
/// sha256 of `tokenizer.json` (HF fast-tokenizer, ~0.7 MB).
const TOKENIZER_SHA256: &str = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
/// Reject a response an order of magnitude past the real sizes before hashing.
const MODEL_SIZE_LIMIT: u64 = 120 * 1024 * 1024;
/// Base URL for the release-asset download. Overridable for tests/mirrors.
const RELEASES_BASE_ENV: &str = "TRAVSR_RERANK_RELEASES_BASE";
/// Shared hermetic hook (also used by `travsr lang install`): skip the network
/// and just report the destination directory.
const SKIP_DOWNLOAD_ENV: &str = "TRAVSR_SKIP_DOWNLOAD";
/// Opt out of the daemon's background auto-fetch without disabling the reranker
/// entirely (an already-installed model still loads).
const NO_AUTOFETCH_ENV: &str = "TRAVSR_NO_RERANK_AUTOFETCH";
const DEFAULT_RELEASES_BASE: &str = "https://github.com/Travsr-com/travsr/releases/download";

/// `true` when both model files are present at the default install location.
pub fn model_installed() -> bool {
    default_rerank_model_dir().is_some_and(|dir| {
        dir.join(travsr_rerank::MODEL_FILE).is_file()
            && dir.join(travsr_rerank::TOKENIZER_FILE).is_file()
    })
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = <sha2::Sha256 as sha2::Digest>::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

async fn download_verified(
    base: &str,
    file: &str,
    expected_sha: &str,
    client: &reqwest::Client,
) -> anyhow::Result<Vec<u8>> {
    use anyhow::bail;
    let url = format!("{base}/{MODEL_TAG}/{file}");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("requesting {url}: {e}"))?;
    if !resp.status().is_success() {
        bail!("download failed ({}): {url}", resp.status());
    }
    if let Some(len) = resp.content_length() {
        if len > MODEL_SIZE_LIMIT {
            bail!("{file} exceeds size limit ({len} bytes)");
        }
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| anyhow::anyhow!("reading {file} body: {e}"))?;
    if bytes.len() as u64 > MODEL_SIZE_LIMIT {
        bail!(
            "{file} exceeds size limit after download ({} bytes)",
            bytes.len()
        );
    }
    let actual = hex_sha256(&bytes);
    if actual != expected_sha {
        bail!("sha256 mismatch for {file}: expected {expected_sha}, got {actual}");
    }
    Ok(bytes.to_vec())
}

fn write_atomic(dir: &std::path::Path, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Per-call sequence so two threads writing the same asset in one process
    // (pid alone is not unique across threads) target distinct temp files and
    // never interleave writes before the rename.
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let dest = dir.join(name);
    let tmp = dir.join(format!(
        "{name}.tmp.{}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, bytes).map_err(|e| anyhow::anyhow!("writing {}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::anyhow!("moving into place {}: {e}", dest.display()));
    }
    Ok(())
}

/// Fetch + verify + install `model_fp16.onnx` + `tokenizer.json` into
/// `~/.travsr/models/rerank`, then write `model.toml` (F8 manifest). Idempotent
/// (re-downloads/overwrites, so it doubles as a repair). `TRAVSR_SKIP_DOWNLOAD`
/// short-circuits the network and returns the target dir (CI/offline).
pub async fn install_model() -> anyhow::Result<PathBuf> {
    let dir = default_rerank_model_dir()
        .ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", dir.display()))?;
    if std::env::var_os(SKIP_DOWNLOAD_ENV).is_some() {
        return Ok(dir);
    }

    let base =
        std::env::var(RELEASES_BASE_ENV).unwrap_or_else(|_| DEFAULT_RELEASES_BASE.to_string());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .user_agent(concat!("travsr-mcp/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| anyhow::anyhow!("building HTTP client: {e}"))?;

    let (model, tokenizer) = tokio::try_join!(
        download_verified(&base, travsr_rerank::MODEL_FILE, MODEL_SHA256, &client),
        download_verified(
            &base,
            travsr_rerank::TOKENIZER_FILE,
            TOKENIZER_SHA256,
            &client
        ),
    )?;

    write_atomic(&dir, travsr_rerank::MODEL_FILE, &model)?;
    write_atomic(&dir, travsr_rerank::TOKENIZER_FILE, &tokenizer)?;
    travsr_rerank::ensure_manifest(&dir);
    Ok(dir)
}

/// Blocking [`install_model`], safe to call from a plain `std::thread` OR from
/// inside a running tokio runtime: it drives the future on a fresh runtime in a
/// scoped worker thread, so it never panics with "runtime within a runtime"
/// (the `travsr rerank install` CLI path runs inside `#[tokio::main]`).
pub fn install_model_blocking() -> anyhow::Result<PathBuf> {
    std::thread::scope(|s| {
        s.spawn(|| {
            tokio::runtime::Runtime::new()
                .map_err(|e| anyhow::anyhow!("creating tokio runtime: {e}"))?
                .block_on(install_model())
        })
        .join()
        .map_err(|_| anyhow::anyhow!("model-download thread panicked"))?
    })
}

/// P5 auto-fetch: on daemon start, if the reranker is enabled and no model is
/// present at the default location, download it (blocking, in the warm thread)
/// BEFORE `reranker()` pins the `OnceLock`, so the model activates this session.
/// Every exit is a silent no-op → ships-dark fallback; never fatal.
fn maybe_autofetch_model() {
    if rerank_disabled() || std::env::var_os(NO_AUTOFETCH_ENV).is_some() {
        return;
    }
    // An explicit model dir is the operator's responsibility — never auto-write
    // into a location they pointed us at on purpose.
    if std::env::var_os("TRAVSR_RERANK_MODEL_DIR").is_some() {
        return;
    }
    if model_installed() {
        return;
    }
    tracing::info!("reranker model absent, fetching in the background");
    match install_model_blocking() {
        Ok(dir) => tracing::info!(dir = %dir.display(), "reranker model ready"),
        Err(error) => tracing::warn!(
            %error,
            "reranker model download failed, continuing without it"
        ),
    }
}

fn reranker() -> Option<&'static TractReranker> {
    if rerank_disabled() {
        return None;
    }
    RERANKER
        .get_or_init(|| {
            let dir = rerank_model_dir()?;
            // catch_unwind around the load itself: `tract` has internal panics
            // reachable via a malformed/truncated ONNX (a bad
            // TRAVSR_RERANK_MODEL_DIR override, or a partially-written install).
            // `OnceLock` does NOT cache a panicking initializer, so an escaping
            // panic would both propagate through the current query and re-run the
            // failed load on every subsequent one. Map a load panic to `None` so
            // the fail-open decision is cached once and the daemon degrades to the
            // lexical gate, matching the `Err` arm (RFC-021 G5/G6 fail-open).
            match catch_unwind(AssertUnwindSafe(|| TractReranker::load(&dir))) {
                Ok(Ok(r)) => {
                    tracing::info!(model_dir = %dir.display(), "reranker loaded");
                    Some(r)
                }
                Ok(Err(error)) => {
                    tracing::warn!(
                        %error,
                        model_dir = %dir.display(),
                        "reranker failed to load, continuing without it"
                    );
                    None
                }
                Err(_) => {
                    tracing::warn!(
                        model_dir = %dir.display(),
                        "reranker crashed while loading, continuing without it"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Spawn a background thread that loads the reranker eagerly, so the first
/// real query doesn't pay the (model-load) cold-start cost. Idempotent and
/// non-blocking — safe to call from every server entrypoint (stdio,
/// stdio-global, SSE); only the first call actually spawns anything.
pub(crate) fn warm_background() {
    WARM_STARTED.call_once(|| {
        std::thread::Builder::new()
            .name("rerank-warm".into())
            .spawn(|| {
                // P5: fetch the model first (no-op when present/disabled) so the
                // load below sees it and the reranker activates this session —
                // `reranker()` pins its OnceLock, so a later download would not
                // be picked up until the next daemon start.
                maybe_autofetch_model();
                reranker();
                // Resolve the manifest here too (self-heal write + one-time
                // sha256 verification) so the first query never pays the model
                // hash on its hot path.
                manifest();
            })
            .ok();
    });
}

/// Score `candidates` against `query`. Returns `None` — "no opinion" — when
/// the reranker is absent, disabled, panicked, errored, or skipped because the
/// circuit breaker is open; the caller must leave ordering/confidence untouched
/// in that case. Never panics itself. `Some(scores)` is always the same length
/// as `candidates`, aligned by index.
///
/// Scores that were actually computed are always returned. An over-budget call
/// feeds [`BREAKER`] so the *next* queries can skip the cost, but its own
/// output is kept: the model is deterministic, so elapsed time says nothing
/// about score quality, and discarding it paid the full latency and then
/// silently reinstated the confident-salad bug RFC-021 exists to cut.
///
/// [`BREAKER`] is a process-global static, so in practice it only opens in a
/// long-lived MCP or daemon process. The one-shot `travsr ask` CLI path runs
/// at most two rerank calls per process (code lane, doc lane) and starts fresh
/// every invocation, so it never accumulates enough over-budget calls to open —
/// the score-keeping half above is what helps that path.
pub(crate) fn rerank(query: &str, candidates: &[&str]) -> Option<Vec<f32>> {
    let reranker = reranker()?;
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    if BREAKER.should_skip() {
        return None;
    }

    let start = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| reranker.rerank(query, candidates)));
    let elapsed_ms = start.elapsed().as_millis();

    let scores = match outcome {
        Ok(Ok(scores)) => scores,
        Ok(Err(error)) => {
            tracing::warn!(%error, "rerank inference failed, falling back to lexical gate");
            return None;
        }
        Err(_) => {
            tracing::warn!("rerank inference panicked, falling back to lexical gate");
            return None;
        }
    };

    let budget = rerank_budget_ms();
    let over_budget = elapsed_ms > budget;
    // Per-call timing at a level nobody sees by default. This is the only place
    // "how close to the budget is this machine" is observable now that `record`
    // logs only the open/close transitions; force it on for a run with
    // `RUST_LOG=travsr_mcp::rerank=debug`.
    tracing::debug!(
        elapsed_ms,
        over_budget,
        budget_ms = budget,
        "rerank complete"
    );
    BREAKER.record(over_budget, elapsed_ms, budget);
    Some(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate process-global env vars; the default test
    /// harness runs test fns on parallel threads and `set_var`/`remove_var`
    /// are process-wide (RFC-021 F6).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn rerank_topk_default_is_thirty() {
        std::env::remove_var("TRAVSR_RERANK_TOPK");
        assert_eq!(rerank_topk(), 30);
    }

    #[test]
    fn rerank_budget_default_is_1200ms() {
        std::env::remove_var("TRAVSR_RERANK_BUDGET_MS");
        assert_eq!(rerank_budget_ms(), 1200);
    }

    #[test]
    fn breaker_stays_closed_for_isolated_slow_calls() {
        // A single slow call is contention from unrelated work, not a verdict
        // on the machine: it must stay a minority of the window, so the reranker
        // keeps running. (Contrast `breaker_opens_when_one_lane_is_consistently_
        // over_budget`, where a slow call recurs every query.)
        let breaker = Breaker::with_limits(BREAKER_TRIP_COUNT, BREAKER_WINDOW);
        for _ in 0..10 {
            assert!(!breaker.should_skip());
            breaker.record(true, 5_000, 1_200);
            assert!(!breaker.should_skip(), "one slow call must not open it");
            // Let it age fully out of the window before the next isolated spike.
            for _ in 0..BREAKER_WINDOW {
                breaker.record(false, 300, 1_200);
            }
        }
    }

    #[test]
    fn breaker_opens_when_one_lane_is_consistently_over_budget() {
        // Every get_context reranks two lanes against one budget through the one
        // shared breaker: the heavier code lane, then the lighter doc lane. On a
        // machine where only the code lane runs over budget the per-query
        // sequence is (over, under). A consecutive-streak counter that any
        // in-budget call reset would never open here (the doc lane resets it
        // every query); the windowed trip count must, because the reranker is
        // over budget on every query. This models the real call path: a skipped
        // call never records.
        let breaker = Breaker::with_limits(BREAKER_TRIP_COUNT, BREAKER_WINDOW);
        let call = |over_budget: bool| {
            if breaker.should_skip() {
                return;
            }
            let elapsed = if over_budget { 5_000 } else { 300 };
            breaker.record(over_budget, elapsed, 1_200);
        };
        let mut opened = false;
        for _ in 0..10 {
            call(true); // code lane, over budget
            call(false); // doc lane, under budget
            if breaker.should_skip() {
                opened = true;
                break;
            }
        }
        assert!(
            opened,
            "the breaker must open when a lane is over budget every query"
        );
    }

    #[test]
    fn breaker_opens_only_after_the_trip_count_then_skips_inference() {
        let breaker = Breaker::with_limits(BREAKER_TRIP_COUNT, BREAKER_WINDOW);
        for _ in 0..BREAKER_TRIP_COUNT - 1 {
            assert!(
                !breaker.should_skip(),
                "must not open before the trip count"
            );
            breaker.record(true, 5_000, 1_200);
        }
        assert!(!breaker.should_skip());
        breaker.record(true, 5_000, 1_200);
        // Open: the cost is now skipped rather than paid and discarded.
        assert!(breaker.should_skip());
    }

    #[test]
    fn open_breaker_probes_and_closes_once_calls_are_back_in_budget() {
        let breaker = Breaker::with_limits(BREAKER_TRIP_COUNT, BREAKER_WINDOW);
        for _ in 0..BREAKER_TRIP_COUNT {
            breaker.record(true, 5_000, 1_200);
        }
        // Skips until the probe interval elapses, then lets exactly one through.
        for _ in 0..BREAKER_PROBE_INTERVAL - 1 {
            assert!(breaker.should_skip());
        }
        assert!(!breaker.should_skip(), "one call in N must probe");
        // The probe came back in budget, so the breaker closes.
        breaker.record(false, 300, 1_200);
        assert!(!breaker.should_skip());
    }

    #[test]
    fn open_breaker_stays_open_while_probes_are_still_over_budget() {
        let breaker = Breaker::with_limits(BREAKER_TRIP_COUNT, BREAKER_WINDOW);
        for _ in 0..BREAKER_TRIP_COUNT {
            breaker.record(true, 5_000, 1_200);
        }
        for _ in 0..BREAKER_PROBE_INTERVAL - 1 {
            assert!(breaker.should_skip());
        }
        assert!(!breaker.should_skip());
        breaker.record(true, 5_000, 1_200);
        assert!(breaker.should_skip(), "a slow probe must not close it");
    }

    /// The window override's sharp edge, which is why it is range-checked at
    /// all: `BreakerLimits::mask` is `(1 << window) - 1`, so a window of 32 or
    /// more overflows the shift — a debug build panics and a release build gets
    /// a mask of 0, which makes `is_open` permanently false and silently deletes
    /// the breaker. Out-of-range and unparseable values must fall back to the
    /// default rather than reach `mask` at all.
    #[test]
    fn breaker_window_override_rejects_values_that_would_break_the_mask() {
        let window_of = |raw: Option<&str>| BreakerLimits::resolve(raw, None).window;

        assert_eq!(window_of(None), BREAKER_WINDOW, "unset uses the default");
        assert_eq!(window_of(Some("8")), 8, "an in-range override applies");
        assert_eq!(window_of(Some(" 8 ")), 8, "surrounding whitespace is fine");
        assert_eq!(
            window_of(Some("31")),
            BREAKER_WINDOW_MAX,
            "the ceiling is usable"
        );

        for rejected in ["32", "64", "4294967295", "0", "-1", "", "banana", "5.5"] {
            assert_eq!(
                window_of(Some(rejected)),
                BREAKER_WINDOW,
                "{rejected:?} must fall back to the default, not reach the mask"
            );
        }

        // The property the range check exists to protect, stated directly.
        for window in 1..=BREAKER_WINDOW_MAX {
            let mask = BreakerLimits { trip: 1, window }.mask();
            assert_eq!(
                mask.count_ones(),
                window,
                "mask must select exactly {window} bits"
            );
        }
    }

    /// A trip count above the window can never be reached, so it would silently
    /// disable the breaker. Clamping covers the override *and* the default: a
    /// window of 2 with the default trip of 3 is the same trap arrived at from
    /// the other side.
    #[test]
    fn breaker_trip_override_is_clamped_to_the_window() {
        let limits = |w: Option<&str>, t: Option<&str>| BreakerLimits::resolve(w, t);

        assert_eq!(
            limits(None, None),
            BreakerLimits {
                trip: BREAKER_TRIP_COUNT,
                window: BREAKER_WINDOW
            }
        );
        assert_eq!(
            limits(None, Some("2")).trip,
            2,
            "an in-range override applies"
        );
        assert_eq!(
            limits(Some("5"), Some("9")).trip,
            5,
            "a trip above the window is clamped to it"
        );
        assert_eq!(
            limits(Some("2"), None).trip,
            2,
            "the default is clamped too"
        );
        for rejected in ["0", "", "nope"] {
            assert_eq!(
                limits(None, Some(rejected)).trip,
                BREAKER_TRIP_COUNT,
                "{rejected:?}"
            );
        }
    }

    /// `resolve` is tested above without touching the environment; this is the
    /// one test that the documented variable *names* are the ones actually read,
    /// which no amount of `resolve` coverage can catch. Safe to mutate env here:
    /// `from_env` holds no `OnceLock` of its own, so nothing is pinned by it.
    ///
    /// The names are spelled out as literals rather than reused from the
    /// constants. Setting `BREAKER_WINDOW_ENV` and asserting that
    /// `BREAKER_WINDOW_ENV` was read passes for any string at all, including a
    /// typo, which is exactly the failure this test exists to catch.
    #[test]
    fn breaker_limits_read_the_documented_env_var_names() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                std::env::remove_var("TRAVSR_RERANK_BREAKER_WINDOW");
                std::env::remove_var("TRAVSR_RERANK_BREAKER_TRIP");
            }
        }
        let _restore = Restore;

        assert_eq!(BREAKER_WINDOW_ENV, "TRAVSR_RERANK_BREAKER_WINDOW");
        assert_eq!(BREAKER_TRIP_ENV, "TRAVSR_RERANK_BREAKER_TRIP");

        std::env::remove_var("TRAVSR_RERANK_BREAKER_WINDOW");
        std::env::remove_var("TRAVSR_RERANK_BREAKER_TRIP");
        assert_eq!(
            BreakerLimits::from_env(),
            BreakerLimits {
                trip: BREAKER_TRIP_COUNT,
                window: BREAKER_WINDOW
            }
        );

        std::env::set_var("TRAVSR_RERANK_BREAKER_WINDOW", "9");
        std::env::set_var("TRAVSR_RERANK_BREAKER_TRIP", "4");
        assert_eq!(
            BreakerLimits::from_env(),
            BreakerLimits { trip: 4, window: 9 },
            "both overrides must be read from the names the docs advertise"
        );
    }

    /// The overrides must reach the decision logic, not just parse. Pinning only
    /// the parser would leave `is_open` and `record` free to keep using the
    /// defaults while the tests above stayed green.
    #[test]
    fn overridden_limits_drive_the_trip_decision() {
        // Trip at 2 of the last 2: opens strictly sooner than the default 3-of-5.
        let breaker = Breaker::with_limits(2, 2);
        breaker.record(true, 5_000, 1_200);
        assert!(!breaker.should_skip(), "one over-budget call is not two");
        breaker.record(true, 5_000, 1_200);
        assert!(breaker.should_skip(), "2 of the last 2 must open it");

        // A wider window with the same trip tolerates the isolated spikes that
        // the narrow one above trips on.
        let breaker = Breaker::with_limits(2, 8);
        for _ in 0..4 {
            breaker.record(true, 5_000, 1_200);
            assert!(
                !breaker.should_skip(),
                "spikes spread across the window must not trip"
            );
            for _ in 0..8 {
                breaker.record(false, 300, 1_200);
            }
        }
    }

    #[test]
    fn no_model_configured_is_none_not_panic() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("TRAVSR_NO_RERANK");
        // Point discovery at a guaranteed-absent dir rather than *removing* the
        // override: with RFC-021 P5 the model is default-installed at
        // ~/.travsr/models/rerank on any machine that has run the daemon, so
        // clearing the override would let reranker() find that real model and
        // load Some — the assertion must stay hermetic regardless of local
        // installs. A nonexistent dir exercises the identical "no usable model
        // -> None" degrade path (loader fails -> cached None), which is the
        // ships-dark contract this test guards.
        std::env::set_var(
            "TRAVSR_RERANK_MODEL_DIR",
            "/nonexistent/travsr-rerank-none-test",
        );
        assert!(reranker().is_none());
        assert_eq!(rerank("anything", &["a", "b"]), None);
        std::env::remove_var("TRAVSR_RERANK_MODEL_DIR");
    }

    #[test]
    fn disabled_flag_short_circuits_even_with_model_dir() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_NO_RERANK", "1");
        std::env::set_var("TRAVSR_RERANK_MODEL_DIR", "/nonexistent/path/for/test");
        let result = rerank("q", &["a"]);
        std::env::remove_var("TRAVSR_NO_RERANK");
        std::env::remove_var("TRAVSR_RERANK_MODEL_DIR");
        assert_eq!(result, None);
    }

    #[test]
    fn rerank_model_dir_prefers_env_override_verbatim() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_RERANK_MODEL_DIR", "/custom/model/dir");
        let dir = rerank_model_dir();
        std::env::remove_var("TRAVSR_RERANK_MODEL_DIR");
        // Override wins and is returned as-is (existence is the loader's job).
        assert_eq!(dir, Some(PathBuf::from("/custom/model/dir")));
    }

    #[test]
    fn default_rerank_model_dir_is_under_travsr_models_rerank() {
        // P5 canonical location, mirroring embed's ~/.travsr/models/<id>.
        let dir = default_rerank_model_dir().expect("home dir resolvable in test env");
        assert!(dir.ends_with("rerank"), "got {dir:?}");
        assert!(dir.parent().unwrap().ends_with("models"), "got {dir:?}");
        assert!(dir.to_string_lossy().contains(".travsr"), "got {dir:?}");
    }

    #[test]
    fn rerank_status_off_when_disabled() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TRAVSR_NO_RERANK", "1");
        let s = rerank_status();
        std::env::remove_var("TRAVSR_NO_RERANK");
        assert_eq!(s, "off");
    }

    #[test]
    fn hex_sha256_matches_known_vector() {
        // sha256("") — canonical empty-input digest.
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn pinned_model_hashes_are_lowercase_hex_64() {
        for h in [MODEL_SHA256, TOKENIZER_SHA256] {
            assert_eq!(h.len(), 64, "sha256 hex is 64 chars: {h}");
            assert!(
                h.bytes()
                    .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase()),
                "pinned hash must be lowercase hex: {h}"
            );
        }
    }
}
