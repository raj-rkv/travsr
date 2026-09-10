//! Axum SSE transport for the cloud/team MCP server (RFC-007).
//!
//! Routes:
//!   GET  /sse      — open a persistent SSE stream (scope: mcp)
//!   POST /rpc      — submit a JSON-RPC 2.0 request (scope: mcp)
//!   GET  /health   — liveness probe (unauthenticated)
//!   GET  /metrics  — Prometheus text format (scope: metrics)

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use dashmap::DashMap;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::auth::{auth_error_status, verify_token, AuthError, TokenScope, VerifiedToken};
use crate::session::SessionStore;
use crate::tools;

// ── Types ─────────────────────────────────────────────────────────────────────

pub type TenantId = String;

/// Shared application state — all fields must be `Send + Sync`.
pub struct AppState {
    /// Active SSE sessions keyed by session UUID.
    pub sessions: DashMap<Uuid, SseSessionHandle>,
    /// Tenant repo registries: tenant_id → repo_name → db_path.
    pub tenant_repos: DashMap<TenantId, HashMap<String, PathBuf>>,
    /// Signing keys for bearer token verification.
    pub signing_keys: Vec<[u8; 32]>,
    /// Ring buffer for SSE event replay (at-most-once delivery on reconnect).
    /// Key: (tenant_id, session_id). Value: VecDeque of (event_id, received_at, json_payload).
    pub ring_buffers: RingBufferMap,
    /// Per-session monotonic event counter.
    pub event_counters: DashMap<Uuid, u64>,
    /// In-flight RPC requests: (tenant_id, request_id) → ().
    ///
    /// `Arc`'d so [`InFlightGuard`] can hold the map independently of the
    /// `AppState` borrow and remove its entry in `Drop` (#736 item 6).
    pub in_flight: Arc<DashMap<(TenantId, String), ()>>,
    /// RBAC session store (RFC-006, S14).
    ///
    /// Held here so the periodic maintenance task spawned in [`router`] can
    /// call `evict_expired()`. Expiry is otherwise only enforced lazily on
    /// `get()`, so a session that is created but never looked up again would
    /// be retained until process exit (#736 item 6).
    pub session_store: SessionStore,
    /// Bounds concurrent `dispatch_tool_call` executions (#736 item 8).
    ///
    /// `spawn_blocking` alone has no concurrency limit — the only backstop is
    /// tokio's default 512-thread blocking pool, and 512 concurrent reranks /
    /// graph walks is an OOM, not a limit. RPC handlers await an owned permit
    /// before spawning, which is the natural backpressure point.
    pub dispatch_permits: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(
        tenant_repos: DashMap<TenantId, HashMap<String, PathBuf>>,
        signing_keys: Vec<[u8; 32]>,
    ) -> Self {
        // #736 item 8: 2× the core count keeps the blocking pool busy while
        // I/O-bound dispatches overlap; the floor of 8 keeps small containers
        // (1–2 vCPU) from serialising every RPC behind a couple of slots.
        let dispatch_slots = std::cmp::max(
            8,
            2 * std::thread::available_parallelism().map_or(4, usize::from),
        );
        Self {
            sessions: DashMap::new(),
            tenant_repos,
            signing_keys,
            ring_buffers: DashMap::new(),
            event_counters: DashMap::new(),
            in_flight: Arc::new(DashMap::new()),
            session_store: SessionStore::new(),
            dispatch_permits: Arc::new(tokio::sync::Semaphore::new(dispatch_slots)),
        }
    }
}

/// Handle to a live SSE session stored in AppState.
pub struct SseSessionHandle {
    pub tenant_id: TenantId,
    pub tx: mpsc::Sender<SseEvent>,
    pub opened_at: Instant,
}

/// Events that can be sent on an SSE stream.
pub enum SseEvent {
    JsonRpc(String),
    Keepalive,
    Close,
}

// Ring buffer limits.
const RING_BUFFER_CAPACITY: usize = 1000;
const RING_BUFFER_TTL: std::time::Duration = std::time::Duration::from_secs(300);
/// Per-session byte budget for buffered payloads (#736 item 6).
///
/// The entry cap alone is not a memory bound: entries are JSON-RPC response
/// chunks of up to `MAX_RESPONSE_BYTES` (512 KB), so 1000 of them is ~500 MB
/// for a single session. 4 MiB still replays ~8 full-size chunks — plenty for
/// the reconnect window the TTL allows — while capping worst-case residency.
const RING_BUFFER_MAX_BYTES: usize = 4 * 1024 * 1024;

/// A single ring-buffer entry: (event_id, received_at, json_payload).
type RingEntry = (u64, Instant, String);
/// The ring buffer map type.
type RingBufferMap = DashMap<(TenantId, Uuid), RingBuffer>;

/// Per-session replay buffer with byte accounting (#736 item 6).
///
/// `total_bytes` is maintained on every push/evict so the byte-budget check is
/// O(evicted) per push instead of re-summing the deque.
#[derive(Default)]
pub struct RingBuffer {
    entries: VecDeque<RingEntry>,
    /// Invariant: always equals the sum of `payload.len()` over `entries`.
    total_bytes: usize,
}

impl RingBuffer {
    /// Push with the production limits. See [`Self::push_bounded`] for the
    /// eviction order.
    fn push(&mut self, entry: RingEntry) {
        self.push_bounded(
            entry,
            RING_BUFFER_CAPACITY,
            RING_BUFFER_TTL,
            RING_BUFFER_MAX_BYTES,
        );
    }

    /// Push `entry`, evicting oldest-first in this order:
    /// 1. entries older than `ttl` (pre-existing TTL-on-push behaviour),
    /// 2. down to `capacity - 1` entries (pre-existing count bound),
    /// 3. until `entry` fits inside `max_bytes` (the #736 byte budget).
    ///
    /// A payload larger than `max_bytes` on its own is REFUSED, making the
    /// budget a hard bound — the same contract as `ParseCache::insert` and
    /// `QueryCache::put` (#737 review). The cost is a replay gap for that one
    /// event id on reconnect, which the ring already tolerates (TTL and the
    /// entry cap drop events too), and `MAX_RESPONSE_BYTES` chunking keeps
    /// real payloads far below the budget, so the refusal is a last-resort
    /// guard rather than an expected path.
    ///
    /// Limits are parameters (rather than reading the consts directly) so the
    /// eviction logic is testable without multi-MiB fixtures.
    fn push_bounded(&mut self, entry: RingEntry, capacity: usize, ttl: Duration, max_bytes: usize) {
        let now = entry.1;
        // Evict expired entries.
        while let Some(front) = self.entries.front() {
            if now.duration_since(front.1) > ttl {
                self.pop_front();
            } else {
                break;
            }
        }
        // Refuse payloads that could never fit: buffering one would drain the
        // whole ring and still leave total_bytes above the budget by up to a
        // whole tool response.
        if entry.2.len() > max_bytes {
            tracing::debug!(
                bytes = entry.2.len(),
                max_bytes,
                "SSE replay entry exceeds the ring budget, not buffered"
            );
            return;
        }
        // Enforce the entry cap.
        while self.entries.len() >= capacity {
            self.pop_front();
        }
        // Enforce the byte budget.
        while self.total_bytes + entry.2.len() > max_bytes {
            self.pop_front();
        }
        self.total_bytes += entry.2.len();
        self.entries.push_back(entry);
    }

    /// Remove the oldest entry, keeping `total_bytes` in sync.
    fn pop_front(&mut self) {
        if let Some((_, _, payload)) = self.entries.pop_front() {
            self.total_bytes -= payload.len();
        }
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// How often the background maintenance task sweeps expired state (#736 item 6).
///
/// 600 s is deliberately coarse: the sweep exists to bound memory over hours of
/// idleness, not to make expiry precise — lookups and pushes still enforce
/// expiry lazily in the meantime.
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(600);

/// Build the axum router. Path-only spans are included; Authorization headers are
/// never captured in trace spans.
pub fn router(state: Arc<AppState>) -> Router {
    use axum::routing::{get, post};

    // RFC-021: background-warm the reranker (idempotent, non-blocking) so the
    // first real request doesn't pay the model-load cost.
    crate::rerank::warm_background();

    // #736 item 6: periodic eviction. Session expiry and the ring-buffer TTL
    // are otherwise only enforced lazily (on lookup / on push), so state owned
    // by idle or abandoned sessions is retained until process exit. Every SSE
    // deployment passes through router(), and `travsr serve` calls it from
    // inside its tokio runtime, so this is the construction point to spawn at.
    // The Handle guard keeps router() callable from non-async contexts (tests)
    // where tokio::spawn would panic.
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let state_maint = state.clone();
        handle.spawn(async move {
            let mut ticker = tokio::time::interval(MAINTENANCE_INTERVAL);
            loop {
                // The first tick completes immediately; both sweeps are cheap
                // no-ops on empty maps, so no special-casing is needed.
                ticker.tick().await;
                state_maint.session_store.evict_expired();
                sweep_idle_ring_buffers(&state_maint);
            }
        });
    }

    Router::new()
        .route("/sse", get(sse_handler))
        .route("/rpc", post(rpc_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        // Reject bodies larger than MAX_BODY_BYTES at middleware level before axum
        // buffers them — prevents multi-MB heap allocations ahead of the in-handler check.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(
            TraceLayer::new_for_http().make_span_with(|request: &axum::http::Request<_>| {
                tracing::info_span!(
                    "http_request",
                    method = %request.method(),
                    path = %request.uri().path(),
                )
            }),
        )
        .with_state(state)
}

// ── Auth helper ───────────────────────────────────────────────────────────────

/// Extract and verify the bearer token from request headers.
fn extract_bearer(
    headers: &HeaderMap,
    signing_keys: &[[u8; 32]],
) -> Result<VerifiedToken, AuthError> {
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .ok_or(AuthError::MissingHeader)?;

    let auth_str = auth_header
        .to_str()
        .map_err(|_| AuthError::BadHeaderFormat)?;

    let token = auth_str
        .strip_prefix("Bearer ")
        .ok_or(AuthError::BadHeaderFormat)?;

    verify_token(token, signing_keys)
}

/// Return a JSON-RPC 2.0 error response string.
fn jsonrpc_error(id: &str, code: i32, message: &str) -> String {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
    .to_string()
}

/// Return a JSON-RPC 2.0 success response string.
fn jsonrpc_ok(id: &str, result: serde_json::Value) -> String {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
    .to_string()
}

// ── Ring buffer helpers ───────────────────────────────────────────────────────

fn ring_buffer_push(
    state: &AppState,
    tenant_id: &str,
    session_id: Uuid,
    event_id: u64,
    payload: String,
) {
    let key = (tenant_id.to_string(), session_id);
    let mut buf = state.ring_buffers.entry(key).or_default();
    // TTL, entry-cap, and byte-budget eviction all happen inside push.
    buf.push((event_id, Instant::now(), payload));
}

fn ring_buffer_replay(
    state: &AppState,
    tenant_id: &str,
    session_id: Uuid,
    after_id: u64,
) -> Vec<(u64, String)> {
    let key = (tenant_id.to_string(), session_id);
    let now = Instant::now();
    match state.ring_buffers.get(&key) {
        None => vec![],
        Some(buf) => buf
            .entries
            .iter()
            .filter(|(id, ts, _)| *id > after_id && now.duration_since(*ts) <= RING_BUFFER_TTL)
            .map(|(id, _, payload)| (*id, payload.clone()))
            .collect(),
    }
}

/// Drop ring buffers whose newest entry is older than `RING_BUFFER_TTL` (#736 item 6).
///
/// The TTL is otherwise only enforced on push, so a session that stops
/// producing events keeps its last buffered window resident forever. Called by
/// the periodic maintenance task spawned in [`router`]. Anything this removes
/// was already unreplayable — every entry in the buffer is past the TTL — so no
/// live reconnect can lose data to the sweep.
fn sweep_idle_ring_buffers(state: &AppState) {
    let now = Instant::now();
    state.ring_buffers.retain(|_, buf| {
        buf.entries
            .back()
            .is_some_and(|(_, ts, _)| now.duration_since(*ts) <= RING_BUFFER_TTL)
    });
}

fn next_event_id(state: &AppState, session_id: Uuid) -> u64 {
    let mut counter = state.event_counters.entry(session_id).or_insert(0);
    *counter += 1;
    *counter
}

// ── GET /health ───────────────────────────────────────────────────────────────

async fn health_handler() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

// ── GET /metrics ──────────────────────────────────────────────────────────────

async fn metrics_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let vt = match extract_bearer(&headers, &state.signing_keys) {
        Ok(vt) => vt,
        Err(e) => {
            let status = auth_error_status(&e);
            return (status, e.to_string()).into_response();
        }
    };
    if vt.scope != TokenScope::Metrics {
        return (
            StatusCode::FORBIDDEN,
            "insufficient scope: metrics required",
        )
            .into_response();
    }

    let active_sessions = state.sessions.len();

    // Sum graph.db sizes across /data/tenants — or 0 if directory not accessible.
    let db_size_bytes: u64 = {
        let base = std::path::Path::new("/data/tenants");
        if base.is_dir() {
            let mut total = 0u64;
            for entry in state.tenant_repos.iter() {
                for db_path in entry.value().values() {
                    if let Ok(meta) = std::fs::metadata(db_path) {
                        total += meta.len();
                    }
                }
            }
            total
        } else {
            0
        }
    };

    let body = format!(
        "# HELP travsr_sse_active_sessions_total Number of active SSE sessions\n\
         # TYPE travsr_sse_active_sessions_total gauge\n\
         travsr_sse_active_sessions_total {active_sessions}\n\
         # HELP travsr_tenant_db_size_bytes Total size of all tenant graph databases in bytes\n\
         # TYPE travsr_tenant_db_size_bytes gauge\n\
         travsr_tenant_db_size_bytes {db_size_bytes}\n"
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

// ── GET /sse ──────────────────────────────────────────────────────────────────

async fn sse_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    // Authenticate — scope must be Mcp.
    let vt = match extract_bearer(&headers, &state.signing_keys) {
        Ok(vt) => vt,
        Err(e) => {
            let status = auth_error_status(&e);
            return (status, e.to_string()).into_response();
        }
    };
    if vt.scope != TokenScope::Mcp {
        return (StatusCode::FORBIDDEN, "insufficient scope: mcp required").into_response();
    }

    let tenant_id = vt.tenant_id.clone();
    let session_id = Uuid::new_v4();

    // Check for reconnect headers.
    let last_event_id: Option<u64> = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());
    let reconnect_session_id: Option<Uuid> = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok());

    // Create mpsc channel for this session.
    let (tx, rx) = mpsc::channel::<SseEvent>(64);

    let handle = SseSessionHandle {
        tenant_id: tenant_id.clone(),
        tx: tx.clone(),
        opened_at: Instant::now(),
    };
    state.sessions.insert(session_id, handle);

    // Keepalive task — cancels the token when the client disconnects (send error).
    let cancel = CancellationToken::new();
    let cancel_for_ka = cancel.clone();
    let tx_ka = tx.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_for_ka.cancelled() => break,
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(15)) => {
                    if tx_ka.send(SseEvent::Keepalive).await.is_err() {
                        // Receiver dropped — client disconnected. Signal cleanup.
                        cancel_for_ka.cancel();
                        break;
                    }
                }
            }
        }
    });

    // Build replay events if reconnecting.
    let replay_events: Vec<(u64, String)> =
        if let (Some(after_id), Some(old_session_id)) = (last_event_id, reconnect_session_id) {
            ring_buffer_replay(&state, &tenant_id, old_session_id, after_id)
        } else {
            vec![]
        };

    // Convert the mpsc receiver into an SSE stream.
    let state_for_stream = state.clone();
    let tenant_id_for_stream = tenant_id.clone();

    let initial_event = {
        let session_json = json!({"session_id": session_id.to_string()}).to_string();
        // session event: id 0, event: session
        Ok::<Event, std::convert::Infallible>(
            Event::default().event("session").id("0").data(session_json),
        )
    };

    // Build the replay stream.
    let replay_stream = tokio_stream::iter(replay_events.into_iter().map(|(id, payload)| {
        Ok::<Event, std::convert::Infallible>(Event::default().id(id.to_string()).data(payload))
    }));

    // Convert the live mpsc channel to a stream.
    let live_stream = {
        let state_s = state_for_stream.clone();
        let tenant_id_s = tenant_id_for_stream.clone();
        tokio_stream::StreamExt::map(ReceiverStream::new(rx), move |ev| match ev {
            SseEvent::JsonRpc(payload) => {
                let eid = next_event_id(&state_s, session_id);
                ring_buffer_push(&state_s, &tenant_id_s, session_id, eid, payload.clone());
                Ok::<Event, std::convert::Infallible>(
                    Event::default().id(eid.to_string()).data(payload),
                )
            }
            SseEvent::Keepalive => Ok(Event::default().comment("keepalive")),
            SseEvent::Close => Ok(Event::default().comment("close")),
        })
    };

    // Chain: initial session event → replayed events → live events.
    use tokio_stream::StreamExt as _;
    let full_stream = tokio_stream::once(initial_event)
        .chain(replay_stream)
        .chain(live_stream);

    // Spawn a cleanup task: when the mpsc tx is dropped (stream consumed / client
    // disconnected), the keepalive task will get a send error and stop. We also
    // remove the session from AppState when the CancellationToken fires.
    //
    // The cleanup task waits until the cancellation token is cancelled (which
    // happens when the keepalive task exits because all tx clones are dropped).
    // Since the SSE stream holds the last tx clone inside `live_stream`, when axum
    // drops the stream, tx is dropped → keepalive errors → cancel fires → cleanup runs.
    let state_cleanup = state.clone();
    let tenant_id_cleanup = tenant_id.clone();
    tokio::spawn(async move {
        cancel.cancelled().await;
        state_cleanup.sessions.remove(&session_id);
        state_cleanup.event_counters.remove(&session_id);
        state_cleanup
            .ring_buffers
            .remove(&(tenant_id_cleanup.clone(), session_id));
        tracing::debug!(
            session = %session_id,
            tenant = %tenant_id_cleanup,
            "SSE session cleaned up"
        );
    });

    axum::response::sse::Sse::new(full_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(tokio::time::Duration::from_secs(15))
                .text("keepalive"),
        )
        .into_response()
}

// ── POST /rpc ─────────────────────────────────────────────────────────────────

/// RAII guard for an `in_flight` dedup entry (#736 item 6).
///
/// The entry used to be removed manually on the success and error paths, which
/// missed the third exit: axum drops the handler future when the HTTP client
/// disconnects mid-await (during `spawn_blocking` or the SSE channel send), and
/// code after a dropped await point never runs. The leaked entry then rejected
/// that rpc id as a duplicate forever. `Drop` runs on every exit path — early
/// return, panic unwind, and future cancellation alike.
struct InFlightGuard {
    map: Arc<DashMap<(TenantId, String), ()>>,
    key: (TenantId, String),
}

impl InFlightGuard {
    /// Atomically insert `key`; `None` means the id is already in flight.
    ///
    /// Uses the entry API rather than contains_key-then-insert so two racing
    /// requests with the same id cannot both pass the dedup check.
    fn try_acquire(
        map: &Arc<DashMap<(TenantId, String), ()>>,
        key: (TenantId, String),
    ) -> Option<Self> {
        match map.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Occupied(_) => None,
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                slot.insert(());
                Some(Self {
                    map: Arc::clone(map),
                    key,
                })
            }
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.map.remove(&self.key);
    }
}

/// Maximum JSON body size accepted (512 KB).
const MAX_BODY_BYTES: usize = 512 * 1024;

/// Maximum JSON response size before chunking (512 KB).
const MAX_RESPONSE_BYTES: usize = 512 * 1024;

async fn rpc_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Authenticate — scope must be Mcp.
    let vt = match extract_bearer(&headers, &state.signing_keys) {
        Ok(vt) => vt,
        Err(e) => {
            let status = auth_error_status(&e);
            return (status, e.to_string()).into_response();
        }
    };
    if vt.scope != TokenScope::Mcp {
        return (StatusCode::FORBIDDEN, "insufficient scope: mcp required").into_response();
    }

    let tenant_id = vt.tenant_id.clone();

    // Body size check.
    if body.len() > MAX_BODY_BYTES {
        return (StatusCode::PAYLOAD_TOO_LARGE, "request body exceeds 512KB").into_response();
    }

    // Parse JSON-RPC 2.0.
    let req_value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")).into_response();
        }
    };

    // Validate JSON-RPC fields.
    if req_value.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
        return (StatusCode::BAD_REQUEST, "jsonrpc field must be \"2.0\"").into_response();
    }

    // id must be a UUID4 string.
    let id_str = match req_value.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return (StatusCode::BAD_REQUEST, "id field must be a UUID4 string").into_response();
        }
    };
    if Uuid::parse_str(id_str).is_err() {
        return (StatusCode::BAD_REQUEST, "id field must be a valid UUID").into_response();
    }
    let rpc_id = id_str.to_string();

    // Check in-flight dedup. The guard removes the entry when it drops — on
    // the normal return, on the no-session early return, and (critically) when
    // axum cancels this future because the client disconnected mid-dispatch.
    let in_flight_key = (tenant_id.clone(), rpc_id.clone());
    let Some(_in_flight_guard) = InFlightGuard::try_acquire(&state.in_flight, in_flight_key) else {
        return (
            StatusCode::BAD_REQUEST,
            "duplicate request id: already in flight",
        )
            .into_response();
    };

    // Find the most recently opened SSE session for this tenant.
    let session_entry = {
        state
            .sessions
            .iter()
            .filter(|e| e.value().tenant_id == tenant_id)
            .max_by_key(|e| e.value().opened_at)
            .map(|e| (*e.key(), e.value().tx.clone()))
    };

    let (_session_id, session_tx) = match session_entry {
        Some((id, tx)) => (id, tx),
        None => {
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                "no active SSE session for tenant",
            )
                .into_response();
            resp.headers_mut()
                .insert("retry-after", axum::http::HeaderValue::from_static("1"));
            return resp;
        }
    };

    // Get repos for this tenant.
    let repos: HashMap<String, PathBuf> = state
        .tenant_repos
        .get(&tenant_id)
        .map(|r| r.clone())
        .unwrap_or_default();

    // Dispatch tool call (same dispatch as server.rs global mode). Rerank and
    // other blocking work must not run on the async/tokio worker thread (RFC-021
    // §15 G5(a)); move the whole synchronous dispatch to a blocking pool.
    let response_payload = {
        // #736 item 8: take a dispatch permit before spawning onto the blocking
        // pool. `acquire_owned().await` is the backpressure point — excess RPCs
        // queue here instead of growing the blocking pool towards its
        // 512-thread ceiling. The permit MOVES INTO the closure so it is
        // released when the blocking work actually finishes, not when this
        // future is dropped: a cancelled handler leaves its spawn_blocking
        // task running to completion, and freeing the permit early would let
        // orphaned dispatches accumulate without bound.
        let permit = state
            .dispatch_permits
            .clone()
            .acquire_owned()
            .await
            .expect("dispatch semaphore is never closed");
        let rpc_id_for_dispatch = rpc_id.clone();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            dispatch_tool_call(&req_value, &rpc_id_for_dispatch, &repos)
        })
        .await
        {
            Ok(payload) => payload,
            Err(join_err) => {
                // A panic that previously unwound through rpc_handler now surfaces
                // as JoinError; answer with a JSON-RPC internal error instead of
                // dropping the request silently.
                tracing::error!(%join_err, "tool dispatch panicked");
                serde_json::json!({
                    "jsonrpc": "2.0", "id": rpc_id,
                    "error": { "code": -32603, "message": "internal error" }
                })
                .to_string()
            }
        }
    };

    // Send response(s) via SSE.
    let response_str = response_payload;
    let bytes = response_str.as_bytes();

    if bytes.len() <= MAX_RESPONSE_BYTES {
        // Single event — the live_stream map will call ring_buffer_push when it maps
        // the SseEvent::JsonRpc variant from the mpsc channel.
        let _ = session_tx.send(SseEvent::JsonRpc(response_str)).await;
    } else {
        // Chunk the response.
        let chunks: Vec<&[u8]> = bytes.chunks(MAX_RESPONSE_BYTES).collect();
        let total = chunks.len();
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_str = String::from_utf8_lossy(chunk).to_string();
            let envelope = json!({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "partial": true,
                    "part": i + 1,
                    "total": total,
                    "data": chunk_str
                }
            })
            .to_string();
            let _ = session_tx.send(SseEvent::JsonRpc(envelope)).await;
        }
    }

    // _in_flight_guard drops here, releasing the dedup entry.
    StatusCode::OK.into_response()
}

/// Dispatch a JSON-RPC 2.0 tool call and return the response string.
///
/// Mirrors the dispatch logic from server.rs global mode without touching that file.
fn dispatch_tool_call(
    req: &serde_json::Value,
    rpc_id: &str,
    repos: &HashMap<String, PathBuf>,
) -> String {
    use crate::protocol::{INVALID_PARAMS, METHOD_NOT_FOUND};
    use travsr_retrieval::OpenFilter;

    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let params = req
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    match method {
        "initialize" => jsonrpc_ok(
            rpc_id,
            json!({
                "protocolVersion": crate::PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": crate::SERVER_NAME, "version": crate::SERVER_VERSION }
            }),
        ),
        "tools/call" => {
            let tool_name = match params["name"].as_str() {
                Some(n) => n,
                None => {
                    return jsonrpc_error(rpc_id, INVALID_PARAMS, "missing tool name");
                }
            };
            let args = &params["arguments"];
            let repo_arg = args["repo"].as_str();

            let _span = tracing::info_span!("mcp.sse.tool_call", tool = tool_name, tenant = "sse")
                .entered();

            let text = match tool_name {
                "get_dependencies" => tools::get_dependencies_global(
                    repos,
                    args["file"].as_str().unwrap_or(""),
                    repo_arg,
                ),
                "get_callers" => tools::get_callers_global(
                    repos,
                    args["symbol"].as_str().unwrap_or(""),
                    args["path"].as_str().filter(|s| !s.is_empty()),
                    repo_arg,
                ),
                "get_blast_radius" => {
                    let mode = match args["analysis"].as_str().unwrap_or("tree-sitter") {
                        "semantic" => tools::AnalysisMode::Semantic,
                        _ => tools::AnalysisMode::TreeSitter,
                    };
                    tools::get_blast_radius_global(
                        repos,
                        args["file"].as_str().unwrap_or(""),
                        repo_arg,
                        mode,
                    )
                }
                "get_lang_status" => tools::get_lang_status_global(
                    repos,
                    args["file"].as_str().unwrap_or(""),
                    repo_arg,
                ),
                "search_symbol" => tools::search_symbol_global(
                    repos,
                    args["name"].as_str().unwrap_or(""),
                    repo_arg,
                    args["exact"].as_bool().unwrap_or(false),
                ),
                "get_repo_map" => tools::get_repo_map_global(repos, repo_arg),
                // TODO(RFC-008 / #197): replace OpenFilter with per-session RbacFilter
                // once token-scoped RBAC is implemented for the SSE path.
                "get_execution_path" => tools::get_execution_path_global(
                    repos,
                    args["source"].as_str().unwrap_or(""),
                    args["sink"].as_str().unwrap_or(""),
                    repo_arg,
                    &OpenFilter,
                ),
                other => {
                    return jsonrpc_error(
                        rpc_id,
                        INVALID_PARAMS,
                        &format!("unknown tool: {other}"),
                    );
                }
            };

            tracing::info!(
                tool = tool_name,
                tool_calls_total = 1u64,
                "mcp.sse.tool_call complete"
            );

            jsonrpc_ok(
                rpc_id,
                json!({ "content": [{ "type": "text", "text": text }] }),
            )
        }
        other => jsonrpc_error(
            rpc_id,
            METHOD_NOT_FOUND,
            &format!("method not found: {other}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, payload: &str) -> RingEntry {
        (id, Instant::now(), payload.to_string())
    }

    /// Sanity check the `total_bytes` invariant against a recount.
    fn recounted_bytes(buf: &RingBuffer) -> usize {
        buf.entries.iter().map(|(_, _, p)| p.len()).sum()
    }

    #[test]
    fn ring_buffer_evicts_oldest_when_byte_budget_exceeded() {
        let mut buf = RingBuffer::default();
        // Budget of 10 bytes, 4-byte payloads: the third push must evict the first.
        buf.push_bounded(entry(1, "aaaa"), 1000, RING_BUFFER_TTL, 10);
        buf.push_bounded(entry(2, "bbbb"), 1000, RING_BUFFER_TTL, 10);
        assert_eq!(buf.total_bytes, 8);
        buf.push_bounded(entry(3, "cccc"), 1000, RING_BUFFER_TTL, 10);
        assert_eq!(buf.entries.len(), 2, "oldest entry must be evicted");
        assert_eq!(buf.entries.front().unwrap().0, 2);
        assert_eq!(buf.entries.back().unwrap().0, 3);
        assert_eq!(buf.total_bytes, 8);
        assert_eq!(buf.total_bytes, recounted_bytes(&buf));
    }

    /// #737 review: the byte budget is a HARD bound. An oversized payload is
    /// refused outright (mirroring ParseCache and QueryCache) instead of
    /// draining the ring and then exceeding the budget by its own size.
    #[test]
    fn ring_buffer_oversized_payload_is_refused() {
        let mut buf = RingBuffer::default();
        buf.push_bounded(entry(1, "aaaa"), 1000, RING_BUFFER_TTL, 10);
        // 16 bytes > the 10-byte budget: refused, and the existing in-budget
        // entries survive untouched.
        buf.push_bounded(entry(2, "0123456789abcdef"), 1000, RING_BUFFER_TTL, 10);
        assert_eq!(buf.entries.len(), 1);
        assert_eq!(buf.entries.front().unwrap().0, 1, "entry 1 must survive");
        assert_eq!(buf.total_bytes, 4);
        assert!(buf.total_bytes <= 10, "budget must be a hard bound");
        // Normal pushes continue to work afterwards.
        buf.push_bounded(entry(3, "dddd"), 1000, RING_BUFFER_TTL, 10);
        assert_eq!(buf.entries.len(), 2);
        assert_eq!(buf.total_bytes, recounted_bytes(&buf));
    }

    #[test]
    fn ring_buffer_entry_cap_still_enforced() {
        let mut buf = RingBuffer::default();
        buf.push_bounded(entry(1, "a"), 2, RING_BUFFER_TTL, usize::MAX);
        buf.push_bounded(entry(2, "b"), 2, RING_BUFFER_TTL, usize::MAX);
        buf.push_bounded(entry(3, "c"), 2, RING_BUFFER_TTL, usize::MAX);
        assert_eq!(buf.entries.len(), 2, "entry cap must still apply");
        assert_eq!(buf.entries.front().unwrap().0, 2);
        assert_eq!(buf.total_bytes, recounted_bytes(&buf));
    }

    #[test]
    fn ring_buffer_ttl_eviction_still_applies_on_push() {
        let mut buf = RingBuffer::default();
        // A zero TTL means any measurable age expires the entry on the next push.
        buf.push_bounded(entry(1, "old"), 1000, Duration::ZERO, usize::MAX);
        std::thread::sleep(Duration::from_millis(5));
        buf.push_bounded(entry(2, "new"), 1000, Duration::ZERO, usize::MAX);
        assert_eq!(buf.entries.len(), 1, "expired entry must be evicted");
        assert_eq!(buf.entries.front().unwrap().0, 2);
        assert_eq!(buf.total_bytes, 3);
    }

    #[test]
    fn sweep_removes_only_buffers_whose_newest_entry_expired() {
        // An Instant far enough in the past to be beyond the ring TTL. On
        // platforms where Instant cannot go back that far (very early after
        // boot) there is nothing meaningful to test, so skip.
        let Some(stale) = Instant::now().checked_sub(RING_BUFFER_TTL + Duration::from_secs(1))
        else {
            return;
        };

        let state = AppState::new(DashMap::new(), vec![]);
        let idle_key = ("tenant-a".to_string(), Uuid::new_v4());
        let live_key = ("tenant-a".to_string(), Uuid::new_v4());

        let mut idle = RingBuffer::default();
        idle.push_bounded((1, stale, "old".into()), 1000, Duration::MAX, usize::MAX);
        state.ring_buffers.insert(idle_key.clone(), idle);

        let mut live = RingBuffer::default();
        live.push_bounded((1, stale, "old".into()), 1000, Duration::MAX, usize::MAX);
        live.push((2, Instant::now(), "fresh".into()));
        state.ring_buffers.insert(live_key.clone(), live);

        sweep_idle_ring_buffers(&state);

        assert!(
            !state.ring_buffers.contains_key(&idle_key),
            "buffer whose newest entry is past the TTL must be swept"
        );
        assert!(
            state.ring_buffers.contains_key(&live_key),
            "buffer with a fresh newest entry must survive the sweep"
        );
    }

    #[test]
    fn in_flight_guard_removes_entry_on_drop() {
        let map: Arc<DashMap<(TenantId, String), ()>> = Arc::new(DashMap::new());
        let key = ("tenant-a".to_string(), "rpc-1".to_string());

        let guard =
            InFlightGuard::try_acquire(&map, key.clone()).expect("first acquire must succeed");
        assert!(map.contains_key(&key), "entry must exist while guard lives");

        // A second acquire for the same key is the duplicate-request case.
        assert!(
            InFlightGuard::try_acquire(&map, key.clone()).is_none(),
            "duplicate id must be rejected while in flight"
        );
        // The failed acquire must not have disturbed the live entry.
        assert!(map.contains_key(&key));

        drop(guard);
        assert!(
            !map.contains_key(&key),
            "entry must be removed when the guard drops"
        );

        // The id is usable again after the guard is gone — this is exactly the
        // property the pre-guard code lost when the handler future was cancelled.
        assert!(InFlightGuard::try_acquire(&map, key).is_some());
    }

    #[test]
    fn in_flight_guards_for_different_keys_do_not_interfere() {
        let map: Arc<DashMap<(TenantId, String), ()>> = Arc::new(DashMap::new());
        let key_a = ("tenant-a".to_string(), "rpc-1".to_string());
        let key_b = ("tenant-b".to_string(), "rpc-1".to_string());

        let guard_a = InFlightGuard::try_acquire(&map, key_a.clone()).expect("acquire a");
        let guard_b = InFlightGuard::try_acquire(&map, key_b.clone()).expect("acquire b");

        drop(guard_a);
        assert!(!map.contains_key(&key_a));
        assert!(
            map.contains_key(&key_b),
            "dropping one guard must not remove another tenant's entry"
        );
        drop(guard_b);
        assert!(map.is_empty());
    }
}
