//! travsr-mcp — Model Context Protocol server.
//!
//! Exposes Travsr's graph retrieval tools to MCP clients (Claude Desktop,
//! VS Code, etc.) over stdio. Per CLAUDE.md principle 4, MCP is the *only*
//! external interface — no REST, no GraphQL.
//!
//! Framing: newline-delimited JSON-RPC 2.0 over stdin/stdout.
//! All diagnostics are written to stderr via `tracing`.

#![forbid(unsafe_code)]

pub mod auth;
mod observability;
mod protocol;
pub mod query;
mod rerank;
mod sanitize;
mod seed;
mod server;
pub use server::{tools_list as stdio_tools_list, tools_list_global as global_tools_list};
pub mod session;
pub mod sse;
mod tools;

// Re-exported for fuzz targets (fuzz/fuzz_targets/fuzz_mcp_parser.rs).
pub use auth::{fetch_signing_keys, is_valid_tenant_id};
pub use protocol::RpcRequest;
pub use session::{session_id_log_hash, Session, SessionId, SessionStore};
pub use sse::{router as sse_router, AppState};
// Re-exported for the `travsr refs` / `travsr pattern` CLI subcommands (#299),
// which run the same occurrence-store read as the MCP tools against a locally
// opened store. The rest of `tools` stays private (MCP-only surface).
pub use tools::{
    find_pattern, find_pattern_raw, find_references, find_references_structured, ResolvedSymbol,
    StructuredReferences,
};
// Re-exported for the `travsr graph` CLI subcommand.
pub use query::AMBIGUOUS_DISPLAY_LIMIT;
// #645 WS-B: the CLI `status` surface reuses this exact classifier so the CLI
// and MCP notes never disagree about an index/HEAD mismatch.
pub use tools::head_index_mismatch_note;
// The CLI surfaces the same call-graph completeness note as the MCP tools.
// Shared rather than reimplemented: the doc on `phase_b_degraded_note` already
// promised the two would never disagree, and until now only the MCP side used
// it, so a terminal user got an empty answer with no indication why.
pub use tools::phase_b_degraded_note;
// #448: exported solely so `travsr-daemon` can assert its own `SKIP_DIRS` still
// matches this copy. The dependency edge runs daemon → mcp, so the equality can
// only be checked from that side; without it a future edit to the daemon's list
// silently widens `find_pattern` past the graph's file set.
pub use tools::SKIP_DIRS;
// RFC-021 P5: model distribution. The daemon auto-fetches on warm; the
// `travsr rerank` CLI subcommand drives the same install path. The rest of
// `rerank` stays private (query-path internals).
pub use rerank::{
    install_model_blocking as install_rerank_model, model_installed as rerank_model_installed,
};

use std::path::Path;

use anyhow::Context as _;
use travsr_store::SqliteStore;

pub(crate) const PROTOCOL_VERSION: &str = "2024-11-05";
pub(crate) const SERVER_NAME: &str = "travsr";

/// The version reported in `initialize`'s `serverInfo`, over both stdio and SSE.
///
/// Reads this crate's own `Cargo.toml`, independently of `travsr-cli`'s. Nothing
/// at the Cargo level keeps the two in lockstep (`verify-version` in release.yml
/// only pins `travsr-cli` and npm's `package.json` to the tag); they agree today
/// only because both files say `1.0.0`. The only thing that would catch a release
/// that bumps one and forgets the other is `check_mcp_server_version_matches_cli`
/// in `travsrAutomation`, which runs on every PR via `run.py --binary` in `ci.yml`.
pub(crate) const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Start the MCP stdio server backed by the graph database at `db_path`.
///
/// Reads JSON-RPC 2.0 requests from stdin, dispatches them to the graph
/// tools, and writes responses to stdout. Runs until stdin is closed.
pub fn serve_stdio(db_path: &Path) -> anyhow::Result<()> {
    // `&mut`: the synonym_* tools perform writes (some via SQLite transactions,
    // which require `&mut Connection`). Read-only tools auto-reborrow `&mut`→`&`.
    let mut store = SqliteStore::open(db_path)
        .with_context(|| format!("opening graph database at {}", db_path.display()))?;
    // #645 WS-B: record the caller's checkout (the IDE launches the stdio server
    // in the workspace dir) so structural tools can flag an index built for a
    // different commit than the live HEAD.
    if let Ok(cwd) = std::env::current_dir() {
        tools::set_launch_cwd(cwd);
    }
    // Inject the embed KNN hook so get_context's semantic seed selection (Step 4)
    // works in standalone `travsr mcp --stdio` mode, not just daemon mode.
    inject_embed_hook(&mut store, db_path);
    // RFC-021: background-warm the reranker (idempotent, non-blocking) so the
    // first real query doesn't pay the model-load cost.
    rerank::warm_background();
    server::run(&mut store)
}

/// Start the global MCP stdio server backed by all repos in `~/.travsr/registry.json`.
///
/// The registry is re-read on every tool call so repos added via `travsr init`
/// after startup are picked up live without restarting the server.
pub fn serve_stdio_global() -> anyhow::Result<()> {
    // #645 WS-B: same checkout capture as `serve_stdio` (see there).
    if let Ok(cwd) = std::env::current_dir() {
        tools::set_launch_cwd(cwd);
    }
    rerank::warm_background();
    server::run_global()
}

/// The lazily-armed embed hooks for one repo, as installed on a `SqliteStore`.
///
/// Cloning is four `Arc` clones — the sidecar behind them is shared, never
/// respawned. Held by [`embed_hook_cache`] so global mode, which opens a fresh
/// read-only store per tool call, does not re-arm on every call.
#[derive(Clone)]
struct EmbedHooks {
    knn: travsr_store::EmbedKnnHook,
    doc: travsr_store::EmbedKnnHook,
    score: travsr_store::EmbedScoreHook,
    readiness: std::sync::Arc<travsr_store::EmbedReadiness>,
}

/// Armed embed hooks, keyed by graph.db path.
///
/// Single-repo mode arms once at startup and holds the store for the life of the
/// process. Global mode has no such store: it opens one per tool call and drops
/// it at the end, which would spawn, model-load and kill a sidecar every time —
/// measured at ~2.1 s of arm cost per `get_context` on a small repo. This keeps
/// the hooks (and so the sidecar) alive across calls.
///
/// One entry per repo the process has actually served a named `get_context`
/// for, capped at [`MAX_CACHED_EMBED_HOOKS`]. Only [`inject_embed_hook`]
/// inserts; the global fan-out reads through [`inject_cached_embed_hook`] so a
/// query with no `repo` argument can never spawn a sidecar per registered repo.
fn embed_hook_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, EmbedHooks>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, EmbedHooks>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Ceiling on cached entries. Each one pins a sidecar process holding a
/// 200-1400 MB ONNX model for the life of the daemon, and global mode puts no
/// bound on how many repos one process is asked to serve. Past the cap the
/// hooks are still built and used for the call, they are just not kept: those
/// repos pay the arm cost again next time instead of the process growing
/// without limit.
const MAX_CACHED_EMBED_HOOKS: usize = 8;

fn install_embed_hooks(store: &mut SqliteStore, hooks: EmbedHooks) {
    store.set_embed_readiness(hooks.readiness);
    store.set_embed_knn_hook(hooks.knn);
    store.set_embed_doc_knn_hook(hooks.doc);
    store.set_embed_score_hook(hooks.score);
}

/// Install already-armed hooks for `db_path`, if some earlier call armed them.
///
/// Never starts a sidecar, so it is safe on the global fan-out path: a repo that
/// a previous named-repo query warmed keeps its semantic lane for free, and one
/// that was never warmed stays lexical-only. Returns whether hooks were installed.
pub(crate) fn inject_cached_embed_hook(store: &mut SqliteStore, db_path: &Path) -> bool {
    // A poisoned cache degrades to "no hook" rather than bringing down the query.
    let hooks = embed_hook_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(db_path).cloned());
    match hooks {
        Some(hooks) => {
            install_embed_hooks(store, hooks);
            true
        }
        None => false,
    }
}

/// Wire the active embed backend's KNN hook into `store`, arming the sidecar on
/// first use for this `db_path` and reusing it on every later call.
///
/// The sidecar loads a 200–1400 MB ONNX model at startup — this takes 15–25 s
/// on a cold start. To keep `serve_stdio` (and the extension's `initialize`
/// handshake) responsive, sidecar startup is moved to a background thread.
///
/// A lazy meta-hook is injected immediately: it returns `Ok(vec![])` while the
/// sidecar is still loading, and delegates to the real KNN hook once it is ready.
/// This produces "embedding in progress" signals in `get_context` responses rather
/// than blank results or a 15–25 s connect stall.
///
/// The lookup and the insert happen under ONE lock. Releasing it around
/// `build_embed_hooks` let two concurrent SSE requests for the same cold repo
/// both miss, both spawn a sidecar, and the loser's process stay alive with its
/// model resident and nothing referencing it. `build_embed_hooks` only starts
/// the arming thread, so holding the lock across it does not serialise the
/// 15-25 s model load.
pub(crate) fn inject_embed_hook(store: &mut SqliteStore, db_path: &Path) {
    // A poisoned cache degrades to "no hook" rather than bringing down the query.
    let Ok(mut cache) = embed_hook_cache().lock() else {
        return;
    };
    let hooks = match cache.get(db_path) {
        Some(hooks) => hooks.clone(),
        None => {
            let Some(hooks) = build_embed_hooks(store, db_path) else {
                return;
            };
            if cache.len() < MAX_CACHED_EMBED_HOOKS {
                cache.insert(db_path.to_path_buf(), hooks.clone());
            } else {
                tracing::debug!(
                    cap = MAX_CACHED_EMBED_HOOKS,
                    "embed hook cache full, not retaining hooks for {}",
                    db_path.display()
                );
            }
            hooks
        }
    };
    drop(cache);
    install_embed_hooks(store, hooks);
}

/// Build (and start arming) the embed hooks for `db_path`, or `None` when this
/// repo has no embedding index, no installed backend binary, or no resolvable
/// backend — in which case no sidecar is started.
fn build_embed_hooks(store: &SqliteStore, db_path: &Path) -> Option<EmbedHooks> {
    use std::sync::{Arc, Mutex};

    use travsr_error::StoreError;
    use travsr_plugin_host::{
        active_backend_id, embed_backends, lookup_embed_backend, repo_backend_id, EmbedQueryHook,
        EmbedSupervisor,
    };
    use travsr_store::{EmbedKnnHook, EmbedReadiness, EmbedScoreHook};

    // Guard: no embed.db → nothing to query; skip to avoid spawning a sidecar
    // against a non-existent HNSW index.
    if !db_path.with_file_name("embed.db").exists() {
        return None;
    }

    let home = dirs::home_dir()?;
    // #481: the embedding backend is a per-repo setting; `~/.travsr/embed.toml`
    // is only the fallback. Reading the machine-global id here started the
    // sidecar with a different model than this repo's index was built with, so
    // `knn_hook`/`doc_knn_hook` were armed against a space that does not exist
    // for that model id. The doc hook then stayed `None` forever and the docs
    // section vanished with no error on every `get_context`, while `travsr ask`
    // (served by the daemon, which resolves the repo model) still rendered it.
    // Same resolution order as travsr-cli's embed paths.
    let backend = db_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(repo_backend_id)
        .or_else(active_backend_id)
        .as_deref()
        .and_then(lookup_embed_backend)
        .or_else(|| embed_backends().first())
        .cloned();
    let backend = backend?;

    // Mirror the daemon's guard (travsr-daemon: `embed model_id mismatch`): if
    // the index records a model, the sidecar's must match it or the hooks would
    // query a space built by a different encoder. Read before the init thread
    // because `store` is not `Send`.
    let stored_model = store.get_meta("current_embed_model").ok().flatten();

    let binary = home
        .join(".travsr")
        .join("bin")
        .join(backend.binary_filename());
    // Fast path: if the binary isn't installed there's nothing to do.
    if !binary.exists() {
        return None;
    }

    let db_path_bg = db_path.to_path_buf();
    let model_id_bg = backend.id.to_string();

    // Shared hook slot: None = sidecar starting; Some = ready to call.
    // `knn_hook()` clones the inner Arc<Mutex<EmbedSidecar>> so the sidecar
    // process stays alive after `EmbedSupervisor` itself is dropped on the thread.
    let slot: Arc<Mutex<Option<EmbedKnnHook>>> = Arc::new(Mutex::new(None));
    let slot_bg = Arc::clone(&slot);

    // #376 Phase 2: parallel slot for the doc-space hook, armed by the same
    // background thread right after the code hook. Stays `None` forever when
    // the sidecar predates doc-space support or the repo has no doc-chunk
    // nodes — `EmbedSupervisor::doc_knn_hook` returns `None` in both cases.
    let doc_slot: Arc<Mutex<Option<EmbedKnnHook>>> = Arc::new(Mutex::new(None));
    let doc_slot_bg = Arc::clone(&doc_slot);

    // RFC-019: parallel slot for the query-embedding hook, armed by the same
    // background init thread. None = sidecar still warming (no direct-cosine
    // scoring yet, so the classifier/ranker use the FTS-only path).
    let score_slot: Arc<Mutex<Option<EmbedQueryHook>>> = Arc::new(Mutex::new(None));
    let score_slot_bg = Arc::clone(&score_slot);

    // Arm-state shared with the query path: the background thread marks it ready
    // the instant the sidecar is warm, so the first get_context can briefly wait
    // (and the header reports `warming` honestly) instead of silently degrading
    // to lexical-only during the ~3 s startup window.
    let readiness = EmbedReadiness::new();
    let readiness_bg = Arc::clone(&readiness);

    std::thread::Builder::new()
        .name("embed-hook-init".into())
        .spawn(move || {
            // Arming lives in a closure so that every exit from it - model
            // mismatch, inactive supervisor, absent knn hook - still reaches the
            // single `mark_ready` below. Readiness means "arming has settled",
            // not "a hook exists": the meta-hooks further down are installed
            // unconditionally, so `has_embed` is true regardless, and a
            // readiness that never flips makes every `get_context` block for
            // `embed_arm_wait_ms()` before degrading to lexical-only.
            //
            // Returns whether a KNN hook was actually installed. A `false` is
            // not a transient state: nothing retries, so semantic search is off
            // for the life of this process and the query path has to be told,
            // or it reports `embeddings: on` over a lane that cannot run.
            let arm = || -> bool {
                let supervisor = EmbedSupervisor::try_start(&binary, &db_path_bg, &model_id_bg);
                if supervisor.is_active() {
                    if let Some(mid) = supervisor.model_id().map(str::to_string) {
                        if stored_model.as_deref().is_some_and(|stored| stored != mid) {
                            tracing::warn!(
                                stored_model = ?stored_model,
                                plugin_model = %mid,
                                "embed model_id mismatch, semantic search disabled. \
                                 Run `travsr embed reindex` to rebuild embeddings with the installed model."
                            );
                            return false;
                        }
                        if let Some(hook) = supervisor.knn_hook(mid.clone()) {
                            // Warm the sidecar (ONNX + HNSW load) BEFORE arming the
                            // hook, so the first real query never pays the cold-start
                            // cost that would trip the host's 600 ms KNN breaker and
                            // silently degrade to FTS. Blocking — we are already on a
                            // background init thread, so this delays nothing visible.
                            supervisor.prewarm();
                            // RFC-019: arm the query-embedding hook before the KNN slot
                            // so a query that observes `Some(knn)` also observes the
                            // score hook (never a half-armed state).
                            if let Some(qhook) = supervisor.embed_query_hook() {
                                if let Ok(mut guard) = score_slot_bg.lock() {
                                    *guard = Some(qhook);
                                }
                            }
                            if let Ok(mut guard) = slot_bg.lock() {
                                *guard = Some(hook);
                            }
                            // #376 Phase 2: arm the doc hook alongside the code hook.
                            // `None` when the sidecar predates doc-space support or
                            // has no doc-space index — `doc_slot` then simply stays
                            // empty forever, and `doc_lane_seeds` (seed.rs) treats an
                            // absent hook as "docs unavailable", not an error.
                            if let Some(doc_hook) = supervisor.doc_knn_hook(mid.clone()) {
                                if let Ok(mut guard) = doc_slot_bg.lock() {
                                    *guard = Some(doc_hook);
                                }
                            }
                            tracing::info!(
                                model_id = %mid,
                                "embed plugin active, Step 4 (semantic ANN) enabled"
                            );
                            return true;
                        }
                    }
                }
                // supervisor drops here; sidecar stays alive via the hook's Arc.
                false
            };
            if !arm() {
                // Recorded before `mark_ready` so a thread woken by it observes
                // both flags, and the query path can say `embeddings: disabled`
                // instead of `on` over a lane that will never answer.
                readiness_bg.mark_disabled();
            }
            // Signal arm-complete AFTER the slots are populated so any thread
            // woken by `mark_ready` sees whatever `arm` managed to install.
            readiness_bg.mark_ready();
        })
        .ok();

    // Inject the meta-hook immediately so `serve_stdio` can respond to
    // `initialize` without waiting for the sidecar to finish loading its model.
    let meta: EmbedKnnHook = Arc::new(move |query: &str, k: u32| {
        let guard = slot
            .lock()
            .map_err(|_| StoreError::Database("embed hook slot poisoned".into()))?;
        match guard.as_ref() {
            None => Ok(vec![]), // sidecar still warming up — return no seeds
            Some(hook) => hook(query, k),
        }
    });

    // #376 Phase 2: doc-space meta-hook, same lazy-slot shape as `meta` above.
    // While `doc_slot` is `None` (sidecar warming, unsupported, or no doc-chunk
    // nodes) this returns an empty vec — `doc_lane_seeds` then emits no docs
    // section, never a stall or an error.
    let meta_doc: EmbedKnnHook = Arc::new(move |query: &str, k: u32| {
        let guard = doc_slot
            .lock()
            .map_err(|_| StoreError::Database("embed doc hook slot poisoned".into()))?;
        match guard.as_ref() {
            None => Ok(vec![]),
            Some(hook) => hook(query, k),
        }
    });

    // RFC-019: meta direct-cosine oracle hook. Reads the lazily-armed query hook;
    // while the sidecar warms (slot None) it scores nothing, so the classifier and
    // ranker use the FTS-only path — identical to the pre-RFC behaviour.
    let embed_db = db_path.with_file_name("embed.db");
    let score_model = backend.id.to_string();
    let meta_score: EmbedScoreHook = Arc::new(move |query: &str, ids: &[travsr_core::NodeId]| {
        let guard = score_slot
            .lock()
            .map_err(|_| StoreError::Database("embed score slot poisoned".into()))?;
        let Some(qhook) = guard.as_ref() else {
            return Ok(vec![]); // sidecar still warming — no scoring yet
        };
        let blob = qhook(query)?;
        match travsr_store::decode_embedding(&blob) {
            Some(qv) => travsr_store::score_candidates(&qv, &embed_db, &score_model, ids),
            None => Ok(vec![]),
        }
    });
    tracing::info!("embed plugin hook installed (lazy, sidecar starting in background)");
    Some(EmbedHooks {
        knn: meta,
        doc: meta_doc,
        score: meta_score,
        readiness,
    })
}
