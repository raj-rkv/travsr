//! MCP stdio event loop.
//!
//! Reads newline-delimited JSON-RPC 2.0 messages from stdin, dispatches them,
//! and writes responses to stdout. All diagnostics go to stderr via tracing
//! so stdout stays clean for the MCP client.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use crate::protocol::{INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR};
use crate::tools;
use crate::{protocol::RpcRequest, PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION};
use travsr_retrieval::OpenFilter;
use travsr_store::{registry, SqliteStore};

pub fn run(store: &mut SqliteStore) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line_result in BufReader::new(stdin.lock()).lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(req) => handle_request(&mut *store, req),
            Err(e) => {
                tracing::debug!("JSON parse error: {e}");
                Some(error_response(
                    serde_json::Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                ))
            }
        };

        if let Some(resp) = response {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

/// Capabilities advertised in the `initialize` response. The `prompts`
/// capability is only declared when `mcp-sampling` is enabled, so that a
/// spec-compliant client can discover and call the `prompts/*` endpoints that
/// feature exposes; without it those endpoints are unreachable from a
/// conformant host.
fn server_capabilities() -> serde_json::Value {
    #[cfg(feature = "mcp-sampling")]
    {
        serde_json::json!({ "tools": {}, "prompts": {} })
    }
    #[cfg(not(feature = "mcp-sampling"))]
    {
        serde_json::json!({ "tools": {} })
    }
}

fn handle_request(store: &mut SqliteStore, req: RpcRequest) -> Option<String> {
    // JSON-RPC notifications have no id — must never receive a response.
    let id = match req.id {
        Some(id) => id,
        None => {
            tracing::debug!("notification received: {}", req.method);
            return None;
        }
    };

    let resp = match req.method.as_str() {
        "initialize" => ok_response(
            id,
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": server_capabilities(),
                "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION }
            }),
        ),

        "tools/list" => ok_response(id, tools_list()),

        "tools/call" => {
            let params = req.params.unwrap_or(serde_json::Value::Null);
            handle_tool_call(store, id, params)
        }

        #[cfg(feature = "mcp-sampling")]
        "prompts/list" => ok_response(id, prompts_list()),

        #[cfg(feature = "mcp-sampling")]
        "prompts/get" => {
            let name = req
                .params
                .as_ref()
                .and_then(|p| p["name"].as_str())
                .unwrap_or("");
            handle_prompts_get(id, name)
        }

        other => {
            tracing::debug!("unknown method: {other}");
            error_response(id, METHOD_NOT_FOUND, format!("method not found: {other}"))
        }
    };

    Some(resp)
}

fn handle_tool_call(
    store: &mut SqliteStore,
    id: serde_json::Value,
    params: serde_json::Value,
) -> String {
    let tool_name = match params["name"].as_str() {
        Some(n) => n,
        None => return error_response(id, INVALID_PARAMS, "missing tool name".into()),
    };
    let args = &params["arguments"];

    // The JSON-RPC id is already unique per in-flight request, so it is the
    // correlation id — no separate one to generate, and it is the same value
    // the client sees in the response. Every event emitted while serving this
    // call inherits it, which is what makes "show me everything that happened
    // for that one slow get_context" an answerable question.
    let _span = tracing::info_span!("mcp.tool_call", tool = tool_name, req = %id).entered();
    let started = std::time::Instant::now();
    let text = match tool_name {
        "get_dependencies" => {
            let file = args["file"].as_str().unwrap_or("");
            let transitive = args["transitive"].as_bool().unwrap_or(false);
            let depth = args["depth"].as_u64().unwrap_or(3).clamp(1, 10) as u32;
            tools::get_dependencies(store, file, transitive, depth)
        }
        "get_callers" => {
            let symbol = args["symbol"].as_str().unwrap_or("");
            let path = args["path"].as_str().filter(|s| !s.is_empty());
            tools::get_callers(store, symbol, path)
        }
        "find_references" => {
            let symbol = args["symbol"].as_str().unwrap_or("");
            let path = args["path"].as_str().filter(|s| !s.is_empty());
            tools::find_references(store, symbol, path)
        }
        "find_pattern" => {
            let pattern = args["pattern"].as_str().unwrap_or("");
            let scope = args["scope"].as_str().filter(|s| !s.is_empty());
            let fixed = args["fixed"].as_bool().unwrap_or(false);
            tools::find_pattern(store, pattern, scope, fixed)
        }
        "get_blast_radius" => {
            let file = args["file"].as_str().unwrap_or("");
            let mode = match args["analysis"].as_str().unwrap_or("tree-sitter") {
                "semantic" => tools::AnalysisMode::Semantic,
                _ => tools::AnalysisMode::TreeSitter,
            };
            tools::get_blast_radius(store, file, mode)
        }
        "get_lang_status" => {
            let file = args["file"].as_str().unwrap_or("");
            tools::get_lang_status(store, file)
        }
        "search_symbol" => {
            let name = args["name"].as_str().unwrap_or("");
            let exact = args["exact"].as_bool().unwrap_or(false);
            tools::search_symbol(store, name, exact)
        }
        "get_repo_map" => tools::get_repo_map(store),
        "get_graph_stats" => tools::get_graph_stats(store),
        "get_execution_path" => {
            let source = args["source"].as_str().unwrap_or("");
            let sink = args["sink"].as_str().unwrap_or("");
            tools::get_execution_path(store, source, sink)
        }
        "get_context" => {
            let query = args["query"].as_str().unwrap_or("");
            // McpClient sends Record<string,string> so numeric args arrive as JSON strings.
            // Try numeric first (direct callers) then string fallback (VS Code extension path).
            let token_budget = args["token_budget"]
                .as_u64()
                .or_else(|| {
                    args["token_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(4096) as usize;
            let include_snippets = args["include_snippets"].as_bool().unwrap_or(false);
            let snippet_budget = args["snippet_budget"]
                .as_u64()
                .or_else(|| {
                    args["snippet_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .map(|v| v as usize);
            // Use the embed-aware path when the daemon has injected a KNN hook
            // (i.e. `travsr embed init` + `travsr embed reindex` have been run).
            // Falls back to fuzzy text search automatically when the hook is absent.
            match store.embed_knn_fn() {
                Some(knn) => tools::get_context_embed(
                    store,
                    query,
                    token_budget,
                    include_snippets,
                    snippet_budget,
                    &knn,
                ),
                None => {
                    tools::get_context(store, query, token_budget, include_snippets, snippet_budget)
                }
            }
        }
        // DIAGNOSTIC (#462 Phase 0): raw embed-KNN ranking. Hidden — not in
        // tools/list; call by name. No shipping path depends on it.
        "embed_knn_probe" => {
            let query = args["query"].as_str().unwrap_or("");
            let k = args["k"]
                .as_u64()
                .or_else(|| args["k"].as_str().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(200) as u32;
            tools::embed_knn_probe(store, query, k)
        }
        // DIAGNOSTIC (#462 WS0): per-query seed-cascade trace (GATE vs DROP-@-stage).
        // Hidden — not in tools/list; call by name. No shipping path depends on it.
        "seed_trace" => {
            let query = args["query"].as_str().unwrap_or("");
            tools::seed_trace(store, query)
        }
        "get_graph_json" => {
            let query = args["query"].as_str().unwrap_or("");
            let direction = args["direction"].as_str().unwrap_or("both");
            // McpClient sends Record<string,string> so depth arrives as a JSON
            // string ("3"), not a number. Try numeric first for direct callers,
            // then fall back to string parsing for the VS Code extension path.
            let depth = args["depth"]
                .as_u64()
                .or_else(|| args["depth"].as_str().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(2)
                .clamp(1, 4) as u8;
            let kind_filter = args["kind_filter"].as_str().unwrap_or("");
            // #318 O6: optional token budget (additive arg). 0 = unlimited,
            // preserving pre-#318 behaviour when the arg is absent.
            let token_budget = args["token_budget"]
                .as_u64()
                .or_else(|| {
                    args["token_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(0) as usize;
            // #319 P3: LOD repo-map overview mode + package drill path_prefix.
            let mode = args["mode"].as_str().unwrap_or("");
            let path_prefix = args["path_prefix"].as_str().unwrap_or("");
            // RFC-027 section 10: optional, and absent means "everything", so an
            // existing caller keeps seeing the fresher graph unchanged.
            let provenance = args["provenance"].as_str().unwrap_or("");
            tools::get_graph_json(
                store,
                &tools::GraphJsonParams {
                    query,
                    direction,
                    depth,
                    kind_filter,
                    token_budget,
                    mode,
                    path_prefix,
                    provenance,
                },
            )
        }
        // RFC-012 A2 F1: dynamic synonym management. Single-repo (stdio) only —
        // see `handle_tool_call_global` for the multi-repo rejection.
        "synonym_add" => tools::synonym_add(
            store,
            args["term"].as_str().unwrap_or(""),
            args["alias"].as_str().unwrap_or(""),
        ),
        "synonym_set" => tools::synonym_set(
            store,
            args["term"].as_str().unwrap_or(""),
            args["aliases"].as_str().unwrap_or(""),
        ),
        "synonym_remove" => tools::synonym_remove(
            store,
            args["term"].as_str().unwrap_or(""),
            args["alias"].as_str().unwrap_or(""),
        ),
        "synonym_remove_term" => {
            tools::synonym_remove_term(store, args["term"].as_str().unwrap_or(""))
        }
        "synonym_list" => tools::synonym_list(store),
        "synonym_reset" => tools::synonym_reset(store),
        // VSCODE-247: global-registry management for the "Registered repos" webview.
        "repos_list" => tools::repos_list(),
        "repos_prune" => tools::repos_prune(),
        "repos_remove" => tools::repos_remove(args["name"].as_str().unwrap_or("")),
        "repo_languages" => tools::repo_languages(store),
        "get_snippets" => {
            let symbols = args["symbols"].as_str().unwrap_or("");
            // McpClient (VS Code) sends args as JSON strings, so token_budget
            // arrives as "6000"; try numeric first, then string.
            let token_budget = args["token_budget"]
                .as_u64()
                .or_else(|| {
                    args["token_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(tools::SNIPPET_DEFAULT_BUDGET as u64)
                as usize;
            let mode = tools::SnippetMode::from_arg(args["mode"].as_str().unwrap_or("auto"));
            tools::get_snippets(store, symbols, token_budget, mode)
        }
        // #636: read-only observability tools.
        "get_index_status" => crate::observability::get_index_status(store),
        "get_daemon_logs" => {
            let tail = args["tail"]
                .as_u64()
                .or_else(|| args["tail"].as_str().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(crate::observability::DEFAULT_TAIL as u64)
                as usize;
            let level = args["level"].as_str().unwrap_or("info");
            crate::observability::get_daemon_logs(store, tail, level)
        }
        "get_graph_health" => crate::observability::get_graph_health(store),
        other => {
            return error_response(id, INVALID_PARAMS, format!("unknown tool: {other}"));
        }
    };

    // tool_calls_total=1 is a log-based counter field for tracing subscribers.
    // TODO(travsr-060): replace with otel Counter metric for proper OTLP aggregation.
    tracing::info!(
        event = "mcp.tool_call.served",
        tool = tool_name,
        elapsed_ms = started.elapsed().as_millis(),
        tool_calls_total = 1u64,
        "mcp.tool_call complete"
    );
    ok_response(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": text }] }),
    )
}

/// The single-repo `tools/list` payload, served by `travsr mcp --stdio`.
///
/// Public so consumers that generate agent guidance for this server (see
/// `travsr connect`) can pin their tool names against what is actually served
/// instead of hand-maintaining a mirror that drifts on the next rename.
pub fn tools_list() -> serde_json::Value {
    serde_json::json!({
        "_schemaVersion": "1.1.0",
        "tools": [
            {
                "name": "get_dependencies",
                "description": "Return the files and modules that a given file imports. Set transitive=true to follow `depends` edges recursively up to `depth` hops.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to look up dependencies for" },
                        "transitive": { "type": "boolean", "description": "Follow dependency edges recursively. Default: false (direct imports only)." },
                        "depth": { "type": "integer", "minimum": 1, "maximum": 10, "description": "Max hops when transitive=true. Default: 3." }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_callers",
                "description": "Return all graph nodes that have an incoming edge to the given symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol name to find callers of (partial match supported)" },
                        "path": { "type": "string", "description": "Optional path hint to scope an overloaded name to a file or directory: a filename, a relative path, a directory prefix, or a path fragment (e.g. ppr.rs, src/ppr.rs, crates/travsr-retrieval, retrieval)" }
                    },
                    "required": ["symbol"],
                    "additionalProperties": false
                }
            },
            {
                "name": "find_references",
                "description": "Enumerate every use site (incl. inline expressions, assignments, and type references) of a symbol as path:line. Works across every indexed language via the occurrence store; falls back to caller definitions when a language has no occurrence lines yet.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol to enumerate references of (bare name or full signature)" },
                        "path": { "type": "string", "description": "Optional path hint to scope an overloaded name to a file or directory: a filename, a relative path, a directory prefix, or a path fragment (e.g. ppr.rs, src/ppr.rs, crates/travsr-retrieval, retrieval)" }
                    },
                    "required": ["symbol"],
                    "additionalProperties": false
                }
            },
            {
                "name": "find_pattern",
                "description": "Textual search (git grep) over the repo's tracked and untracked text files, returning path:line:col: text. Not limited to the files the graph indexes: any file git shows as text can match, with known-binary formats and ignored paths removed. Pattern is a POSIX extended regular expression (ERE); set `fixed: true` for a literal search. Optionally scope to a path prefix or to files-importing(<symbol>).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "POSIX ERE pattern to search for (use fixed:true for a literal string)" },
                        "scope": { "type": "string", "description": "Optional scope: a repo-relative path prefix, or files-importing(<symbol>)" },
                        "fixed": { "type": "boolean", "description": "Treat pattern as a literal string instead of a regular expression" }
                    },
                    "required": ["pattern"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_blast_radius",
                "description": "Return the set of files transitively affected if the given file changes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to compute blast radius for" },
                        "analysis": { "type": "string", "enum": ["tree-sitter", "semantic"], "description": "Edge mode: 'tree-sitter' (default, structural) or 'semantic' (call/reference edges; needs full cross-file analysis to be built)." }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_lang_status",
                "description": "Return whether semantic (full cross-file) analysis is available for the language of the given file, and an install hint if not. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to detect language for" }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "search_symbol",
                "description": "Find symbol definitions matching a name across the indexed graph. Accepts exact symbol names, partial matches, and natural-language queries (e.g. 'auth session validation', 'mcp dispatch tool call'). Query normalisation is deterministic: no model or API key required.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Symbol name or natural-language query (1-200 chars). Partial and NL queries are supported." },
                        "exact": { "type": "boolean", "description": "If true, drops loose substring matches and keeps only exact, word-boundary, and prefix matches. Use for short or common names to cut noise." }
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_repo_map",
                "description": "Return a structural overview of the indexed repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "get_execution_path",
                "description": "Find the lowest-cost path from a source symbol to a sink symbol through the code graph. Answers explicitly when the symbols do not resolve or resolve but are disconnected ('no path found'), so an empty-looking result is never ambiguous.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string", "description": "Source symbol name (partial match supported)" },
                        "sink": { "type": "string", "description": "Sink symbol name (partial match supported)" }
                    },
                    "required": ["source", "sink"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_stats",
                "description": "Return accurate node and edge counts from the indexed graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "get_context",
                "description": "Retrieve the most relevant context for a query within a token budget. Accepts symbol names and natural-language queries (e.g. 'where is the auth session validated?'). Natural-language queries are translated to search terms deterministically, no model or API key required. Set include_snippets=true to get actual source code inline alongside the structural metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Symbol name or natural-language query (1-200 chars)" },
                        "token_budget": { "type": "integer", "description": "Hard token budget (100-32000). Defaults to 4096 if omitted." },
                        "include_snippets": { "type": "boolean", "description": "When true, appends the actual source body of each selected symbol (kind-aware, docblock-stripped). Defaults to false." },
                        "snippet_budget": { "type": "integer", "description": "Optional separate token budget for the appended snippets. When omitted, snippets share the main token_budget (best-effort, stops before overflow). When provided, snippet retrieval uses this independent ceiling and does not affect node selection." }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_json",
                "description": "Return a subgraph around a symbol as structured JSON nodes and edges for graph renderers. Pass mode='overview' with no query for the repository overview tile layout.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Symbol name (1-200 chars). Resolved exact-first: full signature (e.g. 'fn:charge'), then exact bare name; substring matching is only a fallback when neither matches, so prefer the exact name for a precise single-root graph. May be empty when kind_filter is 'file' or mode is 'overview'." },
                        "direction": { "type": "string", "enum": ["deps", "callers", "both"], "description": "Edge direction. Default: both" },
                        "depth": { "type": "integer", "minimum": 1, "maximum": 4, "description": "BFS depth. Default: 2" },
                        "kind_filter": { "type": "string", "enum": ["file", ""], "description": "Restrict nodes to a specific kind. 'file' returns only file nodes and imports edges (project module map). Default: empty (all kinds)." },
                        "provenance": { "type": "string", "enum": ["", "ratified", "tree-sitter", "lsif", "scip", "live"], "description": "Restrict edges by how they were derived. Default empty returns everything, including 'live' edges resolved from uncommitted edits and not yet ratified. Use 'ratified' to exclude those and see only what the commit-gated pipeline has confirmed. Every edge in the response carries its own 'provenance' field." },
                        "token_budget": { "type": "integer", "description": "Cap the payload to roughly this many tokens (0 or omitted = unlimited). Truncation is reported via truncated_by_budget." },
                        "mode": { "type": "string", "enum": ["", "overview"], "description": "'overview' returns directory-level component tiles (each with file_count and dependents) plus cross-component dependency edges from the resolved graph, ranked by how depended-upon each component is. Combine with path_prefix to drill into a component." },
                        "path_prefix": { "type": "string", "description": "When mode='overview', scope to files under this path prefix (e.g. 'src/components/'). Returns file nodes inside the prefix plus external package nodes for cross-boundary dependencies." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_list",
                "description": "List all active query-synonym pairs (term => alias) from the per-repo dynamic synonym table. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_add",
                "description": "Add one alias for a term to the dynamic synonym table (200-row cap). Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": { "type": "string", "description": "The query term to expand" },
                        "alias": { "type": "string", "description": "An alias the term should also match" }
                    },
                    "required": ["term", "alias"],
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_set",
                "description": "Replace ALL aliases for a term atomically. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": { "type": "string", "description": "The query term to expand" },
                        "aliases": { "type": "string", "description": "Comma-separated alias list. Replaces all existing aliases for the term." }
                    },
                    "required": ["term", "aliases"],
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_remove",
                "description": "Remove a single (term, alias) synonym pair. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": { "type": "string", "description": "The query term" },
                        "alias": { "type": "string", "description": "The alias to remove from the term" }
                    },
                    "required": ["term", "alias"],
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_remove_term",
                "description": "Remove ALL aliases for a term. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": { "type": "string", "description": "The term whose aliases should all be removed" }
                    },
                    "required": ["term"],
                    "additionalProperties": false
                }
            },
            {
                "name": "synonym_reset",
                "description": "Reset the dynamic synonym table to the built-in static defaults. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "repos_list",
                "description": "List all globally registered repos as TSV: name, db_path, exists. Call this first to discover available repo names before supplying the `repo` parameter on other tools.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "repos_prune",
                "description": "Remove registry entries whose graph.db no longer exists. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "repos_remove",
                "description": "Remove a single repo from the registry by name. Stdio (single-repo) sessions only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Registry key (repo name) to remove" }
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            },
            {
                "name": "repo_languages",
                "description": "Return per-language node counts indexed in this repo as TSV (language, count), sorted by count descending.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "get_snippets",
                "description": "Return tailored code snippets for one or more symbols by name. Accepts the symbol names returned by get_context, get_callers, and search_symbol. Kind-aware extraction (mode='auto'): functions/methods → up to 40 lines; classes/structs/impls → up to 15 lines (header + fields only); interfaces/traits/enums → up to 60 lines; overflowing definitions fall back to a condensed structural outline. mode='full' returns the complete definition with no line cap; mode='skeleton' returns the structural outline only. Leading docblocks are stripped. Respects a token budget; symbols are included in request order until the budget is reached (the first symbol is always included). Use this after any graph-navigation tool to read the actual code without opening files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "string",
                            "description": "Newline- or comma-separated list of symbol names (e.g. 'PaymentService.charge\\nMAX_RETRIES'). Partial matches accepted, the closest matching non-file symbol is used for each name."
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Hard token cap across all returned snippets. Default: 2000. Higher values return more symbols."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "full", "skeleton"],
                            "description": "Representation per symbol. 'auto' (default): raw body if within the kind line-cap, else a condensed structural outline. 'full': complete definition, no line cap. 'skeleton': structural outline only (signatures, no bodies)."
                        },
                        "repo": {
                            "type": "string",
                            "description": "Restrict to a specific registered repo by name (global / multi-repo mode only)."
                        }
                    },
                    "required": ["symbols"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_index_status",
                "description": "Return index freshness and completeness: schema version, indexed vs HEAD commit staleness, node/edge counts, structural and semantic analysis state (including per-language failed/unavailable/done), and semantic (embeddings/rerank) readiness. Read-only. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_daemon_logs",
                "description": "Return recent daemon log entries (parsed, severity-filtered, and sanitized: home paths and credential-shaped substrings are best-effort redacted; do not assume logs are guaranteed secret-free). Read-only; never creates or writes any file. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tail": { "type": "integer", "minimum": 1, "maximum": 500, "description": "Number of most-recent log lines to consider (default 50, max 500)." },
                        "level": { "type": "string", "enum": ["error", "warn", "info", "debug"], "description": "Minimum severity to include (default 'info'); more severe levels are always included." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_health",
                "description": "Return a read-only graph integrity report: ghost paths (indexed but deleted on disk), orphan edges, and lexical-index parity. Never mutates; this is the report half of `travsr fsck`, not `--fix`. O(tracked file count); slower than most tools. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        ]
    })
}

fn ok_response(id: serde_json::Value, result: serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
}

fn error_response(id: serde_json::Value, code: i32, message: String) -> String {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
    .to_string()
}

// ── Global mode (all registered repos) ───────────────────────────────────────

/// Run the MCP stdio server in global mode.
///
/// Serves all repos registered in `~/.travsr/registry.json`. The registry is
/// re-read on every `tools/call` so repos added via `travsr init` after startup
/// are picked up live without restarting the server.
pub fn run_global() -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line_result in BufReader::new(stdin.lock()).lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(req) => handle_request_global(req),
            Err(e) => {
                tracing::debug!("JSON parse error: {e}");
                Some(error_response(
                    serde_json::Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                ))
            }
        };

        if let Some(resp) = response {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

fn handle_request_global(req: RpcRequest) -> Option<String> {
    let id = match req.id {
        Some(id) => id,
        None => {
            tracing::debug!("notification received: {}", req.method);
            return None;
        }
    };

    let resp = match req.method.as_str() {
        "initialize" => ok_response(
            id,
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": server_capabilities(),
                "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION }
            }),
        ),
        "tools/list" => ok_response(id, tools_list_global()),
        "tools/call" => {
            let repos = registry::all_repos().unwrap_or_default();
            let params = req.params.unwrap_or(serde_json::Value::Null);
            handle_tool_call_global(&repos, id, params)
        }

        #[cfg(feature = "mcp-sampling")]
        "prompts/list" => ok_response(id, prompts_list()),

        #[cfg(feature = "mcp-sampling")]
        "prompts/get" => {
            let name = req
                .params
                .as_ref()
                .and_then(|p| p["name"].as_str())
                .unwrap_or("");
            handle_prompts_get(id, name)
        }

        other => {
            tracing::debug!("unknown method: {other}");
            error_response(id, METHOD_NOT_FOUND, format!("method not found: {other}"))
        }
    };

    Some(resp)
}

fn handle_tool_call_global(
    repos: &HashMap<String, PathBuf>,
    id: serde_json::Value,
    params: serde_json::Value,
) -> String {
    let tool_name = match params["name"].as_str() {
        Some(n) => n,
        None => return error_response(id, INVALID_PARAMS, "missing tool name".into()),
    };
    let args = &params["arguments"];
    let repo_arg = args["repo"].as_str();

    // `repo` is what makes a shared global log readable: one process serves
    // every registered repo, so without it a line cannot be attributed. Carried
    // on the span rather than added at each call site, so events emitted deeper
    // in retrieval inherit it too.
    let _span = tracing::info_span!(
        "mcp.tool_call",
        tool = tool_name,
        req = %id,
        repo = repo_arg.unwrap_or("*"),
        global = true
    )
    .entered();
    let text = match tool_name {
        "get_dependencies" => {
            tools::get_dependencies_global(repos, args["file"].as_str().unwrap_or(""), repo_arg)
        }
        "get_callers" => tools::get_callers_global(
            repos,
            args["symbol"].as_str().unwrap_or(""),
            args["path"].as_str().filter(|s| !s.is_empty()),
            repo_arg,
        ),
        "find_references" => tools::find_references_global(
            repos,
            args["symbol"].as_str().unwrap_or(""),
            args["path"].as_str().filter(|s| !s.is_empty()),
            repo_arg,
        ),
        "find_pattern" => tools::find_pattern_global(
            repos,
            args["pattern"].as_str().unwrap_or(""),
            args["scope"].as_str().filter(|s| !s.is_empty()),
            repo_arg,
            args["fixed"].as_bool().unwrap_or(false),
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
        "get_lang_status" => {
            tools::get_lang_status_global(repos, args["file"].as_str().unwrap_or(""), repo_arg)
        }
        "search_symbol" => {
            let name = args["name"].as_str().unwrap_or("");
            let exact = args["exact"].as_bool().unwrap_or(false);
            tools::search_symbol_global(repos, name, repo_arg, exact)
        }
        "get_repo_map" => tools::get_repo_map_global(repos, repo_arg),
        "get_graph_stats" => tools::get_graph_stats_global(repos, repo_arg),
        "get_execution_path" => {
            let source = args["source"].as_str().unwrap_or("");
            let sink = args["sink"].as_str().unwrap_or("");
            tools::get_execution_path_global(repos, source, sink, repo_arg, &OpenFilter)
        }
        "get_context" => {
            let query = args["query"].as_str().unwrap_or("");
            let token_budget = args["token_budget"]
                .as_u64()
                .or_else(|| {
                    args["token_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(4096) as usize;
            let include_snippets = args["include_snippets"].as_bool().unwrap_or(false);
            let snippet_budget = args["snippet_budget"]
                .as_u64()
                .or_else(|| {
                    args["snippet_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .map(|v| v as usize);
            tools::get_context_global(
                repos,
                query,
                token_budget,
                repo_arg,
                include_snippets,
                snippet_budget,
            )
        }
        "get_graph_json" => {
            let query = args["query"].as_str().unwrap_or("");
            let direction = args["direction"].as_str().unwrap_or("both");
            let depth = args["depth"]
                .as_u64()
                .or_else(|| args["depth"].as_str().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(2)
                .clamp(1, 4) as u8;
            let kind_filter = args["kind_filter"].as_str().unwrap_or("");
            let mode = args["mode"].as_str().unwrap_or("");
            let path_prefix = args["path_prefix"].as_str().unwrap_or("");
            // RFC-027 section 10: optional, and absent means "everything", so an
            // existing caller keeps seeing the fresher graph unchanged.
            let provenance = args["provenance"].as_str().unwrap_or("");
            tools::get_graph_json_global(
                repos,
                repo_arg,
                &tools::GraphJsonParams {
                    query,
                    direction,
                    depth,
                    kind_filter,
                    token_budget: 0,
                    mode,
                    path_prefix,
                    provenance,
                },
            )
        }
        // Synonym tools mutate a single repo's table; ambiguous across the global
        // registry. Reject cleanly rather than silently no-op or fall through to
        // "unknown tool".
        "get_snippets" => {
            let symbols = args["symbols"].as_str().unwrap_or("");
            let token_budget = args["token_budget"]
                .as_u64()
                .or_else(|| {
                    args["token_budget"]
                        .as_str()
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(tools::SNIPPET_DEFAULT_BUDGET as u64)
                as usize;
            let mode = tools::SnippetMode::from_arg(args["mode"].as_str().unwrap_or("auto"));
            tools::get_snippets_global(repos, symbols, token_budget, repo_arg, mode)
        }
        "synonym_add"
        | "synonym_set"
        | "synonym_remove"
        | "synonym_remove_term"
        | "synonym_list"
        | "synonym_reset" => {
            return error_response(
                id,
                INVALID_PARAMS,
                "synonym tools require a single-repo (stdio) session".into(),
            );
        }
        // repos_list is read-only: safe to expose in global mode so agents can
        // discover which repos are indexed before supplying the `repo` param.
        "repos_list" => tools::repos_list(),
        "repos_prune" | "repos_remove" => {
            return error_response(
                id,
                INVALID_PARAMS,
                "repos_prune and repos_remove require a single-repo (stdio) session".into(),
            );
        }
        // #636: read-only observability tools. Never aggregate across repos:
        // each resolves to exactly one repo (`repo_arg`, or the sole live
        // registry entry) or returns an error payload naming `repos_list`.
        "get_index_status" => crate::observability::get_index_status_global(repos, repo_arg),
        "get_daemon_logs" => {
            let tail = args["tail"]
                .as_u64()
                .or_else(|| args["tail"].as_str().and_then(|s| s.parse::<u64>().ok()))
                .unwrap_or(crate::observability::DEFAULT_TAIL as u64)
                as usize;
            let level = args["level"].as_str().unwrap_or("info");
            crate::observability::get_daemon_logs_global(repos, tail, level, repo_arg)
        }
        "get_graph_health" => crate::observability::get_graph_health_global(repos, repo_arg),
        other => return error_response(id, INVALID_PARAMS, format!("unknown tool: {other}")),
    };

    // tool_calls_total=1 is a log-based counter field for tracing subscribers.
    // TODO(travsr-060): replace with otel Counter metric for proper OTLP aggregation.
    tracing::info!(
        tool = tool_name,
        tool_calls_total = 1u64,
        "mcp.tool_call complete"
    );
    ok_response(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": text }] }),
    )
}

/// The multi-repo `tools/list` payload, served by `travsr mcp --global`.
/// Public for the same reason as [`tools_list`].
pub fn tools_list_global() -> serde_json::Value {
    serde_json::json!({
        "_schemaVersion": "1.1.0",
        "tools": [
            {
                "name": "repos_list",
                "description": "List all globally registered repos as TSV: name, db_path, exists. Call this first to discover available repo names before supplying the `repo` parameter on other tools.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "get_dependencies",
                "description": "Return the files and modules that a given file imports. Supply `repo` to scope to a single codebase; omit only for explicit cross-repo queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to look up" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_callers",
                "description": "Return all graph nodes that have an incoming edge to the given symbol. Supply `repo` to scope results; omit only if the symbol may exist in multiple repos.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol name to find callers of (partial match supported)" },
                        "path": { "type": "string", "description": "Optional path hint to scope an overloaded name to a file or directory: a filename, a relative path, a directory prefix, or a path fragment (e.g. ppr.rs, src/ppr.rs, crates/travsr-retrieval, retrieval)" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["symbol"],
                    "additionalProperties": false
                }
            },
            {
                "name": "find_references",
                "description": "Enumerate every use site (incl. inline expressions, assignments, and type references) of a symbol as path:line. Supply `repo` to scope; omit only if the symbol may exist in multiple repos.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol to enumerate references of (bare name or full signature)" },
                        "path": { "type": "string", "description": "Optional path hint to scope an overloaded name to a file or directory: a filename, a relative path, a directory prefix, or a path fragment (e.g. ppr.rs, src/ppr.rs, crates/travsr-retrieval, retrieval)" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["symbol"],
                    "additionalProperties": false
                }
            },
            {
                "name": "find_pattern",
                "description": "Textual search (git grep) over each repo's tracked and untracked text files, returning path:line:col: text. Not limited to the files the graph indexes: any file git shows as text can match, with known-binary formats and ignored paths removed. Pattern is a POSIX extended regular expression (ERE); set `fixed: true` for a literal search. Optionally scope to a path prefix or files-importing(<symbol>). Supply `repo` to scope; omit only to search across all repos.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "POSIX ERE pattern to search for (use fixed:true for a literal string)" },
                        "scope": { "type": "string", "description": "Optional scope: a repo-relative path prefix, or files-importing(<symbol>)" },
                        "fixed": { "type": "boolean", "description": "Treat pattern as a literal string instead of a regular expression" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["pattern"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_blast_radius",
                "description": "Return the set of files transitively affected if the given file changes. Supply `repo` to scope results; omit only if the symbol may exist in multiple repos.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to compute blast radius for" },
                        "analysis": { "type": "string", "enum": ["tree-sitter", "semantic"], "description": "Edge mode: 'tree-sitter' (default, structural) or 'semantic' (call/reference edges; needs full cross-file analysis to be built)." },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_lang_status",
                "description": "Return whether semantic (full cross-file) analysis is available for the language of the given file, and an install hint if not. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Repo-relative file path to detect language for" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["file"],
                    "additionalProperties": false
                }
            },
            {
                "name": "search_symbol",
                "description": "Find symbol definitions matching a name across the indexed graph. Accepts exact symbol names, partial matches, and natural-language queries (e.g. 'auth session validation'). No model or API key required. IMPORTANT: always supply `repo` to avoid cross-repo noise, omit only when explicitly searching across multiple repos. Run repos_list to discover available repo names.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Symbol name or natural-language query (1-200 chars). Partial and NL queries are supported." },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). IMPORTANT: always supply to avoid cross-repo noise; omit only when explicitly searching across multiple repos." },
                        "exact": { "type": "boolean", "description": "If true, drops loose substring matches and keeps only exact, word-boundary, and prefix matches. Use for short or common names to cut noise." }
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_repo_map",
                "description": "Return a structural overview of the indexed repository. IMPORTANT: always supply `repo`, omitting fans out across all indexed repos and produces a very large, hard-to-read response. Run repos_list to discover available repo names.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). IMPORTANT: always supply, omitting fans out across all indexed repos." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "get_execution_path",
                "description": "Find the lowest-cost path from a source symbol to a sink symbol through the code graph. Supply `repo` to scope to a single codebase; scoped queries answer explicitly when the symbols do not resolve or are disconnected ('no path found').",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string", "description": "Source symbol name (partial match supported)" },
                        "sink": { "type": "string", "description": "Sink symbol name (partial match supported)" },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." }
                    },
                    "required": ["source", "sink"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_stats",
                "description": "Return accurate node and edge counts from the indexed graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Sums all repos if omitted." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "get_context",
                "description": "Retrieve the most relevant context for a query within a token budget. Accepts symbol names and natural-language queries (e.g. 'where is the auth session validated?'). Natural-language queries are translated to search terms deterministically, no model or API key required. Set include_snippets=true to get actual source code inline alongside the structural metadata. Supply `repo` to scope to a single codebase; omit only for cross-repo queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Symbol name or natural-language query (1-200 chars)" },
                        "token_budget": { "type": "integer", "description": "Hard token budget (100-32000). Defaults to 4096 if omitted." },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." },
                        "include_snippets": { "type": "boolean", "description": "When true, appends the actual source body of each selected symbol (kind-aware, docblock-stripped). Defaults to false." },
                        "snippet_budget": { "type": "integer", "description": "Optional separate token budget for the appended snippets. When omitted, snippets share the main token_budget (best-effort, stops before overflow). When provided, snippet retrieval uses this independent ceiling and does not affect node selection." }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_json",
                "description": "Return a subgraph around a symbol as structured JSON nodes and edges for graph renderers. Pass mode='overview' with no query for the repository overview tile layout.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Symbol name (1-200 chars). Resolved exact-first: full signature (e.g. 'fn:charge'), then exact bare name; substring matching is only a fallback when neither matches, so prefer the exact name for a precise single-root graph. May be empty when kind_filter is 'file' or mode is 'overview'." },
                        "direction": { "type": "string", "enum": ["deps", "callers", "both"], "description": "Edge direction. Default: both" },
                        "depth": { "type": "integer", "minimum": 1, "maximum": 4, "description": "BFS depth. Default: 2" },
                        "kind_filter": { "type": "string", "enum": ["file", ""], "description": "Restrict nodes to a specific kind. 'file' returns only file nodes and imports edges (project module map). Default: empty (all kinds)." },
                        "provenance": { "type": "string", "enum": ["", "ratified", "tree-sitter", "lsif", "scip", "live"], "description": "Restrict edges by how they were derived. Default empty returns everything, including 'live' edges resolved from uncommitted edits and not yet ratified. Use 'ratified' to exclude those and see only what the commit-gated pipeline has confirmed. Every edge in the response carries its own 'provenance' field." },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos." },
                        "mode": { "type": "string", "enum": ["", "overview"], "description": "'overview' returns directory-level component tiles (each with file_count and dependents) plus cross-component dependency edges from the resolved graph." },
                        "path_prefix": { "type": "string", "description": "When mode='overview', scope to files under this path prefix. Returns file nodes inside the prefix plus external package nodes for cross-boundary dependencies." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_snippets",
                "description": "Return tailored code snippets for one or more symbols by name. Accepts the symbol names returned by get_context, get_callers, and search_symbol. Kind-aware extraction (mode='auto'): functions/methods → up to 40 lines; classes/structs/impls → up to 15 lines (header + fields only); interfaces/traits/enums → up to 60 lines; overflowing definitions fall back to a condensed structural outline. mode='full' returns the complete definition with no line cap; mode='skeleton' returns the structural outline only. Leading docblocks are stripped. Respects a token budget; symbols are included in request order until the budget is reached (the first symbol is always included). Use this after any graph-navigation tool to read the actual code without opening files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "string",
                            "description": "Newline- or comma-separated list of symbol names (e.g. 'PaymentService.charge\\nMAX_RETRIES'). Partial matches accepted, the closest matching non-file symbol is used for each name."
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Hard token cap across all returned snippets. Default: 2000. Higher values return more symbols."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "full", "skeleton"],
                            "description": "Representation per symbol. 'auto' (default): raw body if within the kind line-cap, else a condensed structural outline. 'full': complete definition, no line cap. 'skeleton': structural outline only (signatures, no bodies)."
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repo name (run repos_list to discover). Always supply to avoid cross-repo noise; omit only when explicitly querying across all repos."
                        }
                    },
                    "required": ["symbols"],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_index_status",
                "description": "Return index freshness and completeness for a single repo: schema version, indexed vs HEAD commit staleness, node/edge counts, structural and semantic analysis state (including per-language failed/unavailable/done), and semantic (embeddings/rerank) readiness. Read-only. Never aggregates across repos; supply `repo` when more than one is registered, or the call returns an ambiguity error. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). REQUIRED when more than one repo is registered; this tool never aggregates across repos." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_daemon_logs",
                "description": "Return recent daemon log entries for a single repo (parsed, severity-filtered, and sanitized: home paths and credential-shaped substrings are best-effort redacted; do not assume logs are guaranteed secret-free). Read-only; never creates or writes any file. Never aggregates across repos; supply `repo` when more than one is registered, or the call returns an ambiguity error. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tail": { "type": "integer", "minimum": 1, "maximum": 500, "description": "Number of most-recent log lines to consider (default 50, max 500)." },
                        "level": { "type": "string", "enum": ["error", "warn", "info", "debug"], "description": "Minimum severity to include (default 'info'); more severe levels are always included." },
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). REQUIRED when more than one repo is registered; this tool never aggregates across repos, and never reads more than one repo's logs per call." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            {
                "name": "get_graph_health",
                "description": "Return a read-only graph integrity report for a single repo: ghost paths (indexed but deleted on disk), orphan edges, and lexical-index parity. Never mutates; this is the report half of `travsr fsck`, not `--fix`. O(tracked file count); slower than most tools. Never aggregates across repos; supply `repo` when more than one is registered, or the call returns an ambiguity error. Returns JSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": { "type": "string", "description": "Repo name (run repos_list to discover). REQUIRED when more than one repo is registered; this tool never aggregates across repos." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            }
        ]
    })
}

// ── L2-D: MCP sampling borrow (RFC-012 A2 F3, feature = "mcp-sampling") ────────
// The daemon is passive: it returns a prompt template string with a {{query}}
// placeholder. The host LLM fills the placeholder and calls tools/call itself.
// The daemon never calls the LLM. Security review required before cloud deploy.

#[cfg(feature = "mcp-sampling")]
fn prompts_list() -> serde_json::Value {
    serde_json::json!({
        "prompts": [{
            "name": "search_query_rewrite",
            "description": "Rewrite a natural-language query into a concise \
                            symbol-oriented search term for Travsr's graph index.",
            "arguments": [{
                "name": "query",
                "description": "The original user query",
                "required": true
            }]
        }]
    })
}

#[cfg(feature = "mcp-sampling")]
fn handle_prompts_get(id: serde_json::Value, name: &str) -> String {
    match name {
        "search_query_rewrite" => ok_response(
            id,
            serde_json::json!({
                "description": "Rewrite this query for Travsr symbol search",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Rewrite the following query into a short (1-4 word) \
                                 code-symbol search term that would appear in function \
                                 names, class names, or file names:\n\n{{query}}"
                    }
                }]
            }),
        ),
        _ => error_response(id, INVALID_PARAMS, format!("unknown prompt: {name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TOOLS: &[&str] = &[
        "get_dependencies",
        "get_callers",
        "find_references",
        "find_pattern",
        "get_blast_radius",
        "get_lang_status",
        "search_symbol",
        "get_repo_map",
        "get_execution_path",
        "get_graph_stats",
        "get_context",
        "get_graph_json",
        "get_snippets",
        // #636: read-only observability tools.
        "get_index_status",
        "get_daemon_logs",
        "get_graph_health",
    ];

    /// Tools exposed only on the stdio (single-repo) server — never in the global
    /// registry path. Synonyms (RFC-012 A2 F1) target one repo's table; repos_prune
    /// and repos_remove mutate the registry and must not be exposed in global mode.
    /// repos_list is read-only and IS available in global mode — see #374.
    const STDIO_ONLY_TOOLS: &[&str] = &[
        "synonym_list",
        "synonym_add",
        "synonym_set",
        "synonym_remove",
        "synonym_remove_term",
        "synonym_reset",
        "repos_prune",
        "repos_remove",
    ];

    /// Collect every `description` string value anywhere in a tools/list value.
    fn collect_descriptions(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, val) in map {
                    if k == "description" {
                        if let Some(s) = val.as_str() {
                            out.push(s.to_string());
                        }
                    }
                    collect_descriptions(val, out);
                }
            }
            serde_json::Value::Array(arr) => {
                for val in arr {
                    collect_descriptions(val, out);
                }
            }
            _ => {}
        }
    }

    /// UX-031/032/033: no project-internal jargon may reach a consumer through a
    /// tool `description` in EITHER list. These are implementation terms that are
    /// meaningless outside this repo (and `PCST` was also factually wrong — the
    /// path search is Dijkstra + λ-corridor, not PCST; see issue #527). Wire enum
    /// values like `tree-sitter` are deliberately not covered — this checks prose
    /// descriptions only, so the two lists cannot drift jargon back in.
    #[test]
    fn tool_descriptions_are_jargon_free() {
        // Case-sensitive tokens that do not occur in ordinary English prose, so
        // substring matching is safe (no false positives on words like "approach").
        const BANNED: &[&str] = &[
            "PCST",
            "Phase A",
            "Phase B",
            "RefCall",
            "knapsack",
            "k-core",
            "ghost-port",
            "L2-A",
            "T0 ",
            "PPR",
            "cross-encoder",
            "AST skeleton",
            "AST summary",
            "LOD",
        ];
        for list in [tools_list(), tools_list_global()] {
            let mut descriptions = Vec::new();
            collect_descriptions(&list, &mut descriptions);
            for desc in &descriptions {
                for banned in BANNED {
                    assert!(
                        !desc.contains(banned),
                        "jargon '{banned}' leaked into a tool description: {desc:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn stdio_tools_list_contains_synonym_tools() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expected in STDIO_ONLY_TOOLS {
            assert!(names.contains(expected), "tools/list missing: {expected}");
        }
    }

    #[test]
    fn global_tools_list_excludes_synonym_tools() {
        let list = tools_list_global();
        let tools = list["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for forbidden in STDIO_ONLY_TOOLS {
            assert!(
                !names.contains(forbidden),
                "tools_list_global must not expose stdio-only tool: {forbidden}"
            );
        }
    }

    #[test]
    fn synonym_tools_require_correct_fields() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        let required_map = [
            ("synonym_add", &["term", "alias"][..]),
            ("synonym_set", &["term", "aliases"][..]),
            ("synonym_remove", &["term", "alias"][..]),
            ("synonym_remove_term", &["term"][..]),
            ("repos_remove", &["name"][..]),
        ];
        for (tool_name, fields) in required_map {
            let tool = tools
                .iter()
                .find(|t| t["name"].as_str() == Some(tool_name))
                .unwrap_or_else(|| panic!("tool '{tool_name}' not found"));
            let required = tool["inputSchema"]["required"].as_array().unwrap();
            for field in fields {
                assert!(
                    required.iter().any(|r| r.as_str() == Some(field)),
                    "tool '{tool_name}' must have '{field}' in required"
                );
            }
        }
    }

    #[test]
    fn tools_list_contains_all_tools() {
        let list = tools_list();
        let tools = list["tools"].as_array().expect("tools must be an array");
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expected in ALL_TOOLS {
            assert!(names.contains(expected), "tools/list missing: {expected}");
        }
    }

    #[test]
    fn tools_list_global_contains_all_tools() {
        let list = tools_list_global();
        let tools = list["tools"].as_array().expect("tools must be an array");
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expected in ALL_TOOLS {
            assert!(
                names.contains(expected),
                "tools_list_global missing: {expected}"
            );
        }
    }

    #[test]
    fn global_tools_list_exposes_repos_list() {
        let list = tools_list_global();
        let tools = list["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(
            names.contains(&"repos_list"),
            "tools_list_global must expose repos_list in global mode (#374)"
        );
        assert!(
            !names.contains(&"repos_prune"),
            "tools_list_global must not expose repos_prune"
        );
        assert!(
            !names.contains(&"repos_remove"),
            "tools_list_global must not expose repos_remove"
        );
    }

    #[test]
    fn every_tool_has_input_schema_with_type_object() {
        for list in [tools_list(), tools_list_global()] {
            let tools = list["tools"].as_array().unwrap();
            for tool in tools {
                let name = tool["name"].as_str().unwrap();
                let schema = &tool["inputSchema"];
                assert_eq!(
                    schema["type"].as_str(),
                    Some("object"),
                    "tool '{name}' inputSchema must have type=object"
                );
                assert!(
                    schema["properties"].is_object(),
                    "tool '{name}' inputSchema must have a properties object"
                );
            }
        }
    }

    #[test]
    fn required_tools_have_correct_required_fields() {
        // get_graph_json is intentionally excluded: query is optional when mode="overview".
        let required_map = [
            ("get_dependencies", "file"),
            ("get_callers", "symbol"),
            ("find_references", "symbol"),
            ("find_pattern", "pattern"),
            ("get_blast_radius", "file"),
            ("search_symbol", "name"),
            ("get_execution_path", "source"),
            ("get_context", "query"),
        ];
        for list in [tools_list(), tools_list_global()] {
            let tools = list["tools"].as_array().unwrap();
            for (tool_name, required_field) in required_map {
                let tool = tools
                    .iter()
                    .find(|t| t["name"].as_str() == Some(tool_name))
                    .unwrap_or_else(|| panic!("tool '{tool_name}' not found"));
                let required = tool["inputSchema"]["required"]
                    .as_array()
                    .unwrap_or_else(|| panic!("tool '{tool_name}' must have required array"));
                assert!(
                    required.iter().any(|r| r.as_str() == Some(required_field)),
                    "tool '{tool_name}' must have '{required_field}' in required"
                );
            }
        }
    }

    #[test]
    fn observability_tools_have_no_required_fields() {
        for list in [tools_list(), tools_list_global()] {
            let tools = list["tools"].as_array().unwrap();
            for name in ["get_index_status", "get_daemon_logs", "get_graph_health"] {
                let tool = tools
                    .iter()
                    .find(|t| t["name"].as_str() == Some(name))
                    .unwrap_or_else(|| panic!("tool '{name}' not found"));
                let required = tool["inputSchema"]["required"]
                    .as_array()
                    .unwrap_or_else(|| panic!("tool '{name}' must have a required array"));
                assert!(
                    required.is_empty(),
                    "tool '{name}' must have no required fields"
                );
            }
        }
    }

    #[test]
    fn get_daemon_logs_schema_has_tail_bounds_and_level_enum() {
        for list in [tools_list(), tools_list_global()] {
            let tools = list["tools"].as_array().unwrap();
            let tool = tools
                .iter()
                .find(|t| t["name"].as_str() == Some("get_daemon_logs"))
                .expect("get_daemon_logs must be present");
            let props = &tool["inputSchema"]["properties"];
            assert_eq!(props["tail"]["minimum"].as_i64(), Some(1));
            assert_eq!(props["tail"]["maximum"].as_i64(), Some(500));
            let levels: Vec<&str> = props["level"]["enum"]
                .as_array()
                .expect("level must have an enum")
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(levels, vec!["error", "warn", "info", "debug"]);
        }
    }

    #[test]
    fn observability_tools_expose_repo_only_in_global_mode() {
        let stdio_tools = tools_list()["tools"].as_array().unwrap().clone();
        let global_tools = tools_list_global()["tools"].as_array().unwrap().clone();
        for name in ["get_index_status", "get_daemon_logs", "get_graph_health"] {
            let stdio_tool = stdio_tools
                .iter()
                .find(|t| t["name"].as_str() == Some(name))
                .unwrap();
            assert!(
                stdio_tool["inputSchema"]["properties"]["repo"].is_null(),
                "stdio '{name}' must not expose 'repo'"
            );
            let global_tool = global_tools
                .iter()
                .find(|t| t["name"].as_str() == Some(name))
                .unwrap();
            assert!(
                global_tool["inputSchema"]["properties"]["repo"].is_object(),
                "global '{name}' must expose 'repo'"
            );
        }
    }

    #[test]
    fn schema_version_is_semver() {
        for list in [tools_list(), tools_list_global()] {
            let version = list["_schemaVersion"]
                .as_str()
                .expect("_schemaVersion must be present");
            let parts: Vec<&str> = version.split('.').collect();
            assert_eq!(
                parts.len(),
                3,
                "_schemaVersion must be X.Y.Z, got: {version}"
            );
            for part in &parts {
                part.parse::<u32>()
                    .unwrap_or_else(|_| panic!("semver part must be numeric: {part}"));
            }
        }
    }

    #[test]
    fn all_schemas_have_additional_properties_false() {
        for list in [tools_list(), tools_list_global()] {
            let tools = list["tools"].as_array().unwrap();
            for tool in tools {
                let name = tool["name"].as_str().unwrap();
                assert_eq!(
                    tool["inputSchema"]["additionalProperties"].as_bool(),
                    Some(false),
                    "tool '{name}' inputSchema must set additionalProperties=false"
                );
            }
        }
    }

    /// Protocol consistency: what `initialize` advertises and what the
    /// dispatcher actually answers must agree, in both directions.
    ///
    /// #717 (ISS-005, ISS-006) reported `resources/list` and `prompts/list` as
    /// missing handlers. They are absent on purpose: travsr serves its graph
    /// through tools and exposes no MCP resources at all, and prompts sit
    /// behind the default-off `mcp-sampling` feature pending a security
    /// review. A client that reads `initialize` never calls either here.
    ///
    /// Asserted against the dispatcher rather than the source, because what a
    /// client experiences is the response, and an endpoint that answers while
    /// its capability is unadvertised is a feature leaking past its flag.
    #[test]
    fn unadvertised_capabilities_have_no_live_endpoint() {
        let caps = server_capabilities();
        let call = |method: &str| -> serde_json::Value {
            let req = RpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(serde_json::json!(1)),
                method: method.into(),
                params: None,
            };
            let resp = handle_request_global(req).expect("a request must get a response");
            serde_json::from_str(&resp).unwrap()
        };

        // Resources: never advertised, never answered.
        assert!(
            caps.get("resources").is_none(),
            "no resources capability is advertised"
        );
        assert_eq!(
            call("resources/list")["error"]["code"],
            serde_json::json!(METHOD_NOT_FOUND),
            "an unadvertised capability must not have an endpoint behind it"
        );

        // Prompts: advertised and answered together, or neither.
        let prompts_advertised = caps["prompts"].is_object();
        let prompts_answered = call("prompts/list").get("result").is_some();
        assert_eq!(
            prompts_advertised, prompts_answered,
            "prompts capability and endpoint must agree (advertised={prompts_advertised}, \
             answered={prompts_answered})"
        );
    }

    #[test]
    fn initialize_advertises_capabilities() {
        let caps = server_capabilities();
        assert!(
            caps["tools"].is_object(),
            "initialize must always advertise the tools capability"
        );
        // The prompts capability is present iff mcp-sampling is compiled in.
        assert_eq!(
            caps["prompts"].is_object(),
            cfg!(feature = "mcp-sampling"),
            "prompts capability must be advertised exactly when mcp-sampling is enabled"
        );
    }

    #[cfg(feature = "mcp-sampling")]
    #[test]
    fn prompts_list_exposes_search_query_rewrite() {
        let list = prompts_list();
        let prompts = list["prompts"].as_array().expect("prompts must be array");
        let names: Vec<&str> = prompts
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"search_query_rewrite"),
            "prompts/list must expose search_query_rewrite"
        );
    }

    #[cfg(feature = "mcp-sampling")]
    #[test]
    fn prompts_get_returns_template_with_placeholder() {
        let resp = handle_prompts_get(serde_json::json!(1), "search_query_rewrite");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let text = v["result"]["messages"][0]["content"]["text"]
            .as_str()
            .expect("template text must be present");
        assert!(
            text.contains("{{query}}"),
            "prompt template must contain the {{{{query}}}} placeholder"
        );
        // Passive contract: the daemon returns a template, never an LLM call result.
        assert!(v["error"].is_null(), "valid prompt must not error");
    }

    #[cfg(feature = "mcp-sampling")]
    #[test]
    fn prompts_get_unknown_name_is_invalid_params() {
        let resp = handle_prompts_get(serde_json::json!(2), "does_not_exist");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(
            v["error"]["code"].as_i64(),
            Some(INVALID_PARAMS as i64),
            "unknown prompt name must return INVALID_PARAMS"
        );
    }

    #[cfg(feature = "mcp-sampling")]
    #[test]
    fn global_handler_routes_prompts_list() {
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(7)),
            method: "prompts/list".into(),
            params: None,
        };
        let resp = handle_request_global(req).expect("request must produce a response");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(
            v["result"]["prompts"].is_array(),
            "prompts/list must route through the global handler when mcp-sampling is on"
        );
    }
}
