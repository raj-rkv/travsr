//! `travsr graph` — subgraph rendering around a symbol or file.
//!
//! Data acquisition is shared with the daemon via `travsr_mcp::query`
//! (#318 O1): a running daemon answers from its warm store; otherwise the
//! store is opened directly (read-only fast path). Rendering happens here,
//! from the payload, so both routes produce identical output.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use anyhow::Context as _;
use travsr_mcp::query::{
    self, EdgeEntry, GraphPayload, GraphQueryArgs, NodeEntry, QueryDirection, QueryEdgeMode,
};

use crate::daemon_client;
use crate::repo::find_git_root;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Direction {
    /// Follow outgoing edges (what does this symbol import / define?)
    Deps,
    /// Follow incoming edges (who calls / depends on this symbol?). Containment
    /// edges to the defining file are shown but not expanded.
    Callers,
    /// Follow both directions
    Both,
}

impl From<Direction> for QueryDirection {
    fn from(d: Direction) -> Self {
        match d {
            Direction::Deps => QueryDirection::Deps,
            Direction::Callers => QueryDirection::Callers,
            Direction::Both => QueryDirection::Both,
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Format {
    /// ASCII tree printed to stdout
    Tree,
    /// Graphviz DOT with clusters and shapes — pipe to `dot -Tsvg` to render
    Dot,
    /// Structured JSON for AI / tooling consumption
    Json,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum EdgeMode {
    /// Prefer semantic call edges; fall back to structural if none exist
    Semantic,
    /// Follow all edge kinds (original behaviour)
    All,
}

impl From<EdgeMode> for QueryEdgeMode {
    fn from(m: EdgeMode) -> Self {
        match m {
            EdgeMode::Semantic => QueryEdgeMode::Semantic,
            EdgeMode::All => QueryEdgeMode::All,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    query_str: &str,
    path: Option<String>,
    depth: u8,
    direction: Direction,
    format: Format,
    edge_mode: EdgeMode,
    include_noise: bool,
    budget: usize,
) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("getting current directory")?;
    let repo_root = find_git_root(&cwd)?;
    let db_path = repo_root.join(".travsr/graph.db");

    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    let args = GraphQueryArgs {
        query: query_str.to_string(),
        path: path.clone(),
        depth,
        direction: direction.into(),
        edge_mode: edge_mode.into(),
        include_noise,
    };

    // Callers and blast radius ride ref/call edges, so an incomplete Phase B
    // turns "nothing found" into a wrong answer rather than a small one.
    //
    // `deps` does not: it rides Phase A import and `defines` edges, which are
    // complete whether or not Phase B has run. Warning there told the user their
    // complete answer might be missing something, which is both wrong and the
    // fastest way to teach someone to ignore the warning that matters.
    //
    // The cross-checkout note itself is not subject to that split: it says which
    // tree the answer describes, and a complete `deps` answer about the wrong
    // checkout is still the wrong answer. Its folded-in degraded caveat *is*
    // subject to it, being the Phase B claim above in different words. Emitting
    // the note explicitly here, rather than relying on
    // `warn_if_call_graph_degraded` to delegate to it, keeps the "every
    // direction gets the cross-checkout note" rule visible at the call site that
    // owns the split — a refactor of that delegation cannot silently drop it
    // from `callers` / `blast-radius`.
    //
    // One local drives both halves so they cannot drift apart: it decides
    // whether the standalone Phase B note is printed *and* whether the
    // cross-checkout note carries that same claim as its degraded caveat.
    let reads_call_edges = !matches!(direction, Direction::Deps);
    let cross = daemon_client::warn_if_cross_checkout(&db_path, reads_call_edges);
    if !cross && reads_call_edges {
        daemon_client::warn_if_phase_b_degraded(&db_path);
    }

    // Daemon route first (#318 O1), direct read-only open as fallback.
    let payload: GraphPayload =
        match daemon_client::try_query(&repo_root, "graph", serde_json::to_value(&args)?) {
            Some(p) => p,
            None => {
                let store = daemon_client::open_read_store(&db_path)?;
                query::graph_query(&store, &args)?
            }
        };

    if let Some(candidates) = &payload.candidates {
        let count = candidates.len();
        let limit = travsr_mcp::AMBIGUOUS_DISPLAY_LIMIT;
        // The store caps the candidate set (Tier 1 at NODE_EXACT_LOOKUP_LIMIT,
        // Tier 2 at NODE_NAME_SEARCH_LIMIT), so once we get back more than we
        // display, `count` is a lower bound, not the true total — hence "at
        // least {count}" rather than a definite count (#565 / RFC-002).
        let truncated = count > limit;

        // `graph --format json` is the AI/tooling surface, and disambiguation is
        // exactly the case where an agent needs to read the options and re-query
        // with a `--path`. Emit the candidates as JSON on stdout (still a non-zero
        // exit) so "ambiguous, here are the choices" is machine-distinguishable
        // from "the command failed". `truncated` marks `count` as a lower bound.
        if matches!(format, Format::Json) {
            let candidates_json: Vec<serde_json::Value> = candidates
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "id": n.id.to_string(),
                        // `signature` is the raw form and round-trips through
                        // exact-signature resolution (`--path` re-query);
                        // `label` is the clean display name.
                        "signature": n.signature,
                        "label": n.label,
                        "kind": n.kind,
                        "path": n.path,
                        "line": n.line,
                    })
                })
                .collect();
            let out = serde_json::json!({
                "status": "ambiguous",
                "count": count,
                "truncated": truncated,
                "candidates": candidates_json,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            anyhow::bail!("ambiguous symbol query");
        }

        // #757: `--path` only disambiguates *across* files. When the collision is
        // same-file (a struct and its own impl, both at one path), the hint
        // narrows nothing. The exact signature always resolves uniquely (Tier 1
        // of the resolver), so point at it explicitly and use the first
        // candidate's signature as a concrete example.
        let example_sig = candidates.first().map(|n| n.signature.as_str());
        let escape_hatch = match example_sig {
            Some(sig) => format!(
                "Re-run with a `--path` hint (for cross-file matches) or with one of the \
                 exact signatures listed below (e.g. `{sig}`) to pick one:"
            ),
            None => "Re-run with a `--path` hint to pick one:".to_string(),
        };
        if truncated {
            eprintln!(
                "'{query_str}' is ambiguous, showing {limit} of at least {count} definitions. \
                 {escape_hatch}"
            );
        } else {
            eprintln!("'{query_str}' is ambiguous, {count} definitions. {escape_hatch}");
        }
        for n in candidates.iter().take(limit) {
            let loc = n.line.map(|l| format!(":{l}")).unwrap_or_default();
            // The clean label reads best, but for SCIP-synthesized nodes it is
            // *not* the re-runnable token. Append the exact signature whenever it
            // differs from the label so there is always a copy-pasteable query.
            if n.signature == n.label {
                eprintln!("  {} ({}) \u{2014} {}{}", n.label, n.kind, n.path, loc);
            } else {
                eprintln!(
                    "  {} ({}) \u{2014} {}{}   [re-run: {}]",
                    n.label, n.kind, n.path, loc, n.signature
                );
            }
        }
        if truncated {
            eprintln!("[truncated: additional filtering/narrowing is required]");
        }
        anyhow::bail!("ambiguous symbol query");
    }

    if payload.seed.is_none() {
        if let Some(p) = path {
            anyhow::bail!("no matching definition found for '{query_str}' in path '{p}'");
        } else {
            println!("no symbols matching '{query_str}'");
            return Ok(());
        }
    }

    // C3: a manifest/config file has no inbound edges — no source file depends on
    // a manifest, so `--direction callers` is legitimately empty. Explain that
    // instead of leaving the user to wonder, and point at what does work.
    let manifest_dead_end = matches!(direction, Direction::Callers)
        && matches!(format, Format::Tree)
        && payload.tree.is_empty()
        && payload
            .seed
            .as_ref()
            .is_some_and(|s| s.kind == "file" && is_config_manifest_path(&s.path));

    render(payload, format, budget)?;
    if manifest_dead_end {
        eprintln!(
            "note: manifests are configuration inputs, no source file depends on one, so \
             callers are empty. Use `--direction deps` to see what this manifest declares."
        );
    }
    Ok(())
}

/// True when `path` is a dependency/config manifest (data-format extension or a
/// name-recognized manifest). Used to explain an empty `--direction callers`.
fn is_config_manifest_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    if matches!(ext, "json" | "jsonc" | "yaml" | "yml" | "toml" | "xml") {
        return true;
    }
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    travsr_core::is_manifest_file(name)
}

pub fn run_all(format: Format, budget: usize) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("getting current directory")?;
    let repo_root = find_git_root(&cwd)?;
    let db_path = repo_root.join(".travsr/graph.db");

    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    daemon_client::warn_if_call_graph_degraded(&db_path);
    // --all dumps are large by construction — always computed locally rather
    // than shipped through the daemon socket.
    let store = daemon_client::open_read_store(&db_path)?;
    let payload = query::graph_all_payload(&store)?;
    render(payload, format, budget)
}

fn render(mut payload: GraphPayload, format: Format, budget: usize) -> anyhow::Result<()> {
    // L12: tell the user their budget is unlimited so they know no truncation will happen.
    if budget == 0 {
        eprintln!("token budget: unlimited");
    }
    // #318 O6: token budget — prefix of BFS order, seed always kept.
    let truncated = query::apply_token_budget(&mut payload, budget);

    // A seeded query force-keeps the seed, so `rendered >= 1` even when the budget
    // could not afford a single neighbour; distinguish "only the queried symbol
    // fit" from a real partial graph so the footer stays honest for `--budget 1`.
    let only_seed = payload.seed.is_some() && payload.nodes.len() <= 1;

    match format {
        Format::Tree => {
            let rendered = if let Some(seed) = &payload.seed {
                println!("{} ({})", seed.label, seed.kind);
                print_tree(&payload);
                payload.nodes.len()
            } else {
                // --all tree mode: file listing (historic behaviour).
                let mut files: Vec<&NodeEntry> =
                    payload.nodes.iter().filter(|n| n.kind == "file").collect();
                files.sort_by(|a, b| a.path.cmp(&b.path));
                for node in &files {
                    println!("{}", node.path);
                }
                files.len()
            };
            print_budget_footer(rendered, truncated, budget, "", only_seed);
        }
        Format::Dot => {
            print_dot(&payload)?;
            print_budget_footer(payload.nodes.len(), truncated, budget, "// ", only_seed);
        }
        Format::Json => print_json(&payload, budget, truncated)?,
    }

    Ok(())
}

/// Footer shown after a budgeted graph render. When some neighbours were shown it
/// reports how many more were cut; when the budget was too small to show any node
/// (`--all`) or fit only the force-kept queried symbol (seeded), the "N more"
/// phrasing is misleading — there is no partial list to extend — so it says the
/// budget is too small and how to lift it.
fn print_budget_footer(
    rendered: usize,
    truncated: usize,
    budget: usize,
    prefix: &str,
    only_seed: bool,
) {
    if truncated == 0 {
        return;
    }
    if rendered == 0 {
        println!(
            "{prefix}Budget of {budget} tokens is too small to display any graph nodes, \
             raise it with --budget (or --budget 0 for no limit)."
        );
    } else if only_seed {
        println!(
            "{prefix}Budget of {budget} tokens fits only the queried symbol, not its graph, \
             raise it with --budget (or --budget 0 for no limit)."
        );
    } else {
        println!("{prefix}… {truncated} more nodes beyond budget (raise with --budget)");
    }
}

fn print_tree(payload: &GraphPayload) {
    let nodes_by_id: HashMap<u64, &NodeEntry> = payload.nodes.iter().map(|n| (n.id, n)).collect();
    // Children per parent, in BFS discovery order.
    let mut children: HashMap<u64, Vec<(&str, u64, bool)>> = HashMap::new();
    for step in &payload.tree {
        children.entry(step.parent).or_default().push((
            step.edge_kind.as_str(),
            step.child,
            step.incoming,
        ));
    }
    if let Some(seed) = &payload.seed {
        print_tree_level(seed.id, &nodes_by_id, &children, "");
    }
}

fn print_tree_level(
    node_id: u64,
    nodes_by_id: &HashMap<u64, &NodeEntry>,
    children: &HashMap<u64, Vec<(&str, u64, bool)>>,
    prefix: &str,
) {
    let Some(kids) = children.get(&node_id) else {
        return;
    };
    for (i, (edge_kind, child_id, incoming)) in kids.iter().enumerate() {
        let is_last = i == kids.len() - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let extension = if is_last { "    " } else { "│   " };

        if let Some(child) = nodes_by_id.get(child_id) {
            // #564: the arrow renders the stored edge orientation — `→` for an
            // outgoing edge (parent → child), `←` for an incoming one (the
            // child calls / contains the parent).
            let arrow = if *incoming { "←" } else { "→" };
            println!(
                "{prefix}{connector}{edge_kind} {arrow} {} ({})",
                child.label, child.kind
            );
            print_tree_level(
                *child_id,
                nodes_by_id,
                children,
                &format!("{prefix}{extension}"),
            );
        }
    }
}

fn print_dot(payload: &GraphPayload) -> anyhow::Result<()> {
    let nodes_map: HashMap<u64, &NodeEntry> = payload.nodes.iter().map(|n| (n.id, n)).collect();

    // Resolve import nodes to the file node they reference.
    let mut import_redirect: HashMap<u64, u64> = HashMap::new();
    for node in &payload.nodes {
        if node.kind == "import" {
            // Best-effort local resolution without store lookup in --all mode
            for cand in &payload.nodes {
                if cand.kind != "file" {
                    continue;
                }
                let specifier = node
                    .signature
                    .strip_prefix("import:")
                    .unwrap_or(&node.signature);
                if specifier.starts_with('.') {
                    let basename = std::path::Path::new(specifier)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(specifier);
                    let stem = std::path::Path::new(&cand.path)
                        .file_stem()
                        .and_then(|s| s.to_str());
                    if stem == Some(basename) {
                        import_redirect.insert(node.id, cand.id);
                        break;
                    }
                }
            }
        }
    }

    // Rewrite edges through the redirect table; drop self-loops and duplicates.
    let mut seen: HashSet<(u64, u64, String)> = HashSet::new();
    let mut edges: Vec<(u64, u64, String)> = Vec::new();
    for EdgeEntry { src, dst, kind, .. } in &payload.edges {
        let s = import_redirect.get(src).copied().unwrap_or(*src);
        let d = import_redirect.get(dst).copied().unwrap_or(*dst);
        if s == d {
            continue;
        }
        let key = (s, d, kind.clone());
        if seen.insert(key) {
            edges.push((s, d, kind.clone()));
        }
    }

    // Group visible nodes by kind (skip resolved import stubs).
    let mut by_kind: HashMap<&str, Vec<u64>> = HashMap::new();
    for node in &payload.nodes {
        if node.kind == "import" && import_redirect.contains_key(&node.id) {
            continue;
        }
        by_kind.entry(node.kind.as_str()).or_default().push(node.id);
    }

    // Cluster definitions: (kind, label, shape, fill, border)
    let clusters: &[(&str, &str, &str, &str, &str)] = &[
        ("file", "Files", "folder", "#dbeafe", "#3b82f6"),
        ("class", "Classes", "box3d", "#dcfce7", "#22c55e"),
        ("function", "Functions", "ellipse", "#fef9c3", "#eab308"),
        ("method", "Methods", "ellipse", "#fce7f3", "#ec4899"),
        ("variable", "Variables", "plaintext", "#f3e8ff", "#a855f7"),
        ("import", "Imports", "note", "#f1f5f9", "#94a3b8"),
    ];

    println!("digraph travsr {{");
    println!("  rankdir=LR;");
    println!("  compound=true;");
    println!("  graph [fontname=\"monospace\" fontsize=11];");
    println!("  node  [fontname=\"monospace\" fontsize=10];");
    println!("  edge  [fontname=\"monospace\" fontsize=9];");
    println!();

    for (idx, (kind, label, shape, fill, border)) in clusters.iter().enumerate() {
        let Some(ids) = by_kind.get(kind) else {
            continue;
        };
        if ids.is_empty() {
            continue;
        }
        println!("  subgraph cluster_{idx} {{");
        println!("    label=\"{label}\";");
        println!("    style=filled;");
        println!("    color=\"{border}\";");
        println!("    fillcolor=\"{fill}\";");
        println!();
        for &nid in ids {
            if let Some(node) = nodes_map.get(&nid) {
                let label = escape_dot(&format!("{}\n{}", node.label, node.path));
                println!(
                    "    n{nid} [label=\"{label}\" shape={shape} style=filled \
                     fillcolor=\"{fill}\" color=\"{border}\"];",
                );
            }
        }
        println!("  }}");
        println!();
    }

    // Emit edges; suppress defines/binding labels from containers to members.
    for (src_id, dst_id, kind) in &edges {
        let src_kind = nodes_map.get(src_id).map(|n| n.kind.as_str()).unwrap_or("");
        let dst_kind = nodes_map.get(dst_id).map(|n| n.kind.as_str()).unwrap_or("");

        let suppress = kind == "defines/binding"
            && matches!(src_kind, "file" | "class")
            && matches!(dst_kind, "function" | "method" | "variable" | "class");

        if suppress {
            println!("  n{src_id} -> n{dst_id};");
        } else {
            println!("  n{src_id} -> n{dst_id} [label=\"{kind}\"];");
        }
    }

    println!("}}");
    Ok(())
}

fn print_json(payload: &GraphPayload, budget: usize, truncated: usize) -> anyhow::Result<()> {
    let out = build_graph_json(payload, budget, truncated)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// Build the `graph --format json` document. Split out of [`print_json`] so the
/// schema_version 1 contract (raw `signature`, additive `label`, edge
/// `from_id`/`to_id`) can be pinned by a test without capturing stdout.
fn build_graph_json(
    payload: &GraphPayload,
    budget: usize,
    truncated: usize,
) -> anyhow::Result<serde_json::Value> {
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for node in &payload.nodes {
        *kinds.entry(node.kind.clone()).or_default() += 1;
    }

    let mut node_entries: Vec<serde_json::Value> = payload
        .nodes
        .iter()
        .map(|node| {
            serde_json::json!({
                "id": node.id.to_string(),
                // schema_version 1 documents `signature` as the raw node
                // signature; keep it raw so distinct overloads / same-named
                // types in different packages stay distinguishable, and expose
                // the clean display name as an additive `label` field.
                "signature": node.signature,
                "label": node.label,
                "kind": node.kind,
                "path": node.path,
                "language": node.language,
                "depth_from_seed": node.depth,
            })
        })
        .collect();
    node_entries.sort_by(|a, b| {
        let da = a["depth_from_seed"].as_u64().unwrap_or(0);
        let db = b["depth_from_seed"].as_u64().unwrap_or(0);
        da.cmp(&db).then(
            a["signature"]
                .as_str()
                .unwrap_or("")
                .cmp(b["signature"].as_str().unwrap_or("")),
        )
    });

    // Clean display label per node id; endpoints that are noise nodes (not in
    // `nodes`) still render, so fall back to cleaning the raw endpoint signature.
    let label_by_id: HashMap<u64, &str> = payload
        .nodes
        .iter()
        .map(|n| (n.id, n.label.as_str()))
        .collect();
    let edge_entries: Vec<serde_json::Value> = payload
        .edges
        .iter()
        .filter(|e| !e.src_sig.is_empty() && !e.dst_sig.is_empty())
        .map(|e| {
            let from = label_by_id
                .get(&e.src)
                .map(|l| Cow::Borrowed(*l))
                .unwrap_or_else(|| travsr_core::display_signature(&e.src_sig, ""));
            let to = label_by_id
                .get(&e.dst)
                .map(|l| Cow::Borrowed(*l))
                .unwrap_or_else(|| travsr_core::display_signature(&e.dst_sig, ""));
            serde_json::json!({
                // Node ids keep two edges between collapsed-label endpoints
                // (overloads, same-named types across packages) distinct.
                "from_id": e.src.to_string(),
                "to_id": e.dst.to_string(),
                "from": from,
                "to": to,
                "kind": e.kind,
                "provenance": e.provenance,
            })
        })
        .collect();

    let mut summary = if let Some(s) = &payload.seed {
        serde_json::json!({
            "mode": "query",
            "root": s.label,
            "root_path": s.path,
            "total_nodes": payload.nodes.len(),
            "total_edges": edge_entries.len(),
            "kinds": kinds,
        })
    } else {
        serde_json::json!({
            "mode": "all",
            "total_nodes": payload.nodes.len(),
            "total_edges": edge_entries.len(),
            "kinds": kinds,
        })
    };
    if truncated > 0 {
        summary["token_budget"] = serde_json::json!(budget);
        summary["truncated_nodes"] = serde_json::json!(truncated);
    }
    let mut out = serde_json::json!({
        "schema_version": 1,
        "summary": summary,
        "nodes": node_entries,
        "edges": edge_entries,
    });
    if let Some(cov) = &payload.coverage {
        out["coverage"] = serde_json::to_value(cov)?;
    }

    Ok(out)
}

fn escape_dot(s: &str) -> String {
    s.replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use travsr_mcp::query::{EdgeEntry, GraphPayload, NodeEntry};

    fn node(id: u64, sig: &str, label: &str, path: &str) -> NodeEntry {
        NodeEntry {
            id,
            signature: sig.to_string(),
            kind: "method".to_string(),
            path: path.to_string(),
            language: "java".to_string(),
            depth: 0,
            label: label.to_string(),
            tokens: 1,
            line: Some(3),
        }
    }

    /// Pin the schema_version 1 contract the review regressed and this PR
    /// restored: `signature` stays raw, `label` is the additive clean name, and
    /// edges carry `from_id`/`to_id` so two edges between collapsed-label
    /// endpoints (same type name in different packages) stay distinct.
    #[test]
    fn graph_json_keeps_raw_signature_and_distinct_edge_ids() {
        let n1 = node(
            1,
            "scip:a/Greeter.java:semanticdb maven . . com/a/Greeter#greet().",
            "Greeter.greet",
            "a/Greeter.java",
        );
        let n2 = node(
            2,
            "scip:b/Greeter.java:semanticdb maven . . com/b/Greeter#greet().",
            "Greeter.greet",
            "b/Greeter.java",
        );
        let payload = GraphPayload {
            seed: Some(n1.clone()),
            nodes: vec![n1, n2],
            edges: vec![EdgeEntry {
                src: 1,
                dst: 2,
                kind: "ref/call".to_string(),
                provenance: "scip".to_string(),
                src_sig: "scip:a/Greeter.java:semanticdb maven . . com/a/Greeter#greet()."
                    .to_string(),
                dst_sig: "scip:b/Greeter.java:semanticdb maven . . com/b/Greeter#greet()."
                    .to_string(),
            }],
            tree: vec![],
            coverage: None,
            last_commit: None,
            candidates: None,
        };

        let out = build_graph_json(&payload, 0, 0).unwrap();
        assert_eq!(out["schema_version"], 1);

        let nodes = out["nodes"].as_array().unwrap();
        // `signature` is the raw compiler symbol; `label` is the clean name.
        assert_eq!(
            nodes[0]["signature"],
            "scip:a/Greeter.java:semanticdb maven . . com/a/Greeter#greet()."
        );
        assert_eq!(nodes[0]["label"], "Greeter.greet");

        let edges = out["edges"].as_array().unwrap();
        let e = &edges[0];
        // Collapsed labels are equal, but the ids keep the two endpoints distinct.
        assert_eq!(e["from"], "Greeter.greet");
        assert_eq!(e["to"], "Greeter.greet");
        assert_eq!(e["from_id"], "1");
        assert_eq!(e["to_id"], "2");
        assert_ne!(e["from_id"], e["to_id"]);
    }
}
