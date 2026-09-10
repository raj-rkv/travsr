// SYNC: keep in sync with crates/travsr-indexer/src/phase_b_rust.rs (TODO: extract to shared crate)
//! Native Rust Phase B — zero external-tool dependencies.
//!
//! Sources of edges:
//!   1. Cargo.toml parsing → `Depends` edges between crate nodes
//!   2. Tree-sitter call-site query → `RefCall` edges between functions
//!   3. Tree-sitter field-access query → `RefField` occurrences on struct
//!      fields (#757), keyed to `field:Owner.name` via the recovered receiver
//!      type (fail-closed without one, mirroring the method contract)
//!
//! Accuracy: name-based resolution without a type system. Correct for direct
//! calls; approximate for trait-dispatched generics. Always better than zero
//! edges (the outcome when rust-analyzer is absent). When rust-analyzer is
//! available the caller merges LSIF output on top for higher fidelity.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use streaming_iterator::StreamingIterator as _;
use travsr_core::{Edge, EdgeKind, Node, ScipRef, UnresolvedCall, VName};
use tree_sitter::{Parser, Query, QueryCursor};

// ── Tree-sitter query ─────────────────────────────────────────────────────────

/// Captures three call-site patterns plus field-access references:
///   `call.fn`     — bare identifier: `foo()`
///   `call.method` — field call: `self.process()` / `obj.method()`, with the
///                   receiver expression bound as `call.recv` (#529) so the
///                   extractor can attempt to recover its type instead of
///                   discarding it.
///   `call.scoped` — scoped path: `Type::new()` / `std::mem::swap()`
///   `field.name`  — field read/write: `x.foo` (#757), with the receiver bound
///                   as `field.recv`. This pattern also matches the
///                   field-expression inside a method call (`x.foo()`); the
///                   handler discards those (they are the `call.method` case)
///                   and keeps only genuine field accesses.
const CALL_QUERY: &str = "
(call_expression function: (identifier) @call.fn)
(call_expression function: (field_expression value: (_) @call.recv field: (field_identifier) @call.method))
(call_expression function: (scoped_identifier name: (identifier) @call.scoped))
(field_expression value: (_) @field.recv field: (field_identifier) @field.name)
";

/// Common names that generate enormous noise (ubiquitous in every Rust crate) and
/// provide no meaningful cross-crate signal. Drop them entirely — no edge, no
/// UnresolvedCall.
const NOISE_NAMES: &[&str] = &[
    "new",
    "from",
    "into",
    "clone",
    "default",
    "fmt",
    "drop",
    "iter",
    "next",
    "unwrap",
    "expect",
    "ok",
    "err",
    "map",
    "and_then",
    "unwrap_or",
    "collect",
    "push",
    "len",
    "is_empty",
];

// ── Public API ────────────────────────────────────────────────────────────────

/// Native Rust Phase B output: `(nodes, edges, unresolved_calls, refs)`.
/// `refs` carries same-file call occurrences (issue #299) for `edge_sites`.
pub type NativePhaseB = (Vec<Node>, Vec<Edge>, Vec<UnresolvedCall>, Vec<ScipRef>);

/// Extract native Phase B edges for a Rust corpus rooted at `root`.
///
/// Returns `(nodes, edges, unresolved_calls, refs)`:
///   - crate nodes + `Depends` edges from Cargo.toml dependency graph
///   - `Depends`/structural edges (same-file call edges are carried as `refs`)
///   - `UnresolvedCall`s for cross-crate calls that cannot be anchored locally
///   - `ScipRef` occurrence records for same-file and struct/enum scoped calls
///     (issue #299): carry the call-site line so the store's
///     `write_scip_attributed_batch` records `edge_sites` rows, giving
///     `find_references` exact `path:line` for Rust.
///
/// Note: travsr-analysis does not have daemon plumbing, so `unresolved_calls`
/// are returned to the caller but are NOT resolved against the store here.
///
/// When `files` is `Some`, the caller supplies pre-walked `(abs_path, vname_path)`
/// pairs from the daemon's Phase A walk (P6 — #329); the extractor uses them
/// directly and skips its own directory walk. Pass `None` to fall back to
/// `collect_source_files`.
pub fn extract_native_phase_b(
    corpus: &str,
    root: &Path,
    files: Option<&[(PathBuf, String)]>,
) -> anyhow::Result<NativePhaseB> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut unresolved: Vec<UnresolvedCall> = Vec::new();
    let mut refs: Vec<ScipRef> = Vec::new();

    // Pass 1: crate dependency graph via Cargo.toml
    match extract_cargo_deps(corpus, root) {
        Ok((dep_nodes, dep_edges)) => {
            nodes.extend(dep_nodes);
            edges.extend(dep_edges);
        }
        Err(e) => tracing::debug!(err = %e, "cargo dep extraction skipped"),
    }

    // Pass 2: call-site edges via tree-sitter
    let language = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
    let query = match Query::new(&language, CALL_QUERY) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(err = %e, "rust call-site query compile failed, skipping Phase B calls");
            return Ok((nodes, edges, unresolved, refs));
        }
    };

    // Use the daemon's pre-walked file list when available (P6 — #329); fall back
    // to a local walk for old daemons and the `travsr index` CLI path.
    let walked;
    let file_pairs: &[(PathBuf, String)] = match files {
        Some(f) => f,
        None => {
            walked = collect_source_files(root, &["rs"]);
            &walked
        }
    };

    for (abs_path, vname_path) in file_pairs {
        match extract_file_call_edges(corpus, abs_path, vname_path, &language, &query) {
            Ok((file_unresolved, file_refs)) => {
                unresolved.extend(file_unresolved);
                refs.extend(file_refs);
            }
            Err(e) => {
                tracing::debug!(err = %e, path = %abs_path.display(), "rust call extraction skipped")
            }
        }
    }

    // Dedup — same pattern as Phase A
    nodes.sort_unstable_by_key(|n| n.id);
    nodes.dedup_by_key(|n| n.id);
    edges.sort_unstable_by_key(|e| (e.src, e.dst));
    edges.dedup_by(|a, b| a.src == b.src && a.dst == b.dst && a.kind == b.kind);
    // Dedup on (src, callee_sig, caller_line) so two distinct call sites of the
    // same callee from one caller both survive — find_references (#299) needs
    // every occurrence line, not just the first.
    unresolved.sort_unstable_by(|a, b| {
        a.src
            .0
            .cmp(&b.src.0)
            .then(a.callee_sig.cmp(&b.callee_sig))
            .then(a.caller_line.cmp(&b.caller_line))
    });
    unresolved.dedup_by(|a, b| {
        a.src == b.src && a.callee_sig == b.callee_sig && a.caller_line == b.caller_line
    });
    refs.sort_unstable_by(|a, b| {
        a.caller_path
            .cmp(&b.caller_path)
            .then(a.caller_line.cmp(&b.caller_line))
            .then(a.callee_id.0.cmp(&b.callee_id.0))
    });
    refs.dedup_by(|a, b| {
        a.caller_path == b.caller_path
            && a.caller_line == b.caller_line
            && a.callee_id == b.callee_id
    });

    Ok((nodes, edges, unresolved, refs))
}

// RFC-027 live IsImplementation lane, Rust — DEFERRED, not implemented here.
//
// Rust's `impl Trait for Type` does not fit the live model TS/Python use, for
// reasons that are structural, not effort:
//   1. The `impl` block is a separate item from the type, so the natural edge
//      source (`struct:Foo`) lives in a different file than the clause. Saving
//      the impl file would not invalidate that edge, so deleting an impl would
//      leave a stale (wrong) edge until the next commit — the exact §8.1 failure.
//   2. Rust's own `impl:Foo` node carries no span and collapses every impl of a
//      type into one node, so it cannot serve as a per-impl source either.
//   3. Rust native Phase B emits no `IsImplementation` edge at all, so a live one
//      has nothing to ratify against (swept every commit) and no oracle for the
//      precision meter.
// Making it correct requires teaching Rust's committed Phase B to emit
// IsImplementation from a properly-spanned per-impl node first — a change to the
// deterministic pipeline, tracked separately.

// ── Cargo.toml dependency graph ───────────────────────────────────────────────

fn extract_cargo_deps(corpus: &str, root: &Path) -> anyhow::Result<(Vec<Node>, Vec<Edge>)> {
    let root_cargo = root.join("Cargo.toml");
    if !root_cargo.exists() {
        return Ok((vec![], vec![]));
    }

    let root_text = std::fs::read_to_string(&root_cargo).context("reading root Cargo.toml")?;
    let root_doc: toml::Value = root_text.parse().context("parsing root Cargo.toml")?;

    // Collect all member Cargo.toml paths
    let cargo_paths: Vec<PathBuf> = if let Some(members) = root_doc
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
    {
        members
            .iter()
            .filter_map(|m| m.as_str())
            .flat_map(|member| {
                if member.contains('*') {
                    // Simple glob: enumerate direct children of prefix dir
                    let prefix = member.trim_end_matches("/*").trim_end_matches('*');
                    let base = root.join(prefix);
                    if let Ok(rd) = std::fs::read_dir(base) {
                        rd.flatten()
                            .map(|e| e.path().join("Cargo.toml"))
                            .filter(|p| p.exists())
                            .collect()
                    } else {
                        vec![]
                    }
                } else {
                    let p = root.join(member).join("Cargo.toml");
                    if p.exists() {
                        vec![p]
                    } else {
                        vec![]
                    }
                }
            })
            .collect()
    } else {
        vec![root_cargo]
    };

    // Parse each Cargo.toml: (pkg_name, rel_cargo_path, dep_names)
    let pkg_data: Vec<(String, String, Vec<String>)> = cargo_paths
        .iter()
        .filter_map(|cargo_path| {
            let text = std::fs::read_to_string(cargo_path).ok()?;
            let doc: toml::Value = text.parse().ok()?;
            let pkg_name = doc
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())?
                .to_string();
            let rel = cargo_path
                .strip_prefix(root)
                .unwrap_or(cargo_path)
                .to_string_lossy()
                .replace('\\', "/");
            let mut dep_names: Vec<String> = Vec::new();
            for section in &["dependencies", "dev-dependencies", "build-dependencies"] {
                if let Some(deps) = doc.get(section).and_then(|d| d.as_table()) {
                    dep_names.extend(deps.keys().cloned());
                }
            }
            Some((pkg_name, rel, dep_names))
        })
        .collect();

    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    for (pkg_name, rel_path, dep_names) in &pkg_data {
        let pkg_node = Node::new(
            VName::new(
                corpus,
                "",
                rel_path.as_str(),
                "rust",
                format!("crate:{pkg_name}"),
            ),
            "crate",
        );
        let pkg_id = pkg_node.id;
        nodes.push(pkg_node);

        for dep_name in dep_names {
            let dep_rel = pkg_data
                .iter()
                .find(|(n, _, _)| n == dep_name)
                .map(|(_, p, _)| p.as_str())
                .unwrap_or("");
            let dep_node = Node::new(
                VName::new(corpus, "", dep_rel, "rust", format!("crate:{dep_name}")),
                "crate",
            );
            let dep_id = dep_node.id;
            nodes.push(dep_node);
            edges.push(Edge::new(pkg_id, dep_id, EdgeKind::Depends));
        }
    }

    Ok((nodes, edges))
}

// ── Call-site extraction ──────────────────────────────────────────────────────

fn extract_file_call_edges(
    corpus: &str,
    abs_path: &Path,
    vname_path: &str,
    language: &tree_sitter::Language,
    query: &Query,
) -> anyhow::Result<(Vec<UnresolvedCall>, Vec<ScipRef>)> {
    let source =
        std::fs::read(abs_path).with_context(|| format!("reading {}", abs_path.display()))?;

    let mut parser = Parser::new();
    parser
        .set_language(language)
        .context("loading Rust grammar")?;
    let tree = match parser.parse(&source, None) {
        Some(t) => t,
        None => return Ok((vec![], vec![])),
    };

    let cap_names: Vec<String> = query
        .capture_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut cursor = QueryCursor::new();
    let mut iter = cursor.matches(query, tree.root_node(), source.as_slice());

    let mut unresolved: Vec<UnresolvedCall> = Vec::new();
    // #299 R1: the Rust extractor no longer guesses method-call callee ids
    // (that produced orphaned edge_sites). Method / associated calls are now
    // emitted as UnresolvedCall and resolved against the real node table by the
    // daemon, so this file-local ScipRef vec stays empty and is kept only for
    // the shared return shape.
    let refs: Vec<ScipRef> = Vec::new();

    while let Some(m) = iter.next() {
        for &cap in m.captures {
            let Some(cap_name) = cap_names.get(cap.index as usize) else {
                continue;
            };
            let Ok(callee_name) = cap.node.utf8_text(source.as_slice()) else {
                continue;
            };
            // Skip short names that are likely noise (closures, etc.)
            if callee_name.len() < 2 {
                continue;
            }

            // 1-based call-site line (issue #299). tree-sitter rows are 0-based.
            let occ_line = cap.node.start_position().row.saturating_add(1) as u32;
            let occ_col = cap.node.start_position().column as u32;

            let Some((caller_fn, caller_impl)) = find_enclosing_fn(cap.node, source.as_slice())
            else {
                continue;
            };

            let caller_id = match &caller_impl {
                Some(t) => VName::new(
                    corpus,
                    "",
                    vname_path,
                    "rust",
                    format!("method:{t}.{caller_fn}"),
                )
                .id(),
                None => VName::new(corpus, "", vname_path, "rust", format!("fn:{caller_fn}")).id(),
            };

            match cap_name.as_str() {
                // #299 R1: a method call `recv.method()` targets the receiver's
                // type, which is not knowable syntactically (the enclosing impl
                // is usually NOT the receiver's type — `zoo.add()` inside
                // `Zoo::announce_all`, `a.describe()` where `a: Box<dyn Animal>`).
                // Guessing `fn:{enclosing_impl}.{method}` produced a callee id
                // that matched no node → orphaned edge_sites. Emit an
                // UnresolvedCall keyed on the bare method name and let the daemon
                // resolve it against the real global node table (exact `fn:method`
                // or unique `fn:Type.method` leaf match), which knows every file.
                "call.method" if !NOISE_NAMES.contains(&callee_name) => {
                    // #529: the receiver expression is bound as a sibling
                    // capture (`call.recv`) on this same query match. Recover
                    // its type when syntactically possible (§4.5) so the
                    // daemon can resolve `fn:T.method` exactly instead of
                    // guessing by unique leaf name; `None` keeps the
                    // pre-#529 leaf-fallback behavior unchanged.
                    let recv_type = m
                        .captures
                        .iter()
                        .find(|c| {
                            cap_names.get(c.index as usize).map(String::as_str) == Some("call.recv")
                        })
                        .and_then(|recv_cap| {
                            let enclosing_fn = enclosing_function_item(recv_cap.node)?;
                            resolve_receiver_type(recv_cap.node, enclosing_fn, source.as_slice())
                        });
                    unresolved.push(UnresolvedCall {
                        src: caller_id,
                        callee_sig: format!("fn:{callee_name}"),
                        alt_callee_sig: None,
                        hint_crate: None,
                        caller_line: occ_line,
                        caller_col: Some(occ_col),
                        is_method_call: true,
                        recv_type,
                    });
                }
                "call.scoped" => {
                    // Extract qualifying segment from the scoped path parent node
                    let qual_raw = cap
                        .node
                        .parent()
                        .and_then(|p| p.child_by_field_name("path"))
                        .and_then(|path_node| path_node.utf8_text(source.as_slice()).ok())
                        .and_then(|t| t.split("::").last().map(str::to_string))
                        .filter(|s| !s.is_empty() && s != callee_name);

                    match qual_raw {
                        Some(ref qual) if qual.starts_with(|c: char| c.is_uppercase()) => {
                            // Uppercase qualifier → associated call `Type::method()`
                            // (e.g. `Zoo::new()`, `Dog::new()`). The type may be defined
                            // in another file, so a path-bound guessed id orphaned.
                            // Emit an UnresolvedCall with the precise qualified sig and
                            // let the daemon match `fn:{Type}.{method}` across all files.
                            if !NOISE_NAMES.contains(&callee_name) {
                                unresolved.push(UnresolvedCall {
                                    src: caller_id,
                                    callee_sig: format!("method:{qual}.{callee_name}"),
                                    alt_callee_sig: None,
                                    hint_crate: None,
                                    caller_line: occ_line,
                                    caller_col: Some(occ_col),
                                    is_method_call: false,
                                    recv_type: None,
                                });
                            }
                        }
                        Some(ref qual) => {
                            // Lowercase qualifier → likely a crate/module path; emit UnresolvedCall
                            if !NOISE_NAMES.contains(&callee_name) {
                                unresolved.push(UnresolvedCall {
                                    src: caller_id,
                                    callee_sig: format!("fn:{callee_name}"),
                                    alt_callee_sig: None,
                                    hint_crate: Some(qual.clone()),
                                    caller_line: occ_line,
                                    caller_col: Some(occ_col),
                                    is_method_call: false,
                                    recv_type: None,
                                });
                            }
                        }
                        None => {
                            // No qualifier found — treat as bare call
                            if !NOISE_NAMES.contains(&callee_name) {
                                unresolved.push(UnresolvedCall {
                                    src: caller_id,
                                    callee_sig: format!("fn:{callee_name}"),
                                    alt_callee_sig: None,
                                    hint_crate: None,
                                    caller_line: occ_line,
                                    caller_col: Some(occ_col),
                                    is_method_call: false,
                                    recv_type: None,
                                });
                            }
                        }
                    }
                }
                // Bare identifier — cannot anchor callee to a file; emit UnresolvedCall.
                // Note: travsr-analysis does not have daemon plumbing, so UnresolvedCalls
                // are returned to the caller but NOT resolved against the store here.
                "call.fn" if !NOISE_NAMES.contains(&callee_name) => {
                    unresolved.push(UnresolvedCall {
                        src: caller_id,
                        callee_sig: format!("fn:{callee_name}"),
                        alt_callee_sig: None,
                        hint_crate: None,
                        caller_line: occ_line,
                        caller_col: Some(occ_col),
                        is_method_call: false,
                        recv_type: None,
                    });
                }
                // #757: field-access reference `x.foo`. The `field_expression`
                // pattern also fires on the receiver of a method call
                // (`x.foo()`), so drop any whose parent is a `call_expression`
                // using it as the callee — that is the `call.method` case above.
                // A field read carries no disambiguating signal beyond its
                // receiver's type, so we only emit when that type is recoverable
                // (`recv_type`); otherwise the daemon would fail closed anyway
                // (§1.5, mirroring the #604/#606 method contract). The daemon
                // resolves the emitted `field:{name}` sig against the exact Phase
                // A node `field:{recv_type}.{name}`.
                "field.name" => {
                    let Some(field_expr) = cap.node.parent() else {
                        continue;
                    };
                    let is_method_callee = field_expr.parent().is_some_and(|p| {
                        p.kind() == "call_expression"
                            && p.child_by_field_name("function").map(|f| f.id())
                                == Some(field_expr.id())
                    });
                    if is_method_callee {
                        continue;
                    }
                    let recv_type = m
                        .captures
                        .iter()
                        .find(|c| {
                            cap_names.get(c.index as usize).map(String::as_str)
                                == Some("field.recv")
                        })
                        .and_then(|recv_cap| {
                            let enclosing_fn = enclosing_function_item(recv_cap.node)?;
                            resolve_receiver_type(recv_cap.node, enclosing_fn, source.as_slice())
                        });
                    if recv_type.is_none() {
                        continue;
                    }
                    unresolved.push(UnresolvedCall {
                        src: caller_id,
                        callee_sig: format!("field:{callee_name}"),
                        alt_callee_sig: None,
                        hint_crate: None,
                        caller_line: occ_line,
                        caller_col: Some(occ_col),
                        is_method_call: false,
                        recv_type,
                    });
                }
                _ => {}
            }
        }
    }

    // #299 R1: recover calls that live inside macro invocations
    // (`println!("{}", a.describe())`). tree-sitter keeps a macro's arguments as
    // an unparsed `token_tree`, so the call-expression query above never sees
    // them — yet these are real reference sites. Scan token_trees for the call
    // shape and emit UnresolvedCalls (daemon unique-match resolution guards the
    // false positives an untyped token scan can produce).
    extract_macro_calls(
        tree.root_node(),
        source.as_slice(),
        corpus,
        vname_path,
        &mut unresolved,
    );

    Ok((unresolved, refs))
}

/// Walk the tree and, inside every macro `token_tree`, recover call sites of the
/// form `name ( … )` — a `name` identifier immediately followed by a nested
/// `token_tree`. Classify by the token preceding `name`: `.` → method call,
/// `::` → associated call (with the qualifier), otherwise a bare function call.
/// Each becomes an [`UnresolvedCall`] resolved later against the real node table.
fn extract_macro_calls(
    node: tree_sitter::Node<'_>,
    source: &[u8],
    corpus: &str,
    vname_path: &str,
    out: &mut Vec<UnresolvedCall>,
) {
    if node.kind() == "token_tree" {
        let mut c = node.walk();
        let children: Vec<tree_sitter::Node<'_>> = node.children(&mut c).collect();
        for i in 1..children.len() {
            // A nested token_tree is a call's argument list.
            if children[i].kind() != "token_tree" {
                continue;
            }
            let name_node = children[i - 1];
            if !matches!(name_node.kind(), "identifier" | "field_identifier") {
                continue;
            }
            let Ok(callee_name) = name_node.utf8_text(source) else {
                continue;
            };
            if callee_name.len() < 2 || NOISE_NAMES.contains(&callee_name) {
                continue;
            }
            let occ_line = name_node.start_position().row.saturating_add(1) as u32;
            let occ_col = name_node.start_position().column as u32;
            let Some((caller_fn, caller_impl)) = find_enclosing_fn(name_node, source) else {
                continue;
            };
            let caller_id = match &caller_impl {
                Some(t) => VName::new(
                    corpus,
                    "",
                    vname_path,
                    "rust",
                    format!("method:{t}.{caller_fn}"),
                )
                .id(),
                None => VName::new(corpus, "", vname_path, "rust", format!("fn:{caller_fn}")).id(),
            };
            // Classify by the token before the name.
            let prev = (i >= 2).then(|| children[i - 2].kind());
            // #521 F3: `.` marks a method-call receiver of unknown type — the
            // resolver must not match it against a bare free function.
            let is_method_call = matches!(prev, Some("."));
            let callee_sig = match prev {
                Some("::") => {
                    // Associated call `Type::method` — recover the qualifier.
                    let qual = (i >= 3)
                        .then(|| children[i - 3])
                        .filter(|n| n.kind() == "identifier")
                        .and_then(|n| n.utf8_text(source).ok());
                    match qual {
                        Some(q) => format!("method:{q}.{callee_name}"),
                        None => format!("fn:{callee_name}"),
                    }
                }
                // `.` (method) or anything else (bare call) → resolve by name.
                _ => format!("fn:{callee_name}"),
            };
            out.push(UnresolvedCall {
                src: caller_id,
                callee_sig,
                alt_callee_sig: None,
                hint_crate: None,
                caller_line: occ_line,
                caller_col: Some(occ_col),
                is_method_call,
                // Macro-token recovery has no receiver AST node to inspect
                // (it scans the raw token stream, not a parsed expression) —
                // recv_type stays None, same as before #529 (unchanged tier).
                recv_type: None,
            });
        }
    }

    let mut c = node.walk();
    for child in node.children(&mut c) {
        extract_macro_calls(child, source, corpus, vname_path, out);
    }
}

/// Walk up from `node` to the nearest enclosing `function_item` node itself
/// (as opposed to [`find_enclosing_fn`], which derives its name/impl-type).
/// Shared with `resolve_receiver_type` (#529), which needs the node to scan
/// the function body for local bindings.
fn enclosing_function_item(node: tree_sitter::Node<'_>) -> Option<tree_sitter::Node<'_>> {
    let mut cur = node.parent()?;
    loop {
        match cur.kind() {
            "function_item" => return Some(cur),
            "source_file" => return None,
            _ => {}
        }
        cur = cur.parent()?;
    }
}

/// Walk up the tree-sitter AST to find the nearest enclosing `function_item`.
/// Returns `(fn_name, Option<impl_type>)` or `None` when outside any function.
fn find_enclosing_fn(
    node: tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<(String, Option<String>)> {
    let fn_node = enclosing_function_item(node)?;
    let fn_name = fn_node
        .child_by_field_name("name")?
        .utf8_text(source)
        .ok()?
        .to_string();
    let impl_type = find_parent_impl_type(fn_node, source);
    Some((fn_name, impl_type))
}

/// Walk up from `node` to find the nearest enclosing type-container name.
/// Returns the implementing type (e.g. `"Worker"` for `impl Trait for Worker`)
/// or, for a default method inside a trait, the trait name (N4c) — matching the
/// Phase A node identity `method:Trait.name`, so caller `src` ids and `self`
/// receiver types stay consistent. Stops at `mod_item` boundaries.
fn find_parent_impl_type(node: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let mut cur = node.parent()?;
    loop {
        match cur.kind() {
            "impl_item" => {
                let type_node = cur.child_by_field_name("type")?;
                return match type_node.kind() {
                    "type_identifier" => type_node.utf8_text(source).ok().map(str::to_string),
                    "generic_type" => (0..type_node.child_count())
                        .filter_map(|i| type_node.child(i as u32))
                        .find(|c| c.kind() == "type_identifier")
                        .and_then(|c| c.utf8_text(source).ok().map(str::to_string)),
                    _ => None,
                };
            }
            // N4c: a default method inside a trait belongs to the trait.
            "trait_item" => {
                return cur
                    .child_by_field_name("name")?
                    .utf8_text(source)
                    .ok()
                    .map(str::to_string);
            }
            "mod_item" => return None,
            _ => {}
        }
        cur = cur.parent()?;
    }
}

// ── Receiver-type resolution (#529) ─────────────────────────────────────────

/// Recover the receiver's type name for `recv.method()`, using only information
/// visible inside the enclosing function. File-local by construction: an
/// incremental single-file reindex must reach the same answer as a full build,
/// since nothing outside `enclosing_fn`'s own text is consulted.
///
/// O(k) in the number of statements in the enclosing fn; each call site walks
/// its own function body once. No cross-file lookup, no store access.
///
/// Accepted, in order (see `docs/plans/issue-529-method-call-receiver-resolution.md`
/// §4.5): `self` receivers resolve via the enclosing `impl`; plain-identifier
/// receivers resolve via the nearest preceding `let`/parameter binding of that
/// name. Anything else — chain/temporary receivers (`x.iter().filter()`),
/// unbound identifiers, closure captures, `match`/`if let` bindings, tuple
/// destructuring — returns `None`, which is exactly today's (pre-#529)
/// behavior. `None` is always the safe answer: the daemon's fallback path is
/// unchanged from before this feature existed.
fn resolve_receiver_type(
    recv: tree_sitter::Node<'_>,
    enclosing_fn: tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<String> {
    match recv.kind() {
        "self" => find_parent_impl_type(enclosing_fn, source),
        "identifier" => {
            let name = recv.utf8_text(source).ok()?;
            nearest_preceding_binding_type(enclosing_fn, source, name, recv.start_byte())
        }
        _ => None,
    }
}

/// Find the type of the nearest `let`/parameter binding of `name` that starts
/// before `before_byte` (the receiver's own position). "Nearest" = greatest
/// start byte among candidates, i.e. the textually closest earlier binding.
///
/// Deliberately flat: this scans the whole function body by source position,
/// not by lexical scope. A binding inside a sibling `if`/`match` arm that
/// happens to sit earlier in the text can shadow the real one — accepted per
/// §4.5 ("shadowing across nested blocks beyond nearest-preceding-in-source-order
/// ... rejected deliberately"). §6.1 R-4 measures how often this produces a
/// wrong (non-`None`) answer before this ships.
fn nearest_preceding_binding_type<'a>(
    enclosing_fn: tree_sitter::Node<'a>,
    source: &[u8],
    name: &str,
    before_byte: usize,
) -> Option<String> {
    let mut best: Option<(usize, tree_sitter::Node<'a>)> = None;

    if let Some(params) = enclosing_fn.child_by_field_name("parameters") {
        let mut c = params.walk();
        for p in params.children(&mut c) {
            if p.kind() != "parameter" {
                continue;
            }
            let Some(pat) = p.child_by_field_name("pattern") else {
                continue;
            };
            if pat.kind() != "identifier" || pat.utf8_text(source) != Ok(name) {
                continue;
            }
            if p.start_byte() < before_byte && best.map_or(true, |(b, _)| p.start_byte() > b) {
                best = Some((p.start_byte(), p));
            }
        }
    }

    if let Some(body) = enclosing_fn.child_by_field_name("body") {
        collect_nearest_let_binding(body, source, name, before_byte, &mut best);
    }

    let (_, node) = best?;
    let raw = match node.kind() {
        "parameter" => extract_receiver_type_name(node.child_by_field_name("type")?, source),
        "let_declaration" => extract_type_from_let(node, source),
        _ => None,
    }?;
    // `Self` inside the binding's own type/constructor text (`let cfg = Self::new();`,
    // `fn f(x: Self)`) means the enclosing impl's type, not a literal type named
    // "Self" — resolve it the same way the `self` receiver already does.
    if raw == "Self" {
        return find_parent_impl_type(enclosing_fn, source);
    }
    // A receiver typed as one of the enclosing function's OWN generic type
    // parameters (`fn f<S: Store>(store: &S)`) is not a concrete type — "S" is
    // syntactically indistinguishable from a real type name, so without this
    // check it would be treated as one (measured via §6.1 R-4: this produced a
    // real wrong-resolution case on `crates/travsr-retrieval/src/ppr.rs`).
    if fn_generic_param_names(enclosing_fn, source).contains(raw.as_str()) {
        return None;
    }
    Some(raw)
}

/// Names declared in `enclosing_fn`'s own `<...>` generic parameter list
/// (excluding lifetimes, which can never be a receiver type). Used to reject
/// a resolved type name that is actually a generic parameter, not a concrete
/// type — see the `Self`/generic-parameter guard in
/// [`nearest_preceding_binding_type`].
fn fn_generic_param_names(
    enclosing_fn: tree_sitter::Node<'_>,
    source: &[u8],
) -> std::collections::HashSet<String> {
    let Some(type_params) = enclosing_fn.child_by_field_name("type_parameters") else {
        return std::collections::HashSet::new();
    };
    let mut names = std::collections::HashSet::new();
    let mut c = type_params.walk();
    for p in type_params.children(&mut c) {
        if p.kind() != "type_parameter" {
            continue;
        }
        if let Some(name_node) = p.child_by_field_name("name") {
            if let Ok(text) = name_node.utf8_text(source) {
                names.insert(text.to_string());
            }
        }
    }
    names
}

/// Recurse through `node` collecting `let_declaration`s that bind `name` and
/// start before `before_byte`, keeping only the one with the greatest start
/// byte (nearest preceding) in `best`. Does not descend into nested
/// `function_item`/`closure_expression` bodies — a binding local to a nested
/// fn or closure is never visible to a receiver outside it, and matching one
/// by source position alone (rather than scope) would be actively wrong
/// rather than just imprecise.
fn collect_nearest_let_binding<'a>(
    node: tree_sitter::Node<'a>,
    source: &[u8],
    name: &str,
    before_byte: usize,
    best: &mut Option<(usize, tree_sitter::Node<'a>)>,
) {
    if node.kind() == "let_declaration" {
        if let Some(pat) = node.child_by_field_name("pattern") {
            if pat.kind() == "identifier"
                && pat.utf8_text(source) == Ok(name)
                && node.start_byte() < before_byte
                && best.map_or(true, |(b, _)| node.start_byte() > b)
            {
                *best = Some((node.start_byte(), node));
            }
        }
    }
    if matches!(node.kind(), "function_item" | "closure_expression") {
        return;
    }
    let mut c = node.walk();
    for child in node.children(&mut c) {
        collect_nearest_let_binding(child, source, name, before_byte, best);
    }
}

/// Smart-pointer / interior-mutability constructors whose `Type::ctor(..)`
/// call does NOT return `Type` itself (`Arc::new(x)` / `Arc::clone(&x)` both
/// return `Arc<InnerType>`, not `Arc`). Excluded from the constructor-call
/// inference below rather than peeled: unlike the `type:`-annotation case
/// (`extract_receiver_type_name`), there is no type argument in the call
/// syntax to peel — the inner type is not visible here at all, so `None` is
/// the only correct answer. Measured via §6.1 R-4: `Arc::clone(&r)` was
/// wrongly resolving to receiver type `"Arc"` on two real call sites
/// (`crates/travsr-store/src/lib.rs`, `crates/travsr-mcp/src/lib.rs`).
const WRAPPER_CONSTRUCTOR_NAMES: &[&str] = &[
    "Option", "Arc", "Box", "Rc", "RefCell", "Mutex", "RwLock", "Cell", "Weak",
];

/// Result/Option adapters that hand back the very value the constructor call
/// produced, so an initializer wrapped in them carries exactly the evidence
/// the bare constructor call does. Closed list on purpose: a name outside it
/// stops the peel in [`peel_result_adapters`] and the initializer falls
/// through to `None`, which is the safe answer.
const TRANSPARENT_RESULT_ADAPTERS: &[&str] = &["unwrap", "expect", "context", "with_context"];

/// Peel `?` and the [`TRANSPARENT_RESULT_ADAPTERS`] off a `let` initializer so
/// `let s = T::open(p).with_context(|| ..)?` reaches the same `T::open(p)`
/// constructor call that `let s = T::open(p)` already resolves through. Each
/// step descends into a child node, so the loop always terminates.
///
/// Nothing else is peeled: `T::open(p).into_inner()` or `T::builder().build()`
/// stop here and resolve to `None`, keeping the inference to shapes whose
/// result type is the constructor's own.
fn peel_result_adapters<'a>(
    mut value: tree_sitter::Node<'a>,
    source: &[u8],
) -> tree_sitter::Node<'a> {
    loop {
        match value.kind() {
            "try_expression" => match value.named_child(0) {
                Some(inner) => value = inner,
                None => return value,
            },
            "call_expression" => {
                let Some(func) = value.child_by_field_name("function") else {
                    return value;
                };
                if func.kind() != "field_expression" {
                    return value;
                }
                let Some(field) = func.child_by_field_name("field") else {
                    return value;
                };
                let Ok(name) = field.utf8_text(source) else {
                    return value;
                };
                if !TRANSPARENT_RESULT_ADAPTERS.contains(&name) {
                    return value;
                }
                match func.child_by_field_name("value") {
                    Some(recv) => value = recv,
                    None => return value,
                }
            }
            _ => return value,
        }
    }
}

/// Resolve a `let_declaration`'s bound type per §4.5 bullet 2: prefer an
/// explicit `type:` annotation; otherwise infer from a capitalized
/// constructor call (`SqliteStore::open(..)`) or struct literal
/// (`SqliteStore { .. }`) on the right-hand side, after peeling any
/// `?`/`unwrap`/`expect`/`context` adapters wrapped around it. Anything else
/// (a plain identifier, an arbitrary method chain, a literal) is `None` —
/// inferring through an arbitrary expression is exactly the unbounded problem
/// #529 rejected doing with a denylist; only these shapes are used.
fn extract_type_from_let(let_decl: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    if let Some(ty) = let_decl.child_by_field_name("type") {
        return extract_receiver_type_name(ty, source);
    }
    let value = peel_result_adapters(let_decl.child_by_field_name("value")?, source);
    match value.kind() {
        "call_expression" => {
            let func = value.child_by_field_name("function")?;
            if func.kind() != "scoped_identifier" {
                return None;
            }
            let path_node = func.child_by_field_name("path")?;
            let path_text = path_node.utf8_text(source).ok()?;
            let seg = path_text.rsplit("::").next().unwrap_or(path_text);
            if seg.is_empty()
                || !seg.starts_with(|c: char| c.is_uppercase())
                || WRAPPER_CONSTRUCTOR_NAMES.contains(&seg)
            {
                return None;
            }
            Some(seg.to_string())
        }
        "struct_expression" => {
            let name = value.child_by_field_name("name")?;
            // Only a plain `Type { .. }` is unambiguous. A scoped name here
            // (`SandboxPolicy::Elevated { .. }`) is syntactically identical
            // whether it names a real nested-module type or (far more
            // commonly) a struct-shaped enum variant — tree-sitter cannot
            // tell these apart, and guessing either segment risks a wrong
            // resolution (§6.1 R-4 measured this exact shape). `None`.
            (name.kind() == "type_identifier")
                .then(|| extract_receiver_type_name(name, source))
                .flatten()
        }
        _ => None,
    }
}

/// Extract a receiver-usable type name from a type AST node: peel `&`/`&mut`
/// wrappers, then peel a *single* `Arc<T>`/`Box<T>`/`Rc<T>` layer down to `T`
/// (these smart pointers `Deref<Target = T>`, so `arc_foo.foo_method()` really
/// dispatches to `T`'s method). `Option<T>` is deliberately NOT peeled: it does
/// not implement `Deref`, so a method call on an `Option` receiver dispatches to
/// `Option`'s own methods (`.map`, `.take`, `.filter`, `.as_ref`, ...), never
/// `T`'s — peeling it would confidently mis-resolve those to `method:T.<name>`.
/// `Vec<T>` and every other generic container are likewise NOT unwrapped: the
/// receiver really is the container (`vec.push(x)` calls `Vec`'s own method, not
/// `T`'s) — see §4.5.
fn extract_receiver_type_name(ty: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let ty = peel_references(ty);
    if ty.kind() == "generic_type" {
        let base = peel_references(ty.child_by_field_name("type")?);
        let base_name = type_identifier_text(base, source)?;
        if matches!(base_name.as_str(), "Arc" | "Box" | "Rc") {
            let args = ty.child_by_field_name("type_arguments")?;
            let inner = peel_references(args.named_child(0)?);
            return type_identifier_text(inner, source);
        }
        return Some(base_name);
    }
    type_identifier_text(ty, source)
}

/// Peel `&`/`&mut` reference wrappers down to the underlying type node.
fn peel_references(mut node: tree_sitter::Node<'_>) -> tree_sitter::Node<'_> {
    while node.kind() == "reference_type" {
        match node.child_by_field_name("type") {
            Some(inner) => node = inner,
            None => break,
        }
    }
    node
}

/// The bare name of a `type_identifier` or `scoped_type_identifier` node
/// (`crate_a::SqliteStore` -> `"SqliteStore"`). `None` for any other shape
/// (tuples, `dyn Trait`, primitives) — those are not resolvable receiver
/// types under this design.
fn type_identifier_text(node: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),
        "scoped_type_identifier" => type_identifier_text(node.child_by_field_name("name")?, source),
        _ => None,
    }
}

// ── File walker ───────────────────────────────────────────────────────────────

/// Collect (abs_path, repo-relative vname_path) for all source files with
/// the given extensions under `root`. Skips `target/`, `node_modules/`, hidden dirs.
pub(crate) fn collect_source_files(root: &Path, exts: &[&str]) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    walk(root, root, exts, &mut out);
    out
}

fn walk(root: &Path, dir: &Path, exts: &[&str], out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if matches!(name, "target" | "node_modules" | ".git") {
                continue;
            }
            walk(root, &path, exts, out);
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if exts.contains(&ext) {
                let vname_path = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((path, vname_path));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn parse_rust(source: &[u8]) -> tree_sitter::Tree {
        let language = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
        let mut parser = Parser::new();
        parser.set_language(&language).unwrap();
        parser.parse(source, None).unwrap()
    }

    /// Run the real `CALL_QUERY` + `resolve_receiver_type` path end-to-end
    /// (#529) against `source`, returning the recovered receiver type for the
    /// `.method_name()` call site. Exercises the exact same query/capture
    /// wiring as `extract_file_call_edges`, not a hand-rolled shortcut.
    fn recv_type_for_call(source: &[u8], method_name: &str) -> Option<String> {
        let language = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
        let query = Query::new(&language, CALL_QUERY).unwrap();
        let tree = parse_rust(source);
        let cap_names: Vec<String> = query
            .capture_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut cursor = QueryCursor::new();
        let mut iter = cursor.matches(&query, tree.root_node(), source);
        while let Some(m) = iter.next() {
            let Some(method_cap) = m.captures.iter().find(|c| {
                cap_names.get(c.index as usize).map(String::as_str) == Some("call.method")
            }) else {
                continue;
            };
            if method_cap.node.utf8_text(source) != Ok(method_name) {
                continue;
            }
            let recv_cap = m.captures.iter().find(|c| {
                cap_names.get(c.index as usize).map(String::as_str) == Some("call.recv")
            })?;
            let enclosing_fn = enclosing_function_item(recv_cap.node)?;
            return resolve_receiver_type(recv_cap.node, enclosing_fn, source);
        }
        None
    }

    /// Drive the real `extract_file_call_edges` path over `source` and return
    /// the field-access `UnresolvedCall`s it emits (`callee_sig` starts with
    /// `field:`). Writes to a unique temp file so the on-disk read path is
    /// exercised exactly as in production, not a hand-rolled shortcut.
    fn field_refs(source: &str) -> Vec<UnresolvedCall> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("travsr_field_test_{}_{}.rs", std::process::id(), n));
        std::fs::write(&path, source).unwrap();
        let language = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
        let query = Query::new(&language, CALL_QUERY).unwrap();
        let (unresolved, _refs) =
            extract_file_call_edges("c", &path, "main.rs", &language, &query).unwrap();
        let _ = std::fs::remove_file(&path);
        unresolved
            .into_iter()
            .filter(|u| u.callee_sig.starts_with("field:"))
            .collect()
    }

    #[test]
    fn field_read_on_self_emits_owner_qualifiable_ref() {
        // `self.bar` read → daemon can resolve `field:Foo.bar` from recv_type.
        let refs = field_refs("impl Foo { fn f(&self) { let _x = self.bar; } }");
        assert_eq!(refs.len(), 1, "exactly one field ref: {refs:?}");
        assert_eq!(refs[0].callee_sig, "field:bar");
        assert_eq!(refs[0].recv_type.as_deref(), Some("Foo"));
        assert!(!refs[0].is_method_call);
    }

    #[test]
    fn method_call_receiver_field_expr_is_not_a_field_ref() {
        // `self.bar()` is a method call, not a field read — no `field:bar`.
        let refs = field_refs("impl Foo { fn f(&self) { self.bar(); } }");
        assert!(
            refs.is_empty(),
            "method call must not emit a field ref: {refs:?}"
        );
    }

    #[test]
    fn field_read_without_recoverable_receiver_is_dropped() {
        // `thing` has no visible binding → recv_type None → fail closed.
        let refs = field_refs("fn f() { let _x = thing.bar; }");
        assert!(
            refs.is_empty(),
            "unresolved receiver must not emit: {refs:?}"
        );
    }

    #[test]
    fn field_read_on_typed_parameter_resolves() {
        let refs = field_refs("fn f(s: &SqliteStore) { let _x = s.conn; }");
        assert_eq!(refs.len(), 1, "{refs:?}");
        assert_eq!(refs[0].callee_sig, "field:conn");
        assert_eq!(refs[0].recv_type.as_deref(), Some("SqliteStore"));
    }

    #[test]
    fn nested_field_read_resolves_inner_and_drops_unknowable_outer() {
        // `self.store.conn`: inner `self.store` → field:store (recv self→Foo);
        // outer `.conn` has a field_expression receiver of unknown type → dropped.
        let refs = field_refs("impl Foo { fn f(&self) { let _x = self.store.conn; } }");
        assert_eq!(
            refs.len(),
            1,
            "only the inner field read resolves: {refs:?}"
        );
        assert_eq!(refs[0].callee_sig, "field:store");
        assert_eq!(refs[0].recv_type.as_deref(), Some("Foo"));
    }

    #[test]
    fn field_write_lhs_is_a_use_site() {
        // A write `self.bar = 1` is a reference too (LHS field_expression).
        let refs = field_refs("impl Foo { fn f(&mut self) { self.bar = 1; } }");
        assert_eq!(refs.len(), 1, "{refs:?}");
        assert_eq!(refs[0].callee_sig, "field:bar");
        assert_eq!(refs[0].recv_type.as_deref(), Some("Foo"));
    }

    #[test]
    fn t1_self_receiver_resolves_to_enclosing_impl_type() {
        let source = b"impl Foo { fn f(&self) { self.bar(); } }";
        assert_eq!(recv_type_for_call(source, "bar"), Some("Foo".to_string()));
    }

    #[test]
    fn t2_let_with_type_annotation_resolves() {
        let source = b"fn f() { let s: SqliteStore = x; s.open(); }";
        assert_eq!(
            recv_type_for_call(source, "open"),
            Some("SqliteStore".to_string())
        );
    }

    #[test]
    fn t2_let_with_constructor_call_resolves() {
        let source = b"fn f() { let s = SqliteStore::open(p); s.close(); }";
        assert_eq!(
            recv_type_for_call(source, "close"),
            Some("SqliteStore".to_string())
        );
    }

    #[test]
    fn t2_let_with_struct_literal_resolves() {
        let source = b"fn f() { let s = SqliteStore { a: 1 }; s.close(); }";
        assert_eq!(
            recv_type_for_call(source, "close"),
            Some("SqliteStore".to_string())
        );
    }

    #[test]
    fn t3_generic_container_keeps_its_own_name_not_the_type_argument() {
        // The daemon, not the extractor, decides HashSet is not a graph type
        // (#529 §4.3 branch 2) — here we only assert the extractor recovers
        // "HashSet", not "NodeId".
        let source = b"fn f() { let visited: HashSet<NodeId> = y; visited.insert(x); }";
        assert_eq!(
            recv_type_for_call(source, "insert"),
            Some("HashSet".to_string())
        );
    }

    #[test]
    fn t3b_smart_pointer_wrapper_peels_to_inner_type() {
        // Arc/Box/Rc are unwrapped one layer (they `Deref<Target = T>`, so
        // `arc_foo.foo_method()` dispatches to `T`) — unlike Vec/HashSet above,
        // which are not.
        let source = b"fn f() { let store: Arc<SqliteStore> = z; store.open(); }";
        assert_eq!(
            recv_type_for_call(source, "open"),
            Some("SqliteStore".to_string())
        );
    }

    #[test]
    fn t3b_option_is_not_peeled_to_its_inner_type() {
        // `Option<T>` does NOT `Deref` to `T`: a method call on an Option
        // receiver dispatches to Option's own methods (`.filter`, `.take`,
        // `.map`, ...), never `T`'s. Peeling it would confidently mis-resolve
        // `opt.filter(..)` to `method:QueryBuilder.filter`. The extractor must
        // report the receiver as `Option`; the daemon (Option not being a graph
        // type) then falls through to leaf-fallback rather than a wrong edge.
        let source = b"fn f() { let q: Option<QueryBuilder> = z; q.filter(p); }";
        assert_eq!(
            recv_type_for_call(source, "filter"),
            Some("Option".to_string())
        );
    }

    // ── §6.1 R-4 regressions: real wrong-resolution sites found by the Phase 0
    // measurement (bench/_529-recv-dump.tsv cross-referenced against this
    // repo's own graph). Each of these produced a wrong non-None answer
    // before the corresponding fix. ─────────────────────────────────────────

    #[test]
    fn r4_wrapper_constructor_call_does_not_resolve_to_the_wrapper_itself() {
        // `Arc::clone(&r)` returns `Arc<T>`, not `Arc` — the call syntax gives
        // no way to see `T`, so this must be None, not "Arc" (real repro:
        // crates/travsr-store/src/lib.rs's embed_readiness_wait_returns_on_mark).
        let source = b"fn f() { let r_bg = Arc::clone(&r); r_bg.mark_ready(); }";
        assert_eq!(recv_type_for_call(source, "mark_ready"), None);
    }

    #[test]
    fn r4_self_keyword_resolves_via_enclosing_impl_not_as_a_literal_type() {
        // Real repro: crates/travsr-plugin-host/src/trust.rs's `from_disk`.
        let source = b"impl TrustConfig { fn from_disk() -> Self { let mut cfg = Self::new(); cfg.trust(x); cfg } }";
        assert_eq!(
            recv_type_for_call(source, "trust"),
            Some("TrustConfig".to_string())
        );
    }

    #[test]
    fn r4_generic_type_parameter_is_not_treated_as_a_concrete_type() {
        // Real repro: crates/travsr-retrieval/src/ppr.rs's
        // build_weighted_subgraph<S: Store>(store: &S, ...).
        let source = b"fn f<S: Store>(store: &S) { store.iter_edges_from_batch(x); }";
        assert_eq!(recv_type_for_call(source, "iter_edges_from_batch"), None);
    }

    #[test]
    fn r4_struct_expression_with_scoped_name_is_ambiguous_and_resolves_to_none() {
        // `SandboxPolicy::Elevated { .. }` is syntactically identical whether
        // "Elevated" is a struct-shaped enum variant (the common case here)
        // or a real nested type — tree-sitter cannot disambiguate, so this
        // must stay None rather than guess either "SandboxPolicy" or
        // "Elevated". Real repro: crates/travsr-plugin-host/src/resolver.rs.
        let source =
            b"fn f() { let policy = SandboxPolicy::Elevated { a: 1 }; policy.validate(); }";
        assert_eq!(recv_type_for_call(source, "validate"), None);
    }

    #[test]
    fn t4_chain_receiver_of_unknowable_type_resolves_to_none() {
        let source = b"fn f() { x.iter().filter(|v| *v); }";
        assert_eq!(recv_type_for_call(source, "filter"), None);
        let source2 = b"fn f() { foo().bar(); }";
        assert_eq!(recv_type_for_call(source2, "bar"), None);
    }

    #[test]
    fn t4b_receiver_with_no_visible_binding_resolves_to_none() {
        let source = b"fn f() { thing.process(); }";
        assert_eq!(recv_type_for_call(source, "process"), None);
    }

    #[test]
    fn t4c_parameter_binding_resolves() {
        let source = b"fn f(store: &mut SqliteStore) { store.open(); }";
        assert_eq!(
            recv_type_for_call(source, "open"),
            Some("SqliteStore".to_string())
        );
    }

    #[test]
    fn t4d_constructor_behind_result_adapters_resolves() {
        // Real repro (#877 follow-up): `travsr-daemon`'s init_repo_with_progress
        // binds its store as
        //   `let mut store = SqliteStore::open(&db_path).with_context(|| ..)?;`
        // Before the adapter peel this returned None, so every `store.method()`
        // in that function was emitted with `recv_type: None` and dropped by the
        // daemon's #604 fail-closed gate, leaving those call edges to the
        // fail-open rust-analyzer LSIF path alone.
        let source =
            b"fn f() { let mut store = SqliteStore::open(&p).with_context(|| x)?; store.begin_staging_tables(); }";
        assert_eq!(
            recv_type_for_call(source, "begin_staging_tables"),
            Some("SqliteStore".to_string())
        );
        for src in [
            &b"fn f() { let s = SqliteStore::open(&p)?; s.node_count(); }"[..],
            &b"fn f() { let s = SqliteStore::open(&p).unwrap(); s.node_count(); }"[..],
            &b"fn f() { let s = SqliteStore::open(&p).expect(\"m\"); s.node_count(); }"[..],
            &b"fn f() { let s = SqliteStore::open(&p).context(\"m\")?; s.node_count(); }"[..],
        ] {
            assert_eq!(
                recv_type_for_call(src, "node_count"),
                Some("SqliteStore".to_string()),
                "adapter-wrapped constructor must resolve: {}",
                String::from_utf8_lossy(src)
            );
        }
    }

    #[test]
    fn t4e_non_transparent_adapter_still_resolves_to_none() {
        // Only adapters that return the constructor's own value are peeled. A
        // chain that transforms the type (`.build()`, `.to_str()`) carries no
        // evidence of the binding's type and must stay None.
        let source = b"fn f() { let s = SqliteStore::builder().build(); s.node_count(); }";
        assert_eq!(recv_type_for_call(source, "node_count"), None);
        let source2 = b"fn f() { let s = PathBuf::from(&p).to_str().unwrap(); s.node_count(); }";
        assert_eq!(recv_type_for_call(source2, "node_count"), None);
    }

    /// Manual measurement tool for #529 Phase 0 (plan §6.1 R-2/R-3/R-4) — not
    /// part of the regular suite (`#[ignore]`). Walks this repo's own `.rs`
    /// files with the real extraction path and dumps every method-call site's
    /// receiver classification to `bench/_529-recv-dump.tsv`, which
    /// `bench/phase-b-edge-audit.py` cross-references against `.travsr/graph.db`
    /// to compute the go/no-go gates before the daemon wiring lands. Not
    /// wired into CI; a one-time (re-runnable) gate-check artifact.
    ///
    /// Run with: `cargo test -p travsr-analysis --release -- --ignored
    /// dump_recv_type_measurement_529 --nocapture`
    #[test]
    #[ignore = "manual measurement tool for #529 Phase 0 gates, not a regression test"]
    fn dump_recv_type_measurement_529() {
        use std::io::Write;

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let out_path = repo_root.join("bench/_529-recv-dump.tsv");

        let language = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
        let query = Query::new(&language, CALL_QUERY).unwrap();
        let cap_names: Vec<String> = query
            .capture_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let files = collect_source_files(repo_root, &["rs"]);
        let mut out = std::fs::File::create(&out_path).expect("create dump file");
        writeln!(
            out,
            "path\tline\tmethod\trecv_kind\trecv_multiline\trecv_type"
        )
        .unwrap();

        for (abs_path, vname_path) in &files {
            let Ok(source) = std::fs::read(abs_path) else {
                continue;
            };
            let mut parser = Parser::new();
            parser.set_language(&language).unwrap();
            let Some(tree) = parser.parse(&source, None) else {
                continue;
            };
            let mut cursor = QueryCursor::new();
            let mut iter = cursor.matches(&query, tree.root_node(), source.as_slice());
            while let Some(m) = iter.next() {
                let Some(method_cap) = m.captures.iter().find(|c| {
                    cap_names.get(c.index as usize).map(String::as_str) == Some("call.method")
                }) else {
                    continue;
                };
                let Some(recv_cap) = m.captures.iter().find(|c| {
                    cap_names.get(c.index as usize).map(String::as_str) == Some("call.recv")
                }) else {
                    continue;
                };
                let Ok(method) = method_cap.node.utf8_text(source.as_slice()) else {
                    continue;
                };
                if method.len() < 2 || NOISE_NAMES.contains(&method) {
                    continue;
                }
                let line = method_cap.node.start_position().row + 1;
                let recv_kind = recv_cap.node.kind();
                let recv_multiline =
                    recv_cap.node.end_position().row < method_cap.node.start_position().row;
                let recv_type = enclosing_function_item(recv_cap.node)
                    .and_then(|f| resolve_receiver_type(recv_cap.node, f, source.as_slice()));
                writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    vname_path,
                    line,
                    method,
                    recv_kind,
                    recv_multiline,
                    recv_type.as_deref().unwrap_or("")
                )
                .unwrap();
            }
        }
        eprintln!("wrote {}", out_path.display());
    }

    #[test]
    fn extract_macro_calls_recovers_method_assoc_and_bare_calls() {
        // #299 R1: tree-sitter does not expose call expressions inside a macro
        // token_tree (`println!(z.describe())`), so the call sites are recovered
        // by scanning the raw tokens. Verify the three classifications and that
        // each site is attributed to its enclosing function with the right line.
        let source = br#"
fn run(z: &Zoo) {
    println!("{}", z.describe());
    println!("{}", Animal::speak());
    println!("{}", greet());
}
"#;
        let tree = parse_rust(source);
        let mut out: Vec<UnresolvedCall> = Vec::new();
        extract_macro_calls(tree.root_node(), source, "c", "main.rs", &mut out);

        let by_sig: HashMap<&str, &UnresolvedCall> =
            out.iter().map(|u| (u.callee_sig.as_str(), u)).collect();

        // Method call `z.describe()` (prev token `.`) → resolve by leaf name.
        let describe = by_sig
            .get("fn:describe")
            .expect("method call recovered from macro");
        assert_eq!(describe.caller_line, 3);
        // Associated call `Animal::speak()` (prev `::`) keeps the qualifier.
        let speak = by_sig
            .get("method:Animal.speak")
            .expect("associated call recovered from macro");
        assert_eq!(speak.caller_line, 4);
        // Bare call `greet()`.
        let greet = by_sig
            .get("fn:greet")
            .expect("bare call recovered from macro");
        assert_eq!(greet.caller_line, 5);

        // RFC-027 #813 P2: the captured byte column points at the callee
        // identifier itself, so the editor resolves the exact occurrence.
        let src_str = std::str::from_utf8(source).unwrap();
        let col_points_at = |u: &UnresolvedCall, leaf: &str| {
            let line = src_str.lines().nth((u.caller_line - 1) as usize).unwrap();
            let col = u.caller_col.expect("occurrence column captured") as usize;
            line[col..].starts_with(leaf)
        };
        assert!(col_points_at(describe, "describe"));
        assert!(col_points_at(speak, "speak"));
        assert!(col_points_at(greet, "greet"));

        // No guessed callee ids: every recovered call is attributed to `run`.
        let run_id = VName::new("c", "", "main.rs", "rust", "fn:run").id();
        for u in &out {
            assert_eq!(u.src, run_id, "call {} not attributed to run", u.callee_sig);
        }
        // `println` itself (the macro name, outside the token_tree) is not a call.
        assert!(!by_sig.contains_key("fn:println"));
    }

    #[test]
    fn extract_macro_calls_skips_noise_and_short_names() {
        // NOISE_NAMES and single-char identifiers inside a macro must not
        // produce edges (they carry no cross-crate signal).
        let source = br#"
fn run() {
    vec![x.clone(), y.new()];
}
"#;
        let tree = parse_rust(source);
        let mut out: Vec<UnresolvedCall> = Vec::new();
        extract_macro_calls(tree.root_node(), source, "c", "main.rs", &mut out);
        assert!(
            out.is_empty(),
            "expected no calls, got {:?}",
            out.iter().map(|u| &u.callee_sig).collect::<Vec<_>>()
        );
    }
}
