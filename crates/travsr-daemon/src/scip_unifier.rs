//! G1 unification pass — bridges indexer SCIP output with the store.
//!
//! For each SCIP Phase B node that can be matched to an existing tree-sitter
//! node by (path, signature candidates, line proximity), this module:
//!   1. Registers a `symbol_aliases` row so future lookups resolve correctly.
//!   2. Patches `ScipRef.callee_id` to point to the unified TS node so that
//!      `write_scip_attributed_batch` emits edges on the right node.
//!
//! RFC-014 §G1. Language-agnostic: the SCIP descriptor suffix grammar is
//! shared by every SCIP indexer, and `candidate_signatures` covers every
//! Phase A parser's signature convention.  Non-SCIP node signatures (builtin
//! native Phase B plugins) yield no descriptor parse and fall through.

use std::collections::{HashMap, HashSet};

use travsr_core::{NodeId, ScipRef};
use travsr_store::SqliteStore;

/// Fallback line window used only when span-containment finds no match (E6:
/// degenerate/zero-width Phase A spans). Positional span-containment is the
/// primary matcher; this bound just keeps the proximity fallback conservative.
const MAX_LINE_DELTA: i64 = 5;

/// Outcome of a [`unify_all`] pass: the SCIP→TS alias map plus the counts that
/// feed the E6 unification miss-rate on the `travsr status` degradation channel.
#[derive(Debug, Default)]
pub struct UnifyOutcome {
    /// SCIP `NodeId` → unified tree-sitter `NodeId` (all kinds, for ref/edge
    /// remapping — may exceed `unified`, which is callable/type-scoped).
    pub alias_map: HashMap<NodeId, NodeId>,
    /// Callable/type (`function`/`class`) SCIP defs that were real unification
    /// candidates (parsed to a name *and* carried a definition line).
    pub attempted: usize,
    /// Callable/type candidates that matched an existing Phase A node.
    pub unified: usize,
    /// SCIP def nodes that are Sorbet synthetic DSL meta-scopes with no twin and
    /// no possibility of one (#780): RSpec `describe`/`context`/`it` blocks whose
    /// *leaf* is the block itself, and a def defined inside a block that still
    /// found no Phase A twin after its unreconcilable container was cleared.
    /// Neither is a reconciliation failure, so both are excluded from
    /// `attempted`/`unified` and dropped by the caller (node plus its inbound
    /// refs/edges) so they stop surviving as orphan duplicates that steal edges.
    /// Defs in tree-sitter-unindexed files (vendored gem code) are NOT dropped:
    /// they are real navigable definitions, only excluded from the counters.
    pub dropped: HashSet<NodeId>,
    /// #825: one detail row per distinct callable/type SCIP symbol counted in
    /// `attempted` but not in `unified` — i.e. exactly the defs behind the E6
    /// miss rate. Carried out so `travsr status` can name the symbols instead of
    /// only reporting a count, and so a dev can see which language/construct is
    /// failing. The miss set is deterministic, so re-running never changes it.
    pub misses: Vec<UnifyMiss>,
}

/// A single unreconciled SCIP definition: a callable/type the compiler defined
/// that could not be matched to its Phase A tree-sitter node, so its references
/// attribute to an orphan duplicate. One row per distinct SCIP symbol (its
/// first-seen occurrence), mirroring how the miss rate counts symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifyMiss {
    pub language: String,
    pub symbol: String,
    pub path: String,
    pub line: u32,
    pub kind: String,
}

/// G1 unification pass for all SCIP-indexed languages.
///
/// Must be called **after** Phase A tree-sitter nodes with `end_line` spans
/// are written to the DB, and **before** `write_scip_attributed_batch`.
///
/// Returns the alias map (SCIP `NodeId` → unified tree-sitter `NodeId`) so
/// the caller can drop the now-duplicate SCIP definition nodes from its
/// Phase B batch and rewrite structural edges onto the unified nodes.
pub fn unify_all(
    store: &mut SqliteStore,
    corpus: &str,
    nodes: &[travsr_core::Node],
    refs: &mut [ScipRef],
) -> UnifyOutcome {
    // Maps SCIP NodeId → unified TS NodeId for ref-patching.
    let mut alias_map: HashMap<NodeId, NodeId> = HashMap::new();
    // #780: SCIP def nodes that are synthetic DSL meta-scopes — no twin exists
    // and none can, so they are neither an attempt nor a miss. Collected here so
    // the caller drops them (and their inbound refs/edges) outright.
    let mut dropped: HashSet<NodeId> = HashSet::new();
    // (scip_symbol, ts_id) pairs, registered in one batch transaction below.
    let mut aliases: Vec<(String, NodeId)> = Vec::new();
    // E6: unification attempts/matches for the miss-rate signal. Scoped to
    // callable/type defs (`function`/`class`) — the only kinds that own
    // ref/call edges and can be stolen by an orphaned SCIP twin. `variable`
    // defs (struct fields, module vars) are excluded: many Phase A parsers
    // don't model them at all (e.g. Go struct fields), so counting them would
    // inflate the rate with benign non-matches on every real repo.
    //
    // Counted per *distinct SCIP symbol*, not per occurrence: a symbol whose
    // definition appears in several files (Obj-C `@interface` in the `.h` and
    // `@implementation` in the `.m`, C/C++ `.h` decl + `.cpp` def) is one
    // definition. It unifies against the file that carries the Phase A node,
    // and the other files' occurrences find no same-file tree-sitter node — a
    // benign duplicate, not a real miss. Tracking symbols (rather than adding
    // to the per-occurrence counters on each file) keeps those from inflating
    // the rate regardless of the order files are visited (#596).
    let mut attempted_syms: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut unified_syms: std::collections::HashSet<&str> = std::collections::HashSet::new();
    // SCIP symbol → the tree-sitter node it unified onto (first match wins).
    // Used after the pass to collapse the *other* files' occurrences of the
    // same symbol (Obj-C `@implementation` twin, C/C++ header-vs-source) onto
    // that node so they are dropped as duplicates instead of surviving as
    // orphan def nodes (#596).
    let mut sym_to_ts: HashMap<&str, NodeId> = HashMap::new();
    // Def nodes whose own file carried no matching tree-sitter node — revisited
    // below to see whether the symbol unified in some other file.
    // Carries the candidate signatures too, so the cross-file rung below can
    // re-query without re-parsing the SCIP descriptor.
    // Carries the kind too: the cross-file rung below is restricted by it, and
    // re-deriving it would mean re-parsing the SCIP descriptor.
    // Carries the path and the container-qualified candidates too, for the
    // overload-collapse rung, which is same-file and ignores line distance.
    #[allow(clippy::type_complexity)]
    let mut unmatched: Vec<(NodeId, &str, Vec<String>, &str, &str, Vec<String>)> = Vec::new();
    // #825: first-seen detail for each callable/type SCIP symbol that becomes an
    // attempt, so the residual misses (`attempted - unified`) can be named in
    // `travsr status`. Keyed by scip symbol to match the per-symbol counters.
    let mut miss_detail: HashMap<&str, UnifyMiss> = HashMap::new();

    // #780: paths the tree-sitter parser actually indexed. A SCIP def in a file
    // absent from this set is one only the SCIP tool saw — gitignored vendored
    // code (scip-ruby indexes `vendor/bundle`, tree-sitter skips it) — and can
    // never reconcile, so it is excluded from the miss counters and dropped
    // rather than counted as a failure or kept as an edge-stealing orphan.
    let indexed_paths = store.phase_a_indexed_paths(corpus).unwrap_or_default();

    for node in nodes {
        let scip_sym = travsr_indexer::scip_unifier::scip_symbol_from_sig(&node.vname.signature);
        // Primary: SCIP descriptor grammar (go/java/ruby/c#/c/c++/… + rust/ts/py
        // LSIF). Fallback: bespoke sidecars (kotlin/swift) whose signatures are
        // Phase-A-style (`fn:Container.name`, `swift::Container.name`) and never
        // parse as SCIP. The fallback is gated to those languages so native
        // Phase A/rust nodes — whose signatures look identical — are never
        // re-unified against themselves.
        let (parsed, is_scip) = match travsr_indexer::scip_unifier::scip_name_kind(scip_sym) {
            Some(p) => (p, true),
            // Scala's sidecar reads SemanticDB, whose symbols are a bare
            // descriptor chain with none of SCIP's `<scheme> <mgr> <pkg>
            // <version>` prefix, so `scip_name_kind` rejects all of them. Parsed
            // as SCIP-shaped rather than Phase-A-shaped, because the descriptor
            // grammar is SCIP's: `…/Parsers#phrase().`.
            None if node.vname.language.as_str() == "scala" => {
                match travsr_indexer::scip_unifier::semanticdb_name_kind(&node.vname.signature) {
                    Some(p) => (p, true),
                    None => continue,
                }
            }
            None if matches!(node.vname.language.as_str(), "kotlin" | "swift" | "dart") => {
                match travsr_indexer::scip_unifier::native_name_kind(
                    &node.vname.signature,
                    &node.kind,
                ) {
                    Some(p) => (p, false),
                    None => continue,
                }
            }
            None => continue,
        };
        // #780: Sorbet models RSpec `describe`/`context`/`it` blocks as singleton
        // scopes and emits SCIP defs for them. When the *leaf* itself is the
        // block (`<it 'does x'>` as the def's own name) tree-sitter (correctly)
        // sees a method call with a block, not a definition, so no Phase A twin
        // exists or can — in ANY file, so this is checked before the
        // unindexed-path rule below. Drop it so it stops orphaning as a duplicate
        // that steals spec-file reference edges (~70% of #780's headline rate).
        // (The *container*-only block case — a real `class Helper` / `def helper`
        // defined inside `describe 'Foo' do … end` — is handled after the path
        // check: Phase A emits an unqualified twin, so it can reconcile.)
        if travsr_indexer::scip_unifier::is_dsl_scope_leaf(&parsed) {
            dropped.insert(node.id);
            continue;
        }
        // #780: a SCIP def in a file the tree-sitter parser never indexed
        // (gitignored vendored code that scip-ruby indexes anyway) has no twin
        // and none can exist — it is not a reconciliation failure. Exclude it
        // from the counters but KEEP the node: it is a real definition at a real
        // on-disk file (`vendor/bundle/.../rake/task.rb`), and calls from app
        // code into gem code are exactly the cross-boundary edges the graph is
        // most useful for. Dropping it would take every inbound `ScipRef` with
        // it. Native sidecar nodes (kotlin/swift/dart) are never path-excluded:
        // their twins live in the same indexed sources.
        if is_scip && !indexed_paths.contains(&node.vname.path) {
            continue;
        }
        // Container-only DSL block: Phase A DOES emit a twin, unqualified,
        // because a block is not a `method_container`. Clearing the
        // unreconcilable container lets the def unify onto that twin. It is
        // counted only if it unifies (below), so recovering it can never regress
        // the reported miss rate.
        let dsl_contained = travsr_indexer::scip_unifier::is_dsl_scope_container(&parsed);
        let parsed = if dsl_contained {
            travsr_indexer::scip_unifier::ScipName {
                container: None,
                ..parsed
            }
        } else {
            parsed
        };
        // No definition line means line-proximity matching is meaningless —
        // unwrapping to 0 would let any same-named node on lines 1..=5 of the
        // file match wrongly. Skip instead.
        let Some(line) = node.line else {
            continue;
        };
        let candidates = travsr_indexer::scip_unifier::candidate_signatures(&parsed);
        let scip_line = line as i64;
        let is_callable_type = matches!(parsed.kind, "function" | "class");
        // A normal callable/type def is an attempt up front — a miss raises the
        // rate. A DSL-contained def is credited only when it unifies (in the
        // match arm), so its recovery cannot push the miss rate up.
        if is_callable_type && !dsl_contained {
            attempted_syms.insert(scip_sym);
            miss_detail.entry(scip_sym).or_insert_with(|| UnifyMiss {
                language: node.vname.language.clone(),
                // Readable `Container.name` (or bare `name`) rather than the raw
                // SCIP moniker — enough for a dev to spot the construct/language.
                symbol: match parsed.container {
                    Some(c) => format!("{c}.{}", parsed.name),
                    None => parsed.name.to_string(),
                },
                path: node.vname.path.clone(),
                line,
                kind: parsed.kind.to_string(),
            });
        }

        match store.find_ts_node_for_unification(
            corpus,
            &node.vname.path,
            &candidates,
            scip_line,
            MAX_LINE_DELTA,
        ) {
            Ok(Some(ts_id)) => {
                aliases.push((scip_sym.to_string(), ts_id));
                alias_map.insert(node.id, ts_id);
                sym_to_ts.entry(scip_sym).or_insert(ts_id);
                if is_callable_type {
                    // A DSL-contained def enters `attempted` only now, on the
                    // success path, so the pair stays metric-neutral.
                    attempted_syms.insert(scip_sym);
                    unified_syms.insert(scip_sym);
                }
                tracing::trace!(symbol = %scip_sym, ?ts_id, "G1: unified");
            }
            // A DSL-contained def that still misses after the container clear has
            // no reconcilable twin — drop it un-counted rather than counting a
            // synthetic miss or letting the cross-file rungs key on its
            // block-qualified symbol.
            Ok(None) if dsl_contained => {
                dropped.insert(node.id);
            }
            Ok(None) => unmatched.push((
                node.id,
                scip_sym,
                candidates,
                parsed.kind,
                node.vname.path.as_str(),
                travsr_indexer::scip_unifier::overload_collapse_signatures(&parsed),
            )),
            Err(e) => tracing::warn!(symbol = %scip_sym, "G1: DB lookup: {e:#}"),
        }
    }

    // Cross-file duplicate collapse: an unmatched def whose symbol unified in a
    // different file is the same definition (Obj-C interface/implementation,
    // C/C++ header/source). Alias it onto that node so it is dropped as a
    // duplicate and its edges/refs rewrite onto the real node, and credit its
    // symbol as unified so the miss-rate does not penalize the benign twin.
    for (node_id, sym, candidates, kind, path, overload_sigs) in unmatched {
        // Rung 1: the symbol unified in another file, so this occurrence is
        // the benign twin.
        if let Some(&ts_id) = sym_to_ts.get(sym) {
            aliases.push((sym.to_string(), ts_id));
            alias_map.insert(node_id, ts_id);
            // Only callable/type symbols feed the miss-rate; crediting a
            // `variable` twin would let `unified` exceed `attempted`.
            if attempted_syms.contains(sym) {
                unified_syms.insert(sym);
            }
            continue;
        }

        // Rung 1b: the SCIP def is one overload of a method group whose Phase A
        // node is shared by every overload. Phase A signatures carry no
        // parameter list, so `C#n().`, `C#n(+1).`, ... all belong to the single
        // `method:C.n` node anchored at the first overload's span, and every
        // later overload misses both span-containment and the +/-5 window above.
        // Resolving that by same-file uniqueness on the *container-qualified*
        // signature is exact, not heuristic: one match means one method group.
        //
        // Measured on the C# fixture (commandlineparser): constructors are the
        // dominant case, 113 of the 120 unreconciled callable defs. The first
        // overload of `OptionAttribute` carried 0 ref/call edges while overloads
        // +1..+4 carried all 103, so the candidate fix alone recovered nothing;
        // this rung is what makes those edges reachable from the type.
        //
        // Placed before rung 2: a same-file qualified match is stronger evidence
        // than rung 2's corpus-wide uniqueness, and it is checked first so the
        // wider rung never gets to answer a question this one already can.
        if !overload_sigs.is_empty() {
            match store.find_unique_ts_node_in_file(corpus, path, &overload_sigs) {
                Ok(Some(ts_id)) => {
                    aliases.push((sym.to_string(), ts_id));
                    alias_map.insert(node_id, ts_id);
                    sym_to_ts.entry(sym).or_insert(ts_id);
                    if attempted_syms.contains(sym) {
                        unified_syms.insert(sym);
                    }
                    tracing::trace!(symbol = %sym, ?ts_id, "G1: unified onto overload group");
                    continue;
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(symbol = %sym, "G1: overload lookup: {e:#}"),
            }
        }

        // Rung 2: the declaration is the *only* Phase A node and it lives in
        // another file, so rung 1 has nothing to key on. This is the C/C++
        // out-of-line member definition: `widget.h` declares `Widget::draw`,
        // `widget.cpp` defines it, and Phase A anchored the header. Without
        // this the SCIP definition survives as an orphan and takes every
        // ref/call edge with it, so `travsr references draw` answers zero
        // while the edges exist and point somewhere unreachable.
        //
        // Guarded on uniqueness rather than on position: a declaration's line
        // says nothing about its definition's, and more than one candidate
        // means the name is ambiguous in this repo (two same-named `static`
        // functions in different translation units are different functions).
        //
        // Restricted to callables and types (#708 review). `candidate_signatures`
        // qualifies those by container where it can (`method:Widget.draw`), but
        // a `variable` yields only bare `var:name` / `const:name` /
        // `static:name`. Names like `count`, `size`, `buf` are common enough
        // that an unrelated `static int count;` elsewhere in the repo is often
        // the single other match, and uniqueness cannot tell "the only match"
        // from "the right match": it would alias the definition onto an
        // unrelated node and corrupt its ref/call edges with no error at all.
        if !matches!(kind, "function" | "class") {
            continue;
        }

        // A same-file exclusion was tried here and removed: the #708 review
        // suggested this rung should not overturn rung 1's positional
        // rejection, but rung 1 rejects the *legitimate* same-file case too.
        // `Box<T>::unwrap` is declared inside the class and defined out of
        // line six lines later, past the +/-5 window, and excluding it
        // reintroduced the orphan this rung exists to prevent. For a callable
        // whose name is unique in the whole corpus, distance is not evidence
        // of a different symbol; it is what an out-of-line definition looks
        // like. The kind restriction above is what addresses the risk the
        // review actually described.
        match store.find_unique_ts_node_across_files(corpus, &candidates) {
            Ok(Some(ts_id)) => {
                aliases.push((sym.to_string(), ts_id));
                alias_map.insert(node_id, ts_id);
                sym_to_ts.entry(sym).or_insert(ts_id);
                if attempted_syms.contains(sym) {
                    unified_syms.insert(sym);
                }
                tracing::trace!(symbol = %sym, ?ts_id, "G1: unified across files");
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(symbol = %sym, "G1: cross-file lookup: {e:#}"),
        }
    }

    let attempted = attempted_syms.len();
    let unified = unified_syms.len();
    // #825: the residual misses are exactly the attempted symbols that never
    // unified. Emit one detail row each (sorted by path:line for a stable list)
    // so `travsr status` can name them rather than only counting them.
    let mut misses: Vec<UnifyMiss> = attempted_syms
        .difference(&unified_syms)
        .filter_map(|s| miss_detail.get(s).cloned())
        .collect();
    misses.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));

    if let Err(e) = store.register_symbol_aliases(&aliases) {
        tracing::warn!("G1: register_symbol_aliases batch: {e:#}");
    }

    for r in refs.iter_mut() {
        if let Some(&ts_id) = alias_map.get(&r.callee_id) {
            r.callee_id = ts_id;
        }
    }

    // #780: report the excluded-outright count alongside the attempt/miss
    // figures so a mass silent exclusion (e.g. Phase A having indexed nothing,
    // so every DSL-contained def fails to unify and is dropped) is
    // distinguishable from a genuinely low miss rate on the same debug line.
    tracing::debug!(
        aliased = alias_map.len(),
        callable_unified = unified,
        callable_attempted = attempted,
        dropped = dropped.len(),
        total = nodes.len(),
        "G1: unification complete"
    );
    UnifyOutcome {
        unified,
        attempted,
        alias_map,
        dropped,
        misses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use travsr_core::{Node, VName};

    fn scip_node(path: &str, symbol: &str, line: u32) -> Node {
        // scip-reader packs def signatures as `scip:{rel_path}:{symbol}`.
        let sig = format!("scip:{path}:{symbol}");
        Node::new(VName::new("c", "main", path, "ruby", &sig), "definition").with_line(line)
    }

    #[test]
    fn dsl_scopes_excluded_and_dropped_real_method_counted() {
        // #780: a real `Class#method().` def unifies onto its Phase A twin and
        // counts toward the miss rate; a Sorbet RSpec DSL block def is neither
        // counted nor written — it is dropped so it cannot steal ref edges.
        let mut store = SqliteStore::open_in_memory().unwrap();
        // Phase A twin for the real accessor (as emitted by the Ruby attr_* /
        // method captures), same path + line as the SCIP def.
        let ts = Node::new(
            VName::new(
                "c",
                "main",
                "supply/lib/supply/generated_universal_apk.rb",
                "ruby",
                "method:GeneratedUniversalApk.package_name",
            ),
            "method",
        )
        .with_line(4)
        .with_end_line(4);
        store
            .write_phase_b_batch(std::slice::from_ref(&ts), &[], "scip")
            .unwrap();

        let real = scip_node(
            "supply/lib/supply/generated_universal_apk.rb",
            "scip-ruby gem fastlane 0.0.0 Supply#GeneratedUniversalApk#package_name().",
            4,
        );
        let dsl_method = scip_node(
            "fastlane/spec/actions_specs/carthage_spec.rb",
            "scip-ruby gem fastlane 0.0.0 `<describe 'Fastlane'>`#`<it 'sets the platform to iOS'>`().",
            7,
        );
        let dsl_type = scip_node(
            "match/spec/setup_spec.rb",
            "scip-ruby gem fastlane 0.0.0 `<describe 'Match'>`#`<describe 'Setup'>`#",
            1,
        );

        let nodes = vec![real.clone(), dsl_method.clone(), dsl_type.clone()];
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &nodes, &mut refs);

        assert_eq!(out.attempted, 1, "only the real Class#method is an attempt");
        assert_eq!(out.unified, 1, "the real method unifies onto its twin");
        assert_eq!(out.alias_map.get(&real.id), Some(&ts.id));
        assert!(out.dropped.contains(&dsl_method.id), "DSL block dropped");
        assert!(out.dropped.contains(&dsl_type.id), "DSL type block dropped");
        assert!(
            !out.dropped.contains(&real.id),
            "the real method must not be dropped"
        );
    }

    #[test]
    fn a_real_miss_is_named_in_the_outcome() {
        // #825: an indexed-file callable def with no Phase A twin is a real miss.
        // It must appear in `out.misses` with its language, kind, readable
        // `Container.name`, and path:line, so `travsr status` can name it instead
        // of only counting it.
        let mut store = SqliteStore::open_in_memory().unwrap();
        // A Phase A twin at lib/app.rb makes that path "indexed"; the orphan def
        // lives in the same file but has no twin of its own.
        let ts = Node::new(
            VName::new("c", "main", "lib/app.rb", "ruby", "method:App.run"),
            "method",
        )
        .with_line(2)
        .with_end_line(2);
        store
            .write_phase_b_batch(std::slice::from_ref(&ts), &[], "scip")
            .unwrap();

        let good = scip_node("lib/app.rb", "scip-ruby gem g 0.0.0 App#run().", 2);
        let orphan = scip_node("lib/app.rb", "scip-ruby gem g 0.0.0 App#missing().", 99);

        let nodes = vec![good.clone(), orphan.clone()];
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &nodes, &mut refs);

        assert_eq!(out.attempted, 2, "both callable defs are attempts");
        assert_eq!(out.unified, 1, "only App#run unifies onto its twin");
        assert_eq!(out.misses.len(), 1, "exactly the orphan is a named miss");
        let m = &out.misses[0];
        assert_eq!(m.language, "ruby");
        assert_eq!(m.kind, "function");
        assert_eq!(m.symbol, "App.missing");
        assert_eq!(m.path, "lib/app.rb");
        assert_eq!(m.line, 99);
    }

    #[test]
    fn a_fully_unified_pass_reports_no_misses() {
        // The list must be empty when nothing misses, so status prints no stray
        // rows.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let ts = Node::new(
            VName::new("c", "main", "lib/app.rb", "ruby", "method:App.run"),
            "method",
        )
        .with_line(2)
        .with_end_line(2);
        store
            .write_phase_b_batch(std::slice::from_ref(&ts), &[], "scip")
            .unwrap();
        let good = scip_node("lib/app.rb", "scip-ruby gem g 0.0.0 App#run().", 2);
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &[good], &mut refs);
        assert_eq!(out.unified, out.attempted);
        assert!(out.misses.is_empty(), "no misses => empty list");
    }

    #[test]
    fn scip_def_in_unindexed_file_is_kept_not_counted() {
        // #780: scip-ruby indexes gitignored vendored code the tree-sitter parser
        // skips, so those files hold only SCIP defs and no Phase A node. Such a
        // def can never reconcile, so it is excluded from the miss rate — but it
        // is a real navigable definition (calls from app code into gem code are
        // exactly the cross-boundary edges the graph is most useful for), so it
        // is KEPT, not dropped.
        let mut store = SqliteStore::open_in_memory().unwrap();
        // An indexed app file with a real Phase A twin (so its path is "indexed").
        let ts = Node::new(
            VName::new("c", "main", "lib/app.rb", "ruby", "method:App.run"),
            "method",
        )
        .with_line(2)
        .with_end_line(2);
        store
            .write_phase_b_batch(std::slice::from_ref(&ts), &[], "scip")
            .unwrap();

        let app = scip_node("lib/app.rb", "scip-ruby gem g 0.0.0 App#run().", 2);
        // A vendored def whose file has NO Phase A node at all.
        let vendored = scip_node(
            "vendor/bundle/ruby/3.4.0/gems/rake/lib/rake/task.rb",
            "scip-ruby gem g 0.0.0 Rake#Task#invoke().",
            10,
        );

        let nodes = vec![app.clone(), vendored.clone()];
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &nodes, &mut refs);

        assert_eq!(out.attempted, 1, "only the indexed-file def is an attempt");
        assert_eq!(out.unified, 1);
        assert!(
            !out.dropped.contains(&vendored.id),
            "def in an unindexed (vendored) file must be kept, not dropped"
        );
        assert!(
            !out.alias_map.contains_key(&vendored.id),
            "the vendored def has no twin, so it is not aliased away either"
        );
        assert!(!out.dropped.contains(&app.id));
    }

    #[test]
    fn real_def_inside_describe_block_unifies_not_dropped() {
        // #780 defect 1: a spec routinely defines helpers inside a `describe`
        // block. scip-ruby qualifies them by the block (`<describe 'Foo'>#…`),
        // but Phase A emits an unqualified twin (a block is not a
        // `method_container`). Only the container is a DSL scope, so clearing it
        // lets the def reconcile onto the real twin instead of being deleted with
        // every reference to it.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let spec = "fastlane/spec/foo_spec.rb";
        // Phase A twins: an unqualified helper method and a helper class, both
        // inside the `describe` block, on the lines scip-ruby reports them.
        let helper_fn = Node::new(
            VName::new("c", "main", spec, "ruby", "fn:helper"),
            "function",
        )
        .with_line(4)
        .with_end_line(4);
        let helper_class = Node::new(
            VName::new("c", "main", spec, "ruby", "class:Helper"),
            "class",
        )
        .with_line(2)
        .with_end_line(2);
        store
            .write_phase_b_batch(&[helper_fn.clone(), helper_class.clone()], &[], "scip")
            .unwrap();

        let scip_fn = scip_node(
            spec,
            "scip-ruby gem fastlane 0.0.0 `<describe 'Foo'>`#`helper`().",
            4,
        );
        let scip_class = scip_node(
            spec,
            "scip-ruby gem fastlane 0.0.0 `<describe 'Foo'>`#Helper#",
            2,
        );

        let nodes = vec![scip_fn.clone(), scip_class.clone()];
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &nodes, &mut refs);

        assert_eq!(
            out.alias_map.get(&scip_fn.id),
            Some(&helper_fn.id),
            "describe-block helper method must unify onto its unqualified twin"
        );
        assert_eq!(
            out.alias_map.get(&scip_class.id),
            Some(&helper_class.id),
            "describe-block helper class must unify onto its twin"
        );
        assert!(
            !out.dropped.contains(&scip_fn.id) && !out.dropped.contains(&scip_class.id),
            "recovered defs must not be dropped"
        );
        assert_eq!(out.attempted, 2, "both count once they have unified");
        assert_eq!(out.unified, 2);
    }

    #[test]
    fn unreconcilable_describe_block_def_is_dropped_not_counted() {
        // A def qualified only by a `describe` block that has NO Phase A twin
        // (e.g. metaprogramming the parser cannot see) still misses after the
        // container clear. It is dropped un-counted, so it neither steals edges
        // nor regresses the miss rate.
        let mut store = SqliteStore::open_in_memory().unwrap();
        let spec = "fastlane/spec/bar_spec.rb";
        // An unrelated indexed twin so the path is "indexed".
        let other = Node::new(
            VName::new("c", "main", spec, "ruby", "fn:other"),
            "function",
        )
        .with_line(1)
        .with_end_line(1);
        store
            .write_phase_b_batch(std::slice::from_ref(&other), &[], "scip")
            .unwrap();

        let orphan = scip_node(
            spec,
            "scip-ruby gem fastlane 0.0.0 `<describe 'Bar'>`#`ghost`().",
            9,
        );
        let nodes = vec![orphan.clone()];
        let mut refs: Vec<ScipRef> = Vec::new();
        let out = unify_all(&mut store, "c", &nodes, &mut refs);

        assert!(out.dropped.contains(&orphan.id), "no twin -> dropped");
        assert_eq!(out.attempted, 0, "a DSL-contained miss is not an attempt");
        assert_eq!(out.unified, 0);
    }
}
