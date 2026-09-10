//! Config-driven Phase A parser for all generic languages.
//!
//! All Phase A work is structurally identical across languages: load a grammar,
//! run tree-sitter queries, map capture names to node kinds. `LanguageConfig`
//! expresses that as static data so adding a new language requires only a
//! new module with a `CONFIG` constant — no new Rust logic.
//!
//! Callers in `travsr-plugin-host` may cache the compiled `tree_sitter::Query`
//! internally (via `GenericTreeSitterPlugin`) to avoid re-compiling on every
//! file. Direct callers (e.g. per-language `parse()` wrappers) may compile
//! fresh; query compilation is fast (< 1 µs) and acceptable per-file.

use std::path::Path;

use anyhow::Context as _;
use streaming_iterator::StreamingIterator as _;
use travsr_core::{Edge, EdgeKind, Language, Node, VName};
use tree_sitter::{Parser, Query, QueryCursor};

use crate::ParseOutput;

const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024; // 10 MB

/// A complete Phase A language definition expressed as static data.
/// All fields are `'static` so configs can be declared as `const`.
pub struct LanguageConfig {
    pub language: Language,
    pub extensions: &'static [&'static str],
    /// tree-sitter query string compiled at first use.
    pub queries: &'static str,
    /// Maps `(capture_name, node_kind, signature_prefix)`.
    ///
    /// For regular nodes: `sig = "{prefix}:{captured_text}"`.
    /// Special prefix `"import"` → uses the full node text, strips the leading
    /// keyword (`import`, `use`, `require`, `using`) and trailing `;`.
    /// A prefix of the form `"import:<scheme>"` (e.g. `"import:gem"`) behaves
    /// identically but tags the signature with `<scheme>` so a language config
    /// can distinguish two import forms captured under different tree-sitter
    /// patterns (e.g. Ruby `require` vs `require_relative`, #614) without any
    /// change to this parser's shared logic.
    pub capture_kinds: &'static [(&'static str, &'static str, &'static str)],
    /// AST node kinds that enclose their members (methods and fields) as a
    /// type/namespace, each paired with the signature prefix that container is
    /// itself captured under. When a `fn`- or `field`-prefixed definition
    /// capture is nested inside one of these, it is emitted as
    /// `method:{ContainerName}.{leaf}` / `field:{ContainerName}.{leaf}` and its
    /// `DefinesBinding` edge is parented to the container node instead of the
    /// file (N1 collision fix + N3 containment). Empty ⇒ the container has no
    /// methods; a language with fields but no methods (e.g. C `struct`) still
    /// lists its type node here so field captures qualify (#757).
    pub method_containers: &'static [(&'static str, &'static str)],
    /// AST node kinds that represent a full definition *with its body* (N2).
    /// When a name capture is nested below its declaration (C
    /// `function_declarator`, C++ inline methods, Obj-C C-functions), a single
    /// hop to `name.parent()` ends the span at the signature line and excludes
    /// the body, so call sites inside fail span-containment and mis-attribute
    /// to the file node. The span end is instead taken from the nearest
    /// ancestor whose kind is listed here. Empty ⇒ the 1-hop span is already
    /// correct for this grammar (name is a direct child of its declaration).
    pub decl_kinds: &'static [&'static str],
    /// Optional post-parse expansion hook, run once after the capture pass with
    /// the parsed tree and source so a language can synthesize nodes the shared
    /// capture pipeline cannot express. Used by Ruby (#780) to expand
    /// `attr_accessor`/`attr_reader`/`attr_writer` macros into their reader/writer
    /// accessor method nodes (one macro → N synthetic names with a derived `=`
    /// writer, which no single capture can produce). `None` for every language
    /// whose definitions are all direct captures.
    pub post_parse: Option<PostParseHook>,
    /// N4d refiners for grammars that fold several type kinds into one AST node
    /// with no distinguishing field (unlike Swift's `declaration_kind`).
    /// tree-sitter-kotlin-ng folds `class`/`interface`/`enum class` into
    /// `class_declaration`, distinguished by a signal node: the `interface`
    /// keyword (a direct child) or the `enum` class-modifier token (inside the
    /// `modifiers` subtree). A query-only split would regress bare/constructor
    /// classes, so the broad query stays and the kind is refined in Rust
    /// (`refine_type`). Empty ⇒ no folding to refine (every non-Kotlin config).
    pub type_refinements: &'static [TypeRefinement],
    /// Optional override for the name text a definition capture contributes to
    /// its signature, consulted in place of the captured node's own text.
    ///
    /// The shared pipeline assumes a definition's name is exactly the text of
    /// one captured node. Objective-C breaks that assumption: a selector is
    /// spelled across several sibling keyword identifiers
    /// (`policyWithPinningMode:withPinnedCertificates:`), only the first of
    /// which a single capture can name. Capturing every keyword instead would
    /// emit one node per part, so the name has to be rebuilt in Rust from the
    /// captured node's siblings.
    ///
    /// `Some(name)` replaces the captured text; `None` keeps it. `None` for
    /// every language whose names are a single token.
    pub name_hook: Option<NameHook>,
    /// Returns the tree-sitter grammar for this language.
    /// Stored as a function pointer so `LanguageConfig` is `const`-constructible
    /// (tree-sitter `Language` itself is not directly `const`-constructible).
    pub get_grammar: fn() -> tree_sitter::Language,
}

/// Rebuilds the name a definition capture contributes to its signature
/// (see [`LanguageConfig::name_hook`]).
///
/// Receives the captured node, the signature prefix it was captured under
/// (`"fn"`, `"class"`, ...) and the file bytes. Returning `None` leaves the
/// captured node's own text in place.
pub type NameHook = fn(cap: tree_sitter::Node, sig_prefix: &str, source: &[u8]) -> Option<String>;

/// A language-specific post-parse expansion pass (see [`LanguageConfig::post_parse`]).
/// Receives the parsed tree/source via [`PostParseCtx`] and appends synthetic
/// nodes and their containment edges to the accumulating output.
pub type PostParseHook = fn(&PostParseCtx<'_>, &mut Vec<Node>, &mut Vec<Edge>);

/// Context handed to a [`PostParseHook`]: everything it needs to build VNames and
/// containment edges consistent with the capture pass that ran before it.
pub struct PostParseCtx<'a> {
    /// Root of the parsed tree.
    pub root: tree_sitter::Node<'a>,
    /// File bytes (for `utf8_text` on captured nodes).
    pub source: &'a [u8],
    /// The corpus this file belongs to.
    pub corpus: &'a str,
    /// VName path of the file being parsed.
    pub vname_path: &'a str,
    /// Language string (`config.language.as_str()`), for VName construction.
    pub lang: &'a str,
    /// The file node's id, for edges parented to the file.
    pub file_id: travsr_core::NodeId,
    /// The parser config, so the hook can reuse `method_containers` /
    /// `type_refinements` (e.g. via [`enclosing_container`]).
    pub config: &'a LanguageConfig,
}

/// N4d: refine a folded type declaration's kind by the presence of a signal
/// node (`has_child_kind`) among the decl's direct children OR inside its
/// `modifiers` subtree. The modifiers subtree is scanned because grammars put
/// discriminating class modifiers there (Kotlin `enum` is
/// `class_declaration > modifiers > class_modifier > enum`), while a keyword
/// like `interface` is a direct child. The declaration *body* is deliberately
/// never scanned — a signal there could belong to a nested type. Applied both
/// to the type node's own emission and to the containment prefix of methods
/// nested inside it.
pub struct TypeRefinement {
    /// The folded declaration node kind, e.g. `"class_declaration"`.
    pub decl_kind: &'static str,
    /// A signal node kind (direct child or modifier token) whose presence
    /// identifies the refined kind, e.g. `"interface"` / `"enum"`.
    pub has_child_kind: &'static str,
    /// Refined node kind, e.g. `"enum"` / `"interface"`.
    pub kind: &'static str,
    /// Refined signature prefix, e.g. `"enum"` / `"interface"`.
    pub prefix: &'static str,
}

/// Parse `abs_path` using the given grammar and `LanguageConfig`.
///
/// If the caller already has a pre-compiled `Query` (e.g. `GenericTreeSitterPlugin`),
/// pass it via `compiled_query`. Pass `None` to compile fresh from `config.queries`.
/// The grammar is obtained from `config.get_grammar()` when not already known.
///
/// # O(n) where n = number of AST nodes matching the query
pub fn parse_with_config(
    config: &LanguageConfig,
    grammar: &tree_sitter::Language,
    compiled_query: Option<&Query>,
    corpus: &str,
    abs_path: &Path,
    vname_path: &str,
) -> anyhow::Result<ParseOutput> {
    let size = std::fs::metadata(abs_path)
        .with_context(|| format!("stat {}", abs_path.display()))?
        .len();
    anyhow::ensure!(size <= MAX_FILE_BYTES, "file too large: {size} bytes");

    let source =
        std::fs::read(abs_path).with_context(|| format!("reading {}", abs_path.display()))?;

    let mut parser = Parser::new();
    parser.set_language(grammar).context("set language")?;
    let tree = parser.parse(&source, None).context("parse timeout")?;

    // Compile fresh or use the caller-supplied pre-compiled query.
    let owned_query;
    let query: &Query = match compiled_query {
        Some(q) => q,
        None => {
            owned_query = Query::new(grammar, config.queries).context("compile query")?;
            &owned_query
        }
    };

    let lang_str = config.language.as_str();
    let file_vname = VName::new(corpus, "", vname_path, lang_str, "file");
    let file_node = Node::new(file_vname, "file");
    let file_id = file_node.id;
    let mut nodes = vec![file_node];
    let mut edges: Vec<Edge> = Vec::new();

    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut iter = cursor.matches(query, tree.root_node(), source.as_slice());

    // #479: `@test.entry`/`@test.scope` line-span signals collected during the
    // walk, then applied to `nodes` in a single post-pass keyed off each node's
    // own start line.
    let mut test_signals = crate::test_role::TestSignals::default();

    while let Some(m) = iter.next() {
        for cap in m.captures {
            let cap_name = *capture_names.get(cap.index as usize).unwrap_or(&"");

            // #479: route test captures to the signal collector; they are not in
            // `capture_kinds` so they never emit a node.
            if test_signals.route_capture(
                cap_name,
                cap.node.start_position().row,
                cap.node.end_position().row,
            ) {
                continue;
            }

            let Some(&(_, node_kind, sig_prefix)) = config
                .capture_kinds
                .iter()
                .find(|(name, _, _)| *name == cap_name)
            else {
                continue;
            };

            // N4d: if the captured name's declaration is a folded type node
            // (Kotlin `class_declaration` covering class/interface/enum),
            // refine its kind+prefix from the decl's direct children. Only
            // fires when the parent kind matches a declared refinement, so
            // `fn`/`import`/object captures are untouched.
            let (node_kind, sig_prefix) = match cap
                .node
                .parent()
                .and_then(|p| refine_type(p, config.type_refinements))
            {
                Some(r) => (r.kind, r.prefix),
                None => (node_kind, sig_prefix),
            };

            // A language may spell one name across several sibling nodes
            // (Objective-C selectors); the hook rebuilds it from the captured
            // node. `None` leaves the captured node's own text in place.
            let hooked = config
                .name_hook
                .and_then(|h| h(cap.node, sig_prefix, &source));
            let text = match &hooked {
                Some(name) => name.trim(),
                None => cap.node.utf8_text(&source).unwrap_or("").trim(),
            };
            if text.is_empty() {
                continue;
            }

            let line = cap.node.start_position().row as u32 + 1;
            // N2: end the span at the enclosing full-definition node when the
            // name is nested below it (C/C++/Obj-C function bodies); otherwise
            // one hop to `name.parent()` gives the full span (G2).
            let end_line = decl_end_line(cap.node, config.decl_kinds).unwrap_or(line);

            // N1/N3 + #757: a member definition nested inside a type container is
            // qualified by the enclosing type (collision-free per Invariant #1)
            // and its containment edge is parented to the container node, not the
            // file. `member_qual` maps a member's signature prefix to its
            // qualified `(sig_prefix, node_kind)`: a `fn` capture becomes
            // `method:Type.name`, a `field` capture becomes `field:Type.name`.
            // Every other prefix (var/const/type/class/import) stays file-level.
            let member_qual: Option<(&'static str, &'static str)> = match sig_prefix {
                "fn" => Some(("method", "method")),
                "field" => Some(("field", "field")),
                _ => None,
            };
            let enclosing = if member_qual.is_some() {
                enclosing_container(
                    cap.node,
                    config.method_containers,
                    config.type_refinements,
                    &source,
                )
            } else {
                None
            };

            let (sig, node_kind, parent_id) = match (&enclosing, member_qual) {
                (Some((container_prefix, container_name)), Some((qual_prefix, qual_kind))) => {
                    let container_sig = format!("{container_prefix}:{container_name}");
                    let container_id =
                        VName::new(corpus, "", vname_path, lang_str, &container_sig).id();
                    (
                        format!("{qual_prefix}:{container_name}.{text}"),
                        qual_kind,
                        container_id,
                    )
                }
                _ if is_import_prefix(sig_prefix) => {
                    // Use the full node text, strip leading keyword + trailing
                    // semicolons. N5: also strip surrounding quotes so Dart
                    // `import 'dart:core';` yields `import:dart:core` (the
                    // DartResolver tests `starts_with("dart:")`) and C
                    // `#include "foo.h"` yields `import:foo.h`.
                    let cleaned = text
                        .trim_start_matches("import ")
                        .trim_start_matches("use ")
                        .trim_start_matches("require ")
                        .trim_start_matches("using ")
                        .trim_end_matches(';')
                        .trim()
                        .trim_matches(|c| c == '\'' || c == '"')
                        .to_string();
                    (format!("{sig_prefix}:{cleaned}"), node_kind, file_id)
                }
                // A member not enclosed by any container (top-level fn / property)
                // and every non-member prefix: file-level, unqualified.
                _ => (format!("{sig_prefix}:{text}"), node_kind, file_id),
            };

            let vname = VName::new(corpus, "", vname_path, lang_str, &sig);
            let mut node = Node::new(vname, node_kind).with_line(line);
            if !is_import_prefix(sig_prefix) {
                node = node.with_end_line(end_line);
            }
            let edge_kind = if node_kind == "import" {
                EdgeKind::Depends
            } else {
                EdgeKind::DefinesBinding
            };
            edges.push(Edge::new(parent_id, node.id, edge_kind));
            nodes.push(node);
        }
    }

    // #780: the capture loop emits one node per matched capture, so a symbol
    // whose definition site recurs — a Ruby ivar `@x` reassigned in several
    // methods, a constant redeclared — yields several nodes that share one
    // VName/NodeId at different lines. `flush_staging_to_production` documents
    // the invariant "same NodeId => same VName => same parse output, so all are
    // equal" and breaks any tie with `MAX(line)`, which would anchor the node on
    // the last reassignment and defeat the SCIP def's line-proximity match.
    // Collapse duplicates to the first occurrence in document order (query
    // matches arrive in start-position order) so the invariant holds and the
    // anchor line stays the definition, not a later write.
    {
        let mut seen_ids = std::collections::HashSet::new();
        nodes.retain(|n| seen_ids.insert(n.id));
    }

    // #780: language-specific expansion the shared capture pipeline cannot
    // express (Ruby `attr_*` macros → accessor method nodes). No-op for every
    // config that leaves `post_parse` unset.
    if let Some(hook) = config.post_parse {
        let ctx = PostParseCtx {
            root: tree.root_node(),
            source: &source,
            corpus,
            vname_path,
            lang: lang_str,
            file_id,
            config,
        };
        hook(&ctx, &mut nodes, &mut edges);
    }

    // #479: single language-agnostic post-pass sets test_role from the collected
    // signals (no-op when the file has no test captures).
    crate::test_role::apply_test_roles(&test_signals, &mut nodes);

    Ok(ParseOutput {
        nodes,
        edges,
        ffi_markers: vec![],
        workspace_dep_markers: vec![],
    })
}

/// `true` for the bare `"import"` prefix or a scheme-marked variant
/// (`"import:<scheme>"`, e.g. Ruby's `"import:gem"`, #614). Both forms share
/// the same node-text cleanup and both stay `end_line`-free, so every import
/// capture is treated uniformly regardless of which scheme produced it.
fn is_import_prefix(sig_prefix: &str) -> bool {
    sig_prefix == "import" || sig_prefix.starts_with("import:")
}

/// Compute the 1-based end line for a definition's span (N2).
///
/// Walks up (bounded) from the name capture: if an ancestor's kind is in
/// `decl_kinds` (a full definition with body), that node's end row is the span
/// end. Otherwise falls back to the one-hop `name.parent()` end row, which is
/// correct for grammars where the name is a direct child of its declaration.
/// Either way the row is clamped by [`clamp_to_next_sibling`].
fn decl_end_line(node: tree_sitter::Node<'_>, decl_kinds: &[&str]) -> Option<u32> {
    let line = node.start_position().row as u32 + 1;
    if !decl_kinds.is_empty() {
        let mut cur = node.parent();
        for _ in 0..6 {
            let Some(n) = cur else { break };
            if decl_kinds.contains(&n.kind()) {
                return Some(clamp_to_next_sibling(n, line));
            }
            cur = n.parent();
        }
    }
    node.parent().map(|p| clamp_to_next_sibling(p, line))
}

/// End row of `decl` as a 1-based line, clamped so a definition's span cannot
/// run into the sibling that follows it.
///
/// tree-sitter-scala ends an expression-bodied `function_definition` at the
/// start position of the *next* declaration (column 0 of its line, or its
/// indent column), so the raw end row is that sibling's own line. The two
/// spans then overlap on the boundary line and a reference occurrence there
/// sits inside two equally wide spans, leaving caller attribution to break the
/// tie by NodeId. Never clamps below `line`, the definition's start line.
///
/// Only that overlap is clamped. Merely sharing the end *row* with the next
/// sibling is not one: `}  /* trailing */` puts a comment on the closing brace
/// line, and `... } int after(int y) { .. }` puts a whole second definition
/// there, yet in both cases the definition really does end where it ends and
/// shortening it drops its own last line (and, for the crammed case, the call
/// sites on it). So extras are skipped when picking the sibling, and the clamp
/// applies only when the definition's end position has actually run past where
/// that sibling starts.
fn clamp_to_next_sibling(decl: tree_sitter::Node<'_>, line: u32) -> u32 {
    let end = decl.end_position().row as u32 + 1;
    let mut next = decl.next_sibling();
    while let Some(n) = next {
        if !n.is_extra() {
            break;
        }
        next = n.next_sibling();
    }
    let Some(next) = next else { return end };
    if decl.end_position() < next.start_position() {
        return end;
    }
    // `next.start_position().row` is 0-based, so it is exactly the 1-based
    // line *before* the sibling begins.
    end.min(next.start_position().row as u32).max(line)
}

/// Walk up from a definition capture to the nearest enclosing type container.
///
/// Returns `(container_prefix, container_name)` for the first ancestor whose
/// kind appears in `method_containers`, or `None` if the definition is not
/// nested in any container (i.e. a free function). `container_prefix` is the
/// signature prefix that container is captured under, so the caller can
/// reconstruct the container's own VName (`{prefix}:{name}`) for the
/// containment edge.
pub(crate) fn enclosing_container<'a>(
    node: tree_sitter::Node<'_>,
    method_containers: &[(&'static str, &'a str)],
    type_refinements: &[TypeRefinement],
    source: &[u8],
) -> Option<(&'a str, String)> {
    let mut cur = node.parent();
    while let Some(n) = cur {
        if let Some((_, prefix)) = method_containers.iter().find(|(kind, _)| *kind == n.kind()) {
            if let Some(name) = container_name(n, source) {
                // N4d: grammars that fold several type kinds into one AST node
                // disambiguate either via a `declaration_kind` keyword field
                // (tree-sitter-swift: class/struct/enum/actor/extension) or via
                // a distinguishing direct child (tree-sitter-kotlin-ng:
                // interface/enum via `type_refinements`). When either resolves,
                // the container is emitted under that refined prefix, so the
                // method's containment edge must target `{prefix}:{name}` to
                // match. Absent both, the static `method_containers` prefix is
                // used (every non-Swift/Kotlin grammar).
                let prefix = container_kind_prefix(n, type_refinements, source).unwrap_or(prefix);
                return Some((prefix, name));
            }
            // #757: a C/C++ anonymous aggregate named only via
            // `typedef struct { .. } Name;` has no `name:` on the
            // struct_specifier, so `container_name` is None and its fields would
            // orphan as unqualified `field:name` (colliding across every such
            // struct). The graph node that actually exists for it is the typedef
            // (`type:Name`, from the `type_definition` capture), so borrow that
            // name and prefix. Inert for grammars without `type_definition`.
            if let Some(name) = n
                .parent()
                .filter(|p| p.kind() == "type_definition")
                .and_then(|td| td.child_by_field_name("declarator"))
                .filter(|d| d.kind() == "type_identifier")
                .and_then(|d| d.utf8_text(source).ok())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                return Some(("type", name.to_string()));
            }
        }
        cur = n.parent();
    }
    None
}

/// N4d: resolve the refined signature prefix of a folded type container so the
/// node and its members' containment edges agree on its VName. Checks the
/// child-based `type_refinements` first (Kotlin interface/enum), then the
/// Swift-style `declaration_kind` keyword field. Returns `None` when neither
/// applies, leaving the caller's static prefix in force.
fn container_kind_prefix(
    node: tree_sitter::Node<'_>,
    type_refinements: &[TypeRefinement],
    source: &[u8],
) -> Option<&'static str> {
    if let Some(r) = refine_type(node, type_refinements) {
        return Some(r.prefix);
    }
    let kw = node
        .child_by_field_name("declaration_kind")?
        .utf8_text(source)
        .ok()?
        .trim();
    match kw {
        "struct" => Some("struct"),
        "enum" => Some("enum"),
        "actor" => Some("actor"),
        "extension" => Some("extension"),
        "class" => Some("class"),
        _ => None,
    }
}

/// N4d: refine a folded type declaration by inspecting its direct children and
/// its `modifiers` subtree. Returns the first refinement (list order = the
/// tie-break) whose `decl_kind` matches `decl.kind()` and whose `has_child_kind`
/// appears as a direct child or a `modifiers` descendant. `None` when `decl` is
/// not a folded type node or no refinement matches (the common case:
/// `type_refinements` is empty for every non-Kotlin config).
fn refine_type<'r>(
    decl: tree_sitter::Node<'_>,
    type_refinements: &'r [TypeRefinement],
) -> Option<&'r TypeRefinement> {
    if type_refinements.is_empty() {
        return None;
    }
    let kind = decl.kind();
    // Signal kinds: the decl's direct children, plus every node kind inside a
    // `modifiers` child (where class-level modifiers like `enum`/`sealed` live).
    // The body is not scanned so a nested type cannot leak its keyword upward.
    let mut signals: Vec<&str> = Vec::new();
    let mut walk = decl.walk();
    for child in decl.children(&mut walk) {
        signals.push(child.kind());
        if child.kind() == "modifiers" {
            collect_descendant_kinds(child, &mut signals);
        }
    }
    type_refinements
        .iter()
        .find(|r| r.decl_kind == kind && signals.contains(&r.has_child_kind))
}

/// Collect the kinds of every descendant of `node` (used to flatten a small
/// `modifiers` subtree so nested modifier tokens like `enum` are visible).
fn collect_descendant_kinds<'a>(node: tree_sitter::Node<'a>, out: &mut Vec<&'a str>) {
    let mut walk = node.walk();
    for child in node.children(&mut walk) {
        out.push(child.kind());
        collect_descendant_kinds(child, out);
    }
}

/// Extract a type container's declared name. Prefers the `name` field; falls
/// back to the first identifier-like child (grammars such as tree-sitter-objc
/// anchor the class name positionally with no `name` field).
fn container_name(node: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(t) = name_node.utf8_text(source) {
            let t = t.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    let mut walk = node.walk();
    for child in node.children(&mut walk) {
        if matches!(
            child.kind(),
            "identifier"
                | "type_identifier"
                | "constant"
                | "simple_identifier"
                | "name"
                | "namespace_identifier"
        ) {
            if let Ok(t) = child.utf8_text(source) {
                let t = t.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use travsr_core::EdgeKind;

    fn parse_str(config: &LanguageConfig, name: &str, src: &str) -> ParseOutput {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, src).unwrap();
        let grammar = (config.get_grammar)();
        parse_with_config(config, &grammar, None, "corp", &path, name).unwrap()
    }

    #[test]
    fn keystone_two_same_named_methods_are_distinct_nodes() {
        // N1 / Invariant #1: two classes each defining `run()` must NOT collapse
        // to one VName. Before N1 both were `fn:run` → one NodeId clobbered the
        // other. After N1 they are `method:A.run` and `method:B.run`.
        let out = parse_str(
            &crate::php::CONFIG,
            "collide.php",
            "<?php\nclass A { function run() {} }\nclass B { function run() {} }\n",
        );
        let method_sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "method")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert!(method_sigs.contains(&"method:A.run"), "got {method_sigs:?}");
        assert!(method_sigs.contains(&"method:B.run"), "got {method_sigs:?}");

        let a = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:A.run")
            .unwrap();
        let b = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:B.run")
            .unwrap();
        assert_ne!(
            a.id, b.id,
            "two distinct methods must have distinct NodeIds"
        );

        // No bare `fn:run` collision node survives.
        assert!(
            !out.nodes.iter().any(|n| n.vname.signature == "fn:run"),
            "bare fn:run must not exist after N1"
        );
    }

    #[test]
    fn keystone_containment_edge_parents_method_to_type() {
        // N3: the DefinesBinding edge for a method is parented to its enclosing
        // type node, not the file.
        let out = parse_str(
            &crate::php::CONFIG,
            "contain.php",
            "<?php\nclass A { function run() {} }\n",
        );
        let file_id = out.nodes.iter().find(|n| n.kind == "file").unwrap().id;
        let class_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "class:A")
            .unwrap()
            .id;
        let method_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:A.run")
            .unwrap()
            .id;
        assert!(
            out.edges.iter().any(|e| e.src == class_id
                && e.dst == method_id
                && e.kind == EdgeKind::DefinesBinding),
            "expected class:A → method:A.run containment edge"
        );
        assert!(
            !out.edges
                .iter()
                .any(|e| e.src == file_id && e.dst == method_id),
            "flat file → method edge must be gone (N3)"
        );
    }

    #[test]
    fn n2_c_function_span_includes_body() {
        // N2: a C function's end_line must reach the closing brace so call
        // sites inside its body fall within [line, end_line] and attribute to
        // the function (not the file). Before N2 the 1-hop span ended at the
        // signature line (function_declarator), excluding the body.
        let src = "int compute(int x) {\n    int y = x + 1;\n    return y;\n}\n";
        let out = parse_str(&crate::c::CONFIG, "compute.c", src);
        let f = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "fn:compute")
            .expect("fn:compute node");
        assert_eq!(f.line, Some(1), "starts on the signature line");
        assert_eq!(
            f.end_line,
            Some(4),
            "span must reach the closing brace (line 4), not end at the signature line"
        );
    }

    #[test]
    fn a_comment_after_the_closing_brace_does_not_shorten_the_span() {
        // A `comment` is a tree-sitter extra and still shows up as the next
        // sibling, on the closing brace's own line. Clamping to it dropped the
        // brace line off every definition written this way, classes and structs
        // included.
        let src = "int helper(int x) { return x; }\n\n\
             int trailing(int x) {\n    return helper(x);\n}  /* trailing comment */\n";
        let out = parse_str(&crate::c::CONFIG, "trailing.c", src);
        let f = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "fn:trailing")
            .expect("fn:trailing node");
        assert_eq!(f.line, Some(3));
        assert_eq!(
            f.end_line,
            Some(5),
            "the span must still reach the closing brace"
        );
    }

    #[test]
    fn a_definition_crammed_onto_the_closing_line_does_not_shorten_the_span() {
        // `after` starts on `crammed`'s last line but two columns past where
        // `crammed` ends, so there is no overlap to clamp. Clamping anyway cut
        // `crammed` to lines 2..2 and pushed the `helper(x)` call on line 3 outside
        // it, so `find_narrowest_enclosing` attributed the call to the file.
        let src = "int helper(int x) { return x; }\n\
             int crammed(int x) {\n    return helper(x); }  int after(int y) { return y; }\n";
        let out = parse_str(&crate::c::CONFIG, "crammed.c", src);
        let f = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "fn:crammed")
            .expect("fn:crammed node");
        assert_eq!(f.line, Some(2));
        assert_eq!(
            f.end_line,
            Some(3),
            "the span must cover the line the body actually ends on"
        );
    }

    #[test]
    fn n5_dart_import_strips_quotes() {
        // N5: `import 'dart:core';` must yield `import:dart:core` (no quotes) so
        // the DartResolver's `starts_with("dart:")` test resolves stdlib imports.
        let out = parse_str(
            &crate::dart::CONFIG,
            "q.dart",
            "import 'dart:core';\nimport 'package:foo/bar.dart';\n",
        );
        let imports: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "import")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert!(imports.contains(&"import:dart:core"), "got {imports:?}");
        assert!(
            imports.contains(&"import:package:foo/bar.dart"),
            "got {imports:?}"
        );
    }

    // #757: every declaration-based config language emits owner-qualified
    // `field:Owner.name` nodes contained by their type. One case per language,
    // asserting the field node exists, is kind `field`, and has a containment
    // edge from the owning type node (not the file).
    #[test]
    fn field_nodes_emitted_and_contained_per_language() {
        struct Case {
            cfg: &'static LanguageConfig,
            file: &'static str,
            src: &'static str,
            field_sig: &'static str,
            owner_sig: &'static str,
        }
        let cases = [
            Case {
                cfg: &crate::c::CONFIG,
                file: "p.c",
                src: "struct Point { int x; int y; };\n",
                field_sig: "field:Point.x",
                owner_sig: "struct:Point",
            },
            Case {
                cfg: &crate::cpp::CONFIG,
                file: "p.cpp",
                src: "class C { int count; void m(); };\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::csharp::CONFIG,
                file: "p.cs",
                src: "class C { int count; public int P { get; set; } }\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::csharp::CONFIG,
                file: "prop.cs",
                src: "class C { public int Total { get; set; } }\n",
                field_sig: "field:C.Total",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::kotlin::CONFIG,
                file: "p.kt",
                src: "class C {\n    val count: Int = 0\n    fun m() { val local = 1 }\n}\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::php::CONFIG,
                file: "p.php",
                src: "<?php\nclass C {\n    public $count;\n    private int $total;\n}\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::scala::CONFIG,
                file: "p.scala",
                src: "class C {\n  val count: Int = 0\n  def m(): Int = { val local = 1\n    local }\n}\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::dart::CONFIG,
                file: "p.dart",
                src: "class C {\n  int count = 0;\n  final String name = \"x\";\n}\n",
                field_sig: "field:C.count",
                owner_sig: "class:C",
            },
            Case {
                cfg: &crate::objc::CONFIG,
                file: "p.m",
                src: "@interface C {\n  int _count;\n}\n@property (nonatomic) int total;\n@end\n",
                field_sig: "field:C._count",
                owner_sig: "class:C",
            },
        ];

        for c in &cases {
            let out = parse_str(c.cfg, c.file, c.src);
            let field = out
                .nodes
                .iter()
                .find(|n| n.vname.signature == c.field_sig)
                .unwrap_or_else(|| {
                    panic!(
                        "{:?}: missing {}, have fields: {:?}",
                        c.cfg.language,
                        c.field_sig,
                        out.nodes
                            .iter()
                            .filter(|n| n.kind == "field")
                            .map(|n| &n.vname.signature)
                            .collect::<Vec<_>>()
                    )
                });
            assert_eq!(field.kind, "field", "{:?}: wrong kind", c.cfg.language);
            let owner = out
                .nodes
                .iter()
                .find(|n| n.vname.signature == c.owner_sig)
                .unwrap_or_else(|| panic!("{:?}: missing owner {}", c.cfg.language, c.owner_sig));
            assert!(
                out.edges.iter().any(|e| e.src == owner.id
                    && e.dst == field.id
                    && e.kind == EdgeKind::DefinesBinding),
                "{:?}: expected {} → {} containment edge",
                c.cfg.language,
                c.owner_sig,
                c.field_sig
            );
        }
    }

    // #757: a member `val`/`var`/local declared inside a method body must NOT be
    // captured as a field (would pollute the type and risk collisions).
    #[test]
    fn local_variables_are_not_fields() {
        let out = parse_str(
            &crate::scala::CONFIG,
            "loc.scala",
            "class C {\n  def m(): Int = {\n    val temp = 5\n    temp\n  }\n}\n",
        );
        assert!(
            !out.nodes
                .iter()
                .any(|n| n.vname.signature == "field:C.temp"),
            "local val must not become a field"
        );
    }

    #[test]
    fn free_function_stays_unqualified() {
        // A `fn` capture NOT nested in a container is still a free function.
        let out = parse_str(
            &crate::php::CONFIG,
            "free.php",
            "<?php\nfunction top() {}\n",
        );
        assert!(
            out.nodes
                .iter()
                .any(|n| n.kind == "function" && n.vname.signature == "fn:top"),
            "top-level function must stay fn:top"
        );
    }
}
