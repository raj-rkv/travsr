//! Phase A parser for Kotlin source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig, TypeRefinement};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Kotlin,
    extensions: &["kt", "kts"],
    queries: r#"
(class_declaration name: (identifier) @class.name)
(object_declaration name: (identifier) @object.name)
(function_declaration name: (identifier) @fn.name)
(type_alias (identifier) @typealias.name)
(class_body (property_declaration (variable_declaration (identifier) @field.name)))
(enum_class_body (property_declaration (variable_declaration (identifier) @field.name)))
(import) @import
(function_declaration
  (modifiers (annotation (user_type (identifier) @_ka)))
  name: (identifier) @test.entry
  (#eq? @_ka "Test"))
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("object.name", "object", "class"),
        ("fn.name", "function", "fn"),
        ("typealias.name", "type", "type"),
        // #757: class/object properties → `field:Owner.name`. Anchored to
        // `class_body` so local `val`/`var` in function bodies are not captured.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_declaration", "class"),
        ("object_declaration", "class"),
    ],
    decl_kinds: &[],
    // N4d: tree-sitter-kotlin-ng folds interface/enum into `class_declaration`.
    // Signals (verified via parse dumps, all children()-based; note `enum` and
    // `sealed`/`data`/`abstract` live inside a `modifiers > class_modifier`
    // subtree, which `refine_type` scans in addition to direct children):
    //   * interface: the `interface` keyword is a direct child (present even
    //     with a body).
    //   * enum: an `enum` class-modifier token. NOT `enum_class_body` — that
    //     node also appears via error recovery on plain constructor-classes
    //     (`class C(x) { ... }`) and interfaces, so it is not a sound signal.
    // Interface is checked first (order in this list is the tie-break). Bare /
    // constructor / data / annotation / sealed classes carry neither signal and
    // stay `class`.
    type_refinements: &[
        TypeRefinement {
            decl_kind: "class_declaration",
            has_child_kind: "interface",
            kind: "interface",
            prefix: "interface",
        },
        TypeRefinement {
            decl_kind: "class_declaration",
            has_child_kind: "enum",
            kind: "enum",
            prefix: "enum",
        },
    ],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_kotlin_ng::LANGUAGE),
};

/// Parse a Kotlin source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use travsr_core::EdgeKind;

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.kt");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.kt").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_class_and_function() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.kt");
        std::fs::write(
            &path,
            "import kotlin.text.*\nclass Foo\nobject Bar\nfun baz() {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.kt").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"object"));
        assert!(kinds.contains(&"function"));
        assert!(kinds.contains(&"import"));
    }

    fn parse_src(name: &str, src: &str) -> ParseOutput {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, src).unwrap();
        parse("corp", &path, name).unwrap()
    }

    fn sigs(out: &ParseOutput) -> Vec<&str> {
        out.nodes
            .iter()
            .map(|n| n.vname.signature.as_str())
            .collect()
    }

    #[test]
    fn n4d_refines_interface_and_enum_without_regressing_class() {
        // R5 (N4d): the folded `class_declaration` is refined to distinct kinds
        // by a signal node. Bare/constructor/data classes carry neither the
        // `interface` keyword nor an `enum` modifier, so they stay `class`.
        let out = parse_src(
            "kinds.kt",
            "class C\ninterface I\nenum class E { A }\nclass Bare\nclass P(val x: Int)\ndata class D(val y: Int)\n",
        );
        let kind_of = |sig: &str| -> &str {
            out.nodes
                .iter()
                .find(|n| n.vname.signature == sig)
                .unwrap_or_else(|| panic!("no node {sig}; got {:?}", sigs(&out)))
                .kind
                .as_str()
        };
        assert_eq!(kind_of("class:C"), "class");
        assert_eq!(kind_of("class:Bare"), "class");
        assert_eq!(kind_of("class:P"), "class");
        assert_eq!(kind_of("class:D"), "class");
        assert_eq!(kind_of("interface:I"), "interface");
        assert_eq!(kind_of("enum:E"), "enum");

        // No stale `class:I` / `class:E` twin left behind, and nothing emitted
        // twice.
        assert!(
            !out.nodes.iter().any(|n| n.vname.signature == "class:I"),
            "interface must not also emit class:I"
        );
        assert!(
            !out.nodes.iter().any(|n| n.vname.signature == "class:E"),
            "enum must not also emit class:E"
        );
        for sig in ["class:C", "interface:I", "enum:E"] {
            assert_eq!(
                out.nodes
                    .iter()
                    .filter(|n| n.vname.signature == sig)
                    .count(),
                1,
                "{sig} emitted more than once"
            );
        }
    }

    #[test]
    fn enum_class_property_is_a_qualified_field() {
        // #757 re-review: Kotlin's `enum class` body is `enum_class_body`, not
        // `class_body`, so anchoring the property capture to `class_body` alone
        // dropped enum-class properties. The `enum_class_body` anchor recovers
        // them, owner-qualified from the enclosing `class_declaration`
        // (`field:E.d`). Enum entries (`X`) are `enum_entry`, not
        // `property_declaration`, so they are not captured.
        let out = parse_src("e.kt", "enum class E {\n    X;\n    val d = 4\n}\n");
        let field_sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "field")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert_eq!(
            field_sigs,
            vec!["field:E.d"],
            "enum-class property must be an owner-qualified field; got {field_sigs:?}"
        );
    }

    #[test]
    fn n4d_methods_parent_to_refined_container_no_dangling() {
        // R5: a method inside `enum class E` / `interface I` must have its
        // containment edge parented to `enum:E` / `interface:I` (the refined
        // container VName), not a nonexistent `class:E` / `class:I` — else the
        // edge dangles.
        let out = parse_src(
            "members.kt",
            "interface I {\n    fun ping() {}\n}\nenum class E {\n    A;\n    fun tick() {}\n}\n",
        );
        let node_id = |sig: &str| {
            out.nodes
                .iter()
                .find(|n| n.vname.signature == sig)
                .unwrap_or_else(|| panic!("no node {sig}; got {:?}", sigs(&out)))
                .id
        };
        let iface = node_id("interface:I");
        let enum_e = node_id("enum:E");
        let ping = node_id("method:I.ping");
        let tick = node_id("method:E.tick");
        assert!(
            out.edges
                .iter()
                .any(|e| e.src == iface && e.dst == ping && e.kind == EdgeKind::DefinesBinding),
            "interface:I → method:I.ping containment edge missing"
        );
        assert!(
            out.edges
                .iter()
                .any(|e| e.src == enum_e && e.dst == tick && e.kind == EdgeKind::DefinesBinding),
            "enum:E → method:E.tick containment edge missing"
        );

        // No containment edge whose src is not an emitted node (0 dangling).
        let ids: std::collections::HashSet<_> = out.nodes.iter().map(|n| n.id).collect();
        for e in &out.edges {
            if e.kind == EdgeKind::DefinesBinding {
                assert!(ids.contains(&e.src), "dangling containment edge src");
                assert!(ids.contains(&e.dst), "dangling containment edge dst");
            }
        }
    }
}
