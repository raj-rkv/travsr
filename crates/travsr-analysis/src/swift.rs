//! Phase A parser for Swift source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Swift,
    extensions: &["swift"],
    queries: r#"
(class_declaration declaration_kind: "class"     name: (type_identifier) @class.name)
(class_declaration declaration_kind: "struct"    name: (type_identifier) @struct.name)
(class_declaration declaration_kind: "enum"      name: (type_identifier) @enum.name)
(class_declaration declaration_kind: "actor"     name: (type_identifier) @actor.name)
(class_declaration declaration_kind: "extension" name: (user_type (type_identifier) @extension.name))
(protocol_declaration name: (type_identifier) @protocol.name)
(typealias_declaration name: (type_identifier) @typealias.name)
(function_declaration name: (simple_identifier) @fn.name)
(init_declaration "init" @init.name)
(import_declaration)  @import
(class_body (property_declaration name: (pattern bound_identifier: (simple_identifier) @var.name)))
(enum_class_body (property_declaration name: (pattern bound_identifier: (simple_identifier) @var.name)))
(function_declaration
  (modifiers (attribute (user_type (type_identifier) @_swa)))
  name: (simple_identifier) @test.entry
  (#eq? @_swa "Test"))
(class_declaration
  (inheritance_specifier inherits_from: (user_type (type_identifier) @_swc))
  (#eq? @_swc "XCTestCase")) @test.scope
"#,
    capture_kinds: &[
        // N4d: tree-sitter-swift folds all five type declarations into
        // `class_declaration`, distinguished by the `declaration_kind` keyword.
        // Emit distinct kinds AND distinct signature prefixes; the class-group
        // signatures (struct/enum) already unify via `candidate_signatures`,
        // `actor:` was added there in lockstep, and `extension:` stops an
        // extension from colliding with the class it extends (both were
        // `class:Foo` before). Members' containment edges follow the same
        // prefix via `container_kind_prefix`.
        ("class.name", "class", "class"),
        ("struct.name", "struct", "struct"),
        ("enum.name", "enum", "enum"),
        ("actor.name", "actor", "actor"),
        ("extension.name", "extension", "extension"),
        ("protocol.name", "protocol", "class"),
        ("typealias.name", "type", "type"),
        ("fn.name", "function", "fn"),
        ("init.name", "init", "fn"),
        ("import", "import", "import"),
        // #449/#757: properties (`static let shared`, `var count`) become
        // owner-qualified `field:Type.name` nodes (was unqualified `var:name`,
        // which collided when two types shared a property name). The generic
        // hook parents them to their enclosing type; the SCIP/index unifier
        // now maps `Type.name` field references onto `field:Type.name`
        // (candidate_signatures, #757).
        ("var.name", "field", "field"),
    ],
    method_containers: &[
        ("class_declaration", "class"),
        ("protocol_declaration", "class"),
    ],
    decl_kinds: &[],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_swift::LANGUAGE),
};

/// Parse a Swift source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n4d_distinct_type_kinds() {
        // N4d: struct/enum/actor/extension no longer collapse to kind `class`.
        // Each gets a distinct kind and a distinct signature prefix.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("types.swift");
        std::fs::write(
            &path,
            "class C {}\nstruct S {}\nenum E { case a }\nactor A {}\nextension X {}\nprotocol P {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "types.swift").unwrap();
        let by_sig: std::collections::HashMap<&str, &str> = out
            .nodes
            .iter()
            .map(|n| (n.vname.signature.as_str(), n.kind.as_str()))
            .collect();
        assert_eq!(by_sig.get("class:C"), Some(&"class"));
        assert_eq!(by_sig.get("struct:S"), Some(&"struct"));
        assert_eq!(by_sig.get("enum:E"), Some(&"enum"));
        assert_eq!(by_sig.get("actor:A"), Some(&"actor"));
        assert_eq!(by_sig.get("extension:X"), Some(&"extension"));
        assert_eq!(by_sig.get("class:P"), Some(&"protocol"));
    }

    #[test]
    fn n4d_struct_method_containment_matches_struct_node() {
        // N4d: a method inside a `struct` must have its containment edge parented
        // to the `struct:S` node (via container_kind_prefix), not a nonexistent
        // `class:S` — otherwise the edge dangles.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.swift");
        std::fs::write(&path, "struct S {\n  func run() {}\n}\n").unwrap();
        let out = parse("corp", &path, "m.swift").unwrap();

        let struct_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "struct:S")
            .map(|n| n.id)
            .expect("struct:S node");
        let method = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:S.run")
            .expect("method:S.run node");
        assert_eq!(method.kind, "method");
        // The containment edge from struct:S -> method:S.run must exist.
        let contained = out.edges.iter().any(|e| {
            e.kind == travsr_core::EdgeKind::DefinesBinding
                && e.src == struct_id
                && e.dst == method.id
        });
        assert!(
            contained,
            "struct:S must contain method:S.run (no dangling)"
        );
    }

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.swift");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.swift").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_class_and_protocol() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.swift");
        std::fs::write(
            &path,
            "import Foundation\nclass Foo {}\nprotocol Bar {}\nfunc baz() {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.swift").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"protocol"));
        assert!(kinds.contains(&"function"));
        assert!(kinds.contains(&"import"));
    }

    #[test]
    fn parse_property_declaration() {
        // #449/#757: `static let shared` / `var count` produce owner-qualified
        // field nodes (`field:ClassC.shared`) contained by their type, so Phase B
        // `swift::ClassC.shared` unifies (candidate_signatures now yields
        // `field:ClassC.shared`) and dotted queries resolve.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("singleton.swift");
        std::fs::write(
            &path,
            "class ClassC {\n    static let shared = ClassC()\n    var count: Int = 0\n}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "singleton.swift").unwrap();
        let fields: Vec<&travsr_core::Node> =
            out.nodes.iter().filter(|n| n.kind == "field").collect();
        let sigs: Vec<&str> = fields.iter().map(|n| n.vname.signature.as_str()).collect();
        assert!(
            sigs.contains(&"field:ClassC.shared"),
            "got field sigs: {sigs:?}"
        );
        assert!(
            sigs.contains(&"field:ClassC.count"),
            "got field sigs: {sigs:?}"
        );

        // Containment: field edges parent to the type node, not the file.
        let class_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "class:ClassC")
            .expect("class:ClassC node")
            .id;
        let shared_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "field:ClassC.shared")
            .unwrap()
            .id;
        assert!(
            out.edges.iter().any(|e| e.src == class_id
                && e.dst == shared_id
                && e.kind == travsr_core::EdgeKind::DefinesBinding),
            "field must be contained by its type"
        );
    }

    #[test]
    fn two_swift_types_same_property_no_collision() {
        // #757 Tier-B fix: two types each with a `count` property must NOT
        // collapse to one `var:count` VName. Owner-qualification keeps them
        // distinct (`field:A.count` / `field:B.count`).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("collide.swift");
        std::fs::write(
            &path,
            "class A {\n    var count: Int = 0\n}\nstruct B {\n    var count: Int = 0\n}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "collide.swift").unwrap();
        let a = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "field:A.count")
            .expect("field:A.count");
        let b = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "field:B.count")
            .expect("field:B.count");
        assert_ne!(a.id, b.id, "owner-qualified fields must be distinct nodes");
    }

    #[test]
    fn local_and_toplevel_bindings_are_not_fields() {
        // #757 audit: Swift uses `property_declaration` for top-level and
        // function-local `let`/`var` too, so an unanchored capture emitted
        // spurious unqualified `field:dog` nodes. Anchoring to `class_body`
        // keeps only stored properties.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("main.swift");
        std::fs::write(
            &path,
            "let dog = Dog()\nfunc run() {\n  let local = 1\n}\nclass Cat {\n  let name: String = \"\"\n}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "main.swift").unwrap();
        let field_sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "field")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert_eq!(
            field_sigs,
            vec!["field:Cat.name"],
            "only the stored property is a field; got {field_sigs:?}"
        );
    }

    #[test]
    fn enum_property_is_a_qualified_field() {
        // #757 re-review: Swift enum bodies parse as `enum_class_body`, not
        // `class_body`, so anchoring the property capture to `class_body` alone
        // dropped enum properties master had emitted (as unqualified `var:c`).
        // The `enum_class_body` anchor recovers them, owner-qualified from the
        // enclosing `class_declaration` (`field:E.c`).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.swift");
        std::fs::write(&path, "enum E {\n  var c: Int { 0 }\n}\n").unwrap();
        let out = parse("corp", &path, "e.swift").unwrap();
        let field_sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "field")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert_eq!(
            field_sigs,
            vec!["field:E.c"],
            "enum property must be an owner-qualified field; got {field_sigs:?}"
        );
    }

    // L2: an `init` declaration must emit a bare `method:Cat.init` node, not the
    // full declaration text (`method:Cat.init(name: String) { ... }`) — the bare
    // form is what the travsr-lang Swift emitter and the SCIP unifier expect a
    // `.init` reference to resolve onto.
    #[test]
    fn init_declaration_emits_bare_method_init() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cat.swift");
        std::fs::write(&path, "class Cat {\n  init(name: String) {}\n}\n").unwrap();
        let out = parse("corp", &path, "cat.swift").unwrap();
        let sigs: Vec<&str> = out
            .nodes
            .iter()
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert!(
            sigs.contains(&"method:Cat.init"),
            "expected bare method:Cat.init, got: {sigs:?}"
        );
        assert!(
            !sigs.iter().any(|s| s.starts_with("method:Cat.init(")),
            "must not emit the full declaration text as the signature, got: {sigs:?}"
        );
    }
}
