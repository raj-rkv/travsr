//! Phase A parser for C# source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::CSharp,
    extensions: &["cs"],
    queries: r#"
(class_declaration name: (identifier) @class.name)
(interface_declaration name: (identifier) @interface.name)
(struct_declaration name: (identifier) @struct.name)
(enum_declaration name: (identifier) @enum.name)
(record_declaration name: (identifier) @class.name)
(delegate_declaration name: (identifier) @delegate.name)
(method_declaration name: (identifier) @fn.name)
(constructor_declaration name: (identifier) @fn.name)
(field_declaration (variable_declaration (variable_declarator name: (identifier) @field.name)))
(property_declaration name: (identifier) @field.name)
(using_directive) @import
(class_declaration
  (attribute_list (attribute name: (identifier) @_ca))
  (#any-of? @_ca "TestFixture" "TestClass")) @test.scope
(method_declaration
  (attribute_list (attribute name: (identifier) @_ma))
  name: (identifier) @test.entry
  (#any-of? @_ma "Test" "Fact" "Theory" "TestMethod"))
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("interface.name", "interface", "interface"),
        // N4d: C# structs are a distinct kind, not folded into `class`. The
        // `struct:` signature unifies onto the SCIP struct def (scip-dotnet
        // emits a `#` type descriptor → `candidate_signatures` class-group,
        // which already contains `struct:`).
        ("struct.name", "struct", "struct"),
        ("enum.name", "enum", "enum"),
        ("delegate.name", "delegate", "type"),
        ("fn.name", "function", "fn"),
        // #757: fields and auto-properties → `field:Owner.name`, contained by
        // their type. `field_declaration` and `property_declaration` only occur
        // in a type body, never a method body (locals are
        // `local_declaration_statement`), so no anchoring is needed.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_declaration", "class"),
        // N4d: struct methods parent to the `struct:` node, matching the split
        // above (was `class`, which would dangle now that the node is `struct:`).
        ("struct_declaration", "struct"),
        ("record_declaration", "class"),
        ("interface_declaration", "interface"),
        ("enum_declaration", "enum"),
    ],
    decl_kinds: &[],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_c_sharp::LANGUAGE),
};

/// Parse a C# source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.cs");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.cs").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_class_and_interface() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.cs");
        std::fs::write(
            &path,
            "using System;\nclass Foo : IFoo { void Bar() {} }\ninterface IFoo {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.cs").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"interface"));
        assert!(kinds.contains(&"import"));
        // N1: `void Bar()` inside `class Foo` is a method, qualified by its type.
        assert!(
            out.nodes
                .iter()
                .any(|n| n.kind == "method" && n.vname.signature == "method:Foo.Bar"),
            "expected method:Foo.Bar; got {:?}",
            out.nodes
                .iter()
                .map(|n| &n.vname.signature)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn n4d_struct_distinct_kind_and_method_containment() {
        // N4d: a C# struct is kind `struct` with sig `struct:Point`, and its
        // method parents to the struct node (not a dangling `class:Point`).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.cs");
        std::fs::write(
            &path,
            "struct Point { public int X; public int Norm() { return 0; } }\n",
        )
        .unwrap();
        let out = parse("corp", &path, "p.cs").unwrap();

        let struct_node = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "struct:Point")
            .expect("struct:Point node");
        assert_eq!(struct_node.kind, "struct");
        let method = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:Point.Norm")
            .expect("method:Point.Norm node");
        let contained = out.edges.iter().any(|e| {
            e.kind == travsr_core::EdgeKind::DefinesBinding
                && e.src == struct_node.id
                && e.dst == method.id
        });
        assert!(contained, "struct:Point must contain method:Point.Norm");
    }
}
