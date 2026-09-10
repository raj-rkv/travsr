//! Phase A parser for PHP source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Php,
    extensions: &["php", "phtml", "php8"],
    queries: r#"
(class_declaration name: (name) @class.name)
(interface_declaration name: (name) @interface.name)
(trait_declaration name: (name) @trait.name)
(enum_declaration name: (name) @enum.name)
(function_definition name: (name) @fn.name)
(method_declaration name: (name) @fn.name)
(property_declaration (property_element name: (variable_name (name) @field.name)))
(namespace_use_clause) @import
(class_declaration
  (base_clause (name) @_pb)
  (#eq? @_pb "TestCase")) @test.scope
(method_declaration
  attributes: (attribute_list (attribute_group (attribute (name) @_pa)))
  name: (name) @test.entry
  (#eq? @_pa "Test"))
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("interface.name", "interface", "interface"),
        // N4d: PHP traits are a distinct kind, not folded into `class`. The
        // `trait:` signature unifies onto the SCIP trait def (scip-php emits a
        // `#` type descriptor → `candidate_signatures` class-group, which
        // already contains `trait:`).
        ("trait.name", "trait", "trait"),
        ("enum.name", "enum", "enum"),
        ("fn.name", "function", "fn"),
        // #757: class/interface/trait/enum properties → `field:Owner.name`.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        // N4d: trait methods parent to the `trait:` node (was `class`, which
        // would dangle now that the node is `trait:`).
        ("trait_declaration", "trait"),
        ("enum_declaration", "enum"),
    ],
    decl_kinds: &[],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_php::LANGUAGE_PHP),
};

/// Parse a PHP source file into graph nodes and edges.
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
        let path = dir.path().join("empty.php");
        std::fs::write(&path, "<?php\n").unwrap();
        let out = parse("corp", &path, "empty.php").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_class_and_function() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.php");
        std::fs::write(
            &path,
            "<?php\nuse App\\Models\\User;\nclass Foo {}\ninterface Bar {}\nfunction baz() {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.php").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"interface"));
        assert!(kinds.contains(&"function"));
        assert!(kinds.contains(&"import"));
    }

    #[test]
    fn n4d_trait_distinct_kind_and_method_containment() {
        // N4d: a PHP trait is kind `trait` with sig `trait:Logs`, and its method
        // parents to the trait node (not a dangling `class:Logs`).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.php");
        std::fs::write(
            &path,
            "<?php\ntrait Logs {\n  public function log($m) {}\n}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "t.php").unwrap();

        let trait_node = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "trait:Logs")
            .expect("trait:Logs node");
        assert_eq!(trait_node.kind, "trait");
        let method = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:Logs.log")
            .expect("method:Logs.log node");
        let contained = out.edges.iter().any(|e| {
            e.kind == travsr_core::EdgeKind::DefinesBinding
                && e.src == trait_node.id
                && e.dst == method.id
        });
        assert!(contained, "trait:Logs must contain method:Logs.log");
    }
}
