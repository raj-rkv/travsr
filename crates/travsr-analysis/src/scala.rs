//! Phase A parser for Scala source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Scala,
    extensions: &["scala", "sc"],
    queries: r#"
(class_definition name: (_) @class.name)
(object_definition name: (_) @object.name)
(trait_definition name: (_) @trait.name)
(function_definition name: (_) @fn.name)
(type_definition name: (type_identifier) @typedef.name)
(template_body (val_definition pattern: (identifier) @field.name))
(template_body (var_definition pattern: (identifier) @field.name))
(import_declaration) @import
(function_definition
  (annotation name: (type_identifier) @_sa)
  name: (_) @test.entry
  (#eq? @_sa "Test"))
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("object.name", "object", "class"),
        ("trait.name", "trait", "class"),
        ("fn.name", "function", "fn"),
        ("typedef.name", "type", "type"),
        // #757: class/object/trait `val`/`var` members → `field:Owner.name`.
        // Anchored to `template_body` so local `val`s in method bodies are not
        // captured.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_definition", "class"),
        ("object_definition", "class"),
        ("trait_definition", "class"),
    ],
    decl_kinds: &[],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_scala::LANGUAGE),
};

/// Parse a Scala source file into graph nodes and edges.
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
        let path = dir.path().join("empty.scala");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.scala").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn expression_bodied_def_span_stops_before_the_next_def() {
        // tree-sitter-scala keeps an expression-bodied `function_definition`
        // open across the newline and the next line's indent, so the raw end
        // row lands on the following declaration. Two adjacent defs then own
        // the boundary line and caller attribution has to break the tie by
        // NodeId (#527 fallout). Spans must not overlap.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.scala");
        std::fs::write(&path, "trait T {\n  def a =\n    1\n  def b =\n    2\n}\n").unwrap();
        let out = parse("corp", &path, "t.scala").unwrap();
        let mut spans: Vec<(u32, u32)> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "method")
            .map(|n| (n.line.unwrap_or(0), n.end_line.unwrap_or(0)))
            .collect();
        spans.sort_unstable();
        assert_eq!(spans, vec![(2, 3), (4, 5)]);
    }

    #[test]
    fn parse_class_trait_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.scala");
        std::fs::write(
            &path,
            "import scala.collection._\nclass Foo\ntrait Bar\nobject Baz\ndef qux() = {}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.scala").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"trait"));
        assert!(kinds.contains(&"object"));
        assert!(kinds.contains(&"function"));
        assert!(kinds.contains(&"import"));
    }
}
