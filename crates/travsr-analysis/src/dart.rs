//! Phase A parser for Dart source files using tree-sitter.
//!
//! Phase B (call-site edges via the AOT emitter) lives in `phase_b_dart`.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Dart,
    extensions: &["dart"],
    queries: r#"
(class_declaration     name: (identifier) @class.name)
(mixin_declaration     name: (identifier) @mixin.name)
(extension_declaration name: (identifier) @extension.name)
(enum_declaration      name: (identifier) @enum.name)
(type_alias . (type_identifier) @typedef.name)
(function_signature    name: (identifier) @fn.name)
(class_body (class_member (declaration (initialized_identifier_list (initialized_identifier name: (identifier) @field.name)))))
(import_or_export)     @import
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("mixin.name", "mixin", "class"),
        ("extension.name", "extension", "class"),
        ("enum.name", "enum", "enum"),
        ("typedef.name", "typedef", "type"),
        ("fn.name", "function", "fn"),
        // #757: instance fields → `field:Owner.name`. Anchored under `class_body`
        // so local variables in method bodies are not captured.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_declaration", "class"),
        ("mixin_declaration", "class"),
        ("extension_declaration", "class"),
        ("enum_declaration", "enum"),
    ],
    // N2: the `fn.name` capture sits in a `function_signature`, whose parent is
    // just the signature — its `end_row` is the signature line, excluding the
    // body. Walk up to the enclosing full declaration (`function_declaration`
    // for top-level functions, `method_declaration` for class/mixin methods) so
    // the span covers the body. Without this, every call site inside a Dart
    // function fails span-containment and its `ref/call` mis-attributes to the
    // file node instead of the enclosing function (E5).
    decl_kinds: &["function_declaration", "method_declaration"],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_dart::LANGUAGE),
};

/// Parse a Dart source file into graph nodes and edges (Phase A structural only).
///
/// For Phase B call-site edges use [`crate::phase_b_dart::extract_native_phase_b`].
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n2_function_and_method_spans_cover_body() {
        // N2 (E5): a top-level function and a class method must span their whole
        // body, so a call site inside attributes to the enclosing function (not
        // the file). Before decl_kinds, both spans ended at the signature line.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.dart");
        std::fs::write(
            &path,
            // line 1: class Animal {
            // line 2:   String describe() {
            // line 3:     return "a";
            // line 4:   }
            // line 5: }
            // line 6: void main() {
            // line 7:   print(1);
            // line 8: }
            "class Animal {\n  String describe() {\n    return \"a\";\n  }\n}\nvoid main() {\n  print(1);\n}\n",
        )
        .unwrap();
        let out = parse("corp", &path, "s.dart").unwrap();

        let main = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "fn:main")
            .expect("fn:main");
        assert_eq!(main.line, Some(6));
        assert_eq!(
            main.end_line,
            Some(8),
            "main span must reach the body brace"
        );

        let describe = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:Animal.describe")
            .expect("method:Animal.describe");
        assert_eq!(describe.line, Some(2));
        assert_eq!(
            describe.end_line,
            Some(4),
            "method span must reach the body brace"
        );
    }

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.dart");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.dart").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_class_and_enum() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.dart");
        std::fs::write(
            &path,
            "import 'dart:core';\nclass Foo {}\nmixin Bar {}\nenum Status { ok, fail }\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.dart").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"mixin"));
        assert!(kinds.contains(&"enum"));
        assert!(kinds.contains(&"import"));
    }
}
