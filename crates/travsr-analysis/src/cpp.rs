//! Phase A parser for C++ source files using tree-sitter.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Cpp,
    extensions: &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
    queries: r#"
(class_specifier name: (type_identifier) @class.name)
(struct_specifier name: (type_identifier) @struct.name)
(union_specifier name: (type_identifier) @union.name)
(enum_specifier name: (type_identifier) @enum.name)
(namespace_definition name: (namespace_identifier) @namespace.name)
(alias_declaration name: (type_identifier) @using.name)
(function_declarator declarator: (identifier) @fn.name)
(function_declarator declarator: (field_identifier) @fn.name)
(field_declaration declarator: (field_identifier) @field.name)
(preproc_def name: (identifier) @macro.name)
(preproc_function_def name: (identifier) @macro.name)
(preproc_include path: (_) @import)
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("struct.name", "struct", "struct"),
        ("union.name", "union", "struct"),
        ("enum.name", "enum", "enum"),
        ("namespace.name", "namespace", "namespace"),
        ("using.name", "typedef", "type"),
        ("fn.name", "function", "fn"),
        // #757: data members → `field:Owner.name`. A method's declarator is a
        // `function_declarator` (captured as `fn.name` above), so only true data
        // members reach this capture.
        ("field.name", "field", "field"),
        // N4e: `#define` object-like and function-like macros as first-class nodes.
        ("macro.name", "macro", "macro"),
        ("import", "import", "import"),
    ],
    // #757: `union_specifier` is a data-member container too; without it a
    // union member orphans as an unqualified `field:name` that collides across
    // every union (matching the C fix in `c.rs`).
    method_containers: &[
        ("class_specifier", "class"),
        ("struct_specifier", "struct"),
        ("union_specifier", "struct"),
    ],
    decl_kinds: &["function_definition"],
    type_refinements: &[],
    post_parse: None,
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_cpp::LANGUAGE),
};

/// Whether a `.h` header is C++ rather than C.
///
/// `.h` is genuinely ambiguous and the ecosystem leans C++: LLVM, Google style
/// and most C++ projects use `.h`, not `.hpp`. Extension alone sends all of
/// them to the C grammar, which cannot see a class, a namespace or a template,
/// so a header-declared API is invisible to the graph and its SCIP definitions
/// have nothing to unify against.
///
/// Decided on content, the way clangd and linguist do it, because nothing in
/// the filename can answer it.
///
/// The markers are ones C **cannot** contain. Deliberately excluded:
///
/// - `#ifdef __cplusplus` and `extern "C"`, which are how a *C* header
///   announces it is safe to include from C++. Treating them as C++ markers
///   would misclassify exactly the headers that took care to be portable.
/// - `//` comments and `bool`, which are C99 and later.
///
/// False negatives are the safe direction: a C++ header with none of these is
/// declaration-only and parses acceptably as C, which is today's behaviour. A
/// false positive is not, so every marker is anchored on a boundary rather
/// than matching a bare substring.
pub fn header_is_cxx(source: &str) -> bool {
    // Scope resolution is impossible in C and appears in almost any C++ header
    // that declares something qualified.
    if source.contains("::") {
        return true;
    }
    const MARKERS: &[&str] = &[
        "template<",
        "template <",
        "namespace ",
        "class ",
        "public:",
        "private:",
        "protected:",
        "virtual ",
        "friend ",
        "constexpr ",
        "nullptr",
    ];
    if MARKERS.iter().any(|m| source.contains(m)) {
        return true;
    }
    // `operator` needs a real word boundary on both sides, which a substring
    // match cannot give (#708 review). A bare `contains("operator")` matched
    // ordinary C like `int apply_operator(...)`; anchoring only the right side
    // still matched it, and anchoring only the left side still matched
    // `enum operator_type`. Both sides, or the overload keyword is
    // indistinguishable from an identifier that happens to contain it.
    contains_keyword(source, "operator")
}

/// Whether `word` appears in `source` as a whole identifier, rather than as a
/// substring of a longer one.
///
/// A C identifier continues through ASCII alphanumerics and `_`, so a match is
/// a keyword only when neither neighbour can continue it: `operator+=` and
/// `operator()` qualify, `apply_operator` and `operator_type` do not.
fn contains_keyword(source: &str, word: &str) -> bool {
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let bytes = source.as_bytes();
    let w = word.as_bytes();
    source.match_indices(word).any(|(i, _)| {
        let before_ok = i == 0 || !is_ident(bytes[i - 1]);
        let after = i + w.len();
        let after_ok = after >= bytes.len() || !is_ident(bytes[after]);
        before_ok && after_ok
    })
}

/// Parse a C++ source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #708 review: `"operator"` was an unanchored substring, so ordinary C
    /// identifiers that merely contain it sent the header to the C++ grammar.
    /// A false positive, which is the direction this heuristic exists to avoid.
    #[test]
    fn c_identifiers_containing_operator_are_not_cxx() {
        for c_header in [
            "enum operator_type { OP_ADD, OP_SUB };\n",
            "int apply_operator(int a, int b);\n",
            "static const char *bitwise_operator_table[8];\n",
            "void unary_operator_apply(struct expr *e);\n",
        ] {
            assert!(
                !header_is_cxx(c_header),
                "plain C misclassified as C++: {c_header:?}"
            );
        }
    }

    /// The real C++ spellings must still be caught, or the anchoring traded one
    /// bug for another.
    #[test]
    fn real_operator_declarations_are_cxx() {
        for cxx in [
            "Widget& operator+=(int by);\n",
            "bool operator==(const Widget &a, const Widget &b);\n",
            "int operator()(int x) const;\n",
        ] {
            assert!(header_is_cxx(cxx), "missed a real operator decl: {cxx:?}");
        }
    }

    /// A C header that says nothing C++-shaped stays C, which is the safe
    /// direction and the pre-existing behaviour.
    #[test]
    fn a_plain_c_header_stays_c() {
        assert!(!header_is_cxx(
            "#ifndef H\n#define H\nstruct Point { int x; };\nint add(int a, int b);\n#endif\n"
        ));
    }

    /// A C header written to be included from C++ must not be misread as C++:
    /// those guards are how a C header declares itself portable.
    #[test]
    fn extern_c_guards_do_not_make_a_header_cxx() {
        assert!(!header_is_cxx(
            "#ifdef __cplusplus\nextern \"C\" {\n#endif\nint add(int, int);\n#ifdef __cplusplus\n}\n#endif\n"
        ));
    }

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.cpp");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.cpp").unwrap();
        assert_eq!(out.nodes.len(), 1, "file node only");
    }

    #[test]
    fn parse_class_and_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.cpp");
        std::fs::write(&path, "namespace ns { class Foo {}; }\n").unwrap();
        let out = parse("corp", &path, "sample.cpp").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        assert!(kinds.contains(&"namespace"));
    }

    #[test]
    fn union_members_are_owner_qualified() {
        // #757 re-review: `union_specifier` was captured as a node but was not a
        // `method_container`, so its data members orphaned as unqualified
        // `field:name`, colliding across every union. Adding it (matching
        // `c.rs`) qualifies them to `field:U.name`. Class/struct members are
        // unaffected.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("u.cpp");
        std::fs::write(
            &path,
            "class C { int a; };\nstruct S { int c; };\nunion U { int d; float e; };\n",
        )
        .unwrap();
        let out = parse("corp", &path, "u.cpp").unwrap();
        let field_sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "field")
            .map(|n| n.vname.signature.as_str())
            .collect();
        for want in ["field:C.a", "field:S.c", "field:U.d", "field:U.e"] {
            assert!(
                field_sigs.contains(&want),
                "missing {want}; got {field_sigs:?}"
            );
        }
        assert!(
            !field_sigs
                .iter()
                .any(|s| *s == "field:d" || *s == "field:e"),
            "union members must not orphan unqualified; got {field_sigs:?}"
        );
    }

    #[test]
    fn n4e_define_macros_are_nodes() {
        // N4e: object-like and function-like `#define` macros become `macro:` nodes.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("macros.cpp");
        std::fs::write(&path, "#define PI 3.14\n#define SQ(x) ((x)*(x))\n").unwrap();
        let out = parse("corp", &path, "macros.cpp").unwrap();
        let sigs: Vec<&str> = out
            .nodes
            .iter()
            .filter(|n| n.kind == "macro")
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert!(sigs.contains(&"macro:PI"), "got {sigs:?}");
        assert!(sigs.contains(&"macro:SQ"), "got {sigs:?}");
    }
}
