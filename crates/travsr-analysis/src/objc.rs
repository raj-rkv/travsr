//! Phase A parser for Objective-C source files using tree-sitter.
//!
//! `.mm` files (ObjC++) are accepted; the grammar parses ObjC constructs
//! correctly and silently skips C++-only syntax.

use std::path::Path;

use travsr_core::Language;

use crate::generic::{parse_with_config, LanguageConfig};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::ObjectiveC,
    // `.mm` files are ObjC++ (C++ mixed with ObjC). tree-sitter-objc parses ObjC
    // constructs correctly but silently skips C++-only syntax (templates, lambdas,
    // `std::` usage). Phase A nodes will be incomplete for `.mm` files that lean
    // heavily on C++ — this is acceptable for structural indexing purposes.
    extensions: &["m", "mm"],
    // Class name: "@interface" immediately followed by the first identifier.
    // The positional anchor prevents capturing the superclass identifier.
    //
    // Method name: in tree-sitter-objc's CST, `method_selector_no_list` and
    // `keyword_selector` are NOT named nodes: every selector keyword collapses
    // into a separate direct `(identifier)` child of
    // `method_definition`/`method_declaration`. The anchor after `(method_type)`
    // captures the leading keyword only, so exactly one node is emitted per
    // method; `full_selector` (via `name_hook`) then rebuilds the whole selector
    // from that node's siblings, so `setWidth:(int)w height:(int)h` is stored as
    // `setWidth:height:` rather than as a bare `setWidth` that collides with
    // every other selector starting the same way.
    queries: r#"
(class_interface "@interface" . (identifier) @class.name)
(class_implementation "@implementation" . (identifier) @impl.name)
(protocol_declaration "@protocol" . (identifier) @protocol.name)
(method_definition (method_type) . (identifier) @fn.name)
(method_declaration (method_type) . (identifier) @fn.name)
(function_definition declarator: (function_declarator declarator: (identifier) @fn.name))
(instance_variable (struct_declaration (struct_declarator (identifier) @field.name)))
(property_declaration (struct_declaration (struct_declarator (identifier) @field.name)))
(preproc_include path: (_) @import)
(module_import (identifier) @import)
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("impl.name", "impl", "impl"),
        ("protocol.name", "protocol", "protocol"),
        ("fn.name", "function", "fn"),
        // #757: ivars (`@interface { int _x; }`) and `@property` declarations →
        // `field:Owner.name`, contained by the interface/implementation.
        ("field.name", "field", "field"),
        ("import", "import", "import"),
    ],
    method_containers: &[
        ("class_implementation", "impl"),
        ("class_interface", "class"),
        ("protocol_declaration", "protocol"),
    ],
    decl_kinds: &["function_definition"],
    type_refinements: &[],
    post_parse: None,
    name_hook: Some(full_selector),
    get_grammar: || tree_sitter::Language::new(tree_sitter_objc::LANGUAGE),
};

/// Rebuild an Objective-C method's full selector from the captured leading
/// keyword, so each selector is a distinct node.
///
/// The query captures only the first `identifier` after `(method_type)`, but a
/// selector is spelled across alternating sibling nodes:
///
/// ```text
/// method_definition
///   identifier "policyWithPinningMode"   ← the captured node
///   method_parameter ":(AFSSLPinningMode)pinningMode"
///   identifier "withPinnedCertificates"
///   method_parameter ":(NSSet *)pinnedCertificates"
///   compound_statement "{ ... }"
/// ```
///
/// Parameter *names* (`pinningMode`) are nested inside `method_parameter`, never
/// direct children, so the direct `identifier` children are keyword parts only.
/// The one exception is a trailing argument-less macro: both
/// `- (instancetype)init NS_DESIGNATED_INITIALIZER;` and
/// `- (id)initWithFrame:(CGRect)f NS_DESIGNATED_INITIALIZER;` put the macro in
/// the same position a further keyword would take, so nothing *before* it tells
/// the two apart. What comes *after* does: a real keyword is always followed by
/// its `method_parameter`, a trailing macro never is. A keyword is therefore
/// held pending and committed only once the next sibling proves it, and a macro
/// left pending when the walk ends is dropped. (A macro that carries an
/// argument list, `NS_SWIFT_NAME(...)` or `__attribute__((...))`, parses as its
/// own node kind nested inside the preceding `method_parameter` and never
/// reaches this walk at all.)
///
/// `comment` is a tree-sitter extra and appears as an ordinary sibling between
/// selector parts, so a wrapped Cocoa signature with a trailing `//` comment
/// must step over it rather than end there. A preprocessor conditional inside a
/// selector is not a `preproc_*` sibling: tree-sitter-objc recovers it as an
/// `ERROR` sibling, which is stepped over the same way rather than ending the
/// selector at the `#if`. Recovery there is approximate, since a keyword the
/// `ERROR` node happens to swallow is lost, but the result keeps the selector's
/// arity instead of truncating it onto a genuinely shorter method's key.
///
/// Stepping over `ERROR` needs a bound, because a *body* tree-sitter cannot
/// parse is recovered as `ERROR` siblings too: `- (void)dealloc { [self.p
/// removeObserver:self forKeyPath:@"f"]; }` yields `ERROR "{ [self."`, then
/// `identifier "removeObserver"`, then `method_parameter ":self"`, all as
/// direct children of the method, with no `compound_statement` sibling and no
/// `body` field to stop on. Unbounded, the walk absorbs the body's first
/// message send and names the method `deallocremoveObserver:forKeyPath:`.
/// The bound is positional rather than a judgement about which `ERROR` nodes
/// are safe: a selector cannot continue past the `{` that opens the body, so
/// the walk ends at the first `{` in the source after the captured keyword,
/// wherever it falls. That offset holds whether or not the body parses, since
/// the brace is in the source either way, and a declaration with no body has
/// no `{` and still ends on its `;`. Braces inside a `comment` do not count.
///
/// Each `method_parameter` contributes one `:`, including a nameless one
/// (`- (void)anon:(int)a :(int)b` → `anon::`), matching the real selector.
/// A method with no parameters keeps its bare name and no trailing colon
/// (`- (void)reload` → `reload`), again matching the real selector.
///
/// Returns `None` for any capture that is not a method selector, leaving the
/// captured text in place.
fn full_selector(cap: tree_sitter::Node, sig_prefix: &str, source: &[u8]) -> Option<String> {
    if sig_prefix != "fn" {
        return None;
    }
    let parent = cap.parent()?;
    if !matches!(parent.kind(), "method_definition" | "method_declaration") {
        return None;
    }

    let mut selector = cap.utf8_text(source).ok()?.trim().to_string();
    if selector.is_empty() {
        return None;
    }

    // Walk the siblings after the captured keyword. `pending` holds an
    // identifier that is a keyword part only if a `method_parameter` follows
    // it; if the walk ends first it was a trailing macro and is dropped.
    let mut pending: Option<&str> = None;
    let mut consumed = cap.end_byte();
    let mut sibling = cap.next_sibling();
    while let Some(node) = sibling {
        // Stop at the `{` that opens the body. `comment` is excluded because a
        // brace inside one is text, not the body. `{` is ASCII, so it can never
        // be a UTF-8 continuation byte and a raw byte scan is exact.
        if node.kind() != "comment"
            && source
                .get(consumed..node.end_byte())
                .is_some_and(|span| span.contains(&b'{'))
        {
            break;
        }
        consumed = node.end_byte();
        match node.kind() {
            // Sits between selector parts without being one: step over it and
            // leave `pending` alone.
            "comment" | "ERROR" => {}
            "method_parameter" => {
                if let Some(keyword) = pending.take() {
                    selector.push_str(keyword);
                }
                selector.push(':');
            }
            "identifier" => pending = Some(node.utf8_text(source).ok()?.trim()),
            // The body, the terminating `;`, `, ...` on a variadic method: the
            // selector is complete.
            _ => break,
        }
        sibling = node.next_sibling();
    }

    Some(selector)
}

/// Parse an Objective-C source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

/// Bytes of a header inspected by [`header_is_objc`]. Declarations that
/// identify a header's dialect appear near the top; this bounds the read on a
/// generated header without changing the answer for a real one.
const HEADER_SNIFF_BYTES: usize = 64 * 1024;

/// Whether an ambiguous `.h` header is Objective-C, judged from its own text.
///
/// `.h` is shared by C, C++ and Objective-C, and the extension cannot say
/// which. The caller previously decided this from a single repo-wide "does any
/// `.m`/`.mm` exist" flag applied to every header at once, so one Objective-C
/// file anywhere claimed every header in the repo — including C++ headers in
/// unrelated directories.
///
/// The cost is silent symbol loss, not a broken edge: the Objective-C grammar
/// cannot parse C++ declarations, so a misfiled header yields a file node and
/// nothing else. `class Animal { public: void speak(); };` contributed no
/// `fn:speak`, and `search_symbol` / `find_references` / `get_callers` then
/// answered "not found" for a symbol that is plainly in the source, with no
/// error to explain why.
///
/// The header's own text settles it: Objective-C declarations have no C or C++
/// spelling, so finding one is conclusive. `None` means the text carries no
/// dialect marker either way (a plain C-style declarations header), leaving the
/// choice to the caller's repo-level signal — which is the previous behaviour,
/// and the right default for a repo already known to be Objective-C.
///
/// Sniffing is deliberately lexical: raw substring matching, with no comment or
/// string-literal stripping. A marker inside a comment counts, `#import` counts
/// even though clang accepts it in C and C++, and `class ` matches an
/// Objective-C `@class Foo;` forward declaration. All of those are rare, all
/// only reachable inside a repo already known to contain Objective-C, and the
/// alternative is a second parser to decide which parser to use.
///
/// `Some(false)` means "not Objective-C", which routes to the C grammar. It
/// does not mean the header is parsed as C++ — plain `.h` maps to
/// `Language::C` regardless.
pub fn header_is_objc(source: &str) -> Option<bool> {
    let head = source.get(..HEADER_SNIFF_BYTES).unwrap_or(source);

    // Objective-C wins outright when present: a header carrying `@interface`
    // is Objective-C even if it also uses C++ constructs (ObjC++ headers do).
    const OBJC: &[&str] = &[
        "@interface",
        "@protocol",
        "@implementation",
        "@property",
        "@end",
        "NS_ASSUME_NONNULL",
        "#import",
    ];
    if OBJC.iter().any(|m| head.contains(m)) {
        return Some(true);
    }

    // C++-only spellings. None of these parse as Objective-C's supersets of C,
    // so their presence rules Objective-C out.
    const CPP: &[&str] = &[
        "template<",
        "template <",
        "namespace ",
        "public:",
        "private:",
        "protected:",
        "std::",
        "class ",
    ];
    if CPP.iter().any(|m| head.contains(m)) {
        return Some(false);
    }

    None
}

#[cfg(test)]
mod selector_tests {
    use std::io::Write as _;

    /// Signatures a source snippet produces, in emission order.
    fn signatures(src: &str) -> Vec<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("Fixture.m");
        std::fs::File::create(&path)
            .expect("create")
            .write_all(src.as_bytes())
            .expect("write");
        super::parse("corp", &path, "Fixture.m")
            .expect("parse")
            .nodes
            .into_iter()
            .map(|n| n.vname.signature)
            .collect()
    }

    #[test]
    fn a_multi_part_selector_keeps_every_keyword() {
        // The bug this covers: both selectors used to collapse onto
        // `method:Foo.policyWithPinningMode`, so the unique index kept one node
        // and the two-argument method had none at all.
        let sigs = signatures(
            "@implementation Foo\n\
             + (id)policyWithPinningMode:(int)m { return nil; }\n\
             + (id)policyWithPinningMode:(int)m withPinnedCertificates:(id)c { return nil; }\n\
             @end\n",
        );
        assert!(sigs.contains(&"method:Foo.policyWithPinningMode:".to_string()));
        assert!(
            sigs.contains(&"method:Foo.policyWithPinningMode:withPinnedCertificates:".to_string())
        );
    }

    #[test]
    fn a_no_argument_selector_takes_no_trailing_colon() {
        let sigs = signatures("@implementation Foo\n- (void)reload { }\n@end\n");
        assert!(sigs.contains(&"method:Foo.reload".to_string()));
    }

    #[test]
    fn a_trailing_macro_is_not_read_as_a_selector_keyword() {
        // `NS_DESIGNATED_INITIALIZER` sits in the same direct-child position a
        // second selector keyword would, but no `method_parameter` follows it,
        // so it is never committed. An availability macro parses as its own
        // node kind and is not a keyword either.
        let sigs = signatures(
            "@interface Foo\n\
             - (instancetype)init NS_DESIGNATED_INITIALIZER;\n\
             + (instancetype)new NS_UNAVAILABLE;\n\
             @end\n",
        );
        assert!(sigs.contains(&"method:Foo.init".to_string()));
        assert!(sigs.contains(&"method:Foo.new".to_string()));
    }

    #[test]
    fn a_trailing_macro_after_a_parameter_is_still_not_a_keyword() {
        // The arity-0 guard above held under a lookbehind rule too; these do
        // not. Once the selector has one parameter, a lookbehind sees the macro
        // as licensed and appends it, producing
        // `initWithFrame:NS_DESIGNATED_INITIALIZER`, which no Phase B selector
        // can ever match.
        let sigs = signatures(
            "@interface Foo\n\
             - (instancetype)initWithFrame:(CGRect)f NS_DESIGNATED_INITIALIZER;\n\
             - (id)responseObjectForResponse:(id)r data:(id)d NS_SWIFT_NOTHROW;\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Foo.initWithFrame:".to_string()),
            "got {sigs:?}"
        );
        assert!(
            sigs.contains(&"method:Foo.responseObjectForResponse:data:".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_comment_between_keywords_does_not_truncate_the_selector() {
        // `comment` is a tree-sitter extra, so it lands as a real sibling in the
        // middle of the selector. Ending the walk there gave
        // `requestWithMethod:`, i.e. the collision this hook exists to prevent.
        let sigs = signatures(
            "@implementation Client\n\
             - (id)requestWithMethod:(NSString *)method   // HTTP verb\n\
                           URLString:(NSString *)url\n\
                          parameters:(id)params { return nil; }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Client.requestWithMethod:URLString:parameters:".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_comment_before_the_first_colon_keeps_the_selector_a_selector() {
        // Worse than truncation: ending at the comment produced `method:Pre.pre`,
        // a colon-less leaf indistinguishable from a genuine no-argument method.
        let sigs = signatures("@interface Pre\n- (void)pre /* c */ :(int)a and:(int)b;\n@end\n");
        assert!(
            sigs.contains(&"method:Pre.pre:and:".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_brace_inside_a_comment_is_not_the_body() {
        // The body bound is a `{` in the source, and a comment between selector
        // parts is source too. Counting its braces would truncate the selector
        // to `requestWithMethod:`, the collision this hook exists to prevent.
        let sigs = signatures(
            "@implementation Client\n\
             - (id)requestWithMethod:(NSString *)method   // opens { here\n\
                           URLString:(NSString *)url { return nil; }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Client.requestWithMethod:URLString:".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_nameless_keyword_part_still_contributes_its_colon() {
        // `- (void)anon:(int)a :(int)b` really is the selector `anon::`.
        let sigs = signatures("@interface Foo\n- (void)anon:(int)a :(int)b;\n@end\n");
        assert!(sigs.contains(&"method:Foo.anon::".to_string()));
    }

    #[test]
    fn a_body_that_fails_to_parse_does_not_extend_the_selector() {
        // AFURLSessionManager.m:149. tree-sitter recovers the whole body as
        // `ERROR "{ [self."` plus loose `identifier`/`method_parameter`
        // siblings, so the walk used to read the body's first message send as
        // more selector and emit `method:Foo.deallocremoveObserver:forKeyPath:`.
        let sigs = signatures(
            "@implementation Foo\n\
             - (void)dealloc {\n\
             \x20   [self.downloadProgress removeObserver:self forKeyPath:@\"fractionCompleted\"];\n\
             }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Foo.dealloc".to_string()),
            "got {sigs:?}"
        );
        assert!(
            !sigs
                .iter()
                .any(|s| s.starts_with("method:Foo.dealloc") && s != "method:Foo.dealloc"),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_body_of_nested_brackets_and_macros_does_not_extend_the_selector() {
        // AFURLSessionManager.m:425 and :682. A bare `NSAssert(...)` macro, a
        // nested bracket expression and a message send inside a `return` all
        // recover as loose siblings; none of them is selector.
        let sigs = signatures(
            "@implementation Foo\n\
             - (void)af_resume {\n\
             \x20   NSAssert([self respondsToSelector:@selector(state)], @\"no state\");\n\
             }\n\
             - (NSArray *)tasks {\n\
             \x20   return [self tasksForKeyPath:NSStringFromSelector(_cmd)];\n\
             }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Foo.af_resume".to_string()),
            "got {sigs:?}"
        );
        assert!(
            sigs.contains(&"method:Foo.tasks".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_parameterised_selector_stops_at_its_own_body() {
        // AFURLSessionManager.m:898. The selector is complete before the body,
        // but the body is a chain of `@selector(...)` comparisons that recover
        // as `method_parameter` siblings, so it used to become
        // `respondsToSelector:URLSession:::nil:::::nil::::nil::::`.
        let sigs = signatures(
            "@implementation Foo\n\
             - (BOOL)respondsToSelector:(SEL)selector {\n\
             \x20   if (selector == @selector(URLSession:didReceiveChallenge:completionHandler:)) {\n\
             \x20       return self.sessionDidReceiveAuthenticationChallenge != nil;\n\
             \x20   }\n\
             \x20   return [super respondsToSelector:selector];\n\
             }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Foo.respondsToSelector:".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_preproc_conditional_inside_a_selector_still_does_not_truncate_it() {
        // The `ERROR`-stepping this fix bounds must still hold before the body:
        // a `#if` between selector parts is an `ERROR` sibling, and ending the
        // walk there would key a three-part selector as `sendRequest:`.
        let sigs = signatures(
            "@interface Foo\n\
             - (void)sendRequest:(id)r\n\
             #if TARGET_OS_IOS\n\
             \x20            queue:(id)q\n\
             #endif\n\
             \x20         handler:(id)h;\n\
             @end\n",
        );
        assert!(
            sigs.iter()
                .any(|s| s.starts_with("method:Foo.sendRequest:") && s.matches(':').count() >= 3),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_trailing_macro_on_a_defined_method_is_still_not_a_keyword() {
        // The arity-0 macro guard has to survive the body bound: the macro is
        // pending when the walk stops at `{`, and a pending keyword is dropped.
        let sigs = signatures(
            "@implementation Foo\n\
             - (instancetype)init NS_DESIGNATED_INITIALIZER {\n\
             \x20   return [super init];\n\
             }\n\
             @end\n",
        );
        assert!(
            sigs.contains(&"method:Foo.init".to_string()),
            "got {sigs:?}"
        );
    }

    #[test]
    fn a_c_function_in_an_objc_file_is_untouched() {
        // The hook only fires under a `method_definition`/`method_declaration`
        // parent, so a plain C function keeps the captured text.
        let sigs = signatures("int add(int a, int b) { return a + b; }\n");
        assert!(sigs.contains(&"fn:add".to_string()));
    }
}

#[cfg(test)]
mod header_sniff_tests {
    use super::{header_is_objc, HEADER_SNIFF_BYTES};

    #[test]
    fn objc_declarations_are_conclusive() {
        assert_eq!(header_is_objc("@interface Animal\n@end\n"), Some(true));
        assert_eq!(header_is_objc("@protocol Speaker\n@end\n"), Some(true));
        assert_eq!(
            header_is_objc("#import <Foundation/Foundation.h>\n"),
            Some(true)
        );
    }

    #[test]
    fn cpp_declarations_rule_objc_out() {
        // #610: the exact shape that was being misfiled. A C++ header in a repo
        // that happens to contain an unrelated `.m` file.
        assert_eq!(
            header_is_objc("#pragma once\nclass Animal { public: void speak(); };\n"),
            Some(false)
        );
        assert_eq!(
            header_is_objc("template <typename T> T id(T x);\n"),
            Some(false)
        );
        assert_eq!(header_is_objc("namespace zoo { void f(); }\n"), Some(false));
    }

    #[test]
    fn objc_wins_over_cpp_markers_in_an_objcpp_header() {
        // ObjC++ headers legitimately carry both; the Objective-C parser is the
        // one that can read `@interface`, so it must win.
        let src = "#import <Foundation/Foundation.h>\nclass Impl;\n@interface Wrapper\n@end\n";
        assert_eq!(header_is_objc(src), Some(true));
    }

    #[test]
    fn a_plain_c_header_is_ambiguous_and_defers_to_the_caller() {
        // No dialect marker: the caller's repo-level signal decides, which
        // preserves the pre-#610 behaviour for genuinely ambiguous headers.
        assert_eq!(
            header_is_objc("#pragma once\nint add(int a, int b);\n"),
            None
        );
        assert_eq!(header_is_objc(""), None);
    }

    #[test]
    fn a_non_char_boundary_cut_falls_back_instead_of_panicking() {
        // `get(..N)` returns None when byte 65536 lands inside a multi-byte
        // character, and the whole string is scanned instead. This covers that
        // fallback, not truncation — the marker here is still found.
        let src = format!("// {}\n@interface Late\n@end\n", "é".repeat(40_000));
        assert_eq!(header_is_objc(&src), Some(true));
    }

    #[test]
    fn a_marker_past_the_sniff_bound_is_not_seen() {
        // The truncation path proper: ASCII padding, so HEADER_SNIFF_BYTES is a
        // clean boundary and the slice really is cut. A marker beyond it is
        // invisible, which yields None and leaves the decision to the caller's
        // repo-level signal rather than to a partial read.
        let src = format!(
            "// {}\n@interface Late\n@end\n",
            "x".repeat(HEADER_SNIFF_BYTES)
        );
        assert_eq!(header_is_objc(&src), None);
        // The same marker inside the bound is found, so the difference above is
        // the bound and not the content.
        assert_eq!(header_is_objc("// x\n@interface Early\n@end\n"), Some(true));
    }

    /// The routing boolean is only half the claim. #630 was a *symbol* loss:
    /// the recovered header has to actually yield `fn:speak`.
    ///
    /// Worth pinning at parse level because plain `.h` maps to `Language::C`,
    /// not `Cpp` — only `.hpp`/`.hh`/`.hxx` map to C++ — so the symbol survives
    /// on tree-sitter-c error-recovering `void speak();` out of a C++ class
    /// body. That works today, and a grammar bump could take it away with every
    /// boolean assertion still green.
    #[test]
    fn a_recovered_cpp_header_still_yields_its_symbols() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("animal.h");
        std::fs::write(
            &path,
            "#pragma once\nclass Animal { public: void speak(); };\n",
        )
        .unwrap();

        let out = crate::c::parse("", &path, "cpp/animal.h").expect("C grammar parses the header");
        assert!(
            out.nodes.iter().any(|n| n.vname.signature == "fn:speak"),
            "the C grammar must still recover fn:speak from a C++ header: {:?}",
            out.nodes
                .iter()
                .map(|n| &n.vname.signature)
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.m");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.m").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    #[test]
    fn parse_interface_and_impl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.m");
        std::fs::write(
            &path,
            "#import <Foundation/Foundation.h>\n\
             @interface MyClass : NSObject\n- (void)doThing;\n@end\n\
             @implementation MyClass\n- (void)doThing {}\n@end\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.m").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"), "class from @interface");
        assert!(kinds.contains(&"impl"), "impl from @implementation");
        assert!(kinds.contains(&"import"), "import node");
        // N1: the selector is qualified by its enclosing type.
        assert!(
            out.nodes
                .iter()
                .any(|n| n.kind == "method" && n.vname.signature == "method:MyClass.doThing"),
            "expected method:MyClass.doThing; got {:?}",
            out.nodes
                .iter()
                .map(|n| &n.vname.signature)
                .collect::<Vec<_>>()
        );
    }
}
