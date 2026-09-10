//! RFC-027 section 8.2 — detection-only reference set for the live LSP lane.
//!
//! The three languages with a native Phase B extractor (Rust, TypeScript/JS,
//! Python) get their on-save reference set from that extractor: a fully typed
//! record carrying the caller's identity, the receiver type where it could be
//! recovered, and a per-language signature key. The other thirteen have Phase A
//! structural nodes and no on-save reference-detection pass at all, which is the
//! single reason the live lane never reached them — resolution, node mapping,
//! target production, emit, ratification, and the precision meter are already
//! language-agnostic.
//!
//! This module closes exactly that gap and nothing else. It answers *"which
//! references exist in this buffer, where, and of what kind"* and stops there:
//!
//! - **No receiver-type recovery.** `x.Foo()` yields the name `Foo` and its
//!   line. What `x` is at that position is the editor's language server's
//!   question, and asking it is the whole point of the LSP lane (RFC-027
//!   section 7.3b).
//! - **No signature building.** A signature is a claim about identity, and
//!   identity belongs to SCIP (section 8.2 fencing rule). Nothing here mints a
//!   VName.
//! - **No resolution.** The daemon maps positions to nodes against the graph it
//!   already owns.
//!
//! Because it produces no receiver type and no signature key, the fail-closed
//! lexical floor (section 7.3a) cannot consume this output, so these languages
//! run the **editor lane only** (section 8.3). With no language server installed
//! they abstain and keep today's commit-gated behavior exactly, which is the
//! RFC's zero-regression floor (section 7.3c).
//!
//! ## Adding a language
//!
//! One [`LangSpec`]: a tree-sitter query using the four standard captures, and
//! (only where the grammar reuses one node for both) the [`MemberShape`] that
//! tells a callee apart from a field read. Then an arm in the daemon's
//! `live_language` gate and an entry in the extension's `SUPPORTED_LANGUAGES`.
//! Each language is measured against the per-`(language, kind)` precision gate
//! (section 12) once it is emitting.

use anyhow::Context as _;
use streaming_iterator::StreamingIterator as _;
use tree_sitter::{Node, Parser, Query, QueryCursor};

use travsr_core::{InheritanceRef, Language};

/// Upper bound on a buffer this pass will parse.
///
/// Matches the Phase A parsers' own limit. A generated file past this size is
/// not what the live lane exists for, and parsing it on every save would cost
/// more than the freshness is worth.
const MAX_SOURCE_BYTES: usize = 10 * 1024 * 1024;

/// One detected reference: the name as written and the line it sits on.
///
/// The line, not a column: the editor recovers the column by finding `name` on
/// the line, which is far cheaper than teaching every detector to carry UTF-16
/// offsets, and is what the daemon-driven target contract already asks of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRef {
    /// 1-based source line the reference appears on.
    pub line: u32,
    /// The referenced name exactly as written (`Foo` in `x.Foo()`).
    pub name: String,
}

/// Every reference in one buffer the live lane can act on, bucketed by the edge
/// kind it would become.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiveRefs {
    /// Call sites — `EdgeKind::RefCall`.
    pub calls: Vec<LiveRef>,
    /// Field / property reads that are not calls — `EdgeKind::RefField`.
    pub fields: Vec<LiveRef>,
    /// `extends` / `implements` style clauses — `EdgeKind::IsImplementation`.
    pub inheritance: Vec<InheritanceRef>,
}

impl LiveRefs {
    /// True when nothing was detected, so the caller can skip the whole pass.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.fields.is_empty() && self.inheritance.is_empty()
    }
}

/// How a language nests a member-access name inside a call.
///
/// Several grammars use one node for both `x.foo(…)` and `x.foo` — the member
/// access is simply the callee of a surrounding call in the first case. A
/// tree-sitter query cannot express "not the callee of a call", so a name
/// captured as `@sel.name` is classified from the tree instead. Languages whose
/// grammar already has distinct nodes for the two (Java, PHP, Ruby) need no
/// shape and capture `@call.name` / `@field.name` directly.
///
/// Getting this wrong costs correctness, not just tidiness: a field read that
/// became a `ref/call` would surface as a caller in `get_callers` and
/// `get_blast_radius` (#757).
struct MemberShape {
    /// The member-access expression the captured name sits inside. The captured
    /// node may be nested a level or two below it (Swift's `navigation_suffix`),
    /// so this is found by walking up, not by taking the parent.
    member_kind: &'static str,
    /// The call expression that member may be the callee of.
    call_kind: &'static str,
    /// Field naming the callee on the call node, or `None` when the grammar
    /// leaves it unlabelled and the callee is simply the first named child.
    callee_field: Option<&'static str>,
}

/// A language's detection queries and call shape.
struct LangSpec {
    /// Tree-sitter query using the four standard captures:
    ///
    /// - `@call.name` — unambiguously a call site.
    /// - `@sel.name` — a member-access name; [`MemberShape`] decides which it is.
    /// - `@field.name` — unambiguously a field read.
    /// - `@base.name` — the base type of an inheritance clause.
    query: &'static str,
    /// Required if and only if the query uses `@sel.name`.
    member: Option<MemberShape>,
}

/// Detect the live lane's reference set in `source`, or an empty set for a
/// language with no detector.
///
/// Returning empty rather than erroring for an unhandled language keeps the
/// daemon's per-language gate the single place a language is turned on: this
/// function is safe to call for anything, and adding a language here without
/// adding it there still ships nothing.
pub fn detect_live_refs(lang: Language, source: &[u8]) -> anyhow::Result<LiveRefs> {
    if source.len() > MAX_SOURCE_BYTES {
        return Ok(LiveRefs::default());
    }
    let Some((grammar, spec)) = spec_for(lang) else {
        return Ok(LiveRefs::default());
    };
    detect(lang, grammar, &spec, source)
}

fn detect(
    lang: Language,
    grammar: tree_sitter::Language,
    spec: &LangSpec,
    source: &[u8],
) -> anyhow::Result<LiveRefs> {
    let query = Query::new(&grammar, spec.query)
        .with_context(|| format!("compiling {} live-detect query", lang.as_str()))?;
    let capture_names: Vec<String> = query
        .capture_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .with_context(|| format!("loading {} grammar", lang.as_str()))?;
    // A parse that does not complete yields no references, which abstains for
    // the whole file. Fail-closed, and the commit-gated path still covers it.
    let Some(tree) = parser.parse(source, None) else {
        return Ok(LiveRefs::default());
    };

    let mut out = LiveRefs::default();
    let mut cursor = QueryCursor::new();
    let mut iter = cursor.matches(&query, tree.root_node(), source);
    while let Some(m) = iter.next() {
        for cap in m.captures {
            let Some(cap_name) = capture_names.get(cap.index as usize).map(String::as_str) else {
                continue;
            };
            let Ok(name) = cap.node.utf8_text(source) else {
                continue;
            };
            let line = cap.node.start_position().row as u32 + 1;
            match cap_name {
                "call.name" => out.calls.push(LiveRef {
                    line,
                    name: name.to_string(),
                }),
                "field.name" => out.fields.push(LiveRef {
                    line,
                    name: name.to_string(),
                }),
                "sel.name" => {
                    let r = LiveRef {
                        line,
                        name: name.to_string(),
                    };
                    match spec.member.as_ref() {
                        Some(shape) if is_call_callee(cap.node, shape) => out.calls.push(r),
                        // No shape declared for a `@sel.name` capture is a
                        // detector bug, not a user's. Treat it as a field read,
                        // the conservative half: a missed call costs recall,
                        // while a field wrongly called a call is the #757 defect.
                        _ => out.fields.push(r),
                    }
                }
                "base.name" => out.inheritance.push(InheritanceRef {
                    base_name: name.to_string(),
                    line,
                }),
                _ => {}
            }
        }
    }
    Ok(out)
}

/// True when the member access containing `name` is the callee of a call, so the
/// reference is `x.foo(…)` rather than a bare `x.foo`.
fn is_call_callee(name: Node<'_>, shape: &MemberShape) -> bool {
    // Walk up to the member-access node. Bounded because the capture sits at a
    // fixed, shallow depth inside it (one level for a `field:`-labelled name,
    // two for Swift's `navigation_suffix`), and an unbounded walk would happily
    // find an enclosing call several statements out and call it the callee.
    let mut member = name;
    for _ in 0..MEMBER_LOOKUP_DEPTH {
        if member.kind() == shape.member_kind {
            break;
        }
        match member.parent() {
            Some(p) => member = p,
            None => return false,
        }
    }
    if member.kind() != shape.member_kind {
        return false;
    }
    let Some(call) = member.parent() else {
        return false;
    };
    if call.kind() != shape.call_kind {
        return false;
    }
    match shape.callee_field {
        Some(field) => call.child_by_field_name(field) == Some(member),
        // Unlabelled grammars put the callee first, ahead of the argument list.
        None => call.named_child(0) == Some(member),
    }
}

/// How far above a captured name the member-access node may sit.
const MEMBER_LOOKUP_DEPTH: usize = 3;

/// The grammar and detection spec for `lang`, or `None` when it has no detector.
fn spec_for(lang: Language) -> Option<(tree_sitter::Language, LangSpec)> {
    let (grammar, query, member) = match lang {
        Language::Go => (
            tree_sitter::Language::new(tree_sitter_go::LANGUAGE),
            GO_QUERY,
            Some(MemberShape {
                member_kind: "selector_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::Java => (
            tree_sitter::Language::new(tree_sitter_java::LANGUAGE),
            JAVA_QUERY,
            None,
        ),
        Language::CSharp => (
            tree_sitter::Language::new(tree_sitter_c_sharp::LANGUAGE),
            CSHARP_QUERY,
            Some(MemberShape {
                member_kind: "member_access_expression",
                call_kind: "invocation_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::Cpp => (
            tree_sitter::Language::new(tree_sitter_cpp::LANGUAGE),
            CPP_QUERY,
            Some(MemberShape {
                member_kind: "field_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::C => (
            tree_sitter::Language::new(tree_sitter_c::LANGUAGE),
            C_QUERY,
            Some(MemberShape {
                member_kind: "field_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::ObjectiveC => (
            tree_sitter::Language::new(tree_sitter_objc::LANGUAGE),
            OBJC_QUERY,
            Some(MemberShape {
                member_kind: "field_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::Ruby => (
            tree_sitter::Language::new(tree_sitter_ruby::LANGUAGE),
            RUBY_QUERY,
            None,
        ),
        Language::Php => (
            tree_sitter::Language::new(tree_sitter_php::LANGUAGE_PHP),
            PHP_QUERY,
            None,
        ),
        Language::Kotlin => (
            tree_sitter::Language::new(tree_sitter_kotlin_ng::LANGUAGE),
            KOTLIN_QUERY,
            Some(MemberShape {
                member_kind: "navigation_expression",
                call_kind: "call_expression",
                callee_field: None,
            }),
        ),
        Language::Swift => (
            tree_sitter::Language::new(tree_sitter_swift::LANGUAGE),
            SWIFT_QUERY,
            Some(MemberShape {
                member_kind: "navigation_expression",
                call_kind: "call_expression",
                callee_field: None,
            }),
        ),
        Language::Dart => (
            tree_sitter::Language::new(tree_sitter_dart::LANGUAGE),
            DART_QUERY,
            Some(MemberShape {
                member_kind: "member_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        Language::Scala => (
            tree_sitter::Language::new(tree_sitter_scala::LANGUAGE),
            SCALA_QUERY,
            Some(MemberShape {
                member_kind: "field_expression",
                call_kind: "call_expression",
                callee_field: Some("function"),
            }),
        ),
        _ => return None,
    };
    Some((grammar, LangSpec { query, member }))
}

// ── Per-language queries ─────────────────────────────────────────────────────
//
// Written against each grammar's real node names, not from memory. A wrong name
// fails `Query::new` loudly rather than silently matching nothing, and every
// language below is pinned by a test.

/// **Go contributes no inheritance clauses, by design.** It has no `extends` /
/// `implements`: interface satisfaction is structural and implicit, and struct
/// embedding is composition, not implementation. No clause in the source asserts
/// "this type implements that interface", so emitting one would mean inferring
/// it — precisely the guess section 8.1 forbids. Go's implements relation stays
/// commit-gated, where the SCIP sidecar derives it from real type information.
const GO_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(selector_expression field: (field_identifier) @sel.name)
"#;

/// Java's grammar already separates `method_invocation` from `field_access`, so
/// no shape is needed. `method_invocation` covers both `helper()` and `o.m()`.
const JAVA_QUERY: &str = r#"
(method_invocation name: (identifier) @call.name)
(field_access field: (identifier) @field.name)
(superclass (type_identifier) @base.name)
(superclass (generic_type (type_identifier) @base.name))
(super_interfaces (type_list (type_identifier) @base.name))
(super_interfaces (type_list (generic_type (type_identifier) @base.name)))
(extends_interfaces (type_list (type_identifier) @base.name))
(extends_interfaces (type_list (generic_type (type_identifier) @base.name)))
"#;

/// A C# base list holds plain, generic, and qualified names, and the class-vs-
/// interface distinction is not written in the source at all — both are valid
/// `is-implementation` targets, so both are emitted and the target-kind gate in
/// the daemon decides.
///
/// Calls need the same generic treatment the base list already got. Per the
/// grammar's node-types, `invocation_expression.function` can be a `generic_name`
/// directly and `member_access_expression.name` can be `identifier` OR
/// `generic_name`, so a plain-`identifier`-only pattern silently never matches
/// `Bar<T>()` or `x.Foo<T>()`. The generic arms below capture the `identifier`
/// inside the `generic_name`, which is the bare method name the resolver wants;
/// it sits two parents under the `member_access_expression`, within
/// `MEMBER_LOOKUP_DEPTH`, so `x.Foo<T>()` still classifies as a call and not a
/// field read.
const CSHARP_QUERY: &str = r#"
(invocation_expression function: (identifier) @call.name)
(invocation_expression function: (generic_name (identifier) @call.name))
(member_access_expression name: (identifier) @sel.name)
(member_access_expression name: (generic_name (identifier) @sel.name))
(base_list (identifier) @base.name)
(base_list (generic_name (identifier) @base.name))
(base_list (qualified_name name: (identifier) @base.name))
"#;

/// `->` and `.` are both `field_expression` in C/C++, so one pattern covers a
/// pointer and a value receiver.
const CPP_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(call_expression function: (qualified_identifier name: (identifier) @call.name))
(field_expression field: (field_identifier) @sel.name)
(base_class_clause (type_identifier) @base.name)
"#;

/// C has no inheritance clause to detect.
const C_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(field_expression field: (field_identifier) @sel.name)
"#;

/// A keyword message (`[o setX:1 y:2]`) yields one `method:` capture per keyword.
/// Each resolves to the same selector's definition, so the extra targets produce
/// a duplicate edge rather than a wrong one, and the daemon's upsert absorbs it.
const OBJC_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(message_expression method: (identifier) @call.name)
(field_expression field: (field_identifier) @sel.name)
(class_interface superclass: (identifier) @base.name)
"#;

/// **Ruby has no field reads to detect**, and that is the language, not a gap:
/// `o.attr` is a method call on an attribute reader, which is what `call` here
/// captures. A bare receiverless `helper` is not detectable either — the grammar
/// parses it as a plain `identifier`, indistinguishable from a local variable or
/// a parameter, and emitting every identifier would flood the editor with
/// positions that resolve to locals. Both are recall costs with no precision
/// cost, which is the right side to err on.
///
/// `include M` is Ruby's nearest analogue to `implements`, and it is deliberately
/// **not** treated as one: it parses as an ordinary `call`, so recognising it
/// would mean special-casing a method name and asserting a relation the grammar
/// does not state. That is an inference, not a detection (section 8.1).
const RUBY_QUERY: &str = r#"
(call method: (identifier) @call.name)
(superclass (constant) @base.name)
(superclass (scope_resolution name: (constant) @base.name))
"#;

/// PHP separates every call shape from `member_access_expression`, so no shape is
/// needed. `new Thing()` is included: a constructor call is a reference to the
/// class, the same treatment Python's `Foo()` gets.
const PHP_QUERY: &str = r#"
(function_call_expression function: (name) @call.name)
(member_call_expression name: (name) @call.name)
(scoped_call_expression name: (name) @call.name)
(object_creation_expression (name) @call.name)
(member_access_expression name: (name) @field.name)
(base_clause (name) @base.name)
(class_interface_clause (name) @base.name)
"#;

/// tree-sitter-kotlin-ng labels neither the callee of a `call_expression` nor
/// the parts of a `navigation_expression`, so the callee is the first named
/// child and the accessed name is the second. `(_)` for the receiver keeps a
/// chained `a.b.c` matching at every level.
const KOTLIN_QUERY: &str = r#"
(call_expression (identifier) @call.name)
(navigation_expression (_) (identifier) @sel.name)
(delegation_specifier (user_type (identifier) @base.name))
(delegation_specifier (constructor_invocation (user_type (identifier) @base.name)))
"#;

/// Swift nests the accessed name one level deeper than the other grammars, in a
/// `navigation_suffix` inside the `navigation_expression`, which is why
/// [`is_call_callee`] walks up rather than taking the parent.
const SWIFT_QUERY: &str = r#"
(call_expression (simple_identifier) @call.name)
(navigation_suffix suffix: (simple_identifier) @sel.name)
(inheritance_specifier inherits_from: (user_type (type_identifier) @base.name))
"#;

/// Dart splits the three clause kinds into `superclass`, `mixins`, and
/// `interfaces`; all three name a type this class implements or derives from.
const DART_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(member_expression property: (identifier) @sel.name)
(superclass type: (type (type_identifier) @base.name))
(mixins (type (type_identifier) @base.name))
(interfaces (type (type_identifier) @base.name))
"#;

/// Scala folds `extends B with T with U` into one `extends_clause` holding a
/// `type:` child per base, so one pattern covers every half.
///
/// The base is matched **without** naming the `type:` field on purpose: a field
/// name binds a query to the *first* child carrying it, so `type: (…)` would
/// capture `B` and silently drop every `with` clause after it. Verified against
/// the grammar, not assumed.
const SCALA_QUERY: &str = r#"
(call_expression function: (identifier) @call.name)
(field_expression field: (identifier) @sel.name)
(extends_clause (type_identifier) @base.name)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn names(refs: &[LiveRef]) -> Vec<(u32, &str)> {
        refs.iter().map(|r| (r.line, r.name.as_str())).collect()
    }

    fn bases(refs: &[InheritanceRef]) -> Vec<&str> {
        refs.iter().map(|r| r.base_name.as_str()).collect()
    }

    /// Every language's query must compile against its real grammar. A wrong
    /// node name is a `Query::new` error, not a silent zero-match detector, and
    /// this is what keeps that true as grammars are upgraded.
    #[test]
    fn every_registered_query_compiles() {
        for lang in [
            Language::Go,
            Language::Java,
            Language::CSharp,
            Language::Cpp,
            Language::C,
            Language::ObjectiveC,
            Language::Ruby,
            Language::Php,
            Language::Kotlin,
            Language::Swift,
            Language::Dart,
            Language::Scala,
        ] {
            let (grammar, spec) = spec_for(lang).expect("a registered language has a spec");
            Query::new(&grammar, spec.query)
                .unwrap_or_else(|e| panic!("{} query must compile: {e}", lang.as_str()));
            // A `@sel.name` capture is meaningless without the shape that
            // classifies it, and the fallback would silently call every method
            // call a field read.
            assert_eq!(
                spec.query.contains("@sel.name"),
                spec.member.is_some(),
                "{} must declare a MemberShape if and only if it captures @sel.name",
                lang.as_str()
            );
        }
    }

    // ── Go ───────────────────────────────────────────────────────────────────

    #[test]
    fn go_separates_calls_from_field_reads() {
        let src = br#"
package main

func run(s *Session) int {
	s.Start()
	helper()
	n := s.count
	return n
}
"#;
        let refs = detect_live_refs(Language::Go, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(5, "Start"), (6, "helper")]);
        assert_eq!(names(&refs.fields), vec![(7, "count")]);
    }

    #[test]
    fn go_detects_package_qualified_calls() {
        // `pkg.Foo()` is the shape that carries most of Go's cross-file value:
        // the receiver is a package, not a value, and only the server knows
        // which package the alias binds to.
        let src = br#"
package main

import svc "example.com/m/service"

func run() {
	svc.Start()
}
"#;
        let refs = detect_live_refs(Language::Go, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(7, "Start")]);
        assert!(refs.fields.is_empty());
    }

    #[test]
    fn go_emits_no_inheritance_clauses() {
        // Interface satisfaction is implicit in Go and embedding is composition,
        // so there is no clause to detect and nothing may be inferred.
        let src = br#"
package main

type Reader interface { Read() }

type Base struct{ n int }

type Derived struct {
	Base
}
"#;
        let refs = detect_live_refs(Language::Go, src).unwrap();
        assert!(refs.inheritance.is_empty());
    }

    #[test]
    fn a_method_declaration_is_not_a_reference() {
        let src = br#"
package main

type T struct{}

func (t *T) Foo() {}
"#;
        let refs = detect_live_refs(Language::Go, src).unwrap();
        assert!(
            refs.is_empty(),
            "declarations are definitions, not references"
        );
    }

    // ── Java ─────────────────────────────────────────────────────────────────

    #[test]
    fn java_detects_calls_fields_and_both_clause_kinds() {
        let src = br#"class C extends Base<T> implements Runnable, Closeable {
  void f(Order o) {
    helper();
    o.submit(1);
    int n = o.total;
  }
}
"#;
        let refs = detect_live_refs(Language::Java, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "helper"), (4, "submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "total")]);
        assert_eq!(
            bases(&refs.inheritance),
            vec!["Base", "Runnable", "Closeable"]
        );
    }

    #[test]
    fn java_detects_an_interface_extends_clause() {
        let refs = detect_live_refs(Language::Java, b"interface I extends J, K {}").unwrap();
        assert_eq!(bases(&refs.inheritance), vec!["J", "K"]);
    }

    // ── C# ───────────────────────────────────────────────────────────────────

    #[test]
    fn csharp_separates_calls_from_field_reads() {
        let src = br#"class C : Base<T>, N.IRunnable {
  void F(Order o) {
    Helper();
    o.Submit(1);
    var n = o.Total;
  }
}
"#;
        let refs = detect_live_refs(Language::CSharp, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "Helper"), (4, "Submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "Total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "IRunnable"]);
    }

    /// `x.Foo<int>()` and `Bar<T>()` are ordinary calls that the plain-identifier
    /// patterns never saw, because the grammar wraps a generic callee in a
    /// `generic_name` node. The on-save lane therefore never asked about them.
    #[test]
    fn csharp_detects_generic_calls() {
        let refs = detect_live_refs(
            Language::CSharp,
            b"var a = s.Foo<int>(1); var b = Bar<T>(2);",
        )
        .unwrap();
        assert_eq!(names(&refs.calls), vec![(1, "Foo"), (1, "Bar")]);
        // A generic method call is a call, never a field read.
        assert_eq!(names(&refs.fields), vec![]);
    }

    // ── C / C++ ──────────────────────────────────────────────────────────────

    #[test]
    fn cpp_detects_qualified_calls_and_both_receiver_forms() {
        let src = br#"class C : public Base {
  void f(Order o, Order* p) {
    helper();
    ns::build();
    o.submit(1);
    p->submit(2);
    int n = o.total;
  }
};
"#;
        let refs = detect_live_refs(Language::Cpp, src).unwrap();
        assert_eq!(
            names(&refs.calls),
            vec![(3, "helper"), (4, "build"), (5, "submit"), (6, "submit")]
        );
        assert_eq!(names(&refs.fields), vec![(7, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base"]);
    }

    #[test]
    fn c_detects_calls_and_fields_and_has_no_inheritance() {
        let src = br#"void f(struct S s, struct S* p) {
  helper();
  int a = s.total;
  int b = p->total;
}
"#;
        let refs = detect_live_refs(Language::C, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(2, "helper")]);
        assert_eq!(names(&refs.fields), vec![(3, "total"), (4, "total")]);
        assert!(refs.inheritance.is_empty());
    }

    // ── Objective-C ──────────────────────────────────────────────────────────

    #[test]
    fn objc_detects_message_sends_and_the_superclass() {
        let src = br#"@interface C : Base
@end
void f(Order* o, struct S s) {
  helper();
  [o submit:1];
  int n = s.total;
}
"#;
        let refs = detect_live_refs(Language::ObjectiveC, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(4, "helper"), (5, "submit")]);
        assert_eq!(names(&refs.fields), vec![(6, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base"]);
    }

    // ── Ruby ─────────────────────────────────────────────────────────────────

    #[test]
    fn ruby_treats_attribute_reads_as_the_method_calls_they_are() {
        let src = br#"class C < A::Base
  def f(o)
    o.submit(1)
    o.total
  end
end
"#;
        let refs = detect_live_refs(Language::Ruby, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "submit"), (4, "total")]);
        assert!(
            refs.fields.is_empty(),
            "Ruby has no field-read expression; an attribute read is a call"
        );
        assert_eq!(bases(&refs.inheritance), vec!["Base"]);
    }

    #[test]
    fn ruby_does_not_infer_implements_from_include() {
        // `include M` parses as an ordinary call, so recognising it would mean
        // special-casing a method name to assert a relation the grammar does not
        // state. That is an inference, not a detection (section 8.1).
        let refs = detect_live_refs(Language::Ruby, b"class C\n  include M\nend\n").unwrap();
        assert!(refs.inheritance.is_empty());
        assert_eq!(names(&refs.calls), vec![(2, "include")]);
    }

    // ── PHP ──────────────────────────────────────────────────────────────────

    #[test]
    fn php_detects_every_call_shape_and_both_clause_kinds() {
        let src = br#"<?php
class C extends Base implements Runnable {
  function f($o) {
    helper();
    $o->submit(1);
    Factory::build();
    $x = new Thing();
    return $o->total;
  }
}
"#;
        let refs = detect_live_refs(Language::Php, src).unwrap();
        assert_eq!(
            names(&refs.calls),
            vec![(4, "helper"), (5, "submit"), (6, "build"), (7, "Thing"),]
        );
        assert_eq!(names(&refs.fields), vec![(8, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "Runnable"]);
    }

    // ── Kotlin ───────────────────────────────────────────────────────────────

    #[test]
    fn kotlin_separates_calls_from_property_reads() {
        let src = br#"class C : Base(), Runnable {
    fun f(o: Order) {
        helper()
        o.submit(1)
        val n = o.total
    }
}
"#;
        let refs = detect_live_refs(Language::Kotlin, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "helper"), (4, "submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "Runnable"]);
    }

    // ── Swift ────────────────────────────────────────────────────────────────

    #[test]
    fn swift_separates_calls_from_property_reads() {
        let src = br#"class C: Base, Runnable {
    func f(o: Order) {
        helper()
        o.submit(1)
        let n = o.total
    }
}
"#;
        let refs = detect_live_refs(Language::Swift, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "helper"), (4, "submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "Runnable"]);
    }

    // ── Dart ─────────────────────────────────────────────────────────────────

    /// A clause that can name several types must yield all of them. A
    /// tree-sitter field name binds to the *first* child carrying it, so a
    /// pattern written as `type: (…)` over a repeated field silently keeps only
    /// the first base. This pins that none of the multi-base clauses do that.
    #[test]
    fn a_clause_naming_several_types_yields_all_of_them() {
        let dart = detect_live_refs(
            Language::Dart,
            b"class C extends B with M1, M2 implements I1, I2 {}",
        )
        .unwrap();
        assert_eq!(bases(&dart.inheritance), vec!["B", "M1", "M2", "I1", "I2"]);

        let scala =
            detect_live_refs(Language::Scala, b"class C extends B with R with O { }").unwrap();
        assert_eq!(bases(&scala.inheritance), vec!["B", "R", "O"]);

        let php = detect_live_refs(
            Language::Php,
            b"<?php class C extends B implements I1, I2 {}",
        )
        .unwrap();
        assert_eq!(bases(&php.inheritance), vec!["B", "I1", "I2"]);

        let csharp = detect_live_refs(Language::CSharp, b"class C : B, I1, I2 { }").unwrap();
        assert_eq!(bases(&csharp.inheritance), vec!["B", "I1", "I2"]);

        let kotlin = detect_live_refs(Language::Kotlin, b"class C : B(), I1, I2 { }").unwrap();
        assert_eq!(bases(&kotlin.inheritance), vec!["B", "I1", "I2"]);

        let swift = detect_live_refs(Language::Swift, b"class C: B, P1, P2 { }").unwrap();
        assert_eq!(bases(&swift.inheritance), vec!["B", "P1", "P2"]);

        let java =
            detect_live_refs(Language::Java, b"class C extends B implements I1, I2 {}").unwrap();
        assert_eq!(bases(&java.inheritance), vec!["B", "I1", "I2"]);
    }

    #[test]
    fn dart_detects_all_three_clause_kinds() {
        let src = br#"class C extends Base with Mixin implements Runnable {
  void f(Order o) {
    helper();
    o.submit(1);
    var n = o.total;
  }
}
"#;
        let refs = detect_live_refs(Language::Dart, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "helper"), (4, "submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "Mixin", "Runnable"]);
    }

    // ── Scala ────────────────────────────────────────────────────────────────

    #[test]
    fn scala_folds_extends_and_with_into_one_clause() {
        let src = br#"class C extends Base with Runnable {
  def f(o: Order) = {
    helper()
    o.submit(1)
    val n = o.total
  }
}
"#;
        let refs = detect_live_refs(Language::Scala, src).unwrap();
        assert_eq!(names(&refs.calls), vec![(3, "helper"), (4, "submit")]);
        assert_eq!(names(&refs.fields), vec![(5, "total")]);
        assert_eq!(bases(&refs.inheritance), vec!["Base", "Runnable"]);
    }

    // ── Shared behaviour ─────────────────────────────────────────────────────

    #[test]
    fn a_language_without_a_detector_returns_empty() {
        // The data/config and prose formats carry no call semantics at all.
        for lang in [Language::Json, Language::Yaml, Language::Markdown] {
            assert!(detect_live_refs(lang, b"{}").unwrap().is_empty());
        }
    }

    #[test]
    fn a_native_language_has_no_generic_detector() {
        // Rust, TypeScript and Python are served by their native Phase B
        // extractor, which carries receiver types and signature keys the generic
        // path deliberately drops. Two detectors for one language would disagree.
        for lang in [Language::Rust, Language::TypeScript, Language::Python] {
            assert!(spec_for(lang).is_none());
        }
    }

    #[test]
    fn an_oversized_buffer_is_skipped() {
        let src = vec![b' '; MAX_SOURCE_BYTES + 1];
        assert!(detect_live_refs(Language::Go, &src).unwrap().is_empty());
    }
}
