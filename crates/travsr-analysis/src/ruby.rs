//! Phase A parser for Ruby source files using tree-sitter.

use std::path::Path;

use travsr_core::{Edge, EdgeKind, Language, Node, VName};

use crate::generic::{enclosing_container, parse_with_config, LanguageConfig, PostParseCtx};
use crate::ParseOutput;

pub const CONFIG: LanguageConfig = LanguageConfig {
    language: Language::Ruby,
    extensions: &["rb", "rake", "gemspec"],
    queries: r#"
(class name: (constant) @class.name)
(module name: (constant) @module.name)
(method name: (identifier) @fn.name)
(singleton_method name: (identifier) @fn.name)
; #780: setter (`def x=`) and operator (`def ==`, `def <=>`, `def []`,
; `def []=`, `def <<`, …) methods name their def with a `setter`/`operator`
; node, not an `identifier`, so the identifier pattern above never captured
; them and scip-ruby's twins (`Class#`x=`().`, `Class#`==`().`) orphaned.
(method name: (setter) @fn.name)
(method name: (operator) @fn.name)
(singleton_method name: (setter) @fn.name)
(singleton_method name: (operator) @fn.name)
; #780 (RC-3): constants (`X = …`) and instance variables (`@x = …`) as def
; nodes so scip-ruby const/ivar references unify onto them instead of resolving
; to duplicate orphan scip nodes. The ivar is field-qualified by its enclosing
; type (`field:C.@x`) so `@name` in different classes never collide.
(assignment left: (constant) @const.name)
(assignment left: (instance_variable) @field.name)
(call
  method: (identifier) @require_relative.kw
  arguments: (argument_list (string (string_content) @import))
  (#eq? @require_relative.kw "require_relative"))
(call
  method: (identifier) @require.kw
  arguments: (argument_list (string (string_content) @import.gem))
  (#eq? @require.kw "require"))
(class
  superclass: (superclass (scope_resolution name: (constant) @_rs))
  (#any-of? @_rs "Test" "TestCase")) @test.scope
(class
  superclass: (superclass (scope_resolution name: (constant) @_rs2))
  body: (body_statement (method name: (identifier) @test.entry))
  (#any-of? @_rs2 "Test" "TestCase")
  (#match? @test.entry "^test"))
"#,
    capture_kinds: &[
        ("class.name", "class", "class"),
        ("module.name", "class", "class"),
        ("fn.name", "function", "fn"),
        // N4b/#614: `require_relative` and `require` are split into two
        // patterns/captures so the emitted signature records which keyword
        // produced the import. `require_relative` is importer-relative
        // (`import:<spec>`); `require` is load-path (gem/stdlib/in-repo lib),
        // tagged `import:gem:<spec>` so RubyResolver can tell them apart
        // (#614) instead of both collapsing to `import:<spec>`. Both keep
        // node kind `"import"`, so `EdgeKind::Depends` is unchanged. The
        // `#eq?` predicates are auto-applied by tree-sitter's match iterator,
        // so `@require.kw`/`@require_relative.kw` (no capture_kinds entry)
        // are ignored and any other call (e.g. `puts`) never matches either
        // pattern.
        ("import", "import", "import"),
        ("import.gem", "import", "import:gem"),
        // #780 (RC-3): `X = …` → `const:X` (file-level; the VName's path keeps
        // same-named constants in different files distinct). Matches scip-ruby's
        // `const:X` unification candidate.
        //
        // Deliberately file-level, not container-qualified like `field` below:
        // Ruby constants are overwhelmingly referenced by their bare name and
        // scip-ruby's `variable` candidate ladder falls back to unqualified
        // `const:X` regardless, so a qualified `const:C.X` node would simply not
        // be found. Accepted limitation: two same-named constants in different
        // classes of ONE file (`class B; NAME=…; end; class C; NAME=…; end`)
        // collapse onto a single `const:NAME` node and references to either
        // resolve to it. This is rare, and qualifying would require routing
        // `const` through the shared `member_qual` path in `generic.rs`, which
        // governs every language's `const` capture — out of scope for #780.
        ("const.name", "constant", "const"),
        // #780 (RC-3): `@x = …` → `field:C.@x`, qualified by the enclosing type
        // via the shared `field` member-qualification path, matching scip-ruby's
        // `field:C.@x` candidate. `@name` in different classes stay distinct.
        ("field.name", "field", "field"),
    ],
    method_containers: &[("class", "class"), ("module", "class")],
    decl_kinds: &[],
    type_refinements: &[],
    post_parse: Some(ruby_post_parse),
    name_hook: None,
    get_grammar: || tree_sitter::Language::new(tree_sitter_ruby::LANGUAGE),
};

/// #780 Prong B: Ruby-specific node synthesis the shared capture pipeline cannot
/// express — `attr_*` macro accessors and `Struct.new` classes with their member
/// accessors and block methods.
fn ruby_post_parse(ctx: &PostParseCtx<'_>, nodes: &mut Vec<Node>, edges: &mut Vec<Edge>) {
    expand_attr_accessors(ctx, nodes, edges);
    expand_struct_defs(ctx, nodes, edges);
}

/// The declared name of a `method` / `singleton_method` def as tree-sitter names
/// it: an `identifier` (`foo`), a `setter` (`foo=`), or an `operator` (`==`,
/// `[]`, `<=>`, …). Returns the node's own text, which is exactly the method name
/// scip-ruby uses in its descriptor.
fn method_def_name<'a>(method: tree_sitter::Node<'_>, source: &'a [u8]) -> Option<&'a str> {
    let name = method.child_by_field_name("name")?;
    match name.kind() {
        "identifier" | "setter" | "operator" => name.utf8_text(source).ok().map(str::trim),
        _ => None,
    }
}

/// #780 Prong B: expand `attr_accessor` / `attr_reader` / `attr_writer` macros
/// into the accessor method nodes tree-sitter never emits, so scip-ruby's
/// synthesized reader/writer defs (`Class#name().`, `` Class#`name=`(). ``)
/// unify onto a real Phase A twin instead of surviving as orphan duplicates
/// that steal the accessor's reference edges.
///
/// One macro call yields up to `2 × N` names (a reader `x` and a writer `x=`
/// per symbol), with the writer's `=`-suffixed name derived — which no single
/// tree-sitter capture can express, hence this post-parse pass. Each accessor
/// is anchored on the macro's line (scip-ruby reports the accessor there too,
/// so unification matches with line delta 0) and qualified by the enclosing
/// class/module (matching scip-ruby's container). An accessor whose signature
/// was already emitted by the capture pass — a class with both `attr_accessor
/// :x` and an explicit `def x` / `def x=` — is not re-emitted.
fn expand_attr_accessors(ctx: &PostParseCtx<'_>, nodes: &mut Vec<Node>, edges: &mut Vec<Edge>) {
    // Signatures already present (explicit `def x` / `def x=`, and accessors
    // emitted earlier in this pass) — the double-emission guard.
    let mut seen: std::collections::HashSet<String> =
        nodes.iter().map(|n| n.vname.signature.clone()).collect();

    let mut stack = vec![ctx.root];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
        if node.kind() != "call" {
            continue;
        }
        let Some(method) = node.child_by_field_name("method") else {
            continue;
        };
        if method.kind() != "identifier" {
            continue;
        }
        let (reader, writer) = match method.utf8_text(ctx.source).unwrap_or("") {
            "attr_accessor" => (true, true),
            "attr_reader" => (true, false),
            "attr_writer" => (false, true),
            _ => continue,
        };
        // An accessor macro only defines methods inside a class/module; the
        // enclosing type both qualifies the name (matching scip-ruby's
        // container) and prevents `method:name` collisions across classes.
        let Some((prefix, container)) = enclosing_container(
            node,
            ctx.config.method_containers,
            ctx.config.type_refinements,
            ctx.source,
        ) else {
            continue;
        };
        let Some(args) = node.child_by_field_name("arguments") else {
            continue;
        };
        let line = node.start_position().row as u32 + 1;
        let container_id = VName::new(
            ctx.corpus,
            "",
            ctx.vname_path,
            ctx.lang,
            format!("{prefix}:{container}"),
        )
        .id();

        let mut arg_cursor = args.walk();
        for arg in args.children(&mut arg_cursor) {
            if arg.kind() != "simple_symbol" {
                continue;
            }
            // `:package_name` → `package_name`.
            let Some(name) = arg
                .utf8_text(ctx.source)
                .ok()
                .and_then(|t| t.strip_prefix(':'))
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let mut method_names: Vec<String> = Vec::with_capacity(2);
            if reader {
                method_names.push(name.to_string());
            }
            if writer {
                method_names.push(format!("{name}="));
            }
            for mname in method_names {
                let sig = format!("method:{container}.{mname}");
                if !seen.insert(sig.clone()) {
                    continue;
                }
                let vname = VName::new(ctx.corpus, "", ctx.vname_path, ctx.lang, &sig);
                let accessor = Node::new(vname, "method")
                    .with_line(line)
                    .with_end_line(line);
                edges.push(Edge::new(
                    container_id,
                    accessor.id,
                    EdgeKind::DefinesBinding,
                ));
                nodes.push(accessor);
            }
        }
    }
}

/// #780 Prong B (Struct.new): a `Name = Struct.new(:a, :b) do … end` constant
/// is a class in scip-ruby (`…#Name#`) with a read/write accessor per member and
/// the block's own methods — none of which tree-sitter emits, because it sees a
/// constant assigned a method call, not a class. Synthesize the pieces scip
/// unifies against: the `class:Name` node, `method:Name.member` /
/// `method:Name.member=` accessors, and `method:Name.<block method>` for each
/// method defined in the `do … end` block. Anchored on the assignment line
/// (where scip reports the class and its synthesized accessors).
fn expand_struct_defs(ctx: &PostParseCtx<'_>, nodes: &mut Vec<Node>, edges: &mut Vec<Edge>) {
    let mut seen: std::collections::HashSet<String> =
        nodes.iter().map(|n| n.vname.signature.clone()).collect();

    let mut stack = vec![ctx.root];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
        if node.kind() != "assignment" {
            continue;
        }
        // `Name = <right>` where the constant `Name` becomes the class name.
        let Some(left) = node.child_by_field_name("left") else {
            continue;
        };
        if left.kind() != "constant" {
            continue;
        }
        let Some(name) = left.utf8_text(ctx.source).ok().map(str::trim) else {
            continue;
        };
        // `<right>` must be `Struct.new(...)` — receiver constant `Struct`,
        // method `new`.
        let Some(call) = node
            .child_by_field_name("right")
            .filter(|r| r.kind() == "call")
        else {
            continue;
        };
        let is_struct_new = call
            .child_by_field_name("receiver")
            .and_then(|r| r.utf8_text(ctx.source).ok())
            .is_some_and(|t| t.trim() == "Struct")
            && call
                .child_by_field_name("method")
                .and_then(|m| m.utf8_text(ctx.source).ok())
                .is_some_and(|t| t.trim() == "new");
        if !is_struct_new {
            continue;
        }

        // `Name = Struct.new(...)` is an `assignment`, so the shared constant
        // capture already emitted a `const:Name` node at this line. scip-ruby
        // models the construct as a class (`…#Name#`), which is the node the
        // members and refs unify against, so the constant is a duplicate
        // identity for the same location — exactly what this PR removes
        // elsewhere. Retract it (node plus its file edge) before emitting the
        // class so `travsr graph Name` is unambiguous.
        let const_id = VName::new(
            ctx.corpus,
            "",
            ctx.vname_path,
            ctx.lang,
            format!("const:{name}"),
        )
        .id();
        nodes.retain(|n| n.id != const_id);
        edges.retain(|e| !(e.src == ctx.file_id && e.dst == const_id));

        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let class_sig = format!("class:{name}");
        let class_id = VName::new(ctx.corpus, "", ctx.vname_path, ctx.lang, &class_sig).id();

        // The `class:Name` node (parented to the file, as tree-sitter's own class
        // nodes are). Skip if a real `class Name` already exists.
        if seen.insert(class_sig.clone()) {
            let vname = VName::new(ctx.corpus, "", ctx.vname_path, ctx.lang, &class_sig);
            let class_node = Node::new(vname, "class")
                .with_line(line)
                .with_end_line(end_line);
            edges.push(Edge::new(
                ctx.file_id,
                class_node.id,
                EdgeKind::DefinesBinding,
            ));
            nodes.push(class_node);
        }

        let mut emit_method =
            |mname: &str, at: u32, seen: &mut std::collections::HashSet<String>| {
                let sig = format!("method:{name}.{mname}");
                if !seen.insert(sig.clone()) {
                    return;
                }
                let vname = VName::new(ctx.corpus, "", ctx.vname_path, ctx.lang, &sig);
                let m = Node::new(vname, "method").with_line(at).with_end_line(at);
                edges.push(Edge::new(class_id, m.id, EdgeKind::DefinesBinding));
                nodes.push(m);
            };

        // Member accessors: one reader + writer per positional `:symbol` arg
        // (`keyword_init:` and other pairs are not simple_symbols and are
        // skipped). scip-ruby synthesizes these at the class line, so anchor them
        // on the assignment line.
        if let Some(args) = call.child_by_field_name("arguments") {
            let mut ac = args.walk();
            for arg in args.children(&mut ac) {
                if arg.kind() != "simple_symbol" {
                    continue;
                }
                if let Some(member) = arg
                    .utf8_text(ctx.source)
                    .ok()
                    .and_then(|t| t.strip_prefix(':'))
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    emit_method(member, line, &mut seen);
                    emit_method(&format!("{member}="), line, &mut seen);
                }
            }
        }

        // Methods defined in the `do … end` block belong to the Struct class and
        // are reported by scip-ruby at their own def line (which can be far from
        // the assignment line in a large block), so anchor each on its own line.
        // Emitted before the implicit `initialize` below so an explicit block
        // `def initialize` keeps its own accurate line.
        if let Some(block) = call.child_by_field_name("block") {
            for (mname, at) in block_method_names(block, ctx.source) {
                emit_method(&mname, at, &mut seen);
            }
        }

        // Every Struct has an implicit `initialize` constructor, which scip-ruby
        // emits as a def even when there is no block. Anchor it on the class line
        // (unless an explicit block `def initialize` already claimed it above).
        emit_method("initialize", line, &mut seen);
    }
}

/// `(name, 1-based line)` of `method`/`singleton_method` defs directly inside a
/// Struct.new block, not descending into a nested class/module (whose methods
/// belong to it, not the Struct). Order is irrelevant — the caller dedups by
/// signature.
fn block_method_names(block: tree_sitter::Node<'_>, source: &[u8]) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let mut stack = vec![block];
    while let Some(node) = stack.pop() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            // A nested class/module owns its own methods — do not attribute them
            // to the enclosing Struct.
            if child != block && matches!(child.kind(), "class" | "module") {
                continue;
            }
            if matches!(child.kind(), "method" | "singleton_method") {
                if let Some(n) = method_def_name(child, source) {
                    out.push((n.to_string(), child.start_position().row as u32 + 1));
                }
            }
            stack.push(child);
        }
    }
    out
}

/// Parse a Ruby source file into graph nodes and edges.
pub fn parse(corpus: &str, abs_path: &Path, vname_path: &str) -> anyhow::Result<ParseOutput> {
    let grammar = (CONFIG.get_grammar)();
    parse_with_config(&CONFIG, &grammar, None, corpus, abs_path, vname_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n4b_require_imports_and_depends_edges() {
        // N4b/#614: `require` and `require_relative` produce distinct import
        // signatures, file->import `Depends` edges (Ruby previously emitted
        // zero, #582), and both stay `kind == "import"`. A non-require call
        // (`puts`) must NOT produce an import, the `#eq?` predicates filter
        // the match on each pattern.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("boot.rb");
        std::fs::write(
            &path,
            "require 'set'\nrequire_relative './foo'\nrequire_relative 'bar'\nputs 'hi'\n",
        )
        .unwrap();
        let out = parse("corp", &path, "boot.rb").unwrap();

        let imports: Vec<&travsr_core::Node> =
            out.nodes.iter().filter(|n| n.kind == "import").collect();
        let import_sigs: Vec<&str> = imports.iter().map(|n| n.vname.signature.as_str()).collect();
        assert!(
            import_sigs.contains(&"import:gem:set"),
            "got {import_sigs:?}"
        );
        assert!(import_sigs.contains(&"import:./foo"), "got {import_sigs:?}");
        assert!(import_sigs.contains(&"import:bar"), "got {import_sigs:?}");
        assert_eq!(
            import_sigs.len(),
            3,
            "no import for `puts`: {import_sigs:?}"
        );
        assert!(
            imports.iter().all(|n| n.kind == "import"),
            "both require forms stay kind=import: {import_sigs:?}"
        );

        let depends = out
            .edges
            .iter()
            .filter(|e| e.kind == travsr_core::EdgeKind::Depends)
            .count();
        assert_eq!(depends, 3, "one Depends edge per require");
    }

    #[test]
    fn gem_import_has_no_end_line() {
        // Mutation-test guard: generic.rs's `is_import_prefix` helper governs
        // both the signature-cleanup match arm AND the end_line guard for
        // *every* import-scheme prefix, not just the bare "import" one. The
        // n4b test above only checks emitted signatures, so a regression
        // that scopes the end_line guard back to `sig_prefix != "import"`
        // (leaving `import:gem:*` nodes to fall through and pick up an
        // end_line) would go undetected. Import nodes are single-line
        // requires, not spans, so `import:gem:*` must stay end_line-free
        // exactly like the bare `import:*` (require_relative) form.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("boot.rb");
        std::fs::write(&path, "require 'set'\nrequire_relative './foo'\n").unwrap();
        let out = parse("corp", &path, "boot.rb").unwrap();
        let imports: Vec<&travsr_core::Node> =
            out.nodes.iter().filter(|n| n.kind == "import").collect();
        assert_eq!(imports.len(), 2, "got {imports:?}");
        for n in &imports {
            assert_eq!(
                n.end_line, None,
                "import node {:?} must not carry an end_line",
                n.vname.signature
            );
        }
    }

    #[test]
    fn parse_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.rb");
        std::fs::write(&path, "").unwrap();
        let out = parse("corp", &path, "empty.rb").unwrap();
        assert_eq!(out.nodes.len(), 1);
    }

    fn sigs_of(out: &ParseOutput) -> Vec<&str> {
        out.nodes
            .iter()
            .map(|n| n.vname.signature.as_str())
            .collect()
    }

    #[test]
    fn attr_accessor_emits_reader_and_writer() {
        // #780 Prong B: `attr_accessor :x` synthesizes both `method:C.x`
        // (reader) and `method:C.x=` (writer), qualified by the class, anchored
        // on the macro line so scip-ruby's `x()`/`x=()` twins unify (delta 0).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  attr_accessor :package_name, :version_code\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let sigs = sigs_of(&out);
        for want in [
            "method:C.package_name",
            "method:C.package_name=",
            "method:C.version_code",
            "method:C.version_code=",
        ] {
            assert!(sigs.contains(&want), "missing {want}: {sigs:?}");
        }
        // Both accessors anchor on the macro line (2), so line-proximity holds.
        let reader = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "method:C.package_name")
            .unwrap();
        assert_eq!(reader.line, Some(2));
        assert_eq!(reader.kind, "method");
        // Containment edge is parented to the class, not the file.
        let class_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "class:C")
            .unwrap()
            .id;
        assert!(
            out.edges
                .iter()
                .any(|e| e.src == class_id && e.dst == reader.id),
            "reader must be contained by class:C"
        );
    }

    #[test]
    fn attr_reader_and_writer_are_one_sided() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  attr_reader :ro\n  attr_writer :wo\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let sigs = sigs_of(&out);
        assert!(sigs.contains(&"method:C.ro"), "reader: {sigs:?}");
        assert!(!sigs.contains(&"method:C.ro="), "no writer for attr_reader");
        assert!(sigs.contains(&"method:C.wo="), "writer: {sigs:?}");
        assert!(!sigs.contains(&"method:C.wo"), "no reader for attr_writer");
    }

    #[test]
    fn attr_accessor_does_not_double_emit_explicit_def() {
        // #780 guard: a class with both `attr_accessor :x` and an explicit
        // `def x` / `def x=` yields exactly one node per signature.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  attr_accessor :x\n  def x\n    @x\n  end\n  def x=(v)\n    @x = v\n  end\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        assert_eq!(
            out.nodes
                .iter()
                .filter(|n| n.vname.signature == "method:C.x")
                .count(),
            1,
            "method:C.x must not be double-emitted"
        );
        assert_eq!(
            out.nodes
                .iter()
                .filter(|n| n.vname.signature == "method:C.x=")
                .count(),
            1,
            "method:C.x= must not be double-emitted"
        );
    }

    #[test]
    fn setter_and_operator_methods_captured() {
        // #780 Prong B: setter (`def x=`) and operator (`def ==`, `def <=>`,
        // `def []`, `def []=`) methods are captured as qualified method nodes.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  def name=(v); end\n  def ==(o); end\n  def <=>(o); end\n  def [](i); end\n  def []=(i, v); end\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let sigs = sigs_of(&out);
        for want in [
            "method:C.name=",
            "method:C.==",
            "method:C.<=>",
            "method:C.[]",
            "method:C.[]=",
        ] {
            assert!(sigs.contains(&want), "missing {want}: {sigs:?}");
        }
    }

    #[test]
    fn struct_new_emits_class_members_and_block_methods() {
        // #780 Prong B (Struct.new): `X = Struct.new(:a, :b) do def m; end end` is
        // a class in scip-ruby with member accessors and block methods. Emit
        // class:X, method:X.a/.a=/.b/.b=, and method:X.m.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "module M\n  ServiceOption = Struct.new(:auth_type, :name) do\n    def describe\n    end\n  end\n  URLLog = Struct.new(:url)\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let sigs = sigs_of(&out);
        for want in [
            "class:ServiceOption",
            "method:ServiceOption.auth_type",
            "method:ServiceOption.auth_type=",
            "method:ServiceOption.name",
            "method:ServiceOption.name=",
            "method:ServiceOption.describe",
            // Every Struct has an implicit `initialize` scip-ruby emits, even the
            // block-less `URLLog`.
            "method:ServiceOption.initialize",
            "class:URLLog",
            "method:URLLog.url",
            "method:URLLog.url=",
            "method:URLLog.initialize",
        ] {
            assert!(sigs.contains(&want), "missing {want}: {sigs:?}");
        }
        // Defect 2: `Name = Struct.new(...)` is an `assignment`, so the constant
        // capture emitted `const:ServiceOption` / `const:URLLog` too. The class is
        // the node scip unifies against, so the constant must be retracted — one
        // definition, one node — leaving no duplicate identity for `graph Name`.
        for gone in ["const:ServiceOption", "const:URLLog"] {
            assert!(
                !sigs.contains(&gone),
                "Struct.new constant must be retracted, found {gone}: {sigs:?}"
            );
        }
        // class:ServiceOption is parented to the file (as tree-sitter class nodes
        // are), so scip's `…#ServiceOption#` class ref unifies onto it.
        let file_id = out.nodes.iter().find(|n| n.kind == "file").unwrap().id;
        let class_id = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "class:ServiceOption")
            .unwrap()
            .id;
        assert!(out
            .edges
            .iter()
            .any(|e| e.src == file_id && e.dst == class_id));
        // The retracted constant's file edge is gone too (no orphan edge).
        let const_id =
            travsr_core::VName::new("corp", "", "a.rb", "ruby", "const:ServiceOption").id();
        assert!(
            !out.edges.iter().any(|e| e.dst == const_id),
            "retracted constant's edge must be removed"
        );
    }

    #[test]
    fn constants_and_ivars_emitted_as_def_nodes() {
        // #780 Prong C: `X = …` → `const:X`; `@x = …` → `field:C.@x` (qualified
        // by the enclosing class), so scip-ruby const/ivar refs unify instead of
        // orphaning duplicates.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  VERSION = \"1.0\"\n  def initialize\n    @count = 0\n  end\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let sigs = sigs_of(&out);
        assert!(sigs.contains(&"const:VERSION"), "const: {sigs:?}");
        assert!(sigs.contains(&"field:C.@count"), "ivar: {sigs:?}");
        let ivar = out
            .nodes
            .iter()
            .find(|n| n.vname.signature == "field:C.@count")
            .unwrap();
        assert_eq!(ivar.kind, "field");
    }

    #[test]
    fn reassigned_ivar_is_one_node_anchored_on_first_write() {
        // Defect 3: an ivar assigned in several methods matches the capture once
        // per assignment. All share one VName/NodeId, so they must collapse to a
        // single node anchored on the FIRST write (the definition) — not the last,
        // which `flush_staging_to_production`'s MAX(line) tie-break would pick and
        // which would push the node past the scip def's +/-5 line window.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rb");
        std::fs::write(
            &path,
            "class C\n  def initialize\n    @count = 0\n  end\n  def reset\n    @count = 0\n  end\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "a.rb").unwrap();
        let ivars: Vec<&travsr_core::Node> = out
            .nodes
            .iter()
            .filter(|n| n.vname.signature == "field:C.@count")
            .collect();
        assert_eq!(ivars.len(), 1, "one node per VName, not one per assignment");
        assert_eq!(
            ivars[0].line,
            Some(3),
            "anchored on the first write, not line 6"
        );
    }

    #[test]
    fn parse_class_and_methods() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.rb");
        std::fs::write(
            &path,
            "module M\nclass Foo\ndef bar; end\ndef self.baz; end\nend\nend\n",
        )
        .unwrap();
        let out = parse("corp", &path, "sample.rb").unwrap();
        let kinds: Vec<&str> = out.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert!(kinds.contains(&"class"));
        // N1: both methods are qualified by `Foo`. `def self.baz` no longer
        // collides with a same-named instance method (the ruby.rs defect).
        assert_eq!(kinds.iter().filter(|&&k| k == "method").count(), 2);
        let sigs: Vec<&str> = out
            .nodes
            .iter()
            .map(|n| n.vname.signature.as_str())
            .collect();
        assert!(sigs.contains(&"method:Foo.bar"), "got {sigs:?}");
        assert!(sigs.contains(&"method:Foo.baz"), "got {sigs:?}");
    }
}
