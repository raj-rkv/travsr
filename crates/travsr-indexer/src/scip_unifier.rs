//! G1 — SCIP symbol name/kind extraction helpers for the unification pass.
//!
//! This module is **pure** (no I/O, no DB).  The actual DB writes live in
//! `travsr-daemon::scip_unifier` which bridges indexer output with the store.
//!
//! RFC-014 §G1. The descriptor suffix grammar is defined by the SCIP spec and
//! shared by every SCIP indexer (scip-go, scip-typescript, scip-python,
//! rust-analyzer, scip-java, scip-clang, ...), so one parser covers all
//! languages:
//!   `(<disambiguator>).` → method/function    (kind `"function"`)
//!   `#`                  → type/class/trait   (kind `"class"`)
//!   `.`                  → field/variable     (kind `"variable"`)

/// Parsed identity of a SCIP symbol's leaf descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScipName<'a> {
    /// Enclosing type for methods/fields (`Server#Serve().` → `Server`).
    pub container: Option<&'a str>,
    /// Bare identifier (`Serve`).
    pub name: &'a str,
    /// RFC-014 kind class: `"function"`, `"class"`, or `"variable"`.
    pub kind: &'a str,
}

/// Extract `(container, name, kind)` from any SCIP symbol string.
///
/// SCIP symbol format: `<scheme> <mgr> <pkg> <version> <descriptor-chain>`
/// where the last `/`-separated descriptor encodes the identifier + a suffix.
/// Method descriptors may carry an overload disambiguator between the parens
/// (`add(+1).`, scip-clang hash disambiguators) — accepted and stripped.
///
/// Returns `None` for non-SCIP signatures (no descriptor suffix), package
/// descriptors, parameter descriptors, and macro/meta descriptors — callers
/// can feed every Phase B node through without pre-filtering by language.
pub fn scip_name_kind(symbol: &str) -> Option<ScipName<'_>> {
    // SCIP symbol = `<scheme> <manager> <package> <version> <descriptor-chain>`.
    // The four metadata fields are single-space separated and space-free for
    // every indexer we ingest, but the descriptor chain that follows MAY contain
    // spaces when a segment is backtick-escaped — scip-ruby wraps Sorbet's RSpec
    // DSL scopes that way (`` `<describe 'Foo'>`#`<it 'does x'>`(). ``).
    // `split_whitespace().last()` tears those apart into garbage that still gets
    // counted as a def miss; take everything after the 4th space instead so the
    // whole chain (spaces and all) is parsed as one unit.
    let descriptor_chain = symbol.splitn(5, ' ').nth(4)?;
    descriptor_chain_kind(descriptor_chain)
}

/// Extract `(container, name, kind)` from a SemanticDB symbol.
///
/// SemanticDB symbols are the descriptor chain alone, with no
/// `<scheme> <mgr> <pkg> <version>` metadata prefix and so no spaces:
/// `scala/util/parsing/combinator/Parsers#phrase().`. The descriptor grammar
/// itself is the same one SCIP uses, so only the missing prefix separates them
/// and [`scip_name_kind`]'s `splitn(5, ' ').nth(4)` rejects every one of them.
/// Without this the scala sidecar's defs never became unification candidates:
/// its `sdb:` nodes kept their own identity, ref edges pointed at those while
/// `references <name>` resolved to the Phase A twin, and every scala query
/// answered 0 against a fully populated graph. They were not even counted as
/// misses, so `travsr status` reported nothing wrong.
pub fn semanticdb_name_kind(symbol: &str) -> Option<ScipName<'_>> {
    // `sdb_vname` packs signatures as `sdb:{symbol}`. Strip it so a symbol with
    // no package path (`sdb:Foo#`) does not carry the prefix into the name.
    descriptor_chain_kind(symbol.strip_prefix("sdb:").unwrap_or(symbol))
}

/// Parse a bare SCIP/SemanticDB descriptor chain, the part after any metadata
/// prefix. Shared by [`scip_name_kind`] and [`semanticdb_name_kind`].
fn descriptor_chain_kind(descriptor_chain: &str) -> Option<ScipName<'_>> {
    // Namespace descriptor `Name/`: Obj-C emits protocols this way
    // (`Speakable/`), and a protocol is a unifiable type (Phase A `protocol:`).
    // Restrict to a SINGLE-segment name — multi-segment package paths
    // (`github.com/org/repo/pkg/`, `com/example/`) are Go/Java packages that
    // Phase A does not model, so they stay unparsed to avoid false unification.
    if let Some(name) = descriptor_chain.strip_suffix('/') {
        if name.is_empty() || name.contains('/') {
            return None;
        }
        return Some(ScipName {
            container: None,
            name: strip_backticks(name),
            kind: "class",
        });
    }

    // Strip the package-path prefix at the last `/` that is not inside a
    // backtick-escaped segment — a test-description scope can contain `/`
    // (scip-ruby: `` `<it 'uses /var/tmp'>`(). ``), which is part of the name,
    // not a path separator.
    let leaf = match rfind_outside_backticks(descriptor_chain, '/') {
        Some(i) => &descriptor_chain[i + 1..],
        None => descriptor_chain,
    };

    if leaf.ends_with(").") {
        // Method/function: `Name().`, `Type#Name().`, `Name(+1).`
        let open = rfind_outside_backticks(leaf, '(')?;
        let (container, name) = split_container(&leaf[..open]);
        if name.is_empty() {
            return None;
        }
        Some(ScipName {
            container,
            name,
            kind: "function",
        })
    } else if let Some(stripped) = leaf.strip_suffix('#') {
        let (container, name) = split_container(stripped);
        if name.is_empty() {
            return None;
        }
        Some(ScipName {
            container,
            name,
            kind: "class",
        })
    } else if let Some(stripped) = leaf.strip_suffix('.') {
        // Single `.` suffix — field/var. The `().` case is handled above.
        let (container, name) = split_container(stripped);
        if name.is_empty() || name.contains('/') || name.contains(':') {
            return None;
        }
        Some(ScipName {
            container,
            name,
            kind: "variable",
        })
    } else {
        None
    }
}

/// Split `Type#name` into `(Some("Type"), "name")`; bare names pass through.
/// Nested types (`Outer#Inner#name`) keep only the innermost container, which
/// matches the single-level `Container.name` qualification Phase A emits.
///
/// SCIP descriptors may be backtick-escaped (e.g. scip-go wraps identifiers
/// containing special characters: `` `Kubelet`#`syncPod`(). ``).  One pair of
/// surrounding backticks is stripped from both container and name so they
/// match Phase A signatures.  Backticks containing `/` or `(` (package
/// descriptors) are not tokenized here — those never parse as name/kind.
fn split_container(s: &str) -> (Option<&str>, &str) {
    // Split on the last `#` that is not inside a backtick-escaped segment. A
    // scip-ruby RSpec scope embeds `#` in its description (`` `<describe
    // '#matches_identifiers?'>` ``); those are part of the name, not the
    // `Container#name` separator, so a naive `rsplit('#')` would tear the scope
    // apart into garbage that then escapes DSL detection and mis-parses real
    // names.
    match rfind_outside_backticks(s, '#') {
        Some(i) => {
            let name = &s[i + 1..];
            let pre = &s[..i];
            // Innermost container = the segment after `pre`'s own last
            // outside-backtick `#` (nested types keep only the innermost).
            let container_raw = match rfind_outside_backticks(pre, '#') {
                Some(j) => &pre[j + 1..],
                None => pre,
            };
            let container = unwrap_meta_container(strip_backticks(container_raw));
            (
                (!container.is_empty()).then_some(container),
                // Unwrap the leaf too: a `<Class:X>`/`<Module:X>` leaf is the
                // *singleton class* of a real type (`Tunes#`<Class:Members>`#`),
                // which has no separate Phase A node; reducing it to `X` lets the
                // reference unify onto the real `class:X`. A synthetic DSL leaf
                // (`<describe …>`) is left unchanged and still recognized.
                unwrap_meta_container(strip_backticks(name)),
            )
        }
        None => (None, unwrap_meta_container(strip_backticks(s))),
    }
}

/// Byte index of the last `ch` in `s` that is not inside a backtick-escaped
/// segment. SCIP wraps any identifier containing a structural character
/// (`#`, `/`, `(`, space, `.`) in backticks, so such a character inside a
/// backtick pair belongs to the name and must not be read as a descriptor
/// separator. Backtick regions are delimited by single backticks (the common
/// scip-go / scip-ruby form); the scan simply toggles in/out on each one.
fn rfind_outside_backticks(s: &str, ch: char) -> Option<usize> {
    let mut in_tick = false;
    let mut found = None;
    for (i, c) in s.char_indices() {
        if c == '`' {
            in_tick = !in_tick;
        } else if c == ch && !in_tick {
            found = Some(i);
        }
    }
    found
}

/// Unwrap scip-ruby's Sorbet meta-class container notation to the bare name.
///
/// scip-ruby encodes a class method's enclosing scope as the *singleton class*
/// of the type: `def self.is_supported?` inside `class EnsureBundleExecAction`
/// becomes the descriptor `…#`<Class:EnsureBundleExecAction>`#is_supported?().`.
/// Phase A's Ruby parser names the enclosing scope plainly
/// (`method:EnsureBundleExecAction.is_supported?`), so without unwrapping, the
/// derived candidate `method:<Class:EnsureBundleExecAction>.is_supported?` never
/// matches and the SCIP definition orphans as a duplicate node that silently
/// steals the method's call edges (the twin the tree-sitter node never sees).
/// `<Class:Name>` / `<Module:Name>` → `Name`; anything else (RSpec `<describe
/// '...'>` DSL blocks, which have no Phase A twin) passes through unchanged.
fn unwrap_meta_container(s: &str) -> &str {
    if let Some(inner) = s.strip_prefix('<').and_then(|i| i.strip_suffix('>')) {
        if let Some((tag, name)) = inner.split_once(':') {
            if matches!(tag, "Class" | "Module")
                && !name.is_empty()
                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
            {
                return name;
            }
        }
    }
    s
}

/// `true` when a parsed SCIP name is a Sorbet synthetic DSL meta-scope rather
/// than a real definition — scip-ruby (built on Sorbet) models every RSpec block
/// (`describe`, `context`, `it`, `before`, …) as a singleton scope and emits a
/// descriptor for it (`` `<describe 'Foo'>`#`<it 'does x'>`(). ``). Tree-sitter
/// correctly sees these as method calls with a block, not definitions, so no
/// Phase A twin exists or can exist. Counting them as def misses is what inflated
/// issue #780's rate to ~52%; the daemon excludes them from the miss counters and
/// drops them so they stop stealing spec-file reference edges.
///
/// True when the leaf `name` or its `container` is a DSL meta-scope: a bracketed
/// `<…>` segment that [`unwrap_meta_container`] does NOT resolve to a real
/// `<Class:Name>`/`<Module:Name>`.
///
/// This is the leaf-OR-container disjunction. The unification pass must NOT act
/// on it directly — a real def inside a `describe` block ([`is_dsl_scope_container`])
/// has a Phase A twin and must be reconciled, not dropped. Use the split
/// predicates below and drop only the [`is_dsl_scope_leaf`] case outright.
pub fn is_synthetic_dsl_scope(parsed: &ScipName<'_>) -> bool {
    is_dsl_scope_leaf(parsed) || is_dsl_scope_container(parsed)
}

/// The definition's own leaf `name` is a DSL meta-scope (`<it 'does x'>`,
/// `<describe 'Foo'>` as the name itself). scip-ruby emits this for the RSpec
/// block; tree-sitter sees a method call with a block, not a definition, so no
/// Phase A twin exists or can. Drop it outright.
pub fn is_dsl_scope_leaf(parsed: &ScipName<'_>) -> bool {
    is_dsl_meta_scope(parsed.name)
}

/// The leaf is a real name but its `container` is a DSL meta-scope — a helper
/// (`class Helper`, `def helper`) defined inside an RSpec `describe`/`context`
/// block. Phase A emits it unqualified (a block is not a `method_container`),
/// so the SCIP def CAN reconcile once the unreconcilable container is cleared.
/// The caller retries unification against the unqualified candidates rather than
/// dropping the def, and counts it only if it unifies.
pub fn is_dsl_scope_container(parsed: &ScipName<'_>) -> bool {
    !is_dsl_meta_scope(parsed.name) && parsed.container.is_some_and(is_dsl_meta_scope)
}

/// A single descriptor segment is a Sorbet DSL meta-scope: bracketed `<…>`, its
/// first inner character alphabetic (so a Ruby operator method whose name merely
/// starts with `<` — `<`, `<<`, `<=>` — is NOT misread as a scope), and not a
/// real singleton class/module.
fn is_dsl_meta_scope(s: &str) -> bool {
    let Some(inner) = s.strip_prefix('<').and_then(|i| i.strip_suffix('>')) else {
        return false;
    };
    // A `<Class:…>` / `<Module:…>` singleton marker is a real scope, never a DSL
    // block — even when its name does not reduce via `unwrap_meta_container` (a
    // namespaced `<Class:Foo::Bar>`, whose `::` fails the alphanumeric check).
    // Without this guard such a metaclass would be misread as a DSL scope and
    // its methods dropped.
    if let Some((tag, _)) = inner.split_once(':') {
        if matches!(tag, "Class" | "Module") {
            return false;
        }
    }
    inner.chars().next().is_some_and(|c| c.is_alphabetic()) && unwrap_meta_container(s) == s
}

/// Strip ONE pair of surrounding backticks from a SCIP escaped identifier.
/// `` `name` `` → `name`; anything else passes through unchanged.
fn strip_backticks(s: &str) -> &str {
    if s.len() >= 2 && s.starts_with('`') && s.ends_with('`') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Phase A signature candidates for a parsed SCIP name, ordered most→least
/// specific.  Covers every Phase A parser convention in one list:
///   functions: `method:C.n` (Go/Python/TS), `fn:C.n` (Rust), `fn:n` (all,
///              and the only form for Java/Kotlin/Scala/C/C++/C#/Ruby/PHP/
///              Swift/Dart, whose parsers emit unqualified method names)
///   types:     `class:` `struct:` `interface:` `trait:` `enum:` `type:`
///              `protocol:` `namespace:` `actor:`
///              (Kotlin/Scala `object`, Dart `mixin`/`extension`, and Swift
///              `protocol` all emit sig prefix `class:` — already covered.
///              Obj-C `protocol` emits `protocol:` and C++ `namespace` emits
///              `namespace:` [E6]. N4d split Swift's folded types into distinct
///              prefixes, so `actor:` (a SCIP `#`-type target) is added here;
///              `extension:` is deliberately omitted — an extension is not a
///              SCIP definition target and must not steal the extended type's
///              unification.)
///   terms:     `field:C.n` (owner-qualified field, #757), `var:` `const:`
///              `static:` (unqualified package/global terms)
pub fn candidate_signatures(parsed: &ScipName<'_>) -> Vec<String> {
    let name = parsed.name;
    match parsed.kind {
        "function" => {
            let mut sigs = Vec::with_capacity(6);
            if let Some(c) = parsed.container {
                // A constructor's SCIP leaf is a fixed marker, not the name the
                // Phase A parser saw. scip-dotnet emits `Type#`.ctor`().`, but
                // tree-sitter reads a `constructor_declaration` whose name IS the
                // type, so Phase A wrote `method:Type.Type`. Without this the two
                // never meet: every C# constructor def stayed an orphan SCIP node
                // and took its whole ref/call fan-in with it, so `graph <Type>
                // --direction callers` reached none of the `new Type(...)` sites.
                //
                // Container-qualified forms only. A bare `fn:{c}` would also be
                // fed to rung 2, whose corpus-wide uniqueness gate has no
                // container qualification and no position check, so an
                // unrelated top-level `fn:Foo` anywhere in the repo could be its
                // single match and absorb every `new Foo(...)` edge. It buys
                // nothing anyway: C# routes through `generic.rs`, which writes
                // `class:`/`struct:`/`interface:`/`enum:` for the type and
                // qualifies a `constructor_declaration` by its enclosing type
                // container, so Phase A never emits `fn:Foo` for a C# type.
                if name == ".ctor" {
                    sigs.push(format!("method:{c}.{c}"));
                    sigs.push(format!("fn:{c}.{c}"));
                }
                sigs.push(format!("method:{c}.{name}"));
                sigs.push(format!("fn:{c}.{name}"));
                // A bespoke sidecar (`native_name_kind`: kotlin KLS, swift,
                // dart) reports the FULL dotted nesting path as the container,
                // while Phase A qualifies a member by the single nearest NAMED
                // type container (`enclosing_container`). The two disagree
                // wherever the nesting adds a scope Phase A does not qualify
                // by: a Kotlin companion object
                // (`Annotations.Companion.isList` vs `method:Annotations.isList`),
                // a nested class (`Issue224Test.Data2.toString` vs
                // `method:Data2.toString`), an object expression inside a
                // function (`PathMatcherTest.scatteredObject.pm.onMatch` vs
                // `method:PathMatcherTest.onMatch`). None of those candidates
                // matched, so the definition never unified and survived as a
                // SECOND node covering the same source span as its Phase A
                // twin. Caller attribution (`fetch_all_fn_spans` +
                // `find_narrowest_enclosing`) then had two equal-width
                // candidates and broke the tie on `id`, a content hash, so
                // which one owned the call depended on whether the sidecar's
                // node happened to be there. Measured on klaxon: 42 exact span
                // collisions, 112 of 2337 in-span sites decided that way.
                // Which segment Phase A used is not recoverable from the
                // string, so offer every one and let the same-file + in-span
                // gate in `find_ts_node_for_unification` pick. SCIP
                // descriptors keep only the innermost container
                // (`split_container`) and never contain a `.`, so this arm is
                // inert for every SCIP language.
                if c.contains('.') {
                    for seg in c.split('.') {
                        sigs.push(format!("method:{seg}.{name}"));
                        sigs.push(format!("fn:{seg}.{name}"));
                    }
                }
            }
            sigs.push(format!("fn:{name}"));
            // #449 used to add leading-keyword candidates here
            // (`method:Foo.setWidth` for `setWidth:height:`) because Phase A
            // truncated an Objective-C selector to its leading keyword. Phase A
            // now stores the whole selector, so the full-selector candidates
            // above match directly and a leading-keyword candidate could only
            // ever match a DIFFERENT method: a genuinely zero-argument
            // `- (void)setWidth` in the same class. That is not a harmless
            // last-resort either, because `find_ts_node_for_unification` orders
            // span-containment ahead of candidate priority, so the wrong node
            // wins whenever its span contains the SCIP line.
            sigs
        }
        "class" => [
            "class",
            "struct",
            "interface",
            "trait",
            "enum",
            "type",
            // E6: the only type prefixes not folded into `class:` by Phase A —
            // Obj-C protocols (`protocol:`) and C++ namespaces (`namespace:`).
            "protocol",
            "namespace",
            // N4d: Swift `actor` is now a distinct Phase A prefix and a valid
            // SCIP `#`-type unification target.
            "actor",
        ]
        .iter()
        .map(|p| format!("{p}:{name}"))
        .collect(),
        "variable" => {
            // #757: an owner-qualified field node (`field:Owner.name`, emitted
            // by every declaration-language Phase A parser) is the most-specific
            // target for a SCIP field reference (`Type#name.`,
            // `swift::Type.name`). Try it first when a container is known, then
            // fall back to the unqualified term prefixes some parsers still emit
            // for package-level vars/consts.
            let mut sigs = Vec::with_capacity(4);
            if let Some(c) = parsed.container {
                sigs.push(format!("field:{c}.{name}"));
            }
            sigs.push(format!("var:{name}"));
            sigs.push(format!("const:{name}"));
            sigs.push(format!("static:{name}"));
            sigs
        }
        _ => Vec::new(),
    }
}

/// The container-qualified candidates alone, for the overload-collapse rung.
///
/// Phase A signatures carry no parameter list, so every overload of `C.n` in a
/// file hashes to ONE tree-sitter node, anchored at the first overload's span.
/// SCIP keeps them apart (`C#n().`, `C#n(+1).`, ...), so only the first overload
/// falls inside that span; the rest miss both span-containment and the +/-5
/// proximity window and orphan. That is not a line-distance problem, it is a
/// cardinality mismatch: N SCIP defs legitimately share one Phase A node.
///
/// The container qualification is what makes resolving it by uniqueness safe.
/// The guarantee is about nodes, not method groups: the `nodes` unique index is
/// `(corpus, root, path, language, signature)` and Phase A does not qualify a
/// signature by namespace, so a single match means exactly one NODE named
/// `method:C.n` in that file. Two same-named containers in one file (e.g.
/// `namespace A { class C { void n(); } }` and `namespace B { class C { ... } }`)
/// already share that one node before this rung runs, and the rung then binds
/// both SCIP defs to it. That is accepted: the alternative is an orphan node
/// holding the fan-in, which is worse, and the shape is rare.
/// The unqualified `fn:n` fallback carries no guarantee at all and is
/// deliberately excluded: an unrelated free function of the same name would be
/// the "unique" match and would silently absorb the edges.
///
/// Returns empty for anything that is not a container-qualified callable, which
/// disables the rung rather than widening it.
pub fn overload_collapse_signatures(parsed: &ScipName<'_>) -> Vec<String> {
    if parsed.kind != "function" {
        return Vec::new();
    }
    let Some(c) = parsed.container else {
        return Vec::new();
    };
    let name = parsed.name;
    if name == ".ctor" {
        return vec![format!("method:{c}.{c}"), format!("fn:{c}.{c}")];
    }
    vec![format!("method:{c}.{name}"), format!("fn:{c}.{name}")]
}

/// Parse a *bespoke-sidecar* node signature into a [`ScipName`] for G1
/// unification.
///
/// Some Phase B providers (kotlin-language-server, the Swift index emitter) do
/// not run a SCIP tool and therefore emit node signatures in the Phase A
/// convention (`fn:Container.name`, `class:Type`) or a scheme-prefixed form
/// (`swift::Container.name`) rather than a SCIP descriptor chain. These never
/// parse via [`scip_name_kind`], so their definition node would survive as a
/// duplicate of the tree-sitter node. This parser recovers `(container, name,
/// kind)` from those signatures, using the caller-supplied `node_kind` (the
/// Travsr node-kind string) as the authoritative kind class, so the same
/// `candidate_signatures` / line-proximity matching used for SCIP languages
/// can unify them.
///
/// Returns `None` when `node_kind` is not a function/type/term class or the
/// signature has no usable leaf name.
pub fn native_name_kind<'a>(signature: &'a str, node_kind: &str) -> Option<ScipName<'a>> {
    let kind = match node_kind {
        "function" | "method" | "constructor" => "function",
        "class" | "struct" | "interface" | "trait" | "enum" | "type" | "protocol" | "object"
        | "extension" => "class",
        "field" | "variable" | "const" | "constant" | "property" | "var" | "static" => "variable",
        _ => return None,
    };

    // Strip the bespoke-sidecar scheme so only the dotted `Container.name`
    // remains. Two shapes:
    //   • `<scheme>::<Dotted.Name>` — swift emitter (`swift::Animal.describe`)
    //     and the Dart index emitter, whose symbols embed the *file path* before
    //     the `::` separator (`file:///abs/x.dart::Animal.describe`,
    //     `package:pkg/x.dart::Animal.describe`). The dotted name follows the
    //     LAST `::`, so `rfind("::")` isolates it and drops the path — without
    //     which the Dart container resolves to `…/x.dart::Animal` (path junk)
    //     and the `method:Animal.describe` candidate is never generated.
    //   • `<prefix>:<Dotted.Name>` — a single-word Phase-A-style kind prefix
    //     (`fn:`, `class:`, `var:`, kotlin KLS `enum:`/`interface:`). Only an
    //     all-alphabetic prefix is stripped so a qualified name that legitimately
    //     contains a single colon is never truncated.
    let body = if let Some(idx) = signature.rfind("::") {
        &signature[idx + 2..]
    } else if let Some((prefix, rest)) = signature.split_once(':') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_alphabetic()) {
            rest
        } else {
            signature
        }
    } else {
        signature
    };

    // #825: the body must be a symbol name, not a path. A URI scheme (`file:`
    // in `file:///abs/x.dart`) and a Windows drive letter (`C:` in
    // `C:\src\x.rs`) are both all-alphabetic, so the single-colon strip above
    // cannot tell them from a kind prefix and eats them, and the `.` split
    // below then reads the file extension as the symbol: `file:///C:/src/x.dart`
    // became container `///C:/src/x`, name `dart`. That is a corrupted value in
    // the `travsr status` miss list (#825 Part A prints it verbatim) and, worse,
    // a bogus `class:dart` candidate that could unify a def onto an unrelated
    // node. A path carries no usable leaf name, which is already this function's
    // documented `None` condition. The real emitter shapes always keep a `::`
    // separator before the dotted name and take the branch above, so they are
    // unaffected.
    if body.contains(['/', '\\']) {
        return None;
    }

    // Split `Container.name` on the last `.`; a bare name has no container.
    let (container, name) = match body.rsplit_once('.') {
        Some((c, n)) if !c.is_empty() && !n.is_empty() => (Some(c), n),
        _ => (None, body),
    };
    if name.is_empty() {
        return None;
    }
    Some(ScipName {
        container,
        name,
        kind,
    })
}

/// Extract the raw SCIP symbol string from a node's VName signature.
///
/// scip-reader packs signatures as `"scip:{rel_path}:{symbol}"`.
/// `ingest_scip_g2` stores just the raw symbol.  Both cases are handled.
pub fn scip_symbol_from_sig(sig: &str) -> &str {
    if let Some(rest) = sig.strip_prefix("scip:") {
        if let Some(pos) = rest.find(':') {
            return &rest[pos + 1..];
        }
    }
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed<'a>(container: Option<&'a str>, name: &'a str, kind: &'a str) -> ScipName<'a> {
        ScipName {
            container,
            name,
            kind,
        }
    }

    // A SemanticDB symbol is the descriptor chain with no SCIP metadata
    // prefix. `scip_name_kind` requires that prefix, so scala's defs parsed as
    // nothing and never became unification candidates: ref edges pointed at the
    // `sdb:` node while `references <name>` resolved to the Phase A twin, and
    // every scala query returned 0 against a populated graph.
    #[test]
    fn semanticdb_method_parses_without_a_scip_metadata_prefix() {
        assert_eq!(
            scip_name_kind("scala/util/parsing/combinator/Parsers#phrase()."),
            None,
            "no metadata prefix, so the SCIP parser must still reject it"
        );
        assert_eq!(
            semanticdb_name_kind("sdb:scala/util/parsing/combinator/Parsers#phrase()."),
            Some(parsed(Some("Parsers"), "phrase", "function"))
        );
    }

    #[test]
    fn semanticdb_class_and_prefixless_symbol() {
        assert_eq!(
            semanticdb_name_kind("sdb:scala/util/parsing/combinator/Parsers#"),
            Some(parsed(None, "Parsers", "class"))
        );
        // No package path: the `sdb:` prefix must not leak into the name.
        assert_eq!(
            semanticdb_name_kind("sdb:Foo#"),
            Some(parsed(None, "Foo", "class"))
        );
    }

    #[test]
    fn go_function() {
        assert_eq!(
            scip_name_kind("scip-go go github.com/org/repo v1.0.0 pkg/HandleRequest()."),
            Some(parsed(None, "HandleRequest", "function"))
        );
    }

    #[test]
    fn go_method_has_container() {
        assert_eq!(
            scip_name_kind("scip-go go github.com/org/repo v1.0.0 pkg/Server#Serve()."),
            Some(parsed(Some("Server"), "Serve", "function"))
        );
    }

    #[test]
    fn go_type() {
        assert_eq!(
            scip_name_kind("scip-go go github.com/org/repo v1.0.0 pkg/Server#"),
            Some(parsed(None, "Server", "class"))
        );
    }

    #[test]
    fn go_variable() {
        assert_eq!(
            scip_name_kind("scip-go go github.com/org/repo v1.0.0 pkg/ErrNotFound."),
            Some(parsed(None, "ErrNotFound", "variable"))
        );
    }

    #[test]
    fn java_overload_disambiguator() {
        assert_eq!(
            scip_name_kind("semanticdb maven jdk 11 java/util/List#add(+1)."),
            Some(parsed(Some("List"), "add", "function"))
        );
    }

    #[test]
    fn rust_analyzer_method() {
        assert_eq!(
            scip_name_kind("rust-analyzer cargo serde 1.0.0 ser/Serializer#serialize()."),
            Some(parsed(Some("Serializer"), "serialize", "function"))
        );
    }

    #[test]
    fn python_class_method() {
        assert_eq!(
            scip_name_kind("scip-python python pkg 1.0 mod/Animal#speak()."),
            Some(parsed(Some("Animal"), "speak", "function"))
        );
    }

    #[test]
    fn nested_type_keeps_innermost_container() {
        assert_eq!(
            scip_name_kind("semanticdb maven . . pkg/Outer#Inner#run()."),
            Some(parsed(Some("Inner"), "run", "function"))
        );
    }

    #[test]
    fn class_field_has_container() {
        assert_eq!(
            scip_name_kind("scip-python python pkg 1.0 mod/Animal#name."),
            Some(parsed(Some("Animal"), "name", "variable"))
        );
    }

    #[test]
    fn no_suffix_is_none() {
        assert_eq!(scip_name_kind("scip-go go pkg v1.0.0 something"), None);
    }

    #[test]
    fn objc_protocol_namespace_descriptor_parses_as_class() {
        // #596: ObjC protocols are emitted as `Name/`; they must parse as a
        // type so they unify with the Phase A `protocol:` node.
        let p = scip_name_kind("objc . local/objc 0.0.0 Speakable/").unwrap();
        assert_eq!(p, parsed(None, "Speakable", "class"));
        assert!(candidate_signatures(&p).contains(&"protocol:Speakable".to_string()));
    }

    #[test]
    fn multi_segment_package_descriptor_is_none() {
        // Go/Java package paths end in `/` too but are multi-segment and have
        // no Phase A node — must stay unparsed to avoid false unification.
        assert_eq!(
            scip_name_kind("scip-go go github.com/org/repo v1.0.0 github.com/org/repo/pkg/"),
            None
        );
    }

    #[test]
    fn native_phase_a_sig_is_none() {
        // Non-SCIP signatures from builtin Phase B plugins must fall through.
        assert_eq!(scip_name_kind("fn:Type.method"), None);
        assert_eq!(scip_name_kind("class:Foo"), None);
    }

    #[test]
    fn native_name_kind_kotlin_qualified_method() {
        // kotlin sidecar node signature `fn:Container.method` → drop the `fn:`
        // prefix, split the container, kind from the node kind.
        assert_eq!(
            native_name_kind("fn:Animal.describe", "function"),
            Some(parsed(Some("Animal"), "describe", "function"))
        );
    }

    #[test]
    fn native_name_kind_swift_scheme_prefix() {
        // swift `swift::Container.method` → strip the `swift::` scheme.
        assert_eq!(
            native_name_kind("swift::Animal.describe", "method"),
            Some(parsed(Some("Animal"), "describe", "function"))
        );
        // swift top-level type `swift::Animal`, kind class.
        assert_eq!(
            native_name_kind("swift::Animal", "class"),
            Some(parsed(None, "Animal", "class"))
        );
    }

    #[test]
    fn native_name_kind_path_shaped_signature_is_none() {
        // #825: a URI scheme (`file:`) and a Windows drive letter (`C:`) are both
        // all-alphabetic, so the single-colon kind-prefix strip could not tell
        // them from `fn:` / `class:` and ate them, after which the `.` split read
        // the file extension as the symbol name. `file:///C:/src/x.dart` came out
        // as container `///C:/src/x`, name `dart` — a corrupted row in the
        // `travsr status` miss list, and a bogus `class:dart` candidate that
        // could unify a def onto an unrelated node. A path has no usable leaf
        // name, so the answer is None.
        for sym in [
            "file:///C:/src/x.dart",
            "file:///abs/x.dart",
            "C:\\src\\x.rs",
            "package:pkg/animal.dart",
        ] {
            assert_eq!(native_name_kind(sym, "class"), None, "sym={sym}");
        }
        // The real emitter shapes keep their `::` separator and are unaffected,
        // Windows drive letter included.
        let p = native_name_kind("file:///C:/src/x.dart::Animal.describe", "function").unwrap();
        assert_eq!(p, parsed(Some("Animal"), "describe", "function"));
        // And an ordinary Phase-A-style prefix still strips.
        assert_eq!(
            native_name_kind("class:Foo", "class"),
            Some(parsed(None, "Foo", "class"))
        );
    }

    #[test]
    fn native_name_kind_dart_package_symbol() {
        // dart `package:pkg/file.dart::Type.method` and the real emitter's
        // `file:///abs/x.dart::Type.method` — `rfind("::")` isolates the dotted
        // name so the container leaf (`Animal`) is recovered and the
        // `method:Animal.describe` candidate (Phase A N1 qualification) is
        // generated, not just the bare `fn:describe`.
        for sym in [
            "package:pkg/animal.dart::Animal.describe",
            "file:///private/tmp/app/lib/animal.dart::Animal.describe",
        ] {
            let p = native_name_kind(sym, "function").unwrap();
            assert_eq!(p.container, Some("Animal"), "sym={sym}");
            assert_eq!(p.name, "describe", "sym={sym}");
            assert_eq!(p.kind, "function");
            let cands = candidate_signatures(&p);
            assert!(
                cands.contains(&"method:Animal.describe".to_string()),
                "sym={sym} cands={cands:?}"
            );
            assert!(cands.contains(&"fn:describe".to_string()), "sym={sym}");
        }
    }

    #[test]
    fn native_name_kind_dart_top_level_and_type() {
        // Top-level fn `file:///abs/x.dart::main` → bare `main` (fn:main).
        let p = native_name_kind("file:///abs/x.dart::main", "function").unwrap();
        assert_eq!(p.container, None);
        assert_eq!(p.name, "main");
        // Class twin `file:///abs/x.dart::Animal` (kind class) → class family.
        let c = native_name_kind("file:///abs/x.dart::Animal", "class").unwrap();
        assert_eq!(c.name, "Animal");
        assert!(candidate_signatures(&c).contains(&"class:Animal".to_string()));
    }

    #[test]
    fn native_name_kind_bare_and_unknown_kind() {
        assert_eq!(
            native_name_kind("fn:describe", "function"),
            Some(parsed(None, "describe", "function"))
        );
        // Unknown/non-def kinds fall through.
        assert_eq!(native_name_kind("fn:describe", "file"), None);
    }

    #[test]
    fn ruby_singleton_class_method_unwraps_to_bare_container() {
        // scip-ruby wraps a class method's scope in the Sorbet singleton-class
        // notation `<Class:Name>`. It must reduce to `Name` so the derived
        // candidate `method:EnsureBundleExecAction.is_supported?` matches the
        // Phase A tree-sitter node instead of orphaning a duplicate twin.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 Fastlane#Actions#`<Class:EnsureBundleExecAction>`#`is_supported?`().",
        )
        .unwrap();
        assert_eq!(p.container, Some("EnsureBundleExecAction"));
        assert_eq!(p.name, "is_supported?");
        assert_eq!(p.kind, "function");
        assert!(candidate_signatures(&p)
            .contains(&"method:EnsureBundleExecAction.is_supported?".to_string()));
    }

    #[test]
    fn dsl_descriptor_with_spaces_tokenizes_and_is_synthetic() {
        // #780 RC-1b: an RSpec `it` block's descriptor is backtick-escaped and
        // contains spaces. `split_whitespace().last()` used to tear it into
        // garbage that still counted as a def miss; the 4-space-prefix split now
        // keeps the whole chain, and the result is recognized as synthetic.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 `<describe 'Fastlane'>`#`<it 'sets the platform to iOS'>`().",
        )
        .unwrap();
        assert_eq!(p.name, "<it 'sets the platform to iOS'>");
        assert_eq!(p.container, Some("<describe 'Fastlane'>"));
        assert_eq!(p.kind, "function");
        assert!(is_synthetic_dsl_scope(&p), "DSL block must be synthetic");
    }

    #[test]
    fn dsl_scope_with_hash_in_description_is_synthetic() {
        // #780 RC-1b: an RSpec `describe '#method'` scope embeds a `#` inside its
        // backtick-quoted description. A `#`-naive split tore it into garbage that
        // escaped DSL detection and stayed counted; the backtick-aware split keeps
        // the scope intact so it is recognized and excluded.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 `<describe 'BetaGroup'>`#`<describe '#matches_identifiers?'>`#",
        )
        .unwrap();
        assert_eq!(p.kind, "class");
        assert_eq!(p.name, "<describe '#matches_identifiers?'>");
        assert!(
            is_synthetic_dsl_scope(&p),
            "scope with '#' must be synthetic"
        );
    }

    #[test]
    fn dsl_scope_with_slash_in_description_parses_and_is_synthetic() {
        // #780 RC-1b: a test description can contain `/` (`uses /var/tmp`), which
        // is not a package-path separator. Backtick-aware leaf extraction keeps
        // the whole chain so the scope parses and is recognized as synthetic.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 `<describe 'Client'>`#`<it 'uses /var/tmp if home not available'>`().",
        )
        .unwrap();
        assert_eq!(p.name, "<it 'uses /var/tmp if home not available'>");
        assert!(is_synthetic_dsl_scope(&p));
    }

    #[test]
    fn dsl_type_block_trailing_hash_is_synthetic() {
        // #780 category C: a `describe` block with a trailing `#` parses as a
        // class-kind name that is itself a DSL scope.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 `<describe 'Match'>`#`<describe 'Setup'>`#",
        )
        .unwrap();
        assert_eq!(p.kind, "class");
        assert!(
            is_synthetic_dsl_scope(&p),
            "DSL type block must be synthetic"
        );
    }

    #[test]
    fn real_ruby_method_is_not_synthetic() {
        // A genuine `Class#method().` must NOT be classified synthetic — it is a
        // real def that Phase A should reconcile and the miss rate should count.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 Supply#GeneratedUniversalApk#package_name().",
        )
        .unwrap();
        assert_eq!(p.container, Some("GeneratedUniversalApk"));
        assert_eq!(p.name, "package_name");
        assert!(!is_synthetic_dsl_scope(&p));
    }

    #[test]
    fn describe_block_leaf_vs_container_split() {
        // #780 defect 1: a def whose *leaf* is the block (`<it '...'>`) has no
        // twin and is a leaf-drop; a real def *inside* a `describe` block
        // (`<describe 'Foo'>#helper().`) is a container-only case that the
        // unifier must recover by clearing the container, not drop.
        let leaf =
            scip_name_kind("scip-ruby gem fastlane 0.0.0 `<describe 'Foo'>`#`<it 'does x'>`().")
                .unwrap();
        assert!(is_dsl_scope_leaf(&leaf), "block-as-leaf is a leaf drop");
        assert!(!is_dsl_scope_container(&leaf));

        let contained =
            scip_name_kind("scip-ruby gem fastlane 0.0.0 `<describe 'Foo'>`#`helper`().").unwrap();
        assert!(
            !is_dsl_scope_leaf(&contained),
            "real leaf is not a leaf drop"
        );
        assert!(
            is_dsl_scope_container(&contained),
            "real def inside a describe block is a container case"
        );
        assert_eq!(contained.name, "helper");
        assert_eq!(contained.container, Some("<describe 'Foo'>"));
        // Both are still `synthetic` under the disjunction (existing callers).
        assert!(is_synthetic_dsl_scope(&leaf) && is_synthetic_dsl_scope(&contained));

        // A real class inside the block: container-only, recoverable to class:Helper.
        let cls =
            scip_name_kind("scip-ruby gem fastlane 0.0.0 `<describe 'Foo'>`#Helper#").unwrap();
        assert!(!is_dsl_scope_leaf(&cls));
        assert!(is_dsl_scope_container(&cls));
        assert_eq!(cls.name, "Helper");
    }

    #[test]
    fn namespaced_metaclass_is_not_a_dsl_scope() {
        // Hardening: `<Class:Foo::Bar>` does not reduce via unwrap_meta_container
        // (the `::` fails the alphanumeric check), but a `<Class:…>` singleton
        // marker is a real scope, never a DSL block. It must NOT be classified
        // synthetic, or a namespaced metaclass's methods would be dropped.
        assert!(!is_dsl_meta_scope("<Class:Foo::Bar>"));
        assert!(!is_dsl_meta_scope("<Module:Foo::Bar>"));
        // A genuine DSL block still is.
        assert!(is_dsl_meta_scope("<describe 'Foo'>"));
    }

    #[test]
    fn operator_method_starting_with_angle_is_not_synthetic() {
        // Regression guard: `def <=>` / `def <<` name with `<`, and `<=>` even
        // ends with `>`. Neither is a DSL scope (first inner char is not
        // alphabetic), so they must reconcile as real operator methods.
        for sym in [
            "scip-ruby gem fastlane 0.0.0 Foo#`<=>`().",
            "scip-ruby gem fastlane 0.0.0 Foo#`<<`().",
        ] {
            let p = scip_name_kind(sym).unwrap();
            assert!(
                !is_synthetic_dsl_scope(&p),
                "operator misread as DSL: {sym}"
            );
        }
    }

    #[test]
    fn singleton_class_leaf_unwraps_to_real_type() {
        // #780: a `<Class:X>` leaf (kind class) is the singleton class of `X`; it
        // has no Phase A twin of its own, so it must reduce to `X` and unify onto
        // the real `class:X` instead of orphaning.
        let p = scip_name_kind("scip-ruby gem fastlane 0.0.0 Spaceship#Tunes#`<Class:Members>`#")
            .unwrap();
        assert_eq!(p.kind, "class");
        assert_eq!(p.name, "Members");
        assert_eq!(p.container, Some("Tunes"));
        assert!(!is_synthetic_dsl_scope(&p));
        assert!(candidate_signatures(&p).contains(&"class:Members".to_string()));
    }

    #[test]
    fn class_module_singleton_scope_is_not_synthetic() {
        // `<Class:Name>` unwraps to a real container, so a class method inside it
        // is a real def, not a synthetic scope.
        let p = scip_name_kind(
            "scip-ruby gem fastlane 0.0.0 Fastlane#Actions#`<Class:EnsureBundleExecAction>`#`is_supported?`().",
        )
        .unwrap();
        assert_eq!(p.container, Some("EnsureBundleExecAction"));
        assert!(!is_synthetic_dsl_scope(&p));
    }

    #[test]
    fn rspec_describe_block_container_is_not_unwrapped() {
        // RSpec DSL descriptors (`<describe 'Fastlane'>`) have no Phase A twin;
        // the meta-unwrap must leave them untouched (space/quotes disqualify).
        assert_eq!(
            unwrap_meta_container("<describe 'Fastlane'>"),
            "<describe 'Fastlane'>"
        );
        assert_eq!(unwrap_meta_container("<Class:Helper>"), "Helper");
        assert_eq!(
            unwrap_meta_container("<Module:FastlaneCore>"),
            "FastlaneCore"
        );
        assert_eq!(unwrap_meta_container("Helper"), "Helper");
    }

    #[test]
    fn candidates_function_with_container() {
        let sigs = candidate_signatures(&parsed(Some("Server"), "Serve", "function"));
        assert_eq!(
            sigs,
            vec!["method:Server.Serve", "fn:Server.Serve", "fn:Serve"]
        );
    }

    #[test]
    fn candidates_function_bare() {
        let sigs = candidate_signatures(&parsed(None, "HandleRequest", "function"));
        assert_eq!(sigs, vec!["fn:HandleRequest"]);
    }

    #[test]
    fn objc_method_symbol_parses_as_function() {
        // #449: travsr-lang-objectivec emits `Class#selector().`, SCIP-conformant
        // since the emitter appends the method suffix.
        assert_eq!(
            scip_name_kind("objc . corp 0.0.0 ClassC#registerEnvironments()."),
            Some(parsed(Some("ClassC"), "registerEnvironments", "function"))
        );
        assert_eq!(
            scip_name_kind("objc . corp 0.0.0 Foo#setWidth:height:()."),
            Some(parsed(Some("Foo"), "setWidth:height:", "function"))
        );
    }

    #[test]
    fn candidates_selector_keeps_the_whole_selector() {
        // Phase A stores the whole Objective-C selector, so the full-selector
        // candidates are the only correct targets. A leading-keyword candidate
        // (`method:Foo.setWidth`) would name a different method entirely.
        let sigs = candidate_signatures(&parsed(Some("Foo"), "setWidth:height:", "function"));
        assert_eq!(
            sigs,
            vec![
                "method:Foo.setWidth:height:",
                "fn:Foo.setWidth:height:",
                "fn:setWidth:height:",
            ]
        );
        // Colon-free names are unchanged.
        let sigs = candidate_signatures(&parsed(Some("Foo"), "run", "function"));
        assert_eq!(sigs, vec!["method:Foo.run", "fn:Foo.run", "fn:run"]);
    }

    #[test]
    fn candidates_function_offers_every_segment_of_a_dotted_container() {
        // A bespoke sidecar reports the full nesting path; Phase A qualifies by
        // the nearest NAMED type container, which can be any segment of it: the
        // outermost for a Kotlin companion object (`method:Annotations.isList`),
        // the innermost for a nested class (`method:Data2.toString`). Both must
        // be offered. `candidates_function_with_container` covers the undotted
        // case, which every SCIP language produces and this must not disturb.
        let sigs =
            candidate_signatures(&parsed(Some("Annotations.Companion"), "isList", "function"));
        assert_eq!(
            sigs,
            vec![
                "method:Annotations.Companion.isList",
                "fn:Annotations.Companion.isList",
                "method:Annotations.isList",
                "fn:Annotations.isList",
                "method:Companion.isList",
                "fn:Companion.isList",
                "fn:isList",
            ]
        );
    }

    #[test]
    fn candidates_class_covers_all_type_prefixes() {
        let sigs = candidate_signatures(&parsed(None, "Server", "class"));
        assert_eq!(
            sigs,
            vec![
                "class:Server",
                "struct:Server",
                "interface:Server",
                "trait:Server",
                "enum:Server",
                "type:Server",
                "protocol:Server",
                "namespace:Server",
                "actor:Server"
            ]
        );
    }

    #[test]
    fn candidates_class_covers_objc_protocol_and_cpp_namespace() {
        // E6: an Obj-C protocol node signature is `protocol:Name` and a C++
        // namespace is `namespace:Name` — a SCIP `#` (class) reference to
        // either must produce the matching candidate so it unifies instead of
        // orphaning a duplicate twin.
        let sigs = candidate_signatures(&parsed(None, "Drawable", "class"));
        assert!(sigs.contains(&"protocol:Drawable".to_string()));
        assert!(sigs.contains(&"namespace:Drawable".to_string()));
    }

    #[test]
    fn candidates_class_covers_swift_actor() {
        // N4d: a Swift `actor` node signature is `actor:Name`; a SCIP `#` (class)
        // reference to it must unify onto the Phase A actor node.
        let sigs = candidate_signatures(&parsed(None, "Cache", "class"));
        assert!(sigs.contains(&"actor:Cache".to_string()));
    }

    #[test]
    fn candidates_variable() {
        let sigs = candidate_signatures(&parsed(None, "MAX_LEN", "variable"));
        assert_eq!(sigs, vec!["var:MAX_LEN", "const:MAX_LEN", "static:MAX_LEN"]);
    }

    #[test]
    fn candidates_variable_with_container_prefers_field() {
        // #757: a field reference `Session#name.` must unify onto the
        // owner-qualified Phase A field node `field:Session.name` first, then
        // fall back to the unqualified term prefixes.
        let sigs = candidate_signatures(&parsed(Some("Session"), "name", "variable"));
        assert_eq!(
            sigs,
            vec![
                "field:Session.name",
                "var:name",
                "const:name",
                "static:name"
            ]
        );
    }

    #[test]
    fn scip_symbol_from_scip_reader_sig() {
        let sig = "scip:pkg/foo.go:scip-go go example.com v1.0.0 pkg/Foo#";
        assert_eq!(
            scip_symbol_from_sig(sig),
            "scip-go go example.com v1.0.0 pkg/Foo#"
        );
    }

    #[test]
    fn scip_symbol_from_raw_sig() {
        let sig = "scip-go go example.com v1.0.0 pkg/Bar().";
        assert_eq!(scip_symbol_from_sig(sig), sig);
    }

    #[test]
    fn backtick_escaped_descriptors_are_stripped() {
        // scip-go wraps identifiers that contain special characters in backticks.
        // Both container and name must have surrounding backticks stripped so
        // they can match Phase A signatures like `method:Kubelet.syncPod`.
        assert_eq!(
            scip_name_kind(
                "scip-go go k8s.io/kubernetes v0.0.0 pkg/kubelet/`Kubelet`#`syncPod`()."
            ),
            Some(parsed(Some("Kubelet"), "syncPod", "function"))
        );
        // Bare name without backticks still works.
        assert_eq!(
            scip_name_kind("scip-go go k8s.io/kubernetes v0.0.0 pkg/kubelet/Kubelet#syncPod()."),
            Some(parsed(Some("Kubelet"), "syncPod", "function"))
        );
    }
}
