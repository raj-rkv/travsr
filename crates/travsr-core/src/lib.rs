//! travsr-core — graph primitives for the Travsr code-intelligence daemon.
//!
//! This crate defines the foundational data model: Kythe-style VNames,
//! node identifiers, edges, and the multiplex graph types that every other
//! Travsr crate builds on. It has zero dependencies on any other internal
//! Travsr crate by design (see crate dependency rules in CLAUDE.md).

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub mod exec;
pub mod ident;
pub mod noise;

/// Version of the VName signature format baked into every `NodeId` hash.
///
/// This byte is the **first input** to the BLAKE3 hasher in `VName::id()`.
/// Changing it produces a disjoint `NodeId` space — any `.travsr/graph.db`
/// built with a different version must be fully re-indexed before it can be
/// queried. See `docs/rfcs/RFC-002-vname-signature-versioning.md`.
///
/// Version history:
///   0 — legacy (no version byte; all pre-RFC-002 databases)
///   1 — Tree-sitter vocabulary (`class:X`, `fn:X`, `method:X.Y`, `var:X`)
///   2 — RFC-014 Phase B graph unification. Phase A now captures
///       type-definition nodes and `end_line` spans that the G1/G2 unification
///       passes depend on, so v1 databases lack the tree-sitter nodes that
///       SCIP symbols unify onto. Bumping intentionally invalidates every
///       existing `.travsr/graph.db` so the daemon skew check and the
///       `travsr status` warning force a full re-index (RFC-014 "Re-index
///       Policy").
///   3 - current: Objective-C method signatures carry the WHOLE selector
///       (`method:Class.setWidth:height:`) instead of only its leading keyword
///       (`method:Class.setWidth`). The old form collapsed every selector
///       sharing a first keyword onto one node, so sibling methods lost their
///       own identity and calls between them degenerated into self-loops the
///       store dropped. Node identity therefore changes for every ObjC method.
///       A v2 database cannot be migrated in place: incremental reindex only
///       re-parses files that changed, so an ObjC repo would hold collapsed
///       signatures for untouched files and full selectors for the rest, with
///       nothing able to tell the halves apart.
///
/// The constant does not invalidate anything by itself. It is a marker some code
/// paths compare against, and only those paths act on a bump:
///   * The write paths refuse to advance the graph or the freshness marker when
///     the stored version differs. `reindex_files` (the commit hook and the
///     watcher) indexes nothing and returns success, logging the reason to the
///     daemon log so the hook never blocks a commit; `reconcile_head_drift` and
///     the CLI reindex leave `last_commit` unstamped so freshness is not claimed
///     for a reindex that never ran.
///   * `travsr status` is what tells the user, printing the format skew and
///     asking for a `travsr init`.
///   * `init_repo_with_progress` reads the stored version before re-stamping it
///     and, on a mismatch, purges the graph and clears the file-hash cache so
///     every file is re-parsed, exactly as `--force` does, rather than taking
///     the incremental path.
///
/// Read paths do not check the version at all. `open_read_only` verifies the
/// schema version only, so queries keep answering from the old-format graph: it
/// is stale, not corrupt, and the write-path refusals above are what stop the
/// two formats from ever mixing.
pub const SIGNATURE_FORMAT_VERSION: u8 = 3;

// ── Corpus derivation (ARCH-102) ─────────────────────────────────────────────

/// Derive the canonical Travsr corpus identifier from a git remote URL.
///
/// **Canonical form:** `host/org/repo` — all lowercase, no scheme prefix,
/// no `.git` suffix, no trailing slash. See `docs/rfcs/ARCH-102`.
///
/// Handles all standard git remote URL formats:
///
/// | Input | Output |
/// |---|---|
/// | `https://github.com/acme/foo.git` | `github.com/acme/foo` |
/// | `git@github.com:acme/foo.git`     | `github.com/acme/foo` |
/// | `ssh://git@github.com/acme/foo`   | `github.com/acme/foo` |
/// | `git://github.com/acme/foo.git`   | `github.com/acme/foo` |
pub fn canonical_corpus(remote_url: &str) -> String {
    let s = remote_url.trim();

    // SCP-style SSH: git@host:org/repo[.git]
    if let Some(rest) = s.strip_prefix("git@") {
        if let Some((host, path)) = rest.split_once(':') {
            return format!("{}/{}", host.to_lowercase(), normalize_path(path));
        }
    }

    // URL schemes: https://, http://, ssh://, git://
    let after_scheme = s.split_once("://").map_or(s, |(_, r)| r);
    // Strip userinfo (ssh://git@host/path → host/path)
    let after_at = after_scheme
        .split_once('@')
        .map_or(after_scheme, |(_, r)| r);

    if let Some((host_port, path)) = after_at.split_once('/') {
        // Strip port from host (github.com:443 → github.com)
        let host = host_port.split(':').next().unwrap_or(host_port);
        return format!("{}/{}", host.to_lowercase(), normalize_path(path));
    }

    // No path component — fall back to local name
    format!("local/{}", sanitize_local(s))
}

/// Derive corpus for a local-only repo (no git remote): `local/<basename>`.
///
/// Non-alphanumeric characters (except `-` and `_`) are replaced by `-`.
/// Cross-repo `Exports` edges are impossible for local corpora by definition.
pub fn canonical_corpus_local(repo_root: &Path) -> String {
    let basename = repo_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    format!("local/{}", sanitize_local(basename))
}

fn normalize_path(path: &str) -> String {
    // Lowercase first so .trim_end_matches(".git") also catches ".GIT".
    let lower = path.to_lowercase();
    lower
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .to_string()
}

fn sanitize_local(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .to_lowercase()
}

// ── Language ──────────────────────────────────────────────────────────────────

/// Source language of a graph node.
///
/// Used by the indexer dispatcher ([`Language::from_extension`]) and stored on
/// the `nodes.language` column. The `#[non_exhaustive]` attribute prevents
/// external crates from writing exhaustive matches — Phase 4 will add `Go`
/// without a breaking change. **Within the travsr workspace** the compiler
/// still enforces exhaustive matches, so adding a variant is a compile-time
/// forcing function that updates every dispatch site.
///
/// See RFC-003 and ADR-005 for the design rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Language {
    TypeScript,
    Rust,
    Python,
    Go,
    Java,
    Kotlin,
    Ruby,
    CSharp,
    Php,
    Scala,
    Cpp,
    C,
    Swift,
    Dart,
    ObjectiveC,
    // Data/config formats — Phase A only, never Phase B.
    Json,
    Yaml,
    Toml,
    Xml,
    // Prose — Phase A only, never Phase B (#376). No Tree-sitter grammar and
    // no call-site semantics, so a Phase B sidecar has nothing to contribute.
    Markdown,
}

impl Language {
    /// Map a file extension to a `Language`.
    ///
    /// Returns `None` for unrecognised extensions — callers skip those files.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs" => Some(Self::TypeScript),
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "kt" | "kts" => Some(Self::Kotlin),
            "rb" | "rake" | "gemspec" => Some(Self::Ruby),
            "cs" => Some(Self::CSharp),
            "php" | "phtml" | "php8" => Some(Self::Php),
            "scala" | "sc" => Some(Self::Scala),
            "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some(Self::Cpp),
            "c" | "h" => Some(Self::C),
            "swift" => Some(Self::Swift),
            "dart" => Some(Self::Dart),
            "m" | "mm" => Some(Self::ObjectiveC),
            "json" | "jsonc" => Some(Self::Json),
            "yml" | "yaml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "xml" | "xsd" | "xsl" => Some(Self::Xml),
            "md" | "markdown" | "mdx" => Some(Self::Markdown),
            _ => None,
        }
    }

    /// Human-readable string stored in the `nodes.language` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TypeScript => "typescript",
            Self::Rust => "rust",
            Self::Python => "python",
            Self::Go => "go",
            Self::Java => "java",
            Self::Kotlin => "kotlin",
            Self::Ruby => "ruby",
            Self::CSharp => "csharp",
            Self::Php => "php",
            Self::Scala => "scala",
            Self::Cpp => "cpp",
            Self::C => "c",
            Self::Swift => "swift",
            Self::Dart => "dart",
            Self::ObjectiveC => "objectivec",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Toml => "toml",
            Self::Xml => "xml",
            Self::Markdown => "markdown",
        }
    }

    /// Parse from the storage string produced by [`Language::as_str`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "typescript" => Some(Self::TypeScript),
            "rust" => Some(Self::Rust),
            "python" => Some(Self::Python),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "kotlin" => Some(Self::Kotlin),
            "ruby" => Some(Self::Ruby),
            "csharp" => Some(Self::CSharp),
            "php" => Some(Self::Php),
            "scala" => Some(Self::Scala),
            "cpp" => Some(Self::Cpp),
            "c" => Some(Self::C),
            "swift" => Some(Self::Swift),
            "dart" => Some(Self::Dart),
            // "objc" alias is accepted at the protocol layer (language_map.rs in
            // travsr-plugin-protocol) for external API callers; core uses the
            // canonical form only.
            "objectivec" => Some(Self::ObjectiveC),
            "json" => Some(Self::Json),
            "yaml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "xml" => Some(Self::Xml),
            "markdown" => Some(Self::Markdown),
            _ => None,
        }
    }

    /// Returns `true` for data/config formats that index in Phase A only.
    ///
    /// These variants have no Phase B tool (`travsr lang install` / `CATALOG`),
    /// so they must be excluded from the Phase B `present_languages` set and
    /// from the `language_as_str_covered_by_catalog` coverage gate.
    pub fn is_data_format(self) -> bool {
        matches!(self, Self::Json | Self::Yaml | Self::Toml | Self::Xml)
    }

    /// Returns `true` for every variant that has no Phase B tool at all —
    /// data/config formats plus prose (#376's [`Self::Markdown`]).
    ///
    /// Superset of [`Self::is_data_format`]. Phase B dispatch, the
    /// `present_languages` coverage gate, and the embed-catalog coverage test
    /// all key off this rather than `is_data_format` so that adding a future
    /// Phase-A-only variant only requires touching this one match arm.
    pub fn is_phase_a_only(self) -> bool {
        self.is_data_format() || matches!(self, Self::Markdown)
    }
}

/// True when `file_name` is a dependency manifest that `data_format::parse`
/// handles but whose extension is unmapped or absent, so the extension-based
/// [`Language::from_extension`] gates would otherwise skip the file entirely.
///
/// Recognized by canonical basename (manifests have fixed names). This is the
/// single source of truth for "is this an extensionless/odd-extension manifest";
/// every file-enumeration gate (daemon walk, watcher, CLI walk) and the indexer
/// dispatch route through it via [`is_indexable_path`], so the recognizer set
/// can never drift between call sites.
///
/// Manifests whose extension IS mapped (`package.json`, `Cargo.toml`,
/// `pyproject.toml`, `composer.json`, `pubspec.yaml`) are already admitted by
/// `from_extension`; they are dispatched by name inside `data_format::parse` and
/// do NOT need to be listed here.
pub fn is_manifest_file(file_name: &str) -> bool {
    file_name == "go.mod" || file_name.ends_with(".csproj")
}

/// True when `path` should be indexed at all — either it has a recognized
/// source/data-format extension ([`Language::from_extension`]) or it is a
/// name-recognized manifest ([`is_manifest_file`]). Every enumeration gate calls
/// this so the "what is indexable" decision lives in exactly one place.
pub fn is_indexable_path(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if Language::from_extension(ext).is_some() {
        return true;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    is_manifest_file(name)
}

/// Canonical list of every [`Language`] variant.
///
/// `Language` is `#[non_exhaustive]`, so the compiler cannot enforce that a
/// downstream `for lang in ALL_LANGUAGES` sees every variant. Keep this slice in
/// sync with the enum — the test-role coverage gate (issue #479 §7.2) and any
/// future per-language matrix iterate it, so adding a variant here is the
/// forcing function that makes those gates flag the new language.
pub const ALL_LANGUAGES: &[Language] = &[
    Language::TypeScript,
    Language::Rust,
    Language::Python,
    Language::Go,
    Language::Java,
    Language::Kotlin,
    Language::Ruby,
    Language::CSharp,
    Language::Php,
    Language::Scala,
    Language::Cpp,
    Language::C,
    Language::Swift,
    Language::Dart,
    Language::ObjectiveC,
    Language::Json,
    Language::Yaml,
    Language::Toml,
    Language::Xml,
    Language::Markdown,
];

/// Kythe-style globally unique identifier for a code entity.
///
/// VNames are stable across repos, languages, and time — they form the
/// universal address space of the Travsr graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VName {
    /// Logical corpus (e.g. repo URL or org/project).
    pub corpus: String,
    /// Root within the corpus (e.g. branch or build root).
    pub root: String,
    /// Path within the root (e.g. `src/foo.ts`).
    pub path: String,
    /// Source language identifier (e.g. `typescript`, `rust`).
    pub language: String,
    /// Symbol signature within the file (e.g. `class:PaymentService#charge`).
    pub signature: String,
}

impl VName {
    /// Construct a `VName` from its five components.
    pub fn new(
        corpus: impl Into<String>,
        root: impl Into<String>,
        path: impl Into<String>,
        language: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        Self {
            corpus: corpus.into(),
            root: root.into(),
            path: path.into(),
            language: language.into(),
            signature: signature.into(),
        }
    }

    /// Stable 64-bit identifier derived from the five-field VName.
    ///
    /// The hash is the first 8 bytes of the BLAKE3 digest of:
    ///   `[SIGNATURE_FORMAT_VERSION] || [len_u32_le][corpus] || [len_u32_le][root] || ...`
    ///
    /// Length-prefix encoding (4-byte little-endian field length before each
    /// field) replaces the NUL-separator scheme. This guarantees that no two
    /// distinct VNames share the same byte stream, and that a v0 byte stream
    /// (which starts with raw corpus bytes) can never equal a v1 stream (which
    /// starts with `[version_byte][len]`). See RFC-002.
    pub fn id(&self) -> NodeId {
        let mut hasher = blake3::Hasher::new();
        // Version domain separator — must be first. Changing SIGNATURE_FORMAT_VERSION
        // produces disjoint NodeId spaces; see RFC-002.
        hasher.update(&[SIGNATURE_FORMAT_VERSION]);
        // Length-prefix each field so no two distinct VNames share the same byte
        // stream regardless of field contents (no NUL-injection ambiguity).
        for field in [
            self.corpus.as_str(),
            self.root.as_str(),
            self.path.as_str(),
            self.language.as_str(),
            self.signature.as_str(),
        ] {
            let bytes = field.as_bytes();
            hasher.update(&(bytes.len() as u32).to_le_bytes());
            hasher.update(bytes);
        }
        let digest = hasher.finalize();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&digest.as_bytes()[..8]);
        NodeId(u64::from_le_bytes(buf))
    }
}

/// Opaque, content-addressed identifier for a node in the graph.
///
/// `NodeId` is a stable BLAKE3-derived hash of a `VName` (see
/// [`VName::id`]). It is the SQLite primary key for the `nodes` table.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct NodeId(pub u64);

/// The kinds of edges supported in the Travsr multiplex graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// File / module import. Corresponds to Kythe `%kythe/edge/depends`.
    #[serde(rename = "depends")]
    Depends,
    /// Call-site reference. Corresponds to Kythe `%kythe/edge/ref/call`.
    #[serde(rename = "ref/call")]
    RefCall,
    /// Field-access reference: a read or write of a struct/class field
    /// (`x.foo`), as distinct from a call. Kept separate from `RefCall` so a
    /// field read never appears as a caller in `get_callers` / `get_blast_radius`
    /// while still surfacing as a use-site in `find_references` (#757).
    #[serde(rename = "ref/field")]
    RefField,
    /// Definition-binding edge (parent → child in the AST).
    #[serde(rename = "defines/binding")]
    DefinesBinding,
    /// A symbol exported from a module.
    #[serde(rename = "exports")]
    Exports,
    /// An import node resolved to the file node it targets.
    /// Connects `import:./foo` → `file:foo.ts`, enabling transitive
    /// caller traversal across file boundaries.
    #[serde(rename = "resolves-to")]
    ResolvesTo,
    /// Named import specifier reference emitted by the LSIF pipeline.
    /// Distinguishes semantic import references from file-level `Depends` edges.
    #[serde(rename = "ref/imports")]
    RefImports,
    /// Class-to-interface implementation edge emitted by the LSIF pipeline.
    #[serde(rename = "is-implementation")]
    IsImplementation,
    /// Method override edge emitted by the LSIF pipeline when a subclass
    /// method shadows a same-named method in the base class.
    #[serde(rename = "overrides")]
    Overrides,
    /// Cross-language FFI call edge (RFC-005). Confidence lives on `Edge.confidence`
    /// so `EdgeKind` stays `Copy`. PPR weight: 0.85 (ADR-003 amendment, 2026-05-24).
    #[serde(rename = "ffi/call")]
    FFICall,
    /// Config file → source file or target it configures.
    /// E.g. `tsconfig.json` references a sub-project, `docker-compose.yml` depends_on a service.
    #[serde(rename = "configures")]
    Configures,
    /// Config file → external package node (registry-hosted dependency).
    /// E.g. `package.json` dependency, `Cargo.toml` crate, `pom.xml` artifact.
    #[serde(rename = "external-dependency")]
    ExternalDependency,
}

impl EdgeKind {
    /// Stable string representation used as the storage key.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Depends => "depends",
            Self::RefCall => "ref/call",
            Self::RefField => "ref/field",
            Self::DefinesBinding => "defines/binding",
            Self::Exports => "exports",
            Self::ResolvesTo => "resolves-to",
            Self::RefImports => "ref/imports",
            Self::IsImplementation => "is-implementation",
            Self::Overrides => "overrides",
            Self::FFICall => "ffi/call",
            Self::Configures => "configures",
            Self::ExternalDependency => "external-dependency",
        }
    }

    /// PPR transition weight for this edge kind.
    ///
    /// Weights encode the semantic importance of each edge type for
    /// Personalized PageRank: a higher weight means PPR mass flows more
    /// readily across edges of this kind, producing higher scores for
    /// reachable nodes.
    ///
    /// # Rationale (DEBT-016 / ADR-003)
    ///
    /// | Kind              | Weight | Reasoning                               |
    /// |---|---|---|
    /// | Kind                | Weight | Reasoning                               |
    /// |---|---|---|
    /// | `RefCall`           | 1.00   | Direct call — strongest semantic link   |
    /// | `FFICall`           | 0.85   | Cross-language call — near-call strength|
    /// | `DefinesBinding`    | 0.70   | Parent→child definition — strong structural link |
    /// | `Exports`           | 0.60   | Exported API surface — important for callers |
    /// | `Depends`           | 0.50   | File import — broad but less targeted   |
    /// | `ResolvesTo`        | 0.50   | Import→file resolution — same as Depends |
    /// | `RefImports`        | 0.40   | Named import specifier — narrower than file import |
    /// | `IsImplementation`  | 0.40   | Class implements interface — type-system link |
    /// | `Overrides`         | 0.30   | Method override — weakest semantic tie  |
    /// | `Configures`        | 0.35   | Config→target — below structural, above override |
    /// | `ExternalDependency`| 0.30   | Config→registry package — declaration, not usage |
    ///
    /// Weights are normalised per-node at PPR iteration time so their
    /// absolute scale does not matter — only the ratios between kinds.
    pub fn ppr_weight(self) -> f32 {
        match self {
            Self::RefCall => 1.00,
            // A field read is a genuine data-usage link but weaker than a call;
            // sits alongside a file import in signal strength (#757).
            Self::RefField => 0.50,
            Self::DefinesBinding => 0.70,
            Self::Exports => 0.60,
            Self::Depends => 0.50,
            Self::ResolvesTo => 0.50,
            Self::RefImports => 0.40,
            Self::IsImplementation => 0.40,
            Self::Overrides => 0.30,
            Self::FFICall => 0.85,
            Self::Configures => 0.35,
            Self::ExternalDependency => 0.30,
        }
    }

    /// Parse from the stable string representation.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "depends" => Some(Self::Depends),
            "ref/call" => Some(Self::RefCall),
            "ref/field" => Some(Self::RefField),
            "defines/binding" => Some(Self::DefinesBinding),
            "exports" => Some(Self::Exports),
            "resolves-to" => Some(Self::ResolvesTo),
            "ref/imports" => Some(Self::RefImports),
            "is-implementation" => Some(Self::IsImplementation),
            "overrides" => Some(Self::Overrides),
            "ffi/call" => Some(Self::FFICall),
            "configures" => Some(Self::Configures),
            "external-dependency" => Some(Self::ExternalDependency),
            _ => None,
        }
    }
}

/// Index-time classification of a node as test code (issue #479).
///
/// Derived from tree-sitter captures (`@test.entry` / `@test.scope`) during
/// Phase A parsing. It is **metadata, not identity**: two nodes are the same symbol
/// regardless of `test_role`, so it is **not** part of the BLAKE3 VName id — it
/// sits alongside `package`/`line`, exactly like `is_noise`.
///
/// Retrieval uses it to bucket test declarations into a capped `tests` section
/// below the implementation sections, instead of letting a `#[test]` fn take the
/// top slot of the `exact`/`semantic` groups (the #479 defect).
///
/// The asymmetric-cost rule (a false positive removes a real answer from the
/// section the host reads first) is enforced upstream in the tree-sitter query
/// predicates, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TestRole {
    /// Not test code (production code, or a test-ish name with no corroboration).
    #[default]
    None,
    /// A test entry point — a `#[test]` fn, a `@Test`-annotated method, a
    /// `func TestX` with a `*testing.T` param, etc. The thing a runner invokes.
    EntryPoint,
    /// Support code that lives inside a test unit — a helper fn or fixture inside
    /// a `#[cfg(test)] mod`, a method of a `TestCase` subclass, etc. Detected from
    /// AST `@test.scope` captures only (the path-based fallback was Phase 2, which
    /// regressed the k8s bench and was reverted).
    Support,
}

impl TestRole {
    /// Stable integer representation for the `nodes.test_role` column (v22).
    ///
    /// `None = 0` so the `INTEGER NOT NULL DEFAULT 0` column and the serde
    /// default agree — un-reindexed rows read back as [`TestRole::None`].
    pub fn as_i64(self) -> i64 {
        match self {
            Self::None => 0,
            Self::EntryPoint => 1,
            Self::Support => 2,
        }
    }

    /// Parse from the stored integer. Unknown values fail closed to `None`
    /// (a forward-compatible store row never mislabels code as a test).
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => Self::EntryPoint,
            2 => Self::Support,
            _ => Self::None,
        }
    }

    /// True for any node classified as test code (entry point or support).
    pub fn is_test(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// A node in the code graph.
///
/// `PartialEq` compares all fields including `package`. Use `node.id == other.id`
/// for identity-only comparisons (two nodes are the same symbol regardless of
/// their package annotation).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub vname: VName,
    pub kind: String,
    /// Sub-unit identity within the corpus (ADR-005 Rule 2).
    ///
    /// Stored in `nodes.package`; **not** part of the BLAKE3 hash input.
    /// Empty string for nodes where package identity is unknown or irrelevant.
    ///
    /// | Language   | Value                                       |
    /// |------------|---------------------------------------------|
    /// | TypeScript | npm package name from `package.json`        |
    /// | Rust       | Cargo package name from `Cargo.toml`        |
    /// | Python     | top-level package dir (highest `__init__.py`)|
    pub package: String,
    /// 1-based source line of the symbol's definition site.
    /// `None` for file-kind nodes and synthetic import nodes.
    pub line: Option<u32>,
    /// 1-based last source line of the symbol's body (inclusive).
    /// Used by G2 span attribution to find the enclosing function for a SCIP
    /// reference occurrence. `None` until migration v13 backfills the column.
    pub end_line: Option<u32>,
    /// Index-time test classification (issue #479). Defaults to
    /// [`TestRole::None`]; set by the `travsr-analysis` post-pass from
    /// tree-sitter `@test.entry` / `@test.scope` captures. Stored in
    /// `nodes.test_role` (v22); **not** part of the BLAKE3 id.
    ///
    /// `#[serde(default)]` keeps old plugin-protocol payloads (out-of-process
    /// sidecars, RFC-013 plugins) valid — a missing field deserializes to
    /// `None`, the default-safe value.
    #[serde(default)]
    pub test_role: TestRole,
}

impl Node {
    /// Build a `Node` from a `VName` and a free-form kind string.
    ///
    /// The `id` is derived deterministically from the VName. `package`
    /// defaults to an empty string; use [`Node::with_package`] to set it.
    /// `line` / `end_line` default to `None`; use the builder methods to set them.
    pub fn new(vname: VName, kind: impl Into<String>) -> Self {
        let id = vname.id();
        Self {
            id,
            vname,
            kind: kind.into(),
            package: String::new(),
            line: None,
            end_line: None,
            test_role: TestRole::None,
        }
    }

    /// Set the `package` field and return `self` (builder pattern).
    ///
    /// ```
    /// use travsr_core::{Node, VName};
    /// let n = Node::new(VName::new("github.com/a/b", "", "src/lib.rs", "rust", "fn:main"), "function")
    ///     .with_package("my-crate");
    /// assert_eq!(n.package, "my-crate");
    /// ```
    pub fn with_package(mut self, package: impl Into<String>) -> Self {
        self.package = package.into();
        self
    }

    /// Set the `line` field (1-based) and return `self` (builder pattern).
    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Set the `end_line` field (1-based, inclusive) and return `self` (builder pattern).
    pub fn with_end_line(mut self, end_line: u32) -> Self {
        self.end_line = Some(end_line);
        self
    }

    /// Set the `test_role` field and return `self` (builder pattern).
    ///
    /// Used by the `travsr-analysis` post-pass so the ~10 `emit.rs` constructors
    /// stay one-liners and only test declarations pay the extra call (issue #479).
    pub fn with_test_role(mut self, test_role: TestRole) -> Self {
        self.test_role = test_role;
        self
    }
}

/// A raw SCIP reference occurrence used for G2 call-site attribution.
///
/// The store's `write_scip_attributed_batch` takes a slice of these and emits
/// `ref/call` edges from the enclosing function node (or the file node as fallback)
/// to `callee_id`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScipRef {
    /// Repo-relative path of the file containing the reference (e.g. `pkg/foo/bar.go`).
    pub caller_path: String,
    /// 1-based source line of the reference occurrence.
    pub caller_line: u32,
    /// `NodeId` of the called symbol (from `symbol_map` or `symbol_aliases`).
    pub callee_id: NodeId,
    /// Whether this occurrence is an actual **call** (vs a non-call reference:
    /// type annotation, `self`/`Self`, path segment, field access, bare read).
    ///
    /// Every occurrence — call or not — is recorded in `edge_sites` so
    /// `find_references` enumerates all use sites. Only calls (`is_call == true`)
    /// create a `ref/call` **edge**, keeping `get_callers` / blast-radius /
    /// PageRank a genuine call graph and never a `src == dst` self-loop (#650).
    ///
    /// `#[serde(default)]` is `true`: producers whose occurrences are already
    /// call-scoped by construction (native tree-sitter call-site queries, our
    /// bundled emitters, older sidecars that predate this field) get the prior
    /// edge-emitting behavior with no regression.
    #[serde(default = "default_true")]
    pub is_call: bool,
    /// 0-based UTF-8 BYTE column of the reference occurrence on `caller_line`
    /// (RFC-027 #813 P2). Recorded on the `edge_sites` row so the live overlay
    /// resolves the reference at its exact editor position instead of searching
    /// the line for the name. Each source (SCIP sidecar, rust-analyzer LSIF,
    /// Dart emitter) converts its own occurrence unit to a byte offset before
    /// building this; the daemon converts byte to the editor's UTF-16 column at
    /// use. `None` when the source cannot give a reliable position, in which case
    /// the daemon falls back to its word-boundary search (no regression).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_col: Option<u32>,
}

/// serde default for [`ScipRef::is_call`] / [`LsifPositionalRef::is_call`].
fn default_true() -> bool {
    true
}

/// A positional reference occurrence from a rust-analyzer LSIF dump (E3 W3b).
///
/// Unlike [`ScipRef`], the callee is identified by its **definition location**
/// (`callee_def_path`, `callee_def_line`) rather than a pre-resolved `NodeId`,
/// because rust-analyzer LSIF carries no `travsr_vname` — only monikers and
/// definition ranges. The store resolves the callee positionally (narrowest
/// node span containing the def line) and fails closed when nothing resolves,
/// turning these into `ScipRef`s. This is what replaces the old moniker-synth
/// path whose callee VName (at `path = project_root`) matched no Phase A node.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LsifPositionalRef {
    /// Repo-relative path of the file containing the reference occurrence.
    pub caller_path: String,
    /// 1-based source line of the reference occurrence.
    pub caller_line: u32,
    /// Repo-relative path of the file containing the callee's definition.
    pub callee_def_path: String,
    /// 1-based source line of the callee's definition occurrence.
    pub callee_def_line: u32,
    /// Whether this occurrence is an actual call (see [`ScipRef::is_call`]). The
    /// store carries it through to the resolved `ScipRef` so non-call references
    /// record a `find_references` occurrence without creating a call edge (#650).
    #[serde(default = "default_true")]
    pub is_call: bool,
    /// 0-based UTF-8 BYTE column of the reference occurrence on `caller_line`
    /// (RFC-027 #813 P2). LSIF/LSP ranges are UTF-16 code units, so the indexer
    /// converts the range's start character to a byte offset against the caller
    /// line before setting this. Carried through to the resolved [`ScipRef`] and
    /// recorded on the `edge_sites` row. `None` for a dump built before this
    /// shipped, in which case the daemon name-searches the line (no regression).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_col: Option<u32>,
}

/// A single reference occurrence returned by `find_references` (issue #299):
/// the file path and 1-based line of one use site of a symbol.
///
/// Produced by `SqliteStore::reference_sites` from the `edge_sites` occurrence
/// store. Language-agnostic — every ingestion path (SCIP, LSIF, native
/// tree-sitter) feeds the same table, so one struct describes them all.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RefSite {
    /// Repo-relative path of the file containing the reference occurrence.
    pub path: String,
    /// 1-based source line of the occurrence.
    pub line: u32,
    /// True when at least one `ref/call` edge behind this site carries
    /// `provenance = 'tree-sitter'`, i.e. it was matched by leaf name rather
    /// than resolved by a compiler. Such a site can be wholly fabricated: a
    /// local binding Phase A does not model leaves the only same-named node in
    /// an unrelated file as the unique winner, and every occurrence enumerated
    /// under it points at the wrong symbol. Renderers mark these; a `false`
    /// means either compiler-resolved or (for an occurrence with no edge row of
    /// its own, e.g. a SCIP type reference) nothing to flag.
    ///
    /// `#[serde(default)]` so a payload written before this field existed still
    /// deserializes, reading as "nothing to flag" exactly as it did then.
    #[serde(default)]
    pub heuristic: bool,
}

/// Human-readable label for a node in `graph` / reference output.
///
/// - File nodes (signature is the literal `"file"`): the repo-relative path.
/// - Nodes synthesized from a raw compiler symbol (a reference whose target has
///   no tree-sitter node — an implicit constructor, a cross-file external
///   symbol): a clean `Type.member` name, matching the `Type.member` style
///   native tree-sitter nodes already use (see [`external_symbol_label`]).
/// - Everything else: the signature verbatim (native `kind:name` form).
pub fn display_label(node: &Node) -> Cow<'_, str> {
    if node.kind == "file" && !node.vname.path.is_empty() {
        return Cow::Borrowed(&node.vname.path);
    }
    display_signature(&node.vname.signature, &node.vname.path)
}

/// Clean a raw node signature for display, given the node's path.
///
/// This is the signature-level entry point behind [`display_label`], for the
/// call sites that hold a bare signature string (the `graph --format json` /
/// DOT renderers) rather than a whole [`Node`]. Non-synthesized signatures are
/// returned borrowed and unchanged.
pub fn display_signature<'a>(signature: &'a str, path: &str) -> Cow<'a, str> {
    match external_symbol_label(signature, path) {
        Some(clean) => Cow::Owned(clean),
        None => Cow::Borrowed(signature),
    }
}

/// Turn a synthesized external-symbol signature into a clean `Type.member`
/// name, or return `None` for an ordinary node.
///
/// When a Phase B reference points at a symbol that has no tree-sitter node,
/// the analyzer synthesizes a node named by the raw compiler symbol so the edge
/// has a destination. Those raw names carry indexer-internal prefixes and the
/// full package path (`Greeter#`<init>`().`, `com/demo/Main.g.`) that must never
/// reach a user. This normalizes them to the same short, package-free
/// `Type.member` form a resolved node would show.
///
/// Two synthesis paths use two prefixes for the same concept: scip-reader emits
/// `scip:{path}:{scip-symbol}` (go, java, c#, c/c++, objc), the scala reader
/// emits `sdb:{descriptors}`. Everything else (native tree-sitter, kotlin) is
/// already a clean `kind:name` signature and returns `None` here.
fn external_symbol_label(signature: &str, path: &str) -> Option<String> {
    let descriptors = if let Some(rest) = signature.strip_prefix("scip:") {
        // `{path}:{scip-symbol}`. Prefer stripping the node's own path (robust to
        // spaces in the path). When the path is empty or does not prefix-match —
        // the `graph` edge-endpoint renderers call `display_signature(sig, "")`
        // with no path in hand — fall back to the last `:`: a SCIP global symbol
        // contains no colon and a repo-relative path does not either, so the tail
        // after the final colon is the scip symbol even when the path has spaces.
        let scip_symbol = rest
            .strip_prefix(path)
            .and_then(|r| r.strip_prefix(':'))
            .or_else(|| rest.rsplit_once(':').map(|(_, sym)| sym))
            .unwrap_or(rest);
        // An anonymous SCIP local (`local <id>`) has no type/member structure, so
        // `scip_descriptor_part` cannot parse it and it would otherwise fall
        // through raw. Drop only the `scip:{path}:` noise and keep `local <id>`.
        if let Some(id) = scip_symbol.strip_prefix("local ") {
            return Some(format!("local {}", id.trim()));
        }
        scip_descriptor_part(scip_symbol)?
    } else {
        // SemanticDB global symbols are descriptors only (no scheme prefix);
        // a signature without either prefix is already clean.
        signature.strip_prefix("sdb:")?
    };
    descriptors_to_label(descriptors)
}

/// The descriptor tail of a SCIP global symbol, which is
/// `<scheme> <manager> <name> <version> <descriptors>` — four whitespace-
/// separated leading fields, then the descriptors (which may themselves contain
/// spaces inside backtick-escaped names, so split off exactly four fields rather
/// than taking the last token).
fn scip_descriptor_part(scip_symbol: &str) -> Option<&str> {
    let mut rest = scip_symbol;
    for _ in 0..4 {
        rest = rest.split_once(char::is_whitespace)?.1;
    }
    Some(rest)
}

/// Join the type/member names of a SCIP / SemanticDB descriptor string into a
/// package-free `Type.member` label. Package segments are dropped; an implicit
/// constructor (`<init>` / `<clinit>`) collapses onto its enclosing type.
fn descriptors_to_label(descriptors: &str) -> Option<String> {
    let parts = descriptor_names(descriptors);
    // Drop package descriptors when anything more specific (a type or member)
    // remains; a pure-package symbol keeps its segments so it is not blanked.
    let mut kept: Vec<&str> = parts
        .iter()
        .filter(|(_, term)| *term != '/')
        .map(|(name, _)| name.as_str())
        .collect();
    if kept.is_empty() {
        kept = parts.iter().map(|(name, _)| name.as_str()).collect();
    }
    // `<init>`/`<clinit>` are compiler-internal constructor names; the enclosing
    // type name already identifies the constructor (the node kind says so too).
    kept.retain(|name| *name != "<init>" && *name != "<clinit>" && !name.is_empty());
    if kept.is_empty() {
        return None;
    }
    Some(kept.join("."))
}

/// Split a SCIP / SemanticDB descriptor string into `(name, terminator)` pairs.
///
/// Descriptor grammar: a package ends in `/`, a type in `#`, a term in `.`, a
/// method in `(disambiguator).`, a parameter is `(name)`, a type parameter is
/// `[name]`. Names may be backtick-escaped (`` `<init>` ``), in which case a
/// doubled backtick is a literal backtick.
fn descriptor_names(descriptors: &str) -> Vec<(String, char)> {
    let mut out = Vec::new();
    let mut chars = descriptors.chars().peekable();
    while let Some(&c) = chars.peek() {
        // Parameter `(name)` / type parameter `[name]`: a bracketed name with no
        // preceding identifier.
        if c == '(' || c == '[' {
            let close = if c == '(' { ')' } else { ']' };
            chars.next();
            let mut name = String::new();
            for ch in chars.by_ref() {
                if ch == close {
                    break;
                }
                name.push(ch);
            }
            out.push((name, close));
            continue;
        }

        // A (possibly backtick-escaped) name, then its terminator.
        let mut name = String::new();
        if c == '`' {
            chars.next();
            while let Some(ch) = chars.next() {
                if ch == '`' {
                    if chars.peek() == Some(&'`') {
                        chars.next();
                        name.push('`');
                    } else {
                        break;
                    }
                } else {
                    name.push(ch);
                }
            }
        } else {
            while let Some(&ch) = chars.peek() {
                if matches!(ch, '/' | '#' | '.' | '(' | '[') {
                    break;
                }
                name.push(ch);
                chars.next();
            }
        }

        match chars.peek().copied() {
            Some('/') => {
                chars.next();
                out.push((name, '/'));
            }
            Some('#') => {
                chars.next();
                out.push((name, '#'));
            }
            Some('.') => {
                chars.next();
                out.push((name, '.'));
            }
            Some('(') => {
                // Method disambiguator `(...)` followed by `.`: skip the
                // parenthesized part, then the terminating dot.
                chars.next();
                let mut depth = 1;
                for ch in chars.by_ref() {
                    match ch {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if chars.peek() == Some(&'.') {
                    chars.next();
                }
                out.push((name, '.'));
            }
            _ => {
                if !name.is_empty() {
                    out.push((name, '.'));
                }
                break;
            }
        }
    }
    out
}

/// Returns `true` for nodes that carry no developer-facing signal:
/// vendored paths, OS/build caches, repo-escaping paths, and SCIP anonymous locals.
///
/// This is the single source of truth for structural noise. `is_noise_seed`
/// (travsr-mcp) calls this first and then adds MCP-layer policy on top.
pub fn is_noise_node(node: &Node) -> bool {
    let p = &node.vname.path;
    // Vendored / package-manager directories.
    if p.starts_with("third_party/")
        || p.starts_with("vendor/")
        || p.starts_with("node_modules/")
        || p.contains("/node_modules/")
    {
        return true;
    }
    // Paths that escape the repo root are never real source nodes
    // (e.g. Go build-cache: `../../../Library/Caches/go-build/…`).
    if p.starts_with("../") {
        return true;
    }
    // OS-level build and package caches.
    if p.contains("/Library/Caches/")
        || p.contains("/.cache/")
        || p.contains("/go-build/")
        || p.contains("/go/pkg/mod/")
    {
        return true;
    }
    is_scip_anonymous_local(&node.vname.signature)
}

/// Returns `true` if `sig` is a SCIP anonymous-local symbol (`local N` suffix).
/// Used by G3 ingest filter in `travsr-indexer` and by `is_noise_node`.
pub fn is_scip_anonymous_local(sig: &str) -> bool {
    // SCIP local symbols end with "local <digits>", e.g. "local 27".
    if let Some(pos) = sig.rfind("local ") {
        let suffix = &sig[pos + 6..];
        !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}

/// A directed, typed edge between two nodes.
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: EdgeKind,
    /// Confidence score 0..=100 for cross-language FFI edges (RFC-005).
    /// `None` for all non-FFI edges. Stored in `edges.confidence` (migration v6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<u8>,
    /// How this edge was derived: the `edges.provenance` tag required by
    /// ADR-002 Rule 1 (`tree-sitter` / `lsif` / `scip` / `bridge:<mech>`).
    ///
    /// This is a **read-side** field, populated by the store's `iter_edges_*`
    /// readers so consumers (the MCP surface, `travsr graph --format json`) can
    /// report an edge's true origin instead of assuming `tree-sitter`
    /// (DEBT-75). It is `None` on an edge that was constructed rather than read.
    /// Writers are unaffected: every insert path still takes its provenance as
    /// an explicit argument, so there is exactly one source of truth on write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// Equality is over `(src, dst, kind, confidence)` and deliberately **excludes**
/// `provenance`: an edge's identity in the store is its `(src, dst, kind)`
/// primary key, and provenance is metadata about how that one edge was derived,
/// not a second edge. Two `Edge` values that differ only in provenance denote
/// the same edge, so comparing a constructed edge against a read-back one stays
/// meaningful.
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src
            && self.dst == other.dst
            && self.kind == other.kind
            && self.confidence == other.confidence
    }
}

impl Edge {
    pub fn new(src: NodeId, dst: NodeId, kind: EdgeKind) -> Self {
        Self {
            src,
            dst,
            kind,
            confidence: None,
            provenance: None,
        }
    }

    /// Attach a read-side provenance tag. Used by the store readers; see the
    /// `provenance` field docs.
    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }

    /// Build a cross-language FFI edge with a confidence score (RFC-005).
    ///
    /// `confidence` must be in `0..=100`. Panics in debug builds if violated.
    pub fn ffi_call(src: NodeId, dst: NodeId, confidence: u8) -> Self {
        debug_assert!(
            confidence <= 100,
            "confidence must be 0..=100, got {confidence}"
        );
        Self {
            src,
            dst,
            kind: EdgeKind::FFICall,
            confidence: Some(confidence),
            provenance: None,
        }
    }
}

/// A cross-crate call that Phase B could not resolve to a concrete NodeId.
///
/// Phase B tree-sitter extraction knows the callee name but not its file path
/// (the file path is part of the VName hash). The daemon resolves these after
/// Phase B completes by querying Phase A nodes already in the store.
///
/// Propagated through `InvokeResponse.unresolved_calls`; resolved in the daemon
/// before writing to the graph (see `resolve_unresolved_calls` in travsr-daemon).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UnresolvedCall {
    /// The caller node ID (already resolved — it's in the same file as the call site).
    pub src: NodeId,
    /// Signature of the callee to resolve, e.g. `"fn:ppr_weighted"`.
    pub callee_sig: String,
    /// A second exact signature to try when `callee_sig` misses, for a call
    /// site whose shape is genuinely ambiguous between two node kinds.
    ///
    /// Python's bare `Foo()` is the motivating case (#709): PEP 8 says an
    /// uppercase initial means a class, so it is emitted as `class:Foo`, but a
    /// PascalCase free function (`def Field(...)`) is legal and common. The
    /// extractor cannot tell them apart, and `travsr-analysis` may not consult
    /// the store to find out, so it names both and lets the resolver decide on
    /// evidence.
    ///
    /// Tried **exactly**, never through the leaf pool: the point is to find the
    /// other real definition, not to widen the net (#716 review).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alt_callee_sig: Option<String>,
    /// Last path segment of the qualifying crate/module for `call.scoped`, e.g.
    /// `"travsr_retrieval"` from `travsr_retrieval::ppr_weighted(...)`.
    /// `None` for bare `call.fn` identifiers — resolver uses name-only lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint_crate: Option<String>,
    /// 1-based source line of the call-site occurrence (issue #299). Recorded so
    /// the daemon can emit an `edge_sites` row when it resolves `callee_sig` to a
    /// concrete node, giving `find_references` occurrence-level `path:line` for
    /// cross-crate bare calls. `0` means "unknown" (older extractors); the daemon
    /// skips `edge_sites` emission for zero lines.
    #[serde(default)]
    pub caller_line: u32,
    /// 0-based byte column of the callee-name occurrence on `caller_line`
    /// (RFC-027 #813 P2). It is the start column of the exact identifier node the
    /// extractor captured, so it points at the reference the editor resolves,
    /// which is strictly better than a name search when the name repeats on the
    /// line. Byte offset, in the file's own encoding; the daemon converts it to
    /// the editor's UTF-16 column against the current line. `None` for an emitter
    /// that predates this field, in which case the daemon falls back to its
    /// word-boundary search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_col: Option<u32>,
    /// True when this call came from a method-call receiver (`recv.method()`)
    /// whose type is not known syntactically. A method call can never resolve
    /// to a bare free function — the daemon resolver requires a qualified
    /// `fn:Type.method` / `method:Type.method` match when this is set (#521 F3).
    /// `false` for `call.fn` / `call.scoped` and for non-Rust emitters that
    /// predate this field (serde default keeps old sidecar payloads valid).
    #[serde(default)]
    pub is_method_call: bool,
    /// Receiver type for a method call (`recv.method()`), when the extractor
    /// could recover it from the enclosing function's own text (#529). `Some(T)`
    /// lets the daemon resolve `fn:T.method` exactly instead of guessing by
    /// unique leaf name; `None` keeps the pre-#529 leaf-fallback behavior. Only
    /// ever set alongside `is_method_call`. `#[serde(default)]` keeps older
    /// sidecar payloads valid, matching how `caller_line` and `is_method_call`
    /// were introduced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recv_type: Option<String>,
}

/// An `extends`/`implements` clause the native extractor found but does not
/// resolve cross-file, for RFC-027's live `IsImplementation` lane.
///
/// Like [`UnresolvedCall`] it carries no identity — only the base type's simple
/// name and the line it is written on, which is exactly what the live lane needs
/// to resolve it (lexically against a unique repo-wide definition, or via the
/// editor's definition provider) and to abstain otherwise. The daemon derives
/// the edge's source (the implementing class) from the line and owns both node
/// ids, so nothing here mints a VName.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InheritanceRef {
    /// The base type's simple name (the edge *target*), exactly as written: the
    /// superclass in `class C extends B`, the interface in `implements I`.
    pub base_name: String,
    /// 1-based line the base name appears on — the class declaration line, which
    /// the daemon maps to the implementing class (the edge source) via
    /// `enclosing_definition_at`.
    pub line: u32,
}

// ── Import Resolution ────────────────────────────────────────────────────────
//
// Query-time bridge over missing `ResolvesTo` edges.
// When the indexer has not emitted `ResolvesTo` edges (Java, Go cross-module,
// PHP, C#, C/C++, Swift, Dart), these resolvers answer at query time:
// "does the import statement `import_sig`, written in `importer_path`, point to
//  this `target_path`?"
//
// `importer_path` (the path of the file that holds the import — the import
// node's own `VName::path`) is the deterministic anchor for *relative* imports:
// Ruby `require_relative` and C/C++ quoted `#include "…"` are resolved against
// the importing file's directory, so a bare `import:animal` can only match the
// `animal.rb` next to its importer, never a same-named file under an unrelated
// root. Languages whose imports are absolute/namespace-rooted (Go module paths,
// Java/Kotlin/Scala/C# packages, PHP PSR-4, Swift modules, Python absolute)
// carry their own path in the signature and ignore `importer_path`; their match
// is structural and deliberately over-inclusive, since classpath/module
// resolution is not available in the graph.
//
// Used by `travsr-mcp::tools::get_blast_radius_raw` (Phase 2). Determinism here
// matters because that consumer walks importers transitively — a false match
// would compound across hops.
// Lives in `travsr-core` so all downstream crates can use it without
// violating the crate dependency DAG.

pub trait ImportResolver: Send + Sync {
    fn resolves_to(&self, import_sig: &str, importer_path: &str, target_path: &str) -> bool;
}

/// Return the correct resolver for a `VName::language` string.
/// Falls back to `NoopResolver` for TypeScript, JS, and Rust — `link_imports*`
/// already emits `ResolvesTo` edges for those at index time.
pub fn resolver_for_language(language: &str) -> &'static dyn ImportResolver {
    match language {
        "go" => &GoResolver,
        "java" => &JavaResolver,
        "kotlin" => &KotlinResolver,
        "scala" => &ScalaResolver,
        "python" => &PythonResolver,
        "php" => &PhpResolver,
        "csharp" | "c#" | "cs" => &CsharpResolver,
        "cpp" | "c++" | "c" => &CppResolver,
        "swift" => &SwiftResolver,
        "dart" => &DartResolver,
        "ruby" => &RubyResolver,
        // Objective-C `#import "Foo.h"` / `#include "Foo.h"` are importer-relative
        // exactly like C/C++, so they share the same resolver. Angle-bracket
        // framework imports and `@import Module` resolve to no project file.
        // "objc" is the plugin-protocol alias; "objectivec" is core's canonical
        // storage string (see Language::as_str).
        "objectivec" | "objc" => &CppResolver,
        // TypeScript/JS/Rust: link_imports* already emits ResolvesTo — noop here.
        // Data/config formats (JSON/YAML/TOML/XML) and Markdown never emit
        // `import:` nodes, so NoopResolver is correct for them too.
        _ => &NoopResolver,
    }
}

struct NoopResolver;
impl ImportResolver for NoopResolver {
    fn resolves_to(&self, _: &str, _: &str, _: &str) -> bool {
        false
    }
}

// ── shared helpers ──────────────────────────────────────────────────────────

/// Resolve a relative import `spec` (e.g. `animal`, `../lib/animal`) written in
/// `importer_path` into a normalized, repo-relative path by walking `.` / `..`
/// segments against the importer's directory. Returns `None` if the spec climbs
/// above the repo root. Deterministic: the result depends only on the importer's
/// location, so it can never match a same-named file under an unrelated root.
fn resolve_relative_to_importer(importer_path: &str, spec: &str) -> Option<String> {
    let dir = file_parent_dir(&importer_path.replace('\\', "/"));
    let mut parts: Vec<&str> = if dir.is_empty() {
        Vec::new()
    } else {
        dir.split('/').collect()
    };
    let spec = spec.replace('\\', "/");
    for seg in spec.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?; // None if the spec escapes above the repo root
            }
            other => parts.push(other),
        }
    }
    Some(parts.join("/"))
}

fn file_parent_dir(fp: &str) -> String {
    std::path::Path::new(fp)
        .parent()
        .and_then(|p| p.to_str())
        .filter(|d| !d.is_empty() && *d != ".")
        .unwrap_or("")
        .replace('\\', "/")
}

/// True when `haystack` equals `needle` OR `haystack` ends with `/<needle>`.
fn path_suffix_match(haystack: &str, needle: &str) -> bool {
    haystack == needle || haystack.ends_with(&format!("/{needle}"))
}

/// Shared logic for dot-separated languages (Java, Kotlin, Scala, C#).
/// Handles class-level imports ("import:com.example.Foo") and package-level
/// imports ("import:com.example"), with optional wildcard stripping (".*").
fn dot_lang_resolves_to(import_sig: &str, file_path: &str, exts: &[&str]) -> bool {
    let raw = match import_sig.strip_prefix("import:") {
        Some(p) => p,
        None => return false,
    };
    let import_path = raw.trim_end_matches(".*"); // strip wildcard suffix
    let as_slash = import_path.replace('.', "/");
    let fp = file_path.replace('\\', "/");

    // Class-level: "com/example/Foo.java"
    for ext in exts {
        let class_file = format!("{as_slash}{ext}");
        if path_suffix_match(&fp, &class_file) {
            return true;
        }
    }
    // Package-level: any file whose parent dir ends with the package path
    let dir = file_parent_dir(&fp);
    path_suffix_match(&dir, &as_slash)
}

// ── Go ─────────────────────────────────────────────────────────────────────
// Signature: `import:github.com/Foo/repo/pkg`
// `link_imports_go` emits ResolvesTo for same-module paths; this resolves
// cross-module references where the import path ends with the file's dir.

struct GoResolver;
impl ImportResolver for GoResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        let import_path = match import_sig.strip_prefix("import:") {
            Some(p) => p,
            None => return false,
        };
        let fp = file_path.replace('\\', "/");
        let file_dir = file_parent_dir(&fp);
        if file_dir.is_empty() {
            return false; // root-level Go files are unexportable main packages
        }
        path_suffix_match(import_path, &file_dir)
    }
}

// ── Java ───────────────────────────────────────────────────────────────────
// Signature: `import:org.springframework.web.Controller`
// Dot-to-slash: "org/springframework/web/Controller.java"

struct JavaResolver;
impl ImportResolver for JavaResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        dot_lang_resolves_to(import_sig, file_path, &[".java"])
    }
}

// ── Kotlin ─────────────────────────────────────────────────────────────────
// Signature: `import:kotlin.collections.List` — same convention as Java.

struct KotlinResolver;
impl ImportResolver for KotlinResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        dot_lang_resolves_to(import_sig, file_path, &[".kt", ".kts"])
    }
}

// ── Scala ──────────────────────────────────────────────────────────────────
// Signature: `import:scala.collection.mutable` — same convention as Java.

struct ScalaResolver;
impl ImportResolver for ScalaResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        dot_lang_resolves_to(import_sig, file_path, &[".scala", ".sc"])
    }
}

// ── Python ─────────────────────────────────────────────────────────────────
// Signature: `import:os.path` (absolute), `import:.utils` (relative).
// `link_imports_python` handles both at index time; relative imports require
// the caller's path which is not available here — return false for those.

struct PythonResolver;
impl ImportResolver for PythonResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        let spec = match import_sig.strip_prefix("import:") {
            Some(s) => s,
            None => return false,
        };
        if spec.starts_with('.') {
            return false; // relative: link_imports_python is authoritative
        }
        let as_slash = spec.replace('.', "/");
        let fp = file_path.replace('\\', "/");

        // Module file: "os/path.py"
        if path_suffix_match(&fp, &format!("{as_slash}.py")) {
            return true;
        }
        // Package init: "os/path/__init__.py"
        if path_suffix_match(&fp, &format!("{as_slash}/__init__.py")) {
            return true;
        }
        // Directory match: any .py file in the package dir
        let dir = file_parent_dir(&fp);
        path_suffix_match(&dir, &as_slash)
    }
}

// ── PHP ────────────────────────────────────────────────────────────────────
// Signature: `import:Symfony\Component\HttpKernel\Controller`
// PSR-4: backslash namespace separator → filesystem slash.

struct PhpResolver;
impl ImportResolver for PhpResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        let import_path = match import_sig.strip_prefix("import:") {
            Some(p) => p,
            None => return false,
        };
        let as_slash = import_path.replace('\\', "/");
        let fp = file_path.replace('\\', "/");

        if path_suffix_match(&fp, &format!("{as_slash}.php")) {
            return true;
        }
        let dir = file_parent_dir(&fp);
        path_suffix_match(&dir, &as_slash)
    }
}

// ── C# ─────────────────────────────────────────────────────────────────────
// Signature: `import:System.Collections.Generic`
// By convention (not enforced) C# namespaces map to directory structure.

struct CsharpResolver;
impl ImportResolver for CsharpResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        dot_lang_resolves_to(import_sig, file_path, &[".cs"])
    }
}

// ── C / C++ / Objective-C ─────────────────────────────────────────────────────
// Signature: `import:<stdio.h>` (system) or `import:local/header.h` (local,
// quotes already stripped by the analyzer). Angle-bracket includes are external
// and cannot map to a project file. Quoted `#include "…"` / `#import "…"` are,
// by the C standard, resolved relative to the including file first — so we
// anchor them to the importer's directory and require an exact match. This is
// deterministic: `#include "util.h"` in `src/a.c` can only mean `src/util.h`,
// never a same-named `util.h` under an unrelated directory.

struct CppResolver;
impl ImportResolver for CppResolver {
    fn resolves_to(&self, import_sig: &str, importer_path: &str, target_path: &str) -> bool {
        let include = match import_sig.strip_prefix("import:") {
            Some(p) => p,
            None => return false,
        };
        if include.starts_with('<') {
            return false; // system / framework header
        }
        let bare = include.trim_matches('"');
        let fp = target_path.replace('\\', "/");
        match resolve_relative_to_importer(importer_path, bare) {
            Some(resolved) => fp == resolved,
            None => false,
        }
    }
}

// ── Swift ──────────────────────────────────────────────────────────────────
// Signature: `import:Foundation`, `import:UIKit`, `import:MyModule`
// SPM convention: module name == the directory under Sources/ that holds
// the .swift files. A file belongs to a module when its parent dir name
// matches the module name (handles Sources/MyModule/*.swift).

struct SwiftResolver;
impl ImportResolver for SwiftResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        let module = match import_sig.strip_prefix("import:") {
            Some(m) => m,
            None => return false,
        };
        let fp = file_path.replace('\\', "/");
        if !fp.ends_with(".swift") {
            return false;
        }
        let dir = file_parent_dir(&fp);
        // Last component of parent dir must equal the module name
        let last_dir = dir.rsplit('/').next().unwrap_or(&dir);
        last_dir == module
    }
}

// ── Dart ───────────────────────────────────────────────────────────────────
// Signature: `import:package:myapp/src/util.dart`, `import:dart:core`
// `dart:` → stdlib, no local file.
// Relative (`./`, `../`) → ResolvesTo already emitted by link_imports.
// `package:` → strip scheme + package name, match against remaining path.

struct DartResolver;
impl ImportResolver for DartResolver {
    fn resolves_to(&self, import_sig: &str, _importer_path: &str, file_path: &str) -> bool {
        let uri = match import_sig.strip_prefix("import:") {
            Some(u) => u,
            None => return false,
        };
        if uri.starts_with("dart:") {
            return false; // stdlib
        }
        if uri.starts_with("./") || uri.starts_with("../") {
            return false; // relative: link_imports is authoritative
        }
        let path_part = if let Some(rest) = uri.strip_prefix("package:") {
            // "package:myapp/src/util.dart" → "src/util.dart"
            match rest.find('/') {
                Some(pos) => &rest[pos + 1..],
                None => return false, // "package:myapp" with no path
            }
        } else {
            uri
        };
        let fp = file_path.replace('\\', "/");
        path_suffix_match(&fp, path_part)
    }
}

// ── Ruby ───────────────────────────────────────────────────────────────────
// Two signature forms, tagged by the keyword that produced them (#614):
//   `import:animal` / `import:foo/bar`, from `require_relative`, importer-relative.
//   `import:gem:json`, from `require`, load-path (gem/stdlib/in-repo lib).
// Ruby's `require`/`require_relative` usually take a path without the `.rb`
// extension; the indexer emits the import node but no ResolvesTo chain, so
// resolve the bare require path against the file here.
//
// A `require_relative` path is anchored to the importer's directory and
// matched exactly, so `require_relative '../lib/foo'` in `app/src/main.rb`
// resolves deterministically to `app/lib/foo.rb` and can never over-match a
// same-named file under an unrelated root. A `gem:`-tagged `require` always
// resolves to nothing, since a load-path lookup is not available in the
// graph.
//
// Known limitation (was previously the opposite bug, see #611/#613 history):
// a load-path `require 'my_gem/parser'` that happens to point at an in-repo
// `lib/my_gem/parser.rb` is now unresolved, since `gem:` requires never
// resolve. This is deliberate, over-inclusion (resolving every same-stem
// gem require) is traded for determinism (resolving none of them).

struct RubyResolver;
impl ImportResolver for RubyResolver {
    fn resolves_to(&self, import_sig: &str, importer_path: &str, target_path: &str) -> bool {
        let spec = match import_sig.strip_prefix("import:") {
            Some(s) => s,
            None => return false,
        };
        // #614: `require` (load-path: gem/stdlib/in-repo lib) is tagged
        // `gem:` by the analyzer and resolves to no project file, only
        // `require_relative`'s importer-relative form is resolved here.
        if spec.starts_with("gem:") {
            return false;
        }
        // A require may already carry the extension (`require_relative 'foo.rb'`);
        // strip it so we don't build `foo.rb.rb`.
        let spec = spec.strip_suffix(".rb").unwrap_or(spec);
        let Some(resolved) = resolve_relative_to_importer(importer_path, spec) else {
            return false;
        };
        let fp = target_path.replace('\\', "/");
        fp == format!("{resolved}.rb")
    }
}

// ── Graph GC types ────────────────────────────────────────────────────────────

/// Paths of files that held inbound edges to deleted/changed symbols.
/// The daemon enqueues these for Tier-0 dirty re-resolution.
pub type DirtySet = std::collections::HashSet<String>;

/// Returned by the store's `reindex_replace` operation.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ReplaceReport {
    /// Number of symbols that vanished in this edit (old id not present in new parse).
    pub removed_count: usize,
    /// Paths of files that had inbound edges to the removed symbols.
    pub callers: DirtySet,
    /// RFC-027 #813: the definitions in the reparsed file this edit actually
    /// changed. It is every node this parse produced whose committed edges were
    /// NOT preserved: the edited definitions on a pure body edit, or the whole
    /// file when the edit added or removed a symbol (or no `content` was
    /// supplied, so nothing could be proven unchanged). Empty means every
    /// definition was preserved, so there is no changed region to re-resolve.
    ///
    /// Both lanes scope to exactly these. The save-path lexical lane
    /// (`live_resolve_file`) filters to them so a preserved definition's
    /// already-committed references are not re-recorded as `pending` and the
    /// freshness count stays honest. The request-path editor targets
    /// (`live_resolution_targets`) filter to them too, via the set stashed on the
    /// `EditorPlane` at save, so a preserved definition emits no native provider
    /// round trip. A whole-file re-derive stashes nothing, so the request path
    /// stays whole-file; a dependent file (not reparsed by this save) is also
    /// resolved whole-file.
    #[serde(default)]
    pub changed_defs: Vec<NodeId>,
    /// RFC-027 #813 P2: the committed occurrence rows the reparse is about to
    /// purge for the changed definitions, captured before the purge (see
    /// [`ChangedOccurrence`]). The live lane enumerates
    /// these as editor-resolution targets so it reaches references the
    /// tree-sitter live extractor never detects (macro, desugared,
    /// trait-dispatched) but the committed SCIP occurrence set did capture, and
    /// resolves each at its exact stored column. The `dst` is deliberately NOT
    /// carried: it is the stale committed target, and the editor must re-resolve
    /// the CURRENT buffer position (RFC-027 section 8.2 fencing), so only the
    /// position is trustworthy. Empty on a pure preserve or when no `content`
    /// was supplied.
    #[serde(default)]
    pub changed_occurrences: Vec<ChangedOccurrence>,
    /// RFC-027 #813 (finding 2): whether at least one definition was preserved,
    /// i.e. this was a scoped pure-body edit and `changed_defs` is a proper
    /// subset of the file. When false the whole file was re-derived and
    /// `changed_defs` names every node, so the save path resolves the file
    /// wholesale exactly as before and there is nothing to scope.
    #[serde(default)]
    pub preserved_any: bool,
}

/// RFC-027 #813 P2: one committed occurrence of a changed definition, captured
/// by `reindex_replace` before it purges the file's owned sites, for the live
/// lane to re-resolve at the editor. Carries only the position and kind, never
/// the stale committed `dst` (see [`ReplaceReport::changed_occurrences`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChangedOccurrence {
    /// The changed source definition this occurrence lived in (its enclosing
    /// node), so the daemon can bound the occurrence to that def's current span.
    pub src: NodeId,
    /// 1-based line of the occurrence, already remapped to the current buffer by
    /// the changed definition's block delta when captured.
    pub line: u32,
    /// 0-based UTF-8 byte column of the occurrence, or `None` when the committed
    /// row carried no column (the daemon then name-searches the line).
    pub col: Option<u32>,
    /// The occurrence's edge kind (`ref/call` or `ref/field`), so the served
    /// editor target requests the matching resolution.
    pub kind: String,
    /// Leaf name of the reference identifier at this occurrence (the committed
    /// callee's signature leaf), so the editor can name-search the line when no
    /// column is stored and label the target. Never a node identity: the editor
    /// resolves the CURRENT buffer position and the daemon maps the result to a
    /// SCIP-owned node (RFC-027 section 8.2 fencing), so the stale committed
    /// target is deliberately not carried, only this name hint.
    pub name: String,
}

/// Summary returned by `reconcile` / `travsr fsck`.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct GcReport {
    /// VName paths ghost-deleted from the graph.
    pub ghost_paths: Vec<String>,
    /// Number of orphan edges detected in report mode (`fsck` without `--fix`).
    /// Counted read-only so the default integrity check surfaces edges with a
    /// missing endpoint instead of staying silent about them (issue #580).
    pub orphan_edges_detected: u64,
    /// Number of orphan edges swept (should be 0 in normal operation).
    pub orphan_edges_swept: u64,
    /// #650: self-referential (`src == dst`) `ref/call` edges detected in report
    /// mode. Zero once the write-path guard is in effect; a non-zero count means
    /// the DB predates the guard (or a producer bypassed the choke point).
    pub self_ref_call_edges_detected: u64,
    /// #650: self-referential `ref/call` edges swept under `--fix` (edge + its
    /// occurrence sites). Should be 0 in normal operation.
    pub self_ref_call_edges_swept: u64,
    /// Total node count at reconcile start.
    pub node_count: u64,
    /// Total edge count at reconcile start.
    pub edge_count: u64,
    /// `true` if the mass-delete circuit breaker aborted the reconcile.
    pub aborted: bool,
    /// Human-readable reason for abort (set when `aborted = true`).
    pub abort_reason: Option<String>,
    /// #478 RFC-023 §8: set when `nodes` / `nodes_fts_map` / `nodes_fts_words_map`
    /// row counts disagree — indicates a partial write on one of the lexical
    /// index write paths. Report-only; `fsck --fix` does not auto-repair this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_index_parity_issue: Option<String>,
}

/// Safety guards for reconciliation (§6.5 of the GC architecture doc).
#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    /// Maximum fraction of tracked paths deletable in one pass (0.0–1.0).
    /// Circuit breaker aborts if ghost count exceeds
    /// `max(mass_delete_ceiling_min, db_paths.len() * pct)`. Default `0.50`.
    pub mass_delete_ceiling_pct: f64,
    /// Minimum number of deletions always allowed regardless of `pct`. Default `100`.
    pub mass_delete_ceiling_min: usize,
    /// Re-check `exists_on_disk` immediately before each `delete_file` to guard
    /// against TOCTOU races (§6.5 S3). Default `true`.
    pub toctou_recheck: bool,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            mass_delete_ceiling_pct: 0.50,
            mass_delete_ceiling_min: 100,
            toctou_recheck: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_manifest_file_recognizes_extensionless_and_odd_ext_manifests() {
        assert!(is_manifest_file("go.mod"));
        assert!(is_manifest_file("App.csproj"));
        assert!(is_manifest_file("Directory.Build.csproj"));
        // Not manifests handled by the name recognizer:
        assert!(!is_manifest_file("go.sum"));
        assert!(!is_manifest_file("Cargo.toml")); // ext-mapped, dispatched by name
        assert!(!is_manifest_file("main.go"));
        assert!(!is_manifest_file(""));
    }

    #[test]
    fn is_indexable_path_admits_manifests_extension_and_name_based() {
        // Extension-mapped source / data-format files.
        assert!(is_indexable_path(Path::new("src/main.rs")));
        assert!(is_indexable_path(Path::new("package.json")));
        assert!(is_indexable_path(Path::new("Cargo.toml")));
        assert!(is_indexable_path(Path::new("README.md")));
        // Name-recognized manifests with unmapped/absent extensions.
        assert!(is_indexable_path(Path::new("go.mod")));
        assert!(is_indexable_path(Path::new("src/App.csproj")));
        // Not indexable.
        assert!(!is_indexable_path(Path::new("go.sum")));
        assert!(!is_indexable_path(Path::new("notes.txt")));
        assert!(!is_indexable_path(Path::new("Gemfile")));
    }

    fn sample_vname() -> VName {
        VName::new(
            "github.com/raj-rkv/travsr",
            "main",
            "crates/travsr-core/src/lib.rs",
            "rust",
            "fn:sample",
        )
    }

    #[test]
    fn ruby_resolver_anchors_require_to_importer_dir() {
        let r = resolver_for_language("ruby");
        // `require_relative 'animal'` in cat.rb resolves to the sibling animal.rb.
        assert!(r.resolves_to("import:animal", "ruby/src/cat.rb", "ruby/src/animal.rb"));
        assert!(r.resolves_to("import:./animal", "ruby/src/cat.rb", "ruby/src/animal.rb"));
        // Nested require path, relative to the importer's dir.
        assert!(r.resolves_to("import:lib/animal", "app/main.rb", "app/lib/animal.rb"));
        // A require that already carries the extension must not become foo.rb.rb.
        assert!(r.resolves_to("import:animal.rb", "ruby/src/cat.rb", "ruby/src/animal.rb"));
        // Parent-relative require is resolved *through* ../, not merely stripped.
        assert!(r.resolves_to(
            "import:../lib/animal",
            "app/src/main.rb",
            "app/lib/animal.rb"
        ));
        // Different stem must not match.
        assert!(!r.resolves_to("import:animal", "ruby/src/cat.rb", "ruby/src/cat.rb"));
        // A subpath must actually constrain the match.
        assert!(!r.resolves_to("import:lib/animal", "app/main.rb", "app/other/animal.rb"));
        // A bare import prefix or malformed spec resolves to nothing.
        assert!(!r.resolves_to("animal", "ruby/src/cat.rb", "ruby/src/animal.rb"));
        // Determinism (#613): a same-named file under an unrelated root must NOT
        // match, even though the bare stem is identical. This is the transitive
        // false-positive class the fixpoint would otherwise compound.
        assert!(!r.resolves_to("import:animal", "ruby/src/cat.rb", "pkg/other/animal.rb"));
    }

    #[test]
    fn ruby_resolver_refuses_load_path_require() {
        // #614: `require 'json'` (a gem/stdlib load-path require) is tagged
        // `import:gem:json` by the analyzer and must never resolve, even when
        // a same-stem `json.rb` sits right next to the importer. This is the
        // exact false-positive from the issue.
        let r = resolver_for_language("ruby");
        assert!(!r.resolves_to("import:gem:json", "ruby/src/cat.rb", "ruby/src/json.rb"));
        assert!(!r.resolves_to("import:gem:animal", "ruby/src/cat.rb", "ruby/src/animal.rb"));
    }

    #[test]
    fn ruby_resolver_gem_early_return_is_load_bearing() {
        // Mutation-test guard: `ruby_resolver_refuses_load_path_require`
        // above passes even if the `gem:` early-return in `RubyResolver` is
        // deleted, because the literal "gem:" text gets folded into the path
        // segment by `resolve_relative_to_importer`, so it can never equal a
        // realistic `target_path` and the assertion passes for the wrong
        // reason. This test uses a `target_path` that contains the literal
        // "gem:" text the join would produce, so it fails loudly (mismatched
        // path) if the early-return is ever removed, proving the guard is
        // exercised rather than vacuously true.
        let r = resolver_for_language("ruby");
        assert!(!r.resolves_to("import:gem:json", "ruby/src/cat.rb", "ruby/src/gem:json.rb"));
    }

    #[test]
    fn cpp_and_objc_anchor_quoted_include_to_importer_dir() {
        // `#include "util.h"` in src/net/client.c resolves to src/net/util.h.
        let cpp = resolver_for_language("cpp");
        assert!(cpp.resolves_to("import:util.h", "src/net/client.c", "src/net/util.h"));
        // Parent-relative include is resolved through ../.
        assert!(cpp.resolves_to("import:../util.h", "src/net/client.c", "src/util.h"));
        // Angle-bracket system/framework header never maps to a project file.
        assert!(!cpp.resolves_to("import:<stdio.h>", "src/net/client.c", "src/net/stdio.h"));
        // A same-named header under an unrelated dir must NOT match (#613).
        assert!(!cpp.resolves_to("import:util.h", "src/net/client.c", "src/other/util.h"));

        // Objective-C shares the C/C++ resolver: `#import "Model.h"` is
        // importer-relative. Before this it fell through to NoopResolver and
        // Objective-C had no Phase-2 blast radius at all.
        let objc = resolver_for_language("objectivec");
        assert!(objc.resolves_to("import:Model.h", "app/View.m", "app/Model.h"));
        assert!(!objc.resolves_to("import:Model.h", "app/View.m", "vendor/Model.h"));
        // The plugin-protocol "objc" alias resolves to the same resolver.
        let objc_alias = resolver_for_language("objc");
        assert!(objc_alias.resolves_to("import:Model.h", "app/View.m", "app/Model.h"));
    }

    #[test]
    fn vname_round_trips_through_serde_json() {
        let v = sample_vname();
        let json = serde_json::to_string(&v).unwrap();
        let back: VName = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn vname_id_is_deterministic() {
        assert_eq!(sample_vname().id(), sample_vname().id());
    }

    #[test]
    fn vname_id_differs_on_any_field_change() {
        let base = sample_vname();
        let mut other = base.clone();
        other.signature = "fn:different".into();
        assert_ne!(base.id(), other.id());
    }

    #[test]
    fn edge_kind_round_trips_through_string() {
        for kind in [
            EdgeKind::Depends,
            EdgeKind::RefCall,
            EdgeKind::DefinesBinding,
            EdgeKind::Exports,
            EdgeKind::ResolvesTo,
            EdgeKind::RefImports,
            EdgeKind::IsImplementation,
            EdgeKind::Overrides,
            EdgeKind::FFICall,
            EdgeKind::Configures,
            EdgeKind::ExternalDependency,
        ] {
            assert_eq!(EdgeKind::from_str(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn ppr_weights_are_ordered_by_semantic_strength() {
        // RefCall > DefinesBinding > Exports > Depends == ResolvesTo > RefImports == IsImplementation > Configures > Overrides == ExternalDependency
        assert!(EdgeKind::RefCall.ppr_weight() > EdgeKind::DefinesBinding.ppr_weight());
        assert!(EdgeKind::DefinesBinding.ppr_weight() > EdgeKind::Exports.ppr_weight());
        assert!(EdgeKind::Exports.ppr_weight() > EdgeKind::Depends.ppr_weight());
        assert_eq!(
            EdgeKind::Depends.ppr_weight(),
            EdgeKind::ResolvesTo.ppr_weight()
        );
        assert!(EdgeKind::Depends.ppr_weight() > EdgeKind::RefImports.ppr_weight());
        assert_eq!(
            EdgeKind::RefImports.ppr_weight(),
            EdgeKind::IsImplementation.ppr_weight()
        );
        assert!(EdgeKind::IsImplementation.ppr_weight() > EdgeKind::Configures.ppr_weight());
        assert!(EdgeKind::Configures.ppr_weight() > EdgeKind::Overrides.ppr_weight());
        assert_eq!(
            EdgeKind::Overrides.ppr_weight(),
            EdgeKind::ExternalDependency.ppr_weight()
        );
    }

    #[test]
    fn ppr_weights_are_positive_and_at_most_one() {
        for kind in [
            EdgeKind::Depends,
            EdgeKind::RefCall,
            EdgeKind::DefinesBinding,
            EdgeKind::Exports,
            EdgeKind::ResolvesTo,
            EdgeKind::RefImports,
            EdgeKind::IsImplementation,
            EdgeKind::Overrides,
            EdgeKind::FFICall,
            EdgeKind::Configures,
            EdgeKind::ExternalDependency,
        ] {
            let w = kind.ppr_weight();
            assert!(
                w > 0.0 && w <= 1.0,
                "weight {w} for {kind:?} must be in (0, 1]"
            );
        }
    }

    #[test]
    fn node_id_matches_vname_id() {
        let v = sample_vname();
        let node = Node::new(v.clone(), "function");
        assert_eq!(node.id, v.id());
    }

    #[test]
    fn version_byte_produces_different_id_than_unversioned() {
        // Regression guard: confirms the RFC-002 domain separator is actually
        // prepended and that length-prefix encoding is used. The versioned format
        // starts with [SIGNATURE_FORMAT_VERSION][len][corpus...]; the v0 format
        // starts with raw corpus bytes. These byte streams can never be equal
        // regardless of field contents.
        let v = sample_vname();
        let versioned_id = v.id(); // uses SIGNATURE_FORMAT_VERSION byte + length-prefix

        // Compute the legacy (no version byte, NUL-separated) hash directly.
        let mut hasher = blake3::Hasher::new();
        hasher.update(v.corpus.as_bytes());
        hasher.update(b"\0");
        hasher.update(v.root.as_bytes());
        hasher.update(b"\0");
        hasher.update(v.path.as_bytes());
        hasher.update(b"\0");
        hasher.update(v.language.as_bytes());
        hasher.update(b"\0");
        hasher.update(v.signature.as_bytes());
        let digest = hasher.finalize();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&digest.as_bytes()[..8]);
        let legacy_id = NodeId(u64::from_le_bytes(buf));

        assert_ne!(
            versioned_id, legacy_id,
            "RFC-002 version byte + length-prefix must produce a different NodeId than the legacy NUL-separated hash"
        );
    }

    // ── ARCH-102: canonical_corpus tests ─────────────────────────────────────

    #[test]
    fn canonical_corpus_handles_https_with_git_suffix() {
        assert_eq!(
            canonical_corpus("https://github.com/raj-rkv/travsr.git"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_handles_https_without_git_suffix() {
        assert_eq!(
            canonical_corpus("https://github.com/raj-rkv/travsr"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_handles_scp_style_ssh() {
        assert_eq!(
            canonical_corpus("git@github.com:raj-rkv/travsr.git"),
            "github.com/raj-rkv/travsr"
        );
        assert_eq!(
            canonical_corpus("git@github.com:raj-rkv/travsr"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_handles_ssh_url() {
        assert_eq!(
            canonical_corpus("ssh://git@github.com/raj-rkv/travsr.git"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_handles_git_protocol() {
        assert_eq!(
            canonical_corpus("git://github.com/raj-rkv/travsr.git"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_lowercases_input() {
        assert_eq!(
            canonical_corpus("HTTPS://GITHUB.COM/Raj-Rkv/Travsr.GIT"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_strips_port() {
        assert_eq!(
            canonical_corpus("https://github.com:443/raj-rkv/travsr.git"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_strips_trailing_slash() {
        assert_eq!(
            canonical_corpus("https://github.com/raj-rkv/travsr/"),
            "github.com/raj-rkv/travsr"
        );
    }

    #[test]
    fn canonical_corpus_gitlab() {
        assert_eq!(
            canonical_corpus("https://gitlab.com/acme/payments-api.git"),
            "gitlab.com/acme/payments-api"
        );
    }

    #[test]
    fn canonical_corpus_local_uses_basename() {
        let path = std::path::Path::new("/home/user/my-project");
        assert_eq!(canonical_corpus_local(path), "local/my-project");
    }

    #[test]
    fn canonical_corpus_local_sanitises_special_chars() {
        let path = std::path::Path::new("/tmp/My Project (v2)");
        let result = canonical_corpus_local(path);
        assert!(result.starts_with("local/"));
        assert!(!result.contains(' '), "spaces must be replaced");
        assert!(!result.contains('('), "parens must be replaced");
    }

    #[test]
    fn different_corpus_produces_non_colliding_node_ids() {
        // Regression: same file + same signature in two different repos must
        // produce different NodeIds because corpus is part of the BLAKE3 input.
        let v_repo_a = VName::new(
            "github.com/acme/repo-a",
            "",
            "src/foo.ts",
            "typescript",
            "fn:bar",
        );
        let v_repo_b = VName::new(
            "github.com/acme/repo-b",
            "",
            "src/foo.ts",
            "typescript",
            "fn:bar",
        );
        assert_ne!(
            v_repo_a.id(),
            v_repo_b.id(),
            "different corpora must produce different NodeIds (cross-repo VName collision)"
        );
    }

    // ── Language enum (ADR-005 / RFC-003) ─────────────────────────────────────

    #[test]
    fn language_from_extension_covers_all_variants() {
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("tsx"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("mts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("cts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("pyi"), Some(Language::Python));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("js"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("jsx"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("mjs"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("cjs"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("m"), Some(Language::ObjectiveC));
        assert_eq!(Language::from_extension("mm"), Some(Language::ObjectiveC));
        assert_eq!(Language::from_extension("json"), Some(Language::Json));
        assert_eq!(Language::from_extension("jsonc"), Some(Language::Json));
        assert_eq!(Language::from_extension("yml"), Some(Language::Yaml));
        assert_eq!(Language::from_extension("yaml"), Some(Language::Yaml));
        assert_eq!(Language::from_extension("toml"), Some(Language::Toml));
        assert_eq!(Language::from_extension("xml"), Some(Language::Xml));
        assert_eq!(Language::from_extension("xsd"), Some(Language::Xml));
        assert_eq!(Language::from_extension("xsl"), Some(Language::Xml));
        assert_eq!(Language::from_extension("md"), Some(Language::Markdown));
        assert_eq!(
            Language::from_extension("markdown"),
            Some(Language::Markdown)
        );
        assert_eq!(Language::from_extension("mdx"), Some(Language::Markdown));
        assert_eq!(Language::from_extension(""), None);
    }

    #[test]
    fn language_as_str_and_from_str_round_trip() {
        for lang in [
            Language::TypeScript,
            Language::Rust,
            Language::Python,
            Language::Go,
            Language::ObjectiveC,
            Language::Json,
            Language::Yaml,
            Language::Toml,
            Language::Xml,
            Language::Markdown,
        ] {
            let s = lang.as_str();
            assert_eq!(
                Language::from_str(s),
                Some(lang),
                "round-trip failed for {s}"
            );
        }
    }

    #[test]
    fn language_as_str_values_are_lowercase() {
        assert_eq!(Language::TypeScript.as_str(), "typescript");
        assert_eq!(Language::Rust.as_str(), "rust");
        assert_eq!(Language::Python.as_str(), "python");
        assert_eq!(Language::Go.as_str(), "go");
        assert_eq!(Language::Json.as_str(), "json");
        assert_eq!(Language::Yaml.as_str(), "yaml");
        assert_eq!(Language::Toml.as_str(), "toml");
        assert_eq!(Language::Xml.as_str(), "xml");
        assert_eq!(Language::Markdown.as_str(), "markdown");
    }

    #[test]
    fn language_from_str_returns_none_for_unknown() {
        assert_eq!(Language::from_str("go"), Some(Language::Go));
        assert_eq!(Language::from_str("TypeScript"), None);
        assert_eq!(Language::from_str(""), None);
    }

    #[test]
    fn noise_keeps_external_package_node() {
        // Synthetic registry nodes (npmjs.com, crates.io) must NOT be filtered
        // out by is_noise_node — they have an empty path and a registry corpus.
        for (corpus, sig, lang) in [
            ("npmjs.com", "pkg:express@^4.18.0", "json"),
            ("crates.io", "pkg:serde@1", "toml"),
            (
                "maven.org",
                "pkg:org.springframework:spring-core@6.0.0",
                "xml",
            ),
        ] {
            let node = Node::new(VName::new(corpus, "", "", lang, sig), "package");
            assert!(
                !is_noise_node(&node),
                "external package node for {corpus} must not be noise"
            );
        }
    }

    #[test]
    fn is_data_format_true_for_data_formats_false_for_code() {
        assert!(Language::Json.is_data_format());
        assert!(Language::Yaml.is_data_format());
        assert!(Language::Toml.is_data_format());
        assert!(Language::Xml.is_data_format());
        assert!(!Language::Rust.is_data_format());
        assert!(!Language::TypeScript.is_data_format());
        assert!(!Language::Python.is_data_format());
        // Markdown is Phase-A-only but is prose, not a data/config format —
        // it must stay out of is_data_format() (whose callers key skeleton
        // text on "language path", wrong for prose) while still tripping
        // is_phase_a_only() (whose callers gate Phase B dispatch).
        assert!(!Language::Markdown.is_data_format());
    }

    #[test]
    fn is_phase_a_only_covers_data_formats_and_markdown_not_code() {
        for lang in [
            Language::Json,
            Language::Yaml,
            Language::Toml,
            Language::Xml,
            Language::Markdown,
        ] {
            assert!(lang.is_phase_a_only(), "{lang:?} must be Phase-A-only");
        }
        for lang in [Language::Rust, Language::TypeScript, Language::Python] {
            assert!(!lang.is_phase_a_only(), "{lang:?} must not be Phase-A-only");
        }
    }

    // Regression: two symbols in different languages (same file path, same sig)
    // produce different NodeIds because language is part of the BLAKE3 input.
    #[test]
    fn language_field_prevents_cross_language_vname_collision() {
        let ts = VName::new("github.com/a/b", "", "src/main.rs", "typescript", "fn:main");
        let rs = VName::new("github.com/a/b", "", "src/main.rs", "rust", "fn:main");
        assert_ne!(
            ts.id(),
            rs.id(),
            "different language fields must produce different NodeIds"
        );
    }

    // node.with_package() sets package without changing id.
    #[test]
    fn node_with_package_does_not_change_id() {
        let vname = VName::new("github.com/a/b", "", "src/lib.rs", "rust", "fn:open");
        let plain = Node::new(vname.clone(), "function");
        let packaged = Node::new(vname, "function").with_package("my-crate");
        assert_eq!(plain.id, packaged.id, "package must not affect NodeId");
        assert_eq!(packaged.package, "my-crate");
        assert_eq!(plain.package, "");
    }

    #[test]
    fn edge_ffi_call_builder_sets_confidence() {
        let e = Edge::ffi_call(NodeId(1), NodeId(2), 90);
        assert_eq!(e.kind, EdgeKind::FFICall);
        assert_eq!(e.confidence, Some(90));
    }

    #[test]
    fn edge_new_has_no_confidence() {
        let e = Edge::new(NodeId(1), NodeId(2), EdgeKind::RefCall);
        assert_eq!(e.confidence, None);
    }

    #[test]
    fn edge_kind_ffi_call_roundtrip() {
        assert_eq!(EdgeKind::FFICall.as_str(), "ffi/call");
        assert_eq!(EdgeKind::from_str("ffi/call"), Some(EdgeKind::FFICall));
    }

    #[test]
    fn ppr_weight_ffi_call_is_between_refcall_and_defines_binding() {
        assert!(EdgeKind::FFICall.ppr_weight() < EdgeKind::RefCall.ppr_weight());
        assert!(EdgeKind::FFICall.ppr_weight() > EdgeKind::DefinesBinding.ppr_weight());
        assert!((EdgeKind::FFICall.ppr_weight() - 0.85_f32).abs() < 1e-6);
    }

    #[test]
    fn edge_serde_roundtrip_with_confidence() {
        let e = Edge::ffi_call(NodeId(42), NodeId(99), 75);
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"confidence\":75"));
        let e2: Edge = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.confidence, Some(75));
    }

    #[test]
    fn edge_serde_roundtrip_without_confidence_field() {
        // JSON produced before v6 (no confidence field) must deserialize to None
        let json = r#"{"src":1,"dst":2,"kind":"ref/call"}"#;
        let e: Edge = serde_json::from_str(json).unwrap();
        assert_eq!(e.confidence, None);
    }

    #[test]
    fn unresolved_call_serde_roundtrip_without_recv_type_field() {
        // T-11 (#529): a pre-#529 sidecar payload (no recv_type key at all)
        // must still deserialize, with recv_type defaulting to None.
        let json = r#"{"src":1,"callee_sig":"fn:filter","caller_line":42,"is_method_call":true}"#;
        let u: UnresolvedCall = serde_json::from_str(json).unwrap();
        assert_eq!(u.recv_type, None);
        assert!(u.is_method_call);
    }

    #[test]
    fn unresolved_call_serde_roundtrip_with_recv_type_field() {
        let u = UnresolvedCall {
            src: NodeId(1),
            callee_sig: "fn:filter".to_string(),
            alt_callee_sig: None,
            hint_crate: None,
            caller_line: 42,
            caller_col: None,
            is_method_call: true,
            recv_type: Some("Session".to_string()),
        };
        let json = serde_json::to_string(&u).unwrap();
        assert!(json.contains("\"recv_type\":\"Session\""));
        let u2: UnresolvedCall = serde_json::from_str(&json).unwrap();
        assert_eq!(u2.recv_type, Some("Session".to_string()));
    }

    #[test]
    fn unresolved_call_recv_type_omitted_from_json_when_none() {
        // skip_serializing_if keeps wire payloads small when the extractor
        // could not recover a receiver type (the overwhelming common case).
        let u = UnresolvedCall {
            src: NodeId(1),
            callee_sig: "fn:filter".to_string(),
            alt_callee_sig: None,
            hint_crate: None,
            caller_line: 42,
            caller_col: None,
            is_method_call: true,
            recv_type: None,
        };
        let json = serde_json::to_string(&u).unwrap();
        assert!(!json.contains("recv_type"));
    }

    #[test]
    fn display_label_uses_path_for_file_nodes() {
        let n = Node::new(VName::new("", "", "src/lib.rs", "rust", "file"), "file");
        assert_eq!(display_label(&n).as_ref(), "src/lib.rs");
    }

    #[test]
    fn display_label_uses_signature_otherwise() {
        let n = Node::new(
            VName::new("", "", "src/lib.rs", "rust", "fn:main"),
            "function",
        );
        assert_eq!(display_label(&n).as_ref(), "fn:main");
    }

    fn label_of(signature: &str) -> String {
        label_with_path("p", signature)
    }

    fn label_with_path(path: &str, signature: &str) -> String {
        let n = Node::new(VName::new("c", "", path, "java", signature), "definition");
        display_label(&n).into_owned()
    }

    #[test]
    fn display_label_cleans_scip_constructor() {
        // A synthesized java constructor node (scip-reader path): the raw symbol
        // carries the indexer prefix, package path, and `<init>`; the label must
        // collapse to the enclosing type name. The node's own path is stripped
        // before the descriptor is parsed (here it embeds a space, exercising the
        // path-strip that a "last whitespace token" heuristic would break on).
        assert_eq!(
            label_with_path(
                "src/main/java/com/My Demo/Greeter.java",
                "scip:src/main/java/com/My Demo/Greeter.java:semanticdb maven . . com/demo/Greeter#`<init>`()."
            ),
            "Greeter"
        );
    }

    #[test]
    fn display_label_cleans_scip_go_method() {
        // A real multi-field go SCIP symbol (scheme/manager/name/version present),
        // with a package name that itself contains dots.
        assert_eq!(
            label_with_path(
                "net/http/client.go",
                "scip:net/http/client.go:scip-go gomod github.com/foo/bar v1.2.0 net/http/Client#Do()."
            ),
            "Client.Do"
        );
    }

    #[test]
    fn display_label_cleans_sdb_symbols() {
        // scala reader path (`sdb:` prefix).
        assert_eq!(label_of("sdb:com/demo/Greeter#greet()."), "Greeter.greet");
        assert_eq!(label_of("sdb:com/demo/Main.g."), "Main.g");
        assert_eq!(label_of("sdb:com/demo/Greeter#`<init>`()."), "Greeter");
        // A parameter descriptor, if one ever reaches the display layer.
        assert_eq!(
            label_of("sdb:com/demo/Greeter#greet().(name)"),
            "Greeter.greet.name"
        );
    }

    #[test]
    fn display_label_leaves_native_signatures_untouched() {
        assert_eq!(label_of("method:Greeter.greet"), "method:Greeter.greet");
        assert_eq!(label_of("class:Main"), "class:Main");
    }

    #[test]
    fn display_label_collapses_overload_disambiguator() {
        // Overloads carry a `(+N)` disambiguator on the method descriptor; the
        // display name drops it, so `greet().` and `greet(+1).` both read
        // `Greeter.greet`. The node id (raw signature) is what keeps them
        // distinct downstream — display is intentionally lossy here.
        assert_eq!(
            label_with_path(
                "src/Greeter.java",
                "scip:src/Greeter.java:semanticdb maven . . com/demo/Greeter#greet()."
            ),
            "Greeter.greet"
        );
        assert_eq!(
            label_with_path(
                "src/Greeter.java",
                "scip:src/Greeter.java:semanticdb maven . . com/demo/Greeter#greet(+1)."
            ),
            "Greeter.greet"
        );
    }

    #[test]
    fn display_label_drops_package_across_collisions() {
        // Same type name in two packages collapses to the same package-free
        // label; identity is preserved by the raw signature, not the label.
        assert_eq!(label_of("sdb:com/a/Greeter#greet()."), "Greeter.greet");
        assert_eq!(label_of("sdb:com/b/Greeter#greet()."), "Greeter.greet");
    }

    #[test]
    fn display_label_cleans_scip_local() {
        // An anonymous SCIP local has no type/member structure. Before this fix
        // it fell through `display_signature` raw as `scip:{path}:local N`; now
        // only the indexer noise is dropped, keeping the honest `local N`.
        assert_eq!(
            label_with_path("src/Greeter.java", "scip:src/Greeter.java:local 4"),
            "local 4"
        );
    }

    #[test]
    fn display_label_falls_back_on_malformed_scip() {
        // A scip signature whose global-symbol part has too few fields cannot be
        // parsed into descriptors; the renderer must fall back to the raw
        // signature rather than panic or emit an empty label.
        let sig = "scip:src/Greeter.java:semanticdb maven";
        assert_eq!(label_with_path("src/Greeter.java", sig), sig);
    }

    #[test]
    fn display_signature_empty_path_matches_known_path() {
        // The `graph` edge-endpoint renderers call `display_signature(sig, "")`
        // with no path in hand. It must clean identically to the path-aware call,
        // even when the embedded path contains spaces (which the 4-field split
        // would otherwise miscount).
        let sig = "scip:src/My Demo/Greeter.java:semanticdb maven . . com/demo/Greeter#greet().";
        assert_eq!(
            display_signature(sig, "src/My Demo/Greeter.java").as_ref(),
            "Greeter.greet"
        );
        assert_eq!(display_signature(sig, "").as_ref(), "Greeter.greet");
    }

    #[test]
    fn noise_detects_third_party() {
        let n = Node::new(
            VName::new("", "", "third_party/foo/bar.go", "go", "fn:bar"),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_scip_local() {
        let n = Node::new(
            VName::new("", "", "pkg/eviction/handler.go", "go", "local 27"),
            "variable",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_keeps_production_node() {
        let n = Node::new(
            VName::new("", "", "pkg/eviction/handler.go", "go", "fn:Handle"),
            "function",
        );
        assert!(!is_noise_node(&n));
    }

    #[test]
    fn noise_detects_root_node_modules() {
        let n = Node::new(
            VName::new(
                "",
                "",
                "node_modules/react/index.js",
                "typescript",
                "fn:createElement",
            ),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_repo_escaping_path() {
        let n = Node::new(
            VName::new(
                "",
                "",
                "../../../Library/Caches/go-build/80/abc-d",
                "go",
                "file",
            ),
            "file",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_library_caches() {
        let n = Node::new(
            VName::new("", "", "a/b/Library/Caches/go-build/x", "go", "fn:foo"),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_dot_cache() {
        let n = Node::new(
            VName::new("", "", "x/.cache/yarn/v6/blah", "typescript", "fn:bar"),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_go_build_cache() {
        let n = Node::new(
            VName::new("", "", "home/user/.cache/go-build/ab/abcdef", "go", "fn:f"),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_detects_go_pkg_mod() {
        let n = Node::new(
            VName::new(
                "",
                "",
                "a/go/pkg/mod/github.com/foo/bar@v1.0.0/baz.go",
                "go",
                "fn:Baz",
            ),
            "function",
        );
        assert!(is_noise_node(&n));
    }

    #[test]
    fn noise_keeps_real_source_node() {
        let n = Node::new(
            VName::new(
                "",
                "",
                "pkg/api/servicecidr/servicecidr.go",
                "go",
                "fn:OverlapsPrefix",
            ),
            "function",
        );
        assert!(!is_noise_node(&n));
    }
}
