use crate::ffi_marker::FfiMarker;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use travsr_core::{Edge, Node, ScipRef, UnresolvedCall};

/// Current protocol version. Bump on any breaking wire change.
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseRequest {
    /// Absolute path on disk — used to read/mmap the file.
    pub path: PathBuf,
    /// Repo-relative path used in VName construction (the stable graph key).
    /// Must match the `vname_path` passed to `Indexer::parse_file_with_vname`.
    pub vname_path: String,
    pub corpus: String,
    pub package: String,
    /// Populated only for git-blob indexing (content not on disk).
    pub source: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParseResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub ffi_markers: Vec<FfiMarker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeRequest {
    pub root: PathBuf,
    /// Corpus identifier (e.g. `github.com/org/repo`). Used by SCIP ingest to
    /// produce correct VNames. Defaults to empty string for backwards compatibility
    /// with plugin binaries that predate this field.
    #[serde(default)]
    pub corpus: String,
    /// Sandbox-authorized writable scratch directory, expressed as the SIDECAR
    /// sees it rather than as the host does. Sidecar tools that need to write
    /// temp files (e.g. scip-clang SCIP output, scip-ruby index) SHOULD place
    /// them under this path: the sandbox's write grant covers only this
    /// directory.
    ///
    /// The two views differ on Linux, where bwrap binds the host directory at a
    /// fixed mount point instead of at its own path, so the host path resolves
    /// to nothing inside the namespace. `travsr-plugin-host`'s
    /// `sandbox::sidecar_scratch_path` is what converts one to the other.
    ///
    /// Falling back to `std::env::temp_dir()` is safe rather than a gamble, and
    /// the previous wording here had it backwards: the sandbox sets `TMPDIR` to
    /// that same scratch mount, so the fallback lands in exactly the directory
    /// the write grant covers.
    ///
    /// Validate before trusting it. Empty means a daemon older than the field,
    /// and a daemon older than the remap above sends its own host path, which a
    /// sandboxed sidecar cannot open. Check that it names a directory this
    /// process can see and fall back to a tempdir when it does not.
    #[serde(default)]
    pub scratch: PathBuf,
    /// Pre-walked list of source files for this language (repo-root-relative paths),
    /// forwarded from the daemon's Phase A walk (P6 — #329).
    /// `None` = old daemon that doesn't support P6 → sidecar MUST walk itself.
    /// `Some(paths)` = daemon pre-walked; sidecar SHOULD use this list.
    /// (P1 ensures this is never `Some([])` for a lang without source files —
    /// sidecars for absent languages are never spawned.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InvokeResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// G2 attribution records — reference occurrences with call-site line numbers.
    /// Old sidecar binaries omit this field; `#[serde(default)]` provides `[]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<ScipRef>,
    /// Cross-crate calls that Phase B could not resolve to a concrete NodeId.
    /// The daemon resolves these after Phase B using Phase A nodes in the store.
    /// Old sidecar binaries omit this field; `#[serde(default)]` provides `[]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unresolved_calls: Vec<UnresolvedCall>,
    /// Run-scoped caveats the sidecar wants surfaced regardless of how much it
    /// produced. Old sidecar binaries omit this field; `#[serde(default)]`
    /// provides `[]`, so an empty list means "nothing to say", never "too old
    /// to say it".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<PluginDiagnostic>,
}

impl InvokeResponse {
    pub fn unsupported() -> Self {
        Self::default()
    }
}

/// How loudly the host should surface a [`PluginDiagnostic`].
///
/// Deliberately two-valued for a sidecar to *choose* from. A sidecar that wants
/// finer control is asking the wrong question: anything it would file as `debug`
/// belongs on its stderr, which the host already drains into a bounded ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// The analysis completed but is known to be incomplete or untrustworthy.
    Warning,
    /// Context worth recording that does not qualify the result.
    Info,
    /// A severity this host does not know, from a sidecar newer than it.
    ///
    /// Without this arm serde rejects the whole frame, `decode_message` fails,
    /// and the host discards every node and edge of that run and marks the
    /// sidecar crashed: a language's entire Phase B index lost over a field
    /// designed to be ignorable.
    ///
    /// The host surfaces it as a warning rather than dropping it. An unknown
    /// severity came from a sidecar that thought the record mattered, and a
    /// warning a reader can disregard costs less than a caveat nobody sees.
    #[serde(other)]
    Unknown,
}

/// A run-scoped diagnostic a Phase B sidecar wants its host to surface.
///
/// This exists because a sidecar had no way to be heard. The host echoes the
/// stderr ring only when a run returns zero nodes, so any analyzer that
/// produced *some* output while knowing its result was degraded, java skipping
/// test compilation, a stale emitter binary, wrote its warning into a void.
/// Widening that echo was not an option: it would put every analyzer's routine
/// chatter on every successful index.
///
/// The channel is opt-in per message, which is what makes an unconditional echo
/// safe. A sidecar emits a diagnostic only when it has decided the run carries a
/// caveat, so the host never has to guess which stderr lines mattered.
///
/// Scope: one record per *run*, not per source location. RFC-016's
/// `NormalizedDiagnostic` (per-file, per-line compiler diagnostics) is a
/// separate and finer-grained channel; this does not stand in for it or
/// preempt its design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDiagnostic {
    pub severity: DiagnosticSeverity,
    /// Stable dotted identifier for the condition, e.g.
    /// `java.tests-not-compiled`, `emitter.version-mismatch`. Machine-readable
    /// and greppable across runs; the prose in `message` is not.
    ///
    /// The shape is enforced by the host, not by this type: a sidecar is
    /// untrusted, so `travsr-plugin-host`'s `is_diagnostic_code` accepts only
    /// `[A-Za-z0-9._-]` and logs anything else under a neutral placeholder code.
    ///
    /// `serde(default)`: an absent field costs the record its code, exactly as a
    /// malformed one does, instead of costing the run its whole index.
    #[serde(default)]
    pub code: String,
    /// One line, addressed to the developer running the index. The host strips
    /// control characters and truncates before logging, and caps how many
    /// records of a single response it will echo at all.
    ///
    /// `serde(default)`: same reason as `code` above.
    #[serde(default)]
    pub message: String,
}

impl PluginDiagnostic {
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Info,
            code: code.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub daemon_protocol_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub protocol_version: u32,
    pub plugin_version: String,
    /// Canonical lowercase language string — must match the normative table in language_map.rs.
    pub language: String,
    pub extensions: Vec<String>,
    pub supports_phase_b: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginRequest {
    Handshake(HandshakeRequest),
    Parse(ParseRequest),
    Invoke(InvokeRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginResponse {
    Handshake(HandshakeResponse),
    Parse(ParseResponse),
    Invoke(InvokeResponse),
    Error(PluginError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginError {
    pub file: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_message, encode_message};
    use std::io::Cursor;
    use travsr_core::{EdgeKind, NodeId, VName};

    /// A sidecar newer than this host files a diagnostic at a severity the host
    /// has never heard of, and omits the two string fields.
    ///
    /// Before the `serde(other)` and `serde(default)` arms the whole frame
    /// failed to decode: the host marked the sidecar crashed and discarded the
    /// language's entire node and edge set over an ignorable field.
    #[test]
    fn unknown_severity_keeps_the_runs_nodes_and_edges() {
        let node = Node {
            id: NodeId(1),
            vname: VName {
                corpus: "c".into(),
                root: String::new(),
                path: "a.rs".into(),
                language: "rust".into(),
                signature: "fn:a".into(),
            },
            kind: "function".into(),
            package: String::new(),
            line: Some(1),
            end_line: Some(2),
            test_role: Default::default(),
        };
        let edge = Edge {
            src: NodeId(1),
            dst: NodeId(1),
            kind: EdgeKind::RefCall,
            confidence: None,
            provenance: None,
        };
        let resp = InvokeResponse {
            nodes: vec![node],
            edges: vec![edge],
            refs: Vec::new(),
            unresolved_calls: Vec::new(),
            diagnostics: vec![PluginDiagnostic::warning("a.b", "m")],
        };
        let mut wire =
            serde_json::to_value(PluginResponse::Invoke(resp)).expect("serialize response");
        let diag = &mut wire["diagnostics"][0];
        diag["severity"] = serde_json::json!("error");
        diag.as_object_mut()
            .expect("diagnostic is an object")
            .retain(|k, _| k == "severity");

        let frame = encode_message(&wire).expect("encode frame");
        let decoded: PluginResponse =
            decode_message(&mut Cursor::new(frame)).expect("unknown severity must still decode");

        let PluginResponse::Invoke(out) = decoded else {
            panic!("expected an invoke response");
        };
        assert_eq!(out.nodes.len(), 1, "nodes must survive an unknown severity");
        assert_eq!(out.edges.len(), 1, "edges must survive an unknown severity");
        assert_eq!(out.diagnostics[0].severity, DiagnosticSeverity::Unknown);
        assert_eq!(out.diagnostics[0].code, "");
        assert_eq!(out.diagnostics[0].message, "");
    }
}
