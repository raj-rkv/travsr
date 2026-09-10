use super::parse_output_to_response;
use travsr_core::Language;
use travsr_plugin_protocol::{InvokeRequest, InvokeResponse, ParseRequest, ParseResponse, Plugin};

pub struct RustPlugin;

impl Plugin for RustPlugin {
    fn language(&self) -> Language {
        Language::Rust
    }
    fn extensions(&self) -> &[&str] {
        &["rs"]
    }
    fn supports_phase_b(&self) -> bool {
        true // Native Phase B always available; LSIF enrichment is opt-in
    }
    fn parse(&self, req: &ParseRequest) -> ParseResponse {
        match travsr_indexer::rust_parse(&req.corpus, &req.path, &req.vname_path) {
            Ok(out) => parse_output_to_response(out),
            Err(e) => {
                tracing::warn!("rust parse {}: {e}", req.path.display());
                ParseResponse::default()
            }
        }
    }
    fn invoke_phase_b(&self, req: &InvokeRequest) -> InvokeResponse {
        use travsr_indexer::{ingest_rust, run_ra_lsif, sandbox::SandboxConfig};

        // Convert pre-walked relative paths (P6 — #329) to (abs, vname) pairs for the extractor.
        let files_owned: Option<Vec<(std::path::PathBuf, String)>> = req
            .files
            .as_ref()
            .map(|rel| rel.iter().map(|r| (req.root.join(r), r.clone())).collect());

        // Native Phase B: always runs, zero external-tool requirements.
        let (mut nodes, mut edges, mut unresolved_calls, mut refs) =
            travsr_indexer::phase_b_native_rust(&req.corpus, &req.root, files_owned.as_deref())
                .unwrap_or_else(|e| {
                    tracing::warn!("rust native phase_b: {e}");
                    (vec![], vec![], vec![], vec![])
                });
        tracing::debug!(
            nodes = nodes.len(),
            edges = edges.len(),
            unresolved_calls = unresolved_calls.len(),
            "rust native phase_b complete"
        );

        // LSIF enrichment: merge higher-fidelity edges when rust-analyzer is available.
        let cfg = SandboxConfig {
            repo_root: req.root.clone(),
            allow_unsandboxed: travsr_indexer::sandbox::allow_unsandboxed_opt_in(),
            ..Default::default()
        };
        match run_ra_lsif(&req.root, &cfg) {
            Ok(Some(dump)) => match ingest_rust(&dump, &req.corpus) {
                Ok(lsif_out) => {
                    tracing::debug!(
                        nodes = lsif_out.nodes.len(),
                        edges = lsif_out.edges.len(),
                        "rust lsif enrichment merged"
                    );
                    nodes.extend(lsif_out.nodes);
                    edges.extend(lsif_out.edges);
                }
                Err(e) => tracing::warn!("rust lsif ingest: {e}"),
            },
            Ok(None) => tracing::debug!("rust-analyzer not available, native phase_b only"),
            Err(e) => tracing::warn!("rust-analyzer failed: {e}"),
        }

        // Dedup merged output
        nodes.sort_unstable_by_key(|n| n.id);
        nodes.dedup_by_key(|n| n.id);
        edges.sort_unstable_by_key(|e| (e.src, e.dst));
        edges.dedup_by(|a, b| a.src == b.src && a.dst == b.dst && a.kind == b.kind);
        // Keep caller_line in the dedup key (#299) so every distinct call site
        // survives to edge_sites, not just the first per (caller, callee).
        unresolved_calls.sort_unstable_by(|a, b| {
            a.src
                .0
                .cmp(&b.src.0)
                .then(a.callee_sig.cmp(&b.callee_sig))
                .then(a.caller_line.cmp(&b.caller_line))
        });
        unresolved_calls.dedup_by(|a, b| {
            a.src == b.src && a.callee_sig == b.callee_sig && a.caller_line == b.caller_line
        });
        // #299: native same-file call occurrence lines flow as ScipRefs so the
        // daemon's G2 attributed-write records edge_sites for find_references.
        refs.sort_unstable_by(|a, b| {
            a.caller_path
                .cmp(&b.caller_path)
                .then(a.caller_line.cmp(&b.caller_line))
                .then(a.callee_id.0.cmp(&b.callee_id.0))
        });
        refs.dedup_by(|a, b| {
            a.caller_path == b.caller_path
                && a.caller_line == b.caller_line
                && a.callee_id == b.callee_id
        });

        InvokeResponse {
            diagnostics: Vec::new(),
            nodes,
            edges,
            refs,
            unresolved_calls,
        }
    }
}
