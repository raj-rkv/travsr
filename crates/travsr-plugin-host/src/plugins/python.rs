use super::parse_output_to_response;
use travsr_core::Language;
use travsr_plugin_protocol::{InvokeRequest, InvokeResponse, ParseRequest, ParseResponse, Plugin};

pub struct PythonPlugin;

impl Plugin for PythonPlugin {
    fn language(&self) -> Language {
        Language::Python
    }
    fn extensions(&self) -> &[&str] {
        &["py", "pyi"]
    }
    fn supports_phase_b(&self) -> bool {
        true
    }
    fn parse(&self, req: &ParseRequest) -> ParseResponse {
        match travsr_indexer::python_parse(&req.corpus, &req.path, &req.vname_path) {
            Ok(mut out) => {
                // Best-effort pyright enrichment (same as old Indexer path)
                let pyright = travsr_indexer::python_lsif::parse_python_with_pyright(
                    &req.path,
                    std::time::Duration::from_secs(30),
                )
                .unwrap_or_default();
                out.merge_deduped(pyright);
                parse_output_to_response(out)
            }
            Err(e) => {
                tracing::warn!("python parse {}: {e}", req.path.display());
                ParseResponse::default()
            }
        }
    }
    fn invoke_phase_b(&self, req: &InvokeRequest) -> InvokeResponse {
        // Convert pre-walked relative paths (P6 — #329) to (abs, vname) pairs for the extractor.
        let files_owned: Option<Vec<(std::path::PathBuf, String)>> = req
            .files
            .as_ref()
            .map(|rel| rel.iter().map(|r| (req.root.join(r), r.clone())).collect());

        // Native Phase B: always runs, zero external-tool requirements.
        let (mut nodes, mut edges, unresolved_calls) =
            travsr_indexer::phase_b_native_python(&req.corpus, &req.root, files_owned.as_deref())
                .unwrap_or_else(|e| {
                    tracing::warn!("python native phase_b: {e}");
                    (vec![], vec![], vec![])
                });
        tracing::debug!(
            nodes = nodes.len(),
            edges = edges.len(),
            unresolved_calls = unresolved_calls.len(),
            "python native phase_b complete"
        );

        // LSIF enrichment via travsr-lsif-py (bundled, PATH-independent).
        // Resolves via current_exe walk-up; Ok(None) = not yet built / not found.
        match travsr_indexer::run_lsif_py_emitter(&req.root) {
            Ok(Some(dump)) => match travsr_indexer::ingest_lsif(&dump, &req.corpus) {
                Ok(lsif_out) => {
                    tracing::debug!(
                        nodes = lsif_out.nodes.len(),
                        edges = lsif_out.edges.len(),
                        "python lsif enrichment merged"
                    );
                    nodes.extend(lsif_out.nodes);
                    edges.extend(lsif_out.edges);
                }
                Err(e) => tracing::warn!("python lsif ingest: {e}"),
            },
            Ok(None) => {
                tracing::debug!("travsr-lsif-py not found, native phase_b tree-sitter edges only")
            }
            Err(e) => tracing::warn!("travsr-lsif-py failed: {e}"),
        }

        // Dedup merged output
        nodes.sort_unstable_by_key(|n| n.id);
        nodes.dedup_by_key(|n| n.id);
        edges.sort_unstable_by_key(|e| (e.src, e.dst));
        edges.dedup_by(|a, b| a.src == b.src && a.dst == b.dst && a.kind == b.kind);

        InvokeResponse {
            diagnostics: Vec::new(),
            nodes,
            edges,
            unresolved_calls,
            ..Default::default()
        }
    }
}
