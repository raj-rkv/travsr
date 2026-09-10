use super::parse_output_to_response;
use travsr_core::Language;
use travsr_plugin_protocol::{InvokeRequest, InvokeResponse, ParseRequest, ParseResponse, Plugin};

pub struct TypeScriptPlugin;

impl Plugin for TypeScriptPlugin {
    fn language(&self) -> Language {
        Language::TypeScript
    }
    fn extensions(&self) -> &[&str] {
        &["ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs"]
    }
    fn supports_phase_b(&self) -> bool {
        true
    } // travsr-lsif-ts
    fn parse(&self, req: &ParseRequest) -> ParseResponse {
        if let Some(src_bytes) = &req.source {
            // git-blob path: write to tempfile then parse
            use std::io::Write;
            let ext = req
                .path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("ts");
            match tempfile::Builder::new()
                .suffix(&format!(".{ext}"))
                .tempfile()
            {
                Ok(mut tmp) => {
                    let _ = tmp.write_all(src_bytes);
                    let _ = tmp.flush();
                    match travsr_indexer::typescript_parse(&req.corpus, tmp.path(), &req.vname_path)
                    {
                        Ok(out) => return parse_output_to_response(out),
                        Err(e) => {
                            tracing::warn!("ts parse (blob): {e}");
                            return ParseResponse::default();
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("tempfile: {e}");
                    return ParseResponse::default();
                }
            }
        }
        match travsr_indexer::typescript_parse(&req.corpus, &req.path, &req.vname_path) {
            Ok(out) => parse_output_to_response(out),
            Err(e) => {
                tracing::warn!("ts parse {}: {e}", req.path.display());
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
        let (mut nodes, mut edges, unresolved_calls) = travsr_indexer::phase_b_native_typescript(
            &req.corpus,
            &req.root,
            files_owned.as_deref(),
        )
        .unwrap_or_else(|e| {
            tracing::warn!("ts native phase_b: {e}");
            (vec![], vec![], vec![])
        });
        tracing::debug!(
            nodes = nodes.len(),
            edges = edges.len(),
            unresolved_calls = unresolved_calls.len(),
            "ts native phase_b complete"
        );

        // LSIF enrichment: merge higher-fidelity edges when travsr-lsif-ts is available.
        let tsconfig = req.root.join("tsconfig.json");
        if tsconfig.exists() {
            match travsr_indexer::run_lsif_emitter(&tsconfig) {
                Ok(dump) => match travsr_indexer::ingest_lsif(&dump, &req.corpus) {
                    Ok(lsif_out) => {
                        tracing::debug!(
                            nodes = lsif_out.nodes.len(),
                            edges = lsif_out.edges.len(),
                            "ts lsif enrichment merged"
                        );
                        nodes.extend(lsif_out.nodes);
                        edges.extend(lsif_out.edges);
                    }
                    Err(e) => tracing::warn!("ts lsif ingest: {e}"),
                },
                Err(e) => tracing::debug!("ts lsif emitter not available: {e}"),
            }
        }

        // #833: `.js`/`.jsx`/`.mjs`/`.cjs` are classified as TypeScript, so they
        // reach this plugin, but the project tsconfig above (if any) only covers
        // them with `allowJs` — which plain-JS / CommonJS repos never set, and
        // most have no tsconfig at all. Run a second pass over a synthesized
        // allowJs tsconfig covering exactly this repo's JS files. A no-op when
        // there are no JS files; idempotent where a real allowJs tsconfig
        // already covered them (the dedup below drops the overlap).
        //
        // `req.files` is None only on the legacy "sidecar walks itself"
        // protocol path; every `init --semantic` supplies indexable_paths, so
        // the JS pass simply does not run there.
        if let Some(rel_files) = req.files.as_ref() {
            let js_abs: Vec<std::path::PathBuf> = rel_files
                .iter()
                .filter(|r| {
                    std::path::Path::new(r.as_str())
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| travsr_indexer::JS_EXTENSIONS.contains(&e))
                })
                .map(|r| req.root.join(r))
                .collect();
            match travsr_indexer::synthesize_js_tsconfig(&js_abs) {
                Ok(Some((_scratch, synth_tsconfig))) => {
                    match travsr_indexer::run_lsif_emitter_with_root(&synth_tsconfig, &req.root) {
                        Ok(dump) => match travsr_indexer::ingest_lsif(&dump, &req.corpus) {
                            Ok(lsif_out) => {
                                tracing::debug!(
                                    nodes = lsif_out.nodes.len(),
                                    edges = lsif_out.edges.len(),
                                    "js synthesized-tsconfig lsif enrichment merged"
                                );
                                nodes.extend(lsif_out.nodes);
                                edges.extend(lsif_out.edges);
                            }
                            Err(e) => tracing::warn!("js lsif ingest: {e}"),
                        },
                        // Only a failure to *start* the emitter is "not
                        // available". One that ran and failed (a pre-`--root`
                        // emitter hits SEC-003 here) is a real fault and must be
                        // visible at default verbosity, stderr head included.
                        Err(e) if travsr_indexer::emitter_missing(&e) => {
                            tracing::debug!("js lsif emitter not available: {e}")
                        }
                        Err(e) => tracing::warn!("js lsif emitter failed: {e:#}"),
                    }
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("js synthetic tsconfig: {e}"),
            }
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
