//! `travsr pattern` — graph-scoped textual search (#299).
//!
//! Thin wrapper over the MCP `find_pattern` tool: a `git grep` confined to a
//! bounded file set (whole repo, a path prefix, or `files-importing(<symbol>)`).

use anyhow::Context as _;

use crate::daemon_client;
use crate::repo::find_git_root;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text (default): `path:line:col: text` lines.
    Text,
    /// JSON envelope: `{ "pattern", "scope", "output" }` for scripting.
    Json,
}

pub fn run(
    pattern: &str,
    scope: Option<String>,
    fixed: bool,
    format: OutputFormat,
) -> anyhow::Result<()> {
    if pattern.trim().is_empty() {
        anyhow::bail!("pattern must not be empty; try: travsr pattern FlagSourceHash");
    }
    let cwd = std::env::current_dir().context("getting current directory")?;
    let repo_root = find_git_root(&cwd)?;
    let db_path = repo_root.join(".travsr/graph.db");
    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    let store = daemon_client::open_read_store(&db_path)?;

    // `find_pattern` greps the *served* index's file set, so from a linked
    // worktree served by another checkout every `path:line` printed below is
    // that tree's, rendered as a repo-relative path that reads as this one's.
    // Only this note applies here, and only its "which tree" half: the search
    // reads no call edges, so the Phase B freshness warning would be beside the
    // point — hence `reads_call_edges: false`, which also keeps that claim from
    // riding in as the cross-checkout note's degraded caveat.
    daemon_client::warn_if_cross_checkout(&db_path, false);

    // UX-009: use the raw (un-enveloped, un-escaped) pattern body. The plain
    // `find_pattern` returns the model-facing `<travsr-data>` envelope with
    // source text HTML-entity escaped (`->` → `-&gt;`); that is wrong for a
    // human terminal. `references` / `graph` already print raw — match them.
    let output = travsr_mcp::find_pattern_raw(&store, pattern, scope.as_deref(), fixed);

    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::json!({
                    "pattern": pattern,
                    "scope": scope,
                    "fixed": fixed,
                    "output": output,
                })
            );
        }
        OutputFormat::Text => {
            if output.trim().is_empty() {
                println!("no matches for '{pattern}'");
            } else {
                println!("{output}");
            }
        }
    }
    Ok(())
}
