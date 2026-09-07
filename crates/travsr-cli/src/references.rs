//! `travsr refs` — enumerate every use site (`path:line`) of a symbol (#299).
//!
//! Thin wrapper over the same occurrence-store read the MCP `find_references`
//! tool performs. Opens the repo's store read-only and prints the result.

use anyhow::Context as _;

use crate::daemon_client;
use crate::repo::find_git_root;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text (default): a resolved header + `path:line` lines.
    Text,
    /// Structured JSON (`symbol`, `resolved_to`, `references[]`, …) for scripting.
    Json,
}

/// #661 WS-D: the caller's live short HEAD, read at `cwd` (before the worktree
/// redirect in `find_git_root`, so a linked worktree reports its own commit).
/// `None` when git is unavailable or the dir is not a repo — the mismatch note
/// then correctly never fires. Mirrors `status::head_at`.
fn head_at(cwd: &std::path::Path) -> Option<String> {
    // Bounded: an unbounded `output()` here can never return on Windows when a
    // git child or grandchild inherits the pipe (#717 triage, same mechanism as
    // #503 / #572). A HEAD that does not arrive is the same as no HEAD, which
    // this function already handles.
    // `cwd` goes through as a real path rather than `-C <string>`: a path with
    // bytes that are not valid UTF-8 is legal, and converting it to a string
    // first would mangle it into U+FFFD and lose a repo that exists.
    crate::git_bounded::git_stdout_bounded(Some(cwd), ["rev-parse", "--short", "HEAD"])
}

/// Extract the content between `<travsr-data>` and `</travsr-data>`. Mirrors
/// `pattern::envelope_body`; falls back to the whole string if unwrapped.
fn envelope_body(output: &str) -> &str {
    output
        .strip_prefix("<travsr-data>")
        .and_then(|s| s.strip_suffix("</travsr-data>"))
        .map(|s| s.trim_matches('\n'))
        .unwrap_or(output)
}

pub fn run(symbol: &str, path: Option<String>, format: OutputFormat) -> anyhow::Result<()> {
    if symbol.trim().is_empty() {
        anyhow::bail!("symbol must not be empty; try: travsr refs PaymentService");
    }
    let cwd = std::env::current_dir().context("getting current directory")?;
    // #661 WS-D: read HEAD at cwd before the worktree redirect so a drifted
    // checkout is compared against the served index below.
    //
    // Run concurrently with `find_git_root` below: both are independent,
    // bounded git queries on `cwd` (the latter only shells out in the
    // linked-worktree branch, via `main_worktree_root`). Sequentially, a wedged
    // git would let this command stall for up to 2x `GIT_QUERY_TIMEOUT` instead
    // of 1x. Mirrors `status::run`.
    let head_handle = {
        let cwd = cwd.clone();
        std::thread::spawn(move || head_at(&cwd))
    };
    let repo_root = find_git_root(&cwd)?;
    let head = head_handle.join().ok().flatten();
    let db_path = repo_root.join(".travsr/graph.db");
    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    let cross_checkout = daemon_client::warn_if_call_graph_degraded(&db_path);
    let store = daemon_client::open_read_store(&db_path)?;

    match format {
        OutputFormat::Json => {
            // Structured result: `symbol`, `resolved_to`, and a real `references`
            // array — not the human-readable envelope body wrapped in a string.
            let structured =
                travsr_mcp::find_references_structured(&store, symbol, path.as_deref());
            println!("{}", serde_json::to_string(&structured)?);
        }
        OutputFormat::Text => {
            // Decide not-found from the resolver outcome, never by sniffing the
            // rendered body: an empty body can be a drift note (`with_head_note`)
            // or a validation-reject string, and a genuine not-found can carry a
            // note — so `body.is_empty()` is not a reliable signal.
            let structured =
                travsr_mcp::find_references_structured(&store, symbol, path.as_deref());
            if structured.status == "not_found" && structured.candidates.is_empty() {
                // Resolver saw no definition (or rejected the argument). Its own
                // note is the definitive not-found line or an invalid-argument
                // caveat — distinct from a resolved symbol with zero recorded
                // uses, which prints its `resolved: … 0 reference(s)` line below.
                match &structured.note {
                    Some(note) => println!("{note}"),
                    None => println!("Symbol '{symbol}' was not found."),
                }
            } else {
                // Resolved / ambiguous / pending / path-miss-with-candidates:
                // `find_references` returns the model-facing `<travsr-data>`
                // envelope. That is noise in CLI output — `ask` and `graph`
                // already strip it — so present the inner body to the user.
                let output = travsr_mcp::find_references(&store, symbol, path.as_deref());
                println!("{}", envelope_body(&output));
            }
        }
    }

    // #661 WS-D: warn (on stderr, keeping stdout/JSON clean) when the served
    // index describes a different commit than the caller's checkout, so a
    // confident `path:line` list on a drifted worktree is never taken at face
    // value. cwd-local classifier, identical wording to `travsr status` and the
    // MCP tools (`travsr_mcp::head_index_mismatch_note`).
    //
    // Skipped when this is a linked worktree served by another checkout:
    // `warn_if_call_graph_degraded` above already named both roots definitively,
    // and this note would follow it with a guess ("expected in a linked
    // worktree; otherwise ... wait for the daemon to reconcile") whose advice is
    // wrong in exactly that case. Keyed to `cross_checkout`, the bool that call
    // returned, so the suppression tracks the note that was actually emitted
    // rather than a fresh re-evaluation of the same predicate.
    let stored = store
        .get_meta("last_commit")
        .ok()
        .flatten()
        .unwrap_or_default();
    if let Some(head) = head.as_deref().filter(|_| !cross_checkout) {
        if let Some(note) = travsr_mcp::head_index_mismatch_note(head, &stored) {
            eprintln!("{note}");
        }
    }
    Ok(())
}
