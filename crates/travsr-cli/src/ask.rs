//! `travsr ask` — PPR-ranked, knapsack-budgeted symbol context.
//!
//! Data acquisition is shared with the daemon via `travsr_mcp::query`
//! (#318 O1): a running daemon answers from its warm store; otherwise the
//! store is opened directly (read-only fast path).

use anyhow::Context as _;
use tabled::{settings::Style, Table, Tabled};
use travsr_mcp::query::{self, AskPayload};

use crate::daemon_client;
use crate::repo::find_git_root;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table (default)
    Table,
    /// Machine-readable JSON — emits the full AskPayload
    Json,
}

/// One rendered result line. There is no `Kind` column (#824): most signatures
/// already carry their kind as a prefix (`fn:`, `method:`, `field:`, `var:`,
/// ...), so the column just repeated it. The kind is folded into `signature`
/// instead, and only for the rows that do not already show it — see
/// [`signature_shows_kind`].
#[derive(Tabled)]
struct Row {
    #[tabled(rename = "Signature")]
    signature: String,
    #[tabled(rename = "Path")]
    path: String,
    #[tabled(rename = "Score")]
    score: String,
}

/// Render results as a borderless, space-aligned table (#824). The default
/// `tabled` style drew a `+---+` rule between every row and boxed each cell in
/// `|` bars, which roughly tripled the stdout an agent piping `ask` receives
/// for output no human reads. Columns still align; only the frame is dropped.
/// The machine surface is `--format json`, so this affects the human view only.
///
/// `Style::blank()` pads the last column, so every line would otherwise end in
/// a space: invisible in a terminal, but still a byte per row in the stdout
/// this change exists to shrink, and whitespace noise the moment anyone diffs
/// captured output. Trim it back off.
fn render_rows(rows: Vec<Row>) -> String {
    let table = Table::new(rows).with(Style::blank()).to_string();
    table
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The match-source lanes the grouped human table renders, in backend
/// `trust_rank` order (seed.rs `MatchSource`): exact -> semantic -> docs ->
/// tests -> relevant. Every lane the backend can stamp on a row MUST appear
/// here: the per-tag filter in `run` drops any row whose `match_source` matches
/// no listed tag, so a missing lane is silently dropped from the table (the #479
/// bug: `tests` was absent, so 24 test rows including the top-scored node
/// vanished; the JSON `rows` still carried them). `docs` renders from
/// `payload.docs`, not from `rows`.
const SECTION_TAGS: [&str; 5] = ["exact", "semantic", "docs", "tests", "relevant"];

/// THE RULE: a signature already shows its kind only when the word before its
/// first `:` is an abbreviation of the row's own kind — every letter of that
/// word appearing in order inside the kind name.
///
/// That is language-agnostic because it compares the signature against the kind
/// the indexer stamped on the same node rather than against a list of known
/// prefixes: `fn:`/function, `var:`/variable, `pkg:`/package, `filemod:`/
/// file-module and every other Phase A and Kotlin prefix pass, while the Phase B
/// provider schemes that survive G1 unification — SCIP `scip:<path>:<symbol>`
/// (Java/Go/C#/C/C++/Scala/PHP/Ruby/ObjC), the Swift emitter's `swift::Type.member`
/// and the Dart emitter's `file:///abs/x.dart::Type.member` — do not, because
/// `scip`/`swift`/`file` are not subsequences of `function`, `method`, `class`
/// or any other kind. A bare path (a file node) and a Windows `C:\src\x.rs`
/// have no kind-like prefix either, so they keep their kind too. #824's report
/// came from an iOS/Swift run, which is exactly the set this protects.
fn signature_shows_kind(signature: &str, kind: &str) -> bool {
    let Some((prefix, _)) = signature.split_once(':') else {
        return false;
    };
    if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let mut kind_chars = kind.chars();
    prefix
        .chars()
        .all(|p| kind_chars.any(|k| k.eq_ignore_ascii_case(&p)))
}

fn to_row(r: &query::AskRow) -> Row {
    Row {
        // Keep the signature verbatim — it is what a user copies into
        // `travsr graph` / `travsr references` — and only prepend the kind when
        // the signature does not already carry it.
        signature: if r.kind.is_empty() || signature_shows_kind(&r.signature, &r.kind) {
            r.signature.clone()
        } else {
            format!("{}: {}", r.kind, r.signature)
        },
        path: match r.line {
            Some(l) => format!("{}:{}", r.path, l),
            None => r.path.clone(),
        },
        score: format!("{:.3}", r.score),
    }
}

/// #376 §4.1: print the docs section, or nothing at all when it is empty
/// ("absent, not empty-ish" — a section that appears on every query trains the
/// reader to ignore it).
///
/// Plain lines, no table, and no score column: doc scores and code scores are
/// not commensurable (§8.3 measured doc cosines above each repo's code band),
/// so the header carries the epistemic weight instead of a number. The lines
/// arrive already sanitized from `travsr_mcp::query` — this function must not
/// reformat them in a way that could re-introduce what the sanitizer removed.
fn print_docs(docs: &[String]) {
    if docs.is_empty() {
        return;
    }
    println!(
        "── docs, documentation prose: claims about the code, verify behaviour against the code itself ──"
    );
    for line in docs {
        println!("{line}");
    }
}

/// #376 G1 / §18.7: `TRAVSR_DOCS_ENABLED` is read by whichever process performs
/// retrieval, and for `ask` that is never this one. When the daemon serves the
/// query it reads the flag from its own environment; when it does not, the cold
/// read-only path deliberately never arms the doc KNN hook. So setting the flag
/// on this command looks like it applies and silently does nothing — the exact
/// shape of the #516 bug, where a lane that was wired correctly rendered nothing
/// and reported no error.
///
/// Deliberately phrased as where-to-set-it rather than "docs are off": a query
/// that legitimately matched no doc above the floor renders nothing either
/// (§4.2, "absent, not empty-ish"), and this function cannot tell the two apart
/// from inside the CLI process. Only fires when the user actually set the
/// variable, and goes to stderr so `--format json` stays machine-readable.
///
/// #376 O1 update: there is now a *working* switch to point at.
/// `travsr config set docs.enabled true` writes the repo's
/// `.travsr/config.toml`, which the daemon reads regardless of the environment
/// it was started in, so the note recommends that over restarting the daemon
/// with an exported variable.
fn note_docs_flag_is_read_by_the_daemon() {
    if std::env::var_os("TRAVSR_DOCS_ENABLED").is_none() {
        return;
    }
    eprintln!(
        "note: TRAVSR_DOCS_ENABLED is read by the process that performs retrieval, \
         which for `ask` is the travsr daemon, not this command. Prefer the \
         config key, which the daemon reads whatever environment it was started \
         in: `travsr config set docs.enabled true`."
    );
}

/// #376 O7 / G7: with no daemon running, `ask` is served by the cold read-only
/// path, which can never render a docs section — and said nothing about it.
///
/// The silence was the whole problem: an empty docs section is also what a
/// correctly-working lane produces when nothing cleared the floor (§4.2,
/// "absent, not empty-ish"), so a user who enabled the lane and got nothing had
/// no way to tell "no doc matched" from "this code path structurally cannot
/// answer you". Two of #376's shipped bugs (#516, and the CLI env-var no-op)
/// were the same shape.
///
/// The fix is to say so rather than to arm the hook. Arming it was measured and
/// rejected: [`travsr_daemon::try_inject_embed_hook_readonly`] documents that a
/// per-invocation sidecar costs ~0.6 s of model load, which overruns
/// `ask_query`'s own 600 ms circuit breaker, so the seeds would be discarded
/// anyway while every `ask` churned a throwaway 127 MB-model process.
///
/// Only fires when the lane is actually enabled for this repo — a user who
/// never turned docs on is not owed a message about them — and writes to stderr
/// so `--format json` stays machine-readable.
fn note_cold_path_cannot_render_docs(repo_root: &std::path::Path) {
    // #519: must match travsr-mcp::seed::docs_enabled's own default (now
    // true) - otherwise this warning silently stops firing for a cold-path
    // user on the new default, the exact silent-failure shape this function
    // exists to prevent.
    let enabled = travsr_config::effective_bool("docs.enabled", Some(repo_root)).unwrap_or(true);
    if !enabled {
        return;
    }
    eprintln!(
        "note: docs.enabled is on, but no travsr daemon is running, this query \
         is served by the read-only cold path, which does not load the doc \
         index, so no docs section can appear. Start the daemon with \
         `travsr daemon start`."
    );
}

/// Answer a FAQ entry in whichever shape the caller asked for.
///
/// Every path out of `ask` has to honour `--format json`. Three early returns
/// were added without it, so `ask --format json` could answer a machine with
/// coloured prose and exit 0. `benchmarks/ab-eval/run.js` parses that output,
/// and it survived only because every task in `tasks.json` is a bare symbol
/// name that the FAQ matcher declines; the first natural-language task added
/// there would have broken it, which puts the failure on whoever adds a
/// benchmark rather than on the change that caused it.
fn answer_faq(e: &crate::faq::Entry, format: OutputFormat) -> anyhow::Result<()> {
    use std::io::IsTerminal as _;
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                // `matched` is mandatory on the ordinary payload, so consumers
                // branch on it. Omitting it made ab-eval read a FAQ answer as a
                // zero-recall retrieval result: a silent miss that looks like a
                // search regression rather than a routing one (#746 review).
                "matched": false,
                "kind": "faq",
                "question": e.question,
                "lead": e.lead,
                "detail": e.detail,
                "points": e.points,
                "commands": e.commands,
            })
        );
        return Ok(());
    }
    let pal = crate::progress::Palette::for_stream(std::io::stdout().is_terminal());
    crate::faq::print_entry(e, pal);
    Ok(())
}

pub fn run(query_str: &str, format: OutputFormat) -> anyhow::Result<()> {
    if query_str.trim().is_empty() {
        anyhow::bail!("search query must not be empty; try: travsr ask \"PaymentService\"");
    }
    // The explicit route, checked before the repository is even located: a
    // question about travsr is answerable with no index, which is exactly when
    // someone is most likely to ask one. `match_question` still handles the
    // natural phrasing; this is the spelling that cannot be ambiguous, and the
    // one the agent guidance points at.
    if let Some(rest) = crate::faq::strip_namespace(query_str) {
        match crate::faq::match_namespaced(rest) {
            Some(e) => answer_faq(e, format)?,
            // Listing the questions beats "no match": the reader has already
            // said what kind of answer they want, so show what is on offer.
            None if matches!(format, OutputFormat::Json) => println!(
                "{}",
                serde_json::json!({
                    "kind": "faq",
                    "matched": false,
                    "questions": crate::faq::questions().collect::<Vec<_>>(),
                })
            ),
            None => {
                use std::io::IsTerminal as _;
                crate::faq::print_questions(crate::progress::Palette::for_stream(
                    std::io::stdout().is_terminal(),
                ))
            }
        }
        return Ok(());
    }

    // Matched before the repository is located, not after. This used to sit
    // below `find_git_root` and the `graph.db` existence check, so
    // `travsr ask "what is travsr?"` in a fresh clone answered
    // "not initialized, run travsr init", and outside a git repo it failed in
    // `find_git_root`. That is precisely the reader the catalogue is written
    // for: `faq.rs` says so in its own module doc, and the code contradicted it.
    if let Some(e) = crate::faq::match_question(query_str) {
        return answer_faq(e, format);
    }

    note_docs_flag_is_read_by_the_daemon();
    let cwd = std::env::current_dir().context("getting current directory")?;
    let repo_root = find_git_root(&cwd)?;
    let db_path = repo_root.join(".travsr/graph.db");

    if !db_path.exists() {
        anyhow::bail!("not initialized; run `travsr init`");
    }

    // A question about the repository as a whole is not one the graph can
    // answer. `ask` seeds from the words in the query, so "what is this repo
    // written in?" matches `var:REPO` and returns a screen of bench files with
    // confident-looking scores. That is worse than abstaining: the answer exists
    // one command away, and the result looks like a real one.
    //
    // Caught before retrieval rather than after, because retrieval succeeds
    // here. There is no low-confidence signal to hang an abstention on.
    if let Some(redirect) = meta_question_redirect(query_str) {
        use std::io::IsTerminal as _;
        // In JSON mode, name the command instead of running it. This branch
        // delegates to another command's *human* renderer, with `json: false`
        // hardcoded for `lang list` and no JSON mode at all in `status`, so
        // running it would answer a machine-readable request with a table.
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::json!({
                    "matched": false,
                    "kind": "redirect",
                    "answer": redirect.answer,
                    "command": redirect.command,
                })
            );
            return Ok(());
        }
        let pal = crate::progress::Palette::for_stream(std::io::stdout().is_terminal());
        // Answer here rather than naming another command. Someone who typed the
        // question has already said what they want to know, and "run this other
        // thing" is a worse response than the answer itself.
        match redirect.command {
            "travsr lang list" => {
                println!("{}", pal.dim(redirect.answer));
                crate::lang::run(crate::lang::LangCommand::List { json: false })?;
            }
            "travsr status" => {
                println!("{}", pal.dim(redirect.answer));
                crate::status::run()?;
            }
            other => {
                println!("{}", redirect.answer);
                println!("  {} {}", pal.dim("$"), pal.ident(other));
            }
        }
        return Ok(());
    }
    // Before either path: `ask` ranks over call edges, so an incomplete Phase B
    // changes the answer, not just its completeness.
    daemon_client::warn_if_call_graph_degraded(&db_path);

    let mut served_cold_path = false;
    let mut cold_path_embed_unarmed = false;
    let payload: AskPayload = match daemon_client::try_query(
        &repo_root,
        "ask",
        serde_json::json!({ "query": query_str }),
    ) {
        Some(p) => p,
        None => {
            served_cold_path = true;
            let mut store = daemon_client::open_read_store(&db_path)?;
            // Best-effort: load HNSW embed hook for cold-path KNN. Falls back to
            // FTS-only if the sidecar binary is absent or the index is not built.
            travsr_daemon::try_inject_embed_hook_readonly(&mut store, &db_path);
            let knn = store.embed_knn_fn();
            // An embed.db sibling exists and this process still has no hook, so
            // ranking here is lexical only. `knn.is_none()` is redundant today
            // (`try_inject_embed_hook_readonly` is currently a no-op, so `knn`
            // is always `None` on this path); it is kept because it is what the
            // flag actually means, and dropping it would silently start lying
            // the day that injector gains a body.
            cold_path_embed_unarmed = store.has_embed_db() && knn.is_none();
            let knn_ref = knn
                .as_ref()
                .map(|f| f as &dyn Fn(&str, u32) -> Vec<(travsr_core::NodeId, f32)>);
            query::ask_query(&store, query_str, knn_ref)?
        }
    };

    // UX-010: only nudge about the missing doc index when the cold path actually
    // produced a grounded answer — a real result a docs section could have
    // augmented. On an abstention or an empty result (including nonsense queries)
    // a docs section would not have helped, so the note was pure recurring noise.
    if served_cold_path && payload.matched && !payload.no_results {
        note_cold_path_cannot_render_docs(&repo_root);
    }
    // Unlike the docs note, this one matters most when the query FAILED: the
    // payload's own signal for this state reads "embedding in progress; run
    // `travsr embed status`", which sends the user to a command that may well
    // report the index complete. The cold path declines to arm the hook, by
    // design, and only this process knows that. What it does NOT know is why no
    // daemon answered, or whether the embedding index is finished, so the note
    // below claims neither.
    if served_cold_path && cold_path_embed_unarmed {
        eprintln!(
            "note: this repo has an embedding index, but this query was not served \
             by a daemon, so it fell back to the read-only cold path, which does \
             not load the embedding sidecar. The answer is lexical only. Run \
             `travsr daemon status` to see whether a daemon is available for \
             semantic ranking."
        );
    }

    if matches!(format, OutputFormat::Json) {
        println!("{}", serde_json::to_string(&payload)?);
        return Ok(());
    }

    let display_query = query_str.strip_prefix(':').unwrap_or(query_str).trim();
    if !payload.matched {
        // `ask` is natural-language, graph-grounded retrieval — not a symbol-name
        // lookup — so an abstention means "no confidently relevant code found",
        // not "that symbol does not exist". Word it that way to avoid the misread.
        // "try rephrasing" is not actionable on its own: a user who does not
        // already know what travsr answers cannot tell what to rephrase towards,
        // and this path fires most often on exactly the conceptual questions
        // where they are least sure.
        //
        // A static list of examples is barely better, because the user still has
        // to translate their own question into one of them. So the suggestions
        // below are built from what they actually typed: the nearest real symbol
        // in this repo, and the command that matches the intent their wording
        // signals. Falls back to the catalogue only when nothing specific can be
        // said, rather than leading with it.
        println!("no grounded match for '{display_query}' in this repo");
        let suggestions = suggest_next(&db_path, display_query);
        use std::io::IsTerminal as _;
        if suggestions.is_empty() {
            println!("run `travsr ask --examples` to see what travsr can answer");
        } else {
            let pal = crate::progress::Palette::for_stream(std::io::stdout().is_terminal());
            println!("\ntry one of these:");
            for s in &suggestions {
                println!("  {}", paint_command(pal, &s.command, &s.ident));
                println!("      {}", pal.dim(&s.why));
            }
            println!("\n{}", pal.dim("or `travsr ask --examples` for more"));
        }
        // #376 §4.3: doc hits may appear below the abstain message, but never
        // convert it into a match — `payload.matched` stays false and no
        // confidence, coverage or tier label is derived from them. This is the
        // highest-value case for the lane: §8.5 measured 15/20 k8s rationale
        // queries as hard abstentions with a citable doc section available.
        if !payload.docs.is_empty() {
            println!();
            print_docs(&payload.docs);
        }
        // #826: an abstention may be a setup gap rather than a genuine absence,
        // so say which one. `degraded_note` is the per-query signal the matched
        // path already prints (and `--format json` already carries): it names
        // `travsr embed init` only when semantic search really did not run for
        // THIS query in THIS repo, and says "warming up" / "in progress" /
        // "degraded" in the states where that command is the wrong advice.
        if !payload.degraded_note.is_empty() {
            println!("\n{}", payload.degraded_note);
        }
        return Ok(());
    }
    if payload.no_results {
        println!("no graph results for '{display_query}'");
        if !payload.docs.is_empty() {
            println!();
            print_docs(&payload.docs);
        }
        // #826: same reasoning as the abstain branch above.
        if !payload.degraded_note.is_empty() {
            println!("\n{}", payload.degraded_note);
        }
        return Ok(());
    }

    let n = payload.rows.len();
    // RFC-022 §14: when match-source grouping is on (rows carry `match_source`)
    // and the result is large enough (N>4) that section headers pay for themselves,
    // print one table per Exact → Semantic → Docs → Tests → Relevant section.
    // Otherwise a single flat table (unchanged default). This never reorders the
    // JSON `rows` (that path returned above); it only regroups the human table.
    let grouped = n > 4 && payload.rows.iter().any(|r| r.match_source.is_some());
    if grouped {
        for tag in SECTION_TAGS {
            // #376 §4.1 section order: exact → semantic → docs → relevant. Docs
            // sit above `relevant` because design intent beats graph-adjacent
            // filler, and below the code sections so code leads whenever it has
            // an answer. Docs are not `rows`, so this arm is not a filter.
            if tag == "docs" {
                print_docs(&payload.docs);
                continue;
            }
            let mut section: Vec<&query::AskRow> = payload
                .rows
                .iter()
                .filter(|r| r.match_source.as_deref() == Some(tag))
                .collect();
            if section.is_empty() {
                continue;
            }
            section.sort_by(|a, b| b.score.total_cmp(&a.score));
            // #479: cap the tests lane in the human table the same way
            // `get_context`'s `assemble_context_body` does (TESTS_CAP = 3) so a
            // test lane never dominates the summary. The JSON `rows` keep every row.
            if tag == "tests" {
                section.truncate(3);
            }
            let rows: Vec<Row> = section.iter().map(|r| to_row(r)).collect();
            // U3: titles describe WHAT the group is, not the internal mechanism.
            // The old "semantic — cross-encoder ranked" asserted a reranker pass
            // that may not have run (e.g. KNN timed out → lexical fallback),
            // contradicting the degraded note printed below.
            let header = match tag {
                "exact" => "── exact matches, literal symbol / text ──",
                "semantic" => "── related, ranked by relevance ──",
                "tests" => "── tests, test entry points & fixtures ──",
                _ => "── relevant, graph-adjacent context ──",
            };
            println!("{header}");
            println!("{}", render_rows(rows));
        }
    } else {
        let rows: Vec<Row> = payload.rows.iter().map(to_row).collect();
        println!("{}", render_rows(rows));
        print_docs(&payload.docs);
    }
    let embed_note = if payload.embed_used {
        " · [embed-enhanced]"
    } else {
        ""
    };
    // F9: surface the honest confidence label (parity with get_context's header).
    let confidence_note = if payload.confidence.is_empty() {
        String::new()
    } else {
        format!(" · confidence: {}", payload.confidence)
    };
    println!(
        "\n{n} nodes · ~{} tokens{confidence_note}{embed_note}",
        payload.total_tokens
    );
    if !payload.degraded_note.is_empty() {
        println!("{}", payload.degraded_note);
    }
    Ok(())
}

#[cfg(test)]
mod docs_note_tests {
    /// Serializes the tests below: both mutate `HOME`, which is process-global.
    static HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A hermetic repo + config environment. Redirects **both** file layers into
    /// a tempdir — `HOME` for the global one, the returned path for the repo one
    /// — because a developer with `docs.enabled = true` in their real
    /// `~/.travsr/config.toml` would otherwise flip these assertions.
    struct Env {
        dir: tempfile::TempDir,
        prev_home: Option<std::ffi::OsString>,
    }

    impl Env {
        fn new() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            let prev_home = std::env::var_os("HOME");
            std::env::set_var("HOME", dir.path().join("home"));
            std::fs::create_dir_all(dir.path().join("repo").join(".travsr")).expect("mk repo");
            Self { dir, prev_home }
        }
        fn repo(&self) -> std::path::PathBuf {
            self.dir.path().join("repo")
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            match self.prev_home.take() {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    /// #376 O7: the note must be gated on the lane actually being enabled.
    /// Printing it unconditionally would put a docs warning in front of every
    /// user who never turned docs on, which is how a note becomes noise and then
    /// becomes ignored — the failure mode §20.4 O3 flags for flaky CI gates too.
    ///
    /// #519 flipped the default to on, so the "silent" case this test pins is
    /// now an explicit opt-out rather than the default — was
    /// `cold_path_note_is_silent_when_docs_are_off`, renamed rather than
    /// edited in place.
    #[test]
    fn cold_path_note_is_silent_when_docs_are_explicitly_off() {
        let _g = HOME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = Env::new();
        assert!(
            travsr_config::effective_bool("docs.enabled", Some(&env.repo())).unwrap_or(true),
            "default must be on"
        );
        travsr_config::set(
            "docs.enabled",
            "false",
            travsr_config::Scope::Repo(env.repo()),
        )
        .expect("set");
        assert!(
            !travsr_config::effective_bool("docs.enabled", Some(&env.repo())).unwrap_or(true),
            "explicit opt-out must make note_cold_path_cannot_render_docs return early"
        );
    }

    /// And it must fire once the key is on — the condition the live check in
    /// this session exercised, pinned so a config-plumbing regression is caught
    /// here rather than by a user seeing an unexplained empty docs section.
    #[test]
    fn cold_path_note_fires_when_docs_are_on() {
        let _g = HOME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env = Env::new();
        travsr_config::set(
            "docs.enabled",
            "true",
            travsr_config::Scope::Repo(env.repo()),
        )
        .expect("set");
        assert!(
            travsr_config::effective_bool("docs.enabled", Some(&env.repo())).unwrap_or(true),
            "repo config must drive the note"
        );
    }

    /// #479 regression: every match-source lane the backend can stamp on a row
    /// must be rendered by the grouped table. The backend stamps `tests` (and
    /// `docs`, #376) alongside `exact`/`semantic`/`relevant`; the grouped renderer
    /// filters rows per `SECTION_TAGS`, so a lane missing from that list is
    /// silently dropped from the human table (the #479 bug dropped 24 `tests`
    /// rows, including the top-scored node, while the JSON path still carried
    /// them). Pin the full lane set so a future edit cannot re-introduce the drop.
    #[test]
    fn section_tags_cover_every_backend_match_source_lane() {
        for lane in ["exact", "semantic", "docs", "tests", "relevant"] {
            assert!(
                super::SECTION_TAGS.contains(&lane),
                "match-source lane {lane:?} missing from SECTION_TAGS -> its rows would be silently dropped from the ask table"
            );
        }
        // And the order matches the backend trust_rank so sections read
        // most-trusted first.
        assert_eq!(
            super::SECTION_TAGS,
            ["exact", "semantic", "docs", "tests", "relevant"]
        );
    }
}

/// The questions travsr answers, as templates over a symbol.
///
/// A flat list rather than grouped sections with notes: an earlier version read
/// as a manual page, and the point is to show what can be typed, not to document
/// the CLI. Each entry is a question a reader might actually have, paired with
/// the one command that answers it.
///
/// `(question, command)`. `{sym}` is filled with a real symbol from the reader's
/// own repository, `{term}` with a lowercased form for text search.
///
/// Diagnostics are deliberately absent. `travsr explain` describes itself as
/// "a diagnostic for tuning search; not part of normal use", and `fsck` reports
/// ghost nodes and orphan edges. Both answer questions about travsr's own
/// internals rather than about the reader's code, and a list of "questions you
/// can ask" is not where someone should first meet them.
///
/// Only `ask` shapes measured to return results are included. `ask` is
/// graph-grounded, so it answers when the question is anchored to something in
/// the code and abstains when it is not: "what breaks if I change X" returns
/// nothing today, which is the known conceptual-query gap, so that question is
/// routed to `graph` instead of being suggested as an `ask`.
///
/// "how does X work" was dropped for the same reason after testing it: it
/// returns results for some symbols and abstains for others, and a suggestion
/// that works only sometimes is worse than one fewer. A bare symbol never
/// abstains, so that is the shape offered for `ask`.
const FAQ: &[(&str, &str)] = &[
    // Orientation: what someone asks before they know any symbol.
    ("what is this repo written in?", "travsr lang list"),
    ("is the index ready to query?", "travsr status"),
    // Finding things.
    ("where is {sym} defined?", "travsr ask \"{sym}\""),
    (
        "where does the text \"TODO\" appear?",
        "travsr pattern \"TODO\"",
    ),
    // Structure. These are the questions the graph exists for.
    (
        "what calls {sym}?",
        "travsr graph {sym} --direction callers",
    ),
    (
        "what does {sym} depend on?",
        "travsr graph {sym} --direction deps",
    ),
    (
        "what breaks if I change {sym}?",
        "travsr graph {sym} --direction both",
    ),
    (
        "where is {sym} used, before a rename?",
        "travsr references {sym}",
    ),
];

/// Print the questions, using symbols from the reader's own repository.
pub fn print_examples(db_path: Option<&std::path::Path>) {
    use std::io::IsTerminal as _;
    let pal = crate::progress::Palette::for_stream(std::io::stdout().is_terminal());

    // Several symbols, rotated. One name repeated down the page reads as a
    // template with a variable substituted, which is what it would be.
    let symbols = example_symbols(db_path, FAQ.len());
    let grounded = !symbols.is_empty();
    let symbols = if grounded {
        symbols
    } else {
        vec!["PaymentService".to_string()]
    };

    println!("{}", pal.bold("Questions you can ask"));
    println!();

    // Answered by `ask` itself, so no command is shown: typing the question is
    // the whole action. Listed first because they are what someone asks before
    // they know a symbol to ask about.
    println!("{}", pal.orange("About travsr"));
    for q in crate::faq::questions() {
        println!("  {q}");
    }

    println!();
    println!("{}", pal.orange("About this repo"));
    for (i, (question, command)) in FAQ.iter().enumerate() {
        let sym = &symbols[i % symbols.len()];
        let term = sym.to_lowercase();
        let q = question.replace("{sym}", sym).replace("{term}", &term);
        let c = command.replace("{sym}", sym).replace("{term}", &term);
        println!("  {q}");
        println!("      {}", paint_command(pal, &c, sym));
    }

    println!();
    println!(
        "{}",
        pal.dim("`travsr ask --cmds` for every command travsr supports.")
    );

    if !grounded {
        println!(
            "{}",
            pal.dim("Run `travsr init` first and these fill in with your own symbols.")
        );
    }
}

/// The command groups printed by `--cmds`, as (heading, subcommand names).
///
/// Only the grouping lives here. Every name, alias and description is read from
/// clap at print time, so this cannot describe a command that does not exist or
/// go stale when one changes its help text. A test asserts the two sides agree,
/// which means adding a subcommand fails the build until it is filed under a
/// heading rather than silently going missing from this list.
const COMMAND_GROUPS: &[(&str, &[&str])] = &[
    ("Set up a repo", &["init", "connect", "lang"]),
    (
        "Ask about code",
        &["ask", "graph", "references", "pattern", "explain"],
    ),
    ("Run in the background", &["daemon", "mcp", "serve"]),
    (
        "Inspect and debug",
        &["status", "daemon logs", "repos", "fsck", "index"],
    ),
    ("Tune search", &["embed", "rerank", "synonym", "config"]),
];

/// Print every command travsr supports, grouped by what it is for.
///
/// A reader who has just learned `ask` has no way to discover the other twenty
/// without `--help`, which prints them in declaration order with no shape. The
/// grouping is the whole point: `mcp` and `serve` mean nothing next to each
/// other alphabetically and everything under one heading.
pub fn print_commands() {
    use clap::CommandFactory as _;
    use std::io::IsTerminal as _;
    let pal = crate::progress::Palette::for_stream(std::io::stdout().is_terminal());
    let cli = crate::Cli::command();

    println!("{}", pal.bold("Commands travsr supports"));

    for (heading, names) in COMMAND_GROUPS {
        println!();
        println!("{}", pal.orange(heading));
        for name in *names {
            // A name may be a path. `daemon logs` is the whole logging surface
            // and appeared only as a bare word in its parent's folded list,
            // which is not enough to find it if you do not already know it is
            // there. Naming the path promotes it to a row with a description of
            // its own, while it stays listed under `daemon` for context.
            let Some(sub) = resolve_path(&cli, name) else {
                continue;
            };
            // Only a genuine shorthand is worth showing. `graph` carries eight
            // aliases, all of them MCP tool names in snake and kebab spelling,
            // and printing them ran the name column past the width of the
            // terminal. An alias shorter than the command is something a reader
            // would actually type; the rest are compatibility spellings.
            let label = match sub
                .get_all_aliases()
                .filter(|a| a.len() < name.len())
                .min_by_key(|a| a.len())
            {
                Some(short) => format!("{name}, {short}"),
                None => (*name).to_string(),
            };
            println!("  {:<18} {}", pal.ident(&label), first_sentence(sub));

            // Half the surface is one level down: `daemon logs`, `embed status`,
            // `lang install`. Listing only the parents hid it, which is how a
            // logging surface with filters and follow went undocumented. Names
            // only, since a description each would turn this into fifty lines
            // and the footer already points at `--help` for the detail.
            let nested: Vec<&str> = sub
                .get_subcommands()
                .filter(|c| c.get_name() != "help" && !c.is_hide_set())
                .map(clap::Command::get_name)
                .collect();
            // A promoted path is already a leaf shown for its own sake; listing
            // its children under it would repeat what its parent row shows.
            if !nested.is_empty() && !name.contains(' ') {
                for line in wrap_list(&nested, 55) {
                    println!("  {:<18} {}", "", pal.dim(&line));
                }
            }
        }
    }

    println!();
    println!(
        "{}",
        pal.dim("`travsr <command> --help` for the flags on any of these.")
    );
    println!(
        "{}",
        pal.dim("`travsr ask --examples` for the questions ask can answer.")
    );
}

/// Resolve a possibly-nested command path such as `daemon logs`.
///
/// Returns None for a path naming nothing, which a test turns into a failure
/// rather than a silently missing row.
fn resolve_path<'a>(cli: &'a clap::Command, path: &str) -> Option<&'a clap::Command> {
    let mut cur = cli;
    for part in path.split_whitespace() {
        cur = cur.get_subcommands().find(|c| c.get_name() == part)?;
    }
    Some(cur)
}

/// Wrap a comma-separated list of names to `width`, for the column it sits in.
///
/// `daemon` and `embed` carry eight subcommands each, which is wider than a
/// terminal once indented, so the list has to fold rather than run off the edge.
fn wrap_list(names: &[&str], width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, n) in names.iter().enumerate() {
        let last = i + 1 == names.len();
        let piece = if last {
            (*n).to_string()
        } else {
            format!("{n},")
        };
        if !cur.is_empty() && cur.chars().count() + 1 + piece.chars().count() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&piece);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// The first sentence of a command's help, short enough to sit in a column.
///
/// clap's `about` runs to a paragraph for several commands, and printing all of
/// it turns the list into a wall. The first sentence is what a reader scanning
/// for the right command needs.
fn first_sentence(sub: &clap::Command) -> String {
    let about = sub
        .get_about()
        .map(ToString::to_string)
        .unwrap_or_default()
        .replace('\n', " ");
    let mut out = match about.split_once(". ") {
        Some((head, _)) => head.to_string(),
        None => about.trim_end_matches('.').to_string(),
    };
    // Drop a *trailing* parenthetical. Several of these spell out the parts of
    // a command ("(git hook + file watcher + MCP server)") which is detail for
    // `--help`, not for a line someone is scanning. Anchored to the end of the
    // string on purpose: cutting at the first "(" also truncated the ones that
    // qualify a word mid-sentence, and "Enumerate every use site" lost the
    // "(path:line) of a symbol across the repo" that made it useful.
    if out.ends_with(')') {
        if let Some(i) = out.rfind(" (") {
            out.truncate(i);
        }
    }
    // Collapse the runs of whitespace left behind by unwrapping the help text.
    out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    const WIDTH: usize = 56;
    if out.chars().count() > WIDTH {
        out = out.chars().take(WIDTH - 3).collect::<String>();
        // Cut at a word boundary rather than mid-word.
        if let Some(i) = out.rfind(' ') {
            out.truncate(i);
        }
        out.push_str("...");
    }
    out
}

/// A question `ask` cannot answer, and the command that can.
pub(crate) struct MetaRedirect {
    pub answer: &'static str,
    pub command: &'static str,
}

/// Whether the query *is* this question, rather than merely containing it.
///
/// `contains` let `tech stack` catch "how is the tech stack detected", and
/// anchoring alone was not enough: "what is the tech stack detector doing"
/// genuinely starts with the trigger while asking about a symbol (#746 review).
/// The trigger has to account for the whole query, give or take the filler people
/// add to a question typed at a tool.
fn asks_exactly(query: &str, phrase: &str) -> bool {
    let Some(rest) = query.strip_prefix(phrase) else {
        return false;
    };
    const FILLER: &[&str] = &[
        "here",
        "now",
        "please",
        "in",
        "of",
        "for",
        "this",
        "repo",
        "repository",
        "project",
        "codebase",
        "currently",
        "right",
        "at",
        "the",
        "moment",
    ];
    rest.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .all(|w| FILLER.contains(&w))
}

/// Phrases meaning the question is about the repository as a whole, paired with
/// what actually answers them.
///
/// Only the two routes that run a command live here. Questions about travsr are
/// matched against the FAQ catalogue itself (`faq::match_question`), so there is
/// no parallel phrase list for those to drift out of sync with.
///
/// Matched as whole phrases rather than keywords, deliberately. A bare "repo" or
/// "language" appears in plenty of legitimate code questions, and hijacking those
/// would be a worse failure than the one being fixed.
///
/// Containing a space is not the bar. `contains` matches anywhere in the query,
/// so a mid-sentence fragment is as generic as a bare keyword: `is the index`
/// hijacked "where is the index rebuilt after a commit", `how big is` hijacked
/// "how big is a NodeId", and both are ordinary questions about this codebase.
/// This path runs before retrieval and returns, so the search the reader wanted
/// never happens, which is worse than the noisy failure it replaced. Each
/// trigger now carries enough of its own sentence to be about the repository
/// rather than about something in it.
const META_QUESTIONS: &[(&[&str], MetaRedirect)] = &[
    (
        &[
            "what is this repo written in",
            "what languages does this repo",
            "what languages does the repo",
            "what language is this repo",
            "what language is this project",
            "what is the tech stack",
            "what is this codebase written in",
        ],
        MetaRedirect {
            answer: "That is a question about the repository rather than about a symbol in it.",
            command: "travsr lang list",
        },
    ),
    (
        &[
            "how big is the index",
            "how big is the graph",
            "how many files are indexed",
            "how many nodes are in the graph",
            "is the index ready",
            "is the index fresh",
            "is the index up to date",
            "is my index ready",
            "is my index fresh",
            "is the graph fresh",
        ],
        MetaRedirect {
            answer: "That is a question about the index rather than about the code.",
            command: "travsr status",
        },
    ),
];

/// Route a question about the repository away from graph retrieval.
///
/// Returns `None` for anything else, so an ordinary code question is untouched.
pub(crate) fn meta_question_redirect(query: &str) -> Option<&'static MetaRedirect> {
    let q = query.to_lowercase();
    META_QUESTIONS
        .iter()
        .find(|(phrases, _)| phrases.iter().any(|p| asks_exactly(&q, p)))
        .map(|(_, r)| r)
}

/// Colour a rendered command so its shape reads at a glance.
///
/// Three roles, because they answer three different questions for the reader:
/// which tool (constant, so dimmed), which action (the verb, so it leads), and
/// which part is theirs to replace (the identifier). Without that, the line is a
/// uniform run of words and the reader has to parse it before they can use it.
///
/// Everything routes through `Palette`, so `NO_COLOR`, `CLICOLOR_FORCE` and a
/// non-tty stdout all fall back to plain text with the spacing unchanged.
fn paint_command(pal: crate::progress::Palette, cmd: &str, ident: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for (i, word) in cmd.split(' ').enumerate() {
        let painted = if i == 0 {
            // `travsr` is on every line, so it carries no information.
            pal.dim(word)
        } else if i == 1 {
            // The subcommand is the verb: what this line actually does.
            pal.green(word)
        } else if word.starts_with("--") {
            pal.dim(word)
        } else if word.contains(ident) {
            // The part the reader swaps for their own symbol.
            pal.ident(word)
        } else {
            // A flag's value (`callers`, `both`) belongs with its flag.
            pal.dim(word)
        };
        out.push(painted);
    }
    out.join(" ")
}

/// Up to `want` symbols worth putting in examples: real, in this repo, not noise.
///
/// Ranked by in-degree, because an example that resolves to a leaf with no
/// callers demonstrates the command without demonstrating an answer. Returns
/// several so the printed list varies; one name repeated down the page reads as a
/// filled-in template rather than as questions about this codebase.
fn example_symbols(db_path: Option<&std::path::Path>, want: usize) -> Vec<String> {
    let Some(db_path) = db_path else {
        return Vec::new();
    };
    let Ok(store) = crate::daemon_client::open_read_store(db_path) else {
        return Vec::new();
    };

    // Classes and functions only: a file or module node is a valid graph node but
    // a confusing thing to put in `travsr graph <symbol>`.
    let mut candidates: Vec<travsr_core::Node> = Vec::new();
    for kind in ["class", "function", "method", "struct"] {
        if let Ok(mut ns) = store.nodes_by_kind(kind) {
            candidates.append(&mut ns);
        }
    }
    candidates.retain(|n| {
        !travsr_core::noise::is_structural_noise(n)
            // Test entry points and fixtures make poor examples: a reader
            // pasting one gets a result about the test suite, not about the code.
            && n.test_role == travsr_core::TestRole::None
            // Very short names make confusing examples even when well connected.
            && travsr_core::ident::leaf_of(&n.vname.signature).len() >= 4
    });
    if candidates.is_empty() {
        return Vec::new();
    }

    let ids: Vec<travsr_core::NodeId> = candidates.iter().map(|n| n.id).collect();
    let degrees = store.in_degrees(&ids).unwrap_or_default();
    candidates.sort_by_key(|n| std::cmp::Reverse(degrees.get(&n.id).copied().unwrap_or(0)));

    let mut out: Vec<String> = Vec::new();
    for n in candidates {
        let leaf = travsr_core::ident::leaf_of(&n.vname.signature).to_string();
        // Distinct names only: the same leaf can appear on several nodes, and a
        // repeated one defeats the reason for collecting more than one.
        if !out.contains(&leaf) {
            out.push(leaf);
        }
        if out.len() >= want {
            break;
        }
    }
    out
}

#[cfg(test)]
mod meta_question_tests {
    use super::{meta_question_redirect, META_QUESTIONS};

    /// The reported failure: `ask "what is this repo written in?"` matched the
    /// word "repo" against `var:REPO` and returned a screen of bench files with
    /// confident scores. Retrieval succeeds there, so there is no low-confidence
    /// signal to abstain on; it has to be caught before the search runs.
    #[test]
    fn the_reported_question_is_redirected() {
        let r = meta_question_redirect("what is this repo written in?")
            .expect("must be recognised as a question about the repo");
        assert_eq!(r.command, "travsr lang list");
    }

    /// Questions about travsr are answered by matching the FAQ catalogue itself,
    /// not by this phrase list, so they must NOT appear here. A second list would
    /// be one more thing to keep in sync, which is what this design removed.
    #[test]
    fn travsr_questions_are_not_duplicated_in_the_phrase_list() {
        for q in [
            "what is travsr",
            "how does travsr work",
            "how do I install travsr",
        ] {
            assert!(
                meta_question_redirect(q).is_none(),
                "`{q}` should be handled by the FAQ matcher, not a phrase"
            );
            assert!(
                crate::faq::match_question(q).is_some(),
                "`{q}` must still be answerable, via the FAQ matcher"
            );
        }
    }

    /// The triggers carry enough of their own sentence to be about the index.
    /// `is it up to date` used to be one and was dropped in the #746 review:
    /// `contains` matches it anywhere, so "check if it is up to date before the
    /// write" would have been answered with a status table instead of searched.
    #[test]
    fn questions_about_the_index_go_to_status() {
        for q in [
            "how big is the graph",
            "is my index ready",
            "is the index up to date",
            "is the graph fresh",
        ] {
            let r = meta_question_redirect(q).unwrap_or_else(|| panic!("{q} not matched"));
            assert_eq!(r.command, "travsr status", "{q}");
        }
    }

    /// The failure mode that would be worse than the bug. A user searching for a
    /// symbol whose name happens to contain "repo" or "language" must still get
    /// their search: hijacking a real query is a silent wrong answer, where the
    /// original problem was at least visibly noisy.
    #[test]
    fn real_code_questions_are_never_hijacked() {
        for q in [
            "repo_languages",
            "language_distribution",
            "where is Language defined",
            "what calls repo_root",
            "RepoStats",
            "normalize_repo_root",
            "how does the parser handle a repo path",
            // From the #746 review. Each of these hijacked a real search: the
            // trigger was a mid-sentence fragment, and `contains` matches it
            // anywhere. This path returns before retrieval, so the reader's
            // search never ran.
            "where is the index rebuilt after a commit",
            "how big is a NodeId",
            "how many nodes does the PPR walk visit",
            "which languages does the analysis crate support",
            "is the index_files helper called twice",
        ] {
            assert!(
                meta_question_redirect(q).is_none(),
                "`{q}` is a code question and must reach retrieval"
            );
        }
    }

    /// Whole phrases, not keywords. A single word like "repo" or "language" is
    /// too common in real queries to route on, and this is what keeps the test
    /// above passing rather than luck.
    #[test]
    fn every_trigger_is_a_phrase_not_a_bare_keyword() {
        for (phrases, redirect) in META_QUESTIONS {
            assert!(!phrases.is_empty());
            for p in *phrases {
                // Containing a space was the bar that let `is the index` through,
                // and last round the data was corrected without the check being.
                // `tech stack` passes a space test and is exactly the shape that
                // hijacks (#746 review).
                let words = p.split_whitespace().count();
                let opens_a_question = matches!(
                    p.split_whitespace().next(),
                    Some("how" | "what" | "which" | "where" | "is" | "do" | "does" | "can")
                );
                assert!(
                    words >= 3 && opens_a_question,
                    "`{p}` is too short or does not open a question; even anchored \
                     at the start of a query it would hijack a code search"
                );
                assert_eq!(p.to_lowercase(), *p, "`{p}` must be lowercase to match");
            }
            assert!(!redirect.answer.is_empty());
            assert!(!redirect.command.is_empty());
        }
    }

    /// Every redirect must name a command that exists, for the same reason the
    /// other lists do: #727 was a documented command that did not.
    #[test]
    fn every_redirect_names_a_real_subcommand() {
        use clap::CommandFactory as _;
        let cmd = crate::Cli::command();
        let known: Vec<String> = cmd
            .get_subcommands()
            .flat_map(|c| {
                std::iter::once(c.get_name().to_string())
                    .chain(c.get_all_aliases().map(str::to_string))
            })
            .collect();
        for (_, r) in META_QUESTIONS {
            let sub = r.command.split_whitespace().nth(1).unwrap_or("");
            assert!(
                known.iter().any(|k| k == sub),
                "`{}` names unknown subcommand {sub:?}",
                r.command
            );
        }
    }
}

#[cfg(test)]
mod command_group_tests {
    use super::{first_sentence, COMMAND_GROUPS};

    /// The grouping is the one hand-written part of `--cmds`, so it is the one
    /// part that can drift. Adding a subcommand without filing it here would
    /// leave it silently missing from a list that claims to be every command,
    /// which is the same failure as documenting one that does not exist (#727).
    #[test]
    fn every_subcommand_is_listed_exactly_once() {
        use clap::CommandFactory as _;
        let cli = crate::Cli::command();

        let mut listed: Vec<&str> = COMMAND_GROUPS
            .iter()
            .flat_map(|(_, n)| *n)
            .copied()
            .collect();
        listed.sort_unstable();
        let mut deduped = listed.clone();
        deduped.dedup();
        assert_eq!(listed, deduped, "a command is filed under two headings");

        // A promoted path such as `daemon logs` is a row of its own, so the
        // top-level coverage check below compares on its first word.
        let top = |n: &str| n.split_whitespace().next().unwrap_or(n).to_string();

        for sub in cli.get_subcommands() {
            // clap generates `help`, and `hook-run` is marked hidden because the
            // git hook invokes it rather than a person. `--cmds` lists what a
            // reader can usefully type, which is what `--help` shows.
            if sub.get_name() == "help" || sub.is_hide_set() {
                continue;
            }
            assert!(
                listed.iter().any(|n| top(n) == sub.get_name()),
                "`{}` is a real subcommand but no --cmds heading lists it",
                sub.get_name()
            );
        }
        // Every listed name must resolve, paths included. A row naming a
        // command that does not exist is the dead end this whole file guards
        // against, and a nested path is the easiest way to write one by
        // mistake: `daemon log` looks right and resolves to nothing.
        for name in &listed {
            assert!(
                super::resolve_path(&cli, name).is_some(),
                "--cmds lists `{name}`, which resolves to no command"
            );
        }
    }

    /// Nested names sit in a column that starts 21 characters in, so the list
    /// has to fold to stay inside an 80-column terminal. `daemon` and `embed`
    /// carry eight each, which overflows a single line on its own.
    #[test]
    fn nested_command_lists_fit_the_terminal() {
        use clap::CommandFactory as _;
        let cli = crate::Cli::command();
        for sub in cli.get_subcommands().filter(|c| !c.is_hide_set()) {
            let nested: Vec<&str> = sub
                .get_subcommands()
                .filter(|c| c.get_name() != "help" && !c.is_hide_set())
                .map(clap::Command::get_name)
                .collect();
            if nested.is_empty() {
                continue;
            }
            for line in super::wrap_list(&nested, 55) {
                let rendered = 2 + 18 + 1 + line.chars().count();
                assert!(
                    rendered <= 80,
                    "`{}` renders a {rendered}-column line: {line}",
                    sub.get_name()
                );
            }
            // Folding must not lose one. The whole point of the line is that a
            // reader can see `daemon logs` exists without running --help.
            let joined = super::wrap_list(&nested, 55).join(" ");
            for n in &nested {
                assert!(
                    joined.split([',', ' ']).any(|w| &w == n),
                    "`{}` dropped `{n}` when folding",
                    sub.get_name()
                );
            }
        }
    }

    /// The description column is only useful if it fits beside the name column
    /// on an 80-column terminal, which is what the truncation is for.
    #[test]
    fn descriptions_fit_the_column() {
        use clap::CommandFactory as _;
        let cli = crate::Cli::command();
        for sub in cli.get_subcommands().filter(|c| !c.is_hide_set()) {
            let d = first_sentence(sub);
            assert!(
                d.chars().count() <= 56,
                "`{}` renders a {}-char description: {d}",
                sub.get_name(),
                d.chars().count()
            );
            assert!(
                !d.contains('\n'),
                "`{}` renders a description with a newline",
                sub.get_name()
            );
        }
    }
}

#[cfg(test)]
mod faq_tests {
    use super::FAQ;

    /// Every entry must render with nothing left to substitute. A stray `{sym}`
    /// reaching the terminal would be pasted verbatim and search for that text.
    #[test]
    fn every_entry_renders_cleanly() {
        for (question, command) in FAQ {
            for (label, text) in [("question", question), ("command", command)] {
                let r = text.replace("{sym}", "Widget").replace("{term}", "widget");
                assert!(
                    !r.contains('{') && !r.contains('}'),
                    "{label} `{text}` left a slot unfilled: {r}"
                );
            }
        }
    }

    /// Each entry is scanned for by its question, so it has to read as one.
    /// An earlier version used command labels ("Find a symbol by name"), which
    /// describe the tool rather than the reader's problem.
    #[test]
    fn every_question_reads_as_a_question() {
        for (question, _) in FAQ {
            assert!(
                question.ends_with('?'),
                "`{question}` is a label, not a question"
            );
            assert!(
                question.chars().next().is_some_and(|c| c.is_lowercase()),
                "`{question}` should read as spoken text, not a heading"
            );
        }
    }

    /// Only commands that exist. A list naming a subcommand travsr does not have
    /// sends someone to a dead end while looking authoritative, which is exactly
    /// what #727 was. Read from clap rather than a hand-kept copy.
    #[test]
    fn every_command_names_a_real_subcommand() {
        use clap::CommandFactory as _;
        let cmd = crate::Cli::command();
        let known: Vec<String> = cmd
            .get_subcommands()
            .flat_map(|c| {
                std::iter::once(c.get_name().to_string())
                    .chain(c.get_all_aliases().map(str::to_string))
            })
            .collect();
        assert!(!known.is_empty(), "clap reported no subcommands");

        for (_, template) in FAQ {
            let command = template
                .replace("{sym}", "PaymentService")
                .replace("{term}", "payment");
            let mut words = command.split_whitespace();
            assert_eq!(words.next(), Some("travsr"), "`{command}`");
            let sub = words.next().unwrap_or("");
            assert!(
                known.iter().any(|k| k == sub),
                "`{command}` names unknown subcommand {sub:?}"
            );

            // The name existing is not enough. `faq.rs` learned this when
            // `travsr explain "<query>"` passed a name check and still exited 2
            // for missing its second argument, and this is the list a reader is
            // most likely to paste from, so it gets the same parse.
            let argv: Vec<String> = command.split_whitespace().map(str::to_string).collect();
            if let Err(e) = crate::Cli::command().try_get_matches_from(&argv) {
                use clap::error::ErrorKind::{DisplayHelp, DisplayVersion};
                assert!(
                    matches!(e.kind(), DisplayHelp | DisplayVersion),
                    "`{command}` does not parse: {}",
                    e.render().to_string().lines().next().unwrap_or_default()
                );
            }
        }
    }

    /// Diagnostics stay out of a user-facing list. `explain` describes itself as
    /// "not part of normal use", and `fsck` reports ghost nodes and orphan edges:
    /// both are about travsr's own internals rather than the reader's code.
    /// Pinned because they are easy to add back, being genuinely useful commands.
    #[test]
    fn no_internal_diagnostics_are_offered() {
        for (question, command) in FAQ {
            for internal in ["explain", "fsck"] {
                assert!(
                    !command.contains(&format!("travsr {internal}")),
                    "`{question}` offers the {internal} diagnostic; it answers a \
                     question about travsr's internals, not about this codebase"
                );
            }
        }
    }

    /// `ask` abstains on questions it cannot ground. "what breaks if I change X"
    /// is the measured case: it returns nothing today, so it must be routed to
    /// `graph`, not offered as something to ask. Pins the routing rather than
    /// leaving it to whoever edits the list next.
    #[test]
    fn questions_ask_cannot_answer_are_not_routed_to_ask() {
        for (question, command) in FAQ {
            if question.to_lowercase().contains("what breaks") {
                assert!(
                    command.contains("graph") && command.contains("--direction both"),
                    "`{question}` must route to graph, not `{command}`"
                );
            }
        }
    }
}

/// One concrete next step, phrased as a command the user can paste.
pub(crate) struct Suggestion {
    pub command: String,
    pub why: String,
    /// The part of `command` the reader is expected to swap. Carried rather than
    /// re-derived so the painter highlights exactly what was substituted.
    pub ident: String,
}

/// Intent keywords mapped to the command that actually answers them.
///
/// `ask` is graph-grounded retrieval, so several common questions are better
/// served by a different subcommand entirely. A user who phrases one of those as
/// a question gets an abstention today and no hint that the answer exists one
/// command over.
///
/// Ordered most specific first: "what breaks if" must win over the bare "what",
/// and "who calls" over "call".
const INTENT_ROUTES: &[(&[&str], &str, &str)] = &[
    (
        &[
            "what breaks",
            "blast radius",
            "impact of",
            "safe to change",
            "safe to remove",
        ],
        "travsr graph {sym} --direction both",
        "callers and dependencies together, which is what breaks",
    ),
    (
        &[
            "who calls",
            "what calls",
            "callers of",
            "used by",
            "call sites",
        ],
        "travsr graph {sym} --direction callers",
        "incoming call edges",
    ),
    (
        &[
            "depend on",
            "dependencies of",
            "imports",
            "what does it use",
        ],
        "travsr graph {sym} --direction deps",
        "outgoing dependency edges",
    ),
    (
        &["every use", "all uses", "references to", "rename"],
        "travsr references {sym}",
        "every use site with path:line, wider than callers",
    ),
    (
        &[
            "config",
            "setting",
            "option",
            "env var",
            "environment variable",
        ],
        "travsr config get <key>",
        "configuration is not in the code graph",
    ),
    (
        &["why did", "why is", "ranked", "scored", "not showing"],
        "travsr explain \"{q}\" <symbol>",
        "shows which terms matched and which thresholds failed",
    ),
];

/// Build next steps from the user's own question.
///
/// Two independent sources, because they fail differently. Symbol lookup finds a
/// real name in *this* repo, which is the strongest possible suggestion but only
/// works when the query contains something close to one. Intent routing works on
/// phrasing alone, which covers the case where the user described what they want
/// without naming anything that exists.
///
/// Returns empty rather than padding with generic advice, so the caller can fall
/// back to the catalogue instead of printing suggestions that suggest nothing.
pub(crate) fn suggest_next(db_path: &std::path::Path, query: &str) -> Vec<Suggestion> {
    let lower = query.to_lowercase();
    let mut out: Vec<Suggestion> = Vec::new();

    // The nearest real symbol, if the repo has one. Uses the same fuzzy search
    // `ask` itself uses, so a suggestion can never name something unindexed.
    let nearest = nearest_symbol(db_path, query);

    // #778: only suggest the nearest symbol when it differs from what the user
    // already typed. When the query IS an exact symbol name (`ask "UI"`), the
    // nearest match is that same string, so echoing `ask "UI"` -> `ask "UI"` is a
    // dead-end loop. Skip it. `ask "UI"` matches no `INTENT_ROUTES` key either, so
    // `suggest_next` then returns empty — which is correct: per this function's
    // contract the caller falls back to the catalogue rather than printing a
    // suggestion that suggests nothing.
    if let Some(sym) = nearest
        .as_deref()
        .filter(|sym| !sym.eq_ignore_ascii_case(query.trim()))
    {
        out.push(Suggestion {
            command: format!("travsr ask \"{sym}\""),
            why: "closest symbol in this repo to what you typed".to_string(),
            ident: sym.to_string(),
        });
    }

    // Intent routing. `{sym}` is substituted when a real symbol was found;
    // otherwise a placeholder, so the shape of the answer is still visible.
    let sym_slot = nearest.clone().unwrap_or_else(|| "<symbol>".to_string());
    for (keys, template, why) in INTENT_ROUTES {
        if keys.iter().any(|k| lower.contains(k)) {
            out.push(Suggestion {
                command: render_route(template, &sym_slot, query),
                why: (*why).to_string(),
                ident: sym_slot.clone(),
            });
            // One intent route is a hint; several is a menu the user has to
            // re-read. Stop at the most specific match.
            break;
        }
    }

    // The text escape hatch, offered only when there is a distinctive term to
    // search for. `pattern` answers questions the graph deliberately does not
    // model, which is a large share of what reaches this path.
    if let Some(term) = distinctive_term(query) {
        out.push(Suggestion {
            command: format!("travsr pattern \"{term}\""),
            why: "searches the text of tracked files, for things the graph does not model"
                .to_string(),
            ident: term.clone(),
        });
    }

    out
}

/// Render one `INTENT_ROUTES` template into a runnable command.
///
/// Owns the `{q}` substitution so there is exactly one place that strips
/// quotes, and so a test can drive the real thing. The template wraps `{q}` in
/// double quotes, so a query containing one produced `travsr explain "why is
/// "charge" not showing" <symbol>`, which the shell splits into three words
/// (#746 review).
fn render_route(template: &str, sym: &str, query: &str) -> String {
    template
        .replace("{sym}", sym)
        .replace("{q}", &query.replace('"', ""))
}

/// The closest indexed symbol to `query`, or `None` when nothing is close.
///
/// Deliberately reuses the store's own fuzzy search rather than a bespoke match,
/// so a suggestion is always a name that exists. Suggesting a symbol the repo
/// does not contain would repeat the mistake this whole path exists to fix.
fn nearest_symbol(db_path: &std::path::Path, query: &str) -> Option<String> {
    let store = crate::daemon_client::open_read_store(db_path).ok()?;
    let hits = store.search_nodes_fuzzy(query).ok()?;
    let leaf = hits
        .iter()
        .find(|n| {
            // `is_structural_noise` filters test *paths*, but an inline
            // `#[cfg(test)] mod tests` lives in src/, so a test helper survives
            // it. Suggesting `works_does_not_match_workspace` as the closest
            // symbol is worse than suggesting nothing: it is not code the reader
            // was looking for, and it makes the tool look like it guessed.
            !travsr_core::noise::is_structural_noise(n)
                && n.test_role == travsr_core::TestRole::None
        })
        // No `or_else(|| hits.first())` here. `find` returns None only when every
        // hit is noise or a test, which is exactly the case the filter above
        // argues against, so falling back to the unfiltered first hit handed
        // back the very suggestion the comment calls worse than nothing.
        // `suggest_next` documents the contract: return empty and let the caller
        // fall back to the catalogue, which it already does.
        .map(|n| travsr_core::ident::leaf_of(&n.vname.signature).to_string())?;
    (!leaf.is_empty()).then_some(leaf)
}

/// The longest word in the query that is not a stop word, used as a text search
/// term. Longest because it is the most selective: `pattern "the"` is noise.
fn distinctive_term(query: &str) -> Option<String> {
    const STOP: &[&str] = &[
        "what", "where", "which", "who", "why", "how", "does", "do", "did", "is", "are", "was",
        "the", "this", "that", "for", "from", "with", "and", "not", "can", "should", "would",
        "when", "there", "here", "into", "about", "code", "function", "class", "file",
    ];
    query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() >= 4 && !STOP.contains(&w.to_lowercase().as_str()))
        .max_by_key(|w| w.len())
        .map(str::to_string)
}

#[cfg(test)]
mod suggestion_tests {
    use super::{distinctive_term, render_route, INTENT_ROUTES};

    /// Split a rendered command the way a shell would, so a quoted argument
    /// stays one word.
    fn shell_words(line: &str) -> Vec<String> {
        let mut out = Vec::new();
        let (mut cur, mut quoted) = (String::new(), false);
        for ch in line.chars() {
            match ch {
                '"' => quoted = !quoted,
                c if c.is_whitespace() && !quoted => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Every route must produce a runnable command, not a template with an
    /// unsubstituted slot left in it.
    ///
    /// Rendered through `render_route`, not through a copy of its body: the
    /// previous fixture stripped the quotes itself before substituting, so the
    /// production strip was never executed and deleting it kept the suite green
    /// (#746 review).
    #[test]
    fn routes_have_no_unsubstituted_slots_after_rendering() {
        // One UNBALANCED quote, deliberately. A balanced pair is harmless:
        // the shell concatenates `"why is "charge" missing"` straight back into
        // a single word, so a fixture built from one cannot fail. A lone quote
        // closes the template's own and lets the rest split (#746 review).
        let raw = r#"why is "charge missing"#;
        for (_, template, _) in INTENT_ROUTES {
            let rendered = render_route(template, "Foo", raw);
            // The load-bearing check, for the routes that actually carry the
            // query: it has to come back out as ONE word. Without the strip in
            // `render_route` the inner quote closes the template's own, and
            // this same query arrives as three.
            if template.contains("{q}") {
                assert!(
                    shell_words(&rendered).contains(&"why is charge missing".to_string()),
                    "the query did not survive as a single shell word: {rendered} -> {:?}",
                    shell_words(&rendered)
                );
            }
            assert!(
                !rendered.contains('{') && !rendered.contains('}'),
                "template `{template}` left a slot unfilled: {rendered}"
            );
            assert!(
                !rendered.contains("\"\""),
                "an empty quoted argument means the query was eaten: {rendered}"
            );
            // And it has to parse. That was the ask last round and it landed
            // only on the FAQ list.
            {
                use clap::CommandFactory as _;
                let argv: Vec<String> = shell_words(&rendered);
                if let Err(e) = crate::Cli::command().try_get_matches_from(&argv) {
                    use clap::error::ErrorKind::{DisplayHelp, DisplayVersion};
                    assert!(
                        matches!(e.kind(), DisplayHelp | DisplayVersion),
                        "`{rendered}` does not parse: {}",
                        e.render().to_string().lines().next().unwrap_or_default()
                    );
                }
            }
        }
    }

    /// The specific phrasings must beat the general ones. "what breaks if I
    /// change X" must route to blast radius, not to the bare "what calls" rule,
    /// which is why order is load-bearing rather than incidental.
    #[test]
    fn the_most_specific_intent_wins() {
        let q = "what breaks if i change the ledger";
        let first = INTENT_ROUTES
            .iter()
            .find(|(keys, _, _)| keys.iter().any(|k| q.contains(k)))
            .expect("phrase must match some route");
        assert!(
            first.1.contains("--direction both"),
            "expected the blast-radius route, got `{}`",
            first.1
        );
    }

    /// #778: when the query IS an exact symbol name, `suggest_next` must never
    /// echo it back — `ask "UI"` -> try `ask "UI"` is a dead-end loop. The guard
    /// is load-bearing here, not the absence of a match: `nearest_symbol` DOES
    /// resolve `UI` to `class:UI`, yet the echo filter drops the identical query
    /// so the suggestion list carries no `travsr ask "UI"`.
    #[test]
    fn suggest_next_never_echoes_the_exact_query() {
        use travsr_store::{SqliteStore, Store as _};
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("graph.db");
        {
            let mut store = SqliteStore::open(&db).expect("open");
            store
                .put_node(&travsr_core::Node::new(
                    travsr_core::VName::new("corpus", "root", "app/ui.rb", "ruby", "class:UI"),
                    "class",
                ))
                .expect("put_node");
        }
        // Non-vacuous: the symbol really is found, so what suppresses the echo is
        // the guard, not a missing match.
        assert_eq!(super::nearest_symbol(&db, "UI").as_deref(), Some("UI"));
        let suggestions = super::suggest_next(&db, "UI");
        assert!(
            suggestions.iter().all(|s| s.command != "travsr ask \"UI\""),
            "suggest_next echoed the identical query back: {:?}",
            suggestions.iter().map(|s| &s.command).collect::<Vec<_>>()
        );
    }

    #[test]
    fn caller_phrasings_route_to_callers() {
        for q in [
            "who calls payment service",
            "what calls this",
            "call sites of foo",
        ] {
            let hit = INTENT_ROUTES
                .iter()
                .find(|(keys, _, _)| keys.iter().any(|k| q.contains(k)))
                .unwrap_or_else(|| panic!("no route matched {q:?}"));
            assert!(hit.1.contains("callers"), "{q:?} routed to `{}`", hit.1);
        }
    }

    /// Configuration questions are the clearest case of "the answer exists, just
    /// not in the graph", so they must not route to a graph command.
    #[test]
    fn configuration_questions_route_away_from_the_graph() {
        let hit = INTENT_ROUTES
            .iter()
            .find(|(keys, _, _)| {
                keys.iter()
                    .any(|k| "how do i set the config option".contains(k))
            })
            .expect("must match");
        assert!(hit.1.starts_with("travsr config get"), "got `{}`", hit.1);
    }

    #[test]
    fn the_search_term_is_selective_not_a_stop_word() {
        assert_eq!(
            distinctive_term("where is the retry budget configured"),
            Some("configured".to_string())
        );
        // All stop words or too short: nothing worth searching for.
        assert_eq!(distinctive_term("what is it for"), None);
        assert_eq!(distinctive_term(""), None);
    }
}

#[cfg(test)]
mod render_tests {
    use super::{render_rows, to_row, Row};
    use travsr_mcp::query::AskRow;

    fn row(signature: &str) -> Row {
        Row {
            signature: signature.to_string(),
            path: "crates/travsr-cli/src/ask.rs:1".to_string(),
            score: "0.500".to_string(),
        }
    }

    fn ask_row(kind: &str, signature: &str) -> AskRow {
        AskRow {
            kind: kind.to_string(),
            signature: signature.to_string(),
            path: "x".to_string(),
            line: None,
            score: 0.5,
            match_source: None,
        }
    }

    /// #824: pins the borderless render. The previous default came from a bare
    /// `Table::new(rows)` with no style call, which is easy to reintroduce, and
    /// the `+---+` rule per row was most of the stdout an agent piping `ask`
    /// paid for. Lines are trimmed before comparison so column padding cannot
    /// flip the assertions.
    #[test]
    fn render_rows_is_borderless_and_keeps_the_header() {
        let out = render_rows(vec![row("fn:knapsack"), row("method:Animal.describe")]);
        assert!(!out.contains("+--"), "row rule survived:\n{out}");
        assert!(!out.contains('|'), "cell bars survived:\n{out}");

        let header = out.lines().next().unwrap_or_default().trim();
        assert!(header.starts_with("Signature"), "header lost: {header:?}");
        assert!(header.contains("Path") && header.contains("Score"));
        assert!(!out.contains("Kind"), "Kind column returned:\n{out}");

        // ASCII only: a Windows terminal must render this unchanged.
        assert!(out.is_ascii(), "non-ASCII in table:\n{out}");
        // `path:line` stays one unbroken, clickable token.
        assert!(out.contains("crates/travsr-cli/src/ask.rs:1"));

        // No line ends in whitespace: `Style::blank()` pads the last column and
        // would otherwise leave a trailing space on every row. A stray `\r` is
        // stripped first so this cannot fail on a CRLF checkout.
        for line in out.lines() {
            let line = line.trim_end_matches('\r');
            assert_eq!(line, line.trim_end(), "trailing whitespace: {line:?}");
        }
    }

    /// The kind is folded away only when the signature already spells it.
    /// Phase A (all 16 languages) and Kotlin prefix their signatures; the SCIP,
    /// Swift and Dart Phase B conventions do not, and those rows would
    /// otherwise lose their only kind information in the human view.
    #[test]
    fn kind_is_kept_for_signatures_that_do_not_carry_it() {
        let cases = [
            // Already kinded — the prefix abbreviates the kind.
            ("function", "fn:knapsack", "fn:knapsack"),
            ("method", "method:Animal.describe", "method:Animal.describe"),
            ("variable", "var:LIMIT", "var:LIMIT"),
            ("package", "pkg:serde@1.0", "pkg:serde@1.0"),
            ("file-module", "filemod:knapsack", "filemod:knapsack"),
            ("doc-chunk", "doc:rfc-010/design", "doc:rfc-010/design"),
            // Provider schemes — not kinds, so the kind is prepended.
            (
                "method",
                "scip:src/Foo.java:semanticdb maven . . Foo#bar().",
                "method: scip:src/Foo.java:semanticdb maven . . Foo#bar().",
            ),
            (
                "method",
                "swift::Animal.describe",
                "method: swift::Animal.describe",
            ),
            (
                "method",
                "file:///abs/x.dart::Animal.describe",
                "method: file:///abs/x.dart::Animal.describe",
            ),
            // No prefix at all: a repo-relative path, and a Windows path whose
            // drive letter must not be mistaken for a kind prefix.
            (
                "file",
                "crates/travsr-cli/src/ask.rs",
                "file: crates/travsr-cli/src/ask.rs",
            ),
            ("file", "C:\\src\\x.rs", "file: C:\\src\\x.rs"),
        ];
        for (kind, signature, want) in cases {
            let got = to_row(&ask_row(kind, signature)).signature;
            assert_eq!(got, want, "kind={kind} signature={signature}");
        }
    }
}
