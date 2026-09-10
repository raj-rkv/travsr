# RFC-029: `travsr init` finishes the job it detects

**Status:** Draft
**Author:** Ritik Pal
**Date:** 2026-09-02
**Crates affected:** `travsr-cli`, `travsr-plugin-host`
**Depends on:** RFC-026 (this changes when its adapters write vs print), ADR-017 (Rule 3 bounds what is deliberately NOT changed here)
**Supersedes:** N/A
**Issue:** TBD

---

## Summary

On a fresh repository `travsr init` prints 47 lines, of which three are
near-identical MCP config blobs the user is told to paste by hand, one is a tip
naming a command to run, one is a PATH note, and one is a suggested query naming
a symbol that does not exist in their code. The wiring that RFC-026 added to
remove that work only activates when the repository already contains the tool's
config directory, so it does nothing on the one run where it matters.

This RFC changes `Detection::Auto` to mean "the tool is installed on this
machine" rather than "this repository is already configured for the tool", makes
the first-run suggestion use a symbol from the user's own graph, and turns on
per-repo embeddings when the backend is already installed. Target for the same
run: zero blobs, zero tips, every line a completed action.

---

## Motivation

Measured on a clean 4-file repository (TypeScript, Python, Go) with Claude Code,
Cursor and Zed installed on the machine and none configured in the repo, using
the shipped `travsr init` with no flags:

```
total output lines:  47
add-manually blocks:  3
```

The three blocks are byte-identical apart from the tool name, and each is a
12-line JSON document the user is expected to transcribe into a file the message
does not name. The instructions the user is left holding:

```
6:  tip: go found in this repo, full cross-file analysis is not set up yet:
7:         travsr lang install go
8:  claude-code detected (global), add manually:
21: cursor detected (global), add manually:
34: zed detected (global), add manually:
47: note: `travsr` is not on PATH, so configs use an absolute path.
```

An architect evaluating Travsr against a competitor reported the configuration
as too manual. This output is that report, reproduced.

### The cause is one branch

`Tool::detect` (`crates/travsr-cli/src/connect.rs:297`) gates auto-writing on a
project-local marker:

```rust
Tool::ClaudeCode => {
    if has(repo.join(".claude")) || has(repo.join("CLAUDE.md")) {
        Detection::Auto      // write project config
    } else if home.is_some_and(|h| has(h.join(".claude"))) {
        Detection::Print     // print a blob (connect.rs:1221)
    } else {
        Detection::None
    }
}
```

Cursor and Zed have the same shape. A repository that has never been used with
the tool has no such marker, so every tool falls to `Print`. The consequence is
that **the feature that exists to remove manual configuration is inert for
exactly the users who have not already done it manually**, and degrades to
printing the config it declined to write.

### The safety argument does not survive contact with the same command

RFC-026 is right that Travsr must not silently mutate a user's global config,
and nothing here proposes to. But the objection being applied is broader than
that: it is refusing to create a *project-scoped* file. The same `travsr init`
run, on the same repository, with no prompt:

```
$ git status --porcelain
?? .travsr/
?? .travsrignore
```

`.travsrignore` is created unasked and announced as a courtesy
(`crates/travsr-cli/src/progress.rs:592`). So `init` already writes into the
user's working tree; it declines only for the file that makes the product
reachable. `.mcp.json`, `.cursor/mcp.json` and `.zed/settings.json` are
standard project-scoped paths, RFC-026 already gitignores what it generates, and
`travsr connect --remove` already exists as the undo. The blast radius of
writing them is the blast radius `init` accepts every time it runs.

### Two smaller items in the same 30 seconds

- `crates/travsr-cli/src/progress.rs:600` prints
  `try: travsr ask "what calls PaymentService?"` to every user on every first
  run. `PaymentService` is a literal. A new user's first command therefore
  returns nothing, in the product whose pitch is that it does not guess. The
  substitution mechanism already exists (`crates/travsr-cli/src/ask.rs:634`
  replaces `{sym}` with a real symbol); the first-run tip does not use it.
- `crates/travsr-cli/src/embed.rs:2531` prints
  `tip: embeddings are installed but not enabled for this repo`. The model and
  sidecar are already downloaded at this point; what remains is per-repo
  activation. Meanwhile the heavier `travsr lang install` gets an interactive
  `Set up now? [y/N]` (`crates/travsr-cli/src/init.rs`, `hint_lang_detect`).
  The cheap step is passive and the expensive one is assisted.

### The principle

Every `tip: run X` is a statement that Travsr detected the condition, knows the
remedy, and chose to delegate the typing. `hint_lang_detect`,
`hint_embed_missing` and `path_contains_travsr_bin` are all detection that stops
one step short of the action.

---

## Detailed Design

### D1 - `Detection::Auto` means "tool installed", not "repo already configured"

For tools that have a documented project-scoped MCP config path, `detect`
returns `Auto` when the tool is present anywhere (project marker **or** home
marker). The adapter then writes the project-scoped file and adds it to the
managed `.gitignore` block, exactly as it does today for the project-marker
case. No new write path is introduced; the existing one becomes reachable.

`Detection::Print` is retained and is now reserved for its true meaning: tools
whose MCP servers live only in a home-directory file Travsr does not edit.
`connect.rs` already documents which ones those are, in comments that the
current `detect` does not act on:

- Windsurf (home-only config, per RFC-026)
- Codex ("keeps its MCP servers in a global TOML we do not edit", `connect.rs:357`)
- Antigravity ("MCP servers live in the global `~/.gemini/config/mcp_config.json`", `connect.rs:479`)

VS Code Copilot keeps its current rule unchanged and is called out in Drawbacks.

**Invariants that must hold.** All are already implemented and this RFC weakens
none of them:

- JSON merge upserts only the `travsr` key and preserves every other server.
- A target file that does not parse as strict JSON is skipped with a warning,
  never rewritten.
- Markdown managed blocks stay a single balanced
  `<!-- travsr:begin -->` / `<!-- travsr:end -->` pair; malformed, duplicate or
  nested markers cause a skip, never a destructive replace.
- Global and home-directory files are never written.
- Every file written or skipped is still printed in the one-line summary.

**Failure path.** Unchanged: connect failures are non-fatal and never fail
`travsr init`. A skip is reported, not retried.

**Deliberately unchanged.** `travsr init --no-connect` remains the opt-out and
`travsr connect --remove` remains the undo. The `--rules` default is out of
scope (see Out of Scope).

### D2 - The first-run suggestion names a symbol from the user's own graph

`print_summary` (`crates/travsr-cli/src/progress.rs:600`) takes a symbol
selected from the graph just written and renders it through the existing
`{sym}` substitution rather than the `PaymentService` literal. When no suitable
symbol is available (an empty or trivially small index) the line is omitted
rather than falling back to a name the repository does not contain.

Selection must be deterministic so the line is stable across runs on an
unchanged repo.

### D3 - Per-repo embeddings are enabled when the backend is already installed

When `travsr_plugin_host::active_backend_id()` resolves and the repository has
no embed lane yet, `init` enables it instead of printing the tip at
`embed.rs:2531`. This performs no download; the sidecar and model are already
on disk, which is the precondition for that message being printed at all.

If no backend is installed, behaviour is unchanged: the existing
`travsr embed init` tip still prints, because that path does involve a download
and is a real decision.

### D4 - A registered language with no analyzer is reported, not silently dropped

Independent of the UX work and listed here because it undermines it: a language
that is registered in `lang.toml` but whose wrapper is absent from disk lands in
no `PhaseBOutcome` bucket. `no_analyzer_langs`
(`crates/travsr-plugin-host/src/indexer.rs:492`) is built only from
`CatalogResolver::unresolvable_shims()` (`resolver.rs:256`, npm shims the
Windows sandbox cannot execute) and `CatalogResolver::missing_tool()`
(`resolver.rs:264`, wrapper installed but underlying tool absent). "Wrapper
never installed" is neither.

Reproduced on a Go repository with `registered = ["go"]` and no
`travsr-lang-go` on disk:

```
phase_b_warnings        (empty)
edges                   defines/binding 21, depends 3      # zero ref/call
travsr status           semantic: complete                 # wrong
travsr lang status      go   no analyzer   partial         # right
```

Two surfaces reading the same machine disagree, and the one users are pointed at
reports success. This is the `#760` failure mode with a producer-side cause
rather than a consumer-side one, so the guards added there cannot catch it: they
assert every variant of `PhaseBWarningClass` is handled, and this language never
becomes a variant at all.

The fix is to add the absent-wrapper case to the `skipped_no_analyzer` bucket,
whose existing hint (`travsr lang install <lang>`) is already the correct remedy.

---

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Prompt before writing each config file | Turns three blobs into three prompts. The complaint is the number of decisions, not their format, and `init` is frequently run non-interactively where a prompt cannot be answered. |
| Keep `Print`, but collapse the three blobs into one shared block | Fixes the duplication and none of the manual work. The user still hand-edits three files, and still has to discover which three. |
| Write the config only under an opt-in `travsr connect --write` | The opt-in already exists in the inverse direction (`--no-connect`). Adding a second flag means the default stays broken and only informed users escape it, which is the current situation restated. |
| Auto-grant the ADR-017 per-corpus trust so `travsr lang install` disappears in repos 2..N | Genuinely attractive and probably correct, but it is a change to a security boundary, not a UX change. It needs its own RFC and security sign-off, and bundling it would block this one behind that review. See Out of Scope. |
| Ship `--rules` on by default so agents actually prefer the graph | RFC-026 argues the rules file is what changes agent behaviour, and the shipped default disagrees with it for token-cost reasons. That disagreement is real and unresolved; it is a separate decision from whether the MCP server gets wired at all. |
| Do nothing; document the steps in the README better | The steps are already printed at the moment they are needed, which is strictly better placement than a README, and the feedback is that this is still too manual. Better prose does not reduce the step count. |

---

## Drawbacks

- **Travsr creates files in repositories it did not create files in before.** The
  mitigation is that these are gitignored, removable with
  `travsr connect --remove`, suppressible with `--no-connect`, and of the same
  class as `.travsrignore`. It is still a real behaviour change and some users
  will be surprised by it.
- **A monorepo used by one developer across several tools now accumulates
  several config files.** They are small and ignored, but they are clutter.
- **D1 widens the blast radius of any latent bug in the JSON merge path**, because
  that path now runs for many more users. The strict-JSON-or-skip rule bounds the
  damage to "we did nothing", not "we corrupted your config", but the merge code
  gets materially more exposure than it has today.
- **D2 makes first-run output depend on repository content**, so it is no longer a
  fixed string. Any test asserting the old literal must move to asserting shape.
- **D3 spends CPU on embedding at init time** for users who would not have run
  `travsr embed init`. The lane is already governed by the reindex resource
  config, but init becomes busier than it was.
- **This RFC does not make the first run zero-decision.** With ADR-017 unchanged, a
  non-builtin language in a second repository still requires
  `travsr lang install <lang>`. The 47 lines shrink substantially; they do not
  reach zero.

---

## Unresolved Questions

- **Which symbol does D2 pick?** Highest-degree node, first entry point, or first
  definition in the largest file. Whatever is chosen must be deterministic.
  Settled by trying each against a handful of real repositories and seeing which
  produces a query a newcomer would plausibly have asked. Author decides.
- **Does VS Code Copilot join D1?** Its current rule (auto-write only when
  `.vscode/mcp.json` already exists) is deliberate, because a bare `.vscode/` is
  present in many repositories without Copilot installed. Whether a home-level
  Copilot marker is strong enough evidence to create `.vscode/mcp.json` is a
  judgement about that tool's install footprint that this RFC does not make.
- **Should D3 be bounded by repository size?** Enabling embeddings automatically on
  a very large repository at init time may be surprising. A threshold above which
  it reverts to the tip may be warranted. Needs a number from the bench harness.
- **Is D4 in scope for this RFC at all?** It is a correctness bug, not UX, and
  could ship independently and sooner. It is documented here because it silently
  reports success for a language that did not run, which discredits the rest of
  this work. Reviewer decides whether to split it.

---

## Acceptance Criteria

On a clean repository containing at least one language, with at least two
supported AI tools installed on the machine and none configured in the repo:

```bash
travsr init < /dev/null > out.txt 2>&1
test "$(grep -c 'add manually' out.txt)" -eq 0      # D1
grep -q 'PaymentService' out.txt && exit 1          # D2
```

D1 additionally requires that the written files exist, parse, and are ignored:

```bash
test -f .mcp.json && python3 -m json.tool .mcp.json > /dev/null
git check-ignore -q .mcp.json
```

Non-clobbering must be pinned, not assumed. With a pre-existing `.mcp.json`
holding an unrelated server, after `travsr init` that server is still present and
byte-identical, and only the `travsr` key is added. With a `.mcp.json` that is
not strict JSON, the file is unchanged on disk and the run reports a skip.

D3: on a repo where `travsr embed status` reports an installed backend, after
`travsr init` the repo's embed lane is enabled and no `travsr embed init` tip is
printed.

D4 is verified by the case that produces it today. With `registered = ["go"]`,
no `travsr-lang-go` on disk, and Go files in the repo:

```bash
sqlite3 .travsr/graph.db "select value from meta where key='phase_b_warnings'"
# must contain skipped_no_analyzer:go, and must not be empty
travsr status | grep -q 'semantic: complete' && exit 1
```

`travsr status` and `travsr lang status` must agree about whether Go ran.

Existing suites must stay green, in particular the connect adapter tests that
pin strict-JSON-or-skip and managed-marker handling:

```bash
cargo test -p travsr-cli
cargo test -p travsr-plugin-host
```

---

## Out of Scope

- **ADR-017 Rule 3 per-corpus trust.** `travsr lang install` inside a repository is
  already treated as that repository's consent signal
  (`crates/travsr-cli/src/lang.rs:750`), and there is an argument that running
  `travsr init` there is an equal or stronger signal. That argument may well be
  right, but it changes a security boundary and belongs in its own RFC with
  security sign-off.
- **The `--rules` default.** See Alternatives.
- **PATH installation.** The `note:` about `~/.travsr/bin` is a packaging concern
  for the installer, not for `init`.
- **Any change to what the MCP server exposes.** This RFC only changes whether and
  where its configuration is written.
