//! Identifier segmentation — the single source of truth for "what words is
//! this identifier made of". Used by the FTS tokenizer (index population, via
//! `travsr_store::fts_tokenize::tokenize_identifier`) and by the retrieval
//! boundary predicate (anchor admission, lexical boundary evidence, via
//! `contains_token`). Before this module existed, these two questions were
//! answered by two different segmenters that disagreed (#478 RFC-023
//! Evidence C/D) — camelCase-aware `tokenize_identifier` built the index, while
//! a punctuation-only boundary check gated anchoring, rejecting genuine
//! anchors like `"sqlite"` in `fn:SqliteStore.exec_ddl`. Keeping both on one
//! segmenter is the point.

/// Split `s` into lowercased word segments.
///
/// Delimiters: every non-ASCII-alphanumeric character. Also splits on
/// lower→upper transitions (`getCallers` → `get`, `callers`) and on
/// letter↔digit transitions (`utf8Decode` → `utf`, `8`, `decode`). Repeated
/// segments are suppressed, keeping only the first occurrence (matches the
/// pre-extraction `tokenize_identifier` behaviour exactly — that function
/// builds a bag-of-words FTS field, where position doesn't matter and a
/// repeated word is redundant). Do not reuse this for anything that needs
/// the segments' true position in `s` — see [`raw_segments`].
///
/// KNOWN LIMITATION (deliberate, see RFC-023 §13): consecutive-uppercase
/// acronym runs are not split — `HTTPServer` → `["httpserver"]`. Changing
/// this changes `nodes_fts`/`nodes_fts_words` content and forces a reindex.
///
/// O(n) in the length of `s`, single pass.
pub fn segments(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for tok in raw_segments(s) {
        if !tokens.contains(&tok) {
            tokens.push(tok);
        }
    }
    tokens
}

/// Same splitting as [`segments`], but keeps every occurrence in its original
/// position — no "first occurrence only" dedup. [`contains_token`]'s
/// contiguous-run check needs the text's *true* segment sequence: `segments`'
/// dedup drops a segment that recurs elsewhere in `s`, which can sever the
/// contiguity between that segment and its real neighbour (e.g.
/// `"path_to_path_map"` deduped is `["path","to","map"]` — the second `path`,
/// contiguous with `map` in the source, is gone, so `"pathmap"` would wrongly
/// fail to match). Never use this for FTS indexing — that output must match
/// `tokenize_identifier`'s bag-of-words contract exactly, which is `segments`.
fn raw_segments(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    for i in 0..len {
        let c = chars[i];

        if !c.is_ascii_alphanumeric() {
            flush(&mut current, &mut tokens);
            continue;
        }

        if c.is_ascii_uppercase() {
            if i > 0 && (chars[i - 1].is_ascii_lowercase() || chars[i - 1].is_ascii_digit()) {
                flush(&mut current, &mut tokens);
            }
            current.push(c.to_ascii_lowercase());
        } else if c.is_ascii_digit() {
            if i > 0 && chars[i - 1].is_ascii_alphabetic() {
                flush(&mut current, &mut tokens);
            }
            current.push(c);
        } else {
            if i > 0 && chars[i - 1].is_ascii_digit() {
                flush(&mut current, &mut tokens);
            }
            current.push(c);
        }
    }
    flush(&mut current, &mut tokens);

    tokens
}

#[inline]
fn flush(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

/// True when `token` appears in `text` as a whole word segment, or as a
/// contiguous run of segments joined (`"sqlitestore"` matches
/// `["sqlite","store"]`).
///
/// `token` is normalised the same way as `text` before comparison — every
/// non-ASCII-alphanumeric byte is stripped and the result lowercased — so a
/// token that carries its own delimiters (a raw content token like
/// `"seed_target"`, not yet segmented by the caller) still matches a
/// contiguous run of the text's segments (`"seedtarget"` == `"seed"+"target"`).
/// This is the replacement for `travsr_mcp::seed::token_is_sig_component`,
/// which required non-alphanumeric boundaries and so rejected camelCase/
/// PascalCase anchors.
///
/// Uses [`raw_segments`], not [`segments`]: this check is positional (a
/// contiguous run in `text`), and `segments`' first-occurrence dedup would
/// silently break contiguity for a segment that recurs elsewhere in `text`.
pub fn contains_token(token: &str, text: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let token_norm: String = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if token_norm.is_empty() {
        return false;
    }
    let segs = raw_segments(text);

    for start in 0..segs.len() {
        let mut acc = String::new();
        for seg in &segs[start..] {
            acc.push_str(seg);
            if acc == token_norm {
                return true;
            }
            if acc.len() > token_norm.len() {
                break;
            }
        }
    }
    false
}

/// Leaf identifier of a `kind:Qualified.Name` signature: the segment after the
/// last `.` of the body (`method:HttpResponse.render` -> `render`), or the whole
/// body when unqualified (`class:HttpResponse` -> `HttpResponse`, `fn:index` ->
/// `index`).
///
/// Lives here, in the one crate both sides already depend on, because the
/// daemon's leaf-uniqueness call resolver and the store's fuzzy symbol
/// correction must agree on what counts as "the same callee". They previously
/// carried byte-identical private copies; a future change to the splitting
/// rules would have silently applied to only one of them.
pub fn leaf_of(sig: &str) -> &str {
    let body = sig.split_once(':').map(|(_, r)| r).unwrap_or(sig);
    body.rsplit('.').next().unwrap_or(body)
}

/// The leading keyword of an Objective-C selector, or `name` unchanged.
///
/// Objective-C spells one method name across several keywords
/// (`policyWithPinningMode:withPinnedCertificates:`), and Phase A stores that
/// whole selector as the node's leaf so two selectors sharing a leading keyword
/// stay distinct nodes. A developer, though, types and reads the leading
/// keyword alone, the only part that appears at a call site as a contiguous
/// word, so symbol lookup has to be able to get back from the stored selector
/// to that head.
///
/// A stored selector always carries a trailing `:`, which is what identifies
/// one here. No other language's leaf ends in `:`: a Rust path
/// (`crate::repo::find_git_root`) or a C++ qualified name (`Widget::draw`)
/// spells its separator as `::` and never terminates on it.
///
/// ```text
/// selector_head("policyWithPinningMode:withPinnedCertificates:") == "policyWithPinningMode"
/// selector_head("reload")                                        == "reload"
/// selector_head("crate::repo::find_git_root")   == "crate::repo::find_git_root"
/// ```
pub fn selector_head(name: &str) -> &str {
    if name.ends_with(':') {
        name.split(':').next().unwrap_or(name)
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(s: &str) -> Vec<String> {
        segments(s)
    }

    #[test]
    fn snake_case() {
        assert_eq!(segs("dispatch_tool_call"), vec!["dispatch", "tool", "call"]);
    }

    #[test]
    fn camel_case() {
        assert_eq!(segs("getCallersGlobal"), vec!["get", "callers", "global"]);
    }

    #[test]
    fn pascal_case() {
        assert_eq!(segs("SqliteStore"), vec!["sqlite", "store"]);
    }

    #[test]
    fn screaming_snake() {
        assert_eq!(segs("MAX_CONTEXT_BUDGET"), vec!["max", "context", "budget"]);
    }

    #[test]
    fn utf8_decode_letter_digit_boundary() {
        assert_eq!(segs("utf8Decode"), vec!["utf", "8", "decode"]);
    }

    #[test]
    fn parse2_url_digit_boundary() {
        assert_eq!(segs("parse2Url"), vec!["parse", "2", "url"]);
    }

    #[test]
    fn dotted_signature() {
        assert_eq!(
            segs("fn:SqliteStore.exec_ddl"),
            vec!["fn", "sqlite", "store", "exec", "ddl"]
        );
    }

    #[test]
    fn path_with_extension() {
        assert_eq!(
            segs("packages/travsr-lsif-ts/src/walker.ts"),
            vec!["packages", "travsr", "lsif", "ts", "src", "walker"]
        );
    }

    #[test]
    fn empty_string() {
        assert_eq!(segs(""), Vec::<String>::new());
    }

    #[test]
    fn single_char() {
        assert_eq!(segs("x"), vec!["x"]);
    }

    #[test]
    fn all_punctuation() {
        assert_eq!(segs("::--__"), Vec::<String>::new());
    }

    #[test]
    fn token_longer_than_text() {
        assert!(!contains_token("verylongtoken", "fn:walk"));
    }

    #[test]
    fn non_ascii_treated_as_delimiter() {
        // Non-ASCII alphanumeric (e.g. accented letters) are not
        // ascii_alphanumeric, so they act as delimiters like punctuation.
        assert_eq!(segs("café_bar"), vec!["caf", "bar"]);
    }

    // ── contains_token: the #478 behaviour-change table ──────────────────────

    #[test]
    fn works_does_not_match_workspace() {
        assert!(!contains_token("works", "provideWorkspaceChatContext"));
    }

    #[test]
    fn auth_does_not_match_authentication() {
        assert!(!contains_token("auth", "fn:authentication"));
    }

    #[test]
    fn wal_does_not_match_walk() {
        assert!(!contains_token("wal", "fn:walk"));
    }

    #[test]
    fn ppr_matches_ppr_inner() {
        assert!(contains_token("ppr", "fn:ppr_inner"));
    }

    #[test]
    fn sqlite_matches_sqlite_store_method() {
        assert!(contains_token("sqlite", "fn:SqliteStore.exec_ddl"));
    }

    #[test]
    fn store_matches_sqlite_store_method() {
        assert!(contains_token("store", "fn:SqliteStore.exec_ddl"));
    }

    #[test]
    fn sqlitestore_matches_via_contiguous_run() {
        assert!(contains_token("sqlitestore", "fn:SqliteStore.exec_ddl"));
    }

    #[test]
    fn token_with_its_own_delimiters_matches_contiguous_run() {
        // A raw content token that still carries its own separator (not yet
        // segmented by the caller) must match the same way a fused token
        // does — the token's own delimiters are stripped before comparing.
        assert!(contains_token("seed_target", "fn:seed_target"));
        assert!(contains_token("Seed_Target", "fn:seed_target"));
    }

    #[test]
    fn contiguous_run_matches_across_a_segment_that_recurs_elsewhere() {
        // `segments()` dedups to ["get", "config", "name"] (the second "get"
        // is a repeat of the first, so it's dropped) — using that deduped
        // list for positional matching would sever the contiguity between
        // the *second* "get" and "name", even though "get_name" is a real
        // contiguous run at the end of the source string. `contains_token`
        // must use the non-deduped segmentation for this to match.
        assert!(contains_token("getname", "get_config_get_name"));
        // Same shape, repeated segment in the middle rather than at the front.
        assert!(contains_token("pathmap", "fn:path_to_path_map"));
    }

    #[test]
    fn http_does_not_match_httpserver_acronym_limitation() {
        // Deliberate known limitation (RFC-023 §13): acronym runs are not
        // split, so "http" is not a segment of "httpserver". Pinned here so
        // the acronym-segmentation follow-up has a failing-by-design anchor
        // to flip.
        assert!(!contains_token("http", "HTTPServer"));
    }

    #[test]
    fn selector_head_takes_the_leading_keyword() {
        assert_eq!(
            selector_head("policyWithPinningMode:withPinnedCertificates:"),
            "policyWithPinningMode"
        );
        assert_eq!(
            selector_head("policyWithPinningMode:"),
            "policyWithPinningMode"
        );
        assert_eq!(selector_head("anon::"), "anon");
    }

    #[test]
    fn selector_head_leaves_every_other_leaf_alone() {
        assert_eq!(selector_head("reload"), "reload");
        assert_eq!(
            selector_head("crate::repo::find_git_root"),
            "crate::repo::find_git_root"
        );
        assert_eq!(selector_head("Widget::draw"), "Widget::draw");
        assert_eq!(selector_head(""), "");
    }

    #[test]
    fn leaf_of_keeps_a_whole_objc_selector() {
        // The kind prefix is split at the FIRST colon, so the selector's own
        // colons survive into the leaf.
        assert_eq!(
            leaf_of("method:AFSecurityPolicy.policyWithPinningMode:withPinnedCertificates:"),
            "policyWithPinningMode:withPinnedCertificates:"
        );
    }

    #[test]
    fn leaf_of_splits_kind_prefix_and_qualifier() {
        assert_eq!(leaf_of("method:HttpResponse.render"), "render");
        assert_eq!(leaf_of("fn:Session.filter"), "filter");
        // Unqualified body: the whole body is the leaf.
        assert_eq!(leaf_of("class:HttpResponse"), "HttpResponse");
        assert_eq!(leaf_of("fn:index"), "index");
        // Deeply qualified: only the last segment.
        assert_eq!(leaf_of("fn:a.b.c.d"), "d");
        // No kind prefix at all.
        assert_eq!(leaf_of("bare_name"), "bare_name");
        // The kind prefix is stripped before the qualifier split, so a dot in
        // the prefix position cannot be mistaken for a qualifier boundary.
        assert_eq!(leaf_of("fn:"), "");
    }
}
