//! #863 emission contracts for the cross-checkout note, driven through the real
//! binary from a real linked worktree.
//!
//! Both properties here live in the wiring between `repo.rs`, `daemon_client.rs`
//! and the individual commands, and neither is visible to a unit test: the
//! emitter reads the process's own working directory, and what is being pinned
//! is which note reaches whose stderr. A unit test on the note string alone
//! stays green through either regression.

use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;

fn git(dir: &Path, args: &[&str]) {
    let ok = StdCommand::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("git command failed")
        .success();
    assert!(ok, "git {args:?} failed in {}", dir.display());
}

fn travsr(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("travsr").unwrap();
    c.env("TRAVSR_DISABLE_REGISTRY", "1") // UX-017: don't pollute the real registry
        .env_remove("TRAVSR_NO_WORKTREE_NOTE")
        .current_dir(dir);
    c
}

/// stderr of a successful run, with stdout ignored: every note under test is a
/// warning, and stdout staying clean is the other half of the contract.
fn stderr_of(cmd: &mut Command, args: &[&str]) -> String {
    let out = cmd
        .args(args)
        .assert()
        .success()
        .get_output()
        .stderr
        .clone();
    String::from_utf8_lossy(&out).into_owned()
}

/// A committed git repo holding the TS fixture, indexed by `travsr init`, plus a
/// linked worktree with no `.travsr/` of its own so its reads redirect to the
/// main index (the #302 `PreferLocalElseMain` path).
///
/// `travsr init` runs Phase A only, so the resulting index is genuinely Phase B
/// degraded (`semantic: not run`). That is a precondition rather than an
/// accident, so the tests below assert it rather than assume it — without it the
/// degraded-caveat test would pass while proving nothing.
fn main_and_worktree() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let main = tmp.path();
    git(main, &["-c", "init.defaultBranch=main", "init", "-q"]);
    git(main, &["config", "user.email", "test@test.com"]);
    git(main, &["config", "user.name", "Test"]);
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ts-small/a.ts"),
        main.join("a.ts"),
    )
    .unwrap();
    git(main, &["add", "-A"]);
    git(main, &["commit", "-qm", "init"]);
    travsr(main).arg("init").assert().success();

    let wt = main.join("wt");
    git(main, &["worktree", "add", "-q", wt.to_str().unwrap()]);
    assert!(
        wt.join(".git").is_file(),
        "a linked worktree's .git must be a gitlink file"
    );
    (tmp, wt)
}

/// The folded-in degraded caveat must reach only the commands whose answer rides
/// call edges.
///
/// `pattern` greps a file set and `graph --direction deps` rides Phase A import
/// edges: both are complete whether or not Phase B has run, which is why each
/// deliberately skips the standalone Phase B warning. Attaching the caveat to
/// the cross-checkout note instead of to the commands that ride call edges put
/// that exact claim back in front of both of them, telling their users to
/// distrust a result that is fine. Standing in a worktree must not be enough to
/// bring it back.
#[test]
fn the_degraded_caveat_reaches_only_commands_that_ride_call_edges() {
    let (tmp, wt) = main_and_worktree();
    let main = tmp.path();

    // Precondition: the served index really is Phase B degraded, stated in the
    // wording the caveat replaces. If `init` ever starts running Phase B, this
    // fails here rather than letting the assertions below pass vacuously.
    let control = stderr_of(&mut travsr(main), &["references", "hello"]);
    assert!(
        control.contains("has not caught up"),
        "precondition: the main index must be Phase B degraded, got: {control}"
    );

    for args in [
        &["pattern", "hello"][..],
        &["graph", "a.ts", "--direction", "deps"][..],
    ] {
        let stderr = stderr_of(&mut travsr(&wt), args);
        assert!(
            stderr.contains("different checkout"),
            "{args:?} must still say which tree answered, got: {stderr}"
        );
        assert!(
            !stderr.contains("not authoritative"),
            "{args:?} reads no call edges, so it must not carry the degraded \
             caveat, got: {stderr}"
        );
        // The hatch is discoverable only from the note itself.
        assert!(
            stderr.contains("TRAVSR_NO_WORKTREE_NOTE=1"),
            "{args:?}: the note must name its own escape hatch, got: {stderr}"
        );
    }

    // The command that does ride call edges keeps the caveat, so the exemption
    // above is a split rather than a blanket removal.
    let stderr = stderr_of(&mut travsr(&wt), &["references", "hello"]);
    assert!(
        stderr.contains("different checkout") && stderr.contains("not authoritative"),
        "references rides call edges and must keep the caveat, got: {stderr}"
    );
    assert!(
        !stderr.contains("has not caught up"),
        "the caveat must not resurrect the freshness advice, got: {stderr}"
    );
}

/// `TRAVSR_NO_WORKTREE_NOTE=1` must silence the cross-checkout note without
/// handing back the two notes it replaces.
///
/// Reporting "not a cross-checkout" for a muted note made the hatch a net loss:
/// callers key the suppression of the Phase B freshness note and the #645 drift
/// note to that answer, so opting out of one accurate note printed two
/// inaccurate ones instead, and neither of their remedies ("wait for the daemon
/// to reconcile", "run `travsr status`") can ever apply here.
///
/// Driven with the worktree's HEAD moved past the indexed commit, which is what
/// arms the drift note.
#[test]
fn the_hatch_silences_the_note_without_resurrecting_the_ones_it_replaces() {
    let (_tmp, wt) = main_and_worktree();

    // Move this worktree's HEAD off the indexed commit so the drift note is
    // armed; without this the test could pass with the drift note simply
    // inapplicable.
    std::fs::write(wt.join("a.ts"), "// drift\n").unwrap();
    git(&wt, &["add", "-A"]);
    git(&wt, &["commit", "-qm", "drift"]);

    let armed = stderr_of(&mut travsr(&wt), &["references", "hello"]);
    assert!(
        armed.contains("different checkout"),
        "precondition: the note must fire without the hatch, got: {armed}"
    );

    for args in [&["references", "hello"][..], &["status"][..]] {
        let stderr = stderr_of(travsr(&wt).env("TRAVSR_NO_WORKTREE_NOTE", "1"), args);
        assert!(
            !stderr.contains("different checkout"),
            "{args:?}: the hatch must silence the note, got: {stderr}"
        );
        assert!(
            !stderr.contains("has not caught up"),
            "{args:?}: muting the accurate note must not restore the freshness \
             claim, got: {stderr}"
        );
        assert!(
            !stderr.contains("reconcile"),
            "{args:?}: muting the accurate note must not restore the drift \
             note, got: {stderr}"
        );
    }
}
