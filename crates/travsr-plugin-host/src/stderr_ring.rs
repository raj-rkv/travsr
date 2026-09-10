// Bounded ring buffer that drains a child's stderr on a reader thread so a
// corrupt-model / OOM message (which the sidecar prints then exits) is captured
// and can be surfaced via tracing on non-zero exit (FT-M2). Bounded in BOTH
// directions so a chatty sidecar cannot grow memory without limit: at most
// MAX_LINES lines, each at most MAX_LINE_BYTES.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::process::ChildStderr;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

const MAX_LINES: usize = 64;
// A JVM analyzer (Gradle, sbt, KLS) emits hundreds of stderr lines, and the line
// that explains the failure ("FAILURE:", "* What went wrong:", the sidecar's own
// startup warning) appears EARLY. Evicting from the front dropped exactly that
// line and kept only trailing noise, so the first HEAD_LINES are pinned and the
// middle is discarded instead. Same MAX_LINES line bound.
const HEAD_LINES: usize = 16;
// A line bound is not a memory bound on its own: the sidecar, or the repo build
// tool it drives, chooses where the newlines go, so one newline-free write is
// one line of arbitrary size. Keep this much of a line and discard the rest of
// it; the total is then MAX_LINES * MAX_LINE_BYTES.
const MAX_LINE_BYTES: usize = 8 * 1024;

/// The captured lines plus the count discarded from the middle. Both are only
/// ever touched under the one mutex, so they live together rather than pairing a
/// `Mutex` with an atomic that is correct by accident of lock placement.
#[derive(Default)]
struct Ring {
    lines: VecDeque<String>,
    elided: usize,
}

pub(crate) struct StderrRing {
    ring: Arc<Mutex<Ring>>,
    handle: Option<JoinHandle<()>>,
}

/// Read one line into `out`, keeping at most [`MAX_LINE_BYTES`] of it and
/// discarding the remainder up to the newline. Returns false at EOF.
///
/// `BufReader::lines()` cannot do either half of this. It reads to `\n` with no
/// limit, so a 4 GiB newline-free write is one unbounded `String`; and it yields
/// `Err(InvalidData)` on the first non-UTF-8 byte, which `map_while(Result::ok)`
/// turned into a silent, permanent end of capture for the whole run.
fn read_bounded_line(rdr: &mut impl BufRead, out: &mut Vec<u8>) -> bool {
    out.clear();
    let read = rdr
        .by_ref()
        .take(MAX_LINE_BYTES as u64)
        .read_until(b'\n', out)
        .unwrap_or(0);
    if read == 0 {
        return false;
    }
    // Over the per-line bound: drop the rest of this line so the bound holds for
    // the stream, not just for its first chunk.
    while out.last() != Some(&b'\n') {
        let (consumed, done) = match rdr.fill_buf() {
            Ok([]) | Err(_) => break,
            Ok(buf) => match buf.iter().position(|&b| b == b'\n') {
                Some(i) => (i + 1, true),
                None => (buf.len(), false),
            },
        };
        rdr.consume(consumed);
        if done {
            break;
        }
    }
    true
}

impl StderrRing {
    /// Create a ring with no backing reader (used when stderr is not piped).
    pub(crate) fn spawn_empty() -> Self {
        Self {
            ring: Arc::new(Mutex::new(Ring::default())),
            handle: None,
        }
    }

    /// Take the child's piped stderr and start draining it. Caller must have
    /// spawned with `.stderr(Stdio::piped())`.
    pub(crate) fn spawn(stderr: ChildStderr) -> Self {
        let ring = Arc::new(Mutex::new(Ring {
            lines: VecDeque::with_capacity(MAX_LINES),
            elided: 0,
        }));
        let ring_w = Arc::clone(&ring);
        let handle = std::thread::Builder::new()
            .name("sidecar-stderr".into())
            .spawn(move || {
                let mut rdr = BufReader::new(stderr);
                let mut raw: Vec<u8> = Vec::new();
                while read_bounded_line(&mut rdr, &mut raw) {
                    // Strip the terminator `lines()` used to strip, then decode
                    // lossily: one stray byte costs one replacement character,
                    // not the rest of the run's stderr.
                    let mut end = raw.len();
                    if end > 0 && raw[end - 1] == b'\n' {
                        end -= 1;
                    }
                    if end > 0 && raw[end - 1] == b'\r' {
                        end -= 1;
                    }
                    let line = String::from_utf8_lossy(&raw[..end]).into_owned();
                    let mut r = ring_w.lock().unwrap_or_else(|e| e.into_inner());
                    if r.lines.len() == MAX_LINES {
                        r.lines.remove(HEAD_LINES);
                        r.elided += 1;
                    }
                    r.lines.push_back(line);
                }
            })
            .ok();
        Self { ring, handle }
    }

    /// Snapshot the captured lines, oldest first, joined by newline: the pinned
    /// opening lines, an elision marker once the middle was dropped, then the
    /// most recent lines.
    pub(crate) fn tail(&self) -> String {
        let r = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<String> = r.lines.iter().take(HEAD_LINES).cloned().collect();
        if r.elided > 0 {
            out.push(format!("... {} lines elided ...", r.elided));
        }
        out.extend(r.lines.iter().skip(HEAD_LINES).cloned());
        out.join("\n")
    }
}

impl Drop for StderrRing {
    fn drop(&mut self) {
        // The reader thread ends on stderr EOF (child death closes the write end).
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(unix)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    /// Run `script`, drain its stderr into a ring, and return the ring.
    fn ring_for(script: &str) -> StderrRing {
        let mut child = Command::new("sh")
            .args(["-c", script])
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sh");
        let stderr = child.stderr.take().expect("piped stderr");
        let ring = StderrRing::spawn(stderr);
        let _ = child.wait();
        // Give the reader thread time to drain.
        std::thread::sleep(std::time::Duration::from_millis(200));
        ring
    }

    #[test]
    fn stderr_ring_captures_lines() {
        let tail = ring_for("echo one 1>&2; echo two 1>&2").tail();
        assert!(tail.contains("one"), "expected 'one' in: {tail}");
        assert!(tail.contains("two"), "expected 'two' in: {tail}");
    }

    #[test]
    fn stderr_ring_bounds_to_max_lines() {
        // Write MAX_LINES+10 lines; only MAX_LINES must be kept.
        let ring = ring_for(&format!(
            "for i in $(seq 1 {}); do echo \"line$i\" 1>&2; done",
            MAX_LINES + 10
        ));
        let r = ring.ring.lock().expect("ring lock");
        let buf = &r.lines;
        assert!(
            buf.len() <= MAX_LINES,
            "ring must not exceed MAX_LINES, got {}",
            buf.len()
        );
        // The last line must be the highest-numbered one.
        let last = buf.back().expect("must have lines");
        assert!(
            last.contains(&(MAX_LINES + 10).to_string()),
            "last line should be line{}, got: {last}",
            MAX_LINES + 10
        );
    }

    // The reason a JVM analyzer failed is printed in its FIRST lines. Evicting
    // from the front dropped them; head + tail must keep both ends.
    #[test]
    fn stderr_ring_keeps_head_and_tail() {
        let total = MAX_LINES * 4;
        let rendered = ring_for(&format!(
            "echo 'FAILURE: Build failed with an exception.' 1>&2; \
             for i in $(seq 2 {total}); do echo \"noise$i\" 1>&2; done"
        ))
        .tail();
        assert!(
            rendered.contains("FAILURE: Build failed"),
            "the first line is the cause and must survive: {rendered}"
        );
        assert!(
            rendered.contains(&format!("noise{total}")),
            "the last line must survive: {rendered}"
        );
        assert!(
            rendered.contains("lines elided"),
            "the discarded middle must be marked: {rendered}"
        );
    }

    // A line bound is not a memory bound: the writer chooses where the newlines
    // go. One 200 KB newline-free write must cost MAX_LINE_BYTES, and capture
    // must resynchronise on the next newline rather than swallowing it.
    #[test]
    fn stderr_ring_bounds_line_length() {
        let ring = ring_for(
            "head -c 200000 /dev/zero | tr '\\000' a 1>&2; echo 1>&2; echo tailmarker 1>&2",
        );
        let r = ring.ring.lock().expect("ring lock");
        for line in &r.lines {
            assert!(
                line.len() <= MAX_LINE_BYTES,
                "a single line must not exceed MAX_LINE_BYTES, got {}",
                line.len()
            );
        }
        assert!(
            r.lines.iter().any(|l| l == "tailmarker"),
            "capture must resume after an over-long line: {:?}",
            r.lines
        );
    }

    // `lines()` yields Err on the first non-UTF-8 byte and `map_while` ended the
    // whole run's capture there. One bad byte must cost one line's fidelity.
    #[test]
    fn stderr_ring_survives_invalid_utf8() {
        let tail = ring_for("printf 'good\\n\\377\\nafter\\n' 1>&2").tail();
        assert!(tail.contains("good"), "expected 'good' in: {tail}");
        assert!(
            tail.contains("after"),
            "capture must continue past a non-UTF-8 byte: {tail}"
        );
    }

    // Under the bound nothing is elided and the output is verbatim.
    #[test]
    fn stderr_ring_no_marker_when_under_bound() {
        let rendered = ring_for("for i in $(seq 1 5); do echo \"line$i\" 1>&2; done").tail();
        assert_eq!(rendered, "line1\nline2\nline3\nline4\nline5");
    }
}
