//! Shared plumbing for backends that talk line-delimited text over a child's stdio.

use std::path::Path;
use std::process::Stdio;

use async_process::Command;
use futures::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};
use futures::stream::BoxStream;
use futures::StreamExt as _;

use super::AgentEvent;

/// Spawns `program args…`, writes `stdin_line` plus a newline, then yields one parsed batch
/// per stdout line until the process exits. A non-zero exit with no `Done`/`Error` becomes an
/// `Error` carrying the tail of stderr. The child is killed when the stream is dropped.
pub(crate) fn run_lines(
    program: &str,
    args: Vec<String>,
    cwd: Option<&Path>,
    stdin_line: String,
    mut parse: impl FnMut(&str) -> Vec<AgentEvent> + Send + 'static,
) -> BoxStream<'static, AgentEvent> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    // Headless: no console window flashes up for the child (Windows only).
    #[cfg(windows)]
    {
        use async_process::windows::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let program = program.to_owned();

    Box::pin(async_stream::stream! {
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                yield AgentEvent::Error(format!("Could not start `{program}`: {e}"));
                return;
            }
        };
        if let Some(mut stdin) = child.stdin.take() {
            let write = async {
                stdin.write_all(stdin_line.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await
            };
            if let Err(e) = write.await {
                yield AgentEvent::Error(format!("Could not send the prompt to `{program}`: {e}"));
                return;
            }
            // Dropping stdin closes it, which lets one-shot CLIs finish after the turn.
            drop(stdin);
        }
        let stderr = child.stderr.take();
        let mut lines = BufReader::new(child.stdout.take().expect("piped")).lines();
        let mut finished = false;
        while let Some(Ok(line)) = lines.next().await {
            for ev in parse(&line) {
                finished |= matches!(ev, AgentEvent::Done | AgentEvent::Error(_));
                yield ev;
            }
        }
        if !finished {
            let mut tail = String::new();
            if let Some(mut err) = stderr {
                let _ = err.read_to_string(&mut tail).await;
            }
            let tail = tail.trim();
            let tail: String = tail.chars().rev().take(500).collect::<Vec<_>>().into_iter().rev().collect();
            let ok = child.status().await.map(|s| s.success()).unwrap_or(false);
            yield if ok { AgentEvent::Done } else if tail.is_empty() {
                AgentEvent::Error(format!("`{program}` exited unexpectedly"))
            } else {
                AgentEvent::Error(tail)
            };
        }
    })
}
