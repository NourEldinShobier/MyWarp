//! Shared plumbing for backends that talk line-delimited text over a child's stdio.

use std::path::Path;
use std::process::Stdio;

use async_process::Command;
use futures::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};
use futures::stream::BoxStream;
use futures::{FutureExt as _, StreamExt as _};

use super::AgentEvent;

/// Spawns `program args…`, writes `stdin_line` plus a newline, then yields one parsed batch
/// per stdout line until the process exits. A non-zero exit with no `Done`/`Error` becomes an
/// `Error` carrying the tail of stderr. The child is killed when the stream is dropped.
pub(crate) fn run_lines(
    program: &str,
    args: Vec<String>,
    cwd: Option<&Path>,
    stdin_line: String,
    keep_stdin_open: bool,
    mut parse: impl FnMut(&str, &futures::channel::mpsc::UnboundedSender<String>) -> Vec<AgentEvent> + Send + 'static,
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
        let mut stdin = child.stdin.take();
        if let Some(stdin) = stdin.as_mut() {
            let write = async {
                stdin.write_all(stdin_line.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await
            };
            if let Err(e) = write.await {
                yield AgentEvent::Error(format!("Could not send the prompt to `{program}`: {e}"));
                return;
            }
        }
        // Closing stdin lets one-shot CLIs finish after the turn. It stays open while the CLI may
        // still ask for permission answers, and closes once the turn's result is in.
        if !keep_stdin_open {
            stdin = None;
        }
        let (to_stdin, mut from_parser) = futures::channel::mpsc::unbounded::<String>();
        let stderr = child.stderr.take();
        let mut lines = BufReader::new(child.stdout.take().expect("piped")).lines();
        let mut finished = false;
        loop {
            futures::select! {
                line = lines.next().fuse() => {
                    let Some(Ok(line)) = line else { break };
                    for ev in parse(&line, &to_stdin) {
                        finished |= matches!(ev, AgentEvent::Done | AgentEvent::Error(_));
                        yield ev;
                    }
                    if finished {
                        stdin = None;
                    }
                }
                msg = from_parser.next() => {
                    if let (Some(msg), Some(stdin)) = (msg, stdin.as_mut()) {
                        let _ = stdin.write_all(msg.as_bytes()).await;
                        let _ = stdin.write_all(b"\n").await;
                        let _ = stdin.flush().await;
                    }
                }
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
