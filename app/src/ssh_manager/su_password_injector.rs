//! su password confirmation prompt. Continuously watches PTY output; when the user types `su root` / `su - root`
//! or a similar command to switch to root and a password prompt appears, shows a confirmation menu, and after the user confirms injects the root password
//! or the shared OneKey password.
//!
//! Injects only for the root target; switching to other users (e.g. `su lg`) does not trigger it.
//! Waits for the shell prompt (meaning the SSH login has finished) before detecting, to avoid conflicting with the login password.
//! Uses `spawn_stream_local` + `stream!` for continuous watching, so every `su root` triggers.

use std::sync::Arc;
use std::time::Duration;

use async_broadcast::InactiveReceiver;
use async_stream::stream;
use lazy_static::lazy_static;
use regex::bytes::Regex;
use warpui::r#async::FutureExt;
use warpui::{ViewContext, WeakViewHandle};
use zeroize::Zeroizing;

use crate::ssh_manager::shell_prompt::bytes_look_like_shell_prompt;
use crate::terminal::TerminalView;

const SLIDING_WINDOW_BYTES: usize = 8 * 1024;
const BUFFER_HARD_LIMIT: usize = 16 * 1024;
/// Maximum time phase 1 waits for the shell prompt. On timeout the whole stream is abandoned (and
/// `in_flight` is reset in `on_done`).
const SHELL_READY_TIMEOUT: Duration = Duration::from_secs(30);

lazy_static! {
    /// Password prompt regex — strictly matches two kinds:
    /// 1. `password` / `passphrase` / `密码` lines ending with an ASCII colon `:` or a full-width colon `：`
    /// 2. Kylin V10's colonless `输入密码`
    ///
    /// The old implementation made the colon optional, so any line ending with "password" (such as
    /// `Your password has expired`) produced false positives.
    static ref PASSWORD_PROMPT_REGEX: Regex = Regex::new(
        r"(?im)(?:(?:password|passphrase|密码)[^\n]*(?::|：)\s*$|输入密码\s*$)"
    )
    .expect("su password prompt regex must compile");

    /// su command regex — matches su commands targeting root (at end of line):
    /// `su` / `su -` / `su -l` / `su --login` / `su root` / `su - root` /
    /// `su -l root` / `su --login root`. Does not match forms that switch to other users, such as `su lg` / `su - lg`;
    /// `sudo su` still hits the trailing `su` thanks to the `su` word boundary.
    static ref SU_ROOT_CMD_REGEX: Regex =
        Regex::new(r"(?m)\bsu(?:\s+(?:-l?|--login|-))*(?:\s+root)?\s*$")
            .expect("su root cmd regex must compile");
}

/// Spawns the continuous su password watch stream in the owner context.
pub fn spawn_su_password_injector<O>(
    pty_reads_rx: Option<InactiveReceiver<Arc<Vec<u8>>>>,
    terminal_view: WeakViewHandle<TerminalView>,
    root_password: Option<Zeroizing<String>>,
    ctx: &mut ViewContext<O>,
) where
    O: warpui::View + 'static,
{
    let Some(rx) = pty_reads_rx else {
        log::debug!("ssh su password injector: no pty_reads_rx — skip");
        return;
    };
    let Some(root_password) = root_password.filter(|password| !password.is_empty()) else {
        log::debug!("ssh su password injector: empty root password - skip");
        return;
    };
    // Set the in-flight flag to stop the OneKey credential picker from popping up while waiting for the shell prompt.
    if let Some(view) = terminal_view.upgrade(ctx) {
        view.update(ctx, |view, _| {
            view.set_ssh_secret_auto_injection_in_flight(true);
        });
    }

    let prompt_stream = stream! {
        let mut active = rx.activate_cloned();
        let mut buf: Vec<u8> = Vec::with_capacity(SLIDING_WINDOW_BYTES);

        // Phase 1: wait for the shell prompt (times out after SHELL_READY_TIMEOUT), which means login is complete
        loop {
            match active.recv().with_timeout(SHELL_READY_TIMEOUT).await {
                Ok(Ok(chunk)) => {
                    buf.extend_from_slice(&chunk);
                    if buf.len() > BUFFER_HARD_LIMIT {
                        let drop_n = buf.len() - SLIDING_WINDOW_BYTES;
                        buf.drain(..drop_n);
                    }
                    if bytes_look_like_shell_prompt(&buf) {
                        break;
                    }
                }
                _ => return,
            }
        }

        // Phase 2: continuously detect su root + password prompt, keep watching after each yield
        buf.clear();
        while let Ok(chunk) = active.recv().await {
            buf.extend_from_slice(&chunk);
            if buf.len() > BUFFER_HARD_LIMIT {
                let drop_n = buf.len() - SLIDING_WINDOW_BYTES;
                buf.drain(..drop_n);
            }
            if PASSWORD_PROMPT_REGEX.is_match(&buf) && is_su_to_root(&buf) {
                buf.clear();
                yield ();
            }
        }
    };

    // on_done must reset in_flight: if phase 1 (waiting for the shell prompt) times out or hits EOF and
    // `return`s out of the stream directly, on_item has not run yet; without a reset in on_done,
    // OneKey would be blocked permanently on this terminal.
    let terminal_view_done = terminal_view.clone();
    let _ = ctx.spawn_stream_local(
        prompt_stream,
        move |_owner, (), ctx| {
            let Some(view) = terminal_view.upgrade(ctx) else {
                return;
            };
            view.update(ctx, |view, ctx| {
                view.su_root_password = Some(root_password.clone());
                view.show_su_root_confirm_menu(ctx);
                view.set_ssh_secret_auto_injection_in_flight(false);
            });
        },
        move |_owner, ctx| {
            if let Some(view) = terminal_view_done.upgrade(ctx) {
                view.update(ctx, |view, _| {
                    view.set_ssh_secret_auto_injection_in_flight(false);
                });
            }
        },
    );
}

/// Checks whether the buffer contains an su command targeting root.
fn is_su_to_root(buf: &[u8]) -> bool {
    SU_ROOT_CMD_REGEX.is_match(buf)
}

pub(crate) fn should_spawn_su_password_injector(root_password: Option<&Zeroizing<String>>) -> bool {
    root_password.is_some_and(|password| !password.is_empty())
}

#[cfg(test)]
#[path = "su_password_injector_tests.rs"]
mod tests;
