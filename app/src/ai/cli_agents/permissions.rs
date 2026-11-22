//! Tool-permission prompts from the CLI (`--permission-prompt-tool stdio`).
//!
//! When the CLI wants to run a tool that needs approval it sends a `control_request` and waits.
//! The request is parked here under the tool's card id; the card shows Approve / Reject, and the
//! answer is written back to the CLI's stdin through the channel saved with the request.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use futures::channel::mpsc::UnboundedSender;
use serde_json::{json, Value};

struct Pending {
    request_id: String,
    input: Value,
    stdin: UnboundedSender<String>,
}

static PENDING: LazyLock<Mutex<HashMap<String, Pending>>> = LazyLock::new(Default::default);

fn key(tool_use_id: &str) -> String {
    format!("{}{tool_use_id}", super::TOOL_ID_PREFIX)
}

/// Whether prompts are routed to Warp. Off when the permission mode already allows everything
/// or `WARP_CLAUDE_INTERACTIVE_PERMISSIONS=off`.
pub(crate) fn interactive() -> bool {
    let mode = std::env::var("WARP_CLAUDE_PERMISSION_MODE").unwrap_or_default();
    let off = std::env::var("WARP_CLAUDE_INTERACTIVE_PERMISSIONS").is_ok_and(|v| v.eq_ignore_ascii_case("off"));
    !off && mode != "bypassPermissions"
}

pub(crate) fn register(tool_use_id: &str, request_id: String, input: Value, stdin: UnboundedSender<String>) {
    let k = key(tool_use_id);
    PENDING.lock().unwrap().insert(k.clone(), Pending { request_id, input, stdin });
    // Dev-only UI test hook: approve automatically after a few seconds.
    if let Ok(secs) = std::env::var("WARP_DEV_AUTOAPPROVE").map(|v| v.parse::<u64>().unwrap_or(8)) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            respond(&k, true);
        });
    }
}

/// True while the card with this id waits for the user's answer.
pub(crate) fn is_awaiting(call_id: &str) -> bool {
    call_id.starts_with(super::TOOL_ID_PREFIX) && PENDING.lock().unwrap().contains_key(call_id)
}

/// The `control_response` line for a decision.
pub(crate) fn response_line(request_id: &str, input: &Value, allow: bool) -> String {
    let body = if allow {
        json!({"behavior": "allow", "updatedInput": input})
    } else {
        json!({"behavior": "deny", "message": "Rejected in Warp"})
    };
    json!({"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": body}})
        .to_string()
}

/// Answers a parked request. Returns false if there was none for this card.
pub(crate) fn respond(call_id: &str, allow: bool) -> bool {
    let Some(p) = PENDING.lock().unwrap().remove(call_id) else { return false };
    p.stdin.unbounded_send(response_line(&p.request_id, &p.input, allow)).is_ok()
}

/// Drops requests whose turn is over (answers can no longer be delivered).
pub(crate) fn clear() {
    PENDING.lock().unwrap().clear();
}

#[cfg(test)]
#[path = "permissions_tests.rs"]
mod tests;
