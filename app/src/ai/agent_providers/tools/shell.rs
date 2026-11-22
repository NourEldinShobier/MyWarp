//! `RunShellCommand` adapter.
//!
//! Corresponds to `api::message::tool_call::Tool::RunShellCommand` in warp;
//! after execution the result is `ToolCallResultType::RunShellCommand(RunShellCommandResult)`.

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use warp_multi_agent_api as api;

use super::OpenAiTool;

#[derive(Debug, Deserialize)]
struct Args {
    command: String,
    #[serde(default)]
    is_read_only: bool,
    #[serde(default)]
    uses_pager: bool,
    #[serde(default)]
    is_risky: bool,
    /// `None` (default / true) = return after the command finishes; `Some(false)` = return right after launch with
    /// a LongRunningCommandSnapshot, after which read/write_to_long_running_*
    /// tools can continue the interaction (suited to long-running commands such as dev servers / tail -f).
    #[serde(default)]
    wait_until_complete: Option<bool>,
}

fn parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run (the full command line)."
            },
            "is_read_only": {
                "type": "boolean",
                "description": "Whether the command only reads information and does not modify the filesystem or any external state (no user confirmation is needed when true).",
                "default": false
            },
            "uses_pager": {
                "type": "boolean",
                "description": "Whether the command triggers a pager (less/more, etc.). Prefer false, and append something like | cat to avoid blocking.",
                "default": false
            },
            "is_risky": {
                "type": "boolean",
                "description": "Whether the command is dangerous (rm -rf, changing global config, etc.). Set true to make the user confirm more prominently.",
                "default": false
            },
            "wait_until_complete": {
                "type": "boolean",
                "description": "Defaults to true (return only once the command finishes, suited to one-shot commands). Commands that never exit on their own — dev servers, background processes, tail -f, interactive REPLs — MUST set this to false, otherwise the current turn hangs forever waiting for a result. With false it returns a LongRunningCommandSnapshot immediately, and later turns keep interacting via read/write_to_long_running_shell_command.",
                "default": true
            }
        },
        "required": ["command"],
        "additionalProperties": false
    })
}

fn from_args(args: &str) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call::run_shell_command::WaitUntilCompleteValue;
    let parsed: Args = serde_json::from_str(args)?;
    // When None, explicitly default to true (return only after the command finishes), to avoid the controller's implicit default
    // being ambiguous across warp versions/paths. A model that wants long-running mode must pass false explicitly.
    let wait_until_complete_value = Some(WaitUntilCompleteValue::WaitUntilComplete(
        parsed.wait_until_complete.unwrap_or(true),
    ));
    Ok(api::message::tool_call::Tool::RunShellCommand(
        api::message::tool_call::RunShellCommand {
            command: parsed.command,
            is_read_only: parsed.is_read_only,
            uses_pager: parsed.uses_pager,
            is_risky: parsed.is_risky,
            citations: vec![],
            wait_until_complete_value,
            risk_category: 0,
        },
    ))
}

fn result_to_json(result: &api::message::tool_call_result::Result) -> Option<Value> {
    use api::message::tool_call_result::Result as R;
    use api::run_shell_command_result::Result as ShellR;
    let r = match result {
        R::RunShellCommand(r) => r,
        _ => return None,
    };
    let value = match &r.result {
        Some(ShellR::CommandFinished(f)) => json!({
            "status": "completed",
            "command": r.command,
            "exit_code": f.exit_code,
            "output": f.output,
        }),
        // Long-running command: started but not finished. Expose the snapshot to the model so it can
        // decide whether to keep reading (read_shell_command_output) or write (write_to_long_running_*).
        Some(ShellR::LongRunningCommandSnapshot(s)) => json!({
            "status": "running",
            "command": r.command,
            "command_id": s.command_id,
            "output": s.output,
            "is_alt_screen_active": s.is_alt_screen_active,
        }),
        Some(ShellR::PermissionDenied(_)) => json!({
            "status": "permission_denied",
            "command": r.command,
        }),
        None => json!({ "status": "cancelled", "command": r.command }),
    };
    Some(value)
}

pub static RUN_SHELL_COMMAND: OpenAiTool = OpenAiTool {
    name: "run_shell_command",
    description: include_str!("../prompts/tool_descriptions/run_shell_command.md"),
    parameters,
    from_args,
    result_to_json,
};
