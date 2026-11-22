//! Bidirectional translation registry for OpenAI tool calling in BYOP mode.
//!
//! Each built-in warp tool (a variant of `api::message::tool_call::Tool`) maps to one
//! [`OpenAiTool`] descriptor: function name + JSON Schema + reverse parsing of args +
//! serializing the execution result into a string for the upstream model.
//!
//! ## Currently implemented subset (Phase 3a, first batch)
//!
//! - `run_shell_command`
//! - `read_files`
//!
//! Later rounds extend this with `grep` / `file_glob_v2` / `apply_file_diffs` / `call_mcp_tool`, etc.
//!
//! ## Round trip
//!
//! The model returns `tool_calls` → `from_args` converts to `tool_call::Tool` → we emit
//! `Message::ToolCall { tool_call_id, tool }` → warp's own `convert_from.rs`
//! converts it to `AIAgentAction` → the executor applies profile permissions/prompts → executes → the result
//! is written back to the conversation → triggers the next byop request → our `result_to_json`
//! serializes the result as the content of `role=tool, tool_call_id=...` for the upstream model.

pub mod ask;
pub mod coerce;
pub mod documents;
pub mod edit;
pub mod exa;
pub mod files;
pub mod long_shell;
pub mod markers;
pub mod mcp;
pub mod search;
pub mod shell;
pub mod skill;
pub mod suggest;
pub mod todowrite;
pub mod web_runtime;
pub mod webfetch;
pub mod websearch;

use anyhow::Result;
use serde_json::Value;
use warp_multi_agent_api as api;

use crate::ai::agent::AIAgentActionResult;

/// Bidirectional adapter description for one tool.
///
/// **Naming history**: BYOP originally only supported OpenAI-compatible protocols, then moved to the genai SDK
/// across 5 adapters (OpenAI / OpenAIResp / Gemini / Anthropic / Ollama). The struct name `OpenAiTool` is kept
/// to preserve git blame, but the JSON Schema it carries is the OpenAPI standard; genai rewrites it internally
/// into each adapter's native format (e.g. Anthropic input_schema, Gemini function_declarations).
pub struct OpenAiTool {
    /// Function name for the upstream LLM (the model calls the tool by this name in its response).
    pub name: &'static str,
    /// Description for the LLM.
    pub description: &'static str,
    /// Parameter JSON Schema (OpenAPI standard). Returns a closure to avoid building a serde_json::Value in a const.
    pub parameters: fn() -> Value,
    /// Reverse parsing: the args JSON string returned by the upstream model → warp's internal `tool_call::Tool` variant.
    pub from_args: fn(args: &str) -> Result<api::message::tool_call::Tool>,
    /// Convert the `Result` variant for this tool in a ToolCallResult into JSON readable by the upstream model.
    /// Returns `None` when no variant matches (so the caller falls back to generic serialization).
    pub result_to_json: fn(&api::message::tool_call_result::Result) -> Option<Value>,
}

impl OpenAiTool {
    /// Convert to a genai `Tool` (to feed into `ChatRequest.tools`).
    pub fn to_genai_tool(&self) -> genai::chat::Tool {
        genai::chat::Tool::new(self.name)
            .with_description(self.description)
            .with_schema((self.parameters)())
    }
}

/// Registry: all supported BYOP tools.
pub const REGISTRY: &[&OpenAiTool] = &[
    &shell::RUN_SHELL_COMMAND,
    &files::READ_FILES,
    &search::GREP,
    &search::FILE_GLOB_V2,
    &edit::APPLY_FILE_DIFFS,
    &long_shell::WRITE_TO_LONG_RUNNING_SHELL_COMMAND,
    &long_shell::READ_SHELL_COMMAND_OUTPUT,
    &ask::ASK_USER_QUESTION,
    &skill::READ_SKILL,
    // Local document system (AIDocumentModel)
    &documents::READ_DOCUMENTS,
    &documents::EDIT_DOCUMENTS,
    &documents::CREATE_DOCUMENTS,
    // User suggestions (local channel + UI)
    &suggest::SUGGEST_NEW_CONVERSATION,
    &suggest::SUGGEST_PROMPT,
    // UI markers (no side effects, signal the frontend)
    &markers::OPEN_CODE_REVIEW,
    &markers::TRANSFER_SHELL_CONTROL,
    // Local todo list (BYOP synthesizes Message::UpdateTodos itself, bypassing the protobuf executor)
    &todowrite::TODOWRITE,
    // BYOP-only network tools: not mapped to a protobuf executor variant; chat_stream
    // intercepts them by name before parse_incoming_tool_call and calls web_runtime to run HTTP directly.
    // Gating: when profile.web_search_enabled=false, build_tools_array filters them out.
    &webfetch::WEBFETCH,
    &websearch::WEBSEARCH,
];

/// Look up the registry by OpenAI function name.
pub fn lookup(name: &str) -> Option<&'static OpenAiTool> {
    REGISTRY.iter().copied().find(|t| t.name == name)
}

/// Given a ToolCallResult, first find the matching tool in REGISTRY and serialize with its `result_to_json`;
/// if none is found, try generic MCP serialization; finally fall back to a short description to avoid a panic.
pub fn serialize_result(result: &api::message::ToolCallResult) -> String {
    let inner = match &result.result {
        Some(r) => r,
        None => return r#"{"status":"cancelled"}"#.to_owned(),
    };
    for t in REGISTRY {
        if let Some(json) = (t.result_to_json)(inner) {
            return serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_owned());
        }
    }
    if let Some(json) = mcp::serialize_result(inner) {
        return serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_owned());
    }
    // Fallback: unrecognized variants (tools not yet registered in later rounds also land here).
    r#"{"status":"unsupported_tool_result"}"#.to_owned()
}

/// Serialize an `AIAgentActionResult` that has *just finished executing on the client in the current round*
/// into a JSON string to feed the upstream model (the content of a role=tool message).
///
/// ## Why not use `AIAgentActionResultType::Display` directly
///
/// The `Display` impl renders structured results (especially `LongRunningCommandSnapshot`) into a
/// one-line string like `"Command 'bun repl' is long-running"`, **discarding key fields such as block_id
/// (= command_id), grid_contents and is_alt_screen_active**, so the model in the next round has no command_id
/// and cannot continue with read/write_to_long_running_*, which makes long-running commands useless.
///
/// ## How it works
///
/// 1. Reuse the existing `TryFrom<AIAgentActionResult>
///    for api::request::input::user_inputs::user_input::Input` in `app/src/ai/agent/api/convert_to.rs` (covering all 25+ ActionResult
///    variants) to get `Input::ToolCallResult { result, .. }`
/// 2. The inner `*Result` types (e.g. `RunShellCommandResult`) and `api::message::tool_call_result::Result`
///    share the same protobuf message and differ only in the outer enum's namespace, so the outer enum can be re-wrapped
///    to reuse the existing per-tool `result_to_json` in `tools::REGISTRY`
///    (see `shell.rs::result_to_json`, which flattens `LongRunningCommandSnapshot` into full JSON
///    including command_id/output/is_alt_screen_active)
/// 3. Unrecognized variants return `None`, and the caller falls back to Display
///
/// ## Maintenance note
///
/// When adding a BYOP tool, **the enum match here must get the new variant too**, otherwise that tool's
/// current-round ActionResult falls back to Display and loses its structured fields.
pub fn serialize_action_result(action: &AIAgentActionResult) -> Option<String> {
    let msg_side = action_result_to_msg_result(action)?;
    for t in REGISTRY {
        if let Some(json) = (t.result_to_json)(&msg_side) {
            return Some(serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_owned()));
        }
    }
    if let Some(json) = mcp::serialize_result(&msg_side) {
        return Some(serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_owned()));
    }
    None
}

/// Convert an `AIAgentActionResult` that finished executing on the client in the current round into
/// the `api::message::tool_call_result::Result` enum, for BYOP to persist as task.message.
///
/// Shares the ReqR → MsgR mapping with `serialize_action_result`; the caller wraps the result into
/// `Message::ToolCallResult { result: Some(...), context: None, tool_call_id }`。
pub fn action_result_to_msg_result(
    action: &AIAgentActionResult,
) -> Option<api::message::tool_call_result::Result> {
    use api::message::tool_call_result::Result as MsgR;
    use api::request::input::tool_call_result::Result as ReqR;
    use api::request::input::user_inputs::user_input::Input;

    let input: Input = action.clone().try_into().ok()?;
    let req_input: ReqR = match input {
        Input::ToolCallResult(tcr) => tcr.result?,
        _ => return None,
    };
    let msg_side = match req_input {
        ReqR::RunShellCommand(r) => MsgR::RunShellCommand(r),
        ReqR::WriteToLongRunningShellCommand(r) => MsgR::WriteToLongRunningShellCommand(r),
        ReqR::ReadShellCommandOutput(r) => MsgR::ReadShellCommandOutput(r),
        ReqR::ReadFiles(r) => MsgR::ReadFiles(r),
        ReqR::Grep(r) => MsgR::Grep(r),
        ReqR::FileGlobV2(r) => MsgR::FileGlobV2(r),
        ReqR::ApplyFileDiffs(r) => MsgR::ApplyFileDiffs(r),
        ReqR::CallMcpTool(r) => MsgR::CallMcpTool(r),
        ReqR::ReadMcpResource(r) => MsgR::ReadMcpResource(r),
        ReqR::AskUserQuestion(r) => MsgR::AskUserQuestion(r),
        ReqR::ReadSkill(r) => MsgR::ReadSkill(r),
        ReqR::ReadDocuments(r) => MsgR::ReadDocuments(r),
        ReqR::EditDocuments(r) => MsgR::EditDocuments(r),
        ReqR::CreateDocuments(r) => MsgR::CreateDocuments(r),
        ReqR::SuggestNewConversation(r) => MsgR::SuggestNewConversation(r),
        ReqR::SuggestPrompt(r) => MsgR::SuggestPrompt(r),
        ReqR::OpenCodeReview(r) => MsgR::OpenCodeReview(r),
        ReqR::TransferShellCommandControlToUser(r) => MsgR::TransferShellCommandControlToUser(r),
        _ => return None,
    };
    Some(msg_side)
}
