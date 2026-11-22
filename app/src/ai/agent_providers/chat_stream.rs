//! BYOP-mode chat completion + tool calling adapter layer (based on genai 0.5.3).
//!
//! Translates `RequestParams` into a genai `ChatRequest` and calls the user-configured provider via `Client::exec_chat_stream`;
//! the response is translated back to `warp_multi_agent_api::ResponseEvent`,
//! and the controller's own logic (permissions/popups/execution/result write-back/next-turn trigger) takes over the loop.
//!
//! ## Explicit routing for 5 API protocols
//!
//! No longer treats every provider as OpenAI-compatible; `ServiceTargetResolver` maps
//! the `AgentProviderApiType` chosen in the settings UI one-to-one to genai's `AdapterKind`:
//!
//! | ApiType        | AdapterKind  | Default endpoint                               |
//! |----------------|--------------|------------------------------------------------|
//! | OpenAi         | OpenAI       | https://api.openai.com/v1                      |
//! | OpenAiResp     | OpenAIResp   | https://api.openai.com/v1 (via /v1/responses)  |
//! | Gemini         | Gemini       | https://generativelanguage.googleapis.com/v1beta |
//! | Anthropic      | Anthropic    | https://api.anthropic.com/v1 (via /v1/messages) |
//! | Ollama         | Ollama       | http://localhost:11434                         |
//!
//! The user-provided `base_url` always overrides the default. This way:
//! - OpenAI-compatible providers such as DeepSeek / SiliconFlow / OpenRouter pick `OpenAi` with a custom base_url
//! - Explicitly choosing an adapter fully bypasses genai's default "detect by model name" behavior, avoiding misdetection
//!
//! ## Multi-turn message conversion
//!
//! - system prompt: `ChatRequest::with_system()` (not in the messages array)
//! - user query: `ChatMessage::user(text)`
//! - assistant text: `ChatMessage::assistant(text)`
//! - assistant tool_calls: `ChatMessage::from(Vec<ToolCall>)` (assistant role is set automatically)
//! - tool result: `ChatMessage::from(ToolResponse::new(call_id, content))` (tool role is set automatically)
//!
//! ## Streaming implementation
//!
//! `Client::exec_chat_stream` returns a `ChatStreamResponse` whose `stream` field implements
//! `futures_core::Stream<Item = Result<ChatStreamEvent>>`. Events:
//! - `Start` / `Chunk(text)` / `ReasoningChunk(text)` / `ToolCallChunk(tool_call)` / `End(StreamEnd)`
//!
//! We emit `AppendToMessageContent` immediately for Chunk/ReasoningChunk (typewriter effect),
//! buffer ToolCallChunk by call_id and emit `Message::ToolCall` at the end of the stream,
//! after which the controller takes over automatically.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use instant::Instant;
use serde_json::{json, Value};
use uuid::Uuid;
use warp_multi_agent_api as api;

use genai::adapter::AdapterKind;
use genai::chat::{
    Binary, BinarySource, CacheControl, ChatMessage, ChatOptions, ChatRequest, ChatRole,
    ChatStreamEvent, ContentPart, MessageContent, Tool as GenaiTool, ToolCall, ToolResponse,
};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget, WebConfig};
use http_client::current_proxy_config;

use crate::ai::agent::api::{RequestParams, ResponseStream};
use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::{AIAgentActionResult, AIAgentInput, RunningCommand, UserQueryMode};
use crate::ai::api_error::AIApiError;
use crate::ai::byop_compaction;
use crate::ai::byop_readiness::{
    classify_projection, AcceptedRepair, BlockedByopReadinessError, LiveToolCall,
    LiveToolCallState, ProjectedToolCall, ProjectedToolResult, ProjectionItem, ReadinessCategory,
    ReadinessContext, ReadinessDiagnosticCoalescer, ReadinessDiagnosticContext,
    ReadinessDiagnosticLevel, ReadinessReport, ReadinessState, ReadinessTriggerLayer,
    RedactedToolKind, RepairSource, RepairStateStatus, TerminalResultKind, ToolCallKey,
    ToolCallRef, ToolResultSource,
};
use crate::settings::AgentProviderApiType;
use ai::agent::convert::ConvertToAPITypeError;

use super::header_vars::HeaderVars;
use super::openai_compatible::OpenAiCompatibleError;
use super::tools;

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------
// The system prompt is generated from a minijinja template by `prompt_renderer::render_system`,
// choosing system/{anthropic,gpt,beast,gemini,kimi,codex,trinity,default}.j2 by LLMId model family,
// and renders the AIAgentContext the warp client has already collected (env / git / skills / project_rules / codebase / current_time)
// into the system prompt, so the BYOP path gets environment info comparable to warp's own path.

use super::attachment_caps;
use super::prompt_renderer;
use super::user_context;
use crate::ai::agent::AIAgentContext;

/// Extracts the most recent `UserQuery.context` from the input (equivalent to the one warp's `convert_to.rs::convert_input` takes).
fn latest_input_context(input: &[AIAgentInput]) -> &[AIAgentContext] {
    for i in input.iter().rev() {
        if let Some(ctx) = i.context() {
            return ctx;
        }
    }
    &[]
}

/// In the LRC tag-in scenario, renders an `<attached_running_command>` XML block and prepends it to the user message,
/// so the model sees the actual PTY state (command, grid content, whether alt-screen is active) and
/// correctly chooses `write_to_long_running_shell_command` to send the matching key sequence.
fn render_running_command_context(rc: &RunningCommand) -> String {
    format!(
        "<attached_running_command command_id=\"{}\" is_alt_screen_active=\"{}\">\n  \
         <command>{}</command>\n  \
         <snapshot>\n{}\n  </snapshot>\n  \
         <instructions>This command is already running in the user's terminal. \
         Use `read_shell_command_output` with this command_id to inspect it, and \
         `write_to_long_running_shell_command` with this command_id to operate the program \
         through its PTY (in raw mode, use tokens like `<ESC>` and `<ENTER>` for control \
         keys). This command_id is valid even if the process was started by the user \
         rather than by run_shell_command. Do NOT spawn a new shell to control the same TUI.\
         </instructions>\n\
         </attached_running_command>",
        xml_attr(rc.block_id.as_str()),
        rc.is_alt_screen_active,
        xml_text(&rc.command),
        xml_text(&rc.grid_contents),
    )
}

/// Short fallback version: used when only command_id is available (no full RunningCommand snapshot),
/// so the model at least knows the target PTY id and can fetch the latest content itself with `read_shell_command_output`.
fn render_running_command_id_context(command_id: &str) -> String {
    format!(
        "<attached_running_command command_id=\"{}\">\n  \
         <instructions>This command is already running in the user's terminal. \
         Use `read_shell_command_output` with this command_id to inspect it, and \
         `write_to_long_running_shell_command` with this command_id to operate the program \
         through its PTY. Do NOT spawn a new shell to control the same TUI.</instructions>\n\
         </attached_running_command>",
        xml_attr(command_id),
    )
}

fn render_lrc_request_context(params: &RequestParams) -> Option<String> {
    params
        .lrc_running_command
        .as_ref()
        .map(render_running_command_context)
        .or_else(|| {
            params
                .lrc_command_id
                .as_deref()
                .map(render_running_command_id_context)
        })
}

/// Warp: renders the SSH session state block, appended to the end of the system prompt.
///
/// Trigger: `SessionContext.is_legacy_ssh()` is true (the user manually typed
/// `ssh xx@xx` in a local PTY to reach a remote host that has no warp shell hook installed). For such a session:
/// - `session_type` is still `Local`
/// - the [Environment] section of the whole system prompt describes the **local client** OS / shell,
///   while the PTY is actually running on the **remote host**
///
/// Without telling the model this, the LLM infers from the local OS in the system prompt that
/// "the target is remote, I must ssh over first", and emits nested commands like `ssh xx@xx uname -a`.
///
/// Note: warpified SSH (`SessionType::WarpifiedRemote`) is not handled here; on that path
/// the remote shell hook has re-bootstrapped, host_info / shell are the remote's true values, and the prompt is already correct.
fn render_ssh_session_block(
    session_context: &crate::ai::blocklist::SessionContext,
) -> Option<String> {
    if !session_context.is_legacy_ssh() {
        return None;
    }
    let info = session_context.ssh_connection_info();
    let host = info
        .and_then(|i| i.host.as_deref())
        .map(xml_attr)
        .unwrap_or_else(|| "unknown".to_owned());
    let port = info
        .and_then(|i| i.port.as_deref())
        .map(xml_attr)
        .unwrap_or_else(|| "22".to_owned());

    Some(format!(
        "\n\n<ssh_session host=\"{host}\" port=\"{port}\">\n  \
         <fact>The active terminal PTY is currently inside an SSH session opened by the user from their local machine. \
         All shell commands you run via `run_shell_command` execute on the REMOTE host, not on the local client.</fact>\n  \
         <warning>The [Environment] block (OS / shell / working directory) above describes the LOCAL client and may not match the remote host. \
         If you need precise remote info, probe it directly (e.g. `uname -a`, `cat /etc/os-release`, `pwd`).</warning>\n  \
         <rules>\n    \
         - Run commands DIRECTLY (e.g. `uname -a`, `ls /`). Do NOT prepend `ssh {host} ...` — that opens a NESTED ssh session inside the current one.\n    \
         - Treat the working directory and home directory shown above with skepticism; they may reflect the local client.\n    \
         - When LRC tag-in mode is active (an `<attached_running_command>` block is present), prefer `write_to_long_running_shell_command` with that command_id to inject keystrokes into this same remote PTY. Spawning a new shell would create a separate local-side ssh client, not interact with the remote process the user is watching.\n  \
         </rules>\n\
         </ssh_session>"
    ))
}

/// XML-escapes and strips all illegal/problematic control characters to avoid JSON serialization failures.
///
/// `grid_contents` (alt-screen content extracted from `formatted_terminal_contents_for_input`)
/// may contain raw ANSI escape sequences (0x1b), CSI sequences, SGR codes, box-drawing chars, etc.
/// Control chars `< 0x20` among them make Anthropic's JSON parser report "invalid escaped character in string",
/// because JSON RFC 7159 only accepts the legal escapes `\b \f \n \r \t \" \\ \/ \uXXXX`,
/// and others such as `\v` `\a` `\x1b` are rejected outright.
///
/// Handling:
/// - `\n` `\r` `\t` are kept (legal JSON)
/// - other `< 0x20` control chars are replaced with a space (shown to the model only; ANSI colors etc. need not be kept)
/// - `&` `<` `>` are converted to XML entities
fn xml_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' | '\r' | '\t' => out.push(c),
            c if (c as u32) < 0x20 => out.push(' '),
            // DEL(0x7f) is handled separately; it is also a control character
            '\u{7f}' => out.push(' '),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn xml_attr(s: &str) -> String {
    xml_text(s).replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Multi-turn message conversion
// ---------------------------------------------------------------------------

/// Accumulates the text + tool_calls + reasoning of the same assistant turn, then flushes them as one or two
/// `ChatMessage`s (one for text, one for tool_calls; genai models them as separate messages).
///
/// **thinking-mode reasoning_content echo gate** (bidirectional).
///
/// `force_echo_reasoning` controls two things at once, with unified semantics: "this model's capability declares
/// `interleaved.field = reasoning_content`" (aligned with opencode `model.capabilities.interleaved`):
///
/// - `true` (model_id matches the `reasoning::INTERLEAVED_RULES` substring table: DeepSeek / Kimi /
///   MiMo / Qwen3 / GLM thinking / MiniMax / Hunyuan / Ernie / Doubao / Yi, etc.;
///   or api_type is the DeepSeek whole-adapter): every assistant message must carry `reasoning_content`
///   (use the real reasoning if any, otherwise a non-empty placeholder), satisfying the thinking-mode server's
///   "field must exist" validation.
/// - `false` (everything else): **even if the stream received real reasoning_content, it is dropped on replay**,
///   and `with_reasoning_content` is not attached to history assistant messages.
///
/// Why real reasoning is actively dropped when `false`: many OpenAI-strict providers treat
/// `messages[].reasoning_content` as an illegal field and return 400 (`code: wrong_api_format`):
///
/// - **Cerebras** (culprit of zerx-lab/warp #25, 400 on the second turn of zai-glm-4.7)
/// - **Groq** (its protocol uses `reasoning_format` / `include_reasoning`, and does not accept a message field)
/// - **OpenRouter / Together AI / SambaNova / Anyscale / Replicate** and other relay/inference vendors
/// - **Official OpenAI** (GPT-5/o-series use OpenAIResp; o-series use `reasoning.encrypted_content` and do not accept `reasoning_content`)
///
/// genai 0.6 OpenAI adapter `adapter_shared.rs:367,385-387`, on seeing
/// `ContentPart::ReasoningContent`, **unconditionally** hoists it into the top-level `reasoning_content`
/// field, so the gate must move to the client side: "not attaching a ContentPart means it is never serialized".
///
/// The Anthropic / Gemini adapter serialization layers ignore `ContentPart::ReasoningContent`
/// (they use thinking blocks / thought signature respectively), so they are unaffected by this gate, but for consistency they still take the `false` branch and attach nothing.
const REASONING_ECHO_PLACEHOLDER: &str = " ";

#[derive(Default)]
struct AssistantBuffer {
    text: Option<String>,
    tool_calls: Vec<ToolCall>,
    tool_call_keys: Vec<ToolCallKey>,
    /// Previous turn's AgentReasoning (thinking chain). On flush it is attached to the matching assistant message's
    /// reasoning_content field (genai serializes per adapter internally: DeepSeek/Kimi use reasoning_content,
    /// Anthropic uses thinking blocks).
    reasoning: Option<String>,
    /// Thinking-mode adapters force echoing reasoning_content (non-empty placeholder). Decided by
    /// `super::reasoning::model_requires_reasoning_echo`.
    force_echo_reasoning: bool,
}

impl AssistantBuffer {
    fn new(force_echo_reasoning: bool) -> Self {
        Self {
            force_echo_reasoning,
            ..Default::default()
        }
    }

    fn push_tool_call(&mut self, tool_call: ToolCall, key: ToolCallKey) {
        self.tool_calls.push(tool_call);
        self.tool_call_keys.push(key);
    }

    fn flush_into(&mut self, messages: &mut Vec<ChatMessage>) {
        let _ = self.flush_into_with_group(messages);
    }

    fn flush_into_with_group(
        &mut self,
        messages: &mut Vec<ChatMessage>,
    ) -> Option<OutboundAssistantToolGroup> {
        let reasoning = self.reasoning.take();
        let has_tool_calls = !self.tool_calls.is_empty();
        // Decides the reasoning string to attach to the assistant message in this flush.
        //
        // **Gate inversion**: when `force_echo_reasoning = false`, **never attach**, even if this turn's
        // stream received real reasoning (thinking models such as zai-glm / qwen3-thinking emit reasoning_content chunks
        // on the OpenAI-compatible path), because Cerebras / Groq /
        // OpenRouter and other OpenAI-strict providers, on seeing `messages[].reasoning_content`, directly
        // 400 `wrong_api_format`(zerx-lab/warp #25)。
        //
        // When `force_echo_reasoning = true` (DeepSeek api_type / OpenAI+kimi/moonshot):
        // - real reasoning present: use it
        // - none: non-empty placeholder (satisfies the "field must exist" validation)
        let echo_reasoning: Option<String> = if self.force_echo_reasoning {
            match reasoning {
                Some(r) if !r.is_empty() => Some(r),
                _ => Some(REASONING_ECHO_PLACEHOLDER.to_owned()),
            }
        } else {
            // Note: even if `reasoning` is Some(non-empty), it is dropped; see the gate inversion note above.
            None
        };
        if let Some(t) = self.text.take() {
            let mut msg = ChatMessage::assistant(t);
            if has_tool_calls {
                // DeepSeek thinking mode requires every assistant message to carry
                // reasoning_content. When genai models text + tool_calls as two
                // assistant messages, the text one must also get the placeholder.
                if self.force_echo_reasoning {
                    msg = msg.with_reasoning_content(Some(REASONING_ECHO_PLACEHOLDER.to_owned()));
                }
            } else if let Some(r) = echo_reasoning.clone() {
                msg = msg.with_reasoning_content(Some(r));
            }
            messages.push(msg);
        }
        if has_tool_calls {
            // genai `From<Vec<ToolCall>> for ChatMessage` automatically produces the assistant role +
            // MessageContent::from_tool_calls。
            let mut msg = ChatMessage::from(std::mem::take(&mut self.tool_calls));
            if let Some(r) = echo_reasoning {
                msg = msg.with_reasoning_content(Some(r));
            }
            let message_index = messages.len();
            messages.push(msg);
            Some(OutboundAssistantToolGroup {
                message_index,
                tool_call_keys: std::mem::take(&mut self.tool_call_keys),
            })
        } else {
            self.tool_call_keys.clear();
            None
        }
    }
}

#[derive(Debug, Clone)]
struct OutboundAssistantToolGroup {
    message_index: usize,
    tool_call_keys: Vec<ToolCallKey>,
}

fn flush_assistant_buffer(
    buf: &mut AssistantBuffer,
    messages: &mut Vec<ChatMessage>,
    outbound_tool_groups: &mut Vec<OutboundAssistantToolGroup>,
) {
    if let Some(group) = buf.flush_into_with_group(messages) {
        if !group.tool_call_keys.is_empty() {
            outbound_tool_groups.push(group);
        }
    }
}

/// Builds a user `ChatMessage`, deciding by model capability whether to switch to
/// the `MessageContent::Parts(Text + Binary[])` multimodal form.
///
/// - no binaries: old path `ChatMessage::user(text)` plain text, same as P0 behavior
/// - binaries and the model supports the mime: `Parts(vec![Text(text), Binary(...), ...])`,
///   the genai adapter adapts to the wire protocol automatically (OpenAI image_url/file, Anthropic image/document,
///   Gemini inline_data, etc.)
/// - binaries but the model does not support them: log warn, skip that part and degrade to plain text (the
///   `<image .../>` / `<file binary=true .../>` placeholders in the prefix XML remain, so the LLM at least knows what the user attached)
fn build_user_message_with_binaries(
    text: String,
    binaries: Vec<user_context::UserBinary>,
    caps: attachment_caps::AttachmentCaps,
) -> ChatMessage {
    if binaries.is_empty() {
        return ChatMessage::user(text);
    }

    let mut parts: Vec<ContentPart> = Vec::with_capacity(1 + binaries.len());
    parts.push(ContentPart::Text(text));

    let mut error_replacements: Vec<(String, String)> = Vec::new();
    for bin in binaries {
        if !caps.supports_mime(&bin.content_type) {
            // Warp aligns with opencode `unsupportedParts` (packages/opencode/src/provider/transform.ts:305-341):
            // mimes the model does not support are not silently dropped; instead an ERROR text part is inserted so the LLM can tell the user itself.
            // The wording is copied strictly from opencode's `ERROR: Cannot read {name} (this model does not support
            // {modality} input). Inform the user.`; modality is mapped from the mime prefix, and name prefers the file name.
            let modality = mime_to_modality(&bin.content_type);
            let name = if bin.name.is_empty() {
                modality.to_string()
            } else {
                format!("\"{}\"", bin.name)
            };
            let err_text = format!(
                "ERROR: Cannot read {name} (this model does not support {modality} input). Inform the user."
            );
            error_replacements.push((bin.name.clone(), bin.content_type.clone()));
            parts.push(ContentPart::Text(err_text));
            continue;
        }
        parts.push(ContentPart::Binary(Binary::from_base64(
            bin.content_type,
            bin.data,
            Some(bin.name),
        )));
    }

    if !error_replacements.is_empty() {
        log::info!(
            "[byop] {} attachment(s) replaced with ERROR text — caps={caps:?} does not support: {error_replacements:?}",
            error_replacements.len()
        );
    }

    // If the binaries are all replaced ERROR texts (no real Binary part), the ERROR text parts are still kept
    // so the model sees them. Degenerate case (e.g. empty text + no part added) falls back to plain text.
    if parts.len() == 1 {
        if let Some(ContentPart::Text(t)) = parts.into_iter().next() {
            return ChatMessage::user(t);
        }
        return ChatMessage::user("");
    }

    ChatMessage {
        role: ChatRole::User,
        content: MessageContent::from_parts(parts),
        options: None,
    }
}

/// MIME → modality string mapping. Aligned with opencode `mimeToModality`
/// (packages/opencode/src/provider/transform.ts:12-18)。
fn mime_to_modality(mime: &str) -> &'static str {
    let lower = mime.trim().to_ascii_lowercase();
    if lower.starts_with("image/") {
        "image"
    } else if lower.starts_with("audio/") {
        "audio"
    } else if lower.starts_with("video/") {
        "video"
    } else if lower == "application/pdf" {
        "pdf"
    } else {
        "file"
    }
}

/// Collects `params.tasks` into a stable linear message sequence, fixing Issue #94.
///
/// `params.tasks` comes from [`crate::ai::agent::conversation::AIConversation::compute_active_tasks`],
/// which collects via `HashMap::into_values()`, so the task order is nondeterministic. The old implementation simply did
/// `flat_map(|t| t.messages.iter())` concatenation, which has two defects in multi-task scenarios (LRC CLI subagent /
/// subtasks derived from the `subagent` tool):
///   1. The relative order between tasks is random, so a history-turn user message may land at the end of the sequence and be
///      treated by the upstream model as the "latest user message" (the direct symptom of Issue #94).
///   2. When an LRC subagent spawns, the current turn's UserQuery is copied into **both** the root task and the
///      new subtask (see the subtask UserQuery injection inside the `generate_byop_output` stream;
///      the copy is required by `CLISubagentView` rendering and cannot be removed at the source), so naive concatenation makes the same
///      user query appear twice in the request.
///
/// Fix: do a DFS from the root task (no parent, or parent not in the tasks set), descending into the corresponding subtask on each
/// `Subagent` ToolCall (the same deterministic traversal as [`crate::ai::agent::task_store::TaskStore::all_linearized_messages`]);
/// and dedupe UserQuery by `(request_id, query)`, dropping the subtask copy made by LRC. Different user turns have different
/// `request_id`s and are not removed by mistake; for old data / test stubs with an empty `request_id`, no dedupe is done,
/// to avoid collateral damage. Orphan tasks not reached by the DFS are appended at the end sorted by id as a fallback,
/// ensuring no messages are lost.
fn collect_linearized_task_messages(tasks: &[api::Task]) -> Vec<&api::Message> {
    use std::collections::{HashMap, HashSet};

    if tasks.is_empty() {
        return Vec::new();
    }

    let by_id: HashMap<&str, &api::Task> = tasks.iter().map(|t| (t.id.as_str(), t)).collect();

    // root = a task with no parent, or whose parent is not in the current tasks set.
    let root = tasks.iter().find(|t| match t.dependencies.as_ref() {
        None => true,
        Some(dep) => {
            dep.parent_task_id.is_empty() || !by_id.contains_key(dep.parent_task_id.as_str())
        }
    });

    fn push_msg<'a>(
        msg: &'a api::Message,
        out: &mut Vec<&'a api::Message>,
        seen_user_queries: &mut HashSet<(&'a str, &'a str)>,
    ) {
        if let Some(api::message::Message::UserQuery(u)) = &msg.message {
            if !msg.request_id.is_empty()
                && !seen_user_queries.insert((msg.request_id.as_str(), u.query.as_str()))
            {
                return;
            }
        }
        out.push(msg);
    }

    fn dfs<'a>(
        task: &'a api::Task,
        by_id: &HashMap<&'a str, &'a api::Task>,
        visited_tasks: &mut HashSet<&'a str>,
        out: &mut Vec<&'a api::Message>,
        seen_user_queries: &mut HashSet<(&'a str, &'a str)>,
    ) {
        if !visited_tasks.insert(task.id.as_str()) {
            return;
        }
        for msg in &task.messages {
            push_msg(msg, out, seen_user_queries);
            if let Some(api::message::Message::ToolCall(tc)) = &msg.message {
                if let Some(api::message::tool_call::Tool::Subagent(sub)) = &tc.tool {
                    if let Some(subtask) = by_id.get(sub.task_id.as_str()) {
                        dfs(subtask, by_id, visited_tasks, out, seen_user_queries);
                    }
                }
            }
        }
    }

    let mut out: Vec<&api::Message> = Vec::new();
    let mut visited_tasks: HashSet<&str> = HashSet::new();
    let mut seen_user_queries: HashSet<(&str, &str)> = HashSet::new();

    if let Some(root) = root {
        dfs(
            root,
            &by_id,
            &mut visited_tasks,
            &mut out,
            &mut seen_user_queries,
        );
    }

    // Orphan task fallback: sort by id to ensure determinism.
    let mut orphans: Vec<&api::Task> = tasks
        .iter()
        .filter(|t| !visited_tasks.contains(t.id.as_str()))
        .collect();
    orphans.sort_by(|a, b| a.id.cmp(&b.id));
    for task in orphans {
        if !visited_tasks.insert(task.id.as_str()) {
            continue;
        }
        for msg in &task.messages {
            push_msg(msg, &mut out, &mut seen_user_queries);
        }
    }

    out
}

struct SerializerProjectionBuilder {
    items: Vec<ProjectionItem>,
    pending_tool_calls: Vec<ProjectedToolCall>,
    pending_task_id: Option<String>,
    pending_assistant_message_id: Option<String>,
    skipped_tool_results: HashSet<(String, String)>,
}

impl SerializerProjectionBuilder {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            pending_tool_calls: Vec::new(),
            pending_task_id: None,
            pending_assistant_message_id: None,
            skipped_tool_results: HashSet::new(),
        }
    }

    fn push_user_boundary(&mut self, task_id: String, message_id: String) {
        self.flush_tool_calls();
        self.items
            .push(ProjectionItem::user_boundary(task_id, message_id));
    }

    fn push_assistant_boundary(&mut self, task_id: String, message_id: String) {
        self.flush_tool_calls();
        self.items
            .push(ProjectionItem::assistant_boundary(task_id, message_id));
    }

    fn push_tool_call(
        &mut self,
        task_id: &str,
        message_id: &str,
        tool_call: &api::message::ToolCall,
    ) {
        use crate::ai::agent::task::helper::ToolCallExt;

        if tool_call.subagent().is_some() {
            self.skipped_tool_results
                .insert((task_id.to_owned(), tool_call.tool_call_id.clone()));
            return;
        }

        if self
            .pending_task_id
            .as_deref()
            .is_some_and(|pending_task_id| pending_task_id != task_id)
        {
            self.flush_tool_calls();
        }

        if self.pending_tool_calls.is_empty() {
            self.pending_task_id = Some(task_id.to_owned());
            self.pending_assistant_message_id = Some(message_id.to_owned());
        }

        let assistant_message_id = self
            .pending_assistant_message_id
            .clone()
            .unwrap_or_else(|| message_id.to_owned());
        self.pending_tool_calls.push(ProjectedToolCall::new(
            task_id,
            assistant_message_id,
            tool_call.tool_call_id.clone(),
            redacted_tool_kind_for_tool_call(tool_call),
        ));
    }

    fn push_tool_result(&mut self, result: ProjectedToolResult) {
        self.flush_tool_calls();
        self.items.push(ProjectionItem::tool_result(result));
    }

    fn should_skip_tool_result(&self, task_id: &str, tool_call_id: &str) -> bool {
        self.skipped_tool_results
            .contains(&(task_id.to_owned(), tool_call_id.to_owned()))
    }

    fn finish(mut self) -> Vec<ProjectionItem> {
        self.flush_tool_calls();
        self.items
    }

    fn flush_tool_calls(&mut self) {
        if self.pending_tool_calls.is_empty() {
            return;
        }

        let task_id = self.pending_task_id.take().unwrap_or_default();
        let assistant_message_id = self.pending_assistant_message_id.take().unwrap_or_default();
        self.items.push(ProjectionItem::assistant_tool_calls(
            task_id,
            assistant_message_id,
            std::mem::take(&mut self.pending_tool_calls),
        ));
    }
}

fn redacted_tool_kind_for_tool_call(tool_call: &api::message::ToolCall) -> RedactedToolKind {
    use crate::ai::agent::task::helper::ToolExt;

    tool_call
        .tool
        .as_ref()
        .map(|tool| RedactedToolKind::new(tool.name()))
        .unwrap_or_default()
}

fn current_input_result_kind(result: &AIAgentActionResult) -> TerminalResultKind {
    if result.result.is_cancelled() {
        TerminalResultKind::Cancellation
    } else {
        TerminalResultKind::Real
    }
}

fn persisted_tool_result_kind(
    msg: &api::Message,
    compacted_tool_msg_ids: Option<&std::collections::HashSet<String>>,
) -> TerminalResultKind {
    if compacted_tool_msg_ids.is_some_and(|ids| ids.contains(&msg.id)) {
        return TerminalResultKind::Compacted;
    }

    let Some(api::message::Message::ToolCallResult(tool_call_result)) = msg.message.as_ref() else {
        return TerminalResultKind::Real;
    };
    if tool_call_result.result.is_some() {
        return TerminalResultKind::Real;
    }

    let content = msg.server_message_data.trim();
    if content.is_empty() {
        return TerminalResultKind::Real;
    }

    match serde_json::from_str::<Value>(content) {
        Ok(value) => {
            if value
                .get("_byop_intercepted")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                TerminalResultKind::LocalInterception
            } else if value.get("error").and_then(Value::as_str) == Some("invalid_arguments") {
                TerminalResultKind::StructuredError
            } else {
                TerminalResultKind::Real
            }
        }
        Err(_)
            if content.contains("_byop_intercepted") || content.contains("invalid_arguments") =>
        {
            TerminalResultKind::UnreadableLocalInterception
        }
        Err(_) => TerminalResultKind::Real,
    }
}

fn current_input_task_id(params: &RequestParams) -> String {
    params
        .byop_target_task_id
        .clone()
        .or_else(|| params.tasks.first().map(|task| task.id.clone()))
        .unwrap_or_else(|| "current_input".to_owned())
}

fn build_serializer_readiness_projection(
    params: &RequestParams,
    all_msgs: &[&api::Message],
    summarize_head_end: Option<usize>,
    summary_inserts: &HashMap<String, String>,
    hidden_msg_ids: &std::collections::HashSet<String>,
    compacted_tool_msg_ids: &std::collections::HashSet<String>,
) -> Vec<ProjectionItem> {
    let mut builder = SerializerProjectionBuilder::new();

    for (idx, msg) in all_msgs.iter().enumerate() {
        if let Some(head_end) = summarize_head_end {
            if idx >= head_end {
                continue;
            }
        }

        if hidden_msg_ids.contains(&msg.id) {
            if summary_inserts.contains_key(&msg.id) {
                builder.push_user_boundary(msg.task_id.clone(), format!("summary_user:{}", msg.id));
                builder.push_assistant_boundary(
                    msg.task_id.clone(),
                    format!("summary_assistant:{}", msg.id),
                );
            }
            continue;
        }

        let Some(inner) = &msg.message else {
            continue;
        };

        match inner {
            api::message::Message::UserQuery(_) => {
                builder.push_user_boundary(msg.task_id.clone(), msg.id.clone());
            }
            api::message::Message::AgentOutput(_) => {
                builder.push_assistant_boundary(msg.task_id.clone(), msg.id.clone());
            }
            api::message::Message::AgentReasoning(_) => {}
            api::message::Message::ToolCall(tool_call) => {
                builder.push_tool_call(&msg.task_id, &msg.id, tool_call);
            }
            api::message::Message::ToolCallResult(tool_call_result) => {
                if builder.should_skip_tool_result(&msg.task_id, &tool_call_result.tool_call_id) {
                    continue;
                }
                builder.push_tool_result(ProjectedToolResult::new(
                    msg.task_id.clone(),
                    msg.id.clone(),
                    None,
                    tool_call_result.tool_call_id.clone(),
                    RedactedToolKind::default(),
                    ToolResultSource::PersistedHistory,
                    persisted_tool_result_kind(msg, Some(compacted_tool_msg_ids)),
                ));
            }
            api::message::Message::ServerEvent(_)
            | api::message::Message::SystemQuery(_)
            | api::message::Message::UpdateTodos(_)
            | api::message::Message::Summarization(_)
            | api::message::Message::CodeReview(_)
            | api::message::Message::UpdateReviewComments(_)
            | api::message::Message::WebSearch(_)
            | api::message::Message::WebFetch(_)
            | api::message::Message::DebugOutput(_)
            | api::message::Message::ArtifactEvent(_)
            | api::message::Message::InvokeSkill(_)
            | api::message::Message::MessagesReceivedFromAgents(_)
            | api::message::Message::ModelUsed(_)
            | api::message::Message::EventsFromAgents(_)
            | api::message::Message::PassiveSuggestionResult(_) => {}
        }
    }

    let current_task_id = current_input_task_id(params);
    for (idx, input) in params.input.iter().enumerate() {
        match input {
            AIAgentInput::UserQuery { .. }
            | AIAgentInput::InvokeSkill { .. }
            | AIAgentInput::ResumeConversation { .. }
            | AIAgentInput::SummarizeConversation { .. } => {
                builder.push_user_boundary(
                    current_task_id.clone(),
                    format!("current_input:{idx}:user"),
                );
            }
            AIAgentInput::ActionResult { result, .. } => {
                let tool_call_id = result.id.to_string();
                builder.push_tool_result(ProjectedToolResult::new(
                    result.task_id.to_string(),
                    format!("current_input:{idx}:{tool_call_id}"),
                    None,
                    tool_call_id,
                    RedactedToolKind::default(),
                    ToolResultSource::CurrentInput,
                    current_input_result_kind(result),
                ));
            }
            AIAgentInput::AutoCodeDiffQuery { .. }
            | AIAgentInput::InitProjectRules { .. }
            | AIAgentInput::TriggerPassiveSuggestion { .. }
            | AIAgentInput::CreateNewProject { .. }
            | AIAgentInput::CloneRepository { .. }
            | AIAgentInput::CodeReview { .. }
            | AIAgentInput::FetchReviewComments { .. }
            | AIAgentInput::StartFromAmbientRunPrompt { .. }
            | AIAgentInput::MessagesReceivedFromAgents { .. }
            | AIAgentInput::EventsFromAgents { .. }
            | AIAgentInput::PassiveSuggestionResult { .. } => {}
        }
    }

    builder.finish()
}

pub(crate) fn classify_byop_controller_readiness(params: &RequestParams) -> ReadinessReport {
    classify_byop_controller_readiness_with_live_tool_calls(params, Vec::new())
}

pub(crate) fn classify_byop_controller_readiness_with_live_tool_calls(
    params: &RequestParams,
    live_tool_calls: Vec<LiveToolCall>,
) -> ReadinessReport {
    let skipped_cancellation_results = current_input_cancellation_result_keys(params);
    let projection = build_controller_readiness_projection(params, &skipped_cancellation_results);
    let mut live_tool_calls_for_context =
        cancellation_live_tool_calls(params, &skipped_cancellation_results);
    live_tool_calls_for_context.extend(live_tool_calls);
    let context = ReadinessContext {
        repair_records: params.byop_repair_state.repair_records().to_vec(),
        live_tool_calls: live_tool_calls_for_context,
    };
    classify_projection(&projection, &context)
}

fn current_input_cancellation_result_keys(params: &RequestParams) -> HashSet<(String, String)> {
    params
        .input
        .iter()
        .filter_map(|input| {
            let AIAgentInput::ActionResult { result, .. } = input else {
                return None;
            };
            result
                .result
                .is_cancelled()
                .then(|| (result.task_id.to_string(), result.id.to_string()))
        })
        .collect()
}

fn cancellation_live_tool_calls(
    params: &RequestParams,
    skipped_cancellation_results: &HashSet<(String, String)>,
) -> Vec<LiveToolCall> {
    use crate::ai::agent::task::helper::ToolCallExt;

    params
        .tasks
        .iter()
        .flat_map(|task| task.messages.iter())
        .filter_map(|msg| {
            let api::message::Message::ToolCall(tool_call) = msg.message.as_ref()? else {
                return None;
            };
            if tool_call.subagent().is_some() {
                return None;
            }
            if !skipped_cancellation_results
                .contains(&(msg.task_id.clone(), tool_call.tool_call_id.clone()))
            {
                return None;
            }
            Some(LiveToolCall::new(
                ToolCallRef::new(
                    ToolCallKey::new(&msg.task_id, &msg.id, &tool_call.tool_call_id),
                    redacted_tool_kind_for_tool_call(tool_call),
                ),
                LiveToolCallState::CancellationRequested,
            ))
        })
        .collect()
}

fn build_controller_readiness_projection(
    params: &RequestParams,
    skipped_current_action_results: &HashSet<(String, String)>,
) -> Vec<ProjectionItem> {
    let mut builder = SerializerProjectionBuilder::new();

    for msg in params.tasks.iter().flat_map(|task| task.messages.iter()) {
        let Some(inner) = &msg.message else {
            continue;
        };

        match inner {
            api::message::Message::UserQuery(_) => {
                builder.push_user_boundary(msg.task_id.clone(), msg.id.clone());
            }
            api::message::Message::AgentOutput(_) => {
                builder.push_assistant_boundary(msg.task_id.clone(), msg.id.clone());
            }
            api::message::Message::AgentReasoning(_) => {}
            api::message::Message::ToolCall(tool_call) => {
                builder.push_tool_call(&msg.task_id, &msg.id, tool_call);
            }
            api::message::Message::ToolCallResult(tool_call_result) => {
                if builder.should_skip_tool_result(&msg.task_id, &tool_call_result.tool_call_id) {
                    continue;
                }
                builder.push_tool_result(ProjectedToolResult::new(
                    msg.task_id.clone(),
                    msg.id.clone(),
                    None,
                    tool_call_result.tool_call_id.clone(),
                    RedactedToolKind::default(),
                    ToolResultSource::PersistedHistory,
                    persisted_tool_result_kind(msg, None),
                ));
            }
            api::message::Message::ServerEvent(_)
            | api::message::Message::SystemQuery(_)
            | api::message::Message::UpdateTodos(_)
            | api::message::Message::Summarization(_)
            | api::message::Message::CodeReview(_)
            | api::message::Message::UpdateReviewComments(_)
            | api::message::Message::WebSearch(_)
            | api::message::Message::WebFetch(_)
            | api::message::Message::DebugOutput(_)
            | api::message::Message::ArtifactEvent(_)
            | api::message::Message::InvokeSkill(_)
            | api::message::Message::MessagesReceivedFromAgents(_)
            | api::message::Message::ModelUsed(_)
            | api::message::Message::EventsFromAgents(_)
            | api::message::Message::PassiveSuggestionResult(_) => {}
        }
    }

    let current_task_id = current_input_task_id(params);
    for (idx, input) in params.input.iter().enumerate() {
        match input {
            AIAgentInput::UserQuery { .. }
            | AIAgentInput::InvokeSkill { .. }
            | AIAgentInput::ResumeConversation { .. }
            | AIAgentInput::SummarizeConversation { .. } => {
                builder.push_user_boundary(
                    current_task_id.clone(),
                    format!("current_input:{idx}:user"),
                );
            }
            AIAgentInput::ActionResult { result, .. } => {
                let tool_call_id = result.id.to_string();
                if skipped_current_action_results
                    .contains(&(result.task_id.to_string(), tool_call_id.clone()))
                {
                    continue;
                }
                builder.push_tool_result(ProjectedToolResult::new(
                    result.task_id.to_string(),
                    format!("current_input:{idx}:{tool_call_id}"),
                    None,
                    tool_call_id,
                    RedactedToolKind::default(),
                    ToolResultSource::CurrentInput,
                    current_input_result_kind(result),
                ));
            }
            AIAgentInput::AutoCodeDiffQuery { .. }
            | AIAgentInput::InitProjectRules { .. }
            | AIAgentInput::TriggerPassiveSuggestion { .. }
            | AIAgentInput::CreateNewProject { .. }
            | AIAgentInput::CloneRepository { .. }
            | AIAgentInput::CodeReview { .. }
            | AIAgentInput::FetchReviewComments { .. }
            | AIAgentInput::StartFromAmbientRunPrompt { .. }
            | AIAgentInput::MessagesReceivedFromAgents { .. }
            | AIAgentInput::EventsFromAgents { .. }
            | AIAgentInput::PassiveSuggestionResult { .. } => {}
        }
    }

    builder.finish()
}

fn validate_byop_serializer_readiness(
    params: &RequestParams,
    all_msgs: &[&api::Message],
    summarize_head_end: Option<usize>,
    summary_inserts: &HashMap<String, String>,
    hidden_msg_ids: &std::collections::HashSet<String>,
    compacted_tool_msg_ids: &std::collections::HashSet<String>,
) -> Result<ReadinessReport, ConvertToAPITypeError> {
    let projection = build_serializer_readiness_projection(
        params,
        all_msgs,
        summarize_head_end,
        summary_inserts,
        hidden_msg_ids,
        compacted_tool_msg_ids,
    );
    let conversation_id = params
        .byop_conversation_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unknown".to_owned());
    let request_attempt_id = params
        .byop_readiness_attempt_id
        .clone()
        .unwrap_or_else(|| "serializer-unknown".to_owned());
    validate_serializer_readiness_projection_with_repair_state(
        projection,
        &params.byop_repair_state,
        &ReadinessDiagnosticContext::new(
            &conversation_id,
            &request_attempt_id,
            ReadinessTriggerLayer::SerializerValidation,
        ),
    )
}

fn validate_serializer_readiness_projection(
    projection: Vec<ProjectionItem>,
) -> Result<ReadinessReport, ConvertToAPITypeError> {
    validate_serializer_readiness_projection_with_repair_state(
        projection,
        &RepairStateStatus::default(),
        &ReadinessDiagnosticContext::new(
            "test-conversation",
            "test-attempt",
            ReadinessTriggerLayer::SerializerValidation,
        ),
    )
}

fn validate_serializer_readiness_projection_with_repair_state(
    projection: Vec<ProjectionItem>,
    repair_state: &RepairStateStatus,
    diagnostic_context: &ReadinessDiagnosticContext<'_>,
) -> Result<ReadinessReport, ConvertToAPITypeError> {
    let context = ReadinessContext {
        repair_records: repair_state.repair_records().to_vec(),
        live_tool_calls: Vec::new(),
    };
    let report = classify_projection(&projection, &context);

    match &report.state {
        ReadinessState::Ready => {
            if let Some(error_category) = repair_state.error_category() {
                log::error!(
                    "[byop-readiness] serializer continuing with invalid repair sidecar \
                     category={error_category:?} projection_items={}",
                    projection.len()
                );
            }
            Ok(report)
        }
        ReadinessState::AcceptedHistoryRepair { repairs } => {
            log_accepted_history_repair(repairs, diagnostic_context);
            Ok(report)
        }
        ReadinessState::PendingToolResults { .. }
        | ReadinessState::NeedsCancellationCommit { .. }
        | ReadinessState::DuplicateToolResults { .. }
        | ReadinessState::OrphanToolResult { .. }
        | ReadinessState::OutOfOrderToolResult { .. }
        | ReadinessState::MissingResultWithoutRepairSource { .. } => {
            let category = report.state.category();
            let mut diagnostics = ReadinessDiagnosticCoalescer::default();
            diagnostics.log_state(
                &report.state,
                diagnostic_context,
                ReadinessDiagnosticLevel::Error,
            );
            diagnostics.finish(diagnostic_context, ReadinessDiagnosticLevel::Error);
            log::error!(
                "[byop-readiness] serializer blocked request category={category:?} \
                 projection_items={} ignored_repair_records={} trigger_layer=serializer_validation \
                 request_attempt_id={}",
                projection.len(),
                report.ignored_repair_records.len(),
                diagnostic_context.request_attempt_id
            );

            Err(ConvertToAPITypeError::Other(
                BlockedByopReadinessError::new(category).into(),
            ))
        }
    }
}

fn log_accepted_history_repair(
    repairs: &[AcceptedRepair],
    diagnostic_context: &ReadinessDiagnosticContext<'_>,
) {
    log::info!(
        "{}",
        accepted_history_repair_log_message(repairs, diagnostic_context)
    );
}

fn accepted_history_repair_log_message(
    repairs: &[AcceptedRepair],
    diagnostic_context: &ReadinessDiagnosticContext<'_>,
) -> String {
    let forked_history_count = repairs
        .iter()
        .filter(|repair| matches!(repair.record.source, RepairSource::ForkedHistory))
        .count();
    let restored_legacy_history_count = repairs
        .iter()
        .filter(|repair| matches!(repair.record.source, RepairSource::RestoredLegacyHistory))
        .count();

    format!(
        "[byop-readiness] serializer accepted history repair records={} \
         category={:?} forked_history={} restored_legacy_history={} conversation_id={} \
         trigger_layer=serializer_validation request_attempt_id={} repair_keys={:?}",
        repairs.len(),
        ReadinessCategory::AcceptedHistoryRepair,
        forked_history_count,
        restored_legacy_history_count,
        diagnostic_context.conversation_id,
        diagnostic_context.request_attempt_id,
        repairs
            .iter()
            .map(|repair| format!(
                "task_id={} assistant_tool_call_message_id={} tool_call_id={} redacted_tool_kind={}",
                repair.tool_call.key.task_id,
                repair.tool_call.key.assistant_tool_call_message_id,
                repair.tool_call.key.tool_call_id,
                repair.tool_call.redacted_tool_kind.as_str()
            ))
            .collect::<Vec<_>>()
    )
}

/// Translates RequestParams into a genai `ChatRequest` (including system + messages + tools).
///
/// `force_echo_reasoning`: decided by `super::reasoning::model_requires_reasoning_echo`.
/// When true, all assistant messages are forced to carry reasoning_content (empty-string placeholder),
/// fixing thinking-mode endpoints with tightened validation such as DeepSeek-v4-flash / Kimi.
fn build_chat_request(
    params: &RequestParams,
    force_echo_reasoning: bool,
    api_type: AgentProviderApiType,
    attachment_caps: attachment_caps::AttachmentCaps,
) -> Result<ChatRequest, ConvertToAPITypeError> {
    let agent_ctx = latest_input_context(&params.input);
    let plan_mode = is_plan_mode_turn(&params.input);
    let tool_names = available_tool_names(params);
    let mut system_text = prompt_renderer::render_system(
        api_type,
        &params.model,
        agent_ctx,
        &tool_names,
        plan_mode,
        &params.user_rules,
    );
    // Warp: legacy SSH session profile patch. `render_system` goes through AIAgentContext,
    // so the OS/shell it gets are the local client's; under legacy SSH the PTY is actually remote,
    // so an SSH state block is appended to correct the LLM's inference.
    if let Some(ssh_block) = render_ssh_session_block(&params.session_context) {
        system_text.push_str(&ssh_block);
    }
    // Note: the tool-usage guidance for LRC / long commands (write_to_long_running_shell_command + command_id +
    // the various modes and raw byte sequences) is already fully covered in `prompts/system/default.j2:69-79`.
    // The user's concrete current PTY context (command name / alt-screen flag / grid content) is injected separately via the
    // `<attached_running_command>` XML block prefixed to the user message (see
    // `render_running_command_context` and the UserQuery branch in build_chat_request).
    // Do not hard-code things like TUI exit keys at the system layer, to avoid conflicting with or duplicating the standard guidance in default.j2.

    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut outbound_tool_groups: Vec<OutboundAssistantToolGroup> = Vec::new();

    // Collect the messages of all tasks, and use `collect_linearized_task_messages` for deterministic
    // DFS linearization + UserQuery dedupe (fixes Issue #94: history-turn user messages misordered to the
    // end, or duplicated due to LRC subagent copies). See that function's docs.
    let all_msgs: Vec<&api::Message> = collect_linearized_task_messages(&params.tasks);

    // Warp BYOP local conversation compaction: apply conversation.compaction_state to the message sequence.
    //   1. Filter out (user, assistant) pairs already covered by a compaction (`hidden_message_ids`)
    //   2. At the position of the hidden range, insert a synthetic pair (user "compacted, summary follows" + assistant summary text) message;
    //      this is emitted in place in the main loop via the `summary_inserts` index
    //   3. When a ToolCallResult marker.tool_output_compacted_at is non-empty, a later branch replaces content with a placeholder
    //
    // When the current input is `AIAgentInput::SummarizeConversation`: further use the select algorithm to cut messages
    // to the head (dropping the tail); at the end of the input loop `build_prompt(...)` is appended as the user message
    // (going through the full SUMMARY_TEMPLATE), so the upstream LLM outputs a structured summary.
    let is_summarization_request = params
        .input
        .iter()
        .any(|i| matches!(i, AIAgentInput::SummarizeConversation { .. }));
    let summarization_overflow = params.input.iter().any(|i| {
        matches!(
            i,
            AIAgentInput::SummarizeConversation { overflow: true, .. }
        )
    });
    let _ = summarization_overflow; // used by the follow-up wording branch inside the input loop; silenced as dead for now

    let summary_inserts: std::collections::HashMap<String, String> =
        if let Some(state) = params.compaction_state.as_ref() {
            // user_msg_id → summary_text; when this user_msg_id is hit (it would have been hidden), replace it with the synthetic summary pair
            state
                .completed()
                .iter()
                .filter_map(|c| {
                    c.summary_text
                        .as_ref()
                        .map(|s| (c.user_msg_id.clone(), s.clone()))
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };
    let hidden_msg_ids: std::collections::HashSet<String> = params
        .compaction_state
        .as_ref()
        .map(|s| s.hidden_message_ids())
        .unwrap_or_default();
    let compacted_tool_msg_ids: std::collections::HashSet<String> = params
        .compaction_state
        .as_ref()
        .map(|s| {
            // Collect the message_ids of all ToolCallResults marked with tool_output_compacted_at
            // implemented by iterating all_msgs and checking the marker
            let mut out = std::collections::HashSet::new();
            for msg in &all_msgs {
                if let Some(api::message::Message::ToolCallResult(_)) = &msg.message {
                    if s.marker(&msg.id)
                        .and_then(|m| m.tool_output_compacted_at)
                        .is_some()
                    {
                        out.insert(msg.id.clone());
                    }
                }
            }
            out
        })
        .unwrap_or_default();

    // Summarization request path: use byop_compaction::algorithm::select to cut the head; the tail is not sent upstream
    let summarize_head_end: Option<usize> = if is_summarization_request {
        // Temporarily project into WarpMessageView to compute select
        let state_for_select = params.compaction_state.clone().unwrap_or_default();
        let tool_names =
            byop_compaction::message_view::build_tool_name_lookup(all_msgs.iter().copied());
        let views =
            byop_compaction::message_view::project(&all_msgs, &state_for_select, &tool_names);
        let cfg = byop_compaction::CompactionConfig::default();
        let model_limit = byop_compaction::overflow::ModelLimit::FALLBACK;
        let result = byop_compaction::algorithm::select(&views, &cfg, model_limit, |slice| {
            slice
                .iter()
                .map(byop_compaction::algorithm::MessageRef::estimate_size)
                .sum()
        });
        // head_end is the upper bound of the "head range" in views, in the same order as all_msgs
        Some(result.head_end)
    } else {
        None
    };

    let readiness_report = validate_byop_serializer_readiness(
        params,
        &all_msgs,
        summarize_head_end,
        &summary_inserts,
        &hidden_msg_ids,
        &compacted_tool_msg_ids,
    )?;

    let mut buf = AssistantBuffer::new(force_echo_reasoning);
    // Warp: the call_id of the subagent ToolCall skipped in history; its
    // ToolCallResult must also be skipped, otherwise it becomes an orphan tool_response and Anthropic returns 400 directly.
    // `unexpected tool_use_id ... no corresponding tool_use block`。
    let mut skipped_subagent_call_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // Warp: only allow stripping suspected tool JSON in AgentOutput when a structured ToolCall has landed under the same task_id,
    // to avoid collateral damage to plain-text answers across turns (different task_id) from history-wide evidence.
    let tasks_with_tool_call: std::collections::HashSet<&str> = all_msgs
        .iter()
        .filter(|msg| matches!(msg.message, Some(api::message::Message::ToolCall(_))))
        .map(|msg| msg.task_id.as_str())
        .collect();

    for (idx, msg) in all_msgs.iter().enumerate() {
        // Summarization request: the tail range is not sent upstream (only head + SUMMARY_TEMPLATE appended at the end)
        if let Some(head_end) = summarize_head_end {
            if idx >= head_end {
                continue;
            }
        }
        if hidden_msg_ids.contains(&msg.id) {
            if let Some(summary_text) = summary_inserts.get(&msg.id) {
                flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);
                messages.push(ChatMessage::user(
                    "Conversation history was compacted. Below is the structured summary of all prior turns.".to_string(),
                ));
                messages.push(ChatMessage::assistant(summary_text.clone()));
            }
            continue;
        }
        let Some(inner) = &msg.message else {
            continue;
        };
        match inner {
            api::message::Message::UserQuery(u) => {
                flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);
                // Warp: keep multimodal alive across history turns. Warp's own path relies on the cloud server to re-inject InputContext,
                // which the BYOP direct connection lacks, so `make_user_query_message` persists all binaries
                // (image / pdf / audio) into `UserQuery.context.images`; here they are restored back to
                // UserBinary and passed to `build_user_message_with_binaries`, so later turns' model still sees the previously
                // pasted multimodal attachments. Mimes unsupported by the model caps are handled by build_user_message_with_binaries,
                // which replaces them with ERROR text (opencode unsupportedParts style), never silently dropping them.
                // No binary: fall back to the old path `ChatMessage::user(text)`, equivalent to before the fix.
                let history_binaries: Vec<user_context::UserBinary> = u
                    .context
                    .as_ref()
                    .map(|ctx| {
                        ctx.images
                            .iter()
                            .filter(|b| !b.data.is_empty())
                            .enumerate()
                            .map(|(idx, b)| {
                                use base64::Engine;
                                user_context::UserBinary {
                                    name: format!("history-attachment-{}-{idx}", &msg.id),
                                    content_type: if b.mime_type.is_empty() {
                                        "application/octet-stream".to_string()
                                    } else {
                                        b.mime_type.clone()
                                    },
                                    data: base64::engine::general_purpose::STANDARD.encode(&b.data),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let mut history_prefixes: Vec<String> = Vec::new();
                if let Some(prefix) =
                    user_context::render_api_referenced_attachments(&u.referenced_attachments)
                {
                    history_prefixes.push(prefix);
                }
                let history_text = if history_prefixes.is_empty() {
                    u.query.clone()
                } else {
                    format!("{}\n\n{}", history_prefixes.join("\n\n"), u.query)
                };
                if history_binaries.is_empty() {
                    messages.push(ChatMessage::user(history_text));
                } else {
                    messages.push(build_user_message_with_binaries(
                        history_text,
                        history_binaries,
                        attachment_caps,
                    ));
                }
            }
            api::message::Message::AgentReasoning(r) => {
                // Attach the previous turn's reasoning to the next assistant message to be flushed.
                // genai 0.6's with_reasoning_content serializes per the current adapter:
                // DeepSeek/Kimi → reasoning_content field; Anthropic → thinking blocks.
                // Multiple AgentReasoning segments are accumulated (the same turn may stream several reasoning chunks
                // that land as multiple AgentReasoning).
                let next = r.reasoning.clone();
                if !next.is_empty() {
                    match buf.reasoning.as_mut() {
                        Some(existing) => existing.push_str(&next),
                        None => buf.reasoning = Some(next),
                    }
                }
            }
            api::message::Message::AgentOutput(a) => {
                // Ollama local models often land tool JSON as AgentOutput text; later in the same turn there is also a
                // structured ToolCall + ToolCallResult. Sending it upstream again would drown the real summary and cause C3 amnesia.
                // Strip only when (a) a structured ToolCall has already landed under the same task_id, and (b) this text itself can be
                // extracted by extract_tool_calls_from_assistant_text as an executable call;
                // `contains_tool_shaped_json` returns true as soon as a string `name` key appears (e.g. pasted
                // package.json content), which would misjudge a legitimate answer as tool JSON noise and delete it.
                if api_type == AgentProviderApiType::Ollama
                    && tasks_with_tool_call.contains(msg.task_id.as_str())
                    && !super::content_tool_calls::extract_tool_calls_from_assistant_text(
                        &a.text,
                        &tool_names,
                    )
                    .is_empty()
                {
                    continue;
                }
                if buf.text.is_some() || !buf.tool_calls.is_empty() {
                    flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);
                }
                buf.text = Some(a.text.clone());
            }
            api::message::Message::ToolCall(tc) => {
                // Warp BYOP: **the virtual subagent tool_call is not sent to the upstream model**.
                // In the LRC tag-in scenario, we synthesize `Tool::Subagent { metadata: Cli }` at the head of the chat_stream stream
                // and write it into root.task.messages, only to trigger conversation creation of the cli subtask + spawn of the floating window;
                // it is not a tool call the model actually produced, and the model would be confused by seeing it (an extra tool call + no way to respond).
                // Its corresponding ToolCallResult must likewise be filtered by the branch below, to avoid the
                // imbalance of "tool_response has no matching tool_call".
                use crate::ai::agent::task::helper::ToolCallExt;
                if tc.subagent().is_some() {
                    skipped_subagent_call_ids.insert(tc.tool_call_id.clone());
                    continue;
                }
                if buf
                    .tool_call_keys
                    .first()
                    .is_some_and(|key| key.task_id != msg.task_id)
                {
                    flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);
                }
                let (name, args_json) = serialize_outgoing_tool_call(
                    tc,
                    params.mcp_context.as_ref(),
                    &msg.server_message_data,
                );
                let assistant_message_id = buf
                    .tool_call_keys
                    .first()
                    .map(|key| key.assistant_tool_call_message_id.clone())
                    .unwrap_or_else(|| msg.id.clone());
                let key = ToolCallKey::new(&msg.task_id, assistant_message_id, &tc.tool_call_id);
                buf.push_tool_call(
                    ToolCall {
                        call_id: tc.tool_call_id.clone(),
                        fn_name: name,
                        fn_arguments: args_json,
                        thought_signatures: None,
                    },
                    key,
                );
            }
            api::message::Message::ToolCallResult(tcr) => {
                flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);
                // Warp: the corresponding ToolCall was skipped (virtual subagent call), so the result is skipped too,
                // otherwise an orphan tool_response is left and upstream returns 400.
                if skipped_subagent_call_ids.contains(&tcr.tool_call_id) {
                    continue;
                }
                // BYOP-persisted ToolCallResult goes through server_message_data (content is already a JSON string);
                // server-side emit uses the structured variant of the result oneof; both paths are supported.
                let content = if compacted_tool_msg_ids.contains(&msg.id) {
                    // Compaction projection: pruned tool output is replaced with a placeholder; the actual content is not sent upstream
                    r#"{"status":"compacted","note":"tool output was pruned by local compaction"}"#
                        .to_string()
                } else if tcr.result.is_some() {
                    tools::serialize_result(tcr)
                } else if !msg.server_message_data.is_empty() {
                    msg.server_message_data.clone()
                } else {
                    r#"{"status":"empty"}"#.to_owned()
                };
                messages.push(ChatMessage::from(ToolResponse::new(
                    tcr.tool_call_id.clone(),
                    content,
                )));
            }
            _ => {
                // Other message types (SystemQuery/UpdateTodos/...) are not sent upstream by BYOP for now.
            }
        }
    }
    flush_assistant_buffer(&mut buf, &mut messages, &mut outbound_tool_groups);

    // New input for the current turn: append.
    for input in &params.input {
        match input {
            AIAgentInput::UserQuery {
                query,
                context,
                referenced_attachments,
                running_command,
                ..
            } => {
                // Attachment-type context carried by the current turn's UserQuery (Block / SelectedText / File / Image)
                // strictly follows the semantics of warp's own path, where `api::InputContext.executed_shell_commands` and similar fields
                // are uploaded and the backend injects them into the prompt. BYOP has no backend layer, so they are prepended directly to the user message.
                // Environment-type context (env / git / skills / ...) is rendered into system by prompt_renderer,
                // and does not overlap with this path.
                //
                // Warp: in the LRC tag-in scenario, `running_command: Some(...)` contains the full PTY context
                // (alt-screen grid_contents + command + is_alt_screen_active flag), rendered by
                // `render_running_command_context` into an `<attached_running_command>` XML block.
                // The model uses it to decide whether to call write_to_long_running_shell_command.
                // When absent (normal conversation or not injected by the controller), fall back to the short `lrc_command_id` context.
                //
                // **P1-10 prompt cache optimization**: the LRC context block is **appended after the query** rather
                // than prefixed. Reasons:
                //   - grid_contents changes every second with PTY state and is "high-frequency changing" content.
                //   - Putting it before the query makes the head of the user message unstable, so the hashes written by the last
                //     2 Anthropic breakpoints in the messages segment are always different and reuse value is low.
                //   - Placed after the query, the same query (e.g. "exit nvim") still shares the prefix
                //     "the user's question" across different PTY snapshots, improving cross-call reuse.
                // The model's behavior differs little: whether the instruction or the context comes first, the model understands correctly.
                // The prefix of user_attachments (e.g. SelectedText / Block) still stays in the prefix position, because
                // they correspond to content the user "explicitly selected" and should serve as question background rather than instance supplement.
                let mut suffixes: Vec<String> = Vec::new();
                let request_running_command = running_command
                    .as_ref()
                    .or(params.lrc_running_command.as_ref());
                if let Some(rc) = request_running_command {
                    suffixes.push(render_running_command_context(rc));
                } else if let Some(command_id) = params.lrc_command_id.as_deref() {
                    suffixes.push(render_running_command_id_context(command_id));
                }
                let mut prefixes: Vec<String> = Vec::new();
                let user_attachments = user_context::collect_user_attachments(context);
                if let Some(p) = &user_attachments.prefix {
                    prefixes.push(p.clone());
                }
                if let Some(p) = user_context::render_referenced_attachments(referenced_attachments)
                {
                    prefixes.push(p);
                }
                let full_text = match (prefixes.is_empty(), suffixes.is_empty()) {
                    (true, true) => query.clone(),
                    (false, true) => format!("{}\n\n{query}", prefixes.join("\n\n")),
                    (true, false) => format!("{query}\n\n{}", suffixes.join("\n\n")),
                    (false, false) => format!(
                        "{}\n\n{query}\n\n{}",
                        prefixes.join("\n\n"),
                        suffixes.join("\n\n"),
                    ),
                };
                log::info!(
                    "[byop-diag] build_chat_request UserQuery: query_len={} \
                     running_command={} prefixes={} suffixes={} full_text_len={} binaries={}",
                    query.len(),
                    match request_running_command {
                        Some(rc) => format!(
                            "Some(grid_len={} alt={})",
                            rc.grid_contents.len(),
                            rc.is_alt_screen_active
                        ),
                        None => "None".to_owned(),
                    },
                    prefixes.len(),
                    suffixes.len(),
                    full_text.len(),
                    user_attachments.binaries.len(),
                );
                messages.push(build_user_message_with_binaries(
                    full_text,
                    user_attachments.binaries,
                    attachment_caps,
                ));
            }
            AIAgentInput::ActionResult { result, .. } => {
                // The previous model turn replied with tool_calls; after the client executes them, the result goes through `params.input`
                // rather than the `params.tasks` history. It must be serialized as a ToolResponse here, otherwise
                // genai/upstream will return 400 due to tool_call_id pairing failure.
                let tool_call_id = result.id.to_string();
                let content = tools::serialize_action_result(result).unwrap_or_else(|| {
                    serde_json::json!({ "result": result.result.to_string() }).to_string()
                });
                messages.push(ChatMessage::from(ToolResponse::new(tool_call_id, content)));
            }
            AIAgentInput::InvokeSkill {
                skill, user_query, ..
            } => {
                let mut composed = format!(
                    "Perform the task following the guidance of the skill \"{}\" below:\n\n{}\n\n---\n",
                    skill.name, skill.content,
                );
                if let Some(uq) = user_query {
                    composed.push_str(&format!("Further user instructions: {}", uq.query));
                }
                messages.push(ChatMessage::user(composed));
            }
            AIAgentInput::ResumeConversation { context } => {
                // BYOP has no server-side resume prompt injection layer. On LRC auto-resume the current PTY context must be explicitly
                // re-attached, otherwise the error-recovery turn degrades into a normal conversation and picks a shell tool again.
                let mut prefixes: Vec<String> = Vec::new();
                if let Some(lrc_prefix) = render_lrc_request_context(params) {
                    prefixes.push(lrc_prefix);
                }
                let user_attachments = user_context::collect_user_attachments(context);
                if let Some(p) = &user_attachments.prefix {
                    prefixes.push(p.clone());
                }
                if !prefixes.is_empty() {
                    let full_text = format!("{}\n\nContinue.", prefixes.join("\n\n"));
                    messages.push(build_user_message_with_binaries(
                        full_text,
                        user_attachments.binaries,
                        attachment_caps,
                    ));
                }
            }
            AIAgentInput::SummarizeConversation {
                prompt,
                overflow: _,
            } => {
                // Warp BYOP local conversation compaction entry, 1:1 aligned with opencode `compaction.ts processCompaction`.
                //
                // Earlier the messages loop already cut the sequence to head (dropping tail) per `summarize_head_end`;
                // here the last user message is appended: `build_prompt(previous_summary, plugin_context)`,
                // which contains SUMMARY_TEMPLATE (9-section Markdown template) + the incremental summary anchor.
                //
                // The model emits a structured Markdown summary text; after the stream completes the controller
                // writes it back to conversation.compaction_state (see the Phase 6 controller changes).
                let prev_summary = params
                    .compaction_state
                    .as_ref()
                    .and_then(|s| s.previous_summary())
                    .map(str::to_string);
                let mut anchor_context: Vec<String> = Vec::new();
                if let Some(custom) = prompt.as_ref().filter(|p| !p.is_empty()) {
                    // /compact <custom instructions> goes here; the user's instructions are appended to the plugin_context section
                    anchor_context
                        .push(format!("Additional instructions from the user:\n{custom}"));
                }
                let nextp =
                    byop_compaction::prompt::build_prompt(prev_summary.as_deref(), &anchor_context);
                messages.push(ChatMessage::user(nextp));
            }
            AIAgentInput::AutoCodeDiffQuery { .. }
            | AIAgentInput::CreateNewProject { .. }
            | AIAgentInput::CodeReview { .. } => {
                // Ignored for now
            }
            _ => {}
        }
    }

    if let ReadinessState::AcceptedHistoryRepair { repairs } = &readiness_report.state {
        repair_tool_call_pairs_for_accepted_history_gaps(
            &mut messages,
            repairs,
            &outbound_tool_groups,
        )?;
    }

    // Defensive sanitize: make sure messages does not end with assistant.
    // Anthropic / some gateways do not accept requests ending with assistant (prefill is only supported by specific models),
    // and warp's `AIAgentInput::ResumeConversation` (handoff / auto-resume after error, etc.)
    // does not append a new user message, leaving the sequence ending on a history assistant.
    // Unified fallback here: if it ends with assistant, append an implicit user message to let upstream continue.
    ensure_ends_with_user(&mut messages);

    let mut tools_array = build_tools_array(params);

    // Anthropic path: put a 1h cache_control breakpoint on the **last tool** of the tools array,
    // making the whole tools segment a long-TTL static prefix (aligned with Zed
    // `crates/anthropic/src/completion.rs::254-258`)。
    //
    // Processing order `tools → system → messages`; long TTL must come before short TTL. On this path:
    // - tools tail 1h (here)
    // - system 1h (`apply_caching_anthropic` on the ChatRole::System message)
    // - messages tail 5m (`apply_caching_anthropic` on the last 2 non-system)
    //
    // The tools segment changes least within a session (only on switching web_search / plan_mode / LRC), so the hit rate is
    // very high, and the 1h write at 2× base amortized over many reuses is effectively near 0; it also blocks external reverse proxies from injecting 5m on
    // system, which causes a 1h-after-5m ordering error (letting this one tools 1h breakpoint take over
    // the entire static tools+system prefix).
    if matches!(api_type, AgentProviderApiType::Anthropic) {
        if let Some(last_tool) = tools_array.last_mut() {
            last_tool.cache_control = Some(CacheControl::Ephemeral1h);
        }
    }

    // Outbound message text is passed through to `serde_json` for JSON escaping, with no more aggressive char-level
    // sanitize (see zed `into_anthropic` / opencode `provider/transform.ts`,
    // neither flattens control chars or replaces `\` / `"` at the outbound layer). The Anthropic / OpenAI / Gemini
    // official APIs and mainstream BYOP reverse proxies all handle the legal escapes produced by `serde_json` correctly.

    // Prompt caching (1:1 ported from opencode `provider/transform.ts::applyCaching`):
    // - opencode picks the first 2 system messages + last 2 non-system messages and tags them uniformly with
    //   anthropic.cacheControl / openaiCompatible.cache_control / bedrock.cachePoint
    //   and other multi-SDK-compatible markers. Each provider implementation in the AI SDK reads its own key; irrelevant keys are ignored automatically.
    // - We use rust-genai; the Anthropic adapter supports per-message `cache_control`,
    //   the OpenAI / OpenAiResp adapters only recognize the `ChatOptions`-level prompt_cache_key /
    //   cache_control, and DeepSeek / Gemini / Ollama use implicit server-side caching with no client opt-in.
    // - So here only the Anthropic path is tagged "per-message": push the system text to the head of messages as a
    //   ChatRole::System message tagged Ephemeral, then also tag the last two
    //   non-system messages Ephemeral (matching opencode's system + last 2 pattern).
    //   The OpenAI-family `prompt_cache_key` / `cache_control` are set in `build_chat_options`
    //   (request level), also a downstream fallback from the same set of opencode rules.
    let messages = if matches!(api_type, AgentProviderApiType::Anthropic) {
        let mut msgs: Vec<ChatMessage> = std::iter::once(ChatMessage::system(system_text.clone()))
            .chain(messages)
            .collect();
        apply_caching_anthropic(&mut msgs);
        msgs
    } else {
        messages
    };

    let mut req = ChatRequest::from_messages(messages);
    // On the Anthropic path system is already in messages as a ChatRole::System message,
    // so `with_system` is no longer set, avoiding genai Anthropic adapter's "the first system cannot carry
    // cache_control" restriction (comment in `adapter_impl.rs::into_anthropic_request_parts`).
    if !matches!(api_type, AgentProviderApiType::Anthropic) {
        req = req.with_system(system_text);
    }
    if !tools_array.is_empty() {
        req = req.with_tools(tools_array);
    }
    Ok(req)
}

const BYOP_DIAG_SNIPPET_CHARS: usize = 240;
const REPAIR_PLACEHOLDER_NOTE: &str =
    "tool result was unavailable in repaired conversation history";

fn is_placeholder_tool_response_content(content: &str) -> bool {
    if content == "(tool result not retained)" {
        return true;
    }

    let Ok(Value::Object(object)) = serde_json::from_str::<Value>(content) else {
        return false;
    };

    object.len() == 3
        && object.get("status").and_then(Value::as_str) == Some("unavailable")
        && matches!(
            object.get("reason").and_then(Value::as_str),
            Some("forked_history_repair" | "restored_legacy_history_repair")
        )
        && object.get("note").and_then(Value::as_str) == Some(REPAIR_PLACEHOLDER_NOTE)
}

fn insert_preferred_tool_response(
    responses_by_call_id: &mut HashMap<String, ToolResponse>,
    response: &ToolResponse,
) {
    let should_replace = match responses_by_call_id.get(&response.call_id) {
        None => true,
        Some(existing) => should_replace_tool_response(existing, response),
    };
    if should_replace {
        responses_by_call_id.insert(response.call_id.clone(), response.clone());
    }
}

fn should_replace_tool_response(existing: &ToolResponse, candidate: &ToolResponse) -> bool {
    is_placeholder_tool_response_content(&existing.content)
        || !is_placeholder_tool_response_content(&candidate.content)
}

fn snippet_for_log(s: &str, max_chars: usize) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for (idx, ch) in s.chars().enumerate() {
        if idx >= max_chars {
            out.push_str("...");
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{{{:04x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn json_value_for_log(value: &Value) -> (usize, String) {
    let json = serde_json::to_string(value)
        .unwrap_or_else(|_| "<failed-to-serialize-json-value>".to_owned());
    (json.len(), snippet_for_log(&json, BYOP_DIAG_SNIPPET_CHARS))
}

fn binary_for_log(binary: &Binary) -> String {
    let name = binary
        .name
        .as_deref()
        .map(|n| snippet_for_log(n, 80))
        .unwrap_or_default();
    match &binary.source {
        BinarySource::Base64(data) => format!(
            "mime={} name={} source=base64 chars={}",
            binary.content_type,
            name,
            data.len()
        ),
        BinarySource::Url(url) => format!(
            "mime={} name={} source=url chars={} url={}",
            binary.content_type,
            name,
            url.len(),
            snippet_for_log(url, 120)
        ),
    }
}

fn log_chat_request_details(
    chat_req: &ChatRequest,
    model_id: &str,
    api_type: AgentProviderApiType,
    base_url: &str,
) {
    let system_in_head = matches!(api_type, AgentProviderApiType::Anthropic)
        && chat_req
            .messages
            .first()
            .map(|m| matches!(m.role, ChatRole::System))
            .unwrap_or(false);
    let tool_count = chat_req.tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let tool_names: Vec<String> = chat_req
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(|t| t.name.as_str().to_owned()).collect())
        .unwrap_or_default();
    log::info!(
        "[byop-diag] request summary: adapter={:?} model={} system_len={} \
         system_in_messages_head={} messages={} tools={} tool_names={:?} \
         previous_response_id_present={} store={:?} system_snippet={:?}",
        effective_adapter_kind_for(api_type, model_id, base_url),
        model_id,
        chat_req.system.as_deref().map(str::len).unwrap_or(0),
        system_in_head,
        chat_req.messages.len(),
        tool_count,
        tool_names,
        chat_req.previous_response_id.is_some(),
        chat_req.store,
        chat_req
            .system
            .as_deref()
            .map(|s| snippet_for_log(s, BYOP_DIAG_SNIPPET_CHARS))
            .unwrap_or_default(),
    );

    if let Some(tools) = &chat_req.tools {
        for (idx, tool) in tools.iter().enumerate() {
            let schema_len = tool
                .schema
                .as_ref()
                .and_then(|schema| serde_json::to_string(schema).ok())
                .map(|schema| schema.len())
                .unwrap_or(0);
            log::info!(
                "[byop-diag] request tool[{idx}]: name={} desc_len={} schema_len={} \
                 strict={:?} cache_control={:?}",
                tool.name.as_str(),
                tool.description.as_deref().map(str::len).unwrap_or(0),
                schema_len,
                tool.strict,
                tool.cache_control,
            );
        }
    }

    let flow: Vec<String> = chat_req
        .messages
        .iter()
        .enumerate()
        .map(|(idx, msg)| {
            let text_len: usize = msg.content.texts().iter().map(|t| t.len()).sum();
            let tool_call_ids: Vec<String> = msg
                .content
                .tool_calls()
                .iter()
                .map(|tc| tc.call_id.clone())
                .collect();
            let tool_response_ids: Vec<String> = msg
                .content
                .tool_responses()
                .iter()
                .map(|tr| tr.call_id.clone())
                .collect();
            format!(
                "{idx}:{:?}(text_len={text_len},tool_calls={tool_call_ids:?},tool_responses={tool_response_ids:?})",
                msg.role
            )
        })
        .collect();
    log::info!("[byop-diag] request message_flow={flow:?}");

    for (idx, msg) in chat_req.messages.iter().enumerate() {
        let mut text_count = 0;
        let mut text_total_len = 0;
        let mut first_text_snippet: Option<String> = None;
        let mut binary_summaries: Vec<String> = Vec::new();
        let mut tool_call_summaries: Vec<String> = Vec::new();
        let mut tool_response_summaries: Vec<String> = Vec::new();
        let mut thought_count = 0;
        let mut thought_total_len = 0;
        let mut reasoning_count = 0;
        let mut reasoning_total_len = 0;
        let mut custom_count = 0;

        for part in &msg.content {
            match part {
                ContentPart::Text(text) => {
                    text_count += 1;
                    text_total_len += text.len();
                    if first_text_snippet.is_none() {
                        first_text_snippet = Some(snippet_for_log(text, BYOP_DIAG_SNIPPET_CHARS));
                    }
                }
                ContentPart::Binary(binary) => {
                    binary_summaries.push(binary_for_log(binary));
                }
                ContentPart::ToolCall(tool_call) => {
                    let (args_len, args_snippet) = json_value_for_log(&tool_call.fn_arguments);
                    tool_call_summaries.push(format!(
                        "call_id={} name={} args_len={} args={} thought_signatures={}",
                        tool_call.call_id,
                        tool_call.fn_name,
                        args_len,
                        args_snippet,
                        tool_call
                            .thought_signatures
                            .as_ref()
                            .map(|s| s.len())
                            .unwrap_or(0)
                    ));
                }
                ContentPart::ToolResponse(tool_response) => {
                    tool_response_summaries.push(format!(
                        "call_id={} content_len={} placeholder={} content={}",
                        tool_response.call_id,
                        tool_response.content.len(),
                        is_placeholder_tool_response_content(&tool_response.content),
                        snippet_for_log(&tool_response.content, BYOP_DIAG_SNIPPET_CHARS)
                    ));
                }
                ContentPart::ThoughtSignature(thought) => {
                    thought_count += 1;
                    thought_total_len += thought.len();
                }
                ContentPart::ReasoningContent(reasoning) => {
                    reasoning_count += 1;
                    reasoning_total_len += reasoning.len();
                }
                ContentPart::Custom(_) => {
                    custom_count += 1;
                }
            }
        }

        let cache_control = msg
            .options
            .as_ref()
            .and_then(|options| options.cache_control.as_ref())
            .map(|cache| format!("{cache:?}"))
            .unwrap_or_else(|| "None".to_owned());
        log::info!(
            "[byop-diag] request message[{idx}]: role={:?} parts={} size={} \
             cache_control={} text_parts={} text_total_len={} first_text={:?} \
             binaries={:?} tool_calls={:?} tool_responses={:?} \
             thought_signatures={} thought_total_len={} reasoning_parts={} \
             reasoning_total_len={} custom_parts={}",
            msg.role,
            msg.content.len(),
            msg.content.size(),
            cache_control,
            text_count,
            text_total_len,
            first_text_snippet.unwrap_or_default(),
            binary_summaries,
            tool_call_summaries,
            tool_response_summaries,
            thought_count,
            thought_total_len,
            reasoning_count,
            reasoning_total_len,
            custom_count,
        );
    }

    for (idx, msg) in chat_req.messages.iter().enumerate() {
        let expected_call_ids: Vec<String> = msg
            .content
            .tool_calls()
            .iter()
            .map(|tc| tc.call_id.clone())
            .collect();
        if expected_call_ids.is_empty() {
            continue;
        }
        let next = chat_req.messages.get(idx + 1);
        let next_role = next.map(|m| format!("{:?}", m.role)).unwrap_or_default();
        let response_call_ids: Vec<String> = next
            .filter(|m| matches!(m.role, ChatRole::Tool))
            .map(|m| {
                m.content
                    .tool_responses()
                    .iter()
                    .map(|tr| tr.call_id.clone())
                    .collect()
            })
            .unwrap_or_default();
        let matched = response_call_ids == expected_call_ids;
        if matched {
            log::info!(
                "[byop-diag] request tool_pair idx={idx}: expected_call_ids={expected_call_ids:?} \
                 next_role={next_role} response_call_ids={response_call_ids:?}"
            );
        } else {
            log::warn!(
                "[byop-diag] request tool_pair mismatch idx={idx}: \
                 expected_call_ids={expected_call_ids:?} next_role={next_role} \
                 response_call_ids={response_call_ids:?}"
            );
        }
    }

    for (idx, msg) in chat_req.messages.iter().enumerate() {
        if !matches!(msg.role, ChatRole::Tool) {
            continue;
        }
        let response_call_ids: Vec<String> = msg
            .content
            .tool_responses()
            .iter()
            .map(|tr| tr.call_id.clone())
            .collect();
        let previous_expected: Vec<String> = idx
            .checked_sub(1)
            .and_then(|prev_idx| chat_req.messages.get(prev_idx))
            .filter(|prev| matches!(prev.role, ChatRole::Assistant))
            .map(|prev| {
                prev.content
                    .tool_calls()
                    .iter()
                    .map(|tc| tc.call_id.clone())
                    .collect()
            })
            .unwrap_or_default();
        if response_call_ids != previous_expected {
            log::warn!(
                "[byop-diag] request orphan_or_misordered_tool_response idx={idx}: \
                 response_call_ids={response_call_ids:?} previous_assistant_call_ids={previous_expected:?}"
            );
        }
    }
}

/// 1:1 ported from the Anthropic branch of opencode `provider/transform.ts::applyCaching`:
/// tags the first 2 system messages + last 2 non-system messages with cache markers.
///
/// The genai Anthropic adapter, inside `into_anthropic_request_parts`, places
/// `MessageOptions::cache_control` on the last content part of that message,
/// matching opencode's lastContent.providerOptions.anthropic.cacheControl.
///
/// **TTL choice (revised from P0-4, aligned with Zed `crates/anthropic/src/completion.rs:219-274`)**:
/// the static-prefix system uses 1h; the conversation tail (last 2 non-system) uses 5m; at the same time `build_chat_request`
/// puts 1h on the tools tail on the Anthropic path (the genai Tool struct now has a cache_control field).
///
/// **Motivation for the mixed strategy**:
/// - The old "all 1h" strategy cannot coexist with external 5m breakpoints. When a BYOP reverse proxy / upstream gateway injects default 5m at an earlier
///   position (tools/system), Anthropic's ordering constraint rejects the 1h that follows:
///   `a ttl='1h' cache_control block must not come after a ttl='5m'`, triggering 400.
/// - New strategy: all long TTLs go first (tools / system 1h), short TTL at the tail of the sequence (messages 5m),
///   strictly matching Anthropic's processing order `tools → system → messages`; if an external party stuffs extra 5m on tools/
///   system, it just becomes "two 5m in front + tail 5m" and no longer triggers 1h-after-5m.
///
/// **Cost impact**:
/// - tools/system are a static prefix within a session; the 1h write at 2× base is amortized over many hits within the whole session,
///   effectively near 0 (the total cost drops markedly vs. "all 1h" repeatedly rewriting the tail).
/// - The messages tail changes every turn; the 5m write is 1.25× base, a hit within 5 min on the next turn renews it for free, with no
///   repeated rewriting and no need to submit a large one-time 1h write.
///
/// **TTL ordering constraint**: the Anthropic API requires long-TTL breakpoints to come before short-TTL ones
/// (`https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
///  #mixing-different-ttls`). This function tags system 1h and the non-system tail 5m, so the order
///  naturally satisfies system(1h) → messages(5m); `build_chat_request` also puts 1h on the tools tail,
///  which also precedes system, so the overall order tools(1h) → system(1h) → messages(5m) remains compliant.
/// genai checks the order in `into_anthropic_request_parts` and warns on violation (see
/// `lib/rust-genai/src/adapter/adapters/anthropic/adapter_impl.rs`)。
fn apply_caching_anthropic(messages: &mut Vec<ChatMessage>) {
    let n = messages.len();
    if n == 0 {
        return;
    }
    let mut tag = vec![false; n];

    // first 2 system messages
    let mut sys_seen = 0;
    for (i, m) in messages.iter().enumerate() {
        if matches!(m.role, ChatRole::System) {
            tag[i] = true;
            sys_seen += 1;
            if sys_seen >= 2 {
                break;
            }
        }
    }
    // last 2 non-system messages
    let mut tail_seen = 0;
    for (i, m) in messages.iter().enumerate().rev() {
        if !matches!(m.role, ChatRole::System) {
            tag[i] = true;
            tail_seen += 1;
            if tail_seen >= 2 {
                break;
            }
        }
    }

    let original = std::mem::take(messages);
    *messages = original
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            if tag[i] {
                // ChatMessage has no direct with_cache_control; cache_control hangs on
                // `MessageOptions`, injected via `with_options`.
                // `MessageOptions: From<CacheControl>` is provided by genai
                // (`chat_message.rs::impl From<CacheControl> for MessageOptions`)。
                // Mixed TTL: system uses 1h (static prefix, hit many times within a session to amortize the write),
                // non-system uses 5m (conversation tail changes every turn; a hit within 5 min on the next turn renews for free).
                // Order system(1h) → messages(5m) satisfies Anthropic's ordering constraint.
                let ttl = if matches!(m.role, ChatRole::System) {
                    CacheControl::Ephemeral1h
                } else {
                    CacheControl::Ephemeral5m
                };
                m.with_options(ttl)
            } else {
                m
            }
        })
        .collect();
}

/// Runs only after the serializer has determined `AcceptedHistoryRepair`: converts history gaps
/// explicitly authorized by a RepairRecord into outbound-only structured ToolResponses.
///
/// Ordinary missing, duplicate, orphan, or cross-boundary out-of-order cases were already blocked at the readiness validation stage;
/// no placeholder results are generated for the normal flow here, and nothing is written back to conversation history.
fn repair_tool_call_pairs_for_accepted_history_gaps(
    messages: &mut Vec<ChatMessage>,
    repairs: &[AcceptedRepair],
    outbound_tool_groups: &[OutboundAssistantToolGroup],
) -> Result<(), ConvertToAPITypeError> {
    use std::collections::{HashMap, HashSet};

    if repairs.is_empty() {
        return Ok(());
    }

    let repair_by_key: HashMap<ToolCallKey, &AcceptedRepair> = repairs
        .iter()
        .map(|repair| (repair.tool_call.key.clone(), repair))
        .collect();
    let group_by_message_index: HashMap<usize, &OutboundAssistantToolGroup> = outbound_tool_groups
        .iter()
        .map(|group| (group.message_index, group))
        .collect();
    let mut call_id_counts: HashMap<String, usize> = HashMap::new();
    for group in outbound_tool_groups {
        for key in &group.tool_call_keys {
            *call_id_counts.entry(key.tool_call_id.clone()).or_default() += 1;
        }
    }
    let mut placeholders_inserted: Vec<String> = Vec::new();
    let mut orphan_call_ids: Vec<String> = Vec::new();
    let mut missing_without_repair: Vec<String> = Vec::new();

    let original = std::mem::take(messages);
    let mut late_responses_by_unique_call_id: HashMap<String, ToolResponse> = HashMap::new();
    let mut late_response_call_ids: HashSet<String> = HashSet::new();
    for (idx, msg) in original.iter().enumerate() {
        if msg.role != genai::chat::ChatRole::Tool {
            continue;
        }

        let is_adjacent_to_group =
            idx > 0 && group_by_message_index.contains_key(&(idx.saturating_sub(1)));
        if is_adjacent_to_group {
            continue;
        }

        for resp in msg.content.tool_responses() {
            if call_id_counts.get(&resp.call_id) == Some(&1) {
                insert_preferred_tool_response(&mut late_responses_by_unique_call_id, resp);
                late_response_call_ids.insert(resp.call_id.clone());
            }
        }
    }

    let mut rewritten: Vec<ChatMessage> = Vec::with_capacity(original.len());
    let mut idx = 0;
    while idx < original.len() {
        let msg = original[idx].clone();
        if msg.role == genai::chat::ChatRole::Tool {
            orphan_call_ids.extend(
                msg.content
                    .tool_responses()
                    .iter()
                    .filter(|response| !late_response_call_ids.contains(&response.call_id))
                    .map(|response| response.call_id.clone()),
            );
            idx += 1;
            continue;
        }

        let Some(group) = group_by_message_index.get(&idx).copied() else {
            rewritten.push(msg);
            idx += 1;
            continue;
        };

        rewritten.push(msg);
        idx += 1;

        let mut responses_by_call_id: HashMap<String, ToolResponse> = HashMap::new();
        while idx < original.len() && original[idx].role == genai::chat::ChatRole::Tool {
            for resp in original[idx].content.tool_responses() {
                insert_preferred_tool_response(&mut responses_by_call_id, resp);
            }
            idx += 1;
        }

        let mut bundled: Vec<ToolResponse> = Vec::new();
        for key in &group.tool_call_keys {
            let cid = &key.tool_call_id;
            let mut response = responses_by_call_id.remove(cid);
            if call_id_counts.get(cid) == Some(&1) {
                if let Some(late_response) = late_responses_by_unique_call_id.remove(cid) {
                    response = match response {
                        Some(existing)
                            if !should_replace_tool_response(&existing, &late_response) =>
                        {
                            Some(existing)
                        }
                        _ => Some(late_response),
                    };
                }
            }

            match response {
                Some(resp) => bundled.push(resp),
                None => {
                    if let Some(repair) = repair_by_key.get(key) {
                        placeholders_inserted.push(cid.clone());
                        bundled.push(ToolResponse::new(
                            cid.clone(),
                            repair_placeholder_content(repair.record.source),
                        ));
                    } else {
                        missing_without_repair.push(cid.clone());
                    }
                }
            }
        }

        if !bundled.is_empty() {
            rewritten.push(ChatMessage::from(bundled));
        }

        if !responses_by_call_id.is_empty() {
            orphan_call_ids.extend(responses_by_call_id.into_keys());
        }
    }

    *messages = rewritten;

    if !orphan_call_ids.is_empty() {
        log::warn!(
            "[byop-diag] accepted_history_repair: dropped {} orphan ToolResponses: \
             orphan_call_ids={:?}",
            orphan_call_ids.len(),
            orphan_call_ids
        );
    }
    if !placeholders_inserted.is_empty() {
        log::info!(
            "[byop-diag] accepted_history_repair: added repair placeholder for {} ToolCalls \
             ToolResponse: missing_call_ids={:?}",
            placeholders_inserted.len(),
            placeholders_inserted
        );
    }
    if !missing_without_repair.is_empty() {
        // When the readiness classifier has determined AcceptedHistoryRepair, every missing tool call should
        // have a matching authorization in repairs; if one is still missing here, the classifier and the serializer's
        // tool call key sources have diverged (e.g. a future refactor of projection or outbound_tool_groups construction
        // introduces a difference). In that case we cannot keep emitting an illegal request lacking a ToolResponse; it must be blocked.
        log::error!(
            "[byop-diag] accepted_history_repair: missing ToolResponse not authorized by readiness: \
             missing_call_ids={:?}",
            missing_without_repair
        );
        return Err(ConvertToAPITypeError::Other(
            BlockedByopReadinessError::new(ReadinessCategory::MissingResultWithoutRepairSource)
                .into(),
        ));
    }
    Ok(())
}

fn repair_placeholder_content(source: RepairSource) -> String {
    json!({
        "status": "unavailable",
        "reason": source.placeholder_reason(),
        "note": REPAIR_PLACEHOLDER_NOTE,
    })
    .to_string()
}

/// Fallback: ensure messages ends with user (or a tool response).
///
/// Trigger: `AIAgentInput::ResumeConversation` appends no new user message and resends the history directly.
/// The native Anthropic API rejects requests ending with assistant (`This model does not support
/// assistant message prefill. The conversation must end with a user message.`),
/// Retrying 3 times with the same payload → the UI renders an error block and triggers a flex panic.
///
/// When it ends with assistant, append `ChatMessage::user("Continue.")` to prompt the model to continue.
/// The Tool role counts as a kind of user input (the model treats a tool response as the start of the next turn) and is left alone.
/// Empty messages do not trigger it, to avoid making up content for an empty conversation.
fn ensure_ends_with_user(messages: &mut Vec<ChatMessage>) {
    use genai::chat::ChatRole;
    if let Some(last) = messages.last() {
        if last.role == ChatRole::Assistant {
            messages.push(ChatMessage::user("Continue."));
        }
    }
}

/// Reverse: serialize the internal `tool_call::Tool` variant into (function name, arguments JSON Value)
/// for multi-turn history replay. The (name, args) here must strictly align with each tool's `name` in `tools::REGISTRY`
/// and the schema expected by `from_args`.
fn serialize_outgoing_tool_call(
    tc: &api::message::ToolCall,
    mcp_ctx: Option<&crate::ai::agent::MCPContext>,
    server_message_data: &str,
) -> (String, Value) {
    use api::message::tool_call::Tool;

    // Restore the BYOP from_args parse-failure carrier: written by make_tool_call_carrier_message,
    // tool oneof = None, the original `<fn_name>\n<args_str>` is encoded in server_message_data.
    // It must be recognized first, before the main match, otherwise it falls through to None=>"warp_internal_empty" below,
    // and the upstream model seeing a nonexistent tool name would be even more confused and not know which call failed.
    if tc.tool.is_none() {
        if let Some((fn_name, raw_args)) = server_message_data.split_once('\n') {
            if !fn_name.is_empty() {
                // When raw_args is invalid JSON (e.g. model emitted \e or \` escape),
                // fall back to an empty object so the provider receives valid JSON
                // rather than a JSON string wrapping the invalid content (which causes
                // "Invalid \escape" 400 on the next turn).
                let args_value = serde_json::from_str(raw_args).unwrap_or_else(|_| json!({}));
                return (fn_name.to_owned(), args_value);
            }
        }
    }

    // Most old implementations return (String, String); here it is changed to (String, Value), parsing the string once more.
    let (name, args_str) = match &tc.tool {
        Some(Tool::CallMcpTool(c)) => tools::mcp::serialize_outgoing_call(c, mcp_ctx),
        Some(Tool::ReadMcpResource(r)) => tools::mcp::serialize_outgoing_read_resource(r, mcp_ctx),
        Some(Tool::RunShellCommand(c)) => (
            "run_shell_command".to_owned(),
            json!({
                "command": c.command,
                "is_read_only": c.is_read_only,
                "uses_pager": c.uses_pager,
                "is_risky": c.is_risky,
            })
            .to_string(),
        ),
        Some(Tool::ReadFiles(r)) => {
            let files: Vec<Value> = r
                .files
                .iter()
                .map(|f| {
                    json!({
                        "path": f.name,
                        "line_ranges": f.line_ranges.iter().map(|lr| json!({
                            "start": lr.start, "end": lr.end
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            (
                "read_files".to_owned(),
                json!({ "files": files }).to_string(),
            )
        }
        Some(Tool::Grep(g)) => (
            "grep".to_owned(),
            json!({ "queries": g.queries, "path": g.path }).to_string(),
        ),
        Some(Tool::AskUserQuestion(a)) => {
            let questions: Vec<Value> = a
                .questions
                .iter()
                .map(|q| {
                    let (options, recommended_index, multi_select, supports_other) =
                        match &q.question_type {
                            Some(
                                api::ask_user_question::question::QuestionType::MultipleChoice(mc),
                            ) => (
                                mc.options
                                    .iter()
                                    .map(|o| o.label.clone())
                                    .collect::<Vec<_>>(),
                                mc.recommended_option_index,
                                mc.is_multiselect,
                                mc.supports_other,
                            ),
                            None => (vec![], 0, false, false),
                        };
                    json!({
                        "question": q.question,
                        "options": options,
                        "recommended_index": recommended_index,
                        "multi_select": multi_select,
                        "supports_other": supports_other,
                    })
                })
                .collect();
            (
                "ask_user_question".to_owned(),
                json!({ "questions": questions }).to_string(),
            )
        }
        Some(Tool::FileGlobV2(g)) => (
            "file_glob".to_owned(),
            json!({
                "patterns": g.patterns,
                "search_dir": g.search_dir,
                "limit": g.max_matches,
            })
            .to_string(),
        ),
        Some(Tool::ApplyFileDiffs(a)) => {
            let mut operations: Vec<Value> = Vec::new();
            for d in &a.diffs {
                operations.push(json!({
                    "op": "edit",
                    "file_path": d.file_path,
                    "search": d.search,
                    "replace": d.replace,
                }));
            }
            for f in &a.new_files {
                operations.push(json!({
                    "op": "create",
                    "file_path": f.file_path,
                    "content": f.content,
                }));
            }
            for f in &a.deleted_files {
                operations.push(json!({
                    "op": "delete",
                    "file_path": f.file_path,
                }));
            }
            (
                "apply_file_diffs".to_owned(),
                json!({ "summary": a.summary, "operations": operations }).to_string(),
            )
        }
        Some(Tool::WriteToLongRunningShellCommand(w)) => {
            use api::message::tool_call::write_to_long_running_shell_command::mode::Mode as M;
            let mode = match w.mode.as_ref().and_then(|m| m.mode.as_ref()) {
                Some(M::Raw(_)) => "raw",
                Some(M::Block(_)) => "block",
                _ => "line",
            };
            (
                "write_to_long_running_shell_command".to_owned(),
                json!({
                    "command_id": w.command_id,
                    "input": String::from_utf8_lossy(&w.input).to_string(),
                    "mode": mode,
                })
                .to_string(),
            )
        }
        Some(Tool::ReadDocuments(r)) => {
            let docs: Vec<Value> = r
                .documents
                .iter()
                .map(|d| {
                    json!({
                        "document_id": d.document_id,
                        "line_ranges": d.line_ranges.iter().map(|lr| json!({
                            "start": lr.start, "end": lr.end
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            (
                "read_documents".to_owned(),
                json!({ "documents": docs }).to_string(),
            )
        }
        Some(Tool::EditDocuments(e)) => {
            let diffs: Vec<Value> = e
                .diffs
                .iter()
                .map(|d| {
                    json!({
                        "document_id": d.document_id,
                        "search": d.search,
                        "replace": d.replace,
                    })
                })
                .collect();
            (
                "edit_documents".to_owned(),
                json!({ "diffs": diffs }).to_string(),
            )
        }
        Some(Tool::CreateDocuments(c)) => {
            let new_documents: Vec<Value> = c
                .new_documents
                .iter()
                .map(|d| json!({ "title": d.title, "content": d.content }))
                .collect();
            (
                "create_documents".to_owned(),
                json!({ "new_documents": new_documents }).to_string(),
            )
        }
        Some(Tool::SuggestNewConversation(s)) => (
            "suggest_new_conversation".to_owned(),
            json!({ "message_id": s.message_id }).to_string(),
        ),
        Some(Tool::SuggestPrompt(s)) => {
            use api::message::tool_call::suggest_prompt::DisplayMode;
            let (prompt, label) = match &s.display_mode {
                Some(DisplayMode::PromptChip(c)) => (c.prompt.clone(), c.label.clone()),
                Some(DisplayMode::InlineQueryBanner(b)) => (b.query.clone(), b.title.clone()),
                None => (String::new(), String::new()),
            };
            (
                "suggest_prompt".to_owned(),
                json!({ "prompt": prompt, "label": label }).to_string(),
            )
        }
        Some(Tool::OpenCodeReview(_)) => ("open_code_review".to_owned(), "{}".to_owned()),
        Some(Tool::TransferShellCommandControlToUser(t)) => (
            "transfer_shell_command_control_to_user".to_owned(),
            json!({ "reason": t.reason }).to_string(),
        ),
        Some(Tool::ReadSkill(r)) => {
            use api::message::tool_call::read_skill::SkillReference;
            // The model passed `name` last turn, which `from_args` put into the `SkillPath` slot;
            // when serializing back into the upstream conversation history, it is changed back to the `name` field, consistent with the current JSON schema.
            // BundledSkillId uses the Display form `@warp-skill:<id>`, consistent with SkillReference Display.
            let name = match &r.skill_reference {
                Some(SkillReference::SkillPath(s)) => s.clone(),
                Some(SkillReference::BundledSkillId(id)) => format!("@warp-skill:{id}"),
                None => String::new(),
            };
            ("read_skill".to_owned(), json!({ "name": name }).to_string())
        }
        Some(Tool::ReadShellCommandOutput(r)) => {
            use api::message::tool_call::read_shell_command_output::Delay;
            let delay_seconds = match &r.delay {
                Some(Delay::Duration(d)) => Some(d.seconds),
                Some(Delay::OnCompletion(_)) | None => None,
            };
            let mut args = json!({ "command_id": r.command_id });
            if let Some(s) = delay_seconds {
                args["delay_seconds"] = json!(s);
            }
            ("read_shell_command_output".to_owned(), args.to_string())
        }
        Some(other) => {
            let variant_name = format!("{other:?}")
                .split('(')
                .next()
                .unwrap_or("UnknownVariant")
                .to_owned();
            (format!("warp_internal_{}", variant_name), "{}".to_owned())
        }
        None => ("warp_internal_empty".to_owned(), "{}".to_owned()),
    };
    let args_value: Value =
        serde_json::from_str(&args_str).unwrap_or(Value::Object(Default::default()));
    (name, args_value)
}

// ---------------------------------------------------------------------------
// Tools array
// ---------------------------------------------------------------------------

/// Whether this turn's input contains `UserQueryMode::Plan` triggered by `/plan`.
///
/// Per-turn semantics: only check whether this turn's `params.input` carries the Plan marker. History task messages
/// currently persist (via `make_user_query_message`) with `..Default::default()` into the
/// upstream proto, **without a mode field**; so plan state does not automatically stick across turns, and each query the user
/// wants to keep read-only needs the `/plan ` prefix added again. This is an intentional MVP shape:
/// - lowest implementation cost (no proto schema change, no new session-level state machine)
/// - consistent with the "explicit enter/exit" semantics of Claude Code `EnterPlanMode`; here the
///   exit action is implied by "the next query without /plan"
fn is_plan_mode_turn(input: &[AIAgentInput]) -> bool {
    input.iter().any(|i| {
        matches!(
            i,
            AIAgentInput::UserQuery {
                user_query_mode: UserQueryMode::Plan,
                ..
            }
        )
    })
}

/// Names of write/execute built-in tools hard-filtered under Plan Mode.
///
/// Logic fallback: even if the model ignores the guidance in `partials/plan_mode.j2` it cannot trigger side effects;
/// a tool absent from the tool list cannot be called (the provider protocol layer rejects an unknown function outright).
///
/// **Write-type tools NOT blocked**: `create_documents` / `edit_documents`. They only touch the
/// Warp Drive local document store (AIDocumentModel), not the filesystem or running commands, and semantically
/// are exactly Plan Mode's output archiving action: the model settles the final plan as a Drive document,
/// which the user can later view / edit in the Drive UI or drag into a self-made PLAN folder for reuse.
///
/// Remaining read-only + Drive-write subset: `read_files / grep / file_glob_v2 /
/// read_shell_command_output / ask_user_question / read_skill / read_documents /
/// create_documents / edit_documents / webfetch / websearch / mcp/*`。
const PLAN_MODE_BLOCKED_TOOLS: &[&str] = &[
    "run_shell_command",
    "apply_file_diffs",
    "write_to_long_running_shell_command",
    "open_code_review",
    "transfer_shell_command_control_to_user",
    "suggest_prompt",
];

/// Lists the tool names that will actually be fed to the upstream model this turn (built-in REGISTRY + current MCP tools),
/// sharing the same gating as `build_tools_array` (LRC / `web_search_enabled` /
/// `suggest_new_conversation` / `plan_mode`). Injected by `prompt_renderer` into the
/// system prompt so the template renders dynamically from the actually available list, no longer hard-coding allow/deny lists.
pub fn available_tool_names(params: &RequestParams) -> Vec<String> {
    let is_lrc = params.lrc_command_id.is_some();
    let web_enabled = params.web_search_enabled;
    let plan_mode = is_plan_mode_turn(&params.input);
    let mut names: Vec<String> = tools::REGISTRY
        .iter()
        .filter(|t| {
            if is_lrc && t.name == "run_shell_command" {
                return false;
            }
            if !web_enabled
                && (t.name == tools::webfetch::TOOL_NAME || t.name == tools::websearch::TOOL_NAME)
            {
                return false;
            }
            if t.name == "suggest_new_conversation" {
                return false;
            }
            if plan_mode && PLAN_MODE_BLOCKED_TOOLS.contains(&t.name) {
                return false;
            }
            true
        })
        .map(|t| t.name.to_owned())
        .collect();
    if let Some(ctx) = params.mcp_context.as_ref() {
        for (name, _description, _parameters) in tools::mcp::build_mcp_tool_defs(ctx) {
            names.push(name);
        }
    }
    names
}

fn build_tools_array(params: &RequestParams) -> Vec<GenaiTool> {
    // Warp A2: in the LRC tag-in scenario drop `run_shell_command`, forcing the model to pick PTY-operation tools.
    //
    // With an alt-screen long command (nvim/htop) + the user in tag-in state, the **model's most common mistake** is
    // calling `run_shell_command` to run `taskkill nvim` / `Stop-Process nvim` (spawning a new process),
    // which has nothing to do with the currently running PTY and cannot kill the target. The **correct approach** is to
    // `write_to_long_running_shell_command(command_id, input=":q\n", mode=raw)`,
    // send instructions directly to the current PTY.
    //
    // Measured: system prompt guidance + RunningCommand context prefix are both not strong enough,
    // the model still tends toward run_shell_command (simpler). The cleanest hard constraint is to remove the tool from the tools list
    // directly, so the model can only choose among PTY-operation tools.
    //
    // Other tools are kept (read_files/grep/ask_user_question, etc.), allowing the model to do necessary
    // information gathering and clarifying questions.
    let is_lrc = params.lrc_command_id.is_some();
    let web_enabled = params.web_search_enabled;
    let plan_mode = is_plan_mode_turn(&params.input);
    // Warp BYOP: the `suggest_prompt` chip UI has been restored via view-layer subscription to
    // PromptSuggestionExecutorEvent (see `terminal/view.rs::
    // handle_suggest_prompt_executor_event`), so it can be exposed to the model.
    // `suggest_new_conversation` is still filtered: UX has no ready popup component, and the executor was changed to
    // fast-fail Cancelled (see `action_model/execute/suggest_new_conversation.rs`);
    // the filter is redundant defense to avoid invalid-call noise.
    // Dynamic placeholder replacement: some tool descriptions contain `{{year}}` (e.g. websearch, aligned with opencode
    // websearch.ts:30-32's description getter), replaced with the current year at build time.
    // The description the model sees each time carries the correct year, not polluted by old years in the training data.
    let current_year = chrono::Local::now().format("%Y").to_string();
    let mut out: Vec<GenaiTool> = tools::REGISTRY
        .iter()
        .filter(|t| {
            if is_lrc && t.name == "run_shell_command" {
                return false;
            }
            // BYOP web tools are gated by profile.web_search_enabled (when the user has turned off the privacy
            // switch they are not exposed to the upstream model, avoiding accidental external network requests).
            if !web_enabled
                && (t.name == tools::webfetch::TOOL_NAME || t.name == tools::websearch::TOOL_NAME)
            {
                return false;
            }
            // suggest_new_conversation: no UI implementation; the executor in Warp is changed to
            // fast-fail Cancelled. Filtered out here to avoid pointless
            // tool_call→cancelled round trips by the model (pure token waste).
            if t.name == "suggest_new_conversation" {
                return false;
            }
            // Plan Mode: a read-only-mode hard guardrail triggered by `/plan`, removing write/execute tools.
            // Double insurance with the system prompt's plan_mode.j2 guidance: even if the model ignores
            // the prompt, with the tools not in the list it cannot trigger side effects (the provider protocol layer
            // rejects an unknown function outright).
            if plan_mode && PLAN_MODE_BLOCKED_TOOLS.contains(&t.name) {
                return false;
            }
            true
        })
        .map(|t| {
            let description = if t.description.contains("{{year}}") {
                t.description.replace("{{year}}", &current_year)
            } else {
                t.description.to_owned()
            };
            GenaiTool::new(t.name)
                .with_description(description)
                .with_schema((t.parameters)())
        })
        .collect();

    if let Some(ctx) = params.mcp_context.as_ref() {
        for (name, description, parameters) in tools::mcp::build_mcp_tool_defs(ctx) {
            out.push(
                GenaiTool::new(name)
                    .with_description(description)
                    .with_schema(parameters),
            );
        }
    }
    if is_lrc {
        log::info!(
            "[byop] LRC tag-in: tools array filtered (removed run_shell_command), \
             total tools={}",
            out.len()
        );
    }
    if plan_mode {
        log::info!(
            "[byop] Plan Mode: tools array filtered (removed write/exec tools: {:?}), \
             total tools={}",
            PLAN_MODE_BLOCKED_TOOLS,
            out.len()
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Client / routing
// ---------------------------------------------------------------------------

/// Maps `AgentProviderApiType` one-to-one to genai `AdapterKind`.
fn adapter_kind_for(api_type: AgentProviderApiType) -> AdapterKind {
    match api_type {
        AgentProviderApiType::OpenAi => AdapterKind::OpenAI,
        AgentProviderApiType::OpenAiResp => AdapterKind::OpenAIResp,
        AgentProviderApiType::Gemini => AdapterKind::Gemini,
        AgentProviderApiType::Anthropic => AdapterKind::Anthropic,
        AgentProviderApiType::Ollama => AdapterKind::Ollama,
        AgentProviderApiType::DeepSeek => AdapterKind::DeepSeek,
    }
}

/// Official OpenAI gpt-5 / codex families, with function tools + reasoning_effort,
/// only accept `/v1/responses`; `/v1/chat/completions` returns 400.
fn openai_model_requires_responses_api(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    id.starts_with("gpt-5") || id.starts_with("codex")
}

/// Claude model id heuristic (tolerates namespace prefixes such as `anthropic/claude-sonnet-4-6`).
fn is_claude_model(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    let bare = id.rsplit('/').next().unwrap_or(&id);
    bare.contains("claude-") || bare.starts_with("claude")
}

/// Whether `base_url` points to the official Anthropic Messages API host.
fn is_anthropic_api_host(base_url: &str) -> bool {
    url::Url::parse(base_url.trim())
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "api.anthropic.com" || host.ends_with(".anthropic.com"))
}

/// Whether `base_url` points to the official OpenAI API host. Auto-upgrade to `/v1/responses` only takes effect for the official
/// host: chat-completions-only relays such as one-api / LiteLLM often keep official model ids,
/// and forcing the Responses API would 404.
fn is_openai_api_host(base_url: &str) -> bool {
    url::Url::parse(base_url.trim())
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "api.openai.com" || host.ends_with(".openai.com"))
}

/// Resolves the genai adapter actually to use from the user-configured `api_type`, model id and `base_url`.
///
/// - `OpenAi` + official host + gpt-5.4 etc. → `OpenAIResp` (`/v1/responses`)
/// - `OpenAi` + `api.anthropic.com` + claude model → `Anthropic` (`/v1/messages`)
fn effective_adapter_kind_for(
    api_type: AgentProviderApiType,
    model_id: &str,
    base_url: &str,
) -> AdapterKind {
    let base = adapter_kind_for(api_type);
    if api_type == AgentProviderApiType::OpenAi {
        if is_anthropic_api_host(base_url) && is_claude_model(model_id) {
            return AdapterKind::Anthropic;
        }
        if is_openai_api_host(base_url) && openai_model_requires_responses_api(model_id) {
            return AdapterKind::OpenAIResp;
        }
    }
    base
}

/// Normalizes the user-entered `base_url`, producing the endpoint URL from which genai adapters join the service path.
///
/// All genai 0.6.x adapters assume the endpoint ends with `/` and already includes the version path segment:
/// - Anthropic: `format!("{base_url}messages")` expects `…/v1/`
/// - Gemini: `format!("{base_url}models/{m}:streamGenerateContent")` expects `…/v1beta/`
/// - OpenAI / OpenAIResp / DeepSeek: `Url::join("chat/completions" or "responses")` expects `…/v1/`
/// - Ollama: `format!("{base_url}api/chat")` expects the root path `…/`
///
/// The three ways users actually fill it in:
/// 1. Bare host (`https://ai.zerx.dev`): the early default behavior of only appending a trailing `/` would join into `https://ai.zerx.dev/messages`,
///    missing `/v1/` and causing 404. **Here the default version path segment is appended automatically per api_type** (Anthropic/OpenAI families→`/v1/`,
///    Gemini→`/v1beta/`, Ollama not appended).
/// 2. Full path with version (`https://ai.zerx.dev/v1`): only append the trailing `/`, leave the path alone.
/// 3. Left empty: use [`AgentProviderApiType::default_base_url`].
///
/// Extra Anthropic tolerance: if the user pastes the full endpoint (`…/v1/messages`), strip the trailing
/// `messages` segment, so genai does not append it again giving `…/messages/messages`.
fn normalize_endpoint_url(api_type: AgentProviderApiType, base_url: &str) -> String {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return api_type.default_base_url().to_owned();
    }

    // Parse failure (the user entered a malformed URL): degrade to the original "append trailing /" behavior, letting upstream report the error instead of panicking here.
    let mut parsed = match url::Url::parse(trimmed) {
        Ok(u) => u,
        Err(_) => {
            let stripped = trimmed.trim_end_matches('/');
            return format!("{stripped}/");
        }
    };

    if api_type == AgentProviderApiType::Anthropic || is_anthropic_api_host(trimmed) {
        let path = parsed.path().trim_end_matches('/');
        if let Some(prefix) = path.strip_suffix("/messages") {
            let new_path = if prefix.is_empty() { "/" } else { prefix };
            parsed.set_path(&format!("{new_path}/"));
        }
    }

    // The native Ollama API is at host root `/api/chat` with no `/v1/` prefix. The historical default_base_url
    // wrongly filled `/v1/`, or a `/v1` the user carried over from OpenAI habits, is stripped here to avoid joining into `/v1/api/chat`.
    if api_type == AgentProviderApiType::Ollama {
        let path = parsed.path().trim_end_matches('/');
        if path.is_empty() || path == "/" || path == "/v1" {
            let mut normalized = parsed.clone();
            normalized.set_path("/");
            return normalized.to_string();
        }
    }

    // path == "/" or empty: the user only entered a host; automatically append the api_type default version path segment.
    if parsed.path() == "/" || parsed.path().is_empty() {
        // Extract the path part from default_base_url (e.g. "/v1/" / "/v1beta/" / "/").
        let default_path = url::Url::parse(api_type.default_base_url())
            .ok()
            .map(|u| u.path().to_owned())
            .unwrap_or_else(|| "/".to_owned());
        let host_part = trimmed.trim_end_matches('/');
        return format!("{host_part}{default_path}");
    }

    // The user already supplies a path: only ensure the trailing `/` (genai format!/Url::join both depend on it).
    let normalized = parsed.to_string();
    let stripped = normalized.trim_end_matches('/');
    format!("{stripped}/")
}

/// Builds the genai Client. Created anew per request (low overhead; Client is internally just a reqwest::Client + adapter table).
/// `ServiceTargetResolver` captures the current request's endpoint/key/api_type, then forces every exec_chat_stream
/// to route to the specified AdapterKind, fully bypassing genai's default "detect by model name".
pub(super) fn build_client(
    api_type: AgentProviderApiType,
    base_url: &str,
    api_key: String,
) -> Client {
    let endpoint_url = normalize_endpoint_url(api_type, base_url);
    log::info!("[byop] build_client: api_type={api_type:?} endpoint_url={endpoint_url}");
    let key_for_resolver = api_key.clone();
    let resolver = ServiceTargetResolver::from_resolver_fn(
        move |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
            let ServiceTarget { model, .. } = service_target;
            let endpoint = Endpoint::from_owned(endpoint_url.clone());
            let auth = AuthData::from_single(key_for_resolver.clone());
            let model_name = model.model_name.as_str();
            let adapter_kind = effective_adapter_kind_for(api_type, model_name, &endpoint_url);
            if api_type == AgentProviderApiType::OpenAi && adapter_kind == AdapterKind::OpenAIResp {
                log::info!(
                    "[byop] auto-upgrade OpenAi → OpenAIResp for model={model_name} \
                     (function tools + reasoning_effort require /v1/responses)"
                );
            }
            if api_type == AgentProviderApiType::OpenAi && adapter_kind == AdapterKind::Anthropic {
                log::info!(
                    "[byop] auto-upgrade OpenAi → Anthropic for model={model_name} \
                     (official Anthropic host requires /v1/messages)"
                );
            }
            // Override genai's "by model name" detection result with our specified AdapterKind,
            // but keep model_name so the upstream service addresses the model correctly.
            let model = ModelIden::new(adapter_kind, model.model_name);
            Ok(ServiceTarget {
                endpoint,
                auth,
                model,
            })
        },
    );

    // Warp BYOP: SSE streams must not use gzip. `Accept-Encoding: gzip` makes proxies like nginx
    // compress the response, and the server must flush a complete deflate frame before the client can decode
    // plaintext, which breaks streaming semantics into ~K-byte bursts, felt as a stall every few hundred ms. zed/opencode
    // use native fetch / std HTTP, which do not proactively negotiate gzip on SSE, so the same proxies have no problem.
    //
    // Here we explicitly construct `WebConfig` even though genai default already has `gzip=false` (fork modification).
    //
    // User-Agent is dynamically bound to the current app name (from `ChannelState::app_id().application_name()`,
    // registered by the entry bin: `bin/oss.rs` → "Warp"; other channels bring their own names).
    // This way upstream services can tell which branch build a request comes from, and it follows automatically if renamed later.
    let mut headers = reqwest::header::HeaderMap::new();
    let ua = if is_opencode_host(base_url) {
        // OpenCode Go requires a descriptive UA of the form `warp/<version>`.
        let ua = match warp_core::channel::ChannelState::app_version() {
            Some(v) if !v.is_empty() => format!("warp/{v}"),
            _ => "warp".to_owned(),
        };
        reqwest::header::HeaderValue::from_str(&ua)
    } else {
        build_user_agent_header()
    };
    if let Ok(value) = ua {
        headers.insert(reqwest::header::USER_AGENT, value);
    }
    let mut web_config = WebConfig {
        gzip: false,
        default_headers: Some(headers),
        ..WebConfig::default()
    };
    let proxy_cfg = current_proxy_config();
    match proxy_cfg.mode {
        http_client::ProxyMode::Off => {
            // Fully disable proxies (including system proxy and environment variables), aligned with the Off behavior of
            // http_client::Client::new().
            web_config.no_proxy = true;
        }
        http_client::ProxyMode::System => {
            // Follow the system/environment proxy (reqwest default behavior).
            // WebConfig sets no proxy → reqwest reads macOS SystemConfiguration
            // or the environment variables HTTP_PROXY / HTTPS_PROXY.
        }
        http_client::ProxyMode::Custom => {
            if !proxy_cfg.url.is_empty() {
                if let Err(err) = web_config.set_proxy_settings(
                    &proxy_cfg.url,
                    &proxy_cfg.username,
                    &proxy_cfg.password,
                    &proxy_cfg.no_proxy,
                ) {
                    log::warn!(
                        "[byop] proxy URL '{}' invalid, skipping proxy configuration: {err}",
                        proxy_cfg.url
                    );
                }
            }
        }
    }
    Client::builder()
        .with_web_config(web_config)
        .with_service_target_resolver(resolver)
        .build()
}

/// Builds the `User-Agent` header for BYOP outbound requests, with values like:
/// - `Warp/<git-tag>`: when a release build has `GIT_RELEASE_TAG` injected
/// - `Warp`: Dev / local builds without a version
///
/// The app name is always taken from `ChannelState::app_id().application_name()`, ensuring consistency with the `AppId`
/// registered by the entry bin (`bin/oss.rs` registers "Warp").
fn build_user_agent_header(
) -> Result<reqwest::header::HeaderValue, reqwest::header::InvalidHeaderValue> {
    let app_name = warp_core::channel::ChannelState::app_id()
        .application_name()
        .to_owned();
    let ua = match warp_core::channel::ChannelState::app_version() {
        Some(v) if !v.is_empty() => format!("{app_name}/{v}"),
        _ => app_name,
    };
    reqwest::header::HeaderValue::from_str(&ua)
}

/// Decides whether to inject `enable_thinking: true` for DashScope (Alibaba Cloud Bailian, OpenAI-compatible path).
///
/// Aligned with opencode `transform.ts:931-938` (comment at provider/transform.ts L926+):
/// "DashScope does not enable thinking by default; reasoning models such as qwen3 / qwq / deepseek-r1 / kimi-k2.5 / qwen-plus
/// must explicitly set `enable_thinking: true` to output reasoning_content".
///
/// Match conditions (all must hold):
/// 1. `api_type == OpenAi` (DashScope uses the OpenAI-compatible path)
/// 2. `effort_setting != Off` (when the user actively turns thinking off, respect it and do not inject)
/// 3. base_url contains `dashscope.aliyuncs.com` / `dashscope.cn` / `dashscope-intl.aliyuncs.com`
/// 4. model_id does not contain `kimi-k2-thinking` (opencode excludes it; that model thinks by default)
/// 5. model_id matches the reasoning substring whitelist: `qwen3` / `qwq` / `deepseek-r1` / `kimi-k2.5` /
///    `kimi-k2-` / `qwen-plus` (avoids stuffing it into pure chat models like qwen-turbo / qwen2.5)
fn dashscope_needs_enable_thinking(
    api_type: AgentProviderApiType,
    base_url: &str,
    model_id: &str,
    effort_setting: crate::settings::ReasoningEffortSetting,
) -> bool {
    if !matches!(api_type, AgentProviderApiType::OpenAi) {
        return false;
    }
    if matches!(effort_setting, crate::settings::ReasoningEffortSetting::Off) {
        return false;
    }
    let url = base_url.to_ascii_lowercase();
    let is_dashscope = url.contains("dashscope.aliyuncs.com")
        || url.contains("dashscope.cn")
        || url.contains("dashscope-intl.aliyuncs.com");
    if !is_dashscope {
        return false;
    }
    let id = model_id.to_ascii_lowercase();
    if id.contains("kimi-k2-thinking") {
        return false;
    }
    id.contains("qwen3")
        || id.contains("qwq")
        || id.contains("deepseek-r1")
        || id.contains("kimi-k2.5")
        || id.contains("kimi-k2-")
        || id.contains("qwen-plus")
}

/// Infers the upstream provider from base_url, used **only** to decide whether to send `prompt_cache_key`.
///
/// Aligned with opencode `packages/opencode/src/provider/transform.ts`'s
/// `options()` function: opencode decides by the `providerID` dimension whether to set
/// `promptCacheKey`, and only the five providers `openai` / `azure` / `openrouter` / `venice` /
/// `opencode*` send it. All other providers (including all OpenAI-compatible
/// relays / local services / most domestic clouds) never send it.
///
/// Warp has no `providerID` dimension, only the user's free-form `base_url`.
/// So we infer from `base_url`:
/// - `api.openai.com`           → "openai"
/// - `*.openai.azure.com`       → "azure"
/// - `openrouter.ai/api`        → "openrouter"
/// - `api.venice.ai/api`        → "venice"
/// - `opencode.ai/zen`          → "opencode"
///
/// Everything else returns `None` (equivalent to matching no branch in opencode).
///
/// Data source: the `api` field of the provider table at [models.dev](https://models.dev/api.json).
/// openai / azure / venice have no `api` in models.dev (baked in by the SDK),
/// so they are hard-coded here by each SDK's default endpoint.
///
/// Deliberately covers only these 5 \u{2014} this is opencode's real whitelist; do not expand it on your own because "it looks like
/// other providers also support compatible caching". OpenRouter documents native support for
/// `prompt_cache_key` (snake_case); the other four go through the OpenAI Chat Completions path.
fn opencode_compatible_cache_provider(base_url: &str) -> bool {
    let u = base_url.to_ascii_lowercase();
    u.contains("api.openai.com")
        || u.contains(".openai.azure.com")
        || u.contains("openrouter.ai/api")
        || u.contains("api.venice.ai/api")
        || u.contains("opencode.ai/zen")
}

/// Model prefixes supported by OpenAI 24h Extended Cache (`prompt_cache_retention="24h"`).
/// Judged by literal prefix only; unrecognized models are not sent, avoiding 400 rejection by strictly validating upstreams.
fn openai_supports_extended_cache(model_id: &str) -> bool {
    const PREFIXES: &[&str] = &["gpt-5", "gpt-4.1", "o1", "o3", "o4"];
    let m = model_id.to_ascii_lowercase();
    PREFIXES
        .iter()
        .any(|p| m.starts_with(p) || m.contains(&format!("/{p}")))
}

fn build_chat_options(
    api_type: AgentProviderApiType,
    base_url: &str,
    model_id: &str,
    effort_setting: crate::settings::ReasoningEffortSetting,
    mut extra_headers: Vec<(String, String)>,
    conversation_id: Option<&str>,
) -> ChatOptions {
    let mut opts = ChatOptions::default()
        .with_capture_content(true)
        .with_capture_tool_calls(true)
        .with_capture_reasoning_content(true)
        .with_capture_usage(true)
        // Have genai extract the <think>...</think>
        // segments that DeepSeek-style models interleave in content into reasoning chunks, for a cleaner UI display. Only takes effect for adapters supporting this format.
        .with_normalize_reasoning_content(true);

    // Prompt caching (1:1 aligned with opencode `packages/opencode/src/provider/transform.ts`
    // 's `options()` function). Key points:
    //
    // 1. **`prompt_cache_retention` (genai `ChatOptions::cache_control`) is sent as "24h" only for official
    //    api.openai.com + models supporting 24h (GPT-5 / 4.1 / o-series), and not otherwise**.
    //    opencode does not use this field anywhere in its repo; strictly schema-validating BYOP relays
    //    (OpenCode Go / vLLM / lm-studio / most domestic proxies) reject it with HTTP 400
    //    `Extra inputs are not permitted, field: 'prompt_cache_retention'`
    //    (issue #126). Even official OpenAI has 5min implicit caching by default, so no explicit declaration is needed.
    //
    // 2. **`prompt_cache_key` is sent only for the 5 providers opencode knows**:
    //    `openai` / `azure` / `openrouter` / `venice` / `opencode`. All other providers
    //    (including all OpenAI-compatible relays / domestic clouds / local services) never send it.
    //
    //    opencode decides by `providerID` (the string chosen in the user's config); Warp has no
    //    `providerID`, so it can only infer from `base_url`; see `opencode_compatible_cache_provider`.
    //    This is the only semantic use of base_url; do not extend it to decide behavior beyond caching.
    //
    // 3. Anthropic uses per-message cache_control (in `build_chat_request`),
    //    not here.
    // 4. DeepSeek / Gemini / Ollama use implicit server-side caching; skip.
    if matches!(
        api_type,
        AgentProviderApiType::OpenAi | AgentProviderApiType::OpenAiResp
    ) && opencode_compatible_cache_provider(base_url)
    {
        if let Some(cid) = conversation_id {
            if !cid.is_empty() {
                opts = opts.with_prompt_cache_key(cid.to_owned());
            }
        }
        // 24h retention is only sent to the official api.openai.com (other whitelisted/relay hosts may 400, see #126).
        if base_url.to_ascii_lowercase().contains("api.openai.com")
            && openai_supports_extended_cache(model_id)
        {
            opts = opts.with_cache_control(CacheControl::Ephemeral24h);
        }
    }

    // **Thinking-depth tier dispatch** (aligned with Zed `LanguageModelRequest::thinking_allowed` handling for each
    // provider: when `thinking_allowed=false` no provider sends any thinking
    // field; Anthropic / Google / Bedrock servers default to thinking off).
    //
    // - **Auto**: send nothing, let genai fall back to "infer from model-name suffix" (inside the OpenAI/Anthropic adapters).
    // - **Off + Anthropic / Gemini**: **skip `with_reasoning_effort` entirely**, equivalent to
    //   Auto + a model name without a thinking suffix. The genai adapter takes the `(model, None)` inference branch,
    //   does not call `insert_anthropic_reasoning` / `thinkingConfig`, and sends no thinking field.
    //   ★ This happens to bypass the bug where vendor genai `claude-opus-4-6` / `claude-sonnet-4-6`
    //   `support_adaptive` forcibly injects `thinking:{type:adaptive}`
    //   (`lib/rust-genai/src/adapter/adapters/anthropic/adapter_impl.rs:121-135`
    //   does not read whether effort is `None`).
    // - **Off + DeepSeek**: the server's `thinking_mode` is on by default (deepseek-v4-flash etc.),
    //   and needs an explicit `extra_body.thinking.type=disabled` to turn off. Warp's local fork
    //   of genai already supports top-level merge of `ChatOptions::extra_body`.
    // - **Off + OpenAI / OpenAiResp**: take the `reasoning_effort: "none"` path
    //   (GPT-5 / codex accept the `none` tier; o-series are filtered by the capability table).
    // - **Non-Off + model does not support reasoning**: skip, to avoid injecting thinking params into old models like claude-3-5-haiku / gpt-4o /
    //   gemini-1.5-pro and being rejected upstream with 400.
    use crate::settings::ReasoningEffortSetting as RE;
    match (api_type, effort_setting) {
        // Auto: no parameters sent
        (_, RE::Auto) => {}

        // Anthropic + Off: no thinking field sent
        (AgentProviderApiType::Anthropic, RE::Off) => {
            log::info!(
                "[byop] Anthropic Off → skip reasoning_effort (model={model_id}); \
                 no thinking field sent"
            );
        }

        // Gemini + Off: no thinkingConfig sent
        (AgentProviderApiType::Gemini, RE::Off) => {
            log::info!(
                "[byop] Gemini Off → skip reasoning_effort (model={model_id}); \
                 no thinkingConfig sent"
            );
        }

        // DeepSeek + Off: explicit disabled
        (AgentProviderApiType::DeepSeek, RE::Off) => {
            log::info!(
                "[byop] DeepSeek Off → extra_body thinking.type=disabled (model={model_id})"
            );
            opts = opts.with_extra_body(json!({"thinking": {"type": "disabled"}}));
        }

        // Others (OpenAI / OpenAiResp / Ollama / non-Off tiers of every provider):
        // go through the reasoning_effort injection path filtered by the capability table
        _ => {
            if let Some(effort) = effort_setting.to_genai() {
                if super::reasoning::model_supports_reasoning(api_type, model_id) {
                    log::info!(
                        "[byop] reasoning_effort injected: model={model_id} setting={effort_setting:?}"
                    );
                    opts = opts.with_reasoning_effort(effort);
                } else {
                    log::info!(
                        "[byop] reasoning_effort SKIPPED: model={model_id} not in capability list \
                         (api_type={api_type:?} setting={effort_setting:?}); request sent without thinking params"
                    );
                }
            }
        }
    }

    // DashScope (Alibaba Cloud Bailian) OpenAI-compatible path needs an explicit `enable_thinking: true` to
    // output reasoning. See the `dashscope_needs_enable_thinking` comment.
    // Mutually exclusive with the extra_body of DeepSeek Off above (DeepSeek uses the DeepSeek api_type,
    // DashScope uses the OpenAI api_type), so they never fire together.
    if dashscope_needs_enable_thinking(api_type, base_url, model_id, effort_setting) {
        log::info!(
            "[byop] DashScope reasoning model → extra_body enable_thinking=true \
             (model={model_id} setting={effort_setting:?})"
        );
        opts = opts.with_extra_body(json!({"enable_thinking": true}));
    }
    add_opencode_session_header(&mut extra_headers, base_url, conversation_id);
    if !extra_headers.is_empty() {
        opts = opts.with_extra_headers(extra_headers);
    }

    opts
}

/// `opencode.ai` or its subdomains (exact host match, no substring match).
pub(super) fn is_opencode_host(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|h| h == "opencode.ai" || h.ends_with(".opencode.ai"))
}

/// OpenCode Go requires a stable per-conversation `x-opencode-session` header.
/// Appended only when the host is opencode, the conversation id is non-empty, and the user has not set a header of the same name (case-insensitive).
pub(super) fn add_opencode_session_header(
    headers: &mut Vec<(String, String)>,
    base_url: &str,
    conversation_id: Option<&str>,
) {
    let Some(cid) = conversation_id.filter(|c| !c.is_empty()) else {
        return;
    };
    if is_opencode_host(base_url)
        && !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("x-opencode-session"))
    {
        headers.push(("x-opencode-session".to_owned(), cid.to_owned()));
    }
}

fn map_genai_error(err: genai::Error) -> OpenAiCompatibleError {
    use genai::Error as G;
    match err {
        // True parse failure: the JSON deserialization stage
        G::StreamParse { .. }
        | G::SerdeJson(_)
        | G::JsonValueExt(_)
        | G::InvalidJsonResponseElement { .. } => OpenAiCompatibleError::Decode(format!("{err}")),

        // Network/stream-send stage failure (reqwest connect, TLS, DNS, timeout, stream interruption, etc.)
        G::WebStream { .. } | G::WebAdapterCall { .. } | G::WebModelCall { .. } => {
            OpenAiCompatibleError::Stream(format!("{err}"))
        }

        // HTTP error status returned by the server
        G::HttpError {
            status,
            body,
            canonical_reason,
        } => OpenAiCompatibleError::Status {
            status: status.as_u16(),
            body: if canonical_reason.is_empty() {
                body
            } else {
                format!("{canonical_reason}: {body}")
            },
        },

        // The rest (request construction, auth, unsupported capability, etc.) go to the generic error, avoiding misleading it as "parse failure"
        other => OpenAiCompatibleError::Other(format!("{other}")),
    }
}

// ---------------------------------------------------------------------------
// Main flow
// ---------------------------------------------------------------------------

/// BYOP config needed for title generation. May or may not be the same provider as the main request (the user chose title_model independently in the Profile
/// Editor).
pub struct TitleGenInput {
    pub base_url: String,
    pub api_key: String,
    pub model_id: String,
    pub api_type: AgentProviderApiType,
    pub reasoning_effort: crate::settings::ReasoningEffortSetting,
    /// Extra headers of the provider; `{conversationId}` in values is resolved using `conversation_id` below.
    pub extra_headers: Vec<(String, String)>,
    /// The id of the conversation being titled; `None` if there is no conversation.
    pub conversation_id: Option<AIConversationId>,
}

pub struct ByopOutputInput {
    pub params: RequestParams,
    pub base_url: String,
    pub api_key: String,
    pub model_id: String,
    pub api_type: AgentProviderApiType,
    pub reasoning_effort: crate::settings::ReasoningEffortSetting,
    pub extra_headers: Vec<(String, String)>,
    /// Local conversation id used to resolve `{conversationId}` in headers; unrelated to the server conversation token.
    pub conversation_id: Option<AIConversationId>,
    pub task_id: String,
    pub target_task_id: String,
    pub needs_create_task: bool,
    pub lrc_command_id: Option<String>,
    pub lrc_should_spawn_subagent: bool,
    pub context_window: Option<u32>,
    pub cancellation_rx: futures::channel::oneshot::Receiver<()>,
    /// Attachment caps with user settings applied (three-state Override for image/pdf/audio).
    /// Computed by `resolve_for_model`, so UI display and runtime behavior stay consistent.
    pub attachment_caps: attachment_caps::AttachmentCaps,
}

/// `task_id`: the conversation's root task id (taken from the history model on the controller side).
/// `target_task_id`: the task id this turn's model output should be written to; for a normal conversation equal to root,
/// for follow-up turns of a CLI subagent it is the existing subtask.
/// `needs_create_task`: only the first turn (root still Optimistic) needs to emit `CreateTask`.
pub async fn generate_byop_output(
    input: ByopOutputInput,
) -> Result<ResponseStream, ConvertToAPITypeError> {
    let ByopOutputInput {
        params,
        base_url,
        api_key,
        model_id,
        api_type,
        reasoning_effort,
        extra_headers,
        conversation_id,
        task_id,
        target_task_id,
        needs_create_task,
        lrc_command_id,
        lrc_should_spawn_subagent,
        context_window,
        cancellation_rx: _cancellation_rx,
        attachment_caps,
    } = input;

    let force_echo_reasoning = super::reasoning::model_requires_reasoning_echo(api_type, &model_id);
    // Activate streaming extraction only for known models that interleave reasoning in <think> tags (e.g. MiniMax M3).
    // Other models keep the original Chunk output behavior, avoiding swallowing normal text that contains a literal <think>.
    let use_think_extraction = super::reasoning::model_uses_think_tags_in_content(&model_id);
    // Warp: request shaping (Anthropic cache_control / system-in-messages layout) must follow the
    // adapter actually on the wire rather than the user-configured api_type: OpenAi + official Anthropic host + claude
    // models are routed by effective_adapter_kind_for to the Anthropic adapter; if OpenAi shaping were still used
    // all prompt cache breakpoints would be lost, and agentic multi-turn cost would rise significantly. For other
    // api_types such as Ollama, the effective adapter equals the original adapter, shaping_api_type == api_type.
    let shaping_api_type =
        if effective_adapter_kind_for(api_type, &model_id, &base_url) == AdapterKind::Anthropic {
            AgentProviderApiType::Anthropic
        } else {
            api_type
        };
    let chat_req = build_chat_request(
        &params,
        force_echo_reasoning,
        shaping_api_type,
        attachment_caps,
    )?;
    let server_conversation_token = params
        .conversation_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_default();
    let extra_headers = super::header_vars::resolve_headers(
        &extra_headers,
        &HeaderVars::default().with_conversation_id(conversation_id),
    );
    let chat_opts = build_chat_options(
        api_type,
        &base_url,
        &model_id,
        reasoning_effort,
        extra_headers,
        if server_conversation_token.is_empty() {
            None
        } else {
            Some(server_conversation_token.as_str())
        },
    );
    let client = build_client(api_type, &base_url, api_key);
    let request_id = Uuid::new_v4().to_string();
    let mcp_context = params.mcp_context.clone();
    let tool_names_for_extract = available_tool_names(&params);

    // ⚠️ BYOP persistence key point: on warp's own path, the following ClientActions are all emitted by the server
    // to let the client write non-model-produced messages such as UserQuery / ToolCallResult
    // back into task.messages, so the next request's `params.tasks` snapshot is complete.
    //
    // BYOP is de-clouded and client-managed, with no server side, so we must emit these write-back events ourselves,
    // otherwise the next turn's `compute_active_tasks` sees only model output (reasoning/output/tool_call),
    // missing the corresponding user_query and tool_call_result, and the model context is severely broken.
    //
    // Here, after the stream starts, the current turn's UserQuery / ToolCallResult are written in the original order of `params.input`.
    // When the user interrupts a pending tool and continues typing, the controller passes ActionResult → UserQuery;
    // persistence must not be split into "all UserQuery first, then all ActionResult", otherwise history becomes
    // Assistant(tool_call) → UserQuery → ToolCallResult。
    //
    // The emit must come after CreateTask (the task has been upgraded to Server state),
    // and before the model response begins (UI order: user shown → thinking/answer).
    // Warp: keep multimodal alive across history turns. Besides the query text, all multimodal binaries
    // (image / pdf / audio / ...) in the current turn's UserQuery.context are packed together into `UserQuery.context.images`
    // and persisted (the proto field is named images but is semantically a generic BinaryFile: `bytes data + mime_type`,
    // equivalent to opencode FilePart), so that when build_chat_request rebuilds messages next turn it can restore binaries from history
    // messages and keep injecting them upstream as ContentPart::Binary (mimes unsupported by the model are replaced by
    // build_user_message_with_binaries with ERROR text, consistent with opencode unsupportedParts).
    // Warp's own path does not need this step because the cloud server holds InputContext; BYOP direct connection must manage it client-side.
    let pending_user_queries: Vec<(String, Vec<user_context::UserBinary>)> = params
        .input
        .iter()
        .filter_map(|i| match i {
            AIAgentInput::UserQuery { query, context, .. } => {
                let attachments = user_context::collect_user_attachments(context);
                Some((query.clone(), attachments.binaries))
            }
            _ => None,
        })
        .collect();
    // One INFO-level summary line + one brief line per message (role + text length + tool count + reasoning flag),
    // visible with the default log config, convenient for diagnosing issues like "was the history fully sent".
    //
    // Note: on the Anthropic path, `build_chat_request` pushes the system text as `ChatMessage::system`
    // to messages[0] so it can carry `cache_control`, so `chat_req.system` will be None and `system_len`
    // shows 0; the real system content is still in messages[0] (see the per-message reports below). To avoid
    // misleading the diagnoser, a `system_in_messages_head` hint is added here.
    log_chat_request_details(&chat_req, &model_id, shaping_api_type, &base_url);

    // Diagnostics: build a complete ChatRequest JSON dump containing system / messages / tools, saved to
    // the stream closure. The real Anthropic wire body is converted once more by the genai adapter, but this already
    // covers all raw strings passed into BYOP, enough to locate whether an illegal escape comes from the prompt, tool description,
    // schema or tool result.
    let diag_body_json = serde_json::to_string(&json!({
        "model": &model_id,
        "chat_request": &chat_req,
    }))
    .unwrap_or_default();
    log::info!("[byop] diag_body_approx_len={}", diag_body_json.len());
    log::info!("[byop-diag] full_request_json={diag_body_json}");

    // Proactively scan the raw text for "suspicious backslash sequences": serde_json serializes a literal
    // `\` in the source string as `\\`, so only "two consecutive backslashes + u/x" in the wire body means
    // the original text has a literal `\u` / `\x`, which is the real risk point where a proxy wrongly "restores `\\u` → `\u`"
    // and triggers an invalid escape. `\n` / `\r` / `\t` in the source string are output by serde_json as a single backslash +
    // letter, which is already a legal JSON escape; the proxy will not restore it a second time, so it is not suspicious.
    fn scan_suspicious_backslash(label: &str, s: &str) {
        let bytes = s.as_bytes();
        let mut bs_hits: Vec<(usize, String)> = Vec::new();
        let mut ctrl_hits: Vec<(usize, u8)> = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            // Literal `\\u` / `\\x` sequences (the source string contains `\u` / `\x`).
            if b == b'\\'
                && i + 2 < bytes.len()
                && bytes[i + 1] == b'\\'
                && matches!(bytes[i + 2], b'u' | b'x')
            {
                let end = (i + 10).min(bytes.len());
                let snippet = String::from_utf8_lossy(&bytes[i..end]).to_string();
                if bs_hits.len() < 5 {
                    bs_hits.push((i, snippet));
                }
                // Skip this pair, to avoid triggering multiple times at the same position.
                i += 3;
                continue;
            }
            // Raw control characters (bytes 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F).
            // serde_json escapes them as `\u00XX`, legal JSON; but some strict proxies
            // or base64 / intermediate encoding layers are most error-prone with these bytes.
            if (b < 0x20 && !matches!(b, b'\t' | b'\n' | b'\r')) && ctrl_hits.len() < 10 {
                ctrl_hits.push((i, b));
            }
            i += 1;
        }
        if !bs_hits.is_empty() {
            log::warn!("[byop] {label} suspicious literal '\\\\u'/'\\\\x' patterns: {bs_hits:?}");
        }
        if !ctrl_hits.is_empty() {
            log::warn!("[byop] {label} contains raw control chars (offset, byte): {ctrl_hits:?}");
        }
    }
    scan_suspicious_backslash("full_request_json", &diag_body_json);
    if let Some(sys) = chat_req.system.as_deref() {
        scan_suspicious_backslash("system", sys);
    }
    for (idx, m) in chat_req.messages.iter().enumerate() {
        if let Some(t) = m.content.first_text() {
            scan_suspicious_backslash(&format!("msg[{idx}]"), t);
        }
    }

    let stream = async_stream::stream! {
        // 1) StreamInit: always sent first, so the UI can immediately show "thinking..."
        yield Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(
                api::response_event::StreamInit {
                    request_id: request_id.clone(),
                    conversation_id: server_conversation_token,
                    run_id: String::new(),
                },
            )),
        });

        // 2) First turn: CreateTask upgrades the Optimistic root → Server.
        if needs_create_task {
            yield Ok(create_task_event(&task_id));
        }

        // 3) Persist the UserQuery / ToolCallResult in input to task.messages.
        //    (on the warp server path the backend emits them; BYOP clients must emit them themselves, see the comment above.)
        //    On the tag-in first turn write root first, then the spawn branch below copies to the new subtask; follow-up turns of an existing CLI
        //    subagent write directly to target_task_id.
        let persistence_task_id = if lrc_should_spawn_subagent {
            task_id.as_str()
        } else {
            target_task_id.as_str()
        };
        let mut persistence_messages: Vec<api::Message> = Vec::new();
        let mut persistence_order: Vec<String> = Vec::new();
        for (input_idx, input) in params.input.iter().enumerate() {
            match input {
                AIAgentInput::UserQuery {
                    query,
                    context,
                    running_command,
                    ..
                } => {
                    let attachments = user_context::collect_user_attachments(context);
                    log::info!(
                        "[byop-diag] persistence input[{input_idx}]: task_id={} \
                         kind=UserQuery query_len={} binaries={} running_command={} \
                         lrc_command_id={} query={:?}",
                        persistence_task_id,
                        query.len(),
                        attachments.binaries.len(),
                        running_command.is_some(),
                        lrc_command_id.as_deref().unwrap_or(""),
                        snippet_for_log(query, BYOP_DIAG_SNIPPET_CHARS),
                    );
                    persistence_order.push(format!(
                        "{input_idx}:UserQuery(query_len={},binaries={})",
                        query.len(),
                        attachments.binaries.len()
                    ));
                    persistence_messages.push(make_user_query_message(
                        persistence_task_id,
                        &request_id,
                        query.clone(),
                        &attachments.binaries,
                    ));
                }
                AIAgentInput::ActionResult { result, .. } => {
                    let content = tools::serialize_action_result(result).unwrap_or_else(|| {
                        serde_json::json!({ "result": result.result.to_string() }).to_string()
                    });
                    log::info!(
                        "[byop-diag] persistence input[{input_idx}]: task_id={} \
                         kind=ActionResult call_id={} content_len={} content={:?}",
                        persistence_task_id,
                        result.id,
                        content.len(),
                        snippet_for_log(&content, BYOP_DIAG_SNIPPET_CHARS),
                    );
                    persistence_order.push(format!(
                        "{input_idx}:ActionResult(call_id={},content_len={})",
                        result.id,
                        content.len()
                    ));
                    persistence_messages.push(make_tool_call_result_message(
                        persistence_task_id,
                        &request_id,
                        result.id.to_string(),
                        content,
                    ));
                }
                _ => {}
            }
        }
        log::info!(
            "[byop-diag] persistence summary: request_id={} task_id={} emitted_messages={} \
             input_order={:?}",
            request_id,
            persistence_task_id,
            persistence_messages.len(),
            persistence_order,
        );
        if !persistence_messages.is_empty() {
            yield Ok(make_add_messages_event(persistence_task_id, persistence_messages));
        }

        // 3.5) LRC subagent spawn (aligned with the upstream cloud's cli subagent injection path).
        //
        // When the request comes from alt-screen + agent tagged-in state, `lrc_command_id` carries the id string of the current LRC
        // block. Here the client synthesizes two events:
        //   a) AddMessagesToTask(root, [<virtual subagent tool_call>])
        //      in root.messages, attach a ToolCall::Subagent { task_id=<new subtask>,
        //      metadata: Cli { command_id }, payload: "" }.
        //      conversation `Task::new_subtask` matches this subagent_call by task_id from parent.messages,
        //      extracts SubagentParams and attaches it to the subtask.
        //   b) CreateTask(api::Task { id=<new subtask>, dependencies.parent_task_id=root })
        //      triggers `apply_client_action::CreateTask`; since parent_id is non-empty it goes to `new_subtask`,
        //      then emits `BlocklistAIHistoryEvent::CreatedSubtask` →
        //      `cli_controller::handle_history_model_event` sees cli_subagent_block_id
        //      non-empty, emits `CLISubagentEvent::SpawnedSubagent` → terminal_view creates the
        //      `CLISubagentView` floating window, attached into the `cli_subagent_views` map.
        //
        // Switch the task_id of subsequent chunk emits to subtask_id, so the model's reasoning/output/tool_call
        // all go into the subtask, and the subagent_view renders the floating window content from it.
        //
        // Ordering constraint: must come after root CreateTask + UserQuery persistence and before the model stream.
        // Otherwise the conversation cannot find the root task / the user query reference pair.
        let mut current_task_id = if lrc_should_spawn_subagent {
            task_id.clone()
        } else {
            target_task_id.clone()
        };
        if lrc_should_spawn_subagent {
            let Some(command_id) = lrc_command_id.clone() else {
                log::warn!("[byop] LRC spawn requested without command_id");
                yield Err(Arc::new(AIApiError::Other(anyhow::anyhow!(
                    "BYOP LRC spawn requested without command_id"
                ))));
                return;
            };
            let subtask_id = if target_task_id != task_id {
                target_task_id.clone()
            } else {
                Uuid::new_v4().to_string()
            };
            let tool_call_id = Uuid::new_v4().to_string();
            log::info!(
                "[byop] LRC tag-in: spawning cli subagent subtask={subtask_id} \
                 command_id={command_id} parent={task_id}"
            );

            let subagent_tool = api::message::tool_call::Tool::Subagent(
                api::message::tool_call::Subagent {
                    task_id: subtask_id.clone(),
                    payload: String::new(),
                    metadata: Some(
                        api::message::tool_call::subagent::Metadata::Cli(
                            api::message::tool_call::subagent::CliSubagent {
                                command_id,
                            },
                        ),
                    ),
                },
            );
            let subagent_msg = make_tool_call_message(
                &task_id,
                &request_id,
                &tool_call_id,
                subagent_tool,
            );
            // a) Attach the subagent tool_call to root.messages so new_subtask can look up SubagentParams.
            yield Ok(make_add_messages_event(&task_id, vec![subagent_msg]));
            // b) Create the subtask with parent_task_id; the conversation detects non-empty parent_id →
            //    goes through the `Task::new_subtask` path, automatically binding SubagentParams.
            yield Ok(create_subtask_event(&subtask_id, &task_id));

            // c) Warp A1: also copy the current turn's UserQuery to the subtask, initializing the subtask's
            //    exchange.output.messages. Otherwise when CLISubagentView renders, the subtask's exchanges
            //    output is empty, and the floating window shows only an empty 49.6-high dialog box forever, with no content visible.
            //    The upstream cloud has a full ClientAction sequence on cli subagent tasks filling exchange.output;
            //    a self-managed BYOP client must inject it explicitly.
            //
            //    Copy only this turn's UserQuery (`pending_user_queries`), leave root's copy alone (root
            //    keeps the user query reference to avoid an empty exchange.input confusing the state machine).
            //    Subsequent model chunks go with `current_task_id = subtask_id`, appended after this starting point.
            if !pending_user_queries.is_empty() {
                let mut subtask_messages: Vec<api::Message> = Vec::new();
                for (q, imgs) in &pending_user_queries {
                    subtask_messages.push(make_user_query_message(
                        &subtask_id,
                        &request_id,
                        q.clone(),
                        imgs,
                    ));
                }
                yield Ok(make_add_messages_event(&subtask_id, subtask_messages));
            }

            // Subsequent chunk emits switch to the subtask.
            current_task_id = subtask_id;
        }

        log::info!("[byop] opening stream: model={model_id}");
        // Retry transient open failures (429/5xx/dropped connection) with 2/4/8s backoff.
        // Nothing has been emitted for this reply yet, so a retry cannot duplicate output.
        let mut open_retries: u32 = 0;
        let open_result = loop {
            let err = match client
                .exec_chat_stream(&model_id, chat_req.clone(), Some(&chat_opts))
                .await
            {
                Ok(resp) => {
                    log::info!("[byop] stream opened OK (HTTP request accepted)");
                    break Ok(resp.stream);
                }
                Err(e) => map_genai_error(e),
            };
            let status = match &err {
                OpenAiCompatibleError::Status { status, body } => Some((*status, body.as_str())),
                OpenAiCompatibleError::Stream(_) => None,
                _ => break Err(err),
            };
            match super::retry::next_delay(open_retries, status) {
                Some(wait) => {
                    open_retries += 1;
                    log::warn!("[byop] open stream failed ({err}); retry {open_retries} in {wait:?}");
                    warpui::r#async::Timer::after(wait).await;
                }
                None => break Err(err),
            }
        };
        let mut sdk_stream = match open_result {
            Ok(stream) => stream,
            Err(mapped) => {
                log::error!("[byop] open stream failed: {mapped:#}");
                yield Err(Arc::new(AIApiError::Other(anyhow::anyhow!(
                    "BYOP open stream failed: {mapped}"
                ))));
                return;
            }
        };

        // Streaming state: the message ids for text / reasoning are each generated when the first chunk arrives,
        // after which chunks append incrementally via AppendToMessageContent.
        let mut text_msg_id: Option<String> = None;
        let mut reasoning_msg_id: Option<String> = None;
        // <think>...</think> streaming extraction state: only meaningful when `use_think_extraction` is true.
        // Models known to interleave reasoning in <think> tags (e.g. MiniMax M3) use this extraction.
        let mut think_active = false;
        let mut think_buf = String::new();
        // tool_call accumulates by call_id; genai's streamed ToolCallChunk already carries a complete ToolCall
        // (since 0.4.0 behavior), but the same call_id may appear several times across chunks with args increments,
        // accumulated in a HashMap by id and emitted together at the end of the stream.
        let mut tool_bufs: HashMap<String, ToolCall> = HashMap::new();
        let mut tool_order: Vec<String> = Vec::new();
        // call_id → id of the first-frame placeholder ToolCall message.
        // On the first ToolCallChunk, if parseable, a placeholder card is emitted immediately (so the UI shows "calling tool X" feedback before
        // the stream End), and at the end of the stream update_message refreshes it in place to the final args.
        // A call_id not in the table (first-frame parse failure / web tool) takes the old path and is emitted once after End.
        let mut tool_msg_ids: HashMap<String, String> = HashMap::new();
        // call_id → time of the last incremental update_message refresh.
        // For long-args tools (create_or_edit_document, long grep query) whose args accumulate across many chunks,
        // throttle reparse + update to ≥ 200ms, so it feels as continuous as the text stream rather than freezing on the first frame until End.
        let mut tool_last_update: HashMap<String, Instant> = HashMap::new();
        // Incremental refresh throttle threshold: consecutive chunks below this value no longer update_message, avoiding frequent UI reflow.
        // Note: each ChatStreamEvent of the SDK stream is awaited independently, so with multiple concurrent tools they already arrive sequentially,
        // and same-tick batch emit has little meaning at this layer; the real jitter reduction is in throttling. This comment warns later readers not to add batching blindly.
        const TOOL_ARGS_UPDATE_THROTTLE_MS: u64 = 200;
        // Diagnostics: count each kind of stream event, print an INFO log at stream end.
        // Used to investigate "messages silently disappearing": if chunk_count=0 and tool_count=0, upstream returned empty content.
        let mut start_count: u32 = 0;
        let mut chunk_count: u32 = 0;
        let mut chunk_bytes: usize = 0;
        let mut reasoning_count: u32 = 0;
        let mut reasoning_bytes: usize = 0;
        let mut tool_chunk_count: u32 = 0;
        let mut end_count: u32 = 0;
        let mut other_count: u32 = 0;
        let mut captured_assistant_text: Option<String> = None;
        // End.captured_content of providers like Ollama is sometimes empty, while Chunk events have delivered the body;
        // stream accumulation serves as the reliable source for content→tool extraction (assistant body only; hypothetical
        // command descriptions in reasoning should not be treated as executable tool calls, see extract_sources below).
        let mut streamed_assistant_text = String::new();
        // Accumulate this turn's token usage. genai carries
        // captured_usage (Option<Usage>) in the ChatStreamEvent::End event; its prompt_tokens is the whole history of this turn
        // (Anthropic / OpenAI both count the "complete request prompt"), and completion_tokens is the model output.
        // Their sum divided by context_window is the "context occupancy", consistent with the semantics of warp's own server path.
        let mut captured_prompt_tokens: i32 = 0;
        let mut captured_completion_tokens: i32 = 0;
        // P0-6 prompt cache hit-rate monitoring: assemble from genai `Usage.prompt_tokens_details` the
        // cache_read / cache_create fields returned by Anthropic / OpenAI / Gemini.
        // See the stream End handling logic. DeepSeek / Ollama do not use cache fields themselves, and
        // stay at 0.
        let mut captured_cache_read_tokens: i32 = 0;
        let mut captured_cache_create_tokens: i32 = 0;

        while let Some(item) = sdk_stream.next().await {
            let event = match item {
                Ok(ev) => ev,
                Err(e) => {
                    let mapped = map_genai_error(e);
                    let err_text = format!("{mapped:#}");
                    log::error!("[byop] stream chunk error: {err_text}");
                    log::error!("[byop-diag] full_request_json_on_error={diag_body_json}");
                    // Parse "column N" from the error message, dump diag_body_json with ±200 chars of context around that position + byte hex.
                    if let Some(col) = err_text
                        .split("column ")
                        .nth(1)
                        .and_then(|s| s.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<usize>().ok())
                    {
                        let body = &diag_body_json;
                        let byte_len = body.len();
                        let start = col.saturating_sub(200).min(byte_len);
                        let end = (col + 200).min(byte_len);
                        let context = body.get(start..end).unwrap_or("(slice failed: not a char boundary)");
                        log::error!(
                            "[byop] error column={col} diag_body_len={byte_len} context[{start}..{end}]={context:?}"
                        );
                        let hex_start = col.saturating_sub(20).min(byte_len);
                        let hex_end = (col + 20).min(byte_len);
                        if let Some(slice) = body.as_bytes().get(hex_start..hex_end) {
                            log::error!("[byop] error bytes[{hex_start}..{hex_end}] hex={slice:02x?}");
                        }
                    }
                    yield Err(Arc::new(AIApiError::Other(anyhow::anyhow!(
                        "BYOP stream error: {mapped}"
                    ))));
                    return;
                }
            };

            match event {
                ChatStreamEvent::Start => {
                    // unit event; the UI already shows thinking via StreamInit, no-op here
                    start_count += 1;
                }
                ChatStreamEvent::Chunk(c) if !c.content.is_empty() => {
                    chunk_count += 1;
                    chunk_bytes += c.content.len();
                    streamed_assistant_text.push_str(&c.content);
                    if use_think_extraction {
                        // <think> tag streaming extraction: only activated for models in the THINK_TAG_IN_CONTENT_MODELS whitelist.
                        // Route the <think>...</think> segment in /delta/content to the reasoning channel,
                        // and the rest of the content goes through the text channel as usual. Supports tag content spanning chunk boundaries.
                        //
                        // known limitation: when the `<think>` tag string itself is cut across chunks (e.g.
                        // chunk1 ends with `<thi` and chunk2 starts with `nk>`) it cannot be recognized, and the leftover string
                        // is output as normal text. Most reasoning models output `<think>` as a complete token,
                        // so the actual probability of triggering is very low.
                        let mut rest: &str = &c.content;
                        loop {
                            if think_active {
                                match rest.find("</think>") {
                                    Some(end) => {
                                        think_buf.push_str(&rest[..end]);
                                        let reasoning = std::mem::take(&mut think_buf);
                                        think_active = false;
                                        rest = &rest[end + "</think>".len()..];
                                        if !reasoning.is_empty() {
                                            reasoning_count += 1;
                                            reasoning_bytes += reasoning.len();
                                            if let Some(id) = reasoning_msg_id.clone() {
                                                yield Ok(make_append_event(&current_task_id, &id, AppendKind::Reasoning(reasoning)));
                                            } else {
                                                let new_id = Uuid::new_v4().to_string();
                                                let mut msg = make_reasoning_message(&current_task_id, &request_id, reasoning);
                                                msg.id = new_id.clone();
                                                reasoning_msg_id = Some(new_id);
                                                yield Ok(make_add_messages_event(&current_task_id, vec![msg]));
                                            }
                                        }
                                    }
                                    None => {
                                        think_buf.push_str(rest);
                                        break;
                                    }
                                }
                            } else {
                                match rest.find("<think>") {
                                    Some(start) => {
                                        let before = rest[..start].to_owned();
                                        think_active = true;
                                        rest = &rest[start + "<think>".len()..];
                                        if !before.is_empty() {
                                            if let Some(id) = text_msg_id.clone() {
                                                yield Ok(make_append_event(&current_task_id, &id, AppendKind::Text(before)));
                                            } else {
                                                let new_id = Uuid::new_v4().to_string();
                                                let mut msg = make_agent_output_message(&current_task_id, &request_id, before);
                                                msg.id = new_id.clone();
                                                text_msg_id = Some(new_id);
                                                yield Ok(make_add_messages_event(&current_task_id, vec![msg]));
                                            }
                                        }
                                    }
                                    None => {
                                        let text = rest.to_owned();
                                        if !text.is_empty() {
                                            if let Some(id) = text_msg_id.clone() {
                                                yield Ok(make_append_event(&current_task_id, &id, AppendKind::Text(text)));
                                            } else {
                                                let new_id = Uuid::new_v4().to_string();
                                                let mut msg = make_agent_output_message(&current_task_id, &request_id, text);
                                                msg.id = new_id.clone();
                                                text_msg_id = Some(new_id);
                                                yield Ok(make_add_messages_event(&current_task_id, vec![msg]));
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        if let Some(id) = text_msg_id.clone() {
                            yield Ok(make_append_event(&current_task_id, &id, AppendKind::Text(c.content)));
                        } else {
                            let new_id = Uuid::new_v4().to_string();
                            let mut msg = make_agent_output_message(&current_task_id, &request_id, c.content);
                            msg.id = new_id.clone();
                            text_msg_id = Some(new_id);
                            yield Ok(make_add_messages_event(&current_task_id, vec![msg]));
                        }
                    }
                }
                ChatStreamEvent::Chunk(_) => {}
                ChatStreamEvent::ReasoningChunk(c) if !c.content.is_empty() => {
                    reasoning_count += 1;
                    reasoning_bytes += c.content.len();
                    // Runtime latch: if this (api_type, model_id) ever sent a reasoning chunk →
                    // mark it so that from the next turn reasoning_content is forced to be echoed, covering any domestic/third-party
                    // thinking models outside the INTERLEAVED_RULES static table (aligned with opencode's data-driven approach,
                    // using stream probing in place of an external catalog).
                    super::reasoning::note_reasoning_seen(api_type, &model_id);
                    if let Some(id) = reasoning_msg_id.clone() {
                        yield Ok(make_append_event(&current_task_id, &id, AppendKind::Reasoning(c.content)));
                    } else {
                        let new_id = Uuid::new_v4().to_string();
                        let mut msg = make_reasoning_message(&current_task_id, &request_id, c.content);
                        msg.id = new_id.clone();
                        reasoning_msg_id = Some(new_id);
                        yield Ok(make_add_messages_event(&current_task_id, vec![msg]));
                    }
                }
                ChatStreamEvent::ReasoningChunk(_) => {}
                ChatStreamEvent::ToolCallChunk(tc) => {
                    tool_chunk_count += 1;
                    let mut call = tc.tool_call;
                    // In rare cases a provider (self-hosted ollama proxy etc.) sends no call_id; a local uuid is the fallback.
                    if call.call_id.is_empty() {
                        call.call_id = Uuid::new_v4().to_string();
                    }
                    // On first sight of this call_id → immediately push a placeholder ToolCall message into pending_placeholders,
                    // so the UI shows a "calling tool X" card before the stream End.
                    // When multiple tools arrive in the same tick: emit add_messages once as a batch before this loop ends,
                    // reducing view tree reflow count.
                    // Already in the table (placeholder sent) and args get a new chunk → throttle ≥ 200ms reparse + update_message
                    // to refresh args incrementally, so long-args tools (create_or_edit_document, long grep, etc.) feel continuous.
                    // Web tools (webfetch/websearch) use their own loading-frame chain (around L2102),
                    // skipped here to avoid double cards.
                    // todowrite goes through the BYOP todo interceptor, synthesizing Message::UpdateTodos to trigger the chip,
                    // and the placeholder is skipped here too to avoid a meaningless "calling todowrite" card.
                    if call.fn_name != tools::webfetch::TOOL_NAME
                        && call.fn_name != tools::websearch::TOOL_NAME
                        && call.fn_name != tools::todowrite::TOOL_NAME
                    {
                        if let Some(msg_id) = tool_msg_ids.get(&call.call_id).cloned() {
                            // Placeholder already emitted → throttled incremental refresh.
                            let now = Instant::now();
                            let last = tool_last_update.get(&call.call_id).copied();
                            let elapsed_ok = last
                                .map(|t| now.duration_since(t).as_millis() as u64 >= TOOL_ARGS_UPDATE_THROTTLE_MS)
                                .unwrap_or(true);
                            if elapsed_ok {
                                if let Ok(parsed) =
                                    parse_incoming_tool_call(&call, mcp_context.as_ref())
                                {
                                    let mut updated = make_tool_call_message(
                                        &current_task_id,
                                        &request_id,
                                        &call.call_id,
                                        parsed,
                                    );
                                    updated.id = msg_id;
                                    tool_last_update.insert(call.call_id.clone(), now);
                                    yield Ok(make_update_message_event(
                                        &current_task_id,
                                        updated,
                                        vec!["tool_call".to_owned()],
                                    ));
                                }
                                // reparse failed (intermediate state): stay silent, wait for the next chunk.
                            }
                        } else if let Ok(parsed) =
                            parse_incoming_tool_call(&call, mcp_context.as_ref())
                        {
                            // First parse succeeds → emit the placeholder card immediately.
                            // Every chunk reparses before a placeholder is emitted (i.e. "retry on every
                            // chunk"), so even if first-frame args are incomplete, any later chunk that is complete
                            // triggers the placeholder emit immediately; this is the coverage path of P1-4,
                            // and a generic placeholder variant is no longer needed.
                            let msg_id = Uuid::new_v4().to_string();
                            let mut placeholder = make_tool_call_message(
                                &current_task_id,
                                &request_id,
                                &call.call_id,
                                parsed,
                            );
                            placeholder.id = msg_id.clone();
                            tool_msg_ids.insert(call.call_id.clone(), msg_id);
                            tool_last_update.insert(
                                call.call_id.clone(),
                                Instant::now(),
                            );
                            yield Ok(make_add_messages_event(
                                &current_task_id,
                                vec![placeholder],
                            ));
                        }
                        // First-frame parse failed (args not yet complete / unknown tool): do not emit for now,
                        // retry on the next chunk or take the old path at End, avoiding visual jitter.
                    }
                    // Same call_id in multiple chunks: later one overrides (genai has already merged args).
                    if !tool_bufs.contains_key(&call.call_id) {
                        tool_order.push(call.call_id.clone());
                    }
                    tool_bufs.insert(call.call_id.clone(), call);
                }
                ChatStreamEvent::End(end) => {
                    end_count += 1;
                    // captured_content of genai >= 0.4.0 contains tool_calls.
                    // Prefer the tool_calls in captured_content (more complete),
                    // otherwise use the tool_bufs accumulated during streaming.
                    if let Some(content) = end.captured_content.as_ref() {
                        if let Some(text) = content.first_text() {
                            if !text.is_empty() {
                                captured_assistant_text = Some(text.to_owned());
                            }
                        }
                        let mut captured_order: Vec<String> = Vec::new();
                        for call in content.tool_calls() {
                            if !captured_order.contains(&call.call_id) {
                                captured_order.push(call.call_id.clone());
                            }
                            tool_bufs.insert(call.call_id.clone(), call.clone());
                        }
                        if !captured_order.is_empty() {
                            for call_id in &tool_order {
                                if !captured_order.contains(call_id) {
                                    captured_order.push(call_id.clone());
                                }
                            }
                            tool_order = captured_order;
                        }
                    }
                    if let Some(usage) = end.captured_usage.as_ref() {
                        // On multiple Ends take the maximum as a fallback (in theory a single stream has only one End).
                        if let Some(p) = usage.prompt_tokens {
                            captured_prompt_tokens = captured_prompt_tokens.max(p);
                        }
                        if let Some(c) = usage.completion_tokens {
                            captured_completion_tokens = captured_completion_tokens.max(c);
                        }
                        // P0-6 prompt cache hit-rate monitoring: Anthropic / OpenAI / Gemini return in
                        // `prompt_tokens_details` respectively `cache_read_input_tokens`
                        // (Anthropic) / `cached_tokens`(OpenAI) / `cachedContentTokenCount`
                        // (Gemini). genai has already unified the mapping to `cached_tokens`.
                        // Likewise `cache_creation_tokens` is provided only by Anthropic (a write-billing hint).
                        // On multiple Ends take the maximum as a fallback, same semantics as prompt/completion.
                        if let Some(details) = usage.prompt_tokens_details.as_ref() {
                            if let Some(r) = details.cached_tokens {
                                captured_cache_read_tokens =
                                    captured_cache_read_tokens.max(r);
                            }
                            if let Some(w) = details.cache_creation_tokens {
                                captured_cache_create_tokens =
                                    captured_cache_create_tokens.max(w);
                            }
                        }
                    }
                }
                _ => {
                    other_count += 1;
                    // ThoughtSignatureChunk etc. are not handled for now (Gemini 3 thoughts need to be passed back to later turns,
                    // BYOP currently does not persist thought_signatures; the degradation is accepted)
                }
            }
        }

        // Warp: the content→tool extraction fallback applies only to Ollama, symmetric with the history filtering at :1412;
        // if the body of a cloud provider (OpenAI/Anthropic/...) happens to look like tool JSON (e.g. the model is
        // explaining a JSON example), it should not be wrongly executed as a real ToolCall.
        if tool_bufs.is_empty() && api_type == AgentProviderApiType::Ollama {
            let extract_sources: [&str; 2] = [
                streamed_assistant_text.as_str(),
                captured_assistant_text.as_deref().unwrap_or(""),
            ];
            let mut parsed_any_text = false;
            for text in extract_sources.into_iter().filter(|t| !t.is_empty()) {
                parsed_any_text = true;
                let extracted = super::content_tool_calls::extract_tool_calls_from_assistant_text(
                    text,
                    &tool_names_for_extract,
                );
                if extracted.is_empty() {
                    continue;
                }
                log::info!(
                    "[byop] content_tool_extract: found={} names={:?} source_len={}",
                    extracted.len(),
                    extracted
                        .iter()
                        .map(|call| call.fn_name.as_str())
                        .collect::<Vec<_>>(),
                    text.len()
                );
                for call in extracted {
                    let call_id = call.call_id.clone();
                    if !tool_bufs.contains_key(&call_id) {
                        tool_order.push(call_id.clone());
                    }
                    tool_bufs.insert(call_id, call);
                }
                break;
            }
            if tool_bufs.is_empty() && parsed_any_text {
                let preview: String = streamed_assistant_text.chars().take(240).collect();
                log::info!(
                    "[byop] content_tool_extract: no tools parsed (streamed={}B captured={}B) \
                     preview={preview:?}",
                    streamed_assistant_text.len(),
                    captured_assistant_text.as_ref().map(|t| t.len()).unwrap_or(0),
                );
            } else if tool_bufs.is_empty() {
                log::warn!(
                    "[byop] content_tool_extract: skipped — no assistant text in stream \
                     (chunks={chunk_count} reasoning={reasoning_count})"
                );
            }
        }

        // Stream statistics INFO log. When chunk_count=0 && tool_count=0 upstream returned empty,
        // most likely model_id is not recognized / max_tokens is missing / an Anthropic API-compatible proxy returned 200 with an empty body.
        let total_tools = tool_bufs.len();
        log::info!(
            "[byop] stream stats: start={start_count} chunks={chunk_count} ({chunk_bytes}B) \
             reasoning={reasoning_count} ({reasoning_bytes}B) native_tool_chunks={tool_chunk_count} \
             ends={end_count} other={other_count} captured_tools={total_tools}"
        );
        // P0-6 prompt cache hit-rate log (printed only when the provider returns cache fields).
        // ratio = cache_read / (prompt_tokens.max(1)) is the fraction of this turn's input
        // that directly hit the cache. create > 0 means this turn had a cache write; write price ≈ 1.25x base (5m) or
        // 2x base (1h). Read price ≈ 0.1x base; in the long run, one reuse or more pays back.
        // Use ratio to judge whether the P0 optimization is effective: the 2nd+ turn of the same conversation should show a markedly rising ratio.
        //
        // **P2-16**: additionally append a `compaction=` marker. Compaction itself rewrites history so the messages
        // prefix is inconsistent before vs after compaction → the first turn after compaction is bound to cache miss. Emitting this
        // signal in the log lets later analysis (`script/analyze-prompt-cache.ps1`) distinguish "normal miss" from
        // "compaction-caused miss", avoiding false positives.
        if captured_cache_read_tokens > 0 || captured_cache_create_tokens > 0 {
            let denom = captured_prompt_tokens.max(1);
            let read_ratio = captured_cache_read_tokens as f32 / denom as f32;
            let create_ratio = captured_cache_create_tokens as f32 / denom as f32;
            // Compaction state: none → not enabled / inactive → enabled but unchanged this turn /
            // active (count of hidden message ids) → this turn took the compaction path.
            let compaction_label = match params.compaction_state.as_ref() {
                None => "none".to_owned(),
                Some(s) => {
                    let hidden = s.hidden_message_ids().len();
                    if hidden == 0 {
                        "inactive".to_owned()
                    } else {
                        format!("active(hidden={hidden})")
                    }
                }
            };
            log::info!(
                "[byop-cache] prompt_tokens={captured_prompt_tokens} \
                 cache_read={captured_cache_read_tokens} ({:.1}%) \
                 cache_create={captured_cache_create_tokens} ({:.1}%) \
                 model={model_id} compaction={compaction_label}",
                read_ratio * 100.0,
                create_ratio * 100.0,
            );
        }
        if chunk_count == 0 && reasoning_count == 0 && total_tools == 0 {
            log::warn!(
                "[byop] stream returned 0 content / 0 reasoning / 0 tool_calls — \
                 upstream may have returned an empty response (wrong model_id? missing max_tokens? proxy error?)"
            );
        }

        // End of stream: emit the accumulated tool_calls all at once.
        let mut final_messages: Vec<api::Message> = Vec::new();
        let mut ordered_tool_calls: Vec<ToolCall> = Vec::with_capacity(tool_bufs.len());
        for call_id in tool_order {
            if let Some(call) = tool_bufs.remove(&call_id) {
                ordered_tool_calls.push(call);
            }
        }
        let mut unordered_tool_calls: Vec<ToolCall> = tool_bufs.into_values().collect();
        if !unordered_tool_calls.is_empty() {
            // On the normal path both ChunkArgs / End maintain `tool_order` in sync, so when we get here
            // `tool_bufs` should be empty. This fallback is hit only when a provider is abnormal (e.g. captured_content and ChunkArgs
            // are each missing some call_id). dict-sort guarantees a stable OpenAI-compatible
            // `tool_calls[]` order across calls (no drifting cache prefix), but it should warn.
            log::warn!(
                "[byop] {} tool_calls fell through to dict-sort fallback — \
                 provider inconsistency between ChunkArgs and captured_content; \
                 call_ids={:?}",
                unordered_tool_calls.len(),
                unordered_tool_calls.iter().map(|t| t.call_id.as_str()).collect::<Vec<_>>(),
            );
        }
        unordered_tool_calls.sort_by(|a, b| a.call_id.cmp(&b.call_id));
        ordered_tool_calls.extend(unordered_tool_calls);
        for call in ordered_tool_calls {
            // Diagnostics: dump the raw payload of the tool_call the model actually sent
            // (call_id / fn_name / fn_arguments raw JSON + type annotation),
            // to check whether the model produced inputs/outputs per the schema (common issues: bool fields stringified,
            // numbers quoted, nested objects collapsed into strings, etc.).
            // debug level: only enable RUST_LOG=debug when investigating schema problems, so INFO is not polluted normally.
            // info level keeps a short one-line summary without args, to see streaming timing.
            log::info!(
                "[byop] tool_call_in: name={} call_id={}",
                call.fn_name,
                call.call_id,
            );
            if log::log_enabled!(log::Level::Debug) {
                let args_repr = if call.fn_arguments.is_string() {
                    format!("string({:?})", call.fn_arguments.as_str().unwrap_or(""))
                } else {
                    format!(
                        "{}({})",
                        match &call.fn_arguments {
                            Value::Object(_) => "object",
                            Value::Array(_) => "array",
                            Value::Bool(_) => "bool",
                            Value::Number(_) => "number",
                            Value::Null => "null",
                            Value::String(_) => "string",
                        },
                        call.fn_arguments
                    )
                };
                log::debug!(
                    "[byop] tool_call_in_args: name={} call_id={} args={}",
                    call.fn_name,
                    call.call_id,
                    args_repr,
                );
            }

            // Warp BYOP todowrite interception: not mapped to the protobuf executor, synthesizes
            // `Message::UpdateTodos` written directly to conversation.todo_lists to trigger the chip + popup
            // UI (aligned with the server-side ClientAction::AddMessagesToTask::UpdateTodos path).
            // Then append the carrier ToolCall + ToolCallResult to unblock the model.
            if call.fn_name == tools::todowrite::TOOL_NAME {
                let args_str = if call.fn_arguments.is_string() {
                    call.fn_arguments.as_str().unwrap_or("").to_owned()
                } else {
                    call.fn_arguments.to_string()
                };

                match tools::todowrite::build_update_todos_messages(
                    &args_str,
                    &current_task_id,
                    &request_id,
                ) {
                    Ok(todo_msgs) if !todo_msgs.is_empty() => {
                        // Yield UpdateTodos directly so the UI updates the chip in real time.
                        // Goes through AddMessagesToTask: the apply_client_action path will
                        // hit the Message::UpdateTodos branch → update_todo_list_from_todo_op
                        // → emit BlocklistAIHistoryEvent::UpdatedTodoList, and the UI refreshes automatically.
                        yield Ok(make_add_messages_event(&current_task_id, todo_msgs));
                        let result_payload =
                            tools::todowrite::success_result_to_json("todo list updated");
                        let result_content = serde_json::to_string(&result_payload)
                            .unwrap_or_else(|_| r#"{"status":"ok"}"#.to_owned());
                        final_messages.push(make_tool_call_carrier_message(
                            &current_task_id,
                            &request_id,
                            &call.call_id,
                            &call.fn_name,
                            &args_str,
                        ));
                        final_messages.push(make_tool_call_result_message(
                            &current_task_id,
                            &request_id,
                            call.call_id.clone(),
                            result_content,
                        ));
                    }
                    Ok(_) => {
                        // Empty todos array: do not emit UpdateTodos, but still give the model a result
                        // otherwise the next turn of chat would hang (the model waits for the result of tool_call_id).
                        let result_payload = tools::todowrite::success_result_to_json("no todos");
                        let result_content = serde_json::to_string(&result_payload)
                            .unwrap_or_else(|_| r#"{"status":"ok","message":"no todos"}"#.to_owned());
                        final_messages.push(make_tool_call_carrier_message(
                            &current_task_id,
                            &request_id,
                            &call.call_id,
                            &call.fn_name,
                            &args_str,
                        ));
                        final_messages.push(make_tool_call_result_message(
                            &current_task_id,
                            &request_id,
                            call.call_id.clone(),
                            result_content,
                        ));
                    }
                    Err(e) => {
                        // args parse failed: same as from_args failure, emit an error tool_result.
                        log::warn!(
                            "[byop] todowrite args parse failed: call_id={} err={e:#}",
                            call.call_id
                        );
                        let error_payload = tools::todowrite::invalid_arguments_result_to_json(
                            e.to_string(),
                            &args_str,
                        );
                        let error_content = serde_json::to_string(&error_payload)
                            .unwrap_or_else(|_| r#"{"error":"invalid_arguments"}"#.to_owned());
                        final_messages.push(make_tool_call_carrier_message(
                            &current_task_id,
                            &request_id,
                            &call.call_id,
                            &call.fn_name,
                            &args_str,
                        ));
                        final_messages.push(make_tool_call_result_message(
                            &current_task_id,
                            &request_id,
                            call.call_id.clone(),
                            error_content,
                        ));
                    }
                }
                continue;
            }

            // Warp BYOP web tool interception: webfetch / websearch are not mapped to the protobuf
            // executor variant; local HTTP is run directly here, synthesizing a (carrier ToolCall,
            // ToolCallResult) pair of messages, bypassing parse_incoming_tool_call.
            //
            // UI: aligned with cloud mode, emit one `Message::WebSearch` /
            // `Message::WebFetch` status message before and after, triggering the inline_action `WebSearchView` /
            // `WebFetchView` rendering: Searching/Fetching loading card → Success (URL list)
            // / Error collapsed card. These two do not go into final_messages and are yielded directly so the UI updates in real time;
            // the carrier + result still go through final_messages for the next turn's model reasoning.
            if call.fn_name == tools::webfetch::TOOL_NAME
                || call.fn_name == tools::websearch::TOOL_NAME
            {
                let args_str = if call.fn_arguments.is_string() {
                    call.fn_arguments.as_str().unwrap_or("").to_owned()
                } else {
                    call.fn_arguments.to_string()
                };
                let is_search = call.fn_name == tools::websearch::TOOL_NAME;

                // Pre-parse args to extract query / url for the UI loading card. Emit even if args parsing fails
                // (with empty fields as fallback), ensuring the UI sees at least one loading frame; later dispatch
                // will still return invalid_arguments → switching to the Error card.
                let preview_query = if is_search {
                    serde_json::from_str::<tools::web_runtime::SearchToolArgs>(&args_str)
                        .map(|a| a.query)
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let preview_urls: Vec<String> = if !is_search {
                    serde_json::from_str::<tools::web_runtime::FetchArgs>(&args_str)
                        .map(|a| vec![a.url])
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };

                // The Searching/Fetching loading frame and the final Success/Error frame must share the same
                // message.id: `block.rs::handle_web_search_messages` reuses WebSearchView by id;
                // different ids would create two independent cards.
                let web_msg_id = Uuid::new_v4().to_string();
                let mut loading_msg = if is_search {
                    make_web_search_searching_message(
                        &current_task_id,
                        &request_id,
                        preview_query.clone(),
                    )
                } else {
                    make_web_fetch_fetching_message(
                        &current_task_id,
                        &request_id,
                        preview_urls.clone(),
                    )
                };
                loading_msg.id = web_msg_id.clone();
                yield Ok(make_add_messages_event(&current_task_id, vec![loading_msg]));

                let result_json = dispatch_byop_web_tool(&call.fn_name, &args_str).await;

                let mut done_msg = if is_search {
                    make_web_search_status_from_result(
                        &current_task_id,
                        &request_id,
                        &preview_query,
                        &result_json,
                    )
                } else {
                    make_web_fetch_status_from_result(
                        &current_task_id,
                        &request_id,
                        &preview_urls,
                        &result_json,
                    )
                };
                done_msg.id = web_msg_id;
                // The second frame must not use AddMessagesToTask again: that would append a second record with the same id
                // to task.messages, and the `output.rs::WebSearch` rendering branch add_child's by message count,
                // showing two side-by-side cards. Use UpdateTaskMessage + FieldMask instead: `task::upsert_message`
                // finds the existing message with the same id and merges in place via FieldMaskOperation::update,
                // task.messages still has only one → the UI has one card with set_status switching.
                let mask_path = if is_search { "web_search" } else { "web_fetch" };
                yield Ok(make_update_message_event(
                    &current_task_id,
                    done_msg,
                    vec![mask_path.to_owned()],
                ));

                let result_content = serde_json::to_string(&result_json)
                    .unwrap_or_else(|_| r#"{"status":"serialize_error"}"#.to_owned());
                final_messages.push(make_tool_call_carrier_message(
                    &current_task_id,
                    &request_id,
                    &call.call_id,
                    &call.fn_name,
                    &args_str,
                ));
                final_messages.push(make_tool_call_result_message(
                    &current_task_id,
                    &request_id,
                    call.call_id.clone(),
                    result_content,
                ));
                continue;
            }

            match parse_incoming_tool_call(&call, mcp_context.as_ref()) {
                Ok(warp_tool) => {
                    // If a placeholder card was already emitted at the ToolCallChunk stage (same call_id),
                    // use update_message to refresh in place to the final args (covering any args delta that
                    // may have arrived later in chunks). The placeholder and final frame share the same message.id,
                    // task::upsert_message goes through FieldMaskOperation::update,
                    // task.messages still has only one → the UI card refreshes in place, no double cards.
                    if let Some(msg_id) = tool_msg_ids.get(&call.call_id).cloned() {
                        let mut updated = make_tool_call_message(
                            &current_task_id,
                            &request_id,
                            &call.call_id,
                            warp_tool,
                        );
                        updated.id = msg_id;
                        yield Ok(make_update_message_event(
                            &current_task_id,
                            updated,
                            vec!["tool_call".to_owned()],
                        ));
                    } else {
                        final_messages.push(make_tool_call_message(
                            &current_task_id,
                            &request_id,
                            &call.call_id,
                            warp_tool,
                        ));
                    }
                }
                Err(e) => {
                    // Key: from_args failure is no longer swallowed into plain text (original implementation: emit AgentOutput),
                    // because the model that turn thinks it called a tool and is waiting for the result; seeing a piece of plain assistant text
                    // it has no idea the argument type was wrong and cannot make a targeted correction and retry.
                    // Change to emit a ToolCall (carrier) + ToolCallResult (error JSON) pair,
                    // so the model sees a standard tool_result error next turn and can, as usual, fix args and retry or switch tools.
                    //
                    // The ToolCall's `tool` oneof is left None (no suitable structured variant); the original
                    // fn_name + args_str are carried via server_message_data,
                    // and the carrier branch of serialize_outgoing_tool_call restores them first.
                    let args_str = if call.fn_arguments.is_string() {
                        call.fn_arguments.as_str().unwrap_or("").to_owned()
                    } else {
                        call.fn_arguments.to_string()
                    };
                    log::warn!(
                        "[byop] tool_call parse failed → emit synthetic error tool_result: \
                         tool={} call_id={} err={e:#}",
                        call.fn_name,
                        call.call_id
                    );
                    let error_payload = serde_json::json!({
                        "error": "invalid_arguments",
                        "detail": e.to_string(),
                        "tool": call.fn_name,
                        "received_args": &args_str,
                        "hint": "Arguments did not match the tool's JSON Schema. \
                                 Re-emit the tool call with corrected types / required fields, \
                                 or pick a different tool.",
                    });
                    let error_content = serde_json::to_string(&error_payload)
                        .unwrap_or_else(|_| r#"{"error":"invalid_arguments"}"#.to_owned());
                    final_messages.push(make_tool_call_carrier_message(
                        &current_task_id,
                        &request_id,
                        &call.call_id,
                        &call.fn_name,
                        &args_str,
                    ));
                    final_messages.push(make_tool_call_result_message(
                        &current_task_id,
                        &request_id,
                        call.call_id.clone(),
                        error_content,
                    ));
                }
            }
        }
        if !final_messages.is_empty() {
            yield Ok(make_add_messages_event(&current_task_id, final_messages));
        }

        // Convert the captured token usage into ConversationUsageMetadata.context_window_usage and
        // inject it into StreamFinished; the controller's handle_response_stream_finished writes it to
        // conversation.conversation_usage_metadata, and the footer listening for UpdatedStreamingExchange/
        // AppendedExchange events refreshes the "X% context remaining" tooltip in real time at the end of each turn.
        let usage_metadata = context_window.and_then(|cw| {
            if cw == 0 || (captured_prompt_tokens == 0 && captured_completion_tokens == 0) {
                return None;
            }
            let used = (captured_prompt_tokens + captured_completion_tokens).max(0) as f32;
            let pct = (used / cw as f32).clamp(0.0, 1.0);
            log::info!(
                "[byop] context usage: prompt={} completion={} window={} → {:.1}%",
                captured_prompt_tokens,
                captured_completion_tokens,
                cw,
                pct * 100.0
            );
            Some(api::response_event::stream_finished::ConversationUsageMetadata {
                context_window_usage: pct,
                summarized: false,
                credits_spent: 0.0,
                #[allow(deprecated)]
                token_usage: Vec::new(),
                tool_usage_metadata: None,
                warp_token_usage: std::collections::HashMap::new(),
                byok_token_usage: std::collections::HashMap::new(),
            })
        });
        yield Ok(make_finished_done(usage_metadata));
    };

    Ok(Box::pin(stream))
}

/// Sends a short non-tool request with a separate BYOP config, letting the model generate a conversation title for the first user query.
/// All errors are swallowed (return Err so upstream emits a warn log, without affecting the main flow).
///
/// Implementation delegates to `oneshot::byop_oneshot_streaming_completion`; here we only assemble the prompt and clean the output.
///
/// ## Prompt design
///
/// - **system**: see `prompts/tasks/title_system.md`, structured task/rules/examples,
///   covering bilingual Chinese/English examples, explicitly forbidding "answering the user's question / refusing / adding quotes".
/// - **user**: wrap the original `user_query` in `<user>...</user>`, preceded by an explicit
///   "Generate a title for this conversation:", to prevent weak models from treating the user text as the main instruction and replying directly
///   (typical bad case: user="who are you" → the model answers "I am Claude" which is taken as the title).
/// - **temperature**: 0.3; the opencode title agent uses 0.5, this is more conservative, reducing going off-topic.
pub(crate) async fn generate_title_via_byop(
    tg: &TitleGenInput,
    user_query: &str,
) -> Result<Option<String>, anyhow::Error> {
    let cfg = super::oneshot::OneshotConfig {
        base_url: tg.base_url.clone(),
        api_key: tg.api_key.clone(),
        model_id: tg.model_id.clone(),
        api_type: tg.api_type,
        reasoning_effort: tg.reasoning_effort,
        extra_headers: tg.extra_headers.clone(),
        conversation_id: tg.conversation_id,
    };
    let system = include_str!("prompts/tasks/title_system.md");
    let user_prompt = format!(
        "Generate a title for this conversation:\n<user>{}</user>",
        user_query
    );
    let opts = super::oneshot::OneshotOptions {
        max_chars: Some(1000),
        temperature: Some(0.5),
        ..Default::default()
    };
    let raw = super::oneshot::byop_oneshot_completion(&cfg, system, &user_prompt, &opts).await?;
    Ok(sanitize_title(&raw))
}

/// Cleans the title text. Empty string → None (so upstream skips emit).
///
/// Processing order:
/// 1. Strip `<think>...</think>` / `<reasoning>...</reasoning>` thinking blocks (a common prefix of reasoning models).
/// 2. Take the first non-empty line (models often prepend "Okay, the title is:" then break the line before giving the title).
/// 3. Strip prefixes such as `Title:` / `Thread:` / `Subject:` and their Chinese equivalents (case-insensitive).
/// 4. Strip leading/trailing quotes / backticks (Chinese and English).
/// 5. Strip trailing punctuation.
/// 6. Truncate to 50 characters (by char, protecting CJK); if exceeded, append `…` at the end.
fn sanitize_title(raw: &str) -> Option<String> {
    // 1. Strip reasoning tags (may be multiple, DOTALL mode).
    let mut s = raw.to_owned();
    for tag in &["think", "reasoning", "thought", "scratchpad"] {
        let open = format!("<{}>", tag);
        let close = format!("</{}>", tag);
        while let (Some(start), Some(end_rel)) =
            (s.find(&open), s.find(&close).map(|e| e + close.len()))
        {
            if end_rel <= start {
                break;
            }
            s.replace_range(start..end_rel, "");
        }
    }

    // 2. Take the first non-empty line.
    let first_line = s
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_owned();
    let mut s = first_line;

    // 3. Strip prefixes (strip in a loop, handling double prefixes like "Title: Thread: foo").
    let prefixes = [
        "title:",
        "subject:",
        "thread:",
    ];
    loop {
        let lower = s.to_lowercase();
        let mut stripped = false;
        for p in &prefixes {
            if lower.starts_with(p) {
                s = s[p.len()..].trim_start().to_owned();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }

    // 4. Strip leading/trailing quotes (Chinese and English).
    let quotes = ['"', '\'', '`', '“', '”', '‘', '’', '《', '》', '「', '」'];
    while let Some(c) = s.chars().next() {
        if quotes.contains(&c) {
            s.remove(0);
        } else {
            break;
        }
    }
    while let Some(c) = s.chars().last() {
        if quotes.contains(&c) {
            let new_len = s.len() - c.len_utf8();
            s.truncate(new_len);
        } else {
            break;
        }
    }

    // 5. Strip trailing punctuation.
    while let Some(c) = s.chars().last() {
        if matches!(
            c,
            '.' | '。' | '!' | '！' | '?' | '？' | ',' | '，' | ';' | '；' | ':' | '：'
        ) {
            let new_len = s.len() - c.len_utf8();
            s.truncate(new_len);
        } else {
            break;
        }
    }

    let s = s.trim().to_owned();
    if s.is_empty() {
        return None;
    }

    // 6. Truncate to 50 characters (by char, protecting CJK). Append an ellipsis if too long.
    const MAX_CHARS: usize = 50;
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > MAX_CHARS {
        let mut truncated: String = chars.iter().take(MAX_CHARS - 1).collect();
        truncated.push('…');
        Some(truncated)
    } else {
        Some(s)
    }
}

// ---------------------------------------------------------------------------
// Event construction helpers
// ---------------------------------------------------------------------------

pub(crate) enum AppendKind {
    Reasoning(String),
    Text(String),
}

pub(crate) fn make_add_messages_event(task_id: &str, messages: Vec<api::Message>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_owned(),
                            messages,
                        },
                    )),
                }],
            },
        )),
    }
}

/// Replaces some fields of an existing message with `UpdateTaskMessage` + FieldMask. The controller
/// `conversation::Action::UpdateTaskMessage` → `task::upsert_message` →
/// merges in place with `FieldMaskOperation::update`; if the id already exists no duplicate record is pushed.
/// Used for the BYOP web tool loading → success/error state switch (see the interception branch).
pub(crate) fn make_update_message_event(
    task_id: &str,
    message: api::Message,
    mask_paths: Vec<String>,
) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::UpdateTaskMessage(
                        api::client_action::UpdateTaskMessage {
                            task_id: task_id.to_owned(),
                            message: Some(message),
                            mask: Some(prost_types::FieldMask { paths: mask_paths }),
                        },
                    )),
                }],
            },
        )),
    }
}

pub(crate) fn make_append_event(task_id: &str, message_id: &str, kind: AppendKind) -> api::ResponseEvent {
    let (msg_inner, mask_path) = match kind {
        AppendKind::Reasoning(r) => (
            api::message::Message::AgentReasoning(api::message::AgentReasoning {
                reasoning: r,
                finished_duration: None,
            }),
            "agent_reasoning.reasoning",
        ),
        AppendKind::Text(t) => (
            api::message::Message::AgentOutput(api::message::AgentOutput { text: t }),
            "agent_output.text",
        ),
    };
    let message = api::Message {
        id: message_id.to_owned(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(msg_inner),
        request_id: String::new(),
        timestamp: None,
    };
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AppendToMessageContent(
                        api::client_action::AppendToMessageContent {
                            task_id: task_id.to_owned(),
                            message: Some(message),
                            mask: Some(prost_types::FieldMask {
                                paths: vec![mask_path.to_owned()],
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

/// Local dispatcher for BYOP web tools (`webfetch` / `websearch`).
///
/// Does not go through the protobuf executor; runs HTTP locally with reqwest directly, serializing the structured result
/// into a JSON Value for the upstream LLM. Errors are also serialized as `{status:"error", ...}`,
/// so the model sees a standard tool_result.
async fn dispatch_byop_web_tool(tool_name: &str, args_str: &str) -> Value {
    use tools::web_runtime;
    // Build a client with SSRF protection for webfetch: the custom redirect policy validates the target of every hop.
    let client = match web_runtime::build_ssrf_safe_client() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[byop] reqwest client build failed: {e:#}");
            return web_runtime::error_to_json(tool_name, &anyhow::anyhow!(e.to_string()));
        }
    };
    if tool_name == tools::webfetch::TOOL_NAME {
        match serde_json::from_str::<web_runtime::FetchArgs>(args_str) {
            Ok(args) => match web_runtime::run_webfetch(&client, args).await {
                Ok(out) => web_runtime::fetch_output_to_json(&out),
                Err(e) => {
                    log::warn!("[byop][webfetch] error: {e:#}");
                    web_runtime::error_to_json(tool_name, &e)
                }
            },
            Err(e) => web_runtime::error_to_json(
                tool_name,
                &anyhow::anyhow!(format!("invalid arguments: {e}")),
            ),
        }
    } else {
        // websearch
        match serde_json::from_str::<web_runtime::SearchToolArgs>(args_str) {
            Ok(args) => {
                let api_key = std::env::var("EXA_API_KEY").ok();
                match web_runtime::run_websearch(&client, args, api_key.as_deref(), None).await {
                    Ok(out) => web_runtime::search_output_to_json(&out),
                    Err(e) => {
                        log::warn!("[byop][websearch] error: {e:#}");
                        web_runtime::error_to_json(tool_name, &e)
                    }
                }
            }
            Err(e) => web_runtime::error_to_json(
                tool_name,
                &anyhow::anyhow!(format!("invalid arguments: {e}")),
            ),
        }
    }
}

fn parse_incoming_tool_call(
    call: &ToolCall,
    mcp_ctx: Option<&crate::ai::agent::MCPContext>,
) -> anyhow::Result<api::message::tool_call::Tool> {
    // genai ToolCall.fn_arguments is a Value; the from_args of tools::* expects &str,
    // so the Value is serialized back to a string and passed in (the original protocol is a string JSON).
    let args_str = if call.fn_arguments.is_string() {
        call.fn_arguments.as_str().unwrap_or("").to_owned()
    } else {
        call.fn_arguments.to_string()
    };
    if tools::mcp::is_mcp_function(&call.fn_name) {
        return tools::mcp::parse_mcp_tool_call(&call.fn_name, &args_str, mcp_ctx);
    }
    let Some(tool) = tools::lookup(&call.fn_name) else {
        anyhow::bail!("unknown tool name: {}", call.fn_name);
    };
    match (tool.from_args)(&args_str) {
        Ok(t) => Ok(t),
        Err(e) => {
            // First failure: most likely the model serialized bool/number/array as a string.
            // Run a type coerce with the tool's own schema, then retry.
            let schema = (tool.parameters)();
            if let Some(coerced) = tools::coerce::coerce_args_against_schema(&args_str, &schema) {
                match (tool.from_args)(&coerced) {
                    Ok(t) => {
                        log::info!(
                            "[byop] from_args coerced ok: tool={} original_err={e:#}",
                            call.fn_name
                        );
                        return Ok(t);
                    }
                    Err(e2) => {
                        log::warn!(
                            "[byop] from_args failed (after coerce): tool={} err={e2:#} original_err={e:#} coerced_args={coerced} args_str={args_str}",
                            call.fn_name
                        );
                        return Err(e2);
                    }
                }
            }
            // Diagnostics: on parse failure, print the string from_args actually received verbatim,
            // which together with the upper layer's [byop] tool_call_in args= line can tell:
            //   1. whether the model output the wrong type (bool→"true" / number→"1", etc.)
            //   2. whether escaping went wrong in genai's Value→string conversion
            //   3. whether the whole fn_arguments was stringified (should be an object but is a string)
            log::warn!(
                "[byop] from_args failed: tool={} err={e:#} args_str={args_str}",
                call.fn_name
            );
            Err(e)
        }
    }
}

pub(crate) fn make_reasoning_message(task_id: &str, request_id: &str, reasoning: String) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentReasoning(
            api::message::AgentReasoning {
                reasoning,
                finished_duration: None,
            },
        )),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

pub(crate) fn make_agent_output_message(task_id: &str, request_id: &str, text: String) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text },
        )),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

pub(crate) fn make_user_query_message(
    task_id: &str,
    request_id: &str,
    query: String,
    binaries: &[user_context::UserBinary],
) -> api::Message {
    // Warp: write multimodal binaries (image / pdf / audio, etc.) into `UserQuery.context.images`
    // (an InputContext field; the proto Image is actually a generic container of `bytes data + string mime_type`,
    // the field being named images for historical reasons). UserBinary.data is a base64 string, proto.data is raw bytes,
    // so decode once here; entries that fail to decode are skipped without blocking the model stream (a decode failure already means that entry
    // was not really sent upstream that turn anyway, so dropping it does not affect history consistency).
    let proto_binaries: Vec<api::input_context::Image> = binaries
        .iter()
        .filter_map(|b| {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(&b.data)
                .ok()
                .map(|bytes| api::input_context::Image {
                    data: bytes,
                    mime_type: b.content_type.clone(),
                })
        })
        .collect();
    let context = if proto_binaries.is_empty() {
        None
    } else {
        Some(api::InputContext {
            images: proto_binaries,
            ..Default::default()
        })
    };
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::UserQuery(api::message::UserQuery {
            query,
            context,
            ..Default::default()
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// When BYOP intercepts websearch, emit `Message::WebSearch(Searching{query})`, from which the UI renders
/// the "Searching the web for \"query\"" loading card (`inline_action::web_search`).
pub(crate) fn make_web_search_searching_message(
    task_id: &str,
    request_id: &str,
    query: String,
) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::WebSearch(api::message::WebSearch {
            status: Some(api::message::web_search::Status {
                r#type: Some(api::message::web_search::status::Type::Searching(
                    api::message::web_search::status::Searching { query },
                )),
            }),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// Extracts (url, title) from the results string returned by exa MCP.
///
/// The actual format is line-based metadata blocks, with multiple results separated by `---`:
/// ```
/// Title: Announcing Rust 1.95.0 | Rust Blog
/// URL: https://blog.rust-lang.org/2026/04/16/Rust-1.95.0/
/// Published: 2026-04-16T00:00:00.000Z
/// Author: N/A
/// Highlights:
/// ...
/// ---
/// Title: ...
/// ```
/// On scanning `Title: X`, cache a candidate; the first `URL: Y` right after pairs with it into (Y, X) and is queued, deduplicated.
/// Compatibility fallback: also scan the `[title](url)` markdown link form (in case the exa template switches in the future).
fn extract_search_pages_from_exa_results(s: &str) -> Vec<(String, String)> {
    let mut pages = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Route 1: Title:/URL: line-based
    let mut current_title: Option<String> = None;
    for line in s.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("Title:") {
            current_title = Some(rest.trim().to_owned());
        } else if let Some(rest) = trimmed.strip_prefix("URL:") {
            let url = rest.trim().to_owned();
            let title = current_title.take().unwrap_or_default();
            if (url.starts_with("http://") || url.starts_with("https://"))
                && seen.insert(url.clone())
            {
                pages.push((url, title));
            }
        }
    }

    // Route 2: markdown link `[title](url)` fallback (dedupe already in effect, no duplicates)
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(rel_close_text) = s[i + 1..].find("](") {
                let text_end = i + 1 + rel_close_text;
                let url_start = text_end + 2;
                if let Some(rel_close_url) = s[url_start..].find(')') {
                    let url_end = url_start + rel_close_url;
                    let title = s[i + 1..text_end].trim().to_owned();
                    let url = s[url_start..url_end].trim().to_owned();
                    if (url.starts_with("http://") || url.starts_with("https://"))
                        && seen.insert(url.clone())
                    {
                        pages.push((url, title));
                    }
                    i = url_end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }

    pages
}

/// After BYOP websearch completes, decide the Success / Error state from `result_json`.
///
/// `pages` is extracted by scanning for `[title](url)` in the markdown assembled by exa under `result_json["results"]`.
pub(crate) fn make_web_search_status_from_result(
    task_id: &str,
    request_id: &str,
    query: &str,
    result_json: &Value,
) -> api::Message {
    let is_error = result_json.get("status").and_then(|v| v.as_str()) == Some("error");
    let r#type = if is_error {
        api::message::web_search::status::Type::Error(())
    } else {
        let pages = result_json
            .get("results")
            .and_then(|v| v.as_str())
            .map(extract_search_pages_from_exa_results)
            .unwrap_or_default()
            .into_iter()
            .map(
                |(url, title)| api::message::web_search::status::success::SearchedPage {
                    url,
                    title,
                },
            )
            .collect();
        api::message::web_search::status::Type::Success(api::message::web_search::status::Success {
            query: query.to_owned(),
            pages,
        })
    };
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::WebSearch(api::message::WebSearch {
            status: Some(api::message::web_search::Status {
                r#type: Some(r#type),
            }),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// When BYOP intercepts webfetch, emit `Message::WebFetch(Fetching{urls})`, from which the UI renders
/// the "Fetching N URLs" loading card (`inline_action::web_fetch`).
pub(crate) fn make_web_fetch_fetching_message(
    task_id: &str,
    request_id: &str,
    urls: Vec<String>,
) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::WebFetch(api::message::WebFetch {
            status: Some(api::message::web_fetch::Status {
                r#type: Some(api::message::web_fetch::status::Type::Fetching(
                    api::message::web_fetch::status::Fetching { urls },
                )),
            }),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// After BYOP webfetch completes, extract `url` + HTTP `status` from the `FetchOutput` JSON to assemble the Success
/// card; status="error" goes to the Error card.
pub(crate) fn make_web_fetch_status_from_result(
    task_id: &str,
    request_id: &str,
    fallback_urls: &[String],
    result_json: &Value,
) -> api::Message {
    let is_error = result_json.get("status").and_then(|v| v.as_str()) == Some("error");
    let r#type = if is_error {
        api::message::web_fetch::status::Type::Error(())
    } else {
        let url = result_json
            .get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned())
            .unwrap_or_else(|| fallback_urls.first().cloned().unwrap_or_default());
        // FetchOutput.status is the HTTP status code; 2xx counts as success.
        let success = result_json
            .get("status")
            .and_then(|v| v.as_u64())
            .map(|c| (200..300).contains(&c))
            .unwrap_or(true);
        api::message::web_fetch::status::Type::Success(api::message::web_fetch::status::Success {
            pages: vec![api::message::web_fetch::status::success::FetchedPage {
                url,
                title: String::new(),
                success,
            }],
        })
    };
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::WebFetch(api::message::WebFetch {
            status: Some(api::message::web_fetch::Status {
                r#type: Some(r#type),
            }),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

fn make_tool_call_result_message(
    task_id: &str,
    request_id: &str,
    tool_call_id: String,
    content: String,
) -> api::Message {
    // ToolCallResult persistence: the `tool_call_result.result` oneof of the warp protobuf is all
    // structured variants (RunShellCommand / Grep / ReadFiles / ...), with no generic string
    // fallback variant. BYOP already serializes the result into a JSON string in chat_stream itself,
    // so there is no need to structure it per the warp protocol; just store the string in the free string field
    // `server_message_data` and leave the `result` oneof as None. In the next turn's build_chat_request the
    // `Message::ToolCallResult` branch needs a special case: when result=None read content from server_message_data
    // (otherwise go through tools::serialize_result to deserialize the structured variant).
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: content,
        citations: vec![],
        message: Some(api::message::Message::ToolCallResult(
            api::message::ToolCallResult {
                tool_call_id,
                context: None,
                result: None,
            },
        )),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// When BYOP `from_args` parsing fails, emit a placeholder ToolCall as carrier:
/// the `tool` oneof is left None (no suitable structured variant), and the original fn_name + args_str are encoded into
/// `server_message_data` as `<fn_name>\n<args_str>`. Next turn's build_chat_request →
/// the carrier branch of `serialize_outgoing_tool_call` restores from it, ensuring the upstream model sees
/// tool_use name / args consistent with the original call (otherwise a "warp_internal_empty" placeholder would
/// confuse the model and would not match the error context of the ToolCallResult that follows).
fn make_tool_call_carrier_message(
    task_id: &str,
    request_id: &str,
    tool_call_id: &str,
    fn_name: &str,
    args_str: &str,
) -> api::Message {
    let carrier = format!("{}\n{}", fn_name, args_str);
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: carrier,
        citations: vec![],
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: tool_call_id.to_owned(),
            tool: None,
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

pub(crate) fn make_tool_call_message(
    task_id: &str,
    request_id: &str,
    tool_call_id: &str,
    tool: api::message::tool_call::Tool,
) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: tool_call_id.to_owned(),
            tool: Some(tool),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

pub(crate) fn create_task_event(task_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::CreateTask(
                        api::client_action::CreateTask {
                            task: Some(api::Task {
                                id: task_id.to_owned(),
                                description: String::new(),
                                dependencies: None,
                                messages: vec![],
                                summary: String::new(),
                                server_data: String::new(),
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

/// Builds an `Action::CreateTask` representing a new subtask, with `dependencies.parent_task_id`.
/// When conversation sees non-empty `task.parent_id()` in `apply_client_action::CreateTask` →
/// it goes through the `Task::new_subtask` path, finds the matching subagent tool_call from parent.messages,
/// extracts `SubagentParams` to attach to the subtask, and emits `BlocklistAIHistoryEvent::CreatedSubtask`.
/// The LRC tag-in floating-window spawn chain depends on this event.
fn create_subtask_event(subtask_id: &str, parent_task_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::CreateTask(
                        api::client_action::CreateTask {
                            task: Some(api::Task {
                                id: subtask_id.to_owned(),
                                description: String::new(),
                                dependencies: Some(api::task::Dependencies {
                                    parent_task_id: parent_task_id.to_owned(),
                                }),
                                messages: vec![],
                                summary: String::new(),
                                server_data: String::new(),
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

pub(crate) fn make_finished_done(
    usage_metadata: Option<api::response_event::stream_finished::ConversationUsageMetadata>,
) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                conversation_usage_metadata: usage_metadata,
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
            },
        )),
    }
}

#[cfg(test)]
mod assistant_buffer_tests {
    use super::*;
    use genai::chat::{ChatRole, ToolCall};

    fn reasoning_part(msg: &ChatMessage) -> Option<&str> {
        for p in msg.content.parts() {
            if let ContentPart::ReasoningContent(r) = p {
                return Some(r.as_str());
            }
        }
        None
    }

    /// gate=false + real reasoning → **dropped** (the fix for zerx-lab/warp #25).
    /// OpenAI-strict providers like Cerebras / Groq / OpenRouter return 400 on seeing the field.
    #[test]
    fn no_echo_drops_real_reasoning_text() {
        let mut buf = AssistantBuffer::new(false);
        buf.text = Some("Hi".to_string());
        buf.reasoning = Some("internal thought".to_string());
        let mut msgs = Vec::new();
        buf.flush_into(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, ChatRole::Assistant);
        assert!(
            reasoning_part(&msgs[0]).is_none(),
            "must not echo reasoning"
        );
    }

    /// gate=false + tool_calls + real reasoning → the tool_calls message also carries no reasoning.
    #[test]
    fn no_echo_drops_reasoning_on_tool_calls_message() {
        let mut buf = AssistantBuffer::new(false);
        buf.text = Some("calling".to_string());
        buf.tool_calls = vec![ToolCall {
            call_id: "c1".to_string(),
            fn_name: "echo".to_string(),
            fn_arguments: serde_json::json!({}),
            thought_signatures: None,
        }];
        buf.reasoning = Some("planning".to_string());
        let mut msgs = Vec::new();
        buf.flush_into(&mut msgs);
        assert_eq!(msgs.len(), 2, "text + tool_calls flush into two messages");
        for m in &msgs {
            assert!(
                reasoning_part(m).is_none(),
                "any-msg reasoning must be absent"
            );
        }
    }

    /// gate=true + real reasoning → attach the real value (DeepSeek / Kimi path).
    #[test]
    fn echo_keeps_real_reasoning() {
        let mut buf = AssistantBuffer::new(true);
        buf.text = Some("ok".to_string());
        buf.reasoning = Some("thinking...".to_string());
        let mut msgs = Vec::new();
        buf.flush_into(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(reasoning_part(&msgs[0]), Some("thinking..."));
    }

    /// gate=true + no reasoning → attach a placeholder (satisfies the "field must exist" validation).
    #[test]
    fn echo_inserts_placeholder_when_empty() {
        let mut buf = AssistantBuffer::new(true);
        buf.text = Some("ok".to_string());
        buf.reasoning = None;
        let mut msgs = Vec::new();
        buf.flush_into(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(reasoning_part(&msgs[0]), Some(REASONING_ECHO_PLACEHOLDER));
    }

    /// gate=true + tool_calls + real reasoning → the text message gets the placeholder, the tool_calls message gets the real value.
    #[test]
    fn echo_with_tool_calls_splits_correctly() {
        let mut buf = AssistantBuffer::new(true);
        buf.text = Some("calling".to_string());
        buf.tool_calls = vec![ToolCall {
            call_id: "c1".to_string(),
            fn_name: "echo".to_string(),
            fn_arguments: serde_json::json!({}),
            thought_signatures: None,
        }];
        buf.reasoning = Some("plan".to_string());
        let mut msgs = Vec::new();
        buf.flush_into(&mut msgs);
        assert_eq!(msgs.len(), 2);
        // text message: placeholder
        assert_eq!(reasoning_part(&msgs[0]), Some(REASONING_ECHO_PLACEHOLDER));
        // tool_calls message: real reasoning + contains ToolCall part
        assert_eq!(reasoning_part(&msgs[1]), Some("plan"));
        assert!(
            !msgs[1].content.tool_calls().is_empty(),
            "second message must carry tool_calls"
        );
    }
}

#[cfg(test)]
mod dashscope_thinking_tests {
    use super::*;
    use crate::settings::ReasoningEffortSetting as R;

    const DASHSCOPE_CN: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/";
    const DASHSCOPE_INTL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/";

    #[test]
    fn dashscope_qwen3_triggers() {
        assert!(dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "qwen3-235b-a22b",
            R::High
        ));
    }

    #[test]
    fn dashscope_qwq_triggers() {
        assert!(dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_INTL,
            "qwq-32b",
            R::Medium
        ));
    }

    #[test]
    fn dashscope_deepseek_r1_triggers() {
        assert!(dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "deepseek-r1",
            R::High
        ));
    }

    #[test]
    fn dashscope_kimi_k2_thinking_excluded() {
        // opencode comment: kimi-k2-thinking is on by default, do not inject again
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "kimi-k2-thinking",
            R::High
        ));
    }

    #[test]
    fn dashscope_off_setting_skips() {
        // Respect the user actively turning thinking off
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "qwen3-30b",
            R::Off
        ));
    }

    #[test]
    fn dashscope_non_reasoning_model_skips() {
        // Pure chat models like qwen-turbo / qwen2.5 should not be injected
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "qwen-turbo",
            R::High
        ));
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            DASHSCOPE_CN,
            "qwen2.5-72b",
            R::High
        ));
    }

    #[test]
    fn non_dashscope_url_skips() {
        // base_urls that are not DashScope, such as OpenAI / Cerebras / Groq
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            "https://api.openai.com/v1/",
            "qwen3-30b",
            R::High
        ));
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::OpenAi,
            "https://api.cerebras.ai/v1/",
            "qwen3-30b",
            R::High
        ));
    }

    #[test]
    fn non_openai_api_type_skips() {
        // Anthropic / Gemini / DeepSeek api_types do not take this path
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::Anthropic,
            DASHSCOPE_CN,
            "qwen3-30b",
            R::High
        ));
        assert!(!dashscope_needs_enable_thinking(
            AgentProviderApiType::DeepSeek,
            DASHSCOPE_CN,
            "deepseek-r1",
            R::High
        ));
    }
}

/// Regression tests for "thinking-depth tier dispatch" in `build_chat_options`.
///
/// Aligned with how Zed `LanguageModelRequest::thinking_allowed=false` is handled across providers:
/// **When Off, no provider may let the server think**. The concrete strategy differs per provider:
/// - Anthropic / Gemini: send no thinking field (skip `with_reasoning_effort`)
/// - DeepSeek: `extra_body.thinking.type=disabled` (on by default server-side, must be explicitly turned off)
/// - OpenAI / OpenAiResp: `reasoning_effort: "none"` (accepted by GPT-5)
#[cfg(test)]
mod build_chat_options_off_tests {
    use super::*;
    use crate::settings::ReasoningEffortSetting as R;
    use genai::chat::ReasoningEffort as GE;

    fn opts(api_type: AgentProviderApiType, model: &str, effort: R) -> genai::chat::ChatOptions {
        build_chat_options(
            api_type,
            "https://example.com/v1/",
            model,
            effort,
            vec![],
            None,
        )
    }

    /// claude-sonnet-4-6 (matched by `SUPPORT_ADAPTTIVE_THINK_MODELS`) + Off must **completely
    /// omit** `reasoning_effort`, otherwise the vendor genai adapter would insert it unconditionally
    /// `thinking:{type:adaptive}`(`adapter_impl.rs:121-135`)。
    #[test]
    fn anthropic_sonnet_4_6_off_skips_reasoning_effort() {
        let o = opts(AgentProviderApiType::Anthropic, "claude-sonnet-4-6", R::Off);
        assert!(
            o.reasoning_effort.is_none(),
            "Anthropic+Off must not send reasoning_effort, to avoid the 4.6 family forcibly inserting adaptive thinking"
        );
        assert!(
            o.extra_body.is_none(),
            "Anthropic+Off should not inject extra_body either"
        );
    }

    /// Same as claude-opus-4-6 (double hit: SUPPORT_EFFORT + SUPPORT_ADAPTIVE).
    #[test]
    fn anthropic_opus_4_6_off_skips_reasoning_effort() {
        let o = opts(AgentProviderApiType::Anthropic, "claude-opus-4-6", R::Off);
        assert!(o.reasoning_effort.is_none());
        assert!(o.extra_body.is_none());
    }

    /// claude-opus-4-7+ + Off: although not on the adaptive list (already OK), it should still be skipped consistently.
    #[test]
    fn anthropic_opus_4_7_off_skips_reasoning_effort() {
        let o = opts(AgentProviderApiType::Anthropic, "claude-opus-4-7", R::Off);
        assert!(o.reasoning_effort.is_none());
        assert!(o.extra_body.is_none());
    }

    /// Anthropic + High still takes the original reasoning_effort path.
    #[test]
    fn anthropic_high_injects_reasoning_effort() {
        let o = opts(AgentProviderApiType::Anthropic, "claude-opus-4-7", R::High);
        assert!(matches!(o.reasoning_effort, Some(GE::High)));
    }

    /// Anthropic + Auto sends no parameters.
    #[test]
    fn anthropic_auto_skips() {
        let o = opts(AgentProviderApiType::Anthropic, "claude-opus-4-7", R::Auto);
        assert!(o.reasoning_effort.is_none());
    }

    /// Gemini + Off: no thinkingConfig sent.
    #[test]
    fn gemini_off_skips_reasoning_effort() {
        let o = opts(AgentProviderApiType::Gemini, "gemini-2.5-pro", R::Off);
        assert!(o.reasoning_effort.is_none());
        assert!(o.extra_body.is_none());
    }

    /// Gemini + Medium takes the thinkingBudget path.
    #[test]
    fn gemini_medium_injects_reasoning_effort() {
        let o = opts(AgentProviderApiType::Gemini, "gemini-2.5-pro", R::Medium);
        assert!(matches!(o.reasoning_effort, Some(GE::Medium)));
    }

    /// DeepSeek + Off: must send `extra_body.thinking.type=disabled`,
    /// and **must not** take reasoning_effort=none (server 400 unknown variant).
    #[test]
    fn deepseek_off_uses_extra_body_disabled() {
        let o = opts(AgentProviderApiType::DeepSeek, "deepseek-v4-flash", R::Off);
        assert!(
            o.reasoning_effort.is_none(),
            "DeepSeek+Off must not take reasoning_effort=none"
        );
        let body = o.extra_body.as_ref().expect("extra_body must be set");
        assert_eq!(
            body.pointer("/thinking/type"),
            Some(&serde_json::Value::String("disabled".to_string())),
            "DeepSeek+Off must send thinking.type=disabled"
        );
    }

    /// DeepSeek + High takes the top-level reasoning_effort field.
    #[test]
    fn deepseek_high_injects_reasoning_effort() {
        let o = opts(AgentProviderApiType::DeepSeek, "deepseek-reasoner", R::High);
        assert!(matches!(o.reasoning_effort, Some(GE::High)));
        assert!(o.extra_body.is_none());
    }

    /// OpenAI (GPT-5) + Off: takes reasoning_effort=none (GPT-5 accepts the `none` tier).
    #[test]
    fn openai_gpt5_off_uses_reasoning_effort_none() {
        let o = opts(AgentProviderApiType::OpenAi, "gpt-5", R::Off);
        assert!(
            matches!(o.reasoning_effort, Some(GE::None)),
            "OpenAI+GPT-5+Off should send reasoning_effort=none"
        );
    }

    /// Models not supporting reasoning + any non-Auto tier: skip (avoiding upstream 400).
    #[test]
    fn anthropic_haiku_3_5_off_skips() {
        let o = opts(
            AgentProviderApiType::Anthropic,
            "claude-3-5-haiku-20241022",
            R::Off,
        );
        assert!(o.reasoning_effort.is_none());
        assert!(o.extra_body.is_none());
    }

    #[test]
    fn openai_gpt4o_off_skips() {
        // gpt-4o is not on the reasoning list; Off is skipped too
        let o = opts(AgentProviderApiType::OpenAi, "gpt-4o", R::Off);
        assert!(o.reasoning_effort.is_none());
    }
}

#[cfg(test)]
mod adapter_routing_tests {
    use super::*;
    use genai::adapter::AdapterKind;

    const OPENAI_HOST: &str = "https://api.openai.com/v1/";
    const ANTHROPIC_HOST: &str = "https://api.anthropic.com/v1/";
    const OPENROUTER_HOST: &str = "https://openrouter.ai/api/v1/";

    #[test]
    fn openai_gpt54_auto_upgrades_to_responses_api() {
        assert_eq!(
            effective_adapter_kind_for(AgentProviderApiType::OpenAi, "gpt-5.4", OPENAI_HOST),
            AdapterKind::OpenAIResp
        );
    }

    #[test]
    fn openai_gpt4o_stays_on_chat_completions() {
        assert_eq!(
            effective_adapter_kind_for(AgentProviderApiType::OpenAi, "gpt-4o", OPENAI_HOST),
            AdapterKind::OpenAI
        );
    }

    #[test]
    fn openai_resp_api_type_unchanged() {
        assert_eq!(
            effective_adapter_kind_for(AgentProviderApiType::OpenAiResp, "gpt-5.4", OPENAI_HOST),
            AdapterKind::OpenAIResp
        );
    }

    #[test]
    fn openai_claude_on_anthropic_host_auto_upgrades() {
        assert_eq!(
            effective_adapter_kind_for(
                AgentProviderApiType::OpenAi,
                "claude-sonnet-4-6",
                ANTHROPIC_HOST
            ),
            AdapterKind::Anthropic
        );
    }

    #[test]
    fn openai_claude_on_openrouter_stays_openai() {
        assert_eq!(
            effective_adapter_kind_for(
                AgentProviderApiType::OpenAi,
                "anthropic/claude-sonnet-4-6",
                OPENROUTER_HOST
            ),
            AdapterKind::OpenAI
        );
    }

    #[test]
    fn openai_gpt54_on_relay_host_stays_on_chat_completions() {
        // Chat-completions-only relays such as one-api / LiteLLM keep the official model id,
        // auto-upgrading to the Responses API must only apply to the official host, otherwise 404.
        assert_eq!(
            effective_adapter_kind_for(AgentProviderApiType::OpenAi, "gpt-5.4", OPENROUTER_HOST),
            AdapterKind::OpenAI
        );
    }

    #[test]
    fn anthropic_api_type_unchanged() {
        assert_eq!(
            effective_adapter_kind_for(
                AgentProviderApiType::Anthropic,
                "claude-opus-4-7",
                ANTHROPIC_HOST
            ),
            AdapterKind::Anthropic
        );
    }
}

#[cfg(test)]
mod anthropic_endpoint_tests {
    use super::*;

    #[test]
    fn host_only_gets_v1_prefix() {
        assert_eq!(
            normalize_endpoint_url(AgentProviderApiType::Anthropic, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1/"
        );
    }

    #[test]
    fn strips_trailing_messages_path() {
        assert_eq!(
            normalize_endpoint_url(
                AgentProviderApiType::Anthropic,
                "https://api.anthropic.com/v1/messages"
            ),
            "https://api.anthropic.com/v1/"
        );
    }

    #[test]
    fn strips_messages_even_when_api_type_openai_but_host_is_anthropic() {
        assert_eq!(
            normalize_endpoint_url(
                AgentProviderApiType::OpenAi,
                "https://api.anthropic.com/v1/messages"
            ),
            "https://api.anthropic.com/v1/"
        );
    }
}

#[cfg(test)]
mod ollama_endpoint_tests {
    use super::*;

    #[test]
    fn legacy_v1_default_base_url_is_rescued() {
        // The historical default_base_url was once `http://localhost:11434/v1/` and has been persisted into existing
        // settings; normalize must strip `/v1` back to the host root, otherwise it joins into `/v1/api/chat` 404.
        assert_eq!(
            normalize_endpoint_url(AgentProviderApiType::Ollama, "http://localhost:11434/v1/"),
            "http://localhost:11434/"
        );
    }

    #[test]
    fn host_only_passes_through_with_trailing_slash() {
        assert_eq!(
            normalize_endpoint_url(AgentProviderApiType::Ollama, "http://localhost:11434"),
            "http://localhost:11434/"
        );
    }

    #[test]
    fn custom_path_is_preserved() {
        assert_eq!(
            normalize_endpoint_url(AgentProviderApiType::Ollama, "http://box:11434/ollama"),
            "http://box:11434/ollama/"
        );
    }
}

/// **End-to-end cache boundary stability tests**: verify, under multi-turn conversation simulation, the guarantee of "byte-level prefix identity"
/// that prompt cache needs. These tests do not call the upstream API, only check the determinism of the output of
/// `apply_caching_anthropic` and `build_chat_options`.
///
/// This is the **minimum bar** for cache hits: if the same input yields different output across calls,
/// the upstream hash is bound to differ → 100% miss. Conversely, identical output does not guarantee a hit.
#[cfg(test)]
mod cache_boundary_stability_tests {
    use super::*;
    use genai::chat::{ChatMessage, ChatRole};

    /// Builds a typical multi-turn conversation messages sequence:
    /// system + user_1 + assistant_1 + user_2 + assistant_2 + user_3
    /// (ends with user, consistent with the `ensure_ends_with_user` output).
    fn build_three_turn_conversation() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system(
                "You are a helpful coding assistant for Warp BYOP.\n\
                 Guidelines: be concise, prefer code over prose.",
            ),
            ChatMessage::user("What is rust borrow checker?"),
            ChatMessage::assistant("It enforces ownership rules at compile time."),
            ChatMessage::user("Show me a code example"),
            ChatMessage::assistant("```rust\nfn main() { let s = String::new(); }\n```"),
            ChatMessage::user("Explain the lifetime in that code"),
        ]
    }

    fn extract_cache_control(msg: &ChatMessage) -> Option<CacheControl> {
        // The cache_control of ChatMessage is on `options.cache_control`.
        msg.options.as_ref().and_then(|o| o.cache_control.clone())
    }

    fn cache_signature(msgs: &[ChatMessage]) -> Vec<(usize, ChatRole, Option<CacheControl>)> {
        msgs.iter()
            .enumerate()
            .map(|(i, m)| (i, m.role.clone(), extract_cache_control(m)))
            .collect()
    }

    /// **P0-4 main acceptance**: apply_caching_anthropic called repeatedly on the same input
    /// must produce byte-equal cache marker positions and TTLs.
    #[test]
    fn apply_caching_anthropic_is_deterministic() {
        let mut a = build_three_turn_conversation();
        let mut b = build_three_turn_conversation();
        apply_caching_anthropic(&mut a);
        apply_caching_anthropic(&mut b);
        assert_eq!(
            cache_signature(&a),
            cache_signature(&b),
            "same input × multiple calls must produce consistent cache markers"
        );
    }

    /// **TTL mixed-strategy acceptance**: system uses 1h (static prefix), non-system uses 5m (conversation tail).
    /// Order system(1h) → messages(5m) satisfies Anthropic's ordering constraint and is immune to externally injected 5m.
    #[test]
    fn anthropic_cache_uses_mixed_ttl() {
        let mut msgs = build_three_turn_conversation();
        apply_caching_anthropic(&mut msgs);
        let tagged: Vec<_> = msgs
            .iter()
            .filter(|m| extract_cache_control(m).is_some())
            .collect();
        assert!(!tagged.is_empty(), "at least one breakpoint must be tagged");
        for m in &tagged {
            let cc = extract_cache_control(m).unwrap();
            let expected = if matches!(m.role, ChatRole::System) {
                CacheControl::Ephemeral1h
            } else {
                CacheControl::Ephemeral5m
            };
            assert_eq!(cc, expected, "TTL for role={:?} does not match expectation", m.role);
        }
    }

    /// **P0-4 coverage acceptance**: opencode style, first 2 system + last 2 non-system.
    /// A multi-turn conversation (1 system + 5 non-system) should get 1+2=3 markers.
    #[test]
    fn anthropic_marks_first_2_system_and_last_2_non_system() {
        let mut msgs = build_three_turn_conversation();
        apply_caching_anthropic(&mut msgs);
        let tagged_indices: Vec<usize> = msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| extract_cache_control(m).is_some())
            .map(|(i, _)| i)
            .collect();
        // Verify system (idx=0) and the last 2 non-system (idx=4, idx=5) are all tagged.
        assert!(tagged_indices.contains(&0), "first system was not marked");
        assert!(tagged_indices.contains(&4), "second-to-last was not marked");
        assert!(tagged_indices.contains(&5), "last was not marked");
        assert_eq!(
            tagged_indices.len(),
            3,
            "3 breakpoints in total (1 system + 2 tail)"
        );
    }

    /// **Simulating cache prefix stability in multi-turn conversation**:
    /// the messages of turn N are the messages of turn N-1 + (assistant of turn N-1) + (new user).
    /// The cache markers in the initial part (system + middle turns) should not drift as turns grow.
    #[test]
    fn cache_marks_stable_as_conversation_grows() {
        // turn 1
        let mut t1 = vec![ChatMessage::system("sys"), ChatMessage::user("q1")];
        apply_caching_anthropic(&mut t1);
        let sys_t1_cc = extract_cache_control(&t1[0]);

        // turn 2: add assistant_1 + user_2
        let mut t2 = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("q2"),
        ];
        apply_caching_anthropic(&mut t2);
        let sys_t2_cc = extract_cache_control(&t2[0]);

        // The first system's cache_control is consistent across turns → the upstream hash is unchanged → later hits.
        assert_eq!(
            sys_t1_cc, sys_t2_cc,
            "the first system breakpoint's TTL/position should be consistent across turns"
        );
        // The user position of turn 1 was tagged (at the tail); in turn 2 it no longer is.
        assert!(extract_cache_control(&t1[1]).is_some());
        assert!(
            extract_cache_control(&t2[1]).is_none(),
            "turn 2's old user is no longer the tail"
        );
    }

    /// **build_chat_options output determinism** (same input gives the same result across calls).
    /// The minimum bar for prompt cache hits \u{2014} bit-level identical hashes.
    #[test]
    fn openai_chat_options_is_deterministic() {
        use crate::settings::ReasoningEffortSetting as R;
        let make = || {
            build_chat_options(
                AgentProviderApiType::OpenAi,
                "https://api.openai.com/v1/",
                "gpt-5-mini",
                R::Auto,
                vec![],
                Some("conv-abc-123"),
            )
        };
        let a = make();
        let b = make();
        assert_eq!(a.prompt_cache_key, b.prompt_cache_key);
        assert_eq!(a.cache_control, b.cache_control);
    }

    /// **opencode-compatible whitelisted providers** (api.openai.com / *.openai.azure.com /
    /// openrouter.ai / api.venice.ai / opencode.ai/zen) → send prompt_cache_key,
    /// and **never send cache_control** (corresponding to the prompt_cache_retention field;
    /// the opencode repo does not use that field).
    #[test]
    fn whitelisted_provider_emits_prompt_cache_key_only() {
        use crate::settings::ReasoningEffortSetting as R;
        // Pick 5 whitelist representative URLs, each covering the api_type=OpenAi decision branch.
        let whitelisted = [
            "https://api.openai.com/v1/",
            "https://my-resource.openai.azure.com/openai/v1/",
            "https://openrouter.ai/api/v1/",
            "https://api.venice.ai/api/v1/",
            "https://opencode.ai/zen/v1/",
        ];
        for url in whitelisted {
            let opts = build_chat_options(
                AgentProviderApiType::OpenAi,
                url,
                "gpt-5-mini",
                R::Auto,
                vec![],
                Some("conv-1"),
            );
            assert_eq!(
                opts.prompt_cache_key.as_deref(),
                Some("conv-1"),
                "{url}: a whitelisted provider should send prompt_cache_key=conversation_id"
            );
            let expected = url
                .contains("api.openai.com")
                .then_some(CacheControl::Ephemeral24h);
            assert_eq!(
                opts.cache_control, expected,
                "{url}: only official api.openai.com sends 24h cache_control"
            );
        }
    }

    /// **#126 regression**: OpenAi api_type but base_url not on the whitelist (OpenCode Go
    /// relaying Kimi / GLM, vLLM, lm-studio, DashScope, Moonshot, Zhipu native, etc.)
    /// → neither cache_control nor prompt_cache_key is sent.
    ///
    /// Aligned with the opencode `options()` function: apart from the five providerIDs openai/azure/openrouter/venice/opencode,
    /// no provider sets promptCacheKey.
    #[test]
    fn non_whitelisted_provider_emits_nothing() {
        use crate::settings::ReasoningEffortSetting as R;
        // The two concrete examples from the issue #126 body + the user's follow-up comment, plus other mainstream
        // OpenAI-compatible relays. None of these URLs should send any cache field.
        let byop_urls = [
            ("https://opencode.go/v1/", "kimi-k2.6"),
            ("https://opencode.go/v1/", "glm-5.1"),
            ("https://api.moonshot.cn/v1/", "kimi-k2"),
            ("https://open.bigmodel.cn/api/paas/v4/", "glm-4.6"),
            (
                "https://dashscope.aliyuncs.com/compatible-mode/v1/",
                "qwen-max",
            ),
            ("http://localhost:1234/v1/", "local-model"),
        ];
        for (url, model) in byop_urls {
            let opts = build_chat_options(
                AgentProviderApiType::OpenAi,
                url,
                model,
                R::Auto,
                vec![],
                Some("conv-byop"),
            );
            assert!(
                opts.cache_control.is_none(),
                "{url}: non-whitelisted should not send cache_control"
            );
            assert!(
                opts.prompt_cache_key.is_none(),
                "{url}: non-whitelisted should not send prompt_cache_key"
            );
        }
    }

    /// The OpenAiResp api_type goes through the same decision logic (the genai openai_resp adapter serializes
    /// the same fields); whitelisted ones send, non-whitelisted are blocked.
    #[test]
    fn openai_resp_follows_same_whitelist() {
        use crate::settings::ReasoningEffortSetting as R;
        let on_whitelist = build_chat_options(
            AgentProviderApiType::OpenAiResp,
            "https://api.openai.com/v1/",
            "gpt-5",
            R::Auto,
            vec![],
            Some("conv-resp"),
        );
        assert_eq!(on_whitelist.prompt_cache_key.as_deref(), Some("conv-resp"));
        assert_eq!(on_whitelist.cache_control, Some(CacheControl::Ephemeral24h));

        let off_whitelist = build_chat_options(
            AgentProviderApiType::OpenAiResp,
            "https://custom.relay/v1/",
            "gpt-5",
            R::Auto,
            vec![],
            Some("conv-resp"),
        );
        assert!(off_whitelist.prompt_cache_key.is_none());
        assert!(off_whitelist.cache_control.is_none());
    }

    /// **An empty conversation_id skips prompt_cache_key** (avoiding misattached routing across conversations).
    #[test]
    fn openai_empty_conversation_id_skips_cache_key() {
        use crate::settings::ReasoningEffortSetting as R;
        let opts = build_chat_options(
            AgentProviderApiType::OpenAi,
            "https://api.openai.com/v1/",
            "gpt-5",
            R::Auto,
            vec![],
            Some(""),
        );
        assert!(
            opts.prompt_cache_key.is_none(),
            "an empty conversation_id should skip prompt_cache_key"
        );
        assert_eq!(opts.cache_control, Some(CacheControl::Ephemeral24h));
    }

    #[test]
    fn extended_cache_only_for_supported_models_on_official_endpoint() {
        use crate::settings::ReasoningEffortSetting as R;
        for (model, want) in [("gpt-4o", false), ("gpt-5.5-pro", true), ("o3-mini", true)] {
            let o = build_chat_options(
                AgentProviderApiType::OpenAi,
                "https://api.openai.com/v1/",
                model,
                R::Auto,
                vec![],
                Some("c"),
            );
            assert_eq!(o.cache_control.is_some(), want, "{model}");
        }
    }

    fn opencode_session_headers(
        base_url: &str,
        cid: &str,
        user: Vec<(String, String)>,
    ) -> Vec<(String, String)> {
        use crate::settings::ReasoningEffortSetting as R;
        let opts = build_chat_options(
            AgentProviderApiType::OpenAi,
            base_url,
            "glm-5",
            R::Auto,
            user,
            Some(cid),
        );
        opts.extra_headers
            .map(|h| h.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    #[test]
    fn opencode_session_header_added_only_for_opencode_host() {
        let got = opencode_session_headers("https://opencode.ai/zen/go/v1", "c1", vec![]);
        assert_eq!(got, vec![("x-opencode-session".to_owned(), "c1".to_owned())]);
        assert!(opencode_session_headers("https://example.com/v1", "c1", vec![]).is_empty());
        assert!(opencode_session_headers("https://notopencode.ai/v1", "c1", vec![]).is_empty());
        assert!(opencode_session_headers("https://opencode.ai/zen/go/v1", "", vec![]).is_empty());
        let user = vec![("X-OpenCode-Session".to_owned(), "mine".to_owned())];
        let got = opencode_session_headers("https://opencode.ai/zen/go/v1", "c1", user);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].1, "mine");
    }

    /// **The Anthropic path build_chat_options does not send cache_control**
    /// (Anthropic uses per-message, not the ChatOptions level).
    #[test]
    fn anthropic_chat_options_no_cache_control() {
        use crate::settings::ReasoningEffortSetting as R;
        let opts = build_chat_options(
            AgentProviderApiType::Anthropic,
            "https://api.anthropic.com/v1/",
            "claude-opus-4-7",
            R::Auto,
            vec![],
            Some("conv-3"),
        );
        assert!(
            opts.cache_control.is_none(),
            "Anthropic's ChatOptions must not carry cache_control (it goes per-message)"
        );
        assert!(
            opts.prompt_cache_key.is_none(),
            "Anthropic does not use prompt_cache_key"
        );
    }

    /// **DeepSeek / Gemini / Ollama use implicit server-side caching and send no cache_control**.
    #[test]
    fn implicit_cache_providers_no_cache_control() {
        use crate::settings::ReasoningEffortSetting as R;
        for api in [
            AgentProviderApiType::DeepSeek,
            AgentProviderApiType::Gemini,
            AgentProviderApiType::Ollama,
        ] {
            let opts = build_chat_options(
                api,
                "https://example.com/v1/",
                "some-model",
                R::Auto,
                vec![],
                Some("conv"),
            );
            assert!(
                opts.cache_control.is_none(),
                "{api:?} should not send cache_control"
            );
        }
    }

    /// The resolved value of `{conversationId}` must enter `ChatOptions.extra_headers` as-is.
    #[test]
    fn resolved_conversation_header_reaches_chat_options() {
        use crate::settings::ReasoningEffortSetting as R;
        let conversation_id =
            AIConversationId::try_from("0b0a0c1e-1111-4222-8333-444455556666".to_owned())
                .expect("valid uuid");
        let headers = super::super::header_vars::resolve_headers(
            &[(
                "x-session-id".to_owned(),
                "conv-{conversationId}".to_owned(),
            )],
            &HeaderVars::default().with_conversation_id(Some(conversation_id)),
        );
        let opts = build_chat_options(
            AgentProviderApiType::OpenAi,
            "https://api.openai.com/v1/",
            "gpt-5-mini",
            R::Auto,
            headers,
            None,
        );
        let value = opts
            .extra_headers
            .expect("resolved headers applied")
            .iter()
            .find(|(name, _)| name.as_str() == "x-session-id")
            .map(|(_, value)| value.clone());
        assert_eq!(value, Some(format!("conv-{conversation_id}")));
    }
}

#[cfg(test)]
mod serializer_readiness_tests {
    use super::*;
    use crate::ai::agent::task::TaskId;
    use crate::ai::agent::{AIAgentActionId, AIAgentActionResultType, RequestCommandOutputResult};
    use crate::ai::byop_compaction::state::{CompactionState, CompletedCompaction};
    use crate::ai::byop_readiness::{
        PendingByopToolResultsError, RepairRecord, RepairState, ToolCallKey, ToolCallRef,
        BLOCKED_BYOP_REQUEST_MESSAGE,
    };
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn kind() -> RedactedToolKind {
        RedactedToolKind::new("shell")
    }

    fn assistant_calls(task_id: &str, assistant_id: &str, call_ids: &[&str]) -> ProjectionItem {
        ProjectionItem::assistant_tool_calls(
            task_id,
            assistant_id,
            call_ids
                .iter()
                .map(|call_id| ProjectedToolCall::new(task_id, assistant_id, *call_id, kind()))
                .collect(),
        )
    }

    fn result(
        task_id: &str,
        message_id: &str,
        assistant_id: Option<&str>,
        call_id: &str,
        source: ToolResultSource,
    ) -> ProjectionItem {
        ProjectionItem::tool_result(ProjectedToolResult::new(
            task_id,
            message_id,
            assistant_id.map(str::to_owned),
            call_id,
            kind(),
            source,
            TerminalResultKind::Real,
        ))
    }

    fn assert_blocked_category(projection: Vec<ProjectionItem>, category: &str) {
        let error = validate_serializer_readiness_projection(projection).unwrap_err();
        assert!(
            error.to_string().contains(BLOCKED_BYOP_REQUEST_MESSAGE),
            "error should use blocked-request copy for {category}: {error}"
        );
    }

    fn shell_tool() -> api::message::tool_call::Tool {
        use api::message::tool_call::run_shell_command::WaitUntilCompleteValue;

        api::message::tool_call::Tool::RunShellCommand(api::message::tool_call::RunShellCommand {
            command: "echo hi".to_owned(),
            is_read_only: true,
            uses_pager: false,
            is_risky: false,
            citations: vec![],
            wait_until_complete_value: Some(WaitUntilCompleteValue::WaitUntilComplete(true)),
            risk_category: 0,
        })
    }

    fn subagent_tool() -> api::message::tool_call::Tool {
        api::message::tool_call::Tool::Subagent(api::message::tool_call::Subagent {
            task_id: "subtask-1".to_owned(),
            payload: String::new(),
            metadata: Some(api::message::tool_call::subagent::Metadata::Cli(
                api::message::tool_call::subagent::CliSubagent {
                    command_id: "command-1".to_owned(),
                },
            )),
        })
    }

    fn task_with_messages(messages: Vec<api::Message>) -> api::Task {
        api::Task {
            id: "task-1".to_owned(),
            messages,
            dependencies: None,
            description: String::new(),
            summary: String::new(),
            server_data: String::new(),
        }
    }

    fn user_query_input(query: &str) -> AIAgentInput {
        AIAgentInput::UserQuery {
            query: query.to_owned(),
            context: Arc::<[AIAgentContext]>::from([]),
            static_query_type: None,
            referenced_attachments: HashMap::new(),
            user_query_mode: UserQueryMode::default(),
            running_command: None,
            intended_agent: None,
        }
    }

    fn cancelled_action_result_input(call_id: &str) -> AIAgentInput {
        AIAgentInput::ActionResult {
            result: AIAgentActionResult {
                id: AIAgentActionId::from(call_id.to_owned()),
                task_id: TaskId::new("task-1".to_owned()),
                result: AIAgentActionResultType::RequestCommandOutput(
                    RequestCommandOutputResult::CancelledBeforeExecution,
                ),
            },
            context: Arc::<[AIAgentContext]>::from([]),
        }
    }

    fn request_params(messages: Vec<api::Message>, input: Vec<AIAgentInput>) -> RequestParams {
        RequestParams::new_for_test(input, vec![task_with_messages(messages)])
    }

    fn request_params_with_repair(
        messages: Vec<api::Message>,
        input: Vec<AIAgentInput>,
        repair_state: RepairState,
    ) -> RequestParams {
        let mut params = request_params(messages, input);
        params.byop_repair_state = RepairStateStatus::Valid(repair_state);
        params
    }

    fn build_openai_request(params: &RequestParams) -> Result<ChatRequest, ConvertToAPITypeError> {
        build_chat_request(
            params,
            false,
            AgentProviderApiType::OpenAi,
            attachment_caps::AttachmentCaps::default(),
        )
    }

    fn assert_request_has_no_repair_placeholder(request: &ChatRequest) {
        assert!(
            !request.messages.iter().any(|message| {
                message
                    .content
                    .tool_responses()
                    .iter()
                    .any(|response| is_placeholder_tool_response_content(&response.content))
            }),
            "normal request body must not emit placeholder tool results"
        );
    }

    fn tool_response_contents(request: &ChatRequest, call_id: &str) -> Vec<String> {
        request
            .messages
            .iter()
            .flat_map(|message| message.content.tool_responses())
            .filter(|response| response.call_id == call_id)
            .map(|response| response.content.clone())
            .collect()
    }

    fn assert_build_request_blocked(params: RequestParams, category: &str) {
        let error = build_openai_request(&params).unwrap_err();
        assert!(
            error.to_string().contains(BLOCKED_BYOP_REQUEST_MESSAGE),
            "error should use blocked-request copy for {category}: {error}"
        );
    }

    #[test]
    fn placeholder_then_cancellation_regression_blocks_waits_or_sends_real_cancellation() {
        let user_message = make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]);
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let assistant_message_id = tool_call_message.id.clone();

        assert_build_request_blocked(
            request_params(
                vec![user_message.clone(), tool_call_message.clone()],
                vec![user_query_input("continue")],
            ),
            "MissingResultWithoutRepairSource",
        );

        let pending_report = classify_byop_controller_readiness_with_live_tool_calls(
            &request_params(
                vec![user_message.clone(), tool_call_message.clone()],
                vec![user_query_input("continue")],
            ),
            vec![LiveToolCall::new(
                ToolCallRef::new(
                    ToolCallKey::new("task-1", assistant_message_id, "call-1"),
                    kind(),
                ),
                LiveToolCallState::Running,
            )],
        );
        assert!(matches!(
            pending_report.state,
            ReadinessState::PendingToolResults { ref tool_calls }
                if tool_calls.len() == 1 && tool_calls[0].key.tool_call_id == "call-1"
        ));
        let ReadinessState::PendingToolResults { tool_calls } = pending_report.state else {
            panic!("expected pending tool results");
        };
        let pending_error = PendingByopToolResultsError::new(tool_calls.len());
        assert_eq!(
            pending_error.category(),
            ReadinessCategory::PendingToolResults
        );
        assert_eq!(pending_error.tool_call_count(), 1);
        assert!(
            !pending_error
                .to_string()
                .contains(BLOCKED_BYOP_REQUEST_MESSAGE),
            "pending wait must remain distinct from blocked user-facing errors"
        );

        let request = build_openai_request(&request_params(
            vec![user_message, tool_call_message],
            vec![
                cancelled_action_result_input("call-1"),
                user_query_input("continue"),
            ],
        ))
        .expect("current cancellation result should serialize as a real tool result");
        let errors = strict_chat_completions_ordering_errors(&request.messages);
        assert!(
            errors.is_empty(),
            "request body ordering errors: {errors:?}"
        );
        assert_request_has_no_repair_placeholder(&request);

        let contents = tool_response_contents(&request, "call-1");
        assert_eq!(contents.len(), 1);
        assert!(
            contents[0].to_ascii_lowercase().contains("cancel"),
            "expected real cancellation content, got {}",
            contents[0]
        );
    }

    /// Regression for issues #147 / #222: when the user interrupts the AI during tool execution, the interrupted tool call never produces a result due to cancel
    /// races / the future being dropped, leaving an orphan gap with "a tool_call but no tool_result".
    /// Previously readiness judged it MissingResultWithoutRepairSource and blocked the conversation permanently. After the fix, the controller
    /// preflight synthesizes a cancellation placeholder ToolCallResult for such a gap (server_message_data carrying
    /// {"status":"cancelled",...}, result oneof left None) and lands it in history. This test verifies that once the synthesized message
    /// enters history, readiness turns Ready and the request serializes normally, and the conversation no longer hangs.
    #[test]
    fn synthesized_cancellation_result_unblocks_interrupted_tool_call() {
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());

        // When the gap exists it must block (the hang scenario before the regression).
        assert_build_request_blocked(
            request_params(
                vec![tool_call_message.clone()],
                vec![user_query_input("continue")],
            ),
            "MissingResultWithoutRepairSource",
        );

        // After the synthesized cancellation message lands in history, the gap is filled → Ready.
        let synthesized = make_tool_call_result_message(
            "task-1",
            "req-1",
            "call-1".to_owned(),
            serde_json::json!({
                "status": "cancelled",
                "reason": "interrupted_by_user",
            })
            .to_string(),
        );
        let params = request_params(
            vec![tool_call_message, synthesized],
            vec![user_query_input("continue")],
        );
        assert!(matches!(
            classify_byop_controller_readiness(&params).state,
            ReadinessState::Ready
        ));
        let request = build_openai_request(&params)
            .expect("synthesized cancellation should let the request serialize");
        let contents = tool_response_contents(&request, "call-1");
        assert_eq!(contents.len(), 1);
        assert!(
            contents[0].to_ascii_lowercase().contains("cancel"),
            "expected cancellation content, got {}",
            contents[0]
        );
    }

    #[test]
    fn controller_readiness_requires_cancellation_commit_before_user_boundary() {
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let params = request_params(
            vec![tool_call_message],
            vec![
                cancelled_action_result_input("call-1"),
                user_query_input("continue"),
            ],
        );

        let report = classify_byop_controller_readiness(&params);

        assert!(matches!(
            report.state,
            ReadinessState::NeedsCancellationCommit { ref tool_calls }
                if tool_calls.len() == 1
                    && tool_calls[0].key.tool_call_id == "call-1"
        ));
    }

    #[test]
    fn controller_readiness_ignores_duplicate_current_cancellation_after_persistence() {
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let tool_result_message = make_tool_call_result_message(
            "task-1",
            "req-1",
            "call-1".to_owned(),
            r#"{"status":"cancelled"}"#.to_owned(),
        );
        let params = request_params(
            vec![tool_call_message, tool_result_message],
            vec![
                cancelled_action_result_input("call-1"),
                user_query_input("continue"),
            ],
        );

        let report = classify_byop_controller_readiness(&params);

        assert!(matches!(report.state, ReadinessState::Ready));
    }

    #[test]
    fn controller_readiness_accepts_committed_local_interception_results() {
        let carrier =
            make_tool_call_carrier_message("task-1", "req-1", "call-1", "todowrite", "{}");
        let local_result = make_tool_call_result_message(
            "task-1",
            "req-1",
            "call-1".to_owned(),
            r#"{"_byop_intercepted":true,"status":"ok"}"#.to_owned(),
        );
        let invalid_arguments_result = make_tool_call_result_message(
            "task-1",
            "req-1",
            "call-2".to_owned(),
            r#"{"error":"invalid_arguments","tool":"dummy"}"#.to_owned(),
        );
        let invalid_arguments_carrier =
            make_tool_call_carrier_message("task-1", "req-1", "call-2", "dummy", "{}");
        let params = request_params(
            vec![
                carrier,
                local_result,
                invalid_arguments_carrier,
                invalid_arguments_result,
            ],
            vec![],
        );

        let report = classify_byop_controller_readiness(&params);

        assert!(matches!(report.state, ReadinessState::Ready));
    }

    #[test]
    fn controller_readiness_blocks_unreadable_local_interception_payload() {
        let carrier =
            make_tool_call_carrier_message("task-1", "req-1", "call-1", "todowrite", "{}");
        let unreadable_result = make_tool_call_result_message(
            "task-1",
            "req-1",
            "call-1".to_owned(),
            r#"{"_byop_intercepted":true,"status":"ok""#.to_owned(),
        );
        let params = request_params(vec![carrier, unreadable_result], vec![]);

        let report = classify_byop_controller_readiness(&params);

        assert!(matches!(
            report.state,
            ReadinessState::MissingResultWithoutRepairSource {
                reason: crate::ai::byop_readiness::MissingResultReason::UnreadableLocalInterception,
                ..
            }
        ));
    }

    #[test]
    fn controller_readiness_reports_pending_live_action() {
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let assistant_message_id = tool_call_message.id.clone();
        let params = request_params(vec![tool_call_message], vec![]);

        let report = classify_byop_controller_readiness_with_live_tool_calls(
            &params,
            vec![LiveToolCall::new(
                ToolCallRef::new(
                    ToolCallKey::new("task-1", assistant_message_id, "call-1"),
                    kind(),
                ),
                LiveToolCallState::Running,
            )],
        );

        assert!(matches!(
            report.state,
            ReadinessState::PendingToolResults { ref tool_calls }
                if tool_calls.len() == 1 && tool_calls[0].key.tool_call_id == "call-1"
        ));
    }

    #[test]
    fn normal_flow_missing_result_blocks_before_placeholder_repair() {
        assert_blocked_category(
            vec![assistant_calls("task-1", "assistant-1", &["call-1"])],
            "MissingResultWithoutRepairSource",
        );
    }

    #[test]
    fn duplicate_orphan_and_out_of_order_results_block_serialization() {
        assert_blocked_category(
            vec![
                assistant_calls("task-1", "assistant-1", &["call-1"]),
                result(
                    "task-1",
                    "result-1",
                    Some("assistant-1"),
                    "call-1",
                    ToolResultSource::PersistedHistory,
                ),
                result(
                    "task-1",
                    "result-2",
                    Some("assistant-1"),
                    "call-1",
                    ToolResultSource::PersistedHistory,
                ),
            ],
            "DuplicateToolResults",
        );
        assert_blocked_category(
            vec![result(
                "task-1",
                "result-1",
                Some("assistant-missing"),
                "call-1",
                ToolResultSource::PersistedHistory,
            )],
            "OrphanToolResult",
        );
        assert_blocked_category(
            vec![
                assistant_calls("task-1", "assistant-1", &["call-1"]),
                ProjectionItem::user_boundary("task-1", "user-2"),
                result(
                    "task-1",
                    "result-1",
                    Some("assistant-1"),
                    "call-1",
                    ToolResultSource::PersistedHistory,
                ),
            ],
            "OutOfOrderToolResult",
        );
    }

    #[test]
    fn visible_boundaries_block_pending_tool_groups_but_filtered_messages_do_not() {
        assert_blocked_category(
            vec![
                assistant_calls("task-1", "assistant-1", &["call-1"]),
                ProjectionItem::other_boundary("task-1", "visible-other"),
            ],
            "MissingResultWithoutRepairSource",
        );

        let report = validate_serializer_readiness_projection(vec![
            assistant_calls("task-1", "assistant-1", &["call-1"]),
            result(
                "task-1",
                "result-1",
                Some("assistant-1"),
                "call-1",
                ToolResultSource::PersistedHistory,
            ),
        ])
        .expect("filtered-out messages should not affect readiness");
        assert_eq!(report.state, ReadinessState::Ready);
    }

    #[test]
    fn current_input_action_result_satisfies_serializer_readiness() {
        let report = validate_serializer_readiness_projection(vec![
            assistant_calls("task-1", "assistant-1", &["call-1"]),
            result(
                "task-1",
                "current_input:0:call-1",
                None,
                "call-1",
                ToolResultSource::CurrentInput,
            ),
        ])
        .expect("current input result should satisfy a visible tool call");
        assert_eq!(report.state, ReadinessState::Ready);
    }

    #[test]
    fn accepted_history_repair_is_sendable_but_distinct_from_ready() {
        let repair = RepairRecord::new(
            RepairSource::ForkedHistory,
            ToolCallKey::new("task-1", "assistant-1", "call-1"),
        );
        let report = validate_serializer_readiness_projection_with_repair_state(
            vec![assistant_calls("task-1", "assistant-1", &["call-1"])],
            &RepairStateStatus::Valid(RepairState::new(vec![repair.clone()])),
            &ReadinessDiagnosticContext::new(
                "test-conversation",
                "test-attempt",
                ReadinessTriggerLayer::SerializerValidation,
            ),
        )
        .expect("accepted repair should be sendable");

        assert_eq!(
            report.state,
            ReadinessState::AcceptedHistoryRepair {
                repairs: vec![AcceptedRepair {
                    record: repair,
                    tool_call: ToolCallRef::new(
                        ToolCallKey::new("task-1", "assistant-1", "call-1"),
                        kind(),
                    ),
                }],
            }
        );
    }

    #[test]
    fn invalid_repair_sidecar_does_not_block_ready_projection() {
        let report = validate_serializer_readiness_projection_with_repair_state(
            vec![
                assistant_calls("task-1", "assistant-1", &["call-1"]),
                result(
                    "task-1",
                    "result-1",
                    Some("assistant-1"),
                    "call-1",
                    ToolResultSource::PersistedHistory,
                ),
            ],
            &RepairStateStatus::from_sidecar_json(Some("{not valid json".to_owned())),
            &ReadinessDiagnosticContext::new(
                "test-conversation",
                "test-attempt",
                ReadinessTriggerLayer::SerializerValidation,
            ),
        )
        .expect("valid real results should not need repair authorization");

        assert_eq!(report.state, ReadinessState::Ready);
    }

    fn make_tool_call(call_id: &str) -> ToolCall {
        ToolCall {
            call_id: call_id.to_owned(),
            fn_name: "dummy".to_owned(),
            fn_arguments: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    fn assistant_with_calls(call_ids: &[&str]) -> ChatMessage {
        ChatMessage::from(
            call_ids
                .iter()
                .map(|call_id| make_tool_call(call_id))
                .collect::<Vec<_>>(),
        )
    }

    fn tool_response(call_id: &str) -> ChatMessage {
        ChatMessage::from(ToolResponse::new(call_id.to_owned(), "{}".to_owned()))
    }

    fn strict_chat_completions_ordering_errors(messages: &[ChatMessage]) -> Vec<String> {
        let mut errors = Vec::new();
        let mut pending_call_ids: Vec<String> = Vec::new();
        let mut seen_tool_results: HashSet<String> = HashSet::new();

        for (idx, msg) in messages.iter().enumerate() {
            if !pending_call_ids.is_empty() && msg.role != ChatRole::Tool {
                errors.push(format!(
                    "message {idx} role {:?} appeared before pending tool responses {:?}",
                    msg.role, pending_call_ids
                ));
            }

            if msg.role == ChatRole::Assistant {
                let tool_call_ids: Vec<String> = msg
                    .content
                    .tool_calls()
                    .iter()
                    .map(|tool_call| tool_call.call_id.clone())
                    .collect();
                if !tool_call_ids.is_empty() {
                    pending_call_ids = tool_call_ids;
                }
            } else if msg.role == ChatRole::Tool {
                let responses = msg.content.tool_responses();
                if pending_call_ids.is_empty() {
                    errors.push(format!("message {idx} is an orphan tool response"));
                }
                for response in responses {
                    if !seen_tool_results.insert(response.call_id.clone()) {
                        errors.push(format!("duplicate tool result id {}", response.call_id));
                    }
                    match pending_call_ids.first() {
                        Some(expected_call_id) if expected_call_id == &response.call_id => {
                            pending_call_ids.remove(0);
                        }
                        Some(expected_call_id) => {
                            if let Some(pos) = pending_call_ids
                                .iter()
                                .position(|call_id| call_id == &response.call_id)
                            {
                                errors.push(format!(
                                    "out-of-order tool result id {} expected {}",
                                    response.call_id, expected_call_id
                                ));
                                pending_call_ids.remove(pos);
                            } else {
                                errors.push(format!("orphan tool result id {}", response.call_id));
                            }
                        }
                        None => {
                            errors.push(format!("orphan tool result id {}", response.call_id));
                        }
                    }
                }
            }
        }

        if !pending_call_ids.is_empty() {
            errors.push(format!(
                "request ended with pending tool responses {:?}",
                pending_call_ids
            ));
        }

        errors
    }

    #[test]
    fn strict_request_body_checker_accepts_ordered_tool_responses() {
        let messages = vec![
            ChatMessage::user("hi"),
            assistant_with_calls(&["a", "b"]),
            tool_response("a"),
            tool_response("b"),
            ChatMessage::user("continue"),
        ];

        assert!(strict_chat_completions_ordering_errors(&messages).is_empty());
    }

    #[test]
    fn strict_request_body_checker_rejects_orphans_duplicates_and_early_boundaries() {
        let orphan = vec![ChatMessage::user("hi"), tool_response("a")];
        assert!(strict_chat_completions_ordering_errors(&orphan)
            .iter()
            .any(|error| error.contains("orphan")));

        let duplicate = vec![
            ChatMessage::user("hi"),
            assistant_with_calls(&["a"]),
            tool_response("a"),
            tool_response("a"),
        ];
        assert!(strict_chat_completions_ordering_errors(&duplicate)
            .iter()
            .any(|error| error.contains("duplicate")));

        let early_boundary = vec![
            ChatMessage::user("hi"),
            assistant_with_calls(&["a"]),
            ChatMessage::user("too soon"),
            tool_response("a"),
        ];
        assert!(strict_chat_completions_ordering_errors(&early_boundary)
            .iter()
            .any(|error| error.contains("before pending")));

        let out_of_order = vec![
            ChatMessage::user("hi"),
            assistant_with_calls(&["a", "b"]),
            tool_response("b"),
            tool_response("a"),
        ];
        assert!(strict_chat_completions_ordering_errors(&out_of_order)
            .iter()
            .any(|error| error.contains("out-of-order")));
    }

    #[test]
    fn build_chat_request_body_rejects_missing_duplicate_orphan_and_out_of_order_history() {
        assert_build_request_blocked(
            request_params(
                vec![
                    make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                    make_tool_call_message("task-1", "req-1", "call-1", shell_tool()),
                ],
                vec![],
            ),
            "MissingResultWithoutRepairSource",
        );

        assert_build_request_blocked(
            request_params(
                vec![
                    make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                    make_tool_call_message("task-1", "req-1", "call-1", shell_tool()),
                    make_tool_call_result_message(
                        "task-1",
                        "req-1",
                        "call-1".to_owned(),
                        r#"{"status":"ok"}"#.to_owned(),
                    ),
                    make_tool_call_result_message(
                        "task-1",
                        "req-1",
                        "call-1".to_owned(),
                        r#"{"status":"ok-again"}"#.to_owned(),
                    ),
                ],
                vec![],
            ),
            "DuplicateToolResults",
        );

        assert_build_request_blocked(
            request_params(
                vec![
                    make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                    make_tool_call_result_message(
                        "task-1",
                        "req-1",
                        "call-1".to_owned(),
                        r#"{"status":"orphan"}"#.to_owned(),
                    ),
                ],
                vec![],
            ),
            "OrphanToolResult",
        );

        assert_build_request_blocked(
            request_params(
                vec![
                    make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                    make_tool_call_message("task-1", "req-1", "call-1", shell_tool()),
                    make_user_query_message("task-1", "req-2", "too soon".to_owned(), &[]),
                    make_tool_call_result_message(
                        "task-1",
                        "req-2",
                        "call-1".to_owned(),
                        r#"{"status":"late"}"#.to_owned(),
                    ),
                ],
                vec![],
            ),
            "OutOfOrderToolResult",
        );
    }

    #[test]
    fn build_chat_request_body_accepts_current_input_tool_result() {
        let params = request_params(
            vec![
                make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                make_tool_call_message("task-1", "req-1", "call-1", shell_tool()),
            ],
            vec![
                cancelled_action_result_input("call-1"),
                user_query_input("continue"),
            ],
        );

        let request = build_openai_request(&params).expect("current input result should serialize");
        let errors = strict_chat_completions_ordering_errors(&request.messages);
        assert!(
            errors.is_empty(),
            "request body ordering errors: {errors:?}"
        );
        assert_request_has_no_repair_placeholder(&request);
    }

    #[test]
    fn build_chat_request_body_ignores_filtered_subagent_tool_call_result() {
        let params = request_params(
            vec![
                make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                make_tool_call_message("task-1", "req-1", "subagent-call-1", subagent_tool()),
                make_tool_call_result_message(
                    "task-1",
                    "req-1",
                    "subagent-call-1".to_owned(),
                    r#"{"status":"spawned"}"#.to_owned(),
                ),
            ],
            vec![],
        );

        let request =
            build_openai_request(&params).expect("filtered subagent result should not block");
        let errors = strict_chat_completions_ordering_errors(&request.messages);
        assert!(
            errors.is_empty(),
            "request body ordering errors: {errors:?}"
        );
        assert!(
            request
                .messages
                .iter()
                .flat_map(|message| message.content.tool_responses())
                .all(|response| response.call_id != "subagent-call-1"),
            "filtered subagent ToolCallResult must not be sent outbound"
        );
    }

    #[test]
    fn build_chat_request_body_emits_structured_repair_placeholder_only_for_accepted_repair() {
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let assistant_message_id = tool_call_message.id.clone();
        let repair = RepairRecord::new(
            RepairSource::ForkedHistory,
            ToolCallKey::new("task-1", assistant_message_id, "call-1"),
        );
        let params = request_params_with_repair(
            vec![
                make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]),
                tool_call_message,
            ],
            vec![],
            RepairState::new(vec![repair]),
        );

        let request = build_openai_request(&params).expect("accepted repair should serialize");
        let response = request
            .messages
            .iter()
            .flat_map(|message| message.content.tool_responses())
            .find(|response| response.call_id == "call-1")
            .expect("repair placeholder response should be present");
        let payload: serde_json::Value =
            serde_json::from_str(&response.content).expect("placeholder should be JSON");
        let object = payload
            .as_object()
            .expect("placeholder should be an object");

        assert_eq!(
            object.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from([
                "status".to_string(),
                "reason".to_string(),
                "note".to_string()
            ])
        );
        assert_eq!(payload["status"], "unavailable");
        assert_eq!(payload["reason"], "forked_history_repair");
        assert_eq!(
            payload["note"],
            "tool result was unavailable in repaired conversation history"
        );
        assert!(!response.content.contains("(tool result not retained)"));
        assert!(
            params.tasks[0].messages.iter().all(|message| !matches!(
                message.message,
                Some(api::message::Message::ToolCallResult(_))
            )),
            "accepted repair placeholders must remain outbound-only"
        );
    }

    #[test]
    fn accepted_repair_log_summary_includes_source_counts_and_redacted_keys() {
        let repairs = vec![
            AcceptedRepair {
                record: RepairRecord::new(
                    RepairSource::ForkedHistory,
                    ToolCallKey::new("task-1", "assistant-1", "call-1"),
                ),
                tool_call: ToolCallRef::new(
                    ToolCallKey::new("task-1", "assistant-1", "call-1"),
                    kind(),
                ),
            },
            AcceptedRepair {
                record: RepairRecord::new(
                    RepairSource::RestoredLegacyHistory,
                    ToolCallKey::new("task-1", "assistant-2", "call-2"),
                ),
                tool_call: ToolCallRef::new(
                    ToolCallKey::new("task-1", "assistant-2", "call-2"),
                    RedactedToolKind::new("local_interception:webfetch"),
                ),
            },
        ];
        let context = ReadinessDiagnosticContext::new(
            "conversation-1",
            "attempt-1",
            ReadinessTriggerLayer::SerializerValidation,
        );

        let message = accepted_history_repair_log_message(&repairs, &context);

        assert!(message.contains("serializer accepted history repair"));
        assert!(message.contains("records=2"));
        assert!(message.contains("category=AcceptedHistoryRepair"));
        assert!(message.contains("forked_history=1"));
        assert!(message.contains("restored_legacy_history=1"));
        assert!(message.contains("conversation_id=conversation-1"));
        assert!(message.contains("trigger_layer=serializer_validation"));
        assert!(message.contains("request_attempt_id=attempt-1"));
        assert!(message.contains("task_id=task-1"));
        assert!(message.contains("assistant_tool_call_message_id=assistant-1"));
        assert!(message.contains("tool_call_id=call-1"));
        assert!(message.contains("redacted_tool_kind=local_interception:webfetch"));
        assert!(!message.contains(REPAIR_PLACEHOLDER_NOTE));
        assert!(!message.contains("secret user prompt"));
        assert!(!message.contains("raw tool arguments"));
        assert!(!message.contains("raw tool output"));
        assert!(!message.contains("raw local interception payload"));
    }

    #[test]
    fn build_chat_request_body_honors_compaction_filtering_boundaries() {
        let hidden_user = make_user_query_message("task-1", "req-1", "hidden".to_owned(), &[]);
        let hidden_call = make_tool_call_message("task-1", "req-1", "hidden-call", shell_tool());
        let summary_user = make_user_query_message("task-1", "req-2", "/compact".to_owned(), &[]);
        let summary_assistant =
            make_agent_output_message("task-1", "req-2", "redacted summary".to_owned());
        let visible_user = make_user_query_message("task-1", "req-3", "visible".to_owned(), &[]);

        let mut compaction_state = CompactionState::default();
        compaction_state.push_completed(CompletedCompaction {
            user_msg_id: summary_user.id.clone(),
            assistant_msg_id: summary_assistant.id.clone(),
            head_message_ids: vec![hidden_user.id.clone(), hidden_call.id.clone()],
            tail_start_id: Some(visible_user.id.clone()),
            summary_text: Some("redacted summary".to_owned()),
            auto: false,
            overflow: false,
        });

        let mut params = request_params(
            vec![
                hidden_user.clone(),
                hidden_call.clone(),
                summary_user.clone(),
                summary_assistant.clone(),
                visible_user.clone(),
            ],
            vec![],
        );
        params.compaction_state = Some(compaction_state.clone());

        let request = build_openai_request(&params)
            .expect("hidden historical tool-call gap should not block");
        let errors = strict_chat_completions_ordering_errors(&request.messages);
        assert!(
            errors.is_empty(),
            "request body ordering errors: {errors:?}"
        );
        assert_request_has_no_repair_placeholder(&request);

        let visible_call = make_tool_call_message("task-1", "req-3", "visible-call", shell_tool());
        let mut params = request_params(
            vec![
                hidden_user,
                hidden_call,
                summary_user,
                summary_assistant,
                visible_user,
                visible_call,
            ],
            vec![],
        );
        params.compaction_state = Some(compaction_state);

        assert_build_request_blocked(params, "MissingResultWithoutRepairSource");
    }

    #[test]
    fn smoke_build_chat_request_simple_user_query_succeeds() {
        let params = request_params(
            vec![make_user_query_message(
                "task-1",
                "req-1",
                "hello".to_owned(),
                &[],
            )],
            vec![user_query_input("hello")],
        );
        let request = build_openai_request(&params).expect("simple user query should serialize");
        assert!(
            !request.messages.is_empty(),
            "expected at least one chat message"
        );
        assert_request_has_no_repair_placeholder(&request);
    }

    #[test]
    fn smoke_build_chat_request_pending_live_tool_call_is_pending_not_blocked() {
        let user_message = make_user_query_message("task-1", "req-1", "hi".to_owned(), &[]);
        let tool_call_message = make_tool_call_message("task-1", "req-1", "call-1", shell_tool());
        let assistant_message_id = tool_call_message.id.clone();
        let params = request_params(
            vec![user_message, tool_call_message],
            vec![user_query_input("continue")],
        );

        let pending_report = classify_byop_controller_readiness_with_live_tool_calls(
            &params,
            vec![LiveToolCall::new(
                ToolCallRef::new(
                    ToolCallKey::new("task-1", assistant_message_id, "call-1"),
                    kind(),
                ),
                LiveToolCallState::Running,
            )],
        );
        assert!(
            matches!(
                pending_report.state,
                ReadinessState::PendingToolResults { .. }
            ),
            "running tool should wait, not block: {:?}",
            pending_report.state
        );

        let build_result = build_openai_request(&params);
        assert!(
            build_result.is_err(),
            "serializer must not send while tool is still running"
        );
    }
}

/// **Verification of accepted history repair outbound repair behavior**:
///
/// This module only covers outbound repairs explicitly authorized by `RepairRecord`. For the ordinary normal flow,
/// missing results must first be blocked by readiness, and no longer fill placeholders here.
#[cfg(test)]
mod accepted_history_repair_tests {
    use super::*;
    use crate::ai::byop_readiness::{RepairRecord, ToolCallKey, ToolCallRef};
    use genai::chat::{ChatMessage, ChatRole, ToolCall};

    fn make_tool_call(call_id: &str) -> ToolCall {
        ToolCall {
            call_id: call_id.to_owned(),
            fn_name: "dummy".to_owned(),
            fn_arguments: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    fn assistant_with_calls(call_ids: &[&str]) -> ChatMessage {
        let calls: Vec<ToolCall> = call_ids.iter().map(|cid| make_tool_call(cid)).collect();
        ChatMessage::from(calls)
    }

    fn tool_response(call_id: &str, content: &str) -> ChatMessage {
        ChatMessage::from(ToolResponse::new(call_id.to_owned(), content.to_owned()))
    }

    /// Flattens the (call_id, content) of all ToolResponses in a single Tool message, for convenient assertions.
    fn responses_of(msg: &ChatMessage) -> Vec<(String, String)> {
        msg.content
            .tool_responses()
            .iter()
            .map(|r| (r.call_id.clone(), r.content.clone()))
            .collect()
    }

    fn accepted_repair(key: ToolCallKey, source: RepairSource) -> AcceptedRepair {
        AcceptedRepair {
            record: RepairRecord::new(source, key.clone()),
            tool_call: ToolCallRef::new(key, RedactedToolKind::new("shell")),
        }
    }

    fn outbound_groups_for_messages(messages: &[ChatMessage]) -> Vec<OutboundAssistantToolGroup> {
        let mut groups = Vec::new();
        let mut assistant_group_number = 0;
        for (message_index, message) in messages.iter().enumerate() {
            if message.role != ChatRole::Assistant || message.content.tool_calls().is_empty() {
                continue;
            }

            assistant_group_number += 1;
            let assistant_message_id = format!("assistant-{assistant_group_number}");
            groups.push(OutboundAssistantToolGroup {
                message_index,
                tool_call_keys: message
                    .content
                    .tool_calls()
                    .iter()
                    .map(|tool_call| {
                        ToolCallKey::new("task-1", &assistant_message_id, &tool_call.call_id)
                    })
                    .collect(),
            });
        }

        groups
    }

    fn repairs_for_groups(groups: &[OutboundAssistantToolGroup]) -> Vec<AcceptedRepair> {
        groups
            .iter()
            .flat_map(|group| {
                group
                    .tool_call_keys
                    .iter()
                    .cloned()
                    .map(|key| accepted_repair(key, RepairSource::ForkedHistory))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn repair_messages(messages: &mut Vec<ChatMessage>) {
        let groups = outbound_groups_for_messages(messages);
        let repairs = repairs_for_groups(&groups);
        repair_tool_call_pairs_for_accepted_history_gaps(messages, &repairs, &groups)
            .expect("repair helper expects all gaps to be authorized in tests");
    }

    fn assert_structured_repair_payload(content: &str, reason: &str) {
        let payload: serde_json::Value =
            serde_json::from_str(content).expect("repair placeholder should be JSON");
        assert_eq!(payload["status"], "unavailable");
        assert_eq!(payload["reason"], reason);
        assert_eq!(
            payload["note"],
            "tool result was unavailable in repaired conversation history"
        );
    }

    /// Normal push path: [user, asst(a,b), tool_a, tool_b] → merged into a single bundled message, no padding.
    #[test]
    fn normal_push_path_merges_adjacent_tool_messages_without_placeholder() {
        let mut msgs = vec![
            ChatMessage::user("hi"),
            assistant_with_calls(&["a", "b"]),
            tool_response("a", "resp_a"),
            tool_response("b", "resp_b"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3, "two adjacent Tool messages merge into one");
        assert_eq!(msgs[0].role, ChatRole::User);
        assert_eq!(msgs[1].role, ChatRole::Assistant);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![
                ("a".to_owned(), "resp_a".to_owned()),
                ("b".to_owned(), "resp_b".to_owned()),
            ],
            "bundled response order must match Assistant.tool_calls"
        );
    }

    /// Fork truncation scenario A: the Assistant has tool_calls but **all** ToolCallResults are missing →
    /// insert a Tool message full of placeholders after the Assistant.
    #[test]
    fn fork_truncated_missing_all_tool_responses_inserts_placeholders() {
        let mut msgs = vec![ChatMessage::user("q"), assistant_with_calls(&["a", "b"])];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3, "a Tool message must be padded after the Assistant");
        assert_eq!(msgs[2].role, ChatRole::Tool);
        let responses = responses_of(&msgs[2]);
        assert_eq!(
            responses.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>(),
            vec!["a".to_owned(), "b".to_owned()]
        );
        for (_, content) in &responses {
            assert_structured_repair_payload(content, "forked_history_repair");
        }
    }

    /// Fork truncation scenario B: the Assistant has (a, b) but only tool_a was kept → b gets a placeholder,
    /// the order is still regrouped per Assistant.tool_calls into (real_a, placeholder_b).
    #[test]
    fn fork_truncated_partial_tool_responses_fills_missing_only() {
        let mut msgs = vec![
            ChatMessage::user("q"),
            assistant_with_calls(&["a", "b"]),
            tool_response("a", "real_a"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        let responses = responses_of(&msgs[2]);
        assert_eq!(responses[0], ("a".to_owned(), "real_a".to_owned()));
        assert_eq!(responses[1].0, "b".to_owned());
        assert_structured_repair_payload(&responses[1].1, "forked_history_repair");
    }

    /// An orphan ToolResponse (call_id not in Assistant.tool_calls) is dropped and does not pollute the output.
    #[test]
    fn orphan_tool_response_with_unknown_call_id_is_dropped() {
        let mut msgs = vec![
            ChatMessage::user("q"),
            assistant_with_calls(&["a"]),
            tool_response("a", "real_a"),
            tool_response("z", "orphan_z"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3, "two adjacent Tools merge, orphan z is dropped");
        let responses = responses_of(&msgs[2]);
        assert_eq!(
            responses,
            vec![("a".to_owned(), "real_a".to_owned())],
            "keep only the call_ids the Assistant knows"
        );
    }

    /// Assistant.tool_calls order is (a, b) but the existing Tool order is (b, a) →
    /// bundled output is regrouped into (real_a, real_b), aligned with Anthropic's tool_use order.
    #[test]
    fn out_of_order_tool_responses_are_reordered_per_assistant_calls() {
        let mut msgs = vec![
            ChatMessage::user("q"),
            assistant_with_calls(&["a", "b"]),
            tool_response("b", "real_b"),
            tool_response("a", "real_a"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![
                ("a".to_owned(), "real_a".to_owned()),
                ("b".to_owned(), "real_b".to_owned()),
            ]
        );
    }

    /// When the user interrupts/keeps typing, a completed tool result may be persisted later than the new UserQuery.
    /// The outbound request must move the result back after the corresponding Assistant, otherwise upstream would see an orphan ToolResponse.
    #[test]
    fn late_tool_response_after_user_query_is_moved_back_to_tool_call() {
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a"]),
            ChatMessage::user("interrupt"),
            tool_response("a", "real_a"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, ChatRole::User);
        assert_eq!(msgs[1].role, ChatRole::Assistant);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![("a".to_owned(), "real_a".to_owned())]
        );
        assert_eq!(msgs[3].role, ChatRole::User);
    }

    /// Two consecutive real ToolResponses for the same call_id (duplicate persistence / manual retry, etc.),
    /// the later real value should override the earlier one, keeping consistent with the old implementation's "later insert overrides" semantics.
    #[test]
    fn real_tool_response_is_replaced_by_later_real_tool_response() {
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a"]),
            tool_response("a", "real_a_v1"),
            tool_response("a", "real_a_v2"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![("a".to_owned(), "real_a_v2".to_owned())],
            "multiple real responses for the same call_id, the later one wins"
        );
    }

    /// A placeholder must not override an existing real result.
    /// This scenario arises when: last turn a placeholder was padded as filler → this turn the real result lands →
    /// then a placeholder with the same call_id is encountered again (e.g. concatenation after a fork). We must ensure the real one is not clobbered.
    #[test]
    fn placeholder_does_not_overwrite_existing_real_response() {
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a"]),
            tool_response("a", "real_a"),
            tool_response("a", "(tool result not retained)"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![("a".to_owned(), "real_a".to_owned())],
            "a placeholder must not override the real value"
        );
    }

    /// A polluted history may contain both a placeholder and a late-arriving real result; the real result should override the placeholder.
    #[test]
    fn placeholder_is_replaced_by_late_real_tool_response() {
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a"]),
            tool_response("a", "(tool result not retained)"),
            ChatMessage::user("interrupt"),
            tool_response("a", "real_a"),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![("a".to_owned(), "real_a".to_owned())]
        );
        assert_eq!(msgs[3].role, ChatRole::User);
    }

    /// Multiple Assistant tool_calls segments can each be processed independently: each Assistant + Tool segment does not affect the others.
    #[test]
    fn multiple_assistant_tool_call_groups_handled_independently() {
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a", "b"]),
            tool_response("a", "real_a"),
            tool_response("b", "real_b"),
            ChatMessage::user("q2"),
            assistant_with_calls(&["c"]),
            tool_response("c", "real_c"),
        ];
        repair_messages(&mut msgs);

        // Expected result: user, asst(a,b), bundled(a,b), user, asst(c), bundled(c)
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![
                ("a".to_owned(), "real_a".to_owned()),
                ("b".to_owned(), "real_b".to_owned()),
            ]
        );
        assert_eq!(msgs[5].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[5]),
            vec![("c".to_owned(), "real_c".to_owned())]
        );
    }

    /// An Assistant message without tool_calls is left alone; no extra Tool is added.
    #[test]
    fn assistant_without_tool_calls_is_untouched() {
        let mut msgs = vec![
            ChatMessage::user("q"),
            ChatMessage::assistant("plain reply"),
        ];
        let before = msgs.len();
        repair_messages(&mut msgs);
        assert_eq!(msgs.len(), before);
        assert_eq!(msgs[1].role, ChatRole::Assistant);
        assert!(msgs[1].content.tool_calls().is_empty());
    }

    #[test]
    fn accepted_repair_placeholder_is_authorized_by_full_tool_call_key() {
        let first_key = ToolCallKey::new("task-1", "assistant-1", "dup");
        let second_key = ToolCallKey::new("task-2", "assistant-2", "dup");
        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["dup"]),
            ChatMessage::user("q2"),
            assistant_with_calls(&["dup"]),
            tool_response("dup", "real_second"),
        ];
        let groups = vec![
            OutboundAssistantToolGroup {
                message_index: 1,
                tool_call_keys: vec![first_key.clone()],
            },
            OutboundAssistantToolGroup {
                message_index: 3,
                tool_call_keys: vec![second_key],
            },
        ];
        let repairs = vec![accepted_repair(first_key, RepairSource::ForkedHistory)];

        repair_tool_call_pairs_for_accepted_history_gaps(&mut msgs, &repairs, &groups)
            .expect("second_key has a real response, so repair must succeed");

        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[2].role, ChatRole::Tool);
        let first_responses = responses_of(&msgs[2]);
        assert_eq!(first_responses[0].0, "dup");
        assert_structured_repair_payload(&first_responses[0].1, "forked_history_repair");
        assert_eq!(msgs[5].role, ChatRole::Tool);
        assert_eq!(
            responses_of(&msgs[5]),
            vec![("dup".to_owned(), "real_second".to_owned())],
            "the real result of a duplicate call_id must stay after its own Assistant group"
        );
    }

    #[test]
    fn unavailable_json_with_non_repair_reason_is_not_placeholder() {
        let real_unavailable =
            r#"{"status":"unavailable","reason":"service_down","note":"try later"}"#;
        assert!(!is_placeholder_tool_response_content(real_unavailable));
        assert!(!is_placeholder_tool_response_content(
            r#"{"status":"unavailable","reason":"forked_history_repair","note":"tool result was unavailable in repaired conversation history","extra":true}"#,
        ));
        assert!(is_placeholder_tool_response_content(
            &repair_placeholder_content(RepairSource::ForkedHistory)
        ));

        let mut msgs = vec![
            ChatMessage::user("q1"),
            assistant_with_calls(&["a"]),
            tool_response("a", real_unavailable),
            tool_response(
                "a",
                &repair_placeholder_content(RepairSource::ForkedHistory),
            ),
        ];
        repair_messages(&mut msgs);

        assert_eq!(msgs.len(), 3);
        assert_eq!(
            responses_of(&msgs[2]),
            vec![("a".to_owned(), real_unavailable.to_owned())],
            "a real unavailable JSON must not be overridden by the repair placeholder"
        );
    }
}

// ---------------------------------------------------------------------------
// Test helper: for use by `cache_stability_tests` within the same crate.
// ---------------------------------------------------------------------------

/// Test-only wrapper: lets other test modules in the same crate call the originally file-private
/// `serialize_outgoing_tool_call`. Exposed only under `cfg(test)`, not affecting the production code surface.
#[cfg(test)]
pub(super) fn serialize_outgoing_tool_call_for_test(
    tc: &api::message::ToolCall,
    mcp_ctx: Option<&crate::ai::agent::MCPContext>,
    server_message_data: &str,
) -> (String, Value) {
    serialize_outgoing_tool_call(tc, mcp_ctx, server_message_data)
}

/// Issue #94 regression tests: when `build_chat_request` collects multi-task messages, history-turn user messages
/// must not be misordered to the end, nor duplicated due to LRC subagent copies.
#[cfg(test)]
mod issue_94_task_linearization_tests {
    use super::*;
    use crate::test_util::ai_agent_tasks::{
        create_api_subtask, create_api_task, create_message, create_subagent_tool_call_message,
    };

    /// Builds a UserQuery message with `request_id`.
    fn user_query_msg(id: &str, task_id: &str, request_id: &str, query: &str) -> api::Message {
        api::Message {
            id: id.to_string(),
            task_id: task_id.to_string(),
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                query: query.to_string(),
                ..Default::default()
            })),
            request_id: request_id.to_string(),
            timestamp: None,
        }
    }

    /// Reproduces the 01.json / 02.json scenario of Issue #94: the user sent only one sentence, but after the LRC CLI
    /// subagent spawn, that UserQuery exists in both the root task and the subtask.
    ///
    /// - root: [UserQuery, AgentOutput, Subagent ToolCall→subtask]
    /// - subtask: [UserQuery (copy with the same request_id+query), AgentOutput]
    fn issue_94_tasks() -> (api::Task, api::Task) {
        const REQ: &str = "req-1";
        const QUERY: &str = "帮我在这个服务器上搭建一个dns(doh)分流服务";
        let root = create_api_task(
            "root",
            vec![
                user_query_msg("m1", "root", REQ, QUERY),
                create_message("m2", "root"),
                create_subagent_tool_call_message("m3", "root", "sub-1", None),
            ],
        );
        let subtask = create_api_subtask(
            "sub-1",
            "root",
            vec![
                user_query_msg("s1", "sub-1", REQ, QUERY),
                create_message("s2", "sub-1"),
            ],
        );
        (root, subtask)
    }

    fn count_user_queries(msgs: &[&api::Message]) -> usize {
        msgs.iter()
            .filter(|m| matches!(&m.message, Some(api::message::Message::UserQuery(_))))
            .count()
    }

    fn message_ids(msgs: &[&api::Message]) -> Vec<String> {
        msgs.iter().map(|m| m.id.clone()).collect()
    }

    /// Reproduction: the old implementation `params.tasks.iter().flat_map(|t| t.messages.iter())` naive concatenation:
    /// (1) UserQuery appears twice; (2) the result varies with task input order (`compute_active_tasks`
    /// collects with HashMap::into_values, so the order is nondeterministic).
    #[test]
    fn naive_flat_map_reproduces_issue_94() {
        let (root, subtask) = issue_94_tasks();

        let naive = |tasks: &[api::Task]| -> Vec<String> {
            tasks
                .iter()
                .flat_map(|t| t.messages.iter())
                .map(|m| m.id.clone())
                .collect()
        };

        let root_first = naive(&[root.clone(), subtask.clone()]);
        let subtask_first = naive(&[subtask.clone(), root.clone()]);

        // (1) UserQuery concatenated twice.
        let root_first_refs: Vec<&api::Message> = [&root, &subtask]
            .iter()
            .flat_map(|t| t.messages.iter())
            .collect();
        assert_eq!(
            count_user_queries(&root_first_refs),
            2,
            "naive concatenation makes the same user query appear twice, exactly the bug of Issue #94"
        );

        // (2) the order drifts with the input task order: when the subtask comes first, the history user (m1) is thrown to the end.
        assert_ne!(
            root_first, subtask_first,
            "naive concatenation results depend on task order, nondeterministic"
        );
        assert_eq!(
            subtask_first.last().map(String::as_str),
            Some("m3"),
            "when the subtask is placed first, root's messages shift back as a whole"
        );
        assert!(
            subtask_first.iter().position(|id| id == "s1").unwrap()
                < subtask_first.iter().position(|id| id == "m1").unwrap(),
            "the subtask's UserQuery copy (s1) landed before root's original (m1)"
        );
    }

    /// Fix verification: the output of `collect_linearized_task_messages` is independent of task input order,
    /// after UserQuery dedupe only one remains, and the whole is the DFS linear order root→subtask.
    #[test]
    fn linearized_collection_is_deterministic_and_deduped() {
        let (root, subtask) = issue_94_tasks();

        let root_first = vec![root.clone(), subtask.clone()];
        let subtask_first = vec![subtask.clone(), root.clone()];

        let a = collect_linearized_task_messages(&root_first);
        let b = collect_linearized_task_messages(&subtask_first);

        // Independent of input order.
        assert_eq!(
            message_ids(&a),
            message_ids(&b),
            "the result must be independent of the input order of params.tasks"
        );

        // UserQuery dedupe: the subtask copy (s1) made by LRC is dropped.
        assert_eq!(
            count_user_queries(&a),
            1,
            "the duplicate UserQuery must be deduped to one"
        );

        // DFS linear order: root's messages first, descend into the subtask on a Subagent ToolCall.
        // s1 is deduped, so expect [m1, m2, m3, s2].
        assert_eq!(message_ids(&a), vec!["m1", "m2", "m3", "s2"]);

        // The remaining user query is root's original, at the start of the sequence.
        assert_eq!(a.first().map(|m| m.id.as_str()), Some("m1"));
    }

    /// A normal single-task conversation (no subagent) is unaffected: messages are returned as-is, in original order.
    #[test]
    fn single_task_conversation_unchanged() {
        let root = create_api_task(
            "root",
            vec![
                user_query_msg("m1", "root", "req-1", "你好"),
                create_message("m2", "root"),
            ],
        );
        let out = collect_linearized_task_messages(std::slice::from_ref(&root));
        assert_eq!(message_ids(&out), vec!["m1", "m2"]);
    }

    /// Different user turns, even with the same query text, are not wrongly removed as long as `request_id` differs.
    #[test]
    fn distinct_turns_with_same_text_are_kept() {
        let root = create_api_task(
            "root",
            vec![
                user_query_msg("m1", "root", "req-1", "继续"),
                create_message("m2", "root"),
                user_query_msg("m3", "root", "req-2", "继续"),
            ],
        );
        let out = collect_linearized_task_messages(std::slice::from_ref(&root));
        assert_eq!(
            count_user_queries(&out),
            2,
            "both user messages of the two turns with different request_id must be kept"
        );
        assert_eq!(message_ids(&out), vec!["m1", "m2", "m3"]);
    }
}
