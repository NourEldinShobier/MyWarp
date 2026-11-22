//! BYOP one-shot non-streaming completion adapter.
//!
//! Used by the "proactive AI" sub-flows (prompt suggestions / NLD predict / relevant files /
//! conversation title generation, etc.): they send one short request and get back a piece
//! of text, with **no tool calling, no streaming, and no persistence to task.messages**.
//!
//! Differences from `chat_stream::generate_byop_output` (the main conversation stream):
//! - This goes through `Client::exec_chat` (non-streaming) and gets `ChatResponse::first_text()` in one shot.
//! - It does not use `RequestParams` / `ResponseEvent` / `task_store`; it is plain string in, string out.
//! - Reasoning is disabled by default (proactive AI should not trigger a thinking chain,
//!   which wastes tokens and is slow); it is injected through the capability gate only when
//!   `OneshotOptions.allow_reasoning = true`.
//!
//! The caller chooses the model: `resolve_active_ai_oneshot()` decodes `active_ai_model`
//! (profile fallback to base_model) into a BYOP `OneshotConfig`.
//! On decode failure (BYOP not configured / model not in the BYOP encoding space) it returns `None`,
//! and the caller silently no-ops.

use anyhow::Context as _;
use futures::StreamExt;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatStreamEvent};
use warpui::{AppContext, EntityId, SingletonEntity as _};

use super::chat_stream;
use super::header_vars::HeaderVars;
use crate::ai::agent::conversation::AIConversationId;
use crate::ai::llms::LLMPreferences;
use crate::settings::{AgentProviderApiType, ReasoningEffortSetting};

/// Provider/model info needed for a BYOP one-shot request.
#[derive(Debug, Clone)]
pub struct OneshotConfig {
    pub base_url: String,
    pub api_key: String,
    pub model_id: String,
    pub api_type: AgentProviderApiType,
    pub reasoning_effort: ReasoningEffortSetting,
    /// Extra provider headers (values may contain `{conversationId}`).
    pub extra_headers: Vec<(String, String)>,
    /// Conversation id used to resolve `{conversationId}`; `None` for one-shots without a conversation.
    pub conversation_id: Option<AIConversationId>,
}

/// Optional parameters for a one-shot call.
#[derive(Debug, Clone, Default)]
pub struct OneshotOptions {
    /// Character truncation limit for the user message (per char, to protect CJK). `None` = default 8000.
    pub max_chars: Option<usize>,
    /// Temperature (genai `ChatOptions::temperature`); `None` = provider default.
    pub temperature: Option<f32>,
    /// Whether to require JSON output (OpenAI-compatible providers use response_format).
    /// Note: adapters that don't support it ignore this parameter, so the system prompt must itself ask for JSON.
    pub response_format_json: bool,
    /// Whether reasoning may be triggered. Default `false` (proactive AI calls are low-latency and lightweight).
    pub allow_reasoning: bool,
}

const DEFAULT_MAX_CHARS: usize = 8000;

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    s.chars().take(max).collect()
}

fn build_oneshot_request(
    cfg: &OneshotConfig,
    system: &str,
    user: &str,
    opts: &OneshotOptions,
) -> (ChatRequest, ChatOptions) {
    let mut chat_opts = ChatOptions::default()
        .with_capture_content(true)
        .with_capture_usage(true);
    if let Some(t) = opts.temperature {
        chat_opts = chat_opts.with_temperature(t.into());
    }
    if opts.response_format_json {
        chat_opts = chat_opts.with_response_format(genai::chat::ChatResponseFormat::JsonMode);
    }
    if opts.allow_reasoning {
        if let Some(effort) = cfg.reasoning_effort.to_genai() {
            if super::reasoning::model_supports_reasoning(cfg.api_type, &cfg.model_id) {
                chat_opts = chat_opts.with_reasoning_effort(effort);
            }
        }
    }

    let max_chars = opts.max_chars.unwrap_or(DEFAULT_MAX_CHARS);
    let user_truncated = truncate_chars(user, max_chars);

    // `{conversationId}` in header values is resolved before sending; with no conversation the whole header line is omitted.
    let headers = super::header_vars::resolve_headers(
        &cfg.extra_headers,
        &HeaderVars::default().with_conversation_id(cfg.conversation_id),
    );
    if !headers.is_empty() {
        chat_opts = chat_opts.with_extra_headers(headers);
    }

    let chat_req = ChatRequest::from_messages(vec![ChatMessage::user(user_truncated)])
        .with_system(system.to_owned());

    (chat_req, chat_opts)
}

/// Send one BYOP non-streaming chat completion and return the model reply as plain text.
///
/// Whether to swallow errors is up to the caller; this only propagates `anyhow::Error` and does no logging.
pub async fn byop_oneshot_completion(
    cfg: &OneshotConfig,
    system: &str,
    user: &str,
    opts: &OneshotOptions,
) -> anyhow::Result<String> {
    let client = chat_stream::build_client(cfg.api_type, &cfg.base_url, cfg.api_key.clone());
    let (chat_req, chat_opts) = build_oneshot_request(cfg, system, user, opts);

    let resp = client
        .exec_chat(&cfg.model_id, chat_req, Some(&chat_opts))
        .await
        .with_context(|| format!("byop oneshot exec_chat failed (model={})", cfg.model_id))?;

    Ok(resp.first_text().unwrap_or("").to_owned())
}

/// Send one BYOP streaming chat completion, aggregate all text chunks, and return them.
///
/// For OpenAI Responses-compatible proxies that only accept `stream=true`. The caller still
/// gets the full string, so it can keep reusing the one-shot title cleanup / JSON parsing logic.
pub async fn byop_oneshot_streaming_completion(
    cfg: &OneshotConfig,
    system: &str,
    user: &str,
    opts: &OneshotOptions,
) -> anyhow::Result<String> {
    let client = chat_stream::build_client(cfg.api_type, &cfg.base_url, cfg.api_key.clone());
    let (chat_req, chat_opts) = build_oneshot_request(cfg, system, user, opts);
    let mut resp = client
        .exec_chat_stream(&cfg.model_id, chat_req, Some(&chat_opts))
        .await
        .with_context(|| {
            format!(
                "byop oneshot exec_chat_stream failed (model={})",
                cfg.model_id
            )
        })?
        .stream;

    let mut text = String::new();
    while let Some(event) = resp.next().await {
        match event.with_context(|| {
            format!(
                "byop oneshot exec_chat_stream event failed (model={})",
                cfg.model_id
            )
        })? {
            ChatStreamEvent::Chunk(chunk) => {
                text.push_str(&chunk.content);
            }
            ChatStreamEvent::Start
            | ChatStreamEvent::ReasoningChunk(_)
            | ChatStreamEvent::ThoughtSignatureChunk(_)
            | ChatStreamEvent::ToolCallChunk(_)
            | ChatStreamEvent::End(_) => {}
        }
    }

    Ok(text)
}

/// Resolve the current active profile's `active_ai_model` (falling back to `base_model`);
/// if it decodes to a valid BYOP encoding, return `OneshotConfig`, otherwise `None` (the caller silently no-ops).
pub fn resolve_active_ai_oneshot(
    app: &AppContext,
    terminal_view_id: Option<EntityId>,
) -> Option<OneshotConfig> {
    let llm_prefs = LLMPreferences::as_ref(app);
    let id = llm_prefs
        .get_active_ai_model(app, terminal_view_id)
        .id
        .clone();
    let (provider, api_key, model_id) = super::lookup_byop(app, &id)?;
    let reasoning_effort =
        llm_prefs.get_reasoning_effort(terminal_view_id, provider.api_type, &model_id);
    Some(OneshotConfig {
        base_url: provider.base_url,
        api_key,
        model_id,
        api_type: provider.api_type,
        reasoning_effort,
        extra_headers: provider.extra_headers,
        conversation_id: None,
    })
}

/// Resolve the current active profile's `next_command_model` (falling back to `base_model`);
/// if it decodes to a valid BYOP encoding, return `OneshotConfig`, otherwise `None`.
pub fn resolve_next_command_oneshot(
    app: &AppContext,
    terminal_view_id: Option<EntityId>,
) -> Option<OneshotConfig> {
    let llm_prefs = LLMPreferences::as_ref(app);
    let id = llm_prefs
        .get_active_next_command_model(app, terminal_view_id)
        .id
        .clone();
    let (provider, api_key, model_id) = super::lookup_byop(app, &id)?;
    let reasoning_effort =
        llm_prefs.get_reasoning_effort(terminal_view_id, provider.api_type, &model_id);
    Some(OneshotConfig {
        base_url: provider.base_url,
        api_key,
        model_id,
        api_type: provider.api_type,
        reasoning_effort,
        extra_headers: provider.extra_headers,
        conversation_id: None,
    })
}

#[cfg(test)]
#[path = "oneshot_tests.rs"]
mod tests;
