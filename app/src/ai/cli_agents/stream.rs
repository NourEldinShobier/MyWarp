//! Converts a backend's neutral [`AgentEvent`]s into the `ResponseEvent` stream the
//! conversation model already consumes (the same one the BYOP path produces).

use std::path::PathBuf;
use std::sync::Arc;

use futures::channel::oneshot;
use futures::StreamExt as _;
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::{AgentBackend, AgentEvent, TurnRequest};
use crate::ai::agent::api::{RequestParams, ResponseStream};
use crate::ai::agent::AIAgentInput;
use crate::ai::agent_providers::chat_stream::{
    create_task_event, make_add_messages_event, make_agent_output_message, make_append_event,
    make_finished_done, make_reasoning_message, make_tool_call_message, make_update_message_event,
    make_web_fetch_fetching_message, make_web_fetch_status_from_result,
    make_web_search_searching_message, make_web_search_status_from_result, make_user_query_message, AppendKind,
};
use crate::ai::api_error::AIApiError;

pub(crate) struct CliTurnInput {
    pub params: RequestParams,
    pub backend: &'static dyn AgentBackend,
    pub model: Option<String>,
    pub task_id: String,
    pub needs_create_task: bool,
    pub cancellation_rx: oneshot::Receiver<()>,
}

/// The text block currently being streamed, so deltas append to one message.
#[derive(PartialEq)]
enum Open {
    None,
    Text(String),
    Reasoning(String),
}

pub(crate) fn generate_cli_output(input: CliTurnInput) -> ResponseStream {
    let CliTurnInput { params, backend, model, task_id, needs_create_task, cancellation_rx } = input;

    let prompt = params
        .input
        .iter()
        .rev()
        .find_map(|i| match i {
            AIAgentInput::UserQuery { query, .. } => Some(query.clone()),
            _ => None,
        })
        .unwrap_or_default();
    // The server conversation token doubles as the CLI's session id: generated on the first
    // turn (and echoed in `Init`), passed back by the controller on later turns.
    let existing = params.conversation_token.as_ref().map(|t| t.as_str().to_owned());
    let resume = existing.is_some();
    let session_id = existing.unwrap_or_else(|| Uuid::new_v4().to_string());
    let cwd = params.session_context.current_working_directory().as_ref().map(PathBuf::from);
    let request_id = Uuid::new_v4().to_string();

    Box::pin(async_stream::stream! {
        yield Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(api::response_event::StreamInit {
                request_id: request_id.clone(),
                conversation_id: session_id.clone(),
                run_id: String::new(),
            })),
        });
        if needs_create_task {
            yield Ok(create_task_event(&task_id));
        }
        if !prompt.is_empty() {
            let msg = make_user_query_message(&task_id, &request_id, prompt.clone(), &[]);
            yield Ok(make_add_messages_event(&task_id, vec![msg]));
        }

        let mut events = backend
            .run_turn(TurnRequest { prompt, cwd, model, session_id: session_id.clone(), resume })
            .take_until(cancellation_rx);
        let mut open = Open::None;
        // The open thinking block and when it started, to stamp "thought for N s" on it.
        let mut reasoning: Option<(String, std::time::Instant)> = None;
        let mut failed = None;
        // Tools shown with a native Warp card, by tool id.
        let mut native: std::collections::HashMap<String, Native> = Default::default();
        let styling = super::style::enabled();
        let mut styler = super::style::LineStyler::default();
        'turn: while let Some(raw) = events.next().await {
            // Text is styled line by line; any other event first flushes the unfinished line.
            let mut batch = Vec::with_capacity(2);
            match raw {
                AgentEvent::Text(t) => {
                    let styled = if styling { styler.push(&t) } else { t };
                    if !styled.is_empty() {
                        batch.push(AgentEvent::Text(styled));
                    }
                }
                other => {
                    let rest = styler.flush();
                    if !rest.is_empty() {
                        batch.push(AgentEvent::Text(rest));
                    }
                    batch.push(other);
                }
            }
            for ev in batch {
            if !matches!(ev, AgentEvent::Reasoning(_)) {
                if let Some((id, started)) = reasoning.take() {
                    yield Ok(make_reasoning_done_event(&task_id, &request_id, &id, started.elapsed()));
                }
            }
            match ev {
                AgentEvent::Text(t) => match &open {
                    Open::Text(id) => yield Ok(make_append_event(&task_id, id, AppendKind::Text(t))),
                    _ => {
                        let msg = make_agent_output_message(&task_id, &request_id, t);
                        open = Open::Text(msg.id.clone());
                        yield Ok(make_add_messages_event(&task_id, vec![msg]));
                    }
                },
                AgentEvent::Reasoning(t) => match &open {
                    Open::Reasoning(id) => {
                        yield Ok(make_append_event(&task_id, id, AppendKind::Reasoning(t)))
                    }
                    _ => {
                        let msg = make_reasoning_message(&task_id, &request_id, t);
                        open = Open::Reasoning(msg.id.clone());
                        reasoning = Some((msg.id.clone(), std::time::Instant::now()));
                        yield Ok(make_add_messages_event(&task_id, vec![msg]));
                    }
                },
                AgentEvent::ToolStarted { id, name, input } => {
                    open = Open::None;
                    // Native cards for tools Warp already has a card for.
                    if name == "TodoWrite" {
                        if let Ok(msgs) = crate::ai::agent_providers::tools::todowrite::build_update_todos_messages(
                            &input.to_string(), &task_id, &request_id,
                        ) {
                            native.insert(id, Native::Todo);
                            yield Ok(make_add_messages_event(&task_id, msgs));
                            continue;
                        }
                    }
                    if name == "WebSearch" || name == "WebFetch" {
                        let is_search = name == "WebSearch";
                        let detail = input[if is_search { "query" } else { "url" }].as_str().unwrap_or_default().to_owned();
                        let mut msg = if is_search {
                            make_web_search_searching_message(&task_id, &request_id, detail.clone())
                        } else {
                            make_web_fetch_fetching_message(&task_id, &request_id, vec![detail.clone()])
                        };
                        let msg_id = Uuid::new_v4().to_string();
                        msg.id = msg_id.clone();
                        native.insert(id, Native::Web { msg_id, detail, is_search });
                        yield Ok(make_add_messages_event(&task_id, vec![msg]));
                        continue;
                    }
                    let tool = api::message::tool_call::Tool::CallMcpTool(
                        api::message::tool_call::CallMcpTool {
                            name,
                            args: Some(super::json_to_struct(&input)),
                            server_id: String::new(),
                        },
                    );
                    let call_id = format!("{}{id}", super::TOOL_ID_PREFIX);
                    let msg = make_tool_call_message(&task_id, &request_id, &call_id, tool);
                    yield Ok(make_add_messages_event(&task_id, vec![msg]));
                }
                AgentEvent::ToolFinished { id, is_error, output } => {
                    match native.remove(&id) {
                        Some(Native::Todo) => continue,
                        Some(Native::Web { msg_id, detail, is_search }) => {
                            let json = if is_error {
                                serde_json::json!({"status": "error"})
                            } else {
                                serde_json::json!({"results": output, "url": detail})
                            };
                            let mut done = if is_search {
                                make_web_search_status_from_result(&task_id, &request_id, &detail, &json)
                            } else {
                                make_web_fetch_status_from_result(&task_id, &request_id, &[detail], &json)
                            };
                            done.id = msg_id;
                            let mask = if is_search { "web_search" } else { "web_fetch" };
                            yield Ok(make_update_message_event(&task_id, done, vec![mask.to_owned()]));
                            continue;
                        }
                        None => {}
                    }
                    let call_id = format!("{}{id}", super::TOOL_ID_PREFIX);
                    // Persisted with the conversation so reopened chats still show the result.
                    let msg = make_tool_result_message(&task_id, &request_id, &call_id, is_error, &output);
                    yield Ok(make_add_messages_event(&task_id, vec![msg]));
                    super::record_outcome(super::ToolOutcome {
                        tool_call_id: call_id,
                        task_id: task_id.clone(),
                        is_error,
                        output,
                    });
                }
                AgentEvent::PermissionRequest { tool_name, input, .. } => {
                    open = Open::None;
                    let note = format!("*Waiting for your approval: {}*", super::tool_title(&tool_name, &input));
                    let msg = make_agent_output_message(&task_id, &request_id, note);
                    yield Ok(make_add_messages_event(&task_id, vec![msg]));
                }
                AgentEvent::Commands(names) => super::commands::set_if_empty(names),
                AgentEvent::Usage { line, cost_usd } => {
                    open = Open::None;
                    let line = match cost_usd {
                        Some(c) => format!("{line} · chat {}", chat_total(&session_id, c)),
                        None => line,
                    };
                    let msg = make_agent_output_message(&task_id, &request_id, format!("*{line}*"));
                    yield Ok(make_add_messages_event(&task_id, vec![msg]));
                }
                AgentEvent::Done => break 'turn,
                AgentEvent::Error(e) => {
                    failed = Some(e);
                    break 'turn;
                }
            }
            }
        }
        super::permissions::clear();
        match failed {
            Some(e) => yield Err(Arc::new(AIApiError::Other(anyhow::anyhow!(e)))),
            None => yield Ok(make_finished_done(None)),
        }
    })
}

/// A finished tool as a structured MCP-style result message.
fn make_tool_result_message(
    task_id: &str,
    request_id: &str,
    tool_call_id: &str,
    is_error: bool,
    output: &str,
) -> api::Message {
    use api::call_mcp_tool_result as r;
    let result = if is_error {
        r::Result::Error(r::Error { message: output.to_owned() })
    } else {
        r::Result::Success(r::Success {
            results: vec![r::success::Result {
                result: Some(r::success::result::Result::Text(r::success::result::Text {
                    text: output.to_owned(),
                })),
            }],
        })
    };
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::ToolCallResult(api::message::ToolCallResult {
            tool_call_id: tool_call_id.to_owned(),
            context: None,
            result: Some(api::message::tool_call_result::Result::CallMcpTool(api::CallMcpToolResult {
                result: Some(result),
            })),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    }
}

/// Marks a thinking block finished so Warp can show how long it took.
fn make_reasoning_done_event(
    task_id: &str,
    request_id: &str,
    message_id: &str,
    took: std::time::Duration,
) -> api::ResponseEvent {
    let msg = api::Message {
        id: message_id.to_owned(),
        task_id: task_id.to_owned(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentReasoning(api::message::AgentReasoning {
            reasoning: String::new(),
            finished_duration: Some(prost_types::Duration {
                seconds: took.as_secs() as i64,
                nanos: took.subsec_nanos() as i32,
            }),
        })),
        request_id: request_id.to_owned(),
        timestamp: None,
    };
    make_update_message_event(task_id, msg, vec!["agent_reasoning.finished_duration".to_owned()])
}

/// A tool shown with a native Warp card instead of the generic tool card.
enum Native {
    Todo,
    Web { msg_id: String, detail: String, is_search: bool },
}

/// Running cost of one chat (CLI session), as `$1.23`.
fn chat_total(session_id: &str, turn_cost: f64) -> String {
    static TOTALS: std::sync::Mutex<Vec<(String, f64)>> = std::sync::Mutex::new(Vec::new());
    let mut totals = TOTALS.lock().unwrap();
    let total = match totals.iter_mut().find(|(id, _)| id == session_id) {
        Some((_, t)) => {
            *t += turn_cost;
            *t
        }
        None => {
            totals.push((session_id.to_owned(), turn_cost));
            turn_cost
        }
    };
    format!("${total:.2}")
}
