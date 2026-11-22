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
    make_finished_done, make_reasoning_message, make_tool_call_message, make_user_query_message, AppendKind,
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
            .run_turn(TurnRequest { prompt, cwd, model, session_id, resume })
            .take_until(cancellation_rx);
        let mut open = Open::None;
        let mut failed = None;
        while let Some(ev) = events.next().await {
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
                        yield Ok(make_add_messages_event(&task_id, vec![msg]));
                    }
                },
                AgentEvent::ToolStarted { id, name, input } => {
                    open = Open::None;
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
                    super::record_outcome(super::ToolOutcome {
                        tool_call_id: format!("{}{id}", super::TOOL_ID_PREFIX),
                        task_id: task_id.clone(),
                        is_error,
                        output,
                    });
                }
                AgentEvent::Done => break,
                AgentEvent::Error(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        match failed {
            Some(e) => yield Err(Arc::new(AIApiError::Other(anyhow::anyhow!(e)))),
            None => yield Ok(make_finished_done(None)),
        }
    })
}
