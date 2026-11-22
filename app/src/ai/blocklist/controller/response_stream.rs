use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::ai::api_error::AIApiError;
use anyhow::anyhow;
use chrono::{DateTime, Local, TimeDelta};
use futures::channel::oneshot;
use futures_util::StreamExt;
use uuid::Uuid;
use warp_multi_agent_api::response_event;
use warpui::{Entity, ModelContext};

use crate::{
    ai::agent::{
        api::{self, ConvertToAPITypeError},
        conversation::AIConversationId,
        AIAgentInput, AIIdentifiers, CancellationReason,
    },
    ai::blocklist::BlocklistAIHistoryModel,
    ai::byop_readiness::BlockedByopReadinessError,
    network::NetworkStatus,
    report_error, send_telemetry_from_ctx,
};
use warpui::SingletonEntity;

/// Request-dispatch parameters for the BYOP path. Extracted from LLMId, settings and conversation, then
/// handed to the spawn closure in one go (ctx cannot cross an await boundary).
pub(super) struct PendingTitleGeneration {
    pub(super) input: crate::ai::agent_providers::chat_stream::TitleGenInput,
    pub(super) user_query: String,
    pub(super) task_id: String,
}

struct ByopDispatch {
    base_url: String,
    api_key: String,
    model_id: String,
    /// Explicitly specified API protocol type; chat_stream maps it to genai AdapterKind accordingly.
    api_type: crate::settings::AgentProviderApiType,
    /// Provider-level reasoning effort preference. With `Auto`, no effort is passed to genai,
    /// and the adapter infers it from the model name suffix; non-Auto values are injected after passing the client capability gate.
    reasoning_effort: crate::settings::ReasoningEffortSetting,
    extra_headers: Vec<(String, String)>,
    /// Local conversation id used to resolve `{conversationId}` in headers (differs from the server token).
    conversation_id: Option<AIConversationId>,
    /// Root task id of the conversation — must be a locally registered id,
    /// otherwise downstream `Action::AddMessagesToTask` can't find it in task_store and fails with `TaskNotFound`.
    root_task_id: String,
    /// Task id that this turn's model output should be written to. For a normal conversation it equals the root task; for later turns of a CLI subagent it is a subtask.
    target_task_id: String,
    /// Whether to emit `CreateTask` to upgrade the Optimistic root to a Server task.
    /// Only needed on the first turn (root task has no source yet); sending again triggers `UnexpectedUpgrade`.
    needs_create_task: bool,
    /// Title-generation model parameters. Filled only on the first turn (needs_create_task) when the active title_model
    /// decodes to a valid BYOP id; otherwise background title generation is not started.
    title_gen: Option<TitleGenParams>,
    /// `command_id` bound for the LRC scenario (= LRC block id string).
    lrc_command_id: Option<String>,
    /// Whether chat_stream needs to synthesize a subagent CreateTask to upgrade the optimistic CLI subtask.
    lrc_should_spawn_subagent: bool,
    /// Context window of the selected model (tokens). 0/None ⇒ the user left it empty and the catalog has none either,
    /// chat_stream skips the context_window_usage computation and the UI keeps the 100% placeholder.
    context_window: Option<u32>,
    /// Attachment caps that already reflect the user's settings (image/pdf/audio tri-state override).
    /// Computed by `resolve_for_model`. The UI display and runtime behavior reference the same caps.
    attachment_caps: crate::ai::agent_providers::attachment_caps::AttachmentCaps,
}

/// BYOP config dedicated to title generation (may be the same provider as the main base model or a different one).
pub(crate) struct TitleGenParams {
    pub base_url: String,
    pub api_key: String,
    pub model_id: String,
    pub api_type: crate::settings::AgentProviderApiType,
    pub reasoning_effort: crate::settings::ReasoningEffortSetting,
    /// Extra headers of the title model's provider (may differ from the main provider).
    pub extra_headers: Vec<(String, String)>,
}

/// Dispatch info for `cli:` models (Claude Code and other CLI backends).
struct CliDispatch {
    backend: &'static dyn crate::ai::cli_agents::AgentBackend,
    model: Option<String>,
    task_id: String,
    needs_create_task: bool,
}

fn cli_dispatch_info(
    params: &api::RequestParams,
    ai_identifiers: &AIIdentifiers,
    ctx: &warpui::AppContext,
) -> Option<CliDispatch> {
    let (backend, model) = crate::ai::cli_agents::decode_model_id(params.model.as_str())?;
    let conversation_id = ai_identifiers.client_conversation_id.as_ref()?;
    let conversation = BlocklistAIHistoryModel::as_ref(ctx).conversation(conversation_id)?;
    Some(CliDispatch {
        backend,
        model,
        task_id: params
            .byop_target_task_id
            .clone()
            .unwrap_or_else(|| conversation.get_root_task_id().to_string()),
        needs_create_task: conversation.compute_active_tasks().is_empty(),
    })
}

fn cli_response_stream(
    cli: CliDispatch,
    params: api::RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> Result<api::ResponseStream, ConvertToAPITypeError> {
    Ok(crate::ai::cli_agents::stream::generate_cli_output(
        crate::ai::cli_agents::stream::CliTurnInput {
            params,
            backend: cli.backend,
            model: cli.model,
            task_id: cli.task_id,
            needs_create_task: cli.needs_create_task,
            cancellation_rx,
        },
    ))
}

fn byop_dispatch_info(
    params: &api::RequestParams,
    ai_identifiers: &AIIdentifiers,
    ctx: &warpui::AppContext,
) -> Option<ByopDispatch> {
    let (provider, api_key, model_id) =
        crate::ai::agent_providers::lookup_byop(ctx, &params.model)?;
    let extra_headers = provider.extra_headers.clone();
    // Find the current model entry in provider.models and take its context_window (tokens).
    // 0 counts as unfilled and goes to the None branch later ⇒ chat_stream does not compute usage.
    let context_window = provider
        .models
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| m.context_window)
        .filter(|n| *n > 0);
    let conversation_id = ai_identifiers.client_conversation_id.as_ref()?;
    let history = BlocklistAIHistoryModel::as_ref(ctx);
    let conversation = history.conversation(conversation_id)?;
    let root_task_id = conversation.get_root_task_id().to_string();
    let target_task_id = params
        .byop_target_task_id
        .clone()
        .unwrap_or_else(|| root_task_id.clone());
    let (lrc_should_spawn_subagent, lrc_command_id) =
        resolve_lrc_subagent_params(conversation, &root_task_id, &target_task_id, params);
    // compute_active_tasks only returns tasks with `task.source().is_some()` —
    // so non-empty ⇒ root has already been upgraded to Server state; do not emit CreateTask again.
    let needs_create_task = conversation.compute_active_tasks().is_empty();

    // Title generation: triggered only on the first turn (avoids re-titling every turn).
    // Resolve the active title_model: it may be base_model itself, or another BYOP model chosen separately by the user.
    // If either model is not BYOP-encoded (e.g. a fallback to a non-BYOP default), skip — Warp's main path is all BYOP,
    // and when it actually falls back to base, base itself is BYOP.
    let llm_prefs = crate::ai::llms::LLMPreferences::as_ref(ctx);
    let title_gen = if needs_create_task {
        let title_id = llm_prefs.get_active_title_model(ctx, None).id.clone();
        crate::ai::agent_providers::lookup_byop(ctx, &title_id).map(
            |(t_provider, t_api_key, t_model_id)| {
                let t_effort =
                    llm_prefs.get_reasoning_effort(None, t_provider.api_type, &t_model_id);
                TitleGenParams {
                    base_url: t_provider.base_url,
                    api_key: t_api_key,
                    model_id: t_model_id,
                    api_type: t_provider.api_type,
                    reasoning_effort: t_effort,
                    extra_headers: t_provider.extra_headers,
                }
            },
        )
    } else {
        None
    };

    let reasoning_effort = llm_prefs.get_reasoning_effort(None, provider.api_type, &model_id);
    let attachment_caps = provider
        .models
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| {
            crate::ai::agent_providers::attachment_caps::resolve_for_model(
                &provider.id,
                provider.api_type,
                m,
            )
        })
        .unwrap_or_else(|| {
            log::warn!(
                "[byop] model '{}' not found in provider.models — falling back to caps_for (user overrides ignored)",
                model_id
            );
            crate::ai::agent_providers::attachment_caps::caps_for(provider.api_type, &model_id)
        });
    Some(ByopDispatch {
        base_url: provider.base_url,
        api_key,
        model_id,
        api_type: provider.api_type,
        reasoning_effort,
        extra_headers,
        conversation_id: params.byop_conversation_id,
        root_task_id,
        target_task_id,
        needs_create_task,
        title_gen,
        lrc_command_id,
        lrc_should_spawn_subagent,
        context_window,
        attachment_caps,
    })
}

fn pending_title_generation_from_byop(
    params: &api::RequestParams,
    byop: &ByopDispatch,
) -> Option<PendingTitleGeneration> {
    let title_gen = byop.title_gen.as_ref()?;
    let user_query = params.input.iter().find_map(|input| {
        if let AIAgentInput::UserQuery { query, .. } = input {
            Some(query.clone())
        } else {
            None
        }
    })?;

    Some(PendingTitleGeneration {
        input: crate::ai::agent_providers::chat_stream::TitleGenInput {
            base_url: title_gen.base_url.clone(),
            api_key: title_gen.api_key.clone(),
            model_id: title_gen.model_id.clone(),
            api_type: title_gen.api_type,
            reasoning_effort: title_gen.reasoning_effort,
            extra_headers: title_gen.extra_headers.clone(),
            conversation_id: byop.conversation_id,
        },
        user_query,
        task_id: byop.root_task_id.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResponseStreamId(String);

impl ResponseStreamId {
    pub fn new_local() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn for_shared_session(init_event: &response_event::StreamInit) -> Self {
        // Make the stream ID unique per viewing by appending a local UUID
        // This prevents collisions when replaying the same conversation multiple times
        // (either on close-and-reopen or when viewing the same shared session from multiple terminals)
        Self(format!("{}-{}", init_event.request_id, Uuid::new_v4()))
    }

    #[cfg(test)]
    pub fn new_for_test() -> Self {
        Self::new_local()
    }
}

/// Model wrapping an agent API response stream.
///
/// Emits events when the output corresponding to the stream is updated, typically after receiving
/// each response chunk.
///
/// Handles retries internally - retries are only attempted if no ClientActions events have been
/// received yet, ensuring we don't retry after the AI has started executing actions.
pub struct ResponseStream {
    id: ResponseStreamId,
    params: api::RequestParams,
    retry_count: usize,
    start_time: DateTime<Local>,
    time_to_latest_event: TimeDelta,
    cancellation_tx: Option<oneshot::Sender<()>>,
    /// Store the original error for telemetry when retries succeed
    original_error: Option<String>,
    /// Track whether we've received any client actions
    /// If true, we cannot retry on subsequent errors since actions may have been executed
    has_received_client_actions: bool,
    /// AI identifiers for telemetry emission
    ai_identifiers: AIIdentifiers,

    /// Whether this request can attempt to resume the conversation on error.
    /// This is true for all requests except those that are themselves the result of a resume
    /// triggered by a previous error.
    can_attempt_resume_on_error: bool,

    pending_title_generation: Option<PendingTitleGeneration>,

    /// Whether we should attempt to resume the conversation after the stream finishes.
    ///
    /// This is set when we receive a retryable error after client actions have been received
    /// and `can_attempt_resume_on_error` is true.
    should_resume_conversation_after_stream_finished: bool,

    /// Unique, internal id for the current request.
    ///
    /// This ensures that the model never emits events for a request that was already cancelled (or
    /// retried) and is still receiving lagging events.
    ///
    /// Note this is unique compared to `id`; this is unique across retry requests while the response
    /// stream id remains stable.
    current_request_id: Option<Uuid>,
}

impl ResponseStream {
    pub fn new(
        params: api::RequestParams,
        ai_identifiers: AIIdentifiers,
        can_attempt_resume_on_error: bool,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        Self::new_with_id(
            ResponseStreamId::new_local(),
            params,
            ai_identifiers,
            can_attempt_resume_on_error,
            ctx,
        )
    }

    pub fn new_with_id(
        id: ResponseStreamId,
        params: api::RequestParams,
        ai_identifiers: AIIdentifiers,
        can_attempt_resume_on_error: bool,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        let start_time = Local::now();

        let request_id = Uuid::new_v4();
        let params_clone = params.clone();
        // BYOP path: if the selected base model is an LLMId encoded from a user-defined provider,
        // take (provider, api_key, model_id, root_task_id) from ctx before spawning
        // and go through custom chat completions. Otherwise use warp's own multi-agent endpoint (the original path).
        let cli_dispatch = cli_dispatch_info(&params, &ai_identifiers, ctx);
        let byop_dispatch = byop_dispatch_info(&params, &ai_identifiers, ctx);
        let pending_title_generation = byop_dispatch
            .as_ref()
            .and_then(|byop| pending_title_generation_from_byop(&params, byop));
        let _ = ctx.spawn(
            async move {
                if let Some(cli) = cli_dispatch {
                    cli_response_stream(cli, params_clone, cancellation_rx)
                } else if let Some(byop) = byop_dispatch {
                    crate::ai::agent_providers::chat_stream::generate_byop_output(
                        crate::ai::agent_providers::chat_stream::ByopOutputInput {
                            params: params_clone,
                            base_url: byop.base_url,
                            api_key: byop.api_key,
                            model_id: byop.model_id,
                            api_type: byop.api_type,
                            reasoning_effort: byop.reasoning_effort,
                            extra_headers: byop.extra_headers,
                            conversation_id: byop.conversation_id,
                            task_id: byop.root_task_id,
                            target_task_id: byop.target_task_id,
                            needs_create_task: byop.needs_create_task,
                            lrc_command_id: byop.lrc_command_id,
                            lrc_should_spawn_subagent: byop.lrc_should_spawn_subagent,
                            context_window: byop.context_window,
                            cancellation_rx,
                            attachment_caps: byop.attachment_caps,
                        },
                    )
                    .await
                } else {
                    byop_required_response_stream(cancellation_rx).await
                }
            },
            move |me, stream, ctx| {
                me.handle_response_stream_result(request_id, stream, ctx);
            },
        );
        Self {
            id,
            params: params.clone(),
            start_time,
            time_to_latest_event: TimeDelta::seconds(0),
            cancellation_tx: Some(cancellation_tx),
            retry_count: 0,
            original_error: None,
            has_received_client_actions: false,
            ai_identifiers,
            can_attempt_resume_on_error,
            pending_title_generation,
            should_resume_conversation_after_stream_finished: false,
            current_request_id: Some(request_id),
        }
    }

    pub(super) fn take_pending_title_generation(&mut self) -> Option<PendingTitleGeneration> {
        self.pending_title_generation.take()
    }

    pub fn id(&self) -> &ResponseStreamId {
        &self.id
    }

    pub fn is_lrc_tag_in_request(&self) -> bool {
        self.params.lrc_should_spawn_subagent
    }

    /// Warp BYOP local conversation compaction: returns whether this stream is running SummarizeConversation,
    /// plus the overflow flag. The controller uses it in the Done branch of handle_response_stream_finished
    /// to call commit_summarization and land the summary into conversation.compaction_state.
    pub fn summarization_overflow(&self) -> Option<bool> {
        self.params.input.iter().find_map(|input| match input {
            crate::ai::agent::AIAgentInput::SummarizeConversation { overflow, .. } => {
                Some(*overflow)
            }
            _ => None,
        })
    }

    /// Returns true if we should attempt to resume the conversation after the stream finishes.
    pub fn should_resume_conversation_after_stream_finished(&self) -> bool {
        self.should_resume_conversation_after_stream_finished
    }

    /// Helper function to emit AgentModeError telemetry for error that is retryable (not user visible).
    fn emit_retryable_agent_mode_error_telemetry(
        &self,
        error: String,
        ctx: &mut ModelContext<Self>,
    ) {
        send_telemetry_from_ctx!(
            crate::TelemetryEvent::AgentModeError {
                identifiers: self.ai_identifiers.clone(),
                error,
                is_user_visible: false,
                will_attempt_to_resume: false,
            },
            ctx
        );
    }

    fn retry(&mut self, ctx: &mut ModelContext<Self>) {
        self.retry_count += 1;
        self.has_received_client_actions = false; // Reset for the new attempt

        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        if let Some(old_cancellation_tx) = self.cancellation_tx.take() {
            let _ = old_cancellation_tx.send(());
        }
        self.cancellation_tx = Some(cancellation_tx);

        let request_id = Uuid::new_v4();
        self.current_request_id = Some(request_id);
        let params = self.params.clone();
        let cli_dispatch = cli_dispatch_info(&params, &self.ai_identifiers, ctx);
        let byop_dispatch = byop_dispatch_info(&params, &self.ai_identifiers, ctx);
        let _ = ctx.spawn(
            async move {
                if let Some(cli) = cli_dispatch {
                    cli_response_stream(cli, params, cancellation_rx)
                } else if let Some(byop) = byop_dispatch {
                    crate::ai::agent_providers::chat_stream::generate_byop_output(
                        crate::ai::agent_providers::chat_stream::ByopOutputInput {
                            params,
                            base_url: byop.base_url,
                            api_key: byop.api_key,
                            model_id: byop.model_id,
                            api_type: byop.api_type,
                            reasoning_effort: byop.reasoning_effort,
                            extra_headers: byop.extra_headers,
                            conversation_id: byop.conversation_id,
                            task_id: byop.root_task_id,
                            target_task_id: byop.target_task_id,
                            needs_create_task: byop.needs_create_task,
                            lrc_command_id: byop.lrc_command_id,
                            lrc_should_spawn_subagent: byop.lrc_should_spawn_subagent,
                            context_window: byop.context_window,
                            cancellation_rx,
                            attachment_caps: byop.attachment_caps,
                        },
                    )
                    .await
                } else {
                    byop_required_response_stream(cancellation_rx).await
                }
            },
            move |me, stream, ctx| {
                me.handle_response_stream_result(request_id, stream, ctx);
            },
        );
    }

    /// Cancels the stream. The conversation_id is preserved in the emitted event for async handling.
    pub(super) fn cancel(
        &mut self,
        reason: CancellationReason,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        self.current_request_id = None;
        let Some(cancellation_tx) = self.cancellation_tx.take() else {
            return;
        };
        let _ = cancellation_tx.send(());
        ctx.emit(ResponseStreamEvent::AfterStreamFinished {
            cancellation: Some(StreamCancellation {
                reason,
                conversation_id,
            }),
        });
    }

    fn handle_response_stream_result(
        &mut self,
        request_id: Uuid,
        stream_result: Result<api::ResponseStream, ConvertToAPITypeError>,
        ctx: &mut ModelContext<Self>,
    ) {
        match stream_result {
            Ok(stream) => {
                ctx.spawn_stream_local(
                    stream,
                    move |me, event, ctx| {
                        me.handle_response_stream_event(request_id, event, ctx);
                    },
                    move |me, ctx| {
                        me.on_response_stream_complete(request_id, ctx);
                    },
                );
            }
            Err(e) => {
                log::error!("Failed to send request to multi-agent API: {e:?}");
                let api_error = convert_to_api_error(e);
                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Err(
                    Arc::new(api_error),
                ))));
                self.on_response_stream_complete(request_id, ctx);
            }
        }
    }

    fn handle_response_stream_event(
        &mut self,
        request_id: Uuid,
        event: api::Event,
        ctx: &mut ModelContext<Self>,
    ) {
        if self.current_request_id.is_none_or(|id| id != request_id) {
            return;
        }
        self.time_to_latest_event = Local::now().signed_duration_since(self.start_time);

        match &event {
            Ok(response_event) => {
                if let Some(event_type) = &response_event.r#type {
                    match event_type {
                        warp_multi_agent_api::response_event::Type::Init(init_event) => {
                            // Capture server_output_id from StreamInit event
                            self.ai_identifiers.server_output_id =
                                Some(crate::ai::agent::ServerOutputId::new(
                                    init_event.request_id.clone(),
                                ));
                        }
                        warp_multi_agent_api::response_event::Type::ClientActions(_) => {
                            // Mark that we've received client actions
                            self.has_received_client_actions = true;
                        }
                        warp_multi_agent_api::response_event::Type::Finished(finished_event) => {
                            // Emit retry success telemetry on successful completion
                            if matches!(
                                finished_event.reason,
                                Some(warp_multi_agent_api::response_event::stream_finished::Reason::Done(_)) | None
                            ) {
                                // Emit retry success telemetry if this was a successful completion after retries
                                if self.retry_count > 0 {
                                    if let Some(original_error) = &self.original_error {
                                        send_telemetry_from_ctx!(
                                            crate::TelemetryEvent::AgentModeRequestRetrySucceeded {
                                                identifiers: self.ai_identifiers.clone(),
                                                retry_count: self.retry_count,
                                                original_error: original_error.clone(),
                                            },
                                            ctx
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(event)));
            }
            Err(e) => {
                // Store original error if this is the first error
                if self.retry_count == 0 {
                    self.original_error = Some(format!("{e:?}"));
                }

                // Only retry if:
                // 1. We haven't received any client actions yet (this is the first event or only init events)
                // 2. The error is retryable
                // 3. We haven't exceeded max retries
                // 4. We're online
                const MAX_RETRIES: usize = 3;
                let network_status = NetworkStatus::as_ref(ctx);
                let is_online = network_status.is_online();
                let is_retryable = e.is_retryable();

                let should_retry = !self.has_received_client_actions
                    && is_retryable
                    && self.retry_count < MAX_RETRIES
                    && is_online;

                if should_retry {
                    log::warn!(
                        "MultiAgent request failed, retrying (attempt {}/{}) - Error: {e:?}",
                        self.retry_count + 1,
                        MAX_RETRIES
                    );
                    // Only emit error telemetry here if we're retrying.
                    // Final errors that aren't being retried are emitted elsewhere.
                    self.emit_retryable_agent_mode_error_telemetry(format!("{e:?}"), ctx);
                    self.retry(ctx);
                    // Don't emit the error event, we're retrying
                    // TODO: emit a separate event if controller needs to know about failures that are being retried
                    return;
                }

                // If we can't retry (because client actions were received) but the error is
                // retryable and we're allowed to attempt a resume, signal that the controller
                // should resume the conversation after the stream completes.
                let should_attempt_resume = self.has_received_client_actions
                    && is_retryable
                    && self.can_attempt_resume_on_error;
                if should_attempt_resume {
                    self.should_resume_conversation_after_stream_finished = true;
                }

                log::warn!(
                    "MultiAgent request failed after {} retries: has_received_client_actions={}, is_retryable={}, is_online={is_online}",
                    self.retry_count,
                    self.has_received_client_actions,
                    e.is_retryable()
                );
                report_error!(anyhow!(e.clone()).context(format!(
                    "MultiAgent request failed after {} retries",
                    self.retry_count
                )));

                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(event)));
            }
        }
    }

    fn on_response_stream_complete(&mut self, request_id: Uuid, ctx: &mut ModelContext<Self>) {
        if self.current_request_id.is_none_or(|id| id != request_id) {
            return;
        }
        ctx.emit(ResponseStreamEvent::AfterStreamFinished { cancellation: None });
        self.cancellation_tx = None;
    }
}

fn convert_to_api_error(error: ConvertToAPITypeError) -> AIApiError {
    match &error {
        ConvertToAPITypeError::Other(inner)
            if inner.downcast_ref::<BlockedByopReadinessError>().is_some() =>
        {
            let blocked = inner
                .downcast_ref::<BlockedByopReadinessError>()
                .expect("checked blocked readiness error");
            AIApiError::Other(BlockedByopReadinessError::new(blocked.category()).into())
        }
        ConvertToAPITypeError::Ignore
        | ConvertToAPITypeError::Unimplemented(_)
        | ConvertToAPITypeError::Other(_) => AIApiError::Other(anyhow!(error.to_string())),
    }
}

#[derive(Debug)]
pub struct Consumable<T> {
    value: Rc<RefCell<Option<T>>>,
}

impl<T> Consumable<T> {
    fn new(value: T) -> Self {
        Consumable {
            value: Rc::new(RefCell::new(Some(value))),
        }
    }

    pub(super) fn consume(&self) -> Option<T> {
        self.value.borrow_mut().take()
    }
}

impl<T> Clone for Consumable<T> {
    fn clone(&self) -> Self {
        Consumable {
            value: Rc::clone(&self.value),
        }
    }
}

/// Cancellation context preserved for async event handling.
/// Includes conversation_id because truncation can remove exchange mappings before the event is processed.
#[derive(Debug, Clone)]
pub struct StreamCancellation {
    pub reason: CancellationReason,
    pub conversation_id: AIConversationId,
}

#[derive(Debug, Clone)]
pub enum ResponseStreamEvent {
    ReceivedEvent(Consumable<api::Event>),
    AfterStreamFinished {
        /// Some for cancellation (with context), None for natural completion (uses dynamic lookup).
        cancellation: Option<StreamCancellation>,
    },
}

impl Entity for ResponseStream {
    type Event = ResponseStreamEvent;
}

async fn byop_required_response_stream(
    cancellation_rx: oneshot::Receiver<()>,
) -> Result<api::ResponseStream, ConvertToAPITypeError> {
    log::debug!("No BYOP provider selected for Warp agent request");
    let error_stream = futures::stream::once(async {
        Err(Arc::new(AIApiError::Other(anyhow!(
            "Warp requires a configured BYOP provider in Settings"
        ))))
    })
    .take_until(cancellation_rx);
    Ok(Box::pin(error_stream))
}

fn resolve_lrc_subagent_params(
    conversation: &crate::ai::agent::conversation::AIConversation,
    root_task_id: &str,
    target_task_id: &str,
    params: &api::RequestParams,
) -> (bool, Option<String>) {
    // If `target_task_id` is an Optimistic CLI subagent not yet upgraded to Server,
    // BYOP must take the subagent spawn/upgrade path first, writing the preceding messages to root and upgrading that subtask;
    // otherwise sending `AddMessagesToTask` directly to an Optimistic subtask deterministically fails (`TaskNotInitialized`).
    let target_task =
        conversation.get_task(&crate::ai::agent::task::TaskId::new(target_task_id.to_owned()));
    let optimistic_cli_subagent_block_id = target_task.and_then(|task| {
        (task.id().to_string() != root_task_id && task.source().is_none())
            .then(|| task.cli_subagent_block_id())
            .flatten()
    });
    let lrc_should_spawn_subagent =
        params.lrc_should_spawn_subagent || optimistic_cli_subagent_block_id.is_some();
    let lrc_command_id = params
        .lrc_command_id
        .clone()
        .or_else(|| optimistic_cli_subagent_block_id.map(|block_id| block_id.to_string()));
    (lrc_should_spawn_subagent, lrc_command_id)
}

#[cfg(test)]
mod tests {
    use super::resolve_lrc_subagent_params;
    use crate::ai::agent::api;
    use crate::ai::blocklist::BlocklistAIHistoryModel;
    use crate::terminal::model::BlockId;
    use warpui::{App, EntityId};

    #[test]
    fn optimistic_cli_subagent_target_forces_byop_subagent_spawn_and_command_id() {
        App::test((), |mut app| async move {
            let history_model =
                app.add_singleton_model(|_| BlocklistAIHistoryModel::new(vec![], &[]));
            let terminal_view_id = EntityId::new();
            let lrc_block_id = BlockId::from("precmd-179393590118629-10".to_string());

            let (root_task_id, subtask_id, should_spawn, command_id) =
                history_model.update(&mut app, |history_model, ctx| {
                    let conversation_id =
                        history_model.start_new_conversation(terminal_view_id, false, false, ctx);
                    let root_task_id = history_model
                        .conversation(&conversation_id)
                        .unwrap()
                        .get_root_task_id()
                        .to_string();
                    let subtask_id = history_model
                        .create_cli_subagent_task_for_conversation(
                            lrc_block_id.clone(),
                            conversation_id,
                            terminal_view_id,
                            ctx,
                        )
                        .unwrap()
                        .to_string();
                    let conversation = history_model.conversation(&conversation_id).unwrap();

                    let mut params = api::RequestParams::new_for_test(vec![], vec![]);
                    params.byop_target_task_id = Some(subtask_id.clone());
                    params.lrc_should_spawn_subagent = false;
                    params.lrc_command_id = None;

                    let (should_spawn, command_id) =
                        resolve_lrc_subagent_params(conversation, &root_task_id, &subtask_id, &params);
                    (root_task_id, subtask_id, should_spawn, command_id)
                });

            assert_ne!(subtask_id, root_task_id);
            assert!(should_spawn);
            assert_eq!(command_id, Some(lrc_block_id.to_string()));
        });
    }
}
