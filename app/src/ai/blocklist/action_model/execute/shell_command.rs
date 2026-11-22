use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::{select, FutureExt};
use futures_lite::pin;
use itertools::Itertools;
use parking_lot::FairMutex;
use warp_core::command::ExitCode;
use warp_core::execution_mode::AppExecutionMode;
use warp_util::path::ShellFamily;
use warpui::r#async::{Spawnable, Timer};
use warpui::{Entity, EntityId, ModelContext, ModelHandle, SingletonEntity};

use crate::ai::agent::{
    AIAgentActionId, AIAgentActionType, AIAgentPtyWriteMode, ReadShellCommandOutputResult,
    RequestCommandOutputResult, ShellCommandDelay, ShellCommandError,
    TransferShellCommandControlToUserResult, WriteToLongRunningShellCommandResult,
};
use crate::ai::blocklist::permissions::CommandExecutionPermission;
use crate::ai::blocklist::BlocklistAIPermissions;
use crate::ai::execution_profiles::WriteToPtyPermission;
use crate::terminal::event::BlockMetadataReceivedEvent;
use crate::terminal::model::block::{
    formatted_terminal_contents_for_input, Block, BlockId, CURSOR_MARKER,
};
use crate::terminal::shell::ShellType;
use crate::terminal::ssh::util::parse_interactive_ssh_command;
use crate::{
    ai::agent::AIAgentActionResultType,
    terminal::{
        model::session::active_session::ActiveSession,
        model_events::{ModelEvent, ModelEventDispatcher},
        TerminalModel,
    },
};
use crate::{send_telemetry_from_ctx, TelemetryEvent};

use super::{ActionExecution, AnyActionExecution, ExecuteActionInput, PreprocessActionInput};

/// Text returned to the agent for `run_shell_command` / related tools.
///
/// Prefer unobfuscated grid text; some shells / timing paths leave the primary serialization empty
/// while `output_to_string()` (displayed-output path) or the force-full path still has bytes.
///
/// Fallback: when output grids are empty (e.g. missing preexec / early-complete timing), extract
/// the stdout portion from the command grid (everything after the first line).
fn agent_shell_command_block_output(block: &Block) -> String {
    let primary = block.output_with_secrets_unobfuscated();
    if !primary.trim().is_empty() {
        return primary;
    }

    let displayed = block.output_to_string();
    if !displayed.trim().is_empty() {
        return displayed;
    }

    let forced = block.output_to_string_force_full_grid_contents();
    if !forced.trim().is_empty() {
        return forced;
    }

    let command_grid = block.command_with_secrets_unobfuscated(false);
    command_grid
        .split_once('\n')
        .map(|(_, output)| output.to_owned())
        .filter(|output| !output.trim().is_empty())
        .unwrap_or_default()
}

pub struct ShellCommandExecutor {
    active_session: ModelHandle<ActiveSession>,
    block_finished_senders: HashMap<BlockSelector, oneshot::Sender<()>>,
    /// Senders used by the `Check now` affordance to force a long-running shell command's
    /// pending poll future to resolve immediately with a fresh snapshot, bypassing the
    /// agent-set timeout.
    force_refresh_senders: HashMap<BlockSelector, oneshot::Sender<()>>,
    terminal_model: Arc<FairMutex<TerminalModel>>,
    terminal_view_id: EntityId,
    /// Sender to notify when user hands control back to agent after TransferShellCommandControlToUser.
    control_handback_sender: Option<oneshot::Sender<()>>,
}

impl ShellCommandExecutor {
    pub const MAX_WAIT_DURATION: Duration = Duration::from_secs(2);
    /// Maximum delay we will honor for any agent-requested wait. Applies both  
    /// to finite `ShellCommandDelay::Duration` requests and to  
    /// `ShellCommandDelay::OnCompletion`, which would otherwise wait indefinitely.  
    pub const MAX_AGENT_DELAY_DURATION: Duration = Duration::from_secs(120);
    /// "Pager hang guard": `wait_until_completion=true` (`ActionResultDelay::UntilCompletion`)
    /// path's final fallback timeout, only to prevent `turn_off_pager_for_command` from being
    /// bypassed by user shell config (`~/.zshrc` `export PAGER=less`, `git config --global core.pager less`, etc.)
    /// and leaving the agent hung forever in that extreme case.
    ///
    /// **Not** a general command timeout: 30 minutes is deliberately far above `MAX_AGENT_DELAY_DURATION` to avoid hurting
    /// legitimate long tasks such as `cargo build --release` / `docker build` / large `npm install`.
    /// When it fires, `is_preempted=true` marks the snapshot as preempted rather than "command completed".
    pub const MAX_UNTIL_COMPLETION_DURATION: Duration = Duration::from_secs(30 * 60);

    pub fn new(
        active_session: ModelHandle<ActiveSession>,
        terminal_model: Arc<FairMutex<TerminalModel>>,
        model_event_dispatcher: &ModelHandle<ModelEventDispatcher>,
        terminal_view_id: EntityId,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        ctx.subscribe_to_model(model_event_dispatcher, Self::handle_terminal_model_event);

        Self {
            active_session,
            terminal_model,
            block_finished_senders: HashMap::new(),
            force_refresh_senders: HashMap::new(),
            terminal_view_id,
            control_handback_sender: None,
        }
    }

    fn handle_terminal_model_event(&mut self, event: &ModelEvent, _ctx: &mut ModelContext<Self>) {
        // We wait for precmd for the block _after_ the requested command's block so that
        // downstream checks for current working directory are fresh. The precmd hook is when
        // the shell relays current working directory to warp.
        if let ModelEvent::BlockMetadataReceived(BlockMetadataReceivedEvent { .. }) = event {
            let model = self.terminal_model.lock();
            let block_finished_senders = self.block_finished_senders.drain().collect_vec();
            for (block_selector, block_finished_tx) in block_finished_senders.into_iter() {
                if let Some(block) = block_selector.get_block(&model) {
                    if block.is_command_finished() {
                        if let Err(e) = block_finished_tx.send(()) {
                            log::warn!(
                                "Failed to notify block completion for running requested command: {e:?}"
                            )
                        }
                    } else {
                        self.block_finished_senders
                            .insert(block_selector, block_finished_tx);
                    }
                }
            }
        }
    }

    pub(super) fn should_autoexecute(
        &self,
        input: ExecuteActionInput,
        ctx: &mut ModelContext<Self>,
    ) -> bool {
        let blocklist_permissions = BlocklistAIPermissions::as_ref(ctx);
        match &input.action.action {
            AIAgentActionType::RequestCommandOutput {
                command,
                is_read_only,
                is_risky,
                ..
            } => {
                let Some(escape_char) = self
                    .active_session
                    .as_ref(ctx)
                    .shell_type(ctx)
                    .map(|s| ShellFamily::from(s).escape_char())
                else {
                    return false;
                };
                let autoexecution_permission = blocklist_permissions.can_autoexecute_command(
                    &input.conversation_id,
                    command,
                    escape_char,
                    is_read_only.unwrap_or(false),
                    *is_risky,
                    Some(self.terminal_view_id),
                    ctx,
                );
                if let CommandExecutionPermission::Allowed(reason) = autoexecution_permission {
                    send_telemetry_from_ctx!(
                        TelemetryEvent::AutoexecutedAgentModeRequestedCommand { reason },
                        ctx
                    );
                } else if let CommandExecutionPermission::Denied(reason) = autoexecution_permission
                {
                    if AppExecutionMode::as_ref(ctx).is_autonomous() {
                        log::warn!(
                            "Command denied during autonomous execution, reason: {reason:?}"
                        );
                    }
                }
                autoexecution_permission.is_allowed()
            }
            AIAgentActionType::WriteToLongRunningShellCommand { block_id, .. } => {
                let terminal_model = self.terminal_model.lock();
                let block = terminal_model.block_list().block_with_id(block_id);

                if block.is_none_or(|block| block.finished()) {
                    // If the block is already finished, allow auto-execution - the finished output
                    // will be returned.
                    true
                } else {
                    let should_autoexecute = match blocklist_permissions.can_write_to_pty(
                        &input.conversation_id,
                        Some(self.terminal_view_id),
                        ctx,
                    ) {
                        WriteToPtyPermission::AlwaysAllow => true,
                        WriteToPtyPermission::AskOnFirstWrite => terminal_model
                            .block_list()
                            .active_block()
                            .has_agent_written_to_block(),
                        _ => false,
                    };

                    if should_autoexecute {
                        send_telemetry_from_ctx!(
                            TelemetryEvent::CLISubagentActionExecuted {
                                conversation_id: input.conversation_id,
                                block_id: block_id.clone(),
                                is_autoexecuted: true,
                            },
                            ctx
                        );
                    }

                    should_autoexecute
                }
            }
            AIAgentActionType::ReadShellCommandOutput { .. } => true,
            AIAgentActionType::TransferShellCommandControlToUser { .. } => false,
            _ => false,
        }
    }

    /// Wraps the command with a set of generic pager env vars so it does not enter a pager while **keeping the real exit code**.
    ///
    /// The previous implementation was `(cmd) | cat`; it made stdout a non-tty (so git/man/less etc. skip the pager),
    /// but in bash/zsh `$?` was overwritten by `cat`'s exit code (almost always 0), so the agent saw exit_code=0
    /// even when `cargo check` failed and drew wrong conclusions.
    ///
    /// This now uses `PAGER=cat GIT_PAGER=cat MANPAGER=cat` inside a subshell/script block,
    /// which covers the pager behavior of nearly all CLIs (git/man/bat/kubectl/psql/gh, ...) while the outer `$?` /
    /// `$LASTEXITCODE` comes from the command itself.
    ///
    /// **Two hardening measures** (given that `ActionResultDelay::UntilCompletion` has no short timeout, see #138):
    /// 1. `unset` first, then `export` (the equivalent syntax per shell), clearing values such as `PAGER=less`
    ///    inherited from the parent process and exported by the user's `~/.zshrc` / `~/.bashrc`, then set to `cat`. `export` alone
    ///    can still be overridden by a later `.zshenv` or similar in some edge cases.
    /// 2. Inject `GIT_CONFIG_COUNT=1 / GIT_CONFIG_KEY_0=core.pager / GIT_CONFIG_VALUE_0=cat`
    ///    as a second safeguard: testing shows that in git 2.54 the `GIT_PAGER` env var already takes priority over
    ///    `git config --global core.pager less` in `~/.gitconfig`, but with git >= 2.31 the
    ///    `GIT_CONFIG_COUNT` mechanism adds another in-process config override layer, guarding against future git
    ///    versions changing precedence or edge cases with third-party pager wrappers. It is harmless for non-git commands,
    ///    so there is no need to inspect the first token.
    ///
    /// Even if all of the above fail, the `MAX_UNTIL_COMPLETION_DURATION` fallback of `action_result_future`
    /// guarantees the agent never hangs **forever**.
    fn turn_off_pager_for_command(&self, command: &String, ctx: &mut ModelContext<Self>) -> String {
        wrap_command_without_pager(self.active_session.as_ref(ctx).shell_type(ctx), command)
    }

    pub(super) fn execute(
        &mut self,
        input: ExecuteActionInput,
        ctx: &mut ModelContext<Self>,
    ) -> impl Into<AnyActionExecution> {
        let active_session = self.active_session.as_ref(ctx).session(ctx);
        let current_working_directory = self
            .active_session
            .as_ref(ctx)
            .current_working_directory()
            .cloned();
        let model = self.terminal_model.lock();

        // Determine the action we want to take based on the input.
        let action_id = input.action.id.clone();

        let handle = ctx.handle();
        match &input.action.action {
            AIAgentActionType::RequestCommandOutput {
                command,
                uses_pager,
                wait_until_completion,
                ..
            } => {
                if model
                    .block_list()
                    .active_block()
                    .is_active_and_long_running()
                {
                    // If there is an active block, we can't execute another command.
                    return ActionExecution::Sync(AIAgentActionResultType::RequestCommandOutput(
                        RequestCommandOutputResult::CancelledBeforeExecution,
                    ));
                }
                // Completed commands in remote sessions run through the session's own command
                // executor (RemoteServerCommandExecutor, or the SSH ControlMaster fallback), not
                // the visible PTY: hidden agent blocks over SSH can otherwise report a wrong exit
                // code with empty output. Long-running commands stay on the PTY.
                if *wait_until_completion {
                    if let Some(session) = active_session.filter(|session| !session.is_local()) {
                        let command = command.clone();
                        let command_to_execute = command.clone();
                        drop(model);

                        return ActionExecution::new_async(
                            async move {
                                let result = session
                                    .execute_command(
                                        &command_to_execute,
                                        current_working_directory.as_deref(),
                                        None,
                                        Default::default(),
                                    )
                                    .await;

                                match result {
                                    Ok(output) => {
                                        let exit_code = output.exit_code().unwrap_or_else(|| {
                                            ExitCode::from(if output.success() { 0 } else { 1 })
                                        });
                                        ActionResult::CommandFinished {
                                            block_id: BlockId::new(),
                                            output: String::from_utf8_lossy(output.output())
                                                .into_owned(),
                                            exit_code,
                                        }
                                    }
                                    Err(error) => ActionResult::CommandFinished {
                                        block_id: BlockId::new(),
                                        output: format!(
                                            "Remote command execution failed: {error:#}"
                                        ),
                                        exit_code: ExitCode::from(1),
                                    },
                                }
                            },
                            move |result, _| action_result_for_requested_command(command, result),
                        );
                    }
                }
                // Warp: synchronous wait-style commands (wait_until_completion=true) unconditionally disable the pager.
                //
                // The model's self-reported `uses_pager` is unreliable: small models like deepseek-v4-flash almost never flag it,
                // and once an implicit pager such as `git diff`/`git log`/`man` is hit it stalls at the less prompt;
                // warp downgrades the command to a LongRunningCommandSnapshot and returns it, but the agent is unaware of this contract
                // switch and keeps issuing new tool calls in parallel, locking up both the PTY and the UI (input box disappears).
                //
                // Root-cause logic: since the agent explicitly says "wait until completion", a pager prompt violates that contract; warp
                // must guarantee the pager is never triggered, rather than relying on the model to predict each CLI's paging behavior.
                //
                // Does not affect the explicit async path (wait_until_completion=false); truly long-running commands such as tail -f / dev servers
                // still use the original LongRunningCommandSnapshot flow.
                let _ = uses_pager; // field kept for API compatibility, but semantics no longer depend on it
                let decorated_command = if *wait_until_completion {
                    self.turn_off_pager_for_command(command, ctx)
                } else {
                    command.clone()
                };
                ctx.emit(ShellCommandExecutorEvent::ExecuteCommand {
                    action_id: action_id.clone(),
                    command: decorated_command,
                });

                let block_selector = BlockSelector::RequestedCommandId(action_id.clone());
                let command = command.clone();
                drop(model);

                ActionExecution::new_async(
                    self.action_result_future(
                        block_selector.clone(),
                        action_result_delay_for_requested_command(*wait_until_completion),
                    ),
                    move |result, ctx| {
                        // Remove the senders from the maps.
                        if let Some(handle) = handle.upgrade(ctx) {
                            handle.update(ctx, |me, _| {
                                me.block_finished_senders.remove(&block_selector);
                                me.force_refresh_senders.remove(&block_selector);
                            });
                        }

                        action_result_for_requested_command(command, result)
                    },
                )
            }
            AIAgentActionType::WriteToLongRunningShellCommand {
                block_id,
                input,
                mode,
            } => {
                let Some(block) = model.block_list().block_with_id(block_id) else {
                    return ActionExecution::Sync(
                        AIAgentActionResultType::WriteToLongRunningShellCommand(
                            WriteToLongRunningShellCommandResult::Error(
                                ShellCommandError::BlockNotFound,
                            ),
                        ),
                    );
                };
                if block.finished() {
                    let output: String = agent_shell_command_block_output(block);
                    let exit_code = block.exit_code();
                    return ActionExecution::Sync(
                        AIAgentActionResultType::WriteToLongRunningShellCommand(
                            WriteToLongRunningShellCommandResult::CommandFinished {
                                block_id: block.id().clone(),
                                output,
                                exit_code,
                            },
                        ),
                    );
                }
                // Drop immutable borrow.
                drop(model);

                let mut model = self.terminal_model.lock();
                if let Some(block) = model.block_list_mut().mut_block_from_id(block_id) {
                    block.mark_agent_written_to_block();
                }
                drop(model);

                ctx.emit(ShellCommandExecutorEvent::WriteToPty {
                    input: input.clone(),
                    mode: *mode,
                });

                let block_selector = BlockSelector::Id(block_id.clone());
                ActionExecution::new_async(
                    self.action_result_future(
                        block_selector.clone(),
                        ActionResultDelay::Duration(Duration::from_millis(200)),
                    ),
                    move |result, ctx| {
                        // Remove the senders from the maps.
                        if let Some(handle) = handle.upgrade(ctx) {
                            handle.update(ctx, |me, _| {
                                me.block_finished_senders.remove(&block_selector);
                                me.force_refresh_senders.remove(&block_selector);
                            });
                        }

                        action_result_for_write_to_long_running_shell_command(result)
                    },
                )
            }
            AIAgentActionType::ReadShellCommandOutput { block_id, delay } => {
                let Some(block) = model.block_list().block_with_id(block_id) else {
                    return ActionExecution::Sync(AIAgentActionResultType::ReadShellCommandOutput(
                        ReadShellCommandOutputResult::Error(ShellCommandError::BlockNotFound),
                    ));
                };
                if block.finished() {
                    let command = block.command_with_secrets_unobfuscated(false);
                    let output: String = block.output_with_secrets_unobfuscated();
                    let exit_code = block.exit_code();
                    return ActionExecution::Sync(AIAgentActionResultType::ReadShellCommandOutput(
                        ReadShellCommandOutputResult::CommandFinished {
                            command,
                            block_id: block_id.clone(),
                            output,
                            exit_code,
                        },
                    ));
                }
                let command = block.command_with_secrets_unobfuscated(false);
                // Only on the `ReadShellCommandOutput` path, lower the wait time based on command content: this
                // is the agent's second poll of a block that is **still running**; by default it takes `OnCompletion`
                // and would wait the full `MAX_AGENT_DELAY_DURATION` (120s). For interactive sessions that never exit on their own, such as ssh / mosh / sftp /
                // telnet, such waiting is pointless.
                // `RequestCommandOutput` (first request) uses the default timeout of `MAX_WAIT_DURATION = 2s`,
                // which never stalls for 120s, so it needs no such handling.
                let delay = effective_read_shell_command_delay(&command, delay.clone());
                drop(model);

                let block_selector = BlockSelector::Id(block_id.clone());
                ActionExecution::new_async(
                    self.action_result_future(block_selector.clone(), delay),
                    move |result, ctx| {
                        // Remove the senders from the maps.
                        if let Some(handle) = handle.upgrade(ctx) {
                            handle.update(ctx, |me, _| {
                                me.block_finished_senders.remove(&block_selector);
                                me.force_refresh_senders.remove(&block_selector);
                            });
                        }

                        action_result_for_read_shell_command_output(command.clone(), result)
                    },
                )
            }
            AIAgentActionType::TransferShellCommandControlToUser { reason } => {
                let active_block = model.block_list().active_block();
                if !active_block.is_active_and_long_running() {
                    return ActionExecution::Sync(
                        AIAgentActionResultType::TransferShellCommandControlToUser(
                            TransferShellCommandControlToUserResult::Error(
                                ShellCommandError::BlockNotFound,
                            ),
                        ),
                    );
                }

                let block_id = active_block.id().clone();
                drop(model);

                // Emit event to transfer control to user.
                ctx.emit(ShellCommandExecutorEvent::TransferControlToUser {
                    action_id: action_id.clone(),
                    reason: reason.clone(),
                });

                // Create a channel to wait for control handback.
                let (handback_tx, handback_rx) = oneshot::channel();
                self.control_handback_sender = Some(handback_tx);

                let block_selector = BlockSelector::Id(block_id.clone());

                // Set up a future to also wait for block completion.
                let (block_finished_tx, block_finished_rx) = oneshot::channel();
                self.block_finished_senders
                    .insert(block_selector.clone(), block_finished_tx);

                // Build the future that captures terminal model and block data.
                let transfer_future = {
                    let terminal_model = self.terminal_model.clone();
                    let block_id = block_id.clone();
                    async move {
                        pin!(handback_rx);
                        pin!(block_finished_rx);

                        // Wait for either control handback or block completion.
                        let transfer_result = select! {
                            val = handback_rx => match val {
                                Ok(_) => TransferControlResult::ControlHandedBack,
                                Err(_) => TransferControlResult::Cancelled,
                            },
                            val = block_finished_rx => match val {
                                Ok(_) => TransferControlResult::BlockFinished,
                                Err(_) => TransferControlResult::Cancelled,
                            },
                        };

                        // Convert to ActionResult
                        let model = terminal_model.lock();
                        match transfer_result {
                            TransferControlResult::ControlHandedBack
                            | TransferControlResult::BlockFinished => {
                                match model.block_list().block_with_id(&block_id) {
                                    Some(block) => {
                                        if block.finished() {
                                            ActionResult::CommandFinished {
                                                block_id: block.id().clone(),
                                                output: agent_shell_command_block_output(block),
                                                exit_code: block.exit_code(),
                                            }
                                        } else {
                                            let grid_contents = if model.is_alt_screen_active() {
                                                formatted_terminal_contents_for_input(
                                                    model.alt_screen().grid_handler(),
                                                    None,
                                                    CURSOR_MARKER,
                                                )
                                            } else {
                                                formatted_terminal_contents_for_input(
                                                    block.output_grid().grid_handler(),
                                                    Some(1000),
                                                    CURSOR_MARKER,
                                                )
                                            };
                                            ActionResult::LongRunningCommandSnapshot {
                                                block_id: block.id().clone(),
                                                grid_contents,
                                                cursor: CURSOR_MARKER,
                                                is_alt_screen_active: model.is_alt_screen_active(),
                                                is_preempted: false,
                                            }
                                        }
                                    }
                                    None => ActionResult::BlockNotFound,
                                }
                            }
                            TransferControlResult::Cancelled => ActionResult::Cancelled,
                        }
                    }
                };

                ActionExecution::new_async(transfer_future, move |result, ctx| {
                    // Clean up.
                    if let Some(handle) = handle.upgrade(ctx) {
                        handle.update(ctx, |me, _| {
                            me.block_finished_senders.remove(&block_selector);
                            me.control_handback_sender = None;
                        });
                    }

                    action_result_for_transfer_shell_command_control_to_user(result)
                })
            }
            _ => ActionExecution::InvalidAction,
        }
    }

    /// Called when user hands control back to agent after TransferShellCommandControlToUser.
    pub fn notify_control_handed_back(&mut self) {
        if let Some(sender) = self.control_handback_sender.take() {
            let _ = sender.send(());
        }
    }

    /// Produces a future which resolves when the action is complete and
    /// we have a result to send to the agent.
    fn action_result_future(
        &mut self,
        block_selector: BlockSelector,
        delay: ActionResultDelay,
    ) -> impl Spawnable<Output = ActionResult> {
        // Create a channel to notify us when we receive block metadata.
        let (block_metadata_received_tx, block_metadata_received_rx) = oneshot::channel();
        self.block_finished_senders
            .insert(block_selector.clone(), block_metadata_received_tx);

        // Create a channel so the `Check now` affordance can short-circuit the timeout
        // and deliver the agent a fresh snapshot immediately.
        let (force_refresh_tx, force_refresh_rx) = oneshot::channel();
        self.force_refresh_senders
            .insert(block_selector.clone(), force_refresh_tx);

        // Create a future that resolves when we should send a result to the agent.
        let terminal_model = self.terminal_model.clone();

        async move {
            pin!(block_metadata_received_rx);
            pin!(force_refresh_rx);

            let timeout_duration = match delay {
                ActionResultDelay::UntilCompletion => None,
                ActionResultDelay::Duration(duration) => {
                    // Enforce a maximum allowed delay that the agent may request, never waiting longer than MAX_AGENT_DELAY_DURATION.
                    // If the requested duration exceeds this cap, we'll still behave as if the agent may expect a running command,
                    // so there's no need to signal preemption (the agent already anticipates an incomplete command state).
                    Some(duration.min(Self::MAX_AGENT_DELAY_DURATION))
                }
                ActionResultDelay::OnCompletion { timeout } => {
                    Some(timeout.min(Self::MAX_AGENT_DELAY_DURATION))
                }
                ActionResultDelay::Default => Some(Self::MAX_WAIT_DURATION),
            };

            let wake_reason = if let Some(timeout_duration) = timeout_duration {
                let timeout = Timer::after(timeout_duration).fuse();
                pin!(timeout);
                select! {
                    val = block_metadata_received_rx => match val {
                        Ok(_) => WakeReason::BlockFinished,
                        Err(_) => return ActionResult::Cancelled,
                    },
                    val = force_refresh_rx => match val {
                        // User asked the agent to check now; fall through to the snapshot
                        // code path below. Treated as a preemption (snapshot arrives before
                        // the agent's own timer would have fired).
                        Ok(_) => WakeReason::ForceRefresh,
                        // Sender was dropped (e.g. because the executor is being torn down).
                        Err(_) => return ActionResult::Cancelled,
                    },
                    _ = timeout => WakeReason::Timeout,
                }
            } else {
                // The ActionResultDelay::UntilCompletion path originally had no timeout. Add a hard `MAX_UNTIL_COMPLETION_DURATION`
                // fallback to prevent the agent from hanging forever after `turn_off_pager_for_command` is bypassed by user shell config
                // (see #138). A timeout takes the `(Timeout, UntilCompletion)` branch of `compute_is_preempted` below,
                // and is marked as preempted.
                let hard_timeout = Timer::after(Self::MAX_UNTIL_COMPLETION_DURATION).fuse();
                pin!(hard_timeout);
                select! {
                    val = block_metadata_received_rx => match val {
                        Ok(_) => WakeReason::BlockFinished,
                        Err(_) => return ActionResult::Cancelled,
                    },
                    val = force_refresh_rx => match val {
                        Ok(_) => WakeReason::ForceRefresh,
                        Err(_) => return ActionResult::Cancelled,
                    },
                    _ = hard_timeout => WakeReason::Timeout,
                }
            };

            // Mark the snapshot as preempted if woken early, allowing the server to distinguish
            // true completion from a forced client poll (`ForceRefresh`), a timeout during
            // `on_completion`, or the `UntilCompletion` pager-hang safety-net timeout.
            //
            // Note: `RequestCommandOutputResult::LongRunningCommandSnapshot` currently has no `is_preempted`
            // field (unlike `ReadShellCommandOutputResult` / `TransferShellCommandControlToUserResult`),
            // so on the `RequestCommandOutput` path this flag is dropped by the `..` in `action_result_for_requested_command`;
            // it is still assigned with the correct semantics here so it takes effect automatically once the field is added.
            let is_preempted = compute_is_preempted(wake_reason, delay);

            // At this point, we've either received block metadata or we've timed out.
            // Check the current state of the block and produce a result accordingly.
            let model = terminal_model.lock();
            let result = match block_selector.get_block(&model) {
                Some(block) => {
                    if block.finished() {
                        ActionResult::CommandFinished {
                            block_id: block.id().clone(),
                            output: agent_shell_command_block_output(block),
                            exit_code: block.exit_code(),
                        }
                    } else {
                        let grid_contents = if model.is_alt_screen_active() {
                            formatted_terminal_contents_for_input(
                                model.alt_screen().grid_handler(),
                                None,
                                CURSOR_MARKER,
                            )
                        } else {
                            formatted_terminal_contents_for_input(
                                block.output_grid().grid_handler(),
                                // TODO(vorporeal): This is probably too large.
                                Some(1000),
                                CURSOR_MARKER,
                            )
                        };
                        ActionResult::LongRunningCommandSnapshot {
                            block_id: block.id().clone(),
                            grid_contents,
                            cursor: CURSOR_MARKER,
                            is_alt_screen_active: model.is_alt_screen_active(),
                            is_preempted,
                        }
                    }
                }
                None => ActionResult::BlockNotFound,
            };

            result
        }
    }

    pub(super) fn cancel_execution(&mut self, id: &AIAgentActionId, _ctx: &mut ModelContext<Self>) {
        // The RequestedCommand path uses the action id as selector and cleans up unconditionally.
        // Cannot rely on the `is_active_and_long_running()` guard: in the ~50ms window after the command is spawned
        // (LONG_RUNNING_COMMAND_DURATION_MS) the guard is false, which would leave senders behind,
        // and keep the detached future alive until the command actually ends (especially bad for wait_until_completion=true,
        // i.e. ActionResultDelay::UntilCompletion).
        let requested_selector = BlockSelector::RequestedCommandId(id.clone());
        self.block_finished_senders.remove(&requested_selector);
        self.force_refresh_senders.remove(&requested_selector);

        // No longer use `BlockSelector::Id(active_block.id())` as a fallback cleanup. The sender keys of WriteToLRC /
        // ReadShellCommandOutput / TransferShellCommandControlToUser come from the block_id in the action
        // parameters or the active_block at creation time, which has no reliable correspondence to the active_block at cancel time:
        // if the user switches the active block after the action is spawned, the old active-block fallback would not match;
        // if not, the cleanup is only "occasionally correct". Their
        // senders are cleaned up by their own on_complete callbacks when the future ends naturally; immediate cleanup would need an
        // action_id -> BlockSelector reverse index, which is a separate change outside this issue.
    }

    /// Force any in-flight poll for the given long-running command block to resolve
    /// immediately with a fresh snapshot, bypassing the agent-set timeout.
    ///
    /// Called by the `Check now` affordance in the warping indicator. No-ops if there
    /// is no matching in-flight poll (e.g. because the block already finished or the
    /// agent has transferred control to the user).
    pub fn force_refresh_block(&mut self, block_id: &BlockId) {
        let terminal_model = self.terminal_model.lock();
        // Find a sender whose selector resolves to this block. In practice there is at
        // most one: a given block can have at most one in-flight `action_result_future`
        // at a time.
        let matching_selector = self
            .force_refresh_senders
            .keys()
            .find(|selector| {
                selector
                    .get_block(&terminal_model)
                    .is_some_and(|block| block.id() == block_id)
            })
            .cloned();
        drop(terminal_model);

        if let Some(selector) = matching_selector {
            if let Some(sender) = self.force_refresh_senders.remove(&selector) {
                let _ = sender.send(());
            }
        }
    }

    pub(super) fn preprocess_action(
        &mut self,
        _action: PreprocessActionInput,
        _ctx: &mut ModelContext<Self>,
    ) -> BoxFuture<'static, ()> {
        futures::future::ready(()).boxed()
    }
}

/// Pager-suppression prefix for bash/zsh. Export inside a subshell; the subshell exit code = the last command's exit code,
/// thereby preserving the real `$?`. First unset PAGER/GIT_PAGER/MANPAGER inherited from the parent shell, then export=cat.
const POSIX_NO_PAGER_PRELUDE: &str = "unset PAGER GIT_PAGER MANPAGER; export PAGER=cat GIT_PAGER=cat MANPAGER=cat GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.pager GIT_CONFIG_VALUE_0=cat";

/// fish: `set -lx` is a local export inside a begin/end block, and `$status` takes the last command.
/// Use `set -e` to clear inherited variables first, then `set -lx` to assign cat.
const FISH_NO_PAGER_PRELUDE: &str = "set -e PAGER; set -e GIT_PAGER; set -e MANPAGER; set -lx PAGER cat; set -lx GIT_PAGER cat; set -lx MANPAGER cat; set -lx GIT_CONFIG_COUNT 1; set -lx GIT_CONFIG_KEY_0 core.pager; set -lx GIT_CONFIG_VALUE_0 cat";

/// pwsh: a script block's local `$env:` does not pollute the outer session, and `$LASTEXITCODE` passes through.
/// `Remove-Item Env:` clears inherited values, then assign cat; use `-ErrorAction SilentlyContinue` for variables that do not exist.
const PWSH_NO_PAGER_PRELUDE: &str = "Remove-Item Env:PAGER -ErrorAction SilentlyContinue; Remove-Item Env:GIT_PAGER -ErrorAction SilentlyContinue; Remove-Item Env:MANPAGER -ErrorAction SilentlyContinue; $env:PAGER='cat'; $env:GIT_PAGER='cat'; $env:MANPAGER='cat'; $env:GIT_CONFIG_COUNT='1'; $env:GIT_CONFIG_KEY_0='core.pager'; $env:GIT_CONFIG_VALUE_0='cat'";

/// Wraps `command` in a "pager disabled" scope, see [`ShellCommandExecutor::turn_off_pager_for_command`].
///
/// **Invariant: the closing token of a multi-line command (`)` / `end` / `}`) must be on its own line.**
///
/// Counterexample: the agent sends `python3 - <<'PY' … PY`; naive concatenation gives `…\nPY)`. A heredoc terminator
/// must be alone on its line, `PY)` no longer matches, so the shell sits forever at the PS2 continuation prompt waiting for a line `PY`
/// — the command never ends → precmd never fires → `ActionResultDelay::UntilCompletion` hangs all the way to
/// `MAX_UNTIL_COMPLETION_DURATION` (30 minutes), during which the `is_active_and_long_running()` guard
/// also turns all subsequent agent commands into `CancelledBeforeExecution`, which looks like the whole agent being stuck.
/// A command ending with a trailing `#` comment is the same kind of hole (the closing token gets commented out).
///
/// Switch to the multi-line wrapping form only when the command **is already multi-line**: single-line commands keep their original single-line byte form,
/// to avoid being split into multiple blocks by the `\n` → `\r` replacement in `bytes_to_execute_command`
/// on shells without bracketed paste support (such as the bash 3.2 bundled with macOS). Multi-line commands are already split
/// on those shells, so this function does not make things worse.
fn wrap_command_without_pager(shell_type: Option<ShellType>, command: &str) -> String {
    // Trailing newlines/whitespace would put blank lines before the closing token; no semantic effect but pollutes the blocklist display.
    let command = command.trim_end();
    let is_multiline = command.contains('\n');

    match shell_type {
        Some(ShellType::Zsh) | Some(ShellType::Bash) => {
            let prelude = POSIX_NO_PAGER_PRELUDE;
            if is_multiline {
                format!("({prelude}\n{command}\n)")
            } else {
                format!("({prelude}; {command})")
            }
        }
        Some(ShellType::Fish) => {
            let prelude = FISH_NO_PAGER_PRELUDE;
            if is_multiline {
                format!("begin; {prelude}\n{command}\nend")
            } else {
                format!("begin; {prelude}; {command}; end")
            }
        }
        Some(ShellType::PowerShell) => {
            let prelude = PWSH_NO_PAGER_PRELUDE;
            if is_multiline {
                format!("& {{ {prelude}\n{command}\n}}")
            } else {
                format!("& {{ {prelude}; {command} }}")
            }
        }
        // An unknown shell cannot be decorated safely, so let it through. On this path pager suppression is completely ineffective, and we can only
        // rely on the MAX_UNTIL_COMPLETION_DURATION fallback timeout to avoid hanging forever.
        None => command.to_owned(),
    }
}

/// Wait strategy used internally by `action_result_future`.
///
/// Compared with the external `Option<ShellCommandDelay>`, this promotes the `OnCompletion` timeout
/// from an implicit constant to an explicit field, so it can be adjusted dynamically per command scenario (see
/// `effective_read_shell_command_delay`)。
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ActionResultDelay {
    UntilCompletion,
    Default,
    Duration(Duration),
    OnCompletion { timeout: Duration },
}

impl ActionResultDelay {
    fn from_shell_command_delay(delay: Option<ShellCommandDelay>) -> Self {
        match delay {
            Some(ShellCommandDelay::Duration(duration)) => Self::Duration(duration),
            Some(ShellCommandDelay::OnCompletion) => Self::OnCompletion {
                timeout: ShellCommandExecutor::MAX_AGENT_DELAY_DURATION,
            },
            None => Self::Default,
        }
    }
}

/// The reason in `action_result_future` that decides the value of `is_preempted`. Hoisted to module scope
/// so that `compute_is_preempted` can be called by same-module unit tests.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WakeReason {
    BlockFinished,
    Timeout,
    /// User clicked `Check now` in the warping indicator, short-circuiting  
    /// the agent-set poll timer. Treated as a preemption so the server does  
    /// not interpret the early snapshot as a completion.  
    ForceRefresh,
}

/// Computes whether the snapshot should be marked as preempted (`is_preempted=true`). Extracted as a pure function so that
/// unit tests can verify the table-driven logic (avoiding the need to mock the clock inside an async `select!`).
///
/// Preemption semantics: the server treats the snapshot as "an early peek", not "command completed". Any of the following:
/// - `ForceRefresh` (manually triggered by the user via Check now)
/// - `Timeout` with delay `OnCompletion` (exceeded the on-completion timeout set by the agent)
/// - `Timeout` with delay `UntilCompletion` (hit the pager-hang fallback timeout, see #138)
fn compute_is_preempted(wake: WakeReason, delay: ActionResultDelay) -> bool {
    matches!(wake, WakeReason::ForceRefresh)
        || matches!(
            (wake, delay),
            (WakeReason::Timeout, ActionResultDelay::OnCompletion { .. })
                | (WakeReason::Timeout, ActionResultDelay::UntilCompletion)
        )
}

fn action_result_delay_for_requested_command(wait_until_completion: bool) -> ActionResultDelay {
    if wait_until_completion {
        ActionResultDelay::UntilCompletion
    } else {
        ActionResultDelay::Default
    }
}

/// Maps the `ShellCommandDelay` requested by the agent to the internal `ActionResultDelay`,
/// with special handling for interactive sessions that **never exit on their own** (ssh / mosh / sftp / telnet, etc.):
///
/// 1. `Some(OnCompletion)`: shorten the timeout from `MAX_AGENT_DELAY_DURATION` (120s)
///    to `MAX_WAIT_DURATION` (2s), so the agent does not wait forever on a command that never ends.
/// 2. `None` (default): proactively **upgrade** to `OnCompletion { 2s }` instead of keeping
///    `Default`. Note this also changes the value of `is_preempted` in `action_result_future`:
///    `Default` + `Timeout` is not preemption, while `OnCompletion` + `Timeout`
///    is marked as preempted, so the server interprets this snapshot as "take a look first" rather than "command completed".
///    For interactive sessions this is the correct semantics.
/// 3. `Some(Duration(d))`: keep the agent's explicit request, no rewriting.
///
/// Non-interactive commands always use the original mapping of `from_shell_command_delay`.
fn effective_read_shell_command_delay(
    command: &str,
    delay: Option<ShellCommandDelay>,
) -> ActionResultDelay {
    if command_starts_non_terminating_session(command)
        && matches!(delay, None | Some(ShellCommandDelay::OnCompletion))
    {
        return ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_WAIT_DURATION,
        };
    }

    ActionResultDelay::from_shell_command_delay(delay)
}

/// Determines whether `command` will start an interactive session that **never exits on its own**. Matching rules:
/// - Commands wrapped by the Warp generator wrapper: recurse into the inner command.
/// - Bare `ssh ...` (goes through `parse_interactive_ssh_command`, which correctly excludes `-T` / `-W`
///   and other non-interactive forms).
/// - ssh with a path or `.exe` (rewritten to bare `ssh` and then checked).
/// - `mosh` / `sftp` / `telnet` (including `.exe`) have no non-interactive flag like ssh, so match directly by
///   executable name.
///
/// A heuristic used only by `effective_read_shell_command_delay`; the consequences of a false positive are limited.
fn command_starts_non_terminating_session(command: &str) -> bool {
    let command = command.trim_start();
    in_band_generator_command(command)
        .as_deref()
        .is_some_and(command_starts_non_terminating_session)
        || parse_interactive_ssh_command(command).is_some()
        || normalized_ssh_command(command)
            .as_deref()
            .is_some_and(|command| parse_interactive_ssh_command(command).is_some())
        || first_executable_name(command).is_some_and(|name| {
            matches!(
                name.as_str(),
                "mosh" | "mosh.exe" | "sftp" | "sftp.exe" | "telnet" | "telnet.exe"
            )
        })
}

/// Unwraps Warp's own generator wrapper and extracts the command that actually runs inside.
///
/// The wrapper protocol looks like: `<wrapper> <generator_id> '<inner_command>' [extra flags...]`
/// where:
/// - `<wrapper>` is `warp_run_generator_command` (POSIX shell) or
///   `Warp-Run-GeneratorCommand` (PowerShell, case-insensitive).
/// - `<generator_id>` is a numeric id, not parsed here, just skipped.
/// - `<inner_command>` is the real command string wrapped in single quotes, which is what we return.
///
/// The protocol takes `tokens[2]` by fixed position; if a later wrapper adds optional arguments that break the positional assumption,
/// this silently mismatches (returns None); the worst case is falling back to the old 120s wait, with no incorrect behavior introduced.
fn in_band_generator_command(command: &str) -> Option<String> {
    let tokens = shell_words::split(command.trim_start()).ok()?;
    if tokens.len() >= 3
        && (tokens[0].eq_ignore_ascii_case("Warp-Run-GeneratorCommand")
            || tokens[0] == "warp_run_generator_command")
    {
        Some(tokens[2].clone())
    } else {
        None
    }
}

/// When the command's executable entry is ssh with a path or a `.exe` suffix, rewrite it to bare `ssh`,
/// so the `parse_interactive_ssh_command` parser, which only accepts `^ssh\s+...`, can be reused.
///
/// For example `"C:\Windows\System32\OpenSSH\ssh.exe" host -p 22` is rewritten to
/// `ssh host -p 22`. The remaining arguments are kept as is (`rest` is the string left after `first_executable_token` cut
/// off the first token).
///
/// By name it only does a "prefix rewrite": it does not normalize backslashes/quotes in the path, nor expand escapes.
fn normalized_ssh_command(command: &str) -> Option<String> {
    let (token, rest) = first_executable_token(command)?;
    let name = command_basename(token);
    if name.eq_ignore_ascii_case("ssh") || name.eq_ignore_ascii_case("ssh.exe") {
        Some(format!("ssh{rest}"))
    } else {
        None
    }
}

fn first_executable_name(command: &str) -> Option<String> {
    let (token, _) = first_executable_token(command)?;
    Some(command_basename(token).to_ascii_lowercase())
}

/// Returns the real "executable entry" token of the command, skipping common invocation prefixes:
/// - The PowerShell call operator `&` (must be a standalone token, e.g. `& "C:\...\ssh.exe" host`).
/// - The POSIX `command` builtin (`command ssh host`), used to bypass aliases/functions.
///
/// Only one layer of prefix is stripped, enough for practical cases; `&&` chains, `call`, `exec` and other forms are not supported.
fn first_executable_token(command: &str) -> Option<(&str, &str)> {
    let (token, rest) = first_command_token(command)?;
    if token == "&" || token.eq_ignore_ascii_case("command") {
        first_command_token(rest)
    } else {
        Some((token, rest))
    }
}

/// Heuristic tokenization: take the first token of the command and the raw string remaining after it.
///
/// Deliberately **not** using `shell_words::split`, for two reasons:
/// 1. `shell_words` fails outright on non-POSIX characters such as the PowerShell call operator `&`,
///    and we need to recognize such forms.
/// 2. We only need the two parts "first token + rest", not the full token stream, so hand-writing is more direct.
///
/// Quote handling only recognizes a `"` or `'` at the string **start**, and does not handle escapes. To avoid
/// inputs like `"foo"bar` or `"ssh"hello-world` (characters stuck after the closing quote) being wrongly cut into
/// `foo` / `ssh` and triggering a **false detection** (recognizing an ordinary command as an interactive session), a closing quote
/// must be followed by whitespace or end of string; otherwise return `None`, letting the caller take the safe branch of "fall back to the old wait
/// behavior" rather than risk a false positive by cutting anyway.
fn first_command_token(command: &str) -> Option<(&str, &str)> {
    let command = command.trim_start();
    if command.is_empty() {
        return None;
    }

    let mut chars = command.char_indices();
    let (_, first) = chars.next()?;
    if first == '"' || first == '\'' {
        for (idx, ch) in chars {
            if ch == first {
                let token = &command[first.len_utf8()..idx];
                let rest = &command[idx + ch.len_utf8()..];
                // The closing quote must be followed by whitespace or end of string; otherwise treat it as untokenizable,
                // letting the caller fall back to the non-preempting safe path.
                if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
                    return None;
                }
                return Some((token, rest));
            }
        }

        // No matching closing quote found: likewise treated as untokenizable.
        return None;
    }

    let end = command.find(char::is_whitespace).unwrap_or(command.len());
    Some((&command[..end], &command[end..]))
}

fn command_basename(command_token: &str) -> &str {
    command_token
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(command_token)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum BlockSelector {
    Id(BlockId),
    RequestedCommandId(AIAgentActionId),
}

impl BlockSelector {
    fn get_block<'a>(&self, model: &'a TerminalModel) -> Option<&'a Block> {
        match self {
            BlockSelector::Id(block_id) => model.block_list().block_with_id(block_id),
            BlockSelector::RequestedCommandId(requested_command_id) => model
                .block_list()
                .block_for_ai_action_id(requested_command_id),
        }
    }
}

/// Returns the result from executing a requested command.
fn action_result_for_requested_command(
    command: String,
    result: ActionResult,
) -> AIAgentActionResultType {
    match result {
        ActionResult::CommandFinished {
            block_id,
            output,
            exit_code,
        } => AIAgentActionResultType::RequestCommandOutput(RequestCommandOutputResult::Completed {
            command,
            block_id,
            output,
            exit_code,
        }),
        ActionResult::LongRunningCommandSnapshot {
            block_id,
            grid_contents,
            cursor,
            is_alt_screen_active,
            ..
        } => AIAgentActionResultType::RequestCommandOutput(
            RequestCommandOutputResult::LongRunningCommandSnapshot {
                command,
                block_id,
                grid_contents,
                cursor: cursor.to_owned(),
                is_alt_screen_active,
            },
        ),
        ActionResult::BlockNotFound | ActionResult::Cancelled => {
            AIAgentActionResultType::RequestCommandOutput(
                RequestCommandOutputResult::CancelledBeforeExecution,
            )
        }
    }
}

/// Returns the result from writing to a long-running shell command.
fn action_result_for_write_to_long_running_shell_command(
    result: ActionResult,
) -> AIAgentActionResultType {
    match result {
        ActionResult::CommandFinished {
            block_id,
            output,
            exit_code,
        } => AIAgentActionResultType::WriteToLongRunningShellCommand(
            WriteToLongRunningShellCommandResult::CommandFinished {
                block_id,
                output,
                exit_code,
            },
        ),
        ActionResult::LongRunningCommandSnapshot {
            block_id,
            grid_contents,
            cursor,
            is_alt_screen_active,
            is_preempted,
        } => AIAgentActionResultType::WriteToLongRunningShellCommand(
            WriteToLongRunningShellCommandResult::Snapshot {
                block_id,
                grid_contents,
                cursor: cursor.to_owned(),
                is_alt_screen_active,
                is_preempted,
            },
        ),
        ActionResult::Cancelled => AIAgentActionResultType::WriteToLongRunningShellCommand(
            WriteToLongRunningShellCommandResult::Cancelled,
        ),
        ActionResult::BlockNotFound => AIAgentActionResultType::WriteToLongRunningShellCommand(
            WriteToLongRunningShellCommandResult::Error(ShellCommandError::BlockNotFound),
        ),
    }
}

/// Returns the result from reading shell command output.
fn action_result_for_read_shell_command_output(
    command: String,
    result: ActionResult,
) -> AIAgentActionResultType {
    match result {
        ActionResult::CommandFinished {
            output,
            exit_code,
            block_id,
        } => AIAgentActionResultType::ReadShellCommandOutput(
            ReadShellCommandOutputResult::CommandFinished {
                command,
                block_id,
                output,
                exit_code,
            },
        ),
        ActionResult::LongRunningCommandSnapshot {
            block_id,
            grid_contents,
            cursor,
            is_alt_screen_active,
            is_preempted,
        } => AIAgentActionResultType::ReadShellCommandOutput(
            ReadShellCommandOutputResult::LongRunningCommandSnapshot {
                command,
                block_id,
                grid_contents,
                cursor: cursor.to_owned(),
                is_alt_screen_active,
                is_preempted,
            },
        ),
        ActionResult::Cancelled => {
            AIAgentActionResultType::ReadShellCommandOutput(ReadShellCommandOutputResult::Cancelled)
        }
        ActionResult::BlockNotFound => AIAgentActionResultType::ReadShellCommandOutput(
            ReadShellCommandOutputResult::Error(ShellCommandError::BlockNotFound),
        ),
    }
}

/// Returns the result from transferring shell command control to user.
fn action_result_for_transfer_shell_command_control_to_user(
    result: ActionResult,
) -> AIAgentActionResultType {
    match result {
        ActionResult::CommandFinished {
            block_id,
            output,
            exit_code,
        } => AIAgentActionResultType::TransferShellCommandControlToUser(
            TransferShellCommandControlToUserResult::CommandFinished {
                block_id,
                output,
                exit_code,
            },
        ),
        ActionResult::LongRunningCommandSnapshot {
            block_id,
            grid_contents,
            cursor,
            is_alt_screen_active,
            is_preempted,
        } => AIAgentActionResultType::TransferShellCommandControlToUser(
            TransferShellCommandControlToUserResult::Snapshot {
                block_id,
                grid_contents,
                cursor: cursor.to_owned(),
                is_alt_screen_active,
                is_preempted,
            },
        ),
        ActionResult::Cancelled => AIAgentActionResultType::TransferShellCommandControlToUser(
            TransferShellCommandControlToUserResult::Cancelled,
        ),
        ActionResult::BlockNotFound => AIAgentActionResultType::TransferShellCommandControlToUser(
            TransferShellCommandControlToUserResult::Error(ShellCommandError::BlockNotFound),
        ),
    }
}

#[derive(Debug, Clone)]
pub enum ShellCommandExecutorEvent {
    ExecuteCommand {
        action_id: AIAgentActionId,
        command: String,
    },
    WriteToPty {
        input: Bytes,
        mode: AIAgentPtyWriteMode,
    },
    CancelExecution,
    /// Emitted when the agent requests to transfer control of a long-running command to the user.
    TransferControlToUser {
        action_id: AIAgentActionId,
        reason: String,
    },
}

impl Entity for ShellCommandExecutor {
    type Event = ShellCommandExecutorEvent;
}

/// Result from waiting for control transfer.
#[derive(Debug, Clone)]
enum TransferControlResult {
    ControlHandedBack,
    BlockFinished,
    Cancelled,
}

/// The possible results of taking an action.
#[derive(Debug, Clone)]
enum ActionResult {
    CommandFinished {
        block_id: BlockId,
        output: String,
        exit_code: ExitCode,
    },
    LongRunningCommandSnapshot {
        block_id: BlockId,
        grid_contents: String,
        cursor: &'static str,
        is_alt_screen_active: bool,
        is_preempted: bool,
    },
    Cancelled,
    BlockNotFound,
}

#[cfg(test)]
#[path = "shell_command_tests.rs"]
mod tests;
