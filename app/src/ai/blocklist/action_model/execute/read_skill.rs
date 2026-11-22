use super::{ActionExecution, AnyActionExecution, ExecuteActionInput, PreprocessActionInput};
#[cfg(feature = "local_fs")]
use crate::ai::agent::AIAgentActionResultType;
use crate::ai::skills::{SkillManager, SkillTelemetryEvent};
#[cfg(feature = "local_fs")]
use crate::ai::skills::extract_skill_parent_directory;
use crate::send_telemetry_from_ctx;
use ai::agent::action_result::AnyFileContent;
use ai::skills::SkillReference;
#[cfg(feature = "local_fs")]
use ai::skills::parse_skill;
use std::path::Path;
use warpui::{ModelContext, SingletonEntity};

use crate::ai::agent::AIAgentActionType;
use crate::ai::agent::ReadSkillRequest;
use crate::ai::agent::ReadSkillResult;
use ai::agent::action_result::FileContext;
use futures::future::{BoxFuture, FutureExt};
use warpui::Entity;

pub struct ReadSkillExecutor;

impl ReadSkillExecutor {
    pub fn new() -> Self {
        Self
    }

    pub(super) fn should_autoexecute(
        &self,
        _input: ExecuteActionInput,
        _ctx: &mut ModelContext<Self>,
    ) -> bool {
        // User-created skills are readable on demand.
        true
    }

    pub(super) fn execute(
        &mut self,
        input: ExecuteActionInput,
        ctx: &mut ModelContext<Self>,
    ) -> impl Into<AnyActionExecution> {
        let ExecuteActionInput { action, .. } = input;
        let AIAgentActionType::ReadSkill(ReadSkillRequest { skill: skill_ref }) = &action.action
        else {
            return ActionExecution::InvalidAction;
        };

        let manager = SkillManager::as_ref(ctx);

        // Cache hit: the proto `SkillReference::Path(p)` only hits at this step when p is exactly
        // the real SKILL.md absolute path in the index.
        if let Some(skill) = manager.skill_by_reference(skill_ref) {
            send_telemetry_from_ctx!(
                SkillTelemetryEvent::Read {
                    reference: skill_ref.clone(),
                    name: Some(skill.name.clone()),
                    scope: Some(skill.scope),
                    provider: Some(skill.provider),
                    error: false,
                },
                ctx
            );
            return success_execution(skill);
        }

        // The argument of the BYOP `read_skill` tool is the skill **name**, packed by `from_args` into
        // the `SkillReference::SkillPath(name)` slot (to avoid a proto schema change).
        // On a cache miss, look up the real SKILL.md path by name here, covering all skills
        // visible to the Skill manager (file skills + bundled skills).
        if let SkillReference::Path(p) = skill_ref {
            if let Some(candidate_name) = name_candidate(p) {
                if let Some(skill) = manager.find_skill_by_name(candidate_name) {
                    send_telemetry_from_ctx!(
                        SkillTelemetryEvent::Read {
                            reference: skill_ref.clone(),
                            name: Some(skill.name.clone()),
                            scope: Some(skill.scope),
                            provider: Some(skill.provider),
                            error: false,
                        },
                        ctx
                    );
                    return success_execution(skill);
                }
            }
        }

        // Cache-miss fallback: for a reference of the `SkillReference::Path` form,
        // if the path shape is a valid skill file
        // (`.../<provider>/skills/<name>/SKILL.md` or under a warp managed skill directory),
        // read and parse it from disk directly, fixing the "skill exists but cache is not warm" scenario described in issue #99.
        //
        // Design trade-offs:
        // - Do not proactively warm the SkillManager cache. The cache is maintained one-way by SkillWatcher,
        //   and writing to it here would break the data flow. Repeated read_skill calls on the same path will re-read the disk,
        //   but SKILL.md is usually small, so this is negligible.
        // - `extract_skill_parent_directory` only validates the path shape, at the same safety level as the path
        //   returned on a cache hit - neither restricts a home directory prefix. This is intentional:
        //   skills inside a project (`/some/repo/.agents/skills/...`) must also be readable.
        // - On Windows the regex uses backslash separators, so Linux-style `/home/<u>/...` paths are
        //   rejected; this means the fallback does not work for "Windows main process + WSL session",
        //   a known limitation of issue #99 (see the PR description).
        // The cache-miss fallback is only available in builds with a local file system;
        // in no-fs builds such as WASM, `extract_skill_parent_directory` / `parse_skill`
        // do not exist, so there is naturally nothing to read from disk.
        #[cfg(feature = "local_fs")]
        if let SkillReference::Path(path) = skill_ref {
            if extract_skill_parent_directory(path).is_ok() {
                let path = path.clone();
                let skill_ref_for_async = skill_ref.clone();
                return ActionExecution::new_async(
                    async move { parse_skill(&path) },
                    move |parsed, _app| match parsed {
                        Ok(skill) => AIAgentActionResultType::ReadSkill(
                            ReadSkillResult::Success {
                                content: FileContext::new(
                                    skill.path.to_string_lossy().into_owned(),
                                    AnyFileContent::StringContent(skill.content.clone()),
                                    skill.line_range.clone(),
                                    None,
                                ),
                            },
                        ),
                        Err(err) => AIAgentActionResultType::ReadSkill(
                            ReadSkillResult::Error(format!(
                                "Skill not found: {skill_ref_for_async:?} ({err})"
                            )),
                        ),
                    },
                );
            }
        }

        send_telemetry_from_ctx!(
            SkillTelemetryEvent::Read {
                reference: skill_ref.clone(),
                name: None,
                scope: None,
                provider: None,
                error: true,
            },
            ctx
        );
        ActionExecution::Sync(
            ReadSkillResult::Error(format!("Skill not found: {:?}", skill_ref)).into(),
        )
    }

    pub(super) fn preprocess_action(
        &mut self,
        _input: PreprocessActionInput,
        _ctx: &mut ModelContext<Self>,
    ) -> BoxFuture<'static, ()> {
        futures::future::ready(()).boxed()
    }
}

/// Build a sync success execution from a parsed skill.
///
/// This helper is extracted so that the generic `T` of `ActionExecution<T>` infers to the same type in the
/// `success_execution` and `new_async` paths (otherwise Rust would require the function to declare its return type explicitly).
fn success_execution(skill: &ai::skills::ParsedSkill) -> ActionExecution<anyhow::Result<ai::skills::ParsedSkill>> {
    let content = FileContext::new(
        skill.path.to_string_lossy().into_owned(),
        AnyFileContent::StringContent(skill.content.clone()),
        skill.line_range.clone(),
        None,
    );
    ActionExecution::Sync(ReadSkillResult::Success { content }.into())
}

/// Decides whether the value in `SkillReference::Path` should be treated as a skill **name** to look up.
///
/// A real SKILL.md path contains a path separator (`/` or `\`) or is an absolute path, while the name
/// in a BYOP tool call (e.g. `"build-feature"`) is a plain string. Telling the two apart
/// avoids misreading `/home/.../SKILL.md` as a name and missing the file-system fallback.
fn name_candidate(p: &Path) -> Option<&str> {
    if p.is_absolute() {
        return None;
    }
    let s = p.to_str()?;
    if s.is_empty() || s.contains('/') || s.contains('\\') {
        return None;
    }
    Some(s)
}

impl Entity for ReadSkillExecutor {
    type Event = ();
}

#[cfg(test)]
#[path = "read_skill_tests.rs"]
mod tests;
