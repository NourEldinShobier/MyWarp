//! `read_skill`: reads Warp's Skill markdown templates.
//!
//! A Skill is a reusable workflow predefined by the user/project (a `SKILL.md` file + optional metadata).
//! After reading a skill the model can advance the task in the steps the user expects. warp maintains its own `SkillManager`
//! that indexes all available skills; a skill can be referenced by name (frontmatter `name` field), absolute path, or
//! bundled id.
//!
//! ## Argument contract
//!
//! The BYOP path exposes a `name` field, whose value comes from the system prompt `<available_skills><skill><name>`.
//! `from_args` puts the name into the proto `SkillReference::SkillPath` slot (proto unchanged),
//! and the `read_skill` executor, on a cache miss, first resolves the name to the real SKILL.md absolute path
//! and then reads from disk. This fallback also accepts the older forms where the model passes an absolute path or the bundled form
//! `@warp-skill:<id>` directly.
//!
//! ## Usage hints (written into the description)
//!
//! The model may proactively call it in these cases:
//! - The user mentions a skill name / file name / path
//! - The task matches a skill description (e.g. "do a PR review" triggers the `review` skill)

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use warp_multi_agent_api as api;

use super::OpenAiTool;

#[derive(Debug, Deserialize)]
struct Args {
    name: String,
}

fn parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (exactly as it appears in the <available_skills><skill><name> field of the system prompt)."
            }
        },
        "required": ["name"],
        "additionalProperties": false
    })
}

fn from_args(args: &str) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call::read_skill::SkillReference;
    let parsed: Args = serde_json::from_str(args)?;
    // Reuse the proto `SkillPath` slot to carry the name (avoids a proto schema change);
    // the executor resolves the real SKILL.md path by name on a cache miss.
    Ok(api::message::tool_call::Tool::ReadSkill(
        api::message::tool_call::ReadSkill {
            skill_reference: Some(SkillReference::SkillPath(parsed.name)),
            name: String::new(),
        },
    ))
}

fn result_to_json(result: &api::message::tool_call_result::Result) -> Option<Value> {
    use api::message::tool_call_result::Result as R;
    use api::read_skill_result::Result as SR;
    let r = match result {
        R::ReadSkill(r) => r,
        _ => return None,
    };
    let value = match &r.result {
        Some(SR::Success(s)) => {
            // FileContent { file_path, content, line_range } is a single message directly,
            // not a oneof, so there is no inner content to unwrap.
            let (path, content) = s
                .content
                .as_ref()
                .map(|c| (c.file_path.clone(), c.content.clone()))
                .unwrap_or_default();
            json!({ "status": "ok", "path": path, "content": content })
        }
        Some(SR::Error(e)) => json!({ "status": "error", "message": e.message }),
        None => json!({ "status": "cancelled" }),
    };
    Some(value)
}

pub static READ_SKILL: OpenAiTool = OpenAiTool {
    name: "read_skill",
    description: include_str!("../prompts/tool_descriptions/read_skill.md"),
    parameters,
    from_args,
    result_to_json,
};
