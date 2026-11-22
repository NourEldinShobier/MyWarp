//! Claude Code via `claude -p --input-format stream-json --output-format stream-json`.

use futures::stream::BoxStream;
use serde_json::Value;

use super::{runner, AgentBackend, AgentEvent, TurnRequest};

pub(crate) struct ClaudeCode;

impl AgentBackend for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude"
    }
    fn display_name(&self) -> &'static str {
        "Claude Code"
    }
    fn binary(&self) -> &'static str {
        "claude"
    }
    fn models(&self) -> &'static [&'static str] {
        &["default", "opus", "sonnet", "haiku"]
    }

    fn run_turn(&self, req: TurnRequest) -> BoxStream<'static, AgentEvent> {
        let stdin = serde_json::json!({
            "type": "user",
            "message": { "role": "user", "content": req.prompt },
        })
        .to_string();
        runner::run_lines(self.binary(), args(&req), req.cwd.as_deref(), stdin, parse_line)
    }
}

fn args(req: &TurnRequest) -> Vec<String> {
    let mut a: Vec<String> = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        // Headless runs cannot answer permission prompts; edits are allowed, other
        // tools the user has not pre-approved are denied by the CLI.
        "--permission-mode",
        "acceptEdits",
    ]
    .map(String::from)
    .into();
    // The CLI keys saved sessions by working directory, so a conversation reopened from
    // elsewhere cannot be resumed; start it fresh under the same id instead of failing.
    let resume = req.resume && session_exists(req.cwd.as_deref(), &req.session_id);
    a.extend(["--append-system-prompt".into(), super::style::GUIDANCE.into()]);
    a.push(if resume { "--resume" } else { "--session-id" }.into());
    a.push(req.session_id.clone());
    if let Some(model) = &req.model {
        a.extend(["--model".into(), model.clone()]);
    }
    a
}

/// Maps one stdout line of the stream-json protocol to neutral events.
fn parse_line(line: &str) -> Vec<AgentEvent> {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return vec![];
    };
    match v["type"].as_str() {
        Some("stream_event") => {
            let delta = &v["event"]["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => text(delta["text"].as_str(), AgentEvent::Text),
                Some("thinking_delta") => text(delta["thinking"].as_str(), AgentEvent::Reasoning),
                _ => vec![],
            }
        }
        // Complete assistant message: its text was already streamed, only tools are new.
        Some("assistant") => blocks(&v)
            .filter(|b| b["type"] == "tool_use")
            .map(|b| AgentEvent::ToolStarted {
                id: b["id"].as_str().unwrap_or_default().into(),
                name: b["name"].as_str().unwrap_or("tool").into(),
                input: b["input"].clone(),
            })
            .collect(),
        Some("user") => blocks(&v)
            .filter(|b| b["type"] == "tool_result")
            .map(|b| AgentEvent::ToolFinished {
                id: b["tool_use_id"].as_str().unwrap_or_default().into(),
                is_error: b["is_error"].as_bool().unwrap_or(false),
                output: result_text(&b["content"]),
            })
            .collect(),
        Some("result") if v["is_error"].as_bool().unwrap_or(false) => {
            vec![AgentEvent::Error(failure_text(&v))]
        }
        Some("result") => {
            let mut ev: Vec<AgentEvent> = usage_line(&v).into_iter().map(AgentEvent::Usage).collect();
            ev.push(AgentEvent::Done);
            ev
        }
        _ => vec![],
    }
}

fn text(s: Option<&str>, make: fn(String) -> AgentEvent) -> Vec<AgentEvent> {
    s.filter(|s| !s.is_empty()).map(|s| vec![make(s.into())]).unwrap_or_default()
}

fn blocks(v: &Value) -> impl Iterator<Item = &Value> {
    v["message"]["content"].as_array().into_iter().flatten()
}

/// Whether the CLI has a saved session `id` for `cwd` (`~/.claude/projects/<cwd>/<id>.jsonl`).
fn session_exists(cwd: Option<&std::path::Path>, id: &str) -> bool {
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return true; // cannot tell; let the CLI decide
    };
    let cwd = cwd.map(Into::into).or_else(|| std::env::current_dir().ok()).unwrap_or_default();
    std::path::Path::new(&home)
        .join(".claude/projects")
        .join(project_dir_name(&cwd.to_string_lossy()))
        .join(format!("{id}.jsonl"))
        .is_file()
}

/// The CLI's project folder name: every non-alphanumeric character becomes `-`.
fn project_dir_name(cwd: &str) -> String {
    cwd.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect()
}

/// `12.4s · 1.1k tokens out · $0.86` from the final `result` event; parts that are missing are skipped.
fn usage_line(v: &Value) -> Option<String> {
    let mut parts = vec![];
    if let Some(ms) = v["duration_ms"].as_f64() {
        parts.push(format!("{:.1}s", ms / 1000.0));
    }
    if let Some(t) = v["usage"]["output_tokens"].as_u64() {
        parts.push(if t >= 1000 { format!("{:.1}k tokens out", t as f64 / 1000.0) } else { format!("{t} tokens out") });
    }
    if let Some(c) = v["total_cost_usd"].as_f64() {
        parts.push(format!("${c:.2}"));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// Why a turn failed: the CLI's message when it has one, else its subtype and error list.
fn failure_text(v: &Value) -> String {
    if let Some(s) = v["result"].as_str().filter(|s| !s.is_empty()) {
        return s.into();
    }
    let errors = v["errors"].as_array().into_iter().flatten().filter_map(Value::as_str);
    let detail = errors.collect::<Vec<_>>().join("; ");
    let subtype = v["subtype"].as_str().unwrap_or("error");
    if detail.is_empty() { format!("Claude Code failed ({subtype})") } else { format!("Claude Code failed ({subtype}): {detail}") }
}

/// A tool result's `content` is either a string or a list of `{type:"text", text}` blocks.
fn result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            items.iter().filter_map(|i| i["text"].as_str()).collect::<Vec<_>>().join("
")
        }
        _ => String::new(),
    }
}

#[cfg(test)]
#[path = "claude_tests.rs"]
mod tests;
