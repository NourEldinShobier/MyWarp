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
    a.push(if req.resume { "--resume" } else { "--session-id" }.into());
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
            vec![AgentEvent::Error(v["result"].as_str().unwrap_or("Claude Code failed").into())]
        }
        Some("result") => vec![AgentEvent::Done],
        _ => vec![],
    }
}

fn text(s: Option<&str>, make: fn(String) -> AgentEvent) -> Vec<AgentEvent> {
    s.filter(|s| !s.is_empty()).map(|s| vec![make(s.into())]).unwrap_or_default()
}

fn blocks(v: &Value) -> impl Iterator<Item = &Value> {
    v["message"]["content"].as_array().into_iter().flatten()
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
