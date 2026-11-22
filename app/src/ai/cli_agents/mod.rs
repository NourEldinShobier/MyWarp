//! Drive external coding-agent CLIs (Claude Code, and later Codex / ACP agents) as chat
//! backends, using the user's own installed binary and login.
//!
//! Each backend owns its transport and turns the CLI's output into the neutral
//! [`AgentEvent`] stream; `stream` converts that stream into Warp's `ResponseEvent`s. A new
//! agent is one `AgentBackend` impl plus an entry in [`backends`].

use std::path::PathBuf;

use futures::stream::BoxStream;

pub(crate) mod claude;
pub(crate) mod runner;
pub(crate) mod stream;
pub(crate) mod style;

/// Prefix of tool-call ids that Warp only displays. The CLI already ran those tools, so the
/// action model must not execute them again.
pub(crate) const TOOL_ID_PREFIX: &str = "cli-tool:";

/// How a displayed tool ended.
pub(crate) struct ToolOutcome {
    pub tool_call_id: String,
    pub task_id: String,
    pub is_error: bool,
    pub output: String,
}

static OUTCOMES: std::sync::Mutex<Vec<ToolOutcome>> = std::sync::Mutex::new(Vec::new());

/// Remembers how a displayed tool ended, until the response finishes and the UI collects it.
pub(crate) fn record_outcome(outcome: ToolOutcome) {
    OUTCOMES.lock().unwrap().push(outcome);
}

pub(crate) fn drain_outcomes() -> Vec<ToolOutcome> {
    std::mem::take(&mut *OUTCOMES.lock().unwrap())
}

/// Card title for a displayed tool: `Bash · ls -la`. No `(` so the card keeps the whole title
/// (the card treats the first `(` as the start of the arguments).
pub(crate) fn tool_title(name: &str, input: &serde_json::Value) -> String {
    let pick = ["command", "pattern", "query", "url", "file_path", "path", "description"]
        .iter()
        .find_map(|k| input[*k].as_str());
    let s = pick.or_else(|| input.as_object()?.values().find_map(serde_json::Value::as_str));
    let s = s.and_then(|s| s.lines().next()).unwrap_or("");
    let s: String = s.chars().take(100).collect::<String>().replace('(', "[").replace(')', "]");
    if s.is_empty() { name.to_owned() } else { format!("{name} · {s}") }
}

pub(crate) fn is_display_only(tool_call_id: &str) -> bool {
    tool_call_id.starts_with(TOOL_ID_PREFIX)
}

/// JSON object -> protobuf `Struct` (tool arguments).
pub(crate) fn json_to_struct(v: &serde_json::Value) -> prost_types::Struct {
    use prost_types::value::Kind;
    fn conv(v: &serde_json::Value) -> prost_types::Value {
        use serde_json::Value as J;
        prost_types::Value {
            kind: Some(match v {
                J::Null => Kind::NullValue(0),
                J::Bool(b) => Kind::BoolValue(*b),
                J::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or_default()),
                J::String(s) => Kind::StringValue(s.clone()),
                J::Array(a) => Kind::ListValue(prost_types::ListValue { values: a.iter().map(conv).collect() }),
                J::Object(_) => Kind::StructValue(json_to_struct(v)),
            }),
        }
    }
    prost_types::Struct {
        fields: v.as_object().into_iter().flatten().map(|(k, v)| (k.clone(), conv(v))).collect(),
    }
}

/// Prefix that marks an `LLMId` as "run this through a CLI backend": `cli:<backend>/<model>`.
const ID_PREFIX: &str = "cli:";

/// Provider-neutral events a backend emits while handling one user turn.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AgentEvent {
    /// Visible answer text.
    Text(String),
    /// Model thinking text.
    Reasoning(String),
    /// The agent started a tool. The CLI executes it itself; Warp only displays it.
    ToolStarted { id: String, name: String, input: serde_json::Value },
    /// A tool finished. `is_error` marks failures; `output` is the tool's result text.
    ToolFinished { id: String, is_error: bool, output: String },
    /// One-line turn summary (duration, tokens, cost) shown under the answer.
    Usage(String),
    /// The turn ended normally.
    Done,
    /// The turn failed; the message is shown to the user.
    Error(String),
}

/// Everything a backend needs to run one turn.
#[derive(Debug, Clone)]
pub(crate) struct TurnRequest {
    pub prompt: String,
    pub cwd: Option<PathBuf>,
    /// Backend-specific model name, `None` for the CLI default.
    pub model: Option<String>,
    /// Id of the conversation on the CLI side (`resume` is true once it exists).
    pub session_id: String,
    pub resume: bool,
}

pub(crate) trait AgentBackend: Send + Sync {
    /// Stable id used in model ids and settings.
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    /// Executable looked up on `PATH`.
    fn binary(&self) -> &'static str;
    /// Models offered in the picker (first is the default). Empty means "CLI default only".
    fn models(&self) -> &'static [&'static str];
    /// Runs one turn and streams its events. Dropping the stream kills the process.
    fn run_turn(&self, req: TurnRequest) -> BoxStream<'static, AgentEvent>;
}

pub(crate) fn backends() -> &'static [&'static dyn AgentBackend] {
    &[&claude::ClaudeCode]
}

pub(crate) fn backend_by_id(id: &str) -> Option<&'static dyn AgentBackend> {
    backends().iter().copied().find(|b| b.id() == id)
}

/// Whether the backend's executable is on `PATH` (also tries the Windows executable suffixes).
pub(crate) fn is_installed(backend: &dyn AgentBackend) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    let exts: &[&str] = if cfg!(windows) { &["", ".exe", ".cmd", ".bat"] } else { &[""] };
    std::env::split_paths(&path)
        .any(|dir| exts.iter().any(|e| dir.join(format!("{}{e}", backend.binary())).is_file()))
}

pub(crate) fn encode_model_id(backend: &str, model: &str) -> String {
    format!("{ID_PREFIX}{backend}/{model}")
}

/// `(backend, model)` for a `cli:` id. `model` is empty for the CLI default.
pub(crate) fn decode_model_id(id: &str) -> Option<(&'static dyn AgentBackend, Option<String>)> {
    let rest = id.strip_prefix(ID_PREFIX)?;
    let (backend, model) = rest.split_once('/').unwrap_or((rest, ""));
    let model = (!model.is_empty() && model != "default").then(|| model.to_owned());
    Some((backend_by_id(backend)?, model))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
