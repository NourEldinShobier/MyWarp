//! Slash commands, skills and plugin commands the CLI offers, for Warp's `/` menu.
//!
//! A short-lived probe process asks the CLI for them with the `initialize` control request
//! (names, descriptions) and stops it again; no model call is made. A turn's `init` event
//! (names only) fills the list when the probe has not.

use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CliCommand {
    pub name: String,
    pub description: Option<String>,
}

static COMMANDS: Mutex<Vec<CliCommand>> = Mutex::new(Vec::new());
static PROBED: AtomicBool = AtomicBool::new(false);

pub(crate) fn set(commands: Vec<CliCommand>) {
    *COMMANDS.lock().unwrap() = commands;
}

/// Used for names from a turn's `init` event, which must not replace richer probe results.
pub(crate) fn set_if_empty(commands: Vec<CliCommand>) {
    let mut known = COMMANDS.lock().unwrap();
    if known.is_empty() {
        *known = commands;
    }
}

/// Known commands. The first call starts a background probe when none are known yet.
pub(crate) fn list() -> Vec<CliCommand> {
    let known = COMMANDS.lock().unwrap().clone();
    if known.is_empty() && !PROBED.swap(true, Ordering::Relaxed) {
        std::thread::spawn(probe);
    }
    known
}

fn valid_name(c: &str) -> bool {
    !c.is_empty() && !c.contains(char::is_whitespace)
}

/// Names from a turn's `init` event line, if it is one.
pub(crate) fn from_init_line(v: &serde_json::Value) -> Option<Vec<CliCommand>> {
    (v["type"] == "system" && v["subtype"] == "init").then(|| {
        v["slash_commands"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c.as_str())
            .filter(|c| valid_name(c))
            .map(|name| CliCommand { name: name.into(), description: None })
            .collect()
    })
}

/// Commands from the `initialize` control response, if the line is one.
pub(crate) fn from_control_response(v: &serde_json::Value) -> Option<Vec<CliCommand>> {
    let list = v["response"]["response"]["commands"].as_array()?;
    Some(
        list.iter()
            .filter_map(|c| {
                let name = c["name"].as_str().filter(|n| valid_name(n))?;
                let description = c["description"]
                    .as_str()
                    .map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "))
                    .filter(|d| !d.is_empty())
                    .map(|d| d.chars().take(200).collect());
                Some(CliCommand { name: name.into(), description })
            })
            .collect(),
    )
}

fn probe() {
    let backend = &super::claude::ClaudeCode;
    if !super::is_installed(backend) {
        return;
    }
    let mut cmd = Command::new(super::AgentBackend::binary(backend));
    cmd.args(["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let Ok(mut child) = cmd.spawn() else { return };
    // Keep stdin open (the CLI waits for input) until the answer arrives.
    let mut stdin = child.stdin.take();
    let ask = r#"{"type":"control_request","request_id":"warp-init","request":{"subtype":"initialize"}}"#;
    if let Some(s) = stdin.as_mut() {
        let _ = writeln!(s, "{ask}");
        let _ = s.flush();
    }
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            if let Some(commands) = from_control_response(&v) {
                set(commands);
                break;
            }
        }
    }
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
#[path = "commands_tests.rs"]
mod tests;
