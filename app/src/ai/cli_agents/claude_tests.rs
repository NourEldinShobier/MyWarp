use super::*;

#[test]
fn streams_text_and_thinking_deltas() {
    let t = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}}"#;
    assert_eq!(parse_line(t), vec![AgentEvent::Text("Hi".into())]);
    let r = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}"#;
    assert_eq!(parse_line(r), vec![AgentEvent::Reasoning("hmm".into())]);
}

#[test]
fn assistant_message_yields_only_tool_use() {
    let l = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"x"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls -la\nmore"}}]}}"#;
    assert_eq!(
        parse_line(l),
        vec![AgentEvent::ToolStarted {
            id: "t1".into(),
            name: "Bash".into(),
            input: serde_json::json!({"command": "ls -la
more"}),
        }]
    );
}

#[test]
fn tool_result_and_turn_end() {
    let l = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":[{"type":"text","text":"denied"}]}]}}"#;
    assert_eq!(
        parse_line(l),
        vec![AgentEvent::ToolFinished { id: "t1".into(), is_error: true, output: "denied".into() }]
    );
    assert_eq!(parse_line(r#"{"type":"result","is_error":false,"result":"ok"}"#), vec![AgentEvent::Done]);
    assert_eq!(
        parse_line(r#"{"type":"result","is_error":false,"duration_ms":12400,"usage":{"output_tokens":1084},"total_cost_usd":0.8625}"#),
        vec![
            AgentEvent::Usage { line: "12.4s · 1.1k tokens out · $0.86".into(), cost_usd: Some(0.8625) },
            AgentEvent::Done
        ]
    );
    assert_eq!(
        parse_line(r#"{"type":"result","is_error":true,"result":"boom"}"#),
        vec![AgentEvent::Error("boom".into())]
    );
}

#[test]
fn noise_is_ignored() {
    assert!(parse_line("not json").is_empty());
    assert!(parse_line(r#"{"type":"system","subtype":"hook_started"}"#).is_empty());
    assert_eq!(
        parse_line(r#"{"type":"system","subtype":"init","slash_commands":["compact"]}"#),
        vec![AgentEvent::Commands(vec![crate::ai::cli_agents::commands::CliCommand { name: "compact".into(), description: None }])]
    );
}

#[test]
fn project_dir_matches_cli_naming() {
    assert_eq!(project_dir_name(r"D:\workspaces\warp-terminal\code\warp"), "D--workspaces-warp-terminal-code-warp");
    assert_eq!(project_dir_name(r"C:\Users\noure"), "C--Users-noure");
}

#[test]
fn args_resume_vs_new_session_and_model() {
    let req = |resume, model: Option<&str>| TurnRequest {
        prompt: "p".into(), cwd: None, model: model.map(Into::into), session_id: "sid".into(), resume,
    };
    let a = args(&req(false, None));
    assert!(a.windows(2).any(|w| w == ["--session-id", "sid"]));
    assert!(!a.contains(&"--model".to_string()));
    let a = args(&req(true, Some("opus")));
    // No saved session named "sid" exists, so resume falls back to a fresh one with that id.
    assert!(a.windows(2).any(|w| w == ["--session-id", "sid"]));
    assert!(a.windows(2).any(|w| w == ["--model", "opus"]));
}

#[test]
fn permission_request_is_parsed() {
    let l = r#"{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"touch x"},"tool_use_id":"t7"}}"#;
    assert_eq!(
        parse_line(l),
        vec![AgentEvent::PermissionRequest {
            tool_use_id: "t7".into(),
            request_id: "r1".into(),
            tool_name: "Bash".into(),
            input: serde_json::json!({"command": "touch x"}),
        }]
    );
}
