use super::*;

#[test]
fn allow_echoes_input_and_deny_carries_a_message() {
    let input = json!({"command": "ls"});
    let allow: Value = serde_json::from_str(&response_line("r1", &input, true)).unwrap();
    assert_eq!(allow["response"]["request_id"], "r1");
    assert_eq!(allow["response"]["response"]["behavior"], "allow");
    assert_eq!(allow["response"]["response"]["updatedInput"], input);
    let deny: Value = serde_json::from_str(&response_line("r1", &input, false)).unwrap();
    assert_eq!(deny["response"]["response"]["behavior"], "deny");
    assert_eq!(deny["response"]["response"]["message"], "Rejected in Warp");
}

#[test]
fn respond_sends_once_to_the_parked_channel() {
    let (tx, mut rx) = futures::channel::mpsc::unbounded();
    register("t9", "req9".into(), json!({"a": 1}), tx);
    assert!(is_awaiting("cli-tool:t9"));
    assert!(respond("cli-tool:t9", true));
    assert!(!is_awaiting("cli-tool:t9"));
    assert!(!respond("cli-tool:t9", true));
    let line = rx.try_next().unwrap().unwrap();
    assert!(line.contains("req9") && line.contains("allow"));
}
