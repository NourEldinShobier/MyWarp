use super::*;

fn cmd(name: &str, description: Option<&str>) -> CliCommand {
    CliCommand { name: name.into(), description: description.map(Into::into) }
}

#[test]
fn reads_names_from_init_line() {
    let v: serde_json::Value = serde_json::from_str(
        r#"{"type":"system","subtype":"init","slash_commands":["compact","figma:figma-use","bad name",""]}"#,
    )
    .unwrap();
    assert_eq!(from_init_line(&v), Some(vec![cmd("compact", None), cmd("figma:figma-use", None)]));
}

#[test]
fn other_lines_are_not_init() {
    let v: serde_json::Value = serde_json::from_str(r#"{"type":"assistant"}"#).unwrap();
    assert_eq!(from_init_line(&v), None);
    assert_eq!(from_control_response(&v), None);
}

#[test]
fn reads_descriptors_from_control_response() {
    let v: serde_json::Value = serde_json::from_str(
        r#"{"type":"control_response","response":{"subtype":"success","request_id":"warp-init","response":{"commands":[
            {"name":"compact","description":"Free up\n  context"},{"name":"x","description":""},{"name":"two words"}]}}}"#,
    )
    .unwrap();
    assert_eq!(from_control_response(&v), Some(vec![cmd("compact", Some("Free up context")), cmd("x", None)]));
}
