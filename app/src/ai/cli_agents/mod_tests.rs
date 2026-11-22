use super::*;

#[test]
fn model_id_round_trip() {
    let id = encode_model_id("claude", "opus");
    assert_eq!(id, "cli:claude/opus");
    let (backend, model) = decode_model_id(&id).unwrap();
    assert_eq!(backend.id(), "claude");
    assert_eq!(model.as_deref(), Some("opus"));
}

#[test]
fn default_model_decodes_to_none() {
    let (_, model) = decode_model_id("cli:claude/default").unwrap();
    assert_eq!(model, None);
    let (_, model) = decode_model_id("cli:claude").unwrap();
    assert_eq!(model, None);
}

#[test]
fn unknown_backend_or_prefix_is_rejected() {
    assert!(decode_model_id("cli:nope/x").is_none());
    assert!(decode_model_id("byop:a:b").is_none());
}
