use warp_cli::agent::Harness;

use super::{
    build_local_claude_child_command, build_local_opencode_child_command,
    normalize_local_child_harness, validate_local_harness_shell,
};
use crate::terminal::shell::ShellType;

#[test]
fn normalize_local_child_harness_accepts_supported_aliases() {
    assert_eq!(
        normalize_local_child_harness("claude"),
        Some(Harness::Claude)
    );
    assert_eq!(
        normalize_local_child_harness("claude-code"),
        Some(Harness::Claude)
    );
    assert_eq!(
        normalize_local_child_harness("claude_code"),
        Some(Harness::Claude)
    );
    assert_eq!(
        normalize_local_child_harness("opencode"),
        Some(Harness::OpenCode)
    );
    assert_eq!(
        normalize_local_child_harness("open-code"),
        Some(Harness::OpenCode)
    );
    assert_eq!(
        normalize_local_child_harness("open_code"),
        Some(Harness::OpenCode)
    );
}

#[test]
fn normalize_local_child_harness_rejects_unsupported_values() {
    assert_eq!(normalize_local_child_harness("oz"), None);
    assert_eq!(normalize_local_child_harness("codex"), None);
    assert_eq!(normalize_local_child_harness(""), None);
}

#[test]
fn validate_local_harness_shell_accepts_supported_shells() {
    for shell in [
        ShellType::Bash,
        ShellType::Zsh,
        ShellType::Fish,
        ShellType::PowerShell,
    ] {
        assert_eq!(validate_local_harness_shell(Some(shell)), Ok(()));
    }
}

#[test]
fn validate_local_harness_shell_rejects_unknown_shell() {
    assert_eq!(
        validate_local_harness_shell(None),
        Err(
            "Local child harnesses currently require a detected bash, zsh, fish, or PowerShell session."
                .to_string()
        )
    );
}

#[test]
fn build_local_claude_child_command_quotes_the_prompt() {
    let command = build_local_claude_child_command("hello world", ShellType::Bash);

    assert!(command.starts_with("claude --session-id "));
    assert!(command.ends_with(" --dangerously-skip-permissions 'hello world'"));
}

#[test]
fn build_local_claude_child_command_quotes_the_prompt_for_powershell() {
    let command = build_local_claude_child_command("it's $x", ShellType::PowerShell);

    assert!(command.ends_with(" --dangerously-skip-permissions 'it''s $x'"));
}

#[test]
fn build_local_opencode_child_command_quotes_the_prompt() {
    assert_eq!(
        build_local_opencode_child_command("hello world", ShellType::Zsh),
        "opencode --prompt 'hello world'"
    );
    assert_eq!(
        build_local_opencode_child_command("it's", ShellType::PowerShell),
        "opencode --prompt 'it''s'"
    );
}
