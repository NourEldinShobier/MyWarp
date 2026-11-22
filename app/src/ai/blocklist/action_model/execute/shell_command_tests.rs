use std::time::Duration;

use super::*;

#[test]
fn detects_interactive_session_commands_across_platforms() {
    for command in [
        "ssh root@example.com",
        "command ssh localhost",
        "ssh.exe -p 2222 root@example.com",
        "/usr/bin/ssh host",
        r#""C:\Windows\System32\OpenSSH\ssh.exe" -p 22 host"#,
        r#"& "C:\Program Files\OpenSSH\ssh.exe" host"#,
        "warp_run_generator_command 42 'ssh host'",
        " warp_run_generator_command 42 'ssh host'",
        "Warp-Run-GeneratorCommand 42 'ssh host' -ErrorAction Ignore",
        r#"warp_run_generator_command 42 '"C:\Windows\System32\OpenSSH\ssh.exe" host'"#,
        "gcloud compute ssh --zone us-west1-a my-instance",
        "eb ssh --profile my-profile my-env",
        "doctl compute ssh --region nyc1 my-droplet",
        "mosh root@example.com",
        "sftp root@example.com",
        "telnet example.com",
    ] {
        assert_eq!(
            command_starts_non_terminating_session(command),
            true,
            "{command}"
        );
    }
}

#[test]
fn does_not_detect_unrelated_or_non_interactive_ssh_commands() {
    for command in [
        "",
        "echo ssh",
        "git status",
        "ssh-add-key",
        "ssh -T user@host",
        "ssh -v user@host -W localhost:22",
        "ssh user@host ls",
        "ssh.exe user@host ls",
        r#""C:\Windows\System32\OpenSSH\ssh.exe" user@host ls"#,
        r#"& "C:\Program Files\OpenSSH\ssh.exe" user@host ls"#,
        "warp_run_generator_command 42 'ssh user@host ls'",
        "Warp-Run-GeneratorCommand 42 'git status' -ErrorAction Ignore",
        "rsync myfile.txt ssh://user@server.com",
        // A character stuck right after the closing quote is deliberately rejected by tokenize, to avoid being mis-split into `ssh`
        // and then misjudged as an interactive session via `ssh hello-world`.
        r#""ssh"hello-world"#,
        // An unclosed quote is likewise rejected by tokenize.
        r#""ssh hello world"#,
    ] {
        assert_eq!(
            command_starts_non_terminating_session(command),
            false,
            "{command}"
        );
    }
}

#[test]
fn shortens_on_completion_delay_for_interactive_sessions() {
    assert_eq!(
        effective_read_shell_command_delay("ssh host", Some(ShellCommandDelay::OnCompletion)),
        ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_WAIT_DURATION
        }
    );
    assert_eq!(
        effective_read_shell_command_delay(
            r#"& "C:\Program Files\OpenSSH\ssh.exe" host"#,
            Some(ShellCommandDelay::OnCompletion)
        ),
        ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_WAIT_DURATION
        }
    );
    assert_eq!(
        effective_read_shell_command_delay(
            "warp_run_generator_command 42 'ssh host'",
            Some(ShellCommandDelay::OnCompletion)
        ),
        ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_WAIT_DURATION
        }
    );
    assert_eq!(
        effective_read_shell_command_delay("mosh host", None),
        ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_WAIT_DURATION
        }
    );
}

#[test]
fn preserves_explicit_or_non_interactive_read_delays() {
    assert_eq!(
        effective_read_shell_command_delay(
            "ssh host",
            Some(ShellCommandDelay::Duration(Duration::from_secs(8)))
        ),
        ActionResultDelay::Duration(Duration::from_secs(8))
    );
    assert_eq!(
        effective_read_shell_command_delay("git status", Some(ShellCommandDelay::OnCompletion)),
        ActionResultDelay::OnCompletion {
            timeout: ShellCommandExecutor::MAX_AGENT_DELAY_DURATION
        }
    );
    assert_eq!(
        effective_read_shell_command_delay("git status", None),
        ActionResultDelay::Default
    );
}

#[test]
fn requested_command_wait_until_completion_does_not_use_snapshot_timeout() {
    assert_eq!(
        action_result_delay_for_requested_command(true),
        ActionResultDelay::UntilCompletion
    );
    assert_eq!(
        action_result_delay_for_requested_command(false),
        ActionResultDelay::Default
    );
}

#[test]
fn preemption_logic_covers_until_completion_timeout() {
    use ActionResultDelay::{Default, Duration as DurationDelay, OnCompletion, UntilCompletion};
    use WakeReason::*;

    // BlockFinished never preempts - it is the signal that the command has truly finished.
    assert!(!compute_is_preempted(BlockFinished, UntilCompletion));
    assert!(!compute_is_preempted(BlockFinished, Default));
    assert!(!compute_is_preempted(
        BlockFinished,
        OnCompletion {
            timeout: Duration::from_secs(1)
        }
    ));

    // ForceRefresh always preempts, regardless of delay.
    assert!(compute_is_preempted(ForceRefresh, UntilCompletion));
    assert!(compute_is_preempted(ForceRefresh, Default));

    // Timeout + OnCompletion / UntilCompletion preempt.
    assert!(compute_is_preempted(
        Timeout,
        OnCompletion {
            timeout: Duration::from_secs(1)
        }
    ));
    // #138: the pager-stuck fallback timeout must be marked as preempting, so the server does not read it as "command finished".
    assert!(compute_is_preempted(Timeout, UntilCompletion));

    // Timeout + Default / Duration do not preempt - the agent expects to get intermediate snapshots.
    assert!(!compute_is_preempted(Timeout, Default));
    assert!(!compute_is_preempted(
        Timeout,
        DurationDelay(Duration::from_secs(1))
    ));
}

/// Reproduces the hang case: the heredoc terminator must stay alone on its line, and the closing `)` must not be appended after it.
/// If it becomes `PY)`, the shell never sees the terminator, stays at PS2, and the command never ends.
#[test]
fn multiline_heredoc_keeps_delimiter_and_closer_on_their_own_lines() {
    let command = "python3 - <<'PY'\nprint('ok')\nPY";

    for shell in [ShellType::Bash, ShellType::Zsh] {
        let wrapped = wrap_command_without_pager(Some(shell), command);
        let lines: Vec<&str> = wrapped.lines().collect();

        assert_eq!(
            lines[lines.len() - 2],
            "PY",
            "heredoc terminator corrupted: {wrapped}"
        );
        assert_eq!(lines[lines.len() - 1], ")", "closing paren not alone on its line: {wrapped}");
        assert!(wrapped.contains("PAGER=cat"), "pager suppression lost: {wrapped}");
    }
}

/// Same class of flaw: when a command ends with a trailing `#` comment, the closing token gets commented out.
#[test]
fn multiline_trailing_comment_does_not_swallow_closer() {
    let command = "echo start\necho done # 收尾";

    assert!(wrap_command_without_pager(Some(ShellType::Bash), command).ends_with("\n)"));
    assert!(wrap_command_without_pager(Some(ShellType::Fish), command).ends_with("\nend"));
    assert!(wrap_command_without_pager(Some(ShellType::PowerShell), command).ends_with("\n}"));
}

/// A single-line command must keep its original single-line form: on shells that do not support bracketed
/// paste, `bytes_to_execute_command` turns `\n` into `\r`, and extra newlines would split one command into several blocks.
#[test]
fn single_line_command_stays_on_one_line() {
    let command = "cargo check";

    for shell in [
        ShellType::Bash,
        ShellType::Zsh,
        ShellType::Fish,
        ShellType::PowerShell,
    ] {
        let wrapped = wrap_command_without_pager(Some(shell), command);
        assert!(
            !wrapped.contains('\n'),
            "{shell:?} single-line command was split: {wrapped}"
        );
        assert!(wrapped.contains(command), "{shell:?} command lost: {wrapped}");
    }
}

/// An unknown shell cannot be safely decorated; pass it through unchanged.
#[test]
fn unknown_shell_passes_command_through() {
    let command = "python3 - <<'PY'\nprint('ok')\nPY";
    assert_eq!(wrap_command_without_pager(None, command), command);
}
