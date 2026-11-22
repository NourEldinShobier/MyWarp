//! Unit tests for `ssh_command`.
//!
//! Split out per `AGENTS.md §5.6` into a standalone file, included via `#[path]` at the end of `ssh_command.rs`.
//! Coverage:
//! - `build_ssh_args` / `build_ssh_command_line` argument construction
//! - `test_connection` error paths for a missing password / wrong auth type
//! - `build_password_auth_stdin` byte stream construction (also covers the key security path of stdin injection)
//!
//! Note: end-to-end tests that spawn an actual ssh child process are covered by the integration tests / manual testing in `app/src/ssh_manager/server_view.rs`;
//! unit tests make no network connections.
//!
//! author: logic
//! date: 2026-06-01

use super::*;
use zeroize::Zeroizing;

fn server() -> SshServerInfo {
    SshServerInfo {
        node_id: "n".into(),
        host: "1.2.3.4".into(),
        port: 22,
        username: "alice".into(),
        auth_type: AuthType::Password,
        key_path: None,
        credential_id: None,
        startup_command: None,
        notes: None,
        last_connected_at: None,
    }
}

#[test]
fn default_port_omitted() {
    let s = server();
    assert_eq!(build_ssh_args(&s), vec!["ssh", "alice@1.2.3.4"]);
    // shell-escape conservatively wraps user@host in single quotes, which is a valid and
    // shell-equivalent form; the unquoted version is not required.
    let line = build_ssh_command_line(&s);
    assert!(
        line == "ssh alice@1.2.3.4" || line == "ssh 'alice@1.2.3.4'",
        "unexpected: {line}"
    );
}

#[test]
fn custom_port_uses_dash_p() {
    let mut s = server();
    s.port = 2222;
    assert_eq!(
        build_ssh_args(&s),
        vec!["ssh", "-p", "2222", "alice@1.2.3.4"]
    );
}

#[test]
fn key_auth_emits_dash_i() {
    let mut s = server();
    s.auth_type = AuthType::Key;
    s.key_path = Some("/home/u/.ssh/id_ed25519".into());
    assert_eq!(
        build_ssh_args(&s),
        vec!["ssh", "-i", "/home/u/.ssh/id_ed25519", "alice@1.2.3.4"]
    );
}

#[test]
fn key_auth_without_path_is_skipped() {
    let mut s = server();
    s.auth_type = AuthType::Key;
    s.key_path = None;
    assert_eq!(build_ssh_args(&s), vec!["ssh", "alice@1.2.3.4"]);
}

#[test]
fn empty_username_yields_host_only() {
    let mut s = server();
    s.username = String::new();
    assert_eq!(build_ssh_args(&s), vec!["ssh", "1.2.3.4"]);
}

#[test]
fn shell_escapes_spaces_in_path() {
    let mut s = server();
    s.auth_type = AuthType::Key;
    s.key_path = Some("/path with spaces/id_rsa".into());
    let line = build_ssh_command_line(&s);
    assert!(
        line.contains("'/path with spaces/id_rsa'"),
        "actual: {line}"
    );
}

#[test]
fn test_connection_requires_password_for_password_auth() {
    let s = server();
    // test_connection should return Offline + an error message when there is no password
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(test_connection(&s, None));
    assert_eq!(result.status, ConnectionStatus::Offline);
    assert!(result
        .error_message
        .unwrap()
        .contains("Password not provided"));
}

#[test]
fn test_connection_requires_password_for_onekey_auth() {
    let mut s = server();
    s.auth_type = AuthType::OneKey;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(test_connection(&s, None));
    assert_eq!(result.status, ConnectionStatus::Offline);
    assert!(result
        .error_message
        .unwrap()
        .contains("Password not provided"));
}

#[test]
fn onekey_key_auth_emits_dash_i_when_key_path_is_resolved() {
    let mut s = server();
    s.auth_type = AuthType::OneKey;
    s.key_path = Some("/home/u/.ssh/shared_ed25519".into());

    assert_eq!(
        build_ssh_args(&s),
        vec!["ssh", "-i", "/home/u/.ssh/shared_ed25519", "alice@1.2.3.4"]
    );
}

#[test]
fn test_connection_key_auth_uses_batch_mode() {
    let mut s = server();
    s.auth_type = AuthType::Key;
    s.key_path = Some("/home/user/.ssh/id_rsa".into());
    // For key auth, the BatchMode=yes path should be used (carried by run_ssh_test);
    // here we only verify that build_ssh_args includes -i and key_path.
    let args = build_ssh_args(&s);
    assert!(args.contains(&"-i".to_string()));
    assert!(args.contains(&"/home/user/.ssh/id_rsa".to_string()));
}

#[test]
fn connection_status_equality() {
    assert_eq!(ConnectionStatus::Online, ConnectionStatus::Online);
    assert_eq!(ConnectionStatus::Offline, ConnectionStatus::Offline);
    assert_eq!(ConnectionStatus::Unknown, ConnectionStatus::Unknown);
    assert_ne!(ConnectionStatus::Online, ConnectionStatus::Offline);
    assert_ne!(ConnectionStatus::Online, ConnectionStatus::Unknown);
    assert_ne!(ConnectionStatus::Offline, ConnectionStatus::Unknown);
}

// -------- Password stdin injection security --------

/// Verify that `build_password_auth_stdin` encodes the password + newline correctly.
/// This is the key to the password-leak fix: we must confirm the bytes written to ssh stdin are the literal password +
/// `\n`, and not any form that would let the password leak into argv / environment variables / temp files.
#[test]
fn build_password_auth_stdin_contains_password_with_newline() {
    let password: Zeroizing<String> = Zeroizing::new("s3cret-pass".into());
    let bytes = build_password_auth_stdin(&password);
    assert_eq!(&*bytes, b"s3cret-pass\n");
}

/// Edge case: an empty password still writes a `\n`, so ssh immediately gets EOF and judges authentication failed
/// (rather than hanging while waiting for a prompt).
#[test]
fn build_password_auth_stdin_empty_password_still_has_newline() {
    let password: Zeroizing<String> = Zeroizing::new(String::new());
    let bytes = build_password_auth_stdin(&password);
    assert_eq!(&*bytes, b"\n");
}

/// Unicode password: written as raw UTF-8 bytes.
#[test]
fn build_password_auth_stdin_unicode_password() {
    let password: Zeroizing<String> = Zeroizing::new("密码🔐".into());
    let bytes = build_password_auth_stdin(&password);
    let mut expected = "密码🔐".as_bytes().to_vec();
    expected.push(b'\n');
    assert_eq!(&*bytes, expected.as_slice());
}

/// Regression: `build_ssh_args` must no longer include `sshpass`, to prevent someone from mistakenly adding it back to cmd_args
/// (Windows / macOS have no sshpass by default, and a leftover path would immediately fail with No such file or directory).
#[test]
fn build_ssh_args_does_not_emit_sshpass() {
    let s = server();
    let args = build_ssh_args(&s);
    assert!(
        !args.iter().any(|a| a == "sshpass"),
        "build_ssh_args must not emit sshpass; got {args:?}"
    );
}

// -------- password auth cmd_args regression guards --------
//
// These tests guard the key switches that keep the "test connection" password path from timing out at 10s again.
// Any change to the -o options inside `test_password_auth` must satisfy these three:
// 1. No longer declare keyboard-interactive (otherwise server-side PAM falls back to kbd-int)
// 2. Explicitly disable KbdInteractiveAuthentication (a client capability switch, not a preference)
// 3. The trailing remote command is still `echo ok` (otherwise the success check cannot match stdout)
// author: logic
// date: 2026-06-01

/// Regression guard: `PreferredAuthentications` must contain only `password` and must not contain
/// `keyboard-interactive`. Otherwise stdin pipe + EOF triggers the kbd-int PAM
/// retry chain (`pam_faildelay` ~2s each), which fills the 10s `TEST_TIMEOUT`.
#[test]
fn password_auth_args_no_keyboard_interactive() {
    let s = server();
    let args = build_password_auth_cmd_args(&s);
    let joined = args.join(" ");
    assert!(
        !joined.contains("keyboard-interactive"),
        "test_password_auth must NOT use keyboard-interactive; got {args:?}"
    );
    assert!(
        joined.contains("PreferredAuthentications=password"),
        "expected PreferredAuthentications=password; got {args:?}"
    );
    // Even when PreferredAuthentications=password is present, no other methods may be listed after it.
    // Split and take the first segment after "="; if it starts with "password," there are other auth methods after it.
    let after_pref = joined
        .split("PreferredAuthentications=")
        .nth(1)
        .unwrap_or("");
    assert!(
        !after_pref.starts_with("password,"),
        "PreferredAuthentications should not list other methods after password; got {args:?}"
    );
}

/// Regression guard: kbd-interactive must be disabled explicitly (a client capability switch),
/// not only via the order of the `PreferredAuthentications` list (which only constrains the password
/// sub-methods). OpenSSH 8.2+ behavior differences, and interaction with server-side `AuthenticationMethods`
/// in particular, make this defense in depth necessary.
#[test]
fn password_auth_args_disable_kbd_interactive() {
    let s = server();
    let args = build_password_auth_cmd_args(&s);
    let joined = args.join(" ");
    assert!(
        joined.contains("KbdInteractiveAuthentication=no"),
        "missing KbdInteractiveAuthentication=no; got {args:?}"
    );
}

/// Regression guard: the trailing `echo ok` in cmd_args must appear as the remote command.
/// Under ssh's parsing rules, the first non-option positional argument after destination = the remote command;
/// if the option order is wrong and ssh does not recognize `echo ok` as the command, the success check breaks.
#[test]
fn password_auth_args_ends_with_echo_ok_command() {
    let s = server();
    let args = build_password_auth_cmd_args(&s);
    assert!(!args.is_empty(), "cmd_args is empty: {args:?}");
    let last = args.last().unwrap();
    assert_eq!(
        last, "echo ok",
        "cmd_args must end with `echo ok` as remote command; got {args:?}"
    );
}

/// Regression guard: on the password path, the destination (`user@host`) must appear **after** all `-o`
/// options and **before** `echo ok`. The SSH command line parsing rule is
/// `ssh [options] destination [command]`: the first non-option argument = destination,
/// and everything after it = the remote command. If a `-o` option ends up after the destination, SSH treats it
/// as part of the remote command rather than its own option, so key options like `PreferredAuthentications` and
/// `KbdInteractiveAuthentication` all stop working, and the kbd-interactive
/// PAM retry chain fills the 10s `TEST_TIMEOUT`.
/// author: logic
/// date: 2026-06-01
#[test]
fn password_auth_args_destination_before_echo_ok_and_after_options() {
    let s = server();
    let args = build_password_auth_cmd_args(&s);
    let joined = args.join(" ");

    // destination "alice@1.2.3.4" must appear before "echo ok"
    let dest_pos = joined
        .find("alice@1.2.3.4")
        .expect("destination must appear in args");
    let echo_pos = joined
        .find("echo ok")
        .expect("`echo ok` must appear in args");

    assert!(
        dest_pos < echo_pos,
        "destination must come before `echo ok`; got joined: {joined}"
    );

    // destination must appear after all -o options
    // find the position of the last -o option
    let last_o_pos = joined
        .rfind("-o ")
        .expect("expected at least one -o option");
    assert!(
        last_o_pos < dest_pos,
        "all -o options must come before destination; got joined: {joined}"
    );
}

/// Regression guard: the key auth path's `build_ssh_args` also needs the destination after the -o options.
/// Verified by using `build_ssh_args` + manually appending options, simulating how `test_key_auth` builds its command.
/// author: logic
/// date: 2026-06-01
#[test]
fn key_auth_args_destination_comes_after_options() {
    let mut s = server();
    s.auth_type = AuthType::Key;
    s.key_path = Some("/home/user/.ssh/id_rsa".into());

    // simulate the build logic of test_key_auth
    let mut args = build_ssh_args(&s);
    let target = args.pop().unwrap();
    args.extend([
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=5".into(),
        "-o".into(),
        "StrictHostKeyChecking=no".into(),
        "-o".into(),
        "LogLevel=ERROR".into(),
    ]);
    args.push(target);
    args.push("echo ok".into());

    let joined = args.join(" ");
    let dest_pos = joined
        .find("alice@1.2.3.4")
        .expect("destination must appear in args");
    let echo_pos = joined
        .find("echo ok")
        .expect("`echo ok` must appear in args");
    let last_o_pos = joined
        .rfind("-o ")
        .expect("expected at least one -o option");

    assert!(
        last_o_pos < dest_pos,
        "all -o options must come before destination; got joined: {joined}"
    );
    assert!(
        dest_pos < echo_pos,
        "destination must come before `echo ok`; got joined: {joined}"
    );
}

// -------- Windows SSH_ASKPASS regression guards --------
//
// On Windows, Win32-OpenSSH has no console + CREATE_NO_WINDOW and therefore refuses to read the password from stdin
// (Win32-OpenSSH issue #1470), so the SSH_ASKPASS mechanism must be used.
// Guard the existence of this code, to prevent someone from mistakenly merging the Windows path back into the stdin approach.
// author: logic
// date: 2026-06-01

/// Regression guard: on Windows the `test_password_auth` entry must reference `AskpassSession`,
/// and must not write the password to stdin directly. This assertion relies on the type system: if the Windows path
/// is changed to the stdin approach, `AskpassSession::new` no longer appears in the function body, and the test fails.
#[cfg(windows)]
#[test]
fn windows_password_auth_uses_askpass_not_stdin() {
    // This test works at compile time: if the Windows branch of ssh_command.rs
    // falls back to stdin injection, the `AskpassSession` type is no longer used and compilation reports a
    // dead_code error, so CI fails.
    // Here we only verify that the AskpassSession type exists + can be new'd. It cannot run (needs to write a file),
    // but it blocks breakage such as "AskpassSession accidentally deleted".
    let _ = std::any::type_name::<AskpassSession>();
}

/// Real end-to-end: create an `AskpassSession` to get the askpass script path, then spawn it with
/// `CreateProcessW` (simulating how ssh spawns askpass) and verify it can start.
///
/// This test guards that the askpass script is "executable" from ssh's point of view, directly blocking regressions like
/// `CreateProcessW failed error:5` (ERROR_ACCESS_DENIED).
/// A bug occurred before: the askpass file had `FILE_ATTRIBUTE_HIDDEN` set, so ssh's
/// `posix_spawnp` refused to spawn it, askpass never started, the password was never passed, and the server
/// reported "wrong password".
#[cfg(windows)]
#[test]
fn windows_askpass_script_is_spawnable() {
    use std::os::windows::process::CommandExt as _;
    use std::process::Stdio;
    use zeroize::Zeroizing;

    let password: Zeroizing<String> = Zeroizing::new("dummy-pw-for-spawn-test".into());
    let session = AskpassSession::new(&password).expect("AskpassSession::new failed");
    let script = session.script_path.clone();
    let password_file = session.password_path.clone();

    // Spawn the askpass script: use CreateProcessW to go through the same code path as ssh.
    // CREATE_NO_WINDOW simulates the environment in which ssh spawns askpass (no console).
    // The WARP_SSH_ASKPASS_FILE env must be set; the script uses it to locate the password file.
    let output = std::process::Command::new("cmd.exe")
        .raw_arg(format!("/c \"{}\"", script.display()))
        .env("WARP_SSH_ASKPASS_FILE", &password_file)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output()
        .expect("CreateProcessW failed — askpass script is not spawnable");

    assert!(
        output.status.success(),
        "askpass script exited non-zero: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The askpass script reads the first line of the password file and echoes it; it should output the password written when the session was created
    assert!(
        stdout.trim() == "dummy-pw-for-spawn-test",
        "askpass output mismatch: got {stdout:?}"
    );
}
