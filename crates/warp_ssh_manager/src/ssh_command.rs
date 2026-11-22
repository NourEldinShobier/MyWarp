//! Builds a `SshServerInfo` into a `ssh ...` command and spawns the connection-test child process.
//!
//! `build_ssh_command_line` is called when writing to the PTY and shell-escapes each arg,
//! so spaces or single quotes in the username / host / key_path cannot break the command line.
//!
//! ## Password auth security & cross-platform compatibility
//!
//! **Non-Windows**: `ssh` can read the password from stdin in piped-stdin mode, so a one-shot
//! stdin injection is used (`build_password_auth_stdin`). The password is held only in memory as
//! `Zeroizing<String>`, never in argv, so it does not appear in `/proc/<pid>/cmdline`,
//! `ps` or other process info readable on the same machine (a fix for the sshpass `-p` mode).
//!
//! **Windows**: even when stdin is a pipe, Win32-OpenSSH refuses to read the password from stdin because of
//! `CREATE_NO_WINDOW` (no console); it prints
//! `GetConsoleMode on STD_INPUT_HANDLE failed` and hangs, see
//! PowerShell/Win32-OpenSSH issue #1470. The workaround is `SSH_ASKPASS`:
//! write a temporary .cmd script; ssh spawns it and reads its stdout as the password, bypassing stdin
//! and the console entirely. `SSH_ASKPASS_REQUIRE=force` forces the askpass path. The password itself
//! is passed to the askpass script through a temp file (not an env var, to reduce the leak surface), and the whole lifetime
//! is managed by the `AskpassSession` RAII guard, which cleans up as soon as ssh exits.

use crate::types::{AuthType, ConnectionStatus, SshServerInfo};
#[cfg(not(windows))]
use futures_lite::io::AsyncWriteExt as _;
use std::borrow::Cow;
use std::process::Stdio;
use std::time::Duration;
use zeroize::Zeroizing;

pub fn build_ssh_args(server: &SshServerInfo) -> Vec<String> {
    let mut args: Vec<String> = vec!["ssh".into()];
    if server.port != 22 {
        args.push("-p".into());
        args.push(server.port.to_string());
    }
    if matches!(server.auth_type, AuthType::Key | AuthType::OneKey) {
        if let Some(path) = server.key_path.as_deref() {
            if !path.is_empty() {
                args.push("-i".into());
                args.push(path.to_string());
            }
        }
    }
    let target = if server.username.is_empty() {
        server.host.clone()
    } else {
        format!("{}@{}", server.username, server.host)
    };
    args.push(target);
    args
}

pub fn build_ssh_command_line(server: &SshServerInfo) -> String {
    let args = build_ssh_args(server);
    args.iter()
        .map(|a| shell_escape::unix::escape(Cow::Borrowed(a.as_str())).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

pub struct ConnectionTestResult {
    pub status: ConnectionStatus,
    pub latency_ms: Option<u64>,
    pub error_message: Option<String>,
}

pub async fn test_connection(
    server: &SshServerInfo,
    password: Option<Zeroizing<String>>,
) -> ConnectionTestResult {
    let start = instant::Instant::now();

    let result = match server.auth_type {
        AuthType::Key => test_key_auth(server).await,
        AuthType::Password | AuthType::OneKey => test_password_auth(server, password).await,
    };

    let latency = start.elapsed().as_millis() as u64;

    match result {
        Ok(()) => ConnectionTestResult {
            status: ConnectionStatus::Online,
            latency_ms: Some(latency),
            error_message: None,
        },
        Err(e) => ConnectionTestResult {
            status: ConnectionStatus::Offline,
            latency_ms: Some(latency),
            error_message: Some(e),
        },
    }
}

async fn test_key_auth(server: &SshServerInfo) -> Result<(), String> {
    let mut args = build_ssh_args(server);
    // build_ssh_args ends with the destination (user@host); -o options must be inserted before
    // the destination, otherwise SSH treats -o as part of the remote command instead of its own option.
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
    let cmd_args = args;

    match tokio::time::timeout(TEST_TIMEOUT, run_ssh_test(&cmd_args)).await {
        Ok(Ok(output)) => {
            // Strictly match `echo ok`, so a banner/motd that happens to end in "ok" is not misjudged as success.
            if output.trim() == "ok" {
                Ok(())
            } else {
                Err(format!("Unexpected output: {}", output.trim()))
            }
        }
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("Connection timeout".into()),
    }
}

async fn test_password_auth(
    server: &SshServerInfo,
    password: Option<Zeroizing<String>>,
) -> Result<(), String> {
    let password = password.ok_or("Password not provided")?;

    // Build the ssh command arguments (note: -o options must be inserted before the destination, see that function's comment)
    let cmd_args = build_password_auth_cmd_args(server);

    // Platform branch: Windows uses SSH_ASKPASS, other platforms use stdin injection
    #[cfg(windows)]
    return test_password_auth_windows(cmd_args, &password).await;
    #[cfg(not(windows))]
    test_password_auth_unix(cmd_args, &password).await
}

/// Non-Windows platforms: `ssh` can read the password from piped stdin.
#[cfg(not(windows))]
async fn test_password_auth_unix(
    cmd_args: Vec<String>,
    password: &Zeroizing<String>,
) -> Result<(), String> {
    let stdin_bytes = build_password_auth_stdin(password);

    let mut child = command::r#async::Command::new("ssh")
        .args(&cmd_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to start ssh: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&stdin_bytes)
            .await
            .map_err(|e| format!("Failed to write password: {e}"))?;
    }

    let output = match tokio::time::timeout(TEST_TIMEOUT, child.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("Failed to read ssh output: {e}")),
        Err(_) => return Err("Connection timeout".into()),
    };

    finalize_password_test_result(&output)
}

/// Windows: hands the password to ssh via SSH_ASKPASS, bypassing stdin/console entirely.
#[cfg(windows)]
async fn test_password_auth_windows(
    cmd_args: Vec<String>,
    password: &Zeroizing<String>,
) -> Result<(), String> {
    let askpass = AskpassSession::new(password).map_err(|e| format!("Failed to prepare askpass: {e}"))?;

    let mut cmd = command::r#async::Command::new("ssh");
    cmd.args(&cmd_args)
        // ssh no longer needs to read a password from stdin; set to null so ssh does not think there is a tty
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    askpass.apply_env(&mut cmd);

    let child = cmd.spawn().map_err(|e| format!("Failed to start ssh: {e}"))?;

    // When the timeout hits, the child is dropped -> kill_on_drop kills ssh automatically.
    // The askpass guard is dropped at the end of the function, cleaning up the temp files.
    let output = match tokio::time::timeout(TEST_TIMEOUT, child.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("Failed to read ssh output: {e}")),
        Err(_) => return Err("Connection timeout".into()),
    };
    drop(askpass);

    finalize_password_test_result(&output)
}

/// Parses the ssh child process output into a unified success/failure decision (shared by both platforms).
fn finalize_password_test_result(output: &std::process::Output) -> Result<(), String> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr_trimmed = String::from_utf8_lossy(&output.stderr).trim().to_string();

    // Always log ssh's real stderr, even on success, to help diagnose later
    // cases where "the server accepted the password but the UI reported success" differ.
    if !stderr_trimmed.is_empty() {
        log::warn!("ssh test stderr: {stderr_trimmed}");
    }

    // Success check: strictly match the `echo ok` output. The earlier `ends_with("ok")` fallback
    // misjudged success when a banner / motd happened to end in "ok"; it is removed here.
    if output.status.success() && stdout.trim() == "ok" {
        Ok(())
    } else if stderr_trimmed.contains("Permission denied")
        || stderr_trimmed.contains("Authentication failed")
    {
        // The error message includes a trimmed stderr (<= 200 chars) so the user can tell whether the server
        // has password auth disabled or is configured with kbd-only AuthenticationMethods, etc.
        let detail = if stderr_trimmed.is_empty() {
            String::new()
        } else {
            let snippet: String = stderr_trimmed.chars().take(200).collect();
            if stderr_trimmed.chars().count() > 200 {
                format!(" ({snippet}...)")
            } else {
                format!(" ({snippet})")
            }
        };
        Err(format!("Authentication failed: wrong password{detail}"))
    } else {
        Err(format!(
            "Unexpected output: stdout={} stderr={}",
            stdout.trim(),
            stderr_trimmed
        ))
    }
}

/// Encodes the password into the byte stream written to ssh stdin: password UTF-8 + newline.
/// Kept as a pure function so unit tests can assert "stdin contains the password literal + newline".
/// Only the unix branch actually calls it (Windows uses SSH_ASKPASS), but the function compiles on all platforms,
/// so the `build_password_auth_stdin_*` unit tests can also run on Windows CI.
// On Windows only tests call this function (production uses SSH_ASKPASS); dead_code is suppressed
#[cfg_attr(windows, allow(dead_code))]
fn build_password_auth_stdin(password: &Zeroizing<String>) -> Zeroizing<Vec<u8>> {
    let mut v = Zeroizing::new(Vec::with_capacity(password.len() + 1));
    v.extend_from_slice(password.as_bytes());
    v.push(b'\n');
    v
}

/// Builds the full argv passed to the ssh child process for the password-auth test.
///
/// Unlike `build_ssh_args`, this skips the leading `"ssh"` item (we spawn with
/// `Command::new("ssh")` explicitly) and appends the test `-o` options and the `echo ok` remote command.
///
/// Meaning of the key options:
/// - `BatchMode=no`: lets ssh read the password from stdin / askpass (stdin is needed when askpass is not used)
/// - `PreferredAuthentications=password`: declares **only** password, without
///   `keyboard-interactive`. Otherwise the server's PAM triggers a kbd-interactive fallback after password,
///   the kbd-int sub-prompts get no response and are retried one by one,
///   triggering `pam_faildelay` (~2s each), adding up to ~8-10s and hitting `TEST_TIMEOUT`.
/// - `KbdInteractiveAuthentication=no`: client capability switch that disables the whole kbd-int
///   protocol. `PreferredAuthentications` alone is not enough, since it only limits the number of password sub-method
///   prompts and kbd-int can still be used; setting both is defense in depth.
/// - `NumberOfPasswordPrompts=1`: the password sub-method is allowed only 1 retry.
/// - `ConnectTimeout=5`: timeout for a single TCP connection.
/// - `StrictHostKeyChecking=no`: do not block on known_hosts (avoids false alarms from host key
///   changes in the test scenario; real terminal connections use a different path).
/// - `LogLevel=ERROR`: suppresses noise such as host key prompts / banners.
///
/// `echo ok` is the remote command, and stdout is strictly matched to decide success (avoids misjudging a banner / motd
/// that happens to end in "ok").
///
/// author: logic
/// date: 2026-06-01
fn build_password_auth_cmd_args(server: &SshServerInfo) -> Vec<String> {
    // skip(1) drops "ssh" itself (already given to Command::new), leaving
    // ["-p","2222","user@host"]. -o options must be inserted before the destination,
    // otherwise SSH treats -o as part of the remote command instead of its own option.
    let mut args: Vec<String> = build_ssh_args(server).into_iter().skip(1).collect();
    let target = args.pop().unwrap();
    args.extend([
        "-o".into(),
        "BatchMode=no".into(),
        "-o".into(),
        "PreferredAuthentications=password".into(),
        "-o".into(),
        "KbdInteractiveAuthentication=no".into(),
        "-o".into(),
        "NumberOfPasswordPrompts=1".into(),
        "-o".into(),
        "ConnectTimeout=5".into(),
        "-o".into(),
        "StrictHostKeyChecking=no".into(),
        "-o".into(),
        "LogLevel=ERROR".into(),
    ]);
    args.push(target);
    args.push("echo ok".into());
    args
}

async fn run_ssh_test(args: &[String]) -> Result<String, std::io::Error> {
    // Always spawn via command::r#async; on Windows this adds CREATE_NO_WINDOW,
    // avoiding a flashing console window (see the .clippy.toml ban on tokio::process::Command).
    let output = command::r#async::Command::new(&args[0])
        .args(&args[1..])
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // Success check: process exit code 0, or the remote `echo ok` output was returned (some sshpass
    // warnings make the exit code non-zero while stdout still contains "ok").
    if output.status.success() || stdout.contains("ok") {
        Ok(stdout)
    } else {
        Err(std::io::Error::other(stderr))
    }
}

/// Windows-only askpass session: creates a password file + askpass helper script in the temp dir,
/// exposes them to `ssh` via the `SSH_ASKPASS` environment variable, and deletes both files on drop.
///
/// On Windows, even when stdin is a pipe, `ssh.exe` refuses to read the password from
/// stdin because there is no console (it prints `GetConsoleMode on STD_INPUT_HANDLE failed` and hangs),
/// see PowerShell/Win32-OpenSSH issue #1470. The workaround is `SSH_ASKPASS`:
/// when `ssh` sees that environment variable, it spawns the given program and reads its stdout as the password, bypassing
/// stdin and the console entirely. `SSH_ASKPASS_REQUIRE=force` forces ssh onto the askpass path even if it detects a
/// TTY.
///
/// The password is passed to the askpass script through a temp file (not an env var, to reduce the leak surface): an env var
/// would be visible in the `ssh` child process and all its children. The askpass process lives very briefly (ssh
/// forks, execs immediately, and the process exits after reading), so the on-disk window is limited to milliseconds.
///
/// **Security trade-off**: the two temp files do not set `FILE_ATTRIBUTE_HIDDEN` and do not touch ACLs,
/// relying on the default Windows `%TEMP%` isolation (`C:\Users\<user>\AppData\Local\Temp`,
/// separate for each user). An earlier version tried a hidden attribute + icacls tightening to `(R)`, but
/// `FILE_ATTRIBUTE_HIDDEN` makes `posix_spawnp` return `ERROR_ACCESS_DENIED` (error 5)
/// at the `CreateProcessW` stage, so askpass never started and the
/// password was wrongly sent to the server's password prompt (the user saw "wrong password"
/// but nothing had actually been sent). Per-user isolation of the Windows temp dir is enough,
/// so simple and reliable is ranked above "defense in depth" here.
///
/// author: logic
/// date: 2026-06-01
#[cfg(windows)]
struct AskpassSession {
    password_path: std::path::PathBuf,
    script_path: std::path::PathBuf,
}

#[cfg(windows)]
impl AskpassSession {
    fn new(password: &Zeroizing<String>) -> std::io::Result<Self> {
        use std::io::Write as _;

        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let suffix = format!("{pid}-{nanos}");

        let password_path = dir.join(format!("warp-ssh-askpass-{suffix}.txt"));
        let script_path = dir.join(format!("warp-ssh-askpass-{suffix}.cmd"));

        // Write the password to a temp file (no hidden attribute, no ACL changes; see the trade-off in the type doc)
        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&password_path)?;
            f.write_all(password.as_bytes())?;
            f.sync_all()?;
        }

        // Write the askpass helper script: read the first line of the file pointed to by %WARP_SSH_ASKPASS_FILE%
        // and echo it to stdout. `set /p` reads the first line (newline stripped), `echo !PW!` prints it.
        // Uses `setlocal enabledelayedexpansion` + `!PW!` delayed expansion, so a password containing
        // cmd special characters (&, |, <, >, ^) is not re-parsed and truncated by the immediate expansion of %PW%.
        let body = "@echo off\r\nsetlocal enabledelayedexpansion\r\nset /p PW=<\"%WARP_SSH_ASKPASS_FILE%\"\r\necho !PW!\r\nendlocal\r\n";
        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&script_path)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
        }

        Ok(Self {
            password_path,
            script_path,
        })
    }

    /// Attaches the environment variables SSH_ASKPASS needs to the ssh child process.
    fn apply_env(&self, cmd: &mut command::r#async::Command) {
        cmd.env("SSH_ASKPASS", &self.script_path)
            .env("SSH_ASKPASS_REQUIRE", "force")
            .env("WARP_SSH_ASKPASS_FILE", &self.password_path)
            .env_remove("DISPLAY");
    }
}

#[cfg(windows)]
impl Drop for AskpassSession {
    fn drop(&mut self) {
        // Delete both temp files right after ssh exits, shortening the time the password lives on disk.
        // Errors are swallowed: a cleanup failure must not affect the main flow's return value.
        let _ = std::fs::remove_file(&self.password_path);
        let _ = std::fs::remove_file(&self.script_path);
    }
}

#[cfg(test)]
#[path = "ssh_command_tests.rs"]
mod tests;
