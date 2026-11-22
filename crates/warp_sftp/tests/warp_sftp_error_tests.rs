//! Unit tests for the warp_sftp::error module
//!
//! author: logic
//! date: 2026/05/26

use warp_sftp::error::{SftpChannelError, SftpError};

// ============================================================
// SftpError Display tests
// ============================================================

/// Verifies the ConnectionFailed formatted output
#[test]
fn test_sftp_error_connection_failed() {
    let err = SftpError::ConnectionFailed("host unreachable".to_string());
    assert_eq!(format!("{err}"), "Connection failed: host unreachable");
}

/// Verifies the AuthFailed formatted output
#[test]
fn test_sftp_error_auth_failed() {
    let err = SftpError::AuthFailed("bad password".to_string());
    assert_eq!(format!("{err}"), "Authentication failed: bad password");
}

/// Verifies the Timeout formatted output
#[test]
fn test_sftp_error_timeout() {
    let err = SftpError::Timeout;
    assert_eq!(format!("{err}"), "Operation timed out");
}

/// Verifies the NoSuchFile formatted output
#[test]
fn test_sftp_error_no_such_file() {
    let err = SftpError::NoSuchFile("/tmp/missing.txt".to_string());
    assert_eq!(format!("{err}"), "File not found: /tmp/missing.txt");
}

/// Verifies the PermissionDenied formatted output
#[test]
fn test_sftp_error_permission_denied() {
    let err = SftpError::PermissionDenied("/root/secret".to_string());
    assert_eq!(format!("{err}"), "Permission denied: /root/secret");
}

/// Verifies the General formatted output
#[test]
fn test_sftp_error_general() {
    let err = SftpError::General("something went wrong".to_string());
    assert_eq!(format!("{err}"), "Operation failed: something went wrong");
}

// ============================================================
// SftpChannelError Display tests
// ============================================================

/// Verifies the SendFailed formatted output
#[test]
fn test_sftp_channel_error_send_failed() {
    let err = SftpChannelError::SendFailed("channel closed".to_string());
    assert_eq!(format!("{err}"), "Failed to send request: channel closed");
}

/// Verifies the RecvFailed formatted output
#[test]
fn test_sftp_channel_error_recv_failed() {
    let err = SftpChannelError::RecvFailed("timeout".to_string());
    assert_eq!(format!("{err}"), "Failed to receive response: timeout");
}

// ============================================================
// From<SftpError> for SftpChannelError tests
// ============================================================

/// Verifies that SftpError converts to SftpChannelError::Sftp
#[test]
fn test_sftp_channel_error_from_sftp_error() {
    let sftp_err = SftpError::General("inner error".to_string());
    let channel_err: SftpChannelError = sftp_err.into();
    match channel_err {
        SftpChannelError::Sftp(inner) => {
            assert_eq!(format!("{inner}"), "Operation failed: inner error");
        }
        _ => panic!("expected the SftpChannelError::Sftp variant"),
    }
}
