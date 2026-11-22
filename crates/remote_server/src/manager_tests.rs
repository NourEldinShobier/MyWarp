//! Pure-function unit tests for `manager.rs`.
//!
//! Only pure-function helpers are covered here - `RemoteServerManager` itself is not touched,
//! because it depends on `warpui::Entity` / `ModelContext` and needs a whole App
//! context; the integration testing framework is a better place for that.

use super::*;

// ---------------------------------------------------------------------------
// version_is_compatible
// ---------------------------------------------------------------------------

#[test]
fn version_compat_both_tagged_and_equal() {
    assert!(version_is_compatible(
        Some("v0.2026.05.10.stable"),
        "v0.2026.05.10.stable",
    ));
}

#[test]
fn version_compat_both_tagged_and_different() {
    assert!(!version_is_compatible(
        Some("v0.2026.05.10.stable"),
        "v0.2026.05.10.preview",
    ));
}

#[test]
fn version_compat_both_untagged() {
    // The client has no GIT_RELEASE_TAG (cargo run), and the server also reports an empty string
    // (`script/deploy_remote_server` dev deploy): treated as compatible, kept so
    // the local dev loop is unaffected.
    assert!(version_is_compatible(None, ""));
}

#[test]
fn version_compat_client_tagged_server_untagged() {
    // Client is a release, server is a dev deploy: treated as incompatible, the reinstall
    // flow is triggered as normal.
    assert!(!version_is_compatible(Some("v0.2026.05.10.stable"), ""));
}

#[test]
fn version_compat_client_untagged_server_tagged() {
    // **Key scenario**: the Warp client has no tag (cargo build),
    // while the server is a release downloaded from the official CDN (with a tag). The original helper judged
    // it incompatible, which would trigger `remove_remote_server_binary` and loop forever.
    // This test only records that `version_is_compatible` itself is unchanged;
    // actually "skipping the check" is handled by [`should_enforce_remote_version_check`].
    assert!(!version_is_compatible(None, "v0.2026.05.10.stable"));
}

// ---------------------------------------------------------------------------
// should_enforce_remote_version_check
// ---------------------------------------------------------------------------

#[test]
fn enforce_version_check_skipped_on_oss() {
    // When Warp temporarily reuses the official release binary, the client and server versions
    // are never equal, so strict checking must be skipped.
    assert!(!should_enforce_remote_version_check(Channel::Oss));
}

#[test]
fn enforce_version_check_kept_on_official_channels() {
    // On the official channel the client and server either both come from the same release CI,
    // or both from a local deploy via `script/deploy_remote_server`; strict
    // checking is still necessary - keep the original stale-binary self-healing path.
    for channel in [
        Channel::Stable,
        Channel::Preview,
        Channel::Dev,
        Channel::Local,
        Channel::Integration,
    ] {
        assert!(
            should_enforce_remote_version_check(channel),
            "channel {channel:?} should still enforce version check"
        );
    }
}
