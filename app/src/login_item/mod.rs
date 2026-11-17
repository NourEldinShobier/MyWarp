//! Launch-at-login registration. Only Windows (HKCU Run key) is supported.

#[cfg(target_os = "windows")]
mod windows;

/// Reconciles the HKCU Run entry with the `add_app_as_login_item` setting.
/// Skipped in integration tests and non-release-bundle builds so dev machines
/// don't auto-launch `target/debug/warp`.
#[cfg(target_os = "windows")]
pub fn maybe_register_app_as_login_item(ctx: &mut warpui::AppContext) {
    if std::env::var("WARP_INTEGRATION").is_ok() {
        return;
    }
    if !warp_core::channel::ChannelState::is_release_bundle() {
        log::debug!("Not a release bundle, skipping login-item registration");
        return;
    }
    windows::maybe_register_app_as_login_item(ctx);
}
