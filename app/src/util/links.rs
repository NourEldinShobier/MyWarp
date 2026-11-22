use crate::channel::ChannelState;

// Upstream Warp's docs site, Slack and privacy policy no longer apply to this fork;
// these constants are kept as empty placeholder strings until the fork has its own channels.
// `ctx.open_url("")` is a harmless no-op for UI callers.
pub const USER_DOCS_URL: &str = "";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub const GITHUB_ISSUES_URL: &str = "https://github.com/zerx-lab/warp/issues";
pub const SLACK_URL: &str = "";
pub const PRIVACY_POLICY_URL: &str = "";

pub fn feedback_form_url() -> String {
    let mut url = url::Url::parse("https://github.com/zerx-lab/warp/issues/new/choose")
        .expect("Should not fail to parse");
    if let Some(version) = ChannelState::app_version() {
        url.query_pairs_mut().append_pair("warp-version", version);
    }
    url.query_pairs_mut()
        .append_pair("os-version", &os_info::get().version().to_string());
    url.to_string()
}
