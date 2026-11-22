//! Global HTTP proxy configuration.
//!
//! See Issue #72: Warp needs a globally configurable proxy setting that applies to all outbound HTTP
//! requests (BYOP model list fetches, autoupdate, conversation loading, etc.).
//!
//! Design notes:
//! - Three [`ProxyMode`] values: `System` / `Custom` / `Off`.
//! - `System` falls back to the default reqwest behavior; the workspace reqwest enables the
//!   `system-proxy` + `macos-system-configuration` features, so macOS reads
//!   SystemConfiguration, Windows reads WinINET and Linux reads env vars such as `HTTP_PROXY`,
//!   with no custom implementation needed.
//! - `Custom` explicitly sets the URL / basic auth / no_proxy list.
//! - `Off` calls [`reqwest::ClientBuilder::no_proxy`], disabling the proxy entirely (including env vars).
//!
//! The app injects the config through [`set_global_proxy_config`] at startup / on settings change;
//! every later [`crate::Client::new`] call reads that global value and applies it to reqwest.
//!
//! reqwest cannot switch the proxy on an already-built `Client`, so after a settings change callers must
//! rebuild the Client instance (e.g. `AutoupdateState::new(http_client::Client::new())`).

use std::sync::{OnceLock, RwLock};

/// Global proxy mode.
///
/// Defaults to `Off`: avoids a `Client` built during cold start, before the app-level settings are
/// injected, picking up an unexpected system proxy detected by reqwest. app::ProxyMode has the same default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProxyMode {
    /// Disables the proxy, including env vars. The default.
    #[default]
    Off,
    /// Follows the system / env vars entirely (default reqwest behavior).
    System,
    /// Uses the proxy explicitly configured in [`ProxyConfig::url`].
    Custom,
}

impl ProxyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ProxyMode::System => "system",
            ProxyMode::Custom => "custom",
            ProxyMode::Off => "off",
        }
    }

    pub fn from_str_lenient(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "system" => ProxyMode::System,
            "custom" => ProxyMode::Custom,
            // off / disabled / none / unknown all fall back to Off (the default) to avoid accidentally using the system proxy.
            _ => ProxyMode::Off,
        }
    }
}

/// Resolved global proxy configuration.
///
/// `username` is stored in plain text in settings.toml; `password` is saved separately through `managed_secrets`
/// (same pattern as the BYOP API key), and the caller injects it into [`Self::password`] before assembling this struct.
#[derive(Clone, Debug, Default)]
pub struct ProxyConfig {
    pub mode: ProxyMode,
    /// Example: `http://proxy.corp:8080`. Only takes effect under [`ProxyMode::Custom`].
    pub url: String,
    pub username: String,
    pub password: String,
    /// Comma-separated host list; an empty string means no exceptions.
    pub no_proxy: String,
}

impl ProxyConfig {
    /// Applies this config to `reqwest::ClientBuilder`.
    ///
    /// On error (`Custom` mode with an invalid URL) logs a warning and falls back to the default reqwest
    /// behavior so `Client::new()` never panics.
    pub fn apply(&self, mut builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        match self.mode {
            ProxyMode::System => builder,
            ProxyMode::Off => builder.no_proxy(),
            ProxyMode::Custom => {
                let trimmed = self.url.trim();
                if trimmed.is_empty() {
                    log::warn!("HTTP proxy is set to Custom but the URL is empty; falling back to reqwest default (system proxy)");
                    return builder;
                }

                let proxy_result = reqwest::Proxy::all(trimmed);
                let mut proxy = match proxy_result {
                    Ok(p) => p,
                    Err(err) => {
                        log::warn!("HTTP proxy URL '{trimmed}' is invalid ({err}); falling back to reqwest default");
                        return builder;
                    }
                };

                if !self.username.is_empty() || !self.password.is_empty() {
                    proxy = proxy.basic_auth(&self.username, &self.password);
                }

                if !self.no_proxy.trim().is_empty() {
                    if let Some(no_proxy) = reqwest::NoProxy::from_string(self.no_proxy.trim()) {
                        proxy = proxy.no_proxy(Some(no_proxy));
                    }
                }

                builder = builder.proxy(proxy);
                builder
            }
        }
    }
}

static GLOBAL_PROXY_CONFIG: OnceLock<RwLock<ProxyConfig>> = OnceLock::new();

fn slot() -> &'static RwLock<ProxyConfig> {
    GLOBAL_PROXY_CONFIG.get_or_init(|| RwLock::new(ProxyConfig::default()))
}

/// Installs a new global proxy config.
///
/// Only affects `Client`s constructed after this call. A constructed `reqwest::Client` cannot switch
/// its proxy, so after a settings change the app layer must rebuild all shared Client instances.
pub fn set_global_proxy_config(cfg: ProxyConfig) {
    let lock = slot();
    if let Ok(mut guard) = lock.write() {
        *guard = cfg;
    } else {
        log::error!("Failed to write global HTTP proxy config: RwLock is poisoned");
    }
}

/// Reads the current global proxy config (returns the default if unset).
pub fn current_proxy_config() -> ProxyConfig {
    let lock = slot();
    match lock.read() {
        Ok(guard) => guard.clone(),
        Err(err) => {
            log::error!("Failed to read global HTTP proxy config: RwLock is poisoned ({err})");
            ProxyConfig::default()
        }
    }
}

#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
