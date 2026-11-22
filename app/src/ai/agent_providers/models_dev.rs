//! models.dev data source integration.
//!
//! When the user opens the Providers settings page, fetch `https://models.dev/api.json` asynchronously in the background and
//! cache it at `${cache_dir}/models-dev.json`. The next launch reads the cache directly;
//! if the cache hits and is within the TTL (default 24h), no request is made; fetch again when expired/missing.
//!
//! The data structure follows opencode's `provider/models.ts`: the top level is
//! `{ <provider_id>: Provider }`, and a Provider contains `models: { <model_id>: Model }`.
//! We only care about the few fields the UI "quick select" needs:
//! - provider: id / name / api / env (hints which env var is needed)
//! - model:    id / name / limit.context / limit.output / reasoning / tool_call
//!
//! Fields not listed are tolerated via `serde(default)` + `#[allow(dead_code)]`.
//!
//! Design tradeoff: **synchronous cache read, asynchronous network fetch**. The read side serves the UI and must be fast;
//! the fetch side is spawned in the background, failures only log instead of showing errors, and if the cache cannot be read we return empty data and the UI shows
//! "models.dev not fetched yet, please check the network".

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::{Duration, SystemTime};

use http_client::Client;
use serde::{Deserialize, Serialize};

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const CACHE_FILENAME: &str = "models-dev.json";
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// Top-level `models.dev` data: provider_id -> Provider.
pub type Catalog = BTreeMap<String, Provider>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Provider {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// Upstream API base URL, e.g. `https://api.deepseek.com/v1`.
    #[serde(default)]
    pub api: Option<String>,
    /// Env var names this provider usually needs, e.g. `["DEEPSEEK_API_KEY"]`.
    #[serde(default)]
    pub env: Vec<String>,
    /// Available models, keyed by model id.
    #[serde(default)]
    pub models: BTreeMap<String, Model>,
    /// Documentation URL (some providers have one).
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Model {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default = "default_true")]
    pub tool_call: bool,
    /// Whether file attachments are supported (the attachment field, complementary to modalities:
    /// modalities describes native multimodality; attachment covers PDF / generic file attachment protocols).
    #[serde(default)]
    pub attachment: bool,
    /// Input / output modalities, typical values: `text` / `image` / `audio` / `video` / `pdf`.
    #[serde(default)]
    pub modalities: ModelModalities,
    /// Context window limit.
    #[serde(default)]
    pub limit: ModelLimit,
    /// "alpha" / "beta" / "deprecated" tag.
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelModalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

impl ModelModalities {
    pub fn supports_input(&self, modality: &str) -> bool {
        self.input.iter().any(|m| m.eq_ignore_ascii_case(modality))
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelLimit {
    #[serde(default)]
    pub context: u32,
    #[serde(default)]
    pub output: u32,
}

// -- In-process singleton cache --------------------------------------------------

#[derive(Debug, Default)]
struct State {
    /// The loaded catalog. `None` means it has never loaded successfully.
    catalog: Option<Catalog>,
    /// Cache last-modified time (used to decide whether it is expired).
    loaded_at: Option<SystemTime>,
}

fn state() -> &'static RwLock<State> {
    static S: OnceLock<RwLock<State>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(State::default()))
}

fn cache_path() -> PathBuf {
    let mut p = warp_core::paths::cache_dir();
    p.push(CACHE_FILENAME);
    p
}

/// Read a copy of the loaded catalog (no lock wait - clone directly).
/// Returns `None` when there is no data; the UI should show "fetching" / a retry button.
pub fn cached() -> Option<Catalog> {
    state().read().ok().and_then(|s| s.catalog.clone())
}

/// A snapshot of one model's capabilities extracted from models.dev, used by the BYOP UI / chat_stream to decide attachment types.
#[derive(Debug, Clone, Default)]
pub struct ModelCaps {
    pub vision: bool,
    pub pdf: bool,
    pub audio: bool,
    pub attachment: bool,
}

impl ModelCaps {
    pub fn from_model(m: &Model) -> Self {
        Self {
            vision: m.modalities.supports_input("image"),
            pdf: m.modalities.supports_input("pdf") || m.attachment,
            audio: m.modalities.supports_input("audio"),
            attachment: m.attachment,
        }
    }
}

/// Look up by model_id in the loaded catalog and return the capabilities the model declares on models.dev.
///
/// Prefer an exact match on `provider_id` against the catalog provider key; on a miss, fall back to scanning the whole catalog
/// for the first model.id hit. This gives exact matching (when the user's provider.id equals the one in models.dev),
/// and also handles custom user provider ids (e.g. aggregating providers like "openrouter" or "siliconflow"
/// that forward upstream models, whose ids differ from the upstream provider on models.dev).
pub fn lookup_caps(provider_id: &str, model_id: &str) -> Option<ModelCaps> {
    let s = state().read().ok()?;
    let catalog = s.catalog.as_ref()?;
    if let Some(p) = catalog.get(provider_id) {
        if let Some(m) = p.models.get(model_id) {
            return Some(ModelCaps::from_model(m));
        }
    }
    for p in catalog.values() {
        if let Some(m) = p.models.get(model_id) {
            return Some(ModelCaps::from_model(m));
        }
    }
    None
}

/// Load the disk cache into memory (synchronous, non-blocking; only called at process start or when the UI first needs it).
/// If the disk cache does not exist or fails to parse, returns false and the caller should trigger a network fetch.
pub fn load_from_disk() -> bool {
    let path = cache_path();
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let mtime = std::fs::metadata(&path)
        .ok()
        .and_then(|m| m.modified().ok());
    match serde_json::from_slice::<Catalog>(&bytes) {
        Ok(catalog) => {
            if let Ok(mut s) = state().write() {
                s.catalog = Some(catalog);
                s.loaded_at = mtime;
            }
            true
        }
        Err(e) => {
            log::warn!("[models.dev] failed to parse disk cache ({path:?}): {e}");
            false
        }
    }
}

/// Whether the cache is expired - missing or past the TTL.
pub fn is_stale() -> bool {
    let s = match state().read() {
        Ok(s) => s,
        Err(_) => return true,
    };
    match s.loaded_at {
        Some(t) => SystemTime::now()
            .duration_since(t)
            .map(|d| d > CACHE_TTL)
            .unwrap_or(true),
        None => true,
    }
}

/// Fetch models.dev asynchronously and write it to the disk cache and the in-memory cache.
/// Failures only log and are not propagated (the UI caller decides what to show by whether `cached()` is `Some`).
pub async fn fetch_and_cache(client: Client) -> Result<(), String> {
    let resp = client
        .get(MODELS_DEV_URL)
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read the response body: {e}"))?;

    let catalog: Catalog =
        serde_json::from_slice(&bytes).map_err(|e| format!("JSON parsing failed: {e}"))?;

    // Write to disk - failure is not fatal, only log.
    let path = cache_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, &bytes) {
        log::warn!("[models.dev] failed to write disk cache ({path:?}): {e}");
    }

    if let Ok(mut s) = state().write() {
        s.catalog = Some(catalog);
        s.loaded_at = Some(SystemTime::now());
    }
    Ok(())
}

// -- Chip row collapsed/expanded state (process-level, so widget rebuilds do not lose it) --

static CHIPS_EXPANDED: AtomicBool = AtomicBool::new(false);

pub fn chips_expanded() -> bool {
    CHIPS_EXPANDED.load(Ordering::Relaxed)
}

pub fn toggle_chips_expanded() {
    CHIPS_EXPANDED.fetch_xor(true, Ordering::Relaxed);
}

// -- Flag for the most recent network fetch failure -------------------------------------

static FETCH_FAILED: AtomicBool = AtomicBool::new(false);

/// Whether the most recent network fetch failed (meaningful when cached() == None).
pub fn last_fetch_failed() -> bool {
    FETCH_FAILED.load(Ordering::Relaxed)
}

/// Set by the caller in the spawn callback (true on failure; no reset needed on success since cached() is Some by then).
pub fn set_fetch_failed(failed: bool) {
    FETCH_FAILED.store(failed, Ordering::Relaxed);
}

// -- Search filter for the quick-add chip row ------------------------------------------

fn search_state() -> &'static RwLock<String> {
    static S: OnceLock<RwLock<String>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(String::new()))
}

pub fn search_query() -> String {
    search_state()
        .read()
        .ok()
        .map(|s| s.clone())
        .unwrap_or_default()
}

pub fn set_search_query(q: String) {
    if let Ok(mut s) = search_state().write() {
        *s = q;
    }
}

/// Filter the catalog by the current search query, case-insensitive substring match on provider.name and provider.id.
/// An empty query returns all entries in order. Returns an owned Vec so the UI side can take/iter.
pub fn filter_catalog(catalog: &Catalog, query: &str) -> Vec<(String, Provider)> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return catalog
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
    }
    catalog
        .iter()
        .filter(|(id, p)| id.to_lowercase().contains(&q) || p.name.to_lowercase().contains(&q))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Convert a models.dev Model into the AgentProviderModel used by local settings.
///
/// By default, write the image/pdf/audio inferred from the catalog into the fields (so on first sync / quick-add
/// the user sees the model capabilities synced into the toml directly, without expanding the detail).
/// On later syncs the caller only fills new values into None slots; Some(_) is treated as an explicit user override and skipped.
pub fn into_agent_provider_model(model: &Model) -> crate::settings::AgentProviderModel {
    let caps = ModelCaps::from_model(model);
    crate::settings::AgentProviderModel {
        name: if model.name.is_empty() {
            model.id.clone()
        } else {
            model.name.clone()
        },
        id: model.id.clone(),
        context_window: model.limit.context,
        max_output_tokens: model.limit.output,
        reasoning: model.reasoning,
        tool_call: model.tool_call,
        image: Some(caps.vision),
        pdf: Some(caps.pdf),
        audio: Some(caps.audio),
    }
}
