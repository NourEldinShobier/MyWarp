//! Custom Agent provider support.
//!
//! This module is responsible for:
//! - Storing each Provider's `api_key` securely in the OS keychain (secure_storage),
//!   while Provider metadata (name/base_url/model list) goes in the plain settings.toml.
//! - Calling `${base_url}/models` through `OpenAiCompatibleClient`
//!   to fetch the upstream model list (used by the UI "Fetch models" button).
//!
//! The second phase will implement the `AiProvider` trait on top of this configuration,
//! routing the Agent's multi-agent calls to local Providers.

pub mod active_ai;
pub mod attachment_caps;
pub mod chat_stream;
pub mod content_tool_calls;
pub(crate) mod header_vars;
pub mod llm_id;
pub mod models_dev;
pub mod oneshot;
pub mod openai_compatible;
pub mod prompt_renderer;
pub mod reasoning;
mod retry;
pub mod secrets;
pub mod tools;
pub mod user_context;

#[cfg(test)]
#[path = "mod_test.rs"]
mod tests;

#[cfg(test)]
mod cache_stability_tests;

// Current external use sites:
// - `fetch_openai_compatible_models`: the FetchAgentProviderModels handler in ai_page.rs
// - `AgentProviderSecrets`: several handlers in ai_page.rs and the registration point in lib.rs
// The remaining symbols (`OpenAiCompatibleError`/`OpenAiCompatibleModel`/`AgentProviderSecretsEvent`)
// are still reachable through full paths such as `crate::ai::agent_providers::openai_compatible::*`;
// they are not re-exported here to avoid `unused_imports` warnings.
pub use openai_compatible::fetch_openai_compatible_models;
pub use secrets::AgentProviderSecrets;

// ---------------------------------------------------------------------------
// LLMInfo synthesis: turn the agent_providers configured in settings into a form the picker can use
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use settings::Setting;
use warpui::{AppContext, SingletonEntity};

use crate::ai::llms::{
    AvailableLLMs, DisableReason, LLMContextWindow, LLMInfo, LLMProvider, LLMUsageMetadata,
    ModelsByFeature,
};
use crate::settings::{AISettings, AgentProvider};

/// Synthesizes the LLMInfo list of all valid (provider, model) pairs for the given provider.
///
/// "Valid" = the provider has a non-empty base_url + at least 1 model.
/// **API key is optional**: local providers without auth (ollama / lm-studio / vllm, etc.) may leave it empty;
/// with no key the models are still exposed to the picker, and requests are still sent at runtime, just without `Authorization`.
/// Invalid providers (no base_url or no models) are ignored entirely and their models are not shown in the picker,
/// so users can see at a glance "which providers are incomplete -> do not appear".
fn build_byop_llm_infos(app: &AppContext) -> Vec<LLMInfo> {
    let providers = AISettings::as_ref(app).agent_providers.value().clone();
    let mut out = Vec::new();

    for provider in providers {
        if provider.base_url.trim().is_empty() {
            continue;
        }
        if provider.models.is_empty() {
            continue;
        }

        let provider_label = if provider.name.trim().is_empty() {
            provider.id.clone()
        } else {
            provider.name.clone()
        };

        for model in &provider.models {
            if model.id.trim().is_empty() {
                continue;
            }
            let display_name = if model.name.trim().is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            };
            // Three-level priority resolves the final capability: user force toggle via the settings tri-state chip ->
            // models.dev catalog inference -> substring fallback.
            // This is the same function chat_stream uses to decide whether to put in ContentPart::Binary,
            // so UI display and runtime behavior always agree.
            let resolved_caps =
                attachment_caps::resolve_for_model(&provider.id, provider.api_type, model);
            let vision_supported = resolved_caps.images;
            out.push(LLMInfo {
                display_name: format!("{provider_label} / {display_name}"),
                base_model_name: format!("{provider_label} / {display_name}"),
                id: llm_id::encode(&provider.id, &model.id),
                reasoning_level: None,
                usage_metadata: LLMUsageMetadata {
                    request_multiplier: 1,
                    credit_multiplier: None,
                },
                description: None,
                disable_reason: None,
                vision_supported,
                spec: None,
                provider: LLMProvider::Unknown,
                host_configs: HashMap::new(),
                discount_percentage: None,
                context_window: LLMContextWindow::default(),
            });
        }
    }

    out.extend(cli_llm_infos());
    out
}

/// Picker entries for every installed CLI backend (`cli:<backend>/<model>`).
fn cli_llm_infos() -> Vec<LLMInfo> {
    let mut out = Vec::new();
    for backend in crate::ai::cli_agents::backends().iter().filter(|b| crate::ai::cli_agents::is_installed(**b)) {
        for model in backend.models() {
            let name = format!("{} / {model}", backend.display_name());
            out.push(LLMInfo {
                display_name: name.clone(),
                base_model_name: name,
                id: ai::LLMId::from(crate::ai::cli_agents::encode_model_id(backend.id(), model)),
                reasoning_level: None,
                usage_metadata: LLMUsageMetadata { request_multiplier: 1, credit_multiplier: None },
                description: Some("Runs through your installed CLI and its login".to_owned()),
                disable_reason: None,
                vision_supported: false,
                spec: None,
                provider: LLMProvider::Unknown,
                host_configs: HashMap::new(),
                discount_percentage: None,
                context_window: LLMContextWindow::default(),
            });
        }
    }
    out
}

/// Placeholder entry: when the user has not configured any valid provider, the picker needs at least 1 entry
/// (`AvailableLLMs::new` rejects an empty list). The entry is grayed out with `DisableReason::Unavailable`,
/// cannot be selected, and prompts the user to configure one in settings.
fn placeholder_llm_info() -> LLMInfo {
    LLMInfo {
        display_name: crate::t!("byop-placeholder-no-provider"),
        base_model_name: crate::t!("byop-placeholder-not-configured"),
        id: ai::LLMId::from("byop-placeholder"),
        reasoning_level: None,
        usage_metadata: LLMUsageMetadata {
            request_multiplier: 1,
            credit_multiplier: None,
        },
        description: None,
        disable_reason: Some(DisableReason::Unavailable),
        vision_supported: false,
        spec: None,
        provider: LLMProvider::Unknown,
        host_configs: HashMap::new(),
        discount_percentage: None,
        context_window: LLMContextWindow::default(),
    }
}

/// Builds a `ModelsByFeature` populated entirely with BYOP models.
/// The 4 features (agent_mode / coding / cli_agent / computer_use) use the same model set —
/// custom providers do not distinguish capabilities, so every model can serve any feature.
pub fn build_byop_models_by_feature(app: &AppContext) -> ModelsByFeature {
    let mut choices = build_byop_llm_infos(app);
    if choices.is_empty() {
        choices.push(placeholder_llm_info());
    }

    let default_id = choices[0].id.clone();
    let make = || {
        AvailableLLMs::new(default_id.clone(), choices.clone(), None)
            .expect("choices is non-empty by construction")
    };

    ModelsByFeature {
        agent_mode: make(),
        coding: make(),
        cli_agent: Some(make()),
        computer_use: Some(make()),
    }
}

/// Given a BYOP `LLMId`, looks up `(provider, api_key, model_id)` from `AISettings` and secrets.
/// Returns `None` if any piece is missing (the controller caller should map it to an `InvalidApiKey` error).
pub fn lookup_byop(app: &AppContext, id: &ai::LLMId) -> Option<(AgentProvider, String, String)> {
    let (provider_id, model_id) = llm_id::decode(id)?;
    let providers = AISettings::as_ref(app).agent_providers.value().clone();
    let provider = providers.into_iter().find(|p| p.id == provider_id)?;
    // API key is optional: with no key an empty string is returned, and downstream build_client passes
    // `AuthData::from_single("")` to genai -- no `Authorization` is attached, which suits local no-auth services such as ollama.
    let api_key = AgentProviderSecrets::as_ref(app)
        .get(&provider_id)
        .map(str::to_owned)
        .unwrap_or_default();
    Some((provider, api_key, model_id))
}
