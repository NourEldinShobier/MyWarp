//! Exa MCP wire protocol (pure logic, no HTTP I/O).
//!
//! Mirrors opencode `packages/opencode/src/tool/mcp-exa.ts`:
//! - Endpoint: `https://mcp.exa.ai/mcp` (anonymous by default) or with `?exaApiKey=...`
//! - Protocol: JSON-RPC 2.0 POST, `Accept: application/json, text/event-stream`
//! - Response: SSE; scan lines for the `data: ` prefix and parse `result.content[0].text`
//!
//! All HTTP calls live in `web_runtime.rs`; this module only builds the request body and parses the response string.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const EXA_BASE_URL: &str = "https://mcp.exa.ai/mcp";
pub const SEARCH_TOOL_NAME: &str = "web_search_exa";

/// Builds the final Exa endpoint URL. When `api_key=Some`, the key is appended to the querystring (percent-encoded).
pub fn endpoint_url(api_key: Option<&str>) -> String {
    match api_key {
        Some(k) if !k.trim().is_empty() => {
            let encoded: String = url::form_urlencoded::byte_serialize(k.as_bytes()).collect();
            format!("{EXA_BASE_URL}?exaApiKey={encoded}")
        }
        _ => EXA_BASE_URL.to_owned(),
    }
}

/// `web_search_exa` arguments (sent to Exa as-is).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchArgs {
    pub query: String,
    /// "auto" / "fast" / "deep"
    #[serde(rename = "type")]
    pub search_type: String,
    #[serde(rename = "numResults")]
    pub num_results: u32,
    /// "fallback" / "preferred"
    pub livecrawl: String,
    #[serde(
        rename = "contextMaxCharacters",
        skip_serializing_if = "Option::is_none"
    )]
    pub context_max_characters: Option<u32>,
}

impl SearchArgs {
    /// opencode defaults (websearch.ts:54-58).
    pub fn with_defaults(query: String) -> Self {
        Self {
            query,
            search_type: "auto".to_owned(),
            num_results: 8,
            livecrawl: "fallback".to_owned(),
            context_max_characters: None,
        }
    }
}

/// JSON-RPC 2.0 `tools/call` request body. `id` is fixed at 1 (single call, no id disambiguation needed).
pub fn build_request_body(tool_name: &str, args: &SearchArgs) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args,
        }
    })
}

/// Parses an Exa SSE response: scans each line and, for the first `data: ` line, JSON-parses it and takes `result.content[0].text`.
///
/// Returns `Ok(Some(text))` = content found; `Ok(None)` = no content at all (empty result);
/// `Err` = a data line exists but JSON parsing failed or the structure does not match.
pub fn parse_sse_body(body: &str) -> Result<Option<String>> {
    let mut last_err: Option<anyhow::Error> = None;
    for line in body.split('\n') {
        let Some(payload) = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
        else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(payload) {
            Ok(v) => {
                if let Some(text) = extract_first_text(&v) {
                    return Ok(Some(text));
                }
                // The data: line parsed but has no content; move on to the next one
            }
            Err(e) => {
                last_err = Some(anyhow!("invalid Exa SSE JSON payload: {e}"));
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e).context("no Exa SSE data line yielded usable content");
    }
    Ok(None)
}

fn extract_first_text(v: &Value) -> Option<String> {
    let content = v.get("result")?.get("content")?.as_array()?;
    let first = content.first()?;
    let text = first.get("text")?.as_str()?;
    Some(text.to_owned())
}

#[cfg(test)]
#[path = "exa_tests.rs"]
mod exa_tests;
