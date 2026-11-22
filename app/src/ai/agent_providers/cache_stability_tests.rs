//! Prompt cache serialization stability tests (corresponding to doc items P1-8 / P1-9 / P1-13).
//!
//! The Anthropic docs explicitly warn:
//! > Verify that the keys in your `tool_use` content blocks have stable
//! > ordering as some languages (for example, Swift, Go) randomize key order
//! > during JSON conversion, breaking caches
//!
//! This means any `serde_json::Value` produced on the Rust side **must**:
//!   1. Be byte-equal across calls for the same input (deterministic)
//!   2. Not depend on `HashMap` iteration order
//!   3. Not depend on external state (timestamps, randomness, PID, etc.)
//!
//! This suite is Warp's regression guardrail: any later change to the prompt
//! construction path that breaks byte-level stability fails an assertion here.

use crate::ai::agent::{MCPContext, MCPServer};
use api::message;
use warp_multi_agent_api as api;

use super::chat_stream;
use super::tools;

// ---------------------------------------------------------------------------
// P1-8: tool schema field order stability
// ---------------------------------------------------------------------------

/// Call `(parameters)()` twice for each tool in `REGISTRY` and assert byte-equal.
///
/// Risk: if an enum / oneof embedded in a tool schema is converted to a Value from a
/// `HashMap<String, Schema>`, the order becomes random. The `serde_json::Map` produced by a `json!({...})` literal
/// keeps **insertion order** by default (`preserve_order` is on by default, see Cargo.toml), so
/// hard-coded key order is stable across calls. This test guards that invariant.
#[test]
fn registry_tool_schemas_are_deterministic() {
    for tool in tools::REGISTRY {
        let s1 = (tool.parameters)();
        let s2 = (tool.parameters)();
        let j1 = serde_json::to_string(&s1).unwrap();
        let j2 = serde_json::to_string(&s2).unwrap();
        assert_eq!(
            j1, j2,
            "tool `{}` schema must be byte-equal across calls (prerequisite for prompt cache hits)",
            tool.name
        );
    }
}

/// Call each tool in `REGISTRY` 50 times and assert all outputs are byte-equal.
/// Guards against occasional HashMap iteration order drift (2 runs might match by chance).
#[test]
fn registry_tool_schemas_stable_under_repetition() {
    for tool in tools::REGISTRY {
        let baseline = serde_json::to_string(&(tool.parameters)()).unwrap();
        for i in 0..50 {
            let candidate = serde_json::to_string(&(tool.parameters)()).unwrap();
            assert_eq!(
                baseline, candidate,
                "tool `{}` call #{i} output differs from baseline (possible HashMap order drift)",
                tool.name
            );
        }
    }
}

/// `tools::REGISTRY` order is static, but verify it anyway:
/// iterating several times in the same process yields the same (name, description) sequence.
#[test]
fn registry_iteration_order_is_stable() {
    let names1: Vec<&str> = tools::REGISTRY.iter().map(|t| t.name).collect();
    let names2: Vec<&str> = tools::REGISTRY.iter().map(|t| t.name).collect();
    assert_eq!(names1, names2);
}

// ---------------------------------------------------------------------------
// P1-9: serialize_outgoing_tool_call history replay stability
// ---------------------------------------------------------------------------

/// Simulate a Grep tool call and verify two serializations are byte-equal.
/// `serialize_outgoing_tool_call` reruns on every build_chat_request,
/// converting historical ToolCalls into (name, args Value). Any HashMap / time-related
/// instability would invalidate the cache for the latter part of the messages section.
///
/// Grep is chosen because its fields are the simplest (`queries: Vec<String>`, `path: String`),
/// with no dependence on any implicit prost default fields.
#[test]
fn serialize_grep_tool_call_is_deterministic() {
    let tc = message::ToolCall {
        tool_call_id: "call-grep-1".to_owned(),
        tool: Some(message::tool_call::Tool::Grep(message::tool_call::Grep {
            queries: vec!["fn main".to_owned(), "Result<".to_owned()],
            path: "src/".to_owned(),
        })),
    };

    let (n1, v1) = chat_stream::serialize_outgoing_tool_call_for_test(&tc, None, "");
    let (n2, v2) = chat_stream::serialize_outgoing_tool_call_for_test(&tc, None, "");
    assert_eq!(n1, n2, "tool name must match");
    let j1 = serde_json::to_string(&v1).unwrap();
    let j2 = serde_json::to_string(&v2).unwrap();
    assert_eq!(j1, j2, "the same ToolCall must serialize byte-equal across runs");
}

/// Grep `queries` is a `Vec<String>` whose order must be stable (Vec is naturally stable, but this is a defensive assertion).
/// This reflects a broader rule: any Vec field inside a user ToolCall must preserve input order.
#[test]
fn serialize_grep_preserves_queries_order() {
    let tc = message::ToolCall {
        tool_call_id: "call-grep-2".to_owned(),
        tool: Some(message::tool_call::Tool::Grep(message::tool_call::Grep {
            queries: vec!["zzz".to_owned(), "aaa".to_owned()],
            path: ".".to_owned(),
        })),
    };
    let (_, v) = chat_stream::serialize_outgoing_tool_call_for_test(&tc, None, "");
    let s = serde_json::to_string(&v).unwrap();
    let pos_z = s.find("zzz").expect("queries should contain zzz");
    let pos_a = s.find("aaa").expect("queries should contain aaa");
    assert!(pos_z < pos_a, "Vec order must follow the input (zzz first, aaa second)");
}

/// An MCP tool call contains a `prost_types::Struct`; verify serialization is stable.
/// `prost_types::Struct.fields` uses a `BTreeMap` internally, which is stable by itself; this confirms it.
#[test]
fn serialize_mcp_tool_call_is_deterministic() {
    use prost_types::{value::Kind, Struct, Value as ProstValue};
    use std::collections::BTreeMap;

    let mut fields = BTreeMap::new();
    fields.insert(
        "key_z".to_owned(),
        ProstValue {
            kind: Some(Kind::StringValue("v_z".to_owned())),
        },
    );
    fields.insert(
        "key_a".to_owned(),
        ProstValue {
            kind: Some(Kind::NumberValue(42.0)),
        },
    );

    let server_id = "srv-uuid-1".to_owned();
    let tc = message::ToolCall {
        tool_call_id: "call-mcp-1".to_owned(),
        tool: Some(message::tool_call::Tool::CallMcpTool(
            message::tool_call::CallMcpTool {
                name: "echo".to_owned(),
                args: Some(Struct { fields }),
                server_id: server_id.clone(),
            },
        )),
    };

    // Build an mcp_context so sanitize_server_name can look up the server name
    let ctx = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![MCPServer {
            id: server_id.clone(),
            name: "my-server".to_owned(),
            description: String::new(),
            resources: vec![],
            tools: vec![],
        }],
    };

    let (n1, v1) = chat_stream::serialize_outgoing_tool_call_for_test(&tc, Some(&ctx), "");
    let (n2, v2) = chat_stream::serialize_outgoing_tool_call_for_test(&tc, Some(&ctx), "");
    assert_eq!(n1, n2);
    let j1 = serde_json::to_string(&v1).unwrap();
    let j2 = serde_json::to_string(&v2).unwrap();
    assert_eq!(j1, j2);
    // BTreeMap should emit in key lexicographic order (key_a before key_z)
    let pos_a = j1.find("key_a").expect("should contain key_a");
    let pos_z = j1.find("key_z").expect("should contain key_z");
    assert!(
        pos_a < pos_z,
        "prost_types::Struct should follow BTreeMap key lexicographic order"
    );
}

// ---------------------------------------------------------------------------
// Issue #245: carrier with invalid JSON args must fall back to empty object
// ---------------------------------------------------------------------------

/// When the model emits a tool call with invalid JSON escape sequences (e.g. `\e`, `\``),
/// the carrier message stores the raw string in server_message_data.
/// On the next turn, serialize_outgoing_tool_call must return a Value::Object (not
/// Value::String) so that genai serializes it as a JSON object for `arguments`,
/// not a doubly-wrapped JSON string that the provider would reject with "Invalid \escape".
#[test]
fn carrier_with_invalid_json_args_falls_back_to_empty_object() {
    use api::message;
    // Simulate a carrier message: tool = None, server_message_data = "fn_name\n<invalid_json>"
    let tc = message::ToolCall {
        tool_call_id: "call-invalid".to_owned(),
        tool: None,
    };
    let server_message_data = "shell\n{\"command\": \"echo \\epath\"}";

    let (fn_name, args_value) =
        chat_stream::serialize_outgoing_tool_call_for_test(&tc, None, server_message_data);

    assert_eq!(fn_name, "shell");
    // Must NOT be a String (which would cause double-wrapping and Invalid \escape on the wire)
    assert!(
        !args_value.is_string(),
        "args must be a JSON object/value, not a raw string (would cause Invalid \\escape)"
    );
    // Must be a valid JSON value that serde_json can serialize
    let serialized = serde_json::to_string(&args_value).expect("args_value must be serializable");
    // The serialized form must itself be valid JSON
    serde_json::from_str::<serde_json::Value>(&serialized)
        .expect("serialized args must be valid JSON");
}

// ---------------------------------------------------------------------------
// P1-13: build_tools_array overall stability (together with P0-3 MCP ordering)
// ---------------------------------------------------------------------------

/// End-to-end assertion: assembling the tools array twice from the same `(REGISTRY + same mcp_context)`
/// gives byte-equal strings. This covers the key stability constraint of the tools array in the prompt
/// (Anthropic docs: changing tool definitions → the whole cache is invalidated).
///
/// It does not call `build_tools_array(params: &RequestParams)` directly because `RequestParams`
/// has too many fields and is costly to construct; this replicates its core assembly logic for REGISTRY and the mcp part.
#[test]
fn full_tools_array_serialization_is_stable() {
    let assemble = || -> String {
        let mut buf = String::new();
        // Built-in tools (REGISTRY iteration order is static)
        for t in tools::REGISTRY {
            buf.push_str(t.name);
            buf.push('|');
            buf.push_str(t.description);
            buf.push('|');
            let schema = (t.parameters)();
            buf.push_str(&serde_json::to_string(&schema).unwrap());
            buf.push('\n');
        }
        // MCP tools (already sorted inside build_mcp_tool_defs; empty when there is no ctx)
        buf
    };
    let a = assemble();
    let b = assemble();
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b, "tools array serialization must be byte-equal across calls");
}

/// End-to-end assembly stability with an MCP server (relies on the P0-3 ordering guarantee).
#[test]
fn full_tools_array_with_mcp_is_stable() {
    use rmcp::model::{AnnotateAble, RawResource, Tool as McpTool};
    use serde_json::json;
    use std::sync::Arc;

    let schema_obj = json!({
        "type": "object",
        "properties": { "x": { "type": "string" } }
    })
    .as_object()
    .unwrap()
    .clone();

    let server_a = MCPServer {
        id: "id-a".to_owned(),
        name: "server-a".to_owned(),
        description: String::new(),
        resources: vec![RawResource::new("file:///x.txt", "X").no_annotation()],
        tools: vec![
            McpTool::new("zeta", "Z desc", Arc::new(schema_obj.clone())),
            McpTool::new("alpha", "A desc", Arc::new(schema_obj.clone())),
        ],
    };
    let ctx1 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![server_a.clone()],
    };
    // Rebuild the same ctx once more (servers Vec in the same order):
    let ctx2 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![server_a],
    };

    let assemble = |ctx: &MCPContext| -> String {
        let mut buf = String::new();
        for t in tools::REGISTRY {
            buf.push_str(t.name);
            buf.push('|');
            buf.push_str(t.description);
            buf.push('|');
            let schema = (t.parameters)();
            buf.push_str(&serde_json::to_string(&schema).unwrap());
            buf.push('\n');
        }
        for (name, desc, schema) in tools::mcp::build_mcp_tool_defs(ctx) {
            buf.push_str(&name);
            buf.push('|');
            buf.push_str(&desc);
            buf.push('|');
            buf.push_str(&serde_json::to_string(&schema).unwrap());
            buf.push('\n');
        }
        buf
    };

    let a = assemble(&ctx1);
    let b = assemble(&ctx2);
    assert_eq!(a, b, "tools array with MCP must be byte-equal across calls");
    // Verify MCP tools are in function_name lexicographic order (alpha before zeta)
    let pos_alpha = a.find("mcp__server-a__alpha").expect("should contain alpha");
    let pos_zeta = a.find("mcp__server-a__zeta").expect("should contain zeta");
    assert!(pos_alpha < pos_zeta, "P0-3 ordering guarantees alpha < zeta");
}
