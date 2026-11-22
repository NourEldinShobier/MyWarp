//! Unit tests for `mcp.rs`.
//!
//! Covers the P0-3 prompt cache optimization: `build_mcp_tool_defs` must be **stable in lexicographic order**,
//! so that calling it several times across requests with the same `MCPContext` yields a byte-equal tools list; otherwise
//! Anthropic treats the tools field as changed and invalidates all cache layers.
//!
//! Note: `rmcp::model::Tool` and `rmcp::model::Resource` (= `Annotated<RawResource>`)
//! come from an upstream vendored crate; only their public construction paths (`Tool::new` / `RawResource::new`) are used here.

use rmcp::model::{AnnotateAble, RawResource, Tool};
use serde_json::json;
use std::sync::Arc;

use crate::ai::agent::{MCPContext, MCPServer};

use super::{build_mcp_tool_defs, function_name};

/// Builds an `rmcp::model::Tool` with a minimal input schema.
fn mk_tool(name: &'static str, desc: &'static str) -> Tool {
    let schema: serde_json::Map<String, serde_json::Value> = json!({
        "type": "object",
        "properties": {
            "x": { "type": "string" }
        }
    })
    .as_object()
    .unwrap()
    .clone();
        // `Tool::new` accepts Arc<JsonObject>; pass the Map directly (it implements Into<Arc<JsonObject>>).
    Tool::new(name, desc, Arc::new(schema))
}

/// Builds an MCPServer. The order of tools and resources is kept exactly as passed (simulating the shuffled input
/// that upstream may pass when iterating a HashMap).
fn mk_server(
    id: &str,
    name: &str,
    tools: Vec<Tool>,
    resources: Vec<rmcp::model::Resource>,
) -> MCPServer {
    MCPServer {
        id: id.to_owned(),
        name: name.to_owned(),
        description: String::new(),
        resources,
        tools,
    }
}

fn mk_resource(uri: &str, name: &str) -> rmcp::model::Resource {
        // RawResource -> Annotated<RawResource> (without annotation).
        // The safe conversion entry point provided upstream is `AnnotateAble::no_annotation`.
    RawResource::new(uri, name).no_annotation()
}

/// Building twice with the same ctx must produce byte-equal (name, description, schema) triples.
/// This is the minimum bar for prompt cache hits - if unstable, the entire Anthropic cache is invalidated.
#[test]
fn build_mcp_tool_defs_is_stable_across_calls() {
    let ctx = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![
            mk_server(
                "id-b",
                "server-b",
                vec![mk_tool("zeta", "z"), mk_tool("alpha", "a")],
                vec![],
            ),
            mk_server(
                "id-a",
                "server-a",
                vec![mk_tool("beta", "b"), mk_tool("gamma", "g")],
                vec![],
            ),
        ],
    };
    let r1 = build_mcp_tool_defs(&ctx);
    let r2 = build_mcp_tool_defs(&ctx);
        assert_eq!(r1, r2, "build_mcp_tool_defs must produce deterministic output");
}

/// When servers / tools are given out of order, the output is sorted lexicographically by function_name.
/// This is the core assertion of P0-3: if the upstream ctx.servers order differs across requests (caused by
/// HashMap iteration etc.), the output is still byte-equal.
#[test]
fn build_mcp_tool_defs_outputs_lexicographic_order() {
    let ctx = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![
            mk_server(
                "id-b",
                "server-b",
                // out of order: zeta before alpha
                vec![mk_tool("zeta", "z"), mk_tool("alpha", "a")],
                vec![],
            ),
            mk_server(
                "id-a",
                "server-a",
                vec![mk_tool("beta", "b"), mk_tool("gamma", "g")],
                vec![],
            ),
        ],
    };
    let out = build_mcp_tool_defs(&ctx);
    let names: Vec<&str> = out.iter().map(|(n, _, _)| n.as_str()).collect();
    // after sorting by function_name: server-a/beta < server-a/gamma < server-b/alpha < server-b/zeta
    let expected = [
        function_name(&mk_server("id-a", "server-a", vec![], vec![]), "beta"),
        function_name(&mk_server("id-a", "server-a", vec![], vec![]), "gamma"),
        function_name(&mk_server("id-b", "server-b", vec![], vec![]), "alpha"),
        function_name(&mk_server("id-b", "server-b", vec![], vec![]), "zeta"),
    ];
    assert_eq!(
        names,
        expected.iter().map(|s| s.as_str()).collect::<Vec<_>>()
    );
}

/// A different input server order across requests (simulating a HashMap reshuffle) still yields byte-equal output.
#[test]
fn build_mcp_tool_defs_invariant_under_servers_permutation() {
    let server_a = mk_server(
        "id-a",
        "server-a",
        vec![mk_tool("beta", "b"), mk_tool("gamma", "g")],
        vec![],
    );
    let server_b = mk_server(
        "id-b",
        "server-b",
        vec![mk_tool("zeta", "z"), mk_tool("alpha", "a")],
        vec![],
    );
    let ctx1 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![server_a.clone(), server_b.clone()],
    };
    let ctx2 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![server_b, server_a],
    };
    assert_eq!(build_mcp_tool_defs(&ctx1), build_mcp_tool_defs(&ctx2));
}

/// When any server exposes resources, the available_uris in the read_resource description
/// must also be lexicographically stable, and read_resource is always last in the array.
#[test]
fn read_resource_description_is_stable_and_sorted() {
    let ctx1 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![mk_server(
            "id-a",
            "srv",
            vec![mk_tool("t", "")],
            vec![
                mk_resource("file:///z.txt", "Z"),
                mk_resource("file:///a.txt", "A"),
            ],
        )],
    };
    // same ctx but with the resource order changed
    let ctx2 = MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![mk_server(
            "id-a",
            "srv",
            vec![mk_tool("t", "")],
            vec![
                mk_resource("file:///a.txt", "A"),
                mk_resource("file:///z.txt", "Z"),
            ],
        )],
    };
    let r1 = build_mcp_tool_defs(&ctx1);
    let r2 = build_mcp_tool_defs(&ctx2);
        assert_eq!(r1, r2, "read_resource description must be byte-equal");

        let last = r1.last().expect("should contain at least read_resource");
    assert_eq!(last.0, "mcp_read_resource");
    // after sorting, a.txt comes before z.txt
        let pos_a = last.1.find("a.txt").expect("should contain a.txt");
        let pos_z = last.1.find("z.txt").expect("should contain z.txt");
        assert!(pos_a < pos_z, "available_uris must be sorted lexicographically");
}
