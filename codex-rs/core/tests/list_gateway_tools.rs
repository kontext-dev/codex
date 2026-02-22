//! Quick test to list all available Gateway tools

use std::time::Duration;

use codex_rmcp_client::OAuthCredentialsStoreMode;
use codex_rmcp_client::RmcpClient;
use core_test_support::gateway_auth;
use kontext_dev::build_mcp_url;
use serde_json::json;

#[tokio::test]
async fn list_all_gateway_tools() {
    if gateway_auth::should_skip() {
        println!("Skipping: credentials not set");
        return;
    }

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("Auth failed: {e}");
            return;
        }
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");

    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            println!("Failed to create client: {e}");
            return;
        }
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("tool-lister"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        println!("Init failed: {e}");
        return;
    }

    // Search for ALL tools with empty query
    println!("\n# Available Gateway Tools\n");

    let result = client
        .call_tool(
            "SEARCH_TOOLS".to_string(),
            Some(json!({"query": ""})),
            Some(Duration::from_secs(60)),
        )
        .await;

    match result {
        Ok(r) => {
            // Serialize the response and parse the embedded JSON
            let response_str = serde_json::to_string(&r).unwrap_or_default();

            // Extract the tools array from the response
            if let Ok(response_val) = serde_json::from_str::<serde_json::Value>(&response_str) {
                // Navigate to content[0].resource.text. Gateways may return either a raw
                // array (`[...]`) or an envelope (`{"items":[...],"errors":[...]}`).
                if let Some(text) = response_val
                    .get("content")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("resource"))
                    .and_then(|r| r.get("text"))
                    .and_then(|t| t.as_str())
                {
                    let tools_payload = serde_json::from_str::<serde_json::Value>(text)
                        .unwrap_or_else(|_| serde_json::Value::Array(vec![]));
                    let tools = if let Some(items) = tools_payload.as_array() {
                        items.clone()
                    } else {
                        tools_payload
                            .get("items")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default()
                    };

                    if let Some(errors) = tools_payload.get("errors").and_then(|v| v.as_array())
                        && !errors.is_empty()
                    {
                        println!("Gateway search errors: {errors:?}");
                    }

                    if !tools.is_empty() || tools_payload.get("items").is_some() {
                        println!("Found {} tools:\n", tools.len());

                        // Group by server
                        let mut by_server: std::collections::HashMap<String, Vec<String>> =
                            std::collections::HashMap::new();

                        for tool in &tools {
                            let server_name = tool
                                .get("server")
                                .and_then(|s| s.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            let tool_name = tool
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");

                            by_server
                                .entry(server_name.to_string())
                                .or_default()
                                .push(tool_name.to_string());
                        }

                        for (server, tools) in by_server.iter() {
                            println!("## {} ({} tools)", server, tools.len());
                            for tool in tools.iter().take(30) {
                                println!("  - {tool}");
                            }
                            if tools.len() > 30 {
                                println!("  ... and {} more", tools.len() - 30);
                            }
                            println!();
                        }
                    } else {
                        println!("Failed to parse tools payload: {text}");
                    }
                } else {
                    println!(
                        "Response structure: {}",
                        &response_str[..response_str.len().min(1000)]
                    );
                }
            }
        }
        Err(e) => {
            println!("SEARCH_TOOLS failed: {e}");
        }
    }
}
