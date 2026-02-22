#![cfg(feature = "benchmarking")]
//! Quick test to list all available Gateway tools

use std::time::Duration;

use codex_rmcp_client::OAuthCredentialsStoreMode;
use codex_rmcp_client::RmcpClient;
use core_test_support::gateway_auth;
use kontext_dev::build_mcp_url;
use serde_json::json;

fn init_test_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

#[tokio::test]
async fn list_all_gateway_tools() {
    init_test_tracing();
    if gateway_auth::should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
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
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Failed to create client (gateway should be reachable): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("tool-lister"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("Init failed (gateway should be reachable): {e}");
    }

    // Search for ALL tools with empty query
    tracing::info!("Available Gateway Tools");

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
                        tracing::error!("Gateway search errors: {errors:?}");
                    }

                    if !tools.is_empty() || tools_payload.get("items").is_some() {
                        tracing::info!("Found {} tools", tools.len());

                        assert!(!tools.is_empty(), "Should discover at least one tool");

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
                            tracing::debug!("{} ({} tools)", server, tools.len());
                            for tool in tools.iter().take(30) {
                                tracing::debug!("- {tool}");
                            }
                            if tools.len() > 30 {
                                tracing::debug!("... and {} more", tools.len() - 30);
                            }
                        }
                    } else {
                        tracing::error!("Failed to parse tools payload: {text}");
                    }
                } else {
                    tracing::trace!(
                        "Response structure: {}",
                        &response_str[..response_str.len().min(1000)]
                    );
                }
            }
        }
        Err(e) => {
            tracing::error!("SEARCH_TOOLS failed: {e}");
        }
    }
}
