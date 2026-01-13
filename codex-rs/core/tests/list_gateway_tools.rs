//! Quick test to list all available Gateway tools

use std::env;
use std::time::Duration;

use codex_rmcp_client::ElicitationAction;
use codex_rmcp_client::ElicitationResponse;
use codex_rmcp_client::OAuthCredentialsStoreMode;
use codex_rmcp_client::RmcpClient;
use futures::FutureExt;
use kontext_dev::build_mcp_url;
use kontext_dev::request_access_token;
use kontext_dev::KontextDevConfig;
use kontext_dev::DEFAULT_SCOPE;
use kontext_dev::DEFAULT_SERVER_NAME;
use mcp_types::ClientCapabilities;
use mcp_types::Implementation;
use mcp_types::InitializeRequestParams;
use serde_json::json;

fn should_skip() -> bool {
    env::var("KONTEXT_CLIENT_ID").is_err() || env::var("KONTEXT_CLIENT_SECRET").is_err()
}

fn build_kontext_config() -> Option<KontextDevConfig> {
    let client_id = env::var("KONTEXT_CLIENT_ID").ok()?;
    let client_secret = env::var("KONTEXT_CLIENT_SECRET").ok()?;

    // Try explicit URLs first, then base URL, then defaults
    let (mcp_url, token_url) = if let (Ok(mcp), Ok(token)) = (
        env::var("KONTEXT_MCP_URL"),
        env::var("KONTEXT_TOKEN_URL"),
    ) {
        (mcp, token)
    } else if let Ok(base_url) = env::var("KONTEXT_GATEWAY_URL") {
        let base = base_url.trim_end_matches('/');
        if base.ends_with("/mcp") {
            let prefix = &base[..base.len() - 4];
            (base.to_string(), format!("{}/oauth2/token", prefix))
        } else {
            (format!("{}/mcp", base), format!("{}/oauth2/token", base))
        }
    } else {
        (
            "https://gateway.kontext.dev/mcp".to_string(),
            "https://gateway.kontext.dev/oauth2/token".to_string(),
        )
    };

    Some(KontextDevConfig {
        client_id,
        client_secret,
        mcp_url,
        token_url,
        scope: DEFAULT_SCOPE.to_string(),
        server_name: DEFAULT_SERVER_NAME.to_string(),
    })
}

fn create_init_params() -> InitializeRequestParams {
    InitializeRequestParams {
        capabilities: ClientCapabilities {
            experimental: None,
            roots: None,
            sampling: None,
            elicitation: Some(json!({})),
        },
        client_info: Implementation {
            name: "tool-lister".into(),
            version: "1.0.0".into(),
            title: Some("Gateway Tool Lister".into()),
            user_agent: None,
        },
        protocol_version: mcp_types::MCP_SCHEMA_VERSION.to_string(),
    }
}

fn create_elicitation_handler() -> codex_rmcp_client::SendElicitation {
    Box::new(|_, _| {
        async {
            Ok(ElicitationResponse {
                action: ElicitationAction::Accept,
                content: Some(json!({})),
            })
        }
        .boxed()
    })
}

#[tokio::test]
async fn list_all_gateway_tools() {
    if should_skip() {
        println!("Skipping: credentials not set");
        return;
    }

    let config = build_kontext_config().expect("Config should be valid");
    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("Auth failed: {}", e);
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
            println!("Failed to create client: {}", e);
            return;
        }
    };

    if let Err(e) = client
        .initialize(create_init_params(), Some(Duration::from_secs(30)), create_elicitation_handler())
        .await
    {
        println!("Init failed: {}", e);
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
                // Navigate to content[0].resource.text which contains the JSON array
                if let Some(text) = response_val
                    .get("content")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("resource"))
                    .and_then(|r| r.get("text"))
                    .and_then(|t| t.as_str())
                {
                    if let Ok(tools) = serde_json::from_str::<Vec<serde_json::Value>>(text) {
                        println!("Found {} tools:\n", tools.len());

                        // Group by server
                        let mut by_server: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

                        for tool in &tools {
                            let server_name = tool.get("server")
                                .and_then(|s| s.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            let tool_name = tool.get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");

                            by_server.entry(server_name.to_string())
                                .or_default()
                                .push(tool_name.to_string());
                        }

                        for (server, tools) in by_server.iter() {
                            println!("## {} ({} tools)", server, tools.len());
                            for tool in tools.iter().take(30) {
                                println!("  - {}", tool);
                            }
                            if tools.len() > 30 {
                                println!("  ... and {} more", tools.len() - 30);
                            }
                            println!();
                        }
                    } else {
                        println!("Failed to parse tools array");
                    }
                } else {
                    println!("Response structure: {}", &response_str[..response_str.len().min(1000)]);
                }
            }
        }
        Err(e) => {
            println!("SEARCH_TOOLS failed: {}", e);
        }
    }
}
