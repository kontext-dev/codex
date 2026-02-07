use std::collections::HashMap;
use std::time::Duration;

use anyhow::anyhow;
use anyhow::Result;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::config::Config;
use crate::config::types::McpServerConfig;
use crate::config::types::McpServerTransportConfig;
use kontext_dev::KontextDevClient;

pub(crate) async fn attach_kontext_dev_mcp_server(config: &mut Config) -> Result<()> {
    let Some(settings) = config.kontext_dev.clone() else {
        debug!("Kontext-Dev not configured; skipping attachment.");
        return Ok(());
    };

    let client = KontextDevClient::new(settings.clone());
    let session = client.authenticate_mcp().await?;
    let url = client.mcp_url()?;
    let server_name = settings.server_name.clone();

    let mut http_headers = HashMap::new();
    http_headers.insert(
        "Authorization".to_string(),
        format!("Bearer {}", session.gateway_token.access_token),
    );

    let transport = McpServerTransportConfig::StreamableHttp {
        url,
        bearer_token_env_var: None,
        http_headers: Some(http_headers),
        env_http_headers: None,
    };

    let server_config = McpServerConfig {
        transport,
        enabled: true,
        required: false,
        disabled_reason: None,
        startup_timeout_sec: Some(Duration::from_secs_f64(30.0)),
        tool_timeout_sec: None,
        enabled_tools: None,
        disabled_tools: None,
        scopes: None,
    };

    let mut mcp_servers = (*config.mcp_servers).clone();
    mcp_servers.insert(server_name.clone(), server_config);
    config
        .mcp_servers
        .set(mcp_servers)
        .map_err(|err| anyhow!("failed to set Kontext-Dev MCP server config: {err}"))?;

    info!("Attached Kontext-Dev MCP server '{server_name}'.");

    if settings.open_connect_page_on_login && session.browser_auth_performed {
        match client
            .open_integration_connect_page(&session.gateway_token.access_token)
            .await
        {
            Ok(connect_url) => {
                info!("Opened Kontext integration connect page: {connect_url}");
            }
            Err(err) => {
                warn!("Failed to open Kontext integration connect page: {err:#}");
            }
        }
    }

    Ok(())
}
