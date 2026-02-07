use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use serde::Deserialize;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::config::Config;
use crate::config::types::McpServerConfig;
use crate::config::types::McpServerTransportConfig;
use kontext_dev::KontextDevClient;
use kontext_dev::resolve_server_base_url;

#[derive(Debug, Deserialize)]
struct KontextIntegrationsResponse {
    #[serde(default)]
    items: Vec<KontextIntegrationItem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KontextIntegrationItem {
    #[serde(default)]
    requires_oauth: bool,
    #[serde(default)]
    connection: Option<KontextIntegrationConnection>,
}

#[derive(Debug, Deserialize)]
struct KontextIntegrationConnection {
    #[serde(default)]
    connected: bool,
}

fn has_pending_oauth_integrations(response: &KontextIntegrationsResponse) -> bool {
    response.items.iter().any(|integration| {
        integration.requires_oauth
            && integration
                .connection
                .as_ref()
                .map(|connection| !connection.connected)
                .unwrap_or(true)
    })
}

async fn fetch_pending_oauth_integrations(
    client: &KontextDevClient,
    gateway_access_token: &str,
) -> Result<bool> {
    let server_base = resolve_server_base_url(client.config())?;
    let url = format!("{}/mcp/integrations", server_base.trim_end_matches('/'));

    let response = reqwest::Client::new()
        .get(url.clone())
        .bearer_auth(gateway_access_token)
        .send()
        .await
        .map_err(|err| anyhow!("Kontext integration list request failed for {url}: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Kontext integration list request returned {status} at {url}: {body}"
        ));
    }

    let payload: KontextIntegrationsResponse = response
        .json()
        .await
        .map_err(|err| anyhow!("failed to decode Kontext integrations response at {url}: {err}"))?;

    Ok(has_pending_oauth_integrations(&payload))
}

#[cfg(test)]
mod tests {
    use super::KontextIntegrationsResponse;
    use super::has_pending_oauth_integrations;

    #[test]
    fn detects_pending_oauth_integrations() {
        let payload = serde_json::from_str::<KontextIntegrationsResponse>(
            r#"{
              "items": [
                {
                  "id": "a",
                  "requiresOauth": true,
                  "connection": { "connected": false }
                },
                {
                  "id": "b",
                  "requiresOauth": false,
                  "connection": { "connected": false }
                }
              ]
            }"#,
        )
        .expect("valid integrations payload");

        assert!(has_pending_oauth_integrations(&payload));
    }

    #[test]
    fn ignores_connected_or_non_oauth_integrations() {
        let payload = serde_json::from_str::<KontextIntegrationsResponse>(
            r#"{
              "items": [
                {
                  "id": "a",
                  "requiresOauth": true,
                  "connection": { "connected": true }
                },
                {
                  "id": "b",
                  "requiresOauth": false
                }
              ]
            }"#,
        )
        .expect("valid integrations payload");

        assert!(!has_pending_oauth_integrations(&payload));
    }
}

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

    let pending_integrations = match fetch_pending_oauth_integrations(
        &client,
        &session.gateway_token.access_token,
    )
    .await
    {
        Ok(pending) => pending,
        Err(err) => {
            warn!("Unable to determine Kontext integration connection state: {err:#}");
            false
        }
    };

    let should_open_connect_page = settings.open_connect_page_on_login
        && (session.browser_auth_performed || pending_integrations);

    if should_open_connect_page {
        match client
            .open_integration_connect_page(&session.gateway_token.access_token)
            .await
        {
            Ok(connect_url) => {
                info!("Opened Kontext integration connect page: {connect_url}");
            }
            Err(err) => {
                warn!("Failed to open Kontext integration connect page: {err:#}");
                if let Ok(connect_url) = client
                    .create_integration_connect_url(&session.gateway_token.access_token)
                    .await
                {
                    config.startup_warnings.push(format!(
                        "Kontext integration setup is pending. Open this URL to finish connecting integrations: {connect_url}"
                    ));
                }
            }
        }
    } else if pending_integrations {
        if let Ok(connect_url) = client
            .create_integration_connect_url(&session.gateway_token.access_token)
            .await
        {
            config.startup_warnings.push(format!(
                "Kontext integrations require OAuth connection. Open this URL to connect: {connect_url}"
            ));
        }
    }

    Ok(())
}
