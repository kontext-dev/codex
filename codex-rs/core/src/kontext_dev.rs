use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use tracing::debug;
use tracing::info;

use crate::config::Config;
use crate::config::types::McpServerConfig;
use crate::config::types::McpServerTransportConfig;
use kontext_dev::build_mcp_url;
use kontext_dev::request_access_token;

const DEFAULT_TOKEN_TTL_SECONDS: i64 = 3600;

static KONTEXT_DEV_TOKEN_EXPIRES_AT: OnceLock<Instant> = OnceLock::new();
static KONTEXT_DEV_SERVER_NAME: OnceLock<String> = OnceLock::new();

pub(crate) async fn attach_kontext_dev_mcp_server(config: &mut Config) -> Result<()> {
    let Some(settings) = config.kontext_dev.clone() else {
        debug!("Kontext-Dev not configured; skipping attachment.");
        return Ok(());
    };

    let token = request_access_token(&settings).await?;
    let url = build_mcp_url(&settings, token.access_token.as_str())?;
    let server_name = settings.server_name.clone();

    let transport = McpServerTransportConfig::StreamableHttp {
        url,
        bearer_token_env_var: None,
        http_headers: None,
        env_http_headers: None,
    };

    let server_config = McpServerConfig {
        transport,
        startup_timeout_sec: Some(Duration::from_secs_f64(30.0)),
        tool_timeout_sec: None,
        enabled: true,
        enabled_tools: None,
        disabled_tools: None,
    };

    config
        .mcp_servers
        .insert(server_name.clone(), server_config);

    let expires_in = token.expires_in.unwrap_or(DEFAULT_TOKEN_TTL_SECONDS);
    let expires_in = expires_in.max(0);
    let expires_at = Instant::now() + Duration::from_secs_f64(expires_in as f64);
    let _ = KONTEXT_DEV_TOKEN_EXPIRES_AT.set(expires_at);
    let _ = KONTEXT_DEV_SERVER_NAME.set(server_name.clone());

    info!("Attached Kontext-Dev MCP server '{server_name}'.");
    Ok(())
}

pub(crate) fn kontext_dev_server_name() -> Option<&'static str> {
    KONTEXT_DEV_SERVER_NAME.get().map(String::as_str)
}

pub(crate) fn kontext_dev_token_expired() -> bool {
    KONTEXT_DEV_TOKEN_EXPIRES_AT
        .get()
        .map(|expires_at| Instant::now() >= *expires_at)
        .unwrap_or(false)
}
