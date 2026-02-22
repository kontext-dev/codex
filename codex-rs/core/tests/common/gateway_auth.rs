use std::env;

use anyhow::Result;
use codex_rmcp_client::ElicitationAction;
use codex_rmcp_client::ElicitationResponse;
use codex_rmcp_client::SendElicitation;
use futures::FutureExt;
use kontext_dev::AccessToken;
use kontext_dev::DEFAULT_SERVER_NAME;
use kontext_dev::KontextDevConfig;
use kontext_dev::request_access_token;
use rmcp::model::ClientCapabilities;
use rmcp::model::ElicitationCapability;
use rmcp::model::FormElicitationCapability;
use rmcp::model::Implementation;
use rmcp::model::InitializeRequestParams;
use rmcp::model::ProtocolVersion;
use serde_json::json;

pub fn should_skip() -> bool {
    env::var("KONTEXT_CLIENT_ID").is_err() || env::var("KONTEXT_CLIENT_SECRET").is_err()
}

pub fn build_kontext_config() -> Option<KontextDevConfig> {
    let client_id = env::var("KONTEXT_CLIENT_ID").ok()?;
    let client_secret = env::var("KONTEXT_CLIENT_SECRET").ok()?;

    let (mcp_url, token_url) = if let (Ok(mcp), Ok(token)) =
        (env::var("KONTEXT_MCP_URL"), env::var("KONTEXT_TOKEN_URL"))
    {
        (mcp, token)
    } else if let Ok(base_url) = env::var("KONTEXT_GATEWAY_URL") {
        let base = base_url.trim_end_matches('/');
        if let Some(prefix) = base.strip_suffix("/mcp") {
            (base.to_string(), format!("{prefix}/oauth2/token"))
        } else {
            (format!("{base}/mcp"), format!("{base}/oauth2/token"))
        }
    } else {
        (
            "http://localhost:4000/mcp".to_string(),
            "http://localhost:4000/oauth2/token".to_string(),
        )
    };

    Some(KontextDevConfig {
        client_id,
        client_secret: Some(client_secret),
        mcp_url: Some(mcp_url),
        token_url: Some(token_url),
        scope: "mcp:invoke".to_string(),
        server_name: DEFAULT_SERVER_NAME.to_string(),
        server: None,
        resource: kontext_dev::DEFAULT_RESOURCE.to_string(),
        integration_ui_url: None,
        integration_return_to: None,
        open_connect_page_on_login: false,
        auth_timeout_seconds: kontext_dev::DEFAULT_AUTH_TIMEOUT_SECONDS,
        token_cache_path: None,
        redirect_uri: "http://localhost:3333/callback".to_string(),
    })
}

/// Exchanges a Hydra access token for an MCP-scoped resource token via RFC 8693.
pub async fn exchange_token_for_mcp(
    config: &KontextDevConfig,
    hydra_token: &str,
) -> Option<AccessToken> {
    let token_url = config.token_url.as_deref()?;
    let mcp_url = config.mcp_url.as_deref()?;
    let client = reqwest::Client::new();
    client
        .post(token_url)
        .basic_auth(
            &config.client_id,
            config.client_secret.as_deref(),
        )
        .form(&[
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            ("subject_token", hydra_token),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:access_token",
            ),
            ("resource", mcp_url),
            ("scope", config.scope.as_str()),
        ])
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json::<AccessToken>()
        .await
        .ok()
}

pub async fn authenticate(config: &KontextDevConfig) -> Result<AccessToken> {
    let hydra_token = request_access_token(config).await?;
    Ok(exchange_token_for_mcp(config, &hydra_token.access_token)
        .await
        .unwrap_or(hydra_token))
}

pub fn create_init_params(client_name: &str) -> InitializeRequestParams {
    InitializeRequestParams {
        meta: None,
        capabilities: ClientCapabilities {
            experimental: None,
            extensions: None,
            roots: None,
            sampling: None,
            elicitation: Some(ElicitationCapability {
                form: Some(FormElicitationCapability {
                    schema_validation: None,
                }),
                url: None,
            }),
            tasks: None,
        },
        client_info: Implementation {
            name: client_name.into(),
            version: "1.0.0".into(),
            title: None,
            description: None,
            icons: None,
            website_url: None,
        },
        protocol_version: ProtocolVersion::V_2025_06_18,
    }
}

pub fn create_elicitation_handler() -> SendElicitation {
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
