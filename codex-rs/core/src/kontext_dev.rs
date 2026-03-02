use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use kontext_dev_sdk::KontextClientConfig;
use kontext_dev_sdk::KontextDevError;
use kontext_dev_sdk::build_kontext_prompt_guidance;
use kontext_dev_sdk::create_kontext_orchestrator;
use kontext_dev_sdk::mcp::KontextTool;
use kontext_dev_sdk::orchestrator::KontextOrchestrator;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use sha1::Digest;
use sha1::Sha1;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;
use url::Url;
use uuid::Uuid;

use crate::config::Config;
use crate::default_client::CODEX_INTERNAL_ORIGINATOR_OVERRIDE_ENV_VAR;

const MAX_TOOL_NAME_LENGTH: usize = 64;
const TOOL_NAME_PREFIX: &str = "kontext__";
const KONTEXT_CLIENT_ID: &str = "app_6736f70c-1c16-421e-b56f-2ae2c8c59950";
const KONTEXT_REDIRECT_URI: &str = "http://localhost:3333/callback";
const KONTEXT_SERVER_URL: &str = "https://api.kontext.dev/mcp";
const KONTEXT_RESOURCE: &str = "mcp-gateway";
const KONTEXT_SCOPE: &str = "";
const KONTEXT_SERVER_NAME: &str = "kontext-dev";
const KONTEXT_AUTH_TIMEOUT_SECONDS: i64 = 300;
const KONTEXT_INTEGRATION_UI_URL: &str = "https://app.kontext.dev";

#[derive(Clone, Debug)]
pub(crate) struct InjectedKontextToolSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: Value,
}

#[derive(Clone, Debug)]
struct DisconnectedCapability {
    name: String,
    connect_url: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ToolState {
    tool_id_by_name: HashMap<String, String>,
    disconnected_capabilities: Vec<DisconnectedCapability>,
    needs_connect_page: bool,
    prompt_guidance: Option<String>,
}

pub(crate) struct KontextDevRuntime {
    client: KontextOrchestrator,
    settings: kontext_dev_sdk::KontextDevConfig,
    state: Arc<RwLock<ToolState>>,
}

impl KontextDevRuntime {
    pub(crate) async fn list_tool_specs(&self) -> Result<Vec<InjectedKontextToolSpec>> {
        let should_force_refresh = {
            let state = self.state.read().await;
            state.needs_connect_page || !state.disconnected_capabilities.is_empty()
        };

        if should_force_refresh && let Err(err) = self.reconnect_client().await {
            warn!(
                "Unable to force-refresh Kontext session after connect-state change; retrying with existing session: {err}"
            );
        }

        let mut needs_connect_page = false;
        let tools = match self.client.tools_list().await {
            Ok(tools) => tools,
            Err(err) if is_url_elicitation_required_error(&err) => {
                warn!(
                    "Kontext gateway requires integration connect URL elicitation before tool listing. Continuing startup and showing connect guidance."
                );
                needs_connect_page = true;
                Vec::new()
            }
            Err(err) => return Err(anyhow!("failed to list Kontext tools: {err}")),
        };

        let mut seen_names = HashSet::new();
        let mut tool_specs = Vec::new();
        let mut tool_id_by_name = HashMap::new();

        for tool in tools.as_slice() {
            let (name, spec) = map_kontext_tool(tool.clone(), &mut seen_names)?;
            tool_id_by_name.insert(name.clone(), spec.id);
            tool_specs.push(InjectedKontextToolSpec {
                name,
                description: spec.description,
                input_schema: spec.input_schema,
            });
        }

        let integrations = match self.client.integrations_list().await {
            Ok(integrations) => integrations,
            Err(err) if is_url_elicitation_required_error(&err) => {
                warn!(
                    "Kontext integrations require URL elicitation before listing status. Continuing startup and showing connect guidance."
                );
                needs_connect_page = true;
                Vec::new()
            }
            Err(err) => {
                warn!("Unable to list Kontext integrations: {err}");
                Vec::new()
            }
        };

        let mut disconnected_capabilities = Vec::new();
        for integration in integrations.as_slice() {
            if integration.connected {
                continue;
            }

            disconnected_capabilities.push(DisconnectedCapability {
                name: integration.name.clone(),
                connect_url: integration.connect_url.as_ref().map(|raw_url| {
                    normalize_connect_url(
                        raw_url.as_str(),
                        self.settings.integration_ui_url.as_deref(),
                    )
                }),
            });
        }

        disconnected_capabilities.sort_by(|a, b| a.name.cmp(&b.name));

        let tool_names = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<String>>();
        let prompt_guidance =
            build_kontext_prompt_guidance(tool_names.as_slice(), integrations.as_slice())
                .system_prompt;

        tool_specs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut state = self.state.write().await;
        state.tool_id_by_name = tool_id_by_name;
        state.disconnected_capabilities = disconnected_capabilities;
        state.needs_connect_page = needs_connect_page;
        state.prompt_guidance = prompt_guidance;

        Ok(tool_specs)
    }

    async fn connect_page_url(&self) -> Result<String> {
        let connect = self.client.get_connect_page_url().await?;
        Ok(normalize_connect_url(
            connect.connect_url.as_str(),
            self.settings.integration_ui_url.as_deref(),
        ))
    }

    pub(crate) async fn execute_tool(
        &self,
        model_tool_name: &str,
        args: Map<String, Value>,
    ) -> Result<String> {
        // Keep tool routing fresh on every execution so newly connected
        // integrations and newly discovered tools are immediately available.
        if let Err(err) = self.list_tool_specs().await {
            warn!("Unable to refresh Kontext tool inventory before execute: {err}");
        }

        let mut tool_id = {
            let state = self.state.read().await;
            state.tool_id_by_name.get(model_tool_name).cloned()
        };

        if tool_id.is_none()
            && self.reconnect_client().await.is_ok()
            && self.list_tool_specs().await.is_ok()
        {
            tool_id = {
                let state = self.state.read().await;
                state.tool_id_by_name.get(model_tool_name).cloned()
            };
        }

        let Some(tool_id) = tool_id else {
            return Err(anyhow!(
                "Unknown Kontext tool `{model_tool_name}`. Run tools.list() again to refresh inventory."
            ));
        };

        let result = match self
            .client
            .tools_execute(tool_id.as_str(), Some(args.clone()))
            .await
        {
            Ok(result) => result,
            Err(err) if should_retry_tool_execution(&err) => {
                warn!(
                    "Kontext tool execution for `{tool_id}` failed with a recoverable auth/session error; reconnecting and retrying once: {err}"
                );

                self.reconnect_client().await?;
                self.list_tool_specs().await?;

                let retry_tool_id = {
                    let state = self.state.read().await;
                    state
                        .tool_id_by_name
                        .get(model_tool_name)
                        .cloned()
                        .unwrap_or(tool_id.clone())
                };

                self.client
                    .tools_execute(retry_tool_id.as_str(), Some(args))
                    .await
                    .map_err(|retry_err| {
                        anyhow!(
                            "Kontext tool execution failed for `{retry_tool_id}` after reconnect retry: {retry_err}"
                        )
                    })?
            }
            Err(err) => {
                return Err(anyhow!(
                    "Kontext tool execution failed for `{tool_id}`: {err}"
                ));
            }
        };

        Ok(result.content)
    }

    pub(crate) async fn build_system_prompt(&self) -> Option<String> {
        let state = self.state.read().await;
        build_kontext_system_prompt_from_state(&state)
    }

    pub(crate) async fn maybe_show_connect_guidance(&self) {
        let (disconnected, needs_connect_page) = {
            let state = self.state.read().await;
            (
                state.disconnected_capabilities.clone(),
                state.needs_connect_page,
            )
        };

        if disconnected.is_empty() && !needs_connect_page {
            return;
        }

        if should_auto_open_connect_page(
            self.settings.open_connect_page_on_login,
            disconnected.as_slice(),
            needs_connect_page,
        ) {
            match self.connect_page_url().await {
                Ok(connect_url) => {
                    info!(
                        "Kontext integrations are disconnected. Open this URL to connect: {connect_url}"
                    );
                    if let Err(err) = webbrowser::open(&connect_url) {
                        warn!(
                            "Failed to open Kontext connect URL in browser (showing URL instead): {err}"
                        );
                    } else {
                        info!("Opened Kontext integration connect page in browser.");
                    }
                }
                Err(err) => {
                    warn!("Unable to generate Kontext connect URL: {err}");
                }
            }
            return;
        }

        if let Some(first) = disconnected.first() {
            if let Some(connect_url) = first.connect_url.as_deref() {
                warn!(
                    "Kontext integration `{}` is disconnected. Open this URL to connect: {connect_url}",
                    first.name
                );
            } else {
                match self.connect_page_url().await {
                    Ok(connect_url) => warn!(
                        "Kontext integration `{}` is disconnected. Open this URL to connect/manage integrations: {connect_url}",
                        first.name
                    ),
                    Err(err) => warn!(
                        "Kontext integration `{}` is disconnected and generating a connect URL failed: {err}",
                        first.name
                    ),
                }
            }
        } else if needs_connect_page {
            warn!(
                "Kontext requires integration connection before tools are available. Open the connect page to finish setup."
            );
        }
    }

    async fn reconnect_client(&self) -> Result<()> {
        self.client.disconnect().await;
        self.client
            .connect()
            .await
            .map_err(|err| anyhow!("failed to reconnect Kontext session: {err}"))
    }
}

pub(crate) async fn initialize_kontext_dev_runtime(
    _config: &Config,
) -> Result<Option<Arc<KontextDevRuntime>>> {
    // Keep tests deterministic/offline: harness and nextest runs should not trigger PKCE.
    if cfg!(test)
        || std::env::var_os("RUST_TEST_THREADS").is_some()
        || std::env::var_os("NEXTEST").is_some()
        || std::env::var_os("NEXTEST_STATUS_LEVEL").is_some()
        || std::env::var_os("CARGO_TARGET_TMPDIR").is_some()
    {
        return Ok(None);
    }

    if !env!("CARGO_PKG_VERSION").contains("-kontext.") {
        return Ok(None);
    }

    if should_skip_kontext_runtime_for_originator(
        std::env::var(CODEX_INTERNAL_ORIGINATOR_OVERRIDE_ENV_VAR)
            .ok()
            .as_deref(),
    ) {
        return Ok(None);
    }

    let settings = baked_kontext_dev_settings();
    let client_session_id = Uuid::new_v4().to_string();

    let client = create_kontext_orchestrator(KontextClientConfig {
        client_session_id,
        client_id: settings.client_id.clone(),
        redirect_uri: settings.redirect_uri.clone(),
        url: None,
        server_url: Some(settings.server.clone()),
        client_secret: settings.client_secret.clone(),
        scope: Some(settings.scope.clone()),
        resource: Some(settings.resource.clone()),
        integration_ui_url: settings.integration_ui_url.clone(),
        integration_return_to: settings.integration_return_to.clone(),
        auth_timeout_seconds: Some(settings.auth_timeout_seconds),
        token_cache_path: settings.token_cache_path.clone(),
    });

    match client.connect().await {
        Ok(()) => {}
        Err(err) if is_url_elicitation_required_error(&err) => {
            warn!(
                "Kontext connect reported URL elicitation requirement during startup; continuing and exposing request-capability guidance."
            );
        }
        Err(err) => return Err(anyhow!("Kontext authentication failed: {err}")),
    }

    let runtime = Arc::new(KontextDevRuntime {
        client,
        settings,
        state: Arc::new(RwLock::new(ToolState::default())),
    });

    runtime.list_tool_specs().await?;
    runtime.maybe_show_connect_guidance().await;

    Ok(Some(runtime))
}

fn baked_kontext_dev_settings() -> kontext_dev_sdk::KontextDevConfig {
    kontext_dev_sdk::KontextDevConfig {
        server: KONTEXT_SERVER_URL.to_string(),
        client_id: KONTEXT_CLIENT_ID.to_string(),
        client_secret: None,
        scope: KONTEXT_SCOPE.to_string(),
        server_name: KONTEXT_SERVER_NAME.to_string(),
        resource: KONTEXT_RESOURCE.to_string(),
        integration_ui_url: Some(KONTEXT_INTEGRATION_UI_URL.to_string()),
        integration_return_to: None,
        open_connect_page_on_login: true,
        auth_timeout_seconds: KONTEXT_AUTH_TIMEOUT_SECONDS,
        token_cache_path: baked_token_cache_path(),
        redirect_uri: KONTEXT_REDIRECT_URI.to_string(),
    }
}

fn baked_token_cache_path() -> Option<String> {
    let mut path = dirs::home_dir()?;
    path.push(".kontext-dev");
    path.push("tokens");

    let server_slug = KONTEXT_SERVER_URL
        .replace("https://", "")
        .replace("http://", "")
        .replace(['/', ':'], "_");
    path.push(format!("{KONTEXT_CLIENT_ID}__{server_slug}.json"));

    Some(path.to_string_lossy().into_owned())
}

fn map_kontext_tool(
    tool: KontextTool,
    seen_names: &mut HashSet<String>,
) -> Result<(String, MappedKontextTool)> {
    let raw_name = format!("{TOOL_NAME_PREFIX}{}", tool.name);
    let name = unique_tool_name(raw_name.as_str(), seen_names);

    let description = tool
        .description
        .unwrap_or_else(|| format!("Execute Kontext tool `{}`.", tool.name));

    let input_schema = normalize_input_schema(tool.input_schema);

    Ok((
        name,
        MappedKontextTool {
            id: tool.id,
            description,
            input_schema,
        },
    ))
}

#[derive(Debug)]
struct MappedKontextTool {
    id: String,
    description: String,
    input_schema: Value,
}

fn normalize_input_schema(input_schema: Option<Value>) -> Value {
    let mut schema = input_schema.unwrap_or_else(|| json!({ "type": "object" }));

    if let Value::Object(map) = &mut schema {
        if map.get("type").and_then(Value::as_str).is_none() {
            map.insert("type".to_string(), Value::String("object".to_string()));
        }
        if map.get("properties").is_none_or(Value::is_null) {
            map.insert("properties".to_string(), Value::Object(Map::new()));
        }
    }

    schema
}

fn should_auto_open_connect_page(
    open_connect_page_on_login: bool,
    disconnected: &[DisconnectedCapability],
    needs_connect_page: bool,
) -> bool {
    open_connect_page_on_login && (!disconnected.is_empty() || needs_connect_page)
}

fn normalize_connect_url(raw_url: &str, integration_ui_url: Option<&str>) -> String {
    let Some(base) = integration_ui_url else {
        return raw_url.to_string();
    };

    let Ok(mut base_url) = Url::parse(base) else {
        return raw_url.to_string();
    };
    let Ok(source_url) = Url::parse(raw_url) else {
        return raw_url.to_string();
    };

    base_url.set_path(source_url.path());
    base_url.set_query(source_url.query());
    base_url.set_fragment(None);
    base_url.to_string()
}

fn is_url_elicitation_required_error(err: &KontextDevError) -> bool {
    match err {
        KontextDevError::ConnectSession { message } => {
            let message = message.to_ascii_lowercase();
            message.contains("url elicitation required")
                || message.contains("url elicitations required")
        }
        _ => false,
    }
}

fn should_retry_tool_execution(err: &KontextDevError) -> bool {
    if is_url_elicitation_required_error(err) {
        return true;
    }

    match err {
        KontextDevError::ConnectSession { message }
        | KontextDevError::IntegrationOAuthInit { message }
        | KontextDevError::TokenRequest { message, .. }
        | KontextDevError::TokenExchange { message, .. } => {
            let message = message.to_ascii_lowercase();
            message.contains("unauthorized")
                || message.contains("forbidden")
                || message.contains("denied")
                || message.contains("credentials_required")
                || message.contains("invalid session")
                || message.contains("not connected")
        }
        _ => false,
    }
}

fn should_skip_kontext_runtime_for_originator(originator_override: Option<&str>) -> bool {
    originator_override.is_some_and(|originator| originator.starts_with("codex_sdk_"))
}

fn build_kontext_system_prompt_from_state(state: &ToolState) -> Option<String> {
    state.prompt_guidance.clone()
}

fn unique_tool_name(raw: &str, seen_names: &mut HashSet<String>) -> String {
    let mut candidate = sanitize_responses_api_tool_name(raw);
    if candidate.len() > MAX_TOOL_NAME_LENGTH {
        let sha1 = sha1_hex(raw);
        let prefix_len = MAX_TOOL_NAME_LENGTH - sha1.len();
        candidate = format!("{}{}", &candidate[..prefix_len], sha1);
    }

    if seen_names.insert(candidate.clone()) {
        return candidate;
    }

    let sha1 = sha1_hex(raw);
    let max_prefix_len = MAX_TOOL_NAME_LENGTH.saturating_sub(sha1.len());
    let prefix = if candidate.len() > max_prefix_len {
        &candidate[..max_prefix_len]
    } else {
        &candidate
    };
    let deduped = format!("{prefix}{sha1}");
    seen_names.insert(deduped.clone());
    deduped
}

fn sanitize_responses_api_tool_name(name: &str) -> String {
    let mut sanitized = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            sanitized.push(c);
        } else {
            sanitized.push('_');
        }
    }

    if sanitized.is_empty() {
        "_".to_string()
    } else {
        sanitized
    }
}

fn sha1_hex(value: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn unique_tool_name_sanitizes_and_dedupes() {
        let mut seen = HashSet::new();

        let first = unique_tool_name("kontext__tool.with.dot", &mut seen);
        let second = unique_tool_name("kontext__tool_with_dot", &mut seen);

        assert_eq!(first, "kontext__tool_with_dot");
        assert_ne!(first, second);
        assert!(second.starts_with("kontext__tool_with_dot"));
    }

    #[test]
    fn normalize_input_schema_inserts_defaults() {
        let normalized = normalize_input_schema(Some(json!({ "type": "object" })));
        let properties = normalized
            .get("properties")
            .expect("properties should be present");
        assert_eq!(properties, &Value::Object(Map::new()));
    }

    #[test]
    fn should_auto_open_connect_page_when_enabled_and_disconnected_or_elicited() {
        let disconnected = vec![DisconnectedCapability {
            name: "Linear".to_string(),
            connect_url: None,
        }];

        assert!(should_auto_open_connect_page(
            true,
            disconnected.as_slice(),
            false
        ));
        assert!(should_auto_open_connect_page(true, &[], true));
        assert!(!should_auto_open_connect_page(
            false,
            disconnected.as_slice(),
            false
        ));
        assert!(!should_auto_open_connect_page(false, &[], true));
        assert!(!should_auto_open_connect_page(true, &[], false));
    }

    #[test]
    fn normalize_connect_url_uses_baked_ui_host_with_same_path_and_query() {
        let normalized = normalize_connect_url(
            "https://app.kontext.dev/oauth/connect?session=test-session",
            Some("http://localhost:3000"),
        );

        assert_eq!(
            normalized,
            "http://localhost:3000/oauth/connect?session=test-session"
        );
    }

    #[test]
    fn normalize_connect_url_falls_back_to_raw_url_when_base_is_invalid() {
        let raw = "https://app.kontext.dev/oauth/connect?session=test-session";
        let normalized = normalize_connect_url(raw, Some("not a valid url"));
        assert_eq!(normalized, raw);
    }

    #[test]
    fn url_elicitation_required_error_detection_is_specific() {
        let required_error = KontextDevError::ConnectSession {
            message: "MCP error -32042: URL elicitations required".to_string(),
        };
        let other_connect_error = KontextDevError::ConnectSession {
            message: "MCP request failed: invalid session".to_string(),
        };
        let other_error = KontextDevError::TokenExchange {
            resource: "mcp-gateway".to_string(),
            message: "exchange failed".to_string(),
        };

        assert!(is_url_elicitation_required_error(&required_error));
        assert!(!is_url_elicitation_required_error(&other_connect_error));
        assert!(!is_url_elicitation_required_error(&other_error));
    }

    #[test]
    fn should_retry_tool_execution_for_auth_and_session_errors() {
        let denied = KontextDevError::ConnectSession {
            message: "unauthorized: tool invocation denied".to_string(),
        };
        let invalid_session = KontextDevError::ConnectSession {
            message: "MCP request failed: invalid session".to_string(),
        };
        let non_retryable = KontextDevError::ConnectSession {
            message: "MCP request failed: input schema invalid".to_string(),
        };

        assert!(should_retry_tool_execution(&denied));
        assert!(should_retry_tool_execution(&invalid_session));
        assert!(!should_retry_tool_execution(&non_retryable));
    }

    #[test]
    fn baked_settings_match_fork_defaults() {
        let settings = baked_kontext_dev_settings();

        assert_eq!(settings.client_id, KONTEXT_CLIENT_ID);
        assert_eq!(settings.redirect_uri, KONTEXT_REDIRECT_URI);
        assert_eq!(settings.server, KONTEXT_SERVER_URL);
        assert_eq!(settings.resource, KONTEXT_RESOURCE);
        assert_eq!(
            settings.integration_ui_url.as_deref(),
            Some(KONTEXT_INTEGRATION_UI_URL)
        );
        let server_slug = KONTEXT_SERVER_URL
            .replace("https://", "")
            .replace("http://", "")
            .replace(['/', ':'], "_");
        let expected_token_file = format!("{KONTEXT_CLIENT_ID}__{server_slug}.json");
        assert!(
            settings
                .token_cache_path
                .as_deref()
                .is_some_and(|path| { path.contains(expected_token_file.as_str()) })
        );
        assert!(settings.open_connect_page_on_login);
    }

    #[tokio::test]
    async fn initialize_runtime_skips_under_test_harness() {
        let config = crate::config::test_config();
        let runtime = initialize_kontext_dev_runtime(&config)
            .await
            .expect("runtime init should not fail in tests");
        assert!(runtime.is_none());
    }

    #[test]
    fn sdk_originator_override_skips_runtime_init() {
        assert!(should_skip_kontext_runtime_for_originator(Some(
            "codex_sdk_ts"
        )));
        assert!(should_skip_kontext_runtime_for_originator(Some(
            "codex_sdk_python"
        )));
        assert!(!should_skip_kontext_runtime_for_originator(Some(
            "codex_exec"
        )));
        assert!(!should_skip_kontext_runtime_for_originator(Some(
            "codex-kontext-cli"
        )));
        assert!(!should_skip_kontext_runtime_for_originator(None));
    }

    #[test]
    fn build_system_prompt_returns_none_without_tools_or_integrations() {
        let state = ToolState::default();
        assert_eq!(build_kontext_system_prompt_from_state(&state), None);
    }

    #[test]
    fn build_system_prompt_returns_cached_sdk_prompt_text() {
        let state = ToolState {
            prompt_guidance: Some(
                "Always call `REQUEST_CAPABILITY` for fresh integration links.".to_string(),
            ),
            tool_id_by_name: HashMap::new(),
            disconnected_capabilities: Vec::new(),
            needs_connect_page: false,
        };

        let prompt = build_kontext_system_prompt_from_state(&state)
            .expect("prompt should be present with integration context");
        assert_eq!(
            prompt,
            "Always call `REQUEST_CAPABILITY` for fresh integration links."
        );
    }
}
