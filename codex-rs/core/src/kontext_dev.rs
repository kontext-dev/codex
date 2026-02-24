use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use kontext_dev_sdk::KontextClientConfig;
use kontext_dev_sdk::create_kontext_orchestrator;
use kontext_dev_sdk::mcp::KontextTool;
use kontext_dev_sdk::orchestrator::KontextOrchestrator;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use sha1::Digest;
use sha1::Sha1;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::config::Config;

const MAX_TOOL_NAME_LENGTH: usize = 64;
const TOOL_NAME_PREFIX: &str = "kontext__";
const REQUEST_CAPABILITY_TOOL_NAME: &str = "kontext__request_capability";

#[derive(Clone, Debug)]
pub(crate) struct InjectedKontextToolSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: Value,
}

#[derive(Clone, Debug)]
struct DisconnectedCapability {
    id: String,
    name: String,
    connect_url: String,
}

#[derive(Clone, Debug, Default)]
struct ToolState {
    tool_id_by_name: HashMap<String, String>,
    disconnected_capabilities: Vec<DisconnectedCapability>,
}

pub(crate) struct KontextDevRuntime {
    client: KontextOrchestrator,
    settings: kontext_dev_sdk::KontextDevConfig,
    state: Arc<RwLock<ToolState>>,
}

impl KontextDevRuntime {
    pub(crate) async fn list_tool_specs(&self) -> Result<Vec<InjectedKontextToolSpec>> {
        let tools = self
            .client
            .tools_list()
            .await
            .map_err(|err| anyhow!("failed to list Kontext tools: {err}"))?;

        let mut seen_names = HashSet::new();
        let mut tool_specs = Vec::new();
        let mut tool_id_by_name = HashMap::new();

        for tool in tools {
            let (name, spec) = map_kontext_tool(tool, &mut seen_names)?;
            tool_id_by_name.insert(name.clone(), spec.id);
            tool_specs.push(InjectedKontextToolSpec {
                name,
                description: spec.description,
                input_schema: spec.input_schema,
            });
        }

        let disconnected_capabilities = match self.client.integrations_list().await {
            Ok(integrations) => integrations
                .into_iter()
                .filter_map(|integration| {
                    if integration.connected {
                        return None;
                    }
                    integration
                        .connect_url
                        .map(|connect_url| DisconnectedCapability {
                            id: integration.id,
                            name: integration.name,
                            connect_url,
                        })
                })
                .collect::<Vec<_>>(),
            Err(err) => {
                warn!("Unable to list Kontext integrations: {err}");
                Vec::new()
            }
        };

        if !disconnected_capabilities.is_empty() {
            tool_specs.push(request_capability_tool_spec(&disconnected_capabilities));
        }

        tool_specs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut state = self.state.write().await;
        state.tool_id_by_name = tool_id_by_name;
        state.disconnected_capabilities = disconnected_capabilities;

        Ok(tool_specs)
    }

    pub(crate) async fn execute_tool(
        &self,
        model_tool_name: &str,
        args: Map<String, Value>,
    ) -> Result<String> {
        if model_tool_name == REQUEST_CAPABILITY_TOOL_NAME {
            return self.execute_request_capability(args).await;
        }

        // Keep tool routing fresh on every execution so newly connected
        // integrations and newly discovered tools are immediately available.
        if let Err(err) = self.list_tool_specs().await {
            warn!("Unable to refresh Kontext tool inventory before execute: {err}");
        }

        let tool_id = {
            let state = self.state.read().await;
            state.tool_id_by_name.get(model_tool_name).cloned()
        };

        let Some(tool_id) = tool_id else {
            return Err(anyhow!(
                "Unknown Kontext tool `{model_tool_name}`. Run tools.list() again to refresh inventory."
            ));
        };

        let result = self
            .client
            .tools_execute(tool_id.as_str(), Some(args))
            .await
            .map_err(|err| anyhow!("Kontext tool execution failed for `{tool_id}`: {err}"))?;

        Ok(result.content)
    }

    pub(crate) async fn maybe_show_connect_guidance(&self, config: &mut Config) {
        let disconnected = {
            let state = self.state.read().await;
            state.disconnected_capabilities.clone()
        };

        if disconnected.is_empty() {
            return;
        }

        if should_auto_open_connect_page(
            self.settings.open_connect_page_on_login,
            disconnected.as_slice(),
        ) {
            match self.client.get_connect_page_url().await {
                Ok(connect) => {
                    info!(
                        "Kontext integrations are disconnected. Open this URL to connect: {}",
                        connect.connect_url
                    );
                    if let Err(err) = webbrowser::open(&connect.connect_url) {
                        warn!(
                            "Failed to open Kontext connect URL in browser (showing URL instead): {err}"
                        );
                    } else {
                        info!("Opened Kontext integration connect page in browser.");
                    }
                    config.startup_warnings.push(format!(
                        "Kontext integrations require connection. Open this URL to finish setup: {}",
                        connect.connect_url
                    ));
                }
                Err(err) => {
                    warn!("Unable to generate Kontext connect URL: {err}");
                }
            }
            return;
        }

        if let Some(first) = disconnected.first() {
            config.startup_warnings.push(format!(
                "Kontext integration `{}` is disconnected. Open this URL to connect: {}",
                first.name, first.connect_url
            ));
        }
    }

    async fn execute_request_capability(&self, args: Map<String, Value>) -> Result<String> {
        let requested = args
            .get("capability_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();

        let disconnected = {
            let state = self.state.read().await;
            state.disconnected_capabilities.clone()
        };

        if disconnected.is_empty() {
            return Ok("All Kontext integrations are already connected.".to_string());
        }

        if requested.is_empty()
            && disconnected.len() == 1
            && let Some(capability) = disconnected.first()
        {
            return Ok(format!(
                "{} requires authorization. Connect it here: {}",
                capability.name, capability.connect_url
            ));
        }

        let requested_lower = requested.to_ascii_lowercase();
        if let Some(capability) = disconnected.iter().find(|capability| {
            capability.name.to_ascii_lowercase() == requested_lower
                || capability.id.to_ascii_lowercase() == requested_lower
        }) {
            return Ok(format!(
                "{} requires authorization. Connect it here: {}",
                capability.name, capability.connect_url
            ));
        }

        let available = disconnected
            .iter()
            .map(|capability| capability.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        if requested.is_empty() {
            Ok(format!(
                "Please specify `capability_name`. Disconnected capabilities: {available}."
            ))
        } else {
            Ok(format!(
                "Capability `{requested}` is not disconnected or not recognized. Disconnected capabilities: {available}."
            ))
        }
    }
}

pub(crate) async fn initialize_kontext_dev_runtime(
    config: &mut Config,
) -> Result<Option<Arc<KontextDevRuntime>>> {
    let Some(settings) = config.kontext_dev.clone() else {
        debug!("Kontext-Dev not configured; skipping runtime initialization.");
        return Ok(None);
    };

    let client = create_kontext_orchestrator(KontextClientConfig {
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

    client
        .connect()
        .await
        .map_err(|err| anyhow!("Kontext authentication failed: {err}"))?;

    let runtime = Arc::new(KontextDevRuntime {
        client,
        settings,
        state: Arc::new(RwLock::new(ToolState::default())),
    });

    runtime.list_tool_specs().await?;
    runtime.maybe_show_connect_guidance(config).await;

    Ok(Some(runtime))
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

fn request_capability_tool_spec(
    disconnected: &[DisconnectedCapability],
) -> InjectedKontextToolSpec {
    let names = disconnected
        .iter()
        .map(|capability| capability.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let description = format!(
        "Request authorization for a disconnected Kontext integration. Disconnected integrations: {names}."
    );

    InjectedKontextToolSpec {
        name: REQUEST_CAPABILITY_TOOL_NAME.to_string(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {
                "capability_name": {
                    "type": "string",
                    "description": "Integration name to connect, for example `Linear`"
                }
            },
            "required": ["capability_name"]
        }),
    }
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
) -> bool {
    open_connect_page_on_login && !disconnected.is_empty()
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
    fn should_auto_open_connect_page_only_when_disconnected_and_enabled() {
        let disconnected = vec![DisconnectedCapability {
            id: "linear".to_string(),
            name: "Linear".to_string(),
            connect_url: "https://app.kontext.dev/oauth/connect?session=123".to_string(),
        }];

        assert!(should_auto_open_connect_page(true, disconnected.as_slice()));
        assert!(!should_auto_open_connect_page(
            false,
            disconnected.as_slice()
        ));
        assert!(!should_auto_open_connect_page(true, &[]));
    }
}
