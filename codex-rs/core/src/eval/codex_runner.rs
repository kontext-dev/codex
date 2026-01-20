//! Codex Agent Task Runner
//!
//! Uses the real Codex agent via `ConversationManager` for MCP-Atlas evaluation.
//! This mode runs tasks with the full Codex system prompts and agent loop.

use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use mcp_types::ContentBlock;
use tempfile::TempDir;

use crate::AuthManager;
use crate::CodexAuth;
use crate::ConversationManager;
use crate::config::ConfigOverrides;
use crate::config::Constrained;
use crate::config::types::McpServerConfig;
use crate::config::types::McpServerTransportConfig;
use crate::protocol::AskForApproval;
use crate::protocol::EventMsg;
use crate::protocol::Op;
use crate::protocol::SandboxPolicy;
use crate::protocol::SessionSource;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::user_input::UserInput;

use super::ExecutionMode;
use super::McpAtlasTask;
use super::TaskResult;
use super::ToolCallRecord;

/// Task runner that uses the real Codex agent via ConversationManager
pub struct CodexTaskRunner {
    /// Gateway URL for MCP server
    gateway_url: String,
    /// Access token for authentication
    access_token: String,
    /// Model to use for the agent
    model: String,
    /// Maximum turns before stopping
    max_turns: usize,
    /// Temporary home directory for Codex
    _temp_home: TempDir,
    /// Path to the Codex home directory
    codex_home: PathBuf,
}

impl CodexTaskRunner {
    /// Create a new Codex task runner
    pub async fn new(gateway_url: &str, access_token: &str, model: &str) -> Result<Self> {
        let temp_home = TempDir::new().context("Failed to create temp Codex home")?;
        let codex_home = temp_home.path().to_path_buf();

        Ok(Self {
            gateway_url: gateway_url.to_string(),
            access_token: access_token.to_string(),
            model: model.to_string(),
            max_turns: 10,
            _temp_home: temp_home,
            codex_home,
        })
    }

    /// Set the maximum number of turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Build the MCP server configuration for Kontext Gateway
    fn build_gateway_mcp_config(&self) -> McpServerConfig {
        // Build the URL with token
        let url = format!("{}?token={}", self.gateway_url, self.access_token);

        McpServerConfig {
            transport: McpServerTransportConfig::StreamableHttp {
                url,
                bearer_token_env_var: None,
                http_headers: None,
                env_http_headers: None,
            },
            startup_timeout_sec: Some(Duration::from_secs(30)),
            tool_timeout_sec: Some(Duration::from_secs(120)),
            enabled: true,
            enabled_tools: None,
            disabled_tools: None,
        }
    }

    /// Build a Config for running Codex with the Gateway MCP server
    async fn build_config(&self, cwd: &std::path::Path) -> Result<crate::config::Config> {
        use crate::config::ConfigBuilder;

        let overrides = ConfigOverrides {
            cwd: Some(cwd.to_path_buf()),
            ..Default::default()
        };

        let mut config = ConfigBuilder::default()
            .codex_home(self.codex_home.clone())
            .harness_overrides(overrides)
            .build()
            .await
            .context("Failed to build Codex config")?;

        // Set model
        config.model = Some(self.model.clone());

        // Add the Gateway MCP server
        config
            .mcp_servers
            .insert("kontext-gateway".to_string(), self.build_gateway_mcp_config());

        // Set approval policy to never ask (for automated eval)
        config.approval_policy = Constrained::allow_any(AskForApproval::Never);
        config.sandbox_policy = Constrained::allow_any(SandboxPolicy::DangerFullAccess);

        Ok(config)
    }

    /// Run a task using the Codex agent
    pub async fn run_task(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let mut tool_calls = Vec::new();
        let mut total_context_tokens: i64 = 0;
        let mut final_answer = String::new();
        let mut error: Option<String> = None;

        // Create temporary working directory
        let cwd = match TempDir::new() {
            Ok(d) => d,
            Err(e) => {
                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: String::new(),
                    tool_calls,
                    mode: ExecutionMode::CodexAgent,
                    context_tokens: 0,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Failed to create temp dir: {}", e)),
                };
            }
        };

        // Build config
        let config = match self.build_config(cwd.path()).await {
            Ok(c) => c,
            Err(e) => {
                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: String::new(),
                    tool_calls,
                    mode: ExecutionMode::CodexAgent,
                    context_tokens: 0,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Failed to build config: {}", e)),
                };
            }
        };

        // Create auth manager with API key from environment
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "dummy".to_string());
        let auth = CodexAuth::from_api_key(&api_key);
        let auth_manager = AuthManager::from_auth_for_testing_with_home(auth, self.codex_home.clone());

        // Create conversation manager
        let conversation_manager = ConversationManager::new(auth_manager, SessionSource::Exec);

        // Start a new conversation
        let new_conversation = match conversation_manager.new_conversation(config.clone()).await {
            Ok(c) => c,
            Err(e) => {
                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: String::new(),
                    tool_calls,
                    mode: ExecutionMode::CodexAgent,
                    context_tokens: 0,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Failed to start conversation: {}", e)),
                };
            }
        };

        let conversation = new_conversation.conversation;
        let session_model = new_conversation.session_configured.model.clone();

        // Submit the task prompt
        if let Err(e) = conversation
            .submit(Op::UserTurn {
                items: vec![UserInput::Text {
                    text: task.prompt.clone(),
                }],
                final_output_json_schema: None,
                cwd: cwd.path().to_path_buf(),
                approval_policy: AskForApproval::Never,
                sandbox_policy: SandboxPolicy::DangerFullAccess,
                model: session_model,
                effort: None,
                summary: ReasoningSummary::Auto,
            })
            .await
        {
            return TaskResult {
                task_id: task.task_id.clone(),
                final_answer: String::new(),
                tool_calls,
                mode: ExecutionMode::CodexAgent,
                context_tokens: 0,
                latency_ms: start.elapsed().as_millis() as u64,
                error: Some(format!("Failed to submit prompt: {}", e)),
            };
        }

        // Event loop - process events until TaskComplete or error
        let mut turns = 0;
        loop {
            if turns >= self.max_turns {
                error = Some(format!("Max turns ({}) reached", self.max_turns));
                break;
            }

            let event = match conversation.next_event().await {
                Ok(e) => e,
                Err(e) => {
                    error = Some(format!("Event error: {}", e));
                    break;
                }
            };

            match event.msg {
                EventMsg::McpToolCallBegin(begin) => {
                    eprintln!(
                        "[CODEX AGENT] MCP Tool Call Begin: {} - {}",
                        begin.invocation.server, begin.invocation.tool
                    );
                }
                EventMsg::McpToolCallEnd(end) => {
                    eprintln!(
                        "[CODEX AGENT] MCP Tool Call End: {}",
                        end.invocation.tool
                    );

                    // Record the tool call
                    let result_content = match &end.result {
                        Ok(r) => r
                            .content
                            .iter()
                            .filter_map(|c| {
                                if let ContentBlock::TextContent(text) = c {
                                    Some(text.text.clone())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                        Err(e) => format!("Error: {}", e),
                    };

                    let result_tokens = (result_content.len() / 4) as i64;

                    tool_calls.push(ToolCallRecord {
                        name: end.invocation.tool.clone(),
                        arguments: end.invocation.arguments.clone().unwrap_or_default(),
                        result: result_content,
                        result_tokens,
                        stored_in_corpus: false,
                    });
                }
                EventMsg::TokenCount(tc) => {
                    if let Some(ref info) = tc.info {
                        total_context_tokens = info.total_token_usage.total_tokens;
                    }
                    eprintln!(
                        "[CODEX AGENT] Token count: {:?}",
                        tc.info.as_ref().map(|i| i.total_token_usage.total_tokens)
                    );
                }
                EventMsg::AgentMessage(msg) => {
                    // Capture the final answer from agent messages
                    final_answer = msg.message.clone();
                    eprintln!(
                        "[CODEX AGENT] Agent message: {}...",
                        &msg.message[..msg.message.len().min(100)]
                    );
                }
                EventMsg::AgentMessageDelta(delta) => {
                    // Accumulate delta messages
                    final_answer.push_str(&delta.delta);
                }
                EventMsg::TaskComplete(_) => {
                    eprintln!("[CODEX AGENT] Task complete");
                    break;
                }
                EventMsg::Error(e) => {
                    error = Some(format!("Agent error: {}", e.message));
                    eprintln!("[CODEX AGENT] Error: {}", e.message);
                    break;
                }
                _ => {
                    // Ignore other events
                }
            }

            turns += 1;
        }

        TaskResult {
            task_id: task.task_id.clone(),
            final_answer,
            tool_calls,
            mode: ExecutionMode::CodexAgent,
            context_tokens: total_context_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_gateway_mcp_config() {
        let runner = CodexTaskRunner::new(
            "https://gateway.example.com/mcp",
            "test-token",
            "gpt-4o",
        )
        .await
        .expect("create runner");

        let config = runner.build_gateway_mcp_config();
        assert!(config.enabled);

        if let McpServerTransportConfig::StreamableHttp { url, .. } = config.transport {
            assert!(url.contains("gateway.example.com"));
            assert!(url.contains("token=test-token"));
        } else {
            panic!("Expected StreamableHttp transport");
        }
    }
}
