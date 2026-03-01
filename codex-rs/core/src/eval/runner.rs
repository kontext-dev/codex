//! Multi-Mode Task Runner
//!
//! Executes MCP-Atlas tasks in different modes:
//! - Baseline: Direct tool calls with full results in context
//! - CodeMode: Tool calls via EXECUTE_CODE with summarized results
//! - BaselineRlm: Tool calls with RLM routing for large results
//! - Rlm: True RLM mode with Python REPL and sub-LLM calls
//! - CodeMode+RLM: RLM REPL with Gateway tool execution via execute_code()
//! - RlmNative: RLM REPL with Python-native tool wrappers and field projection

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use async_openai::Client;
use async_openai::config::Config;
use async_openai::config::OpenAIConfig;
use async_openai::types::ChatCompletionMessageToolCall;
use async_openai::types::ChatCompletionRequestAssistantMessageArgs;
use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionRequestSystemMessageArgs;
use async_openai::types::ChatCompletionRequestToolMessageArgs;
use async_openai::types::ChatCompletionRequestUserMessageArgs;
use async_openai::types::ChatCompletionTool;
use async_openai::types::ChatCompletionToolArgs;
use async_openai::types::ChatCompletionToolType;
use async_openai::types::CreateChatCompletionRequest;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::CreateChatCompletionResponse;
use async_openai::types::FunctionObjectArgs;
use codex_rmcp_client::RmcpClient;
use regex_lite::Regex;
use serde_json::json;
use tokio::sync::RwLock;

use super::McpAtlasTask;
use crate::rlm::BudgetManager;
use crate::rlm::ContextMetadata;
use crate::rlm::EvidenceStore;
use crate::rlm::FinalAnswer;
use crate::rlm::GatewayResultRouter;
use crate::rlm::LmHandler;
use crate::rlm::LmHandlerConfig;
use crate::rlm::LocalRepl;
use crate::rlm::ProcessedResult;
use crate::rlm::ReplResult;
use crate::rlm::RlmConfig;
use crate::rlm::RlmCorpus;
use crate::rlm::build_rlm_codemode_system_prompt;
use crate::rlm::build_rlm_native_system_prompt;
use crate::rlm::build_rlm_system_prompt;
use crate::rlm::build_rlm_user_prompt;
use crate::rlm::find_code_blocks;
use crate::rlm::find_final_answer;
use crate::rlm::format_repl_feedback;

const RLM_LLM_CALL_TIMEOUT_SECS: u64 = 90;

/// Execution mode for tool-calling client
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolCallingMode {
    /// Direct EXECUTE_TOOL calls with full results in context
    Baseline,
    /// EXECUTE_CODE with summarized results
    CodeMode,
    /// EXECUTE_TOOL with RLM routing for large results
    BaselineRlm,
    /// True RLM mode: Python REPL with iterative LLM loop and sub-LLM calls
    Rlm,
    /// CodeMode+RLM: Python REPL with Gateway tool execution via execute_code()
    RlmCodeMode,
    /// RLM with Python-native tool wrappers: `tools.Server.method(fields=[...])`
    /// Combines RLM's iterative REPL with CodeMode's token efficiency via
    /// field projection, without requiring the LLM to write TypeScript.
    RlmNative,
}

impl std::fmt::Display for ToolCallingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolCallingMode::Baseline => write!(f, "Baseline"),
            ToolCallingMode::CodeMode => write!(f, "CodeMode"),
            ToolCallingMode::BaselineRlm => write!(f, "Baseline+RLM"),
            ToolCallingMode::Rlm => write!(f, "RLM"),
            ToolCallingMode::RlmCodeMode => write!(f, "CodeMode+RLM"),
            ToolCallingMode::RlmNative => write!(f, "RLM+Native"),
        }
    }
}

/// Result of executing a task
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Final answer from the agent
    pub final_answer: String,
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallRecord>,
    /// Execution mode name (e.g., "Baseline", "CodeMode", "Codex")
    pub mode_name: String,
    /// Total context tokens used (last root LLM turn's usage.total_tokens)
    pub context_tokens: i64,
    /// Total LLM tokens across all root LLM turns + sub-LM handler tokens
    pub total_llm_tokens: i64,
    /// Execution latency in milliseconds
    pub latency_ms: u64,
    /// Sum of all LLM inference call durations in milliseconds
    pub llm_time_ms: u64,
    /// Sum of all tool execution call durations in milliseconds
    pub tool_time_ms: u64,
    /// RLM setup overhead in milliseconds (0 for non-RLM modes)
    pub setup_ms: u64,
    /// Total number of LLM API calls made
    pub llm_total_calls: u32,
    /// Total LLM input tokens across all calls
    pub llm_input_tokens: u64,
    /// Total LLM output tokens across all calls
    pub llm_output_tokens: u64,
    /// Error if task failed
    pub error: Option<String>,
}

/// Record of a single tool call
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Tool name
    pub name: String,
    /// Arguments passed
    pub arguments: serde_json::Value,
    /// Result content
    pub result: String,
    /// Token count of result
    pub result_tokens: i64,
    /// Whether result was stored in corpus (RLM mode)
    pub stored_in_corpus: bool,
    /// Wall-clock latency of this tool call in milliseconds
    pub latency_ms: u64,
}

/// Tool information discovered from Gateway
#[derive(Debug, Clone)]
pub struct GatewayTool {
    /// Full tool ID (server:tool_name)
    pub id: String,
    /// Server name
    pub server: String,
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema
    pub input_schema: serde_json::Value,
}

impl GatewayTool {
    /// Prefixed name for use in LLM function definitions (e.g. `Linear_list_issues`).
    /// Guarantees uniqueness when multiple servers expose identically-named tools.
    pub fn prefixed_name(&self) -> String {
        format!("{}_{}", self.server, self.name)
    }
}

/// Tool-calling client runner that supports multiple execution modes
pub struct ToolCallingRunner {
    /// MCP client for Gateway calls
    mcp_client: Arc<RmcpClient>,
    /// LLM client for agent (OpenAI-compatible)
    llm_client: Client<OpenAIConfig>,
    /// RLM router for BaselineRlm mode
    rlm_router: Option<Arc<GatewayResultRouter>>,
    /// Model to use for agent
    agent_model: String,
    /// Maximum agent turns before stopping
    max_turns: usize,
    /// Discovered tools from Gateway (cached)
    gateway_tools: Vec<GatewayTool>,
    /// Tool lookup by name (for quick access)
    tool_lookup: HashMap<String, GatewayTool>,
}

impl ToolCallingRunner {
    /// Create a new task runner and discover available tools from Gateway
    ///
    /// `base_url` and `api_key` configure the OpenAI-compatible endpoint used by
    /// the agent LLM. Both should be provided via `EVAL_API_KEY` / `EVAL_BASE_URL`.
    pub async fn new(
        mcp_client: Arc<RmcpClient>,
        rlm_router: Option<Arc<GatewayResultRouter>>,
        base_url: Option<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let mut config = OpenAIConfig::default();
        if let Some(base) = base_url {
            config = config.with_api_base(base);
        }
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        let llm_client = Client::with_config(config);

        // Discover tools from Gateway
        let gateway_tools = discover_gateway_tools(&mcp_client).await?;

        // Build lookup map
        let mut tool_lookup = HashMap::new();
        for tool in &gateway_tools {
            // Index by various name formats for flexible lookup
            tool_lookup.insert(tool.name.clone(), tool.clone());
            tool_lookup.insert(tool.id.clone(), tool.clone());
            tool_lookup.insert(format!("{}_{}", tool.server, tool.name), tool.clone());
        }

        Ok(Self {
            mcp_client,
            llm_client,
            rlm_router,
            agent_model: std::env::var("EVAL_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()),
            max_turns: 25,
            gateway_tools,
            tool_lookup,
        })
    }

    /// Create with custom model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.agent_model = model.into();
        self
    }

    /// Create with custom max turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Get list of available tools
    pub fn available_tools(&self) -> &[GatewayTool] {
        &self.gateway_tools
    }

    /// Build a tool name lookup map for the LM handler.
    /// Maps various name forms (name, id, server_name, lowercased variants)
    /// to the canonical Gateway tool ID.
    fn build_tool_lookup_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for tool in &self.gateway_tools {
            // Canonical: name → id
            map.insert(tool.name.clone(), tool.id.clone());
            // Exact ID
            map.insert(tool.id.clone(), tool.id.clone());
            // Prefixed: Server_name → id
            map.insert(tool.prefixed_name(), tool.id.clone());
            // Lowercased variants
            map.insert(tool.name.to_lowercase(), tool.id.clone());
            map.insert(tool.id.to_lowercase(), tool.id.clone());
            map.insert(tool.prefixed_name().to_lowercase(), tool.id.clone());
            // Common LLM guesses: server:name without proper casing
            let colon_form = format!(
                "{}:{}",
                tool.server.to_lowercase(),
                tool.name.to_lowercase()
            );
            map.insert(colon_form, tool.id.clone());
        }
        map
    }

    /// Format a tool list section for inclusion in RLM system prompts.
    fn format_tool_list_for_prompt(&self) -> String {
        if self.gateway_tools.is_empty() {
            return String::new();
        }

        let mut section = String::from(
            "\n\n# Available Gateway Tools\n\nYou can call these tools using `execute_tool(tool_id, args)`. Use the EXACT tool_id shown below:\n\n",
        );
        for tool in &self.gateway_tools {
            section.push_str(&format!(
                "- `\"{}\"` — {} (server: {})\n",
                tool.id, tool.description, tool.server
            ));
        }
        section.push_str("\nExample: `execute_tool(\"");
        if let Some(first) = self.gateway_tools.first() {
            section.push_str(&first.id);
        }
        section.push_str("\")` or `execute_tool(\"");
        if let Some(first) = self.gateway_tools.first() {
            section.push_str(&first.id);
        }
        section.push_str("\", {\"key\": \"value\"})`\n");
        section
    }

    /// Format a tool list section for RLM+CodeMode prompts (TypeScript execute_code style).
    ///
    /// Uses `prefixed_name()` (e.g. `Linear_list_projects`) — the same identifiers
    /// that the Gateway VM resolves — instead of raw UUIDs which are not valid JS
    /// identifiers and cause repeated "Invalid or unexpected token" errors.
    fn format_tool_list_for_codemode_prompt(&self) -> String {
        if self.gateway_tools.is_empty() {
            return String::new();
        }

        let mut section = String::from(
            "\n\n# Available Gateway Tools\n\nCall these via `execute_code()` using `codemode.<FunctionName>({...})`. Available functions:\n\n",
        );
        for tool in &self.gateway_tools {
            section.push_str(&format!(
                "- `codemode.{}({{}})` — {}\n",
                tool.prefixed_name(),
                tool.description,
            ));
        }

        // Pick a representative tool for the example
        let sample_name = self
            .gateway_tools
            .iter()
            .find(|t| t.name == "list_projects" || t.name.contains("list"))
            .or(self.gateway_tools.first())
            .map(|t| t.prefixed_name())
            .unwrap_or_else(|| "Linear_list_projects".to_string());

        section.push_str(&format!(
            "\nExample:\n```repl\nresult = execute_code('''\nconst data = await codemode.{sample}({{}});\nreturn {{ summary: data.map(d => d.name || d.title).slice(0, 20) }};\n''')\nprint(result[:500])\n```\n",
            sample = sample_name,
        ));
        section
    }

    async fn chat_create(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, String> {
        let config = self.llm_client.config();
        let url = config.url("/chat/completions");
        let headers = config.headers();
        let client = reqwest::Client::new();
        let mut req = client.post(&url).json(&request);
        for (k, v) in headers.iter() {
            if let Ok(s) = v.to_str() {
                req = req.header(k.as_str(), s);
            }
        }
        let response = req.send().await.map_err(|e| format!("{e}"))?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|e| format!("{e}"))?;
        if !status.is_success() {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                // OpenAI format: {"error": {"message": "..."}}
                let err_obj = v
                    .get("error")
                    // Gemini format: [{"error": {"message": "..."}}]
                    .or_else(|| v.get(0).and_then(|item| item.get("error")));
                if let Some(err) = err_obj {
                    if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                        return Err(format!("API error ({status}): {msg}"));
                    }
                }
            }
            let body_str = String::from_utf8_lossy(&bytes);
            return Err(format!(
                "API error ({status}): {}",
                &body_str[..body_str.len().min(500)]
            ));
        }
        serde_json::from_slice(&bytes).map_err(|e| format!("deserialize response: {e}"))
    }

    /// Execute a task in the specified mode
    pub async fn run_task(&self, task: &McpAtlasTask, mode: ToolCallingMode) -> TaskResult {
        // Dispatch to mode-specific runners
        match mode {
            ToolCallingMode::CodeMode => return self.run_task_codemode(task).await,
            ToolCallingMode::Rlm => return self.run_task_rlm(task).await,
            ToolCallingMode::RlmCodeMode => return self.run_task_rlm_codemode(task).await,
            ToolCallingMode::RlmNative => return self.run_task_rlm_native(task).await,
            _ => {} // Baseline and BaselineRlm continue below
        }

        // Baseline and BaselineRlm use the standard tool-call approach
        let start = Instant::now();
        let mut tool_calls = Vec::new();
        let mut total_context_tokens: i64 = 0;
        let mut total_llm_tokens: i64 = 0;
        let mut llm_time_ms: u64 = 0;
        let mut tool_time_ms: u64 = 0;
        let mut llm_total_calls: u32 = 0;
        let mut llm_input_tokens: u64 = 0;
        let mut llm_output_tokens: u64 = 0;

        // Build tool definitions - include RLM tools for BaselineRlm mode
        let tools = if mode == ToolCallingMode::BaselineRlm {
            self.build_tool_definitions_for_rlm()
        } else {
            self.build_tool_definitions_from_gateway()
        };

        // Build tool list for system prompt (uses prefixed names to match function definitions)
        let tool_list = self
            .gateway_tools
            .iter()
            .map(|t| format!("- {}: {}", t.prefixed_name(), t.description))
            .collect::<Vec<_>>()
            .join("\n");

        // System prompt for the agent with discovered tools
        let system_prompt = if mode == ToolCallingMode::BaselineRlm {
            format!(
                r#"You are an AI assistant executing tasks using available tools.
Complete the task by making appropriate tool calls. When you have gathered
all necessary information, provide a final answer.

# Available Tools

{tool_list}

# Instructions

1. Analyze the task and determine which tools to use
2. Make tool calls to gather information
3. When you have enough information, provide your final answer directly

Note: Tool calls are executed via a Gateway. Use the exact tool names shown above.

Important:
- Do not ask follow-up questions. Answer using only the data from tool calls.
- When the task asks for a count, always include the explicit numerical count.
- If gathering the answer requires multiple tool calls, make all necessary calls before answering.

In rare cases a tool result may be summarized automatically. If so, you can use rlm_search, rlm_get_chunk, or rlm_list_chunks to retrieve the full data."#
            )
        } else {
            format!(
                r#"You are an AI assistant executing tasks using available tools.
Complete the task by making appropriate tool calls. When you have gathered
all necessary information, provide a final answer.

# Available Tools

{tool_list}

# Instructions

1. Analyze the task and determine which tools to use
2. Make tool calls to gather information
3. When you have enough information, provide your final answer directly

Note: Tool calls are executed via a Gateway. Use the exact tool names shown above.

Important:
- Do not ask follow-up questions. Answer using only the data from tool calls.
- When the task asks for a count, always include the explicit numerical count.
- If gathering the answer requires multiple tool calls, make all necessary calls before answering."#
            )
        };

        // Build initial messages
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()
                    .unwrap(),
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(task.prompt.clone())
                    .build()
                    .unwrap(),
            ),
        ];

        tracing::trace!("[TOKEN DEBUG] Task: {}, Mode: {:?}", task.task_id, mode);

        // Agent loop
        for _turn in 0..self.max_turns {
            // Build request
            let request_result = CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .tools(tools.clone())
                .build();

            let request = match request_result {
                Ok(r) => r,
                Err(e) => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode_name: mode.to_string(),
                        context_tokens: total_context_tokens,
                        total_llm_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        llm_time_ms,
                        tool_time_ms,
                        setup_ms: 0,
                        llm_total_calls,
                        llm_input_tokens,
                        llm_output_tokens,
                        error: Some(format!("Failed to build request: {e}")),
                    };
                }
            };

            // Call the agent LLM
            let llm_t = Instant::now();
            let response = match self.chat_create(request).await {
                Ok(r) => r,
                Err(e) => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode_name: mode.to_string(),
                        context_tokens: total_context_tokens,
                        total_llm_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        llm_time_ms,
                        tool_time_ms,
                        setup_ms: 0,
                        llm_total_calls,
                        llm_input_tokens,
                        llm_output_tokens,
                        error: Some(format!("Agent LLM call failed: {e}")),
                    };
                }
            };
            llm_time_ms += llm_t.elapsed().as_millis() as u64;
            llm_total_calls += 1;

            // Track token usage
            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
                total_llm_tokens += usage.total_tokens as i64;
                llm_input_tokens += usage.prompt_tokens as u64;
                llm_output_tokens += usage.completion_tokens as u64;
            }

            let choice = match response.choices.first() {
                Some(c) => c,
                None => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode_name: mode.to_string(),
                        context_tokens: total_context_tokens,
                        total_llm_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        llm_time_ms,
                        tool_time_ms,
                        setup_ms: 0,
                        llm_total_calls,
                        llm_input_tokens,
                        llm_output_tokens,
                        error: Some("No response from agent".to_string()),
                    };
                }
            };

            // Check if there are tool calls
            if let Some(ref calls) = choice.message.tool_calls
                && !calls.is_empty()
            {
                // Process each tool call
                let mut tool_results = Vec::new();

                for call in calls {
                    let tool_t = Instant::now();
                    let result = self.execute_tool_call(call, mode, &task.task_id).await;
                    let tool_latency = tool_t.elapsed().as_millis() as u64;
                    tool_time_ms += tool_latency;

                    // Track the call
                    let record = ToolCallRecord {
                        name: call.function.name.clone(),
                        arguments: serde_json::from_str(&call.function.arguments)
                            .unwrap_or(json!({})),
                        result: result.content.clone(),
                        result_tokens: result.tokens,
                        stored_in_corpus: result.stored_in_corpus,
                        latency_ms: tool_latency,
                    };
                    tool_calls.push(record);

                    tracing::trace!(
                        "[TOKEN DEBUG] Tool: {}, ResultTokens: {}, StoredInCorpus: {}",
                        call.function.name,
                        result.tokens,
                        result.stored_in_corpus,
                    );

                    tool_results.push((call.id.clone(), result.content));
                }

                // Add assistant message with tool calls
                messages.push(ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .tool_calls(calls.clone())
                        .build()
                        .unwrap(),
                ));

                // Add tool result messages
                for (tool_call_id, content) in tool_results {
                    messages.push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(tool_call_id)
                            .content(content)
                            .build()
                            .unwrap(),
                    ));
                }

                continue;
            }

            // No tool calls - this is the final answer
            let final_answer = choice.message.content.clone().unwrap_or_default();

            return TaskResult {
                task_id: task.task_id.clone(),
                final_answer,
                tool_calls,
                mode_name: mode.to_string(),
                context_tokens: total_context_tokens,
                total_llm_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                llm_time_ms,
                tool_time_ms,
                setup_ms: 0,
                llm_total_calls,
                llm_input_tokens,
                llm_output_tokens,
                error: None,
            };
        }

        // Max turns reached
        TaskResult {
            task_id: task.task_id.clone(),
            final_answer: format!(
                "Max turns ({}) reached without final answer",
                self.max_turns
            ),
            tool_calls,
            mode_name: mode.to_string(),
            context_tokens: total_context_tokens,
            total_llm_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
            llm_time_ms,
            tool_time_ms,
            setup_ms: 0,
            llm_total_calls,
            llm_input_tokens,
            llm_output_tokens,
            error: Some("Max turns reached".to_string()),
        }
    }

    /// Build tool definitions for the LLM from discovered Gateway tools.
    /// Uses server-prefixed names (e.g. `Linear_list_issues`) to avoid
    /// duplicate function names when multiple servers expose the same tool.
    fn build_tool_definitions_from_gateway(&self) -> Vec<ChatCompletionTool> {
        self.gateway_tools
            .iter()
            .map(|tool| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(
                        FunctionObjectArgs::default()
                            .name(tool.prefixed_name())
                            .description(tool.description.clone())
                            .parameters(tool.input_schema.clone())
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap()
            })
            .collect()
    }

    /// Build tool definitions for RLM mode (includes Gateway tools + corpus access tools)
    fn build_tool_definitions_for_rlm(&self) -> Vec<ChatCompletionTool> {
        let mut tools = self.build_tool_definitions_from_gateway();
        tools.extend(self.build_rlm_tool_definitions());
        tools
    }

    /// Build only the RLM corpus access tool definitions (without Gateway tools)
    fn build_rlm_tool_definitions(&self) -> Vec<ChatCompletionTool> {
        let mut tools = Vec::new();

        // Add rlm_search tool
        tools.push(
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name("rlm_search")
                        .description("Search tool results by semantic query.")
                        .parameters(json!({
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant data"
                                }
                            },
                            "required": ["query"]
                        }))
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        // Add rlm_get_chunk tool
        tools.push(
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name("rlm_get_chunk")
                        .description("Retrieve a specific chunk of stored data by its ID.")
                        .parameters(json!({
                            "type": "object",
                            "properties": {
                                "chunk_id": {
                                    "type": "string",
                                    "description": "The chunk ID to retrieve"
                                }
                            },
                            "required": ["chunk_id"]
                        }))
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        // Add rlm_list_chunks tool
        tools.push(
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name("rlm_list_chunks")
                        .description("List all available data chunks with their summaries.")
                        .parameters(json!({
                            "type": "object",
                            "properties": {}
                        }))
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        tools
    }

    /// Resolve a tool name to its full ID using substring matching
    pub fn resolve_tool(&self, name: &str) -> Option<&GatewayTool> {
        // First try exact match
        if let Some(tool) = self.tool_lookup.get(name) {
            return Some(tool);
        }

        // Try substring matching - find tools where the name contains or is contained
        let name_lower = name.to_lowercase();

        for tool in &self.gateway_tools {
            let tool_name_lower = tool.name.to_lowercase();

            // Check if the requested name contains the tool name or vice versa
            // e.g., "git_git_status" contains "status", "filesystem_read_file" contains "read_file"
            if name_lower.contains(&tool_name_lower) || tool_name_lower.contains(&name_lower) {
                return Some(tool);
            }

            // Also check against server_toolname pattern
            let combined = format!("{}_{}", tool.server.to_lowercase(), tool_name_lower);
            if name_lower.contains(&combined) || combined.contains(&name_lower) {
                return Some(tool);
            }
        }

        None
    }

    /// Check if a task can be executed with available tools
    /// Returns true if at least one of the task's enabled tools matches an available Gateway tool
    pub fn can_execute_task(&self, task: &McpAtlasTask) -> bool {
        task.enabled_tools
            .iter()
            .any(|tool_name| self.resolve_tool(tool_name).is_some())
    }

    /// Get matching tools for a task
    pub fn get_matching_tools<'a>(
        &'a self,
        task: &'a McpAtlasTask,
    ) -> Vec<(&'a str, &'a GatewayTool)> {
        task.enabled_tools
            .iter()
            .filter_map(|tool_name| {
                self.resolve_tool(tool_name)
                    .map(|t| (tool_name.as_str(), t))
            })
            .collect()
    }

    /// Filter tasks to only those that can be executed with available tools
    /// and match the specified server filter (e.g., ["git", "CLI", "Code Executor"])
    pub fn filter_tasks_by_servers<'a>(
        &self,
        tasks: &'a [McpAtlasTask],
        server_filter: &[&str],
    ) -> Vec<&'a McpAtlasTask> {
        tasks
            .iter()
            .filter(|task| {
                // Check if any of the task's tools match a Gateway tool from allowed servers
                task.enabled_tools.iter().any(|tool_name| {
                    if let Some(gateway_tool) = self.resolve_tool(tool_name) {
                        server_filter.iter().any(|s| {
                            gateway_tool
                                .server
                                .to_lowercase()
                                .contains(&s.to_lowercase())
                        })
                    } else {
                        false
                    }
                })
            })
            .collect()
    }

    /// Get tool coverage ratio for a task (0.0 to 1.0)
    pub fn get_tool_coverage(&self, task: &McpAtlasTask) -> f64 {
        if task.enabled_tools.is_empty() {
            return 0.0;
        }
        let matched = task
            .enabled_tools
            .iter()
            .filter(|t| self.resolve_tool(t).is_some())
            .count();
        matched as f64 / task.enabled_tools.len() as f64
    }

    /// Filter to only fully solvable tasks (100% tool coverage)
    pub fn filter_fully_solvable<'a>(&self, tasks: &'a [McpAtlasTask]) -> Vec<&'a McpAtlasTask> {
        tasks
            .iter()
            .filter(|task| {
                !task.enabled_tools.is_empty()
                    && task
                        .enabled_tools
                        .iter()
                        .all(|t| self.resolve_tool(t).is_some())
            })
            .collect()
    }

    /// Filter to tasks with at least the given tool coverage ratio
    pub fn filter_by_coverage<'a>(
        &self,
        tasks: &'a [McpAtlasTask],
        min_coverage: f64,
    ) -> Vec<&'a McpAtlasTask> {
        tasks
            .iter()
            .filter(|task| self.get_tool_coverage(task) >= min_coverage)
            .collect()
    }

    /// Execute a tool call in the specified mode
    async fn execute_tool_call(
        &self,
        call: &ChatCompletionMessageToolCall,
        mode: ToolCallingMode,
        task_id: &str,
    ) -> ToolCallResult {
        let tool_name = &call.function.name;
        let args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));

        // Handle RLM-specific tools in BaselineRlm mode
        if mode == ToolCallingMode::BaselineRlm {
            match tool_name.as_str() {
                "rlm_search" => return self.execute_rlm_search(&args).await,
                "rlm_get_chunk" => return self.execute_rlm_get_chunk(&args).await,
                "rlm_list_chunks" => return self.execute_rlm_list_chunks().await,
                _ => {} // Fall through to normal tool execution
            }
        }

        // Resolve the tool to get its full ID
        let tool_id = self
            .resolve_tool(tool_name)
            .map(|t| t.id.clone())
            .unwrap_or_else(|| tool_name.clone());

        match mode {
            ToolCallingMode::Baseline => self.execute_baseline(&tool_id, &args).await,
            ToolCallingMode::CodeMode => self.execute_codemode(&tool_id, &args).await,
            ToolCallingMode::Rlm => {
                // Rlm is dispatched to run_task_rlm() directly and does not use tool calls
                self.execute_baseline(&tool_id, &args).await
            }
            ToolCallingMode::BaselineRlm => {
                self.execute_with_rlm(&tool_id, &args, task_id, &call.id)
                    .await
            }
            ToolCallingMode::RlmCodeMode => {
                // RlmCodeMode is dispatched to run_task_rlm_codemode() directly
                // and does not use standard tool call dispatch
                self.execute_baseline(&tool_id, &args).await
            }
            ToolCallingMode::RlmNative => {
                // RlmNative is dispatched to run_task_rlm_native() directly
                // and does not use standard tool call dispatch
                self.execute_baseline(&tool_id, &args).await
            }
        }
    }

    /// Execute rlm_search tool
    async fn execute_rlm_search(&self, args: &serde_json::Value) -> ToolCallResult {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        if let Some(ref router) = self.rlm_router {
            let results = router.search_corpus(query, 5).await;
            let content = serde_json::to_string_pretty(&results).unwrap_or_default();
            let tokens = estimate_tokens(&content);
            ToolCallResult {
                content,
                tokens,
                stored_in_corpus: false,
            }
        } else {
            ToolCallResult {
                content: "RLM router not available".to_string(),
                tokens: 10,
                stored_in_corpus: false,
            }
        }
    }

    /// Execute rlm_get_chunk tool
    async fn execute_rlm_get_chunk(&self, args: &serde_json::Value) -> ToolCallResult {
        let chunk_id = args.get("chunk_id").and_then(|v| v.as_str()).unwrap_or("");

        if let Some(ref router) = self.rlm_router {
            match router.get_chunk(chunk_id).await {
                Some(content) => {
                    let tokens = estimate_tokens(&content);
                    ToolCallResult {
                        content,
                        tokens,
                        stored_in_corpus: false,
                    }
                }
                None => ToolCallResult {
                    content: format!("Chunk '{chunk_id}' not found"),
                    tokens: 10,
                    stored_in_corpus: false,
                },
            }
        } else {
            ToolCallResult {
                content: "RLM router not available".to_string(),
                tokens: 10,
                stored_in_corpus: false,
            }
        }
    }

    /// Execute rlm_list_chunks tool
    async fn execute_rlm_list_chunks(&self) -> ToolCallResult {
        if let Some(ref router) = self.rlm_router {
            let chunks = router.list_chunks().await;
            let content = serde_json::to_string_pretty(&chunks).unwrap_or_default();
            let tokens = estimate_tokens(&content);
            ToolCallResult {
                content,
                tokens,
                stored_in_corpus: false,
            }
        } else {
            ToolCallResult {
                content: "RLM router not available".to_string(),
                tokens: 10,
                stored_in_corpus: false,
            }
        }
    }

    /// Execute tool in baseline mode (full result)
    async fn execute_baseline(&self, tool_id: &str, args: &serde_json::Value) -> ToolCallResult {
        // Gateway's EXECUTE_TOOL expects "tool_arguments" not "args"
        let execute_args = json!({
            "tool_id": tool_id,
            "tool_arguments": args
        });

        match self
            .mcp_client
            .call_tool(
                "EXECUTE_TOOL".to_string(),
                Some(execute_args),
                Some(Duration::from_secs(120)),
            )
            .await
        {
            Ok(result) => {
                let content = serde_json::to_string(&result).unwrap_or_default();
                let tokens = estimate_tokens(&content);
                ToolCallResult {
                    content,
                    tokens,
                    stored_in_corpus: false,
                }
            }
            Err(e) => ToolCallResult {
                content: format!("Tool call failed: {e}"),
                tokens: 10,
                stored_in_corpus: false,
            },
        }
    }

    /// Execute tool in code mode - call EXECUTE_CODE and return FULL result
    /// The server runs the code in a VM and returns complete tool output
    async fn execute_codemode(&self, tool_id: &str, args: &serde_json::Value) -> ToolCallResult {
        // Gateway VM uses `new AsyncFunction("tools", "console", code)` — code is the
        // function body, so bare statements with `return`/`await` work directly.
        let code = format!(
            r#"const result = await tools.EXECUTE_TOOL({{ tool_id: "{}", tool_arguments: {} }});
return result;"#,
            tool_id,
            serde_json::to_string(args).unwrap_or("{}".to_string())
        );

        match self
            .mcp_client
            .call_tool(
                "EXECUTE_CODE".to_string(),
                Some(json!({ "code": code })),
                Some(Duration::from_secs(120)),
            )
            .await
        {
            Ok(result) => {
                let content = serde_json::to_string(&result).unwrap_or_default();
                let tokens = estimate_tokens(&content);
                ToolCallResult {
                    content,
                    tokens,
                    stored_in_corpus: false,
                }
            }
            Err(e) => ToolCallResult {
                content: format!("Code execution failed: {e}"),
                tokens: 10,
                stored_in_corpus: false,
            },
        }
    }

    /// Build system prompt for CodeMode - instructs LLM to generate JavaScript code
    fn build_codemode_system_prompt(&self) -> String {
        let generated_types = super::codemode_types::generate_codemode_types(&self.gateway_tools);

        // Get a sample tool name for the example
        let sample_name = self
            .gateway_tools
            .iter()
            .find(|t| t.name == "list_projects" || t.name.contains("list"))
            .map(|t| t.prefixed_name())
            .unwrap_or_else(|| "Linear_list_projects".to_string());

        format!(
            r#"You are an AI assistant. Complete tasks by writing JavaScript code.

# Available Tools (type reference)

{generated_types}

# How to Call Tools

Write a SINGLE ```javascript code block containing an async arrow function that calls the `codemode` object. `codemode.*` calls return clean, unwrapped data — if the upstream API returns a single list, you get the array directly.

```javascript
async () => {{
  const items = await codemode.{sample_name}({{}});
  // items is typically an array — map directly
  const names = items.map(i => i.name ?? i.title ?? i.key);
  return {{ count: names.length, names }};
}}
```

# Rules

- Write an async arrow function: `async () => {{ ... }}`. Do NOT write bare statements.
- Call tools via `codemode.<ToolName>({{ ... }})` — NOT via `tools.EXECUTE_TOOL`.
- `codemode.*` calls return clean, unwrapped data. List endpoints return arrays directly — use `.map()`, `.filter()`, `.length` on the result immediately. Do NOT access `.nodes`, `.items`, or other wrapper properties.
- `codemode.*` calls throw on errors with a descriptive message. If a call fails, the error will tell you which tool and what went wrong.
- Chain ALL tool calls in one code block. Do NOT make one call per block.
- Extract only the fields you need and return a compact summary. Do NOT return raw tool results.
- When the task asks for a count, compute it in code and include it in your return value.
- Do not ask follow-up questions. Answer using only the data from tool calls.
- Write plain JavaScript only. Do NOT use TypeScript syntax (no type annotations, no `as` casts, no interfaces).
- When accessing nested properties (e.g. `item.status.name`), guard against undefined: use `item.status && item.status.name` or optional chaining `item?.status?.name`.

# Workflow

1. Write ONE ```javascript code block with an async arrow function that calls all needed tools via `codemode.*` and returns extracted data
2. Read the compact result and provide a plain text answer to the task

Your final answer must be plain text that directly answers the task question, not a code block."#
        )
    }

    /// Execute code via EXECUTE_CODE tool and return the result.
    /// Normalizes LLM-generated code (unwraps async arrow functions) so the
    /// Gateway's AsyncFunction body receives bare statements.
    /// The Gateway already provides the `codemode` runtime object — no
    /// client-side preamble is needed.
    async fn execute_code(&self, code: &str) -> String {
        // Gateway VM uses `new AsyncFunction("tools", "console", code)` — code is the
        // function body. Unwrap async arrows the LLM may generate into bare statements.
        let code = unwrap_async_arrow(code);
        match self
            .mcp_client
            .call_tool(
                "EXECUTE_CODE".to_string(),
                Some(json!({ "code": code })),
                Some(Duration::from_secs(120)),
            )
            .await
        {
            Ok(result) => extract_execute_code_result(&result),
            Err(e) => format!("Code execution failed: {e}"),
        }
    }

    /// Execute a task in CodeMode - LLM generates code, we execute it
    pub async fn run_task_codemode(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let system_prompt = self.build_codemode_system_prompt();
        tracing::warn!(
            "  [codemode] system prompt built ({} chars, {} tools, types: {})",
            system_prompt.len(),
            self.gateway_tools.len(),
            system_prompt.contains("declare const codemode"),
        );
        let mut total_context_tokens: i64 = 0;
        let mut total_llm_tokens: i64 = 0;
        let mut tool_calls = Vec::new();
        let mut llm_time_ms: u64 = 0;
        let mut tool_time_ms: u64 = 0;
        let mut llm_total_calls: u32 = 0;
        let mut llm_input_tokens: u64 = 0;
        let mut llm_output_tokens: u64 = 0;

        // Build initial messages - NO tool definitions, we want code generation
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()
                    .unwrap(),
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(task.prompt.clone())
                    .build()
                    .unwrap(),
            ),
        ];

        // Agent loop
        for turn in 0..self.max_turns {
            tracing::warn!(
                "  [codemode] turn {}/{} — calling LLM...",
                turn + 1,
                self.max_turns
            );
            // Call LLM WITHOUT tools - we want code generation, not tool calls
            let request_result = CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build();

            let request = match request_result {
                Ok(r) => r,
                Err(e) => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode_name: "CodeMode".to_string(),
                        context_tokens: total_context_tokens,
                        total_llm_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        llm_time_ms,
                        tool_time_ms,
                        setup_ms: 0,
                        llm_total_calls,
                        llm_input_tokens,
                        llm_output_tokens,
                        error: Some(format!("Failed to build request: {e}")),
                    };
                }
            };

            // Call the agent LLM (with retry for transient errors / empty completions)
            let llm_t = Instant::now();
            let response = {
                let mut last_err = String::new();
                let mut resp = None;
                for attempt in 0..3u32 {
                    if attempt > 0 {
                        let delay = Duration::from_millis(1000 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                        tracing::warn!(
                            "  [codemode] retrying LLM call (attempt {}/3)...",
                            attempt + 1
                        );
                    }
                    match tokio::time::timeout(
                        Duration::from_secs(RLM_LLM_CALL_TIMEOUT_SECS),
                        self.chat_create(request.clone()),
                    )
                    .await
                    {
                        Ok(Ok(r)) => {
                            let has_content = r
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                                .map(|s| !s.is_empty())
                                .unwrap_or(false);
                            if has_content || attempt == 2 {
                                resp = Some(r);
                                break;
                            }
                            tracing::warn!(
                                "  [codemode] empty completion on attempt {}",
                                attempt + 1
                            );
                            resp = Some(r);
                        }
                        Ok(Err(e)) => {
                            last_err = format!("Agent LLM call failed: {e}");
                            tracing::warn!("  [codemode] {last_err} (attempt {})", attempt + 1);
                        }
                        Err(_) => {
                            last_err =
                                format!("LLM call timed out after {RLM_LLM_CALL_TIMEOUT_SECS}s");
                            tracing::warn!("  [codemode] {last_err} (attempt {})", attempt + 1);
                        }
                    }
                }
                match resp {
                    Some(r) => r,
                    None => {
                        return TaskResult {
                            task_id: task.task_id.clone(),
                            final_answer: String::new(),
                            tool_calls,
                            mode_name: "CodeMode".to_string(),
                            context_tokens: total_context_tokens,
                            total_llm_tokens,
                            latency_ms: start.elapsed().as_millis() as u64,
                            llm_time_ms,
                            tool_time_ms,
                            setup_ms: 0,
                            llm_total_calls,
                            llm_input_tokens,
                            llm_output_tokens,
                            error: Some(last_err),
                        };
                    }
                }
            };
            llm_time_ms += llm_t.elapsed().as_millis() as u64;
            llm_total_calls += 1;

            // Track token usage
            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
                total_llm_tokens += usage.total_tokens as i64;
                llm_input_tokens += usage.prompt_tokens as u64;
                llm_output_tokens += usage.completion_tokens as u64;
            }

            let choice = match response.choices.first() {
                Some(c) => c,
                None => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode_name: "CodeMode".to_string(),
                        context_tokens: total_context_tokens,
                        total_llm_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        llm_time_ms,
                        tool_time_ms,
                        setup_ms: 0,
                        llm_total_calls,
                        llm_input_tokens,
                        llm_output_tokens,
                        error: Some("No response from agent".to_string()),
                    };
                }
            };

            let content = choice.message.content.clone().unwrap_or_default();
            tracing::warn!(
                "  [codemode] LLM responded ({} chars), uses codemode.*: {}",
                content.len(),
                content.contains("codemode."),
            );
            // Debug: log the full LLM response
            tracing::warn!(
                "  [codemode] LLM content:\n{}",
                &content[..content.len().min(600)]
            );

            // Check if response contains code block
            if let Some(code) = extract_code_block(&content) {
                tracing::warn!(
                    "  [codemode] extracted code block ({} chars):\n{}",
                    code.len(),
                    &code[..code.len().min(500)],
                );
                // Execute the code via EXECUTE_CODE
                let tool_t = Instant::now();
                let exec_result = self.execute_code(&code).await;
                let tool_latency = tool_t.elapsed().as_millis() as u64;
                tool_time_ms += tool_latency;
                tracing::warn!(
                    "  [codemode] EXECUTE_CODE returned ({} chars)",
                    exec_result.len(),
                );

                tool_calls.push(ToolCallRecord {
                    name: "EXECUTE_CODE".to_string(),
                    arguments: json!({ "code": code }),
                    result: exec_result.clone(),
                    result_tokens: estimate_tokens(&exec_result),
                    stored_in_corpus: false,
                    latency_ms: tool_latency,
                });

                // Add assistant message
                messages.push(ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(content.clone())
                        .build()
                        .unwrap(),
                ));

                // Add execution result as user message
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(format!(
                            "Code execution result:\n```json\n{exec_result}\n```"
                        ))
                        .build()
                        .unwrap(),
                ));

                continue;
            }

            tracing::warn!("  [codemode] no code block — treating as final answer");
            // No code block - this is the final answer
            return TaskResult {
                task_id: task.task_id.clone(),
                final_answer: content,
                tool_calls,
                mode_name: "CodeMode".to_string(),
                context_tokens: total_context_tokens,
                total_llm_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                llm_time_ms,
                tool_time_ms,
                setup_ms: 0,
                llm_total_calls,
                llm_input_tokens,
                llm_output_tokens,
                error: None,
            };
        }

        // Max turns reached
        TaskResult {
            task_id: task.task_id.clone(),
            final_answer: format!(
                "Max turns ({}) reached without final answer",
                self.max_turns
            ),
            tool_calls,
            mode_name: "CodeMode".to_string(),
            context_tokens: total_context_tokens,
            total_llm_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
            llm_time_ms,
            tool_time_ms,
            setup_ms: 0,
            llm_total_calls,
            llm_input_tokens,
            llm_output_tokens,
            error: Some("Max turns reached".to_string()),
        }
    }

    /// Execute a task in true RLM mode: Python REPL with iterative LLM loop
    ///
    /// This mode does NOT use tool definitions. Instead, the LLM generates Python
    /// code in ```repl blocks which are executed in a persistent Python subprocess.
    /// Sub-LLM calls are available via `llm_query()` / `llm_query_batched()` in the
    /// REPL. The loop runs until the LLM emits FINAL(...) or FINAL_VAR(...), or
    /// max iterations are reached.
    pub async fn run_task_rlm(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let setup_t = Instant::now();

        // 1. Create RLM infrastructure
        let rlm_config = RlmConfig::default();
        let budget = Arc::new(BudgetManager::new(&rlm_config));

        let api_key = std::env::var("EVAL_API_KEY").unwrap_or_default();
        let base_url = std::env::var("EVAL_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty());

        // Create corpus + router so RLM can bound tool result growth.
        let temp_dir = match tempfile::TempDir::new() {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    task,
                    "RLM",
                    start,
                    &format!("Temp dir creation failed: {e}"),
                );
            }
        };
        let corpus =
            match crate::rlm::RlmCorpus::new(temp_dir.path().to_path_buf(), rlm_config.clone())
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    return error_result(
                        task,
                        "RLM",
                        start,
                        &format!("Corpus creation failed: {e}"),
                    );
                }
            };
        if let Err(e) = corpus.ingest_prompt(&task.prompt).await {
            return error_result(task, "RLM", start, &format!("Corpus ingest failed: {e}"));
        }
        let corpus_arc = Arc::new(RwLock::new(Some(corpus)));
        let evidence_store = Arc::new(RwLock::new(EvidenceStore::new()));
        let router = Arc::new(GatewayResultRouter::new(
            corpus_arc,
            evidence_store,
            rlm_config.clone(),
        ));

        let lm_config = LmHandlerConfig {
            root_model: self.agent_model.clone(),
            sub_model: None,
            api_key,
            base_url,
            mcp_client: Some(Arc::clone(&self.mcp_client)),
            rlm_router: Some(Arc::clone(&router)),
            tool_lookup: self.build_tool_lookup_map(),
        };

        let lm_handler = match LmHandler::start(lm_config, budget).await {
            Ok(h) => h,
            Err(e) => {
                return error_result(task, "RLM", start, &format!("LM handler start failed: {e}"));
            }
        };

        // 2. Create REPL
        let mut repl = match LocalRepl::new(lm_handler.port()).await {
            Ok(r) => r,
            Err(e) => {
                lm_handler.stop().await;
                return error_result(task, "RLM", start, &format!("REPL start failed: {e}"));
            }
        };

        // 3. Inject context (the task prompt is the context for RLM)
        if let Err(e) = repl.set_context(&task.prompt).await {
            repl.cleanup().await;
            lm_handler.stop().await;
            return error_result(task, "RLM", start, &format!("Set context failed: {e}"));
        }

        // 4. Build context metadata
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: task.prompt.len(),
            context_lengths: vec![task.prompt.len()],
        };

        // Inject available tools into the system prompt so the LLM knows what to call
        let tool_list_section = self.format_tool_list_for_prompt();
        let system_prompt = format!(
            "{}{}",
            build_rlm_system_prompt(&metadata),
            tool_list_section
        );
        let setup_ms = setup_t.elapsed().as_millis() as u64;
        let mut total_context_tokens: i64 = 0;
        let mut total_llm_tokens: i64 = 0;
        let mut tool_calls = Vec::new();
        let mut last_assistant_content = String::new();
        let mut last_repl_stdout = String::new();
        let mut llm_time_ms: u64 = 0;
        let mut tool_time_ms: u64 = 0;
        let mut llm_total_calls: u32 = 0;
        let mut llm_input_tokens: u64 = 0;
        let mut llm_output_tokens: u64 = 0;
        let mut last_stdout_hash: u64 = 0;
        let mut consecutive_same_output: u32 = 0;
        let mut consecutive_empty: u32 = 0;

        // 5. Build conversation messages
        let mut messages: Vec<ChatCompletionRequestMessage> =
            vec![ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()
                    .unwrap(),
            )];

        // 6. Iterative REPL loop
        for iteration in 0..self.max_turns {
            // Add user prompt for this iteration
            let user_prompt = build_rlm_user_prompt(iteration, self.max_turns, &task.prompt);
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_prompt.clone())
                    .build()
                    .unwrap(),
            ));

            // Call LLM (NO tools — we want free-form text with ```repl blocks)
            let request = match CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    repl.cleanup().await;
                    lm_handler.stop().await;
                    return error_result(task, "RLM", start, &format!("Request build failed: {e}"));
                }
            };

            let llm_t = Instant::now();
            let response = {
                let mut last_err = String::new();
                let mut resp = None;
                for attempt in 0..3u32 {
                    if attempt > 0 {
                        let delay = Duration::from_millis(1000 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                        tracing::warn!("  [rlm] retrying LLM call (attempt {}/3)...", attempt + 1);
                    }
                    match tokio::time::timeout(
                        Duration::from_secs(RLM_LLM_CALL_TIMEOUT_SECS),
                        self.chat_create(request.clone()),
                    )
                    .await
                    {
                        Ok(Ok(r)) => {
                            let has_content = r
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                                .map(|s| !s.is_empty())
                                .unwrap_or(false);
                            if has_content || attempt == 2 {
                                resp = Some(r);
                                break;
                            }
                            tracing::warn!("  [rlm] empty completion on attempt {}", attempt + 1);
                            resp = Some(r);
                        }
                        Ok(Err(e)) => {
                            last_err = format!("LLM call failed: {e}");
                            tracing::warn!("  [rlm] {last_err} (attempt {})", attempt + 1);
                        }
                        Err(_) => {
                            last_err =
                                format!("LLM call timed out after {RLM_LLM_CALL_TIMEOUT_SECS}s");
                            tracing::warn!("  [rlm] {last_err} (attempt {})", attempt + 1);
                        }
                    }
                }
                match resp {
                    Some(r) => r,
                    None => {
                        repl.cleanup().await;
                        lm_handler.stop().await;
                        return error_result(task, "RLM", start, &last_err);
                    }
                }
            };
            llm_time_ms += llm_t.elapsed().as_millis() as u64;
            llm_total_calls += 1;

            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
                total_llm_tokens += usage.total_tokens as i64;
                llm_input_tokens += usage.prompt_tokens as u64;
                llm_output_tokens += usage.completion_tokens as u64;
            }

            let content = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            last_assistant_content = content.clone();

            if content.is_empty() {
                tracing::warn!("  [rlm] empty LLM response on iteration {}", iteration);
                continue;
            }

            // Add assistant message
            messages.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content.clone())
                    .build()
                    .unwrap(),
            ));

            // Check for FINAL answer
            if let Some(final_answer) = find_final_answer(&content) {
                let answer = resolve_final_answer_text(&mut repl, final_answer).await;
                if answer.trim().is_empty() {
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content("Your FINAL(...) was empty. Provide a non-empty final answer based on your computed results. If needed, run one more ```repl block to print the final values, then call FINAL(...) with actual content.".to_string())
                            .build()
                            .unwrap(),
                    ));
                    continue;
                }

                let sub_usage = lm_handler.usage().await;
                let total_llm_tokens_final =
                    total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
                let sub_input = sub_usage.total_input_tokens as u64;
                let sub_output = sub_usage.total_output_tokens as u64;
                repl.cleanup().await;
                lm_handler.stop().await;

                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: answer,
                    tool_calls,
                    mode_name: "RLM".to_string(),
                    context_tokens: total_context_tokens,
                    total_llm_tokens: total_llm_tokens_final,
                    latency_ms: start.elapsed().as_millis() as u64,
                    llm_time_ms,
                    tool_time_ms,
                    setup_ms,
                    llm_total_calls,
                    llm_input_tokens: llm_input_tokens + sub_input,
                    llm_output_tokens: llm_output_tokens + sub_output,
                    error: None,
                };
            }

            // Extract and execute code blocks
            let code_blocks = find_code_blocks(&content);
            if code_blocks.is_empty() && iteration > 0 {
                // No code blocks and no FINAL — if content is substantive, use it and break
                // to let force-final run; otherwise nudge the model to keep going
                if content.len() > 80 {
                    last_assistant_content = content;
                    break;
                }
                // Ask the model to print its stored variables and wrap with FINAL()
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Your response had no code block and no FINAL(). You must either write a ```repl code block to continue working, or wrap your final answer with FINAL(your answer). If you have already computed the answer in a variable, write a ```repl block with print(your_variable) and then use FINAL().".to_string())
                        .build()
                        .unwrap(),
                ));
                continue;
            }

            // Execute each code block
            for code in &code_blocks {
                let tool_t = Instant::now();
                let result = match repl.execute(code).await {
                    Ok(r) => r,
                    Err(e) => ReplResult {
                        stdout: String::new(),
                        stderr: format!("REPL execution error: {e}"),
                        locals_summary: String::new(),
                        execution_time_ms: 0,
                    },
                };
                let tool_latency = tool_t.elapsed().as_millis() as u64;
                tool_time_ms += tool_latency;

                if !result.stdout.trim().is_empty() {
                    last_repl_stdout = result.stdout.clone();
                }

                // Track consecutive identical output (Fix 5)
                {
                    let mut hasher = std::hash::DefaultHasher::new();
                    result.stdout.hash(&mut hasher);
                    let stdout_hash = hasher.finish();
                    if stdout_hash == last_stdout_hash && !result.stdout.trim().is_empty() {
                        consecutive_same_output += 1;
                    } else {
                        consecutive_same_output = 0;
                        last_stdout_hash = stdout_hash;
                    }
                }

                // Track consecutive empty output (Fix 6)
                if result.stdout.trim().is_empty() && result.stderr.trim().is_empty() {
                    consecutive_empty += 1;
                } else {
                    consecutive_empty = 0;
                }

                tool_calls.push(ToolCallRecord {
                    name: "repl_execute".to_string(),
                    arguments: json!({ "code": code }),
                    result: result.stdout.clone(),
                    result_tokens: estimate_tokens(&result.stdout),
                    stored_in_corpus: false,
                    latency_ms: tool_latency,
                });

                // Format feedback for LLM
                let feedback = format_repl_feedback(
                    code,
                    &result.stdout,
                    &result.stderr,
                    &result.locals_summary,
                    20_000,
                );

                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(feedback.clone())
                        .build()
                        .unwrap(),
                ));
            }

            // Fix 5: Break if same output repeats too many times
            if consecutive_same_output >= 3 {
                tracing::warn!(
                    "  [rlm] identical output {} consecutive times, breaking to force-final",
                    consecutive_same_output + 1
                );
                break;
            }

            // Fix 6: Nudge if too many consecutive empty outputs
            if consecutive_empty >= 5 {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("WARNING: Your last 5 code executions produced no output. Your code may have silent errors. Try adding print() statements to debug, or simplify your approach. Check that variable names are correct and that execute_tool() results are being captured properly.".to_string())
                        .build()
                        .unwrap(),
                ));
                consecutive_empty = 0;
            }

            // Fix 4: Detect final_answer variable assignment
            if code_blocks
                .iter()
                .any(|c| c.contains("final_answer =") || c.contains("final_answer="))
            {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Note: You assigned to a Python variable `final_answer` but did NOT call FINAL(). To submit your answer, write FINAL(your_answer) as plain text (outside a code block), or use FINAL_VAR(final_answer) to return the variable's value.".to_string())
                        .build()
                        .unwrap(),
                ));
            }
        }

        // Max iterations reached — force one more LLM call for final answer
        {
            let force_prompt = "You have reached the maximum number of iterations. Based on ALL the data and analysis from your previous steps, provide your final answer NOW. Use FINAL(your answer) to wrap it.".to_string();
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(force_prompt)
                    .build()
                    .unwrap(),
            ));

            if let Ok(request) = CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                let llm_t = Instant::now();
                if let Ok(response) = self.chat_create(request).await {
                    llm_time_ms += llm_t.elapsed().as_millis() as u64;
                    llm_total_calls += 1;
                    if let Some(usage) = &response.usage {
                        total_context_tokens = usage.total_tokens as i64;
                        total_llm_tokens += usage.total_tokens as i64;
                        llm_input_tokens += usage.prompt_tokens as u64;
                        llm_output_tokens += usage.completion_tokens as u64;
                    }
                    if let Some(content) = response
                        .choices
                        .first()
                        .and_then(|c| c.message.content.clone())
                    {
                        if !content.trim().is_empty() {
                            last_assistant_content = content;
                        }
                    }
                }
            }
        }

        // If force-final produced a useful answer, prefer it over stale REPL stdout.
        if !last_assistant_content.trim().is_empty() {
            last_repl_stdout.clear();
        }

        let sub_usage = lm_handler.usage().await;
        let total_llm_tokens_final =
            total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
        let sub_input = sub_usage.total_input_tokens as u64;
        let sub_output = sub_usage.total_output_tokens as u64;
        repl.cleanup().await;
        lm_handler.stop().await;

        let final_answer = if !last_repl_stdout.trim().is_empty() {
            last_repl_stdout
        } else {
            last_assistant_content
        };
        let final_error = if final_answer.trim().is_empty() {
            Some("No final answer produced (empty completion)".to_string())
        } else {
            None
        };

        TaskResult {
            task_id: task.task_id.clone(),
            final_answer,
            tool_calls,
            mode_name: "RLM".to_string(),
            context_tokens: total_context_tokens,
            total_llm_tokens: total_llm_tokens_final,
            latency_ms: start.elapsed().as_millis() as u64,
            llm_time_ms,
            tool_time_ms,
            setup_ms,
            llm_total_calls,
            llm_input_tokens: llm_input_tokens + sub_input,
            llm_output_tokens: llm_output_tokens + sub_output,
            error: final_error,
        }
    }

    /// Execute a task in RLM+CodeMode: Python REPL with Gateway tool execution.
    ///
    /// Like `run_task_rlm`, this uses a persistent Python REPL with iterative
    /// LLM loop and sub-LLM calls. The key difference is that the REPL also
    /// has `execute_code()` for calling Gateway tools via EXECUTE_CODE, and
    /// `corpus_search()` / `corpus_get_chunk()` for accessing large results
    /// that were auto-stored in the corpus.
    pub async fn run_task_rlm_codemode(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let setup_t = Instant::now();

        // 1. Create RLM infrastructure
        let rlm_config = RlmConfig::default();
        let budget = Arc::new(BudgetManager::new(&rlm_config));

        let api_key = std::env::var("EVAL_API_KEY").unwrap_or_default();
        let base_url = std::env::var("EVAL_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty());

        // Create corpus + router for large result storage
        let temp_dir = match tempfile::TempDir::new() {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    task,
                    "CodeMode+RLM",
                    start,
                    &format!("Temp dir creation failed: {e}"),
                );
            }
        };
        let corpus =
            match crate::rlm::RlmCorpus::new(temp_dir.path().to_path_buf(), rlm_config.clone())
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    return error_result(
                        task,
                        "CodeMode+RLM",
                        start,
                        &format!("Corpus creation failed: {e}"),
                    );
                }
            };
        if let Err(e) = corpus.ingest_prompt(&task.prompt).await {
            return error_result(
                task,
                "CodeMode+RLM",
                start,
                &format!("Corpus ingest failed: {e}"),
            );
        }

        let corpus_arc = Arc::new(RwLock::new(Some(corpus)));
        let evidence_store = Arc::new(RwLock::new(EvidenceStore::new()));
        let router = Arc::new(GatewayResultRouter::new(
            corpus_arc,
            evidence_store,
            rlm_config,
        ));

        // 2. Create LM handler WITH Gateway proxy (mcp_client + rlm_router)
        let lm_config = LmHandlerConfig {
            root_model: self.agent_model.clone(),
            sub_model: None,
            api_key,
            base_url,
            mcp_client: Some(Arc::clone(&self.mcp_client)),
            rlm_router: Some(Arc::clone(&router)),
            tool_lookup: self.build_tool_lookup_map(),
        };

        let lm_handler = match LmHandler::start(lm_config, budget).await {
            Ok(h) => h,
            Err(e) => {
                return error_result(
                    task,
                    "CodeMode+RLM",
                    start,
                    &format!("LM handler start failed: {e}"),
                );
            }
        };

        // 3. Create REPL (will have execute_code, corpus_search, corpus_get_chunk available)
        let mut repl = match LocalRepl::new(lm_handler.port()).await {
            Ok(r) => r,
            Err(e) => {
                lm_handler.stop().await;
                return error_result(
                    task,
                    "CodeMode+RLM",
                    start,
                    &format!("REPL start failed: {e}"),
                );
            }
        };

        // 4. Inject context
        if let Err(e) = repl.set_context(&task.prompt).await {
            repl.cleanup().await;
            lm_handler.stop().await;
            return error_result(
                task,
                "CodeMode+RLM",
                start,
                &format!("Set context failed: {e}"),
            );
        }

        // 5. Build context metadata and system prompt with injected tool list
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: task.prompt.len(),
            context_lengths: vec![task.prompt.len()],
        };

        let tool_list_section = self.format_tool_list_for_codemode_prompt();
        let system_prompt = format!(
            "{}{}",
            build_rlm_codemode_system_prompt(&metadata),
            tool_list_section
        );
        let setup_ms = setup_t.elapsed().as_millis() as u64;
        let mut total_context_tokens: i64 = 0;
        let mut total_llm_tokens: i64 = 0;
        let mut tool_calls = Vec::new();
        let mut last_assistant_content = String::new();
        let mut last_repl_stdout = String::new();
        let mut llm_time_ms: u64 = 0;
        let mut tool_time_ms: u64 = 0;
        let mut llm_total_calls: u32 = 0;
        let mut llm_input_tokens: u64 = 0;
        let mut llm_output_tokens: u64 = 0;

        let mut messages: Vec<ChatCompletionRequestMessage> =
            vec![ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()
                    .unwrap(),
            )];

        let mut last_stdout_hash: u64 = 0;
        let mut consecutive_same_output: u32 = 0;
        let mut consecutive_empty: u32 = 0;

        // 6. Iterative REPL loop (same structure as run_task_rlm)
        for iteration in 0..self.max_turns {
            let user_prompt = build_rlm_user_prompt(iteration, self.max_turns, &task.prompt);
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_prompt.clone())
                    .build()
                    .unwrap(),
            ));

            // Call LLM (NO tools — free-form text with ```repl blocks)
            let request = match CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    repl.cleanup().await;
                    lm_handler.stop().await;
                    return error_result(
                        task,
                        "CodeMode+RLM",
                        start,
                        &format!("Request build failed: {e}"),
                    );
                }
            };

            let llm_t = Instant::now();
            let response = {
                let mut last_err = String::new();
                let mut resp = None;
                for attempt in 0..3u32 {
                    if attempt > 0 {
                        let delay = Duration::from_millis(1000 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                        tracing::warn!(
                            "  [rlm-code] retrying LLM call (attempt {}/3)...",
                            attempt + 1
                        );
                    }
                    match tokio::time::timeout(
                        Duration::from_secs(RLM_LLM_CALL_TIMEOUT_SECS),
                        self.chat_create(request.clone()),
                    )
                    .await
                    {
                        Ok(Ok(r)) => {
                            let has_content = r
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                                .map(|s| !s.is_empty())
                                .unwrap_or(false);
                            if has_content || attempt == 2 {
                                resp = Some(r);
                                break;
                            }
                            tracing::warn!(
                                "  [rlm-code] empty completion on attempt {}",
                                attempt + 1
                            );
                            resp = Some(r);
                        }
                        Ok(Err(e)) => {
                            last_err = format!("LLM call failed: {e}");
                            tracing::warn!("  [rlm-code] {last_err} (attempt {})", attempt + 1);
                        }
                        Err(_) => {
                            last_err =
                                format!("LLM call timed out after {RLM_LLM_CALL_TIMEOUT_SECS}s");
                            tracing::warn!("  [rlm-code] {last_err} (attempt {})", attempt + 1);
                        }
                    }
                }
                match resp {
                    Some(r) => r,
                    None => {
                        repl.cleanup().await;
                        lm_handler.stop().await;
                        return error_result(task, "CodeMode+RLM", start, &last_err);
                    }
                }
            };
            llm_time_ms += llm_t.elapsed().as_millis() as u64;
            llm_total_calls += 1;

            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
                total_llm_tokens += usage.total_tokens as i64;
                llm_input_tokens += usage.prompt_tokens as u64;
                llm_output_tokens += usage.completion_tokens as u64;
            }

            let content = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            last_assistant_content = content.clone();

            if content.is_empty() {
                tracing::warn!("  [rlm-code] empty LLM response on iteration {}", iteration);
                continue;
            }

            messages.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content.clone())
                    .build()
                    .unwrap(),
            ));

            // Check for FINAL answer
            if let Some(final_answer) = find_final_answer(&content) {
                let answer = resolve_final_answer_text(&mut repl, final_answer).await;
                if answer.trim().is_empty() {
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content("Your FINAL(...) was empty. Provide a non-empty final answer based on your computed results. If needed, run one more ```repl block to print the final values, then call FINAL(...) with actual content.".to_string())
                            .build()
                            .unwrap(),
                    ));
                    continue;
                }

                let sub_usage = lm_handler.usage().await;
                let total_llm_tokens_final =
                    total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
                let sub_input = sub_usage.total_input_tokens as u64;
                let sub_output = sub_usage.total_output_tokens as u64;
                repl.cleanup().await;
                lm_handler.stop().await;

                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: answer,
                    tool_calls,
                    mode_name: "CodeMode+RLM".to_string(),
                    context_tokens: total_context_tokens,
                    total_llm_tokens: total_llm_tokens_final,
                    latency_ms: start.elapsed().as_millis() as u64,
                    llm_time_ms,
                    tool_time_ms,
                    setup_ms,
                    llm_total_calls,
                    llm_input_tokens: llm_input_tokens + sub_input,
                    llm_output_tokens: llm_output_tokens + sub_output,
                    error: None,
                };
            }

            // Extract and execute code blocks
            let code_blocks = find_code_blocks(&content);
            if code_blocks.is_empty() && iteration > 0 {
                // No code blocks and no FINAL — if content is substantive, use it and break
                // to let force-final run; otherwise nudge the model to keep going
                if content.len() > 80 {
                    last_assistant_content = content;
                    break;
                }
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Your response had no code block and no FINAL(). You must either write a ```repl code block to continue working, or wrap your final answer with FINAL(your answer). If you have already computed the answer in a variable, write a ```repl block with print(your_variable) and then use FINAL().".to_string())
                        .build()
                        .unwrap(),
                ));
                continue;
            }

            for code in &code_blocks {
                let tool_t = Instant::now();
                let result = match repl.execute(code).await {
                    Ok(r) => r,
                    Err(e) => ReplResult {
                        stdout: String::new(),
                        stderr: format!("REPL execution error: {e}"),
                        locals_summary: String::new(),
                        execution_time_ms: 0,
                    },
                };
                let tool_latency = tool_t.elapsed().as_millis() as u64;
                tool_time_ms += tool_latency;

                // Track last non-empty REPL stdout for fallback answer
                if !result.stdout.trim().is_empty() {
                    last_repl_stdout = result.stdout.clone();
                }

                // Track consecutive identical output (Fix 5)
                {
                    let mut hasher = std::hash::DefaultHasher::new();
                    result.stdout.hash(&mut hasher);
                    let stdout_hash = hasher.finish();
                    if stdout_hash == last_stdout_hash && !result.stdout.trim().is_empty() {
                        consecutive_same_output += 1;
                    } else {
                        consecutive_same_output = 0;
                        last_stdout_hash = stdout_hash;
                    }
                }

                // Track consecutive empty output (Fix 6)
                if result.stdout.trim().is_empty() && result.stderr.trim().is_empty() {
                    consecutive_empty += 1;
                } else {
                    consecutive_empty = 0;
                }

                tool_calls.push(ToolCallRecord {
                    name: "repl_execute".to_string(),
                    arguments: json!({ "code": code }),
                    result: result.stdout.clone(),
                    result_tokens: estimate_tokens(&result.stdout),
                    stored_in_corpus: false,
                    latency_ms: tool_latency,
                });

                let feedback = format_repl_feedback(
                    code,
                    &result.stdout,
                    &result.stderr,
                    &result.locals_summary,
                    20_000,
                );

                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(feedback.clone())
                        .build()
                        .unwrap(),
                ));
            }

            // Fix 5: Break if same output repeats too many times
            if consecutive_same_output >= 3 {
                tracing::warn!(
                    "  [rlm-code] identical output {} consecutive times, breaking to force-final",
                    consecutive_same_output + 1
                );
                break;
            }

            // Fix 6: Nudge if too many consecutive empty outputs
            if consecutive_empty >= 5 {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("WARNING: Your last 5 code executions produced no output. Your code may have silent errors. Try adding print() statements to debug, or simplify your approach. Check that variable names are correct and that execute_tool() results are being captured properly.".to_string())
                        .build()
                        .unwrap(),
                ));
                consecutive_empty = 0;
            }

            // Fix 4: Detect final_answer variable assignment
            if code_blocks
                .iter()
                .any(|c| c.contains("final_answer =") || c.contains("final_answer="))
            {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Note: You assigned to a Python variable `final_answer` but did NOT call FINAL(). To submit your answer, write FINAL(your_answer) as plain text (outside a code block), or use FINAL_VAR(final_answer) to return the variable's value.".to_string())
                        .build()
                        .unwrap(),
                ));
            }
        }

        // Max iterations reached — force one more LLM call for final answer
        {
            let force_prompt = "You have reached the maximum number of iterations. Based on ALL the data and analysis from your previous steps, provide your final answer NOW. Use FINAL(your answer) to wrap it.".to_string();
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(force_prompt)
                    .build()
                    .unwrap(),
            ));

            if let Ok(request) = CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                let llm_t = Instant::now();
                if let Ok(response) = self.chat_create(request).await {
                    llm_time_ms += llm_t.elapsed().as_millis() as u64;
                    llm_total_calls += 1;
                    if let Some(usage) = &response.usage {
                        total_context_tokens = usage.total_tokens as i64;
                        total_llm_tokens += usage.total_tokens as i64;
                        llm_input_tokens += usage.prompt_tokens as u64;
                        llm_output_tokens += usage.completion_tokens as u64;
                    }
                    if let Some(content) = response
                        .choices
                        .first()
                        .and_then(|c| c.message.content.clone())
                    {
                        if !content.trim().is_empty() {
                            last_assistant_content = content;
                        }
                    }
                }
            }
        }

        // If force-final produced a useful answer, prefer it over stale REPL stdout
        if !last_assistant_content.trim().is_empty() {
            last_repl_stdout.clear();
        }

        let sub_usage = lm_handler.usage().await;
        let total_llm_tokens_final =
            total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
        let sub_input = sub_usage.total_input_tokens as u64;
        let sub_output = sub_usage.total_output_tokens as u64;
        repl.cleanup().await;
        lm_handler.stop().await;

        TaskResult {
            task_id: task.task_id.clone(),
            final_answer: if !last_repl_stdout.trim().is_empty() {
                last_repl_stdout
            } else {
                last_assistant_content
            },
            tool_calls,
            mode_name: "CodeMode+RLM".to_string(),
            context_tokens: total_context_tokens,
            total_llm_tokens: total_llm_tokens_final,
            latency_ms: start.elapsed().as_millis() as u64,
            llm_time_ms,
            tool_time_ms,
            setup_ms,
            llm_total_calls,
            llm_input_tokens: llm_input_tokens + sub_input,
            llm_output_tokens: llm_output_tokens + sub_output,
            error: None,
        }
    }

    /// RLM+Native mode: Python REPL with native tool wrappers and field projection.
    ///
    /// Combines RLM's iterative REPL with CodeMode's token efficiency by exposing
    /// Gateway tools as `tools.Server.method(fields=[...])` Python methods. The LLM
    /// stays in Python the entire time — no TypeScript embedding.
    pub async fn run_task_rlm_native(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let setup_t = Instant::now();

        // 1. Create RLM infrastructure (same as RlmCodeMode)
        let rlm_config = RlmConfig::default();
        let budget = Arc::new(BudgetManager::new(&rlm_config));

        let api_key = std::env::var("EVAL_API_KEY").unwrap_or_default();
        let base_url = std::env::var("EVAL_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty());

        // Create corpus + router so native mode shares the same bounded-runtime
        // guarantees as generic RLM.
        let temp_dir = match tempfile::TempDir::new() {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    task,
                    "RLM+Native",
                    start,
                    &format!("Temp dir creation failed: {e}"),
                );
            }
        };
        let corpus =
            match crate::rlm::RlmCorpus::new(temp_dir.path().to_path_buf(), rlm_config.clone())
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    return error_result(
                        task,
                        "RLM+Native",
                        start,
                        &format!("Corpus creation failed: {e}"),
                    );
                }
            };
        if let Err(e) = corpus.ingest_prompt(&task.prompt).await {
            return error_result(
                task,
                "RLM+Native",
                start,
                &format!("Corpus ingest failed: {e}"),
            );
        }
        let corpus_arc = Arc::new(RwLock::new(Some(corpus)));
        let evidence_store = Arc::new(RwLock::new(EvidenceStore::new()));
        let router = Arc::new(GatewayResultRouter::new(
            corpus_arc,
            evidence_store,
            rlm_config.clone(),
        ));

        // 2. Create LM handler with Gateway proxy and RLM router.
        let lm_config = LmHandlerConfig {
            root_model: self.agent_model.clone(),
            sub_model: None,
            api_key,
            base_url,
            mcp_client: Some(Arc::clone(&self.mcp_client)),
            rlm_router: Some(Arc::clone(&router)),
            tool_lookup: self.build_tool_lookup_map(),
        };

        let lm_handler = match LmHandler::start(lm_config, budget).await {
            Ok(h) => h,
            Err(e) => {
                return error_result(
                    task,
                    "RLM+Native",
                    start,
                    &format!("LM handler start failed: {e}"),
                );
            }
        };

        // 3. Create REPL
        let mut repl = match LocalRepl::new(lm_handler.port()).await {
            Ok(r) => r,
            Err(e) => {
                lm_handler.stop().await;
                return error_result(
                    task,
                    "RLM+Native",
                    start,
                    &format!("REPL start failed: {e}"),
                );
            }
        };

        // 4. Generate and inject Python tool wrappers into REPL
        let wrapper_code =
            super::pythonic_tools::generate_pythonic_tool_wrappers(&self.gateway_tools);
        if let Err(e) = repl.execute(&wrapper_code).await {
            repl.cleanup().await;
            lm_handler.stop().await;
            return error_result(
                task,
                "RLM+Native",
                start,
                &format!("Tool wrapper injection failed: {e}"),
            );
        }

        // 5. Inject context
        if let Err(e) = repl.set_context(&task.prompt).await {
            repl.cleanup().await;
            lm_handler.stop().await;
            return error_result(
                task,
                "RLM+Native",
                start,
                &format!("Set context failed: {e}"),
            );
        }

        // 6. Build system prompt with tool list
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: task.prompt.len(),
            context_lengths: vec![task.prompt.len()],
        };

        let tool_list_section =
            super::pythonic_tools::format_tool_list_for_native_prompt(&self.gateway_tools);
        let tools_dir_summary =
            super::pythonic_tools::generate_tools_dir_summary(&self.gateway_tools);
        let system_prompt = format!(
            "{}{}{}",
            build_rlm_native_system_prompt(&metadata),
            tool_list_section,
            tools_dir_summary
        );
        let setup_ms = setup_t.elapsed().as_millis() as u64;
        let mut total_context_tokens: i64 = 0;
        let mut total_llm_tokens: i64 = 0;
        let mut tool_calls = Vec::new();
        let mut last_assistant_content = String::new();
        let mut last_repl_stdout = String::new();
        let mut llm_time_ms: u64 = 0;
        let mut tool_time_ms: u64 = 0;
        let mut llm_total_calls: u32 = 0;
        let mut llm_input_tokens: u64 = 0;
        let mut llm_output_tokens: u64 = 0;

        let mut messages: Vec<ChatCompletionRequestMessage> =
            vec![ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()
                    .unwrap(),
            )];

        let mut last_stdout_hash: u64 = 0;
        let mut consecutive_same_output: u32 = 0;
        let mut consecutive_empty: u32 = 0;

        // 7. Iterative REPL loop (same structure as RlmCodeMode)
        for iteration in 0..self.max_turns {
            let user_prompt = build_rlm_user_prompt(iteration, self.max_turns, &task.prompt);
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_prompt.clone())
                    .build()
                    .unwrap(),
            ));

            // Call LLM (NO tools — free-form text with ```repl blocks)
            let request = match CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    repl.cleanup().await;
                    lm_handler.stop().await;
                    return error_result(
                        task,
                        "RLM+Native",
                        start,
                        &format!("Request build failed: {e}"),
                    );
                }
            };

            let llm_t = Instant::now();
            let response = {
                let mut last_err = String::new();
                let mut resp = None;
                for attempt in 0..3u32 {
                    if attempt > 0 {
                        let delay = Duration::from_millis(1000 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                        tracing::warn!(
                            "  [rlm-native] retrying LLM call (attempt {}/3)...",
                            attempt + 1
                        );
                    }
                    match tokio::time::timeout(
                        Duration::from_secs(RLM_LLM_CALL_TIMEOUT_SECS),
                        self.chat_create(request.clone()),
                    )
                    .await
                    {
                        Ok(Ok(r)) => {
                            let has_content = r
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                                .map(|s| !s.is_empty())
                                .unwrap_or(false);
                            if has_content || attempt == 2 {
                                resp = Some(r);
                                break;
                            }
                            tracing::warn!(
                                "  [rlm-native] empty completion on attempt {}",
                                attempt + 1
                            );
                            resp = Some(r);
                        }
                        Ok(Err(e)) => {
                            last_err = format!("LLM call failed: {e}");
                            tracing::warn!("  [rlm-native] {last_err} (attempt {})", attempt + 1);
                        }
                        Err(_) => {
                            last_err =
                                format!("LLM call timed out after {RLM_LLM_CALL_TIMEOUT_SECS}s");
                            tracing::warn!("  [rlm-native] {last_err} (attempt {})", attempt + 1);
                        }
                    }
                }
                match resp {
                    Some(r) => r,
                    None => {
                        repl.cleanup().await;
                        lm_handler.stop().await;
                        return error_result(task, "RLM+Native", start, &last_err);
                    }
                }
            };
            llm_time_ms += llm_t.elapsed().as_millis() as u64;
            llm_total_calls += 1;

            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
                total_llm_tokens += usage.total_tokens as i64;
                llm_input_tokens += usage.prompt_tokens as u64;
                llm_output_tokens += usage.completion_tokens as u64;
            }

            let content = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            last_assistant_content = content.clone();

            if content.is_empty() {
                tracing::warn!(
                    "  [rlm-native] empty LLM response on iteration {}",
                    iteration
                );
                continue;
            }

            messages.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content.clone())
                    .build()
                    .unwrap(),
            ));

            // Check for FINAL answer
            if let Some(final_answer) = find_final_answer(&content) {
                let answer = resolve_final_answer_text(&mut repl, final_answer).await;
                if answer.trim().is_empty() {
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content("Your FINAL(...) was empty. Provide a non-empty final answer based on your computed results. If needed, run one more ```repl block to print the final values, then call FINAL(...) with actual content.".to_string())
                            .build()
                            .unwrap(),
                    ));
                    continue;
                }

                let sub_usage = lm_handler.usage().await;
                let total_llm_tokens_final =
                    total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
                let sub_input = sub_usage.total_input_tokens as u64;
                let sub_output = sub_usage.total_output_tokens as u64;
                repl.cleanup().await;
                lm_handler.stop().await;

                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: answer,
                    tool_calls,
                    mode_name: "RLM+Native".to_string(),
                    context_tokens: total_context_tokens,
                    total_llm_tokens: total_llm_tokens_final,
                    latency_ms: start.elapsed().as_millis() as u64,
                    llm_time_ms,
                    tool_time_ms,
                    setup_ms,
                    llm_total_calls,
                    llm_input_tokens: llm_input_tokens + sub_input,
                    llm_output_tokens: llm_output_tokens + sub_output,
                    error: None,
                };
            }

            // Extract and execute code blocks
            let code_blocks = find_code_blocks(&content);
            if code_blocks.is_empty() && iteration > 0 {
                if content.len() > 80 {
                    last_assistant_content = content;
                    break;
                }
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Your response had no code block and no FINAL(). You must either write a ```repl code block to continue working, or wrap your final answer with FINAL(your answer). If you have already computed the answer in a variable, write a ```repl block with print(your_variable) and then use FINAL().".to_string())
                        .build()
                        .unwrap(),
                ));
                continue;
            }

            for code in &code_blocks {
                let tool_t = Instant::now();
                let result = match repl.execute(code).await {
                    Ok(r) => r,
                    Err(e) => ReplResult {
                        stdout: String::new(),
                        stderr: format!("REPL execution error: {e}"),
                        locals_summary: String::new(),
                        execution_time_ms: 0,
                    },
                };
                let tool_latency = tool_t.elapsed().as_millis() as u64;
                tool_time_ms += tool_latency;

                if !result.stdout.trim().is_empty() {
                    last_repl_stdout = result.stdout.clone();
                }

                // Track consecutive identical output (Fix 5)
                {
                    let mut hasher = std::hash::DefaultHasher::new();
                    result.stdout.hash(&mut hasher);
                    let stdout_hash = hasher.finish();
                    if stdout_hash == last_stdout_hash && !result.stdout.trim().is_empty() {
                        consecutive_same_output += 1;
                    } else {
                        consecutive_same_output = 0;
                        last_stdout_hash = stdout_hash;
                    }
                }

                // Track consecutive empty output (Fix 6)
                if result.stdout.trim().is_empty() && result.stderr.trim().is_empty() {
                    consecutive_empty += 1;
                } else {
                    consecutive_empty = 0;
                }

                tool_calls.push(ToolCallRecord {
                    name: "repl_execute".to_string(),
                    arguments: json!({ "code": code }),
                    result: result.stdout.clone(),
                    result_tokens: estimate_tokens(&result.stdout),
                    stored_in_corpus: false,
                    latency_ms: tool_latency,
                });

                let feedback = format_repl_feedback(
                    code,
                    &result.stdout,
                    &result.stderr,
                    &result.locals_summary,
                    20_000,
                );

                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(feedback.clone())
                        .build()
                        .unwrap(),
                ));
            }

            // Fix 5: Break if same output repeats too many times
            if consecutive_same_output >= 3 {
                tracing::warn!(
                    "  [rlm-native] identical output {} consecutive times, breaking to force-final",
                    consecutive_same_output + 1
                );
                break;
            }

            // Fix 6: Nudge if too many consecutive empty outputs
            if consecutive_empty >= 5 {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("WARNING: Your last 5 code executions produced no output. Your code may have silent errors. Try adding print() statements to debug, or simplify your approach. Check that variable names are correct and that execute_tool() results are being captured properly.".to_string())
                        .build()
                        .unwrap(),
                ));
                consecutive_empty = 0;
            }

            // Fix 4: Detect final_answer variable assignment
            if code_blocks
                .iter()
                .any(|c| c.contains("final_answer =") || c.contains("final_answer="))
            {
                messages.push(ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Note: You assigned to a Python variable `final_answer` but did NOT call FINAL(). To submit your answer, write FINAL(your_answer) as plain text (outside a code block), or use FINAL_VAR(final_answer) to return the variable's value.".to_string())
                        .build()
                        .unwrap(),
                ));
            }
        }

        // Max iterations reached — force one more LLM call for final answer
        {
            let force_prompt = "You have reached the maximum number of iterations. Based on ALL the data and analysis from your previous steps, provide your final answer NOW. Use FINAL(your answer) to wrap it.".to_string();
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(force_prompt)
                    .build()
                    .unwrap(),
            ));

            if let Ok(request) = CreateChatCompletionRequestArgs::default()
                .model(&self.agent_model)
                .messages(messages.clone())
                .build()
            {
                let llm_t = Instant::now();
                if let Ok(response) = self.chat_create(request).await {
                    llm_time_ms += llm_t.elapsed().as_millis() as u64;
                    llm_total_calls += 1;
                    if let Some(usage) = &response.usage {
                        total_context_tokens = usage.total_tokens as i64;
                        total_llm_tokens += usage.total_tokens as i64;
                        llm_input_tokens += usage.prompt_tokens as u64;
                        llm_output_tokens += usage.completion_tokens as u64;
                    }
                    if let Some(content) = response
                        .choices
                        .first()
                        .and_then(|c| c.message.content.clone())
                    {
                        if !content.trim().is_empty() {
                            last_assistant_content = content;
                        }
                    }
                }
            }
        }

        // If force-final produced a useful answer, prefer it over stale REPL stdout
        if !last_assistant_content.trim().is_empty() {
            last_repl_stdout.clear();
        }

        // Max iterations reached without FINAL()
        let sub_usage = lm_handler.usage().await;
        let total_llm_tokens_final =
            total_llm_tokens + sub_usage.total_input_tokens + sub_usage.total_output_tokens;
        let sub_input = sub_usage.total_input_tokens as u64;
        let sub_output = sub_usage.total_output_tokens as u64;
        repl.cleanup().await;
        lm_handler.stop().await;

        let final_answer = if !last_repl_stdout.trim().is_empty() {
            last_repl_stdout
        } else {
            last_assistant_content
        };
        let final_error = if final_answer.trim().is_empty() {
            Some("No final answer produced (empty completion)".to_string())
        } else {
            None
        };

        TaskResult {
            task_id: task.task_id.clone(),
            final_answer,
            tool_calls,
            mode_name: "RLM+Native".to_string(),
            context_tokens: total_context_tokens,
            total_llm_tokens: total_llm_tokens_final,
            latency_ms: start.elapsed().as_millis() as u64,
            llm_time_ms,
            tool_time_ms,
            setup_ms,
            llm_total_calls,
            llm_input_tokens: llm_input_tokens + sub_input,
            llm_output_tokens: llm_output_tokens + sub_output,
            error: final_error,
        }
    }

    /// Execute tool with RLM routing
    async fn execute_with_rlm(
        &self,
        tool_id: &str,
        args: &serde_json::Value,
        task_id: &str,
        call_id: &str,
    ) -> ToolCallResult {
        // Gateway's EXECUTE_TOOL expects "tool_arguments" not "args"
        let execute_args = json!({
            "tool_id": tool_id,
            "tool_arguments": args
        });

        let result = match self
            .mcp_client
            .call_tool(
                "EXECUTE_TOOL".to_string(),
                Some(execute_args),
                Some(Duration::from_secs(120)),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return ToolCallResult {
                    content: format!("Tool call failed: {e}"),
                    tokens: 10,
                    stored_in_corpus: false,
                };
            }
        };

        let content = serde_json::to_string(&result).unwrap_or_default();

        // Route through RLM if available
        if let Some(ref router) = self.rlm_router {
            let evidence_id = format!("{task_id}_{call_id}");
            match router
                .process_result(&evidence_id, "kontext-dev", tool_id, &content)
                .await
            {
                Ok(ProcessedResult::StoredInCorpus { summary, .. }) => {
                    // Large result stored - return summary
                    ToolCallResult {
                        content: summary,
                        tokens: 100, // Summary is small
                        stored_in_corpus: true,
                    }
                }
                Ok(ProcessedResult::PassThrough { content }) => {
                    // Small result passed through
                    let tokens = estimate_tokens(&content);
                    ToolCallResult {
                        content,
                        tokens,
                        stored_in_corpus: false,
                    }
                }
                Err(_) => {
                    // RLM error - fall back to full content
                    let tokens = estimate_tokens(&content);
                    ToolCallResult {
                        content,
                        tokens,
                        stored_in_corpus: false,
                    }
                }
            }
        } else {
            // No RLM - return full content
            let tokens = estimate_tokens(&content);
            ToolCallResult {
                content,
                tokens,
                stored_in_corpus: false,
            }
        }
    }
}

/// Result from executing a tool call
struct ToolCallResult {
    content: String,
    tokens: i64,
    stored_in_corpus: bool,
}

fn is_bare_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

async fn resolve_final_answer_text(repl: &mut LocalRepl, final_answer: FinalAnswer) -> String {
    match final_answer {
        FinalAnswer::Variable(name) => repl
            .resolve_var(&name)
            .await
            .unwrap_or_else(|e| format!("Failed to resolve variable '{}': {}", name, e)),
        FinalAnswer::Direct(text) => {
            let trimmed = text.trim();
            // Robustness for common model slip: FINAL(report) instead of FINAL_VAR(report).
            if is_bare_identifier(trimmed) {
                if let Ok(value) = repl.resolve_var(trimmed).await {
                    return value;
                }
            }
            text
        }
    }
}

/// Build an error TaskResult for early-exit paths in run_task_rlm.
fn error_result(task: &McpAtlasTask, mode: &str, start: Instant, msg: &str) -> TaskResult {
    TaskResult {
        task_id: task.task_id.clone(),
        final_answer: String::new(),
        tool_calls: Vec::new(),
        mode_name: mode.to_string(),
        context_tokens: 0,
        total_llm_tokens: 0,
        latency_ms: start.elapsed().as_millis() as u64,
        llm_time_ms: 0,
        tool_time_ms: 0,
        setup_ms: 0,
        llm_total_calls: 0,
        llm_input_tokens: 0,
        llm_output_tokens: 0,
        error: Some(msg.to_string()),
    }
}

/// Estimate tokens from content (rough: 4 chars per token)
fn estimate_tokens(content: &str) -> i64 {
    (content.len() / 4) as i64
}

/// Extract the code's return value from an EXECUTE_CODE MCP response.
///
/// The Gateway returns: MCP envelope → `content[0].resource.text` (JSON string)
/// → `{ result, logs?, error? }`.  We parse through the MCP envelope and
/// return a compact string with just the `result` field.
fn extract_execute_code_result(mcp_response: &rmcp::model::CallToolResult) -> String {
    let raw = serde_json::to_string(mcp_response).unwrap_or_default();

    // Parse outer MCP envelope → content[0].resource.text or content[0].text
    let text = (|| -> Option<String> {
        let outer: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let content = outer.get("content")?.as_array()?;
        let first = content.first()?;
        first
            .get("resource")
            .and_then(|r| r.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| first.get("text").and_then(|t| t.as_str()))
            .map(|s| s.to_string())
    })();

    let Some(inner_json) = text else {
        return raw;
    };

    // Parse the inner { result, logs?, error? } object
    let Ok(inner) = serde_json::from_str::<serde_json::Value>(&inner_json) else {
        return inner_json;
    };

    // Log execution details for debugging
    if let Some(error) = inner.get("error").and_then(|e| e.as_str()) {
        tracing::warn!("  [codemode] VM error: {}", error);
    }
    if let Some(logs) = inner.get("logs").and_then(|l| l.as_array()) {
        for log in logs.iter().take(10) {
            if let Some(s) = log.as_str() {
                tracing::warn!("  [codemode] VM log: {}", &s[..s.len().min(200)]);
            }
        }
    }

    // Extract "result" — this is the code's return value
    if let Some(result) = inner.get("result") {
        if result.is_null() {
            // null result with error means execution failed
            if let Some(error) = inner.get("error").and_then(|e| e.as_str()) {
                return format!("Code execution error: {error}");
            }
        }
        if let Some(s) = result.as_str() {
            return s.to_string();
        }
        return serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());
    }

    // Fallback: return the inner JSON
    inner_json
}

/// Extract TypeScript/JavaScript code from markdown code blocks
fn extract_code_block(content: &str) -> Option<String> {
    // Match ```typescript, ```ts, ```javascript, ```js, or plain ```
    // Allow flexible whitespace - newline is optional after language identifier
    let re = Regex::new(r"```(?:typescript|ts|javascript|js)?[\s]*([\s\S]*?)```").unwrap();
    re.captures(content)
        .map(|c| c.get(1).unwrap().as_str().trim().to_string())
        .filter(|s| !s.is_empty()) // Don't return empty code blocks
}

/// Unwrap async arrow functions into bare statements for the Gateway VM.
///
/// The Gateway VM uses `new AsyncFunction("tools", "console", code)` where
/// `code` is the **body** of an async function. If the LLM writes
/// `async () => { body }`, the body alone is what the VM needs — the async
/// arrow wrapper is redundant and would just create an uncalled function.
///
/// Examples:
/// - `async () => { const x = 1; return x; }` → `const x = 1; return x;`
/// - `async()=>{ return 42; }` → `return 42;`
/// - `const x = await foo();` → unchanged (already bare statements)
fn unwrap_async_arrow(code: &str) -> String {
    let trimmed = code.trim();

    // Check for async arrow prefix variants
    let body_start = if trimmed.starts_with("async () =>") {
        Some("async () =>".len())
    } else if trimmed.starts_with("async() =>") {
        Some("async() =>".len())
    } else if trimmed.starts_with("async ()=>") {
        Some("async ()=>".len())
    } else if trimmed.starts_with("async()=>") {
        Some("async()=>".len())
    } else {
        None
    };

    let Some(start) = body_start else {
        return trimmed.to_string();
    };

    let rest = trimmed[start..].trim();

    // Must be a block body: `{ ... }`
    if rest.starts_with('{') && rest.ends_with('}') {
        // Extract the body between the outer braces
        rest[1..rest.len() - 1].trim().to_string()
    } else {
        // Expression body: `async () => expr` — wrap with return
        format!("return {rest};")
    }
}

/// Normalize TypeScript/JavaScript code for execution in the Gateway VM.
///
/// Combines `unwrap_async_arrow` (unwrap `async () => { body }` into bare
/// statements) with stripping of TypeScript-specific syntax (type annotations
/// like `: Type` and `as Type` casts) that the Gateway VM cannot handle.
fn normalize_code_for_async_fn(code: &str) -> String {
    let unwrapped = unwrap_async_arrow(code);

    // Strip TypeScript type annotations: `: Type` after identifiers (but not in strings)
    let re_type_ann =
        Regex::new(r"(?m)(const|let|var)\s+(\w+)\s*:\s*[\w\[\]<>,\s|&]+\s*=").unwrap();
    let stripped = re_type_ann.replace_all(&unwrapped, "$1 $2 =");

    // Strip `as Type` casts
    let re_as_cast = Regex::new(r"\s+as\s+\w+").unwrap();
    let result = re_as_cast.replace_all(&stripped, "");

    result.to_string()
}

/// Discover available tools from Gateway using SEARCH_TOOLS
async fn discover_gateway_tools(client: &RmcpClient) -> Result<Vec<GatewayTool>> {
    // Call SEARCH_TOOLS with empty query to get all tools
    let result = client
        .call_tool(
            "SEARCH_TOOLS".to_string(),
            Some(json!({"query": ""})),
            Some(Duration::from_secs(60)),
        )
        .await
        .with_context(|| "Failed to call SEARCH_TOOLS")?;

    // Parse the response
    let response_str = serde_json::to_string(&result).unwrap_or_default();
    let response_val: serde_json::Value = serde_json::from_str(&response_str)
        .with_context(|| "Failed to parse SEARCH_TOOLS response")?;

    // Navigate to content[0].resource.text. Gateways may return either a raw array
    // (`[...]`) or an envelope (`{"items":[...],"errors":[...]}`).
    let tools_json = response_val
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("resource"))
        .and_then(|r| r.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("[]");

    let tools_value: serde_json::Value =
        serde_json::from_str(tools_json).with_context(|| "Failed to parse tools payload")?;

    // Surface servers that need OAuth re-auth
    if let Some(errors) = tools_value.get("errors").and_then(|v| v.as_array()) {
        for err in errors {
            let server = err
                .get("serverName")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let reason = err
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            tracing::warn!("Gateway: server {server} not available ({reason})");
        }
    }

    let tools_array: Vec<serde_json::Value> = if let Some(items) = tools_value.as_array() {
        items.clone()
    } else if let Some(items) = tools_value.get("items").and_then(|v| v.as_array()) {
        items.clone()
    } else if let Some(items) = tools_value.get("tools").and_then(|v| v.as_array()) {
        items.clone()
    } else {
        Vec::new()
    };

    let mut gateway_tools = Vec::new();

    for tool_val in tools_array {
        let id = tool_val
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let server = tool_val
            .get("server")
            .and_then(|s| s.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown")
            .to_string();
        let name = tool_val
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        let description = tool_val
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let input_schema = tool_val
            .get("inputSchema")
            .cloned()
            .unwrap_or(json!({"type": "object", "properties": {}}));

        if !name.is_empty() {
            gateway_tools.push(GatewayTool {
                id,
                server,
                name,
                description,
                input_schema,
            });
        }
    }

    Ok(gateway_tools)
}

/// Create an RLM router for evaluation
pub async fn create_rlm_router(temp_path: &std::path::Path) -> Result<GatewayResultRouter> {
    let rlm_config = RlmConfig::default();
    let corpus = RlmCorpus::new(temp_path.to_path_buf(), rlm_config.clone())
        .await
        .with_context(|| "Failed to create RLM corpus")?;

    corpus
        .ingest_prompt("MCP-Atlas evaluation corpus.")
        .await
        .with_context(|| "Failed to initialize corpus")?;

    let router = GatewayResultRouter::new(
        Arc::new(RwLock::new(Some(corpus))),
        Arc::new(RwLock::new(EvidenceStore::new())),
        rlm_config,
    );

    Ok(router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_calling_mode_display() {
        assert_eq!(format!("{}", ToolCallingMode::Baseline), "Baseline");
        assert_eq!(format!("{}", ToolCallingMode::CodeMode), "CodeMode");
        assert_eq!(format!("{}", ToolCallingMode::BaselineRlm), "Baseline+RLM");
        assert_eq!(format!("{}", ToolCallingMode::Rlm), "RLM");
        assert_eq!(format!("{}", ToolCallingMode::RlmCodeMode), "CodeMode+RLM");
        assert_eq!(format!("{}", ToolCallingMode::RlmNative), "RLM+Native");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("1234"), 1);
        assert_eq!(estimate_tokens("12345678"), 2);
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_is_bare_identifier() {
        assert!(is_bare_identifier("report"));
        assert!(is_bare_identifier("_final_answer_2"));
        assert!(!is_bare_identifier("final answer"));
        assert!(!is_bare_identifier("42"));
        assert!(!is_bare_identifier("'report'"));
    }

    #[test]
    fn test_normalize_bare_statements_passthrough() {
        let code = r#"const projects = await codemode.Linear_list_projects({});
return { count: projects.length };"#;
        let result = normalize_code_for_async_fn(code);
        // Bare statements should be used as-is (no IIFE wrapping)
        assert!(result.contains("await codemode.Linear_list_projects"));
        assert!(result.contains("return { count: projects.length }"));
        // Should NOT be wrapped in IIFE
        assert!(!result.contains("(async () =>"));
    }

    #[test]
    fn test_normalize_async_arrow_body_extraction() {
        let code = r#"async () => {
  const projects = await codemode.Linear_list_projects({});
  return { count: projects.length };
}"#;
        let result = normalize_code_for_async_fn(code);
        // Should extract the body, not wrap in IIFE
        assert!(result.contains("await codemode.Linear_list_projects"));
        assert!(result.contains("return { count: projects.length }"));
        // Should NOT contain async () => or IIFE wrapping
        assert!(!result.contains("async () =>"));
        assert!(!result.contains("})()"));
    }

    #[test]
    fn test_normalize_async_arrow_expression() {
        let code = "async () => 42";
        let result = normalize_code_for_async_fn(code);
        // unwrap_async_arrow adds a semicolon: `return 42;`
        assert_eq!(result, "return 42;");
    }

    #[test]
    fn test_normalize_async_arrow_nested_braces() {
        let code = r#"async () => {
  const data = { a: 1, b: { c: 2 } };
  return data;
}"#;
        let result = normalize_code_for_async_fn(code);
        assert!(result.contains("const data = { a: 1, b: { c: 2 } }"));
        assert!(result.contains("return data;"));
        assert!(!result.contains("async () =>"));
    }

    #[test]
    fn test_normalize_strips_ts_type_annotations() {
        let code =
            "const projects: any[] = await codemode.Linear_list_projects({});\nreturn projects;";
        let result = normalize_code_for_async_fn(code);
        // Should not contain type annotation
        assert!(!result.contains(": any[]"));
        // Should still have the variable declaration
        assert!(result.contains("const projects =") || result.contains("const projects="));
    }

    #[test]
    fn test_normalize_strips_as_cast() {
        let code = "const data = result as unknown;\nreturn data;";
        let result = normalize_code_for_async_fn(code);
        assert!(!result.contains("as unknown"));
    }

    #[test]
    fn test_extract_code_block_javascript() {
        let content =
            "Here is the code:\n```javascript\nasync () => {\n  return 42;\n}\n```\nDone.";
        let block = extract_code_block(content);
        assert!(block.is_some());
        assert!(block.unwrap().contains("return 42"));
    }

    #[test]
    fn test_extract_code_block_typescript() {
        let content = "```typescript\nconst x = 1;\n```";
        let block = extract_code_block(content);
        assert!(block.is_some());
        assert!(block.unwrap().contains("const x = 1"));
    }
}
