//! Three-Mode Task Runner
//!
//! Executes MCP-Atlas tasks in three different modes:
//! - Baseline: Direct tool calls with full results in context
//! - CodeMode: Tool calls via EXECUTE_CODE with summarized results
//! - BaselineRlm: Tool calls with RLM routing for large results

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
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
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::FunctionObjectArgs;
use async_openai::Client;
use codex_rmcp_client::RmcpClient;
use regex::Regex;
use serde_json::json;
use tokio::sync::RwLock;

use super::McpAtlasTask;
use crate::rlm::EvidenceStore;
use crate::rlm::GatewayResultRouter;
use crate::rlm::ProcessedResult;
use crate::rlm::RlmConfig;
use crate::rlm::RlmCorpus;

/// Execution mode for task running
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    /// Direct EXECUTE_TOOL calls with full results in context
    Baseline,
    /// EXECUTE_CODE with summarized results
    CodeMode,
    /// EXECUTE_TOOL with RLM routing for large results
    BaselineRlm,
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Baseline => write!(f, "Baseline"),
            ExecutionMode::CodeMode => write!(f, "CodeMode"),
            ExecutionMode::BaselineRlm => write!(f, "Baseline+RLM"),
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
    /// Execution mode used
    pub mode: ExecutionMode,
    /// Total context tokens used
    pub context_tokens: i64,
    /// Execution latency in milliseconds
    pub latency_ms: u64,
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

/// Task runner that supports three execution modes
pub struct TaskRunner {
    /// MCP client for Gateway calls
    mcp_client: Arc<RmcpClient>,
    /// OpenAI client for agent LLM
    openai_client: Client<OpenAIConfig>,
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

impl TaskRunner {
    /// Create a new task runner and discover available tools from Gateway
    pub async fn new(
        mcp_client: Arc<RmcpClient>,
        rlm_router: Option<Arc<GatewayResultRouter>>,
    ) -> Result<Self> {
        let config = OpenAIConfig::default();
        let openai_client = Client::with_config(config);

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
            openai_client,
            rlm_router,
            agent_model: "gpt-4o".to_string(),
            max_turns: 10,
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

    /// Execute a task in the specified mode
    pub async fn run_task(&self, task: &McpAtlasTask, mode: ExecutionMode) -> TaskResult {
        // Dispatch to mode-specific runner for CodeMode
        if mode == ExecutionMode::CodeMode {
            return self.run_task_codemode(task).await;
        }

        // Baseline and BaselineRlm use the standard tool-call approach
        let start = Instant::now();
        let mut tool_calls = Vec::new();
        let mut total_context_tokens: i64 = 0;

        // Build tool definitions - include RLM tools for BaselineRlm mode
        let tools = if mode == ExecutionMode::BaselineRlm {
            self.build_tool_definitions_for_rlm()
        } else {
            self.build_tool_definitions_from_gateway()
        };

        // Build tool list for system prompt
        let tool_list = self
            .gateway_tools
            .iter()
            .map(|t| format!("- {}: {} ({})", t.name, t.description, t.server))
            .collect::<Vec<_>>()
            .join("\n");

        // System prompt for the agent with discovered tools
        let system_prompt = if mode == ExecutionMode::BaselineRlm {
            format!(
                r#"You are an AI assistant executing tasks using available tools.
Complete the task by making appropriate tool calls. When you have gathered
all necessary information, provide a final answer.

# Available Tools

{}

# Corpus Access Tools (for large results)

When tool results are too large, they are stored in a corpus and you'll see a summary.
Use these tools to access the stored data:

- rlm_search(query): Search stored results by semantic query
- rlm_get_chunk(chunk_id): Retrieve a specific chunk by ID
- rlm_list_chunks(): List all stored chunks with summaries

# Instructions

1. Analyze the task and determine which tools to use
2. Make tool calls to gather information
3. If you receive a "stored in corpus" message, use rlm_search or rlm_get_chunk to access the data
4. When you have enough information, provide your final answer directly

Note: Tool calls are executed via a Gateway. Use the exact tool names shown above."#,
                tool_list
            )
        } else {
            format!(
                r#"You are an AI assistant executing tasks using available tools.
Complete the task by making appropriate tool calls. When you have gathered
all necessary information, provide a final answer.

# Available Tools

{}

# Instructions

1. Analyze the task and determine which tools to use
2. Make tool calls to gather information
3. When you have enough information, provide your final answer directly

Note: Tool calls are executed via a Gateway. Use the exact tool names shown above."#,
                tool_list
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

        // Estimate initial context
        total_context_tokens += estimate_tokens(&system_prompt);
        total_context_tokens += estimate_tokens(&task.prompt);

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
                        mode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("Failed to build request: {}", e)),
                    };
                }
            };

            // Call the agent LLM
            let response = match self.openai_client.chat().create(request).await {
                Ok(r) => r,
                Err(e) => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("Agent LLM call failed: {}", e)),
                    };
                }
            };

            // Track token usage
            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
            }

            let choice = match response.choices.first() {
                Some(c) => c,
                None => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some("No response from agent".to_string()),
                    };
                }
            };

            // Check if there are tool calls
            if let Some(ref calls) = choice.message.tool_calls {
                if !calls.is_empty() {
                    // Process each tool call
                    let mut tool_results = Vec::new();

                    for call in calls {
                        let result = self.execute_tool_call(call, mode, &task.task_id).await;

                        // Track the call
                        let record = ToolCallRecord {
                            name: call.function.name.clone(),
                            arguments: serde_json::from_str(&call.function.arguments)
                                .unwrap_or(json!({})),
                            result: result.content.clone(),
                            result_tokens: result.tokens,
                            stored_in_corpus: result.stored_in_corpus,
                        };
                        tool_calls.push(record);

                        // Add to context unless stored in corpus
                        if !result.stored_in_corpus {
                            total_context_tokens += result.tokens;
                        }

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
            }

            // No tool calls - this is the final answer
            let final_answer = choice.message.content.clone().unwrap_or_default();

            return TaskResult {
                task_id: task.task_id.clone(),
                final_answer,
                tool_calls,
                mode,
                context_tokens: total_context_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                error: None,
            };
        }

        // Max turns reached
        TaskResult {
            task_id: task.task_id.clone(),
            final_answer: format!("Max turns ({}) reached without final answer", self.max_turns),
            tool_calls,
            mode,
            context_tokens: total_context_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
            error: Some("Max turns reached".to_string()),
        }
    }

    /// Build tool definitions for OpenAI from discovered Gateway tools
    fn build_tool_definitions_from_gateway(&self) -> Vec<ChatCompletionTool> {
        self.gateway_tools
            .iter()
            .map(|tool| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(
                        FunctionObjectArgs::default()
                            .name(tool.name.clone())
                            .description(format!("{} ({})", tool.description, tool.server))
                            .parameters(tool.input_schema.clone())
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap()
            })
            .collect()
    }

    /// Build tool definitions for RLM mode (includes corpus access tools)
    fn build_tool_definitions_for_rlm(&self) -> Vec<ChatCompletionTool> {
        let mut tools = self.build_tool_definitions_from_gateway();

        // Add rlm_search tool
        tools.push(
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name("rlm_search")
                        .description("Search stored tool results by semantic query. Use this to find specific data from large results that were stored.")
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
                        .description("List all stored chunks with their summaries. Use this to see what data is available.")
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
        task.enabled_tools.iter().any(|tool_name| self.resolve_tool(tool_name).is_some())
    }

    /// Get matching tools for a task
    pub fn get_matching_tools<'a>(&'a self, task: &'a McpAtlasTask) -> Vec<(&'a str, &'a GatewayTool)> {
        task.enabled_tools
            .iter()
            .filter_map(|tool_name| {
                self.resolve_tool(tool_name).map(|t| (tool_name.as_str(), t))
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
                            gateway_tool.server.to_lowercase().contains(&s.to_lowercase())
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
                    && task.enabled_tools.iter().all(|t| self.resolve_tool(t).is_some())
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
        mode: ExecutionMode,
        task_id: &str,
    ) -> ToolCallResult {
        let tool_name = &call.function.name;
        let args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));

        // Handle RLM-specific tools in BaselineRlm mode
        if mode == ExecutionMode::BaselineRlm {
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
            ExecutionMode::Baseline => self.execute_baseline(&tool_id, &args).await,
            ExecutionMode::CodeMode => self.execute_codemode(&tool_id, &args).await,
            ExecutionMode::BaselineRlm => {
                self.execute_with_rlm(&tool_id, &args, task_id, &call.id)
                    .await
            }
        }
    }

    /// Execute rlm_search tool
    async fn execute_rlm_search(&self, args: &serde_json::Value) -> ToolCallResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

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
        let chunk_id = args
            .get("chunk_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

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
                    content: format!("Chunk '{}' not found", chunk_id),
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
                content: format!("Tool call failed: {}", e),
                tokens: 10,
                stored_in_corpus: false,
            },
        }
    }

    /// Execute tool in code mode - call EXECUTE_CODE and return FULL result
    /// The server runs the code in a VM and returns complete tool output
    async fn execute_codemode(&self, tool_id: &str, args: &serde_json::Value) -> ToolCallResult {
        // Generate code that calls the tool and returns full result (not a summary!)
        // Gateway's EXECUTE_TOOL expects "tool_arguments" not "args"
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
                content: format!("Code execution failed: {}", e),
                tokens: 10,
                stored_in_corpus: false,
            },
        }
    }

    /// Build system prompt for CodeMode - instructs LLM to generate TypeScript code
    fn build_codemode_system_prompt(&self) -> String {
        // Build tool list with actual tool IDs for accurate code generation
        let tool_list = self
            .gateway_tools
            .iter()
            .map(|t| format!("- tool_id: \"{}\" - {} ({})", t.id, t.description, t.server))
            .collect::<Vec<_>>()
            .join("\n");

        // Get a sample tool ID for the example - build outside raw string for safety
        let sample_tool_id = self
            .gateway_tools
            .iter()
            .find(|t| t.name == "list_projects" || t.name.contains("list"))
            .map(|t| t.id.clone())
            .unwrap_or_else(|| "Linear:list_projects".to_string());

        // Build the example code block with actual tool ID substituted
        let example_code = format!(
            r#"const result = await tools.EXECUTE_TOOL({{
    tool_id: "{}",
    tool_arguments: {{}}
}});
console.log(JSON.stringify(result, null, 2));
return result;"#,
            sample_tool_id
        );

        format!(
            r#"You are an AI assistant. Complete tasks using TypeScript code.

# Available Tools

{tool_list}

# How to Call Tools

Write code in a ```typescript code block:

```typescript
{example_code}
```

# Workflow

1. First: Write TypeScript code to fetch data (in a code block)
2. After seeing results: Provide a PLAIN TEXT answer to the task

CRITICAL: After code execution, your FINAL response must be plain text that directly answers the task question. Do NOT respond with another code block - write a natural language answer using the data you fetched."#,
            tool_list = tool_list,
            example_code = example_code
        )
    }

    /// Execute code via EXECUTE_CODE tool and return full results
    async fn execute_code(&self, code: &str) -> String {
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
                // Return full result - server returns: { result, logs, toolCalls, durationMs }
                serde_json::to_string_pretty(&result).unwrap_or_default()
            }
            Err(e) => format!("Code execution failed: {}", e),
        }
    }

    /// Execute a task in CodeMode - LLM generates code, we execute it
    pub async fn run_task_codemode(&self, task: &McpAtlasTask) -> TaskResult {
        let start = Instant::now();
        let system_prompt = self.build_codemode_system_prompt();
        let mut total_context_tokens: i64 = 0;
        let mut tool_calls = Vec::new();

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

        total_context_tokens += estimate_tokens(&system_prompt);
        total_context_tokens += estimate_tokens(&task.prompt);

        // Agent loop
        for _turn in 0..self.max_turns {
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
                        mode: ExecutionMode::CodeMode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("Failed to build request: {}", e)),
                    };
                }
            };

            // Call the agent LLM
            let response = match self.openai_client.chat().create(request).await {
                Ok(r) => r,
                Err(e) => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode: ExecutionMode::CodeMode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("Agent LLM call failed: {}", e)),
                    };
                }
            };

            // Track token usage
            if let Some(usage) = &response.usage {
                total_context_tokens = usage.total_tokens as i64;
            }

            let choice = match response.choices.first() {
                Some(c) => c,
                None => {
                    return TaskResult {
                        task_id: task.task_id.clone(),
                        final_answer: String::new(),
                        tool_calls,
                        mode: ExecutionMode::CodeMode,
                        context_tokens: total_context_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some("No response from agent".to_string()),
                    };
                }
            };

            let content = choice.message.content.clone().unwrap_or_default();

            // Check if response contains code block
            if let Some(code) = extract_code_block(&content) {
                // Execute the code via EXECUTE_CODE
                let exec_result = self.execute_code(&code).await;

                tool_calls.push(ToolCallRecord {
                    name: "EXECUTE_CODE".to_string(),
                    arguments: json!({ "code": code }),
                    result: exec_result.clone(),
                    result_tokens: estimate_tokens(&exec_result),
                    stored_in_corpus: false,
                });

                total_context_tokens += estimate_tokens(&exec_result);

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
                            "Code execution result:\n```json\n{}\n```\n\nBased on this result, provide your final answer to the original task.",
                            exec_result
                        ))
                        .build()
                        .unwrap(),
                ));

                continue;
            }

            // No code block - this is the final answer
            return TaskResult {
                task_id: task.task_id.clone(),
                final_answer: content,
                tool_calls,
                mode: ExecutionMode::CodeMode,
                context_tokens: total_context_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                error: None,
            };
        }

        // Max turns reached
        TaskResult {
            task_id: task.task_id.clone(),
            final_answer: format!("Max turns ({}) reached without final answer", self.max_turns),
            tool_calls,
            mode: ExecutionMode::CodeMode,
            context_tokens: total_context_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
            error: Some("Max turns reached".to_string()),
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
                    content: format!("Tool call failed: {}", e),
                    tokens: 10,
                    stored_in_corpus: false,
                }
            }
        };

        let content = serde_json::to_string(&result).unwrap_or_default();

        // Route through RLM if available
        if let Some(ref router) = self.rlm_router {
            let evidence_id = format!("{}_{}", task_id, call_id);
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

/// Estimate tokens from content (rough: 4 chars per token)
fn estimate_tokens(content: &str) -> i64 {
    (content.len() / 4) as i64
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
    let response_val: serde_json::Value =
        serde_json::from_str(&response_str).with_context(|| "Failed to parse SEARCH_TOOLS response")?;

    // Navigate to content[0].resource.text which contains the JSON array
    let tools_json = response_val
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("resource"))
        .and_then(|r| r.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("[]");

    let tools_array: Vec<serde_json::Value> =
        serde_json::from_str(tools_json).with_context(|| "Failed to parse tools array")?;

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
    fn test_execution_mode_display() {
        assert_eq!(format!("{}", ExecutionMode::Baseline), "Baseline");
        assert_eq!(format!("{}", ExecutionMode::CodeMode), "CodeMode");
        assert_eq!(format!("{}", ExecutionMode::BaselineRlm), "Baseline+RLM");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("1234"), 1);
        assert_eq!(estimate_tokens("12345678"), 2);
        assert_eq!(estimate_tokens(""), 0);
    }
}
