//! LM Handler — lightweight HTTP server that proxies sub-LLM calls from the
//! Python REPL back to the OpenAI-compatible API, and optionally proxies
//! Gateway tool calls and corpus operations for RLM+CodeMode.
//!
//! The handler is started on `127.0.0.1:0` (OS-assigned port) and exposes:
//!
//! - `POST /llm_query`         — single prompt completion
//! - `POST /llm_query_batched` — batch of prompts (concurrent)
//! - `POST /execute_code`      — proxy to Gateway EXECUTE_CODE (optional)
//! - `POST /corpus_search`     — search RLM corpus (optional)
//! - `POST /corpus_get_chunk`  — retrieve corpus chunk (optional)
//! - `GET  /health`            — liveness check
//!
//! All endpoints exchange JSON.  The HTTP parsing is intentionally minimal
//! because only our own Python REPL talks to this server.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use async_openai::Client;
use codex_rmcp_client::RmcpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

use super::BudgetManager;
use super::GatewayResultRouter;
use super::ProcessedResult;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the LM Handler.
#[derive(Clone)]
pub struct LmHandlerConfig {
    /// Model name used for depth-0 (root) calls.
    pub root_model: String,
    /// Model name used for depth >= 1 (sub) calls.  Falls back to `root_model`
    /// when `None`.
    pub sub_model: Option<String>,
    /// API key for the OpenAI-compatible endpoint.
    pub api_key: String,
    /// Optional base URL override (e.g. for Azure / local proxies).
    pub base_url: Option<String>,
    /// Optional MCP client for proxying Gateway tool calls (RLM+CodeMode).
    pub mcp_client: Option<Arc<RmcpClient>>,
    /// Optional RLM router for large-result corpus management (RLM+CodeMode).
    pub rlm_router: Option<Arc<GatewayResultRouter>>,
    /// Tool name lookup: maps various name forms (fuzzy) to canonical Gateway tool IDs.
    /// Used by `/execute_tool` to resolve LLM-guessed tool names to real IDs.
    pub tool_lookup: HashMap<String, String>,
}

impl std::fmt::Debug for LmHandlerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmHandlerConfig")
            .field("root_model", &self.root_model)
            .field("sub_model", &self.sub_model)
            .field("base_url", &self.base_url)
            .field("mcp_client", &self.mcp_client.as_ref().map(|_| "<RmcpClient>"))
            .field("rlm_router", &self.rlm_router.as_ref().map(|_| "<GatewayResultRouter>"))
            .finish()
    }
}

/// Aggregated token usage across all calls handled by this server.
#[derive(Debug, Clone, Default)]
pub struct UsageSummary {
    pub total_calls: u32,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
}

/// A running LM Handler server.
pub struct LmHandler {
    port: u16,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<tokio::task::JoinHandle<()>>,
    usage: Arc<RwLock<UsageSummary>>,
}

// ---------------------------------------------------------------------------
// Wire types (JSON request/response bodies)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct QueryRequest {
    prompt: String,
    model: Option<String>,
    #[serde(default = "default_depth")]
    depth: u32,
}

fn default_depth() -> u32 {
    1
}

#[derive(Serialize)]
struct QueryResponse {
    response: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct BatchedRequest {
    prompts: Vec<String>,
    model: Option<String>,
    #[serde(default = "default_depth")]
    depth: u32,
}

#[derive(Serialize)]
struct BatchedResponse {
    responses: Option<Vec<String>>,
    error: Option<String>,
}

// -- Gateway proxy types (RLM+CodeMode) --

#[derive(Deserialize)]
struct ExecuteCodeRequest {
    code: String,
}

#[derive(Serialize)]
struct ExecuteCodeResponse {
    result: Option<String>,
    stored_in_corpus: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct CorpusSearchRequest {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    5
}

#[derive(Deserialize)]
struct CorpusGetChunkRequest {
    chunk_id: String,
}

#[derive(Deserialize)]
struct ExecuteToolRequest {
    tool_id: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Serialize)]
struct ExecuteToolResponse {
    result: Option<String>,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared state handed to every connection handler
// ---------------------------------------------------------------------------

struct SharedState {
    client: Client<OpenAIConfig>,
    root_model: String,
    sub_model: String,
    budget: Arc<BudgetManager>,
    usage: Arc<RwLock<UsageSummary>>,
    /// Optional MCP client for proxying EXECUTE_CODE to Gateway.
    mcp_client: Option<Arc<RmcpClient>>,
    /// Optional RLM router for large-result corpus routing.
    rlm_router: Option<Arc<GatewayResultRouter>>,
    /// Tool name lookup for resolving fuzzy names to canonical IDs.
    tool_lookup: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// LmHandler implementation
// ---------------------------------------------------------------------------

impl LmHandler {
    /// Start the HTTP server and return immediately.
    pub async fn start(config: LmHandlerConfig, budget: Arc<BudgetManager>) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .context("Failed to bind LM Handler TCP listener")?;
        let port = listener.local_addr()?.port();

        let mut oai_config = OpenAIConfig::default();
        if let Some(ref base) = config.base_url {
            oai_config = oai_config.with_api_base(base);
        }
        oai_config = oai_config.with_api_key(&config.api_key);
        let client = Client::with_config(oai_config);

        let sub_model = config
            .sub_model
            .clone()
            .unwrap_or_else(|| config.root_model.clone());

        let usage: Arc<RwLock<UsageSummary>> = Arc::new(RwLock::new(UsageSummary::default()));

        let state = Arc::new(SharedState {
            client,
            root_model: config.root_model.clone(),
            sub_model,
            budget,
            usage: Arc::clone(&usage),
            mcp_client: config.mcp_client.clone(),
            rlm_router: config.rlm_router.clone(),
            tool_lookup: config.tool_lookup.clone(),
        });

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let join_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        match accept {
                            Ok((stream, _addr)) => {
                                let st = Arc::clone(&state);
                                tokio::spawn(async move {
                                    if let Err(e) = handle_connection(stream, st).await {
                                        tracing::debug!("LmHandler connection error: {e:#}");
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::debug!("LmHandler accept error: {e}");
                            }
                        }
                    }
                    _ = &mut shutdown_rx => {
                        tracing::debug!("LmHandler shutting down");
                        break;
                    }
                }
            }
        });

        tracing::debug!("LmHandler listening on 127.0.0.1:{port}");

        Ok(Self {
            port,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
            usage,
        })
    }

    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Snapshot of the accumulated usage.
    pub async fn usage(&self) -> UsageSummary {
        self.usage.read().await.clone()
    }

    /// Gracefully stop the server.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.await;
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

async fn handle_connection(mut stream: TcpStream, state: Arc<SharedState>) -> Result<()> {
    let (method, path, body) = read_request(&mut stream).await?;

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            write_response(&mut stream, 200, b"{\"status\":\"ok\"}").await?;
        }
        ("POST", "/llm_query") => {
            let req: QueryRequest =
                serde_json::from_slice(&body).context("Invalid /llm_query JSON")?;
            let resp = handle_query(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        ("POST", "/llm_query_batched") => {
            let req: BatchedRequest =
                serde_json::from_slice(&body).context("Invalid /llm_query_batched JSON")?;
            let resp = handle_batched(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        ("POST", "/execute_code") => {
            let req: ExecuteCodeRequest =
                serde_json::from_slice(&body).context("Invalid /execute_code JSON")?;
            let resp = handle_execute_code(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        ("POST", "/corpus_search") => {
            let req: CorpusSearchRequest =
                serde_json::from_slice(&body).context("Invalid /corpus_search JSON")?;
            let resp = handle_corpus_search(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        ("POST", "/corpus_get_chunk") => {
            let req: CorpusGetChunkRequest =
                serde_json::from_slice(&body).context("Invalid /corpus_get_chunk JSON")?;
            let resp = handle_corpus_get_chunk(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        ("POST", "/execute_tool") => {
            let req: ExecuteToolRequest =
                serde_json::from_slice(&body).context("Invalid /execute_tool JSON")?;
            let resp = handle_execute_tool(&state, req).await;
            let json = serde_json::to_vec(&resp)?;
            write_response(&mut stream, 200, &json).await?;
        }
        _ => {
            write_response(&mut stream, 404, b"{\"error\":\"not found\"}").await?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// LLM query handlers
// ---------------------------------------------------------------------------

async fn handle_query(state: &SharedState, req: QueryRequest) -> QueryResponse {
    match do_llm_call(state, &req.prompt, req.model.as_deref(), req.depth).await {
        Ok(text) => QueryResponse {
            response: Some(text),
            error: None,
        },
        Err(e) => QueryResponse {
            response: None,
            error: Some(format!("{e:#}")),
        },
    }
}

async fn handle_batched(state: &SharedState, req: BatchedRequest) -> BatchedResponse {
    let model = req.model.as_deref();
    let depth = req.depth;

    let futures: Vec<_> = req
        .prompts
        .iter()
        .map(|p| do_llm_call(state, p, model, depth))
        .collect();

    let results = futures::future::join_all(futures).await;

    let mut responses = Vec::with_capacity(results.len());
    for r in &results {
        match r {
            Ok(text) => responses.push(text.clone()),
            Err(e) => {
                return BatchedResponse {
                    responses: None,
                    error: Some(format!("{e:#}")),
                };
            }
        }
    }

    BatchedResponse {
        responses: Some(responses),
        error: None,
    }
}

// ---------------------------------------------------------------------------
// Gateway proxy handlers (RLM+CodeMode)
// ---------------------------------------------------------------------------

/// Execute TypeScript/JS code on the Gateway via EXECUTE_CODE, routing large
/// results through the RLM corpus.
async fn handle_execute_code(state: &SharedState, req: ExecuteCodeRequest) -> ExecuteCodeResponse {
    let Some(ref mcp) = state.mcp_client else {
        return ExecuteCodeResponse {
            result: None,
            stored_in_corpus: false,
            error: Some("Gateway not configured (no mcp_client)".to_string()),
        };
    };

    tracing::debug!(
        "LmHandler /execute_code: code_len={}",
        req.code.len()
    );

    // Call Gateway's EXECUTE_CODE
    let raw = match mcp
        .call_tool(
            "EXECUTE_CODE".to_string(),
            Some(json!({ "code": req.code })),
            Some(Duration::from_secs(120)),
        )
        .await
    {
        Ok(r) => serde_json::to_string(&r).unwrap_or_default(),
        Err(e) => {
            return ExecuteCodeResponse {
                result: None,
                stored_in_corpus: false,
                error: Some(format!("EXECUTE_CODE failed: {e}")),
            };
        }
    };

    // Route through corpus if router is available
    if let Some(ref router) = state.rlm_router {
        let call_id = format!("repl_exec_{}", uuid::Uuid::new_v4());
        match router
            .process_result(&call_id, "kontext-dev", "EXECUTE_CODE", &raw)
            .await
        {
            Ok(ProcessedResult::StoredInCorpus { summary, .. }) => ExecuteCodeResponse {
                result: Some(summary),
                stored_in_corpus: true,
                error: None,
            },
            Ok(ProcessedResult::PassThrough { content }) => ExecuteCodeResponse {
                result: Some(content),
                stored_in_corpus: false,
                error: None,
            },
            Err(e) => ExecuteCodeResponse {
                result: Some(raw),
                stored_in_corpus: false,
                error: Some(format!("Corpus routing failed (returning raw): {e}")),
            },
        }
    } else {
        ExecuteCodeResponse {
            result: Some(raw),
            stored_in_corpus: false,
            error: None,
        }
    }
}

/// Search the RLM corpus for chunks matching a query.
async fn handle_corpus_search(
    state: &SharedState,
    req: CorpusSearchRequest,
) -> serde_json::Value {
    let Some(ref router) = state.rlm_router else {
        return json!({ "error": "Corpus not configured (no rlm_router)", "results": [] });
    };

    let results = router.search_corpus(&req.query, req.max_results).await;
    json!({ "results": results })
}

/// Retrieve a specific chunk from the RLM corpus by ID.
async fn handle_corpus_get_chunk(
    state: &SharedState,
    req: CorpusGetChunkRequest,
) -> serde_json::Value {
    let Some(ref router) = state.rlm_router else {
        return json!({ "error": "Corpus not configured (no rlm_router)", "content": null });
    };

    match router.get_chunk(&req.chunk_id).await {
        Some(content) => json!({ "content": content }),
        None => json!({ "error": "Chunk not found", "content": null }),
    }
}

/// Execute a Gateway tool directly via EXECUTE_TOOL (like Baseline mode).
/// Returns the full result without corpus routing.
///
/// Performs fuzzy tool name resolution: the LLM may guess tool names like
/// `linear_list_projects` or `list_projects` which need to be resolved to
/// the actual Gateway ID (e.g., `Linear:list_projects`).
async fn handle_execute_tool(state: &SharedState, req: ExecuteToolRequest) -> ExecuteToolResponse {
    let Some(ref mcp) = state.mcp_client else {
        return ExecuteToolResponse {
            result: None,
            error: Some("Gateway not configured (no mcp_client)".to_string()),
        };
    };

    // Resolve tool_id: try exact match first, then fuzzy lookup
    let resolved_id = resolve_tool_id(&req.tool_id, &state.tool_lookup);

    tracing::debug!(
        "LmHandler /execute_tool: requested={} resolved={} args_len={}",
        req.tool_id,
        resolved_id,
        req.args.to_string().len()
    );

    let execute_args = json!({
        "tool_id": resolved_id,
        "tool_arguments": req.args,
    });

    match mcp
        .call_tool(
            "EXECUTE_TOOL".to_string(),
            Some(execute_args),
            Some(Duration::from_secs(120)),
        )
        .await
    {
        Ok(r) => ExecuteToolResponse {
            result: Some(serde_json::to_string(&r).unwrap_or_default()),
            error: None,
        },
        Err(e) => ExecuteToolResponse {
            result: None,
            error: Some(format!("EXECUTE_TOOL failed: {e}")),
        },
    }
}

/// Resolve a potentially fuzzy tool ID to the canonical Gateway tool ID.
///
/// Tries in order:
/// 1. Exact match in lookup table
/// 2. Case-insensitive exact match
/// 3. Substring containment match (e.g., "list_projects" matches "Linear:list_projects")
/// 4. Falls back to the original ID if no match
fn resolve_tool_id(requested: &str, lookup: &HashMap<String, String>) -> String {
    // 1. Exact match
    if let Some(id) = lookup.get(requested) {
        return id.clone();
    }

    // 2. Case-insensitive exact match
    let requested_lower = requested.to_lowercase();
    for (key, id) in lookup {
        if key.to_lowercase() == requested_lower {
            return id.clone();
        }
    }

    // 3. Substring containment — find the best match
    // Normalize common separators: dots and hyphens become underscores
    let normalized = requested_lower.replace(['.', '-', ':'], "_");
    for (key, id) in lookup {
        let key_normalized = key.to_lowercase().replace(['.', '-', ':'], "_");
        if normalized.contains(&key_normalized) || key_normalized.contains(&normalized) {
            return id.clone();
        }
    }

    // 4. Fall back to original
    requested.to_string()
}

// ---------------------------------------------------------------------------
// LLM query core
// ---------------------------------------------------------------------------

/// Execute a single LLM chat completion with budget checks.
async fn do_llm_call(
    state: &SharedState,
    prompt: &str,
    model_override: Option<&str>,
    depth: u32,
) -> Result<String> {
    // Estimate tokens for the request.
    let estimated = (prompt.len() / 4).max(1) as i64;

    // Budget gate.
    let check = state.budget.can_proceed(estimated).await;
    if !check.can_proceed() {
        anyhow::bail!("Budget exhausted: {check:?}");
    }

    // Pick model based on depth.
    let model = model_override.unwrap_or(if depth == 0 {
        &state.root_model
    } else {
        &state.sub_model
    });

    tracing::debug!(
        "LmHandler: calling model={model} depth={depth} prompt_len={}",
        prompt.len()
    );

    let user_msg = ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?,
    );

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(vec![user_msg])
        .temperature(0.0_f32)
        .build()?;

    let response = state
        .client
        .chat()
        .create(request)
        .await
        .context("OpenAI chat completion failed")?;

    // Extract usage information.
    let (input_tokens, output_tokens) = if let Some(u) = &response.usage {
        (u.prompt_tokens as i64, u.completion_tokens as i64)
    } else {
        (0, 0)
    };

    // Record in budget manager.
    let total_tokens = input_tokens + output_tokens;
    state.budget.record_usage(total_tokens).await;

    // Record in handler-level usage summary.
    {
        let mut usage = state.usage.write().await;
        usage.total_calls += 1;
        usage.total_input_tokens += input_tokens;
        usage.total_output_tokens += output_tokens;
    }

    // Extract response text.
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    tracing::debug!(
        "LmHandler: response len={} input_tokens={input_tokens} output_tokens={output_tokens}",
        text.len()
    );

    Ok(text)
}

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 parsing
// ---------------------------------------------------------------------------

/// Read an HTTP/1.1 request from `stream` and return `(method, path, body)`.
async fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 1024];

    // Read until we find the end of headers (\r\n\r\n).
    let header_end;
    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .context("read_request: read failed")?;
        if n == 0 {
            anyhow::bail!("read_request: connection closed before headers complete");
        }
        buf.extend_from_slice(&tmp[..n]);

        if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
            header_end = pos;
            break;
        }
    }

    let header_bytes = &buf[..header_end];
    let header_str =
        std::str::from_utf8(header_bytes).context("read_request: headers are not valid UTF-8")?;

    // Parse request line: METHOD PATH HTTP/1.x
    let request_line = header_str
        .lines()
        .next()
        .context("read_request: empty request")?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .context("read_request: missing method")?
        .to_string();
    let path = parts
        .next()
        .context("read_request: missing path")?
        .to_string();

    // Find Content-Length (case-insensitive).
    let content_length: usize = header_str
        .lines()
        .find_map(|line| {
            let lower = line.to_lowercase();
            if lower.starts_with("content-length:") {
                lower
                    .strip_prefix("content-length:")
                    .and_then(|v| v.trim().parse().ok())
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Body starts right after \r\n\r\n (4 bytes).
    let body_start = header_end + 4;
    let already_read = buf.len() - body_start;

    // Read remaining body bytes if needed.
    if already_read < content_length {
        let remaining = content_length - already_read;
        let mut extra = vec![0u8; remaining];
        stream
            .read_exact(&mut extra)
            .await
            .context("read_request: failed to read full body")?;
        buf.extend_from_slice(&extra);
    }

    let body = buf[body_start..body_start + content_length].to_vec();
    Ok((method, path, body))
}

/// Write a minimal HTTP/1.1 response.
async fn write_response(stream: &mut TcpStream, status: u16, body: &[u8]) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

/// Find the first occurrence of `needle` in `haystack`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_starts_and_health_check() {
        use crate::rlm::RlmConfig;

        let config = LmHandlerConfig {
            root_model: "gpt-4o-mini".to_string(),
            sub_model: None,
            api_key: "test-key".to_string(),
            base_url: None,
            mcp_client: None,
            rlm_router: None,
            tool_lookup: HashMap::new(),
        };
        let budget = Arc::new(BudgetManager::new(&RlmConfig::default()));
        let handler = LmHandler::start(config, budget).await.unwrap();
        assert!(handler.port() > 0);

        // Hit the health endpoint.
        let url = format!("http://127.0.0.1:{}/health", handler.port());
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");

        // Clean up.
        handler.stop().await;
    }

    #[tokio::test]
    async fn test_404_for_unknown_path() {
        use crate::rlm::RlmConfig;

        let config = LmHandlerConfig {
            root_model: "gpt-4o-mini".to_string(),
            sub_model: None,
            api_key: "test-key".to_string(),
            base_url: None,
            mcp_client: None,
            rlm_router: None,
            tool_lookup: HashMap::new(),
        };
        let budget = Arc::new(BudgetManager::new(&RlmConfig::default()));
        let handler = LmHandler::start(config, budget).await.unwrap();

        let url = format!("http://127.0.0.1:{}/nonexistent", handler.port());
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 404);

        handler.stop().await;
    }

    #[tokio::test]
    async fn test_usage_starts_at_zero() {
        use crate::rlm::RlmConfig;

        let config = LmHandlerConfig {
            root_model: "gpt-4o-mini".to_string(),
            sub_model: None,
            api_key: "test-key".to_string(),
            base_url: None,
            mcp_client: None,
            rlm_router: None,
            tool_lookup: HashMap::new(),
        };
        let budget = Arc::new(BudgetManager::new(&RlmConfig::default()));
        let handler = LmHandler::start(config, budget).await.unwrap();

        let usage = handler.usage().await;
        assert_eq!(usage.total_calls, 0);
        assert_eq!(usage.total_input_tokens, 0);
        assert_eq!(usage.total_output_tokens, 0);

        handler.stop().await;
    }
}
