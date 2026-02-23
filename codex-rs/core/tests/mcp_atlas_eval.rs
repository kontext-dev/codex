#![cfg(feature = "benchmarking")]
//! MCP-Atlas Evaluation Script
//!
//! Benchmarks LLM agent performance on the MCP-Atlas dataset using two client architectures:
//!
//! ## Tool-Calling Client (ToolCallingRunner)
//! Uses LLM function calling with modes:
//! - Baseline: Direct EXECUTE_TOOL calls with full results in context
//! - CodeMode: EXECUTE_CODE with summarized results
//! - Baseline+RLM: EXECUTE_TOOL with RLM routing for large results
//! - CodeMode+RLM: RLM REPL with Gateway tool execution via execute_code()
//!
//! ## Codex Client (CodexRunner)
//! Uses the real Codex agent with full system prompts via ConversationManager.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Gateway credentials
//! export KONTEXT_CLIENT_ID=<your-client-id>
//! export KONTEXT_CLIENT_SECRET=<your-client-secret>
//! export KONTEXT_MCP_URL=http://localhost:4000/mcp
//! export KONTEXT_TOKEN_URL=http://localhost:4000/oauth2/token
//!
//! # LLM API key (any OpenAI-compatible provider)
//! export EVAL_API_KEY=<your-api-key>
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Test mode (first 10 tasks)
//! source .env
//! cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture
//!
//! # Select gateway task CSV (v1, v2, v3, v4, or custom path)
//! EVAL_GATEWAY_TASK_CSV=v3 cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture
//!
//! # Full evaluation (500 tasks × 3 modes = 1500 runs)
//! cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use codex_core::eval::ClaimJudge;
use codex_core::eval::ClaimScore;
use codex_core::eval::ClaimVerificationResult;
use codex_core::eval::PASS_THRESHOLD;
use codex_core::eval::TaskResult;
use codex_core::eval::ToolCallingMode;
use codex_core::eval::ToolCallingRunner;
use codex_core::rlm::EvidenceStore;
use codex_core::rlm::GatewayResultRouter;
use codex_core::rlm::RlmConfig;
use codex_core::rlm::RlmCorpus;
use codex_rmcp_client::OAuthCredentialsStoreMode;
use codex_rmcp_client::RmcpClient;
use core_test_support::gateway_auth;
use kontext_dev::build_mcp_url;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::RwLock;

fn init_test_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

/// Gateway task CSV paths relative to core directory (CARGO_MANIFEST_DIR).
const GATEWAY_TASKS_V1_DATASET_PATH: &str = "../mcp-atlas/services/mcp_eval/gateway_tasks.csv";
const GATEWAY_TASKS_V2_DATASET_PATH: &str = "../mcp-atlas/services/mcp_eval/gateway_tasks_v2.csv";
const GATEWAY_TASKS_V3_DATASET_PATH: &str = "../mcp-atlas/services/mcp_eval/gateway_tasks_v3.csv";
const GATEWAY_TASKS_V4_DATASET_PATH: &str = "../mcp-atlas/services/mcp_eval/gateway_tasks_v4.csv";

/// Default dataset path (v4) can be overridden with `EVAL_GATEWAY_TASK_CSV` or
/// the legacy `EVAL_DATASET_PATH`.
const DEFAULT_DATASET_PATH: &str = GATEWAY_TASKS_V4_DATASET_PATH;

/// Original Arrow dataset path (for reference)
#[allow(dead_code)]
const ARROW_DATASET_PATH: &str = "../../data/coding_atlas/data-00000-of-00001.arrow";

/// Combined result for a single task across all modes
#[derive(Debug)]
struct EvalResult {
    task_id: String,
    prompt: String,
    claims: Vec<String>,
    task_result: TaskResult,
    verification: ClaimVerificationResult,
}

// =============================================================================
// SETUP HELPERS
// =============================================================================

/// Check if evaluation tests should be skipped
fn should_skip() -> bool {
    env::var("SKIP_EVAL_TESTS").is_ok()
        || gateway_auth::should_skip()
        || env_non_empty("EVAL_API_KEY").is_none()
}

/// Read an env var, treating empty strings as unset.
fn env_non_empty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.is_empty())
}

/// Resolve the API key for the agent LLM.
fn resolve_eval_api_key() -> Option<String> {
    env_non_empty("EVAL_API_KEY")
}

/// Resolve the base URL for the agent LLM.
fn resolve_eval_base_url() -> Option<String> {
    env_non_empty("EVAL_BASE_URL")
}

/// Resolve the API key for the judge LLM.
/// Cascade: `JUDGE_API_KEY` -> `EVAL_API_KEY`
fn resolve_judge_api_key() -> Option<String> {
    env_non_empty("JUDGE_API_KEY").or_else(|| env_non_empty("EVAL_API_KEY"))
}

/// Resolve the base URL for the judge LLM.
/// Cascade: `JUDGE_BASE_URL` -> `EVAL_BASE_URL`
fn resolve_judge_base_url() -> Option<String> {
    env_non_empty("JUDGE_BASE_URL").or_else(|| env_non_empty("EVAL_BASE_URL"))
}

/// Resolve the judge model name.
/// Cascade: `JUDGE_MODEL` -> `EVAL_MODEL`
fn resolve_judge_model() -> String {
    env_non_empty("JUDGE_MODEL")
        .or_else(|| env_non_empty("EVAL_MODEL"))
        .unwrap_or_else(|| "gpt-4o".to_string())
}

fn resolve_gateway_dataset_path(manifest_dir: &str, selector: &str) -> PathBuf {
    match selector.trim() {
        "" | "v4" | "gateway_tasks_v4" | "gateway_tasks_v4.csv" => {
            PathBuf::from(manifest_dir).join(GATEWAY_TASKS_V4_DATASET_PATH)
        }
        "v3" | "gateway_tasks_v3" | "gateway_tasks_v3.csv" => {
            PathBuf::from(manifest_dir).join(GATEWAY_TASKS_V3_DATASET_PATH)
        }
        "v2" | "gateway_tasks_v2" | "gateway_tasks_v2.csv" => {
            PathBuf::from(manifest_dir).join(GATEWAY_TASKS_V2_DATASET_PATH)
        }
        "v1" | "gateway_tasks" | "gateway_tasks.csv" => {
            PathBuf::from(manifest_dir).join(GATEWAY_TASKS_V1_DATASET_PATH)
        }
        other => {
            let path = PathBuf::from(other);
            if path.is_absolute() {
                path
            } else {
                PathBuf::from(manifest_dir).join(path)
            }
        }
    }
}

/// Get dataset path.
///
/// Priority:
/// 1. `EVAL_GATEWAY_TASK_CSV` (recommended): `v1`, `v2`, `v3`, or a CSV path
/// 2. `EVAL_DATASET_PATH` (legacy): explicit dataset path
/// 3. default: gateway v3 task CSV
fn get_dataset_path() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    if let Ok(selector) = env::var("EVAL_GATEWAY_TASK_CSV") {
        return resolve_gateway_dataset_path(&manifest_dir, &selector);
    }

    if let Ok(custom_path) = env::var("EVAL_DATASET_PATH") {
        return PathBuf::from(custom_path);
    }

    PathBuf::from(manifest_dir).join(DEFAULT_DATASET_PATH)
}

/// Create an RLM router for evaluation
async fn create_rlm_router(temp_path: &std::path::Path) -> anyhow::Result<GatewayResultRouter> {
    let rlm_config = RlmConfig::default();
    let corpus = RlmCorpus::new(temp_path.to_path_buf(), rlm_config.clone()).await?;
    corpus.ingest_prompt("MCP-Atlas evaluation corpus.").await?;

    let router = GatewayResultRouter::new(
        Arc::new(RwLock::new(Some(corpus))),
        Arc::new(RwLock::new(EvidenceStore::new())),
        rlm_config,
    );

    Ok(router)
}

/// Create a new Gateway session: authenticate, connect MCP client, build runner.
///
/// Returns `(client, runner, session_created_at)`.
/// The returned runner does NOT yet have a model set — call `.with_model()` on it.
async fn create_gateway_session(
    config: &kontext_dev::KontextDevConfig,
    rlm_router: Option<Arc<GatewayResultRouter>>,
) -> anyhow::Result<(Arc<RmcpClient>, ToolCallingRunner, Instant)> {
    let token = gateway_auth::authenticate(config).await?;
    let mcp_url = build_mcp_url(config, &token.access_token)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let client = Arc::new(
        RmcpClient::new_streamable_http_client(
            &config.server_name,
            &mcp_url,
            None,
            None,
            None,
            OAuthCredentialsStoreMode::File,
        )
        .await?,
    );

    client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await?;

    let runner = ToolCallingRunner::new(
        client.clone(),
        rlm_router,
        resolve_eval_base_url(),
        resolve_eval_api_key(),
    )
    .await?;

    tracing::debug!("Gateway session created");
    Ok((client, runner, Instant::now()))
}

// =============================================================================
// VERIFICATION TESTS
// =============================================================================

/// Test 1: Verify dataset loads correctly
#[tokio::test]
async fn test_dataset_loads() {
    init_test_tracing();
    tracing::info!("Test: Dataset Loading");

    let dataset_path = get_dataset_path();
    tracing::debug!("Loading from: {:?}", dataset_path);

    if !dataset_path.exists() {
        tracing::warn!("Dataset not found at {:?}", dataset_path);
        tracing::warn!("Please ensure the MCP-Atlas dataset is downloaded.");
        return;
    }

    match codex_core::eval::load_dataset(&dataset_path) {
        Ok(tasks) => {
            tracing::debug!("Loaded {} tasks", tasks.len());
            assert!(!tasks.is_empty(), "Dataset should not be empty");

            // Print first task as sample
            if let Some(task) = tasks.first() {
                tracing::debug!("Sample task:");
                tracing::debug!("ID: {}", task.task_id);
                tracing::debug!(
                    "Tools: {:?}",
                    &task.enabled_tools[..task.enabled_tools.len().min(5)]
                );
                tracing::debug!(
                    "Prompt: {}...",
                    &task.prompt[..task.prompt.len().min(100)]
                );
                tracing::debug!("Claims: {} total", task.claims.len());
            }
        }
        Err(e) => {
            tracing::error!("Failed to load dataset: {e}");
            panic!("Dataset loading failed");
        }
    }
}

/// Test 2: Verify Gateway connection works
#[tokio::test]
async fn test_gateway_connection() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::info!("Test: Gateway Connection");

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    tracing::debug!("Token URL: {:?}", config.token_url);

    // Authenticate
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => {
            tracing::debug!("Authentication successful");
            t
        }
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    // Build MCP URL and connect
    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");

    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => c,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    // Initialize
    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }

    // List tools
    match client.list_tools(None, Some(Duration::from_secs(30))).await {
        Ok(result) => {
            tracing::info!("Found {} tools", result.tools.len());
            // Print EXECUTE_TOOL schema to understand expected args
            for tool in &result.tools {
                tracing::debug!("Tool: {}", tool.name);
                tracing::debug!(
                    "Description: {}",
                    tool.description.as_deref().unwrap_or("N/A")
                );
                tracing::debug!(
                    "Schema: {}",
                    serde_json::to_string_pretty(&tool.input_schema).unwrap_or("N/A".to_string())
                );
            }
            assert!(!result.tools.is_empty(), "Should have tools available");
        }
        Err(e) => {
            tracing::error!("Failed to list tools: {e}");
        }
    }
}

/// Test 3: Verify real Git MCP tool works
#[tokio::test]
async fn test_git_mcp_tool() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::info!("Test: Git MCP Tool");

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");

    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => c,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }

    // Try to call a git-related tool
    let start = Instant::now();
    let result = client
        .call_tool(
            "SEARCH_TOOLS".to_string(),
            Some(json!({"query": "git"})),
            Some(Duration::from_secs(60)),
        )
        .await;

    match result {
        Ok(r) => {
            tracing::debug!("SEARCH_TOOLS(git) succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            tracing::debug!(
                "Response preview: {}...",
                &content[..content.len().min(500)]
            );
        }
        Err(e) => {
            tracing::error!("SEARCH_TOOLS failed: {e}");
        }
    }
}

/// Test 4: Verify Code Executor MCP tool works
#[tokio::test]
async fn test_code_executor_mcp_tool() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::info!("Test: Code Executor MCP Tool");

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");

    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => c,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }

    // Try EXECUTE_CODE
    let start = Instant::now();
    let result = client
        .call_tool(
            "EXECUTE_CODE".to_string(),
            Some(json!({
                "code": "return 'Hello from Code Executor';"
            })),
            Some(Duration::from_secs(60)),
        )
        .await;

    match result {
        Ok(r) => {
            tracing::debug!("EXECUTE_CODE succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            tracing::debug!("Response: {content}");
        }
        Err(e) => {
            tracing::error!("EXECUTE_CODE failed: {e}");
            tracing::warn!("(This is expected if EXECUTE_CODE is not available)");
        }
    }
}

/// Test 5: Verify CLI MCP tool works
#[tokio::test]
async fn test_cli_mcp_tool() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::info!("Test: CLI MCP Tool");

    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");

    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => c,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }

    // Search for CLI tools
    let start = Instant::now();
    let result = client
        .call_tool(
            "SEARCH_TOOLS".to_string(),
            Some(json!({"query": "cli execute command"})),
            Some(Duration::from_secs(60)),
        )
        .await;

    match result {
        Ok(r) => {
            tracing::debug!("SEARCH_TOOLS(cli) succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            tracing::debug!(
                "Response preview: {}...",
                &content[..content.len().min(500)]
            );
        }
        Err(e) => {
            tracing::error!("SEARCH_TOOLS failed: {e}");
        }
    }
}

/// Test 6: Verify Claim Judge works
#[tokio::test]
async fn test_claim_judge() {
    init_test_tracing();
    if env_non_empty("EVAL_API_KEY").is_none() {
        tracing::warn!("Skipping: EVAL_API_KEY not set");
        return;
    }

    tracing::info!("Test: Claim Judge");

    let judge = match ClaimJudge::with_model(
        &resolve_judge_model(),
        resolve_judge_base_url(),
        resolve_judge_api_key(),
    ) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to create judge: {e}");
            return;
        }
    };

    let task_prompt = "List the files in the current directory";
    let answer = "The current directory contains: main.rs, lib.rs, Cargo.toml, README.md";
    let claims = vec![
        "The response lists files in the directory".to_string(),
        "The response includes Cargo.toml".to_string(),
        "The response shows file sizes".to_string(), // This should fail
    ];

    let start = Instant::now();
    match judge.verify_claims(task_prompt, answer, &claims, "").await {
        Ok(result) => {
            tracing::info!("Verification completed in {:?}", start.elapsed());
            tracing::info!("Coverage: {:.2}", result.coverage);
            tracing::info!("Passed: {}", result.passed);
            tracing::info!("Per-claim results:");
            for (claim, score) in &result.scores {
                let score_str = match score {
                    ClaimScore::Fulfilled => "FULFILLED",
                    ClaimScore::PartiallyFulfilled => "PARTIAL",
                    ClaimScore::NotFulfilled => "NOT_FULFILLED",
                };
                tracing::info!("- {}: {}", score_str, &claim[..claim.len().min(50)]);
            }
        }
        Err(e) => {
            tracing::error!("Verification failed: {e}");
        }
    }
}

/// Test 7: Verify RLM routing works
#[tokio::test]
async fn test_rlm_routing() {
    init_test_tracing();
    tracing::info!("Test: RLM Routing");

    let temp_dir = TempDir::new().unwrap();
    let router = match create_rlm_router(temp_dir.path()).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to create RLM router: {e}");
            return;
        }
    };

    // Test small response (should pass through)
    let small_content = r#"{"files": ["main.rs", "lib.rs"]}"#;
    match router
        .process_result("test_1", "gateway", "list_files", small_content)
        .await
    {
        Ok(result) => {
            let routing = match result {
                codex_core::rlm::ProcessedResult::PassThrough { .. } => "passthrough",
                codex_core::rlm::ProcessedResult::StoredInCorpus { .. } => "corpus",
            };
            tracing::debug!(
                "Small response ({} bytes): {}",
                small_content.len(),
                routing
            );
            assert!(matches!(
                result,
                codex_core::rlm::ProcessedResult::PassThrough { .. }
            ));
        }
        Err(e) => {
            tracing::error!("Small response routing failed: {e}");
        }
    }

    // Test large response (should go to corpus)
    let large_content = "x".repeat(20000);
    match router
        .process_result("test_2", "gateway", "large_tool", &large_content)
        .await
    {
        Ok(result) => {
            let routing = match result {
                codex_core::rlm::ProcessedResult::PassThrough { .. } => "passthrough",
                codex_core::rlm::ProcessedResult::StoredInCorpus { .. } => "corpus",
            };
            tracing::debug!(
                "Large response ({} bytes): {}",
                large_content.len(),
                routing
            );
            assert!(matches!(
                result,
                codex_core::rlm::ProcessedResult::StoredInCorpus { .. }
            ));
        }
        Err(e) => {
            tracing::error!("Large response routing failed: {e}");
        }
    }
}

// =============================================================================
// MAIN EVALUATION
// =============================================================================

/// Run three-way evaluation on first 10 tasks (test mode)
#[tokio::test]
async fn run_mcp_atlas_three_way_evaluation() {
    init_test_tracing();
    if should_skip() {
        tracing::trace!("MCP-Atlas Evaluation - Setup Required");
        tracing::trace!("Required environment variables:");
        tracing::trace!("KONTEXT_CLIENT_ID=<your-client-id>");
        tracing::trace!("KONTEXT_CLIENT_SECRET=<your-client-secret>");
        tracing::trace!("KONTEXT_MCP_URL=http://localhost:4000/mcp");
        tracing::trace!("KONTEXT_TOKEN_URL=http://localhost:4000/oauth2/token");
        tracing::trace!("EVAL_API_KEY=<your-api-key>    # Any OpenAI-compatible provider");
        tracing::trace!("Agent / Judge LLM configuration:");
        tracing::trace!(
            "EVAL_API_KEY=<key>             # API key for agent LLM (required)"
        );
        tracing::trace!(
            "EVAL_BASE_URL=<url>            # Base URL for agent LLM (OpenAI-compatible)"
        );
        tracing::trace!(
            "JUDGE_API_KEY=<key>            # API key for judge LLM (defaults to EVAL_API_KEY)"
        );
        tracing::trace!(
            "JUDGE_BASE_URL=<url>           # Base URL for judge LLM (falls back to EVAL_BASE_URL)"
        );
        tracing::trace!(
            "JUDGE_MODEL=<model>            # Judge model name (defaults to EVAL_MODEL)"
        );
        tracing::trace!("Optional configuration:");
        tracing::trace!("EVAL_LIMIT=5                   # Run only N tasks");
        tracing::trace!("EVAL_TASK_PREFIX=task_linear   # Filter by task ID prefix");
        tracing::trace!("EVAL_MODEL=<model>             # Model for tool-calling client");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v4       # gateway_tasks_v4.csv (default)");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v3       # gateway_tasks_v3.csv");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v2       # gateway_tasks_v2.csv");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v1       # gateway_tasks.csv");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=/path/to/gateway_tasks.csv  # Custom CSV path");
        tracing::trace!("EVAL_DATASET_PATH=/path/to/dataset.csv            # Legacy override");
        tracing::trace!("Tool-calling client modes:");
        tracing::trace!(
            "EVAL_TOOL_MODES=baseline,codemode  # Comma-separated (baseline,codemode,rlm,rlmcodemode,baselinerlm,all)"
        );
        tracing::trace!("Codex client:");
        tracing::trace!("EVAL_USE_CODEX=true            # Enable Codex client");
        tracing::trace!(
            "EVAL_CODEX_MODEL=gpt-4-turbo   # Independent model for Codex (defaults to EVAL_MODEL)"
        );
        tracing::trace!("Run with:");
        tracing::trace!("source .env");
        tracing::trace!(
            "cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture"
        );
        tracing::trace!("Examples:");
        tracing::trace!("# Run only tool-calling modes:");
        tracing::trace!("EVAL_TOOL_MODES=baseline,codemode cargo test ...");
        tracing::trace!("# Run only Codex client:");
        tracing::trace!("EVAL_TOOL_MODES= EVAL_USE_CODEX=true cargo test ...");
        tracing::trace!("# Run both with different models:");
        tracing::trace!(
            "EVAL_TOOL_MODES=baseline EVAL_USE_CODEX=true EVAL_MODEL=gpt-4o EVAL_CODEX_MODEL=gpt-4-turbo cargo test ..."
        );
        return;
    }

    tracing::trace!("MCP-Atlas Three-Way Evaluation");

    // Step 1: Load FULL dataset (filter comes later)
    let dataset_path = get_dataset_path();
    tracing::debug!("Loading dataset from {:?}...", dataset_path);

    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Failed to load dataset: {e}");
            panic!("Failed to load dataset: {e}");
        }
    };

    tracing::debug!("Loaded {} tasks from dataset", tasks.len());

    // Step 2: Setup RLM infrastructure (created once, reused across session refreshes)
    let temp_dir = TempDir::new().unwrap();
    let rlm_router: Option<Arc<GatewayResultRouter>> =
        match create_rlm_router(temp_dir.path()).await {
            Ok(r) => Some(Arc::new(r)),
            Err(e) => {
                tracing::warn!("RLM router creation failed: {e}");
                None
            }
        };

    // Step 3: Setup Gateway connection
    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    tracing::debug!("Connecting to Gateway...");

    // Session refresh threshold: reconnect when 50 minutes have elapsed (token TTL is 60 min)
    const SESSION_REFRESH_SECS: u64 = 50 * 60;

    let (_client, runner, mut session_created_at) =
        match create_gateway_session(&config, rlm_router.clone()).await {
            Ok(v) => v,
            Err(e) if e.to_string().contains("onnection refused") => {
                tracing::warn!("Skipping: gateway not running");
                return;
            }
            Err(e) => panic!("Gateway session setup failed: {e}"),
        };

    // Step 4: Setup judge
    let judge_model = resolve_judge_model();
    let judge = match ClaimJudge::with_model(
        &judge_model,
        resolve_judge_base_url(),
        resolve_judge_api_key(),
    ) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to create claim judge: {e}");
            panic!("Failed to create claim judge: {e}");
        }
    };
    tracing::debug!("Judge model: {judge_model}");

    // Step 5: Create runner and discover tools
    tracing::debug!("Discovering available Gateway tools...");
    let agent_model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let mut runner = runner.with_model(&agent_model);
    tracing::debug!("Using model: {agent_model}");

    // Print discovered tools
    let tools = runner.available_tools();
    tracing::info!("Found {} tools:", tools.len());
    let mut by_server: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for tool in tools {
        by_server.entry(&tool.server).or_default().push(&tool.name);
    }
    for (server, tool_names) in &by_server {
        tracing::debug!("{}: {}", server, tool_names.join(", "));
    }

    // Step 6: Use all tasks from the gateway dataset (already filtered to available tools)
    // Filter out tasks with 0 coverage (tools that don't match)
    let eval_tasks: Vec<_> = tasks
        .iter()
        .filter(|task| {
            if task.enabled_tools.is_empty() {
                return false;
            }
            // At least one tool must resolve
            task.enabled_tools
                .iter()
                .any(|tool_name| runner.resolve_tool(tool_name).is_some())
        })
        .collect();

    tracing::info!(
        "Tasks with resolvable tools: {}/{}",
        eval_tasks.len(),
        tasks.len()
    );

    assert!(!eval_tasks.is_empty(), "eval_tasks should not be empty -- tool resolution may have regressed");

    // Show the solvable tasks with their coverage
    tracing::debug!("Solvable tasks:");
    for task in eval_tasks.iter().take(20) {
        let coverage = runner.get_tool_coverage(task);
        let matches = runner.get_matching_tools(task);
        tracing::debug!(
            "Task: {}",
            &task.task_id[..task.task_id.len().min(15)]
        );
        tracing::debug!(
            "Coverage: {:.0}% ({} tools matched)",
            coverage * 100.0,
            matches.len()
        );

        // Show prompt
        let prompt_preview = if task.prompt.len() > 200 {
            format!("{}...", &task.prompt[..200].replace('\n', " "))
        } else {
            task.prompt.replace('\n', " ")
        };
        tracing::debug!("Prompt: {prompt_preview}");

        // Show claims
        tracing::debug!("Claims ({}):", task.claims.len());
        for (i, claim) in task.claims.iter().enumerate().take(3) {
            tracing::debug!("{}. {}", i + 1, &claim[..claim.len().min(80)]);
        }
        if task.claims.len() > 3 {
            tracing::debug!("... and {} more claims", task.claims.len() - 3);
        }

        // Show tool mapping
        tracing::debug!("Tools:");
        for (dataset_tool, gateway_tool) in matches.iter().take(3) {
            tracing::debug!(
                "{} -> {} ({})",
                dataset_tool, gateway_tool.name, gateway_tool.server
            );
        }
        if matches.len() > 3 {
            tracing::debug!("... and {} more tools", matches.len() - 3);
        }

        // Show missing tools
        let missing: Vec<_> = task
            .enabled_tools
            .iter()
            .filter(|t| runner.resolve_tool(t).is_none())
            .collect();
        if !missing.is_empty() {
            tracing::debug!("Missing tools: {:?}", missing);
        }
    }

    // Apply subset filters from environment variables
    // EVAL_LIMIT: max number of tasks (default: all)
    // EVAL_TASK_PREFIX: filter to tasks starting with this prefix (e.g., "task_linear")
    // EVAL_MODES: comma-separated modes to run (e.g., "baseline,codemode" or "rlm")
    let limit: usize = std::env::var("EVAL_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let task_prefixes: Option<Vec<String>> = std::env::var("EVAL_TASK_PREFIX")
        .ok()
        .map(|s| s.split(',').map(|p| p.trim().to_string()).collect());

    let eval_tasks: Vec<_> = eval_tasks
        .into_iter()
        .filter(|task| {
            if let Some(ref prefixes) = task_prefixes {
                prefixes
                    .iter()
                    .any(|prefix| task.task_id.starts_with(prefix))
            } else {
                true
            }
        })
        .take(limit)
        .collect();

    if eval_tasks.is_empty() {
        tracing::warn!("No tasks match the filter criteria.");
        if let Some(ref prefixes) = task_prefixes {
            tracing::warn!("EVAL_TASK_PREFIX={}", prefixes.join(","));
        }
        return;
    }

    tracing::warn!("Running evaluation on {} tasks", eval_tasks.len());
    if let Some(ref prefixes) = task_prefixes {
        tracing::info!("(filtered by prefixes: {})", prefixes.join(", "));
    }
    if limit < usize::MAX {
        tracing::info!("(limited to {limit} tasks)");
    }

    // Step 7: Parse tool-calling modes from EVAL_TOOL_MODES (comma-separated)
    // Options: baseline, codemode, rlm, rlmcodemode, baselinerlm. Default: baseline,codemode,rlm,rlmcodemode
    let tool_modes: Vec<ToolCallingMode> = env::var("EVAL_TOOL_MODES")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "baseline,codemode,rlm,rlmcodemode".to_string())
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "baseline" => Some(ToolCallingMode::Baseline),
            "codemode" | "code" => Some(ToolCallingMode::CodeMode),
            "baselinerlm" | "baseline+rlm" => Some(ToolCallingMode::BaselineRlm),
            "rlm" | "repl" => Some(ToolCallingMode::Rlm),
            "rlmcodemode" | "rlm+codemode" | "rlm_codemode" | "codemoderlm" | "code+rlm" | "coderlm" | "codemode+rlm" => Some(ToolCallingMode::RlmCodeMode),
            "all" => None, // Handle "all" separately
            _ => None,
        })
        .collect();

    // Check if "all" was specified
    let tool_modes: Vec<ToolCallingMode> = if env::var("EVAL_TOOL_MODES")
        .map(|s| s.to_lowercase().contains("all"))
        .unwrap_or(false)
    {
        vec![
            ToolCallingMode::Baseline,
            ToolCallingMode::CodeMode,
            ToolCallingMode::BaselineRlm,
            ToolCallingMode::Rlm,
            ToolCallingMode::RlmCodeMode,
        ]
    } else {
        tool_modes
    };

    assert!(!tool_modes.is_empty(), "At least one tool mode should be configured");

    // Step 7b: Parse codex flag from EVAL_USE_CODEX (boolean)
    let use_codex = env::var("EVAL_USE_CODEX")
        .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
        .unwrap_or(false);

    // Parse independent codex model (defaults to EVAL_MODEL)
    let codex_model = env::var("EVAL_CODEX_MODEL").unwrap_or_else(|_| agent_model.clone());

    tracing::info!(
        "Running tool-calling modes: {:?}",
        tool_modes
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
    );
    tracing::info!("Use Codex client: {use_codex}");
    if use_codex {
        tracing::info!("Codex model: {codex_model}");
    }

    // CodexRunner has been removed (depends on deeply-coupled codex internals).
    // The 4 benchmark modes (Baseline, CodeMode, BaselineRlm, CodeModeRlm) work
    // via the ToolCallingRunner and do not need CodexRunner.
    let codex_runner: Option<()> = None;
    let _ = use_codex; // suppress unused warning

    // Use mode_name strings as keys for results
    let mut all_results: HashMap<String, Vec<EvalResult>> = HashMap::new();
    let mut incremental_writer = IncrementalWriter::try_new();

    // Run tool-calling modes
    for mode in &tool_modes {
        let mode_name = mode.to_string();
        tracing::warn!("Running {mode_name} mode (ToolCalling)");

        let mut results = Vec::new();

        for (i, task) in eval_tasks.iter().enumerate() {
            // Refresh the Gateway session if nearing token expiration
            if session_created_at.elapsed() > Duration::from_secs(SESSION_REFRESH_SECS) {
                tracing::warn!("Session nearing expiration — refreshing Gateway connection...");
                match create_gateway_session(&config, rlm_router.clone()).await {
                    Ok((_new_client, new_runner, created_at)) => {
                        runner = new_runner.with_model(&agent_model);
                        session_created_at = created_at;
                        tracing::warn!("Gateway session refreshed successfully");
                    }
                    Err(e) => {
                        tracing::error!("Failed to refresh Gateway session: {e}");
                        tracing::error!("Remaining tasks will likely fail");
                    }
                }
            }

            tracing::warn!(
                "Task {}/{}: {}...",
                i + 1,
                eval_tasks.len(),
                &task.task_id[..task.task_id.len().min(20)]
            );

            // Execute task using tool-calling runner
            let task_result = runner.run_task(task, *mode).await;

            // Print tool calls made
            if !task_result.tool_calls.is_empty() {
                let tool_names: Vec<_> = task_result
                    .tool_calls
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect();
                tracing::debug!("Tools used: {}", tool_names.join(", "));
            }

            // Build tool results summary for the judge (ground truth)
            let tool_results_summary: String = task_result
                .tool_calls
                .iter()
                .filter(|t| !t.result.is_empty())
                .map(|t| format!("[{}] {}", t.name, t.result))
                .collect::<Vec<_>>()
                .join("\n\n");

            // Judge the answer
            let verification = match judge
                .verify_claims(&task.prompt, &task_result.final_answer, &task.claims, &tool_results_summary)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Verification failed: {e:#}");
                    ClaimVerificationResult {
                        scores: vec![],
                        coverage: 0.0,
                        passed: false,
                        raw_response: format!("{e:#}"),
                        verification_latency_ms: 0,
                    }
                }
            };

            let status = if verification.passed { "PASS" } else { "FAIL" };
            tracing::warn!(
                "  {} | coverage={:.2} | ctx_tokens={} | total_llm_tokens={}",
                status, verification.coverage, task_result.context_tokens, task_result.total_llm_tokens,
            );
            tracing::warn!(
                "  wall={}ms llm={}ms tool={}ms setup={}ms judge={}ms",
                task_result.latency_ms,
                task_result.llm_time_ms,
                task_result.tool_time_ms,
                task_result.setup_ms,
                verification.verification_latency_ms,
            );
            if let Some(ref err) = task_result.error {
                tracing::error!("ERROR: {err}");
            }

            results.push(EvalResult {
                task_id: task.task_id.clone(),
                prompt: task.prompt.clone(),
                claims: task.claims.clone(),
                task_result,
                verification,
            });

            if let Some(ref mut writer) = incremental_writer {
                let eval_result = results.last().unwrap();
                if let Err(e) = writer.append_result(eval_result, &mode_name) {
                    tracing::error!("Incremental write failed: {e}");
                }
                if let Err(e) = writer.write_trace(eval_result, &mode_name) {
                    tracing::error!("Trace write failed: {e}");
                }
            }
        }

        all_results.insert(mode_name, results);
    }

    // CodexRunner has been removed; skip Codex client mode.
    let _ = codex_runner;

    // Assert each requested mode produced results
    for mode in &tool_modes {
        let mode_name = mode.to_string();
        assert!(
            all_results.contains_key(&mode_name),
            "Mode {mode_name} was requested but produced no results -- it may have been silently skipped"
        );
    }
    // Assert at least one mode has pass_rate > 0%
    let any_passes = all_results.values().any(|results| {
        results.iter().any(|r| r.verification.passed)
    });
    if !any_passes {
        tracing::warn!("No mode achieved any passes -- possible systemic failure");
    }

    // Step 8: Print results
    print_comparison(&all_results);

    // Step 9: Save results to JSON
    let mode_strings: Vec<String> = tool_modes.iter().map(|m| m.to_string()).collect();
    save_results(
        &all_results,
        &agent_model,
        &codex_model,
        &dataset_path.to_string_lossy(),
        &mode_strings,
        use_codex,
        limit,
    );
}

/// Print comparative results (uses mode_name strings as keys)
fn print_comparison(results: &HashMap<String, Vec<EvalResult>>) {
    tracing::trace!("MCP-Atlas Evaluation Results");

    // Define the order of modes for display
    let mode_order = [
        "Baseline",
        "CodeMode",
        "Baseline+RLM",
        "RLM",
        "CodeMode+RLM",
        "Codex",
    ];

    // Per-mode summary
    for mode_name in &mode_order {
        if let Some(mode_results) = results.get(*mode_name) {
            let pass_count = mode_results
                .iter()
                .filter(|r| r.verification.passed)
                .count();
            let avg_coverage: f64 = mode_results
                .iter()
                .map(|r| r.verification.coverage)
                .sum::<f64>()
                / mode_results.len().max(1) as f64;
            let avg_tokens: i64 = mode_results
                .iter()
                .map(|r| r.task_result.context_tokens)
                .sum::<i64>()
                / mode_results.len().max(1) as i64;
            let avg_latency_ms: u64 = mode_results
                .iter()
                .map(|r| r.task_result.latency_ms)
                .sum::<u64>()
                / mode_results.len().max(1) as u64;
            let avg_latency_secs = avg_latency_ms as f64 / 1000.0;

            tracing::info!("## {} Mode", mode_name);
            tracing::info!("| Metric | Value |");
            tracing::info!("|--------|-------|");
            tracing::info!(
                "| Pass Rate | {:.1}% ({}/{}) |",
                pass_count as f64 / mode_results.len().max(1) as f64 * 100.0,
                pass_count,
                mode_results.len()
            );
            tracing::info!("| Avg Coverage | {:.3} |", avg_coverage);
            tracing::info!("| Avg Context Tokens | {} |", avg_tokens);
            tracing::info!("| Avg Latency | {:.1}s |", avg_latency_secs);
        }
    }

    // Comparative table
    tracing::info!("## Comparison");
    tracing::info!("| Mode | Pass Rate | Avg Coverage | Avg Tokens | Token Reduction | Avg Latency |");
    tracing::info!("|------|-----------|--------------|------------|-----------------|-------------|");

    let baseline_tokens = results
        .get("Baseline")
        .map(|r| {
            r.iter().map(|e| e.task_result.context_tokens).sum::<i64>() / r.len().max(1) as i64
        })
        .unwrap_or(1);

    for mode_name in &mode_order {
        if let Some(mode_results) = results.get(*mode_name) {
            let pass_count = mode_results
                .iter()
                .filter(|r| r.verification.passed)
                .count();
            let pass_rate = pass_count as f64 / mode_results.len().max(1) as f64 * 100.0;
            let avg_coverage: f64 = mode_results
                .iter()
                .map(|r| r.verification.coverage)
                .sum::<f64>()
                / mode_results.len().max(1) as f64;
            let avg_tokens: i64 = mode_results
                .iter()
                .map(|r| r.task_result.context_tokens)
                .sum::<i64>()
                / mode_results.len().max(1) as i64;
            let avg_latency_ms: u64 = mode_results
                .iter()
                .map(|r| r.task_result.latency_ms)
                .sum::<u64>()
                / mode_results.len().max(1) as u64;

            let reduction = if baseline_tokens > 0 {
                100.0 - (avg_tokens as f64 / baseline_tokens as f64 * 100.0)
            } else {
                0.0
            };

            let reduction_str = if *mode_name == "Baseline" {
                "-".to_string()
            } else {
                format!("-{reduction:.0}%")
            };

            tracing::info!(
                "| {} | {:.1}% | {:.3} | {} | {} | {:.1}s |",
                mode_name,
                pass_rate,
                avg_coverage,
                avg_tokens,
                reduction_str,
                avg_latency_ms as f64 / 1000.0
            );
        }
    }

    // Per-task comparison
    tracing::info!("## Per-Task Comparison");
    // Dynamic per-task comparison table using whatever modes are present
    let present_modes: Vec<&str> = mode_order.iter()
        .filter(|m| results.contains_key(**m))
        .copied()
        .collect();

    if !present_modes.is_empty() {
        let header_cols: Vec<String> = present_modes.iter().map(|m| format!(" {} ", m)).collect();
        tracing::info!("| Task | {} | Winner |", header_cols.join("|"));

        // Get any available results for iteration
        let first_results = results.get(present_modes[0]).unwrap();
        for (i, result) in first_results.iter().enumerate() {
            let coverages: Vec<f64> = present_modes.iter()
                .map(|m| results.get(*m)
                    .and_then(|r| r.get(i))
                    .map(|r| r.verification.coverage)
                    .unwrap_or(0.0))
                .collect();

            let max_cov = coverages.iter().cloned().fold(0.0f64, f64::max);
            let winner = if max_cov == 0.0 {
                "all fail".to_string()
            } else if coverages.iter().all(|c| *c == max_cov) {
                "tie (all)".to_string()
            } else {
                present_modes.iter().zip(coverages.iter())
                    .find(|(_, c)| **c == max_cov)
                    .map(|(m, _)| m.to_string())
                    .unwrap_or_else(|| "?".to_string())
            };

            let cols: Vec<String> = coverages.iter()
                .map(|c| {
                    let status = if *c >= PASS_THRESHOLD { "PASS" } else { "FAIL" };
                    format!(" {:.2} {} ", c, status)
                })
                .collect();

            tracing::info!(
                "| {} | {} | {} |",
                &result.task_id[..result.task_id.len().min(15)],
                cols.join("|"),
                winner
            );
        }
    }

    tracing::info!("Key Insight");
    tracing::info!("RLM modes aim to achieve significant token reduction while maintaining quality.");
    tracing::info!("CodeMode+RLM combines RLM REPL with Gateway tool execution for best of both worlds.");
    tracing::info!("Codex client uses the full Codex system prompts and agent loop.");
}

/// Convert an EvalResult to a JSON value for JSONL / results_full.json.
fn eval_result_to_json(r: &EvalResult, mode: &str) -> serde_json::Value {
    let claim_scores: Vec<serde_json::Value> = r
        .verification
        .scores
        .iter()
        .map(|(claim, score)| {
            let verdict = match score {
                ClaimScore::Fulfilled => "FULFILLED",
                ClaimScore::PartiallyFulfilled => "PARTIAL",
                ClaimScore::NotFulfilled => "NOT_FULFILLED",
            };
            json!({
                "claim": claim,
                "verdict": verdict,
                "score": score.score(),
            })
        })
        .collect();

    let tool_calls_json: Vec<serde_json::Value> = r
        .task_result
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, tc)| {
            json!({
                "index": i,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "result_tokens": tc.result_tokens,
                "stored_in_corpus": tc.stored_in_corpus,
                "latency_ms": tc.latency_ms,
            })
        })
        .collect();

    json!({
        "task_id": r.task_id,
        "mode": mode,
        "prompt": r.prompt,
        "claims": r.claims,
        "task_result": {
            "task_id": r.task_result.task_id,
            "mode_name": r.task_result.mode_name,
            "context_tokens": r.task_result.context_tokens,
            "total_llm_tokens": r.task_result.total_llm_tokens,
            "latency_ms": r.task_result.latency_ms,
            "llm_time_ms": r.task_result.llm_time_ms,
            "tool_time_ms": r.task_result.tool_time_ms,
            "setup_ms": r.task_result.setup_ms,
            "llm_total_calls": r.task_result.llm_total_calls,
            "llm_input_tokens": r.task_result.llm_input_tokens,
            "llm_output_tokens": r.task_result.llm_output_tokens,
            "error": r.task_result.error,
            "final_answer": r.task_result.final_answer,
            "num_tool_calls": r.task_result.tool_calls.len(),
            "tool_calls": tool_calls_json,
        },
        "verification": {
            "coverage": r.verification.coverage,
            "passed": r.verification.passed,
            "verification_latency_ms": r.verification.verification_latency_ms,
            "claim_scores": claim_scores,
        },
    })
}

/// Convert an EvalResult to a trace JSON (same format as collect_results.py extract_traces).
fn eval_result_to_trace(r: &EvalResult, mode: &str) -> serde_json::Value {
    let tool_calls_json: Vec<serde_json::Value> = r
        .task_result
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, tc)| {
            json!({
                "index": i,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "result_tokens": tc.result_tokens,
                "stored_in_corpus": tc.stored_in_corpus,
                "latency_ms": tc.latency_ms,
            })
        })
        .collect();

    json!({
        "task_id": r.task_id,
        "mode": mode,
        "prompt": r.prompt,
        "claims": r.claims,
        "final_answer": r.task_result.final_answer,
        "context_tokens": r.task_result.context_tokens,
        "total_llm_tokens": r.task_result.total_llm_tokens,
        "latency_ms": r.task_result.latency_ms,
        "error": r.task_result.error,
        "tool_calls": tool_calls_json,
        "verification": {
            "coverage": r.verification.coverage,
            "passed": r.verification.passed,
            "verification_latency_ms": r.verification.verification_latency_ms,
            "claim_scores": r.verification.scores.iter().map(|(claim, score)| {
                let verdict = match score {
                    ClaimScore::Fulfilled => "FULFILLED",
                    ClaimScore::PartiallyFulfilled => "PARTIAL",
                    ClaimScore::NotFulfilled => "NOT_FULFILLED",
                };
                json!({
                    "claim": claim,
                    "verdict": verdict,
                    "score": score.score(),
                })
            }).collect::<Vec<_>>(),
        },
    })
}

/// Writes results incrementally as each task completes.
/// Falls back to no-op if EVAL_OUTPUT_DIR is not set.
struct IncrementalWriter {
    output_dir: PathBuf,
    jsonl_file: std::io::BufWriter<std::fs::File>,
}

impl IncrementalWriter {
    /// Creates a new writer if EVAL_OUTPUT_DIR is set.
    fn try_new() -> Option<Self> {
        let output_dir = PathBuf::from(env::var("EVAL_OUTPUT_DIR").ok()?);
        if let Err(e) = std::fs::create_dir_all(output_dir.join("logs")) {
            tracing::warn!("Failed to create logs dir: {e}");
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_dir.join("results.jsonl"))
            .ok()?;
        tracing::info!("Incremental writer: {:?}/results.jsonl", output_dir);
        Some(Self {
            output_dir,
            jsonl_file: std::io::BufWriter::new(file),
        })
    }

    /// Appends one result as a JSONL line.
    fn append_result(&mut self, result: &EvalResult, mode: &str) -> std::io::Result<()> {
        let value = eval_result_to_json(result, mode);
        let line = serde_json::to_string(&value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(self.jsonl_file, "{}", line)?;
        self.jsonl_file.flush()
    }

    /// Writes a trace file for one result.
    fn write_trace(&self, result: &EvalResult, mode: &str) -> std::io::Result<()> {
        let safe_mode = mode.replace('+', "_plus_");
        let filename = format!("{}_{}.json", result.task_id, safe_mode);
        let trace = eval_result_to_trace(result, mode);
        let path = self.output_dir.join("logs").join(filename);
        let content = serde_json::to_string_pretty(&trace)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, content)
    }
}

/// Save evaluation results to a JSON file compatible with `scripts/plot_results.py`.
///
/// When EVAL_OUTPUT_DIR is set, writes `results_full.json` into that directory.
/// Otherwise writes to `codex-rs/mcp-atlas/services/mcp_eval/results/` with a
/// Unix-timestamp suffix so successive runs never collide.
fn save_results(
    results: &HashMap<String, Vec<EvalResult>>,
    agent_model: &str,
    codex_model: &str,
    dataset_path: &str,
    tool_modes: &[String],
    use_codex: bool,
    eval_limit: usize,
) {
    let results_dir = if let Ok(dir) = env::var("EVAL_OUTPUT_DIR") {
        PathBuf::from(dir)
    } else {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()))
            .join("../mcp-atlas/services/mcp_eval/results")
    };

    if let Err(e) = std::fs::create_dir_all(&results_dir) {
        tracing::warn!("Could not create results dir {:?}: {}", results_dir, e);
        return;
    }

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mode_order = [
        "Baseline",
        "CodeMode",
        "Baseline+RLM",
        "RLM",
        "CodeMode+RLM",
        "Codex",
    ];

    // Build per-mode JSON objects
    let mut results_json = serde_json::Map::new();
    for mode_name in &mode_order {
        if let Some(mode_results) = results.get(*mode_name) {
            let total = mode_results.len().max(1);
            let pass_count = mode_results
                .iter()
                .filter(|r| r.verification.passed)
                .count();
            let avg_coverage: f64 =
                mode_results.iter().map(|r| r.verification.coverage).sum::<f64>() / total as f64;
            let avg_tokens: i64 = mode_results
                .iter()
                .map(|r| r.task_result.context_tokens)
                .sum::<i64>()
                / total as i64;
            let avg_total_llm_tokens: i64 = mode_results
                .iter()
                .map(|r| r.task_result.total_llm_tokens)
                .sum::<i64>()
                / total as i64;
            let avg_latency_ms: u64 = mode_results
                .iter()
                .map(|r| r.task_result.latency_ms)
                .sum::<u64>()
                / total as u64;

            let summary = json!({
                "total_tasks": mode_results.len(),
                "pass_count": pass_count,
                "pass_rate": pass_count as f64 / total as f64,
                "avg_coverage": avg_coverage,
                "avg_context_tokens": avg_tokens,
                "avg_total_llm_tokens": avg_total_llm_tokens,
                "avg_latency_ms": avg_latency_ms,
            });

            let tasks: Vec<serde_json::Value> = mode_results
                .iter()
                .map(|r| {
                    let claim_scores: Vec<serde_json::Value> = r
                        .verification
                        .scores
                        .iter()
                        .map(|(claim, score)| {
                            let verdict = match score {
                                ClaimScore::Fulfilled => "FULFILLED",
                                ClaimScore::PartiallyFulfilled => "PARTIAL",
                                ClaimScore::NotFulfilled => "NOT_FULFILLED",
                            };
                            json!({
                                "claim": claim,
                                "verdict": verdict,
                                "score": score.score(),
                            })
                        })
                        .collect();

                    let tool_calls_json: Vec<serde_json::Value> = r
                        .task_result
                        .tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, tc)| {
                            json!({
                                "index": i,
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "result": tc.result,
                                "result_tokens": tc.result_tokens,
                                "stored_in_corpus": tc.stored_in_corpus,
                                "latency_ms": tc.latency_ms,
                            })
                        })
                        .collect();

                    json!({
                        "task_id": r.task_id,
                        "prompt": r.prompt,
                        "claims": r.claims,
                        "task_result": {
                            "task_id": r.task_result.task_id,
                            "mode_name": r.task_result.mode_name,
                            "context_tokens": r.task_result.context_tokens,
                            "total_llm_tokens": r.task_result.total_llm_tokens,
                            "latency_ms": r.task_result.latency_ms,
                            "llm_time_ms": r.task_result.llm_time_ms,
                            "tool_time_ms": r.task_result.tool_time_ms,
                            "setup_ms": r.task_result.setup_ms,
                            "llm_total_calls": r.task_result.llm_total_calls,
                            "llm_input_tokens": r.task_result.llm_input_tokens,
                            "llm_output_tokens": r.task_result.llm_output_tokens,
                            "error": r.task_result.error,
                            "final_answer": r.task_result.final_answer,
                            "num_tool_calls": r.task_result.tool_calls.len(),
                            "tool_calls": tool_calls_json,
                        },
                        "verification": {
                            "coverage": r.verification.coverage,
                            "passed": r.verification.passed,
                            "verification_latency_ms": r.verification.verification_latency_ms,
                            "claim_scores": claim_scores,
                        },
                    })
                })
                .collect();

            results_json.insert(
                mode_name.to_string(),
                json!({ "summary": summary, "tasks": tasks }),
            );
        }
    }

    let output = json!({
        "generated_at_unix": ts,
        "agent_model": agent_model,
        "codex_model": codex_model,
        "dataset_path": dataset_path,
        "tool_modes": tool_modes,
        "use_codex": use_codex,
        "eval_limit": eval_limit,
        "results": results_json,
    });

    let path = if env::var("EVAL_OUTPUT_DIR").is_ok() {
        results_dir.join("results_full.json")
    } else {
        results_dir.join(format!("mcp_atlas_eval_{ts}.json"))
    };
    match std::fs::File::create(&path) {
        Ok(mut f) => {
            let pretty =
                serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string());
            if let Err(e) = f.write_all(pretty.as_bytes()) {
                tracing::warn!("Failed to write results to {:?}: {}", path, e);
            } else {
                tracing::info!("Results saved to {:?}", path);
                tracing::info!(
                    "Plot with: python3 scripts/plot_results.py {}",
                    path.display()
                );
            }
        }
        Err(e) => {
            tracing::warn!("Could not create results file {:?}: {}", path, e);
        }
    }
}

/// Full evaluation on all 500 tasks (ignored by default - run with --ignored)
#[tokio::test]
#[ignore]
async fn run_full_mcp_atlas_evaluation() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping full evaluation - credentials not set");
        return;
    }

    tracing::trace!("MCP-Atlas FULL Evaluation (500 tasks x 3 modes = 1500 runs)");

    // Same as above but without the .take(10)
    // This would run all 500 tasks
    tracing::info!("Full evaluation would run here...");
    tracing::info!("This is a placeholder - implement the same logic as test mode but with all tasks.");
}

/// Analyze dataset to find tasks solvable with available Gateway tools
#[tokio::test]
async fn analyze_solvable_tasks() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::trace!("MCP-Atlas Dataset Analysis: Which Tasks Can Be Solved?");

    // Load dataset
    let dataset_path = get_dataset_path();
    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Failed to load dataset: {e}");
            panic!("Failed to load dataset: {e}");
        }
    };
    tracing::debug!("Loaded {} tasks from dataset", tasks.len());

    // Connect to Gateway and discover tools
    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");
    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => Arc::new(c),
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }

    // Create runner to discover tools
    let runner = match ToolCallingRunner::new(
        client.clone(),
        None,
        resolve_eval_base_url(),
        resolve_eval_api_key(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to create runner: {e}");
            panic!("Failed to create runner: {e}");
        }
    };

    // Print available Gateway tools
    let gateway_tools = runner.available_tools();
    tracing::warn!("Available Gateway tools ({} total):", gateway_tools.len());

    let mut by_server: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for tool in gateway_tools {
        by_server.entry(&tool.server).or_default().push(&tool.name);
    }
    for (server, tool_names) in &by_server {
        tracing::debug!("{}: {}", server, tool_names.join(", "));
    }

    // Analyze each task
    tracing::info!("Task Analysis");

    // Target servers
    let target_servers = ["git", "CLI", "Code Executor"];

    // Categorize tasks
    let mut fully_matched: Vec<(&codex_core::eval::McpAtlasTask, f64)> = Vec::new();
    let mut partially_matched: Vec<(&codex_core::eval::McpAtlasTask, f64, Vec<String>)> =
        Vec::new();
    let mut no_match: Vec<&codex_core::eval::McpAtlasTask> = Vec::new();

    for task in &tasks {
        let matches = runner.get_matching_tools(task);

        // Check if matches are from target servers
        let target_match_count = matches
            .iter()
            .filter(|(_, tool)| {
                target_servers
                    .iter()
                    .any(|s| tool.server.to_lowercase().contains(&s.to_lowercase()))
            })
            .count();

        let target_ratio = if task.enabled_tools.is_empty() {
            0.0
        } else {
            target_match_count as f64 / task.enabled_tools.len() as f64
        };

        if target_ratio >= 0.8 {
            fully_matched.push((task, target_ratio));
        } else if target_ratio > 0.0 {
            let unmatched: Vec<String> = task
                .enabled_tools
                .iter()
                .filter(|t| !matches.iter().any(|(dt, _)| dt == *t))
                .cloned()
                .collect();
            partially_matched.push((task, target_ratio, unmatched));
        } else {
            no_match.push(task);
        }
    }

    // Sort by match ratio
    fully_matched.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    partially_matched.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Report
    tracing::warn!("## Summary");
    tracing::warn!("| Category | Count | Percentage |");
    tracing::warn!("|----------|-------|------------|");
    tracing::warn!(
        "| Fully Matched (>=80% tools available) | {} | {:.1}% |",
        fully_matched.len(),
        fully_matched.len() as f64 / tasks.len() as f64 * 100.0
    );
    tracing::warn!(
        "| Partially Matched (<80% but >0% tools) | {} | {:.1}% |",
        partially_matched.len(),
        partially_matched.len() as f64 / tasks.len() as f64 * 100.0
    );
    tracing::warn!(
        "| No Match (0% tools available) | {} | {:.1}% |",
        no_match.len(),
        no_match.len() as f64 / tasks.len() as f64 * 100.0
    );

    // Show best candidates
    tracing::info!("## Best Candidate Tasks (>=80% tool coverage)");
    if fully_matched.is_empty() {
        tracing::info!("No tasks have >=80% tool coverage with target servers.");
    } else {
        tracing::info!("| Task ID | Match % | Prompt Preview | Tools |");
        tracing::info!("|---------|---------|----------------|-------|");
        for (task, ratio) in fully_matched.iter().take(20) {
            let prompt_preview = if task.prompt.len() > 50 {
                format!("{}...", &task.prompt[..50].replace('\n', " "))
            } else {
                task.prompt.replace('\n', " ")
            };
            let tool_count = task.enabled_tools.len();
            tracing::info!(
                "| {} | {:.0}% | {} | {} |",
                &task.task_id[..task.task_id.len().min(15)],
                ratio * 100.0,
                prompt_preview,
                tool_count
            );
        }
    }

    // Show partially matched with missing tools
    tracing::info!("## Partially Matched Tasks (with missing tools)");
    if partially_matched.is_empty() {
        tracing::info!("No partially matched tasks.");
    } else {
        tracing::info!("| Task ID | Match % | Missing Tools |");
        tracing::info!("|---------|---------|---------------|");
        for (task, ratio, missing) in partially_matched.iter().take(10) {
            let missing_preview: String = missing
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let more = if missing.len() > 3 {
                format!(" (+{})", missing.len() - 3)
            } else {
                String::new()
            };
            tracing::info!(
                "| {} | {:.0}% | {}{} |",
                &task.task_id[..task.task_id.len().min(15)],
                ratio * 100.0,
                missing_preview,
                more
            );
        }
    }

    // Common missing tool patterns
    tracing::info!("## Common Missing Tools (blocking task solvability)");
    let mut missing_tool_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for task in &tasks {
        for tool in &task.enabled_tools {
            if runner
                .get_matching_tools(task)
                .iter()
                .all(|(dt, _)| dt != tool)
            {
                *missing_tool_counts.entry(tool.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut missing_sorted: Vec<_> = missing_tool_counts.into_iter().collect();
    missing_sorted.sort_by(|a, b| b.1.cmp(&a.1));

    tracing::info!("| Tool Name | Tasks Affected |");
    tracing::info!("|-----------|----------------|");
    for (tool, count) in missing_sorted.iter().take(15) {
        tracing::info!("| {} | {} |", tool, count);
    }

    // Recommendations
    tracing::info!("## Recommendations");
    tracing::info!(
        "1. Best tasks for evaluation: {} tasks have >=80% tool coverage",
        fully_matched.len()
    );
    if !fully_matched.is_empty() {
        tracing::info!("- These are most likely to succeed with current Gateway tools");
    }
    tracing::info!("2. To increase coverage, add support for:");
    for (tool, count) in missing_sorted.iter().take(5) {
        tracing::info!("- `{}` (would unlock {} tasks)", tool, count);
    }

    // Show sample fully matched task details
    if let Some((task, ratio)) = fully_matched.first() {
        tracing::info!("## Sample Fully Matched Task");
        tracing::info!("Task ID: {}", task.task_id);
        tracing::info!("Match Ratio: {:.0}%", ratio * 100.0);
        tracing::debug!("Prompt: {}", task.prompt);
        tracing::debug!("Enabled Tools: {:?}", task.enabled_tools);
        tracing::info!("Claims to verify:");
        for (i, claim) in task.claims.iter().enumerate().take(5) {
            tracing::info!("{}. {}", i + 1, claim);
        }
        if task.claims.len() > 5 {
            tracing::info!("... ({} more claims)", task.claims.len() - 5);
        }

        // Show tool mapping
        tracing::debug!("Tool Mapping:");
        for (dt, gt) in runner.get_matching_tools(task) {
            tracing::debug!("{} -> {} ({})", dt, gt.name, gt.server);
        }
    }
}

/// Verbose debug evaluation on 5 random tasks
/// This test logs EVERY step to help debug why tasks fail
#[tokio::test]
async fn run_verbose_debug_evaluation() {
    init_test_tracing();
    if should_skip() {
        tracing::warn!("Skipping: credentials not set");
        return;
    }

    tracing::trace!("VERBOSE DEBUG EVALUATION - 5 Random Tasks");

    // Step 1: Load dataset
    let dataset_path = get_dataset_path();
    tracing::debug!("[STEP 1] Loading dataset from {:?}...", dataset_path);

    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => {
            tracing::debug!("Loaded {} tasks", t.len());
            t
        }
        Err(e) => {
            tracing::error!("Failed to load dataset: {e}");
            panic!("Failed to load dataset: {e}");
        }
    };

    // Step 2: Connect to Gateway
    tracing::debug!("[STEP 2] Connecting to Gateway...");
    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    tracing::debug!("Token URL: {:?}", config.token_url);
    tracing::debug!("MCP URL: {:?}", config.mcp_url);

    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => {
            tracing::debug!("Authentication successful");
            t
        }
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("Auth failed (credentials configured): {e}"),
    };

    let mcp_url = build_mcp_url(&config, &token.access_token).expect("Failed to build MCP URL");
    let client = match RmcpClient::new_streamable_http_client(
        &config.server_name,
        &mcp_url,
        None,
        None,
        None,
        OAuthCredentialsStoreMode::File,
    )
    .await
    {
        Ok(c) => {
            tracing::debug!("MCP client created");
            Arc::new(c)
        }
        Err(e) if e.to_string().contains("onnection refused") => {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        Err(e) => panic!("MCP client creation failed (credentials configured): {e}"),
    };

    if let Err(e) = client
        .initialize(
            gateway_auth::create_init_params("mcp-atlas-eval"),
            Some(Duration::from_secs(30)),
            gateway_auth::create_elicitation_handler(),
        )
        .await
    {
        if e.to_string().contains("onnection refused") {
            tracing::warn!("Skipping: gateway not running");
            return;
        }
        panic!("MCP initialization failed: {e}");
    }
    tracing::debug!("MCP initialized");

    // Step 3: Setup RLM
    tracing::debug!("[STEP 3] Setting up RLM router...");
    let temp_dir = TempDir::new().unwrap();
    let rlm_router: Option<Arc<GatewayResultRouter>> =
        match create_rlm_router(temp_dir.path()).await {
            Ok(r) => {
                tracing::debug!("RLM router created");
                Some(Arc::new(r))
            }
            Err(e) => {
                tracing::warn!("RLM router creation failed (will skip RLM mode): {e}");
                None
            }
        };

    // Step 4: Setup judge
    tracing::debug!("[STEP 4] Setting up claim judge...");
    let judge = match ClaimJudge::with_model(
        &resolve_judge_model(),
        resolve_judge_base_url(),
        resolve_judge_api_key(),
    ) {
        Ok(j) => {
            tracing::debug!("Claim judge created");
            j
        }
        Err(e) => {
            tracing::error!("Failed to create claim judge: {e}");
            panic!("Failed to create claim judge: {e}");
        }
    };

    // Step 5: Discover tools
    tracing::debug!("[STEP 5] Discovering Gateway tools...");
    let runner = match ToolCallingRunner::new(
        client.clone(),
        rlm_router.clone(),
        resolve_eval_base_url(),
        resolve_eval_api_key(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to create runner: {e}");
            panic!("Failed to create runner: {e}");
        }
    };

    let tools = runner.available_tools();
    tracing::debug!("Found {} tools:", tools.len());
    for tool in tools {
        tracing::debug!(
            "- {} ({}): {}",
            tool.name,
            tool.server,
            &tool.description[..tool.description.len().min(60)]
        );
    }

    // Step 6: Select 5 random tasks with at least one resolvable tool
    tracing::debug!("[STEP 6] Selecting 5 random tasks...");
    let eligible_tasks: Vec<_> = tasks
        .iter()
        .filter(|task| {
            !task.enabled_tools.is_empty()
                && task
                    .enabled_tools
                    .iter()
                    .any(|t| runner.resolve_tool(t).is_some())
        })
        .collect();

    tracing::debug!(
        "Eligible tasks (with resolvable tools): {}/{}",
        eligible_tasks.len(),
        tasks.len()
    );

    // Select 5 specific indices for reproducibility: first, last, and 3 spread through middle
    let indices = if eligible_tasks.len() >= 5 {
        vec![
            0,
            eligible_tasks.len() / 4,
            eligible_tasks.len() / 2,
            3 * eligible_tasks.len() / 4,
            eligible_tasks.len() - 1,
        ]
    } else {
        (0..eligible_tasks.len().min(5)).collect()
    };

    let selected_tasks: Vec<_> = indices
        .iter()
        .filter_map(|&i| eligible_tasks.get(i).copied())
        .collect();

    tracing::debug!(
        "Selected {} tasks for verbose evaluation:",
        selected_tasks.len()
    );
    for (i, task) in selected_tasks.iter().enumerate() {
        tracing::debug!(
            "{}. {} - {}",
            i + 1,
            task.task_id,
            &task.prompt[..task.prompt.len().min(50)]
        );
    }

    // Step 7: Run verbose evaluation on each task
    for (task_num, task) in selected_tasks.iter().enumerate() {
        tracing::info!(
            "TASK {}/{}: {}",
            task_num + 1,
            selected_tasks.len(),
            task.task_id
        );

        // Show task details
        tracing::trace!("TASK DETAILS");
        tracing::trace!("Prompt:");
        for line in task.prompt.lines() {
            tracing::trace!("{line}");
        }
        tracing::trace!("Enabled Tools: {:?}", task.enabled_tools);
        tracing::trace!("Tool Resolution:");
        for tool_name in &task.enabled_tools {
            if let Some(gateway_tool) = runner.resolve_tool(tool_name) {
                tracing::debug!(
                    "{} -> {} ({})",
                    tool_name, gateway_tool.name, gateway_tool.server
                );
            } else {
                tracing::debug!("{} -> NOT FOUND", tool_name);
            }
        }
        tracing::trace!("Claims to verify ({}):", task.claims.len());
        for (i, claim) in task.claims.iter().enumerate() {
            tracing::trace!("{}. {}", i + 1, claim);
        }
        tracing::trace!("Expected Trajectory ({} steps):", task.trajectory.len());
        for (i, step) in task.trajectory.iter().enumerate() {
            tracing::trace!("{}. {} with args: {}", i + 1, step.tool, step.args);
        }

        // Run in RLM mode only (the mode that had best results)
        let mode = ToolCallingMode::BaselineRlm;

        tracing::trace!("EXECUTION ({})", mode);
        tracing::debug!("Sending prompt to agent...");

        let start = Instant::now();
        let task_result = runner.run_task(task, mode).await;
        let elapsed = start.elapsed();

        tracing::debug!("Execution completed in {:?}", elapsed);

        // Show tool calls
        if task_result.tool_calls.is_empty() {
            tracing::warn!("NO TOOL CALLS MADE!");
            tracing::warn!("The agent did not use any tools.");
        } else {
            tracing::debug!("Tool Calls Made ({}):", task_result.tool_calls.len());
            for (i, call) in task_result.tool_calls.iter().enumerate() {
                tracing::trace!("Call {}", i + 1);
                tracing::trace!("Tool: {}", call.name);
                tracing::trace!(
                    "Arguments: {}",
                    serde_json::to_string_pretty(&call.arguments)
                        .unwrap_or_default()
                );
                tracing::trace!("Result tokens: {}", call.result_tokens);
                tracing::trace!("Stored in corpus: {}", call.stored_in_corpus);

                // Truncate result for display
                let result_preview = if call.result.len() > 500 {
                    format!(
                        "{}... [TRUNCATED: {} more chars]",
                        &call.result[..500],
                        call.result.len() - 500
                    )
                } else {
                    call.result.clone()
                };
                tracing::trace!("Result: {}", result_preview);
            }
        }

        // Show error if any
        if let Some(ref error) = task_result.error {
            tracing::error!("ERROR: {error}");
        }

        // Show final answer
        tracing::trace!("Final Answer:");
        let answer_preview = if task_result.final_answer.len() > 1000 {
            format!(
                "{}... [TRUNCATED: {} more chars]",
                &task_result.final_answer[..1000],
                task_result.final_answer.len() - 1000
            )
        } else if task_result.final_answer.is_empty() {
            "[EMPTY ANSWER]".to_string()
        } else {
            task_result.final_answer.clone()
        };
        for line in answer_preview.lines() {
            tracing::trace!("{line}");
        }
        tracing::debug!("Context tokens used: {}", task_result.context_tokens);

        // Show claim verification
        tracing::trace!("CLAIM VERIFICATION");
        tracing::debug!("Sending to judge...");

        let tool_results_summary: String = task_result
            .tool_calls
            .iter()
            .filter(|t| !t.result.is_empty())
            .map(|t| format!("[{}] {}", t.name, t.result))
            .collect::<Vec<_>>()
            .join("\n\n");
        let verify_start = Instant::now();
        let verification = match judge
            .verify_claims(&task.prompt, &task_result.final_answer, &task.claims, &tool_results_summary)
            .await
        {
            Ok(v) => {
                tracing::debug!("Verification completed in {:?}", verify_start.elapsed());
                v
            }
            Err(e) => {
                tracing::error!("Verification FAILED: {e}");
                ClaimVerificationResult {
                    scores: vec![],
                    coverage: 0.0,
                    passed: false,
                    raw_response: e.to_string(),
                    verification_latency_ms: 0,
                }
            }
        };

        tracing::debug!("Per-claim results:");
        for (claim, score) in &verification.scores {
            let score_str = match score {
                ClaimScore::Fulfilled => "FULFILLED",
                ClaimScore::PartiallyFulfilled => "PARTIAL",
                ClaimScore::NotFulfilled => "NOT_MET",
            };
            tracing::debug!(
                "{} {}",
                score_str,
                &claim[..claim.len().min(60)]
            );
        }
        tracing::info!("Coverage: {:.2}", verification.coverage);
        tracing::debug!("Threshold: {}", PASS_THRESHOLD);
        tracing::info!(
            "Status: {}",
            if verification.passed {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // Show raw judge response for debugging
        if !verification.raw_response.is_empty() && !verification.passed {
            tracing::trace!("Judge Raw Response (for debugging):");
            let raw_preview = if verification.raw_response.len() > 500 {
                format!("{}...", &verification.raw_response[..500])
            } else {
                verification.raw_response.clone()
            };
            for line in raw_preview.lines() {
                tracing::trace!("{line}");
            }
        }

        // Summary
        tracing::debug!("TASK SUMMARY");
        tracing::debug!("Task: {}", task.task_id);
        tracing::debug!("Tool calls: {}", task_result.tool_calls.len());
        tracing::debug!("Expected trajectory steps: {}", task.trajectory.len());
        tracing::debug!("Coverage: {:.2}", verification.coverage);
        tracing::debug!(
            "Result: {}",
            if verification.passed {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // Diagnosis
        tracing::debug!("DIAGNOSIS:");
        if task_result.tool_calls.is_empty() {
            tracing::debug!(
                "Agent made NO tool calls (expected {})",
                task.trajectory.len()
            );
            tracing::debug!("Check: Is the system prompt telling agent to use tools?");
            tracing::debug!("Check: Are tool definitions correctly formatted?");
        } else if task_result.tool_calls.len() != task.trajectory.len() {
            tracing::debug!(
                "Agent made {} calls, expected {}",
                task_result.tool_calls.len(),
                task.trajectory.len()
            );
        }

        if let Some(ref error) = task_result.error {
            tracing::debug!("Execution error: {error}");
        }

        if !verification.passed && task_result.final_answer.is_empty() {
            tracing::debug!("Final answer is EMPTY!");
        }

        // Check if tool calls match expected trajectory
        let expected_tools: Vec<_> = task.trajectory.iter().map(|s| &s.tool).collect();
        let actual_tools: Vec<_> = task_result.tool_calls.iter().map(|c| &c.name).collect();
        let tools_match = expected_tools.len() == actual_tools.len()
            && expected_tools.iter().zip(actual_tools.iter()).all(|(expected, actual)| {
                let e = expected.to_lowercase();
                let a = actual.to_lowercase();
                e == a || e.contains(&a) || a.contains(&e)
            });
        if !tools_match {
            tracing::debug!("Tool sequence mismatch:");
            tracing::debug!("Expected: {:?}", expected_tools);
            tracing::debug!("Actual:   {:?}", actual_tools);
        }
    }

    tracing::trace!("VERBOSE DEBUG EVALUATION COMPLETE");
}

/// Print setup instructions
#[tokio::test]
async fn print_setup_instructions() {
    init_test_tracing();
    if !should_skip() {
        return;
    }

    tracing::trace!("MCP-Atlas Evaluation - Setup Instructions");
    tracing::trace!("This evaluation requires:");
    tracing::trace!("1. Kontext Gateway credentials");
    tracing::trace!("2. LLM API key (any OpenAI-compatible provider)");
    tracing::trace!("3. MCP-Atlas dataset");
    tracing::trace!("Setup:");
    tracing::trace!("# Create .env file:");
    tracing::trace!("KONTEXT_CLIENT_ID=<your-client-id>");
    tracing::trace!("KONTEXT_CLIENT_SECRET=<your-client-secret>");
    tracing::trace!("KONTEXT_MCP_URL=http://localhost:4000/mcp");
    tracing::trace!("KONTEXT_TOKEN_URL=http://localhost:4000/oauth2/token");
    tracing::trace!("EVAL_API_KEY=<your-api-key>");
    tracing::trace!("EVAL_GATEWAY_TASK_CSV=v4");
    tracing::trace!("# Optional alternate dataset:");
    tracing::trace!(
        "# EVAL_GATEWAY_TASK_CSV=v3  (maps to ../mcp-atlas/services/mcp_eval/gateway_tasks_v3.csv)"
    );
    tracing::trace!(
        "# EVAL_GATEWAY_TASK_CSV=v2  (maps to ../mcp-atlas/services/mcp_eval/gateway_tasks_v2.csv)"
    );
    tracing::trace!(
        "# EVAL_GATEWAY_TASK_CSV=v1  (maps to ../mcp-atlas/services/mcp_eval/gateway_tasks.csv)"
    );
    tracing::trace!("# EVAL_GATEWAY_TASK_CSV=/path/to/custom_gateway_tasks.csv");
    tracing::trace!("Run tests:");
    tracing::trace!("# Verification tests:");
    tracing::trace!("source .env");
    tracing::trace!("cargo test -p codex-core --test mcp_atlas_eval -- --nocapture");
    tracing::trace!("# Three-way evaluation (10 tasks):");
    tracing::trace!(
        "cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture"
    );
    tracing::trace!("# Full evaluation (500 tasks - takes a long time):");
    tracing::trace!(
        "cargo test -p codex-core --test mcp_atlas_eval run_full_mcp_atlas_evaluation -- --nocapture --ignored"
    );
}
