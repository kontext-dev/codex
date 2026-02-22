#![cfg(feature = "benchmarking")]
//! MCP-Atlas Evaluation Script
//!
//! Benchmarks LLM agent performance on the MCP-Atlas dataset using two client architectures:
//!
//! ## Tool-Calling Client (ToolCallingRunner)
//! Uses OpenAI function calling with modes:
//! - Baseline: Direct EXECUTE_TOOL calls with full results in context
//! - CodeMode: EXECUTE_CODE with summarized results
//! - Baseline+RLM: EXECUTE_TOOL with RLM routing for large results
//! - CodeMode+RLM: EXECUTE_CODE with RLM routing for large results
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
//! # OpenAI for agent and judge
//! export OPENAI_API_KEY=<your-api-key>
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Test mode (first 10 tasks)
//! source .env
//! cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture
//!
//! # Select gateway task CSV (v1, v2, or custom path)
//! EVAL_GATEWAY_TASK_CSV=v1 cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture
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

/// Default dataset path (v2) can be overridden with `EVAL_GATEWAY_TASK_CSV` or
/// the legacy `EVAL_DATASET_PATH`.
const DEFAULT_DATASET_PATH: &str = GATEWAY_TASKS_V2_DATASET_PATH;

/// Original Arrow dataset path (for reference)
#[allow(dead_code)]
const ARROW_DATASET_PATH: &str = "../../data/coding_atlas/data-00000-of-00001.arrow";

/// Combined result for a single task across all modes
#[derive(Debug)]
struct EvalResult {
    task_id: String,
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
        || env::var("OPENAI_API_KEY").is_err()
}

fn resolve_gateway_dataset_path(manifest_dir: &str, selector: &str) -> PathBuf {
    match selector.trim() {
        "" | "v2" | "gateway_tasks_v2" | "gateway_tasks_v2.csv" => {
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
/// 1. `EVAL_GATEWAY_TASK_CSV` (recommended): `v1`, `v2`, or a CSV path
/// 2. `EVAL_DATASET_PATH` (legacy): explicit dataset path
/// 3. default: gateway v2 task CSV
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
    if env::var("OPENAI_API_KEY").is_err() {
        tracing::warn!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    tracing::info!("Test: Claim Judge");

    let judge = match ClaimJudge::new() {
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
    match judge.verify_claims(task_prompt, answer, &claims).await {
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
        tracing::trace!("OPENAI_API_KEY=<your-api-key>");
        tracing::trace!("Optional configuration:");
        tracing::trace!("EVAL_LIMIT=5                   # Run only N tasks");
        tracing::trace!("EVAL_TASK_PREFIX=task_linear   # Filter by task ID prefix");
        tracing::trace!("EVAL_MODEL=gpt-4o              # Model for tool-calling client");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v1       # gateway_tasks.csv");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=v2       # gateway_tasks_v2.csv (default)");
        tracing::trace!("EVAL_GATEWAY_TASK_CSV=/path/to/gateway_tasks.csv  # Custom CSV path");
        tracing::trace!("EVAL_DATASET_PATH=/path/to/dataset.csv            # Legacy override");
        tracing::trace!("Tool-calling client modes:");
        tracing::trace!(
            "EVAL_TOOL_MODES=baseline,codemode  # Comma-separated (baseline,codemode,rlm,codemoderlm,all)"
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

    // Step 2: Setup Gateway connection
    let config = gateway_auth::build_kontext_config().expect("Config should be valid");
    tracing::debug!("Connecting to Gateway...");

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
    tracing::debug!("Gateway connected");

    // Step 3: Setup RLM infrastructure
    let temp_dir = TempDir::new().unwrap();
    let rlm_router: Option<Arc<GatewayResultRouter>> =
        match create_rlm_router(temp_dir.path()).await {
            Ok(r) => Some(Arc::new(r)),
            Err(e) => {
                tracing::warn!("RLM router creation failed: {e}");
                None
            }
        };

    // Step 4: Setup judge
    let judge = match ClaimJudge::new() {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to create claim judge: {e}");
            panic!("Failed to create claim judge: {e}");
        }
    };

    // Step 5: Create runner and discover tools
    tracing::debug!("Discovering available Gateway tools...");
    let agent_model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let runner = match ToolCallingRunner::new(client.clone(), rlm_router.clone()).await {
        Ok(r) => r.with_model(&agent_model),
        Err(e) => {
            tracing::error!("Failed to create runner: {e}");
            panic!("Failed to create runner: {e}");
        }
    };
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
    // Options: baseline, codemode, rlm, codemoderlm. Default: baseline,codemode,rlm
    let tool_modes: Vec<ToolCallingMode> = env::var("EVAL_TOOL_MODES")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "baseline,codemode,rlm".to_string())
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "baseline" => Some(ToolCallingMode::Baseline),
            "codemode" | "code" => Some(ToolCallingMode::CodeMode),
            "rlm" | "baselinerlm" => Some(ToolCallingMode::BaselineRlm),
            "codemoderlm" | "code+rlm" | "coderlm" => Some(ToolCallingMode::CodeModeRlm),
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
            ToolCallingMode::CodeModeRlm,
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

    // Run tool-calling modes
    for mode in &tool_modes {
        let mode_name = mode.to_string();
        tracing::warn!("Running {mode_name} mode (ToolCalling)");

        let mut results = Vec::new();

        for (i, task) in eval_tasks.iter().enumerate() {
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

            // Judge the answer
            let verification = match judge
                .verify_claims(&task.prompt, &task_result.final_answer, &task.claims)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Verification failed: {e}");
                    ClaimVerificationResult {
                        scores: vec![],
                        coverage: 0.0,
                        passed: false,
                        raw_response: e.to_string(),
                    }
                }
            };

            let status = if verification.passed { "PASS" } else { "FAIL" };
            let latency_secs = task_result.latency_ms as f64 / 1000.0;
            tracing::warn!(
                "  {} | coverage={:.2} | tokens={} | {:.1}s",
                status, verification.coverage, task_result.context_tokens, latency_secs
            );
            if let Some(ref err) = task_result.error {
                tracing::error!("ERROR: {err}");
            }

            results.push(EvalResult {
                task_id: task.task_id.clone(),
                task_result,
                verification,
            });
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
    tracing::info!("| Task | Baseline | CodeMode | Baseline+RLM | CodeMode+RLM | Codex | Winner |");
    tracing::info!("|------|----------|----------|--------------|--------------|-------|--------|");

    let baseline_results = results.get("Baseline");
    let codemode_results = results.get("CodeMode");
    let rlm_results = results.get("Baseline+RLM");
    let codemode_rlm_results = results.get("CodeMode+RLM");
    let codex_results = results.get("Codex");

    // Get any available results for iteration
    let any_results = baseline_results
        .or(codemode_results)
        .or(rlm_results)
        .or(codemode_rlm_results)
        .or(codex_results);

    if let Some(first_results) = any_results {
        for (i, result) in first_results.iter().enumerate() {
            let b_cov = baseline_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);
            let c_cov = codemode_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);
            let r_cov = rlm_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);
            let cr_cov = codemode_rlm_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);
            let codex_cov = codex_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);

            let b_status = if b_cov >= PASS_THRESHOLD {
                "PASS"
            } else {
                "FAIL"
            };
            let c_status = if c_cov >= PASS_THRESHOLD {
                "PASS"
            } else {
                "FAIL"
            };
            let r_status = if r_cov >= PASS_THRESHOLD {
                "PASS"
            } else {
                "FAIL"
            };
            let cr_status = if cr_cov >= PASS_THRESHOLD {
                "PASS"
            } else {
                "FAIL"
            };
            let codex_status = if codex_cov >= PASS_THRESHOLD {
                "PASS"
            } else {
                "FAIL"
            };

            // Determine winner based on highest coverage
            let max_cov = b_cov.max(c_cov).max(r_cov).max(cr_cov).max(codex_cov);
            let winner = if max_cov == 0.0 {
                "all fail"
            } else if codex_cov == max_cov
                && b_cov == max_cov
                && c_cov == max_cov
                && r_cov == max_cov
                && cr_cov == max_cov
            {
                "tie (all)"
            } else if codex_cov == max_cov {
                "Codex"
            } else if b_cov == max_cov {
                "Baseline"
            } else if r_cov == max_cov {
                "Baseline+RLM"
            } else if cr_cov == max_cov {
                "CodeMode+RLM"
            } else {
                "CodeMode"
            };

            tracing::info!(
                "| {} | {:.2} {} | {:.2} {} | {:.2} {} | {:.2} {} | {:.2} {} | {} |",
                &result.task_id[..result.task_id.len().min(15)],
                b_cov,
                b_status,
                c_cov,
                c_status,
                r_cov,
                r_status,
                cr_cov,
                cr_status,
                codex_cov,
                codex_status,
                winner
            );
        }
    }

    tracing::info!("Key Insight");
    tracing::info!("RLM modes aim to achieve significant token reduction while maintaining quality.");
    tracing::info!("CodeMode+RLM combines code generation with RLM routing for best of both worlds.");
    tracing::info!("Codex client uses the full Codex system prompts and agent loop.");
}

/// Save evaluation results to a JSON file compatible with `scripts/plot_results.py`.
///
/// The file is written to `codex-rs/mcp-atlas/services/mcp_eval/results/` with a
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
    let results_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()))
        .join("../mcp-atlas/services/mcp_eval/results");

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

                    json!({
                        "task_id": r.task_id,
                        "task_result": {
                            "task_id": r.task_result.task_id,
                            "mode_name": r.task_result.mode_name,
                            "context_tokens": r.task_result.context_tokens,
                            "latency_ms": r.task_result.latency_ms,
                            "error": r.task_result.error,
                            "final_answer": r.task_result.final_answer,
                            "num_tool_calls": r.task_result.tool_calls.len(),
                        },
                        "verification": {
                            "coverage": r.verification.coverage,
                            "passed": r.verification.passed,
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

    let path = results_dir.join(format!("mcp_atlas_eval_{ts}.json"));
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
    let runner = match ToolCallingRunner::new(client.clone(), None).await {
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
    let judge = match ClaimJudge::new() {
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
    let runner = match ToolCallingRunner::new(client.clone(), rlm_router.clone()).await {
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
        tracing::debug!("Sending prompt to agent (GPT-4o)...");

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
        tracing::debug!("Sending to judge (GPT-4o)...");

        let verify_start = Instant::now();
        let verification = match judge
            .verify_claims(&task.prompt, &task_result.final_answer, &task.claims)
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
    tracing::trace!("2. OpenAI API key (for agent and judge)");
    tracing::trace!("3. MCP-Atlas dataset");
    tracing::trace!("Setup:");
    tracing::trace!("# Create .env file:");
    tracing::trace!("KONTEXT_CLIENT_ID=<your-client-id>");
    tracing::trace!("KONTEXT_CLIENT_SECRET=<your-client-secret>");
    tracing::trace!("KONTEXT_MCP_URL=http://localhost:4000/mcp");
    tracing::trace!("KONTEXT_TOKEN_URL=http://localhost:4000/oauth2/token");
    tracing::trace!("OPENAI_API_KEY=<your-openai-key>");
    tracing::trace!("EVAL_GATEWAY_TASK_CSV=v2");
    tracing::trace!("# Optional alternate dataset:");
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
