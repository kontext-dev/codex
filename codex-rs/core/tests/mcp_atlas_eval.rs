//! MCP-Atlas Evaluation Script
//!
//! Benchmarks LLM agent performance on the MCP-Atlas dataset across three execution modes:
//! - Baseline: Direct EXECUTE_TOOL calls with full results in context
//! - CodeMode: EXECUTE_CODE with summarized results
//! - Baseline+RLM: EXECUTE_TOOL with RLM routing for large results
//!
//! ## Prerequisites
//!
//! ```bash
//! # Gateway credentials
//! export KONTEXT_CLIENT_ID=<your-client-id>
//! export KONTEXT_CLIENT_SECRET=<your-client-secret>
//! export KONTEXT_GATEWAY_URL=https://gateway.kontext.dev
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
//! # Full evaluation (500 tasks × 3 modes = 1500 runs)
//! cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use codex_core::eval::ClaimJudge;
use codex_core::eval::ClaimScore;
use codex_core::eval::ClaimVerificationResult;
use codex_core::eval::ExecutionMode;
use codex_core::eval::TaskResult;
use codex_core::eval::TaskRunner;
use codex_core::eval::PASS_THRESHOLD;
use codex_core::rlm::EvidenceStore;
use codex_core::rlm::GatewayResultRouter;
use codex_core::rlm::RlmConfig;
use codex_core::rlm::RlmCorpus;
use codex_rmcp_client::ElicitationAction;
use codex_rmcp_client::ElicitationResponse;
use codex_rmcp_client::OAuthCredentialsStoreMode;
use codex_rmcp_client::RmcpClient;
use futures::FutureExt;
use kontext_dev::build_mcp_url;
use kontext_dev::request_access_token;
use kontext_dev::KontextDevConfig;
use kontext_dev::DEFAULT_SCOPE;
use kontext_dev::DEFAULT_SERVER_NAME;
use mcp_types::ClientCapabilities;
use mcp_types::Implementation;
use mcp_types::InitializeRequestParams;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Dataset path relative to core directory (CARGO_MANIFEST_DIR)
/// Use the v2 dataset focused on Linear, DeepWiki, Context7 tools (no CLI security issues)
const DATASET_PATH: &str = "../mcp-atlas/services/mcp_eval/gateway_tasks_v2.csv";

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
        || env::var("KONTEXT_CLIENT_ID").is_err()
        || env::var("KONTEXT_CLIENT_SECRET").is_err()
        || env::var("OPENAI_API_KEY").is_err()
}

/// Build KontextDevConfig from environment variables
fn build_kontext_config() -> Option<KontextDevConfig> {
    let client_id = env::var("KONTEXT_CLIENT_ID").ok()?;
    let client_secret = env::var("KONTEXT_CLIENT_SECRET").ok()?;

    let (mcp_url, token_url) = if let (Ok(mcp), Ok(token)) = (
        env::var("KONTEXT_MCP_URL"),
        env::var("KONTEXT_TOKEN_URL"),
    ) {
        (mcp, token)
    } else if let Ok(base_url) = env::var("KONTEXT_GATEWAY_URL") {
        let base = base_url.trim_end_matches('/');
        if base.ends_with("/mcp") {
            let prefix = &base[..base.len() - 4];
            (base.to_string(), format!("{}/oauth2/token", prefix))
        } else {
            (format!("{}/mcp", base), format!("{}/oauth2/token", base))
        }
    } else {
        (
            "https://gateway.kontext.dev/mcp".to_string(),
            "https://gateway.kontext.dev/oauth2/token".to_string(),
        )
    };

    Some(KontextDevConfig {
        client_id,
        client_secret,
        mcp_url,
        token_url,
        scope: DEFAULT_SCOPE.to_string(),
        server_name: DEFAULT_SERVER_NAME.to_string(),
    })
}

/// Create MCP initialization parameters
fn create_init_params() -> InitializeRequestParams {
    InitializeRequestParams {
        capabilities: ClientCapabilities {
            experimental: None,
            roots: None,
            sampling: None,
            elicitation: Some(json!({})),
        },
        client_info: Implementation {
            name: "mcp-atlas-eval".into(),
            version: "1.0.0".into(),
            title: Some("MCP-Atlas Evaluation".into()),
            user_agent: None,
        },
        protocol_version: mcp_types::MCP_SCHEMA_VERSION.to_string(),
    }
}

/// Create elicitation handler
fn create_elicitation_handler() -> codex_rmcp_client::SendElicitation {
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

/// Get dataset path
fn get_dataset_path() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join(DATASET_PATH)
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
    println!("\n# Test: Dataset Loading\n");

    let dataset_path = get_dataset_path();
    println!("  Loading from: {:?}", dataset_path);

    if !dataset_path.exists() {
        println!("  Dataset not found at {:?}", dataset_path);
        println!("  Please ensure the MCP-Atlas dataset is downloaded.");
        return;
    }

    match codex_core::eval::load_dataset(&dataset_path) {
        Ok(tasks) => {
            println!("  Loaded {} tasks", tasks.len());
            assert!(!tasks.is_empty(), "Dataset should not be empty");

            // Print first task as sample
            if let Some(task) = tasks.first() {
                println!("\n  Sample task:");
                println!("    ID: {}", task.task_id);
                println!("    Tools: {:?}", &task.enabled_tools[..task.enabled_tools.len().min(5)]);
                println!("    Prompt: {}...", &task.prompt[..task.prompt.len().min(100)]);
                println!("    Claims: {} total", task.claims.len());
            }
        }
        Err(e) => {
            println!("  Failed to load dataset: {}", e);
            panic!("Dataset loading failed");
        }
    }
}

/// Test 2: Verify Gateway connection works
#[tokio::test]
async fn test_gateway_connection() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n# Test: Gateway Connection\n");

    let config = build_kontext_config().expect("Config should be valid");
    println!("  Token URL: {}", config.token_url);

    // Authenticate
    let token = match request_access_token(&config).await {
        Ok(t) => {
            println!("  Authentication successful");
            t
        }
        Err(e) => {
            println!("  Authentication failed: {}", e);
            println!("  Gateway may not be running");
            return;
        }
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
        Err(e) => {
            println!("  Failed to create MCP client: {}", e);
            return;
        }
    };

    // Initialize
    match client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
    {
        Ok(result) => {
            println!(
                "  MCP initialized: {} v{}",
                result.server_info.name, result.server_info.version
            );
        }
        Err(e) => {
            println!("  MCP initialization failed: {}", e);
            return;
        }
    }

    // List tools
    match client.list_tools(None, Some(Duration::from_secs(30))).await {
        Ok(result) => {
            println!("  Found {} tools", result.tools.len());
            // Print EXECUTE_TOOL schema to understand expected args
            for tool in &result.tools {
                println!("\n  Tool: {}", tool.name);
                println!("  Description: {}", tool.description.as_deref().unwrap_or("N/A"));
                println!("  Schema: {}", serde_json::to_string_pretty(&tool.input_schema).unwrap_or("N/A".to_string()));
            }
            assert!(!result.tools.is_empty(), "Should have tools available");
        }
        Err(e) => {
            println!("  Failed to list tools: {}", e);
        }
    }
}

/// Test 3: Verify real Git MCP tool works
#[tokio::test]
async fn test_git_mcp_tool() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n# Test: Git MCP Tool\n");

    let config = build_kontext_config().expect("Config should be valid");
    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("  Gateway not reachable: {}", e);
            return;
        }
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
        Err(e) => {
            println!("  Failed to create MCP client: {}", e);
            return;
        }
    };

    if client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
        .is_err()
    {
        println!("  MCP initialization failed");
        return;
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
            println!("  SEARCH_TOOLS(git) succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            println!("  Response preview: {}...", &content[..content.len().min(500)]);
        }
        Err(e) => {
            println!("  SEARCH_TOOLS failed: {}", e);
        }
    }
}

/// Test 4: Verify Code Executor MCP tool works
#[tokio::test]
async fn test_code_executor_mcp_tool() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n# Test: Code Executor MCP Tool\n");

    let config = build_kontext_config().expect("Config should be valid");
    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("  Gateway not reachable: {}", e);
            return;
        }
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
        Err(e) => {
            println!("  Failed to create MCP client: {}", e);
            return;
        }
    };

    if client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
        .is_err()
    {
        println!("  MCP initialization failed");
        return;
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
            println!("  EXECUTE_CODE succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            println!("  Response: {}", content);
        }
        Err(e) => {
            println!("  EXECUTE_CODE failed: {}", e);
            println!("  (This is expected if EXECUTE_CODE is not available)");
        }
    }
}

/// Test 5: Verify CLI MCP tool works
#[tokio::test]
async fn test_cli_mcp_tool() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n# Test: CLI MCP Tool\n");

    let config = build_kontext_config().expect("Config should be valid");
    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("  Gateway not reachable: {}", e);
            return;
        }
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
        Err(e) => {
            println!("  Failed to create MCP client: {}", e);
            return;
        }
    };

    if client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
        .is_err()
    {
        println!("  MCP initialization failed");
        return;
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
            println!("  SEARCH_TOOLS(cli) succeeded in {:?}", start.elapsed());
            let content = serde_json::to_string_pretty(&r).unwrap_or_default();
            println!("  Response preview: {}...", &content[..content.len().min(500)]);
        }
        Err(e) => {
            println!("  SEARCH_TOOLS failed: {}", e);
        }
    }
}

/// Test 6: Verify Claim Judge works
#[tokio::test]
async fn test_claim_judge() {
    if env::var("OPENAI_API_KEY").is_err() {
        println!("\n⏭️  Skipping: OPENAI_API_KEY not set\n");
        return;
    }

    println!("\n# Test: Claim Judge\n");

    let judge = match ClaimJudge::new() {
        Ok(j) => j,
        Err(e) => {
            println!("  Failed to create judge: {}", e);
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
            println!("  Verification completed in {:?}", start.elapsed());
            println!("  Coverage: {:.2}", result.coverage);
            println!("  Passed: {}", result.passed);
            println!("\n  Per-claim results:");
            for (claim, score) in &result.scores {
                let score_str = match score {
                    ClaimScore::Fulfilled => "FULFILLED",
                    ClaimScore::PartiallyFulfilled => "PARTIAL",
                    ClaimScore::NotFulfilled => "NOT_FULFILLED",
                };
                println!("    - {}: {}", score_str, &claim[..claim.len().min(50)]);
            }
        }
        Err(e) => {
            println!("  Verification failed: {}", e);
        }
    }
}

/// Test 7: Verify RLM routing works
#[tokio::test]
async fn test_rlm_routing() {
    println!("\n# Test: RLM Routing\n");

    let temp_dir = TempDir::new().unwrap();
    let router = match create_rlm_router(temp_dir.path()).await {
        Ok(r) => r,
        Err(e) => {
            println!("  Failed to create RLM router: {}", e);
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
            println!("  Small response ({} bytes): {}", small_content.len(), routing);
            assert!(matches!(result, codex_core::rlm::ProcessedResult::PassThrough { .. }));
        }
        Err(e) => {
            println!("  Small response routing failed: {}", e);
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
            println!("  Large response ({} bytes): {}", large_content.len(), routing);
            assert!(matches!(result, codex_core::rlm::ProcessedResult::StoredInCorpus { .. }));
        }
        Err(e) => {
            println!("  Large response routing failed: {}", e);
        }
    }
}

// =============================================================================
// MAIN EVALUATION
// =============================================================================

/// Run three-way evaluation on first 10 tasks (test mode)
#[tokio::test]
async fn run_mcp_atlas_three_way_evaluation() {
    if should_skip() {
        println!("\n");
        println!("═══════════════════════════════════════════════════════════════");
        println!("       MCP-Atlas Evaluation - Setup Required");
        println!("═══════════════════════════════════════════════════════════════");
        println!();
        println!("Required environment variables:");
        println!("  KONTEXT_CLIENT_ID=<your-client-id>");
        println!("  KONTEXT_CLIENT_SECRET=<your-client-secret>");
        println!("  KONTEXT_GATEWAY_URL=https://gateway.kontext.dev");
        println!("  OPENAI_API_KEY=<your-api-key>");
        println!();
        println!("Optional filter variables:");
        println!("  EVAL_LIMIT=5              # Run only N tasks");
        println!("  EVAL_TASK_PREFIX=task_linear  # Filter by task ID prefix");
        println!("  EVAL_MODES=codemode       # Run specific modes (baseline,codemode,rlm)");
        println!("  EVAL_MODEL=gpt-4.1-2025-04-14  # Model to use (default: gpt-4o)");
        println!();
        println!("Run with:");
        println!("  source .env");
        println!("  cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture");
        println!();
        println!("Examples:");
        println!("  # Run 5 Linear tasks in RLM mode with gpt-4.1:");
        println!("  EVAL_LIMIT=5 EVAL_MODES=rlm EVAL_MODEL=gpt-4.1-2025-04-14 cargo test ...");
        println!();
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("       MCP-Atlas Three-Way Evaluation");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Step 1: Load FULL dataset (filter comes later)
    let dataset_path = get_dataset_path();
    println!("Loading dataset from {:?}...", dataset_path);

    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to load dataset: {}", e);
            return;
        }
    };

    println!("Loaded {} tasks from dataset\n", tasks.len());

    // Step 2: Setup Gateway connection
    let config = build_kontext_config().expect("Config should be valid");
    println!("Connecting to Gateway...");

    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("Gateway authentication failed: {}", e);
            return;
        }
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
        Err(e) => {
            println!("Failed to create MCP client: {}", e);
            return;
        }
    };

    if let Err(e) = client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
    {
        println!("MCP initialization failed: {}", e);
        return;
    }
    println!("Gateway connected\n");

    // Step 3: Setup RLM infrastructure
    let temp_dir = TempDir::new().unwrap();
    let rlm_router: Option<Arc<GatewayResultRouter>> = match create_rlm_router(temp_dir.path()).await {
        Ok(r) => Some(Arc::new(r)),
        Err(e) => {
            println!("Warning: RLM router creation failed: {}", e);
            None
        }
    };

    // Step 4: Setup judge
    let judge = match ClaimJudge::new() {
        Ok(j) => j,
        Err(e) => {
            println!("Failed to create claim judge: {}", e);
            return;
        }
    };

    // Step 5: Create runner and discover tools
    println!("Discovering available Gateway tools...");
    let agent_model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let runner = match TaskRunner::new(client.clone(), rlm_router.clone()).await {
        Ok(r) => r.with_model(&agent_model),
        Err(e) => {
            println!("Failed to create runner: {}", e);
            return;
        }
    };
    println!("Using model: {}", agent_model);

    // Print discovered tools
    let tools = runner.available_tools();
    println!("Found {} tools:\n", tools.len());
    let mut by_server: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for tool in tools {
        by_server.entry(&tool.server).or_default().push(&tool.name);
    }
    for (server, tool_names) in &by_server {
        println!("  {}: {}", server, tool_names.join(", "));
    }
    println!();

    // Step 6: Use all tasks from the gateway dataset (already filtered to available tools)
    // Filter out tasks with 0 coverage (tools that don't match)
    let eval_tasks: Vec<_> = tasks.iter()
        .filter(|task| {
            if task.enabled_tools.is_empty() {
                return false;
            }
            // At least one tool must resolve
            task.enabled_tools.iter().any(|tool_name| runner.resolve_tool(tool_name).is_some())
        })
        .collect();

    println!("Tasks with resolvable tools: {}/{}\n", eval_tasks.len(), tasks.len());

    if eval_tasks.is_empty() {
        println!("No tasks meet the minimum coverage threshold. Exiting.");
        return;
    }

    // Show the solvable tasks with their coverage
    println!("Solvable tasks:\n");
    for task in eval_tasks.iter().take(20) {
        let coverage = runner.get_tool_coverage(task);
        let matches = runner.get_matching_tools(task);
        println!("═══ Task: {} ═══", &task.task_id[..task.task_id.len().min(15)]);
        println!("Coverage: {:.0}% ({} tools matched)", coverage * 100.0, matches.len());

        // Show prompt
        let prompt_preview = if task.prompt.len() > 200 {
            format!("{}...", &task.prompt[..200].replace('\n', " "))
        } else {
            task.prompt.replace('\n', " ")
        };
        println!("Prompt: {}", prompt_preview);

        // Show claims
        println!("Claims ({}):", task.claims.len());
        for (i, claim) in task.claims.iter().enumerate().take(3) {
            println!("  {}. {}", i + 1, &claim[..claim.len().min(80)]);
        }
        if task.claims.len() > 3 {
            println!("  ... and {} more claims", task.claims.len() - 3);
        }

        // Show tool mapping
        println!("Tools:");
        for (dataset_tool, gateway_tool) in matches.iter().take(3) {
            println!("  {} → {} ({})", dataset_tool, gateway_tool.name, gateway_tool.server);
        }
        if matches.len() > 3 {
            println!("  ... and {} more tools", matches.len() - 3);
        }

        // Show missing tools
        let missing: Vec<_> = task.enabled_tools.iter()
            .filter(|t| runner.resolve_tool(t).is_none())
            .collect();
        if !missing.is_empty() {
            println!("Missing tools: {:?}", missing);
        }
        println!();
    }
    println!();

    // Apply subset filters from environment variables
    // EVAL_LIMIT: max number of tasks (default: all)
    // EVAL_TASK_PREFIX: filter to tasks starting with this prefix (e.g., "task_linear")
    // EVAL_MODES: comma-separated modes to run (e.g., "baseline,codemode" or "rlm")
    let limit: usize = std::env::var("EVAL_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let task_prefix = std::env::var("EVAL_TASK_PREFIX").ok();

    let eval_tasks: Vec<_> = eval_tasks
        .into_iter()
        .filter(|task| {
            if let Some(ref prefix) = task_prefix {
                task.task_id.starts_with(prefix)
            } else {
                true
            }
        })
        .take(limit)
        .collect();

    if eval_tasks.is_empty() {
        println!("No tasks match the filter criteria.");
        if let Some(ref prefix) = task_prefix {
            println!("  EVAL_TASK_PREFIX={}", prefix);
        }
        return;
    }

    println!("Running evaluation on {} tasks", eval_tasks.len());
    if let Some(ref prefix) = task_prefix {
        println!("  (filtered by prefix: {})", prefix);
    }
    if limit < usize::MAX {
        println!("  (limited to {} tasks)", limit);
    }
    println!();

    // Step 7: Run evaluation with selected modes
    // EVAL_MODES: comma-separated modes (baseline, codemode, rlm). Default: all
    let mode_filter = std::env::var("EVAL_MODES").ok();
    let modes: Vec<ExecutionMode> = if let Some(ref filter) = mode_filter {
        let filter_lower = filter.to_lowercase();
        let mut selected = Vec::new();
        // Parse comma-separated modes explicitly
        for part in filter_lower.split(',') {
            let part = part.trim();
            match part {
                "baseline" => selected.push(ExecutionMode::Baseline),
                "codemode" | "code" => selected.push(ExecutionMode::CodeMode),
                "rlm" | "baselinerlm" => selected.push(ExecutionMode::BaselineRlm),
                _ => {}
            }
        }
        if selected.is_empty() {
            println!("Warning: No valid modes in EVAL_MODES='{}'. Running all modes.", filter);
            vec![ExecutionMode::Baseline, ExecutionMode::CodeMode, ExecutionMode::BaselineRlm]
        } else {
            println!("Running modes: {:?}\n", selected.iter().map(|m| m.to_string()).collect::<Vec<_>>());
            selected
        }
    } else {
        vec![ExecutionMode::Baseline, ExecutionMode::CodeMode, ExecutionMode::BaselineRlm]
    };

    let mut all_results: HashMap<ExecutionMode, Vec<EvalResult>> = HashMap::new();

    for mode in &modes {
        println!("\n═══ Running {} mode ═══\n", mode);

        let mut results = Vec::new();

        for (i, task) in eval_tasks.iter().enumerate() {
            println!("  Task {}/{}: {}...", i + 1, eval_tasks.len(), &task.task_id[..task.task_id.len().min(20)]);

            // Execute task
            let task_result = runner.run_task(task, *mode).await;

            // Print tool calls made
            if !task_result.tool_calls.is_empty() {
                let tool_names: Vec<_> = task_result.tool_calls.iter().map(|t| t.name.as_str()).collect();
                println!("    Tools used: {}", tool_names.join(", "));
            }

            // Judge the answer
            let verification = match judge
                .verify_claims(&task.prompt, &task_result.final_answer, &task.claims)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    println!("    Verification failed: {}", e);
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
            println!(
                "    Coverage: {:.2}, Status: {}, Tokens: {}, Latency: {:.1}s",
                verification.coverage, status, task_result.context_tokens, latency_secs
            );

            results.push(EvalResult {
                task_id: task.task_id.clone(),
                task_result,
                verification,
            });
        }

        all_results.insert(*mode, results);
    }

    // Step 8: Print results
    print_three_way_comparison(&all_results);
}

/// Print comparative results
fn print_three_way_comparison(results: &HashMap<ExecutionMode, Vec<EvalResult>>) {
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                    MCP-Atlas Three-Way Evaluation Results");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Per-mode summary
    for mode in &[ExecutionMode::Baseline, ExecutionMode::CodeMode, ExecutionMode::BaselineRlm] {
        if let Some(mode_results) = results.get(mode) {
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

            println!("## {} Mode", mode);
            println!();
            println!("| Metric | Value |");
            println!("|--------|-------|");
            println!(
                "| Pass Rate | {:.1}% ({}/{}) |",
                pass_count as f64 / mode_results.len().max(1) as f64 * 100.0,
                pass_count,
                mode_results.len()
            );
            println!("| Avg Coverage | {:.3} |", avg_coverage);
            println!("| Avg Context Tokens | {} |", avg_tokens);
            println!("| Avg Latency | {:.1}s |", avg_latency_secs);
            println!();
        }
    }

    // Comparative table
    println!("## Comparison\n");
    println!("| Mode | Pass Rate | Avg Coverage | Avg Tokens | Token Reduction |");
    println!("|------|-----------|--------------|------------|-----------------|");

    let baseline_tokens = results
        .get(&ExecutionMode::Baseline)
        .map(|r| {
            r.iter().map(|e| e.task_result.context_tokens).sum::<i64>() / r.len().max(1) as i64
        })
        .unwrap_or(1);

    for mode in &[ExecutionMode::Baseline, ExecutionMode::CodeMode, ExecutionMode::BaselineRlm] {
        if let Some(mode_results) = results.get(mode) {
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

            let reduction = if baseline_tokens > 0 {
                100.0 - (avg_tokens as f64 / baseline_tokens as f64 * 100.0)
            } else {
                0.0
            };

            let reduction_str = if *mode == ExecutionMode::Baseline {
                "-".to_string()
            } else {
                format!("-{:.0}%", reduction)
            };

            println!(
                "| {} | {:.1}% | {:.3} | {} | {} |",
                mode, pass_rate, avg_coverage, avg_tokens, reduction_str
            );
        }
    }

    println!();

    // Per-task comparison
    println!("## Per-Task Comparison\n");
    println!("| Task | Baseline | CodeMode | RLM | Winner |");
    println!("|------|----------|----------|-----|--------|");

    let baseline_results = results.get(&ExecutionMode::Baseline);
    let codemode_results = results.get(&ExecutionMode::CodeMode);
    let rlm_results = results.get(&ExecutionMode::BaselineRlm);

    if let Some(baseline) = baseline_results {
        for (i, b_result) in baseline.iter().enumerate() {
            let b_cov = b_result.verification.coverage;
            let c_cov = codemode_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);
            let r_cov = rlm_results
                .and_then(|r| r.get(i))
                .map(|r| r.verification.coverage)
                .unwrap_or(0.0);

            let b_status = if b_cov >= PASS_THRESHOLD { "PASS" } else { "FAIL" };
            let c_status = if c_cov >= PASS_THRESHOLD { "PASS" } else { "FAIL" };
            let r_status = if r_cov >= PASS_THRESHOLD { "PASS" } else { "FAIL" };

            let winner = if b_cov >= c_cov && b_cov >= r_cov && b_cov > 0.0 {
                if b_cov == r_cov {
                    "Baseline/RLM"
                } else {
                    "Baseline"
                }
            } else if r_cov >= b_cov && r_cov >= c_cov && r_cov > 0.0 {
                "RLM"
            } else if c_cov > 0.0 {
                "CodeMode"
            } else {
                "tie (all fail)"
            };

            println!(
                "| {} | {:.2} {} | {:.2} {} | {:.2} {} | {} |",
                &b_result.task_id[..b_result.task_id.len().min(15)],
                b_cov,
                b_status,
                c_cov,
                c_status,
                r_cov,
                r_status,
                winner
            );
        }
    }

    println!();
    println!("## Key Insight\n");
    println!("RLM aims to achieve significant token reduction while maintaining baseline quality.");
    println!("(vs CodeMode which may lose quality for token reduction)");
}

/// Full evaluation on all 500 tasks (ignored by default - run with --ignored)
#[tokio::test]
#[ignore]
async fn run_full_mcp_atlas_evaluation() {
    if should_skip() {
        println!("Skipping full evaluation - credentials not set");
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("       MCP-Atlas FULL Evaluation (500 tasks × 3 modes = 1500 runs)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Same as above but without the .take(10)
    // This would run all 500 tasks
    println!("Full evaluation would run here...");
    println!("This is a placeholder - implement the same logic as test mode but with all tasks.");
}

/// Analyze dataset to find tasks solvable with available Gateway tools
#[tokio::test]
async fn analyze_solvable_tasks() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("       MCP-Atlas Dataset Analysis: Which Tasks Can Be Solved?");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Load dataset
    let dataset_path = get_dataset_path();
    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to load dataset: {}", e);
            return;
        }
    };
    println!("Loaded {} tasks from dataset\n", tasks.len());

    // Connect to Gateway and discover tools
    let config = build_kontext_config().expect("Config should be valid");
    let token = match request_access_token(&config).await {
        Ok(t) => t,
        Err(e) => {
            println!("Gateway authentication failed: {}", e);
            return;
        }
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
        Err(e) => {
            println!("Failed to create MCP client: {}", e);
            return;
        }
    };

    if let Err(e) = client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
    {
        println!("MCP initialization failed: {}", e);
        return;
    }

    // Create runner to discover tools
    let runner = match TaskRunner::new(client.clone(), None).await {
        Ok(r) => r,
        Err(e) => {
            println!("Failed to create runner: {}", e);
            return;
        }
    };

    // Print available Gateway tools
    let gateway_tools = runner.available_tools();
    println!("Available Gateway tools ({} total):\n", gateway_tools.len());

    let mut by_server: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for tool in gateway_tools {
        by_server.entry(&tool.server).or_default().push(&tool.name);
    }
    for (server, tool_names) in &by_server {
        println!("  {}: {}", server, tool_names.join(", "));
    }
    println!();

    // Analyze each task
    println!("═══ Task Analysis ═══\n");

    // Target servers
    let target_servers = ["git", "CLI", "Code Executor"];

    // Categorize tasks
    let mut fully_matched: Vec<(&codex_core::eval::McpAtlasTask, f64)> = Vec::new();
    let mut partially_matched: Vec<(&codex_core::eval::McpAtlasTask, f64, Vec<String>)> = Vec::new();
    let mut no_match: Vec<&codex_core::eval::McpAtlasTask> = Vec::new();

    for task in &tasks {
        let matches = runner.get_matching_tools(task);
        let match_ratio = if task.enabled_tools.is_empty() {
            0.0
        } else {
            matches.len() as f64 / task.enabled_tools.len() as f64
        };

        // Check if matches are from target servers
        let target_matches: Vec<_> = matches.iter()
            .filter(|(_, tool)| target_servers.iter().any(|s| tool.server.to_lowercase().contains(&s.to_lowercase())))
            .collect();

        let target_ratio = if task.enabled_tools.is_empty() {
            0.0
        } else {
            target_matches.len() as f64 / task.enabled_tools.len() as f64
        };

        if target_ratio >= 0.8 {
            fully_matched.push((task, target_ratio));
        } else if target_ratio > 0.0 {
            let unmatched: Vec<String> = task.enabled_tools.iter()
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
    println!("## Summary\n");
    println!("| Category | Count | Percentage |");
    println!("|----------|-------|------------|");
    println!("| Fully Matched (≥80% tools available) | {} | {:.1}% |",
        fully_matched.len(),
        fully_matched.len() as f64 / tasks.len() as f64 * 100.0);
    println!("| Partially Matched (<80% but >0% tools) | {} | {:.1}% |",
        partially_matched.len(),
        partially_matched.len() as f64 / tasks.len() as f64 * 100.0);
    println!("| No Match (0% tools available) | {} | {:.1}% |",
        no_match.len(),
        no_match.len() as f64 / tasks.len() as f64 * 100.0);
    println!();

    // Show best candidates
    println!("## Best Candidate Tasks (≥80% tool coverage)\n");
    if fully_matched.is_empty() {
        println!("No tasks have ≥80% tool coverage with target servers.\n");
    } else {
        println!("| Task ID | Match % | Prompt Preview | Tools |");
        println!("|---------|---------|----------------|-------|");
        for (task, ratio) in fully_matched.iter().take(20) {
            let prompt_preview = if task.prompt.len() > 50 {
                format!("{}...", &task.prompt[..50].replace('\n', " "))
            } else {
                task.prompt.replace('\n', " ")
            };
            let tool_count = task.enabled_tools.len();
            println!("| {} | {:.0}% | {} | {} |",
                &task.task_id[..task.task_id.len().min(15)],
                ratio * 100.0,
                prompt_preview,
                tool_count);
        }
        println!();
    }

    // Show partially matched with missing tools
    println!("## Partially Matched Tasks (with missing tools)\n");
    if partially_matched.is_empty() {
        println!("No partially matched tasks.\n");
    } else {
        println!("| Task ID | Match % | Missing Tools |");
        println!("|---------|---------|---------------|");
        for (task, ratio, missing) in partially_matched.iter().take(10) {
            let missing_preview: String = missing.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
            let more = if missing.len() > 3 { format!(" (+{})", missing.len() - 3) } else { String::new() };
            println!("| {} | {:.0}% | {}{} |",
                &task.task_id[..task.task_id.len().min(15)],
                ratio * 100.0,
                missing_preview,
                more);
        }
        println!();
    }

    // Common missing tool patterns
    println!("## Common Missing Tools (blocking task solvability)\n");
    let mut missing_tool_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for task in &tasks {
        for tool in &task.enabled_tools {
            if runner.get_matching_tools(task).iter().all(|(dt, _)| dt != tool) {
                *missing_tool_counts.entry(tool.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut missing_sorted: Vec<_> = missing_tool_counts.into_iter().collect();
    missing_sorted.sort_by(|a, b| b.1.cmp(&a.1));

    println!("| Tool Name | Tasks Affected |");
    println!("|-----------|----------------|");
    for (tool, count) in missing_sorted.iter().take(15) {
        println!("| {} | {} |", tool, count);
    }
    println!();

    // Recommendations
    println!("## Recommendations\n");
    println!("1. **Best tasks for evaluation**: {} tasks have ≥80% tool coverage", fully_matched.len());
    if !fully_matched.is_empty() {
        println!("   - These are most likely to succeed with current Gateway tools");
    }
    println!();
    println!("2. **To increase coverage, add support for**:");
    for (tool, count) in missing_sorted.iter().take(5) {
        println!("   - `{}` (would unlock {} tasks)", tool, count);
    }
    println!();

    // Show sample fully matched task details
    if let Some((task, ratio)) = fully_matched.first() {
        println!("## Sample Fully Matched Task\n");
        println!("**Task ID**: {}\n", task.task_id);
        println!("**Match Ratio**: {:.0}%\n", ratio * 100.0);
        println!("**Prompt**:\n```\n{}\n```\n", task.prompt);
        println!("**Enabled Tools**: {:?}\n", task.enabled_tools);
        println!("**Claims to verify**:");
        for (i, claim) in task.claims.iter().enumerate().take(5) {
            println!("  {}. {}", i + 1, claim);
        }
        if task.claims.len() > 5 {
            println!("  ... ({} more claims)", task.claims.len() - 5);
        }
        println!();

        // Show tool mapping
        println!("**Tool Mapping**:");
        for (dt, gt) in runner.get_matching_tools(task) {
            println!("  {} → {} ({})", dt, gt.name, gt.server);
        }
    }
}

/// Verbose debug evaluation on 5 random tasks
/// This test logs EVERY step to help debug why tasks fail
#[tokio::test]
async fn run_verbose_debug_evaluation() {
    if should_skip() {
        println!("\n⏭️  Skipping: credentials not set\n");
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("       VERBOSE DEBUG EVALUATION - 5 Random Tasks");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Step 1: Load dataset
    let dataset_path = get_dataset_path();
    println!("[STEP 1] Loading dataset from {:?}...", dataset_path);

    let tasks = match codex_core::eval::load_dataset(&dataset_path) {
        Ok(t) => {
            println!("  ✓ Loaded {} tasks\n", t.len());
            t
        }
        Err(e) => {
            println!("  ✗ Failed to load dataset: {}\n", e);
            return;
        }
    };

    // Step 2: Connect to Gateway
    println!("[STEP 2] Connecting to Gateway...");
    let config = build_kontext_config().expect("Config should be valid");
    println!("  Token URL: {}", config.token_url);
    println!("  MCP URL: {}", config.mcp_url);

    let token = match request_access_token(&config).await {
        Ok(t) => {
            println!("  ✓ Authentication successful\n");
            t
        }
        Err(e) => {
            println!("  ✗ Authentication failed: {}\n", e);
            return;
        }
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
            println!("  ✓ MCP client created\n");
            Arc::new(c)
        }
        Err(e) => {
            println!("  ✗ Failed to create MCP client: {}\n", e);
            return;
        }
    };

    if let Err(e) = client
        .initialize(
            create_init_params(),
            Some(Duration::from_secs(30)),
            create_elicitation_handler(),
        )
        .await
    {
        println!("  ✗ MCP initialization failed: {}\n", e);
        return;
    }
    println!("  ✓ MCP initialized\n");

    // Step 3: Setup RLM
    println!("[STEP 3] Setting up RLM router...");
    let temp_dir = TempDir::new().unwrap();
    let rlm_router: Option<Arc<GatewayResultRouter>> = match create_rlm_router(temp_dir.path()).await {
        Ok(r) => {
            println!("  ✓ RLM router created\n");
            Some(Arc::new(r))
        }
        Err(e) => {
            println!("  ⚠ RLM router creation failed (will skip RLM mode): {}\n", e);
            None
        }
    };

    // Step 4: Setup judge
    println!("[STEP 4] Setting up claim judge...");
    let judge = match ClaimJudge::new() {
        Ok(j) => {
            println!("  ✓ Claim judge created\n");
            j
        }
        Err(e) => {
            println!("  ✗ Failed to create claim judge: {}\n", e);
            return;
        }
    };

    // Step 5: Discover tools
    println!("[STEP 5] Discovering Gateway tools...");
    let runner = match TaskRunner::new(client.clone(), rlm_router.clone()).await {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ Failed to create runner: {}\n", e);
            return;
        }
    };

    let tools = runner.available_tools();
    println!("  Found {} tools:\n", tools.len());
    for tool in tools {
        println!("    - {} ({}): {}", tool.name, tool.server, &tool.description[..tool.description.len().min(60)]);
    }
    println!();

    // Step 6: Select 5 random tasks with at least one resolvable tool
    println!("[STEP 6] Selecting 5 random tasks...");
    let eligible_tasks: Vec<_> = tasks.iter()
        .filter(|task| {
            !task.enabled_tools.is_empty() &&
            task.enabled_tools.iter().any(|t| runner.resolve_tool(t).is_some())
        })
        .collect();

    println!("  Eligible tasks (with resolvable tools): {}/{}\n", eligible_tasks.len(), tasks.len());

    // Select 5 specific indices for reproducibility: first, last, and 3 spread through middle
    let indices = if eligible_tasks.len() >= 5 {
        vec![0, eligible_tasks.len() / 4, eligible_tasks.len() / 2, 3 * eligible_tasks.len() / 4, eligible_tasks.len() - 1]
    } else {
        (0..eligible_tasks.len().min(5)).collect()
    };

    let selected_tasks: Vec<_> = indices.iter()
        .filter_map(|&i| eligible_tasks.get(i).copied())
        .collect();

    println!("  Selected {} tasks for verbose evaluation:\n", selected_tasks.len());
    for (i, task) in selected_tasks.iter().enumerate() {
        println!("    {}. {} - {}", i + 1, task.task_id, &task.prompt[..task.prompt.len().min(50)]);
    }
    println!();

    // Step 7: Run verbose evaluation on each task
    for (task_num, task) in selected_tasks.iter().enumerate() {
        println!("\n");
        println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
        println!("║  TASK {}/{}: {}  ", task_num + 1, selected_tasks.len(), task.task_id);
        println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
        println!();

        // Show task details
        println!("┌─ TASK DETAILS ─────────────────────────────────────────────────────────────────┐");
        println!("│ Prompt:");
        for line in task.prompt.lines() {
            println!("│   {}", line);
        }
        println!("│");
        println!("│ Enabled Tools: {:?}", task.enabled_tools);
        println!("│");
        println!("│ Tool Resolution:");
        for tool_name in &task.enabled_tools {
            if let Some(gateway_tool) = runner.resolve_tool(tool_name) {
                println!("│   ✓ {} → {} ({})", tool_name, gateway_tool.name, gateway_tool.server);
            } else {
                println!("│   ✗ {} → NOT FOUND", tool_name);
            }
        }
        println!("│");
        println!("│ Claims to verify ({}):", task.claims.len());
        for (i, claim) in task.claims.iter().enumerate() {
            println!("│   {}. {}", i + 1, claim);
        }
        println!("│");
        println!("│ Expected Trajectory ({} steps):", task.trajectory.len());
        for (i, step) in task.trajectory.iter().enumerate() {
            println!("│   {}. {} with args: {}", i + 1, step.tool, step.args);
        }
        println!("└────────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Run in RLM mode only (the mode that had best results)
        let mode = ExecutionMode::BaselineRlm;

        println!("┌─ EXECUTION ({}) ────────────────────────────────────────────────────────────┐", mode);
        println!("│");
        println!("│ Sending prompt to agent (GPT-4o)...");

        let start = Instant::now();
        let task_result = runner.run_task(task, mode).await;
        let elapsed = start.elapsed();

        println!("│ Execution completed in {:?}", elapsed);
        println!("│");

        // Show tool calls
        if task_result.tool_calls.is_empty() {
            println!("│ ⚠ NO TOOL CALLS MADE!");
            println!("│ The agent did not use any tools.");
        } else {
            println!("│ Tool Calls Made ({}):", task_result.tool_calls.len());
            for (i, call) in task_result.tool_calls.iter().enumerate() {
                println!("│   ── Call {} ──", i + 1);
                println!("│   Tool: {}", call.name);
                println!("│   Arguments: {}", serde_json::to_string_pretty(&call.arguments).unwrap_or_default().replace('\n', "\n│   "));
                println!("│   Result tokens: {}", call.result_tokens);
                println!("│   Stored in corpus: {}", call.stored_in_corpus);

                // Truncate result for display
                let result_preview = if call.result.len() > 500 {
                    format!("{}...\n│   [TRUNCATED: {} more chars]", &call.result[..500], call.result.len() - 500)
                } else {
                    call.result.clone()
                };
                println!("│   Result: {}", result_preview.replace('\n', "\n│   "));
            }
        }
        println!("│");

        // Show error if any
        if let Some(ref error) = task_result.error {
            println!("│ ✗ ERROR: {}", error);
            println!("│");
        }

        // Show final answer
        println!("│ Final Answer:");
        let answer_preview = if task_result.final_answer.len() > 1000 {
            format!("{}...\n│   [TRUNCATED: {} more chars]", &task_result.final_answer[..1000], task_result.final_answer.len() - 1000)
        } else if task_result.final_answer.is_empty() {
            "[EMPTY ANSWER]".to_string()
        } else {
            task_result.final_answer.clone()
        };
        for line in answer_preview.lines() {
            println!("│   {}", line);
        }
        println!("│");
        println!("│ Context tokens used: {}", task_result.context_tokens);
        println!("└────────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Show claim verification
        println!("┌─ CLAIM VERIFICATION ───────────────────────────────────────────────────────────┐");
        println!("│");
        println!("│ Sending to judge (GPT-4o)...");

        let verify_start = Instant::now();
        let verification = match judge.verify_claims(&task.prompt, &task_result.final_answer, &task.claims).await {
            Ok(v) => {
                println!("│ Verification completed in {:?}", verify_start.elapsed());
                v
            }
            Err(e) => {
                println!("│ ✗ Verification FAILED: {}", e);
                ClaimVerificationResult {
                    scores: vec![],
                    coverage: 0.0,
                    passed: false,
                    raw_response: e.to_string(),
                }
            }
        };

        println!("│");
        println!("│ Per-claim results:");
        for (claim, score) in &verification.scores {
            let (score_str, symbol) = match score {
                ClaimScore::Fulfilled => ("FULFILLED", "✓"),
                ClaimScore::PartiallyFulfilled => ("PARTIAL  ", "~"),
                ClaimScore::NotFulfilled => ("NOT_MET  ", "✗"),
            };
            println!("│   {} {} {}", symbol, score_str, &claim[..claim.len().min(60)]);
        }
        println!("│");
        println!("│ Coverage: {:.2}", verification.coverage);
        println!("│ Threshold: {:.2}", PASS_THRESHOLD);
        println!("│ Status: {}", if verification.passed { "✓ PASS" } else { "✗ FAIL" });
        println!("│");

        // Show raw judge response for debugging
        if !verification.raw_response.is_empty() && !verification.passed {
            println!("│ Judge Raw Response (for debugging):");
            let raw_preview = if verification.raw_response.len() > 500 {
                format!("{}...", &verification.raw_response[..500])
            } else {
                verification.raw_response.clone()
            };
            for line in raw_preview.lines() {
                println!("│   {}", line);
            }
        }
        println!("└────────────────────────────────────────────────────────────────────────────────┘");

        // Summary
        println!();
        println!("┌─ TASK SUMMARY ─────────────────────────────────────────────────────────────────┐");
        println!("│ Task: {}", task.task_id);
        println!("│ Tool calls: {}", task_result.tool_calls.len());
        println!("│ Expected trajectory steps: {}", task.trajectory.len());
        println!("│ Coverage: {:.2}", verification.coverage);
        println!("│ Result: {}", if verification.passed { "✓ PASS" } else { "✗ FAIL" });

        // Diagnosis
        println!("│");
        println!("│ DIAGNOSIS:");
        if task_result.tool_calls.is_empty() {
            println!("│   - Agent made NO tool calls (expected {})", task.trajectory.len());
            println!("│   - Check: Is the system prompt telling agent to use tools?");
            println!("│   - Check: Are tool definitions correctly formatted?");
        } else if task_result.tool_calls.len() != task.trajectory.len() {
            println!("│   - Agent made {} calls, expected {}", task_result.tool_calls.len(), task.trajectory.len());
        }

        if let Some(ref error) = task_result.error {
            println!("│   - Execution error: {}", error);
        }

        if !verification.passed && task_result.final_answer.is_empty() {
            println!("│   - Final answer is EMPTY!");
        }

        // Check if tool calls match expected trajectory
        let expected_tools: Vec<_> = task.trajectory.iter().map(|s| &s.tool).collect();
        let actual_tools: Vec<_> = task_result.tool_calls.iter().map(|c| &c.name).collect();
        if expected_tools != actual_tools {
            println!("│   - Tool sequence mismatch:");
            println!("│     Expected: {:?}", expected_tools);
            println!("│     Actual:   {:?}", actual_tools);
        }

        println!("└────────────────────────────────────────────────────────────────────────────────┘");
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("       VERBOSE DEBUG EVALUATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

/// Print setup instructions
#[tokio::test]
async fn print_setup_instructions() {
    if !should_skip() {
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("           MCP-Atlas Evaluation - Setup Instructions");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("This evaluation requires:");
    println!();
    println!("  1. Kontext Gateway credentials");
    println!("  2. OpenAI API key (for agent and judge)");
    println!("  3. MCP-Atlas dataset");
    println!();
    println!("Setup:");
    println!();
    println!("  # Create .env file:");
    println!("  KONTEXT_CLIENT_ID=<your-client-id>");
    println!("  KONTEXT_CLIENT_SECRET=<your-client-secret>");
    println!("  KONTEXT_GATEWAY_URL=https://gateway.kontext.dev");
    println!("  OPENAI_API_KEY=<your-openai-key>");
    println!();
    println!("  # Download dataset (if not present):");
    println!("  # Dataset should be at: ../data/coding_atlas/data-00000-of-00001.arrow");
    println!();
    println!("Run tests:");
    println!();
    println!("  # Verification tests:");
    println!("  source .env");
    println!("  cargo test -p codex-core --test mcp_atlas_eval -- --nocapture");
    println!();
    println!("  # Three-way evaluation (10 tasks):");
    println!("  cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture");
    println!();
    println!("  # Full evaluation (500 tasks - takes a long time):");
    println!("  cargo test -p codex-core --test mcp_atlas_eval run_full_mcp_atlas_evaluation -- --nocapture --ignored");
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!();
}
