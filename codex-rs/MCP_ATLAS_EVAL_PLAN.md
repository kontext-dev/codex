# MCP-Atlas Evaluation Script - Execution Plan

> **IMPORTANT**: This implementation uses **REAL MCP servers** (Git, Code Executor, CLI) through the Kontext Gateway. No mock data.

## Goal

Create an evaluation script that benchmarks LLM agent performance on the MCP-Atlas dataset across **three execution modes**:

| Mode | Description | Tool Execution |
|------|-------------|----------------|
| **Baseline** | Direct tool calls | EXECUTE_TOOL → full results in context |
| **Code Mode** | Batched execution | EXECUTE_CODE → summarized results |
| **Baseline + RLM** | Direct + routing | EXECUTE_TOOL → large results to corpus |

This gives us **empirical quality measurements** (not hardcoded estimates) via claims-based scoring.

---

## Dataset

**Location**: `/Users/vishi/repos/kontext-codex/codex/data/coding_atlas/`

**Files**:
- `data-00000-of-00001.arrow` (16MB, 500 tasks)
- `dataset_info.json` (schema definition)

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| TASK | string | Task identifier |
| ENABLED_TOOLS | string (JSON) | Tools to expose (Texpose) |
| PROMPT | string | Natural language request |
| GTFA_CLAIMS | string (JSON) | Ground truth claims (C*) |
| TRAJECTORY | string (JSON) | Reference solution (π*) |

---

## Evaluation Methodology

### Primary Scoring: Claims-Based Rubric

1. **Per-claim grading** (GPT-4o judge):
   - `fulfilled` = 1.0
   - `partially_fulfilled` = 0.5
   - `not_fulfilled` = 0.0

2. **Coverage score**:
   ```
   Coverage = Σ score(claim_i) / |C*|
   ```

3. **Pass/Fail threshold**:
   - Task passes if **Coverage ≥ 0.75**

---

## Implementation Plan

### File Structure

```
codex-rs/
├── core/tests/
│   └── mcp_atlas_eval.rs       # Main evaluation script (NEW)
├── core/src/
│   └── eval/                   # Evaluation module (NEW)
│       ├── mod.rs
│       ├── dataset.rs          # Arrow dataset loader
│       ├── judge.rs            # GPT-4o claim verification
│       └── runner.rs           # Task execution
```

---

### Step 1: Dataset Loader (`dataset.rs`)

**Purpose**: Load MCP-Atlas tasks from Arrow IPC file

```rust
use arrow::array::StringArray;
use arrow::ipc::reader::FileReader;
use std::fs::File;

pub struct McpAtlasTask {
    pub task_id: String,
    pub enabled_tools: Vec<String>,  // Parsed from JSON
    pub prompt: String,
    pub claims: Vec<String>,         // Parsed from JSON
    pub trajectory: Option<String>,  // For diagnostics only
}

pub fn load_dataset(path: &str) -> Result<Vec<McpAtlasTask>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = FileReader::try_new(file, None)?;

    let mut tasks = Vec::new();
    for batch in reader {
        let batch = batch?;
        let task_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let tools_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let prompt_col = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let claims_col = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let traj_col = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..batch.num_rows() {
            tasks.push(McpAtlasTask {
                task_id: task_col.value(i).to_string(),
                enabled_tools: serde_json::from_str(tools_col.value(i))?,
                prompt: prompt_col.value(i).to_string(),
                claims: serde_json::from_str(claims_col.value(i))?,
                trajectory: Some(traj_col.value(i).to_string()),
            });
        }
    }
    Ok(tasks)
}
```

---

### Step 2: Task Runner (`runner.rs`)

**Purpose**: Execute tasks in three different modes

```rust
use codex_rmcp_client::RmcpClient;
use codex_core::rlm::gateway_intercept::{GatewayResultRouter, ProcessedResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    Baseline,      // EXECUTE_TOOL → full results in context
    CodeMode,      // EXECUTE_CODE → summarized results
    BaselineRlm,   // EXECUTE_TOOL → large results routed to corpus
}

pub struct TaskRunner {
    gateway_client: RmcpClient,
    openai_api_key: String,
    rlm_router: GatewayResultRouter,
}

impl TaskRunner {
    pub async fn run_task(&self, task: &McpAtlasTask, mode: ExecutionMode) -> TaskResult {
        // 1. Build tool definitions from task.enabled_tools
        let tools = self.build_tool_definitions(&task.enabled_tools);

        // 2. Agent loop: prompt → LLM → tool calls → results → final answer
        let mut messages = vec![ChatMessage::user(&task.prompt)];
        let mut tool_calls_log = Vec::new();
        let mut total_context_tokens = 0i64;

        loop {
            // Call GPT-4o with tools
            let response = self.call_openai(&messages, &tools).await?;
            total_context_tokens += response.usage.prompt_tokens;

            if let Some(tool_calls) = response.tool_calls {
                for call in tool_calls {
                    // Execute tool via Gateway based on mode
                    let result = match mode {
                        ExecutionMode::Baseline => {
                            self.execute_tool_baseline(&call).await
                        }
                        ExecutionMode::CodeMode => {
                            self.execute_tool_codemode(&call).await
                        }
                        ExecutionMode::BaselineRlm => {
                            self.execute_tool_with_rlm(&call).await
                        }
                    };
                    tool_calls_log.push(call.clone());
                    messages.push(ChatMessage::tool_result(&call.id, &result));
                }
            } else {
                // Final answer reached
                return TaskResult {
                    task_id: task.task_id.clone(),
                    final_answer: response.content.unwrap_or_default(),
                    tool_calls: tool_calls_log,
                    mode,
                    context_tokens: total_context_tokens,
                };
            }
        }
    }

    async fn execute_tool_baseline(&self, call: &ToolCall) -> String {
        // Call EXECUTE_TOOL, return full result
        let result = self.gateway_client.call_tool(
            "EXECUTE_TOOL".to_string(),
            Some(serde_json::json!({
                "tool": call.function.name,
                "args": call.function.arguments
            })),
            Some(Duration::from_secs(120)),
        ).await;
        serialize_result(&result)
    }

    async fn execute_tool_codemode(&self, call: &ToolCall) -> String {
        // Call EXECUTE_CODE with summarizing code
        let code = format!(
            r#"const r = await tools.{}({});
               return JSON.stringify({{count: Array.isArray(r) ? r.length : 1, summary: 'processed'}});"#,
            call.function.name, call.function.arguments
        );
        let result = self.gateway_client.call_tool(
            "EXECUTE_CODE".to_string(),
            Some(serde_json::json!({"code": code})),
            Some(Duration::from_secs(120)),
        ).await;
        serialize_result(&result)
    }

    async fn execute_tool_with_rlm(&self, call: &ToolCall) -> String {
        // Call EXECUTE_TOOL, route through RLM
        let result = self.execute_tool_baseline(call).await;
        let call_id = uuid::Uuid::new_v4().to_string();

        match self.rlm_router.process_result(&call_id, "gateway", &call.function.name, &result).await {
            Ok(ProcessedResult::PassThrough { content }) => content,
            Ok(ProcessedResult::StoredInCorpus { summary, .. }) => summary,
            Err(_) => result,
        }
    }
}

pub struct TaskResult {
    pub task_id: String,
    pub final_answer: String,
    pub tool_calls: Vec<ToolCall>,
    pub mode: ExecutionMode,
    pub context_tokens: i64,
}
```

---

### Step 3: Claim Judge (`judge.rs`)

**Purpose**: Verify claims against agent answers using GPT-4o

```rust
pub struct ClaimJudge {
    openai_api_key: String,
}

impl ClaimJudge {
    pub async fn verify_claims(
        &self,
        answer: &str,
        claims: &[String],
    ) -> ClaimVerificationResult {
        let prompt = format!(
            r#"You are a claim verification judge. Given an answer and a list of claims,
            verify each claim against the answer.

            Answer:
            {}

            Claims to verify:
            {}

            For each claim, respond with one of:
            - "fulfilled" (claim is fully supported by the answer)
            - "partially_fulfilled" (claim is partially supported)
            - "not_fulfilled" (claim is not supported or contradicted)

            Respond in JSON format:
            {{"scores": ["fulfilled", "partially_fulfilled", ...]}}"#,
            answer,
            claims.iter().enumerate()
                .map(|(i, c)| format!("{}. {}", i+1, c))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let response = self.call_openai_json(&prompt).await?;
        let scores: Vec<ClaimScore> = response.scores.iter()
            .map(|s| match s.as_str() {
                "fulfilled" => ClaimScore::Fulfilled,
                "partially_fulfilled" => ClaimScore::PartiallyFulfilled,
                _ => ClaimScore::NotFulfilled,
            })
            .collect();

        let coverage = scores.iter()
            .map(|s| s.value())
            .sum::<f64>() / scores.len() as f64;

        ClaimVerificationResult {
            scores,
            coverage,
            passed: coverage >= 0.75,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ClaimScore {
    Fulfilled,           // 1.0
    PartiallyFulfilled,  // 0.5
    NotFulfilled,        // 0.0
}

impl ClaimScore {
    pub fn value(&self) -> f64 {
        match self {
            ClaimScore::Fulfilled => 1.0,
            ClaimScore::PartiallyFulfilled => 0.5,
            ClaimScore::NotFulfilled => 0.0,
        }
    }
}

pub struct ClaimVerificationResult {
    pub scores: Vec<ClaimScore>,
    pub coverage: f64,
    pub passed: bool,
}
```

---

### Step 4: Main Evaluation Script (`mcp_atlas_eval.rs`)

**Purpose**: Orchestrate the three-way evaluation

```rust
#[tokio::test]
#[ignore] // Run with --ignored for full evaluation
async fn run_mcp_atlas_three_way_evaluation() {
    // 1. Load dataset
    let dataset_path = "../../data/coding_atlas/data-00000-of-00001.arrow";
    let tasks = load_dataset(dataset_path).expect("Failed to load dataset");

    // Subset for testing (remove for full eval)
    let tasks = &tasks[0..10];

    println!("Loaded {} tasks", tasks.len());

    // 2. Setup Gateway connection + RLM
    let gateway_client = create_gateway_client().await;
    let rlm_router = GatewayResultRouter::new(RlmConfig::default());
    let runner = TaskRunner::new(gateway_client, rlm_router);

    // 3. Setup judge
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");
    let judge = ClaimJudge::new(api_key);

    // 4. Run THREE-WAY evaluation
    let modes = [
        ExecutionMode::Baseline,
        ExecutionMode::CodeMode,
        ExecutionMode::BaselineRlm,
    ];

    let mut all_results: HashMap<ExecutionMode, Vec<EvalResult>> = HashMap::new();

    for mode in &modes {
        println!("\n{'═'.repeat(60)}");
        println!("Running {:?} mode", mode);
        println!("{'═'.repeat(60)}\n");

        let mut results = Vec::new();

        for (i, task) in tasks.iter().enumerate() {
            println!("  Task {}/{}: {}", i+1, tasks.len(), task.task_id);

            // Execute task in this mode
            let task_result = runner.run_task(task, *mode).await;

            // Judge the answer
            let verification = judge.verify_claims(
                &task_result.final_answer,
                &task.claims,
            ).await;

            println!("    Coverage: {:.2} {}",
                verification.coverage,
                if verification.passed { "✓" } else { "✗" }
            );

            results.push(EvalResult {
                task_id: task.task_id.clone(),
                task_result,
                verification,
            });
        }

        all_results.insert(*mode, results);
    }

    // 5. Report comparative results
    print_three_way_comparison(&all_results);
}
```

---

### Output Format

```
# MCP-Atlas Three-Way Evaluation Results

## Mode Comparison

| Mode | Pass Rate | Avg Coverage | Avg Tokens | Token Reduction |
|------|-----------|--------------|------------|-----------------|
| Baseline | 68.4% | 0.72 | 45,000 | - |
| Code Mode | 52.0% | 0.58 | 4,500 | -90% |
| Baseline + RLM | 67.2% | 0.71 | 2,100 | -95% |

## Key Insight

RLM achieves **95% token reduction** while maintaining **98% of baseline quality**
(vs Code Mode which loses 24% quality for 90% token reduction)

## Per-Task Comparison (sample)

| Task | Baseline | Code Mode | RLM | Winner |
|------|----------|-----------|-----|--------|
| task_001 | 1.00 ✓ | 0.75 ✓ | 1.00 ✓ | tie |
| task_002 | 0.80 ✓ | 0.40 ✗ | 0.80 ✓ | Baseline/RLM |
| task_003 | 0.50 ✗ | 0.25 ✗ | 0.50 ✗ | tie (all fail) |
| task_004 | OVERFLOW | 0.60 ✗ | 0.90 ✓ | RLM |

## Coverage Distribution by Mode

| Threshold | Baseline | Code Mode | RLM |
|-----------|----------|-----------|-----|
| ≥0.65 | 75.2% | 58.4% | 74.0% |
| ≥0.75 | 68.4% | 52.0% | 67.2% |
| ≥0.85 | 52.1% | 38.6% | 51.4% |
```

---

## Dependencies to Add

```toml
# core/Cargo.toml [dev-dependencies]
arrow = { version = "53", default-features = false, features = ["ipc"] }
async-openai = "0.28"
```

---

## Tool Mapping (Real MCP Servers)

**IMPORTANT**: This evaluation uses **real MCP servers** through the Kontext Gateway - NO mock data.

Gateway MCP servers available:
| MCP Server | Gateway Name | Operations |
|------------|--------------|------------|
| **Git** | `git` | clone, status, diff, commit, log, branch |
| **Code Executor** | `code_executor` | execute code in sandboxed environment |
| **CLI** | `cli` | run shell commands |

These are called via EXECUTE_TOOL:
```rust
// Real MCP call through Gateway
client.call_tool("EXECUTE_TOOL", json!({
    "tool": "git_status",  // or "cli_run", "code_executor_run"
    "args": { "repo": "/path/to/repo" }
}))
```

The ENABLED_TOOLS column in the dataset specifies which tools each task can use.

---

## Verification Tests

### Test 1: Dataset Loading (`test_dataset_loading`)

Verify Arrow dataset can be loaded and parsed correctly.

```rust
#[tokio::test]
async fn test_dataset_loading() {
    let tasks = load_dataset("../../data/coding_atlas/data-00000-of-00001.arrow")
        .expect("Failed to load dataset");

    // Verify dataset loaded
    assert_eq!(tasks.len(), 500, "Expected 500 tasks");

    // Verify first task has required fields
    let first = &tasks[0];
    assert!(!first.task_id.is_empty(), "Task ID should not be empty");
    assert!(!first.prompt.is_empty(), "Prompt should not be empty");
    assert!(!first.claims.is_empty(), "Claims should not be empty");
    assert!(!first.enabled_tools.is_empty(), "Enabled tools should not be empty");

    println!("✓ Loaded {} tasks", tasks.len());
    println!("  First task: {}", first.task_id);
    println!("  Enabled tools: {:?}", first.enabled_tools);
    println!("  Claims count: {}", first.claims.len());
}
```

### Test 2: Real Gateway Connection (`test_gateway_connection`)

Verify connection to Kontext Gateway with OAuth.

```rust
#[tokio::test]
async fn test_gateway_connection() {
    // Skip if credentials not configured
    let client_id = std::env::var("KONTEXT_CLIENT_ID")
        .expect("KONTEXT_CLIENT_ID required");
    let client_secret = std::env::var("KONTEXT_CLIENT_SECRET")
        .expect("KONTEXT_CLIENT_SECRET required");

    let client = create_gateway_client().await
        .expect("Failed to connect to Gateway");

    // List available tools
    let tools = client.list_tools().await.expect("Failed to list tools");

    println!("✓ Connected to Gateway");
    println!("  Available tools: {}", tools.len());

    // Verify expected MCP servers are available
    let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.iter().any(|n| n.contains("git") || n.contains("EXECUTE")),
        "Expected git or EXECUTE tools");

    println!("  Tools: {:?}", &tool_names[..tool_names.len().min(10)]);
}
```

### Test 3: Real MCP Tool Calls (`test_real_mcp_tool_calls`)

Verify actual MCP tool execution through Gateway.

```rust
#[tokio::test]
async fn test_real_mcp_tool_calls() {
    let client = create_gateway_client().await
        .expect("Failed to connect to Gateway");

    // Test 1: SEARCH_TOOLS (meta-tool)
    println!("\n--- Testing SEARCH_TOOLS ---");
    let search_result = client.call_tool(
        "SEARCH_TOOLS".to_string(),
        Some(json!({"query": "git"})),
        Some(Duration::from_secs(30)),
    ).await;
    assert!(search_result.is_ok(), "SEARCH_TOOLS should succeed");
    println!("✓ SEARCH_TOOLS returned {} bytes",
        serialize_result(&search_result).len());

    // Test 2: EXECUTE_TOOL with CLI
    println!("\n--- Testing EXECUTE_TOOL (CLI) ---");
    let cli_result = client.call_tool(
        "EXECUTE_TOOL".to_string(),
        Some(json!({
            "tool": "cli_run",
            "args": {"command": "echo 'hello from MCP'"}
        })),
        Some(Duration::from_secs(30)),
    ).await;
    println!("CLI result: {:?}", cli_result);

    // Test 3: EXECUTE_CODE
    println!("\n--- Testing EXECUTE_CODE ---");
    let code_result = client.call_tool(
        "EXECUTE_CODE".to_string(),
        Some(json!({
            "code": "return 'Code execution works!';"
        })),
        Some(Duration::from_secs(30)),
    ).await;
    println!("Code result: {:?}", code_result);

    println!("\n✓ All real MCP tool calls completed");
}
```

### Test 4: Claim Judge Verification (`test_claim_judge`)

Verify GPT-4o claim verification works correctly.

```rust
#[tokio::test]
async fn test_claim_judge() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY required");

    let judge = ClaimJudge::new(api_key);

    // Test with known answer and claims
    let answer = "The repository has 3 branches: main, develop, and feature-x. \
                  The latest commit was made by Alice on 2024-01-15.";

    let claims = vec![
        "The repository has exactly 3 branches".to_string(),
        "One of the branches is named 'main'".to_string(),
        "The latest commit was made in January 2024".to_string(),
        "Bob made the latest commit".to_string(),  // Should fail
    ];

    let result = judge.verify_claims(&answer, &claims).await
        .expect("Claim verification failed");

    println!("✓ Claim judge results:");
    for (i, (claim, score)) in claims.iter().zip(&result.scores).enumerate() {
        println!("  {}. {:?} - {}", i+1, score, claim);
    }
    println!("  Coverage: {:.2}", result.coverage);
    println!("  Passed: {}", result.passed);

    // First 3 should be fulfilled, last should not
    assert!(matches!(result.scores[0], ClaimScore::Fulfilled | ClaimScore::PartiallyFulfilled));
    assert!(matches!(result.scores[3], ClaimScore::NotFulfilled));
}
```

### Test 5: Single Task Execution (`test_single_task_execution`)

Run a single MCP-Atlas task through all three modes.

```rust
#[tokio::test]
async fn test_single_task_execution() {
    // Load first task from dataset
    let tasks = load_dataset("../../data/coding_atlas/data-00000-of-00001.arrow")
        .expect("Failed to load dataset");
    let task = &tasks[0];

    println!("Testing task: {}", task.task_id);
    println!("Prompt: {}", &task.prompt[..task.prompt.len().min(100)]);
    println!("Enabled tools: {:?}", task.enabled_tools);
    println!("Claims: {} total", task.claims.len());

    // Setup
    let gateway_client = create_gateway_client().await.expect("Gateway connection failed");
    let rlm_router = GatewayResultRouter::new(RlmConfig::default());
    let runner = TaskRunner::new(gateway_client, rlm_router);
    let judge = ClaimJudge::new(std::env::var("OPENAI_API_KEY").unwrap());

    // Run in all three modes
    let modes = [
        ExecutionMode::Baseline,
        ExecutionMode::CodeMode,
        ExecutionMode::BaselineRlm,
    ];

    println!("\n| Mode | Tokens | Coverage | Status |");
    println!("|------|--------|----------|--------|");

    for mode in &modes {
        let result = runner.run_task(task, *mode).await;
        let verification = judge.verify_claims(&result.final_answer, &task.claims).await
            .expect("Verification failed");

        println!("| {:?} | {} | {:.2} | {} |",
            mode,
            result.context_tokens,
            verification.coverage,
            if verification.passed { "✓" } else { "✗" }
        );
    }
}
```

### Test 6: Three-Way Comparison (10 tasks) (`test_three_way_comparison_subset`)

Run full three-way comparison on first 10 tasks.

```rust
#[tokio::test]
async fn test_three_way_comparison_subset() {
    let tasks = load_dataset("../../data/coding_atlas/data-00000-of-00001.arrow")
        .expect("Failed to load dataset");
    let tasks = &tasks[0..10];  // First 10 tasks

    let gateway_client = create_gateway_client().await.expect("Gateway failed");
    let rlm_router = GatewayResultRouter::new(RlmConfig::default());
    let runner = TaskRunner::new(gateway_client, rlm_router);
    let judge = ClaimJudge::new(std::env::var("OPENAI_API_KEY").unwrap());

    let modes = [
        ExecutionMode::Baseline,
        ExecutionMode::CodeMode,
        ExecutionMode::BaselineRlm,
    ];

    let mut mode_stats: HashMap<ExecutionMode, (usize, f64, i64)> = HashMap::new();

    for mode in &modes {
        let mut pass_count = 0;
        let mut total_coverage = 0.0;
        let mut total_tokens = 0i64;

        for task in tasks {
            let result = runner.run_task(task, *mode).await;
            let verification = judge.verify_claims(&result.final_answer, &task.claims).await?;

            if verification.passed { pass_count += 1; }
            total_coverage += verification.coverage;
            total_tokens += result.context_tokens;
        }

        mode_stats.insert(*mode, (pass_count, total_coverage, total_tokens));
    }

    // Print comparison
    println!("\n# Three-Way Comparison (10 tasks)\n");
    println!("| Mode | Pass Rate | Avg Coverage | Avg Tokens |");
    println!("|------|-----------|--------------|------------|");

    for mode in &modes {
        let (pass, cov, tok) = mode_stats.get(mode).unwrap();
        println!("| {:?} | {}/10 ({:.0}%) | {:.3} | {} |",
            mode,
            pass,
            *pass as f64 / 10.0 * 100.0,
            cov / 10.0,
            tok / 10
        );
    }
}
```

---

## Verification Commands

```bash
# 1. Test dataset loading
cargo test -p codex-core --test mcp_atlas_eval test_dataset_loading -- --nocapture

# 2. Test Gateway connection (requires .env)
export $(grep -v '^#' .env | xargs)
cargo test -p codex-core --test mcp_atlas_eval test_gateway_connection -- --nocapture

# 3. Test real MCP tool calls
cargo test -p codex-core --test mcp_atlas_eval test_real_mcp_tool_calls -- --nocapture

# 4. Test claim judge (requires OPENAI_API_KEY)
cargo test -p codex-core --test mcp_atlas_eval test_claim_judge -- --nocapture

# 5. Test single task execution (all three modes)
cargo test -p codex-core --test mcp_atlas_eval test_single_task_execution -- --nocapture

# 6. Run 10-task comparison
cargo test -p codex-core --test mcp_atlas_eval test_three_way_comparison_subset -- --nocapture

# 7. Full evaluation (500 tasks × 3 modes)
cargo test -p codex-core --test mcp_atlas_eval run_mcp_atlas_three_way_evaluation -- --nocapture --ignored
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/src/eval/mod.rs` | Module declaration and exports |
| `core/src/eval/dataset.rs` | Arrow dataset loader for MCP-Atlas |
| `core/src/eval/judge.rs` | GPT-4o claim verification |
| `core/src/eval/runner.rs` | Three-mode task execution via real Gateway |
| `core/tests/mcp_atlas_eval.rs` | Main evaluation test + 6 verification tests |

### Test Coverage

| Test | What it Verifies |
|------|------------------|
| `test_dataset_loading` | Arrow file parsing, schema validation |
| `test_gateway_connection` | OAuth auth, Gateway connectivity |
| `test_real_mcp_tool_calls` | SEARCH_TOOLS, EXECUTE_TOOL, EXECUTE_CODE |
| `test_claim_judge` | GPT-4o claim verification accuracy |
| `test_single_task_execution` | End-to-end single task in all 3 modes |
| `test_three_way_comparison_subset` | 10-task comparison with metrics |
| `run_mcp_atlas_three_way_evaluation` | Full 500-task evaluation (ignored) |

---

## Expected Outcome

This evaluation will provide **empirical evidence** for RLM quality:
- Current benchmarks use hardcoded quality estimates (100% for RLM, 40% for CodeMode)
- MCP-Atlas provides ground-truth claims for objective measurement
- Results will validate (or refute) the claim that RLM preserves quality while reducing tokens
