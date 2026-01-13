# RLM Benchmarking Methodology

This document describes the benchmarking methodology for evaluating RLM  performance in Codex CLI. The methodology compares three approaches for handling large prompts and tool results.

## Problem Statement

When prompts or tool results exceed context windows:

| Problem | Impact |
|---------|--------|
| **Truncation loses context** | Critical information at the end is lost |
| **No iterative inspection** | Model can't "look around" the corpus |
| **No provenance** | No way to trace answers back to sources |
| **No decomposition** | Large analysis done in single pass (or fails) |

## Solution: RLM Layer on Codex

The RLM pattern treats long prompts as an external environment:

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query (arbitrarily long)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  RLM Corpus Storage Layer                        │
│  • Prompt stored as files in workspace (not in context)         │
│  • Manifest: doc boundaries, chunk IDs, sizes                   │
│  • Index: keyword/regex searchable                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Root Codex Session (RLM Controller)                 │
│  • Sees: corpus metadata + RLM toolkit (not full corpus)        │
│  • System prompt: "corpus in workspace, use tools to inspect"   │
│  • Loop: plan → query → reduce → recurse? → finalize            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RLM Toolkit (Tools)                           │
│  A) Bounded inspection: size/structure, chunk retrieval         │
│  B) Search: keyword/regex → references (not full text)          │
│  C) Evidence bundle: assemble citations with provenance         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Sub-LM Sessions (Recursion)                         │
│  • Root selects snippet → spawns sub-task                       │
│  • Sub-agent: "Given ONLY this snippet, extract X"              │
│  • Returns: structured result (facts + local citations)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Evidence Bundle + Trace                       │
│  • Answer + citations + reasoning artifacts                     │
│  • Trace: which chunks accessed and why                         │
│  • Provenance: chunk → sub-agent → conclusion                   │
└─────────────────────────────────────────────────────────────────┘
```

## Three-Way Comparison

The benchmark compares three approaches:

### 1. Baseline (Truncation)

Traditional approach where tool results are added directly to conversation history:

```
Turn 1: SEARCH_TOOLS → full schemas (9000 tokens)
Turn 2: EXECUTE_TOOL(list_issues) → 200 issues (45,000 tokens added)
Turn 3: EXECUTE_TOOL(list_projects) → 15 projects (8,000 tokens added)
Turn 4: Final response

Total: ~62,000 tokens, 4 turns
Risk: Context overflow with large results
```

### 2. Code Mode (Batched Execution)

Uses `EXECUTE_CODE` to batch tool calls and return summarized results:

```
Turn 1: SEARCH_TOOLS → minimal list (300 tokens)
Turn 2: EXECUTE_CODE batches both calls, returns counts only
Turn 3: Final response

Total: ~2,500 tokens, 3 turns
Limitation: Fixed summarization logic, no LLM reasoning over details
```

### 3. RLM Mode (Bounded Decomposition)

Stores results as evidence, uses sub-LM calls for detailed analysis:

```
Phase 1: SEARCH_TOOLS → minimal list (300 tokens)
Phase 2: EXECUTE_CODE batches calls
  → Results stored as evidence (not in conversation)
Phase 3: Sub-LM decomposition for detailed analysis
Phase 4: Evidence-grounded synthesis with provenance

Total: Variable (budget-controlled)
Benefit: Full provenance DAG, LLM reasoning at each step
```

## Benchmark Suite Structure

```
codex-rs/core/benches/rlm/
├── lib/
│   ├── corpus_generator.rs   # Generate test corpora of various sizes
│   ├── quality_eval.rs       # Evaluate answer quality vs. ground truth
│   ├── token_tracker.rs      # Track token usage across tools/sub-agents
│   └── prompts.rs            # Test prompts with known answers
├── truncation_baseline.bench.rs  # Compare vs. simple truncation
├── rlm_corpus.bench.rs           # Corpus ingestion/chunking
├── rlm_tools.bench.rs            # Tool execution latency
├── rlm_subagent.bench.rs         # Sub-agent spawning via codex_delegate
├── rlm_e2e.bench.rs              # Full RLM workflow
└── results/
    └── [benchmark outputs]
```

## Metrics Captured

### Token Metrics

| Metric | Description |
|--------|-------------|
| `total_tokens` | Cumulative tokens across all LLM calls |
| `tokens_per_call` | Average tokens per sub-LM invocation |
| `token_overhead` | Instruction + context formatting cost |
| `token_utilization` | `total_tokens / max_total_tokens` |

### Latency Metrics

| Metric | Description |
|--------|-------------|
| `e2e_latency_ms` | Wall-clock time from start to finish |
| `sub_lm_latency_ms` | Per-invocation latency |
| `budget_check_ns` | Time for `can_proceed()` calls |
| `evidence_record_ns` | Time for `EvidenceStore::record()` |
| `bundle_build_ns` | Time for `build_bundle()` |

### Recursion Metrics

| Metric | Description |
|--------|-------------|
| `max_depth_reached` | Deepest recursion level used |
| `call_count` | Total sub-LM invocations |
| `depth_distribution` | Calls per depth level |
| `branching_factor` | Avg sub-calls per parent |

### Evidence Metrics

| Metric | Description |
|--------|-------------|
| `evidence_count` | Total evidence items created |
| `evidence_types` | Distribution by EvidenceKind |
| `provenance_edges` | Edges in provenance DAG |
| `bundle_token_usage` | Tokens in final evidence bundle |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| `source_coverage` | % of input sources cited in output |
| `evidence_grounding` | % of claims backed by evidence |
| `factual_accuracy` | Correctness vs. ground truth (manual) |

## Benchmark Prompts

Prompts are designed to test different complexity levels:

```rust
const RLM_BENCHMARK_PROMPTS: &[BenchmarkPrompt] = &[
    BenchmarkPrompt {
        id: "simple_no_recursion",
        prompt: "Summarize this README file.",
        input_tokens: 2_000,
        expected_depth: 0,
        description: "Simple task, no decomposition needed",
    },
    BenchmarkPrompt {
        id: "moderate_single_depth",
        prompt: "Find all error handling patterns in this crate.",
        input_tokens: 50_000,
        expected_depth: 1,
        description: "Moderate task, single level of sub-LM calls",
    },
    BenchmarkPrompt {
        id: "complex_multi_depth",
        prompt: "Analyze security model across all modules.",
        input_tokens: 200_000,
        expected_depth: 3,
        description: "Complex task requiring deep decomposition",
    },
    BenchmarkPrompt {
        id: "exhaustive_max_depth",
        prompt: "Create comprehensive documentation from codebase.",
        input_tokens: 500_000,
        expected_depth: 5,
        description: "Exhaustive task hitting recursion limits",
    },
];
```

## Example Scenarios

### Example 1: Simple Task (2 Parallel Tool Calls)

**Prompt**: "Get the count of issues and count of projects."

| Mode | Tokens | Turns | Sub-LMs | Quality |
|------|--------|-------|---------|---------|
| Baseline | 62,000 | 4 | 0 | 100% |
| Code Mode | 2,500 | 3 | 0 | 100% |
| RLM | 3,000 | 4 | 0 | 100% |

**Result**: For simple tasks, RLM adds ~20% overhead for provenance tracking.

### Example 2: Moderate Task (Fan-Out Pattern)

**Prompt**: "List all projects, then summarize the issues in each one."

| Mode | Tokens | Turns | Sub-LMs | Quality |
|------|--------|-------|---------|---------|
| Baseline | 95,000 | 8 | 0 | 100% |
| Code Mode | 2,800 | 3 | 0 | 40%* |
| RLM | 28,300 | 4 | 5 | 100% |

*Code Mode quality limited by fixed summarization (no LLM reasoning over details)

### Example 3: Complex Task (Cross-Entity Analysis)

**Prompt**: "Analyze team workload distribution and identify bottlenecks."

| Mode | Tokens | Turns | Sub-LMs | Quality |
|------|--------|-------|---------|---------|
| Baseline | 150,000+ | 15 | 0 | FAIL* |
| Code Mode | 3,500 | 3 | 0 | 25% |
| RLM | 95,000 | 6 | 11 | 100% |

*Baseline fails due to context overflow (>128K tokens)

### Example 4: Budget Exhaustion

**Prompt**: "Generate comprehensive quarterly report."
**Config**: `max_total_tokens = 100,000`

```
Phase 1: Tool Discovery (300 tokens)
Phase 2: Data Gathering → Evidence stored (not in context)
Phase 3: Sub-LM Decomposition (within budget)
  - Sub-LM 1-3: Monthly summaries (~30,000 tokens)
  - Sub-LM 4: Cross-month trends (~8,000 tokens)
  - Sub-LM 5-8: Category analyses (~30,000 tokens)
Phase 4: Synthesis (~13,000 tokens)

Total: 80,200 tokens (within 100K budget)
Budget remaining: 19,800 (19.8% unused)
```

## Budget Enforcement Tests

```rust
#[bench]
fn bench_budget_at_50_percent() {
    // Configure: max_total_tokens = 100_000
    // Run until: ~50K tokens consumed
    // Measure: Warning rate, accuracy of remaining estimate
}

#[bench]
fn bench_budget_at_80_percent() {
    // Configure: max_total_tokens = 100_000
    // Run until: ~80K tokens (warning threshold)
    // Measure: Warning triggered, can_proceed() behavior
}

#[bench]
fn bench_budget_at_99_percent() {
    // Configure: max_total_tokens = 100_000
    // Run until: ~99K tokens
    // Measure: Hard stop behavior, final output quality
}

#[bench]
fn bench_depth_limit_reached() {
    // Configure: max_recursion_depth = 3
    // Prompt requiring depth 5
    // Measure: Graceful degradation, partial output quality
}
```

## Evidence Store Scaling

Test evidence operations at scale:

```rust
#[bench] fn bench_evidence_store_100_items() { /* baseline */ }
#[bench] fn bench_evidence_store_1k_items() { /* moderate */ }
#[bench] fn bench_evidence_store_10k_items() { /* large */ }
#[bench] fn bench_evidence_store_100k_items() { /* stress */ }

// Measure per-operation:
// - record() latency
// - get() lookup latency
// - build_bundle() latency
// - ancestors() traversal
// - Memory footprint
```

## Running Benchmarks

### Prerequisites

```bash
# Required environment variables
OPENAI_API_KEY=sk-xxx
KONTEXT_CLIENT_ID=<agent-client-id>
KONTEXT_CLIENT_SECRET=<agent-secret>
KONTEXT_GATEWAY_URL=https://gateway.kontext.dev
```

### Commands

```bash
# Run all RLM benchmarks
cargo bench -p codex-core --bench rlm

# Run specific benchmark
cargo bench -p codex-core --bench rlm_corpus

# Run with custom configuration
RLM_MAX_TOTAL_TOKENS=50000 cargo bench -p codex-core --bench rlm_e2e

# Run unit tests for benchmark utilities
cargo test -p codex-core --lib rlm::
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for main LLM calls |
| `OPENAI_SUBLM_MODEL` | `gpt-4o-mini` | Model for sub-LM calls |
| `RLM_MAX_TOTAL_TOKENS` | 500000 | Token budget for RLM mode |
| `RLM_MAX_RECURSION_DEPTH` | 5 | Max sub-LM nesting depth |
| `RLM_PER_CALL_LIMIT` | 50000 | Per sub-LM call token limit |

## Interpreting Results

### Output Format

```
══════════════════════════════════════════════════════════════════
Prompt: fan_out
   "List all projects, then summarize the issues in each one."
══════════════════════════════════════════════════════════════════

   Running baseline...
     Summary: 8 turns, 95,000 tokens, 6 tool calls, 0 evidence

   Running code mode...
     Summary: 3 turns, 2,800 tokens, 1 execute call, 0 evidence

   Running RLM mode...
     Summary: 4 phases, 28,300 tokens, 5 sub-LM calls, 6 evidence

   Comparison:
     Baseline vs Code Mode: 97% token savings
     Baseline vs RLM: 70% token savings
     Code Mode vs RLM: RLM uses 10x more tokens but with LLM reasoning
```

### Results Table

```
┌─────────────────┬──────────┬─────────┬───────┬──────────┬──────────┐
│ Prompt          │   Mode   │ tokens  │ turns │ sub-LMs  │ quality  │
├─────────────────┼──────────┼─────────┼───────┼──────────┼──────────┤
│ simple          │ baseline │  62,000 │     4 │        0 │   100%   │
│                 │ codemode │   2,500 │     3 │        0 │   100%   │
│                 │ rlm      │   3,000 │     4 │        0 │   100%   │
├─────────────────┼──────────┼─────────┼───────┼──────────┼──────────┤
│ fan_out         │ baseline │  95,000 │     8 │        0 │   100%   │
│                 │ codemode │   2,800 │     3 │        0 │    40%   │
│                 │ rlm      │  28,300 │     4 │        5 │   100%   │
├─────────────────┼──────────┼─────────┼───────┼──────────┼──────────┤
│ cross_entity    │ baseline │ 150,000+│    15 │        0 │   FAIL   │
│                 │ codemode │   3,500 │     3 │        0 │    25%   │
│                 │ rlm      │  95,000 │     6 │       11 │   100%   │
└─────────────────┴──────────┴─────────┴───────┴──────────┴──────────┘
```

## When to Use Each Mode

| Use Case | Recommended Mode | Rationale |
|----------|------------------|-----------|
| Quick counts, simple aggregations | Code Mode | Minimal tokens, fast |
| Per-entity analysis, fan-out patterns | RLM | Needs LLM reasoning |
| Cross-entity reasoning | RLM | Only viable option |
| Audit trails, compliance | RLM | Provenance required |
| Real-time dashboards | Code Mode | Latency critical |
| Large data (>128K) | RLM | Only viable option |

## Trade-offs Summary

| Dimension | Baseline | Code Mode | RLM |
|-----------|----------|-----------|-----|
| **Token Usage** | Highest | Lowest | Medium |
| **Quality** | High (until overflow) | Low-Medium | High |
| **Latency** | Medium | Lowest | Highest |
| **Provenance** | None | None | Full DAG |
| **Max Data Size** | ~128K | Unlimited* | Unlimited |
| **LLM Reasoning** | Full | None | Per sub-LM |

*Code Mode can handle unlimited data but with lossy summarization

## Key Insights

1. **Simple tasks**: Code Mode wins (minimal tokens, fast, quality sufficient)
2. **Moderate tasks with fan-out**: RLM adds value (Code Mode loses quality)
3. **Complex cross-entity tasks**: RLM is the only option that works
4. **Exhaustive analysis (>context window)**: RLM with graceful budget handling
5. **Audit requirements**: RLM provides full provenance DAG
