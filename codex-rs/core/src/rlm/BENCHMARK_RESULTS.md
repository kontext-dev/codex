# RLM Benchmark Results

Generated: 2026-01-10 16:10:50 UTC

## System Information

- Platform: macOS
- Architecture: aarch64 (Apple Silicon)
- Rust: codex-core v0.0.0

---

## Budget Manager Benchmarks

The Budget Manager tracks token usage, recursion depth, and time limits for RLM sessions.

| Operation | Iterations | Min | Avg | Max |
|-----------|------------|-----|-----|-----|
| can_proceed | 1000 | 0.000ms | 0.000ms | 0.022ms |
| record_usage | 1000 | 0.000ms | 0.000ms | 0.004ms |
| increment_depth | 100 | 0.001ms | 0.001ms | 0.003ms |
| snapshot | 1000 | 0.000ms | 0.000ms | 0.004ms |
| remaining_tokens | 1000 | 0.000ms | 0.000ms | 0.003ms |

**Key Findings:**
- All budget operations are sub-microsecond on average
- Budget checks add negligible overhead to RLM operations
- Thread-safe async operations (using `RwLock`) perform well

---

## Corpus Benchmarks

The Corpus storage layer handles ingestion, chunking, and retrieval of large prompts.

| Operation | Iterations | Min | Avg | Max |
|-----------|------------|-----|-----|-----|
| ingest (1KB) | 10 | 0.160ms | 0.193ms | 0.425ms |
| ingest (100KB) | 10 | 6.732ms | 7.660ms | 9.563ms |
| ingest (1MB) | 5 | 66.833ms | 69.553ms | 78.106ms |
| list_chunks | 100 | 0.001ms | 0.001ms | 0.019ms |
| get_chunk | 100 | 0.020ms | 0.024ms | 0.041ms |
| search | 100 | 2.901ms | 2.969ms | 3.099ms |
| build_evidence_bundle | 100 | 0.011ms | 0.011ms | 0.021ms |
| build_system_prompt | 100 | 0.001ms | 0.001ms | 0.005ms |

**Key Findings:**
- Ingestion scales roughly linearly with corpus size (~70ms/MB)
- Chunk listing and retrieval are very fast (<0.1ms)
- Keyword search takes ~3ms for 100KB corpus (acceptable for interactive use)
- Evidence bundle building is negligible overhead

**Ingestion Throughput:**
- 1KB: ~5,200 KB/s
- 100KB: ~13,000 KB/s
- 1MB: ~14,300 KB/s

---

## Evidence Store Benchmarks

The Evidence Store tracks provenance and manages evidence bundles with token budgets.

| Operation | Iterations | Min | Avg | Max |
|-----------|------------|-----|-----|-----|
| record (empty) | 1000 | 0.002ms | 0.002ms | 0.013ms |
| record (100 items) | 1000 | 0.013ms | 0.014ms | 0.031ms |
| record (1000 items) | 100 | 0.123ms | 0.127ms | 0.147ms |
| build_bundle (100 items) | 100 | 0.008ms | 0.009ms | 0.011ms |
| build_bundle (1000 items) | 100 | 0.076ms | 0.080ms | 0.097ms |
| export (100 items) | 100 | 0.006ms | 0.007ms | 0.009ms |
| restore (100 items) | 100 | 0.077ms | 0.083ms | 0.106ms |

**Key Findings:**
- Record operations scale with store size (HashMap index updates)
- Bundle building is efficient even with 1000 items (~0.08ms)
- Export/restore enables session persistence with minimal overhead

---

## Memory Scaling (Evidence Store)

Tests how the Evidence Store performs as the number of items grows.

| Items | Record Time | Bundle Time | Export Size |
|-------|-------------|-------------|-------------|
| 100 | 0.194ms | 0.013ms | 100 items |
| 500 | 1.061ms | 0.058ms | 500 items |
| 1000 | 2.071ms | 0.112ms | 1000 items |
| 5000 | 9.340ms | 0.609ms | 5000 items |
| 10000 | 18.670ms | 0.625ms | 10000 items |

**Key Findings:**
- Linear scaling for record operations (~2ms per 1000 items)
- Bundle building time is relatively constant at high item counts
- 10,000 evidence items can be handled efficiently

---

## End-to-End Workflow Benchmarks

Complete RLM workflows including ingestion, search, chunk retrieval, and evidence bundling.

| Workflow | Corpus Size | Time | Chunks | Evidence Items |
|----------|-------------|------|--------|----------------|
| Small (4KB) | 4KB | 0.564ms | 1 | 3 |
| Medium (100KB) | 100KB | 6.683ms | 3 | 3 |
| Large (1MB) | 1MB | 64.816ms | 29 | 3 |

**Key Findings:**
- Small prompts: Sub-millisecond total processing
- Medium prompts: ~7ms total (suitable for interactive use)
- Large prompts: ~65ms total (acceptable for batch processing)
- Chunk count scales with corpus size (1MB = 29 chunks @ 8K tokens each)

---

## Performance Summary

### Latency Expectations

| Corpus Size | Ingestion | Search | Get Chunk | Total Workflow |
|-------------|-----------|--------|-----------|----------------|
| 1KB | <1ms | <1ms | <1ms | <1ms |
| 100KB | ~8ms | ~3ms | <1ms | ~7ms |
| 1MB | ~70ms | ~3ms | <1ms | ~65ms |
| 10MB | ~700ms* | ~10ms* | <1ms | ~650ms* |

*Estimated based on linear scaling

### Throughput

- **Ingestion**: ~14 MB/s
- **Search**: ~33 queries/s (100KB corpus)
- **Chunk Retrieval**: ~40,000 chunks/s
- **Budget Checks**: >1,000,000 checks/s

### Memory Efficiency

- Evidence store overhead: ~2KB per 1000 items
- Chunk caching: Automatic for chunks <10KB
- Provenance graph: O(n) edges for n evidence items

---

## Recommendations

1. **For Interactive Use (< 100ms target)**:
   - Corpus size up to 1MB
   - Pre-warm corpus ingestion if possible
   - Limit evidence store to <5000 items

2. **For Batch Processing**:
   - No practical corpus size limit
   - Use budget enforcement to prevent runaway costs
   - Consider evidence pruning for long sessions

3. **For High-Throughput**:
   - Budget checks are negligible overhead
   - Chunk retrieval is the fastest operation
   - Search is the main bottleneck (consider caching results)

---

## Running Benchmarks

```bash
# Run all RLM benchmarks
cargo test -p codex-core --test rlm_benchmark -- --nocapture --test-threads=1

# Run specific benchmark
cargo test -p codex-core --test rlm_benchmark benchmark_corpus -- --nocapture
```

---

## Comparison: RLM vs Direct Context Loading

| Approach | 100KB Prompt | 1MB Prompt | 10MB Prompt |
|----------|--------------|------------|-------------|
| **Direct Load** | 100K tokens in context | Context overflow | Not possible |
| **RLM Mode** | ~7ms setup, bounded access | ~65ms setup, bounded access | ~650ms setup, bounded access |

**RLM Advantages:**
- Handles arbitrarily large prompts
- Bounded token usage per operation
- Full provenance tracking
- Budget enforcement prevents runaway costs

**Trade-offs:**
- Initial ingestion overhead (~70ms/MB)
- Search latency (~3ms per query)
- Additional complexity in workflow
