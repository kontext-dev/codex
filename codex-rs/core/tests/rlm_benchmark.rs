#![cfg(feature = "benchmarking")]
//! RLM Benchmarks - Performance measurement for RLM components
//!
//! Run with: cargo test -p codex-core --test rlm_benchmark -- --nocapture

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tempfile::TempDir;

use codex_core::rlm::BudgetManager;
use codex_core::rlm::EvidenceItem;
use codex_core::rlm::EvidenceKind;
use codex_core::rlm::EvidenceSource;
use codex_core::rlm::EvidenceStore;
use codex_core::rlm::RlmConfig;
use codex_core::rlm::RlmController;

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

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    iterations: u32,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
    avg_time: Duration,
}

impl BenchmarkResult {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            iterations: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_time: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.iterations += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.avg_time = self.total_time / self.iterations;
    }

    fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:.3}ms | {:.3}ms | {:.3}ms |",
            self.name,
            self.iterations,
            self.min_time.as_secs_f64() * 1000.0,
            self.avg_time.as_secs_f64() * 1000.0,
            self.max_time.as_secs_f64() * 1000.0,
        )
    }
}

/// Run a benchmark with the given closure
fn bench<F>(name: &str, iterations: u32, mut f: F) -> BenchmarkResult
where
    F: FnMut(),
{
    let mut result = BenchmarkResult::new(name);

    // Warmup
    for _ in 0..3 {
        f();
    }

    // Actual benchmark
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        result.record(start.elapsed());
    }

    result
}

/// Run an async benchmark
async fn bench_async<F, Fut>(name: &str, iterations: u32, mut f: F) -> BenchmarkResult
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut result = BenchmarkResult::new(name);

    // Warmup
    for _ in 0..3 {
        f().await;
    }

    // Actual benchmark
    for _ in 0..iterations {
        let start = Instant::now();
        f().await;
        result.record(start.elapsed());
    }

    result
}

// ============================================================================
// Budget Manager Benchmarks
// ============================================================================

#[tokio::test]
async fn benchmark_budget_manager() {
    init_test_tracing();
    tracing::info!("\n## Budget Manager Benchmarks\n");
    tracing::debug!("| Operation | Iterations | Min | Avg | Max |");
    tracing::debug!("|-----------|------------|-----|-----|-----|");

    let config = RlmConfig {
        enabled: true,
        max_total_tokens: 1_000_000,
        max_recursion_depth: 10,
        ..Default::default()
    };

    // Benchmark: can_proceed check
    let manager = BudgetManager::new(&config);
    let result = bench_async("can_proceed", 1000, || async {
        let _ = manager.can_proceed(1000).await;
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    assert!(
        result.avg_time < Duration::from_millis(1),
        "can_proceed should take <1ms, took {:?}",
        result.avg_time
    );

    // Benchmark: record_usage
    let manager = BudgetManager::new(&config);
    let result = bench_async("record_usage", 1000, || async {
        manager.record_usage(100).await;
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: increment_depth
    let manager = BudgetManager::new(&config);
    let result = bench_async("increment_depth", 100, || async {
        let _ = manager.increment_depth().await;
        manager.decrement_depth().await;
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: snapshot
    let manager = BudgetManager::new(&config);
    let result = bench_async("snapshot", 1000, || async {
        let _ = manager.snapshot().await;
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: remaining_tokens
    let manager = BudgetManager::new(&config);
    let result = bench_async("remaining_tokens", 1000, || async {
        let _ = manager.remaining_tokens().await;
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());
}

// ============================================================================
// Evidence Store Benchmarks
// ============================================================================

#[tokio::test]
async fn benchmark_evidence_store() {
    init_test_tracing();
    tracing::info!("\n## Evidence Store Benchmarks\n");
    tracing::debug!("| Operation | Iterations | Min | Avg | Max |");
    tracing::debug!("|-----------|------------|-----|-----|-----|");

    // Benchmark: record (empty store)
    let result = bench("record (empty)", 1000, || {
        let mut store = EvidenceStore::new();
        let item = EvidenceItem::new(
            EvidenceKind::UserQuery,
            "test content".to_string(),
            EvidenceSource::from_search("test"),
        );
        store.record(item);
    });
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: record (populated store - 100 items)
    let mut base_store = EvidenceStore::new();
    for i in 0..100 {
        base_store.record(EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!("content {i}"),
            EvidenceSource::from_search("test"),
        ));
    }
    let result = bench("record (100 items)", 1000, || {
        let mut store = base_store.clone();
        let item = EvidenceItem::new(
            EvidenceKind::UserQuery,
            "new content".to_string(),
            EvidenceSource::from_search("test"),
        );
        store.record(item);
    });
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: record (populated store - 1000 items)
    let mut base_store = EvidenceStore::new();
    for i in 0..1000 {
        base_store.record(EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!("content {i}"),
            EvidenceSource::from_search("test"),
        ));
    }
    let result = bench("record (1000 items)", 100, || {
        let mut store = base_store.clone();
        let item = EvidenceItem::new(
            EvidenceKind::UserQuery,
            "new content".to_string(),
            EvidenceSource::from_search("test"),
        );
        store.record(item);
    });
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: build_bundle (100 items)
    let mut store = EvidenceStore::new();
    for i in 0..100 {
        store.record(EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!("content {i} with some extra text to increase token count"),
            EvidenceSource::from_search("test"),
        ));
    }
    let result = bench("build_bundle (100 items)", 100, || {
        let _ = store.build_bundle(10000);
    });
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: build_bundle (1000 items)
    let mut store = EvidenceStore::new();
    for i in 0..1000 {
        store.record(EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!("content {i} with some extra text"),
            EvidenceSource::from_search("test"),
        ));
    }
    let result = bench("build_bundle (1000 items)", 100, || {
        let _ = store.build_bundle(10000);
    });
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: export/restore
    let mut store = EvidenceStore::new();
    for i in 0..100 {
        store.record(EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!("content {i}"),
            EvidenceSource::from_search("test"),
        ));
    }
    let result = bench("export (100 items)", 100, || {
        let _ = store.export();
    });
    tracing::debug!("{}", result.to_markdown_row());

    let exported = store.export();
    let result = bench("restore (100 items)", 100, || {
        let _ = EvidenceStore::restore(exported.clone());
    });
    tracing::debug!("{}", result.to_markdown_row());
}

// ============================================================================
// Corpus Benchmarks
// ============================================================================

#[tokio::test]
async fn benchmark_corpus() {
    init_test_tracing();
    tracing::info!("\n## Corpus Benchmarks\n");
    tracing::debug!("| Operation | Iterations | Min | Avg | Max |");
    tracing::debug!("|-----------|------------|-----|-----|-----|");

    let config = RlmConfig::default();

    // Small corpus (1KB)
    let small_content = "word ".repeat(200); // ~1KB
    let temp_dir = TempDir::new().unwrap();

    // Benchmark ingest operations individually (not in a loop due to state)
    let mut ingest_results_1k = BenchmarkResult::new("ingest (1KB)");
    for i in 0..10 {
        let ctrl = RlmController::new(config.clone());
        let path = temp_dir.path().join(format!("corpus_1k_{i}"));
        let start = Instant::now();
        let _ = ctrl.initialize_with_prompt(&small_content, path).await;
        ingest_results_1k.record(start.elapsed());
    }
    tracing::debug!("{}", ingest_results_1k.to_markdown_row());

    // Medium corpus (100KB)
    let medium_content = "word ".repeat(20000); // ~100KB
    let mut ingest_results_100k = BenchmarkResult::new("ingest (100KB)");
    for i in 0..10 {
        let ctrl = RlmController::new(config.clone());
        let path = temp_dir.path().join(format!("corpus_100k_{i}"));
        let start = Instant::now();
        let _ = ctrl.initialize_with_prompt(&medium_content, path).await;
        ingest_results_100k.record(start.elapsed());
    }
    tracing::debug!("{}", ingest_results_100k.to_markdown_row());

    // Large corpus (1MB)
    let large_content = "word ".repeat(200000); // ~1MB
    let mut ingest_results_1m = BenchmarkResult::new("ingest (1MB)");
    for i in 0..5 {
        let ctrl = RlmController::new(config.clone());
        let path = temp_dir.path().join(format!("corpus_1m_{i}"));
        let start = Instant::now();
        let _ = ctrl.initialize_with_prompt(&large_content, path).await;
        ingest_results_1m.record(start.elapsed());
    }
    tracing::debug!("{}", ingest_results_1m.to_markdown_row());

    assert!(
        ingest_results_1m.avg_time < Duration::from_millis(500),
        "1MB corpus ingest should take <500ms, took {:?}",
        ingest_results_1m.avg_time
    );

    // Setup for chunk operations
    let ctrl = Arc::new(RlmController::new(config.clone()));
    let corpus_path = temp_dir.path().join("bench_corpus");
    ctrl.initialize_with_prompt(&medium_content, corpus_path)
        .await
        .unwrap();

    // Benchmark: list_chunks
    let ctrl_ref = Arc::clone(&ctrl);
    let result = bench_async("list_chunks", 100, || {
        let c = Arc::clone(&ctrl_ref);
        async move {
            let _ = c.list_chunks().await;
        }
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: get_chunk
    let chunks = ctrl.list_chunks().await.unwrap();
    if !chunks.is_empty() {
        let chunk_id = chunks[0].id.clone();
        let ctrl_ref = Arc::clone(&ctrl);
        let result = bench_async("get_chunk", 100, || {
            let c = Arc::clone(&ctrl_ref);
            let id = chunk_id.clone();
            async move {
                let _ = c.get_chunk(&id, None).await;
            }
        })
        .await;
        tracing::debug!("{}", result.to_markdown_row());
    }

    // Benchmark: search
    let ctrl_ref = Arc::clone(&ctrl);
    let result = bench_async("search", 100, || {
        let c = Arc::clone(&ctrl_ref);
        async move {
            let _ = c.search("word", None).await;
        }
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: build_evidence_bundle
    let ctrl_ref = Arc::clone(&ctrl);
    let result = bench_async("build_evidence_bundle", 100, || {
        let c = Arc::clone(&ctrl_ref);
        async move {
            let _ = c.build_evidence_bundle().await;
        }
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());

    // Benchmark: system_prompt generation
    let ctrl_ref = Arc::clone(&ctrl);
    let result = bench_async("build_system_prompt", 100, || {
        let c = Arc::clone(&ctrl_ref);
        async move {
            let _ = c.build_system_prompt().await;
        }
    })
    .await;
    tracing::debug!("{}", result.to_markdown_row());
}

// ============================================================================
// End-to-End Workflow Benchmarks
// ============================================================================

#[tokio::test]
async fn benchmark_e2e_workflow() {
    init_test_tracing();
    tracing::info!("\n## End-to-End Workflow Benchmarks\n");
    tracing::debug!("| Workflow | Corpus Size | Time | Chunks | Evidence Items |");
    tracing::debug!("|----------|-------------|------|--------|----------------|");

    let config = RlmConfig {
        enabled: true,
        max_total_tokens: 500_000,
        max_recursion_depth: 5,
        ..Default::default()
    };

    // Small workflow
    let temp_dir = TempDir::new().unwrap();
    let small_content = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let start = Instant::now();
    let ctrl = RlmController::new(config.clone());
    ctrl.initialize_with_prompt(&small_content, temp_dir.path().join("small"))
        .await
        .unwrap();
    let _ = ctrl.search("fox", None).await.unwrap();
    let chunks = ctrl.list_chunks().await.unwrap();
    if !chunks.is_empty() {
        let _ = ctrl.get_chunk(&chunks[0].id, None).await.unwrap();
    }
    let bundle = ctrl.build_evidence_bundle().await;
    let elapsed = start.elapsed();
    tracing::debug!(
        "| Small (4KB) | 4KB | {:.3}ms | {} | {} |",
        elapsed.as_secs_f64() * 1000.0,
        chunks.len(),
        bundle.items.len()
    );

    // Medium workflow
    let medium_content = "The quick brown fox jumps over the lazy dog. ".repeat(2000);
    let start = Instant::now();
    let ctrl = RlmController::new(config.clone());
    ctrl.initialize_with_prompt(&medium_content, temp_dir.path().join("medium"))
        .await
        .unwrap();
    let _results = ctrl.search("fox", None).await.unwrap();
    let chunks = ctrl.list_chunks().await.unwrap();
    for chunk in chunks.iter().take(3) {
        let _ = ctrl.get_chunk(&chunk.id, None).await.unwrap();
    }
    let bundle = ctrl.build_evidence_bundle().await;
    let elapsed = start.elapsed();
    tracing::debug!(
        "| Medium (100KB) | 100KB | {:.3}ms | {} | {} |",
        elapsed.as_secs_f64() * 1000.0,
        chunks.len(),
        bundle.items.len()
    );

    // Large workflow
    let large_content = "The quick brown fox jumps over the lazy dog. ".repeat(20000);
    let start = Instant::now();
    let ctrl = RlmController::new(config.clone());
    ctrl.initialize_with_prompt(&large_content, temp_dir.path().join("large"))
        .await
        .unwrap();
    let _results = ctrl.search("fox", None).await.unwrap();
    let chunks = ctrl.list_chunks().await.unwrap();
    for chunk in chunks.iter().take(5) {
        let _ = ctrl.get_chunk(&chunk.id, None).await.unwrap();
    }
    let bundle = ctrl.build_evidence_bundle().await;
    let elapsed = start.elapsed();
    tracing::debug!(
        "| Large (1MB) | 1MB | {:.3}ms | {} | {} |",
        elapsed.as_secs_f64() * 1000.0,
        chunks.len(),
        bundle.items.len()
    );
}

// ============================================================================
// Memory Usage Benchmarks
// ============================================================================

#[tokio::test]
async fn benchmark_memory_scaling() {
    init_test_tracing();
    tracing::info!("\n## Memory Scaling (Evidence Store)\n");
    tracing::debug!("| Items | Record Time | Bundle Time | Export Size |");
    tracing::debug!("|-------|-------------|-------------|-------------|");

    for count in [100, 500, 1000, 5000, 10000] {
        let mut store = EvidenceStore::new();
        let content = "Sample evidence content with reasonable length for testing purposes.";

        let start = Instant::now();
        for i in 0..count {
            store.record(EvidenceItem::new(
                EvidenceKind::FileRead {
                    path: format!("/test/file_{i}.rs"),
                },
                content.to_string(),
                EvidenceSource::from_file(&format!("/test/file_{i}.rs"), Some(0), Some(100)),
            ));
        }
        let record_time = start.elapsed();

        let start = Instant::now();
        let _ = store.build_bundle(100000);
        let bundle_time = start.elapsed();

        let exported = store.export();
        let export_size = exported.len();

        tracing::debug!(
            "| {} | {:.3}ms | {:.3}ms | {} items |",
            count,
            record_time.as_secs_f64() * 1000.0,
            bundle_time.as_secs_f64() * 1000.0,
            export_size
        );
    }
}

// ============================================================================
// Summary Output
// ============================================================================

#[tokio::test]
async fn print_benchmark_summary() {
    init_test_tracing();
    tracing::info!("\n# RLM Benchmark Results\n");
    tracing::info!(
        "Generated: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    tracing::info!("## System Information\n");
    tracing::info!("- Platform: {}", std::env::consts::OS);
    tracing::info!("- Architecture: {}", std::env::consts::ARCH);
    tracing::info!("- Rust version: {}", env!("CARGO_PKG_RUST_VERSION"));
}
