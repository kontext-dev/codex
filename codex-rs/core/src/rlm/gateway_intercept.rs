//! Gateway result interception layer for RLM mode.
//!
//! This module routes Gateway (MCP) tool results to the RLM corpus,
//! allowing large results to be stored and queried without filling the context window.

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::corpus::ChunkInfo;
use super::corpus::RlmCorpus;
use super::evidence::EvidenceItem;
use super::evidence::EvidenceKind;
use super::evidence::EvidenceSource;
use super::evidence::EvidenceStore;
use super::RlmConfig;
use super::RlmError;

/// Default token threshold for routing to corpus (2000 tokens ~ 8KB).
pub const DEFAULT_CORPUS_THRESHOLD_TOKENS: i64 = 2000;

/// Result of processing a Gateway tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessedResult {
    /// Result is small enough to pass through directly.
    PassThrough {
        /// The content to include in conversation
        content: String,
    },
    /// Result was stored in corpus and should be referenced by evidence.
    StoredInCorpus {
        /// Evidence ID for the stored result
        evidence_id: Uuid,
        /// Chunk IDs where the result was stored
        chunk_ids: Vec<String>,
        /// Summary to include in conversation instead of full content
        summary: String,
        /// Total tokens in the original result
        total_tokens: i64,
    },
}

/// Routes Gateway tool results to RLM corpus based on size.
pub struct GatewayResultRouter {
    /// The RLM corpus for storing large results.
    corpus: Arc<RwLock<Option<RlmCorpus>>>,
    /// Evidence store for tracking all results.
    evidence_store: Arc<RwLock<EvidenceStore>>,
    /// Configuration.
    config: RlmConfig,
}

impl GatewayResultRouter {
    /// Create a new Gateway result router.
    pub fn new(
        corpus: Arc<RwLock<Option<RlmCorpus>>>,
        evidence_store: Arc<RwLock<EvidenceStore>>,
        config: RlmConfig,
    ) -> Self {
        Self {
            corpus,
            evidence_store,
            config,
        }
    }

    /// Get the token threshold for routing to corpus.
    fn corpus_threshold(&self) -> i64 {
        self.config
            .env_tools
            .corpus_threshold_tokens
            .unwrap_or(DEFAULT_CORPUS_THRESHOLD_TOKENS)
    }

    /// Process a Gateway tool result.
    ///
    /// Large results (> threshold tokens) are stored in the corpus and
    /// referenced via evidence. Small results pass through directly.
    pub async fn process_result(
        &self,
        call_id: &str,
        server: &str,
        tool_name: &str,
        result_content: &str,
    ) -> Result<ProcessedResult, RlmError> {
        let tokens = estimate_tokens(result_content);
        let threshold = self.corpus_threshold();

        if tokens > threshold {
            self.store_in_corpus(call_id, server, tool_name, result_content, tokens)
                .await
        } else {
            self.pass_through(call_id, server, tool_name, result_content, tokens)
                .await
        }
    }

    /// Store a large result in the corpus.
    async fn store_in_corpus(
        &self,
        call_id: &str,
        server: &str,
        tool_name: &str,
        content: &str,
        tokens: i64,
    ) -> Result<ProcessedResult, RlmError> {
        let corpus_guard = self.corpus.read().await;
        let corpus = corpus_guard.as_ref().ok_or_else(|| {
            RlmError::Internal("Corpus not initialized for Gateway result storage".to_string())
        })?;

        // Create source ID for provenance
        let source_id = format!("{}:{}:{}", server, tool_name, call_id);

        // Append to corpus
        let chunks = corpus.append_content(content, &source_id).await?;

        drop(corpus_guard);

        // Record as evidence
        let evidence = EvidenceItem::new(
            EvidenceKind::ToolResult {
                server: server.to_string(),
                tool: tool_name.to_string(),
                call_id: call_id.to_string(),
            },
            content.to_string(),
            EvidenceSource::from_tool(server, tool_name, call_id),
        );
        let evidence_id = evidence.id;

        let mut store = self.evidence_store.write().await;
        store.record(evidence);

        // Build summary for conversation with content preview
        let chunk_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();

        // Include a preview of the content so the agent knows what data is available
        let preview = if content.len() > 500 {
            format!("{}...", &content[..500])
        } else {
            content.to_string()
        };

        let summary = format!(
            "[Tool result stored: {} chunks, {} tokens]\n\nPreview:\n{}\n\n[Use rlm_search(query) to find specific data, or rlm_get_chunk(chunk_id) to retrieve: {}]",
            chunks.len(),
            tokens,
            preview,
            chunk_ids.first().map(|s| s.as_str()).unwrap_or("no chunks")
        );

        Ok(ProcessedResult::StoredInCorpus {
            evidence_id,
            chunk_ids,
            summary,
            total_tokens: tokens,
        })
    }

    /// Pass through a small result directly.
    async fn pass_through(
        &self,
        call_id: &str,
        server: &str,
        tool_name: &str,
        content: &str,
        tokens: i64,
    ) -> Result<ProcessedResult, RlmError> {
        // Record as evidence even for small results (for provenance)
        let evidence = EvidenceItem::new(
            EvidenceKind::ToolResult {
                server: server.to_string(),
                tool: tool_name.to_string(),
                call_id: call_id.to_string(),
            },
            content.to_string(),
            EvidenceSource::from_tool(server, tool_name, call_id),
        );

        let mut store = self.evidence_store.write().await;
        store.record(evidence);

        // For small results, also add a bounded content marker
        tracing::debug!(
            "Gateway result pass-through: server={}, tool={}, tokens={}",
            server,
            tool_name,
            tokens
        );

        Ok(ProcessedResult::PassThrough {
            content: content.to_string(),
        })
    }

    /// Get the chunk IDs for a specific tool result by evidence ID.
    pub async fn get_chunks_for_evidence(&self, evidence_id: Uuid) -> Vec<ChunkInfo> {
        let store = self.evidence_store.read().await;
        let Some(evidence) = store.get(evidence_id) else {
            return Vec::new();
        };

        // Extract source_id from evidence
        let source_id = match &evidence.kind {
            EvidenceKind::ToolResult {
                server,
                tool,
                call_id,
            } => format!("{}:{}:{}", server, tool, call_id),
            _ => return Vec::new(),
        };

        drop(store);

        // Get chunks by source
        let corpus_guard = self.corpus.read().await;
        let Some(corpus) = corpus_guard.as_ref() else {
            return Vec::new();
        };

        corpus
            .get_chunks_by_source(&source_id)
            .await
            .into_iter()
            .map(|summary| ChunkInfo {
                id: summary.id,
                file: std::path::PathBuf::new(), // Not needed for lookup
                offset: summary.offset,
                length: 0,
                tokens: summary.tokens,
                summary: summary.summary,
                source_id: Some(source_id.clone()),
            })
            .collect()
    }

    /// Generate an evidence summary for the system prompt.
    pub async fn generate_evidence_summary(&self) -> String {
        let store = self.evidence_store.read().await;
        let items = store.all();

        if items.is_empty() {
            return "No evidence collected yet.".to_string();
        }

        let mut summary = String::new();
        summary.push_str(&format!("Total evidence items: {}\n", items.len()));

        // Group by kind
        let mut tool_results = Vec::new();
        let mut other = Vec::new();

        for item in items {
            match &item.kind {
                EvidenceKind::ToolResult { server, tool, .. } => {
                    tool_results.push(format!(
                        "- [{}] {}:{} ({} tokens)",
                        item.id, server, tool, item.token_count
                    ));
                }
                _ => {
                    other.push(format!("- [{}] {:?} ({} tokens)", item.id, item.kind, item.token_count));
                }
            }
        }

        if !tool_results.is_empty() {
            summary.push_str("\nTool Results:\n");
            for r in tool_results.iter().take(10) {
                summary.push_str(r);
                summary.push('\n');
            }
            if tool_results.len() > 10 {
                summary.push_str(&format!("... and {} more\n", tool_results.len() - 10));
            }
        }

        if !other.is_empty() {
            summary.push_str("\nOther Evidence:\n");
            for r in other.iter().take(5) {
                summary.push_str(r);
                summary.push('\n');
            }
            if other.len() > 5 {
                summary.push_str(&format!("... and {} more\n", other.len() - 5));
            }
        }

        summary
    }

    /// Search the corpus for chunks matching a query.
    /// Returns relevant chunks with their content.
    pub async fn search_corpus(&self, query: &str, max_results: usize) -> Vec<SearchResultInfo> {
        let corpus_guard = self.corpus.read().await;
        let Some(corpus) = corpus_guard.as_ref() else {
            return Vec::new();
        };

        corpus
            .search(query, max_results)
            .await
            .into_iter()
            .map(|r| SearchResultInfo {
                chunk_id: r.chunk_id,
                score: r.score as f32,
                snippet: r.snippet,
            })
            .collect()
    }

    /// Get a specific chunk by ID.
    pub async fn get_chunk(&self, chunk_id: &str) -> Option<String> {
        let corpus_guard = self.corpus.read().await;
        let corpus = corpus_guard.as_ref()?;

        corpus
            .get_chunk(chunk_id)
            .await
            .ok()
            .map(|bc| bc.content)
    }

    /// List all chunks in the corpus with their summaries.
    pub async fn list_chunks(&self) -> Vec<ChunkSummaryInfo> {
        let corpus_guard = self.corpus.read().await;
        let Some(corpus) = corpus_guard.as_ref() else {
            return Vec::new();
        };

        corpus
            .list_chunks()
            .await
            .into_iter()
            .map(|s| ChunkSummaryInfo {
                id: s.id,
                tokens: s.tokens,
                summary: s.summary.unwrap_or_default(),
            })
            .collect()
    }
}

/// Information about a search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultInfo {
    /// Chunk ID
    pub chunk_id: String,
    /// Relevance score
    pub score: f32,
    /// Snippet around the match
    pub snippet: String,
}

/// Summary information about a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSummaryInfo {
    /// Chunk ID
    pub id: String,
    /// Token count
    pub tokens: i64,
    /// Summary
    pub summary: String,
}

/// Simple token estimation based on character count.
fn estimate_tokens(text: &str) -> i64 {
    (text.len() / 4).max(1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_router() -> (TempDir, GatewayResultRouter) {
        let temp_dir = TempDir::new().unwrap();
        let config = RlmConfig::default();

        let corpus = RlmCorpus::new(temp_dir.path().to_path_buf(), config.clone())
            .await
            .unwrap();
        // Initialize with minimal content
        corpus.ingest_prompt("Initial corpus.").await.unwrap();

        let router = GatewayResultRouter::new(
            Arc::new(RwLock::new(Some(corpus))),
            Arc::new(RwLock::new(EvidenceStore::new())),
            config,
        );

        (temp_dir, router)
    }

    #[tokio::test]
    async fn test_small_result_passes_through() {
        let (_temp, router) = setup_router().await;

        let result = router
            .process_result("call_1", "kontext-dev", "get_user", r#"{"id": 1, "name": "Test"}"#)
            .await
            .unwrap();

        match result {
            ProcessedResult::PassThrough { content } => {
                assert!(content.contains("Test"));
            }
            _ => panic!("Expected PassThrough"),
        }
    }

    #[tokio::test]
    async fn test_large_result_stored_in_corpus() {
        let (_temp, router) = setup_router().await;

        // Create a large result (> 2000 tokens ~ 8000 chars)
        let large_content = "data ".repeat(3000); // ~15KB

        let result = router
            .process_result("call_2", "kontext-dev", "list_issues", &large_content)
            .await
            .unwrap();

        match result {
            ProcessedResult::StoredInCorpus {
                evidence_id,
                chunk_ids,
                summary,
                total_tokens,
            } => {
                assert!(!evidence_id.is_nil());
                assert!(!chunk_ids.is_empty());
                assert!(summary.contains("chunks"));
                assert!(total_tokens > 2000);
            }
            _ => panic!("Expected StoredInCorpus"),
        }
    }

    #[tokio::test]
    async fn test_evidence_recorded_for_all_results() {
        let (_temp, router) = setup_router().await;

        // Process small result
        router
            .process_result("call_1", "server_a", "tool_a", "small")
            .await
            .unwrap();

        // Process large result
        let large = "x".repeat(10000);
        router
            .process_result("call_2", "server_b", "tool_b", &large)
            .await
            .unwrap();

        let store = router.evidence_store.read().await;
        assert_eq!(store.all().len(), 2);
    }

    #[tokio::test]
    async fn test_generate_evidence_summary() {
        let (_temp, router) = setup_router().await;

        router
            .process_result("call_1", "kontext-dev", "list_projects", "project data")
            .await
            .unwrap();

        let summary = router.generate_evidence_summary().await;
        assert!(summary.contains("kontext-dev"));
        assert!(summary.contains("list_projects"));
    }
}
