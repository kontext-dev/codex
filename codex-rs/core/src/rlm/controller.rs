//! RLM Controller for orchestrating RLM execution.
//!
//! This module provides the main controller that manages the RLM workflow:
//! - Corpus ingestion and management
//! - Evidence tracking and bundling
//! - Budget enforcement
//! - Sub-LM invocation coordination
//! - System prompt generation for RLM mode

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;
use uuid::Uuid;

use super::BudgetManager;
use super::EvidenceBundle;
use super::EvidenceItem;
use super::EvidenceKind;
use super::EvidenceSource;
use super::EvidenceStore;
use super::RlmConfig;
use super::RlmError;
use super::SubLmInvoker;
use super::corpus::CorpusManifest;
use super::corpus::CorpusMetadata;
use super::corpus::RlmCorpus;

/// RLM Controller manages the RLM execution lifecycle.
pub struct RlmController {
    /// The corpus being processed.
    corpus: Arc<RwLock<Option<RlmCorpus>>>,
    /// Evidence store for tracking provenance.
    evidence_store: Arc<RwLock<EvidenceStore>>,
    /// Budget manager for enforcing limits.
    budget_manager: Arc<BudgetManager>,
    /// Sub-LM invoker for recursive decomposition.
    sub_lm_invoker: Arc<SubLmInvoker>,
    /// Configuration.
    config: RlmConfig,
    /// Session identifier.
    session_id: Uuid,
}

/// State of the RLM session for persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RlmSessionState {
    /// Session identifier.
    pub session_id: Uuid,
    /// Corpus manifest (if any).
    pub corpus_manifest: Option<CorpusManifest>,
    /// Exported evidence items.
    pub evidence_items: Vec<EvidenceItem>,
    /// Budget snapshot.
    pub budget: super::RlmBudget,
}

impl RlmController {
    /// Create a new RLM controller.
    pub fn new(config: RlmConfig) -> Self {
        let budget_manager = Arc::new(BudgetManager::new(&config));
        Self {
            corpus: Arc::new(RwLock::new(None)),
            evidence_store: Arc::new(RwLock::new(EvidenceStore::new())),
            budget_manager: Arc::clone(&budget_manager),
            sub_lm_invoker: Arc::new(SubLmInvoker::new(budget_manager)),
            config,
            session_id: Uuid::new_v4(),
        }
    }

    /// Create a controller from a saved session state (for replay).
    pub async fn from_state(config: RlmConfig, state: RlmSessionState) -> Result<Self, RlmError> {
        let budget_manager = Arc::new(BudgetManager::from_snapshot(&config, state.budget));
        let evidence_store = EvidenceStore::restore(state.evidence_items);

        let controller = Self {
            corpus: Arc::new(RwLock::new(None)),
            evidence_store: Arc::new(RwLock::new(evidence_store)),
            budget_manager: Arc::clone(&budget_manager),
            sub_lm_invoker: Arc::new(SubLmInvoker::new(budget_manager)),
            config: config.clone(),
            session_id: state.session_id,
        };

        // Restore corpus if manifest exists
        if let Some(manifest) = state.corpus_manifest {
            // For replay, we need the corpus root to be the same
            // This is a simplification; in practice, we'd need to store the root path
            let corpus_root =
                PathBuf::from(format!("/tmp/rlm-corpus-{}", state.session_id));
            let corpus = RlmCorpus::from_manifest(corpus_root, manifest, config).await?;
            *controller.corpus.write().await = Some(corpus);
        }

        Ok(controller)
    }

    /// Initialize RLM mode by ingesting a large prompt.
    pub async fn initialize_with_prompt(
        &self,
        prompt: &str,
        corpus_root: PathBuf,
    ) -> Result<CorpusManifest, RlmError> {
        // Create corpus and ingest prompt
        let corpus = RlmCorpus::new(corpus_root, self.config.clone()).await?;
        let manifest = corpus.ingest_prompt(prompt).await?;

        // Record the original query as evidence
        let user_query_evidence = EvidenceItem::new(
            EvidenceKind::UserQuery,
            format!(
                "User query ingested as corpus with {} chunks ({} tokens total)",
                manifest.chunks.len(),
                manifest.total_tokens
            ),
            EvidenceSource {
                identifier: "user_query".to_string(),
                offset: None,
                length: Some(prompt.len()),
                method: "corpus_ingestion".to_string(),
            },
        );
        self.evidence_store.write().await.record(user_query_evidence);

        // Store corpus
        *self.corpus.write().await = Some(corpus);

        Ok(manifest)
    }

    /// Get corpus metadata.
    pub async fn get_corpus_metadata(&self) -> Option<CorpusMetadata> {
        let corpus = self.corpus.read().await;
        if let Some(ref c) = *corpus {
            Some(c.get_metadata().await)
        } else {
            None
        }
    }

    /// List available chunks.
    pub async fn list_chunks(&self) -> Result<Vec<super::corpus::ChunkSummary>, RlmError> {
        let corpus = self.corpus.read().await;
        let c = corpus
            .as_ref()
            .ok_or(RlmError::Internal("No corpus initialized".to_string()))?;
        Ok(c.list_chunks().await)
    }

    /// Get a chunk by ID with bounded content.
    pub async fn get_chunk(
        &self,
        chunk_id: &str,
        max_tokens: Option<i64>,
    ) -> Result<super::corpus::BoundedContent, RlmError> {
        let corpus = self.corpus.read().await;
        let c = corpus
            .as_ref()
            .ok_or(RlmError::Internal("No corpus initialized".to_string()))?;

        let content = if let Some(max) = max_tokens {
            c.get_chunk_bounded(chunk_id, max).await?
        } else {
            c.get_chunk(chunk_id).await?
        };

        // Record file read as evidence
        let evidence = EvidenceItem::new(
            EvidenceKind::FileRead {
                path: chunk_id.to_string(),
            },
            if content.truncated {
                format!(
                    "[Truncated to {} tokens] {}",
                    content.tokens, content.content
                )
            } else {
                content.content.clone()
            },
            EvidenceSource::from_file(chunk_id, Some(content.offset), Some(content.original_length)),
        );
        self.evidence_store.write().await.record(evidence);

        Ok(content)
    }

    /// Search the corpus.
    pub async fn search(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> Result<Vec<super::corpus::SearchResult>, RlmError> {
        let corpus = self.corpus.read().await;
        let c = corpus
            .as_ref()
            .ok_or(RlmError::Internal("No corpus initialized".to_string()))?;

        let max = max_results.unwrap_or(self.config.env_tools.max_search_results);
        let results = c.search(query, max).await;

        // Record search as evidence
        let evidence = EvidenceItem::new(
            EvidenceKind::SearchResult {
                query: query.to_string(),
            },
            format!(
                "Search for '{}' returned {} results: {:?}",
                query,
                results.len(),
                results.iter().map(|r| &r.chunk_id).collect::<Vec<_>>()
            ),
            EvidenceSource::from_search(query),
        );
        self.evidence_store.write().await.record(evidence);

        Ok(results)
    }

    /// Record evidence item.
    pub async fn record_evidence(&self, item: EvidenceItem) {
        self.evidence_store.write().await.record(item);
    }

    /// Record evidence with provenance links.
    pub async fn record_evidence_with_provenance(&self, item: EvidenceItem, parents: &[Uuid]) {
        self.evidence_store
            .write()
            .await
            .record_with_provenance(item, parents);
    }

    /// Build an evidence bundle within the configured token budget.
    pub async fn build_evidence_bundle(&self) -> EvidenceBundle {
        let store = self.evidence_store.read().await;
        store.build_bundle(self.config.max_evidence_bundle_tokens)
    }

    /// Build an evidence bundle with a custom token budget.
    pub async fn build_evidence_bundle_with_budget(&self, max_tokens: i64) -> EvidenceBundle {
        let store = self.evidence_store.read().await;
        store.build_bundle(max_tokens)
    }

    /// Check if an operation can proceed within budget.
    pub async fn can_proceed(&self, estimated_tokens: i64) -> super::BudgetCheckResult {
        self.budget_manager.can_proceed(estimated_tokens).await
    }

    /// Record token usage.
    pub async fn record_usage(&self, tokens: i64) {
        self.budget_manager.record_usage(tokens).await;
    }

    /// Get remaining token budget.
    pub async fn remaining_tokens(&self) -> i64 {
        self.budget_manager.remaining_tokens().await
    }

    /// Get the sub-LM invoker for spawning sub-tasks.
    pub fn sub_lm_invoker(&self) -> Arc<SubLmInvoker> {
        Arc::clone(&self.sub_lm_invoker)
    }

    /// Get the budget manager.
    pub fn budget_manager(&self) -> Arc<BudgetManager> {
        Arc::clone(&self.budget_manager)
    }

    /// Generate the RLM system prompt based on current state.
    pub async fn build_system_prompt(&self) -> String {
        let metadata = self.get_corpus_metadata().await;
        let remaining = self.remaining_tokens().await;
        let budget = self.budget_manager.snapshot().await;

        let corpus_section = if let Some(meta) = metadata {
            format!(
                r#"
**CORPUS SUMMARY:**
- Corpus ID: {}
- Total size: {} tokens across {} chunks
- Document boundaries: {}
- Ingestion complete: {}
"#,
                meta.corpus_id,
                meta.total_tokens,
                meta.chunk_count,
                meta.doc_count,
                meta.ingestion_complete
            )
        } else {
            "**CORPUS:** Not yet initialized. Use rlm_ingest_prompt to store large prompts.".to_string()
        };

        format!(
            r#"You are operating in RLM (Retrieval-augmented Language Model) mode.

**CRITICAL CONSTRAINTS:**
- Large prompts are stored in the workspace as a corpus, NOT in your context
- You MUST use the RLM tools to inspect the corpus
- Do NOT attempt to read the entire corpus at once
- Always start with search/metadata before reading chunks
- Maintain bounded token usage per operation

**AVAILABLE OPERATIONS:**
1. `rlm_corpus_info` - Get corpus metadata (size, structure, chunk count)
2. `rlm_list_chunks` - List available chunk IDs with summaries
3. `rlm_search` - Search corpus, returns chunk IDs with snippets (not full text)
4. `rlm_get_chunk` - Fetch a bounded chunk by ID (respects token limits)
5. `rlm_build_evidence` - Assemble citations for final answer with provenance
6. `rlm_spawn_subtask` - Delegate narrow analysis to sub-agent

**WORKFLOW:**
1. Call `rlm_corpus_info` to understand corpus structure
2. Call `rlm_search` to find relevant sections
3. Call `rlm_get_chunk` on specific chunks (bounded)
4. If chunk requires detailed analysis, use `rlm_spawn_subtask`
5. Call `rlm_build_evidence` to prepare citations
6. Provide final answer with evidence references

**BUDGET STATUS:**
- Tokens remaining: {} of {} total
- Calls made: {}
- Current recursion depth: {} (max: {})

{}
"#,
            remaining,
            self.config.max_total_tokens,
            budget.call_count,
            budget.current_depth,
            self.config.max_recursion_depth,
            corpus_section
        )
    }

    /// Export session state for persistence.
    pub async fn export_state(&self) -> RlmSessionState {
        let corpus = self.corpus.read().await;
        let corpus_manifest = if let Some(ref c) = *corpus {
            Some(c.get_manifest().await)
        } else {
            None
        };

        let evidence_items = self.evidence_store.read().await.export();
        let budget = self.budget_manager.snapshot().await;

        RlmSessionState {
            session_id: self.session_id,
            corpus_manifest,
            evidence_items,
            budget,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get the config.
    pub fn config(&self) -> &RlmConfig {
        &self.config
    }

    /// Get access to the corpus (for gateway intercept).
    pub fn corpus(&self) -> Arc<RwLock<Option<RlmCorpus>>> {
        Arc::clone(&self.corpus)
    }

    /// Get access to the evidence store (for gateway intercept).
    pub fn evidence_store(&self) -> Arc<RwLock<EvidenceStore>> {
        Arc::clone(&self.evidence_store)
    }

    /// Append content to the corpus (for tool results).
    pub async fn append_to_corpus(
        &self,
        content: &str,
        source_id: &str,
    ) -> Result<Vec<super::corpus::ChunkInfo>, RlmError> {
        let corpus = self.corpus.read().await;
        let c = corpus
            .as_ref()
            .ok_or(RlmError::Internal("No corpus initialized".to_string()))?;
        c.append_content(content, source_id).await
    }

    /// Build a system prompt for Gateway + RLM mode.
    ///
    /// This version includes evidence summary from tool results instead of
    /// just corpus metadata.
    pub async fn build_gateway_system_prompt(&self, evidence_summary: &str) -> String {
        let remaining = self.remaining_tokens().await;
        let budget = self.budget_manager.snapshot().await;
        let threshold = self.config
            .env_tools
            .corpus_threshold_tokens
            .unwrap_or(super::DEFAULT_CORPUS_THRESHOLD_TOKENS);

        format!(
            r#"You are operating in Gateway + RLM mode.

**EVIDENCE STORE:**
Tool results from the Gateway are stored as evidence, NOT in your context.
Large results (>{threshold} tokens) are chunked and searchable.

**CURRENT EVIDENCE:**
{evidence_summary}

**AVAILABLE OPERATIONS:**
1. `rlm_list_evidence` - List stored evidence items
2. `rlm_search` - Search across all evidence (tool results, chunks)
3. `rlm_get_chunk` - Retrieve bounded chunk by ID
4. `rlm_get_evidence` - Get specific evidence item (bounded)
5. `rlm_spawn_subtask` - Delegate analysis to sub-agent

**WORKFLOW:**
1. Review evidence summary above
2. Use `rlm_search` to find relevant results
3. Use `rlm_get_chunk` for detailed inspection (bounded)
4. Spawn sub-tasks for complex analysis
5. Build response with evidence citations

**DO NOT:**
- Request full tool results directly (they may be too large)
- Assume you have all data in context
- Skip evidence citations in your response

**BUDGET STATUS:**
- Tokens remaining: {remaining} of {total} total
- Calls made: {calls}
- Current recursion depth: {depth} (max: {max_depth})
"#,
            threshold = threshold,
            evidence_summary = evidence_summary,
            remaining = remaining,
            total = self.config.max_total_tokens,
            calls = budget.call_count,
            depth = budget.current_depth,
            max_depth = self.config.max_recursion_depth,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> RlmConfig {
        RlmConfig {
            enabled: true,
            max_total_tokens: 100_000,
            max_recursion_depth: 3,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_controller_creation() {
        let config = test_config();
        let controller = RlmController::new(config);
        assert!(controller.get_corpus_metadata().await.is_none());
    }

    #[tokio::test]
    async fn test_initialize_with_prompt() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "This is a test prompt for RLM initialization.";
        let manifest = controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        assert!(manifest.ingestion_complete);
        assert!(!manifest.chunks.is_empty());

        let meta = controller.get_corpus_metadata().await.unwrap();
        assert_eq!(meta.chunk_count, manifest.chunks.len());
    }

    #[tokio::test]
    async fn test_chunk_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "Test content for chunk operations in RLM mode.";
        controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let chunks = controller.list_chunks().await.unwrap();
        assert!(!chunks.is_empty());

        let content = controller.get_chunk(&chunks[0].id, None).await.unwrap();
        assert!(!content.content.is_empty());
    }

    #[tokio::test]
    async fn test_search_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "The quick brown fox jumps over the lazy dog.";
        controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let results = controller.search("fox", None).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_evidence_bundle() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "Test prompt for evidence bundling.";
        controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Initial evidence from ingestion
        let bundle = controller.build_evidence_bundle().await;
        assert!(!bundle.items.is_empty());
    }

    #[tokio::test]
    async fn test_system_prompt_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "Test prompt for system prompt generation.";
        controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let system_prompt = controller.build_system_prompt().await;
        assert!(system_prompt.contains("RLM"));
        assert!(system_prompt.contains("CRITICAL CONSTRAINTS"));
        assert!(system_prompt.contains("CORPUS SUMMARY"));
    }

    #[tokio::test]
    async fn test_state_export() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        let prompt = "Test prompt for state export.";
        controller
            .initialize_with_prompt(prompt, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let state = controller.export_state().await;
        assert!(state.corpus_manifest.is_some());
        assert!(!state.evidence_items.is_empty());
    }

    #[tokio::test]
    async fn test_gateway_system_prompt() {
        let config = test_config();
        let controller = RlmController::new(config);

        let evidence_summary = "Tool Results:\n- [id1] kontext-dev:list_projects (500 tokens)";
        let prompt = controller.build_gateway_system_prompt(evidence_summary).await;

        assert!(prompt.contains("Gateway + RLM mode"));
        assert!(prompt.contains("EVIDENCE STORE"));
        assert!(prompt.contains("kontext-dev:list_projects"));
        assert!(prompt.contains("rlm_list_evidence"));
    }

    #[tokio::test]
    async fn test_append_to_corpus() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config();
        let controller = RlmController::new(config);

        // Initialize corpus first
        controller
            .initialize_with_prompt("Initial content.", temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Append new content
        let chunks = controller
            .append_to_corpus("Tool result content", "tool:test:call_1")
            .await
            .unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks[0].source_id.is_some());
    }
}
