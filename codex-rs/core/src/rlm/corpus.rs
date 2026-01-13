//! Corpus storage layer for RLM mode.
//!
//! This module provides the infrastructure for treating large prompts as an
//! external environment that can be queried programmatically. Instead of loading
//! the entire corpus into the context window, we store it in workspace files
//! and provide bounded access through the RLM toolkit.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::RlmConfig;
use super::RlmError;

/// Default chunk size in tokens (~32KB of text).
#[allow(dead_code)]
const DEFAULT_CHUNK_SIZE_TOKENS: i64 = 8000;

/// Simple token estimation based on character count (~4 chars per token).
fn estimate_tokens(text: &str) -> i64 {
    (text.len() / 4).max(1) as i64
}

/// Truncate text to fit within a token budget.
fn truncate_to_tokens(text: &str, max_tokens: i64) -> String {
    let max_chars = (max_tokens * 4) as usize;
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...[truncated]", &text[..max_chars])
    }
}

/// Represents the corpus stored in the workspace.
pub struct RlmCorpus {
    /// Root directory for corpus files.
    root: PathBuf,
    /// Manifest describing the corpus structure.
    manifest: Arc<RwLock<CorpusManifest>>,
    /// Index for search operations.
    index: Arc<RwLock<CorpusIndex>>,
    /// Configuration for corpus operations.
    config: RlmConfig,
}

/// Metadata about the corpus structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorpusManifest {
    /// Unique identifier for this corpus.
    pub corpus_id: Uuid,
    /// Total tokens across all chunks.
    pub total_tokens: i64,
    /// Total bytes across all chunks.
    pub total_bytes: usize,
    /// Information about each chunk.
    pub chunks: Vec<ChunkInfo>,
    /// Document boundaries (if applicable).
    pub doc_boundaries: Vec<DocBoundary>,
    /// Whether the corpus has been fully ingested.
    pub ingestion_complete: bool,
}

impl CorpusManifest {
    /// Create a new manifest for a corpus.
    pub fn new() -> Self {
        Self {
            corpus_id: Uuid::new_v4(),
            ..Default::default()
        }
    }
}

/// Information about a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Unique chunk identifier.
    pub id: String,
    /// File path relative to corpus root.
    pub file: PathBuf,
    /// Byte offset within the original content.
    pub offset: usize,
    /// Length in bytes.
    pub length: usize,
    /// Estimated token count.
    pub tokens: i64,
    /// Optional summary/header for quick reference.
    pub summary: Option<String>,
    /// Source identifier for provenance (e.g., "tool:list_issues:call_123").
    #[serde(default)]
    pub source_id: Option<String>,
}

/// Document boundary marker for multi-document corpora.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocBoundary {
    /// Document identifier.
    pub doc_id: String,
    /// Starting chunk index.
    pub start_chunk: usize,
    /// Ending chunk index (exclusive).
    pub end_chunk: usize,
    /// Document title or description.
    pub title: Option<String>,
}

/// Index for corpus search operations.
#[derive(Debug, Default)]
pub struct CorpusIndex {
    /// Keyword to chunk IDs mapping.
    keyword_index: HashMap<String, Vec<String>>,
    /// Chunk ID to content cache (for small corpora).
    content_cache: HashMap<String, String>,
}

/// Summary information for a chunk (returned by list operations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSummary {
    /// Chunk identifier.
    pub id: String,
    /// Estimated token count.
    pub tokens: i64,
    /// Optional summary text.
    pub summary: Option<String>,
    /// Byte offset in original content.
    pub offset: usize,
}

/// Bounded content returned from chunk retrieval.
#[derive(Debug, Clone)]
pub struct BoundedContent {
    /// The content (possibly truncated).
    pub content: String,
    /// Actual token count.
    pub tokens: i64,
    /// Whether content was truncated.
    pub truncated: bool,
    /// Original byte offset.
    pub offset: usize,
    /// Original length in bytes.
    pub original_length: usize,
}

/// Search result from corpus search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Chunk ID containing the match.
    pub chunk_id: String,
    /// Byte offset of match within chunk.
    pub match_offset: usize,
    /// Short snippet around the match.
    pub snippet: String,
    /// Relevance score (higher is better).
    pub score: f64,
}

impl RlmCorpus {
    /// Create a new corpus in the specified directory.
    pub async fn new(root: PathBuf, config: RlmConfig) -> Result<Self, RlmError> {
        // Ensure the directory exists
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|e| RlmError::Internal(format!("Failed to create corpus directory: {}", e)))?;

        Ok(Self {
            root,
            manifest: Arc::new(RwLock::new(CorpusManifest::new())),
            index: Arc::new(RwLock::new(CorpusIndex::default())),
            config,
        })
    }

    /// Create a corpus from an existing manifest (for replay).
    pub async fn from_manifest(
        root: PathBuf,
        manifest: CorpusManifest,
        config: RlmConfig,
    ) -> Result<Self, RlmError> {
        Ok(Self {
            root,
            manifest: Arc::new(RwLock::new(manifest)),
            index: Arc::new(RwLock::new(CorpusIndex::default())),
            config,
        })
    }

    /// Ingest a large prompt, storing it as chunks in the workspace.
    pub async fn ingest_prompt(&self, prompt: &str) -> Result<CorpusManifest, RlmError> {
        let mut manifest = self.manifest.write().await;
        let mut index = self.index.write().await;

        // Calculate chunk size based on config
        let chunk_size_tokens = self.config.env_tools.max_file_read_tokens;
        let chunk_size_bytes = (chunk_size_tokens * 4) as usize;

        let total_bytes = prompt.len();
        let total_tokens = estimate_tokens(prompt);

        // Create chunks
        let mut offset = 0;
        let mut chunk_idx = 0;

        while offset < total_bytes {
            let end = (offset + chunk_size_bytes).min(total_bytes);

            // Try to break at a word boundary
            let actual_end = if end < total_bytes {
                prompt[offset..end]
                    .rfind(char::is_whitespace)
                    .map(|pos| offset + pos + 1)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_content = &prompt[offset..actual_end];
            let chunk_id = format!("c{:04}", chunk_idx);
            let chunk_file = PathBuf::from(format!("chunk_{}.txt", chunk_id));
            let chunk_path = self.root.join(&chunk_file);

            // Write chunk to file
            let mut file = tokio::fs::File::create(&chunk_path)
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to create chunk file: {}", e)))?;
            file.write_all(chunk_content.as_bytes())
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to write chunk: {}", e)))?;

            // Generate summary (first 100 chars)
            let summary = if chunk_content.len() > 100 {
                Some(format!("{}...", &chunk_content[..100]))
            } else {
                Some(chunk_content.to_string())
            };

            // Add to manifest
            let chunk_info = ChunkInfo {
                id: chunk_id.clone(),
                file: chunk_file,
                offset,
                length: actual_end - offset,
                tokens: estimate_tokens(chunk_content),
                summary,
                source_id: None, // Initial prompt has no source
            };
            manifest.chunks.push(chunk_info);

            // Build keyword index
            for word in chunk_content.split_whitespace() {
                let word_lower = word.to_lowercase();
                if word_lower.len() >= 3 {
                    index
                        .keyword_index
                        .entry(word_lower)
                        .or_default()
                        .push(chunk_id.clone());
                }
            }

            // Cache small chunks for fast access
            if chunk_content.len() < 10000 {
                index
                    .content_cache
                    .insert(chunk_id.clone(), chunk_content.to_string());
            }

            offset = actual_end;
            chunk_idx += 1;
        }

        manifest.total_bytes = total_bytes;
        manifest.total_tokens = total_tokens;
        manifest.ingestion_complete = true;

        Ok(manifest.clone())
    }

    /// Get corpus metadata without content.
    pub async fn get_metadata(&self) -> CorpusMetadata {
        let manifest = self.manifest.read().await;
        CorpusMetadata {
            corpus_id: manifest.corpus_id,
            total_tokens: manifest.total_tokens,
            total_bytes: manifest.total_bytes,
            chunk_count: manifest.chunks.len(),
            doc_count: manifest.doc_boundaries.len(),
            ingestion_complete: manifest.ingestion_complete,
        }
    }

    /// Get a chunk by ID with bounded content.
    pub async fn get_chunk(&self, chunk_id: &str) -> Result<BoundedContent, RlmError> {
        let manifest = self.manifest.read().await;
        let index = self.index.read().await;

        // Find chunk info
        let chunk_info = manifest
            .chunks
            .iter()
            .find(|c| c.id == chunk_id)
            .ok_or_else(|| RlmError::Internal(format!("Chunk not found: {}", chunk_id)))?;

        // Try cache first
        let content = if let Some(cached) = index.content_cache.get(chunk_id) {
            cached.clone()
        } else {
            // Read from file
            let chunk_path = self.root.join(&chunk_info.file);
            tokio::fs::read_to_string(&chunk_path)
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to read chunk: {}", e)))?
        };

        let max_tokens = self.config.env_tools.max_file_read_tokens;
        let truncated = estimate_tokens(&content) > max_tokens;
        let bounded_content = if truncated {
            truncate_to_tokens(&content, max_tokens)
        } else {
            content.clone()
        };

        Ok(BoundedContent {
            content: bounded_content,
            tokens: estimate_tokens(&content).min(max_tokens),
            truncated,
            offset: chunk_info.offset,
            original_length: chunk_info.length,
        })
    }

    /// Get bounded content by specifying a custom token limit.
    pub async fn get_chunk_bounded(
        &self,
        chunk_id: &str,
        max_tokens: i64,
    ) -> Result<BoundedContent, RlmError> {
        let manifest = self.manifest.read().await;
        let index = self.index.read().await;

        let chunk_info = manifest
            .chunks
            .iter()
            .find(|c| c.id == chunk_id)
            .ok_or_else(|| RlmError::Internal(format!("Chunk not found: {}", chunk_id)))?;

        let content = if let Some(cached) = index.content_cache.get(chunk_id) {
            cached.clone()
        } else {
            let chunk_path = self.root.join(&chunk_info.file);
            tokio::fs::read_to_string(&chunk_path)
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to read chunk: {}", e)))?
        };

        let truncated = estimate_tokens(&content) > max_tokens;
        let bounded_content = if truncated {
            truncate_to_tokens(&content, max_tokens)
        } else {
            content.clone()
        };

        Ok(BoundedContent {
            content: bounded_content,
            tokens: estimate_tokens(&content).min(max_tokens),
            truncated,
            offset: chunk_info.offset,
            original_length: chunk_info.length,
        })
    }

    /// List available chunks with summaries.
    pub async fn list_chunks(&self) -> Vec<ChunkSummary> {
        let manifest = self.manifest.read().await;
        manifest
            .chunks
            .iter()
            .map(|c| ChunkSummary {
                id: c.id.clone(),
                tokens: c.tokens,
                summary: c.summary.clone(),
                offset: c.offset,
            })
            .collect()
    }

    /// Search the corpus for keyword matches.
    pub async fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
        let index = self.index.read().await;
        let manifest = self.manifest.read().await;

        let query_words: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();

        // Score chunks by keyword matches
        let mut chunk_scores: HashMap<String, f64> = HashMap::new();
        for word in &query_words {
            if let Some(chunk_ids) = index.keyword_index.get(word) {
                for chunk_id in chunk_ids {
                    *chunk_scores.entry(chunk_id.clone()).or_default() += 1.0;
                }
            }
        }

        // Sort by score and take top results
        let mut results: Vec<_> = chunk_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        // Build search results with snippets
        results
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                let chunk_info = manifest.chunks.iter().find(|c| c.id == chunk_id)?;

                // Use cached content or summary for snippet
                let snippet = if let Some(cached) = index.content_cache.get(&chunk_id) {
                    // Find first occurrence of any query word
                    let mut best_pos = 0;
                    for word in &query_words {
                        if let Some(pos) = cached.to_lowercase().find(word) {
                            best_pos = pos;
                            break;
                        }
                    }
                    // Extract snippet around match
                    let start = best_pos.saturating_sub(30);
                    let end = (best_pos + 70).min(cached.len());
                    cached[start..end].to_string()
                } else {
                    chunk_info.summary.clone().unwrap_or_default()
                };

                Some(SearchResult {
                    chunk_id,
                    match_offset: 0,
                    snippet,
                    score,
                })
            })
            .collect()
    }

    /// Get the current manifest (for persistence).
    pub async fn get_manifest(&self) -> CorpusManifest {
        self.manifest.read().await.clone()
    }

    /// Get the corpus root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Append new content to the corpus (for tool results, LLM responses).
    ///
    /// This allows dynamic addition of content after initial corpus creation,
    /// useful for storing Gateway tool results in RLM mode.
    ///
    /// # Arguments
    /// * `content` - The content to append
    /// * `source_id` - Source identifier for provenance (e.g., "tool:list_issues:call_123")
    ///
    /// # Returns
    /// Vector of ChunkInfo for the newly created chunks
    pub async fn append_content(
        &self,
        content: &str,
        source_id: &str,
    ) -> Result<Vec<ChunkInfo>, RlmError> {
        let mut manifest = self.manifest.write().await;
        let mut index = self.index.write().await;

        // Calculate chunk size based on config
        let chunk_size_tokens = self.config.env_tools.max_file_read_tokens;
        let chunk_size_bytes = (chunk_size_tokens * 4) as usize;

        let content_bytes = content.len();
        let content_tokens = estimate_tokens(content);

        // Generate unique prefix for this append operation
        let append_prefix = format!("a{:04}", manifest.chunks.len());

        let mut new_chunks = Vec::new();
        let mut offset = 0;
        let mut chunk_idx = 0;

        while offset < content_bytes {
            let end = (offset + chunk_size_bytes).min(content_bytes);

            // Try to break at a word boundary
            let actual_end = if end < content_bytes {
                content[offset..end]
                    .rfind(char::is_whitespace)
                    .map(|pos| offset + pos + 1)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_content = &content[offset..actual_end];
            let chunk_id = format!("{}_{:04}", append_prefix, chunk_idx);
            let chunk_file = PathBuf::from(format!("chunk_{}.txt", chunk_id));
            let chunk_path = self.root.join(&chunk_file);

            // Write chunk to file
            let mut file = tokio::fs::File::create(&chunk_path)
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to create chunk file: {}", e)))?;
            file.write_all(chunk_content.as_bytes())
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to write chunk: {}", e)))?;

            // Generate summary (first 100 chars)
            let summary = if chunk_content.len() > 100 {
                Some(format!("{}...", &chunk_content[..100]))
            } else {
                Some(chunk_content.to_string())
            };

            // Create chunk info with source tracking
            let chunk_info = ChunkInfo {
                id: chunk_id.clone(),
                file: chunk_file,
                offset: manifest.total_bytes + offset, // Global offset
                length: actual_end - offset,
                tokens: estimate_tokens(chunk_content),
                summary,
                source_id: Some(source_id.to_string()),
            };

            // Update keyword index
            for word in chunk_content.split_whitespace() {
                let word_lower = word.to_lowercase();
                if word_lower.len() >= 3 {
                    index
                        .keyword_index
                        .entry(word_lower)
                        .or_default()
                        .push(chunk_id.clone());
                }
            }

            // Cache small chunks for fast access
            if chunk_content.len() < 10000 {
                index
                    .content_cache
                    .insert(chunk_id.clone(), chunk_content.to_string());
            }

            new_chunks.push(chunk_info.clone());
            manifest.chunks.push(chunk_info);

            offset = actual_end;
            chunk_idx += 1;
        }

        // Update manifest totals
        manifest.total_bytes += content_bytes;
        manifest.total_tokens += content_tokens;

        Ok(new_chunks)
    }

    /// Get chunks by source ID (for finding all chunks from a specific tool result).
    pub async fn get_chunks_by_source(&self, source_id: &str) -> Vec<ChunkSummary> {
        let manifest = self.manifest.read().await;
        manifest
            .chunks
            .iter()
            .filter(|c| c.source_id.as_deref() == Some(source_id))
            .map(|c| ChunkSummary {
                id: c.id.clone(),
                tokens: c.tokens,
                summary: c.summary.clone(),
                offset: c.offset,
            })
            .collect()
    }

    /// Rebuild the search index from scratch.
    ///
    /// Useful after restoring a corpus from manifest where the index wasn't persisted.
    pub async fn rebuild_index(&self) -> Result<(), RlmError> {
        let manifest = self.manifest.read().await;
        let mut index = self.index.write().await;

        // Clear existing index
        index.keyword_index.clear();
        index.content_cache.clear();

        // Rebuild from all chunks
        for chunk_info in &manifest.chunks {
            let chunk_path = self.root.join(&chunk_info.file);
            let content = tokio::fs::read_to_string(&chunk_path)
                .await
                .map_err(|e| RlmError::Internal(format!("Failed to read chunk: {}", e)))?;

            // Build keyword index
            for word in content.split_whitespace() {
                let word_lower = word.to_lowercase();
                if word_lower.len() >= 3 {
                    index
                        .keyword_index
                        .entry(word_lower)
                        .or_default()
                        .push(chunk_info.id.clone());
                }
            }

            // Cache small chunks
            if content.len() < 10000 {
                index
                    .content_cache
                    .insert(chunk_info.id.clone(), content);
            }
        }

        Ok(())
    }
}

/// Metadata about the corpus (returned without content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusMetadata {
    /// Unique corpus identifier.
    pub corpus_id: Uuid,
    /// Total tokens across all chunks.
    pub total_tokens: i64,
    /// Total bytes across all chunks.
    pub total_bytes: usize,
    /// Number of chunks.
    pub chunk_count: usize,
    /// Number of document boundaries.
    pub doc_count: usize,
    /// Whether ingestion is complete.
    pub ingestion_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_corpus(content: &str) -> (TempDir, RlmCorpus) {
        let temp_dir = TempDir::new().unwrap();
        let config = RlmConfig::default();
        let corpus = RlmCorpus::new(temp_dir.path().to_path_buf(), config)
            .await
            .unwrap();
        corpus.ingest_prompt(content).await.unwrap();
        (temp_dir, corpus)
    }

    #[tokio::test]
    async fn test_ingest_small_prompt() {
        let content = "This is a small test prompt.";
        let (_temp, corpus) = test_corpus(content).await;

        let meta = corpus.get_metadata().await;
        assert!(meta.ingestion_complete);
        assert_eq!(meta.chunk_count, 1);
        assert!(meta.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_ingest_large_prompt() {
        // Create a large prompt that will span multiple chunks
        let content = "word ".repeat(10000);
        let (_temp, corpus) = test_corpus(&content).await;

        let meta = corpus.get_metadata().await;
        assert!(meta.ingestion_complete);
        assert!(meta.chunk_count > 1);
    }

    #[tokio::test]
    async fn test_get_chunk() {
        let content = "This is a test prompt for chunk retrieval.";
        let (_temp, corpus) = test_corpus(content).await;

        let chunks = corpus.list_chunks().await;
        assert!(!chunks.is_empty());

        let chunk = corpus.get_chunk(&chunks[0].id).await.unwrap();
        assert!(!chunk.content.is_empty());
        assert!(!chunk.truncated);
    }

    #[tokio::test]
    async fn test_search() {
        let content = "The quick brown fox jumps over the lazy dog.";
        let (_temp, corpus) = test_corpus(content).await;

        let results = corpus.search("fox", 10).await;
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_list_chunks() {
        let content = "Test content for listing chunks.";
        let (_temp, corpus) = test_corpus(content).await;

        let chunks = corpus.list_chunks().await;
        assert!(!chunks.is_empty());
        assert!(chunks[0].summary.is_some());
    }

    #[tokio::test]
    async fn test_append_content() {
        let initial = "Initial corpus content.";
        let (_temp, corpus) = test_corpus(initial).await;

        let initial_meta = corpus.get_metadata().await;
        let initial_count = initial_meta.chunk_count;

        // Append tool result
        let tool_result = "This is a tool result with important data.";
        let new_chunks = corpus
            .append_content(tool_result, "tool:list_issues:call_123")
            .await
            .unwrap();

        assert!(!new_chunks.is_empty());
        assert!(new_chunks[0].source_id.is_some());
        assert_eq!(
            new_chunks[0].source_id.as_deref(),
            Some("tool:list_issues:call_123")
        );

        // Verify corpus grew
        let final_meta = corpus.get_metadata().await;
        assert!(final_meta.chunk_count > initial_count);
        assert!(final_meta.total_tokens > initial_meta.total_tokens);
    }

    #[tokio::test]
    async fn test_append_large_content() {
        let initial = "Initial.";
        let (_temp, corpus) = test_corpus(initial).await;

        // Append large content that spans multiple chunks
        let large_result = "word ".repeat(10000); // ~50KB
        let new_chunks = corpus
            .append_content(&large_result, "tool:list_all:call_456")
            .await
            .unwrap();

        // Should create multiple chunks
        assert!(new_chunks.len() > 1);

        // All chunks should have the same source_id
        for chunk in &new_chunks {
            assert_eq!(
                chunk.source_id.as_deref(),
                Some("tool:list_all:call_456")
            );
        }
    }

    #[tokio::test]
    async fn test_get_chunks_by_source() {
        let initial = "Initial content.";
        let (_temp, corpus) = test_corpus(initial).await;

        // Append from two different sources
        corpus
            .append_content("Source A content", "source_a")
            .await
            .unwrap();
        corpus
            .append_content("Source B content", "source_b")
            .await
            .unwrap();

        // Get chunks by source
        let source_a_chunks = corpus.get_chunks_by_source("source_a").await;
        let source_b_chunks = corpus.get_chunks_by_source("source_b").await;

        assert!(!source_a_chunks.is_empty());
        assert!(!source_b_chunks.is_empty());
    }

    #[tokio::test]
    async fn test_search_appended_content() {
        let initial = "Initial corpus without special keywords.";
        let (_temp, corpus) = test_corpus(initial).await;

        // Append content with unique keyword
        corpus
            .append_content("The flamingo flies gracefully.", "tool:search:call_789")
            .await
            .unwrap();

        // Search should find the appended content
        let results = corpus.search("flamingo", 10).await;
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_rebuild_index() {
        let content = "Test content for index rebuild with flamingo bird.";
        let temp_dir = TempDir::new().unwrap();
        let config = RlmConfig::default();

        // Create corpus and ingest
        let corpus = RlmCorpus::new(temp_dir.path().to_path_buf(), config.clone())
            .await
            .unwrap();
        let manifest = corpus.ingest_prompt(content).await.unwrap();

        // Create new corpus from manifest (simulating restore)
        let restored = RlmCorpus::from_manifest(
            temp_dir.path().to_path_buf(),
            manifest,
            config,
        )
        .await
        .unwrap();

        // Rebuild index
        restored.rebuild_index().await.unwrap();

        // Search should work after rebuild
        let results = restored.search("flamingo", 10).await;
        assert!(!results.is_empty());
    }
}
