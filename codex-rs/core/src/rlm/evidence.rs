//! Evidence tracking and provenance management (FR-15, FR-16, FR-17).
//!
//! This module provides:
//! - `EvidenceItem`: Individual pieces of evidence with source tracking
//! - `EvidenceBundle`: Collection of evidence items within a token budget
//! - `ProvenanceGraph`: DAG tracking how evidence items relate to each other
//! - `EvidenceStore`: Central storage for all evidence in an RLM session

use std::collections::HashMap;

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

/// A single piece of evidence gathered during RLM execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// Unique identifier for this evidence item.
    pub id: Uuid,
    /// Type of evidence.
    pub kind: EvidenceKind,
    /// The actual content of the evidence.
    pub content: String,
    /// Truncated/bounded content for context window (if applicable).
    pub bounded_content: Option<String>,
    /// Source information for provenance.
    pub source: EvidenceSource,
    /// When this evidence was gathered.
    pub created_at: DateTime<Utc>,
    /// Estimated token count.
    pub token_count: i64,
}

impl EvidenceItem {
    /// Create a new evidence item.
    pub fn new(kind: EvidenceKind, content: String, source: EvidenceSource) -> Self {
        let token_count = estimate_tokens(&content);
        Self {
            id: Uuid::new_v4(),
            kind,
            content,
            bounded_content: None,
            source,
            created_at: Utc::now(),
            token_count,
        }
    }

    /// Create an evidence item with bounded content.
    pub fn with_bounded_content(mut self, bounded: String) -> Self {
        self.bounded_content = Some(bounded);
        self
    }
}

/// Types of evidence that can be gathered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvidenceKind {
    /// The original user query.
    UserQuery,
    /// Content read from a file.
    FileRead { path: String },
    /// Result from a search operation.
    SearchResult { query: String },
    /// Output from a sub-LM call.
    SubLmOutput { call_id: Uuid },
    /// Synthesized content combining multiple sources.
    Synthesis,
    /// Verification result.
    Verification { claim: String },
    /// Result from a Gateway/MCP tool call.
    ToolResult {
        /// Server name (e.g., "kontext-dev")
        server: String,
        /// Tool name (e.g., "list_issues")
        tool: String,
        /// Unique call identifier
        call_id: String,
    },
    /// Summary of a tool result (for context window, references stored content).
    ToolResultSummary {
        /// Reference to the full evidence item ID
        evidence_id: Uuid,
        /// Number of chunks the result was split into
        chunk_count: usize,
        /// Total tokens in the original result
        total_tokens: i64,
    },
}

/// Source information for an evidence item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    /// Identifier for the source (file path, search query, etc.).
    pub identifier: String,
    /// Byte offset within the source (if applicable).
    pub offset: Option<usize>,
    /// Length in bytes (if applicable).
    pub length: Option<usize>,
    /// Retrieval method used.
    pub method: String,
}

impl EvidenceSource {
    /// Create a source from a file read.
    pub fn from_file(path: &str, offset: Option<usize>, length: Option<usize>) -> Self {
        Self {
            identifier: path.to_string(),
            offset,
            length,
            method: "file_read".to_string(),
        }
    }

    /// Create a source from a search operation.
    pub fn from_search(query: &str) -> Self {
        Self {
            identifier: query.to_string(),
            offset: None,
            length: None,
            method: "search".to_string(),
        }
    }

    /// Create a source from a sub-LM call.
    pub fn from_sub_lm(call_id: Uuid) -> Self {
        Self {
            identifier: call_id.to_string(),
            offset: None,
            length: None,
            method: "sub_lm".to_string(),
        }
    }

    /// Create a source from a Gateway/MCP tool call.
    pub fn from_tool(server: &str, tool: &str, call_id: &str) -> Self {
        Self {
            identifier: format!("{}:{}:{}", server, tool, call_id),
            offset: None,
            length: None,
            method: "tool_call".to_string(),
        }
    }
}

/// A collection of evidence items with provenance (FR-7).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvidenceBundle {
    /// Selected evidence items.
    pub items: Vec<EvidenceItem>,
    /// Optional summary of the evidence.
    pub summary: Option<String>,
    /// Total tokens in the bundle.
    pub total_tokens: i64,
}

impl EvidenceBundle {
    /// Create an empty evidence bundle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item to the bundle.
    pub fn add(&mut self, item: EvidenceItem) {
        self.total_tokens += item.token_count;
        self.items.push(item);
    }

    /// Check if the bundle is within a token budget.
    pub fn within_budget(&self, max_tokens: i64) -> bool {
        self.total_tokens <= max_tokens
    }
}

/// Directed acyclic graph tracking evidence provenance (FR-16).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvenanceGraph {
    /// Nodes in the graph (evidence item IDs).
    nodes: HashMap<Uuid, ProvenanceNode>,
    /// Edges: (parent_id, child_id) pairs.
    edges: Vec<(Uuid, Uuid)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProvenanceNode {
    id: Uuid,
    kind: String,
    created_at: DateTime<Utc>,
}

impl ProvenanceGraph {
    /// Create an empty provenance graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, item: &EvidenceItem) {
        self.nodes.insert(
            item.id,
            ProvenanceNode {
                id: item.id,
                kind: format!("{:?}", item.kind),
                created_at: item.created_at,
            },
        );
    }

    /// Add an edge indicating that `child` was derived from `parent`.
    pub fn add_edge(&mut self, parent: Uuid, child: Uuid) {
        if self.nodes.contains_key(&parent) && self.nodes.contains_key(&child) {
            self.edges.push((parent, child));
        }
    }

    /// Get all ancestors of a node.
    pub fn ancestors(&self, node_id: Uuid) -> Vec<Uuid> {
        let mut result = Vec::new();
        let mut to_visit = vec![node_id];
        let mut visited = std::collections::HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for (parent, child) in &self.edges {
                if *child == current && !visited.contains(parent) {
                    result.push(*parent);
                    to_visit.push(*parent);
                }
            }
        }

        result
    }

    /// Get all descendants of a node.
    pub fn descendants(&self, node_id: Uuid) -> Vec<Uuid> {
        let mut result = Vec::new();
        let mut to_visit = vec![node_id];
        let mut visited = std::collections::HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for (parent, child) in &self.edges {
                if *parent == current && !visited.contains(child) {
                    result.push(*child);
                    to_visit.push(*child);
                }
            }
        }

        result
    }
}

/// Central storage for all evidence in an RLM session (FR-15).
#[derive(Debug, Clone, Default)]
pub struct EvidenceStore {
    /// All evidence items gathered.
    items: Vec<EvidenceItem>,
    /// Index by ID for fast lookup.
    index: HashMap<Uuid, usize>,
    /// Provenance graph.
    provenance: ProvenanceGraph,
}

impl EvidenceStore {
    /// Create an empty evidence store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new evidence item.
    pub fn record(&mut self, item: EvidenceItem) {
        let id = item.id;
        self.provenance.add_node(&item);
        self.index.insert(id, self.items.len());
        self.items.push(item);
    }

    /// Record an evidence item with provenance links to parent items.
    pub fn record_with_provenance(&mut self, item: EvidenceItem, parents: &[Uuid]) {
        let id = item.id;
        self.provenance.add_node(&item);
        for parent in parents {
            self.provenance.add_edge(*parent, id);
        }
        self.index.insert(id, self.items.len());
        self.items.push(item);
    }

    /// Get an evidence item by ID.
    pub fn get(&self, id: Uuid) -> Option<&EvidenceItem> {
        self.index.get(&id).map(|&idx| &self.items[idx])
    }

    /// Get all evidence items.
    pub fn all(&self) -> &[EvidenceItem] {
        &self.items
    }

    /// Build an evidence bundle within a token budget (FR-7).
    pub fn build_bundle(&self, max_tokens: i64) -> EvidenceBundle {
        let mut bundle = EvidenceBundle::new();
        let mut remaining = max_tokens;

        // Select most recent evidence first (reverse order)
        for item in self.items.iter().rev() {
            if item.token_count <= remaining {
                bundle.add(item.clone());
                remaining -= item.token_count;
            }
        }

        // Reverse to maintain chronological order
        bundle.items.reverse();
        bundle
    }

    /// Get the provenance graph.
    pub fn provenance(&self) -> &ProvenanceGraph {
        &self.provenance
    }

    /// Export all items for persistence (FR-17).
    pub fn export(&self) -> Vec<EvidenceItem> {
        self.items.clone()
    }

    /// Restore from exported items (FR-17).
    pub fn restore(items: Vec<EvidenceItem>) -> Self {
        let mut store = Self::new();
        for item in items {
            store.record(item);
        }
        store
    }
}

/// Simple token estimation based on character count.
/// Approximation: ~4 characters per token for English text.
fn estimate_tokens(text: &str) -> i64 {
    (text.len() / 4).max(1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_item_creation() {
        let item = EvidenceItem::new(
            EvidenceKind::FileRead {
                path: "/test/file.rs".to_string(),
            },
            "fn main() {}".to_string(),
            EvidenceSource::from_file("/test/file.rs", Some(0), Some(12)),
        );

        assert!(!item.id.is_nil());
        assert_eq!(item.content, "fn main() {}");
        assert!(item.token_count > 0);
    }

    #[test]
    fn test_evidence_bundle_budget() {
        let mut bundle = EvidenceBundle::new();

        let item1 = EvidenceItem::new(
            EvidenceKind::FileRead {
                path: "a.rs".to_string(),
            },
            "a".repeat(100),
            EvidenceSource::from_file("a.rs", None, None),
        );
        let item2 = EvidenceItem::new(
            EvidenceKind::FileRead {
                path: "b.rs".to_string(),
            },
            "b".repeat(100),
            EvidenceSource::from_file("b.rs", None, None),
        );

        bundle.add(item1);
        bundle.add(item2);

        assert_eq!(bundle.items.len(), 2);
        assert!(bundle.total_tokens > 0);
    }

    #[test]
    fn test_evidence_store_provenance() {
        let mut store = EvidenceStore::new();

        let parent = EvidenceItem::new(
            EvidenceKind::FileRead {
                path: "source.rs".to_string(),
            },
            "source content".to_string(),
            EvidenceSource::from_file("source.rs", None, None),
        );
        let parent_id = parent.id;
        store.record(parent);

        let child = EvidenceItem::new(
            EvidenceKind::SubLmOutput {
                call_id: Uuid::new_v4(),
            },
            "derived content".to_string(),
            EvidenceSource::from_sub_lm(Uuid::new_v4()),
        );
        let child_id = child.id;
        store.record_with_provenance(child, &[parent_id]);

        let ancestors = store.provenance().ancestors(child_id);
        assert!(ancestors.contains(&parent_id));

        let descendants = store.provenance().descendants(parent_id);
        assert!(descendants.contains(&child_id));
    }

    #[test]
    fn test_evidence_store_export_restore() {
        let mut store = EvidenceStore::new();

        let item = EvidenceItem::new(
            EvidenceKind::UserQuery,
            "test query".to_string(),
            EvidenceSource {
                identifier: "user".to_string(),
                offset: None,
                length: None,
                method: "input".to_string(),
            },
        );
        let item_id = item.id;
        store.record(item);

        let exported = store.export();
        let restored = EvidenceStore::restore(exported);

        assert!(restored.get(item_id).is_some());
        assert_eq!(restored.get(item_id).unwrap().content, "test query");
    }
}
