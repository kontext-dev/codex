//! Retrieval-augmented Language Model (RLM) support for Codex.
//!
//! This module provides RLM capabilities that enable the agent to work with
//! arbitrarily long prompts via bounded environment access, recursive sub-LM
//! decomposition, and evidence-grounded outputs.
//!
//! # Architecture
//!
//! The RLM system consists of:
//!
//! - **Controller**: Orchestrates RLM execution, manages sub-LM calls, enforces budgets
//! - **Sub-LM Invoker**: Handles recursive LLM calls using the parent session's auth
//! - **Evidence Store**: Tracks provenance and manages evidence bundles
//! - **Budget Manager**: Enforces hard limits on tokens, time, and recursion depth
//!
//! # Feature Flag
//!
//! RLM mode is gated behind the `rlm_mode` feature flag. Enable it in config:
//!
//! ```toml
//! [features]
//! rlm_mode = true
//! ```

mod budget;
mod config;
mod controller;
pub mod corpus;
mod evidence;
mod gateway_intercept;
mod sub_lm;

pub use budget::BudgetCheckResult;
pub use budget::BudgetManager;
pub use budget::RlmBudget;
pub use config::DataMinimizationPolicy;
pub use config::EnvToolsConfig;
pub use config::RlmConfig;
pub use controller::RlmController;
pub use controller::RlmSessionState;
pub use corpus::BoundedContent;
pub use corpus::ChunkInfo;
pub use corpus::ChunkSummary;
pub use corpus::CorpusManifest;
pub use corpus::CorpusMetadata;
pub use corpus::RlmCorpus;
pub use corpus::SearchResult;
pub use evidence::EvidenceBundle;
pub use evidence::EvidenceItem;
pub use evidence::EvidenceKind;
pub use evidence::EvidenceSource;
pub use evidence::EvidenceStore;
pub use evidence::ProvenanceGraph;
pub use gateway_intercept::ChunkSummaryInfo;
pub use gateway_intercept::GatewayResultRouter;
pub use gateway_intercept::ProcessedResult;
pub use gateway_intercept::SearchResultInfo;
pub use gateway_intercept::DEFAULT_CORPUS_THRESHOLD_TOKENS;
pub use sub_lm::SubLmInvoker;
pub use sub_lm::SubLmParams;
pub use sub_lm::SubLmResult;

/// Error types for RLM operations.
#[derive(Debug, thiserror::Error)]
pub enum RlmError {
    #[error("RLM budget exhausted: {0}")]
    BudgetExhausted(String),

    #[error("Maximum recursion depth reached: {0}")]
    MaxDepthReached(u32),

    #[error("Sub-LM invocation failed: {0}")]
    SubLmFailed(String),

    #[error("Evidence store error: {0}")]
    EvidenceError(String),

    #[error("RLM mode not enabled")]
    NotEnabled,

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<anyhow::Error> for RlmError {
    fn from(err: anyhow::Error) -> Self {
        RlmError::Internal(err.to_string())
    }
}
