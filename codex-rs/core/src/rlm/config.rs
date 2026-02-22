//! RLM configuration types.

use serde::Deserialize;
use serde::Serialize;

/// Configuration for RLM mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmConfig {
    /// Whether RLM mode is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Maximum total tokens across all LLM calls (FR-18).
    #[serde(default = "default_max_total_tokens")]
    pub max_total_tokens: i64,

    /// Maximum duration in seconds (FR-18).
    #[serde(default = "default_max_duration_sec")]
    pub max_duration_sec: u64,

    /// Maximum recursion depth for sub-LM calls (FR-3).
    #[serde(default = "default_max_recursion_depth")]
    pub max_recursion_depth: u32,

    /// Per-call token limit for tail-risk prevention (FR-19).
    #[serde(default = "default_per_call_token_limit")]
    pub per_call_token_limit: i64,

    /// Whether to require evidence grounding in outputs (FR-21).
    #[serde(default = "default_require_evidence_grounding")]
    pub require_evidence_grounding: bool,

    /// Maximum tokens in the evidence bundle (FR-7).
    #[serde(default = "default_max_evidence_bundle_tokens")]
    pub max_evidence_bundle_tokens: i64,

    /// Data minimization policy (FR-20).
    #[serde(default)]
    pub data_minimization: DataMinimizationPolicy,

    /// Environment tools configuration.
    #[serde(default)]
    pub env_tools: EnvToolsConfig,
}

/// Data minimization policy levels (FR-20).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataMinimizationPolicy {
    /// No data minimization - full content access.
    None,
    /// Moderate minimization - truncate large files, limit search results.
    #[default]
    Moderate,
    /// Aggressive minimization - strict limits on all operations.
    Aggressive,
}

/// Configuration for RLM environment tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvToolsConfig {
    /// Maximum tokens when reading a file (FR-5).
    #[serde(default = "default_max_file_read_tokens")]
    pub max_file_read_tokens: i64,

    /// Maximum number of search results (FR-6).
    #[serde(default = "default_max_search_results")]
    pub max_search_results: usize,

    /// Token threshold for routing Gateway results to corpus.
    /// Results larger than this are stored in corpus instead of conversation.
    #[serde(default)]
    pub corpus_threshold_tokens: Option<i64>,
}

impl Default for EnvToolsConfig {
    fn default() -> Self {
        Self {
            max_file_read_tokens: default_max_file_read_tokens(),
            max_search_results: default_max_search_results(),
            corpus_threshold_tokens: None, // Use gateway_intercept::DEFAULT_CORPUS_THRESHOLD_TOKENS
        }
    }
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_total_tokens: default_max_total_tokens(),
            max_duration_sec: default_max_duration_sec(),
            max_recursion_depth: default_max_recursion_depth(),
            per_call_token_limit: default_per_call_token_limit(),
            require_evidence_grounding: default_require_evidence_grounding(),
            max_evidence_bundle_tokens: default_max_evidence_bundle_tokens(),
            data_minimization: DataMinimizationPolicy::default(),
            env_tools: EnvToolsConfig::default(),
        }
    }
}

// Default value functions for serde.

fn default_max_total_tokens() -> i64 {
    500_000
}

fn default_max_duration_sec() -> u64 {
    300
}

fn default_max_recursion_depth() -> u32 {
    5
}

fn default_per_call_token_limit() -> i64 {
    50_000
}

fn default_require_evidence_grounding() -> bool {
    true
}

fn default_max_evidence_bundle_tokens() -> i64 {
    10_000
}

fn default_max_file_read_tokens() -> i64 {
    8_000
}

fn default_max_search_results() -> usize {
    20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RlmConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_total_tokens, 500_000);
        assert_eq!(config.max_recursion_depth, 5);
        assert_eq!(config.data_minimization, DataMinimizationPolicy::Moderate);
    }

    #[test]
    fn test_deserialize_config() {
        let toml = r#"
            enabled = true
            max_total_tokens = 100000
            max_recursion_depth = 3
            data_minimization = "aggressive"

            [env_tools]
            max_file_read_tokens = 4000
            max_search_results = 10
        "#;

        let config: RlmConfig = toml::from_str(toml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.max_total_tokens, 100_000);
        assert_eq!(config.max_recursion_depth, 3);
        assert_eq!(config.data_minimization, DataMinimizationPolicy::Aggressive);
        assert_eq!(config.env_tools.max_file_read_tokens, 4_000);
        assert_eq!(config.env_tools.max_search_results, 10);
    }
}
