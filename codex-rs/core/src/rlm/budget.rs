//! RLM budget tracking and enforcement (FR-18, FR-19).
//!
//! This module provides hard limit enforcement for:
//! - Total tokens consumed across all LLM calls
//! - Maximum recursion depth for sub-LM calls
//! - Time budget for the entire RLM session
//! - Per-call token limits for tail-risk prevention

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use serde::Serialize;
use tokio::sync::RwLock;

use super::RlmConfig;

/// Result of a budget check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetCheckResult {
    /// Operation can proceed.
    Proceed,
    /// Operation can proceed but budget is getting low.
    ProceedWithWarning(String),
    /// Hard token limit reached.
    HardLimitReached,
    /// Time limit reached.
    TimeLimitReached,
    /// Maximum recursion depth reached.
    DepthLimitReached,
}

impl BudgetCheckResult {
    /// Returns true if the operation can proceed.
    pub fn can_proceed(&self) -> bool {
        matches!(self, Self::Proceed | Self::ProceedWithWarning(_))
    }
}

/// Budget state that can be persisted for replay (FR-17).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmBudget {
    /// Total tokens consumed so far.
    pub total_consumed: i64,
    /// Number of LLM calls made.
    pub call_count: u32,
    /// Current recursion depth.
    pub current_depth: u32,
    /// Elapsed time in milliseconds (for replay).
    pub elapsed_ms: u64,
}

impl Default for RlmBudget {
    fn default() -> Self {
        Self {
            total_consumed: 0,
            call_count: 0,
            current_depth: 0,
            elapsed_ms: 0,
        }
    }
}

/// Manages budget tracking and enforcement.
pub struct BudgetManager {
    /// Hard token limit.
    max_total_tokens: i64,
    /// Warning threshold (80% of hard limit).
    warning_threshold: f64,
    /// Per-call limit for tail-risk prevention.
    per_call_limit: Option<i64>,
    /// Time limit.
    max_duration: Option<Duration>,
    /// Max recursion depth.
    max_depth: u32,
    /// Mutable state.
    state: Arc<RwLock<BudgetState>>,
}

struct BudgetState {
    budget: RlmBudget,
    start_time: Instant,
    #[allow(dead_code)] // Reserved for future use
    warnings_emitted: Vec<String>,
}

impl BudgetManager {
    /// Create a new budget manager from config.
    pub fn new(config: &RlmConfig) -> Self {
        Self {
            max_total_tokens: config.max_total_tokens,
            warning_threshold: 0.8,
            per_call_limit: Some(config.per_call_token_limit),
            max_duration: if config.max_duration_sec > 0 {
                Some(Duration::from_secs(config.max_duration_sec))
            } else {
                None
            },
            max_depth: config.max_recursion_depth,
            state: Arc::new(RwLock::new(BudgetState {
                budget: RlmBudget::default(),
                start_time: Instant::now(),
                warnings_emitted: Vec::new(),
            })),
        }
    }

    /// Create a budget manager for replay with existing state.
    pub fn from_snapshot(config: &RlmConfig, snapshot: RlmBudget) -> Self {
        let mut manager = Self::new(config);
        // We can't restore the actual start time, but we can restore the budget state
        let state = Arc::get_mut(&mut manager.state).unwrap().get_mut();
        state.budget = snapshot;
        manager
    }

    /// Check if an operation with estimated token usage can proceed (FR-18).
    pub async fn can_proceed(&self, estimated_tokens: i64) -> BudgetCheckResult {
        let state = self.state.read().await;

        // Hard token limit check
        if state.budget.total_consumed + estimated_tokens > self.max_total_tokens {
            return BudgetCheckResult::HardLimitReached;
        }

        // Per-call limit check (tail-risk prevention FR-19)
        if let Some(per_call_limit) = self.per_call_limit {
            if estimated_tokens > per_call_limit {
                return BudgetCheckResult::HardLimitReached;
            }
        }

        // Time limit check
        if let Some(max_dur) = self.max_duration {
            if state.start_time.elapsed() >= max_dur {
                return BudgetCheckResult::TimeLimitReached;
            }
        }

        // Depth limit check
        if state.budget.current_depth >= self.max_depth {
            return BudgetCheckResult::DepthLimitReached;
        }

        // Warning threshold check
        let usage_ratio =
            (state.budget.total_consumed + estimated_tokens) as f64 / self.max_total_tokens as f64;
        if usage_ratio >= self.warning_threshold {
            return BudgetCheckResult::ProceedWithWarning(format!(
                "Budget usage at {:.0}%",
                usage_ratio * 100.0
            ));
        }

        BudgetCheckResult::Proceed
    }

    /// Record token usage after an operation.
    pub async fn record_usage(&self, tokens: i64) {
        let mut state = self.state.write().await;
        state.budget.total_consumed += tokens;
        state.budget.call_count += 1;
        state.budget.elapsed_ms = state.start_time.elapsed().as_millis() as u64;
    }

    /// Increment recursion depth before a sub-LM call.
    pub async fn increment_depth(&self) -> Result<(), super::RlmError> {
        let mut state = self.state.write().await;
        if state.budget.current_depth >= self.max_depth {
            return Err(super::RlmError::MaxDepthReached(self.max_depth));
        }
        state.budget.current_depth += 1;
        Ok(())
    }

    /// Decrement recursion depth after a sub-LM call completes.
    pub async fn decrement_depth(&self) {
        let mut state = self.state.write().await;
        state.budget.current_depth = state.budget.current_depth.saturating_sub(1);
    }

    /// Get a snapshot of the current budget state for persistence (FR-17).
    pub async fn snapshot(&self) -> RlmBudget {
        let state = self.state.read().await;
        let mut budget = state.budget.clone();
        budget.elapsed_ms = state.start_time.elapsed().as_millis() as u64;
        budget
    }

    /// Get remaining token budget.
    pub async fn remaining_tokens(&self) -> i64 {
        let state = self.state.read().await;
        self.max_total_tokens - state.budget.total_consumed
    }

    /// Get remaining time budget.
    pub async fn remaining_time(&self) -> Option<Duration> {
        self.max_duration.map(|max| {
            let state = futures::executor::block_on(self.state.read());
            max.saturating_sub(state.start_time.elapsed())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RlmConfig {
        RlmConfig {
            enabled: true,
            max_total_tokens: 1000,
            max_duration_sec: 60,
            max_recursion_depth: 3,
            per_call_token_limit: 500,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_budget_proceed() {
        let manager = BudgetManager::new(&test_config());
        let result = manager.can_proceed(100).await;
        assert!(result.can_proceed());
        assert_eq!(result, BudgetCheckResult::Proceed);
    }

    #[tokio::test]
    async fn test_budget_warning() {
        let manager = BudgetManager::new(&test_config());
        // Consume 750 tokens (75%), then check for 100 more (85% total)
        manager.record_usage(750).await;
        let result = manager.can_proceed(100).await;
        assert!(result.can_proceed());
        assert!(matches!(result, BudgetCheckResult::ProceedWithWarning(_)));
    }

    #[tokio::test]
    async fn test_budget_hard_limit() {
        let manager = BudgetManager::new(&test_config());
        manager.record_usage(900).await;
        let result = manager.can_proceed(200).await;
        assert!(!result.can_proceed());
        assert_eq!(result, BudgetCheckResult::HardLimitReached);
    }

    #[tokio::test]
    async fn test_per_call_limit() {
        let manager = BudgetManager::new(&test_config());
        // Try to use 600 tokens in one call (exceeds 500 per-call limit)
        let result = manager.can_proceed(600).await;
        assert!(!result.can_proceed());
        assert_eq!(result, BudgetCheckResult::HardLimitReached);
    }

    #[tokio::test]
    async fn test_depth_limit() {
        let manager = BudgetManager::new(&test_config());
        // Increment depth 3 times (max is 3)
        manager.increment_depth().await.unwrap();
        manager.increment_depth().await.unwrap();
        manager.increment_depth().await.unwrap();
        // Fourth increment should fail
        let result = manager.increment_depth().await;
        assert!(result.is_err());
        // And can_proceed should return depth limit reached
        let result = manager.can_proceed(100).await;
        assert_eq!(result, BudgetCheckResult::DepthLimitReached);
    }

    #[tokio::test]
    async fn test_snapshot_restore() {
        let config = test_config();
        let manager = BudgetManager::new(&config);
        manager.record_usage(500).await;
        manager.increment_depth().await.unwrap();

        let snapshot = manager.snapshot().await;
        assert_eq!(snapshot.total_consumed, 500);
        assert_eq!(snapshot.current_depth, 1);
        assert_eq!(snapshot.call_count, 1);

        // Restore from snapshot
        let restored = BudgetManager::from_snapshot(&config, snapshot.clone());
        let restored_snapshot = restored.snapshot().await;
        assert_eq!(restored_snapshot.total_consumed, snapshot.total_consumed);
        assert_eq!(restored_snapshot.current_depth, snapshot.current_depth);
    }
}
