//! Sub-LM invocation for RLM recursive decomposition (FR-11 to FR-14).
//!
//! This module provides the infrastructure for making recursive LLM calls
//! from within an RLM session, using the parent session's authentication
//! and model configuration.

use std::sync::Arc;

use uuid::Uuid;

use super::BudgetManager;
use super::RlmError;
use super::evidence::EvidenceItem;
use super::evidence::EvidenceKind;
use super::evidence::EvidenceSource;
use super::lm_handler::LmHandler;

/// Parameters for a sub-LM invocation.
#[derive(Debug, Clone)]
pub struct SubLmParams {
    /// Unique identifier for this sub-LM call.
    pub call_id: Uuid,
    /// The prompt/instruction for the sub-LM.
    pub instruction: String,
    /// Context snippet to include (bounded content from evidence).
    pub context: Option<String>,
    /// Task type for logging/auditing.
    pub task_type: SubLmTaskType,
    /// Estimated tokens for budget checking.
    pub estimated_tokens: i64,
}

impl SubLmParams {
    /// Create parameters for an extraction task.
    pub fn extraction(instruction: String, context: Option<String>) -> Self {
        let estimated = estimate_tokens(&instruction)
            + context.as_ref().map(|c| estimate_tokens(c)).unwrap_or(0);
        Self {
            call_id: Uuid::new_v4(),
            instruction,
            context,
            task_type: SubLmTaskType::Extract,
            estimated_tokens: estimated,
        }
    }

    /// Create parameters for a verification task (FR-14).
    pub fn verification(claim: String, evidence: String) -> Self {
        let estimated = estimate_tokens(&claim) + estimate_tokens(&evidence);
        Self {
            call_id: Uuid::new_v4(),
            instruction: format!(
                "Verify the following claim using the provided evidence:\n\nClaim: {claim}\n\nEvidence: {evidence}"
            ),
            context: None,
            task_type: SubLmTaskType::Verify,
            estimated_tokens: estimated,
        }
    }

    /// Create parameters for a transformation task.
    pub fn transform(instruction: String, input: String) -> Self {
        let estimated = estimate_tokens(&instruction) + estimate_tokens(&input);
        Self {
            call_id: Uuid::new_v4(),
            instruction,
            context: Some(input),
            task_type: SubLmTaskType::Transform,
            estimated_tokens: estimated,
        }
    }

    /// Create parameters for a classification task.
    pub fn classify(instruction: String, content: String) -> Self {
        let estimated = estimate_tokens(&instruction) + estimate_tokens(&content);
        Self {
            call_id: Uuid::new_v4(),
            instruction,
            context: Some(content),
            task_type: SubLmTaskType::Classify,
            estimated_tokens: estimated,
        }
    }
}

/// Type of sub-LM task for auditing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubLmTaskType {
    /// Extract specific information from content.
    Extract,
    /// Transform content in some way.
    Transform,
    /// Classify content.
    Classify,
    /// Verify a claim against evidence.
    Verify,
    /// Custom task type.
    Custom,
}

/// Result from a sub-LM invocation.
#[derive(Debug, Clone)]
pub struct SubLmResult {
    /// The call ID that produced this result.
    pub call_id: Uuid,
    /// The output text from the sub-LM.
    pub output: String,
    /// Tokens consumed by this call.
    pub tokens_used: i64,
    /// Whether the sub-LM completed successfully.
    pub success: bool,
    /// Task type for provenance.
    pub task_type: SubLmTaskType,
}

impl SubLmResult {
    /// Convert this result to an evidence item for the store.
    pub fn to_evidence_item(&self) -> EvidenceItem {
        EvidenceItem::new(
            EvidenceKind::SubLmOutput {
                call_id: self.call_id,
            },
            self.output.clone(),
            EvidenceSource::from_sub_lm(self.call_id),
        )
    }
}

/// Handles sub-LM invocations with budget tracking.
pub struct SubLmInvoker {
    budget_manager: Arc<BudgetManager>,
}

impl SubLmInvoker {
    /// Create a new sub-LM invoker with the given budget manager.
    pub fn new(budget_manager: Arc<BudgetManager>) -> Self {
        Self { budget_manager }
    }

    /// Get a reference to the budget manager.
    pub fn budget_manager(&self) -> &Arc<BudgetManager> {
        &self.budget_manager
    }

    /// Invoke a sub-LM call with budget tracking, routing through the LM Handler.
    pub async fn invoke(
        &self,
        params: SubLmParams,
        lm_handler: &LmHandler,
    ) -> Result<SubLmResult, RlmError> {
        // 1. Budget check.
        let check = self
            .budget_manager
            .can_proceed(params.estimated_tokens)
            .await;
        if !check.can_proceed() {
            return Err(RlmError::BudgetExhausted(format!(
                "Cannot proceed with estimated {} tokens: {check:?}",
                params.estimated_tokens
            )));
        }

        // 2. Increment recursion depth.
        self.budget_manager.increment_depth().await?;

        // 3. Build prompt from params.
        let prompt = if let Some(ref ctx) = params.context {
            format!("{}\n\nContext:\n{}", params.instruction, ctx)
        } else {
            params.instruction.clone()
        };

        // 4. POST to the LM Handler's /llm_query endpoint.
        let url = format!("http://127.0.0.1:{}/llm_query", lm_handler.port());
        let req_body = serde_json::json!({
            "prompt": prompt,
            "model": serde_json::Value::Null,
            "depth": 1,
        });

        let result = reqwest::Client::new()
            .post(&url)
            .json(&req_body)
            .send()
            .await;

        // 5. Decrement depth regardless of outcome.
        self.budget_manager.decrement_depth().await;

        // 6. Process result.
        let resp = result.map_err(|e| RlmError::SubLmFailed(e.to_string()))?;
        let resp_body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| RlmError::SubLmFailed(format!("Failed to parse response: {e}")))?;

        if let Some(err) = resp_body.get("error").and_then(|v| v.as_str()) {
            return Err(RlmError::SubLmFailed(err.to_string()));
        }

        let output = resp_body
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Estimate tokens used from output (budget recording is done by LmHandler).
        let tokens_used = estimate_tokens(&output) + params.estimated_tokens;

        Ok(SubLmResult {
            call_id: params.call_id,
            output,
            tokens_used,
            success: true,
            task_type: params.task_type,
        })
    }
}

/// Simple token estimation based on character count.
fn estimate_tokens(text: &str) -> i64 {
    (text.len() / 4).max(1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_lm_params_extraction() {
        let params = SubLmParams::extraction(
            "Extract the main function".to_string(),
            Some("fn main() { println!(\"Hello\"); }".to_string()),
        );

        assert!(!params.call_id.is_nil());
        assert_eq!(params.task_type, SubLmTaskType::Extract);
        assert!(params.estimated_tokens > 0);
    }

    #[test]
    fn test_sub_lm_params_verification() {
        let params = SubLmParams::verification(
            "The function returns a Result type".to_string(),
            "fn process() -> Result<(), Error> { Ok(()) }".to_string(),
        );

        assert_eq!(params.task_type, SubLmTaskType::Verify);
        assert!(params.instruction.contains("Verify"));
    }

    #[test]
    fn test_sub_lm_result_to_evidence() {
        let result = SubLmResult {
            call_id: Uuid::new_v4(),
            output: "Extracted content".to_string(),
            tokens_used: 100,
            success: true,
            task_type: SubLmTaskType::Extract,
        };

        let evidence = result.to_evidence_item();
        assert!(matches!(evidence.kind, EvidenceKind::SubLmOutput { .. }));
        assert_eq!(evidence.content, "Extracted content");
    }
}
