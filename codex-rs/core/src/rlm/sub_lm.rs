//! Sub-LM invocation for RLM recursive decomposition (FR-11 to FR-14).
//!
//! This module provides the infrastructure for making recursive LLM calls
//! from within an RLM session, using the parent session's authentication
//! and model configuration.

use std::sync::Arc;

use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::AuthManager;
use crate::codex::Codex;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::codex_delegate::run_codex_conversation_one_shot;
use crate::config::Config;
use crate::models_manager::manager::ModelsManager;

use super::BudgetManager;
use super::RlmError;
use super::evidence::EvidenceItem;
use super::evidence::EvidenceKind;
use super::evidence::EvidenceSource;

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
                "Verify the following claim using the provided evidence:\n\nClaim: {}\n\nEvidence: {}",
                claim, evidence
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

    /// Invoke a sub-LM call (FR-11).
    ///
    /// This uses the parent session's authentication and model configuration.
    /// Budget is tracked across the call, including incrementing/decrementing
    /// recursion depth.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn invoke(
        &self,
        params: SubLmParams,
        config: Config,
        auth_manager: Arc<AuthManager>,
        models_manager: Arc<ModelsManager>,
        parent_session: Arc<Session>,
        parent_ctx: Arc<TurnContext>,
        cancellation_token: CancellationToken,
    ) -> Result<SubLmResult, RlmError> {
        // Check budget before proceeding
        let budget_check = self.budget_manager.can_proceed(params.estimated_tokens).await;
        if !budget_check.can_proceed() {
            return Err(RlmError::BudgetExhausted(format!("{:?}", budget_check)));
        }

        // Increment depth before the call
        self.budget_manager.increment_depth().await?;

        // Build the input for the sub-LM
        let input = self.build_input(&params);

        // Make the sub-LM call using the delegate infrastructure
        let result = self
            .execute_sub_lm(
                params.call_id,
                params.task_type,
                input,
                config,
                auth_manager,
                models_manager,
                parent_session,
                parent_ctx,
                cancellation_token,
            )
            .await;

        // Decrement depth after the call
        self.budget_manager.decrement_depth().await;

        // Record token usage
        if let Ok(ref res) = result {
            self.budget_manager.record_usage(res.tokens_used).await;
        }

        result
    }

    /// Build the user input for the sub-LM.
    fn build_input(&self, params: &SubLmParams) -> Vec<UserInput> {
        let mut prompt = params.instruction.clone();
        if let Some(context) = &params.context {
            prompt = format!("{}\n\n---\n\n{}", prompt, context);
        }

        vec![UserInput::Text { text: prompt }]
    }

    /// Execute the sub-LM call using the delegate infrastructure.
    #[allow(clippy::too_many_arguments)]
    async fn execute_sub_lm(
        &self,
        call_id: Uuid,
        task_type: SubLmTaskType,
        input: Vec<UserInput>,
        config: Config,
        auth_manager: Arc<AuthManager>,
        models_manager: Arc<ModelsManager>,
        parent_session: Arc<Session>,
        parent_ctx: Arc<TurnContext>,
        cancellation_token: CancellationToken,
    ) -> Result<SubLmResult, RlmError> {
        // Use the existing delegate infrastructure
        let codex = run_codex_conversation_one_shot(
            config,
            auth_manager,
            models_manager,
            input,
            parent_session,
            parent_ctx,
            cancellation_token,
            None,
        )
        .await
        .map_err(|e| RlmError::SubLmFailed(e.to_string()))?;

        // Collect the output from the sub-LM
        let (output, tokens_used, success) = self.collect_output(codex).await;

        Ok(SubLmResult {
            call_id,
            output,
            tokens_used,
            success,
            task_type,
        })
    }

    /// Collect output from the sub-LM conversation.
    async fn collect_output(&self, codex: Codex) -> (String, i64, bool) {
        let mut output = String::new();
        let mut tokens_used: i64 = 0;
        let mut success = false;

        loop {
            match codex.next_event().await {
                Ok(event) => match event.msg {
                    EventMsg::AgentMessage(msg) => {
                        output.push_str(&msg.message);
                    }
                    EventMsg::AgentMessageContentDelta(delta) => {
                        output.push_str(&delta.delta);
                    }
                    EventMsg::TokenCount(tc) => {
                        if let Some(info) = tc.info {
                            tokens_used = info.total_token_usage.total_tokens;
                        }
                    }
                    EventMsg::TaskComplete(_) => {
                        success = true;
                        break;
                    }
                    EventMsg::TurnAborted(_) => {
                        break;
                    }
                    _ => {}
                },
                Err(_) => break,
            }
        }

        // Estimate tokens if not reported
        if tokens_used == 0 {
            tokens_used = estimate_tokens(&output);
        }

        (output, tokens_used, success)
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
