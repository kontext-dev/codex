//! LLM Claim Judge
//!
//! Verifies claims against agent answers using an LLM as the judge.
//! Follows MCP-Atlas methodology: per-claim evaluation with structured output.

use anyhow::Context;
use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionRequestSystemMessageArgs;
use async_openai::types::ChatCompletionRequestUserMessageArgs;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::ResponseFormat;
use async_openai::types::ResponseFormatJsonSchema;

/// Score for a single claim
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClaimScore {
    /// Claim is fully satisfied (1.0)
    Fulfilled,
    /// Claim is partially satisfied (0.5)
    PartiallyFulfilled,
    /// Claim is not satisfied (0.0)
    NotFulfilled,
}

impl ClaimScore {
    /// Convert to numeric score
    pub fn score(&self) -> f64 {
        match self {
            ClaimScore::Fulfilled => 1.0,
            ClaimScore::PartiallyFulfilled => 0.5,
            ClaimScore::NotFulfilled => 0.0,
        }
    }
}

/// Result of verifying claims against an answer
#[derive(Debug, Clone)]
pub struct ClaimVerificationResult {
    /// Per-claim scores
    pub scores: Vec<(String, ClaimScore)>,
    /// Mean coverage score
    pub coverage: f64,
    /// Whether the task passed (coverage >= 0.75)
    pub passed: bool,
    /// Raw judge response for debugging
    pub raw_response: String,
}

/// System prompt aligned with MCP-Atlas CoverageEvaluator methodology.
const JUDGE_SYSTEM_PROMPT: &str = r#"You are evaluating how well a model's response addresses a specific expert-defined claim.

SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.

NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match

INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
5. Provide a confidence level (0.0-1.0) for your assessment

Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."#;

/// JSON schema for structured judge output, matching MCP-Atlas's
/// `get_single_claim_evaluation_schema`.
fn judge_response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "claim_text": {
                "type": "string",
                "description": "The claim being evaluated"
            },
            "coverage_outcome": {
                "type": "string",
                "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"],
                "description": "Whether the claim is fulfilled, partially fulfilled, or not fulfilled"
            },
            "justification": {
                "type": "string",
                "description": "Specific justification referencing what was/wasn't covered"
            },
            "confidence_level": {
                "type": "number",
                "description": "Confidence level from 0.0 to 1.0 for the assessment"
            }
        },
        "required": ["claim_text", "coverage_outcome", "justification", "confidence_level"],
        "additionalProperties": false
    })
}

/// LLM-based claim judge (any OpenAI-compatible provider)
pub struct ClaimJudge {
    client: Client<OpenAIConfig>,
    model: String,
}

impl ClaimJudge {
    /// Create a new claim judge
    ///
    /// `base_url` and `api_key` configure the OpenAI-compatible endpoint.
    /// Both should be provided via `EVAL_API_KEY` / `EVAL_BASE_URL` env vars.
    pub fn new(base_url: Option<String>, api_key: Option<String>) -> Result<Self> {
        let mut config = OpenAIConfig::default();
        if let Some(base) = base_url {
            config = config.with_api_base(base);
        }
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        let client = Client::with_config(config);

        Ok(Self {
            client,
            model: "gpt-4o".to_string(),
        })
    }

    /// Create with a specific model
    pub fn with_model(
        model: impl Into<String>,
        base_url: Option<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let mut config = OpenAIConfig::default();
        if let Some(base) = base_url {
            config = config.with_api_base(base);
        }
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        let client = Client::with_config(config);

        Ok(Self {
            client,
            model: model.into(),
        })
    }

    /// Evaluate a single claim against the agent's answer.
    async fn evaluate_single_claim(
        &self,
        task_prompt: &str,
        answer: &str,
        claim: &str,
    ) -> Result<(ClaimScore, String)> {
        let user_prompt = format!(
            "CLAIM TO EVALUATE:\n{claim}\n\nTASK PROMPT:\n{task_prompt}\n\nMODEL RESPONSE TO ANALYZE:\n{answer}"
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(JUDGE_SYSTEM_PROMPT)
                        .build()?,
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(user_prompt)
                        .build()?,
                ),
            ])
            .temperature(0.0)
            .response_format(ResponseFormat::JsonSchema {
                json_schema: ResponseFormatJsonSchema {
                    description: Some(
                        "Evaluation of a single claim against the model response".to_string(),
                    ),
                    name: "claim_evaluation".to_string(),
                    schema: Some(judge_response_schema()),
                    strict: Some(true),
                },
            })
            .build()?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .with_context(|| {
                format!(
                    "Failed to call LLM for claim verification (model={})",
                    self.model
                )
            })?;

        let raw = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let score = parse_single_verdict(&raw);
        Ok((score, raw))
    }

    /// Verify claims against an agent's answer.
    ///
    /// Each claim is evaluated in an independent LLM call (per-claim isolation),
    /// matching MCP-Atlas methodology.
    pub async fn verify_claims(
        &self,
        task_prompt: &str,
        answer: &str,
        claims: &[String],
    ) -> Result<ClaimVerificationResult> {
        if claims.is_empty() {
            return Ok(ClaimVerificationResult {
                scores: vec![],
                coverage: 1.0,
                passed: true,
                raw_response: "No claims to verify".to_string(),
            });
        }

        // Evaluate each claim independently (concurrent)
        let futures: Vec<_> = claims
            .iter()
            .map(|claim| self.evaluate_single_claim(task_prompt, answer, claim))
            .collect();

        let results = futures::future::join_all(futures).await;

        let mut scores = Vec::new();
        let mut raw_responses = Vec::new();

        for (i, result) in results.into_iter().enumerate() {
            let claim = &claims[i];
            match result {
                Ok((score, raw)) => {
                    raw_responses.push(format!("claim {}: {}", i + 1, raw));
                    scores.push((claim.clone(), score));
                }
                Err(e) => {
                    raw_responses.push(format!("claim {}: ERROR: {}", i + 1, e));
                    scores.push((claim.clone(), ClaimScore::NotFulfilled));
                }
            }
        }

        let total_score: f64 = scores.iter().map(|(_, s)| s.score()).sum();
        let coverage = total_score / scores.len() as f64;
        let passed = coverage >= super::PASS_THRESHOLD;

        Ok(ClaimVerificationResult {
            scores,
            coverage,
            passed,
            raw_response: raw_responses.join("\n"),
        })
    }
}

/// Parse a single structured verdict from the judge response JSON.
fn parse_single_verdict(response: &str) -> ClaimScore {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(response) {
        match v.get("coverage_outcome").and_then(|o| o.as_str()) {
            Some("fulfilled") => return ClaimScore::Fulfilled,
            Some("partially_fulfilled") => return ClaimScore::PartiallyFulfilled,
            Some("not_fulfilled") => return ClaimScore::NotFulfilled,
            _ => {}
        }
    }

    // Structured output should always parse, but if it doesn't, try
    // extracting JSON from surrounding text (e.g. markdown fences).
    let json_str = extract_json_object(response);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
        match v.get("coverage_outcome").and_then(|o| o.as_str()) {
            Some("fulfilled") => return ClaimScore::Fulfilled,
            Some("partially_fulfilled") => return ClaimScore::PartiallyFulfilled,
            Some("not_fulfilled") => return ClaimScore::NotFulfilled,
            _ => {}
        }
    }

    ClaimScore::NotFulfilled
}

/// Extract a JSON object from text that may contain markdown or other wrapping.
fn extract_json_object(text: &str) -> String {
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                return text[start..=end].to_string();
            }
        }
    }
    "{}".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claim_score_values() {
        assert_eq!(ClaimScore::Fulfilled.score(), 1.0);
        assert_eq!(ClaimScore::PartiallyFulfilled.score(), 0.5);
        assert_eq!(ClaimScore::NotFulfilled.score(), 0.0);
    }

    #[test]
    fn test_parse_single_verdict_fulfilled() {
        let json = r#"{"claim_text":"test","coverage_outcome":"fulfilled","justification":"ok","confidence_level":0.95}"#;
        assert_eq!(parse_single_verdict(json), ClaimScore::Fulfilled);
    }

    #[test]
    fn test_parse_single_verdict_partial() {
        let json = r#"{"claim_text":"test","coverage_outcome":"partially_fulfilled","justification":"partial","confidence_level":0.6}"#;
        assert_eq!(parse_single_verdict(json), ClaimScore::PartiallyFulfilled);
    }

    #[test]
    fn test_parse_single_verdict_not_fulfilled() {
        let json = r#"{"claim_text":"test","coverage_outcome":"not_fulfilled","justification":"missing","confidence_level":0.9}"#;
        assert_eq!(parse_single_verdict(json), ClaimScore::NotFulfilled);
    }

    #[test]
    fn test_parse_single_verdict_malformed() {
        assert_eq!(parse_single_verdict("garbage"), ClaimScore::NotFulfilled);
    }

    #[test]
    fn test_parse_single_verdict_wrapped_in_markdown() {
        let text = "```json\n{\"claim_text\":\"x\",\"coverage_outcome\":\"fulfilled\",\"justification\":\"y\",\"confidence_level\":1.0}\n```";
        assert_eq!(parse_single_verdict(text), ClaimScore::Fulfilled);
    }

    #[test]
    fn test_extract_json_object() {
        let text = r#"Here is my evaluation: {"key": "value"} done."#;
        let json = extract_json_object(text);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn test_judge_response_schema_is_valid() {
        let schema = judge_response_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema.get("required").is_some());
    }
}
