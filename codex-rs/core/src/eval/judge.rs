//! GPT-4o Claim Judge
//!
//! Verifies claims against agent answers using GPT-4o as the judge.

use anyhow::Context;
use anyhow::Result;
use async_openai::config::OpenAIConfig;
use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionRequestSystemMessageArgs;
use async_openai::types::ChatCompletionRequestUserMessageArgs;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::Client;

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

/// GPT-4o-based claim judge
pub struct ClaimJudge {
    client: Client<OpenAIConfig>,
    model: String,
}

impl ClaimJudge {
    /// Create a new claim judge
    ///
    /// Uses OPENAI_API_KEY from environment
    pub fn new() -> Result<Self> {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);

        Ok(Self {
            client,
            model: "gpt-4o".to_string(),
        })
    }

    /// Create with a specific model
    pub fn with_model(model: impl Into<String>) -> Result<Self> {
        let config = OpenAIConfig::default();
        let client = Client::with_config(config);

        Ok(Self {
            client,
            model: model.into(),
        })
    }

    /// Verify claims against an agent's answer
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

        // Build the verification prompt
        let system_prompt = r#"You are an expert evaluator for AI agent responses. Your task is to verify whether claims are fulfilled by the agent's answer.

For each claim, respond with exactly one of:
- FULFILLED: The claim is completely satisfied by the answer
- PARTIALLY_FULFILLED: The claim is partially satisfied (some aspects addressed, others missing)
- NOT_FULFILLED: The claim is not satisfied by the answer

Format your response as a JSON array with one object per claim:
[
  {"claim": "...", "verdict": "FULFILLED|PARTIALLY_FULFILLED|NOT_FULFILLED", "reason": "brief explanation"},
  ...
]

Be strict but fair. A claim is FULFILLED only if the answer directly addresses it with correct information."#;

        let claims_list = claims
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!(
            r#"## Task Prompt
{}

## Agent's Answer
{}

## Claims to Verify
{}

Evaluate each claim and respond with JSON array."#,
            task_prompt, answer, claims_list
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(system_prompt)
                        .build()?,
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(user_prompt)
                        .build()?,
                ),
            ])
            .temperature(0.0)
            .build()?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .with_context(|| "Failed to call GPT-4o for claim verification")?;

        let raw_response = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        // Parse the response
        let scores = self.parse_verification_response(&raw_response, claims)?;

        // Calculate coverage
        let total_score: f64 = scores.iter().map(|(_, s)| s.score()).sum();
        let coverage = if scores.is_empty() {
            1.0
        } else {
            total_score / scores.len() as f64
        };

        let passed = coverage >= super::PASS_THRESHOLD;

        Ok(ClaimVerificationResult {
            scores,
            coverage,
            passed,
            raw_response,
        })
    }

    /// Parse the JSON response from GPT-4o
    fn parse_verification_response(
        &self,
        response: &str,
        claims: &[String],
    ) -> Result<Vec<(String, ClaimScore)>> {
        // Try to extract JSON from the response
        let json_str = extract_json_array(response);

        let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(&json_str);

        match parsed {
            Ok(verdicts) => {
                let mut scores = Vec::new();

                for (i, claim) in claims.iter().enumerate() {
                    let verdict = verdicts.get(i).and_then(|v| v.get("verdict"));
                    let score = match verdict.and_then(|v| v.as_str()) {
                        Some("FULFILLED") => ClaimScore::Fulfilled,
                        Some("PARTIALLY_FULFILLED") => ClaimScore::PartiallyFulfilled,
                        _ => ClaimScore::NotFulfilled,
                    };
                    scores.push((claim.clone(), score));
                }

                Ok(scores)
            }
            Err(_) => {
                // Fallback: try to parse verdict keywords from text
                let mut scores = Vec::new();
                let response_upper = response.to_uppercase();

                for claim in claims {
                    let score = if response_upper.contains("FULFILLED")
                        && !response_upper.contains("NOT_FULFILLED")
                        && !response_upper.contains("PARTIALLY")
                    {
                        ClaimScore::Fulfilled
                    } else if response_upper.contains("PARTIALLY") {
                        ClaimScore::PartiallyFulfilled
                    } else {
                        ClaimScore::NotFulfilled
                    };
                    scores.push((claim.clone(), score));
                }

                Ok(scores)
            }
        }
    }
}

/// Extract JSON array from a response that may contain markdown or other text
fn extract_json_array(text: &str) -> String {
    // Try to find JSON array in the text
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return text[start..=end].to_string();
            }
        }
    }

    // Return empty array as fallback
    "[]".to_string()
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
    fn test_extract_json_array() {
        let text = r#"Here is my evaluation:
[{"claim": "test", "verdict": "FULFILLED"}]
That's my analysis."#;
        let json = extract_json_array(text);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
    }
}
