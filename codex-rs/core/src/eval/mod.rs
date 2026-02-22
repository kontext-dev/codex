//! MCP-Atlas Evaluation Module
//!
//! This module provides evaluation capabilities for benchmarking LLM agent
//! performance using the MCP-Atlas dataset. It supports the tool-calling client
//! architecture with four modes:
//!
//! ## Tool-Calling Client (ToolCallingRunner)
//!
//! Uses OpenAI function calling with modes:
//! - **Baseline**: Direct EXECUTE_TOOL calls with full results in context
//! - **CodeMode**: EXECUTE_CODE with summarized results
//! - **BaselineRlm**: EXECUTE_TOOL with RLM routing for large results
//! - **CodeModeRlm**: EXECUTE_CODE with RLM routing for large results
//!
//! ## Scoring
//!
//! Claims-based scoring using GPT-4o as judge:
//! - `fulfilled` = 1.0
//! - `partially_fulfilled` = 0.5
//! - `not_fulfilled` = 0.0
//!
//! Task passes if Coverage >= 0.75

pub mod dataset;
pub mod judge;
pub mod runner;

pub use dataset::McpAtlasTask;
pub use dataset::TrajectoryStep;
pub use dataset::load_dataset;
pub use judge::ClaimJudge;
pub use judge::ClaimScore;
pub use judge::ClaimVerificationResult;
pub use runner::GatewayTool;
pub use runner::TaskResult;
pub use runner::ToolCallRecord;
pub use runner::ToolCallingMode;
pub use runner::ToolCallingRunner;

/// Default pass threshold for MCP-Atlas evaluation
pub const PASS_THRESHOLD: f64 = 0.75;
