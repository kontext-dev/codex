//! MCP-Atlas Evaluation Module
//!
//! This module provides evaluation capabilities for benchmarking LLM agent
//! performance using the MCP-Atlas dataset. It supports the tool-calling client
//! architecture with multiple modes:
//!
//! ## Tool-Calling Client (ToolCallingRunner)
//!
//! Uses LLM function calling with modes:
//! - **Baseline**: Direct EXECUTE_TOOL calls with full results in context
//! - **CodeMode**: EXECUTE_CODE with summarized results
//! - **BaselineRlm**: EXECUTE_TOOL with RLM routing for large results
//! - **Rlm**: True RLM mode with Python REPL and sub-LLM calls
//! - **CodeMode+RLM**: RLM REPL with Gateway tool execution via execute_code()
//! - **RLM+Native**: RLM REPL with Python-native tool wrappers and field projection
//!
//! ## Scoring
//!
//! Claims-based scoring using LLM as judge (MCP-Atlas methodology):
//! - Each claim is evaluated in an independent LLM call (per-claim isolation)
//! - Structured JSON schema output enforces valid verdicts
//! - `fulfilled` = 1.0
//! - `partially_fulfilled` = 0.5
//! - `not_fulfilled` = 0.0
//!
//! Task passes if Coverage >= 0.75

pub mod codemode_types;
pub mod dataset;
pub mod judge;
pub mod pythonic_tools;
pub mod runner;

pub use codemode_types::generate_codemode_js_preamble;
pub use codemode_types::generate_codemode_types;
pub use dataset::McpAtlasTask;
pub use dataset::TrajectoryStep;
pub use dataset::load_dataset;
pub use judge::ClaimJudge;
pub use judge::ClaimScore;
pub use judge::ClaimVerificationResult;
pub use pythonic_tools::format_tool_list_for_native_prompt;
pub use pythonic_tools::generate_pythonic_tool_wrappers;
pub use runner::GatewayTool;
pub use runner::TaskResult;
pub use runner::ToolCallRecord;
pub use runner::ToolCallingMode;
pub use runner::ToolCallingRunner;

/// Default pass threshold for MCP-Atlas evaluation
pub const PASS_THRESHOLD: f64 = 0.75;
