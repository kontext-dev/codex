//! Code block extraction and FINAL answer detection for RLM mode.
//!
//! This module provides utilities for parsing LLM responses in the RLM loop:
//! - Extracting executable code from fenced `repl` blocks
//! - Detecting FINAL answer markers (both direct and variable-based)
//! - Formatting REPL output for inclusion in subsequent LLM turns

use regex_lite::Regex;

use super::repl::ReplResult;

/// Represents a final answer extracted from an LLM response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalAnswer {
    /// A direct final answer: `FINAL(content)`.
    Direct(String),
    /// A variable-based final answer: `FINAL_VAR(name)`.
    Variable(String),
}

/// Extract all code blocks fenced with triple-backtick `repl` markers.
///
/// Matches blocks of the form:
/// ````text
/// ```repl
/// <code>
/// ```
/// ````
///
/// Returns the inner code content of each block, in order of appearance.
pub fn find_code_blocks(response: &str) -> Vec<String> {
    // Accept `repl`, `python`, `py`, and unlabeled fenced blocks to be robust
    // against provider-specific markdown formatting drift.
    let re = Regex::new(r"```(?:\s*(?:repl|python|py))?\s*\n([\s\S]*?)```").unwrap();
    re.captures_iter(response)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Detect a FINAL answer marker in the LLM response.
///
/// Checks for `FINAL_VAR(name)` first (variable lookup), then falls back to
/// `FINAL(content)` (direct answer). Returns `None` if neither is found.
pub fn find_final_answer(response: &str) -> Option<FinalAnswer> {
    // Check for FINAL_VAR(name) first — variable-based answer
    let var_re = Regex::new(r"(?m)^\s*FINAL_VAR\((.+?)\)").unwrap();
    if let Some(cap) = var_re.captures(response) {
        let name = cap.get(1).unwrap().as_str().trim().to_string();
        return Some(FinalAnswer::Variable(name));
    }

    // Check for FINAL(content) — direct answer
    // Use a greedy match to capture multi-line content within FINAL(...)
    let direct_re = Regex::new(r"(?m)^\s*FINAL\(([\s\S]*)\)\s*$").unwrap();
    if let Some(cap) = direct_re.captures(response) {
        let content = cap.get(1).unwrap().as_str().trim().to_string();
        if content.is_empty() {
            return None;
        }
        return Some(FinalAnswer::Direct(content));
    }

    None
}

/// Format REPL output for inclusion in the next LLM turn.
///
/// Combines stdout, stderr, and locals summary into a single string,
/// truncated to `max_chars` characters (default 20000).
pub fn format_repl_output(result: &ReplResult, max_chars: usize) -> String {
    let mut parts = Vec::new();

    if !result.stdout.is_empty() {
        parts.push(format!("[stdout]\n{}", result.stdout));
    }
    if !result.stderr.is_empty() {
        parts.push(format!("[stderr]\n{}", result.stderr));
    }
    if !result.locals_summary.is_empty() {
        parts.push(format!("[locals]\n{}", result.locals_summary));
    }

    let combined = if parts.is_empty() {
        "(no output)".to_string()
    } else {
        parts.join("\n\n")
    };

    if combined.len() > max_chars {
        format!(
            "{}...\n[truncated, {} chars total]",
            &combined[..max_chars],
            combined.len()
        )
    } else {
        combined
    }
}

/// Default maximum characters for REPL output formatting.
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_single_code_block() {
        let response = r#"Let me compute the answer:

```repl
x = 2 + 3
print(x)
```

The result is 5."#;

        let blocks = find_code_blocks(response);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "x = 2 + 3\nprint(x)");
    }

    #[test]
    fn test_find_multiple_code_blocks() {
        let response = r#"First step:

```repl
data = [1, 2, 3]
```

Second step:

```repl
result = sum(data)
print(result)
```
"#;

        let blocks = find_code_blocks(response);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], "data = [1, 2, 3]");
        assert_eq!(blocks[1], "result = sum(data)\nprint(result)");
    }

    #[test]
    fn test_find_no_code_blocks() {
        let response = "No code here, just text.\n\nFINAL(42)";
        let blocks = find_code_blocks(response);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_ignore_non_repl_blocks() {
        let response = r#"Here is some python:

```python
print("hello")
```

And some repl:

```repl
x = 1
```
"#;

        let blocks = find_code_blocks(response);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], r#"print("hello")"#);
        assert_eq!(blocks[1], "x = 1");
    }

    #[test]
    fn test_unlabeled_fence_is_accepted() {
        let response = r#"``` 
a = 1
print(a)
```"#;
        let blocks = find_code_blocks(response);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "a = 1\nprint(a)");
    }

    #[test]
    fn test_final_direct() {
        let response = "After analysis:\n\nFINAL(The answer is 42)";
        let result = find_final_answer(response);
        assert_eq!(
            result,
            Some(FinalAnswer::Direct("The answer is 42".to_string()))
        );
    }

    #[test]
    fn test_final_var() {
        let response = "I stored the result:\n\nFINAL_VAR(result)";
        let result = find_final_answer(response);
        assert_eq!(result, Some(FinalAnswer::Variable("result".to_string())));
    }

    #[test]
    fn test_final_var_takes_precedence() {
        let response = "FINAL_VAR(answer)\nFINAL(should not match)";
        let result = find_final_answer(response);
        assert_eq!(result, Some(FinalAnswer::Variable("answer".to_string())));
    }

    #[test]
    fn test_no_final() {
        let response = "Still thinking...\n\n```repl\nx = 1\n```";
        let result = find_final_answer(response);
        assert!(result.is_none());
    }

    #[test]
    fn test_final_empty_returns_none() {
        let response = "FINAL()";
        let result = find_final_answer(response);
        assert!(result.is_none());
    }

    #[test]
    fn test_final_whitespace_only_returns_none() {
        let response = "FINAL(   )";
        let result = find_final_answer(response);
        assert!(result.is_none());
    }

    #[test]
    fn test_final_multiline() {
        let response = "FINAL(line one\nline two\nline three)";
        let result = find_final_answer(response);
        assert_eq!(
            result,
            Some(FinalAnswer::Direct(
                "line one\nline two\nline three".to_string()
            ))
        );
    }

    #[test]
    fn test_final_with_leading_whitespace() {
        let response = "  FINAL_VAR(my_var)";
        let result = find_final_answer(response);
        assert_eq!(result, Some(FinalAnswer::Variable("my_var".to_string())));
    }

    #[test]
    fn test_format_repl_output_all_fields() {
        let result = ReplResult {
            stdout: "hello world".to_string(),
            stderr: "warning: something".to_string(),
            locals_summary: "x = 5, y = 10".to_string(),
            execution_time_ms: 42,
        };
        let formatted = format_repl_output(&result, DEFAULT_MAX_OUTPUT_CHARS);
        assert!(formatted.contains("[stdout]"));
        assert!(formatted.contains("hello world"));
        assert!(formatted.contains("[stderr]"));
        assert!(formatted.contains("warning: something"));
        assert!(formatted.contains("[locals]"));
        assert!(formatted.contains("x = 5, y = 10"));
    }

    #[test]
    fn test_format_repl_output_empty() {
        let result = ReplResult {
            stdout: String::new(),
            stderr: String::new(),
            locals_summary: String::new(),
            execution_time_ms: 0,
        };
        let formatted = format_repl_output(&result, DEFAULT_MAX_OUTPUT_CHARS);
        assert_eq!(formatted, "(no output)");
    }

    #[test]
    fn test_format_repl_output_truncation() {
        let result = ReplResult {
            stdout: "a".repeat(100),
            stderr: String::new(),
            locals_summary: String::new(),
            execution_time_ms: 0,
        };
        let formatted = format_repl_output(&result, 50);
        assert!(formatted.contains("[truncated"));
        assert!(formatted.len() < 200); // Well within bounds
    }

    #[test]
    fn test_format_repl_output_only_stdout() {
        let result = ReplResult {
            stdout: "output".to_string(),
            stderr: String::new(),
            locals_summary: String::new(),
            execution_time_ms: 10,
        };
        let formatted = format_repl_output(&result, DEFAULT_MAX_OUTPUT_CHARS);
        assert_eq!(formatted, "[stdout]\noutput");
    }
}
