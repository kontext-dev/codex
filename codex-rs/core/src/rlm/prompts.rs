//! RLM system prompts for true Recursive Language Model execution.
//!
//! These prompts instruct the LLM to use a Python REPL environment
//! for programmatic context examination and recursive sub-LLM calls.

/// Metadata about the context injected into the REPL.
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    /// Type description (e.g., "str", "dict", "list")
    pub context_type: String,
    /// Total character length of the context
    pub context_total_length: usize,
    /// Per-chunk lengths (truncated display if >100)
    pub context_lengths: Vec<usize>,
}

/// The core RLM system prompt that instructs the LLM how to use the REPL.
pub const RLM_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible.

Your REPL environment is initialized with the following:
1. `context` — a variable that contains extremely important information about your query. It may be a string, dictionary, or list depending on the data.
2. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM that can handle around 500K characters. Use this to analyze chunks of the context.
3. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently for parallel analysis.
4. `SHOW_VARS()` — returns all variables you have created in the REPL.
5. `print()` statements to view the output of your REPL code.

IMPORTANT:
- You will only be able to see truncated outputs from the REPL environment, so you should use the llm_query function on variables you want to analyze in detail.
- Make sure to explicitly look through the entire context in REPL before answering your query.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.

Here are some example strategies you can use:

**Strategy 1 — Examine the context structure:**
```repl
# First, examine the context structure
print(type(context))
print(len(context))
if isinstance(context, str):
    print(context[:500])
elif isinstance(context, dict):
    print(list(context.keys()))
elif isinstance(context, list):
    print(f"List with {len(context)} items")
    print(context[0][:200] if context else "empty")
```

**Strategy 2 — Chunk and analyze with llm_query:**
```repl
# Split into manageable chunks and analyze each
chunk_size = 50000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
print(f"Split into {len(chunks)} chunks")

# Analyze first chunk
summary = llm_query(f"Summarize the following text:\n{chunks[0]}")
print(summary)
```

**Strategy 3 — Batch processing with llm_query_batched:**
```repl
# Process all chunks in parallel
prompts = [f"Extract key facts from:\n{chunk}" for chunk in chunks]
results = llm_query_batched(prompts)
combined = "\n".join(results)
print(f"Got {len(results)} chunk summaries")
```

**Strategy 4 — Aggregation and final synthesis:**
```repl
# Final aggregation
final_answer = llm_query(f"Based on these summaries, answer the original question:\n{combined}")
answer = final_answer
```

To provide your final answer, use one of these formats:
- `FINAL(your final answer here)` — for direct text answers
- `FINAL_VAR(variable_name)` — to return the value of a REPL variable

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl block FIRST, then call FINAL_VAR in a SEPARATE step.

Think step by step carefully, plan, and execute this plan immediately in your response — do not just say "I will do this". Output to the REPL environment and recursive LLMs as much as possible."#;

/// Build the complete system prompt with context metadata appended.
pub fn build_rlm_system_prompt(metadata: &ContextMetadata) -> String {
    let lengths_display = if metadata.context_lengths.len() > 100 {
        let shown: Vec<String> = metadata.context_lengths[..100]
            .iter()
            .map(|l| l.to_string())
            .collect();
        format!("[{}, ... ({} more)]", shown.join(", "), metadata.context_lengths.len() - 100)
    } else {
        let shown: Vec<String> = metadata.context_lengths.iter().map(|l| l.to_string()).collect();
        format!("[{}]", shown.join(", "))
    };

    format!(
        "{}\n\nCONTEXT METADATA:\n- Type: {}\n- Total length: {} characters\n- Chunk lengths: {}",
        RLM_SYSTEM_PROMPT,
        metadata.context_type,
        metadata.context_total_length,
        lengths_display,
    )
}

/// Build the user prompt for a specific iteration.
pub fn build_rlm_user_prompt(iteration: usize, query: &str) -> String {
    if iteration == 0 {
        format!(
            "{}\n\nIMPORTANT: Do not provide a final answer yet. \
             You haven't interacted with the REPL or examined the context. \
             Start by exploring the context variable in a ```repl block.",
            query
        )
    } else {
        "Continue your analysis. The history above contains your previous REPL \
         interactions and their outputs. If you have enough information, provide \
         your final answer using FINAL(...)."
            .to_string()
    }
}

/// Build user prompt for presenting REPL output back to the LLM.
pub fn format_repl_feedback(
    code: &str,
    stdout: &str,
    stderr: &str,
    locals: &str,
    max_chars: usize,
) -> String {
    let mut output = format!(
        "Code executed:\n```python\n{}\n```\n\nREPL Output:\n{}",
        code, stdout,
    );

    if !stderr.is_empty() {
        output.push_str(&format!("\n\nStderr:\n{}", stderr));
    }

    output.push_str(&format!("\n\nREPL Variables:\n{}", locals));

    if output.len() > max_chars {
        let omitted = output.len() - max_chars;
        output.truncate(max_chars);
        output.push_str(&format!("\n[truncated, {} chars omitted]", omitted));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rlm_system_prompt_contains_key_sections() {
        assert!(RLM_SYSTEM_PROMPT.contains("REPL environment"));
        assert!(RLM_SYSTEM_PROMPT.contains("llm_query"));
        assert!(RLM_SYSTEM_PROMPT.contains("llm_query_batched"));
        assert!(RLM_SYSTEM_PROMPT.contains("SHOW_VARS()"));
        assert!(RLM_SYSTEM_PROMPT.contains("FINAL("));
        assert!(RLM_SYSTEM_PROMPT.contains("FINAL_VAR("));
        assert!(RLM_SYSTEM_PROMPT.contains("```repl"));
        assert!(RLM_SYSTEM_PROMPT.contains("Think step by step"));
    }

    #[test]
    fn test_build_rlm_system_prompt_appends_metadata() {
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: 150_000,
            context_lengths: vec![50_000, 50_000, 50_000],
        };

        let prompt = build_rlm_system_prompt(&metadata);
        assert!(prompt.starts_with(RLM_SYSTEM_PROMPT));
        assert!(prompt.contains("CONTEXT METADATA:"));
        assert!(prompt.contains("Type: str"));
        assert!(prompt.contains("Total length: 150000 characters"));
        assert!(prompt.contains("[50000, 50000, 50000]"));
    }

    #[test]
    fn test_build_rlm_system_prompt_truncates_many_chunks() {
        let metadata = ContextMetadata {
            context_type: "list".to_string(),
            context_total_length: 1_000_000,
            context_lengths: (0..150).map(|_| 6666).collect(),
        };

        let prompt = build_rlm_system_prompt(&metadata);
        assert!(prompt.contains("... (50 more)"));
    }

    #[test]
    fn test_build_rlm_user_prompt_iteration_zero() {
        let prompt = build_rlm_user_prompt(0, "What is the main topic?");
        assert!(prompt.contains("What is the main topic?"));
        assert!(prompt.contains("Do not provide a final answer yet"));
        assert!(prompt.contains("```repl block"));
    }

    #[test]
    fn test_build_rlm_user_prompt_subsequent_iteration() {
        let prompt = build_rlm_user_prompt(1, "What is the main topic?");
        assert!(prompt.contains("Continue your analysis"));
        assert!(prompt.contains("FINAL("));
        // The query is not repeated on subsequent iterations
        assert!(!prompt.contains("What is the main topic?"));
    }

    #[test]
    fn test_format_repl_feedback_basic() {
        let feedback = format_repl_feedback(
            "print(len(context))",
            "150000\n",
            "",
            "context: str (150000 chars)",
            10_000,
        );

        assert!(feedback.contains("Code executed:"));
        assert!(feedback.contains("print(len(context))"));
        assert!(feedback.contains("150000"));
        assert!(feedback.contains("REPL Variables:"));
        assert!(!feedback.contains("Stderr:"));
    }

    #[test]
    fn test_format_repl_feedback_with_stderr() {
        let feedback = format_repl_feedback(
            "x = 1/0",
            "",
            "ZeroDivisionError: division by zero",
            "",
            10_000,
        );

        assert!(feedback.contains("Stderr:"));
        assert!(feedback.contains("ZeroDivisionError"));
    }

    #[test]
    fn test_format_repl_feedback_truncation() {
        let long_stdout = "x".repeat(5000);
        let feedback = format_repl_feedback("print('a' * 5000)", &long_stdout, "", "x: str", 100);

        assert!(feedback.contains("[truncated,"));
        assert!(feedback.contains("chars omitted]"));
        // Total length should be around max_chars + the truncation message
        assert!(feedback.len() < 200);
    }
}
