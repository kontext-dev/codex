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
pub const RLM_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs and call external tools, which you are strongly encouraged to use as much as possible.

Your REPL environment is initialized with the following:
1. `context` — a variable that contains extremely important information about your query. It may be a string, dictionary, or list depending on the data.
2. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM that can handle around 500K characters. Use this to analyze chunks of the context.
3. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently for parallel analysis.
4. `execute_tool(tool_id: str, args: dict = None) -> str` — call a Gateway tool directly. Tool IDs use the format `"Server:tool_name"` (e.g., `execute_tool("Linear:list_projects")` or `execute_tool("GitHub:get_repo", {"owner": "facebook", "repo": "react"})`). Returns the full tool result as JSON. See the "Available Gateway Tools" section below for exact tool IDs.
5. `SHOW_VARS()` — returns all variables you have created in the REPL.
6. `print()` statements to view the output of your REPL code.

IMPORTANT:
- Use `execute_tool()` to retrieve live data from external tools (Linear, GitHub, DeepWiki, Context7, etc.).
- Use `llm_query()` for reasoning, synthesis, and analysis of data you've collected.
- You will only be able to see truncated outputs from the REPL environment, so you should use the llm_query function on variables you want to analyze in detail.
- Make sure to explicitly look through the entire context in REPL before answering your query.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.

Here are some example strategies you can use:

**Strategy 1 — Examine the context and call tools:**
```repl
# Read the task
print(context)

# Call a tool to get live data (use exact tool_id from Available Gateway Tools)
projects = execute_tool("Linear:list_projects")
print(projects[:500])
```

**Strategy 2 — Analyze tool results with llm_query:**
```repl
# Get detailed data
import json
data = json.loads(projects)
summary = llm_query(f"Summarize these projects:\n{projects}")
print(summary)
```

**Strategy 3 — Batch processing with llm_query_batched:**
```repl
# Process multiple items in parallel
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

/// System prompt for RLM+CodeMode: RLM REPL with Gateway tool execution.
///
/// This extends the base RLM prompt with `execute_code()`, `corpus_search()`,
/// and `corpus_get_chunk()` functions that allow the REPL to call Gateway
/// tools (Linear, GitHub, etc.) via server-side code execution.
pub const RLM_CODEMODE_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query using a Python REPL that can execute Gateway tools via TypeScript and call sub-LLMs for reasoning.

Your REPL environment provides:
1. `context` — the task description (you already see it in the user prompt, so do NOT waste a step printing it).
2. `execute_code(ts_code: str) -> str` — execute TypeScript on the Gateway. Use `codemode.*` methods to call tools (Linear, GitHub, DeepWiki, etc.). In your JS, extract only the fields you need using `.map()` — do NOT return raw tool results.
3. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM. Use ONLY when you need to reason over large content (>2000 chars) that cannot be processed in JS. Never use it just to reformat data you already have.
4. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently for parallel analysis.
5. `corpus_search(query: str, max_results: int = 5) -> list[dict]` — search stored results. Only use if you see "stored in corpus" in a result. Do not speculatively search the corpus.
6. `corpus_get_chunk(chunk_id: str) -> str` — retrieve full content of a stored chunk.
7. `SHOW_VARS()` — list all REPL variables.
8. `print()` — standard output (captured and returned to you).

KEY RULES:
- For tasks answerable with a single tool call, call `execute_code()` once, read the result, and answer directly with `FINAL()` — do NOT use `llm_query()` for simple formatting or extraction.
- In your `execute_code()` TypeScript, extract only the fields you need using `.map()`, `.filter()`, or direct property access. Do NOT return full JSON payloads.
- Use `llm_query()` ONLY when you need to reason over large content (>2000 chars) that cannot be processed in JS. Never use it just to reformat data you already have.
- Large results are auto-stored in corpus. Only use `corpus_search()`/`corpus_get_chunk()` if you see "stored in corpus" in the result.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.

Example — single tool call task:
```repl
result = execute_code('''
const projects = await codemode.Linear_list_projects({});
return { names: projects.map(p => p.name).slice(0, 20) };
''')
print(result)
```
FINAL(Based on the result, the projects are: ...)

Example — multi-step analysis:
```repl
result = execute_code('''
const issues = await codemode.Linear_list_issues({
  filter: { state: { name: { eq: "In Progress" } } },
  first: 10
});
return { issues: issues.map(i => ({ title: i.title, assignee: i.assignee?.name })) };
''')
print(result)
```
```repl
# Only if result was too large and stored in corpus
analysis = llm_query(f"Summarize the key themes from these issues:\n{result}")
print(analysis)
```
FINAL(Here is the summary: ...)

To provide your final answer:
- `FINAL(your final answer here)` — for direct text answers
- `FINAL_VAR(variable_name)` — to return the value of a REPL variable

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl block FIRST, then call FINAL_VAR in a SEPARATE step.

Think step by step carefully, plan, and execute immediately — do not just say "I will do this"."#;

/// Build the complete RLM+CodeMode system prompt with context metadata appended.
pub fn build_rlm_codemode_system_prompt(metadata: &ContextMetadata) -> String {
    let lengths_display = if metadata.context_lengths.len() > 100 {
        let shown: Vec<String> = metadata.context_lengths[..100]
            .iter()
            .map(|l| l.to_string())
            .collect();
        format!(
            "[{}, ... ({} more)]",
            shown.join(", "),
            metadata.context_lengths.len() - 100
        )
    } else {
        let shown: Vec<String> = metadata.context_lengths.iter().map(|l| l.to_string()).collect();
        format!("[{}]", shown.join(", "))
    };

    format!(
        "{}\n\nCONTEXT METADATA:\n- Type: {}\n- Total length: {} characters\n- Chunk lengths: {}",
        RLM_CODEMODE_SYSTEM_PROMPT,
        metadata.context_type,
        metadata.context_total_length,
        lengths_display,
    )
}

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
///
/// Follows the reference RLM pattern: always include the original query so the
/// model never loses sight of the task, and end with a directive ("Your next
/// action:") to force generation.
pub fn build_rlm_user_prompt(iteration: usize, max_iterations: usize, query: &str) -> String {
    if iteration == 0 {
        format!(
            "You have not interacted with the REPL environment or seen your context yet. \
             Your next action should be to look through and figure out how to answer the prompt, \
             so don't just provide a final answer yet.\n\n\
             Think step-by-step on what to do using the REPL environment (which contains the context) \
             to answer the original prompt: \"{query}\"\n\n\
             Continue using the REPL environment, which has the `context` variable, \
             and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. \
             Your next action:",
            query = query
        )
    } else if iteration >= max_iterations - 1 {
        format!(
            "The history above is your previous interactions with the REPL environment. \
             This is your final iteration — you must provide your answer now.\n\n\
             Think step-by-step to answer the original prompt: \"{query}\"\n\n\
             Synthesize everything you have gathered so far into a complete answer \
             using FINAL(your answer) or FINAL_VAR(variable_name). Your next action:",
            query = query
        )
    } else {
        format!(
            "The history above is your previous interactions with the REPL environment. \
             Think step-by-step on what to do using the REPL environment to answer the original prompt: \"{query}\"\n\n\
             Continue using the REPL environment and querying sub-LLMs by writing to ```repl``` tags. \
             If you have enough information, provide your final answer using FINAL(...). \
             Your next action:",
            query = query
        )
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
    fn test_rlm_codemode_system_prompt_contains_key_sections() {
        // Must use correct codemode.* API, not tools.EXECUTE_TOOL
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("codemode."));
        assert!(!RLM_CODEMODE_SYSTEM_PROMPT.contains("tools.EXECUTE_TOOL"));

        // Must contain field filtering guidance
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains(".map("));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("extract only the fields you need"));

        // Must contain FINAL/FINAL_VAR
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("FINAL("));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("FINAL_VAR("));

        // Must contain direct-answer guidance
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("single tool call"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("answer directly with `FINAL()`"));

        // Must contain corpus guidance
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("stored in corpus"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("Do not speculatively search the corpus"));

        // Must contain llm_query restriction
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("ONLY when you need to reason over large content"));

        // Must NOT contain wasteful print(context) step
        assert!(!RLM_CODEMODE_SYSTEM_PROMPT.contains("print(context)"));

        // Must contain REPL basics
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("execute_code"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("llm_query"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("```repl"));
    }

    #[test]
    fn test_build_rlm_codemode_system_prompt_appends_metadata() {
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: 50_000,
            context_lengths: vec![50_000],
        };

        let prompt = build_rlm_codemode_system_prompt(&metadata);
        assert!(prompt.starts_with(RLM_CODEMODE_SYSTEM_PROMPT));
        assert!(prompt.contains("CONTEXT METADATA:"));
        assert!(prompt.contains("Type: str"));
        assert!(prompt.contains("Total length: 50000 characters"));
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
        let prompt = build_rlm_user_prompt(0, 10, "What is the main topic?");
        assert!(prompt.contains("What is the main topic?"));
        assert!(prompt.contains("don't just provide a final answer yet"));
        assert!(prompt.contains("Your next action:"));
    }

    #[test]
    fn test_build_rlm_user_prompt_subsequent_iteration() {
        let prompt = build_rlm_user_prompt(1, 10, "What is the main topic?");
        assert!(prompt.contains("previous interactions"));
        assert!(prompt.contains("FINAL("));
        // Query IS repeated on subsequent iterations (matches reference RLM)
        assert!(prompt.contains("What is the main topic?"));
        assert!(prompt.contains("Your next action:"));
    }

    #[test]
    fn test_build_rlm_user_prompt_final_iteration() {
        let prompt = build_rlm_user_prompt(9, 10, "What is the main topic?");
        assert!(prompt.contains("final iteration"));
        assert!(prompt.contains("What is the main topic?"));
        assert!(prompt.contains("Your next action:"));
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
