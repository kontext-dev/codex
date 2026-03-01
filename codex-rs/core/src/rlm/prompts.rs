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
pub const RLM_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query using a Python REPL with access to external tools and sub-LLM reasoning.

Your REPL environment provides:
1. `context` — the task description (you already see it in the user prompt, so do NOT waste a step printing it).
2. `execute_tool_json(tool_id: str, args: dict = None, fields: list[str] = None) -> Any` — preferred tool call helper. Returns parsed Python objects (dict/list) when possible. If a large result is stored in corpus, returns metadata: `{"_stored_in_corpus": true, "summary": "...", "chunk_ids": [...], "tool_id": "..."}`.
3. `execute_tool(tool_id: str, args: dict = None, fields: list[str] = None) -> str` — legacy helper that returns a JSON string.
4. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM. Use ONLY when you need to reason over large content (>2000 chars). Never use it just to reformat data you already have.
5. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently.
6. `corpus_search(query: str, max_results: int = 5) -> list[dict]` — search stored results for relevant chunks.
7. `corpus_get_chunk(chunk_id: str) -> str` — retrieve full content for a stored chunk.
8. `SHOW_VARS()` — list all REPL variables.
9. `print()` — standard output (captured and returned to you).

WORKFLOW:
- Start by calling the tools you need. You can fetch data across multiple steps.
- Inspect results with print(), then continue with the next step.
- When you have all the data, compute the answer in Python and print it.
- Prefer fewer turns, but correctness matters more than minimizing turns.

KEY RULES:
- Use `execute_tool_json()` for ALL data retrieval from external tools (Linear, GitHub, DeepWiki, Context7, etc.).
- Use `execute_tool()` only if you explicitly need the raw JSON string.
- Do computations (counting, filtering, matching, sorting, aggregation) in Python directly — do NOT delegate simple data processing to llm_query.
- Use `llm_query()` ONLY for reasoning over large content that you cannot process directly. Never call it in a loop over many items.
- If a tool result has `_stored_in_corpus == True`, use `chunk_ids` and `corpus_get_chunk()` (or `corpus_search()`) to retrieve needed content.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.
- IMPORTANT: Only `print()` output is returned to you as feedback. Data stored in variables is NOT visible between turns. You MUST print() any results you want to inspect or verify. Tool calls auto-print a one-line status (e.g. `-> tool_id: 1234 chars`), but you must print() the actual data you need.

Example — multi-tool task:
```repl
# Fetch data
projects = execute_tool_json("Linear:list_projects")
print(f"Got {len(projects)} projects")
# Process in Python
names = [p["name"] for p in projects]
print(names[:10])
```

Example — analysis requiring sub-LLM:
```repl
wiki = execute_tool("DeepWiki:read_wiki_contents", {"repoName": "facebook/react"})
summary = llm_query(f"Count the top-level topics in this structure:\n{wiki}")
print(summary)
```

To provide your final answer:
- `FINAL(your final answer here)` — for direct text answers
- `FINAL_VAR(variable_name)` — to return the value of a REPL variable

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl block FIRST, then call FINAL_VAR in a SEPARATE step."#;

/// System prompt for RLM+CodeMode: RLM REPL with Gateway tool execution.
///
/// This extends the base RLM prompt with `execute_code()`, `corpus_search()`,
/// and `corpus_get_chunk()` functions that allow the REPL to call Gateway
/// tools (Linear, GitHub, etc.) via server-side code execution.
pub const RLM_CODEMODE_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query using a Python REPL with Gateway tool execution via TypeScript and sub-LLM reasoning.

Your REPL environment provides:
1. `context` — the task description (you already see it in the user prompt, so do NOT waste a step printing it).
2. `execute_code(ts_code: str) -> str` — execute TypeScript on the Gateway. Use `codemode.*` methods to call tools (Linear, GitHub, DeepWiki, etc.). In your JS, extract only the fields you need using `.map()`, `.filter()`, or direct property access.
3. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM. Use ONLY when you need to reason over large content (>2000 chars). Never use it just to reformat data you already have.
4. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently.
5. `corpus_search(query: str, max_results: int = 5) -> list[dict]` — search stored results for relevant chunks.
6. `corpus_get_chunk(chunk_id: str) -> str` — retrieve full content for a stored chunk.
7. `SHOW_VARS()` — list all REPL variables.
8. `print()` — standard output (captured and returned to you).

WORKFLOW:
- Start by calling tools to fetch data. You can fetch data across multiple steps.
- Inspect results with print(), then continue with the next step.
- When you have all the data, compute the answer in Python and print it.
- Prefer fewer turns, but correctness matters more than minimizing turns.

KEY RULES:
- Use `execute_code()` for ALL data retrieval from external tools via `codemode.*`.
- Do computations (counting, filtering, matching, sorting, aggregation) in Python directly — do NOT delegate simple data processing to llm_query.
- Use `llm_query()` ONLY for reasoning over large content that you cannot process directly. Never call it in a loop over many items.
- Large results may be stored in corpus. If you see "stored in corpus" in a result, use `corpus_search()` and `corpus_get_chunk()` to inspect full content.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.
- IMPORTANT: Only `print()` output is returned to you as feedback. Data stored in variables is NOT visible between turns. You MUST print() any results you want to inspect or verify. Tool calls auto-print a one-line status, but you must print() the actual data you need.

Example — fetch and process:
```repl
result = execute_code('''
const projects = await codemode.Linear_list_projects({});
return { names: projects.map(p => p.name).slice(0, 20) };
''')
print(result)
```

Example — analysis requiring sub-LLM:
```repl
chunk = corpus_get_chunk(hits[0]["chunk_id"])
analysis = llm_query(f"Summarize key findings and risks:\n{chunk}")
print(analysis)
```

To provide your final answer:
- `FINAL(your final answer here)` — for direct text answers
- `FINAL_VAR(variable_name)` — to return the value of a REPL variable

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl block FIRST, then call FINAL_VAR in a SEPARATE step."#;

/// System prompt for RLM+Native: Python-native tool wrappers with field projection.
///
/// The LLM stays in Python the entire time. Tools are accessed via `tools.Server.method()`,
/// with an optional `fields` parameter for projection. No TypeScript, no `execute_code()`.
pub const RLM_NATIVE_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query using a Python REPL with direct access to Gateway tools and sub-LLM reasoning.

Your REPL environment provides:
1. `context` — the task description (you already see it in the user prompt, so do NOT waste a step printing it).
2. `tools` — a namespace with methods for each available Gateway tool. Call them directly:
   - `tools.Linear.list_projects()` — returns a Python list/dict
   - `tools.GitHub.get_repo(owner="facebook", repo="react")` — with typed arguments
   - `tools.Linear.list_projects(fields=["name", "issueCount"])` — returns ONLY those fields (much smaller output)
3. `llm_query(prompt: str, model: str = None) -> str` — query a sub-LLM. Use ONLY when you need to reason over large content (>2000 chars). Never use it just to reformat data you already have.
4. `llm_query_batched(prompts: list[str], model: str = None) -> list[str]` — query multiple prompts concurrently.
5. `corpus_search(query: str, max_results: int = 5) -> list[dict]` — search stored results for relevant chunks.
6. `corpus_get_chunk(chunk_id: str) -> str` — retrieve full content for a stored chunk.
7. `SHOW_VARS()` — list all REPL variables.
8. `print()` — standard output (captured and returned to you).

WORKFLOW:
- Start by calling the tools you need. You can fetch data across multiple steps.
- Inspect results with print(), then continue with the next step.
- When you have all the data, print your final answer.
- Prefer fewer turns, but correctness matters more than minimizing turns.

KEY RULES:
- Use `tools.*` for ALL data retrieval. Call tool methods directly — they return parsed Python objects (dicts/lists), not raw JSON strings.
- Use `fields=[...]` whenever you only need specific fields from a result. This keeps output small and focused.
- Use `llm_query()` ONLY for reasoning over large content that you cannot process directly.
- If a result is stored in corpus, call `corpus_search()` / `corpus_get_chunk()` to retrieve what you need.
- All code MUST be wrapped in triple backticks with the `repl` language identifier.
- IMPORTANT: Only `print()` output is returned to you as feedback. Data stored in variables is NOT visible between turns. You MUST print() any results you want to inspect or verify. Tool calls auto-print a one-line status (e.g. `-> tool_id: list, 5 items`), but you must print() the actual data you need.

Example — multi-tool task (ONE block):
```repl
# Fetch all data in one block
projects = tools.Linear.list_projects(fields=["name", "issueCount"])
first_name = projects["projects"][0]["name"]
repos = tools.GitHub.search_repositories(query=first_name, perPage=1)
repo = repos["items"][0] if repos["items"] else None
if repo:
    wiki = tools.DeepWiki.read_wiki_structure(repoName=repo["full_name"])
else:
    wiki = "No matching repo found"
print(f"Project: {first_name}")
print(f"Repo: {repo['full_name'] if repo else 'None'}, Stars: {repo['stargazers_count'] if repo else 'N/A'}")
print(f"Wiki: {wiki}")
```
FINAL(The first project is ... with repo ... having N stars and M wiki topics.)

Example — analysis requiring sub-LLM (ONE block):
```repl
wiki = tools.DeepWiki.read_wiki_structure(repoName="facebook/react")
summary = llm_query(f"Count the top-level topics in this structure:\n{wiki}")
print(summary)
```
FINAL(The wiki has N top-level topics.)

To provide your final answer:
- `FINAL(your final answer here)` — for direct text answers
- `FINAL_VAR(variable_name)` — to return the value of a REPL variable

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl block FIRST, then call FINAL_VAR in a SEPARATE step."#;

/// Build the complete RLM+Native system prompt with context metadata appended.
pub fn build_rlm_native_system_prompt(metadata: &ContextMetadata) -> String {
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
        let shown: Vec<String> = metadata
            .context_lengths
            .iter()
            .map(|l| l.to_string())
            .collect();
        format!("[{}]", shown.join(", "))
    };

    format!(
        "{}\n\nCONTEXT METADATA:\n- Type: {}\n- Total length: {} characters\n- Chunk lengths: {}",
        RLM_NATIVE_SYSTEM_PROMPT,
        metadata.context_type,
        metadata.context_total_length,
        lengths_display,
    )
}

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
        let shown: Vec<String> = metadata
            .context_lengths
            .iter()
            .map(|l| l.to_string())
            .collect();
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
        format!(
            "[{}, ... ({} more)]",
            shown.join(", "),
            metadata.context_lengths.len() - 100
        )
    } else {
        let shown: Vec<String> = metadata
            .context_lengths
            .iter()
            .map(|l| l.to_string())
            .collect();
        format!("[{}]", shown.join(", "))
    };

    format!(
        "{}\n\nCONTEXT METADATA:\n- Type: {}\n- Total length: {} characters\n- Chunk lengths: {}",
        RLM_SYSTEM_PROMPT, metadata.context_type, metadata.context_total_length, lengths_display,
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
        "Code executed:\n```repl\n{}\n```\n\nREPL Output:\n{}",
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
        assert!(RLM_SYSTEM_PROMPT.contains("REPL"));
        assert!(RLM_SYSTEM_PROMPT.contains("llm_query"));
        assert!(RLM_SYSTEM_PROMPT.contains("llm_query_batched"));
        assert!(RLM_SYSTEM_PROMPT.contains("SHOW_VARS()"));
        assert!(RLM_SYSTEM_PROMPT.contains("FINAL("));
        assert!(RLM_SYSTEM_PROMPT.contains("FINAL_VAR("));
        assert!(RLM_SYSTEM_PROMPT.contains("```repl"));
        // Should discourage excessive llm_query usage
        assert!(RLM_SYSTEM_PROMPT.contains("ONLY"));
        assert!(RLM_SYSTEM_PROMPT.contains("do NOT delegate simple data processing"));
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

        // Must contain corpus guidance
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("stored in corpus"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("corpus_search()"));

        // Should discourage excessive llm_query usage
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("ONLY"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("do NOT delegate simple data processing"));

        // Must contain REPL basics
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("execute_code"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("llm_query"));
        assert!(RLM_CODEMODE_SYSTEM_PROMPT.contains("```repl"));
    }

    #[test]
    fn test_rlm_native_system_prompt_contains_key_sections() {
        // Must use tools.* Python API
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("tools.Linear.list_projects"));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("tools.GitHub.get_repo"));

        // Must contain fields parameter guidance
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("fields="));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("fields=[\"name\""));

        // Must NOT contain TypeScript/CodeMode concepts
        assert!(!RLM_NATIVE_SYSTEM_PROMPT.contains("execute_code"));
        assert!(!RLM_NATIVE_SYSTEM_PROMPT.contains("codemode."));
        assert!(!RLM_NATIVE_SYSTEM_PROMPT.contains("TypeScript"));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("corpus_search"));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("corpus_get_chunk"));

        // Must contain FINAL/FINAL_VAR
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("FINAL("));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("FINAL_VAR("));

        // Must contain llm_query
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("llm_query"));
        assert!(RLM_NATIVE_SYSTEM_PROMPT.contains("```repl"));
    }

    #[test]
    fn test_build_rlm_native_system_prompt_appends_metadata() {
        let metadata = ContextMetadata {
            context_type: "str".to_string(),
            context_total_length: 50_000,
            context_lengths: vec![50_000],
        };

        let prompt = build_rlm_native_system_prompt(&metadata);
        assert!(prompt.starts_with(RLM_NATIVE_SYSTEM_PROMPT));
        assert!(prompt.contains("CONTEXT METADATA:"));
        assert!(prompt.contains("Type: str"));
        assert!(prompt.contains("Total length: 50000 characters"));
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
