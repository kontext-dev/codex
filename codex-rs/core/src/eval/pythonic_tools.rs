//! JSON Schema → Python tool wrapper generator for RlmNative mode.
//!
//! Generates Python code that creates a `tools` namespace with typed methods
//! for each Gateway tool. Each method calls `_execute_tool()` internally and
//! supports an optional `fields` parameter for server-side projection.
//!
//! Example output:
//! ```python
//! projects = tools.Linear.list_projects(fields=["name", "issueCount"])
//! ```

use super::runner::GatewayTool;
use serde_json::Value;

/// Sanitize a name into a valid Python identifier.
/// Replaces non-alphanumeric/underscore characters with `_`.
/// Prepends `_` if the result starts with a digit.
fn sanitize_py_identifier(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

/// Extract parameter info from a JSON Schema for a tool's input_schema.
/// Returns a list of (param_name, is_required, has_default) tuples.
fn extract_params(schema: &Value) -> Vec<(String, bool)> {
    let props = match schema.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let mut params: Vec<(String, bool)> = Vec::new();
    // Required params first, then optional
    for (key, _) in props {
        if required.contains(&key.as_str()) {
            params.push((key.clone(), true));
        }
    }
    for (key, _) in props {
        if !required.contains(&key.as_str()) {
            params.push((key.clone(), false));
        }
    }
    params
}

/// Generate a Python type hint string from a JSON Schema value.
fn json_schema_to_py_type(schema: &Value) -> String {
    if schema.get("enum").is_some() {
        return "str".to_string();
    }

    match schema.get("type").and_then(|t| t.as_str()) {
        Some("string") => "str".to_string(),
        Some("number") | Some("integer") => "int".to_string(),
        Some("boolean") => "bool".to_string(),
        Some("array") => "list".to_string(),
        Some("object") => "dict".to_string(),
        Some("null") => "None".to_string(),
        _ => "any".to_string(),
    }
}

/// Generate Python wrapper code for all available tools.
///
/// Creates:
/// 1. A `_execute_tool_projected()` helper that wraps `_execute_tool()` with
///    JSON parsing and optional field projection
/// 2. Per-server namespace classes (e.g., `_LinearTools`) with methods for
///    each tool
/// 3. A `tools` object with server attributes (e.g., `tools.Linear.list_projects()`)
///
/// The generated code is meant to be injected into the REPL namespace via
/// `repl.execute()`.
pub fn generate_pythonic_tool_wrappers(tools: &[GatewayTool]) -> String {
    let mut code = String::new();

    // IMPORTANT: Everything is wrapped in a `_setup_tools()` closure function.
    // This is required because the REPL uses `exec(code, _globals, _locals)`,
    // and class methods defined in exec'd code can only see _globals, NOT _locals.
    // By wrapping in a function, all definitions share a common enclosing scope
    // and classes can find `_call_tool` via normal Python closure scoping.
    code.push_str(
        r#"
def _setup_tools(_et):
    def _call_tool(tool_id, args=None, fields=None):
        resp = _et(tool_id, args, fields)
        if isinstance(resp, dict):
            if resp.get("error") and not resp.get("result") and not resp.get("stored_in_corpus"):
                raise RuntimeError(resp.get("error"))
            if resp.get("stored_in_corpus"):
                summary = resp.get("summary") or "(stored in corpus)"
                chunk_ids = resp.get("chunk_ids") or []
                first_chunk = chunk_ids[0] if chunk_ids else "unknown"
                print(f"-> {tool_id}: stored in corpus (first_chunk={first_chunk})")
                return {
                    "_stored_in_corpus": True,
                    "summary": summary,
                    "chunk_ids": chunk_ids,
                    "tool_id": tool_id,
                }
            result = resp.get("result")
        else:
            result = resp
        if isinstance(result, list):
            print(f"-> {tool_id}: list, {len(result)} items")
        elif isinstance(result, dict):
            _ks = list(result.keys())[:5]
            print(f"-> {tool_id}: dict, keys={_ks}")
        else:
            print(f"-> {tool_id}: {type(result).__name__}")
        return result

"#,
    );

    // Group tools by server
    let mut servers: std::collections::BTreeMap<String, Vec<&GatewayTool>> =
        std::collections::BTreeMap::new();
    for tool in tools {
        servers.entry(tool.server.clone()).or_default().push(tool);
    }

    // Generate per-server classes (indented inside _setup_tools)
    for (server, server_tools) in &servers {
        let class_name = format!("_{}Tools", sanitize_py_identifier(server));
        code.push_str(&format!("    class {}:\n", class_name));

        for tool in server_tools {
            let method_name = sanitize_py_identifier(&tool.name);
            let params = extract_params(&tool.input_schema);

            // Build method signature
            let mut sig_parts: Vec<String> = vec!["self".to_string()];
            for (name, is_required) in &params {
                let safe_name = sanitize_py_identifier(name);
                if *is_required {
                    sig_parts.push(safe_name);
                } else {
                    sig_parts.push(format!("{}=None", safe_name));
                }
            }
            sig_parts.push("fields=None".to_string());

            code.push_str(&format!(
                "        def {}({}):\n",
                method_name,
                sig_parts.join(", ")
            ));

            // Docstring
            let desc = tool.description.replace('\\', "\\\\").replace('"', "\\\"");
            code.push_str(&format!("            \"\"\"{}\"\"\"\n", desc));

            // Build args dict
            code.push_str("            args = {}\n");
            for (name, _) in &params {
                let safe_name = sanitize_py_identifier(name);
                code.push_str(&format!(
                    "            if {} is not None:\n                args[\"{}\"] = {}\n",
                    safe_name, name, safe_name
                ));
            }

            // Call and return — uses `_call_tool` from enclosing scope (closure)
            code.push_str(&format!(
                "            return _call_tool(\"{}\", args, fields)\n\n",
                tool.id
            ));
        }
    }

    // Build the namespace class (indented inside _setup_tools)
    code.push_str("    class _Ns:\n");
    code.push_str("        def __init__(self):\n");
    if servers.is_empty() {
        code.push_str("            pass\n");
    }
    for server in servers.keys() {
        let attr_name = sanitize_py_identifier(server);
        let class_name = format!("_{}Tools", sanitize_py_identifier(server));
        code.push_str(&format!(
            "            self.{} = {}()\n",
            attr_name, class_name
        ));
    }
    code.push_str("        def __repr__(self):\n");
    code.push_str("            servers = [a for a in dir(self) if not a.startswith('_')]\n");
    code.push_str("            return f\"tools({', '.join(servers)})\"\n");

    // Return the namespace object
    code.push_str("\n    return _Ns()\n\n");

    // Call the setup function and assign to `tools`
    code.push_str("tools = _setup_tools(execute_tool_object)\n");

    code
}

/// Generate a compact `dir(tools)`-style summary of available methods.
///
/// Produces output like:
/// ```text
/// Available methods in REPL:
///   tools.Linear  → list_projects(), list_issues(), ...
///   tools.GitHub  → search_repositories(), get_repo(), ...
/// ```
pub fn generate_tools_dir_summary(tools: &[GatewayTool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut servers: std::collections::BTreeMap<String, Vec<&GatewayTool>> =
        std::collections::BTreeMap::new();
    for tool in tools {
        servers.entry(tool.server.clone()).or_default().push(tool);
    }

    let mut lines = vec!["\nAvailable methods in REPL:".to_string()];
    for (server, server_tools) in &servers {
        let safe_server = sanitize_py_identifier(server);
        let method_names: Vec<String> = server_tools
            .iter()
            .map(|t| format!("{}()", sanitize_py_identifier(&t.name)))
            .collect();
        lines.push(format!(
            "  tools.{}  \u{2192} {}",
            safe_server,
            method_names.join(", ")
        ));
    }
    lines.push(String::new()); // trailing newline
    lines.join("\n")
}

/// Generate a tool list section for the RlmNative system prompt.
///
/// Lists all available tools in `tools.Server.method_name()` format
/// with parameter signatures and descriptions.
pub fn format_tool_list_for_native_prompt(tools: &[GatewayTool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut section = String::from(
        "\n\n# Available Tools\n\nCall tools using the `tools` object. All methods accept an optional `fields` parameter for projection.\n\n",
    );

    // Group by server
    let mut servers: std::collections::BTreeMap<String, Vec<&GatewayTool>> =
        std::collections::BTreeMap::new();
    for tool in tools {
        servers.entry(tool.server.clone()).or_default().push(tool);
    }

    for (server, server_tools) in &servers {
        let safe_server = sanitize_py_identifier(server);
        section.push_str(&format!("## {}\n", server));
        for tool in server_tools {
            let method_name = sanitize_py_identifier(&tool.name);
            let params = extract_params(&tool.input_schema);

            let param_str: String = if params.is_empty() {
                String::new()
            } else {
                let parts: Vec<String> = params
                    .iter()
                    .map(|(name, required)| {
                        let safe = sanitize_py_identifier(name);
                        let ty = tool
                            .input_schema
                            .get("properties")
                            .and_then(|p| p.get(name))
                            .map(|s| json_schema_to_py_type(s))
                            .unwrap_or_else(|| "any".to_string());
                        if *required {
                            format!("{}: {}", safe, ty)
                        } else {
                            format!("{}?: {}", safe, ty)
                        }
                    })
                    .collect();
                parts.join(", ")
            };

            section.push_str(&format!(
                "- `tools.{}.{}({})` — {}\n",
                safe_server, method_name, param_str, tool.description
            ));
        }
        section.push('\n');
    }

    section.push_str("**Projection example:** `tools.Linear.list_projects(fields=[\"name\", \"issueCount\"])` — returns only the specified fields, reducing output size.\n");

    section
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_tool(server: &str, name: &str, desc: &str, schema: Value) -> GatewayTool {
        GatewayTool {
            id: format!("{}:{}", server, name),
            server: server.to_string(),
            name: name.to_string(),
            description: desc.to_string(),
            input_schema: schema,
        }
    }

    #[test]
    fn test_sanitize_py_identifier() {
        assert_eq!(sanitize_py_identifier("list_projects"), "list_projects");
        assert_eq!(sanitize_py_identifier("get-repo"), "get_repo");
        assert_eq!(sanitize_py_identifier("123abc"), "_123abc");
        assert_eq!(sanitize_py_identifier("Code Executor"), "Code_Executor");
    }

    #[test]
    fn test_extract_params_required_and_optional() {
        let schema = json!({
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "page": {"type": "number"}
            },
            "required": ["owner", "repo"]
        });
        let params = extract_params(&schema);
        assert_eq!(params.len(), 3);
        // Required params come first
        assert!(params[0].1); // owner required
        assert!(params[1].1); // repo required
        assert!(!params[2].1); // page optional
    }

    #[test]
    fn test_extract_params_empty_schema() {
        let schema = json!({});
        let params = extract_params(&schema);
        assert!(params.is_empty());
    }

    #[test]
    fn test_generate_wrappers_single_tool() {
        let tools = vec![make_tool(
            "Linear",
            "list_projects",
            "List all projects",
            json!({
                "type": "object",
                "properties": {
                    "filter": {"type": "object"},
                    "first": {"type": "number"}
                }
            }),
        )];

        let code = generate_pythonic_tool_wrappers(&tools);

        assert!(code.contains("class _LinearTools:"));
        assert!(code.contains("def list_projects(self"));
        assert!(code.contains("fields=None"));
        assert!(code.contains("_call_tool(\"Linear:list_projects\""));
        assert!(code.contains("tools = _setup_tools(execute_tool_object)"));
        assert!(code.contains("self.Linear = _LinearTools()"));
        // Must be wrapped in _setup_tools closure with _et parameter
        assert!(code.contains("def _setup_tools(_et):"));
    }

    #[test]
    fn test_generate_wrappers_multiple_servers() {
        let tools = vec![
            make_tool(
                "Linear",
                "list_projects",
                "List projects",
                json!({"type": "object", "properties": {"first": {"type": "number"}}}),
            ),
            make_tool(
                "GitHub",
                "get_repo",
                "Get a repository",
                json!({
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"}
                    },
                    "required": ["owner", "repo"]
                }),
            ),
        ];

        let code = generate_pythonic_tool_wrappers(&tools);

        assert!(code.contains("class _LinearTools:"));
        assert!(code.contains("class _GitHubTools:"));
        assert!(code.contains("self.Linear = _LinearTools()"));
        assert!(code.contains("self.GitHub = _GitHubTools()"));
        // Required params should not have defaults
        assert!(code.contains("def get_repo(self, owner, repo, fields=None)"));
    }

    #[test]
    fn test_generate_wrappers_empty_tools() {
        let code = generate_pythonic_tool_wrappers(&[]);
        assert!(code.contains("_call_tool"));
        assert!(code.contains("tools = _setup_tools(execute_tool_object)"));
    }

    #[test]
    fn test_format_tool_list_for_native_prompt() {
        let tools = vec![
            make_tool(
                "Linear",
                "list_projects",
                "List all projects",
                json!({"type": "object", "properties": {"first": {"type": "number"}}}),
            ),
            make_tool(
                "GitHub",
                "get_repo",
                "Get a repository",
                json!({
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"}
                    },
                    "required": ["owner", "repo"]
                }),
            ),
        ];

        let section = format_tool_list_for_native_prompt(&tools);
        assert!(section.contains("tools.Linear.list_projects"));
        assert!(section.contains("tools.GitHub.get_repo"));
        assert!(section.contains("owner: str, repo: str"));
        assert!(section.contains("Projection example"));
    }

    #[test]
    fn test_generate_tools_dir_summary() {
        let tools = vec![
            make_tool(
                "Linear",
                "list_projects",
                "List projects",
                json!({"type": "object", "properties": {}}),
            ),
            make_tool(
                "Linear",
                "list_issues",
                "List issues",
                json!({"type": "object", "properties": {}}),
            ),
            make_tool(
                "GitHub",
                "get_repo",
                "Get a repository",
                json!({"type": "object", "properties": {}}),
            ),
        ];

        let summary = generate_tools_dir_summary(&tools);
        assert!(summary.contains("Available methods in REPL:"));
        assert!(summary.contains("tools.Linear"));
        assert!(summary.contains("list_projects()"));
        assert!(summary.contains("list_issues()"));
        assert!(summary.contains("tools.GitHub"));
        assert!(summary.contains("get_repo()"));
    }

    #[test]
    fn test_generate_tools_dir_summary_empty() {
        let summary = generate_tools_dir_summary(&[]);
        assert!(summary.is_empty());
    }
}
