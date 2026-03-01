//! JSON Schema → TypeScript type declaration converter for CodeMode.
//!
//! Generates TypeScript `declare const codemode` blocks from `GatewayTool`
//! schemas so the LLM sees typed function signatures in its system prompt.

use super::runner::GatewayTool;
use serde_json::Value;

/// Sanitize a tool name into a valid JavaScript/TypeScript identifier.
/// Replaces any character that isn't alphanumeric or `_` with `_`.
/// Prepends `_` if the result starts with a digit (identifiers can't start with digits).
/// Mirrors Cloudflare's `sanitizeToolName()` which also handles digit-leading names.
fn sanitize_identifier(name: &str) -> String {
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

/// Convert a JSON Schema value to a TypeScript type string.
///
/// Handles: string, number, integer, boolean, array (with items),
/// object (with properties), enum, nullable. Unknown → `unknown`.
pub fn json_schema_to_ts_type(schema: &Value) -> String {
    // Handle enum first — works at any level
    if let Some(variants) = schema.get("enum").and_then(|v| v.as_array()) {
        let literals: Vec<String> = variants
            .iter()
            .map(|v| match v {
                Value::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => "null".to_string(),
                _ => "unknown".to_string(),
            })
            .collect();
        return literals.join(" | ");
    }

    let type_val = schema.get("type").and_then(|t| t.as_str());

    match type_val {
        Some("string") => "string".to_string(),
        Some("number") | Some("integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("array") => {
            let items_type = schema
                .get("items")
                .map(|i| json_schema_to_ts_type(i))
                .unwrap_or_else(|| "unknown".to_string());
            format!("{}[]", wrap_if_union(&items_type))
        }
        Some("object") => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                let required: Vec<&str> = schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                    .unwrap_or_default();

                let fields: Vec<String> = props
                    .iter()
                    .map(|(key, val)| {
                        let ts_type = json_schema_to_ts_type(val);
                        let optional = if required.contains(&key.as_str()) {
                            ""
                        } else {
                            "?"
                        };
                        format!("{}{}: {}", key, optional, ts_type)
                    })
                    .collect();

                format!("{{ {} }}", fields.join("; "))
            } else {
                "object".to_string()
            }
        }
        _ => {
            // Handle anyOf / oneOf as union types
            let union_key = if schema.get("anyOf").is_some() {
                "anyOf"
            } else if schema.get("oneOf").is_some() {
                "oneOf"
            } else {
                return "unknown".to_string();
            };

            if let Some(variants) = schema.get(union_key).and_then(|v| v.as_array()) {
                let types: Vec<String> = variants.iter().map(json_schema_to_ts_type).collect();
                types.join(" | ")
            } else {
                "unknown".to_string()
            }
        }
    }
}

/// Wrap a type string in parens if it contains `|` (union), for use in array types.
fn wrap_if_union(ts: &str) -> String {
    if ts.contains('|') {
        format!("({})", ts)
    } else {
        ts.to_string()
    }
}

/// Generate TypeScript type declarations for all available tools.
///
/// Produces:
/// ```text
/// type Linear_list_projects_Input = { filter?: object; first?: number; }
///
/// declare const codemode: {
///   /** List all projects */
///   Linear_list_projects: (input?: Linear_list_projects_Input) => Promise<unknown>;
/// }
/// ```
pub fn generate_codemode_types(tools: &[GatewayTool]) -> String {
    let mut type_defs = Vec::new();
    let mut method_defs = Vec::new();

    for tool in tools {
        let safe_name = sanitize_identifier(&tool.prefixed_name());
        let input_type_name = format!("{}_Input", safe_name);

        // Generate the input type from the tool's input_schema
        let input_type =
            if tool.input_schema.is_object() && tool.input_schema.get("properties").is_some() {
                json_schema_to_ts_type(&tool.input_schema)
            } else {
                // No properties — accept an empty object
                "object".to_string()
            };

        type_defs.push(format!("type {} = {};", input_type_name, input_type));

        // Escape description for JSDoc
        let desc = tool.description.replace("*/", "* /");
        method_defs.push(format!(
            "  /** {} */\n  {}: (input?: {}) => Promise<unknown>;",
            desc, safe_name, input_type_name,
        ));
    }

    let types_block = type_defs.join("\n");
    let methods_block = method_defs.join("\n");

    format!(
        "{}\n\ndeclare const codemode: {{\n{}\n}};",
        types_block, methods_block
    )
}

/// Generate a JavaScript preamble that creates the runtime `codemode` object.
///
/// This is injected into the code sent to EXECUTE_CODE so the LLM-generated
/// `codemode.*` calls resolve at runtime. Includes an `__extractResult` helper
/// that unwraps MCP response envelopes.
///
/// Uses `var` so it won't conflict if the Gateway also injects a preamble.
pub fn generate_codemode_js_preamble(tools: &[GatewayTool]) -> String {
    // Two-layer unwrap: Gateway responses are often double-wrapped:
    //   Layer 1: content[0].resource.text or content[0].text → JSON string
    //   Layer 2: parsed result may itself have content[0].text → another JSON string
    // ES5-compatible helper — avoids ?. and ?? which some VMs don't support
    let extract_helper = r#"function __extractResult(raw) {
  if (raw == null) return raw;
  var c = raw.content && raw.content[0];
  var text = c && ((c.resource && c.resource.text) || c.text);
  if (typeof text === 'string') {
    try { raw = JSON.parse(text); } catch(e) { return text; }
  }
  var c2 = raw.content && raw.content[0];
  var inner = c2 && c2.text;
  if (typeof inner === 'string') {
    try { raw = JSON.parse(inner); } catch(e) { return inner; }
  }
  if (raw && typeof raw === 'object' && !Array.isArray(raw) && raw.error && typeof raw.error === 'string') {
    throw new Error(raw.message || raw.error);
  }
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    var arrays = [];
    var keys = Object.keys(raw);
    for (var i = 0; i < keys.length; i++) {
      if (Array.isArray(raw[keys[i]])) arrays.push(keys[i]);
    }
    if (arrays.length === 1) return raw[arrays[0]];
  }
  return raw;
}"#;

    let entries: Vec<String> = tools
        .iter()
        .map(|t| {
            let raw_name = t.prefixed_name();
            let safe_name = sanitize_identifier(&raw_name);
            if safe_name != raw_name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_") {
                tracing::warn!("  [codemode] sanitized tool key: {} -> {}", raw_name, safe_name);
            }
            let id_json = serde_json::to_string(&t.id).unwrap_or_else(|_| format!("\"{}\"", t.id));
            format!(
                "  {safe}: async function(args) {{ try {{ var r = __extractResult(await tools.EXECUTE_TOOL({{ tool_id: {id}, tool_arguments: args || {{}} }})); if (r == null) throw new Error(\"returned null/empty\"); return r; }} catch(e) {{ throw new Error(\"{safe}: \" + (e && e.message || e)); }} }}",
                safe = safe_name, id = id_json
            )
        })
        .collect();

    let preamble = format!(
        "{}\nvar codemode = {{\n{}\n}};\n",
        extract_helper,
        entries.join(",\n")
    );

    // Diagnostic: check for any remaining invalid JS identifiers as keys
    for line in preamble.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() && trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            tracing::warn!(
                "  [codemode] INVALID JS KEY (starts with digit): {}",
                &trimmed[..trimmed.len().min(60)]
            );
        }
    }

    preamble
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── json_schema_to_ts_type ───────────────────────────────────────

    #[test]
    fn test_string_type() {
        let schema = json!({"type": "string"});
        assert_eq!(json_schema_to_ts_type(&schema), "string");
    }

    #[test]
    fn test_number_type() {
        let schema = json!({"type": "number"});
        assert_eq!(json_schema_to_ts_type(&schema), "number");
    }

    #[test]
    fn test_integer_type() {
        let schema = json!({"type": "integer"});
        assert_eq!(json_schema_to_ts_type(&schema), "number");
    }

    #[test]
    fn test_boolean_type() {
        let schema = json!({"type": "boolean"});
        assert_eq!(json_schema_to_ts_type(&schema), "boolean");
    }

    #[test]
    fn test_array_of_strings() {
        let schema = json!({"type": "array", "items": {"type": "string"}});
        assert_eq!(json_schema_to_ts_type(&schema), "string[]");
    }

    #[test]
    fn test_array_without_items() {
        let schema = json!({"type": "array"});
        assert_eq!(json_schema_to_ts_type(&schema), "unknown[]");
    }

    #[test]
    fn test_object_with_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        });
        let result = json_schema_to_ts_type(&schema);
        assert!(result.contains("name: string"));
        assert!(result.contains("age?: number"));
    }

    #[test]
    fn test_object_without_properties() {
        let schema = json!({"type": "object"});
        assert_eq!(json_schema_to_ts_type(&schema), "object");
    }

    #[test]
    fn test_enum_strings() {
        let schema = json!({"enum": ["asc", "desc"]});
        assert_eq!(json_schema_to_ts_type(&schema), "\"asc\" | \"desc\"");
    }

    #[test]
    fn test_enum_mixed() {
        let schema = json!({"enum": ["a", 1, true, null]});
        assert_eq!(json_schema_to_ts_type(&schema), "\"a\" | 1 | true | null");
    }

    #[test]
    fn test_nullable_via_anyof() {
        let schema = json!({"anyOf": [{"type": "string"}, {"type": "null"}]});
        assert_eq!(json_schema_to_ts_type(&schema), "string | null");
    }

    #[test]
    fn test_unknown_type() {
        let schema = json!({});
        assert_eq!(json_schema_to_ts_type(&schema), "unknown");
    }

    #[test]
    fn test_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string"}
                    }
                }
            }
        });
        let result = json_schema_to_ts_type(&schema);
        assert!(result.contains("filter?:"));
        assert!(result.contains("state?: string"));
    }

    #[test]
    fn test_array_of_union() {
        let schema = json!({
            "type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}
        });
        assert_eq!(json_schema_to_ts_type(&schema), "(string | number)[]");
    }

    // ── generate_codemode_types ──────────────────────────────────────

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
    fn test_generate_single_tool() {
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

        let output = generate_codemode_types(&tools);

        assert!(output.contains("type Linear_list_projects_Input ="));
        assert!(output.contains("filter?: object"));
        assert!(output.contains("first?: number"));
        assert!(output.contains("declare const codemode:"));
        assert!(output.contains("/** List all projects */"));
        assert!(output.contains(
            "Linear_list_projects: (input?: Linear_list_projects_Input) => Promise<unknown>;"
        ));
    }

    #[test]
    fn test_generate_multiple_tools() {
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

        let output = generate_codemode_types(&tools);

        assert!(output.contains("type Linear_list_projects_Input ="));
        assert!(output.contains("type GitHub_get_repo_Input ="));
        assert!(output.contains("owner: string"));
        assert!(output.contains("repo: string"));
        assert!(output.contains("Linear_list_projects:"));
        assert!(output.contains("GitHub_get_repo:"));
    }

    #[test]
    fn test_generate_empty_tools() {
        let output = generate_codemode_types(&[]);
        assert!(output.contains("declare const codemode:"));
    }

    #[test]
    fn test_tool_with_no_schema_properties() {
        let tools = vec![make_tool("Ping", "ping", "Ping the server", json!({}))];

        let output = generate_codemode_types(&tools);
        assert!(output.contains("type Ping_ping_Input = object;"));
    }

    // ── generate_codemode_js_preamble ────────────────────────────────

    #[test]
    fn test_js_preamble_single_tool() {
        let tools = vec![make_tool(
            "Linear",
            "list_projects",
            "List all projects",
            json!({}),
        )];
        let output = generate_codemode_js_preamble(&tools);

        assert!(output.contains("function __extractResult(raw)"));
        assert!(output.contains("var codemode = {"));
        assert!(output.contains("Linear_list_projects: async function(args)"));
        assert!(output.contains("tool_id: \"Linear:list_projects\""));
        assert!(output.contains("__extractResult(await tools.EXECUTE_TOOL("));
        // Error handling: proxy wraps in try/catch with tool name
        assert!(output.contains("\"Linear_list_projects: \""));
        // Null guard
        assert!(output.contains("if (r == null) throw new Error"));
    }

    #[test]
    fn test_js_preamble_multiple_tools() {
        let tools = vec![
            make_tool("Linear", "list_projects", "List", json!({})),
            make_tool("GitHub", "get_repo", "Get repo", json!({})),
        ];
        let output = generate_codemode_js_preamble(&tools);

        assert!(output.contains("Linear_list_projects:"));
        assert!(output.contains("GitHub_get_repo:"));
    }

    #[test]
    fn test_js_preamble_sanitizes_server_names_with_spaces() {
        let tools = vec![make_tool(
            "Code Executor",
            "run-code",
            "Execute code",
            json!({}),
        )];
        let output = generate_codemode_js_preamble(&tools);

        // Spaces and hyphens replaced with underscores
        assert!(output.contains("Code_Executor_run_code: async function(args)"));
        // But tool_id stays original
        assert!(output.contains("tool_id: \"Code Executor:run-code\""));
    }

    #[test]
    fn test_js_preamble_empty() {
        let output = generate_codemode_js_preamble(&[]);
        assert!(output.contains("var codemode = {"));
        assert!(output.contains("function __extractResult(raw)"));
    }
}
