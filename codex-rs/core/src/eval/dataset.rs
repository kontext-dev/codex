//! MCP-Atlas Dataset Loader
//!
//! Loads tasks from Arrow IPC format or CSV format datasets.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use arrow::array::Array;
use arrow::array::StringArray;
use arrow::ipc::reader::StreamReader;

/// A single task from the MCP-Atlas dataset
#[derive(Debug, Clone)]
pub struct McpAtlasTask {
    /// Task identifier
    pub task_id: String,
    /// Tools to expose for this task (parsed from JSON)
    pub enabled_tools: Vec<String>,
    /// Natural language request
    pub prompt: String,
    /// Ground truth claims for evaluation (parsed from JSON)
    pub claims: Vec<String>,
    /// Reference solution trajectory (for diagnostics)
    pub trajectory: Vec<TrajectoryStep>,
}

/// A step in the reference trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// Tool called
    pub tool: String,
    /// Arguments passed
    pub args: serde_json::Value,
    /// Expected result (optional)
    pub result: Option<String>,
}

/// Load the MCP-Atlas dataset from an Arrow IPC file or CSV file
///
/// # Arguments
///
/// * `path` - Path to the dataset file (.arrow or .csv)
///
/// # Returns
///
/// Vector of parsed tasks
pub fn load_dataset(path: impl AsRef<Path>) -> Result<Vec<McpAtlasTask>> {
    let path = path.as_ref();

    // Check file extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "csv" => load_csv_dataset(path),
        "arrow" => load_arrow_dataset(path),
        _ => {
            // Try Arrow first, fall back to CSV
            load_arrow_dataset(path).or_else(|_| load_csv_dataset(path))
        }
    }
}

/// Load dataset from CSV format
fn load_csv_dataset(path: &Path) -> Result<Vec<McpAtlasTask>> {
    let file = File::open(path).with_context(|| format!("Failed to open CSV at {:?}", path))?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::new();
    let mut lines = reader.lines();

    // Read header
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty CSV file"))?
        .with_context(|| "Failed to read header")?;

    let columns: Vec<&str> = header.split(',').collect();

    // Find column indices
    let task_idx = columns
        .iter()
        .position(|c| *c == "TASK")
        .ok_or_else(|| anyhow::anyhow!("TASK column not found"))?;
    let tools_idx = columns
        .iter()
        .position(|c| *c == "ENABLED_TOOLS")
        .ok_or_else(|| anyhow::anyhow!("ENABLED_TOOLS column not found"))?;
    let prompt_idx = columns
        .iter()
        .position(|c| *c == "PROMPT")
        .ok_or_else(|| anyhow::anyhow!("PROMPT column not found"))?;
    let claims_idx = columns
        .iter()
        .position(|c| *c == "GTFA_CLAIMS")
        .ok_or_else(|| anyhow::anyhow!("GTFA_CLAIMS column not found"))?;
    let trajectory_idx = columns
        .iter()
        .position(|c| *c == "TRAJECTORY")
        .ok_or_else(|| anyhow::anyhow!("TRAJECTORY column not found"))?;

    // Parse rows
    for (row_num, line_result) in lines.enumerate() {
        let line = line_result.with_context(|| format!("Failed to read line {}", row_num + 2))?;

        // Parse CSV line (handling quoted fields)
        let fields = parse_csv_line(&line);

        if fields.len() <= trajectory_idx.max(claims_idx).max(prompt_idx).max(tools_idx).max(task_idx) {
            continue; // Skip malformed rows
        }

        let task_id = fields[task_idx].clone();
        let enabled_tools = parse_json_string_array(&fields[tools_idx])
            .with_context(|| format!("Failed to parse ENABLED_TOOLS for task {} (row {})", task_id, row_num + 2))?;
        let prompt = fields[prompt_idx].clone();
        let claims = parse_json_string_array(&fields[claims_idx])
            .with_context(|| format!("Failed to parse GTFA_CLAIMS for task {} (row {})", task_id, row_num + 2))?;
        let trajectory = parse_trajectory(&fields[trajectory_idx])
            .with_context(|| format!("Failed to parse TRAJECTORY for task {} (row {})", task_id, row_num + 2))?;

        tasks.push(McpAtlasTask {
            task_id,
            enabled_tools,
            prompt,
            claims,
            trajectory,
        });
    }

    Ok(tasks)
}

/// Parse a CSV line handling quoted fields
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quotes => {
                in_quotes = true;
            }
            '"' if in_quotes => {
                // Check for escaped quote
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    fields.push(current);
    fields
}

/// Load dataset from Arrow IPC format
fn load_arrow_dataset(path: &Path) -> Result<Vec<McpAtlasTask>> {
    let file = File::open(path).with_context(|| format!("Failed to open dataset at {:?}", path))?;
    let buf_reader = BufReader::new(file);

    let reader =
        StreamReader::try_new(buf_reader, None).with_context(|| "Failed to create Arrow stream reader")?;

    let mut tasks = Vec::new();

    for batch_result in reader {
        let batch = batch_result.with_context(|| "Failed to read record batch")?;

        // Get column indices
        let schema = batch.schema();
        let task_idx = schema
            .index_of("TASK")
            .with_context(|| "TASK column not found")?;
        let tools_idx = schema
            .index_of("ENABLED_TOOLS")
            .with_context(|| "ENABLED_TOOLS column not found")?;
        let prompt_idx = schema
            .index_of("PROMPT")
            .with_context(|| "PROMPT column not found")?;
        let claims_idx = schema
            .index_of("GTFA_CLAIMS")
            .with_context(|| "GTFA_CLAIMS column not found")?;
        let trajectory_idx = schema
            .index_of("TRAJECTORY")
            .with_context(|| "TRAJECTORY column not found")?;

        // Get arrays
        let task_array = batch
            .column(task_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| "TASK column is not a string array")?;
        let tools_array = batch
            .column(tools_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| "ENABLED_TOOLS column is not a string array")?;
        let prompt_array = batch
            .column(prompt_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| "PROMPT column is not a string array")?;
        let claims_array = batch
            .column(claims_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| "GTFA_CLAIMS column is not a string array")?;
        let trajectory_array = batch
            .column(trajectory_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| "TRAJECTORY column is not a string array")?;

        // Parse each row
        for i in 0..batch.num_rows() {
            let task_id = task_array.value(i).to_string();
            let enabled_tools = parse_json_string_array(tools_array.value(i))
                .with_context(|| format!("Failed to parse ENABLED_TOOLS for task {} (row {})", task_id, i))?;
            let prompt = prompt_array.value(i).to_string();
            let claims = parse_json_string_array(claims_array.value(i))
                .with_context(|| format!("Failed to parse GTFA_CLAIMS for task {} (row {})", task_id, i))?;
            let trajectory = parse_trajectory(trajectory_array.value(i))
                .with_context(|| format!("Failed to parse TRAJECTORY for task {} (row {})", task_id, i))?;

            tasks.push(McpAtlasTask {
                task_id,
                enabled_tools,
                prompt,
                claims,
                trajectory,
            });
        }
    }

    Ok(tasks)
}

/// Parse a string array field (supports both JSON and Python list syntax)
fn parse_json_string_array(json_str: &str) -> Result<Vec<String>> {
    if json_str.is_empty() {
        return Ok(Vec::new());
    }

    // First try JSON parsing
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
        return match parsed {
            serde_json::Value::Array(arr) => {
                let strings: Vec<String> = arr
                    .into_iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        _ => v.as_str().map(String::from),
                    })
                    .collect();
                Ok(strings)
            }
            serde_json::Value::String(s) => {
                // Sometimes claims are stored as newline-separated strings
                Ok(s.lines().map(String::from).collect())
            }
            _ => Ok(Vec::new()),
        };
    }

    // Try parsing Python list format (single quotes)
    if json_str.starts_with('[') && json_str.ends_with(']') {
        return parse_python_list(json_str);
    }

    // Fallback: treat as newline-separated
    Ok(json_str.lines().map(String::from).collect())
}

/// Parse Python list format with single quotes
fn parse_python_list(s: &str) -> Result<Vec<String>> {
    let inner = &s[1..s.len() - 1]; // Remove [ and ]
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escape_next = false;
    let mut quote_char = '\'';

    for ch in inner.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => {
                escape_next = true;
                current.push(ch);
            }
            '\'' | '"' if !in_string => {
                in_string = true;
                quote_char = ch;
            }
            c if c == quote_char && in_string => {
                in_string = false;
                results.push(current.replace("\\n", "\n").replace("\\'", "'"));
                current = String::new();
            }
            ',' if !in_string => {
                // Skip commas between elements
            }
            ' ' if !in_string => {
                // Skip whitespace between elements
            }
            _ => {
                if in_string {
                    current.push(ch);
                }
            }
        }
    }

    Ok(results)
}

/// Parse trajectory JSON into TrajectoryStep structs
fn parse_trajectory(json_str: &str) -> Result<Vec<TrajectoryStep>> {
    if json_str.is_empty() {
        return Ok(Vec::new());
    }

    let parsed: serde_json::Value =
        serde_json::from_str(json_str).with_context(|| "Failed to parse trajectory JSON")?;

    match parsed {
        serde_json::Value::Array(arr) => {
            let steps: Vec<TrajectoryStep> = arr
                .into_iter()
                .filter_map(|v| {
                    let obj = v.as_object()?;
                    let tool = obj
                        .get("tool")
                        .or_else(|| obj.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from)?;
                    let args = obj
                        .get("args")
                        .or_else(|| obj.get("arguments"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    let result = obj
                        .get("result")
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    Some(TrajectoryStep { tool, args, result })
                })
                .collect();
            Ok(steps)
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_string_array() {
        let json = r#"["tool1", "tool2", "tool3"]"#;
        let result = parse_json_string_array(json).unwrap();
        assert_eq!(result, vec!["tool1", "tool2", "tool3"]);
    }

    #[test]
    fn test_parse_json_string_array_empty() {
        let result = parse_json_string_array("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_trajectory() {
        let json = r#"[{"tool": "git", "args": {"command": "status"}}, {"tool": "cli", "args": {}}]"#;
        let result = parse_trajectory(json).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tool, "git");
        assert_eq!(result[1].tool, "cli");
    }
}
