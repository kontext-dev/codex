use async_trait::async_trait;
use serde_json::Map;
use serde_json::Value;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub struct KontextDevHandler;

#[async_trait]
impl ToolHandler for KontextDevHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<FunctionToolOutput, FunctionCallError> {
        let ToolInvocation {
            session,
            tool_name,
            payload,
            ..
        } = invocation;

        let ToolPayload::Function { arguments } = payload else {
            return Err(FunctionCallError::RespondToModel(
                "Kontext handler received unsupported payload".to_string(),
            ));
        };

        let args = parse_function_arguments(arguments.as_str())?;

        let Some(runtime) = session.services.kontext_dev_runtime.clone() else {
            return Err(FunctionCallError::RespondToModel(
                "Kontext runtime is not configured for this session.".to_string(),
            ));
        };

        let output = runtime
            .execute_tool(tool_name.as_str(), args)
            .await
            .map_err(|err| {
                FunctionCallError::RespondToModel(format!(
                    "Kontext tool `{tool_name}` failed: {err}"
                ))
            })?;

        Ok(FunctionToolOutput::from_text(output, Some(true)))
    }
}

fn parse_function_arguments(arguments: &str) -> Result<Map<String, Value>, FunctionCallError> {
    let parsed = serde_json::from_str::<Value>(arguments).map_err(|err| {
        FunctionCallError::RespondToModel(format!("failed to parse function arguments: {err}"))
    })?;

    match parsed {
        Value::Object(map) => Ok(map),
        Value::Null => Ok(Map::new()),
        _ => Err(FunctionCallError::RespondToModel(
            "function arguments must be a JSON object".to_string(),
        )),
    }
}
