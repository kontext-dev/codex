use codex_utils_absolute_path::AbsolutePathBuf;

use crate::posix::escalate_protocol::EscalateAction;

/// Decides what action to take in response to an execve request from a client.
#[async_trait::async_trait]
pub(crate) trait EscalationPolicy: Send + Sync {
    async fn determine_action(
        &self,
        file: &AbsolutePathBuf,
        argv: &[String],
        workdir: &Path,
    ) -> Result<EscalateAction, rmcp::ErrorData>;
}
