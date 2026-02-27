use std::collections::HashMap;
use std::os::fd::RawFd;
use std::path::PathBuf;

use codex_protocol::approvals::EscalationPermissions;
use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Deserialize;
use serde::Serialize;

/// 'exec-server escalate' reads this to find the inherited FD for the escalate socket.
pub(super) const ESCALATE_SOCKET_ENV_VAR: &str = "CODEX_ESCALATE_SOCKET";

/// Patched shells use this to wrap exec() calls.
pub(super) const EXEC_WRAPPER_ENV_VAR: &str = "EXEC_WRAPPER";

/// Compatibility alias for older patched bash builds.
pub(super) const LEGACY_BASH_EXEC_WRAPPER_ENV_VAR: &str = "BASH_EXEC_WRAPPER";

/// The client sends this to the server to request an exec() call.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub(super) struct EscalateRequest {
    /// The absolute path to the executable to run, i.e. the first arg to exec.
    pub(super) file: PathBuf,
    /// The argv, including the program name (argv[0]).
    pub(super) argv: Vec<String>,
    pub(super) workdir: PathBuf,
    pub(super) env: HashMap<String, String>,
}

/// The server sends this to the client to respond to an exec() request.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub(super) struct EscalateResponse {
    pub(super) action: EscalateAction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EscalationDecision {
    Run,
    Escalate(EscalationExecution),
    Deny { reason: Option<String> },
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EscalationExecution {
    /// Rerun the intercepted command outside any sandbox wrapper.
    Unsandboxed,
    /// Rerun using the turn's current sandbox configuration.
    TurnDefault,
    /// Rerun using an explicit sandbox configuration attached to the request.
    Permissions(EscalationPermissions),
}

impl EscalationDecision {
    pub fn run() -> Self {
        Self::Run
    }

    pub fn escalate(execution: EscalationExecution) -> Self {
        Self::Escalate(execution)
    }

    pub fn deny(reason: Option<String>) -> Self {
        Self::Deny { reason }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub(super) enum EscalateAction {
    /// The command should be run directly by the client.
    Run,
    /// The command should be escalated to the server for execution.
    Escalate,
    /// The command should not be executed.
    Deny { reason: Option<String> },
}

/// The client sends this to the server to forward its open FDs.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub(super) struct SuperExecMessage {
    pub(super) fds: Vec<RawFd>,
}

/// The server responds when the exec()'d command has exited.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub(super) struct SuperExecResult {
    pub(super) exit_code: i32,
}
