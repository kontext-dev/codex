use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use path_absolutize::Absolutize as _;

use codex_core::SandboxState;
use codex_core::exec::process_exec_tool_call;
use codex_core::sandboxing::SandboxPermissions;
use codex_protocol::config_types::WindowsSandboxLevel;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::posix::escalate_protocol::ESCALATE_SOCKET_ENV_VAR;
use crate::posix::escalate_protocol::EXEC_WRAPPER_ENV_VAR;
use crate::posix::escalate_protocol::EscalateAction;
use crate::posix::escalate_protocol::EscalateRequest;
use crate::posix::escalate_protocol::EscalateResponse;
use crate::posix::escalate_protocol::LEGACY_BASH_EXEC_WRAPPER_ENV_VAR;
use crate::posix::escalate_protocol::SuperExecMessage;
use crate::posix::escalate_protocol::SuperExecResult;
use crate::posix::escalation_policy::EscalationPolicy;
use crate::posix::mcp::ExecParams;
use crate::posix::socket::AsyncDatagramSocket;
use crate::posix::socket::AsyncSocket;
use codex_core::exec::ExecExpiration;

pub(crate) struct EscalateServer {
    bash_path: PathBuf,
    execve_wrapper: PathBuf,
    policy: Arc<dyn EscalationPolicy>,
}

impl EscalateServer {
    pub fn new<P>(bash_path: PathBuf, execve_wrapper: PathBuf, policy: P) -> Self
    where
        P: EscalationPolicy + Send + Sync + 'static,
    {
        Self {
            bash_path,
            execve_wrapper,
            policy: Arc::new(policy),
        }
    }

    pub async fn exec(
        &self,
        params: ExecParams,
        cancel_rx: CancellationToken,
        sandbox_state: &SandboxState,
    ) -> anyhow::Result<ExecResult> {
        let (escalate_server, escalate_client) = AsyncDatagramSocket::pair()?;
        let client_socket = escalate_client.into_inner();
        client_socket.set_cloexec(false)?;

        let escalate_task = tokio::spawn(escalate_task(escalate_server, self.policy.clone()));
        let mut env = std::env::vars().collect::<HashMap<String, String>>();
        env.insert(
            ESCALATE_SOCKET_ENV_VAR.to_string(),
            client_socket.as_raw_fd().to_string(),
        );
        env.insert(
            EXEC_WRAPPER_ENV_VAR.to_string(),
            self.execve_wrapper.to_string_lossy().to_string(),
        );
        env.insert(
            LEGACY_BASH_EXEC_WRAPPER_ENV_VAR.to_string(),
            self.execve_wrapper.to_string_lossy().to_string(),
        );

        let ExecParams {
            command,
            workdir,
            timeout_ms: _,
            login,
        } = params;
        let result = process_exec_tool_call(
            codex_core::exec::ExecParams {
                command: vec![
                    self.bash_path.to_string_lossy().to_string(),
                    if login == Some(false) {
                        "-c".to_string()
                    } else {
                        "-lc".to_string()
                    },
                    command,
                ],
                cwd: PathBuf::from(&workdir),
                expiration: ExecExpiration::Cancellation(cancel_rx),
                env,
                network: None,
                sandbox_permissions: SandboxPermissions::UseDefault,
                windows_sandbox_level: WindowsSandboxLevel::Disabled,
                justification: None,
                arg0: None,
            },
            &sandbox_state.sandbox_policy,
            &sandbox_state.sandbox_cwd,
            &sandbox_state.codex_linux_sandbox_exe,
            sandbox_state.use_linux_sandbox_bwrap,
            None,
        )
        .await?;
        escalate_task.abort();
        let result = ExecResult {
            exit_code: result.exit_code,
            output: result.aggregated_output.text,
            duration: result.duration,
            timed_out: result.timed_out,
        };
        Ok(result)
    }
}

async fn escalate_task(
    socket: AsyncDatagramSocket,
    policy: Arc<dyn EscalationPolicy>,
) -> anyhow::Result<()> {
    loop {
        let (_, mut fds) = socket.receive_with_fds().await?;
        if fds.len() != 1 {
            tracing::error!("expected 1 fd in datagram handshake, got {}", fds.len());
            continue;
        }
        let stream_socket = AsyncSocket::from_fd(fds.remove(0))?;
        let policy = policy.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_escalate_session_with_policy(stream_socket, policy).await {
                tracing::error!("escalate session failed: {err:?}");
            }
        });
    }
}

#[derive(Debug)]
pub(crate) struct ExecResult {
    pub(crate) exit_code: i32,
    pub(crate) output: String,
    pub(crate) duration: Duration,
    pub(crate) timed_out: bool,
}

async fn handle_escalate_session_with_policy(
    socket: AsyncSocket,
    policy: Arc<dyn EscalationPolicy>,
) -> anyhow::Result<()> {
    let EscalateRequest {
        file,
        argv,
        workdir,
        env,
    } = socket.receive::<EscalateRequest>().await?;
    let program = AbsolutePathBuf::resolve_path_against_base(file, workdir.as_path())?;
    let action = policy
        .determine_action(file.as_path(), &argv, &workdir)
        .await?;

    tracing::debug!("decided {action:?} for {program:?} {argv:?} {workdir:?}");

    match action {
        EscalateAction::Run => {
            socket
                .send(EscalateResponse {
                    action: EscalateAction::Run,
                })
                .await?;
        }
        EscalateAction::Escalate => {
            socket
                .send(EscalateResponse {
                    action: EscalateAction::Escalate,
                })
                .await?;
            let (msg, fds) = socket
                .receive_with_fds::<SuperExecMessage>()
                .await
                .context("failed to receive SuperExecMessage")?;
            if fds.len() != msg.fds.len() {
                return Err(anyhow::anyhow!(
                    "mismatched number of fds in SuperExecMessage: {} in the message, {} from the control message",
                    msg.fds.len(),
                    fds.len()
                ));
            }

            if msg
                .fds
                .iter()
                .any(|src_fd| fds.iter().any(|dst_fd| dst_fd.as_raw_fd() == *src_fd))
            {
                return Err(anyhow::anyhow!(
                    "overlapping fds not yet supported in SuperExecMessage"
                ));
            }

            let mut command = Command::new(program.as_path());
            command
                .args(&argv[1..])
                .arg0(argv[0].clone())
                .envs(&env)
                .current_dir(&workdir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            unsafe {
                command.pre_exec(move || {
                    for (dst_fd, src_fd) in msg.fds.iter().zip(&fds) {
                        libc::dup2(src_fd.as_raw_fd(), *dst_fd);
                    }
                    Ok(())
                });
            }
            let mut child = command.spawn()?;
            let exit_status = child.wait().await?;
            socket
                .send(SuperExecResult {
                    exit_code: exit_status.code().unwrap_or(127),
                })
                .await?;
        }
        EscalateAction::Deny { reason } => {
            socket
                .send(EscalateResponse {
                    action: EscalateAction::Deny { reason },
                })
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use pretty_assertions::assert_eq;
    use std::collections::HashMap;
    use std::path::PathBuf;

    struct DeterministicEscalationPolicy {
        action: EscalateAction,
    }

    #[async_trait::async_trait]
    impl EscalationPolicy for DeterministicEscalationPolicy {
        async fn determine_action(
            &self,
            _file: &AbsolutePathBuf,
            _argv: &[String],
            _workdir: &Path,
        ) -> Result<EscalateAction, rmcp::ErrorData> {
            Ok(self.action.clone())
        }
    }

    struct AssertingEscalationPolicy {
        expected_file: AbsolutePathBuf,
        expected_workdir: AbsolutePathBuf,
    }

    #[async_trait::async_trait]
    impl EscalationPolicy for AssertingEscalationPolicy {
        async fn determine_action(
            &self,
            file: &AbsolutePathBuf,
            _argv: &[String],
            workdir: &AbsolutePathBuf,
        ) -> anyhow::Result<EscalateAction> {
            assert_eq!(file, &self.expected_file);
            assert_eq!(workdir, &self.expected_workdir);
            Ok(EscalateAction::Run)
        }
    }

    #[tokio::test]
    async fn handle_escalate_session_respects_run_in_sandbox_decision() -> anyhow::Result<()> {
        let (server, client) = AsyncSocket::pair()?;
        let server_task = tokio::spawn(handle_escalate_session_with_policy(
            server,
            Arc::new(DeterministicEscalationPolicy {
                action: EscalateAction::Run,
            }),
        ));

        let mut env = HashMap::new();
        for i in 0..10 {
            let value = "A".repeat(1024);
            env.insert(format!("CODEX_TEST_VAR{i}"), value);
        }

        client
            .send(EscalateRequest {
                file: PathBuf::from("/bin/echo"),
                argv: vec!["echo".to_string()],
                workdir: AbsolutePathBuf::try_from(PathBuf::from("/tmp"))?,
                env,
            })
            .await?;

        let response = client.receive::<EscalateResponse>().await?;
        assert_eq!(
            EscalateResponse {
                action: EscalateAction::Run,
            },
            response
        );
        server_task.await?
    }

    #[tokio::test]
    async fn handle_escalate_session_resolves_relative_file_against_request_workdir()
    -> anyhow::Result<()> {
        let (server, client) = AsyncSocket::pair()?;
        let tmp = tempfile::TempDir::new()?;
        let workdir = tmp.path().join("workspace");
        std::fs::create_dir(&workdir)?;
        let workdir = AbsolutePathBuf::try_from(workdir)?;
        let expected_file = workdir.join("bin/tool")?;
        let server_task = tokio::spawn(handle_escalate_session_with_policy(
            server,
            Arc::new(AssertingEscalationPolicy {
                expected_file,
                expected_workdir: workdir.clone(),
            }),
        ));

        client
            .send(EscalateRequest {
                file: PathBuf::from("./bin/tool"),
                argv: vec!["./bin/tool".to_string()],
                workdir,
                env: HashMap::new(),
            })
            .await?;

        let response = client.receive::<EscalateResponse>().await?;
        assert_eq!(
            EscalateResponse {
                action: EscalateAction::Run,
            },
            response
        );
        server_task.await?
    }

    #[tokio::test]
    async fn handle_escalate_session_executes_escalated_command() -> anyhow::Result<()> {
        let (server, client) = AsyncSocket::pair()?;
        let server_task = tokio::spawn(handle_escalate_session_with_policy(
            server,
            Arc::new(DeterministicEscalationPolicy {
                action: EscalateAction::Escalate,
            }),
        ));

        client
            .send(EscalateRequest {
                file: PathBuf::from("/bin/sh"),
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    r#"if [ "$KEY" = VALUE ]; then exit 42; else exit 1; fi"#.to_string(),
                ],
                workdir: AbsolutePathBuf::current_dir()?,
                env: HashMap::from([("KEY".to_string(), "VALUE".to_string())]),
            })
            .await?;

        let response = client.receive::<EscalateResponse>().await?;
        assert_eq!(
            EscalateResponse {
                action: EscalateAction::Escalate,
            },
            response
        );

        client
            .send_with_fds(SuperExecMessage { fds: Vec::new() }, &[])
            .await?;

        let result = client.receive::<SuperExecResult>().await?;
        assert_eq!(42, result.exit_code);

        server_task.await?
    }
}
