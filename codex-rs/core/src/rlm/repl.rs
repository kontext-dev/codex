//! Local REPL execution environment for true RLM mode.
//!
//! Manages a persistent Python subprocess that maintains state across
//! code executions. Communication uses a line-delimited JSON protocol
//! over stdin/stdout. The Python side provides helper functions:
//! - `llm_query(prompt, model=None)` — sub-LLM calls via HTTP
//! - `llm_query_batched(prompts, model=None)` — batched sub-LLM calls
//! - `FINAL_VAR(name)` — mark a variable as the final answer
//! - `SHOW_VARS()` — list all user-defined variables
//! - `print()` — standard output (captured)

use anyhow::Context;
use anyhow::Result;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;

/// Result of executing a code block in the REPL.
#[derive(Debug, Clone, Default)]
pub struct ReplResult {
    /// Captured stdout from the execution.
    pub stdout: String,
    /// Captured stderr from the execution.
    pub stderr: String,
    /// Summary of local variables after execution.
    pub locals_summary: String,
    /// Wall-clock execution time in milliseconds.
    pub execution_time_ms: u64,
}

/// A persistent Python REPL subprocess.
pub struct LocalRepl {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    #[allow(dead_code)]
    lm_handler_port: u16,
}

impl LocalRepl {
    /// Spawn a new Python REPL subprocess.
    ///
    /// The `lm_handler_port` is the local HTTP port where the LM handler
    /// listens for sub-LLM query requests from the Python code.
    pub async fn new(lm_handler_port: u16) -> Result<Self> {
        let script = repl_script(lm_handler_port);

        let mut child = Command::new("python3")
            .arg("-u") // unbuffered
            .arg("-c")
            .arg(&script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("Failed to spawn Python REPL process")?;

        let stdin = child.stdin.take().context("Failed to get REPL stdin")?;
        let stdout_raw = child.stdout.take().context("Failed to get REPL stdout")?;
        let stdout = BufReader::new(stdout_raw);

        tracing::debug!("Spawned Python REPL subprocess (lm_handler_port={lm_handler_port})");

        Ok(Self {
            child,
            stdin,
            stdout,
            lm_handler_port,
        })
    }

    /// Execute a code block in the REPL and return the result.
    pub async fn execute(&mut self, code: &str) -> Result<ReplResult> {
        let cmd = serde_json::json!({
            "cmd": "execute",
            "code": code,
        });
        self.send_command(&cmd).await?;
        self.read_result().await
    }

    /// Set the `context` variable in the REPL namespace.
    pub async fn set_context(&mut self, context: &str) -> Result<()> {
        let cmd = serde_json::json!({
            "cmd": "set_context",
            "context": context,
        });
        self.send_command(&cmd).await?;
        let result = self.read_result().await?;
        if !result.stderr.is_empty() {
            tracing::debug!("set_context stderr: {}", result.stderr);
        }
        Ok(())
    }

    /// Resolve a variable name to its string value in the REPL namespace.
    pub async fn resolve_var(&mut self, name: &str) -> Result<String> {
        let cmd = serde_json::json!({
            "cmd": "resolve_var",
            "name": name,
        });
        self.send_command(&cmd).await?;
        let result = self.read_result().await?;
        Ok(result.stdout)
    }

    /// Gracefully shut down the Python process.
    pub async fn cleanup(&mut self) {
        let cmd = serde_json::json!({"cmd": "shutdown"});
        // Best-effort: ignore errors during shutdown
        let _ = self.send_command(&cmd).await;
        let _ = self.child.kill().await;
        tracing::debug!("REPL subprocess cleaned up");
    }

    /// Send a JSON command to the Python process (one line).
    async fn send_command(&mut self, cmd: &serde_json::Value) -> Result<()> {
        let mut line = serde_json::to_string(cmd)?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .await
            .context("Failed to write to REPL stdin")?;
        self.stdin
            .flush()
            .await
            .context("Failed to flush REPL stdin")?;
        Ok(())
    }

    /// Read a single JSON response line from the Python process.
    async fn read_result(&mut self) -> Result<ReplResult> {
        let mut line = String::new();
        let bytes_read = self
            .stdout
            .read_line(&mut line)
            .await
            .context("Failed to read from REPL stdout")?;

        if bytes_read == 0 {
            anyhow::bail!("REPL process closed stdout (process may have crashed)");
        }

        let resp: serde_json::Value =
            serde_json::from_str(line.trim()).context("Invalid JSON from REPL")?;

        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            Ok(ReplResult {
                stdout: resp
                    .get("stdout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                stderr: resp
                    .get("stderr")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                locals_summary: resp
                    .get("locals")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                execution_time_ms: resp
                    .get("time_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            })
        } else {
            let error = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error")
                .to_string();
            // Return the error in stderr rather than failing the whole call,
            // because runtime errors in user code are expected.
            Ok(ReplResult {
                stdout: String::new(),
                stderr: error,
                locals_summary: String::new(),
                execution_time_ms: 0,
            })
        }
    }
}

impl Drop for LocalRepl {
    fn drop(&mut self) {
        // Best-effort synchronous kill. The async cleanup should be preferred.
        let _ = self.child.start_kill();
    }
}

/// Generate the embedded Python REPL script.
///
/// The script reads line-delimited JSON from stdin, executes code in a
/// persistent namespace, and writes JSON results to stdout. Uses only
/// Python stdlib modules.
fn repl_script(lm_handler_port: u16) -> String {
    format!(
        r##"
import json
import sys
import io
import traceback
import time
import urllib.request

# ---------- configuration ----------
LM_HANDLER_PORT = {port}

# ---------- persistent namespace ----------
_globals = {{"__builtins__": {{}}}}
_locals = {{}}

# ---------- restricted builtins ----------
import builtins as _builtins

_ALLOWED_BUILTINS = {{
    "abs": _builtins.abs,
    "all": _builtins.all,
    "any": _builtins.any,
    "bool": _builtins.bool,
    "bytes": _builtins.bytes,
    "chr": _builtins.chr,
    "dict": _builtins.dict,
    "dir": _builtins.dir,
    "divmod": _builtins.divmod,
    "enumerate": _builtins.enumerate,
    "filter": _builtins.filter,
    "float": _builtins.float,
    "format": _builtins.format,
    "frozenset": _builtins.frozenset,
    "getattr": _builtins.getattr,
    "hasattr": _builtins.hasattr,
    "hash": _builtins.hash,
    "hex": _builtins.hex,
    "id": _builtins.id,
    "int": _builtins.int,
    "isinstance": _builtins.isinstance,
    "issubclass": _builtins.issubclass,
    "iter": _builtins.iter,
    "len": _builtins.len,
    "list": _builtins.list,
    "map": _builtins.map,
    "max": _builtins.max,
    "min": _builtins.min,
    "next": _builtins.next,
    "oct": _builtins.oct,
    "open": _builtins.open,
    "ord": _builtins.ord,
    "pow": _builtins.pow,
    "print": _builtins.print,
    "range": _builtins.range,
    "repr": _builtins.repr,
    "reversed": _builtins.reversed,
    "round": _builtins.round,
    "set": _builtins.set,
    "slice": _builtins.slice,
    "sorted": _builtins.sorted,
    "str": _builtins.str,
    "sum": _builtins.sum,
    "super": _builtins.super,
    "tuple": _builtins.tuple,
    "type": _builtins.type,
    "vars": _builtins.vars,
    "zip": _builtins.zip,
    "__import__": _builtins.__import__,
    "KeyError": _builtins.KeyError,
    "ValueError": _builtins.ValueError,
    "TypeError": _builtins.TypeError,
    "IndexError": _builtins.IndexError,
    "AttributeError": _builtins.AttributeError,
    "RuntimeError": _builtins.RuntimeError,
    "StopIteration": _builtins.StopIteration,
    "Exception": _builtins.Exception,
    "True": True,
    "False": False,
    "None": None,
}}

_globals["__builtins__"] = _ALLOWED_BUILTINS

# ---------- helper functions injected into namespace ----------

def _llm_query(prompt, model=None):
    """Send a sub-LLM query to the LM handler."""
    payload = json.dumps({{"prompt": prompt, "model": model}}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:{port}/llm_query",
        data=payload,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "")
    except Exception as e:
        return f"[llm_query error: {{e}}]"

def _llm_query_batched(prompts, model=None):
    """Send batched sub-LLM queries to the LM handler."""
    payload = json.dumps({{"prompts": prompts, "model": model}}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:{port}/llm_query_batched",
        data=payload,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode())
            return body.get("responses", [])
    except Exception as e:
        return [f"[llm_query_batched error: {{e}}]"]

def _final_var(name):
    """Mark a variable as the final answer."""
    if name in _locals:
        return str(_locals[name])
    raise KeyError(f"Variable '{{name}}' not found in namespace")

def _show_vars():
    """List all user-defined variables."""
    items = []
    for k, v in sorted(_locals.items()):
        if not k.startswith("_"):
            val_repr = repr(v)
            if len(val_repr) > 200:
                val_repr = val_repr[:200] + "..."
            items.append(f"  {{k}} = {{val_repr}}")
    return "\n".join(items) if items else "(no variables)"

_locals["llm_query"] = _llm_query
_locals["llm_query_batched"] = _llm_query_batched
_locals["FINAL_VAR"] = _final_var
_locals["SHOW_VARS"] = _show_vars

# ---------- response helper ----------

def _respond(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

# ---------- locals summary ----------

def _locals_summary():
    items = []
    for k, v in sorted(_locals.items()):
        if k.startswith("_"):
            continue
        if callable(v) and k in ("llm_query", "llm_query_batched", "FINAL_VAR", "SHOW_VARS"):
            continue
        val_repr = repr(v)
        if len(val_repr) > 200:
            val_repr = val_repr[:200] + "..."
        items.append(f"{{k}} = {{val_repr}}")
    return "; ".join(items)

# ---------- command loop ----------

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue

    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        _respond({{"ok": False, "error": f"Invalid JSON: {{e}}"}})
        continue

    cmd = msg.get("cmd", "")

    if cmd == "execute":
        code = msg.get("code", "")
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        t0 = time.time()
        try:
            exec(code, _globals, _locals)
        except Exception:
            traceback.print_exc(file=stderr_capture)
        elapsed_ms = int((time.time() - t0) * 1000)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        _respond({{
            "ok": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "locals": _locals_summary(),
            "time_ms": elapsed_ms,
        }})

    elif cmd == "set_context":
        ctx = msg.get("context", "")
        _locals["context"] = ctx
        _respond({{"ok": True, "stdout": "", "stderr": "", "locals": "", "time_ms": 0}})

    elif cmd == "resolve_var":
        name = msg.get("name", "")
        if name in _locals:
            _respond({{
                "ok": True,
                "stdout": str(_locals[name]),
                "stderr": "",
                "locals": "",
                "time_ms": 0,
            }})
        else:
            _respond({{
                "ok": False,
                "error": f"Variable '{{name}}' not found",
            }})

    elif cmd == "shutdown":
        _respond({{"ok": True, "stdout": "", "stderr": "", "locals": "", "time_ms": 0}})
        break

    else:
        _respond({{"ok": False, "error": f"Unknown command: {{cmd}}"}})
"##,
        port = lm_handler_port
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repl_basic_execution() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        let result = repl.execute("x = 2 + 3\nprint(x)").await.unwrap();
        assert_eq!(result.stdout.trim(), "5");
        assert!(result.stderr.is_empty());
        assert!(result.locals_summary.contains("x = 5"));

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_persistent_state() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        repl.execute("a = 10").await.unwrap();
        repl.execute("b = 20").await.unwrap();
        let result = repl.execute("print(a + b)").await.unwrap();
        assert_eq!(result.stdout.trim(), "30");

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_set_context() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        repl.set_context("Hello, world!").await.unwrap();
        let result = repl.execute("print(context)").await.unwrap();
        assert_eq!(result.stdout.trim(), "Hello, world!");

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_resolve_var() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        repl.execute("answer = 42").await.unwrap();
        let value = repl.resolve_var("answer").await.unwrap();
        assert_eq!(value, "42");

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_resolve_missing_var() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        let value = repl.resolve_var("nonexistent").await.unwrap();
        // Should return empty string (error mapped to stderr in read_result)
        assert!(value.is_empty());

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_error_handling() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        let result = repl.execute("1 / 0").await.unwrap();
        assert!(result.stderr.contains("ZeroDivisionError"));
        assert!(result.stdout.is_empty());

        // REPL should still work after an error
        let result = repl.execute("print('ok')").await.unwrap();
        assert_eq!(result.stdout.trim(), "ok");

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_restricted_builtins() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        // eval should not be available
        let result = repl.execute("eval('1+1')").await.unwrap();
        assert!(result.stderr.contains("NameError") || result.stderr.contains("name 'eval'"));

        // But len should work
        let result = repl.execute("print(len([1,2,3]))").await.unwrap();
        assert_eq!(result.stdout.trim(), "3");

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_imports() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        let result = repl.execute("import json\nprint(json.dumps({'a': 1}))").await.unwrap();
        assert_eq!(result.stdout.trim(), r#"{"a": 1}"#);

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_show_vars() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        repl.execute("x = 1\ny = 'hello'").await.unwrap();
        let result = repl.execute("print(SHOW_VARS())").await.unwrap();
        assert!(result.stdout.contains("x = 1"));
        assert!(result.stdout.contains("y = 'hello'"));

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_execution_time() {
        let mut repl = LocalRepl::new(0).await.unwrap();

        let result = repl
            .execute("import time\ntime.sleep(0.05)")
            .await
            .unwrap();
        assert!(result.execution_time_ms >= 40);

        repl.cleanup().await;
    }

    #[tokio::test]
    async fn test_repl_shutdown() {
        let mut repl = LocalRepl::new(0).await.unwrap();
        repl.cleanup().await;
        // Should not panic — cleanup is idempotent
    }
}
