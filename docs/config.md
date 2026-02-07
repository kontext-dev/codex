# Configuration

For basic configuration instructions, see [this documentation](https://developers.openai.com/codex/config-basic).

For advanced configuration instructions, see [this documentation](https://developers.openai.com/codex/config-advanced).

For a full configuration reference, see [this documentation](https://developers.openai.com/codex/config-reference).

## Connecting to MCP servers

Codex can connect to MCP servers configured in `~/.codex/config.toml`. See the configuration reference for the latest MCP server options:

- https://developers.openai.com/codex/config-reference

### Kontext-Dev fork configuration

This fork also supports a top-level `[kontext-dev]` table. It authenticates with PKCE, exchanges an identity token for an `mcp-gateway` token, and injects a streamable HTTP MCP server automatically.

```toml
[kontext-dev]
server = "https://api.kontext.dev"
client_id = "<application-client-id>"
# optional for confidential clients
# client_secret = "<application-client-secret>"

# optional; defaults shown
scope = "openid offline"
resource = "mcp-gateway"
server_name = "kontext-dev"
auth_timeout_seconds = 300
open_connect_page_on_login = true
integration_ui_url = "https://app.kontext.dev"
# integration_return_to = "https://app.kontext.dev/oauth/complete"
# token_cache_path = "/Users/<you>/.codex/kontext-dev-token.json"
```

## Apps (Connectors)

Use `$` in the composer to insert a ChatGPT connector; the popover lists accessible
apps. The `/apps` command lists available and installed apps. Connected apps appear first
and are labeled as connected; others are marked as can be installed.

## Notify

Codex can run a notification hook when the agent finishes a turn. See the configuration reference for the latest notification settings:

- https://developers.openai.com/codex/config-reference

## JSON Schema

The generated JSON Schema for `config.toml` lives at `codex-rs/core/config.schema.json`.

## Notices

Codex stores "do not show again" flags for some UI prompts under the `[notice]` table.

Ctrl+C/Ctrl+D quitting uses a ~1 second double-press hint (`ctrl + c again to quit`).
