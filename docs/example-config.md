# Sample configuration

For a sample configuration file, see [this documentation](https://developers.openai.com/codex/config-sample).

For this fork, add the following block to `~/.codex/config.toml` to enable Kontext-Dev:

```toml
[kontext-dev]
client_id = "<application-client-id>"
redirect_uri = "http://localhost:3000/callback"
```

Optional values (defaults shown):

```toml
[kontext-dev]
client_id = "<application-client-id>"
redirect_uri = "http://localhost:3000/callback"

server = "https://api.kontext.dev"
scope = ""
resource = "mcp-gateway"
server_name = "kontext-dev"
auth_timeout_seconds = 300
open_connect_page_on_login = true
integration_ui_url = "https://app.kontext.dev"
# integration_return_to = "https://app.kontext.dev/oauth/complete"
# token_cache_path = "/Users/<you>/.codex/kontext-dev-token.json"
```

With this configuration, Kontext tools are added directly to the normal Codex tool list (they do not appear under MCP server inventory).
