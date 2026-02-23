# Sample configuration

For a sample configuration file, see [this documentation](https://developers.openai.com/codex/config-sample).

For this fork, add the following block to `~/.codex/config.toml` to enable Kontext-Dev:

```toml
[kontext-dev]
server = "https://api.kontext.dev"
client_id = "<application-client-id>"

# optional values (defaults shown)
# leave empty unless your app explicitly allows OAuth scopes
scope = ""
resource = "mcp-gateway"
server_name = "kontext-dev"
open_connect_page_on_login = true
# redirect_uri must match your OAuth app configuration
redirect_uri = "http://localhost:3000/callback"
```

With this configuration, Kontext tools are added directly to the normal Codex tool list (they do not appear under MCP server inventory).
