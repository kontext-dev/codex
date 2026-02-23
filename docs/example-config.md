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
```
