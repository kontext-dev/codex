# Sample configuration

For a sample configuration file, see [this documentation](https://developers.openai.com/codex/config-sample).

This fork does not require a `[kontext-dev]` block anymore.

Kontext is configured with baked defaults in the binary:

- `client_id`: baked into the fork build
- `redirect_uri`: `http://localhost:3333/callback`
- `server`: `http://localhost:4000/mcp`
- `resource`: `mcp-gateway`
- `integration_ui_url`: `http://localhost:3000`
- `open_connect_page_on_login`: `true`

Flow rule: local gateway handles API/MCP/auth endpoints on `localhost:4000`,
and integrations connect runs on local web `localhost:3000`. Do not expect
`http://localhost:4000/oauth/connect` to exist.

On first run, Codex handles normal login, then Kontext PKCE and integration connect flow run automatically.
