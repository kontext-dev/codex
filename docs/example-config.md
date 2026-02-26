# Sample configuration

For a sample configuration file, see [this documentation](https://developers.openai.com/codex/config-sample).

This fork does not require a `[kontext-dev]` block anymore.

Kontext is configured with baked defaults in the binary:

- `client_id`: baked into the fork build
- `redirect_uri`: `http://localhost:3333/callback`
- `server`: `https://api.kontext.dev/mcp`
- `resource`: `mcp-gateway`
- `integration_ui_url`: `https://app.kontext.dev`
- `open_connect_page_on_login`: `true`

On first run, Codex handles normal login, then Kontext PKCE and integration connect flow run automatically.
