# Kontext Codex CLI

`@kontext-dev/codex` is the Kontext-maintained Codex CLI fork.

## 1. Install

Create a GitHub Personal Access Token (classic) with `read:packages`.

Run this once:

```bash
npm config set @kontext-dev:registry https://npm.pkg.github.com && npm config set //npm.pkg.github.com/:_authToken YOUR_GITHUB_PAT
```

Install globally:

```bash
npm i -g @kontext-dev/codex@latest
```

## 2. Run

```bash
codex-kontext
```

On first run:

1. Sign in to Codex when prompted.
2. If needed, a browser page opens for Kontext auth/integrations.
3. After success, return to terminal.

## 3. Upgrade

```bash
npm i -g @kontext-dev/codex@latest
```

## 4. Verify

```bash
codex-kontext --version
```

## Built-in Kontext Defaults

- `client_id`: baked into this fork build
- `redirect_uri`: `http://localhost:3333/callback`
- `server`: `https://api.kontext.dev/mcp`
- `resource`: `mcp-gateway`
- `integration_ui_url`: `https://app.kontext.dev`
- `open_connect_page_on_login`: `true`

No `[kontext-dev]` block is required in `~/.codex/config.toml`.

## Troubleshooting

If install fails with `404 Not Found` on `registry.npmjs.org`, re-run:

```bash
npm config set @kontext-dev:registry https://npm.pkg.github.com
```

If install fails with `EEXIST ... codex-kontext`, run:

```bash
npm i -g @kontext-dev/codex@latest --force
```

## Versioning

Versions follow `<upstream_version>-kontext.<N>`, for example `0.105.0-kontext.3`.

## Docs

- [Codex Documentation](https://developers.openai.com/codex/)
- [Installing and Building](https://github.com/kontext-dev/codex/blob/kontext-dev/INSTALL.md)
- [Contributing](https://github.com/kontext-dev/codex/blob/kontext-dev/CONTRIBUTING.md)

Licensed under Apache-2.0.
