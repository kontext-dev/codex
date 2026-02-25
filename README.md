<p align="center"><code>npm i -g @kontext-dev/codex</code></p>
<p align="center"><strong>Codex CLI</strong> is a coding agent from OpenAI that runs locally on your computer.
<p align="center">
  <img src="https://github.com/openai/codex/blob/main/.github/codex-cli-splash.png" alt="Codex CLI splash" width="80%" />
</p>
</br>
If you want Codex in your code editor (VS Code, Cursor, Windsurf), <a href="https://developers.openai.com/codex/ide">install in your IDE.</a>
</br>If you want the desktop app experience, run <code>codex-kontext app</code> or visit <a href="https://chatgpt.com/codex?app-landing-page=true">the Codex App page</a>.
</br>If you are looking for the <em>cloud-based agent</em> from OpenAI, <strong>Codex Web</strong>, go to <a href="https://chatgpt.com/codex">chatgpt.com/codex</a>.</p>

---

## Kontext-Dev fork

This fork wires Codex CLI to the Kontext-Dev MCP server using the `kontext-dev-sdk`
Rust SDK. It authenticates with PKCE, exchanges tokens for `mcp-gateway`, and
attaches a single MCP server automatically.

### Configuration (local)

This fork uses baked Kontext defaults. No `[kontext-dev]` block is required in
`~/.codex/config.toml`.

Built-in values:

- `client_id`: baked into this fork
- `redirect_uri`: `http://localhost:3333/callback`
- `server`: `http://localhost:4000/mcp`
- `resource`: `mcp-gateway`
- `integration_ui_url`: `http://localhost:3000`
- `open_connect_page_on_login`: `true`

Flow rule: local gateway handles API/MCP/auth endpoints at port `4000`, and
integrations connect UI runs on local web at port `3000`
(`http://localhost:4000/oauth/connect` is not a valid route).

### Running locally

```bash
cd codex-rs
cargo run --bin codex
```

Branch model: `main` mirrors `openai/codex`, and `kontext-dev` carries Kontext
customizations on top of `main`.

## Quickstart

### Installing and running Codex CLI

Configure npm for GitHub Packages and install globally:

```shell
# Authenticate to GitHub Packages (@kontext-dev scope)
npm config set @kontext-dev:registry https://npm.pkg.github.com
npm config set //npm.pkg.github.com/:_authToken YOUR_GITHUB_PAT

# Install the private fork package
npm install -g @kontext-dev/codex
```

Then run `codex-kontext` to get started.

<details>
<summary>You can also go to the <a href="https://github.com/kontext-dev/codex/releases/latest">latest GitHub Release</a> and download the appropriate binary for your platform.</summary>

Each GitHub Release contains many executables, but in practice, you likely want one of these:

- macOS
  - Apple Silicon/arm64: `codex-aarch64-apple-darwin.tar.gz`
  - x86_64 (older Mac hardware): `codex-x86_64-apple-darwin.tar.gz`
- Linux
  - x86_64: `codex-x86_64-unknown-linux-musl.tar.gz`
  - arm64: `codex-aarch64-unknown-linux-musl.tar.gz`

Each archive contains a single entry with the platform baked into the name (e.g., `codex-x86_64-unknown-linux-musl`), so you likely want to rename it to `codex` after extracting it.

</details>

### Using Codex with your ChatGPT plan

Run `codex-kontext` and select **Sign in with ChatGPT**. We recommend signing into your ChatGPT account to use Codex as part of your Plus, Pro, Team, Edu, or Enterprise plan. [Learn more about what's included in your ChatGPT plan](https://help.openai.com/en/articles/11369540-codex-in-chatgpt).

You can also use Codex with an API key, but this requires [additional setup](https://developers.openai.com/codex/auth#sign-in-with-an-api-key).

### First-run Kontext flow

After Codex login completes, this fork automatically runs Kontext PKCE. If
your integrations are disconnected, the Kontext connect page opens
automatically. On later runs, prompts only reappear when token or integration
state changes.

## Docs

- [**Codex Documentation**](https://developers.openai.com/codex)
- [**Contributing**](./docs/contributing.md)
- [**Installing & building**](./docs/install.md)
- [**Open source fund**](./docs/open-source-fund.md)

This repository is licensed under the [Apache-2.0 License](LICENSE).
