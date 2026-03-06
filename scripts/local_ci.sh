#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pnpm install --frozen-lockfile

./scripts/asciicheck.py README.md
python3 scripts/readme_toc.py README.md
./scripts/asciicheck.py codex-cli/README.md
python3 scripts/readme_toc.py codex-cli/README.md
pnpm run format

(
  cd codex-rs
  cargo deny check
  cargo build --bin codex-kontext
)

CODEX_INTERNAL_ORIGINATOR_OVERRIDE=codex_sdk_ts pnpm -r --filter ./sdk/typescript run build
CODEX_INTERNAL_ORIGINATOR_OVERRIDE=codex_sdk_ts pnpm -r --filter ./sdk/typescript run lint
CODEX_INTERNAL_ORIGINATOR_OVERRIDE=codex_sdk_ts pnpm -r --filter ./sdk/typescript exec jest --testTimeout=30000
