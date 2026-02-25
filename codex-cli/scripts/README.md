# npm releases

Use the staging helper in the repo root to generate npm tarballs for a release.
For this fork, only stage the CLI package. Example with version
`0.6.0-kontext.1`:

```bash
./scripts/stage_npm_packages.py \
  --release-version 0.6.0-kontext.1 \
  --package codex
```

This downloads the native artifacts once, hydrates `vendor/` for each package, and writes
tarballs to `dist/npm/`.

When `--package codex` is provided, the staging helper builds the lightweight
`@kontext-dev/codex` meta package plus all platform-native `@kontext-dev/codex` variants
that are later published under platform-specific dist-tags.

If you need to invoke `build_npm_package.py` directly, run
`codex-cli/scripts/install_native_deps.py` first and pass `--vendor-src` pointing to the
directory that contains the populated `vendor/` tree.
