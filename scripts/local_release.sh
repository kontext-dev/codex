#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/local_release.sh <version>

Environment:
  CODEX_RELEASE_TARGET   Target triple to build and package. Defaults to aarch64-apple-darwin.
  CODEX_NPM_SCOPE        npm scope used for staged packages. Defaults to kontext-dev.
  NODE_AUTH_TOKEN        Required when npm publishing is enabled.
  SKIP_GH_RELEASE        Set to 1 to skip creating/updating the GitHub release.
  SKIP_NPM_PUBLISH       Set to 1 to skip npm publishing.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

version="$1"
target="${CODEX_RELEASE_TARGET:-aarch64-apple-darwin}"
npm_scope="${CODEX_NPM_SCOPE:-kontext-dev}"
skip_gh_release="${SKIP_GH_RELEASE:-0}"
skip_npm_publish="${SKIP_NPM_PUBLISH:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tag="rust-v${version}"
dist_dir="$repo_root/dist/local-release"
npm_dir="$dist_dir/npm"
vendor_root="$(mktemp -d "${TMPDIR:-/tmp}/codex-local-vendor.XXXXXX")"
vendor_src="$vendor_root/vendor"
cleanup() {
  rm -rf "$vendor_root"
}
trap cleanup EXIT

./scripts/local_ci.sh

cli_bin="$(
  cd codex-rs
  cargo metadata --no-deps --format-version=1 \
    | python3 -c 'import json, sys; data=json.load(sys.stdin); bins={t["name"] for p in data["packages"] for t in p["targets"] if "bin" in t["kind"]}; print("codex" if "codex" in bins else "codex-kontext")'
)"

(
  cd codex-rs
  cargo build --target "$target" --release --bin "$cli_bin" --bin codex-responses-api-proxy
)

mkdir -p "$vendor_src/$target/codex" "$vendor_src/$target/codex-responses-api-proxy"
cp "codex-rs/target/$target/release/$cli_bin" "$vendor_src/$target/codex/codex"
cp "codex-rs/target/$target/release/codex-responses-api-proxy" "$vendor_src/$target/codex-responses-api-proxy/codex-responses-api-proxy"
python3 codex-cli/scripts/install_native_deps.py --component rg --target "$target" "$vendor_root"

rm -rf "$dist_dir"
mkdir -p "$npm_dir"
cp "codex-rs/target/$target/release/$cli_bin" "$dist_dir/codex-$target"
cp "codex-rs/target/$target/release/codex-responses-api-proxy" "$dist_dir/codex-responses-api-proxy-$target"
cp scripts/install/install.sh "$dist_dir/install.sh"
cp scripts/install/install.ps1 "$dist_dir/install.ps1"
if [[ -f codex-rs/core/config.schema.json ]]; then
  cp codex-rs/core/config.schema.json "$dist_dir/config.schema.json"
fi

CODEX_NPM_SCOPE="$npm_scope" python3 scripts/stage_npm_packages.py \
  --release-version "$version" \
  --target "$target" \
  --package codex \
  --package codex-responses-api-proxy \
  --package codex-sdk \
  --vendor-src "$vendor_src" \
  --output-dir "$npm_dir"

if ! git rev-parse --verify --quiet "$tag" >/dev/null; then
  git tag -a "$tag" -m "Release $version"
fi
git push origin "$tag"

release_assets=()
for asset in "$dist_dir"/* "$npm_dir"/*; do
  if [[ -f "$asset" ]]; then
    release_assets+=("$asset")
  fi
done

if [[ "$skip_gh_release" != "1" ]]; then
  if gh release view "$tag" >/dev/null 2>&1; then
    gh release upload "$tag" "${release_assets[@]}" --clobber
  else
    gh release create "$tag" "${release_assets[@]}" --verify-tag --title "$version" --notes "Local release $version"
  fi
fi

should_publish="false"
npm_tag=""
npm_registry="https://registry.npmjs.org"
if [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  should_publish="true"
elif [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-alpha\.[0-9]+$ ]]; then
  should_publish="true"
  npm_tag="alpha"
elif [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-kontext\.[0-9]+$ ]]; then
  should_publish="true"
  npm_tag="latest"
  npm_registry="https://npm.pkg.github.com"
fi

if [[ "$should_publish" == "true" && "$skip_npm_publish" != "1" ]]; then
  if [[ -z "${NODE_AUTH_TOKEN:-}" && "$npm_registry" == "https://npm.pkg.github.com" ]]; then
    if gh_token="$(gh auth token 2>/dev/null)"; then
      export NODE_AUTH_TOKEN="$gh_token"
    fi
  fi

  if [[ -z "${NODE_AUTH_TOKEN:-}" ]]; then
    echo "NODE_AUTH_TOKEN is required for npm publishing." >&2
    exit 1
  fi

  prefix=""
  if [[ -n "$npm_tag" ]]; then
    prefix="${npm_tag}-"
  fi

  shopt -s nullglob
  tarballs=("$npm_dir"/*-"${version}".tgz)
  if [[ ${#tarballs[@]} -eq 0 ]]; then
    echo "No npm tarballs found for version ${version}" >&2
    exit 1
  fi

  for tarball in "${tarballs[@]}"; do
    filename="$(basename "$tarball")"
    publish_tag=""

    case "$filename" in
      codex-npm-*-"${version}".tgz)
        platform="${filename#codex-npm-}"
        platform="${platform%-${version}.tgz}"
        publish_tag="${prefix}${platform}"
        ;;
      codex-npm-"${version}".tgz|codex-responses-api-proxy-npm-"${version}.tgz|codex-sdk-npm-"${version}.tgz)
        publish_tag="$npm_tag"
        ;;
      *)
        echo "Unexpected npm tarball: $filename" >&2
        exit 1
        ;;
    esac

    publish_cmd=(npm publish "$tarball" --registry "$npm_registry")
    if [[ -n "$publish_tag" ]]; then
      publish_cmd+=(--tag "$publish_tag")
    fi

    "${publish_cmd[@]}"
  done
fi

echo "Local release complete for ${version}"
