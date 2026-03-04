#![cfg(not(debug_assertions))]

use crate::update_action;
use chrono::DateTime;
use chrono::Duration;
use chrono::Utc;
use codex_core::config::Config;
use semver::Version;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;
use std::path::PathBuf;

use crate::version::CODEX_CLI_VERSION;

pub fn get_upgrade_version(config: &Config) -> Option<String> {
    if !config.check_for_update_on_startup {
        return None;
    }
    if update_action::get_update_action().is_none() {
        return None;
    }

    let version_file = version_filepath(config);
    let info = read_version_info(&version_file).ok();

    if match &info {
        None => true,
        Some(info) => info.last_checked_at < Utc::now() - Duration::hours(20),
    } {
        // Refresh the cached latest version in the background so TUI startup
        // isn’t blocked by a network call. The UI reads the previously cached
        // value (if any) for this run; the next run shows the banner if needed.
        tokio::spawn(async move {
            check_for_update(&version_file)
                .await
                .inspect_err(|e| tracing::error!("Failed to update version: {e}"))
        });
    }

    info.and_then(|info| {
        if is_newer(&info.latest_version, CODEX_CLI_VERSION).unwrap_or(false) {
            Some(info.latest_version)
        } else {
            None
        }
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VersionInfo {
    latest_version: String,
    // ISO-8601 timestamp (RFC3339)
    last_checked_at: DateTime<Utc>,
    #[serde(default)]
    dismissed_version: Option<String>,
}

const VERSION_FILENAME: &str = "version.json";
const NPM_PACKAGE_FOR_UPDATES: &str = "@kontext-dev/codex";

fn version_filepath(config: &Config) -> PathBuf {
    config.codex_home.join(VERSION_FILENAME)
}

fn read_version_info(version_file: &Path) -> anyhow::Result<VersionInfo> {
    let contents = std::fs::read_to_string(version_file)?;
    Ok(serde_json::from_str(&contents)?)
}

async fn check_for_update(version_file: &Path) -> anyhow::Result<()> {
    let output = tokio::process::Command::new("npm")
        .args(["view", NPM_PACKAGE_FOR_UPDATES, "version", "--json"])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        anyhow::bail!("Failed to query npm for latest {NPM_PACKAGE_FOR_UPDATES} version: {stderr}");
    }

    let stdout = std::str::from_utf8(&output.stdout)?;
    let latest_version = parse_npm_view_version_output(stdout)?;

    // Preserve any previously dismissed version if present.
    let prev_info = read_version_info(version_file).ok();
    let info = VersionInfo {
        latest_version,
        last_checked_at: Utc::now(),
        dismissed_version: prev_info.and_then(|p| p.dismissed_version),
    };

    let json_line = format!("{}\n", serde_json::to_string(&info)?);
    if let Some(parent) = version_file.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(version_file, json_line).await?;
    Ok(())
}

fn is_newer(latest: &str, current: &str) -> Option<bool> {
    match (parse_version(latest), parse_version(current)) {
        (Some(l), Some(c)) => Some(l > c),
        _ => None,
    }
}

fn parse_npm_view_version_output(stdout: &str) -> anyhow::Result<String> {
    let version = serde_json::from_str::<String>(stdout.trim())?;
    let normalized = version.trim();
    if normalized.is_empty() {
        anyhow::bail!("npm returned an empty version for {NPM_PACKAGE_FOR_UPDATES}");
    }

    Ok(normalized.to_string())
}

/// Returns the latest version to show in a popup, if it should be shown.
/// This respects the user's dismissal choice for the current latest version.
pub fn get_upgrade_version_for_popup(config: &Config) -> Option<String> {
    if !config.check_for_update_on_startup {
        return None;
    }

    let version_file = version_filepath(config);
    let latest = get_upgrade_version(config)?;
    // If the user dismissed this exact version previously, do not show the popup.
    if let Ok(info) = read_version_info(&version_file)
        && info.dismissed_version.as_deref() == Some(latest.as_str())
    {
        return None;
    }
    Some(latest)
}

/// Persist a dismissal for the current latest version so we don't show
/// the update popup again for this version.
pub async fn dismiss_version(config: &Config, version: &str) -> anyhow::Result<()> {
    let version_file = version_filepath(config);
    let mut info = match read_version_info(&version_file) {
        Ok(info) => info,
        Err(_) => return Ok(()),
    };
    info.dismissed_version = Some(version.to_string());
    let json_line = format!("{}\n", serde_json::to_string(&info)?);
    if let Some(parent) = version_file.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(version_file, json_line).await?;
    Ok(())
}

fn parse_version(v: &str) -> Option<Version> {
    Version::parse(v.trim()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_npm_view_json_string_output() {
        assert_eq!(
            parse_npm_view_version_output("\"0.108.0-kontext.1\"\n")
                .expect("failed to parse npm version output"),
            "0.108.0-kontext.1"
        );
    }

    #[test]
    fn parses_npm_view_output_and_trims_inner_whitespace() {
        assert_eq!(
            parse_npm_view_version_output("\"  0.108.0-kontext.2  \"")
                .expect("failed to parse npm version output"),
            "0.108.0-kontext.2"
        );
    }

    #[test]
    fn rejects_non_string_npm_view_output() {
        assert!(parse_npm_view_version_output("{\"version\":\"0.108.0\"}").is_err());
    }

    #[test]
    fn semver_comparisons_work_for_kontext_versions() {
        assert_eq!(
            is_newer("0.108.0-kontext.2", "0.108.0-kontext.1"),
            Some(true)
        );
        assert_eq!(
            is_newer("0.108.0-kontext.1", "0.108.0-kontext.2"),
            Some(false)
        );
        assert_eq!(
            is_newer("0.109.0-kontext.1", "0.108.9-kontext.99"),
            Some(true)
        );
        assert_eq!(is_newer("0.108.0", "0.108.0-kontext.9"), Some(true));
    }

    #[test]
    fn invalid_versions_return_none_for_comparison() {
        assert_eq!(is_newer("0.108.0-kontext.1", "invalid"), None);
        assert_eq!(is_newer("invalid", "0.108.0-kontext.1"), None);
    }

    #[test]
    fn whitespace_is_ignored() {
        assert_eq!(
            parse_version(" 0.108.0-kontext.3 \n")
                .expect("expected semver version")
                .to_string(),
            "0.108.0-kontext.3"
        );
        assert_eq!(
            is_newer(" 0.108.0-kontext.3 ", "0.108.0-kontext.2"),
            Some(true)
        );
    }
}
