#![cfg(feature = "benchmarking")]
//! Opens the Kontext integration connect page in the browser.
//!
//! Usage:
//!   make connect
//!   # or directly:
//!   cargo test -p codex-core --features benchmarking --test connect_integrations -- --nocapture

use core_test_support::gateway_auth;
use kontext_dev::KontextDevClient;

#[tokio::test]
async fn open_connect_page() {
    if gateway_auth::should_skip() {
        eprintln!("KONTEXT_CLIENT_ID / KONTEXT_CLIENT_SECRET not set.");
        eprintln!("Add them to .env or export them before running `make connect`.");
        return;
    }

    let mut config = gateway_auth::build_kontext_config().expect("Config should be valid");
    config.open_connect_page_on_login = true;

    // The web UI serving /oauth/connect runs on a separate port from the API gateway.
    // Default: localhost:3000 for local dev, app.kontext.dev for production.
    if config.integration_ui_url.is_none() {
        let ui_url = std::env::var("KONTEXT_UI_URL").unwrap_or_else(|_| {
            // Derive from gateway URL: if localhost, use port 3000 for the web app
            let mcp = config.mcp_url.as_deref().unwrap_or("http://localhost:4000/mcp");
            if mcp.contains("localhost") || mcp.contains("127.0.0.1") {
                "http://localhost:3000".to_string()
            } else if mcp.contains("api.kontext.dev") {
                "https://app.kontext.dev".to_string()
            } else {
                // Non-local, non-production: assume web app is on same host, port 3000
                mcp.split(':').take(2).collect::<Vec<_>>().join(":") + ":3000"
            }
        });
        config.integration_ui_url = Some(ui_url);
    }

    // Authenticate (client-credentials flow) to get a gateway token
    let token = match gateway_auth::authenticate(&config).await {
        Ok(t) => t,
        Err(e) if e.to_string().contains("onnection refused") => {
            eprintln!("Gateway not running at {:?}", config.mcp_url);
            eprintln!("Start it with: make setup");
            return;
        }
        Err(e) => panic!("Auth failed: {e}"),
    };

    // Create a connect session and open the page
    let client = KontextDevClient::new(config);
    match client
        .open_integration_connect_page(&token.access_token)
        .await
    {
        Ok(url) => {
            eprintln!("Opened integration connect page:");
            eprintln!("  {url}");
            eprintln!();
            eprintln!("Connect your integrations (e.g. Linear) in the browser,");
            eprintln!("then re-run `make test` to verify tools are available.");
        }
        Err(e) => {
            // Fall back to printing the URL manually
            eprintln!("Could not open browser automatically: {e}");
            if let Ok(url) = client.create_integration_connect_url(&token.access_token).await {
                eprintln!("Open this URL manually:");
                eprintln!("  {url}");
            }
        }
    }
}
