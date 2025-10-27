use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_ansi(true)
            .with_writer(std::io::stderr)
            .init();
    });
}
