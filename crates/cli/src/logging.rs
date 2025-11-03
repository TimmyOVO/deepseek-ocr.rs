use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init(quiet: bool) {
    INIT.call_once(|| {
        if quiet {
            // In quiet mode, disable all logs or log to stderr at error level
            let filter = EnvFilter::new("error");
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(true)
                .with_writer(std::io::stderr)
                .init();
        } else {
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(true)
                .init();
        }
    });
}
