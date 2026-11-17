//! Module containing utilities to query the currently running antivirus / EDR software on the
//! user's machine.

mod telemetry;
#[cfg(windows)]
mod windows;

use warpui::{Entity, ModelContext, SingletonEntity};

/// Singleton model that reports the currently running antivirus software.
#[derive(Debug, Clone)]
pub struct AntivirusInfo(Option<String>);

impl AntivirusInfo {
    pub fn new(_ctx: &mut ModelContext<Self>) -> Self {
        #[cfg(windows)]
        {
            // COM calls block, and `ctx.spawn` futures are polled on the UI thread, so scan on
            // a dedicated thread and await the result over a channel.
            let (tx, rx) = async_channel::bounded(1);
            std::thread::spawn(move || {
                let start = std::time::Instant::now();
                let result = futures_lite::future::block_on(Self::scan());
                log::info!(
                    "startup perf: antivirus scan took {} ms",
                    start.elapsed().as_millis()
                );
                let _ = tx.send_blocking(result);
            });
            _ctx.spawn(
                async move {
                    rx.recv()
                        .await
                        .unwrap_or_else(|err| Err(anyhow::anyhow!(err)))
                },
                Self::on_scan_complete,
            );
        }

        Self(None)
    }

    /// Returns the currently running antivirus software if any.
    /// If called before the antivirus is computed (i.e. before
    /// [`AntivirusInfoEvent::ScannedComplete`] is emitted), this function returns [`None`].
    ///
    /// ## Platform-specific
    /// This function always returns `None` on non-Windows platforms.
    pub fn get(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

pub enum AntivirusInfoEvent {
    #[allow(dead_code)]
    ScannedComplete,
}

impl Entity for AntivirusInfo {
    type Event = AntivirusInfoEvent;
}

impl SingletonEntity for AntivirusInfo {}
