use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{info_span, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

use burn_wgpu::{Wgpu, WgpuDevice};

// Global flag to enable/disable sync
static SYNC_ENABLED: AtomicBool = AtomicBool::new(false);

// Tracing layer for sync events
#[derive(Default)]
pub struct SyncLayer {
    device: WgpuDevice,
}

impl SyncLayer {
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }
}

impl<S> Layer<S> for SyncLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        if SYNC_ENABLED.load(Ordering::Relaxed) {
            let metadata = ctx.metadata(&id).unwrap();

            if metadata.is_span() && metadata.fields().field("sync_burn").is_some() {
                let _span = info_span!("GPU Wait", name = metadata.name()).entered();

                <Wgpu as burn::prelude::Backend>::sync(
                    &self.device,
                    burn::tensor::backend::SyncType::Wait,
                );
            }
        }
    }
}

pub fn is_enabled() -> bool {
    SYNC_ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(enabled: bool) {
    SYNC_ENABLED.store(enabled, Ordering::Relaxed);
}
