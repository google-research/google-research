use eframe::egui_wgpu::WgpuConfiguration;
use std::sync::Arc;

pub fn get_config() -> WgpuConfiguration {
    WgpuConfiguration {
        device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("egui+burn wgpu device"),
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            // cube already batches allocations.
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        }),
        supported_backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    }
}
