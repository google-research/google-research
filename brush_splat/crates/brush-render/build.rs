use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    brush_wgsl::build_modules(
        &[
            "src/shaders/project_forward.wgsl",
            "src/shaders/project_visible.wgsl",
            "src/shaders/map_gaussian_to_intersects.wgsl",
            "src/shaders/get_tile_bin_edges.wgsl",
            "src/shaders/rasterize.wgsl",
            "src/shaders/rasterize_backwards.wgsl",
            "src/shaders/gather_grads.wgsl",
            "src/shaders/project_backwards.wgsl",
        ],
        &["src/shaders/helpers.wgsl"],
        "src/shaders",
        "src/shaders/mod.rs",
    )
    .into_diagnostic()
}
