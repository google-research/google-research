use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    brush_wgsl::build_modules(
        &[
            "src/shaders/sort_count.wgsl",
            "src/shaders/sort_reduce.wgsl",
            "src/shaders/sort_scan_add.wgsl",
            "src/shaders/sort_scan.wgsl",
            "src/shaders/sort_scatter.wgsl",
        ],
        &["src/shaders/sorting.wgsl"],
        "src/shaders",
        "src/shaders/mod.rs",
    )
    .into_diagnostic()
}
