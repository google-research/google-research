use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    brush_wgsl::build_modules(
        &[
            "src/shaders/prefix_sum_add_scanned_sums.wgsl",
            "src/shaders/prefix_sum_scan_sums.wgsl",
            "src/shaders/prefix_sum_scan.wgsl",
        ],
        &["src/shaders/prefix_sum_helpers.wgsl"],
        "src/shaders",
        "src/shaders/mod.rs",
    )
    .into_diagnostic()
}
