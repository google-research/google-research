use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    brush_wgsl::build_modules(
        &["src/shaders/wg.wgsl"],
        &[],
        "src/shaders",
        "src/shaders/mod.rs",
    )
    .into_diagnostic()
}
