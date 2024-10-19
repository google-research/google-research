#[cfg(target_os = "android")]
pub mod android;

use anyhow::Result;

#[derive(Clone, Debug)]
pub struct PickedFile {
    pub data: Vec<u8>,
    pub file_name: String,
}

#[cfg(not(target_os = "android"))]
pub async fn pick_file_rfd() -> Result<PickedFile> {
    use anyhow::Context;
    let file = rfd::AsyncFileDialog::new()
        .pick_file()
        .await
        .context("Failed to pick file")?;

    let file_data = file.read().await;

    Ok(PickedFile {
        data: file_data,
        file_name: file.file_name(),
    })
}

pub async fn pick_file() -> Result<PickedFile> {
    #[cfg(not(target_os = "android"))]
    {
        pick_file_rfd().await
    }

    #[cfg(target_os = "android")]
    {
        android::pick_file().await
    }
}
