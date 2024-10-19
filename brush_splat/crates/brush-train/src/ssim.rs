use burn::tensor::{backend::Backend, module::conv2d, ops::ConvOptions, Tensor};

fn gaussian<B: Backend>(window_size: usize, sigma: f32, device: &B::Device) -> Tensor<B, 1> {
    let sigma2 = 2.0 * sigma.powf(2.0);
    let window_extent = (window_size / 2) as f32;
    let gauss_floats: Vec<_> = (0..window_size)
        .map(|x| (-(x as f32 - window_extent).powf(2.0) / sigma2).exp())
        .collect();
    let gauss = Tensor::from_floats(gauss_floats.as_slice(), device);
    gauss.clone() / gauss.sum()
}

fn create_window<B: Backend>(
    window_size: usize,
    channel: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let window1d = gaussian(window_size, 1.5, device).reshape([window_size, 1]);
    let window2d = window1d.clone() * window1d.transpose();
    window2d.unsqueeze().repeat_dim(1, channel)
}

pub fn ssim<B: Backend>(
    img1: Tensor<B, 4>,
    img2: Tensor<B, 4>,
    window_size: usize,
) -> Tensor<B, 1> {
    let device = &img1.device();

    let [_, height, width, channel] = img1.dims();

    let real_size = window_size.min(height).min(width);
    let window = create_window::<B>(real_size, channel, device);

    let conv_options = ConvOptions::new([1, 1], [0, 0], [0, 0], channel);
    let mu1 = conv2d(img1.clone(), window.clone(), None, conv_options.clone());
    let mu2 = conv2d(img2.clone(), window.clone(), None, conv_options.clone());

    let mu1_sq = mu1.clone().powf_scalar(2);
    let mu2_sq = mu2.clone().powf_scalar(2);
    let mu1_mu2 = mu1 * mu2;

    let sigma1_sq = conv2d(
        img1.clone().powf_scalar(2.0),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu1_sq.clone();
    let sigma2_sq = conv2d(
        img2.clone().powf_scalar(2.0),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu2_sq.clone();
    let sigma12 = conv2d(
        img1.clone() * img2.clone(),
        window.clone(),
        None,
        conv_options.clone(),
    ) - mu1_mu2.clone();

    let c1 = (0.01f32).powf(2.0);
    let c2 = (0.03f32).powf(2.0);

    let v1 = sigma12 * 2.0 + c2;
    let v2 = sigma1_sq + sigma2_sq + c2;

    let ssim_map = ((mu1_mu2 * 2.0 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2);
    ssim_map.mean()
}
