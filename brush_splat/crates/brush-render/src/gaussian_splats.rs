use crate::camera::Camera;
use crate::safetensor_utils::safetensor_to_burn;
use crate::{render::num_sh_coeffs, Backend};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use burn::{
    module::{Module, Param, ParamId},
    tensor::Device,
};
use safetensors::SafeTensors;

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub means: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 2>>,
    pub rotation: Param<Tensor<B, 2>>,
    pub raw_opacity: Param<Tensor<B, 1>>,
    pub log_scales: Param<Tensor<B, 2>>,

    // Dummy input to track screenspace gradient.
    pub xys_dummy: Tensor<B, 2>,
}

fn map_param<B: Backend, const D: usize>(
    tensor: &mut Param<Tensor<B, D>>,
    f: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
) {
    *tensor = tensor.clone().map(|x| f(x).detach().require_grad());
}

impl<B: Backend> Splats<B> {
    pub fn init_random(num_points: usize, aabb_scale: f32, device: &Device<B>) -> Splats<B> {
        let extent = (aabb_scale as f64) / 2.0;
        let means = Tensor::random(
            [num_points, 3],
            Distribution::Uniform(-extent, extent),
            device,
        );

        let num_coeffs = num_sh_coeffs(0);

        let sh_coeffs = Tensor::random(
            [num_points, num_coeffs * 3],
            Distribution::Uniform(-0.5, 0.5),
            device,
        );
        let init_rotation = Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat_dim(0, num_points);

        let init_raw_opacity =
            Tensor::random([num_points], Distribution::Uniform(-2.0, -1.0), device);

        // TODO: Fancy KNN init.
        let init_scale = Tensor::random([num_points, 3], Distribution::Uniform(-3.0, -2.0), device);

        Self::from_data(
            means,
            sh_coeffs,
            init_rotation,
            init_raw_opacity,
            init_scale,
            device,
        )
    }

    pub fn from_data(
        means: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        raw_opacity: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
        device: &Device<B>,
    ) -> Self {
        let num_points = means.shape().dims[0];
        Splats {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            rotation: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
            xys_dummy: Tensor::zeros([num_points, 2], device).require_grad(),
        }
    }

    pub fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        bg_color: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<B, 3>, crate::RenderAux) {
        B::render_splats(
            camera,
            img_size,
            self.means.val(),
            self.xys_dummy.clone(),
            self.log_scales.val(),
            self.rotation.val(),
            self.sh_coeffs.val(),
            self.raw_opacity.val(),
            bg_color,
            render_u32_buffer,
        )
    }

    pub fn num_splats(&self) -> usize {
        self.means.dims()[0]
    }

    pub fn concat_splats(
        &mut self,
        means: Tensor<B, 2>,
        rotations: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 2>,
        raw_opacities: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
    ) {
        map_param(&mut self.means, |x| Tensor::cat(vec![x, means.clone()], 0));
        map_param(&mut self.rotation, |x| {
            Tensor::cat(vec![x, rotations.clone()], 0)
        });
        map_param(&mut self.sh_coeffs, |x| {
            Tensor::cat(vec![x, sh_coeffs.clone()], 0)
        });
        map_param(&mut self.raw_opacity, |x| {
            Tensor::cat(vec![x, raw_opacities.clone()], 0)
        });
        map_param(&mut self.log_scales, |x| {
            Tensor::cat(vec![x, log_scales.clone()], 0)
        });
    }

    pub fn norm_rotations(&mut self) {
        map_param(&mut self.rotation, |x| {
            x.clone() / Tensor::clamp_min(Tensor::sum_dim(x.powf_scalar(2.0), 1).sqrt(), 1e-6)
        });
    }

    pub fn from_safetensors(tensors: &SafeTensors, device: &B::Device) -> anyhow::Result<Self> {
        let means = safetensor_to_burn::<B, 2>(tensors.tensor("means")?, device);
        let num_points = means.dims()[0];
        let log_scales = safetensor_to_burn::<B, 2>(tensors.tensor("scales")?, device);

        // TODO: This doesn't really handle SH properly. Probably should serialize this in the format
        // we expect and save this reshape hassle.
        let sh_coeffs =
            safetensor_to_burn::<B, 3>(tensors.tensor("coeffs")?, device).reshape([num_points, 3]);
        let quats = safetensor_to_burn::<B, 2>(tensors.tensor("quats")?, device);
        let raw_opacity = safetensor_to_burn::<B, 1>(tensors.tensor("opacities")?, device);

        Ok(Self::from_data(
            means,
            sh_coeffs,
            quats,
            raw_opacity,
            log_scales,
            device,
        ))
    }
}
