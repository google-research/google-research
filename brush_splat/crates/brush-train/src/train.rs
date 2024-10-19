use anyhow::Result;
use brush_render::gaussian_splats::Splats;
use brush_render::{AutodiffBackend, Backend, RenderAux};
use burn::lr_scheduler::linear::{LinearLrScheduler, LinearLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::nn::loss::HuberLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::tensor::{Bool, Distribution};
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use tracing::info_span;

use crate::scene::SceneBatch;

#[derive(Config)]
pub struct LrConfig {
    #[config(default = 3e-3)]
    min_lr: f64,
    #[config(default = 3e-3)]
    max_lr: f64,
}

#[derive(Config)]
pub struct TrainConfig {
    pub lr_mean: LrConfig,
    pub lr_opac: LrConfig,
    pub lr_rest: LrConfig,

    #[config(default = 5000)]
    pub(crate) schedule_steps: u32,

    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 200)]
    pub(crate) warmup_steps: u32,
    #[config(default = 100)]
    pub(crate) refine_every: u32,

    #[config(default = 0.0)]
    pub(crate) ssim_weight: f32,
    // threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    #[config(default = 0.05)]
    pub(crate) prune_alpha_thresh: f32,
    #[config(default = 0.005)]
    pub(crate) prune_scale_thresh: f32,

    #[config(default = 0.0001)]
    pub(crate) clone_split_grad_threshold: f32,
    #[config(default = 0.01)]
    pub(crate) split_clone_size_threshold: f32,
    // threshold of scale for culling huge gaussians.
    #[config(default = 0.5)]
    pub(crate) cull_scale_thresh: f32,
    #[config(default = 30)]
    pub(crate) reset_alpha_every: u32,

    #[config(default = false)]
    pub(crate) random_bck_color: bool,
    #[config(default = 100)]
    pub visualize_every: u32,
    #[config(default = 250)]
    pub visualize_splats_every: u32,

    #[config(default = 1000)]
    pub init_splat_count: usize,
}

pub struct TrainStepStats<B: AutodiffBackend> {
    pub pred_images: Tensor<B, 4>,
    pub auxes: Vec<RenderAux>,
    pub loss: Tensor<B, 1>,
    pub lr_mean: f64,
    pub lr_opac: f64,
    pub lr_rest: f64,
    pub iter: u32,
}

pub struct SplatTrainer<B: AutodiffBackend>
where
    B::InnerBackend: Backend,
{
    pub iter: u32,

    config: TrainConfig,

    sched_mean: LinearLrScheduler,
    sched_opac: LinearLrScheduler,
    sched_rest: LinearLrScheduler,

    opt_config: AdamConfig,
    optim: OptimizerAdaptor<Adam<B::InnerBackend>, Splats<B>, B>,

    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    xy_grad_norm_accum: Tensor<B, 1>,
}

pub(crate) fn quat_multiply<B: Backend>(q: Tensor<B, 2>, r: Tensor<B, 2>) -> Tensor<B, 2> {
    let num = q.dims()[0];

    let (qw, qx, qy, qz) = (
        q.clone().slice([0..num, 0..1]),
        q.clone().slice([0..num, 1..2]),
        q.clone().slice([0..num, 2..3]),
        q.clone().slice([0..num, 3..4]),
    );
    let (rw, rx, ry, rz) = (
        r.clone().slice([0..num, 0..1]),
        r.clone().slice([0..num, 1..2]),
        r.clone().slice([0..num, 2..3]),
        r.clone().slice([0..num, 3..4]),
    );

    // Slightly odd hack to make cloning easier.
    let (qw, qx, qy, qz) = (|| qw.clone(), || qx.clone(), || qy.clone(), || qz.clone());
    let (rw, rx, ry, rz) = (|| rw.clone(), || rx.clone(), || ry.clone(), || rz.clone());

    let sw = qw() * rw() - qx() * rx() - qy() * ry() - qz() * rz();
    let sx = qw() * rx() + qx() * rw() + qy() * rz() - qz() * ry();
    let sy = qw() * ry() - qx() * rz() + qy() * rw() + qz() * rx();
    let sz = qw() * rz() + qx() * ry() - qy() * rx() + qz() * rw();

    Tensor::cat(vec![sw, sx, sy, sz], 1)
}

pub(crate) fn quaternion_rotation<B: Backend>(
    vectors: Tensor<B, 2>,
    quaternions: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num = vectors.dims()[0];
    // Convert vectors to quaternions with zero real part
    let vector_quats = Tensor::cat(
        vec![
            Tensor::zeros_like(&vectors.clone().slice([0..num, 0..1])),
            vectors,
        ],
        1,
    );

    // Calculate the conjugate of quaternions
    let quat_conj = quaternions.clone().slice_assign(
        [0..num, 1..4],
        quaternions.clone().slice([0..num, 1..4]) * -1,
    );

    // Rotate vectors: v' = q * v * q_conjugate
    let rotated_vectors = quat_multiply(quat_conj, quat_multiply(vector_quats, quaternions));

    // Return only the vector part (imaginary components)
    rotated_vectors.slice([0..num, 1..4])
}

impl<B: AutodiffBackend> SplatTrainer<B>
where
    B::InnerBackend: Backend,
{
    pub fn new(num_points: usize, config: &TrainConfig, splats: &Splats<B>) -> Self {
        let opt_config = AdamConfig::new().with_epsilon(1e-15);
        let optim = opt_config.init::<B, Splats<B>>();

        let device = &splats.means.device();

        let sched_mean = LinearLrSchedulerConfig::new(
            config.lr_mean.max_lr,
            config.lr_mean.min_lr,
            config.schedule_steps as usize,
        )
        .init();
        let sched_opac = LinearLrSchedulerConfig::new(
            config.lr_opac.max_lr,
            config.lr_opac.min_lr,
            config.schedule_steps as usize,
        )
        .init();
        let sched_rest = LinearLrSchedulerConfig::new(
            config.lr_rest.max_lr,
            config.lr_rest.min_lr,
            config.schedule_steps as usize,
        )
        .init();

        Self {
            config: config.clone(),
            iter: 0,
            optim,
            opt_config,
            sched_mean,
            sched_opac,
            sched_rest,
            xy_grad_norm_accum: Tensor::zeros([num_points], device),
        }
    }

    fn reset_stats(&mut self, num_points: usize, device: &B::Device) {
        self.xy_grad_norm_accum = Tensor::zeros([num_points], device);
    }

    // Densifies and prunes the Gaussians.
    pub async fn densify_and_prune(
        &mut self,
        splats: &mut Splats<B>,
        grad_threshold: f32,
        max_world_size_threshold: Option<f32>,
        clone_vs_split_size_threshold: f32,
        device: &B::Device,
    ) {
        if let Some(threshold) = max_world_size_threshold {
            // Delete Gaussians with too large of a radius in world-units.
            let prune_mask = splats
                .log_scales
                .val()
                .exp()
                .max_dim(1)
                .squeeze(1)
                .greater_elem(threshold);
            self.prune_points(splats, prune_mask).await;
        }

        // Compute average magnitude of the gradient for each Gaussian in
        // pixel-units while accounting for the number of times each Gaussian was
        // seen during training.
        let grads = self.xy_grad_norm_accum.clone();

        let big_grad_mask = grads.greater_equal_elem(grad_threshold);
        let split_clone_size_mask = splats
            .log_scales
            .val()
            .exp()
            .max_dim(1)
            .squeeze(1)
            .lower_elem(clone_vs_split_size_threshold);

        let clone_mask = Tensor::stack::<2>(
            vec![split_clone_size_mask.clone(), big_grad_mask.clone()],
            1,
        )
        .all_dim(1)
        .squeeze::<1>(1);
        let split_mask =
            Tensor::stack::<2>(vec![split_clone_size_mask.bool_not(), big_grad_mask], 1)
                .all_dim(1)
                .squeeze::<1>(1);

        // Need to be very careful not to do any operations with this tensor, as it might be
        // less than the minimum size wgpu can support :/
        let clone_where = clone_mask.clone().argwhere_async().await;

        if clone_where.dims()[0] >= 4 {
            let clone_inds = clone_where.squeeze(1);

            let new_means = splats.means.val().select(0, clone_inds.clone());
            let new_rots = splats.rotation.val().select(0, clone_inds.clone());
            let new_coeffs = splats.sh_coeffs.val().select(0, clone_inds.clone());
            let new_opac = splats.raw_opacity.val().select(0, clone_inds.clone());
            let new_scales = splats.log_scales.val().select(0, clone_inds.clone());
            splats.concat_splats(new_means, new_rots, new_coeffs, new_opac, new_scales);
        }

        let split_where = split_mask.clone().argwhere_async().await;
        if split_where.dims()[0] >= 4 {
            let split_inds = split_where.squeeze(1);
            let samps = split_inds.dims()[0];

            let scales = splats
                .log_scales
                .val()
                .select(0, split_inds.clone())
                .exp()
                .repeat_dim(0, 2);

            let new_rots = splats
                .rotation
                .val()
                .select(0, split_inds.clone())
                .repeat_dim(0, 2);

            let rotated_scale = quaternion_rotation(scales, new_rots.clone());

            let positions = splats
                .means
                .val()
                .select(0, split_inds.clone())
                .repeat_dim(0, 2);
            let new_means = positions
                + rotated_scale
                    * Tensor::random([samps * 2, 3], Distribution::Normal(0.0, 1.0), device);

            let new_coeffs = splats
                .sh_coeffs
                .val()
                .select(0, split_inds.clone())
                .repeat_dim(0, 2);
            let new_opac = splats
                .raw_opacity
                .val()
                .select(0, split_inds.clone())
                .repeat_dim(0, 2);
            let new_scales = (splats.log_scales.val().select(0, split_inds.clone()).exp() / 1.6)
                .log()
                .repeat_dim(0, 2);

            self.prune_points(splats, split_mask.clone()).await;

            splats.concat_splats(new_means, new_rots, new_coeffs, new_opac, new_scales);
        }
    }

    pub(crate) fn reset_opacity(&self, splats: &mut Splats<B>) {
        splats.raw_opacity = splats
            .raw_opacity
            .clone()
            .map(|x| Tensor::from_inner((x - 1.0).inner()).require_grad());
    }

    // Prunes points based on the given mask.
    //
    // Args:
    //   mask: bool[n]. If True, prune this Gaussian.
    pub async fn prune_points(&mut self, splats: &mut Splats<B>, prune: Tensor<B, 1, Bool>) {
        // bool[n]. If True, delete these Gaussians.
        let valid_inds = prune.bool_not().argwhere_async().await.squeeze(1);

        let start_splats = splats.num_splats();
        let new_points = valid_inds.dims()[0];

        if new_points < start_splats {
            self.xy_grad_norm_accum = self
                .xy_grad_norm_accum
                .clone()
                .select(0, valid_inds.clone());

            splats.means = splats.means.clone().map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            });
            splats.sh_coeffs = splats.sh_coeffs.clone().map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            });
            splats.rotation = splats.rotation.clone().map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            });
            splats.raw_opacity = splats.raw_opacity.clone().map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            });
            splats.log_scales = splats.log_scales.clone().map(|x| {
                Tensor::from_inner(x.select(0, valid_inds.clone()).inner()).require_grad()
            });
        }
    }

    pub async fn step(
        &mut self,
        batch: SceneBatch<B>,
        splats: Splats<B>,
    ) -> Result<(Splats<B>, TrainStepStats<B>), anyhow::Error> {
        let device = &splats.means.device();
        let _span = info_span!("Train step").entered();

        let background_color = if self.config.random_bck_color {
            glam::vec3(rand::random(), rand::random(), rand::random())
        } else {
            glam::Vec3::ZERO
        };

        let [batch_size, img_h, img_w, _] = batch.gt_images.dims();

        let (pred_images, auxes, loss) = {
            let mut renders = vec![];
            let mut auxes = vec![];

            for i in 0..batch.cameras.len() {
                let camera = &batch.cameras[i];

                let (pred_image, aux) = splats.render(
                    camera,
                    glam::uvec2(img_w as u32, img_h as u32),
                    background_color,
                    false,
                );

                renders.push(pred_image);
                auxes.push(aux);
            }

            // TODO: Could probably handle this in Burn.
            let pred_images = if renders.len() == 1 {
                renders[0].clone().reshape([1, img_h, img_w, 4])
            } else {
                Tensor::stack(renders, 0)
            };

            let _span = info_span!("Calculate losses", sync_burn = true).entered();

            // There might be some marginal benefit to caching the "loss objects". I wish Burn had a more
            // functional style for this.
            let huber = HuberLossConfig::new(0.05).init();
            let mut loss = huber.forward(
                pred_images.clone(),
                batch.gt_images.clone(),
                burn::nn::loss::Reduction::Mean,
            );

            if self.config.ssim_weight > 0.0 {
                let pred_rgb = pred_images
                    .clone()
                    .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
                let gt_rgb =
                    batch
                        .gt_images
                        .clone()
                        .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
                let ssim_loss = crate::ssim::ssim(
                    pred_rgb.clone().permute([0, 3, 1, 2]),
                    gt_rgb.clone().permute([0, 3, 1, 2]),
                    11,
                );
                loss = loss * (1.0 - self.config.ssim_weight)
                    + (-ssim_loss + 1.0) * self.config.ssim_weight;
            }
            (pred_images, auxes, loss)
        };

        let mut grads = info_span!("Backward pass", sync_burn = true).in_scope(|| loss.backward());

        // There's an annoying issue in Burn where the scheduler step
        // is a trait function, which requires the backen to be known,
        // which is otherwise unconstrained, leading to needing this ugly call.
        let lr_mean = LrScheduler::<B>::step(&mut self.sched_mean);
        let lr_opac = LrScheduler::<B>::step(&mut self.sched_opac);
        let lr_rest = LrScheduler::<B>::step(&mut self.sched_rest);

        let mut splats = info_span!("Optimizer step", sync_burn = true).in_scope(|| {
            // Burn doesn't have a great way to use multiple different learning rates
            // or different optimizers. The current best way seems to be to "distribute" the gradients
            // to different GradientParams. Basically each optimizer step call only sees a
            // a subset of parameter gradients.

            let mut grad_means = GradientsParams::new();
            grad_means.register(
                splats.means.id.clone(),
                splats.means.grad_remove(&mut grads).unwrap(),
            );
            let mut grad_opac = GradientsParams::new();
            grad_opac.register(
                splats.raw_opacity.id.clone(),
                splats.raw_opacity.grad_remove(&mut grads).unwrap(),
            );
            let mut grad_rest = GradientsParams::new();
            grad_rest.register(
                splats.sh_coeffs.id.clone(),
                splats.sh_coeffs.grad_remove(&mut grads).unwrap(),
            );
            grad_rest.register(
                splats.rotation.id.clone(),
                splats.rotation.grad_remove(&mut grads).unwrap(),
            );
            grad_rest.register(
                splats.log_scales.id.clone(),
                splats.log_scales.grad_remove(&mut grads).unwrap(),
            );

            let mut splats = splats;
            splats = self.optim.step(lr_mean, splats, grad_means);
            splats = self.optim.step(lr_opac, splats, grad_opac);
            splats = self.optim.step(lr_rest, splats, grad_rest);
            splats
        });

        info_span!("Housekeeping", sync_burn = true).in_scope(|| {
            splats.norm_rotations();

            // TODO: Maybe can batch this.
            let xys_grad = Tensor::from_inner(
                splats
                    .xys_dummy
                    .grad_remove(&mut grads)
                    .expect("XY gradients should be calculated."),
            );

            // TODO: Original implementation has a running average instead. That seems wrong to me -
            // but might need some proper ablation.
            self.xy_grad_norm_accum = Tensor::max_pair(
                self.xy_grad_norm_accum.clone(),
                xys_grad
                    .clone()
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .squeeze(1)
                    .sqrt(),
            );
        });

        if self.iter % self.config.refine_every == 0 {
            // Remove barely visible gaussians.
            let prule_alpha_thresh = self.config.prune_alpha_thresh;
            let alpha_mask = burn::tensor::activation::sigmoid(splats.raw_opacity.val())
                .lower_elem(prule_alpha_thresh);
            self.prune_points(&mut splats, alpha_mask).await;

            let prune_scale_thresh = self.config.prune_scale_thresh;
            let scale_mask = splats
                .log_scales
                .val()
                .exp()
                .max_dim(1)
                .squeeze(1)
                .lower_elem(prune_scale_thresh);
            self.prune_points(&mut splats, scale_mask).await;

            if self.iter > self.config.warmup_steps {
                let max_img_size = img_w.max(img_h) as f32;
                self.densify_and_prune(
                    &mut splats,
                    self.config.clone_split_grad_threshold / max_img_size,
                    Some(self.config.cull_scale_thresh),
                    self.config.split_clone_size_threshold,
                    device,
                )
                .await;

                if self.iter % (self.config.refine_every * self.config.reset_alpha_every) == 0 {
                    self.reset_opacity(&mut splats);
                }
            }

            self.reset_stats(splats.num_splats(), device);
            self.optim = self.opt_config.init::<B, Splats<B>>();
        }

        self.iter += 1;

        let stats = TrainStepStats {
            pred_images,
            auxes,
            loss,
            lr_mean,
            lr_opac,
            lr_rest,
            iter: self.iter,
        };

        Ok((splats, stats))
    }
}
