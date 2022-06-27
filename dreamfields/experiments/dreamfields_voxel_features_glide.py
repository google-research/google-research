# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fit a volume with CLIP and diffusion guidance."""

# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import argparse
import random
from typing import Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from tqdm.auto import trange
import wandb

# pylint: disable=g-bad-import-order
# pylint: disable=g-multiple-import
# Data structures and functions for rendering.
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
                                RayBundle, VolumeRenderer, VolumeSampler,
                                NDCGridRaysampler, EmissionAbsorptionRaymarcher)
from pytorch3d.structures import Volumes

# Import GLIDE.
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import create_model_and_diffusion
from glide_text2im.model_creation import model_and_diffusion_defaults
# pylint: enable=g-multiple-import

from coco_val_queries import queries
from dreamfields_voxel import clamp_and_detach
from dreamfields_voxel import compute_query_rank
from dreamfields_voxel import DEFAULT_VOLUME_SIZE
from dreamfields_voxel import load_clip
from dreamfields_voxel import render_rotating_volume
from dreamfields_voxel import render_validation_view
from dreamfields_voxel import sample_backgrounds
# pylint: enable=g-bad-import-order

parser = argparse.ArgumentParser()
# Logging.
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--exp-name-prefix", type=str, default="")
# Query and seed.
parser.add_argument("--query-idx", type=int)
parser.add_argument("--seed", type=int, default=0)
# Scene parameterization.
parser.add_argument("--volume-size", type=int, default=128)
parser.add_argument(
    "--sample-mode",
    type=str,
    default="bilinear",
    choices=["bilinear", "nearest", "bicubic"],
    help="RGB and density interpolation mode for"
    "torch.nn.functional.grid_sample")
parser.add_argument("--voxel-features-dim", type=int, default=32)
parser.add_argument("--mlp-hidden-dim", type=int, default=64)
# Data augmentation
parser.add_argument("--n-aug", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--crop-scale-range", default=(0.5, 1.0))
parser.add_argument("--nsq", type=int, default=8)
parser.add_argument("--bg-blur-std-range", default=(0., 10.))
# Optimization.
parser.add_argument("--loss-model", type=str, default="ViT-B/16")
parser.add_argument("--retrieve-model", type=str, default="ViT-B/32")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--n-iter", type=int, default=5000)
parser.add_argument("--target-transmittance", type=float, default=0.95)
parser.add_argument("--transmittance-lam", type=float, default=1.0)
parser.add_argument("--density-tvd-lam", type=float, default=0.)
parser.add_argument("--color-tvd-lam", type=float, default=0.)
parser.add_argument("--clip-lam", type=float, default=1.0)
# GLIDE.
parser.add_argument("--guidance-scale", type=float, default=5.0)
parser.add_argument(
    "--t-schedule", type=str, choices=["anneal", "uniform"], default="anneal")
parser.add_argument("--t-respace", type=int, default=100)
parser.add_argument("--t-max", type=int, default=40)  # initial
parser.add_argument("--t-min", type=int, default=1)  # final
parser.add_argument("--denoise-n-steps", type=int, default=1)
parser.add_argument("--denoise-stop-grad", action="store_true")
parser.add_argument("--denoise-augmented", action="store_true")
parser.add_argument("--diffusion-lam", type=float, default=0.1)
# Camera.
# Our rendered scene is centered around (0,0,0)
# and is enclosed inside a bounding box
# whose side is roughly equal to 2.0 (world units).
parser.add_argument("--volume-extent-world", type=float, default=2.0)
parser.add_argument("--camera-radius", type=float, default=4.0)
# render_size describes the size of both sides of the
# rendered images in pixels.
parser.add_argument("--render-size", type=int, default=168)
parser.add_argument("--n-pts-per-ray", type=int, default=192)
args = parser.parse_args()


class FeatureVolumeSampler(VolumeSampler):
  """Sample a batch of volumes `Volumes` at points along projection rays."""

  def __init__(self,
               volumes,
               sample_mode = "bilinear",
               densities_dim = 1,
               output_dim = 3,
               features_dim = args.voxel_features_dim,
               hidden_dim = args.mlp_hidden_dim):
    """Feature volume sampler.

    Args:
      volumes: An instance of the `Volumes` class representing a batch of
        volumes that are being rendered.
      sample_mode: Defines the algorithm used to sample the volumetric voxel
        grid. Can be either "bilinear" or "nearest".
    """
    super().__init__(volumes, sample_mode)

    self.output_dim = output_dim
    self.densities_dim = densities_dim
    self.features_dim = features_dim

    self.output = nn.Linear(features_dim, output_dim)

  def forward(self, ray_bundle,
              **kwargs):
    """Samples `self._volumes` at the respective 3D ray-points.

    Args:
      ray_bundle: A RayBundle object with the following fields:
        rays_origins_world: A tensor of shape `(minibatch, ..., 3)` denoting
          the origins of the sampling rays in world coords.
        rays_directions_world: A tensor of shape `(minibatch, ..., 3)`
          containing the direction vectors of sampling rays in world coords.
        rays_lengths: A tensor of shape `(minibatch, ...,
          num_points_per_ray)` containing the lengths at which the rays are
          sampled.

    Returns:
      rays_densities: A tensor of shape
        `(minibatch, ..., num_points_per_ray, opacity_dim)` containing the
        density vectors sampled from the volume at the locations of the
        ray points.
      rays_features: A tensor of shape
        `(minibatch, ..., num_points_per_ray, feature_dim)` containing
        the feature vectors sampled from the volume at the locations of the
        ray points.
    """
    rays_densities, rays_features = super().forward(ray_bundle, **kwargs)

    # project to output feature dimension
    rays_features = self.output(rays_features)

    return rays_densities, rays_features


class FeatureVolumeRenderer(VolumeRenderer):
  """A class for rendering a batch of Volumes.

  The class should be initialized with a raysampler and a raymarcher class
  which both have to be a `Callable`.
  """

  def forward(self, cameras, volumes,
              **kwargs):
    """Render images using raymarching over rays cast through input `Volumes`.

    Args:
      cameras: A batch of cameras that render the scene. A `self.raysampler`
        takes the cameras as input and samples rays that pass through the
        domain of the volumetric function.
      volumes: An instance of the `Volumes` class representing a batch of
        volumes that are being rendered.

    Returns:
      images: A tensor of shape `(minibatch, ..., (feature_dim +
      opacity_dim)`
          containing the result of the rendering.
      ray_bundle: A `RayBundle` containing the parametrizations of the
          sampled rendering rays.
    """
    volumetric_function = FeatureVolumeSampler(
        volumes, sample_mode=self._sample_mode)
    volumetric_function = volumetric_function.to(volumes.device)
    return self.renderer(
        cameras=cameras, volumetric_function=volumetric_function, **kwargs)


class VolumeModel(torch.nn.Module):

  def __init__(self,
               renderer,
               volume_size=DEFAULT_VOLUME_SIZE,
               voxel_size=0.1,
               features_dim=args.voxel_features_dim):
    super().__init__()
    # Densities close to zero.
    self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
    self.features = torch.nn.Parameter(torch.randn(features_dim, *volume_size))
    self._voxel_size = voxel_size
    # Store the renderer module as well.
    self._renderer = renderer

  def forward(self, cameras, custom_renderer=None):
    batch_size = cameras.R.shape[0]

    # Convert the log-space values to the densities/colors.
    densities = torch.sigmoid(self.log_densities)

    # Instantiate the Volumes object, making sure the densities and colors are
    # correctly expanded batch_size-times.
    volumes = Volumes(
        densities=densities[None].expand(batch_size, *self.log_densities.shape),
        features=self.features[None].expand(batch_size, *self.features.shape),
        voxel_size=self._voxel_size,
    )

    # Given cameras and volumes, run the renderer
    # and return only the first output value (the 2nd output is a representation
    # of the sampled rays which can be omitted for our purpose).
    if custom_renderer is not None:
      return custom_renderer(cameras=cameras, volumes=volumes)[0]

    rendered = self._renderer(cameras=cameras, volumes=volumes)[0]
    rgb, silhouette = rendered.split([3, 1], dim=-1)
    return rgb, silhouette, densities


def get_renderer(resolution, n_pts_per_ray):
  ## Initialize the volumetric renderer
  # The following initializes a volumetric renderer that emits a ray from each
  # pixel of a target image and samples a set of uniformly-spaced points along
  # the ray. At each ray-point, the corresponding density and color value is
  # obtained by querying the corresponding location in the volumetric model of
  # the scene (the model is described & instantiated in a later cell).

  # The renderer is composed of a *raymarcher* and a *raysampler*.
  # - The *raysampler* is responsible for emitting rays from image pixels and
  # sampling the points along them. Here, we use the `NDCGridRaysampler` which
  # follows the standard PyTorch3D coordinate grid convention (+X from right to
  # left; +Y from bottom to top; +Z away from the user).
  # - The *raymarcher* takes the densities and colors sampled along each ray and
  # renders each ray into a color and an opacity value of the ray's source
  # pixel. Here we use the `EmissionAbsorptionRaymarcher` which implements the
  # standard Emission-Absorption raymarching algorithm.

  # Next we instantiate a volumetric model of the scene. This quantizes the 3D
  # space to cubical voxels, where each voxel is described with a 3D vector
  # representing the voxel's RGB color and a density scalar which describes the
  # opacity of the voxel (ranging between [0-1], the higher the more opaque).

  # In order to ensure the range of densities and colors is between [0-1], we
  # represent both volume colors and densities in the logarithmic space. During
  # the forward function of the model, the log-space values are passed through
  # the sigmoid function to bring the log-space values to the correct range.

  # Additionally, `VolumeModel` contains the renderer object. This object stays
  # unaltered throughout the optimization.

  # 1) Instantiate the raysampler.
  # Here, NDCGridRaysampler generates a rectangular image
  # grid of rays whose coordinates follow the PyTorch3D
  # coordinate conventions.
  # Since we use a volume of size 128^3, we sample n_pts_per_ray=150,
  # which roughly corresponds to a one ray-point per voxel.
  # We further set the min_depth=0.1 since there is no surface within
  # 0.1 units of any camera plane.

  # Changing the rendering resolution is a bit involved.
  raysampler = NDCGridRaysampler(
      image_width=resolution,
      image_height=resolution,
      n_pts_per_ray=n_pts_per_ray,
      min_depth=args.camera_radius - args.volume_extent_world * np.sqrt(3) / 2,
      max_depth=args.camera_radius + args.volume_extent_world * np.sqrt(3) / 2,
  )

  # 2) Instantiate the raymarcher.
  # Here, we use the standard EmissionAbsorptionRaymarcher
  # which marches along each ray in order to render
  # each ray into a single 3D color vector
  # and an opacity scalar.
  raymarcher = EmissionAbsorptionRaymarcher()

  # Finally, instantiate the volumetric render
  # with the raysampler and raymarcher objects.

  renderer = FeatureVolumeRenderer(
      raysampler=raysampler,
      raymarcher=raymarcher,
      sample_mode=args.sample_mode)
  return renderer


def load_diffusion(name, device):
  """Create base model."""
  options = model_and_diffusion_defaults()
  options["inpaint"] = False
  options["use_fp16"] = has_cuda
  options["timestep_respacing"] = str(
      args.t_respace)  # Use few diffusion steps for fast sampling.
  model, diffusion = create_model_and_diffusion(**options)
  model.eval()
  if has_cuda:
    model.convert_to_fp16()
  model.to(device)
  model.load_state_dict(load_checkpoint(name, device))
  print("total base parameters", sum(x.numel() for x in model.parameters()))
  return model, diffusion, options


if __name__ == "__main__":
  ## 1. Setup logging.
  query = queries[args.query_idx]
  exp_name = f"{args.exp_name_prefix}_{args.query_idx:03d}_{query}"
  wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=exp_name,
      config=args)

  # Reproducibility.
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)

  ## 2. Obtain the utilized device.
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    has_cuda = True
  else:
    device = torch.device("cpu")
    has_cuda = False

  ## Initialize CLIP.
  image_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device)
  image_mean = image_mean[None, :, None, None]
  image_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
  image_std = image_std[None, :, None, None]

  if args.clip_lam:
    model, preprocess, clip_size = load_clip(args.loss_model, device)

  if args.retrieve_model == args.loss_model and args.clip_lam:
    test_model, test_preprocess, test_clip_size = model, preprocess, clip_size
  else:
    test_model, test_preprocess, test_clip_size = load_clip(
        args.retrieve_model, device)

  ## Initialize GLIDE.
  # Create base model.
  base_glide_model, diffusion, base_glide_options = load_diffusion(
      "base", device)
  base_glide_model.eval()

  ## Initialize the volumetric model.
  renderer = get_renderer(args.render_size, args.n_pts_per_ray)

  # Here we carry out the volume fitting with differentiable rendering.

  # Instantiate the volumetric model.
  volume_model = VolumeModel(
      renderer,
      volume_size=[args.volume_size] * 3,
      voxel_size=args.volume_extent_world / args.volume_size,
  ).to(device)

  def get_parameters():
    return volume_model.parameters()

  # Instantiate the Adam optimizer.
  optimizer = torch.optim.Adam(get_parameters(), lr=args.lr)

  ## Embed the target caption with CLIP.
  if args.clip_lam:
    query_tok = clip.tokenize(query).to(device)
    z_clip = model.encode_text(query_tok).detach()
    z_clip = F.normalize(z_clip, dim=-1)

  ## Embed the target caption with GLIDE.
  denoise_batch_size = (
      args.n_aug *
      args.batch_size if args.denoise_augmented else args.batch_size)
  tokens = base_glide_model.tokenizer.encode(query)
  tokens, mask = base_glide_model.tokenizer.padded_tokens_and_mask(
      tokens, base_glide_options["text_ctx"])

  # Create the classifier-free guidance tokens (empty).
  full_batch_size = denoise_batch_size * 2
  uncond_tokens, uncond_mask = (
      base_glide_model.tokenizer.padded_tokens_and_mask(
          [], base_glide_options["text_ctx"]))

  # Pack the tokens together into model kwargs.
  base_model_kwargs = dict(
      tokens=torch.tensor(
          [tokens] * denoise_batch_size + [uncond_tokens] * denoise_batch_size,
          device=device),
      mask=torch.tensor(
          [mask] * denoise_batch_size + [uncond_mask] * denoise_batch_size,
          dtype=torch.bool,
          device=device,
      ),
  )

  # Create an classifier-free guidance sampling function.
  def base_model_fn(x_t, ts, **kwargs):
    half = x_t[:len(x_t) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = base_glide_model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

  def preprocess_glide(x, order="NHWC"):
    if order == "NHWC":
      # x is [NHWC]. Reshape to NCHW.
      x = x.movedim(-1, 1)
    x = x * 2 - 1  # Scale from [0, 1] to [-1, 1].
    x = F.interpolate(x, (64, 64), mode="bilinear")
    return x

  def unprocess_glide(x):
    # x is [NCHW] Reshape to [NHWC].
    x = x.movedim(1, -1)
    x = (x + 1) / 2  # Scale from [-1, 1] to [0, 1].
    return x

  denoised_fn = lambda x_start: x_start

  glide_context_manager = (
      torch.no_grad if args.denoise_stop_grad else torch.enable_grad)

  ## We do Adam iterations and sample 10 random images in each minibatch.
  if args.clip_lam:
    clip_aug_fn = torchvision.transforms.RandomResizedCrop(
        clip_size, scale=args.crop_scale_range, ratio=(1.0, 1.0))
  denoise_aug_fn = torchvision.transforms.RandomResizedCrop(
      64, scale=args.crop_scale_range, ratio=(1.0, 1.0))

  pbar = trange(1, args.n_iter + 1)
  for iteration in pbar:
    metrics = {}
    visualize_images = iteration % 100 == 0 or iteration == 1

    # In case we reached the last 75% of iterations,
    # decrease the learning rate of the optimizer 10-fold.
    if iteration == round(args.n_iter * 0.75):
      tqdm.write("Decreasing LR 10-fold ...")
      for g in optimizer.param_groups:
        g["lr"] = args.lr * 0.1

    # Zero the optimizer gradient.
    optimizer.zero_grad()

    # Get a batch of viewing angles.
    elev = torch.rand(args.batch_size) * 10 + 20.
    azim = torch.rand(args.batch_size) * 360
    radius_mult = 1 + 0.1 * (torch.rand(args.batch_size) - 0.5)
    radius = radius_mult * args.camera_radius / 1.2

    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)
    batch_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Evaluate the volumetric model.
    rendered_images, rendered_silhouettes, densities = volume_model(
        batch_cameras)

    # Transmittance loss.
    tau = 1 - (1 - args.target_transmittance) / (radius_mult**2)
    tau = tau.to(device)
    avg_transmittance = 1 - rendered_silhouettes.mean(
        dim=tuple(range(1, rendered_silhouettes.ndim)))
    transmittance_loss = -torch.min(tau, avg_transmittance).mean()

    if (args.diffusion_lam > 0 and args.denoise_augmented) or args.clip_lam > 0:
      # NOTE: this background is at the render resolution, not the resize.
      # Generate random backgrounds.
      bgs = sample_backgrounds(
          args.n_aug * args.batch_size, args.render_size, device=device)

      # Composite renderings with backgrounds.
      bgs = bgs.view(args.n_aug, args.batch_size, *bgs.shape[1:])  # A,N,C,H,W
      bgs = bgs.movedim(2, -1)  # ANCHW to ANHWC
      composite_images = (
          rendered_silhouettes[None] * rendered_images[None] +
          (1 - rendered_silhouettes[None]) * bgs)
      composite_images = composite_images.reshape(  # to A*N,H,W,C
          args.n_aug * args.batch_size, args.render_size, args.render_size, 3)
      composite_images = composite_images.movedim(3, 1)  # NHWC to NCHW.

    # Compute GLIDE loss.
    # Sample from the base model.
    if args.diffusion_lam:
      base_glide_model.del_cache()
      if args.denoise_augmented:
        denoise_aug_images = denoise_aug_fn(composite_images)
        x = preprocess_glide(denoise_aug_images, order="NCHW")
      else:
        # White background.
        x = rendered_silhouettes * rendered_images + 1 - rendered_silhouettes
        x = preprocess_glide(x, order="NHWC")

      inp = x.detach() if args.denoise_stop_grad else x

      assert args.t_schedule == "anneal"
      progress = (iteration - 1) / (args.n_iter - 1)
      start_t = int(args.t_max - (args.t_max - args.t_min) * progress)
      diffuse_steps = []
      denoise_steps = []
      with glide_context_manager():
        for t in torch.linspace(
            start_t, args.t_min, args.denoise_n_steps, dtype=int):
          inp = diffusion.q_sample(
              inp, torch.tensor([t] * denoise_batch_size, device=device))
          diffuse_steps.append(inp.detach().to("cpu", non_blocking=True))
          inp = diffusion.p_sample(
              base_model_fn,
              x=torch.cat([inp, inp], dim=0),
              t=torch.tensor([t] * full_batch_size, device=device),
              clip_denoised=True,
              model_kwargs=base_model_kwargs,
              cond_fn=None,
              denoised_fn=denoised_fn,
          )["pred_xstart"][:denoise_batch_size]
          denoise_steps.append(inp.detach().to("cpu", non_blocking=True))
      base_glide_model.del_cache()
      diffusion_loss = F.mse_loss(x, inp)
      metrics["diffusion/start_t"] = start_t
      metrics["loss/diffusion_mse"] = diffusion_loss
    else:
      diffusion_loss = torch.tensor([0.], device=device)

    # Compute the CLIP loss.
    if args.clip_lam:
      clip_aug_images = clip_aug_fn(composite_images)
      x = preprocess(clip_aug_images)  # Resize and normalize.
      z_est = model.encode_image(x)
      z_est = F.normalize(z_est, dim=-1)
      clip_loss = -torch.sum(z_est * z_clip, dim=-1).mean()
    else:
      clip_loss = torch.tensor([0.], device=device)

    # Compute total loss and take an optimization step.
    loss = (
        args.clip_lam * clip_loss +
        args.transmittance_lam * transmittance_loss +
        args.diffusion_lam * diffusion_loss)

    if args.density_tvd_lam:
      loss.backward()

      with torch.no_grad():
        d = densities.squeeze(0)
        dc = torch.cat([d, d.transpose(1, 2), d.transpose(0, 2)], axis=-1)
        density_tvd_loss = torch.square(dc[Ellipsis, :-1] - dc[Ellipsis, 1:]).sum()
        loss = loss + args.density_tvd_lam * density_tvd_loss
        metrics["loss/density_tvd"] = density_tvd_loss.item()

      grad_sigmoid = lambda x: torch.exp(-x) / torch.square(1 + torch.exp(-x))
      log_d = volume_model.log_densities
      grad = log_d.grad
      grad_in = grad_sigmoid(log_d.detach())  # d log(density) / d density.
      s = args.density_tvd_lam * 2  # gradient below assumes scale is 1/2.

      d = densities.detach().squeeze(0)

      grad[0, :-1] += s * (d[:-1] - d[1:]) * grad_in[0, :-1]
      grad[0, 1:] += s * (d[1:] - d[:-1]) * grad_in[0, 1:]

      grad[0, :, :-1] += s * (d[:, :-1] - d[:, 1:]) * grad_in[0, :, :-1]
      grad[0, :, 1:] += s * (d[:, 1:] - d[:, :-1]) * grad_in[0, :, 1:]

      grad[0, :, :, :-1] += s * (d[:, :, :-1] -
                                 d[:, :, 1:]) * grad_in[0, :, :, :-1]
      grad[0, :, :,
           1:] += s * (d[:, :, 1:] - d[:, :, :-1]) * grad_in[0, :, :, 1:]
    else:
      loss.backward()

    optimizer.step()

    ## Logging.
    with torch.no_grad():
      metrics.update({
          "loss/total_loss": loss.item(),
          "loss/clip": clip_loss.item(),
          "loss/transmittance": transmittance_loss.item(),
      })

      # Print the current values of the losses.
      if iteration % 10 == 0:
        avg_transmittance_float = float(avg_transmittance.mean().item())
        pbar.set_description(
            f"Iteration {iteration:05d}:" +
            f" clip_loss = {float(clip_loss.item()):1.2f}" +
            f" diffusion_loss = {float(diffusion_loss.item()):1.5f}" +
            f" avg transmittance = {avg_transmittance_float:1.2f}" +
            f" tau = {str(tau)}")

      # Visualize the renders every 100 iterations.
      if visualize_images:
        wandb_image = lambda x: wandb.Image(clamp_and_detach(x))

        # Visualize only a single randomly selected element of the batch.
        im_show_idx = int(torch.randint(low=0, high=args.batch_size, size=(1,)))

        rendering = rendered_images[im_show_idx]
        silhouette = rendered_silhouettes[im_show_idx]
        metrics["render/rendered"] = wandb.Image(clamp_and_detach(rendering))

        if args.clip_lam > 0:
          aug_image = clip_aug_images[im_show_idx].movedim(0, 2)
          metrics["render/augmented"] = wandb_image(aug_image)

        render_white_bg = rendering * silhouette + 1 - silhouette
        metrics["render/rendered_white_bg"] = wandb_image(render_white_bg)
        metrics["render/rendered_silhouettes"] = wandb_image(
            silhouette.squeeze(-1))

        diffused = torch.stack(
            diffuse_steps, dim=1)  # [batch_size, n_steps, C, 64, 64].
        diffused = unprocess_glide(
            diffused[im_show_idx])  # [n_steps, 64, 64, C]
        diffused = diffused.movedim(0, 1).reshape(64, args.denoise_n_steps * 64,
                                                  3)
        metrics["render/diffused"] = wandb_image(diffused)

        denoised = torch.stack(
            denoise_steps, dim=1)  # [batch_size, n_steps, C, 64, 64].
        denoised = unprocess_glide(
            denoised[im_show_idx])  # [n_steps, 64, 64, C]
        denoised = denoised.movedim(0, 1).reshape(64, args.denoise_n_steps * 64,
                                                  3)
        metrics["render/denoised"] = wandb_image(denoised)

      # Validate from a held-out view.
      if iteration % 250 == 0 or iteration == 1:
        validation_view = render_validation_view(
            volume_model, test_clip_size, device=device)
        metrics["val/render"] = wandb.Image(clamp_and_detach(validation_view))

        rank, cosine_sim = compute_query_rank(
            test_model,
            test_preprocess,
            render=validation_view.movedim(-1, 0).unsqueeze(0),
            query=query,
            queries_r=queries,
            device=device)

        metrics["val/rank"] = rank
        metrics["val/acc"] = int(rank == 0)
        metrics["val/cosine_sim"] = cosine_sim

      if iteration % 500 == 0 or iteration == 1:
        # Visualize the optimized volume by rendering from multiple viewpoints
        # that rotate around the volume's y-axis.
        rotating_volume_frames = render_rotating_volume(
            volume_model, n_frames=240, device=device)
        metrics["render/video"] = wandb.Video(
            (rotating_volume_frames * 255.).astype(np.uint8),
            fps=30,
            format="mp4")

      wandb.log(metrics, iteration)
