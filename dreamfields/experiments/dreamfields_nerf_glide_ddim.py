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

"""Partial port Dream Fields to PyTorch, with some experimental features."""

import argparse
import math
import random

import clip
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import wandb

# pylint: disable=g-bad-import-order
# Import GLIDE.
from glide_text2im.download import load_checkpoint
from glide_text2im.gaussian_diffusion import _extract_into_tensor
from glide_text2im.model_creation import create_model_and_diffusion
from glide_text2im.model_creation import model_and_diffusion_defaults

from coco_val_queries import queries
import augment
import nerf
import scene
import schedule
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
parser.add_argument(
    "--volume-extent-world",
    type=float,
    default=2.0,
    help="Our rendered scene is centered around (0,0,0) and is"
    "enclosed inside a bounding box whose side is roughly"
    "equal to 2.0 (world units).")
parser.add_argument("--scene-extent", type=float, default=1.)
parser.add_argument("--track-scene-origin", action="store_true")
# Camera.
parser.add_argument("--camera-radius", type=float, default=4.0)
parser.add_argument("--n-pts-per-ray", type=int, default=128)
parser.add_argument("--azimuth-range", type=float, default=[0, 360], nargs="+")
parser.add_argument(
    "--elevation-range", type=float, default=[-30, -30], nargs="+")
parser.add_argument(
    "--render-size",
    type=int,
    default=64,
    help="Describes the size of both"
    "sides of the rendered images in pixels.")
parser.add_argument(
    "--video-size",
    type=int,
    default=96,
    help="Describes the size of both sides of the"
    "rendered frames of a rotating video in pixels.")
parser.add_argument("--video-n-frames", type=int, default=48)
parser.add_argument(
    "--max-validation-size",
    type=int,
    default=128,
    help="Describes the size of both sides of the"
    "rendered frames of a validation view in pixels.")
# Positional encoding.
parser.add_argument("--posenc-deg", type=int, default=8)
parser.add_argument("--fourfeat", action="store_true")
parser.add_argument("--ipe", action="store_true")  # integrated pos enc
# Data augmentation.
parser.add_argument("--n-aug", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument(
    "--n-views", type=int, default=8, help="Number of fixed views to render.")
parser.add_argument(
    "--n-optimize", type=int, default=8, help="Number of views to optimize.")
parser.add_argument("--crop-scale-range", default=(0.5, 1.0))
parser.add_argument("--nsq", type=int, default=8)
parser.add_argument("--bg-blur-std-range", default=(0., 10.))
# Optimization.
parser.add_argument("--loss-model", type=str, default="ViT-B/16")
parser.add_argument("--retrieve-model", type=str, default="ViT-B/32")
parser.add_argument(
    "--lr-init", type=float, default=5e-4, help="The initial learning rate.")
parser.add_argument(
    "--lr-final", type=float, default=5e-5, help="The final learning rate.")
parser.add_argument(
    "--lr-delay-mult",
    type=float,
    default=0.01,
    help="How severe the warmup should be.")
parser.add_argument("--adam-eps", type=float, default=1e-8)
parser.add_argument("--n-iter", type=int, default=5000)
parser.add_argument("--target-transmittance0", type=float, default=0.5)
parser.add_argument("--target-transmittance1", type=float, default=0.9)
parser.add_argument(
    "--target-transmittance-anneal-iters", type=int, default=500)
parser.add_argument("--transmittance-lam", type=float, default=0.1)
parser.add_argument("--clip-lam", type=float, default=1.0)
parser.add_argument("--sigma-noise-std", type=float, default=0.)
parser.add_argument("--denoise-every", type=int, default=1)
# GLIDE.
parser.add_argument("--guidance-scale", type=float, default=5.0)
parser.add_argument(
    "--t-schedule", type=str, choices=["anneal", "uniform"], default="anneal")
parser.add_argument("--t-respace", type=int, default=100)
parser.add_argument("--denoise-n-steps", type=int, default=1)
parser.add_argument("--denoise-stop-grad", action="store_true")
parser.add_argument("--denoise-augmented", action="store_true")
parser.add_argument("--diffusion-lam", type=float, default=0.1)
parser.add_argument(
    "--ddim-eta",
    type=float,
    default=0.0,
    help="DDIM eta parameter in [0, 1], controlling degree "
    "of stochasticity")
parser.add_argument(
    "--independent-sampling-steps",
    type=int,
    default=0,
    help="Number of steps to sample viewpoints without NeRF")
args = parser.parse_args()


def quantize(image):
  """Quantize a [0, 1] scaled float image to uint8 [0, 255]."""
  return (image * 255.).astype(np.uint8)


def clamp_and_detach(a):
  return a.clamp(0.0, 1.0).cpu().detach().numpy()


def render_rotating_volume(volume_model, scene_origin, n_frames,
                           video_size, **render_kwargs):
  """Render frames from a camera orbiting the volume."""

  print("Rendering rotating volume ...")
  render_azimuths = np.linspace(0., 360., n_frames)
  cam2worlds = [
      scene.pose_spherical(azim, phi=-30, radius=4) for azim in render_azimuths
  ]
  frames = []

  height, width, focal = scene.scale_intrinsics(video_size)

  for cam2world in tqdm.tqdm(cam2worlds, desc="Rendering rotating volume"):
    rays = scene.camera_rays(cam2world, height, width, focal)
    rendered, _ = nerf.render_rays_mip(
        rays,
        volume_model,
        origin=scene_origin.value,
        white_bkgd=True,
        **render_kwargs)  # rgb, depth, disparity, silhouette
    rendered = torch.cat(rendered, dim=-1)  # [H, W, 6]
    frames.append(rendered)

  frames = torch.stack(frames, dim=0)  # [n_frames, H, W, 6]
  frames = frames.cpu()
  rgb, depth, disparity, silhouette = torch.split(frames, [3, 1, 1, 1], dim=-1)
  return (
      rgb.numpy(),  # [T, H, W, 3]
      depth.numpy(),  # [T, H, W, 1]
      disparity.numpy(),  # [T, H, W, 1]
      silhouette.numpy())  # [T, H, W, 1]


def render_validation_view(volume_model, scene_origin, render_size, max_size,
                           **render_kwargs):
  """Render a frame from a camera at a new perspective."""
  cam2world = scene.pose_spherical(theta=30, phi=-45, radius=4)
  height, width, focal = scene.scale_intrinsics(min(render_size, max_size))
  rays = scene.camera_rays(cam2world, height, width, focal)
  (rgb, _, _, _), _ = nerf.render_rays_mip(
      rays,
      volume_model,
      origin=scene_origin.value,
      white_bkgd=True,
      chunksize=2**17,
      **render_kwargs)  # [H, W, 6]

  if height != render_size:
    rgb = rgb.movedim(-1, 0)[None]  # HWC to 1CHW
    rgb = F.interpolate(rgb, render_size, mode="bilinear")
    rgb = rgb[0].movedim(0, -1)  # 1CHW to HWC

  assert rgb.ndim == 3
  return rgb


def load_clip(name, device):
  """Load CLIP models."""
  image_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device)
  image_mean = image_mean[None, :, None, None]
  image_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
  image_std = image_std[None, :, None, None]

  model, _ = clip.load(name, device=device, jit=False)

  def preprocess(images):
    images = F.interpolate(
        images, size=model.visual.input_resolution, mode="bicubic")
    images = (images - image_mean) / image_std
    return images

  return model, preprocess, model.visual.input_resolution


def load_diffusion(name, device, has_cuda):
  """Create base model."""
  options = model_and_diffusion_defaults()
  options["inpaint"] = False
  options["use_fp16"] = has_cuda
  options["timestep_respacing"] = str(
      args.t_respace)  # use few diffusion steps for fast sampling
  model, diffusion = create_model_and_diffusion(**options)
  model.load_state_dict(load_checkpoint(name, torch.device("cpu")))
  model.eval()
  if has_cuda:
    model.convert_to_fp16()
  model.to(device)
  print("total base parameters", sum(x.numel() for x in model.parameters()))
  return model, diffusion, options


def embed_queries(model, queries_list, device):
  """Embed captions."""
  # TODO(jainajay): This can be cached.
  queries_tok = clip.tokenize(queries_list).to(device)
  z_queries = model.encode_text(queries_tok).detach()
  z_queries = F.normalize(z_queries, dim=-1)
  return z_queries


def compute_query_rank(model, preprocess, render, query, queries_r, device):
  """Compute rank of `query` among `queries_r` according to CLIP.

  The score <CLIP_image(render), CLIP_text(query)> is used for ranking.

  Args:
    model:
    preprocess:
    render:
    query:
    queries_r:
    device:

  Returns:
    rank (int), cosine_similarity (float)
  """
  if query not in queries_r:
    print(f"WARN: query \"{query}\" not in retrieval set. Adding it.")
    queries_r = queries_r + [query]
    query_idx = len(queries_r) - 1
  else:
    query_idx = queries_r.index(query)

  with torch.inference_mode():
    # Embed the retrieval set of captions.
    z_queries = embed_queries(model, queries_r, device)

    # Embed render.
    assert render.ndim == 4
    assert render.shape[1] == 3
    x = preprocess(render)
    z_render = model.encode_image(x)
    z_render = F.normalize(z_render, dim=-1)

    sim = torch.sum(z_render * z_queries, dim=-1)
    ranks = torch.argsort(sim, dim=0, descending=True)
    return torch.nonzero(ranks == query_idx)[0].item(), sim[query_idx].item()


def grid(a, **kwargs):
  if a.shape[-1] == 3 or a.shape[-1] == 1:
    a = a.movedim(-1, 1)  # NHWC to NCHW.
  a = torchvision.utils.make_grid(a, **kwargs)  # NCHW to CH'W'.
  return a.movedim(0, 2)  # Return in HWC order.


wandb_image = lambda a: wandb.Image(clamp_and_detach(a))
wandb_grid = lambda a, nrow=8: wandb_image(grid(a, nrow=nrow))


def main():
  # Reproducibility.
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)

  # Obtain the utilized device.
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    has_cuda = True
  else:
    device = torch.device("cpu")
    has_cuda = False

  # Setup logging.
  query = queries[args.query_idx]
  exp_name = f"{args.exp_name_prefix}_{args.query_idx:03d}_{query}"
  wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=exp_name,
      config=args)

  # Initialize CLIP
  if args.clip_lam:
    model, preprocess, clip_size = load_clip(args.loss_model, device)
    model.eval()

  if args.retrieve_model == args.loss_model and args.clip_lam:
    test_model, test_preprocess, test_clip_size = model, preprocess, clip_size
  else:
    test_model, test_preprocess, test_clip_size = load_clip(
        args.retrieve_model, device)
    test_model.eval()

  # Initialize the volumetric model.
  volume_model = nerf.DreamFieldsMLP(
      activation="SiLU",
      features_early=[96],  # Dense layers before residual blocks.
      features_residual=[(128, 96)] * 3,  # Resid block feature dimensions.
      features_late=[96, 4],  # Features dimensions at end.
      fourfeat=args.fourfeat,
      max_deg=args.posenc_deg,
      ipe=args.ipe,
  )
  volume_model = nn.DataParallel(volume_model)
  volume_model = volume_model.to(device)
  scene_origin = scene.EMA(np.zeros(3, dtype=np.float64), decay=0.999)
  render_kwargs = dict(
      sigma_noise_std=args.sigma_noise_std,
      near=4. - math.sqrt(3) * args.volume_extent_world / 2,
      far=4. + math.sqrt(3) * args.volume_extent_world / 2,
      mask_rad=args.volume_extent_world / 2,
      n_pts_per_ray=args.n_pts_per_ray,
      device=device,
  )

  # Instantiate the Adam optimizer.
  optimizer = torch.optim.Adam(
      volume_model.parameters(), lr=args.lr_init, eps=args.adam_eps)
  scaler = torch.cuda.amp.GradScaler()

  # Embed the target caption with CLIP.
  if args.clip_lam:
    query_tok = clip.tokenize(query).to(device)
    z_clip = model.encode_text(query_tok).detach()
    z_clip = F.normalize(z_clip, dim=-1)

    clip_aug_fn = torchvision.transforms.RandomResizedCrop(
        clip_size, scale=args.crop_scale_range, ratio=(1.0, 1.0))

  if args.diffusion_lam:
    # Initialize GLIDE. Create base model.
    base_glide_model, diffusion, base_glide_options = load_diffusion(
        "base", device, has_cuda=has_cuda)
    base_glide_model.eval()

    # Embed the target caption with GLIDE.
    denoise_batch_size = (
        args.n_aug * args.n_views if args.denoise_augmented else args.n_views)
    tokens = base_glide_model.tokenizer.encode(query)
    tokens, mask = base_glide_model.tokenizer.padded_tokens_and_mask(
        tokens, base_glide_options["text_ctx"])

    # Create the classifier-free guidance tokens (empty).
    uncond_tokens, uncond_mask = base_glide_model.tokenizer.padded_tokens_and_mask(
        [], base_glide_options["text_ctx"])

    # Pack the tokens together into model kwargs.
    base_model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * denoise_batch_size +
            [uncond_tokens] * denoise_batch_size,
            device=device),
        mask=torch.tensor(
            [mask] * denoise_batch_size + [uncond_mask] * denoise_batch_size,
            dtype=torch.bool,
            device=device),
    )

    parallel_glide = nn.DataParallel(base_glide_model)

    # Create an classifier-free guidance sampling function.
    def base_model_fn(x_t, ts, **kwargs):
      half = x_t[:len(x_t) // 2]
      combined = torch.cat([half, half], dim=0)
      model_out = parallel_glide(combined, ts, **kwargs)
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
      if x.shape[-2:] != (64, 64):
        x = F.interpolate(x, (64, 64), mode="bilinear")
      return x

    def unprocess_glide(x):
      return (x + 1) / 2  # Scale from [-1, 1] to [0, 1].

    denoised_fn = lambda x_start: x_start
    denoise_aug_fn = torchvision.transforms.RandomResizedCrop(
        64, scale=args.crop_scale_range, ratio=(1.0, 1.0))

    glide_context_manager = (
        torch.no_grad if args.denoise_stop_grad else torch.enable_grad)

    # Initialize each chain.
    diffusion_x = torch.randn((args.n_views, 3, 64, 64),
                              device=device,
                              requires_grad=False)
    diffusion_t = torch.full(
        size=(args.n_views,),
        fill_value=args.t_respace - 1,
        requires_grad=False,
        dtype=torch.long,
        device=device)

    # Training uses n_iter iterations: 1 to n_iter (inclusive).
    # Diffusion uses t_respace timesteps: t_respace-1 to 0 (inclusive).
    # For now, check they are equal.
    # TODO(jainajay): implement sampling with non-unit timesteps.
    assert args.t_respace * args.denoise_every == args.n_iter

  # Get a batch of viewing angles and pre-generate rays.
  azimuths = np.arange(args.n_views) * 360. / args.n_views
  rads = np.full(args.n_views, 4.)
  focal_mults = np.full(args.n_views, 1.2)
  elevations = [
      scene.uniform_in_interval(args.elevation_range)
      for _ in range(args.n_views)
  ]
  cam2worlds = [
      scene.pose_spherical(azim, phi=elev, radius=rad)
      for azim, elev, rad in zip(azimuths, elevations, rads)
  ]
  height, width, focal = scene.scale_intrinsics(args.render_size)
  # Generate rays: 3-tuple of [n_views, H, W, n_pts_per_ray, 3 or 1].
  rays_all_views = scene.camera_rays_batched(cam2worlds, height, width,
                                             focal_mults * focal)

  pbar = tqdm.trange(1, args.n_iter + 1)
  for iteration in pbar:
    metrics = {}
    visualize_images = iteration % 25 == 0 or iteration == 1

    # Set learning rate
    lr = schedule.learning_rate_decay(
        iteration,
        args.lr_init,
        args.lr_final,
        args.n_iter,
        lr_delay_steps=min(args.n_iter // 8, 2500),
        lr_delay_mult=args.lr_delay_mult)
    for g in optimizer.param_groups:
      g["lr"] = float(lr)

    # Zero the optimizer gradient.
    optimizer.zero_grad()

    # Render the volumetric model from random perspectives.
    batch_idx = np.random.choice(
        args.n_views, size=args.batch_size, replace=False)
    rays_batched = [r[batch_idx] for r in rays_all_views]

    # Runs the forward pass with automatic precision casting.
    with torch.cuda.amp.autocast():
      (images, depths, disparities, silhouettes), _ = nerf.render_rays_mip(
          rays_batched,
          volume_model,
          origin=scene_origin.value,
          **render_kwargs)
      assert images.ndim == 4
      assert images.shape[0] == args.batch_size
      assert images.shape[-1] == 3

      # Transmittance loss. Anneal target opacity (1 - transmittance).
      target_opacity = schedule.anneal_logarithmically(
          iteration, args.target_transmittance_anneal_iters,
          1 - args.target_transmittance0, 1 - args.target_transmittance1)

      # The area of an object on the image plane grows with the focal length
      # and shrinks with increasing camera radius. Scale target opacity
      # proportionally with the squared focal multiplier and inversely
      # proportionally with the squared camera radius.
      target_opacities = np.minimum(
          np.ones(args.batch_size), focal_mults[batch_idx]**2 /
          (rads[batch_idx] / 4.)**2 * target_opacity)
      taus = torch.tensor(1 - target_opacities, device=device)
      avg_transmittance = 1 - silhouettes.mean(
          dim=tuple(range(1, silhouettes.ndim)))
      # NOTE(jainajay): Using a modified, two-sided transmittance loss that
      # differs from Dream Fields. It can encourage reducing transmittance if
      # the scene becomes too sparse. The original loss would penalize
      # -torch.mean(torch.min(avg_transmittance, taus)).
      transmittance_loss = torch.mean(torch.abs(avg_transmittance - taus))

      # Data augmentation.
      if (args.diffusion_lam > 0 and
          args.denoise_augmented) or args.clip_lam > 0:
        # NOTE(jainajay): this background is at the render resolution,
        #       not the resize, unlike Dream Fields.
        # Generate random backgrounds.
        bgs = augment.sample_backgrounds(
            num=args.n_aug * args.batch_size,
            res=args.render_size,
            checkerboard_nsq=args.nsq,
            min_blur_std=args.bg_blur_std_range[0],
            max_blur_std=args.bg_blur_std_range[1],
            device=device)

        # Composite renders with backgrounds.
        bgs = bgs.view(args.n_aug, args.batch_size, *bgs.shape[1:])  # ANCHW.
        bgs = bgs.movedim(2, -1)  # Convert ANCHW to ANHWC.
        composite_images = (
            silhouettes[None] * images[None] + (1 - silhouettes[None]) * bgs)
        composite_images = composite_images.reshape(  # to A*N,H,W,C.
            args.n_aug * args.batch_size, args.render_size, args.render_size, 3)
        composite_images = composite_images.movedim(3, 1)  # NHWC to NCHW.

      # Compute GLIDE loss.
      # Sample from the base model.
      if args.diffusion_lam:
        # Preprocess rendering (scale to [-1, 1]).
        if args.denoise_augmented:
          denoise_aug_images = denoise_aug_fn(composite_images)
          inp = preprocess_glide(denoise_aug_images, order="NCHW")
        else:
          inp = silhouettes * images + 1 - silhouettes  # white bg
          inp = preprocess_glide(inp, order="NHWC")

        if (iteration - 1) % args.denoise_every == 0:
          base_glide_model.del_cache()

          # Sampling step for every view in the cache.
          with glide_context_manager():
            assert diffusion_t.dtype == torch.long
            assert torch.all(diffusion_t == diffusion_t[0])
            metrics["diffusion/t"] = diffusion_t[0].item()

            xt = diffusion_x  # || x_hat(x_t) - render ||^2

            # Enable for loss: || x_hat(diffuse(render)) - x_hat(x_t) ||^2
            # x = diffusion.q_sample(
            #     inp, torch.tensor([diffusion_t] * denoise_batch_size,
            #     device=device))

            # Sample x_s from x_t using DDIM.
            # Based on glide-text2im/glide_text2im/gaussian_diffusion.py#L453
            assert args.batch_size == args.n_views  # Updating all chains.
            out = diffusion.p_mean_variance(
                base_model_fn,
                torch.cat([xt, xt], dim=0),
                torch.cat([diffusion_t, diffusion_t], dim=0),
                clip_denoised=True,
                denoised_fn=denoised_fn,  # TODO(jainajay): look into this,
                model_kwargs=base_model_kwargs,
            )
            assert out["pred_xstart"].shape[0] == 2 * args.batch_size
            pred_xstart = out["pred_xstart"][:args.batch_size]

            if iteration < args.independent_sampling_steps * args.denoise_every:
              # Ours: eps = pred_eps(x_t, t, tilde_x).
              # Ours: x_{t-1} = a * tilde_x + b * eps + sigma * noise.
              x0_for_sampling = pred_xstart
            else:
              # GLIDE: eps = pred_eps(x_t, t, x_hat(x_t)).
              # GLIDE: x_{t-1} = a * x_hat(x_t) + b * eps + sigma * noise.
              x0_for_sampling = inp.detach()

            # pylint: disable=protected-access
            eps = diffusion._predict_eps_from_xstart(diffusion_x, diffusion_t,
                                                     x0_for_sampling)
            # pylint: enable=protected-access

            assert eps.shape[0] == args.batch_size

            alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod,
                                             diffusion_t, xt.shape)
            metrics["diffusion/alpha_bar"] = alpha_bar.mean().item()
            alpha_bar_prev = _extract_into_tensor(diffusion.alphas_cumprod_prev,
                                                  diffusion_t, xt.shape)
            metrics["diffusion/alpha_bar_prev"] = alpha_bar_prev.mean().item()
            sigma = (
                args.ddim_eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar)) *
                torch.sqrt(1 - alpha_bar / alpha_bar_prev))
            metrics["diffusion/sigma"] = sigma.mean().item()
            # Equation 12.
            mean_pred = (
                x0_for_sampling * torch.sqrt(alpha_bar_prev) +
                torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
            nonzero_mask = (
                (diffusion_t != 0).float().view(-1,
                                                *([1] * (len(xt.shape) - 1)))
            )  # No noise when t == 0.
            noise = torch.randn_like(xt)
            sample = mean_pred + nonzero_mask * sigma * noise

            # Update multiview sampling chains.
            diffusion_x_prev = diffusion_x
            diffusion_x = sample
            diffusion_t = diffusion_t - 1

            # Don't backprop through the denoiser (forces stop_grad True).
            assert args.denoise_stop_grad
            pred_xstart = pred_xstart.detach()

          base_glide_model.del_cache()

        # Loss: ||x_hat(x_t) - render||^2.
        # Slicing the predictions only optimizes a few views.
        diffusion_loss = F.mse_loss(pred_xstart[:args.n_optimize],
                                    inp[:args.n_optimize])

        # TODO(jainajay): Try other losses. Some possibilities:
        #   ||x_hat(render) - render||^2 (change L480)
        #   ||x_hat(x_t) - x_hat(diffuse(render))||^2
        #         (change denosing code to denoise render and x_t)
        #   ||eps - eps_hat(diffuse(render), eps)||^2
        #   ||eps_hat(x_t) - eps_hat(diffuse(render), eps)||^2
        #       only makes sense if that's the eps in x_t
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

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if args.track_scene_origin:
      raise NotImplementedError

    # Logging.
    with torch.inference_mode():
      volume_model.eval()

      metrics["train/depths/min"] = depths.min()
      metrics["train/depths/max"] = depths.max()
      metrics["train/disparities/min"] = disparities.min()
      metrics["train/disparities/max"] = disparities.max()

      metrics.update({
          "schedule/lr": lr,
          "loss/total_loss": loss.item(),
          "loss/clip": clip_loss.item(),
          "loss/transmittance": transmittance_loss.item(),
          "train/avg_transmittance": avg_transmittance.mean().item()
      })

      # Print the current values of the losses.
      if iteration % 10 == 0:
        pbar.set_description(
            f"Iteration {iteration:05d}:" +
            f" clip_loss = {float(clip_loss.item()):1.2f}" +
            f" diffusion_loss = {float(diffusion_loss.item()):1.5f}" +
            f" avg transmittance = {float(avg_transmittance.mean().item()):1.2f}"
        )

      # Visualize the renders.
      if visualize_images:
        metrics["render/rendered"] = wandb_grid(images)
        metrics["render/silhouettes"] = wandb_grid(silhouettes)
        metrics["render/rendered_depth"] = wandb_grid(depths)

        if args.clip_lam > 0:
          metrics["render/augmented"] = wandb_grid(clip_aug_images)

        if args.diffusion_lam:
          # Show diffusion_x_prev, diffusion_x (sample), out['pred_xstart'].
          for name, val in zip(["x_t", "x_tm1", "pred_xstart"],
                               [diffusion_x_prev, diffusion_x, pred_xstart]):
            print("diffusion", name, val.shape, val.min(), val.max())
            val = unprocess_glide(val)  # [n_views, C, 64, 64]
            metrics[f"diffusion/{name}"] = wandb_grid(val)

      # Validate from a held-out view.
      if iteration % 250 == 0 or iteration == 1:
        validation_view = render_validation_view(
            volume_model,
            scene_origin,
            test_clip_size,
            args.max_validation_size,
            **render_kwargs)
        assert validation_view.ndim == 3
        assert validation_view.shape[-1] == 3
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

      if iteration % 250 == 0 or iteration == 1:
        # Visualize the optimized volume by rendering from multiple viewpoints
        # that rotate around the volume's y-axis.
        video_frames = render_rotating_volume(
            volume_model,
            scene_origin=scene_origin,
            video_size=args.video_size,
            n_frames=args.video_n_frames,
            **render_kwargs)

        for name, frames in zip(["rgb", "depth", "disparity", "silhouette"],
                                video_frames):
          # frames is in THWC order.
          filename = f"/tmp/{iteration:05d}_{name}.mp4"
          if frames.shape[-1] == 1:
            media.write_video(filename, frames[Ellipsis, 0], fps=30)
          else:
            media.write_video(filename, frames, fps=30)
          print("wrote", filename,
                f"range: [{frames.min():.4f}, {frames.max():.4f}]")

          metrics[f"render/video/{name}"] = wandb.Video(
              filename, fps=30, format="mp4")

      wandb.log(metrics, iteration)

      volume_model.train()


if __name__ == "__main__":
  main()
