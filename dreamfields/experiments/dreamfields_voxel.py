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

"""Fit a textured volume with CLIP guidance.

This fits a volume given a caption using differentiable volumetric rendering.
Based on pytorch3d/main/docs/tutorials/fit_textured_volume.ipynb.

More specifically, this will:
1. Create a differentiable volumetric renderer.
2. Create a Volumetric model (including how to use the `Volumes` class).
3. Fit the volume based on text using the differentiable volumetric renderer.
4. Visualize the predicted volume.

Based on pytorch3d/main/docs/tutorials/fit_textured_volume.ipynb.
"""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import argparse
import functools
import random

import clip
import jax
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from tqdm.auto import trange
import wandb

# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
# Data structures and functions for rendering.
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
                                VolumeRenderer, NDCGridRaysampler,
                                EmissionAbsorptionRaymarcher)
from pytorch3d.structures import Volumes

from augment import checkerboard
from coco_val_queries import queries
import optvis
# pylint: enable=g-bad-import-order
# pylint: enable=g-multiple-import

parser = argparse.ArgumentParser()
# Logging.
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--exp-name-prefix", type=str, default="")
# Query and seed.
parser.add_argument("--query-idx", type=int)
parser.add_argument("--seed", type=int, default=0)
# Scene parameterization.
parser.add_argument("--optimize-pixels", action="store_true")
parser.add_argument("--volume-size", type=int, default=128)
parser.add_argument(
    "--sample-mode",
    type=str,
    default="bilinear",
    choices=["bilinear", "nearest", "bicubic"],
    help="RGB and density interpolation mode for"
    "torch.nn.functional.grid_sample")
# Data augmentation.
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
parser.add_argument("--density-tvd-lam", type=float, default=0.1)
parser.add_argument("--grouped-tvd", action="store_true")
parser.add_argument("--color-tvd-lam", type=float, default=0.1)
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

DEFAULT_VOLUME_SIZE = (64, 64, 64)


class VolumeModel(torch.nn.Module):

  def __init__(self, renderer, volume_size=DEFAULT_VOLUME_SIZE, voxel_size=0.1):
    super().__init__()
    # After evaluating torch.sigmoid(self.log_colors), we get
    # densities close to zero.
    self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
    # After evaluating torch.sigmoid(self.log_colors), we get
    # a neutral gray color everywhere.
    self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
    self._voxel_size = voxel_size
    # Store the renderer module as well.
    self._renderer = renderer

  def forward(self, cameras, custom_renderer=None):
    batch_size = cameras.R.shape[0]

    # Convert the log-space values to the densities/colors.
    densities = torch.sigmoid(self.log_densities)
    colors = torch.sigmoid(self.log_colors)

    # Instantiate the Volumes object, making sure
    # the densities and colors are correctly
    # expanded batch_size-times.
    volumes = Volumes(
        densities=densities[None].expand(batch_size, *self.log_densities.shape),
        features=colors[None].expand(batch_size, *self.log_colors.shape),
        voxel_size=self._voxel_size,
    )

    # Given cameras and volumes, run the renderer
    # and return only the first output value
    # (the 2nd output is a representation of the sampled
    # rays which can be omitted for our purpose).
    if custom_renderer is not None:
      return custom_renderer(cameras=cameras, volumes=volumes)[0]

    rendered = self._renderer(cameras=cameras, volumes=volumes)[0]
    rgb, silhouette = rendered.split([3, 1], dim=-1)
    return rgb, silhouette, colors, densities


def get_renderer(resolution, n_pts_per_ray):
  # Changing the rendering resolution is a bit involved.
  raysampler = NDCGridRaysampler(
      image_width=resolution,
      image_height=resolution,
      n_pts_per_ray=n_pts_per_ray,
      min_depth=args.camera_radius - args.volume_extent_world * np.sqrt(3) / 2,
      max_depth=args.camera_radius + args.volume_extent_world * np.sqrt(3) / 2,
  )
  raymarcher = EmissionAbsorptionRaymarcher()
  renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)
  return renderer


def render_rotating_volume(volume_model,
                           device,
                           n_frames=50,
                           video_size=400,
                           n_pts_per_ray=192):
  renderer = get_renderer(video_size, n_pts_per_ray)

  # Render frames.
  with torch.inference_mode():
    print("Generating rotating volume ...")
    elev = 30
    azimuths = torch.linspace(0., 360., n_frames, device=device)
    frames = []

    for azim in tqdm(azimuths):
      R, T = look_at_view_transform(
          dist=args.camera_radius, elev=elev, azim=azim)
      batch_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
      rgbo = volume_model(batch_cameras, renderer)
      rgb = rgbo[Ellipsis, :3]
      opacity = rgbo[Ellipsis, 3:4]
      frame = opacity * rgb + 1 - opacity
      frame = frame.clamp(0.0, 1.0)
      frames.append(frame)

    frames = torch.cat(frames).clamp(0., 1.)
    frames = frames.movedim(-1, 1)  # THWC to TCHW.
    return frames.cpu().numpy()


def load_clip(name, device):
  model, _ = clip.load(name, device=device, jit=False)

  def preprocess(images):
    images = F.interpolate(
        images, size=model.visual.input_resolution, mode="bicubic")
    images = (images - image_mean) / image_std
    return images

  return model, preprocess, model.visual.input_resolution


def render_validation_view(volume_model, render_size, device):
  with torch.inference_mode():
    test_renderer = get_renderer(render_size, n_pts_per_ray=192)

    R, T = look_at_view_transform(dist=4.0, elev=45, azim=30)
    camera = FoVPerspectiveCameras(device=device, R=R, T=T)
    rgbo = volume_model(camera, test_renderer)
    rgb = rgbo[Ellipsis, :3]
    opacity = rgbo[Ellipsis, 3:4]
    rendering = opacity * rgb + (1 - opacity)
    rendering = rendering.clamp(0.0, 1.0)

    return rendering.squeeze(0)


def compute_query_rank(model, preprocess, rendering, query, queries_r, device):
  if query not in queries_r:
    print(f"WARN: query \"{query}\" not in retrieval set. Adding it.")
    queries_r = queries_r + [query]
    query_idx = len(queries_r) - 1
  else:
    query_idx = queries_r.index(query)

  with torch.inference_mode():
    # Embed the retrieval set of captions.
    queries_tok = clip.tokenize(queries).to(device)
    z_queries = model.encode_text(queries_tok).detach()
    z_queries = F.normalize(z_queries, dim=-1)

    # Embed render.
    assert rendering.ndim == 4
    assert rendering.shape[1] == 3
    x = preprocess(rendering)
    z_rendering = model.encode_image(x)
    z_rendering = F.normalize(z_rendering, dim=-1)

    sim = torch.sum(z_rendering * z_queries, dim=-1)
    ranks = torch.argsort(sim, dim=0, descending=True)
    return torch.nonzero(ranks == query_idx)[0].item(), sim[query_idx].item()


def clamp_and_detach(x):
  return x.clamp(0.0, 1.0).cpu().detach().numpy()


def sample_backgrounds(num, res, device):
  rand = np.random.uniform()
  if rand <= 0.3333:
    # Randomly colored checkerboard.
    bg = checkerboard(num, args.nsq, res, device=device)
  elif rand <= 0.6666:
    # Random noise.
    bg = torch.rand((num, 3, res, res), device=device)
  else:
    # Random FFT backgrounds from optvis.py.
    key = jax.random.PRNGKey(np.random.randint(1000000))
    keys = jax.random.split(key, num)

    fn = functools.partial(
        optvis.image_sample, shape=[1, res, res, 3], sd=0.2, decay_power=1.5)

    bg = jax.vmap(fn)(keys)[:, 0]  # NHWC
    bg = torch.from_numpy(np.asarray(bg)).to(device)
    bg = bg.movedim(-1, 1)  # NHWC to NCHW

  # Blur the background.
  min_blur, max_blur = args.bg_blur_std_range
  blur_std = np.random.uniform() * (max_blur - min_blur) + min_blur
  bg = torchvision.transforms.functional.gaussian_blur(
      bg, kernel_size=15, sigma=blur_std)

  return bg


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
  else:
    device = torch.device("cpu")

  ## 3. Initialize CLIP.
  image_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device)
  image_mean = image_mean[None, :, None, None]
  image_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
  image_std = image_std[None, :, None, None]

  # NOTE: wasteful of memory to keep both loaded.
  model, preprocess, clip_size = load_clip(args.loss_model, device)
  test_model, test_preprocess, test_clip_size = load_clip(
      args.retrieve_model, device)

  ## 4. Initialize the volumetric renderer.
  # 0.1 units of any camera plane.
  raysampler = NDCGridRaysampler(
      image_width=args.render_size,
      image_height=args.render_size,
      n_pts_per_ray=args.n_pts_per_ray,
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
  renderer = VolumeRenderer(
      raysampler=raysampler,
      raymarcher=raymarcher,
      sample_mode=args.sample_mode)

  ## 5. Initialize the volumetric model.
  if args.optimize_pixels:
    volume_model = torch.rand((args.batch_size, 128, 128, 4),
                              device=device,
                              requires_grad=True)
  else:
    # Instantiate the volumetric model.
    # We use a cubical volume with the size of
    # one side = 128. The size of each voxel of the volume
    # is set to args.volume_extent_world / volume_size s.t. the
    # volume represents the space enclosed in a 3D bounding box
    # centered at (0, 0, 0) with the size of each side equal to 3.
    volume_model = VolumeModel(
        renderer,
        volume_size=[args.volume_size] * 3,
        voxel_size=args.volume_extent_world / args.volume_size,
    ).to(device)

  def get_parameters():
    if args.optimize_pixels:
      return [volume_model]

    return volume_model.parameters()

  # Instantiate the Adam optimizer.
  optimizer = torch.optim.Adam(get_parameters(), lr=args.lr)

  # Embed the target caption
  query_tok = clip.tokenize(query).to(device)
  z_clip = model.encode_text(query_tok).detach()
  z_clip = F.normalize(z_clip, dim=-1)

  # We do 300 Adam iterations and sample 10 random images in each minibatch.
  aug_fn = torchvision.transforms.RandomResizedCrop(
      clip_size, scale=args.crop_scale_range, ratio=(1.0, 1.0))

  pbar = trange(1, args.n_iter + 1)
  for iteration in pbar:
    metrics = {}

    # In case we reached the last 75% of iterations,
    # decrease the learning rate of the optimizer 10-fold.
    if iteration == round(args.n_iter * 0.75):
      tqdm.write("Decreasing LR 10-fold ...")
      for g in optimizer.param_groups:
        g["lr"] = args.lr * 0.1

    # Zero the optimizer gradient.
    optimizer.zero_grad()

    # Get a batch of viewing angles.
    elev = 30  # keep constant
    azim = torch.rand(args.batch_size) * 360

    # Initialize an OpenGL perspective camera that represents a batch of
    # different viewing angles. All the cameras helper methods support mixed
    # type inputs and broadcasting. So we can view the camera from the a
    # distance of dist=2.7, and then specify elevation and azimuth angles for
    # each viewpoint as tensors.
    R, T = look_at_view_transform(
        dist=args.camera_radius / 1.2, elev=elev, azim=azim)
    batch_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    if args.optimize_pixels:
      # Evaluate the volumetric model.
      rendered_images = volume_model[Ellipsis, :3]
      rendered_silhouettes = volume_model[Ellipsis, 3:4]
    else:
      rendered_images, rendered_silhouettes, colors, densities = volume_model(
          batch_cameras)

    # Transmittance loss.
    tau = torch.tensor((args.target_transmittance), device=device)
    avg_transmittance = 1 - rendered_silhouettes.mean()
    transmittance_loss = -torch.min(tau, avg_transmittance)

    # Compute the CLIP loss.
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

    # Augment.
    aug_images = aug_fn(composite_images)

    # Embed.
    x = preprocess(aug_images)  # Resize and normalize.
    z_est = model.encode_image(x)
    z_est = F.normalize(z_est, dim=-1)
    clip_loss = -torch.sum(z_est * z_clip, dim=-1).mean()

    # Compute total loss and take an optimization step.
    loss = clip_loss + args.transmittance_lam * transmittance_loss

    if args.density_tvd_lam and not args.optimize_pixels and args.grouped_tvd:
      eps = 1e-8

      d = densities.squeeze(0)
      dc = torch.stack([d, d.transpose(1, 2), d.transpose(0, 2)], dim=0)
      dc = F.pad(dc, (1, 1))
      density_diff = (dc[Ellipsis, :-1] - dc[Ellipsis, 1:]) * volume_model._voxel_size
      density_tvd_loss = torch.square(density_diff).sum(dim=0)
      density_tvd_loss = torch.sqrt(density_tvd_loss + eps).mean()
      if torch.all(torch.isfinite(density_tvd_loss)):
        loss = loss + args.density_tvd_lam * density_tvd_loss
      else:
        print("WARN: density_tvd_loss is not all finite")
      metrics["loss/grouped_density_tvd"] = density_tvd_loss.item()

      c = colors  # [C, R, R, R]
      cc = torch.stack(
          [c, c.transpose(2, 3), c.transpose(1, 3)], dim=0)  # [3, C, R, R, R]
      color_diff = (cc[Ellipsis, :-1] -
                    cc[Ellipsis, 1:]) * volume_model._voxel_size  # [3, C, R, R, R-1]
      color_tvd_loss = torch.square(color_diff).sum(dim=0)  # [C, R, R, R-1]
      color_tvd_loss = torch.sqrt(color_tvd_loss + eps).mean()
      if torch.all(torch.isfinite(color_tvd_loss)):
        loss = loss + args.color_tvd_lam * color_tvd_loss
      else:
        print("WARN: color_tvd_loss is not all finite")
      metrics["loss/grouped_rgb_tvd"] = color_tvd_loss.item()

      loss.backward()
    elif args.density_tvd_lam and not args.optimize_pixels:
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

      # Manually compute the gradient of a TV loss on the density.
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
    metrics.update({
        "loss/total_loss": loss.item(),
        "loss/clip": clip_loss.item(),
        "loss/transmittance": transmittance_loss.item(),
    })

    # Print the current values of the losses.
    if iteration % 10 == 0:
      pbar.set_description(
          f"Iteration {iteration:05d}:" +
          f" clip_loss = {float(clip_loss.item()):1.2f}" +
          f" avg transmittance = {float(avg_transmittance.item()):1.2f}")

    # Visualize the renders every 100 iterations.
    if iteration % 100 == 0 or iteration == 1:
      # Visualize only a single randomly selected element of the batch.
      im_show_idx = int(torch.randint(low=0, high=args.batch_size, size=(1,)))

      rendering = rendered_images[im_show_idx]
      silhouette = rendered_silhouettes[im_show_idx]
      metrics["render/rendered"] = wandb.Image(clamp_and_detach(rendering))
      aug_image = aug_images[im_show_idx].movedim(0, 2)
      metrics["render/augmented"] = wandb.Image(clamp_and_detach(aug_image))
      render_white_bg = rendering * silhouette + 1 - silhouette
      metrics["render/rendered_white_bg"] = wandb.Image(
          clamp_and_detach(render_white_bg))
      metrics["render/rendered_silhouettes"] = wandb.Image(
          clamp_and_detach(silhouette.squeeze(-1)))

    # Validate from a held-out view.
    if iteration % 250 == 0 or iteration == 1:
      validation_view = render_validation_view(
          volume_model, test_clip_size, device=device)
      metrics["val/render"] = wandb.Image(clamp_and_detach(validation_view))

      rank, cosine_sim = compute_query_rank(
          test_model,
          test_preprocess,
          rendering=validation_view.movedim(-1, 0).unsqueeze(0),
          query=query,
          queries_r=queries,
          device=device)

      metrics["val/rank"] = rank
      metrics["val/acc"] = int(rank == 0)
      metrics["val/cosine_sim"] = cosine_sim

    if iteration % 500 == 0 or iteration == 1:
      # We visualize the optimized volume by rendering from multiple viewpoints
      # that rotate around the volume's y-axis.
      rotating_volume_frames = render_rotating_volume(
          volume_model, n_frames=240, device=device)
      metrics["render/video"] = wandb.Video(
          (rotating_volume_frames * 255.).astype(np.uint8),
          fps=30,
          format="mp4")

    wandb.log(metrics, iteration)
