# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# pylint: disable=invalid-name
"""utils to generate panoramic sky."""
import math

from external.gsn.models.nerf_utils import get_sample_points
import numpy as np
import torch

# --------------------------------------------------------------------------


def make_grid(G_pano):
  """construct extended coord grid for panorama."""
  input_layer = G_pano.synthesis.input

  # make secondary grid
  theta = torch.eye(2, 3, device='cuda')
  theta[0, 0] = 0.5 * input_layer.size[0] / input_layer.sampling_rate
  theta[1, 1] = 0.5 * input_layer.size[1] / input_layer.sampling_rate
  grid_width = (
      input_layer.size[0] - 2 * input_layer.margin_size
  ) * 360 // input_layer.fov + 2 * input_layer.margin_size
  grids = torch.nn.functional.affine_grid(
      theta.unsqueeze(0),
      [1, 1, input_layer.size[1], grid_width],
      align_corners=False,
  )
  # ensure that the x coordinate completes a full circle without padding
  base_width = grid_width - 2 * input_layer.margin_size
  new_x = (
      torch.arange(
          -input_layer.margin_size,
          base_width * 2 + input_layer.margin_size,
          device=grids.device,
      )
      / base_width
      * 2
      - 1
  )
  new_y = grids[0, :, 0, 1]
  new_grids = torch.cat(
      [
          new_x.view(1, 1, -1, 1).repeat(1, input_layer.size[1], 1, 1),
          new_y.view(1, -1, 1, 1).repeat(1, 1, grid_width + base_width * 1, 1),
      ],
      dim=3,
  )
  return new_grids


def generate_start_grid(seed, input_layer, grid):
  rng = np.random.RandomState(seed)
  start_shift = rng.randint(360 / input_layer.fov * input_layer.frame_size[0])
  start_grid = grid[:, :, start_shift : start_shift + input_layer.size[0], :]
  return start_grid


def generate_pano_transform(G_pano, z, ws_encoder, grid):
  """Generate pano image from latent z and encoded image."""
  input_layer = G_pano.synthesis.input
  num_frames = int(360 / input_layer.fov)

  images = []
  with torch.no_grad():
    for tx in range(num_frames):
      transform = torch.eye(3)[None].to(z.device)
      transform[0, 0, 2] = tx
      crop_fn = (
          lambda x: grid
      )  # fix the coord grid, and shift using transform matr

      ws_mapping = G_pano.mapping(z, c=None)
      num_layers = ws_mapping.shape[1]
      features = ws_encoder[:, None].repeat(1, num_layers, 1)
      ws_in = torch.cat([ws_mapping, features], dim=2)
      out = G_pano.synthesis(
          ws_in, mapped_scale=None, transform=transform, crop_fn=crop_fn
      )  # , truncation_psi=0.5)

      images.append(out[0].cpu())
  pano = torch.cat(images, dim=2)
  return pano


# --------------------------------------------------------------------------
# utils to sample sky as a function of Rt or view directions


def composite_sky(img, mask, sky):
  return img * mask + (1 - mask) * sky


def sample_sky_from_viewdirs(sky_texture, viewdirs, sample_res, fov):
  """get corresponding part of sky_texture based on viewdir."""
  v = viewdirs.view(1, sample_res, sample_res, -1)
  x, y, z = v.unbind(3)
  theta = (torch.arctan2(z, x) * 180 / math.pi) % 360
  diag = torch.sqrt(x**2 + z**2)
  psi = torch.arctan(y / diag) * 180 / math.pi
  theta_norm = (theta - 180) / 180
  psi_norm = psi / (fov / 2) * -1
  patch = torch.nn.functional.grid_sample(
      sky_texture,
      torch.stack([theta_norm, psi_norm], dim=-1),
      padding_mode='reflection',
  )
  return patch


def sample_sky_from_Rt(sky_texture, Rt, sample_res, fov):
  """get corresponding part of sky_texture based on Rt."""
  # sky_texture = [B, 3, H, W]
  # Rt = [B, 4, 4]
  fx = (sample_res / 2) / np.tan(np.deg2rad(fov) / 2)
  fy = -(sample_res / 2) / np.tan(np.deg2rad(fov) / 2)
  _, viewdirs, _, _, _ = get_sample_points(
      tform_cam2world=Rt.inverse().cpu(),
      F=(fx, fy),
      H=sample_res,
      W=sample_res,
      samples_per_ray=2,
      near=1,
      far=8,
      perturb=False,
      mask=None,
  )
  v = viewdirs[:, :, 0].view(1, sample_res, sample_res, -1)
  x, y, z = v.unbind(3)
  theta = (torch.arctan2(z, x) * 180 / math.pi) % 360
  diag = torch.sqrt(x**2 + z**2)
  psi = torch.arctan(y / diag) * 180 / math.pi
  theta_norm = (theta - 180) / 180
  psi_norm = psi / (fov / 2) * -1
  patch = torch.nn.functional.grid_sample(
      sky_texture,
      torch.stack([theta_norm, psi_norm], dim=-1),
      padding_mode='reflection',
  )
  return patch
