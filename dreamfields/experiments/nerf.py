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

"""Neural rendering utilities and MLP architecture."""

# pylint: disable=g-bad-import-order
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scene
# pylint: enable=g-bad-import-order


class DreamFieldsMLP(nn.Module):
  """MLP architecture."""

  def __init__(self, activation, features_early,
               features_residual,
               features_late, fourfeat, max_deg,
               ipe, num_fourier_features = 128):
    super().__init__()

    self.ipe = ipe
    if fourfeat:
      # Random Fourier Feature positional encoding. Fix the matrix used for the
      # fourier feature basis so the encoding does not change over iterations.
      dirs = torch.randn((3, num_fourier_features))
      dirs = F.normalize(dirs, dim=-1)
      rads = 2**(max_deg * torch.rand((num_fourier_features,)))
      feats = (rads * dirs).long()
      # Include axis aligned features with scale 1.
      feature_matrix = torch.cat([torch.eye(3), feats], 1).long().T
    else:
      # Axis-aligned positional encoding.
      feature_matrix = 2**torch.arange(max_deg)[:, None, None] * torch.eye(3)
      feature_matrix = feature_matrix.reshape(-1, 3)
    self.register_buffer('feature_matrix', feature_matrix)

    dim = 2 * feature_matrix.size(0)
    activation = nn.__getattribute__(activation)

    # Early network.
    layers_early = []
    for feat in features_early:
      layers_early.append(nn.Linear(dim, feat))
      layers_early.append(activation())
      dim = feat
    self.layers_early = nn.Sequential(*layers_early)

    # Residual blocks.
    layers_residual = []
    for feat_block in features_residual:
      layers_residual.append(nn.LayerNorm(dim))

      # Create a stack of layers.
      block = []
      for l, feat in enumerate(feat_block):
        block.append(nn.Linear(dim, feat))
        if l < len(feat_block) - 1:  # Don't activate right before the residual.
          block.append(activation())
        dim = feat
      block = nn.Sequential(*block)

      layers_residual.append(block)
    self.layers_residual = nn.ModuleList(layers_residual)

    # Late layers.
    layers_late = []
    for l, feat in enumerate(features_late):
      layers_late.append(nn.Linear(dim, feat))
      if l < len(features_late) - 1:  # Don't activate output.
        layers_late.append(activation())
      dim = feat
    self.layers_late = nn.Sequential(*layers_late)

  def reset_parameters(self):
    """Match the default flax initialization."""
    for layer in self.children():
      if isinstance(layer, nn.Linear):
        torch.nn.init.lecun_normal_(layer.weight)
        torch.nn.init.zero_(layer.bias)

  def forward(self, mean, cov=None, decayscale=1.):
    """Run MLP. mean is [*batch, 3] and cov is [*batch, 3, 3]."""
    fm = self.feature_matrix.type(mean.dtype).T  # [3, dim]
    mean_proj = mean.matmul(fm)  # [*batch, dim]
    if self.ipe:
      # Integrated positional encoding (IPE).
      cov_diag_proj = (cov.matmul(fm) * fm).sum(dim=-2)  # [*batch, dim]
      decay = torch.exp(-.5 * cov_diag_proj * decayscale**2)
    else:
      # Disable IPE.
      decay = 1.
    x = torch.cat([decay * torch.cos(mean_proj), decay * torch.sin(mean_proj)],
                  -1)
    x = x.float()

    # Network
    x = self.layers_early(x)

    for i in range(len(self.layers_residual) // 2):
      norm, block = self.layers_residual[(2 * i):(2 * i + 2)]
      x = x + block(norm(x))

    x = self.layers_late(x)
    return x


def intersect_box(rays, box_width):
  # r_o + t * r_d = +/- box_width
  r_o, r_d = rays[:2]
  t0 = np.max((-np.sign(r_d) * box_width - r_o) / r_d, axis=-1)
  t1 = np.min((np.sign(r_d) * box_width - r_o) / r_d, axis=-1)
  return t0, t1


def dists_to_samples(rays, t):
  """Convert mipnerf frustums to gaussians."""
  t_mids = .5 * (t[Ellipsis, 1:] + t[Ellipsis, :-1])
  mean = rays[0][Ellipsis, None, :] + rays[1][Ellipsis, None, :] * t_mids[Ellipsis, None]

  d = rays[1]
  d_mag_sq = np.maximum(1e-10, np.sum(d**2, axis=-1, keepdims=True))
  t_half = .5 * (t[Ellipsis, 1:] - t[Ellipsis, :-1])
  t_var = t_half**2 / 3.
  r_var = (rays[2] * t_mids)**2 / 12.

  d_outer = d[Ellipsis, :, None] * d[Ellipsis, None, :]
  eye = np.eye(d.shape[-1])
  null_outer = eye - d[Ellipsis, :, None] * (d / d_mag_sq)[Ellipsis, None, :]
  t_cov = t_var[Ellipsis, None, None] * d_outer[Ellipsis, None, :, :]
  xy_cov = r_var[Ellipsis, None, None] * null_outer[Ellipsis, None, :, :]
  cov = t_cov + xy_cov

  return mean, cov


def render_rays_mip(rays,
                    model,
                    *,
                    near,
                    far,
                    device,
                    white_bkgd=True,
                    mask_rad=1.,
                    mask_rad_norm='inf',
                    sigma_noise_std=0.,
                    jitter=True,
                    decayscale=1.,
                    n_pts_per_ray=192,
                    origin=None,
                    train=False,
                    eps=1e-6,
                    chunksize=None):
  """Volumetric rendering.

  Args:
    rays: tuple of (ray_origins, ray_directions, ray_diffs). Each is a
      torch.tensor. ray_origins is (..., 3), ray_directions is (..., 3),
      and ray_diffs is (..., 1).
    model (torch.nn.Module): NeRF MLP model.
    near (float): Distance from camera origin to near plane.
    far (float): Distance from camera origin to far plane.
    device: Torch device, e.g. CUDA or CPU. white_bkgd mask_rad mask_rad_norm
      sigma_noise_std jitter.
    white_bkgd (bool):
    mask_rad (float):
    mask_rad_norm (str):
    sigma_noise_std:
    jitter:
    decayscale (float): coarse-to-fine positional encoding. When this is greater
      than 1, high frequency positional encodings are blurred.
    n_pts_per_ray:
    origin: 3-dimensional origin of the scene.
    train (bool): used to decide whether to add noise to density chunksize.
      (Optional[int])
    eps:
    chunksize:

  Returns:
    (rgb, depth, silhouette), aux where:
      rgb is a (*batch, H, W, 3) image, in range [0, 1].
      depth is a (*batch, H, W, 1) image, in range [0, 1].
      disparity is a (*batch, H, W, 1) image.
      silhouette is a (*batch, H, W, 1) grayscale foreground mask image, in
      range [0, 1]. Values closer to 1 indicate foreground. This is sometimes
      called "acc" in NeRF implementations.

  TODO(jainajay):
    - Implement coarse to fine sampling, with the same network
    - Render texture-free images, with lighting given by autodiff normals
  """
  if origin is None:
    origin = np.zeros(3)
  rays_shift = [rays[0] + origin, rays[1], rays[2]]
  r_o, r_d = rays_shift[:2]

  near_ = np.array([near])
  far_ = np.array([far])

  if mask_rad_norm is not None:
    # Per shifted ray, only sample within the bounding box of scene
    # the computed near and far are [H, W] arrays.
    near_, far_ = intersect_box(rays_shift, mask_rad)
    far_ = np.maximum(far_, near_ + 1e-3)  # minimum sized interval

  # Get sample points.
  sh = list(r_o.shape[:-1])
  t = np.linspace(
      near_, far_, n_pts_per_ray + 1, axis=-1)  # [*batch, n_samples+1]
  if jitter:
    delta = (far_ - near_) / n_pts_per_ray  # [*batch]
    jitter_sh = sh + [t.shape[-1]]
    t += (np.random.uniform(size=jitter_sh) - 0.5) * delta[Ellipsis, None]
  endpts = r_o[Ellipsis, None, :] + r_d[Ellipsis, None, :] * t[Ellipsis, None]
  t_mids = .5 * (t[Ellipsis, 1:] + t[Ellipsis, :-1])
  mean, cov = dists_to_samples(rays_shift, t)

  # Run model.
  mean_torch = torch.from_numpy(mean).to(device, non_blocking=True)
  cov_torch = torch.from_numpy(cov).to(device, non_blocking=True)
  if chunksize:
    raw_outputs = []
    for i in range(0, mean_torch.shape[0], chunksize):
      batch_outputs = model(
          mean=mean_torch[i:i + chunksize],
          cov=cov_torch[i:i + chunksize],
          decayscale=decayscale)
      raw_outputs.append(batch_outputs)
    raw_outputs = torch.cat(raw_outputs, dim=0)
  else:
    raw_outputs = model(mean=mean_torch, cov=cov_torch, decayscale=decayscale)

  # Activations to get rgb, sigma.
  rgb = F.sigmoid(raw_outputs[Ellipsis, :3])
  if train and sigma_noise_std:  # Don't add noise at test time.
    sigma_noise = sigma_noise_std * torch.randn(
        raw_outputs.shape[:-1], dtype=raw_outputs.dtype)
    sigma = F.softplus(raw_outputs[Ellipsis, 3] + sigma_noise)
  else:
    sigma = F.softplus(raw_outputs[Ellipsis, 3])

  sigma = scene.mask_sigma(sigma, mean_torch, mask_rad, mask_rad_norm)

  # Estimate updated center of the scene as density-weighted average ray points.
  with torch.no_grad():
    origin, total_sigma = scene.pts_center(mean_torch, sigma)

  aux = {
      'losses': {},
      'scene_origin': origin.cpu().numpy(),
      'scene_origin_sigma': total_sigma.cpu().numpy(),
  }

  # Volume rendering.
  delta = np.linalg.norm(endpts[Ellipsis, 1:, :] - endpts[Ellipsis, :-1, :], axis=-1)
  sigma_delta = sigma * torch.from_numpy(delta).float().to(device)
  sigma_delta_shifted = torch.cat(
      [torch.zeros_like(sigma_delta[Ellipsis, :1]), sigma_delta[Ellipsis, :-1]], axis=-1)
  alpha = 1. - torch.exp(-sigma_delta)
  trans = torch.exp(-torch.cumsum(sigma_delta_shifted, dim=-1))
  weights = alpha * trans

  rgb = torch.sum(weights[Ellipsis, None] * rgb, dim=-2)  # [*batch, H, W, 3]

  t_mids_torch = torch.from_numpy(t_mids).to(device)
  depth = torch.sum(
      weights.double() * t_mids_torch, dim=-1,
      keepdim=True)  # [*batch, H, W, 1]
  depth = depth.float()

  # Scale disparity.
  disp_min, disp_max = 1. / (far + eps), 1. / (near + eps)
  disparity = 1. / (depth + eps)  # [*batch, H, W, 1]
  disparity = (disparity - disp_min) / (disp_max - disp_min)

  # Scale depth between [0, 1]. Depth is originally [0, far].
  # Depth should really be [near, far], but isn't actually, perhaps due
  # to rays that don't intersect the clipping box.
  depth = depth / far

  silhouette = 1 - torch.exp(-torch.sum(sigma_delta, dim=-1))
  silhouette = silhouette[Ellipsis, None]  # [*batch, H, W, 1]

  if white_bkgd:
    rgb += 1 - silhouette

  return (rgb, depth, disparity, silhouette), aux
