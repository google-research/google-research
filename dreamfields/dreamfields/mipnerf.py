# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Concise implementation of MIPNeRF primitives.

Based on mip-NeRF: https://github.com/google/mipnerf
"""

from typing import Sequence

from . import helpers
from . import scene

import flax.linen as nn
import jax
import jax.numpy as np
import jax.random as random


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


class MipMLPLate(nn.Module):
  """MLP architecture."""
  activation: str
  features_early: Sequence[int]
  features_residual: Sequence[Sequence[int]]
  features_late: Sequence[int]
  fourfeat: bool
  max_deg: int
  use_cov: bool
  dropout_rate: float

  @nn.compact
  def __call__(self, mean, cov=None, x_late=None, decayscale=1.,
               *, deterministic):
    """Run MLP."""
    # Integrate the positional encoding over a region centered at mean.
    if not self.fourfeat:
      # Axis-aligned positional encoding.
      feat = 2**np.arange(self.max_deg)[:, None, None] * np.eye(3)
      feat = feat.reshape(-1, 3)
    else:
      # Random Fourier Feature positional encoding. Fix the PRNGKey used for the
      # fourier feature basis so the encoding does not change over iterations.
      fourfeat_key = random.PRNGKey(124124)
      dirs = random.normal(fourfeat_key, (3, 128))
      dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
      rads = 2 ** (self.max_deg * random.uniform(fourfeat_key, (128,)))
      feats = (rads * dirs).astype(np.int32)
      feats = np.concatenate([np.eye(3), feats], 1).astype(np.float32)
      feat = feats.T

    mean_proj = (mean[Ellipsis, None] * feat.T).sum(-2)
    if self.use_cov:
      cov_diag_proj = ((cov[Ellipsis, None] * feat.T).sum(-2) * feat.T).sum(-2)
      decay = np.exp(-.5 * cov_diag_proj * decayscale**2)
    else:
      # Disable IPE
      decay = 1.
    x = np.concatenate([decay * np.cos(mean_proj),
                        decay * np.sin(mean_proj)], -1)

    # Network
    activation = nn.__getattribute__(self.activation)
    for feat in self.features_early:
      x = activation(nn.Dense(feat)(x))
      x = nn.Dropout(self.dropout_rate)(
          x, deterministic=deterministic)

    for feat_block in self.features_residual:
      h = nn.LayerNorm()(x)
      for l, feat in enumerate(feat_block):
        h = nn.Dense(feat)(h)
        h = nn.Dropout(self.dropout_rate)(
            h, deterministic=deterministic)
        if l < len(feat_block) - 1:  # don't activate right before the residual
          h = activation(h)
      x = x + h

    if x_late is not None:
      x = np.concatenate([x, x_late], axis=-1)
    for feat in self.features_late[:-1]:
      x = activation(nn.Dense(feat)(x))
      x = nn.Dropout(self.dropout_rate)(
          x, deterministic=deterministic)
    x = nn.Dense(self.features_late[-1])(x)  # don't activate output
    return x


def render_rays_mip(rays, model, variables, rng, config, **kwargs):
  """Volumetric rendering."""
  sc = kwargs.get('sc', 1.)
  sigma_noise_std = kwargs.get('sigma_noise_std', 0.)
  mask_rad = kwargs.get('mask_rad', 1.)
  origin = kwargs.get('origin', np.zeros(3))
  assert 'origin' in kwargs
  train = kwargs.get('train', False)
  rays_shift = [rays[0] + origin, rays[1], rays[2]]
  r_o, r_d = rays_shift[:2]

  near = np.array([config.near])
  far = np.array([config.far])

  if config.get('intersect_box', True) and config.mr_norm is not None:
    # Per shifted ray, only sample within the bounding box of scene
    # the computed near and far are [H, W] arrays
    near, far = intersect_box(rays_shift, mask_rad)
    far = np.maximum(far, near + 1e-3)  # minimum sized interval

  # Get sample points
  sh = list(r_o.shape[:-1])
  t = np.linspace(near, far, config.num_samples + 1).T  # [batch, n_samples+1]
  if config.jitter:
    delta = (far - near) / config.num_samples  # [batch]
    jitter_key, rng = random.split(rng)
    jitter_sh = sh + [t.shape[-1]]
    t += random.uniform(
        jitter_key, jitter_sh, minval=-.5, maxval=.5) * delta[Ellipsis, None]
  endpts = r_o[Ellipsis, None, :] + r_d[Ellipsis, None, :] * t[Ellipsis, None]
  t_mids = .5 * (t[Ellipsis, 1:] + t[Ellipsis, :-1])
  mean, cov = dists_to_samples(rays_shift, t)

  # Run model
  if config.viewdirs:
    dirs = helpers.normalize(r_d)
    dirs = np.broadcast_to(dirs[Ellipsis, None, :], mean.shape)
    dirs_enc = scene.posenc(dirs, config.posenc_dirs_deg)
    raw_outputs = model.apply(variables, mean, cov, dirs_enc, decayscale=sc,
                              deterministic=not train)
  else:
    raw_outputs = model.apply(variables, mean, cov, x_late=None, decayscale=sc,
                              deterministic=not train)

  # Activations to get rgb, sigma
  rgb = jax.nn.sigmoid(raw_outputs[Ellipsis, :3])
  if config.mipnerf.sigma_activation == 'exp':
    sigma_activation = np.exp
  else:
    sigma_activation = jax.nn.__getattribute__(config.mipnerf.sigma_activation)
  if rng is None:  # don't add noise at test time
    sigma = sigma_activation(raw_outputs[Ellipsis, 3])
  else:
    sigma_key, rng = random.split(rng)
    sigma_noise = sigma_noise_std * random.normal(
        sigma_key, raw_outputs.shape[:-1], dtype=raw_outputs.dtype)
    sigma = sigma_activation(raw_outputs[Ellipsis, 3] + sigma_noise)

  sigma = scene.mask_sigma(sigma, mean, mask_rad, config)

  # Estimate updated center of the scene as density-weighted average ray points
  origin, total_sigma = scene.pts_center(mean, sigma)

  aux = {
      'losses': {},
      'scene_origin': origin,
      'scene_origin_sigma': total_sigma,
  }

  # Volume rendering
  delta = np.linalg.norm(endpts[Ellipsis, 1:, :] - endpts[Ellipsis, :-1, :], axis=-1)
  sigma_delta = sigma * delta
  sigma_delta_shifted = np.concatenate([np.zeros_like(sigma_delta[Ellipsis, :1]),
                                        sigma_delta[Ellipsis, :-1]], axis=-1)
  alpha = 1. - np.exp(-sigma_delta)
  trans = np.exp(-np.cumsum(sigma_delta_shifted, axis=-1))
  weights = alpha * trans

  rgb = np.sum(weights[Ellipsis, None] * rgb, axis=-2)
  depth = np.sum(weights * t_mids, axis=-1)
  acc = 1 - np.exp(-np.sum(sigma_delta, axis=-1))

  if config.white_bkgd:
    rgb += 1 - acc[Ellipsis, None]

  return (rgb, depth, acc), aux
