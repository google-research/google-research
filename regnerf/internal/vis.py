# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Helper functions for visualizing things."""
import dm_pix as pix
from internal import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.cm as cm


def sinebow(h):
  """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
  f = lambda x: jnp.sin(jnp.pi * x)**2
  return jnp.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def convolve2d(z, f):
  return jsp.signal.convolve2d(
      z, f, mode='same', precision=jax.lax.Precision.HIGHEST)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
  """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
  bg_mask = jnp.logical_xor(
      (jnp.arange(acc.shape[0]) % (2 * width) // width)[:, None],
      (jnp.arange(acc.shape[1]) % (2 * width) // width)[None, :])
  bg = jnp.where(bg_mask, light, dark)
  return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def depth_to_normals(depth):
  """Assuming `depth` is orthographic, linearize it to a set of normals."""
  f_blur = jnp.array([1, 2, 1]) / 4
  f_edge = jnp.array([-1, 0, 1]) / 2
  dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
  dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
  inv_denom = 1 / jnp.sqrt(1 + dx**2 + dy**2)
  normals = jnp.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
  return normals


def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
  """Visualize a 1D image and a 1D weighting according to some colormap.

  Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

  Returns:
    A colormap rendering.
  """
  # Identify the values that bound the middle of `value' according to `weight`.
  lo_auto, hi_auto = math.weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

  # If `lo` or `hi` are None, use the automatically-computed bounds above.
  eps = jnp.finfo(jnp.float32).eps
  lo = lo or (lo_auto - eps)
  hi = hi or (hi_auto + eps)

  # Curve all values.
  value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

  # Wrap the values around if requested.
  if modulus:
    value = jnp.mod(value, modulus) / modulus
  else:
    # Otherwise, just scale to [0, 1].
    value = jnp.nan_to_num(
        jnp.clip((value - jnp.minimum(lo, hi)) / jnp.abs(hi - lo), 0, 1))

  if colormap:
    colorized = colormap(value)[:, :, :3]
  else:
    assert len(value.shape) == 3 and value.shape[-1] == 3
    colorized = value

  return matte(colorized, weight) if matte_background else colorized


def visualize_normals(depth, acc, scaling=None):
  """Visualize fake normals of `depth` (optionally scaled to be isotropic)."""
  if scaling is None:
    mask = ~jnp.isnan(depth)
    x, y = jnp.meshgrid(
        jnp.arange(depth.shape[1]), jnp.arange(depth.shape[0]), indexing='xy')
    xy_var = (jnp.var(x[mask]) + jnp.var(y[mask])) / 2
    z_var = jnp.var(depth[mask])
    scaling = jnp.sqrt(xy_var / z_var)

  scaled_depth = scaling * depth
  normals = depth_to_normals(scaled_depth)
  return matte(jnp.isnan(normals) + jnp.nan_to_num((normals + 1) / 2, 0), acc)


def visualize_coord_mod(coords, acc):
  """Visualize the coordinate of each point within its "cell"."""
  return matte(((coords + 1) % 2) / 2, acc)


def visualize_coord_fix(coords, acc, percentile=99.):
  """Visualize the "cell" each coordinate lives in, and highlight its edges."""

  # Round towards zero.
  coords_fix = jnp.int32(jnp.fix(coords))

  # A very hacky plus-shaped edge detector.
  coords_fix_pad = jnp.pad(coords_fix, [(1, 1), (1, 1), (0, 0)], 'edge')
  mask = ((coords_fix == coords_fix_pad[2:, 1:-1, :]) &
          (coords_fix == coords_fix_pad[:-2, 1:-1, :])
          & (coords_fix == coords_fix_pad[1:-1, 2:, :])
          & (coords_fix == coords_fix_pad[1:-1, :-2, :]))

  # Scale according to `acc` and clip to lie in [-1, 1].
  max_val = jnp.maximum(
      1,
      math.weighted_percentile(
          jnp.max(jnp.abs(coords_fix), axis=2), acc, percentile))
  coords_fix_unit = jnp.clip(coords_fix / max_val, -1, 1)

  # The [-1, 1] center cube is gray, and every other integer boundary gets
  # colored with xyz \propto rgb - gray. Edge pixels are highlighted.
  return matte(
      jnp.where(mask, (coords_fix_unit + 1) / 2, 1 - jnp.abs(coords_fix_unit)),
      acc)


def visualize_coord_norm(coords, acc, percentile=99.):
  """Visualize the distance of each coordinate from the origin."""
  # Euclidean distance from the origin.
  coords_norm = jnp.linalg.norm(coords, axis=2)

  # Shrink each distance towards its floor, so "shells" are visible.
  coords_norm_floor = jnp.floor(coords_norm)
  value = (coords_norm - coords_norm_floor) / 2 + coords_norm_floor

  # Normalize and map near to red and blue to far.
  colormap = cm.get_cmap('turbo')
  max_val = math.weighted_percentile(value, acc, percentile)
  return matte(colormap(1 - jnp.clip(value / max_val, 0, 1))[:, :, :3], acc)


def maximize_saturation(rgb):
  """Rescale the maximum saturation in `rgb` to be 1."""
  hsv = pix.rgb_to_hsv(rgb)
  scaling = jnp.maximum(1, jnp.nan_to_num(1 / jnp.max(hsv[Ellipsis, 1]), nan=1))
  rgb_scaled = pix.hsv_to_rgb(
      jnp.stack([hsv[Ellipsis, 0], scaling * hsv[Ellipsis, 1], hsv[Ellipsis, 2]], axis=-1))
  return rgb_scaled


def visualize_rays(t_vals,
                   weights,
                   rgbs,
                   t_range,
                   accumulate=False,
                   renormalize=False,
                   resolution=512,
                   oversample=1024,
                   bg_color=0.8):
  """Visualize a bundle of rays."""
  t_vis = jnp.linspace(*t_range, oversample * resolution)
  vis_rgb, vis_alpha = [], []
  for ts, ws, rs in zip(t_vals, weights, rgbs):
    vis_rs, vis_ws = [], []
    for t, w, r in zip(ts, ws, rs):
      if accumulate:
        # Produce the accumulated color and weight at each point along the ray.
        w_csum = jnp.cumsum(w, axis=0)
        rw_csum = jnp.cumsum((r * w[:, None]), axis=0)
        eps = jnp.finfo(jnp.float32).eps
        r, w = (rw_csum + eps) / (w_csum[:, None] + 2 * eps), w_csum
      idx = jnp.searchsorted(t, t_vis) - 1
      bounds = 0, len(t) - 2
      mask = (idx >= bounds[0]) & (idx <= bounds[1])
      r_mat = jnp.where(mask[:, None], r[jnp.clip(idx, *bounds), :], 0)
      w_mat = jnp.where(mask, w[jnp.clip(idx, *bounds)], 0)
      # Grab the highest-weighted value in each oversampled span.
      r_mat = r_mat.reshape(resolution, oversample, -1)
      w_mat = w_mat.reshape(resolution, oversample)
      mask = w_mat == w_mat.max(axis=1, keepdims=True)
      r_ray = (mask[Ellipsis, None] * r_mat).sum(axis=1) / jnp.maximum(
          1, mask.sum(axis=1))[:, None]
      w_ray = (mask * w_mat).sum(axis=1) / jnp.maximum(1, mask.sum(axis=1))
      vis_rs.append(r_ray)
      vis_ws.append(w_ray)
    vis_rgb.append(jnp.stack(vis_rs))
    vis_alpha.append(jnp.stack(vis_ws))
  vis_rgb = jnp.stack(vis_rgb, axis=1)
  vis_alpha = jnp.stack(vis_alpha, axis=1)

  if renormalize:
    # Scale the alphas so that the largest value is 1, for visualization.
    vis_alpha /= jnp.maximum(jnp.finfo(jnp.float32).eps, jnp.max(vis_alpha))

  if resolution > vis_rgb.shape[0]:
    rep = resolution // (vis_rgb.shape[0] * vis_rgb.shape[1] + 1)
    stride = rep * vis_rgb.shape[1]

    vis_rgb = vis_rgb.tile((1, 1, rep, 1)).reshape((-1,) + vis_rgb.shape[2:])
    vis_alpha = vis_alpha.tile((1, 1, rep)).reshape((-1,) + vis_alpha.shape[2:])

    # Add a strip of background pixels after each set of levels of rays.
    vis_rgb = vis_rgb.reshape((-1, stride) + vis_rgb.shape[1:])
    vis_alpha = vis_alpha.reshape((-1, stride) + vis_alpha.shape[1:])
    vis_rgb = jnp.concatenate([vis_rgb, jnp.zeros_like(vis_rgb[:, :1])],
                              axis=1).reshape((-1,) + vis_rgb.shape[2:])
    vis_alpha = jnp.concatenate(
        [vis_alpha, jnp.zeros_like(vis_alpha[:, :1])],
        axis=1).reshape((-1,) + vis_alpha.shape[2:])

  # Matte the RGB image over the background.
  vis = vis_rgb * vis_alpha[Ellipsis, None] + (bg_color * (1 - vis_alpha))[Ellipsis, None]

  # Remove the final row of background pixels.
  vis = vis[:-1]
  vis_alpha = vis_alpha[:-1]
  return vis, vis_alpha


def visualize_suite(rendering, rays, config):
  """A wrapper around other visualizations for easy integration."""
  del config  # Unused.
  depth_curve_fn = lambda x: -jnp.log(x + jnp.finfo(jnp.float32).eps)

  rgb = rendering['rgb']
  acc = rendering['acc']
  normals = rendering['normals'] / 2. + 0.5
  distance_mean = rendering['distance_mean']
  distance_median = rendering['distance_median']
  distance_std = rendering['distance_std']
  distance_p5 = rendering['distance_percentile_5']  # pylint: disable=unused-variable
  distance_p95 = rendering['distance_percentile_95']  # pylint: disable=unused-variable

  acc = jnp.where(jnp.isnan(distance_mean), jnp.zeros_like(acc), acc)

  # The xyz coordinates where rays terminate.
  coords = rays.origins + rays.directions * distance_mean[:, :, None]  # pylint: disable=unused-variable

  rgb_matte = matte(rgb, acc)

  vis_mean, vis_median = [
      visualize_cmap(x, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn)
      for x in [distance_mean, distance_median]
  ]

  vis_mean_mod, vis_median_mod = [
      visualize_cmap(x, acc, sinebow, modulus=0.1, curve_fn=depth_curve_fn)
      for x in [distance_mean, distance_median]
  ]

  vis_std = visualize_cmap(distance_std, acc, cm.get_cmap('plasma'))

  vis = {
      'color': rgb,
      'acc': acc,
      'normals': normals,
      'color_matte': rgb_matte,
      'depth_mean': vis_mean,
      'depth_median': vis_median,
      'depth_mean_mod': vis_mean_mod,
      'depth_median_mod': vis_median_mod,
      'depth_std': vis_std,
  }

  return vis


def visualize_depth(x, acc, lo=None, hi=None):
  """Visualizes depth maps."""

  depth_curve_fn = lambda x: -jnp.log(x + jnp.finfo(jnp.float32).eps)
  return visualize_cmap(
      x, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn, lo=lo, hi=hi)
