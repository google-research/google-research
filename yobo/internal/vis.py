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

# pylint: skip-file
"""Helper functions for visualizing things."""

from google_research.yobo.internal import stepfun
import jax.numpy as jnp
from matplotlib import cm


def colorize(
    x,
    percentile=99.0,
    pos_color=(1, 0, 0),
    neg_color=(0, 0, 1),
    bg_color=(1, 1, 1),
):
  """Map +x to pos_color, -x to neg_color, and x==0 to bg_color."""
  max_val = jnp.percentile(jnp.abs(x), percentile)
  x_norm = x / max_val
  x_pos = jnp.clip(x_norm, 0, 1)
  x_neg = jnp.clip(-x_norm, 0, 1)
  x_zero = jnp.maximum(0, 1 - jnp.abs(x_norm))
  vis = (
      x_pos[Ellipsis, None] * jnp.array(pos_color)[None, None]
      + x_neg[Ellipsis, None] * jnp.array(neg_color)[None, None]
      + x_zero[Ellipsis, None] * jnp.array(bg_color)[None, None]
  )
  return vis


def weighted_percentile(x, w, ps, assume_sorted=False):
  """Compute the weighted percentile(s) of a single vector."""
  x = x.reshape([-1])
  w = w.reshape([-1])
  if not assume_sorted:
    sortidx = jnp.argsort(x)
    x, w = x[sortidx], w[sortidx]
  acc_w = jnp.cumsum(w)
  return jnp.interp(jnp.array(ps) * (acc_w[-1] / 100), acc_w, x)


def sinebow(h):
  """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
  f = lambda x: jnp.sin(jnp.pi * x) ** 2
  return jnp.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
  """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
  bg_mask = jnp.logical_xor(
      (jnp.arange(acc.shape[-2]) % (2 * width) // width)[Ellipsis, :, None],
      (jnp.arange(acc.shape[-1]) % (2 * width) // width)[Ellipsis, None, :],
  )
  bg = jnp.where(bg_mask, light, dark)
  return vis * acc[Ellipsis, None] + (bg * (1 - acc))[Ellipsis, None]


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
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
  lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2]
  )

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
    value = jnp.clip((value - jnp.minimum(lo, hi)) / jnp.abs(hi - lo), 0, 1)
  value = jnp.nan_to_num(value)

  if colormap:
    colorized = colormap(value)[Ellipsis, :3]
  else:
    if value.shape[-1] != 3:
      raise ValueError(f'value must have 3 channels but has {value.shape[-1]}')
    colorized = value

  return matte(colorized, weight) if matte_background else colorized


def visualize_rays(
    dist,
    dist_range,
    weights,
    rgbs,
    accumulate=False,
    renormalize=False,
    resolution=2048,
    bg_color=0.8,
):
  """Visualize a bundle of rays."""
  dist_vis = jnp.linspace(*dist_range, resolution + 1)
  vis_rgb, vis_alpha = [], []
  for ds, ws, rs in zip(dist, weights, rgbs):
    vis_rs, vis_ws = [], []
    for d, w, r in zip(ds, ws, rs):
      if accumulate:
        # Produce the accumulated color and weight at each point along the ray.
        w_csum = jnp.cumsum(w, axis=0)
        rw_csum = jnp.cumsum((r * w[:, None]), axis=0)
        eps = jnp.finfo(jnp.float32).eps
        r, w = (rw_csum + eps) / (w_csum[:, None] + 2 * eps), w_csum
      vis_rs.append(stepfun.resample(dist_vis, d, r.T, use_avg=True).T)
      vis_ws.append(stepfun.resample(dist_vis, d, w.T, use_avg=True).T)
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

    vis_rgb = jnp.tile(vis_rgb, (1, 1, rep, 1)).reshape(
        (-1,) + vis_rgb.shape[2:]
    )
    vis_alpha = jnp.tile(vis_alpha, (1, 1, rep)).reshape(
        (-1,) + vis_alpha.shape[2:]
    )

    # Add a strip of background pixels after each set of levels of rays.
    vis_rgb = vis_rgb.reshape((-1, stride) + vis_rgb.shape[1:])
    vis_alpha = vis_alpha.reshape((-1, stride) + vis_alpha.shape[1:])
    vis_rgb = jnp.concatenate(
        [vis_rgb, jnp.zeros_like(vis_rgb[:, :1])], axis=1
    ).reshape((-1,) + vis_rgb.shape[2:])
    vis_alpha = jnp.concatenate(
        [vis_alpha, jnp.zeros_like(vis_alpha[:, :1])], axis=1
    ).reshape((-1,) + vis_alpha.shape[2:])

  # Matte the RGB image over the background.
  vis = vis_rgb * vis_alpha[Ellipsis, None] + (bg_color * (1 - vis_alpha))[Ellipsis, None]

  # Remove the final row of background pixels.
  vis = vis[:-1]
  vis_alpha = vis_alpha[:-1]
  return vis, vis_alpha


def visualize_suite(rendering):
  """A wrapper around other visualizations for easy integration."""

  depth_curve_fn = lambda x: -jnp.log(x + jnp.finfo(jnp.float32).eps)

  rgb = rendering['rgb']
  acc = jnp.stack(rendering['acc'], axis=0)

  distance_mean = jnp.stack(rendering['distance_mean'], axis=0)
  distance_median = jnp.stack(rendering['distance_median'], axis=0)
  distance_p5 = jnp.stack(rendering['distance_percentile_5'], axis=0)
  distance_p95 = jnp.stack(rendering['distance_percentile_95'], axis=0)
  acc = jnp.where(jnp.isnan(distance_mean), jnp.zeros_like(acc), acc)

  p = 99.0
  lo, hi = weighted_percentile(distance_median, acc, [50 - p / 2, 50 + p / 2])

  def vis_fn(x):
    return visualize_cmap(
        x, acc, cm.get_cmap('turbo'), lo=lo, hi=hi, curve_fn=depth_curve_fn
    )

  vis_depth_mean, vis_depth_median = [
      vis_fn(x) for x in [distance_mean, distance_median]
  ]

  # Render three depth percentiles directly to RGB channels, where the spacing
  # determines the color. delta == big change, epsilon = small change.
  #   Gray: A strong discontinuitiy, [x-epsilon, x, x+epsilon]
  #   Purple: A thin but even density, [x-delta, x, x+delta]
  #   Red: A thin density, then a thick density, [x-delta, x, x+epsilon]
  #   Blue: A thick density, then a thin density, [x-epsilon, x, x+delta]
  vis_depth_triplet = visualize_cmap(
      jnp.stack(
          [2 * distance_median - distance_p5, distance_median, distance_p95],
          axis=-1,
      ),
      acc,
      None,
      curve_fn=lambda x: jnp.log(x + jnp.finfo(jnp.float32).eps),
  )

  dist = rendering['ray_sdist']
  dist_range = (0, 1)
  weights = rendering['ray_weights']
  rgbs = [jnp.clip(r, 0, 1) for r in rendering['ray_rgbs']]

  vis_ray_colors, _ = visualize_rays(dist, dist_range, weights, rgbs)

  sqrt_weights = [jnp.sqrt(w) for w in weights]
  sqrt_ray_weights, ray_alpha = visualize_rays(
      dist,
      dist_range,
      [jnp.ones_like(lw) for lw in sqrt_weights],
      [lw[Ellipsis, None] for lw in sqrt_weights],
      bg_color=0,
  )
  sqrt_ray_weights = sqrt_ray_weights[Ellipsis, 0]

  null_color = jnp.array([1.0, 0.0, 0.0])
  vis_ray_weights = jnp.where(
      ray_alpha[:, :, None] == 0,
      null_color[None, None],
      visualize_cmap(
          sqrt_ray_weights,
          jnp.ones_like(sqrt_ray_weights),
          cm.get_cmap('gray'),
          lo=0,
          hi=1,
          matte_background=False,
      ),
  )

  # Need to convert from array back to list for tensorboard logging.
  # This applies to all buffers except rgb and roughness.
  is_list = isinstance(rendering['acc'], list)
  unstack_fn = lambda z: list(z) if is_list else z
  unstack1_fn = lambda z: list(z)[-1:] if is_list else z
  vis = {
      'acc': unstack_fn(acc),
      'color_matte': unstack1_fn(matte(rgb, acc)),
      'depth_mean': unstack1_fn(vis_depth_mean),
      'depth_median': unstack1_fn(vis_depth_median),
      'depth_triplet': unstack_fn(vis_depth_triplet),
      'ray_colors': vis_ray_colors,
      'ray_weights': vis_ray_weights,
  }

  if 'rgb_cc' in rendering:
    vis['color_corrected'] = rendering['rgb_cc']

  # Render every item named "normals*".
  for key, val in rendering.items():
    if key.startswith('normals'):
      vis[key] = unstack_fn(matte(jnp.stack(val, axis=0) / 2.0 + 0.5, acc))

  if 'roughness' in rendering:
    roughness = jnp.stack(rendering['roughness'], axis=0)
    vis['roughness'] = unstack_fn(matte(jnp.tanh(roughness), acc))

  return vis
