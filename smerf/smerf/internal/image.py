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

"""Functions for processing images."""

import types
from typing import Optional, Union

from camp_zipnerf.internal import math as teacher_math
import dm_pix
import jax
import jax.numpy as jnp
from matplotlib import cm
import numpy as np


_Array = Union[np.ndarray, jnp.ndarray]


def imgs_to_psnr(gt, pred):
  mse = jnp.mean(jnp.square(gt - pred))
  return mse_to_psnr(mse)


def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10.0 / jnp.log(10.0) * jnp.log(mse)


def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return jnp.exp(-0.1 * jnp.log(10.0) * psnr)


def ssim_to_dssim(ssim):
  """Compute DSSIM given an SSIM."""
  return (1 - ssim) / 2


def dssim_to_ssim(dssim):
  """Compute DSSIM given an SSIM."""
  return 1 - 2 * dssim


def linear_to_srgb(
    linear, eps = None, xnp = jnp
):
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = xnp.finfo(xnp.float32).eps
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * xnp.maximum(eps, linear) ** (5 / 12) - 11) / 200
  return xnp.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(
    srgb, eps = None, xnp = jnp
):
  """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = xnp.finfo(xnp.float32).eps
  linear0 = 25 / 323 * srgb
  linear1 = xnp.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
  return xnp.where(srgb <= 0.04045, linear0, linear1)


def downsample(img, factor):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  if not (sh[0] % factor == 0 and sh[1] % factor == 0):
    raise ValueError(
        f'Downsampling factor {factor} does not '
        f'evenly divide image shape {sh[:2]}'
    )
  img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
  img = img.mean((1, 3))
  return img


def compute_vignette(coords, weights, powers=(1, 2, 3)):
  """Compute a vignetting as a polynomial function of image plane radius."""
  radius_squared = jnp.sum(jnp.square(coords), axis=-1)
  features = radius_squared[Ellipsis, None] ** jnp.array(powers)
  scaling = jnp.exp(-jnp.sum(jnp.abs(weights) * features, axis=-1))
  return scaling


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
  """Warp `img` to match the colors in `ref_img`."""
  if img.shape[-1] != ref.shape[-1]:
    raise ValueError(
        f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
    )
  num_channels = img.shape[-1]
  img_mat = img.reshape([-1, num_channels])
  ref_mat = ref.reshape([-1, num_channels])
  is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
  mask0 = is_unclipped(img_mat)
  # Because the set of saturated pixels may change after solving for a
  # transformation, we repeatedly solve a system `num_iters` times and update
  # our estimate of which pixels are saturated.
  for _ in range(num_iters):
    # Construct the left hand side of a linear system that contains a quadratic
    # expansion of each pixel of `img`.
    a_mat = []
    for c in range(num_channels):
      a_mat.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])  # Quadratic term.
    a_mat.append(img_mat)  # Linear term.
    a_mat.append(jnp.ones_like(img_mat[:, :1]))  # Bias term.
    a_mat = jnp.concatenate(a_mat, axis=-1)
    warp = []
    for c in range(num_channels):
      # Construct the right hand side of a linear system containing each color
      # of `ref`.
      b = ref_mat[:, c]
      # Ignore rows of the linear system that were saturated in the input or are
      # saturated in the current corrected color estimate.
      mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
      ma_mat = jnp.where(mask[:, None], a_mat, 0)
      mb = jnp.where(mask, b, 0)
      # Solve the linear system. We're using the np.lstsq instead of jnp because
      # it's significantly more stable in this case, for some reason.
      w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
      assert jnp.all(jnp.isfinite(w))
      warp.append(w)
    warp = jnp.stack(warp, axis=-1)
    # Apply the warp to update img_mat.
    img_mat = jnp.clip(
        jnp.matmul(a_mat, warp, precision=jax.lax.Precision.HIGHEST), 0, 1
    )
  corrected_img = jnp.reshape(img_mat, img.shape)
  return corrected_img


def compute_inlier_bounds(x, q, xnp=jnp):
  """Computes lower and upper bounds for values in an array.

  Args:
    x: f32[...]. Array of arbitrary shape.
    q: float. Value in [0, 1]. Inner quantile.
    xnp: numpy or jax.numpy.

  Returns:
    lower: float. Lower bound for q-th quantile.
    upper: float. Upper bound for q-th quantile.
  """
  low = (1-q)/2
  high = low + q
  low, high = xnp.quantile(x, xnp.array([low, high]))
  return low, high


def colorize_depth(depth, bounds=None):
  """Colorizes one or more depth maps.

  Depth maps are colored such that red is close and blue is far. Color scale
  is calculated after a power_ladder transform.

  Args:
    depth: f32[...]. Depth values.
    bounds: tuple[float, float] or None. Lower and upper bounds for depth.
      If omitted, do no clipping.

  Returns:
    f32[..., 3]. Colorized depth values. Stored on host memory.
  """

  img = depth

  # Apply the power ladder transform to ensure that most of the colormap's
  # dynamic range is close to the camera.
  transform = lambda x: teacher_math.power_ladder(x, p=-1.5, premult=2.0)
  img = transform(img)

  # Clip outlier pixels.
  if bounds is None:
    low = jnp.min(img)
    high = jnp.max(img)
  else:
    low, high = bounds
    img = jnp.clip(img, transform(low), transform(high))

  # Drop outliers
  img = (img - low) / (high - low)

  # Colorize with cmap
  img = cm.get_cmap('turbo')(1 - img)[Ellipsis, :3]
  return np.array(img)


class MetricHarness:
  """A helper class for evaluating several error metrics."""

  def __init__(self, disable_ssim=False, disable_lpips=True, device=None):
    _ = device
    self.ssim_fn = lambda x, y: np.nan
    if not disable_ssim:
      self.ssim_fn = jax.jit(dm_pix.ssim)

    if not disable_lpips:
      raise NotImplementedError('LPIPS is not implemented in this codebase.')

  def __call__(self, rgb_pred, rgb_gt, name_fn=lambda s: s):
    """Evaluate the error between a predicted rgb image and the true image.

    Args:
      rgb_pred: f32[h,w,3]. Predicted RGB. Values in [0, 1].
      rgb_gt: f32[h,w,3]. Ground truth RGB. Values in [0, 1].
      name_fn: ...

    Returns:
      {str: float} metric values.
    """
    psnr = float(imgs_to_psnr(rgb_pred, rgb_gt))
    ssim = float(self.ssim_fn(rgb_pred, rgb_gt))

    return {
        name_fn('psnr'): psnr,
        name_fn('ssim'): ssim,
    }
