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

import dm_pix
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

_Array = Union[np.ndarray, jnp.ndarray]


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
  scaling = jnp.exp(-jnp.sum(jnp.abs(weights) * features[Ellipsis, None], axis=-2))
  return scaling


def render_histogram(x, **kwargs):
  """Call pyplot's hist() and render it to a numpy buffer."""
  fig = plt.figure()
  fig.gca().hist(x, **kwargs)
  fig.canvas.draw()
  hw = fig.canvas.get_width_height()[::-1]
  buf = fig.canvas.tostring_rgb()
  array = np.frombuffer(buf, dtype=np.uint8).reshape(hw + (3,))
  plt.close(fig)
  return array


class MetricHarness:
  """A helper class for evaluating several error metrics."""

  def __init__(
      self,
      disable_ssim=False,
  ):
    if disable_ssim:
      self.ssim_fn = None
    else:
      self.ssim_fn = jax.jit(dm_pix.ssim)

  def __call__(self, rgb_pred, rgb_gt, name_fn=lambda s: s):
    """Evaluate the error between a predicted rgb image and the true image."""
    metrics = {}
    metrics['psnr'] = mse_to_psnr(((rgb_pred - rgb_gt) ** 2).mean())
    if self.ssim_fn is not None:
      metrics['ssim'] = self.ssim_fn(rgb_pred, rgb_gt)

    # Apply the name function and cast all metrics down to a scalar float.
    return {name_fn(k): float(v) for (k, v) in metrics.items()}
