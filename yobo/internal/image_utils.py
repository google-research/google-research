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
"""Functions for processing images."""

import itertools
import types
from typing import Optional, Union

import chex
import dm_pix
from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

matplotlib.use('Agg')

LPIPS_TFHUB_PATH = '@spectra/metrics/lpips/net-lin_vgg_v0.1/4'

_Array = Union[np.ndarray, jnp.ndarray]


def rgb_to_yuv(rgb):
  # Taken from _rgb_to_yuv_kernel in tf.image.rgb_to_yuv().
  mat = jnp.array([
      [0.299, -0.14714119, +0.61497538],
      [0.587, -0.28886916, -0.51496512],
      [0.114, +0.43601035, -0.10001026],
  ])
  return rgb @ mat


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


def compute_shift_invariant_metric(
    im0,
    im1,
    metric_fn,
    reduction,
    search_radii,
    window_halfwidth,
    boundary='reflect',
):
  """Makes an error function between two images shift-invariant.

  This is a helper function for turning a regular error metric on pairs of color
  images into a shift-invariant version of that metric. This works by manually
  applying the metric on shifted versions of the input image, applying a box
  filter to those blurred errors, and using the argmin/argmax of those blurred
  metrics to return a per-pixel error map. This box filter forces the argmin/
  argmax to pick the offset for each pixel that yields an image *patch* that
  best minimizes the error.

  Args:
    im0: (x, y, c), one color image.
    im1: (x, y, c), another color image.
    metric_fn: a function that takes in two color images and produces a
      per-pixel error map.
    reduction: a string in ['argmin', 'argmax'] that determines whether to use
      the argmin or argmax of error metrics. If metric_fn returns errors, use
      'argmin', but if it returns "scores", use 'argmax'.
    search_radii: a tuple of integers (i, j) that determines how far to search
      when trying different shifts along the first two spatial axes of the
      image.
    window_halfwidth: an integer that determines the half-width of the window
      size used when box-filtering. This effectively determines the tile size
      used when selecting offets, which is `(2 * window_halfwidth + 1,) * 2`.
    boundary: a string that determines the boundary condition to use when
      shifting the image, which must be a valid "mode" for jnp.pad().

  Returns:
    opt_metric: a per-pixel map of whatever format metric_fn outputs.
    opt_di: The integer per-pixel alignment along the first spatial axis that
      minimizes or maximizes the metric with regard to offsets.
    opt_dj: The same as opt_di, but over the second spatial axis.
  """
  if (len(im0.shape) != 3) or (im0.shape[2] != 3) or (im0.shape != im1.shape):
    raise ValueError(f'Invalid input image shapes {im0.shape}, {im1.shape}')

  # Pad im0 by the search radii.
  i_radius, j_radius = search_radii
  im0_pad = jnp.pad(im0, [[i_radius] * 2, [j_radius] * 2, (0, 0)], boundary)

  # Search over all offsets in the search radius.
  opt_metric_pooled = None
  for di in range(-i_radius, i_radius + 1):
    for dj in range(-j_radius, j_radius + 1):
      # Roll and crop the padded im0 to shift it.
      im0_rolled = jnp.roll(jnp.roll(im0_pad, -di, 0), -dj, 1)
      im0_cropped = im0_rolled[
          i_radius : im0_rolled.shape[0] - i_radius,
          j_radius : im0_rolled.shape[1] - j_radius,
          :,
      ]

      # Evaluate the metric map on the shifted im0 and im1.
      metric = metric_fn(im0_cropped, im1)

      # Blur the metric map with a box filter.
      metric_pooled = nn.avg_pool(
          metric[None, :, :, None],
          (2 * window_halfwidth + 1,) * 2,
          padding='same',
      )[0, :, :, 0]

      if opt_metric_pooled is None:
        # If this is the first offset we've looked at, just take it.
        opt_metric_pooled = metric_pooled
        opt_metric = metric
        opt_di = di
        opt_dj = dj
      else:
        # For all pixels where the new pooled metric is better than the previous
        # best, record their error and offset.
        if reduction == 'argmax':
          take = metric_pooled >= opt_metric_pooled
        elif reduction == 'argmin':
          take = metric_pooled <= opt_metric_pooled
        else:
          raise ValueError("reduction must either 'argmax' or 'argmin'")
        opt_metric_pooled = jnp.where(take, metric_pooled, opt_metric_pooled)

        opt_metric = jnp.where(take, metric, opt_metric)
        opt_di = jnp.where(take, di, opt_di)
        opt_dj = jnp.where(take, dj, opt_dj)

  return opt_metric, opt_di, opt_dj


def shift_invariant_ssim(img0, img1, *args):
  """A shift-invariant version of SSIM, see compute_shift_invariant_metric()."""
  pad = 5  # Pad images by 5 because SSIM uses an 11 x 11 window by default.

  def score_fn(x, y):
    def pad_fn(z):
      return jnp.pad(z, [[pad] * 2, [pad] * 2, [0] * 2], mode='reflect')

    return jnp.mean(dm_pix.ssim(pad_fn(x), pad_fn(y), return_map=True), axis=-1)

  opt_score, opt_di, opt_dj = compute_shift_invariant_metric(
      img0, img1, score_fn, 'argmax', *args
  )
  opt_score = jnp.mean(opt_score[pad:-pad, pad:-pad])
  return opt_score, opt_di, opt_dj


def shift_invariant_mse(img0, img1, *args):
  """A shift-invariant version of MSE, see compute_shift_invariant_metric()."""
  err_fn = lambda x, y: jnp.mean((x - y) ** 2, axis=-1)
  opt_score, opt_di, opt_dj = compute_shift_invariant_metric(
      img0, img1, err_fn, 'argmin', *args
  )
  opt_score = jnp.mean(opt_score)
  return opt_score, opt_di, opt_dj


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


def precompute_nlinear_weights(coords, grid_shape):
  """Precompute n-linear interpolation weights.

  Constructs the weights for n-linear (linear, bilinear, trilinear, etc)
  interpolation for a set of n-dimensional coordinates and a desired
  grid/histogram shape with the same dimensionality. We assume the input
  coordinates have been scaled and shifted to lie within the grid shape.

  Args:
    coords: A matrix (n, d) of coordinates.
    grid_shape: A d-length list encoding the shape of the grid/histogram.

  Returns:
    idxs: A list of indices of the grid cells that each row of `coords` maps to.
    weights: the weights to use for each element of `idxs`.
  """
  if coords.shape[-1] != len(grid_shape):
    raise ValueError(
        f'`coord` dim {coords.shape} does not match grid dim {len(grid_shape)}'
    )

  # The index of the *0* corner of the grid, clamped so that out-of-bounds
  # behavior will be linear extrapolation. We're manually writing out the min
  # and max instead of using jnp.clip() because we want to prioritize the max
  # with zero over the min with the grid size in case there are grid dimensions
  # with a size of 1.
  idx0 = jnp.maximum(
      0,
      jnp.minimum(
          jnp.floor(coords).astype(jnp.int32), jnp.array(grid_shape) - 1
      ),
  )

  # The n-linear weight of the *0* corner of the grid.
  weight0 = 1.0 - (coords - idx0.astype(coords.dtype))

  idxs, weights = [], []
  for bitstring in list(itertools.product([0, 1], repeat=len(grid_shape))):
    idxs.append(idx0 + jnp.array(bitstring))
    weights.append(
        jnp.prod(
            weight0 * (1 - 2 * jnp.array(bitstring)) + jnp.array(bitstring), 1
        )
    )

  return idxs, weights


def splat_to_grid(idxs, weights, hist, values):
  """Splats `values` into `hist` according to precomputed (idxs, weights)."""
  hist_idxs = tuple(jnp.concatenate(idxs).T)
  splat_vals = jnp.concatenate([w * values for w in weights])
  hist = hist.at[hist_idxs].add(splat_vals)
  return hist


def slice_from_grid(idxs, weights, hist):
  """Slices values out of `hist` according to precomputed (idxs, weights)."""
  return sum([w * hist[tuple(i.T)] for w, i in zip(weights, idxs)])


def fit_and_apply_grid_warp(x, y, coords, grid_shape, lstsq_eps):
  """A helper function for correct_local_color with no image logic."""

  chex.assert_equal_shape([x, y])
  chex.assert_equal_shape([x[Ellipsis, -1], y[Ellipsis, -1], coords[Ellipsis, -1]])
  assert coords.shape[-1] == len(grid_shape)

  # Flatten the x, y, and coords input tensors to be 2D matrices.
  xy_shape = x.shape
  m = y.shape[-1]
  y = y.reshape([-1, m])
  x = x.reshape([-1, m])
  x = jnp.concatenate([x, jnp.ones_like(x[Ellipsis, :1])], axis=-1)
  n = x.shape[-1]
  coords = coords.reshape([-1, coords.shape[-1]])

  idxs, weights = precompute_nlinear_weights(coords, grid_shape)

  # Construct the left and right sides of the least squares equation.
  w_sq = [w**2 for w in weights]

  def vec2(fn):
    for _ in range(2):
      fn = jax.vmap(fn, in_axes=-1, out_axes=-1)
    return fn

  splat_fn = vec2(lambda z: splat_to_grid(idxs, w_sq, jnp.zeros(grid_shape), z))

  # Splat x^T x and x^T y to get the left and right hand sides of the
  # bilateral-space least squares equation.
  # Note: there's a potential 2x speedup here because x is symmetric, and
  # we're splatting all of it instead of just the upper triangular part.
  a = splat_fn(x[Ellipsis, :, None] * x[Ellipsis, None, :]) + lstsq_eps * jnp.eye(n, n)
  b = splat_fn(x[Ellipsis, :, None] * y[Ellipsis, None, :])

  # Solve the least-squares problem to get a grid of affine transformations.
  t_grid = jnp.vectorize(
      lambda x, y: jnp.linalg.lstsq(x, y)[0],
      signature=f'({n},{n}),({n},{m})->({n},{m})',
  )(a, b)

  # Slice out of the grid to get an image of affine transformations.
  slice_fn = vec2(lambda z: slice_from_grid(idxs, weights, z))
  t = slice_fn(t_grid).reshape(x.shape[:-1] + (n, m))

  # Apply the per-pixel affine transformations.
  z = jnp.vectorize(jnp.matmul, signature=f'({n}),({n},{m})->({m})')(x, t)

  z = z.reshape(xy_shape)

  return z


def correct_local_color(
    im,
    im_true,
    *,
    num_spatial_bins,
    num_luma_bins,
    num_chroma_bins,
    lstsq_eps=1e-5,
):
  """Fit a spatially varying curve that best fits `im` to `im_true`.

  This function implements an algorithm that is functionally equivalent to
  a simplified bilateral guided upsampling, where instead of fitting a
  spatially-varying curve at a low resolution and applying it at a high
  resolution, we fit and apply the curve at the full resolution. This produces
  a modified version of `im` in which structure and high-frequency edges are
  preserved by low-frequency (in bilateral-space) variation is warped to match
  that of `im_true`. This is useful for constructing an error metric between
  images that ignores low-frequency photometric variation (ie, renderings from
  models that use latent appearance embeddings like GLO).

  Args:
    im: (x, y, c), a color image, presumably the output of some rendering. We
      assume (without checking) that image intensities are in [0, 1], and this
      code will behave weirdly if that assumption is violated.
    im_true: (x, y, c), the ground-truth color image we want to reproduce.
    num_spatial_bins: a tuple of two integers that encodes the size of the
      bilateral grid along the spatial dimensions that we're constructing. If
      your image is something like 480 x 640, you should probably use something
      like (3, 4) (note that we're using 'ij' pixel coordinates).
    num_luma_bins: an integer that encodes the size of the bilateral grid along
      the luma dimensions. We use a YUV colorspace.
    num_chroma_bins: Like num_luma bins, but along the chroma (UV) dimensions.
    lstsq_eps: a small float that regularizes the least squares solve to avoid
      numerical instability or NaNs.

  Returns:
    im_curved: (x, y, c) a curve version of `im` that approximates `im_true`.
  """

  if (len(im.shape) != 3) or (im.shape[2] != 3) or (im.shape != im_true.shape):
    raise ValueError(f'Invalid input image shapes {im.shape}, {im_true.shape}')

  # Construct an (num_pixels, 5) matrix of YUV-XY coordinates scaled to fit
  # inside the grid.
  color_grid_shape = [num_luma_bins] + [num_chroma_bins] * 2
  grid_shape = color_grid_shape + num_spatial_bins

  im_yuv = rgb_to_yuv(im) + jnp.array([0.0, 0.5, 0.5])
  coords_color = im_yuv * (jnp.array(color_grid_shape) - 1)
  coords_spatial = jnp.stack(
      jnp.meshgrid(
          *[
              jnp.linspace(0, r - 1, s)
              for s, r in zip(im.shape[:-1], num_spatial_bins)
          ],
          indexing='ij',
      ),
      axis=-1,
  )
  coords = jnp.concatenate([coords_color, coords_spatial], axis=-1)

  x_clc = fit_and_apply_grid_warp(im, im_true, coords, grid_shape, lstsq_eps)
  im_clc = jnp.clip(x_clc, 0, 1)

  return im_clc


class MetricHarness:
  """A helper class for evaluating several error metrics."""

  def __init__(
      self,
      disable_ssim=False,
      disable_lpips=False,
      disable_search_invariant=False,
      search_radii=(4, 4),
      window_halfwidth=8,
  ):
    if disable_ssim:
      self.ssim_fn = None
    else:
      self.ssim_fn = jax.jit(dm_pix.ssim)

    if disable_search_invariant:
      self.si_mse_fn = None
    else:
      self.si_mse_fn = jax.jit(
          lambda x, y: shift_invariant_mse(  # pylint: disable=g-long-lambda
              x, y, search_radii, window_halfwidth
          )[0]
      )

    if disable_ssim or disable_search_invariant:
      self.si_ssim_fn = None
    else:
      self.si_ssim_fn = jax.jit(
          lambda x, y: shift_invariant_ssim(  # pylint: disable=g-long-lambda
              x, y, search_radii, window_halfwidth
          )[0]
      )

    # Stringify the parameters of the shift-invariant metrics for use later.
    self.si_suffix = f' {search_radii} {window_halfwidth}'.replace(' ', '_')

    if disable_lpips:
      self.lpips_fn = None
    else:
      # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
      # LPIPS computation or dataset loading.
      tf.config.experimental.set_visible_devices([], 'GPU')
      tf.config.experimental.set_visible_devices([], 'TPU')
      lpips_model = tf_hub.load(LPIPS_TFHUB_PATH)

      def lpips_fn(x, y):
        # The LPIPS model expects a batch dimension and requires float32 inputs.
        x = tf.convert_to_tensor(x[None], dtype=tf.float32)
        y = tf.convert_to_tensor(y[None], dtype=tf.float32)
        return lpips_model(x, y)[0]

      self.lpips_fn = lpips_fn

  def __call__(self, rgb_pred, rgb_gt, name_fn=lambda s: s):
    """Evaluate the error between a predicted rgb image and the true image."""
    metrics = {}
    metrics['psnr'] = mse_to_psnr(((rgb_pred - rgb_gt) ** 2).mean())
    if self.ssim_fn is not None:
      metrics['ssim'] = self.ssim_fn(rgb_pred, rgb_gt)
    if self.lpips_fn is not None:
      metrics['lpips'] = self.lpips_fn(rgb_pred, rgb_gt)
    if self.si_mse_fn is not None:
      metrics['psnr_si' + self.si_suffix] = mse_to_psnr(
          self.si_mse_fn(rgb_pred, rgb_gt)
      )
    if self.si_ssim_fn is not None:
      metrics['ssim_si' + self.si_suffix] = self.si_ssim_fn(rgb_pred, rgb_gt)

    if ('psnr' in metrics) and ('ssim' in metrics) and ('lpips' in metrics):
      # The geometric mean of MSE, sqrt(DSSIM), and LPIPS.
      # Note: in past papers barron accidentally used sqrt(1-ssim) instead of
      # sqrt((1 - ssim) / 2), so average errors produced by this code will be
      # different than previously published numbers.
      mse = psnr_to_mse(metrics['psnr'])
      sqrt_dssim = jnp.sqrt(ssim_to_dssim(metrics['ssim']))
      lpips = metrics['lpips']
      metrics['avg_err'] = jnp.exp(
          jnp.mean(jnp.log(jnp.array([mse, sqrt_dssim, lpips])))
      )

    # Apply the name function and cast all metrics down to a scalar float.
    return {name_fn(k): float(v) for (k, v) in metrics.items()}
