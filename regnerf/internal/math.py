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

"""Mathy utility functions."""
import jax
import jax.numpy as jnp
import numpy as np


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.where(jnp.abs(x) < t, x, x % t))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / jnp.log(10.) * jnp.log(mse)


def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return jnp.exp(-0.1 * jnp.log(10.) * psnr)


def weighted_percentile(x, w, ps, assume_sorted=False):
  """Compute the weighted percentile(s) of a single vector."""
  x = x.reshape([-1])
  w = w.reshape([-1])
  if not assume_sorted:
    sortidx = jnp.argsort(jax.lax.stop_gradient(x))
    x, w = x[sortidx], w[sortidx]
  acc_w = jnp.cumsum(w)
  return jnp.interp(jnp.array(ps) * (acc_w[-1] / 100), acc_w, x)


def compute_avg_error(psnr, ssim, lpips):
  """The 'average' error used in the paper."""
  mse = psnr_to_mse(psnr)
  dssim = jnp.sqrt(1 - ssim)
  return jnp.exp(jnp.mean(jnp.log(jnp.array([mse, dssim, lpips]))))


def compute_ternary_census(im, epsilon, boundary='edge'):
  """Compute the ternary census transform for an image."""
  assert len(im.shape) == 3  # Expects a single image [height, width, channels]
  assert epsilon >= 0
  im_pad = jnp.pad(im, [(1, 1), (1, 1), (0, 0)], boundary)
  census = []
  for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
      if di == dj == 0:
        continue
      im_rolled = jnp.roll(jnp.roll(im_pad, di, -2), dj, -3)[1:-1, 1:-1, :]
      delta = im - im_rolled
      census.append(
          jnp.where(jnp.abs(delta) <= epsilon, 0, jnp.int8(jnp.sign(delta))))
  return jnp.stack(census, -1)


def compute_census_err(im0, im1, epsilon):
  """Computes an error between two images using a (ternary) census transform.

  This error is large when things are very wrong, and will be 0 when the match
  is perfect --- but can also be "gamed" to produce low errors just by producing
  and all-constant prediction, which will produce a zero error. As such, this
  metric should not be used in isolation: do not rely on it!

  Args:
    im0: array. A image of size [width, height, num_channels].
    im1: array. Another image of size [width, height, num_channels].
    epsilon: float > 0. The difference in intensities to be invariant to. Should
      probably be something like the size of the quantization intervals of the
      input images.

  Returns:
    The frequency of opposite-signed ternary census transforms of the images.
  """
  census0, census1 = [compute_ternary_census(x, epsilon) for x in [im0, im1]]
  return jnp.mean(jnp.abs(census0 - census1) > 1)


def linear_to_srgb(linear):
  # Assumes `linear` is in [0, 1]. https://en.wikipedia.org/wiki/SRGB
  eps = jnp.finfo(jnp.float32).eps
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * jnp.maximum(eps, linear)**(5 / 12) - 11) / 200
  return jnp.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb):
  # Assumes `srgb` is in [0, 1]. https://en.wikipedia.org/wiki/SRGB
  eps = jnp.finfo(jnp.float32).eps
  linear0 = 25 / 323 * srgb
  linear1 = jnp.maximum(eps, ((200 * srgb + 11) / (211)))**(12 / 5)
  return jnp.where(srgb <= 0.04045, linear0, linear1)


def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = jnp.log(v0)
  lv1 = jnp.log(v1)
  return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def sorted_piecewise_constant_pdf(rng,
                                  bins,
                                  weights,
                                  num_samples,
                                  single_jitter=False):
  """Piecewise-Constant PDF sampling from sorted bins.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    bins: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    weights: [..., num_bins], bin interval weights (must be non-negative).
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = jnp.finfo('float32').eps

  # Pad each weight vector (only if necessary) to bring its mean to `eps`. This
  # avoids NaNs when the input is zeros or small, but has no effect otherwise.
  weights += jnp.maximum(0, eps - jnp.sum(weights, axis=-1, keepdims=True))
  weight_sum = jnp.sum(weights, axis=-1, keepdims=True)

  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0 and ends with exactly 1.
  pdf = weights / weight_sum
  cdf = jnp.minimum(1, jnp.cumsum(pdf[Ellipsis, :-1], axis=-1))
  cdf = jnp.concatenate([
      jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf,
      jnp.ones(list(cdf.shape[:-1]) + [1])
  ],
                        axis=-1)

  # Draw uniform samples.
  if rng is None:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    u = jnp.linspace(0., 1. - eps, num_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
  else:
    s = 1 / num_samples
    u = jnp.arange(num_samples) * s
    d = 1 if single_jitter else num_samples
    u += jax.random.uniform(rng, list(cdf.shape[:-1]) + [d], maxval=s - eps)

    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u = jnp.minimum(u, 1. - eps)

  # Identify the location in `cdf` that corresponds to a random sample.
  # The final `True` index in `mask` will be the start of the sampled interval.
  mask = u[Ellipsis, None, :] >= cdf[Ellipsis, :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0 = jnp.max(jnp.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
    return x0, x1

  bins_g0, bins_g1 = find_interval(bins)
  cdf_g0, cdf_g1 = find_interval(cdf)

  t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
  samples = bins_g0 + t * (bins_g1 - bins_g0)
  return samples


def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
  """Returns TV norm for input values.

  Note: The weighting / masking term was necessary to avoid degenerate
  solutions on GPU; only observed on individual DTU scenes.
  """
  v00 = values[:, :-1, :-1]
  v01 = values[:, :-1, 1:]
  v10 = values[:, 1:, :-1]

  if losstype == 'l2':
    loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
  elif losstype == 'l1':
    loss = jnp.abs(v00 - v01) + jnp.abs(v00 - v10)
  else:
    raise ValueError('Not supported losstype.')

  if weighting is not None:
    loss = loss * weighting
  return loss


def compute_tvnorm_weight(step, max_step, weight_start=0.0, weight_end=0.0):
  """Computes loss weight for tv norm."""
  w = np.clip(step * 1.0 / (1 if (max_step < 1) else max_step), 0, 1)
  return weight_start * (1 - w) + w * weight_end
