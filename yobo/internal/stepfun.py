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
"""Tools for manipulating step functions (piecewise-constant 1D functions).

We have a shared naming and dimension convention for these functions.
All input/output step functions are assumed to be aligned along the last axis.
`t` always indicates the x coordinates of the *endpoints* of a step function.
`y` indicates unconstrained values for the *bins* of a step function
`w` indicates bin weights that sum to <= 1. `p` indicates non-negative bin
values that *integrate* to <= 1.
"""

from google_research.yobo.internal import linspline
from google_research.yobo.internal import math
from google_research.yobo.internal import utils
import jax
import jax.numpy as jnp
import numpy as np


def query(tq, t, y, outside_value=0):
  """Look up the values of the step function (t, y) at locations tq."""
  utils.assert_valid_stepfun(t, y)
  idx_lo, idx_hi = math.searchsorted(t, tq, utils.device_is_tpu())
  # Note that `take_along_axis` may be slow on a TPU. This code isn't used
  # by the model and is just a hook for unit tests, so it doesn't matter.
  yq = jnp.where(
      idx_lo == idx_hi, outside_value, jnp.take_along_axis(y, idx_lo, axis=-1)
  )
  return yq


def inner_outer(t0, t1, y1):
  """Construct inner and outer measures on (t1, y1) for t0."""
  utils.assert_valid_stepfun(t1, y1)
  cy1 = jnp.concatenate(
      [jnp.zeros_like(y1[Ellipsis, :1]), jnp.cumsum(y1, axis=-1)], axis=-1
  )
  (idx_lo, idx_hi), ((cy1_lo, cy1_hi),) = math.sorted_lookup(
      t0, t1, (cy1,), utils.device_is_tpu()
  )

  y0_outer = cy1_hi[Ellipsis, 1:] - cy1_lo[Ellipsis, :-1]
  y0_inner = jnp.where(
      idx_hi[Ellipsis, :-1] <= idx_lo[Ellipsis, 1:], cy1_lo[Ellipsis, 1:] - cy1_hi[Ellipsis, :-1], 0
  )
  return y0_inner, y0_outer


def lossfun_outer(t, w, t_env, w_env, eps=jnp.finfo(jnp.float32).eps):
  """The proposal weight should be an upper envelope on the nerf weight."""
  utils.assert_valid_stepfun(t, w)
  utils.assert_valid_stepfun(t_env, w_env)
  _, w_outer = inner_outer(t, t_env, w_env)
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return jnp.maximum(0, w - w_outer) ** 2 / (w + eps)


def weight_to_pdf(t, w):
  """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
  utils.assert_valid_stepfun(t, w)
  td = jnp.diff(t)
  return jnp.where(td < np.finfo(np.float32).tiny, 0, math.safe_div(w, td))


def pdf_to_weight(t, p):
  """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
  utils.assert_valid_stepfun(t, p)
  return p * jnp.diff(t)


def max_dilate(t, w, dilation, domain=(-jnp.inf, jnp.inf)):
  """Dilate (via max-pooling) a non-negative step function."""
  utils.assert_valid_stepfun(t, w)
  t0 = t[Ellipsis, :-1] - dilation
  t1 = t[Ellipsis, 1:] + dilation
  t_dilate = jnp.sort(jnp.concatenate([t, t0, t1], axis=-1), axis=-1)
  t_dilate = jnp.clip(t_dilate, *domain)
  w_dilate = jnp.max(
      jnp.where(
          (t0[Ellipsis, None, :] <= t_dilate[Ellipsis, None])
          & (t1[Ellipsis, None, :] > t_dilate[Ellipsis, None]),
          w[Ellipsis, None, :],
          0,
      ),
      axis=-1,
  )[Ellipsis, :-1]
  return t_dilate, w_dilate


def max_dilate_weights(
    t,
    w,
    dilation,
    domain=(-jnp.inf, jnp.inf),
    renormalize=False,
    eps=jnp.finfo(jnp.float32).eps ** 2,
):
  """Dilate (via max-pooling) a set of weights."""
  utils.assert_valid_stepfun(t, w)
  p = weight_to_pdf(t, w)
  t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
  w_dilate = pdf_to_weight(t_dilate, p_dilate)
  if renormalize:
    w_dilate /= jnp.maximum(eps, jnp.sum(w_dilate, axis=-1, keepdims=True))
  return t_dilate, w_dilate


def integrate_weights(w):
  """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
  cw = jnp.minimum(1, jnp.cumsum(w[Ellipsis, :-1], axis=-1))
  shape = cw.shape[:-1] + (1,)
  # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
  cw0 = jnp.concatenate([jnp.zeros(shape), cw, jnp.ones(shape)], axis=-1)
  return cw0


def invert_cdf(u, t, w_logits):
  """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
  utils.assert_valid_stepfun(t, w_logits)
  # Compute the PDF and CDF for each weight vector.
  w = jax.nn.softmax(w_logits, axis=-1)
  cw = integrate_weights(w)
  # Interpolate into the inverse CDF.
  t_new = math.sorted_interp(u, cw, t, utils.device_is_tpu())
  return t_new


def sample(
    rng,
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    deterministic_center=False,
    eps=jnp.finfo(jnp.float32).eps,
):
  """Piecewise-Constant PDF sampling from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rng` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.
    eps: float, something like numerical epsilon.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  utils.assert_valid_stepfun(t, w_logits)

  # Draw uniform samples.
  if rng is None:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    if deterministic_center:
      pad = 1 / (2 * num_samples)
      u = jnp.linspace(pad, 1.0 - pad - eps, num_samples)
    else:
      u = jnp.linspace(0, 1.0 - eps, num_samples)
    u = jnp.broadcast_to(u, t.shape[:-1] + (num_samples,))
  else:
    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps
    d = 1 if single_jitter else num_samples
    u = jnp.linspace(0, 1 - u_max, num_samples) + jax.random.uniform(
        rng, t.shape[:-1] + (d,), maxval=max_jitter
    )

  return invert_cdf(u, t, w_logits)


def sample_intervals(
    rng,
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    domain=(-jnp.inf, jnp.inf),
):
  """Sample *intervals* (rather than points) from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of intervals to sample.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    domain: (minval, maxval), the range of valid values for `t`.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  utils.assert_valid_stepfun(t, w_logits)
  if num_samples <= 1:
    raise ValueError(f'num_samples must be > 1, is {num_samples}.')

  # Sample a set of points from the step function.
  centers = sample(
      rng, t, w_logits, num_samples, single_jitter, deterministic_center=True
  )

  # The intervals we return will span the midpoints of each adjacent sample.
  mid = (centers[Ellipsis, 1:] + centers[Ellipsis, :-1]) / 2

  # Each first/last fencepost is the reflection of the first/last midpoint
  # around the first/last sampled center.
  first = 2 * centers[Ellipsis, :1] - mid[Ellipsis, :1]
  last = 2 * centers[Ellipsis, -1:] - mid[Ellipsis, -1:]
  samples = jnp.concatenate([first, mid, last], axis=-1)

  # We clamp to the limits of the input domain, provided by the caller.
  samples = jnp.clip(samples, *domain)
  return samples


def lossfun_distortion(t, w, normalize=False):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  utils.assert_valid_stepfun(t, w)

  if normalize:
    w += jnp.finfo(jnp.float32).eps ** 2
    w /= jnp.sum(w, axis=-1, keepdims=True)

  # The loss incurred between all pairs of intervals.
  ut = (t[Ellipsis, 1:] + t[Ellipsis, :-1]) / 2
  dut = jnp.abs(ut[Ellipsis, :, None] - ut[Ellipsis, None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * jnp.diff(t), axis=-1) / 3

  return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
  """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
  # Distortion when the intervals do not overlap.
  d_disjoint = jnp.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

  # Distortion when the intervals overlap.
  d_overlap = (
      2 * (jnp.minimum(t0_hi, t1_hi) ** 3 - jnp.maximum(t0_lo, t1_lo) ** 3)
      + 3
      * (
          t1_hi * t0_hi * jnp.abs(t1_hi - t0_hi)
          + t1_lo * t0_lo * jnp.abs(t1_lo - t0_lo)
          + t1_hi * t0_lo * (t0_lo - t1_hi)
          + t1_lo * t0_hi * (t1_lo - t0_hi)
      )
  ) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

  # Are the two intervals not overlapping?
  are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

  return jnp.where(are_disjoint, d_disjoint, d_overlap)


def weighted_percentile(t, w, ps):
  """Compute the weighted percentiles of a step function. w's must sum to 1."""
  utils.assert_valid_stepfun(t, w)
  cw = integrate_weights(w)
  # We want to interpolate into the integrated weights according to `ps`.
  wprctile = jnp.vectorize(jnp.interp, signature='(n),(m),(m)->(n)')(
      jnp.array(ps) / 100, cw, t
  )
  return wprctile


def resample(t, tp, vp, use_avg=False, eps=jnp.finfo(jnp.float32).eps):
  """Resample a step function defined by (tp, vp) into intervals t.

  Notation roughly matches jnp.interp. Resamples by summation by default.

  Args:
    t: tensor with shape (..., n+1), the endpoints to resample into.
    tp: tensor with shape (..., m+1), the endpoints of the step function being
      resampled.
    vp: tensor with shape (..., m), the values of the step function being
      resampled.
    use_avg: bool, if False, return the sum of the step function for each
      interval in `t`. If True, return the average, weighted by the width of
      each interval in `t`.
    eps: float, a small value to prevent division by zero when use_avg=True.

  Returns:
    v: tensor with shape (..., n), the values of the resampled step function.
  """
  utils.assert_valid_stepfun(tp, vp)
  if use_avg:
    wp = jnp.diff(tp)
    v_numer = resample(t, tp, vp * wp, use_avg=False)
    v_denom = resample(t, tp, wp, use_avg=False)
    v = v_numer / jnp.maximum(eps, v_denom)
    return v

  acc = jnp.cumsum(vp, axis=-1)
  acc0 = jnp.concatenate([jnp.zeros(acc.shape[:-1] + (1,)), acc], axis=-1)
  acc0_resampled = jnp.vectorize(jnp.interp, signature='(n),(m),(m)->(n)')(
      t, tp, acc0
  )
  v = jnp.diff(acc0_resampled, axis=-1)
  return v


def resolve_collisions(t, y, reverse=False):
  """Resolve the weights of a step function with possibly-repeated t-values.

  We scan through (t, y) values, and when identical t-values are detected, we
  replace their y-values with the first/last y-value in each streak. By default
  this uses the first values in each streak, set reverse=True to use the last
  value.

  Args:
    t: tensor with shape (..., n+1), the step function endpoints.
    y: tensor with shape (..., n), the step function values.
    reverse: bool, if True, scan from from back to front.

  Returns:
    y: tensor with shape (..., n), a fixed version of `y`.
  """
  utils.assert_valid_stepfun(t, y)

  def scan_fn(last, x):
    is_same = last[Ellipsis, 0] == x[Ellipsis, 0]
    carry = jnp.where(is_same[Ellipsis, None], last, x)
    output = jnp.where(is_same, last[Ellipsis, 1], x[Ellipsis, 1])
    return carry, output

  # Stack t and y along axis=-1 (no need to worry about the last t value).
  ty = jnp.stack([t[Ellipsis, :-1], y], axis=-1)

  ty_perm = jnp.moveaxis(ty, -2, 0)

  # Initialize to the first/last bin depending on `reverse`.
  init_perm = ty_perm[-1 if reverse else 0]

  # Scan from front to back (reverse=False) or back to front (reverse=True).
  _, y_perm = jax.lax.scan(scan_fn, init_perm, ty_perm, reverse=reverse)

  # Undo the permutation that moved the step function dimension to the front.
  y_resolved = jnp.moveaxis(y_perm, 0, -1)

  return y_resolved


def convolve_with_weighted_diracs(t, w, diracs):
  """Convolve (t, w) with diracs = [(shift0, scale0), (shift1, scale1), ...]."""
  utils.assert_valid_stepfun(t, w)

  # The difference in adjacent `w` values (zero padded).
  dw = jnp.diff(
      jnp.concatenate(
          [jnp.zeros_like(w[Ellipsis, :1]), w, jnp.zeros_like(w[Ellipsis, :1])], axis=-1
      )
  )

  # Construct a set of shifted and scaled derivatives.
  tp, dwp = [
      jnp.concatenate(x, axis=-1)
      for x in zip(*[(t + dt, dw * w_mult) for (dt, w_mult) in diracs])
  ]

  # Sort the convolved t-values and their accompanying derivative weights.
  idx = jnp.argsort(tp, axis=-1)
  tp = jnp.take_along_axis(tp, idx, axis=-1)
  dwp = jnp.take_along_axis(dwp, idx[Ellipsis, :-1], axis=-1)

  # Integrate the derivatives back into a step function.
  wp = jnp.cumsum(dwp, axis=-1)

  # The convolved step function may be incorrect if there are multiple identical
  # values in `tp`. We resolve this by scanning backwards through each stepfun
  # and replacing values corresponding to each "streak" of matching `tp`` values
  # with the last value in `wp` of each streak (which is correct because `wp``
  # was generated by cumsum, and we want to use the sum of all colliding
  # wp-values).
  wp = resolve_collisions(tp, wp, reverse=True)

  return tp, wp


def generate_binomial_diracs(num_taps, num_sigmas=None):
  """Generates a set of weighted diracs that approximate a standard Gaussian."""
  if num_taps < 2 or num_taps > 64:  # Numerical issues crop up after 64.
    raise ValueError(f'num_taps must be in [2, 64], but is {num_taps}')

  # Construct a binomial filter with length `num_taps`.
  w = np.array([1], dtype=np.int64)
  for _ in range(num_taps - 1):
    w = np.convolve(w, np.array([1, 1], dtype=np.int64))

  # Make the filter sum to 1.
  w = w / 2 ** (num_taps - 1)

  # Space the t-values of the diracs such that variance = 1.
  t_max = np.sqrt(num_taps - 1)
  t = np.linspace(-t_max, t_max, num_taps)

  if num_sigmas is not None:
    # Throw out any diracs that are outside of the truncation range.
    keep = np.abs(t) <= num_sigmas
    t, w = t[keep], w[keep]
    w /= np.sum(w)

  return t, w


def blur_with_binomial_diracs(t, y, sigma, num_taps, num_sigmas=3):
  """Blurs stepfun (t, y) with a binomial/dirac approximation of a Gaussian."""
  utils.assert_valid_stepfun(t, y)
  t_diract, w_diract = generate_binomial_diracs(num_taps, num_sigmas=num_sigmas)
  return convolve_with_weighted_diracs(t, y, zip(t_diract * sigma, w_diract))


def blur_and_resample_weights(tq, t, w, blur_halfwidth):
  """Blur the (t, w) histogram by blur_halfwidth, then resample it into tq."""
  utils.assert_valid_stepfun(t, w)

  # Convert the histogram to a PDF.
  p = weight_to_pdf(t, w)

  # Blur the PDF step function into a piecewise linear spline PDF.
  t_linspline, p_linspline = linspline.blur_stepfun(t, p, blur_halfwidth)

  # Integrate the spline PDF, then query it to get integrated weights.
  quad = linspline.compute_integral(t_linspline, p_linspline)
  acc_wq = linspline.interpolate_integral(tq, t_linspline, *quad)

  # Undo the integration to get weights.
  wq = jnp.diff(acc_wq, axis=-1)

  # Fix negative values to 0, as they should never happen but may due to
  # numerical issues.
  wq = jnp.maximum(0, wq)
  return wq
