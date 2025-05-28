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

"""Tools for manipulating step functions (piecewise-constant 1D functions).

We have a shared naming and dimension convention for these functions.
All input/output step functions are assumed to be aligned along the last axis.
`t` always indicates the x coordinates of the *endpoints* of a step function.
`y` indicates unconstrained values for the *bins* of a step function
`w` indicates bin weights that sum to <= 1. `p` indicates non-negative bin
values that *integrate* to <= 1.
"""

from internal import linspline
from internal import math
from internal import utils
import jax
import jax.numpy as jnp
import numpy as np


def query(tq, t, y, left=None, right=None):
  """Query step function (t, y) at locations tq. Edges repeat by default."""
  utils.assert_valid_stepfun(t, y)
  # Query the step function to recover the interval value.
  (i0, i1), ((yq, _),) = math.sorted_lookup(tq, t, (y,), utils.device_is_tpu())
  # Apply boundary conditions.
  left = y[Ellipsis, :1] if left is None else left
  right = y[Ellipsis, -1:] if right is None else right
  yq = math.select([(i1 == 0, left), (i0 == y.shape[-1], right)], yq)
  return yq


def weight_to_pdf(t, w):
  """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
  utils.assert_valid_stepfun(t, w)
  td = jnp.diff(t)
  return jnp.where(td < np.finfo(np.float32).tiny, 0, math.safe_div(w, td))


def pdf_to_weight(t, p):
  """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
  utils.assert_valid_stepfun(t, p)
  return p * jnp.diff(t)


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


def lossfun_distortion(t, w):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  utils.assert_valid_stepfun(t, w)

  # The loss incurred between all pairs of intervals.
  ut = (t[Ellipsis, 1:] + t[Ellipsis, :-1]) / 2
  dut = jnp.abs(ut[Ellipsis, :, None] - ut[Ellipsis, None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * jnp.diff(t), axis=-1) / 3

  return loss_inter + loss_intra


def weighted_percentile(t, w, ps):
  """Compute the weighted percentiles of a step function. w's must sum to 1."""
  utils.assert_valid_stepfun(t, w)
  cw = integrate_weights(w)
  # We want to interpolate into the integrated weights according to `ps`.
  wprctile = jnp.vectorize(jnp.interp, signature='(n),(m),(m)->(n)')(
      jnp.array(ps) / 100, cw, t
  )
  return wprctile


def resample(t, tp, vp, use_avg=False):
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

  Returns:
    v: tensor with shape (..., n), the values of the resampled step function.
  """
  utils.assert_valid_stepfun(tp, vp)
  if use_avg:
    wp = jnp.diff(tp)
    v_numer = resample(t, tp, vp * wp, use_avg=False)
    v_denom = resample(t, tp, wp, use_avg=False)
    v = math.safe_div(v_numer, v_denom)
    return v

  acc = jnp.cumsum(vp, axis=-1)
  acc0 = jnp.concatenate([jnp.zeros(acc.shape[:-1] + (1,)), acc], axis=-1)
  acc0_resampled = jnp.vectorize(jnp.interp, signature='(n),(m),(m)->(n)')(
      t, tp, acc0
  )
  v = jnp.diff(acc0_resampled, axis=-1)
  return v


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
