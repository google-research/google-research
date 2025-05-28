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

import chex
import jax
import jax.numpy as jnp
from smerf.internal import math
from smerf.internal import utils


def inner_outer(t0, t1, y1):
  """Construct inner and outer measures on (t1, y1) for t0."""
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
  _, w_outer = inner_outer(t, t_env, w_env)
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return jnp.maximum(0, w - w_outer) ** 2 / (w + eps)


def weight_to_pdf(t, w, eps=jnp.finfo(jnp.float32).eps ** 2):
  """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
  return w / jnp.maximum(eps, (t[Ellipsis, 1:] - t[Ellipsis, :-1]))


def pdf_to_weight(t, p):
  """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
  return p * (t[Ellipsis, 1:] - t[Ellipsis, :-1])


def max_dilate(t, w, dilation, domain=(-jnp.inf, jnp.inf)):
  """Dilate (via max-pooling) a non-negative step function."""
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

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = jnp.finfo(jnp.float32).eps

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
  if num_samples <= 1:
    raise ValueError(f'num_samples must be > 1, is {num_samples}.')

  # Sample a set of points from the step function.
  centers = sample(
      rng, t, w_logits, num_samples, single_jitter, deterministic_center=True
  )

  # The intervals we return will span the midpoints of each adjacent sample.
  mid = (centers[Ellipsis, 1:] + centers[Ellipsis, :-1]) / 2

  # Each first/last fencepost is the reflection of the first/last midpoint
  # around the first/last sampled center. We clamp to the limits of the input
  # domain, provided by the caller.
  minval, maxval = domain
  first = jnp.maximum(minval, 2 * centers[Ellipsis, :1] - mid[Ellipsis, :1])
  last = jnp.minimum(maxval, 2 * centers[Ellipsis, -1:] - mid[Ellipsis, -1:])

  t_samples = jnp.concatenate([first, mid, last], axis=-1)
  return t_samples


def lossfun_distortion(t, w, normalize=True):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  # The loss incurred between all pairs of intervals.

  if normalize:
    w += jnp.finfo(jnp.float32).eps ** 2
    w /= jnp.sum(w, axis=-1, keepdims=True)

  ut = (t[Ellipsis, 1:] + t[Ellipsis, :-1]) / 2
  dut = jnp.abs(ut[Ellipsis, :, None] - ut[Ellipsis, None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * (t[Ellipsis, 1:] - t[Ellipsis, :-1]), axis=-1) / 3

  return loss_inter + loss_intra


def interval_contains(a, b):
  """True if intervals a is a superset of intervals b.

  Args:
    a: f32[n,2]. Reference interval.
    b: f32[m,2]. Query intervals.

  Returns:
    f32[n,m]. True if intervals a completely contains intervals b.
  """
  n = a.shape[0]
  m = b.shape[0]
  chex.assert_shape(a, (n, 2))
  chex.assert_shape(b, (m, 2))

  def fn(reference):
    return (reference[0] <= b[:, 0]) & (b[:, 1] <= reference[1])

  result = jax.vmap(fn)(a)
  chex.assert_shape(result, (n, m))

  return result


def interval_intersects(a, b):
  """True if intervals a intersect intervals b.

  Args:
    a: f32[n,2]. Reference interval.
    b: f32[m,2]. Query intervals.

  Returns:
    f32[n,m]. True if intervals a has a non-trivial overlap with intervals b.
  """
  n = a.shape[0]
  m = b.shape[0]
  chex.assert_shape(a, (n, 2))
  chex.assert_shape(b, (m, 2))

  def fn(reference):
    # intervals intersect they have a non-trivial overlap. It is not enough
    # for the intervals to "touch" each other. A length-zero interval cannot
    # contain nor be contained by anything.
    return (reference[0] < b[:, 1]) & (b[:, 0] < reference[1])

  result = jax.vmap(fn)(a)
  chex.assert_shape(result, (n, m))

  return result


def fenceposts_to_intervals(x):
  """Converts [x0, x1, x2, ...] to [[x0, x1], [x1, x2], ...].

  Args:
    x: f32[..., k+1]. Fenceposts.

  Returns:
    f32[..., k, 2]. Start and end posts of each interval.
  """
  return jnp.stack([x[Ellipsis, :-1], x[Ellipsis, 1:]], axis=-1)


def intervals_contain(x, y):
  """True if intervals in 'y' are wholly contained by intervals in 'x'.

  Args:
    x: f32[..., n, 2]. Reference intervals.
    y: f32[..., m, 2]. Query intervals.

  Returns:
    f32[..., m, n]. True if intervals in y[..., m] is completely contained
      within interval x[..., n].
  """
  z = jnp.vectorize(interval_contains, signature='(n,k),(m,k)->(n,m)')(x, y)
  return z


def intervals_intersect(x, y):
  """True if intervals in 'y' intersect intervals in 'x'.

  Args:
    x: f32[..., n, 2]. Reference intervals.
    y: f32[..., m, 2]. Query intervals.

  Returns:
    f32[..., m, n]. True if intervals in y[..., m] intersects interval x[..., n]
  """
  z = jnp.vectorize(interval_intersects, signature='(n,k),(m,k)->(n,m)')(x, y)
  return z


def inner_outer2(tdist0, tdist1, weights1):
  """Computes lower and upper bounds for intervals tdist0.

  Intervals where start == end have weight=0.

  Args:
    tdist0: f32[..., m+1]. Query ray intervals.
    tdist1: f32[..., n+1]. Reference ray intervals.
    weights1: f32[..., n]. Weights assigned to reference ray intervals.

  Returns:
    weights0_lower: f32[..., m]. Lower bounds for intervals weights0.
    weights0_upper: f32[..., m]. Upper bounds for intervals weights1.
  """
  batch_dims = tdist0.shape[:-1]
  m = tdist0.shape[-1] - 1
  n = tdist1.shape[-1] - 1
  chex.assert_shape(tdist0, (*batch_dims, m + 1))
  chex.assert_shape(tdist1, (*batch_dims, n + 1))
  chex.assert_shape(weights1, (*batch_dims, n))

  query = fenceposts_to_intervals(tdist0)  # [..., n, 2]
  reference = fenceposts_to_intervals(tdist1)  # [..., m, 2]

  # Ensure that reference intervals where start=end have weight=0.
  weights1 = jnp.where(reference[Ellipsis, 0] == reference[Ellipsis, 1], 0, weights1)

  # The upper bound for a query interval's weight is the sum of weights
  # assigned to all reference intervals that intersect the query interval.
  upper = intervals_intersect(query, reference)  # [..., m, n]
  upper = jnp.sum(upper * weights1[Ellipsis, jnp.newaxis, :], axis=-1)  # [..., m]
  chex.assert_shape(upper, (*batch_dims, m))

  # The lower bound for a query interval's weight is the sum of weights
  # assigned to all reference intervals that are wholly contained within the
  # query interval.
  lower = intervals_contain(query, reference)  # [..., m, n]
  lower = jnp.sum(lower * weights1[Ellipsis, jnp.newaxis, :], axis=-1)  # [..., m]
  chex.assert_shape(lower, (*batch_dims, m))

  # Ensure that output intervals where start=end have loose weight bounds.
  is_empty_interval = query[Ellipsis, 0] == query[Ellipsis, 1]  # [..., m]
  max_weight = jnp.sum(weights1, axis=-1, keepdims=True)  # [..., 1]
  lower = jnp.where(is_empty_interval, 0.0, lower)
  upper = jnp.where(is_empty_interval, max_weight, upper)

  return lower, upper
