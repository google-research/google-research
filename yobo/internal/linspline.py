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
"""Helper functions for linear splines."""

import functools

from google_research.yobo.internal import math
from google_research.yobo.internal import utils
import jax
from jax.experimental import checkify
import jax.numpy as jnp


def check_zero_endpoints(y):
  checkify.check(jnp.all(y[Ellipsis, 0] == 0), 'Splines must all start with 0.')
  checkify.check(jnp.all(y[Ellipsis, -1] == 0), 'Splines must all end with 0.')


def query(tq, t, v):
  """Query linear spline (t, v) at tq."""
  utils.assert_valid_linspline(t, v)
  interp = functools.partial(jnp.interp, left=0, right=0)
  return jnp.vectorize(interp, signature='(n),(m),(m)->(n)')(tq, t, v)


def integrate(t, w):
  """Integrate (t, w) according to the trapezoid rule."""
  utils.assert_valid_linspline(t, w)
  return 0.5 * jnp.sum((w[Ellipsis, :-1] + w[Ellipsis, 1:]) * jnp.diff(t), axis=-1)


def normalize(t, w, eps=jnp.finfo(jnp.float32).eps ** 2):
  """Make w integrate to 1."""
  utils.assert_valid_linspline(t, w)
  return w / jnp.maximum(eps, integrate(t, w))[Ellipsis, None]


def insert_knot(ti, t, y):
  """Inserts knots ti into the linear spline (t, w). Assumes zero endpoints."""
  utils.assert_valid_linspline(t, y)
  check_zero_endpoints(y)

  # Compute the spline value at the insertion points.
  yi = query(ti, t, y)

  # Concatenate the insertion points and values onto the end of each spline.
  ti_ex = jnp.broadcast_to(ti, t.shape[: -len(ti.shape)] + ti.shape)
  yi_ex = jnp.broadcast_to(yi, y.shape[: -len(yi.shape)] + yi.shape)
  to = jnp.concatenate([t, ti_ex], axis=-1)
  yo = jnp.concatenate([y, yi_ex], axis=-1)

  # Sort the spline according to t.
  sort_idx = jnp.argsort(to)
  to = jnp.take_along_axis(to, sort_idx, axis=-1)
  yo = jnp.take_along_axis(yo, sort_idx, axis=-1)
  return to, yo


def clamp(t, y, minval, maxval):
  """Clamp (t, y) to be zero outside of t in [minval, maxval]."""
  utils.assert_valid_linspline(t, y)
  check_zero_endpoints(y)

  # Add in extra points at and immediately above/below the min/max vals.
  ti = jnp.concatenate(
      [
          math.minus_eps(minval),
          minval,
          maxval,
          math.plus_eps(maxval),
      ],
      axis=-1,
  )
  tc, yo = insert_knot(ti, t, y)

  # Zero the spline values outside of [minval, maxval].
  yc = jnp.where(tc > maxval, 0, jnp.where(tc < minval, 0, yo))
  return tc, yc


def compute_integral(t, y):
  """Integrate a linear spline into a piecewise quadratic spline."""
  utils.assert_valid_linspline(t, y)
  eps = jnp.finfo(jnp.float32).eps ** 2
  dt = jnp.diff(t)
  a = jnp.diff(y) / jnp.maximum(eps, 2 * dt)
  b = y[Ellipsis, :-1]
  # The integral has an ambiguous global offset here, which we set to 0.
  c1 = 0.5 * jnp.cumsum(dt[Ellipsis, :-1] * (y[Ellipsis, :-2] + y[Ellipsis, 1:-1]), axis=-1)
  c = jnp.concatenate([jnp.zeros_like(y[Ellipsis, :1]), c1], axis=-1)
  # This quadratic is parameterized as:
  #   (t - t[i])**2 * a[i] + (t - t[i]) * b[i] + c[i]
  return a, b, c


def sorted_lookup(x, xp):
  """Lookup `x` at sorted locations `xp`."""
  # jnp.searchsorted() has slightly different conventions for boundary
  # handling than the rest of this codebase.
  idx = jnp.vectorize(
      functools.partial(jnp.searchsorted, side='right'),
      signature='(n),(m)->(m)',
  )(xp, x)
  idx0 = jnp.maximum(idx - 1, 0)
  idx1 = jnp.minimum(idx, xp.shape[-1] - 1)
  return idx0, idx1


def interpolate_integral(tq, t, a, b, c):
  """Interpolate into the piecewise quadratic returned by compute_integral()."""
  utils.assert_valid_stepfun(t, a)
  utils.assert_valid_stepfun(t, b)
  utils.assert_valid_stepfun(t, c)

  # Clip to valid inputs (assumes repeating boundaries).
  tq = jnp.clip(tq, t[Ellipsis, :1], math.minus_eps(t[Ellipsis, -1:]))

  # Lookup the quadratic coefficients corresponding to each input query.
  idx0, _ = sorted_lookup(tq, t)
  t0 = jnp.take_along_axis(t, idx0, axis=-1)
  a0 = jnp.take_along_axis(a, idx0, axis=-1)
  b0 = jnp.take_along_axis(b, idx0, axis=-1)
  c0 = jnp.take_along_axis(c, idx0, axis=-1)

  td = tq - t0
  v = a0 * td**2 + b0 * td + c0
  return v


def interpolate_inverse_integral(uq, t, a, b, c):
  """Interpolate into the inverse of a piecewise quadratic."""
  utils.assert_valid_stepfun(t, a)
  utils.assert_valid_stepfun(t, b)
  utils.assert_valid_stepfun(t, c)

  # Interpolate into the piecewise quadratic at the knots of the spline, to use
  # as the knots of the inverse spline. Note that t[...,:-1] == u.
  u = interpolate_integral(t, t, a, b, c)

  # Use those knots to clip to valid inputs (assumes repeating boundaries).
  uq = jnp.clip(uq, u[Ellipsis, :1], math.minus_eps(u[Ellipsis, -1:]))

  # Lookup the quadratic coefficients corresponding to each input query.
  idx0, idx1 = sorted_lookup(uq, u)
  t0 = jnp.take_along_axis(t, idx0, axis=-1)
  a0 = jnp.take_along_axis(a, idx0, axis=-1)
  b0 = jnp.take_along_axis(b, idx0, axis=-1)
  c0 = jnp.take_along_axis(c, idx0, axis=-1)

  # Evaluate the inverse of each quadratic. This is done by using just the
  # positive half of the quadratic equation after converting to standard form.
  eps = jnp.finfo(jnp.float32).eps ** 2
  a0_safe = jnp.where(a0 >= 0, 1, -1) * jnp.maximum(eps, jnp.abs(a0))
  z = b0**2 - 4 * a0 * (c0 - uq)
  v_quadratic = t0 + (jnp.sqrt(jnp.maximum(eps, z)) - b0) / (2 * a0_safe)

  # If the quadratic is flat, evaluate the inverse of the linear system.
  d0 = b0 - 2 * a0 * t0
  d0_safe = jnp.where(d0 >= 0, 1, -1) * jnp.maximum(eps, jnp.abs(d0))
  v_linear = t0 + (a0 * t0**2 - (c0 - uq)) / d0_safe

  use_linear = (jnp.abs(a0) <= eps) | (z <= eps)
  v = jnp.where(use_linear, v_linear, v_quadratic)

  # clip v to the values of the previous and next knot, which we may otherwise
  # exceed due to numerical instability.
  t0 = jnp.take_along_axis(t, idx0, axis=-1)
  t1 = jnp.take_along_axis(t, idx1, axis=-1)
  v = jnp.clip(v, jnp.minimum(t0, t1), jnp.maximum(t0, t1))
  return v


def blur_stepfun(ts, ys, halfwidth):
  """Convolve a step function (ts, ys) with a box filter of size `halfwidth`."""
  utils.assert_valid_stepfun(ts, ys)

  # Dilate the t-values by at least numerical epsilon in each direction.
  ts_lo = jnp.minimum(math.minus_eps(ts), ts - halfwidth)
  ts_hi = jnp.maximum(math.plus_eps(ts), ts + halfwidth)

  # The difference in adjacent `y` values (zero padded) divided by the
  # difference in adjacent `t` values.
  ys0 = jnp.concatenate(
      [jnp.zeros_like(ys[Ellipsis, :1]), ys, jnp.zeros_like(ys[Ellipsis, :1])], axis=-1
  )
  dy = jnp.diff(ys0) / (ts_hi - ts_lo)

  # When decreasing t splat a positive second derivative, and when increasing
  # t splat a negative second derivative.
  tp = jnp.concatenate([ts_lo, ts_hi], axis=-1)
  dyp = jnp.concatenate([dy, -dy], axis=-1)

  # Sort the dilated t-values and their accompanying derivative weights.
  idx = jnp.argsort(tp, axis=-1)
  tp = jnp.take_along_axis(tp, idx, axis=-1)
  dyp = jnp.take_along_axis(dyp, idx[Ellipsis, :-2], axis=-1)

  # A ramp is the double integral of a delta function, so if we double-integrate
  # these derivatives you get the sum of a bunch of trapezoids.
  yp = jnp.cumsum(jnp.diff(tp)[Ellipsis, :-1] * jnp.cumsum(dyp, axis=-1), axis=-1)

  # Add in the missing first and last endpoint values, which must be zero
  # because we assume zero padding on `ys`.
  yp = jnp.concatenate(
      [jnp.zeros_like(yp[Ellipsis, :1]), yp, jnp.zeros_like(yp[Ellipsis, -1:])], axis=-1
  )
  return tp, yp


def sample(
    rng,
    t,
    w,
    num_samples,
    single_jitter=False,
    deterministic_center=False,
    eps=jnp.finfo(jnp.float32).eps,
):
  """Draw samples from a piecewise linear PDF."""
  utils.assert_valid_linspline(t, w)

  # Normalize the input spline to be a normalized PDF.
  w = normalize(t, w)

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

  quad = compute_integral(t, w)
  samples = interpolate_inverse_integral(u, t, *quad)
  return samples


def merge(t0, y0, t1, y1, insert_intersections=True):
  """Merge two splines that start and end with 0 into a common set of knots."""
  utils.assert_valid_linspline(t0, y0)
  utils.assert_valid_linspline(t1, y1)
  check_zero_endpoints(y0)
  check_zero_endpoints(y1)

  tm = jnp.sort(jnp.concatenate([t0, t1], axis=-1), axis=-1)
  ym0 = query(tm, t0, y0)
  ym1 = query(tm, t1, y1)

  if insert_intersections:
    eps = jnp.finfo(jnp.float32).eps ** 2
    tmd = jnp.diff(tm)
    tmd0 = tmd <= eps
    slope0 = jnp.where(tmd0, 0, jnp.diff(ym0) / jnp.where(tmd0, 1, tmd))
    slope1 = jnp.where(tmd0, 0, jnp.diff(ym1) / jnp.where(tmd0, 1, tmd))
    dslope = slope0 - slope1
    dslope0 = jnp.abs(dslope) <= eps
    shift = (ym1[Ellipsis, :-1] - ym0[Ellipsis, :-1]) / jnp.where(dslope0, 1, dslope)
    invalid_shift = (shift <= 0) | (shift >= tmd) | dslope0 | tmd0
    shift = jnp.where(invalid_shift, 0, shift)
    t_int = shift + tm[Ellipsis, :-1]

    # This is marginally wasteful because 1) we compute t_int twice, and 2) we
    # have already done most of the math required to interpolate into (tm, ym*)
    # at t_int.
    _, ym0 = insert_knot(t_int, tm, ym0)  # Don't clobber `tm` yet, we need it.
    tm, ym1 = insert_knot(t_int, tm, ym1)

  return tm, ym0, ym1


def excess_area(t_lo, y_lo, t_hi, y_hi):
  """The amount of area by which (t_lo, y_lo) exceeds (t_hi, y_hi)."""
  utils.assert_valid_linspline(t_lo, y_lo)
  utils.assert_valid_linspline(t_hi, y_hi)
  check_zero_endpoints(y_lo)
  check_zero_endpoints(y_hi)

  tm, ym_lo, ym_hi = merge(t_lo, y_lo, t_hi, y_hi)

  # The trapezoid rule, with some reordering to avoid catastrophic cancellation.
  delta = jnp.maximum(
      0, (ym_lo[Ellipsis, 1:] - ym_hi[Ellipsis, 1:]) + (ym_lo[Ellipsis, :-1] - ym_hi[Ellipsis, :-1])
  )
  return 0.5 * jnp.sum(jnp.diff(tm, axis=-1) * delta, axis=-1)
