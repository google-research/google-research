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

"""Helper functions for linear splines."""

import functools

from internal import math
from internal import utils
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
  # TODO(barron): It might be faster to stack (a, c, b) during generation and
  # do a single gather.
  t0 = jnp.take_along_axis(t, idx0, axis=-1)
  a0 = jnp.take_along_axis(a, idx0, axis=-1)
  b0 = jnp.take_along_axis(b, idx0, axis=-1)
  c0 = jnp.take_along_axis(c, idx0, axis=-1)

  td = tq - t0
  v = a0 * td**2 + b0 * td + c0
  return v


def blur_stepfun(ts, ys, halfwidth):
  """Convolve a step function (ts, ys) with a box filter of size `halfwidth`."""

  utils.assert_valid_stepfun(ts, ys)

  # Blur each entire step function by a single `halfwidth` value.

  # Dilate the t-values by at least numerical epsilon in each direction.
  ts_lo = ts - halfwidth
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

  # A ramp is the double integral of a delta function, so if we double-
  # integrate these derivatives you get the sum of a bunch of trapezoids.
  yp = jnp.cumsum(jnp.diff(tp)[Ellipsis, :-1] * jnp.cumsum(dyp, axis=-1), axis=-1)

  # Add in the missing first and last endpoint values, which must be zero
  # because we assume zero padding on `ys`.
  yp = jnp.concatenate(
      [jnp.zeros_like(yp[Ellipsis, :1]), yp, jnp.zeros_like(yp[Ellipsis, -1:])], axis=-1
  )

  return tp, yp
