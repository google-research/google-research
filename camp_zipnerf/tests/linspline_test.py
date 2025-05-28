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

"""Unit tests for linspline."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import linspline
from internal import math
from internal import utils
import jax
from jax import random
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np

# Some of the tests here will fail on GPU, because we're comparing against
# Numpy which runs on a CPU.
jax.config.update('jax_platform_name', 'cpu')


def zero_endpoints(w):
  """Replace w[..., 0] and w[..., -1] with zeros."""
  return w * jnp.concatenate(
      [jnp.array([0.0]), jnp.ones(w.shape[-1] - 2), jnp.array([0])]
  )


def query_stepfun(tq, t, y, outside_value=0):
  """Look up the values of the step function (t, y) at locations tq."""
  idx_lo, idx_hi = math.searchsorted(t, tq, utils.device_is_tpu())
  yq = jnp.where(
      idx_lo == idx_hi, outside_value, jnp.take_along_axis(y, idx_lo, axis=-1)
  )
  return yq


class LinsplineTest(chex.TestCase, parameterized.TestCase):

  def wrap_fn(self, fn):
    return lambda *args: checkify.checkify(self.variant(fn))(*args)[1]

  @chex.all_variants()
  def test_insert_knot(self):
    n, d = 100, 8

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.sort(random.normal(key, shape=[n, d]), axis=-1)
    key, rng = random.split(rng)
    y = random.normal(key, shape=[n, d])
    key, rng = random.split(rng)
    ti = random.normal(key, shape=[n])
    y = zero_endpoints(y)  # insert_knot() assumes zero endpoints.

    to, yo = self.wrap_fn(linspline.insert_knot)(ti[:, None], t, y)

    # Test that each inserted knot exists in the output spline.
    np.testing.assert_equal(
        bool(jnp.all(jnp.any(jnp.abs(to - ti[:, None]) == 0, axis=-1))), True
    )

    # Test that the post-insertion spline describes the same 1D function.
    tq = jnp.linspace(-5, 5, 10000)
    np.testing.assert_allclose(
        linspline.query(tq, t, y), linspline.query(tq, to, yo), atol=1e-5
    )

  @chex.all_variants()
  def test_clamp(self):
    n, d = 100, 8

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.sort(random.normal(key, shape=[n, d]), axis=-1)
    key, rng = random.split(rng)
    y = random.normal(key, shape=[n, d])
    key, rng = random.split(rng)
    minval, maxval = tuple(jnp.sort(random.normal(key, shape=[2, n]), axis=0))
    y = zero_endpoints(y)  # clamp() assumes zero enpoints.

    fn = self.wrap_fn(linspline.clamp)
    tc, yc = fn(t, y, minval[Ellipsis, None], maxval[Ellipsis, None])

    # Query the clamped and unclamped splines at a lot of locations.
    tq = jnp.linspace(-5, 5, 1000)
    yq = linspline.query(tq, t, y)
    yq_clamped = linspline.query(tq, tc, yc)

    # Check that values inside the clamped range are the same as before, and
    # values outside of the clamped range are zero.
    mask = (tq[None, :] >= minval[:, None]) & (tq[None, :] <= maxval[:, None])
    np.testing.assert_allclose(yq_clamped, jnp.where(mask, yq, 0), atol=1e-5)

  @chex.all_variants()
  def test_normalize(self):
    """Test that normalize() gives splines that integrate to 1."""
    n, d = 100, 8

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.cumsum(jnp.exp(random.normal(key, shape=[n, d])), axis=-1)
    key, rng = random.split(rng)
    w = jnp.exp(random.normal(key, shape=[n, d]) - 1)

    integral = self.wrap_fn(linspline.integrate)(t, linspline.normalize(t, w))

    np.testing.assert_allclose(integral, 1.0, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def test_compute_and_interpolate_integral(self):
    n, d = 100, 8

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = jnp.cumsum(0.1 + random.uniform(key, shape=[n, d]), axis=-1)
    key, rng = random.split(rng)
    y = random.normal(key, shape=[n, d])

    # Flatten out some splines to probe some cases where the slope is 0.
    y = np.array(y)
    y[-5, :] = 1
    y[-4, :3] = 1
    y[-3, -3:] = 1
    y[-2, 1:4] = 1
    y[-1, 3:5] = 1

    # Linspace some sorted query points.
    tq = jnp.linspace(jnp.min(t) - 1, jnp.max(t) + 1, 10000)

    # Interpolate into our integral at the query points.
    quad = self.wrap_fn(linspline.compute_integral)(t, y)
    int_ours = self.wrap_fn(linspline.interpolate_integral)(tq, t, *quad)

    # Query the spline and compute the cumulative sum.
    int_ref = (tq[1] - tq[0]) * np.cumsum(
        np.float64(linspline.query(tq, t, y)), axis=-1
    )

    np.testing.assert_allclose(int_ours, int_ref, atol=0.003)

  @chex.all_variants()
  def test_compute_integral_with_repeated_knots(self):
    """Test that integration doesn't nan-out when knots are repeated."""
    t = jnp.array([1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8])
    a, b, c = self.wrap_fn(linspline.compute_integral)(t, t)
    np.testing.assert_equal(bool(jnp.all(jnp.isfinite(a))), True)
    np.testing.assert_equal(bool(jnp.all(jnp.isfinite(b))), True)
    np.testing.assert_equal(bool(jnp.all(jnp.isfinite(c))), True)

  @chex.all_variants()
  @parameterized.product(is_scalar=[False, True])
  def test_blur_stepfun_matches_convolution(self, is_scalar):
    n, d = 10, 8
    radius = 0.7

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    ts = jnp.cumsum(jnp.exp(random.normal(key, shape=[n, d + 1])), axis=-1)
    key, rng = random.split(rng)
    ws = jnp.exp(random.normal(key, shape=[n, d]) - 1)

    x = jnp.linspace(jnp.min(ts) - 3, jnp.max(ts) + 3, 1000)

    # Rasterize the spline and convolve that with a discretized box filter.
    y = query_stepfun(x, ts, ws)
    hw = radius / (x[1] - x[0])
    filt = jnp.arange(-jnp.ceil(hw), jnp.ceil(hw) + 1) <= hw
    filt /= jnp.sum(filt)
    y_blur_ref = jax.vmap(lambda x: jnp.convolve(x, filt, 'same'))(y)

    tp, wp = self.wrap_fn(linspline.blur_stepfun)(ts, ws, radius)
    y_blur_ours = linspline.query(x, tp, wp)

    ps = jnp.array([50, 90, 95, 99, 100])
    # Using a really high absolute tolerance for the max error, because the
    # cumsum() in blur_stepfun() really hurts numerical precision.
    tols = jnp.array([1e-5, 0.01, 0.05, 0.1, 0.5])
    errs = jnp.percentile(jnp.abs(y_blur_ref - y_blur_ours), ps)
    np.testing.assert_array_less(errs, tols)

  @chex.all_variants()
  @parameterized.product(is_scalar=[False, True])
  def test_blur_stepfun_and_integrate(self, is_scalar):
    """Blurring a step function should preserve its integral."""
    n, d = 100, 8
    radius = 0.7

    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    ts = jnp.cumsum(jnp.exp(random.normal(key, shape=[n, d + 1])), axis=-1)
    key, rng = random.split(rng)
    ws = jnp.exp(random.normal(key, shape=[n, d]) - 1)

    tp, wp = self.wrap_fn(linspline.blur_stepfun)(ts, ws, radius)

    # The area under the step function.
    integral_ref = jnp.sum(ws * (ts[Ellipsis, 1:] - ts[Ellipsis, :-1]), axis=-1)

    # The area under the linear spline.
    integral_ours = self.wrap_fn(linspline.integrate)(tp, wp)

    np.testing.assert_allclose(
        integral_ours, integral_ref, atol=1e-5, rtol=1e-5
    )

  @chex.all_variants()
  @parameterized.product(
      hw=[0.0, 1e-50, 1e-30, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
      is_scalar=[False, True],
  )
  def test_blur_toy_stepfun(self, hw, is_scalar):
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0])
    tp_true = np.array(
        [-hw, hw, 1 - hw, 1 + hw, 2 - hw, 2 + hw, 3 - hw, 3 + hw]
    )
    yp_true = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    tt = jnp.linspace(-1, 5, 10001)

    tp, yp = self.wrap_fn(
        functools.partial(linspline.blur_stepfun, halfwidth=hw)
    )(t, y)

    # TODO(barron): Tighten the tolerance for the vectorized case.
    np.testing.assert_allclose(
        linspline.query(tt, tp, yp),
        linspline.query(tt, tp_true, yp_true),
        atol=1e-6 if is_scalar else 1e-3,
    )


if __name__ == '__main__':
  absltest.main()
