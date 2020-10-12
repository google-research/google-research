# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for cubic_spline.py."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax.random as random
from robust_loss_jax import cubic_spline


class CubicSplineTest(chex.TestCase):

  def _setup_toy_data(self, n=32768):
    x = jnp.float32(jnp.arange(n))
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    values = random.normal(key, shape=[n])
    rng, key = random.split(rng)
    tangents = random.normal(key, shape=[n])
    return x, values, tangents

  def _interpolate1d(self, x, values, tangents):
    """Compute interpolate1d(x, values, tangents) and its derivative.

    This is just a helper function around cubic_spline.interpolate1d() that
    computes a tensor of values and gradients.

    Args:
      x: A tensor of values to interpolate with.
      values: A tensor of knot values for the spline.
      tangents: A tensor of knot tangents for the spline.

    Returns:
      A tuple containing:
       (An tensor of interpolated values,
        A tensor of derivatives of interpolated values wrt `x`)

    Typical usage example:
      y, dy_dx = self._interpolate1d(x, values, tangents)
    """
    fn = self.variant(cubic_spline.interpolate1d)
    y = fn(x, values, tangents)
    dy_dx = jax.grad(lambda z: jnp.sum(fn(z, values, tangents)))(x)
    return y, dy_dx

  @chex.all_variants()
  def testInterpolationReproducesValuesAtKnots(self):
    """Check that interpolating at a knot produces the value at that knot."""
    x, values, tangents = self._setup_toy_data()
    y = self.variant(cubic_spline.interpolate1d)(x, values, tangents)
    chex.assert_tree_all_close(y, values, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testInterpolationReproducesTangentsAtKnots(self):
    """Check that the derivative at a knot produces the tangent at that knot."""
    x, values, tangents = self._setup_toy_data()
    _, dy_dx = self.variant(self._interpolate1d)(x, values, tangents)
    chex.assert_tree_all_close(dy_dx, tangents, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testZeroTangentMidpointValuesAndDerivativesAreCorrect(self):
    """Check that splines with zero tangents behave correctly at midpoints.

    Make a spline whose tangents are all zeros, and then verify that
    midpoints between each pair of knots have the mean value of their adjacent
    knots, and have a derivative that is 1.5x the difference between their
    adjacent knots.
    """
    # Make a spline with random values and all-zero tangents.
    _, values, _ = self._setup_toy_data()
    tangents = jnp.zeros_like(values)

    # Query n-1 points placed exactly in between each pair of knots.
    x = jnp.arange(len(values) - 1) + 0.5

    # Get the interpolated values and derivatives.
    y, dy_dx = self._interpolate1d(x, values, tangents)

    # Check that the interpolated values of all queries lies at the midpoint of
    # its surrounding knot values.
    y_true = (values[0:-1] + values[1:]) / 2.
    chex.assert_tree_all_close(y, y_true, atol=1e-5, rtol=1e-5)

    # Check that the derivative of all interpolated values is (fun fact!) 1.5x
    # the numerical difference between adjacent knot values.
    dy_dx_true = 1.5 * (values[1:] - values[0:-1])
    chex.assert_tree_all_close(dy_dx, dy_dx_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testZeroTangentIntermediateValuesAndDerivativesDoNotOvershoot(self):
    """Check that splines with zero tangents behave correctly between knots.

    Make a spline whose tangents are all zeros, and then verify that points
    between each knot lie in between the knot values, and have derivatives
    are between 0 and 1.5x the numerical difference between knot values
    (mathematically, 1.5x is the max derivative if the tangents are zero).
    """
    # Make a spline with all-zero tangents and random values.
    _, values, _ = self._setup_toy_data()
    tangents = jnp.zeros_like(values)

    # Query n-1 points placed somewhere randomly in between all adjacent knots.
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    x = jnp.arange(len(values) - 1) + random.uniform(
        key, shape=[len(values) - 1])

    # Get the interpolated values and derivatives.
    y, dy_dx = self._interpolate1d(x, values, tangents)

    # Check that the interpolated values of all queries lies between its
    # surrounding knot values.
    self.assertTrue(
        jnp.all(((values[0:-1] <= y) & (y <= values[1:]))
                | ((values[0:-1] >= y) & (y >= values[1:]))))

    # Check that all derivatives of interpolated values are between 0 and 1.5x
    # the numerical difference between adjacent knot values.
    max_dy_dx = (1.5 + 1e-3) * (values[1:] - values[0:-1])
    self.assertTrue(
        jnp.all(((0 <= dy_dx) & (dy_dx <= max_dy_dx))
                | ((0 >= dy_dx) & (dy_dx >= max_dy_dx))))

  @chex.all_variants()
  def testLinearRampsReproduceCorrectly(self):
    """Check that interpolating a ramp reproduces a ramp.

    Generate linear ramps, render them into splines, and then interpolate and
    extrapolate the splines and verify that they reproduce the ramp.
    """
    n = 256
    # Generate queries inside and outside the support of the spline.
    rng, key = random.split(random.PRNGKey(0))
    x = (random.uniform(key, shape=[1024]) * 2 - 0.5) * (n - 1)
    idx = jnp.float32(jnp.arange(n))
    fn = self.variant(cubic_spline.interpolate1d)
    for _ in range(8):
      rng, key = random.split(rng)
      slope = random.normal(key)
      rng, key = random.split(rng)
      bias = random.normal(key)
      values = slope * idx + bias
      tangents = jnp.ones_like(values) * slope
      y = fn(x, values, tangents)
      y_true = slope * x + bias
      chex.assert_tree_all_close(y, y_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testExtrapolationIsLinear(self):
    """Check that extrapolation is linear with respect to the endpoint knots.

    Generate random splines and query them outside of the support of the
    spline, and veify that extrapolation is linear with respect to the
    endpoint knots.
    """
    n = 256
    # Generate queries above and below the support of the spline.
    rng, key = random.split(random.PRNGKey(0))
    x_below = -(random.uniform(key, shape=[1024])) * (n - 1)
    rng, key = random.split(rng)
    x_above = (random.uniform(key, shape=[1024]) + 1.) * (n - 1)
    fn = self.variant(cubic_spline.interpolate1d)
    for _ in range(8):
      rng, key = random.split(rng)
      values = random.normal(key, shape=[n])
      rng, key = random.split(rng)
      tangents = random.normal(key, shape=[n])

      # Query the spline below its support and check that it's a linear ramp
      # with the slope and bias of the beginning of the spline.
      y_below = fn(x_below, values, tangents)
      y_below_true = tangents[0] * x_below + values[0]
      chex.assert_tree_all_close(y_below, y_below_true, atol=1e-5, rtol=1e-5)

      # Query the spline above its support and check that it's a linear ramp
      # with the slope and bias of the end of the spline.
      y_above = fn(x_above, values, tangents)
      y_above_true = tangents[-1] * (x_above - (n - 1)) + values[-1]
      chex.assert_tree_all_close(y_above, y_above_true, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
