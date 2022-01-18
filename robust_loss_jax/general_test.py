# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for general.py."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import jax.random as random
from robust_loss_jax import general


class LossfunTest(chex.TestCase):

  def _precompute_lossfun_inputs(self):
    """Precompute a loss and its derivatives for random inputs and parameters.

    Generates a large number of random inputs to the loss, and random
    shape/scale parameters for the loss function at each sample, and
    computes the loss and its derivative with respect to all inputs and
    parameters, returning everything to be used to assert various properties
    in our unit tests.

    Returns:
      A tuple containing:
       (the number (int) of samples, and the length of all following arrays,
        A tensor of losses for each sample,
        A tensor of residuals of each sample (the loss inputs),
        A tensor of shape parameters of each loss,
        A tensor of scale parameters of each loss,
        A tensor of derivatives of each loss wrt each x,
        A tensor of derivatives of each loss wrt each alpha,
        A tensor of derivatives of each loss wrt each scale)

    Typical usage example:
    (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)
        = self._precompute_lossfun_inputs()
    """
    num_samples = 100000
    rng = random.PRNGKey(0)

    # Normally distributed inputs.
    rng, key = random.split(rng)
    x = random.normal(key, shape=[num_samples])

    # Uniformly distributed values in (-16, 3), quantized to the nearest 0.1
    # to ensure that we hit the special cases at 0, 2.
    rng, key = random.split(rng)
    alpha = jnp.round(
        random.uniform(key, shape=[num_samples], minval=-16, maxval=3) *
        10) / 10.
    # Push the sampled alphas at the extents of the range to +/- infinity, so
    # that we probe those cases too.
    alpha = jnp.where(alpha == 3, jnp.inf, alpha)
    alpha = jnp.where(alpha == -16, -jnp.inf, alpha)

    # Random log-normally distributed values in approx (1e-5, 100000):
    rng, key = random.split(rng)
    scale = jnp.exp(random.normal(key, shape=[num_samples]) * 4.) + 1e-5

    fn = self.variant(general.lossfun)
    loss = fn(x, alpha, scale)
    d_x, d_alpha, d_scale = (
        jax.grad(lambda x, a, s: jnp.sum(fn(x, a, s)), [0, 1, 2])(x, alpha,
                                                                  scale))

    return (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)

  @chex.all_variants()
  def testDerivativeIsMonotonicWrtX(self):
    # Check that the loss increases monotonically with |x|.
    _, _, x, alpha, _, d_x, _, _ = self._precompute_lossfun_inputs()
    # This is just to suppress a warning below.
    d_x = jnp.where(jnp.isfinite(d_x), d_x, jnp.zeros_like(d_x))
    mask = jnp.isfinite(alpha) & (
        jnp.abs(d_x) > (300. * jnp.finfo(jnp.float32).eps))
    chex.assert_tree_all_close(jnp.sign(d_x[mask]), jnp.sign(x[mask]))

  @chex.all_variants()
  def testLossIsNearZeroAtOrigin(self):
    # Check that the loss is near-zero when x is near-zero.
    _, loss, x, _, _, _, _, _ = self._precompute_lossfun_inputs()
    loss_near_zero = loss[jnp.abs(x) < 1e-5]
    chex.assert_tree_all_close(
        loss_near_zero, jnp.zeros_like(loss_near_zero), atol=1e-5)

  @chex.all_variants()
  def testLossIsQuadraticNearOrigin(self):
    # Check that the loss is well-approximated by a quadratic bowl when
    # |x| < scale
    _, loss, x, _, scale, _, _, _ = self._precompute_lossfun_inputs()
    mask = jnp.abs(x) < (0.5 * scale)
    loss_quad = 0.5 * jnp.square(x / scale)
    chex.assert_tree_all_close(
        loss_quad[mask], loss[mask], rtol=1e-5, atol=1e-2)

  @chex.all_variants()
  def testLossIsBoundedWhenAlphaIsNegative(self):
    # Assert that loss < (alpha - 2)/alpha when alpha < 0.
    _, loss, _, alpha, _, _, _, _ = self._precompute_lossfun_inputs()
    mask = alpha < 0.
    min_val = jnp.finfo(jnp.float32).min
    alpha_clipped = jnp.maximum(min_val, alpha[mask])
    self.assertTrue(
        jnp.all(loss[mask] <= ((alpha_clipped - 2.) / alpha_clipped)))

  @chex.all_variants()
  def testDerivativeIsBoundedWhenAlphaIsBelow2(self):
    # Assert that |d_x| < |x|/scale^2 when alpha <= 2.
    _, _, x, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs()
    mask = jnp.isfinite(alpha) & (alpha <= 2)
    grad = jnp.abs(d_x[mask])
    bound = ((jnp.abs(x[mask]) + (300. * jnp.finfo(jnp.float32).eps)) /
             scale[mask]**2)
    self.assertTrue(jnp.all(grad <= bound))

  @chex.all_variants()
  def testDerivativeIsBoundedWhenAlphaIsBelow1(self):
    # Assert that |d_x| < 1/scale when alpha <= 1.
    _, _, _, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs()
    mask = jnp.isfinite(alpha) & (alpha <= 1)
    grad = jnp.abs(d_x[mask])
    bound = ((1. + (300. * jnp.finfo(jnp.float32).eps)) / scale[mask])
    self.assertTrue(jnp.all(grad <= bound))

  @chex.all_variants()
  def testAlphaDerivativeIsPositive(self):
    # Assert that d_loss / d_alpha > 0.
    _, _, _, alpha, _, _, d_alpha, _ = self._precompute_lossfun_inputs()
    mask = jnp.isfinite(alpha)
    self.assertTrue(
        jnp.all(d_alpha[mask] > (-300. * jnp.finfo(jnp.float32).eps)))

  @chex.all_variants()
  def testScaleDerivativeIsNegative(self):
    # Assert that d_loss / d_scale < 0.
    _, _, _, alpha, _, _, _, d_scale = self._precompute_lossfun_inputs()
    mask = jnp.isfinite(alpha)
    self.assertTrue(
        jnp.all(d_scale[mask] < (300. * jnp.finfo(jnp.float32).eps)))

  @chex.all_variants()
  def testLossIsScaleInvariant(self):
    # Check that loss(mult * x, alpha, mult * scale) == loss(x, alpha, scale)
    (num_samples, loss, x, alpha, scale, _, _, _) = (
        self._precompute_lossfun_inputs())
    # Random log-normally distributed scalings in ~(0.2, 20)

    rng = random.PRNGKey(0)
    mult = jnp.maximum(0.2, jnp.exp(random.normal(rng, shape=[num_samples])))

    # Compute the scaled loss.
    loss_scaled = general.lossfun(mult * x, alpha, mult * scale)
    chex.assert_tree_all_close(loss, loss_scaled, atol=1e-4, rtol=1e-4)

  @chex.all_variants()
  def testAlphaEqualsNegativeInfinity(self):
    # Check that alpha == -Infinity reproduces Welsch aka Leclerc loss.
    x = jnp.linspace(-20, 20, 1000)
    alpha = -float('inf')
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # Welsch/Leclerc loss.
    loss_true = (1. - jnp.exp(-0.5 * jnp.square(x / scale)))

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsNegativeTwo(self):
    # Check that alpha == -2 reproduces Geman-McClure loss.
    x = jnp.linspace(-20, 20, 1000)
    alpha = -2.
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # Geman-McClure loss.
    loss_true = (2. * jnp.square(x / scale) / (jnp.square(x / scale) + 4.))

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsZero(self):
    # Check that alpha == 0 reproduces Cauchy aka Lorentzian loss.
    x = jnp.linspace(-20, 20, 1000)
    alpha = 0.
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # Cauchy/Lorentzian loss.
    loss_true = (jnp.log(0.5 * jnp.square(x / scale) + 1.))

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsOne(self):
    # Check that alpha == 1 reproduces Charbonnier aka pseudo-Huber loss.
    x = jnp.linspace(-20, 20, 1000)
    alpha = 1.
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # Charbonnier loss.
    loss_true = (jnp.sqrt(jnp.square(x / scale) + 1) - 1)

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsTwo(self):
    # Check that alpha == 2 reproduces L2 loss.
    x = jnp.linspace(-20, 20, 1000)
    alpha = 2.
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # L2 Loss.
    loss_true = (0.5 * jnp.square(x / scale))

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsFour(self):
    # Check that alpha == 4 reproduces a quartic.
    x = jnp.linspace(-20, 20, 1000)
    alpha = 4.
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # The true loss.
    loss_true = (
        jnp.square(jnp.square(x / scale)) / 8. + jnp.square(x / scale) / 2.)

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testAlphaEqualsInfinity(self):
    # Check that alpha == Infinity takes the correct form.
    x = jnp.linspace(-20, 20, 1000)
    alpha = float('inf')
    scale = 1.7

    # Our loss.
    loss = self.variant(general.lossfun)(x, alpha, scale)

    # The true loss.
    loss_true = (jnp.exp(0.5 * jnp.square(x / scale)) - 1.)

    chex.assert_tree_all_close(loss, loss_true, atol=1e-5, rtol=1e-5)

  @chex.all_variants()
  def testLossAndGradientsAreFinite(self):
    # Test that the loss and its approximation both give finite losses and
    # derivatives everywhere that they should for a wide range of values.
    num_samples = 100000
    rng = random.PRNGKey(0)

    # Normally distributed inputs.
    rng, key = random.split(rng)
    x = random.normal(key, shape=[num_samples])

    # Uniformly distributed values in (-16, 3), quantized to the nearest 0.1
    # to ensure that we hit the special cases at 0, 2.
    rng, key = random.split(rng)
    alpha = jnp.round(
        random.uniform(key, shape=[num_samples], minval=-16, maxval=3) *
        10) / 10.

    # Random log-normally distributed values in approx (1e-5, 100000):
    rng, key = random.split(rng)
    scale = jnp.exp(random.normal(key, shape=[num_samples]) * 4.) + 1e-5

    fn = self.variant(general.lossfun)
    loss = fn(x, alpha, scale)
    d_x, d_alpha, d_scale = (
        jax.grad(lambda x, a, s: jnp.sum(fn(x, a, s)), [0, 1, 2])(x, alpha,
                                                                  scale))

    for v in [loss, d_x, d_alpha, d_scale]:
      chex.assert_tree_all_finite(v)

  @chex.all_variants()
  def testGradientMatchesFiniteDifferences(self):
    # Test that the loss and its approximation both return gradients that are
    # close to the numerical gradient from finite differences, with forward
    # differencing. Returning correct gradients is JAX's job, so this is
    # just an aggressive sanity check.
    num_samples = 100000
    rng = random.PRNGKey(0)

    # Normally distributed inputs.
    rng, key = random.split(rng)
    x = random.normal(key, shape=[num_samples])

    # Uniformly distributed values in (-16, 3), quantized to the nearest
    # 0.1 and then shifted by 0.05 so that we avoid the special cases at
    # 0 and 2 where the analytical gradient wont match finite differences.
    rng, key = random.split(rng)
    alpha = jnp.round(
        random.uniform(key, shape=[num_samples], minval=-16, maxval=3) *
        10) / 10. + 0.05

    # Random log-normally distributed values in approx (1e-5, 100000):
    rng, key = random.split(rng)
    scale = random.uniform(key, shape=[num_samples], minval=0.5, maxval=1.5)

    loss = general.lossfun(x, alpha, scale)
    d_x, d_alpha, d_scale = (
        jax.grad(lambda x, a, s: jnp.sum(general.lossfun(x, a, s)),
                 [0, 1, 2])(x, alpha, scale))

    step_size = 1e-3
    fn = self.variant(general.lossfun)
    n_x = (fn(x + step_size, alpha, scale) - loss) / step_size
    n_alpha = (fn(x, alpha + step_size, scale) - loss) / step_size
    n_scale = (fn(x, alpha, scale + step_size) - loss) / step_size

    chex.assert_tree_all_close(n_x, d_x, atol=1e-2, rtol=1e-2)
    chex.assert_tree_all_close(n_alpha, d_alpha, atol=1e-2, rtol=1e-2)
    chex.assert_tree_all_close(n_scale, d_scale, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
