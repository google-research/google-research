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

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from robust_loss import general

tf.enable_v2_behavior()


class LossfunTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(LossfunTest, self).setUp()
    np.random.seed(0)

  def _assert_all_close_according_to_type(self, a, b):
    """AssertAllClose() with tighter thresholds for float64 than float32."""
    self.assertAllCloseAccordingToType(
        a, b, rtol=1e-15, atol=1e-15, float_rtol=1e-6, float_atol=1e-6)

  def _precompute_lossfun_inputs(self, float_dtype):
    """Precompute a loss and its derivatives for random inputs and parameters.

    Generates a large number of random inputs to the loss, and random
    shape/scale parameters for the loss function at each sample, and
    computes the loss and its derivative with respect to all inputs and
    parameters, returning everything to be used to assert various properties
    in our unit tests.

    Args:
      float_dtype: The float precision to be used (np.float32 or np.float64).

    Returns:
      A tuple containing:
       (the number (int) of samples, and the length of all following arrays,
        A np.array (float_dtype) of losses for each sample,
        A np.array (float_dtype) of residuals of each sample (the loss inputs),
        A np array (float_dtype) of shape parameters of each loss,
        A np.array (float_dtype) of scale parameters of each loss,
        A np.array (float_dtype) of derivatives of each loss wrt each x,
        A np.array (float_dtype) of derivatives of each loss wrt each alpha,
        A np.array (float_dtype) of derivatives of each loss wrt each scale)

    Typical usage example:
    (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)
        = self._precompute_lossfun_inputs(np.float32)
    """
    num_samples = 100000
    # Normally distributed inputs.
    x = float_dtype(np.random.normal(size=num_samples))

    # Uniformly distributed values in (-16, 3), quantized to the nearest 0.1
    # to ensure that we hit the special cases at 0, 2.
    alpha = float_dtype(
        np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.)
    # Push the sampled alphas at the extents of the range to +/- infinity, so
    # that we probe those cases too.
    alpha[alpha == 3.] = float_dtype(float('inf'))
    alpha[alpha == -16.] = -float_dtype(float('inf'))

    # Random log-normally distributed values in approx (1e-5, 100000):
    scale = float_dtype(
        np.exp(np.random.normal(size=num_samples) * 4.) + 1e-5)

    x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
    with tf.GradientTape(persistent=True) as tape:
      for z in (x, alpha, scale):
        tape.watch(z)
      loss = general.lossfun(x, alpha, scale)
      d_x, d_alpha, d_scale = [
          tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
      ]
    return (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossfunPreservesDtype(self, float_dtype):
    """Check the loss's output has the same precision as its input."""
    n = 16
    x = float_dtype(np.random.normal(size=n))
    alpha = float_dtype(np.random.normal(size=n))
    scale = float_dtype(np.exp(np.random.normal(size=n)))
    y = general.lossfun(x, alpha, scale)
    self.assertDTypeEqual(y, float_dtype)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testDerivativeIsMonotonicWrtX(self, float_dtype):
    # Check that the loss increases monotonically with |x|.
    _, _, x, alpha, _, d_x, _, _ = self._precompute_lossfun_inputs(float_dtype)
    # This is just to suppress a warning below.
    d_x = tf.where(tf.math.is_finite(d_x), d_x, tf.zeros_like(d_x))
    mask = np.isfinite(alpha) & (
        np.abs(d_x) > (300. * np.finfo(float_dtype).eps))
    self.assertAllEqual(np.sign(d_x[mask]), np.sign(x[mask]))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossIsNearZeroAtOrigin(self, float_dtype):
    # Check that the loss is near-zero when x is near-zero.
    _, loss, x, _, _, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    self.assertTrue(np.all(np.abs(loss[np.abs(x) < 1e-5]) < 1e-5))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossIsQuadraticNearOrigin(self, float_dtype):
    # Check that the loss is well-approximated by a quadratic bowl when
    # |x| < scale
    _, loss, x, _, scale, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    mask = np.abs(x) < (0.5 * scale)
    loss_quad = 0.5 * np.square(x / scale)
    self.assertAllClose(loss_quad[mask], loss[mask], rtol=1e-5, atol=1e-2)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossIsBoundedWhenAlphaIsNegative(self, float_dtype):
    # Assert that loss < (alpha - 2)/alpha when alpha < 0.
    _, loss, _, alpha, _, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    mask = alpha < 0.
    min_val = np.finfo(float_dtype).min
    alpha_clipped = np.maximum(min_val, alpha[mask])
    self.assertTrue(
        np.all(loss[mask] <= ((alpha_clipped - 2.) / alpha_clipped)))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testDerivativeIsBoundedWhenAlphaIsBelow2(self, float_dtype):
    # Assert that |d_x| < |x|/scale^2 when alpha <= 2.
    _, _, x, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 2)
    self.assertTrue(
        np.all((np.abs(d_x[mask]) <=
                ((np.abs(x[mask]) +
                  (300. * np.finfo(float_dtype).eps)) / scale[mask]**2))))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testDerivativeIsBoundedWhenAlphaIsBelow1(self, float_dtype):
    # Assert that |d_x| < 1/scale when alpha <= 1.
    _, _, _, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 1)
    self.assertTrue(
        np.all((np.abs(d_x[mask]) <=
                ((1. + (300. * np.finfo(float_dtype).eps)) / scale[mask]))))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaDerivativeIsPositive(self, float_dtype):
    # Assert that d_loss / d_alpha > 0.
    _, _, _, alpha, _, _, d_alpha, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha)
    self.assertTrue(np.all(d_alpha[mask] > (-300. * np.finfo(float_dtype).eps)))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testScaleDerivativeIsNegative(self, float_dtype):
    # Assert that d_loss / d_scale < 0.
    _, _, _, alpha, _, _, _, d_scale = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha)
    self.assertTrue(np.all(d_scale[mask] < (300. * np.finfo(float_dtype).eps)))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossIsScaleInvariant(self, float_dtype):
    # Check that loss(mult * x, alpha, mult * scale) == loss(x, alpha, scale)
    (num_samples, loss, x, alpha, scale, _, _, _) = (
        self._precompute_lossfun_inputs(float_dtype))
    # Random log-normally distributed scalings in ~(0.2, 20)
    mult = float_dtype(
        np.maximum(0.2, np.exp(np.random.normal(size=num_samples))))
    # Compute the scaled loss.
    loss_scaled = general.lossfun(mult * x, alpha, mult * scale)
    self.assertAllClose(loss, loss_scaled, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsNegativeInfinity(self, float_dtype):
    # Check that alpha == -Infinity reproduces Welsch aka Leclerc loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-float('inf'))
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # Welsch/Leclerc loss.
    loss_true = (1. - tf.math.exp(-0.5 * tf.square(x / scale)))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsNegativeTwo(self, float_dtype):
    # Check that alpha == -2 reproduces Geman-McClure loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-2.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # Geman-McClure loss.
    loss_true = (2. * tf.square(x / scale) / (tf.square(x / scale) + 4.))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsZero(self, float_dtype):
    # Check that alpha == 0 reproduces Cauchy aka Lorentzian loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(0.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # Cauchy/Lorentzian loss.
    loss_true = (tf.math.log(0.5 * tf.square(x / scale) + 1.))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsOne(self, float_dtype):
    # Check that alpha == 1 reproduces Charbonnier aka pseudo-Huber loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(1.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # Charbonnier loss.
    loss_true = (tf.sqrt(tf.square(x / scale) + 1.) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsTwo(self, float_dtype):
    # Check that alpha == 2 reproduces L2 loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(2.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # L2 Loss.
    loss_true = (0.5 * tf.square(x / scale))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsFour(self, float_dtype):
    # Check that alpha == 4 reproduces a quartic.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(4.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # The true loss.
    loss_true = (
        tf.square(tf.square(x / scale)) / 8. + tf.square(x / scale) / 2.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testAlphaEqualsInfinity(self, float_dtype):
    # Check that alpha == Infinity takes the correct form.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(float('inf'))
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale)

    # The true loss.
    loss_true = (tf.math.exp(0.5 * tf.square(x / scale)) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testApproximateLossIsAccurate(self, float_dtype):
    # Check that the approximate loss (lossfun() with epsilon=1e-6) reasonably
    # approximates the true loss (lossfun() with epsilon=0.) for a range of
    # values of alpha (skipping alpha=0, where the approximation is poor).
    x = np.arange(-10, 10, 0.1, float_dtype)
    scale = float_dtype(1.7)
    for alpha in [-4, -2, -0.2, -0.01, 0.01, 0.2, 1, 1.99, 2, 2.01, 4]:
      alpha = float_dtype(alpha)
      loss = general.lossfun(x, alpha, scale)
      loss_approx = general.lossfun(x, alpha, scale, approximate=True)
      self.assertAllClose(
          loss, loss_approx, rtol=1e-5, atol=1e-4, msg='alpha=%g' % (alpha))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossAndGradientsAreFinite(self, float_dtype):
    # Test that the loss and its approximation both give finite losses and
    # derivatives everywhere that they should for a wide range of values.
    for approximate in [False, True]:
      num_samples = 100000

      # Normally distributed inputs.
      x = float_dtype(np.random.normal(size=num_samples))

      # Uniformly distributed values in (-16, 3), quantized to the nearest
      # 0.1 to ensure that we hit the special cases at 0, 2.
      alpha = float_dtype(
          np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.)

      # Random log-normally distributed values in approx (1e-5, 100000):
      scale = float_dtype(
          np.exp(np.random.normal(size=num_samples) * 4.) + 1e-5)

      # Compute the loss and its derivative with respect to all three inputs.
      x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
      with tf.GradientTape(persistent=True) as tape:
        for z in (x, alpha, scale):
          tape.watch(z)
        loss = general.lossfun(x, alpha, scale, approximate=approximate)
        d_x, d_alpha, d_scale = [
            tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
        ]

      for v in [loss, d_x, d_alpha, d_scale]:
        self.assertTrue(np.all(np.isfinite(v)))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testGradientMatchesFiniteDifferences(self, float_dtype):
    # Test that the loss and its approximation both return gradients that are
    # close to the numerical gradient from finite differences, with forward
    # differencing. Returning correct gradients is TensorFlow's job, so this is
    # just an aggressive sanity check in case some implementation detail causes
    # gradients to incorrectly go to zero due to quantization or stop_gradients
    # in some op that is used by the loss.
    for approximate in [False, True]:
      num_samples = 100000

      # Normally distributed inputs.
      x = float_dtype(np.random.normal(size=num_samples))

      # Uniformly distributed values in (-16, 3), quantized to the nearest
      # 0.1 and then shifted by 0.05 so that we avoid the special cases at
      # 0 and 2 where the analytical gradient won't match finite differences.
      alpha = float_dtype(
          np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.) + 0.05

      # Random uniformly distributed values in [0.5, 1.5]
      scale = float_dtype(np.random.uniform(0.5, 1.5, num_samples))

      # Compute the loss and its derivative with respect to all three inputs.
      x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
      with tf.GradientTape(persistent=True) as tape:
        for z in (x, alpha, scale):
          tape.watch(z)
        loss = general.lossfun(x, alpha, scale, approximate=approximate)
        d_x, d_alpha, d_scale = [
            tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
        ]

      step_size = float_dtype(1e-3)
      n_x = (general.lossfun(x + step_size, alpha, scale) - loss) / step_size
      n_alpha = (general.lossfun(x, alpha + step_size, scale) -
                 loss) / step_size
      n_scale = (general.lossfun(x, alpha, scale + step_size) -
                 loss) / step_size

      self.assertAllClose(n_x, d_x, rtol=1e-2, atol=1e-2)
      self.assertAllClose(n_alpha, d_alpha, rtol=1e-2, atol=1e-2)
      self.assertAllClose(n_scale, d_scale, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()
