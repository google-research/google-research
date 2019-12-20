# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for adaptive.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf
from robust_loss import adaptive
from robust_loss import util
from robust_loss import wavelet


def _generate_pixel_toy_image_data(image_width, num_samples, _):
  """Generates pixel data for _test_fitting_toy_image_data_is_correct().

  Constructs a "mean" image in RGB pixel space (parametrized by `image_width`)
  and draws `num_samples` samples from a normal distribution using that mean,
  and returns those samples and their empirical mean as reference.

  Args:
    image_width: The width and height in pixels of the images being produced.
    num_samples: The number of samples to generate.
    _: Dummy argument so that this function's interface matches
      _generate_wavelet_toy_image_data()

  Returns:
    A tuple of (samples, reference, color_space, representation), where
    samples = A set of sampled images of size
      (`num_samples`, `image_width`, `image_width`, 3)
    reference = The empirical mean of `samples` of size
      (`image_width`, `image_width`, 3).
    color_space = 'RGB'
    representation = 'PIXEL'
  """
  color_space = 'RGB'
  representation = 'PIXEL'
  mu = np.random.uniform(size=(image_width, image_width, 3))
  samples = np.random.normal(
      loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
  reference = np.mean(samples, 0)
  return samples, reference, color_space, representation


def _generate_wavelet_toy_image_data(image_width, num_samples,
                                     wavelet_num_levels):
  """Generates wavelet data for testFittingImageDataIsCorrect().

  Constructs a "mean" image in the YUV wavelet domain (parametrized by
  `image_width`, and `wavelet_num_levels`) and draws `num_samples` samples
  from a normal distribution using that mean, and returns RGB images
  corresponding to those samples and to the mean (computed in the
  specified latent space) of those samples.

  Args:
    image_width: The width and height in pixels of the images being produced.
    num_samples: The number of samples to generate.
    wavelet_num_levels: The number of levels in the wavelet decompositions of
      the generated images.

  Returns:
    A tuple of (samples, reference, color_space, representation), where
    samples = A set of sampled images of size
      (`num_samples`, `image_width`, `image_width`, 3)
    reference = The empirical mean of `samples` (computed in YUV Wavelet space
      but returned as an RGB image) of size (`image_width`, `image_width`, 3).
    color_space = 'YUV'
    representation = 'CDF9/7'
  """
  color_space = 'YUV'
  representation = 'CDF9/7'
  samples = []
  reference = []
  for level in range(wavelet_num_levels):
    samples.append([])
    reference.append([])
    w = image_width // 2**(level + 1)
    scaling = 2**level
    for _ in range(3):
      # Construct the ground-truth pixel band mean.
      mu = scaling * np.random.uniform(size=(3, w, w))
      # Draw samples from the ground-truth mean.
      band_samples = np.random.normal(
          loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
      # Take the empirical mean of the samples as a reference.
      band_reference = np.mean(band_samples, 0)
      samples[-1].append(np.reshape(band_samples, [-1, w, w]))
      reference[-1].append(band_reference)
  # Handle the residual band.
  mu = scaling * np.random.uniform(size=(3, w, w))
  band_samples = np.random.normal(
      loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
  band_reference = np.mean(band_samples, 0)
  samples.append(np.reshape(band_samples, [-1, w, w]))
  reference.append(band_reference)
  # Collapse and reshape wavelets to be ({_,} width, height, 3).
  samples = wavelet.collapse(samples, representation)
  reference = wavelet.collapse(reference, representation)
  samples = tf.transpose(
      tf.reshape(samples, [num_samples, 3, image_width, image_width]),
      [0, 2, 3, 1])
  reference = tf.transpose(reference, [1, 2, 0])
  # Convert into RGB space.
  samples = util.syuv_to_rgb(samples)
  reference = util.syuv_to_rgb(reference)
  with tf.Session() as sess:
    samples, reference = sess.run((samples, reference))
  return samples, reference, color_space, representation


class AdaptiveTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(AdaptiveTest, self).setUp()
    np.random.seed(0)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testInitialAlphaAndScaleAreCorrect(self, float_dtype):
    """Tests that `alpha` and `scale` are initialized as expected."""
    for i in range(8):
      # Generate random ranges for alpha and scale.
      alpha_lo = float_dtype(np.random.uniform())
      alpha_hi = float_dtype(np.random.uniform() + 1.)
      # Half of the time pick a random initialization for alpha, the other half
      # use the default value.
      if i % 2 == 0:
        alpha_init = float_dtype(alpha_lo + np.random.uniform() *
                                 (alpha_hi - alpha_lo))
        true_alpha_init = alpha_init
      else:
        alpha_init = None
        true_alpha_init = (alpha_lo + alpha_hi) / 2.
      scale_init = float_dtype(np.random.uniform() + 0.5)
      scale_lo = float_dtype(np.random.uniform() * 0.1)
      with tf.compat.v1.variable_scope('trial_' + str(i)):
        _, alpha, scale = adaptive.lossfun(
            tf.constant(np.zeros((10, 10), float_dtype)),
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            alpha_init=alpha_init,
            scale_lo=scale_lo,
            scale_init=scale_init)
        with self.session() as sess:
          sess.run(tf.global_variables_initializer())
          # Check that `alpha` and `scale` are what we expect them to be.
          alpha, scale = sess.run([alpha, scale])
          self.assertAllClose(alpha, true_alpha_init * np.ones_like(alpha))
          self.assertAllClose(scale, scale_init * np.ones_like(alpha))

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testFixedAlphaAndScaleAreCorrect(self, float_dtype):
    """Tests that fixed alphas and scales do not change during optimization)."""
    for i in range(8):
      alpha_lo = float_dtype(np.random.uniform() * 2.)
      alpha_hi = alpha_lo
      scale_init = float_dtype(np.random.uniform() + 0.5)
      scale_lo = scale_init
      samples = float_dtype(np.random.uniform(size=(10, 10)))
      # We must construct some variable for TF to attempt to optimize.
      mu = tf.Variable(
          tf.zeros(tf.shape(samples)[1], float_dtype), name='DummyMu')
      x = samples - mu[tf.newaxis, :]
      with tf.compat.v1.variable_scope('trial_' + str(i)):
        loss, alpha, scale = adaptive.lossfun(
            x,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            scale_lo=scale_lo,
            scale_init=scale_init)
        with self.session() as sess:
          # Do one giant gradient descent step (usually a bad idea).
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=1000.)
          step = optimizer.minimize(tf.reduce_sum(loss))
          sess.run(tf.global_variables_initializer())
          _ = sess.run(step)
          # Check that `alpha` and `scale` have not changed from their initial
          # values.
          alpha, scale = sess.run([alpha, scale])
          alpha_init = (alpha_lo + alpha_hi) / 2.
          self.assertAllClose(alpha, alpha_init * np.ones_like(alpha))
          self.assertAllClose(scale, scale_init * np.ones_like(alpha))

  def _sample_cauchy_ppf(self, num_samples):
    """Draws ``num_samples'' samples from a Cauchy distribution.

    Because actual sampling is expensive and requires many samples to converge,
    here we sample by drawing `num_samples` evenly-spaced values in [0, 1]
    and then interpolate into the inverse CDF (aka PPF) of a Cauchy
    distribution. This produces "samples" where maximum-likelihood estimation
    likely recovers the true distribution even if `num_samples` is small.

    Args:
      num_samples: The number of samples to draw.

    Returns:
      A numpy array containing `num_samples` evenly-spaced "samples" from a
      zero-mean Cauchy distribution whose scale matches our distribution/loss
      when our scale = 1.
    """
    spacing = 1. / num_samples
    p = np.arange(0., 1., spacing) + spacing / 2.
    return scipy.stats.cauchy(0., np.sqrt(2.)).ppf(p)

  def _sample_normal_ppf(self, num_samples):
    """Draws ``num_samples'' samples from a Normal distribution.

    Because actual sampling is expensive and requires many samples to converge,
    here we sample by drawing `num_samples` evenly-spaced values in [0, 1]
    and then interpolate into the inverse CDF (aka PPF) of a Normal
    distribution. This produces "samples" where maximum-likelihood estimation
    likely recovers the true distribution even if `num_samples` is small.

    Args:
      num_samples: The number of samples to draw.

    Returns:
      A numpy array containing `num_samples` evenly-spaced "samples" from a
      zero-mean unit-scale Normal distribution.
    """
    spacing = 1. / num_samples
    p = np.arange(0., 1., spacing) + spacing / 2.
    return scipy.stats.norm(0., 1.).ppf(p)

  def _sample_nd_mixed_data(self, n, m, float_dtype):
    """`n` Samples from `m` scaled+shifted Cauchy and Normal distributions."""
    samples0 = self._sample_cauchy_ppf(n)
    samples2 = self._sample_normal_ppf(n)
    mu = np.random.normal(size=m)
    alpha = (np.random.uniform(size=m) > 0.5) * 2
    scale = np.exp(np.clip(np.random.normal(size=m), -3., 3.))
    samples = (
        np.tile(samples0[:, np.newaxis], [1, m]) *
        (alpha[np.newaxis, :] == 0.) +
        np.tile(samples2[:, np.newaxis], [1, m]) *
        (alpha[np.newaxis, :] == 2.)) * scale[np.newaxis, :] + mu[np.newaxis, :]
    return [float_dtype(x) for x in [samples, mu, alpha, scale]]

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testFittingToyNdMixedDataIsCorrect(self, float_dtype):
    """Tests that minimizing the adaptive loss recovers the true model.

    Here we generate a 2D array of samples drawn from a mix of scaled and
    shifted Cauchy and Normal distributions. We then minimize our loss with
    respect to the mean, scale, and shape of each distribution, and check that
    after minimization the shape parameter is near-zero for the Cauchy data and
    near 2 for the Normal data, and that the estimated means and scales are
    accurate.

    Args:
      float_dtype: The type (np.float32 or np.float64) of data to test.
    """
    samples, mu_true, alpha_true, scale_true = self._sample_nd_mixed_data(
        100, 8, float_dtype)
    mu = tf.Variable(tf.zeros(tf.shape(samples)[1], float_dtype))
    x = samples - mu[tf.newaxis, :]
    losses, alpha, scale = adaptive.lossfun(x)
    loss = tf.reduce_mean(losses)

    with self.session() as sess:
      init_rate = 1.
      final_rate = 0.1
      num_iters = 201
      global_step = tf.Variable(0, trainable=False)
      t = tf.cast(global_step, tf.float32) / (num_iters - 1)
      rate = tf.math.exp(
          tf.math.log(init_rate) * (1. - t) + tf.math.log(final_rate) * t)
      optimizer = tf.train.AdamOptimizer(
          learning_rate=rate, beta1=0.5, beta2=0.9, epsilon=1e-08)
      step = optimizer.minimize(loss, global_step=global_step)
      sess.run(tf.global_variables_initializer())

      for _ in range(num_iters):
        _ = sess.run(step)
      mu, alpha, scale = sess.run([mu, alpha[0, :], scale[0, :]])

    for a, b in [(alpha, alpha_true), (scale, scale_true), (mu, mu_true)]:
      self.assertAllClose(a, b * np.ones_like(a), rtol=0.1, atol=0.1)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testFittingToyNdMixedDataIsCorrectStudentsT(self, float_dtype):
    """Tests that minimizing the Student's T loss recovers the true model.

    Here we generate a 2D array of samples drawn from a mix of scaled and
    shifted Cauchy and Normal distributions. We then minimize our loss with
    respect to the mean, scale, and shape of each distribution, and check that
    after minimization the log-df parameter is near-zero for the Cauchy data and
    very large for the Normal data, and that the estimated means and scales are
    accurate.

    Args:
      float_dtype: The type (np.float32 or np.float64) of data to test.
    """
    samples, mu_true, alpha_true, scale_true = self._sample_nd_mixed_data(
        100, 8, float_dtype)
    mu = tf.Variable(tf.zeros(tf.shape(samples)[1], float_dtype))
    x = samples - mu[tf.newaxis, :]
    losses, log_df, scale = adaptive.lossfun_students(x)
    loss = tf.reduce_mean(losses)

    with self.session() as sess:
      init_rate = 1.
      final_rate = 0.1
      num_iters = 201
      global_step = tf.Variable(0, trainable=False)
      t = tf.cast(global_step, tf.float32) / (num_iters - 1)
      rate = tf.math.exp(
          tf.math.log(init_rate) * (1. - t) + tf.math.log(final_rate) * t)
      optimizer = tf.train.AdamOptimizer(
          learning_rate=rate, beta1=0.5, beta2=0.9, epsilon=1e-08)
      step = optimizer.minimize(loss, global_step=global_step)
      sess.run(tf.global_variables_initializer())

      for _ in range(num_iters):
        _ = sess.run(step)
      mu, log_df, scale = sess.run([mu, log_df[0, :], scale[0, :]])

    for ldf, a_true in zip(log_df, alpha_true):
      if a_true == 0:
        self.assertAllClose(ldf, 0., rtol=0.1, atol=0.1)
      elif a_true == 2:
        self.assertAllGreater(ldf, 4)
    scale /= np.sqrt(2. - (alpha_true / 2.))
    for a, b in [(scale, scale_true), (mu, mu_true)]:
      self.assertAllClose(a, b * np.ones_like(a), rtol=0.1, atol=0.1)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testLossfunPreservesDtype(self, float_dtype):
    """Checks the loss's outputs have the same precisions as its input."""
    samples, _, _, _ = self._sample_nd_mixed_data(100, 8, float_dtype)
    loss, alpha, scale = adaptive.lossfun(samples)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      loss, alpha, scale = sess.run([loss, alpha, scale])
    self.assertDTypeEqual(loss, float_dtype)
    self.assertDTypeEqual(alpha, float_dtype)
    self.assertDTypeEqual(scale, float_dtype)

  @parameterized.named_parameters(('Single', np.float32),
                                  ('Double', np.float64))
  def testImageLossfunPreservesDtype(self, float_dtype):
    """Tests that image_lossfun's outputs precisions match its input."""
    x = float_dtype(np.random.uniform(size=(10, 64, 64, 3)))
    loss, alpha, scale = adaptive.image_lossfun(x)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      loss, alpha, scale = sess.run([loss, alpha, scale])
    self.assertDTypeEqual(loss, float_dtype)
    self.assertDTypeEqual(alpha, float_dtype)
    self.assertDTypeEqual(scale, float_dtype)

  @parameterized.named_parameters(('Wavelet', _generate_wavelet_toy_image_data),
                                  ('Pixel', _generate_pixel_toy_image_data))
  def testFittingImageDataIsCorrect(self, image_data_callback):
    """Tests that minimizing the adaptive image loss recovers the true model.

    Here we generate a stack of color images drawn from a normal distribution,
    and then minimize image_lossfun() with respect to the mean and scale of each
    distribution, and check that after minimization the estimated means are
    close to the true means.

    Args:
      image_data_callback: The function used to generate the training data and
        parameters used during optimization.
    """
    # Generate toy data.
    image_width = 4
    num_samples = 10
    wavelet_num_levels = 2  # Ignored by _generate_pixel_toy_image_data().
    (samples, reference, color_space, representation) = image_data_callback(
        image_width, num_samples, wavelet_num_levels)

    # Construct the loss.
    prediction = tf.Variable(tf.zeros(tf.shape(reference), tf.float64))
    x = samples - prediction[tf.newaxis, :]
    loss, alpha, scale = adaptive.image_lossfun(
        x,
        color_space=color_space,
        representation=representation,
        wavelet_num_levels=wavelet_num_levels,
        alpha_lo=2,
        alpha_hi=2)
    loss = tf.reduce_mean(loss)

    # Minimize the loss.
    with self.session() as sess:
      init_rate = 0.1
      final_rate = 0.01
      num_iters = 201
      global_step = tf.Variable(0, trainable=False)
      t = tf.cast(global_step, tf.float32) / (num_iters - 1)
      rate = tf.math.exp(
          tf.math.log(init_rate) * (1. - t) + tf.math.log(final_rate) * t)
      optimizer = tf.train.AdamOptimizer(
          learning_rate=rate, beta1=0.5, beta2=0.9, epsilon=1e-08)
      step = optimizer.minimize(loss, global_step=global_step)
      sess.run(tf.global_variables_initializer())
      for _ in range(num_iters):
        _ = sess.run(step)
      scale, alpha, prediction = sess.run([scale, alpha, prediction])
    self.assertAllClose(prediction, reference, rtol=0.01, atol=0.01)


if __name__ == '__main__':
  tf.test.main()
