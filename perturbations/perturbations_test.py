# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Tests for differentiable_programming.perturbations."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from perturbations import perturbations


def reduce_sign_any(input_tensor, axis=-1):
  """A logical or of the signs of a tensor along an axis.

  Args:
   input_tensor: Tensor<float> of any shape.
   axis: the axis along which we want to compute a logical or of the signs of
     the values.

  Returns:
   A Tensor<float>, which as the same shape as the input tensor, but without the
    axis on which we reduced.
  """
  boolean_sign = tf.math.reduce_any(
      tf.cast((tf.sign(input_tensor) + 1) / 2.0, dtype=tf.bool), axis=axis)
  return tf.cast(boolean_sign, dtype=input_tensor.dtype) * 2.0 - 1.0


class PerturbationsTest(parameterized.TestCase, tf.test.TestCase):
  """Testing the perturbations module."""

  def setUp(self):
    super(PerturbationsTest, self).setUp()
    tf.random.set_seed(0)

  @parameterized.parameters([perturbations._GUMBEL, perturbations._NORMAL])
  def test_sample_noise_with_gradients(self, noise):
    shape = (3, 2, 4)
    samples, gradients = perturbations.sample_noise_with_gradients(noise, shape)
    self.assertAllEqual(samples.shape, shape)
    self.assertAllEqual(gradients.shape, shape)

  def test_sample_noise_with_gradients_raise(self):
    with self.assertRaises(ValueError):
      _, _ = perturbations.sample_noise_with_gradients('unknown', (3, 2, 4))

  @parameterized.parameters([1e-3, 1e-2, 1e-1])
  def test_perturbed_reduce_sign_any(self, sigma):
    input_tensor = tf.constant([[-0.3, -1.2, 1.6], [-0.4, -2.4, -1.0]])
    soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)
    output_tensor = soft_reduce_any(input_tensor, axis=-1)
    self.assertAllClose(output_tensor, [1.0, -1.0])

  def test_perturbed_reduce_sign_any_gradients(self):
    # We choose a point where the gradient should be above noise, that is
    # to say the distance to 0 along one direction is about sigma.
    sigma = 0.1
    input_tensor = tf.constant(
        [[-0.6, -1.2, 0.5 * sigma], [-2 * sigma, -2.4, -1.0]])
    soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)
    with tf.GradientTape() as tape:
      tape.watch(input_tensor)
      output_tensor = soft_reduce_any(input_tensor)
    gradient = tape.gradient(output_tensor, input_tensor)
    # The two values that could change the soft logical or should be the one
    # with real positive impact on the final values.
    self.assertAllGreater(gradient[0, 2], 0.0)
    self.assertAllGreater(gradient[1, 0], 0.0)
    # The value that is more on the fence should bring more gradient than any
    # other one.
    self.assertAllLessEqual(gradient, gradient[0, 2].numpy())

  def test_unbatched_rank_one_raise(self):
    with self.assertRaises(ValueError):
      input_tensor = tf.constant([-0.6, -0.5, 0.5])
      dim = len(input_tensor)
      n = 10000000

      argmax = lambda t: tf.one_hot(tf.argmax(t, 1), dim)
      soft_argmax = perturbations.perturbed(argmax, sigma=0.5, num_samples=n)
      _ = soft_argmax(input_tensor)

  def test_perturbed_argmax_gradients_without_minibatch(self):
    input_tensor = tf.constant([-0.6, -0.5, 0.5])
    dim = len(input_tensor)
    eps = 1e-2
    n = 10000000

    argmax = lambda t: tf.one_hot(tf.argmax(t, 1), dim)
    soft_argmax = perturbations.perturbed(
        argmax, sigma=0.5, num_samples=n, batched=False)
    norm_argmax = lambda t: tf.reduce_sum(tf.square(soft_argmax(t)))

    w = tf.random.normal(input_tensor.shape)
    w /= tf.linalg.norm(w)
    var = tf.Variable(input_tensor)
    with tf.GradientTape() as tape:
      value = norm_argmax(var)

    grad = tape.gradient(value, var)
    grad = tf.reshape(grad, input_tensor.shape)

    value_minus = norm_argmax(input_tensor - eps * w)
    value_plus = norm_argmax(input_tensor + eps * w)

    lhs = tf.reduce_sum(w * grad)
    rhs = (value_plus - value_minus) * 1./(2*eps)
    self.assertAllLess(tf.abs(lhs - rhs), 0.05)

  def test_perturbed_argmax_gradients_with_minibatch(self):
    input_tensor = tf.constant([[-0.6, -0.7, 0.5], [0.9, -0.6, -0.5]])
    dim = len(input_tensor)
    eps = 1e-2
    n = 10000000

    argmax = lambda t: tf.one_hot(tf.argmax(t, -1), dim)
    soft_argmax = perturbations.perturbed(argmax, sigma=2.5, num_samples=n)
    norm_argmax = lambda t: tf.reduce_sum(tf.square(soft_argmax(t)))

    w = tf.random.normal(input_tensor.shape)
    w /= tf.linalg.norm(w)
    var = tf.Variable(input_tensor)
    with tf.GradientTape() as tape:
      value = norm_argmax(var)

    grad = tape.gradient(value, var)
    grad = tf.reshape(grad, input_tensor.shape)

    value_minus = norm_argmax(input_tensor - eps * w)
    value_plus = norm_argmax(input_tensor + eps * w)

    lhs = tf.reduce_sum(w * grad)
    rhs = (value_plus - value_minus) * 1./(2*eps)
    self.assertAllLess(tf.abs(lhs - rhs), 0.05)

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
