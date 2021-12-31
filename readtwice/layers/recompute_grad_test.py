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

"""Tests for recompute_grad."""
from typing import Sequence
import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from readtwice.layers import recompute_grad as recompute_grad_lib
from readtwice.layers import stateless_dropout as stateless_dropout_lib


def _compute_gradients(model, x):
  with tf.GradientTape() as tape:
    y = model(x)
  return tape.gradient(
      y, model.trainable_variables
      if hasattr(model, 'trainable_variables') else tape.watched_variables())


def _make_gradients_op(model, x):
  f = lambda x: _compute_gradients(model, x)
  return (tf.function(experimental_compile=True)(lambda: f(x))
          if tf.executing_eagerly() else tf.compat.v1.tpu.rewrite(f, (x,)))


class RecomputeDense(tf.keras.layers.Layer):
  """Dense layer that recomputes the forward pass during backpropagation."""

  def __init__(self, units, **kwargs):
    super(RecomputeDense, self).__init__(**kwargs)
    self._units = tf.nest.flatten(units)

  def build(self, input_shape):
    units = input_shape[-1:] + self._units
    kernels = []
    biases = []
    for i in range(1, len(units)):
      kernels.append(
          self.add_weight('kernel_{}'.format(i), (units[i - 1], units[i])))
      biases.append(self.add_weight('bias_{}'.format(i), (units[i],)))
    self._kernels = kernels
    self._biases = biases
    super(RecomputeDense, self).build(input_shape)

  def call(self, inputs, **kwargs):

    @recompute_grad_lib.recompute_grad
    def f(x):
      for kernel, bias in zip(self._kernels, self._biases):
        x = tf.nn.tanh(tf.matmul(x, kernel) + bias)
      return x

    return f(inputs)


class RecomputeDense2Args(RecomputeDense):
  """Extension of `RecomputeDense` that takes and returns 2 arguments."""

  def build(self, input_shape):
    super(RecomputeDense2Args, self).build(input_shape[0])

  def call(self, inputs, **kwargs):

    @recompute_grad_lib.recompute_grad
    def f(x1, x2):
      for kernel, bias in zip(self._kernels, self._biases):
        x1 = tf.nn.tanh(tf.matmul(x1, kernel) + bias)
      for kernel, bias in zip(self._kernels, self._biases):
        x2 = tf.nn.tanh(tf.matmul(x2, kernel) + bias)
      return x1, x2

    return f(*inputs)


class RecomputeDenseDropout(RecomputeDense):
  """Extension of `RecomputeDense` that uses dropout."""

  def __init__(self, units, rate, **kwargs):
    super(RecomputeDenseDropout, self).__init__(units, **kwargs)
    self._rate = rate

  def call(self, inputs, recompute_grad=False, **kwargs):

    def f(x):
      if recompute_grad_lib.get_recompute_context() is None:
        generator = tf.random.experimental.get_global_generator()
        recompute_grad_seed = tf.stack(
            (generator.uniform_full_int([], tf.int32, name='seed'), 0))
      else:
        recompute_grad_seed = tf.stack(
            (recompute_grad_lib.get_recompute_context().seed, 0))
      seeds = tf.random.stateless_uniform([len(self._units), 2],
                                          recompute_grad_seed,
                                          minval=-2**31,
                                          maxval=2**31 - 1,
                                          dtype=tf.int32,
                                          name='dropout_seeds')
      for i, (kernel, bias) in enumerate(zip(self._kernels, self._biases)):
        x = tf.nn.tanh(tf.matmul(x, kernel) + bias)
        x = stateless_dropout_lib.stateless_dropout(x, self._rate, seeds[i])
      return x

    if recompute_grad:
      f = recompute_grad_lib.recompute_grad(f)
    return f(inputs)


class RecomputeGradXlaTest(tf.test.TestCase):
  """Tests for recompute_grad_lib.recompute_grad with XLA."""

  @property
  def device(self):
    if tf.config.list_logical_devices('TPU'):
      return sorted(tf.config.list_logical_devices('TPU'))[0]
    elif tf.config.list_logical_devices('GPU'):
      return sorted(tf.config.list_logical_devices('GPU'))[0]
    else:
      return sorted(tf.config.list_logical_devices('CPU'))[0]

  def test_xla_model_correctness(self):
    """Tests correctness of the gradient calculation."""

    def _make_model(input_size):
      inputs = tf.keras.Input((input_size,))
      x = inputs
      for _ in range(2):
        x = RecomputeDense([16] * 2)(x)
      outputs = tf.keras.layers.Dense(1)(x)
      return tf.keras.Model(inputs, outputs)

    with tf.device(self.device):
      recompute_model = _make_model(4)
      control_model = tf.keras.Sequential([
          tf.keras.layers.Dense(16, activation='tanh', input_shape=(4,)),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(16, activation='tanh'),
          tf.keras.layers.Dense(1),
      ])
      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.tpu.initialize_system())
        self.evaluate(tf.compat.v1.initializers.global_variables())
      for source, target in zip(control_model.trainable_variables,
                                recompute_model.trainable_variables):
        self.evaluate(target.assign(source))
      x = tf.ones((32, 4))
      actual_gradients = self.evaluate(_make_gradients_op(recompute_model, x))
      expected_gradients = self.evaluate(_make_gradients_op(control_model, x))
    for actual, expected in zip(actual_gradients, expected_gradients):
      self.assertAllClose(actual, expected)

  def test_xla_model_2_argument_case(self):
    """Tests for a recomputed function that takes and returns multiple args.

    We don't test correctness of the gradients here; we're just making sure
    `recompute_grad` runs without error in this case.
    """

    def _make_model(input_size):
      input1 = tf.keras.Input((input_size,))
      input2 = tf.keras.Input((input_size,))
      x = (input1, input2)
      for _ in range(2):
        x = RecomputeDense2Args([16] * 2)(x)
      outputs = tf.keras.layers.Dense(1)(x[0] + x[1])
      return tf.keras.Model((input1, input2), outputs)

    with tf.device(self.device):
      recompute_model = _make_model(4)
      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.tpu.initialize_system())
        self.evaluate(tf.compat.v1.initializers.global_variables())
      x1 = tf.ones((32, 4))
      x2 = 2 * tf.ones((32, 4))
      _ = self.evaluate(_make_gradients_op(recompute_model, (x1, x2)))

  @unittest.skip
  def test_dropout(self):
    """Tests recomputing gradients of functions that have randomness."""

    def _make_model(input_size, recompute_grad=False):
      inputs = tf.keras.Input((input_size,))
      x = RecomputeDenseDropout(
          units=[10, 20, 10], rate=0.5)(
              inputs, recompute_grad=recompute_grad)
      x = RecomputeDenseDropout(
          units=[10, 20, 10], rate=0.5)(
              x, recompute_grad=recompute_grad)
      outputs = tf.keras.layers.Dense(1)(x)
      return tf.keras.Model(inputs, outputs)

    # Initialize models and generator.
    with tf.device(self.device):
      model = _make_model(8)
      recompute_model = _make_model(8, recompute_grad=True)
      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.tpu.initialize_system())
        self.evaluate(tf.compat.v1.initializers.global_variables())
      x = tf.ones((32, 8))
      _ = self.evaluate(
          tf.function(
              experimental_compile=True)(model)(x) if tf.executing_eagerly()
          else tf.compat.v1.tpu.rewrite(lambda x: model(x), (x,)))  # pylint: disable=unnecessary-lambda
      for source, target in zip(model.trainable_variables,
                                recompute_model.trainable_variables):
        self.evaluate(target.assign(source))

    # Compare gradients.
    gradients_op = _make_gradients_op(model, x)
    recompute_gradients_op = _make_gradients_op(recompute_model, x)
    with tf.device(self.device):
      # Capture state and compute gradients in undecorated model.
      state = self.evaluate(tf.random.experimental.get_global_generator().state)
      gradients = self.evaluate(gradients_op)
      # Restore state and compute gradients in decorated model.
      self.evaluate(
          tf.random.experimental.get_global_generator().state.assign(state))
      recomputed_gradients = self.evaluate(recompute_gradients_op)
    for gradient, recomputed_gradient in zip(gradients, recomputed_gradients):
      self.assertAllClose(gradient, recomputed_gradient, rtol=1e-5, atol=1e-5)


class RecomputeGradTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for recompute_grad_lib.recompute_grad."""

  def test_seed(self):
    """Tests that a seed can be provided to recompute_grad."""
    seed = tf.constant(2020, dtype=tf.int32)

    def _make_model():
      inputs = tf.keras.Input((4,))
      outputs = tf.keras.layers.Dense(10)(inputs)
      return tf.keras.Model(inputs, outputs)

    model = _make_model()
    if not tf.executing_eagerly():
      self.evaluate(tf.compat.v1.initializers.global_variables())
    # Set up functions to take gradients with respect to variables.
    def f(x, seed=np.array(1, dtype=np.int32)):
      if recompute_grad_lib.get_recompute_context() is not None:
        seed = recompute_grad_lib.get_recompute_context().seed
      return stateless_dropout_lib.stateless_dropout(
          model(x), rate=0.5, seed=tf.stack([seed, 0]))

    f_recompute = recompute_grad_lib.recompute_grad(f, seed=seed)
    # Compute gradients and compare them.
    x = tf.ones((2, 4))
    gradients = self.evaluate(_compute_gradients(lambda x: f(x, seed), x))
    recomputed_gradients = self.evaluate(_compute_gradients(f_recompute, x))
    for gradient, recomputed_gradient in zip(gradients, recomputed_gradients):
      self.assertAllClose(gradient, recomputed_gradient)


if __name__ == '__main__':
  tf.test.main()
