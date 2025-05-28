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

"""Tests for spin_math."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from internal import spin_math
import jax
from jax import numpy as jnp
from jax import random
import numpy as np


TEST_BATCH_SIZE = 128


class SpinMathTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = 42
    self._key = random.PRNGKey(self._seed)

  def get_random_vector(self, func, shape):
    if func == random.uniform:
      self._key, _ = random.split(self._key)
      return func(shape=shape, key=self._key)
    else:
      return func(shape=shape)

  @parameterized.parameters([-1.0, 0.0, 0.1, 1.0, 2.0, 100.0])
  def test_safe_sqrt_grad(self, x):
    grad_fn = jax.grad(spin_math.safe_sqrt)
    self.assertTrue(np.isfinite(grad_fn(x)))

  @parameterized.product(
      x=[-1.0, 0.0, 1e-10, 1e-8, 0.1, 1.0, 2.0, 100.0],
      eps=[jnp.finfo(jnp.float32).eps],
      value_at_zero=[0.0, 1e-7, jnp.finfo(jnp.float32).eps],
  )
  def test_safe_sqrt(self, x, eps, value_at_zero):
    safe_sqrt = functools.partial(
        spin_math.safe_sqrt, eps=eps, value_at_zero=value_at_zero
    )
    orig_grad_fn = jax.grad(jnp.sqrt)
    grad_fn = jax.grad(safe_sqrt)
    if x < eps:
      self.assertEqual(grad_fn(x), 0.0)
      self.assertEqual(jnp.sqrt(value_at_zero), safe_sqrt(x))
    else:
      self.assertEqual(grad_fn(x), orig_grad_fn(x))
      self.assertEqual(jnp.sqrt(x), safe_sqrt(x))

  @parameterized.parameters([-1.0, 0.0, 0.1, 1.0, 2.0, 100.0])
  def test_safe_acos_grad(self, x):
    grad_fn = jax.grad(spin_math.safe_acos)
    self.assertTrue(np.isfinite(grad_fn(x)))

  @parameterized.product(
      x=[0.0, 1e-10, 1e-8, 0.1, 1.0, 2.0, 100.0],
      eps=[jnp.finfo(jnp.float32).eps],
      value_at_zero=[0.0, 1e-7, jnp.finfo(jnp.float32).eps],
  )
  def test_safe_log(self, x, eps, value_at_zero):
    safe_log = functools.partial(
        spin_math.safe_log, eps=eps, value_at_zero=value_at_zero
    )
    orig_grad_fn = jax.grad(jnp.log)
    grad_fn = jax.grad(safe_log)
    if x < eps:
      self.assertEqual(grad_fn(x), 0.0)
      self.assertEqual(jnp.log(value_at_zero), safe_log(x))
    else:
      self.assertEqual(grad_fn(x), orig_grad_fn(x))
      self.assertEqual(jnp.log(x), safe_log(x))

  @parameterized.parameters([-1.0, 0.0, 0.1, 1.0, 2.0, 100.0])
  def test_safe_log_grad(self, x):
    grad_fn = jax.grad(spin_math.safe_log)
    self.assertTrue(np.isfinite(grad_fn(x)))

  @parameterized.product(
      batch=[None, TEST_BATCH_SIZE],
      func=[random.uniform, jnp.ones],
      sign=[-1, 1],
      ndim=[1, 2, 3, 4],
  )
  def test_from_homogenous(self, batch, func, sign, ndim):
    shape = (batch, ndim + 1) if batch else (ndim + 1,)
    vector = sign * self.get_random_vector(func, shape=shape)
    output = spin_math.from_homogeneous(vector)
    self.assertEqual(output.shape, (*shape[:-1], ndim))
    np.testing.assert_array_equal(output, vector[Ellipsis, :-1] / vector[Ellipsis, -1:])

  @parameterized.product(
      batch=[None, TEST_BATCH_SIZE],
      func=[random.uniform, jnp.ones, jnp.zeros],
      sign=[-1, 1],
      ndim=[1, 2, 3, 4],
  )
  def test_to_homogenous(self, batch, func, sign, ndim):
    shape = (batch, ndim) if batch else (ndim,)
    vector = sign * self.get_random_vector(func, shape=shape)
    output = spin_math.to_homogeneous(vector)
    self.assertEqual(output.shape, (*shape[:-1], ndim + 1))
    np.testing.assert_array_equal(output[Ellipsis, :-1], vector)
    np.testing.assert_array_equal(output[Ellipsis, -1:], 1.0)

  @parameterized.product(
      batch=[None, (1,), (100,), (32, 32)],
      func=[random.uniform, jnp.ones],
      scale=[1.0, 2.0, 3.0, 4.0],
      ndim=[1, 2, 3, 4],
  )
  def test_transform_vectors_scale(self, batch, func, scale, ndim):
    shape = (*batch, ndim) if batch else (ndim,)
    vectors = self.get_random_vector(func, shape)
    matrix = jnp.diag(jnp.array([*([scale] * ndim), 1.0]))
    transformed_points = spin_math.apply_homogeneous_transform(matrix, vectors)
    np.testing.assert_allclose(scale * vectors, transformed_points)

  @parameterized.product(
      batch=[None, (1,), (100,), (32, 32)], ndim=[1, 2, 3, 4]
  )
  def test_normalize_zero(self, batch, ndim):
    shape = (*batch, ndim) if batch else (ndim,)
    vector = self.get_random_vector(jnp.zeros, shape=shape)
    normalized = spin_math.normalize(vector)
    np.testing.assert_array_equal(normalized, jnp.zeros_like(normalized))
    norm_grad = jax.grad(lambda x: spin_math.normalize(x).sum())(vector)
    self.assertTrue(np.all(np.isfinite(norm_grad)))

  @parameterized.product(
      batch=[None, (1,), (100,), (32, 32)], ndim=[1, 2, 3, 4]
  )
  def test_normalize_nonzero(self, batch, ndim):
    shape = (*batch, ndim) if batch else (ndim,)
    vector = self.get_random_vector(random.uniform, shape=shape)
    normalized = spin_math.normalize(vector)
    np.testing.assert_allclose(
        jnp.linalg.norm(normalized, axis=-1),
        jnp.ones_like(normalized[Ellipsis, 0]),
        atol=1e-6,
    )
    norm_grad = jax.grad(lambda x: spin_math.normalize(x).sum())(vector)
    self.assertTrue(np.all(np.isfinite(norm_grad)))

  @parameterized.product(
      slope=[0.1, 2, 200, 2000], threshold=(0, 0.1, 0.2, 0.5, 0.8, 1.0)
  )
  def test_generalized_bias_and_gain_equal_at_threshold(self, slope, threshold):
    """Tests that the output is equal to the input at the threshold."""
    y = spin_math.generalized_bias_and_gain(threshold, slope, threshold)
    np.testing.assert_allclose(y, threshold, rtol=1e-5)

  def test_generalized_bias_and_gain_line(self):
    """Tests that the output is equal to the input at the threshold."""
    x = jnp.linspace(0, 1, 1000)
    y = spin_math.generalized_bias_and_gain(x, slope=1, threshold=0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    np.testing.assert_allclose(y, x, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
