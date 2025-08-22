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

"""Tests for diffusion.py."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from d3pm.text import diffusion
from d3pm.text import losses
from d3pm.text import model_utils


class MatrixDiffusionsTestUtilsTest(absltest.TestCase):
  """Test Diffusion (discrete and continuous)."""

  def test_beta_diagonal_diffusion(self):
    """Test the Diffusion noise diffusion."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(dim=100, schedule=schedule)

    expected = (1 - 1e-3) * jnp.eye(100) + 1e-3 * jnp.ones((100, 100)) / 100
    np.testing.assert_array_almost_equal(diff.get(0), expected)

    expected = (1 - 1e-1) * jnp.eye(100) + 1e-1 * jnp.ones((100, 100)) / 100
    np.testing.assert_array_almost_equal(diff.get(100), expected)

  def test_product_the_hard_way(self):
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(
        dim=100, schedule=schedule, use_fast_inference=False)

    self.assertFalse(diff.supports_efficient_inference())

    product = diff.get_qt_matrix(0)
    np.testing.assert_array_almost_equal(product, jnp.eye(100))

    product = diff.get_qt_matrix(1)
    np.testing.assert_array_almost_equal(product, diff.get(0))

  def test_product_fast(self):
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(
        dim=100, schedule=schedule, use_fast_inference=True)

    self.assertTrue(diff.supports_efficient_inference())

    product = diff.get_qt_matrix(0)
    np.testing.assert_array_almost_equal(product, jnp.eye(100))

    product = diff.get_qt_matrix(1)
    np.testing.assert_array_almost_equal(product, diff.get(0))

  def test_product_constant(self):
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(dim=100, schedule=schedule)

    self.assertTrue(diff.supports_efficient_inference())

    product = diff.get_qt_matrix(0)
    np.testing.assert_array_almost_equal(product, jnp.eye(100))

    product = diff.get_qt_matrix(1)
    np.testing.assert_array_almost_equal(product, diff.get(0))

    product = diff.get_qt_matrix(10)
    expected = jnp.linalg.matrix_power(diff.get(0), 10)
    np.testing.assert_array_almost_equal(product, expected)

  def test_sample_and_posterior(self):
    """Tests that the discrete process predicted probabilities are correct."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(dim=100, schedule=schedule)

    key = jrandom.PRNGKey(0)
    inputs = jnp.ones((1,), jnp.int32)

    probs, sample = diff.sample_and_compute_posterior_q(
        key, inputs, 0, return_logits=False)

    self.assertEqual(probs.shape, (1, 100))
    self.assertAlmostEqual(probs[0, 1], 1.0, places=5)

    self.assertEqual(sample.shape, (1,))
    np.testing.assert_array_equal(sample, jnp.array([1]))

  def test_toy_example(self):
    """Tests that the discrete process predicted probabilities are correct."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(dim=2, schedule=schedule)

    key = jrandom.PRNGKey(0)
    transition = np.array([[1 - 1e-3 / 2, 1e-3 / 2], [1e-3 / 2, 1 - 1e-3 / 2]],
                          dtype=np.float64)
    np.testing.assert_array_almost_equal(transition, diff.get(0))

    mat_power = np.linalg.matrix_power(transition, 5)
    np.testing.assert_array_almost_equal(mat_power, diff.get_qt_matrix(5))

    ## test starting ins tate 0
    inputs = jnp.zeros((1,), jnp.int32)
    probs = diff.get_qt_given_q0(inputs, t=5, make_one_hot=True)
    expected_probs = mat_power[:, 0]
    np.testing.assert_array_almost_equal(probs[0], expected_probs)

    ## test starting in state 1
    inputs = jnp.ones((1,), jnp.int32)
    probs = diff.get_qt_given_q0(inputs, 5, make_one_hot=True)
    expected_probs = mat_power[:, 1]
    np.testing.assert_array_almost_equal(probs[0], expected_probs)

    probs, _ = diff.sample_and_compute_posterior_q(
        key,
        inputs,
        5,
        return_logits=False,
        samples=jnp.ones((1, 1), jnp.int32))

    expected_logits = transition[1] * mat_power[:, 1]
    expected_probs = expected_logits / expected_logits.sum()

    np.testing.assert_array_almost_equal(probs[0, 0], expected_probs)

  def test_compute_posterior(self):
    """Tests that the discrete process predicted probabilities are correct."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BetaDiagonalDiffusion(dim=100, schedule=schedule)

    inputs = jnp.ones((2,), jnp.int32)
    q_t = diff.get_qt_given_q0(inputs, 0, make_one_hot=True)

    self.assertEqual(q_t.shape, (2, 100))
    self.assertAlmostEqual(float(q_t[0][1]), 1.0)
    self.assertAlmostEqual(float(q_t[0][0]), 0.0)

  def test_slow_and_fast(self):
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='standard',
        beta_min=5e-4,
        beta_max=5e-2,
        num_steps=100,
    )

    x0 = jnp.array([0, 1, 2])
    key = jrandom.PRNGKey(0)
    dim = 16

    fast_diff = diffusion.BetaDiagonalDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=True)

    slow_diff = diffusion.BetaDiagonalDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=False)

    for t in range(100):
      qt_slow = slow_diff.get_qt_matrix(t)
      qt_fast = fast_diff.get_qt_matrix(t)

      np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)

      qt_slow = slow_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
      qt_fast = fast_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)

      np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)

      posterior_slow, samples_slow = slow_diff.sample_and_compute_posterior_q(
          key, x_0=x0, t=t, make_one_hot=True)
      posterior_fast, samples_fast = fast_diff.sample_and_compute_posterior_q(
          key, x_0=x0, t=t, make_one_hot=True)

      np.testing.assert_array_almost_equal(
          posterior_slow, posterior_fast, decimal=3)
      np.testing.assert_array_equal(samples_slow, samples_fast)

    qt = fast_diff.get_qt_given_q0(q0=x0, t=100, make_one_hot=True)
    np.testing.assert_allclose(qt, 1 / dim, rtol=1e-5)

    qt = slow_diff.get_qt_given_q0(q0=x0, t=100, make_one_hot=True)
    np.testing.assert_allclose(qt, 1 / dim, rtol=1e-5)


class AllDiffusionTest(parameterized.TestCase):
  """Test BandDiffusion schedule."""

  @parameterized.parameters([diffusion.MaskDiffusion],
                            [diffusion.BetaDiagonalDiffusion],
                            [diffusion.AutoRegressiveDiffusion])
  def test_all_models(self, diffusion_cls):
    """Test the Diffusion noise diffusion."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='standard',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    dim = 100
    length = 100
    key = jrandom.PRNGKey(0)

    x0 = jrandom.randint(key, (length,), 0, dim)
    diff = diffusion_cls(dim=100, schedule=schedule)

    if hasattr(diffusion, 'get'):
      np.testing.assert_allclose(diff.get(0).sum(0), 1.0, rtol=1e-6)
      np.testing.assert_allclose(diff.get(10).sum(0), 1.0, rtol=1e-6)
      np.testing.assert_allclose(diff.get(99).sum(0), 1.0, rtol=1e-6)

      np.testing.assert_allclose(diff.get_qt_matrix(0), jnp.eye(100), rtol=1e-6)

    expected = losses.onehot(x0, dim)
    result = diff.get_qt_given_q0(q0=x0, t=0, make_one_hot=True)
    np.testing.assert_allclose(result, expected)

    expected = jax.nn.softmax(jrandom.normal(key, (length, dim)))
    result = diff.get_qt_given_q0(q0=expected, t=0, make_one_hot=False)
    np.testing.assert_allclose(result, expected)

    q0 = jax.nn.softmax(jrandom.normal(key, (length, dim)))
    result = diff.get_qt_given_q0(q0=q0, t=0, make_one_hot=False)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-6)

    expected = diff.stationary_probs(x0.shape)
    result = diff.get_qt_given_q0(q0=x0, t=100, make_one_hot=True)
    np.testing.assert_allclose(result, expected)


class BandDiffusionTest(absltest.TestCase):
  """Test BandDiffusion schedule."""

  def test_band_diagonal(self):
    """Test the Diffusion noise diffusion."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-3,
        num_steps=100,
    )

    diff = diffusion.BandDiagonalDiffusion(dim=100, schedule=schedule, width=5)

    np.testing.assert_allclose(diff.get(0).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(diff.get(10).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        diff.get(0)[0, 0], 1 - schedule(0) + schedule(0) / 3, rtol=1e-6)

    np.testing.assert_allclose(diff.get_qt_matrix(0), jnp.eye(100), rtol=1e-6)


class MaskDiffusionTest(absltest.TestCase):

  def test_mask_diffusion(self):
    """Test the Diffusion noise diffusion."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)

    np.testing.assert_allclose(diff.get(0).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(diff.get(10).sum(0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(diff.get(0)[0, 0], 1.0 - schedule(0), rtol=1e-6)
    np.testing.assert_allclose(diff.get(1)[0, 0], 1.0 - schedule(1), rtol=1e-6)

    np.testing.assert_allclose(diff.get_qt_matrix(0), jnp.eye(100), rtol=1e-6)

  def test_slow_and_fast(self):
    """Compares fast and slow inference."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='standard',
        beta_min=5e-4,
        beta_max=5e-2,
        num_steps=100,
    )

    dim = 16
    length = 16

    fast_diff = diffusion.MaskDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=True)

    slow_diff = diffusion.MaskDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=False)

    key = jrandom.PRNGKey(0)
    x0 = jrandom.randint(key, (length,), 0, dim)

    for t in range(100):
      qt_slow = slow_diff.get_qt_matrix(t)
      qt_fast = fast_diff.get_qt_matrix(t)

      np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)

      qt_slow = slow_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
      qt_fast = fast_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)

      np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)

      np.testing.assert_array_almost_equal(qt_slow.sum(axis=-1), 1., decimal=3)
      np.testing.assert_array_almost_equal(qt_fast.sum(axis=-1), 1., decimal=3)

      posterior_slow, samples_slow = slow_diff.sample_and_compute_posterior_q(
          key, x_0=x0, t=t, make_one_hot=True)
      posterior_fast, samples_fast = fast_diff.sample_and_compute_posterior_q(
          key, x_0=x0, t=t, make_one_hot=True)

      np.testing.assert_array_almost_equal(
          posterior_slow, posterior_fast, decimal=3)
      np.testing.assert_array_equal(samples_slow, samples_fast)

    qt = fast_diff.get_qt_given_q0(q0=x0, t=100, make_one_hot=True)
    np.testing.assert_allclose(
        qt, losses.onehot(jnp.full(x0.shape, dim - 1), dim), rtol=1e-6)

    qt = slow_diff.get_qt_given_q0(q0=x0, t=100, make_one_hot=True)
    np.testing.assert_allclose(
        qt, losses.onehot(jnp.full(x0.shape, dim - 1), dim), rtol=1e-6)

  def test_large_matrices(self):
    """Tests precision for large matrices."""
    key = jrandom.PRNGKey(0)
    dim = 1000
    length = 64
    x0 = jrandom.randint(key, (length,), 0, dim)

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=5e-4,
        beta_max=5e-2,
        num_steps=100,
    )

    diff = diffusion.MaskDiffusion(dim, schedule, use_fast_inference=True)
    fn = functools.partial(diff.get_qt_given_q0, make_one_hot=True)
    result = fn(x0, 100)
    np.testing.assert_array_almost_equal(result.sum(axis=-1), 1.0)


class AutoRegressiveDiffusionTest(absltest.TestCase):

  def test_autoregressive_diffusion(self):
    """Test the Diffusion noise diffusion."""
    seq_len = 100
    dim = 100
    sequence = jnp.arange(seq_len, dtype=jnp.int32)
    sequence2 = jnp.arange(seq_len, dtype=jnp.int32) + 1

    q0 = (losses.onehot(sequence, dim) + losses.onehot(sequence2, dim)) / 2

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=seq_len,
    )

    diff = diffusion.AutoRegressiveDiffusion(dim=dim, schedule=schedule)

    for t in range(1, seq_len):
      xt = diff.get_qt_given_q0(q0=sequence, t=t, make_one_hot=True)

      argmax = xt.argmax(-1)
      np.testing.assert_array_equal(argmax[:-t], sequence[:-t])
      np.testing.assert_array_equal(argmax[-t:], 99)

      xt = diff.get_qt_given_q0(q0=q0, t=t)

      np.testing.assert_array_equal(xt[:-t], q0[:-t])
      np.testing.assert_array_equal(xt[-t:][:, 99], 1.)

    key = jrandom.PRNGKey(0)
    _, sample = diff.sample_and_compute_posterior_q(
        key=key, x_0=sequence, t=0, make_one_hot=True, return_logits=False)

    np.testing.assert_array_equal(sample[:-1], sequence[:-1])
    np.testing.assert_array_equal(sample[-1], 99)

    qt, sample = diff.sample_and_compute_posterior_q(
        key=key, x_0=q0, t=10, make_one_hot=False, return_logits=False)

    sample_q = losses.onehot(sample, dim)
    np.testing.assert_array_equal(qt[:-12], sample_q[:-12])
    np.testing.assert_array_equal(qt[-11], q0[-11])
    np.testing.assert_array_equal(qt[-10:][:, 99], 1.0)


class NearestNeighborDiffusionTest(absltest.TestCase):

  def test_state_init(self):
    """Tests that the discrete process predicted probabilities are correct."""

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear',
        beta_min=1e-3,
        beta_max=1e-2,
        num_steps=100,
    )

    diff = diffusion.NearestNeighborDiffusion(dim=4, schedule=schedule, knn=2)
    embeddings = jnp.array([[0., 0.], [1., 1.], [3., 3.], [1., 0.]])
    matrix = jnp.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
    mat = matrix + matrix.T

    for _ in range(diff.num_sinkhorn_iterations):
      mat = mat / mat.sum(1, keepdims=True)
      mat = mat / mat.sum(0, keepdims=True)

    mat = mat / mat.sum(0, keepdims=True)

    state = diff.update_state(embeddings)
    np.testing.assert_array_almost_equal(mat, state, decimal=3)

  def test_cached_reverse(self):
    """Test cached diffusion."""
    num_steps = 100
    dim = 20
    key = jrandom.PRNGKey(0)

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear', beta_min=5e-4, beta_max=5e-3, num_steps=num_steps)

    embedding = jrandom.normal(key, (dim, 32))

    slow = diffusion.NearestNeighborCachedDiffusion(
        dim, schedule, use_slow_get=True)
    fast = diffusion.NearestNeighborCachedDiffusion(
        dim, schedule, use_slow_get=False)

    state = slow.update_state(embedding)

    slow.set_state(state)
    fast.set_state(state)

    slow_q, slow_sample = slow.sample_and_compute_posterior_q(
        key, jnp.array([0, 1, 2]), 30, make_one_hot=True)
    fast_q, fast_sample = fast.sample_and_compute_posterior_q(
        key, jnp.array([0, 1, 2]), 30, make_one_hot=True)

    np.testing.assert_array_almost_equal(slow_q, fast_q, decimal=3)
    np.testing.assert_array_equal(slow_sample, fast_sample)

    reverse_probs = slow.qt_reverse(
        jnp.array([0, 1, 2]), 30, make_one_hot=True)[0]
    expected = slow.get(30)[0]
    np.testing.assert_array_almost_equal(reverse_probs, expected, decimal=3)

  def test_precision(self):
    dim = 1000
    key = jrandom.PRNGKey(0)
    embedding = jrandom.normal(key, (dim, 32))
    num_steps = 100

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear', beta_min=5e-4, beta_max=5e-3, num_steps=num_steps)

    diff = diffusion.NearestNeighborCachedDiffusion(
        dim, schedule, use_numpy=True)

    state = diff.update_state(embedding)
    diff.set_state(state)

    q = diff.get_qt_given_q0(
        jnp.array([0, 1, 2]), 99, make_one_hot=True, return_logits=False)[0]
    expected = np.linalg.matrix_power(
        np.array(state['matrix_power_state'].cache[0], np.float64),
        diff.powers[99])[0]

    np.testing.assert_array_almost_equal(q, expected)


class PrecisionExpmTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(['jax', 'scipy'], [200, 1000], [5, 10, 25, 100]))
  def test_expm_precision(self, expm_type, dim, knn):
    key = jrandom.PRNGKey(0)
    embeddings = jrandom.normal(key, (dim, 32))
    x0 = jrandom.randint(key, (64,), 0, dim)

    num_steps = 128

    schedule = diffusion.create_discrete_diffusion_schedule(
        kind='linear', beta_min=5e-3, beta_max=5e-2, num_steps=num_steps)

    diff = diffusion.NearestNeighborCachedDiffusion(
        dim,
        schedule,
        use_numpy=True,
        use_matrix_exponential=True,
        expm_type=expm_type,
        knn=knn)

    state = diff.update_state(embeddings)
    diff.set_state(state)

    neighbors = model_utils.get_nearest_neighbors(
        embeddings, k=knn, include_self=False, num_chunks=10)

    matrix = jnp.zeros((dim, dim), jnp.float32)
    matrix = matrix.at[neighbors, jnp.arange(dim)[:, None]].set(1.)

    matrix = matrix + matrix.T
    transition_rate = matrix - jnp.diagflat(jnp.sum(matrix, axis=1))

    beta_min = diff.min_exponent

    for t in range(num_steps, 5):
      q_t = diff.get_qt_given_q0(x0, t, make_one_hot=True)

      power = diff.powers[t]
      transition = jax.scipy.linalg.expm(beta_min * power * transition_rate)
      expected = transition[x0]

      np.testing.assert_array_almost_equal(q_t, expected)


if __name__ == '__main__':
  absltest.main()
