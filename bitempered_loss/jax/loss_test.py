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

"""Tests for bi-tempered loss."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy.testing as npt
from six.moves import zip

from bitempered_loss.jax import loss

python_version = "PY3"


class LossTest(absltest.TestCase):

  def test_normalization(self):
    """Test the normalization constant."""
    rng = random.PRNGKey(seed=1335)
    activations = random.normal(rng, shape=[100, 50000])
    for t in [0.99, 1.01]:
      normalization_constants = loss.compute_normalization(
          activations, t, num_iters=20)
      npt.assert_allclose(normalization_constants.shape, (100, 1))
      probabilities = jnp.sum(
          loss.exp_t(activations - normalization_constants, t), -1)
      npt.assert_allclose(probabilities, [1.0] * 100, atol=1e-5)
    for t in [0.1, 2.0]:
      normalization_constants = loss.compute_normalization(
          activations, t, num_iters=20)
      probabilities = jnp.sum(
          loss.exp_t(activations - normalization_constants, t), -1)
      npt.assert_allclose(probabilities, [1.0] * 100, atol=1e-5)

  def test_limit_case_logistic_loss(self):
    """Test for checking if t1 = t2 = 1.0 yields the logistic loss."""
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rng = random.PRNGKey(seed=1335)
    activations = random.normal(rng, shape=[3, 3])
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 1.0,
                                                 1.0)
    logistic_loss = loss._cross_entropy_loss(
        logits=activations, labels=labels)
    npt.assert_allclose(actual_loss, logistic_loss)

  def test_loss_value(self):
    """Test the loss based on precomputed values."""
    labels = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]])
    activations = jnp.array([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0],
                             [4.0, -3.0, -6.0]])
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5, 1.5)
    npt.assert_allclose(actual_loss,
                        jnp.array([0.02301914, 0.18972909, 0.93874922]),
                        atol=1e-4)
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5,
                                                 0.8, num_iters=20)
    npt.assert_allclose(actual_loss,
                        jnp.array([0.21646356, 0.41836615, 1.33997854]),
                        atol=1e-4)

  def test_constant_shift(self):
    """Test if adding a constant to all activations is vacuous."""
    labels = jnp.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]])
    rng = random.PRNGKey(seed=1335)
    rng, use_key = random.split(rng)
    activations = random.normal(use_key, shape=[3, 3])
    bias = random.normal(rng, shape=[3, 1])
    for t2 in [0.8, 1.2]:
      actual_loss = loss.bi_tempered_logistic_loss(
          activations, labels, 0.5, t2)
      shifted_loss = loss.bi_tempered_logistic_loss(
          activations + bias, labels, 0.5, t2)
      npt.assert_allclose(actual_loss, shifted_loss, atol=1e-6)

  def test_label_smoothing(self):
    """Test label smoothing."""
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = jnp.array([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0],
                             [4.0, -3.0, -6.0]])
    actual_loss = loss.bi_tempered_logistic_loss(
        activations, labels, 0.5, 1.5, label_smoothing=0.1)
    npt.assert_allclose(
        actual_loss, jnp.array([0.76652711, 0.08627685, 1.35443510]),
        atol=1e-5)

  def test_binary_logistic_loss(self):
    """Test binary logistic loss."""
    labels = jnp.array([1.0, 0.0])
    activations = jnp.array([0.0, 0.0])
    actual_loss = loss.bi_tempered_binary_logistic_loss(activations, labels,
                                                        1.0, 1.0)
    npt.assert_allclose(actual_loss, jnp.array([0.69314718, 0.69314718]),
                        atol=1e-5)

  def test_dynamic_temperatures(self):
    """Test changing temperatures dynamically."""
    labels = jnp.array([[0.2, 0.5, 0.3]])
    activations = jnp.array([[-0.5, 0.1, 2.0]])
    t1_values = [1.0, 0.9, 0.8, 0.7]
    t2_values = [1.0, 1.1, 1.2, 1.3]
    loss_values = [[0.628705], [0.45677936], [0.34298314], [0.26295574]]
    loss_out = []
    for t1_value, t2_value in zip(t1_values, t2_values):
      loss_out.append(loss.bi_tempered_logistic_loss(
          activations, labels, t1_value, t2_value, num_iters=5))
    npt.assert_allclose(loss_values, loss_out, atol=1e-5)

  def test_tempered_softmax(self):
    # Test softmax function with different temperatures.
    activations = jnp.array(
        [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]])
    # Test with temperature = 1.0, which should recover regular
    # softmax probabilities.
    softmax_probabilities_t_1 = loss.tempered_softmax(
        activations, t=1.0)
    vanilla_softmax_probabilties = jax.nn.softmax(activations)
    npt.assert_allclose(vanilla_softmax_probabilties,
                        softmax_probabilities_t_1, atol=1e-6)
    softmax_probabilities_t_4 = loss.tempered_softmax(
        activations, t=4.0)
    expected_softmax_probabilities_t_4 = jnp.array([[
        0.3205458, 0.32714278, 0.3523402
    ], [0.3430056, 0.36491093,
        0.29220778], [0.41369352, 0.30534995, 0.28299212]])
    npt.assert_allclose(expected_softmax_probabilities_t_4,
                        softmax_probabilities_t_4, atol=1e-6)

  def test_tempered_sigmoid(self):
    # Test sigmoid function with different temperatures.
    activations = jnp.array([0.0, 3.0, 6.0])
    # Test with temperature = 1.0, which should recover regular
    # sigmoid probabilities.
    sigmoid_probabilities_t_1 = loss.tempered_sigmoid(
        activations, t=1.0)
    vanilla_softmax_probabilties = jax.nn.sigmoid(activations)
    npt.assert_allclose(vanilla_softmax_probabilties,
                        sigmoid_probabilities_t_1, atol=1e-6)
    sigmoid_probabilities_t_4 = loss.tempered_sigmoid(
        activations, t=4.0)
    expected_sigmoid_probabilities_t_4 = jnp.array([0.5, 0.58516014, 0.6421035])
    npt.assert_allclose(expected_sigmoid_probabilities_t_4,
                        sigmoid_probabilities_t_4, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
