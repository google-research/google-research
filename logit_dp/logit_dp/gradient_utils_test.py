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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from logit_dp.logit_dp import encoders
from logit_dp.logit_dp import gradient_utils


def compute_expected_gradient(w_0, x1, x2, y1, y2):
  """Computes expected gradient for embedding model v=w_0*x."""
  nv1 = x1
  nv2 = x2

  nu1 = y1
  nu2 = y2

  u1 = w_0 * y1
  u2 = w_0 * y2

  v1 = w_0 * x1
  v2 = w_0 * x2

  expected_dot_prods = [[v1 * u1, v1 * u2], [v2 * u1, v2 * u2]]
  expected_vec_jac_prods = [
      [nv1 * u1 + nu1 * v1, nv1 * u2 + nu2 * v1],
      [nv2 * u1 + nu1 * v2, nv2 * u2 + nu2 * v2],
  ]
  gradient_t_1 = (nv1 * u1 + nu1 * v1) + (nv2 * u2 + nu2 * v2)

  d1 = jnp.exp(v1 * u1) + jnp.exp(v1 * u2)
  d2 = jnp.exp(v2 * u1) + jnp.exp(v2 * u2)

  num1 = jnp.exp(v1 * u1) * (nv1 * u1 + nu1 * v1) + jnp.exp(v1 * u2) * (
      nv1 * u2 + nu2 * v1
  )
  num2 = jnp.exp(v2 * u1) * (nv2 * u1 + nu1 * v2) + jnp.exp(v2 * u2) * (
      nv2 * u2 + nu2 * v2
  )

  expected_gradient = -(gradient_t_1 - (num1 / d1) - (num2 / d2)) / 2
  return expected_gradient, expected_dot_prods, expected_vec_jac_prods


def loss_fn(params, rng, x, y, forward):
  v = jax.vmap(forward.apply, in_axes=(None, None, 0))(params, rng, x)
  u = jax.vmap(forward.apply, in_axes=(None, None, 0))(params, rng, y)
  similarities = jnp.matmul(v, u.T)
  labels = jax.nn.one_hot([0, 1], similarities.shape[-1])

  return -jnp.sum(labels * jax.nn.log_softmax(similarities)) / labels.shape[0]


class GradientUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.w_0 = 0.15
    # Queries
    self.x1 = 3.14
    self.x2 = 2.71

    # Docs
    self.y1 = 0.15
    self.y2 = 4.35
    self.input_pairs = jnp.array([[self.x1, self.y1], [self.x2, self.y2]])

    class MyOneWeight(hk.Module):

      def __init__(self, w_0, name=None):
        super().__init__(name=name)
        self.output_size = 1
        self.w_0 = w_0

      def __call__(self, x):
        # j, k = x.shape[-1], self.output_size
        w_init = lambda x, y: jnp.array([self.w_0])
        w = hk.get_parameter('w', shape=[1], dtype=x.dtype, init=w_init)
        embedding = jnp.dot(x, w)
        return embedding

    def simple_forward_fn(x):
      model = MyOneWeight(self.w_0)
      return model(x)

    self.forward_fn = hk.transform(simple_forward_fn)

  @parameterized.parameters(5, 10)
  def test_sensitivity_bound_large_batch(self, batch_size):
    """Test a simple case where we can compute manually the result."""
    similarity_bound = 0.05
    l2_norm_clip = 1.0
    expected_sensitivity = 1 / batch_size + 2 * jnp.exp(
        2 * similarity_bound
    ) / (batch_size + jnp.exp(2 * similarity_bound) - 1)
    sensitivity = gradient_utils.compute_contrastive_loss_gradient_sensitivity(
        l2_norm_clip=l2_norm_clip,
        batch_size=batch_size,
        similarity_bound=similarity_bound,
    )
    self.assertAlmostEqual(sensitivity, expected_sensitivity, places=6)

  def test_sensitivity_bound_small_batch(self):
    """Tests simple case with batch_size=1."""
    similarity_bound = 5
    expected_sensitivity = 3
    l2_norm_clip = 1.0
    batch_size = 1
    bound = gradient_utils.compute_contrastive_loss_gradient_sensitivity(
        l2_norm_clip=l2_norm_clip,
        batch_size=batch_size,
        similarity_bound=similarity_bound,
    )
    self.assertAlmostEqual(bound, expected_sensitivity, places=6)

  def test_compute_contrastive_loss_gradient(self):
    batch = self.input_pairs

    key = hk.PRNGSequence(429)
    initial_params = self.forward_fn.init(next(key), batch[0][0])

    key = jax.random.PRNGKey(421)

    grad_through_loss = jax.grad(loss_fn)(
        initial_params, key, batch[0], batch[1], self.forward_fn
    )
    expected_grad, _, _ = compute_expected_gradient(
        self.w_0, self.x1, self.x2, self.y1, self.y2
    )

    self.assertAllClose(
        grad_through_loss['my_one_weight']['w'][0],
        expected_grad,
        rtol=1e-1,
    )

    jax_key = jax.random.PRNGKey(0)
    cosine_values, cosine_grads = (
        gradient_utils.compute_batch_cosine_values_and_clipped_gradients(
            batch,
            self.forward_fn,
            initial_params,
            jax_key,
        )
    )
    final_grad = gradient_utils.compute_contrastive_loss_gradient(
        cosine_values, cosine_grads, temperature=1.0
    )
    self.assertAllClose(
        final_grad['my_one_weight']['w'][0],
        expected_grad,
        rtol=1e-3,
    )

  @parameterized.product(l2_norm_clip=[0.01, 0.1, 1.14], temperature=[0.2, 1.9])
  def test_compute_clipped_contrastive_loss_gradient(
      self, l2_norm_clip, temperature
  ):
    batch = self.input_pairs

    key = hk.PRNGSequence(429)
    initial_params = self.forward_fn.init(next(key), batch[0][0])

    key = jax.random.PRNGKey(421)

    jax_key = jax.random.PRNGKey(0)
    cosine_values, cosine_grads = (
        gradient_utils.compute_batch_cosine_values_and_clipped_gradients(
            batch_input_pairs=batch,
            forward_fn=self.forward_fn,
            params=initial_params,
            key=jax_key,
            l2_norm_clip=l2_norm_clip,
        )
    )
    # Checks that pairwise similarity gradients are clipped.
    self.assertAllLessEqual(
        cosine_grads[0]['my_one_weight']['w'], l2_norm_clip + 1e-5
    )

    # Checks that the final gradient is different than the one computed with
    # unclipped gradients.
    final_grad = gradient_utils.compute_contrastive_loss_gradient(
        cosine_values, cosine_grads, temperature=temperature
    )
    expected_grad_no_clip, _, _ = compute_expected_gradient(
        self.w_0, self.x1, self.x2, self.y1, self.y2
    )
    self.assertNotAllClose(
        expected_grad_no_clip, final_grad['my_one_weight']['w']
    )

  @parameterized.product(
      gradients=[[np.array([1, 1])], [np.array([1]), np.array([1])]],
      clip_and_expected_norm=[(1.0, 1.0), (10, np.sqrt(2))],
  )
  def test_clip_gradients(self, gradients, clip_and_expected_norm):
    clip_norm, expected_norm = clip_and_expected_norm
    clipped_gradient = gradient_utils.compute_clipped_gradients(
        gradients, clip_norm
    )
    self.assertAlmostEqual(
        expected_norm,
        optax.global_norm(clipped_gradient),
        places=6,
    )

  def test_compute_dp_gradients(self):
    batch = self.input_pairs

    key = hk.PRNGSequence(429)
    initial_params = self.forward_fn.init(next(key), batch[0][0])

    jax_key = jax.random.PRNGKey(421)

    no_noise_gradient = gradient_utils.compute_dp_gradients(
        initial_params,
        batch,
        self.forward_fn,
        l2_norm_clip=None,
        noise_multiplier=0.0,
        temperature=1.0,
        key=jax_key,
        sequential_computation_steps=1,
    )
    expected_grad, _, _ = compute_expected_gradient(
        self.w_0, self.x1, self.x2, self.y1, self.y2
    )
    self.assertAllClose(
        no_noise_gradient['my_one_weight']['w'][0],
        expected_grad,
        rtol=1e-3,
    )

  @parameterized.parameters(1, 2)
  def test_compute_gradient_sequentially(self, sequential_computation_steps):
    expected_grad, _, _ = compute_expected_gradient(
        self.w_0, self.x1, self.x2, self.y1, self.y2
    )
    hk_key = hk.PRNGSequence(429)
    initial_params = self.forward_fn.init(next(hk_key), self.input_pairs[0][0])

    jax_key = jax.random.PRNGKey(421)
    result_grad = gradient_utils.compute_contrastive_loss_gradient_sequentially(
        self.input_pairs,
        self.forward_fn,
        initial_params,
        jax_key,
        sequential_computation_steps=sequential_computation_steps,
        temperature=1.0,
        l2_norm_clip=None,
    )['my_one_weight']['w'][0]

    self.assertAllClose(expected_grad, result_grad)

  @parameterized.product(
      sequential_computation_steps=[2, 4, 8],
      temperature=[0.2, 1.0],
      l2_norm_clip=[None, 0.01, 0.1],
  )
  def test_gradients_sequentially_cifar(
      self, sequential_computation_steps, temperature, l2_norm_clip
  ):
    batch_size = 8
    input_key = jax.random.PRNGKey(0)
    input_pairs = jax.random.uniform(input_key, (batch_size, 2, 3, 32, 32))

    encoder = encoders.SmallEmbeddingNet
    forward_fn = encoders.get_forward_fn_from_module(encoder)
    hk_key = hk.PRNGSequence(0)
    input_shape = (3, 32, 32)

    initial_params = forward_fn.init(next(hk_key), jnp.zeros(input_shape))
    jax_key = jax.random.PRNGKey(421)

    expected_grad = gradient_utils.compute_dp_gradients(
        params=initial_params,
        input_pairs=input_pairs,
        forward_fn=forward_fn,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=0.0,
        temperature=temperature,
        key=jax_key,
        sequential_computation_steps=1,
    )
    result_grad = gradient_utils.compute_contrastive_loss_gradient_sequentially(
        input_pairs,
        forward_fn,
        initial_params,
        jax_key,
        sequential_computation_steps=sequential_computation_steps,
        temperature=temperature,
        l2_norm_clip=l2_norm_clip,
    )
    logging.info('Result')
    logging.info(result_grad)
    logging.info('Expected')
    logging.info(expected_grad)
    self.assertAllClose(expected_grad, result_grad)


if __name__ == '__main__':
  absltest.main()
