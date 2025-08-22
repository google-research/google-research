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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from imp.max.modeling import routing
from imp.max.utils import sharding


_INPUTS_SHARDINGS = (('expert', 'data'), None, 'model')
_LOGITS_SHARDINGS = (('expert', 'data'), None, None)
_KERNEL_SHARDINGS = ('model', None)


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1, 1),
                                      dcn_mesh_shape=(1, 1, 1)),
      ['expert', 'data', 'model'],
  )


class RoutingTest(parameterized.TestCase):

  def test_load_balancing_loss(self):
    num_tokens = 5
    num_experts = 2
    num_selected_experts = 1
    rng = jax.random.key(0)

    router_probs = jax.random.uniform(
        rng, (num_tokens, num_experts), minval=0, maxval=1)
    expert_indices = jax.random.randint(
        rng, (num_tokens, num_selected_experts), minval=0, maxval=2)

    self.assertEqual(
        routing._load_balancing_loss(router_probs, expert_indices), 0.8741045)

  def test_tokens_choose_one_expert_scatter_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 1  # Switch routing case
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseScatterRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )
    @jax.jit
    def _run_forward(token_inputs):
      router_indices, variables = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_indices, variables

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_indices, variables = _run_forward(token_inputs)

    expected_indices = jnp.array([
        [
            [[2, 1]],
            [[3, 0]],
            [[2, 0]],
            [[2, 2]],
        ],
        [
            [[3, 0]],
            [[2, 1]],
            [[3, 1]],
            [[2, 0]],
        ],
    ], dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [[0.25390625], [0.2578125], [0.25585938], [0.]],
        [[0.2578125], [0.25390625], [0.25585938], [0.25585938]],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertEqual(router_indices.auxiliary_loss, 1.0154836)
    self.assertEqual(router_indices.router_z_loss, 1.8987045)

    # Assert shardings are propagated properly
    self.assertEqual(variables['params']['router_weights']['w']['kernel'].names,
                     _KERNEL_SHARDINGS)

  def test_tokens_choose_one_expert_scatter_router_no_bpr(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 1  # Switch routing case
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseScatterRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=False,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )
    @jax.jit
    def _run_forward(token_inputs):
      router_indices, variables = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_indices, variables

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_indices, _ = _run_forward(token_inputs)

    expected_indices = jnp.array([
        [
            [[2, 0]],
            [[3, 0]],
            [[2, 1]],
            [[2, 2]],
        ],
        [
            [[3, 0]],
            [[2, 0]],
            [[3, 1]],
            [[2, 1]],
        ],
    ], dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [[0.25390625], [0.2578125], [0.25585938], [0.]],
        [[0.2578125], [0.25390625], [0.25585938], [0.25585938]],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertEqual(router_indices.auxiliary_loss, 1.0154836)
    self.assertEqual(router_indices.router_z_loss, 1.8987045)

  def test_tokens_choose_multiple_experts_scatter_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 2
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseScatterRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )
    @jax.jit
    def _run_forward(token_inputs):
      router_indices, variables = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_indices, variables

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_indices, _ = _run_forward(token_inputs)

    expected_indices = jnp.array([
        [
            [[2, 1], [3, 2]],
            [[3, 0], [2, 3]],
            [[2, 0], [3, 1]],
            [[2, 2], [0, 0]],
        ],
        [
            [[3, 0], [2, 2]],
            [[2, 1], [3, 3]],
            [[3, 1], [2, 3]],
            [[2, 0], [3, 2]],
        ],
    ], dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [
            [0.25390625, 0.],
            [0.2578125, 0.],
            [0.25585938, 0.25390625],
            [0., 0.25],
        ],
        [
            [0.2578125, 0.],
            [0.25390625, 0.],
            [0.25585938, 0.],
            [0.25585938, 0.],
        ],
    ], dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertAlmostEqual(router_indices.auxiliary_loss, 2.026483, delta=1e-5)
    self.assertAlmostEqual(router_indices.router_z_loss, 1.8987045, delta=1e-5)

  def test_tokens_choose_one_expert_mask_router(self):
    num_groups = 2
    tokens_per_group = 3
    hidden_dim = 4
    num_experts = 2
    num_selected_experts = 1  # Switch routing case
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseMaskedRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )
    @jax.jit
    def _run_forward(token_inputs):
      router_mask, variables = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_mask, variables

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_mask, variables = _run_forward(token_inputs)

    expected_mask = jnp.array([
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
    ], dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.51171875], [0.]],
            [[0.], [0.50390625]],
            [[0.], [0.]],
        ],
        [
            [[0.50390625], [0.]],
            [[0.], [0.5078125]],
            [[0.], [0.]],
        ],
    ], dtype=jnp.float32)

    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 1.0015408)
    self.assertEqual(router_mask.router_z_loss, 0.47536469)

    # Assert shardings are propagated properly
    self.assertEqual(variables['params']['router_weights']['w']['kernel'].names,
                     _KERNEL_SHARDINGS)

  def test_tokens_choose_one_expert_mask_router_no_bpr(self):
    num_groups = 2
    tokens_per_group = 3
    hidden_dim = 4
    num_experts = 2
    num_selected_experts = 1  # Switch routing case
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseMaskedRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=False,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )

    @jax.jit
    def _run_forward(token_inputs):
      router_mask, _ = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_mask

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_mask = _run_forward(token_inputs)

    expected_mask = jnp.array([
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
    ], dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.51171875], [0.]],
            [[0.], [0.50390625]],
            [[0.], [0.]],
        ],
        [
            [[0.50390625], [0.]],
            [[0.], [0.5078125]],
            [[0.], [0.]],
        ],
    ], dtype=jnp.float32)
    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 1.0015408)
    self.assertEqual(router_mask.router_z_loss, 0.47536469)

  def test_tokens_choose_multiple_experts_mask_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 3
    num_selected_experts = 2
    expert_capacity = 1
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.TokensChooseMaskedRouter(
        router_weights=router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
        dtype=jnp.float32,
    )

    @jax.jit
    def _run_forward(token_inputs):
      router_mask, _ = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_mask

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_mask = _run_forward(token_inputs)

    expected_mask = jnp.array([
        [
            [[True], [False], [True]],
            [[False], [True], [False]],
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
        [
            [[True], [True], [False]],
            [[False], [False], [True]],
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
    ], dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.33203125], [0.], [0.3359375]],
            [[0.], [0.3359375], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ],
        [
            [[0.33007812], [0.34179688], [0.]],
            [[0.], [0.], [0.3359375]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ],
    ], dtype=jnp.float32)

    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 2.0019858)
    self.assertEqual(router_mask.router_z_loss, 1.2701721)

  def test_experts_choose_mask_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 2
    expert_capacity = 2
    router_weights = routing.RouterWeights(
        kernel_shardings=_KERNEL_SHARDINGS, name='router_weights')
    router_layer = routing.ExpertsChooseMaskedRouter(
        router_weights=router_weights,
        jitter_noise=0.,
        dtype=jnp.float32,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
    )

    @jax.jit
    def _run_forward(token_inputs):
      router_mask, _ = router_layer.init_with_output(
          {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
          token_inputs, num_experts, expert_capacity)
      return router_mask

    with _create_global_mesh():
      token_inputs = jax.random.uniform(
          jax.random.key(0),
          (num_groups, tokens_per_group, hidden_dim),
          minval=0, maxval=1)
      router_mask = _run_forward(token_inputs)

    expected_mask = jnp.array([
        [
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ],
        [
            [[1, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [0, 0]],
        ],
    ], dtype=jnp.int32)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0., 0.49609375], [0., 0.]],
            [[0., 0.], [0.5078125, 0.]],
            [[0.49804688, 0.], [0., 0.]],
            [[0., 0.], [0., 0.5078125]],
        ],
        [
            [[0.49609375, 0.], [0., 0.]],
            [[0., 0.49609375], [0.5078125, 0.]],
            [[0., 0.], [0., 0.5078125]],
            [[0., 0.], [0., 0.]],
        ],
    ], dtype=jnp.float32)

    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    # Auxiliary loss is always 0. for experts choose tokens routing.
    self.assertEqual(router_mask.auxiliary_loss, 0.)
    self.assertEqual(router_mask.router_z_loss, 0.50351524)

  def test_scatter_and_mask_dispatch_equal(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 3
    num_selected_experts = 1
    expert_capacity = 2
    rng = jax.random.key(0)

    router_weights = routing.RouterWeights(name='router_weights')

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)

    router_mask, _ = routing.TokensChooseMaskedRouter(
        router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS).init_with_output(
            jax.random.key(0), token_inputs, num_experts, expert_capacity)
    # Manipulate masked router dispatch and combine arrays to match format of
    # scatter router output.
    # Ignore capacity. Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
    masked_router_says_dispatched = jnp.max(router_mask.dispatch_mask, axis=-1)
    # Ignore particular expert and capacity for combine array.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP]
    masked_router_combine_array = jnp.max(
        router_mask.combine_array, axis=(-1, -2))

    router_indices, _ = routing.TokensChooseScatterRouter(
        router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS).init_with_output(
            jax.random.key(0), token_inputs, num_experts, expert_capacity)
    # Manipulate scatter router dispatch and combine indices to match format of
    # masked router output.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
    successfully_routed = router_indices.dispatch_indices[Ellipsis,
                                                          1] < expert_capacity
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
    scatter_router_says_dispatched = successfully_routed * jax.nn.one_hot(
        router_indices.dispatch_indices[Ellipsis, 0].squeeze(axis=-1), num_experts)
    # Remove trivial selected expert axis.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP].
    scatter_router_combine_array = router_indices.combine_weights.squeeze(
        axis=-1)

    np.testing.assert_allclose(masked_router_says_dispatched,
                               scatter_router_says_dispatched)
    np.testing.assert_allclose(masked_router_combine_array,
                               scatter_router_combine_array)
    np.testing.assert_allclose(router_mask.auxiliary_loss,
                               router_indices.auxiliary_loss)
    np.testing.assert_allclose(router_mask.router_z_loss,
                               router_indices.router_z_loss)

  def test_routers_ignore_padding(self):
    num_groups = 2
    tokens_per_group = 6
    hidden_dim = 2
    num_experts = 2
    num_selected_experts = 2
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    rng = jax.random.key(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    # Simulate masked inputs.
    padding_mask = jax.random.randint(
        rng, (num_groups, tokens_per_group, 1), minval=0, maxval=2)
    token_inputs *= padding_mask

    router_weights = routing.RouterWeights(name='router_weights')

    mesh = _create_global_mesh()

    with self.subTest(name='tokens_choose_masked_router'):
      router_layer = routing.TokensChooseMaskedRouter(
          router_weights=router_weights,
          num_selected_experts=num_selected_experts,
          jitter_noise=0.,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          router_kwargs={},
          inputs_shardings=_INPUTS_SHARDINGS,
          logits_shardings=_LOGITS_SHARDINGS,
          dtype=jnp.float32,
      )

      @jax.jit
      def _run_forward(token_inputs):
        router_mask, _ = router_layer.init_with_output(
            {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
            token_inputs, num_experts, expert_capacity)
        return router_mask

      with mesh:
        router_mask = _run_forward(token_inputs)

      expected_mask = jnp.array([
          [
              [[False], [False]],
              [[True], [True]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
          ],
          [
              [[False], [False]],
              [[True], [True]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
          ],
      ], dtype=jnp.bool_)

      np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

      expected_weights = jnp.array([
          [
              [[0.], [0.]],
              [[0.5019608], [0.49803922]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
          ],
          [
              [[0.], [0.]],
              [[0.503937], [0.4940945]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
          ],
      ], dtype=jnp.float32)

      np.testing.assert_allclose(router_mask.combine_array, expected_weights)

      self.assertEqual(router_mask.auxiliary_loss, 0.6936807)
      self.assertEqual(router_mask.router_z_loss, 0.48541257)

    with self.subTest(name='tokens_choose_scatter_router'):
      router_layer = routing.TokensChooseScatterRouter(
          router_weights=router_weights,
          num_selected_experts=num_selected_experts,
          jitter_noise=0.,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          router_kwargs={},
          inputs_shardings=_INPUTS_SHARDINGS,
          logits_shardings=_LOGITS_SHARDINGS,
          dtype=jnp.float32,
      )

      @jax.jit
      def _run_forward(token_inputs):  # pylint: disable=function-redefined
        router_indices, _ = router_layer.init_with_output(
            {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
            token_inputs, num_experts, expert_capacity)
        return router_indices

      with mesh:
        router_indices = _run_forward(token_inputs)

      expected_indices = jnp.array([
          [
              [[0, 4], [1, 4]],
              [[0, 0], [1, 0]],
              [[0, 1], [1, 1]],
              [[0, 5], [1, 5]],
              [[0, 2], [1, 2]],
              [[0, 3], [1, 3]],
          ],
          [
              [[0, 3], [1, 3]],
              [[0, 0], [1, 0]],
              [[0, 2], [1, 2]],
              [[0, 4], [1, 4]],
              [[0, 1], [1, 1]],
              [[0, 5], [1, 5]],
          ],
      ], dtype=jnp.int32)

      np.testing.assert_allclose(router_indices.dispatch_indices,
                                 expected_indices)

      expected_weights = jnp.array([
          [
              [0., 0.],
              [0.5019608, 0.49803922],
              [0., 0.],
              [0., 0.],
              [0., 0.],
              [0., 0.],
          ],
          [
              [0., 0.],
              [0.503937, 0.4940945],
              [0., 0.],
              [0., 0.],
              [0., 0.],
              [0., 0.],
          ],
      ], dtype=jnp.float32)
      np.testing.assert_allclose(router_indices.combine_weights,
                                 expected_weights)

      self.assertEqual(router_indices.auxiliary_loss, 1.165357)
      self.assertEqual(router_indices.router_z_loss, 0.48541257)

    with self.subTest(name='experts_choose_masked_router'):
      router_layer = routing.ExpertsChooseMaskedRouter(
          router_weights=router_weights,
          jitter_noise=0.,
          ignore_padding_tokens=True,
          router_kwargs={},
          inputs_shardings=_INPUTS_SHARDINGS,
          logits_shardings=_LOGITS_SHARDINGS,
          dtype=jnp.float32,
      )

      @jax.jit
      def _run_forward(token_inputs):  # pylint: disable=function-redefined
        router_mask, _ = router_layer.init_with_output(
            {'params': jax.random.key(0), 'jitter': jax.random.key(0)},
            token_inputs, num_experts, expert_capacity)
        return router_mask

      with mesh:
        router_mask = _run_forward(token_inputs)

      expected_mask = jnp.array([
          [
              [[0], [0]],
              [[1], [1]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
          ],
          [
              [[0], [0]],
              [[1], [0]],
              [[0], [1]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
          ],
      ], dtype=jnp.bool_)

      np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

      expected_weights = jnp.array([
          [
              [[0.], [0.]],
              [[0.5019608], [0.49803922]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
          ],
          [
              [[0.], [0.]],
              [[0.503937], [0.]],
              [[0.], [0.49803922]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
          ],
      ], dtype=jnp.float32)
      np.testing.assert_allclose(router_mask.combine_array, expected_weights)

      self.assertEqual(router_mask.auxiliary_loss, 0.)
      self.assertEqual(router_mask.router_z_loss, 0.48541257)

  def test_router_z_loss(self):
    num_groups = 2
    num_tokens = 6
    num_experts = 4
    rng = jax.random.key(0)

    router_logits = jax.random.uniform(
        rng, (num_groups, num_tokens, num_experts), minval=-5, maxval=5)

    self.assertEqual(routing._router_z_loss(router_logits), 13.786719)


class NormalizedRouterWeightsTest(absltest.TestCase):

  def test_init(self):
    weights = routing.NormalizedRouterWeights()
    x = jnp.zeros((1, 3, 5), jnp.float32)
    variables = weights.init(jax.random.key(0), x, num_slots=4)
    self.assertIn('params', variables)
    unboxed_params = nn.unbox(variables['params'])
    expected_params_shape = {
        'mu': jax.ShapeDtypeStruct((5, 4), jnp.float32),
        'scale': jax.ShapeDtypeStruct((), jnp.float32),
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_params, expected_params_shape)

  def test_apply(self):
    # Each token in token_inputs has norm 1, except for the last one, which
    # after normalization will be have a value of 1.
    token_inputs = jnp.asarray([[[-1.0], [1.0], [0.5]]])
    variables = {
        'params': {
            'mu': jnp.asarray([[-1.0, 1.0]]),  # Each column has norm 1.
            'scale': jnp.asarray(2.0),
        }
    }
    logits = routing.NormalizedRouterWeights().apply(variables, token_inputs, 2)
    expected_logits = jnp.asarray([[[2.0, -2.0], [-2.0, 2.0], [-2, 2.0]]])
    chex.assert_trees_all_close(logits, expected_logits)


class SoftRouterTest(absltest.TestCase):

  def test_apply(self):
    router_weights = mock.create_autospec(routing.RouterWeights, instance=True)
    router_weights.return_value = jnp.log(
        jnp.asarray([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    )
    # Expected output dispatch and combine weights, given the previous router
    # weights.
    dispatch_weights = jnp.asarray([[
        [[1 / 6, 2 / 8], [3 / 10, 4 / 12]],
        [[5 / 6, 6 / 8], [7 / 10, 8 / 12]],
    ]])
    combine_weights = jnp.asarray([[
        [[1 / 10, 2 / 10], [3 / 10, 4 / 10]],
        [[5 / 26, 6 / 26], [7 / 26, 8 / 26]],
    ]])
    router = routing.SoftRouter(
        router_weights,
        jitter_noise=0.0,
        dtype=jnp.float32,
        ignore_padding_tokens=False,
        router_kwargs={},
        inputs_shardings=_INPUTS_SHARDINGS,
        logits_shardings=_LOGITS_SHARDINGS,
    )
    token_inputs = jnp.zeros((1, 2, 8))  # Not really used.
    num_experts, expert_capacity = 2, 2
    router_mask = router.apply(
        {'params': {}},
        token_inputs,
        num_experts,
        expert_capacity,
    )
    self.assertIsInstance(router_mask, routing.RouterMask)
    chex.assert_trees_all_close(router_mask.dispatch_mask, dispatch_weights)
    chex.assert_trees_all_close(router_mask.combine_array, combine_weights)


if __name__ == '__main__':
  absltest.main()
