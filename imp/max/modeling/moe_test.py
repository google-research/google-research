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

from imp.max.modeling import linear
from imp.max.modeling import moe
from imp.max.modeling import routing
from imp.max.utils import sharding


_TOKENS_SHARDINGS = (('expert', 'data'), None, None, None)  # (b, n, t, d)
_EXPERT_KERNEL_SHARDINGS = (None, 'model')
_EXPERT_INTERMEDIATE_SHARDINGS = ('data', 'model')


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1, 1),
                                      dcn_mesh_shape=(1, 1, 1)),
      ['expert', 'data', 'model'],
  )


class Expert(nn.Module):

  features: int

  @nn.compact
  def __call__(self, x):
    x = linear.DenseGeneral(self.features,
                            kernel_shardings=_EXPERT_KERNEL_SHARDINGS,
                            name='dense')(x)
    x = sharding.shard_array(x, _EXPERT_INTERMEDIATE_SHARDINGS)
    return x


class Identity(nn.Module):

  @nn.compact
  def __call__(self, x):
    return sharding.shard_array(x, _EXPERT_INTERMEDIATE_SHARDINGS)


class SparseMoEwithExpertTest(parameterized.TestCase):

  def test_sparse_moe(self):
    moe_layer = moe.SparseMoEwithExpert(
        num_experts=2,
        min_expert_capacity=0,
        ignore_padding_tokens=False,
        jitter_noise=0,
        comm_dtype=jnp.float32,
        split_params=False,
        optimize_parallel_comms=False,
        capacity_factor=1.0,
        max_group_size=2 * 8,  # experts * length
        strict_group_size=False,
        num_selected_experts=1,
        batch_prioritized_routing=False,
        router_type='ExpertsChooseMasked',
        router_bias=True,
        router_kwargs=(),
        expert=Expert(4),
        rng_keys=(),
        router_kernel_shardings=(),
        tokens_shardings=_TOKENS_SHARDINGS,
        model_axis_size=1,
        model_axis_name='model',
    )

    @jax.jit
    def _run_forward(inputs):
      variables = moe_layer.init(
          rngs=jax.random.key(0),
          inputs=inputs,
          deterministic_routing=True)
      outputs = moe_layer.apply(variables, inputs, deterministic_routing=True)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones((1, 2, 8, 5))
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      outputs, variables = _run_forward(inputs)

    # Assert params structure and shape
    unboxed_params = nn.unbox(variables['params'])
    expected_params_shape_dtype = {
        'router': {
            'w': {'kernel': jax.ShapeDtypeStruct((5, 2), jnp.float32),
                  'bias': jax.ShapeDtypeStruct((2,), jnp.float32)}
        },
        'expert': {
            'dense': {'kernel': jax.ShapeDtypeStruct((2, 5, 4), jnp.float32),
                      'bias': jax.ShapeDtypeStruct((2, 4), jnp.float32)}
        }
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_params, expected_params_shape_dtype)

    # Assert sharding annotations are propagated properly
    self.assertEqual(
        variables['params']['expert']['dense']['kernel'].names,
        ('expert',) + _EXPERT_KERNEL_SHARDINGS,
    )

    # Assert output shape
    expected_output_shape = (1, 2, 8, 4)
    chex.assert_shape(outputs, expected_output_shape)

  def test_wrong_shardings(self):
    moe_layer = moe.SparseMoEwithExpert(
        num_experts=2,
        min_expert_capacity=0,
        ignore_padding_tokens=False,
        jitter_noise=0,
        comm_dtype=jnp.float32,
        split_params=False,
        optimize_parallel_comms=False,
        capacity_factor=1.0,
        max_group_size=2 * 8,  # experts * length
        strict_group_size=False,
        num_selected_experts=1,
        batch_prioritized_routing=False,
        router_type='ExpertsChooseMasked',
        router_bias=True,
        router_kwargs=(),
        expert=Expert(4),
        rng_keys=(),
        router_kernel_shardings=(),
        tokens_shardings=('data', None),
        model_axis_size=1,
        model_axis_name='model',
    )
    with self.assertRaises(ValueError):
      _ = moe_layer.init(
          rngs=jax.random.key(0),
          inputs=jnp.ones((1, 2, 8, 5)),
          deterministic_routing=True)


class SoftMoEwithExpertTest(parameterized.TestCase):

  def test_soft_moe(self):
    moe_layer = moe.SoftMoEwithExpert(
        num_experts=2,
        expert_capacity=3,
        ignore_padding_tokens=False,
        jitter_noise=0,
        comm_dtype=jnp.float32,
        split_params=False,
        optimize_parallel_comms=False,
        router_kwargs=(),
        expert=Expert(4),
        rng_keys=(),
        router_kernel_shardings=(),
        tokens_shardings=_TOKENS_SHARDINGS,
        model_axis_size=1,
        model_axis_name='model',
    )

    @jax.jit
    def _run_forward(inputs):
      variables = moe_layer.init(
          rngs=jax.random.key(0),
          inputs=inputs,
          deterministic_routing=True)
      outputs = moe_layer.apply(variables, inputs, deterministic_routing=True)
      return outputs, variables

    with _create_global_mesh():
      inputs = jnp.ones((1, 2, 8, 5))
      inputs = sharding.shard_array(inputs, _TOKENS_SHARDINGS)
      outputs, variables = _run_forward(inputs)

    # Assert params structure and shape
    unboxed_params = nn.unbox(variables['params'])
    expected_params_shape_dtype = {
        'router': {
            'mu': jax.ShapeDtypeStruct((5, 2 * 3), jnp.float32),
            'scale': jax.ShapeDtypeStruct((), jnp.float32),
        },
        'expert': {
            'dense': {'kernel': jax.ShapeDtypeStruct((2, 5, 4), jnp.float32),
                      'bias': jax.ShapeDtypeStruct((2, 4), jnp.float32)}
        }
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_params, expected_params_shape_dtype)

    # Assert sharding annotations are propagated properly
    self.assertEqual(
        variables['params']['expert']['dense']['kernel'].names,
        ('expert',) + _EXPERT_KERNEL_SHARDINGS,
    )

    # Assert output shape
    expected_output_shape = (1, 2, 8, 4)
    chex.assert_shape(outputs, expected_output_shape)

  @mock.patch.object(routing.SoftRouter, '__call__', autospec=True)
  def test_router_call(self, mock_router_call):
    mock_router_call.return_value = routing.RouterMask(
        dispatch_mask=jnp.asarray([
            [[[.1], [.4]], [[.2], [.3]], [[.7], [.3]]],
            [[[.1], [.4]], [[.2], [.3]], [[.7], [.3]]],
        ]),
        combine_array=jnp.asarray([
            [[[.5], [.5]], [[.2], [.8]], [[.8], [.2]]],
            [[[.5], [.5]], [[.2], [.8]], [[.8], [.2]]],
        ]),
        auxiliary_loss=jnp.zeros((), dtype=jnp.float32),
        router_z_loss=jnp.zeros((), dtype=jnp.float32),
        router_probs=None,
    )
    layer = moe.SoftMoEwithExpert(
        num_experts=2,
        expert_capacity=1,
        ignore_padding_tokens=False,
        jitter_noise=0,
        comm_dtype=jnp.float32,
        split_params=False,
        optimize_parallel_comms=False,
        router_kwargs=(),
        expert=Identity(),
        rng_keys=(),
        router_kernel_shardings=(),
        tokens_shardings=_TOKENS_SHARDINGS,
        model_axis_size=1,
        model_axis_name='model',
    )
    # batch_size=2, instances=1, seq_length=3, hidden_dim=1.
    x = jnp.asarray([
        [[[0.], [1.], [2.]]],
        [[[3.], [4.], [5.]]],
    ])
    output = layer.apply({}, x, deterministic_routing=True)
    # Expected inputs to experts = [[[1.6], [0.9]], [[4.6], [3.9]]].
    # Each expert is the identity function, so the expected outputs are:
    expected_output = jnp.asarray([
        [[[.5 * 1.6 + .5 * 0.9], [.2 * 1.6 + .8 * 0.9], [.8 * 1.6 + .2 * 0.9]]],
        [[[.5 * 4.6 + .5 * 3.9], [.2 * 4.6 + .8 * 3.9], [.8 * 4.6 + .2 * 3.9]]],
    ])
    chex.assert_trees_all_equal_shapes_and_dtypes(output, expected_output)

  def test_wrong_shardings(self):
    moe_layer = moe.SoftMoEwithExpert(
        num_experts=2,
        expert_capacity=1,
        ignore_padding_tokens=False,
        jitter_noise=0,
        comm_dtype=jnp.float32,
        split_params=False,
        optimize_parallel_comms=False,
        router_kwargs=(),
        expert=Identity(),
        rng_keys=(),
        router_kernel_shardings=(),
        tokens_shardings=('data', None),
        model_axis_size=1,
        model_axis_name='model',
    )
    with self.assertRaises(ValueError):
      _ = moe_layer.init(
          rngs=jax.random.key(0),
          inputs=jnp.ones((1, 2, 8, 5)),
          deterministic_routing=True)

if __name__ == '__main__':
  absltest.main()
