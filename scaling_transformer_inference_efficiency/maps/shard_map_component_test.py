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

"""Test xmap and pjit layer equivalency.

Separate from inference test as this uses 8 devices.
"""
import functools

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.maps import shard_map

jax.config.update('jax_array', True)  # required for jax < 0.4.0
# jax.config.update('experimental_xmap_spmd_lowering', True)
# jax.config.update('experimental_xmap_spmd_lowering_manual', True)


# pylint: disable = line-too-long
def print_pjit_info(x_pjit, x_sharding, params_pjit, params_sharding,
                    name_assist):

  def print_info(name, param, sharding):
    print(name, param.shape, sharding)

  _ = jax.tree_map(print_info, name_assist, params_pjit, params_sharding)
  print('x', x_pjit.shape, x_sharding)


def materialise_host_tensors(key, batch, seqlen, h, dtype):
  key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
  x = jax.random.normal(k1, (batch, seqlen, h.embed), dtype)
  q_wi = jax.random.normal(k2, (h.layers, h.heads, h.embed, h.q_wi_per_head),
                           dtype)
  kv = jax.random.normal(k3, (h.layers, h.embed, 1, 2 * h.qkv), dtype)
  o_wo = jax.random.normal(k4, (h.layers, h.heads, h.o_wo_per_head, h.embed),
                           dtype)
  embedding = jax.random.normal(k5, (h.vocab, h.embed), dtype)
  sin = jnp.ones((h.max_len, h.qkv // 2), dtype)
  cos = jnp.ones((h.max_len, h.qkv // 2), dtype)

  return x, weights.Weights(weights.Layer(q_wi, kv, o_wo), sin, cos, embedding)


def create_test_weights(dtype):
  X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name
  assert len(jax.devices()) == X * Y * Z
  devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))

  key_0 = jax.random.PRNGKey(0)
  seqlen = 1
  batch = 4
  h = checkpoint.HParams(
      layers=8, embed=8, ff=32, heads=16, qkv=4, max_len=256, vocab=32)

  # checks for our sharding pattern
  assert h.heads % (Y * Z) == 0
  assert h.embed % X == 0
  assert batch % Z == 0
  assert h.embed % (X * Y) == 0
  assert h.heads % (Y * Z * X) == 0

  x_pjit, params_pjit = materialise_host_tensors(key_0, batch, seqlen, h, dtype)
  x_sharding, params_sharding = P(
      'residual_batch', 'residual_time',
      'residual_embed'), weights.Weights.logical_axes()
  return mesh, x_pjit, x_sharding, params_pjit, params_sharding


def fold_out_to_per_device(mesh, x_pjit, x_sharding, params_pjit,
                           params_sharding):
  fold_out_for_mesh = functools.partial(shard_map.fold_out, mesh)
  x_xmap, x_layout = fold_out_for_mesh(x_pjit, x_sharding)
  folded_out = jax.tree_map(fold_out_for_mesh, params_pjit, params_sharding)
  params_xmap, params_layouts = shard_map.unzip_tree(params_pjit, folded_out)
  return x_xmap, x_layout, params_xmap, params_layouts


class MapsTest(absltest.TestCase):

  def test_map_to_and_from(self):

    mesh, x_pjit, x_sharding, params_pjit, params_sharding = create_test_weights(
        dtype=jnp.float32)
    name_assist = weights.Weights(
        weights.Layer('q_wi', 'kv', 'o_wo'), 'sin', 'cos', 'embedding')
    print('---- These are our initial shapes and sharding definitions ----')
    print_pjit_info(x_pjit, x_sharding, params_pjit, params_sharding,
                    name_assist)
    print('--- This shows how we can change to and from hard xmap ---')
    _, _, params_xmap, params_layouts = fold_out_to_per_device(
        mesh, x_pjit, x_sharding, params_pjit, params_sharding)
    print('Params pjit: ', jax.tree_map(jnp.shape, params_pjit))
    print('Params xmap: ', jax.tree_map(jnp.shape, params_xmap))
    print('Layouts: ', params_layouts)
    folded_in_params = jax.tree_map(shard_map.fold_in, params_xmap,
                                    params_sharding)
    print('Back to pjit: ', jax.tree_map(jnp.shape, folded_in_params))
    equality_checks = jax.tree_map(lambda a, b: (a == b).all(), params_pjit,
                                   folded_in_params)
    equality_checks, _ = jax.tree_util.tree_flatten(equality_checks)
    assert False not in equality_checks

  def test_addition_equivalence(self):

    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(partitioning.AttnAllToAll.NONE))
    with rules:
      mesh, x_pjit, x_sharding, params_pjit, params_sharding = create_test_weights(
          dtype=jnp.float32)
      x_xmap, x_layout, _, _ = fold_out_to_per_device(mesh, x_pjit, x_sharding,
                                                      params_pjit,
                                                      params_sharding)

      @functools.partial(pjit)
      def add_one_pjit(x):
        x = partitioning._with_sharding_constraint(x, x_sharding)
        return x + 1

      @functools.partial(
          xmap,
          in_axes=x_layout,
          out_axes=x_layout,
          axis_resources={
              'x': 'x',
              'y': 'y',
              'z': 'z'
          })
      def add_one_xmap(x):
        return x + 1

      with mesh:
        y1 = add_one_pjit(x_pjit)

      with mesh:
        y2 = add_one_xmap(x_xmap)
      y2 = shard_map.fold_in(y2, x_sharding)

      assert (y1 == y2).all()

  def test_matmul_equivalence(self):
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(partitioning.AttnAllToAll.NONE))
    with rules:
      mesh, x_pjit, x_sharding, params_pjit, params_sharding = create_test_weights(
          dtype=jnp.float32)
      (x_sharding,
       params_sharding) = jax.tree_map(partitioning.logical_to_physical,
                                       (x_sharding, params_sharding))
      out_sharding = jax.tree_map(partitioning.logical_to_physical,
                                  (P('batch', 'time', 'post_norm_embed'),
                                   P('batch', 'time', 'heads', 'qkv')))

      @functools.partial(pjit)
      def matmul_pjit(x, params):
        # x: ['batch.Z', 'time', 'embed.XY']
        x = partitioning._with_sharding_constraint(x, x_sharding)
        # q_wi[0]: ['heads.YZ', 'embed.X', 'qkv']
        # ['batch.Z', 'time', 'embed.XY'] @ ['heads.YZ', 'embed.X', 'qkv']
        # all_gather LHS on YZ
        # ['batch', 'time', 'embed.X'] @ ['heads.YZ', 'embed.X', 'qkv']
        #  ----> ['batch', 'time', 'heads.YZ', 'qkv'] {unreduced.X}
        # reduce_scatter on X
        y = jnp.einsum('bte,hed->bthd', x, params.layer.q_wi[0])
        # -----> ['batch', 'time', 'heads.XYZ', 'qkv']
        y = partitioning._with_sharding_constraint(
            y, P('batch', 'time'
                 'heads', 'qkv'))
        return x, y

      def matmul_xmap(x, params):
        # x: ['batch.Z', 'time', 'embed.XY']
        # ['heads.YZ', 'embed.X', 'qkv']
        q_wi_1 = params.layer.q_wi[0]

        x = lax.all_gather(
            x, 'y', axis=2, tiled=True)  # ['batch.Z', 'time', 'embed.X']
        x = lax.all_gather(
            x, 'z', axis=0, tiled=True)  # ['batch', 'time', 'embed.X']

        # ['batch', 'time', 'embed.X'] @ ['heads.YZ', 'embed.X', 'qkv']
        #  ----> ['batch', 'time', 'heads.YZ', 'qkv'] {unreduced.X}

        y = jnp.einsum('bte,hed->bthd', x, q_wi_1)
        # print(y.shape)
        # -----> ['batch', 'time', 'heads.YZX', 'qkv']
        y = lax.psum_scatter(y, 'x', scatter_dimension=2, tiled=True)

        return x, y

      matmul_shardmap = shard_map.shard_map(
          matmul_xmap,
          mesh=mesh,
          in_specs=(x_sharding, params_sharding),
          out_specs=out_sharding)
      with mesh:
        x1, y1 = matmul_pjit(x_pjit, params_pjit)
        x2, y2 = matmul_shardmap(x_pjit, params_pjit)

      np.testing.assert_allclose(x1, x2, rtol=1e-04, atol=1e-04)
      np.testing.assert_allclose(y1, y2, rtol=1e-04, atol=1e-04)


if __name__ == '__main__':
  absltest.main()
