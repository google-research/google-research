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

"""Tests for collective einsum implementations."""

from absl.testing import absltest
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import collectives

X_SIZE = 8
NUM_LAYERS = 2
LAYER = 1

# pylint:disable = invalid-name


def mesh4():
  devices = jax.devices()
  assert len(devices) == X_SIZE
  return Mesh(devices, ('x',))


def make(shape):
  x = jnp.float32(jnp.arange(np.prod(shape)))
  return jnp.reshape(x, shape)


class CollectivesTest(absltest.TestCase):

  def test_matmul_reducescatter_no_collective(self):
    # matmul_reducescatter. Basic form is:
    #   [a, b.X] @ [b.X, c] -> [a, c.X]
    #
    # To express this in hard xmap we split out the dimensions that will be
    # sharded, so it becomes:
    #
    #   [a, B.X, b] @ [B.X, b, C, c] -> [a, C.X, c]
    lhs = make((3, X_SIZE, 2))
    rhs = make((NUM_LAYERS, X_SIZE, 2, X_SIZE, 5))
    expected = jnp.einsum('aBb,BbCc->aCc', lhs, rhs[LAYER])
    expected = jax.device_get(expected)

    def fn(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'ab,bCc->aCc', lhs, rhs, scatter_axis=1, axis_name='x', layer=LAYER
      )

    y = shard_map(
        lambda lhs_batch, rhs_batch: fn(
            jnp.squeeze(lhs_batch, axis=1), jnp.squeeze(rhs_batch, axis=1)
        ),
        mesh4(),
        in_specs=(P(None, 'x'), P(None, 'x')),
        out_specs=P(None, 'x'),
    )(lhs, rhs)
    y = jax.device_get(y)
    np.testing.assert_allclose(expected, y)

  def test_allgather_matmul_one_way(self):

    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, H, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, Y, Z, B, T, H // (X * Y * Z), D))
    rhs = rhs.reshape((X, Y, Z, L, H // (Y * Z), D, E // X))

    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    expected = shard_map(
        lambda lhs_block, rhs_block: matmul_allgather_no_collective(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    def allgather_matmul_one_way(lhs, rhs):
      return collectives.allgather_matmul_one_way(
          'bthd,hde->bte',
          lhs,
          rhs,
          rhs_split_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0)

    result = shard_map(
        lambda lhs_block, rhs_block: allgather_matmul_one_way(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh=mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def allgather_matmul_throughput(self):
    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, H, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, Y, Z, B, T, H // (X * Y * Z), D))
    rhs = rhs.reshape((X, Y, Z, L, H // (Y * Z), D, E // X))

    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    expected = shard_map(
        lambda lhs_block, rhs_block: matmul_allgather_no_collective(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    def allgather_matmul_throughput(lhs, rhs):
      rhs = collectives.preshuffle_for_allgather_matmul_throughput(
          rhs, shuffle_axis=1, axis_name='x')
      return collectives.allgather_matmul_throughput(
          'bthd,hde->bte',
          lhs,
          rhs, (0, None),
          'x',
          layer=0,
          layer_axis=0,
          subsplit_axis=2)

    result = shard_map(
        lambda lhs_block, rhs_block: allgather_matmul_throughput(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh=mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_allgather_matmul_latency(self):
    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, H, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, Y, Z, B, T, H // (X * Y * Z), D))
    rhs = rhs.reshape((L, Y, Z, H // (Y * Z), D, X, E // X))

    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    expected = shard_map(
        lambda lhs_spec, rhs_spec: jnp.expand_dims(
            matmul_allgather_no_collective(
                jnp.squeeze(lhs_spec, axis=(0, 1, 2)),
                jnp.squeeze(rhs_spec, axis=(1, 2, 5)),
            ),
            axis=(2, 3),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P(None, 'y', 'z', None, None, 'x', None)),
        out_specs=P('y', 'z', None, None, 'x', None),
    )(lhs, rhs)

    def allgather_matmul_latency(lhs, rhs):
      # In normal operation, you would do the shuffle in a separate
      # proprocessing call instead of with the weights
      # lhs: [B,T, H.YZX, D]
      # ag(lhs, x) -> [B, T, H.YZX, D]
      # rhs: [H.YZ, D, E.X]
      # lhs @ rhs -> [B, T, E.X] {yz unreduced}
      rhs = collectives.preshuffle_for_allgather_matmul_latency(
          rhs, shuffle_axis=1, axis_name='x')
      return collectives.allgather_matmul_latency(
          'bthd,hde->bte',
          lhs,
          rhs, 0,
          'x',
          layer=0,
          layer_axis=0,
          subsplit_axis=2)

    result = shard_map(
        lambda lhs_spec, rhs_spec: jnp.expand_dims(
            allgather_matmul_latency(
                jnp.squeeze(lhs_spec, axis=(0, 1, 2)),
                jnp.squeeze(rhs_spec, axis=(1, 2, 5)),
            ),
            axis=(2, 3),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P(None, 'y', 'z', None, None, 'x', None)),
        out_specs=P('y', 'z', None, None, 'x', None),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_matmul_reducescatter_oneway(self):
    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, B, T, D // X))
    rhs = rhs.reshape((X, Y, Z, L, H // (Y * Z), D // X, E))

    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    expected = shard_map(
        lambda lhs_block, rhs_block: matmul_reducescatter_no_collective(
            jnp.squeeze(lhs_block, axis=0),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    def matmul_reducescatter_oneway(lhs, rhs):
      return collectives.matmul_reducescatter_oneway(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0)

    result = shard_map(
        lambda lhs_block, rhs_block: matmul_reducescatter_oneway(
            jnp.squeeze(lhs_block, axis=0),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_matmul_reducescatter_throughput(self):
    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, B, T, D // X))
    rhs = rhs.reshape((X, Y, Z, L, H // (Y * Z), D // X, E))

    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    expected = shard_map(
        lambda lhs_block, rhs_block: matmul_reducescatter_no_collective(
            jnp.squeeze(lhs_block, axis=0),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    def matmul_reducescatter_throughput(lhs, rhs):
      # scatter along heads, subsplit along embed
      rhs = collectives.preshuffle_for_reducescatter_throughput(
          rhs, axis_name='x', scatter_axis=1, subsplit_axis=3)
      return collectives.matmul_reducescatter_throughput(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0,
          subsplit_axis=3)

    result = shard_map(
        lambda lhs_block, rhs_block: matmul_reducescatter_throughput(
            jnp.squeeze(lhs_block, axis=0),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_matmul_reducescatter_latency(self):
    L = 1  # num-layers
    B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    X, Y, Z = 4, 1, 1  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    lhs = jax.random.normal(key0, (B, T, D), dtype=jnp.float32)
    rhs = jax.random.normal(key1, (L, H, D, E), dtype=jnp.float32)
    lhs = lhs.reshape((X, B, T, D // X))
    rhs = rhs.reshape((L, Y, Z, H // (Y * Z), X, D // X, E))

    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    expected = shard_map(
        lambda lhs_block, rhs_block: jnp.expand_dims(
            matmul_reducescatter_no_collective(
                jnp.squeeze(lhs_block, axis=0),
                jnp.squeeze(rhs_block, axis=(1, 2, 4)),
            ),
            axis=(0, 3),
        ),
        mesh,
        in_specs=(P('x'), P(None, 'y', 'z', None, 'x', None, None)),
        out_specs=P(None, 'y', 'z', None, 'x', None, None),
    )(lhs, rhs)

    def shuffle(rhs):
      return collectives.preshuffle_for_reducescatter_latency(
          rhs, axis_name='x', scatter_axis=1
      )

    def matmul_reducescatter_latency(lhs, rhs):
      # you can do this beforehand, but this saves us doing two
      # different shard_maps in the same test. In real operation, you
      # would want to separate shuffling from calling the matmul.
      rhs = shuffle(rhs)
      return collectives.matmul_reducescatter_latency(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0,
          subsplit_axis=2)

    result = shard_map(
        lambda lhs_block, rhs_block: jnp.expand_dims(
            matmul_reducescatter_latency(
                jnp.squeeze(lhs_block, axis=0),
                jnp.squeeze(rhs_block, axis=(1, 2, 4)),
            ),
            axis=(0, 3),
        ),
        mesh,
        in_specs=(P('x'), P(None, 'y', 'z', None, 'x', None, None)),
        out_specs=P(None, 'y', 'z', None, 'x', None, None),
    )(lhs, rhs)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_matmul_collective_weights_gather_q_wi(self):
    # L = 1  # num-layers
    # B, T, H, D, E = 1, 1, 16, 128, 256  # dim sizes
    B, T, E, H, D = 8, 1, 64, 16, 64  # dim sizes
    X, Y, Z = 2, 2, 2  # slice sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    x_norm = jax.random.normal(key0, (B, T, E), dtype=jnp.float32)
    q_wi = jax.random.normal(key1, (H, E, D), dtype=jnp.float32)
    x_norm = x_norm.reshape((X, Y, Z, B // (X * Y * Z), T, E))
    q_wi = q_wi.reshape((X, Y, Z, H // (Y * Z), E // X, D))

    # [batch.XYZ, t, e] @ [heads.YZ, e.X, q_wi_per_head]
    #  -> [batch.XYZ, t, h, q_wi_per_head]
    def baseline_q_wi(x_norm, q_wi):
      gathered_weights = jax.lax.all_gather(q_wi, 'x', axis=1, tiled=True)
      gathered_weights = jax.lax.all_gather(
          gathered_weights, ('y', 'z'), axis=0, tiled=True)
      q_wi = jnp.einsum('bte,hed->bthd', x_norm, gathered_weights)
      return q_wi

    expected = shard_map(
        lambda lhs_block, rhs_block: baseline_q_wi(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(x_norm, q_wi)

    def test_q_wi(x_norm, q_wi):
      return collectives.matmul_collective_weights_gather_q_wi(
          'bte,hed->bthd',
          x_norm,
          q_wi,
          lhs_split_axis=2,
      )

    result = shard_map(
        lambda lhs_block, rhs_block: test_q_wi(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(x_norm, q_wi)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)

  def test_matmul_collective_weights_gather_o_wo(self):
    X, Y, Z = 2, 2, 2  # slice sizes
    B, T, E, H, D = 8, 1, 64, 16, 64  # dim sizes
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    x_norm = jax.random.normal(key0, (B, T, H, D), dtype=jnp.float32)
    o_wo = jax.random.normal(key1, (H, D, E), dtype=jnp.float32)
    x_norm = x_norm.reshape((X, Y, Z, B // (X * Y * Z), T, H, D))
    o_wo = o_wo.reshape((X, Y, Z, H // (Y * Z), D, E // X))

    def baseline_o_wo(x_norm, o_wo):
      gathered_weights = jax.lax.all_gather(o_wo, 'x', axis=2, tiled=True)
      gathered_weights = jax.lax.all_gather(
          gathered_weights, ('y', 'z'), axis=0, tiled=True)
      o_wo = jnp.einsum('bthd,hde->bte', x_norm, gathered_weights)
      return o_wo

    expected = shard_map(
        lambda lhs_block, rhs_block: baseline_o_wo(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(x_norm, o_wo)

    def test_o_wo(x_norm, o_wo):
      return collectives.matmul_collective_weights_gather_o_wo(
          'bthd,hde->bte', x_norm, o_wo, lhs_split_axis=2
      )

    result = shard_map(
        lambda lhs_block, rhs_block: test_o_wo(
            jnp.squeeze(lhs_block, axis=(0, 1, 2)),
            jnp.squeeze(rhs_block, axis=(0, 1, 2)),
        ),
        mesh,
        in_specs=(P('x', 'y', 'z'), P('x', 'y', 'z')),
        out_specs=P('x', 'y', 'z'),
    )(x_norm, o_wo)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)


if __name__ == '__main__':
  absltest.main()
