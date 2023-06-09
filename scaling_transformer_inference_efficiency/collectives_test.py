# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

import functools

from absl.testing import absltest
import jax
from jax.experimental.maps import xmap
import jax.numpy as jnp
from jax.sharding import Mesh
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
      result = collectives.matmul_reducescatter_no_collective(
          'ab,bCc->aCc',
          lhs,
          rhs,
          scatter_axis=1,
          axis_name='x',
          layer=LAYER)
      # Squeeze out C.X dim, which is now 1 on each device.
      result = jax.lax.squeeze(result, (1,))
      return result

    with mesh4():
      y = xmap(
          fn,
          in_axes=({
              1: 'x'
          }, {
              1: 'x'
          }),
          out_axes={1: 'x'},
          axis_resources={'x': 'x'})(lhs, rhs)
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

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    with mesh:
      expected = matmul_allgather_no_collective(lhs, rhs)

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def allgather_matmul_one_way(lhs, rhs):
      return collectives.allgather_matmul_one_way(
          'bthd,hde->bte',
          lhs,
          rhs,
          rhs_split_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0)

    with mesh:
      result = allgather_matmul_one_way(lhs, rhs)

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

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    with mesh:
      expected = matmul_allgather_no_collective(lhs, rhs)

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
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

    with mesh:
      result = allgather_matmul_throughput(lhs, rhs)

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

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], [None, 'y', 'z', None, None, 'x', None]),
        out_axes=['y', 'z', None, None, 'x', None],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_allgather_no_collective(lhs, rhs):
      return collectives.matmul_allgather_no_collective(
          'bthd,hde->bte', lhs, rhs, 2, 'x', layer=0, layer_axis=0)

    with mesh:
      expected = matmul_allgather_no_collective(lhs, rhs)

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], [None, 'y', 'z', None, None, 'x', None]),
        out_axes=['y', 'z', None, None, 'x', None],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
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

    with mesh:
      result = allgather_matmul_latency(lhs, rhs)

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

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    with mesh:
      expected = matmul_reducescatter_no_collective(lhs, rhs)

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_reducescatter_oneway(lhs, rhs):
      return collectives.matmul_reducescatter_oneway(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=0,
          axis_name='x',
          layer=0,
          layer_axis=0)

    with mesh:
      result = matmul_reducescatter_oneway(lhs, rhs)

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

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    with mesh:
      expected = matmul_reducescatter_no_collective(lhs, rhs)

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
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

    with mesh:
      result = matmul_reducescatter_throughput(lhs, rhs)

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

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], [None, 'y', 'z', None, 'x', None, None]),
        out_axes=[None, 'y', 'z', None, 'x', None, None],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_reducescatter_no_collective(lhs, rhs):
      return collectives.matmul_reducescatter_no_collective(
          'btd,hde->bthe',
          lhs,
          rhs,
          scatter_axis=2,
          axis_name='x',
          layer=0,
          layer_axis=0)

    with mesh:
      expected = matmul_reducescatter_no_collective(lhs, rhs)

    def shuffle(rhs):
      return collectives.preshuffle_for_reducescatter_latency(
          rhs, axis_name='x', scatter_axis=1)

    @functools.partial(
        xmap,
        in_axes=(['x', Ellipsis], [None, 'y', 'z', None, 'x', None, None]),
        out_axes=[None, 'y', 'z', None, 'x', None, None],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def matmul_reducescatter_latency(lhs, rhs):
      # you can do this beforehand, but this saves us doing two
      # different xmaps in the same test. In real operation, you
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

    with mesh:
      result = matmul_reducescatter_latency(lhs, rhs)

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
    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def baseline_q_wi(x_norm, q_wi):
      gathered_weights = jax.lax.all_gather(q_wi, 'x', axis=1, tiled=True)
      gathered_weights = jax.lax.all_gather(
          gathered_weights, ('y', 'z'), axis=0, tiled=True)
      q_wi = jnp.einsum('bte,hed->bthd', x_norm, gathered_weights)
      return q_wi

    with mesh:
      expected = baseline_q_wi(x_norm, q_wi)

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def test_q_wi(x_norm, q_wi):
      y = collectives.matmul_collective_weights_gather_q_wi(
          'bte,hed->bthd',
          x_norm,
          q_wi,
          lhs_split_axis=2,
      )
      return y

    with mesh:
      result = test_q_wi(x_norm, q_wi)

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

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def baseline_o_wo(x_norm, o_wo):
      gathered_weights = jax.lax.all_gather(o_wo, 'x', axis=2, tiled=True)
      gathered_weights = jax.lax.all_gather(
          gathered_weights, ('y', 'z'), axis=0, tiled=True)
      o_wo = jnp.einsum('bthd,hde->bte', x_norm, gathered_weights)
      return o_wo

    with mesh:
      expected = baseline_o_wo(x_norm, o_wo)

    @functools.partial(
        xmap,
        in_axes=(['x', 'y', 'z', Ellipsis], ['x', 'y', 'z', Ellipsis]),
        out_axes=['x', 'y', 'z', Ellipsis],
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def test_o_wo(x_norm, o_wo):
      result = collectives.matmul_collective_weights_gather_o_wo(
          'bthd,hde->bte',
          x_norm,
          o_wo,
          lhs_split_axis=2)
      return result

    with mesh:
      result = test_o_wo(x_norm, o_wo)

    np.testing.assert_allclose(expected, result, rtol=1e-03, atol=1e-04)


if __name__ == '__main__':
  absltest.main()
