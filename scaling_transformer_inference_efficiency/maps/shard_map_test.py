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

"""Test shard map."""

from dataclasses import dataclass  # pylint: disable = g-importing-member
from functools import partial  # pylint: disable = g-importing-member

from absl.testing import absltest
from flax import struct
import jax
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import special2


shard_map = partial(shard_map, check_rep=False)


def create_inputs(a_sharding, b_sharding, B=8):  # pylint: disable = invalid-name
  X, Y, Z = 2, 2, 2  # pylint: disable=invalid-name
  devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
  E, F = 8, 8  # pylint: disable=invalid-name
  a = jax.device_put(
      jnp.arange(B * E).reshape((B, E)),
      jax.sharding.NamedSharding(mesh, a_sharding))
  b = jax.device_put(
      jnp.arange(E * F).reshape((E, F)),
      jax.sharding.NamedSharding(mesh, b_sharding))
  # a: [B.z, E.xy], b: [E.y, F]
  return mesh, a, b


def dynamic_index_and_slice(index_axis, index, slice_axis,
                            slice_start, slice_length,
                            x):
  """Does multi axis slicing."""
  assert index_axis != slice_axis, f'{index_axis} != {slice_axis}'
  sizes = list(x.shape)
  starts = [0] * len(sizes)
  starts[index_axis] = index
  starts[slice_axis] = slice_start
  sizes[index_axis] = 1
  sizes[slice_axis] = slice_length
  x = lax.dynamic_slice(x, starts, sizes)
  x = lax.squeeze(x, [index_axis])
  return x


# pylint: disable = invalid-name
def matmul_reducescatter_oneway(
    einsum_spec,
    lhs,
    rhs,
    rhs_scatter_axis,
    axis_name,
    layer,
    subsplit_axis,
    axis_size,
    layer_axis=0,
    REPRO_BUG=False,
):
  """Only for debugging XLA compilation issues.
  """
  del subsplit_axis, axis_name, layer_axis, layer
  # [batch, maxlen, dmodel.X] @ [heads.YZ, dmodel.X, q_wi_per_head]
  # -> (matmul)
  # -> [batch, maxlen, heads.YZ, q_wi_per_head]{X unreduced}
  # -> (reducescatter over X into heads)
  # -> [batch, maxlen, heads.YZX, q_wi_per_head]

  chunk_index = 0
  chunk_size = rhs.shape[rhs_scatter_axis] // axis_size
  first_chunk = lax.dynamic_slice_in_dim(rhs, chunk_index * chunk_size,
                                         chunk_size, rhs_scatter_axis)

  p = jnp.einsum(einsum_spec, lhs, first_chunk)
  accum = jnp.zeros(p.shape, dtype=lhs.dtype)

  def collective_matmul(i, carrys):
    del i  # this is just to test loops under XLA
    accum, p = carrys
    c = lax.dynamic_slice_in_dim(rhs, chunk_index * chunk_size, chunk_size,
                                 rhs_scatter_axis)
    if REPRO_BUG:
      p = jnp.einsum(einsum_spec, lhs, c)

    return accum, p

  accum, p = jax.lax.fori_loop(1, 2, collective_matmul, (accum, p))

  return accum + p


class ShardMapTest(absltest.TestCase):

  # [B.z, E.xy] -> [B.z, E.xy]
  def test_identity(self):

    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))

    assert a.addressable_data(0).shape == (4, 2)

    def identity(x):
      return x

    def fwd(a):
      c = shard_map(
          identity,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P('z', ('x', 'y')))(
              a)
      return c

    with mesh:
      c = jax.jit(fwd)(a)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('z', ('x', 'y')),),
      #     out_shardings=P('z', ('x', 'y')))(a)
    assert c.addressable_data(0).shape == (4, 2)

  #   ###############################################################################

  # [B.z, E.xy] -> [B, E.xy]

  def test_all_gather(self):

    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))

    assert a.addressable_data(0).shape == (4, 2)

    def all_gather(x):
      return lax.all_gather(x, 'z', axis=0, tiled=True)

    # @jax.jit
    def fwd(a):
      c = shard_map(
          all_gather,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P(None, ('x', 'y')))(
              a)
      return c

    with mesh:
      c = jax.jit(fwd)(a)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('z', ('x', 'y')),),
      #     out_shardings=P(None, ('x', 'y')))(a)
    assert c.addressable_data(0).shape == (8, 2)

  ##########################################################################

  # # [B.z, E.y] @ [E.y, F] -> RS(y, 0) -> [B.zy, F]

  def test_matmul_partial(self):

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))

    assert a.addressable_data(0).shape == (4, 4)

    def matmul_partial(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    def fwd(a, b):
      c = shard_map(
          matmul_partial,
          mesh,
          in_specs=(P('z', 'y'), P('y', None)),
          out_specs=P('z', 'y'))(a, b)
      return c

    with mesh:
      c = jax.jit(fwd)(a, b)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('z', 'y'), P('y', None)),
      #     out_shardings=P('z', 'y'))(a, b)
    assert c.addressable_data(0).shape == (4, 8)

  ##########################################################################

  #   # [B.z, E.y] @ [E.y, F] -> RS(y, 0) -> [B.zy, F]

  def test_matmul_reduce_scatter(self):

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))

    assert a.addressable_data(0).shape == (4, 4)

    def matmul_reduce_scatter(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    # @jax.jit
    def fwd(a, b):
      c = shard_map(
          matmul_reduce_scatter,
          mesh,
          in_specs=(P('z', 'y'), P('y', None)),
          out_specs=P(('z', 'y'), None))(a, b)
      return c

    with mesh:
      c = jax.jit(fwd)(a, b)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('z', 'y'), P('y', None)),
      #     out_shardings=P(('z', 'y'), None))(a, b)
    assert c.addressable_data(0).shape == (2, 8)

  ##########################################################################

  def test_collective_permute(self):

    X, Y, Z = 8, 1, 1
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    def collective_permute(a):
      axis_size = lax.psum(1, 'x')
      return lax.ppermute(
          a, 'x', perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

    # @jax.jit
    def fwd(a):
      c = shard_map(
          collective_permute,
          mesh,
          in_specs=(P('x', None),),
          out_specs=P('x', None))(
              a)
      return c

    with mesh:
      c = jax.jit(fwd)(a)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('x', None),),
      #     out_shardings=P('x', None))(a)
    assert (c[1, :] == a[0, :]).all()

  ##########################################################################

  def test_all_to_all(self):

    X, Y, Z = 8, 1, 1
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    a = jax.device_put(
        jnp.float32(jnp.arange(8 * 8).reshape((8, 8))),
        jax.sharding.NamedSharding(mesh, P('x', None)),
    )

    def all_to_all(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    def fwd(a):
      c = shard_map(
          all_to_all, mesh, in_specs=(P('x', None),), out_specs=P(None, 'x'))(
              a)
      return c

    def loss(a):
      return fwd(a).mean()

    def grad(a):
      return jax.value_and_grad(loss)(a)

    with mesh:
      c = jax.jit(fwd)(a)
      _, _ = jax.jit(grad)(a)
      # c = pjit(
      #     fwd,
      #     in_shardings=(P('x', None),),
      #     out_shardings=P(None, 'x'))(a)

    assert (c == jnp.reshape(a.T, (1, 64))).all()

  ###########################################################################

  def test_fwd_pass(self):

    X, Y, Z = 2, 2, 2
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    dtype = jnp.float32
    key = jax.random.PRNGKey(0)
    seqlen = 1
    batch = 4

    @dataclass
    class HParams:
      """Hyperparameters for a PaLM model."""
      layers: int
      embed: int
      ff: int
      heads: int
      qkv: int
      max_len: int  # Max length supported by attention
      vocab: int

      @property
      def q_wi_per_head(self):
        """In the fused q_wi layer, dimension size per head."""
        assert self.ff % self.heads == 0
        return (self.ff * 2 // self.heads) + self.qkv

      @property
      def o_wo_per_head(self):
        """In the fused o_wo layer, dimension size per head."""
        assert self.ff % self.heads == 0
        return (self.ff // self.heads) + self.qkv

    h = HParams(
        layers=8, embed=8, ff=32, heads=16, qkv=4, max_len=256, vocab=32)

    x_sharding = P('z', None, ('x', 'y'))
    q_wi_sharding = P(None, ('y', 'z'), 'x', None)
    kv_sharding = P(None, 'x', None, None)
    o_wo_sharding = P(None, ('y', 'z'), None, 'x')

    x = jax.device_put(
        jax.random.normal(key, (batch, seqlen, h.embed), dtype),
        jax.sharding.NamedSharding(mesh, x_sharding))
    q_wi = jax.device_put(
        jax.random.normal(key,
                          (h.layers, h.heads, h.embed, h.q_wi_per_head), dtype),
        jax.sharding.NamedSharding(mesh, q_wi_sharding))
    kv = jax.device_put(
        jax.random.normal(key, (h.layers, h.embed, 1, 2 * h.qkv), dtype),
        jax.sharding.NamedSharding(mesh, kv_sharding))
    o_wo = jax.device_put(
        jax.random.normal(key,
                          (h.layers, h.heads, h.o_wo_per_head, h.embed), dtype),
        jax.sharding.NamedSharding(mesh, o_wo_sharding))

    @struct.dataclass
    class Layer:
      q_wi: jnp.ndarray
      kv: jnp.ndarray
      o_wo: jnp.ndarray

    layer = Layer(q_wi, kv, o_wo)
    layer_sharding = Layer(q_wi_sharding, kv_sharding, o_wo_sharding)

    def fwd(hparams, x, layer, kv_cache):
      del kv_cache
      q_wi, kv, o_wo = layer.q_wi, layer.kv, layer.o_wo
      # x: ['batch.Z', 'time', 'embed.XY']
      epsilon = 1e-6
      # xgather: ['batch.Z', 'time', 'embed.X']
      xgather = lax.all_gather(x, 'y', axis=2, tiled=True)
      xgather = jnp.float32(xgather)
      # but less comms
      # print('xgather', xgather.shape)
      mean2 = lax.pmean(
          jnp.mean(lax.square(xgather), axis=-1, keepdims=True), axis_name='x')
      # xnorm_z: ['batch.Z', 'time', 'embed.X']
      xnorm_z = jnp.bfloat16(xgather * lax.rsqrt(mean2 + epsilon))

      xnorm = lax.all_gather(
          xnorm_z, 'z', axis=0, tiled=True)  # ['batch', 'time', 'embed.X']
      # print('xnorm',xnorm.shape)

      # ['batch', 'time', 'embed.X'] @ ['heads.YZ', 'embed.X', 'query']
      #  ----> ['batch', 'time', 'heads.YZ', 'query'] {unreduced.X}
      # reduceScatter(x, 2)
      # -----> ['batch', time', 'head.XYZ', 'query']
      with jax.named_scope('q_wi'):
        # -----> ['batch', 'time', 'heads.YZX', 'query']

        q_wi = collectives.matmul_reducescatter_oneway(
            'bte,hed->bthd',
            jnp.bfloat16(xnorm),
            jnp.bfloat16(q_wi),
            scatter_axis=0,
            axis_name='x',
            layer=0,
        )

        wi0 = q_wi[:, :, :,
                   hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
        # wi1: ['batch', 'time', 'heads.YZX', 'ff']
        wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]

      with jax.named_scope('kv'):
        xnorm_sliced = xnorm_z
        kv_unreduced = jnp.einsum('bte,ezd->btzd', xnorm_sliced, kv[0])

        # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
        # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
        # --slice--> [batch.ZB, maxlen, 1, 2*qkv]
        # --AGZ-->   [batch.B, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        # TODO(sholto) - confirm we no longer need
        # kv = lax.dynamic_slice_in_dim(kv, b_index*batch_zb, batch_zb, axis=0)
        kv = lax.all_gather(kv, 'z', axis=0, tiled=True)

      k = kv[:, :, 0, :hparams.qkv]
      v = kv[:, :, 0, hparams.qkv:]

      with jax.named_scope('attention'):
        # y:att [batch, maxlen, heads.YZX, qkv]
        q = q_wi[:, :, :, :hparams.qkv]
        y_att = attention.attend(q, k, v, [], 0)

      with jax.named_scope('SwiGLU'):
        # wi0: ['batch', 'time', 'heads.YZX', 'ff']

        y_mlp = special2.swish2(wi0) * wi1

      with jax.named_scope('o_wo'):
        # y_fused: ['batch', 'time', 'heads.YZX', 'ff+qkv']
        # allgatherx
        # y_fused: ['batch', 'time', 'heads.YZ', 'ff+qkv']
        # @
        #  [heads.YZ, ff+qkv, embed.x]
        # -> # ->  y_out: [batch, maxlen, embed.X]{YZ unreduced}
        y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
        # y_fused = lax.all_gather(y_fused, 'x', axis=0, tiled=True)
        # print(y_fused.shape, params.o_wo.shape)
        # y_out = jnp.einsum('bthd,hde->bte', y_fused, params.o_wo[0])
        rhs = lax.dynamic_index_in_dim(o_wo, 0, 0, keepdims=False)
        lhs = lax.all_gather(y_fused, 'x', axis=2, tiled=True)
        y_out = jnp.einsum('bthd,hde->bte', lhs, rhs)
        # y_out: [batch, maxlen, embed.XY]{Z unreduced}
        # y_out: [batch.Z, maxlen, embed.XY]
        y_out = lax.psum_scatter(y_out, 'y', scatter_dimension=2, tiled=True)
        y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=0, tiled=True)

        with jax.named_scope('residual'):
          z = jnp.float32(y_out + x)

      return z

    fwd = partial(fwd, h)  # partial w/ the hparams

    def wrapped_shardmap(layer, x, kv_caches):
      result = shard_map(
          fwd,
          mesh,
          in_specs=(x_sharding, layer_sharding, P()),
          out_specs=x_sharding,
      )(x, layer, kv_caches)
      return result

    def loss(layer, x, kv_caches):
      return wrapped_shardmap(layer, x, kv_caches).mean()

    with mesh:
      z = jax.jit(wrapped_shardmap)(layer, x, [])

      _, _ = jax.jit(jax.value_and_grad(loss))(layer, x, [])
      # z = pjit(wrapped_shardmap,
      #          in_shardings=(x_sharding, layer_sharding),
      #          out_shardings=x_sharding)(x, layer)

    return z


if __name__ == '__main__':
  absltest.main()
