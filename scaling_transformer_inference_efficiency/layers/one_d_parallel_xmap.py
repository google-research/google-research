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

"""1D weight stationary xmap layer."""

from typing import Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers.layers_pjit import _rope
from scaling_transformer_inference_efficiency.weights import Layer

HParams = checkpoint.HParams
CheckpointSpec = checkpoint.CheckpointSpec


# pylint: disable = protected-access
# pylint: disable = g-doc-return-or-yield
# pylint: disable = g-doc-args
# TODO(sholto): Update
def weight_stationary_simple(
    hparams,
    layer,
    params,
    sin,
    cos,
    kv_caches,
    x,
    x_axis,
    y_axis,
    z_axis,
    latency_collectives,
    intermediate_dtype = jnp.bfloat16,
):
  """Forward pass through a single layer, returning output, K, V.

  Partitioning:
  * 'x' is the only axis (most chips).
  * weights are sharded [dmodel, heads.XYZ, *_per_head]
  * hidden-dimension activations are sharded [batch, time, heads.XYZ,
  *_per_head]
  * embed-dimension activations are "naturally" sharded [batch, time, dmodel]
    but may be "oversharded" after reducescatter operations.

  To support XYZ>heads, we simply increase the number of heads, by padding. The
  FFN can be sharded finer and continues to gain speedup with more chips, but
  the ATTN will just be padded and not gain speedup as we add chips.
  """
  if latency_collectives:
    matmul_reducescatter = collectives.matmul_reducescatter_latency
    # reducescatter = collectives.reducescatter_latency
    matmul_allgather = collectives.allgather_matmul_latency
  else:
    # matmul_reducescatter = partial(
    #     collectives.matmul_reducescatter_throughput, subsplit_axis=0
    # )
    # # reducescatter = collectives.reducescatter_throughput
    # matmul_allgather = partial(
    #     collectives.allgather_matmul_throughput, subsplit_axis=2
    # )
    matmul_reducescatter = collectives.matmul_reducescatter_oneway
    # reducescatter = collectives.reducescatter_throughput
    matmul_allgather = collectives.allgather_matmul_one_way

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  batch, max_len, _ = x.shape

  with jax.named_scope('layernorm'):
    # x: [batch, maxlen, dmodel.X]
    # mean2: [batch, maxlen]
    # xnorm: [batch, maxlen, dmodel.X]
    epsilon = 1e-6
    mean2 = lax.pmean(
        jnp.mean(lax.square(x), axis=-1, keepdims=True), axis_name='x'
    )
    xnorm = intermediate_dtype(x * lax.rsqrt(mean2 + epsilon))  # pytype: disable=not-callable  # jnp-type

  # einsum(xnorm, q_wi):
  # [batch, maxlen, dmodel.X] @ [heads.XYZ, dmodel, q_wi_per_head]
  # -> (allgather lhs)   (fused with matmul)
  # -> [batch, maxlen, dmodel]
  # -> (matmul)
  # -> [batch, maxlen, heads.XYZ, q_wi_per_head]
  with jax.named_scope('q_wi'):
    q_wi = matmul_allgather(
        'bte,hed->bthd',
        xnorm,
        params.q_wi,
        rhs_split_axis=1,
        axis_name='x',
        layer=layer,
    )

    # No need to scatter over y and z, as y and z will always be 1 in here.

    two_d_parallel_xmap.assert_equal(
        q_wi.shape,
        (
            batch,
            max_len,
            hparams.heads // (x_axis * y_axis * z_axis),
            hparams.q_wi_per_head,
        ),
    )

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = q_wi.shape
      q_wi = intermediate_dtype(q_wi * jnp.squeeze(my_layer(params.q_wi_scale)))
      two_d_parallel_xmap.assert_equal(prev_shape, q_wi.shape)

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    wi0 = q_wi[
        :, :, :, hparams.qkv : hparams.qkv + (hparams.ff // hparams.heads)
    ]
    wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads) :]

  # einsum(xnorm, kv):
  #
  # [batch, maxlen, dmodel.X] @ [dmodel.X, 1, 2*qkv]
  # -> (matmul)
  # -> [batch, maxlen, 1, 2*qkv]{x unreduced}
  # -> (reducescatter over x into batch)
  #         *NOT* collective matmul, because it's batch
  # -> { Attn.NONE:      [batch, maxlen,  1, 2*qkv]
  with jax.named_scope('kv'):

    def kv_einsum(lhs):
      return jnp.einsum('bte,ezd->btzd', lhs, my_layer(params.kv))

    # kv_unreduced = jnp.einsum('bte,ezd->btzd', xnorm,
    #                           my_layer(params.kv))
    # [batch, maxlen, 1, 2*qkv]{x_unreduced}
    # --ARx-->   [batch, maxlen, 1, 2*qkv]
    kv = lax.psum(kv_einsum(xnorm), 'x')

    if isinstance(params, inference.QuantizedLayer):
      prev_shape = kv.shape
      kv = intermediate_dtype(kv * jnp.squeeze(my_layer(params.kv_scale)))
      two_d_parallel_xmap.assert_equal(prev_shape, kv.shape)

    k = kv[:, :, 0, : hparams.qkv]
    v = kv[:, :, 0, hparams.qkv :]

  with jax.named_scope('attn'):
    k = _rope(sin, cos, k)

    # q: [batch, maxlen, heads.XYZ, qkv]
    q = q_wi[:, :, :, : hparams.qkv]
    q = _rope(sin, cos, q)

    # y_att: -> [batch.B, maxlen, heads.XYZ, qkv]
    y_att = intermediate_dtype(attention.attend(q, k, v, kv_caches, layer))  # pytype: disable=not-callable  # jnp-type

  with jax.named_scope('SwiGLU'):
    y_mlp = special2.swish2(wi0) * wi1

  # einsum(y_fused, o_wo):
  # [batch, maxlen, heads.XYZ, o_wo_per_head]
  #   @ [heads.XYZ, o_wo_per_head, dmodel]
  # -> (matmul)
  # -> [batch, maxlen, dmodel]{XYZ unreduced}
  # -> (fused reducescatter over X)
  # -> [batch, maxlen, dmodel.X]{YZ unreduced}
  # -> (non-fused allreduce)
  # -> [batch, maxlen, dmodel.X]
  with jax.named_scope('o_wo'):
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
    two_d_parallel_xmap.assert_equal(
        y_fused.shape,
        (
            batch,
            max_len,
            hparams.heads // (x_axis * y_axis * z_axis),
            hparams.o_wo_per_head,
        ),
    )

    y_out = matmul_reducescatter(
        'bthd,hde->bte',
        y_fused,
        params.o_wo,
        scatter_axis=2,
        axis_name='x',
        layer=layer,
    )

    # No output psum because this is for only x
    # y_out = lax.psum(y_out, axis_name=('y', 'z'))

    if isinstance(params, inference.QuantizedLayer):
      prev_shape = y_out.shape
      y_out = intermediate_dtype(
          y_out * jnp.squeeze(my_layer(params.o_wo_scale))
      )
      two_d_parallel_xmap.assert_equal(y_out.shape, prev_shape)

  with jax.named_scope('residual'):
    z = intermediate_dtype(y_out + x)  # pytype: disable=not-callable  # jnp-type
  k, v = k.astype(intermediate_dtype), v.astype(intermediate_dtype)
  return z, k, v


def weight_stationary(
    hparams,
    layer,
    params,
    sin,
    cos,
    kv_caches,
    x,
    x_axis,
    y_axis,
    z_axis,
    attn_all_to_all,
    latency_collectives,
):
  """Forward pass through a single layer, returning output, K, V.

  Partitioning:
  * 'x' is the longest axis (most chips), then 'y', then 'z'.
  * weights are sharded [dmodel, heads.XYZ, *_per_head]
  * hidden-dimension activations are sharded [batch, time, heads.XYZ,
  *_per_head]
  * embed-dimension activations are "naturally" sharded [batch, time, dmodel]
    but may be "oversharded" after reducescatter operations.

  To support XYZ>heads, we simply increase the number of heads, by padding. The
  FFN can be sharded finer and continues to gain speedup with more chips, but
  the ATTN will just be padded and not gain speedup as we add chips.
  """
  if latency_collectives:
    matmul_reducescatter = collectives.matmul_reducescatter_latency
    # reducescatter = collectives.reducescatter_latency
    matmul_allgather = collectives.allgather_matmul_latency
  else:
    matmul_reducescatter = collectives.matmul_reducescatter_throughput
    # reducescatter = collectives.reducescatter_throughput
    matmul_allgather = collectives.allgather_matmul_throughput

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  batch, max_len, _ = x.shape
  batch_z = batch // z_axis
  batch_yz = batch_z // y_axis
  batch_xyz = batch_yz // x_axis

  # x_index = lax.axis_index('x')
  y_index = lax.axis_index('y')
  z_index = lax.axis_index('z')
  yz_index = y_index * z_axis + z_index

  with jax.named_scope('layernorm'):
    # x: [batch, maxlen, dmodel.X]
    # mean2: [batch, maxlen]
    # xnorm: [batch, maxlen, dmodel.X]
    epsilon = 1e-6
    mean2 = lax.pmean(
        jnp.mean(lax.square(x), axis=-1, keepdims=True), axis_name='x'
    )
    xnorm = jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))

  # einsum(xnorm, q_wi):
  # [batch, maxlen, dmodel.X] @ [heads.XYZ, dmodel, q_wi_per_head]
  # -> (allgather lhs)   (fused with matmul)
  # -> [batch, maxlen, dmodel]
  # -> (matmul)
  # -> [batch, maxlen, heads.XYZ, q_wi_per_head]
  with jax.named_scope('q_wi'):
    q_wi = matmul_allgather(
        'bte,hed->bthd',
        xnorm,
        params.q_wi,
        rhs_split_axis=1,
        axis_name='x',
        layer=layer,
        subsplit_axis=2,
    )

    two_d_parallel_xmap.assert_equal(
        q_wi.shape,
        (
            batch,
            max_len,
            hparams.heads // (x_axis * y_axis * z_axis),
            hparams.q_wi_per_head,
        ),
    )

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = q_wi.shape
      q_wi = jnp.bfloat16(q_wi * jnp.squeeze(my_layer(params.q_wi_scale)))
      two_d_parallel_xmap.assert_equal(prev_shape, q_wi.shape)

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    wi0 = q_wi[
        :, :, :, hparams.qkv : hparams.qkv + (hparams.ff // hparams.heads)
    ]
    wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads) :]

  # einsum(xnorm, kv):
  #
  # [batch, maxlen, dmodel.X] @ [dmodel.X, 1, 2*qkv]
  # -> (matmul)
  # -> [batch, maxlen, 1, 2*qkv]{x unreduced}
  # -> (reducescatter over x into batch)
  #         *NOT* collective matmul, because it's batch
  # -> { Attn.NONE:      [batch.B, maxlen,  1, 2*qkv]
  #    { Attn.AXIS_Z:    [batch.ZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZ:   [batch.YZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZX:  [batch.YZXB, maxlen, 1, 2*qkv]
  with jax.named_scope('kv'):

    def kv_einsum(lhs):
      return jnp.einsum('bte,ezd->btzd', lhs, my_layer(params.kv))

    # kv_unreduced = jnp.einsum('bte,ezd->btzd', xnorm,
    #                           my_layer(params.kv))

    if attn_all_to_all == partitioning.AttnAllToAll.NONE:
      # [batch, maxlen, 1, 2*qkv]{x_unreduced}
      # --ARx-->   [batch, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_einsum(xnorm), 'x')
    elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
      assert batch_z >= 1, 'Batch size too small for AXIS_Z and this chip count'
      # xnorm: [batch, maxlen, dmodel.X] -> [batch.Z, maxlen, dmodel.X]
      xnorm = lax.dynamic_slice_in_dim(
          xnorm, z_index * batch_z, batch_z, axis=0
      )
      # [batch.Z, maxlen, dmodel.X] @ [dmodel.X, 1, 2*qkv]
      # --matmul--> [batch.Z, maxlen, 1, 2*qkv]{x unreduced}
      # --ARx-->    [batch.Z, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_einsum(xnorm), 'x')
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
      assert (
          batch_yz >= 1
      ), 'Batch size too small for AXES_YZ and this chip count'
      # xnorm: [batch, maxlen, dmodel.X] -> [batch.YZ, maxlen, dmodel.X]
      xnorm = lax.dynamic_slice_in_dim(
          xnorm, yz_index * batch_yz, batch_yz, axis=0
      )
      # [batch.YZ, maxlen, dmodel.X] @ [dmodel.X, 1, 2*qkv]
      # --matmul--> [batch.YZ, maxlen, 1, 2*qkv]{x unreduced}
      # --ARx-->    [batch.YZ, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_einsum(xnorm), 'x')
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
      assert (
          batch_xyz >= 1
      ), 'Batch size too small for AXES_XYZ and this chip count'
      # xnorm: [batch, maxlen, dmodel.X] -> [batch.YZ, maxlen, dmodel.X]
      xnorm = lax.dynamic_slice_in_dim(
          xnorm, yz_index * batch_yz, batch_yz, axis=0
      )
      # [batch.YZ, maxlen, dmodel.X] @ [dmodel.X, 1, 2*qkv]
      # --matmul--> [batch.YZ, maxlen, 1, 2*qkv]{x unreduced}
      # --RSx-->    [batch.YZ, maxlen, 1, 2*qkv]
      kv = lax.psum_scatter(
          kv_einsum(xnorm), 'x', scatter_dimension=0, tiled=True
      )

    if isinstance(params, inference.QuantizedLayer):
      prev_shape = kv.shape
      kv = jnp.bfloat16(kv * jnp.squeeze(my_layer(params.kv_scale)))
      two_d_parallel_xmap.assert_equal(prev_shape, kv.shape)

    k = kv[:, :, 0, : hparams.qkv]
    v = kv[:, :, 0, hparams.qkv :]

  with jax.named_scope('attn'):
    k = _rope(sin, cos, k)

    # q: [batch, maxlen, heads.XYZ, qkv]
    # -> { NONE:                   [batch,  maxlen, heads.XYZ, qkv]
    #    { AXIS_Z:                 [batch.Z, maxlen, heads.XY, qkv]
    #    { AXES_YZ:                [batch.YZ, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZX, maxlen, heads,  qkv]
    q = q_wi[:, :, :, : hparams.qkv]
    if attn_all_to_all == partitioning.AttnAllToAll.NONE:
      pass
    elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
      q = lax.all_to_all(
          q, axis_name='z', split_axis=0, concat_axis=2, tiled=True
      )
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
      q = lax.all_to_all(
          q, axis_name=('y', 'z'), split_axis=0, concat_axis=2, tiled=True
      )
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
      q = lax.all_to_all(
          q, axis_name='x', split_axis=0, concat_axis=2, tiled=True
      )
      q = lax.all_to_all(
          q, axis_name=('y', 'z'), split_axis=0, concat_axis=2, tiled=True
      )

    q = _rope(sin, cos, q)

    y_att = jnp.bfloat16(attention.attend(q, k, v, kv_caches, layer))
    # y_att:
    #    { NONE:                   [batch,  maxlen, heads.YZX, qkv]
    #    { AXIS_Z:                 [batch.Z, maxlen, heads.YX, qkv]
    #    { AXES_YZ:                [batch.YZ, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZX, maxlen, heads,  qkv]
    # -> [batch.B, maxlen, heads.YZX, qkv]
    if attn_all_to_all == partitioning.AttnAllToAll.NONE:
      pass
    elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
      y_att = lax.all_to_all(
          y_att, axis_name='z', split_axis=2, concat_axis=0, tiled=True
      )
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
      y_att = lax.all_to_all(
          y_att, axis_name=('y', 'z'), split_axis=2, concat_axis=0, tiled=True
      )
    elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
      y_att = lax.all_to_all(
          y_att, axis_name=('y', 'z'), split_axis=2, concat_axis=0, tiled=True
      )
      y_att = lax.all_to_all(
          y_att, axis_name='x', split_axis=2, concat_axis=0, tiled=True
      )

  with jax.named_scope('SwiGLU'):
    y_mlp = special2.swish2(wi0) * wi1

  # einsum(y_fused, o_wo):
  # [batch, maxlen, heads.XYZ, o_wo_per_head]
  #   @ [heads.XYZ, o_wo_per_head, dmodel]
  # -> (matmul)
  # -> [batch, maxlen, dmodel]{XYZ unreduced}
  # -> (fused reducescatter over X)
  # -> [batch, maxlen, dmodel.X]{YZ unreduced}
  # -> (non-fused allreduce)
  # -> [batch, maxlen, dmodel.X]
  with jax.named_scope('o_wo'):
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
    two_d_parallel_xmap.assert_equal(
        y_fused.shape,
        (
            batch,
            max_len,
            hparams.heads // (x_axis * y_axis * z_axis),
            hparams.o_wo_per_head,
        ),
    )

    y_out = matmul_reducescatter(
        'bthd,hde->bte',
        y_fused,
        params.o_wo,
        scatter_axis=2,
        axis_name='x',
        layer=layer,
        subsplit_axis=2,
    )

    # TODO(sholto): Explore psum-scatter?
    y_out = lax.psum(y_out, axis_name=('y', 'z'))

    if isinstance(params, inference.QuantizedLayer):
      prev_shape = y_out.shape
      y_out = jnp.bfloat16(y_out * jnp.squeeze(my_layer(params.o_wo_scale)))
      two_d_parallel_xmap.assert_equal(y_out.shape, prev_shape)

  with jax.named_scope('residual'):
    z = jnp.bfloat16(y_out + x)
  return z, k[:batch_xyz], v[:batch_xyz]
