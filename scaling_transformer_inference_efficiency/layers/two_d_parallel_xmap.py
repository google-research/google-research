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

"""One Transformer layer, in hard xmap."""

from functools import partial  # pylint: disable = g-importing-member
from typing import Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.layers.layers_pjit import _rope
from scaling_transformer_inference_efficiency.partitioning import AttnAllToAll
from scaling_transformer_inference_efficiency.weights import Layer

HParams = checkpoint.HParams
CheckpointSpec = checkpoint.CheckpointSpec
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer
Weights = weights.Weights

ATTN_3D_SHARDING_THRESHOLD_PER_CHIP = 2

# pylint: disable = invalid-name
# pylint: disable = protected-access
# pylint: disable = g-bare-generic


def assert_equal(x, y):
  assert x == y, f'{x} != {y}'


def allgather_layernorm(x,  # pytype: disable=annotation-type-mismatch
                        shard_seqlen_vs_batch,
                        batch_unsharded = False,
                        scale = None):
  """All gathers around layernorm, minimises comms by first doing per-chip."""
  with jax.named_scope('allgather_layernorm'):
    # allgather xnorm: [batch.Z, maxlen, embed.XY] || [batch, maxlen, embed.XYZ]
    # -> [batch.Z, maxlen, embed.X]    (xnorm_z)
    # -> [batch, maxlen, embed.X]
    xgather = x
    if batch_unsharded:
      # [batch, maxlen, embed.XY]
      xgather = lax.all_gather(xgather, 'z', axis=2, tiled=True)
    # [batch.Z, maxlen, embed.X] || [batch, maxlen, embed.X]
    xgather = lax.all_gather(xgather, 'y', axis=2, tiled=True)

    epsilon = 1e-6
    xgather = jnp.float32(xgather)
    mean2 = lax.pmean(
        jnp.mean(lax.square(xgather), axis=-1, keepdims=True), axis_name='x')
    xnorm_z = jnp.bfloat16(xgather * lax.rsqrt(mean2 + epsilon))
    if scale is not None:
      scale += 1.0  # 'center_scale_at_zero' option in T5X
      xnorm_z = jnp.bfloat16(xnorm_z * scale)
    # when attention_all_to_all is None we can partition over sequence len not
    # batch
    if shard_seqlen_vs_batch:
      xnorm = lax.all_gather(xnorm_z, 'z', axis=1, tiled=True)
    else:
      if batch_unsharded:  # in this case already done above
        xnorm = xnorm_z
      else:
        xnorm = lax.all_gather(xnorm_z, 'z', axis=0, tiled=True)
  # [batch, maxlen, embed.X]
  return xnorm, xnorm_z


@partial(jax.jit, static_argnums=(3, 4, 5))
def embed_manual(
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk,
    shard_seqlen_vs_batch = False,
    batch_unsharded = False,
    one_d = False,
):
  """Embeds a chunk of logits.

  Args:
    params: Weights object
    kv_caches: List of chunks preprocessed earlier
    token_chunk: An unsharded token chunk. Assume .tokens is int32[batch,
      maxlen]
    shard_seqlen_vs_batch: Whether to shard seqlen or batch by z.
    batch_unsharded:  global_batch is less than z so we cannot shard along
    one_d: whether it is one dimensional

  Returns:
    embeddings: bfloat16[[batch.Z, time, embed.XY] || [batch, time, embed.XYZ]
    sin: RoPE embeddings starting at the appropriate index determined by
         pre-existing kv_cache for each index in the batch.
    cos: ""
  """

  z_axis = lax.psum(1, 'z')
  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  prefix_batch, = start_indices.shape
  batch, max_length = token_chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'
  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  # x: int32[batch, maxlen]
  # embed: bfloat16[vocab.YZ, embed.X]
  x = token_chunk.tokens
  vocab_yz, _ = params.embedding.shape

  yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
  vocab_start = yz_index * vocab_yz

  # Initial embedding lookup:
  with jax.named_scope('embed'):
    one_x = x - vocab_start
    embeds = params.embedding[one_x, :]
    one_x = one_x[:, :, jnp.newaxis]
    embeds = jnp.where((one_x >= 0) & (one_x < vocab_yz), embeds, 0)
    # [batch, time, embed.X]
    if one_d:
      return embeds, sin, cos
    # [batch, time, embed.XY]
    embeds = lax.psum_scatter(embeds, 'y', scatter_dimension=2, tiled=True)

    if shard_seqlen_vs_batch:
      # [batch, time.Z, embed.XY]
      embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=1, tiled=True)
    else:
      if batch_unsharded:
        # [batch, time, embed.XYZ]
        embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=2, tiled=True)
      else:
        # [batch.Z, time, embed.XY]
        embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=0, tiled=True)

  return embeds, sin, cos


def unembed_manual(
    xnorm,
    params,
    batch_unsharded = False,
    one_d = False,
):
  """Unembedding function for 2D."""
  # x: bfloat16[batch, maxlen, dmodel.X] # [vocab.YZ, embed.X]
  # TODO(sholto): We could squeeze out more memory by doing this
  # with a collective
  with jax.named_scope('unembed'):
    logits_unreduced = jnp.einsum(
        'bte,ve->btv', jnp.float32(xnorm), jnp.float32(params.embedding)
    )
    # x: [batch, maxlen, vocab.YZ] {X unreduced}
    if batch_unsharded or one_d:
      # logits: float32[batch, maxlen, vocab.YZX]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=2, tiled=True
      )
    else:
      # logits: float32[batch.X, maxlen, vocab.YZ]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=0, tiled=True
      )
  return logits


# pylint: disable = g-doc-return-or-yield
# pylint: disable = g-doc-args
# TODO(sholto): Update to new, tested parsing collectives.


def transformer_layer_weight_stationary(
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
    *,
    attn_all_to_all,
    latency_collectives,
    shard_seqlen_vs_batch = False,
    batch_unsharded = False,
    intermediate_dtype = jnp.bfloat16,
):
  """Wraps _fn so that we can use remat while bug is fixed."""
  return jax.checkpoint(
      partial(
          _transformer_layer_weight_stationary,
          attn_all_to_all=attn_all_to_all,
          latency_collectives=latency_collectives,
          shard_seqlen_vs_batch=shard_seqlen_vs_batch,
          batch_unsharded=batch_unsharded,
          intermediate_dtype=intermediate_dtype,
      ),
      static_argnums=(0, 7, 8, 9),
      prevent_cse=True,
  )(hparams, layer, params, sin, cos, kv_caches, x, x_axis, y_axis, z_axis)


def _transformer_layer_weight_stationary(
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
    *,
    attn_all_to_all,
    latency_collectives,
    shard_seqlen_vs_batch = False,
    batch_unsharded = False,
    intermediate_dtype = jnp.bfloat16,
):
  """Forward pass through a single layer, returning output, K, V.

  This implementation has 'x'=d_model sharding,
  ('y', 'z')=d_ff sharding.
  * params are assumed already sharded this way, i.e. embed.X and heads.YZ
  * sin and cos are sharded by batch.YZx (or batch.YZ or batch.Y as necessary)
  * kv_cache is sharded by batch.YZx (or batch.YZ or batch.Y as necessary)
  * x: [batch.Z, maxlen, embed.XY]
  """
  intermediate_dtype = jax.core.concrete_or_error(None, intermediate_dtype)
  if latency_collectives:
    matmul_reducescatter = partial(
        collectives.matmul_reducescatter_latency, subsplit_axis=2)
    # reducescatter = collectives.reducescatter_latency
    # subsplit along heads as they are indepedent
    # partial here because the one-way algorithm does not use subsplit
    matmul_allgather = partial(
        collectives.allgather_matmul_latency, subsplit_axis=2)
  else:
    matmul_reducescatter = collectives.matmul_reducescatter_oneway
    # reducescatter = collectives.reducescatter_oneway
    matmul_allgather = collectives.allgather_matmul_one_way

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # Compare
  # flaxformer/architectures/t5/parallel_fused_decoder.py
  # flaxformer/components/attention/dense_attention.py;l=1147;
  # flaxformer/components/attention/dense_attention.py;l=252;

  batch_z, max_len, _ = x.shape
  if shard_seqlen_vs_batch:
    max_len *= z_axis
    batch = batch_z
    batch_xyz = batch // (x_axis * y_axis * z_axis)
  else:
    if batch_unsharded:
      batch = x.shape[0]
    else:
      batch = batch_z * z_axis
    batch_xyz = batch // (x_axis * y_axis * z_axis)
    batch_yz = batch // (y_axis * z_axis)
    batch_z = batch // (z_axis)

  if isinstance(params, weights.QuantizedLayer):
    xnorm, xnorm_z = allgather_layernorm(
        x,
        shard_seqlen_vs_batch,
        batch_unsharded,
        scale=my_layer(params.layernorm_scale))
  else:
    xnorm, xnorm_z = allgather_layernorm(x, shard_seqlen_vs_batch,
                                         batch_unsharded)

  # einsum(xnorm, q_wi):
  # [batch, maxlen, embed.X] @ [heads.YZ, embed.X, q_wi_per_head]
  # -> (matmul)
  # -> [batch, maxlen, heads.YZ, q_wi_per_head]{x unreduced}
  # -> (reducescatter over x into X heads, B batches)
  # -> [batch, maxlen, heads.YZX, q_wi_per_head]
  with jax.named_scope('q_wi'):
    xnorm = intermediate_dtype(xnorm)
    q_wi = matmul_reducescatter(
        'bte,hed->bthd',
        xnorm,
        params.q_wi,
        scatter_axis=0,
        axis_name='x',
        layer=layer)

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = q_wi.shape
      q_wi = intermediate_dtype(q_wi * jnp.squeeze(my_layer(params.q_wi_scale)))
      assert_equal(prev_shape, q_wi.shape)

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // (hparams.heads - hparams.padded_heads))]  # pylint: disable = line-too-long
    wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // (hparams.heads - hparams.padded_heads)):]  # pylint: disable = line-too-long

  # einsum(xnorm, kv):
  #
  # if attn>=AXES_YZ:
  #   xnorm_z: [batch.Z, maxlen, embed.X]
  #     -> [batch.(X?)YZ, maxlen, embed.X]  (slice down)
  #
  # Then:
  #
  # [batch.(Y?)Z, maxlen, embed.X] @ [embed.X, 1, 2*qkv]
  # -> (matmul)
  # -> [batch.(Y?)Z, maxlen, 1, 2*qkv]{x unreduced}
  # -> (reducescatter over x into batch)
  #         *NOT* collective matmul, because it's batch
  # -> { Attn.NONE:      [batch.B, maxlen,  1, 2*qkv]
  #    { Attn.AXIS_Z:    [batch.ZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZ:   [batch.YZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZX:  [batch.YZXB, maxlen, 1, 2*qkv]
  with jax.named_scope('kv'):
    # TODO(sholto): update this in oversharded
    yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
    # TODO(reinerp): Consider using xnorm instead of xnorm_z in NONE case?
    # I don't know yet if that's better.
    if attn_all_to_all.value >= AttnAllToAll.AXES_YZ.value:
      xnorm_sliced = lax.dynamic_slice_in_dim(
          xnorm, yz_index * batch_yz, batch_yz, axis=0)
    else:
      xnorm_sliced = xnorm_z

    kv_unreduced = jnp.einsum('bte,ezd->btzd', xnorm_sliced,
                              my_layer(params.kv))

    if attn_all_to_all == AttnAllToAll.NONE:
      if shard_seqlen_vs_batch:
        # [batch, maxlen.Z, 1, 2*qkv]{x_unreduced}
        # -> [batch.B, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        kv = lax.all_gather(kv, 'z', axis=1, tiled=True)
      else:
        # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced} || [b, ml, 1, 2qkv] {x_unred}
        # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
        # --AGZ-->   [batch, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        if not batch_unsharded:
          kv = lax.all_gather(kv, 'z', axis=0, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
      # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_unreduced, 'x')
      # print('kv2', kv.shape, kv.named_shape)
    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
      # --ARx-->   [batch.YZ, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_unreduced, 'x')
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
      # --RSx-->   [batch.YZX, maxlen, 1, 2*qkv]
      assert batch_xyz >= 1, ('Batch size too small for AXES_XYZ and this chip '
                              'count')
      kv = lax.psum_scatter(kv_unreduced, 'x', scatter_dimension=0, tiled=True)

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = kv.shape
      kv = intermediate_dtype(kv * jnp.squeeze(my_layer(params.kv_scale)))
      assert_equal(prev_shape, kv.shape)

    k = kv[:, :, 0, :hparams.qkv]
    v = kv[:, :, 0, hparams.qkv:]

  with jax.named_scope('attn'):
    k = _rope(sin, cos, k)

    # print(f'batch_yzb: {batch_yzb}')
    # q: [batch, maxlen, heads.YZX, qkv]
    # -> { NONE:                   [batch., maxlen, heads.YZX, qkv]
    #    { AXIS_Z:                 [batch.Z, maxlen, heads.YX, qkv]
    #    { AXES_YZ:                [batch.YZ, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZX, maxlen, heads, qkv]
    q = q_wi[:, :, :, :hparams.qkv]
    if attn_all_to_all == AttnAllToAll.NONE:
      pass
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      q = lax.all_to_all(
          q, axis_name='z', split_axis=0, concat_axis=2, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      q = lax.all_to_all(
          q, axis_name=('y', 'z'), split_axis=0, concat_axis=2, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      q = lax.all_to_all(
          q, axis_name=('y', 'z', 'x'), split_axis=0, concat_axis=2, tiled=True)

    q = _rope(sin, cos, q)

    y_att = intermediate_dtype(attention.attend(q, k, v, kv_caches, layer))
    # y_att:
    #    { NONE:                   [batch.B, maxlen, heads.YZX, qkv]
    #    { AXIS_Z:                 [batch.ZB, maxlen, heads.YX, qkv]
    #    { AXES_YZ:                [batch.YZB, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZX, maxlen, heads, qkv]
    # -> [batch, maxlen, heads.YZX, qkv]
    if attn_all_to_all == AttnAllToAll.NONE:
      pass
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      y_att = lax.all_to_all(
          y_att, axis_name='z', split_axis=2, concat_axis=0, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      y_att = lax.all_to_all(
          y_att, axis_name=('y', 'z'), split_axis=2, concat_axis=0, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      y_att = lax.all_to_all(
          y_att,
          axis_name=('y', 'z', 'x'),
          split_axis=2,
          concat_axis=0,
          tiled=True)

  with jax.named_scope('SwiGLU'):
    y_mlp = special2.swish2(wi0) * wi1

  # einsum(y_fused, o_wo):
  # [batch, maxlen, heads.YZ, o_wo_per_head] @
  #       [heads.YZ, o_wo_per_head, embed.X]
  # -> (matmul)
  # -> [batch, maxlen, embed.X]{YZ unreduced}
  # -> (fused reducescatter)
  # -> [batch, maxlen, embed.XY]
  # -> (non-fused reducescatter)
  # -> [batch.Z, maxlen, embed.XY]
  with jax.named_scope('o_wo'):
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)

    # do the second half of the mlp and the self-attn projection in parallel
    # allgather y_fused: [batch, maxlen, heads.YZX, o_wo_per_head]
    #       -> [batch, maxlen, heads.YZ, o_wo_per_head]
    # we use the collective matmul/reducescatter instead
    # print(f'o_wo: {params.o_wo.shape}')
    y_out = matmul_allgather(
        'bthd,hde->bte',
        y_fused,
        params.o_wo,
        rhs_split_axis=0,
        axis_name='x',
        layer=layer)
    # y_out = reducescatter(
    #     y_out, scatter_dimension=2, axis_name='y', subsplit_axis=2)

    y_out = lax.psum_scatter(y_out, 'y', scatter_dimension=2, tiled=True)

    if shard_seqlen_vs_batch:
      # y_out = reducescatter(
      #     y_out, scatter_dimension=1, axis_name='z', subsplit_axis=0)
      # [batch, maxlen.Z, embed.XY]
      y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=1, tiled=True)
    else:
      # y_out = reducescatter(
      #     y_out, scatter_dimension=0, axis_name='z', subsplit_axis=0)
      # TODO(sholto): Test if manual faster, update
      if batch_unsharded:
        # [batch, maxlen, embed.XYZ]
        y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=2, tiled=True)
      else:
        # [batch.Z, maxlen, embed.XY]
        y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=0, tiled=True)

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = y_out.shape
      y_out = intermediate_dtype(y_out *
                                 jnp.squeeze(my_layer(params.o_wo_scale)))
      assert_equal(y_out.shape, prev_shape)

  with jax.named_scope('residual'):
    z = intermediate_dtype(y_out + x)

  k, v = k.astype(intermediate_dtype), v.astype(intermediate_dtype)
  return z, k, v


def transformer_layer_weight_gathered(
    hparams, layer, params, sin,
    cos, kv_caches, x,
    x_axis, y_axis,
    z_axis):
  """Weight gathered parallel layer. Typically prefill."""
  del x_axis, y_axis, z_axis  # for API compatibility
  # x: [batch.XYZ, t, e]
  with jax.named_scope('allgather_layernorm'):
    # No need to communicate across batch, so everything is local
    x_prec = jnp.float32(x)
    epsilon = 1e-6
    mean2 = jnp.mean(lax.square(x_prec), axis=-1, keepdims=True)
    xnorm = jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # [batch.XYZ, t, e] @ [heads.YZ, e.X, q_wi_per_head]
  with jax.named_scope('q_wi'):
    q_wi = collectives.matmul_collective_weights_gather_q_wi(
        'bte,hed->bthd',
        xnorm,
        my_layer(
            params.q_wi
        ),  # in this case it makes sense to do this here because its once
        lhs_split_axis=2)  #   -> [batch.XYZ, t, h, q_wi_per_head]

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = q_wi.shape
      q_wi = jnp.bfloat16(q_wi * jnp.squeeze(my_layer(params.q_wi_scale)))
      assert_equal(prev_shape, q_wi.shape)

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // (hparams.heads - hparams.padded_heads))]  # pylint: disable = line-too-long
    wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // (hparams.heads - hparams.padded_heads)):]  # pylint: disable = line-too-long

    # kv is only batch sharded
    with jax.named_scope('kv'):
      # [batch.XYZ, t, e] @ [e, 1, 2*qkv] -> [batch.XYZ, t, 1, 2*qkv]
      # Two options here:
      # a) Split along x, and then all reduce along x
      # b) We fully replicate kv
      kv = jnp.einsum('bte,ezd->btzd', xnorm, my_layer(params.kv))

      if isinstance(params, weights.QuantizedLayer):
        prev_shape = kv.shape
        kv = jnp.bfloat16(kv * jnp.squeeze(my_layer(params.kv_scale)))
        assert_equal(prev_shape, kv.shape)

      k = kv[:, :, 0, :hparams.qkv]  # [batch.XYZ, t, qkv]
      v = kv[:, :, 0, hparams.qkv:]  # [batch.XYZ, t, qkv]

    with jax.named_scope('attn'):
      k = _rope(sin, cos, k)  # [batch.XYZ, t, qkv]
      q = q_wi[:, :, :, :hparams.qkv]
      q = _rope(sin, cos, q)  # [batch.XYZ, t, h, qkv]

      # [batch.XYZ, t, h, qkv]
      y_att = jnp.bfloat16(attention.attend(q, k, v, kv_caches, layer))

    with jax.named_scope('SwiGLU'):
      y_mlp = special2.swish2(wi0) * wi1  # [batch.XYZ, t, h, ff_per_head]

    # [bach.XYZ, t , h, d] @ [h.YZ, d, e.X] -> [batch.XYZ, t, e]
    with jax.named_scope('o_wo'):
      y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)

      # previously concat yz, contracting over x - reconstructing heads dim
      # here, we contract over yz, concat over x to reconstruct embed dim
      y_out = collectives.matmul_collective_weights_gather_o_wo(
          'bthd,hde->bte', y_fused, my_layer(params.o_wo),
          lhs_split_axis=2)  # -> [batch.XYZ, t, e]

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = y_out.shape
      y_out = jnp.bfloat16(y_out * jnp.squeeze(my_layer(params.o_wo_scale)))
      assert_equal(y_out.shape, prev_shape)

    with jax.named_scope('residual'):
      z = jnp.bfloat16(y_out + x)

    return z, k, v
