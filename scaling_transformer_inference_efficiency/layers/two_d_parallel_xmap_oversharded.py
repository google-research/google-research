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

"""Logic  for scaling to many more chips than batch/heads. Brittle. To fix."""

from typing import Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
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
# pylint: disable = g-doc-return-or-yield
# pylint: disable = g-doc-args


# latency sensitive experiments on 128 / 256 chips)
# TODO(sholto): The logic is highly confusing and insanely brittle, fix.
def transformer_layer_weight_stationary_oversharded(
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
    shard_seqlen_vs_batch = False,
    intermediate_dtype = jnp.bfloat16,
):
  """Forward pass through a single layer, returning output, K, V.

  The 'x' physical axis plays multiple roles:
  * for d_model sharding, we shard 100% over 'x', which we call '.x' in our
  notation
  * for heads-sharding (on reducescatter), we break the 'x' physical axis up
  into a product axis:
    * ".X" in our notation represents head-sharding
    * ".B" in our notation represents batch-sharding. We use this if we run out
    of head-sharding. It is the "inner-most" part of this product axis.
  In the extreme with X=1 and B=8, this lets us scale up to a pf_8x8x8 slice
  before this partitioning runs out of steam. We need a custom reducescatter
  implementation that supports scattering over two different axes X and B. We do
  that as follows:
  * when B=1, all our implementations work
  * when B=2, X>1, reducescatter_latency works, and we still get
    compute/communication overlap
  * (theoretically) when B>2, we won't get any compute/communication overlap. We
  do a lax.psum_scatter over B and a lax.psum_scatter over X.
  For slice sizes <= pf_8x8x8, we only care about the B=1 or B=2 cases.

  This implementation is for pmap, with 'x'=d_model sharding,
  ('y', 'z')=d_ff sharding.
  * params are assumed already sharded this way, i.e. dmodel.x and heads.YZ
  * sin and cos are sharded by batch.YZx (or batch.YZ or batch.Y as necessary)
  * kv_cache is sharded by batch.YZx (or batch.YZ or batch.Y as necessary)
  * x: [batch.Z, maxlen, dmodel.xY]
  """
  if latency_collectives:
    matmul_reducescatter = collectives.matmul_reducescatter_latency
    reducescatter = collectives.reducescatter_latency
    matmul_allgather = collectives.allgather_matmul_latency
  else:
    if len(jax.local_devices()) <= 32:
      matmul_reducescatter = collectives.matmul_reducescatter_oneway
      reducescatter = collectives.reducescatter_oneway
      matmul_allgather = collectives.allgather_matmul_one_way
      # matmul_reducescatter = collectives.matmul_reducescatter_no_collective
      # reducescatter = collectives.plain_reducescatter
      # matmul_allgather = collectives.matmul_allgather_no_collective
    else:
      matmul_reducescatter = collectives.matmul_reducescatter_throughput
      reducescatter = collectives.reducescatter_throughput
      matmul_allgather = collectives.allgather_matmul_throughput

  heads_yz = hparams.heads // (y_axis * z_axis)
  if heads_yz >= x_axis:
    B = 1
    X = x_axis
  else:
    B = x_axis // heads_yz
    X = heads_yz

  if B >= 2:
    # There are X many b_groups, each of size B
    # b_groups = [list(a) for a in np.reshape(np.arange(x_axis), (X, B))]
    # There are B many x_groups, each of size X
    x_groups = [list(a) for a in np.reshape(np.arange(x_axis), (X, B)).T]
  else:
    x_groups = None
    # b_groups = None

  b_index = lax.axis_index('x') % B

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # Compare
  # flaxformer/architectures/t5/parallel_fused_decoder.py
  # flaxformer/components/attention/dense_attention.py;l=1147;
  # flaxformer/components/attention/dense_attention.py;l=252;

  # prefix_batch = sin.shape[0]

  batch_z, max_len, _ = x.shape
  if shard_seqlen_vs_batch:
    max_len *= z_axis
    batch = batch_z
    batch_xyz = batch // (x_axis * y_axis * z_axis)
  else:
    batch = batch_z * z_axis
    batch_xyz = batch // (x_axis * y_axis * z_axis)
    batch_yz = batch // (y_axis * z_axis)
    batch_yzb = batch_yz // B
    batch_zb = batch // (z_axis * B)
    # beam = batch // prefix_batch # TODO(reinerp): Do we need this

  if batch == 1 and max_len == 1:
    raise ValueError('sharded batch-1 matmul is broken on VLC, b/246436629')

  # einsum(xnorm, q_wi):
  # [batch, maxlen, dmodel.x] @ [heads.YZ, dmodel.x, q_wi_per_head]
  # -> (matmul)
  # -> [batch, maxlen, heads.YZ, q_wi_per_head]{x unreduced}
  # -> (reducescatter over x into X heads, B batches)
  # -> [batch.B, maxlen, heads.YZX, q_wi_per_head]
  # TODO(reinerp): For chips>64, need to reducescatter over batch instead.

  if isinstance(params, weights.QuantizedLayer):
    xnorm, xnorm_z = two_d_parallel_xmap.allgather_layernorm(
        x, shard_seqlen_vs_batch, scale=my_layer(params.layernorm_scale))
  else:
    xnorm, xnorm_z = two_d_parallel_xmap.allgather_layernorm(
        x, shard_seqlen_vs_batch)

  with jax.named_scope('q_wi'):
    if B == 1:
      xnorm = intermediate_dtype(xnorm)
      q_wi = matmul_reducescatter(
          'bte,hed->bthd',
          xnorm,
          params.q_wi,
          scatter_dimension=(0, 2),
          axis_name='x',
          layer=layer,
          subsplit_axis=0)
    else:
      q_wi_unreduced = jnp.einsum('bte,hed->bthd', xnorm, my_layer(params.q_wi))
      # Combine batch into heads, reducescatter over heads, split batch back out
      two_d_parallel_xmap.assert_equal(
          q_wi_unreduced.shape,
          (batch, max_len, heads_yz, hparams.q_wi_per_head))
      q_wi_unreduced = jnp.reshape(
          q_wi_unreduced,
          (B, batch // B, max_len, heads_yz, hparams.q_wi_per_head))
      q_wi_unreduced = jnp.transpose(q_wi_unreduced, (1, 2, 0, 3, 4))
      q_wi_unreduced = jnp.reshape(
          q_wi_unreduced,
          (batch // B, max_len, B * heads_yz, hparams.q_wi_per_head))
      q_wi = collectives.reducescatter_latency(
          q_wi_unreduced, scatter_dimension=2, axis_name='x')

    if shard_seqlen_vs_batch:
      two_d_parallel_xmap.assert_equal(
          q_wi.shape, (batch, max_len, heads_yz // X, hparams.q_wi_per_head))
    else:
      two_d_parallel_xmap.assert_equal(
          q_wi.shape,
          (batch // B, max_len, heads_yz // X, hparams.q_wi_per_head))

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = q_wi.shape
      q_wi = q_wi * jnp.squeeze(my_layer(params.q_wi_scale))
      two_d_parallel_xmap.assert_equal(prev_shape, q_wi.shape)

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
    wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]

  # einsum(xnorm, kv):
  #
  # if attn>=AXES_YZ:
  #   xnorm_z: [batch.Z, maxlen, dmodel.x]
  #     -> [batch.YZ, maxlen, dmodel.x]  (slice down)
  #
  # Then:
  #
  # [batch.Y? Z, maxlen, dmodel.x] @ [dmodel.x, 1, 2*qkv]
  # -> (matmul)
  # -> [batch.Y? Z, maxlen, 1, 2*qkv]{x unreduced}
  # -> (reducescatter over x into batch)
  #         *NOT* collective matmul, because it's batch
  # -> { Attn.NONE:      [batch.B, maxlen,  1, 2*qkv]
  #    { Attn.AXIS_Z:    [batch.ZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZ:   [batch.YZB, maxlen, 1, 2*qkv]
  #    { Attn.AXES_YZX:  [batch.YZXB, maxlen, 1, 2*qkv]
  with jax.named_scope('kv'):
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
        assert B == 1
        kv = lax.psum(kv_unreduced, 'x')
        kv = lax.all_gather(kv, 'z', axis=1, tiled=True)
      else:
        # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
        # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
        # --slice--> [batch.ZB, maxlen, 1, 2*qkv]
        # --AGZ-->   [batch.B, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        # TODO(sholto) - confirm we no longer need
        # kv = lax.dynamic_slice_in_dim(kv, b_index*batch_zb, batch_zb, axis=0)
        kv = lax.all_gather(kv, 'z', axis=0, tiled=True)
    elif attn_all_to_all == AttnAllToAll.AXIS_Z:
      # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
      # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
      # --slice--> [batch.ZB, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_unreduced, 'x')

      kv = lax.dynamic_slice_in_dim(kv, b_index * batch_zb, batch_zb, axis=0)

    elif attn_all_to_all == AttnAllToAll.AXES_YZ:
      # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
      # --ARx-->   [batch.YZ, maxlen, 1, 2*qkv]
      # --slice--> [batch.YZB, maxlen, 1, 2*qkv]
      kv = lax.psum(kv_unreduced, 'x')
      kv = lax.dynamic_slice_in_dim(kv, b_index * batch_yzb, batch_yzb, axis=0)
    elif attn_all_to_all == AttnAllToAll.AXES_YZX:
      # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
      # --RSx-->   [batch.YZXB, maxlen, 1, 2*qkv]
      assert batch_xyz >= 1, ('Batch size too small for AXES_XYZ and this chip '
                              'count')
      kv = lax.psum_scatter(kv_unreduced, 'x', scatter_dimension=0, tiled=True)

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = kv.shape
      kv = kv * jnp.squeeze(my_layer(params.kv_scale))
      two_d_parallel_xmap.assert_equal(prev_shape, kv.shape)

    k = kv[:, :, 0, :hparams.qkv]
    v = kv[:, :, 0, hparams.qkv:]

  with jax.named_scope('attn'):
    k = _rope(sin, cos, k)

    # q: [batch.B, maxlen, heads.YZX, qkv]
    # -> { NONE:                   [batch.B, maxlen, heads.YZX, qkv]
    #    { AXIS_Z:                 [batch.ZB, maxlen, heads.YX, qkv]
    #    { AXES_YZ:                [batch.YZB, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZXB, maxlen, heads, qkv]
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
          q,
          axis_name='x',
          split_axis=0,
          concat_axis=2,
          tiled=True,
          axis_index_groups=x_groups)
      q = lax.all_to_all(
          q, axis_name=('y', 'z'), split_axis=0, concat_axis=2, tiled=True)

    q = _rope(sin, cos, q)

    y_att = intermediate_dtype(attention.attend(q, k, v, kv_caches, layer))
    # y_att:
    #    { NONE:                   [batch.B, maxlen, heads.YZX, qkv]
    #    { AXIS_Z:                 [batch.ZB, maxlen, heads.YX, qkv]
    #    { AXES_YZ:                [batch.YZB, maxlen, heads.X, qkv]
    #    { AXES_YZX:               [batch.YZXB, maxlen, heads, qkv]
    # -> [batch.B, maxlen, heads.YZX, qkv]
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
          y_att, axis_name=('y', 'z'), split_axis=2, concat_axis=0, tiled=True)
      y_att = lax.all_to_all(
          y_att,
          axis_name='x',
          split_axis=2,
          concat_axis=0,
          tiled=True,
          axis_index_groups=x_groups)

  with jax.named_scope('SwiGLU'):
    y_mlp = special2.swish2(wi0) * wi1

  # einsum(y_fused, o_wo):
  # [batch, maxlen, heads.YZ, o_wo_per_head] @
  #       [heads.YZ, o_wo_per_head, dmodel.x]
  # -> (matmul)
  # -> [batch, maxlen, dmodel.x]{YZ unreduced}
  # -> (fused reducescatter)
  # -> [batch, maxlen, dmodel.xY]
  # -> (non-fused reducescatter)
  # -> [batch.Z, maxlen, dmodel.xY]
  with jax.named_scope('o_wo'):
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)

    # do the second half of the mlp and the self-attn projection in parallel
    # allgather y_fused: [batch.B, maxlen, heads.YZX, o_wo_per_head]
    #       -> [batch, maxlen, heads.YZ, o_wo_per_head]
    if True and B == 1:
      # We don't have a B=2 collective allgather/matmul implementation yet, so
      # we use the collective matmul/reducescatter instead
      y_out = matmul_allgather(
          'bthd,hde->bte',
          y_fused,
          params.o_wo,
          rhs_split_axis=0,
          axis_name='x',
          layer=layer,
          subsplit_axis=2)
      # y_out = reducescatter(
      #     y_out, scatter_dimension=2, axis_name='y', subsplit_axis=2)
      # TODO(sholto): Test if manual faster, update
      y_out = lax.psum_scatter(y_out, 'y', scatter_dimension=2, tiled=True)
    else:
      # y_fused: [batch.B, maxlen, heads.YZX, o_wo_per_head]
      # -> (allgather)
      # -> [batch.B, maxlen, B * heads.YZ, o_wo_per_head]
      # -> (if B>1, reshape)
      # -> [batch, maxlen, heads.YZ, o_wo_per_head]
      y_fused = lax.all_gather(y_fused, axis_name='x', axis=2, tiled=True)
      if B > 1:
        two_d_parallel_xmap.assert_equal(
            y_fused.shape,
            (batch // B, max_len, heads_yz * B, hparams.o_wo_per_head))
        y_fused = jnp.reshape(
            y_fused, (batch // B, max_len, B, heads_yz, hparams.o_wo_per_head))
        y_fused = jnp.swapaxes(y_fused, 1, 2)
        y_fused = jnp.reshape(y_fused,
                              (batch, max_len, heads_yz, hparams.o_wo_per_head))

      two_d_parallel_xmap.assert_equal(
          y_fused.shape, (batch, max_len, heads_yz, hparams.o_wo_per_head))

      y_out = matmul_reducescatter(
          'bthd,hde->bte',
          y_fused,
          params.o_wo,
          scatter_dimension=(2, 2),
          axis_name='y',
          layer=layer,
          subsplit_axis=2)

    if shard_seqlen_vs_batch:
      y_out = reducescatter(
          y_out, scatter_dimension=1, axis_name='z', subsplit_axis=0)
      # TODO(sholto): Test if manual faster, update
      y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=1, tiled=True)
    else:
      # y_out = reducescatter(
      #     y_out, scatter_dimension=0, axis_name='z', subsplit_axis=0)
      # TODO(sholto): Test if manual faster, update
      y_out = lax.psum_scatter(y_out, 'z', scatter_dimension=0, tiled=True)

    if isinstance(params, weights.QuantizedLayer):
      prev_shape = y_out.shape
      y_out = intermediate_dtype(y_out *
                                 jnp.squeeze(my_layer(params.o_wo_scale)))
      two_d_parallel_xmap.assert_equal(y_out.shape, prev_shape)

  with jax.named_scope('residual'):
    z = intermediate_dtype(y_out + x)

  # TODO(sholto): Test correctness
  k, v = k.astype(intermediate_dtype), v.astype(intermediate_dtype)
  if attn_all_to_all == AttnAllToAll.NONE:
    if shard_seqlen_vs_batch:
      return z, k, v
    else:
      return z, k, v  # TODO(sholto), does this need to be updated for .B?
  elif attn_all_to_all == AttnAllToAll.AXIS_Z:
    return z, k[:batch_zb], v[:batch_zb]
  elif attn_all_to_all == AttnAllToAll.AXES_YZ:
    return z, k[:batch_yzb], v[:batch_yzb]
  elif attn_all_to_all == AttnAllToAll.AXES_YZX:
    return z, k[:batch_xyz], v[:batch_xyz]
  else:
    return z, k, v
