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

"""Code required to run non-parallel layer ablation."""

from typing import Sequence, Tuple

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy
import numpy as np

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

# pylint: disable = invalid-name
# pylint: disable = protected-access


@struct.dataclass
class SerialLayer:
  """Weights for the Transformer layers of PaLM."""

  q: jnp.ndarray
  wi: jnp.ndarray
  kv: jnp.ndarray
  o: jnp.ndarray
  wo: jnp.ndarray


def transformer_layer_weight_stationary_serial(
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
    swiglu = True,
):
  """Serial transformer layer block."""

  if isinstance(params, weights.QuantizedLayer):
    raise NotImplementedError

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  if latency_collectives:
    matmul_reducescatter = collectives.matmul_reducescatter_latency
    reducescatter = collectives.reducescatter_latency
    matmul_allgather = collectives.allgather_matmul_latency
  else:
    matmul_reducescatter = collectives.matmul_reducescatter_throughput
    reducescatter = collectives.reducescatter_throughput
    matmul_allgather = collectives.allgather_matmul_throughput

  heads_yz = hparams.heads // (y_axis * z_axis)
  if heads_yz >= x_axis:
    B = 1
    X = x_axis
  else:
    # print('Heads yz, x axis', heads_yz, x_axis)
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

  # prefix_batch = sin.shape[0]
  batch_z, max_len, _ = x.shape
  batch = batch_z * z_axis
  batch_xyz = batch // (x_axis * y_axis * z_axis)
  batch_yz = batch // (y_axis * z_axis)
  batch_yzb = batch_yz // B
  batch_zb = batch // (z_axis * B)

  # einsum(xnorm, q_wi):
  # [batch, maxlen, dmodel.x] @ [heads.YZ, dmodel.x, q_wi_per_head]
  # -> (matmul)
  # -> [batch, maxlen, heads.YZ, q_wi_per_head]{x unreduced}
  # -> (reducescatter over x into X heads, B batches)
  # -> [batch.B, maxlen, heads.YZX, q_wi_per_head]
  # TODO(reinerp): For chips>64, need to reducescatter over batch instead.
  # print('---- begin fwd ---------')
  # print(hparams)
  # print('residual_embed', x_axis * y_axis)
  # print('residual_batch', z_axis)
  # print('---- end globals --------')
  # print('x', x.shape, 'batch/z), seqlen, embed/(x*y)')

  with jax.named_scope('allgather_layernorm'):
    # allgather xnorm: [batch.Z, maxlen, dmodel.XY]
    # -> [batch.Z, maxlen, dmodel.X]    (xnorm_z)
    # -> [batch, maxlen, dmodel.X]
    xgather = x
    xgather = lax.all_gather(xgather, 'y', axis=2, tiled=True)

    epsilon = 1e-6
    xgather = jnp.float32(xgather)
    mean2 = lax.pmean(
        jnp.mean(lax.square(xgather), axis=-1, keepdims=True), axis_name='x'
    )
    xnorm_z = jnp.bfloat16(xgather * lax.rsqrt(mean2 + epsilon))
    xnorm = lax.all_gather(xnorm_z, 'z', axis=0, tiled=True)
  # print(f'xnorm {xnorm.shape}, [batch, seqlen, embed]')

  with jax.named_scope('attention'):
    with jax.named_scope('q'):
      # print('--------Q: --------')

      # [B, T, E.x] @ [H.yz, E.X, qkv] - > [B, T, H.yzx, qkv]
      if B == 1:
        q = matmul_reducescatter(
            'bte,hed->bthd',
            xnorm,
            params.q,
            scatter_axis=0,
            axis_name='x',
            layer=layer,
            subsplit_axis=2 if latency_collectives else 0,
        )
      else:
        q_unreduced = jnp.einsum('bte,hed->bthd', xnorm, my_layer(params.q))
        # Combine batch into heads, reducescatter over heads,
        # split batch back out.
        two_d_parallel_xmap.assert_equal(
            q_unreduced.shape, (batch, max_len, heads_yz, hparams.qkv)
        )
        q_unreduced = jnp.reshape(
            q_unreduced, (B, batch // B, max_len, heads_yz, hparams.qkv)
        )
        q_unreduced = jnp.transpose(q_unreduced, (1, 2, 0, 3, 4))
        q_unreduced = jnp.reshape(
            q_unreduced, (batch // B, max_len, B * heads_yz, hparams.qkv)
        )
        q = collectives.reducescatter_latency(
            q_unreduced, scatter_dimension=2, axis_name='x'
        )

      if isinstance(params, inference.QuantizedLayer):
        raise NotImplementedError
      # print(
      #     f'Attention: xnorm: {xnorm.shape}, [B, T, E.x]
      # q_weight: {params.q.shape} [H.yz, E.X, qkv]
      # -> q {q.shape} [B,T,H.xyz,D]'
      # )

    with jax.named_scope('kv'):
      # y_index = lax.axis_index('y')
      # TODO(reinerp): Consider using xnorm instead of xnorm_z in NONE case?
      # I don't know yet if that's better.

      # if attn_all_to_all.value >= partitioning.AttnAllToAll.AXES_YZ.value:
      #   xnorm_sliced = lax.dynamic_slice_in_dim(
      #       xnorm_z, y_index * batch_yz, batch_yz, axis=0)
      # else:
      #   xnorm_sliced = xnorm_z

      # [B, T, E.x], [E.x, h.yz, qkv*2] -> [B, T, H.xyz, qkv*2]
      # print(
      #     f'kv: xnorm: {xnorm.shape}, [B, T, E.x]
      # kv_weight: {params.kv.shape} [E.x, h.yz, qkv*2]'
      # )
      kv_unreduced = matmul_reducescatter(
          'bte,ehd->bthd',
          xnorm,
          params.kv,
          scatter_axis=1,
          axis_name='x',
          layer=layer,
          subsplit_axis=2,
      )
      # print(
      #     f'kv: xnorm: {xnorm.shape}, [B, T, E.x] kv_weight:
      #     {params.kv.shape} [E.x, h.yz, qkv*2] -> kv {kv_unreduced.shape}
      #     [B, T, H.xyz, qkv*2]'
      # )
      # kv_unreduced = jnp.einsum('bte,ehd->bthd', xnorm_sliced,
      #                           my_layer(params.kv))
      if attn_all_to_all == partitioning.AttnAllToAll.NONE:
        # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
        # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
        # --slice--> [batch.ZB, maxlen, 1, 2*qkv]
        # --AGZ-->   [batch.B, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        kv = lax.dynamic_slice_in_dim(kv, b_index * batch_zb, batch_zb, axis=0)
        kv = lax.all_gather(kv, 'z', axis=0, tiled=True)

      elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
        # [batch.Z, maxlen, 1, 2*qkv]{x_unreduced}
        # --ARx-->   [batch.Z, maxlen, 1, 2*qkv]
        # --slice--> [batch.ZB, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        kv = lax.dynamic_slice_in_dim(kv, b_index * batch_zb, batch_zb, axis=0)
      elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
        # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
        # --ARx-->   [batch.YZ, maxlen, 1, 2*qkv]
        # --slice--> [batch.YZB, maxlen, 1, 2*qkv]
        kv = lax.psum(kv_unreduced, 'x')
        kv = lax.dynamic_slice_in_dim(
            kv, b_index * batch_yzb, batch_yzb, axis=0
        )
      elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
        # [batch.YZ, maxlen, 1, 2*qkv]{x_unreduced}
        # --RSx-->   [batch.YZXB, maxlen, 1, 2*qkv]
        assert (
            batch_xyz >= 1
        ), 'Batch size too small for AXES_XYZ and this chip count'
        kv = lax.psum_scatter(
            kv_unreduced, 'x', scatter_dimension=0, tiled=True
        )

      if isinstance(params, inference.QuantizedLayer):
        prev_shape = kv.shape
        kv = jnp.bfloat16(kv * jnp.squeeze(my_layer(params.kv_scale)))
        two_d_parallel_xmap.assert_equal(prev_shape, kv.shape)

      k = kv[:, :, :, : hparams.qkv]
      v = kv[:, :, :, hparams.qkv :]

    # print(f'q {q.shape}, k {k.shape}, v {v.shape}')
    with jax.named_scope('attn'):
      k = _rope(sin, cos, k)

      # print(f'batch_yzb: {batch_yzb}')
      # q: [batch.B, maxlen, heads.YZX, qkv]
      # -> { NONE:                   [batch.B, maxlen, heads.YZX, qkv]
      #    { AXIS_Z:                 [batch.ZB, maxlen, heads.YX, qkv]
      #    { AXES_YZ:                [batch.YZB, maxlen, heads.X, qkv]
      #    { AXES_YZX:               [batch.YZXB, maxlen, heads, qkv]
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
            q,
            axis_name='x',
            split_axis=0,
            concat_axis=2,
            tiled=True,
            axis_index_groups=x_groups,
        )
        q = lax.all_to_all(
            q, axis_name=('y', 'z'), split_axis=0, concat_axis=2, tiled=True
        )

      q = _rope(sin, cos, q)
      caches = []
      for cache in kv_caches:
        cache = cache.replace(k=jnp.swapaxes(cache.k, 0, 3))
        cache = cache.replace(v=jnp.swapaxes(cache.v, 0, 3))
        caches.append(cache)

      y_att = jnp.bfloat16(attention.attend(q, k, v, caches, layer))
      # y_att:
      #    { NONE:                   [batch.B, maxlen, heads.YZX, qkv]
      #    { AXIS_Z:                 [batch.ZB, maxlen, heads.YX, qkv]
      #    { AXES_YZ:                [batch.YZB, maxlen, heads.X, qkv]
      #    { AXES_YZX:               [batch.YZXB, maxlen, heads, qkv]
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
            y_att,
            axis_name='x',
            split_axis=2,
            concat_axis=0,
            tiled=True,
            axis_index_groups=x_groups,
        )
  # print('y_att', y_att.shape)

  with jax.named_scope('projection'):
    if False and B == 1:
      # We don't have a B=2 collective allgather/matmul implementation yet, so
      # we use the collective matmul/reducescatter instead
      # print(f'o_wo: {params.o_wo.shape}')
      # print('collective projection')
      # [B,T H, D] @ [H, D ,E] -> [B,T,E]
      # FLOPs = B * T * (HDE)*2
      #
      proj_out = matmul_allgather(
          'bthd,hde->bte',
          y_att,
          params.o,
          rhs_split_axis=0,
          axis_name='x',
          layer=layer,
          subsplit_axis=2,
      )
      proj_out = reducescatter(
          proj_out, scatter_dimension=2, axis_name='y', subsplit_axis=2
      )
    else:
      # y_att: [batch.B, maxlen, heads.YZX, qkv]
      # -> (allgather)
      # -> [batch.B, maxlen, B * heads.YZ, qkv]
      # -> (if B>1, reshape)
      # -> [batch, maxlen, heads.YZ, qkv]
      y_att = lax.all_gather(y_att, axis_name='x', axis=2, tiled=True)
      if B > 1:
        two_d_parallel_xmap.assert_equal(
            y_att.shape, (batch // B, max_len, heads_yz * B, hparams.qkv)
        )
        y_att = jnp.reshape(
            y_att, (batch // B, max_len, B, heads_yz, hparams.qkv)
        )
        y_att = jnp.swapaxes(y_att, 1, 2)
        y_att = jnp.reshape(y_att, (batch, max_len, heads_yz, hparams.qkv))

      two_d_parallel_xmap.assert_equal(
          y_att.shape, (batch, max_len, heads_yz, hparams.qkv)
      )
      # print(f'y_att', y_att.shape, 'proj', params.o.shape)

      proj_out = matmul_reducescatter(
          'bthd,hde->bte',
          y_att,
          params.o,
          scatter_axis=2,
          axis_name='y',
          layer=layer,
          subsplit_axis=2,
      )

    proj_out = reducescatter(
        proj_out, scatter_dimension=0, axis_name='z', subsplit_axis=0
    )
  # print(f'x {x.shape} y out: {proj_out.shape}')

  with jax.named_scope('add back'):
    x2 = x + proj_out

  with jax.named_scope('allgather_layernorm2'):
    # allgather xnorm: [batch.Z, maxlen, dmodel.XY]
    # -> [batch.Z, maxlen, dmodel.X]    (xnorm_z)
    # -> [batch, maxlen, dmodel.X]
    xgather2 = x2
    xgather2 = lax.all_gather(xgather2, 'y', axis=2, tiled=True)

    epsilon = 1e-6
    xgather2 = jnp.float32(xgather2)
    mean2 = lax.pmean(
        jnp.mean(lax.square(xgather2), axis=-1, keepdims=True), axis_name='x'
    )
    xnorm_z2 = jnp.bfloat16(xgather2 * lax.rsqrt(mean2 + epsilon))
    xnorm2 = lax.all_gather(xnorm_z2, 'z', axis=0, tiled=True)
  # print(f'xnorm {xnorm2.shape}')

  with jax.named_scope('mlp'):
    if B == 1:
      wi = matmul_reducescatter(
          'bte,hed->bthd',
          xnorm2,
          params.wi,
          scatter_axis=0,
          axis_name='x',
          layer=layer,
          subsplit_axis=2 if latency_collectives else 0,
      )
    else:
      wi_unreduced = jnp.einsum('bte,hed->bthd', xnorm2, my_layer(params.wi))
      # Combine batch into heads, reducescatter over heads,
      # split batch back out.
      # print(swiglu)
      if swiglu:
        q_wi_per_head = hparams.q_wi_per_head
      else:
        q_wi_per_head = hparams.o_wo_per_head

      # print('q_wi_per_head', q_wi_per_head, hparams.o_wo_per_head)

      two_d_parallel_xmap.assert_equal(
          wi_unreduced.shape,
          (batch, max_len, heads_yz, q_wi_per_head - hparams.qkv),
      )
      wi_unreduced = jnp.reshape(
          wi_unreduced,
          (B, batch // B, max_len, heads_yz, q_wi_per_head - hparams.qkv),
      )
      wi_unreduced = jnp.transpose(wi_unreduced, (1, 2, 0, 3, 4))
      wi_unreduced = jnp.reshape(
          wi_unreduced,
          (batch // B, max_len, B * heads_yz, q_wi_per_head - hparams.qkv),
      )
      wi = collectives.reducescatter_latency(
          wi_unreduced, scatter_dimension=2, axis_name='x'
      )

    if isinstance(params, inference.QuantizedLayer):
      raise NotImplementedError

    # print('wi', wi.shape, hparams.ff, hparams.heads)
    if swiglu:
      wi0 = wi[:, :, :, : (hparams.ff // hparams.heads)]
      wi1 = wi[:, :, :, (hparams.ff // hparams.heads) :]

      with jax.named_scope('SwiGLU'):
        y_mlp = special2.swish2(wi0) * wi1  # [batch.XYZ, t, h, ff_per_head]
    else:
      y_mlp = special2.swish2(wi)

  # print(f'y_mlp {y_mlp.shape}')

  with jax.named_scope('wo'):
    if False and B == 1:
      # We don't have a B=2 collective allgather/matmul implementation yet, so
      # we use the collective matmul/reducescatter instead
      # print(f'o_wo: {params.o_wo.shape}')
      # print('collective projection')
      y_out = matmul_allgather(
          'bthd,hde->bte',
          y_mlp,
          params.wo,
          rhs_split_axis=0,
          axis_name='x',
          layer=layer,
          subsplit_axis=2,
      )
      with jax.named_scope('post_wo_reduce_scatter'):
        y_out = reducescatter(
            y_out, scatter_dimension=2, axis_name='y', subsplit_axis=2
        )
    else:
      # y_mlp: [batch.B, maxlen, heads.YZX, qkv]
      # -> (allgather)
      # -> [batch.B, maxlen, B * heads.YZ, qkv]
      # -> (if B>1, reshape)
      # -> [batch, maxlen, heads.YZ, qkv]
      y_mlp = lax.all_gather(y_mlp, axis_name='x', axis=2, tiled=True)
      if B > 1:
        two_d_parallel_xmap.assert_equal(
            y_mlp.shape,
            (
                batch // B,
                max_len,
                heads_yz * B,
                hparams.o_wo_per_head - hparams.qkv,
            ),
        )
        y_mlp = jnp.reshape(
            y_mlp,
            (
                batch // B,
                max_len,
                B,
                heads_yz,
                hparams.o_wo_per_head - hparams.qkv,
            ),
        )
        y_mlp = jnp.swapaxes(y_mlp, 1, 2)
        y_mlp = jnp.reshape(
            y_mlp,
            (batch, max_len, heads_yz, hparams.o_wo_per_head - hparams.qkv),
        )

      two_d_parallel_xmap.assert_equal(
          y_mlp.shape,
          (batch, max_len, heads_yz, hparams.o_wo_per_head - hparams.qkv),
      )
      # print(f'y_mlp', y_mlp.shape, 'proj', params.o.shape)

      y_out = matmul_reducescatter(
          'bthd,hde->bte',
          y_mlp,
          params.wo,
          scatter_axis=2,
          axis_name='y',
          layer=layer,
          subsplit_axis=2,
      )

    y_out = reducescatter(
        y_out, scatter_dimension=0, axis_name='z', subsplit_axis=0
    )

    with jax.named_scope('residual'):
      z = jnp.bfloat16(y_out + x)
    # print("k,", k.shape, "v", v.shape, batch_xyz)
    # print(jnp.swapaxes(k, 0, 1).shape, jnp.swapaxes(v, 0, 1).shape, batch_xyz)
    return z, k[:batch_xyz], v[:batch_xyz]


def transformer_layer_weight_gathered_serial(
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
):
  """Serial form of weight gathered transformer block."""

  if isinstance(params, inference.QuantizedLayer):
    raise NotImplementedError

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # x: [batch.XYZ, t, e]
  with jax.named_scope('excess_slicing'):
    b, _, _ = x.shape
    # print(x.shape)
    q_wi_weights = my_layer(params.q_wi)  # [h, e, q_wi_per_head]
    q_weight = q_wi_weights[:, :, : hparams.qkv]  # [h ,e , q]
    wi_weight = q_wi_weights[:, :, hparams.qkv :]  # [h ,e ,ff]
    o_wo_weights = my_layer(params.o_wo)  # [h.YZ, d, e.X]
    proj = o_wo_weights[:, : hparams.qkv, :]
    wo_weights = o_wo_weights[:, hparams.qkv :, :]
  # print(x_axis, y_axis, z_axis)
  # print(f'q_wi {q_wi_weights.shape}, o_wo {o_wo_weights.shape}')
  # print(
  #     f'q_weight {q_weight.shape} wi_weight {wi_weight.shape},
  #       proj {proj.shape} wo_weights {wo_weights.shape}'
  # )

  with jax.named_scope('layernorm1'):
    # No need to communicate across batch, so everything is local
    x_prec = jnp.float32(x)
    epsilon = 1e-6
    mean2 = jnp.mean(lax.square(x_prec), axis=-1, keepdims=True)
    xnorm = jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))

  batch_xyz = b // (x_axis * y_axis * z_axis)

  with jax.named_scope('attention'):
    q = collectives.matmul_collective_weights_gather_q_wi(
        'bte,hed->bthd', xnorm, q_weight, lhs_split_axis=2
    )  #   -> [batch.XYZ, t, h, q]

    with jax.named_scope('kv'):
      # [batch.XYZ, t, e] @ [e, 1, 2*qkv] -> [batch.XYZ, t, 1, 2*qkv]
      # Two options here:
      # a) Split along x, and then all reduce along x
      # b) We fully replicate kv
      kv = jnp.einsum('bte,ezd->btzd', xnorm, my_layer(params.kv))

      if isinstance(params, inference.QuantizedLayer):
        prev_shape = kv.shape
        kv = jnp.bfloat16(kv * jnp.squeeze(my_layer(params.kv_scale)))
        two_d_parallel_xmap.assert_equal(prev_shape, kv.shape)

      k = kv[:, :, 0, : hparams.qkv]  # [batch.XYZ, t, qkv]
      v = kv[:, :, 0, hparams.qkv :]  # [batch.XYZ, t, qkv]

    with jax.named_scope('attend'):
      k = _rope(sin, cos, k)  # [batch.XYZ, t, qkv]
      q = _rope(sin, cos, q)  # [batch.XYZ, t, h, qkv]

      # [batch.XYZ, t, h, qkv]
      y_att = jnp.bfloat16(attention.attend(q, k, v, kv_caches, layer))
    # print(f'y_att {y_att.shape}')

  with jax.named_scope('projection'):
    gathered_weights = jax.lax.all_gather(proj, 'x', axis=2, tiled=True)
    gathered_weights = jax.lax.all_gather(
        gathered_weights, ('y', 'z'), axis=0, tiled=True
    )

    x = jnp.einsum('bthd,hde->bte', y_att, gathered_weights)

  with jax.named_scope('layernorm2'):
    # No need to communicate across batch, so everything is local
    x_prec = jnp.float32(x)
    epsilon = 1e-6
    mean2 = jnp.mean(lax.square(x_prec), axis=-1, keepdims=True)
    xnorm = jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))

  with jax.named_scope('mlp'):
    # print(f'xnorm {xnorm.shape} wi_weight {wi_weight.shape}')
    wi = collectives.matmul_collective_weights_gather_q_wi(
        'bte,hed->bthd', xnorm, wi_weight, lhs_split_axis=2
    )  #   -> [batch.XYZ, t, h, wi_per_head]

    # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
    # swiGLU with full d_ff dimension, rather than 2/3 scaled
    # print('wi', wi.shape, hparams.ff, hparams.heads)
    wi0 = wi[:, :, :, : (hparams.ff // hparams.heads)]
    wi1 = wi[:, :, :, (hparams.ff // hparams.heads) :]

  with jax.named_scope('SwiGLU'):
    y_mlp = special2.swish2(wi0) * wi1  # [batch.XYZ, t, h, ff_per_head]

  with jax.named_scope('wo'):
    y_out = collectives.matmul_collective_weights_gather_o_wo(
        'bthd,hde->bte', y_mlp, wo_weights, lhs_split_axis=2
    )  # -> [batch.XYZ, t, e]

  with jax.named_scope('residual'):
    z = jnp.bfloat16(y_out + x)

  return z, k[:batch_xyz], v[:batch_xyz]
