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

from typing import Sequence, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.partitioning import _with_sharding_constraint

HParams = checkpoint.HParams
CheckpointSpec = checkpoint.CheckpointSpec
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer
Weights = weights.Weights


def contracting_dims_from_einsum_spec(spec):
  """Gets lax.dot_general contracting dims from an einsum spec - easier to read.
  """
  with jax.named_scope(spec):
    lhs, rhs = spec.split('->')  # e.g. ['bte,hed', 'bthd']
    a1, a2 = lhs.split(',')  # e.g. ['bte', 'hed']
    contracted = [
        c for c in set(lhs) if (c not in rhs and c in a1 and c in a2)
    ]  # e.g. ['e']
    a1_contact, a2_contract = [], []
    for c in contracted:
      a1_contact.append(a1.index(c))
      a2_contract.append(a2.index(c))
    return (tuple(a1_contact), tuple(a2_contract))


def _layernorm(x):
  """Computes t5 layer norm on the input."""
  # flaxformer/components/layer_norm.py
  # 'scale' factor is folded into downstream matrix in 'preprocess'.
  epsilon = 1e-6
  x = jnp.float32(x)
  mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
  return jnp.bfloat16(x * lax.rsqrt(mean2 + epsilon))


def _rope(sin, cos, x):
  """Applies RoPE position encoding to the tensor. Multiquery attention only."""
  # Compare
  # flaxformer/components/embedding.py;l=546
  #
  # Unlike flaxformer rope:
  # * we only process one tensor (x) at a time, not two (k and v)
  # * the decode=True support is different, and unconditionally enabled here.
  #   In Flaxformer, decode=True allows for different batch elements being at
  #   different indices in the sequence. In this codebase, we assume that's
  #   unconditionally true, although beams of the same batch are assumed to
  #   share the same index within the sequence. Additionally, the Gather from
  #   the precalculated sin/cos tables is done once at the beginning of the
  #   entire inference, rather than redundantly once per layer.
  prefix_batch, seqlen, f2 = sin.shape  # f2 = features // 2
  seqlen = x.shape[1]
  x1, x2 = jnp.split(x, 2, axis=-1)
  if x.ndim == 4:
    batch, seqlen, heads, f2 = x1.shape
    shape = (prefix_batch, batch // prefix_batch, seqlen, heads, f2)
    sin = sin[:, np.newaxis, :, np.newaxis, :]
    cos = cos[:, np.newaxis, :, np.newaxis, :]
  else:
    batch, seqlen, f2 = x1.shape
    shape = (prefix_batch, batch // prefix_batch, seqlen, f2)
    sin = sin[:, np.newaxis, :, :]
    cos = cos[:, np.newaxis, :, :]
  x1 = jnp.reshape(x1, shape)
  x2 = jnp.reshape(x2, shape)

  result1 = (x1 * cos) - (x2 * sin)
  result2 = (x2 * cos) + (x1 * sin)
  return jnp.reshape(
      jnp.concatenate(
          [jnp.bfloat16(result1), jnp.bfloat16(result2)], axis=-1), x.shape)


def pjit_transformer_layer(
    hparams, layer, params, sin,
    cos, kv_caches,
    x):
  """Forward pass through a single layer, returning output, K, V."""

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # Compare
  # flaxformer/architectures/t5/parallel_fused_decoder.py
  # flaxformer/components/attention/dense_attention.py;l=1147;
  # flaxformer/components/attention/dense_attention.py;l=252;

  # prefix_batch = sin.shape[0]
  # beam = batch // prefix_batch # TODO(reinerp): Do we need this

  # 2D: [batch.Z, time, embed.XY]
  x = _with_sharding_constraint(
      x, ('residual_batch', 'residual_time', 'residual_embed'))
  xnorm = _layernorm(x)
  # 2D: [batch, time, embed.X]
  xnorm = _with_sharding_constraint(
      xnorm, ('post_norm_batch', 'time', 'post_norm_embed'))
  # in PaLM, ff and attn are parallel so we can compute q and wi together
  q_wi = jnp.einsum('bte,hed->bthd', xnorm, my_layer(params.q_wi))
  # 2D: [batch, time, heads.YZX, None]
  q_wi = _with_sharding_constraint(q_wi,
                                   ('post_norm_batch', 'time', 'heads', 'qkv'))
  q = q_wi[:, :, :, :hparams.qkv]
  q = _rope(sin, cos, q)
  # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
  # swiGLU with full d_ff dimension, rather than 2/3 scaled
  wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
  wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]
  kv = jnp.einsum('bte,ezd->btzd', xnorm, my_layer(params.kv))
  k = kv[:, :, 0, :hparams.qkv]
  v = kv[:, :, 0, hparams.qkv:]
  k = _rope(sin, cos, k)

  y_att = jnp.bfloat16(attention.attend(q, k, v, kv_caches, layer))

  y_mlp = special2.swish2(wi0) * wi1
  # 2D: [batch, time, heads.YZX, None]
  y_mlp = _with_sharding_constraint(y_mlp,
                                    ('post_norm_batch', 'time', 'heads', None))

  y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
  # do the second half of the mlp and the self-attn projection in parallel
  y_out = jnp.einsum('bthd,hde->bte', y_fused, my_layer(params.o_wo))
  # 2D: [batch.Z, time, embed.XY]
  y_out = _with_sharding_constraint(
      y_out, ('residual_batch', 'residual_time', 'residual_embed'))
  z = y_out + x
  z = _with_sharding_constraint(
      z, ('residual_batch', 'residual_time', 'residual_embed'))
  return jnp.bfloat16(z), k, v


################################################################################
################# Quantized weights and layer fprop ############################
################################################################################


def _scaled_layernorm(x, scale):
  """Computes t5 layer norm on the input."""
  # flaxformer/components/layer_norm.py
  # 'scale' factor is folded into downstream matrix in 'preprocess'.
  epsilon = 1e-6
  x = jnp.float32(x)
  mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
  # dbg('mean2', layer, jnp.mean(mean2))
  y = x * lax.rsqrt(mean2 + epsilon)
  scale += 1.0  # 'center_scale_at_zero' option in T5X
  return jnp.bfloat16(y * scale)


def quantized_dot_general(spec,
                          a,
                          w,
                          w_scale,
                          input_dtype=jnp.bfloat16,
                          accum_dtype=jnp.bfloat16):
  """Performs a @ w_quantized and scales by w_scale terms."""

  a, w = a.astype(input_dtype), w.astype(input_dtype)
  dimension_numbers = (contracting_dims_from_einsum_spec(spec), ((), ()))
  # TODO(sholto): Remove once cl/476805949 is submitted.
  a = jax.lax.dot_general(
      a, w, dimension_numbers, preferred_element_type=accum_dtype)
  return a * w_scale.squeeze()


def quantized_pjit_transformer_layer(
    hparams, layer, params, sin,
    cos, kv_caches,
    x):
  """Forward pass through a single layer, returning output, K, V."""

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # Compare

  # prefix_batch = sin.shape[0]

  x = _with_sharding_constraint(
      x, ('residual_batch', 'residual_time', 'residual_embed'))
  # When quantized, we do not fold in layernorm scale to the weights
  xnorm = _scaled_layernorm(x, my_layer(params.layernorm_scale))
  xnorm = _with_sharding_constraint(
      xnorm, ('post_norm_batch', 'time', 'post_norm_embed'))

  q_wi = quantized_dot_general('bte,hed->bthd', xnorm, my_layer(params.q_wi),
                               my_layer(params.q_wi_scale))
  # 2D: [batch, time, heads.YZX, None]
  q_wi = _with_sharding_constraint(q_wi,
                                   ('post_norm_batch', 'time', 'heads', 'qkv'))
  q = q_wi[:, :, :, :hparams.qkv]
  q = _rope(sin, cos, q)
  wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
  wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]

  kv = quantized_dot_general('bte,ezd->btzd', xnorm, my_layer(params.kv),
                             my_layer(params.kv_scale))
  k = kv[:, :, 0, :hparams.qkv]
  v = kv[:, :, 0, hparams.qkv:]
  k = _rope(sin, cos, k)

  y_att = attention.attend(q, k, v, kv_caches, layer)

  y_mlp = special2.swish2(wi0) * wi1
  y_mlp = _with_sharding_constraint(y_mlp,
                                    ('post_norm_batch', 'time', 'heads', None))

  y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
  y_out = quantized_dot_general('bthd,hde->bte', y_fused, my_layer(params.o_wo),
                                my_layer(params.o_wo_scale))
  # 2D: [batch.Z, time, embed.XY]
  y_out = _with_sharding_constraint(
      y_out, ('residual_batch', 'residual_time', 'residual_embed'))
  z = y_out + x
  z = _with_sharding_constraint(
      z, ('residual_batch', 'residual_time', 'residual_embed'))

  return jnp.bfloat16(z), k, v
