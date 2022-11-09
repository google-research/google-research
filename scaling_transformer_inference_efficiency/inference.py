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

"""Minimalist codebase for PaLM model inference.

Relative to the t5x implementation of PaLM, this codebase does not aim for
configurability, and instead aims for peak performance inference, including in
ways that would require significant changes to how t5x's APIs are structured.

Test this with :inference_test
"""

from functools import lru_cache  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
from typing import Callable, Optional, Sequence, Tuple

from flax import struct
import jax
from jax import lax
from jax import pxla
from jax import sharding
from jax.experimental import pjit
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P
import jax.numpy as jnp
import jax.scipy
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import FullChunkResult

HParams = checkpoint.HParams
CheckpointSpec = checkpoint.CheckpointSpec


# cache this until the cpp pathway is built
@lru_cache
def create_mesh_pspec_sharding(mesh, pspec):
  return sharding.MeshPspecSharding(mesh, pspec)


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


def _generate_fixed_pos_embedding(features,
                                  length,
                                  min_timescale=1.0,
                                  max_timescale=10000.0):
  """Generate Sin/Cos for Rotary Embeddings.

  Generates sinusoids at (features//2) different timescales, where the
  timescales form a geometric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).

  Sinusoids are evaluated at integer positions i in [0, length).

  The outputs are computed as:

    output_sin[i, j] = sin(i / timescale[j])
    output_cos[i, j] = cos(i / timescale[j])

  Args:
    features: an integer
    length: an integer
    min_timescale: an optional float
    max_timescale: an optional float

  Returns:
    output_sin: a float32 Tensor with shape [length, features // 2]
    output_cos: a float32 Tensor with shape [length, features // 2]
  """
  # Forked from
  # flaxformer/components/embedding.py;l=592
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = min_timescale * (max_timescale / min_timescale)**fraction
  rotational_frequency = 1. / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(length),
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST)
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


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


def copy_to_device_with_mesh(mesh, x, spec, expected):
  spec = partitioning.logical_to_physical(spec)
  s = create_mesh_pspec_sharding(mesh, spec)
  return partitioning.copy_to_device(x, s, expected)


################################################################################
################# Unquantized weights and layer fprop ##########################
################################################################################
@struct.dataclass
class Layer:
  """Weights for the Transformer layers of PaLM."""
  q_wi: jnp.ndarray
  kv: jnp.ndarray
  o_wo: jnp.ndarray


@struct.dataclass
class Weights:
  """Weights for a model, as stored in HBM.

  This layout may differ from Checkpoint layout, as it is optimized for
  inference.
  """
  layer: Layer

  # weights.sin and weights.cos are precomputed tables of sin and cos at various
  # frequencies as specified by the rotary position encoding. These are not
  # trained, so they don't show up in the checkpoint file on disk. Since they're
  # the same across every inference call, we precalculate them at model load
  # time and keep them in HBM alongside the weights.
  #
  # An alternative would be to compute these sin and cos values on the fly on
  # every layer, rather than precomputing them and storing them in HBM.
  # That is almost as good but not quite.
  sin: jnp.ndarray
  cos: jnp.ndarray
  embedding: jnp.ndarray

  @classmethod
  def make_shaped_arrays(cls, h):
    """Creates weights populated with zero-footprint shaped arrays."""
    q_wi = jax.ShapedArray((h.layers, h.heads, h.embed, h.q_wi_per_head),
                           jnp.bfloat16)
    kv = jax.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.bfloat16)
    o_wo = jax.ShapedArray((h.layers, h.heads, h.o_wo_per_head, h.embed),
                           jnp.bfloat16)
    sin = jax.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    cos = jax.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    embedding = jax.ShapedArray((h.vocab, h.embed), jnp.bfloat16)
    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)

  @classmethod
  def logical_axes(cls):
    """Returns the partition specs for the weights in their logical axes."""
    q_wi = P('layers', 'heads', 'embed', 'query')
    kv = P('layers', 'embed', None, 'query')
    o_wo = P('layers', 'heads', 'query', 'embed')
    sin = P(None, None)
    cos = P(None, None)
    embedding = P('table_vocab', 'table_embed')

    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)

  @classmethod
  def physical_axes(cls):
    """Returns the partition specs for the weights in their physical axes."""
    return jax.tree_map(partitioning.logical_to_physical,
                        Weights.logical_axes())

  @classmethod
  def from_checkpoint(cls, h, mesh,
                      c):
    """Initializes weights in HBM from the checkpoint."""

    axes = Weights.logical_axes()

    def fold_in_wi0_constants(q_wi):
      hidden_channel_iota = jax.lax.broadcasted_iota(jnp.int32, q_wi.shape, 3)
      wi0_mask = (hidden_channel_iota >= h.qkv) & (
          hidden_channel_iota < (h.qkv + (h.ff // h.heads)))
      # Constant 0.5: We need to multiply wi_0 by 0.5 to correct special2.swish2
      # to be equivalent to jnp.swish. More efficient to do this once to the
      # weights than every time we call the fn.
      wi0_constants = 0.5
      return q_wi * jnp.where(wi0_mask, wi0_constants, 1.0)

    def fold_in_q_constants(q_wi):
      hidden_channel_iota = jax.lax.broadcasted_iota(jnp.int32, q_wi.shape, 3)
      q_mask = hidden_channel_iota < h.qkv
      # Constant LOG2_E: comes from using special2.exp2 instead of lax.exp.
      # Constant lax.rsqrt(h.qkv): comes from Transformer attention definition.
      q_constants = special2.LOG2_E * lax.rsqrt(jnp.float32(h.qkv))
      return q_wi * jnp.where(q_mask, q_constants, 1.0)

    def fold_in_layernorm(q_wi, kv,
                          layernorm_scale):
      # Fold in layernorm scale to remove a multiplication
      layernorm_scale = 1.0 + layernorm_scale[:, :, np.newaxis, np.newaxis]
      return q_wi * layernorm_scale, kv * layernorm_scale

    def fold_in_unembedding_constants(o_wo,
                                      embedding):
      # Constant LOG2_E: comes from using special2.exp2 instead of lax.exp.
      # Constant lax.rsqrt(h.embed): comes from t5x definition.
      unembedding_constants = special2.LOG2_E * lax.rsqrt(jnp.float32(h.embed))

      # Define `s=unembedding_constants`. Mathematically we'd like to apply `s`
      # on the last use of the embedding table but not the first use. Since the
      # weights of the first and last use are tied together, this is tricky. We
      # achieve a mathematically identical effect by folding in the factor `s`
      # into _both_ `embedding` and `o_wo`.
      #
      # We now explain why that's mathematically identical. Recall the math of
      # PaLM:
      #
      #   x[0] = embedding[token_ids]
      #   for i in range(layers):
      #     xnorm[i] = layernorm(x[i])
      #     y[i] = concat(
      #       attention_no_output_proj(xnorm[i]),
      #       swish(wi0 @ xnorm[i]) * (wi1 @ xnorm[i]))
      #     z[i] = o_wo[i] @ y[i]
      #     x[i+1] = z[i] + x[i]
      #   pre_logits = layernorm(x[-1])
      #   logits = pre_logits @ embedding
      #
      # Suppose we multiply `embedding` and `o_wo` by `s` and run the same
      # computation. We'll use `x2[i]`, `xnorm2[i]`, etc to refer to the
      # intermediates in this modified computation. Then the following are true:
      #
      #   x2[0]     = (s * embedding)[token_ids] = s * x[0]
      #   xnorm2[0] = layenorm(x2[0])
      #             = layernorm(s * x[0])
      #             = layernorm(x[0])
      #             = xnorm[0]
      #   y2[0]     = y[0] (because y[i] depends only on xnorm[i])
      #   z2[0]     = (s * o_wo[0]) @ y2[0]
      #             = s * (o_wo[0] @ y[0])
      #             = s * z[0]
      #   x2[1]     = x2[0] + z2[0]
      #             = (s * x[0]) + (s * z[0])
      #             = s * (x[0] + z[0])
      #             = s * x[1]
      #   ... (continues likewise for all transformer layers) ...
      #   x2[-1]    = s * x[-1]
      #   pre_logits2 = layernorm(x2[-1])
      #               = layernorm(s * x[-1])
      #               = layernorm(x[-1])
      #               = pre_logits
      #   logits2   = pre_logits2 @ (s * embedding)
      #             = s * (pre_logits @ embedding)
      #             = s * logits
      #
      # This shows that the end result of multiplying `embedding` and `o_wo` by
      # `s` is that `logits` gets multiplied by `s`, which is what we were
      # trying to achieve.
      return o_wo * unembedding_constants, embedding * unembedding_constants

    def preprocess(q_wi, kv, o_wo, layernorm_scale, embedding):
      q_wi, kv = fold_in_layernorm(
          jnp.float32(q_wi), jnp.float32(kv), layernorm_scale)
      q_wi = fold_in_q_constants(q_wi)
      q_wi = fold_in_wi0_constants(q_wi)
      o_wo, embedding = fold_in_unembedding_constants(
          jnp.float32(o_wo), jnp.float32(embedding))

      # Change layout:
      #   (layers, embed, heads, query) -> (layers, heads, embed, query)
      # to avoid XLA doing that same transformation on every inference.
      q_wi = jnp.swapaxes(q_wi, 1, 2)
      return jnp.bfloat16(q_wi), jnp.bfloat16(kv), jnp.bfloat16(
          o_wo), jnp.bfloat16(embedding)

    expected_shapes = Weights.make_shaped_arrays(h)

    copy_to_device = partial(copy_to_device_with_mesh, mesh)

    sin, cos = _generate_fixed_pos_embedding(h.qkv, h.max_len)
    sin = copy_to_device(sin, axes.sin, expected_shapes.sin)
    cos = copy_to_device(cos, axes.cos, expected_shapes.cos)

    q_wi_input_axes = ('layers', 'embed', 'heads', 'query')
    q_wi = copy_to_device(
        c.q_wi, q_wi_input_axes,
        jax.ShapedArray((h.layers, h.embed, h.heads, h.q_wi_per_head),
                        jnp.bfloat16))
    kv = copy_to_device(c.kv, axes.layer.kv, expected_shapes.layer.kv)
    o_wo = copy_to_device(c.o_wo, axes.layer.o_wo, expected_shapes.layer.o_wo)
    layernorm_scale_axes = ('layers', 'embed')
    layernorm_scale = copy_to_device(
        c.layernorm_scale, layernorm_scale_axes,
        jax.ShapedArray((h.layers, h.embed), jnp.float32))
    embedding = copy_to_device(c.embedding, axes.embedding,
                               expected_shapes.embedding)

    q_wi_input_axes = partitioning.logical_to_physical(q_wi_input_axes)
    q_wi_output_axes = partitioning.logical_to_physical(axes.layer.q_wi)
    kv_axes = partitioning.logical_to_physical(axes.layer.kv)
    o_wo_axes = partitioning.logical_to_physical(axes.layer.o_wo)
    layernorm_scale_axes = partitioning.logical_to_physical(
        layernorm_scale_axes)
    embedding_axes = partitioning.logical_to_physical(axes.embedding)

    with mesh:
      q_wi, kv, o_wo, embedding = pjit.pjit(
          preprocess,
          in_axis_resources=(q_wi_input_axes, kv_axes, o_wo_axes,
                             layernorm_scale_axes, embedding_axes),
          out_axis_resources=(q_wi_output_axes, kv_axes, o_wo_axes,
                              embedding_axes),
          donate_argnums=(1, 2, 4))(q_wi, kv, o_wo, layernorm_scale, embedding)

    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)


def transformer_layer(
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
  batch, max_len, _ = x.shape
  # beam = batch // prefix_batch # TODO(reinerp): Do we need this

  if batch == 1 and max_len == 1:
    raise ValueError('sharded batch-1 matmul is broken on VLC, b/246436629')

  x = _with_sharding_constraint(x, ('batch', 'time', 'embed'))
  xnorm = _layernorm(x)
  xnorm = _with_sharding_constraint(xnorm, ('batch', 'time', 'embed'))
  # in PaLM, ff and attn are parallel so we can compute q and wi together
  q_wi = jnp.einsum('bte,hed->bthd', xnorm, my_layer(params.layer.q_wi))
  q_wi = _with_sharding_constraint(q_wi, ('batch', 'time', 'heads', None))
  q = q_wi[:, :, :, :hparams.qkv]
  q = _rope(sin, cos, q)
  # unlike in https://arxiv.org/pdf/2002.05202.pdf, PaLM implements
  # swiGLU with full d_ff dimension, rather than 2/3 scaled
  wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
  wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]
  kv = jnp.einsum('bte,ezd->btzd', xnorm, my_layer(params.layer.kv))
  k = kv[:, :, 0, :hparams.qkv]
  v = kv[:, :, 0, hparams.qkv:]
  k = _rope(sin, cos, k)

  y_att = jnp.bfloat16(attention.attend(q, k, v, kv_caches, layer))

  y_mlp = special2.swish2(wi0) * wi1
  y_mlp = _with_sharding_constraint(y_mlp, ('batch', 'time', 'heads', None))

  y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
  # do the second half of the mlp and the self-attn projection in parallel
  y_out = jnp.einsum('bthd,hde->bte', y_fused, my_layer(params.layer.o_wo))
  y_out = _with_sharding_constraint(y_out, ('batch', 'time', 'embed'))
  z = y_out + x
  z = _with_sharding_constraint(z, ('batch', 'time', 'embed'))
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


@struct.dataclass
class QuantizedLayer:
  """Weights for the Transformer layers of PaLM."""
  q_wi: jnp.ndarray
  q_wi_scale: jnp.ndarray
  kv: jnp.ndarray
  kv_scale: jnp.ndarray
  o_wo: jnp.ndarray
  o_wo_scale: jnp.ndarray
  layernorm_scale: jnp.ndarray


@struct.dataclass
class QuantizedWeights:
  """Weights for a model, as stored in HBM.

  This layout may differ from Checkpoint layout, as it is optimized for
  inference.
  """
  layer: QuantizedLayer

  # See unquantized weights class for notes.
  sin: jnp.ndarray
  cos: jnp.ndarray
  embedding: jnp.ndarray

  @classmethod
  def make_shaped_arrays(cls, h):
    """Creates weights populated with zero-footprint shaped arrays."""
    q_wi = jax.ShapedArray((h.layers, h.heads, h.embed, h.q_wi_per_head),
                           jnp.int8)
    q_wi_scale = jax.ShapedArray((h.layers, h.heads, 1, h.q_wi_per_head),
                                 jnp.float32)
    kv = jax.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.int8)
    kv_scale = jax.ShapedArray((h.layers, 1, 1, 2 * h.qkv), jnp.float32)
    o_wo = jax.ShapedArray((h.layers, h.heads, h.o_wo_per_head, h.embed),
                           jnp.int8)
    o_wo_scale = jax.ShapedArray((h.layers, 1, 1, h.embed), jnp.float32)
    sin = jax.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    cos = jax.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    embedding = jax.ShapedArray((h.vocab, h.embed), jnp.bfloat16)
    layernorm_scale = jax.ShapedArray((h.layers, h.embed), jnp.bfloat16)
    return QuantizedWeights(
        QuantizedLayer(q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale,
                       layernorm_scale),
        sin=sin,
        cos=cos,
        embedding=embedding)

  @classmethod
  def logical_axes(cls):
    """Returns the partition specs for the weights in their logical axes."""
    q_wi = P('layers', 'heads', 'embed', 'query')
    # Scale Axes can not shard along a singleton dimension
    q_wi_scale = P('layers', 'heads', None, 'query')
    kv = P('layers', 'embed', None, 'query')
    kv_scale = P('layers', None, None, 'query')
    o_wo = P('layers', 'heads', 'query', 'embed')
    o_wo_scale = P('layers', None, None, 'embed')
    sin = P(None, None)
    cos = P(None, None)
    # Embedding table wants different sharding than Transformer layers, to work
    # around b/244232479.
    embedding = P('table_vocab', 'table_embed')
    layernorm_scale = P('layers', 'embed')

    return QuantizedWeights(
        QuantizedLayer(q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale,
                       layernorm_scale),
        sin=sin,
        cos=cos,
        embedding=embedding)

  @classmethod
  def physical_axes(cls):
    """Returns the partition specs for the weights in their physical axes."""
    return jax.tree_map(partitioning.logical_to_physical,
                        QuantizedWeights.logical_axes())

  @classmethod
  def from_checkpoint(cls, h, mesh,
                      c):
    """Initializes weights in HBM, copying from a host-resident checkpoint."""

    axes = QuantizedWeights.logical_axes()

    def fold_in_wi0_constants(q_wi_scale):
      hidden_channel_iota = jax.lax.broadcasted_iota(jnp.int32,
                                                     q_wi_scale.shape, 3)
      wi0_mask = (hidden_channel_iota >= h.qkv) & (
          hidden_channel_iota < (h.qkv + (h.ff // h.heads)))
      # Constant 0.5: We need to multiply wi_0 by 0.5 to correct special2.swish2
      # to be equivalent to jnp.swish. More efficient to do this once to the
      # weights than every time we call the fn.
      wi0_constants = 0.5
      return q_wi_scale * jnp.where(wi0_mask, wi0_constants, 1.0)

    def fold_in_q_constants(q_wi_scale):
      hidden_channel_iota = jax.lax.broadcasted_iota(jnp.int32,
                                                     q_wi_scale.shape, 3)
      q_mask = hidden_channel_iota < h.qkv
      # Constant LOG2_E: comes from using special2.exp2 instead of lax.exp.
      # Constant lax.rsqrt(h.qkv): comes from Transformer attention definition.
      q_constants = special2.LOG2_E * lax.rsqrt(jnp.float32(h.qkv))
      return q_wi_scale * jnp.where(q_mask, q_constants, 1.0)

    def fold_in_unembedding_constants(o_wo_scale, embedding):
      # Constant LOG2_E: comes from using special2.exp2 instead of lax.exp.
      # Constant lax.rsqrt(h.embed): comes from t5x definition.
      unembedding_constants = special2.LOG2_E * lax.rsqrt(jnp.float32(h.embed))
      # See unquantized class for an explanation of why we do this
      return o_wo_scale * unembedding_constants, embedding * unembedding_constants

    @jax.jit
    def transpose_q_wi(q_wi, q_wi_scale):
      # Change layout:
      # (layers, embed, heads, query) -> (layers, heads, embed, query)
      # to avoid XLA doing that same transformation on every inference.
      q_wi = jnp.swapaxes(q_wi, 1, 2)
      q_wi_scale = jnp.swapaxes(q_wi_scale, 1, 2)
      return q_wi, q_wi_scale

    # TODO(sholto): Why is a shard of an array not being donated?
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def preprocess(q_wi_scale, o_wo_scale, kv_scale, embedding):
      # They are used as reciprocals later, this slightly improves
      # efficiency at forward pass time
      q_wi_scale, o_wo_scale, kv_scale = 1.0 / q_wi_scale, 1.0 / o_wo_scale, 1.0 / kv_scale
      # With 62B, if we upcast q_wi and kv to float32, then we
      # run out of memory, so we can't fold the layernorm in
      # This is where we would have called fold_in_layernorm(q_wi, kv)
      q_wi_scale = fold_in_q_constants(q_wi_scale)
      q_wi_scale = fold_in_wi0_constants(q_wi_scale)

      o_wo_scale, embedding = fold_in_unembedding_constants(
          o_wo_scale, jnp.float32(embedding))

      # embedding is returned as bfloat16, so do not donate
      return q_wi_scale, o_wo_scale, kv_scale, jnp.bfloat16(embedding)

    copy_to_device = partial(copy_to_device_with_mesh, mesh)

    expected_shapes = QuantizedWeights.make_shaped_arrays(h)

    sin, cos = _generate_fixed_pos_embedding(h.qkv, h.max_len)
    sin = copy_to_device(sin, axes.sin, expected_shapes.sin)
    cos = copy_to_device(cos, axes.cos, expected_shapes.cos)

    q_wi_input_axes = P('layers', 'embed', 'heads', 'query')
    q_wi_scale_input_axes = P('layers', None, 'heads', 'query')
    q_wi = copy_to_device(
        c.q_wi, q_wi_input_axes,
        jax.ShapedArray((h.layers, h.embed, h.heads, h.q_wi_per_head),
                        jnp.int8))
    q_wi_scale = copy_to_device(
        c.q_wi_scale, q_wi_scale_input_axes,
        jax.ShapedArray((h.layers, 1, h.heads, h.q_wi_per_head), jnp.float32))
    kv = copy_to_device(c.kv, axes.layer.kv, expected_shapes.layer.kv)
    kv_scale = copy_to_device(c.kv_scale, axes.layer.kv_scale,
                              expected_shapes.layer.kv_scale)
    o_wo = copy_to_device(c.o_wo, axes.layer.o_wo, expected_shapes.layer.o_wo)
    o_wo_scale = copy_to_device(c.o_wo_scale, axes.layer.o_wo_scale,
                                expected_shapes.layer.o_wo_scale)
    layernorm_scale_axes = P('layers', 'embed')
    layernorm_scale = copy_to_device(c.layernorm_scale, layernorm_scale_axes,
                                     expected_shapes.layer.layernorm_scale)
    embedding = copy_to_device(c.embedding, axes.embedding,
                               expected_shapes.embedding)

    with mesh:
      # We do each step of pre-processing in separate pjit calls to save memory
      q_wi_scale, o_wo_scale, kv_scale, embedding = preprocess(
          q_wi_scale, o_wo_scale, kv_scale, embedding)  # pylint: disable=line-too-long
      # on this call we do not donate argnums as the input/output buffers are
      # different sizes
      q_wi, q_wi_scale = transpose_q_wi(q_wi, q_wi_scale)

    return QuantizedWeights(
        QuantizedLayer(q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale,
                       layernorm_scale),
        sin=sin,
        cos=cos,
        embedding=embedding)


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


def quantized_transformer_layer(
    hparams, layer, params, sin,
    cos, kv_caches,
    x):
  """Forward pass through a single layer, returning output, K, V."""

  def my_layer(t, axis=0):
    """Gets the parameters corresponding to a given layer."""
    return lax.dynamic_index_in_dim(t, layer, axis=axis, keepdims=False)

  # Compare

  # prefix_batch = sin.shape[0]
  batch, max_len, _ = x.shape

  if batch == 1 and max_len == 1:
    raise ValueError('sharded batch-1 matmul is broken on VLC, b/246436629')

  x = _with_sharding_constraint(x, ('batch', 'time', 'embed'))
  # When quantized, we do not fold in layernorm scale to the weights
  xnorm = _scaled_layernorm(x, my_layer(params.layer.layernorm_scale))
  xnorm = _with_sharding_constraint(xnorm, ('batch', 'time', 'embed'))

  q_wi = quantized_dot_general('bte,hed->bthd', xnorm,
                               my_layer(params.layer.q_wi),
                               my_layer(params.layer.q_wi_scale))
  q_wi = _with_sharding_constraint(q_wi, ('batch', 'time', 'heads', None))
  q = q_wi[:, :, :, :hparams.qkv]
  q = _rope(sin, cos, q)
  wi0 = q_wi[:, :, :, hparams.qkv:hparams.qkv + (hparams.ff // hparams.heads)]
  wi1 = q_wi[:, :, :, hparams.qkv + (hparams.ff // hparams.heads):]

  kv = quantized_dot_general('bte,ezd->btzd', xnorm, my_layer(params.layer.kv),
                             my_layer(params.layer.kv_scale))
  k = kv[:, :, 0, :hparams.qkv]
  v = kv[:, :, 0, hparams.qkv:]
  k = _rope(sin, cos, k)

  y_att = attention.attend(q, k, v, kv_caches, layer)

  y_mlp = special2.swish2(wi0) * wi1
  y_mlp = _with_sharding_constraint(y_mlp, ('batch', 'time', 'heads', None))

  y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
  y_out = quantized_dot_general('bthd,hde->bte', y_fused,
                                my_layer(params.layer.o_wo),
                                my_layer(params.layer.o_wo_scale))
  y_out = _with_sharding_constraint(y_out, ('batch', 'time', 'embed'))
  z = y_out + x
  z = _with_sharding_constraint(z, ('batch', 'time', 'embed'))

  return jnp.bfloat16(z), k, v


################################################################################
################################################################################
################################################################################


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def infer(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk):
  """Forward pass through model, returning per-token logits."""

  # flaxformer/architectures/t5/t5_architecture.py;l=1516;

  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  prefix_batch, = start_indices.shape
  batch, max_length = chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'

  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    # print(f'table: {table.shape}')
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  token_ids = _with_sharding_constraint(chunk.tokens, (None, None))
  x = params.embedding[token_ids, :]
  x = _with_sharding_constraint(x, (None, None, 'table_embed'))
  x = _with_sharding_constraint(x, ('batch', 'time', 'embed'))

  def loop_body(layer, carry):
    x, k, v = carry

    x, layer_k, layer_v = _transformer_layer_fn(h, layer, params, sin, cos,
                                                kv_caches, x)
    k = lax.dynamic_update_index_in_dim(k, jnp.swapaxes(layer_k, 0, 1), layer,
                                        0)
    v = lax.dynamic_update_index_in_dim(v, jnp.swapaxes(layer_v, 0, 1), layer,
                                        0)

    return x, k, v

  # Initialize output KV cache.
  k = jnp.zeros((h.layers, max_length, batch, h.qkv), jnp.bfloat16)
  k = _with_sharding_constraint(k, ('layers', 'time', 'batch', None))
  v = jnp.zeros((h.layers, max_length, batch, h.qkv), jnp.bfloat16)
  v = _with_sharding_constraint(v, ('layers', 'time', 'batch', None))

  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))

  k = jnp.swapaxes(k, 0, 1)
  v = jnp.swapaxes(v, 0, 1)

  x = _layernorm(x)

  x = _with_sharding_constraint(x, (None, None, None))
  x = _with_sharding_constraint(x, (None, None, 'table_embed'))
  logits = jnp.einsum('bte,ve->btv', jnp.float32(x),
                      jnp.float32(params.embedding))
  logits = _with_sharding_constraint(logits, (None, None, 'table_vocab'))

  return FullChunkResult(
      logits=logits, kv_cache=attention.KVCache(chunk.lengths, k, v))


_ALLOW_UNEVEN_SHARDING = True


def _with_sharding_constraint(t,
                              spec):
  """Applies a logical sharding constraint to a tensor."""
  axes = partitioning.logical_to_physical(spec)

  # First check that the sharding is equally sized on all chips. While the SPMD
  # partitioner is _designed_ to support unequal sharding on chips, in practice
  # this seems to be a fertile ground for XLA bugs such as b/245966065 and
  # possibly the underlying bug for cr/455700040. So we just ban it, and push
  # the padding complexity on the caller.
  mesh = pxla.thread_resources.env.physical_mesh
  name_to_size = dict(zip(mesh.axis_names, mesh.devices.shape))
  for size, axis in zip(t.shape, axes):
    if axis is None or axis not in name_to_size:
      continue
    axis_size = name_to_size[axis]
    assert size % axis_size == 0 or _ALLOW_UNEVEN_SHARDING, f'Uneven sharding. Shape: {t.shape}, spec: {spec}, axis: {axis}, axis size: {axis_size}'
  return pjit.with_sharding_constraint(t, axes)
