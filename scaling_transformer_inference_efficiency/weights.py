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

"""Minimalist codebase for PaLM model inference.

Relative to the t5x implementation of PaLM, this codebase does not aim for
configurability, and instead aims for peak performance inference, including in
ways that would require significant changes to how t5x's APIs are structured.

Test this with :inference_test
"""

from functools import lru_cache  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member

from flax import struct
import jax
from jax import core
from jax import lax
from jax import sharding
from jax.experimental import pjit
import jax.numpy as jnp
import jax.scipy
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import special2

HParams = checkpoint.HParams
CheckpointSpec = checkpoint.CheckpointSpec


# cache this until the cpp pathway is built
@lru_cache
def create_mesh_pspec_sharding(mesh, pspec):
  return sharding.NamedSharding(mesh, pspec)


def copy_to_device_with_mesh(mesh, x, spec, expected):
  spec = partitioning.logical_to_physical(spec)
  s = create_mesh_pspec_sharding(mesh, spec)
  return partitioning.copy_to_device(x, s, expected)


def _generate_fixed_pos_embedding(
    features, length, min_timescale=1.0, max_timescale=10000.0
):
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
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(length),
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


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
    q_wi = core.ShapedArray(
        (h.layers, h.heads - h.padded_heads, h.embed, h.q_wi_per_head),
        jnp.bfloat16,
    )
    kv = core.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.bfloat16)
    o_wo = core.ShapedArray(
        (h.layers, h.heads - h.padded_heads, h.o_wo_per_head, h.embed),
        jnp.bfloat16,
    )
    sin = core.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    cos = core.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    embedding = core.ShapedArray((h.vocab, h.embed), jnp.bfloat16)
    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)  # pytype: disable=wrong-arg-types  # jax-types

  @classmethod
  def logical_axes(cls):
    """Returns the partition specs for the weights in their logical axes."""
    q_wi = P('layers', 'params_heads', 'params_embed', 'qkv')
    kv = P('layers', 'params_kv_embed', None, 'qkv')
    o_wo = P('layers', 'params_heads', 'qkv', 'params_embed')
    sin = P(None, None)
    cos = P(None, None)
    embedding = P('params_vocab', 'embedding_embed')

    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)  # pytype: disable=wrong-arg-types  # jax-ndarray

  @classmethod
  def physical_axes(cls):
    """Returns the partition specs for the weights in their physical axes."""
    return jax.tree.map(
        partitioning.logical_to_physical, Weights.logical_axes()
    )

  @partial(jax.jit, static_argnums=(0, 1), donate_argnums=(2, 3))
  def pad_heads(self, h, o_wo, q_wi):
    """Pads heads so we can shard them further."""
    if h.padded_heads > 0:
      q_wi_padding = jnp.zeros(
          (h.layers, h.padded_heads, h.embed, h.q_wi_per_head), dtype=q_wi.dtype
      )
      q_wi = jax.lax.with_sharding_constraint(
          jnp.concatenate([q_wi, q_wi_padding], axis=1), self.physical_axes.q_wi
      )
      o_wo_padding = jnp.zeros(
          (h.layers, h.padded_heads, h.o_wo_per_head, h.embed), dtype=o_wo.dtype
      )
      o_wo = jax.lax.with_sharding_constraint(
          jnp.concatenate([o_wo, o_wo_padding], axis=1), self.physical_axes.q_wi
      )
      return o_wo, q_wi

  @classmethod
  def from_checkpoint(
      cls, h, mesh, c
  ):
    """Initializes weights in HBM from the checkpoint."""

    axes = Weights.logical_axes()

    def fold_in_wi0_constants(q_wi):
      hidden_channel_iota = jax.lax.broadcasted_iota(jnp.int32, q_wi.shape, 3)
      wi0_mask = (hidden_channel_iota >= h.qkv) & (
          hidden_channel_iota < (h.qkv + (h.ff // h.heads))
      )
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

    def fold_in_layernorm(
        q_wi, kv, layernorm_scale
    ):
      # Fold in layernorm scale to remove a multiplication
      layernorm_scale = 1.0 + layernorm_scale[:, :, np.newaxis, np.newaxis]
      return q_wi * layernorm_scale, kv * layernorm_scale

    def fold_in_unembedding_constants(
        o_wo, embedding
    ):
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
          jnp.float32(q_wi), jnp.float32(kv), layernorm_scale
      )
      q_wi = fold_in_q_constants(q_wi)
      q_wi = fold_in_wi0_constants(q_wi)
      o_wo, embedding = fold_in_unembedding_constants(
          jnp.float32(o_wo), jnp.float32(embedding)
      )

      # Change layout:
      #   (layers, embed, heads, query) -> (layers, heads, embed, query)
      # to avoid XLA doing that same transformation on every inference.
      q_wi = jnp.swapaxes(q_wi, 1, 2)

      return (
          jnp.bfloat16(q_wi),
          jnp.bfloat16(kv),
          jnp.bfloat16(o_wo),
          jnp.bfloat16(embedding),
      )

    expected_shapes = Weights.make_shaped_arrays(h)

    copy_to_device = partial(copy_to_device_with_mesh, mesh)

    sin, cos = _generate_fixed_pos_embedding(h.qkv, h.max_len)
    sin = copy_to_device(sin, axes.sin, expected_shapes.sin)
    cos = copy_to_device(cos, axes.cos, expected_shapes.cos)

    q_wi_input_axes = P(
        'layers', 'weight_load_embed', 'weight_load_heads', 'qkv'
    )
    q_wi = copy_to_device(
        c.q_wi,
        q_wi_input_axes,
        core.ShapedArray(
            (h.layers, h.embed, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.bfloat16,
        ),
    )
    kv = copy_to_device(c.kv, axes.layer.kv, expected_shapes.layer.kv)
    o_wo = copy_to_device(c.o_wo, axes.layer.o_wo, expected_shapes.layer.o_wo)
    layernorm_scale_axes = ('layers', 'params_embed')
    layernorm_scale = copy_to_device(
        c.layernorm_scale,
        layernorm_scale_axes,
        core.ShapedArray((h.layers, h.embed), jnp.float32),
    )
    embedding = copy_to_device(
        c.embedding, axes.embedding, expected_shapes.embedding
    )

    q_wi_input_axes = partitioning.logical_to_physical(q_wi_input_axes)
    q_wi_output_axes = partitioning.logical_to_physical(axes.layer.q_wi)
    kv_axes = partitioning.logical_to_physical(axes.layer.kv)
    o_wo_axes = partitioning.logical_to_physical(axes.layer.o_wo)
    layernorm_scale_axes = partitioning.logical_to_physical(
        layernorm_scale_axes
    )
    embedding_axes = partitioning.logical_to_physical(axes.embedding)

    with mesh:
      q_wi, kv, o_wo, embedding = pjit.pjit(
          preprocess,
          in_shardings=(
              q_wi_input_axes,
              kv_axes,
              o_wo_axes,
              layernorm_scale_axes,
              embedding_axes,
          ),
          out_shardings=(
              q_wi_output_axes,
              kv_axes,
              o_wo_axes,
              embedding_axes,
          ),
          donate_argnums=(1, 2, 4),
      )(q_wi, kv, o_wo, layernorm_scale, embedding)

    return Weights(Layer(q_wi, kv, o_wo), sin=sin, cos=cos, embedding=embedding)


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
    q_wi = core.ShapedArray(
        (h.layers, h.heads - h.padded_heads, h.embed, h.q_wi_per_head), jnp.int8
    )
    q_wi_scale = core.ShapedArray(
        (h.layers, h.heads - h.padded_heads, 1, h.q_wi_per_head), jnp.float32
    )
    kv = core.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.int8)
    kv_scale = core.ShapedArray((h.layers, 1, 1, 2 * h.qkv), jnp.float32)
    o_wo = core.ShapedArray(
        (h.layers, h.heads - h.padded_heads, h.o_wo_per_head, h.embed), jnp.int8
    )
    o_wo_scale = core.ShapedArray((h.layers, 1, 1, h.embed), jnp.float32)
    sin = core.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    cos = core.ShapedArray((h.max_len, h.qkv // 2), jnp.float32)
    embedding = core.ShapedArray((h.vocab, h.embed), jnp.bfloat16)
    layernorm_scale = core.ShapedArray((h.layers, h.embed), jnp.bfloat16)
    return QuantizedWeights(  # pytype: disable=wrong-arg-types  # jax-types
        QuantizedLayer(  # pytype: disable=wrong-arg-types  # jax-types
            q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale, layernorm_scale
        ),
        sin=sin,
        cos=cos,
        embedding=embedding,
    )

  @classmethod
  def logical_axes(cls):
    """Returns the partition specs for the weights in their logical axes."""
    q_wi = P('layers', 'params_heads', 'params_embed', 'qkv')
    # Scale Axes can not shard along a singleton dimension
    q_wi_scale = P('layers', 'heads', None, 'qkv')
    kv = P('layers', 'params_kv_embed', None, 'qkv')
    kv_scale = P('layers', None, None, 'qkv')
    o_wo = P('layers', 'params_heads', 'qkv', 'params_embed')
    o_wo_scale = P('layers', None, None, 'residual_embed')
    sin = P(None, None)
    cos = P(None, None)
    # Embedding table wants different sharding than Transformer layers, to work
    # around b/244232479.
    embedding = P('params_vocab', 'embedding_embed')
    layernorm_scale = P('layers', 'params_embed')

    return QuantizedWeights(  # pytype: disable=wrong-arg-types  # jax-ndarray
        QuantizedLayer(  # pytype: disable=wrong-arg-types  # jax-ndarray
            q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale, layernorm_scale
        ),
        sin=sin,
        cos=cos,
        embedding=embedding,
    )

  @classmethod
  def physical_axes(cls):
    """Returns the partition specs for the weights in their physical axes."""
    return jax.tree.map(
        partitioning.logical_to_physical, QuantizedWeights.logical_axes()
    )

  @classmethod
  def from_checkpoint(
      cls, h, mesh, c
  ):
    """Initializes weights in HBM, copying from a host-resident checkpoint."""

    axes = QuantizedWeights.logical_axes()

    def fold_in_wi0_constants(q_wi_scale):
      hidden_channel_iota = jax.lax.broadcasted_iota(
          jnp.int32, q_wi_scale.shape, 3
      )
      wi0_mask = (hidden_channel_iota >= h.qkv) & (
          hidden_channel_iota < (h.qkv + (h.ff // h.heads))
      )
      # Constant 0.5: We need to multiply wi_0 by 0.5 to correct special2.swish2
      # to be equivalent to jnp.swish. More efficient to do this once to the
      # weights than every time we call the fn.
      wi0_constants = 0.5
      return q_wi_scale * jnp.where(wi0_mask, wi0_constants, 1.0)

    def fold_in_q_constants(q_wi_scale):
      hidden_channel_iota = jax.lax.broadcasted_iota(
          jnp.int32, q_wi_scale.shape, 3
      )
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
      return (
          o_wo_scale * unembedding_constants,
          embedding * unembedding_constants,
      )

    @jax.jit
    def transpose_q_wi(q_wi, q_wi_scale):
      # Change layout:
      # (layers, embed, heads, query) -> (layers, heads, embed, query)
      # to avoid XLA doing that same transformation on every inference.
      q_wi = jnp.swapaxes(q_wi, 1, 2)
      q_wi_scale = jnp.swapaxes(q_wi_scale, 1, 2)
      q_wi_input_axes = P(
          'layers', 'weight_load_heads', 'weight_load_embed', 'qkv'
      )
      q_wi_scale_input_axes = P('layers', 'weight_load_heads', None, 'qkv')
      # pylint: disable = protected-access
      q_wi = partitioning._with_sharding_constraint(q_wi, q_wi_input_axes)
      q_wi_scale = partitioning._with_sharding_constraint(
          q_wi_scale, q_wi_scale_input_axes
      )
      return q_wi, q_wi_scale

    @partial(jax.jit, donate_argnums=(0,))
    def reshard_q_wi(q_wi):
      return jax.lax.with_sharding_constraint(
          q_wi, QuantizedWeights.physical_axes().layer.q_wi
      )

    @jax.jit
    def reshard_q_wi_scale(q_wi):
      return jax.lax.with_sharding_constraint(
          q_wi, QuantizedWeights.physical_axes().layer.q_wi_scale
      )

    @partial(jax.jit, donate_argnums=(0,))
    def reshard_o_wo(o_wo):
      return jax.lax.with_sharding_constraint(
          o_wo, QuantizedWeights.physical_axes().layer.o_wo
      )

    def pad_heads(
        h,
        q_wi,
        q_wi_scale,
        o_wo,
    ):
      """Pads heads so we can shard them further, post transpose."""
      if h.padded_heads > 0:
        q_wi = jnp.pad(
            q_wi, ((0, 0), (0, h.padded_heads), (0, 0), (0, 0))
        ).block_until_ready()
        q_wi_scale = jnp.pad(
            q_wi_scale, ((0, 0), (0, h.padded_heads), (0, 0), (0, 0))
        ).block_until_ready()
        o_wo = jnp.pad(
            o_wo, ((0, 0), (0, h.padded_heads), (0, 0), (0, 0))
        ).block_until_ready()
      return q_wi, q_wi_scale, o_wo

    # TODO(sholto): Why is a shard of an array not being donated?
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def preprocess(q_wi_scale, o_wo_scale, kv_scale, embedding):
      # They are used as reciprocals later, this slightly improves
      # efficiency at forward pass time
      q_wi_scale, o_wo_scale, kv_scale = (
          1.0 / q_wi_scale,
          1.0 / o_wo_scale,
          1.0 / kv_scale,
      )
      # With 62B, if we upcast q_wi and kv to float32, then we
      # run out of memory, so we can't fold the layernorm in
      # This is where we would have called fold_in_layernorm(q_wi, kv)
      q_wi_scale = fold_in_q_constants(q_wi_scale)
      q_wi_scale = fold_in_wi0_constants(q_wi_scale)

      o_wo_scale, embedding = fold_in_unembedding_constants(
          o_wo_scale, jnp.float32(embedding)
      )

      # embedding is returned as bfloat16, so do not donate
      return q_wi_scale, o_wo_scale, kv_scale, jnp.bfloat16(embedding)

    copy_to_device = partial(copy_to_device_with_mesh, mesh)

    expected_shapes = QuantizedWeights.make_shaped_arrays(h)

    sin, cos = _generate_fixed_pos_embedding(h.qkv, h.max_len)
    sin = copy_to_device(sin, axes.sin, expected_shapes.sin)
    cos = copy_to_device(cos, axes.cos, expected_shapes.cos)

    q_wi_input_axes = P(
        'layers', 'weight_load_embed', 'weight_load_heads', 'qkv'
    )
    q_wi_scale_input_axes = P('layers', None, 'weight_load_heads', 'qkv')
    q_wi = copy_to_device(
        c.q_wi,
        q_wi_input_axes,
        core.ShapedArray(
            (h.layers, h.embed, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.int8,
        ),
    )
    q_wi_scale = copy_to_device(
        c.q_wi_scale,
        q_wi_scale_input_axes,
        core.ShapedArray(
            (h.layers, 1, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.float32,
        ),
    )
    kv = copy_to_device(c.kv, axes.layer.kv, expected_shapes.layer.kv)
    kv_scale = copy_to_device(
        c.kv_scale, axes.layer.kv_scale, expected_shapes.layer.kv_scale
    )
    o_wo_input_axes = P(
        'layers', 'weight_load_heads', 'qkv', 'weight_load_embed'
    )
    o_wo = copy_to_device(
        c.o_wo,
        o_wo_input_axes,
        core.ShapedArray(
            (h.layers, h.heads - h.padded_heads, h.o_wo_per_head, h.embed),
            jnp.int8,
        ),
    )
    o_wo_scale = copy_to_device(
        c.o_wo_scale, axes.layer.o_wo_scale, expected_shapes.layer.o_wo_scale
    )
    layernorm_scale_axes = P('layers', 'params_embed')
    layernorm_scale = copy_to_device(
        c.layernorm_scale,
        layernorm_scale_axes,
        expected_shapes.layer.layernorm_scale,
    )
    embedding = copy_to_device(
        c.embedding, axes.embedding, expected_shapes.embedding
    )

    with mesh:
      # We do each step of pre-processing in separate pjit calls to save memory
      q_wi_scale, o_wo_scale, kv_scale, embedding = preprocess(
          q_wi_scale, o_wo_scale, kv_scale, embedding
      )  # pylint: disable=line-too-long
      # on this call we do not donate argnums as the input/output buffers are
      # different sizes
      q_wi, q_wi_scale = transpose_q_wi(q_wi, q_wi_scale)
      q_wi, q_wi_scale, o_wo = pad_heads(h, q_wi, q_wi_scale, o_wo)

      q_wi = reshard_q_wi(q_wi).block_until_ready()
      q_wi_scale = reshard_q_wi_scale(q_wi_scale).block_until_ready()
      o_wo = reshard_o_wo(o_wo).block_until_ready()

    params = QuantizedWeights(
        QuantizedLayer(
            q_wi, q_wi_scale, kv, kv_scale, o_wo, o_wo_scale, layernorm_scale
        ),
        sin=sin,
        cos=cos,
        embedding=embedding,
    )
    return params
