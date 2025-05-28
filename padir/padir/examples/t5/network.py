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

"""T5.1.1 Transformer model.

A fork of
https://github.com/google-research/t5x/blob/main/t5x/examples/t5/network.py
modified for non-autoregressive decoding.
"""

from typing import Any, Mapping, Optional, Sequence, Union

import flax
from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp

from padir.padir.examples.t5 import layers
from padir.padir.utils import padir_utils


param_with_axes = nn_partitioning.param_with_axes


@struct.dataclass
class PadirModelConfig:
  """Similar to T5Config but with additional parameters for PaDIR."""

  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  # Whether to accumulate attention logits in float32 regardless of dtype.
  float32_attention_logits: bool = False
  positional_attention: bool = False

  length_predictor_cls: str = 'TokenLengthPredictor'
  top_k_lengths: int = 1
  max_target_len: int = 128
  length_id: Optional[int] = None
  # Maximum fertility per source token (FertilityLengthPredictor only).
  max_fertility: int = 2

  rejecter_cls: str = 'PassthroughRejecter'
  bottom_k_to_mask: Optional[int] = None
  bottom_percent_to_mask: Optional[float] = None

  freeze_encoder: bool = False
  freeze_decoder: bool = False
  freeze_length_predictor: bool = False
  freeze_rejecter: bool = False


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: PadirModelConfig
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Relative position embedding as attention biases.
    encoder_bias = self.relative_embedding(inputs.shape[-2], inputs.shape[-2],
                                           True)

    # Attention block.
    assert inputs.ndim == 3
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='attention')(
            x, x, x, encoder_mask, encoder_bias, deterministic=deterministic)
    x = x.out
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    return y


@flax.struct.dataclass
class DecoderLayerOutput:
  """Holds decoder layer outputs.

  Attributes:
    out: attention layer output.
    self_attentions: self attention scores.
    cross_attentions: cross attention scores.
  """
  out: jnp.ndarray
  self_attentions: jnp.ndarray
  cross_attentions: jnp.ndarray


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: PadirModelConfig
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None,
               pos_vecs=None,
               output_self_attentions=False,
               output_cross_attentions=False,
  ):  # pylint: disable=bad-continuation
    cfg = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relative_embedding(l, l, True)

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='self_attention')(
            x,
            x,
            x,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode,
            output_attentions=output_self_attentions,
        )
    self_attentions = jnp.zeros(0, dtype=jnp.float32)
    if output_self_attentions:
      self_attentions = x.attn_weights
    x = x.out
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # optional positional attention
    if pos_vecs is not None:
      pos_vecs = layers.LayerNorm(
          dtype=cfg.dtype, name='pre_positional_attention_layer_norm')(pos_vecs)
      z = layers.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          head_dim=cfg.head_dim,
          dropout_rate=cfg.dropout_rate,
          float32_logits=cfg.float32_attention_logits,
          name='positional_attention')(
              pos_vecs,
              pos_vecs,
              x,
              decoder_mask,
              decoder_bias,
              deterministic=deterministic,
              decode=decode)
      z = z.out
      x = x + z

    # Encoder-Decoder block.
    y = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='encoder_decoder_attention')(
            y, encoded, encoded, encoder_decoder_mask,
            deterministic=deterministic,
            output_attentions=output_cross_attentions,
        )
    cross_attentions = jnp.zeros(0, dtype=jnp.float32)
    if output_cross_attentions:
      cross_attentions = y.attn_weights
    y = y.out
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    return DecoderLayerOutput(
        out=z,
        self_attentions=self_attentions,
        cross_attentions=cross_attentions,
    )


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: PadirModelConfig
  shared_embedding: nn.Module

  @nn.compact
  def __call__(
      self,
      encoder_input_tokens,
      encoder_mask=None,
      deterministic=False,
  ):
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # [batch, length]
    rel_emb = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    x = self.shared_embedding(encoder_input_tokens.astype('int32'))
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
          config=cfg, relative_embedding=rel_emb,
          name=f'layers_{lyr}')(x, encoder_mask, deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: PadirModelConfig
  shared_embedding: nn.Module

  def setup(self):
    cfg = self.config
    self.pos_embed = None
    if cfg.positional_attention:
      self.pos_embed = layers.Embed(
          num_embeddings=256,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          one_hot=False,
          name='abs_pos_embed')

  @nn.compact
  def __call__(
      self,
      encoded,
      decoder_input_tokens,
      decoder_mask = None,
      encoder_decoder_mask = None,
      deterministic = False,
      decode = False,
      max_decode_length = None,
      output_self_attentions = False,
      output_cross_attentions = False,
  ):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    rel_emb = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))

    pos_vecs = None
    if self.pos_embed is not None:
      batch, length = decoder_input_tokens.shape
      iota = jnp.reshape(jnp.arange(length), [1, -1])
      positions = jnp.tile(iota, [batch, 1])
      pos_vecs = self.pos_embed(positions)

    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    self_attentions = []
    cross_attentions = []
    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = DecoderLayer(
          config=cfg, relative_embedding=rel_emb, name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length,
              pos_vecs=pos_vecs,
              output_self_attentions=output_self_attentions,
              output_cross_attentions=output_cross_attentions,
          )
      if output_self_attentions:
        self_attentions.append(y.self_attentions)
      if output_cross_attentions:
        cross_attentions.append(y.cross_attentions)
      y = y.out

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = layers.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense',
      )(y)

    out = {'embeddings': y, 'logits': logits}
    if output_self_attentions:
      self_attentions = jnp.asarray(self_attentions, dtype=jnp.float32)
      # (decoder_num_layers, batch_size, ...)
      # -> (batch_size, decoder_num_layers, ...)
      self_attentions = jnp.swapaxes(self_attentions, 0, 1)
    else:
      self_attentions = jnp.zeros(0, dtype=jnp.float32)
    if output_cross_attentions:
      cross_attentions = jnp.asarray(cross_attentions, dtype=jnp.float32)
      # (decoder_num_layers, batch_size, ...)
      # -> (batch_size, decoder_num_layers, ...)
      cross_attentions = jnp.swapaxes(cross_attentions, 0, 1)
    else:
      cross_attentions = jnp.zeros(0, dtype=jnp.float32)
    out['self_attentions'] = self_attentions
    out['cross_attentions'] = cross_attentions
    return out


class LengthOracle(nn.Module):
  """Oracle length predictor."""

  config: PadirModelConfig

  @nn.compact
  def __call__(
      self, decoder_target_tokens, **kwargs
  ):
    assert decoder_target_tokens.ndim == 2  # [batch, decoder length]
    assert (
        self.config.top_k_lengths == 1
    ), 'top_k_lengths > 1 not supported by oracle length predictor.'

    num_target_tokens = padir_utils.token_lengths(decoder_target_tokens)
    return num_target_tokens, None


class AffineLengthPredictor(nn.Module):
  """Affine length predictor."""

  config: PadirModelConfig

  def setup(self):
    self.len_bias = param_with_axes(
        'len_bias',
        nn.initializers.zeros_init(),
        (1,),
        jnp.float32,
        axes=('embed',),
    )
    self.len_factor = param_with_axes(
        'len_factor',
        nn.initializers.ones_init(),
        (1,),
        jnp.float32,
        axes=('embed',),
    )

  @nn.compact
  def __call__(
      self, encoder_input_tokens, **kwargs
  ):
    assert encoder_input_tokens.ndim == 2  # [batch, encoder length]
    assert (
        self.config.top_k_lengths == 1
    ), 'top_k_lengths > 1 not supported by affine length predictor.'

    num_input_tokens = padir_utils.token_lengths(encoder_input_tokens)
    float_predictions = self.len_factor * num_input_tokens + self.len_bias

    int_predictions = jnp.rint(float_predictions).astype(jnp.int32)
    int_predictions = jnp.clip(int_predictions, 2, self.config.max_target_len)
    return int_predictions, float_predictions


class FertilityLengthPredictor(nn.Module):
  """Fertility model layer."""

  config: PadirModelConfig

  def setup(self):
    cfg = self.config
    intermediate_dim = 4 * cfg.emb_dim
    self.mlp = layers.DenseGeneral(
        intermediate_dim, kernel_axes=('embed', 'mlp'), name='mlp'
    )
    self.mlp_offset = param_with_axes(
        'mlp_offset',
        nn.initializers.zeros_init(),
        [intermediate_dim],
        jnp.float32,
        axes=('embed',),
    )
    self.len_factor = param_with_axes(
        'len_factor',
        nn.initializers.ones_init(),
        [1],
        jnp.float32,
        axes=('embed',),
    )
    self.offset = param_with_axes(
        'offset', nn.initializers.ones_init(), [1], jnp.float32, axes=('embed',)
    )
    self.vocab_projection = layers.DenseGeneral(
        2 * self.config.max_fertility + 2,
        kernel_axes=('embed', 'vocab'),
        name='logits_dense',
    )

  @nn.compact
  def __call__(
      self,
      encoded,
      encoder_input_tokens,
      **kwargs,
  ):
    assert (
        self.config.top_k_lengths == 1
    ), 'top_k_lengths > 1 not supported by fertility length predictor.'
    assert encoded.ndim == 3  # [batch, encoder len, embedding dim]
    assert encoder_input_tokens.ndim == 2  # [batch, encoder len]

    s_mask = (encoder_input_tokens > 0).astype(jnp.float32)
    x = nn.gelu(self.mlp(jax.lax.stop_gradient(encoded)) + self.mlp_offset)
    logits = self.vocab_projection(x)
    probs = nn.softmax(logits)
    # range must have even number of elements, due to sharding.
    k = self.config.max_fertility
    iota = jnp.arange(-k - 1, k + 1)[jnp.newaxis, jnp.newaxis, :]
    relative = jnp.sum(probs * iota, axis=-1) * s_mask
    absolute = s_mask + relative
    fertil_sum = jnp.sum(absolute, axis=-1, keepdims=True)
    float_predictions = self.len_factor * fertil_sum + self.offset

    int_predictions = jnp.rint(float_predictions).astype(jnp.int32)
    int_predictions = jnp.clip(int_predictions, 2, self.config.max_target_len)
    return int_predictions, float_predictions


class TokenLengthPredictor(nn.Module):
  """Length predictor from encoded [LENGTH] token."""

  config: PadirModelConfig

  def setup(self):
    cfg = self.config
    self.length_embed = layers.Embed(
        num_embeddings=cfg.max_target_len,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=False,
        name='length_embed',
    )

  @nn.compact
  def __call__(
      self, length_embeddings, **kwargs
  ):
    assert length_embeddings is not None
    assert length_embeddings.ndim == 2  # [B, E]
    # [B, E] * [L, E].T -> [B, L]
    logits = jnp.matmul(
        length_embeddings,
        self.length_embed.embedding.transpose(),
    )

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    _, pred_idx = jax.lax.top_k(log_probs, k=self.config.top_k_lengths)
    pred_len = pred_idx + 1
    # [B, top_k_lengths], [B, L]
    return pred_len, logits


class DecoderEmbeddingsRejecter(nn.Module):
  """Rejecter trained from binary projection of decoder output embeddings."""

  config: PadirModelConfig

  def setup(self):
    self.binary_projection = layers.DenseGeneral(
        2,
        kernel_axes=('embed', 'vocab'),
        name='binary_projection',
    )

  @nn.compact
  def __call__(
      self,
      decoder_output_embeddings,
      **kwargs,
  ):
    decoder_output_embeddings = jax.lax.stop_gradient(decoder_output_embeddings)
    binary_logits = self.binary_projection(decoder_output_embeddings)
    verified_mask = jnp.argmax(binary_logits, axis=-1)
    return verified_mask, binary_logits


class LinearScheduleRejecter(nn.Module):
  """Remasks low confidence tokens according to a linear schedule."""

  config: PadirModelConfig

  def _mask_ratio(
      self,
      decode_iteration_idx,
      num_decode_iterations,
  ):
    return (
        num_decode_iterations - decode_iteration_idx - 1
    ) / num_decode_iterations

  @nn.compact
  def __call__(
      self,
      decoder_logits,
      decoder_output_tokens,
      decode_iteration_idx,
      num_decode_iterations,
      **kwargs,
  ):
    mask_ratio = self._mask_ratio(decode_iteration_idx, num_decode_iterations)
    token_lengths = padir_utils.token_lengths(decoder_output_tokens)[:, 0]
    num_to_mask = (mask_ratio * token_lengths).astype(jnp.int32)

    probs = jax.nn.softmax(decoder_logits, axis=-1)
    top_probs, _ = jax.lax.top_k(probs, k=1)
    unpadded_mask = (decoder_output_tokens > 0).astype(jnp.int32)
    rejected_mask = padir_utils.lowest_scores_mask(
        top_probs[Ellipsis, 0], unpadded_mask, num_to_mask
    )

    approved_mask = jnp.logical_not(rejected_mask) * unpadded_mask
    return approved_mask, None


class ConstantScheduleRejecter(nn.Module):
  """Remasks tokens with the lowest confidence."""

  config: PadirModelConfig

  def setup(self):
    cfg = self.config
    assert bool(cfg.bottom_k_to_mask) ^ bool(cfg.bottom_percent_to_mask), (
        'ConstantScheduleRejecter: exactly one of bottom_k_to_mask or'
        ' bottom_percent_to_mask must be set.'
    )

  @nn.compact
  def __call__(
      self,
      decoder_logits,
      decoder_output_tokens,
      **kwargs,
  ):
    num_to_mask = self.config.bottom_k_to_mask
    if not num_to_mask:
      token_lengths = padir_utils.token_lengths(decoder_output_tokens)[:, 0]
      bottom_ratios = self.config.bottom_percent_to_mask / 100.0
      num_to_mask = (token_lengths * bottom_ratios).astype(jnp.int32)

    probs = jax.nn.softmax(decoder_logits, axis=-1)
    top_probs, _ = jax.lax.top_k(probs, k=1)
    unpadded_mask = (decoder_output_tokens > 0).astype(jnp.int32)
    rejected_mask = padir_utils.lowest_scores_mask(
        top_probs[Ellipsis, 0], unpadded_mask, num_to_mask
    )

    approved_mask = jnp.logical_not(rejected_mask) * unpadded_mask
    return approved_mask, None


class StutterRejecter(nn.Module):
  """Remask token stutter.

  Note that we found remasking stutter even after the last decoding iteration
  to be beneficial. To do so, use the remask_stutter options in models.py.
  """

  config: PadirModelConfig

  @nn.compact
  def __call__(
      self,
      decoder_output_tokens,
      **kwargs,
  ):
    rejected_mask = padir_utils.stutter_mask(decoder_output_tokens)
    unpadded_mask = (decoder_output_tokens > 0).astype(jnp.int32)
    approved_mask = jnp.logical_not(rejected_mask) * unpadded_mask
    return approved_mask, None


class PassthroughRejecter(nn.Module):
  """No remasking."""

  config: PadirModelConfig

  @nn.compact
  def __call__(
      self,
      decoder_output_tokens,
      **kwargs,
  ):
    return (decoder_output_tokens > 0).astype(jnp.int32), None


LENGTH_PREDICTORS = [
    AffineLengthPredictor,
    FertilityLengthPredictor,
    LengthOracle,
    TokenLengthPredictor,
]


REJECTERS = [
    ConstantScheduleRejecter,
    DecoderEmbeddingsRejecter,
    LinearScheduleRejecter,
    StutterRejecter,
    PassthroughRejecter,
]


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: PadirModelConfig

  def setup(self):
    cfg = self.config
    self.shared_embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')
    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

    length_predictors = {lp.__name__: lp for lp in LENGTH_PREDICTORS}
    assert cfg.length_predictor_cls in length_predictors, (
        f'Unsupported length predictor, {cfg.length_predictor_cls} not in'
        f' {length_predictors.keys()}.'
    )
    self.length_predictor = length_predictors[cfg.length_predictor_cls](
        config=cfg
    )

    rejecters = {rejecter.__name__: rejecter for rejecter in REJECTERS}
    assert (
        cfg.rejecter_cls in rejecters
    ), f'Unsupported rejecter, {cfg.rejecter_cls} not in {rejecters.keys()}.'
    self.rejecter = rejecters[cfg.rejecter_cls](config=cfg)

  def encode(
      self,
      encoder_input_tokens,
      enable_dropout = True,
  ):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert encoder_input_tokens.ndim == 2, (
        f'Expected `encoder_input_tokens` to be of shape (batch, len). '
        f'Got {encoder_input_tokens.shape}')

    # Ideally we'd have unified logic to handle BOS, LENGTH, etc.,
    # outside of Transformer across training and inference.
    embed_length = self.config.length_predictor_cls == 'TokenLengthPredictor'
    if embed_length:
      bs = encoder_input_tokens.shape[0]
      len_tokens = jnp.full([bs, 1], self.config.length_id, dtype=jnp.int32)
      # NB: we don't trim encoder_input_tokens to keep EOS, so
      # encoder_input_tokens is now of length max_target_len+1.
      encoder_input_tokens = jnp.concatenate(
          [len_tokens, encoder_input_tokens], axis=1
      )

    # Make padding attention mask.
    encoder_mask = layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype
    )

    encoded = self.encoder(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout
    )

    if embed_length:
      return {
          'length_embeddings': encoded[:, 0, :],
          'encoded': encoded[:, 1:, :],
      }
    return {'encoded': encoded}

  def predict_length(
      self,
      encoded,
      length_embeddings,
      encoder_input_tokens,
      decoder_target_tokens,
      enable_dropout,
  ):
    """Returns length predictions."""

    params = {}
    params['encoded'] = encoded
    params['length_embeddings'] = length_embeddings
    params['encoder_input_tokens'] = encoder_input_tokens
    params['decoder_target_tokens'] = decoder_target_tokens
    # Should support dropout.
    params['deterministic'] = not enable_dropout

    pred_length, raw_length = self.length_predictor(**params)
    assert pred_length.ndim == 2, pred_length.shape
    return pred_length, raw_length

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      enable_dropout = True,
      decode = False,
      max_decode_length = None,
      output_self_attentions = False,
      output_cross_attentions = False,
  ):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    batch_size = encoded.shape[0]
    num_decodes = decoder_input_tokens.shape[0] // batch_size

    # Decoder inputs are already expanded, but not encoder inputs.
    if num_decodes > 1:
      encoder_input_tokens = padir_utils.flat_batch_beam_expand(
          encoder_input_tokens, num_decodes
      )
      encoded = padir_utils.flat_batch_beam_expand(encoded, num_decodes)

    # Make padding attention masks.
    encoder_decoder_mask = layers.make_attention_mask(
        jnp.ones_like(decoder_target_tokens),
        encoder_input_tokens > 0,
        dtype=cfg.dtype,
    )

    onez = jnp.ones_like(decoder_target_tokens, dtype=jnp.int32)
    mask = (decoder_input_tokens > 0).astype(jnp.int32)
    # Attend to the first position, even if the token is 0.
    mask = jnp.concatenate([onez[:, :1], mask[:, 1:]], axis=1)
    decoder_mask = layers.make_attention_mask(mask, mask, dtype=cfg.dtype)

    decoded_dict = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        output_self_attentions=output_self_attentions,
        output_cross_attentions=output_cross_attentions,
    )
    return decoded_dict

  def reject(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks # pylint: disable=unused-argument
      decoder_output_embeddings,
      decoder_logits,
      decoder_self_attentions,
      decoder_output_tokens,
      decode_iteration_idx,
      num_decode_iterations,
      enable_dropout = True,
      decode = False,
      max_decode_length = None,
  ):
    """Applies rejecter on decoded outputs."""
    params = {}
    params['encoded'] = encoded
    params['decoder_output_embeddings'] = decoder_output_embeddings
    params['decoder_logits'] = decoder_logits
    params['decoder_self_attentions'] = decoder_self_attentions
    params['decoder_output_tokens'] = decoder_output_tokens
    params['decode_iteration_idx'] = decode_iteration_idx
    params['num_decode_iterations'] = num_decode_iterations
    params['deterministic'] = not enable_dropout
    params['decode'] = decode
    params['max_decode_length'] = max_decode_length

    approved_mask, rejecter_binary_logits = self.rejecter(**params)
    assert approved_mask.ndim == 2  # [B, L]
    return {
        'approved_mask': approved_mask,
        'rejecter_binary_logits': rejecter_binary_logits,
        'decoder_self_attentions': decoder_self_attentions,
    }

  def __call__(
      self,
      encoder_input_tokens,
      decoder_input_tokens,
      decoder_target_tokens,
      decode_iteration_idx,
      num_decode_iterations,
      *,
      enable_dropout = True,
      decode = False,
      output_self_attentions = False,
      output_cross_attentions = False,
  ):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      decode_iteration_idx: index of current decoding iteration.
      num_decode_iterations: total number of decoding iterations.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      output_self_attentions: if True then returns self attention weights too.
      output_cross_attentions: if True then returns cross attention weights too.

    Returns:
      A dict with:
      - decoder logits
      - rejecter predictions
      - rejecter logits (None if a heuristic is used, e.g., remask stutter)
      - raw length predictions (None if not learnt, e.g., length oracle)
    """
    encoded_dict = self.encode(
        encoder_input_tokens,
        enable_dropout=enable_dropout,
    )
    encoded = encoded_dict['encoded']
    length_embeddings = encoded_dict.get('length_embeddings', None)
    if self.config.freeze_encoder:
      encoded = jax.lax.stop_gradient(encoded)

    # Simply train the length predictor; the decoder gets the oracle length.
    _, raw_len_predictions = self.predict_length(
        encoded=encoded,
        length_embeddings=length_embeddings,
        encoder_input_tokens=encoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=enable_dropout,
    )
    if self.config.freeze_length_predictor:
      raw_len_predictions = jax.lax.stop_gradient(raw_len_predictions)

    decoded_dict = self.decode(
        encoded,
        encoder_input_tokens,  # only used for masks
        decoder_input_tokens,
        decoder_target_tokens,
        enable_dropout=enable_dropout,
        decode=decode,
        output_self_attentions=output_self_attentions,
        output_cross_attentions=output_cross_attentions,
    )
    if self.config.freeze_decoder:
      decoded_dict = {
          k: jax.lax.stop_gradient(v) for k, v in decoded_dict.items()
      }
    decoder_output_tokens = jnp.argmax(decoded_dict['logits'], axis=-1).astype(
        jnp.int32
    )

    rejecter_outputs = self.reject(
        encoded=encoded,
        encoder_input_tokens=encoder_input_tokens,
        decoder_output_embeddings=decoded_dict['embeddings'],
        decoder_logits=decoded_dict['logits'],
        decoder_self_attentions=decoded_dict['self_attentions'],
        decoder_output_tokens=decoder_output_tokens,
        decode_iteration_idx=decode_iteration_idx,
        num_decode_iterations=num_decode_iterations,
        enable_dropout=enable_dropout,
        decode=decode,
    )
    rejecter_binary_logits = rejecter_outputs['rejecter_binary_logits']
    if self.config.freeze_rejecter:
      rejecter_binary_logits = jax.lax.stop_gradient(rejecter_binary_logits)

    return {
        'raw_length_predictions': raw_len_predictions,
        'decoder_logits': decoded_dict['logits'],
        'rejecter_logits': rejecter_binary_logits,
        'rejecter_predictions': rejecter_outputs['approved_mask'],
        'decoder_self_attentions': rejecter_outputs['decoder_self_attentions'],
        'cross_attentions': decoded_dict['cross_attentions'],
    }
