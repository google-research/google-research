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

"""Transformer-based machine translation model."""

import copy
import dataclasses
import typing
from typing import Any, Optional, Tuple, Type, TypeVar

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from aqt.jax import flax_attention as aqt_flax_attention
from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import quant_config
from aqt.jax import shape_utils
from aqt.jax.flax import struct as flax_struct

T = TypeVar('T')

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


def hardware_bernoulli(rng_key, p=np.float32(0.5), shape=None):
  return lax.rng_uniform(lax.tie_in(rng_key, 0.0), 1.0, shape) < p


def set_hardware_bernoulli():
  jax.random.bernoulli = hardware_bernoulli


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    max_len: maximum length of sequence during eval.
    min_timescale: minimum scale that will be applied at each position
    max_timescale: maximum scale that will be applied at each position
    decode: whether to run in decoding mode (one input fed in at a time)
  """

  max_len: int
  min_timescale: float
  max_timescale: float
  decode: bool

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Adds positional embeddings to the inputs.

    Args:
      inputs: input data
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    _, sequence_length, num_channels = inputs.shape
    num_timescales = num_channels // 2
    max_timescale = self.max_timescale
    min_timescale = self.min_timescale
    max_len = self.max_len
    log_timescale_increment = (
        np.log(max_timescale / min_timescale) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype('float32') * -log_timescale_increment)

    if inputs_positions is None:
      inputs_positions = np.expand_dims(
          np.arange(sequence_length).astype('float32'), 0)

    if self.decode:
      inputs_positions = np.expand_dims(np.arange(max_len).astype('float32'), 0)

    scaled_time = (
        jnp.expand_dims(inputs_positions.astype('float32'), 2) *
        jnp.expand_dims(np.expand_dims(inv_timescales, 0), 0))
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2)
    signal = jnp.pad(
        signal, [[0, 0], [0, 0], [0, np.mod(num_channels, 2)]],
        mode='constant',
        constant_values=inputs.dtype.type(0))

    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        i = cache_index.value
        one = jnp.array(1, jnp.int32)
        cache_index.value = i + one
        _, _, df = signal.shape
        signal = lax.dynamic_slice(signal, jnp.array((0, i, 0)), (1, 1, df))
    if self.decode:
      # just needed to set correct shape on init.
      return inputs + signal[:, :1, :]
    else:
      return inputs + signal


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  @dataclass
  class HParams:
    dense_1: aqt_flax_layers.DenseAqt.HParams
    dense_2: aqt_flax_layers.DenseAqt.HParams

  hparams: HParams
  train: bool
  mlp_dim: int
  quant_context: quant_config.QuantContext
  dtype: Any
  dropout_rate: float
  deterministic: bool
  kernel_init: Any = nn.initializers.xavier_uniform()
  bias_init: Any = nn.initializers.normal(stddev=1e-6)

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  @nn.compact
  def __call__(
      self,
      inputs,
      *,
      padding_mask,
  ):
    """Applies Transformer MlpBlock module."""

    batch_size, sequence_length, channel_size = inputs.shape
    inputs = inputs.reshape((batch_size * sequence_length, channel_size))
    shape_utils.assert_shapes_equal(padding_mask.shape,
                                    (batch_size, sequence_length, 1))
    padding_mask = padding_mask.reshape((batch_size * sequence_length, 1))
    x = aqt_flax_layers.DenseAqt(
        features=self.mlp_dim,
        dtype=self.dtype,
        paxis_name='batch',
        train=self.train,
        quant_context=self.quant_context,
        hparams=self.hparams.dense_1,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense_1')(
            inputs, padding_mask=padding_mask)

    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)

    output = aqt_flax_layers.DenseAqt(
        # We have relu before this layer, x would only contain positive values.
        features=channel_size,
        dtype=self.dtype,
        paxis_name='batch',
        train=self.train,
        quant_context=self.quant_context,
        hparams=self.hparams.dense_2,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='dense_2')(
            x, padding_mask=padding_mask)

    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=self.deterministic)
    output = output.reshape((batch_size, sequence_length, channel_size))
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    update_bounds: Bool whether to update activation bounds.
    train: Whether model is training.
    hparams: hyperparameters
    collect_acts_stats: Whether to tag activations to record statistics.
    dtype: the dtype of the computation (default: float32)
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
  """

  @dataclass
  class HParams:
    mlp_block: MlpBlock.HParams
    attention: aqt_flax_attention.MultiHeadDotProductAttentionAqt.HParams
    layer_norm: aqt_flax_layers.LayerNormAqt.HParams

  hparams: HParams
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  train: bool
  quant_context: quant_config.QuantContext
  dtype: Any
  dropout_rate: float
  attention_dropout_rate: float
  deterministic: bool

  # TODO(shivaniagrawal): parameterize paxis_name instead of hardcoding.
  @nn.compact
  def __call__(self,
               inputs,
               padding_mask,
               inputs_segmentation=None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data
      padding_mask: bool, mask padding tokens
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output after transformer block.
    """

    # Attention block.
    batch_size, sequence_length, channel_size = inputs.shape
    shape_utils.assert_shapes_equal(padding_mask.shape,
                                    (batch_size, sequence_length, 1))
    x = aqt_flax_layers.LayerNormAqt(
        dtype=self.dtype,
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            inputs)
    x = aqt_flax_attention.SelfAttentionAqt(
        hparams=self.hparams.attention,
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        paxis_name='batch',
        train=self.train,
        quant_context=self.quant_context,
        causal_mask=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        decode=False,
        name='enc_self_att')(
            x, padding_mask=padding_mask, segmentation=inputs_segmentation)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + inputs

    # MLP block.
    y = aqt_flax_layers.LayerNormAqt(
        dtype=self.dtype,
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        hparams=self.hparams.mlp_block,
        # inputs would be signed, called after attention layer.
        train=self.train,
        quant_context=self.quant_context,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        name='mlp_block')(
            y, padding_mask=padding_mask)
    out = x + y
    shape_utils.assert_shapes_equal(out.shape,
                                    (batch_size, sequence_length, channel_size))
    return out


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    hparams: hyperparams
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    update_bounds: Bool whether to update activation bounds.
    train: Whether model is training.
    collect_acts_stats: Whether to tag activations to record statistics.
    dtype: the dtype of the computation (default: float32)
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    decode: whether to run in decoding mode
  """

  @dataclass
  class HParams:
    mlp_block: MlpBlock.HParams
    self_attention: aqt_flax_attention.MultiHeadDotProductAttentionAqt.HParams
    enc_dec_attention: aqt_flax_attention.MultiHeadDotProductAttentionAqt.HParams
    layer_norm: aqt_flax_layers.LayerNormAqt.HParams

  hparams: HParams
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  train: bool
  quant_context: quant_config.QuantContext
  dtype: Any
  dropout_rate: float
  attention_dropout_rate: float
  deterministic: bool
  decode: bool

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  @nn.compact
  def __call__(
      self,
      targets,
      encoded,
      padding_mask,
      key_padding_mask,
      inputs_segmentation=None,
      targets_segmentation=None,
  ):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      output after transformer block.
    """

    # Decoder block.
    batch_size, sequence_length, num_channels = targets.shape
    encoded_sequence_length = encoded.shape[1]
    shape_utils.assert_shapes_equal(padding_mask.shape,
                                    (batch_size, sequence_length, 1))
    shape_utils.assert_shapes_equal(
        encoded.shape, (batch_size, encoded_sequence_length, num_channels))
    shape_utils.assert_shapes_equal(key_padding_mask.shape,
                                    (batch_size, encoded_sequence_length, 1))

    x = aqt_flax_layers.LayerNormAqt(
        dtype=self.dtype,
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            targets)
    x = aqt_flax_attention.SelfAttentionAqt(
        hparams=self.hparams.self_attention,
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        paxis_name='batch',
        train=self.train,
        quant_context=self.quant_context,
        causal_mask=True,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        name='dec_self_att',
        decode=self.decode)(
            x,
            padding_mask=padding_mask,
            segmentation=targets_segmentation,
        )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = aqt_flax_layers.LayerNormAqt(
        dtype=self.dtype,
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            x)
    y = aqt_flax_attention.MultiHeadDotProductAttentionAqt(
        hparams=self.hparams.enc_dec_attention,
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        paxis_name='batch',
        train=self.train,
        quant_context=self.quant_context,
        causal_mask=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        decode=False,
        name='dec_enc_att')(
            inputs_q=y,
            inputs_kv=encoded,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=targets_segmentation,
            key_segmentation=inputs_segmentation)
    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=self.deterministic)
    y = y + x

    # MLP block.
    z = aqt_flax_layers.LayerNormAqt(
        dtype=self.dtype,
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            y)
    z = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        hparams=self.hparams.mlp_block,
        # inputs would be signed, called after attention layer.
        train=self.train,
        quant_context=self.quant_context,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        name='mlp_block')(
            z, padding_mask=padding_mask)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    vocab_size: size of the vocabulary
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    shared_embedding: a shared embedding layer to use.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding
    num_heads: number of heads
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    train: if it is training,
    collect_acts_stats: Whether to tag activations to record statistics.
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
  """

  @dataclass
  class HParams:
    """Encoder hyperparameters."""

    embedding: aqt_flax_layers.EmbedAqt.HParams
    encoder_1d_blocks: Tuple[Encoder1DBlock.HParams, Ellipsis]
    layer_norm: aqt_flax_layers.LayerNormAqt.HParams

    # TODO(b/175245107): Remove this function
    @classmethod
    def create_from_block_template(
        cls, *,
        embedding,
        block_template,
        layer_norm,
        num_layers):
      return cls(
          embedding=embedding,
          layer_norm=layer_norm,
          encoder_1d_blocks=tuple(
              copy.deepcopy(block_template) for _ in range(num_layers)))

  vocab_size: int
  hparams: HParams
  max_len: int
  shared_embedding: Any
  use_bfloat16: bool
  emb_dim: int
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  train: bool
  quant_context: quant_config.QuantContext
  dropout_rate: float
  attention_dropout_rate: float

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  @nn.compact
  def __call__(self, inputs, inputs_positions=None, inputs_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer decoder.

    """
    batch_size, sequence_length = inputs.shape

    # Padding Masks
    src_padding_mask = (inputs > 0)[Ellipsis, None]
    shape_utils.assert_shapes_equal(src_padding_mask.shape,
                                    (batch_size, sequence_length, 1))

    if self.use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = aqt_flax_layers.EmbedAqt(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          hparams=self.hparams.embedding,
          embedding_init=nn.initializers.normal(stddev=self.emb_dim**-0.5),
          dtype=dtype,
          name='input_embed',
          paxis_name='batch',
          train=self.train,
          quant_context=self.quant_context)
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x) * jnp.sqrt(self.emb_dim)
    x = AddPositionEmbs(
        name='posembed_input',
        max_len=self.max_len,
        min_timescale=1.0,
        max_timescale=10000.0,
        decode=False)(
            x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)

    if self.use_bfloat16:
      x = x.astype(jnp.bfloat16)

    # Input Encoder
    num_layers = len(self.hparams.encoder_1d_blocks)
    for lyr in range(num_layers):
      x = Encoder1DBlock(
          train=self.train,
          quant_context=self.quant_context,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          hparams=self.hparams.encoder_1d_blocks[lyr],
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f'encoderblock_{lyr}')(
              x,
              padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation)
    encoded = aqt_flax_layers.LayerNormAqt(
        dtype=dtype,
        name='encoder_norm',
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            x)
    shape_utils.assert_shapes_equal(encoded.shape,
                                    (batch_size, sequence_length, self.emb_dim))
    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    output_vocab_size: size of the vocabulary.
    update_bounds: Bool whether to update activation bounds.
    hparams: hyperparameters
    max_len: maximum length.
    shared_embedding: a shared embedding layer to use.
    logits_via_embedding: bool: whether final logit transform shares embedding
      weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding
    num_heads: number of heads
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    train: if it is training.
    collect_acts_stats: Whether to tag activations to record statistics.
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    paxis_name: name of parallel reduction axis
    decode: whether to run in decoding mode.
  """

  @dataclass
  class HParams:
    """Decoder hyperparameters."""

    embedding: aqt_flax_layers.EmbedAqt.HParams
    encoder_decoder_1d_blocks: Tuple[EncoderDecoder1DBlock.HParams, Ellipsis]
    # logits field must be set if logits are not shared.
    logits: Optional[aqt_flax_layers.DenseAqt.HParams]
    layer_norm: aqt_flax_layers.LayerNormAqt.HParams

    # TODO(b/175245107): Remove this function
    @classmethod
    def create_from_block_template(
        cls,
        *,
        embedding,
        block_template,
        num_layers,
        layer_norm,
        logits = None):
      return cls(
          embedding=embedding,
          layer_norm=layer_norm,
          encoder_decoder_1d_blocks=tuple(
              copy.deepcopy(block_template) for _ in range(num_layers)),
          logits=logits)

  output_vocab_size: int
  hparams: HParams
  max_len: int
  shared_embedding: Any
  logits_via_embedding: bool
  use_bfloat16: bool
  emb_dim: int
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  train: bool
  quant_context: quant_config.QuantContext
  dropout_rate: float
  attention_dropout_rate: float
  paxis_name: Optional[str]
  decode: bool

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  @nn.compact
  def __call__(
      self,
      encoded,
      src_padding_mask,
      targets,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
      tgt_padding_mask=None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.

    Returns:
      output of a transformer decoder.

    """
    batch_size, sequence_length, channel_size = encoded.shape  # pylint: disable=unused-variable
    target_batch_size, target_sequence_length = targets.shape  # pylint: disable=unused-variable
    shape_utils.assert_shapes_equal(targets.shape,
                                    (batch_size, target_sequence_length))

    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[Ellipsis, None]
    shape_utils.assert_shapes_equal(tgt_padding_mask.shape,
                                    (batch_size, target_sequence_length, 1))

    if self.use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = aqt_flax_layers.EmbedAqt(
          num_embeddings=self.output_vocab_size,
          features=self.emb_dim,
          hparams=self.hparams.embedding,
          embedding_init=nn.initializers.normal(stddev=self.emb_dim**-0.5),
          dtype=dtype,
          name='target_embed',
          train=self.train,
          quant_context=self.quant_context,
          paxis_name='batch')
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if not self.decode:
      y = shift_right(y)
    y = output_embed(y) * jnp.sqrt(self.emb_dim)
    y = AddPositionEmbs(
        name='posembed_targets',
        max_len=self.max_len,
        decode=self.decode,
        min_timescale=1.0,
        max_timescale=10000.0)(
            y, inputs_positions=targets_positions)
    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not self.train)

    if self.use_bfloat16:
      y = y.astype(jnp.bfloat16)

    # Target-Input Decoder
    num_layers = len(self.hparams.encoder_decoder_1d_blocks)
    for lyr in range(num_layers):
      y = EncoderDecoder1DBlock(
          train=self.train,
          quant_context=self.quant_context,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          hparams=self.hparams.encoder_decoder_1d_blocks[lyr],
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f'encoderdecoderblock_{lyr}',
          decode=self.decode)(
              y,
              encoded,
              padding_mask=tgt_padding_mask,
              key_padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation,
              targets_segmentation=targets_segmentation)
    y = aqt_flax_layers.LayerNormAqt(
        dtype=dtype,
        name='encoderdecoder_norm',
        hparams=self.hparams.layer_norm,
        quant_context=self.quant_context)(
            y)
    y = y.reshape((batch_size * target_sequence_length, channel_size))
    tgt_padding_mask = tgt_padding_mask.reshape(
        (batch_size * target_sequence_length, 1))
    # Decoded Logits
    if self.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(
          query=y,
          padding_mask=tgt_padding_mask,
          paxis_name=self.paxis_name,
          train=self.train)
    else:
      if self.hparams.logits is None:
        raise ValueError('If logits_via_embedding is False, then the hparams '
                         'for the logits layer have to be provided.')
      logits = aqt_flax_layers.DenseAqt(
          features=self.output_vocab_size,
          dtype=dtype,
          paxis_name='batch',
          train=self.train,
          quant_context=self.quant_context,
          hparams=self.hparams.logits,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logits_dense')(
              y, padding_mask=tgt_padding_mask)
    return logits


# The following final model is simple but looks verbose due to all the
# repetitive keyword argument plumbing.  It just sticks the Encoder and
# Decoder in series for training, but allows running them separately for
# inference.
# TODO(malmaud): Remove this comment once the new hyperparam threading
# logic is complete.


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    update_bounds: Bool whether to update activation bounds.
    hparams: hyperparameters
    max_len: maximum length.
    vocab_size: size of the input vocabulary
    output_vocab_size: size of the output vocabulary
    use_bfloat16: bool: whether use bfloat16.
    train: if it is training,
    collect_acts_stats: Whether to tag activations to record statistics.
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    should_decode: whether to run in decoding mode
  """

  @dataclass
  class HParams:
    """Hyperparameters for the transformer.

    TODO(wanglisa): Push encoder / decoder specific params (num_heads, emb_dim
    qkv_dim, mlp_dim) down to Encoder.HParams and Decoder.HParams.
    """

    encoder: Encoder.HParams
    decoder: Decoder.HParams
    emb_dim: int  # dimension of embedding.
    num_heads: int  # Number of heads.
    qkv_dim: int  # dimension of the query/key/value.
    mlp_dim: int  # dimension of the mlp on top of attention block.
    share_embeddings: bool  # share embedding layer for inputs and targets.
    logits_via_embedding: bool  # share emb weights w/ final logit transform.

  hparams: HParams
  max_len: int
  vocab_size: Optional[int]
  output_vocab_size: Optional[int]
  use_bfloat16: bool
  train: bool
  quant_context: quant_config.QuantContext
  dropout_rate: float
  attention_dropout_rate: float
  # We call this 'should_decode' instead of 'decode' so it doesn't conflict with
  # the 'decode' method name.
  should_decode: bool

  def setup(self):

    if self.use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    if self.hparams.share_embeddings:
      if self.output_vocab_size is not None:
        assert self.output_vocab_size == self.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = aqt_flax_layers.EmbedAqt(  # pylint: disable=missing-from-attributes
          num_embeddings=self.vocab_size,
          features=self.hparams.emb_dim,
          hparams=self.hparams.encoder.embedding,
          dtype=dtype,
          embedding_init=nn.initializers.normal(
              stddev=self.hparams.emb_dim**-0.5),
          train=self.train,
          quant_context=self.quant_context,
          paxis_name='batch')
    else:
      self.shared_embedding = None

    self.encoder = Encoder(  # pylint: disable=missing-from-attributes
        hparams=self.hparams.encoder,
        vocab_size=self.vocab_size,
        shared_embedding=self.shared_embedding,
        use_bfloat16=self.use_bfloat16,
        emb_dim=self.hparams.emb_dim,
        num_heads=self.hparams.num_heads,
        qkv_dim=self.hparams.qkv_dim,
        mlp_dim=self.hparams.mlp_dim,
        max_len=self.max_len,
        train=self.train,
        quant_context=self.quant_context,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
    )

    self.decoder = Decoder(  # pylint: disable=missing-from-attributes
        hparams=self.hparams.decoder,
        output_vocab_size=self.output_vocab_size,
        shared_embedding=self.shared_embedding,
        logits_via_embedding=self.hparams.logits_via_embedding,
        use_bfloat16=self.use_bfloat16,
        emb_dim=self.hparams.emb_dim,
        num_heads=self.hparams.num_heads,
        qkv_dim=self.hparams.qkv_dim,
        mlp_dim=self.hparams.mlp_dim,
        max_len=self.max_len,
        train=self.train,
        quant_context=self.quant_context,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        paxis_name='batch',
        decode=self.should_decode)

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  def __call__(
      self,
      inputs,
      targets,
      inputs_positions=None,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      targets: target data
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      output of a transformer decoder.

    """
    batch_size, sequence_length = inputs.shape
    assert batch_size == targets.shape[
        0], 'Input and targets must have the same batch size'

    src_padding_mask = (inputs > 0)[Ellipsis, None]
    shape_utils.assert_shapes_equal(src_padding_mask.shape,
                                    (batch_size, sequence_length, 1))

    encoded = self.encode(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    logits = self.decode(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=None)
    return logits.astype(jnp.float32) if self.use_bfloat16 else logits

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while providing a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  def encode(self, inputs, inputs_positions=None, inputs_segmentation=None):

    encoded = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    return encoded

  # TODO(shivaniagrawal): parametrize paxis_name instead of hardcoding.
  def decode(
      self,
      encoded,
      src_padding_mask,
      targets,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
      tgt_padding_mask=None,
  ):
    """Run the decoder part of the transformer given encoded inputs."""
    logits = self.decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask,
    )

    return logits
