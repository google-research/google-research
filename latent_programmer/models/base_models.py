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

# coding=utf-8
"""Baseline language models for program synthesis using jax/flax.

Adapts transformer code from: flax/examples
"""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

import collections
from typing import Optional, Any, Callable

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np

from latent_programmer.models import attention

Array = Any


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  shift: bool = True  # Whether to shift input or not (for fast decoding)
  dtype: Any = jnp.float32
  emb_dim: int = 128
  num_heads: int = 4
  num_layers: int = 3
  qkv_dim: int = 128
  mlp_dim: int = 512
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  use_relative_attention: bool = False
  num_input_relative_position_buckets: int = 41
  max_input_distance: int = 20
  num_output_relative_position_buckets: int = 160
  max_output_distance: int = 200
  num_input_cross_output_relative_position_buckets: int = 160
  max_input_cross_output_distance: int = 200
  num_program_relative_position_buckets: int = 101
  max_program_distance: int = 100
  num_program_cross_embed_relative_position_buckets: int = 128
  max_program_cross_embed_distance: int = 800
  bidirectional_program_attention: bool = False
  deterministic: bool = False
  decode: bool = False
  bos_token: int = 1
  output_head: Optional[str] = 'logits'
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None


# Utility functions to handle multi-i/o examples.


def flatten_num_io_dim(x, axis=1):
  """Flattens num_io dimension of a non-scalar array into length."""
  if x.ndim == 0:
    return x
  return x.reshape(
      x.shape[:axis] + (x.shape[axis] * x.shape[axis+1],) + x.shape[axis+2:])


def unflatten_num_io_dim(x, num_io, axis=1):
  """Unflattens length dimension of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  assert x.shape[axis] % num_io == 0
  return x.reshape(
      x.shape[:axis] + (num_io, x.shape[axis] // num_io) + x.shape[axis+1:])


# Flax modules from flax/examples.


def shift_right(x, bos_token=0):
  """Shift the input to the right by padding on axis -1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[-1] = (1, 0)  # Padding on axis=-1
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(bos_token))
  return padded[Ellipsis, :-1]


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      Init function returning `[1, max_len, d_feature]`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig containing hyperparameters.
    cache: whether to use cache for fast single-position decoding.
  """

  config: TransformerConfig
  cache: bool = False

  @nn.compact
  def __call__(self,
               inputs):
    """Applies AddPositionEmbs module.

    Args:
      inputs: input data `[batch_size, ..., length, dim]`

    Returns:
      New embedding `[batch_size, ..., length, dim]`
    """
    cfg = self.config

    assert inputs.ndim >= 3
    flat_inputs = inputs.reshape((-1, inputs.shape[-2], inputs.shape[-1]))
    length = flat_inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, flat_inputs.shape[-1])
    if cfg.posemb_init is None:
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if self.cache:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    return (flat_inputs + pe).reshape(inputs.shape)


class MLPBlock(nn.Module):
  """MLP block.

  Attributes:
    config: TransformerConfig containing hyperparameters.
    out_dim: output dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies MLP block of dense layers."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class EncoderBlock(nn.Module):
  """Transformer encoder block."""

  config: TransformerConfig
  bidirectional_attention: bool = False
  num_relative_position_buckets: int = 32
  max_distance: int = 128

  @nn.compact
  def __call__(self,
               inputs,
               encoder_mask = None,
               encoder_relative_position = None):
    """Applies Transformer block.

    Args:
      inputs: input data `[batch_size, ..., length, dim]`
      encoder_mask: encoder self-attention mask
      encoder_relative_position: encoder relative positions tensor
          `[batch_sizes..., length, length]'

    Returns:
      Encoded input data `[batch_size, ..., length, mlp_dim]`
    """
    cfg = self.config

    # Attention block.
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    if cfg.use_relative_attention:
      x, aux = attention.RelativeSelfAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          bidirectional=self.bidirectional_attention,
          num_relative_position_buckets=self.num_relative_position_buckets,
          max_distance=self.max_distance)(
              x, encoder_mask, encoder_relative_position)
    else:
      x, aux = attention.SelfAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic)(x, encoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MLPBlock(config=cfg)(y)

    return x + y, aux


class EncoderDecoderBlock(nn.Module):
  """Transformer encoder-decoder block."""

  config: TransformerConfig
  bidirectional_attention: bool = False
  num_relative_position_buckets: int = 32
  max_distance: int = 128

  relative_cross_attention: bool = False
  bidirectional_cross_attention: bool = False
  num_relative_position_buckets_cross_attention: int = 32
  max_distance_cross_attention: int = 128

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask = None,
               encoder_decoder_mask = None,
               decoder_relative_position = None,
               encoder_decoder_relative_position = None):
    """Applies Transformer block.

    Args:
      targets: input data for decoder `[batch_size, ..., length, dim]`
      encoded: input data from encoder `[batch_size, ..., length2, dim2]`
      decoder_mask: decoder self-attention mask
      encoder_decoder_mask: encoder-decoder attention mask
      decoder_relative_position: decoder relative positions tensor
          `[batch_sizes..., length2, length2]'
      encoder_decoder_relative_position: encoder-decoder relative tensor
          `[batch_sizes..., length2, length]'

    Returns:
      Decoded data `[batch_size, ..., length2, mlp_dim]`
    """
    cfg = self.config

    # Decoder block.
    x = nn.LayerNorm(dtype=cfg.dtype)(targets)
    if cfg.use_relative_attention:
      x, aux = attention.RelativeSelfAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          bidirectional=self.bidirectional_attention,
          num_relative_position_buckets=self.num_relative_position_buckets,
          max_distance=self.max_distance)(
              x, decoder_mask, decoder_relative_position)
    else:
      x, aux = attention.SelfAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic)(x, decoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    if self.relative_cross_attention:
      y, aux2 = attention.RelativeMultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          bidirectional=self.bidirectional_cross_attention,
          num_relative_position_buckets=(
              self.num_relative_position_buckets_cross_attention),
          max_distance=self.max_distance_cross_attention)(
              y, encoded, encoder_decoder_mask,
              encoder_decoder_relative_position)
    else:
      y, aux2 = attention.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic)(y, encoded, encoder_decoder_mask)

    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(y)
    z = MLPBlock(config=cfg)(z)

    return (y + z,
            {'self_attn_weights': aux['attn_weights'],
             'cross_attn_weights': aux2['attn_weights']}
            )


# Transformer baseline model


class TransformerDecoder(nn.Module):
  """Transformer decoder for sequence to sequence models."""

  config: TransformerConfig

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask = None,
               encoder_decoder_mask = None,
               decoder_relative_position = None,
               encoder_decoder_relative_position = None,
               get_cross_attention_weights = False):
    """Applies Transformer to decode the targets.

    Args:
      targets: target outputs.
      encoded: encoded input data from encoder [batch, ..., length, mlp_dim].
      decoder_mask: decoder self-attention mask
      encoder_decoder_mask: encoder-decoder attention mask
      decoder_relative_position: decoder relative positions tensor
          `[batch_sizes..., length2, length2]'
      encoder_decoder_relative_position: encoder-decoder relative tensor
          `[batch_sizes..., length2, length]'
      get_cross_attention_weights: whether to get target-encoded cross-attention weights
          `[num_layers, batch_sizes..., num_heads, length2, length]'
    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    assert encoded.ndim == targets.ndim + 1

    output_embed = nn.Embed(
        num_embeddings=cfg.output_vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='embed_output')

    heads = dict()
    y = targets.astype('int32')
    if cfg.shift:
      y = shift_right(y, cfg.bos_token)

    y = output_embed(y)
    if not cfg.use_relative_attention:
      y = AddPositionEmbs(config=cfg, cache=cfg.decode,
                          name='posembed_output')(y)
    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)

    attn_weights = []
    y = y.astype(cfg.dtype)
    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y, aux = EncoderDecoderBlock( 
          config=cfg,
          bidirectional_attention=cfg.bidirectional_program_attention,
          num_relative_position_buckets=(
              cfg.num_program_relative_position_buckets),
          max_distance=cfg.max_program_distance,
          relative_cross_attention=cfg.use_relative_attention,
          bidirectional_cross_attention=True,
          num_relative_position_buckets_cross_attention=(
              cfg.num_program_cross_embed_relative_position_buckets),
          max_distance_cross_attention=cfg.max_program_cross_embed_distance,
          name=f'encoderdecoderblock_{lyr}')(
              y, encoded, decoder_mask, encoder_decoder_mask,
              decoder_relative_position, encoder_decoder_relative_position)
      attn_weights.append(aux['cross_attn_weights'])
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    heads['output_emb'] = y * (
        jnp.where(targets > 0, 1, 0).astype(jnp.float32)[Ellipsis, None])

    logits = nn.Dense(
        cfg.output_vocab_size,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        name='logitdense')(y)
    heads['logits'] = logits
    if cfg.output_head:
      if get_attention_weights:
        return heads[cfg.output_head], jnp.array(attn_weights)
      else:
        return heads[cfg.output_head]
    else:
      # Return both output embeddings and logits.
      if get_attention_weights:
        return heads, jnp.array(attn_weights)
      else:
        return heads  


# Transformer components modified to handle IO examples as input.


class TransformerIOEncoder(nn.Module):
  """Transformer encoder for i/o examples using double-attention."""

  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               outputs,
               inputs_encoder_mask=None,
               outputs_encoder_mask=None,
               io_mask=None):
    """Applies Transformer model to encode the IO specification.

    Args:
      inputs: input data [batch_size, num_io, length]
      outputs: output data [batch_size, num_io, length2]
      inputs_encoder_mask: decoder self-attention mask

    Returns:
      Encoded IO data `[batch_size, num_io, length2, dim]`
    """
    cfg = self.config

    # Inputs and outputs shared embeddings.
    embed = nn.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='embed')

    x = inputs.astype('int32')
    y = outputs.astype('int32')

    # Make attention masks.
    if inputs_encoder_mask is None:
      inputs_encoder_mask = nn.make_attention_mask(
          x > 0, x > 0, dtype=cfg.dtype)
    if outputs_encoder_mask is None:
      outputs_encoder_mask = nn.make_attention_mask(
          y > 0, y > 0, dtype=cfg.dtype)
    if io_mask is None:
      io_mask = nn.make_attention_mask(
          y > 0, x > 0, dtype=cfg.dtype)

    # Embed inputs.
    x = embed(x)
    if not cfg.use_relative_attention:
      pos_emb = AddPositionEmbs(config=cfg, cache=False, name='posembed_io')
      x = pos_emb(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)

    x = x.astype(cfg.dtype)
    for lyr in range(cfg.num_layers):
      x, _ = EncoderBlock(   # Attend to inputs.
          config=cfg,
          bidirectional_attention=True,
          num_relative_position_buckets=(
              cfg.num_input_relative_position_buckets),
          max_distance=cfg.max_input_distance,
          name=f'encoderblock_{lyr}')(x, inputs_encoder_mask)
    x = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    # Embed outputs.
    y = embed(y)
    if not cfg.use_relative_attention:
      y = pos_emb(y)
    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)

    encode_decoder_cfg = cfg.replace(decode=False)
    for lyr in range(cfg.num_layers):
      y, _ = EncoderDecoderBlock(   # Double attend to inputs and outputs.
          config=encode_decoder_cfg,
          bidirectional_attention=True,
          num_relative_position_buckets=(
              cfg.num_output_relative_position_buckets),
          max_distance=cfg.max_output_distance,
          relative_cross_attention=cfg.use_relative_attention,
          bidirectional_cross_attention=True,
          num_relative_position_buckets_cross_attention=(
              cfg.num_input_cross_output_relative_position_buckets),
          max_distance_cross_attention=cfg.max_input_cross_output_distance,
          name=f'encoderdecoderblock_{lyr}')(
              y, x, outputs_encoder_mask, io_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    return y


class ProgramTransformer(nn.Module):
  """Transformer model for program synthesis with i/o examples."""

  config: TransformerConfig

  def setup(self):
    cfg = self.config

    self.encoder = TransformerIOEncoder(config=cfg, name='encoder')
    self.decoder = TransformerDecoder(config=cfg, name='decoder')

  def encode(self,
             inputs,
             outputs):
    """Applies encoder on input specification."""
    # i/o shape = (batch_size, num_io, length)
    assert inputs.ndim == 3, ('Number of i/o dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    assert outputs.ndim == inputs.ndim

    return self.encoder(inputs, outputs)

  def decode(self,
             programs,
             encoded,
             encoded_padding_mask):
    """Applies decoder on programs and encoded specification."""
    cfg = self.config

    assert programs.ndim == 2, ('Number of program dimensions should be 2,'
                                ' but it is: %d' % programs.ndim)
    assert encoded.ndim == 4, ('Number of encoded dimensions should be 4,'
                               ' but it is: %d' % encoded.ndim)

    # Collapse num_io dimension
    flat_encoded = flatten_num_io_dim(encoded)
    flat_encoded_padding_mask = flatten_num_io_dim(encoded_padding_mask)

    # Make attention masks.
    if cfg.decode:
      # For fast decode with caching, programs shape == [batch_size, 1] and
      # cfg.shift = False, cfg.decode = True.
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(programs), flat_encoded_padding_mask, dtype=cfg.dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(programs > 0, programs > 0, dtype=cfg.dtype),
          nn.make_causal_mask(programs, dtype=cfg.dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          programs > 0, flat_encoded_padding_mask, dtype=cfg.dtype)

    return self.decoder(
        programs, flat_encoded, decoder_mask, encoder_decoder_mask)

  def __call__(self,
               inputs,
               outputs,
               programs):
    """Applies Transformer model on the inputs."""
    encoded = self.encode(inputs, outputs)
    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)

    return self.decode(programs, encoded, encoded_padding_mask)


class TransformerEncoder(nn.Module):
  """Vanilla Transformer encoder."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, dummy):
    """Vanilla Transformer encoder.

    Args:
      inputs: input data [batch_size, num_io, length]
      dummy: unused for SCAN dataset.
    Returns:
      Encoded inputs `[batch_size, num_io, length, dim]`
    """
    del dummy
    # TODO(kshi): possibly use dummy for RobustFill.

    cfg = self.config

    # Inputs and outputs shared embeddings.
    embed = nn.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='embed')

    x = inputs.astype('int32')
    encoder_mask = nn.make_attention_mask(x > 0, x > 0, dtype=cfg.dtype)

    # Embed outputs.
    x = embed(x)
    if not cfg.use_relative_attention:
      pos_emb = AddPositionEmbs(config=cfg, cache=False, name='posembed_io')
      x = pos_emb(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)

    for lyr in range(cfg.num_layers):
      x, _ = EncoderBlock(   # Attend to inputs.
          config=cfg,
          name=f'encoderblock_{lyr}')(x, encoder_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return y
