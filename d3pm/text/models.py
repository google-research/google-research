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

"""Neural network modules for diffusion text processes."""

import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence, Type

from absl import logging
import chex
from flax import struct
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import convolution
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import initializers
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention
from flaxformer.types import Initializer
from d3pm.text import model_utils
from flaxformer.types import Array
from flaxformer.types import DType


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  one_hot_embedding: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  head_dim: Optional[int] = None
  # Integer precision for MLP weights, or None for unquantized.
  mlp_weight_precision: Optional[int] = None
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('relu',)
  mlp_use_bias: bool = False
  mlp_intermediate_dropout_rate: float = 0.1
  position_embeddings: str = 'relative'
  absolute_attention_max_length: int = 2048
  relative_attention_num_buckets: int = 32
  relative_attention_max_distance: int = 128
  relative_attention_per_layer: bool = False
  rescale_logits: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  enable_dropout: bool = True
  decode: bool = False
  sow_intermediates: bool = False
  max_decode_length: int = 0
  temperature: float = 0.0001
  topk: int = 0
  eos_id: int = 1
  # MTF-matching T5 initializers.
  # These initializers are overridden in get_configs for scales other than 1.0.
  attention_kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal')
  mlp_kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  vocab_embed_init: Initializer = nn.initializers.normal(stddev=1.0)
  relative_position_bias_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_avg', 'uniform')
  final_kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None  # pylint: disable=g-bare-generic
  attention_cls: Type = dense_attention.MultiHeadDotProductAttention  # pylint: disable=g-bare-generic
  activation_partitioning_dims: int = 1
  final_layer_norm_use_scale: bool = True
  parallel_layer: bool = False
  use_extra_logit: bool = False
  input_conv_radius: Optional[int] = None
  mlp_conv_radius: Optional[int] = None
  attention_float32_logits: bool = False
  # Whether to store attention parameters qkvo with an explicit head dimension.
  split_head_kernel: bool = False
  # Whether to fuse qkv and the mlp input matrices for improved performance of
  # gradient calculations.
  fuse_kernels: bool = False


class TimestepEmbedBlock(nn.Module):
  """A simple FiLM layer to condition on a particular noise scale."""
  vocab_embed_init: Any
  num_steps: int = 1000
  dim: int = 768
  activation_fn: Any = nn.gelu
  use_sinusoidal_embeddings: bool = False

  @nn.compact
  def __call__(self, x, timestep):
    logging.info(
        'Using Timestep Embed Block with num_steps: %d, use_sinusoidal_embeddings: %s',
        self.num_steps, self.use_sinusoidal_embeddings)
    chex.assert_rank(x, 3)
    chex.assert_rank(timestep, 1)

    timestep = timestep.astype(jnp.int32)

    if self.use_sinusoidal_embeddings:
      embed = model_utils.get_timestep_embedding(
          timestep, self.dim, max_time=self.num_steps, dtype=x.dtype)
      embed = nn.Dense(features=self.dim)(embed)
      embed = nn.Dense(features=self.dim)(self.activation_fn(embed))
    else:
      embed = embedding.Embed(
          num_embeddings=self.num_steps,
          features=x.shape[-1],
          dtype=x.dtype,
          attend_dtype=jnp.float32,  # for logit training stability
          embedding_init=self.vocab_embed_init,
          name='timestep_embedder')(
              timestep)

    return x + embed[:, None]


class FiLMBlock(nn.Module):
  """A simple FiLM layer to condition on a particular noise scale."""
  vocab_embed_init: Any
  num_steps: int = 1000
  activation_fn: Any = nn.gelu

  @nn.compact
  def __call__(self, x, timestep):
    logging.info('Using FiLM block with num_steps %d.', self.num_steps)

    dim = x.shape[-1]

    chex.assert_rank(x, 3)
    chex.assert_rank(timestep, 1)

    timestep = timestep.astype(jnp.int32)

    embed = embedding.Embed(
        num_embeddings=self.num_steps,
        features=2 * dim,
        dtype=x.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=self.vocab_embed_init,
        name='film_timestep_embedder')(
            timestep)

    x = x * embed[:, None, dim:]
    x = x + embed[:, None, :dim]

    return self.activation_fn(x)


class EncoderDecoder(nn.Module, param_remapping.ParameterRemappable):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    encoder_factory: A callable that returns the lower-level Encoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `encoder_factory`.
    decoder_factory: A callable that returns the lower-level Decoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `decoder_factory`.
    dtype: DType for encoder/decoder to cast embedded inputs, and for attention
      mask generation.
    shared_token_embedder_factory: A callable that returns an embedder that can
      be shared between the encoder and decoder.
  """
  # Core components: encoder and decoder embedders and layers.
  encoder_factory: Any
  decoder_factory: Any

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: DType = jnp.float32

  shared_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None

  def setup(self):
    self.token_embedder = (
        self.shared_token_embedder_factory()  # pylint: disable=not-callable
        if self.shared_token_embedder_factory else None)
    self.encoder = self.encoder_factory(
        shared_token_embedder=self.token_embedder)
    self.decoder = self.decoder_factory(
        shared_token_embedder=self.token_embedder)

  def encode(self,
             encoder_input_tokens,
             encoder_padding_mask,
             *,
             deterministic = False):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_padding_mask: padding mask for encoder
      deterministic: Disables dropout if set to True.

    Returns:
      encoded feature array from the transformer encoder.
    """
    # Make padding attention mask.
    encoder_mask = dense_attention.make_attention_mask(
        encoder_padding_mask, encoder_padding_mask, dtype=self.dtype)

    return self.encoder(  # pytype: disable=attribute-error
        encoder_input_tokens,
        encoder_mask=encoder_mask,
        deterministic=deterministic)

  def decode(self,
             encoded,
             decoder_input_tokens,
             encoder_padding_mask,
             decoder_padding_mask,
             timestep=None,
             *,
             deterministic = False):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      decoder_input_tokens: input token to the decoder.
      encoder_padding_mask: padding mask for encoder.
      decoder_padding_mask: padding mask for decoder.
      timestep: timestep used to condition model.
      deterministic: Disables dropout if set to True.

    Returns:
      logits array from transformer decoder.
    """
    decoder_mask = dense_attention.make_attention_mask(
        decoder_padding_mask, decoder_padding_mask, dtype=self.dtype)

    if encoder_padding_mask is None:
      encoder_decoder_mask = None
    else:
      encoder_decoder_mask = dense_attention.make_attention_mask(
          decoder_padding_mask, encoder_padding_mask, dtype=self.dtype)

    # When computing the logits, we don't need decoder_target_tokens, which is
    # needed for computing the loss.
    return self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        timestep=timestep,
        deterministic=deterministic,
    )

  @property
  def encoder_embedder(self):
    return self.encoder.embedder

  def __call__(
      self,
      encoder_input_tokens,
      decoder_input_tokens,
      encoder_padding_mask,
      decoder_padding_mask,
      *,
      deterministic = False,
  ):
    """Applies Transformer model on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      encoder_padding_mask: padding mask for encoder.
      decoder_padding_mask: padding mask for decoder.
      deterministic: Disables dropout if set to True.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        encoder_input_tokens,
        encoder_padding_mask=encoder_padding_mask,
        deterministic=deterministic)

    return self.decode(
        encoded=encoded,
        decoder_input_tokens=decoder_input_tokens,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
        deterministic=deterministic,
    )


class DecoderOnly(nn.Module, param_remapping.ParameterRemappable):
  """Decoder-only model.

  This model sets up the relevant masking and uses Decoder to do the heavy
  lifting.

  Attributes:
    decoder_factory: Factory which will make the lower-level Decoder object. In
      the DecoderOnly usage, it will always be called with
      `shared_token_embedder` as None.
    dtype: DType for encoder/decoder to cast embedded inputs, and for attention
      mask generation.
  """
  # Core sub-component.
  decoder_factory: Any

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: DType = jnp.float32

  def setup(self):
    self.decoder = self.decoder_factory(shared_token_embedder=None)

  def __call__(self,
               decoder_input_tokens,
               decoder_padding_mask,
               *,
               deterministic = False):
    """Applies a decoder-only (more precisely, encoder-only) model.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is typically a shifted version of the former. For a packed dataset, it
    usually has additional processing applied. For example, the first element of
    each sequence has id 0 instead of the shifted EOS id from the previous
    sequence.

    Args:
      decoder_input_tokens: input token to the decoder.
      decoder_padding_mask: padding mask for decoder.
      deterministic: Disables dropout if set to True.

    Returns:
      logits array from DecoderOnly model.
    """
    decoder_mask = dense_attention.make_attention_mask(
        decoder_padding_mask, decoder_padding_mask, dtype=self.dtype)

    # We reuse Decoder class, which can optionally takes in encoded and
    # encoder_decoder_mask. These are used when Decoder is used in the context
    # of encoder-decoder model. Here, we don't have an encoder.
    return self.decoder(  # pytype: disable=attribute-error
        encoder_outputs=None,
        decoder_input_tokens=decoder_input_tokens,
        decoder_mask=decoder_mask,
        deterministic=deterministic)


class DecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer encoder-decoder layer.

  Attributes:
    self_attention: An instance of a self-attention module.
    encoder_decoder_attention: Encoder-decoder attention module. This must be
      non-None if attending to encoded representations.
    mlp: The MLP module, applied after both attention modules.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relative_position_bias_factory: A callable that returns relative position
      bias instances. This should only be used for per-layer relative position
      biases; please use `shared_relative_position_bias` if they are shared
      among layers.
    shared_relative_position_bias: An instance of a shared relative position
      bias module, usually owned by the Decoder.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the decoder layer.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
  """
  self_attention: nn.Module
  encoder_decoder_attention: Optional[nn.Module]
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  vocab_embed_init: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  parallel: bool = False
  sow_intermediates: bool = False
  use_film_layers: bool = True
  num_steps: int = 1000

  def setup(self):
    if (self.relative_position_bias_factory is not None and
        self.shared_relative_position_bias is not None):
      raise ValueError(
          'Please set at most one of relative_position_bias_factory and shared_relative_position_bias. '
          '(They can both be None however, e.g. for absolute position embeds.)')
    self.relpos_bias = (
        self.relative_position_bias_factory()  # pylint: disable=not-callable
        if self.relative_position_bias_factory is not None else
        self.shared_relative_position_bias)

    if self.parallel:
      self.layer_norm = self.layer_norm_factory()
      self.dropout = self.dropout_factory()
    else:
      self.pre_self_attention_layer_norm = self.layer_norm_factory()
      self.post_self_attention_dropout = self.dropout_factory()
      self.pre_cross_attention_layer_norm = self.layer_norm_factory()
      self.post_cross_attention_dropout = self.dropout_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_mlp_dropout = self.dropout_factory()

    self.film_block = FiLMBlock(
        num_steps=self.num_steps, vocab_embed_init=self.vocab_embed_init)

  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               timestep=None,
               *,
               deterministic = False,
               decode = False,
               max_decode_length = None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Input data from encoder with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, block is Decoder
        only.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask with shape [
        batch_size, 1, decoder_seq_length, encoder_seq_length].
      timestep: timestep for FiLM layer.
      deterministic: Disables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.

    Returns:
      output after transformer encoder-decoder block.
    """
    layer_input = targets
    del targets
    # Shared relative position embedding attention biases.
    if self.relpos_bias:
      if decode and max_decode_length:
        decoder_bias = self.relpos_bias(max_decode_length, max_decode_length,
                                        False)
      else:
        decoder_bias = self.relpos_bias(layer_input.shape[-2],
                                        layer_input.shape[-2], False)
    else:
      decoder_bias = None
    # No relative embeddings are used for encoder-decoder cross attention.
    encoder_decoder_bias = None

    # Decoder block.
    assert layer_input.ndim == 3
    layer_input = activation_partitioning.with_sharding(
        layer_input, self.activation_partitioning_dims)

    if self.parallel:
      x = self.layer_norm(layer_input, decode=decode)
      x = activation_partitioning.with_sharding(
          x, self.activation_partitioning_dims)
      x = activation_partitioning.with_sharding(x, 1)

      if timestep is not None and self.use_film_layers:
        logging.info('Using FiLM decoder block with num_steps: %d.',
                     self.num_steps)
        x = self.film_block(x, timestep)
      else:
        logging.info('FiLM decoder block is disabled.')

      y = (
          self.self_attention(x, x, decoder_mask, decoder_bias, decode=decode) +
          self.mlp(x, decode=decode))
      if encoded is not None:
        y += self.encoder_decoder_attention(x, encoded, encoder_decoder_mask,
                                            encoder_decoder_bias)
      y *= (3 if encoded is not None else 2)**-0.5
      z = layer_input + self.dropout(y, deterministic=deterministic)
    else:
      # layer_input is derived from decoder_input_tokens.
      x = self.pre_self_attention_layer_norm(layer_input, decode=decode)
      x = activation_partitioning.with_sharding(
          x, self.activation_partitioning_dims)
      x = activation_partitioning.with_sharding(x, 1)
      # The first and second arguments to the attention are the same,
      # i.e., this is a self-attention layer.

      if timestep is not None and self.use_film_layers:
        logging.info('Using FiLM decoder block.')
        x = self.film_block(x, timestep)
      else:
        logging.info('FiLM decoder block is disabled.')

      x = self.self_attention(x, x, decoder_mask, decoder_bias, decode=decode)
      x = layer_input + self.post_self_attention_dropout(
          x, deterministic=deterministic)
      x = activation_partitioning.with_sharding(
          x, self.activation_partitioning_dims)

      # Encoder-Decoder block.
      if encoded is None:
        # If encoder outputs not provided, skip attending from decoder to
        # encoder.  This results in a decoder only block.
        y = x
      else:
        if self.encoder_decoder_attention is None:
          raise ValueError('Expected encoder_decoder_attention to be populated '
                           'when called with `encoded` inputs.')
        y = self.pre_cross_attention_layer_norm(x, decode=decode)
        y = activation_partitioning.with_sharding(
            y, self.activation_partitioning_dims)
        y = activation_partitioning.with_sharding(y, 1)
        y = self.encoder_decoder_attention(y, encoded, encoder_decoder_mask,
                                           encoder_decoder_bias)
        y = x + self.post_cross_attention_dropout(
            y, deterministic=deterministic)
        y = activation_partitioning.with_sharding(
            y, self.activation_partitioning_dims)

      # MLP block.
      z = self.pre_mlp_layer_norm(y, decode=decode)
      z = activation_partitioning.with_sharding(
          z, self.activation_partitioning_dims)
      z = activation_partitioning.with_sharding(z, 1)
      z = self.mlp(z, decode=decode)
      z = y + self.post_mlp_dropout(z, deterministic=deterministic)

    z = activation_partitioning.with_sharding(z,
                                              self.activation_partitioning_dims)
    if self.sow_intermediates:
      self.sow('intermediates', 'activations', z)
    return z


class Decoder(nn.Module, param_remapping.ParameterRemappable):
  """A stack of decoder layers.

  This module can be used with or without the encoder stack. To use without an
  encoder, pass in encoded=None. This will bypass the encoder-decoder attention.

  Attributes:
    layer_factory: A callable that returns a DecoderLayer.
    dropout_factory: A callable that returns the dropout to apply to the input
      and before the final logits.
    layer_norm_factory: A callable that returns a layer norm.
    output_logits_factory: A callable that returns the output logits. If not
      provided, then the token embedders are used.
    num_layers: Number of layers to generate.
    dtype: DType to cast the embedded inputs.
    shared_relative_position_bias_factory: A callable that returns a relative
      position bias instance which will be shared for all encoder layers. Only
      set this if using shared relative position biases.
    token_embedder_factory: A callable that returns a token embedder. Please
      provide either this or `shared_token_embedder`.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
    position_embedder_factory: A callable that returns an absolute position
      embedder. Only provide this if you want absolute position embeddings.
    sow_intermediates: whether to track intermediates using Module.sow.
  """
  layer_factory: Any
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: DType = jnp.float32

  shared_relative_position_bias_factory: Optional[Callable[[],
                                                           nn.Module]] = None
  output_logits_factory: Optional[Callable[[], nn.Module]] = None

  # Embedders: Either a token_embedder_factory factory or shared token embedder
  # must be provided. The position embedder is optional and provided when
  # absolute position embeddings are desired.
  token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  shared_token_embedder: Optional[embedding.Embed] = None
  position_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  timestep_embed_factory: Optional[Callable[[], nn.Module]] = None

  sow_intermediates: bool = False

  def setup(self):
    # Set up the embedders.
    if (self.token_embedder_factory,
        self.shared_token_embedder).count(None) != 1:
      raise ValueError(
          'Please set exactly one of token_embedder_factory or '
          'shared_token_embedder. token_embedder_factory was %s, and '
          'shared_token_embedder was %s.' %
          (self.token_embedder_factory, self.shared_token_embedder))
    if self.shared_token_embedder is not None:
      embedders = {'token_ids': self.shared_token_embedder}
    else:
      self.token_embedder_factory: Callable[[], embedding.Embed]
      self.token_embedder = self.token_embedder_factory()
      embedders = {'token_ids': self.token_embedder}
    if self.position_embedder_factory is not None:
      self.position_embedder_factory: Callable[[], embedding.Embed]
      self.position_embedder = self.position_embedder_factory()
      embedders['position_ids'] = self.position_embedder
    self.embedder = embedding.MultiEmbed(embedders)

    self.input_dropout = self.dropout_factory()

    self.relpos_bias = (
        self.shared_relative_position_bias_factory()  # pylint: disable=not-callable
        if self.shared_relative_position_bias_factory is not None else None)
    self.layers = [
        self.layer_factory(shared_relative_position_bias=self.relpos_bias)
        for _ in range(self.num_layers)
    ]
    self.decoder = common.TransparentLayerSequence(self.layers)

    self.decoder_norm = self.layer_norm_factory()
    self.output_dropout = self.dropout_factory()

    if self.output_logits_factory:
      self.output_logits_factory: Callable[[], nn.Module]
      self.logits_dense = self.output_logits_factory()
    else:
      self.logits_dense = None

    if self.timestep_embed_factory is not None:
      logging.info('Using timestep embedding layer.')
      self.timestep_embed = self.timestep_embed_factory()  # pylint: disable=not-callable
    else:
      logging.info('Timestep embedding layer is disabled.')
      self.timestep_embed = None

  def embed_and_combine_inputs(
      self,
      decoder_input_tokens,
      decoder_positions=None,
      timestep=None,
      *,
      deterministic = False,
      decode = False,
  ):
    """Returns the combined embedded decoder inputs for further processing."""
    assert decoder_input_tokens.ndim == 2  # (batch, len)

    if 'position_ids' in self.embedder.embedders:
      if decoder_positions is None:
        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = jnp.arange(seq_length)[None, :]
      embedded_inputs = self.embedder(
          token_ids=decoder_input_tokens,
          position_ids=decoder_positions,
          decode=decode)
    else:
      embedded_inputs = self.embedder(
          token_ids=decoder_input_tokens, decode=decode)

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=deterministic)

    embedded_inputs = embedded_inputs.astype(self.dtype)

    if self.timestep_embed is not None:
      embedded_inputs = self.timestep_embed(embedded_inputs, timestep)

    return embedded_inputs

  def decode_from_continuous_inputs(
      self,
      embedded_inputs,
      encoder_outputs,
      decoder_positions=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
      timestep=None,
      *,
      deterministic = False,
      decode = False,
      max_decode_length = None,
  ):
    """Applies the decoder on the continuous (embedded) inputs."""
    # If encoded is not given, this block is decoder only and does not contain
    # attention from decoder to encoder.
    if encoder_outputs is not None:
      assert encoder_outputs.ndim == 3  # (batch, len, depth)

    # Apply the decoder layers, attending to the encoder outputs (if provided),
    # and attending to previous decoder inputs (by masking future inputs).
    decoder_outputs = self.decoder(
        embedded_inputs,
        encoder_outputs,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        timestep=timestep,
        deterministic=deterministic,
        decode=decode,
        max_decode_length=max_decode_length)

    # Post-process final decoder layer outputs.
    decoder_outputs = self.decoder_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=deterministic)

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    # Decoded Logits
    if self.logits_dense is not None:
      logits = self.logits_dense(decoder_outputs)
    else:
      # Use the transpose of embedding matrix for logit transform.

      logits = self.embedder.embedders['token_ids'].attend(decoder_outputs)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])
    return logits

  def __call__(self,
               encoder_outputs,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               timestep=None,
               *,
               deterministic = False,
               decode = False,
               max_decode_length = None):
    """Applies Transformer model on the inputs.

    Args:
      encoder_outputs: The outputs from the encoder. If None, do not attend to
        encoder outputs, resulting in a decoder only model (i.e. language
        model).
      decoder_input_tokens: The decoder input token IDs.
      decoder_positions: Decoder subsequence positions for packed examples.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      timestep: timestep for FiLM-style embedding.
      deterministic: Disables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.

    Returns:
      The decoder output logits for next token prediction.
    """
    embedded_inputs = self.embed_and_combine_inputs(
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        timestep=timestep,
        deterministic=deterministic,
        decode=decode,
    )
    logits = self.decode_from_continuous_inputs(
        embedded_inputs,
        encoder_outputs,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        timestep=timestep,
        deterministic=deterministic,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits


@dataclasses.dataclass
class ModelFactory:
  """Makes the EncoderDecoder or DecoderOnly models.

  Attributes:
    config: Legacy transformer config.
    legacy_different_encoder_output_dropout: Use a different encoder output
      dropout module to exactly mirror legacy T5X behavior. This may or may not
      have been intentional.
    use_timestep_embeddings: whether to embed timestep in embeddings.
    use_film_layers: whether to use FiLM layers at each decoder block.
    num_steps: number of steps in process.
  """
  config: TransformerConfig
  legacy_different_encoder_output_dropout: bool = True
  use_timestep_embeddings: bool = True
  use_film_layers: bool = True
  num_steps: int = 1000

  @property
  def _abs_posemb(self):
    return self.config.position_embeddings == 'absolute'

  @property
  def _shared_relative_position_bias(self):
    return (self.config.position_embeddings == 'relative' and
            not self.config.relative_attention_per_layer)

  @property
  def _per_layer_relative_position_bias(self):
    return (self.config.position_embeddings == 'relative' and
            self.config.relative_attention_per_layer)

  def _make_dropout(self):
    """Dropout for everything except the last encoder layer."""
    return nn.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))

  def _make_encoder_output_dropout(self):
    if self.legacy_different_encoder_output_dropout:
      return nn.Dropout(rate=self.config.dropout_rate)
    else:
      return self._make_dropout()

  def _make_encoder_layer_norm(self):
    if self.config.input_conv_radius:
      conv = convolution.Depthwise1dConv(
          radius=self.config.input_conv_radius,
          dtype=self.config.dtype,
          autoregressive=False)
      return layer_norm.T5LayerNorm(use_scale=False, conv=conv)
    else:
      return layer_norm.T5LayerNorm(use_scale=True)

  def _make_decoder_layer_norm(self):
    if self.config.input_conv_radius:
      conv = convolution.Depthwise1dConv(
          radius=self.config.input_conv_radius,
          dtype=self.config.dtype,
          autoregressive=True)
      return layer_norm.T5LayerNorm(use_scale=False, conv=conv)
    else:
      return layer_norm.T5LayerNorm(use_scale=True)

  def _make_final_layer_norm(self):
    return layer_norm.T5LayerNorm(
        use_scale=self.config.final_layer_norm_use_scale)

  def _make_attention(self):
    """Makes the attention module (for encoder and both decoder attentions)."""
    return self.config.attention_cls(
        num_heads=self.config.num_heads,
        dtype=self.config.dtype,
        qkv_features=self.config.qkv_dim,
        head_dim=self.config.head_dim,
        kernel_init=self.config.attention_kernel_init,
        bias_init=self.config.bias_init,
        use_bias=False,
        broadcast_dropout=True,
        rescale_logits=self.config.rescale_logits,
        dropout_rate=self.config.attention_dropout_rate,
        use_extra_logit=self.config.use_extra_logit,
        float32_logits=self.config.attention_float32_logits)

  def _make_mlp_conv(self):
    if self.config.mlp_conv_radius:
      return convolution.Depthwise1dConv(
          radius=self.config.mlp_conv_radius, dtype=self.config.dtype)
    else:
      return None

  def _make_mlp(self):
    """Makes the MLP module (for encoder and decoder)."""
    return dense.MlpBlock(
        use_bias=self.config.mlp_use_bias,
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        kernel_init=self.config.mlp_kernel_init,
        bias_init=self.config.bias_init,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        intermediate_conv=self._make_mlp_conv(),
        final_dropout_rate=0,
    )

  def _make_token_embedder(self, vocab_size):
    """Makes a token embedder (shared or separate for encoder/decoder)."""
    return embedding.Embed(
        num_embeddings=vocab_size,
        features=self.config.emb_dim,
        cast_input_dtype=jnp.int32,
        dtype=self.config.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=self.config.vocab_embed_init,
        one_hot=self.config.one_hot_embedding,
        name='token_embedder')

  def _make_shared_token_embedder(self):
    """Makes a shared token embedder, if it is configured."""
    assert self.config.share_embeddings, "Shouldn't be called w/o shared emb."
    if self.config.output_vocab_size is not None:
      if self.config.output_vocab_size != self.config.vocab_size:
        raise ValueError("Can't share embedding with different vocab sizes; "
                         f'got encoder vocab size {self.config.vocab_size} and '
                         f'decoder vocab size {self.config.output_vocab_size}.')
    return self._make_token_embedder(self.config.vocab_size)

  def _make_abs_position_embed(self):
    """Makes absolute position embeddings."""
    if self.config.posemb_init is None:
      return embedding.FixedEmbed(
          embedding_init=initializers.sinusoidal(),
          features=self.config.emb_dim,
          max_length=self.config.absolute_attention_max_length,
          dtype=jnp.float32,  # always use float32 for embeddings.
          name='position_embedder')
    else:
      return embedding.PositionEmbed(
          num_embeddings=self.config.absolute_attention_max_length,
          features=self.config.emb_dim,
          embedding_init=self.config.posemb_init,
          dtype=jnp.float32,  # always use float32 for embeddings.
          name='position_embedder')

  def _make_encoder_token_embedder(self):
    return self._make_token_embedder(self.config.vocab_size)

  def _make_decoder_token_embedder(self):
    return self._make_token_embedder(self.config.output_vocab_size)

  def _make_encoder_position_embedder(self):
    return self._make_abs_position_embed()

  def _make_decoder_position_embedder(self):
    return self._make_abs_position_embed()

  def _make_relative_position_bias(self):
    return relative_position_biases.RelativePositionBiases(
        num_buckets=self.config.relative_attention_num_buckets,
        max_distance=self.config.relative_attention_max_distance,
        num_heads=self.config.num_heads,
        dtype=self.config.dtype,
        embedding_init=self.config.relative_position_bias_init)

  def _make_encoder_layer(self, *, shared_relative_position_bias):
    return t5_architecture.EncoderLayer(
        attention=self._make_attention(),
        mlp=self._make_mlp(),
        dropout_factory=self._make_dropout,
        layer_norm_factory=self._make_encoder_layer_norm,
        relative_position_bias_factory=(self._make_relative_position_bias if
                                        self._per_layer_relative_position_bias
                                        else None),
        shared_relative_position_bias=shared_relative_position_bias,
        activation_partitioning_dims=self.config.activation_partitioning_dims,
        parallel=self.config.parallel_layer,
        sow_intermediates=self.config.sow_intermediates)  # pytype: disable=wrong-keyword-args

  def _make_decoder_layer(self, *, shared_relative_position_bias):
    return DecoderLayer(
        self_attention=self._make_attention(),
        encoder_decoder_attention=self._make_attention(),
        mlp=self._make_mlp(),
        dropout_factory=self._make_dropout,
        layer_norm_factory=self._make_decoder_layer_norm,
        relative_position_bias_factory=(self._make_relative_position_bias if
                                        self._per_layer_relative_position_bias
                                        else None),
        shared_relative_position_bias=shared_relative_position_bias,
        activation_partitioning_dims=self.config.activation_partitioning_dims,
        use_film_layers=self.use_film_layers,
        parallel=self.config.parallel_layer,
        sow_intermediates=self.config.sow_intermediates,
        vocab_embed_init=self.config.vocab_embed_init,
        num_steps=self.num_steps)  # pytype: disable=wrong-keyword-args

  def _make_decoder_output_logits(self):
    return dense.DenseGeneral(
        self.config.output_vocab_size,
        dtype=self.config.dtype,
        kernel_init=self.config.final_kernel_init,
        bias_init=self.config.bias_init,
        use_bias=False)

  def make_low_level_encoder(
      self,
      *,
      shared_token_embedder = None,
  ):
    """Please use make_encoder_decoder and call encode() if you can."""
    return t5_architecture.Encoder(
        shared_token_embedder=shared_token_embedder,
        token_embedder_factory=(None if self.config.share_embeddings else
                                self._make_encoder_token_embedder),
        position_embedder_factory=(self._make_encoder_position_embedder
                                   if self._abs_posemb else None),
        shared_relative_position_bias_factory=(
            self._make_relative_position_bias
            if self._shared_relative_position_bias else None),
        layer_factory=self._make_encoder_layer,
        input_dropout_factory=self._make_dropout,
        output_dropout_factory=self._make_encoder_output_dropout,
        layer_norm_factory=self._make_final_layer_norm,
        num_layers=self.config.num_encoder_layers,
        dtype=self.config.dtype,  # pytype: disable=wrong-keyword-args
    )

  def _make_timestep_embedder(self):
    return TimestepEmbedBlock(
        num_steps=self.num_steps,
        dim=self.config.emb_dim,
        vocab_embed_init=self.config.vocab_embed_init)

  def make_low_level_decoder(
      self,
      *,
      shared_token_embedder = None,
  ):
    """Please use make_decoder_only if you can."""
    return Decoder(
        shared_token_embedder=shared_token_embedder,
        token_embedder_factory=(None if self.config.share_embeddings else
                                self._make_decoder_token_embedder),
        position_embedder_factory=(self._make_decoder_position_embedder
                                   if self._abs_posemb else None),
        shared_relative_position_bias_factory=(
            self._make_relative_position_bias
            if self._shared_relative_position_bias else None),
        layer_factory=self._make_decoder_layer,
        dropout_factory=self._make_dropout,
        layer_norm_factory=self._make_final_layer_norm,
        output_logits_factory=(self._make_decoder_output_logits if
                               not self.config.logits_via_embedding else None),
        num_layers=self.config.num_decoder_layers,
        dtype=self.config.dtype,
        timestep_embed_factory=self._make_timestep_embedder
        if self.use_timestep_embeddings else None,
        sow_intermediates=self.config.sow_intermediates,  # pytype: disable=wrong-keyword-args
    )

  def make_encoder_decoder(self):
    return EncoderDecoder(
        shared_token_embedder_factory=(self._make_shared_token_embedder if
                                       self.config.share_embeddings else None),
        encoder_factory=self.make_low_level_encoder,
        decoder_factory=self.make_low_level_decoder,
        dtype=self.config.dtype,  # pytype: disable=wrong-keyword-args
    )

  def make_decoder_only(self):
    return DecoderOnly(
        decoder_factory=self.make_low_level_decoder,
        dtype=self.config.dtype,  # pytype: disable=wrong-keyword-args
    )


def make_decoder_only(config):
  return ModelFactory(config=config).make_decoder_only()


def make_encoder_decoder(config,
                         num_steps=1000,
                         use_timestep_embeddings=False,
                         use_film_layers=False):
  return ModelFactory(
      config=config,
      num_steps=num_steps,
      use_timestep_embeddings=use_timestep_embeddings,
      use_film_layers=use_film_layers).make_encoder_decoder()


def expand_dims(*args):
  return jax.tree.map(lambda x: x[None], args)


@gin.configurable(denylist=['train'])
class CategoricalDiffusionModel(nn.Module):
  """A wrapper for a simple dual-encoder Transformer.

  Attributes:
    config: a TransformerConfig object.
    encoder_decoder_model: the model class to use.
    train: whether or not we are currently training.
  """

  config: TransformerConfig
  encoder_decoder_model: Callable[[Any, Ellipsis], Any] = make_encoder_decoder
  num_steps: int = 1000
  use_timestep_embeddings: bool = True
  use_film_layers: bool = True
  train: bool = False

  def setup(self):
    cfg = self.config

    self.module = self.encoder_decoder_model(  # pylint: disable=redundant-keyword-arg
        config=cfg,
        num_steps=self.num_steps,
        use_timestep_embeddings=self.use_timestep_embeddings,
        use_film_layers=self.use_film_layers)

    logging.log_first_n(logging.INFO,
                        'CategoricalDiffusionModel configuration:', 1)
    logging.log_first_n(logging.INFO, 'module is: %s', 1, self.module)
    logging.log_first_n(logging.INFO, 'config is: %s', 1, cfg)

  def encode(self, encoder_input_tokens, encoder_padding_mask):
    """Encodes an input sequence with a particular encoder_mask.

    Args:
      encoder_input_tokens: (seq_len,) array to be encoded.
      encoder_padding_mask: (seq_len,) mask array to use for masking.

    Returns:
      A (seq_len, emb_dim) array of encoded inputs.
    """

    chex.assert_rank([encoder_input_tokens, encoder_padding_mask], [1, 1])
    chex.assert_type([encoder_input_tokens, encoder_padding_mask], [int, bool])

    (encoder_input_tokens,
     encoder_padding_mask) = expand_dims(encoder_input_tokens,
                                         encoder_padding_mask)

    encoded = self.module.encode(
        encoder_input_tokens=encoder_input_tokens,
        encoder_padding_mask=encoder_padding_mask,
        deterministic=not self.train)

    chex.assert_rank(encoded, 3)
    chex.assert_equal_shape_prefix([encoded, encoder_input_tokens], 2)

    return encoded[0]

  @nn.compact
  def predict_length(self, encoded_inputs, targets):
    """Predicts the length of the target sequence from the encoded inputs."""

    return nn.Dense(features=targets.shape[-1])(encoded_inputs[0])

  def decode(self,
             encoded,
             decoder_input_tokens,
             encoder_padding_mask,
             decoder_padding_mask,
             timestep=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      decoder_input_tokens: input token to the decoder.
      encoder_padding_mask: padding mask for encoder.
      decoder_padding_mask: padding mask for decoder.
      timestep: optionally, a timestep to condition the input on.

    Returns:
      logits array from transformer decoder.
    """
    chex.assert_rank([decoder_input_tokens, decoder_padding_mask], [1, 1])

    if encoded is not None:
      chex.assert_rank([encoded, encoder_padding_mask], [2, 1])

    (encoded, decoder_input_tokens, encoder_padding_mask, decoder_padding_mask,
     timestep) = expand_dims(encoded, decoder_input_tokens,
                             encoder_padding_mask, decoder_padding_mask,
                             timestep)

    logits = self.module.decode(
        encoded=encoded,
        decoder_input_tokens=decoder_input_tokens,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
        timestep=timestep,
        deterministic=not self.train,
    )

    chex.assert_equal_shape_prefix([logits, decoder_input_tokens], 2)

    return logits[0]

  def custom_init(self, targets, inputs=None):
    if inputs is not None:
      encoded_inputs = self.encode(inputs, inputs > 0)
      length_logits = self.predict_length(encoded_inputs, targets)
      encoder_padding_mask = inputs > 0
    else:
      encoded_inputs, length_logits, encoder_padding_mask = None, None, None

    timestep = jnp.ones((), jnp.float32)
    logits = self.decode(
        encoded_inputs,
        targets,
        encoder_padding_mask,
        targets > 0,
        timestep=timestep)
    return logits, length_logits

  def __call__(self, inputs, targets):
    return self.custom_init(inputs, targets)  # pylint: disable=arguments-out-of-order


@gin.configurable(module='models')
def build_transformer_config(**kwargs):
  """Build Transformer model config."""
  return TransformerConfig(**kwargs)


def transformer_init(model_cls, task, dataset_info):
  """Sets the vocab_size argument on our transformer models."""
  del task

  config = build_transformer_config(
      vocab_size=dataset_info.vocab.vocab_size,
      output_vocab_size=dataset_info.vocab.vocab_size)

  return functools.partial(model_cls, config=config)
