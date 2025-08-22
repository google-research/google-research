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

"""Variation Transformer Network for Layout Generation."""

from typing import Any, Callable, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class MlpBlock(nn.Module):
  """MLP / feed-forward block.

  Attributes:
    config: ml_collections.ConfigDict dataclass containing hyperparameters.
  """
  config: ml_collections.ConfigDict
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros
  deterministic: bool = False
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    config = self.config
    out_dim = inputs.shape[-1]
    x = nn.Dense(config.mlp_dim,
                 dtype=self.dtype,
                 kernel_init=self.kernel_init,
                 bias_init=self.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=self.deterministic)
    output = nn.Dense(out_dim,
                      dtype=self.dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)(x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=self.deterministic)
    return output


class Layer(nn.Module):
  """Transformer layer.

  Attributes:
    config: ml_collections.ConfigDict dataclass containing hyperparameters.
  """
  config: ml_collections.ConfigDict
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros
  deterministic: bool = False
  is_train: bool = True
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self,
               inputs,
               attn_mask=None):
    """Applies Layer module.

    Args:
      inputs: input data.
      attn_mask: self-attention mask.

    Returns:
      output from Layer: [batch_size, seq_length, model_dim].
    """
    config = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=self.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=self.deterministic,
        decode=not self.is_train)(x, attn_mask)

    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=self.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(config, self.kernel_init, self.bias_init, self.deterministic,
                 self.dtype)(
                     y)

    return x + y


class Decoder(nn.Module):
  """Transformer Decoder for layout generation.

  Attributes:
    config: ml_collections.ConfigDict dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: ml_collections.ConfigDict
  shared_embedding: Any = None
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros
  deterministic: bool = False
  is_train: bool = True
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self,
               targets,
               label=None,
               latent_z=None,
               decoder_mask=None,
               decode_step=0):
    """Applies decoder on the inputs and latent vectors.

    Args:
      targets: target inputs.
      label: training input vertical info (always None for now).
      latent_z: latent variable sampled from encoder z ~ q(z|x).
      decoder_mask: decoder self-attention mask.
      decode_step: current decode step during inference process.

    Returns:
      output: [batch_size, seq_length, vocab_size]
    """
    config = self.config
    assert targets.ndim == 2  # (batch, len)
    label_embed = nn.Embed(
        num_embeddings=32,
        features=config.emb_dim,
        embedding_init=nn.initializers.normal(stddev=config.emb_dim**-0.5),
        name='label_embedding')

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=config.vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=config.emb_dim ** -0.5))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    y = output_embed(y)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=self.deterministic)
    y = y.astype(self.dtype)
    if label is not None:
      label = label.astype('int32')
      label_emb = label_embed(label)
      y = jnp.where(
          jnp.logical_or(self.is_train, decode_step == 0),
          jax.lax.dynamic_update_slice(y, label_emb, (0, 0, 0)), y)

    if config.use_vae:
      latent_z = jnp.expand_dims(latent_z, 1)
      # # Use the latent vector to y as the kick-off representation.
      y = jnp.where(
          jnp.logical_or(self.is_train, decode_step == 0),
          jax.lax.dynamic_update_slice(y, latent_z, (0, 0, 0)), y)

    for lyr in range(config.num_layers):
      y = Layer(
          config,
          self.kernel_init,
          self.bias_init,
          self.deterministic,
          self.is_train,
          self.dtype,
          name=f'decoderblock_{lyr}')(y, decoder_mask)

    y = nn.LayerNorm(dtype=self.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    # Use the transpose of embedding matrix for logit transform.
    logits = output_embed.attend(y.astype(jnp.float32))
    return logits


class TransformerDecoder(nn.Module):
  """Transformer Model for layout generation.

  Attributes:
    config: ml_collections.ConfigDict dataclass containing hyperparameters.
    dtype: the dtype of the computation (default: float32).
    deterministic: whether enable storcastic process such as dropout.
    is_train: training or decoding time.
    kernel_init: initializer for the kernel.
    bias_init: initializer for the bias.
  """
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  deterministic: bool = False
  is_train: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros

  def setup(self):
    # Update config dict.
    config = self.config

    if config.share_embeddings:
      self.shared_embedding = nn.Embed(
          num_embeddings=config.vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=config.emb_dim ** -0.5))
    else:
      self.shared_embedding = None
    self.decoder = Decoder(config,
                           self.shared_embedding,
                           self.kernel_init,
                           self.bias_init,
                           self.deterministic,
                           self.is_train,
                           self.dtype)

  def decode(self, layout_seq, label=None, latent_z=None, decode_step=0):
    """Applies Transformer decoder on decoder input and sampled latent vector.

    Args:
      layout_seq: input data for the model [batch_size, layout_seq_len].
      label: training input vertical info (always None for now).
      latent_z: latent variable sampled from encoder z ~ q(z|x).
      decode_step: decode_step: current decode step during inference process.

    Returns:
      logits array: [batch_size, seq_length, vocab_size]
    """
    # During training, Remove the last token of the seq as the decoder input.
    # Our target will be the y[:, 1:, :]
    if self.is_train:
      dec_input = layout_seq[:, :-1]
    else:
      dec_input = layout_seq

    # Make padding attention masks.
    if not self.is_train:
      decoder_mask = None
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(
              dec_input > 0, dec_input > 0, dtype=self.dtype),
          nn.make_causal_mask(dec_input, dtype=self.dtype))

    logits = self.decoder(
        dec_input,
        label,
        latent_z,
        decoder_mask=decoder_mask,
        decode_step=decode_step)
    return logits.astype(self.dtype)

  def __call__(self, layout_seq, label, z_rng):
    """Applies Transformer decoder on the input layout sequence.

    Args:
      layout_seq: input data for the model [batch_size, layout_seq_len].
      label: training input vertical info (always None for now).
      z_rng: PRNGKey for latent vector sampling.

    Returns:
      logits array [batch_size, layout_seq_length, vocab_size]
    """
    return (self.decode(layout_seq, label), None)
