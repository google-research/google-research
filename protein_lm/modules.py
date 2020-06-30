# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Flax Modules."""
from flax import nn
from jax import lax
import jax.numpy as jnp
import numpy as np


def shift_right(x, train=True, bos_token=0):
  """Shift the input to the right by padding on axis 1 at train time."""
  if train:
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)  # Padding on axis=1
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(bos_token))
    return padded[:, :-1]
  else:
    # Do nothing when not in train mode, as then the sequence length is 1.
    return x


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def get_positional_encodings(max_len, emb_size, concatenate=False):
  """Compute positional encodings as described in the Transformer paper.

  Positional encoddings use sine and cosine functions of different frequencies:

    PE(pos, 2i) = sin(pos / (10000^(2i / emb_size)))
    PE(pos, 2i + 1) = cos(pos / (10000^(2i / emb_size))

  where pos is the position and i is the dimension

  Reference: Section 3.5 in
    [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

  Args:
    max_len: An int with the maximum possible length for the input.
    emb_size: An int with the embedding size.
    concatenate: A bool indicating whether to concatenate or interleave the
      sines and cosines. The default is False to match the Transformer paper.

  Returns:
    An array of shape (1, max_len, emb_size) with positional embeddings.
  """
  def _get_angles_per_position(position, dim, emb_size):
    denominator = np.power(10000, (2 * (dim // 2)) / np.float32(emb_size))
    return position / denominator

  # Create the arguments for the sines and cosines.
  angles = _get_angles_per_position(np.arange(max_len)[:, np.newaxis],
                                    np.arange(emb_size)[np.newaxis, :],
                                    emb_size)

  # Apply sine to the odd positions.
  sines = np.sin(angles[:, 0::2])

  # Apply cosine to the even positions.
  cosines = np.cos(angles[:, 1::2])

  if concatenate:
    # See e.g. http://jalammar.github.io/illustrated-transformer/.
    output = np.concatenate([sines, cosines], axis=-1)
  else:
    # See e.g.
    # https://kazemnejad.com/blog/transformer_architecture_positional_encoding/.
    output = np.zeros_like(angles)
    output[:, 0::2] = sines
    output[:, 1::2] = cosines

  output = output[np.newaxis, :, :]
  return output


def sinusoidal_init(max_len=2048):
  """Weight initializer based on sinusoial positional embeddings.

  Args:
    max_len: An int with the maximum possible length for the input.

  Returns:
    Callable taking as input a key and a shape (..., emb_size) and returning
      positional embeddings of shape (1, max_len, emb_size).
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    return jnp.array(get_positional_encodings(max_len, shape[-1]))
  return init


class AddLearnedPositionalEncodings(nn.Module):
  """Adds learned positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=2048,
            posemb_init=nn.initializers.normal(stddev=1.0),
            cache=None):
    """Applies the AddLearnedPositionalEncodings module.

    Args:
      inputs: input data
      inputs_positions: input position indices for packed sequences.
      max_len: maximum possible length for the input
      posemb_init: positional embedding initializer
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    if inputs.ndim != 3:
      raise ValueError('Wrong number of dimensions: found %d expected 3' %
                       inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one)
        cache.store(cache_entry)
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class AddSinusoidalPositionalEncodings(nn.Module):
  """Adds the standard sinusoidal positional encodings to the inputs."""

  def apply(self, inputs, max_len=2048):
    """Applies the AddSinusoidalPositionalEncodings module.

    Args:
      inputs: An array of shape (batch_size, length, emb_size) with the token
        embeddings.
      max_len: An int with the maximum possible length for the input.

    Returns:
      An array of shape (batch_size, length, emb_size).
    """
    if inputs.ndim != 3:
      raise ValueError('Wrong number of dimensions: found %d expected 3' %
                       inputs.ndim)

    seq_len = inputs.shape[1]
    emb_size = inputs.shape[2]
    positional_encodings = get_positional_encodings(max_len, emb_size)
    positional_encodings = positional_encodings[:, :seq_len, :]
    return inputs + positional_encodings


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  def apply(self,
            inputs,
            mlp_dim,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(inputs, mlp_dim, kernel_init=kernel_init, bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x, actual_out_dim, kernel_init=kernel_init, bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Transformer1DBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            attention_fn=nn.dot_product_attention,
            cache=None):
    """Applies Transformer1DBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      attention_fn: dot product function to use inside attention.
      cache: Cache for decoding.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        attention_fn=attention_fn,
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


# TODO(levskaya): modify for 3 modes: train, eval and fast predict.
class TransformerLM(nn.Module):
  """Transformer Model for language modeling."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            causal=True,
            cache=None,
            positional_encoding_module=AddLearnedPositionalEncodings,
            attention_fn=nn.dot_product_attention,
            bos_token=None,
            output_head='logits'):
    """Applies Transformer model on the inputs.

    Args:
      inputs: An array of shape (batch_size, length) with the input sequences.
      vocab_size: An int with the size of the vocabulary.
      emb_dim: An int with the token embedding dimension.
      num_heads: An int with the number of attention heads.
      num_layers: An int with the number of transformer encoder layers.
      qkv_dim: An int with the dimension of the query/key/value vectors.
      mlp_dim: An int with the inner dimension of the feed-forward network which
        follows the attention block.
      max_len: An int with the maximum training sequence length.
      train: A bool denoting whether we are currently training.
      dropout_rate: A float with the dropout rate.
      attention_dropout_rate: A float with a dropout rate for attention weights.
      causal: Whether to apply causal masking.
      cache: Cache for decoding.
      positional_encoding_module: A module used for adding positional encodings.
      attention_fn: A callable to use in place of dot product attention.
      bos_token: An int token to shift with and ignore in loss.
      output_head: Output head of model to return. Used for embeddings.

    Returns:
      output of a transformer decoder.

    """
    assert inputs.ndim == 2  # (batch, len)
    if bos_token is None:
      raise ValueError('Must provide a bos_token.')

    # Mask out padding tokens.
    padding_mask = jnp.where(inputs != bos_token, 1,
                             0).astype(jnp.float32)[Ellipsis, None]

    heads = dict()
    ## TODO(ddohan): Add mode train/eval/predict
    x = shift_right(inputs, train=train, bos_token=bos_token)
    x = x.astype('int32')
    x = Embed(x, num_embeddings=vocab_size, features=emb_dim, name='embed')
    if positional_encoding_module == AddLearnedPositionalEncodings:
      x = positional_encoding_module(
          x,
          max_len=max_len,
          cache=cache,
          posemb_init=sinusoidal_init(max_len=max_len))
    else:
      x = positional_encoding_module(x, max_len=max_len)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    heads['input_emb'] = x
    for i in range(num_layers):
      x = Transformer1DBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=causal,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          attention_fn=attention_fn,
          cache=cache,
      )
      heads['layer_%s' % i] = x
    x = nn.LayerNorm(x)
    heads['output_emb'] = x
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    heads['logits'] = logits
    return heads[output_head]
