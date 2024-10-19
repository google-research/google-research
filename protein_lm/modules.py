# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Flax Modules."""
from flax.deprecated import nn
from jax import lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def shift_right(x, bos_token):
  """Shift the input to the right by padding on axis 1 at train time."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(
      x,
      pad_widths,
      mode='constant',
      constant_values=jnp.asarray(bos_token, dtype=x.dtype))
  return padded[:, :-1]


def mask_uniform(inputs, rate, rng, mask_value):
  """Applies a random dropout mask to the input.

  Args:
    inputs: the inputs that should be randomly masked.
    rate: the probablity of masking out a value.
    rng: an optional `jax.random.PRNGKey`. By default `nn.make_rng()` will be
      used.
    mask_value: Value to mask with.

  Returns:
    The masked inputs.
  """
  if rate == 0.:
    return inputs
  keep_prob = 1. - rate
  mask = jrandom.bernoulli(rng, p=keep_prob, shape=inputs.shape)
  return lax.select(mask, inputs, jnp.full_like(inputs, mask_value))


class Tag(nn.Module):
  """Save a value to global state when running in stateful mode."""

  def apply(self, x):
    if self.is_stateful():
      tagged = self.state('tag')
      tagged.value = x
    return x


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            num_features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies the Embed module.

    Args:
      inputs: An array of shape (batch_size, length) or (batch_size, length,
        vocab_size) with the input sequences. When 2-dimensional, the array
        contains sequences of int tokens. Otherwise, the array contains
        next-token distributions over tokens (e.g. one-hot representations).
      num_embeddings: An int with the number of embeddings.
      num_features: An int with the size of the embedding dimension.
      mode: A string, 'input' or 'output' -> to share input/output embeddings.
      emb_init: A callable, the embedding initializer function.

    Returns:
      An array of shape (batch_size, length, num_features) with embedded data.
    """
    if inputs.ndim != 2 and inputs.ndim != 3:
      raise ValueError('Expected 2 or 3 dimensions, found %d.' % inputs.ndim)

    embedding = self.param('embedding', (num_embeddings, num_features),
                           emb_init)
    if mode == 'input':
      if inputs.ndim == 2:  # Inputs are lists of integers.
        if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
          raise ValueError('Input type must be an integer or unsigned integer.')
        return jnp.take(embedding, inputs, axis=0)

      # Inputs contain per-token probabilities.
      if inputs.shape[2] != num_embeddings:
        raise ValueError('Expected shape (..., %d), found (..., %d)' %
                         (num_embeddings, inputs.shape[2]))
      batch_size, length, _ = tuple(inputs.shape)

      # Tile embeddings to (batch_size, length, num_features, num_embeddings).
      emb = jnp.transpose(embedding)
      tiled_emb = jnp.tile(emb[None, None, Ellipsis], [batch_size, length, 1, 1])

      # Accumulate embeddings proportional to token probabilities.
      accum_emb = jnp.matmul(tiled_emb, inputs[Ellipsis, None])
      return accum_emb[Ellipsis, 0]
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
                               (1, 1, df))
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
            self_attention_module=nn.SelfAttention,
            attention_fn=None,
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
      self_attention_module: Self attention module.
      attention_fn: dot product function to use inside attention.
      cache: Cache for decoding.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    if attention_fn is not None:
      self_attention_module = self_attention_module.partial(
          attention_fn=attention_fn)
    x = self_attention_module(
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
class Transformer(nn.Module):
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
            self_attention_module=nn.SelfAttention,
            attention_fn=None,
            pad_token=None,
            output_head='logits'):
    """Applies Transformer model on the inputs.

    Args:
      inputs: An array of shape (batch_size, length) or (batch_size, length,
        vocab_size) with the input sequences. When 2-dimensional, the array
        contains sequences of int tokens. Otherwise, the array contains
        next-token distributions over tokens (e.g. one-hot representations).
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
      self_attention_module: Self attention module.
      attention_fn: Method to use in place of dot product attention.
      pad_token: Token to ignore in attention.
      output_head: String or iterable over strings containing the model's output
        head(s) to return.

    Returns:
      Output of a transformer decoder. If output_head is a string, we return a
        single output head output; if output_head is an iterable, we return a
        dict with (output head name, output head output) key-value pairs.
    """
    if inputs.ndim != 2 and inputs.ndim != 3:
      raise ValueError('Expected 2 or 3 dimensions, found %d.' % inputs.ndim)

    if inputs.ndim == 3:
      padding_mask = jnp.ones_like(inputs[Ellipsis, 0])
    elif pad_token is None:
      padding_mask = jnp.ones_like(inputs)
    else:
      # Mask out padding tokens.
      padding_mask = jnp.where(inputs != pad_token, 1, 0).astype(jnp.float32)
    padding_mask = padding_mask[Ellipsis, None]  # Add embedding dimension.

    heads = dict()
    x = inputs
    if inputs.ndim == 2:
      x = x.astype('int32')
    x = Embed(x, num_embeddings=vocab_size, num_features=emb_dim, name='embed')

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
          self_attention_module=self_attention_module,
          deterministic=not train,
          attention_fn=attention_fn,
          cache=cache,
      )
      heads['layer_%s' % i] = x
    x = nn.LayerNorm(x)
    heads['output_emb'] = x * padding_mask  # Zero out PAD positions.
    if 'logits' in output_head:
      logits = nn.Dense(
          x,
          vocab_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      heads['logits'] = logits

    if 'regression' in output_head:
      regression = nn.Dense(
          x,
          1,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      regression = jnp.squeeze(regression, axis=-1)
      heads['regression'] = regression

    if isinstance(output_head, (tuple, list)):
      return {head: heads[head] for head in output_head}
    return heads[output_head]
