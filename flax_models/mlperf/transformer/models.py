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
"""Transformer-based machine translation model."""

from collections.abc import Iterable  # pylint: disable=g-importing-member
from flax import nn
from flax import struct
from flax.nn.attention import _invert_perm
from flax.nn.attention import _make_causal_mask
from flax.nn.attention import make_padding_mask
import jax
from jax import lax
from jax import random
from jax.interpreters.sharded_jit import PartitionSpec as P
from jax.interpreters.sharded_jit import with_sharding_constraint
import jax.numpy as jnp
import numpy as np


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


# Below we use a slightly modified version of the main Flax multi-headed
# attention layer that uses an optimized autoregressive Cache layout that
# has better TPU performance.


@struct.dataclass
class _CacheEntry:
  key: np.ndarray
  value: np.ndarray
  i: np.ndarray


class Cache(nn.base.Collection):
  """Collect intermediate activations for efficient autoregressive decoding."""

  def initialize_cache(self, shape):
    """Initialize the cache for the given input shape.

    Args:
      shape: the shape of the batch and attention dimensions.
    Returns:
      the initialized cache
    """
    def _init(shape_data):
      ndim = int(shape_data[0])
      tail_shape = tuple(shape_data[1:])
      full_shape = (shape[0],) + tail_shape + shape[1:]
      if len(full_shape) != ndim:
        raise ValueError('Shape should be a tuple with the shape of the batch'
                         'and attention dims.')
      return _CacheEntry(key=jnp.zeros(full_shape, jnp.bfloat16),
                         value=jnp.zeros(full_shape, jnp.bfloat16),
                         i=jnp.zeros((), jnp.uint32))
    return Cache(jax.tree_map(_init, self.state))


jax.tree_util.register_pytree_node(
    Cache, nn.base.iterate_collection, nn.base.collection_from_iterable)


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs. This version is modified to
  move the softmax division after the dot product.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  decoding = attn_weights.shape[-2] != 256
  if decoding:
    attn_weights = lax.exp(attn_weights - jax.scipy.special.logsumexp(
        attn_weights, axis=norm_dims, keepdims=True))
  else:
    # move the division by the softmax denominator to after the dot product
    attn_weights = jnp.exp(attn_weights - lax.stop_gradient(
        jnp.max(attn_weights, axis=norm_dims, keepdims=True)))
    softmax_denominator = jnp.sum(attn_weights, axis=norm_dims, keepdims=False)
  attn_weights = attn_weights.astype(dtype)

  # apply dropout
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      dropout_rng = nn.make_rng()
    keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = attn_weights.shape[-(2 * len(axis)):]
      dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (keep.astype(attn_weights.dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)
  if not decoding:
    # divide by the denominator of the attention softmax now, when the array is
    # O(N*H) rather than O(N^2)
    y = y / jnp.expand_dims(softmax_denominator, -1)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


class MultiHeadDotProductAttention(nn.base.Module):
  """Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=nn.initializers.zeros,
            bias=True,
            num_partitions=2):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token.
      key_padding_mask: boolean specifying key-value tokens that are pad token.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      num_partitions: number of ways to partition (i.e. how many devices to run
        across).

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))
    if num_partitions > 1:
      partitions = P(1, 1, num_partitions, 1)
      query = with_sharding_constraint(query, partitions)
      key = with_sharding_constraint(key, partitions)
      value = with_sharding_constraint(value, partitions)

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(np.array((key.ndim,) + key.shape[-2:], dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, _CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        i = cache_entry.i
        one_hot_indices = jax.nn.one_hot(
            i, cshape[3], dtype=key.dtype).reshape((1, 1, 1, cshape[3]))
        key = key.transpose((0, 2, 3, 1))
        key = cache_entry.key + key * one_hot_indices
        value = value.transpose((0, 2, 3, 1))
        value = cache_entry.value + value * one_hot_indices

        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

        key = key.transpose((0, 3, 1, 2))
        value = value.transpose((0, 3, 1, 2))
        cshape = (cshape[0], cshape[3], cshape[1], cshape[2])

        # TODO(levskaya): verify this is still needed in translation decoding.
        key_padding_mask = jnp.broadcast_to(
            (jnp.arange(cshape[1]) < cache_entry.i), cshape[:2])
        key_padding_mask = key_padding_mask.astype(jnp.float32)[Ellipsis, None]

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(np.take(key.shape, attention_axis))
        attn_size = np.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if padding_mask is not None:
      if key_padding_mask is None:
        key_padding_mask = padding_mask
      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = dot_product_attention(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')
    if num_partitions > 1:
      x = with_sharding_constraint(x, None)

    return out


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            min_timescale=1.0,
            max_timescale=10000.0,
            max_len=147,
            cache=None):
    """Adds positional embeddings to the inputs.

    Args:
      inputs: input data
      inputs_positions: input position indices for packed sequences.
      min_timescale: minimum scale that will be applied at each position
      max_timescale: maximum scale that will be applied at each position
      max_len: int: maximum length of sequence during eval.
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    channels = inputs.shape[2]
    num_timescales = channels // 2
    log_timescale_increment = (np.log(max_timescale / min_timescale) /
                               (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype('float32') * -log_timescale_increment)

    if inputs_positions is None:
      inputs_positions = np.expand_dims(np.arange(length).astype('float32'), 0)

    if cache:
      inputs_positions = np.expand_dims(np.arange(max_len).astype('float32'), 0)

    scaled_time = (jnp.expand_dims(inputs_positions.astype('float32'), 2) *
                   jnp.expand_dims(np.expand_dims(inv_timescales, 0), 0))
    signal = jnp.concatenate([jnp.sin(scaled_time),
                              jnp.cos(scaled_time)],
                             axis=2)
    signal = jnp.pad(signal, [[0, 0], [0, 0], [0, np.mod(channels, 2)]],
                     mode='constant', constant_values=inputs.dtype.type(0))

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
        _, _, df = signal.shape
        signal = lax.dynamic_slice(signal,
                                   jnp.array((0, i, 0)),
                                   jnp.array((1, 1, df)))
    if cache:
      # just needed to set correct shape on init.
      return inputs + signal[:, :1, :]
    else:
      return inputs + signal


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            num_partitions=2):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    inputs_shape = inputs.shape
    inputs = inputs.reshape((-1, inputs_shape[-1]))
    x = nn.Dense(inputs, mlp_dim, dtype=dtype, kernel_init=kernel_init,
                 bias_init=bias_init)
    x = nn.relu(x)
    if num_partitions > 1:
      x = with_sharding_constraint(x, P(1, num_partitions))
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(x, actual_out_dim, dtype=dtype, kernel_init=kernel_init,
                      bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    output = output.reshape(inputs_shape[:-1] + (actual_out_dim,))
    return output


class Encoder1DBlock(nn.Module):
  """Transformer decoder layer."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            num_partitions=2):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      num_partitions: number of ways to partition (i.e. how many devices to run
        across).

    Returns:
      output after transformer block.
    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs, dtype=dtype)
    x = MultiHeadDotProductAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        num_partitions=num_partitions)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        num_partitions=num_partitions)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer."""

  def apply(self,
            targets,
            encoded,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            targets_segmentation=None,
            padding_mask=None,
            key_padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            num_partitions=2):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax attention cache for fast decoding.
      num_partitions: number of ways to partition (i.e. how many devices
        to run across).

    Returns:
      output after transformer block.
    """

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(targets, dtype=dtype)
    x = MultiHeadDotProductAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        padding_mask=padding_mask,
        segmentation=targets_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        num_partitions=num_partitions)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = MultiHeadDotProductAttention(
        y,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=encoded,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=targets_segmentation,
        key_segmentation=inputs_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        num_partitions=num_partitions)
    y = nn.dropout(y, rate=dropout_rate, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(y, dtype=dtype)
    z = MlpBlock(
        z,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        num_partitions=num_partitions)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            num_partitions=2):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      num_partitions: number of ways to partition (i.e. how many devices
        to run across).

    Returns:
      output of a transformer encoder.

    """
    assert inputs.ndim == 2  # (batch, len)

    if use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Padding Masks
    src_padding_mask = (inputs > 0)[Ellipsis, None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=emb_dim**-0.5),
          dtype=dtype,
          num_partitions=num_partitions)()
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed[x] * jnp.sqrt(emb_dim)
    x = x.astype(dtype)
    x = AddPositionEmbs(x,
                        inputs_positions=inputs_positions,
                        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    # Input Encoder
    for lyr in range(num_layers):
      x = Encoder1DBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}',
          num_partitions=num_partitions)
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""

  def apply(self,
            encoded,
            src_padding_mask,
            targets,
            output_vocab_size,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            shared_embedding=None,
            logits_via_embedding=False,
            shift=True,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=True,
            cache=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            num_partitions=1):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      output_vocab_size: size of the vocabulary.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      shared_embedding: a shared embedding matrix to use.
      logits_via_embedding: bool: whether final logit transform shares
        embedding weights.
      shift: whether to shift or not (for fast decoding).
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      cache: flax attention cache for fast decoding.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      num_partitions: number of ways to partition (i.e. how many devices
        to run across).

    Returns:
      output of a transformer decoder.
    """
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    if use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[Ellipsis, None]

    # Target Embedding
    if shared_embedding is None:
      output_embed = Embed.shared(
          num_embeddings=output_vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=emb_dim**-0.5),
          dtype=dtype,
          num_partitions=num_partitions)()
    else:
      output_embed = shared_embedding

    y = targets.astype('int32')
    if shift:
      y = shift_right(y)
    y = output_embed[y] * jnp.sqrt(emb_dim)
    y = y.astype(dtype)
    y = AddPositionEmbs(y, inputs_positions=targets_positions, cache=cache,
                        name='posembed_targets')
    y = nn.dropout(y, rate=dropout_rate, deterministic=not train)

    # Target-Input Decoder
    for lyr in range(num_layers):
      y = EncoderDecoder1DBlock(
          y,
          encoded,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=tgt_padding_mask,
          key_padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          num_partitions=num_partitions,
          name=f'encoderdecoderblock_{lyr}')
    y = nn.LayerNorm(y, dtype=dtype, name='encoderdecoder_norm')
    y = y.reshape((-1, y.shape[-1]))

    # Decoded Logits
    if logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = lax.dot_general(
          y, output_embed, (((y.ndim - 1,), (1,)), ((), ())))
    else:
      logits = nn.Dense(
          y,
          output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')
    return logits


# The following final model is simple but looks verbose due to all the
# repetitive keyword argument plumbing.  It just sticks the Encoder and
# Decoder in series for training, but allows running them separately for
# inference.


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation."""

  def apply(self,
            inputs,
            targets,
            vocab_size=None,
            output_vocab_size=None,
            inputs_positions=None,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            share_embeddings=False,
            logits_via_embedding=False,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            num_partitions=2):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      targets: target data
      vocab_size: size of the input vocabulary
      output_vocab_size: size of the output vocabulary
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      share_embeddings: bool: share embedding layer for inputs and targets.
      logits_via_embedding: bool: whether final logit transform shares
        embedding weights.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      shift: whether to right-shift targets.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      cache: flax autoregressive cache for fast decoding.
      num_partitions: number of ways to partition (i.e. how many devices
        to run across).

    Returns:
      output of a transformer decoder.

    """
    if use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    src_padding_mask = (inputs > 0)[Ellipsis, None]

    if share_embeddings:
      if output_vocab_size is not None:
        assert output_vocab_size == vocab_size, (
            "can't share embedding with different vocab sizes.")
      shared_embedding = Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=emb_dim**-0.5),
          dtype=dtype,
          num_partitions=num_partitions)()
    else:
      shared_embedding = None

    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        num_partitions=num_partitions,
        name='encoder')

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        num_partitions=num_partitions,
        name='decoder')
    return logits.astype(dtype)

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while providing a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  @nn.module_method
  def encode(self,
             inputs,
             vocab_size=None,
             output_vocab_size=None,
             inputs_positions=None,
             inputs_segmentation=None,
             targets_positions=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None,
             num_partitions=2):
    del (output_vocab_size, shift, targets_positions,
         targets_segmentation, tgt_padding_mask, logits_via_embedding,
         cache)

    if use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    if share_embeddings:
      shared_embedding = Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=emb_dim**-0.5),
          dtype=dtype,
          num_partitions=num_partitions)()
    else:
      shared_embedding = None

    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        num_partitions=num_partitions,
        name='encoder')

    return encoded

  @nn.module_method
  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             inputs_positions=None,
             vocab_size=None,
             output_vocab_size=None,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None,
             num_partitions=2):
    del inputs_positions

    if use_bfloat16:
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    if share_embeddings:
      shared_embedding = Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=emb_dim**-0.5),
          dtype=dtype,
          num_partitions=num_partitions)()
    else:
      shared_embedding = None

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        shift=shift,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        num_partitions=num_partitions,
        name='decoder')

    return logits


# A copy of nn.Embed with added sharding constraints.
default_embed_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal',
                                                      out_axis=0)


class Embed(nn.base.Module):
  """Embedding Module with sharding constraints.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            num_embeddings,
            features,
            embedding_init=default_embed_init,
            dtype=jnp.float32,
            num_partitions=2):
    """Layer that returns an embedding matrix.

    Args:
      num_embeddings: number of embeddings.
      features: Number of feature dimensions for each embedding.
      embedding_init: embedding initializer.
      dtype: dtype to use for activations.
      num_partitions: number of ways to partition (i.e. how many devices to run
        across).

    Returns:
      An embedding matrix suitable for embedding[inputs].
    """
    embedding = self.param('embedding', (num_embeddings, features),
                           embedding_init)
    embedding = jnp.asarray(embedding, dtype)
    if num_partitions > 1:
      embedding = with_sharding_constraint(embedding, P(num_partitions, 1))
    return embedding
