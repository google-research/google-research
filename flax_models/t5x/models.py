# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""T5 Transformer model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.linen.module import compact
from flax.linen.module import Module
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

# Type Stubs
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


#------------------------------------------------------------------------------
# Fast nondeterministic hardware RNG for faster Dropout
#------------------------------------------------------------------------------
def hardware_bernoulli(rng_key,
                       p = np.float32(0.5),
                       shape = None):
  return lax.rng_uniform(lax.tie_in(rng_key, 0.0), 1.0, shape) < p


def set_hardware_bernoulli():
  jax.random.bernoulli = hardware_bernoulli


#------------------------------------------------------------------------------
# Adafactor-compatible DenseGeneral for attention layers.
#------------------------------------------------------------------------------
def _normalize_axes(axes, ndim):
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

    Kernel stored as 2d parameter for compatibility with Adafactor optimizer.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      use_bias: whether to add a bias to the output (default: False).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
  """
  features: int
  axis: int = -1
  use_bias: bool = False
  dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  precision: Any = None

  def setup(self):
    """Normalize hyperparameters."""
    if not isinstance(self.features, Iterable):
      self.features = (self.features,)
    if not isinstance(self.axis, Iterable):
      self.axis = (self.axis,)
    self.features = tuple(self.features)
    self.axis = tuple(self.axis)

  @nn.compact
  def __call__(self, inputs):
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(self.axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + self.features
    kernel_2d_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                       np.prod(self.features))
    kernel = self.param('kernel', self.kernel_init, kernel_2d_shape, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    out = lax.dot_general(
        inputs,
        kernel, ((axis, contract_ind), ((), ())),
        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (np.prod(self.features),),
                        self.dtype)
      bias = jnp.reshape(bias, self.features)
      # Reshape bias for broadcast.
      expand_dims = sorted(set(range(inputs.ndim)) - set(axis))
      for ax in expand_dims:
        bias = jnp.expand_dims(bias, ax)
      bias = jnp.asarray(bias, self.dtype)
      out = out + bias
    return out


#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------
class LayerNorm(Module):
  """T5 Layer normalization.

  Operates on the last axis of the input data.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    scale_init: Initializer for scale, by default, one.
  """
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param('scale', self.scale_init, (features,), self.dtype)
    return y * scale


#------------------------------------------------------------------------------
# Fast attention layers.
#------------------------------------------------------------------------------
def dot_product_attention(query,
                          key,
                          value,
                          bias = None,
                          broadcast_dropout = True,
                          rescale_logits = False,
                          dropout_rng = None,
                          dropout_rate = 0.,
                          deterministic = False,
                          dtype = jnp.float32,
                          precision = None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # T5 broadcasts along the "length" dim, but unclear which one that
      # corresponds to in positional dimensions here, assuming query dim.
      dropout_shape = list(attn_weights.shape)
      dropout_shape[-2] = 1
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      keep = jnp.broadcast_to(keep, attn_weights.shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # return weighted sum over values for each query position
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision)


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: bool = False
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  use_bias: bool = True
  rescale_logits: bool = False
  attention_fn: Callable[[Array, Array, Array],
                         Array] = staticmethod(dot_product_attention)
  decode: bool = False

  @compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               mask = None,
               bias = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = self.kernel_init
    else:
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    (query, key,
     value) = (dense(dtype=self.dtype, kernel_init=query_init,
                     name='query')(inputs_q),
               dense(
                   dtype=self.dtype, kernel_init=self.kernel_init,
                   name='key')(inputs_kv),
               dense(
                   dtype=self.dtype, kernel_init=self.kernel_init,
                   name='value')(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # NOTE: We store cached key, value with layout:
      # [batch..., heads, features, length] as a TPU fusion optimization
      # and to enable the "scatter via one-hot broadcast" trick.
      # This requires some transposes to and from this layout below.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, num_heads, depth_per_head, max_length = (
            cached_key.value.shape)
        # Sanity shape check of cached key against input query.
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # Update key, value caches with our new 1d spatial slices.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, max_length, dtype=key.dtype)
        key = jnp.moveaxis(key, -3, -1)
        value = jnp.moveaxis(value, -3, -1)
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + key * one_hot_indices
        value = cached_value.value + value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)
        # Causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(max_length) <= cur_index,
                tuple(batch_dims) + (1, 1, max_length)))
        # Grab the correct relative attention bias during decoding.
        if bias is not None:
          # equivalent to bias[..., cur_index:cur_index+1, :]
          bias = lax.dynamic_slice_in_dim(bias, cur_index, 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not self.deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        broadcast_dropout=self.broadcast_dropout,
        rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        dtype=self.dtype,
        precision=self.precision)

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(
            x)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(self,
               inputs_q,
               mask = None,
               bias = None):
    return super().__call__(inputs_q, inputs_q, mask, bias)


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input,
                        key_input,
                        pairwise_fn = jnp.multiply,
                        extra_batch_dims = 0,
                        dtype = jnp.float32):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  mask = pairwise_fn(
      jnp.expand_dims(query_input, axis=-1),
      jnp.expand_dims(key_input, axis=-2))
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x,
                     extra_batch_dims = 0,
                     dtype = jnp.float32):
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Args:
    x: input array of shape `[batch..., len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks, dtype = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


#------------------------------------------------------------------------------
# Transformer Model
#------------------------------------------------------------------------------
@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('relu',)
  max_len: int = 2048
  position_embeddings: str = 'relative'
  relative_attention_num_buckets: int = 32
  relative_attention_max_distance: int = 128
  rescale_logits: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  max_decode_len: int = 0
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_targets(x, segment_ids=None, axis=1):
  """Shift targets and replace EOS by 0 for packed targets."""
  # Remove T5 token masking "sign bit" for placeholders.
  x = jnp.abs(x)
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token (e.g. 1) for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted


class AddPositionEmbs(nn.Module):
  """Adds sinusoid or learned absolute positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether we're running in single-position autoregressive decode mode.
  """
  config: TransformerConfig
  decode: bool = False

  @staticmethod
  def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
      """Sinusoidal init."""
      del key, dtype
      d_feature = shape[-1]
      pe = np.zeros((max_len, d_feature), dtype=np.float32)
      position = np.arange(0, max_len)[:, np.newaxis]
      scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
      div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
      pe[:, :d_feature // 2] = np.sin(position * div_term)
      pe[:, d_feature // 2:2 * (d_feature // 2)] = np.cos(position * div_term)
      pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
      return jnp.array(pe)

    return init

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = self.sinusoidal_init(max_len=cfg.max_len)(None,
                                                                pos_emb_shape,
                                                                None)
    else:
      pos_embedding = self.param('pos_embedding', cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class RelativePositionEmbs(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    embedding_init: initializer for relative embedding table.
  """
  config: TransformerConfig
  embedding_init: Callable = nn.linear.default_embed_init

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(np.int32) * num_buckets
      n = np.abs(n)
    else:
      n = np.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret

  @nn.compact
  def __call__(self, qlen, klen, bidirectional=True):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
    cfg = self.config
    # TODO(levskaya): should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=cfg.relative_attention_num_buckets,
        max_distance=cfg.relative_attention_max_distance)
    relative_attention_bias = self.param(
        'rel_embedding', self.embedding_init,
        (cfg.num_heads, cfg.relative_attention_num_buckets), cfg.dtype)
    relative_attention_bias = relative_attention_bias.astype(cfg.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(
        jnp.int32, (cfg.relative_attention_num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota, dtype=cfg.dtype)
    # --> shape (qlen, klen, num_heads)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    return values[jnp.newaxis, Ellipsis]


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)
    inputs_shape = inputs.shape
    inputs = inputs.reshape((-1, inputs_shape[-1]))
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('linear', 'gelu') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(cfg.mlp_activations):
      dense_name = 'wi' if len(cfg.mlp_activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
          cfg.mlp_dim,
          dtype=cfg.dtype,
          kernel_init=cfg.kernel_init,
          use_bias=False,
          bias_init=cfg.bias_init,
          name=dense_name)(
              inputs)
      if act_fn != 'linear':
        x = getattr(nn, act_fn)(x)
      activations.append(x)
    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic,
        broadcast_dims=(-2,))  # broadcast along length
    output = DenseGeneral(
        actual_out_dim,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        use_bias=False,
        bias_init=cfg.bias_init,
        name='wo')(
            x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic, broadcast_dims=(-2,))
    output = output.reshape(inputs_shape[:-1] + (actual_out_dim,))
    return output


class Encoder1DBlock(nn.Module):
  """Transformer decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig
  relative_embedding: Optional[nn.Module] = None

  @nn.compact
  def __call__(self, inputs, encoder_mask=None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Shared relative position embedding attention biases.
    if self.relative_embedding:
      encoder_bias = self.relative_embedding(  # pylint: disable=not-callable
          inputs.shape[-2], inputs.shape[-2], True)
    else:
      encoder_bias = None

    # Attention block.
    assert inputs.ndim == 3
    x = LayerNorm(dtype=cfg.dtype)(inputs)
    x = SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=True,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(x, encoder_mask, encoder_bias)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic, broadcast_dims=(-2,))
    x = x + inputs

    # MLP block.
    y = LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y)
    y = y + x

    return y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig
  relative_embedding: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder. If None, block is Decoder only.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Shared relative position embedding attention biases.
    if self.relative_embedding:
      if cfg.decode:
        decoder_bias = self.relative_embedding(  # pylint: disable=not-callable
            cfg.max_decode_len, cfg.max_decode_len, False)
      else:
        decoder_bias = self.relative_embedding(  # pylint: disable=not-callable
            targets.shape[-2], targets.shape[-2], False)
    else:
      decoder_bias = None
    # No relative embeddings are used for encoder-decoder cross attention.
    encoder_decoder_bias = None

    # Decoder block.
    assert targets.ndim == 3
    x = LayerNorm(dtype=cfg.dtype)(targets)
    x = SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=True,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode)(x, decoder_mask, decoder_bias)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic, broadcast_dims=(-2,))
    x = x + targets

    # Encoder-Decoder block.
    if encoded is None:
      # If encoder outputs not provides, skip attending from decoder to encoder.
      # This results in a decoder only block.
      y = x
    else:
      y = LayerNorm(dtype=cfg.dtype)(x)
      y = MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          broadcast_dropout=True,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic)(y, encoded, encoder_decoder_mask,
                                           encoder_decoder_bias)

      y = nn.Dropout(rate=cfg.dropout_rate)(
          y, deterministic=cfg.deterministic, broadcast_dims=(-2,))
      y = y + x

    # MLP block.
    z = LayerNorm(dtype=cfg.dtype)(y)
    z = MlpBlock(config=cfg)(z)
    z = z + y

    return z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Optional[nn.Module] = None

  @nn.compact
  def __call__(self, inputs, inputs_positions=None, encoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config
    assert inputs.ndim == 2  # (batch, len)

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding

    # Shared relative position embedding attention biases.
    if cfg.position_embeddings == 'relative':
      relative_embedding = RelativePositionEmbs(
          cfg, name='encoder_relative_posemb')
    else:
      relative_embedding = None

    x = inputs.astype('int32')
    x = input_embed(x)
    if cfg.position_embeddings == 'absolute':
      x = AddPositionEmbs(
          config=cfg, decode=False, name='posembed_input')(
              x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic, broadcast_dims=(-2,))

    x = x.astype(cfg.dtype)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(
          config=cfg,
          relative_embedding=relative_embedding,
          name=f'encoderblock_{lyr}')(x, encoder_mask)

    encoded = LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    encoded = nn.Dropout(rate=cfg.dropout_rate)(
        encoded, deterministic=cfg.deterministic)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               targets_positions=None,
               targets_segmentation=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder. If None, do not attend to
        encoded inputs, resulting in a decoder only model (i.e. language model).
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    # If encoded is not given, this block is decoder only and does not contain
    # attention from decoder to encoder.
    if encoded is not None:
      assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=cfg.output_vocab_size,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    # Shared relative position embedding attention biases.
    if cfg.position_embeddings == 'relative':
      relative_embedding = RelativePositionEmbs(
          cfg, name='decoder_relative_posemb')
    else:
      relative_embedding = None

    y = targets.astype('int32')
    if not cfg.decode:
      y = shift_targets(y, segment_ids=targets_segmentation)
    y = output_embed(y)
    if cfg.position_embeddings == 'absolute':
      y = AddPositionEmbs(
          config=cfg, decode=cfg.decode, name='posembed_output')(
              y, inputs_positions=targets_positions)
    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic, broadcast_dims=(-2,))

    y = y.astype(cfg.dtype)

    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg,
          relative_embedding=relative_embedding,
          name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask)
    y = LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)
    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic, broadcast_dims=(-2,))

    # We flatten batch-dims+sequence-dim for a small performance gain here.
    # This requires using flattened targets and weights when computing cross-
    # entropy and accuracy.  (If this proves annoying it can be removed later
    # with only a small loss in performance.)
    y = y.reshape((-1, y.shape[-1]))

    # Decoded Logits
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = DenseGeneral(
          cfg.output_vocab_size,
          dtype=cfg.dtype,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          use_bias=False,
          name='logits_dense')(
              y)
    return logits


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def setup(self):
    cfg = self.config

    if cfg.share_embeddings:
      if cfg.output_vocab_size is not None:
        assert cfg.output_vocab_size == cfg.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      self.shared_embedding = None

    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

  def encode(self, inputs, inputs_positions=None, inputs_segmentation=None):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      encoded feature array from the transformer encoder.
    """
    cfg = self.config
    # Make padding attention mask.
    encoder_mask = make_attention_mask(inputs > 0, inputs > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if inputs_segmentation is not None:
      encoder_mask = combine_masks(
          encoder_mask,
          make_attention_mask(
              inputs_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
    return self.encoder(  # pytype: disable=attribute-error
        inputs,
        inputs_positions=inputs_positions,
        encoder_mask=encoder_mask)

  def decode(
      self,
      encoded,
      inputs,  # only needed for masks
      targets,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.config

    # Make padding attention masks.
    if cfg.decode:
      # fast autoregressive decoding uses only a special encoder-decoder mask
      decoder_mask = None
      encoder_decoder_mask = make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0, dtype=cfg.dtype)
    else:
      decoder_mask = combine_masks(
          make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          make_causal_mask(targets, dtype=cfg.dtype))
      encoder_decoder_mask = make_attention_mask(
          targets > 0, inputs > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = combine_masks(
          decoder_mask,
          make_attention_mask(
              targets_segmentation,
              targets_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
      encoder_decoder_mask = combine_masks(
          encoder_decoder_mask,
          make_attention_mask(
              targets_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
    logits = self.decoder(  # pytype: disable=attribute-error
        encoded,
        targets,
        targets_positions=targets_positions,
        targets_segmentation=targets_segmentation,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask)
    return logits.astype(self.config.dtype)

  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    return self.decode(
        encoded,
        inputs,  # only used for masks
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation)
