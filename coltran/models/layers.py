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

"""Various base layers for the colorization transformer."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import math
import operator
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from coltran.utils import att_utils
from coltran.utils import base_utils

# pylint: disable=duplicate-string-formatting-argument


def residual_dropout(inputs, output, dropout, training):
  """out = inputs + dropout(output)."""
  if training and dropout:
    output = tf.nn.dropout(output, dropout)
  output += inputs
  return output


class Shift(layers.Layer):
  """Shifts an input tensor either down or right to preserve causal ordering."""

  def __init__(self, dimension, resolution, **kwargs):
    """Init.

    Args:
      dimension: int, 0 to shift down, 1 to shift right.
      resolution: list of 2 ints, [H, W].
      **kwargs:
    """
    super(Shift, self).__init__(**kwargs)
    self.dimension = dimension
    self.resolution = resolution

  def call(self, x):
    shape = x.shape
    rank = len(shape)
    dim = self.dimension + 1

    # Assume 1 batch_dim.
    index = [0] * len(self.resolution)
    y = x
    paddings = np.zeros((rank, 2), dtype=np.int32)
    paddings[dim, 0] = 1
    y = tf.pad(y, paddings)

    rem_dims = rank - 1 - len(index[:dim])
    slice_inds = [0]  + index[:dim] + [0] * rem_dims
    return tf.slice(y, slice_inds, shape)


class Cache(layers.Layer):
  """Keras layer for cacheing.

  Values are cached in a tensor of shape (B, canvas_shape, D).
  B and D are inferred from the inputs to the call method.

  Every call to the cache instance is assumed to be a tuple of (index, values).
  It updates the cache such that cache[:, index:, :] = values
  """

  def __init__(self, canvas_shape,
               num_batch_axes=1,
               dtype=tf.float32,
               **kwargs):
    super(Cache, self).__init__(trainable=False, **kwargs)
    self.canvas_shape = canvas_shape
    self.num_batch_axes = num_batch_axes
    self._dtype = dtype

  def build(self, input_shapes):
    num_canvas_dim = len(self.canvas_shape)
    value, _ = input_shapes
    features_shape = value[self.num_batch_axes + num_canvas_dim:]
    cache_shape = (value[:self.num_batch_axes] + self.canvas_shape +
                   features_shape)
    self.cache = tf.zeros(shape=cache_shape, dtype=self._dtype)
    super(Cache, self).build(input_shapes)

  def reset(self):
    self.cache = tf.zeros(shape=self.cache.shape, dtype=self._dtype)

  def call(self, inputs):
    value, index = inputs
    if self.cache.shape == inputs[0].shape:
      self.cache = value
      return value

    shape = self.cache.shape.as_list()
    num_index_axes = index.shape[0]
    num_batch_axes = self.num_batch_axes
    num_feature_axes = len(shape) - num_index_axes - num_batch_axes
    features_shape = shape[num_batch_axes + num_index_axes:]
    batch_shape = shape[:num_batch_axes]

    value_index_shape = tf.shape(value)[num_batch_axes:-num_feature_axes]
    if tf.reduce_max(value_index_shape) > 1:
      # This is a block update starting at index.
      value_ranges = []
      for i, s in enumerate(tf.unstack(value_index_shape)):
        curr_range = tf.range(index[i], index[i] + s)
        value_ranges.append(curr_range)

      batch_ranges = [tf.range(s) for s in batch_shape]

      mesh = tf.meshgrid(*(batch_ranges + value_ranges), indexing='ij')
      indices = tf.stack(mesh, axis=-1)
      indices = tf.reshape(indices, [-1, num_index_axes + num_batch_axes])
    else:
      # This is a single update at index position.
      batch_ranges = [tf.range(s) for s in batch_shape]
      mesh = tf.meshgrid(*batch_ranges, indexing='ij')
      batch_indices = tf.stack(mesh, axis=-1)
      batch_indices = tf.reshape(batch_indices, [-1, num_batch_axes])

      # Add leading axes to nd-index and tile to get batched indices.
      shape_indices = tf.reshape(index, [1] * num_batch_axes + [-1])
      shape_indices = tf.tile(shape_indices, batch_shape + [1])
      shape_indices = tf.reshape(shape_indices, [-1, num_index_axes])

      indices = tf.concat([batch_indices, shape_indices], axis=-1)

    # We need to squeeze nd-axes from value before updating.
    value = tf.reshape(value, [-1] + features_shape)
    self.cache = tf.tensor_scatter_nd_update(self.cache, indices, value)
    return self.cache


class Masking(object):
  """Masking options for self-attention.

  We can either mask the entire future, i.e. allow looking into the past and
  the current element, or we can mask in addition the present as well, i.e.,
  we can look only to the past.
  """

  FUTURE = 'future'
  FUTURE_PRESENT = 'future_present'


class PositionEmbed(layers.Layer):
  """Adds factorized positional embeddings for specified axes."""

  def __init__(self, axes, max_lengths=None, **kwargs):
    """Init.

    Args:
      axes: list of ints, axis over which to apply the positional embeddings.
      max_lengths: list of ints, maximum length over each axis.
      **kwargs:
    """
    super(PositionEmbed, self).__init__(**kwargs)
    if not isinstance(axes, (list, tuple)):
      axes = [axes]
    self.axes = axes
    self.max_lengths = None
    if max_lengths:
      if not isinstance(max_lengths, (list, tuple)):
        max_lengths = [max_lengths]
      self.max_lengths = max_lengths

  def build(self, input_shape):
    rank = len(input_shape)
    self.axes = sorted([rank + a if a < 0 else a for a in self.axes])
    self.max_lengths = self.max_lengths or [input_shape[a] for a in self.axes]
    self.embeddings = []
    for i, axis in enumerate(self.axes):
      shape = [self.max_lengths[i]] + [1] * (rank - axis - 2)
      shape.append(input_shape[-1])
      init = tf.keras.initializers.RandomNormal(stddev=shape[-1]**-0.5)
      self.embeddings.append(
          self.add_weight(
              name='position_embedding_%d' % i,
              shape=shape,
              initializer=init,
              trainable=True))
    super(PositionEmbed, self).build(input_shape)

  def call(self, inputs):
    out = inputs
    for e in self.embeddings:
      out += e
    return out


class DenseND(layers.Layer):
  """Maps a rank-m tensor to a rank-n tensor through a dense contraction."""

  def __init__(self,
               filters,
               contract_axes=1,
               use_bias=False,
               activation=None,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(DenseND, self).__init__(**kwargs)
    if isinstance(filters, int):
      filters = [filters]
    self.filters = tuple(filters)
    self.contract_axes = contract_axes
    self.use_bias = use_bias
    self.activation = tf.keras.activations.get(activation)
    self.bias_initializer = bias_initializer
    self._kernel_initializer = kernel_initializer

    # Behaviours differ when shape(weights) > 2.
    # see: https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/init_ops_v2.py#L733 pylint: disable=line-too-long
    if self._kernel_initializer == 'glorot_uniform_nd':
      self._kernel_initializer = self._glorot_uniform

  def _num_batch_axes(self, input_shape):
    """Returns number of batch axes in inputs."""
    return len(input_shape) - len(self.contract_shape)

  def _glorot_uniform(self, shape, dtype=tf.float32):
    """Glorot uniform initializer."""
    fan_out = functools.reduce(operator.mul, self.filters)
    fan_in = functools.reduce(operator.mul, shape[:self.contract_axes])
    scale = 1. / max(1., (fan_in + fan_out) / 2.)
    limit = math.sqrt(3.0 * scale)
    return tf.random.uniform(shape, -limit, limit, dtype)

  def build(self, input_shape):
    # Infer matrix multiplication if no contract shape specified.
    self.contract_shape = input_shape[-self.contract_axes:]
    w_shape = self.contract_shape + self.filters
    self.w = self.add_weight(
        name='kernel',
        shape=w_shape,
        initializer=self._kernel_initializer,
        trainable=True)
    if self.use_bias:
      self.b = self.add_weight(
          name='bias', shape=self.filters, initializer=self.bias_initializer,
          trainable=True)
    super(DenseND, self).build(input_shape)

  def call(self, inputs):
    # Workaround lack of ellipsis support.
    # pyformat: disable
    num_batch_axes = self._num_batch_axes(inputs.shape)
    batch_str = 'abcdefghijklm'[:num_batch_axes]
    contract_str = 'ABCDEFGHIJKLM'[:len(self.contract_shape)]
    output_str = 'nopqrstuvwxyz'[:len(self.filters)]
    # pyformat: enable
    einsum_str = '{}{},{}{}->{}{}'.format(batch_str, contract_str, contract_str,
                                          output_str, batch_str, output_str)
    result = tf.einsum(einsum_str, inputs, self.w)
    if self.use_bias:
      result += self.b
    if self.activation is not None:
      result = self.activation(result)
    return result


class RelativeAttentionBiasND(layers.Layer):
  """Relative attention bias in nd factorizes over dimensions."""

  def __init__(self, lengths, num_heads, **kwargs):
    self.num_heads = num_heads
    self.lengths = lengths
    super(RelativeAttentionBiasND, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.biases = []
    self.total_length = 1
    for i, l in enumerate(self.lengths):
      self.total_length *= l
      if l > 1:
        weight = self.add_weight(
            name='relative_attention_bias_%d' % i,
            shape=[self.num_heads, 2 * l],
            initializer=tf.keras.initializers.Zeros(), trainable=True)
      else:
        weight = None
      self.biases.append(weight)

    super(RelativeAttentionBiasND, self).build(input_shapes)

  def call(self, inputs=None):
    tile, index, biases = 1, None, []
    len_q = self.total_length

    for i, s in enumerate(self.lengths):
      # Relative attention in every dimension separately.
      if s > 1:
        new_bias = att_utils.relative_attn_bias(
            self.biases[i], self.num_heads, index)
        repeat = self.total_length // (tile * s)
        if repeat > 1:
          new_bias = tf.expand_dims(new_bias, -1)
          new_bias = tf.tile(new_bias, [tile, repeat, tile, repeat])
          new_bias = tf.reshape(new_bias,
                                [len_q, self.num_heads, self.total_length])
        elif tile > 1:
          new_bias = tf.tile(new_bias, [tile, 1, tile])
        tile *= s
        biases.append(new_bias)

    return tf.add_n(biases)


class ConditionalLayerNorm(layers.Layer):
  """Conditional Layer Norm.

  Normalization of the input with the scale and shift as a function of 3-D
  context. Transforms 3-D spatial context into 1-D shift and scale of the
  layer-norm parameters. This is done via two dense projections:
    1. Spatial averaging via spatial_average='mean' or 'learnable'.
    2. Pointwise dense projection across channels.
  """

  def __init__(self,
               spatial_average='learnable',
               sequence='sc',
               out_init='glorot_uniform',
               out_act='identity', **kwargs):
    super(ConditionalLayerNorm, self).__init__(**kwargs)
    self.spatial_average = spatial_average
    self.sequence = sequence
    self.out_init = out_init
    self.out_act = out_act
    self.out_act_func = base_utils.act_to_func(out_act)
    if self.spatial_average not in ['mean', 'learnable']:
      raise ValueError('Expected spatial average to be "mean" or "learnable" ,'
                       'got %s' % self.spatial_average)
    if self.sequence not in ['sc', 'cs']:
      raise ValueError('Expected sequence to be "sc" or "cs" ,'
                       'got %s' % self.sequence)

  def build(self, input_shape):
    x_shape = input_shape[0]
    height, width, features = x_shape[-3:]
    self.layer_norm = layers.LayerNormalization(
        trainable=False, name='normalize')

    if self.spatial_average == 'learnable':
      self.spatial_weights = self.add_weight(
          name='spatial_average', shape=(1, height, width, 1),
          initializer=tf.keras.initializers.Ones())
    self.channel_dense = layers.Dense(
        units=2*features, kernel_initializer=self.out_init)
    super(ConditionalLayerNorm, self).build(input_shape)

  def spatial_projection(self, cond_inputs):
    if self.spatial_average == 'learnable':
      cond_inputs = self.spatial_weights * cond_inputs
    return tf.reduce_mean(cond_inputs, axis=(1, 2), keepdims=True)

  def call(self, inputs):
    inputs, cond_inputs = inputs

    if self.sequence == 'sc':
      ops = [self.spatial_projection, self.channel_dense]
    elif self.sequence == 'cs':
      ops = [self.channel_dense, self.spatial_projection]

    for op in ops:
      cond_inputs = op(cond_inputs)

    scale, shift = tf.split(cond_inputs, num_or_size_splits=2, axis=-1)
    scale = self.out_act_func(scale)
    shift = self.out_act_func(shift)
    inputs_norm = self.layer_norm(inputs)
    inputs_norm *= scale
    inputs_norm += shift
    return inputs_norm


class SelfAttentionND(layers.Layer):
  """Transforms input through a N-D self-attention layer.

  Assume key, query and memory tensors are N-D tensors.

  1. Project key, query and value tensors into (N+2)-D tensors using
     dense layers where the outer two dimensions are
     [num_heads, num_channels_per_head].
     num_channels_per_head is set to num_channels // num_heads by default.
  2. Computes self-attention tensor using 2 dot products.
     The first computes similarity between the key and query tensors.
     The second uses this similarity to perform a weighted average over
     the value tensors. Done in _dot_product and _weighted_sum.
  3. The default behaviour, i.e if nd_block is not set, is to do global
     self attention. If nd_block_set is set, the above self-attention is limited
     to a block-size of nd_block_size.
     For instance, in case of 2D inputs (images), setting nd_block_size to
     [1, num_columns] or [num_rows, 1] to limit attention to column
     and rows respectively.
  4. If mask=='future', zero out the contribution of the values that
     violate raster ordering. Done in _apply_mask_and_bias
     for more details.
  5. Project the transformed tensor into hidden_size number of channels
     using a dense layer.

  Self-attention can be optionally conditioned with an tuple of two values
  where the second argument is the conditional input. Supports:
  1. Biasing: By setting cond_q, cond_k or cond_v to be True.
  2. Scaling: By setting cond_scale to be True.
  """

  def __init__(self,
               hidden_size,
               num_heads=1,
               num_channels_per_head=None,
               mask=None,
               kernel_initializer='glorot_uniform',
               nd_block_size=None,
               resolution=None,
               cond_init='glorot_uniform',
               cond_k=False,
               cond_q=False,
               cond_v=False,
               cond_scale=False,
               cond_act='identity',
               **kwargs):
    super(SelfAttentionND, self).__init__(**kwargs)
    if nd_block_size:
      nd_block_size = list(nd_block_size)
    num_channels_per_head = num_channels_per_head or hidden_size // num_heads
    self.num_filters = [num_heads, num_channels_per_head]
    self.kernel_initializer = kernel_initializer
    self.hidden_size = hidden_size
    self.cond_k = cond_k
    self.cond_q = cond_q
    self.cond_v = cond_v
    self.cond_scale = cond_scale
    self.cond_init = cond_init
    self.cond_act_func = base_utils.act_to_func(cond_act)
    self.project_cond_q, self.project_cond_k, self.project_cond_v = None, None, None
    self.cond_filters = self.num_filters
    if cond_scale:
      self.cond_filters = [num_heads, 2*num_channels_per_head]

    self.nd_block_size = nd_block_size
    self.resolution = resolution
    self.mask = mask
    self.num_channels_per_head = num_channels_per_head
    self.num_heads = num_heads
    self.hidden_size = hidden_size

    # By default, apply attention in third last dimension.
    # Last 2 dimensions are heads, channels.
    self.attention_dim_q = self.attention_dim_k = -3

    # Self attention type.
    self.is_block_attention = True if self.nd_block_size else False

  def get_num_filters(self, is_cond):
    if not is_cond:
      return self.num_filters
    num_heads, num_channels_per_head = self.num_filters
    return [num_heads, 2*num_channels_per_head]

  def cond_shift_and_scale(self, inputs, cond_inputs, is_cond, layer):
    if not is_cond:
      return inputs
    cond_out = layer(cond_inputs)
    if self.cond_scale:
      scale, shift = tf.split(cond_out, num_or_size_splits=2, axis=-1)
      scale = self.cond_act_func(scale)
      shift = self.cond_act_func(shift)
      inputs *= scale
      inputs += shift
    else:
      inputs += cond_out
    return inputs

  def build(self, input_shape):
    if not isinstance(input_shape[-1], int):
      input_shape = input_shape[0]
    lengths = self.nd_block_size or self.resolution or input_shape[1:-1]

    self.project_q = DenseND(
        self.num_filters, kernel_initializer=self.kernel_initializer, name='q')
    self.project_k = DenseND(
        self.num_filters, kernel_initializer=self.kernel_initializer, name='k')
    self.project_v = DenseND(
        self.num_filters, kernel_initializer=self.kernel_initializer, name='v')
    self.project_final = DenseND(
        self.hidden_size, kernel_initializer=self.kernel_initializer,
        contract_axes=2, name='output')

    self.relative_attention = RelativeAttentionBiasND(
        lengths, self.num_heads)
    self.relative_attention.build([])

    if self.cond_k:
      self.project_cond_k = DenseND(
          self.cond_filters, kernel_initializer=self.cond_init, name='cond_k')
    if self.cond_q:
      self.project_cond_q = DenseND(
          self.cond_filters, kernel_initializer=self.cond_init, name='cond_q')
    if self.cond_v:
      self.project_cond_v = DenseND(
          self.cond_filters, kernel_initializer=self.cond_init, name='cond_v')

    self.is_one_dim_attention = (
        self.is_block_attention and
        sum(s != 1 for s in self.nd_block_size) == 1)
    if self.is_one_dim_attention:
      max_dim = self.nd_block_size.index(max(self.nd_block_size))
      if self.nd_block_size[max_dim] == lengths[max_dim]:
        self.is_block_attention = False
        self.attention_dim_q = max_dim - len(self.nd_block_size) - 2
        self.attention_dim_k = self.attention_dim_q
      else:
        self.is_one_dim_attention = False

    if self.mask:
      total_length = functools.reduce(operator.mul, lengths, 1)
      self._mask = np.triu(np.ones([total_length, total_length], np.float32))
      if self.mask != Masking.FUTURE_PRESENT:
        self._mask *= (1.0 - np.eye(total_length))
      self._mask *= -1e6
      self._mask = tf.constant(
          np.reshape(self._mask, [total_length, 1, total_length]))

    super(SelfAttentionND, self).build(input_shape)

  def _apply_mask_and_bias(self, alphas):
    bias = self.relative_attention(None)
    if self.mask:
      bias += self._mask

    expand_bias_dims = -self.attention_dim_q - 3
    if expand_bias_dims:
      bias = tf.reshape(bias, [-1] + [1] * expand_bias_dims +
                        list(bias.shape[1:]))
    return alphas + bias

  def _dot_product(self, q, k, contract_dim_q=-3, contract_dim_k=-3):
    num_batch_axes = len(q.shape) + contract_dim_q
    pre_str = 'abcdefghij' [:num_batch_axes]
    in_dim_q = -contract_dim_q - 2
    in_dim_k = -contract_dim_k - 2

    in_str_q = 'zyxwv' [:in_dim_q]
    in_str_k = 'zyxwv' [:in_dim_k]
    einsum_str = '{}Q{}C,{}M{}C->{}Q{}M'.format(pre_str, in_str_q, pre_str,
                                                in_str_k, pre_str, in_str_q)
    return tf.einsum(einsum_str, q, k)

  def _weighted_sum(self, alphas, v, contract_dim_a=-3, contract_dim_v=-3):
    num_batch_axes = len(alphas.shape) + contract_dim_a
    pre_str = 'abcdefghij' [:num_batch_axes]
    in_dim_a = -contract_dim_a - 2
    in_dim_v = -contract_dim_v - 2
    in_str_a = 'zyxwv' [:in_dim_a]
    in_str_v = 'zyxwv' [:in_dim_v]
    einsum_str = '{}Q{}M,{}M{}C->{}Q{}C'.format(pre_str, in_str_a, pre_str,
                                                in_str_v, pre_str, in_str_a)
    return tf.einsum(einsum_str, alphas, v)

  def _prepare_block_attention(self, x):
    return att_utils.divide_nd_blocks(x, self.nd_block_size, collapse=True)

  def _prepare_full_attention(self, x):
    return tf.reshape(x, [x.shape[0], -1, x.shape[-1]])

  def call(self, inputs):
    cond_inputs = memory = None
    cond_qkv = self.cond_v or self.cond_q or self.cond_k
    if cond_qkv:
      if tf.is_tensor(inputs) or len(inputs) != 2:
        raise ValueError('Expected tuple of (inputs, cond_inputs)')
      inputs, cond_inputs = inputs

    x = inputs
    if not self.is_one_dim_attention:
      # We flatten the index axes here. [B, ..., D] --> [B, M, D].
      if self.is_block_attention:
        x = self._prepare_block_attention(x)
      else:
        x = self._prepare_full_attention(x)
    memory = x
    q, k, v = self.project_q(x), self.project_k(memory), self.project_v(memory)

    q = self.cond_shift_and_scale(
        q, cond_inputs, self.cond_q, self.project_cond_q)
    k = self.cond_shift_and_scale(
        k, cond_inputs, self.cond_k, self.project_cond_k)
    v = self.cond_shift_and_scale(
        v, cond_inputs, self.cond_v, self.project_cond_v)

    q *= q.shape[-1]**-0.5
    alphas = self._dot_product(q, k, self.attention_dim_q, self.attention_dim_k)
    alphas = self._apply_mask_and_bias(alphas)
    weights = tf.nn.softmax(alphas)
    output = self._weighted_sum(weights, v, self.attention_dim_q,
                                self.attention_dim_k)
    output = self.project_final(output)
    return output


class FactorizedAttention(layers.Layer):
  """Encodes image into 2-D spatial context with factorized attention layers."""

  def __init__(self, config, **kwargs):
    super(FactorizedAttention, self).__init__(**kwargs)
    self.config = config
    self.dropout = self.config.get('dropout', 0.0)

  def build(self, input_shapes):
    ff_size, hidden_size = self.config.ff_size, self.config.hidden_size
    num_heads = self.config.num_heads
    height, width = input_shapes[1:3]

    self.pos_embed = PositionEmbed(axes=[1, 2], max_lengths=[height, width])

    self.residual_layers = []
    num_norms = 4 * self.config.num_encoder_layers
    self.layer_norms = [layers.LayerNormalization() for _ in range(num_norms)]

    for _ in range(self.config.num_encoder_layers):
      # unmasked row
      unmask_row = SelfAttentionND(
          hidden_size=hidden_size, num_heads=num_heads,
          nd_block_size=[1, width], resolution=[height, width])

      ff_row = tf.keras.Sequential([
          layers.Dense(units=ff_size, activation='relu'),
          layers.Dense(units=hidden_size)
      ])

      # unmasked column,
      unmask_col = SelfAttentionND(
          hidden_size=hidden_size, num_heads=num_heads,
          nd_block_size=[height, 1], resolution=[height, width])

      ff_col = tf.keras.Sequential([
          layers.Dense(units=ff_size, activation='relu'),
          layers.Dense(units=hidden_size)
      ])

      self.residual_layers.append(unmask_row)
      self.residual_layers.append(ff_row)
      self.residual_layers.append(unmask_col)
      self.residual_layers.append(ff_col)

  def call(self, inputs, training=True):
    inputs = self.pos_embed(inputs)

    # Apply a stack of unmaked row and column attention layers.
    for layer, norm in zip(self.residual_layers, self.layer_norms):
      output = layer(inputs)
      output = residual_dropout(inputs, output, self.dropout, training)
      inputs = norm(output)

    return inputs
