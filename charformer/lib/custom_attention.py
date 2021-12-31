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

"""Custom attention modules for Charformer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer.transformer import sinusoid_positional_embedding_weights
import tensorflow.compat.v1 as tf


def local_attention_1d(q,
                       k,
                       v,
                       length_dim,
                       key_dim,
                       value_dim,
                       fully_autoregressive=True,
                       length_dim_num_splits=1,
                       radius=128,
                       sequence_id=1,
                       write_priority=None,
                       read_priority=None,
                       attention_kwargs=None,
                       context=None):
  """Attention to the a neighborood around the source.

  If fully_autoregressive, then query position p can only see memory positions
  in the range (p - radius, p].

  If not fully_autoregressive, then query position p can only see memory
  positions in the range (p - window_size, p + radius].

  In addition, if write_priority and read_priority are provided, then attention
  is limited to position pairs where
  read_priority[query position] >= write_priority[memory position]

  Args:
    q: a Tensor containing length_dim
    k: a Tensor containing length_dim
    v: an optional Tensor containing length_dim.  If none then uses v=k.
    length_dim: a Dimension
    key_dim: a Dimension (the channels dimension of q and k)
    value_dim: a Dimension (the channels dimension of v)
    fully_autoregressive: a boolean
    length_dim_num_splits: an optional integer indicating how many ways the
      length dimension is split
    radius: an integer
    sequence_id: a Tensor or an integer
    write_priority: an optional Tensor containing length_dim
    read_priority: an optional Tensor containing length_dim
    attention_kwargs: optional keyword arguments for attention()
    context: optional context.

  Returns:
    a Tensor with the shape x.shape - key_dim + value_dim

  Raises:
    ValueError: if channels or depth don't match.
  """
  # Choose a suitable block size.
  # We choose the greatest divisor of length_per_split less than or equal
  # to max(window_size, 128)
  tf.logging.info(attention_kwargs)
  length_per_split = length_dim.size // length_dim_num_splits
  block_length = max(radius, 128)
  while length_per_split % block_length != 0:
    block_length -= 1
  query_block_length = mtf.Dimension("query_block_length", block_length)
  memory_block_length = mtf.Dimension("memory_block_length", block_length)
  # The num_blocks dimension gets the same name as the length dimension,
  # so it will be split in the same way.
  num_blocks = mtf.Dimension(length_dim.name, length_dim.size // block_length)
  def _reshape_query(x):
    return mtf.replace_dimensions(
        x, length_dim, [num_blocks, query_block_length])
  def _reshape_memory(x):
    x = mtf.replace_dimensions(
        x, length_dim, [num_blocks, memory_block_length])
    return (mtf.left_halo_exchange if fully_autoregressive
            else mtf.halo_exchange)(
                x, num_blocks, memory_block_length, radius)
  q = _reshape_query(q)
  k = _reshape_memory(k)
  if v:
    v = _reshape_memory(v)
  else:
    v = k
  if sequence_id is None:
    sequence_id = 1
  if (not isinstance(sequence_id, mtf.Tensor) or
      length_dim not in sequence_id.shape.dims):
    sequence_id += mtf.zeros(q.mesh, [length_dim], tf.int32)
  q_sequence_id = _reshape_query(sequence_id)
  m_sequence_id = _reshape_memory(sequence_id)
  pos = mtf.range(q.mesh, length_dim, dtype=tf.int32)
  q_pos = _reshape_query(pos)
  m_pos = _reshape_memory(pos)

  padded_memory_block_length = mtf.Dimension(
      "memory_block_length",
      (1 if fully_autoregressive else 2) * radius + block_length)

  relative_position = m_pos - q_pos
  visible = mtf.equal(q_sequence_id, m_sequence_id)
  visible = mtf.logical_and(visible, mtf.greater(relative_position, -radius))
  visible = mtf.logical_and(visible, mtf.less_equal(
      relative_position, 0 if fully_autoregressive else radius))
  if read_priority is not None:
    write_priority = _reshape_memory(write_priority)
    read_priority = _reshape_query(read_priority)
    visible = mtf.logical_and(
        visible, mtf.greater_equal(read_priority, write_priority))

  bias = attention.visibility_mask_to_attention_bias(visible, q.dtype)
  o = attention.attention(q, k, v, padded_memory_block_length, key_dim,
                          value_dim, bias, context=context,
                          **attention_kwargs)
  return mtf.replace_dimensions(o, [num_blocks, query_block_length], length_dim)


def gradient_based_subword_tokenization(x,
                                        length_dim,
                                        max_subword_length=4,
                                        downsample=None,
                                        use_offsets=False,
                                        consider_chars_as_blocks=False,
                                        use_block_pos_embedding=False,
                                        share_block_kernel=False,
                                        memory_embeddings=0,
                                        context=None,
                                        block_mixing_mode=None,
                                        activation="softmax",
                                        downsample_function="mean"):
  """Implements GBSWT from Charformer.

  Args:
    x: a Tensor containing length_dim
    length_dim: a Dimension
    max_subword_length: integer
    downsample: integer.
    use_offsets: boolean.
    consider_chars_as_blocks: boolean.
    use_block_pos_embedding: boolean.
    share_block_kernel: boolean.
    memory_embeddings: integer.
    context: Context.
    block_mixing_mode: Str for block mixing.
    activation: Str for block ranking.
    downsample_function: Str, supports mean/linformer for now.

  Returns:
    a Tensor with the same shape as x.

  Raises:
    ValueError: if channels or depth don't match.
  """
  # don't use this for now.
  del max_subword_length
  del memory_embeddings
  all_blocks = []
  all_scores = []
  tf.logging.info("GSW block layer")

  def _tile(x, n, tile_dim):
    # Simple tile function in MTF.
    return mtf.concat([x] * n, tile_dim.name)

  def _repeat(x, n, repeat_dim):
    # repeat function in MTF
    tmp_dim = mtf.Dimension("tmp", 1)
    expand_shape = mtf.Shape(x.shape.dims + [tmp_dim])
    x = mtf.reshape(x, expand_shape)
    x = _tile(x, n, tmp_dim)
    output_shape = []
    for dim in x.shape.dims:
      if dim.name == "tmp":
        continue
      if dim.name == repeat_dim.name:
        dim = mtf.Dimension(dim.name, dim.size * n)
      output_shape.append(dim)
    output_shape = mtf.Shape(output_shape)
    x = mtf.reshape(x, output_shape)
    return x

  def _combined_dim(dims):
    return mtf.Dimension(dims[0].name, mtf.Shape(dims).size)

  # compute all subword blocks
  # TODO(yitay): handle offsets to get all blocks
  if activation == "sigtanh":
    # one score for sigmoid
    tmp_dim = mtf.Dimension("block_score", 2)
  else:
    tmp_dim = mtf.Dimension("block_score", 1)

  model_dim = x.shape[-1]
  subword_blocks_width = [2, 3, 4]

  if consider_chars_as_blocks:
    subword_blocks_width += [1]

  if share_block_kernel:
    block_kernel_shape = mtf.Shape([model_dim, tmp_dim])
    block_kernel = mtf.get_variable(
        x.mesh, "block_kernel", block_kernel_shape, initializer=None,
        dtype=context.variable_dtype)
  else:
    block_kernel = None

  for subword_len in subword_blocks_width:
    if use_block_pos_embedding:
      # this is turn off by default. It is meant to support cases like
      # parameterized pooling or other features.
      block_len_dim = mtf.Dimension(length_dim.name, subword_len)
      # TODO(vqtran): Consider other positional embeddings.
      block_pos_emb = sinusoid_positional_embedding_weights(
          context.mesh, block_len_dim, x.shape[-1],
          context.variable_dtype.activation_dtype)
      block_pos_emb = _repeat(block_pos_emb,
                              math.ceil(length_dim.size / float(subword_len)),
                              block_len_dim)
    if use_offsets:
      offset_space = subword_len
    else:
      offset_space = 1
    for offsets in range(offset_space):
      if offsets > 0:
        xoff = mtf.shift(x, offsets, length_dim, wrap=False)
        if use_block_pos_embedding:
          block_pos_emb = mtf.shift(
              block_pos_emb, offsets, block_pos_emb.shape[-2], wrap=False)
      else:
        xoff = x
      tf.logging.info("SW len=%d offset=%d", subword_len, offsets)
      if length_dim.size % subword_len != 0:
        tf.logging.info("Not divisible by length")
        # add extra padding tokens
        pad_amt = int(subword_len) - int(
            length_dim.size % subword_len)
        kp = mtf.pad(xoff, [0, pad_amt], length_dim.name)
      else:
        kp = xoff

      if use_block_pos_embedding:
        kp += block_pos_emb

      bx = mtf.pool_tensor_1d(
          kp,
          pool_dim=kp.shape.get_dim_by_name("length"),
          reduce_fn=mtf.reduce_mean,
          pool_size=int(subword_len))
      block_score = mtf.layers.dense(
          bx, [tmp_dim],
          use_bias=False,
          name="bx",
          reduced_dims=[model_dim],
          variable_dtype=None,
          kernel_weights=block_kernel)

      expand_bx = _repeat(bx, subword_len, length_dim)
      expand_scores = _repeat(block_score, subword_len, length_dim)
      if offsets > 0:
        # add offset.
        expand_bx = mtf.pad(expand_bx, [offsets, 0], length_dim.name)
        expand_scores = mtf.pad(expand_scores, [offsets, 0], length_dim.name)
      new_len = expand_bx.shape.get_dim_by_name(length_dim.name)
      if new_len.size < length_dim.size:
        pad_amt = new_len.size - length_dim.size
        expand_bx = mtf.pad(expand_bx, [0, pad_amt], length_dim.name)
        expand_scores = mtf.pad(expand_scores, [0, pad_amt], length_dim.name)
      elif new_len.size > length_dim.size:
        expand_bx = mtf.slice(expand_bx, 0, length_dim.size, length_dim.name)
        expand_scores = mtf.slice(expand_scores, 0, length_dim.size,
                                  length_dim.name)

      new_tmp_dim = mtf.Dimension("extra_dim", 1)
      expand_shape = mtf.Shape(expand_bx.shape.dims + [new_tmp_dim])
      expand_scores_shape = mtf.Shape(expand_scores.shape.dims + [new_tmp_dim])
      expand_bx = mtf.reshape(expand_bx, expand_shape)
      expand_scores = mtf.reshape(expand_scores, expand_scores_shape)
      all_blocks.append(expand_bx)
      all_scores.append(expand_scores)

  all_blocks = mtf.concat(all_blocks, new_tmp_dim.name)
  all_scores = mtf.concat(all_scores, new_tmp_dim.name)
  tf.logging.info(all_blocks)
  new_tmp_dim = all_blocks.shape.get_dim_by_name("extra_dim")
  combined_dim = _combined_dim([new_tmp_dim, tmp_dim])
  block_net_shape = all_scores.shape - tmp_dim - new_tmp_dim + combined_dim
  block_net = mtf.reshape(all_scores, block_net_shape)

  if block_mixing_mode == "score_attention":
    tf.logging.info("Using score attention")
    att = mtf.einsum([block_net, block_net], reduced_dims=[new_tmp_dim])
    tf.logging.info(block_net)
    att = mtf.softmax(att, reduced_dim=att.shape[-1])
    block_net = mtf.einsum([att, block_net], output_shape=block_net.shape)
    tf.logging.info(block_net)

  if activation == "softmax":
    block_net = mtf.softmax(block_net, reduced_dim=new_tmp_dim)
  elif activation == "tanh":
    tf.logging.info("Using tanh")
    block_net = mtf.tanh(block_net)

  all_blocks = block_net * all_blocks
  all_blocks = mtf.reduce_sum(all_blocks, reduced_dim=new_tmp_dim)
  output = all_blocks

  if downsample:
    output_length = output.shape.get_dim_by_name("length")
    if output_length.size % int(downsample) != 0:
      pad_amt = int(downsample) - int(output_length.size % int(downsample))
      output = mtf.pad(output, [0, pad_amt], output_length.name)
    if downsample_function == "mean":
      output = mtf.pool_tensor_1d(
          output,
          pool_dim=output.shape.get_dim_by_name("length"),
          reduce_fn=mtf.reduce_mean,
          pool_size=int(downsample))
    else:
      raise ValueError("Downsampling function not implemeneted.")

  return output

