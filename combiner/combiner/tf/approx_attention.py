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

# pylint: skip-file
import tensorflow.compat.v1 as tf
import math
from combiner.tf import attention
from combiner.tf import ops
import functools


def shift_right(x, axis):
  """Shift input x to the right along given axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = tf.pad(x, pad_widths)
  return tf.slice(padded, begin=[0]*len(x.shape), size=x.shape)


def shift_left(x, axis):
  """Shift input x to the left along given axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  padded = tf.pad(x, pad_widths)
  begin = [0]*len(x.shape)
  begin[axis] = 1
  return tf.slice(padded, begin=begin, size=x.shape)


def approx_cummax(x, axis, exclusive=False, reverse=False):
  """Approximate the cummax operation in jax."""
  sum_x = tf.math.cumsum(x, axis, exclusive=exclusive, reverse=reverse)
  # return tf.math.cumsum(tf.nn.relu(x), axis, reverse=reverse)
  return sum_x


def get_causal_mask(x, axis, is_strict, upper=False):
  """Get attention mask bias (keep a lower triangle).

  Args:
    x: input tensor
    axis: across which dim to make mask
    is_strict: if True, the diagonal will be masked out as well.
    upper: upper or lower triangle

  Returns:
    mask: tensor of {0, -1e9} ^ (x.shape[axis], x.shape[axis])
  """
  seq_len = tf.shape(x)[axis]
  if is_strict:
    if upper:
      mask = tf.linalg.band_part(
          tf.ones([seq_len, seq_len], dtype=x.dtype),
          num_lower=-1, num_upper=0)
    else:
      mask = tf.linalg.band_part(
          tf.ones([seq_len, seq_len], dtype=x.dtype),
          num_lower=0, num_upper=-1)
  else:
    if upper:
      mask = 1.0 - tf.linalg.band_part(
          tf.ones([seq_len, seq_len], dtype=x.dtype),
          num_lower=0, num_upper=-1)
    else:
      mask = 1.0 - tf.linalg.band_part(
          tf.ones([seq_len, seq_len], dtype=x.dtype),
          num_lower=-1, num_upper=0)
  mask = -1e9 * mask
  return mask


def pooling_summary(x, axis, local_summary, keepdims=False):
  """Perform a cheap pooling summary of a span.

  Args:
    x: input tensor
    axis: over which axis to summarize
    local_summary: str of format activation-pooling, choose
      from {relu, identity}-{max, sum, mean}
    keepdims: whether to keep the summarized singleton axis.

  Returns:
    y: the same shape as x for other axis,
      except y.shape[axis] = 1 if keepdims=True,
      otherwise y.rank = x.rank + 1
  """
  act, pool = local_summary.split('-')
  if act == 'relu':
    x = tf.nn.relu(x)
  elif act == 'identity':
    pass
  elif act == 'deepset':
    x = ops.trail_dense(x, x.shape.as_list()[-1], bias=False)
    x = tf.nn.relu(x)
  else:
    raise ValueError('Unsupported activation: %s' % act)
  if pool == 'mean':
    x = tf.math.reduce_mean(x, axis=axis, keepdims=keepdims)
  elif pool == 'max':
    x = tf.math.reduce_max(x, axis=axis, keepdims=keepdims)
  elif pool == 'sum':
    x = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
  else:
    raise ValueError('Unsupported pooling: %s' % pool)
  return x


def axial_mixture_unidir(x, config, is_training=True, causal=True):
  """Full attention matrix with axial pattern as local and mixture for global summary."""
  del is_training
  assert causal
  bsize = x.shape[0]
  query, key, value = attention.get_qkv(x, x, x, hidden_size=config.model_size,
                                        num_heads=config.num_heads, bias=config.dense_use_bias)
  head_dim = config.model_size // config.num_heads
  assert config.max_seq_len % config.max_seg_len == 0
  num_seg = config.max_seq_len // config.max_seg_len
  cur_query = tf.reshape(query, [bsize,
                                 num_seg,
                                 config.max_seg_len,
                                 config.num_heads,
                                 head_dim])
  cur_key = tf.reshape(key, cur_query.shape)
  cur_val = tf.reshape(value, cur_query.shape)

  col_logit_expr = 'BSUNK,BTUNK->BUNST'
  col_attn_expr = 'BUNST,BTUNK->BSUNK'
  col_strict_mask = get_causal_mask(cur_query,
                             axis=1,
                             is_strict=True)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  row_logit_expr = 'BUSNK,BUTNK->BUNST'
  row_attn_expr = 'BUNST,BUTNK->BUSNK'
  row_mask = get_causal_mask(cur_query,
                             axis=2,
                             is_strict=False)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  col_logits = tf.einsum(col_logit_expr, cur_query, cur_key) + col_strict_mask
  row_logits = tf.einsum(row_logit_expr, cur_query, cur_key) + row_mask

  ###################
  col_up2down_query = approx_cummax(cur_query, axis=1)
  col_up2down_key = shift_right(approx_cummax(cur_key, axis=1), axis=1)
  col_mask = get_causal_mask(
      cur_query, axis=1, is_strict=False)[tf.newaxis, tf.newaxis,
                                          tf.newaxis, :, :]
  col_up2down_logits = tf.einsum(col_logit_expr, col_up2down_query,
                                 cur_key) + col_mask
  col_up2down_attn_weights = attention.float32_softmax(
      col_up2down_logits, axis=-1)
  col_up2down_summary = tf.einsum(col_attn_expr, col_up2down_attn_weights,
                                  cur_val)
  col_up2down_summary = shift_right(col_up2down_summary, axis=1)

  row_only_myself_mask = tf.eye(tf.shape(cur_query)[2], dtype=cur_query.dtype)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  row_without_myself_mask = -1e9 * row_only_myself_mask
  all_maskout = tf.cast(tf.fill(row_without_myself_mask.shape, -1e9), cur_query.dtype)
  row_without_myself_mask = tf.concat([all_maskout] + [row_without_myself_mask] * (cur_query.shape[1] - 1),
                                      axis=1)
  previous_row_logits = tf.einsum(row_logit_expr, cur_query, col_up2down_key) + row_without_myself_mask
  ###################

  row_left2right_query = approx_cummax(cur_query, axis=2)
  row_left2right_key = shift_right(approx_cummax(cur_key, axis=2), axis=2)
  row_left2right_logits = tf.einsum(row_logit_expr, row_left2right_query,
                                    cur_key) + row_mask
  row_left2right_attn_weights = attention.float32_softmax(
      row_left2right_logits, axis=-1)
  row_left2right_summary = tf.einsum(row_attn_expr, row_left2right_attn_weights,
                                     cur_val)
  row_left2right_summary = shift_right(row_left2right_summary, axis=2)

  all_maskout = tf.cast(tf.fill(col_strict_mask.shape, -1e9), cur_query.dtype)
  col_strict_without_first_mask = tf.concat(
      [all_maskout] + [col_strict_mask] * (cur_query.shape[2] - 1), axis=1)
  top_left_col_logits = tf.einsum(
      col_logit_expr, cur_query,
      row_left2right_key) + col_strict_without_first_mask
  ###################

  row_right2left_query = approx_cummax(cur_query, axis=2, reverse=True)
  row_right2left_key = shift_left(
      approx_cummax(cur_key, axis=2, reverse=True), axis=2)
  row_upper_mask = get_causal_mask(
      cur_query, axis=2, is_strict=False, upper=True)[tf.newaxis, tf.newaxis,
                                                      tf.newaxis, :, :]
  row_right2left_logits = tf.einsum(row_logit_expr, row_right2left_query,
                                    cur_key) + row_upper_mask
  row_right2left_attn_weights = attention.float32_softmax(
      row_right2left_logits, axis=-1)
  row_right2left_summary = tf.einsum(row_attn_expr, row_right2left_attn_weights,
                                     cur_val)
  row_right2left_summary = shift_left(row_right2left_summary, axis=2)
  col_strict_without_last_mask = tf.concat(
      [col_strict_mask] * (cur_query.shape[2] - 1) + [all_maskout], axis=1)
  top_right_col_logits = tf.einsum(
      col_logit_expr, cur_query,
      row_right2left_key) + col_strict_without_last_mask
  ###################

  joint_logits = tf.concat([
      tf.transpose(col_logits, perm=[0, 3, 2, 1, 4]), row_logits,
      previous_row_logits,
      tf.transpose(top_left_col_logits, perm=[0, 3, 2, 1, 4]),
      tf.transpose(top_right_col_logits, perm=[0, 3, 2, 1, 4])
  ],
                           axis=-1)
  attn_weights = attention.float32_softmax(joint_logits, axis=-1)
  col_att, row_att, previous_row_att, top_left_col_att, top_right_col_att = tf.split(attn_weights,
                                                                                     [num_seg,
                                                                                      config.max_seg_len,
                                                                                      config.max_seg_len,
                                                                                      num_seg,
                                                                                      num_seg], axis=-1)
  col_att = tf.transpose(col_att, [0, 3, 2, 1, 4])
  top_left_col_att = tf.transpose(top_left_col_att, [0, 3, 2, 1, 4])
  top_right_col_att = tf.transpose(top_right_col_att, [0, 3, 2, 1, 4])
  col_merged = tf.einsum(col_attn_expr, col_att, cur_val)
  row_merged = tf.einsum(row_attn_expr, row_att, cur_val)
  previous_row_merged = tf.einsum(row_attn_expr, previous_row_att,
                                  col_up2down_summary)
  top_left_merged = tf.einsum(col_attn_expr, top_left_col_att,
                              row_left2right_summary)
  top_right_merged = tf.einsum(col_attn_expr, top_right_col_att,
                               row_right2left_summary)

  joint_merged = tf.reshape(
      col_merged + row_merged + previous_row_merged + top_left_merged +
      top_right_merged,
      [bsize, num_seg * config.max_seg_len, config.num_heads, head_dim])
  output = ops.trail_dense(joint_merged, config.model_size, begin_axis=-2)
  return output


def sqrt_fixed_full(x, config, is_training=True, causal=True):
  """Full attention matrix with sqrt decomposition."""
  bsize = x.shape[0]
  query, key, value = attention.get_qkv(x, x, x, hidden_size=config.model_size,
                                        num_heads=config.num_heads,
                                        bias=config.dense_use_bias)
  head_dim = config.model_size // config.num_heads
  assert config.max_seq_len % config.max_seg_len == 0
  num_seg = config.max_seq_len // config.max_seg_len
  cur_query = tf.reshape(query, [-1,
                                 num_seg,
                                 config.max_seg_len,
                                 config.num_heads,
                                 head_dim])
  with tf.variable_scope('pooling_query'):
    merged_query = pooling_summary(cur_query, axis=2,
                                   local_summary=config.local_summary,
                                   keepdims=True)
  cur_key = tf.reshape(key, cur_query.shape)
  cur_val = tf.reshape(value, cur_query.shape)
  span_val = attention.dot_product_attention(merged_query,
                                             cur_key,
                                             cur_val,
                                             is_training=is_training,
                                             attn_axis=1,
                                             dropatt=config.dropatt)
  span_val = tf.squeeze(span_val, axis=2)
  with tf.variable_scope('pooling_key'):
    span_key = pooling_summary(cur_key, axis=2,
                               local_summary=config.local_summary,
                               keepdims=False)
  local_logits = tf.einsum('bsqhd,bskhd->bsqhk', cur_query, cur_key)
  if causal:
    local_mask = get_causal_mask(cur_query, axis=2, is_strict=False)
    local_mask = tf.expand_dims(local_mask, axis=-2)
    local_logits += local_mask
  prev_logits = tf.einsum('bqhd,bkhd->bqhk', query, span_key)
  if causal:
    prev_mask = get_causal_mask(cur_query, axis=1, is_strict=True)
    prev_mask = tf.repeat(prev_mask, [config.max_seg_len] * num_seg, axis=0)
    prev_logits += tf.expand_dims(prev_mask, axis=1)
  joint_logits = tf.concat([tf.reshape(local_logits,
                                       [bsize, config.max_seq_len,
                                        config.num_heads, -1]),
                            prev_logits], axis=-1)
  attn_weights = attention.float32_softmax(joint_logits, axis=-1)
  local_att, prev_att = tf.split(attn_weights, [config.max_seg_len, num_seg],
                                 axis=-1)
  if is_training:
    local_att = tf.nn.dropout(local_att, rate=config.dropatt)
  local_att = tf.reshape(local_att, [bsize, num_seg,
                                     config.max_seg_len,
                                     config.num_heads,
                                     config.max_seg_len])
  local_merged = tf.einsum('bsqhk,bskhd->bsqhd', local_att, cur_val)
  prev_merged = tf.einsum('bqhk,bkhd->bqhd', prev_att, span_val)
  joint_merged = prev_merged + tf.reshape(local_merged, prev_merged.shape)
  output = ops.trail_dense(joint_merged, config.model_size, begin_axis=-2)
  return output


def axial_rowmajor(x, config, is_training=True, causal=True):
  """Full attention matrix with sqrt decomposition."""
  bsize = x.shape[0]
  seq_len = x.shape.as_list()[1]
  head_dim = config.model_size // config.num_heads
  assert seq_len % config.max_seg_len == 0
  num_seg = seq_len // config.max_seg_len
  x_sqr = tf.reshape(x,
                     [bsize, num_seg, config.max_seg_len, config.model_size])
  q_row_local, key_row_local, value_row_local = attention.get_qkv(
      x_sqr, x_sqr, x_sqr, hidden_size=config.model_size,
      num_heads=config.num_heads, bias=config.dense_use_bias)
  local_logits = tf.einsum('bsqhd,bskhd->bsqhk', q_row_local, key_row_local)
  row_probs = attention.float32_softmax(local_logits, axis=-1)
  if is_training:
    row_probs = tf.nn.dropout(row_probs, rate=config.dropatt)

  row_attn_out = tf.einsum('bsqhk,bskhd->bsqhd', row_probs, value_row_local)
  if config.row_summary == 'none':
    key_row = key_row_local
  elif config.row_summary in ['wsum', 'proj', 'wsum_proj']:
    if 'wsum' in config.row_summary:
      pre_summary = tf.einsum('bsqhk,bskhd->bsqhd', row_probs, key_row_local)
    else:
      pre_summary = row_attn_out
    if 'proj' in config.row_summary:
      with tf.variable_scope('rowmajor_param_post'):
        key_row = ops.trail_dense(pre_summary, config.model_size, begin_axis=-2,
                                  bias=config.dense_use_bias)
        key_row = ops.postprocess(x_sqr, key_row, config, is_training)
        _, key_row = ops.preprocess(key_row, config)
        key_row = ops.trail_dense(key_row, [config.num_heads, head_dim],
                                  bias=config.dense_use_bias)
    else:
      key_row = pre_summary
  else:
    raise ValueError('Unknown row summary %s' % config.row_summary)
  if causal:
    local_mask = get_causal_mask(q_row_local, axis=2, is_strict=False)
    local_logits += local_mask[:, tf.newaxis, :]

  global_logits = tf.einsum('bqlhd,bklhd->bqlhk', q_row_local, key_row)
  if causal:
    global_mask = get_causal_mask(q_row_local, axis=1, is_strict=True)
    global_logits += global_mask[:, tf.newaxis, tf.newaxis, :]
  # (bsize, num_seg, seg_len, n_head, seg_len + num_seg)
  joint_logits = tf.concat([local_logits, global_logits], axis=-1)
  attn_probs = attention.float32_softmax(joint_logits, axis=-1)
  local_att, global_att = tf.split(attn_probs,
                                   [config.max_seg_len, num_seg],
                                   axis=-1)
  if is_training:
    local_att = tf.nn.dropout(local_att, rate=config.dropatt)
  local_merged = tf.einsum('bsqhk,bskhd->bsqhd', local_att, value_row_local)
  global_merged = tf.einsum('bqlhv,bvlhd->bqlhd', global_att, row_attn_out)
  joint_merged = tf.reshape(local_merged + global_merged,
                            [bsize, seq_len,
                             config.num_heads, head_dim])
  output = ops.trail_dense(joint_merged, config.model_size,
                           begin_axis=-2, bias=config.dense_use_bias)
  return output


def axial_mixture_bidir(x, config, is_training=True, causal=False):
  """Full attention matrix with axial mixture decomposition."""
  assert not causal
  bsize = x.shape[0]
  seq_len = x.shape.as_list()[1]
  head_dim = config.model_size // config.num_heads
  assert seq_len % config.max_seg_len == 0
  num_seg = seq_len // config.max_seg_len
  x_sqr = tf.reshape(x,
                     [bsize, num_seg, config.max_seg_len, config.model_size])
  query, key, value = attention.get_qkv(
      x_sqr, x_sqr, x_sqr, hidden_size=config.model_size,
      num_heads=config.num_heads, bias=config.dense_use_bias)
  local_row_logits = tf.einsum('bushd,buthd->bhust', query, key)
  local_col_logits = tf.einsum('bsuhd,btuhd->bhsut', query, key)
  # TODO: add self-mask for local_col_logits

  span_attn_fn = functools.partial(attention.dot_product_attention,
                                   key_heads=key,
                                   value_heads=value,
                                   is_training=is_training,
                                   dropatt=config.dropatt)

  # === top-down summary ===
  col_query_topdown = approx_cummax(query, 1, exclusive=True)
  col_key_topdown = approx_cummax(key, 1, exclusive=True)
  col_t2d_mask = get_causal_mask(x_sqr, axis=1, is_strict=True)
  col_t2d_val = span_attn_fn(query_heads=col_query_topdown,
                             attn_axis=0,
                             attn_bias=col_t2d_mask)

  # === bottom-up summary ===
  col_query_bottomup = approx_cummax(query, 1, exclusive=True, reverse=True)
  col_key_bottomup = approx_cummax(key, 1, exclusive=True, reverse=True)
  col_b2t_mask = get_causal_mask(x_sqr, axis=1, is_strict=True, upper=True)
  col_b2t_val = span_attn_fn(query_heads=col_query_bottomup,
                             attn_axis=0,
                             attn_bias=col_b2t_mask)

  # === left2right summary ===
  row_query_left2right = approx_cummax(query, 2, exclusive=True)
  row_key_left2right = approx_cummax(key, 2, exclusive=True)
  row_l2r_mask = get_causal_mask(x_sqr, axis=2, is_strict=True)
  row_l2r_val = span_attn_fn(query_heads=row_query_left2right,
                             attn_axis=1,
                             attn_bias=row_l2r_mask)

  # === right2left summary ===
  row_query_right2left = approx_cummax(query, 2, exclusive=True, reverse=True)
  row_key_right2left = approx_cummax(key, 2, exclusive=True, reverse=True)
  row_r2l_mask = get_causal_mask(x_sqr, axis=2, is_strict=True, upper=True)
  row_r2l_val = span_attn_fn(query_heads=row_query_right2left,
                             attn_axis=1,
                             attn_bias=row_r2l_mask)

  global_t2d_logits = tf.einsum('bushd,buthd->bhust', query, col_key_topdown)
  global_b2t_logits = tf.einsum('bushd,buthd->bhust', query, col_key_bottomup)
  global_l2r_logits = tf.einsum('bsuhd,btuhd->bhsut', query, row_key_left2right)
  global_r2l_logits = tf.einsum('bsuhd,btuhd->bhsut', query, row_key_right2left)
  joint_logits = tf.concat([local_row_logits, local_col_logits,
                            global_t2d_logits, global_b2t_logits,
                            global_l2r_logits, global_r2l_logits], axis=-1)
  attn_probs = attention.float32_softmax(joint_logits, axis=-1)
  prow, pcol, pt2d, pb2t, pl2r, pr2l = tf.split(
      attn_probs, [config.max_seg_len, num_seg, config.max_seg_len,
                   config.max_seg_len, num_seg, num_seg], axis=-1)
  mrow = tf.einsum('bhust,buthd->bushd', prow, value)
  mcol = tf.einsum('bhsut,btuhd->bsuhd', pcol, value)
  mt2d = tf.einsum('bhust,buthd->bushd', pt2d, col_t2d_val)
  mb2t = tf.einsum('bhust,buthd->bushd', pb2t, col_b2t_val)
  ml2r = tf.einsum('bhsut,btuhd->bsuhd', pl2r, row_l2r_val)
  mr2l = tf.einsum('bhsut,btuhd->bsuhd', pr2l, row_r2l_val)
  joint_merged = mrow + mcol + mt2d + mb2t + ml2r + mr2l
  joint_merged = tf.reshape(joint_merged,
                            [bsize, seq_len, config.num_heads, head_dim])
  output = ops.trail_dense(joint_merged, config.model_size,
                           begin_axis=-2, bias=config.dense_use_bias)
  return output
