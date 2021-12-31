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

"""Layers for attention."""
import copy

from combiner.tf import ops
import tensorflow.compat.v1 as tf


# Util functions
def _create_expr(symbols, prefix='B', suffix='NK'):
  """Create einsum expr with prefix and suffix."""
  return prefix + ''.join(symbols) + suffix


def _insert(l, index, elem):
  """Insert elem into the index-th position of l."""
  l = copy.deepcopy(l)
  l.insert(index, elem)
  return l


def float32_softmax(x, *args, **kwargs):
  """Perform softmax with float32 precision."""
  y = tf.cast(tf.nn.softmax(tf.cast(x, tf.float32), *args, **kwargs), x.dtype)
  return y


def get_qkv(query, key, value, hidden_size, num_heads, head_size=None,
            bias=True):
  """Get QKV projected results, with query scaled by 1/sqrt(head_size)."""
  if head_size is None:
    head_size = hidden_size // num_heads
  # query, key, value: [B x A1 x ... x An x D]
  if head_size is None:
    head_size = hidden_size // num_heads

  query_heads = ops.trail_dense(
      query, [num_heads, head_size], name='q', bias=bias)
  key_heads = ops.trail_dense(
      key, [num_heads, head_size], name='k', bias=bias)
  value_heads = ops.trail_dense(
      value, [num_heads, head_size], name='v', bias=bias)

  query_heads *= head_size ** -0.5
  return query_heads, key_heads, value_heads


def attn_bias_from_mask(hidden, padding_mask=None, causal=False):
  """Turn padding mask and causal constaint into attention bias."""
  attn_rank = hidden.shape.rank - 2
  if causal:
    causal_masks = []
    for i in range(0, attn_rank, 1):
      # [Ai x Ai]
      seq_len = tf.shape(hidden)[1 + i]
      causal_mask_i = 1.0 - tf.linalg.band_part(
          tf.ones([seq_len, seq_len], dtype=hidden.dtype),
          num_lower=-1, num_upper=0)
      causal_masks.append(causal_mask_i)

  if padding_mask:
    # [B x A1 x A2 x ...]
    padding_masks = []
    for i in range(0, attn_rank, 1):
      # [B x A1 x ... x N x Ai x Ai]
      perm = list(k for k in range(attn_rank) if k != i) + [i]
      padding_mask_i = tf.transpose(padding_mask, perm=perm)
      padding_mask_i = tf.expand_dims(padding_mask_i, axis=-2)
      padding_mask_i = tf.expand_dims(padding_mask_i, axis=-3)
      padding_mask_i = tf.cast(padding_mask_i, hidden.dtype)
      padding_masks.append(padding_mask_i)

  attn_biases = [None] * attn_rank
  for i in range(attn_rank):
    if causal and padding_mask:
      attn_biases[i] = -1e9 * tf.cast(
          causal_masks[i] + padding_masks[i] > 0, hidden.dtype)
    elif causal:
      attn_biases[i] = -1e9 * causal_masks[i]
    elif padding_mask:
      attn_biases[i] = -1e9 * padding_masks[i]

  tf.logging.info('Attn biases: %s', attn_biases)

  if isinstance(attn_biases, list) and len(attn_biases) == 1:
    return attn_biases[0]
  else:
    return attn_biases


def dot_product_attention(query_heads, key_heads, value_heads,
                          is_training, attn_axis=0, dropatt=0.0,
                          attn_bias=None):
  """Perform dot-product attention."""
  # Einsum expression:
  #   B = batch_size
  #   N = num_heads
  #   K = head_size
  #   S = query_len (of the given attn_axis)
  #   T = key/value_len (of the given attn_axis)
  #   [U-Z] = length of other attension axes
  # Example for 5D query_heads, (e.g. images [B x H x W x N x K])
  # - when attn_axis = 0 (H axis):
  #     symbols = ['U']   => num_attn_dims = 2
  #     q_expr = 'BSUNK'  => 'S' is inserted, prefix = 'B', suffix = 'NK'
  #     k_expr = 'BTUNK'  => 'T' is inserted, prefix = 'B', suffix = 'NK'
  #     a_expr = 'BUNST'  => 'S x T' attention map
  #     logit_expr = 'BSUNK,BTUNK->BUNST'
  #     atten_expr = 'BUNST,BTUNK->BSUNK'
  num_attn_dims = query_heads.shape.rank - 3   # bsz, num_heads, head_size
  symbols = [chr(ord('U') + i) for i in range(num_attn_dims - 1)]
  q_expr = _create_expr(_insert(symbols, attn_axis, 'S'))
  k_expr = _create_expr(_insert(symbols, attn_axis, 'T'))
  v_expr = _create_expr(_insert(symbols, attn_axis, 'T'))
  a_expr = _create_expr(symbols, suffix='NST')
  logit_expr = '{},{}->{}'.format(q_expr, k_expr, a_expr)
  atten_expr = '{},{}->{}'.format(a_expr, v_expr, q_expr)
  tf.logging.debug('logit expr: %s', logit_expr)
  tf.logging.debug('atten expr: %s', atten_expr)

  # attention
  attn_logits = tf.einsum(logit_expr, query_heads, key_heads)
  if attn_bias is not None:
    attn_logits += attn_bias
  attn_probs = float32_softmax(attn_logits, axis=-1)
  # tf.logging.info('Attention matrix shape: %s', attn_probs.shape)
  if is_training:
    attn_probs = tf.nn.dropout(attn_probs, rate=dropatt)
  attn_out = tf.einsum(atten_expr, attn_probs, value_heads)
  return attn_out


############################
# Softmax attention
############################
def multihead_attention(query, key, value, hidden_size, num_heads, is_training,
                        attn_axis=0, dropatt=0.1, head_size=None,
                        attn_bias=None, bias=True):
  """Attention along a specified axis."""
  query_heads, key_heads, value_heads = get_qkv(query, key, value,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                head_size=head_size,
                                                bias=bias)
  attn_out = dot_product_attention(query_heads, key_heads, value_heads,
                                   is_training, attn_axis=attn_axis,
                                   dropatt=dropatt, attn_bias=attn_bias)
  output = ops.trail_dense(attn_out, hidden_size, begin_axis=-2)
  # tf.logging.info('Attention output shape: %s', output.shape)

  return output

