# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import numpy as np
from combiner.tf import attention
from combiner.tf import approx_attention
from combiner.tf import ops
import functools


def get_embedding(vocab_size,
                  embed_size,
                  dtype,
                  num_heads=None,
                  init_std=1.0,
                  name='embeddings'):
  """Initialize embedding weights."""
  # compatible with both 3d and 4d cases
  if num_heads is not None:
    shape = [vocab_size, num_heads, embed_size // num_heads]
  else:
    shape = [vocab_size, embed_size]
  with tf.variable_scope(None, name):
    weights = tf.get_variable(
        'weights', shape=shape,
        initializer=tf.random_normal_initializer(stddev=init_std), dtype=dtype)
    return weights


def get_position_embedding(max_len, emb_size, dtype,
                           name='position_embeddings', concatenate=False):
  """Get position encoding."""
  def _get_angles_per_position(position, dim, emb_size):
    if dtype == tf.float32:
      denominator = np.power(10000, (2 * (dim // 2)) / np.float32(emb_size))
    else:
      denominator = np.power(10000, (2 * (dim // 2)) / np.float16(emb_size))
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
    output = np.zeros_like(
        angles, dtype=np.float32 if dtype==tf.float32 else np.float16)
    output[:, 0::2] = sines
    output[:, 1::2] = cosines

  with tf.variable_scope(None, name):
    weights = tf.get_variable(
        'weights', shape=[max_len, emb_size],
        initializer=tf.constant_initializer(output), dtype=dtype)
    return weights


def embedding_lookup(embeddings, indices, implementation='lookup'):
  """Different types of embedding approaches."""
  if implementation == 'lookup':
    return tf.nn.embedding_lookup(embeddings, indices)
  elif implementation == 'matmul':
    onehot = tf.one_hot(indices, depth=embeddings.shape[0].value, axis=-1,
                        dtype=embeddings.dtype)
    return tf.einsum('BLV,VD->BLD', onehot, embeddings)
  else:
    raise ValueError('Unsupported embedding lookup implementation %s'
                     % implementation)

def vanilla_transformer_layer(x, config, is_training=True, attn_bias=None,
                      layer_idx=0):  # pylint: disable=unused-argument
  """transformer layer: attention + ffn."""
  # Attention
  with tf.variable_scope('attn'):
    shortcut, x = ops.preprocess(x, config)
    x = attention.multihead_attention(x, x, x,
                                      config.model_size,
                                      config.num_heads,
                                      is_training=is_training,
                                      dropatt=config.dropatt,
                                      attn_bias=attn_bias,
                                      bias=config.dense_use_bias)
    x = ops.postprocess(shortcut, x, config, is_training)

  # FFN
  with tf.variable_scope('ffn'):
    shortcut, x = ops.preprocess(x, config)
    x = ops.ffn(x, is_training, config.dropout)
    x = ops.postprocess(shortcut, x, config, is_training)

  return x


def transformer_approx_att_layer(x,
                                 config,
                                 is_training=True,
                                 attn_bias=None,
                                 attn_impl=None,
                                 layer_idx=0):
  """transformer layer: approximated attention + ffn."""
  # Attention
  with tf.variable_scope('attn'):
    shortcut, x = ops.preprocess(x, config)
    x = attn_impl(x, config, is_training=is_training)
    x = ops.postprocess(shortcut, x, config, is_training)

  # FFN
  with tf.variable_scope('ffn'):
    shortcut, x = ops.preprocess(x, config)
    x = ops.ffn(x, is_training, config.dropout)
    x = ops.postprocess(shortcut, x, config, is_training)

  return x


def transformer(inputs, config, is_training=True, input_mask=None,
                segment=None, causal=False):
  """Transformer encoder."""
  outputs = {}

  #### Embeddings
  word_embeddings = get_embedding(
      config.vocab_size, config.embed_size,
      dtype=config.dtype,
      init_std=config.embedding_init_std, name='word_embeddings')

  outputs['word_embeddings'] = word_embeddings
  x = embedding_lookup(word_embeddings, inputs)

  if config.pos_sine_init:
    pos_embeddings = get_position_embedding(
        config.max_seq_len, config.embed_size,
        dtype=config.dtype, name='pos_embeddings')
  else:
    pos_embeddings = get_embedding(
        config.max_seq_len, config.embed_size,
        dtype=config.dtype, init_std=0.02, name='pos_embeddings')
  outputs['pos_embeddings'] = pos_embeddings
  x += pos_embeddings[:tf.shape(x)[1]]

  if segment is not None:
    # Use for multi-segment input (GLUE datasets)
    seg_embeddings = get_embedding(config.max_num_seg, config.embed_size,
                                   dtype=config.dtype, name='seg_embeddings')
    outputs['seg_embeddings'] = seg_embeddings
    x += embedding_lookup(seg_embeddings, segment)

  x = ops.dropout(x, is_training, config.dropout)
  tf.logging.info('Embedding output: shape %s, dtype %s.', x.shape, x.dtype)

  att_type_spec = getattr(config, 'att_type', None)
  if att_type_spec is None:
    layer_fn = vanilla_transformer_layer
  else:
    attn_impl = functools.partial(
        getattr(approx_attention, att_type_spec, None),
        causal=causal)
    assert attn_impl is not None, 'unknown attention type %s' % att_type_spec
    layer_fn = functools.partial(transformer_approx_att_layer,
                                 attn_impl=attn_impl)

  #### Attn bias
  attn_bias = attention.attn_bias_from_mask(x, input_mask, causal=causal)
  #### Attention blocks
  hiddens = []
  for idx in range(config.num_layers):
    with tf.variable_scope('layer_{:0>3d}'.format(idx)):
      x = layer_fn(x,
                   config,
                   is_training=is_training,
                   attn_bias=attn_bias,
                   layer_idx=idx)
      hiddens.append(x)

  if att_type_spec == 'axial':
    # [B x A1 x A2 x ... x D] -> [B x L x D]
    x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])

  outputs['hiddens'] = hiddens
  outputs['output'] = x

  return outputs


def lm_head(hidden, config, embeddings=None, hidden_mapping=None):
  """Compute the logits used for LM/MLM."""
  hidden = ops.layer_norm(hidden, name='final_norm')
  if hidden_mapping is not None:
    # `hidden_mapping` is usually used in MLM to retrieve masked positions
    hidden = tf.einsum('BMD,BLM->BLD', hidden,
                       tf.cast(hidden_mapping, hidden.dtype))
  if embeddings is None or not config.inner_prod:
    softmax_weight = tf.get_variable(
        'softmax_weight', shape=[config.vocab_size, config.embed_size],
        initializer=ops.WEIGHT_INITIALIZER, dtype=config.dtype)
  else:
    softmax_weight = embeddings
  softmax_bias = tf.get_variable(
      'softmax_bias', shape=[config.vocab_size],
      initializer=ops.BIAS_INITIALIZER, dtype=config.dtype)
  logits = tf.einsum('BLD,VD->BLV', hidden, softmax_weight) + softmax_bias
  return logits
