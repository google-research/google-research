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

"""Functions for embedding tokens."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf


def embed_tokens(tokens, task_vocab_size, hidden_size, hparams,
                 embed_scope=None):
  """Embeds tokens."""
  with tf.variable_scope("embed_tokens" if embed_scope is None else embed_scope,
                         reuse=tf.AUTO_REUSE) as scope:
    input_embeddings = tf.nn.embedding_lookup(
        params=tf.get_variable(
            name="task_embed_w",
            shape=[task_vocab_size, hidden_size]),
        ids=tokens, name="embed_tokens")
  if hparams.get("freeze_reference_model", False):
    input_embeddings = tf.stop_gradient(input_embeddings)
  return input_embeddings, scope


def average_bag_of_embeds(embeddings, mask, use_bigrams=False,
                          bigram_embed_scope=None, append_start_end=False):
  """Averages a bag of embeds.

  Args:
    embeddings: a float Tensor of shape [None, length, depth]
    mask: a boolean Tensor of shape [None, length]
    use_bigrams: whether to use bigrams.
    bigram_embed_scope: the variable scope.
    append_start_end: whether to append start and end tokens.
  Returns:
    word_embed: a Tensor of shape [None, embed_size]
  """
  if bigram_embed_scope is None:
    var_scope = "average_bow"
  else:
    var_scope = bigram_embed_scope
  with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(embeddings), 3, summarize=100),
        tf.assert_equal(tf.rank(mask), 2, summarize=100),
    ]):
      lengths = tf.cast(
          tf.reduce_sum(tf.cast(mask, tf.int32), -1, keepdims=True), tf.float32)
    batch_size = common_layers.shape_list(embeddings)[0]
    length = common_layers.shape_list(embeddings)[1]
    depth = common_layers.shape_list(embeddings)[2]
    embeddings = tf.where(
        tf.tile(tf.expand_dims(mask, 2), [1, 1, depth]), embeddings,
        tf.zeros_like(embeddings))
    if use_bigrams:
      if append_start_end:
        span_start_embed = tf.get_variable(name="span_start_embed",
                                           shape=[depth])
        span_end_embed = tf.get_variable(name="span_end_embed",
                                         shape=[depth])
        span_end_embed = tf.expand_dims(tf.expand_dims(span_end_embed, 0), 0)
        start = tf.expand_dims(
            tf.tile(tf.expand_dims(span_start_embed, 0), [batch_size, 1]), 1)
        # Prefix the start
        embeddings = tf.concat([start, embeddings], axis=1)
        # Pad for the end slot
        embeddings = tf.pad(embeddings, [[0, 0], [0, 1], [0, 0]])
        span_end_embed = tf.tile(span_end_embed, [batch_size, length + 2, 1])
        mask_with_start = tf.pad(
            tf.pad(tf.to_int32(mask), [[0, 0], [1, 0]],
                   constant_values=1), [[0, 0], [0, 1]],
            constant_values=0)
        mask_with_end = tf.pad(mask_with_start, [[0, 0], [1, 0]],
                               constant_values=1)[:, :-1]
        mask = tf.cast(mask_with_end, tf.bool)
        mask_of_end = tf.expand_dims(mask_with_end - mask_with_start, 2)
        embeddings = embeddings + span_end_embed * tf.to_float(mask_of_end)
      bigram_embeddings = tf.layers.dense(
          tf.concat([embeddings[:, :-1, :], embeddings[:, 1:, :]], axis=-1),
          units=depth)
      bigram_mask = tf.to_float(tf.expand_dims(mask[:, 1:], 2))
      masked_bigram_embeddings = bigram_embeddings * bigram_mask
      embeddings = tf.concat(
          [embeddings, masked_bigram_embeddings], axis=1)
      lengths = lengths + lengths - 1
    avg_embeddings = tf.div(tf.reduce_sum(embeddings, axis=1),
                            tf.maximum(lengths, 1.0))
  return avg_embeddings
