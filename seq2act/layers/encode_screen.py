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

"""Utils for encoding a UI screen."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
import tensorflow.compat.v1 as tf
from seq2act.layers import common_embed


def prepare_encoder_input(features, hparams, embed_scope=None,
                          embed_token_fn=common_embed.embed_tokens):
  """Prepares the input for the screen encoder.

  Args:
    features: the feature dict.
    hparams: the hyperparameter.
    embed_scope: the embedding variable scope.
    embed_token_fn: the function for embedding tokens.
  Returns:
    object_embedding: a Tensor of shape
        [batch_size, num_steps, max_object_count, embed_depth]
    object_mask: a binary tensor of shape
        [batch_size, num_steps, max_object_count]
    nonpadding_bias: a Tensor of shape
        [batch_size, num_steps, max_object_count]
  """
  with tf.control_dependencies([
      tf.assert_equal(tf.rank(features["obj_text"]), 4)]):
    if hparams.get("synthetic_screen_noise", 0.) > 0.:
      num_objects = tf.shape(features["obj_text"])[2]
      # [batch, length, num_objects]
      target_obj_mask = tf.cast(
          tf.one_hot(features["objects"], depth=num_objects), tf.bool)
      num_tokens = tf.shape(features["obj_text"])[-1]
      target_obj_mask = tf.tile(
          tf.expand_dims(target_obj_mask, 3),
          [1, 1, 1, num_tokens])
      # Randomly keep tokens
      keep_mask = tf.greater_equal(
          tf.random_uniform(shape=tf.shape(features["obj_text"])),
          hparams.synthetic_screen_noise)
      # Keep paddings
      keep_mask = tf.logical_or(tf.equal(features["obj_text"], 0),
                                keep_mask)
      # Keep targets
      target_obj_mask = tf.logical_or(target_obj_mask, keep_mask)
      features["obj_text"] = tf.where(
          target_obj_mask, features["obj_text"],
          tf.random_uniform(shape=tf.shape(features["obj_text"]),
                            maxval=50000, dtype=tf.int32))
    text_embeddings, _ = embed_token_fn(
        features["obj_text"],
        hparams.task_vocab_size,
        hparams.hidden_size, hparams,
        embed_scope=embed_scope)
    with tf.variable_scope("obj_text_embed", reuse=tf.AUTO_REUSE):
      if hparams.obj_text_aggregation == "max":
        embed_bias = tf.cast(tf.less(features["obj_text"], 2),
                             tf.float32) * -1e7
        with tf.control_dependencies([tf.assert_equal(tf.rank(embed_bias), 4)]):
          text_embeddings = tf.reduce_max(
              text_embeddings + tf.expand_dims(embed_bias, 4), -2)
          no_txt_embed = tf.get_variable(
              name="no_txt_embed", shape=[hparams.hidden_size])
          shape = common_layers.shape_list(text_embeddings)
          no_txt_embed = tf.tile(
              tf.reshape(no_txt_embed, [1, 1, 1, hparams.hidden_size]),
              [shape[0], shape[1], shape[2], 1])
          text_embeddings = tf.maximum(text_embeddings, no_txt_embed)
      elif hparams.obj_text_aggregation == "sum":
        # [batch, step, #max_obj, #max_token]  0 for padded tokens
        real_objects = tf.cast(
            tf.greater_equal(features["obj_text"], 2), tf.float32)
        # [batch, step, #max_obj, hidden]   0s for padded objects
        text_embeddings = tf.reduce_sum(
            text_embeddings * tf.expand_dims(real_objects, 4), -2)
      elif hparams.obj_text_aggregation == "mean":
        shape_list = common_layers.shape_list(text_embeddings)
        embeddings = tf.reshape(text_embeddings, [-1] + shape_list[3:])
        emb_sum = tf.reduce_sum(tf.abs(embeddings), axis=-1)
        non_paddings = tf.not_equal(emb_sum, 0.0)
        embeddings = common_embed.average_bag_of_embeds(
            embeddings, non_paddings, use_bigrams=True,
            bigram_embed_scope=embed_scope, append_start_end=True)
        text_embeddings = tf.reshape(
            embeddings, shape_list[:3] + [hparams.hidden_size])
      else:
        raise ValueError("Unrecognized token aggregation %s" % (
            hparams.obj_text_aggregation))
  with tf.control_dependencies([
      tf.assert_equal(tf.rank(features["obj_type"]), 3),
      tf.assert_equal(tf.rank(features["obj_clickable"]), 3)]):
    with tf.variable_scope("encode_object_attr", reuse=tf.AUTO_REUSE):
      type_embedding = tf.nn.embedding_lookup(
          params=tf.get_variable(
              name="embed_type_w", shape=[hparams.get("num_types", 100),
                                          hparams.hidden_size]),
          ids=tf.maximum(features["obj_type"], 0))
      clickable_embedding = tf.nn.embedding_lookup(
          params=tf.get_variable(
              name="embed_clickable_w", shape=[2, hparams.hidden_size]),
          ids=features["obj_clickable"])
  with tf.control_dependencies([
      tf.assert_equal(tf.rank(features["obj_screen_pos"]), 4)]):
    def _create_embed(feature_name, vocab_size, depth):
      """Embed a position feature."""
      pos_embedding_list = []
      with tf.variable_scope("encode_object_" + feature_name,
                             reuse=tf.AUTO_REUSE):
        num_featues = common_layers.shape_list(features[feature_name])[-1]
        for i in range(num_featues):
          pos_embedding_list.append(tf.nn.embedding_lookup(
              params=tf.get_variable(
                  name=feature_name + "_embed_w_%d" % i,
                  shape=[vocab_size, depth]),
              ids=features[feature_name][:, :, :, i]))
        pos_embedding = tf.add_n(pos_embedding_list)
        return pos_embedding
    pos_embedding = _create_embed("obj_screen_pos",
                                  hparams.max_pixel_pos,
                                  hparams.hidden_size)
  if "all" == hparams.screen_embedding_feature or (
      "dom" in hparams.screen_embedding_feature):
    dom_embedding = _create_embed("obj_dom_pos",
                                  hparams.max_dom_pos,
                                  hparams.hidden_size)
  object_embed = tf.zeros_like(text_embeddings, dtype=tf.float32)
  if hparams.screen_embedding_feature == "all":
    object_embed = (
        text_embeddings + type_embedding + pos_embedding + dom_embedding)
  elif "text" in hparams.screen_embedding_feature:
    object_embed += text_embeddings
  elif "type" in hparams.screen_embedding_feature:
    object_embed += type_embedding
  elif "pos" in hparams.screen_embedding_feature:
    object_embed += pos_embedding
  elif "dom" in hparams.screen_embedding_feature:
    object_embed += dom_embedding
  elif "click" in hparams.screen_embedding_feature:
    object_embed += clickable_embedding
  object_mask = tf.cast(tf.not_equal(features["obj_type"], -1), tf.float32)
  object_embed = object_embed * tf.expand_dims(object_mask, 3)
  att_bias = (1. - object_mask) * common_attention.large_compatible_negative(
      object_embed.dtype)
  return object_embed, object_mask, att_bias


def transformer_encoder(features, hparams,
                        embed_scope=None,
                        embed_token_fn=common_embed.embed_tokens,
                        attention_weights=None):
  """Encodes a screen using Transformer.

  Args:
    features: the feature dict.
    hparams: the hyperparameter.
    embed_scope: the scope for token embedding.
    embed_token_fn: the embed function.
    attention_weights: the attention_weights dict.
  Returns:
    encoder_outputs: a Tensor of shape
        [batch_size, num_steps, max_object_count, hidden_size]
    encoder_attn_bias: A tensor of shape
        [batch_size, num_steps, max_object_count]
  """
  tf.logging.info("Using Transformer screen encoder")
  # Remove the default positional encoding in Transformer
  object_embed, object_mask, encoder_attn_bias = prepare_encoder_input(
      features=features, hparams=hparams, embed_scope=embed_scope,
      embed_token_fn=embed_token_fn)
  with tf.variable_scope("encode_screen", reuse=tf.AUTO_REUSE):
    shape = tf.shape(object_embed)
    with tf.control_dependencies([
        tf.assert_equal(shape[3], hparams.hidden_size)]):
      object_embed = tf.reshape(object_embed,
                                [shape[0] * shape[1], shape[2],
                                 hparams.hidden_size])
    encoder_input = tf.nn.dropout(
        object_embed,
        keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
    self_attention_bias = tf.expand_dims(tf.expand_dims(
        tf.reshape(encoder_attn_bias, [shape[0] * shape[1], shape[2]]),
        axis=1), axis=1)
    encoder_output = transformer.transformer_encoder(
        encoder_input=encoder_input,
        encoder_self_attention_bias=self_attention_bias,
        hparams=hparams,
        save_weights_to=attention_weights,
        make_image_summary=not common_layers.is_xla_compiled())
    encoder_output = tf.reshape(encoder_output,
                                [shape[0], shape[1], shape[2], shape[3]])
    return encoder_output, object_mask, encoder_attn_bias


def gcn_encoder(features, hparams, embed_scope,
                embed_token_fn=common_embed.embed_tokens,
                adjcency_feature="obj_dom_dist",
                discretize=True):
  """Encodes a screen using Graph Convolution Networks.

  Args:
    features: the feature dict.
    hparams: the hyperparameter.
    embed_scope: the variable scope for token embedding.
    embed_token_fn: the embed function.
    adjcency_feature: the feature name for the adjacency matrix.
    discretize: whether to discretize the matrix.
  Returns:
    encoder_outputs: a Tensor of shape
        [batch_size, num_steps, max_object_count, hidden_size]
    encoder_attn_bias: A tensor of shape
        [batch_size, num_steps, max_object_count]
  """
  tf.logging.info("Using GCN screen encoder")
  # [batch_size, num_steps, max_num_objects, depth]
  inputs, object_mask, encoder_attn_bias = prepare_encoder_input(
      features=features, hparams=hparams, embed_scope=embed_scope,
      embed_token_fn=embed_token_fn)
  # [batch_size, num_steps, max_num_objects, max_num_objects]
  if discretize:
    adjacency_matrix = tf.cast(tf.where(
        tf.greater(features[adjcency_feature], 1),
        tf.zeros_like(features[adjcency_feature]),
        tf.ones_like(features[adjcency_feature])), tf.float32)
  else:
    adjacency_matrix = tf.cast(features[adjcency_feature], tf.float32)
    dom_dist_variance = 0.1
    numerator = tf.exp(
        adjacency_matrix * adjacency_matrix / (-2.0 * dom_dist_variance))
    denominator = tf.sqrt(2.0 * 3.141 * dom_dist_variance)
    adjacency_matrix = numerator / denominator
  encoder_outputs = graph_cnn(inputs, object_mask, hparams.num_hidden_layers,
                              hparams.hidden_size,
                              dropout=hparams.layer_prepostprocess_dropout,
                              adjacency_matrix=adjacency_matrix,
                              norm_type=hparams.norm_type,
                              norm_epsilon=hparams.norm_epsilon)
  return encoder_outputs, object_mask, encoder_attn_bias


def graph_cnn(inputs, object_mask, num_layers, hidden_size, dropout,
              adjacency_matrix, norm_type="layer", norm_epsilon=0.001,
              test=False):
  """Encodes a screen using Graph Convolution Networks.

  Args:
    inputs: [batch_size, num_steps, max_object_count, depth].
    object_mask: [batch_size, num_steps, max_object_count].
    num_layers: the number of layers.
    hidden_size: the hidden layer size.
    dropout: dropout ratio.
    adjacency_matrix: the adjacency matrix
        [batch_size, num_steps, max_object_count, max_object_count].
    norm_type: the norm_type.
    norm_epsilon: norm_epsilon.
    test: whether it's in the test mode.
  Returns:
    hidden: a Tensor of shape
        [batch_size, num_steps, max_object_count, depth]
  """
  # [batch_size, num_steps, max_num_objects, max_num_objects]
  normalizer = tf.div(1., tf.sqrt(tf.reduce_sum(
      adjacency_matrix, -1, keepdims=True)))
  normalizer = normalizer * tf.expand_dims(tf.expand_dims(
      tf.eye(tf.shape(normalizer)[-2]), 0), 0)
  adjacency_matrix = tf.matmul(
      tf.matmul(normalizer, adjacency_matrix), normalizer)
  hidden = inputs
  for layer in range(num_layers):
    with tf.variable_scope("gcn_layer_" + str(layer), reuse=tf.AUTO_REUSE):
      hidden = tf.matmul(adjacency_matrix, hidden)
      # [batch_size, num_steps, max_num_objects, depth]
      if not test:
        hidden = tf.layers.dense(inputs=hidden, units=hidden_size)
        hidden = common_layers.apply_norm(
            hidden, norm_type, hidden_size,
            epsilon=norm_epsilon)
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=1.0 - dropout)
      # zero out padding objects
      hidden = hidden * tf.expand_dims(object_mask, 3)
  return hidden
