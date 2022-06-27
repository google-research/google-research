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

"""The Tensorflow model function for the node-level field classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import logging
import numpy as np
from six.moves import reduce
from tensor2tensor.models import transformer
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_addons as tfa
from simpdom import block_lstm
from simpdom import seq_tagging_metric_util


tf.set_random_seed(42)


def _index_table_from_file(filepath, num_oov_buckets):
  return tf.lookup.StaticVocabularyTable(
      tf.lookup.TextFileInitializer(
          filepath,
          key_dtype=tf.string,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
          delimiter="\t"), num_oov_buckets)


def _bidirectional_lstm(lstm_input, lstm_size, sequence_length):
  """Get forward and backward lstm output for the given input."""
  lstm_cell_fw = block_lstm.LSTMBlockFusedCell(lstm_size)
  lstm_cell_bw = block_lstm.LSTMBlockFusedCell(lstm_size)
  lstm_cell_bw = block_lstm.TimeReversedFusedLSTM(lstm_cell_bw)
  output_fw, _ = lstm_cell_fw(
      lstm_input,
      dtype=tf.float32,
      sequence_length=sequence_length)
  output_bw, _ = lstm_cell_bw(
      lstm_input,
      dtype=tf.float32,
      sequence_length=sequence_length)

  return output_fw, output_bw


def masked_conv1d_and_max(t, weights, filters, kernel_size, reducemax=True):
  """Applies 1d convolution and a masked max-pooling.

  Args:
    t : tf.Tensor A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool A Tensor of shape [d1, d2, dn-1]
    filters : int number of filters
    kernel_size : int kernel size for the temporal convolution
    reducemax :  if reduce the result by max pooling.

  Returns:
    tf.Tensor: A tensor of shape [d1, d2, dn-1, filters]
  """
  # Get shape and parameters.
  shape = tf.shape(t)
  ndims = t.shape.ndims
  dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
  dim2 = shape[-2]
  dim3 = t.shape[-1]

  # Reshape weights.
  weights = tf.reshape(weights, shape=[dim1, dim2, 1])
  weights = tf.to_float(weights)

  # Reshape input and apply weights.
  flat_shape = [dim1, dim2, dim3]
  t = tf.reshape(t, shape=flat_shape)
  t *= weights

  # Apply convolution.
  t_conv = tf.layers.conv1d(t, filters, kernel_size, padding="same")
  t_conv *= weights

  if not reducemax:
    return t_conv

  # Reduce max -- set to zero if all padded.
  t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
  t_max = tf.reduce_max(t_conv, axis=-2)

  # Reshape the output
  final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
  t_max = tf.reshape(t_max, shape=final_shape)

  return t_max


def infer_shape(x):
  """Infers the shape of a dynamic tensor."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape.
  if x.shape.dims is None:
    return tf.shape(x)

  static_shape = x.shape.as_list()
  dynamic_shape = tf.shape(x)

  ret = []
  for i in range(len(static_shape)):
    dim = static_shape[i]
    if dim is None:
      dim = dynamic_shape[i]
    ret.append(dim)
  return ret


def merge_first_two_dims(tensor):
  """Merges the first two dimensions of a tf tensor."""
  shape = infer_shape(tensor)
  if len(shape) < 2:
    return tensor
  shape[0] *= shape[1]
  shape.pop(1)
  return tf.reshape(tensor, shape)


def split_first_two_dims_by_example(tensor, example_tensor):
  """Splits a tensor by its first two dimensions following the example."""
  example_shape = infer_shape(example_tensor)
  self_shape = infer_shape(tensor)
  new_shape = [example_shape[0]] + [example_shape[1]] + self_shape[1:]
  return tf.reshape(tensor, new_shape)


def transformer_encoding(node_seq_input, num_nodes, params, mode):
  """Construct a node-level encoder based on the transformer module.

  Args:
    node_seq_input : tf.Tensor. A tensor with 3 dimensions.
    num_nodes: tf.Tensor. Number of nodes per instance.
    params : dict. A parameter dictionary.
    mode : tf.estimator.ModeKeys object.

  Returns:
    node_seq_output: tf.Tensor. A tensor with 3 dimensions.
  """
  node_weights = tf.sequence_mask(num_nodes)

  hparams = transformer.transformer_tiny()
  hparams.hidden_size = params["transformer_hidden_unit"]
  hparams.num_heads = params["transformer_head"]
  hparams.num_hidden_layers = params["transformer_hidden_layer"]

  if hparams.hidden_size % hparams.num_heads != 0:
    raise ValueError("The hidden_size needs to be divisible by trans_head.")

  transformer_encoder = transformer.TransformerEncoder(hparams, mode=mode)
  # Input shape [batch_size, sequence_length, 1, hidden_dim].
  node_seq_input = tf.layers.dense(node_seq_input, hparams.hidden_size)
  node_seq_input_reshape = tf.expand_dims(node_seq_input, 2)
  # Targets and target_space_id are required by decoder of transformer,
  # are both set as 0 for encoder.
  node_seq_output = transformer_encoder(
      {
          "inputs": node_seq_input_reshape,
          "targets": 0,
          "target_space_id": 0,
      },
      nonpadding=node_weights)
  node_seq_output = tf.squeeze(node_seq_output[0], 2)
  # Construct a residue network by adding up the input and output
  node_seq_output = tf.add(node_seq_input, node_seq_output)
  return node_seq_output


def friend_attention(friends_embeddings, params):
  """Applies attention mechanism to optimize friend embeddings.

  Args:
    friends_embeddings: List of friend word-level embeddings.
    params: A parameter dictionary.

  Returns:
    friends_embeddings: List of friend node-level embeddings.
  """

  # Learn a context vector to compute attention weights on the word-level.
  context = tf.get_variable("context", [params["dim_word_embedding"], 1],
                            tf.float32)
  # Compute the normalized attention weights for each word.
  friends_attention = tf.nn.softmax(
      tf.matmul(friends_embeddings, context), axis=-2)
  # Each friend is composed of several words. Achieve an embedding for each
  # friend of each node.
  friends_embeddings = tf.reduce_sum(
      tf.math.multiply(friends_embeddings, friends_attention), axis=-2)
  # Learn a context vector to compute attention weights on the friend-level.
  sig = tf.get_variable("sig", [params["dim_word_embedding"], 1], tf.float32)
  friends_sig = tf.nn.softmax(tf.matmul(friends_embeddings, sig), axis=-2)
  # Each node has multiple friends. Achieve a friend embedding for each node.
  friends_embeddings = tf.reduce_sum(
      tf.math.multiply(friends_embeddings, friends_sig), axis=-2)
  # The dim of final friend embedding is a hyperparameter.
  friends_embeddings = tf.layers.dense(friends_embeddings,
                                       params["friend_hidden_size"])
  return friends_embeddings


def friend_self_attention(friends_embeddings, word_embeddings, params):
  """Applies self attention mechanism to optimize friend embeddings.

  Args:
    friends_embeddings: List of friend word-level embeddings.
    word_embeddings: The current node word-level embeddings.
    params: A parameter dictionary.

  Returns:
    friends_embeddings: List of friend node-level embeddings.
  """
  # Compute the average word embeddings to achieve a friend embedding.
  friends_embeddings = tf.reduce_mean(friends_embeddings, axis=-2)
  # Compute the normalized attention weights on the friend level.
  friends_attention = tf.nn.softmax(
      tf.matmul(friends_embeddings, word_embeddings), axis=-2)
  friends_embeddings = tf.reduce_sum(
      tf.math.multiply(friends_embeddings, friends_attention), axis=-2)
  # The dim of final friend embedding is a hyperparameter.
  friends_embeddings = tf.layers.dense(friends_embeddings,
                                       params["friend_hidden_size"])
  return friends_embeddings


def circle_feature_modeling(variable, vocab_words, partner_words, friends_words,
                            n_friends_words, friends_fix, friends_var,
                            word_embeddings, dropout, training, params):
  """Encodes partner and friends features."""

  # Partner Embeddings.
  partner_ids = vocab_words.lookup(partner_words)
  partner_embeddings = tf.nn.embedding_lookup(variable, partner_ids)
  logging.info("partner_embeddings.shape: %s", partner_embeddings.shape)
  partner_representation = tf.reduce_mean(partner_embeddings, 2)
  logging.info("partner_representation.shape: %s", partner_representation.shape)

  if params["circle_features"] == "partner":
    return partner_embeddings, partner_representation

  # Friends Embeddings.
  friends_ids = vocab_words.lookup(friends_words)
  friends_embeddings = tf.nn.embedding_lookup(variable, friends_ids)
  logging.info("friends_embeddings.shape: %s", friends_embeddings.shape)

  friends_fix_ids = vocab_words.lookup(friends_fix)
  friends_fix_embeddings = tf.nn.embedding_lookup(variable, friends_fix_ids)
  friends_var_ids = vocab_words.lookup(friends_var)
  friends_var_embeddings = tf.nn.embedding_lookup(variable, friends_var_ids)
  logging.info("friends_fix_embeddings1.shape: %s",
               friends_fix_embeddings.shape)

  friends_fix_embeddings = tf.layers.dense(
      friends_fix_embeddings, params["dim_word_embedding"], activation="relu")
  friends_var_embeddings = tf.layers.dense(
      friends_var_embeddings, params["dim_word_embedding"], activation="relu")

  logging.info("friends_fix_embeddings2.shape: %s",
               friends_fix_embeddings.shape)

  if params["friend_encoder"] == "cnn":
    friends_weights = tf.sequence_mask(n_friends_words)
    friends_representation = masked_conv1d_and_max(
        friends_embeddings,
        friends_weights,
        params["node_filters"],
        params["node_kernel_size"],
        reducemax=True)

  elif params["friend_encoder"] == "average":
    friends_embeddings = tf.layers.dense(friends_embeddings,
                                         params["dim_word_embedding"])
    friends_embeddings = tf.reduce_mean(friends_embeddings, 2)
    friends_representation = tf.layers.dense(friends_embeddings,
                                             params["friend_hidden_size"])
    friends_representation = tf.layers.dropout(
        friends_representation, rate=dropout, training=training)

  elif params["friend_encoder"] == "max":
    friends_embeddings = tf.layers.dense(friends_embeddings,
                                         params["dim_word_embedding"])
    friends_embeddings = tf.reduce_max(friends_embeddings, 2)
    friends_representation = tf.layers.dense(friends_embeddings,
                                             params["friend_hidden_size"])
    friends_representation = tf.layers.dropout(
        friends_representation, rate=dropout, training=training)

  elif params["friend_encoder"] == "attention":
    # Apply the attention mechanism to both friends_fix and friends_var.
    friends_fix_embeddings = friend_attention(friends_fix_embeddings, params)
    friends_var_embeddings = friend_attention(friends_var_embeddings, params)
    friends_representation = tf.concat(
        [friends_fix_embeddings, friends_var_embeddings], 2)

  elif params["friend_encoder"] == "self-attention":
    # Use the current node embedding as the context vector in the self attention
    # mechanism.
    word_embeddings = tf.expand_dims(
        tf.reduce_mean(word_embeddings, axis=-2), axis=-1)
    # Apply the self attention mechanism to both friends_fix and friends_var.
    friends_fix_embeddings = friend_self_attention(friends_fix_embeddings,
                                                   word_embeddings, params)
    friends_var_embeddings = friend_self_attention(friends_var_embeddings,
                                                   word_embeddings, params)
    friends_representation = tf.concat(
        [friends_fix_embeddings, friends_var_embeddings], 2)

  else:
    # Compute the average embeddings over all the words.
    friends_representation = tf.reduce_mean(friends_embeddings, 2)
  logging.info("friends_representation.shape: %s", friends_representation.shape)

  if params["circle_features"] == "friends":
    return partner_embeddings, friends_representation
  elif params["circle_features"] == "all":
    return partner_embeddings, tf.concat(
        [partner_representation, friends_representation], axis=2)
  else:
    return None


def xpath_lstm(params, head_node_xpath_embeddings, head_node_xpath_len_list,
               training):
  """Returns BiLSTM encoded results."""
  dropout = params["dropout"]
  h_t = merge_first_two_dims(head_node_xpath_embeddings)
  h_t = tf.transpose(h_t, perm=[1, 0, 2])  # Need time-major.
  h_output_fw, h_output_bw = _bidirectional_lstm(
      h_t, params["xpath_lstm_size"],
      merge_first_two_dims(head_node_xpath_len_list))
  h_output = tf.concat([h_output_fw, h_output_bw], axis=-1)
  h_output = tf.reduce_mean(h_output, 0)
  h_output = tf.layers.dropout(h_output, rate=dropout, training=training)
  print("output.shape (after reduce_mean):", h_output.shape, file=sys.stderr)
  h_output = split_first_two_dims_by_example(h_output,
                                             head_node_xpath_embeddings)
  print("output.shape (after split)", h_output.shape)
  return h_output


def xpath_feature_modeling(node_xpath_list, node_xpath_len_list, training,
                           params):
  """Loads xpath features and encode them with LSTM units."""
  vocab_xpath_units = _index_table_from_file(
      params["xpath_units"], num_oov_buckets=params["num_oov_buckets"])
  with tf.gfile.Open(params["xpath_units"]) as f:
    num_xpath_units = sum(1 for _ in f) + params["num_oov_buckets"]
  variable_xpath_unit = tf.get_variable(
      "xpath_unit_embeddings", [num_xpath_units + 1, params["dim_xpath_units"]],
      tf.float32)
  node_xpath_ids = vocab_xpath_units.lookup(node_xpath_list)

  node_xpath_embeddings = tf.nn.embedding_lookup(variable_xpath_unit,
                                                 node_xpath_ids)
  h_output = xpath_lstm(params, node_xpath_embeddings, node_xpath_len_list,
                        training)
  return h_output


def position_modeling(position_list, params):
  """Encodes absolute position of each node to the node embeddings."""

  vocab_positions = _index_table_from_file(
      params["positions"], num_oov_buckets=params["num_oov_buckets"])
  with tf.gfile.Open(params["positions"]) as f:
    num_positions = sum(1 for _ in f) + params["num_oov_buckets"]
  variable_positions = tf.get_variable(
      "position_embeddings", [num_positions + 1, params["dim_positions"]],
      tf.float32)
  node_pos_embs = tf.nn.embedding_lookup(variable_positions,
                                         vocab_positions.lookup(position_list))

  return node_pos_embs


def semantic_similarity(variable, vocab_words, partner_embeddings, labels,
                        params):
  """Computes the semantic similarity between partner and labels."""

  def normalize(v):
    return tf.math.divide_no_nan(
        v, tf.sqrt(tf.reduce_sum(tf.multiply(v, v), axis=-1, keep_dims=True)))

  partner_embeddings = tf.layers.dense(partner_embeddings,
                                       params["dim_word_embedding"])
  shapes = tf.shape(partner_embeddings)
  # Reshape the shape into three dims to facilitate the tf.matmul.
  partner_embeddings = tf.reshape(
      partner_embeddings, shape=[shapes[0], shapes[1] * shapes[2], shapes[3]])

  if params["semantic_encoder"] == "cos_sim_randomized":
    # Randomly initialize the label embeddings.
    with tf.gfile.Open(params["tags"]) as f:
      num_tags = sum(1 for _ in f) + params["num_oov_buckets"]
    vocab_tags = _index_table_from_file(
        params["tags"], num_oov_buckets=params["num_oov_buckets"])
    label_ids = vocab_tags.lookup(labels)
    variable = tf.get_variable(
        "label_embeddings", [num_tags + 1, params["dim_word_embedding"]],
        tf.float32,
        trainable=True)
    label_embeddings = tf.nn.embedding_lookup(variable, label_ids)
    label_embeddings = tf.layers.dense(label_embeddings,
                                       params["dim_word_embedding"])
    logging.info("label_embeddings.shape: %s", label_embeddings.shape)
    semantic_representation = tf.matmul(
        normalize(partner_embeddings),
        tf.transpose(normalize(label_embeddings), [0, 2, 1]))

  else:
    # Load label embeddings from GloVe.
    label_ids = vocab_words.lookup(labels)
    label_embeddings = tf.nn.embedding_lookup(variable, label_ids)
    logging.info("label_embeddings.shape: %s", label_embeddings.shape)
    if params["semantic_encoder"] == "inner_prod":
      semantic_representation = tf.matmul(
          partner_embeddings, tf.transpose(label_embeddings, [0, 2, 1]))
    elif params["semantic_encoder"] == "cos_sim":
      semantic_representation = tf.matmul(
          normalize(partner_embeddings),
          tf.transpose(normalize(label_embeddings), [0, 2, 1]))
  logging.info("semantic_representation.shape: %s",
               semantic_representation.shape)
  # Recover the original shape.
  last_dim = semantic_representation.shape[-1]
  semantic_representation = tf.reshape(
      semantic_representation,
      shape=[shapes[0], shapes[1], shapes[2], last_dim])
  # Select the highest similarity score over all the words in one friend.
  semantic_representation = tf.reduce_max(
      tf.transpose(semantic_representation, [0, 1, 3, 2]), axis=-1)
  return semantic_representation


def semantic_scorer(labels, node_embeddings, params):
  """Directly outputs the similarity scores as prediction logits."""

  def normalize(v):
    return tf.math.divide_no_nan(
        v, tf.sqrt(tf.reduce_sum(tf.multiply(v, v), axis=-1, keep_dims=True)))

  with tf.gfile.Open(params["tags"]) as f:
    num_tags = sum(1 for _ in f) + params["num_oov_buckets"]
  vocab_tags = _index_table_from_file(
      params["tags"], num_oov_buckets=params["num_oov_buckets"])
  tags_ids = vocab_tags.lookup(labels)
  # Randomly initialize the label embeddings.
  variable = tf.get_variable(
      "label_embeddings", [num_tags + 1, params["dim_label_embedding"]],
      tf.float32,
      trainable=True)
  label_embeddings = tf.nn.embedding_lookup(variable, tags_ids)
  logging.info("label_embeddings.shape: %s", label_embeddings.shape)
  node_embeddings = tf.layers.dense(node_embeddings,
                                    params["dim_label_embedding"])
  semantic_representation = tf.matmul(
      normalize(node_embeddings),
      tf.transpose(normalize(label_embeddings), [0, 2, 1]))

  logging.info("semantic_representation.shape: %s",
               semantic_representation.shape)
  return semantic_representation


def binary_scorer(labels, node_embeddings, training, params):
  """Conducts binary classification on node_emb plus label_emb."""

  if params["use_uniform_label"]:
    with tf.gfile.Open(params["tags-all"]) as f:
      num_tags = sum(1 for _ in f) + params["num_oov_buckets"]
  else:
    with tf.gfile.Open(params["tags"]) as f:
      num_tags = sum(1 for _ in f) + params["num_oov_buckets"]
  vocab_tags = _index_table_from_file(
      params["tags"], num_oov_buckets=params["num_oov_buckets"])
  tags_ids = vocab_tags.lookup(labels)
  # Randomly initialize the label embeddings.
  variable = tf.get_variable(
      "label_embeddings", [num_tags + 1, params["dim_label_embedding"]],
      tf.float32,
      trainable=True)
  label_embeddings = tf.nn.embedding_lookup(variable, tags_ids)

  label_embeddings = tf.layers.dense(
      label_embeddings, params["dim_label_embedding"], activation="relu")

  node_embeddings = tf.layers.dense(node_embeddings, 300, activation="relu")
  # node_embeddings: (page_num, node_num, emb_dim1).
  # label_embeddings: (page_num, label_num, emb_dim2).
  # concatenate -> (page_num, node_num, label_num, emb_dim1+emb_dim2).
  label_num = label_embeddings.shape[1]
  label_emb = label_embeddings.shape[2]
  shapes = tf.shape(node_embeddings)
  node_embeddings = tf.einsum("ijk,mk->ijmk", node_embeddings,
                              tf.ones([label_num, shapes[-1]]))
  label_embeddings = tf.einsum("ijk,km->imjk", label_embeddings,
                               tf.ones([label_emb, shapes[1]]))
  logging.info("label_embeddings.shape: %s", label_embeddings.shape)
  logging.info("node_embeddings.shape: %s", node_embeddings.shape)
  node_embeddings = tf.concat([node_embeddings, label_embeddings], axis=-1)

  node_embeddings = tf.layers.dense(
      node_embeddings, params["dim_word_embedding"], activation="relu")
  node_embeddings = tf.layers.dropout(
      node_embeddings, rate=params["dropout"], training=training)
  node_embeddings = tf.layers.dense(
      node_embeddings, params["dim_word_embedding"], activation="relu")
  node_embeddings = tf.layers.dropout(
      node_embeddings, rate=params["dropout"], training=training)
  node_embeddings = tf.layers.dense(node_embeddings,
                                    params["last_hidden_layer_size"])
  logits = tf.layers.dense(node_embeddings, 1)
  logits = tf.squeeze(logits, axis=-1)

  return logits


def joint_extraction_model_fn(features, labels, mode, params):
  """Runs the node-level sequence labeling model."""
  logging.info("joint_extraction_model_fn")
  inputs = features  # Arg "features" is the overall inputs.

  # Read vocabs and inputs.
  dropout = params["dropout"]
  if params["circle_features"]:
    nnodes, friend_has_label, (words, nwords), (
        prev_text_words,
        n_prev_text_words), (chars_list, chars_len_list), (partner_words, _), (
            friends_words, n_friends_words), (friends_fix, friends_var), (
                leaf_type_list, goldmine_feat_list), (_, _), (
                    node_xpath_list,
                    node_xpath_len_list), (attributes, attributes_plus_none), (
                        position_list) = inputs
  else:
    nnodes, (words, nwords), (prev_text_words, n_prev_text_words), (
        chars_list, chars_len_list), (leaf_type_list, goldmine_feat_list), (
            _, _), (node_xpath_list,
                    node_xpath_len_list), (attributes), (position_list) = inputs

  # nnodes, the number of nodes in each page;
  #    shape is [?]; length is the number of pages.
  # words, nwords are the node_text feature, shape is [?, ?, ?]
  #    the first two dimension is the batch * pages,
  #    the last one is the maximum length of the word lists
  # prev_text_words, n_prev_text_words, similar as above for previous nodes'text
  # chars_list, chars_len_list, shape is [?,?,?,?] also for node_text features
  #    the additional dim is for the length of the character sequences.
  # friends_words, shape is [?, ?, ?], gathers all the words from different
  #    friends of one node.
  # friends_fix, friends_var, shapes are [?, ?, ?, ?]
  #    the first two dimension is the batch * pages,
  #    the last two are the maximum length of friend nodes and words.

  nnodes = merge_first_two_dims(nnodes)
  training = (mode == tf_estimator.ModeKeys.TRAIN)
  vocab_words = _index_table_from_file(
      params["words"], num_oov_buckets=params["num_oov_buckets"])
  with tf.gfile.Open(params["tags"]) as f:
    indices = [idx for idx, tag in enumerate(f) if tag.strip() != "none"]
    num_tags = len(indices) + 1  # Make "None" as the tag with the last index.

  # NodeText Char Embeddings.
  with tf.gfile.Open(params["chars"]) as f:
    num_chars = sum(1 for _ in f) + params["num_oov_buckets"]
  vocab_chars = _index_table_from_file(
      params["chars"], num_oov_buckets=params["num_oov_buckets"])
  char_ids = vocab_chars.lookup(chars_list)
  variable = tf.get_variable("chars_embeddings",
                             [num_chars + 1, params["dim_chars"]], tf.float32)
  char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
  char_embeddings = tf.layers.dropout(
      char_embeddings, rate=dropout, training=training)
  logging.info("char_embeddings.shape: %s", char_embeddings.shape)
  # Char 1d convolution.
  weights = tf.sequence_mask(chars_len_list)
  char_embeddings = masked_conv1d_and_max(char_embeddings, weights,
                                          params["filters"],
                                          params["kernel_size"])
  logging.info("char_embeddings.shape after CNN: %s", char_embeddings.shape)

  # Word Embeddings.
  word_ids = vocab_words.lookup(words)
  glove = np.load(tf.gfile.Open(params["glove"],
                                "rb"))["embeddings"]  # np.array
  variable = np.vstack([glove, [[0.] * params["dim_word_embedding"]]])
  # To finetune the GloVe embedding by setting trainable as True.
  variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  logging.info("word_embeddings.shape: %s", word_embeddings.shape)

  # Prev_Text Representations.
  prev_text_word_ids = vocab_words.lookup(prev_text_words)
  prev_text_word_embeddings = tf.nn.embedding_lookup(variable,
                                                     prev_text_word_ids)
  if params["use_prev_text_lstm"]:
    # PREV_text LSTM.
    logging.info("prev_text_representation using lstm")

    prev_t = merge_first_two_dims(prev_text_word_embeddings)
    # Seq * batch * input
    prev_t = tf.transpose(prev_t, perm=[1, 0, 2])  # Need time-major.
    prev_output_fw, prev_output_bw = _bidirectional_lstm(
        prev_t, params["lstm_size"], merge_first_two_dims(n_prev_text_words))
    prev_output = tf.concat([prev_output_fw, prev_output_bw], axis=-1)
    prev_output = tf.reduce_mean(prev_output, 0)
    prev_output = tf.layers.dropout(
        prev_output, rate=dropout, training=training)
    logging.info("prev_output.shape (after reduce_mean): %s", prev_output.shape)
    context_representation = split_first_two_dims_by_example(
        prev_output, prev_text_word_embeddings)
    logging.info("context_representation.shape (after split): %s",
                 context_representation.shape)

  else:
    logging.info("prev_text_word_embeddings.shape: %s",
                 prev_text_word_embeddings.shape)
    context_representation = tf.reduce_mean(prev_text_word_embeddings, 2)
    logging.info("context_representation.shape: %s",
                 context_representation.shape)

  if params["circle_features"]:
    partner_embeddings, circle_representation = circle_feature_modeling(
        variable, vocab_words, partner_words, friends_words, n_friends_words,
        friends_fix, friends_var, word_embeddings, dropout, training, params)
    context_representation = circle_representation

    if params["use_friend_semantic"]:
      friends_ids = vocab_words.lookup(friends_words)
      friend_embeddings = tf.nn.embedding_lookup(variable, friends_ids)

  if params["use_xpath_lstm"]:
    h_output = xpath_feature_modeling(node_xpath_list, node_xpath_len_list,
                                      training, params)
    context_representation = tf.concat([h_output, context_representation],
                                       axis=2)

  if params["use_position_embedding"]:
    position_representation = position_modeling(position_list, params)
    context_representation = tf.concat(
        [context_representation, position_representation], axis=2)

  # Text Embeddings: Concatenate Word and Char and Feature Embeddings.
  embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
  embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

  logging.info("embeddings.shape: %s", embeddings.shape)

  # LSTM inside node texts.
  t = merge_first_two_dims(embeddings)
  t = tf.transpose(t, perm=[1, 0, 2])  # Need time-major.
  output_fw, output_bw = _bidirectional_lstm(t, params["lstm_size"],
                                             merge_first_two_dims(nwords))
  output = tf.concat([output_fw, output_bw], axis=-1)
  output = tf.reduce_mean(output, 0)
  output = tf.layers.dropout(output, rate=dropout, training=training)
  logging.info("output.shape (after reduce_mean): %s", output.shape)
  output = split_first_two_dims_by_example(output, embeddings)
  logging.info("output.shape (after split): %s", output.shape)

  node_seq_input = tf.concat([output, context_representation], axis=2)
  logging.info("output.shape (after + prev): %s", node_seq_input.shape)

  # Leaf Type Features.
  if params["add_leaf_types"]:
    with tf.gfile.Open(params["leaf_types"]) as f:
      num_leaf_types = sum(1 for _ in f) + params["num_oov_buckets"]
    vocab_leaf_types = _index_table_from_file(
        params["leaf_types"], num_oov_buckets=params["num_oov_buckets"])
    leaf_type_ids = vocab_leaf_types.lookup(leaf_type_list)
    leaf_variable = tf.get_variable(
        "leaf_type_embeddings", [num_leaf_types + 1, params["dim_leaf_type"]],
        tf.float32)
    leaf_type_embeddings = tf.nn.embedding_lookup(leaf_variable, leaf_type_ids)
    leaf_type_embeddings = tf.layers.dropout(
        leaf_type_embeddings, rate=dropout, training=training)
    logging.info("leaf_type_embeddings.shape: %s", char_embeddings.shape)
    logging.info("node_seq_input.shape before leaf: %s", node_seq_input.shape)
    node_seq_input = tf.concat([node_seq_input, leaf_type_embeddings], axis=2)
    logging.info("node_seq_input.shape after leaf: %s", node_seq_input.shape)

  # Goldmine Feat Embeddings.
  if params["add_goldmine"]:
    vocab_goldmine_features = _index_table_from_file(
        params["goldmine_features"], num_oov_buckets=1)
    goldmine_feature_variable = tf.get_variable("goldmine_feature_embeddings",
                                                [8 + 1, params["dim_goldmine"]],
                                                tf.float32)
    goldmine_feat_ids = vocab_goldmine_features.lookup(goldmine_feat_list)
    goldmine_feat_embeddings = tf.nn.embedding_lookup(goldmine_feature_variable,
                                                      goldmine_feat_ids)
    goldmine_feat_embeddings = tf.reduce_sum(goldmine_feat_embeddings, 2)
    logging.info("goldmine_feat_embeddings.shape: %s",
                 goldmine_feat_embeddings.shape)
    node_seq_input = tf.concat([node_seq_input, goldmine_feat_embeddings],
                               axis=2)
    logging.info("node_seq_input.shape after goldmine: %s",
                 node_seq_input.shape)

  # Node-level LSTM modeling.
  if params["node_encoder"] == "lstm":
    # Node-Sequence-LSTM.
    n_t = tf.transpose(node_seq_input, perm=[1, 0, 2])  # Need time-major.
    node_output_fw, node_output_bw = _bidirectional_lstm(
        n_t, params["node_lstm_size"], nnodes)
    node_seq_output = tf.concat([node_output_fw, node_output_bw], axis=-1)
    node_seq_output = tf.transpose(node_seq_output, perm=[1, 0, 2])
  elif params["node_encoder"] == "cnn":
    node_weights = tf.sequence_mask(nnodes)
    node_seq_output = masked_conv1d_and_max(
        node_seq_input,
        node_weights,
        params["node_filters"],
        params["node_kernel_size"],
        reducemax=False)
  elif params["node_encoder"] == "transformer":
    # Node-Sequence-Transformer.
    node_seq_output = transformer_encoding(node_seq_input, nnodes, params, mode)
  else:
    node_seq_output = node_seq_input

  logging.info("node_seq_input.shape after encoder: %s", node_seq_output.shape)

  if params["node_encoder"] != "transformer":
    # Add the dropout layer if the encoder is not a transformer.
    node_seq_output = tf.layers.dropout(
        node_seq_output, rate=dropout, training=training)

  if params["use_friends_discrete_feature"] and params["circle_features"]:
    friend_has_label = tf.expand_dims(friend_has_label, axis=-1)
    node_seq_output = tf.concat([node_seq_output, friend_has_label], axis=-1)
    logging.info("node_seq_input.shape after friend_has_label: %s",
                 node_seq_output.shape)
    node_seq_output = tf.layers.dense(node_seq_output,
                                      params["last_hidden_layer_size"])

  logits = tf.layers.dense(node_seq_output, num_tags, name="label_dense_1")

  if params["semantic_encoder"] and params["circle_features"]:

    partner_similarity_emb = semantic_similarity(variable, vocab_words,
                                                 partner_embeddings, attributes,
                                                 params)
    node_seq_output = tf.concat(
        [node_seq_output,
         tf.nn.softmax(partner_similarity_emb)], axis=-1)
    logging.info("node_seq_output.shape after semantic encoder: %s",
                 node_seq_output.shape)

    if params["use_friend_semantic"]:
      friends_similarity_emb = semantic_similarity(variable, vocab_words,
                                                   friend_embeddings,
                                                   attributes, params)

      node_seq_output = tf.concat([node_seq_output, friends_similarity_emb],
                                  axis=-1)

    if params["objective"] == "classification":
      node_seq_output = tf.layers.dense(
          node_seq_output, params["dim_word_embedding"], activation="relu")
      node_seq_output = tf.layers.dense(node_seq_output,
                                        params["last_hidden_layer_size"])
      logging.info("node_seq_output.shape after semantic encoder: %s",
                   node_seq_output.shape)
      logits = tf.layers.dense(node_seq_output, num_tags, name="label_dense_2")

    elif params["objective"] == "semantic_scorer":
      logits = semantic_scorer(attributes_plus_none, node_seq_output, params)

    elif params["objective"] == "binary_scorer":
      logits = binary_scorer(attributes_plus_none, node_seq_output, training,
                             params)

  if params["use_crf"]:
    # CRF Layer.
    logging.info("logits.shape: %s", logits.shape)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tfa.text.crf.crf_decode(logits, crf_params, nnodes)
    logging.info("pred_ids.shape: %s", pred_ids.shape)
  else:
    pred_ids = tf.argmax(logits, 2)
    logging.info("pred_ids.shape: %s", pred_ids.shape)
  # Predict for new sentences in target set.
  if mode == tf_estimator.ModeKeys.PREDICT:
    reverse_vocab_tags = _index_table_from_file(params["tags"], 1)
    pred_strings = reverse_vocab_tags.lookup(tf.strings.as_string(pred_ids))
    predictions = {
        "pred_ids": pred_ids,
        "tags": pred_strings,
        "scores": tf.nn.softmax(logits),
        "raw_scores": logits,
    }
    # Store the intermediate weights.
    if params["semantic_encoder"]:
      predictions["similarity"] = partner_similarity_emb
    if params["friend_encoder"]:
      predictions["friends_embs"] = circle_representation
    if params["extract_node_emb"]:
      predictions["node_embs"] = node_seq_output
    return tf_estimator.EstimatorSpec(mode, predictions=predictions)

  vocab_tags = _index_table_from_file(params["tags"], 1)
  tags = vocab_tags.lookup(labels)
  logging.info("tags.shape: %s", logits.shape)

  logging.info(
      "Parameter size: %s",
      np.sum(
          [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

  if params["use_crf"]:
    log_likelihood, _ = tfa.text.crf.crf_log_likelihood(logits, tags, nnodes,
                                                        crf_params)
    loss = tf.reduce_mean(-log_likelihood)
  else:
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tags, logits=logits)
  #  Processing the metrics.
  weights = tf.sequence_mask(nnodes)
  metrics = {
      "acc":
          tf.metrics.accuracy(tags, pred_ids, weights),
      "precision":
          seq_tagging_metric_util.precision(tags, pred_ids, num_tags, indices,
                                            weights),
      "recall":
          seq_tagging_metric_util.recall(tags, pred_ids, num_tags, indices,
                                         weights),
      "f1":
          seq_tagging_metric_util.f1(tags, pred_ids, num_tags, indices,
                                     weights),
  }
  for metric_name, op in metrics.items():
    tf.summary.scalar(metric_name, op[1])

  if mode == tf_estimator.ModeKeys.TRAIN:
    with tf.name_scope("train_scope"):
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(
          loss, global_step=tf.train.get_or_create_global_step())
    return tf_estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  return tf_estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=metrics)
