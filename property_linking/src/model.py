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

"""Model: Given string, predict desired set.

Simple version: Keep category attributes in.
Model should learn something like the identity
"""
import language.nql.nql as nql
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers

from property_linking.src import util
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import rnn as contrib_rnn

# Uses these global flags:
#   max_query_length
#   layer_size
#   num_layers
#   max_properties
#   logits
#   enforce_type
#   dropout
#   smoothing_param
#   weight_examples
#   weight_regularizer
FLAGS = tf.flags.FLAGS
IS_A = "i/P31"  # id of the "is-a" or "instance-of" relation in wikidata


class Similarity(object):
  """Biaffine (qAr) layer for predicting starting states.
  """

  def __init__(self, layer_size, embs, num_gpu):
    self._bert = embs
    self._num_gpu = num_gpu
    scorer = tf.get_variable(
        "start_scorer",
        shape=[layer_size, self._bert.shape[0]],
        initializer=tf.initializers.glorot_uniform,
        trainable=True)
    self._scorer = scorer

  def compute_logits(self, input_vec):
    """Returns a logits vector, possibly distributing across gpus.

    Args:
      input_vec: input vector to multiplication

    Returns:
      input_vec * self._scorer * self._bert, the logits of input_vec.
    """

    scaled_input = tf.matmul(input_vec, self._scorer)
    # The distribution in this section is
    # we reserve GPU:0 to handle other computations, e.g. embedding lookup,
    # softmax, losses, etc.
    num_shards = self._num_gpu - 1
    if num_shards > 0:
      sharded_outputs = [None] * num_shards
      shard_offset = 0
      # evenly sized shards
      size_per_shard = int(self._bert.shape[1]) + 1
      for shard_id in range(num_shards):
        with tf.device("/gpu:{}".format(shard_id + 1)):
          x = tf.matmul(
              scaled_input,
              self._bert[:, shard_offset:shard_offset + size_per_shard])
          shard_offset += size_per_shard
          sharded_outputs[shard_id] = x

      with tf.device("/gpu:0"):
        output = tf.concat(sharded_outputs, axis=1)
      return output
    else:
      return tf.matmul(scaled_input, self._bert)


class Prior(object):
  """Safely log sparse features.

  A Prior class is created to allow extensions in the future
  """

  def compute_logits(self, input_vec):
    # epsilon should be smaller than KB size but not 0 since that leads to nan
    return tf.log(input_vec + tf.constant(1e-6))


class Model(object):
  """Example model adapted from NQL Template-Free Gridworld.
  """

  def __init__(self,
               nqc,
               value_encodings,
               relation_encodings,
               num_gpus=1,
               encoder=None):
    """Builds a simple, fully-connected model to predict the outcome set given a query string.

    Args:
      nqc: NeuralQueryContext
      value_encodings: (bert features for values, length of value span)
      relation_encodings: (bert features for relations, length of relation span)
      num_gpus: number of gpus for distributed computation
      encoder: encoder (layers.RNN) for parameter sharing between train and dev

    Needs:
      self.input_ph: input to encoder (either one-hot or BERT layers)
      self.mask_ph: mask for the input
      self.correct_set_ph.name: target labels (if loss or accuracy is computed)
      self.prior_start: sparse matrix for string similarity features
      self.is_training: whether the model should is training (for dropout)

    Exposes:
      self.loss: objective for loss
      self.accuracy: mean accuracy metric (P_{predicted}(gold))
      self.accuracy_per_ex: detailed per example accuracy
      self.log_nql_pred_set: predicted entity set (in nql)
      self.log_decoded_relations: predicted relations (as indices)
      self.log_start_values: predicted start values (in nql)
      self.log_start_cmps: components of predicted start values (in nql)
    """
    # Encodings should have the same dimensions
    assert value_encodings[0].shape[-1] == relation_encodings[0].shape[-1]
    self.context = nqc
    self.input_ph = tf.placeholder(
        tf.float32,
        shape=(None, FLAGS.max_query_length, value_encodings[0].shape[-1]),
        name="oh_seq_ph"
    )
    self.mask_ph = tf.placeholder(
        tf.float32,
        shape=(None, FLAGS.max_query_length),
        name="oh_mask_ph"
    )
    self.debug = None
    layer_size = FLAGS.layer_size
    num_layers = FLAGS.num_layers
    max_properties = FLAGS.max_properties
    logits_strategy = FLAGS.logits
    dropout_rate = FLAGS.dropout

    inferred_batch_size = tf.shape(self.input_ph)[0]
    self.is_training = tf.placeholder(tf.bool, shape=[])
    value_tensor = util.reshape_to_tensor(value_encodings[0],
                                          value_encodings[1])
    relation_tensor = util.reshape_to_tensor(relation_encodings[0],
                                             relation_encodings[1])
    # The last state of LSTM encoder is the representation of the input string
    with tf.variable_scope("model"):
      # Build all the model parts:

      #   encoder: LSTM encoder
      #   prior: string features
      #   {value, relation}_similarity: learned embedding similarty
      #   decoder: LSTM decoder
      #   value_model: map from encoder to key for attention
      #   attention: Luong (dot product) attention

      # Builds encoder - note that this is in keras
      self.encoder = self._build_encoder(encoder, layer_size, num_layers)

      # Build module to turn prior (string features) into logits
      self.prior_start = tf.sparse.placeholder(
          tf.float32,
          name="prior_start_ph",
          shape=[inferred_batch_size, value_tensor.shape[1]]
      )

      with tf.variable_scope("prior"):
        prior = Prior()

      # Build similarity module - biaffine qAr
      with tf.variable_scope("value_similarity"):
        value_similarity = Similarity(layer_size, value_tensor, num_gpus)
      # Build relation decoder
      with tf.variable_scope("relation_decoder"):
        rel_dec_rnn_layers = [
            contrib_rnn.LSTMBlockCell(layer_size, name=("attr_lstm_%d" % i))
            for (i, layer_size) in enumerate([layer_size] * num_layers)
        ]
        relation_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(rel_dec_rnn_layers)
        tf.logging.info("relation decoder lstm has state of size: {}".format(
            relation_decoder_cell.state_size))

      # Build similarity module - biaffine qAr
      with tf.variable_scope("relation_similarity"):
        relation_similarity = Similarity(layer_size, relation_tensor, 1)
      with tf.variable_scope("attention"):
        attention = layers.Attention()
      value_model = tf.get_variable(
          "value_transform",
          shape=[layer_size, relation_decoder_cell.output_size],
          trainable=True)

    # Initialization for logging, variables shouldn't be used elsewhere
    log_decoded_starts = []
    log_start_logits = []
    log_decoded_relations = []

    # Initialization to prepare before first iteration of loop
    prior_logits_0 = prior.compute_logits(tf.sparse.to_dense(self.prior_start))
    cumulative_entities = nqc.all("id_t")
    relation_decoder_out = tf.zeros([inferred_batch_size, layer_size])
    encoder_output = self.encoder(self.input_ph, mask=self.mask_ph)
    query_encoder_out = encoder_output[0]
    relation_decoder_state = encoder_output[1:]

    # Initialization for property loss, equal to log vars but separating
    value_dist = []
    relation_dist = []

    for i in range(max_properties):
      prior_logits = tf.layers.dropout(prior_logits_0, rate=dropout_rate,
                                       training=self.is_training)
      # Use the last state to determine key; more stable than last output
      query_key = tf.nn.relu(
          tf.matmul(tf.expand_dims(relation_decoder_state[-1][-1],
                                   axis=1),
                    value_model))

      query_emb = tf.squeeze(attention(
          [query_key, query_encoder_out],
          mask=[None, tf.cast(self.mask_ph, tf.bool)]),
                             axis=1)

      similarity_logits = value_similarity.compute_logits(query_emb)
      if logits_strategy == "prior":
        total_logits = prior_logits
      elif logits_strategy == "sim":
        total_logits = similarity_logits
      elif logits_strategy == "mixed":
        total_logits = prior_logits + similarity_logits
      total_dist = contrib_layers.softmax(total_logits)
      values_pred = nqc.as_nql(total_dist, "val_g")
      with tf.variable_scope("start_follow_{}".format(i)):
        start_pred = nqc.all("v_t").follow(values_pred)  # find starting nodes

      # Given the previous set of attributes, where are we going?
      (relation_decoder_out,
       relation_decoder_state) = relation_decoder_cell(
           relation_decoder_out,
           relation_decoder_state)
      pred_relation = tf.nn.softmax(
          relation_similarity.compute_logits(relation_decoder_out))
      if FLAGS.enforce_type:
        if i == 0:
          is_adjust = nqc.as_tf(nqc.one(IS_A, "rel_g"))
        else:
          is_adjust = 1 - nqc.as_tf(nqc.one(IS_A, "rel_g"))
        pred_relation = pred_relation * is_adjust
      nql_pred_relation = nqc.as_nql(pred_relation, "rel_g")
      # Conjunctive (& start.follow() & start.follow()...).
      with tf.variable_scope("relation_follow_{}".format(i)):
        current_entities = start_pred.follow(nql_pred_relation)
      cumulative_entities = cumulative_entities & current_entities

      # For property loss and regularization
      value_dist.append(total_dist)
      relation_dist.append(pred_relation)

      # Store predictions for logging
      log_decoded_starts.append(start_pred)
      log_decoded_relations.append(pred_relation)
      log_start_logits.append([prior_logits, similarity_logits])

    (loss,
     pred_set_tf,
     pred_set_tf_norm) = self._compute_loss(cumulative_entities)
    property_loss = self._compute_property_loss(value_dist, relation_dist)
    (accuracy_per_ex,
     accuracy) = self._compute_accuracy(cumulative_entities, pred_set_tf)
    value_loss = self._compute_distribution_regularizer(value_dist)
    relation_loss = self._compute_distribution_regularizer(relation_dist)
    self.regularization = FLAGS.time_reg * (value_loss + relation_loss)
    self.loss = loss - self.regularization
    self.property_loss = property_loss
    self.accuracy_per_ex = accuracy_per_ex
    self.accuracy = accuracy

    # Debugging/logging information
    log_decoded_relations = tf.transpose(tf.stack(log_decoded_relations),
                                         [1, 0, 2])
    tf.logging.info("decoded relations has shape: {}".format(
        log_decoded_relations.shape))
    self.log_start_values = log_decoded_starts
    self.log_start_cmps = [[nqc.as_nql(logits, "val_g") for logits in comp]
                           for comp in log_start_logits]
    self.log_decoded_relations = tf.nn.top_k(log_decoded_relations, k=5)
    self.log_nql_pred_set = nqc.as_nql(pred_set_tf_norm, "id_t")

  def _build_encoder(self, encoder, layer_size, num_layers):
    if encoder is None:
      enc_rnn_cells = [
          layers.LSTMCell(layer_size, name=("enc_lstm_%d" % i))
          for (i, layer_size) in enumerate([layer_size] * num_layers)
      ]
      encoder = layers.RNN(enc_rnn_cells,
                           return_state=True,
                           return_sequences=True)
    return encoder

  def _compute_loss(self, cumulative_entities):
    """Compute loss.

    Args:
      cumulative_entities: predicted entity distribution

    Returns:
      negative log probability of targets under the predicted distribution
    """
    # Smoothing param to prevent underflow, might also be useful in optimization
    smoothing_param = tf.cond(self.is_training,
                              lambda: FLAGS.smoothing_param,
                              lambda: 0.0)

    # Compute distribution of predicted set
    pred_set_tf = self.context.as_tf(cumulative_entities)
    pred_denominator = (tf.reduce_sum(pred_set_tf, 1) + smoothing_param)
    pred_set_tf_norm = pred_set_tf / tf.expand_dims(pred_denominator, 1)

    # Compute distribution of target set; assumes nonempty
    self.correct_set_ph = self.context.placeholder("correct_ids", "id_t")
    targets = self.correct_set_ph.tf / (tf.expand_dims(tf.math.reduce_sum(
        self.correct_set_ph.tf, axis=1), -1))

    # Need to add regularization
    if FLAGS.weight_examples:
      loss = util.weighted_nonneg_crossentropy(pred_set_tf_norm,
                                               targets,
                                               pred_denominator)
      loss -= (FLAGS.weight_regularizer *
               tf.math.reduce_mean(tf.minimum(1.0, pred_denominator), 0))
    else:
      loss = util.weighted_nonneg_crossentropy(pred_set_tf_norm,
                                               targets)
    return (loss, pred_set_tf, pred_set_tf_norm)

  def _compute_property_loss(self, value_dist, relation_dist):
    """Compute property loss directly.

    Args:
      value_dist: distribution over values per timestep
      relation_dist: distribution over relations per timestep

    Returns:
      negative log probability of correct property
    """

    self.correct_vals = self.context.placeholder("correct_vals", "val_g")
    self.correct_rels = self.context.placeholder("correct_rels", "rel_g")
    target_vals = self.correct_vals.tf
    target_rels = self.correct_rels.tf
    all_vals = tf.reduce_sum(tf.stack(value_dist, axis=1), 1)
    all_rels = tf.reduce_sum(tf.stack(relation_dist, axis=1), 1)
    property_loss = (
        nql.nonneg_crossentropy(all_vals, target_vals) +
        nql.nonneg_crossentropy(all_rels, target_rels)
    )
    return property_loss

  def _compute_accuracy(self, cumulative_entities, pred_set_tf):
    """Compute accuracy. P(gold) under predicted distribution.

    Not the correct formulation of accuracy, but an approximate
    measure for overlap.

    Args:
      cumulative_entities: predicted entity distribution
      pred_set_tf: predicted set in tensorflow

    Returns:
      precision, recall, and f1
    """
    accuracy_set = self.correct_set_ph * cumulative_entities
    unsmoothed_pred_denominator = tf.reduce_sum(pred_set_tf, 1)

    # Compute per example accuracy (useful for explicit debugging)
    accuracy_per_ex = (tf.reduce_sum(self.context.as_tf(accuracy_set), 1) /
                       unsmoothed_pred_denominator)
    accuracy = tf.reduce_mean(accuracy_per_ex)
    return (accuracy_per_ex, accuracy)

  def _compute_distribution_regularizer(self, dist):
    """Compute a regularizer over the distributions per timestep.

    Args:
      dist: a list of distributions per timestep

    Returns:
      sum of L1 distances per timestep, averaged over batch dim
    """
    total_distance = []
    for i in range(len(dist) - 1):
      # computes distance of i to the next timestep
      total_distance.append(tf.reduce_sum(tf.abs(dist[i] - dist[i + 1]), -1))
    return tf.reduce_mean(tf.reduce_mean(tf.stack(total_distance, axis=1), 1))
