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

"""Defines architecture of the model networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn as contrib_rnn


def partial_sequence_encoder(
    features,
    symbolic_properties,
    numerical_points,
    num_production_rules,
    embedding_size):
  """Encodes partial sequence to embedding.

  Symbolic properties and numerical points will be concatenated to each
  production rule embedding.

  Args:
    features: Dict of tensors. This dict need to have:
        'partial_sequence': an int32 tensor with shape [batch_size, max_length].

        If symbolic_properties is not an empty list, features need to
        contain all the symbolic_property in symbolic_properties. Each tensor
        is a float32 tensor with shape [batch_size].

        If numerical_points is not an empty list, features need to
        contain key 'numerical_values': a float32 tensor with shape
        [batch_size, num_numerical_points].
    symbolic_properties: List of strings, symbolic properties to concatenate on
        embedding as conditions.
    numerical_points: List of floats, points to evaluate expression values.
    num_production_rules: Integer, the total number of production rules in
        grammar.
    embedding_size: Integer, the size of the embedding for each production rule.

  Returns:
    Float tensor with shape [batch_size, max_length, num_output_features].
    num_output_features = (
        embedding_size + len(symbolic_properties) + len(numerical_points))
  """
  with tf.variable_scope('partial_sequence_encoder'):
    partial_sequence = features['partial_sequence']
    max_length = partial_sequence.shape[1]
    one_hot_partial_sequence = tf.one_hot(
        partial_sequence,
        depth=num_production_rules,
        axis=-1,
        name='one_hot_partial_sequence')

    # Shape [batch_size, max_length, embedding_size].
    embedding_layer = tf.layers.dense(
        one_hot_partial_sequence,
        units=embedding_size,
        use_bias=False,
        name='embedding_layer')

    if symbolic_properties:
      condition_tensors = []
      for symbolic_property in symbolic_properties:
        condition_tensors.append(
            tf.expand_dims(
                tf.tile(
                    tf.expand_dims(features[symbolic_property], axis=1),
                    multiples=[1, max_length]),
                axis=2,
                name='symbolic_property_%s' % symbolic_property))
      # Shape [batch_size,
      #        max_length, embedding_size + num_symbolic_properties].
      embedding_layer = tf.concat([embedding_layer] + condition_tensors, axis=2)

    if numerical_points:
      numerical_values = tf.tile(
          tf.expand_dims(features['numerical_values'], axis=1),
          multiples=[1, max_length, 1],
          name='numerical_points')
      # Shape [batch_size,
      #        max_length, embedding_layer.shape[2] + num_numerical_points].
      embedding_layer = tf.concat([embedding_layer, numerical_values], axis=2)
    return embedding_layer


def build_stacked_gru_model(
    embedding_layer,
    partial_sequence_length,
    gru_hidden_sizes,
    num_output_features,
    bidirectional):
  """Predicts next production rule from partial sequence with stacked GRUs.

  Args:
    embedding_layer: Float32 tensor with shape
        [batch_size, max_length, num_features]. Input to the model.
    partial_sequence_length: Int32 tensor with shape [batch_size].
        This tensor is used for sequence_length in tf.nn.dynamic_rnn().
    gru_hidden_sizes: List of integers, number of units for each GRU layer.
    num_output_features: Integer, the number of output features.
    bidirectional: Boolean, whether to use bidirectional RNN.

  Returns:
    Float tensor with shape [batch_size, num_output_features]
  """
  with tf.variable_scope('stacked_gru_model'):
    gru_cells = [
        tf.nn.rnn_cell.GRUCell(gru_hidden_size)
        for gru_hidden_size in gru_hidden_sizes
    ]
    forward_stacked_gru = contrib_rnn.MultiRNNCell(gru_cells)
    if bidirectional:
      gru_cells = [
          tf.nn.rnn_cell.GRUCell(gru_hidden_size)
          for gru_hidden_size in gru_hidden_sizes
      ]
      backward_stacked_gru = contrib_rnn.MultiRNNCell(gru_cells)

      _, final_states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=forward_stacked_gru,
          cell_bw=backward_stacked_gru,
          inputs=embedding_layer,
          sequence_length=partial_sequence_length,
          dtype=embedding_layer.dtype,
          time_major=False)
      # final_states is a tuple of tuples:
      # (
      #     (forward_gru_0, forward_gru_1, ...),
      #     (backward_gru_0, backward_gru_1, ...)
      # )
      # Flatten the tuple as
      # (forward_gru_0, ..., backward_gru_0, ...)
      final_states = final_states[0] + final_states[1]
    else:
      _, final_states = tf.nn.dynamic_rnn(
          cell=forward_stacked_gru,
          inputs=embedding_layer,
          sequence_length=partial_sequence_length,
          dtype=embedding_layer.dtype,
          time_major=False)

    concat_final_states = tf.concat(
        final_states, axis=1, name='concatenate_gru_final_states')

    logits = tf.layers.dense(
        concat_final_states, num_output_features, name='logits')
    return logits
