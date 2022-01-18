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

"""Utility functions for constructing networks."""

import tensorflow as tf


def _merge_inputs(shape, name, merged_embedding, inputs):
  rnn_input = tf.keras.layers.Input(shape=shape, name=name)
  merged_embedding = tf.keras.layers.concatenate([merged_embedding, rnn_input],
                                                 axis=-1)
  inputs.append(rnn_input)
  return merged_embedding, inputs


def construct_creator_rnn_inputs(document_feature_size=10,
                                 creator_feature_size=None,
                                 num_creators=None,
                                 creator_id_embedding_size=0,
                                 trajectory_length=None):
  """Return creator RNN inputs.

  Creator RNN input at each time step is the current time step's
  [#recommendations, #user-clicks, summed user_reward,
  weighted_clicked_document_topics, creator_current_feature], where
  'weighted_clicked_document_topics` has length document_feature_size, and is a
  weighted average of user_clicked document topics weighted by user_reward;
  `creator_current_feature` is creator current context such as the creator
  observed saturated satisfaction if creator_feature_size is not None.

  Args:
    document_feature_size: Integer, length of document features, which is the
      number of topics on the platform.
    creator_feature_size: Integer or None, length of creator features. If None,
      no features about creators will be used as input.
    num_creators: Integer, used for embedding creator id.
    creator_id_embedding_size: Integer, if greater than zero, embed creator id.
    trajectory_length: Integer, specify the trajectory length.

  Returns:
    rnn_inputs: Keras input layer of shape (episode_length,
    3 + document_feature_size+creator_feature_size)
  """
  number_recommendations = tf.keras.layers.Input(
      shape=(trajectory_length, 1), name='creator_number_recommendations')
  number_clicks = tf.keras.layers.Input(
      shape=(trajectory_length, 1), name='creator_number_clicks')
  creator_rewards = tf.keras.layers.Input(
      shape=(trajectory_length, 1), name='creator_user_rewards')
  weighted_clicked_doc_topics = tf.keras.layers.Input(
      shape=(trajectory_length, document_feature_size),
      name='creator_weighted_clicked_doc_topics')
  merged_embedding = tf.keras.layers.concatenate([
      number_recommendations, number_clicks, creator_rewards,
      weighted_clicked_doc_topics
  ],
                                                 axis=-1)
  inputs = [
      number_recommendations, number_clicks, creator_rewards,
      weighted_clicked_doc_topics
  ]
  if creator_feature_size is not None:
    merged_embedding, inputs = _merge_inputs(
        (trajectory_length, creator_feature_size), 'creator_current_feature',
        merged_embedding, inputs)
  if creator_id_embedding_size > 0:
    creator_id_inputs = tf.keras.layers.Input(
        shape=(trajectory_length), name='creator_id')
    creator_embeddings = tf.keras.layers.Embedding(
        input_dim=num_creators + 1,  # One additional token for padding.
        output_dim=creator_id_embedding_size,
        mask_zero=False,
        name='creator_id_embedding_layer')(
            creator_id_inputs)
    inputs.append(creator_id_inputs)
    merged_embedding = tf.keras.layers.concatenate(
        [merged_embedding, creator_embeddings], axis=-1)

  return merged_embedding, inputs


def construct_user_rnn_inputs(document_feature_size=10,
                              creator_feature_size=None,
                              user_feature_size=None,
                              input_reward=False):
  """Returns user RNN inputs.

  Args:
    document_feature_size: Integer, length of document features.
    creator_feature_size: Integer or None, length of creator features. If None,
      no features about creators will be input.
    user_feature_size: Integer or None, length of user features. If None, no
      features about users will be input.
    input_reward: Boolean, whether to input previous reward to RNN layer.
  """

  # Previous consumed document.
  rnn_input_doc_feature = tf.keras.layers.Input(
      shape=(None, document_feature_size), name='user_consumed_doc_feature')
  merged_embedding = rnn_input_doc_feature
  inputs = [rnn_input_doc_feature]
  # Previous consumed document-associated creator.
  if creator_feature_size is not None:
    # This vector includes creator's observable features and/or creator's hidden
    # states inferred by creator model.
    merged_embedding, inputs = _merge_inputs(
        (None, creator_feature_size), 'user_consumed_doc-creator_feature',
        merged_embedding, inputs)

  # User current context.
  if user_feature_size is not None:
    merged_embedding, inputs = _merge_inputs(
        (None, user_feature_size), 'user_current_feature', merged_embedding,
        inputs)
  # Previous reward.
  if input_reward:
    merged_embedding, inputs = _merge_inputs((None, 1), 'user_previous_reward',
                                             merged_embedding, inputs)

  return merged_embedding, inputs


def construct_rnn_layer(rnn_type, rnn_merged_embedding, hidden_size,
                        regularizer_obj):
  """Returns a recurrent layer for the given type or raises ValueError."""
  if rnn_type == 'LSTM':
    rnn_layer = tf.keras.layers.LSTM(
        units=hidden_size,
        return_sequences=True,
        return_state=True,
        name='LSTM',
        kernel_regularizer=regularizer_obj)
    whole_seq_output, final_memory_state, final_carry_state = rnn_layer(
        rnn_merged_embedding)
    final_state = [final_memory_state, final_carry_state]
    return rnn_layer, whole_seq_output, final_state
  elif rnn_type == 'GRU':
    rnn_layer = tf.keras.layers.GRU(
        units=hidden_size,
        return_sequences=True,
        return_state=True,
        name='GRU',
        kernel_regularizer=regularizer_obj)
    whole_seq_output, final_state = rnn_layer(rnn_merged_embedding)
    return rnn_layer, whole_seq_output, final_state
  else:
    raise NotImplementedError(
        "Use recurrent cell as one out of 'LSTM' and 'GRU'.")
