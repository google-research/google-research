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

"""Utility functions for processing data."""

import collections
import itertools

import numpy as np
import tensorflow as tf


def _get_utilities(rewards, gamma):
  """Get accumulated utilities with decay rate gamma based on observed rewards.

  Args:
    rewards: 1d float array representing observed rewards.
    gamma: Float, decay rate.

  Returns:
    Accumulated future utilities for each time step.
  """
  reversed_rewards = rewards[::-1]
  reversed_utilities = list(
      itertools.accumulate(reversed_rewards, lambda x, y: x * gamma + y))
  return np.array(reversed_utilities[::-1])


def get_user_hidden_state(user_dict, user_model):
  """Get user hidden states based on user history.

  This function is used for agent.step(), thus all viable users have the
  history of same length, which means no padding or masking is needed.

  If at the start of simulation and there is no history yet, return all zeros
  as user hidden states.
  Otherwise, use user_model to embed user history(user_clicked_docs) into user
  hidden states.

  Args:
    user_dict: A dictionary of user observed information including: (*) user_obs
      = A dictionary of key=user_id, value=a list of user observations at all
      time steps. (*) user_clicked_docs = A dictionary of key=user_id, value=a
      list of user consumed documents (doc, reward, index in the candidate set).
      (*) user_terminates = A dictionary of key=user_id, value=boolean denoting
      whether this user has terminated or not at the end of simulation.
    user_model: RNN user value model to embed user history into user hidden
      states and predict user utility.

  Returns:
    A dictionary of viable_user_ids and user_hidden_states.
  """
  viable_user_ids = [
      user_id for user_id, user_tmnt in user_dict['user_terminates'].items()
      if not user_tmnt
  ]
  if not user_dict['user_clicked_docs'][viable_user_ids[0]]:
    # All viable users have the same length of histories and the user has no
    # option of no-click. We only need to check if the first viable user has
    # history.
    user_hidden_states = np.zeros(
        (len(viable_user_ids), user_model.embedding_size))
  else:
    user_consumed_docs = []
    for u_id in viable_user_ids:
      u_consumed_docs = []
      for doc, _, _, _, _, _, _ in user_dict['user_clicked_docs'][u_id]:
        u_consumed_docs.append(doc['topic'])
      user_consumed_docs.append(u_consumed_docs)
    user_hidden_states, _ = user_model.get_embedding(
        np.array(user_consumed_docs))
  return dict(zip(viable_user_ids, user_hidden_states))


def get_creator_hidden_state(creator_dict, creator_model):
  """Get creator hidden states based on creator history.

  This function is used for agent.step(), thus all viable creators have the
  history of same length, which means no padding or masking is needed.

  If at the start of simulation and there is no history yet, return all zeros
  as creator hidden states.
  Otherwise, use creator_model to embed creator history(#recs, #clicks,
  summed_user_reward, average_user_click_doc_topics) into creator
  hidden states.

  Args:
    creator_dict:  A dictionary of creator observed information including:: (*)
      creator_obs = A dictionary of key=creator_id, value=a list of creator
      observations at all time steps. (*) creator_recommended_docs = A
      dictionary of key=creator_id, value=a list of sublists, where each sublist
      represents the recommended documents at current time steps. (*)
      creator_clicked_docs = A dictionary of key=creator_id, value=a list of
      sublists, where each sublist represents the user clicked documents
      (document object, user reward) at current time steps. (*) creator_actions
      = A dictionary of key=creator_id, value=a list of creator actions(one of
      'create'/'stay'/'leave') at current time step. (*) creator_terminates = A
      dictionary of key=creator_id, value=boolean denoting whether this creator
      has terminated at the end of simulation.
    creator_model: RNN creator value model to embed creator history into creator
      hidden states and predict creator utility.

  Returns:
    A dictionary of viable creator_hidden_states and a dictionary of viable
      creator_rnn_state, with both ids to be creator_ids.

  """
  viable_creator_ids = [
      creator_id for creator_id, creator_tmnt in
      creator_dict['creator_terminates'].items() if not creator_tmnt
  ]
  history_len = len(
      creator_dict['creator_recommended_docs'][viable_creator_ids[0]])
  creator_is_saturation_dict = {
      c_id: creator_dict['creator_is_saturation'][c_id]
      for c_id in viable_creator_ids
  }
  if history_len == 0:
    # All viable creators have the same length of histories. We only need to
    # check if the first viable creator has history.
    creator_embedding_states = np.zeros(
        (len(viable_creator_ids), creator_model.embedding_size))
    if creator_model.rnn_type == 'LSTM':
      creator_rnn_states = np.zeros(
          (len(viable_creator_ids), 2, creator_model.embedding_size))
    else:
      creator_rnn_states = np.zeros(
          (len(viable_creator_ids), creator_model.embedding_size))
  else:
    creator_num_recs = []
    creator_num_clicks = []
    creator_user_rewards = []
    creator_satisfaction = []
    creator_clicked_doc_topics = []
    for c_id in viable_creator_ids:
      c_num_recs = [
          [len(recs)] for recs in creator_dict['creator_recommended_docs'][c_id]
      ]
      # We skip the current observation when embedding the history.
      c_satisfaction = [[
          c_obs['creator_satisfaction']
      ] for c_obs in creator_dict['creator_obs'][c_id][:history_len]]
      c_num_clicks = []
      c_user_rewards = []
      clicked_doc_topic = []
      for per_time_clicks in creator_dict['creator_clicked_docs'][c_id]:
        c_num_clicks.append([len(per_time_clicks)])

        if per_time_clicks:
          (per_time_clicked_docs, per_time_user_rewards) = zip(*per_time_clicks)
          per_time_clicked_doc_topics = [
              doc['topic'] for doc in per_time_clicked_docs
          ]
          clicked_doc_topic.append(
              np.average(
                  per_time_clicked_doc_topics,
                  axis=0,
                  weights=per_time_user_rewards))
          c_user_rewards.append([np.sum(per_time_user_rewards)])
        else:
          clicked_doc_topic.append(
              np.zeros(creator_model.document_feature_size))
          c_user_rewards.append([0])
      pad_length = creator_model.trajectory_length - history_len
      creator_num_recs.append(c_num_recs + [[0]] * pad_length)
      creator_num_clicks.append(c_num_clicks + [[0]] * pad_length)
      creator_user_rewards.append(c_user_rewards + [[0]] * pad_length)
      creator_satisfaction.append(c_satisfaction + [[0]] * pad_length)
      creator_clicked_doc_topics.append(
          clicked_doc_topic +
          [np.zeros(creator_model.document_feature_size)] * pad_length)

    creator_id = np.tile(
        np.reshape(viable_creator_ids, [-1, 1]),
        (1, creator_model.trajectory_length))
    creator_mask = np.zeros_like(creator_id)
    creator_mask[:, :history_len] = 1
    creator_mask = tf.convert_to_tensor(creator_mask, bool)

    inputs = [
        np.array(creator_num_recs),
        np.array(creator_num_clicks),
        np.array(creator_user_rewards),
        np.array(creator_clicked_doc_topics),
        np.array(creator_satisfaction),
    ]
    if creator_model.creator_id_embedding_size > 0:
      inputs.append(creator_id)
    creator_embedding_states, creator_rnn_states = creator_model.get_embedding(
        inputs, mask=creator_mask)
    if creator_model.rnn_type == 'LSTM':
      creator_rnn_states = np.swapaxes(
          creator_rnn_states, 0, 1)  # Shape(num_creators, 2, rnn_state_size).
  return (dict(zip(viable_creator_ids, creator_embedding_states)),
          dict(zip(viable_creator_ids,
                   creator_rnn_states)), creator_is_saturation_dict)


def align_document_creator(creator_hidden_state_dict, creator_rnn_state_dict,
                           creator_is_saturation_dict, docs):
  """Align documents with associating creators in a candidate set."""
  doc_input = []
  creator_input = []
  creator_rnn_final_state = []
  creator_is_saturation = []
  creator_id = []
  for doc in docs.values():
    doc_input.append(doc['topic'])
    c_id = doc['creator_id']
    creator_input.append(creator_hidden_state_dict[c_id])
    creator_rnn_final_state.append(creator_rnn_state_dict[c_id])
    creator_is_saturation.append(creator_is_saturation_dict[c_id])
    creator_id.append(c_id)
  return np.array(creator_input), np.array(creator_rnn_final_state), np.array(
      creator_is_saturation), np.array(creator_id), np.array(doc_input)


class DummyValue:

  def predict_value(self, inputs):
    if isinstance(inputs, list):
      batch_size, max_episode_length, _ = inputs[0].shape
    else:
      batch_size, max_episode_length, _ = inputs.shape
    return np.zeros((batch_size, max_episode_length, 1))


class ExperienceReplay:
  """Class to store data."""

  def __init__(self,
               nsteps=10,
               doc_feature_size=10,
               num_candidates=10,
               user_gamma=0.99,
               creator_gamma=0):
    """Initializes an experience_replay buffer.

    Args:
      nsteps: Int representing the maximimum length of a simulated trajectory.
      doc_feature_size: Int, number of document topics.
      num_candidates: Int, size of candidate set.
      user_gamma: Float, user reward discount factor when calculating utilities.
      creator_gamma: Float, creator reward discount factor when calculating
        utilities.
    """
    self.trajectory_length = nsteps
    self.doc_feature_size = doc_feature_size
    self.num_candidates = num_candidates
    self.user_gamma = user_gamma
    self.creator_gamma = creator_gamma
    self.reset()

  def reset(self):
    """Reset buffer."""
    self.num_runs = 0  # Record how many simulations the buffer holds.

    # User data. Key = f`{self.num_runs}_{u_id}`.
    self.user_accumulated_reward = collections.OrderedDict()
    self.user_utilities = collections.OrderedDict()
    self.user_consumed_docs = collections.OrderedDict()
    self.user_masks = collections.OrderedDict()
    self.user_trajectory_lengths = collections.OrderedDict()
    self.user_click_labels = collections.OrderedDict()
    self.user_current_rewards = collections.OrderedDict()
    self.user_click_creator_rnn_states = collections.OrderedDict()
    self.user_click_creator_previous_satisfaction = collections.OrderedDict()
    self.user_click_creator_is_saturation = collections.OrderedDict()
    self.user_click_creator_id = collections.OrderedDict()

    # Creator data. Key = f`{self.num_runs}_{c_id}`.
    self.creator_utilities = collections.OrderedDict()
    self.creator_num_recs = collections.OrderedDict()
    self.creator_num_clicks = collections.OrderedDict()
    self.creator_clicked_doc_topics = collections.OrderedDict()
    self.creator_accumulated_reward = collections.OrderedDict()
    self.creator_masks = collections.OrderedDict()
    self.creator_trajectory_lengths = collections.OrderedDict()
    self.creator_user_rewards = collections.OrderedDict()
    self.creator_previous_satisfaction = collections.OrderedDict()
    self.creator_current_rewards = collections.OrderedDict()
    self.creator_is_saturation = collections.OrderedDict()

    # Actor data.
    ## Actor inputs.
    self.actor_user_embedding_states = []
    self.actor_creator_embedding_states = []
    self.actor_documents = []
    ## Actor label.
    self.actor_labels = []
    ## Actor social rewards.
    self.actor_click_creator_rnn_states = []
    self.actor_user_rewards = []
    self.actor_user_clicked_docs = []
    self.actor_click_creator_satisfaction = []
    self.actor_click_creator_is_saturation = []  # For normalization.
    self.actor_click_creator_id = []
    ## Actor action utilities.
    self.actor_user_utilities = []
    self.actor_creator_uplift_utilities = []
    self.need_calculate_creator_uplift_utility = True

  def update_experience(self,
                        user_dict,
                        creator_dict,
                        preprocessed_user_documents=None,
                        update_actor=True):
    """Update buffer.

    We format the data of one simulation to train value models and actor model.

    The user value model takes the input of user currently consumed documents
    and outputs the current step's utility (including current reward).

    Similarly, the creator value model takes the input of current user and
    recommender feedback and outputs the current step's utility (including
    current reward).

    The actor model takes the current time step's user hidden states, candidate
    documents and corresponding creator hidden states. Since we only have the
    user and creator history to get hidden states starting from t=2, we thus
    correspondingly only use the candidate sets starting from t=2. Thus the
    first candidate set is discarded.

    As a result, the formatted trajectory length is 1 less than the real
    simulation trajectory.

    Args:
      user_dict: A dictionary of user observed information including
        user_clicked_docs, which is a dictionary of key=user_id, value=a list of
        user consumed documents (doc, reward, index in the candidate set).
      creator_dict:  A dictionary of creator observed information including::
        (*) creator_obs = A dictionary of key=creator_id, value=a list of
        creator observations at all time steps. (*) creator_recommended_docs = A
        dictionary of key=creator_id, value=a list of sublists, where each
        sublist represents the recommended documents at current time steps. (*)
        creator_clicked_docs = A dictionary of key=creator_id, value=a list of
        sublists, where each sublist represents the user clicked documents
        (document object, user reward) at current time steps.
      preprocessed_user_documents: Includes viable user_embedding_states,
        candidate document topics, and candidate creator embedding states at
        each time step across the simulation.
      update_actor: Boolean, whether or not updating actor data, useful when we
        only train utility models.
    """

    self.update_user_experience(user_dict)
    self.update_creator_experience(creator_dict)
    if update_actor:
      self.update_actor_experience(preprocessed_user_documents)

    self.num_runs += 1

  def update_user_experience(self, user_dict):
    """Update buffer with user-related data."""
    user_clicked_docs = user_dict['user_clicked_docs']

    for u_id in user_clicked_docs:
      u_key = f'{self.num_runs}_{u_id}'

      (
          consumed_docs,
          rewards,
          labels,
          clicked_creator_rnn_states,
          click_creator_previous_satisfaction,
          clicked_creator_is_saturation,
          clicked_creator_id,
      ) = zip(*user_clicked_docs[u_id])
      consumed_docs = [doc['topic'] for doc in consumed_docs]

      self.user_accumulated_reward[u_key] = np.sum(rewards)

      utilities = _get_utilities(rewards, self.user_gamma)

      # Calculate mask for user trajectories. Since we are padding user
      # trajectories to the same length, we use a mask to denote a padding
      # wherever mask=0.
      mask = np.zeros(self.trajectory_length)
      user_trajectory_length = len(utilities)
      mask[:user_trajectory_length] = 1
      self.user_trajectory_lengths[u_key] = user_trajectory_length

      # Padding the user utilities to the same length for computation in case
      # there are users terminating.
      pad_length = self.trajectory_length - user_trajectory_length
      utilities = list(utilities) + [0] * pad_length
      consumed_docs = consumed_docs + [np.zeros(self.doc_feature_size)
                                      ] * pad_length

      # For user value model training.
      self.user_consumed_docs[u_key] = consumed_docs
      self.user_utilities[u_key] = utilities
      self.user_masks[u_key] = mask

      # For actor model training.
      self.user_click_labels[u_key] = labels
      self.user_current_rewards[u_key] = np.array(rewards)[:, np.newaxis]
      self.user_click_creator_rnn_states[u_key] = np.array(
          clicked_creator_rnn_states)
      self.user_click_creator_previous_satisfaction[u_key] = np.array(
          click_creator_previous_satisfaction)[:, np.newaxis]
      self.user_click_creator_is_saturation[
          u_key] = clicked_creator_is_saturation
      self.user_click_creator_id[u_key] = clicked_creator_id

  def update_creator_experience(self, creator_dict):
    """Update buffer with creator-related data."""
    creator_obs = creator_dict['creator_obs']
    creator_rewards = creator_dict['creator_rewards']
    creator_recommended_docs = creator_dict['creator_recommended_docs']
    creator_clicked_docs = creator_dict['creator_clicked_docs']
    creator_is_saturation = creator_dict['creator_is_saturation']

    for c_id in creator_obs:
      c_key = f'{self.num_runs}_{c_id}'

      c_rewards = creator_rewards[c_id]
      self.creator_current_rewards[c_key] = np.array(c_rewards)
      self.creator_accumulated_reward[c_key] = np.sum(c_rewards)
      self.creator_is_saturation[c_key] = creator_is_saturation[c_id]
      # Calculate creator forward discounted utility.
      c_utilities = _get_utilities(c_rewards, self.creator_gamma)

      # Format feedback from users and recommender.
      num_recs = [[len(rec)] for rec in creator_recommended_docs[c_id]]
      num_clicks = []
      creator_user_rewards = []
      clicked_doc_topic = []
      for per_time_clicks in creator_clicked_docs[c_id]:
        num_clicks.append([len(per_time_clicks)])
        if per_time_clicks:
          # If there are user-clicks, calculate summed user_reward and average
          # user-clicked doc topic weighted by user rewards.
          (per_time_clicked_docs, per_time_user_rewards) = zip(*per_time_clicks)
          per_time_clicked_doc_topics = [
              doc['topic'] for doc in per_time_clicked_docs
          ]
          clicked_doc_topic.append(
              np.average(
                  per_time_clicked_doc_topics,
                  axis=0,
                  weights=per_time_user_rewards))
          creator_user_rewards.append([np.sum(per_time_user_rewards)])
        else:
          # If there is no user-click, use zero instead.
          clicked_doc_topic.append(np.zeros(self.doc_feature_size))
          creator_user_rewards.append([0])

      # Padding the creator utilities to the same length for computation in case
      # there are terminated creators.
      creator_trajectory_length = len(c_utilities)
      pad_length = self.trajectory_length - creator_trajectory_length
      c_utilities = list(c_utilities) + [0] * pad_length
      num_recs = num_recs + [[0]] * pad_length
      num_clicks = num_clicks + [[0]] * pad_length
      creator_user_rewards = creator_user_rewards + [[0]] * pad_length
      clicked_doc_topic = clicked_doc_topic + [np.zeros(self.doc_feature_size)
                                              ] * pad_length
      c_satisfaction = np.array(
          [c_obs['creator_satisfaction'] for c_obs in creator_obs[c_id]])
      c_previous_satisfaction = np.zeros((self.trajectory_length, 1))
      c_previous_satisfaction[:creator_trajectory_length, 0] = c_satisfaction
      mask = np.zeros(self.trajectory_length)
      mask[:creator_trajectory_length] = 1

      # For creator model training.
      self.creator_num_recs[c_key] = np.array(num_recs)
      self.creator_num_clicks[c_key] = np.array(num_clicks)
      self.creator_user_rewards[c_key] = np.array(creator_user_rewards)

      self.creator_clicked_doc_topics[c_key] = np.array(clicked_doc_topic)
      self.creator_previous_satisfaction[c_key] = c_previous_satisfaction
      self.creator_utilities[c_key] = np.array(c_utilities)
      self.creator_masks[c_key] = mask
      self.creator_trajectory_lengths[c_key] = creator_trajectory_length

  def update_actor_experience(self, preprocessed_user_documents):
    """Update buffer with actor-related data."""
    self.need_calculate_creator_uplift_utility = True
    (user_embedding_states, creator_embedding_states,
     documents) = preprocessed_user_documents
    # First data point is not used since we don't have user and creator
    # histories to embed at the beginning.
    for t in range(1, len(user_embedding_states)):
      num_users_t = len(user_embedding_states[t])
      self.actor_creator_embedding_states.extend([creator_embedding_states[t]] *
                                                 num_users_t)
      self.actor_documents.extend([documents[t]] * num_users_t)
      for u_id in user_embedding_states[t]:
        self.actor_user_embedding_states.append(user_embedding_states[t][u_id])

        u_key = f'{self.num_runs}_{u_id}'
        label = self.user_click_labels[u_key][t]
        self.actor_labels.append(label)
        self.actor_user_utilities.append(self.user_utilities[u_key][t])

        # For creator uplift modeling.
        self.actor_user_rewards.append(self.user_current_rewards[u_key][t])
        self.actor_user_clicked_docs.append(documents[t][label])
        self.actor_click_creator_rnn_states.append(
            self.user_click_creator_rnn_states[u_key][t])
        self.actor_click_creator_satisfaction.append(
            self.user_click_creator_previous_satisfaction[u_key][t])
        self.actor_click_creator_is_saturation.append(
            self.user_click_creator_is_saturation[u_key][t])
        self.actor_click_creator_id.append(self.user_click_creator_id[u_key][t])

  def user_data_generator(self, batch_size=32):
    """Yield batch data to train user_value_model."""
    start_idx = 0
    user_consumed_docs = np.array(list(self.user_consumed_docs.values()))
    user_utilities = np.array(list(self.user_utilities.values()))
    user_masks = np.array(list(self.user_masks.values()))
    num_samples = len(user_consumed_docs)
    while True:
      end_idx = min(start_idx + batch_size, num_samples)
      yield (user_consumed_docs[start_idx:end_idx],
             user_utilities[start_idx:end_idx, :,
                            np.newaxis], user_masks[start_idx:end_idx])
      if end_idx >= num_samples:
        break
      start_idx = end_idx

  def creator_data_generator(self, batch_size=32, creator_id_embedding_size=0):
    """Yield batch data to train creator_value_model."""
    start_idx = 0
    creator_num_recs = np.array(list(self.creator_num_recs.values()))
    creator_num_clicks = np.array(list(self.creator_num_clicks.values()))
    creator_clicked_doc_topics = np.array(
        list(self.creator_clicked_doc_topics.values()))
    creator_utilities = np.array(list(self.creator_utilities.values()))
    creator_user_rewards = np.array(list(self.creator_user_rewards.values()))
    creator_saturated_satifaction = np.array(
        list(self.creator_previous_satisfaction.values()))
    creator_id = np.array([[int(c_key.split('_')[1])] * self.trajectory_length
                           for c_key in self.creator_num_recs.keys()])
    creator_masks = np.array(list(self.creator_masks.values()))
    num_samples = len(creator_num_recs)
    while True:
      end_idx = min(start_idx + batch_size, num_samples)
      inputs = [
          creator_num_recs[start_idx:end_idx],
          creator_num_clicks[start_idx:end_idx],
          creator_user_rewards[start_idx:end_idx],
          creator_clicked_doc_topics[start_idx:end_idx],
          creator_saturated_satifaction[start_idx:end_idx],
      ]
      if creator_id_embedding_size > 0:
        inputs.append(creator_id[start_idx:end_idx])
      yield (inputs, creator_utilities[start_idx:end_idx, :, np.newaxis],
             creator_masks[start_idx:end_idx])
      if end_idx >= num_samples:
        break
      start_idx = end_idx

  def get_click_creator_uplift_utility(self, creator_embedding_model):
    """Calculates creator uplift utility."""
    num_samples = len(self.actor_user_embedding_states)
    actor_click_creator_rnn_states = np.array(
        self.actor_click_creator_rnn_states)
    if creator_embedding_model.rnn_type == 'LSTM':
      initial_state = [
          tf.convert_to_tensor(
              actor_click_creator_rnn_states[:, 0], dtype='float32'),
          tf.convert_to_tensor(
              actor_click_creator_rnn_states[:, 1], dtype='float32')
      ]
    else:
      initial_state = tf.convert_to_tensor(
          actor_click_creator_rnn_states, dtype='float32')

    mask = np.zeros((num_samples, self.trajectory_length))
    mask[:, 0] = 1
    mask = tf.convert_to_tensor(mask, bool)
    # Predict creator utilities given the recommendation:
    actor_user_rewards = np.zeros((num_samples, self.trajectory_length, 1))
    actor_user_rewards[:, 0] = self.actor_user_rewards
    actor_user_clicked_docs = np.zeros(
        (num_samples, self.trajectory_length, self.doc_feature_size))
    actor_user_clicked_docs[:, 0] = self.actor_user_clicked_docs
    actor_click_creator_satisfaction = np.zeros(
        (num_samples, self.trajectory_length, 1))
    actor_click_creator_satisfaction[:,
                                     0] = self.actor_click_creator_satisfaction
    actor_click_creator_id = np.zeros((num_samples, self.trajectory_length))
    actor_click_creator_id[:, 0] = self.actor_click_creator_id
    rec_inputs = [
        np.ones((num_samples, self.trajectory_length, 1)),
        np.ones((num_samples, self.trajectory_length, 1)),
        actor_user_rewards,
        actor_user_clicked_docs,
        actor_click_creator_satisfaction,
    ]
    if creator_embedding_model.creator_id_embedding_size > 0:
      rec_inputs.append(actor_click_creator_id)
    ul_rec_creator_utilities = creator_embedding_model.predict_value(
        rec_inputs, initial_state=initial_state, mask=mask)

    # Predict Creator utilities without the recommendation:
    rec_inputs = [
        np.zeros((num_samples, self.trajectory_length, 1)),
        np.zeros((num_samples, self.trajectory_length, 1)),
        np.zeros((num_samples, self.trajectory_length, 1)),
        np.zeros((num_samples, self.trajectory_length, self.doc_feature_size)),
        actor_click_creator_satisfaction,
    ]
    if creator_embedding_model.creator_id_embedding_size > 0:
      rec_inputs.append(actor_click_creator_id)
    ul_norec_creator_utilities = creator_embedding_model.predict_value(
        rec_inputs, initial_state=initial_state, mask=mask)

    # Calculate social reward.
    actor_creator_uplift_utilities = ul_rec_creator_utilities - ul_norec_creator_utilities
    self.actor_creator_uplift_utilities = np.array(
        actor_creator_uplift_utilities).flatten()

  def actor_data_generator(self, creator_embedding_model, batch_size=32):
    """Generate batch data for actor agent with user and creator embedding models.

    To train the PolicyGradient actor agent, we need to input batch data
    (inputs, labels, rewards).
    inputs: [user_embedding_state, doc_features, creator_embedding_states] where
     `user_embedding_state` and `creator_embedding_states` are learned
     separately from user and creator embedding models (value model).
     user_embedding_state: (batch_size, user_embedding_size)
     doc_features: (batch_size, num_candidates, document_feature_size)
     creator_embedding_states: (batch_size, num_candidates,
                                         creator_embedding_size)
    labels: Indices of user-clicked documents in the candidate sets, with
      shape (batch_size).
    rewards: Utilities (of both users and creators) for the action.

    Args:
      creator_embedding_model: An object generating creator embedding states
        based on creator history. To call, creator_hidden_state, _ =
        creator_embedding_model.get_embedding(creator_history).
      batch_size: Integer, size of mini-batch.

    Yields:
      (inputs, labels, rewards): Mini-batch training data for actor agent.
    """

    if self.need_calculate_creator_uplift_utility:  # pylint: disable=g-explicit-length-test
      self.get_click_creator_uplift_utility(creator_embedding_model)
      self.need_calculate_creator_uplift_utility = False

    actor_user_embedding_states = np.array(self.actor_user_embedding_states)
    actor_creator_embedding_states = np.array(
        self.actor_creator_embedding_states)
    actor_documents = np.array(self.actor_documents)
    actor_labels = np.array(self.actor_labels)
    # TODO(team) potentially update this name.
    actor_click_creator_is_saturation = np.array(
        self.actor_click_creator_is_saturation)
    actor_user_utilities = np.array(self.actor_user_utilities)
    num_samples = len(self.actor_user_embedding_states)

    start_idx = 0
    while True:
      end_idx = min(start_idx + batch_size, num_samples)
      inputs = [
          actor_user_embedding_states[start_idx:end_idx],
          actor_documents[start_idx:end_idx],
          actor_creator_embedding_states[start_idx:end_idx]
      ]
      yield (inputs, actor_labels[start_idx:end_idx],
             actor_user_utilities[start_idx:end_idx],
             self.actor_creator_uplift_utilities[start_idx:end_idx],
             actor_click_creator_is_saturation[start_idx:end_idx])
      if end_idx >= num_samples:
        break
      start_idx = end_idx

  def get_all_data(self):
    return (np.array(self.user_consumed_docs),
            np.array(self.user_utilities)[:, :, np.newaxis],
            np.array(self.user_masks))
