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

"""Agent models to generate recommendations."""

import abc

import numpy as np
import scipy
import six
import tensorflow as tf

from recs_ecosystem_creator_rl.recommender import data_utils


@six.add_metaclass(abc.ABCMeta)
class AbstractAgent:
  """Abstract class to generate recommendations."""

  def __init__(self, slate_size):
    self.slate_size = slate_size

  @abc.abstractmethod
  def step(self, user_dict, creator_dict, docs):
    """Generates recommendations for each user given observable features of users and candidate documents.

    Args:
      user_dict: A dictionary of user observed information including user_obs =
        A dictionary of key=user_id, value=a list of user observations at all
        time steps. user_clicked_docs = A dictionary of key=user_id, value=a
        list of user consumed documents (doc, reward, index in the candidate
        set). user_terminates = A dictionary of key=user_id, value=boolean
        denoting whether this user has terminated or not at the end of
        simulation.
      creator_dict: A dictionary of creator observed information including
        creator_obs = A dict describing all creator observation history, with
        key=creator_id, value=a list of creator's all past observations.
        creator_recommended_docs = A dict describing all creator recommendation
        history, with key=creator_id, value=a list of recommended doc objects.
        creator_clicked_docs = A dict describing all creator user-clicked
        document history, with key=creator_id, value=a list of user-clicked docs
        (document object, user reward). creator_actions = A dictionary of
        key=creator_id, value=a list of creator actions(one of
        'create'/'stay'/'leave') at current time step. creator_terminates = A
        dict to show whether creator terminates or not at current time step,
        with key=creator_id, value=True if creator terminates otherwise False.
      docs: An ordered dictionary of current document candidate set with
        key=doc_id, value=document object.
    """


class RandomAgent(AbstractAgent):
  """Random agent class."""

  def __init__(self, slate_size=2):
    self.name = 'RandomAgent'
    super(RandomAgent, self).__init__(slate_size)

  def step(self, user_dict, docs):
    return generate_random_slate(self.slate_size, user_dict, docs)


def generate_random_slate(slate_size, user_dict, docs):
  """Generate random slate."""
  viable_user_ids = [
      u_id for u_id, u_tmnt in user_dict['user_terminates'].items()
      if not u_tmnt
  ]
  num_doc = len(docs)
  slates = {
      u_id: np.random.choice(num_doc, size=slate_size)
      for u_id in viable_user_ids
  }
  probs = {u_id: np.ones(num_doc) / num_doc for u_id in viable_user_ids}
  return slates, probs, None


class PolicyGradientAgent(AbstractAgent):
  """PolicyGradient agent."""

  def __init__(self,
               slate_size=2,
               user_embedding_size=10,
               document_embedding_size=10,
               creator_embedding_size=1,
               num_candidates=10,
               hidden_sizes=(32, 16),
               weight_size=10,
               lr=1e-3,
               user_model=None,
               creator_model=None,
               entropy_coeff=0.01,
               regularization_coeff=None,
               model_path=None,
               seed=None,
               loss_denom_decay=-1.0,
               social_reward_coeff=0.0):
    if seed:
      tf.random.set_seed(seed)
    super(PolicyGradientAgent, self).__init__(slate_size)
    self.name = 'EcoAgent'
    self.entropy_coeff = entropy_coeff
    self.social_reward_coeff = social_reward_coeff
    self.user_model = user_model
    self.creator_model = creator_model

    # Moving average user_utlities and social_rewards denom.
    self.sum_label_weights_var = tf.Variable(
        0.0, name='sum_label_weights', dtype=tf.float32, trainable=False)
    self.loss_denom_decay = loss_denom_decay
    self.num_updates = 0

    # For environment step preprocessing candidates.
    self.creator_hidden_state = None
    self.doc_feature = None

    # Model.
    inputs, outputs = self._construct_graph(user_embedding_size,
                                            document_embedding_size,
                                            creator_embedding_size,
                                            num_candidates, hidden_sizes,
                                            weight_size, regularization_coeff)
    self.actor_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    self.optimizer = tf.keras.optimizers.Adagrad(lr)
    # Metrics.
    self.train_loss = tf.keras.metrics.Mean('train_loss')
    self.train_utility_loss = tf.keras.metrics.Mean('train_utility_loss')
    self.train_entropy_loss = tf.keras.metrics.Mean('train_entropy_loss')
    self.ckpt = tf.train.Checkpoint(
        step=tf.Variable(1),
        optimizer=self.optimizer,
        value_model=self.actor_model)
    self.manager = tf.train.CheckpointManager(
        self.ckpt, model_path, max_to_keep=3)
    self.ckpt.restore(self.manager.latest_checkpoint)
    if self.manager.latest_checkpoint:
      print('Restored from {}.'.format(self.manager.latest_checkpoint))
    else:
      print('Initializing from scratch.')

  def _construct_graph(self,
                       user_embedding_size,
                       document_embedding_size,
                       creator_embedding_size,
                       num_candidates,
                       hidden_sizes,
                       weight_size,
                       regularization_coeff=None):
    """Construct network architecture of policy gradient agent."""
    if regularization_coeff is not None:
      regularizer_obj = tf.keras.regularizers.l2(regularization_coeff)
    else:
      regularizer_obj = None
    user_input_state = tf.keras.layers.Input(
        shape=(user_embedding_size), name='user_embedding_state')
    document_input_state = tf.keras.layers.Input(
        shape=(num_candidates, document_embedding_size),
        name='document_feature')
    creator_input_state = tf.keras.layers.Input(
        shape=(num_candidates, creator_embedding_size),
        name='creator_embedding_state')

    # User hidden layer is used to embed user to calculate softmax logits.
    user_hidden_layer = user_input_state
    for i, hidden_size in enumerate(hidden_sizes, 1):
      user_hidden_layer = tf.keras.layers.Dense(
          units=hidden_size,
          activation='relu',
          kernel_regularizer=regularizer_obj,
          name=f'user_actor_hidden_layer_{i}')(
              user_hidden_layer)
    user_embedding_weights = tf.keras.layers.Dense(
        units=weight_size, activation=None, kernel_regularizer=regularizer_obj)(
            user_hidden_layer)
    user_embedding_weights = tf.nn.l2_normalize(
        user_embedding_weights, axis=-1, name='user_weights')
    # User sensitivity to document bias, range [0, 1].
    user_sensitivity = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_regularizer=regularizer_obj,
        name='user_sensitivity')(
            user_hidden_layer)
    # We can also use fixed effects from both users and creators.

    # Document hidden layer to embed candidate documents.
    candidate_hidden_layer = tf.keras.layers.concatenate(
        [document_input_state, creator_input_state], axis=-1)
    for i, hidden_size in enumerate(hidden_sizes, 1):
      candidate_hidden_layer = tf.keras.layers.Dense(
          units=hidden_size,
          activation='relu',
          kernel_regularizer=regularizer_obj,
          name=f'doc-creator_actor_hidden_layer_{i}')(
              candidate_hidden_layer)
    candidate_embedding_weights = tf.keras.layers.Dense(
        units=weight_size, activation=None, kernel_regularizer=regularizer_obj)(
            candidate_hidden_layer)
    candidate_embedding_weights = tf.nn.l2_normalize(
        candidate_embedding_weights, axis=-1, name='document_weights')
    # Bias within [-1, 1].
    candidate_embedding_bias = tf.squeeze(
        tf.keras.layers.Dense(
            units=1, activation='tanh',
            kernel_regularizer=regularizer_obj)(candidate_hidden_layer),
        axis=-1,
        name='document_bias')

    # Softmax logits = (1 - user_sensitivity) * < user_weights,
    #            document_weights > + user_sensitivity * document_bias.
    # TODO(rhzhan): Experiment with other architecture. For example, add bias
    # terms from both users and creators; only bias from creators; etc.
    output_log_logits = (1 - user_sensitivity) * tf.linalg.matvec(
        candidate_embedding_weights,
        user_embedding_weights) + user_sensitivity * candidate_embedding_bias
    inputs = [user_input_state, document_input_state, creator_input_state]
    return inputs, output_log_logits

  def train_step(self, inputs, labels, user_utilities, social_rewards):
    """Training step given mini-batch data."""
    self.ckpt.step.assign_add(1)
    self.num_updates += 1
    user_utilities = tf.cast(user_utilities, dtype=tf.float32)
    social_rewards = tf.cast(social_rewards, dtype=tf.float32)
    label_weights = (
        1 - self.social_reward_coeff
    ) * user_utilities + self.social_reward_coeff * social_rewards
    with tf.GradientTape() as tape:
      logits = self.actor_model(inputs, training=True)
      p = tf.nn.softmax(logits=logits)
      neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      if self.loss_denom_decay >= 0:
        # Batch normalization on label weights.
        label_weights_denom = tf.reduce_sum(tf.abs(label_weights))
        tf.keras.backend.moving_average_update(
            self.sum_label_weights_var,
            value=label_weights_denom,
            momentum=self.loss_denom_decay)  # Update moving average.
        label_weights_denom = self.sum_label_weights_var / (
            1 - self.loss_denom_decay**self.num_updates)  # Debias.
        label_weights = label_weights / label_weights_denom
      utility_loss = tf.reduce_mean(label_weights * neglogp)
      entropy = tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=logits)
      entropy_loss = -tf.reduce_mean(entropy)
      loss = utility_loss + self.entropy_coeff * entropy_loss
    grad = tape.gradient(loss, self.actor_model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(grad, self.actor_model.trainable_variables))
    self.train_loss(loss)
    self.train_utility_loss(utility_loss)
    self.train_entropy_loss(entropy_loss)

  def preprocess_candidates(self, creator_dict, docs):
    """Preprocess candidates into creator features and doc features."""
    # We are learning creator hidden state using self.creator_model separately.
    (creator_hidden_state_dict, creator_rnn_state_dict,
     creator_is_saturation_dict) = data_utils.get_creator_hidden_state(
         creator_dict, self.creator_model)
    # Concatenate document_topic with corresponding creator_hidden_state.
    (self.creator_hidden_state, creator_rnn_state, creator_is_saturation,
     creator_id, self.doc_feature) = data_utils.align_document_creator(
         creator_hidden_state_dict, creator_rnn_state_dict,
         creator_is_saturation_dict, docs)
    return (self.creator_hidden_state, creator_rnn_state, creator_is_saturation,
            creator_id, self.doc_feature)

  def step(self, user_dict, docs):
    viable_user_ids = [
        user_id for user_id, user_tmnt in user_dict['user_terminates'].items()
        if not user_tmnt
    ]
    if not user_dict['user_clicked_docs'][viable_user_ids[0]]:
      # When no history, generate random slates.
      return generate_random_slate(self.slate_size, user_dict, docs)
    policy, preprocessed_users = self.get_policy(user_dict)
    user_id, p = list(policy.keys()), list(policy.values())
    slates = np.argsort(p, axis=-1)[Ellipsis, -self.slate_size:]
    return dict(zip(user_id, slates)), policy, preprocessed_users

  def get_policy(self, user_dict):
    """Generate policy of given observations."""
    # We are learning user hidden state using self.user_model separately.
    user_hidden_state_dict = data_utils.get_user_hidden_state(
        user_dict, self.user_model)
    user_id, user_hidden_state = zip(*user_hidden_state_dict.items())
    user_hidden_state = np.array(list(user_hidden_state))

    creator_input = np.tile(self.creator_hidden_state,
                            (len(user_hidden_state), 1, 1))
    doc_input = np.tile(self.doc_feature, (len(user_hidden_state), 1, 1))
    model_inputs = [user_hidden_state, doc_input, creator_input]
    logits = self.actor_model.predict(model_inputs)
    p = scipy.special.softmax(logits, axis=-1)
    return dict(zip(user_id, p)), dict(zip(user_id, user_hidden_state))

  def save(self):
    save_path = self.manager.save()
    print('Saved checkpoint for step {}: {}'.format(
        int(self.ckpt.step), save_path))
