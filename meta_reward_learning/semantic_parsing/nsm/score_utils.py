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

"""Helper class for score functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
# from tensorflow.keras.constraints import non_neg
from meta_reward_learning.semantic_parsing.nsm import tf_utils
EPS = 1e-8


class ScoreFunction(object):
  """Class for creating score functions to be used for meta learning."""

  def __init__(self,
               score_model,
               num_envs,
               trainable=False,
               score_norm_fn=None,
               max_programs=30,
               num_features=11,
               score_temperature=1.0,
               **kwargs):
    if score_model not in [
        'constant',
        'tabular',
        'linear',
        'local_linear'  # 'local_attn', 'attn'
    ]:
      raise ValueError('{} is not a valid score model'.format(score_model))
    self.score_model = score_model
    self.trainable = trainable
    self._score_norm_fn = score_norm_fn
    self._is_linear_softmax = (
        self.score_model == 'linear' and self._score_norm_fn == 'softmax')
    self._num_envs = num_envs
    self._max_programs = max_programs
    self._num_features = num_features
    self._score_temperature = score_temperature
    self._score_helper(**kwargs)

  def create_sim_features(self):
    # TODO(rishabhagarwal): Fix the harcoded values
    if self.score_model == 'tabular' or self._is_linear_softmax:
      num_features = 2
      dtype = tf.int32
    else:
      num_features = self._num_features
      dtype = tf.float32
    return tf.placeholder(
        dtype=dtype, shape=[None, num_features], name='sim_features')

  def create_attn_based_scores(self, scores_per_timestep, sequence_length):
    """Creates the scores for the attn based score function."""
    with tf.variable_scope('score_fn', reuse=tf.AUTO_REUSE):
      maxlen = len(scores_per_timestep)
      scores_per_timestep = tf.stack(scores_per_timestep, axis=1)
      scores_per_timestep = tf.squeeze(scores_per_timestep, axis=-1)
      sequence_mask = tf.sequence_mask(
          sequence_length, maxlen=maxlen, dtype=tf.float32)
      lengths = tf.cast(sequence_length, dtype=tf.float32)
      score_vals = scores_per_timestep * sequence_mask
      score_vals = tf.reduce_sum(score_vals, axis=-1) / lengths
      if self.score_model == 'attn':
        score_vals = tf.nn.relu(score_vals + 1.0)
        self.scores = score_vals
      elif self.score_model == 'local_attn':
        self.score_bias = score_vals

  def get_var_from_placeholder(self, name, **kwargs):
    var, pc, init = tf_utils.create_var_and_placeholder(name, **kwargs)
    self._score_dict.update({
        '{}_placeholder'.format(name): pc,
        '{}_init'.format(name): init
    })
    return var

  def _score_helper(self, **kwargs):
    """Add the rewards weights to the graph in case of a tabular setting."""
    self._score_dict = {}
    if self.score_model == 'constant':
      return
    with tf.variable_scope('score_fn', reuse=tf.AUTO_REUSE):
      self._score_dict['sim_features'] = self.create_sim_features()
      score_features = self._score_dict['sim_features']
      if self.score_model == 'tabular' or self._is_linear_softmax:
        if self._is_linear_softmax:
          score_shape = [self._num_envs, self._max_programs, self._num_features]
        else:
          score_shape = [self._num_envs, self._max_programs]
        score_features = self.get_var_from_placeholder(
            'scores',
            shape=score_shape,
            dtype=tf.float32,
            trainable=self.trainable)
      elif self.score_model == 'local_linear':
        env_indices = tf.placeholder(
            dtype=tf.int32, shape=[None], name='env_indices')
        self._score_dict.update(env_indices=env_indices)
      scores = self._create_scores(score_features, **kwargs)
      if self._score_norm_fn == 'softmax':
        num_trajs = self.get_var_from_placeholder(
            'num_trajs',
            shape=[self._num_envs],
            dtype=tf.int32,
            trainable=False)
        score_mask = tf.sequence_mask(
            num_trajs, maxlen=self._max_programs, dtype=tf.float32)
        scores /= self._score_temperature  # Temperature parameter
        alpha = tf.get_variable(
            'alpha',
            dtype=tf.float32,
            trainable=self.trainable,
            initializer=0.0)
        scores = score_mask * scores + (1.0 - score_mask) * alpha
        self.reward_weights = tf.nn.softmax(scores, axis=-1)
      elif self._score_norm_fn == 'sigmoid':
        self.reward_weights = 2 * tf.sigmoid(scores)
      elif self._score_norm_fn == 'exp':
        # This should only be used when the training objective is MML
        scores /= self._score_temperature
        self.reward_weights = tf.exp(scores)
      elif self._score_norm_fn == 'identity':
        self.reward_weights = scores

  @property
  def score_dict(self):
    return self._score_dict

  @property
  def num_features(self):
    return self._num_features

  @property
  def is_linear_softmax(self):
    return self._is_linear_softmax

  def _create_scores(self, score_features, **kwargs):
    """Creates the weights and biases for scores."""
    if self.score_model == 'tabular':
      return score_features
    elif self.score_model in ['linear', 'local_linear']:
      # Code for initialization
      len_features = score_features.shape.as_list()[-1]
      if 'score_fn_init' in kwargs:
        score_fn_init = kwargs['score_fn_init']
        initializer = score_fn_init[:-1]
        bias_initializer = score_fn_init[-1]
      else:
        initializer = len_features * [1.0]
        bias_initializer = 0.0
      # Represents the feature similar to jaccard similarity
      weights = tf.get_variable(
          'weights',
          dtype=tf.float32,
          trainable=self.trainable,
          initializer=list(initializer))
      if self.score_model == 'linear' and not self._is_linear_softmax:
        # Linear combination of features with a bias
        bias = tf.get_variable(
            'bias',
            dtype=tf.float32,
            trainable=self.trainable,
            initializer=[bias_initializer])
      elif self.score_model == 'local_linear':
        # Linear combination of features with a context dependent bias
        reward_bias = tf.get_variable(
            'reward_bias',
            trainable=self.trainable,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            shape=[self._num_envs])
        bias = tf.gather(
            params=reward_bias,
            indices=self._score_dict['env_indices'],
            name='bias')
      scores = tf.tensordot(score_features, weights, axes=1)
      if not self._is_linear_softmax:
        scores += bias
      return scores

  def get_scores(self, **kwargs):
    """Returns the scores corresponding to score features."""
    if self.score_model == 'tabular' or self._is_linear_softmax:
      # Use sim_features as reward_indices
      sim_features = self._score_dict.get('sim_features', None)
      scores = tf.gather_nd(
          self.reward_weights, sim_features, name='reward_weights')
    elif self.score_model in ['linear', 'local_linear']:
      scores = self.reward_weights
    elif self.score_model == 'constant':
      scores = tf.ones(shape=kwargs['n'], name='scores')
    else:
      raise ValueError('Other score models not defined yet!')
    log_scores = tf.log(scores + EPS)
    return scores, log_scores


def create_env_index(replay_buffer):
  """Creates the mapping from env_name to their index in the replay buffer."""
  env_index = {}
  buffer_items = sorted(
      replay_buffer.traj_buffer.iteritems(), key=lambda k: k[0])
  for idx, (env_name, _) in enumerate(buffer_items):
    env_index[env_name] = idx
  return env_index


def get_features(replay_buffer, num_envs, max_programs=30):
  buffer_items = sorted(
      replay_buffer.traj_buffer.iteritems(), key=lambda k: k[0])
  feature_length = len(buffer_items[0][1][0].sim_features)
  zero_feature = [0] * feature_length
  features = [[zero_feature] * max_programs for _ in range(num_envs)]
  for idx, (_, trajs) in enumerate(buffer_items):
    features[idx][:len(trajs)] = [t.sim_features for t in trajs]
  return features


def get_num_trajs(replay_buffer, num_envs):
  buffer_items = sorted(
      replay_buffer.traj_buffer.iteritems(), key=lambda k: k[0])
  num_trajs = [0] * num_envs
  for idx, (_, trajs) in enumerate(buffer_items):
    num_trajs[idx] = len(trajs)
  return num_trajs


def create_init_weights(score_norm_fn,
                        replay_buffer,
                        num_envs=None,
                        max_programs=30,
                        agent=None):
  """Get the initialization weights for tabular score model."""

  def pad(l, maxlen, pad_val=0):
    return l + (maxlen - len(l)) * [pad_val]

  # Init weights
  inv_prob_fn = None
  if score_norm_fn == 'softmax':
    # Ideally, -np.inf should work but it causes tf gradient to return nan
    init_val, pad_value = 1.0, -1e8
    inv_prob_fn = lambda x: np.log(x + EPS)
  elif score_norm_fn == 'sigmoid':
    init_val, pad_value = 0.0, -1e8
    inv_prob_fn = lambda x: np.log(x + EPS) - np.log(1 - x + EPS)
  elif score_norm_fn is None:
    init_val, pad_value = 1.0, 0.0
    inv_prob_fn = lambda x: x
  else:
    raise ValueError('{} is not a valid score_norm_fn'.format(score_norm_fn))
  if inv_prob_fn is not None:
    inv_prob_fn = np.vectorize(inv_prob_fn)

  init_weights = []
  # Need to sort the items to keep ordering consistent for starting from a
  # previous checkpointed model
  buffer_items = sorted(
      replay_buffer.traj_buffer.iteritems(), key=lambda k: k[0])
  for _, trajs in buffer_items:
    weights = [pad_value] * max_programs
    if agent is not None:
      probs = agent.compute_probs(trajs)
      traj_weights = inv_prob_fn(probs)
    else:
      traj_weights = [init_val] * len(trajs)
    for t, w in zip(trajs, traj_weights):
      weights[t.idx] = w
    init_weights.append(weights)
  if num_envs is not None:
    pad_arr = pad([], max_programs, pad_value)
    for _ in range(num_envs - len(init_weights)):
      init_weights.append(pad_arr)

  return init_weights
