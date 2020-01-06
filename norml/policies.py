# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Policies for MAML RL.

A policy computes the probability of actions given the output of a
neural network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class GaussianPolicy(object):
  """Gaussian policy using a given neural network.

  p(action|state) =
    N(network_output, diag(exp(logstd)))
  """

  def __init__(self, network_input, network_output, action_dimensions, logstd):
    """Creates a new Gaussian Policy.

    Args:
      network_input: input to the neural network (1 dimensional)
      network_output: neural network output.
        There should be action_dimensions network_output units.
      action_dimensions: size of action vectors.
      logstd: log standard deviation of the actions (TF variable or constant).
        A network output works too, but is discouraged (difficult to learn).
    """
    self._input = network_input  # state
    self._output = network_output  # mean, log std
    self._dim = action_dimensions  # number of actions
    self._mean = self._output[:, :self._dim]
    self._log_std = logstd
    self._std = tf.exp(self._log_std)
    self._action_dist = tfp.distributions.Normal(
        loc=self._mean, scale=self._std)
    self._action_sample = {1: self._action_dist.sample()}

  def sample(self, state, session, feed_dict):
    """Generate sample(s) for the given input(state).

    This is just sampling from a normal distribution with mean and std
    deviations given by the output of the neural network.

    Args:
      state: neural network input values
      session: tf session
      feed_dict: feed dict for session (e.g. if logstd is not a constant).

    Returns:
      a matrix with samples (one row per row in state).
    """
    feed_dict[self._input] = state
    return session.run(self.sample_op()[0], feed_dict)

  def sample_op(self, num_samples=1):
    if num_samples not in self._action_sample:
      self._action_sample[num_samples] = self._action_dist.sample([num_samples])

    return self._action_sample[num_samples], self._input

  def mean_op(self):
    return self._mean, self._input

  def likelihood_op(self, actions):
    # Reshape to prevent unexpected broadcasting
    return tf.reshape(
        tf.reduce_prod(self._action_dist.prob(actions), 1), [-1, 1])

  def log_likelihood_op(self, actions):
    # Reshape to prevent unexpected broadcasting
    return tf.reshape(
        tf.reduce_sum(self._action_dist.log_prob(actions), 1), [-1, 1])

  def mean_std_log_std_op(self):
    return self._mean, self._std, self._log_std
