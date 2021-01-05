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

"""Neural network configuration for MAML."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow.compat.v1 as tf


def serialize_weights(session, weights, feed_dict=None):
  """Serializes the weights of current network into a 1-d array for protobuf.

  The order in which weights are serialized depends on the alphabetical
  ordering of the name of the weights.

  Args:
    session: a TF session in which the values are computed
    weights: a dictionary that maps weight name to corresponding TF variables.
    feed_dict: feed_dict for TF evaluation

  Returns:
    A 1-d numpy array containing the serialized weights
  """
  flattened_weights = []
  for name in sorted(weights.keys()):
    materialized_weight = session.run(weights[name], feed_dict=feed_dict)
    flattened_weights.append(materialized_weight.reshape([-1]))
  return np.hstack(flattened_weights)


def deserialize_weights(weights_variable, flattened_weights):
  """Deserializes the weights into a dictionary that maps name to values.

  The schema is provided by the weights_variable, which is a dictionary that
  maps weight names to corresponding TF variables (i.e. the output of
  construct_network)

  Args:
    weights_variable: a dictionary that maps weight names to corresponding TF
      variables
    flattened_weights: a 1-d array of weights to deserialize

  Returns:
    A dictionary that maps weight names to weight values
  """
  ans = {}
  idx = 0
  for name in sorted(weights_variable.keys()):
    len_current_weight = np.prod(weights_variable[name].shape)
    flattened_weight = np.array(flattened_weights[idx:idx + len_current_weight])
    ans[name] = flattened_weight.reshape(weights_variable[name].shape)
    idx += len_current_weight
  return ans


class MAMLNetworkGenerator(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def construct_network_weights(self, scope='weights'):
    pass

  @abstractmethod
  def construct_network(self, network_input, weights, scope='network'):
    pass


class FullyConnectedNetworkGenerator(MAMLNetworkGenerator):
  """Generator for fully connected networks."""

  def __init__(self,
               dim_input=1,
               dim_output=1,
               layer_sizes=(64,),
               activation_fn=tf.nn.tanh):
    """Creates fully connected neural networks.

    Args:
      dim_input: Dimensionality of input (integer > 0).
      dim_output: Dimensionality of output (integer > 0).
      layer_sizes: non-empty list with number of neurons per internal layer.
      activation_fn: activation function for hidden layers
    """
    self.dim_input = dim_input
    self.dim_output = dim_output
    self.layer_sizes = layer_sizes
    self.activation_fn = activation_fn

  def construct_network_weights(self, scope='weights'):
    """Creates weights for fully connected neural network.

    Args:
      scope: variable scope

    Returns:
      A dict with weights (network parameters).
    """
    weights = {}
    with tf.variable_scope(scope):
      weights['w_0'] = tf.Variable(
          tf.truncated_normal([self.dim_input, self.layer_sizes[0]],
                              stddev=0.1),
          name='w_0')
      weights['b_0'] = tf.Variable(tf.zeros([self.layer_sizes[0]]), name='b_0')
      for i in range(1, len(self.layer_sizes)):
        weights['w_%d' % i] = tf.Variable(
            tf.truncated_normal([self.layer_sizes[i - 1], self.layer_sizes[i]],
                                stddev=0.1),
            name='w_%d' % i)
        weights['b_%d' % i] = tf.Variable(
            tf.zeros([self.layer_sizes[i]]), name='b_%d' % i)
      weights['w_out'] = tf.Variable(
          tf.truncated_normal([self.layer_sizes[-1], self.dim_output],
                              stddev=0.1),
          name='w_out')
      weights['b_out'] = tf.Variable(tf.zeros([self.dim_output]), name='b_out')
    return weights

  def construct_network(self, network_input, weights, scope='network'):
    """Creates a fully connected neural network with given weights and input.

    Args:
     network_input: Network input (1d).
     weights: network parameters (see construct_network_weights).
     scope: name scope.

    Returns:
      neural network output op
    """
    num_layers = len(self.layer_sizes)
    with tf.name_scope(scope):
      hidden = self.activation_fn(
          tf.nn.xw_plus_b(
              network_input, weights['w_0'], weights['b_0'], name='hidden_0'))
      for i in range(1, num_layers):
        hidden = self.activation_fn(
            tf.nn.xw_plus_b(
                hidden,
                weights['w_%d' % i],
                weights['b_%d' % i],
                name='hidden_%d' % i))
      return tf.nn.xw_plus_b(
          hidden, weights['w_out'], weights['b_out'], name='output')


class LinearNetworkGenerator(MAMLNetworkGenerator):
  """Generator for simple linear connections (Y = W*X+b)."""

  def __init__(self, dim_input=1, dim_output=1):
    """Linear transformation with dim_input inputs and dim_output outputs.

    Args:
      dim_input: Dimensionality of input (integer > 0).
      dim_output: Dimensionality of output (integer > 0).
    """
    self.dim_input = dim_input
    self.dim_output = dim_output

  def construct_network_weights(self, scope='weights'):
    """Create weights for linear transformation.

    Args:
      scope: variable scope

    Returns:
      A dict with weights (network parameters).
    """
    with tf.variable_scope(scope):
      return {
          'w_out':
              tf.Variable(
                  tf.truncated_normal([self.dim_input, self.dim_output],
                                      stddev=0.1),
                  name='w_out'),
          'b_out':
              tf.Variable(tf.zeros([self.dim_output]), name='b_out'),
      }

  def construct_network(self, network_input, weights, scope='network'):
    """Create ops for linear transformation.

    Args:
     network_input: Network input (1d).
     weights: network parameters (see construct_network_weights).
     scope: name scope.

    Returns:
      output op
    """
    with tf.name_scope(scope):
      return tf.nn.xw_plus_b(
          network_input, weights['w_out'], weights['b_out'], name='output')
