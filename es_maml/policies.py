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

"""Contains policies used in MAML."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class Policy(object):
  r"""Abstract class for different policies \Pi: S -> A.

  Class is responsible for creating different policies and provides an interface
  for computing actions recommended by policies in different input states.
  In particular, this class provides an interface that accepts compressed
  vectorized form of the policy and decompresses it.

  Standard procedure for improving the parameters of the policy with an
  interface given by the class:

  policy = policies.ParticularClassThatInheritsFromBaseClass(...)
  vectorized_network = policy.get_initial()
  while(...):
    new_vectorized_network = SomeTransformationOf(vectorized_network)
    policy.update(new_vectorized_network)

  and SomeTransformationOf is a single step of some optimization procedure such
  as gradient descent that sees the policy in the vectorized form.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def update(self, vectorized_parameters):
    """Updates the policy using new parameters from <vectorized_parameters>.

    Updates the parameters of the policy using new parameters encoded by
    <vectorized_parameters>. The size of the vector <vectorized_parameters>
    should be the number of all biases and weights of the neural network.
    We use the convention where parameters encoding matrices of connections of
    the neural network come in <vectorized_parameters> before parameters
    encoding biases and furthermore the order in <vectorized_parameters> of
    parameters encoding weights for different matrices/biases-vectors is
    inherited from the order of these matrices/biases-vectors in the
    decompressed neural network. Details regarding compression depend on
    different neural network architectures used (such as: structured and
    unstructured) and are given in the implementations of that abstract method
    in specific classes that inherit from Policy.

    Args:
      vectorized_parameters: parameters of the neural network in the vectorized
        form.

    Returns:
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_action(self, state):
    """Returns the action proposed by a policy in a given state.

    Returns an action proposed by the policy in <state>.

    Args:
      state: input state

    Returns:
      Action proposed by the policy represented by an object of the class in a
      given state.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_initial(self):
    """Returns the default parameters of the policy in the vectorized form.

    Initial parameters of the policy are output in the vectorized form.

    Args:

    Returns:
      Numpy array encoding in the vectorized form initial parameters of the
      policy.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_total_num_parameters(self):
    """Outputs total number of parameters of the policy.

    Args:

    Returns:
      Total number of parameters used by the policy.
    """
    raise NotImplementedError('Abstract method')


class BasicTFPolicy(Policy):
  """Basic Policy implemented in Tensorflow."""

  def __init__(self, state_dimensionality, action_dimensionality, hidden_layers,
               scope):
    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality

    self.input_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.state_dimensionality])
    self.output_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.action_dimensionality])

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      self.out = self.input_ph
      for i, layer_size in enumerate(hidden_layers):
        self.out = tf.layers.dense(
            self.out, layer_size, activation=tf.nn.relu, name='h' + str(i))

      self.main_out = tf.layers.dense(
          self.out, self.action_dimensionality, name='main_out')
      self.secondary_out = tf.layers.dense(
          self.out, self.action_dimensionality, name='secondary_out')

    self.action = tfp.distributions.Normal(
        loc=self.main_out, scale=self.secondary_out).sample()

    self.loss = tf.losses.mean_squared_error(self.main_out, self.output_ph)
    self.obj_tensor = -1.0 * self.loss
    self.tf_params = tf.trainable_variables(scope)

    self.shapes = [v.shape.as_list() for v in self.tf_params]
    self.sizes = [int(np.prod(s)) for s in self.shapes]
    self.total_nb_parameters = sum(self.sizes)

    self.assign_ph_dict = {
        v: tf.placeholder(dtype=tf.float32, shape=v.shape.as_list())
        for v in self.tf_params
    }
    self.assign_ops = []
    for v in self.tf_params:
      self.assign_ops.append(v.assign(self.assign_ph_dict[v]))

    with tf.control_dependencies(self.assign_ops):
      # This is needed to input Numpy Params into network temporarily
      self.action = tf.identity(self.action)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.np_params = np.concatenate([
        self.sess.run(tf.reshape(tf_param, [-1])) for tf_param in self.tf_params
    ])

  def update(self, flattened_weights):
    self.np_params = flattened_weights

  def get_action(self, state):

    ph_dict = {}
    for ind, v in enumerate(self.tf_params):
      numpy_flat_val = self.np_params[sum(self.sizes[:ind]
                                         ):sum(self.sizes[:ind + 1])]
      numpy_reshaped = np.reshape(numpy_flat_val, self.shapes[ind])
      v_ph = self.assign_ph_dict[v]
      ph_dict[v_ph] = numpy_reshaped

    ph_dict[self.input_ph] = state.reshape(-1, self.state_dimensionality)

    action_numpy = self.sess.run(self.action, feed_dict=ph_dict)
    return action_numpy.flatten()

  def get_initial(self):
    return self.np_params

  def get_total_num_parameters(self):
    return self.total_nb_parameters


class DeterministicNumpyPolicy(Policy):
  """Deterministic Policy implemented in Numpy."""

  def __init__(self,
               state_dimensionality,
               action_dimensionality,
               hidden_layers,
               init_sd=None):
    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality

    self.layers = hidden_layers + [action_dimensionality]
    self.layers.insert(0, state_dimensionality)
    self.weights = []
    self.biases = []
    self.weight_positions = []
    self.bias_positions = []

    self.init_params = []
    flat_pos = 0
    for dims in zip(self.layers[:-1], self.layers[1:]):
      in_size = dims[0]
      out_size = dims[1]

      if init_sd is None:
        init_sd = np.sqrt(2.0 / (in_size))
      init_weights = init_sd * np.random.normal(0, 1, size=(out_size * in_size))
      self.init_params.extend(init_weights.tolist())
      self.weights.append(np.reshape(init_weights, (out_size, in_size)))
      self.weight_positions.append(flat_pos)
      flat_pos += out_size * in_size

      init_biases = np.zeros(out_size)
      self.init_params.extend(init_biases.tolist())
      self.biases.append(init_biases)
      self.bias_positions.append(flat_pos)
      flat_pos += out_size
    self.weight_positions.append(flat_pos)

  def update(self, flat_weights):
    for i, dims in enumerate(zip(self.layers[:-1], self.layers[1:])):
      in_size = dims[0]
      out_size = dims[1]
      start_pos = self.weight_positions[i]
      end_pos = start_pos + (out_size * in_size)
      self.weights[i] = np.reshape(
          np.array(flat_weights[start_pos:end_pos]), (out_size, in_size))

      start_pos = self.bias_positions[i]
      end_pos = start_pos + out_size
      self.biases[i] = np.reshape(
          np.array(flat_weights[start_pos:end_pos]), (out_size))

  def get_action(self, state):
    neuron_values = np.reshape(np.array(state), (self.state_dimensionality))
    for i in range(len(self.weights)):
      neuron_values = np.matmul(self.weights[i], neuron_values)
      neuron_values += self.biases[i]
      if i < len(self.weights) - 1:
        np.maximum(neuron_values, 0, neuron_values)
    np.tanh(neuron_values, neuron_values)  # this is sometimes not needed
    return neuron_values

  def get_initial(self):
    return np.array(self.init_params)

  def get_total_num_parameters(self):
    return self.weight_positions[-1]
