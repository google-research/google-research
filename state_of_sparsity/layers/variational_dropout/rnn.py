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

"""Defines variational dropout recurrent layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from state_of_sparsity.layers.utils import rnn_checks
from state_of_sparsity.layers.variational_dropout import common
from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import


# TODO(tgale): This RNN cell and the following LSTM cell share a large
# amount of common code. It would be best if we could extract a
# common base class, and the implement the recurrent functionality
# from scratch (as opposed to  deriving from the non-variational
# reccurent cells.
class RNNCell(tf.nn.rnn_cell.BasicRNNCell):
  """RNN cell trained with variational dropout.

  This class implements an RNN cell trained with variational dropout following
  the technique from https://arxiv.org/abs/1708.00077.
  """

  def __init__(self,
               kernel_weights,
               bias_weights,
               num_units,
               training=True,
               threshold=3.0,
               eps=common.EPSILON,
               activation=None,
               name=None):
    R"""Initialize the variational RNN cell.

    Args:
      kernel_weights: 2-tuple of Tensors, where the first tensor is the \theta
        values and the second contains the log of the \sigma^2 values.
      bias_weights: The weight matrix to use for the biases.
      num_units: int, The number of units in the RNN cell.
      training: boolean, Whether the model is training or being evaluated.
      threshold: Weights with a log \alpha_{ij} value greater than this will
       be set to zero.
      eps: Small constant value to add to the term inside the square-root
        operation to avoid NaNs.
      activation: Activation function of the inner states. Defaults to `tanh`.
      name: String, the name of the layer.

    Raises:
      RuntimeError: If the input variational_params is not a 2-tuple of Tensors
        that have the same shape.
    """
    super(RNNCell, self).__init__(
        num_units=num_units,
        activation=activation,
        reuse=None,
        name=name,
        dtype=None)

    # Verify and save the weight matrices
    rnn_checks.check_rnn_weight_shapes(kernel_weights, bias_weights, num_units)
    self._variational_params = kernel_weights
    self._bias = bias_weights

    self._training = training
    self._threshold = threshold
    self._eps = eps

  def build(self, inputs_shape):
    """Initializes noise samples for the RNN.

    Args:
      inputs_shape: The shape of the input batch.

    Raises:
      RuntimeError: If the first and last dimensions of the input shape are
        not defined.
    """
    inputs_shape = inputs_shape.as_list()
    if inputs_shape[-1] is None:
      raise RuntimeError("Expected inputs.shape[-1] to be known, saw shape {}"
                         .format(inputs_shape))
    if inputs_shape[0] is None:
      raise RuntimeError("Expected inputs.shape[0] to be known, saw shape {}"
                         .format(inputs_shape))
    self._batch_size = inputs_shape[0]
    self._data_size = inputs_shape[-1]

    with ops.init_scope():
      if self._training:
        # Setup the random noise which should be sampled once per-iteration
        self._input_noise = tf.random_normal(
            [self._batch_size, self._num_units])
        self._hidden_noise = tf.random_normal(
            [self._num_units, self._num_units])
      else:
        # Mask the weights ahead of time for efficiency
        theta, log_sigma2 = self._variational_params
        log_alpha = common.compute_log_alpha(
            log_sigma2, theta, self._eps, value_limit=None)

        weight_mask = tf.cast(tf.less(log_alpha, self._threshold), tf.float32)
        self._masked_weights = weight_mask * theta
    self.built = True

  def _compute_gate_inputs(
      self,
      inputs,
      state,
      input_parameters,
      hidden_parameters,
      input_noise,
      hidden_noise):
    """Compute a gate pre-activation with variational dropout.

    Args:
      inputs: The input batch feature timesteps.
      state: The input hidden state from the last timestep.
      input_parameters: The posterior parameters for the input-to-hidden
        connections.
      hidden_parameters: The posterior parameters for the hidden-to-hidden
        connections.
      input_noise: Normally distributed random noise used to for the
        sampling of pre-activations from the input-to-hidden weight
        posterior.
      hidden_noise: Normally distribution random noise use for the
        sampling of pre-activations from the hidden-to-hidden weight
        posterior.

    Returns:
      A tf.Tensor containing the computed pre-activations.
    """
    input_theta, input_log_sigma2 = input_parameters
    hidden_theta, hidden_log_sigma2 = hidden_parameters

    # Compute the input-to-hidden connections
    input_mu = tf.matmul(inputs, input_theta)
    input_sigma = tf.sqrt(tf.matmul(
        tf.square(inputs),
        tf.exp(input_log_sigma2)) + self._eps)

    input_to_hidden = input_mu + input_sigma * input_noise

    # Compute the hidden-to-hidden connections
    hidden_sigma = tf.sqrt(tf.exp(hidden_log_sigma2) + self._eps)
    hidden_weights = hidden_theta + hidden_sigma * hidden_noise
    hidden_to_hidden = tf.matmul(state, hidden_weights)

    # Sum the results
    return tf.add(input_to_hidden, hidden_to_hidden)

  def _forward_train(self, inputs, state):
    # Split the input-to-hidden and hidden-to-hidden weights
    theta, log_sigma2 = self._variational_params
    input_theta, hidden_theta = tf.split(
        theta, [self._data_size, self._num_units])
    input_log_sigma2, hidden_log_sigma2 = tf.split(
        log_sigma2, [self._data_size, self._num_units])

    gate_inputs = self._compute_gate_inputs(
        inputs,
        state,
        (input_theta, input_log_sigma2),
        (hidden_theta, hidden_log_sigma2),
        self._input_noise,
        self._hidden_noise)

    # Add bias, and apply the activation
    gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

  def _forward_eval(self, inputs, state):
    # At eval time, we use the masked mean values for the input-to-hidden
    # and hidden-to-hidden weights.
    gate_inputs = tf.matmul(
        tf.concat([inputs, state], axis=1),
        self._masked_weights)

    gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

  def call(self, inputs, state):
    if self._training:
      return self._forward_train(inputs, state)
    return self._forward_eval(inputs, state)


class LSTMCell(tf.nn.rnn_cell.LSTMCell):
  """LSTM cell trained with variational dropout.

  This class implements an LSTM cell trained with variational dropout following
  the technique from https://arxiv.org/abs/1708.00077.
  """

  def __init__(self,
               kernel_weights,
               bias_weights,
               num_units,
               training=True,
               threshold=3.0,
               eps=common.EPSILON,
               forget_bias=1.0,
               activation=None,
               name="lstm_cell"):
    R"""Initialize the LSTM cell.

    Args:
      kernel_weights: 2-tuple of Tensors, where the first tensor is the \theta
        values and the second contains the log of the \sigma^2 values.
      bias_weights: the weight matrix to use for the biases.
      num_units: int, The number of units in the LSTM cell.
      training: boolean, Whether the model is training or being evaluated.
      threshold: Weights with a log \alpha_{ij} value greater than this will
        be set to zero.
      eps: Small constant value to add to the term inside the square-root
        operation to avoid NaNs.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states. Defaults to `tanh`.
        It could also be string that is within Keras activation function names.
      name: String, the name of the layer.

    Raises:
      RuntimeError: If the input variational_params is not a 2-tuple of Tensors
       that have the same shape.
    """
    super(LSTMCell, self).__init__(
        num_units=num_units,
        forget_bias=forget_bias,
        state_is_tuple=True,
        activation=activation,
        name=name)

    # Verify and save the weight matrices
    rnn_checks.check_lstm_weight_shapes(kernel_weights, bias_weights, num_units)
    self._variational_params = kernel_weights
    self._bias = bias_weights

    self._training = training
    self._threshold = threshold
    self._eps = eps

  def build(self, inputs_shape):
    """Initializes noise samples for the LSTM.

    Args:
      inputs_shape: The shape of the input batch.

    Raises:
      RuntimeError: If the first and last dimensions of the input shape are
        not defined.
    """
    inputs_shape = inputs_shape.as_list()
    if inputs_shape[-1] is None:
      raise RuntimeError("Expected inputs.shape[-1] to be known, saw shape {}"
                         .format(inputs_shape))
    if inputs_shape[0] is None:
      raise RuntimeError("Expected inputs.shape[0] to be known, saw shape {}"
                         .format(inputs_shape))
    self._batch_size = inputs_shape[0]
    self._data_size = inputs_shape[-1]

    with ops.init_scope():
      if self._training:
        # Setup the random noise which should be sampled once per-iteration
        self._input_noise = tf.random_normal(
            [self._batch_size, 4 * self._num_units])
        self._hidden_noise = tf.random_normal(
            [self._num_units, 4 * self._num_units])
      else:
        # Mask the weights ahead of time for efficiency
        theta, log_sigma2 = self._variational_params
        log_alpha = common.compute_log_alpha(
            log_sigma2, theta, self._eps, value_limit=None)

        weight_mask = tf.cast(tf.less(log_alpha, self._threshold), tf.float32)
        self._masked_weights = weight_mask * theta
    self.built = True

  def _compute_gate_inputs(
      self,
      inputs,
      state,
      input_parameters,
      hidden_parameters,
      input_noise,
      hidden_noise):
    """Compute a gate pre-activation with variational dropout.

    Args:
      inputs: The input batch feature timesteps.
      state: The input hidden state from the last timestep.
      input_parameters: The posterior parameters for the input-to-hidden
        connections.
      hidden_parameters: The posterior parameters for the hidden-to-hidden
        connections.
      input_noise: Normally distributed random noise used to for the
        sampling of pre-activations from the input-to-hidden weight
        posterior.
      hidden_noise: Normally distribution random noise use for the
        sampling of pre-activations from the hidden-to-hidden weight
        posterior.

    Returns:
      A tf.Tensor containing the computed pre-activations.
    """
    input_theta, input_log_sigma2 = input_parameters
    hidden_theta, hidden_log_sigma2 = hidden_parameters

    # Compute the input-to-hidden connections
    input_mu = tf.matmul(inputs, input_theta)
    input_sigma = tf.sqrt(tf.matmul(
        tf.square(inputs),
        tf.exp(input_log_sigma2)) + self._eps)

    input_to_hidden = input_mu + input_sigma * input_noise

    # Compute the hidden-to-hidden connections
    hidden_sigma = tf.sqrt(tf.exp(hidden_log_sigma2) + self._eps)
    hidden_weights = hidden_theta + hidden_sigma * hidden_noise
    hidden_to_hidden = tf.matmul(state, hidden_weights)

    # Sum the results
    return tf.add(input_to_hidden, hidden_to_hidden)

  def _finish_lstm_computation(self, lstm_matrix, c_prev):
    i, j, f, o = tf.split(
        value=lstm_matrix,
        num_or_size_splits=4,
        axis=1)

    sigmoid = tf.sigmoid
    c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
         self._activation(j))
    m = sigmoid(o) * self._activation(c)

    new_state = tf.nn.rnn_cell.LSTMStateTuple(c, m)
    return m, new_state

  def _forward_train(self, inputs, state):
    # Split the input-to-hidden and hidden-to-hidden weights
    theta, log_sigma2 = self._variational_params
    input_theta, hidden_theta = tf.split(
        theta, [self._data_size, self._num_units])
    input_log_sigma2, hidden_log_sigma2 = tf.split(
        log_sigma2, [self._data_size, self._num_units])

    (c_prev, m_prev) = state
    lstm_matrix = self._compute_gate_inputs(
        inputs,
        m_prev,
        (input_theta, input_log_sigma2),
        (hidden_theta, hidden_log_sigma2),
        self._input_noise,
        self._hidden_noise)
    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    return self._finish_lstm_computation(lstm_matrix, c_prev)

  def _forward_eval(self, inputs, state):
    (c_prev, m_prev) = state

    lstm_matrix = tf.matmul(
        tf.concat([inputs, m_prev], axis=1),
        self._masked_weights)
    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    return self._finish_lstm_computation(lstm_matrix, c_prev)

  def call(self, inputs, state):
    if self._training:
      return self._forward_train(inputs, state)
    return self._forward_eval(inputs, state)
