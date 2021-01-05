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

"""Defines recurrent network layers that train using l0 regularization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from state_of_sparsity.layers.l0_regularization import common
from state_of_sparsity.layers.utils import rnn_checks
from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import


class RNNCell(tf.nn.rnn_cell.BasicRNNCell):
  """RNN cell trained with l0 regularization.

  This class implements an RNN cell trained with l0 regularization following
  the technique from https://arxiv.org/abs/1712.01312.
  """

  def __init__(
      self,
      kernel_weights,
      bias_weights,
      num_units,
      beta=common.BETA,
      gamma=common.GAMMA,
      zeta=common.ZETA,
      training=True,
      eps=common.EPSILON,
      activation=None,
      name=None):
    R"""Initialize the RNN cell.

    Args:
      kernel_weights: 2-tuple of Tensors, where the first tensor is the unscaled
        weight values and the second is the log of the alpha values for the hard
        concrete distribution.
      bias_weights: The weight matrix to use for the biases.
      num_units: int, The number of units in the RNN cell.
      beta: The beta parameter, which controls the "temperature" of
        the distribution. Defaults to 2/3 from the above paper.
      gamma: The gamma parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      zeta: The zeta parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
      training: boolean, Whether the model is training or being evaluated.
      eps: Small constant value to add to the term inside the square-root
        operation to avoid NaNs.
      activation: Activation function of the inner states. Defaults to `tanh`.
      name: String, the name of the layer.

    Raises:
      RuntimeError: If the input kernel_weights is not a 2-tuple of Tensors
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
    self._weight_parameters = kernel_weights
    self._bias = bias_weights

    self._beta = beta
    self._gamma = gamma
    self._zeta = zeta

    self._training = training
    self._eps = eps

  def build(self, _):
    """Initializes the weights for the RNN."""
    with ops.init_scope():
      theta, log_alpha = self._weight_parameters
      if self._training:
        weight_noise = common.hard_concrete_sample(
            log_alpha,
            self._beta,
            self._gamma,
            self._zeta,
            self._eps)
      else:
        weight_noise = common.hard_concrete_mean(
            log_alpha,
            self._gamma,
            self._zeta)
      self._weights = weight_noise * theta
    self.built = True

  def call(self, inputs, state):
    gate_inputs = tf.matmul(
        tf.concat([inputs, state], axis=1),
        self._weights)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output


class LSTMCell(tf.nn.rnn_cell.LSTMCell):
  """LSTM cell trained with l0 regularization.

  This class implements an LSTM cell trained with l0 regularization following
  the technique from https://arxiv.org/abs/1712.01312.
  """

  def __init__(
      self,
      kernel_weights,
      bias_weights,
      num_units,
      beta=common.BETA,
      gamma=common.GAMMA,
      zeta=common.ZETA,
      training=True,
      eps=common.EPSILON,
      forget_bias=1.0,
      activation=None,
      name="lstm_cell"):
    R"""Initialize the LSTM cell.

    Args:
      kernel_weights: 2-tuple of Tensors, where the first tensor is the unscaled
        weight values and the second is the log of the alpha values for the hard
        concrete distribution.
      bias_weights: the weight matrix to use for the biases.
      num_units: int, The number of units in the LSTM cell.
      beta: The beta parameter, which controls the "temperature" of
        the distribution. Defaults to 2/3 from the above paper.
      gamma: The gamma parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      zeta: The zeta parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
      training: boolean, Whether the model is training or being evaluated.
      eps: Small constant value to add to the term inside the square-root
        operation to avoid NaNs.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states. Defaults to `tanh`.
        It could also be string that is within Keras activation function names.
      name: String, the name of the layer.

    Raises:
      RuntimeError: If the input kernel_weights is not a 2-tuple of Tensors
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
    self._weight_parameters = kernel_weights
    self._bias = bias_weights

    self._beta = beta
    self._gamma = gamma
    self._zeta = zeta

    self._training = training
    self._eps = eps

  def build(self, _):
    """Initialize the weights for the LSTM."""
    with ops.init_scope():
      theta, log_alpha = self._weight_parameters
      if self._training:
        weight_noise = common.hard_concrete_sample(
            log_alpha,
            self._beta,
            self._gamma,
            self._zeta,
            self._eps)
      else:
        weight_noise = common.hard_concrete_mean(
            log_alpha,
            self._gamma,
            self._zeta)
      self._weights = weight_noise * theta
    self.built = True

  def call(self, inputs, state):
    (c_prev, m_prev) = state
    lstm_matrix = tf.matmul(
        tf.concat([inputs, m_prev], axis=1),
        self._weights)
    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
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
