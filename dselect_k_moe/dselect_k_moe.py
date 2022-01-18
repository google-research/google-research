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

"""A Keras layer for the dSelect-K MoE gate."""

import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# Small constant used to ensure numerical stability.
EPSILON = 1e-6


class SmoothStep(tf.keras.layers.Layer):
  """A smooth-step function.

  For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.

  See https://arxiv.org/abs/2002.07772 for more details on this function.
  """

  def __init__(self, gamma = 1.0):
    """Initializes the layer.

    Args:
      gamma: Scaling parameter controlling the width of the polynomial region.
    """
    super(SmoothStep, self).__init__()
    self._lower_bound = -gamma / 2
    self._upper_bound = gamma / 2
    self._a3 = -2 / (gamma**3)
    self._a1 = 3 / (2 * gamma)
    self._a0 = 0.5

  def call(self, inputs):
    return tf.where(
        inputs <= self._lower_bound, tf.zeros_like(inputs),
        tf.where(inputs >= self._upper_bound, tf.ones_like(inputs),
                 self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


class EntropyRegularizer(tf.keras.layers.Layer):
  """A regularizer that minimizes entropy.

  This class maintains a counter:
    num_calls = num_calls_per_training_step * num_training_steps
  to allow the regularization parameter to change over the course of training.
  """

  def __init__(
      self,
      schedule_fn = lambda x: 1e-6):
    """Initializes the layer.

    Args:
      schedule_fn: A callable that returns the regularization parameter based on
        the num_calls counter.
    """
    super(EntropyRegularizer, self).__init__()
    self._num_calls = self.add_weight(
        name="num_calls", shape=[], initializer="zeros", trainable=False)
    self._schedule_fn = schedule_fn

  def call(self, inputs):
    assign_op = self._num_calls.assign_add(1, read_value=False)

    preconditions = [] if assign_op is None else [assign_op]
    with tf.control_dependencies(preconditions):
      reg_param = self._schedule_fn(self._num_calls)
      entropy = -tf.math.reduce_sum(inputs * tf.math.log(inputs + EPSILON))
      return reg_param * entropy


class DSelectKGate(tf.keras.layers.Layer):
  """A custom layer for selecting a sparse mixture of experts.

  Let f_1, f_2, ..., f_n be the experts. The layer returns:

              a_1 * f_1 + a_2 * f_2 + ... + a_n * f_n,

  where the mixture weights satisfy a_1, ..., a_n >= 0 and a_1 + ... + a_n = 1.
  The number of non-zeros in the mixture weights can be directly controlled.
  The layer is differentiable and can be trained using first-order methods like
  SGD.

  Input: For task-only conditioning, the input should be a list of tensors,
    each corresponding to an expert. For example conditioning, the input should
    be a tuple of the form: (experts, routing_inputs), where experts is a list
    of expert tensors, and routing_inputs is a 2D tensor of input examples.
    Note: In both cases, the expert tensors should have the same shape.
  Output: Tensor, with the same shape as the expert tensors.

  Example:
    # Construct a DSelectKGate to select 2 out of 4 experts.
    gate = DSelectKGate(num_nonzeros=2)
    # output_tensor is a sparse mixture of the 4 tensors in the inputs.
    output_tensor = gate(inputs)
  """

  def __init__(
      self,
      num_nonzeros,
      gamma = 1.0,
      entropy_reg = None,
      z_initializer = None,
      w_initializer = None
  ):
    """DSelectKGate constructor.

    Args:
      num_nonzeros: An upper bound on the number of non-zeros in the mixture
        weights (a_1, ..., a_n), where n is the number of experts inferred from
        the inputs to this layer.
      gamma: A scaling parameter for the smooth-step function.
      entropy_reg: An optional `EntropyRegularizer`. This regularizer, if set,
        can be used to control the speed of convergence of the gate during
        training to the desired number of non-zeros.
      z_initializer: An optional initializer for z_logits.
      w_initializer: An optional initializer for w_logits.
    """
    super(DSelectKGate, self).__init__()
    self._num_nonzeros = num_nonzeros
    self._smooth_step = SmoothStep(gamma)
    self._entropy_reg = entropy_reg
    self._z_initializer = z_initializer or tf.keras.initializers.RandomUniform(
        -gamma / 100, gamma / 100)
    self._w_initializer = w_initializer or tf.keras.initializers.RandomUniform()

  def build(
      self, input_shape
  ):
    """Creates the layer's internal variables."""

    if isinstance(input_shape, tuple):
      expert_shapes, routing_input_shape = input_shape
    else:
      expert_shapes, routing_input_shape = input_shape, None
    num_experts = len(expert_shapes)
    # num_binary is the number of binary vars required to encode the
    # num_experts choices.
    self._num_binary = math.ceil(math.log2(num_experts))
    # Boolean to check if num_experts is a power of 2.
    self._power_of_2 = (num_experts == 2**self._num_binary)
    if routing_input_shape is None:
      # z_logits is a trainable 3D tensor used for selecting the experts.
      # Axis 0: Number of non-zero experts to select.
      # Axis 1: Dummy axis of length 1 used for broadcasting.
      # Axis 2: Each num_binary-dimensional row corresponds to a "single-expert"
      # selector.
      self._z_logits = self.add_weight(
          name="z_logits",
          shape=(self._num_nonzeros, 1, self._num_binary),
          initializer=self._z_initializer,
          trainable=True)
      # w_logits is a trainable tensor used to assign weights to the
      # single-expert selectors. Each element of w_logits is a logit.
      self._w_logits = self.add_weight(
          name="w_logits",
          shape=(self._num_nonzeros, 1),
          initializer=self._w_initializer,
          trainable=True)
    else:
      self._z_logits = tf.keras.layers.Dense(
          self._num_nonzeros * self._num_binary,
          kernel_initializer=self._z_initializer,
          bias_initializer=self._z_initializer)
      self._w_logits = tf.keras.layers.Dense(
          self._num_nonzeros,
          kernel_initializer=self._w_initializer,
          bias_initializer=self._w_initializer)
    # binary_matrix is a (num_experts, num_binary)-matrix used for binary
    # encoding. The i-th row contains a num_binary-digit binary encoding of the
    # integer i.
    binary_matrix = np.array([
        list(np.binary_repr(val, width=self._num_binary))
        for val in range(num_experts)
    ]).astype(bool)
    # A constant tensor = binary_matrix, with an additional dimension for
    # broadcasting.
    self._binary_codes = tf.expand_dims(
        tf.constant(binary_matrix, dtype=bool), axis=0)
    self.built = True

  def call(self,
           inputs,
           training = False):
    """Calls the layer.

    Args:
      inputs: For task-only conditioning, the input should be a list of tensors,
      each corresponding to an expert. For example conditioning, the input
      should be a tuple of the form: (experts, routing_inputs), where experts is
      a list of expert tensors, and routing_inputs is a 2D tensor of input
      examples.
      training: True if the call is in training mode.

    Returns:
      A (single) tensor representing the mixture of experts.
    """
    if isinstance(inputs, tuple):
      experts, routing_inputs = inputs
    else:
      experts, routing_inputs = inputs, None
    if routing_inputs is not None:
      # Example-conditioned routing.
      expert_weights, selector_outputs = (
          self._compute_example_conditioned_expert_weights(routing_inputs))
      output = tf.math.accumulate_n([
          tf.reshape(expert_weights[:, i], [-1, 1]) * experts[i]
          for i in range(len(experts))
      ])
    else:
      # Task-only routing.
      expert_weights, selector_outputs = self._compute_expert_weights()
      output = tf.math.accumulate_n(
          [expert_weights[i] * experts[i] for i in range(len(experts))])
    if training:
      self._add_regularization_loss(selector_outputs)

    return output

  def _compute_expert_weights(self):
    """Computes the weight vector for the experts.

    Args: None.

    Returns:
      A tuple: (expert_weights, selector_outputs).
        expert_weights is the final weight vector of the experts.
        selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
        represents the outputs of the i-th single-expert selector.
    """
    # Shape = (num_nonzero, 1, num_binary).
    smooth_step_activations = self._smooth_step(self._z_logits)
    # Shape = (num_nonzero, num_experts).
    selector_outputs = tf.math.reduce_prod(
        tf.where(self._binary_codes, smooth_step_activations,
                 1 - smooth_step_activations),
        axis=2)
    # Weights for the single-expert selectors: shape = (num_nonzero, 1).
    selector_weights = tf.nn.softmax(self._w_logits, axis=0)
    expert_weights = tf.math.reduce_sum(
        selector_weights * selector_outputs, axis=0)

    return expert_weights, selector_outputs

  def _compute_example_conditioned_expert_weights(
      self, routing_inputs):
    """Computes the example-conditioned weights for the experts.

    Args:
      routing_inputs: a tensor of shape=(batch_size, num_features) containing
        the input examples.

    Returns:
      A tuple: (expert_weights, selector_outputs).
        expert_weights is a tensor with shape=(batch_size, num_experts),
        containing the expert weights for each example in routing_inputs.
        selector_outputs is a tensor with
        shape=(batch_size, num_nonzero, num_experts), which contains the outputs
        of the single-expert selectors for all the examples in routing_inputs.
    """
    sample_logits = tf.reshape(
        self._z_logits(routing_inputs),
        [-1, self._num_nonzeros, 1, self._num_binary])
    smooth_step_activations = self._smooth_step(sample_logits)
    # Shape = (batch_size, num_nonzeros, num_experts).
    selector_outputs = tf.math.reduce_prod(
        tf.where(
            tf.expand_dims(self._binary_codes, 0), smooth_step_activations,
            1 - smooth_step_activations), 3)
    # Weights for the single-expert selectors.
    # Shape = (batch_size, num_nonzeros, 1).
    selector_weights = tf.expand_dims(self._w_logits(routing_inputs), 2)
    selector_weights = tf.nn.softmax(selector_weights, axis=1)
    # Sum over the single-expert selectors. Shape = (batch_size, num_experts).
    expert_weights = tf.math.reduce_sum(
        selector_weights * selector_outputs, axis=1)

    return expert_weights, selector_outputs

  def _add_regularization_loss(self, selector_outputs):
    """Adds regularization loss based on the selector outputs.

    Args:
      selector_outputs: a tensor with shape (batch_size, num_nonzero,
        num_experts) or (num_nonzero, num_experts), where the last dimension
        stores the weight vector of each single-expert selector.
    """
    if self._entropy_reg is not None:
      # Add entropy regularization to each single-expert selector to encourage
      # sparsity.
      self.add_loss(self._entropy_reg(selector_outputs))

    if not self._power_of_2:
      # If the number of experts is not a power of 2, we add a regularization
      # term to prevent the "non-reachable" experts from getting all the nonzero
      # weights for any single-expert selector. The regularization term is equal
      # to 1/sum(weights of reachable experts) so that the reachable experts
      # cannot get zero weights.
      # In case of example conditioning, this regularizer is added per example.
      # NOTE: This regularization term has no effect once the sum of the weights
      # of the reachable experts reaches 1, which is the typical/expected case.
      self.add_loss(
          tf.math.reduce_sum(1 / tf.math.reduce_sum(selector_outputs, axis=-1)))
