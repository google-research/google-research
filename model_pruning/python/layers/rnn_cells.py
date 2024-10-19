# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Module implementing RNN Cells with pruning.

This module implements BasicLSTMCell and LSTMCell with pruning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from model_pruning.python import pruning


def _CreateLSTMPruneVariables(lstm_obj, input_depth, h_depth):
  """Function to create additional variables for pruning."""

  mask = lstm_obj.add_variable(
      name="mask",
      shape=[input_depth + h_depth, 4 * h_depth],
      initializer=tf.ones_initializer(),
      trainable=False,
      dtype=lstm_obj.dtype)
  threshold = lstm_obj.add_variable(
      name="threshold",
      shape=[],
      initializer=tf.zeros_initializer(),
      trainable=False,
      dtype=lstm_obj.dtype)
  # Add old_weights, old_old_weights, gradient for gradient
  # based pruning.
  old_weight = lstm_obj.add_variable(
      name="old_weight",
      shape=[input_depth + h_depth, 4 * h_depth],
      initializer=tf.zeros_initializer(),
      trainable=False,
      dtype=lstm_obj.dtype)
  old_old_weight = lstm_obj.add_variable(
      name="old_old_weight",
      shape=[input_depth + h_depth, 4 * h_depth],
      initializer=tf.zeros_initializer(),
      trainable=False,
      dtype=lstm_obj.dtype)
  gradient = lstm_obj.add_variable(
      name="gradient",
      shape=[input_depth + h_depth, 4 * h_depth],
      initializer=tf.zeros_initializer(),
      trainable=False,
      dtype=lstm_obj.dtype)

  return mask, threshold, old_weight, old_old_weight, gradient


class MaskedBasicLSTMCell(tf.compat.v1.nn.rnn_cell.BasicLSTMCell):
  """Basic LSTM recurrent network cell with pruning.

  Overrides the call method of tensorflow BasicLSTMCell and injects the weight
  masks

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full `tf.compat.v1.nn.rnn_cell.LSTMCell`
  that follows.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None):
    """Initialize the basic LSTM cell with pruning.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above). Must set
        to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases. When
        restoring from CudnnLSTM-trained checkpoints, must use
        CudnnCompatibleLSTMCell instead.
    """
    super(MaskedBasicLSTMCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple,
        activation=activation,
        reuse=reuse,
        name=name)

  def build(self, inputs_shape):
    # Call the build method of the parent class.
    super(MaskedBasicLSTMCell, self).build(inputs_shape)

    self.built = False

    input_depth = inputs_shape.dims[1].value
    h_depth = self._num_units

    (self._mask, self._threshold, self._old_weight, self._old_old_weight,
     self._gradient) = _CreateLSTMPruneVariables(self, input_depth, h_depth)
    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    self._masked_kernel = tf.multiply(self._mask, self._kernel,
                                      pruning.MASKED_WEIGHT_NAME)

    if self._mask not in tf.get_collection_ref(pruning.MASK_COLLECTION):
      tf.add_to_collection(pruning.MASK_COLLECTION, self._mask)
      tf.add_to_collection(pruning.MASKED_WEIGHT_COLLECTION,
                           self._masked_kernel)
      tf.add_to_collection(pruning.THRESHOLD_COLLECTION, self._threshold)
      tf.add_to_collection(pruning.WEIGHT_COLLECTION, self._kernel)
      tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION, self._old_weight)
      tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION,
                           self._old_old_weight)
      tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION, self._gradient)

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM) with masks for pruning.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped `[batch_size,
        self.state_size]`, if `state_is_tuple` has been set to `True`.
        Otherwise, a `Tensor` shaped `[batch_size, 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = tf.sigmoid
    one = tf.constant(1, dtype=tf.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = tf.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = tf.matmul(tf.concat([inputs, h], 1), self._masked_kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = tf.add
    multiply = tf.multiply
    new_c = add(
        multiply(c, sigmoid(add(f, forget_bias_tensor))),
        multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
      new_state = tf.concat([new_c, new_h], 1)
    return new_h, new_state


class MaskedLSTMCell(tf.compat.v1.nn.rnn_cell.LSTMCell):
  """LSTMCell with pruning.

  Overrides the call method of tensorflow LSTMCell and injects the weight masks.
  Masks are applied to only the weight matrix of the LSTM and not the
  projection matrix.
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=None,
               num_proj_shards=None,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None):
    """Initialize the parameters for an LSTM cell with masks for pruning.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1 in
        order to reduce the scale of forgetting at the beginning of the
        training. Must set it manually to `0.0` when restoring from CudnnLSTM
        trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.  When restoring from
        CudnnLSTM-trained checkpoints, must use CudnnCompatibleLSTMCell instead.
    """
    super(MaskedLSTMCell, self).__init__(
        num_units,
        use_peepholes=use_peepholes,
        cell_clip=cell_clip,
        initializer=initializer,
        num_proj=num_proj,
        proj_clip=proj_clip,
        num_unit_shards=num_unit_shards,
        num_proj_shards=num_proj_shards,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple,
        activation=activation,
        reuse=reuse)

  def build(self, inputs_shape):
    # Call the build method of the parent class.
    super(MaskedLSTMCell, self).build(inputs_shape)

    self.built = False

    input_depth = inputs_shape.dims[1].value
    h_depth = self._num_units

    (self._mask, self._threshold, self._old_weight, self._old_old_weight,
     self._gradient) = _CreateLSTMPruneVariables(self, input_depth, h_depth)
    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    self._masked_kernel = tf.multiply(self._mask, self._kernel,
                                      pruning.MASKED_WEIGHT_NAME)

    if self._mask not in tf.get_collection_ref(pruning.MASK_COLLECTION):
      tf.add_to_collection(pruning.MASK_COLLECTION, self._mask)
      tf.add_to_collection(pruning.MASKED_WEIGHT_COLLECTION,
                           self._masked_kernel)
      tf.add_to_collection(pruning.THRESHOLD_COLLECTION, self._threshold)
      tf.add_to_collection(pruning.WEIGHT_COLLECTION, self._kernel)
      tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION, self._old_weight)
      tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION,
                           self._old_old_weight)
      tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION, self._gradient)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,
        [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple
        of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = tf.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2).dims[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = tf.matmul(tf.concat([inputs, m_prev], 1), self._masked_kernel)
    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes:
      c = (
          sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
          sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (
          sigmoid(f + self._forget_bias) * c_prev +
          sigmoid(i) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = tf.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = tf.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (
        tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c, m)
        if self._state_is_tuple else tf.concat([c, m], 1))
    return m, new_state
