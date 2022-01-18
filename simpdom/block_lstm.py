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

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""LSTM Block Cell ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow.compat.v1 as tf


class LSTMBlockWrapper(tf.keras.layers.Layer):
  """This is a helper class that provides housekeeping for LSTM cells.

  This may be useful for alternative LSTM and similar type of cells.
  The subclasses must implement `_call_cell` method and `num_units` property.
  """

  @abc.abstractproperty
  def num_units(self):
    """Number of units in this cell (output dimension)."""
    pass

  @abc.abstractmethod
  def _call_cell(self, inputs, initial_cell_state, initial_output, dtype,
                 sequence_length):
    """Run this LSTM on inputs, starting from the given state.

    This method must be implemented by subclasses and does the actual work
    of calling the cell.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      initial_cell_state: initial value for cell state, shape `[batch_size,
        self._num_units]`
      initial_output: initial value of cell output, shape `[batch_size,
        self._num_units]`
      dtype: The data type for the initial state and expected output.
      sequence_length: Specifies the length of each sequence in inputs. An int32
        or int64 vector (tensor) size [batch_size], values in [0, time_len) or
          None.

    Returns:
      A pair containing:

      - State: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
      - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
    """
    pass

  def call(self, inputs, initial_state=None, dtype=None, sequence_length=None,
           mask_output=False):
    """Run this LSTM on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple `(initial_cell_state, initial_output)` with tensors
        of shape `[batch_size, self._num_units]`. If this is not provided, the
        cell is expected to create a zero initial state of type `dtype`.
      dtype: The data type for the initial state and expected output. Required
        if `initial_state` is not provided or RNN state has a heterogeneous
        dtype.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len).`
        Defaults to `time_len` for each element.
      mask_output: ...

    Returns:
      A pair containing:

      - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
        or a list of time_len tensors of shape `[batch_size, output_size]`,
        to match the type of the `inputs`.
      - Final state: a tuple `(cell_state, output)` matching `initial_state`.

    Raises:
      ValueError: in case of shape mismatches
    """
    is_list = isinstance(inputs, list)
    if is_list:
      inputs = tf.stack(inputs)
    inputs_shape = inputs.get_shape().with_rank(3)
    if not inputs_shape[2]:
      raise ValueError("Expecting inputs_shape[2] to be set: %s" % inputs_shape)
    batch_size = inputs_shape[1]
    if batch_size is None:
      batch_size = tf.shape(inputs)[1]
    time_len = inputs_shape[0]
    if time_len is None:
      time_len = tf.shape(inputs)[0]

    # Provide default values for initial_state and dtype
    if initial_state is None:
      if dtype is None:
        raise ValueError("Either initial_state or dtype needs to be specified")
      z = tf.zeros(
          tf.stack([batch_size, self.num_units]), dtype=dtype)
      initial_state = z, z
    else:
      if len(initial_state) != 2:
        raise ValueError(
            "Expecting initial_state to be a tuple with length 2 or None")
      if dtype is None:
        dtype = initial_state[0].dtype

    # create the actual cell
    if sequence_length is not None:
      sequence_length = tf.convert_to_tensor(sequence_length)
    initial_cell_state, initial_output = initial_state  # pylint: disable=unpacking-non-sequence
    cell_states, outputs = self._call_cell(
        inputs, initial_cell_state, initial_output, dtype, sequence_length)

    if sequence_length is not None:
      if mask_output:
        # Mask out the part beyond sequence_length.
        # In MLPerf we don't do it b.c output is masked when computing loss.
        # And in inference we don't use this layer.
        mask = tf.transpose(
            tf.sequence_mask(sequence_length, time_len, dtype=dtype),
            [1, 0])
        mask = tf.tile(
            tf.expand_dims(mask, axis=-1), [1, 1, self.num_units])
        outputs *= mask
      # sequence_length can't be zero in our impl, pass sequence_length -1 for
      # indices.
      mod_cell_states = cell_states
      mod_outputs = outputs
      final_cell_state = self._gather_states(mod_cell_states,
                                             sequence_length - 1, batch_size)
      final_output = self._gather_states(mod_outputs, sequence_length - 1,
                                         batch_size)
    else:
      # No sequence_lengths used: final state is the last state
      final_cell_state = cell_states[-1]
      final_output = outputs[-1]

    if is_list:
      # Input was a list, so return a list
      outputs = tf.unstack(outputs)

    final_state = tf.nn.rnn_cell.LSTMStateTuple(final_cell_state, final_output)
    return outputs, final_state

  def _gather_states(self, data, indices, batch_size):
    """Produce `out`, s.t. out(i, j) = data(indices(i), i, j)."""
    gather_indices = tf.stack([indices, tf.range(batch_size)], axis=1)
    # TODO(jamesqin): ScatterNd doesn't support fp16 on GPU.
    return tf.gather_nd(data, gather_indices)


class LSTMBlockFusedCell(LSTMBlockWrapper):
  """FusedRNNCell implementation of LSTM.

  This is an extremely efficient LSTM implementation, that uses a single TF op
  for the entire LSTM. It should be both faster and more memory-efficient than
  LSTMBlockCell defined above.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  The variable naming is consistent with `rnn_cell_impl.LSTMCell`.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               dtype=None,
               name="lstm_cell"):
    """Initialize the LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      cell_clip: clip the cell to this value. Default is no cell clipping.
      use_peephole: Whether to use peephole connections or not.
      dtype: the dtype of variables of this layer.
      name: String, the name of the layer. By default this is "lstm_cell", for
        variable-name compatibility with `tf.nn.rnn_cell.LSTMCell`.
    """
    super(LSTMBlockFusedCell, self).__init__(
        name=name, dtype=dtype)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._use_peephole = use_peephole

    # Inputs must be 3-dimensional.
    self.input_spec = tf.keras.layers.InputSpec(ndim=3)

  @property
  def num_units(self):
    """Number of units in this cell (output dimension)."""
    return self._num_units

  def build(self, input_shape):
    input_size = input_shape[2]
    self._kernel = self.add_variable(
        "kernel", [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        "bias", [self._num_units * 4],
        initializer=tf.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable("w_i_diag", [self._num_units])
      self._w_f_diag = self.add_variable("w_f_diag", [self._num_units])
      self._w_o_diag = self.add_variable("w_o_diag", [self._num_units])

    self.built = True

  def _call_cell(self,
                 inputs,
                 initial_cell_state=None,
                 initial_output=None,
                 dtype=None,
                 sequence_length=None):
    """Run this LSTM on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      initial_cell_state: initial value for cell state, shape `[batch_size,
        self._num_units]`
      initial_output: initial value of cell output, shape `[batch_size,
        self._num_units]`
      dtype: The data type for the initial state and expected output.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len)` or None.

    Returns:
      A pair containing:

      - Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                         output_size]`
      - Output (h): A `3-D` tensor of shape `[time_len, batch_size,
                    output_size]`
    """

    inputs_shape = inputs.get_shape().with_rank(3)
    time_len = inputs_shape[0]
    if time_len is None:
      time_len = tf.shape(inputs)[0]

    if self._use_peephole:
      wci = self._w_i_diag
      wco = self._w_o_diag
      wcf = self._w_f_diag
    else:
      wci = wcf = wco = tf.zeros([self._num_units], dtype=dtype)

    if sequence_length is None:
      max_seq_len = tf.to_int64(time_len)
    else:
      max_seq_len = tf.to_int64(tf.reduce_max(sequence_length))

    _, cs, _, _, _, _, h = tf.raw_ops.BlockLSTM(
        seq_len_max=max_seq_len,
        x=inputs,
        cs_prev=initial_cell_state,
        h_prev=initial_output,
        w=self._kernel,
        wci=wci,
        wcf=wcf,
        wco=wco,
        b=self._bias,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)
    return cs, h


class TimeReversedFusedLSTM(object):
  """This is an adaptor to time-reverse a LSTMBlockFusedCell.
  """

  def __init__(self, cell):
    self._cell = cell

  def _reverse(self, t, lengths):
    """Time reverse the provided tensor or list of tensors.

    Assumes the top dimension is the time dimension.

    Args:
      t: 3D tensor or list of 2D tensors to be reversed
      lengths: 1D tensor of lengths, or `None`

    Returns:
      A reversed tensor or list of tensors
    """
    if isinstance(t, list):
      return list(reversed(t))
    else:
      if lengths is None:
        return tf.reverse(t, [0])
      else:
        return tf.reverse_sequence(t, lengths, 0, 1)

  def __call__(self,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None):
    inputs = self._reverse(inputs, sequence_length)
    outputs, state = self._cell(
        inputs,
        initial_state=initial_state,
        dtype=dtype,
        sequence_length=sequence_length)
    outputs = self._reverse(outputs, sequence_length)
    return outputs, state
