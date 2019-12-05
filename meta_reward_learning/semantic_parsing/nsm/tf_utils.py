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

"""TensorFlow utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

import tensorflow as tf
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

rnn = contrib_rnn
_BIAS_VARIABLE_NAME = 'biases'
_WEIGHTS_VARIABLE_NAME = 'weights'
ACTIVATION_DICT = dict(sigmoid=tf.nn.sigmoid, tanh=tf.nn.tanh)


def dice(x):
  """DiCE: The Infinitely Differentiable Monte-Carlo Estimator."""
  return tf.exp(x - tf.stop_gradient(x))


def st_estimator(f, g):
  """Function which acts as f in the forward pass and g in the backward pass."""
  return tf.stop_gradient(f - g) + g


def create_var_and_placeholder(name,
                               shape,
                               dtype,
                               trainable=False,
                               initializer=None,
                               default=None):
  """Creates a variable and a corresponding initializer op with placeholder."""
  if default is not None:
    placeholder = tf.placeholder_with_default(
        tf.constant(default, dtype), shape=shape)
  else:
    placeholder = tf.placeholder(
        dtype, shape=shape, name='{}_init_pc'.format(name))

  variable = tf.get_variable(
      name,
      dtype=dtype,
      shape=shape,
      initializer=initializer,
      trainable=trainable)
  init_op = variable.assign(placeholder)
  return variable, placeholder, init_op


def tensormul(t1, t2):
  """Basically matmul, but t1 can have more dimensions than t2."""
  dim1 = t1.get_shape().as_list()[-1]
  dim2 = t2.get_shape().as_list()[-1]
  result_shape_tensors = tf.unstack(tf.shape(t1))
  result_shape_tensors[-1] = dim2
  result_shape_tensor = tf.stack(result_shape_tensors)
  t1 = tf.reshape(t1, [-1, dim1])
  result = tf.matmul(t1, t2)
  result = tf.reshape(result, result_shape_tensors)
  return result


@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs['reuse'] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, '_scope'):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            'Attempt to reuse RNNCell %s with a different variable scope than '
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            'cell if you would like it to use a different set of weights.  '
            'If before you were using: MultiRNNCell([%s(...)] * num_layers), '
            'change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  '
            'If before you were using the same cell instance as both the '
            'forward and reverse cell of a bidirectional RNN, simply create '
            'two instances (one for forward, one for reverse).  '
            "In May 2017, we will start transitioning this cell's behavior "
            'to use existing stored weights, if any, when it is called '
            'with scope=None (which can lead to silent model degradation, so '
            'this error will remain until then.)' %
            (cell, cell_scope.name, scope_name, type(cell).__name__,
             type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable(_WEIGHTS_VARIABLE_NAME)
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            'Attempt to have a second RNNCell use the weights of a variable '
            "scope that already has weights: '%s'; and the cell was not "
            'constructed as %s(..., reuse=True).  '
            'To share the weights of an RNNCell, simply '
            'reuse it in your second calculation, or create a new one with '
            'the argument reuse=True.' % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope


class SeqAttentionCellWrapper(tf.nn.rnn_cell.RNNCell):
  """Basic attention cell wrapper.

  Implementation based on https://arxiv.org/abs/1409.0473.
  """

  def __init__(self,
               cell,
               attn_inputs,
               attn_size,
               attn_vec_size,
               output_size=None,
               input_size=None,
               state_is_tuple=True,
               attn_masks=None,
               merge_output_attn='linear',
               reuse=None):
    """Create a cell with attention.

    Args:
      cell: an RNNCell, an attention is added to it.
      attn_inputs: a Tensor.
      attn_size: integer, the size of an attention vector. Equal to
        cell.output_size by default.
      attn_vec_size: integer, the number of convolutional features calculated on
        attention state and a size of the hidden layer built from base cell
        state. Equal to attn_size by default.
      input_size: integer, the size of a hidden linear layer, built from inputs
        and attention. Derived from the input tensor by default.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all concatenated
        along the column axis.
      attn_mask: mask that should be applied to attention. If None, no masks
        will be applied.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    if not isinstance(cell, rnn.RNNCell):
      raise TypeError('The parameter cell is not RNNCell.')
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError(
          'Cell returns tuple of states, but the flag '
          'state_is_tuple is not set. State size is: %s' % str(cell.state_size))
    if not state_is_tuple:
      logging.warn(
          '%s: Using a concatenated state is slower and will soon be '
          'deprecated.  Use state_is_tuple=True.', self)

    self._state_is_tuple = state_is_tuple

    if not state_is_tuple:
      raise NotImplementedError

    self._cell = cell
    self._input_size = input_size
    self._output_size = output_size
    if output_size is None:
      self._output_size = cell.output_size
    self._attn_size = attn_size
    self._reuse = reuse
    self._attn_inputs = attn_inputs
    self._attn_vec_size = attn_vec_size
    self.attn_masks = attn_masks
    self.merge_output_attn = merge_output_attn

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype=tf.float32):
    zero_state = self._cell.zero_state(batch_size, dtype=dtype)
    return zero_state

  def __call__(self, inputs, state, scope=None):
    """Seq Attention wrapper."""
    with _checked_scope(
        self, scope or 'attention_cell_wrapper', reuse=self._reuse):
      inner_output, new_state = self._cell(inputs, state)
      new_attns = self._attention(inner_output, self._attn_inputs)
      if self.merge_output_attn == 'linear':
        with vs.variable_scope('attn_output_projection'):
          output = linear([inner_output, new_attns], self._output_size, True)
      elif self.merge_output_attn == 'concat':
        output = tf.concat([inner_output, new_attns], axis=-1)
      else:
        raise ValueError(
            'Unknown method to merge output and attention: {}'.format(
                self.merge_output_attn))
      return output, new_state

  def _attention(self, query, attn_inputs):
    with vs.variable_scope('attention'):
      attn_query = tf.layers.dense(
          inputs=query, units=self._attn_vec_size, use_bias=True)
      attn_keys = tf.layers.dense(
          inputs=attn_inputs, units=self._attn_vec_size, use_bias=True)
      attn_contents = tf.layers.dense(
          inputs=attn_inputs, units=self._attn_size, use_bias=True)

      v_attn = vs.get_variable('attn_v', [self._attn_vec_size])
      scores = attn_sum_bahdanau(v_attn, attn_keys, attn_query)

      if self.attn_masks is not None:
        score_masks = self.attn_masks
        scores = scores * score_masks + (1.0 - score_masks) * tf.float32.min

      attn_weights = nn_ops.softmax(scores)
      new_attns = math_ops.reduce_sum(
          tf.expand_dims(attn_weights, -1) * attn_contents, [1])
      return new_attns


def attn_sum_bahdanau(v_attn, keys, query):
  """Calculates a batch and timewise dot product with a variable."""
  return tf.reduce_sum(v_attn * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


def attn_sum_dot(keys, query):
  """Calculates a batch and timewise dot product."""
  return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


def linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError('`args` must be specified')
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError('linear is expecting 2D arguments: %s' % shapes)
    if shape[1].value is None:
      raise ValueError('linear expects shape[1] to be provided for shape %s, '
                       'but saw %s' % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)


class ScoreWrapper(tf.nn.rnn_cell.RNNCell):
  """Creates a cell which outputs a scalar score value at each time step."""

  def __init__(self, cell, activation='tanh'):
    self._cell = cell
    self._activation = ACTIVATION_DICT[activation]

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return (self._cell.output_size, 1)

  def __call__(self, inputs, state, *args, **kwargs):
    inner_output, next_state = self._cell(inputs, state, *args, **kwargs)
    with vs.variable_scope('score_fn', reuse=tf.AUTO_REUSE):
      # No need for explicitly stop_gradient since the gradients contribution
      # from scores is stopped now using stop_gradients in `tf.gradients`
      score_input = inner_output
      # Moving to a 2 layer architechture since the one layer didn't improve
      # the meta learning loss much
      score_val = tf.layers.dense(
          score_input,
          units=16,
          use_bias=True,
          activation=tf.nn.relu,
          kernel_initializer=tf.ones_initializer(),
          bias_initializer=tf.ones_initializer())
      score_val = tf.layers.dense(
          score_val,
          units=1,
          use_bias=True,
          activation=self._activation,
          kernel_initializer=tf.ones_initializer(),
          bias_initializer=tf.ones_initializer())
    output = (inner_output, score_val)
    return output, next_state


MemoryStateTuple = collections.namedtuple('MemoryStateTuple',
                                          ('memory', 'inner_state'))
MemoryInputTuple = collections.namedtuple(
    'MemoryInputTuple', ('read_ind', 'write_ind', 'valid_indices'))


class MemoryWrapper(tf.nn.rnn_cell.RNNCell):
  """Augment RNNCell with a memory that the RNN can write to and read from.

  Each time step, 3 things are happening:

  1) the RNNCell reads from one memory location (read_ind)
  as input to the inner RNN.

  2) It also writes the output of the inner RNN to one
  memory location (write_ind). 1 indicates no writing.

  3) It use the output of the inner RNN to compute the
  logits for the valid_indices, which will be used as input
  to compute a softmax distribution over them. Note that
  valid_indices always has the dimension
  max_n_valid_indices, use -1 to pad the dimensions the
  actual number of valid indices are less.

  """

  def __init__(self,
               cell,
               mem_size,
               embed_size,
               max_n_valid_indices,
               use_score_wrapper=False,
               **kwargs):
    """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      mem_size: size of the memory.
      embed_size: the size/dimension of the embedding in each memory location.
      max_n_valid_indices: maximum number of valid_indices.
      use_score_wrapper: Whether a score wrapper was used prior to passing the
      cell.
      **kwargs: Keyword arguments for score wrapper.

    """
    self._use_score_wrapper = use_score_wrapper
    if use_score_wrapper:
      self._cell = ScoreWrapper(cell, **kwargs)
    else:
      self._cell = cell
    self._mem_size = mem_size
    self._embed_size = embed_size
    self._max_n_valid_indices = max_n_valid_indices

  @property
  def state_size(self):
    # This will be used to create zero states.
    return MemoryStateTuple(
        tf.TensorShape([self._mem_size, self._embed_size]),
        self._cell.state_size)

  @property
  def output_size(self):
    # The output is the logits of the valid_dices.
    if self._use_score_wrapper:
      return (self._max_n_valid_indices, 1)
    else:
      return self._max_n_valid_indices

  def __call__(self, inputs, state, scope=None, debug=False):
    """Unroll the memory augmented cell for one step.

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of MemoryStateTuple containing tensors from the
        previous time step.
    """
    # B is batch size.

    # 1) Use read_ind to find memory location to read from
    # as input.

    # inputs.read_ind: (B, 1)
    # memory: (B, mem_size, embed_size)
    read_ind = tf.to_int32(inputs.read_ind)
    batch_size = tf.shape(read_ind)[0]
    if debug:
      print('batch size is', batch_size)
      print('read ind is', read_ind)
    mem_ind = tf.range(batch_size)

    # read_mem_ind: (B, 1)
    read_mem_ind = tf.expand_dims(mem_ind, axis=1)
    # read_ind: (B, 2)
    read_ind = tf.concat([read_mem_ind, read_ind], axis=1)
    if debug:
      print('processed read ind is', read_ind)

    # inner_inputs: (B, embed_size)
    inner_inputs = tf.gather_nd(state.memory, read_ind)
    if debug:
      print('inner_inputs is', inner_inputs)

    inner_state = state.inner_state

    # 2) Run the inner RNNCell.

    # inner_outputs: (B, embed_size)
    cell_outputs, new_inner_state = self._cell(
        inner_inputs, inner_state, scope=scope)
    if self._use_score_wrapper:
      inner_outputs, score_val = cell_outputs
    else:
      inner_outputs = cell_outputs

    if debug:
      print('inner_outputs is', inner_outputs)

    # 3) Compute logits for valid indices (using logit_masks
    # to mask out padded valid indices (-1)).

    # valid_indices: (B, max_n_valid_indices)
    valid_indices = tf.to_int32(inputs.valid_indices)
    if debug:
      print('valid_indices is', valid_indices)

    # Logit mask: (B, max_n_valid_indices)
    logit_masks = tf.greater_equal(inputs.valid_indices, 0)
    logit_masks = tf.cast(logit_masks, tf.float32)

    if debug:
      print('logit_masks is', logit_masks)

    # Normalize indices to be at least 0.
    valid_indices = tf.maximum(valid_indices, 0)

    # valid_indices: (B, max_n_valid_indices, 1)
    valid_indices = tf.expand_dims(valid_indices, -1)
    if debug:
      print('valid_indices is', valid_indices)
      print('mem_ind is', mem_ind)

    valid_mem_ind = tf.expand_dims(mem_ind, axis=1)

    # valid_mem_ind: (B, 1, 1)
    valid_mem_ind = tf.expand_dims(valid_mem_ind, axis=2)
    if debug:
      print('valid_mem_ind is', valid_mem_ind)

    # valid_mem_ind: (B, max_n_valid_indices, 1)
    valid_mem_ind = tf.tile(valid_mem_ind, [1, self._max_n_valid_indices, 1])

    # valid_indices: (B, max_n_valid_indices, 2)
    # Third dimension of valid_indices is [b_i, valid_index] so that it can
    # index into the right memory location.
    valid_indices = tf.concat([valid_mem_ind, valid_indices], axis=2)

    if debug:
      print('valid_indices is', valid_indices)

    # select all the valid slots.
    # valid_values: (B, max_n_valid_indices, embed_size)
    valid_values = tf.gather_nd(state.memory, valid_indices)
    if debug:
      print('valid_values is', valid_values)

    # expanded_inner_outputs: (B, 1, embed_size)
    expanded_inner_outputs = tf.expand_dims(inner_outputs, 1)
    if debug:
      print('expanded_inner_outputs is', expanded_inner_outputs)

    # valid_values: (B, embed_size, max_n_valid_indices)
    valid_values = tf.transpose(valid_values, [0, 2, 1])
    if debug:
      print('valid_values is', valid_values)

    # logits: (B, 1, max_n_valid_indices)
    logits = tf.matmul(expanded_inner_outputs, valid_values)

    if debug:
      print('logits is', logits)

    # logits: (B, max_n_valid_indices)
    logits = tf.squeeze(logits, axis=[1])
    if debug:
      print('logits is', logits)

    # masked_logits = (logits * logit_masks) - (1 - logit_masks) * 1e6
    masked_logits = logits * logit_masks + (1 - logit_masks) * tf.float32.min
    # masked_logits = tf.Print(masked_logits, [masked_logits],
    # message='masked_logits')

    outputs = masked_logits
    # 4) Write the output of the inner RNN to a memory
    # location (write_ind), using write_masks to mask out
    # padded write_ind (-1).

    # write_ind: (B, 1)
    write_ind = tf.cast(inputs.write_ind, tf.int32)
    if debug:
      print('write_ind is', write_ind)

    # write mask: (B, 1)
    write_masks = tf.greater_equal(inputs.write_ind, 0)
    if debug:
      print('write_masks greater_equal', write_masks)

    write_masks = tf.cast(write_masks, tf.float32)

    # write mask: (B, 1, 1)
    write_masks = tf.expand_dims(write_masks, [-1])

    if debug:
      print('write_masks is', write_masks)

    # Normalize write_ind to be above 0.
    # write_ind: (B, 1)
    write_ind = tf.maximum(write_ind, 0)

    # write_mem_ind: (B, 1)
    write_mem_ind = tf.expand_dims(mem_ind, axis=1)

    # write_ind: (B, 2)
    # Second dimension is [b_i, write_index]
    write_ind = tf.concat([write_mem_ind, write_ind], axis=1)
    if debug:
      print('write_ind is', write_ind)

    if debug:
      print('masked_logits is', masked_logits)
      print('memory is', state.memory)

    # write_mat: (B, mem_size, embed_size)
    write_mat = tf.scatter_nd(
        write_ind, inner_outputs, shape=tf.shape(state.memory))

    if debug:
      print('@' * 50)
      print('write_mat is', write_mat)
      print('write_mask is', write_masks)
      print('@' * 50)

    masked_write_mat = write_mat * write_masks
    new_memory = state.memory + masked_write_mat

    state = MemoryStateTuple(new_memory, new_inner_state)
    if debug:
      print('state is', state)

    if self._use_score_wrapper:
      outputs = (outputs, score_val)
    return (outputs, state)
