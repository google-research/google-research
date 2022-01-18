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

# Lint as: python2, python3
"""Light-weight library for constructing model layers for architecture searches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from tunas import custom_layers
from tunas import depthwise_initializers


def _get_tiny_float(dtype):
  if dtype == tf.bfloat16:
    # Numpy doesn't tell us what the smallest possible value of a bfloat16 is,
    # so we use a hard-coded value based on bfloat16.
    return tf.constant(2e-38, tf.bfloat16)
  else:
    return np.finfo(dtype.as_numpy_dtype).tiny


def _compute_explicit_padding(kernel_size, dilation_rate):
  """Compute the necessary padding based on kernel size and dilation rate."""
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size, kernel_size]
  if isinstance(dilation_rate, int):
    dilation_rate = [dilation_rate, dilation_rate]
  kernel_size_effective = [
      kernel_size[0] + (kernel_size[0] - 1) * (dilation_rate[0] - 1),
      kernel_size[1] + (kernel_size[1] - 1) * (dilation_rate[1] - 1)
  ]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  return [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]]


def with_data_dependencies(dependencies, output_tensors):
  """Add data dependencies that can't be optimized away by XLA.

  In certain cases, we may wish to run one TensorFlow op to run before another.
  In pure TensorFlow, we'd usually do this with control dependencies. But XLA
  can ignore control dependencies in certain cases. We instead create a fake
  data dependency, which XLA can't ignore.

  Based on the implementation of the recompute_grad decorator by rsepassi@.

  Args:
    dependencies: List of tensors which must be evaluated before any element
        of output_tensors can be evaluated.
    output_tensors: List of output tensors.

  Returns:
    A list of tensors with the same shapes and types as `output_tensors`.
  """
  # Compute a data dependency.
  data_dependencies = []
  for dependency in dependencies:
    # Extract the scalar value dependency[0, 0, ..., 0] and append it to
    # `data_dependencies`.
    begin = tf.zeros([dependency.shape.ndims], tf.int32)
    size = tf.ones([dependency.shape.ndims], tf.int32)
    data_dependency = tf.reshape(tf.slice(dependency, begin, size), [])
    data_dependencies.append(tf.cast(data_dependency, dependencies[0].dtype))

  sum_dependency = tf.stop_gradient(tf.add_n(data_dependencies))

  # Apply it to each tensor in `output_tensors`.
  results = []
  for tensor in output_tensors:
    tiny_float = _get_tiny_float(tensor.dtype)
    last_dep = tiny_float * tf.cast(sum_dependency, tensor.dtype)
    results.append(tensor + last_dep)

  return results


def _mask_regularizer(regularizer, mask):
  """Multiply a variable regularizer's value by a binary (0-1) mask."""
  def compute_masked_loss(value):
    loss = regularizer(value)
    if loss is None:
      return None

    if loss.shape.rank != 0:
      raise ValueError('loss must be scalar: {}'.format(loss))
    if mask.shape.rank != 0:
      raise ValueError('mask must be scalar: {}'.format(mask))
    return loss * tf.cast(mask, loss.dtype)

  return compute_masked_loss


def _maximum_regularizer(regularizer1, regularizer2):
  """Take the maximum of two variable regularizers."""
  def compute_loss(value):
    loss1 = regularizer1(value)
    loss2 = regularizer2(value)

    if loss1 is None:
      return loss2
    elif loss2 is None:
      return loss1
    else:
      return tf.maximum(loss1, loss2)

  return compute_loss


class Layer(six.with_metaclass(abc.ABCMeta, object)):
  """Abstract base class representing a neural network layer."""

  def __init__(self):
    self._trainable_variables = []
    self._trainable_tensors = collections.OrderedDict()
    self._var_regularizers = collections.OrderedDict()
    self._moving_average_variables = collections.OrderedDict()
    self._tracked_layers = []
    self._updates = []

  def _create_trainable_variable(self,
                                 name,
                                 shape=None,
                                 dtype=None,
                                 initializer=None,
                                 regularizer=None):
    """Protected helper function to create a new trainable variable."""
    if name in self._trainable_tensors:
      raise ValueError('Variable with name {!r} already exists'.format(name))

    variable = tf.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=True)

    if regularizer is not None:
      self._var_regularizers[variable] = regularizer

    self._trainable_variables.append(variable)

    # NOTE(b/123532966): We store both a list of trainable variables and a
    # dictionary containing Tensor snapshots of their values. In most cases,
    # we work with the snapshots rather than handling the variables directly.
    # This is needed to prevent TensorFlow from generating invalid graphs in
    # the body of Switch.apply().
    #
    # We can end up generating invalid TensorFlow graphs if we try to mix
    # conditional control flow (tf.cond), custom gradients, and TensorFlow
    # variables. The apply() function of a Switch layer makes use of tf.cond
    # and custom gradients. The call to variable.read_value() here basically
    # ensures that the function doesn't have to deal with variables directly.
    # At the beginning of each training step, we call read_value() on each
    # trainable variable to obtain a Tensor snapshot of its most recent value.
    # For the rest of the training step, work with these Tensor snapshots
    # instead of trying to manipulate the variables directly.
    self._trainable_tensors[name] = variable.read_value()

  def _get_trainable_tensor(self, name):
    """Protected helper function to look up a trainable variable by name."""
    return self._trainable_tensors[name]

  def _create_moving_average_variable(self,
                                      name,
                                      shape,
                                      initializer,
                                      dtype=tf.float32):
    self._moving_average_variables[name] = tf.get_variable(
        name=name,
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        collections=[
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES
        ],
        trainable=False)

  def _get_moving_average_variable(self, name):
    return self._moving_average_variables[name]

  def _update_moving_average_variable(self, name, value, momentum):
    """Protected helper method to update a moving average variable.

    WARNING: This method comes with a few caveats:
    1. It should not be used inside Switch layers or other layers that rely on
       conditional control flow.
    2. If using with rematerialization, the same variable may receive multiple
       updates.

    Args:
      name: String, name of the variable to create.
      value: Tensor, value for the moving average update.
      momentum: Float between 0 and 1, the momentum to use for the moving
          average update.
    """
    var = self._moving_average_variables[name]
    update_op = tf.assign_sub(var, (var - value) * (1 - momentum))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
    self._updates.append(update_op)

  def _track_layer(self, layer):
    """Protected helper function to mark `layer` as a child of the callee."""
    self._tracked_layers.append(layer)

  def trainable_tensors(self):
    """Return a list of tensors corresponding to trainable variables."""
    # This logic was originally added to work around a TF bug which caused some
    # bad interactions between tf.cond() statements, custom gradients, and
    # TensorFlow variables. However, the root cause of the problem involved
    # implementation details of COND_V1. We're not sure if it's still necessary,
    # since we've updated the code to use COND_V2 instead.

    # Ensure that no tensor is added more than once, and that tensors are added
    # in a predictable order. Ideally, we'd use an OrderedSet to keep track of
    # all the tensors, but Python doesn't have one, so we use an OrderedDict
    # instead.
    result_dict = collections.OrderedDict()

    for tensor in self._trainable_tensors.values():
      result_dict[tensor] = None

    for layer in self._tracked_layers:
      for tensor in layer.trainable_tensors():
        result_dict[tensor] = None

    return list(result_dict.keys())

  def trainable_variables(self):
    """Returns a list of trainable variables for the layer and its children.

    WARNING: This function is intended for use outside of model construction,
    such as when calling

        Optimizer.minimize(var_list=model.trainable_variables())

    You should use the trainable_tensors() method instead during model
    construction, especially within the apply() method.

    Returns:
      A list of tf.Variable objects.
    """
    # Use an OrderedDict to deduplicate the list of variables while ensuring
    # that they're returned in a predictable order.
    result_dict = collections.OrderedDict()

    for variable in self._trainable_variables:
      result_dict[variable] = None

    for layer in self._tracked_layers:
      for variable in layer.trainable_variables():
        result_dict[variable] = None

    return list(result_dict.keys())

  def _get_all_variable_regularizers(self):
    """Returns a list of (variable, regularizer) pairs."""
    # Deduplicate the list of regularizers so that no variable is regularized
    # more than once, even if it's used in multiple layers.
    result = collections.OrderedDict()
    result.update(self._var_regularizers)
    for layer in self._tracked_layers:
      for var, regularizer in layer._get_all_variable_regularizers().items():  # pylint:disable=protected-access
        if var in result:
          # The same variable can be used by more than one child of the current
          # layer. The regularizer might be masked out (i.e., multiplied by
          # zero) in some but not all of the children. We mask out a variable's
          # regularizer only when it is masked out by *every* child. See the
          # implementation of the Switch class for details.
          result[var] = _maximum_regularizer(result[var], regularizer)
        else:
          result[var] = regularizer

    return result

  def regularization_loss(self):
    """Compute the total regularization loss for a layer and its children.

    Returns:
      A scalar float Tensor.
    """
    losses = []
    for var, regularizer in self._get_all_variable_regularizers().items():
      current_loss = regularizer(var)
      if current_loss is not None:
        losses.append(current_loss)

    if losses:
      return tf.add_n(losses)
    else:
      return tf.zeros(shape=(), dtype=tf.float32)

  def updates(self):
    """Get a list of update ops to apply for the current training step.

    Returns:
      A list of TensorFlow Operations.
    """
    result = list(self._updates)
    for layer in self._tracked_layers:
      result.extend(layer.updates())
    return result

  @abc.abstractmethod
  def build(self, input_shape):
    """Create any layer-specific variables and compute the output shape.

    Args:
      input_shape: tf.Shape, shape of the input tensor for this layer.

    Returns:
      tf.Shape of the output tensor returned by this layer.
    """
    pass

  @abc.abstractmethod
  def apply(self, inputs, training):
    """Apply the current layer to the specified input tensor.

    Args:
      inputs: Tensor of input values.
      training: Boolean. True during model training, false during inference
          and evaluation.

    Returns:
      Tensor of output values.
    """
    pass


class Identity(Layer):
  """Network layer corresponding to an identity function."""

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.identity(inputs)


class Zeros(Layer):
  """Network layer that returns an all-zeros tensor.

  If output shape is not specified, return an all-zeros tensor with the same
  shape as the input tensor. Note the batch dimension of the output will be
  adjusted according to the input.
  """

  def __init__(self, output_shape=None):
    super(Zeros, self).__init__()
    self._output_shape = output_shape

  def build(self, input_shape):
    if self._output_shape:
      # Adjust batch dimension of the output shape based on the input shape.
      batch_dim = input_shape[0]
      remaining_dims = [None] * (self._output_shape.rank - 1)
      return self._output_shape.merge_with([batch_dim] + remaining_dims)
    return input_shape

  def apply(self, inputs, training):
    del training
    if self._output_shape:
      # Batch dimension is allowed to vary based on inputs. Useful for cases
      # where self.apply is called twice with different batch sizes.
      batch_dim = tf.shape(inputs)[0]
      output_shape = tf.stack([batch_dim] + self._output_shape.as_list()[1:])
      self._output_shape[:1].assert_is_compatible_with(inputs.shape[:1])
      return tf.zeros(shape=output_shape, dtype=inputs.dtype)
    return tf.zeros_like(inputs)


class ReLU(Layer):
  """Network layer corresponding to a ReLU activation function."""

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.nn.relu(inputs)


class ReLU6(Layer):
  """Network layer corresponding to a ReLU6 activation function."""

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.nn.relu6(inputs)


class Sigmoid(Layer):
  """Network layer corresponding to a sigmoid activation function."""

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.nn.sigmoid(inputs)


class Swish(Layer):
  """Network layer corresponding to a SiLU/Swish activation function.

  References:
  Hendrycks and Gimpel. "Gaussian Error Linear Units (GELUs)."
  https://arxiv.org/pdf/1606.08415.pdf

  Elfwing, Uchibe, and Doya.
  "Sigmoid-weighted linear units for neural network function approximation
  in reinforcement learning." Neural Networks, 107:3-11, 2018

  Ramachandran, Zoph, and Le. "Searching for Activation Functions."
  https://arxiv.org/abs/1710.05941

  """

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.nn.swish(inputs)


class Swish6(Layer):
  """Network layer corresponding to a Swish6/H-Swish activation.

  Swish6 is a modified variation of the SiLU/Swish activation function
  proposed in the MobileNet V3 paper.

  Reference: Section 5.2 of Howard et al. "Searching for MobileNet V3."
  https://arxiv.org/pdf/1905.02244.pdf
  """

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    with tf.name_scope('Swish6'):
      return inputs * tf.nn.relu6(inputs + np.float32(3)) * np.float32(1. / 6.)


class ELU(Layer):
  """Network layer corresponding to an ELU activation function.

  Reference:
  Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  https://arxiv.org/pdf/1511.07289.pdf
  """

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    del training
    return tf.nn.elu(inputs)


class SpaceToDepth(Layer):
  """Network layer corresponding to a space to depth function."""

  def __init__(self, block_size):
    super(SpaceToDepth, self).__init__()
    self._block_size = block_size
    self._built = False

  def build(self, input_shape):
    assert len(input_shape) == 4, input_shape

    height_dim = tf.compat.dimension_value(input_shape[1])
    if height_dim is not None:
      if height_dim % self._block_size != 0:
        raise ValueError('Image height {} must be a multiple of {}'.format(
            height_dim, self._block_size))
      height_dim //= self._block_size

    width_dim = tf.compat.dimension_value(input_shape[2])
    if width_dim is not None:
      if width_dim % self._block_size != 0:
        raise ValueError('Image width {} must be a multiple of {}'.format(
            width_dim, self._block_size))
      width_dim //= self._block_size

    channel_dim = tf.compat.dimension_value(input_shape[3])
    if channel_dim is not None:
      channel_dim *= pow(self._block_size, 2)

    output_shape = [input_shape[0], height_dim, width_dim, channel_dim]
    self._built = True

    return tf.TensorShape(output_shape)

  def apply(self, inputs, training):
    del training
    assert self._built

    return tf.nn.space_to_depth(inputs, self._block_size)


class DepthPadding(Layer):
  """Network layer corresponding to a depth padding function."""

  def __init__(self, filters):
    super(DepthPadding, self).__init__()
    self._filters = filters
    self._built = False

  def build(self, input_shape):
    assert len(input_shape) == 4, input_shape
    if int(input_shape[3]) > self._filters:
      raise ValueError('Output filters is smaller than input filters.')

    output_shape = input_shape.as_list()
    output_shape[3] = self._filters
    self._built = True

    return tf.TensorShape(output_shape)

  def apply(self, inputs, training):
    del training
    assert len(inputs.shape) == 4, inputs
    assert self._built

    input_filters = int(inputs.shape[3])
    if input_filters > self._filters:
      raise ValueError('Output filters is smaller than input filters.')
    elif input_filters == self._filters:
      return inputs
    else:  # input_filters < self._filters
      filters_padding = self._filters - tf.shape(inputs)[3]
      return tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, filters_padding]])


def _get_pool_output_shape(input_shape, strides):
  """Get output shape for pooling ops with the 'SAME' padding scheme."""
  filter_dim = int(input_shape[3])
  return get_conv_output_shape(input_shape, strides, filter_dim)


class SpatialMasking(Layer):
  """Network layer that masks the input tensor along spatial dimensions."""

  def __init__(self, mask, name=None):
    super(SpatialMasking, self).__init__()
    assert len(mask.shape) == 2, mask
    self._mask = mask
    self._name = name
    self._built = False

  def build(self, input_shape):
    assert len(input_shape) == 4, input_shape
    with tf.name_scope(self._name, 'SpatialMasking') as scope:
      self._scope = scope
      self._spatial_mask = tf.expand_dims(tf.expand_dims(self._mask, 0), -1)
    self._built = True
    return input_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.name_scope(self._scope):
      return inputs * tf.cast(
          tf.stop_gradient(self._spatial_mask), inputs.dtype)


class MaxPool(Layer):
  """Network layer corresponding to a max pooling function."""

  def __init__(self, kernel_size, strides, use_explicit_padding=False):
    super(MaxPool, self).__init__()
    self._kernel_size = kernel_size
    self._strides = (strides, strides) if isinstance(strides, int) else strides
    self._built = False
    self._use_explicit_padding = use_explicit_padding

  def build(self, input_shape):
    self._built = True
    assert len(input_shape) == 4, input_shape
    return _get_pool_output_shape(input_shape, self._strides)

  def apply(self, inputs, training):
    del training
    assert self._built
    padding = 'SAME'
    if self._use_explicit_padding:
      padding = 'VALID'
      inputs = tf.pad(
          tensor=inputs,
          paddings=_compute_explicit_padding(self._kernel_size, (1, 1)))
    return tf.nn.max_pool(
        inputs, self._kernel_size, self._strides, padding=padding)


class AveragePool(Layer):
  """Network layer corresponding to an average pooling function."""

  def __init__(self, kernel_size, strides):
    super(AveragePool, self).__init__()
    self._kernel_size = kernel_size
    self._strides = (strides, strides) if isinstance(strides, int) else strides
    self._built = False

  def build(self, input_shape):
    assert len(input_shape) == 4, input_shape
    self._built = True
    return _get_pool_output_shape(input_shape, self._strides)

  def apply(self, inputs, training):
    del training
    assert self._built
    return tf.nn.avg_pool(
        inputs, self._kernel_size, self._strides, padding='SAME')


class GlobalAveragePool(Layer):
  """Network layer corresponding to a global average pooling function."""

  def __init__(self, keepdims=False):
    super(GlobalAveragePool, self).__init__()
    self._keepdims = keepdims

  def build(self, input_shape):
    assert len(input_shape) == 4, input_shape
    if self._keepdims:
      return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])
    else:
      return tf.TensorShape([input_shape[0], input_shape[3]])

  def apply(self, inputs, training):
    del training

    height = tf.compat.dimension_value(inputs.shape[1])
    width = tf.compat.dimension_value(inputs.shape[2])
    if not height or not width:
      # Use tf.reduce_mean() instead of tf.nn.avg_pool() because average-pooling
      # ops with unknown kernel sizes are not supported by TensorFlow.
      return tf.reduce_mean(inputs, axis=[1, 2], keepdims=self._keepdims)

    if height == 1 and width == 1:
      if self._keepdims:
        return tf.identity(inputs)
      else:
        return tf.squeeze(inputs, axis=[1, 2])

    # We try to use tf.nn.avg_pool() rather than tf.reduce_mean() wherever
    # possible, since tf.reduce_mean() is incompatible with certain mobile GPUs.
    result = tf.nn.avg_pool(
        inputs, [height, width], strides=1, padding='VALID')
    if not self._keepdims:
      result = tf.squeeze(result, axis=[1, 2])

    return result


class Dropout(Layer):
  """Network layer that implements dropout.

  Each element of the input is kept or dropped independently.
  """

  def __init__(self, rate=0.5):
    """Class initializer.

    Args:
      rate: Float or scalar float tensor between 0 and 1. The fraction of
          input units to drop.
    """
    super(Dropout, self).__init__()
    self._rate = rate

  def build(self, input_shape):
    return input_shape

  def apply(self, inputs, training):
    if training:
      return tf.nn.dropout(inputs, rate=self._rate)
    else:
      return tf.identity(inputs)


class MultiplyByConstant(Layer):
  """Multiply the input by a non-trainable mask. Active only during training."""

  def __init__(self, scale, name=None):
    super(MultiplyByConstant, self).__init__()
    self._scale = scale
    self._name = name
    self._scope = None
    self._built = False

  def build(self, input_shape):
    with tf.name_scope(self._name, 'MultiplyByConstant') as scope:
      self._scope = scope
      self._scale = tf.convert_to_tensor(self._scale)

    self._built = True
    return merge_shapes_with_broadcast(input_shape, self._scale.shape)

  def apply(self, inputs, training):
    del training  # Unused
    assert self._built
    with tf.name_scope(self._scope):
      scale = tf.stop_gradient(tf.cast(self._scale, inputs.dtype))
      return scale * inputs


def _cond_v2(pred, true_fn, false_fn):
  """Hack to access tf.cond_v2(), which isn't part of TF's public interface.

  NOTE: This function will have the side effect of enabling tf.cond_v2()
  and while_v2 within true_fn and false_fn.

  Args:
    pred: Logical predicate, bool or scalar tf.bool Tensor.
    true_fn: Function to evaluate if pred is true.
    false_fn: Function to evaluate if pred is false.

  Returns:
    The output of true_fn or false_fn.
  """
  is_cond_v2_enabled = tf.control_flow_v2_enabled()
  if not is_cond_v2_enabled:
    tf.enable_control_flow_v2()
  result = tf.cond(pred, true_fn, false_fn)
  if not is_cond_v2_enabled:
    tf.disable_control_flow_v2()
  return result


def _make_cond(condition, if_true_fn, if_true_inputs, if_false):
  """Add a tf.cond() statement to the model."""
  # NOTE: The code below is equivalent to
  #
  #     return tf.cond(
  #         condition,
  #         lambda: if_true_fn(*if_true_inputs),
  #         lambda: if_false)
  #
  # However, in an early version of the code, we were able to improve model
  # training throughput by 20-30% in test runs by flipping the `if` and `else`
  # branches. This should be fixed in the latest version of TensorFlow, but we
  # haven't tested it yet.
  #
  # We use cond_v2() instead of tf.cond(). Although the two operations are (in
  # theory) equivalent, cond_v2() typically works better on TPUs.
  return _cond_v2(
      tf.logical_not(condition),
      lambda: if_false,
      lambda: if_true_fn(*if_true_inputs))


class Sequential(Layer):
  """Sequence of layers, where the output of one is the input to the next."""

  def __init__(self, layers, aux_outputs=None, name=None):
    super(Sequential, self).__init__()
    self._layers = layers
    self._name = name
    self._built = False

    for layer in self._layers:
      self._track_layer(layer)

    if aux_outputs is not None:
      self._aux_output_indices = []
      for layer in aux_outputs:
        try:
          self._aux_output_indices.append(layers.index(layer))
        except ValueError:
          # Raise a new ValueError with a more informative error message
          raise ValueError(
              'element of aux_outputs does not appear in layers: {}'.format(
                  layer))

    else:
      self._aux_output_indices = None

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'Sequential') as scope:
      self._scope = scope

      shape = input_shape
      for layer in self._layers:
        shape = layer.build(shape)

      self._built = True
      return shape

  def apply(self, inputs, training):
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      value = inputs

      intermediate_values = []
      for layer in self._layers:
        value = layer.apply(value, training)
        intermediate_values.append(value)

      if self._aux_output_indices is not None:
        aux_output_values = [
            intermediate_values[i]
            for i in self._aux_output_indices
        ]
        return value, aux_output_values
      else:
        return value


def merge_shapes_with_broadcast(shape1, shape2):
  """Compute the output shape for a binary op that supports broadcasting."""
  shape1 = tf.TensorShape(shape1)
  shape2 = tf.TensorShape(shape2)

  # Handle the case where shape1 or shape2 contains no information.
  if not shape1:
    return shape2

  if not shape2:
    return shape1

  # Handle the case where one of the inputs is a scalar.
  if shape1.rank == 0:
    return shape2

  if shape2.rank == 0:
    return shape1

  # Make sure both shapes have the same rank.
  if shape1.rank != shape2.rank:
    raise ValueError('Tensor shapes must have the same rank: {} and {}'.format(
        shape1, shape2))

  # Make sure each dimension is either equal or supports broadcasting.
  output_dims = []
  for dim1, dim2 in zip(shape1.as_list(), shape2.as_list()):
    if dim1 is None:
      output_dims.append(dim2)
    elif dim2 is None:
      output_dims.append(dim1)
    elif dim1 == 1:
      output_dims.append(dim2)
    elif dim2 == 1:
      output_dims.append(dim1)
    elif dim1 == dim2:
      output_dims.append(dim1)
    else:  # dim1 != dim2
      raise ValueError('Tensor shapes are not compatible: {} vs {}'.format(
          shape1, shape2))

  return tf.TensorShape(output_dims)


class _ParallelAggregation(Layer):
  """Apply several layers to the same input, and combine their results."""

  def __init__(self, branches, name=None):
    super(_ParallelAggregation, self).__init__()
    self._branches = branches
    self._name = name
    self._built = False

    for layer in branches:
      self._track_layer(layer)

  def build(self, input_shape):
    with tf.variable_scope(self._name, self.__class__.__name__) as scope:
      self._scope = scope

      output_shape = tf.TensorShape(None)
      for branch in self._branches:
        branch_shape = branch.build(input_shape)
        output_shape = merge_shapes_with_broadcast(output_shape, branch_shape)

      self._built = True
      return output_shape

  def apply(self, inputs, training):
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      return self._reduce([
          branch.apply(inputs, training) for branch in self._branches
      ])

  @abc.abstractmethod
  def _reduce(self, tensors):
    pass


class ParallelSum(_ParallelAggregation):
  """Apply several layers to the same input, and sum their results."""

  def _reduce(self, tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
      result = result + tensor
    return result


class ParallelProduct(_ParallelAggregation):
  """Apply several layers to the same input, and multiply their results."""

  def _reduce(self, tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
      result = result * tensor
    return result


class Switch(Layer):
  """Take a weighted combination of N possible options.

  Options whose weights are zero will be optimized away.
  """

  def __init__(self, mask, options, name=None):
    """Class initializer.

    Args:
      mask: A float Tensor of shape [len(options)].
      options: List of Layer instances.
      name: Optional string, name for the current layer.
    """
    super(Switch, self).__init__()
    self._mask = mask
    self._options = options
    self._name = name
    self._built = False

    for layer in options:
      self._track_layer(layer)

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'Switch') as scope:
      self._scope = scope

      output_shape = tf.TensorShape(None)
      for branch in self._options:
        branch_shape = branch.build(input_shape)
        output_shape = output_shape.merge_with(branch_shape)

      self._output_shape = output_shape
      self._built = True
      return output_shape

  def apply(self, inputs, training):
    assert self._built

    def apply_branch_fn(branch, weight, bias):
      # The first element of `all_inputs` is always equal to `inputs`.
      def fn(*all_inputs):
        result = branch.apply(all_inputs[0], training)
        return result * tf.cast(weight, result.dtype) + bias
      return fn

    @tf.custom_gradient
    def impl(*all_inputs):
      """Returns the output tensor and a function that computes its gradient."""
      # Select which branch to take based on a discrete (integer-valued) tensor.
      mask = tf.stop_gradient(self._mask)
      mask.shape.assert_is_compatible_with([len(self._options)])

      # During the apply pass, we evaluate one branch and throw away all the
      # intermediate outputs.
      with tf.variable_scope(self._scope, reuse=True):
        # Compute the output shape. The dimensions will generally be the same as
        # those returned by self.build(), but the batch size can be different.
        batch_size = tf.shape(all_inputs[0])[0]
        output_shape = tf.stack([batch_size] + self._output_shape.as_list()[1:])

        # Forward pass
        output = tf.zeros(output_shape, dtype=all_inputs[0].dtype)
        for i, branch in enumerate(self._options):
          output = _make_cond(
              tf.not_equal(mask[i], 0),
              # If mask[i] != 0 then apply the current branch to `all_inputs`,
              # and add the result to `output`.
              apply_branch_fn(branch, mask[i], output),
              all_inputs,
              # Otherwise, leave the output unchanged
              output)

      def grad_fn(*output_grads):
        """Compute gradients for the switch statement."""
        def update_grads_fn(grads, branch, weight):
          """Returns a function that adds gradients for `branch` to `grads`."""
          def fn(*all_inputs):
            """Rematerializes `branch` and adds its gradients to `grads`."""
            rematerialized_output = branch.apply(all_inputs[0], training)
            rematerialized_output *= tf.cast(
                weight, rematerialized_output.dtype)

            # Replace any `None` gradients with zeros. The gradients computed
            # here are returned in the `else` clause of a tf.cond() statement
            # later in the code, and trying to return a `None` value inside a
            # tf.cond statement would trigger an exception.
            grad_updates = tf.gradients(
                rematerialized_output, all_inputs, output_grads)

            sum_grads = []
            for i in range(len(grad_updates)):
              if grad_updates[i] is not None:
                sum_grads.append(grads[i] + grad_updates[i])
              else:
                sum_grads.append(grads[i])

            return sum_grads
          return fn

        with tf.variable_scope(self._scope, reuse=True):
          grads = [tf.zeros_like(x) for x in all_inputs]
          for i, branch in enumerate(self._options):
            grads = _make_cond(
                tf.not_equal(mask[i], 0),
                # If mask[i] != 0 then take gradients w.r.t. current branch
                update_grads_fn(grads, branch, mask[i]),
                all_inputs,
                # Otherwise, leave the gradients unchanged
                grads)

        return grads

      return output, grad_fn

    # Main logic.
    all_inputs = [inputs] + self.trainable_tensors()
    return impl(*all_inputs)

  def _get_all_variable_regularizers(self):
    # Override the parent class's implementation to only regularizer variables
    # associated with the branch that's used for the current batch of training
    # examples.
    result = collections.OrderedDict()
    for i, option in enumerate(self._options):
      for var, regularizer in option._get_all_variable_regularizers().items():  # pylint:disable=protected-access
        # If a variable is not used by the selected `option` then we mask out
        # its regularizer (i.e., we multiply the regularizer by 0).
        var_regularizer = _mask_regularizer(regularizer, self._mask[i])

        if var in result:
          # The same variable can be used by more than one of `self._options`.
          # For example, if both `self._options[0]` and `self._options[1]` make
          # use of the variable `var` with regularizer `reg` then the masked
          # regularizers for the two options will be
          #     `reg(var) * tf.equal(self._selection, 0)` and
          #     `reg(var) * tf.equal(self._selection, 1)`
          # respectively. By taking the maximum, we'll regularize the variable
          # if `0 <= self._selection <= 1` but not if `self._selection > 1`.
          result[var] = _maximum_regularizer(result[var], var_regularizer)
        else:
          result[var] = var_regularizer
    return result


def _is_array_one_hot(array):
  """Returns true if `array` is one-hot."""
  assert len(array.shape) == 1, array
  argmax = np.argmax(array)

  # One element of the array should have a value of 1
  if array[argmax] != 1:
    return False

  # Each remaining element should have a value of 0
  if np.count_nonzero(array) != 1:
    return False

  return True


def maybe_switch_v2(mask, options, name=None):
  """Apply a Switch layer, optimizing it away if possible.

  Args:
    mask: A one-hot Tensor.
    options: A list of Layer objects.
    name: Optional string.

  Returns:
    A Layer object.
  """
  if mask is None:
    if len(options) != 1:
      raise ValueError(
          'Mask cannot be None unless len(options) == 1, but options = {}'
          .format(options))
    return Sequential([options[0]], name=name)
  elif mask.shape == tf.TensorShape([1]):
    # We avoid using a Switch layer when mask.shape == [1]. This allows us to
    # use masks to train stand-alone models with path dropout. We can't use
    # Switch layers in this case because we need to maintain and update moving
    # average accumulators.
    if len(options) != 1:
      raise ValueError(
          'Mask has shape [1] but options has length {:d}: {}'
          .format(len(options), options))

    return Sequential([options[0], MultiplyByConstant(mask[0])], name=name)
  else:
    assert mask.shape.rank == 1, mask
    return Switch(mask, options, name)


def get_conv_output_shape(input_shape, strides, output_filters):
  """Get output shape for conv/pooling ops with the 'SAME' padding scheme."""
  if isinstance(strides, int):
    strides = (strides, strides)

  height_dim = tf.compat.dimension_value(input_shape[1])
  if height_dim is not None:
    height_dim = int(math.ceil(height_dim / strides[0]))

  width_dim = tf.compat.dimension_value(input_shape[2])
  if width_dim is not None:
    width_dim = int(math.ceil(width_dim / strides[1]))

  output_shape = tf.TensorShape(
      [input_shape[0], height_dim, width_dim, output_filters])
  return output_shape


class Conv2D(Layer):
  """2D convolution network layer."""

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               dilation_rates=(1, 1),
               kernel_initializer=tf.initializers.he_normal(),
               kernel_regularizer=None,
               bias_initializer=tf.initializers.zeros(),
               bias_regularizer=None,
               use_bias=False,
               use_explicit_padding=False,
               name=None):
    super(Conv2D, self).__init__()
    self._filters = filters

    if isinstance(kernel_size, int):
      self._kernel_size = (kernel_size, kernel_size)
    else:
      self._kernel_size = kernel_size

    if isinstance(strides, int):
      self._strides = (strides, strides)
    else:
      self._strides = strides

    if isinstance(dilation_rates, int):
      self._dilation_rates = (dilation_rates, dilation_rates)
    else:
      self._dilation_rates = dilation_rates

    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bias = use_bias
    self._name = name
    self._built = False
    self._use_explicit_padding = use_explicit_padding

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'Conv2D') as scope:
      self._scope = scope

      input_filters = int(input_shape[-1])
      kernel_shape = tuple(self._kernel_size) + (input_filters, self._filters)
      if not self._built:
        self._create_trainable_variable(
            name='kernel',
            shape=kernel_shape,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer)
        if self._use_bias:
          self._create_trainable_variable(
              name='bias',
              shape=(self._filters,),
              initializer=self._bias_initializer,
              regularizer=self._bias_regularizer)

      output_shape = get_conv_output_shape(
          input_shape, self._strides, output_filters=self._filters)
      self._built = True
      return output_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      kernel = self._get_trainable_tensor('kernel')
      kernel = tf.cast(kernel, inputs.dtype)
      padding = 'SAME'
      if self._use_explicit_padding:
        padding = 'VALID'
        inputs = tf.pad(
            tensor=inputs,
            paddings=_compute_explicit_padding(self._kernel_size,
                                               self._dilation_rates))
      result = tf.nn.conv2d(
          input=inputs,
          filters=kernel,
          strides=[1] + list(self._strides) + [1],
          padding=padding,
          dilations=[1] + list(self._dilation_rates) + [1])
      if self._use_bias:
        bias = self._get_trainable_tensor('bias')
        bias = tf.cast(bias, result.dtype)
        return tf.nn.bias_add(result, bias)
      else:
        return result


class DepthwiseConv2D(Layer):
  """2D depthwise convolution layer."""

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               dilation_rates=(1, 1),
               depthwise_initializer=
               depthwise_initializers.depthwise_he_normal(),
               depthwise_regularizer=None,
               use_explicit_padding=False,
               name=None):
    super(DepthwiseConv2D, self).__init__()

    if isinstance(kernel_size, int):
      self._kernel_size = (kernel_size, kernel_size)
    else:
      self._kernel_size = kernel_size

    if isinstance(strides, int):
      self._strides = (strides, strides)
    else:
      self._strides = strides

    if isinstance(dilation_rates, int):
      self._dilation_rates = (dilation_rates, dilation_rates)
    else:
      self._dilation_rates = dilation_rates

    # tf.nn.depthwise_conv2d restricts that if dilation rates are
    # greater than 1, then all strides must be equal to 1.
    if self._dilation_rates != (1, 1) and self._strides != (1, 1):
      raise ValueError(
          'Non-unit dilations {0} can only be used with unit strides {1}. '
          .format(self._dilation_rates, self._strides))

    self._depthwise_initializer = depthwise_initializer
    self._depthwise_regularizer = depthwise_regularizer
    self._name = name
    self._built = False
    self._use_explicit_padding = use_explicit_padding

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'DepthwiseConv2D') as scope:
      self._scope = scope

      input_filters = int(input_shape[-1])
      kernel_shape = tuple(self._kernel_size) + (input_filters, 1)
      if not self._built:
        self._create_trainable_variable(
            name='kernel',
            shape=kernel_shape,
            initializer=self._depthwise_initializer,
            regularizer=self._depthwise_regularizer)

      output_shape = get_conv_output_shape(
          input_shape, self._strides, output_filters=int(input_shape[3]))
      self._built = True
      return output_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      kernel = self._get_trainable_tensor('kernel')
      kernel = tf.cast(kernel, inputs.dtype)
      padding = 'SAME'
      if self._use_explicit_padding:
        padding = 'VALID'
        inputs = tf.pad(
            tensor=inputs,
            paddings=_compute_explicit_padding(self._kernel_size,
                                               self._dilation_rates))
      result = tf.nn.depthwise_conv2d(
          input=inputs,
          filter=kernel,
          strides=[1] + list(self._strides) + [1],
          padding=padding,
          dilations=list(self._dilation_rates))
      return result


class BatchNorm(Layer):
  """Abstract base class representing a batch normalization layer."""

  def __init__(self,
               epsilon=1e-12,
               center=True,
               scale=True,
               beta_initializer=tf.initializers.zeros(),
               gamma_initializer=tf.initializers.ones(),
               momentum=0.99,
               stateful=True,
               name=None):
    super(BatchNorm, self).__init__()
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._beta_initializer = beta_initializer
    self._gamma_initializer = gamma_initializer
    self._momentum = momentum
    self._stateful = stateful
    self._name = name
    self._built = False

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'BatchNorm') as scope:
      self._scope = scope

      if not self._built:
        if self._center:
          self._create_trainable_variable(
              name='beta',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=self._beta_initializer)

        if self._scale:
          self._create_trainable_variable(
              name='gamma',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=self._gamma_initializer)

        if self._stateful:
          self._create_moving_average_variable(
              name='moving_mean',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=tf.initializers.zeros())
          self._create_moving_average_variable(
              name='moving_variance',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=tf.initializers.ones())

      self._built = True
      return input_shape

  def apply(self, inputs, training):
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      scale = self._get_trainable_tensor('gamma') if self._scale else None
      offset = self._get_trainable_tensor('beta') if self._center else None

      if self._stateful and not training:
        moving_mean = self._get_moving_average_variable('moving_mean')
        moving_variance = self._get_moving_average_variable('moving_variance')
      else:
        moving_mean = None
        moving_variance = None

      result, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          scale=scale,
          offset=offset,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=self._epsilon,
          is_training=training or not self._stateful)

      if self._stateful and training:
        self._update_moving_average_variable(
            'moving_mean', mean, self._momentum)
        self._update_moving_average_variable(
            'moving_variance', variance, self._momentum)

      return result


def _regularizer_over_masked_variable(regularizer, mask, transpose=False):
  """return a regularizer over the masked tensor."""
  if regularizer is None:
    return None

  def compute_regularizer(value):
    if transpose:
      value = tf.transpose(value, perm=[0, 1, 3, 2])
    return regularizer(value * mask)

  return compute_regularizer


class MaskedDepthwiseConv2D(Layer):
  """2D masked depthwise convolution layer."""

  def __init__(self,
               kernel_size,
               mask,
               strides=(1, 1),
               depthwise_initializer=
               depthwise_initializers.depthwise_he_normal(),
               depthwise_regularizer=None,
               transpose_depthwise_kernels=False,
               use_explicit_padding=False,
               name=None):
    super(MaskedDepthwiseConv2D, self).__init__()

    if isinstance(kernel_size, int):
      self._kernel_size = (kernel_size, kernel_size)
    else:
      self._kernel_size = kernel_size

    if isinstance(strides, int):
      self._strides = (strides, strides)
    else:
      self._strides = strides

    self._depthwise_initializer = depthwise_initializer
    self._depthwise_regularizer = depthwise_regularizer
    self._name = name
    self._built = False
    self._transpose_depthwise_kernels = transpose_depthwise_kernels
    self._use_explicit_padding = use_explicit_padding

    # NOTE(gbender, hanxiaol): Be careful that TF might try to back-propagate
    # through the masks and cause issues with the 'Switch' statements (which
    # use custom gradients). A potential solution would be integrating the logic
    # of masks also into custom gradients.
    self._mask = tf.reshape(mask, [1, 1, -1, 1])

  def build(self, input_shape):

    with tf.variable_scope(self._name, 'MaskedDepthwiseConv2D') as scope:
      self._scope = scope

      max_filters = int(self._mask.shape[2])
      if int(input_shape[-1]) != max_filters:
        raise ValueError(
            'padded input filter size ({:d}) must match the max possible '
            'effective filter size ({:d}).'
            .format(int(input_shape[-1]), max_filters))

      if not self._built:
        mask = tf.stop_gradient(self._mask)
        masked_depthwise_regularizer = _regularizer_over_masked_variable(
            self._depthwise_regularizer,
            mask,
            transpose=self._transpose_depthwise_kernels)

        if self._transpose_depthwise_kernels:
          kernel_shape = tuple(self._kernel_size) + (1, max_filters)
          depthwise_initializer = custom_layers.TransposedInitializer(
              self._depthwise_initializer)
        else:
          kernel_shape = tuple(self._kernel_size) + (max_filters, 1)
          depthwise_initializer = self._depthwise_initializer

        self._create_trainable_variable(
            name='kernel',
            shape=kernel_shape,
            initializer=depthwise_initializer,
            regularizer=masked_depthwise_regularizer)

      output_shape = get_conv_output_shape(
          input_shape, self._strides, output_filters=max_filters)
      self._built = True
      return output_shape

  def apply(self, inputs, training):
    del training
    assert self._built

    with tf.variable_scope(self._scope, reuse=True):
      kernel = self._get_trainable_tensor('kernel')
      if self._transpose_depthwise_kernels:
        # Transpose the depthwise kernel back to the right shape.
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 2])
      mask = tf.stop_gradient(self._mask)
      masked_kernel = kernel * mask
      masked_kernel = tf.cast(masked_kernel, inputs.dtype)
      padding = 'SAME'
      if self._use_explicit_padding:
        padding = 'VALID'
        inputs = tf.pad(
            tensor=inputs,
            paddings=_compute_explicit_padding(self._kernel_size, (1, 1)))
      result = tf.nn.depthwise_conv2d(
          inputs,
          masked_kernel,
          strides=(1,) + tuple(self._strides) + (1,),
          padding=padding)
      return result


class MaskedConv2D(Layer):
  """2D masked convolution network layer."""

  def __init__(self,
               kernel_size,
               input_mask,
               output_mask,
               strides=(1, 1),
               kernel_initializer=tf.initializers.he_normal(),
               kernel_regularizer=None,
               bias_initializer=tf.initializers.zeros(),
               bias_regularizer=None,
               use_bias=False,
               use_explicit_padding=False,
               name=None):
    super(MaskedConv2D, self).__init__()

    if isinstance(kernel_size, int):
      self._kernel_size = (kernel_size, kernel_size)
    else:
      self._kernel_size = kernel_size

    if isinstance(strides, int):
      self._strides = (strides, strides)
    else:
      self._strides = strides

    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bias = use_bias
    self._name = name
    self._built = False
    self._use_explicit_padding = use_explicit_padding

    if input_mask is None:
      self._input_mask = None
    else:
      self._input_mask = tf.reshape(input_mask, [1, 1, -1, 1])
    self._output_mask = tf.reshape(output_mask, [1, 1, 1, -1])
    if use_bias:
      self._bias_mask = output_mask

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'MaskedConv2D') as scope:
      self._scope = scope

      if self._input_mask is None:
        max_input_filters = int(input_shape[-1])
      else:
        max_input_filters = int(self._input_mask.shape[2])
      max_output_filters = int(self._output_mask.shape[3])
      if int(input_shape[-1]) != max_input_filters:
        raise ValueError(
            'padded input filter size ({:d}) must match the max possible '
            'effective input filter size ({:d}) in scope: {}.'
            .format(int(input_shape[-1]), max_input_filters, scope.name))

      if not self._built:
        mask = tf.stop_gradient(self._output_mask)
        if self._input_mask is not None:
          input_mask = tf.stop_gradient(self._input_mask)
          mask = mask * input_mask

        masked_kernel_regularizer = _regularizer_over_masked_variable(
            self._kernel_regularizer, mask)

        kernel_shape = tuple(self._kernel_size) + (
            max_input_filters, max_output_filters)
        self._create_trainable_variable(
            name='kernel',
            shape=kernel_shape,
            initializer=self._kernel_initializer,
            regularizer=masked_kernel_regularizer)

        if self._use_bias:
          masked_bias_regularizer = _regularizer_over_masked_variable(
              self._bias_regularizer, self._bias_mask)
          self._create_trainable_variable(
              name='bias',
              shape=(max_output_filters,),
              initializer=self._bias_initializer,
              regularizer=masked_bias_regularizer)

      output_shape = get_conv_output_shape(
          input_shape, self._strides, output_filters=max_output_filters)
      self._built = True
      return output_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      kernel = self._get_trainable_tensor('kernel')
      mask = tf.stop_gradient(self._output_mask)
      if self._input_mask is not None:
        input_mask = tf.stop_gradient(self._input_mask)
        mask = mask * input_mask
      masked_kernel = kernel * mask
      masked_kernel = tf.cast(masked_kernel, inputs.dtype)
      padding = 'SAME'
      if self._use_explicit_padding:
        padding = 'VALID'
        inputs = tf.pad(
            tensor=inputs,
            paddings=_compute_explicit_padding(self._kernel_size, (1, 1)))
      result = tf.nn.conv2d(
          inputs,
          masked_kernel,
          strides=[1] + list(self._strides) + [1],
          padding=padding)

      if self._use_bias:
        bias = self._get_trainable_tensor('bias')
        bias_mask = tf.stop_gradient(self._bias_mask)
        masked_bias = bias * bias_mask
        masked_bias = tf.cast(masked_bias, result.dtype)
        return tf.nn.bias_add(result, masked_bias)
      else:
        return result


class MaskedStatelessBatchNorm(Layer):
  """Masked stateless batch normalization layer."""

  def __init__(self,
               mask,
               epsilon=1e-12,
               center=True,
               scale=True,
               beta_initializer=tf.initializers.zeros(),
               gamma_initializer=tf.initializers.ones(),
               name=None):
    super(MaskedStatelessBatchNorm, self).__init__()
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._beta_initializer = beta_initializer
    self._gamma_initializer = gamma_initializer
    self._name = name
    self._built = False

    self._mask = tf.reshape(mask, [1, 1, 1, -1])

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'MaskedStatelessBatchNorm') as scope:
      self._scope = scope

      max_filters = int(self._mask.shape[3])
      if int(input_shape[-1]) != max_filters:
        raise ValueError(
            'padded input filter size ({:d}) must match the max possible '
            'effective filter size ({:d}).'
            .format(int(input_shape[-1]), max_filters))

      if not self._built:
        if self._center:
          self._create_trainable_variable(
              name='beta',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=self._beta_initializer)

        if self._scale:
          self._create_trainable_variable(
              name='gamma',
              shape=[int(input_shape[-1])],
              dtype=tf.float32,
              initializer=self._gamma_initializer)

      self._built = True
      return input_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      scale = self._get_trainable_tensor('gamma') if self._scale else None
      offset = self._get_trainable_tensor('beta') if self._center else None
      mask = tf.stop_gradient(self._mask)
      mask = tf.cast(mask, inputs.dtype)
      result, unused_mean, unused_var = tf.nn.fused_batch_norm(
          inputs,
          scale=scale,
          offset=offset,
          epsilon=self._epsilon,
          is_training=True)
      result = result * mask
      return result


def create_mask(choices, selection):
  """Create a 1-dimensional mask for the given choices and selection."""
  # Opt out tf.gather if there's only a single option
  if len(choices) == 1:
    k = choices[0]
  else:
    k = tf.gather(choices, selection)
  n = max(choices)
  mask = tf.sequence_mask(k, n, dtype=tf.float32)
  return mask
