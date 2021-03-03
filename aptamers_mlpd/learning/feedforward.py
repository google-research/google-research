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

# Lint as: python3
"""Provides a simple feed forward neural net class for regression tasks.

The FeedForward class constructs a tensorflow model. The constructor takes a
config object and expects metaparameter settings to be stored as config
attributes. Any object that has the appropriate attributes can be passed in to
configure the model since FeedForward does not assume the config object is
anything more than a dumb struct and does not attempt to serialize it.
Nonetheless, it will usually be an instance of tf.HParams. The model
constructor sets up TF variables to hold the weights, but the fprop method
builds the fprop graph for the model using the weights.

A few notes on what metaparameters will be needed in general:

For each fully connected (FC) layer, we need to select a size, an activation
function, a dropout rate, and an initialiation scheme. At construction time, we
only need the sizes and initialization.

Right now there is no support for convolutional layers.

Eventually, for each convolutional layer we need the activation function, a
dropout rate, a filter size, a number of filters, an initialization scheme, and
in principle padding and strides, but we will fix those. At construction time,
we only need the filter size, number of filters, and initialization.

Although in principle we can interleave FC and conv layers, life is complicated
enough as it is. Let's do zero or more conv layers followed by zero or more FC
layers. During metaparameter search, based on limitations of metaparater tuning
policies, we will need to fix the number of layers of each type in a given
study. We also might need to use introspection to add attributes to the hpconfig
object the tuner gives us, since the tuner interface has limited flexibility
for multi-dimensional metaparameters.

The model class doesn't know anything about training, so training
metaparameters in the config object will be ignored.
"""

import collections

from six.moves import map

import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor as lt
from xxx import layers as contrib_layers
from xxx import framework as contrib_framework


nonlinearities = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'tanh': tf.tanh,
    'sigmoid': tf.sigmoid
}


def _stack_inputs_by_rank(inputs):
  """Create 2D and 3D input tensors from a dictionary of inputs.

  3D inputs are stacked together for use in (optional) convolutional layers.
  2D inputs are only used in fully-connected layers.

  Args:
    inputs: Dict[str, lt.LabeledTensor] providing input features. All features
      must be 2D or 3D labeled tensors with a 'batch' axis as their first
      dimension. 3D tensors must have 'position' as their second axis. The last
      axis of all tensors is allowed to vary, because raw input features may
      have different names for labels that are more meaningful than generic
      "features" or "channels".

  Returns:
    Tuple[Optional[lt.LabeledTensor], Optional[lt.LabeledTensor]], where the
    first labeled tensor, if present, has axes ['batch', 'feature'] and the
    second labeled tensor, if present, has axes ['batch', 'position',
    'channel'].

  Raises:
    ValueError: if the result tensors do not have the same batch axis.
  """
  inputs_2d = []
  inputs_3d = []
  for key in sorted(inputs):
    # outputs should be fixed across randomized dict iteration order
    tensor = inputs[key]
    if len(tensor.axes) == 2:
      tensor = lt.rename_axis(tensor, list(tensor.axes.keys())[-1], 'feature')
      inputs_2d.append(tensor)
    elif len(tensor.axes) == 3:
      assert list(tensor.axes.values())[1].name == 'position'
      tensor = lt.rename_axis(tensor, list(tensor.axes.keys())[-1], 'channel')
      inputs_3d.append(tensor)
    else:
      raise AssertionError('unexpected rank')

  combined_2d = lt.concat(inputs_2d, 'feature') if inputs_2d else None
  combined_3d = lt.concat(inputs_3d, 'channel') if inputs_3d else None
  if combined_2d is not None and combined_3d is not None:
    if list(combined_2d.axes.values())[0] != list(combined_2d.axes.values())[0]:
      raise ValueError('mismatched batch axis')
  return combined_2d, combined_3d


class FeedForward:
  """Class implementing a simple feedforward neural net in tensorflow.

  Attributes:
    batch_axis: lt.Axis for batches of examples.
    input_position_axis: lt.Axis for input positions.
    input_channel_axis: lt.Axis for input channels.
    logit_axis: lt.Axis for logit channels, output from the `frop` method.
    config: a reference to the config object we used to specify the model. In
      general we expect it to be an instance of tf.HParams, but it could be
      anything with the right attributes.
    params: list of weights and biases
  """

  def __init__(self, dummy_inputs, logit_axis, config):

    self.logit_axis = logit_axis
    self.config = config

    self.fc_sizes = getattr(config, 'fc_hid_sizes', []) + [len(logit_axis)]
    self.fc_init_factors = (
        getattr(config, 'fc_init_factors', []) + [config.output_init_factor])

    if not dummy_inputs:
      raise ValueError('network has size 0 input')
    if logit_axis.size == 0:
      raise ValueError('network has size 0 output')

    if len({
        len(self.fc_sizes), len(self.fc_init_factors), len(config.dropouts)
    }) != 1:
      raise ValueError('invalid hyperparameter config for fc layers')
    self.num_fc_layers = len(self.fc_sizes)

    self._conv_config = _ConvConfig(*[
        getattr(config, 'conv_' + field, []) for field in _ConvConfig._fields
    ])
    if len(set(map(len, self._conv_config))) != 1:
      raise ValueError('invalid hyperparameter config for conv layers')
    self.num_conv_layers = len(self._conv_config.depths)

    self.fprop = tf.make_template('feedforward', self._fprop)
    # create variables
    self.fprop(dummy_inputs, mode='test')
    self.params = contrib_framework.get_variables(
        scope=self.fprop.variable_scope.name)

  def _fprop(self, inputs, mode):
    """Builds the fprop graph from inputs up to logits.

    Args:
      inputs: input LabeledTensor with axes [batch_axis, input_position_axis,
        input_channel_axis].
      mode: either 'test' or 'train', determines whether we add dropout nodes

    Returns:
      Logits tensor with axes [batch_axis, logit_axis].

    Raises:
      ValueError: mode must be 'train' or 'test'
    """
    if mode not in ['test', 'train']:
      raise ValueError('mode must be one of "train" or "test"')
    is_training = mode == 'train'

    inputs_2d, inputs_3d = _stack_inputs_by_rank(inputs)

    if inputs_2d is None and inputs_3d is None:
      raise ValueError('feedforward model has no inputs')

    # Get the batch axis from the actual inputs, because we set up the graph
    # with unknown batch size.
    example_inputs = inputs_3d if inputs_2d is None else inputs_2d
    batch_axis = example_inputs.axes['batch']

    w_initializer = tf.uniform_unit_scaling_initializer
    nonlinearity = nonlinearities[self.config.nonlinearity]

    if inputs_3d is not None:
      conv_args = list(zip(*self._conv_config))
      net = contrib_layers.stack(
          inputs_3d,
          conv1d,
          conv_args,
          scope='conv',
          padding='SAME',
          activation_fn=nonlinearity,
          w_initializer=w_initializer)
      net = contrib_layers.flatten(net)
      if inputs_2d is not None:
        net = tf.concat([net, inputs_2d], 1)
    else:
      net = inputs_2d

    if net.get_shape()[-1].value == 0:
      raise ValueError('feature dimension has size 0')

    keep_probs = [1 - d for d in self.config.dropouts]
    fc_args = list(zip(self.fc_sizes, keep_probs, self.fc_init_factors))

    net = contrib_layers.stack(
        net,
        dropout_and_fully_connected,
        fc_args[:-1],
        scope='fc',
        is_training=is_training,
        activation_fn=nonlinearity,
        w_initializer=w_initializer)

    # the last layer should not have a non-linearity
    net = dropout_and_fully_connected(
        net, *fc_args[-1], scope='fc_final', is_training=is_training,
        activation_fn=None, w_initializer=w_initializer)

    logits = lt.LabeledTensor(net, [batch_axis, self.logit_axis])
    return logits


# must match the order of conv1d's arguments
_ConvConfig = collections.namedtuple(
    '_ConvConfig', 'depths, widths, strides, rates, init_factors')


def conv1d(inputs,
           filter_depth,
           filter_width,
           stride=1,
           rate=1,
           init_factor=1.0,
           w_initializer=None,
           **kwargs):
  """Adds a convolutional 1d layer.

  If rate is 1 then a standard convolutional layer will be added,
  if rate is > 1 then an dilated (atrous) convolutional layer will
  be added.

  Args:
    inputs: a 3-D tensor  `[batch_size, in_width, in_channels]`.
    filter_depth: integer, the number of output channels.
    filter_width: integer, size of the convolution kernel.
    stride: integer, size of the convolution stride.
    rate: integer, the size of the convolution dilation.
    init_factor: passed to `w_initializer`.
    w_initializer: function to call to create a weights initializer.
    **kwargs: passed on to `layers.conv2d`.

  Returns:
    A tensor variable representing the result of the series of operations.
  Raises:
    Error if rate > 1 and stride != 1. Current implementation of
    atrous_conv2d does not allow a stride other than 1.
  """
  with tf.name_scope('conv1d'):
    # expand from 1d to 2d convolutions to match conv2d API
    # take inputs (which are only the inputs_3d layers) from
    # ['batch', 'position', 'channel'] to ['batch', 1, 'position', 'channel']
    # convolutions are done over the middle 2 dimensions.
    inputs_2d = tf.expand_dims(inputs, 1)
    kernel_size_2d = [1, filter_width]
    stride_2d = [1, stride]
    rate_2d = [1, rate]
    weights_initializer = w_initializer(factor=init_factor)
    output_2d = contrib_layers.conv2d(
        inputs_2d,
        filter_depth,
        kernel_size_2d,
        stride_2d,
        rate=rate_2d,
        weights_initializer=weights_initializer,
        **kwargs)

    output = tf.squeeze(output_2d, [1])
    return output


def dropout_and_fully_connected(inputs,
                                num_outputs,
                                keep_prob=0.5,
                                init_factor=1.0,
                                is_training=True,
                                w_initializer=None,
                                **kwargs):
  """Apply dropout followed by a fully connected layer.

  Args:
    inputs: A tensor of with at least rank 2 and value for the last dimension,
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer or long, the number of output units in the layer.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    init_factor: passed to `w_initializer`.
    is_training: A bool `Tensor` indicating whether or not the model
      is in training mode. If so, dropout is applied and values scaled.
      Otherwise, dropout is skipped.
    w_initializer: Function to call to create a weights initializer.
    **kwargs: passed on to `layers.fully_connected`.

  Returns:
    A tensor variable representing the result of the series of operations.
  """
  net = contrib_layers.dropout(
      inputs, keep_prob=keep_prob, is_training=is_training)
  weights_initializer = w_initializer(factor=init_factor)
  net = contrib_layers.fully_connected(
      net, num_outputs, weights_initializer=weights_initializer, **kwargs)
  return net
