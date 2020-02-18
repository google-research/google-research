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

"""Locally connected layers with specific rank.

Classic locally connected layers include filters of size of input
(eg height x width of an image).
Here, rank represents number of filters. These filters is considered as bases
and are linearly combined to produce filters of size of input.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils


class InputDependentCombiningWeights(tf.keras.Model):
  """Input-dependent weights that combines the locally connected kernels."""

  def __init__(self, spatial_rank):
    super(InputDependentCombiningWeights, self).__init__()

    self.dim_reduction_layer = tf.keras.layers.Conv2D(
        filters=spatial_rank,
        kernel_size=1,
        strides=(1, 1),
        dilation_rate=1,
        padding='same',
        use_bias=True,
        data_format='channels_last')
    self.dilations = [1, 2, 4, 8]
    self.multiscale_layers = []
    for r in self.dilations:
      self.multiscale_layers.append(
          tf.keras.layers.DepthwiseConv2D(
              kernel_size=3,
              strides=(1, 1),
              dilation_rate=r,
              padding='valid',
              use_bias=True,
              data_format='channels_last'))
    ch = (len(self.multiscale_layers) + 2) * spatial_rank
    self.squeeze_layer = tf.keras.layers.Conv2D(
        filters=int(ch // 2),
        kernel_size=1,
        strides=(1, 1),
        activation=None,
        use_bias=True,
        data_format='channels_last')

    self.excite_layer = tf.keras.layers.Conv2D(
        filters=ch,
        kernel_size=1,
        strides=(1, 1),
        activation=None,
        use_bias=True,
        data_format='channels_last')

    self.proj_layer = tf.keras.layers.Conv2D(
        filters=spatial_rank,
        kernel_size=1,
        strides=(1, 1),
        use_bias=True,
        data_format='channels_last')

  def __call__(self, x, size):
    """Call function.

    Args:
      x: 4D input tensor of size [batch, height, width, channels].
      size: 2D tensor specifying the size of the layer output.

    Returns:
      4D tensor of size [batch, height, width, spatial_rank]
    """
    x_lowd = self.dim_reduction_layer(x)
    x_pool = tf.reduce_mean(x_lowd, axis=[1, 2], keepdims=True)
    input_size = min(x_lowd.shape.as_list()[1:3])

    x_multiscale = [
        tf.image.resize_bilinear(x_lowd, size, align_corners=True),
        tf.image.resize_bilinear(x_pool, size, align_corners=True),]

    for r, layer in zip(self.dilations, self.multiscale_layers):
      if r <= int((input_size - 1) / 2):
        x_multiscale.append(
            tf.image.resize_bilinear(layer(x_lowd), size, align_corners=True))

    x_multiscale = tf.concat(x_multiscale, axis=-1)
    x_s = self.squeeze_layer(x_multiscale)
    x_s = tf.nn.relu(x_s)
    x_e = self.excite_layer(x_s)
    x_e = tf.nn.sigmoid(x_e)
    output = self.proj_layer(x_e)
    return output


class LowRankLocallyConnected2D(tf.keras.layers.LocallyConnected2D):
  """Locally-connected layer for 2D inputs with low rank on spatial dimensions.

  The `LocallyConnected2D` layer works similarly
  to the `Conv2D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each
  different patch of the input.
  Examples:
  ```python
      # apply a 3x3 unshared weights convolution with 64 output filters on a
      32x32 image
      # with `data_format="channels_last"`:
      model = Sequential()
      model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
      # now model.output_shape == (None, 30, 30, 64)
      # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
      parameters
      # add a 3x3 unshared weights convolution on top, with 32 output filters:
      model.add(LocallyConnected2D(32, (3, 3)))
      # now model.output_shape == (None, 28, 28, 32)
  ```
  Attributes:
      spatial_rank: (Integer) Number of filter basis.
      normalize_weights: (String) Type of combining weights. Can be either:
        '': for no normalization,
        'norm': for normalizing to unit norm,
        'softmax': for normalizing to unit sum.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: tuple of 2 integers or list of tuples, specifying the width
        and height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      dilations: An int or tuple of ints that has length 1, 2 or 4, defaults to
        1. When using different dilations across kernel bases use list
        containing dilation values.
        The dilation factor for each dimension of input. If a single value is
        given it is replicated in the H and W dimension.
        By default the N and C dimensions are set to 1. If set to k > 1,
        there will be k-1 skipped cells between each filter element on that
        dimension. The dimension order is determined by the value of
        data_format, see below for details. Dilations in the batch and
        depth dimensions if a 4-d tensor must be 1.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the width and height. Can be a single integer to
        specify the same value for all spatial dimensions.
      padding: Currently only support `"valid"` (case-insensitive). `"same"`
        will be supported in future.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, height, width,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
      share_row_combining_weights: (Boolean) Allows sharing row kernel combining
        weights and biases across columns.
      share_col_combining_weights: (Boolean) Allows sharing columns
        kernel combining weights and biases across rows.
      combining_weights_initializer: Initializer for the `combining_weights`
        weights matrix. If 'conv_init', it initializes the combining weights to
        constant, which corresponds to initialize the layer to a convolution
        layer.
      combining_weights_regularizer: Regularizer function applied to the
        `combining_weights` weights matrix.
      combining_weights_constraint: Constraint function applied to the
        combining_weights matrix.
      input_dependent: (Boolean) whether combining weights are
        input dependent or fixed. If True share_row_combining_weights,
        and share_col_combining_weights needs both to be False.

  Input shape:
      4D tensor with shape: `(samples, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, rows, cols, channels)` if
        data_format='channels_last'.
  Output shape:
      4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have changed
        due to padding.
  """

  def __init__(self,
               spatial_rank,
               filters,
               kernel_size,
               dilations=1,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               activation=None,
               use_bias=True,
               kernel_initializer='he_uniform',
               combining_weights_initializer='conv_init',
               bias_initializer='zeros',
               kernel_regularizer=None,
               combining_weights_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               combining_weights_constraint=None,
               bias_constraint=None,
               share_row_combining_weights=True,
               share_col_combining_weights=True,
               normalize_weights='softmax',
               use_spatial_bias=True,
               input_dependent=False,
               **kwargs):
    super(LowRankLocallyConnected2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size[0] if isinstance(kernel_size, list)
        else kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        implementation=2,
        **kwargs)
    self.spatial_rank = spatial_rank
    self.use_spatial_bias = use_spatial_bias
    self.normalize_weights = normalize_weights
    self.share_row_combining_weights = share_row_combining_weights
    self.share_col_combining_weights = share_col_combining_weights
    self.combining_weights_initializer = combining_weights_initializer
    self.combining_weights_regularizer = combining_weights_regularizer
    self.combining_weights_constraint = combining_weights_constraint
    self.input_dependent = input_dependent

    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)

    if isinstance(kernel_size, list) and not isinstance(dilations, list):
      # Repeat the dilations to length of kernel sizes list.
      dilations = [dilations] * len(kernel_size)
      # Convert any int instances to (row, col) tuple.
      for i, size in enumerate(kernel_size):
        if isinstance(size, int):
          kernel_size[i] = (size, size)

    elif isinstance(dilations, list) and not isinstance(kernel_size, list):
      # Repeat the kernel size to length of dilations list.
      kernel_size = [kernel_size] * len(dilations)

    # Note here a list is reserved specifically for different kernel operations.
    # To provide kernel_size for different axes, one needs to use tuple.
    if not isinstance(kernel_size, list):
      # Make it a list.
      kernel_size = [kernel_size]

    # Note here a list is reserved specifically for different kernel operations.
    # To provide dilations for different axes, one needs to use tuple.
    if not isinstance(dilations, list):
      # Make it a list.
      dilations = [dilations]

    if ((len(kernel_size) != len(dilations)) or
        ((len(kernel_size) > 1) and (len(kernel_size) != spatial_rank))):
      raise ValueError('kernel_sizes and dilations must be a list of size'
                       'spatial_rank if different kernel bases are used.')

    if len(kernel_size) > 1 and padding.upper() != 'SAME':
      raise ValueError(
          'Padding should be same, if different filter bases are specified')

    if len(kernel_size) == 1:
      # Identical kernel bases.
      self.kernel_size = kernel_size[0]
      self.dilations = dilations[0]
    else:
      # Different kernel bases.
      self.kernel_size = kernel_size
      self.dilations = dilations

    if self.normalize_weights not in ['', 'softmax', 'norm']:
      raise ValueError('Weights normalization type is incorrect')

    if self.input_dependent and (
        self.share_row_combining_weights or self.share_col_combining_weights):
      raise ValueError('Can not share combining weights across'
                       ' rows and/or columns when using input-dependent mode.')

    if self.input_dependent:
      self.combining_weights_layer = InputDependentCombiningWeights(
          self.spatial_rank)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if self.data_format == 'channels_last':
      channel_axis = -1
      input_row, input_col = input_shape[1:-1]
    else:
      channel_axis = 1
      input_row, input_col = input_shape[2:]

    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_filter = int(input_shape[channel_axis])

    if self.data_format == 'channels_last':
      input_row, input_col = input_shape[1:-1]
      input_filter = input_shape[3]
    else:
      input_row, input_col = input_shape[2:]
      input_filter = input_shape[1]

    if (((input_row is None) and (
        (self.share_row_combining_weights, self.share_col_combining_weights) in
        [(True, False), (False, False)])) or
        ((input_col is None) and (
            (self.share_row_combining_weights, self.share_col_combining_weights)
            in [(False, True), (False, False)]))):
      raise ValueError('The spatial dimensions of the inputs to '
                       ' a LowRankLocallyConnected2D layer '
                       'should be fully-defined, but layer received '
                       'the inputs shape ' + str(input_shape))

    # Compute output shapes.
    # Compute using the first filter since output will be same across filters.
    kernel_size = self.kernel_size[0] if isinstance(
        self.kernel_size, list) else self.kernel_size

    dilations = self.dilations[0] if isinstance(
        self.dilations, list) else self.dilations

    output_row = conv_utils.conv_output_length(
        input_row, kernel_size[0], self.padding, self.strides[0],
        dilation=dilations)
    output_col = conv_utils.conv_output_length(
        input_col, kernel_size[1], self.padding, self.strides[1],
        dilation=dilations)

    if isinstance(self.kernel_size, list):
      # Different filters.
      self.kernel_bases = []
      for i, kernel_size in enumerate(self.kernel_size):
        kernel_bases_shape = (
            kernel_size[0], kernel_size[1], input_filter, self.filters)
        self.kernel_bases.append(
            self.add_weight(
                shape=kernel_bases_shape,
                initializer=self.kernel_initializer,
                name='kernel_bases%d' %i,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint))
    else:
      self.kernel_bases_shape = (self.kernel_size[0],
                                 self.kernel_size[1], input_filter,
                                 self.spatial_rank * self.filters)
      self.kernel_shape = (output_row, output_col, self.kernel_size[0],
                           self.kernel_size[1], input_filter, self.filters)
      self.kernel_bases = self.add_weight(
          shape=self.kernel_bases_shape,
          initializer=self.kernel_initializer,
          name='kernel_bases',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    self.output_row = output_row
    self.output_col = output_col

    if not (
        self.share_row_combining_weights or self.share_col_combining_weights):
      if self.input_dependent:
        self.combining_weights = None
      else:
        self.combining_weights_shape = (
            output_row, output_col, self.spatial_rank)

        initializer = (tf.constant_initializer(1./np.sqrt(self.spatial_rank)) if
                       self.combining_weights_initializer == 'conv_init' else
                       self.combining_weights_initializer)

        self.wts = self.add_weight(
            shape=self.combining_weights_shape,
            initializer=initializer,
            name='combining_weights',
            regularizer=self.combining_weights_regularizer,
            constraint=self.combining_weights_constraint)
        # If self.wts is overwritten it is removed from layer.weights.
        # Thus, below assignment is necessary.
        self.combining_weights = self.wts

    else:
      c = 1. / (float(self.share_row_combining_weights) +
                float(self.share_col_combining_weights))      # Scale for init.
      initializer = (tf.constant_initializer(c/np.sqrt(self.spatial_rank)) if
                     self.combining_weights_initializer == 'conv_init' else
                     self.combining_weights_initializer)
      combining_weights_shape_row = (output_row, self.spatial_rank)
      combining_weights_shape_col = (output_col, self.spatial_rank)

      self.wts_row = tf.constant([[0.]])
      self.wts_col = tf.constant([[0.]])
      if self.share_row_combining_weights:
        self.wts_row = self.add_weight(
            shape=combining_weights_shape_row,
            initializer=initializer,
            name='combining_weights_row',
            regularizer=self.combining_weights_regularizer,
            constraint=self.combining_weights_constraint)

      if self.share_col_combining_weights:
        self.wts_col = self.add_weight(
            shape=combining_weights_shape_col,
            initializer=tf.constant_initializer(c/np.sqrt(self.spatial_rank)) if
            self.combining_weights_initializer == 'conv_init' else
            self.combining_weights_initializer,
            name='combining_weights_col',
            regularizer=self.combining_weights_regularizer,
            constraint=self.combining_weights_constraint)

      if self.share_row_combining_weights and self.share_col_combining_weights:
        self.combining_weights = tf.math.add(
            self.wts_col[tf.newaxis],
            self.wts_row[:, tf.newaxis],
            name='combining_weights')
        self.combining_weights_shape = (
            output_row, output_col, self.spatial_rank)

      elif self.share_row_combining_weights:
        self.combining_weights = tf.identity(
            self.wts_row, name='combining_weights')
        self.combining_weights_shape = combining_weights_shape_row

      elif self.share_col_combining_weights:
        self.combining_weights = tf.identity(
            self.wts_col, name='combining_weights')
        self.combining_weights_shape = combining_weights_shape_col

    if not self.input_dependent:
      if self.normalize_weights == 'softmax':
        # Normalize the weights to sum to 1.
        self.combining_weights = tf.nn.softmax(
            self.combining_weights,
            axis=-1,
            name='normalized_combining_weights')
      elif self.normalize_weights == 'norm':
        # Normalize the weights to sum to preserve kernel var.
        self.combining_weights = tf.math.l2_normalize(
            self.combining_weights, axis=-1, epsilon=1e-12,
            name='normalized_combining_weights')

    if (self.input_dependent or
        isinstance(self.kernel_size, list) or
        ((self.share_row_combining_weights, self.share_col_combining_weights)
         in [(True, False), (False, True)])):
      # Different kernel bases can not be combined.
      # Shape may not be defined for one of axes in one dimension separate wts.
      self.kernel = None
    else:
      self.kernel = tf.tensordot(
          self.combining_weights,
          tf.reshape(self.kernel_bases, (self.kernel_size[0],
                                         self.kernel_size[1],
                                         input_filter,
                                         self.spatial_rank,
                                         self.filters)),
          [[-1], [-2]],
          name='kernel')

    self.bias_spatial = 0.
    self.bias_channels = 0.
    if self.use_spatial_bias:
      if not (self.share_row_combining_weights or
              self.share_col_combining_weights):
        self.bias_spatial = self.add_weight(
            shape=(output_row, output_col, 1),
            initializer=self.bias_initializer,
            name='spatial_bias',
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

      else:
        self.bias_row = 0.
        self.bias_col = 0.
        if self.share_row_combining_weights:
          self.bias_row = self.add_weight(
              shape=(output_row, 1, 1),
              initializer=self.bias_initializer,
              name='bias_row',
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)

        if self.share_col_combining_weights:
          self.bias_col = self.add_weight(
              shape=(1, output_col, 1),
              initializer=self.bias_initializer,
              name='bias_col',
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        self.bias_spatial = tf.math.add(
            self.bias_row, self.bias_col, name='spatial_bias')

    if self.use_bias:
      self.bias_channels = self.add_weight(
          shape=(1, 1, self.filters),
          initializer=self.bias_initializer,
          name='bias_channels',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)

    self.bias = tf.math.add(self.bias_spatial, self.bias_channels, name='bias')

    if self.data_format == 'channels_last':
      self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
    else:
      self.input_spec = InputSpec(ndim=4, axes={1: input_filter})

    self.built = True

  def call(self, inputs):
    if isinstance(self.kernel_size, list):
      # Different filters.
      convs = []
      for dilations, kernel in zip(
          self.dilations, self.kernel_bases):
        convs.append(tf.nn.conv2d(
            inputs,
            filter=kernel,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format='NHWC' if self.data_format == 'channels_last'
            else 'NCHW',
            dilations=dilations))

      if self.data_format == 'channels_last':
        convs_reshaped = tf.stack(convs, axis=3)

      else:
        convs_reshaped = tf.stack(convs, axis=1)

    else:
      # Same structure filters.
      convs = tf.nn.conv2d(
          inputs,
          filter=self.kernel_bases,
          strides=self.strides,
          padding=self.padding.upper(),
          data_format='NHWC' if self.data_format == 'channels_last'
          else 'NCHW',
          dilations=self.dilations)

      batch_size = convs.shape[0]
      if self.data_format == 'channels_last':
        convs_reshaped = tf.reshape(convs, [
            batch_size,
            -1 if self.output_row is None else self.output_row,
            -1 if self.output_col is None else self.output_col,
            self.spatial_rank,
            self.filters,
            ])

      elif self.data_format == 'channels_first':
        convs_reshaped = tf.reshape(convs, [
            batch_size,
            self.spatial_rank,
            self.filters,
            -1 if self.output_row is None else self.output_row,
            -1 if self.output_col is None else self.output_col,
            ])

    # Input-dependent combining weights.
    if self.input_dependent:
      size = [self.output_row, self.output_col]
      if self.data_format == 'channels_last':
        self.combining_weights = self.combining_weights_layer(
            inputs, size)
      elif self.data_format == 'channels_first':
        self.combining_weights = self.combining_weights_layer(
            tf.transpose(inputs, [0, 2, 3, 1]), size)
      if self.normalize_weights == 'softmax':
        # Normalize the weights to sum to 1.
        self.combining_weights = tf.nn.softmax(
            self.combining_weights, axis=-1,
            name='normalized_combining_weights')
      elif self.normalize_weights == 'norm':
        # Normalize the weights to sum to preserve kernel var.
        self.combining_weights = tf.math.l2_normalize(
            self.combining_weights, axis=-1, epsilon=1e-12,
            name='normalized_combining_weights')

    # Combine weights with output.
    share_combining_weights = (
        self.share_row_combining_weights, self.share_col_combining_weights)
    if self.data_format == 'channels_last':
      if share_combining_weights == (True, False):
        equation = 'ijklm,jl->ijkm'
      elif share_combining_weights == (False, True):
        equation = 'ijklm,kl->ijkm'
      elif share_combining_weights == (True, True):
        equation = 'ijklm,jkl->ijkm'
      elif share_combining_weights == (False, False):
        if self.input_dependent:
          equation = 'ijklm,ijkl->ijkm'
        else:
          equation = 'ijklm,jkl->ijkm'

    elif self.data_format == 'channels_first':
      if share_combining_weights == (True, False):
        equation = 'ijklm,lj->iklm'
      elif share_combining_weights == (False, True):
        equation = 'ijklm,mj->iklm'
      elif share_combining_weights == (True, True):
        equation = 'ijklm,lmj->iklm'
      elif share_combining_weights == (False, False):
        if self.input_dependent:
          equation = 'ijklm,ilmj->iklm'
        else:
          equation = 'ijklm,lmj->iklm'

    outputs = tf.einsum(equation, convs_reshaped, self.combining_weights)
    bias = self.bias
    if bias != 0 and self.data_format == 'channels_first':
      bias = tf.transpose(bias, [2, 0, 1])

    outputs = outputs + bias
    outputs = self.activation(outputs)
    return outputs

