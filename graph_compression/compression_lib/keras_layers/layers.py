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

"""Keras compressed layers.

Contains keras layers that work with the compression operator. See
example usage in examples/mnist_eager_mode/mnist_compression.py
"""

from absl import logging
import tensorflow as tf
from graph_compression.compression_lib import compression_op as compression


class CompressedLinearLayer(tf.keras.layers.Layer):
  """A Compressed linear layer with the compression op.

  In a CompressedLinearLayer, the W matrix is replaced by the compressed version
  of W, where specific form of the compressed version of W is determined by the
  compressor. For example, if compressor is
  compression_op.LowRankDecompMatrixCompressor, then W is replaced by
  alpha * W + (1 - alpha) * tf.matmul(B, C), see compression_op.py for more
  details.
  """

  def __init__(self, input_dim, num_hidden_nodes, compressor):
    """Initializer.

    Args:
      input_dim: int
      num_hidden_nodes: int
      compressor: a matrix compressor object (instance of a subclass of
        compression_op.MatrixCompressorInferface)
    """
    super(CompressedLinearLayer, self).__init__()
    self.num_hidden_nodes = num_hidden_nodes
    self.compressor = compressor
    self.w = self.add_weight(
        shape=(input_dim, self.num_hidden_nodes),
        initializer='random_normal',
        trainable=True)
    self.b = self.add_weight(
        shape=(self.num_hidden_nodes,),
        initializer='random_normal',
        trainable=True)

  def set_up_variables(self):
    """Set up variables used by compression_op."""
    self.compression_op = compression.CompressionOpEager()
    self.compression_op.set_up_variables(self.w, self.compressor)

  @tf.function
  def call(self, inputs):
    self.compressed_w = self.compression_op.get_apply_compression()
    return tf.matmul(inputs, self.compressed_w) + self.b

  def run_alpha_update(self, step_number):
    """Run alpha update for the alpha parameter in compression_op.

    Args:
      step_number: step number in the training process.
    Note: This method should only be called during training.
    """
    self.compression_op.run_update_step(step_number)


class CompressedDense(tf.keras.layers.Dense):
  """A compressed Dense keras layer with the compression op.

  The compression_obj.get_spec().rank must be divisibe by
  compression_obj.get_spec().input_block_size. The input size to the layer
  must be divisible by compression_obj.get_spec().input_block_size which
  in turn must be divisible by compression_obj.get_spec().rank.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               compression_obj=None,
               **kwargs):
    """Initializer.

    Args:
      units: Positive integer, dimensionality of the output space.
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
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      bias_constraint: Constraint function applied to the bias vector.
      compression_obj: Compression object contaning compression parameters  The
        compression_obj.get_spec().rank must be divisibe by
        compression_obj.get_spec().input_block_size. The input size to the layer
        must be divisible by compression_obj.get_spec().input_block_size which
        in turn must be divisible by compression_obj.get_spec().rank.
      **kwargs: additional keyword arguments.
    """

    super().__init__(units, activation, use_bias, kernel_initializer,
                     bias_initializer, kernel_regularizer, bias_regularizer,
                     activity_regularizer, kernel_constraint, bias_constraint,
                     **kwargs)
    self.compression_obj = compression_obj

  def build(self, input_shape):
    super().build(input_shape)
    # Pass self.kernel to compression_op.
    self.compression_obj.apply_compression_keras(self.kernel, layer=self)
    self.compression_op = self.compression_obj.get_last_compression_op()
    logging.info('self.compression_op is %s', self.compression_op)
    logging.info('input_shape units  is %s %s', input_shape,
                 self.units)
    input_block_size = self.compression_obj.get_spec().input_block_size
    rank = self.compression_obj.get_spec().rank
    assert (
        input_shape[-1] % input_block_size == 0
    ), 'input_shape[-1] {} must be divisible by input_block_size {}'.format(
        input_shape[-1], input_block_size)
    assert (
        input_block_size %
        rank == 0), 'input_block_size {} must be divisible by rank {}'.format(
            input_block_size, rank)
    logging.info(
        'in build kernel a_matrix b_matrix and c_matrix shape is %s %s %s %s',
        self.kernel.shape, self.compression_op.a_matrix_tfvar.shape,
        self.compression_op.b_matrix_tfvar.shape,
        self.compression_op.c_matrix_tfvar.shape)

  @tf.function
  def call(self, inputs, training=True):
    return self.activation(
        self.compression_op.get_apply_matmul_keras(inputs) + self.bias)


class CompressedConv2D(tf.keras.layers.Conv2D):
  """A compressed Conv2D keras layer.

  Uses the simple idea of reducing the number of output filters by the
  rank parameter passed int he compression_obj and then projecting it back
  to the right number (see the b_matrix variable).
  Note that output filters must be divisible by compression_obj.get_spec().rank.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               compression_obj=None,
               **kwargs):
    """Initializer.

    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Specifying any stride
        value != 1 is incompatible with specifying any `dilation_rate` value !=
        1.
      padding: one of `"valid"` or `"same"` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding evenly to the left/right or
        up/down of the input such that output has the same height/width
        dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch_size, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        `channels_last`.
      dilation_rate: an integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Can be a single integer to
        specify the same value for all spatial dimensions. Currently, specifying
        any `dilation_rate` value != 1 is incompatible with specifying any
        stride value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved
        separately with `filters / groups` filters. The output is the
        concatenation of all the `groups` results along the channel axis. Input
        channels and `filters` must both be divisible by `groups`.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix (see
        `keras.initializers`).
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`).
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation") (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix (see
        `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).
      compression_obj: Compression object specifying the rank that is used to
        compress the number of output filters.  filters must be divisible by
        compression_obj.get_spec().rank.
      **kwargs: additional keyword arguments.
    """

    self.orig_filters = filters
    rank = compression_obj.get_spec().rank
    assert (
        filters % rank == 0
    ), 'Number of filters {} must be divisible by compression_op_spec.rank {}'.format(
        filters, rank)
    self.compressed_filters = filters // rank
    super().__init__(
        filters=self.compressed_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

    self.compression_obj = compression_obj

  def build(self, input_shape):
    super().build(input_shape)
    self.b_matrix = self.add_weight(
        'b_matrix',
        shape=[self.compressed_filters, self.orig_filters],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

  @tf.function
  def call(self, inputs):
    if self.data_format == 'channels_first':
      return tf.einsum('ijkl,jm->imkl', super().call(inputs), self.b_matrix)
    else:
      return tf.einsum('iklj,jm->iklm', super().call(inputs), self.b_matrix)
