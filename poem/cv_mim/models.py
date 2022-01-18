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

"""Defines model architectures."""

import tensorflow as tf

from poem.core import data_utils

layers = tf.keras.layers

# Define embedder type.
TYPE_EMBEDDER_POINT = 'POINT'
TYPE_EMBEDDER_GAUSSIAN = 'GAUSSIAN'


def build_linear(hidden_dim, weight_max_norm=0.0,
                 weight_initializer='he_normal', name=None, **_):
  """Builds a linear layer.

  Args:
    hidden_dim: An integer for the dimension of the output.
    weight_max_norm: A float for the maximum weight norm to clip at. Use
      non-positive to ignore.
    weight_initializer: A string for kernel weight initializer.
    name: A string for the name of the layer.
    **_: Unused bindings for keyword arguments.

  Returns:
    A configured linear layer.
  """
  if weight_max_norm > 0.0:
    weight_constraint = tf.keras.constraints.MaxNorm(max_value=weight_max_norm)
  else:
    weight_constraint = None
  return layers.Dense(hidden_dim,
                      kernel_initializer=weight_initializer,
                      bias_initializer=weight_initializer,
                      kernel_constraint=weight_constraint,
                      bias_constraint=weight_constraint,
                      name=name)


def build_linear_layers(hidden_dim, num_layers, dropout_rate=0.0,
                        use_batch_norm=True, name=None, **kwargs):
  """Builds a number of linear layers.

  Note that each layer contains a sequence of Linear, Batch Normalization,
  Dropout and RELU.

  Args:
    hidden_dim: An integer for the dimension of the linear layer.
    num_layers: An integer for the number of layers to build.
    dropout_rate: A float for the dropout rate. Use non-positive to ignore.
    use_batch_norm: A boolean indicating whether to use batch normalization.
    name: A string for the name of the layer.
    **kwargs: A dictionary for additional arguments. Supported arguments include
      `weight_max_norm` and `weight_initializer`.

  Returns:
    A configured sequence of linear layers.
  """
  linear_layers = tf.keras.Sequential(name=name)

  for _ in range(num_layers):
    linear_layers.add(build_linear(hidden_dim, **kwargs))
    if use_batch_norm:
      linear_layers.add(layers.BatchNormalization())
    linear_layers.add(layers.ReLU())
    if dropout_rate > 0.0:
      linear_layers.add(layers.Dropout(dropout_rate))

  return linear_layers


class ResLinearBlock(layers.Layer):
  """Residual linear block."""

  def __init__(self, hidden_dim, num_layers, name=None, **kwargs):
    """Initializer.

    Args:
      hidden_dim: An integer for the dimension of the linear layer.
      num_layers: An integer for the number of linear layers in the block.
      name: A string for the name of the layer.
      **kwargs: A dictionary for additional arguments. Supported arguments
        include `weight_max_norm`, `weight_initializer`, `dropout_rate` and
        `use_batch_norm`.
    """
    super(ResLinearBlock, self).__init__(name=name)
    self.linear_layers = build_linear_layers(hidden_dim, num_layers, **kwargs)

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor. Shape = [..., hidden_dim].
    """
    return self.linear_layers(inputs, training=training) + inputs


class Sampling(layers.Layer):
  """Layer sampling tensors from unit Gaussian distribution."""

  def call(self, inputs):
    """Computes a forward pass.

    Args:
      inputs: A list of input tensors containing the mean and the log var of the
        Gaussian distribution to draw samples.

    Returns:
      An output tensor.
    """
    z_mean, z_log_var = inputs
    epsilon = tf.random.normal(
        shape=z_mean.shape, mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class PointEmbedder(layers.Layer):
  """Implements a point embedder."""

  def __init__(self, embedding_layer, name=None):
    """Initializer.

    Args:
      embedding_layer: A `tf.keras.Layer` object for the layer of the embedding.
      name: A string for the name of the layer.
    """
    super(PointEmbedder, self).__init__(name=name)
    self.embedder = embedding_layer

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor. Shape = [..., embedding_dim].
    """
    return self.embedder(inputs, training=training)


class GaussianEmbedder(layers.Layer):
  """Implements a Gaussian embedder."""

  def __init__(self, embedding_mean_layer, embedding_logvar_layer, name=None):
    """Initializer.

    Args:
      embedding_mean_layer: A `tf.keras.Layer` object for the mean of the
        embedding.
      embedding_logvar_layer: A `tf.keras.Layer` object for the logvar of the
        embedding.
      name: A string for the name of the layer.
    """
    super(GaussianEmbedder, self).__init__(name=name)
    self.embedder_mean = embedding_mean_layer
    self.embedder_logvar = embedding_logvar_layer
    self.sampling = Sampling()

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Note that the embedder returns the output drawn from the Gaussian
    distribution during training while it returns the mean during inference.
    During training, the embedder adds the KL-divergence regularization loss to
    the `self.losses` attribute.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor. Shape = [..., embedding_dim].
    """
    z_mean = self.embedder_mean(inputs)

    if training:
      # TODO(longzh,liuti): Refactor the scale parameter out as an argument.
      z_logvar = tf.nn.tanh(self.embedder_logvar(inputs)) * 10.0
      z = self.sampling((z_mean, z_logvar))

      # Add KL-divergence regularization loss.
      kl_loss = -0.5 * tf.reduce_sum(
          z_logvar - tf.square(z_mean) - tf.exp(z_logvar) + 1, axis=-1)
      self.add_loss(tf.reduce_mean(kl_loss))
    else:
      z = z_mean

    return z


class LegacyGaussianEmbedder(layers.Layer):
  """Implements a Gaussian embedder."""

  def __init__(self, embedding_mean_layer, embedding_stddev_layer, name=None):
    """Initializer.

    Args:
      embedding_mean_layer: A `tf.keras.Layer` object for the mean of the
        embedding.
      embedding_stddev_layer: A `tf.keras.Layer` object for the stddev of the
        embedding.
      name: A string for the name of the layer.
    """
    super(LegacyGaussianEmbedder, self).__init__(name=name)
    self.embedder_mean = embedding_mean_layer
    self.embedder_stddev = embedding_stddev_layer
    self.sampling = Sampling()

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Note that the embedder returns the output drawn from the Gaussian
    distribution during training while it returns the mean during inference.
    During training, the embedder adds the KL-divergence regularization loss to
    the `self.losses` attribute.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor. Shape = [..., embedding_dim].
    """
    z_mean = self.embedder_mean(inputs)

    if training:
      z_stddev = tf.nn.elu(self.embedder_stddev(inputs)) + 1.0
      epsilon = tf.random.normal(shape=z_mean.shape, mean=0.0, stddev=1.0)
      z = z_mean + z_stddev * epsilon

      # Add KL-divergence regularization loss.
      kl_loss = -0.5 * tf.reduce_sum(
          tf.math.log(tf.math.maximum(1e-12, tf.math.square(z_stddev))) -
          tf.math.square(z_mean) - tf.math.square(z_stddev) + 1,
          axis=-1)
      self.add_loss(tf.reduce_mean(kl_loss))
    else:
      z = z_mean

    return z


class SimpleModel(tf.keras.Model):
  """Simple model architecture with a point or Gaussian embedder."""

  def __init__(self,
               output_shape,
               embedder=TYPE_EMBEDDER_POINT,
               hidden_dim=1024,
               num_residual_linear_blocks=2,
               num_layers_per_block=2,
               **kwargs):
    """Initializer.

    Args:
      output_shape: A tuple for the shape of the output.
      embedder: A string for the type of the embedder.
      hidden_dim: An integer for the dimension of linear layers.
      num_residual_linear_blocks: An integer for the number of residual linear
        blocks.
      num_layers_per_block: An integer for the number of layers in each block.
      **kwargs: A dictionary for additional arguments. Supported arguments
        include `weight_max_norm`, `weight_initializer`, `dropout_rate` and
        `use_batch_norm`.
    """
    super(SimpleModel, self).__init__()

    self.blocks = [layers.Flatten(name='flatten')]
    self.blocks.append(
        build_linear_layers(hidden_dim, 1, name='fc0', **kwargs))
    for i in range(num_residual_linear_blocks):
      self.blocks.append(
          ResLinearBlock(
              hidden_dim,
              num_layers_per_block,
              name='res_fcs' + str(i + 1),
              **kwargs))

    self.embedder_output_shape = output_shape
    output_dim = tf.math.reduce_prod(output_shape)

    if embedder == TYPE_EMBEDDER_POINT:
      self.blocks.append(
          PointEmbedder(build_linear(output_dim, **kwargs), name='embedder'))
    elif embedder == TYPE_EMBEDDER_GAUSSIAN:
      self.blocks.append(
          GaussianEmbedder(
              build_linear(output_dim, **kwargs),
              build_linear(output_dim, **kwargs),
              name='embedder'))
    else:
      raise ValueError('Unknown embedder: {}'.format(embedder))

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor and a list of output activations from all the layers.
    """
    activations = {}
    x = inputs

    for block in self.blocks:
      x = block(x, training=training)
      activations[block.name] = x

    output = activations['embedder']
    if len(self.embedder_output_shape) > 1:
      output = data_utils.recursively_expand_dims(
          output, axes=[-1] * (len(self.embedder_output_shape) - 1))
      output = data_utils.reshape_by_last_dims(
          output, last_dim_shape=self.embedder_output_shape)
      activations['embedder'] = output

    return output, activations


class SemGraphConv(layers.Layer):
  """Implements Semantic Graph Convolution.

  Reference:
    Zhao et al. Semantic Graph Convolutional Networks for 3D Human Pose
    Regression. https://arxiv.org/pdf/1904.03345.pdf.
  """

  def __init__(self,
               units,
               affinity_matrix,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    """Initializer.

    Args:
      units: An integer for the output dimension of the layer.
      affinity_matrix: A tensor for the keypoint affinity matrix.
      activation: Activation function to use.
      use_bias: A boolean for whether the layer uses a bias vector.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function for the kernel weights matrix.
      bias_regularizer: Regularizer function for the bias vector.
      activity_regularizer: Regularizer function for the output of the layer.
      kernel_constraint: Constraint function for the kernel weights matrix.
      bias_constraint: Constraint function for the bias vector.
      **kwargs: A dictionary for additional arguments.
    """
    super(SemGraphConv, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units)
    self.affinity_matrix = tf.convert_to_tensor(
        affinity_matrix, dtype=tf.dtypes.float32) > 0.

    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    """Builds the layer.

    Args:
      input_shape: A TensorShape for the shape of the input tensor.
    """
    last_dim = input_shape[-1]

    self.kernel = self.add_weight(
        'kernel',
        shape=(2, last_dim, self.units),
        initializer=self.kernel_initializer,
        constraint=self.kernel_constraint,
        trainable=True)

    self.affinity_weight = self.add_weight(
        'affinity_weight',
        shape=self.affinity_matrix.shape,
        initializer=tf.constant_initializer(1.),
        trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=(self.units,),
          initializer=self.bias_initializer,
          constraint=self.bias_constraint,
          trainable=True)
    else:
      self.bias = None

    self.input_spec = layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.built = True

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor.
    """
    eye_outputs = tf.matmul(inputs, self.kernel[0])
    noneye_outputs = tf.matmul(inputs, self.kernel[1])

    affinity_matrix = tf.where(
        self.affinity_matrix, self.affinity_weight, -9e15)
    affinity_matrix = tf.nn.softmax(affinity_matrix, axis=1)

    eye_matrix = tf.eye(tf.shape(affinity_matrix)[0], dtype=tf.dtypes.float32)
    outputs = tf.matmul(affinity_matrix * eye_matrix, eye_outputs) + \
        tf.matmul(affinity_matrix * (1. - eye_matrix), noneye_outputs)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation:
      outputs = self.activation(outputs)

    return outputs


class PreAggGraphConv(layers.Layer):
  """Implements Pre-Aggregation Graph Convolution.

  Reference:
    Liu et al. A Comprehensive Study of Weight Sharing in Graph Networks for 3D
    Human Pose Estimation.
    http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550324.pdf.
  """

  def __init__(self,
               units,
               affinity_matrix,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    """Initializer.

    Args:
      units: An integer for the output dimension of the layer.
      affinity_matrix: A tensor for the keypoint affinity matrix.
      activation: Activation function to use.
      use_bias: A boolean for whether the layer uses a bias vector.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function for the kernel weights matrix.
      bias_regularizer: Regularizer function for the bias vector.
      activity_regularizer: Regularizer function for the output of the layer.
      kernel_constraint: Constraint function for the kernel weights matrix.
      bias_constraint: Constraint function for the bias vector.
      **kwargs: A dictionary for additional arguments.
    """
    super(PreAggGraphConv, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units)

    def normalize(x):
      """Row-normalization of the matrix."""
      rowsum = tf.math.reduce_sum(x, axis=1)
      rowsum = tf.math.maximum(rowsum, 1e-12)
      rowinv = rowsum ** -1
      return tf.matmul(tf.linalg.diag(rowinv), x)

    self.affinity_matrix = tf.convert_to_tensor(
        affinity_matrix, dtype=tf.dtypes.float32)
    self.affinity_matrix = normalize(self.affinity_matrix)

    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, input_shape):
    """Builds the layer.

    Args:
      input_shape: A TensorShape for the shape of the input tensor.
    """
    last_dim = input_shape[-1]

    self.kernel = self.add_weight(
        'kernel',
        shape=(2, self.affinity_matrix.shape[0], last_dim, self.units),
        initializer=self.kernel_initializer,
        constraint=self.kernel_constraint,
        trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=(self.units,),
          initializer=self.bias_initializer,
          constraint=self.bias_constraint,
          trainable=True)
    else:
      self.bias = None

    self.input_spec = layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.built = True

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor.
    """
    affinity_matrix = tf.expand_dims(self.affinity_matrix, axis=1)
    eye_matrix = tf.eye(affinity_matrix.shape[0], dtype=tf.dtypes.float32)
    eye_matrix = tf.expand_dims(eye_matrix, axis=1)
    x = tf.expand_dims(inputs, axis=-3)

    eye_outputs = tf.matmul(x, self.kernel[0])
    eye_outputs = tf.matmul(affinity_matrix * eye_matrix, eye_outputs)
    eye_outputs = tf.squeeze(eye_outputs, axis=-2)

    noneye_outputs = tf.matmul(x, self.kernel[1])
    noneye_outputs = tf.matmul(affinity_matrix * (1. - eye_matrix),
                               noneye_outputs)
    noneye_outputs = tf.squeeze(noneye_outputs, axis=-2)

    outputs = eye_outputs + noneye_outputs

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation:
      outputs = self.activation(outputs)

    return outputs


class GraphNonLocal(layers.Layer):
  """Implements graph non-local layer.

  Reference:
    Wang et al. Non-local Neural Networks. https://arxiv.org/pdf/1711.07971.pdf.
  """

  def __init__(self,
               inter_filters=None,
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    """Initializer.

    Args:
      inter_filters: An integer for the output dimension of the layer.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function for the kernel weights matrix.
      bias_regularizer: Regularizer function for the bias vector.
      activity_regularizer: Regularizer function for the output of the layer.
      kernel_constraint: Constraint function for the kernel weights matrix.
      bias_constraint: Constraint function for the bias vector.
      **kwargs: A dictionary for additional arguments.
    """
    super(GraphNonLocal, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.inter_filters = inter_filters
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

  def build(self, input_shape):
    """Builds the layer.

    Args:
      input_shape: A TensorShape for the shape of the input tensor.
    """
    last_dim = input_shape[-1]
    conv1d_kwargs = dict(
        kernel_size=1,
        strides=1,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        activity_regularizer=self.activity_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_constraint=self.bias_constraint
    )

    if self.inter_filters is None:
      self.inter_filters = last_dim // 2

    self.g = layers.Conv1D(self.inter_filters, **conv1d_kwargs)
    self.theta = layers.Conv1D(self.inter_filters, **conv1d_kwargs)
    self.phi = layers.Conv1D(self.inter_filters, **conv1d_kwargs)
    self.concat_project = layers.Conv1D(
        1, activation='relu', use_bias=False, **conv1d_kwargs)

    self.w = tf.keras.Sequential([
        layers.Conv1D(last_dim, **conv1d_kwargs),
        layers.BatchNormalization(
            beta_initializer='zeros', gamma_initializer='zeros')
    ])

    self.input_spec = layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.built = True

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor.
    """
    # Shape = [batch_size, num_nodes, hidden_dim].
    g_x = self.g(inputs, training=training)

    # Shape = [batch_size, 1, num_nodes, hidden_dim].
    theta_x = self.theta(inputs, training=training)
    theta_x = tf.expand_dims(theta_x, axis=1)

    # Shape = [batch_size, num_nodes, 1, hidden_dim].
    phi_x = self.phi(inputs, training=training)
    phi_x = tf.expand_dims(phi_x, axis=2)

    theta_x = tf.tile(theta_x, tf.constant([1, phi_x.shape[1], 1, 1], tf.int32))
    phi_x = tf.tile(phi_x, tf.constant([1, 1, theta_x.shape[2], 1], tf.int32))

    # Shape = [batch_size, num_nodes, num_nodes, 2 * hidden_dim].
    concat_feature = tf.concat([theta_x, phi_x], axis=-1)

    # Shape = [batch_size, num_nodes, num_nodes].
    f = self.concat_project(concat_feature, training=training)
    f = tf.squeeze(f, axis=-1)

    f_div_c = f / inputs.shape[1]
    y = tf.matmul(f_div_c, g_x)
    y = self.w(y, training=training)
    return y + inputs


def build_graph_conv_layers(hidden_dim,
                            affinity_matrix,
                            num_layers,
                            gconv_class,
                            dropout_rate=0.0,
                            use_batch_norm=True,
                            name=None,
                            **kwargs):
  """Builds a number of graph convolutional layers.

  Note that each layer contains a sequence of GraphConv, Batch Normalization,
  Dropout and RELU.

  Args:
    hidden_dim: An integer for the dimension of the linear layer.
    affinity_matrix: A tensor for the keypoint affinity matrix.
    num_layers: An integer for the number of layers to build.
    gconv_class: A graph convolutional class to use.
    dropout_rate: A float for the dropout rate. Use non-positive to ignore.
    use_batch_norm: A boolean indicating whether to use batch normalization.
    name: A string for the name of the layer.
    **kwargs: A dictionary for additional arguments. Supported arguments
        include `kernel_initializer` and `bias_initializer`.

  Returns:
    A configured sequence of graph convolutional layers.
  """
  graph_conv_layers = tf.keras.Sequential(name=name)

  for _ in range(num_layers):
    graph_conv_layers.add(gconv_class(hidden_dim, affinity_matrix, **kwargs))
    if use_batch_norm:
      graph_conv_layers.add(layers.BatchNormalization())
    graph_conv_layers.add(layers.ReLU())
    if dropout_rate > 0.0:
      graph_conv_layers.add(layers.Dropout(dropout_rate))

  return graph_conv_layers


class ResGraphConvBlock(layers.Layer):
  """Residual graph convolutional block."""

  def __init__(self,
               hidden_dim,
               affinity_matrix,
               num_layers,
               gconv_class,
               name=None,
               **kwargs):
    """Initializer.

    Args:
      hidden_dim: An integer for the dimension of the layer.
      affinity_matrix: A tensor for the keypoint affinity matrix.
      num_layers: An integer for the number of layers in the block.
      gconv_class: A graph convolutional class to use.
      name: A string for the name of the layer.
      **kwargs: A dictionary for additional arguments. Supported arguments
        include `dropout_rate` and `use_batch_norm`.
    """
    super(ResGraphConvBlock, self).__init__(name=name)
    self.graph_conv_layers = build_graph_conv_layers(
        hidden_dim, affinity_matrix, num_layers, gconv_class, **kwargs)

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor. Shape = [..., hidden_dim].
    """
    return self.graph_conv_layers(inputs, training=training) + inputs


class GCN(tf.keras.Model):
  """Implements Graph Convolutional Network."""

  def __init__(self,
               output_dim,
               affinity_matrix,
               gconv_class,
               hidden_dim=128,
               num_residual_gconv_blocks=4,
               num_layers_per_block=2,
               use_non_local=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    """Initializer.

    Args:
      output_dim: An integer for the dimension of the output.
      affinity_matrix: A tensor for the keypoint affinity matrix.
      gconv_class: A graph convolutional class to use.
      hidden_dim: An integer for the dimension of layers.
      num_residual_gconv_blocks: An integer for the number of residual blocks.
      num_layers_per_block: An integer for the number of layers in each block.
      use_non_local: A boolean indicating whether to use non-local layers.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      **kwargs: A dictionary for additional arguments. Supported arguments
        include `dropout_rate` and `use_batch_norm`.
    """
    super(GCN, self).__init__()

    self.blocks = [
        build_graph_conv_layers(
            hidden_dim,
            affinity_matrix,
            1,
            gconv_class,
            name='gconv0',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs)
    ]
    if use_non_local:
      self.blocks.append(GraphNonLocal())

    for i in range(num_residual_gconv_blocks):
      self.blocks.append(
          ResGraphConvBlock(
              hidden_dim,
              affinity_matrix,
              num_layers_per_block,
              gconv_class,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              name='res_gconvs' + str(i + 1),
              **kwargs))
      if use_non_local:
        self.blocks.append(GraphNonLocal())

    self.blocks.append(
        gconv_class(
            output_dim,
            affinity_matrix,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='gconv' + str(num_residual_gconv_blocks + 1)))

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      An output tensor and a list of output activations from all the layers.
    """
    activations = {}
    x = inputs

    for block in self.blocks:
      x = block(x, training=training)
      activations[block.name] = x

    return x, activations


class LikelihoodEstimator(tf.keras.Model):
  """Network to estimate the log-likelihood."""

  def __init__(self, output_dim):
    """Initializer.

    Args:
      output_dim: An integer for the dimension of the output.
    """
    super(LikelihoodEstimator, self).__init__()

    self.mean_block = tf.keras.Sequential([
        layers.Dense(output_dim),
        layers.BatchNormalization(),
        layers.ELU(),
        layers.Dense(output_dim)
    ])

    self.logvar_block = tf.keras.Sequential([
        layers.Dense(output_dim),
        layers.BatchNormalization(),
        layers.ELU(),
        layers.Dense(output_dim, activation=tf.nn.tanh)
    ])

  def call(self, inputs, training=False):
    """Computes a forward pass.

    Args:
      inputs: An input tensor.
      training: A boolean indicating whether the call is for training or not.

    Returns:
      Two output tensors for mean and logvar.
    """
    z_mean = self.mean_block(inputs, training=training)
    z_logvar = self.logvar_block(inputs, training=training)
    return z_mean, z_logvar
