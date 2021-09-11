# coding=utf-8
"""Small ResNet for CIFAR."""

from __future__ import absolute_import
from __future__ import division

import functools

from absl import flags
import tensorflow.compat.v1 as tf

from ieg.models.networks import StrategyNetBase

FLAGS = flags.FLAGS

L2 = 1e-4


def resnet_layer(inputs,
                 training,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs (tensor): input tensor from input image or previous layer
    training: if in traning stage
    num_filters (int): Conv2D number of filters
    kernel_size (int): Conv2D square kernel dimensions
    strides (int): Conv2D square stride dimensions
    activation (string): activation name
    batch_normalization (bool): whether to include batch normalization
    conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)

  Returns:
    x (tensor): tensor as input to the next layer
  """
  regularizer = tf.keras.regularizers.L2(L2)
  conv = functools.partial(
      tf.layers.conv2d,
      filters=num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer=tf.initializers.glorot_normal(),
      kernel_regularizer=regularizer)

  batchnormalization_layer = functools.partial(
      tf.layers.batch_normalization,
      momentum=0.9,
      epsilon=1e-5,
      fused=True,
      training=training)

  x = inputs
  if conv_first:
    x = conv(inputs=x)
    if batch_normalization:
      x = batchnormalization_layer(inputs=x)
    if activation is not None:
      x = tf.nn.relu(x)
  else:
    if batch_normalization:
      x = batchnormalization_layer(inputs=x)
    if activation is not None:
      x = tf.nn.relu(x)
    x = conv(x)

  return x


def resnet_model_fn(inputs,
                    training,
                    num_classes,
                    num_res_blocks,
                    weight_decay=1e-4,
                    num_filters_in=16):
  """Create model for ResNet-v1.

  Args:
    inputs: input tensor.
    training: bool for training or evaluation mode.
    num_classes: number of classes.
    num_res_blocks: number of residual blocks depending on depth.
    weight_decay: weight decay.
    num_filters_in: initial filter size.

  Returns:
    logits tensor.
  """
  global L2
  L2 = weight_decay

  x = resnet_layer(
      inputs=inputs,
      num_filters=num_filters_in,
      conv_first=True,
      training=training)

  # Instantiate the stack of residual units
  for stage in range(3):
    for res_block in range(num_res_blocks):
      with tf.variable_scope('stage%d_block%d' % (stage, res_block)):
        batch_normalization = True
        strides = 1
        if stage == 0:
          num_filters_out = num_filters_in * 4
          if res_block == 0:  # first layer and first stage
            activation = None
            batch_normalization = False
        else:
          num_filters_out = num_filters_in * 2
          if res_block == 0:  # first layer but not first stage
            strides = 2  # downsample

        # bottleneck residual unit
        y = resnet_layer(
            inputs=x,
            num_filters=num_filters_in,
            kernel_size=1,
            strides=strides,
            activation=activation,
            batch_normalization=batch_normalization,
            conv_first=False,
            training=training)
        y = resnet_layer(
            inputs=y,
            num_filters=num_filters_in,
            conv_first=False,
            training=training)
        y = resnet_layer(
            inputs=y,
            num_filters=num_filters_out,
            kernel_size=1,
            conv_first=False,
            training=training)
        if res_block == 0:
          # linear projection residual shortcut connection to match
          # changed dimsresnet_v2
          x = resnet_layer(
              inputs=x,
              num_filters=num_filters_out,
              kernel_size=1,
              strides=strides,
              activation=None,
              batch_normalization=False,
              training=training)
        x = tf.math.add(x, y)

    num_filters_in = num_filters_out

  batchnormalization_layer = functools.partial(
      tf.layers.batch_normalization,
      momentum=0.9,
      epsilon=1e-5,
      fused=True,
      training=training)
  x = batchnormalization_layer(inputs=x)
  x = tf.nn.relu(x)
  x = tf.layers.average_pooling2d(x, (8, 8), (1, 1))
  y = tf.layers.flatten(x)

  regularizer = tf.keras.regularizers.L2(L2)
  outputs = tf.layers.dense(
      y, units=num_classes, kernel_regularizer=regularizer)

  return outputs


class ResNet(StrategyNetBase):
  """ResNet meta class for CIFAR."""

  def __init__(self, num_classes, depth):
    super(ResNet, self).__init__()

    if (depth - 2) % 9 != 0:
      raise ValueError('depth == 9n+2')

    self.depth = depth
    self.num_classes = num_classes
    self.regularization_loss = 0
    # Used by MetaImage to compute l2 regularizer
    self.wd = 1e-4

  def save(self, path):
    self.model.save(path)

  def load_weights(self, path):
    self.model.load_weights(path)

  def get_partial_variables(self, level=-1):
    vs = []
    if level == 0:
      # only get last fc layer
      for v in self.trainable_variables:
        if 'dense' in v.name:
          vs.append(v)
    elif level == 1:
      prefix = self.name + '/' + 'stage2'
      vs = self.get_vars_before(self.trainable_variables, prefix)
    elif level == 2:
      prefix = self.name + '/' + 'stage1'
      vs = self.get_vars_before(self.trainable_variables, prefix)
    elif level < 0 or level > 2:
      vs = self.trainable_variables
    else:
      raise ValueError
    assert vs, 'Length of obtained partial variable can`t be 0'
    return vs

  def __call__(self, inputs, name, training, reuse=True, custom_getter=None):

    num_res_blocks = int((self.depth - 2) / 9)
    if self.created and not reuse:
      tf.logging.error('Warning: are you parallel?')

    with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
      outputs = resnet_model_fn(
          inputs,
          training,
          self.num_classes,
          num_res_blocks,
          weight_decay=self.wd)

      if not isinstance(reuse, bool) or not reuse:
        # If it is tf.AUTO_REUSE or True to make sure regularization_loss is
        # added once.
        self.regularization_loss = self.get_regularization_loss(
            scope_name=name + '/')
        self.init(name, with_name='batch_normalization', outputs=outputs)
        self.count_parameters(name)

    return outputs
