# coding=utf-8
"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

from __future__ import absolute_import
from __future__ import division

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from ieg.models.networks import StrategyNetBase

# TODO(zizhaoz): Remove in the future.
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
FLAGS = flags.FLAGS


def variable(name, shape, dtype, initializer, trainable):
  """Returns a TF variable with the passed in specifications."""
  var = tf.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      trainable=trainable)
  return var


class ResNet(object):
  """ResNet model."""

  def __init__(self, is_training, data_format, batch_norm_decay,
               batch_norm_epsilon):
    """ResNet constructor."""
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format

  def set_training_mode(self, mode):
    self._is_training = mode

  def forward_pass(self, x):
    raise NotImplementedError(
        'forward_pass() is implemented in ResNet sub classes')

  def _residual_v1(self,
                   x,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_v1') as name_scope:
      orig_x = x

      x = self._conv(x, 3, out_filter, stride)
      x = self._batch_norm(x)
      x = self._relu(x)

      x = self._conv(x, 3, out_filter, 1)
      x = self._batch_norm(x)

      if in_filter != out_filter:
        orig_x = self._avg_pool(orig_x, stride, stride)
        pad = (out_filter - in_filter) // 2
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = self._relu(tf.add(x, orig_x))

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _residual_v2(self,
                   x,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers with preactivation, plan A shortcut."""

    with tf.name_scope('residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 3, out_filter, stride)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 3, out_filter, 1)
      # x = self._conv(x, 3, out_filter, [1, 1, 1, 1])

      if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        orig_x = self._avg_pool(orig_x, stride, stride)
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _bottleneck_residual_v2(self,
                              x,
                              in_filter,
                              out_filter,
                              stride,
                              activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

    with tf.name_scope('bottle_residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      # pad when stride isn't unit
      x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 1, out_filter, 1, is_atrous=True)

      if in_filter != out_filter:
        orig_x = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      if self._data_format == 'channels_first':
        x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
      else:
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      axis = 1
    else:
      axis = -1
    return tf.layers.batch_normalization(
        x,
        momentum=self._batch_norm_decay,
        center=True,
        scale=True,
        epsilon=self._batch_norm_epsilon,
        training=self._is_training,
        trainable=True,
        fused=True,
        axis=axis)

  def _relu(self, x):
    return tf.nn.relu(x)

  def _fully_connected(self, x, out_dim, loss_type):
    """Fully connected layers."""
    with tf.name_scope('fully_connected') as name_scope:
      if loss_type == 'softmax':
        x = tf.layers.dense(x, out_dim)
      elif loss_type == 'sigmoid' or loss_type == 'focal':
        x = tf.layers.dense(
            inputs=x,
            units=out_dim,
            bias_initializer=tf.constant_initializer(-np.log(out_dim - 1)))

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _avg_pool(self, x, pool_size, stride):
    with tf.name_scope('avg_pool') as name_scope:
      x = tf.layers.average_pooling2d(
          x, pool_size, stride, 'SAME', data_format=self._data_format)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _global_avg_pool(self, x):
    with tf.name_scope('global_avg_pool') as name_scope:
      assert x.get_shape().ndims == 4
      if self._data_format == 'channels_first':
        x = tf.reduce_mean(x, [2, 3])
      else:
        x = tf.reduce_mean(x, [1, 2])
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x


class ResNetCifar(ResNet):
  """Cifar model with ResNetV1 and basic residual block."""

  def __init__(self,
               num_layers,
               is_training=False,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5,
               version='v1',
               num_classes=10,
               data_format='channels_last',
               loss_type='softmax'):
    super(ResNetCifar, self).__init__(is_training, data_format,
                                      batch_norm_decay, batch_norm_epsilon)
    self.n = (num_layers - 2) // 6
    self.num_classes = num_classes
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    self.version = version
    self.loss_type = loss_type

  def forward_pass(self, x, input_data_format='channels_last', as_dict=False):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # # Image standardization.
    # x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    if self.version == 'v1':
      # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
      res_func = self._residual_v1
    elif self.version == 'v2':
      # Use basic (non-bottleneck) block and ResNet V2 (pre-activation).
      res_func = self._residual_v2
    else:  # 'bv2'
      # Use bottleneck block and ResNet V2 (pre-activation).
      res_func = self._bottleneck_residual_v2

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, self.filters[i + 1], self.filters[i + 1], 1)

    hiddens = x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes, self.loss_type)
    if as_dict:
      return {'logits': x, 'hiddens': hiddens}
    return x


def decay_weights(weight_decay_rate, trainable_vars):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  tf.logging.info('{} variables to decay, rate {}'.format(
      len(trainable_vars), weight_decay_rate))
  costs = []
  for var in trainable_vars:
    costs.append(tf.cast(tf.nn.l2_loss(var), tf.float32))
  return tf.multiply(weight_decay_rate, tf.add_n(costs))


def get_var(list_of_tensors, prefix_name=None, with_name=None):
  """Gets specific variable.

  Args:
    list_of_tensors: A list of candidate tensors
    prefix_name:  Variable name starts with prefix_name
    with_name: with_name in the variable name

  Returns:
    Obtained tensor list
  """
  if prefix_name is None:
    return list_of_tensors
  else:
    specific_tensor = []
    specific_tensor_name = []
    if prefix_name is not None:
      for var in list_of_tensors:
        if var.name.startswith(prefix_name):
          if with_name is None or with_name in var.name:
            specific_tensor.append(var)
            specific_tensor_name.append(var.name)
    return specific_tensor


class ResNetBuilder(StrategyNetBase):
  """Builder of ResNet for IEG."""

  def __init__(self, *args, **kwargs):
    super(ResNetBuilder, self).__init__()
    self.model = ResNetCifar(*args, **kwargs)
    self.wd = FLAGS.weight_decay  # Original 2e-4 # REMOVE

  def get_partial_variables(self, level=-1):
    vs = []
    if level == 0:
      # only get last fc layer
      for v in self.trainable_variables:
        if 'dense' in v.name:
          vs.append(v)
    elif level < 0:
      vs = self.trainable_variables
    else:
      raise ValueError
    assert vs, 'Length of obtained partial variable is 0'
    return vs

  def __call__(self, inputs, name, training, reuse=True, custom_getter=None):
    with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
      self.model.set_training_mode(training)  # Ugly design
      outputs = self.model.forward_pass(inputs, as_dict=True)['logits']

      if not isinstance(reuse, bool) or not reuse:
        # If it is tf.AUTO_REUSE or True to make sure regularization_loss is
        # added once.
        self.regularization_loss = decay_weights(
            self.wd, get_var(tf.trainable_variables(), name))
        self.init(name, with_name='batch_normalization', outputs=outputs)
        self.count_parameters(name)
    return outputs
