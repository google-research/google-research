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

"""CNN builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

import numpy as np

import tensorflow as tf
from cnn_quantization.tf_cnn_benchmarks import mlperf
from tensorflow.contrib.quantize.python import common
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.training import moving_averages


class ConvNetBuilder(object):
  """Builder of cnn net."""

  def __init__(self,
               input_op,
               input_nchan,
               phase_train,
               use_tf_layers,
               data_format='NCHW',
               dtype=tf.float32,
               variable_dtype=tf.float32,
               params=None):
    self.top_layer = input_op
    self.top_size = input_nchan
    self.phase_train = phase_train
    self.use_tf_layers = use_tf_layers
    self.data_format = data_format
    self.dtype = dtype
    self.variable_dtype = variable_dtype
    self.counts = collections.defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = (
        'channels_last' if data_format == 'NHWC' else 'channels_first')
    self.aux_top_layer = None
    self.aux_top_size = 0
    self.params = params

  def get_custom_getter(self):
    """Returns a custom getter that this class's methods must be called under.

    All methods of this class must be called under a variable scope that was
    passed this custom getter. Example:

    ```python
    network = ConvNetBuilder(...)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      network.conv(...)
      # Call more methods of network here
    ```

    Currently, this custom getter only does anything if self.use_tf_layers is
    True. In that case, it causes variables to be stored as dtype
    self.variable_type, then casted to the requested dtype, instead of directly
    storing the variable as the requested dtype.
    """
    def inner_custom_getter(getter, *args, **kwargs):
      """Custom getter that forces variables to have type self.variable_type."""
      if not self.use_tf_layers:
        return getter(*args, **kwargs)
      requested_dtype = kwargs['dtype']
      if not (requested_dtype == tf.float32 and
              self.variable_dtype == tf.float16):
        # Only change the variable dtype if doing so does not decrease variable
        # precision.
        kwargs['dtype'] = self.variable_dtype
      var = getter(*args, **kwargs)
      # This if statement is needed to guard the cast, because batch norm
      # assigns directly to the return value of this custom getter. The cast
      # makes the return value not a variable so it cannot be assigned. Batch
      # norm variables are always in fp32 so this if statement is never
      # triggered for them.
      if var.dtype.base_dtype != requested_dtype:
        var = tf.cast(var, requested_dtype)
      return var
    return inner_custom_getter

  @contextlib.contextmanager
  def switch_to_aux_top_layer(self):
    """Context that construct cnn in the auxiliary arm."""
    if self.aux_top_layer is None:
      raise RuntimeError('Empty auxiliary top layer in the network.')
    saved_top_layer = self.top_layer
    saved_top_size = self.top_size
    self.top_layer = self.aux_top_layer
    self.top_size = self.aux_top_size
    yield
    self.aux_top_layer = self.top_layer
    self.aux_top_size = self.top_size
    self.top_layer = saved_top_layer
    self.top_size = saved_top_size

  def get_variable(self, name, shape, dtype, cast_dtype, *args, **kwargs):
    # TODO(reedwm): Currently variables and gradients are transferred to other
    # devices and machines as type `dtype`, not `cast_dtype`. In particular,
    # this means in fp16 mode, variables are transferred as fp32 values, not
    # fp16 values, which uses extra bandwidth.
    var = tf.get_variable(name, shape, dtype, *args, **kwargs)
    return tf.cast(var, cast_dtype)

  def _fake_quant_with_min_max_vars(self, inputs, min_var, max_var, per_channel,
                                    num_bits, narrow_range):
    """Adds a fake quantization operation."""
    if per_channel:
      assert len(min_var.get_shape()) == 1
      assert len(max_var.get_shape()) == 1
      return tf.fake_quant_with_min_max_vars_per_channel(
          inputs, min_var, max_var,
          num_bits=num_bits, narrow_range=narrow_range)
    else:
      assert min_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
      assert max_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
      return tf.fake_quant_with_min_max_vars(
          inputs, min_var, max_var,
          num_bits=num_bits, narrow_range=narrow_range)

  def delayed_quant(self,
                    inputs,
                    quant_min,
                    quant_max,
                    per_channel=False,
                    num_bits=4,
                    narrow_range=True,
                    quant_delay=None):
    """Turn on fake quantization after certain delay."""

    # The fake quantization operation does not support tf.float16 yet.
    quant = self._fake_quant_with_min_max_vars(
        tf.cast(inputs, tf.float32),
        tf.cast(quant_min, tf.float32),
        tf.cast(quant_max, tf.float32),
        per_channel=per_channel,
        num_bits=num_bits,
        narrow_range=narrow_range)
    quant = tf.cast(quant, self.dtype)
    if quant_delay and quant_delay > 0:
      activate_quant = tf.greater_equal(
          common.CreateOrGetQuantizationStep(),
          quant_delay,
          name='activate_quant')
      quant = tf.cond(
          activate_quant,
          lambda: quant,
          lambda: inputs,
          name='delayed_quant')
    return quant

  def relu(self,
           inputs,
           init_x=None):
    """Construct a relu/relu_x layer on top of cnn."""
    if ((not self.params.use_relu_x)
        or self.params.last_act_name in tf.get_variable_scope().name):
      return tf.nn.relu(inputs)

    if self.params.relu_x_per_channel:
      if self.params.data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
      shape = [inputs.get_shape()[3]]
      reduce_dim = [0, 1, 2]
    else:
      shape = []
      reduce_dim = None

    if init_x is None:
      init_x = self.params.init_relu_x
    with tf.variable_scope('relu_x'):
      act = inputs
      trainable_x = self.params.relu_x_update == 'gradient_descent'
      x = tf.get_variable('x', shape, tf.float32,
                          initializer=tf.constant_initializer(init_x),
                          trainable=trainable_x)
      if self.params.relu_x_update == 'moving_average':
        act = tf.maximum(tf.minimum(inputs, init_x), 0)
        batch_max = tf.reduce_max(act, axis=reduce_dim, name='BatchMax')
        x = moving_averages.assign_moving_average(
            x, tf.cast(batch_max, tf.float32), 0.999, zero_debias=False,
            name='MovingAvgX')
      x = tf.cast(x, self.dtype)
      if self.params.relu_x_per_channel:
        act = tf.maximum(tf.minimum(act, tf.reshape(x, [1, 1, 1, -1])), 0)
      else:
        act = tf.maximum(tf.minimum(act, x), 0)
      if self.params.quant_act:
        print('Quantizing activation %s' % act.name)
        if self.params.relu_x_per_channel:
          zeros = tf.constant(0, dtype=tf.float32, shape=shape)
        else:
          zeros = 0
        act = self.delayed_quant(
            act,
            zeros,
            x,
            per_channel=self.params.relu_x_per_channel,
            num_bits=self.params.quant_act_bits,
            narrow_range=False,
            quant_delay=self.params.quant_act_delay)
      if self.params.relu_x_per_channel and self.params.data_format == 'NCHW':
        act = tf.transpose(act, [0, 3, 1, 2])

    return act

  def dorefa_weight_quantize(self, weights, quantize, num_bits,
                             per_channel, quant_delay):
    """Weights transformation and quantization described in the DoReFa paper."""

    weights_shape = weights.get_shape()
    weights_dim = len(weights_shape)

    tanh_weights = tf.tanh(weights)
    if per_channel:
      if weights_dim == 4:
        max_per_channel = tf.reduce_max(tf.abs(tanh_weights), axis=[0, 1, 2])
        norm_weights = tanh_weights/tf.reshape(max_per_channel, [1, 1, 1, -1])
      elif weights_dim == 2:
        max_per_channel = tf.reduce_max(tf.abs(tanh_weights), axis=[0])
        norm_weights = tanh_weights/tf.reshape(max_per_channel, [1, -1])
    else:
      norm_weights = tanh_weights/tf.reduce_max(tf.abs(tanh_weights))
    if not quantize:
      return norm_weights
    quant_max = tf.constant(1.0, dtype=tf.float32)
    quant_min = tf.constant(-1.0, dtype=tf.float32)
    quant_weights = self.delayed_quant(
        norm_weights,
        quant_min,
        quant_max,
        per_channel=False,
        num_bits=num_bits,
        narrow_range=True,
        quant_delay=quant_delay)
    return quant_weights

  def last_value_quantize(self,
                          inputs,
                          per_channel=False,
                          init_min=-6.0,
                          init_max=6.0,
                          name_prefix='FixedValueQuant',
                          reuse=None,
                          is_training=False,
                          num_bits=8,
                          narrow_range=False,
                          relative_quantile=0,
                          freeze=False,
                          quant_delay=False):
    """Adds a layer that collects quantization ranges as last input ranges.

    LastValueQuantize creates variables called 'min' and 'max', representing the
    interval used for quantization and clamping.

    Args:
      inputs: a tensor containing values to be quantized.
      per_channel: (Optional) a boolean specifying whether to use different
        quantization ranges per output channel.
      init_min: a float scalar, the initial value for variable min.
      init_max: a float scalar, the initial value for variable max.
      name_prefix: name_prefix for created nodes.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      is_training: Whether the op is applied to a training or eval graph.
      num_bits: Number of bits to use for quantization, must be between 2 and 8.
      narrow_range: Whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
      relative_quantile: Specify the location of quantization min and max
        parameters. relative_quantile = 0 is equivalent to using min and max
        of input; relative_quantile = 1 set min and max the optimal location
        assuming the input distribution is uniform. In reality, a good value
        should be in the range [0 1].
      freeze: If True, the min and max variables are calculated once at the
        begining of training and then freeze. This is used for quantized
        fine-tuning of a pretrained checkpoint. If False, the min and max are
        calculated and updated every cycle.
      quant_delay: The number of global steps after which the fake quantization
        are turned on. Used for performing fine-tuning experiment without
        starting from a pre-trained checkpoint.
    Returns:
      a tensor containing quantized values.
    """

    with tf.variable_scope(
        None, default_name=name_prefix, values=[inputs], reuse=reuse) as scope:
      scope.set_partitioner(None)
      input_shape = inputs.get_shape()
      input_dim = len(input_shape)
      if per_channel:
        # Only support quantizing 1-, 2- and 4-dimensional tensors.
        assert input_dim in [1, 2, 4]
        min_max_shape = [input_shape[-1]]
      else:
        min_max_shape = []

      min_var = tf.get_variable('min',
                                min_max_shape,
                                tf.float32,
                                initializer=tf.constant_initializer(init_min),
                                trainable=False)
      max_var = tf.get_variable('max',
                                min_max_shape,
                                tf.float32,
                                initializer=tf.constant_initializer(init_max),
                                trainable=False)
      if not is_training:
        return self.delayed_quant(
            inputs,
            min_var,
            max_var,
            per_channel=per_channel,
            num_bits=num_bits,
            narrow_range=narrow_range,
            quant_delay=None)

      if per_channel:
        if input_dim == 2:
          reduce_dims = [0]
        elif input_dim == 4:
          reduce_dims = [0, 1, 2]

      if num_bits >= 4:
        quantile = 0
      else:
        quantile = (1.0 / 2.0**(num_bits + 1.0)) * relative_quantile * 100

      if per_channel:
        if input_dim >= 2:
          batch_min = tf.contrib.distributions.percentile(
              inputs, q=quantile, axis=reduce_dims, name='BatchMin')
        else:
          batch_min = inputs
      else:
        batch_min = tf.contrib.distributions.percentile(
            inputs, q=quantile, name='BatchMin')

      if per_channel:
        if input_dim >= 2:
          batch_max = tf.contrib.distributions.percentile(
              inputs, q=100 - quantile, axis=reduce_dims, name='BatchMax')
        else:
          batch_max = inputs
      else:
        batch_max = tf.contrib.distributions.percentile(
            inputs, q=100 - quantile, name='BatchMax')

      if narrow_range:
        multiplier = 1.0
      else:
        multiplier = 1.0 + 1.0 / (2.0**(num_bits-1.0) - 1.0)

      batch_abs_max = tf.maximum(tf.abs(batch_min), tf.abs(batch_max))

      if narrow_range:
        batch_adjusted_min = 0 - batch_abs_max
      else:
        multiplier = 1.0 + 1.0 / (2.0**(num_bits-1.0) - 1.0)
        batch_adjusted_min = 0 - tf.scalar_mul(multiplier, batch_abs_max)

      batch_abs_max = tf.cast(batch_abs_max, tf.float32)
      batch_adjusted_min = tf.cast(batch_adjusted_min, tf.float32)

      if freeze:
        def make_var_op(var):
          def f():
            return var
          return f

        quant_step = common.CreateOrGetQuantizationStep()
        min_max_assign = tf.less_equal(
            quant_step, 1, name='MinMaxAssign')
        min_value = tf.cond(min_max_assign,
                            make_var_op(batch_adjusted_min),
                            make_var_op(min_var),
                            name='AssignMinCond')
        max_value = tf.cond(min_max_assign,
                            make_var_op(batch_abs_max),
                            make_var_op(max_var),
                            name='AssignMaxCond')
      else:
        min_value = batch_adjusted_min
        max_value = batch_abs_max

      assign_min = tf.assign(min_var, min_value)
      assign_max = tf.assign(max_var, max_value)

      return self.delayed_quant(
          inputs,
          assign_min,
          assign_max,
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range,
          quant_delay=quant_delay)

  def _conv2d_impl(self, input_layer, num_channels_in, filters, kernel_size,
                   strides, padding, kernel_initializer):
    """Construct a custom convolution layer."""
    if self.use_tf_layers:
      assert not (self.params.tanh_weight_transform or self.params.quant_weight)
      return conv_layers.conv2d(input_layer, filters, kernel_size, strides,
                                padding, self.channel_pos,
                                kernel_initializer=kernel_initializer,
                                use_bias=False)
    else:
      weights_shape = [kernel_size[0], kernel_size[1], num_channels_in, filters]
      # We use the name 'conv2d/kernel' so the variable has the same name as its
      # tf.layers equivalent. This way, if a checkpoint is written when
      # self.use_tf_layers == True, it can be loaded when
      # self.use_tf_layers == False, and vice versa.
      weights = self.get_variable('conv2d/kernel', weights_shape,
                                  self.variable_dtype, self.dtype,
                                  initializer=kernel_initializer)
      if self.params.tanh_weight_transform:
        if not (self.params.first_weight_name in weights.name
                or self.params.last_weight_name in weights.name):
          print('Dorefa quantizing weight %s' % weights.name)
          weights = self.dorefa_weight_quantize(
              weights,
              self.params.quant_weight,
              self.params.quant_weight_bits,
              self.params.quant_weight_per_channel,
              self.params.quant_weight_delay)
      elif self.params.quant_weight:
        if not (self.params.first_weight_name in weights.name
                or self.params.last_weight_name in weights.name):
          print('Quantizing weight %s' % weights.name)
          weights = self.last_value_quantize(
              weights,
              per_channel=self.params.quant_weight_per_channel,
              is_training=self.phase_train,
              num_bits=self.params.quant_weight_bits,
              narrow_range=self.params.quant_weight_narrow_range,
              relative_quantile=self.params.quant_weight_relative_quantile,
              freeze=self.params.freeze_weight_range,
              quant_delay=self.params.quant_weight_delay)

      if self.data_format == 'NHWC':
        strides = [1] + strides + [1]
      else:
        strides = [1, 1] + strides
      return tf.nn.conv2d(input_layer, weights, strides, padding,
                          data_format=self.data_format)

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           use_batch_norm=None,
           stddev=None,
           activation='relu',
           bias=0.0,
           kernel_initializer=None):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    if stddev is not None and kernel_initializer is None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    if kernel_initializer is None:
      kernel_initializer = tf.variance_scaling_initializer()
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = self._conv2d_impl(input_layer, num_channels_in, num_out_channels,
                                 kernel_size=[k_height, k_width],
                                 strides=[d_height, d_width], padding=mode,
                                 kernel_initializer=kernel_initializer)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = self._conv2d_impl(input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='SAME',
                                   kernel_initializer=kernel_initializer)
        else:
          rate = 1  # Unused (for 'a trous' convolutions)
          kernel_height_effective = k_height + (k_height - 1) * (rate - 1)
          pad_h_beg = (kernel_height_effective - 1) // 2
          pad_h_end = kernel_height_effective - 1 - pad_h_beg
          kernel_width_effective = k_width + (k_width - 1) * (rate - 1)
          pad_w_beg = (kernel_width_effective - 1) // 2
          pad_w_end = kernel_width_effective - 1 - pad_w_beg
          padding = [[0, 0], [pad_h_beg, pad_h_end],
                     [pad_w_beg, pad_w_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          padded_input_layer = tf.pad(input_layer, padding)
          conv = self._conv2d_impl(padded_input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='VALID',
                                   kernel_initializer=kernel_initializer)
      if use_batch_norm is None:
        use_batch_norm = self.use_batch_norm
      mlperf.logger.log_conv2d(input_tensor=input_layer, output_tensor=conv,
                               stride_height=d_height, stride_width=d_width,
                               filters=num_out_channels,
                               initializer=kernel_initializer,
                               use_bias=not use_batch_norm and bias is not None)
      if not use_batch_norm:
        if bias is not None:
          biases = self.get_variable('biases', [num_out_channels],
                                     self.variable_dtype, self.dtype,
                                     initializer=tf.constant_initializer(bias))
          biased = tf.reshape(
              tf.nn.bias_add(conv, biases, data_format=self.data_format),
              conv.get_shape())
        else:
          biased = conv
      else:
        self.top_layer = conv
        self.top_size = num_out_channels
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
        conv1 = self.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = num_out_channels
      return conv1

  def _pool(self,
            pool_name,
            pool_function,
            k_height,
            k_width,
            d_height,
            d_width,
            mode,
            input_layer,
            num_channels_in):
    """Construct a pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = pool_name + str(self.counts[pool_name])
    self.counts[pool_name] += 1
    if self.use_tf_layers:
      pool = pool_function(
          input_layer, [k_height, k_width], [d_height, d_width],
          padding=mode,
          data_format=self.channel_pos,
          name=name)
    else:
      if self.data_format == 'NHWC':
        ksize = [1, k_height, k_width, 1]
        strides = [1, d_height, d_width, 1]
      else:
        ksize = [1, 1, k_height, k_width]
        strides = [1, 1, d_height, d_width]
      pool = tf.nn.max_pool(input_layer, ksize, strides, padding=mode,
                            data_format=self.data_format, name=name)
    if pool_name == 'mpool':
      mlperf.logger.log_max_pool(input_tensor=input_layer,
                                 output_tensor=pool)
    self.top_layer = pool
    return pool

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    return self._pool('mpool', pooling_layers.max_pooling2d, k_height, k_width,
                      d_height, d_width, mode, input_layer, num_channels_in)

  def apool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct an average pooling layer."""
    return self._pool('apool', pooling_layers.average_pooling2d, k_height,
                      k_width, d_height, d_width, mode, input_layer,
                      num_channels_in)

  def reshape(self, shape, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    self.top_layer = tf.reshape(input_layer, shape)
    self.top_size = shape[-1]  # HACK This may not always work
    return self.top_layer

  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             bias=0.0,
             stddev=None,
             activation='relu'):
    """Add an affine transformation layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      stddev = stddev or np.sqrt(init_factor / num_channels_in)
      kernel = self.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.variable_dtype, self.dtype,
          initializer=tf.truncated_normal_initializer(stddev=stddev))

      if self.params.tanh_weight_transform:
        if not (self.params.first_weight_name in kernel.name
                or self.params.last_weight_name in kernel.name):
          print('Dorefa quantizing weight %s' % kernel.name)
          kernel = self.dorefa_weight_quantize(
              kernel,
              self.params.quant_weight,
              self.params.quant_weight_bits,
              self.params.quant_weight_per_channel,
              self.params.quant_weight_delay)
      elif self.params.quant_weight:
        if not (self.params.first_weight_name in kernel.name
                or self.params.last_weight_name in kernel.name):
          print('Quantizing weight %s' % kernel.name)
          kernel = self.last_value_quantize(
              kernel,
              per_channel=self.params.quant_weight_per_channel,
              is_training=self.phase_train,
              num_bits=self.params.quant_weight_bits,
              narrow_range=self.params.quant_weight_narrow_range,
              relative_quantile=self.params.quant_weight_relative_quantile,
              freeze=self.params.freeze_weight_range,
              quant_delay=self.params.quant_weight_delay)

      biases = self.get_variable('biases', [num_out_channels],
                                 self.variable_dtype, self.dtype,
                                 initializer=tf.constant_initializer(bias))
      mlperf.logger.log(key=mlperf.tags.MODEL_HP_DENSE,
                        value=num_out_channels)
      logits = tf.nn.xw_plus_b(input_layer, kernel, biases)
      if activation == 'relu':
        mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
        affine1 = self.relu(logits)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

  def inception_module(self, name, cols, input_layer=None, in_size=None):
    """Add an inception module on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name += str(self.counts[name])
    self.counts[name] += 1
    with tf.variable_scope(name):
      col_layers = []
      col_layer_sizes = []
      for c, col in enumerate(cols):
        col_layers.append([])
        col_layer_sizes.append([])
        for l, layer in enumerate(col):
          ltype, args = layer[0], layer[1:]
          if l == 0:
            kwargs = {
                'input_layer': input_layer,
                'num_channels_in': in_size
            }
          else:
            kwargs = {}
          if ltype == 'conv':
            self.conv(*args, **kwargs)
          elif ltype == 'mpool':
            self.mpool(*args, **kwargs)
          elif ltype == 'apool':
            self.apool(*args, **kwargs)
          elif ltype == 'share':  # Share matching layer from previous column
            self.top_layer = col_layers[c - 1][l]
            self.top_size = col_layer_sizes[c - 1][l]
          else:
            raise KeyError(
                'Invalid layer type for inception module: \'%s\'' % ltype)
          col_layers[c].append(self.top_layer)
          col_layer_sizes[c].append(self.top_size)
      catdim = 3 if self.data_format == 'NHWC' else 1
      self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
      self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
      return self.top_layer

  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keepdims=keep_dims, name=name)
    return self.top_layer

  def dropout(self, keep_prob=0.5, input_layer=None):
    """Add a dropout layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'dropout' + str(self.counts['dropout'])
    with tf.variable_scope(name):
      if not self.phase_train:
        keep_prob = 1.0
      if self.use_tf_layers:
        dropout = core_layers.dropout(input_layer, 1. - keep_prob,
                                      training=self.phase_train)
      else:
        dropout = tf.nn.dropout(input_layer, keep_prob)
      self.top_layer = dropout
      return dropout

  def _batch_norm_without_layers(self, input_layer, decay, use_scale, epsilon):
    """Batch normalization on `input_layer` without tf.layers."""
    # We make this function as similar as possible to the
    # tf.contrib.layers.batch_norm, to minimize the differences between using
    # layers and not using layers.
    shape = input_layer.shape
    num_channels = shape[3] if self.data_format == 'NHWC' else shape[1]
    beta = self.get_variable('beta', [num_channels], tf.float32, tf.float32,
                             initializer=tf.zeros_initializer())
    if use_scale:
      gamma = self.get_variable('gamma', [num_channels], tf.float32,
                                tf.float32, initializer=tf.ones_initializer())
    else:
      gamma = tf.constant(1.0, tf.float32, [num_channels])
    # For moving variables, we use tf.get_variable instead of self.get_variable,
    # since self.get_variable returns the result of tf.cast which we cannot
    # assign to.
    moving_mean = tf.get_variable('moving_mean', [num_channels],
                                  tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False)
    moving_variance = tf.get_variable('moving_variance', [num_channels],
                                      tf.float32,
                                      initializer=tf.ones_initializer(),
                                      trainable=False)
    if self.phase_train:
      bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(
          input_layer, gamma, beta, epsilon=epsilon,
          data_format=self.data_format, is_training=True)
      mean_update = moving_averages.assign_moving_average(
          moving_mean, batch_mean, decay=decay, zero_debias=False)
      variance_update = moving_averages.assign_moving_average(
          moving_variance, batch_variance, decay=decay, zero_debias=False)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
    else:
      bn, _, _ = tf.nn.fused_batch_norm(
          input_layer, gamma, beta, mean=moving_mean,
          variance=moving_variance, epsilon=epsilon,
          data_format=self.data_format, is_training=False)
    return bn

  def batch_norm(self, input_layer=None, decay=0.999, scale=False,
                 epsilon=0.001):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    center = True
    with tf.variable_scope(name) as scope:
      if self.use_tf_layers:
        bn = tf.contrib.layers.batch_norm(
            input_layer,
            decay=decay,
            scale=scale,
            epsilon=epsilon,
            is_training=self.phase_train,
            fused=True,
            data_format=self.data_format,
            scope=scope,
            center=center)
      else:
        bn = self._batch_norm_without_layers(input_layer, decay, scale, epsilon)
    self.top_layer = bn
    self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
    self.top_size = int(self.top_size)
    mlperf.logger.log_batch_norm(
        input_tensor=input_layer, output_tensor=bn, momentum=decay,
        epsilon=epsilon, center=center, scale=scale, training=self.phase_train)
    return bn

  def lrn(self, depth_radius, bias, alpha, beta):
    """Adds a local response normalization layer."""
    name = 'lrn' + str(self.counts['lrn'])
    self.counts['lrn'] += 1
    self.top_layer = tf.nn.lrn(
        self.top_layer, depth_radius, bias, alpha, beta, name=name)
    return self.top_layer
