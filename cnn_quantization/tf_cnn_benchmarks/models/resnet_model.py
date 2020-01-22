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

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import datasets
from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks.models import model as model_lib


def bottleneck_block_v1(cnn, depth, depth_bottleneck, stride):
  """Bottleneck block with identity short-cut for ResNet v1.

  Args:
    cnn: the network to append bottleneck blocks.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
  """
  input_layer = cnn.top_layer
  in_size = cnn.top_size
  name_key = 'resnet_v1'
  name = name_key + str(cnn.counts[name_key])
  cnn.counts[name_key] += 1

  with tf.variable_scope(name):
    if depth == in_size:
      if stride == 1:
        shortcut = input_layer
      else:
        shortcut = cnn.apool(
            1, 1, stride, stride, input_layer=input_layer,
            num_channels_in=in_size)
        mlperf.logger.log_projection(input_tensor=input_layer,
                                     output_tensor=shortcut)
    else:
      shortcut = cnn.conv(
          depth, 1, 1, stride, stride, activation=None,
          use_batch_norm=True, input_layer=input_layer,
          num_channels_in=in_size, bias=None)
    cnn.conv(depth_bottleneck, 1, 1, stride, stride,
             input_layer=input_layer, num_channels_in=in_size,
             use_batch_norm=True, bias=None)
    cnn.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET',
             use_batch_norm=True, bias=None)
    res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                   use_batch_norm=True, bias=None)
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_SHORTCUT_ADD)
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
    with tf.variable_scope('sum'):
      output = cnn.relu(shortcut + res)
    cnn.top_layer = output
    cnn.top_size = depth


def bottleneck_block_v1_5(cnn, depth, depth_bottleneck, stride):
  """Bottleneck block with identity short-cut for ResNet v1.5.

  ResNet v1.5 is the informal name for ResNet v1 where stride 2 is used in the
  first 3x3 convolution of each block instead of the first 1x1 convolution.

  First seen at https://github.com/facebook/fb.resnet.torch. Used in the paper
  "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
  (arXiv:1706.02677v2) and by fast.ai to train to accuracy in 45 epochs using
  multiple image sizes.

  Args:
    cnn: the network to append bottleneck blocks.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
  """
  input_layer = cnn.top_layer
  in_size = cnn.top_size
  name_key = 'resnet_v1.5'
  name = name_key + str(cnn.counts[name_key])
  cnn.counts[name_key] += 1

  with tf.variable_scope(name):
    if depth == in_size:
      if stride == 1:
        shortcut = input_layer
      else:
        shortcut = cnn.apool(
            1, 1, stride, stride, input_layer=input_layer,
            num_channels_in=in_size)
        mlperf.logger.log_projection(input_tensor=input_layer,
                                     output_tensor=shortcut)
    else:
      shortcut = cnn.conv(
          depth, 1, 1, stride, stride, activation=None,
          use_batch_norm=True, input_layer=input_layer,
          num_channels_in=in_size, bias=None)
      mlperf.logger.log_projection(input_tensor=input_layer,
                                   output_tensor=shortcut)
    cnn.conv(depth_bottleneck, 1, 1, 1, 1,
             input_layer=input_layer, num_channels_in=in_size,
             use_batch_norm=True, bias=None)
    cnn.conv(depth_bottleneck, 3, 3, stride, stride, mode='SAME_RESNET',
             use_batch_norm=True, bias=None)
    res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                   use_batch_norm=True, bias=None)
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_SHORTCUT_ADD)
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
    with tf.variable_scope('sum'):
      output = cnn.relu(shortcut + res)
    cnn.top_layer = output
    cnn.top_size = depth


def bottleneck_block_v2(cnn, depth, depth_bottleneck, stride):
  """Bottleneck block with identity short-cut for ResNet v2.

  The main difference from v1 is that a batch norm and relu are done at the
  start of the block, instead of the end. This initial batch norm and relu is
  collectively called a pre-activation.

  Args:
    cnn: the network to append bottleneck blocks.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
  """
  input_layer = cnn.top_layer
  in_size = cnn.top_size
  name_key = 'resnet_v2'
  name = name_key + str(cnn.counts[name_key])
  cnn.counts[name_key] += 1

  with tf.variable_scope(name):
    with tf.variable_scope('preact'):
      preact = cnn.batch_norm()
      mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
      preact = cnn.relu(preact)
    if depth == in_size:
      if stride == 1:
        shortcut = input_layer
      else:
        shortcut = cnn.apool(
            1, 1, stride, stride, input_layer=input_layer,
            num_channels_in=in_size)
        mlperf.logger.log_projection(input_tensor=input_layer,
                                     output_tensor=shortcut)
    else:
      shortcut = cnn.conv(
          depth, 1, 1, stride, stride, activation=None, use_batch_norm=False,
          input_layer=preact, num_channels_in=in_size, bias=None)
    cnn.conv(depth_bottleneck, 1, 1, 1, 1,
             input_layer=preact, num_channels_in=in_size,
             use_batch_norm=True, bias=None)
    cnn.conv(depth_bottleneck, 3, 3, stride, stride, mode='SAME_RESNET',
             use_batch_norm=True, bias=None)
    res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                   use_batch_norm=False, bias=None)
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_SHORTCUT_ADD)
    output = shortcut + res
    cnn.top_layer = output
    cnn.top_size = depth


def bottleneck_block(cnn, depth, depth_bottleneck, stride, version):
  """Bottleneck block with identity short-cut.

  Args:
    cnn: the network to append bottleneck blocks.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
    version: version of ResNet to build.
  """
  mlperf.logger.log(key=mlperf.tags.MODEL_HP_BLOCK_TYPE,
                    value=mlperf.tags.BOTTLENECK_BLOCK)
  mlperf.logger.log_begin_block(
      input_tensor=cnn.top_layer, block_type=mlperf.tags.BOTTLENECK_BLOCK)
  if version == 'v2':
    bottleneck_block_v2(cnn, depth, depth_bottleneck, stride)
  elif version == 'v1.5':
    bottleneck_block_v1_5(cnn, depth, depth_bottleneck, stride)
  else:
    bottleneck_block_v1(cnn, depth, depth_bottleneck, stride)
  mlperf.logger.log_end_block(output_tensor=cnn.top_layer)


def residual_block(cnn, depth, stride, version, projection_shortcut=False):
  """Residual block with identity short-cut.

  Args:
    cnn: the network to append residual blocks.
    depth: the number of output filters for this residual block.
    stride: Stride used in the first layer of the residual block.
    version: version of ResNet to build.
    projection_shortcut: indicator of using projection shortcut, even if top
      size and depth are equal
  """
  pre_activation = True if version == 'v2' else False
  input_layer = cnn.top_layer
  in_size = cnn.top_size
  name_key = 'resnet_v2'
  name = name_key + str(cnn.counts[name_key])
  cnn.counts[name_key] += 1

  with tf.variable_scope(name):
    if projection_shortcut:
      shortcut = cnn.conv(
          depth, 1, 1, stride, stride, activation=None,
          use_batch_norm=True, input_layer=input_layer,
          num_channels_in=in_size, bias=None)
    elif in_size != depth:
      # Plan A of shortcut.
      shortcut = cnn.apool(1, 1, stride, stride,
                           input_layer=input_layer,
                           num_channels_in=in_size)
      padding = (depth - in_size) // 2
      if cnn.channel_pos == 'channels_last':
        shortcut = tf.pad(
            shortcut, [[0, 0], [0, 0], [0, 0], [padding, padding]])
      else:
        shortcut = tf.pad(
            shortcut, [[0, 0], [padding, padding], [0, 0], [0, 0]])
    else:
      shortcut = input_layer
    if pre_activation:
      res = cnn.batch_norm(input_layer)
      res = cnn.relu(res)
    else:
      res = input_layer
    cnn.conv(depth, 3, 3, stride, stride,
             input_layer=res, num_channels_in=in_size,
             use_batch_norm=True, bias=None)
    if pre_activation:
      res = cnn.conv(depth, 3, 3, 1, 1, activation=None,
                     use_batch_norm=False, bias=None)
      output = shortcut + res
    else:
      res = cnn.conv(depth, 3, 3, 1, 1, activation=None,
                     use_batch_norm=True, bias=None)
      cnn.relu(shortcut + res)
    cnn.top_layer = output
    cnn.top_size = depth


class ResnetModel(model_lib.CNNModel):
  """Resnet cnn network configuration."""

  def __init__(self, model, layer_counts, params=None):
    default_batch_sizes = {
        'resnet50': 64,
        'resnet101': 32,
        'resnet152': 32,
        'resnet50_v1.5': 64,
        'resnet101_v1.5': 32,
        'resnet152_v1.5': 32,
        'resnet50_v2': 64,
        'resnet101_v2': 32,
        'resnet152_v2': 32,
    }
    batch_size = default_batch_sizes.get(model, 32)
    # The ResNet paper uses a starting lr of .1 at bs=256.
    self.base_lr_batch_size = 256
    base_lr = 0.128
    if params and params.resnet_base_lr:
      base_lr = params.resnet_base_lr

    super(ResnetModel, self).__init__(model, 224, batch_size, base_lr,
                                      layer_counts, params=params)
    if 'v2' in model:
      self.version = 'v2'
    elif 'v1.5' in model:
      self.version = 'v1.5'
    else:
      self.version = 'v1'

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())
    # Drop batch size from shape logging.
    mlperf.logger.log(key=mlperf.tags.MODEL_HP_INITIAL_SHAPE,
                      value=cnn.top_layer.shape.as_list()[1:])
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
    if self.version == 'v2':
      cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET',
               use_batch_norm=False, activation=None, bias=None)
    else:
      cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    for _ in xrange(self.layer_counts[0]):
      bottleneck_block(cnn, 256, 64, 1, self.version)
    for i in xrange(self.layer_counts[1]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 512, 128, stride, self.version)
    for i in xrange(self.layer_counts[2]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 1024, 256, stride, self.version)
    for i in xrange(self.layer_counts[3]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 2048, 512, stride, self.version)
    if self.version == 'v2':
      with tf.variable_scope('final'):
        cnn.batch_norm()
        cnn.top_layer = cnn.relu(cnn.top_layer)
    cnn.spatial_mean()

  def get_learning_rate(self, global_step, batch_size):
    rescaled_lr = self.get_scaled_base_learning_rate(batch_size)
    num_batches_per_epoch = (
        datasets.IMAGENET_NUM_TRAIN_IMAGES / batch_size)
    boundaries = [int(num_batches_per_epoch * x) for x in [30, 60, 85, 95]]
    values = [1, 0.1, 0.01, 0.001, 0.0001]
    values = [rescaled_lr * v for v in values]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    warmup_steps = int(num_batches_per_epoch * 5)
    mlperf.logger.log(key=mlperf.tags.OPT_LR_WARMUP_STEPS, value=warmup_steps)
    warmup_lr = (
        rescaled_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  def get_scaled_base_learning_rate(self, batch_size):
    """Calculates base learning rate for creating lr schedule.

    In replicated mode, gradients are summed rather than averaged which, with
    the sgd and momentum optimizers, increases the effective learning rate by
    lr * num_gpus. Dividing the base lr by num_gpus negates the increase.

    Args:
      batch_size: Total batch-size.

    Returns:
      Base learning rate to use to create lr schedule.
    """
    base_lr = self.learning_rate
    if self.params.variable_update == 'replicated':
      base_lr = self.learning_rate / self.params.num_gpus
    scaled_lr = base_lr * (batch_size / self.base_lr_batch_size)
    return scaled_lr


def create_resnet50_model(params):
  return ResnetModel('resnet50', (3, 4, 6, 3), params=params)


def create_resnet50_v1_5_model(params):
  return ResnetModel('resnet50_v1.5', (3, 4, 6, 3), params=params)


def create_resnet50_v2_model(params):
  return ResnetModel('resnet50_v2', (3, 4, 6, 3), params=params)


def create_resnet101_model(params):
  return ResnetModel('resnet101', (3, 4, 23, 3), params=params)


def create_resnet101_v2_model(params):
  return ResnetModel('resnet101_v2', (3, 4, 23, 3), params=params)


def create_resnet152_model(params):
  return ResnetModel('resnet152', (3, 8, 36, 3), params=params)


def create_resnet152_v2_model(params):
  return ResnetModel('resnet152_v2', (3, 8, 36, 3), params=params)


class ResnetCifar10Model(model_lib.CNNModel):
  """Resnet cnn network configuration for Cifar 10 dataset.

  V1 model architecture follows the one defined in the paper:
  https://arxiv.org/pdf/1512.03385.pdf.

  V2 model architecture follows the one defined in the paper:
  https://arxiv.org/pdf/1603.05027.pdf.
  """

  def __init__(self, model, layer_counts, params=None):
    if 'v2' in model:
      self.version = 'v2'
    else:
      self.version = 'v1'
    super(ResnetCifar10Model, self).__init__(
        model, 32, 128, 0.1, layer_counts, params=params)

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())

    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}
    if self.version == 'v2':
      cnn.conv(16, 3, 3, 1, 1, use_batch_norm=True)
    else:
      cnn.conv(16, 3, 3, 1, 1, activation=None, use_batch_norm=True)
    for i in xrange(self.layer_counts[0]):
      # reshape to batch_size x 16 x 32 x 32
      residual_block(cnn, 16, 1, self.version)
    for i in xrange(self.layer_counts[1]):
      # Subsampling is performed at the first convolution with a stride of 2
      stride = 2 if i == 0 else 1
      # reshape to batch_size x 32 x 16 x 16
      residual_block(cnn, 32, stride, self.version)
    for i in xrange(self.layer_counts[2]):
      stride = 2 if i == 0 else 1
      # reshape to batch_size x 64 x 8 x 8
      residual_block(cnn, 64, stride, self.version)
    if self.version == 'v2':
      with tf.variable_scope('final'):
        cnn.batch_norm()
        cnn.top_layer = cnn.relu(cnn.top_layer)
    cnn.spatial_mean()

  def get_learning_rate(self, global_step, batch_size):
    num_batches_per_epoch = int(50000 / batch_size)
    boundaries = num_batches_per_epoch * np.array([82, 123, 300],
                                                  dtype=np.int64)
    boundaries = [x for x in boundaries]
    values = [0.1, 0.01, 0.001, 0.0002]
    return tf.train.piecewise_constant(global_step, boundaries, values)


def create_resnet20_cifar_model(params):
  return ResnetCifar10Model('resnet20', (3, 3, 3), params=params)


def create_resnet20_v2_cifar_model(params):
  return ResnetCifar10Model('resnet20_v2', (3, 3, 3), params=params)


def create_resnet32_cifar_model(params):
  return ResnetCifar10Model('resnet32', (5, 5, 5), params=params)


def create_resnet32_v2_cifar_model(params):
  return ResnetCifar10Model('resnet32_v2', (5, 5, 5), params=params)


def create_resnet44_cifar_model(params):
  return ResnetCifar10Model('resnet44', (7, 7, 7), params=params)


def create_resnet44_v2_cifar_model(params):
  return ResnetCifar10Model('resnet44_v2', (7, 7, 7), params=params)


def create_resnet56_cifar_model(params):
  return ResnetCifar10Model('resnet56', (9, 9, 9), params=params)


def create_resnet56_v2_cifar_model(params):
  return ResnetCifar10Model('resnet56_v2', (9, 9, 9), params=params)


def create_resnet110_cifar_model(params):
  return ResnetCifar10Model('resnet110', (18, 18, 18), params=params)


def create_resnet110_v2_cifar_model(params):
  return ResnetCifar10Model('resnet110_v2', (18, 18, 18), params=params)
