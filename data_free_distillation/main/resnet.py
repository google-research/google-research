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
"""Defines Residual Networks with basic blocks.

The networks implemented in this module is for training on CIFAR-10 dataset.
However for the purpose of reproducing the results in [2], the network structure
is different from that proposed in [1] for CIFAR-10. The original version [1]
has down sampling for three times, while in [2] there are four times which is
similar to the networks for ImageNet.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Hongxu Yin et al.
    Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion.
    arXiv: 1912.08795
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Optional, Tuple

import tensorflow.compat.v1 as tf
import tf_slim as slim
import tf_slim.nets as nets

utils = nets.resnet_v1.utils
resnet_utils = nets.resnet_utils

# Alias conv2d.
conv2d = slim.conv2d


class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def build_basic_residual_path(inputs, depth,
                              stride):
  """Builds the basic block residual path for ResNet.

  Args:
    inputs: A `Tensor` of size `[batch, height, width, channels]`.
    depth: The number of output filters for the two convolutions.
    stride: The ResNet unit's stride. Determines the amount of down sampling of
      the unit's output compared to its input.

  Returns:
    The residual pathway to be used in a ResNet basic block.
  """
  residual = conv2d(inputs, depth, kernel_size=3, stride=stride, scope='conv1')
  residual = conv2d(
      residual,
      depth,
      kernel_size=3,
      stride=1,
      activation_fn=None,
      scope='conv2')

  return residual


def build_bottleneck_residual_path(inputs, depth,
                                   depth_bottleneck,
                                   stride):
  """Builds the bottleneck residual path for ResNet.

  Args:
    inputs: A `Tensor` of size `[batch, height, width, channels]`.
    depth: The number of output filters for the two convolutions.
    depth_bottleneck: The number of output filters for the bottleneck.
    stride: The ResNet unit's stride. Determines the amount of down sampling of
      the unit's output compared to its input.

  Returns:
    The residual pathway to be used in a ResNet bottleneck.
  """
  residual = conv2d(
      inputs, depth_bottleneck, kernel_size=1, stride=1, scope='conv1')
  residual = resnet_utils.conv2d_same(
      residual, depth_bottleneck, kernel_size=3, stride=stride, scope='conv2')
  residual = conv2d(
      residual,
      depth,
      kernel_size=1,
      stride=1,
      activation_fn=None,
      scope='conv3')

  return residual


@slim.add_arg_scope
def basic_block(inputs,
                depth,
                stride,
                rate = None,
                outputs_collections=None,
                scope=None):
  """Basic block residual unit with BN after convolutions.

  This is the standard building block proposed in [1].

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the first unit of the second block.

  Args:
    inputs: A `Tensor` of size `[batch, height, width, channels]`.
    depth: The number of output filters for the two convolutions.
    stride: The ResNet unit's stride. Determines the amount of down sampling of
      the unit's output compared to its input.
    rate: Unused. To be compatible with `resnet_utils.stack_blocks_dense`.
    outputs_collections: Collections to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  del rate  # Unused.

  with tf.variable_scope(scope, 'basic_block', [inputs]) as sc:
    residual = build_basic_residual_path(inputs, depth, stride)

    depth_in = utils.last_dimension(inputs.shape, min_rank=4)
    depth_residual = utils.last_dimension(residual.shape, min_rank=4)

    if stride != 1 or depth_in != depth_residual:
      shortcut = conv2d(
          inputs,
          depth,
          kernel_size=1,
          stride=stride,
          activation_fn=None,
          scope='shortcut')
    else:
      shortcut = inputs

    output = tf.nn.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate = None,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit with BN after convolutions.

  This is the bottleneck building block proposed in [1].

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the first unit of the second block.

  Args:
    inputs: A `Tensor` of size `[batch, height, width, channels]`.
    depth: The number of output filters for the last convolutions.
    depth_bottleneck: The number of output filters for the bottleneck.
    stride: The ResNet unit's stride. Determines the amount of down sampling of
      the unit's output compared to its input.
    rate: Unused. To be compatible with `resnet_utils.stack_blocks_dense`.
    outputs_collections: Collections to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  del rate  # Unused.

  with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
    residual = build_bottleneck_residual_path(inputs, depth, depth_bottleneck,
                                              stride)

    depth_in = utils.last_dimension(inputs.shape, min_rank=4)
    depth_residual = utils.last_dimension(residual.shape, min_rank=4)

    if stride != 1 or depth_in != depth_residual:
      shortcut = conv2d(
          inputs,
          depth,
          kernel_size=1,
          stride=stride,
          activation_fn=None,
          scope='shortcut')
    else:
      shortcut = inputs

    output = tf.nn.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def _use_small_root_block(inputs):
  """Whether to use small root block in ResNet.

  [2] uses different root block in their ResNet structure for CIFAR-10 and
  ImageNet. For 32x32 CIFAR inputs, a 3x3 convolutional layer is employed, while
  for ImageNet, a 7x7 convolutional layer followed by a 3x3 max pooling layer is
  used, same as in [1].

  This function checks whether to use the small root block (3x3 conv) according
  to the image size of the inputs.

  Args:
    inputs: The inputs passed to ResNet.

  Returns:
    Whether to use small root block in ResNet.
  """
  dims = inputs.shape.dims
  size = dims[1]  # dims is like [B, H, W, C]
  return size <= 64


def _skip_first_max_pooling(inputs):
  """Whether to skip the first max pooling layer in ResNet.

  For 128x128 inputs, we skip the first 3x3 2-stride max pooling layer.

  Args:
    inputs: The inputs passed to ResNet.

  Returns:
    Whether to skip the first max pooling layer in ResNet.
  """
  dims = inputs.shape.dims
  size = dims[1]  # dims is like [B, H, W, C]
  return size == 128


def resnet(inputs,
           blocks,
           num_classes = None,
           is_training = True,
           global_pool = True,
           include_root_block = True,
           conv1_depth = 64,
           reuse=None,
           scope=None):
  """Generator for ResNet models.

  Args:
    inputs: A `Tensor` of size `[batch, height_in, width_in, channels_in]`.
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a `resnet_utils.Block` object describing the units in the block.
    num_classes: The number of predicted classes for classification tasks.
    is_training: Whether batch norm layers are in training mode.
    global_pool: If True, performs global average pooling before computing the
      logits.
    include_root_block: If True, includes the initial convolution, otherwise
      excludes it.
    conv1_depth: The number of filters of the first convolutional layer.
    reuse: Whether the network and its variables should be reused. To be able to
      reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 `Tensor` of size `[batch, height_out, width_out,
      channels_out]`.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  with tf.variable_scope(scope, 'resnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points'
    # noinspection PyCallingNonCallable
    with slim.arg_scope(
        [conv2d, basic_block, bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      # noinspection PyCallingNonCallable
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
            if is_training is not None else NoOpScope()):
        net = inputs
        if include_root_block:
          if _use_small_root_block(inputs):
            net = resnet_utils.conv2d_same(
                net, conv1_depth, kernel_size=3, stride=1, scope='conv1')
          else:
            net = resnet_utils.conv2d_same(
                net, conv1_depth, kernel_size=7, stride=2, scope='conv1')
            if not _skip_first_max_pooling(inputs):
              net = slim.max_pool2d(
                  net, kernel_size=3, stride=2, padding='SAME', scope='pool1')

        net = resnet_utils.stack_blocks_dense(net, blocks)

        end_points = utils.convert_collection_to_dict(
            end_points_collection, clear_collection=True)
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, axis=[1, 2], name='pool5', keepdims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = conv2d(
              net,
              num_classes,
              kernel_size=1,
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
          end_points[sc.name + '/logits'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def resnet_block(scope,
                 depth,
                 num_units,
                 stride,
                 standard = True):
  """Helper function for creating a resnet block.

  Args:
    scope: The scope of the block.
    depth: The depth of each residual block for standard one; or the depth of
      each bottleneck.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the first unit,
      while all other units have stride=1.
    standard: If true, use standard basic block; else bottleneck.

  Returns:
    A resnet block.
  """
  if standard:
    return resnet_utils.Block(scope, basic_block, [{
        'depth': depth,
        'stride': stride,
    }] + (num_units - 1) * [{
        'depth': depth,
        'stride': 1,
    }])
  else:
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': depth * 4,
        'depth_bottleneck': depth,
        'stride': stride,
    }] + (num_units - 1) * [{
        'depth': depth * 4,
        'depth_bottleneck': depth,
        'stride': 1,
    }])


def _create_blocks(num_units_list,
                   skip_first_n_strides,
                   standard = True):
  """Creates block definitions."""
  # skip_first_n_strides is designed for low-resolution ImageNet inputs.
  # Reference: https://arxiv.org/pdf/1909.03205.pdf
  # ----------------------------------------------------------
  # skip_first_n_strides | stride scheme | suitable input size
  #                    0 |  [1, 2, 2, 2] |             56 | 64
  #                    1 |  [1, 1, 2, 2] |             28 | 32
  #                    2 |  [1, 1, 1, 2] |             14 | 16
  #                    3 |  [1, 1, 1, 1] |              7 |  8
  # ----------------------------------------------------------
  assert 0 <= skip_first_n_strides <= 3
  stride_fn = lambda i: 1 if i <= skip_first_n_strides else 2

  depths = [64, 128, 256, 512]
  blocks = [
      resnet_block(  # pylint: disable=g-complex-comprehension
          f'block{i + 1}',
          depth=depth,
          num_units=num_units,
          stride=stride_fn(i),
          standard=standard,
      ) for i, (depth, num_units) in enumerate(zip(depths, num_units_list))
  ]
  return blocks


def resnet_18(inputs,
              num_classes = None,
              is_training = True,
              global_pool = True,
              weight_decay = 5e-4,
              batch_norm_decay = 0.9,
              skip_first_n_strides = 0,
              reuse=None,
              scope='resnet_18'):
  """ResNet-18 model of [2]. See resnet() for arg and return description."""
  blocks = _create_blocks([2, 2, 2, 2], skip_first_n_strides)
  # noinspection PyCallingNonCallable
  with slim.arg_scope(
      resnet_utils.resnet_arg_scope(
          weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
    return resnet(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        conv1_depth=64,
        reuse=reuse,
        scope=scope)


def resnet_34(inputs,
              num_classes = None,
              is_training = True,
              global_pool = True,
              weight_decay = 5e-4,
              batch_norm_decay = 0.9,
              skip_first_n_strides = 0,
              reuse=None,
              scope='resnet_34'):
  """ResNet-34 model of [2]. See resnet() for arg and return description."""
  blocks = _create_blocks([3, 4, 6, 3], skip_first_n_strides)
  # noinspection PyCallingNonCallable
  with slim.arg_scope(
      resnet_utils.resnet_arg_scope(
          weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
    return resnet(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        conv1_depth=64,
        reuse=reuse,
        scope=scope)


def resnet_50(inputs,
              num_classes = None,
              is_training = True,
              global_pool = True,
              weight_decay = 5e-4,
              batch_norm_decay = 0.9,
              skip_first_n_strides = 0,
              reuse=None,
              scope='resnet_50'):
  """ResNet-34 model of [2]. See resnet() for arg and return description."""
  blocks = _create_blocks([3, 4, 6, 3], skip_first_n_strides, standard=False)
  # noinspection PyCallingNonCallable
  with slim.arg_scope(
      resnet_utils.resnet_arg_scope(
          weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
    return resnet(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        conv1_depth=64,
        reuse=reuse,
        scope=scope)


def resnet_101(inputs,
               num_classes = None,
               is_training = True,
               global_pool = True,
               weight_decay = 5e-4,
               batch_norm_decay = 0.9,
               skip_first_n_strides = 0,
               reuse=None,
               scope='resnet_101'):
  """ResNet-101 model of [2]. See resnet() for arg and return description."""
  blocks = _create_blocks([3, 4, 23, 3], skip_first_n_strides, standard=False)
  # noinspection PyCallingNonCallable
  with slim.arg_scope(
      resnet_utils.resnet_arg_scope(
          weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
    return resnet(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        conv1_depth=64,
        reuse=reuse,
        scope=scope)


def model_fn(architecture):
  """Returns the model function corresponding to the given architecture."""
  model_fns = {
      'resnet_18': resnet_18,
      'resnet_34': resnet_34,
      'resnet_50': resnet_50,
      'resnet_101': resnet_101,
  }
  assert architecture in model_fns, 'Unsupported architecture %s' % architecture
  return model_fns[architecture]
