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

"""MobileNetV1 model and builder functions. Only for inference."""
import functools
import tensorflow.compat.v1 as tf

from sgk.mbv1 import layers

MOVING_AVERAGE_DECAY = 0.9
EPSILON = 1e-5


def _make_divisible(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def batch_norm_relu(x, fuse_batch_norm=False):
  """Batch normalization + ReLU."""
  if not fuse_batch_norm:
    inputs = tf.layers.batch_normalization(
        inputs=x,
        axis=1,
        momentum=MOVING_AVERAGE_DECAY,
        epsilon=EPSILON,
        center=True,
        scale=True,
        training=False,
        fused=True)
    return tf.nn.relu(inputs)
  return x


def mbv1_block_(inputs, filters, stride, block_id, cfg):
  """Standard building block for mobilenetv1 networks.

  Args:
    inputs:  Input tensor, float32 of size [batch, channels, height, width].
    filters: Int specifying number of filters for the first two convolutions.
    stride: Int specifying the stride. If stride >1, the input is downsampled.
    block_id: which block this is.
    cfg: Configuration for the model.

  Returns:
    The output activation tensor.
  """
  # Setup the depthwise convolution layer.
  depthwise_conv = layers.DepthwiseConv2D(
      kernel_size=3,
      strides=[1, 1, stride, stride],
      padding=[0, 0, 1, 1],
      activation=tf.nn.relu if cfg.fuse_bnbr else None,
      use_bias=cfg.fuse_bnbr,
      name='depthwise_nxn_%s' % block_id)

  # Depthwise convolution, batch norm, relu.
  depthwise_out = batch_norm_relu(depthwise_conv(inputs), cfg.fuse_bnbr)

  # Setup the 1x1 convolution layer.
  out_filters = _make_divisible(
      int(cfg.width * filters), divisor=1 if block_id == 0 else 8)
  end_point = 'contraction_1x1_%s' % block_id

  conv_fn = layers.Conv2D
  if cfg.block_config[block_id] == 'sparse':
    conv_fn = functools.partial(
        layers.SparseConv2D, nonzeros=cfg.block_nonzeros[block_id])

  contraction = conv_fn(
      out_filters,
      activation=tf.nn.relu if cfg.fuse_bnbr else None,
      use_bias=cfg.fuse_bnbr,
      name=end_point)

  # Run the 1x1 convolution followed by batch norm and relu.
  return batch_norm_relu(contraction(depthwise_out), cfg.fuse_bnbr)


def mobilenet_generator(cfg):
  """Generator for mobilenet v2 models.

  Args:
    cfg: Configuration for the model.

  Returns:
    Model `function` that takes in `inputs` and returns the output `Tensor`
    of the model.
  """

  def model(inputs):
    """Creation of the model graph."""
    with tf.variable_scope('mobilenet_model', reuse=tf.AUTO_REUSE):
      # Initial convolutional layer.
      initial_conv_filters = _make_divisible(32 * cfg.width)
      initial_conv = layers.Conv2D(
          filters=initial_conv_filters,
          kernel_size=3,
          stride=2,
          padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
          activation=tf.nn.relu,
          use_bias=True,
          name='initial_conv')
      inputs = batch_norm_relu(initial_conv(inputs), cfg.fuse_bnbr)

      mb_block = functools.partial(mbv1_block_, cfg=cfg)

      # Core MobileNetV1 blocks.
      inputs = mb_block(inputs, filters=64, stride=1, block_id=0)
      inputs = mb_block(inputs, filters=128, stride=2, block_id=1)
      inputs = mb_block(inputs, filters=128, stride=1, block_id=2)
      inputs = mb_block(inputs, filters=256, stride=2, block_id=3)
      inputs = mb_block(inputs, filters=256, stride=1, block_id=4)
      inputs = mb_block(inputs, filters=512, stride=2, block_id=5)
      inputs = mb_block(inputs, filters=512, stride=1, block_id=6)
      inputs = mb_block(inputs, filters=512, stride=1, block_id=7)
      inputs = mb_block(inputs, filters=512, stride=1, block_id=8)
      inputs = mb_block(inputs, filters=512, stride=1, block_id=9)
      inputs = mb_block(inputs, filters=512, stride=1, block_id=10)
      inputs = mb_block(inputs, filters=1024, stride=2, block_id=11)
      inputs = mb_block(inputs, filters=1024, stride=1, block_id=12)

      # Pooling layer.
      inputs = tf.layers.average_pooling2d(
          inputs=inputs,
          pool_size=(inputs.shape[2], inputs.shape[3]),
          strides=1,
          padding='VALID',
          data_format='channels_first',
          name='final_avg_pool')

      # Reshape the output of the pooling layer to 2D for the
      # final fully-connected layer.
      last_block_filters = _make_divisible(int(1024 * cfg.width), 8)
      inputs = tf.reshape(inputs, [-1, last_block_filters])

      # Final fully-connected layer.
      inputs = tf.layers.dense(
          inputs=inputs,
          units=cfg.num_classes,
          activation=None,
          use_bias=True,
          name='final_dense')
    return inputs

  return model


def build_model(features, cfg):
  """Builds the MobileNetV1 model and returns the output logits.

  Args:
    features: Input features tensor for the model.
    cfg: Configuration for the model.

  Returns:
    Computed logits from the model.
  """
  model = mobilenet_generator(cfg)
  return model(features)
