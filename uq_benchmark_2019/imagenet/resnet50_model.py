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

# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.compat.v1.keras import backend
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import regularizers
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def identity_block_base(input_tensor, kernel_size, filters, stage, block,
                        num_updates, dropout_rate=0.,
                        use_variational_layers=False):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      num_updates: integer, total steps in an epoch (for weighting the loss)
      dropout_rate: float, always-on dropout rate.
      use_variational_layers: boolean, if true train a variational model

  Returns:
      x: Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  divergence_fn = lambda q, p, ignore: (tfd.kl_divergence(q, p)/num_updates)
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  if not use_variational_layers:
    first_conv_2d = layers.Conv2D(
        filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(input_tensor, training=True)
      x = first_conv_2d(x)
    else:
      x = first_conv_2d(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Conv2D(filters2, kernel_size, use_bias=False,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
  else:
    x = tfpl.Convolution2DFlipout(
        filters1, kernel_size=(1, 1), padding='SAME',
        name=conv_name_base + '2a',
        kernel_divergence_fn=divergence_fn,
        )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = tfpl.Convolution2DFlipout(
        filters2, kernel_size=kernel_size, padding='SAME',
        activation=None, name=conv_name_base + '2b',
        kernel_divergence_fn=divergence_fn,
        )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = tfpl.Convolution2DFlipout(
        filters3, kernel_size=(1, 1), padding='SAME',
        activation=None, name=conv_name_base + '2c',
        kernel_divergence_fn=divergence_fn,
        )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block_base(input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    num_updates=1,
                    dropout_rate=0.,
                    use_variational_layers=False):
  """A block that has a conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
      num_updates: integer, total steps in an epoch (for weighting the loss)
      dropout_rate: float, always-on dropout rate.
      use_variational_layers: boolean, if true train a variational model

  Returns:
      x: Output tensor for the block.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  divergence_fn = lambda q, p, ignore: (tfd.kl_divergence(q, p)/num_updates)
  if not use_variational_layers:
    conv2d_layer = layers.Conv2D(
        filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(input_tensor, training=True)
      x = conv2d_layer(x)
    else:
      x = conv2d_layer(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    if dropout_rate > 0.:
      x = layers.Dropout(dropout_rate)(x, training=True)
    shortcut = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '1')(
            input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON,
                                         name=bn_name_base + '1')(shortcut)

  else:
    x = tfpl.Convolution2DFlipout(
        filters1, kernel_size=(1, 1), padding='SAME',
        activation=None, name=conv_name_base + '2a',
        kernel_divergence_fn=divergence_fn,
        )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = tfpl.Convolution2DFlipout(
        filters2, kernel_size=kernel_size, strides=strides, padding='SAME',
        activation=None, name=conv_name_base + '2b',
        kernel_divergence_fn=divergence_fn,
        )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = tfpl.Convolution2DFlipout(
        filters3, kernel_size=(1, 1), padding='SAME',
        activation=None, name=conv_name_base + '2c',
        kernel_divergence_fn=divergence_fn,
        )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    shortcut = tfpl.Convolution2DFlipout(
        filters3, kernel_size=(1, 1), strides=strides, padding='SAME',
        activation=None, name=conv_name_base + '1',
        kernel_divergence_fn=divergence_fn,
        )(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON,
                                         name=bn_name_base + '1')(shortcut)
  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  """Posterior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  variable_layer = tfpl.VariableLayer(
      2 * n,
      dtype=dtype,
      initializer=tfpl.BlockwiseInitializer([
          tf.keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
          tf.keras.initializers.Constant(np.log(np.expm1(1e-5)))],
                                            sizes=[n, n]))

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                           reinterpreted_batch_ndims=1)
  distribution_layer = tfpl.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])


def prior_trainable(kernel_size, bias_size=0, dtype=None, num_updates=1):
  """Prior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))

  def regularizer(t):
    out = tfd.LogNormal(0., 1.).log_prob(1e-5 + tf.nn.softplus(c + t[Ellipsis, -1]))
    return -tf.reduce_sum(out) / num_updates

  # Include the prior on the scale parameter as a regularizer in the loss.
  variable_layer = tfpl.VariableLayer(n, dtype=dtype, regularizer=regularizer)

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, -1])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                           reinterpreted_batch_ndims=1)

  distribution_layer = tfpl.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])


# pylint: disable=invalid-name
def ResNet50(method, num_classes, num_updates, dropout_rate):
  """Instantiates the ResNet50 architecture.

  Args:
    method: `str`, method for accounting for uncertainty. Must be one of
      ['vanilla', 'll_dropout', 'll_svi', 'dropout', 'svi', 'dropout_nofirst']
    num_classes: `int` number of classes for image classification.
    num_updates: integer, total steps in an epoch (for weighting the loss)
    dropout_rate: Dropout rate for ll_dropout, dropout methods.

  Returns:
      A Keras model instance.
  pylint: disable=invalid-name
  """

  # Determine proper input shape
  if backend.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
    bn_axis = 1
  else:
    input_shape = (224, 224, 3)
    bn_axis = 3

  if (method in ['dropout', 'll_dropout', 'dropout_nofirst']
     ) != (dropout_rate > 0.):
    raise ValueError(
        'Dropout rate should be nonzero iff a dropout method is used.'
        'Method is {}, dropout is {}.'.format(method, dropout_rate))

  use_variational_layers = method == 'svi'
  hidden_layer_dropout = dropout_rate if method in [
      'dropout', 'dropout_nofirst'] else 0.

  img_input = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  if (dropout_rate > 0.) and (method != 'dropout_nofirst'):
    x = layers.Dropout(hidden_layer_dropout)(x, training=True)
  x = layers.Conv2D(64, (7, 7), use_bias=False,
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name='conv1')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name='bn_conv1')(x)
  x = layers.Activation('relu')(x)
  x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  conv_block = functools.partial(
      conv_block_base,
      num_updates=num_updates,
      dropout_rate=hidden_layer_dropout,
      use_variational_layers=use_variational_layers)
  identity_block = functools.partial(
      identity_block_base,
      num_updates=num_updates,
      dropout_rate=hidden_layer_dropout,
      use_variational_layers=use_variational_layers)

  x = conv_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='a',
      strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

  if dropout_rate > 0.:
    x = layers.Dropout(dropout_rate)(x, training=True)

  if method in ['ll_svi', 'svi']:

    x = tfpl.dense_variational_v2.DenseVariational(
        units=num_classes,
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=functools.partial(
            prior_trainable, num_updates=num_updates),
        use_bias=True,
        kl_weight=1./num_updates,
        kl_use_exact=True,
        name='fc1000'
        )(x)
  else:
    x = layers.Dense(
        num_classes,
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc1000')(x)

  # Create model.
  return models.Model(img_input, x, name='resnet50')
