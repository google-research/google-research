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

"""Model Zoo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v2 as tf

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

from tcc.config import CONFIG

FLAGS = flags.FLAGS
layers = tf.keras.layers


def get_pretrained_ckpt(network):
  """Return path to pretrained ckpt."""
  pretrained_paths = {
      'Resnet50_pretrained': CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS,
  }
  ckpt = pretrained_paths.get(network, None)
  return ckpt


def get_vggm_conv_block(x, conv_layers, use_bn, max_pool_size, name):
  """Conv block."""
  l2_reg_weight = CONFIG.model.l2_reg_weight
  for (channels, kernel_size) in conv_layers:
    x = layers.Conv2D(channels, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg_weight),
                      bias_regularizer=regularizers.l2(l2_reg_weight))(x)
    if use_bn:
      x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
  if max_pool_size > 0:
    x = layers.MaxPooling2D(pool_size=max_pool_size, strides=2,
                            padding='same', name=name)(x)
  else:
    # Identity layer
    x = layers.MaxPooling2D(pool_size=1, strides=1,
                            padding='same', name=name)(x)

  return x


def vggm_net(image_size):
  """VGG-M: VGGM-esque (not exactly VGGM) small network."""
  use_bn = CONFIG.model.vggm.use_bn

  x = layers.Input(shape=(image_size, image_size, 3),
                   dtype='float32')
  inputs = x

  x = layers.ZeroPadding2D(padding=(3, 3))(x)
  x = layers.Conv2D(64, (7, 7),
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv1')(x)
  if use_bn:
    x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.ZeroPadding2D(padding=(1, 1))(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

  conv_blocks = [(128, 3, 2), (256, 3, 2), (512, 3, 0)]

  for i, conv_block in enumerate(conv_blocks):
    channels, filter_size, max_pool_size = conv_block
    x = get_vggm_conv_block(
        x,
        2 * [(channels, filter_size)],
        use_bn,
        max_pool_size,
        name='conv%s' % (i + 2))

  model = Model(inputs=inputs, outputs=x)
  return model


class BaseModel(tf.keras.Model):
  """CNN to extract features from frames.
  """

  def __init__(self, num_steps):
    """Passes frames through base CNNs and return feature.

    Args:
      num_steps: int, Number of steps being passed through CNN.

    Raises:
      ValueError: if invalid network config is passed.
    """
    super(BaseModel, self).__init__()
    layer = CONFIG.MODEL.BASE_MODEL.LAYER
    network = CONFIG.MODEL.BASE_MODEL.NETWORK
    local_ckpt = get_pretrained_ckpt(network)

    if network in ['Resnet50', 'Resnet50_pretrained']:
      base_model = tf.keras.applications.ResNet50(
          include_top=False, weights=local_ckpt, pooling='max')

    elif CONFIG.model.base_model.network == 'VGGM':
      base_model = vggm_net(CONFIG.IMAGE_SIZE)

    else:
      raise ValueError('%s not supported.' % CONFIG.MODEL.BASE_MODEL.NETWORK)

    self.base_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer).output)

    self.num_steps = num_steps

  def call(self, inputs):
    # Reorganize frames such that they can be passed through base_model.
    batch_size, num_steps, h, w, c = inputs.shape
    images = tf.reshape(inputs, [batch_size * num_steps, h, w, c])

    x = self.base_model(images, training=CONFIG.MODEL.TRAIN_BASE != 'frozen')

    _, h, w, c = x.shape
    x = tf.reshape(x, [batch_size, num_steps, h, w, c])
    return x


class ConvEmbedder(tf.keras.Model):
  """Embedder network.
  """

  def __init__(self):
    """Passes convolutional features through embedding network.
    """
    super(ConvEmbedder, self).__init__()

    self.num_steps = CONFIG.DATA.NUM_STEPS

    conv_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS
    fc_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS
    use_bn = CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN
    l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
    embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
    cap_scalar = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR

    conv_params = [(cap_scalar*x[0], x[1], x[2]) for x in conv_params]
    fc_params = [(cap_scalar*x[0], x[1]) for x in fc_params]
    conv_bn_activations = get_conv_bn_layers(conv_params, use_bn, conv_dims=3)
    self.conv_layers = conv_bn_activations[0]
    self.bn_layers = conv_bn_activations[1]
    self.activations = conv_bn_activations[2]

    self.fc_layers = get_fc_layers(fc_params)

    self.embedding_layer = layers.Dense(
        embedding_size,
        kernel_regularizer=regularizers.l2(l2_reg_weight),
        bias_regularizer=regularizers.l2(l2_reg_weight))

  def call(self, x, num_frames):
    base_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE
    fc_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE

    batch_size, total_num_steps, h, w, c = x.shape
    num_context = total_num_steps // num_frames
    x = tf.reshape(x, [batch_size * num_frames, num_context, h, w, c])

    # Dropout on output tensor from base.
    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL:
      x = layers.SpatialDropout3D(base_dropout_rate)(x)
    else:
      x = layers.Dropout(base_dropout_rate)(x)

    # Pass through convolution layers
    for i, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x)
      if CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN:
        bn_layer = self.bn_layers[i]
        x = bn_layer(x)
      if self.activations[i]:
        x = self.activations[i](x)

    # Perform spatial pooling
    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'max_pool':
      x = layers.GlobalMaxPooling3D()(x)
    elif CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'avg_pool':
      x = layers.GlobalAveragePooling3D()(x)
    elif CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD == 'flatten':
      x = layers.Flatten()(x)
    else:
      raise ValueError('Supported flatten methods: max_pool, avg_pool and '
                       'flatten.')

    # Pass through fully connected layers
    for fc_layer in self.fc_layers:
      x = layers.Dropout(fc_dropout_rate)(x)
      x = fc_layer(x)

    x = self.embedding_layer(x)

    if CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE:
      x = tf.nn.l2_normalize(x, axis=-1)

    return x


class ConvGRUEmbedder(tf.keras.Model):
  """Embedder network which uses ConvGRU.
  """

  def __init__(self):
    """Passes convolutional features through embedding network.
    """
    super(ConvGRUEmbedder, self).__init__()

    if CONFIG.data.num_steps != 1:
      raise ValueError('Cannot use GRU with context frames.')

    conv_params = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.CONV_LAYERS
    use_bn = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN
    gru_params = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.GRU_LAYERS

    conv_bn_activations = get_conv_bn_layers(conv_params, use_bn)
    self.conv_layers = conv_bn_activations[0]
    self.bn_layers = conv_bn_activations[1]
    self.activations = conv_bn_activations[2]

    self.gru_layers = get_gru_layers(gru_params)

    dropout_rate = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.DROPOUT_RATE
    self.dropout = layers.Dropout(dropout_rate)

  def call(self, x, num_frames):
    batch_size, num_steps, h, w, c = x.shape
    x = tf.reshape(x, [batch_size * num_steps, h, w, c])
    # Pass through convolution layers
    for i, conv_layer in enumerate(self.conv_layers):
      x = self.dropout(x)
      x = conv_layer(x)
      if CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN:
        bn_layer = self.bn_layers[i]
        x = bn_layer(x)
      if self.activations[i]:
        x = self.activations[i](x)

    # Perform spatial pooling
    x = layers.GlobalMaxPooling2D()(x)

    # Get number of channels after conv layers.
    c = x.shape[-1]
    x = tf.reshape(x, [batch_size, num_steps, c])

    for gru_layer in self.gru_layers:
      x = gru_layer(x)

    # Get number of channels after GRU layers.
    c = x.shape[-1]
    x = tf.reshape(x, [batch_size * num_steps, c])
    return x


class Classifier(tf.keras.Model):
  """Classifier network.
  """

  def __init__(self, fc_layers, dropout_rate):
    """Passes concatenated features through fully connected layers.

    Each layer is preceded by dropout layer.

    Args:
      fc_layers: List, List of tuples of (size, activate). Each tuple represents
        a dully connected layer of size units and ReLU activation if activate is
        True.
      dropout_rate: float, dropout rate.
    """
    super(Classifier, self).__init__()

    self.dropout = layers.Dropout(dropout_rate)
    self.fc_layers = get_fc_layers(fc_layers)

  def call(self, x):
    # Pass through fully connected layers.
    for fc_layer in self.fc_layers:
      x = self.dropout(x)
      x = fc_layer(x)

    return x


def get_model():
  """Returns model dict."""
  model = {}
  num_steps = CONFIG.TRAIN.NUM_FRAMES

  # Keeping the 2 models separate.
  # cnn is per-frame feature extractor.
  # emb is (context frames + frame) feature embedder.
  cnn = BaseModel(num_steps=num_steps)

  if CONFIG.MODEL.EMBEDDER_TYPE == 'conv':
    emb = ConvEmbedder()
  elif CONFIG.MODEL.EMBEDDER_TYPE == 'convgru':
    emb = ConvGRUEmbedder()
  else:
    raise ValueError('%s not supported.' % CONFIG.MODEL.EMBEDDER_TYPE)

  model['cnn'] = cnn
  model['emb'] = emb

  return model


def get_conv_bn_layers(conv_params, use_bn, conv_dims=2):
  """Returns convolution and batch norm layers."""
  if conv_dims == 1:
    conv_layer = layers.Conv1D
  elif conv_dims == 2:
    conv_layer = layers.Conv2D
  elif conv_dims == 3:
    conv_layer = layers.Conv3D
  else:
    raise ValueError('Invalid number of conv_dims')
  l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT

  conv_layers = []
  bn_layers = []
  activations = []
  for channels, kernel_size, activate in conv_params:
    if activate:
      activation = tf.nn.relu
    else:
      activation = None
    conv_layers.append(conv_layer(
        channels, kernel_size,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg_weight),
        bias_regularizer=regularizers.l2(l2_reg_weight),
        kernel_initializer='he_normal',
        ))
    if use_bn:
      bn_layers.append(layers.BatchNormalization())
    activations.append(activation)

  return conv_layers, bn_layers, activations


def get_gru_layers(gru_params):
  """Returns GRU layers."""
  l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
  gru_layers = []
  for units in gru_params:
    gru_layers.append(
        layers.CuDNNGRU(units=units,
                        kernel_regularizer=regularizers.l2(l2_reg_weight),
                        bias_regularizer=regularizers.l2(l2_reg_weight),
                        return_sequences=True
                       ))
  return gru_layers


def get_fc_layers(fc_params):
  """Return fully connected layers."""
  l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
  fc_layers = []
  for channels, activate in fc_params:
    if activate:
      activation = tf.nn.relu
    else:
      activation = None
    fc_layers.append(
        layers.Dense(channels, activation=activation,
                     kernel_regularizer=regularizers.l2(l2_reg_weight),
                     bias_regularizer=regularizers.l2(l2_reg_weight)))
  return fc_layers
