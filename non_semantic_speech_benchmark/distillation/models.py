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
"""Models for distillation.

"""

from absl import logging
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from non_semantic_speech_benchmark.distillation.layers import CompressedDense
from non_semantic_speech_benchmark.export_model import tf_frontend
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras.applications import mobilenet_v3 as v3_util
# pylint: enable=g-direct-tensorflow-import


@tf.function
def _sample_to_features(x, tflite):
  return tf_frontend.compute_frontend_features(
      x, 16000, frame_hop=17, tflite=tflite)


def _get_feats_map_fn(tflite):
  """Returns a function mapping audio to features, suitable for keras Lambda."""
  if tflite:
    def feats_map_fn(x):
      # Keras Input needs a batch (which we statically fix to 1), but that
      # causes unexpected shapes in the frontend graph. So we squeeze out that
      # dim here.
      x = tf.squeeze(x)
      return _sample_to_features(x, tflite=True)
  else:

    def feats_map_fn(x):
      return tf.map_fn(
          lambda y: _sample_to_features(y, tflite=False), x, dtype=tf.float64)

  return feats_map_fn


def get_keras_model(bottleneck_dimension,
                    output_dimension,
                    alpha=1.0,
                    mobilenet_size='small',
                    frontend=True,
                    avg_pool=False,
                    compressor=None,
                    quantize_aware_training=False,
                    tflite=False):
  """Make a Keras student model."""
  # For debugging, log hyperparameter values.
  logging.info('bottleneck_dimension: %i', bottleneck_dimension)
  logging.info('output_dimension: %i', output_dimension)
  logging.info('alpha: %s', alpha)
  logging.info('frontend: %s', frontend)
  logging.info('avg_pool: %s', avg_pool)
  logging.info('compressor: %s', compressor)
  logging.info('quantize_aware_training: %s', quantize_aware_training)
  logging.info('tflite: %s', tflite)

  output_dict = {}  # Dictionary of model outputs.

  def _map_mobilenet_func(mnet_size):
    mnet_size_map = {
        'tiny': mobilenetv3_tiny,
        'small': tf.keras.applications.MobileNetV3Small,
        'large': tf.keras.applications.MobileNetV3Large,
    }
    if mnet_size.lower() not in mnet_size_map:
      raise ValueError('Unknown MobileNet size %s.' % mnet_size)
    return mnet_size_map[mnet_size.lower()]

  # TFLite use-cases usually use non-batched inference, and this also enables
  # hardware acceleration.
  num_batches = 1 if tflite else None
  if frontend:
    model_in = tf.keras.Input((None,),
                              name='audio_samples',
                              batch_size=num_batches)
    feats = tf.keras.layers.Lambda(_get_feats_map_fn(tflite))(model_in)
    feats = tf.reshape(feats, [-1, 96, 64, 1])
  else:
    model_in = tf.keras.Input((96, 64, 1), name='log_mel_spectrogram')
    feats = model_in
  inputs = [model_in]

  model = _map_mobilenet_func(mobilenet_size)(
      input_shape=[96, 64, 1],
      alpha=alpha,
      minimalistic=False,
      include_top=False,
      weights=None,
      pooling='avg' if avg_pool else None,
      dropout_rate=0.0)
  model_out = model(feats)
  if avg_pool:
    model_out.shape.assert_is_compatible_with([None, None])
  else:
    model_out.shape.assert_is_compatible_with([None, 1, 1, None])
  if bottleneck_dimension:
    if compressor is not None:
      bottleneck = CompressedDense(
          bottleneck_dimension,
          compression_obj=compressor,
          name='distilled_output')
    else:
      bottleneck = tf.keras.layers.Dense(
          bottleneck_dimension, name='distilled_output')
      if quantize_aware_training:
        bottleneck = tfmot.quantization.keras.quantize_annotate_layer(
            bottleneck)
    embeddings = tf.keras.layers.Flatten()(model_out)
    embeddings = bottleneck(embeddings)
  else:
    embeddings = tf.keras.layers.Flatten(name='distilled_output')(model_out)

  # Construct optional final layer, and create output dictionary.
  output_dict['embedding'] = embeddings
  if output_dimension:
    output = tf.keras.layers.Dense(
        output_dimension, name='embedding_to_target')(embeddings)
    output_dict['embedding_to_target'] = output
  output_model = tf.keras.Model(inputs=inputs, outputs=output_dict)

  # Optional modifications to the model for TFLite.
  if tflite:
    if compressor is not None:
      # If model employs compression, this ensures that the TFLite model
      # just uses the smaller matrices for inference.
      output_model.get_layer('distilled_output').kernel = None
      output_model.get_layer(
          'distilled_output').compression_op.a_matrix_tfvar = None

  return output_model


def mobilenetv3_tiny(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):
  """Makes MobileNetV3 model."""

  def stack_fn(x, kernel, activation, se_ratio):

    # Using blocks from MobileNetV3 saves a lot of code duplication.
    # pylint: disable=protected-access
    def depth(d):
      return v3_util._depth(d * alpha)

    x = v3_util._inverted_res_block(x, 1, depth(16), 3, 2, se_ratio,
                                    v3_util.relu, 0)
    x = v3_util._inverted_res_block(x, 72. / 16, depth(24), 3, 2, None,
                                    v3_util.relu, 1)
    x = v3_util._inverted_res_block(x, 88. / 24, depth(24), 3, 1, None,
                                    v3_util.relu, 2)
    x = v3_util._inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio,
                                    activation, 3)
    x = v3_util._inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio,
                                    activation, 4)
    x = v3_util._inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio,
                                    activation, 5)
    x = v3_util._inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio,
                                    activation, 6)
    x = v3_util._inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio,
                                    activation, 8)
    x = v3_util._inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio,
                                    activation, 9)
    # pylint: enable=protected-access
    return x

  return v3_util.MobileNetV3(stack_fn, 512, input_shape, alpha, 'tiny',
                             minimalistic, include_top, weights, input_tensor,
                             classes, pooling, dropout_rate,
                             classifier_activation)
