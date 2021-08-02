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


# TODO(joelshor): Tracing might not work when passing a python dictionary as an
# arg.
@tf.function
def _sample_to_features(x, frontend_args, tflite):
  if frontend_args is None:
    frontend_args = {}
  return tf_frontend.compute_frontend_features(
      x, 16000, tflite=tflite, **frontend_args)


def get_frontend_output_shape():
  frontend_args = tf_frontend.frontend_args_from_flags()
  x = tf.zeros([frontend_args['n_required']], dtype=tf.float32)
  return _sample_to_features(x, frontend_args, tflite=False).shape


def _get_feats_map_fn(tflite, frontend_args):
  """Returns a function mapping audio to features, suitable for keras Lambda.

  Args:
    tflite: A boolean whether the frontend should be suitable for tflite.
    frontend_args: A dictionary of key-value pairs for the frontend. Keys
      should be arguments to `tf_frontend.compute_frontend_features`.

  Returns:
    A python function mapping samples to features.
  """
  if tflite:
    def feats_map_fn(x):
      # Keras Input needs a batch (which we statically fix to 1), but that
      # causes unexpected shapes in the frontend graph. So we squeeze out that
      # dim here.
      x = tf.squeeze(x)
      return _sample_to_features(x, frontend_args, tflite=True)
  else:
    def feats_map_fn(x):
      map_fn = lambda y: _sample_to_features(y, frontend_args, tflite=False)
      return tf.map_fn(map_fn, x, dtype=tf.float64)

  return feats_map_fn


def _map_mobilenet_func(mnet_size):
  return {
      'small': tf.keras.applications.MobileNetV3Small,
      'large': tf.keras.applications.MobileNetV3Large,
      'debug': _debug_net,
  }[mnet_size.lower()]


def _debug_net(pooling, *args, **kwargs):
  """Small net for debugging."""
  del args, kwargs
  final_shape = [-1, 1] if pooling else [-1, 1, 1, 1]
  layers = [
      tf.keras.layers.Lambda(lambda x: tf.reshape(  # pylint: disable=g-long-lambda
          tf.reduce_mean(x, axis=[1, 2, 3]), final_shape)),
  ]
  return tf.keras.Sequential(layers)


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

  # TFLite use-cases usually use non-batched inference, and this also enables
  # hardware acceleration.
  num_batches = 1 if tflite else None
  frontend_args = tf_frontend.frontend_args_from_flags()
  feats_inner_dim = get_frontend_output_shape()[0]
  if frontend:
    logging.info('frontend_args: %s', frontend_args)
    model_in = tf.keras.Input((None,),
                              name='audio_samples',
                              batch_size=num_batches)
    frontend_fn = _get_feats_map_fn(tflite, frontend_args)
    feats = tf.keras.layers.Lambda(frontend_fn)(model_in)
    feats.shape.assert_is_compatible_with(
        [num_batches, feats_inner_dim, frontend_args['frame_width'],
         frontend_args['num_mel_bins']])
    feats = tf.reshape(
        feats, [-1, feats_inner_dim * frontend_args['frame_width'],
                frontend_args['num_mel_bins'], 1])
  else:
    model_in = tf.keras.Input(
        (feats_inner_dim * frontend_args['frame_width'],
         frontend_args['num_mel_bins'], 1),
        batch_size=num_batches,
        name='log_mel_spectrogram')
    feats = model_in
  inputs = [model_in]

  model = _map_mobilenet_func(mobilenet_size)(
      input_shape=(feats_inner_dim * frontend_args['frame_width'],
                   frontend_args['num_mel_bins'], 1),
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
