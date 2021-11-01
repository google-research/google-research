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
import os
from typing import Optional

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot
from non_semantic_speech_benchmark.distillation import frontend_lib
from non_semantic_speech_benchmark.distillation.layers import CompressedDense



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


def get_keras_model(model_type,
                    bottleneck_dimension,
                    output_dimension,
                    frontend = True,
                    compressor = None,
                    quantize_aware_training = False,
                    tflite = False):
  """Make a Keras student model."""
  # For debugging, log hyperparameter values.
  logging.info('model name: %s', model_type)
  logging.info('bottleneck_dimension: %i', bottleneck_dimension)
  logging.info('output_dimension: %i', output_dimension)
  logging.info('frontend: %s', frontend)
  logging.info('compressor: %s', compressor)
  logging.info('quantize_aware_training: %s', quantize_aware_training)
  logging.info('tflite: %s', tflite)

  output_dict = {}  # Dictionary of model outputs.

  # TFLite use-cases usually use non-batched inference, and this also enables
  # hardware acceleration.
  num_batches = 1 if tflite else None
  frontend_args = frontend_lib.frontend_args_from_flags()
  feats_inner_dim = frontend_lib.get_frontend_output_shape()[0]
  if frontend:
    logging.info('frontend_args: %s', frontend_args)
    model_in = tf.keras.Input((None,),
                              name='audio_samples',
                              batch_size=num_batches)
    frontend_fn = frontend_lib.get_feats_map_fn(tflite, frontend_args)
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
  logging.info('Features shape: %s', feats.shape)

  # Build network.
  if model_type.startswith('mobilenet_'):
    # Format is "mobilenet_{size}_{alpha}_{avg_pool}"
    _, mobilenet_size, alpha, avg_pool = model_type.split('_')
    alpha = float(alpha)
    avg_pool = bool(avg_pool)
    logging.info('mobilenet_size: %s', mobilenet_size)
    logging.info('alpha: %f', alpha)
    logging.info('avg_pool: %s', avg_pool)
    model = _map_mobilenet_func(mobilenet_size)(
        input_shape=(feats_inner_dim * frontend_args['frame_width'],
                     frontend_args['num_mel_bins'], 1),
        alpha=alpha,
        minimalistic=False,
        include_top=False,
        weights=None,
        pooling='avg' if avg_pool else None,
        dropout_rate=0.0)
    expected_output_shape = [None, None] if avg_pool else [None, 1, 1, None]
  elif model_type.startswith('efficientnet'):
    model_fn, final_dim = {
        'efficientnetb0': (tf.keras.applications.EfficientNetB0, 1280),
        'efficientnetb1': (tf.keras.applications.EfficientNetB1, 1280),
        'efficientnetb2': (tf.keras.applications.EfficientNetB2, 1408),
        'efficientnetb3': (tf.keras.applications.EfficientNetB3, 1536),
    }[model_type]
    model = model_fn(
        include_top=False,
        weights=None,  # could be pretrained from imagenet.
        input_shape=(feats_inner_dim * frontend_args['frame_width'],
                     frontend_args['num_mel_bins'], 1),
        pooling='avg',
    )
    expected_output_shape = [None, final_dim]
  else:
    raise ValueError(f'`model_type` not recognized: {model_type}')

  # TODO(joelshor): Consider checking that there are trainable weights in
  # `model`.
  model_out = model(feats)
  model_out.shape.assert_is_compatible_with(expected_output_shape)

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
