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
from typing import Tuple

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
from non_semantic_speech_benchmark.distillation import frontend_lib



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
                    output_dimension,
                    truncate_output = False,
                    frontend = True,
                    tflite = False):
  """Make a Keras student model."""
  # For debugging, log hyperparameter values.
  logging.info('model name: %s', model_type)
  logging.info('truncate_output: %s', truncate_output)
  logging.info('output_dimension: %i', output_dimension)
  logging.info('frontend: %s', frontend)
  logging.info('tflite: %s', tflite)

  output_dict = {}  # Dictionary of model outputs.

  # Construct model input and frontend.
  model_in, feats = _frontend_keras(frontend, tflite)
  feats.shape.assert_is_compatible_with([None, None, None, 1])
  inputs = [model_in]
  logging.info('Features shape: %s', feats.shape)

  # Build network.
  model_out = _build_main_net(model_type, feats)
  embeddings = tf.keras.layers.Flatten(name='distilled_output')(model_out)

  # The last fully-connected layer can sometimes be the single largest
  # layer in the entire network. It's also not always very valuable. We try
  # two methods of getting the right output dimension:
  # 1) A FC layer
  # 2) Taking the first `output_dimension` elements.
  need_final_layer = (output_dimension and
                      embeddings.shape[1] != output_dimension)

  # If we need to truncate, do it before we save the embedding. Otherwise,
  # the embedding will contain some garbage dimensions.
  if need_final_layer and truncate_output:
    if embeddings.shape[1] < output_dimension:
      embeddings = tf.pad(
          embeddings, [[0, 0], [0, output_dimension - embeddings.shape[1]]])
    else:
      embeddings = embeddings[:, :output_dimension]

  # Construct optional final layer, and create output dictionary.
  output_dict['embedding'] = embeddings

  target = embeddings
  if need_final_layer and not truncate_output:
    target = tf.keras.layers.Dense(
        output_dimension, name='embedding_to_target')(target)
  output_dict['embedding_to_target'] = target
  output_model = tf.keras.Model(inputs=inputs, outputs=output_dict)

  return output_model


def _frontend_keras(
    frontend,
    tflite):
  """Returns model input and features."""
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

  # `model_in` can be wavs or spectral features, but `feats` must be a 4D
  # spectrogram.
  feats.shape.assert_is_compatible_with(
      [None, feats_inner_dim * frontend_args['frame_width'],
       frontend_args['num_mel_bins'], 1])

  return (model_in, feats)


def _build_main_net(
    model_type,
    feats,
    ):
  """Constructs main network."""
  if model_type.startswith('mobilenet_'):
    # Format is "mobilenet_{size}_{alpha}_{avg_pool}"
    _, mobilenet_size, alpha, avg_pool = model_type.split('_')
    alpha = float(alpha)
    avg_pool = bool(avg_pool)
    logging.info('mobilenet_size: %s', mobilenet_size)
    logging.info('alpha: %f', alpha)
    logging.info('avg_pool: %s', avg_pool)
    model = _map_mobilenet_func(mobilenet_size)(
        input_shape=feats.shape[1:],
        alpha=alpha,
        minimalistic=False,
        include_top=False,
        weights=None,
        pooling='avg' if avg_pool else None,
        dropout_rate=0.0)
    expected_output_shape = [None, None] if avg_pool else [None, 1, 1, None]
  elif model_type.startswith('efficientnet'):
    # pylint:disable=line-too-long
    model_fn, final_dim = {
        'efficientnetb0': (tf.keras.applications.EfficientNetB0, 1280),
        'efficientnetb1': (tf.keras.applications.EfficientNetB1, 1280),
        'efficientnetb2': (tf.keras.applications.EfficientNetB2, 1408),
        'efficientnetb3': (tf.keras.applications.EfficientNetB3, 1536),
        'efficientnetb4': (tf.keras.applications.EfficientNetB4, 1792),
        'efficientnetb5': (tf.keras.applications.EfficientNetB5, 2048),
        'efficientnetb6': (tf.keras.applications.EfficientNetB6, 2304),
        'efficientnetb7': (tf.keras.applications.EfficientNetB7, 2560),
        # V2
        'efficientnetv2b0': (tf.keras.applications.efficientnet_v2.EfficientNetV2B0, 1280),
        'efficientnetv2b1': (tf.keras.applications.efficientnet_v2.EfficientNetV2B1, 1280),
        'efficientnetv2b2': (tf.keras.applications.efficientnet_v2.EfficientNetV2B2, 1408),
        'efficientnetv2b3': (tf.keras.applications.efficientnet_v2.EfficientNetV2B3, 1536),
        'efficientnetv2bL': (tf.keras.applications.efficientnet_v2.EfficientNetV2L, 1280),
        'efficientnetv2bM': (tf.keras.applications.efficientnet_v2.EfficientNetV2M, 1280),
        'efficientnetv2bS': (tf.keras.applications.efficientnet_v2.EfficientNetV2S, 1280),
    }[model_type]
    # pylint:enable=line-too-long
    model = model_fn(
        include_top=False,
        weights=None,  # could be pretrained from imagenet.
        input_shape=feats.shape[1:],
        pooling='avg',
    )
    expected_output_shape = [None, final_dim]
  else:
    raise ValueError(f'`model_type` not recognized: {model_type}')

  # TODO(joelshor): Consider checking that there are trainable weights in
  # `model`.
  model_out = model(feats)
  model_out.shape.assert_is_compatible_with(expected_output_shape)

  return model_out
