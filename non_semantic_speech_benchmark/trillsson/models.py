# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Keras models with only the features needed for TRILLsson."""
from absl import logging
import tensorflow as tf
from non_semantic_speech_benchmark.distillation import frontend_lib
from non_semantic_speech_benchmark.distillation import models


def get_keras_model(model_type,
                    manually_average,
                    frontend = True,
                    output_dimension = 1024):
  """Make a Keras student model."""
  # For debugging, log hyperparameter values.
  logging.info('model name: %s', model_type)
  logging.info('frontend: %s', frontend)

  output_dict = {}  # Dictionary of model outputs.

  # Construct model input and frontend.
  model_in = tf.keras.Input((None,), name='audio_samples')
  feats = frontend_keras(model_in, manually_average)
  feats.shape.assert_is_compatible_with([None, None, None, 1])

  # Build network.
  logging.info('Features shape: %s', feats.shape)
  model_out = models.build_main_net(model_type, feats)
  logging.info('Model output shape: %s', model_out.shape)

  if manually_average:
    # Reshape back to batch dimension, and average.
    bs = tf.shape(model_in)[0]
    model_out = tf.reshape(model_out, [bs, -1, output_dimension])
    model_out = tf.math.reduce_mean(model_out, axis=1, keepdims=False)

  # Construct optional final layer, and create output dictionary.
  if model_out.shape[1] == output_dimension:
    embedding = model_out
  else:
    embedding = tf.keras.layers.Dense(output_dimension)(model_out)
  output_dict['embedding'] = embedding
  output_model = tf.keras.Model(inputs=[model_in], outputs=output_dict)

  return output_model


def frontend_keras(model_in, manually_average):
  """Returns features."""
  frontend_args = frontend_lib.frontend_args_from_flags()
  logging.info('frontend_args: %s', frontend_args)

  bs = tf.shape(model_in)[0]
  feats = frontend_lib.SamplesToFeats(
      tflite=False, frontend_args=frontend_args)(
          model_in)
  feats.shape.assert_is_compatible_with(
      [None, None, frontend_args['frame_width'], frontend_args['num_mel_bins']])
  if manually_average:
    # Change the number of batches so we can run all chunks through the model
    # at once.
    # Note: A priori, it's not clear that reshape will group frames from the
    # first clip into the first N elements. We've verified this.
    feats = tf.reshape(
        feats,
        [-1, frontend_args['frame_width'], frontend_args['num_mel_bins'], 1])
  else:
    # Keep the number of batches, and change the first dimension. We expect the
    # model to automatically average.
    feats = tf.reshape(feats, [bs, -1, frontend_args['num_mel_bins'], 1])
    feats.shape.assert_is_compatible_with(
        [None, None, frontend_args['num_mel_bins'], 1])

  return feats
