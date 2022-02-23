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

from typing import Optional
from absl import flags
from absl import logging
import tensorflow as tf
from non_semantic_speech_benchmark.distillation import models
from non_semantic_speech_benchmark.trillsson import frontend_lib


def get_keras_model(model_type,
                    output_dimension = 1024,
                    frame_hop = None):
  """Make a Keras student model.

  Args:
    model_type: A string defining the architecture.
    output_dimension: The dimension of the output.
    frame_hop: Frontend framehop.

  Returns:
    A keras model encapsulating everything.
  """
  if frame_hop is None:
    frame_hop = flags.FLAGS.frame_hop
  # For debugging, log hyperparameter values.
  logging.info('model name: %s', model_type)
  logging.info('frame_hop: %s', frame_hop)

  output_dict = {}  # Dictionary of model outputs.

  # Construct model input and frontend.
  # TODO(joelshor): Consider using a ragged Tensor.
  model_in = tf.keras.Input((None,), name='audio_samples')
  feats = frontend_keras(model_in, frame_hop)
  feats.shape.assert_is_compatible_with([None, None, None, 1])

  # Build network.
  logging.info('Features shape: %s', feats.shape)
  model_out = models.build_main_net(model_type, feats)
  logging.info('Model output shape: %s', model_out.shape)

  # Reshape back to batch dimension, and average.
  bs = tf.shape(model_in)[0]
  main_net_output_dim = tf.shape(model_out)[-1]
  model_out = tf.reshape(model_out, [bs, -1, main_net_output_dim])
  model_out = tf.math.reduce_mean(model_out, axis=1, keepdims=False)

  # Construct optional final layer, and create output dictionary.
  if model_out.shape[1] == output_dimension:
    embedding = model_out
  else:
    embedding = tf.keras.layers.Dense(output_dimension)(model_out)
  output_dict['embedding'] = embedding
  output_model = tf.keras.Model(inputs=[model_in], outputs=output_dict)

  return output_model


def frontend_keras(
    model_in,
    frame_hop):
  """Returns features."""
  frontend_args = dict(
      frame_width=195,
      frame_hop=frame_hop,
      n_required=32000,
      num_mel_bins=80,
      pad_mode='SYMMETRIC')
  logging.info('frontend_args overrides: %s', frontend_args)

  feats = frontend_lib.SamplesToFeats(frontend_args)(model_in)
  feats.shape.assert_is_compatible_with(
      [None, None, frontend_args['frame_width'], frontend_args['num_mel_bins']])

  # Change the number of batches so we can run all chunks through the model
  # at once.
  # Note: A priori, it's not clear that reshape will group frames from the
  # first clip into the first N elements. We've verified this.
  feats = tf.reshape(
      feats,
      [-1, frontend_args['frame_width'], frontend_args['num_mel_bins'], 1])

  return feats
