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

"""Frontend for features."""

from typing import Any, Dict

from absl import logging
import tensorflow as tf
from non_semantic_speech_benchmark.export_model import tf_frontend


frontend_args_from_flags = tf_frontend.frontend_args_from_flags


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


# TODO(joelshor): Deprecate this.
def get_feats_map_fn(tflite, frontend_args):
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


class SamplesToFeats(tf.keras.layers.Layer):
  """Compute features from samples."""

  def __init__(self, tflite, frontend_args):
    super(SamplesToFeats, self).__init__()
    self.tflite = tflite
    self.frontend_args = frontend_args
    logging.info('[SamplesToFeats] frontend_args: %s', self.frontend_args)

  def call(self, samples):
    if samples.shape.ndims != 2:
      raise ValueError(f'Frontend input must be ndim 2: {samples.shape.ndims}')
    if self.tflite:
      x = tf.squeeze(samples)
      return _sample_to_features(x, self.frontend_args, self.tflite)
    else:
      # TODO(joelshor): When we get a frontend function that works on batches,
      # remove this loop.
      map_fn = lambda s: _sample_to_features(s, self.frontend_args, self.tflite)
      return tf.map_fn(map_fn, samples, dtype=tf.float64)
