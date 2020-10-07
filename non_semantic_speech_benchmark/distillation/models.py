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

"""Models for distillation."""

import tensorflow as tf
from non_semantic_speech_benchmark.export_model import tf_frontend


@tf.function
def _sample_to_features(x):
  return tf_frontend.compute_frontend_features(x, 16000, overlap_seconds=79)


def get_keras_model(bottleneck_dimension, output_dimension, alpha=1.0):
  """Make a model."""
  audio_tensor = tf.keras.Input((None,))
  def _map_fn_lambda(x):
    return tf.map_fn(_sample_to_features, x, dtype=tf.float64)
  feats = tf.keras.layers.Lambda(_map_fn_lambda)(audio_tensor)
  feats.shape.assert_is_compatible_with([None, None, 96, 64])
  feats = tf.transpose(feats, [0, 2, 1, 3])
  feats = tf.reshape(feats, [-1, 96, 64, 1])
  model = tf.keras.applications.MobileNetV3Large(
      input_shape=[96, 64, 1],
      alpha=alpha,
      minimalistic=False,
      include_top=False,
      weights=None,
      pooling=None,
      dropout_rate=0.0)
  model_out = model(feats)
  model_out.shape.assert_is_compatible_with([None, 3, 2, 1280])
  embeddings = tf.keras.backend.batch_flatten(model_out)
  embeddings.set_shape([None, 3 * 2 * 1280])
  # TODO(joelshor): These final layers can be large. Investigate the compression
  # techniques described in
  # https://blog.tensorflow.org/2020/02/matrix-compression-operator-tensorflow.html?m=1
  embeddings = tf.keras.layers.Dense(
      bottleneck_dimension, name='distilled_output')(embeddings)
  output = tf.keras.layers.Dense(
      output_dimension, name='embedding_to_target')(embeddings)

  model = tf.keras.Model(inputs=audio_tensor, outputs=output)

  return model
