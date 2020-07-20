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

# Lint as: python3
"""Small models to be finetuned on embeddings."""

import tensorflow.compat.v2 as tf
from tensorflow_addons.layers.netvlad import NetVLAD
import tensorflow_hub as hub


def get_keras_model(num_classes, input_length, use_batchnorm=True, l2=1e-5,
                    num_clusters=None):
  """Make a model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input((input_length,)))  # Input is [bs, input_length]
  trill_layer = hub.KerasLayer(
      handle='https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/2',
      trainable=True,
      arguments={'sample_rate': tf.constant(16000, tf.int32)},
      output_key='embedding',
      output_shape=[None, 2048]
  )
  model.add(trill_layer)
  if num_clusters > 0:
    model.add(NetVLAD(num_clusters=num_clusters))
    if use_batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
  else:
    model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
  model.add(tf.keras.layers.Dense(
      num_classes, kernel_regularizer=tf.keras.regularizers.l2(l=l2)))

  return model
