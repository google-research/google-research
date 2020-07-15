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


class TrillOnBatches(tf.keras.layers.Layer):
  """Run TRILL on a batch of samples."""

  def __init__(self):
    super(TrillOnBatches, self).__init__()
    self.module = hub.load(
        'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/2')  # pylint:disable=line-too-long
    assert self.module
    assert self.module.trill_module.variables

  def call(self, batched_samples, sample_rate=16000):
    batched_samples.shape.assert_has_rank(2)
    batched_embeddings = self.module(
        batched_samples, tf.constant(sample_rate, tf.int32))['embedding']
    batched_embeddings.shape.assert_has_rank(3)
    return batched_embeddings


def get_keras_model(num_classes, input_length, use_batchnorm=True, l2=1e-5,
                    num_clusters=None):
  """Make a model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input((input_length,)))  # Input is [bs, input_length]
  model.add(TrillOnBatches())
  if num_clusters > 0:
    model.add(NetVLAD(num_clusters=num_clusters))
    if use_batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
  else:
    model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
  model.add(tf.keras.layers.Dense(
      num_classes, kernel_regularizer=tf.keras.regularizers.l2(l=l2)))

  return model
