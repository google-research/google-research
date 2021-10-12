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

"""LayerNormalization layer with abs op."""
from kws_streaming.layers.compat import tf


class LayerNormalizationAbs(tf.keras.layers.Layer):
  """LayerNormalization layer with abs op.

  It uses abs instead of sqr during deviation computation.
  As a result it simplifies the model and removes both sqr and sqrt ops.
  It is a basic LayerNormalization layer which is applied on last dim only.
  """

  def __init__(self, epsilon=0.001, **kwargs):
    super(LayerNormalizationAbs, self).__init__(**kwargs)
    self.epsilon = epsilon

  def call(self, inputs):

    mean = tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
    deviation_abs = tf.math.reduce_mean(
        tf.abs(inputs - mean), axis=-1, keepdims=True)
    return (inputs - mean) / (deviation_abs + self.epsilon)

  def get_config(self):
    config = {
        'epsilon': self.epsilon,
    }
    base_config = super(LayerNormalizationAbs, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
