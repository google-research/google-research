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

"""Normalization layer."""
from kws_streaming.layers.compat import tf


class Normalizer(tf.keras.layers.Layer):
  """Normalize data by subtracting mean and dividing by stddev.

  It is useful for model convergence during training.
  Both mean and stddev have to be precomputed before training/inference.
  Normalization is applied on the last dim of input data
  """

  def __init__(self, mean=None, stddev=None, **kwargs):
    super(Normalizer, self).__init__(**kwargs)
    self.mean = mean
    self.stddev = stddev

  def build(self, input_shape):
    super(Normalizer, self).build(input_shape)
    feature_size = int(input_shape[-1])
    if self.mean is None:
      self.mean = [0.0] * feature_size
    if self.stddev is None:
      self.stddev = [1.0] * feature_size

  def call(self, inputs):
    return (inputs - self.mean) / self.stddev

  def get_config(self):
    config = {
        'mean': self.mean,
        'stddev': self.stddev
    }
    base_config = super(Normalizer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
