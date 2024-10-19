# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""A layer which computes direct forward DCT II on input speech signal."""
import numpy as np
from kws_streaming.layers.compat import tf


class DCT(tf.keras.layers.Layer):
  """Computes forward DCT transofmation.

  It is based on direct implementation described at
  https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
  This is useful for speech feature extraction.
  """

  def __init__(self, num_features=None, use_tf=False, **kwargs):
    super(DCT, self).__init__(**kwargs)
    self.num_features = num_features
    self.use_tf = use_tf

  def build(self, input_shape):
    super(DCT, self).build(input_shape)

    # dct is computed on last dim
    feature_size = int(input_shape[-1])
    if self.num_features is None:
      self.num_features = int(input_shape[-1])
    if self.num_features > feature_size:
      raise ValueError('num_features: %d can not be > feature_size: %d' %
                       (self.num_features, feature_size))

    # DCT normalization
    self.norm = 1.0 / np.sqrt(2.0 * feature_size)

    if not self.use_tf:
      # precompute forward dct transformation
      self.dct = 2.0 * np.cos(np.pi * np.outer(
          np.arange(feature_size) * 2.0 + 1.0, np.arange(feature_size)) /
                              (2.0 * feature_size))

      # reduce dims, so that DCT is computed only on returned features
      # with size num_features
      self.dct = (self.dct[:, :self.num_features] * self.norm).astype(
          np.float32)

  def call(self, inputs):

    if self.use_tf:
      # compute DCT with tf function
      output = tf.signal.dct(inputs, type=2, n=self.num_features, norm=None)
      return output * self.norm
    else:
      # compute DCT direct matmul
      return tf.matmul(inputs, self.dct)

  def get_config(self):
    config = {
        'num_features': self.num_features,
        'use_tf': self.use_tf,
    }
    base_config = super(DCT, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
