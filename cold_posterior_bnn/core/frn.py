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
"""Filter Response Normalization (FRN) layer for Keras.

Filter Response Normalization is a recently published method that serves as a
drop-in replacement for batch normalization in convolutional neural networks.

The paper is due to Singh and Krishnan, https://arxiv.org/pdf/1911.09737.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf   # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow.keras.regularizers as regularizers   # pylint: disable=g-explicit-tensorflow-version-import


# FRN from https://arxiv.org/pdf/1911.09737.pdf
class FRN(tf.keras.layers.Layer):
  """Filter Response Normalization (FRN) layer."""

  def __init__(self,
               reg_epsilon=1.0e-6,
               tau_regularizer=None,
               beta_regularizer=None,
               gamma_regularizer=None,
               **kwargs):
    """Initialize the FRN layer.

    Args:
      reg_epsilon: float, the regularization parameter preventing a division by
        zero.
      tau_regularizer: tf.keras.regularizer for tau.
      beta_regularizer: tf.keras.regularizer for beta.
      gamma_regularizer: tf.keras.regularizer for gamma.
      **kwargs: keyword arguments passed to the Keras layer base class.
    """
    self.reg_epsilon = reg_epsilon
    self.tau_regularizer = regularizers.get(tau_regularizer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    super(FRN, self).__init__(**kwargs)

  def build(self, input_shape):
    par_shape = (1, 1, 1, input_shape[-1])  # [1,1,1,C]
    self.tau = self.add_weight('tau', shape=par_shape, initializer='zeros',
                               regularizer=self.tau_regularizer,
                               trainable=True)
    self.beta = self.add_weight('beta', shape=par_shape, initializer='zeros',
                                regularizer=self.beta_regularizer,
                                trainable=True)
    self.gamma = self.add_weight('gamma', shape=par_shape, initializer='ones',
                                 regularizer=self.gamma_regularizer,
                                 trainable=True)

  def call(self, x):
    nu2 = tf.reduce_mean(tf.math.square(x), axis=[1, 2], keepdims=True)
    x = x * tf.math.rsqrt(nu2 + self.reg_epsilon)
    y = self.gamma*x + self.beta
    z = tf.maximum(y, self.tau)
    return z

  def get_config(self):
    config = super(FRN, self).get_config()
    config.update({
        'reg_epsilon': self.reg_epsilon,
        'tau_regularizer': regularizers.serialize(self.tau_regularizer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
    })
    return config


class TLU(tf.keras.layers.Layer):
  """Thresholded linear unit (TLU) layer."""

  def __init__(self,
               tau_regularizer=None,
               **kwargs):
    """Initialize the TLU layer.

    Args:
      tau_regularizer: tf.keras.regularizer for tau.
      **kwargs: keyword arguments passed to the Keras layer base class.
    """
    self.tau_regularizer = regularizers.get(tau_regularizer)
    super(TLU, self).__init__(**kwargs)

  def build(self, input_shape):
    par_shape = (1, 1, 1, input_shape[-1])  # [1,1,1,C]
    self.tau = self.add_weight('tau', shape=par_shape, initializer='zeros',
                               regularizer=self.tau_regularizer,
                               trainable=True)

  def call(self, x):
    return tf.maximum(x, self.tau)

  def get_config(self):
    config = super(TLU, self).get_config()
    config.update({
        'tau_regularizer': regularizers.serialize(self.tau_regularizer),
    })
    return config

