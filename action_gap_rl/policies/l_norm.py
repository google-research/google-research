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
"""Exponential family policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from action_gap_rl.policies import layers_lib
import tensorflow.compat.v2 as tf


class LNormPolicy(tf.keras.Model):
  """A policy that takes an arbitrary function as the un-normalized log pdf."""

  def __init__(self, config, name=None):
    super(LNormPolicy, self).__init__(name=name or self.__class__.__name__)
    self._config = config
    hidden_widths = config.hidden_widths
    self.p = config.p
    self.q = config.q
    if config.embed:
      transformation_layers = [layers_lib.soft_hot_layer(**config.embed)]
    else:
      transformation_layers = []
    self._body = tf.keras.Sequential(
        transformation_layers
        + [tf.keras.layers.Dense(w, activation='relu') for w in hidden_widths]
        + [tf.keras.layers.Dense(1, activation=None)]
    )

  def call(self, states, actions):
    return self._body(states) - actions

  def regularizer(self, states):
    return 0.0

  def loss(self, states, actions, targets):
    return tf.reduce_mean(
        tf.abs(-tf.abs(self(states, actions))**self.p - targets)**self.q)

  def argmax(self, states):
    return self._body(states)
