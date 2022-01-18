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

"""Implement a 2-Layer MLP."""

import tensorflow as tf

from extreme_memorization import alignment


class MLP(tf.keras.Model):
  """Simple 2-Layer MLP."""

  def __init__(self,
               num_units,
               stddev,
               activation_fn=tf.nn.relu,
               custom_init=False,
               num_labels=10):
    super(MLP, self).__init__(name="MLP")
    self.custom_init = custom_init
    self.num_labels = num_labels
    if custom_init:
      self.hidden = tf.keras.layers.Dense(
          num_units,
          activation=activation_fn,
          kernel_initializer=tf.keras.initializers.RandomNormal(
              mean=0.0, stddev=stddev),
          use_bias=False,
          name="Dense")
    else:
      self.hidden = tf.keras.layers.Dense(
          num_units, activation=activation_fn, use_bias=False, name="Dense")
    self.top = tf.keras.layers.Dense(num_labels, use_bias=False, name="Top")

  def call(self, input_, labels, training=False, step=0):
    x = tf.keras.layers.Flatten()(input_)
    x = self.hidden(x)

    alignment.plot_class_alignment(
        x,
        labels,
        self.num_labels,
        step,
        tf_summary_key="representation_alignment")

    x = self.top(x)
    return x
