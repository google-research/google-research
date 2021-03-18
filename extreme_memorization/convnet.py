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

"""Implement ConvNet architecture."""

import tensorflow as tf

from extreme_memorization import alignment


class ConvNet(tf.keras.Model):
  """Neural network used to train CIFAR10.

  Uses Keras to define convolutional and fully connected layers.
  """

  def __init__(self, num_labels=10):
    super(ConvNet, self).__init__()
    self.num_labels = num_labels
    self.conv1 = tf.keras.layers.Conv2D(
        32, 5, padding='same', activation='relu')
    self.max_pool1 = tf.keras.layers.MaxPooling2D((3, 3), (2, 2),
                                                  padding='valid')
    self.conv2 = tf.keras.layers.Conv2D(
        64, 5, padding='valid', activation='relu')
    self.max_pool2 = tf.keras.layers.MaxPooling2D((3, 3), (2, 2),
                                                  padding='valid')
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1024, activation='relu', name='hidden')
    self.fc2 = tf.keras.layers.Dense(num_labels, name='top')

  def call(self, x, labels, training=False, step=0):
    """Used to perform a forward pass."""
    # Assume channels_last
    input_shape = [32, 32, 3]
    x = tf.keras.layers.Reshape(
        target_shape=input_shape, input_shape=(32 * 32 * 3,))(
            x)

    if tf.keras.backend.image_data_format() == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      x = tf.transpose(a=x, perm=[0, 3, 1, 2])

    x = self.max_pool1(self.conv1(x))
    x = self.max_pool2(self.conv2(x))
    x = self.flatten(x)
    x = self.fc1(x)

    alignment.plot_class_alignment(
        x,
        labels,
        self.num_labels,
        step,
        tf_summary_key='representation_alignment')

    x = self.fc2(x)
    return x
