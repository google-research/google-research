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

"""Matching module."""

import numpy as np
from ravens import utils
from ravens.models import ResNet36_4s
from ravens.models import ResNet43_8s
import tensorflow as tf
import tensorflow_addons as tfa


class Matching:
  """Matching module."""

  def __init__(self,
               input_shape,
               descriptor_dim,
               num_rotations,
               preprocess,
               lite=False):
    self.preprocess = preprocess
    self.num_rotations = num_rotations
    self.descriptor_dim = descriptor_dim

    max_dim = np.max(input_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(input_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)

    input_shape = np.array(input_shape)
    input_shape += np.sum(self.padding, axis=1)
    input_shape = tuple(input_shape)

    # Initialize fully convolutional Residual Network with 43 layers and
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
    if lite:
      d_in, d_out = ResNet36_4s(input_shape, self.descriptor_dim)
    else:
      d_in, d_out = ResNet43_8s(input_shape, self.descriptor_dim)
    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-5)
    self.metric = tf.keras.metrics.Mean(name='attention_loss')

  def forward(self, input_image):
    """Forward pass."""
    input_data = np.pad(input_image, self.padding, mode='constant')
    input_data = self.preprocess(input_data)
    input_shape = (1,) + input_data.shape
    input_data = input_data.reshape(input_shape)
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Rotate input.
    pivot = np.array(input_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.num_rotations, pivot)
    input_tensor = tf.repeat(input_tensor, repeats=self.num_rotations, axis=0)
    input_tensor = tfa.image.transform(
        input_tensor, rvecs, interpolation='NEAREST')

    # Forward pass.
    input_tensor = tf.split(input_tensor, self.num_rotations)
    logits = ()
    for x in input_tensor:
      logits += (self.model(x),)
    logits = tf.concat(logits, axis=0)

    # Rotate back output.
    rvecs = self.get_se2(self.num_rotations, pivot, reverse=True)
    logits = tfa.image.transform(logits, rvecs, interpolation='NEAREST')
    c0 = self.padding[:2, 0]
    c1 = c0 + input_image.shape[:2]
    output = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]
    return output

  def train(self, input_image, p, q, theta):
    """Train function."""
    self.metric.reset_states()
    with tf.GradientTape() as tape:
      output = self.forward(input_image)

      p_descriptor = output[0, p[0], p[1], :]
      itheta = theta / (2 * np.pi / self.num_rotations)
      itheta = np.int32(np.round(itheta)) % self.num_rotations
      q_descriptor = output[itheta, q[0], q[1], :]

      # Positives.
      positive_distances = tf.linalg.norm(p_descriptor - q_descriptor)
      positive_distances = tf.reshape(positive_distances, (1,))
      positive_labels = tf.constant([1], dtype=tf.int32)
      positive_loss = tfa.losses.contrastive_loss(positive_labels,
                                                  positive_distances)

      # Negatives.
      num_samples = 100
      sample_map = np.zeros(input_image.shape[:2] + (self.num_rotations,))
      sample_map[p[0], p[1], 0] = 1
      sample_map[q[0], q[1], itheta] = 1
      inegative = utils.sample_distribution(1 - sample_map, num_samples)
      negative_distances = ()
      negative_labels = ()
      for i in range(num_samples):
        descriptor = output[inegative[i, 2], inegative[i, 0], inegative[i,
                                                                        1], :]
        distance = tf.linalg.norm(p_descriptor - descriptor)
        distance = tf.reshape(distance, (1,))
        negative_distances += (distance,)
        negative_labels += (tf.constant([0], dtype=tf.int32),)
      negative_distances = tf.concat(negative_distances, axis=0)
      negative_labels = tf.concat(negative_labels, axis=0)
      negative_loss = tfa.losses.contrastive_loss(negative_labels,
                                                  negative_distances)
      negative_loss = tf.reduce_mean(negative_loss)

      loss = tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)

    # Backpropagate.
    grad = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

    self.metric(loss)
    return np.float32(loss)

  def load(self, path):
    self.model.load_weights(path)

  def save(self, filename):
    self.model.save(filename)

  def get_se2(self, num_rotations, pivot, reverse=False):
    """Get SE2 rotations discretized into num_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(num_rotations):
      theta = i * 2 * np.pi / num_rotations
      theta = -theta if reverse else theta
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)
