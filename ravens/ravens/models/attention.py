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

#!/usr/bin/env python
"""Attention module."""

import numpy as np
from ravens import utils
from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s
import tensorflow as tf
import tensorflow_addons as tfa


class Attention:
  """Attention module."""

  def __init__(self, in_shape, n_rotations, preprocess, lite=False):
    self.n_rotations = n_rotations
    self.preprocess = preprocess

    max_dim = np.max(in_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(in_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)

    in_shape = np.array(in_shape)
    in_shape += np.sum(self.padding, axis=1)
    in_shape = tuple(in_shape)

    # Initialize fully convolutional Residual Network with 43 layers and
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
    if lite:
      d_in, d_out = ResNet36_4s(in_shape, 1)
    else:
      d_in, d_out = ResNet43_8s(in_shape, 1)

    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_attention')

  def forward(self, in_img, softmax=True):
    """Forward pass."""
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = self.preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = tf.convert_to_tensor(in_data, dtype=tf.float32)

    # Rotate input.
    pivot = np.array(in_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.n_rotations, pivot)
    in_tens = tf.repeat(in_tens, repeats=self.n_rotations, axis=0)
    in_tens = tfa.image.transform(in_tens, rvecs, interpolation='NEAREST')

    # Forward pass.
    in_tens = tf.split(in_tens, self.n_rotations)
    logits = ()
    for x in in_tens:
      logits += (self.model(x),)
    logits = tf.concat(logits, axis=0)

    # Rotate back output.
    rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
    logits = tfa.image.transform(logits, rvecs, interpolation='NEAREST')
    c0 = self.padding[:2, 0]
    c1 = c0 + in_img.shape[:2]
    logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

    logits = tf.transpose(logits, [3, 1, 2, 0])
    output = tf.reshape(logits, (1, np.prod(logits.shape)))
    if softmax:
      output = tf.nn.softmax(output)
      output = np.float32(output).reshape(logits.shape[1:])
    return output

  def train(self, in_img, p, theta, backprop=True):
    """Train."""
    self.metric.reset_states()
    with tf.GradientTape() as tape:
      output = self.forward(in_img, softmax=False)

      # Get label.
      theta_i = theta / (2 * np.pi / self.n_rotations)
      theta_i = np.int32(np.round(theta_i)) % self.n_rotations
      label_size = in_img.shape[:2] + (self.n_rotations,)
      label = np.zeros(label_size)
      label[p[0], p[1], theta_i] = 1
      label = label.reshape(1, np.prod(label.shape))
      label = tf.convert_to_tensor(label, dtype=tf.float32)

      # Get loss.
      loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
      loss = tf.reduce_mean(loss)

    # Backpropagate
    if backprop:
      grad = tape.gradient(loss, self.model.trainable_variables)
      self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
      self.metric(loss)

    return np.float32(loss)

  def load(self, path):
    self.model.load_weights(path)

  def save(self, filename):
    self.model.save(filename)

  def get_se2(self, n_rotations, pivot, reverse=False):
    """Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(n_rotations):
      theta = i * 2 * np.pi / n_rotations
      theta = -theta if reverse else theta
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)
