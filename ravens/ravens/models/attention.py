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

"""Attention module."""

import numpy as np
from ravens import utils
from ravens.models import ResNet36_4s
from ravens.models import ResNet43_8s
import tensorflow as tf
import tensorflow_addons as tfa


class Attention:
  """Attention module."""

  def __init__(self, input_shape, num_rotations, preprocess, lite=False):
    self.num_rotations = num_rotations
    self.preprocess = preprocess
    self.lr = 1e-5

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
      d_in, d_out = ResNet36_4s(input_shape, 1)
    else:
      d_in, d_out = ResNet43_8s(input_shape, 1)
    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
    self.metric = tf.keras.metrics.Mean(name='attention_loss')

  def forward(self, in_img, apply_softmax=True):
    """Forward pass."""
    input_data = np.pad(in_img, self.padding, mode='constant')
    input_data = self.preprocess(input_data)
    input_shape = (1,) + input_data.shape
    input_data = input_data.reshape(input_shape)
    in_tens = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Rotate input
    pivot = np.array(input_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.num_rotations, pivot)
    in_tens = tf.repeat(in_tens, repeats=self.num_rotations, axis=0)
    in_tens = tfa.image.transform(in_tens, rvecs, interpolation='NEAREST')

    # Forward pass
    in_tens = tf.split(in_tens, self.num_rotations)
    logits = ()
    for x in in_tens:
      logits += (self.model(x),)
    logits = tf.concat(logits, axis=0)

    # Rotate back output
    rvecs = self.get_se2(self.num_rotations, pivot, reverse=True)
    logits = tfa.image.transform(logits, rvecs, interpolation='NEAREST')
    c0 = self.padding[:2, 0]
    c1 = c0 + in_img.shape[:2]
    logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

    logits = tf.transpose(logits, [3, 1, 2, 0])
    output = tf.reshape(logits, (1, np.prod(logits.shape)))
    if apply_softmax:
      output = tf.nn.softmax(output)
      output = np.float32(output).reshape(logits.shape[1:])
    return output

  def train(self, in_img, p, theta):
    """Train."""
    self.metric.reset_states()
    with tf.GradientTape() as tape:
      output = self.forward(in_img, apply_softmax=False)

      # Compute label
      theta_i = theta / (2 * np.pi / self.num_rotations)
      theta_i = np.int32(np.round(theta_i)) % self.num_rotations
      label_size = in_img.shape[:2] + (self.num_rotations,)
      label = np.zeros(label_size)
      label[p[0], p[1], theta_i] = 1
      label = label.reshape(1, np.prod(label.shape))
      label = tf.convert_to_tensor(label, dtype=tf.float32)

      # Compute loss
      loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
      loss = tf.reduce_mean(loss)

    # Backpropagate
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
