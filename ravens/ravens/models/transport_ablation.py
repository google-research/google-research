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

"""Transport ablations."""

import numpy as np
from ravens import utils
from ravens.models.transport import Transport
import tensorflow as tf


class TransportPerPixelLoss(Transport):
  """Transport + per-pixel loss ablation."""

  def __init__(self, in_shape, n_rotations, crop_size, preprocess):
    self.output_dim = 6
    super().__init__(in_shape, n_rotations, crop_size, preprocess)

  def correlate(self, in0, in1, softmax):
    output0 = tf.nn.convolution(in0[Ellipsis, :3], in1, data_format="NHWC")
    output1 = tf.nn.convolution(in0[Ellipsis, 3:], in1, data_format="NHWC")
    output = tf.concat((output0, output1), axis=0)
    output = tf.transpose(output, [1, 2, 3, 0])
    if softmax:
      output_shape = output.shape
      output = tf.reshape(output, (np.prod(output.shape[:-1]), 2))
      output = tf.nn.softmax(output)
      output = np.float32(output[:, 1]).reshape(output_shape[:-1])
    return output

  def train(self, in_img, p, q, theta, backprop=True):
    self.metric.reset_states()
    with tf.GradientTape() as tape:
      output = self.forward(in_img, p, softmax=False)

      itheta = theta / (2 * np.pi / self.n_rotations)
      itheta = np.int32(np.round(itheta)) % self.n_rotations
      label_size = in_img.shape[:2] + (self.n_rotations,)
      label = np.zeros(label_size)
      label[q[0], q[1], itheta] = 1

      # Get per-pixel sampling loss.
      sampling = True  # Sampling negatives seems to converge faster.
      if sampling:
        num_samples = 100
        inegative = utils.sample_distribution(1 - label, num_samples)
        inegative = [np.ravel_multi_index(i, label.shape) for i in inegative]
        ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
        output = tf.reshape(output, (-1, 2))
        output_samples = ()
        for i in inegative:
          output_samples += (tf.reshape(output[i, :], (1, 2)),)
        output_samples += (tf.reshape(output[ipositive, :], (1, 2)),)
        output = tf.concat(output_samples, axis=0)
        label = np.int32([0] * num_samples + [1])[Ellipsis, None]
        label = np.hstack((1 - label, label))
        weights = np.ones(label.shape[0])
        weights[:num_samples] = 1. / num_samples
        weights = weights / np.sum(weights)

      else:
        ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
        output = tf.reshape(output, (-1, 2))
        label = np.int32(np.reshape(label, (int(np.prod(label.shape)), 1)))
        label = np.hstack((1 - label, label))
        weights = np.ones(label.shape[0]) * 0.0025  # Magic constant.
        weights[ipositive] = 1

      label = tf.convert_to_tensor(label, dtype=tf.int32)
      weights = tf.convert_to_tensor(weights, dtype=tf.float32)
      loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
      loss = tf.reduce_mean(loss * weights)

      train_vars = self.model.trainable_variables
      if backprop:
        grad = tape.gradient(loss, train_vars)
        self.optim.apply_gradients(zip(grad, train_vars))
      self.metric(loss)

    self.iters += 1
    return np.float32(loss)
