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

"""Tests for frequency analysis."""

import numpy as np
import tensorflow.compat.v1 as tf
from frequency_analysis import freq_heatmap


class FreqHeatmapTest(tf.test.TestCase):

  def test_heatmap(self):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3))
    y = tf.placeholder(dtype=tf.int64, shape=None)
    flat_x = tf.reshape(x, [-1, 9])
    np.random.seed(8629)
    row_1 = 0.3 + 0.5 * np.random.randn(9)
    init_val = np.transpose(np.array([row_1, (-1.0) * row_1]))
    init_w = tf.constant_initializer(init_val)
    w = tf.get_variable('weights', shape=(9, 2), initializer=init_w)
    logits = tf.matmul(flat_x, w)
    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      x_np = np.array([0.2 * np.ones([3, 3]), (-0.2) * np.ones([3, 3])])
      y_np = np.array([0, 1])
      data_dict = {x: x_np, y: y_np}
      sess.run(init, feed_dict=data_dict)

      # Compute Fourier heatmaps and test error using generate_freq_heatmap.
      neural_network = freq_heatmap.TensorFlowNeuralNetwork(
          sess, x, y, [logits], accuracy)
      heatmaps, test_acc, clean_test_acc = freq_heatmap.generate_freq_heatmap(
          neural_network, x_np, y_np)

      # Compute the Fourier heatmaps without using generate_freq_heatmap.
      heatmap_check = np.zeros([3, 3])
      test_acc_check = np.zeros([3, 3])
      for pos in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        fft_basis = np.zeros([3, 3], dtype=np.complex64)
        if pos == (1, 1):
          fft_basis[1, 1] = 1.0
        else:
          fft_basis[pos[0], pos[1]] = 0.5 + 0.5j
          fft_basis[2 - pos[0], 2 - pos[1]] = 0.5 - 0.5j
        basis = 3 * np.real(np.fft.ifft2(np.fft.ifftshift(fft_basis)))
        basis_flat = np.reshape(basis, [1, 9])
        logit_change = np.matmul(basis_flat, init_val)
        change_norm = np.linalg.norm(logit_change)
        heatmap_check[pos[0], pos[1]] = change_norm
        heatmap_check[2 - pos[0], 2 - pos[1]] = change_norm
        data_dict_basis = {x: x_np + basis, y: y_np}
        test_acc_basis = sess.run(accuracy, feed_dict=data_dict_basis)
        test_acc_check[pos[0], pos[1]] = test_acc_basis
        test_acc_check[2 - pos[0], 2 - pos[1]] = test_acc_basis
      heatmap_check /= np.amax(heatmap_check)
      clean_test_acc_check = sess.run(accuracy, feed_dict=data_dict)

      self.assertAllClose([heatmap_check], heatmaps)
      self.assertAllClose(test_acc_check, test_acc)
      self.assertEqual(clean_test_acc_check, clean_test_acc)


if __name__ == '__main__':
  tf.test.main()
