# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for load_model."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from loss_functions_transfer import load_model


class LoadModelTest(parameterized.TestCase):

  @parameterized.parameters(list(load_model.LOSS_HYPERPARAMETERS))
  def test_build_and_restore(self, loss_name):
    seed = 0
    inputs_np = np.zeros((1, 224, 224, 3))
    labels_np = np.pad([[1]], ((0, 0), (1000, 0)))

    with tf.Graph().as_default():
      inputs = tf.compat.v1.placeholder(tf.float32, (None, 224, 224, 3))
      labels = tf.compat.v1.placeholder(tf.float32, (None, 1001))
      loss, endpoints = load_model.build_model_and_compute_loss(
          loss_name=loss_name, inputs=inputs, labels=labels,
          is_training=False)
      with tf.compat.v1.Session() as sess:
        load_model.restore_checkpoint(loss_name, seed, sess)
        sess.run((loss, endpoints),
                 feed_dict={inputs: inputs_np, labels: labels_np})


if __name__ == '__main__':
  absltest.main()
