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

"""Tests for helper utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from qanet import squad_helper

tf.enable_eager_execution()


class SquadHelperTest(tf.test.TestCase):

  def test_constrain_prob_mat(self):
    batch_size = 2
    context_len = 10
    max_answer_size = 4

    logits = tf.constant(np.random.rand(batch_size, context_len, context_len),
                         dtype=tf.float32)
    prob_mat = tf.nn.softmax(logits)

    # New way
    prob_mat1 = squad_helper._constrain_prob_mat(prob_mat, max_answer_size)

    # Old way
    MAX_CONTEXT_SIZE = 100
    max_x_len = tf.shape(prob_mat)[1]
    upper_tri_mat = tf.slice(
        np.triu(
            np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32') -
            np.triu(
                np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32'),
                k=max_answer_size)), [0, 0], [max_x_len, max_x_len])
    prob_mat2 = prob_mat * tf.expand_dims(upper_tri_mat, 0)

    # Compare
    diff_value = np.sum(np.power(prob_mat1.numpy() - prob_mat2.numpy(), 2))
    self.assertAlmostEqual(0.0, diff_value)

if __name__ == '__main__':
  tf.test.main()
