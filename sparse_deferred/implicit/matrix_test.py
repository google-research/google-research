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

"""Tests `Matrix` via implementation `NumpyMatrix`."""

import numpy as np
import tensorflow as tf

import sparse_deferred.np as sdnp


ADJ = np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 1, 0]])
NONZERO = ADJ.nonzero()
TGT = NONZERO[0]  # adj[i, j] == 1  <==> i->j.
SRC = NONZERO[1]

AM = sdnp.NumpyMatrix(ADJ)
NP_RND = np.array(np.random.uniform(size=[3, 5, 2], low=-1, high=1), 'float32')


class MatrixTest(tf.test.TestCase):

  def test_colsums(self):
    self.assertAllEqual(AM.colsums(), ADJ.sum(0))

  def test_rowsums(self):
    self.assertAllEqual(AM.rowsums(), ADJ.sum(1))

  def test_matmul(self):
    result = AM @ NP_RND
    expected_result0 = ADJ.dot(NP_RND[Ellipsis, 0])
    expected_result1 = ADJ.dot(NP_RND[Ellipsis, 1])
    expected_result = np.stack([expected_result0, expected_result1], -1)
    self.assertAllClose(expected_result, result)

  def test_rmatmul(self):
    result = tf.transpose(NP_RND) @ AM
    expected_result0 = NP_RND.T[0].dot(ADJ)
    expected_result1 = NP_RND.T[1].dot(ADJ)
    expected_result = np.stack([expected_result0, expected_result1], 0)
    self.assertAllClose(expected_result, result)

  def test_add_eye(self):
    rnd = NP_RND[:, :, 0]
    self.assertAllClose(AM.add_eye() @ rnd, (ADJ + np.eye(3)).dot(rnd))
    self.assertAllClose(
        tf.transpose(rnd) @ AM.add_eye(), rnd.T.dot(ADJ + np.eye(3)))

  def test_normalize_right(self):
    rnd = NP_RND[:, :, 0]
    anorm_right = ADJ / ADJ.sum(1, keepdims=True)
    right_stochastic = AM.normalize_right()
    self.assertAllEqual(right_stochastic.rowsums(), tf.ones([3]))
    self.assertAllClose(anorm_right.dot(rnd), right_stochastic @ rnd)
    self.assertAllClose(rnd.T.dot(anorm_right),
                        tf.transpose(rnd) @ right_stochastic)

  def test_normalize_left(self):
    rnd = NP_RND[:, :, 0]
    anorm_left = ADJ / ADJ.sum(0, keepdims=True)
    left_stochastic = AM.normalize_left()
    self.assertAllEqual(left_stochastic.colsums(), tf.ones([3]))
    self.assertAllClose(anorm_left.dot(rnd), left_stochastic @ rnd)
    self.assertAllClose(rnd.T.dot(anorm_left),
                        tf.transpose(rnd) @ left_stochastic)

  def test_normalize_leftright(self):
    rnd = NP_RND[:, :, 0]

    anorm_leftright = ((ADJ / np.sqrt(ADJ.sum(1, keepdims=True)))
                       / np.sqrt(ADJ.sum(0, keepdims=True)))
    normalized_leftright = AM.normalize_leftright()
    self.assertAllClose(anorm_leftright.dot(rnd), normalized_leftright @ rnd)
    self.assertAllClose(rnd.T.dot(anorm_leftright),
                        tf.transpose(rnd) @ normalized_leftright)


if __name__ == '__main__':
  tf.test.main()
