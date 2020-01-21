# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from numpy import linalg as LA

from dictionary_learning import dictionary_learning


class DictionaryLearningTest(absltest.TestCase):

  def testBasicDictionaryLearning(self):
    embedding_table = np.array([[1, 2, 3, 4, 5], [10, 9, 8, 7, 6],
                                [-5, -6, -7, -8, -9], [100, 101, 102, 103, 104],
                                [20, 19, 18, 17, 16], [30, 39, 38, 37, 36],
                                [45, 44, 43, 42, 41], [88, 89, 89, 80, 91],
                                [54, 53, 52, 51, 50]])

    [code0, dictionary0] = dictionary_learning.dictionary_learning(
        embedding_table,
        row_percentage=0.5,
        col_percentage=0.5,
        version_num=1,
        num_buckets=1)

    [code1, dictionary1] = dictionary_learning.dictionary_learning(
        embedding_table,
        row_percentage=0.8,
        col_percentage=0.8,
        version_num=1,
        num_buckets=1)

    approx0 = np.matmul(code0, dictionary0)
    error0 = 100.0 * LA.norm(embedding_table - approx0,
                             "fro") / LA.norm(embedding_table)

    approx1 = np.matmul(code1, dictionary1)
    error1 = 100.0 * LA.norm(embedding_table - approx1,
                             "fro") / LA.norm(embedding_table)

    # Expect that error0 > error1 because the sizes of code1 & dictionary1 are
    # larger than the sizes of code0 & dictionary0.
    self.assertGreaterEqual(error0, error1)

  def testDictionaryLearningWithZeroMatrix(self):
    embedding_table = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

    [code,
     dictionary] = dictionary_learning.dictionary_learning(embedding_table)

    approx = np.matmul(code, dictionary)

    # Expect that 'approx' is a 0-matrix
    self.assertEqual(np.count_nonzero(approx), 0)

  def testDictionaryLearningWithSimulatedData(self):
    # Generate random matrices a, b, and c.
    b = np.random.normal(0, 1, [30, 20])
    sub_thresh_indices = (b <= 0.7)
    b[sub_thresh_indices] = 0
    b[:, 0] = np.ones(shape=b[:, 0].shape)
    c = np.random.normal(0, 1, [20, 10])
    a = np.matmul(b, c)

    [b_out, c_out] = dictionary_learning.dictionary_learning(
        a,
        row_percentage=0.5,
        col_percentage=0.3,
        n_iterations=0,
        version_num=1)
    a_recovered = np.matmul(b_out, c_out)
    error = np.linalg.norm(a - a_recovered) / np.linalg.norm(a)
    # Expect that the approximation 'a_recovered' is not too bad.
    self.assertGreaterEqual(error, 0.00005)

  def testDictionaryLearningWithSimulatedDataLsh(self):
    # The test case has been modified to smaller matrices, the original runtime
    # is recorded as follows:for b.shape = [10000, 5000], c.shape = [5000, 650],
    # one iteration of DL takes about 1 hour.
    # Generate random matrices a, b, and c.
    b = np.random.normal(0, 1, [1000, 500])
    sub_thresh_indices = (b <= 0.7)
    b[sub_thresh_indices] = 0
    b[:, 0] = np.ones(shape=b[:, 0].shape)
    c = np.random.normal(0, 1, [500, 65])
    a = np.matmul(b, c)

    [b_out, c_out] = dictionary_learning.dictionary_learning(
        a,
        row_percentage=0.5,
        col_percentage=0.3,
        n_iterations=3,
        use_lsh=True,
        version_num=1)
    a_recovered = np.matmul(b_out, c_out)
    error = np.linalg.norm(a - a_recovered) / np.linalg.norm(a)

    # Expect that the approximation 'a_recovered' is not too bad.
    self.assertLessEqual(error, 0.2)
    self.assertGreaterEqual(error, 0.00005)

  def testDictionaryLearningWithSimulatedDataLshWithProjection(self):
    # The test case has been modified to smaller matrices, the original runtime
    # recorded as follows: for b.shape = [10000, 5000], c.shape = [5000, 650],
    # one iteration of DL takes about 10 minutes.
    # Generate random matrices a, b, and c.
    b = np.random.normal(0, 1, [1000, 500])
    sub_thresh_indices = (b <= 0.7)
    b[sub_thresh_indices] = 0
    b[:, 0] = np.ones(shape=b[:, 0].shape)
    c = np.random.normal(0, 1, [500, 65])
    a = np.matmul(b, c)

    [b_out, c_out] = dictionary_learning.dictionary_learning(
        a,
        row_percentage=0.5,
        col_percentage=0.3,
        n_iterations=1,
        use_lsh=True,
        use_projection=True,
        projection_dim=10,
        version_num=1)
    a_recovered = np.matmul(b_out, c_out)
    error = np.linalg.norm(a - a_recovered) / np.linalg.norm(a)

    # Expect that the approximation 'a_recovered' is not too bad.
    self.assertLessEqual(error, 0.98)

  def testDictionaryLearningWithSimulatedDataLshProjectionComparison(self):
    # Generate random matrices a, b, and c.
    b = np.random.normal(0, 1, [1000, 500])
    sub_thresh_indices = (b <= 0.7)
    b[sub_thresh_indices] = 0
    b[:, 0] = np.ones(shape=b[:, 0].shape)
    c = np.random.normal(0, 1, [500, 65])
    a = np.matmul(b, c)

    [b_out, c_out] = dictionary_learning.dictionary_learning(
        a,
        row_percentage=0.5,
        col_percentage=0.3,
        n_iterations=1,
        use_lsh=True,
        use_projection=True,
        projection_dim=10,
        version_num=1)
    a_recovered = np.matmul(b_out, c_out)
    error_with_projection = np.linalg.norm(a - a_recovered) / np.linalg.norm(a)

    [b_out_prime, c_out_prime] = dictionary_learning.dictionary_learning(
        a,
        row_percentage=0.5,
        col_percentage=0.3,
        n_iterations=1,
        use_lsh=True,
        version_num=1)
    a_recovered = np.matmul(b_out_prime, c_out_prime)
    error = np.linalg.norm(a - a_recovered) / np.linalg.norm(a)

    self.assertLessEqual(error, error_with_projection)


if __name__ == "__main__":
  absltest.main()
