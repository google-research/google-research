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
        embedding_table, row_percentage=0.5, col_percentage=0.5)

    [code1, dictionary1] = dictionary_learning.dictionary_learning(
        embedding_table, row_percentage=0.8, col_percentage=0.8)

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


if __name__ == "__main__":
  absltest.main()
