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

# Lint as: python3
"""Tests for soft sorting tensorflow layers."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from soft_sort.matrix_factorization import data
from soft_sort.matrix_factorization import nmf
from soft_sort.matrix_factorization import qmf
from soft_sort.matrix_factorization import qmfq
from soft_sort.matrix_factorization import training


class FactorizersTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self.num_individuals = 100
    self.num_features = 30
    self.rank = 6
    self.synthetizer = data.SyntheticData(
        num_individuals=self.num_individuals,
        num_features=self.num_features,
        low_rank=self.rank)
    self.matrix = self.synthetizer.make()

  @parameterized.parameters(
      [nmf.NMF(low_rank=6, num_iterations=20),
       qmf.QMF(low_rank=6, num_quantiles=4, epsilon=1e-2, batch_size=4),
       qmfq.QMFQ(low_rank=6, num_quantiles=4, epsilon=1e-2, num_nmf_updates=2),
       ])
  def test_nmf(self, factorizer):
    factorizer(self.matrix, epochs=2)
    self.assertAllGreater(factorizer.u, 0.0)
    self.assertAllGreater(factorizer.v, 0.0)
    self.assertEqual(factorizer.u.shape, (self.num_features, self.rank))
    self.assertEqual(factorizer.v.shape, (self.rank, self.num_individuals))

  def test_train(self):
    factorizer = nmf.NMF(low_rank=6, num_iterations=20)
    loop = training.TrainingLoop(
        None, data_loader=self.synthetizer, factorizer=factorizer)
    loop.run(steps=2)
    self.assertEqual(factorizer.u.shape, (self.num_features, self.rank))
    self.assertEqual(factorizer.v.shape, (self.rank, self.num_individuals))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
