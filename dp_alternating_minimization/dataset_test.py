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

"""Tests for dataset."""
import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from dp_alternating_minimization import dataset


def _construct_input_matrix(indices0, indices1, values):
  indices0 = np.array(list(indices0))
  indices1 = np.array(list(indices1))
  values = np.array(list(values))
  return dataset.InputMatrix(
      indices=np.array([[x, y] for x, y in zip(indices0, indices1)]),
      values=values,
      weights=None,
      row_reg=None,
      num_rows=indices0.max() + 1,
      num_cols=indices1.max() + 1)


class DatasetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (100, 20),
      (100, 1),
      (100, 100),
      (100, 27),
      )
  def test_batch_gd(self, n, batch_size):
    input_data = _construct_input_matrix(np.arange(n), np.arange(n) * 10,
                                         np.arange(n) / 10)
    d = input_data.batch_gd(user_axis=0, batch_size=batch_size)
    d = iter(d)
    for _ in range(n):
      ids, values, weights = next(d)
      self.assertLen(ids[dataset.OUTPUT_USER_KEY], batch_size)
      self.assertLen(ids[dataset.OUTPUT_ITEM_KEY], batch_size)
      self.assertLen(values, batch_size)
      self.assertLen(weights, batch_size)

  @parameterized.parameters(
      (3, 2),
      (2, 3),
      (1, 10),
      (10, 1),
      )
  def test_batch_gd_by_user(self, num_examples_per_user, num_users_per_batch):
    data = itertools.chain(*[[(i, i, i)] * (i + 1) for i in range(10)])
    input_matrix = _construct_input_matrix(*zip(*data))
    d = input_matrix.batch_gd_by_user(
        0, num_examples_per_user, num_users_per_batch)

    def _check_batch(batch):
      """Checks if one user's data is consecutive in the batch."""
      batch_size = num_examples_per_user * num_users_per_batch
      ids, values, weights = batch
      self.assertSetEqual(
          set(ids.keys()),
          set([dataset.OUTPUT_USER_KEY, dataset.OUTPUT_ITEM_KEY]))
      users = ids[dataset.OUTPUT_USER_KEY].numpy()
      items = ids[dataset.OUTPUT_ITEM_KEY].numpy()
      ratings = values.numpy()
      weights = weights.numpy()
      self.assertLen(users, batch_size)
      self.assertLen(items, batch_size)
      self.assertLen(ratings, batch_size)
      self.assertLen(weights, batch_size)
      # As we have set user = item = rating.
      self.assertAllClose(users, items)
      self.assertAllClose(users, ratings)
      self.assertAllClose(weights, np.ones(batch_size))
      user_ids = []
      for i in range(0, batch_size, num_examples_per_user):
        self.assertAllEqual(users[i:i+num_examples_per_user],
                            users[i] * np.ones(num_examples_per_user))
        user_ids.append(users[i])
      self.assertLen(set(user_ids), num_users_per_batch)

    d = iter(d)
    batch1 = next(d)
    batch2 = next(d)
    batch3 = next(d)
    batch4 = next(d)
    batch5 = next(d)

    _check_batch(batch1)
    _check_batch(batch2)
    _check_batch(batch3)
    _check_batch(batch4)
    _check_batch(batch5)

  def test_batch_gd_by_user_users_per_batch_large(self):
    data = itertools.chain(*[[(i, i, i)] * (i + 1) for i in range(10)])
    input_matrix = _construct_input_matrix(*zip(*data))
    self.assertRaises(ValueError,
                      input_matrix.batch_gd_by_user,
                      user_axis=0,
                      num_examples_per_user=100,
                      # num_users_per_batch exceeds total users
                      num_users_per_batch=11)


if __name__ == "__main__":
  tf.test.main()
