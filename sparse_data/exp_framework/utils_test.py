# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for utils.py in the exp_framework module."""

from absl.testing import absltest
from sparse_data.exp_framework import utils

NUM_REPLICATE = 20
SEED = 49 + 32 + 67 + 111 + 114 + 32 + 49 + 51 + 58 + 52 + 45 + 56


class TestUtils(absltest.TestCase):

  def test_generate_param_configs(self):
    param_grid = {'a': [1, 2, 3], 'b': [4, 5], 'c': [6]}
    exp_param_list = [{
        'a': 1,
        'b': 4,
        'c': 6
    }, {
        'a': 1,
        'b': 5,
        'c': 6
    }, {
        'a': 2,
        'b': 4,
        'c': 6
    }, {
        'a': 2,
        'b': 5,
        'c': 6
    }, {
        'a': 3,
        'b': 4,
        'c': 6
    }, {
        'a': 3,
        'b': 5,
        'c': 6
    }]

    # test exhaustive grid search
    num_search = len(exp_param_list) + 1
    param_list = utils.generate_param_configs(
        param_grid, num_iteration=num_search)
    param_list = sorted(
        param_list, key=lambda x: str(x['a']) + str(x['b']) + str(['c']))
    self.assertEqual(param_list, exp_param_list)

    # test random search
    num_search = len(exp_param_list) - 1
    param_list = utils.generate_param_configs(
        param_grid, num_iteration=num_search)
    for param_dict in param_list:
      for k, v in param_dict.items():
        self.assertIn(v, param_grid[k])
    self.assertLen(param_list, num_search)

    # test reproducibility
    num_search = 20
    param_grid = {'a': range(10), 'b': range(10, 20), 'c': range(30, 40)}
    old_param_list = utils.generate_param_configs(
        param_grid, num_iteration=num_search, seed=SEED)

    for _ in range(NUM_REPLICATE):
      param_list = utils.generate_param_configs(
          param_grid, num_iteration=num_search, seed=SEED)
      self.assertEqual(param_list, old_param_list)
      old_param_list = param_list


if __name__ == '__main__':
  absltest.main()
