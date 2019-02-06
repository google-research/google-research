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

"""Tests for random_forest.py in the exp_framework module."""

from absl.testing import absltest
from sparse_data.exp_framework import random_forest
from sparse_data.exp_framework.dnn_test import TestModel


class TestRandomForest(TestModel):

  def setUp(self):
    super(TestRandomForest, self).setUp()
    self.submodule = random_forest


if __name__ == '__main__':
  absltest.main()
