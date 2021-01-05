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

# python3
"""Tests for task_set.tasks.fixed_language_modeling."""

from absl.testing import parameterized

from task_set import registry
from task_set.tasks import family_test_utils
from task_set.tasks.fixed import fixed_language_modeling  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf


class FixedMLPAETest(family_test_utils.SingleTaskTestCase):

  def test_right_number_of_tasks(self):
    task_names = registry.task_registry.get_all_fixed_config_names()
    self.assertLen(task_names, 5)

  @parameterized.parameters(registry.task_registry.get_all_fixed_config_names())
  def test_tasks(self, task_name):
    self.task_test(registry.task_registry.get_instance(task_name))


if __name__ == "__main__":
  tf.test.main()
