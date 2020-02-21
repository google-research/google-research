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

# python3
"""Tests for task_set.tasks.mlp."""
from task_set.tasks import family_test_utils
from task_set.tasks import mlp
import tensorflow.compat.v1 as tf


class MlpTest(family_test_utils.TaskFamilyTestCase):

  def __init__(self, *args, **kwargs):
    super(MlpTest, self).__init__(mlp.sample_mlp_family_cfg, mlp.get_mlp_family,
                                  *args, **kwargs)


if __name__ == "__main__":
  tf.test.main()
