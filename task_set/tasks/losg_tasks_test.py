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

"""Tests for task_set.tasks.losg_tasks."""
from task_set.tasks import family_test_utils
from task_set.tasks import losg_tasks
import tensorflow.compat.v1 as tf


class LOSGTasksTest(family_test_utils.TaskFamilyTestCase):

  def __init__(self, *args, **kwargs):
    super(LOSGTasksTest,
          self).__init__(losg_tasks.sample_losg_tasks_family_cfg,
                         losg_tasks.get_losg_tasks_family, *args, **kwargs)


if __name__ == "__main__":
  tf.test.main()
