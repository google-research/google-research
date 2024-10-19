# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for task_set.tasks.conv_fc."""
from task_set.tasks import conv_fc
from task_set.tasks import family_test_utils
import tensorflow.compat.v1 as tf


class ConvFCTest(family_test_utils.TaskFamilyTestCase):

  def __init__(self, *args, **kwargs):
    super(ConvFCTest,
          self).__init__(conv_fc.sample_conv_fc_family_cfg,
                         conv_fc.get_conv_fc_family, *args, **kwargs)


if __name__ == "__main__":
  tf.test.main()
