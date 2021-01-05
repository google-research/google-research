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

"""Tests for task_set.registry."""

from task_set import registry
# pylint: disable=unused-import
from task_set.optimizers import adam
from task_set.tasks import mlp
# pylint: enable=unused-import
import tensorflow.compat.v1 as tf


class RegistryTest(tf.test.TestCase):

  def test_optimizer_registry(self):
    optimizer_instance = registry.optimizers_registry.get_instance(
        "adam_lr_-5.00")
    loss = tf.get_variable(shape=[], dtype=tf.float32, name="var")
    _ = optimizer_instance.minimize(loss)

  def test_task_registry(self):
    task_instance = registry.task_registry.get_instance("mlp_family_seed10")
    self.assertEqual(task_instance.name, "mlp_family_seed10")


if __name__ == "__main__":
  tf.test.main()
