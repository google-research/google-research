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
"""Tests for task_set.tasks.base."""

import sonnet as snt

from task_set import datasets
from task_set.tasks import base
import tensorflow.compat.v1 as tf


class BaseTest(tf.test.TestCase):

  def test_dataset_model_task(self):

    def one_dataset_fn(scale):
      dataset = tf.data.Dataset.from_tensor_slices([scale * tf.ones([10, 2])])
      return dataset.repeat()

    all_datasets = datasets.Datasets(
        one_dataset_fn(1), one_dataset_fn(2), one_dataset_fn(3),
        one_dataset_fn(4))

    def fn(inp):
      out = snt.Linear(10, initializers={"w": tf.initializers.ones()})(inp)
      loss = tf.reduce_mean(out)
      return loss

    task = base.DatasetModelTask(lambda: snt.Module(fn), all_datasets)

    param_dict = task.initial_params()

    self.assertLen(param_dict, 2)

    with self.test_session():
      train_loss = task.call_split(param_dict, datasets.Split.TRAIN)
      self.assertNear(train_loss.eval(), 2.0, 1e-8)
      test_loss = task.call_split(param_dict, datasets.Split.TEST)
      self.assertNear(test_loss.eval(), 8.0, 1e-8)
      grads = task.gradients(train_loss, param_dict)
      np_grad = grads["BaseModel/fn/linear/w"].eval()
      self.assertNear(np_grad[0, 0], 0.1, 1e-5)


if __name__ == "__main__":
  tf.test.main()
