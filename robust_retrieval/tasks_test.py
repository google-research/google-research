# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint-as: python3
"""Tests tasks."""

import numpy as np
import tensorflow as tf

from robust_retrieval.tasks import RobustRetrieval


def _sigmoid(x):
  return 1. / (1 + np.exp(-x))


class TasksTest(tf.test.TestCase):

  def test_robust_retrieval_task_with_loss_dro(self):
    query = tf.constant([[1, 2, 3], [2, 3, 4]], dtype=tf.float32)
    candidate = tf.constant([[1, 1, 1], [1, 1, 0]], dtype=tf.float32)

    task = RobustRetrieval(
        group_labels=['a', 'b'],
        group_loss_init=[0.0, 0.0],
        group_metric_init=[0.0, 0.0],
        group_weight_init=[1.0, 1.0],
        group_reweight_strategy='loss-dro',
        streaming_group_loss=False,
        dro_temperature=1.0,
    )

    # All_pair_scores: [[6, 3], [9, 5]].
    # Normalized logits: [[3, 0], [4, 0]].
    first_sample_loss = -np.log(_sigmoid(3.0))
    second_sample_loss = -np.log(1 - _sigmoid(4.0))
    group_loss = np.array([first_sample_loss, second_sample_loss])
    group_weight = np.exp(group_loss) / np.sum(np.exp(group_loss))
    expected_loss = np.sum(group_loss * group_weight) * 2

    loss = task(
        query_embeddings=query,
        candidate_embeddings=candidate,
        group_identity=['a', 'b'],
        step_count=1)

    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, loss)


if __name__ == '__main__':
  tf.test.main()
