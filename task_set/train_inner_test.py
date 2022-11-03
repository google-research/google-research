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

"""Tests for task_set.train_inner."""
import json
import os
import tempfile
import numpy as np

from task_set import datasets
from task_set import train_inner
from task_set.tasks import base
import tensorflow.compat.v1 as tf


class DummyTask(base.BaseTask):
  """Dummy task used for tests."""

  def call_split(self, params, split, with_metrics=False):
    r = tf.random_normal(shape=[], dtype=tf.float32)
    offset = {
        datasets.Split.TRAIN: 1.0,
        datasets.Split.VALID_INNER: 2.0,
        datasets.Split.VALID_OUTER: 3.0,
        datasets.Split.TEST: 4.0,
    }
    loss = offset[split] + r

    if with_metrics:
      return loss, {"metric": -1 * loss}
    else:
      return loss

  def get_batch(self, split):
    return None

  def current_params(self):
    return {}

  def gradients(self, loss):
    return {}

  def initial_params(self):
    return {}

  def get_variables(self):
    return []


class TrainInnerTest(tf.test.TestCase):

  def test_compute_averaged_loss(self):
    task = DummyTask()
    params = task.initial_params()
    losses, _ = train_inner.compute_averaged_loss(
        task, params, num_batches=100, with_metrics=False)

    with self.test_session() as sess:
      all_np_losses = []
      for _ in range(10):
        all_np_losses.append(sess.run(losses))

    tr, vai, vao, te = zip(*all_np_losses)
    # We are averaging over 100 with 10 replications evaluatons.
    # This means the std. error of the mean should be 1/sqrt(1000) or 0.03.
    # We use a threshold of 0.15, corresponding to a 5-sigma test.
    self.assertNear(np.mean(tr), 1.0, 0.15)
    self.assertNear(np.mean(vai), 2.0, 0.15)
    self.assertNear(np.mean(vao), 3.0, 0.15)
    self.assertNear(np.mean(te), 4.0, 0.15)

    # ensure that each sample is also different.
    self.assertLess(1e-5, np.var(tr), 0.5)
    self.assertLess(1e-5, np.var(vai), 0.5)
    self.assertLess(1e-5, np.var(vao), 0.5)
    self.assertLess(1e-5, np.var(te), 0.5)

    losses, metrics = train_inner.compute_averaged_loss(
        task, params, num_batches=100, with_metrics=True)
    tr_metrics, vai_metrics, vao_metrics, te_metrics = metrics
    with self.test_session() as sess:
      # this std. error is 1/sqrt(100), or 0.1. 5 std out is 0.5
      self.assertNear(sess.run(tr_metrics["metric"]), -1.0, 0.5)
      self.assertNear(sess.run(vai_metrics["metric"]), -2.0, 0.5)
      self.assertNear(sess.run(vao_metrics["metric"]), -3.0, 0.5)
      self.assertNear(sess.run(te_metrics["metric"]), -4.0, 0.5)

  def test_train(self):
    tmp_dir = tempfile.mkdtemp()

    # TODO(lmetz) when toy tasks are done, switch this away from an mlp.
    train_inner.train(
        tmp_dir,
        task_name="mlp_family_seed12",
        optimizer_name="adam8p_wide_grid_seed21",
        training_steps=10,
        eval_every_n=5)

    with tf.gfile.Open(os.path.join(tmp_dir, "result")) as f:
      result_data = json.loads(f.read())

    self.assertEqual(len(result_data), 3)
    # 4 losses logged out per timestep
    self.assertEqual(len(result_data["5"]), 4)

    with tf.gfile.Open(os.path.join(tmp_dir, "time_per_step")) as f:
      time_per_step_data = json.loads(f.read())

    self.assertIn("mean_last_half", time_per_step_data)
    self.assertIn("mean_time", time_per_step_data)
    self.assertIn("median_time", time_per_step_data)


if __name__ == "__main__":
  tf.test.main()
