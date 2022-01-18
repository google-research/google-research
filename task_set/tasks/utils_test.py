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

"""Tests for task_set.tasks.utils."""
import numpy as np
from task_set.tasks import utils
import tensorflow.compat.v1 as tf


class UtilsTest(tf.test.TestCase):

  def test_sample_get_activation(self):
    rng = np.random.RandomState(123)
    sampled_acts = []
    num = 4000
    for _ in range(num):
      aname = utils.sample_activation(rng)
      sampled_acts.append(aname)
      # smoke test to ensure graph builds
      out = utils.get_activation(aname)(tf.constant(1.0))
      self.assertIsInstance(out, tf.Tensor)

    uniques, counts = np.unique(sampled_acts, return_counts=True)
    counts_map = {str(u): c for u, c in zip(uniques, counts)}
    # 16 is the total sum of unnormalized probs
    amount_per_n = num / float(16)
    self.assertNear(counts_map["relu"], amount_per_n * 6, 40)
    self.assertNear(counts_map["tanh"], amount_per_n * 3, 40)
    self.assertNear(counts_map["swish"], amount_per_n, 40)

  def test_sample_get_initializer(self):
    rng = np.random.RandomState(123)
    sampled_init = []
    num = 3000
    for _ in range(num):
      init_name, args = utils.sample_initializer(rng)
      sampled_init.append(init_name)
      # smoke test to ensure graph builds
      out = utils.get_initializer((init_name, args))((10, 10))
      self.assertIsInstance(out, tf.Tensor)

    uniques, counts = np.unique(sampled_init, return_counts=True)
    counts_map = {str(u): c for u, c in zip(uniques, counts)}
    # 13 is the total sum of unnormalized probs
    amount_per_n = num / float(13)
    self.assertNear(counts_map["he_normal"], amount_per_n * 2, 40)
    self.assertNear(counts_map["orthogonal"], amount_per_n, 40)
    self.assertNear(counts_map["glorot_normal"], amount_per_n * 2, 40)


if __name__ == "__main__":
  tf.test.main()
