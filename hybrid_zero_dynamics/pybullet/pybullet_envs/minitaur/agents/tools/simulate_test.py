# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the simulation operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google3.robotics.reinforcement_learning.agents import tools


class SimulateTest(tf.test.TestCase):

  def test_done_automatic(self):
    batch_env = self._create_test_batch_env((1, 2, 3, 4))
    algo = tools.MockAlgorithm(batch_env)
    done, _, _ = tools.simulate(batch_env, algo, log=False, reset=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([True, False, False, False], sess.run(done))
      self.assertAllEqual([True, True, False, False], sess.run(done))
      self.assertAllEqual([True, False, True, False], sess.run(done))
      self.assertAllEqual([True, True, False, True], sess.run(done))

  def test_done_forced(self):
    reset = tf.placeholder_with_default(False, ())
    batch_env = self._create_test_batch_env((2, 4))
    algo = tools.MockAlgorithm(batch_env)
    done, _, _ = tools.simulate(batch_env, algo, False, reset)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([False, False], sess.run(done))
      self.assertAllEqual([False, False], sess.run(done, {reset: True}))
      self.assertAllEqual([True, False], sess.run(done))
      self.assertAllEqual([False, False], sess.run(done, {reset: True}))
      self.assertAllEqual([True, False], sess.run(done))
      self.assertAllEqual([False, False], sess.run(done))
      self.assertAllEqual([True, True], sess.run(done))

  def test_reset_automatic(self):
    batch_env = self._create_test_batch_env((1, 2, 3, 4))
    algo = tools.MockAlgorithm(batch_env)
    done, _, _ = tools.simulate(batch_env, algo, log=False, reset=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(10):
        sess.run(done)
    self.assertAllEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], batch_env[0].steps)
    self.assertAllEqual([2, 2, 2, 2, 2], batch_env[1].steps)
    self.assertAllEqual([3, 3, 3, 1], batch_env[2].steps)
    self.assertAllEqual([4, 4, 2], batch_env[3].steps)

  def test_reset_forced(self):
    reset = tf.placeholder_with_default(False, ())
    batch_env = self._create_test_batch_env((2, 4))
    algo = tools.MockAlgorithm(batch_env)
    done, _, _ = tools.simulate(batch_env, algo, False, reset)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(done)
      sess.run(done, {reset: True})
      sess.run(done)
      sess.run(done, {reset: True})
      sess.run(done)
      sess.run(done)
      sess.run(done)
    self.assertAllEqual([1, 2, 2, 2], batch_env[0].steps)
    self.assertAllEqual([1, 2, 4], batch_env[1].steps)

  def _create_test_batch_env(self, durations):
    envs = []
    for duration in durations:
      env = tools.MockEnvironment(
          observ_shape=(2, 3), action_shape=(3,),
          min_duration=duration, max_duration=duration)
      env = tools.wrappers.ConvertTo32Bit(env)
      envs.append(env)
    batch_env = tools.BatchEnv(envs, blocking=True)
    batch_env = tools.InGraphBatchEnv(batch_env)
    return batch_env


if __name__ == '__main__':
  tf.test.main()
