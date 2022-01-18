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

"""Tests for ModelPruningHook."""

import tensorflow.compat.v1 as tf

from model_pruning.python import pruning_hook


class MockPruningObject(object):
  """Mock Pruning Object that has a run_update_step() function."""

  def __init__(self):
    self.logged_steps = []

  def run_update_step(self, session, global_step):  # pylint: disable=unused-argument
    self.logged_steps.append(global_step)


class PruningHookTest(tf.test.TestCase):

  def test_prune_after_session_creation(self):
    every_steps = 10
    pruning_obj = MockPruningObject()
    listener = pruning_hook.ModelPruningListener(pruning_obj)
    hook = pruning_hook.ModelPruningHook(every_steps=every_steps,
                                         listeners=[listener])
    mon_sess = tf.train.MonitoredSession(hooks=[hook])  # pylint: disable=unused-variable.
    self.evaluate(tf.global_variables_initializer())

    self.assertEqual(len(pruning_obj.logged_steps), 1)
    self.assertEqual(pruning_obj.logged_steps[0], 0)

  def test_prune_every_n_steps(self):
    every_steps = 10
    pruning_obj = MockPruningObject()

    with tf.Graph().as_default():
      listener = pruning_hook.ModelPruningListener(pruning_obj)
      hook = pruning_hook.ModelPruningHook(every_steps=every_steps,
                                           listeners=[listener])
      global_step = tf.train.get_or_create_global_step()
      train_op = tf.constant(0)
      global_step_increment_op = tf.assign_add(global_step, 1)
      with tf.train.MonitoredSession(tf.train.ChiefSessionCreator(),
                                     hooks=[hook]) as mon_sess:
        mon_sess.run(tf.global_variables_initializer())

        mon_sess.run(train_op)
        mon_sess.run(global_step_increment_op)
        # ModelPruningHook runs once after session creation, at step 0.
        self.assertEqual(len(pruning_obj.logged_steps), 1)
        self.assertEqual(pruning_obj.logged_steps[0], 0)

        for _ in range(every_steps-1):
          mon_sess.run(train_op)
          mon_sess.run(global_step_increment_op)

        self.assertEqual(len(pruning_obj.logged_steps), 2)
        self.assertSameElements(pruning_obj.logged_steps, [0, every_steps])

        for _ in range(every_steps-1):
          mon_sess.run(train_op)
          mon_sess.run(global_step_increment_op)

        self.assertEqual(len(pruning_obj.logged_steps), 2)
        self.assertSameElements(pruning_obj.logged_steps, [0, every_steps])


if __name__ == '__main__':
  tf.test.main()
