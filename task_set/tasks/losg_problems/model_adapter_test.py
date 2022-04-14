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

"""Tests for learning.brain.models.learned_optimizer.problems.model_adapter."""

from task_set.tasks.losg_problems import model_adapter
import tensorflow.compat.v1 as tf


class ModelAdapterTest(tf.test.TestCase):

  def testModelAdapter(self):
    # A test problem that adds one parameter to one constant.
    def make_loss_and_init_fn():
      def make_loss():
        x = tf.get_variable("x", [])
        c = tf.get_variable("c", [], trainable=False)
        return x + c
      def make_init_fn(params):
        init_op = tf.group(*[tf.assign(p, 2.0) for p in params])
        def init_fn(sess):
          sess.run(init_op)
        return init_fn
      return make_loss, make_init_fn

    problem = model_adapter.ModelAdapter(make_loss_and_init_fn)
    # There's only one parameter in this problem.
    init_tensors = problem.init_tensors()
    self.assertEqual(1, len(init_tensors))

    init_objective = problem.objective(init_tensors)
    # Test replacement of variables with some constant.
    new_objective = problem.objective([tf.constant(0.0)])

    with self.test_session() as sess:
      problem.init_fn(sess)
      # init_fn will initialize all variables to 2.0
      self.assertEqual(2.0, init_tensors[0].eval())
      # 2 + 2 = 4
      self.assertEqual(4.0, init_objective.eval())
      # 2 + 0 = 2
      self.assertEqual(2.0, new_objective.eval())


if __name__ == "__main__":
  tf.test.main()
