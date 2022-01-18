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

"""Tests for learning.brain.models.learned_optimizer.problems.problem_generator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from task_set.tasks.losg_problems import problem_generator as pg
from task_set.tasks.losg_problems import problem_spec
import tensorflow.compat.v1 as tf


class ProblemGeneratorTest(tf.test.TestCase):

  def testProblem(self):
    param_shapes = [(5, 1), (2, 2)]
    random_seed = 200
    noise_stdev = 1.0
    problem = pg.Problem(param_shapes, random_seed, noise_stdev)

    init = problem.init_tensors()
    self.assertLen(init, len(param_shapes))
    for i, ps in enumerate(param_shapes):
      self.assertEqual(list(ps), init[i].get_shape().as_list())

    init = problem.init_variables()
    self.assertLen(init, len(param_shapes))
    for i, ps in enumerate(param_shapes):
      self.assertSequenceEqual(ps, init[i].get_shape())
      self.assertIsInstance(init[i], tf.Variable)

  def testProblemGradients(self):
    param_shapes = [(1, 1)]
    random_seed = 200
    noise_stdev = 1.0
    problem = pg.Problem(param_shapes, random_seed, noise_stdev)

    x = tf.constant(2.)
    y = tf.constant(20.)
    parameters = [x, y]
    objective = x**2 + y**2
    grads = problem.gradients(objective, parameters)
    self.assertLen(grads, len(parameters))
    self.assertNotEqual(grads[0], grads[1])

  def testSparseProblem_neverZero(self):
    zero_prob = 0.0
    problem = pg.SparseProblem(
        problem_spec.Spec(pg.Quadratic, (5,), {}), zero_prob)
    self.assertEqual(zero_prob, problem.zero_prob)

    parameters = problem.init_tensors(seed=1234)
    objective = problem.objective(parameters)
    gradients = problem.gradients(objective, parameters)
    self.assertLen(gradients, 1)

    with self.test_session() as sess:
      self.assertTrue(all(sess.run(gradients[0])))

  def testSparseProblem_alwaysZero(self):
    zero_prob = 1.0
    problem = pg.SparseProblem(
        problem_spec.Spec(pg.Quadratic, (5,), {}), zero_prob)
    self.assertEqual(zero_prob, problem.zero_prob)

    parameters = problem.init_tensors(seed=1234)
    objective = problem.objective(parameters)
    gradients = problem.gradients(objective, parameters)
    self.assertLen(gradients, 1)

    with self.test_session() as sess:
      self.assertFalse(any(sess.run(gradients[0])))

  def testSparseProblem_someProbability(self):
    tf.set_random_seed(1234)
    zero_prob = 0.5
    problem = pg.SparseProblem(
        problem_spec.Spec(pg.Quadratic, (5,), {}), zero_prob)
    self.assertEqual(zero_prob, problem.zero_prob)

    parameters = problem.init_tensors(seed=1234)
    objective = problem.objective(parameters)
    gradients = problem.gradients(objective, parameters)
    self.assertLen(gradients, 1)

    with self.test_session() as sess:
      self.assertTrue(any(sess.run(gradients[0])))
      self.assertFalse(all(sess.run(gradients[0])))


if __name__ == "__main__":
  tf.test.main()
