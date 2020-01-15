# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Abstraction to manage how long many truncations.

As of now, this is used to manage out of graph execution.

A Truncation strategy has 2 functions:
unroll_n_steps()
should_do_truncation(learner)

When executed in a training loop,
First, ops created by should_do_truncation is executed. If this returns true,
one unroll of unroll_n_steps is run.
This repeats untill should_do_truncation returns False at which point
the learner is reset to an initial state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import sonnet as snt
import tensorflow as tf

nest = tf.contrib.framework.nest


class AbstractTruncationStrategy(snt.AbstractModule):
  """Interface to control how truncation works.

  In standard backprop through time, this would just return a fixed number for
  unroll_n_steps and true while training_steps < max length and false otherwise.

  We have found that sampling and or scheduling truncation lengths and how long
  to unroll increases performance. Different subclasses represent different
  strategies to do this.
  """

  def __init__(self, name="AbstratractTruncationStrategy"):
    super(AbstractTruncationStrategy, self).__init__(name=name)
    self()

  def _build(self):
    pass

  def unroll_n_steps(self):
    pass

  def should_do_truncation(self, learner):
    pass


@gin.configurable
class FixedLengthIncreasingTruncationStrategy(AbstractTruncationStrategy):
  """Truncation strategy used for experiments.

  Increase linearly, from `min_unroll_steps` to `truncation_length` over
  `increase_over` steps.

  If the loss diverges, stop the current truncation.
  """

  def __init__(
      self,
      truncation_length=10000,
      increase_over=20000,
      min_unroll_steps=10,
      name="FixedLengthIncreasingTruncationStrategy",
  ):
    super(FixedLengthIncreasingTruncationStrategy, self).__init__(name=name)
    self.increase_over = increase_over
    self.min_unroll_steps = min_unroll_steps
    self.truncation_length = truncation_length

  def unroll_n_steps(self, learner):
    # min_unroll --> truncation_length.
    # never go longer than truncation_length.
    training_step = learner.current_state().training_step

    gs = tf.train.get_global_step()

    # compute the linear interpolation
    ratio = tf.minimum(tf.to_float(gs) / float(self.increase_over), 1.)

    total_steps = self.min_unroll_steps * (1. -
                                           ratio) + self.truncation_length * (
                                               ratio)

    # fudge by 20%
    fudge = tf.random_uniform([], 0.8, 1.2)
    float_steps = total_steps * fudge
    candidate = tf.to_int32(float_steps)

    # clip step amount to never go above truncation length
    steps_left = self.truncation_length - training_step

    steps = tf.minimum(steps_left, candidate)
    return steps

  def should_do_truncation(self, learner):
    training_step = learner.current_state().training_step
    return tf.less(training_step, self.truncation_length)


@gin.configurable
class ConstantTruncationStrategy(AbstractTruncationStrategy):
  """Constant length unrolls."""

  def __init__(self,
               steps_per_unroll=10,
               max_steps=1000,
               name="ConstantTruncationStrategy"):
    super(ConstantTruncationStrategy, self).__init__(name=name)
    self.steps_per_unroll = steps_per_unroll
    self.max_steps = max_steps

  def unroll_n_steps(self, learner):
    steps_left = self.max_steps - learner.current_state().training_step
    candidate = tf.minimum(steps_left, self.steps_per_unroll)
    return candidate

  def should_do_truncation(self, learner):
    training_step = learner.current_state().training_step
    return tf.less(training_step, self.max_steps)
