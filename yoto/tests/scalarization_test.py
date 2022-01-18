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

"""Tests for the scalarization optimizers."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from yoto.optimizers import scalarization
import yoto.problems as problems


class ProblemWithConstantLosses(problems.Problem):

  def __init__(self, losses_values):
    self._losses_values = losses_values
    self._dummy = tf.Variable(0.)

  def losses_and_metrics(self, inputs, inputs_extra=None, training=False):
    """Map the inputs to a {loss_name: loss_tensor} dictionary."""
    del inputs
    del inputs_extra
    losses = {key: value + 0 * self._dummy
              for key, value in self._losses_values.items()}
    return losses, {}

  @property
  def losses_keys(self):
    return tuple(sorted(self._losses_values.keys()))

  def initialize_model(self):
    pass

  def module_spec(self):
    pass


class LinearlyScalarizedOptimizerTest(parameterized.TestCase,
                                      tf.test.TestCase):

  def test_check_weighted_value_on_constant_losses(self):
    weights = {"a": tf.constant(0.5),
               "b": tf.constant(0.3),
               "c": tf.constant(0.4)}
    losses = {"a": -15, "b": .4, "c": .3}
    optimizer = scalarization.LinearlyScalarizedOptimizer(
        problem=ProblemWithConstantLosses(losses), weights=weights)
    loss, _ = optimizer.compute_train_loss_and_update_op(
        inputs=dict(), base_optimizer=tf.train.GradientDescentOptimizer(0.))
    with self.cached_session() as session:
      session.run(tf.initializers.global_variables())
    self.assertAllClose(loss,
                        sum(weights[key] * losses[key] for key in weights))

  def test_exception_thrown_when_weights_is_of_invalid_type(self):
    losses = {"a": -15, "b": .4, "c": .3}
    # Should fail as `weights` is neither a dict nor in the enum.
    with self.assertRaises(TypeError):
      _ = scalarization.LinearlyScalarizedOptimizer(
          problem=ProblemWithConstantLosses(losses), weights=123)

  def test_throws_exception_when_weights_key_is_missing(self):
    losses = {"a": -15, "b": .4, "c": .3}
    weights = {"a": tf.constant(0.5),
               "b": tf.constant(0.3)}  # Misses the key "c".
    with self.assertRaises(ValueError):
      _ = scalarization.LinearlyScalarizedOptimizer(
          problem=ProblemWithConstantLosses(losses), weights=weights)

if __name__ == "__main__":
  tf.disable_eager_execution()
  tf.test.main()
