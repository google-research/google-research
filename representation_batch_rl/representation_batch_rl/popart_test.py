# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import tensorflow.compat.v2 as tf

from representation_batch_rl.representation_batch_rl import popart
from representation_batch_rl.representation_batch_rl import running_statistics
from representation_batch_rl.representation_batch_rl import test_utils


def setUpModule():
  # We want our tests to run on several devices with a mirrored strategy.
  test_utils.simulate_two_devices()


class PopArtTest(test_utils.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup(self, beta):
    """Sets up the reward normalizer and a distribution strategy to run."""
    reward_normalizer = popart.PopArt(running_statistics.EMAMeanStd(beta))
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      reward_normalizer.init()
    return reward_normalizer, strategy

  def _update_normalization_statistics(self, reward_normalizer, strategy,
                                       targets):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(targets)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(reward_normalizer.update_normalization_statistics)
    strategy.run(f, (distributed_values,))

  def test_normalization(self):
    """Tests that the normalization works."""
    reward_normalizer, strategy = self._setup(1.)

    # Apply update.
    targets = tf.range(20, dtype=tf.float32)
    self._update_normalization_statistics(reward_normalizer, strategy, targets)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = reward_normalizer.normalize_target(targets)
    self.assertAllClose(tf.reduce_mean(normalized), 0.)
    self.assertAllClose(tf.math.reduce_std(normalized), 1.)

  def test_variables(self):
    """Tests tha we have the correct number of variables."""
    reward_normalizer, _ = self._setup(.5)
    self.assertLen(reward_normalizer.variables, 4)
    self.assertLen(reward_normalizer.trainable_variables, 2)

  def test_invariance(self):
    """Tests that the update keeps the convolution invariant."""
    reward_normalizer, strategy = self._setup(.5)
    targets = tf.range(20, dtype=tf.float32)

    # Run the convolution of `correct_prediction(...)` and
    # `unnormalize_prediction(...)` before the update of the statistics.
    with strategy.scope():
      before = reward_normalizer.correct_prediction(targets)
      before = reward_normalizer.unnormalize_prediction(before)

    self._update_normalization_statistics(reward_normalizer, strategy, targets)

    # Run the convolution of `correct_prediction(...)` and
    # `unnormalize_prediction(...)` after the update of the statistics.
    with strategy.scope():
      after = reward_normalizer.correct_prediction(targets)
      after = reward_normalizer.unnormalize_prediction(after)

    # Verify that update is correct.
    self.assertAllClose(before, after)

  def test_invertible(self):
    """Tests that the normalization is invertible."""
    reward_normalizer, strategy = self._setup(.5)

    # Apply update.
    targets = tf.range(20, dtype=tf.float32)
    self._update_normalization_statistics(reward_normalizer, strategy, targets)

    # Verify that update is correct.
    with strategy.scope():
      normalized = reward_normalizer.normalize_target(targets)
      inverted = reward_normalizer.unnormalize_prediction(normalized)

    self.assertAllClose(targets, inverted)

  def test_advantage(self):
    """Tests that the advantage normalization works."""
    reward_normalizer, strategy = self._setup(.5)

    # Apply update.
    targets = tf.range(20, dtype=tf.float32)
    self._update_normalization_statistics(reward_normalizer, strategy, targets)

    # Verify that normalization works correctly by seeing if the unnormalized
    # advantage of normalized targets is the same as the normalized advantage
    # of the unnormalized targets.
    advantage = targets - tf.reverse(targets, [0])
    with strategy.scope():
      normalized_advantage = reward_normalizer.normalize_advantage(advantage)
      normalized_targets = reward_normalizer.normalize_target(targets)

    self.assertAllClose(normalized_advantage,
                        normalized_targets - tf.reverse(normalized_targets, [0])
                       )


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
