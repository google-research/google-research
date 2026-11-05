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

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from representation_batch_rl.representation_batch_rl import running_statistics
from representation_batch_rl.representation_batch_rl import test_utils


def setUpModule():
  # We want our tests to run on several devices with a mirrored strategy.
  test_utils.simulate_two_devices()


class EMAMeanStdTest(test_utils.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup_normalizer(self, beta, size):
    """Sets up the input normalizer and a distribution strategy to run."""
    normalizer = running_statistics.EMAMeanStd(beta=beta)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      normalizer.init(size)
    return normalizer, strategy

  def _update(self, normalizer, strategy, tensor):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(tensor)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(normalizer.update)
    strategy.run(f, (distributed_values,))

  def test_moment_update(self):
    """Tests that the normalization moments are correctly updated."""
    normalizer, strategy = self._setup_normalizer(.5, 10)

    # Apply update.
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    self._update(normalizer, strategy, tensor)

    # Verify that update is correct.
    target_first_moment = 0.5 * tf.reduce_mean(tensor, 0)
    self.assertAllClose(normalizer.first_moment, target_first_moment)
    target_second_moment = 0.5 + 0.5 * tf.reduce_mean(tensor**2, 0)
    self.assertAllClose(normalizer.second_moment, target_second_moment)

  def test_normalization(self):
    """Tests that the normalization works."""
    normalizer, strategy = self._setup_normalizer(1., 10)

    # Apply update.
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    self._update(normalizer, strategy, tensor)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)

    mean = tf.reduce_mean(normalized, 0)
    std = tf.math.reduce_std(normalized, 0)
    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  def test_variables(self):
    """Tests that we have the correct number of variables."""
    normalizer, _ = self._setup_normalizer(.5, 10)
    self.assertLen(normalizer.variables, 2)
    self.assertEmpty(normalizer.trainable_variables, 0)

  def test_invertible(self):
    """Tests that the normalization is invertible."""
    normalizer, strategy = self._setup_normalizer(.5, 10)
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Apply update.
    self._update(normalizer, strategy, tensor)

    # Verify that update is correct.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)
      inverted = normalizer.unnormalize(normalized)

    self.assertAllClose(tensor, inverted)


class AverageMeanStdTest(test_utils.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup_normalizer(self, size, std_min_value=None):
    """Sets up the input normalizer and a distribution strategy to run."""
    args = {}
    if std_min_value is not None:
      args.update(std_min_value=std_min_value)
    normalizer = running_statistics.AverageMeanStd(**args)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      normalizer.init(size)
    return normalizer, strategy

  def _update(self, normalizer, strategy, tensor):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(tensor)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(normalizer.update)
    strategy.run(f, (distributed_values,))

  def test_normalization(self):
    """Tests that the normalization works."""
    normalizer, strategy = self._setup_normalizer(10)

    # Apply two updates to make sure that everything works as intended.
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    tensor1, tensor2 = tf.split(tensor, 2, axis=0)
    self._update(normalizer, strategy, tensor1)
    self._update(normalizer, strategy, tensor2)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)

    mean = tf.reduce_mean(normalized, 0)
    std = tf.math.reduce_std(normalized, 0)
    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  def test_variables(self):
    """Tests that we have the correct number of variables."""
    normalizer, _ = self._setup_normalizer(10)
    self.assertLen(normalizer.variables, 4)
    self.assertEmpty(normalizer.trainable_variables, 0)

  def test_invertible(self):
    """Tests that the normalization is invertible."""
    normalizer, strategy = self._setup_normalizer(10)
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Apply update.
    self._update(normalizer, strategy, tensor)

    # Verify that update is correct.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)
      inverted = normalizer.unnormalize(normalized)

    self.assertAllClose(tensor, inverted)

  def test_init(self):
    """Tests that the mean and standard deviation are initialized to 0 and 1."""
    normalizer, _ = self._setup_normalizer(10)
    mean, std = normalizer.get_mean_std()

    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  def test_init2(self):
    """Tests that the mean and standard deviation are initialized to 0 and 1."""
    normalizer, _ = self._setup_normalizer(10, std_min_value=0.1)
    mean, std = normalizer.get_mean_std()

    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))


class TwoLevelAverageMeanStdTest(test_utils.TestCase, parameterized.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup_normalizer(self, size, buffer_size=1e5):
    """Sets up the input normalizer and a distribution strategy to run."""
    normalizer = running_statistics.TwoLevelAverageMeanStd(
        buffer_size=buffer_size)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      normalizer.init(size)
    return normalizer, strategy

  def _update(self, normalizer, strategy, tensor):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(tensor)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(normalizer.update)
    strategy.run(f, (distributed_values,))

  def test_normalization(self):
    """Tests that the normalization works."""
    normalizer, strategy = self._setup_normalizer(10)

    # Apply two updates to make sure that everything works as intended.
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))
    tensor1, tensor2 = tf.split(tensor, 2, axis=0)
    self._update(normalizer, strategy, tensor1)
    self._update(normalizer, strategy, tensor2)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)

    mean = tf.reduce_mean(normalized, 0)
    std = tf.math.reduce_std(normalized, 0)
    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  @parameterized.parameters(
      {'num_updates': 3},
      {'num_updates': 9},
      {'num_updates': 10},
      {'num_updates': 11},
      {'num_updates': 19},
      {'num_updates': 20},
      {'num_updates': 21},
      {'num_updates': 25},
  )
  def test_normalization_two_levels(self, num_updates):
    """Tests normalization when the stats buffer is reset multiple times."""
    tf.print('Num updates:', num_updates)
    normalizer, strategy = self._setup_normalizer(10, buffer_size=10)
    tf.random.set_seed(123)
    tensors = [tf.random.normal([10, 10]) for _ in range(num_updates)]

    for tensor in tensors:
      self._update(normalizer, strategy, tensor)

    # Verify that normalization works correctly.
    with strategy.scope():
      normalized = normalizer.normalize(tf.reshape(tf.stack(tensors), [-1, 10]))

    mean = tf.reduce_mean(normalized, 0)
    std = tf.math.reduce_std(normalized, 0)
    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))

  def test_variables(self):
    """Tests that we have the correct number of variables."""
    normalizer, _ = self._setup_normalizer(10)
    self.assertLen(normalizer.variables, 8)
    self.assertEmpty(normalizer.trainable_variables, 0)

  def test_invertible(self):
    """Tests that the normalization is invertible."""
    normalizer, strategy = self._setup_normalizer(10)
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Apply update.
    self._update(normalizer, strategy, tensor)

    # Verify that update is correct.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)
      inverted = normalizer.unnormalize(normalized)

    self.assertAllClose(tensor, inverted)

  def test_init(self):
    """Tests that the mean and standard deviation are initialized to 0 and 1."""
    normalizer, _ = self._setup_normalizer(10)
    mean, std = normalizer.get_mean_std()

    self.assertAllClose(mean, tf.zeros_like(mean))
    self.assertAllClose(std, tf.ones_like(std))


class FixedMeanStdTest(test_utils.TestCase):
  # We manually enter a strategy scope in our tests.
  ENTER_PRIMARY_DEVICE = False

  def _setup_normalizer(self, size, mean, std):
    """Sets up the input normalizer and a distribution strategy to run."""
    normalizer = running_statistics.FixedMeanStd(mean, std)
    strategy = test_utils.create_distribution_strategy(
        use_tpu=self.primary_device == 'TPU')
    self.assertEqual(strategy.num_replicas_in_sync, 2)
    with strategy.scope():
      normalizer.init(size)
    return normalizer, strategy

  def _update(self, normalizer, strategy, tensor):
    """Updates the normalization statistics via the strategy."""
    dataset = tf.data.Dataset.from_tensors(tensor)
    dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    distributed_values = next(dataset_iterator)
    f = tf.function(normalizer.update)
    strategy.run(f, (distributed_values,))

  def test_normalization(self):
    """Tests that the normalization works."""
    normalizer, strategy = self._setup_normalizer(10, 1., 10.)
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Get the result before the update step.
    with strategy.scope():
      before = normalizer.normalize(tensor)

    # Apply an update (should have no effect).
    self._update(normalizer, strategy, tensor)

    # Get the result after the update step.
    with strategy.scope():
      after = normalizer.normalize(tensor)

    shouldbe = (tensor - 1.) / 10.
    self.assertAllClose(before, shouldbe)
    self.assertAllClose(after, shouldbe)

  def test_variables(self):
    """Tests that we have the correct number of variables."""
    normalizer, _ = self._setup_normalizer(10, 1., 10.)
    self.assertEmpty(normalizer.variables, 0)
    self.assertEmpty(normalizer.trainable_variables, 0)

  def test_invertible(self):
    """Tests that the normalization is invertible."""
    normalizer, strategy = self._setup_normalizer(10, 1., 10.)
    tensor = tf.reshape(tf.range(200, dtype=tf.float32), (20, 10))

    # Apply update.
    self._update(normalizer, strategy, tensor)

    # Verify that update is correct.
    with strategy.scope():
      normalized = normalizer.normalize(tensor)
      inverted = normalizer.unnormalize(normalized)

    self.assertAllClose(tensor, inverted)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
