# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tests for flax_cifar.training_utils.flax_training."""

import os
from typing import Tuple

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import flax
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import pandas as pd
import tensorflow as tf
from tensorflow.io import gfile

from flax_cifar.datasets import dataset_source
from flax_cifar.training_utils import flax_training


FLAGS = flags.FLAGS


class MockDatasetSource(dataset_source.DatasetSource):
  """Simple linearly separable dataset for testing.

  See base class for more details.
  """

  def __init__(self):
    positive_input = tf.constant([[1.0 + i/20] for i in range(20)])
    negative_input = tf.constant([[-1.0 - i/20] for i in range(20)])
    positive_labels = tf.constant([[1, 0] for _ in range(20)])
    negative_labels = tf.constant([[0, 1] for _ in range(20)])
    inputs = tf.concat((positive_input, negative_input), 0)
    labels = tf.concat((positive_labels, negative_labels), 0)
    self.inputs, self.labels = inputs.numpy(), labels.numpy()
    self._ds = tf.data.Dataset.from_tensor_slices({
        'image': inputs,
        'label': labels
    })
    self.num_training_obs = 40
    self.batch_size = 16

  def get_train(self, use_augmentations):
    """Returns the training set.

    Args:
      use_augmentations: Ignored (see base class for more details).
    """
    del use_augmentations
    return self._ds.batch(self.batch_size)

  def get_test(self):
    """Returns the test set."""
    return self._ds.batch(self.batch_size)


def _get_linear_model():
  """Returns a linear model and its state."""

  class LinearModel(flax.nn.Module):
    """Defines the linear model."""

    def apply(self,
              x,
              num_outputs,
              train = False):
      """Forward pass with a linear model.

      Args:
        x: Input of shape [batch_size, num_features].
        num_outputs: Number of classes.
        train: Has no effect.

      Returns:
        A tensor of shape [batch_size, num_outputs].
      """
      del train
      x = flax.nn.Dense(x, num_outputs)
      return x

  input_shape, num_outputs = [1], 2
  module = LinearModel.partial(num_outputs=num_outputs)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = module.init_by_shape(
          jax.random.PRNGKey(0), [(input_shape, jnp.float32)])
      model = flax.nn.Model(module, initial_params)
  return model, init_state


def tensorboard_event_to_dataframe(path):
  """Helper to get events written by tests.

  Args:
    path: Path where the tensorboard records were saved.

  Returns:
    The metric saved by tensorboard, as a dataframe.
  """
  records = []
  all_tb_path = gfile.glob(os.path.join(path, 'events.*.v2'))
  for tb_event_path in all_tb_path:
    for e in tf.compat.v1.train.summary_iterator(tb_event_path):
      if e.step:
        for v in e.summary.value:
          records.append(dict(
              step=e.step, metric=v.tag,
              value=float(tf.make_ndarray(v.tensor))))
  df = pd.DataFrame.from_records(records)
  return df


prev_xla_flags = None


class FlaxTrainingTest(absltest.TestCase):

  # Run all tests with 8 CPU devices.
  # As in third_party/py/jax/tests/pmap_test.py
  def setUp(self):
    super(FlaxTrainingTest, self).setUp()
    global prev_xla_flags
    prev_xla_flags = os.getenv('XLA_FLAGS')
    flags_str = prev_xla_flags or ''
    # Don't override user-specified device count, or other XLA flags.
    if 'xla_force_host_platform_device_count' not in flags_str:
      os.environ['XLA_FLAGS'] = (
          flags_str + ' --xla_force_host_platform_device_count=8')
    # Clear any cached backends so new CPU backend will pick up the env var.
    xla_bridge.get_backend.cache_clear()

  # Reset to previous configuration in case other test modules will be run.
  def tearDown(self):
    super(FlaxTrainingTest, self).tearDown()
    if prev_xla_flags is None:
      del os.environ['XLA_FLAGS']
    else:
      os.environ['XLA_FLAGS'] = prev_xla_flags
    xla_bridge.get_backend.cache_clear()

  @flagsaver.flagsaver
  def test_TrainSimpleModel(self):
    """Model should reach 100% accuracy easily."""
    model, state = _get_linear_model()
    dataset = MockDatasetSource()
    num_epochs = 10
    optimizer = flax_training.create_optimizer(model, 0.0)
    training_dir = self.create_tempdir().full_path
    FLAGS.learning_rate = 0.01
    flax_training.train(
        optimizer, state, dataset, training_dir, num_epochs)
    records = tensorboard_event_to_dataframe(training_dir)
    # Train error rate at the last step should be 0.
    records = records[records.metric == 'train_error_rate']
    records = records.sort_values('step')
    self.assertEqual(records.value.values[-1], 0.0)

  @flagsaver.flagsaver
  def test_ResumeTrainingAfterInterruption(self):
    """Resuming training should match a run without interruption."""
    model, state = _get_linear_model()
    dataset = MockDatasetSource()
    optimizer = flax_training.create_optimizer(model, 0.0)
    training_dir = self.create_tempdir().full_path
    FLAGS.learning_rate = 0.01
    FLAGS.use_learning_rate_schedule = False
    # First we train for 10 epochs and get the logs.
    num_epochs = 10
    reference_run_dir = os.path.join(training_dir, 'reference')
    flax_training.train(
        optimizer, state, dataset, reference_run_dir, num_epochs)
    records = tensorboard_event_to_dataframe(reference_run_dir)
    # In another directory (new experiment), we run the model for 4 epochs and
    # then for 10 epochs, to simulate an interruption.
    interrupted_run_dir = os.path.join(training_dir, 'interrupted')
    flax_training.train(
        optimizer, state, dataset, interrupted_run_dir, 4)
    flax_training.train(
        optimizer, state, dataset, interrupted_run_dir, 10)
    records_interrupted = tensorboard_event_to_dataframe(interrupted_run_dir)

    # Logs should match (order doesn't matter as it is a dataframe in tidy
    # format).
    def _make_hashable(row):
      return str([e if not isinstance(e, float) else round(e, 5) for e in row])

    self.assertEqual(
        set([_make_hashable(e) for e in records_interrupted.values]),
        set([_make_hashable(e) for e in records.values]))

  def test_RecomputeTestLoss(self):
    """Recomputes the loss of the final model to check the value logged."""
    model, state = _get_linear_model()
    dataset = MockDatasetSource()
    num_epochs = 2
    optimizer = flax_training.create_optimizer(model, 0.0)
    training_dir = self.create_tempdir().full_path
    flax_training.train(
        optimizer, state, dataset, training_dir, num_epochs)
    records = tensorboard_event_to_dataframe(training_dir)
    records = records[records.metric == 'test_loss']
    final_test_loss = records.sort_values('step').value.values[-1]
    # Loads final model and state.
    optimizer, state, _ = flax_training.restore_checkpoint(
        optimizer, state, os.path.join(training_dir, 'checkpoints'))
    # Averages over the first dimension as we will use only one device (no
    # pmapped operation.)
    optimizer = jax.tree_map(lambda x: jax.numpy.mean(x, axis=0), optimizer)
    state = jax.tree_map(lambda x: jax.numpy.mean(x, axis=0), state)
    logits = optimizer.target(dataset.inputs)
    loss = flax_training.cross_entropy_loss(logits, dataset.labels)
    self.assertLess(abs(final_test_loss -loss), 1e-7)


if __name__ == '__main__':
  absltest.main()
