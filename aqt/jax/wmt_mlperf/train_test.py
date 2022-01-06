# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

import concurrent.futures
import math
import re

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from aqt.jax.wmt_mlperf import train
from aqt.jax.wmt_mlperf import training_hparams
from aqt.jax.wmt_mlperf.hparams_configs.experimental import minimal_model_8bit_weights_and_auto_acts
from aqt.jax.wmt_mlperf.hparams_configs.experimental import minimal_model_bfloat16
from aqt.jax.wmt_mlperf.training_hparams_generator_lib import BaseConfigSize
from aqt.jax.wmt_mlperf.training_hparams_generator_lib import create_training_hparams_from_flags
from aqt.utils import hparams_utils as os_hparams_utils

FLAGS = flags.FLAGS


def create_1x1():
  return np.ones((1, 1), dtype=np.int64)


def create_mock_data():
  """Create the absolute smallest dataset that can be supplied to the training loop."""

  # Note that not all these keys are necessary for each kind of dataset
  # (train, eval, predict),
  # but it doesn't hurt to have them.
  data = {
      'inputs': create_1x1(),
      'inputs_position': create_1x1(),
      'inputs_segmentation': create_1x1(),
      'targets': create_1x1(),
      'targets_position': create_1x1(),
      'targets_segmentation': create_1x1(),
  }
  dataset = tf.data.Dataset.from_tensor_slices(data).batch(1)
  return dataset


# The training loop requires a decoder for its target language (ie, word
# index->human-readable string) to produce human-readable output during
# evaluation, so we mock the decoder.
class MockEncoder():
  """Mocks the tensorflow_text.SentencepieceTokenizer used in input_pipeline.py."""

  def vocab_size(self):
    # Not '1' so that 'low_confidence' calculation in train.py
    # doesn't divide-by-zero.
    return 2

  def detokenize(self, _):
    return tf.constant('<mock>')


# TODO(wanglisa): Once the remaining transformer kwargs are moved to flags,
# these test cases can all be specified in terms of flags (and thus we gain
# additional test coverage of the flag-processing functions in
# training_hparams.py).
class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.datasets = train.Datasets(
        train_ds=create_mock_data(),
        eval_ds_dict={'mock_data': create_mock_data()},
        train_eval_ds=create_mock_data(),
        predict_ds_dict={'mock_data': create_mock_data()},
        encoder=MockEncoder())

  @parameterized.named_parameters(
      dict(
          testcase_name='minimal_bfloat16',
          hparams_config_filename=minimal_model_bfloat16),)
  def test_train_with_config_file(self, hparams_config_filename):
    with flagsaver.flagsaver(
        num_eval_steps=1,
        eval_batch_size=1,
        max_target_length=1,
        eval_dataset_name='mock_data',
        max_eval_target_length=1,
        max_predict_length=1,
        model_dir=FLAGS.test_tmpdir,
    ):
      hparams = os_hparams_utils.load_dataclass_from_config_dict(
          training_hparams.TrainingHParams,
          hparams_config_filename.get_config())
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as io_executor:
        train.run_training(
            datasets=self.datasets,
            hparams=hparams,
            io_executor=io_executor,
        )

  def test_checkpointing(self):
    hparams = os_hparams_utils.load_dataclass_from_config_dict(
        training_hparams.TrainingHParams,
        minimal_model_8bit_weights_and_auto_acts.get_config())
    training_state_initial = train.TrainingState.initialize(
        encoder=self.datasets.encoder, hparams=hparams)

    ckpt_dir = FLAGS.test_tmpdir + '/ckpt_dir'
    tf.io.gfile.makedirs(ckpt_dir)
    # Delete existing files in the temp directory to clear out old checkpoints.
    for dir_name, _, file_names in tf.io.gfile.walk(ckpt_dir):
      for filename in file_names:
        file_path = tf.io.gfile.join(dir_name, filename)
        tf.io.gfile.remove(file_path)

    training_state, _ = train.run_train_step(
        training_state=training_state_initial,
        batch=next(iter(self.datasets.train_ds)),
        step=0,
        hparams=hparams)

    self.assertFalse(train.does_checkpoint_exist(ckpt_dir))
    training_state.save_checkpoint(model_dir=ckpt_dir, step=0)
    self.assertTrue(train.does_checkpoint_exist(ckpt_dir))

    self.assertNotEqual(training_state.flax_state,
                        training_state_initial.flax_state)
    with np.testing.assert_raises(AssertionError):
      np.testing.assert_array_equal(training_state.dropout_rngs,
                                    training_state_initial.dropout_rngs)
    training_state_restored = training_state_initial.restore_checkpoint(
        model_dir=ckpt_dir)

    leaf_equality_tree = jax.tree_multimap(lambda x, y: jnp.all(x == y),
                                           training_state.flax_state,
                                           training_state_restored.flax_state)
    self.assertTrue(
        all(jax.tree_leaves(leaf_equality_tree)),
        'Training state was altered during restoration.')

    np.testing.assert_array_equal(training_state.dropout_rngs,
                                  training_state_restored.dropout_rngs)


class LearningRateSchedulerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.hparams = training_hparams.LearningRateSchedulerHParams(
        factors='',
        base_learning_rate=2.0,
        warmup_steps=0,
        decay_factor=None,
        steps_per_decay=None,
        steps_per_cycle=None)

  def test_constant(self):
    hparams = self.hparams
    hparams.factors = 'constant'
    schedule = train.create_learning_rate_scheduler(hparams)
    self.assertEqual(schedule(0), 2.0)
    self.assertEqual(schedule(100), 2.0)

  def test_linear_warmup(self):
    hparams = self.hparams
    hparams.factors = 'constant*linear_warmup'
    hparams.warmup_steps = 10
    schedule = train.create_learning_rate_scheduler(hparams)
    self.assertEqual(schedule(0), 0.0)
    self.assertEqual(schedule(5), 2.0 / 2)
    self.assertEqual(schedule(10), 2.0)
    self.assertEqual(schedule(100), 2.0)

  def test_rsqrt_decay(self):
    hparams = self.hparams
    hparams.factors = 'constant*rsqrt_decay'
    hparams.warmup_steps = 9
    hparams.base_learning_rate = 1.0
    schedule = train.create_learning_rate_scheduler(hparams)
    self.assertEqual(schedule(0), 1 / math.sqrt(9))
    self.assertEqual(schedule(9), 1 / math.sqrt(9))
    self.assertEqual(schedule(100), 1 / math.sqrt(100))

  def test_mlperf_schedule(self):
    # Parameters modified slightly from the defaults to avoid dealing with
    # floating point error considerations when taking square roots.
    hparams = training_hparams.LearningRateSchedulerHParams(
        factors='constant * linear_warmup * rsqrt_decay',
        base_learning_rate=0.5,
        warmup_steps=100,
        decay_factor=0.5,
        steps_per_decay=20000,
        steps_per_cycle=100000)
    schedule = train.create_learning_rate_scheduler(hparams)
    self.assertEqual(schedule(0), 0.0)
    self.assertEqual(schedule(50), 0.5 / 2 * 1 / math.sqrt(100))
    self.assertEqual(schedule(100), 0.5 * 1 / math.sqrt(100))
    self.assertEqual(schedule(900), 0.5 * 1 / math.sqrt(900))


class BestCheckpointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_dir = FLAGS.test_tmpdir + '/model_dir'
    tf.io.gfile.makedirs(self.model_dir)
    # Delete existing files in the temp directory to clear out old checkpoints.
    for dir_name, _, file_names in tf.io.gfile.walk(self.model_dir):
      for filename in file_names:
        file_path = tf.io.gfile.join(dir_name, filename)
        tf.io.gfile.remove(file_path)

  def save_checkpoint(self, loss):
    with flagsaver.flagsaver(base_config_size=BaseConfigSize.MINIMAL_MODEL):
      hparams = create_training_hparams_from_flags()
    training_state = train.TrainingState.initialize(
        encoder=MockEncoder(), hparams=hparams)
    train.save_best_checkpoint(
        model_dir=self.model_dir, training_state=training_state, loss=loss)

  def get_model_dir_files(self):
    filenames = list(tf.io.gfile.walk(self.model_dir))[0][2]
    return filenames

  def get_checkpoint(self):
    """Gets the name of a saved checkpoint."""
    files = self.get_model_dir_files()
    # Test there is exactly one saved checkpoint.
    self.assertLen(
        files, 1, f'Expected exactly one checkpoint to be found, got {files}.')

    # Extract the loss associated with the checkpoint.
    checkpoint_match = re.match(r'.*best_checkpoint_eval_loss_(.*)$',
                                str(files[0]))
    self.assertIsNotNone(checkpoint_match)
    self.assertLen(checkpoint_match.groups(), 1)
    return checkpoint_match.groups()[0]

  def test_min_loss_saved(self):
    self.save_checkpoint(2.0)
    self.save_checkpoint(1.0)
    self.save_checkpoint(3.0)
    self.assertEqual(self.get_checkpoint(), '1.0')

  @parameterized.named_parameters(
      dict(testcase_name='nan', bad_value=math.nan),
      dict(testcase_name='inf', bad_value=math.inf),
      dict(testcase_name='-inf', bad_value=-math.inf))
  def test_nonfinite_ignored(self, bad_value):
    self.save_checkpoint(2.0)
    self.save_checkpoint(bad_value)
    self.save_checkpoint(1.0)
    self.save_checkpoint(3.0)
    self.assertEqual(self.get_checkpoint(), '1.0')

  def test_duplicate_loss(self):
    self.save_checkpoint(1.0)
    self.save_checkpoint(2.0)
    self.save_checkpoint(1.0)
    self.save_checkpoint(3.0)
    self.assertEqual(self.get_checkpoint(), '1.0')

  def test_no_checkpoint_saved_if_all_nan(self):
    for _ in range(5):
      self.save_checkpoint(math.nan)
    self.assertEmpty(self.get_model_dir_files())


if __name__ == '__main__':
  absltest.main()
