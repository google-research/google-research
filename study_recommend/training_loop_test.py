# coding=utf-8
# Copyright 2025 The Google Research Authors.
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


import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import orbax.checkpoint
import pandas as pd
from study_recommend import config
from study_recommend import datasource as datasource_lib
from study_recommend import training_loop
from study_recommend import types

SEQ_LEN = 10


def get_test_datasource():
  fields = types.StudentActivityFields
  student_activity_dataframe = pd.DataFrame(
      data=[
          [1, '2020-12-31', 'A'],
          [1, '2020-12-31', 'B'],
          [1, '2021-01-01', 'B'],
      ],
      columns=[
          fields.STUDENT_ID.value,
          fields.DATE.value,
          fields.BOOK_ID.value,
      ],
  ).groupby(fields.STUDENT_ID)
  student_info = pd.DataFrame(
      data=[[1, 2, 3]],
      columns=[
          fields.STUDENT_ID.value,
          fields.SCHOOL_ID.value,
          fields.GRADE_LEVEL.value,
      ],
  )
  my_datasource, vocab = (
      datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
          student_activity_dataframe,
          student_info,
          seq_len=10,
          vocab_size=10,
          with_replacement=True,
      )
  )
  return my_datasource, vocab


def get_test_config(vocab):
  cfg = config.get_config(
      vocab, per_device_batch_size=8, seq_len=SEQ_LEN, working_dir=''
  )
  cfg.save_checkpoints = False
  cfg.restore_checkpoints = False
  cfg.num_layers = 1
  cfg.qkv_dim = 4
  cfg.mlp_dim = 4
  cfg.num_heads = 1
  cfg.reference_train_batch_size = 4
  cfg.reference_valid_batch_size = 4
  cfg.num_train_steps = 1
  return cfg


class TrainingLoopTest(unittest.TestCase):

  def test_non_nan_returns_individual(self):
    """Assert running a few steps of training individual does not produce NaNs."""
    datasource, vocab = get_test_datasource()
    cfg = get_test_config(vocab)
    cfg.model_class = 'individual'
    state, _ = training_loop.training_loop(
        config=cfg, train_datasource=datasource, valid_datasource=datasource
    )
    isfinite = jax.tree.map(lambda x: jnp.isfinite(x).all(), state)
    isfinite_leaves = [
        leaf for (_, leaf) in jax.tree_util.tree_flatten_with_path(isfinite)[0]
    ]
    self.assertTrue(all(isfinite_leaves))

  def test_non_nan_returns_study(self):
    """Assert running a few steps of training study does not produce NaN parameters."""
    datasource, vocab = get_test_datasource()
    cfg = get_test_config(vocab)
    cfg.model_class = 'study'
    state, _ = training_loop.training_loop(
        config=cfg, train_datasource=datasource, valid_datasource=datasource
    )
    isfinite = jax.tree.map(lambda x: jnp.isfinite(x).all(), state)
    isfinite_leaves = [
        leaf for (_, leaf) in jax.tree_util.tree_flatten_with_path(isfinite)[0]
    ]
    self.assertTrue(all(isfinite_leaves))

  def test_tensorboard_logging(self):
    """Assert values are logged to tensorboard."""
    datasource, vocab = get_test_datasource()
    cfg = get_test_config(vocab)
    # This path will not be written to as the metric writer will be mocked.
    cfg.tensorboard_dir = '/a/random/path'
    mock_summary_writer = unittest.mock.create_autospec(
        training_loop.ParallelSummaryWriter, instance=True
    )

    with unittest.mock.patch.object(
        training_loop,
        'ParallelSummaryWriter',
        side_effect=[mock_summary_writer],
    ):
      training_loop.training_loop(
          config=cfg, train_datasource=datasource, valid_datasource=datasource
      )
      mock_summary_writer.write_scalars.assert_called_with(
          0,
          {
              'train_accuracy': mock.ANY,
              'train_loss': mock.ANY,
              'train_oov_corrected_accuracy': mock.ANY,
              'valid_accuracy': mock.ANY,
              'valid_loss': mock.ANY,
              'valid_oov_corrected_accuracy': mock.ANY,
          },
      )

  def test_saving_checkpoints(self):
    """Assert correct checkpoint files are saved to disk."""
    datasource, vocab = get_test_datasource()
    cfg = get_test_config(vocab)
    # This path will not be written to as the metric writer will be mocked.
    cfg.working_dir = '/a/random/path'
    cfg.save_checkpoints = True

    mock_file_handle = mock.MagicMock()
    mock_file_handle.__enter__.return_value = mock_file_handle

    mock_checkpointer = mock.MagicMock()
    with mock.patch.object(
        orbax.checkpoint, 'Checkpointer', side_effect=[mock_checkpointer]
    ), mock.patch.object(
        training_loop, 'file_exists', return_value=False
    ), mock.patch.object(
        training_loop, 'file_open', side_effect=[mock_file_handle]
    ) as mock_open, mock.patch.object(
        training_loop, 'make_dirs'
    ):
      training_loop.training_loop(
          config=cfg, train_datasource=datasource, valid_datasource=datasource
      )
      save_args, _ = mock_checkpointer.save.call_args_list[-1]
      self.assertEqual(
          save_args[0], '/a/random/path/checkpoints/checkpoint_000000000000000'
      )
      mock_open.assert_called_with('/a/random/path/config.json', 'w')
      mock_file_handle.write.assert_called_once()


if __name__ == '__main__':
  unittest.main()
