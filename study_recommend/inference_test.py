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

"""Unit tests for inference."""

import unittest
from unittest import mock
from flax.training import common_utils
import jax
import pandas as pd
from study_recommend import config
from study_recommend import datasource as datasource_lib
from study_recommend import inference
from study_recommend import training_loop
from study_recommend import types

SEP_TOKEN = 3
CHUNK_SIZE = 2
SEQ_LEN = 10
FIELDS = types.ModelInputFields


def get_test_datasource():
  fields = types.StudentActivityFields
  student_activity_dataframe = pd.DataFrame(
      data=[
          [1, '2020-12-31', 'A'],
          [1, '2020-12-31', 'B'],
          [1, '2021-01-01', 'B'],
          [2, '2020-12-31', 'A'],
          [2, '2020-12-31', 'B'],
          [2, '2021-01-01', 'B'],
      ],
      columns=[
          fields.STUDENT_ID.value,
          fields.DATE.value,
          fields.BOOK_ID.value,
      ],
  ).groupby(fields.STUDENT_ID)
  student_info = pd.DataFrame(
      data=[[1, 10, 100], [2, 10, 100]],
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
          seq_len=SEQ_LEN,
          student_chunk_len=CHUNK_SIZE,
          vocab_size=10,
          with_replacement=False,
          ordered_within_student=True,
      )
  )

  my_datasource_with_replacement, _ = (
      datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
          student_activity_dataframe,
          student_info,
          seq_len=SEQ_LEN,
          student_chunk_len=CHUNK_SIZE,
          vocab_size=10,
          with_replacement=True,
          ordered_within_student=True,
      )
  )

  return my_datasource, my_datasource_with_replacement, vocab


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


class InferenceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Prevent just-in-time compilation (jit) from interfering with patching.
    jax.config.update('jax_disable_jit', True)

  def test_inference(self):
    datasource, datasource_with_replacement, vocab = get_test_datasource()

    cfg = get_test_config(vocab=vocab)
    train_state, _, eval_config, _ = training_loop.init_or_load(
        cfg, datasource_with_replacement
    )

    model_class = mock.MagicMock()
    instance = mock.MagicMock()
    model_class.side_effect = [instance]

    def mock_forward(_, inputs):
      """Mock forward predicts the next book to read as the recommendation."""
      return common_utils.onehot(
          inputs[FIELDS.TITLES], num_classes=len(vocab) + 1
      )

    instance.apply.side_effect = mock_forward

    recommendations = inference.recommend_from_datasource(
        eval_config,
        model_class,
        train_state.params,
        datasource,
        n_recommendations=2,
        vocab=vocab,
        per_device_batch_size=4,
    )

    # We are recommending 2 titles per timestep. First recommendation would be
    # the book actually read. Second recommendation would be previous book
    reference_recommendations = {
        1: [['A', 'B'], ['B', 'A'], ['B', 'A']],
        2: [['A', 'B'], ['B', 'A'], ['B', 'A']],
    }

    self.assertEqual(recommendations, reference_recommendations)


if __name__ == '__main__':
  unittest.main()
