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

"""Utilities for restoring saved artifacts."""

import contextlib
import json
from typing import TextIO, Union
import flax.linen as nn
import ml_collections
import pandas as pd
from study_recommend import datasource as datasource_lib
from study_recommend import models
from study_recommend import training_loop
from study_recommend import types
from study_recommend.utils import input_pipeline_utils
file_open = open


def restore_model(
    config_file,
):
  """Restore a model from a config file path or text file buffer.

  Args:
    config_file: Path or file text file buffer for config.json to restore from.

  Returns:
    params: Jax model parameters.
    eval_config: Evaluation config for model.
    model_class: The class of model saved in config_file.
    cfg: The experiment config read from config_file.
  """
  # Load the experiment config.
  if isinstance(config_file, TextIO):
    file_context = contextlib.nullcontext(config_file)
  else:
    file_context = file_open(config_file, 'r')
  with file_context as f:
    tmp = json.load(f)
  # Cast it into ml_collections.ConfigDict().
  cfg = ml_collections.ConfigDict()
  cfg.update(tmp)

  # Load the model from the path specified in the config.
  state, _, eval_config, model_class = training_loop.init_or_load(
      cfg, _get_test_datasource()
  )
  return state.params, eval_config, model_class, cfg


def restore_vocabulary(
    vocabulary_file
):
  """Restore vocabulary from a vocabulary file path or text file buffer."""
  return input_pipeline_utils.Vocabulary().deserialize(vocabulary_file)


def _get_test_datasource():
  """Get a datasource with dummy data to initialize a model."""
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

  datasource, _ = (
      datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
          student_activity_dataframe,
          student_info,
          seq_len=10,
          student_chunk_len=10,
          vocab_size=10,
          with_replacement=True,
          ordered_within_student=True,
          max_grade_level=20,
      )
  )

  return datasource
