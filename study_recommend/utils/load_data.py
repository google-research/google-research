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

"""Utility functions for loading and preprocessing the audioboook data."""

import pandas as pd
from pandas.core import groupby as pd_groupby
from study_recommend import types
open_file = open


FIELDS = types.StudentActivityFields


def load_student_activity_files(
    *file_paths,
):
  """Load student activity files as Grouped Dataframes.

  Args:
    *file_paths: A tuple of file paths. The files will be concatenated into a
      single dataframe.

  Returns:
    grouped_reading_activity: A pandas dataframe of reading activity,
      grouped by student and sorted by activity date.
  """
  pandas_dataframes = []
  for file_path in file_paths:
    with open_file(file_path, 'r') as file_io:
      pandas_dataframes.append(pd.read_csv(file_io))

  joint_data = pd.concat(pandas_dataframes, ignore_index=True)

  return group_reading_activity_df(joint_data)


def group_reading_activity_df(
    reading_activity,
):
  """Sort and group the reading activity dataframe for convenience.

  Args:
    reading_activity: A pandas dataframe with student reading activity

  Returns:
    grouped_reading_activity: A pandas dataframe of reading activity,
      grouped by student and sorted by activity date.
  """
  reading_activity[FIELDS.INTENSITY] = reading_activity[FIELDS.DURATION]
  grouped_df = reading_activity.sort_values(
      [FIELDS.STUDENT_ID, FIELDS.DATE], kind='stable'
  ).groupby(FIELDS.STUDENT_ID)
  return grouped_df


def create_title_index_lookup(filepath_to_title_info):
  """Build a dictionary mapping titles ids to integer indices."""
  with open_file(filepath_to_title_info, 'r') as file_io:
    title_info = pd.read_csv(file_io)

  book_id_to_indx = {
      book_id: i for i, book_id in enumerate(title_info[FIELDS.BOOK_ID])
  }

  return book_id_to_indx
