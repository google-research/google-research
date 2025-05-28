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

"""Standard types used throughout STUDY."""

import enum
from typing import Any, NewType


class StudentActivityFields(str, enum.Enum):
  """Contains labels of columns in StudentActivity data records."""
  INTENSITY = 'Intensity'
  STUDENT_ID = 'StudentID'
  DATE = 'Date'
  DURATION = 'Seconds Read'
  BOOK_ID = 'SHLF_NUM'
  GRADE_LEVEL = 'GRADE_LVL_NBR'
  SCHOOL_ID = 'OrgnID'
  DISTRICT_ID = 'Parent_Building_ID'


class EvalTypes(str, enum.Enum):
  """Different types of evaluation subsets which can be used to score models."""
  ALL = 'all'
  NON_CONTINUATION = 'non_continuation'
  # Non-history evaluation is referred to as novel in the STUDY paper.
  NON_HISTORY = 'non_history'


class ModelInputFields(str, enum.Enum):
  """Expected fields in the model input data."""
  TITLES = 'titles'
  INPUT_POSITIONS = 'input_positions'
  STUDENT_IDS = 'student_ids'
  TIMESTAMPS = 'timestamps'
  GRADE_LEVELS = 'grade_level'


class ResultsRecordFields(str, enum.Enum):
  """Expected fields in the model input data."""

  STUDENT_ID = 'Student_ID'
  EVAL_TYPE = 'Evaluation_Type'
  N_RECOMMENDATIONS = 'N_Recommendations'
  HITS_AT_N = 'Hits_at_N'


TokenIndex = NewType('TokenIndex', int)
Token = NewType('Token', str)
StudentID = NewType('StudentID', int)
ClassroomID = NewType('ClassroomID', tuple[int, int])
Timestamp = NewType('Timestamp', int)
StartIndex = int
NumItems = int
StudentIndexRange = NewType('StudentIndexRange', tuple[StartIndex, NumItems])
PyTree = Any  # A type for nested python containers with jax.ndarrays as
# leaves. Used to store network parameters, inputs and outputs.
