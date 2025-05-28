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

"""Utility Classes and Enums for AP parsing."""

import dataclasses
import enum


@enum.unique
class LabeledSpanType(enum.IntEnum):
  """Span types for AP parsing."""
  UNKNOWN_TYPE = 0
  PROBLEM_TITLE = 1
  PROBLEM_DESCRIPTION = 2
  ACTION_ITEM = 3


@enum.unique
class ActionItemType(enum.IntEnum):
  """Action item types for AP parsing action item spans."""
  NOT_SET = 0
  MEDICATIONS = 1
  IMAGING = 2
  OBSERVATIONS_LABS = 3
  CONSULTS = 4
  NUTRITION = 5
  THERAPEUTIC_PROCEDURES = 6
  OTHER_DIAGNOSTIC_PROCEDURES = 7
  OTHER = 8


@dataclasses.dataclass
class LabeledCharSpan:
  """A character level container of AP parsing spans."""
  start_char: int
  end_char: int
  span_type: LabeledSpanType
  action_item_type: ActionItemType = ActionItemType.NOT_SET


@dataclasses.dataclass
class LabeledTokenSpan:
  """A token level container of AP parsing spans."""
  start_token: int
  end_token: int
  span_type: LabeledSpanType
  action_item_type: ActionItemType = ActionItemType.NOT_SET
