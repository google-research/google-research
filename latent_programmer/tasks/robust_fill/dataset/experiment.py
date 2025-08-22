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

"""The Experiment enum describing different datasets."""

import enum


@enum.unique
class Experiment(enum.Enum):
  NONE = 0
  LENGTH_1_6_TO_7_10 = 1
  LENGTH_6_TO_1_10 = 2
  COMPOSE_DIFFERENT_CONCEPTS = 3
  SWITCH_CONCEPT_ORDER = 4
  COMPOSE_NEW_OP = 5
  EXTEND_OP_FUNCTIONALITY = 6
