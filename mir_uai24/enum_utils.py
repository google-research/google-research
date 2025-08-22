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

"""Useful enum definitions."""

import dataclasses
import enum
from typing import Optional


class Dataset(str, enum.Enum):
  SYNTHETIC = 'synthetic'
  US_CENSUS = 'us_census'


class FeatureType(str, enum.Enum):
  REAL = 'real'
  CATEGORICAL = 'categorical'


@dataclasses.dataclass(frozen=True)
class Feature:
  key: str
  type: FeatureType
  num_categories: Optional[int] = None  # Not used for real features


@dataclasses.dataclass(frozen=True)
class DatasetInstance:
  bag_id: int
  instance_id: int
  bag_id_x_instance_id: int


@dataclasses.dataclass(frozen=True)
class DatasetMembershipInfo:
  instances: dict[int, set[DatasetInstance]]
  bags: dict[int, set[DatasetInstance]]


@dataclasses.dataclass(frozen=False)
class DatasetInfo:
  bag_id: str
  instance_id: str
  bag_id_x_instance_id: str
  bag_size: int
  n_instances: int
  features: list[Feature]
  label: str
  memberships: DatasetMembershipInfo
