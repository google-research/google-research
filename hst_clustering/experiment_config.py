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

"""Data classes to specify parameters of different algorithms."""


import dataclasses


@dataclasses.dataclass
class OneMedianDPConfig:
  epsilon: float  # Epsilon for differential privacy
  delta: float  # Delta for differential privacy
  smoothing_factor: float  # The smoothing factor lambda used in
  # https://dl.acm.org/doi/pdf/10.1145/3534678.3539409. It is used to generate
  # a smooth approximation to the euclidean norm (bottom of page 227). As
  # this value goes to zero, the smooth approximation approaches the norm.
  gamma: float  # Accuracy expected to be satisfied by the optimizer of the
  # one median.


@dataclasses.dataclass
class KVariatesConfig:
  num_splits: int
  split_strategy: str
  num_centers: int
  # Only one of one_median_dp_config and random_selection should be set.
  one_median_dp_config: OneMedianDPConfig
  random_selection: bool


@dataclasses.dataclass
class KMedsPlusPlusConfig:
  niter: int
  num_clusters: int


@dataclasses.dataclass
class ClusteringConfig:
  # Only one should be set.
  kvariates_config: KVariatesConfig
  kmeds_plus_plus_config: KMedsPlusPlusConfig
