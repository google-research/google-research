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

"""Base class definition for general data splitting strategies."""

import abc
from typing import Iterable, Tuple

import numpy as np


class DataSplittingStrategy(abc.ABC):
  """Base class for data splitting strategies for Makita pipelines."""

  @abc.abstractmethod
  def split(self, example_pool,
            target_pool):
    """Generates training set instances from a base training set.

    Args:
      example_pool: List of feature lists (training inputs) to construct data
        sets from.
      target_pool: List of target values associated with examples. The i^th
        value in target_pool will be associated with the i^th value in
        example_pool.

    Returns:
      A generator that will iterate through all data set combinations. Each
      combination is a tuple of two numpy arrays. The first element of the
      tuple will be input features. The second element is a list of target
      values.
    """
    raise NotImplementedError
