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

"""Makita data 'sampler' that returns the full dataset."""
from typing import Iterable, Tuple
import numpy as np

from al_for_fep.data import data_splitting_strategy


class NoOpSplit(data_splitting_strategy.DataSplittingStrategy):
  """Class for passing a full data set through sampling."""

  def split(self, example_pool,
            target_pool):
    """Does not change anything about the passed pools.

    Args:
      example_pool: List of feature lists (training inputs) to construct data
        sets from.
      target_pool: List of target values associated with examples. The i^th
        value in target_pool will be associated with the i^th value in
        example_pool.

    Yields:
      Generator with one element - the pools passed as input.
    """
    yield (example_pool, target_pool)
