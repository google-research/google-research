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

"""Aggregate functions."""

import abc
from typing import Sequence

import numpy as np


class AggregateFunction(metaclass=abc.ABCMeta):
  """Interface for aggregate function APIs."""

  @abc.abstractmethod
  def __call__(self, scores, **kwargs):
    raise NotImplementedError()


class MinAggregateFunction(AggregateFunction):
  """Minimum aggregate function."""

  def __call__(self, scores, **kwargs):
    return np.min(scores)


class MaxAggregateFunction(AggregateFunction):
  """Maximum aggregate function."""

  def __call__(self, scores, **kwargs):
    return np.max(scores)


class MeanAggregateFunction(AggregateFunction):
  """Mean aggregate function."""

  def __call__(self, scores, **kwargs):
    return np.mean(scores)


class WeightedMeanAggregateFunction(AggregateFunction):
  """Weighted mean aggregate function."""

  def __call__(self, scores, **kwargs):
    if 'weights' not in kwargs:
      raise ValueError('weights must be provided.')
    return np.sum(np.array(scores) * np.array(kwargs['weights']))
