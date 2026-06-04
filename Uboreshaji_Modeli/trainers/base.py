# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Base classes and protocols for trainer strategies."""

import abc
from collections.abc import Mapping
from typing import Any

import ml_collections

from Uboreshaji_Modeli.engines import base


class TrainerStrategy(abc.ABC):
  """A base strategy for composed training."""

  @abc.abstractmethod
  def train(
      self,
      engine,
      dataset,
      cfg,
      **kwargs,
  ):
    """Executes the modality-specific training loop."""
    Ellipsis
