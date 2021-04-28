# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Contains the hparams class used for ResNet."""

import typing
from typing import Any

import dataclasses

from aqt.jax.flax import struct as flax_struct
from aqt.utils import hparams_utils

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


@dataclass
class TrainingHParams:
  """Hyperparameters used for training."""

  # Metadata
  metadata: hparams_utils.HParamsMetadata

  # General hparams
  base_learning_rate: float
  momentum: float
  weight_decay: float

  # Auto-clip activation quantization hparams. See
  # train_utils.should_update_bounds for more details. We use -1 instead of None
  # to indicate that these parameters should be inactive since if these
  # attributes are marked as optional, Dacite will set them to None even if
  # they are missing from the JSON-serialized version of this dataclass, while
  # we want Dacite to raise an error in this case to alert users about an
  # incomplete configuration.
  activation_bound_update_freq: int
  activation_bound_start_step: int

  # Model hparams
  model_hparams: Any
