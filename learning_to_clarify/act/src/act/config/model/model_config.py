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

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


@dataclasses.dataclass()
class ModelConfig:
  """Configuration for a model."""
  # Primary reason for setting None here is Error Checking and
  # avoiding having the dataclass be kw_only, since that would require
  # an explicit dependence on Python 3.10.
  #
  # Can also set to None to load the model from model_path instead.
  model_id: Optional[str] = None

  # Load the model from Google Storage instead. The model will be loaded using
  # Hugging Face APIs.
  model_path: Optional[str] = None
