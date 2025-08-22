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

import abc
from typing import Any, Optional, Union

import torch

from act.config.base_config import BaseConfig


class BaseModelABC(abc.ABC):

  def __init__(
      self,
      config,
  ):
    raise NotImplementedError

  def generate(self, inputs, **generation_kwargs):
    raise NotImplementedError


class BaseModel(BaseModelABC):

  def __init__(
      self,
      config,
  ):
    self.config = config
