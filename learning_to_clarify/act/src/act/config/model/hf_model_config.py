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
from typing import Optional

from act.config.model.model_config import ModelConfig


@dataclasses.dataclass
class HFModelConfig(ModelConfig):
  token: Optional[str] = None
  revision: str = 'main'
  trust_remote_code: bool = False
  use_flash_attention_2: bool = False
  model_code_revision: Optional[str] = None
  torch_dtype: Optional[str] = 'auto'
  tokenizer_name_or_path: Optional[str] = None
  gradient_checkpointing: bool = True
  load_in_4bit: bool = False
  load_in_8bit: bool = False
  use_bnb_nested_quant: bool = False
  bnb_4bit_quant_type: Optional[str] = None
  bnb_4bit_quant_storage: Optional[str] = None


@dataclasses.dataclass
class Gemma2Config(HFModelConfig):
  attn_implementation: str = 'eager'
