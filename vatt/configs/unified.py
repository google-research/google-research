# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Config definition for unified backbones."""

import dataclasses
from typing import Optional

from vatt.configs import base_config


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """General common configuration for unified models.

  Attributes:
    name: name of the model
  """
  name: str = "unified_model"
  modality: str = ""
  num_classes: Optional[int] = None
  cls_dropout_rate: float = 0.2


@dataclasses.dataclass
class UniTBase(ModelConfig):
  """Base Unified Transformer model config.

  """
  name: str = "ut_base"
  # input parameters
  vid_temporal_patch_size: int = 4
  vid_spatial_patch_size: int = 16
  aud_temporal_patch_size: int = 128
  txt_vocab_size: int = 2**16
  txt_embedding_dim: int = 300
  txt_embedding_trainable: bool = False
  max_vid_temporal_buckets: int = 8  # 32 // 4
  max_vid_spatial_buckets: int = 14  # 224 // 16
  max_aud_temporal_buckets: int = 1200  # 3.2*48k // 128
  max_txt_temporal_buckets: int = 16
  random_patch_sampling: bool = True
  patch_sampling_rate: float = 0.5
  # network size parameters
  d_model: int = 768
  d_kv: int = 64
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  # interinsic parameters
  pre_norm: bool = True
  use_bias: bool = True
  activation: str = "gelu"
  dropout_rate: float = 0.1
  layer_norm_epsilon: float = 1e-6
  # post-Transformer parameters
  d_post_proj: int = 1024


@dataclasses.dataclass
class UniTMedium(UniTBase):
  """Configuration of the Medium Unified Transformer model."""

  name: str = "ut_medium"
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 12
  num_heads: int = 16


@dataclasses.dataclass
class UniTLarge(UniTBase):
  """Configuration of the Large Unified Transformer model."""

  name: str = "ut_large"
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16
