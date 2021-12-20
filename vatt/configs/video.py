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

# Lint as: python3
"""Config definition for video models."""

import dataclasses
from typing import Optional

from vatt.configs import base_config


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """General common configuration for video models."""

  name: str = ''
  freeze_backbone: bool = False
  final_endpoint: Optional[str] = None
  num_classes: Optional[int] = None
  cls_dropout_rate: float = 0.5
  num_test_samples: Optional[int] = None
  backbone_config: Optional[base_config.Config] = None


@dataclasses.dataclass
class I3D(ModelConfig):
  """Configuration of the I3D model."""

  name: str = 'i3d'
  width_multiplier: int = 1
  conv_weight_decay: float = 0.
  batch_norm_decay: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_norm_scale: bool = True
  use_xreplica_bn: bool = True
  data_format: str = 'channels_last'
  final_endpoint: str = 'mixed_5c'
  backbone_config: Optional[base_config.Config] = None


@dataclasses.dataclass
class ViTBase(ModelConfig):
  """Configuration of the Base Vision Transformer model."""

  name: str = 'vit_base'
  # input parameters
  temporal_patch_size: int = 4
  spatial_patch_size: int = 16
  max_temporal_buckets: int = 8  # 32 // 4
  max_vertical_buckets: int = 14  # 224 // 16
  max_horizontal_buckets: int = 14  # 224 // 16
  random_patch_sampling: bool = True  # apply DropToken
  patch_sampling_rate: float = 0.5  # DropToken rate
  # pre-tx projection
  pre_projection: bool = False  # True if finetuning a modality-agnostic model
  # network size parameters
  d_model: int = 768
  d_kv: int = 64
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  # interinsic parameters
  pre_norm: bool = True
  use_bias: bool = True
  activation: str = 'gelu'
  # masking parameters
  dropout_rate: float = 0.1
  layer_norm_epsilon: float = 1e-6
  use_masking: bool = False
  mask_rate: float = 0.2
  # output parameters
  post_projection: bool = False  # True if finetuning a modality-agnostic model
  d_post_proj: int = 1024
  post_proj_activation: str = activation
  final_endpoint: str = 'predictions_3d'
  num_classes: Optional[int] = None


@dataclasses.dataclass
class ViTMedium(ViTBase):
  """Configuration of the Medium Vision Transformer model."""

  name: str = 'vit_medium'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 12
  num_heads: int = 16


@dataclasses.dataclass
class ViTLarge(ViTBase):
  """Configuration of the Large Vision Transformer model."""

  name: str = 'vit_large'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16


@dataclasses.dataclass
class ViTXLarge(ViTBase):
  """Configuration of the X-Large Vision Transformer model."""

  name: str = 'vit_xlarge'
  d_model: int = 1536
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 24
