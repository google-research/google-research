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

"""Config definition for audio models."""

import dataclasses
from typing import Optional

from vatt.configs import base_config


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """General common configuration for audio models."""

  name: str = ''
  freeze_backbone: bool = False
  final_endpoint: Optional[str] = None
  num_classes: Optional[int] = None
  cls_dropout_rate: float = 0.0
  num_test_samples: Optional[int] = None


@dataclasses.dataclass
class Resnet2DBase(ModelConfig):
  """ResNet2D model config."""

  name: str = 'resnet2d'
  backbone: str = ''
  width_multiplier: int = 1
  cifar_stem: bool = False
  data_format: str = 'channels_last'
  stop_gradient_point: int = -1
  use_xreplica_bn: bool = True
  batch_norm_decay: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_norm_scale: bool = True
  dropblock_keep_probs: Optional[float] = None
  dropblock_size: Optional[int] = None
  final_endpoint: str = 'last_conv'
  num_classes: Optional[int] = None


@dataclasses.dataclass
class Resnet2D50(Resnet2DBase):
  """R50-2D model config."""

  name: str = 'resnet2d_50'
  backbone: str = 'resnet50'


@dataclasses.dataclass
class Resnet2D101(Resnet2DBase):
  """R101-2D model config."""

  name: str = 'resnet2d_101'
  backbone: str = 'resnet101'


@dataclasses.dataclass
class WaTBase(ModelConfig):
  """Configuration of the Base Waveform Transformer model."""

  name: str = 'wat_base'
  # input parameters
  temporal_patch_size: int = 128
  temporal_patch_stride: int = 128
  max_temporal_buckets: int = 1200  # 3.2*48k // 128
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
  dropout_rate: float = 0.1
  layer_norm_epsilon: float = 1e-6
  # masking parameters
  use_masking: bool = False
  mask_rate: float = 0.2
  # output parameters
  post_projection: bool = False  # True if finetuning a modality-agnostic model
  d_post_proj: int = 1024
  post_proj_activation: str = activation
  final_endpoint: str = 'predictions_1d'
  num_classes: Optional[int] = None


@dataclasses.dataclass
class WaTMedium(WaTBase):
  """Configuration of the Medium Waveform Transformer model."""

  name: str = 'wat_medium'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 12
  num_heads: int = 16


@dataclasses.dataclass
class WaTLarge(WaTBase):
  """Configuration of the Large Waveform Transformer model."""

  name: str = 'wat_large'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16


@dataclasses.dataclass
class SpTBase(ModelConfig):
  """Configuration of the Base Spectrogram Transformer model.

  Attributes:
    name: .
  """
  name: str = 'spt_base'
  # input parameters
  temporal_patch_size: int = 16
  spectoral_patch_size: int = 5
  max_temporal_buckets: int = 9  # 150 // 16
  max_spectoral_buckets: int = 16  # 80 // 5
  # pre-tx projection
  pre_projection: bool = False
  # network size parameters
  d_model: int = 768
  d_kv: int = 64
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  # interinsic parameters
  pre_norm: bool = True
  use_bias: bool = False
  activation: str = 'gelu'
  dropout_rate: float = 0.1
  layer_norm_epsilon: float = 1e-6
  # masking parameters
  use_masking: bool = False
  mask_rate: float = 0.2
  # output parameters
  post_projection: bool = False
  d_post_proj: int = 1024
  post_proj_activation: str = activation
  final_endpoint: str = 'predictions_2d'
  num_classes: Optional[int] = None


@dataclasses.dataclass
class SpTMedium(SpTBase):
  """Configuration of the Medium Spectrogram Transformer model."""

  name: str = 'spt_medium'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 12
  num_heads: int = 16


@dataclasses.dataclass
class SpTLarge(SpTBase):
  """Configuration of the Large Spectrogram Transformer model."""

  name: str = 'spt_large'
  d_model: int = 1024
  d_kv: int = 64
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16
