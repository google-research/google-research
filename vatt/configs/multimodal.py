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
"""Config definition for multimodal models."""

import dataclasses
from typing import Optional

from vatt.configs import base_config
from vatt.configs import head as head_cfg
from vatt.configs import loss as loss_cfg


@dataclasses.dataclass
class BackboneConfig(base_config.Config):
  """General common configuration for backbone stack."""

  name: str = 'backbone_stack'
  video_backbone: str = ''
  audio_backbone: str = ''
  text_backbone: str = ''
  video_model_kwargs: base_config.Config = base_config.Config()
  audio_model_kwargs: base_config.Config = base_config.Config()
  text_model_kwargs: base_config.Config = base_config.Config()


@dataclasses.dataclass
class UnifiedBackboneConfig(base_config.Config):
  """General common configuration for unified backbone stack."""

  name: str = 'unified_backbone'
  unified_backbone: str = ''


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """General common configuration for all models."""

  model_name: str = ''
  backbone_config: BackboneConfig = BackboneConfig()
  head_config: head_cfg.HeadStack = head_cfg.HeadStack()
  loss_config: loss_cfg.LossStack = loss_cfg.LossStack()


@dataclasses.dataclass
class CNNkwargs(base_config.Config):
  """Common kwargs for CNN-based backbones (esp. ResNet)."""

  num_classes: Optional[int] = None
  width_multiplier: int = 1
  batch_norm_decay: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_norm_scale: bool = True
  use_xreplica_bn: bool = True


@dataclasses.dataclass
class CNNBackboneConfig(BackboneConfig):
  """General common configuration for CNN-based backbone stack."""

  bn_config: head_cfg.BatchNormConfig = head_cfg.BatchNormConfig()
  use_xreplica_bn: bool = True
  video_backbone: str = 'i3d'
  audio_backbone: str = 'resnet2d_50'
  text_backbone: str = 'linear'
  video_model_kwargs: CNNkwargs = CNNkwargs(
      num_classes=None,
      width_multiplier=2,
      batch_norm_decay=bn_config.momentum,
      batch_norm_epsilon=bn_config.epsilon,
      batch_norm_scale=bn_config.scale,
      use_xreplica_bn=use_xreplica_bn,
    )
  audio_model_kwargs: CNNkwargs = CNNkwargs(
      num_classes=None,
      width_multiplier=1,
      batch_norm_decay=bn_config.momentum,
      batch_norm_epsilon=bn_config.epsilon,
      batch_norm_scale=bn_config.scale,
      use_xreplica_bn=use_xreplica_bn,
      )


@dataclasses.dataclass
class TxBackboneConfig(BackboneConfig):
  """General common configuration for Transformer-based backbone stack."""

  video_backbone: str = 'vit_medium'
  audio_backbone: str = 'wat_base'
  text_backbone: str = 't5_small'


@dataclasses.dataclass
class UTBackboneConfig(UnifiedBackboneConfig):
  """General common configuration for Unified Transformer backbone."""

  unified_backbone: str = 'ut_medium'


@dataclasses.dataclass
class MMVFACModel(ModelConfig):
  """Configs for MMV + MLP-FAC baseline."""

  model_name: str = 'mmv_fac'
  backbone_config: BackboneConfig = CNNBackboneConfig()
  head_config: head_cfg.HeadStack = head_cfg.HeadStack(
      bridge=(
          head_cfg.FACBridge(),
      )
      )
  loss_config: loss_cfg.LossStack = loss_cfg.LossStack(
      bridge=(
          loss_cfg.AsymmetricNCE(),
      )
      )


@dataclasses.dataclass
class TxFACModel(ModelConfig):
  """Configs for Tx + MLP-FAC."""

  model_name: str = 'tx_mlp_fac'
  backbone_config: BackboneConfig = TxBackboneConfig()
  head_config: head_cfg.HeadStack = head_cfg.HeadStack(
      bridge=(
          head_cfg.FACBridge(),
      )
      )
  loss_config: loss_cfg.LossStack = loss_cfg.LossStack(
      bridge=(
          loss_cfg.AsymmetricNCE(),
      )
      )


@dataclasses.dataclass
class UnifiedTxFACModel(ModelConfig):
  """Configs for Unified VATT Tx + MLP-FAC."""

  model_name: str = 'uvatt_mlp_fac'
  backbone_config: UTBackboneConfig = UTBackboneConfig()
  head_config: head_cfg.HeadStack = head_cfg.HeadStack(
      bridge=(
          head_cfg.FACBridge(),
      )
      )
  loss_config: loss_cfg.LossStack = loss_cfg.LossStack(
      bridge=(
          loss_cfg.AsymmetricNCE(),
      )
      )
