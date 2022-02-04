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

"""Dataclasses for network parameter containers."""

import dataclasses


@dataclasses.dataclass
class ConvScopeParams:
  """Parameters for tf.slim arg_scope for conv layers."""
  # Whether to use dropout for conv layers.
  dropout: bool = False

  # Dropout regularization strength for conv layers.
  dropout_keep_prob: float = 0.8

  # Whether to use batch_norm for conv layers.
  batch_norm: bool = True

  # Decay factor for batch_norm in conv layers.
  batch_norm_decay: float = 0.99

  # L2 regularization strength on conv and clf weights.
  l2_weight_decay: float = 0.00004


@dataclasses.dataclass
class InceptionV3FCNParams:
  """Parameters for configuring an InceptionV3FCN network."""
  # Prelogit dropout regularization strength.
  prelogit_dropout_keep_prob: float = 0.8

  # Scale number of filters in each Inception(V3) layer by this factor.
  # Minimum number of filters defaults to 16.
  depth_multiplier: float = 0.1

  # Minimum depth for the conv layers. Relevant only when depth_multiplier < 1.
  min_depth: int = 16

  # Stride used in inference. This stride should be a multiple of 16 as
  # InceptionV3 downsamples by 16 by the time it reaches its logits layer. If
  # set to 0, non-FCN mode is assumed and the output is squeezed from
  # (?, 1, 1, classes) to (?, classes).
  inception_fcn_stride: int = 0
