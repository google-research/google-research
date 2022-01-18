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
"""Config definition for text models."""

import dataclasses
from typing import Optional

from vatt.configs import base_config


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """General common configuration for text models.

  Attributes:
    name: name of the model
  """
  name: str = 'language_model'
  d_pre_proj: Optional[int] = None
  d_post_proj: Optional[int] = None
  d_model: int = 2048
  d_embedding: int = 300
  vocab_size: int = 2**16
  trainable_embeddings: bool = False
  use_agg_token: bool = False
  is_transformer: bool = False
  activation: Optional[str] = None


@dataclasses.dataclass
class LinearModel(ModelConfig):
  """Configuration of the Linear projection model."""

  name: str = 'linear_lm'


@dataclasses.dataclass
class T5Base(ModelConfig):
  """Configuration of the Base T5 Transformer model."""

  name: str = 't5_base'
  # input parameters
  d_embedding: int = 300  # word2vec: 300, bert: 512
  d_pre_proj: Optional[int] = 768  # word2vec: required, bert: optional
  vocab_size: int = 2**16  # word2vec: 2**16, bert: 30522
  num_relative_buckets: int = 32
  max_relative_distance: int = 64
  use_agg_token: bool = True
  # network size parameters
  d_model: int = 768
  d_kv: int = 64
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  # interinsic parameters
  activation: str = 'gelu'
  dropout_rate: float = 0.1
  layer_norm_epsilon: float = 1e-6
  is_transformer: bool = True


@dataclasses.dataclass
class T5Small(T5Base):
  """Configuration of the Small T5 Transformer model."""

  name: str = 't5_small'
  d_model: int = 512
  d_kv: int = 64
  d_ff: int = 2048
  num_layers: int = 6
  num_heads: int = 8
  d_pre_proj: Optional[int] = 512


@dataclasses.dataclass
class BertBase(ModelConfig):
  """Configuration of the Base BERT Transformer model."""

  name: str = 'bert_base'
  # input parameters
  d_embedding: int = 300  # word2vec: 300, bert: 512
  d_pre_proj: Optional[int] = 768  # word2vec: required, bert: optional
  vocab_size: int = 2**16  # word2vec: 2**16, bert: 30522
  use_agg_token: bool = False
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
  is_transformer: bool = True


@dataclasses.dataclass
class BertSmall(T5Base):
  """Configuration of the Base BERT Transformer model."""

  name: str = 'bert_small'
  d_model: int = 512
  d_kv: int = 64
  d_ff: int = 2048
  num_layers: int = 6
  num_heads: int = 8
  d_pre_proj: Optional[int] = 512
