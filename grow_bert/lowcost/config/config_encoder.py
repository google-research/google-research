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

"""Configurations for Encoders."""

from typing import Optional
import dataclasses

from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders


@dataclasses.dataclass
class SmallEncoderConfig(base_config.Config):
  """Encoder config for the lowcost stage."""
  net2net_ratio: float = 0.25
  net2net_layers: Optional[int] = None
  fcfact_ratio: float = 0.2
  fcfact_layers: Optional[int] = None
  kq_ratio: Optional[float] = None
  lightatt_layers: Optional[int] = None
  input_pool_name: Optional[str] = None
  input_pool_size: Optional[int] = None
  override_num_layers: Optional[int] = None


@dataclasses.dataclass
class TransformerEncoderConfig(base_config.Config):
  """BERT encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = 'gelu'
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None

  # Small model configuration
  net2net_ratio: float = 0.25
  net2net_layers: Optional[int] = None
  fcfact_ratio: float = 0.2
  fcfact_layers: Optional[int] = None
  kq_ratio: Optional[float] = None
  lightatt_layers: Optional[int] = None
  input_pool_name: Optional[str] = None
  input_pool_size: Optional[int] = None


def from_bert_encoder_config(official_cfg, small_cfg):
  """Create the encoder from a config object."""
  if isinstance(official_cfg, encoders.BertEncoderConfig):
    official_cfg = official_cfg.as_dict()
  if isinstance(small_cfg, SmallEncoderConfig):
    small_cfg = small_cfg.as_dict()
  num_layers = official_cfg['num_layers']
  if small_cfg['override_num_layers'] is not None:
    num_layers = small_cfg['override_num_layers']
  assert small_cfg['fcfact_layers'] is None or small_cfg[
      'net2net_layers'] is None
  return TransformerEncoderConfig(
      vocab_size=official_cfg['vocab_size'],
      hidden_size=official_cfg['hidden_size'],
      num_layers=num_layers,
      num_attention_heads=official_cfg['num_attention_heads'],
      hidden_activation=official_cfg['hidden_activation'],
      intermediate_size=official_cfg['intermediate_size'],
      dropout_rate=official_cfg['dropout_rate'],
      attention_dropout_rate=official_cfg['attention_dropout_rate'],
      max_position_embeddings=official_cfg['max_position_embeddings'],
      type_vocab_size=official_cfg['type_vocab_size'],
      initializer_range=official_cfg['initializer_range'],
      embedding_size=official_cfg['embedding_size'],
      net2net_ratio=small_cfg['net2net_ratio'],
      net2net_layers=small_cfg['net2net_layers'],
      fcfact_ratio=small_cfg['fcfact_ratio'],
      fcfact_layers=small_cfg['fcfact_layers'],
      kq_ratio=small_cfg['kq_ratio'],
      lightatt_layers=small_cfg['lightatt_layers'],
      input_pool_name=small_cfg['input_pool_name'],
      input_pool_size=small_cfg['input_pool_size'],
  )
