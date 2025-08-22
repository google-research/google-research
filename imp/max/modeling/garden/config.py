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

"""Model Garden Base Configs."""

import dataclasses

from imp.max.config import registry
from imp.max.config import validators
from imp.max.core import constants
from imp.max.modeling import config as mdl_config
from imp.max.modeling import garden
from imp.max.utils import typing

AggregationType = constants.AggregationType
ClassificationHead = typing.ClassificationHead
ShardingAxes = typing.ShardingAxes
register_with_class = registry.Registrar.register_with_class


# pylint: disable=line-too-long
@register_with_class(garden.ViT)
@validators.lock
@dataclasses.dataclass
class ViT(mdl_config.RematScan, mdl_config.Model):
  """Base ViT config (86M)."""

  name: str = 'vit_base'
  # Input params
  batch_size: int = 8
  image_size: tuple[int, int, int, int] = (1, 224, 224, 3)
  patch_size: tuple[int, Ellipsis] = (1, 16, 16)
  # Input sharding annotations
  pos_encode_embed_shardings: ShardingAxes = ()
  pos_encode_layernorm_shardings: ShardingAxes = ()
  token_raw_to_embed_kernel_shardings: ShardingAxes = ()
  tokens_shardings: ShardingAxes = ('data', None, None, None)  # (b, n, t, d)
  # Encoder params
  d_model: int = 768
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  use_bias: bool = True
  dropout_rate: float = 0.1
  d_post_proj: int | None = None
  post_proj_position: str | None = None
  qk_layernorm: bool = False
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.
  approximate_gelu: bool = True
  # Backbone sharding annotations
  scan_sharding_axis: str | None = None  # Pipelining is off by default
  layernorm_shardings: ShardingAxes = ()
  mha_qkv_kernel_shardings: ShardingAxes = (None, 'model', None)  # (d, h, d_h)
  mha_out_kernel_shardings: ShardingAxes = ('model', None, None)  # (h, d_h, d)
  mha_activation_shardings: ShardingAxes = ('data', None, None, 'model', None)  # (b, n, t, h, d_h)
  ffn_inner_kernel_shardings: ShardingAxes = (None, 'model')  # (d, d_ff)
  ffn_outer_kernel_shardings: ShardingAxes = ('model', None)  # (d_ff, d)
  ffn_intermediate_shardings: ShardingAxes = ('data', None, None, 'model')  # (b, n, t, d_ff)
  # Classification
  num_classes: ClassificationHead = None
  # Misc
  head_bias_init: float = 0.
  aggregation_type: str = AggregationType.SPECIAL_TOKEN
  positional_embedding: str = 'learned_1d'
# pylint: enable=line-too-long


# pylint: disable=line-too-long
@dataclasses.dataclass
class BaseIntegratedMultimodalModel(mdl_config.RematScan, mdl_config.Model):
  """Common config for IMP/IMPeGe variants.

  Attributes:
    input_batch_size: The size of the input batch.
    vision_input_size: The size of the input vision modality, i.e.,
      (num_frames, height, width, num_channels).
    vision_patch_size: The size of each patch to divide the vision input.
    waveform_input_size: The total samples of the input audio waveform.
    waveform_patch_size: The patching window size for waveform tokenization.
    spectrogram_input_size: The expected spectrogram input size.
    spectrogram_patch_size: The patching window size for spectrogram
      tokenization.
    text_input_size: The max number of tokens in the text input.
    text_vocab_size: The size of the text vocabulary.
    text_embed_size: The dimension of the text embedding.
    dropout_rate: The rate to apply dropout to channels.
    d_model: The dimension of the model embeddings in the vector space.
    d_ff: The dimension of the feedforward layer embeddings in the vector space.
    num_heads: The number of heads in the multi-head attention layers.
    use_bias: Whether to use bias in the model layers.
    common_space_type: The type of the common space projection (e.g. disjoint,
      joint, or FAC as in VATT).
    d_common: The dimension of the common space projection.
    aggregation_type: The aggregation type in the aggregator head (e.g. global
      average pooling, etc.)
    vision_classes: The number of classes for vision perception.
    waveform_classes: The number of classes for audio waveform perception.
    spectrogram_classes: The number of classes for spectrogram perception.
    text_classes: The number of classes (i.e., vocab size) for text perception.
    vision_targets: The number of targets for vision generation.
    waveform_targets: The number of targets for audio waveform generation.
    spectrogram_targets: The number of targets for spectrogram generation.
    text_targets: The number of targets (i.e., vocab size) for text generation.
    freeze_embeddings: A set of modalities to freeze their embedding projection.
    temperature_init: The initial value of the temperature for the NCE loss.
  """
  # Input params
  input_batch_size: int = 8
  vision_input_size: tuple[int, int, int, int] = (32, 224, 224, 3)
  vision_patch_size: tuple[int, int, int] = (4, 16, 16)
  vision_vocab_size: int = 2048
  vision_embed_size: int = 768
  waveform_input_size: int = 153600
  waveform_patch_size: int = 256
  waveform_vocab_size: int = 2048
  waveform_embed_size: int = 768
  spectrogram_input_size: tuple[int, int] = (30, 80)
  spectrogram_patch_size: tuple[int, int] = (5, 16)
  spectrogram_vocab_size: int = 2048
  spectrogram_embed_size: int = 768
  text_input_size: int = 32
  text_vocab_size: int = 32100
  text_embed_size: int = 768
  dropout_rate: float = 0.
  # Input sharding annotations
  pos_encode_embed_shardings: ShardingAxes = ()
  pos_encode_layernorm_shardings: ShardingAxes = ()
  token_raw_to_embed_kernel_shardings: ShardingAxes = ()
  token_id_to_embed_kernel_shardings: ShardingAxes = ()
  tokens_shardings: ShardingAxes = ('data', None, None, None)  # (b, n, t, d)
  # Backbone params
  d_model: int = 768
  d_ff: int = 3072
  num_heads: int = 12
  use_bias: bool = False
  qk_layernorm: bool = False
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.
  approximate_gelu: bool = True
  # Backbone sharding annotations
  scan_sharding_axis: str | None = None  # Pipelining is off by default
  layernorm_shardings: ShardingAxes = ()
  mha_qkv_kernel_shardings: ShardingAxes = (None, 'model', None)  # (d, h, d_h)
  mha_out_kernel_shardings: ShardingAxes = ('model', None, None)  # (h, d_h, d)
  mha_activation_shardings: ShardingAxes = ('data', None, None, 'model', None)  # (b, n, t, h, d_h)
  ffn_inner_kernel_shardings: ShardingAxes = (None, 'model')  # (d, d_ff)
  ffn_outer_kernel_shardings: ShardingAxes = ('model', None)  # (d_ff, d)
  ffn_intermediate_shardings: ShardingAxes = ('data', None, None, 'model')  # (b, n, t, d_ff)
  # Common space params
  common_space_type: str = constants.CommonSpace.DISJOINT
  d_common: int = 1024
  aggregation_type: str = AggregationType.GLOBAL_AVERAGE_POOL
  # Classification projection head
  vision_classes: ClassificationHead = None
  waveform_classes: ClassificationHead = None
  spectrogram_classes: ClassificationHead = None
  text_classes: ClassificationHead = None
  # Target projection head
  vision_targets: int | None = None
  waveform_targets: int | None = None
  spectrogram_targets: int | None = None
  text_targets: int | None = None
  # Initialization
  freeze_embeddings: tuple[str, Ellipsis] = ()
  temperature_init: float = 0.2
# pylint: enable=line-too-long


@register_with_class(garden.IMP)
@validators.lock
@dataclasses.dataclass
class IMP(BaseIntegratedMultimodalModel):
  """Integrated Multimodal Perception (IMP) config."""

  name: str = 'imp'
  num_layers: int = 12
  d_post_proj: int = BaseIntegratedMultimodalModel.d_model


# pylint: disable=line-too-long
@register_with_class(garden.SparseMoeIMP)
@validators.lock
@dataclasses.dataclass
class SparseMoeIMP(mdl_config.SparseMixtureOfExperts, IMP):
  """Sparse MoE IMP config."""

  name: str = 'sparse_moe_imp'
  qk_layernorm: bool = True
  model_axis_size: int = 1
  model_axis_name: str = 'model'
  tokens_shardings: ShardingAxes = (('expert', 'data'), None, None, None)  # (b, n, t, d)
  mha_activation_shardings: ShardingAxes = (('expert', 'data'), None, None, 'model', None)  # (b, n, t, h, d_h)
  ffn_intermediate_shardings: ShardingAxes = (('expert', 'data'), None, None, 'model')  # (b, n, t, d_ff)
  routed_ffn_intermediate_shardings: ShardingAxes = ('data', 'model')  # (r, d_ff)


@register_with_class(garden.SoftMoeIMP)
@validators.lock
@dataclasses.dataclass
class SoftMoeIMP(mdl_config.SoftMixtureOfExperts, IMP):
  """Soft MoE IMP config."""

  name: str = 'soft_moe_imp'
  qk_layernorm: bool = True
  model_axis_size: int = 1
  model_axis_name: str = 'model'
  tokens_shardings: ShardingAxes = (('expert', 'data'), None, None, None)  # (b, n, t, d)
  mha_activation_shardings: ShardingAxes = (('expert', 'data'), None, None, 'model', None)  # (b, n, t, h, d_h)
  ffn_intermediate_shardings: ShardingAxes = (('expert', 'data'), None, None, 'model')  # (b, n, t, d_ff)
  routed_ffn_intermediate_shardings: ShardingAxes = ('data', 'model')  # (r, d_ff)
# pylint: enable=line-too-long


@register_with_class(garden.IMPeGe)
@validators.lock
@dataclasses.dataclass
class IMPeGe(BaseIntegratedMultimodalModel):
  """Integrated Multimodal Perception and Generation (IMPeGe) config."""

  name: str = 'impege'
  # Backbone arguments.
  num_encoder_layers: int = 12
  num_decoder_layers: int = 12
  d_post_encoder_proj: int = BaseIntegratedMultimodalModel.d_model
  d_post_decoder_proj: int = BaseIntegratedMultimodalModel.d_model
