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

"""Configuration of the IMP models."""

import dataclasses

from imp.max.config import registry
from imp.max.config import validators
from imp.max.modeling import garden
from imp.max.modeling.garden import config as garden_config

register_with_class = registry.Registrar.register_with_class

# ----------------------------------------------------------------------
# ------------- Integrated Multimodal Perception Configs. --------------
# ----------------------------------------------------------------------


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class SmallIMP(garden_config.IMP):
  """Small IMP config (21M)."""

  name: str = 'imp_small'
  d_model: int = 384
  d_ff: int = 1536
  num_layers: int = 12
  num_heads: int = 6
  d_post_proj: int = 384
  text_embed_size: int = d_model


@register_with_class(garden.SparseMoeIMP)
@validators.validate
@dataclasses.dataclass
class SparseMoeSmallIMP(garden_config.SparseMoeIMP):
  """Sparse MoE Small IMP config (21M)."""

  name: str = 'sparse_imp_small'
  d_model: int = 384
  d_ff: int = 1536
  num_layers: int = 12
  num_heads: int = 6
  d_post_proj: int = 384
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class BaseIMP(garden_config.IMP):
  """Base Integrated Multimodal Perceiver config (88M)."""

  name: str = 'imp_base'
  d_model: int = 768
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  d_post_proj: int = 768
  text_embed_size: int = d_model


@register_with_class(garden.SparseMoeIMP)
@validators.validate
@dataclasses.dataclass
class SparseMoeBaseIMP(garden_config.SparseMoeIMP):
  """Sparse MoE MoE Base Integrated Multimodal Perceiver config (90M)."""

  name: str = 'sparse_imp_base'
  d_model: int = 768
  d_ff: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  d_post_proj: int = 768
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class MediumIMP(garden_config.IMP):
  """Configuration of the Medium Integrated Multimodal Transformer (155M)."""

  name: str = 'imp_medium'
  d_model: int = 1024
  d_ff: int = 4096
  num_layers: int = 12
  num_heads: int = 16
  d_post_proj: int = 1024
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class LargeIMP(garden_config.IMP):
  """Configuration of the Large Integrated Multimodal Transformer (306M)."""

  name: str = 'imp_large'
  d_model: int = 1024
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16
  d_post_proj: int = 1024
  text_embed_size: int = d_model


@register_with_class(garden.SparseMoeIMP)
@validators.validate
@dataclasses.dataclass
class SparseMoeLargeIMP(garden_config.SparseMoeIMP):
  """Sparse MoE MoE Large Integrated Multimodal Perceiver config (350M)."""

  name: str = 'sparse_imp_large'
  d_model: int = 1024
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 16
  d_post_proj: int = 1024
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class XLargeIMP(garden_config.IMP):
  """Configuration of the X-Large Integrated Multimodal Transformer (535M)."""

  name: str = 'imp_xlarge'
  d_model: int = 1536
  d_ff: int = 4096
  num_layers: int = 24
  num_heads: int = 24
  d_post_proj: int = 1536
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class XXLargeIMP(garden_config.IMP):
  """Configuration of the XX-Large Integrated Multimodal Transformer (713M)."""

  name: str = 'imp_xxlarge'
  d_model: int = 1536
  d_ff: int = 4096
  num_layers: int = 32
  num_heads: int = 24
  d_post_proj: int = 1536
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class HugeIMP(garden_config.IMP):
  """Configuration of the Huge Integrated Multimodal Transformer (1.2B)."""

  name: str = 'imp_huge'
  d_model: int = 1536
  d_ff: int = 8192
  num_layers: int = 32
  num_heads: int = 24
  d_post_proj: int = 1536
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class XHugeIMP(garden_config.IMP):
  """Configuration of the Huge Integrated Multimodal Transformer (1.6B)."""

  name: str = 'imp_xhuge'
  d_model: int = 2048
  d_ff: int = 8192
  num_layers: int = 32
  num_heads: int = 32
  d_post_proj: int = 2048
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class GiantIMP(garden_config.IMP):
  """Configuration of the Giant Integrated Multimodal Transformer (3.2B)."""

  name: str = 'imp_giant'
  d_model: int = 3072
  d_ff: int = 10240
  num_layers: int = 32
  num_heads: int = 48
  d_post_proj: int = 3072
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class GiganticIMP(garden_config.IMP):
  """Configuration of the Gigantic Integrated Multimodal Transformer (6.5B)."""

  name: str = 'imp_gigantic'
  d_model: int = 4096
  d_ff: int = 16384
  num_layers: int = 32
  num_heads: int = 64
  d_post_proj: int = 4096
  text_embed_size: int = d_model


@register_with_class(garden.IMP)
@validators.validate
@dataclasses.dataclass
class GinormousIMP(garden_config.IMP):
  """Configuration of the Ginormous Integrated Multimodal Transformer (11B)."""

  name: str = 'imp_ginormous'
  d_model: int = 4096
  d_ff: int = 32768
  num_layers: int = 32
  num_heads: int = 64
  d_post_proj: int = 4096
  text_embed_size: int = d_model


# ----------------------------------------------------------------------
# ------ Integrated Multimodal Perception & Generation Configs. --------
# ----------------------------------------------------------------------


@register_with_class(garden.IMPeGe)
@validators.validate
@dataclasses.dataclass
class SmallIMPeGe(garden_config.IMPeGe):
  """Small IMPeGe config (21M)."""

  name: str = 'impege_small'
  d_model: int = 384
  d_ff: int = 1536
  num_heads: int = 6
  num_encoder_layers: int = 12
  num_decoder_layers: int = 12
  d_post_encoder_proj: int = 384
  d_post_decoder_proj: int = 384
  text_embed_size: int = d_model


@register_with_class(garden.IMPeGe)
@validators.validate
@dataclasses.dataclass
class BaseIMPeGe(garden_config.IMPeGe):
  """Base IMPeGe config (88M)."""

  name: str = 'impege_base'
  d_model: int = 768
  d_ff: int = 3072
  num_heads: int = 12
  num_encoder_layers: int = 12
  num_decoder_layers: int = 12
  d_post_encoder_proj: int = 768
  d_post_decoder_proj: int = 768
  text_embed_size: int = d_model
