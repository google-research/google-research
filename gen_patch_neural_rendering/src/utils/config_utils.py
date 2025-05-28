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

"""Config utilities."""

import dataclasses
from typing import Any, Callable, Optional

from jax import lax


#-------------------------------------------------------
# Rendering parameters
#-------------------------------------------------------
@dataclasses.dataclass
class RenderParams:
  """Parameters related to rendering."""
  near: float
  far: float
  white_bkgd: bool
  num_rgb_channels: int
  rgb_activation: Callable[Ellipsis, Any]
  sigma_activation: Optional[Callable[Ellipsis, Any]] = None


def get_render_params(config, rgb_activation, sigma_activation=None):
  return RenderParams(
      near=config.model.near,
      far=config.model.far,
      white_bkgd=config.model.white_bkgd,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      num_rgb_channels=config.model.num_rgb_channels,
  )


#-------------------------------------------------------
# Position Encoding parameters
#-------------------------------------------------------
@dataclasses.dataclass
class EncodingParams:
  """Parameters for poisitonal encoding."""
  name: str
  min_deg_point: int
  max_deg_point: int
  deg_view: int


def get_encoding_params(config):
  return EncodingParams(
      name=config.model.mapping_type,
      min_deg_point=config.model.min_deg_point,
      max_deg_point=config.model.max_deg_point,
      deg_view=config.model.deg_view,
  )


#-------------------------------------------------------
# LightField parameters
#-------------------------------------------------------
@dataclasses.dataclass
class LightFieldParams:
  """Parameter of lightfield representation."""
  name: str
  # Encoding parameters
  encoding_name: bool
  min_deg_point: int
  max_deg_point: int


def get_lightfield_params(config):
  return LightFieldParams(
      name=config.lightfield.name,
      encoding_name=config.lightfield.encoding_name,
      min_deg_point=config.lightfield.min_deg_point,
      max_deg_point=config.lightfield.max_deg_point,
  )


#-------------------------------------------------------
# Transformer parameters
#-------------------------------------------------------
@dataclasses.dataclass
class TransformerParams:
  """Parameters for Transformer."""
  num_layers: int
  attention_heads: int
  qkv_params: Optional[int] = None
  mlp_params: Optional[int] = None
  dropout_rate: float = 0.

  def __post_init__(self):
    assert self.dropout_rate == 0, "Dropout not supported yet."


def get_epipolar_transformer_params(config):
  return TransformerParams(
      num_layers=config.model.epi_transformer_layers,
      attention_heads=config.model.transformer_heads,
      qkv_params=config.model.qkv_dim,
      mlp_params=config.model.transformer_mlp_dim,
      dropout_rate=0.)


def get_view_transformer_params(config):
  return TransformerParams(
      num_layers=config.model.view_transformer_layers,
      attention_heads=config.model.transformer_heads,
      qkv_params=config.model.qkv_dim,
      mlp_params=config.model.transformer_mlp_dim,
      dropout_rate=0.)


#-------------------------------------------------------
# Epipolar Projection parameters
#-------------------------------------------------------
@dataclasses.dataclass
class EpipolarParams:
  """Parameters for epipolar projection."""
  use_pixel_centers: bool
  min_depth: int
  max_depth: int
  num_projections: int
  mask_invalid_projection: bool
  conv_feature_dim: int
  patch_size: int
  interpolation_type: str
  precision: lax.Precision
  normalize_ref_image: bool = False

  def __post_init__(self):
    if self.interpolation_type == "linear":
      assert (not self.use_pixel_centers
             ), "Cannot use pixel center with linear interpolation"


def get_epipolar_params(config):
  assert config.model.near != 0, "0 depth projections can lead to error"

  return EpipolarParams(
      use_pixel_centers=config.dataset.use_pixel_centers,
      min_depth=config.model.near,
      max_depth=config.model.far,
      num_projections=config.model.num_projections,
      mask_invalid_projection=config.model.mask_invalid_projection,
      conv_feature_dim=config.model.conv_feature_dim,
      patch_size=config.model.patch_size,
      interpolation_type=config.model.interpolation_type,
      precision=getattr(lax.Precision, config.model.init_final_precision),
      normalize_ref_image=config.model.normalize_ref_image,
  )
