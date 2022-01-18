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

"""Config utilities."""

import dataclasses
from typing import Any, Callable, Optional

from jax import lax


#-------------------------------------------------------
# MLP parameters
#-------------------------------------------------------
@dataclasses.dataclass
class MLPParams:
  """Parameters for NeRF MLP."""
  net_depth: int
  net_width: int
  net_activation: Callable[Ellipsis, Any]
  num_rgb_channels: int
  skip_layer: int


def get_mlp_config(config, net_activation):
  return MLPParams(
      net_depth=config.model.net_depth,
      net_width=config.model.net_width,
      net_activation=net_activation,
      num_rgb_channels=config.model.num_rgb_channels,
      skip_layer=config.model.skip_layer,
  )


#-------------------------------------------------------
# Rendering parameters
#-------------------------------------------------------
@dataclasses.dataclass
class RenderParams:
  """Parameters related to rendering"""
  near: float
  far: float
  lindisp: bool
  white_bkgd: bool
  num_coarse_samples: int
  num_fine_samples: int
  use_viewdirs: bool
  noise_std: float
  num_rgb_channels: int
  rgb_activation: Callable
  sigma_activation: Optional[Callable] = None


def get_render_params(config, rgb_activation, sigma_activation=None):
  return RenderParams(
      near=config.model.near,
      far=config.model.far,
      white_bkgd=config.model.white_bkgd,
      lindisp=config.model.lindisp,
      num_coarse_samples=config.model.num_coarse_samples,
      num_fine_samples=config.model.num_fine_samples,
      use_viewdirs=config.model.use_viewdirs,
      noise_std=config.model.noise_std,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      num_rgb_channels=config.model.num_rgb_channels,
  )


#-------------------------------------------------------
# Position Encoding parameters
#-------------------------------------------------------
@dataclasses.dataclass
class EncodingParams:
  """Parameters for poisitonal encoding"""
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
  """Parameter of lightfield representation"""
  name: str
  # Light Slab parameters
  st_plane: float
  uv_plane: float
  # Encoding parameters
  encoding_name: bool
  min_deg_point: int
  max_deg_point: int


def get_lightfield_params(config):
  config.lightfield.st_plane = config.model.near
  config.lightfield.uv_plane = config.model.far
  return LightFieldParams(
      name=config.lightfield.name,
      st_plane=config.lightfield.st_plane,
      uv_plane=config.lightfield.uv_plane,
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
      num_layers=config.model.transformer_layers,
      attention_heads=config.model.transformer_heads,
      qkv_params=config.model.qkv_dim,
      mlp_params=config.model.transformer_mlp_dim,
      dropout_rate=0.)


def get_view_transformer_params(config):
  return TransformerParams(
      num_layers=config.model.transformer_layers,
      attention_heads=config.model.transformer_heads,
      qkv_params=config.model.qkv_dim,
      mlp_params=config.model.transformer_mlp_dim,
      dropout_rate=0.)


#-------------------------------------------------------
# Epipolar Projection parameters
#-------------------------------------------------------
@dataclasses.dataclass
class EpipolarParams:
  """Parameters for epipolar projection"""
  use_pixel_centers: bool
  min_depth: int
  max_depth: int
  image_height: int
  image_width: int
  num_projections: int
  num_train_views: int
  use_learned_embedding: bool
  learned_embedding_mode: str
  mask_invalid_projection: bool
  use_conv_features: bool
  conv_feature_dim: int
  ksize1: int
  ksize2: int
  interpolation_type: str
  precision: lax.Precision

  def __post_init__(self):
    if self.interpolation_type == "linear":
      assert (self.use_pixel_centers == False
             ), "Cannot use pixel center with linear interpolation"


def get_epipolar_params(config):
  assert config.dataset.image_height != -1, ("Image height for dataset was not "
                                             "set")
  assert config.dataset.image_width != -1, "Image width for dataset was not set"
  assert config.model.near != 0, "0 depth projections can lead to error"
  assert config.dataset.num_train_views != -1, ("Number of train views should "
                                                "be set")
  return EpipolarParams(
      use_pixel_centers=config.dataset.use_pixel_centers,
      min_depth=config.model.near,
      max_depth=config.model.far,
      image_height=config.dataset.image_height,
      image_width=config.dataset.image_width,
      num_projections=config.model.num_projections,
      num_train_views=config.dataset.num_train_views,
      use_learned_embedding=config.model.use_learned_embedding,
      learned_embedding_mode=config.model.learned_embedding_mode,
      mask_invalid_projection=config.model.mask_invalid_projection,
      use_conv_features=config.model.use_conv_features,
      conv_feature_dim=config.model.conv_feature_dim,
      ksize1=config.model.ksize1,
      ksize2=config.model.ksize2,
      interpolation_type=config.model.interpolation_type,
      precision=getattr(lax.Precision, config.model.init_final_precision),
  )
