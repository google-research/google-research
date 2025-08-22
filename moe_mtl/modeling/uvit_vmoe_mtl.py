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

"""Universal Vision Transformer + VMoE with Multi-task Learning."""
import enum
import functools
from typing import Any, Dict, Mapping, Optional
from absl import logging

from flax import linen as nn
import gin
import numpy as np


from moe_mtl.modeling import vmoe_mtl


KwArgs = Mapping[str, Any]


@gin.register
class UViTVMoEMTL(base.BaseModel):
  """Universal Vision Transformer (https://arxiv.org/pdf/2112.09747.pdf)."""
  encoder: KwArgs
  patch_size_det: int = 8
  patch_size_cls: int = 8
  num_layers: int = 12
  hidden_dim: int = 384
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 6
  window_size_det: Optional[int] = None
  window_size_cls: Optional[int] = None
  pool_type_cls: str = 'gap'
  pool_type_det: str = 'gap'
  patch_overlap: int = 0
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  use_taps: bool = False
  use_shared_embed_conv: bool = False
  use_single_router: bool = False
  use_stable: bool = False

  @nn.compact
  def __call__(self, x_det, x_cls, second_stage=False):

    logging.info(self.use_stable)

    if not self.use_stable:
      cls = vmoe_mtl.VisionTransformerMoeMTL
    else:
      raise NotImplementedError
    model = cls(
        encoder=self.encoder,
        patch_size_det=(self.patch_size_det, self.patch_size_det),
        patch_size_cls=(self.patch_size_cls, self.patch_size_cls),
        width=self.hidden_dim,
        depth=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        window_size_det=self.window_size_det,
        window_size_cls=self.window_size_cls,
        pool_type_cls=self.pool_type_cls,
        pool_type_det=self.pool_type_det,
        patch_overlap=self.patch_overlap,
        dropout=self.dropout_rate,
        use_taps=self.use_taps,
        use_shared_embed_conv=self.use_shared_embed_conv,
        use_single_router=self.use_single_router,

        stoch_depth=self.stochastic_depth)
    if self.use_stable:
      x_det, x_cls, metrics = model(
          x_det,
          x_cls,
          train=self.mode == base.ExecutionMode.TRAIN,
          second_stage=second_stage)
    else:
      x_det, x_cls, metrics = model(
          x_det, x_cls, train=self.mode == base.ExecutionMode.TRAIN)

    level = round(np.log2(self.patch_size_det))
    return {level: x_det}, x_cls, metrics


@gin.constants_from_enum
class UVitVMoEMTLSize(enum.Enum):
  """UViT model scales including vanilla ViT scales and UViT scales."""

  # ViT scaling rule
  HALF_SMALL = 'half_small'
  HALF_SMALL_LARGER = 'half_small_larger'
  HALF_SMALL_LARGER_2 = 'half_small_larger_2'
  TINY = 'tiny'
  SMALL = 'small'
  SMALL_LARGER = 'small_larger'
  SMALL_LARGER_2 = 'small_larger_2'
  BASE = 'base'
  BASE_LARGER = 'base_larger'
  BASE_LARGER_2 = 'base_larger_2'
  LARGE = 'large'
  HUGE = 'huge'
  GIANT = 'giant'
  # UViT scaling rule. Better for COCO detection
  UVIT_VMOE_MTL_EXTRA_TINY = 'uvit_vmoe_mtl_extra_tiny'
  UVIT_VMOE_MTL_TINY = 'uvit_vmoe_mtl_tiny'
  UVIT_VMOE_MTL_SMALL = 'uvit_vmoe_mtl_small'
  UVIT_VMOE_MTL_BASE = 'uvit_vmoe_mtl_base'
  UVIT_VMOE_MTL_BASE_LARGER = 'uvit_vmoe_mtl_base_larger'


MODEL_CONFIGS = {
    # Training data-efficient image transformers & distillation through attention  # pylint: disable=line-too-long
    # https://arxiv.org/abs/2012.12877
    UVitVMoEMTLSize.HALF_SMALL: {
        'num_layers': 6,
        'hidden_dim': 384,
        'mlp_dim': 1536,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.HALF_SMALL_LARGER: {
        'num_layers': 6,
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.HALF_SMALL_LARGER_2: {
        'num_layers': 6,
        'hidden_dim': 480,
        'mlp_dim': 1920,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.TINY: {  # 6M params.
        'num_layers': 12,
        'hidden_dim': 192,
        'mlp_dim': 768,
        'num_heads': 3,
    },
    UVitVMoEMTLSize.SMALL: {  # 22M params.
        'num_layers': 12,
        'hidden_dim': 384,
        'mlp_dim': 1536,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.SMALL_LARGER: {  # 22M params.
        'num_layers': 12,
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.SMALL_LARGER_2: {  # 22M params.
        'num_layers': 12,
        'hidden_dim': 480,
        'mlp_dim': 1920,
        'num_heads': 6,
    },
    # An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    # https://arxiv.org/abs/2010.11929
    UVitVMoEMTLSize.BASE: {  # 86M params.
        'num_layers': 12,
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
    },
    UVitVMoEMTLSize.BASE_LARGER: {  # 86M params.
        'num_layers': 12,
        'hidden_dim': 1536,
        'mlp_dim': 6144,
        'num_heads': 12,
    },
    UVitVMoEMTLSize.BASE_LARGER_2: {  # 86M params.
        'num_layers': 12,
        'hidden_dim': 900,
        'mlp_dim': 3600,
        'num_heads': 12,
    },
    UVitVMoEMTLSize.LARGE: {  # 307M params.
        'num_layers': 24,
        'hidden_dim': 1024,
        'mlp_dim': 4096,
        'num_heads': 16,
    },
    UVitVMoEMTLSize.HUGE: {  # 632M params.
        'num_layers': 32,
        'hidden_dim': 1280,
        'mlp_dim': 5120,
        'num_heads': 16,
    },
    # Scaling Vision Transformers
    # https://arxiv.org/abs/2106.04560
    UVitVMoEMTLSize.GIANT: {  # 1843M params.
        'num_layers': 48,
        'hidden_dim': 1664,
        'mlp_dim': 8192,
        'num_heads': 16,
    },
    # A Simple Single-Scale Vision Transformer for Object Localization
    # https://arxiv.org/pdf/2112.09747.pdf
    UVitVMoEMTLSize.UVIT_VMOE_MTL_EXTRA_TINY: {  # 13.5M params.
        'num_layers': 2,
        'hidden_dim': 2,
        'mlp_dim': 8,
        'num_heads': 1,
    },
    UVitVMoEMTLSize.UVIT_VMOE_MTL_TINY: {  # 13.5M params.
        'num_layers': 18,
        'hidden_dim': 222,
        'mlp_dim': 888,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.UVIT_VMOE_MTL_SMALL: {  # 21.7M params.
        'num_layers': 18,
        'hidden_dim': 288,
        'mlp_dim': 1152,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.UVIT_VMOE_MTL_BASE: {  # 36.9M params.
        'num_layers': 18,
        'hidden_dim': 384,
        'mlp_dim': 1536,
        'num_heads': 6,
    },
    UVitVMoEMTLSize.UVIT_VMOE_MTL_BASE_LARGER: {  # 36.9M params.
        'num_layers': 18,
        'hidden_dim': 480,
        'mlp_dim': 1920,
        'num_heads': 6,
    },
}


@gin.register
def get_uvit_vmoe_mtl_model_fn(
    model_size,
    encoder_config,
    patch_size_det,
    patch_size_cls,
    window_size_det = None,
    window_size_cls = None,
    patch_overlap = 0,
    pool_type_cls = 'gap',
    pool_type_det = 'gap',
    use_stable = False,
    use_taps = False,
    use_single_router = False,
    kwargs = None):
  """Returns a function that constructs a ViT MTL module."""
  if kwargs is None:
    kwargs = {}
  return functools.partial(
      UViTVMoEMTL,
      **kwargs,
      **MODEL_CONFIGS[model_size],
      use_stable=use_stable,
      use_taps=use_taps,
      use_single_router=use_single_router,
      encoder=encoder_config,
      patch_size_det=patch_size_det,
      patch_size_cls=patch_size_cls,
      pool_type_det=pool_type_det,
      pool_type_cls=pool_type_cls,
      window_size_det=window_size_det,
      window_size_cls=window_size_cls,
      patch_overlap=patch_overlap)

