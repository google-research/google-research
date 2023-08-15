# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""OpenAI's CLIP models in Flax.

The implementation is based on an initial port of code in
https://github.com/openai/CLIP to JAX by Ben Poole.
"""

import functools
from typing import Any, Optional, Sequence, Union

import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as np

# Match PyTorch default LayerNorm epsilon of 1e-5 (FLAX defaults to 1e-6).
# LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-6)

# For each layer, we set window_attention_size = 4, which is equivalent to 16
# window_size (for 1024 input, 16 patch size), and similar to 14 used in ViTDet.
# Other layers with size None are with global attention.
_WINDOW_ATTENTION_SIZE_VIT_BASE = (
    4, 4, None, 4, 4, None, 4, 4, None, 4, 4, None,
)
_WINDOW_ATTENTION_SIZE_VIT_LARGE = (
    4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4, None,
    4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4, None,
)
_WINDOW_ATTENTION_SIZE_VIT_HUGE = (
    4, 4, 4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4, 4, 4, None,
    4, 4, 4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4, 4, 4, None,
)
_WINDOW_ATTENTION_SIZE_FROZEN_VIT_BASE = (
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None,
)
_WINDOW_ATTENTION_SIZE_FROZEN_VIT_LARGE = (
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None,
)
_WINDOW_ATTENTION_SIZE_FROZEN_VIT_HUGE = (
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None,
)

_CLIP_MODEL_CFG = {
    'vit_b32': dict(embed_dim=512,
                    vocab_size=49408,
                    vision_num_layers=12,
                    vision_features=768,
                    vision_patch_size=32,
                    vision_num_heads=12,
                    window_attention_size=_WINDOW_ATTENTION_SIZE_VIT_BASE,
                    text_features=512,
                    text_num_heads=8,
                    text_num_layers=12),
    'vit_b16': dict(embed_dim=512,
                    vocab_size=49408,
                    vision_num_layers=12,
                    vision_features=768,
                    vision_patch_size=16,
                    vision_num_heads=12,
                    window_attention_size=_WINDOW_ATTENTION_SIZE_VIT_BASE,
                    window_attention_size_frozen=(
                        _WINDOW_ATTENTION_SIZE_FROZEN_VIT_BASE),
                    text_features=512,
                    text_num_heads=8,
                    text_num_layers=12),
    'vit_l14': dict(embed_dim=768,
                    vocab_size=49408,
                    vision_num_layers=24,
                    vision_features=1024,
                    vision_patch_size=14,
                    vision_num_heads=16,
                    window_attention_size=_WINDOW_ATTENTION_SIZE_VIT_LARGE,
                    window_attention_size_frozen=(
                        _WINDOW_ATTENTION_SIZE_FROZEN_VIT_LARGE),
                    text_features=768,
                    text_num_heads=12,
                    text_num_layers=12),
    'vit_l16': dict(embed_dim=768,
                    vocab_size=49408,
                    vision_num_layers=24,
                    vision_features=1024,
                    vision_patch_size=16,
                    vision_num_heads=16,
                    window_attention_size=_WINDOW_ATTENTION_SIZE_VIT_LARGE,
                    window_attention_size_frozen=(
                        _WINDOW_ATTENTION_SIZE_FROZEN_VIT_LARGE),
                    text_features=768,
                    text_num_heads=12,
                    text_num_layers=12),
    'vit_h16': dict(embed_dim=768,
                    vocab_size=49408,
                    vision_num_layers=32,
                    vision_features=1280,
                    vision_patch_size=16,
                    vision_num_heads=16,
                    window_attention_size=_WINDOW_ATTENTION_SIZE_VIT_HUGE,
                    window_attention_size_frozen=(
                        _WINDOW_ATTENTION_SIZE_FROZEN_VIT_HUGE),
                    text_features=768,
                    text_num_heads=12,
                    text_num_layers=12),
    'resnet_50': dict(embed_dim=1024,
                      vocab_size=49408,
                      vision_num_layers=(3, 4, 6, 3),
                      vision_features=64,
                      text_features=512,
                      text_num_heads=8,
                      text_num_layers=12),
    'resnet_50x4': dict(embed_dim=640,
                        vocab_size=49408,
                        vision_num_layers=(4, 6, 10, 6),
                        vision_features=80,
                        text_features=640,
                        text_num_heads=10,
                        text_num_layers=12),
    'resnet_50x16': dict(embed_dim=768,
                         vocab_size=49408,
                         vision_num_layers=(6, 8, 18, 8),
                         vision_features=96,
                         text_features=768,
                         text_num_heads=12,
                         text_num_layers=12),
    'resnet_50x64': dict(embed_dim=1024,
                         vocab_size=49408,
                         vision_num_layers=(3, 15, 36, 10),
                         vision_features=128,
                         text_features=1024,
                         text_num_heads=16,
                         text_num_layers=12),
    'resnet_101': dict(embed_dim=512,
                       vocab_size=49408,
                       vision_num_layers=(3, 4, 23, 3),
                       vision_features=64,
                       text_features=512,
                       text_num_heads=8,
                       text_num_layers=12)
}


def quick_gelu(x):
  return x * jax.nn.sigmoid(1.702 * x)


class Shortcut(nn.Module):
  """Shortcut in ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
    dtype: Data type of CLIP Bottleneck.
  """
  features: int
  stride: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = nn.avg_pool(x, (self.stride, self.stride), (self.stride, self.stride))
    x = nn.Conv(
        self.features, (1, 1), strides=(1, 1), use_bias=False, dtype=self.dtype,
        name='0')(x)
    x = nn.BatchNorm(use_running_average=True, dtype=self.dtype,
                     name='1')(x)
    return x


class Bottleneck(nn.Module):
  """Bottleneck layer of ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
    expansion: Expansion of feature dimension.
    dtype: Data type of CLIP Bottleneck.
  """
  features: int
  stride: int = 1
  expansion: int = 4
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    conv1 = nn.Conv(self.features, (1, 1), use_bias=False, dtype=self.dtype,
                    name='conv1')
    bn1 = nn.BatchNorm(use_running_average=True, dtype=self.dtype,
                       name='bn1')

    conv2 = nn.Conv(self.features, (3, 3), padding=[(1, 1), (1, 1)],
                    use_bias=False, dtype=self.dtype,
                    name='conv2')
    bn2 = nn.BatchNorm(use_running_average=True, dtype=self.dtype,
                       name='bn2')

    conv3 = nn.Conv(
        self.features * self.expansion, (1, 1),
        use_bias=False,
        dtype=self.dtype,
        name='conv3')
    bn3 = nn.BatchNorm(use_running_average=True, dtype=self.dtype,
                       name='bn3')

    out = nn.relu(bn1(conv1(x)))
    out = nn.relu(bn2(conv2(out)))
    out = nn.avg_pool(out, (self.stride, self.stride),
                      (self.stride, self.stride))
    out = bn3(conv3(out))

    downsample = (
        self.stride > 1 or x.shape[-1] != self.features * self.expansion)
    if downsample:
      x = Shortcut(features=self.features * self.expansion,
                   stride=self.stride, dtype=self.dtype,
                   name='downsample')(x)

    out += x
    out = nn.relu(out)
    return out


@gin.register
class AveragePool(nn.Module):
  """Average pooling layer."""

  @nn.compact
  def __call__(self, x):
    """Call function.

    Args:
      x: An array of shape [batch, ..., h, w, c], where ... is arbitrary number
      of dimensions.

    Returns:
      pooled features of shape [batch, ..., features], where ... is the same as
      the input x dimensions.
    """
    leading_dims = x.shape[:-3]
    # Flatten the leading dimensions before the spatial and feature dimensions.
    x = x.reshape(np.prod(leading_dims), -1, x.shape[-1])
    return x.mean(axis=1).reshape(leading_dims + (-1,))  # [batch, ..., c]


@gin.register
class AttentionPool(nn.Module):
  """Attention pooling layer.

  Attributes:
    num_heads: Number of heads.
    features: Number of features.
    dtype: Data type of CLIP Attention pool.
  """
  num_heads: int
  features: Optional[int] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    """Call function.

    Args:
      x: An array of shape [batch, h, w, c] or [batch, ..., h, w, c], where
        ... is arbitrary number of dimensions.

    Returns:
      pooled features of shape [batch, features] or [batch, ..., features],
        where ... is the same as the input x dimensions.
    """
    leading_dims = x.shape[:-3]
    # Flatten the leading dimensions before the spatial and feature dimensions.
    x = x.reshape(np.prod(leading_dims), -1, x.shape[-1])
    x = jnp.concatenate([x.mean(axis=1, keepdims=True), x], axis=1)

    positional_embedding = self.param(
        'positional_embedding',
        jax.nn.initializers.normal(1. / x.shape[-1]**0.5),
        (x.shape[1], x.shape[2]), self.dtype)
    attn = nn.MultiHeadDotProductAttention(
        self.num_heads,
        qkv_features=x.shape[-1],
        use_bias=True,
        out_features=self.features,
        dtype=self.dtype,
        name='attn')

    x = x + positional_embedding[jnp.newaxis].astype(self.dtype)
    x = attn(x[:, :1], x)
    return x[:, 0].reshape(leading_dims + (-1,))


class ResNetStage(nn.Module):
  """Attention pooling layer.

  Attributes:
    features: Number of features.
    num_layers: Number of bottleneck blocks.
    stride: Stride in the Bottleneck module.
    dtype: Data type of CLIP ResNetStage.
  """
  features: int
  num_layers: int
  stride: int = 1
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = Bottleneck(self.features, self.stride, dtype=self.dtype, name='0')(x)
    for i in range(1, self.num_layers):
      x = Bottleneck(self.features, dtype=self.dtype, name=str(i))(x)
    return x


class ModifiedResNet(nn.Module):
  """A ResNet class that is similar to torchvision's with changes.

  - There are now 3 "stem" convolutions as opposed to 1, with an average pool
  instead of a max pool.
  - Performs anti-aliasing strided convolutions, where an avgpool is
  prepended to convolutions with stride > 1 - The final pooling layer is a
  QKV attention instead of an average pool.

  Attributes:
    features: Number of features.
    out_features: Number of output features. If None, return resnet feature-map.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    dtype: Data type of CLIP.
  """
  features: int
  out_features: Optional[int]
  num_layers: Sequence[int]
  num_heads: Optional[int]
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    # The 3-layer stem.
    self.conv1 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        dtype=self.dtype,
        name='conv1')
    self.bn1 = nn.BatchNorm(use_running_average=True,
                            dtype=self.dtype,
                            name='bn1')
    self.conv2 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        dtype=self.dtype,
        name='conv2')
    self.bn2 = nn.BatchNorm(use_running_average=True,
                            dtype=self.dtype,
                            name='bn2')
    self.conv3 = nn.Conv(
        self.features,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        dtype=self.dtype,
        name='conv3')
    self.bn3 = nn.BatchNorm(use_running_average=True,
                            dtype=self.dtype,
                            name='bn3')

    # Residual layers.
    self.layer1 = ResNetStage(self.features, self.num_layers[0],
                              dtype=self.dtype, name='layer1')
    self.layer2 = ResNetStage(
        self.features * 2, self.num_layers[1], stride=2, dtype=self.dtype,
        name='layer2')
    self.layer3 = ResNetStage(
        self.features * 4, self.num_layers[2], stride=2, dtype=self.dtype,
        name='layer3')
    self.layer4 = ResNetStage(
        self.features * 8, self.num_layers[3], stride=2, dtype=self.dtype,
        name='layer4')
    if self.out_features is not None:
      self.attnpool = AttentionPool(
          self.num_heads, self.out_features, dtype=self.dtype, name='attnpool')

  def __call__(self, x):

    def stem(x):
      for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                       (self.conv3, self.bn3)]:
        x = nn.relu(bn(conv(x)))
      x = nn.avg_pool(x, (2, 2), (2, 2))
      return x

    feature_map = {}
    layers_to_apply = [self.layer1, self.layer2, self.layer3, self.layer4]

    x = stem(x)
    for idx, layer in enumerate(layers_to_apply):
      x = layer(x)
      feature_map[idx + 2] = x

    if self.out_features is None:
      return feature_map
    else:
      return self.attnpool(x), feature_map


class MLP(nn.Module):
  """Simple MLP for Transformer.

  Attributes:
    use_quick_gelu: Whether to use quick gelu or gelu.
  """
  use_quick_gelu: bool = True

  @nn.compact
  def __call__(self, x):
    ch = x.shape[-1]
    x = nn.Dense(4 * ch, name='c_fc')(x)
    if self.use_quick_gelu:
      x = quick_gelu(x)
    else:
      x = jax.nn.gelu(x, approximate=True)
    x = nn.Dense(ch, name='c_proj')(x)
    return x


class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Attributes:
    num_heads: Number of heads.
    use_quick_gelu: Whether to use quick gelu or gelu.
  """
  num_heads: int
  use_quick_gelu: bool = True
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self, x, attn_mask = None,
               deterministic = True):
    xn = LayerNorm(name='ln_1')(x)
    x_attn = nn.SelfAttention(
        self.num_heads, name='attn', deterministic=True)(xn, attn_mask)
    x_attn = StochasticDepth(rate=self.stochastic_depth, name='droppath_1')(
        x_attn, deterministic=deterministic)
    x = x + x_attn
    xn = LayerNorm(name='ln_2')(x)
    x_mlp = MLP(name='mlp', use_quick_gelu=self.use_quick_gelu)(xn)
    x_mlp = StochasticDepth(rate=self.stochastic_depth, name='droppath_2')(
        x_mlp, deterministic=deterministic)
    x = x + x_mlp
    return x


class Transformer(nn.Module):
  """Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    use_quick_gelu: Whether to use quick gelu or gelu.
  """
  features: int
  num_layers: int
  num_heads: int
  use_quick_gelu: bool = True

  @nn.compact
  def __call__(self,
               x,
               attn_mask = None):
    for i in range(self.num_layers):
      x = ResidualAttentionBlock(
          num_heads=self.num_heads, use_quick_gelu=self.use_quick_gelu,
          name=f'resblocks.{i}')(x, attn_mask)
    return x


class StochasticDepth(nn.Module):
  """Performs layer-dropout (also known as stochastic depth)."""

  rate: float = 0.0
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self,
               x,
               deterministic = None):
    """Applies a stochastic depth mask to the inputs.

    Args:
      x: Input tensor.
      deterministic: If false (e.g. in training) the inputs are scaled by `1 /
        (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned
        as is.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    broadcast_dims = range(1, x.ndim)
    return nn.Dropout(
        rate=self.rate, broadcast_dims=broadcast_dims)(x, deterministic)


@gin.configurable
def get_clip_vision_model(model_name):
  """Returns ResNet of ViT model."""
  cfg = _CLIP_MODEL_CFG[model_name]
  if 'resnet' in model_name:
    return functools.partial(
        ModifiedResNet,
        num_layers=cfg['vision_num_layers'],
        features=cfg['vision_features'],
        num_heads=cfg['vision_features'] // 2,
        out_features=None)
  else:
    raise ValueError(f'model_name {model_name} is not supported')


@gin.configurable
def get_clip_frozen_vision_model(model_name):
  """Returns frozen ViT model."""
  if 'vit' in model_name:
    # Frozen vision backbone performs better with window attention size v2.
    return get_clip_vision_model(model_name)
  else:
    raise ValueError(f'model_name {model_name} is not supported with frozen'
                     'backbone inference.')
