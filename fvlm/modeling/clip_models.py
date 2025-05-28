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

"""OpenAI's CLIP models in Flax.

The implementation is based on an initial port of code in
https://github.com/openai/CLIP to JAX by Ben Poole.
"""

import functools
from typing import Any, Mapping, Optional, Sequence, Union

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


class WindowTransformer(Transformer):
  """Window Transformer module without the patchifying stem layer.

  Attributes:
    window_attention_size: Sequence of window attention grid size of all
      transformer layers.
    feat_h: int, height of the 2D features.
    feat_w: int, width of the 2D features.
    droppath: Drop path ratio. Defaults to 0.
  """
  window_attention_size: Optional[Union[int, Sequence[Optional[int]]]] = None
  feat_h: Optional[int] = None
  feat_w: Optional[int] = None
  droppath: float = 0.0

  def window_split(self, seq,
                   window_attention_size):
    """Window-partitions a sequence then merge along the batch dimension.

    Args:
      seq: jnp.ndarray, [B, N, D].
      window_attention_size: int, number of windows along height and width.

    Returns:
      windowed_feats: jnp.ndarray, [K*K*B, N/K**2, D].
    """
    assert self.feat_h % window_attention_size == 0
    assert self.feat_w % window_attention_size == 0

    batch_size, feat_c = seq.shape[0], seq.shape[-1]
    feat = jnp.reshape(seq, [-1, self.feat_h, self.feat_w, feat_c])
    h_new = self.feat_h // window_attention_size
    w_new = self.feat_w // window_attention_size

    tmp = [
        jnp.split(f, window_attention_size, axis=2)
        for f in jnp.split(feat, window_attention_size, axis=1)
    ]
    window_feats = []
    for t in tmp:
      window_feats += t

    # Concate window splits at the batch dimension.
    window_feats = jnp.concatenate([
        jnp.reshape(
            x, [batch_size, h_new * w_new, feat_c]) for x in window_feats
        ], axis=0)
    return window_feats

  def window_merge(self, seqs, batch_size_model,
                   window_attention_size):
    """Merges sequences along spatial dimensions then reshape to a sequence.

    Args:
      seqs: jnp.ndarray, [K*K*B, N/K**2, D].
      batch_size_model: int, the original batch size of the model.
      window_attention_size: int, number of windows along height or width.

    Returns:
      feats: jnp.ndarray, [B, N, D].
    """
    batch_size, feat_c = seqs.shape[0], seqs.shape[-1]

    # Return if no window partition.
    if batch_size == batch_size_model:
      return seqs

    n_windows = batch_size // batch_size_model
    h_new = self.feat_h // window_attention_size
    w_new = self.feat_w // window_attention_size
    seqs = jnp.split(seqs, n_windows, axis=0)
    window_feats = [
        jnp.reshape(seq, [-1, h_new, w_new, feat_c]) for seq in seqs]

    column_feats = []
    for i in range(window_attention_size):
      column_feats.append(
          jnp.concatenate(
              window_feats[i * window_attention_size:(i + 1) *
                           window_attention_size], axis=2))
    merged_feats = jnp.concatenate(column_feats, axis=1)
    return jnp.reshape(
        merged_feats, [batch_size_model, self.feat_h * self.feat_w, feat_c])

  @nn.compact
  def __call__(self,
               x,
               attn_mask = None,
               deterministic = True,
               shift_size = 0):
    window_attention_size = self.window_attention_size
    if window_attention_size is None:
      x_out = super().__call__(x, attn_mask=attn_mask)
      return x_out

    # Prepend and append 'None' to indicate the full attention window at
    # input and outputs.
    assert len(window_attention_size) == self.num_layers
    window_attention_size = [None] + list(window_attention_size) + [None]

    x_out = x
    for layer_idx in range(self.num_layers):
      x_in = x_out
      window_size = window_attention_size[layer_idx + 1]

      # Only reshape input/output when next layer has different window size.
      reshape_input = window_size and (
          window_attention_size[layer_idx] != window_size)
      reshape_output = window_size and (
          window_attention_size[layer_idx + 2] != window_size)

      if reshape_input:
        inputs_shape = x_in.shape
        assert self.feat_h * self.feat_w == inputs_shape[1]
        x_in = self.window_split(
            seq=x_in,
            window_attention_size=window_size)

      attn_mask = None
      x_out = ResidualAttentionBlock(
          num_heads=self.num_heads, use_quick_gelu=self.use_quick_gelu,
          stochastic_depth=(
              layer_idx / max(self.num_layers - 1, 1)) * self.droppath,
          name=f'resblocks.{layer_idx}')(x_in, attn_mask,
                                         deterministic=deterministic)

      if reshape_output:
        x_out = self.window_merge(
            seqs=x_out,
            batch_size_model=inputs_shape[0],
            window_attention_size=window_size)
    return x_out


class VisionTransformer(nn.Module):
  """Vision Transformer backbone.

  Attributes:
    patch_size: The size of the patches to embed.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    window_attention_size: Window attention size.
    droppath: Drop path ratio.
    pretrained_pos_emb_size: Height/width of 2D positional embeddings of the
      pretrained ViT.
    window_attention_shift_size: Window attention shift size.
  """
  patch_size: int
  features: int
  num_layers: int
  num_heads: int
  window_attention_size: Sequence[Optional[int]] = (
      _WINDOW_ATTENTION_SIZE_VIT_BASE)
  droppath: float = 0.0
  pretrained_pos_emb_size: int = 14  # Not used if 0.
  window_attention_shift_size: int = 0

  def interpolate_embedding_2d(self, emb,
                               source_emb_shape,
                               target_emb_shape):
    """Interpolates a 2D positional embedding to a new shape.

    Args:
      emb: JTensor, (1, H1xW1, D), flattened 2D positional embedding.
      source_emb_shape: Tuple, (H1, W1), height and width of source embedding.
      target_emb_shape: Tuple, (H2, W2), height and width of target embedding.

    Returns:
      Flattened, interpolated embedding of shape (1, H2xW2, D)
    """
    if len(emb.shape) > 3 or emb.shape[0] != 1:
      raise ValueError('The shape of the embedding should be (1, H * W, D)')

    if emb.shape[1] != source_emb_shape[0] * source_emb_shape[1]:
      raise ValueError('The shape of the embedding does NOT match input specs.')

    emb_dims = emb.shape[2]
    emb = jnp.reshape(emb, (source_emb_shape[0], source_emb_shape[1], emb_dims))

    target_emb = jax.image.resize(
        emb, (target_emb_shape[0], target_emb_shape[1], emb_dims),
        method='bilinear')
    target_emb = jnp.reshape(
        target_emb, (1, target_emb_shape[0] * target_emb_shape[1], emb_dims))

    return target_emb

  def _forward(self,
               x,
               conv,
               positional_embedding,
               window_transformer,
               shift_size = 0):
    """Forward function."""
    _, height, width, _ = x.shape
    x = conv(x)
    # Shift
    x = jnp.roll(x, shift=(-shift_size, shift_size), axis=(1, 2))
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    if self.pretrained_pos_emb_size != 0:
      num_patches = self.pretrained_pos_emb_size**2
    else:
      num_patches = x.shape[1]
    row_count = height // self.patch_size
    col_count = width // self.patch_size
    if x.shape[1] != num_patches:
      positional_embedding = self.interpolate_embedding_2d(
          positional_embedding,
          (self.pretrained_pos_emb_size, self.pretrained_pos_emb_size),
          (row_count, col_count),)
    x = x + positional_embedding
    x = window_transformer(x, shift_size=shift_size)
    x = jnp.reshape(x, [x.shape[0], row_count, col_count, x.shape[-1]])
    # Unshift
    x = jnp.roll(x, shift=(shift_size, -shift_size), axis=(1, 2))
    return x

  @nn.compact
  def __call__(self,
               x,
               attn_mask = None,
               train = False):
    _, height, width, _ = x.shape
    conv = nn.Conv(
        self.features,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        use_bias=True,
        name='conv1')
    scale = 1.0 / jnp.sqrt(self.features)
    if self.pretrained_pos_emb_size != 0:
      num_patches = self.pretrained_pos_emb_size**2
    else:
      num_patches = x.shape[1]
    row_count = height // self.patch_size
    col_count = width // self.patch_size
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (num_patches, self.features))[None]
    window_transformer = WindowTransformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        use_quick_gelu=False,
        window_attention_size=self.window_attention_size,
        feat_h=row_count,
        feat_w=col_count,
        droppath=self.droppath,
        name='transformer')
    ln_post = LayerNorm(name='ln_post')

    x_out = self._forward(
        x, conv, positional_embedding, window_transformer)
    if self.window_attention_shift_size != 0:
      x_out_shift = self._forward(
          x, conv, positional_embedding, window_transformer,
          shift_size=self.window_attention_shift_size)
      # Average the shifted and unshifted outputs.
      x = (x_out + x_out_shift) * 0.5
    else:
      x = x_out

    x = jnp.reshape(x, [x.shape[0], row_count * col_count, x.shape[-1]])
    x = ln_post(x)
    x = jnp.reshape(x, [x.shape[0], row_count, col_count, x.shape[-1]])
    feature_map = {4: x}
    return feature_map


@gin.configurable
def get_clip_vision_model(model_name,
                          use_window_attention_size_frozen = False,
                          window_attention_shift_size = 0):
  """Returns ResNet of ViT model.

  Args:
    model_name: Model name.
    use_window_attention_size_frozen: If set True, it uses window attention in
      all layers but the last global attention layer, instead of having four
      global attention layers evenly spaced throughout the vision transformer.
    window_attention_shift_size: The roll shift size of window attention.
      Defaults to 0 which is non-shifted window attention baseline. For shifted
      window learning, it is set to 8 which is half the window attention cell
      size (e.g., 16 with image size 1024, patch size 16).
  """
  cfg = _CLIP_MODEL_CFG[model_name]
  if 'resnet' in model_name:
    return functools.partial(
        ModifiedResNet,
        num_layers=cfg['vision_num_layers'],
        features=cfg['vision_features'],
        num_heads=cfg['vision_features'] // 2,
        out_features=None)
  elif 'vit' in model_name:
    return functools.partial(
        VisionTransformer,
        patch_size=cfg['vision_patch_size'],
        num_layers=cfg['vision_num_layers'],
        features=cfg['vision_features'],
        num_heads=cfg['vision_num_heads'],
        window_attention_size=(
            cfg['window_attention_size']
            if not use_window_attention_size_frozen
            else cfg['window_attention_size_frozen']),
        window_attention_shift_size=window_attention_shift_size)
  else:
    raise ValueError(f'model_name {model_name} is not supported')


@gin.configurable
def get_clip_frozen_vision_model(model_name):
  """Returns frozen ViT model."""
  if 'vit' in model_name:
    # Frozen vision backbone performs better with window attention size v2.
    return get_clip_vision_model(model_name,
                                 use_window_attention_size_frozen=True)
  else:
    raise ValueError(f'model_name {model_name} is not supported with frozen'
                     'backbone inference.')
