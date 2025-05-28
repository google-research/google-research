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

"""U-Net v3.

Borrowed from universal_diffusion and adapted to work both on 1d and 2d data.
original from //third_party/py/universal_diffusion/nn/unet_v3.py
"""
from typing import List, Optional, Tuple

from absl import logging
from flax import linen as nn
from flax.linen import GroupNorm
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


def default_init(scale = 1e-10):
  return nn.initializers.variance_scaling(
      scale=scale, mode="fan_avg", distribution="uniform")


class CombineResidualSkip(nn.Module):
  """Combine `residual` and `skip` connections.

    If project_skip=True, use linear layer for skip path.
  """

  @nn.compact
  def __call__(self,
               *,
               residual,
               skip,
               project_skip = False):
    if project_skip:
      skip = nn.Dense(residual.shape[-1], kernel_init=default_init(1.0))(skip)

    return (skip + residual) / np.sqrt(2.0)


class CombineConditionalEmb(nn.Module):
  """Combine `cemb` to `x` with FiLM layer.

  Conditional information is projected to two vectors of length c where c is
  the number of channels of x, then x is scaled channelwise by first vector
  and offset channelwise by the second vector.
  This method is now standard practice for conditioning with diffusion models,
  see e.g. https://arxiv.org/abs/2105.05233, and for the
  more general FiLM technique see https://arxiv.org/abs/1709.07871.
  """

  @nn.compact
  def __call__(self, x, cemb):
    cemb = nn.swish(cemb)

    cemb = nn.Dense(
        features=x.shape[-1] * 2,
        kernel_init=default_init(1.0),
        name="cemb_proj")(
            cemb)
    for _ in range(len(x.shape) - 2):
      cemb = cemb[:, None]  # unsqueeze to allow broadcasting
    scale, bias = jnp.split(cemb, 2, axis=-1)

    return x * (scale + 1.0) + bias


class MultiHeadAttentionBlock(nn.Module):
  """MultiHead Attention Block.

  Attributes:
    num_heads: int. Number of attention heads.
    per_head_channels: int. Number of channels per head in the attention blocks.
      Only one of (num_attn_heads, per_head_channels) should be set to > 0, the
      other should be set to -1 and will not be used.
    skip_rescale: bool. If true, rescale the output by 1/sqrt(2).
  """

  norm_type: str = "group"
  num_heads: int = 1
  per_head_channels: int = -1

  @nn.compact
  def __call__(self,
               input_q,
               input_kv = None,
               kv_mask = None,
               kv_norm_type = "group"):
    """Multi-Head Attention Block (Adapted to 1D sequences).

    If input_kv is not specified, we do self-attention. Note that in this case,
    we assume that input_q is of shape [B, L, C].

    If input_kv is specified, we do cross-attention. Currently, we assume
    input_kv to be of shape [B, L, C] e.g. for cross attention on text
    embedding.

    Args:
      input_q: Input query of shape `[B, L, C]`.
      input_kv: Input key-value pair of shape  `[B, L, C]`.
      kv_mask: Mask for input_kv `[B, L]`, 1s for valid, 0 for invalid position.
      kv_norm_type: Type of normalization to use for key-value.

    Returns:
      output of shape `[B, L, C]`.
    """
    assert len(
        input_q.shape) == 3, f"MHA requires shape [B,L,C], got {input_q.shape}"
    B, L, C = input_q.shape  # pylint: disable=invalid-name,unused-variable

    # Decide over the number of attention heads to use.
    if self.per_head_channels == -1:
      assert C % self.num_heads == 0, f"Cannot divide {C} channels into {self.num_heads} heads"
      per_head_channels = C // self.num_heads
      num_heads = self.num_heads
    else:
      assert self.num_heads == -1, ("both per_head_channels and num_heads ",
                                    "should not be set at the same time.")
      assert C % self.per_head_channels == 0
      num_heads = C // self.per_head_channels
      per_head_channels = self.per_head_channels
    logging.info("%s: q=%r, num_heads=%r, per_head_channels=%r", self.name,
                 input_q.shape, num_heads, per_head_channels)

    if input_kv is None:
      # If input_kv is not specified, do self-attention.
      # Normalize the input.
      q = GroupNorm(min(C // 4, 32), name="norm")(input_q)
      # Projection before query-key-value split.
      qkv = nn.Conv(
          features=C * 3,
          kernel_size=(1,),
          strides=(1,),
          kernel_init=default_init(1.0),
          name="qkv_conv")(
              q)
      qkv = jnp.reshape(qkv, [B, L, 3 * C])
      qkv = jnp.reshape(qkv, [B, L, num_heads, per_head_channels * 3])
      q, k, v = jnp.split(qkv, indices_or_sections=3, axis=-1)
    else:
      # If kv is specified, do cross-attention.
      assert len(input_kv.shape) == 3
      # Normalize the inputs.
      q = GroupNorm(min(C // 4, 32), name="qnorm")(input_q)
      if kv_norm_type == "group":
        kv = GroupNorm(min(C // 4, 32), name="kvnorm")(input_kv)
      elif kv_norm_type == "layer":
        kv = nn.normalization.LayerNorm(name="kvnorm")(input_kv)
      else:
        raise ValueError("Unknown normalization type: %s" % kv_norm_type)

      # Projection the query.
      q = nn.Conv(
          features=C,
          kernel_size=(1,),
          strides=(1,),
          kernel_init=default_init(1.0),
          name="q_conv")(
              q)
      # Projection the kev-value pair.
      kv = nn.Dense(
          features=C * 2, kernel_init=default_init(1.0), name="kv_dense")(
              kv)
      q = jnp.reshape(q, [B, L, C])
      q = jnp.reshape(q, [B, L, num_heads, per_head_channels])

      kv = jnp.reshape(kv, [B, -1, num_heads, per_head_channels * 2])
      k, v = jnp.split(kv, indices_or_sections=2, axis=-1)

    scale = 1. / jnp.sqrt(per_head_channels)
    logits = jnp.einsum("btnh,bfnh->bnft", k, q * scale)

    # Apply attention mask (Only used for cross-attention right now).
    if kv_mask is not None:
      assert kv is not None
      mask = jnp.expand_dims(kv_mask, [1, 2])
      logits = ((1. - mask) * -1e9) + logits

    weights = jax.nn.softmax(logits, axis=-1)
    h = jnp.einsum("bnft,btnh->bfnh", weights, v)

    h = jnp.reshape(h, [B, L, C])
    h = jnp.reshape(h, input_q.shape)
    h = nn.Conv(
        features=C,
        kernel_size=(1,),
        strides=(1,),
        kernel_init=default_init(),
        name="proj_layer")(
            h)

    # Zero out the final activations if the masks are all zero.
    if kv_mask is not None:
      is_not_empty = jnp.any(kv_mask == 1, axis=-1, keepdims=True)
      h = h * is_not_empty.astype(h.dtype)[Ellipsis, None, None]

    return CombineResidualSkip()(residual=h, skip=input_q, project_skip=False)


class FFN(nn.Module):
  """1 hidden layer Feed-Forward Network with Residual connection."""

  hidden_dim: int
  channels: int

  @nn.compact
  def __call__(self, x):
    skip = x
    nd = len(
        x.shape) - 2  # number of spatial dimensions, 1 for seq, 2 for image
    x = nn.Conv(
        features=self.hidden_dim,
        kernel_size=nd * (1,),
        kernel_init=default_init(1.0))(
            x)
    x = nn.swish(x)
    x = nn.Conv(
        features=self.channels,
        kernel_size=nd * (1,),
        kernel_init=default_init(1.0))(
            x)

    return CombineResidualSkip()(residual=x, skip=skip, project_skip=False)


class ConvBlock(nn.Module):
  """Basic conv block.

  There are two paths, the main conv path `h` and the shorcut path `s`.

  main conv path:
  --> GroupNorm
  --> Swish
  --> Conv(kernel_size=kernel_size_0)
  --> GroupNorm
  --> Swish
  --> Conv(kernel_size=kernel_size_1)
  shortcut path:
  --> Linear

  Attributes:
    channels: The number of output channels.
    kernel_size_0: 1st kernel width to use.
    kernel_size_1: 2nd kernel width to use.
    dropout: The dropout rate.
    deterministic: bool = False.
  """

  channels: int
  kernel_size_0: Tuple[int,]
  kernel_size_1: Tuple[int,]
  dropout: float = 0.0
  deterministic: bool = False

  @nn.compact
  def __call__(self, x):
    # `h` for the main conv path, `s` for shorcut path.
    h = x
    s = x
    del x
    nd = len(h.shape) - 2  # number of spatial dimensions 1 for seq, 2 for img

    # Main conv path.
    h = GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = nn.swish(h)
    h = nn.Conv(
        self.channels,
        kernel_size=nd * self.kernel_size_0,
        kernel_init=default_init(1.0),
        name="conv_0")(
            h)

    h = GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = nn.swish(h)
    h = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(h)
    h = nn.Conv(
        self.channels,
        kernel_size=nd * self.kernel_size_1,
        kernel_init=default_init(1.0),
        name="conv_1")(
            h)

    return CombineResidualSkip()(residual=h, skip=s, project_skip=True)


class DStack(nn.Module):
  """Downsampling Stack.

  Repeated convolutional blocks with occaisonal strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, cemb, *,
               train):
    nd = len(x.shape) - 2  # number of spatial dimensions, 1 for seq, 2 for img
    # Note, TPUs benefits from the last dimension to be a multiple of 128,
    # so this is set to 128.
    h_list = [
        nn.Conv(
            128,
            kernel_size=nd * (3,),
            kernel_init=default_init(1.0),
            name="conv_in")(x)
    ]
    h = h_list[-1]

    for d_idx, dblock in enumerate(self.config.blocks):
      dblock = ml_collections.ConfigDict(dblock)

      logging.info("Building dblock_%d starting with shape %s.", d_idx, h.shape)

      # If strides is provided, the block starts with downsampling
      # the feature map. Otherwise proceeds with the original resolution.
      if dblock.get("strides"):
        h = nn.Conv(
            features=dblock.get("hidden_size", dblock.d_channels),
            kernel_size=nd * dblock.strides,
            strides=nd * dblock.strides,
            kernel_init=default_init(1.0),
            name=f"dblock.{dblock.name}.conv_downsample")(
                h)

      for s_idx in range(dblock.num_subblocks):
        # Combine conditioning embedding with activation h.
        h = CombineConditionalEmb(name=f"dblock.{dblock.name}.{s_idx}.cemb")(
            h, cemb)

        # If kernel_size_0 is provided, then we use the ResNet block.
        if dblock.get("kernel_size_0"):
          h = ConvBlock(
              channels=dblock.d_channels,
              kernel_size_0=dblock.kernel_size_0,
              kernel_size_1=dblock.kernel_size_1,
              dropout=self.config.dropout,
              deterministic=not train,
              name=f"dblock.{dblock.name}.{s_idx}.cblock")(
                  h)

        if dblock.get("self_attention"):
          kv, attn_mask = None, None
          norm_type = "group"
          B, *hw, C = h.shape  # pylint: disable=invalid-name
          h = MultiHeadAttentionBlock(
              num_heads=dblock.num_attn_heads,
              per_head_channels=dblock.get("per_head_channels", -1),
              name=f"dblock.{dblock.name}.{s_idx}.attention")(
                  h.reshape(B, -1, C), kv, attn_mask, kv_norm_type=norm_type)

          h = FFN(
              dblock.get("hidden_size", dblock.d_channels * 2),
              dblock.d_channels,
              name=f"dblock.{dblock.name}.{s_idx}.ffn")(h).reshape(B, *hw, C)

        h_list.append(h)

      logging.info("dblock_%d built with shape %s.", d_idx, h.shape)
    return h_list


def depth_to_space_2d(x, block_size):
  """Rearrange data from the depth (e.g., channels) to spatial.

  Args:
    x: The array to rearrange, in NHWC format.
    block_size: The spatial expansion, in [H, W] format.

  Returns:
    The rearranged data.
  """
  assert x.ndim == 4, f"Array should be NHWC for depth2space, got {x.ndim} vs 4"

  in_channels = jnp.shape(x)[-1]
  assert in_channels % np.prod(block_size) == 0, "Invalid reshape sizes"
  out_channels = in_channels // np.prod(block_size)

  x = jnp.reshape(x, x.shape[:-1] + tuple(block_size) + (out_channels,))
  x = jnp.swapaxes(x, -4, -3)
  x = jax.lax.collapse(x, -5, -3)
  x = jax.lax.collapse(x, -3, -1)

  return x


class UStack(nn.Module):
  """Upsampling Stack.

  Takes in features at intermediate resolutions from the downsampling stack
  as well as final output, and applies upsampling with convolutional blocks
  and combines together with skip connections in typical UNet style.
  Optionally can use self attention at low spatial resolutions.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, cemb,
               dstack, *, train):
    h = x
    nd = len(h.shape) - 2  # (number of array dims, 1 for seq, 2 for img)
    for u_idx, ublock in enumerate(reversed(self.config.blocks)):
      ublock = ml_collections.ConfigDict(ublock)

      logging.info(
          "Building ublock_%d starting with shape %s and dstack[-1] "
          "shape %s.", u_idx, h.shape, dstack[-1].shape)
      for s_idx in range(ublock.num_subblocks):
        # Get the skip connection activations from the dstack, and combine with
        # the current activations.
        skip = dstack.pop()
        assert skip.shape[1:3] == h.shape[1:3], (
            "Skip connection has shape: %s, while we are shape %s.\n"
            "Rest of DStack: %s" %
            (skip.shape, h.shape, [x.shape for x in dstack]))
        h = CombineResidualSkip()(
            residual=h, skip=skip, project_skip=skip.shape[-1] != h.shape[-1])
        del skip

        # Combine conditioning embedding with activation h.
        h = CombineConditionalEmb(name=f"ublock.{ublock.name}.{s_idx}.cemb")(
            h, cemb)

        if ublock.get("kernel_size_0"):
          h = ConvBlock(
              channels=ublock.u_channels,
              kernel_size_0=ublock.kernel_size_0,
              kernel_size_1=ublock.kernel_size_1,
              dropout=self.config.dropout,
              deterministic=not train,
              name=f"ublock.{ublock.name}.{s_idx}.cblock")(
                  h)

        if ublock.get("self_attention"):
          kv, attn_mask = None, None
          norm_type = "group"
          B, *hw, c = h.shape  # pylint: disable=invalid-name
          h = MultiHeadAttentionBlock(
              num_heads=ublock.num_attn_heads,
              per_head_channels=ublock.get("per_head_channels", -1),
              name=f"ublock.{ublock.name}.{s_idx}.attention")(
                  h.reshape(B, -1, c), kv, attn_mask, kv_norm_type=norm_type)

          h = FFN(
              ublock.get("hidden_size", ublock.u_channels * 2),
              ublock.u_channels,
              name=f"ublock.{ublock.name}.{s_idx}.ffn")(h).reshape(B, *hw, c)

      if ublock.get("strides"):
        h = nn.Conv(
            features=ublock.get("hidden_size", ublock.u_channels),
            kernel_size=nd * (3,),
            kernel_init=default_init(1.0),
            name=f"ublock.{ublock.name}.conv_upsample")(
                h)
        if nd == 1:  # if input is a sequence
          b, l, c = h.shape
          h = h.reshape(b, l * ublock.strides[0], c // ublock.strides[0])
        elif nd == 2:  # if input is an image
          h = depth_to_space_2d(h, 2 * ublock.strides)
        else:
          raise NotImplementedError

      logging.info("ublock_%d built with shape %s.", u_idx, h.shape)

    # Skip connect, from the DStack immediately after its 3x3 conv (before any
    # blocks).
    skip = dstack.pop()
    assert skip.shape[1:2] == h.shape[1:2], (
        "Skip connection has shape: %s, while we are shape %s.\n"
        "Rest of DStack: %s" % (skip.shape, h.shape, [x.shape for x in dstack]))
    h = CombineResidualSkip()(
        residual=h, skip=skip, project_skip=skip.shape[-1] != h.shape[-1])
    del skip

    assert not dstack, ("Dstack supposed to be empty after full UStack"
                        " but has %d remaining elements." % len(dstack))
    # Final mixing.
    h = nn.Conv(
        128,
        kernel_size=nd * (3,),
        kernel_init=default_init(1.0),
        name="conv_final")(
            h)

    return h


class UNet(nn.Module):
  """UNet architecture designed for diffusion on 1d or 2d data.

  Uses improved Unet from https://imagen.research.google/paper.pdf
  Original Unet first introduced in https://arxiv.org/pdf/2006.11239.pdf
  Optionally feeds in additional data to condition on.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self,
               x,
               t,
               train = True,
               cond = None):
    """UNet architecture designed for diffusion on 1d or 2d data.

    Args:
      x: of shape (bs,h,w,c) for image or (bs,n,c) for seq
      t: diffusion time in (0,1] with shape (bs,)
      train: whether to use train mode or not (for dropout)
      cond: input conditioning of shape (bs,d) or None

    Returns:
      Unet output of same shape as x
    """
    x = jnp.array(x)
    if cond is not None:
      cond = cond.reshape(x.shape[0], -1)
    ch = self.config.embed_dim
    assert t is not None
    B, *hw, _ = x.shape  # pylint: disable=invalid-name
    nd = len(hw)  # pylint: disable=invalid-name
    assert t.shape == (B,)
    assert x.dtype in (jnp.bfloat16, jnp.float32, jnp.float64)
    # Timestep embedding.
    emb = embed_time(t, ch * 4, name="dense")
    assert emb.shape == (B, ch * 4)
    if cond is not None:
      emb = jnp.concatenate(
          [emb, embed_features(cond / 4, ch * 4, name="condemb")], axis=-1)
      # padded_cond = 0*x
      # #padded_cond[:,:cond.shape[1]] = cond
      # padded_cond = padded_cond.at[:,:cond.shape[1]].set(cond)
      # x = jnp.concatenate([x,padded_cond], -1)
    dstack = DStack(self.config)(x, emb, train=train)
    h = dstack[-1]
    h = UStack(self.config)(h, emb, dstack, train=train)

    h = nn.Conv(
        features=self.config.output_ch,
        kernel_size=nd * (3,),
        strides=nd * (1,),
        kernel_init=default_init(),
        name="conv_out")(
            h)
    return h


def embed_time(
    t,
    embed_dim = 64,
    hidden_dim = 256,
    fmax = 2e4,  # highest sinusoid frequency
    name = "dense"):
  """Embed the difussion time t (bs,) using the sinusoidal embedding."""
  frequencies = jnp.exp(jnp.linspace(0, jnp.log(fmax), hidden_dim // 2))
  scaled_time = jnp.pi * frequencies[None, :] * t[:, None]
  c = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
  c = nn.Dense(features=hidden_dim, name=f"{name}0")(c)
  c = nn.swish(c)
  c = nn.Dense(features=embed_dim, name=f"{name}1")(c)
  return c


def embed_features(
    z,
    embed_dim = 64,
    hidden_dim = 256,
    fmax = 2e4,  # highest sinusoid frequency
    name = "dense"):
  """Embed other conditioning features (bs,c) using the sinusoidal embedding."""
  frequencies = jnp.exp(jnp.linspace(0, jnp.log(fmax), hidden_dim // 2))
  scaled_z = (frequencies[None, :, None] * z[:, None, :]).reshape(
      z.shape[0], -1)
  c = jnp.concatenate([jnp.sin(jnp.pi * scaled_z),
                       jnp.cos(jnp.pi * scaled_z)],
                      axis=-1)
  c = nn.Dense(features=hidden_dim, name=f"{name}0")(c)
  c = nn.swish(c)
  c = nn.Dense(features=embed_dim, name=f"{name}1")(c)
  return c


def config(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def unet_64_config(out_dim,
                   base_channels = 24,
                   attention = False,
                   **kwargs):
  """Unet config for 64x64 input resolution, approx 600m params."""
  c = base_channels
  cfg = config(
      name="unet_v3",
      dropout=0.0,
      embed_dim=c,
      blocks=[
          config(
              d_channels=c,
              u_channels=c,
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=4,
              name="64x",
          ),
          config(
              d_channels=2 * c,
              u_channels=2 * c,
              strides=(2,),
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=8,
              name="32x",
          ),
          config(
              d_channels=4 * c,
              u_channels=4 * c,
              strides=(2,),
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=8,
              self_attention=attention,
              num_attn_heads=4,
              name="16x",
          ),
      ],
      mblock=None,
      output_ch=out_dim,
      num_classes=1,
  )

  cfg.update(kwargs)
  return cfg


def unet_128_config(out_dim,
                    base_channels = 24,
                    **kwargs):
  """Unet config for 128x128 input resolution."""
  c = base_channels
  cfg = config(
      name="unet_v3",
      dropout=0.0,
      embed_dim=c,
      blocks=[
          config(
              d_channels=c,
              u_channels=c,
              strides=(2,),
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=4,
              name="64x",
          ),
          config(
              d_channels=2 * c,
              u_channels=2 * c,
              strides=(2,),
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=8,
              name="32x",
          ),
          config(
              d_channels=4 * c,
              u_channels=4 * c,
              strides=(2,),
              kernel_size_0=(3,),
              kernel_size_1=(3,),
              num_subblocks=8,
              self_attention=True,
              num_attn_heads=6,
              name="16x",
          ),
      ],
      mblock=None,
      output_ch=out_dim,
      num_classes=1,
  )

  cfg.update(kwargs)
  return cfg
