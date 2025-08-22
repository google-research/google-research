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

"""A 3D U-Net Architecture from Denoising Diffusion Probabilistic Models."""

from typing import Any, Dict, Optional, Tuple

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


def add_text_features(*args, **kwargs):
  raise NotImplementedError


def add_score_features(*args, **kwargs):
  raise NotImplementedError


def process_conditional_signal(x,
                               cond,
                               embed_dim = 1024,
                               num_classes = 1000,
                               max_time = 2.0e4):
  """Handle conditional variable, project to embedding.

  Args:
    x: If diffusion model, the "x_t". If regression model, None.
    cond: A dictionary of conditionining signals.
    embed_dim: The returned conditonining embedding dim.
    num_classes: The number of classes for class conditional models.
    max_time: The maximum number of timesteps in get_timing_signal_1d.

  Returns:
    The new `x` (potentially different dim than the input `x`), and the
    conditioning signal `emb`.
  """
  emb = None

  if "class_id" in cond:
    cond_label = cond["class_id"]
    assert cond_label.dtype == jnp.int32
    c = jax.nn.one_hot(cond_label, num_classes=num_classes,
                       dtype=jnp.float32)
    c = nn.Dense(features=embed_dim, name="class_condition")(c)
    emb = c if emb is None else emb + c

    del cond_label

  if "text_embs_pooled" in cond:
    cond_embed = cond["text_embs_pooled"]
    assert cond_embed.dtype == jnp.float32
    assert len(cond_embed.shape) == 2
    c = nn.Dense(features=embed_dim, name="embed_condition")(cond_embed)
    emb = c if emb is None else emb + c

    del cond_embed

  if "image" in cond:
    cond_image = cond["image"]
    assert cond_image.dtype == jnp.float32
    if x is None:
      x = cond_image
    else:
      assert cond_image.shape[0] == x.shape[0], f"{cond_image.shape}, {x.shape}"
      assert cond_image.shape[:-1] == x.shape[:-1], (
          f"{cond_image.shape}, {x.shape}")
      x = jnp.concatenate([cond_image, x], -1)

    del cond_image

  for key in cond:
    if key.endswith("noise_cond_t"):
      c = get_timestep_embedding_with_projection(
          cond[key], embed_dim, max_time,
          "%sdense_cond" % key[:-len("noise_cond_t")])
      emb = c if emb is None else emb + c

  if "clip_score" in cond:
    c = get_timestep_embedding_with_projection(
        cond["clip_score"], embed_dim, max_time, name="clip_score_cond")
    emb = c if emb is None else emb + c

  return x, emb


def default_init(scale = 1e-10):
  return nn.initializers.variance_scaling(
      scale=scale,
      mode="fan_avg",
      distribution="uniform")


def naive_upsample_3d(x, factor = 2):
  """Upsample a 2D array by duplication."""
  _, h, w, d, c = x.shape
  x = jnp.reshape(x, [-1, h, 1, w, 1, d, 1, c])
  x = jnp.tile(x, [1, 1, factor, 1, factor, 1, factor, 1])
  return jnp.reshape(x, [-1, h * factor, w * factor, d * factor, c])


def naive_downsample_3d(x, factor = 2):
  """Downsample a 2D array by average-pooling."""
  _, h, w, d, c = x.shape
  x = jnp.reshape(
      x, [-1, h // factor, factor, w // factor, factor, d // factor, factor, c])
  return jnp.mean(x, axis=[2, 4, 6])


def nearest_neighbor_upsample_3d(x):
  B, H, W, D, C = x.shape  # pylint: disable=invalid-name
  x = x.reshape(B, H, 1, W, 1, D, 1, C)
  x = jnp.broadcast_to(x, (B, H, 2, W, 2, D, 2, C))
  return x.reshape(B, H * 2, W * 2, D * 2, C)


def normalization_layer(norm_type, **kwargs):
  """Normalization Layer."""
  if norm_type == "none":
    return jnp.array
  elif norm_type == "group":
    return nn.normalization.GroupNorm(
        num_groups=kwargs.get("num_groups", 32),
        name=kwargs.get("name", "group_norm"))
  elif norm_type == "layer":
    return nn.normalization.LayerNorm(
        name=kwargs.get("name", "layer_norm"))
  else:
    raise ValueError("Normalization type %s not known." % norm_type)


def get_timing_signal_1d(position,
                         num_channels,
                         min_timescale = 1.0,
                         max_timescale = 2.0e4):
  """Returns the positional encoding (same as Tensor2Tensor).

  Args:
    position: An array of shape [batch_size].
    num_channels: The number of output channels.
    min_timescale: The smallest time unit (should probably be 0.0).
    max_timescale: The largest time unit.

  Returns:
    a Tensor of timing signals [1, length, num_channels]
  """
  assert position.ndim == 1
  assert num_channels % 2 == 0
  num_timescales = float(num_channels // 2)
  log_timescale_increment = (
      np.log(max_timescale / min_timescale) / (num_timescales - 1.0))
  inv_timescales = min_timescale * jnp.exp(
      jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  scaled_time = (jnp.expand_dims(position, 1) *
                 jnp.expand_dims(inv_timescales, 0))
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
  signal = jnp.reshape(signal, [jnp.shape(position)[0], num_channels])
  return signal


def get_timestep_embedding_with_projection(t,
                                           embed_dim = 1024,
                                           max_time = 2.0e4,
                                           name="dense"):
  """Get timestep embedding."""
  c = get_timing_signal_1d(t, embed_dim // 4, max_timescale=max_time)
  c = nn.Dense(features=embed_dim, name=f"{name}0")(c)
  c = nn.swish(c)
  c = nn.Dense(features=embed_dim, name=f"{name}1")(c)

  return c


def depth_to_space_2d(x, block_size):
  """Rearrange data from the depth (e.g., channels) to spatial.

  Args:
    x: The array to rearrange, in NHWC format.
    block_size: The spatial expansion, in [H, W] format.

  Returns:
    The rearranged data.
  """
  assert jnp.ndim(x) == 4

  in_channels = jnp.shape(x)[-1]
  assert in_channels % np.prod(block_size) == 0
  out_channels = in_channels // np.prod(block_size)

  x = jnp.reshape(x, x.shape[:-1] + tuple(block_size) + (out_channels,))
  x = jnp.swapaxes(x, -4, -3)
  x = jax.lax.collapse(x, -5, -3)
  x = jax.lax.collapse(x, -3, -1)

  return x


class ResnetBlock(nn.Module):
  """ResNet Building Block for UNet.

  Attributes:
    out_channel: int. Number of output channels.
    dropout: float.
    conv_shortcut: bool = False. If true, use conv layer to shortcut-project the
        input. Otherwise use a dense layer.
    upsample: bool = False. Whether to upsample.
    downsample: bool = False. Whether to downsample.
    skip_rescale: bool. If true, rescale the output by 1/sqrt(2).
    use_scale_shift_norm: bool = False.
    res_block_type: str = "biggan". "biggan" or "ddpm".
    deterministic: bool = False.
  """

  out_channel: int
  dropout: float
  conv_shortcut: bool = False
  upsample: bool = False
  downsample: bool = False
  norm_type: str = "group"
  skip_rescale: bool = False
  use_scale_shift_norm: bool = False
  res_block_type: str = "biggan"
  deterministic: bool = False

  def setup(self):
    if self.res_block_type not in ["biggan", "ddpm"]:
      raise NotImplementedError(
          f"res_block_type {self.res_block_type} not implemented!")

    if self.upsample or self.downsample:
      if self.res_block_type != "biggan":
        raise ValueError(
            "upsample or downsample is only supported when res_block_type is"
            "biggan")

    if self.res_block_type == "biggan":
      self.shortcut_proj = nn.Conv(
          features=self.out_channel,
          kernel_size=(1, 1, 1),
          strides=(1, 1, 1),
          kernel_init=default_init(1.0),
          name="shortcut_proj")
    elif self.res_block_type == "ddpm":
      if self.conv_shortcut:
        self.shortcut_proj = nn.Conv(
            features=self.out_channel,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            kernel_init=default_init(1.0),
            name="shortcut_proj")
      else:
        self.shortcut_proj = nn.Dense(
            features=self.out_channel,
            kernel_init=default_init(1.0),
            name="shortcut_proj")

  @nn.compact
  def __call__(self, x, time_emb):
    h = nn.swish(normalization_layer(
        self.norm_type, num_groups=min(x.shape[-1] // 4, 32),
        name="group_norm_1")(x))

    if self.downsample:
      h = naive_downsample_3d(h, factor=2)
      x = naive_downsample_3d(x, factor=2)
    elif self.upsample:
      h = naive_upsample_3d(h, factor=2)
      x = naive_upsample_3d(x, factor=2)

    h = nn.Conv(
        features=self.out_channel,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        kernel_init=default_init(1.0),
        name="conv_1")(
            h)

    group_norm_2 = normalization_layer(
        self.norm_type, num_groups=min(self.out_channel // 4, 32),
        name="group_norm_2")

    if time_emb is not None:
      assert x.shape[0] == time_emb.shape[0]

      # Time-step embedding projection.
      time_proj_channel = (self.out_channel * 2 if self.use_scale_shift_norm
                           else self.out_channel)
      time_emb_proj = nn.Dense(
          features=time_proj_channel,
          kernel_init=default_init(1.0),
          name="time_emb_proj")(nn.swish(time_emb))[:, None, None, None, :]

      if self.use_scale_shift_norm:
        time_w, time_b = jnp.split(time_emb_proj, 2, axis=-1)
        h = group_norm_2(h) * (time_w + 1.0) + time_b
      else:
        h += time_emb_proj
        h = group_norm_2(h)
    else:
      # No time emb.
      h = group_norm_2(h)

    h = nn.swish(h)
    h = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(h)
    h = nn.Conv(
        features=self.out_channel,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        kernel_init=default_init(),
        name="conv_2")(
            h)

    # Input short-cut projection.
    if x.shape[-1] != self.out_channel:
      x = self.shortcut_proj(x)

    # logging.info("%s: x=%r, temb=%r", self.name, x.shape,
    # None if time_emb is None else time_emb.shape)
    logging.log_first_n(logging.INFO, "%s: x=%r, temb=%r", 5, self.name,
                        x.shape, None if time_emb is None else time_emb.shape)

    if self.skip_rescale:
      return (h + x) / np.sqrt(2.)
    else:
      return h + x


class DownsampleBlock(nn.Module):
  """Downsampling Block.

  Attributes:
    use_conv: bool. Whether to use convolutional downsampling.
  """

  use_conv: bool

  @nn.compact
  def __call__(self, x):
    if self.use_conv:
      x = nn.Conv(
          features=x.shape[-1], kernel_size=(3, 3, 3), strides=(2, 2, 2))(
              x)
    else:
      x = nn.avg_pool(
          x, window_shape=(2, 2, 2), strides=(2, 2, 2), padding="SAME")

    return x


class UpsampleBlock(nn.Module):
  """Upsampling Block.

  Attributes:
    use_conv: bool. Whether to use convolutional upsampling.
  """

  use_conv: bool

  @nn.compact
  def __call__(self, x):
    x = nearest_neighbor_upsample_3d(x)

    if self.use_conv:
      x = nn.Conv(
          features=x.shape[-1], kernel_size=(3, 3, 3), strides=(1, 1, 1))(
              x)

    return x


class MultiHeadAttentionBlock(nn.Module):
  """MultiHead Attention Block.

  Attributes:
    num_heads: int. Number of attention heads.
    per_head_channels: int. Number of channels per head in the attention
        blocks. Only one of (num_attn_heads, per_head_channels) should be
        set to > 0, the other should be set to -1 and will not be used.
    skip_rescale: bool. If true, rescale the output by 1/sqrt(2).
  """

  norm_type: str = "group"
  num_heads: int = 1
  per_head_channels: int = -1
  skip_rescale: bool = False

  @nn.compact
  def __call__(self, input_q,
               input_kv = None,
               kv_mask = None,
               kv_norm_type = "group"):
    """Multi-Head Attention Block (Adapted to 2D images).

    If input_kv is not specified, we do self-attention. Note that in this case,
    we assume that input_q is of shape [B, H, W, C].

    If input_kv is specified, we do cross-attention. Currently, we assume
    input_kv to be of shape [B, L, C] e.g. for cross attention on text
    embedding.

    Args:
      input_q: Input query of shape `[B, H, W, C]`.
      input_kv: Input key-value pair of shape  `[B, L, C]`.
      kv_mask: Mask for input_kv `[B, L]`, 1s for valid, 0 for invalid position.
      kv_norm_type: Type of normalization to use for key-value.

    Returns:
      output of shape `[B, H, W, C]`.
    """
    assert len(input_q.shape) == 5
    B, H, W, D, C = input_q.shape  # pylint: disable=invalid-name,unused-variable

    # Decide over the number of attention heads to use.
    if self.per_head_channels == -1:
      assert C % self.num_heads == 0
      per_head_channels = C // self.num_heads
      num_heads = self.num_heads
    else:
      assert self.num_heads == -1, ("both per_head_channels and num_heads ",
                                    "should not be set at the same time.")
      assert C % self.per_head_channels == 0
      num_heads = C // self.per_head_channels
      per_head_channels = self.per_head_channels
    # logging.info("%s: q=%r, num_heads=%r, per_head_channels=%r",
    # self.name, input_q.shape, num_heads, per_head_channels)
    logging.log_first_n(logging.INFO,
                        "%s: q=%r, num_heads=%r, per_head_channels=%r", 5,
                        self.name, input_q.shape, num_heads, per_head_channels)

    if input_kv is None:
      # If input_kv is not specified, do self-attention.
      # Normalize the input.
      q = normalization_layer(
          self.norm_type, num_groups=min(C // 4, 32), name="norm")(input_q)
      # Projection before query-key-value split.
      qkv = nn.Conv(
          features=C * 3,
          kernel_size=(1, 1, 1),
          strides=(1, 1, 1),
          kernel_init=default_init(1.0),
          name="qkv_conv")(q)
      qkv = jnp.reshape(qkv, [B, H * W * D, 3 * C])
      qkv = jnp.reshape(qkv, [B, H * W * D, num_heads, per_head_channels * 3])
      q, k, v = jnp.split(qkv, indices_or_sections=3, axis=-1)
    else:
      # If kv is specified, do cross-attention.
      assert len(input_kv.shape) == 3
      # Normalize the inputs.
      q = normalization_layer(
          self.norm_type, num_groups=min(C // 4, 32), name="qnorm")(input_q)
      kv = normalization_layer(
          kv_norm_type, num_groups=min(C // 4, 32), name="kvnorm")(input_kv)

      # Projection the query.
      q = nn.Conv(
          features=C,
          kernel_size=(1, 1, 1),
          strides=(1, 1, 1),
          kernel_init=default_init(1.0),
          name="q_conv")(q)
      # Projection the kev-value pair.
      kv = nn.Dense(
          features=C * 2,
          kernel_init=default_init(1.0),
          name="kv_dense")(kv)
      q = jnp.reshape(q, [B, H * W * D, C])
      q = jnp.reshape(q, [B, H * W * D, num_heads, per_head_channels])

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

    h = jnp.reshape(h, [B, H * W * D, C])
    h = jnp.reshape(h, input_q.shape)
    h = nn.Conv(
        features=C,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        kernel_init=default_init(),
        name="proj_layer")(h)

    # Zero out the final activations if the masks are all zero.
    if kv_mask is not None:
      is_not_empty = jnp.any(kv_mask == 1, axis=-1, keepdims=True)
      h = h * is_not_empty.astype(h.dtype)[Ellipsis, None, None]

    if self.skip_rescale:
      return (h + input_q) / jnp.sqrt(2.)
    else:
      return h + input_q


class DownsampleStack(nn.Module):
  """Downsampling Stack.

  Attributes:
    config: Config. Fields in config:
        # (deprecated) num_res_blocks: int. Number of ResNet blocks in the
        # upsample stack.
        num_res_blocks:
          Sequential[int]. Number of ResNet blocks in the upsample stack.
        attn_resolutions: Tuple[int]. At these resolutions an attention block is
            used.
        embed_dim: int. The base hidden dimension.
        channel_mult: Tuple[int]. After each ResNet block, the number of
            channels is embed_dim multiplied by the channel multiplier. The
            tuple is in the order from largest resolution to lowest resolution.
        dropout: float. Dropout for ResNet blocks.
        use_downsample_conv: bool. Whether to use convolutional downsampling.
            Only applicable if res_block_type is "ddpm".
        res_block_type: str. ResNet block type, "ddpm" or "biggan".
        skip_rescale: bool. Whether to rescale output in resnet and attention
            blocks.
        use_scale_shift_norm: bool. Whether to scale and shift the group norm in
            the resnet blocks.
        num_attn_heads: int. Number of attention heads in the attention blocks.
        per_head_channels: int. Number of channels per head in the attention
            blocks. Only one of (num_attn_heads, per_head_channels) should be
            set to > 0.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, cond, time_emb, *, train):
    h = x
    h_list = [
        nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            kernel_init=default_init(1.0),
            name="conv_in")(x)
    ]

    if self.config.get("add_pos_embs"):
      pos_embs = self.param(
          "pos_emb", nn.initializers.normal(
              stddev=h.shape[-1]**-0.5), h.shape[1:])
      h += pos_embs
      h_list[-1] += pos_embs

    num_resolutions = len(self.config.channel_mult)
    for i_layer in range(num_resolutions):
      num_res_blocks = self.config.num_res_blocks
      if not isinstance(num_res_blocks, int):
        num_res_blocks = list(num_res_blocks)[i_layer]
      # for i_block in range(self.config.num_res_blocks):
      for i_block in range(num_res_blocks):
        h = ResnetBlock(
            out_channel=(self.config.embed_dim *
                         self.config.channel_mult[i_layer]),
            dropout=self.config.dropout,
            skip_rescale=self.config.skip_rescale,
            use_scale_shift_norm=self.config.use_scale_shift_norm,
            res_block_type=self.config.res_block_type,
            deterministic=not train,
            name=f"down_{i_layer}.block_{i_block}")(h_list[-1], time_emb)

        # Self Attention.
        if h.shape[2] in self.config.attn_resolutions:
          kv, attn_mask = None, None
          norm_type_ = "group"
          h = MultiHeadAttentionBlock(
              num_heads=self.config.num_attn_heads,
              per_head_channels=self.config.per_head_channels,
              skip_rescale=self.config.skip_rescale,
              name=f"down_{i_layer}.attn_{i_block}")(
                  h, kv, attn_mask, kv_norm_type=norm_type_)

        h_list.append(h)

      # Downsample.
      if i_layer != num_resolutions - 1:
        if self.config.res_block_type == "ddpm":
          h_list.append(
              DownsampleBlock(
                  self.config.use_downsample_conv,
                  name=f"down_{i_layer}.downsample_{self.config.res_block_type}"
              )(h_list[-1]))
        else:
          h_list.append(
              ResnetBlock(
                  out_channel=h_list[-1].shape[-1],
                  dropout=self.config.dropout,
                  downsample=True,
                  skip_rescale=self.config.skip_rescale,
                  use_scale_shift_norm=self.config.use_scale_shift_norm,
                  res_block_type=self.config.res_block_type,
                  deterministic=not train,
                  name=f"down_{i_layer}.downsample_{self.config.res_block_type}"
              )(h_list[-1], time_emb))

    return h_list


class MiddleStack(nn.Module):
  """Middle Stack.

  Attributes:
    config: Config. Fields in config:
        dropout: float. Dropout for ResNet blocks.
        res_block_type: str. ResNet block type, "ddpm" or "biggan".
        skip_rescale: bool. Whether to rescale output in resnet and attention
            blocks.
        use_scale_shift_norm: bool. Whether to scale and shift the group norm in
            the resnet blocks.
        num_attn_heads: int. Number of attention heads in the attention blocks.
        per_head_channels: int. Number of channels per head in the attention
            blocks. Only one of (num_attn_heads, per_head_channels) should be
            set to > 0.
        use_middle_stack_attn: bool. Whether to use an attention block in this
            stack.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, cond, time_emb, *, train):
    h = ResnetBlock(
        out_channel=x.shape[-1],
        dropout=self.config.dropout,
        skip_rescale=self.config.skip_rescale,
        use_scale_shift_norm=self.config.use_scale_shift_norm,
        res_block_type=self.config.res_block_type,
        deterministic=not train,
        name="mid.block_1")(x, time_emb)

    if self.config.use_middle_stack_attn:
      kv, attn_mask = None, None
      norm_type_ = "group"
      h = MultiHeadAttentionBlock(
          num_heads=self.config.num_attn_heads,
          skip_rescale=self.config.skip_rescale,
          per_head_channels=self.config.per_head_channels,
          name="mid.attn_1")(h, kv, attn_mask, kv_norm_type=norm_type_)

    h = ResnetBlock(
        out_channel=x.shape[-1],
        dropout=self.config.dropout,
        skip_rescale=self.config.skip_rescale,
        use_scale_shift_norm=self.config.use_scale_shift_norm,
        res_block_type=self.config.res_block_type,
        deterministic=not train,
        name="mid.block_2")(h, time_emb)
    return h


class UpsampleStack(nn.Module):
  """Upsampling Stack.

  Attributes:
    config: Config. Fields in config:
        num_res_blocks: int. Number of ResNet blocks in the upsample stack.
        attn_resolutions: Tuple[int]. At these resolutions an attention block is
            used.
        embed_dim: int. The base hidden dimension.
        channel_mult: Tuple[int]. After each ResNet block, the number of
            channels is embed_dim multiplied by the channel multiplier. The
            tuple is in the order from largest resolution to lowest resolution.
        dropout: float. Dropout for ResNet blocks.
        use_upsample_conv: bool. Whether to use convolutional upsampling. Only
            applicable if res_block_type is "ddpm".
        res_block_type: str. ResNet block type, "ddpm" or "biggan".
        skip_rescale: bool. Whether to rescale output in resnet and attention
            blocks.
        use_scale_shift_norm: bool. Whether to scale and shift the group norm in
            the resnet blocks.
        num_attn_heads: int. Number of attention heads in the attention blocks.
        per_head_channels: int. Number of channels per head in the attention
            blocks. Only one of (num_attn_heads, per_head_channels) should be
            set to > 0.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, cond, time_emb, downsample_stack, *, train):
    h = x
    num_resolutions = len(self.config.channel_mult)
    for i_layer in reversed(range(num_resolutions)):
      num_res_blocks = self.config.num_res_blocks
      if not isinstance(num_res_blocks, int):
        num_res_blocks = list(num_res_blocks)[i_layer]
      # for i_block in range(self.config.num_res_blocks + 1):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(
            out_channel=(self.config.embed_dim *
                         self.config.channel_mult[i_layer]),
            dropout=self.config.dropout,
            skip_rescale=self.config.skip_rescale,
            use_scale_shift_norm=self.config.use_scale_shift_norm,
            res_block_type=self.config.res_block_type,
            deterministic=not train,
            name=f"up_{i_layer}.block_{i_block}")(jnp.concatenate(
                [h, downsample_stack.pop()], axis=-1), time_emb)

        # Self-Attention.
        if h.shape[2] in self.config.attn_resolutions:
          kv, attn_mask = None, None
          norm_type_ = "group"
          h = MultiHeadAttentionBlock(
              num_heads=self.config.num_attn_heads,
              skip_rescale=self.config.skip_rescale,
              per_head_channels=self.config.per_head_channels,
              name=f"up_{i_layer}.attn_{i_block}")(
                  h, kv, attn_mask, kv_norm_type=norm_type_)

        # Cross-Attention with the conditioning signal.
        if (self.config.get("text_cond_spec") and h.shape[2] in
            self.config.text_cond_spec.get("cross_attn_resolutions", [])):
          h = MultiHeadAttentionBlock(
              num_heads=self.config.num_attn_heads,
              skip_rescale=self.config.skip_rescale,
              per_head_channels=self.config.per_head_channels,
              name=f"up_{i_layer}.cross_attn_{i_block}")(
                  h, input_kv=cond["text_embs"], kv_mask=cond["text_mask"],
                  kv_norm_type="layer")

      # Upsample.
      if i_layer != 0:
        if self.config.res_block_type == "ddpm":
          h = UpsampleBlock(
              use_conv=self.config.use_upsample_conv,
              name=f"up_{i_layer}.upsample_{self.config.res_block_type}")(
                  h)
        else:
          h = ResnetBlock(
              out_channel=h.shape[-1],
              dropout=self.config.dropout,
              upsample=True,
              skip_rescale=self.config.skip_rescale,
              use_scale_shift_norm=self.config.use_scale_shift_norm,
              res_block_type=self.config.res_block_type,
              deterministic=not train,
              name=f"up_{i_layer}.upsample_{self.config.res_block_type}")(
                  h, time_emb)
    assert not downsample_stack
    return h


class UNet3D(nn.Module):
  """A UNet architecture.

  Attributes:
    config: Config. Fields in config:
        attn_resolutions: Tuple[int]. At these resolutions an attention block is
            used.
        embed_dim: int. The base hidden dimension.
        channel_mult: Tuple[int]. After each ResNet block, the number of
            channels is embed_dim multiplied by the channel multiplier. The
            tuple is in the order from largest resolution to lowest resolution.
        dropout: float. Dropout for ResNet blocks.
        norm_type: str. Type of normalization layer to use {`group`, `none`}.
        use_downsample_conv: bool. Whether to use convolutional downsampling.
            Only applicable if res_block_type is "ddpm".
        use_upsample_conv: bool. Whether to use convolutional upsampling. Only
            applicable if res_block_type is "ddpm".
        res_block_type: str. ResNet block type, "ddpm" or "biggan".
        skip_rescale: bool. Whether to rescale output in resnet and attention
            blocks.
        use_scale_shift_norm: bool. Whether to scale and shift the group norm in
            the resnet blocks.
        num_attn_heads: int. Number of attention heads in the attention blocks.
        per_head_channels: int. Number of channels per head in the attention
            blocks. Only one of (num_attn_heads, per_head_channels) should be
            set to > 0.
        use_middle_stack_attn: bool. Whether to use an attention block in the
            middle stack.
        output_ch: int. Number of output channels.
  """

  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self,
               x=None,
               t=None,
               cond = None,
               train = True,
               **kwargs):
    """UNet forward pass with input x, conditioning input c, and time t.

    Args:
      x: noisy image for diffusion models otherwise None
      cond: conditioning dict
      t: time-step for diffusion models otherwise None
      train: whether this is during training (for dropout, etc.)

    Returns:
      network output.
    """
    config = self.config

    assert cond is None or isinstance(cond, dict), "cond isn't None or dict."

    assert x is not None  # diffusion model
    assert t is not None
    B, _, _, _, _ = x.shape  # pylint: disable=invalid-name
    assert t.shape == (B,)
    assert x.dtype in (jnp.float32, jnp.float64)
    # Timestep embedding.
    emb = get_timestep_embedding_with_projection(
        t, config.embed_dim * 4, name="dense")
    assert emb.shape == (B, config.embed_dim * 4)

    if cond is not None:
      # Text feature and/or CLIP-score feature is not supported yet
      # Text conditioning is involved.
      if config.get("text_cond_spec"):
        cond.update(add_text_features(config, cond, train=train))
        raise NotImplementedError

      # In-graph score conditioning (e.g. CLIP).
      if config.get("score_cond_spec"):
        cond.update(add_score_features(config, cond))
        raise NotImplementedError

      x, cemb = process_conditional_signal(
          x, cond, config.embed_dim * 4, config.get("num_classes"),
          config.get("max_time"))
      if cemb is not None:
        emb = emb + cemb if emb is not None else cemb

    # Downsample, middle and upsample stacks.
    h_list = DownsampleStack(config)(x, cond, emb, train=train)
    h = h_list[-1]
    h = MiddleStack(config)(h, cond, emb, train=train)
    h = UpsampleStack(config)(h, cond, emb, h_list, train=train)

    # End.
    h = nn.swish(normalization_layer(
        self.config.norm_type, name="norm_out")(h))

    h = nn.Conv(
        features=config.output_ch,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        kernel_init=default_init(),
        name="conv_out")(h)
    return h
