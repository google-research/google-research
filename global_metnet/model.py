# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Global MetNet model."""

import functools
from typing import Any, Callable, Optional, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

Dtype = Any
PRNGKey = Any
Shape = Sequence[int]

DENSE_INIT = nn.initializers.variance_scaling(
    scale=1.0 / 3, mode='fan_out', distribution='uniform'
)


def rescale_initializer(initializer, scale):
  def scaled_initializer(key, shape, dtype=jnp.float32):
    return scale * initializer(key, shape, dtype)

  return scaled_initializer


class DenseStack(nn.Module):
  """A stack of dense layers."""

  features: int = 4096
  output_channels: int = 512
  num_layers: int = 1
  activation: Callable[Ellipsis, Any] = nn.relu
  last_layer_init_scale: Optional[float] = None
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    out = inputs

    for _ in range(self.num_layers):
      out = nn.Dense(
          features=self.features,
          use_bias=False,
          kernel_init=DENSE_INIT,
          dtype=self.dtype,
      )(out)
      out = self.activation(out)

    pre_out = out
    kernel_initializer = DENSE_INIT
    if self.last_layer_init_scale is not None:
      kernel_initializer = rescale_initializer(
          nn.initializers.lecun_normal(), self.last_layer_init_scale
      )
    out = nn.Dense(
        features=self.output_channels,
        use_bias=False,
        kernel_init=kernel_initializer,
        dtype=self.dtype,
    )(pre_out)
    return out, pre_out


class ResidualBlock(nn.Module):
  """Full pre-activation ResNet block from https://arxiv.org/pdf/1603.05027.pdf.

  Initialized to identity.
  """

  filters: int
  dtype: Dtype = jnp.float32
  groupnorm: Optional[int] = None
  activation: Callable[[Any], Any] = nn.relu
  kernel_dilation: int = 1
  rescale_second_conv_init: float = 1e-3

  @nn.compact
  def __call__(self, x):
    needs_projection = x.shape[-1] != self.filters

    kernel_init = nn.initializers.lecun_normal(in_axis=[0, 1, 2])

    conv = lambda features, kernel_size, name, kernel_init=kernel_init: nn.Conv(
        features=features,
        use_bias=False,
        dtype=self.dtype,
        kernel_init=kernel_init,
        kernel_dilation=(self.kernel_dilation, self.kernel_dilation),
        kernel_size=kernel_size,
        name=name,
    )

    def norm(x):
      if self.groupnorm is None:
        return x
      else:
        return nn.GroupNorm(
            num_groups=self.groupnorm,
            use_scale=True,
            use_bias=True,
            dtype=self.dtype,
        )(x)

    r = x
    if needs_projection:
      r = conv(self.filters, (1, 1), name='proj_conv')(r)

    y = conv(self.filters, (3, 3), name='conv1')(x)

    y = norm(y)
    y = self.activation(y)
    y = conv(
        self.filters,
        (3, 3),
        name='conv2',
        kernel_init=rescale_initializer(
            kernel_init, self.rescale_second_conv_init
        ),
    )(y)

    output = r + y
    return output


class ResidualStack(nn.Module):
  """Stack of residual modules."""

  num_blocks: int
  filters: int
  groupnorm: Optional[int] = None
  kernel_dilations: tuple[int, Ellipsis] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
  remat: bool = False
  dtype: Dtype = jnp.float32
  extra_resnet_block_kwargs: Optional[dict[str, Any]] = None
  fix_cond_init: Optional[bool] = False

  @nn.compact
  def __call__(self, x):
    """Applies a stack of residual modules.

    Args:
      x: The input array.

    Returns:
      The result of applying the residual stack and passing it through a relu.
    """
    rematfn = nn.remat if self.remat else lambda f: f
    dense = functools.partial(nn.Dense, features=self.filters, dtype=self.dtype)
    residual_block = functools.partial(
        ResidualBlock,
        filters=self.filters,
        groupnorm=self.groupnorm,
        dtype=self.dtype,
        **(self.extra_resnet_block_kwargs or {}),
    )
    x_tot = dense(name='skip_init')(x)

    for i in range(self.num_blocks):

      def fn(module, x, x_tot, i=i):
        del module
        dilation = self.kernel_dilations[i % len(self.kernel_dilations)]
        x = residual_block(name=f'block{i}', kernel_dilation=dilation)(x)
        x_tot += dense(name=f'skip{i}')(x)
        return x, x_tot

      x, x_tot = rematfn(fn)(self, x, x_tot)
    return nn.relu(x_tot)


def onehot_range(x, num_classes):
  x = jnp.asarray(x, jnp.int32)
  return jnp.where(
      x[Ellipsis, 0] == -1,
      jnp.zeros(x.shape[:-1] + (num_classes,)),
      jax.nn.one_hot(x[Ellipsis, 0], num_classes),
  )


def upsample_by_repeat(x, times):
  """Upsample a tensor by repeating elements along axis 2 and 3."""
  b, h, w, c = x.shape
  x = jnp.reshape(x, [b, h, 1, w, 1, c])
  x = jnp.tile(x, [1, 1, times, 1, times, 1])
  return jnp.reshape(x, [b, h * times, w * times, c])


def space_to_depth(x, block_size):
  """Space to depth transform."""
  if x.ndim != 4:
    raise ValueError(f'Input tensor must have 4 dimensions, but has {x.ndim}')
  b, h, w, c = x.shape
  if h % block_size != 0 or w % block_size != 0:
    raise ValueError(
        f'Spatial dimensions must be divisible by {block_size}, '
        f'but are {h} and {w}'
    )
  x = jnp.reshape(
      x, [b, h // block_size, block_size, w // block_size, block_size, c]
  )
  x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
  x = jnp.reshape(
      x, [b, h // block_size, w // block_size, c * block_size * block_size]
  )
  return x


def history_encoder(inputs, target_index, hps):
  """Concatenate different inputs and aggregate history."""
  num_time_classes = len(hps.target_tds)
  target_features = hps.get('target_features', 512)

  dtype = (
      jnp.bfloat16 if hps.get('dtype', 'float32') == 'bfloat16' else jnp.float32
  )

  # Remove disabled inputs (constant zeros).
  inputs = {k: v for k, v in inputs.items() if v.ndim > 1}

  model_target_index = jnp.stack([target_index, target_index], axis=-1)
  target_index = onehot_range(model_target_index, num_classes=num_time_classes)
  target_index = target_index.astype(dtype)
  target_index = jnp.reshape(target_index, [-1, 1, 1, 1, num_time_classes])

  def flatten_input(input_):
    input_ = jnp.transpose(input_, [0, 2, 3, 1, 4])  # [b, w, h, t, f]
    input_ = jnp.reshape(input_, input_.shape[:3] + (-1,))
    return input_

  inputs = jax.tree.map(flatten_input, inputs)
  inputs = [inputs[k] for k in sorted(inputs.keys())]
  inputs = jnp.concatenate(inputs, axis=-1)
  out, target_embed = DenseStack(
      output_channels=2 * inputs.shape[-1],
      features=target_features,
      num_layers=1,
      dtype=dtype,
      name='DenseStack_1',
  )(target_index)
  target_bias, target_scale = jnp.split(out, 2, axis=-1)
  inputs += target_bias
  inputs *= target_scale

  return inputs, target_embed


def global_metnet_encoder(inputs, target_index, hps):
  """Dilated global convnet encoder. Multiple cropping stages."""
  # Expand context by repeating for global model
  def expand_context(data):
    context_size = sum(hps.encoder_crops) * hps.get('space_to_depth_factor', 1)

    def global_(x):
      n = len(x.shape)
      if n == 5:
        x_right = x[:, :, :, -context_size:, :]
        x_left = x[:, :, :, :context_size, :]
        x = jnp.concatenate((x_right, x, x_left), axis=-2)
      return x

    return jax.tree.map(global_, data)

  inputs = expand_context(inputs)

  encoded_input, _ = history_encoder(inputs, target_index, hps)

  encoder_channels = hps.encoder_channels
  encoder_num_blocks = hps.encoder_num_blocks
  encoder_crops = hps.encoder_crops
  # Per encoder hyperparams

  assert len(encoder_channels) == len(encoder_num_blocks) == len(encoder_crops)

  dtype = (
      jnp.bfloat16 if hps.get('dtype', 'float32') == 'bfloat16' else jnp.float32
  )
  groupnorm = None
  remat = True

  def make_encoder(i):
    return ResidualStack(
        num_blocks=encoder_num_blocks[i],
        filters=encoder_channels[i],
        remat=remat,
        groupnorm=groupnorm,
        name=f'encoder{i}',
        dtype=dtype,
    )

  encoded_input = space_to_depth(
      encoded_input, block_size=hps.get('space_to_depth_factor', 1)
  )

  for i in range(len(encoder_channels)):
    # Process
    encoded_input = make_encoder(i)(encoded_input)

    # Crop
    crop = encoder_crops[i]
    encoded_input = encoded_input[:, :, crop:-crop, :]

  encoded_input = upsample_by_repeat(
      encoded_input, hps.get('space_to_depth_factor', 1)
  )
  encoded_input = nn.relu(
      nn.Conv(
          features=encoder_channels[-1],
          padding='SAME',
          kernel_init=nn.initializers.orthogonal(),
          kernel_size=(3, 3),
          dtype=dtype,
          name='upsample_for_s2d_conv',
      )(encoded_input)
  )

  return encoded_input


class GlobalMetNet(nn.Module):
  """Global MetNet."""

  num_output_channels: int = 512
  hps: Optional[config_dict.ConfigDict] = None

  def __post_init__(self):
    self.input_keys = self.hps['inputs'].keys() + [
        'target_index',
        'lonlat_e6',
    ]
    super().__post_init__()

  @nn.compact
  def __call__(self, target_index, lonlat_e6, train, **inputs):
    hps = self.hps
    if hps is None:
      hps = config_dict.ConfigDict()

    final_block_hidden_size = 128
    dtype = (
        jnp.bfloat16
        if hps.get('dtype', 'float32') == 'bfloat16'
        else jnp.float32
    )
    encoded_input = global_metnet_encoder(inputs, target_index, hps)

    def final_block_fn(heads, hidden_size, prefix):
      def compute_outputs(module, input_):
        del module
        if not heads:
          return {}
        pre_final_conv = nn.relu(
            nn.Conv(
                features=hidden_size,
                padding='SAME',
                kernel_init=nn.initializers.orthogonal(),
                kernel_size=(3, 3),
                dtype=dtype,
                name=f'{prefix}conv_prefinal',
            )(input_)
        )
        outputs = {}
        for key, head in sorted(heads.items()):
          output = nn.Conv(
              features=head.num_output_channels,
              padding='SAME',
              kernel_init=nn.initializers.orthogonal(),
              kernel_size=(3, 3),
              dtype=dtype,
              name=f'{key}_{prefix}conv_final',
          )(pre_final_conv)
          output = jnp.expand_dims(output, axis=1)
          outputs[key] = output
        return outputs

      return compute_outputs

    outputs = final_block_fn(
        {
            k: head
            for k, head in hps.heads.items()
            if head.resolution_km == hps.input_resolution_km
            and head.requires_model_output()
        },
        hidden_size=final_block_hidden_size,
        prefix=f'{hps.input_resolution_km}km_',
    )(self, encoded_input)

    outputs = {k: v.astype(jnp.float32) for k, v in outputs.items()}
    return outputs
