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

"""ResNet V1."""

import collections
import functools

from absl import logging
from flax.deprecated import nn
import jax.numpy as jnp
import ml_collections

from gift.data import dataset_utils
from gift.models import base_model
from gift.nn import dann_utils
from gift.utils import tensor_util


class ResidualBlock(nn.Module):
  """ResNet block."""

  def apply(self,
            x,
            filters,
            strides=(1, 1),
            dropout_rate=0.0,
            epsilon=1e-5,
            momentum=0.9,
            norm_layer='batch_norm',
            train=True,
            dtype=jnp.float32):

    # TODO(samirabnar): Make 4 a parameter.
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    norm_layer_name = ''
    if norm_layer == 'batch_norm':
      norm_layer = nn.BatchNorm.partial(
          use_running_average=not train,
          momentum=momentum,
          epsilon=epsilon,
          dtype=dtype)
      norm_layer_name = 'bn'
    elif norm_layer == 'group_norm':
      norm_layer = nn.GroupNorm.partial(num_groups=16, dtype=dtype)
      norm_layer_name = 'gn'

    conv = nn.Conv.partial(bias=False, dtype=dtype)

    residual = x
    if needs_projection:
      residual = conv(residual, filters * 4, (1, 1), strides, name='proj_conv')
      residual = norm_layer(residual, name=f'proj_{norm_layer_name}')

    y = conv(x, filters, (1, 1), name='conv1')
    y = norm_layer(y, name=f'{norm_layer_name}1')
    y = nn.relu(y)

    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = norm_layer(y, name=f'{norm_layer_name}2')
    y = nn.relu(y)

    if dropout_rate > 0.0:
      y = nn.dropout(y, dropout_rate, deterministic=not train)
    y = conv(y, filters * 4, (1, 1), name='conv3')
    y = norm_layer(
        y, name=f'{norm_layer_name}3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)

    return y


class ResNet(base_model.BaseModel):
  """ResNetV1."""

  # A dictionary mapping the number of layers in a resnet to the number of
  # blocks in each stage of the model.
  _block_size_options = {
      1: [1],
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  def apply(self,
            inputs,
            num_outputs,
            num_filters=64,
            num_layers=50,
            dropout_rate=0.0,
            input_dropout_rate=0.0,
            train=True,
            dtype=jnp.float32,
            head_bias_init=jnp.zeros,
            return_activations=False,
            input_layer_key='input',
            has_discriminator=False,
            discriminator=False):
    """Apply a ResNet network on the input.

    Args:
      inputs: jnp array; Inputs.
      num_outputs: int; Number of output units.
      num_filters: int; Determines base number of filters. Number of filters in
        block i is  num_filters * 2 ** i.
      num_layers: int; Number of layers (should be one of the predefined ones.)
      dropout_rate: float; Rate of dropping out the output of different hidden
        layers.
      input_dropout_rate: float; Rate of dropping out the input units.
      train: bool; Is train?
      dtype: jnp type; Type of the outputs.
      head_bias_init: fn(rng_key, shape)--> jnp array; Initializer for head bias
        parameters.
      return_activations: bool; If True hidden activation are also returned.
      input_layer_key: str; Determines where to plugin the input (this is to
        enable providing inputs to slices of the model). If `input_layer_key` is
        `layer_i` we assume the inputs are the activations of `layer_i` and pass
        them to `layer_{i+1}`.
      has_discriminator: bool; Whether the model should have discriminator
        layer.
      discriminator: bool; Whether we should return discriminator logits.

    Returns:
      Unnormalized Logits with shape `[bs, num_outputs]`,
      if return_activations:
        Logits, dict of hidden activations and the key to the representation(s)
        which will be used in as ``The Representation'', e.g., for computing
        losses.
    """
    if num_layers not in ResNet._block_size_options:
      raise ValueError('Please provide a valid number of layers')

    block_sizes = ResNet._block_size_options[num_layers]

    layer_activations = collections.OrderedDict()
    input_is_set = False
    current_rep_key = 'input'
    if input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True

    if input_is_set:
      # Input dropout
      x = nn.dropout(x, input_dropout_rate, deterministic=not train)
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'init_conv'
    if input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_is_set:
      # First block
      x = nn.Conv(
          x,
          num_filters, (7, 7), (2, 2),
          padding=[(3, 3), (3, 3)],
          bias=False,
          dtype=dtype,
          name='init_conv')
      x = nn.BatchNorm(
          x,
          use_running_average=not train,
          momentum=0.9,
          epsilon=1e-5,
          dtype=dtype,
          name='init_bn')
      x = nn.relu(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    # Residual blocks
    for i, block_size in enumerate(block_sizes):

      # Stage i (each stage contains blocks of the same size).
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        current_rep_key = f'block_{i + 1}+{j}'
        if input_layer_key == current_rep_key:
          x = inputs
          input_is_set = True
          layer_activations[current_rep_key] = x
          rep_key = current_rep_key
        elif input_is_set:
          x = ResidualBlock(
              x,
              num_filters * 2**i,
              strides=strides,
              dropout_rate=dropout_rate,
              train=train,
              dtype=dtype,
              name=f'block_{i + 1}_{j}')
          layer_activations[current_rep_key] = x
          rep_key = current_rep_key

    current_rep_key = 'avg_pool'
    if input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_is_set:
      # Global Average Pool
      x = jnp.mean(x, axis=(1, 2))
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    # DANN module
    if has_discriminator:
      z = dann_utils.flip_grad_identity(x)
      z = nn.Dense(z, 2, name='disc_l1', bias=True)
      z = nn.relu(z)
      z = nn.Dense(z, 2, name='disc_l2', bias=True)

    current_rep_key = 'head'
    if input_layer_key == current_rep_key:
      x = inputs
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

      logging.warn('Input was never used')
    elif input_is_set:
      x = nn.Dense(
          x, num_outputs, dtype=dtype, bias_init=head_bias_init, name='head')

    # Make sure that the output is float32, even if our previous computations
    # are in float16, or other types.
    x = jnp.asarray(x, jnp.float32)

    outputs = x
    if return_activations:
      outputs = (x, layer_activations, rep_key)
      if discriminator and has_discriminator:
        outputs = outputs + (z,)
    else:
      del layer_activations
      if discriminator and has_discriminator:
        outputs = (x, z)
    if discriminator and (not has_discriminator):
      raise ValueError(
          'Incosistent values passed for discriminator and has_discriminator')
    return outputs

  @classmethod
  def build_flax_module(cls, hparams=None, dataset_metadata=None):
    """Build flax module (partially build by passing the hparams).

    API use to initialize a flax Model:
      ```
        model_def = model_cls.build_flax_module(hparams)
        _, initial_params = model_def.init_by_shape(
              rng, [((device_batch_size,)+dataset.meta_data['input_shape'][1:],
              jnp.float32)])
        model = nn.Model(model_def, initial_params)
      ```

    Args:
      hparams: ConfigDict; contains the hyperparams of the model architecture.
      dataset_metadata: dict; if hparams is None, dataset_meta data should be
        passed to provide the output_dim for the default hparam set.

    Returns:
      partially build class and hparams.
    """

    hparams = super(ResNet, cls).build_flax_module(hparams, dataset_metadata)
    model_dtype = dataset_utils.DATA_TYPE[hparams.get('model_dtype_str',
                                                      'float32')].jax_dtype
    return cls.partial(
        num_outputs=hparams.output_dim,
        num_filters=hparams.num_filters,
        num_layers=hparams.num_layers,
        dropout_rate=hparams.dropout_rate,
        input_dropout_rate=hparams.input_dropout_rate,
        head_bias_init=functools.partial(
            tensor_util.constant_initializer,
            fill_value=jnp.float32(hparams.get('head_bias_init', 0.0))),
        dtype=model_dtype,
        has_discriminator=hparams.get('has_discriminator', False)), hparams

  @classmethod
  def default_flax_module_hparams(cls, dataset_metadata):
    """Default hparams for the flax module that is built in `build_flax_module`.

    This function in particular serves the testing functions and supposed to
    provide hparams tha are passed to the flax_module when it's build in
    `build_flax_module` function, e.g., `model_dtype_str`.

    Args:
      dataset_metadata: dict; Passed to provide output dim.

    Returns:
      default hparams.
    """
    return ml_collections.ConfigDict(
        dict(
            output_dim=dataset_metadata['num_classes'],
            num_filters=32,
            num_layers=1,
            dropout_rate=0.1,
            input_dropout_rate=0.0,
            data_dtype_str='float32',
            has_discriminator=False,
        ))
