# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Wide ResNet model.

This is based on the flax example for cifar-10.
"""

import collections
from absl import logging
from flax.deprecated import nn
import jax
import jax.numpy as jnp
import ml_collections
from gift.data import dataset_utils
from gift.models import base_model
from gift.nn import dann_utils


class WideResnetBlock(nn.Module):
  """Defines a single wide ResNet block."""

  def apply(self,
            x,
            channels,
            strides=(1, 1),
            dropout_rate=0.0,
            norm_layer='group_norm',
            train=True):
    norm_layer_name = ''
    if norm_layer == 'batch_norm':
      norm_layer = nn.BatchNorm.partial(use_running_average=not train)
      norm_layer_name = 'bn'
    elif norm_layer == 'group_norm':
      norm_layer = nn.GroupNorm.partial(num_groups=16)
      norm_layer_name = 'gn'

    y = norm_layer(x, name=f'{norm_layer_name}1')
    y = jax.nn.relu(y)
    y = nn.Conv(y, channels, (3, 3), strides, padding='SAME', name='conv1')
    y = norm_layer(y, name=f'{norm_layer_name}2')
    y = jax.nn.relu(y)
    if dropout_rate > 0.0:
      y = nn.dropout(y, dropout_rate, deterministic=not train)
    y = nn.Conv(y, channels, (3, 3), padding='SAME', name='conv2')

    # Apply an up projection in case of channel mismatch
    if (x.shape[-1] != channels) or strides != (1, 1):
      x = nn.Conv(x, channels, (3, 3), strides, padding='SAME')
    return x + y


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            dropout_rate=0.0,
            norm_layer='group_norm',
            train=True):
    for i in range(blocks_per_group):
      x = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          dropout_rate,
          norm_layer=norm_layer,
          train=train)
    return x


class WideResnet(base_model.BaseModel):
  """Defines the WideResnet Model."""

  def apply(
      self,
      inputs,
      blocks_per_group,
      channel_multiplier,
      num_outputs,
      kernel_size=(3, 3),
      strides=None,
      maxpool=False,
      dropout_rate=0.0,
      dtype=jnp.float32,
      norm_layer='group_norm',
      train=True,
      return_activations=False,
      input_layer_key='input',
      has_discriminator=False,
      discriminator=False,
  ):

    norm_layer_name = ''
    if norm_layer == 'batch_norm':
      norm_layer = nn.BatchNorm.partial(use_running_average=not train)
      norm_layer_name = 'bn'
    elif norm_layer == 'group_norm':
      norm_layer = nn.GroupNorm.partial(num_groups=16)
      norm_layer_name = 'gn'

    layer_activations = collections.OrderedDict()
    input_is_set = False
    current_rep_key = 'input'
    if input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'init_conv'
    if input_is_set:
      x = nn.Conv(
          x,
          16,
          kernel_size=kernel_size,
          strides=strides,
          padding='SAME',
          name='init_conv')
      if maxpool:
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'l1'
    if input_is_set:
      x = WideResnetGroup(
          x,
          blocks_per_group,
          16 * channel_multiplier,
          dropout_rate=dropout_rate,
          norm_layer=norm_layer,
          train=train,
          name='l1')
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'l2'
    if input_is_set:
      x = WideResnetGroup(
          x,
          blocks_per_group,
          32 * channel_multiplier, (2, 2),
          dropout_rate=dropout_rate,
          norm_layer=norm_layer,
          train=train,
          name='l2')
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'l3'
    if input_is_set:
      x = WideResnetGroup(
          x,
          blocks_per_group,
          64 * channel_multiplier, (2, 2),
          dropout_rate=dropout_rate,
          norm_layer=norm_layer,
          train=train,
          name='l3')
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    current_rep_key = 'l4'
    if input_is_set:
      x = norm_layer(x, name=f'{norm_layer_name}')
      x = jax.nn.relu(x)
      x = nn.avg_pool(x, (8, 8))
      x = x.reshape((x.shape[0], -1))
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key
    elif input_layer_key == current_rep_key:
      x = inputs
      input_is_set = True
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

    # DANN module
    if has_discriminator:
      z = dann_utils.flip_grad_identity(x)
      z = nn.Dense(z, 2, name='disc_l1', bias=True)
      z = nn.relu(z)
      z = nn.Dense(z, 2, name='disc_l2', bias=True)

    current_rep_key = 'head'
    if input_is_set:
      x = nn.Dense(x, num_outputs, dtype=dtype, name='head')
    else:
      x = inputs
      layer_activations[current_rep_key] = x
      rep_key = current_rep_key

      logging.warn('Input was never used')

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

    print(
        'in build_flax_module: has_discriminator',
        hparams.get('has_discriminator', False),
    )
    hparams = super(WideResnet, cls).build_flax_module(hparams,
                                                       dataset_metadata)
    model_dtype = dataset_utils.DATA_TYPE[hparams.get('model_dtype_str',
                                                      'float32')].jax_dtype

    return cls.partial(
        blocks_per_group=hparams.blocks_per_group,
        channel_multiplier=hparams.channel_multiplier,
        num_outputs=hparams.num_outputs,
        kernel_size=hparams.get('kernel_size', (3, 3)),
        strides=hparams.get('strides'),
        maxpool=hparams.get('maxpool', False),
        dropout_rate=hparams.dropout_rate,
        norm_layer=hparams.get('norm_layer', 'group_norm'),
        has_discriminator=hparams.get('has_discriminator', False),
        dtype=model_dtype), hparams

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
            num_outputs=dataset_metadata['num_classes'],
            blocks_per_group=4,
            channel_multiplier=10,
            dropout_rate=0.0,
            data_dtype_str='float32',
            has_discriminator=False,
        ))
