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

# Lint as: python3
"""Build FLAX models for image classification."""

from typing import Optional, Tuple
import flax
import jax
from jax import numpy as jnp
from jax import random

from flax_models.cifar.models import pyramidnet
from flax_models.cifar.models import wide_resnet
from flax_models.cifar.models import wide_resnet_shakeshake


_AVAILABLE_MODEL_NAMES = [
    'WideResnet28x10',
    'WideResnet28x6_ShakeShake',
    'Pyramid_ShakeDrop',
    'WideResnet_mini',  # For testing/debugging purposes.
    'WideResnet_ShakeShake_mini',  # For testing/debugging purposes.
    'Pyramid_ShakeDrop_mini',  # For testing/debugging purposes.
]


def _create_image_model(
    prng_key, batch_size, image_size,
    module,
    num_channels = 3):
  """Instantiates a FLAX model and its state.

  Args:
    prng_key: PRNG key to use to sample the initial weights.
    batch_size: Batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    module: FLAX module describing the model to instantiates.
    num_channels: Number of channels for the images.

  Returns:
    A FLAX model and its state.
  """
  input_shape = (batch_size, image_size, image_size, num_channels)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = module.init_by_shape(
          prng_key, [(input_shape, jnp.float32)])
      model = flax.nn.Model(module, initial_params)
  return model, init_state


def get_model(
    model_name,
    batch_size,
    image_size,
    num_classes,
    num_channels = 3,
    prng_key = None,
    ):
  """Returns an initialized model of the chosen architecture.

  Args:
    model_name: Name of the architecture to use. Should be one of
      _AVAILABLE_MODEL_NAMES.
    batch_size: The batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    num_classes: Dimension of the output layer.
    num_channels: Number of channels for the images.
    prng_key: PRNG key to use to sample the weights.

  Returns:
    The initialized model and its state.

  Raises:
    ValueError if the name of the architecture is not recognized.
  """
  if model_name == 'WideResnet28x10':
    module = wide_resnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=num_classes)
  elif model_name == 'WideResnet28x6_ShakeShake':
    module = wide_resnet_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=4,
        channel_multiplier=6,
        num_outputs=num_classes)
  elif model_name == 'Pyramid_ShakeDrop':
    module = pyramidnet.PyramidNetShakeDrop.partial(num_outputs=num_classes)
  elif model_name == 'WideResnet_mini':  # For testing.
    module = wide_resnet.WideResnet.partial(
        blocks_per_group=2,
        channel_multiplier=1,
        num_outputs=num_classes)
  elif model_name == 'WideResnet_ShakeShake_mini':  # For testing.
    module = wide_resnet_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=2,
        channel_multiplier=1,
        num_outputs=num_classes)
  elif model_name == 'Pyramid_ShakeDrop_mini':
    module = pyramidnet.PyramidNetShakeDrop.partial(num_outputs=num_classes,
                                                    pyramid_depth=11)
  else:
    raise ValueError('Unrecognized model name.')
  if not prng_key:
    prng_key = random.PRNGKey(0)

  model, init_state = _create_image_model(prng_key, batch_size, image_size,
                                          module, num_channels)
  return model, init_state
