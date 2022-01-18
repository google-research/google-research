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

"""This contians a layer that handles input processing."""

from typing import Optional

from absl import logging
from flax import linen as nn
import jax.numpy as jnp

from autoregressive_diffusion.model.architecture_components import layers
from autoregressive_diffusion.model.architecture_components import time_embedding

Array = jnp.ndarray


class InputProcessingImage(nn.Module):
  """Inputs embedding for images."""

  num_classes: int
  num_channels: int
  max_time: float

  @nn.compact
  def __call__(self, x, t, mask, train):
    assert self.num_classes >= 1
    assert x.dtype == jnp.int32

    # Keep copy of integer version of x so that we can use it for embeddings if
    # needed.
    x_int = x

    if t is None:
      logging.info('Warning: Using architecture that does not condition on the '
                   'time/step variable. This may seriously impede performance.')
      t = jnp.zeros(x.shape[0], dtype=jnp.float32)

    # Normalizes the image by dividing by the number of classes.
    x = x * 2. / float(self.num_classes) - 1.

    # Concatenate mask (the mask can have multiple channels and is assumed to
    # contain only zeros and ones.
    x = jnp.concatenate([x, mask], axis=3)

    # Timestep embedding.
    temb = time_embedding.TimeEmbedding(self.num_channels, self.max_time)(t)

    # Assign 3/4 of channels to the standard 'float' representation of the
    # inputs.
    assert self.num_channels % 4 == 0
    h_first = nn.Conv(
        features=self.num_channels * 3 // 4,
        kernel_size=(3, 3),
        strides=(1, 1),
        name='conv_in')(x)

    # Here a 4th of the channels will be dedicated to the class embeddings.
    emb_ch = self.num_channels // 4

    emb_x = nn.Embed(self.num_classes, emb_ch)(x_int)
    emb_x = emb_x.reshape(*x_int.shape[:-1], emb_ch * x_int.shape[-1])

    # Dense general, projects down for the case that self.ch > 1, otherwise it's
    # Just a linear layer to the same dimensionality.
    h_emb_x = nn.Dense(features=emb_ch, name='emb_x_proj')(emb_x)

    # Combine h_extra with h_first.
    h_first = jnp.concatenate([h_first, h_emb_x], axis=3)

    return h_first, temb


def shift_one(x):
  """Shift time dimension by one to the right."""
  return jnp.pad(x[:, :-1, :], [(0, 0), (1, 0), (0, 0)])


class InputProcessingAudio(nn.Module):
  """Inputs embedding for audio."""

  num_classes: int
  num_channels: int
  max_time: float
  is_causal: bool = False

  @nn.compact
  def __call__(self,
               x,
               t,
               mask = None,
               train = False):
    assert self.num_classes >= 1
    assert x.dtype == jnp.int32

    # Keep copy of integer version of x so that we can use it for embeddings if
    # needed.
    x_int = x

    if t is None:
      logging.info('Warning: Using architecture that does not condition on the '
                   'time/step variable. This may seriously impede performance.')
      t = jnp.zeros(x.shape[0], dtype=jnp.float32)

    # Normalizes the audio by dividing by the number of classes.
    x = x * 2. / float(self.num_classes) - 1.

    # Concatenate mask (the mask can have multiple channels and is assumed to
    # contain only zeros and ones.
    if mask is not None:
      x = jnp.concatenate([x, mask], axis=-1)

    # Timestep embedding.
    if self.max_time > 0:
      t_embed = time_embedding.TimeEmbedding(self.num_channels, self.max_time)(
          t)
    else:
      t_embed = None

    assert self.num_channels % 4 == 0

    # Assign 3/4 of channels to the standard 'float' representation of the
    # inputs.
    x_causal = shift_one(x) if self.is_causal else x
    h_first = layers.CausalConv(
        features=self.num_channels * 3 // 4,
        kernel_size=(3,),
        strides=(1,),
        is_causal=self.is_causal,
        padding='VALID' if self.is_causal else 'SAME',
        name='conv_in')(
            x_causal)

    # Here a 4th of the channels will be dedicated to the class embeddings.
    emb_ch = self.num_channels // 4

    emb_x = nn.Embed(self.num_classes, emb_ch)(
        x_int)
    emb_x = emb_x.reshape(*x_int.shape[:-1], emb_ch * x_int.shape[-1])

    # Projects down for the case that self.ch > 1, otherwise it's just a linear
    # layer to the same dimensionality.
    h_emb_x = nn.Dense(features=emb_ch, name='emb_x_proj')(
        emb_x)

    if self.is_causal:
      h_emb_x = shift_one(h_emb_x)

    # Combine h_extra with h_first.
    h_first = jnp.concatenate([h_first, h_emb_x], axis=-1)

    return h_first, t_embed
