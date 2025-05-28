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

"""Initializers module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn

import jax
import jax.numpy as jnp

from invariant_slot_attention.lib import utils
from invariant_slot_attention.modules import misc
from invariant_slot_attention.modules import video

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class ParamStateInit(nn.Module):
  """Fixed, learnable state initalization.

  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del inputs, train  # Unused.

    if self.init_fn == "normal":
      init_fn = functools.partial(nn.initializers.normal, stddev=1.)
    elif self.init_fn == "zeros":
      init_fn = lambda: nn.initializers.zeros
    else:
      raise ValueError("Unknown init_fn: {}.".format(self.init_fn))

    param = self.param("state_init", init_fn(), self.shape)
    return utils.broadcast_across_batch(param, batch_size=batch_size)


class GaussianStateInit(nn.Module):
  """Random state initialization with zero-mean, unit-variance Gaussian.

  Note: This module does not contain any trainable parameters and requires
    providing a jax.PRNGKey both at training and at test time. Note: This module
    also ignores any conditional input (by design).
  """

  shape: Sequence[int]

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del inputs, train  # Unused.
    rng = self.make_rng("state_init")
    return jax.random.normal(rng, shape=[batch_size] + list(self.shape))


class SegmentationEncoderStateInit(nn.Module):
  """State init that encodes segmentation masks as conditional input."""

  max_num_slots: int
  backbone: Callable[[], nn.Module]
  pos_emb: Callable[[], nn.Module] = misc.Identity
  reduction: Optional[str] = "all_flatten"  # Reduce spatial dim by default.
  output_transform: Callable[[], nn.Module] = misc.Identity
  zero_background: bool = False

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del batch_size  # Unused.

    # inputs.shape = (batch_size, seq_len, height, width)
    inputs = inputs[:, 0]  # Only condition on first time step.

    # Convert mask index to one-hot.
    inputs_oh = jax.nn.one_hot(inputs, self.max_num_slots)
    # inputs_oh.shape = (batch_size, height, width, n_slots)
    # NOTE: 0th entry inputs_oh[..., 0] will typically correspond to background.

    # Set background slot to all-zeros.
    if self.zero_background:
      inputs_oh = inputs_oh.at[:, :, :, 0].set(0)

    # Switch one-hot axis into 1st position (i.e. sequence axis).
    inputs_oh = jnp.transpose(inputs_oh, (0, 3, 1, 2))
    # inputs_oh.shape = (batch_size, max_num_slots, height, width)

    # Append dummy feature axis.
    inputs_oh = jnp.expand_dims(inputs_oh, axis=-1)

    # Vmapped encoder over seq. axis (i.e. we process each slot independently).
    encoder = video.FrameEncoder(
        backbone=self.backbone,
        pos_emb=self.pos_emb,
        reduction=self.reduction,
        output_transform=self.output_transform)  # type: ignore

    # encoder(inputs_oh).shape = (batch_size, n_slots, n_features)
    slots = encoder(inputs_oh, None, train)

    return slots


class CoordinateEncoderStateInit(nn.Module):
  """State init that encodes bounding box coordinates as conditional input.

  Attributes:
    embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
    prepend_background: Boolean flag; whether to prepend a special, zero-valued
      background bounding box to the input. Default: false.
    center_of_mass: Boolean flag; whether to convert bounding boxes to center
      of mass coordinates. Default: false.
    background_value: Default value to fill in the background.
  """

  embedding_transform: Callable[[], nn.Module]
  prepend_background: bool = False
  center_of_mass: bool = False
  background_value: float = 0.

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del batch_size  # Unused.

    # inputs.shape = (batch_size, seq_len, bboxes, 4)
    inputs = inputs[:, 0]  # Only condition on first time step.
    # inputs.shape = (batch_size, bboxes, 4)

    if self.prepend_background:
      # Adds a fake background box [0, 0, 0, 0] at the beginning.
      batch_size = inputs.shape[0]

      # Encode the background as specified by background_value.
      background = jnp.full(
          (batch_size, 1, 4), self.background_value, dtype=inputs.dtype)

      inputs = jnp.concatenate((background, inputs), axis=1)

    if self.center_of_mass:
      y_pos = (inputs[:, :, 0] + inputs[:, :, 2]) / 2
      x_pos = (inputs[:, :, 1] + inputs[:, :, 3]) / 2
      inputs = jnp.stack((y_pos, x_pos), axis=-1)

    slots = self.embedding_transform()(inputs, train=train)  # pytype: disable=not-callable

    return slots
