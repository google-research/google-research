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

"""Initializers module library for equivariant slot attention."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
from invariant_slot_attention.lib import utils

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


def get_uniform_initializer(vmin, vmax):
  """Get an uniform initializer with an arbitrary range."""
  init = nn.initializers.uniform(scale=vmax - vmin)

  def fn(*args, **kwargs):
    return init(*args, **kwargs) + vmin

  return fn


def get_normal_initializer(mean, sd):
  """Get a normal initializer with an arbitrary mean."""
  init = nn.initializers.normal(stddev=sd)

  def fn(*args, **kwargs):
    return init(*args, **kwargs) + mean

  return fn


class ParamStateInitRandomPositions(nn.Module):
  """Fixed, learnable state initalization with random positions.

  Random slot positions sampled from U[-1, 1] are concatenated
    as the last two dimensions.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.

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

    out = utils.broadcast_across_batch(param, batch_size=batch_size)
    shape = out.shape[:-1]
    rng = self.make_rng("state_init")
    slot_positions = jax.random.uniform(
        rng, shape=[*shape, 2], minval=self.slot_positions_min,
        maxval=self.slot_positions_max)
    out = jnp.concatenate((out, slot_positions), axis=-1)
    return out


class ParamStateInitLearnablePositions(nn.Module):
  """Fixed, learnable state initalization with learnable positions.

  Learnable initial positions are concatenated at the end of slots.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del inputs, train  # Unused.

    if self.init_fn == "normal":
      init_fn_state = functools.partial(nn.initializers.normal, stddev=1.)
    elif self.init_fn == "zeros":
      init_fn_state = lambda: nn.initializers.zeros
    else:
      raise ValueError("Unknown init_fn: {}.".format(self.init_fn))

    init_fn_state = init_fn_state()
    init_fn_pos = get_uniform_initializer(
        self.slot_positions_min, self.slot_positions_max)

    param_state = self.param("state_init", init_fn_state, self.shape)
    param_pos = self.param(
        "state_init_position", init_fn_pos, (*self.shape[:-1], 2))

    param = jnp.concatenate((param_state, param_pos), axis=-1)

    return utils.broadcast_across_batch(param, batch_size=batch_size)  # pytype: disable=bad-return-type  # jax-ndarray


class ParamStateInitRandomPositionsScales(nn.Module):
  """Fixed, learnable state initalization with random positions and scales.

  Random slot positions and scales sampled from U[-1, 1] and N(0.1, 0.1)
    are concatenated as the last four dimensions.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.
  slot_scales_mean: float = 0.1
  slot_scales_sd: float = 0.1

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

    out = utils.broadcast_across_batch(param, batch_size=batch_size)
    shape = out.shape[:-1]
    rng = self.make_rng("state_init")
    slot_positions = jax.random.uniform(
        rng, shape=[*shape, 2], minval=self.slot_positions_min,
        maxval=self.slot_positions_max)
    slot_scales = jax.random.normal(rng, shape=[*shape, 2])
    slot_scales = self.slot_scales_mean + self.slot_scales_sd * slot_scales
    out = jnp.concatenate((out, slot_positions, slot_scales), axis=-1)
    return out


class ParamStateInitLearnablePositionsScales(nn.Module):
  """Fixed, learnable state initalization with random positions and scales.

  Lernable initial positions and scales are concatenated at the end of slots.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.
  slot_scales_mean: float = 0.1
  slot_scales_sd: float = 0.01

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del inputs, train  # Unused.

    if self.init_fn == "normal":
      init_fn_state = functools.partial(nn.initializers.normal, stddev=1.)
    elif self.init_fn == "zeros":
      init_fn_state = lambda: nn.initializers.zeros
    else:
      raise ValueError("Unknown init_fn: {}.".format(self.init_fn))

    init_fn_state = init_fn_state()
    init_fn_pos = get_uniform_initializer(
        self.slot_positions_min, self.slot_positions_max)
    init_fn_scales = get_normal_initializer(
        self.slot_scales_mean, self.slot_scales_sd)

    param_state = self.param("state_init", init_fn_state, self.shape)
    param_pos = self.param(
        "state_init_position", init_fn_pos, (*self.shape[:-1], 2))
    param_scales = self.param(
        "state_init_scale", init_fn_scales, (*self.shape[:-1], 2))

    param = jnp.concatenate((param_state, param_pos, param_scales), axis=-1)

    return utils.broadcast_across_batch(param, batch_size=batch_size)  # pytype: disable=bad-return-type  # jax-ndarray


class ParamStateInitLearnablePositionsRotationsScales(nn.Module):
  """Fixed, learnable state initalization.

  Learnable initial positions, rotations and  scales are concatenated
    at the end of slots. The rotation matrix is flattened.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.
  slot_scales_mean: float = 0.1
  slot_scales_sd: float = 0.01
  slot_angles_mean: float = 0.
  slot_angles_sd: float = 0.1

  @nn.compact
  def __call__(self, inputs, batch_size,
               train = False):
    del inputs, train  # Unused.

    if self.init_fn == "normal":
      init_fn_state = functools.partial(nn.initializers.normal, stddev=1.)
    elif self.init_fn == "zeros":
      init_fn_state = lambda: nn.initializers.zeros
    else:
      raise ValueError("Unknown init_fn: {}.".format(self.init_fn))

    init_fn_state = init_fn_state()
    init_fn_pos = get_uniform_initializer(
        self.slot_positions_min, self.slot_positions_max)
    init_fn_scales = get_normal_initializer(
        self.slot_scales_mean, self.slot_scales_sd)
    init_fn_angles = get_normal_initializer(
        self.slot_angles_mean, self.slot_angles_sd)

    param_state = self.param("state_init", init_fn_state, self.shape)
    param_pos = self.param(
        "state_init_position", init_fn_pos, (*self.shape[:-1], 2))
    param_scales = self.param(
        "state_init_scale", init_fn_scales, (*self.shape[:-1], 2))
    param_angles = self.param(
        "state_init_angles", init_fn_angles, (*self.shape[:-1], 1))

    # Initial angles in the range of (-pi / 4, pi / 4) <=> (-45, 45) degrees.
    angles = jnp.tanh(param_angles) * (jnp.pi / 4)
    rotm = jnp.concatenate(
        [jnp.cos(angles), jnp.sin(angles),
         -jnp.sin(angles), jnp.cos(angles)], axis=-1)

    param = jnp.concatenate(
        (param_state, param_pos, param_scales, rotm), axis=-1)

    return utils.broadcast_across_batch(param, batch_size=batch_size)  # pytype: disable=bad-return-type  # jax-ndarray


class ParamStateInitRandomPositionsRotationsScales(nn.Module):
  """Fixed, learnable state initialization with random pos., rot. and scales.

  Random slot positions and scales sampled from U[-1, 1] and N(0.1, 0.1)
    are concatenated as the last four dimensions. Rotations are sampled
    from +- 45 degrees.
  Note: This module ignores any conditional input (by design).
  """

  shape: Sequence[int]
  init_fn: str = "normal"  # Default init with unit variance.
  conditioning_key: Optional[str] = None
  slot_positions_min: float = -1.
  slot_positions_max: float = 1.
  slot_scales_mean: float = 0.1
  slot_scales_sd: float = 0.1
  slot_angles_min: float = -jnp.pi / 4.
  slot_angles_max: float = jnp.pi / 4.

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

    out = utils.broadcast_across_batch(param, batch_size=batch_size)
    shape = out.shape[:-1]
    rng = self.make_rng("state_init")
    slot_positions = jax.random.uniform(
        rng, shape=[*shape, 2], minval=self.slot_positions_min,
        maxval=self.slot_positions_max)
    rng = self.make_rng("state_init")
    slot_scales = jax.random.normal(rng, shape=[*shape, 2])
    slot_scales = self.slot_scales_mean + self.slot_scales_sd * slot_scales
    rng = self.make_rng("state_init")
    slot_angles = jax.random.uniform(rng, shape=[*shape, 1])
    slot_angles = (slot_angles * (self.slot_angles_max - self.slot_angles_min)
                   ) + self.slot_angles_min
    slot_rotm = jnp.concatenate(
        [jnp.cos(slot_angles), jnp.sin(slot_angles),
         -jnp.sin(slot_angles), jnp.cos(slot_angles)], axis=-1)
    out = jnp.concatenate(
        (out, slot_positions, slot_scales, slot_rotm), axis=-1)
    return out
