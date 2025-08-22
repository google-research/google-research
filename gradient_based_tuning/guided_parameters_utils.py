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
"""Guided Parameters utilities."""
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np


def get_init_by_name(name):
  """Returns an initialization function corresponding to 'name'."""
  lookup = {
      'const': init_const_fn,
      'linear': init_linear_fn,
  }
  if name not in lookup:
    raise ValueError('Unrecognized init type: %s' % name)
  return lookup[name]


def get_activation_fn_by_name(name):
  """Returns the default activation function corresponding to 'name'."""
  lookup = {
      'exp': activation_exp_fn,
      'sig': activation_sig_fn,
      'sigcent': activation_sig_centered_fn,
      'softplus': activation_softplus_fn,
      'self': activation_self_fn,
      'linear': activation_linear_fn,
      'relu': activation_relu_fn,
  }
  if name not in lookup:
    raise ValueError('Unrecognized activation_fn type: %s' % name)
  return lookup[name]


def get_activation_inverter_by_name(name):
  """Returns the default inverted activation fn corresponding to 'name'."""
  lookup = {
      'exp': activation_exp_inverter,
      'sig': activation_sig_inverter,
      'softplus': activation_softplus_inverter,
      'self': activation_self_fn,
      'linear': activation_linear_inverter,
      'relu': activation_linear_inverter,
  }
  if name not in lookup:
    raise ValueError('Unrecognized activation inverter type: %s' % name)
  return lookup[name]


# utility functions
def init_linear_fn(
    num_params, const = None
):
  """Returns a linear interpolation between 0 and 1 of length 'num_params'.

  Args:
    num_params: number of values to have in interpolation
    const: unused, needed for consistency with other init functions

  Returns:
    an array of num_params values linearly interpolated between 0 and 1
  """
  del const
  gen = (x / float(num_params - 1) for x in range(num_params))
  npret = np.fromiter(gen, dtype=np.float32)
  return jnp.array(npret, dtype=jnp.float32)


def init_const_fn(coordinate_shape, const):
  """Returns a jnp.ndarray with value 'const', length 'num_params'.

  Args:
    coordinate_shape: size of array
    const: value of all elements in array

  Returns:
    jnp.array
  """
  return jnp.full(coordinate_shape, const, dtype=np.float32)


# Activation functions applied to guided parameters. See guided_parameters.py
# for details.
def activation_exp_fn(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = 0.0,
):
  """Exponentiates the scaled input array."""
  del ceiling
  return jnp.exp(steepness * input_array) + floor


def activation_exp_inverter(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = 0.0,
):
  """Inverts activation_exp_fn."""
  del ceiling
  mapped = jnp.log(input_array - floor) / steepness
  return jnp.clip(mapped, -50)


def activation_self_fn(
    input_array,
    steepness = None,
    ceiling = None,
    floor = None,
):
  """Dummy function, returns input_array with no modification."""
  del steepness, ceiling, floor
  return input_array


def activation_relu_fn(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = None,
):
  """Passes scaled input_array through relu activation function."""
  del ceiling, floor
  return input_array * steepness * (input_array > 0)


# See https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html for
# details on softplus (via it's implementation in PyTorch).
def activation_softplus_fn(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = None,
):
  """Passes scaled input_array through softplus activation function."""
  del ceiling, floor
  return jnp.log(1 + jnp.exp(steepness * input_array))


def activation_softplus_inverter(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = None,
):
  """Inverts activation_softplus_fn."""
  del ceiling, floor
  return np.clip(jnp.log(jnp.exp(input_array) - 1) / steepness, -10, 99999999)


def activation_linear_fn(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = None,
):
  """Returns scaled input_array."""
  del ceiling, floor
  return input_array * steepness


def activation_linear_inverter(
    input_array,
    steepness = 1.0,
    ceiling = None,
    floor = None,
):
  """Inverts activation_linear_fn."""
  del ceiling, floor
  return input_array / steepness


def activation_sig_fn(
    input_array,
    steepness = 1.0,
    ceiling = 2.0,
    floor = 0.0,
):
  """Passes input_array through a sigmoid between [floor, ceiling]."""
  return (ceiling - floor) * (1 /
                              (1 + jnp.exp(-steepness * input_array))) + floor


def activation_sig_inverter(
    input_array,
    steepness = 1.0,
    ceiling = 2.0,
    floor = 0.0,
):
  """Inverts activation_sig_fn."""
  return jnp.clip((-1 / steepness) * (jnp.log(((ceiling - floor) /
                                               (input_array - floor)) - 1)),
                  -500, 500)


def activation_sig_centered_fn(
    input_array,
    steepness = 1.0,
    ceiling = 10.0,
    floor = 0.0,
):
  """Passes input_array through a sigmoid between [floor, ceiling]."""
  return (ceiling - floor) * (1.0 / (1.0 + jnp.exp(
      -steepness *
      (input_array + activation_sig_inverter(1.0, steepness, ceiling, floor))))
                             ) + floor
