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

"""Utilities for synapse handling."""

import dataclasses as dc
import enum
import functools as ft
from typing import Callable, List, Sequence, Text, Union, Optional

import jax.numpy as jp
import numpy as np
import tensorflow.compat.v1 as tf

from blur import blur_env

TensorShape = tf.TensorShape
Tensor = Union[tf.Tensor, np.ndarray, jp.array]


@dc.dataclass
class SynapseInitializerParams:
  shape: TensorShape
  in_neurons: int
  out_neurons: int


class UpdateType(enum.Enum):
  FORWARD = 1
  BACKWARD = 2
  BOTH = 3
  NONE = 4


SynapseInitializer = Callable[[SynapseInitializerParams], Tensor]

# A callable that takes a sequence of layers and SynapseInitializer and creates
# appropriately shaped list of Synapses.
CreateSynapseFn = Callable[[Sequence[Tensor], SynapseInitializer], List[Tensor]]


def random_uniform_symmetric(shape, seed):
  return (tf.random.uniform(shape, seed=seed) - 0.5) * 2


def random_initializer(start_seed=0,
                       scale_by_channels=False,
                       scale=1,
                       bias=0,
                       random_fn=random_uniform_symmetric):
  """Returns initializer that generates random sequence."""
  seed = [hash(str(start_seed))]

  def impl(params):
    if len(params.shape) >= 3:
      # shape: species x (in+out) x (in+out) x states
      num_channels = int(params.shape[-2])
    seed[0] += 1
    v = random_fn(params.shape, seed[0])
    apply_scale = scale(params) if callable(scale) else scale
    r = v * apply_scale + bias
    if scale_by_channels:
      r = r / (num_channels**0.5)
    return r

  return impl


def _random_uniform_fn(start_seed):
  rng = np.random.RandomState(start_seed)
  return lambda shape: tf.constant(  # pylint: disable=g-long-lambda
      rng.uniform(low=-1, high=1, size=shape), dtype=np.float32)


def fixed_random_initializer(start_seed=0,
                             scale_by_channels=False,
                             scale=1,
                             bias=0,
                             random_fn=None):
  """Returns an initializer that generates random (but fixed) sequence.

  The resulting tensors are backed by a constant so they produce the same
  value across all calls.

  This initializer uses its own random state that is independent of default
  random sequence.

  Args:
    start_seed: initial seed passed to np.random.RandomStates
    scale_by_channels:  whether to scale by number of channels.
    scale: target scale (default: 1)
    bias: mean of the resulting distribution.
    random_fn: random generator if none will use use _random_uniform_fn

  Returns:
    callable that accepts shape and returns tensorflow constant tensor.
  """
  if random_fn is None:
    random_fn = _random_uniform_fn(start_seed)

  def impl(params):
    if len(params.shape) >= 3:
      # shape: species x (in+out) x (in+out) x states
      num_channels = int(params.shape[-2])
    v = random_fn(shape=params.shape)
    apply_scale = scale(params) if callable(scale) else scale
    r = v * apply_scale + bias
    if scale_by_channels:
      r = r / (num_channels**0.5)
    return r

  return impl


def create_synapse_init_fns(
    layers,
    initializer):
  """Generates network synapse initializers.

  Arguments:
    layers: Sequence of network layers (used for shape calculation).
    initializer: SynapseInitializer used to initialize synapse tensors.

  Returns:
    A list of functions that produce synapse tensors for all layers upon
    execution.
  """
  synapse_init_fns = []
  for pre, post in zip(layers, layers[1:]):
    # shape: population_dims, batch_size, in_channels, neuron_state
    pop_dims = pre.shape[:-3]
    # -2: is the number of channels
    num_inputs = pre.shape[-2] + post.shape[-2] + 1
    # -1: is the number of states in a single neuron.
    synapse_shape = (*pop_dims, num_inputs, num_inputs, pre.shape[-1])
    params = SynapseInitializerParams(
        shape=synapse_shape,
        in_neurons=pre.shape[-2],
        out_neurons=post.shape[-2])
    synapse_init_fns.append(ft.partial(initializer, params))
  return synapse_init_fns


def create_synapses(layers,
                    initializer):
  """Generates arbitrary form synapses.

  Arguments:
    layers: Sequence of network layers (used for shape calculation).
    initializer: SynapseInitializer used to initialize synapse tensors.

  Returns:
    A list of created synapse tensors for all layers.
  """
  return [init_fn() for init_fn in create_synapse_init_fns(layers, initializer)]


def transpose_synapse(synapse, env):
  num_batch_dims = len(synapse.shape[:-3])
  perm = [
      *range(num_batch_dims), num_batch_dims + 1, num_batch_dims,
      num_batch_dims + 2
  ]
  return env.transpose(synapse, perm)


def synapse_submatrix(synapse,
                      in_channels,
                      update_type,
                      include_bias = True):
  """Returns a submatrix of a synapse matrix given the update type."""
  bias = 1 if include_bias else 0
  if update_type == UpdateType.FORWARD:
    return synapse[Ellipsis, :(in_channels + bias), (in_channels + bias):, :]
  if update_type == UpdateType.BACKWARD:
    return synapse[Ellipsis, (in_channels + 1):, :(in_channels + bias), :]


def combine_in_out_synapses(in_out_synapse, out_in_synapse,
                            env):
  """Combines forward and backward synapses into a single matrix."""
  batch_dims = in_out_synapse.shape[:-3]
  out_channels, in_channels, num_states = in_out_synapse.shape[-3:]
  synapse = env.concat([
      env.concat([
          env.zeros((*batch_dims, out_channels, out_channels, num_states)),
          in_out_synapse
      ],
                 axis=-2),
      env.concat([
          out_in_synapse,
          env.zeros((*batch_dims, in_channels, in_channels, num_states))
      ],
                 axis=-2)
  ],
                       axis=-3)
  return synapse


def sync_all_synapses(synapses, layers, env):
  """Sync synapses across all layers.

  For each synapse, syncs its first state forward synapse with backward synapse
  and copies it arocess all the states.

  Args:
    synapses: list of synapses in the network.
    layers: list of layers in the network.
    env: Environment

  Returns:
    Synchronized synapses.
  """
  for i in range(len(synapses)):
    synapses[i] = sync_in_and_out_synapse(synapses[i], layers[i].shape[-2], env)
  return synapses


def sync_in_and_out_synapse(synapse, in_channels, env):
  """Copies forward synapse to backward one."""
  in_out_synapse = synapse_submatrix(
      synapse,
      in_channels=in_channels,
      update_type=UpdateType.FORWARD,
      include_bias=True)
  return combine_in_out_synapses(in_out_synapse,
                                 transpose_synapse(in_out_synapse, env), env)


def sync_states_synapse(synapse, env, num_states=None):
  """Sync synapse's first state across all the other states."""
  if num_states is None:
    num_states = synapse.shape[-1]
  return env.stack(num_states * [synapse[Ellipsis, 0]], axis=-1)


def normalize_synapses(synapses,
                       rescale_to,
                       env,
                       axis = -3):
  """Normalizes synapses across a particular axis (across input by def.)."""
  # Default value axis=-3 corresponds to normalizing across the input neuron
  # dimension.
  squared = env.sum(synapses**2, axis=axis, keepdims=True)
  synapses /= env.sqrt(squared + 1e-9)
  if rescale_to is not None:
    synapses *= rescale_to
  return synapses
