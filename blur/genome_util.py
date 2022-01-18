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

"""Utilities for genome handling."""

import dataclasses as dc
import functools as ft
from typing import Any, Callable, Optional, Union

import numpy as np
import tensorflow.compat.v1 as tf

from blur import blur_env



Tensor = Union[tf.Tensor, np.ndarray]


@dc.dataclass
class NeuronGenome:
  transform: Tensor
  keep: Union[float, Tensor] = 1.0
  update: Union[float, Tensor] = 1.0
  norm_multiplier: Union[float, Tensor] = 1.0
  norm_shift: Union[float, Tensor] = 0.0


@dc.dataclass
class HebbianTransform:
  pre: Tensor
  post: Tensor
  ojas_multiplier: Union[float, Tensor] = 1.0


@dc.dataclass
class SynapticGenome:
  transform: HebbianTransform
  synapse_init_std: Union[float, Tensor] = 1e-1
  synapse_init_xavier_std: Union[float, Tensor] = 0.0
  keep: Union[float, Tensor] = 1.0
  update: Union[float, Tensor] = 1.0
  saturation: Union[float, Tensor] = 1
  rescale_to: Union[float, Tensor] = 1.0


@dc.dataclass
class Genome:
  """Genome."""
  neuron: NeuronGenome
  synapse: SynapticGenome
  forward_synapse: Optional[SynapticGenome] = None

  def num_states_per_neuron(self):
    return get_num_states_in_genome(self)

  def num_species(self):
    return get_num_species_in_genome(self)

  def __post_init__(self):
    # By default we start with the same forward pass synapse genome that is
    # used on the backward pass; whether to do synaptic weight update on the
    # forward pass is decided in `network_step` based on the value of
    # `forward_synapse_update` in the network specification.
    if self.forward_synapse is None:
      self.forward_synapse = self.synapse


def _safe_shape(t):
  if hasattr(t, 'shape'):
    return t.shape
  else:
    return np.array(t).shape


def get_num_states_in_genome(g):
  return _safe_shape(g.synapse.transform.pre)[-1]


def transform_genome(g, map_fn, prefix=''):
  """Applies transformation to genome using map_fn."""
  r = {}
  for k, v in vars(g).items():
    if dc.is_dataclass(v):
      r[k] = transform_genome(v, map_fn=map_fn, prefix=f'{prefix}{k}/')
    else:
      mapped_value = map_fn(v, prefix + k)
      if mapped_value is not None:
        r[k] = mapped_value
  return dc.replace(g, **r)


def copy_genome(genome):
  return transform_genome(genome, lambda x, _: x)


def get_genome_slice(g, i):
  def fn(x, unused_name):
    # Necessary to avoid issues with tests restoring checkpoints.
    if isinstance(x, int) or isinstance(x, float):
      return x
    return x[i]
  return transform_genome(g, fn)


def get_genome(g, layer_index, per_layer_genome=False):
  if per_layer_genome:
    return  get_genome_slice(g, layer_index)
  else:
    return g


def convert_genome_to_tf_variables(g, prefix=''):
  """Converts genome to tensorflow variables with initialized to constant."""

  def map_fn(v, name):
    return tf.Variable(initial_value=v, dtype=tf.float32, name=name)

  return transform_genome(g, map_fn, prefix=prefix)


def convert_genome_to_dict(g):
  res = {}
  map_fn = lambda v, name: res.update([(name, v)])
  transform_genome(g, map_fn)
  return res


def _assign_from_values(v, name, values, index=None, prefix='', suffix=''):
  key = prefix + name + suffix
  if key not in values:
    tf.logging.warning(f'Genome parameter "{key}" cannot be found in the '
                       'dictionary.')
    return None
  if hasattr(v, 'shape') and index is not None:
    return values[key][index]
  else:
    return values[key]


def get_num_species_in_genome(g):
  shape = _safe_shape(g.synapse.transform.pre)
  return shape[0] if len(shape) == 3 else None


def genome_from_dict(values, index=None, prefix='', suffix=''):
  num_states = _safe_shape(values['synapse/transform/pre'])[-1]
  transform_fn = ft.partial(
      _assign_from_values,
      values=values,
      index=index,
      prefix=prefix,
      suffix=suffix)
  return transform_genome(create_random_genome(num_states), transform_fn)


def replicate_across_dims(value, shared_update_params, num_species, num_layers):
  if num_species is not None and not shared_update_params:
    value = np.array([value] * num_species)
  if num_layers is not None:
    value = np.array([value] * num_layers)
  return value


def create_random_genome(num_states,
                         num_species=None,
                         shared_update_params=True,
                         neuron_transform_std=1.0,
                         synapse_transform_std=1.0,
                         synapse_update=-1e-3,
                         synapse_init_std=1e-1,
                         separate_forward_synapse=False,
                         num_layers=None):
  """Creates random genome with that many species."""

  species_dims = (num_species,) if num_species is not None else ()
  if num_layers is not None:
    species_dims = (num_layers, *species_dims)

  maybe_shared = ft.partial(replicate_across_dims,
                            shared_update_params=shared_update_params,
                            num_species=num_species,
                            num_layers=num_layers)
  def _synaptic_genome(pre_transform, post_transform):
    return SynapticGenome(
        update=maybe_shared(synapse_update),
        keep=maybe_shared(1.0),
        synapse_init_std=maybe_shared(synapse_init_std),
        synapse_init_xavier_std=maybe_shared(0.0),
        saturation=maybe_shared(1.0),
        rescale_to=maybe_shared(1.0),
        transform=HebbianTransform(
            pre=pre_transform,
            post=post_transform,
            ojas_multiplier=maybe_shared(1.0)))

  matrix_shape = (*species_dims, num_states, num_states)
  o = np.ones(matrix_shape)
  z = np.zeros(matrix_shape)
  init_matrix = lambda: np.random.randn(*matrix_shape) * synapse_transform_std
  pre, post = init_matrix(), init_matrix()
  g = Genome(
      neuron=NeuronGenome(
          transform=(
              neuron_transform_std *
              np.random.randn(*species_dims, 2 * num_states, 2 * num_states) *
              np.block([[z, o], [o, z]])),
          update=maybe_shared(1.0),
          keep=maybe_shared(1.0),
          norm_multiplier=maybe_shared(1.0),
          norm_shift=maybe_shared(0.0)),
      synapse=_synaptic_genome(pre, post))
  if separate_forward_synapse:
    fwd_pre, fwd_post = init_matrix(), init_matrix()
    g.forward_synapse = _synaptic_genome(fwd_pre, fwd_post)
  return g




# Neuron transformation matrix \mu before being fed to synapse
# Rows describe contribution of corresponding state to all outputs
# Columns describe of all inputs to a corresponding output
#
# row 0: sensory(i) ('pre')
# row 1: feedback(i)
# row 2: sensory(j) ('post')
# row 3: feedback(j)
_grad_neuron_genome = np.array(
    [[0, 0, 1, 1],
     [0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0]], dtype=blur_env.NP_FLOATING_TYPE)  # pyformat: disable

# ΔW(i, j, o) = Σ_{k, l} n(i, k) @ pre(i, o) @ post(o, l) @ n(j, l)
# where n(i, k) is concatenation of input and output activations.
_grad_hebbian_genome = HebbianTransform(
    pre=np.array([[1, 0],
                  [0, 1]], dtype=blur_env.NP_FLOATING_TYPE),
    post=np.array([[0, 1],
                   [1, 0]], dtype=blur_env.NP_FLOATING_TYPE))
