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

"""Fully-connected model experiment functions."""

import collections
import sys
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from fast_gradient_clipping.src import custom_registry_functions
from fast_gradient_clipping.src import model_generators
from fast_gradient_clipping.src import profiling_tools


def fc_runner(param, runtimes, peak_memories, registry, repeats):
  """Fully-connected model runner."""
  p, q, r, m, batch_size = param
  print(
      f'Params: p = {p}, q = {q}, r = {r}, m = {m}, batch_size = {batch_size}'
  )
  sys.stdout.flush()
  # Direct model.
  direct_model = model_generators.make_direct_bias_model(p, q, r, m)
  cur_time, cur_mem = profiling_tools.get_compute_profile(
      direct_model, batch_size, registry, repeats=repeats
  )
  runtimes['direct_model'].append(cur_time)
  peak_memories['direct_model'].append(cur_mem)
  # Indirect model.
  indirect_model = model_generators.make_indirect_bias_model(p, q, r, m)
  cur_time, cur_mem = profiling_tools.get_compute_profile(
      indirect_model, batch_size, registry, repeats=repeats
  )
  runtimes['indirect_model'].append(cur_time)
  peak_memories['indirect_model'].append(cur_mem)


def get_fully_connected_compute_profile(params, repeats=20):
  """Main FC wrapper."""
  runtimes = collections.defaultdict(list)
  peak_memories = collections.defaultdict(list)
  registry = layer_registry.LayerRegistry()
  registry.insert(
      tf.keras.layers.Dense, custom_registry_functions.dense_layer_computation
  )
  registry.insert(
      tf.keras.layers.EinsumDense,
      custom_registry_functions.einsum_layer_computation,
  )
  for p in params:
    fc_runner(p, runtimes, peak_memories, registry, repeats)
  return runtimes, peak_memories
