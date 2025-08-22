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

"""Embedding model experiment functions."""

import collections
import sys
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from fast_gradient_clipping.src import custom_registry_functions
from fast_gradient_clipping.src import model_generators
from fast_gradient_clipping.src import profiling_tools


def emb_runner(param, runtimes, peak_memories, repeats):
  """Embedding runner."""
  vocab_size, num_queries, output_dim = param
  print(
      f'Params: vocab_size = {vocab_size}, num_queries = {num_queries}, '
      f'output_dim = {output_dim}'
  )
  sys.stdout.flush()
  model = model_generators.make_embedding_model(
      vocab_size, num_queries, output_dim
  )
  # Direct model.
  direct_lr = layer_registry.make_default_layer_registry()
  cur_time, cur_mem = profiling_tools.get_compute_profile_with_vocab(
      model, vocab_size, num_queries, direct_lr, repeats=repeats
  )
  runtimes['direct_model'].append(cur_time)
  peak_memories['direct_model'].append(cur_mem)
  # Indirect model.
  indirect_lr = layer_registry.LayerRegistry()
  indirect_lr.insert(
      tf.keras.layers.Embedding,
      custom_registry_functions.naive_embedding_layer_computation,
  )
  cur_time, cur_mem = profiling_tools.get_compute_profile_with_vocab(
      model, vocab_size, num_queries, indirect_lr, repeats=repeats
  )
  runtimes['indirect_model'].append(cur_time)
  peak_memories['indirect_model'].append(cur_mem)


def get_embedding_compute_profile(params, repeats=20):
  """Embedding main wrapper."""
  runtimes = collections.defaultdict(list)
  peak_memories = collections.defaultdict(list)
  for p in params:
    emb_runner(p, runtimes, peak_memories, repeats)
  return runtimes, peak_memories
