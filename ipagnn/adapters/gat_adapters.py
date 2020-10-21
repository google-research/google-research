# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Adapters for GAT models."""

import jax
import jax.numpy as jnp

from ipagnn.adapters import common_adapters


class GATAdapter(common_adapters.SequenceAdapter):
  """Adapter for GAT model."""

  def as_example(self, dataset_item):
    inputs = jax.tree_map(lambda x: x.numpy(), dataset_item)
    example = {
        'start_index': inputs['cfg']['start_index'],
        'exit_index': inputs['cfg']['exit_index'],
        'data': inputs['cfg']['data'],
        'edge_types': inputs['cfg']['edge_types'],
        'source_indices': inputs['cfg']['adjacency_list/source_indices'],
        'dest_indices': inputs['cfg']['adjacency_list/dest_indices'],
        'steps': inputs['cfg']['steps'],
        'target_output': inputs['target_output'],
        'target_output_length': inputs['target_output_length'],
        'human_readable_target_output': inputs['human_readable_target_output'],
        'human_readable_code': inputs['human_readable_code'],
    }
    if 'error_type' in inputs:
      example['error_type'] = inputs['error_type']
    return example

  def get_train_inputs(self, example):
    return {key: value for key, value in example.items()
            if value.dtype != jnp.dtype('O')}


class GGNNAdapter(common_adapters.SequenceAdapter):
  """Adapter for GGNN model."""

  def as_example(self, dataset_item):
    inputs = jax.tree_map(lambda x: x.numpy(), dataset_item)
    example = {
        'start_index': inputs['cfg']['start_index'],
        'exit_index': inputs['cfg']['exit_index'],
        'data': inputs['cfg']['data'],
        'edge_types': inputs['cfg']['edge_types'],
        'source_indices': inputs['cfg']['adjacency_list/source_indices'],
        'dest_indices': inputs['cfg']['adjacency_list/dest_indices'],
        'steps': inputs['cfg']['steps'],
        'target_output': inputs['target_output'],
        'target_output_length': inputs['target_output_length'],
        'human_readable_target_output': inputs['human_readable_target_output'],
        'human_readable_code': inputs['human_readable_code'],
    }
    if 'error_type' in inputs:
      example['error_type'] = inputs['error_type']
    return example

  def get_train_inputs(self, example):
    return {key: value for key, value in example.items()
            if value.dtype != jnp.dtype('O')}
