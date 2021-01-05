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

"""Adapters for IPAGNN models."""

import jax
import jax.numpy as jnp

from ipagnn.adapters import common_adapters
from ipagnn.lib import figure_utils


class IPAGNNAdapter(common_adapters.SequenceAdapter):
  """Adapter for IPAGNN model."""

  def as_example(self, dataset_item):
    inputs = jax.tree_map(lambda x: x.numpy(), dataset_item)
    example = {
        'true_branch_nodes': inputs['cfg']['true_branch_nodes'],
        'false_branch_nodes': inputs['cfg']['false_branch_nodes'],
        'start_index': inputs['cfg']['start_index'],
        'exit_index': inputs['cfg']['exit_index'],
        'data': inputs['cfg']['data'],
        'steps': inputs['cfg']['steps'],
        'target': inputs[self.info.supervised_keys[1]],
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

  def generate_plots(self, state, summary_writer=None, step=None):
    """Generates plots."""
    instruction_pointer = state['/instruction_pointer_before'][0, 0].T
    # instruction_pointer.shape: num_nodes, timesteps (excluding final)
    fig = figure_utils.make_figure(
        data=instruction_pointer,
        title='Instruction Pointer',
        xlabel='Timestep',
        ylabel='Node')
    if summary_writer:
      image = figure_utils.figure_to_image(fig)
      summary_writer.image('instruction-pointer', image, step)


class IPAGNNInterpolantAdapter(IPAGNNAdapter):
  """Use the same adapter for the interpolant models."""
  pass
