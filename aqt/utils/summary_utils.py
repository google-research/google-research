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

"""Utility functions for retrieving state values from flax models."""

from typing import Any, Iterable, Mapping

from flax.metrics import tensorboard
import numpy as np


def get_state_dict_summary(state_dict,
                           keys):
  """Gets tagged distributions from state dictionary.

  Retrieves values for specified keys from state_dict if recorded.

  Args:
     state_dict: model state as dictionary.
     keys: The keys to look up in state_dict.

  Returns:
    dictionary containing state values corresponding to keys if recorded.

  """

  return _get_state_dict_summary_recursive(state_dict, keys, '', {})


def _get_state_dict_summary_recursive(parent_module, keys,
                                      path_to_parent_module, summary):
  for module_name, module in parent_module.items():
    path_to_module = f'{path_to_parent_module}/{module_name}'
    for key in keys:
      if key == module_name:
        summary[path_to_module] = module
    if hasattr(module, 'items'):  # Test if module is dictionary-like
      _get_state_dict_summary_recursive(module, keys, path_to_module, summary)
  return summary


def write_state_dict_summaries_to_tb(
    state_dict_summary_all,
    train_summary_writer,
    state_dict_summary_freq,
    step):
  """Write out state dict summaries to Tensorboard."""
  for key, val in state_dict_summary_all.items():
    val = np.asarray(val).astype(np.float32)
    steps_recorded = val.shape[0]
    if key.endswith('bounds'):
      # For bounds, also record mean bound as a scalar.
      for i in range(0, steps_recorded, state_dict_summary_freq):
        train_summary_writer.scalar(f'{key}_mean', val[i:i + 1].mean(), step)

    for i in range(0, steps_recorded, state_dict_summary_freq):
      train_summary_writer.histogram(
          key,
          val[i:i + 1],
          step=step - steps_recorded + i,
          bins=100)
