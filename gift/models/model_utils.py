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

"""Model utils."""

import json
import operator
from absl import logging
import jax
from jax.tree_util import tree_map


def log_param_shapes(flax_module):
  """Prints out shape of parameters and total number of trainable parameters.

  Args:
    flax_module: A flax module.

  Returns:
    int; Total number of trainable parameters.
  """
  shape_dict = tree_map(lambda x: str(x.shape), flax_module.params)
  # we use json.dumps for pretty printing nested dicts
  logging.info('Printing model param shape:/n%s',
               json.dumps(shape_dict, sort_keys=True, indent=4))
  total_params = jax.tree_util.tree_reduce(
      operator.add, tree_map(lambda x: x.size, flax_module.params))
  logging.info('Total params of %s: %d', flax_module.module.__name__,
               total_params)
  return total_params
