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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for incontext library."""

import random
from typing import List, Union, Any
from absl import flags
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]
Dtype = Any


class ConfigDict(object):
  """Simple config dict."""

  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, my_dict):
    self.initial_dict = my_dict
    for key in my_dict:
      setattr(self, key, my_dict[key])

  def __str__(self):
    name = ""
    for k, v in self.initial_dict.items():
      name += str(k) + "=>" + str(v) + "\n"
    return name


def set_seed(seed):
  """Sets the seed for the random number generator.

  Args:
    seed (int): seed for the random number generator.

  Returns:
    List[int]: list of seeds used.
  """
  random.seed(seed)
  np.random.seed(seed)
  return [seed]


def flags_to_args():
  # pylint: disable=protected-access
  flags_dict = {k: v.value for k, v in flags.FLAGS.__flags.items()}
  # pylint: enable=protected-access
  return ConfigDict(flags_dict)
