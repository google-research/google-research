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

# Lint as: python3
"""An optimizer that simply zeros out weights.

This is a sanity check optimizer. All other optimizer should obtain higher loss
than this. Additionally values produced by this optimizer are used for
normalization.
"""

from typing import List

from task_set import registry
from task_set.optimizers import base
import tensorflow.compat.v1 as tf


@registry.optimizers_registry.register_fixed("zero")
class ZeroOptimizer(base.BaseOptimizer):
  r"""Zero out all weights each step.

  This optimizer is only to be used as a sanity check as it should only work
  well when parameters are initialized extremely poorly.
  """

  def minimize(self, loss, global_step,
               var_list):
    """Create op that zeros out all weights."""

    assign_ops = [v.assign(v * 0.) for v in var_list]
    return tf.group(global_step.assign_add(1), *assign_ops, name="minimize")
