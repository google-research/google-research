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

"""Utility functions for handling device groups for JAX models."""

import jax


def get_device_groups(group_batch_size,
                      device_batch_size):
  """Get the device group ids.

  This function partitions all devices into equal sized groups so that each
  partition processes `group_batch_size` items. This is a copy from
  experimental/brain/selfattention/utils.py. The output of this function can be
  passed to lax collectives like `pmean`.

  Args:
    group_batch_size: the batch size per group.
    device_batch_size: the batch size per device.

  Returns:
    device_group_idx: a list of list where the inner list contains the device
      ids belonging to the same group, and the outer list contains the lists
      of all groups.
  """
  if group_batch_size % device_batch_size != 0:
    raise ValueError('Group batch size must be divisible by device batch size.')

  group_size = group_batch_size // device_batch_size
  if jax.device_count() % group_size != 0:
    raise ValueError('Number of devices must be divisible by group size')

  device_group_idx = [
      list(j
           for j in range(i, i + group_size))
      for i in range(0, jax.device_count(), group_size)
  ]
  return device_group_idx
