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

"""Utilities for reading and writing sharded checkpoints."""

from typing import Optional

from flax.training import checkpoints
import jax
from jax import numpy as jnp
import numpy as np

from alx import als
from alx import multihost_utils


def save_checkpoint(state, work_dir):
  """Given ALSState, serializes embedding tables and saves a checkpoint."""
  step = state.step
  print(f"Saving checkpoint after epoch {step}.")

  host_dir = multihost_utils.get_host_dir(work_dir, host_id=jax.process_index())

  # Jax arrays can be sharded across devices, first device_get to host.
  state = jax.device_get(state)
  checkpoints.save_checkpoint(host_dir, state, step=step, keep=3)

  # Dummy psum to sync devices so that we wait for all host to finish saving
  # checkpoints to workdir.
  multihost_utils.sync_devices()


def restore_checkpoint(work_dir):
  """Given a valid dir, restores a checkpoint and returns ALSState object."""
  print(f"Attempting restore_checkpoint from dir: {work_dir}.")

  # Each host stores state in a seperate subdir.
  host_dir = multihost_utils.get_host_dir(work_dir, host_id=jax.process_index())

  # First retore to host then device_put sharded array.
  state = checkpoints.restore_checkpoint(host_dir, target=None)

  def device_put_sharded(x):
    if not isinstance(x, (jnp.ndarray, np.ndarray)):
      return x

    # Later, device_put_sharded takes a sequence of tensors, one tensor for
    # every local device. So we split it on the zeroth (device) dimension.
    x = np.reshape(x, [jax.local_device_count(), -1, x.shape[2]])
    x_list = np.split(x, x.shape[0], axis=0)

    # Squeeze out the dummy dimension.
    x_list = jax.tree_map(lambda y: np.squeeze(y, axis=0), x_list)

    # Send the sharded array in devices.
    return jax.device_put_sharded(x_list, jax.local_devices())

  state = jax.tree_map(device_put_sharded, state)
  return state
