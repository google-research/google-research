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

# Lint as: python3
"""Fixed Replay Buffer."""

from absl import logging
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.replay_memory import circular_replay_buffer
import gin
import numpy as np
import tensorflow as tf


@gin.configurable
class FixedReplayBuffer(fixed_replay_buffer.FixedReplayBuffer):
  """Replay Buffers for loading existing data."""

  def __init__(self, data_dir,
               replay_suffix,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               replay_start_index=0,
               num_buffers_to_load=5,
               update_horizon=1,
               gamma=0.99,
               observation_dtype=np.uint8):
    """Initialize the FixedReplayBuffer class.

    Args:
      data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory. This can be
        used with `replay_start_index` to read a subset of replay data starting
        from a specific position.
      batch_size: int, Batch size for sampling data from buffer.
      replay_start_index: int, Starting index for loading the data from files
        in `data_dir`. This can be used to read a file starting from any index.
      num_buffers_to_load: int, number of replay buffers to load randomly in
        memory at every iteration from all buffers saved in `data_dir`.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
    """

    self._replay_start_index = replay_start_index
    self._num_buffers_to_load = num_buffers_to_load
    super().__init__(
        data_dir=data_dir,
        replay_suffix=replay_suffix,
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        observation_dtype=observation_dtype)

  def _load_buffer(self, suffix):
    """Loads a OutOfGraphReplayBuffer replay buffer."""
    try:
      # pytype: disable=attribute-error
      logging.info(
          'Starting to load from ckpt %d from %s', int(suffix), self._data_dir)

      replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
          *self._args, **self._kwargs)
      replay_buffer.load(self._data_dir, suffix)
      # pylint: disable = protected-access
      replay_capacity = replay_buffer._replay_capacity
      logging.info('Capacity: %d', replay_buffer._replay_capacity)
      logging.info('Start index: %d', self._replay_start_index)
      for name, array in replay_buffer._store.items():
        # This frees unused RAM if replay_capacity is smaller than 1M
        end_index = (
            self._replay_start_index + replay_capacity +
            replay_buffer._stack_size)
        replay_buffer._store[name] = array[
            self._replay_start_index: end_index].copy()
        logging.info('%s: %s', name, array.shape)
      logging.info('Loaded replay buffer from ckpt %d from %s',
                   int(suffix), self._data_dir)
      # pylint: enable=protected-access
      # pytype: enable=attribute-error
      return replay_buffer
    except tf.errors.NotFoundError:
      return None

  @property
  def replay_capacity(self):
    return self._replay_buffers[0]._replay_capacity  # pylint: disable = protected-access

  def reload_data(self):
    super().reload_buffer(num_buffers=self._num_buffers_to_load)
