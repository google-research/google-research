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

"""Uniform replay buffer in Python with compressed storage.

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).
"""

import pickle
import threading

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory

from abps import py_uniform_replay_buffer


class FrameBuffer(tf.train.experimental.PythonState):
  """Saves some frames in a memory efficient way.

  Thread safety: cannot add multiple frames in parallel.
  """

  def __init__(self):
    self._frames = {}

  def add_frame(self, frame):
    """Add a frame to the buffer.

    Args:
      frame: Numpy array.

    Returns:
      A deduplicated frame.
    """
    h = hash(frame.tostring())
    if h in self._frames:
      _, refcount = self._frames[h]
      self._frames[h] = (frame, refcount + 1)
      return h
    self._frames[h] = (frame, 1)
    return h

  def __len__(self):
    return len(self._frames)

  def serialize(self):
    """Callback for `PythonStateWrapper` to serialize the dictionary."""
    return pickle.dumps(self._frames)

  def deserialize(self, string_value):
    """Callback for `PythonStateWrapper` to deserialize the array."""
    self._frames = pickle.loads(string_value)

  def compress(self, observation, split_axis=-1):
    # e.g. When split_axis is -1, turns an array of size 84x84x4
    # into a list of arrays of size 84x84x1.
    frame_list = np.split(observation, observation.shape[split_axis],
                          split_axis)
    return np.array([self.add_frame(f) for f in frame_list])

  def decompress(self, observation, split_axis=-1):
    frames = [self._frames[h][0] for h in observation]
    return np.concatenate(frames, axis=split_axis)

  def on_delete(self, observation, split_axis=-1):
    for h in observation:
      frame, refcount = self._frames[h]
      if refcount > 1:
        self._frames[h] = (frame, refcount - 1)
      else:
        del self._frames[h]

  def clear(self):
    self._frames = {}


class PyHashedReplayBuffer(py_uniform_replay_buffer.PyUniformReplayBuffer):
  """A Python-based replay buffer with optimized underlying storage.

  This replay buffer deduplicates data in the stored trajectories along the
  last axis of the observation, which is useful, e.g., if you are performing
  something like frame stacking. For example, if each observation is 4 stacked
  84x84 grayscale images forming a shape [84, 84, 4], then the replay buffer
  will separate out each of the images and depuplicate across each trajectory
  in case an image is repeated.

  Note: This replay buffer assumes that the items being stored are
  trajectory.Trajectory instances.
  """

  def __init__(self, data_spec, capacity, log_interval=None):
    if not isinstance(data_spec, trajectory.Trajectory):
      raise ValueError(
          'data_spec must be the spec of a trajectory: {}'.format(data_spec))
    super(PyHashedReplayBuffer, self).__init__(data_spec, capacity)

    self._frame_buffer = FrameBuffer()
    self._lock_frame_buffer = threading.Lock()
    self._log_interval = log_interval

  def _encoded_data_spec(self):
    observation = self._data_spec.observation
    observation = array_spec.ArraySpec(
        shape=(observation.shape[-1],), dtype=np.int64)
    return self._data_spec._replace(observation=observation)

  def _encode(self, traj):
    """Encodes a trajectory for efficient storage.

    The observations in this trajectory are replaced by a compressed
    version of the observations: each frame is only stored exactly once.

    Args:
      traj: The original trajectory.

    Returns:
      The same trajectory where frames in the observation have been
      de-duplicated.
    """
    with self._lock_frame_buffer:
      observation = self._frame_buffer.compress(traj.observation)

    if (self._log_interval and
        self._np_state.item_count % self._log_interval == 0):
      logging.info(
          '%s', 'Effective Replay buffer frame count: {}'.format(
              len(self._frame_buffer)))

    return traj._replace(observation=observation)

  def _decode(self, encoded_trajectory):
    """Decodes a trajectory.

    The observation in the trajectory has been compressed so that no frame
    is present more than once in the replay buffer. Uncompress the observations
    in this trajectory.

    Args:
      encoded_trajectory: The compressed version of the trajectory.

    Returns:
      The original trajectory (uncompressed).
    """
    observation = self._frame_buffer.decompress(encoded_trajectory.observation)
    return encoded_trajectory._replace(observation=observation)

  def _on_delete(self, encoded_trajectory):
    with self._lock_frame_buffer:
      self._frame_buffer.on_delete(encoded_trajectory.observation)

  def _clear(self):
    super(PyHashedReplayBuffer, self)._clear()
    self._frame_buffer.clear()
