# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Tests for video_structure.vision."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf
from video_structure import hyperparameters
from video_structure import losses


class LossesTest(tf.test.TestCase):

  def setUp(self):

    # Hyperparameter config for test models:
    self.cfg = hyperparameters.get_config()
    self.cfg.batch_size = 1
    self.cfg.observed_steps = 2
    self.cfg.predicted_steps = 2
    self.cfg.num_keypoints = 3

    super().setUp()

  def _create_parallel_coords(self):
    """Create 3 keypoints that move along straight, parallel trajectories."""
    self.cfg.separation_loss_sigma = 0.01
    num_timesteps = self.cfg.observed_steps + self.cfg.predicted_steps
    # Create three points:
    coords = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float32)
    # Expand in time:
    coords = np.stack([coords] * num_timesteps, axis=0)
    # Add identical linear motion to all points:
    coords += np.linspace(-1, 1, num_timesteps)[:, np.newaxis, np.newaxis]
    return coords[np.newaxis, Ellipsis]  # Add batch dimension.

  def testTemporalSeparationLossParallelMovement(self):
    """Temporal separation loss should be high for parallel-moving keypoints."""
    with self.session() as sess:
      coords = tf.convert_to_tensor(self._create_parallel_coords())
      loss = sess.run(losses.temporal_separation_loss(self.cfg, coords))
    np.testing.assert_almost_equal(loss, 1.0, decimal=4)

  def testTemporalSeparationLossDifferentMovement(self):
    """Temporal separation loss should be low for nonparallel movement."""
    # Create trajectories in which all keypoints move differently:
    coords = self._create_parallel_coords()
    coords[:, 0, :] = -coords[:, 0, :]
    coords[:, 1, :] = 0.0
    with self.session() as sess:
      coords = tf.convert_to_tensor(coords)
      loss = sess.run(losses.temporal_separation_loss(self.cfg, coords))
    np.testing.assert_almost_equal(loss, 0.0, decimal=4)

if __name__ == '__main__':
  absltest.main()
