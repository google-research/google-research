# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Adroit pen environment variant with RGB and proprioceptive observations."""

from d4rl.hand_manipulation_suite import PenEnvV0


class VisualPenEnvV0(PenEnvV0):
  """Pen environment with visual and proprioceptive observations."""

  def __init__(self, camera_id, im_size, **kwargs):
    self._camera_id = camera_id
    self.im_size = im_size
    super().__init__(**kwargs)

  def get_obs(self):
    rgb = self.physics.render(
        self.im_size, self.im_size, camera_id=self._camera_id)
    qp = self.physics.data.qpos.ravel().copy()
    qv = self.physics.data.qvel.ravel().copy()
    palm_pos = self.physics.data.site_xpos[self.S_grasp_sid].ravel().copy()
    sensordata = self.physics.data.sensordata.ravel().copy()[20:41]
    original_obs = super().get_obs()
    return {'rgb': rgb, 'qpos': qp[:-6], 'qvel': qv[:-6], 'palm_pos': palm_pos,
            'tactile': sensordata, 'original_obs': original_obs}
