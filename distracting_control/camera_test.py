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

"""Tests for camera movement code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import suite as dm_control_suite
from dm_control.suite import cartpole
from dm_control.suite.wrappers import pixels
import numpy as np

from distracting_control import camera


def get_camera_params(domain_name, scale, dynamic):
  return dict(
      vertical_delta=np.pi / 2 * scale,
      horizontal_delta=np.pi / 2 * scale,
      # Limit camera to -90 / 90 degree rolls.
      roll_delta=np.pi / 2. * scale,
      vel_std=.1 * scale if dynamic else 0.,
      max_vel=.4 * scale if dynamic else 0.,
      roll_std=np.pi / 300 * scale if dynamic else 0.,
      max_roll_vel=np.pi / 50 * scale if dynamic else 0.,
      # Allow the camera to zoom in at most 50%.
      max_zoom_in_percent=.5 * scale,
      # Allow the camera to zoom out at most 200%.
      max_zoom_out_percent=1.5 * scale,
      limit_to_upper_quadrant='reacher' not in domain_name,
  )


def distraction_wrap(env, domain_name):
  camera_kwargs = get_camera_params(
      domain_name=domain_name, scale=0.0, dynamic=True)
  return camera.DistractingCameraEnv(env, camera_id=0, **camera_kwargs)


class CameraTest(absltest.TestCase):

  def test_dynamic(self):
    camera_kwargs = get_camera_params(
        domain_name='cartpole', scale=0.1, dynamic=True)
    env = cartpole.swingup()
    env = camera.DistractingCameraEnv(env, camera_id=0, **camera_kwargs)
    env = pixels.Wrapper(env, render_kwargs={'camera_id': 0})
    action_spec = env.action_spec()
    time_step = env.reset()
    frames = []
    while not time_step.last() and len(frames) < 10:
      action = np.random.uniform(
          action_spec.minimum, action_spec.maximum, size=action_spec.shape)
      time_step = env.step(action)
      frames.append(time_step.observation['pixels'])
    self.assertEqual(frames[0].shape, (240, 320, 3))

  def test_get_lookat_mat(self):
    agent_pos = np.array([1., -3., 4.])
    cam_position = np.array([0., 0., 0.])
    mat = camera.get_lookat_xmat_no_roll(agent_pos, cam_position)
    agent_pos = agent_pos / np.sqrt(np.sum(agent_pos**2.))
    start = np.array([0., 0., -1.])  # Cam starts looking down Z.
    out = np.dot(mat.reshape((3, 3)), start)
    self.assertTrue(np.isclose(np.max(np.abs(out - agent_pos)), 0.))

  def test_spherical_conversion(self):
    cart = np.array([1.4, -2.8, 3.9])
    sphere = camera.cart2sphere(cart)
    cart2 = camera.sphere2cart(sphere)
    self.assertTrue(np.isclose(np.max(np.abs(cart2 - cart)), 0.))

  def test_envs_same(self):
    # Test that the camera augmentations with magnitude 0 gives the same results
    # as when no camera augmentations are used.
    render_kwargs = {'width': 84, 'height': 84, 'camera_id': 0}
    domain_and_task = [('cartpole', 'swingup'),
                       ('reacher', 'easy'),
                       ('finger', 'spin'),
                       ('cheetah', 'run'),
                       ('ball_in_cup', 'catch'),
                       ('walker', 'walk')]
    for (domain, task) in domain_and_task:
      seed = 42
      envs = [('baseline',
               pixels.Wrapper(
                   dm_control_suite.load(
                       domain, task, task_kwargs={'random': seed}),
                   render_kwargs=render_kwargs)),
              ('no-wrapper',
               pixels.Wrapper(
                   dm_control_suite.load(
                       domain, task, task_kwargs={'random': seed}),
                   render_kwargs=render_kwargs)),
              ('w/-camera_kwargs',
               pixels.Wrapper(
                   distraction_wrap(
                       dm_control_suite.load(
                           domain, task, task_kwargs={'random': seed}), domain),
                   render_kwargs=render_kwargs))]
      frames = []
      for _, env in envs:
        random_state = np.random.RandomState(42)
        action_spec = env.action_spec()
        time_step = env.reset()
        frames.append([])
        while not time_step.last() and len(frames[-1]) < 20:
          action = random_state.uniform(
              action_spec.minimum, action_spec.maximum, size=action_spec.shape)
          time_step = env.step(action)
          frame = time_step.observation['pixels'][:, :, 0:3]
          frames[-1].append(frame)
      frames_np = np.array(frames)
      for i in range(1, len(envs)):
        difference = np.mean(abs(frames_np[0] - frames_np[i]))
        self.assertEqual(difference, 0.)

if __name__ == '__main__':
  absltest.main()
