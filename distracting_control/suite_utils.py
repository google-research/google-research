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

"""A collection of MuJoCo-based Reinforcement Learning environments.

The suite provides a similar API to the original dm_control suite.
Users can configure the distractions on top of the original tasks. The suite is
targeted for loading environments directly with similar configurations as those
used in the original paper. Each distraction wrapper can be used independently
though.
"""
import numpy as np

DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)
DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)
DEFAULT_BACKGROUND_PATH = "$HOME/davis/"


def get_color_kwargs(scale, dynamic):
  max_delta = scale
  step_std = 0.03 * scale if dynamic else 0.0
  return dict(max_delta=max_delta, step_std=step_std)


def get_camera_kwargs(domain_name, scale, dynamic):
  assert domain_name in ['reacher', 'cartpole', 'finger', 'cheetah',
                         'ball_in_cup', 'walker']
  assert scale >= 0.0
  assert scale <= 1.0
  return dict(
      vertical_delta=np.pi / 2 * scale,
      horizontal_delta=np.pi / 2 * scale,
      # Limit camera to -90 / 90 degree rolls.
      roll_delta=np.pi / 2. * scale,
      vel_std=.1 * scale if dynamic else 0.,
      max_vel=.4 * scale if dynamic else 0.,
      roll_std=np.pi / 300 * scale if dynamic else 0.,
      max_roll_vel=np.pi / 50 * scale if dynamic else 0.,
      max_zoom_in_percent=.5 * scale,
      max_zoom_out_percent=1.5 * scale,
      limit_to_upper_quadrant='reacher' not in domain_name,
  )


def get_background_kwargs(domain_name,
                          num_videos,
                          dynamic,
                          dataset_path,
                          dataset_videos=None,
                          shuffle=False,
                          video_alpha=1.0):
  assert domain_name in ['reacher', 'cartpole', 'finger', 'cheetah',
                         'ball_in_cup', 'walker']
  if domain_name == 'reacher':
    ground_plane_alpha = 0.0
  elif domain_name in ['walker', 'cheetah']:
    ground_plane_alpha = 1.0
  else:
    ground_plane_alpha = 0.3

  return dict(
      num_videos=num_videos,
      video_alpha=video_alpha,
      ground_plane_alpha=ground_plane_alpha,
      dynamic=dynamic,
      dataset_path=dataset_path,
      dataset_videos=dataset_videos,
      shuffle_buffer_size=100 if shuffle else None,
  )
