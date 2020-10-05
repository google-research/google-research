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

"""Dummy Agent."""

import os

import numpy as np

from ravens import cameras
from ravens import utils


class DummyAgent:
  """Dummy Agent."""

  def __init__(self, name, task):
    self.name = name
    self.task = task
    self.total_iter = 0

    # Share same camera configuration as Transporter.
    self.camera_config = cameras.RealSenseD415.CONFIG

    # [Optional] Heightmap parameters.
    self.pixel_size = 0.003125
    self.bounds = np.float32([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

    # A place to save pre-trained models.
    self.models_dir = os.path.join('checkpoints', self.name)
    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)

  def train(self, dataset, num_iter, writer, validation_dataset):
    """Train on dataset for a specific number of iterations."""
    del writer
    del validation_dataset

    for i in range(num_iter):
      obs, act, _ = dataset.random_sample()

      # [Optional] Get heightmap from RGB-D images.
      configs = act['camera_config']
      colormap, heightmap = self.get_heightmap(obs, configs)  # pylint: disable=unused-variable

      # Do something here.

      # Compute training loss here.
      loss = 0.
      print(f'Train Iter: {self.total_iter + i} Loss: {loss:.4f}')

    self.total_iter += num_iter
    self.save()

  def act(self, obs, info):
    """Run inference and return best action given visual observations."""
    del info

    act = {'camera_config': self.camera_config, 'primitive': None}
    if not obs:
      return act

    # [Optional] Get heightmap from RGB-D images.
    colormap, heightmap = self.get_heightmap(obs, self.camera_config)  # pylint: disable=unused-variable

    # Do something here.

    # Dummy behavior: move to the middle of the workspace.
    p0_position = (self.bounds[:, 1] - self.bounds[:, 0]) / 2
    p0_position += self.bounds[:, 0]
    p1_position = p0_position
    rotation = utils.get_pybullet_quaternion_from_rot((0, 0, 0))

    # Select task-specific motion primitive.
    act['primitive'] = 'pick_place'
    if self.task == 'sweeping':
      act['primitive'] = 'sweep'
    elif self.task == 'pushing':
      act['primitive'] = 'push'

    params = {
        'pose0': (p0_position, rotation),
        'pose1': (p1_position, rotation)
    }
    act['params'] = params
    return act

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def load(self, num_iter):
    """Load something."""

    # Do something here.

    # self.model.load(os.path.join(self.models_dir, model_fname))

    # Update total training iterations of agent.
    self.total_iter = num_iter

  def save(self):
    """Save models."""

    # Do something here.

    # self.model.save(os.path.join(self.models_dir, model_fname))

  def preprocess(self, image):
    """Pre-process images (subtract mean, divide by std)."""
    color_mean = 0.18877631
    depth_mean = 0.00509261
    color_std = 0.07276466
    depth_std = 0.00903967
    image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
    image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
    return image

  def get_heightmap(self, obs, configs):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = utils.reconstruct_heightmaps(
        obs['color'], obs['depth'], configs, self.bounds, self.pixel_size)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    colormap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    colormap = np.uint8(np.round(colormap))
    heightmap = np.sum(heightmaps, axis=0) / repeat

    return colormap, heightmap
