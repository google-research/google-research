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

"""Pixel-handling utilities."""
from typing import Optional, Tuple

from acme import types
from acme.wrappers import base
import cv2
import dm_env
from dm_env import specs as dm_env_specs
import numpy as np


RGB_INDEX = "RGB"  # Observation index for RGB data.
NUM_COLOR_CHANNELS = 3  # Number of color channels in RGB data.


class MeltingPotPixelsWrapper(base.EnvironmentWrapper):
  """Wrapper that stacks observations along a new final axis."""

  def __init__(self,
               environment,
               grayscale = False,
               scale_dims = (84, 84)):
    """Initializes a new MeltingPotPixelsWrapper.

    Args:
      environment: Environment.
      grayscale: Whether to convert rgb images to grayscale.
      scale_dims: Image size for the rescaling step after grayscaling, given as
        `(height, width)`. Set to `None` to disable resizing.
    """
    self._environment = environment
    self._grayscale = grayscale
    self._scale_dims = scale_dims
    self._observation_spec = self._init_observation_spec()

  def _init_observation_spec(self):
    obs_spec = self._environment.observation_spec()

    # init shape for pixels observations
    for player_key in obs_spec:
      pixels = obs_spec[player_key][RGB_INDEX]

      if self._scale_dims:
        self._height, self._width = self._scale_dims
      else:
        # Pixels is an RGB image with shape (H, W, 3).
        # By default, each agent's RGB has the same dimensions,
        # but we allow for agent-specific RGB shapes (for environments
        # in which RGB sizes might differ for each agent).
        self._height, self._width = pixels.shape[:2]

      num_channels = 1 if self._grayscale else NUM_COLOR_CHANNELS
      new_pixels_shape = (self._height, self._width, num_channels)

      pixel_spec = dm_env_specs.Array(
          shape=new_pixels_shape, dtype=pixels.dtype, name=pixels.name)
      obs_spec[player_key][RGB_INDEX] = pixel_spec

    return obs_spec

  def _process_pixels(self, observation):
    for player_key in observation:
      pixels = observation[player_key][RGB_INDEX]

      # rgb to grayscale
      if self._grayscale:
        # converts RGB frame to grayscale, using the color conversion
        # params described here: https://en.wikipedia.org/wiki/Luma_%28video%29
        # These are the same conversion params as used in Acme's
        # Atari wrapper: acme/wrappers/atari_wrapper.py
        pixels = np.tensordot(pixels, [0.299, 0.587, 1 - (0.299 + 0.587)],
                              (-1, 0))

      # resize
      pixels = np.round(pixels).astype(np.uint8)
      if self._scale_dims != pixels.shape[:2]:
        pixels = cv2.resize(
            pixels, (self._width, self._height), interpolation=cv2.INTER_AREA)
        pixels = np.round(pixels).astype(np.uint8)

      if self._grayscale:
        observation[player_key][RGB_INDEX] = pixels[Ellipsis, np.newaxis]
      else:
        observation[player_key][RGB_INDEX] = pixels
    return observation

  def _convert_timestep(self, source):
    """Returns multiplayer timestep from dmlab2d observations."""
    return dm_env.TimeStep(
        step_type=source.step_type,
        reward=source.reward,
        discount=0. if source.discount is None else source.discount,
        observation=self._process_pixels(source.observation))

  def reset(self):
    return self._convert_timestep(self._environment.reset())

  def step(self, action):
    return self._convert_timestep(self._environment.step(action))

  def observation_spec(self):
    return self._observation_spec
