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

"""Lightweight in-memory replay buffer.

Adapted from https://github.com/ikostrikov/jaxrl.
"""

import abc
import collections
from typing import Tuple, Optional

import cv2
import numpy as np

import torch
from xirl.models import SelfSupervisedModel

Batch = collections.namedtuple(
    "Batch", ["obses", "actions", "rewards", "next_obses", "masks"])
TensorType = torch.Tensor
ModelType = SelfSupervisedModel


class ReplayBuffer:
  """Buffer to store environment transitions."""

  def __init__(
      self,
      obs_shape: Tuple[int, ...],
      action_shape: Tuple[int, ...],
      capacity: int,
      device: torch.device,
  ):
    """Constructor.

        Args:
            obs_shape: The dimensions of the observation space.
            action_shape: The dimensions of the action space.
            capacity: The maximum length of the replay buffer.
            device: The torch device wherein to return sampled transitions.
        """
    self.capacity = capacity
    self.device = device

    obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
    self.obses = self._empty_arr(obs_shape, obs_dtype)
    self.next_obses = self._empty_arr(obs_shape, obs_dtype)
    self.actions = self._empty_arr(action_shape, np.float32)
    self.rewards = self._empty_arr((1,), np.float32)
    self.masks = self._empty_arr((1,), np.float32)

    self.idx = 0
    self.size = 0

  def _empty_arr(self, shape: Tuple[int], dtype: np.dtype) -> np.ndarray:
    """Creates an empty array of specified shape and type."""
    return np.empty((self.capacity, *shape), dtype=dtype)

  def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
    """Convert an ndarray to a torch Tensor and move it to the device."""
    return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

  def insert(
      self,
      obs: np.ndarray,
      action: np.ndarray,
      reward: float,
      next_obs: np.ndarray,
      mask: float,
  ):
    """Insert an episode transition into the buffer."""
    np.copyto(self.obses[self.idx], obs)
    np.copyto(self.actions[self.idx], action)
    np.copyto(self.rewards[self.idx], reward)
    np.copyto(self.next_obses[self.idx], next_obs)
    np.copyto(self.masks[self.idx], mask)

    self.idx = (self.idx + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int) -> Batch:
    """Sample an episode transition from the buffer."""
    idxs = np.random.randint(low=0, high=self.size, size=(batch_size,))

    return Batch(
        obses=self._to_tensor(self.obses[idxs]),
        actions=self._to_tensor(self.actions[idxs]),
        rewards=self._to_tensor(self.rewards[idxs]),
        next_obses=self._to_tensor(self.next_obses[idxs]),
        masks=self._to_tensor(self.masks[idxs]),
    )

  def __len__(self) -> int:
    return self.size


class ReplayBufferLearnedReward(abc.ABC, ReplayBuffer):
  """Buffer that replaces the environment reward with a learned one.

  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      model: ModelType,
      res_hw: Optional[Tuple[int, int]] = None,
      batch_size: int = 64,
      **base_kwargs,
  ) -> None:
    """Constructor.

    Args:
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
      batch_size: How many samples to forward through the model to compute the
        learned reward. Controls the size of the staging lists.
    """
    super().__init__(**base_kwargs)

    self.model = model
    self.res_hw = res_hw
    self.batch_size = batch_size

    self._reset_staging()

  def _reset_staging(self):
    self.obses_staging = []
    self.next_obses_staging = []
    self.actions_staging = []
    self.rewards_staging = []
    self.masks_staging = []
    self.pixels_staging = []

  def _pixel_to_tensor(self, arr: np.ndarray) -> TensorType:
    arr = torch.from_numpy(arr).permute(2, 0, 1).float()[None, None, Ellipsis]
    arr = arr / 255.0
    arr = arr.to(self.device)
    return arr

  @abc.abstractmethod
  def _get_reward_from_image(self) -> float:
    """Forward the pixels through the model and compute the reward."""

  def insert(
      self,
      obs: np.ndarray,
      action: np.ndarray,
      reward: float,
      next_obs: np.ndarray,
      mask: float,
      pixels: np.ndarray,
  ):
    if len(self.obses_staging) < self.batch_size:
      self.obses_staging.append(obs)
      self.next_obses_staging.append(next_obs)
      self.actions_staging.append(action)
      self.rewards_staging.append(reward)
      self.masks_staging.append(mask)
      if self.res_hw is not None:
        h, w = self.res_hw
        pixels = cv2.resize(pixels, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
      self.pixels_staging.append(pixels)
    else:
      for obs_s, action_s, reward_s, next_obs_s, mask_s in zip(
          self.obses_staging,
          self.actions_staging,
          self._get_reward_from_image(),
          self.next_obses_staging,
          self.masks_staging,
      ):
        super().insert(obs_s, action_s, reward_s, next_obs_s, mask_s)
      self._reset_staging()


class ReplayBufferDistanceToGoal(ReplayBufferLearnedReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb: np.ndarray,
      distance_scale: Optional[float] = 1.0,
      **base_kwargs,
  ):
    super().__init__(**base_kwargs)

    self.goal_emb = goal_emb
    self.distance_scale = distance_scale

  def _get_reward_from_image(self) -> float:
    image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
    image_tensors = torch.cat(image_tensors, dim=1)
    embs = self.model.infer(image_tensors).numpy().embs
    dists = -1.0 * np.linalg.norm(embs - self.goal_emb, axis=-1)
    dists *= self.distance_scale
    return dists


class ReplayBufferGoalClassifier(ReplayBufferLearnedReward):
  """Replace the environment reward with the output of a goal classifier."""

  def _get_reward_from_image(self) -> float:
    image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
    image_tensors = torch.cat(image_tensors, dim=1)
    prob = torch.sigmoid(self.model.infer(image_tensors).embs)
    return prob.item()
