import abc
import collections
from sac.wrappers import TensorType
import typing

import numpy as np

import torch

Batch = collections.namedtuple(
    "Batch", ["obses", "actions", "rewards", "next_obses", "masks"])


class ReplayBuffer:
  """Buffer to store environment transitions."""

  def __init__(
      self,
      obs_shape: typing.Tuple[int, ...],
      action_shape: typing.Tuple[int, ...],
      capacity: int,
      device: torch.device,
  ) -> None:
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

  def _empty_arr(
      self,
      shape: typing.Tuple[int],
      dtype: np.dtype,
  ) -> np.ndarray:
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
  ) -> None:
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


class ReplayBufferLearnedVisualReward(abc.ABC, ReplayBuffer):
  """Buffer that replaces the environment reward with a learned one.
  
  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      model,
      goal_emb,
      res_hw,
      distance_func,
      batch_size,
      **base_kwargs,
  ) -> None:
    super().__init__(**base_kwargs)

    self.model = model
    self.goal_emb = goal_emb
    self.res_hw = res_hw
    self.distance_func = distance_func
    self.batch_size = batch_size

    self._reset_staging()

  def _reset_staging(self):
    self.obses_staging = []
    self.next_obses_staging = []
    self.actions_staging = []
    self.rewards_staging = []
    self.masks_staging = []

  def _to_tensor(self, x: np.ndarray) -> TensorType:
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    x = x / 255.0
    x = x.to(self._device)
    return x

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
  ) -> None:
    if len(self.obses_staging) < self.batch_size:
      self.obses_staging.append(obs)
      self.next_obses_staging.append(next_obs)
      self.actions_staging.append(action)
      self.rewards_staging.append(reward)
      self.masks_staging.append(mask)
    else:
      del reward
      for obs, action, reward, next_obs, mask in zip(
          self.obses_staging,
          self.actions_staging,
          self._replace_rew(),
          self.next_obses_staging,
          self.masks_staging,
      ):
        super().insert(obs, action, reward, next_obs, mask)


class ReplayBufferDistanceToGoalLearnedVisualReward(
    ReplayBufferLearnedVisualReward):

  def _get_reward_from_image(self) -> float:
    image_tensors = [self._to_tensor(i) for i in self.obses_staging]
    image_tensors = torch.cat(image_tensors, dim=0)
    emb = self.model.infer(image_tensors).numpy().embs
    dists = np.linalg.norm(emb - self.goal_emb, axis=-1)
    dists *= -1.0
    return dists


class ReplayBufferGoalClassifierLearnedVisualReward(
    ReplayBufferLearnedVisualReward):

  def _get_reward_from_image(self) -> float:
    image_tensors = [self._to_tensor(i) for i in self.obses_staging]
    image_tensors = torch.cat(image_tensors, dim=0)
    prob = torch.sigmoid(self._model.infer(image_tensors).embs)
    return prob.item()
