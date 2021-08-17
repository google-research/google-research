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

"""Environment wrappers."""

import abc
import collections

import time
import typing
import os

import imageio
import cv2
import gym
import numpy as np
import torch

from xirl.models import SelfSupervisedModel

TimeStep = typing.Tuple[np.ndarray, float, bool, dict]
ModelType = SelfSupervisedModel
TensorType = torch.Tensor
DistanceFuncType = typing.Callable[[float], float]
InfoMetric = typing.Mapping[str, typing.Mapping[str, typing.Any]]


class FrameStack(gym.Wrapper):
  """Stack the last k frames of the env into a flat array.

  This is useful for allowing the RL policy to infer temporal information.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env: gym.Env, k: int) -> None:
    """Constructor.

    Args:
      env: A gym env.
      k: The number of frames to stack.
    """
    super().__init__(env)

    assert isinstance(k, int), "k must be an integer."

    self._k = k
    self._frames = collections.deque([], maxlen=k)

    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low.min(),
        high=env.observation_space.high.max(),
        shape=((shp[0] * k,) + shp[1:]),
        dtype=env.observation_space.dtype,
    )

  def reset(self) -> np.ndarray:
    obs = self.env.reset()
    for _ in range(self._k):
      self._frames.append(obs)
    return self._get_obs()

  def step(self, action: np.ndarray) -> TimeStep:
    obs, reward, done, info = self.env.step(action)
    self._frames.append(obs)
    return self._get_obs(), reward, done, info

  def _get_obs(self) -> np.ndarray:
    assert len(self._frames) == self._k
    return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env: gym.Env, repeat: int) -> None:
    """Constructor.

    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action: np.ndarray) -> TimeStep:
    total_reward = 0.0
    for _ in range(self._repeat):
      obs, rew, done, info = self.env.step(action)
      total_reward += rew
      if done:
        break
    return obs, total_reward, done, info


class RewardScale(gym.Wrapper):
  """Scale the environment reward."""

  def __init__(self, env: gym.Env, scale: float) -> None:
    """Constructor.

    Args:
      env: A gym env.
      scale: How much to scale the reward by.
    """
    super().__init__(env)

    self._scale = scale

  def step(self, action: np.ndarray) -> TimeStep:
    obs, reward, done, info = self.env.step(action)
    reward *= self._scale
    return obs, reward, done, info


class VisualRewardWrapper(abc.ABC, gym.Wrapper):
  """Base wrapper class that replaces the env reward with a learned one.
  
  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      env: gym.Env,
      model: ModelType,
      device: torch.device,
      res_hw: typing.Optional[typing.Tuple[int, int]] = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      device: Compute device.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._res_hw = res_hw

  def _to_tensor(self, x: np.ndarray) -> TensorType:
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self) -> np.ndarray:
    """Render the pixels at the desired resolution."""
    pixels = self.env.render(mode="rgb_array")
    if self._res_hw is not None:
      h, w = self._res_hw
      pixels = cv2.resize(
          pixels,
          dsize=(w, h),
          interpolation=cv2.INTER_CUBIC,
      )
    return pixels

  @abc.abstractmethod
  def _get_reward_from_image(self, image: np.ndarray) -> float:
    """Forward the pixels through the model and compute the reward."""

  def step(self, action: np.ndarray) -> TimeStep:
    obs, env_reward, done, info = self.env.step(action)
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    return obs, learned_reward, done, info


class DistanceToGoalVisualReward(VisualRewardWrapper):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb: np.ndarray,
      distance_func: DistanceFuncType = None,
      **base_kwargs,
  ):
    super().__init__(**base_kwargs)

    self._goal_emb = goal_emb
    self._distance_func = distance_func

  def _get_reward_from_image(self, image: np.ndarray) -> float:
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    dist = np.linalg.norm(emb - self._goal_emb)
    if self._distance_func is not None:
      dist = self._distance_func(dist)
    else:
      dist = -1.0 * dist
    return dist


class GoalClassifierVisualReward(VisualRewardWrapper):
  """Replace the environment reward with the output of a goal classifier."""

  def _get_reward_from_image(self, image: np.ndarray) -> float:
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    prob = torch.sigmoid(self._model.infer(image_tensor).embs)
    return prob.item()


class EpisodeMonitor(gym.ActionWrapper):
  """A class that computes episode metrics.

  At minimum, episode return, length and duration are computed. Additional
  metrics that are logged in the environment's info dict can be monitored by
  specifying them via `info_metrics`.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env: gym.Env, info_metrics: InfoMetric = {}) -> None:
      """Constructor.

      Args:
        env: A gym env.
        info_metrics: Additional keys to monitor from the info dict returned
            by the env. This should be a mapping from the metric's str key
            in the info dict to an initial value.
      """
      super().__init__(env)

      self._info_metrics = info_metrics

      self._reset_stats()

  def _reset_stats(self) -> None:
    self.reward_sum = 0.0
    self.episode_length = 0
    self.start_time = time.time()
    self.extra_metrics = {}

  def step(self, action: np.ndarray) -> TimeStep:
    obs, rew, done, info = self.env.step(action)

    self.reward_sum += rew
    self.episode_length += 1
    for k in self._info_metrics.keys():
      self.extra_metrics[k] = info[k]

    info["metrics"] = {
        "episode_return": self.reward_sum,
        "episode_length": self.episode_length,
        "episode_duration": time.time() - self.start_time,
    }
    for k, v in self.extra_metrics.items():
      if "alias" in self._info_metrics[k]:
        key = self._info_metrics[k]["alias"]
      else:
        key = k
      info["metrics"][key] = v

    return obs, rew, done, info

  def reset(self) -> np.ndarray:
    self._reset_stats()
    return self.env.reset()


class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.
  
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env: gym.Env,
      save_dir: str,
      resolution: typing.Tuple[int, int] = (128, 128),
      fps: float = 30,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    self.height, self.width = resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = 0
    self.frames = []

  def step(self, action: np.ndarray) -> TimeStep:
    frame = self.env.render(mode="rgb_array")
    if frame.shape[:2] != (self.height, self.width):
      frame = cv2.resize(
          frame,
          dsize=(self.width, self.height),
          interpolation=cv2.INTER_CUBIC,
      )
    self.frames.append(frame)
    observation, reward, done, info = self.env.step(action)
    if done:
      filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
      imageio.mimsave(filename, self.frames, fps=self.fps)
      self.frames = []
      self.current_episode += 1
    return observation, reward, done, info
