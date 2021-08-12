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

import collections
import functools
import math
import os
import pickle
import typing

import cv2
import gym
from ml_collections import ConfigDict
import numpy as np
import torch
from torchkit import checkpoint
from xirl import common

TimeStep = typing.Tuple[np.ndarray, float, bool, dict]
ModelType = torch.nn.Module
DistanceFuncType = typing.Callable[[float], float]


def sigmoid(x, t = 1.0):
  return 1 / (1 + math.exp(-x / t))


def load_model(
    pretrained_path,
    load_goal_emb,
    device,
):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  config = common.load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = checkpoint.CheckpointManager(
      checkpoint.Checkpoint(model=model), checkpoint_dir, device)
  global_step = checkpoint_manager.restore_or_initialize()
  if load_goal_emb:
    print("Loading goal embedding.")
    with open(os.path.join(pretrained_path, "goal_emb.pkl"), "rb") as fp:
      goal_emb = pickle.load(fp)
    model.goal_emb = goal_emb
  print(f"Restored model from checkpoint @{global_step}.")
  return config, model


class DistanceToGoalVisualReward(gym.Wrapper):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      env,
      model,
      device,
      goal_emb,
      res_hw = None,
      distance_func = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings.
      device: Compute device.
      goal_emb: The goal embedding of shape (D,).
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
      distance_func: Optional function to apply on the embedding distance.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._goal_emb = goal_emb
    self._res_hw = res_hw
    self._distance_func = distance_func

  def _to_tensor(self, x):
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self):
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

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    dist = np.linalg.norm(emb - self._goal_emb)
    if self._distance_func is not None:
      dist = self._distance_func(dist)
    else:
      dist = -1.0 * dist
    return dist

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    return obs, learned_reward, done, info


class GoalClassifierVisualReward(gym.Wrapper):
  """Replace the environment reward with the output of a goal classifier."""

  def __init__(
      self,
      env,
      model,
      device,
      res_hw = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings.
      device: Compute device.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._res_hw = res_hw

  def _to_tensor(self, x):
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self):
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

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    prob = torch.sigmoid(self._model.infer(image_tensor).embs)
    return prob.item()

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    return obs, learned_reward, done, info


class FrameStack(gym.Wrapper):
  """Stack the last k frames of the env into a flat array.

  This is useful for allowing the RL policy to infer temporal information.
  """

  def __init__(self, env, k):
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

  def reset(self):
    obs = self.env.reset()
    for _ in range(self._k):
      self._frames.append(obs)
    return self._get_obs()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._frames.append(obs)
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    assert len(self._frames) == self._k
    return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment."""

  def __init__(self, env, repeat):
    """Constructor.

    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      obs, rew, done, info = self.env.step(action)
      total_reward += rew
      if done:
        break
    return obs, total_reward, done, info


def wrapper_from_config(config, env,
                        device):
  """Wrap the environment based on values in the config."""
  if config.action_repeat > 1:
    env = ActionRepeat(env, config.action_repeat)
  if config.frame_stack > 1:
    env = FrameStack(env, config.frame_stack)
  if config.reward_wrapper.type != "none":
    model_config, model = load_model(
        config.reward_wrapper.pretrained_path,
        # The goal classifier does not use a goal embedding.
        config.reward_wrapper.type != "goal_classifier",
        device,
    )
    kwargs = {
        "env": env,
        "model": model,
        "device": device,
        "res_hw": model_config.DATA_AUGMENTATION.IMAGE_SIZE,
    }
    if config.reward_wrapper.type == "distance_to_goal":
      kwargs["goal_emb"] = model.goal_emb
      if config.reward_wrapper.distance_func == "sigmoid":
        kwargs["distance_func"] = functools.partial(
            sigmoid,
            config.reward_wrapper.distance_func_temperature,
        )
      env = DistanceToGoalVisualReward(**kwargs)
    elif config.reward_wrapper.type == "goal_classifier":
      env = GoalClassifierVisualReward(**kwargs)
    else:
      raise ValueError(
          f"{config.reward_wrapper.type} is not a supported reward wrapper.")
  return env
