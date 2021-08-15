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

import functools
import math
import random
import numpy as np

import gym
import torch
from ml_collections import ConfigDict

from sac import wrappers
from xirl import common


def seed_rngs(
    seed: int,
    cudnn_deterministic: bool = False,
    cudnn_benchmark: bool = True
):
  """Seeds python, numpy, and torch RNGs."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = cudnn_deterministic
  torch.backends.cudnn.benchmark = cudnn_benchmark


def wrap_env(env: gym.Env, config: ConfigDict, device: torch.device) -> gym.Env:
  """Wrap the environment based on values in the config."""
  if config.action_repeat > 1:
    env = wrappers.ActionRepeat(env, config.action_repeat)

  if config.frame_stack > 1:
    env = wrappers.FrameStack(env, config.frame_stack)

  if config.reward_wrapper.type != "none":
    model_config, model = common.load_model_checkpoint(
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
      kwargs["distance_scale"] = config.reward_wrapper.distance_scale
      if config.reward_wrapper.distance_func == "sigmoid":
        def sigmoid(x, t = 1.0):
          return 1 / (1 + math.exp(-x / t))

        kwargs["distance_func"] = functools.partial(
            sigmoid,
            config.reward_wrapper.distance_func_temperature,
        )
      env = wrappers.DistanceToGoalVisualReward(**kwargs)
    elif config.reward_wrapper.type == "goal_classifier":
      env = wrappers.GoalClassifierVisualReward(**kwargs)
    else:
      raise ValueError(
          f"{config.reward_wrapper.type} is not a supported reward wrapper.")

  env = wrappers.EpisodeMonitor(env)

  return env


def make_xmagical_env(embodiment: str) -> gym.Env:
  import xmagical
  xmagical.register_envs()
  return gym.make(f"SweepToTop-{embodiment.capitalize()}-State-Allo-TestLayout-v0")


def make_rlv_env() -> gym.Env:
  raise NotImplementedError
