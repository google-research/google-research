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

"""Useful methods shared by all scripts."""

import os
import pickle
import typing
from typing import Any, Dict, Optional

from absl import logging
import gym
from gym.wrappers import RescaleAction
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from sac import replay_buffer
from sac import wrappers
import torch
from torchkit import CheckpointManager
from torchkit.experiment import git_revision_hash
from xirl import common
import xmagical
import yaml

# pylint: disable=logging-fstring-interpolation

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict

# ========================================= #
# Experiment utils.
# ========================================= #


def setup_experiment(exp_dir, config, resume = False):
  """Initializes a pretraining or RL experiment."""
  #  If the experiment directory doesn't exist yet, creates it and dumps the
  # config dict as a yaml file and git hash as a text file.
  # If it exists already, raises a ValueError to prevent overwriting
  # unless resume is set to True.
  if os.path.exists(exp_dir):
    if not resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    load_config_from_dir(exp_dir, config)
  else:
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(config), fp)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
      fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir,
    config = None,
):
  """Load experiment config."""
  with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
  # Inplace update the config if one is provided.
  if config is not None:
    config.update(cfg)
    return
  return ConfigDict(cfg)


def dump_config(exp_dir, config):
  """Dump config to disk."""
  # Note: No need to explicitly delete the previous config file as "w" will
  # overwrite the file if it already exists.
  with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
    yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
    config,
    update_dict = None,
    freeze = False,
):
  """Makes a copy of a config and optionally updates its values."""
  # Using the ConfigDict constructor leaves the `FieldReferences` untouched
  # unlike `ConfigDict.copy_and_resolve_references`.
  new_config = ConfigDict(config)
  if update_dict is not None:
    new_config.update(update_dict)
  if freeze:
    return FrozenConfigDict(new_config)
  return new_config


def load_model_checkpoint(pretrained_path, device):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  config = load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  model.to(device).eval()
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
  global_step = checkpoint_manager.restore_or_initialize()
  logging.info("Restored model from checkpoint %d.", global_step)
  return config, model


def save_pickle(experiment_path, arr, name):
  """Save an array as a pickle file."""
  filename = os.path.join(experiment_path, name)
  with open(filename, "wb") as fp:
    pickle.dump(arr, fp)
  logging.info("Saved %s to %s", name, filename)


def load_pickle(pretrained_path, name):
  """Load a pickled array."""
  filename = os.path.join(pretrained_path, name)
  with open(filename, "rb") as fp:
    arr = pickle.load(fp)
  logging.info("Successfully loaded %s from %s", name, filename)
  return arr


# ========================================= #
# RL utils.
# ========================================= #


def make_env(
    env_name,
    seed,
    save_dir = None,
    add_episode_monitor = True,
    action_repeat = 1,
    frame_stack = 1,
):
  """Env factory with wrapping.

  Args:
    env_name: The name of the environment.
    seed: The RNG seed.
    save_dir: Specifiy a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.

  Returns:
    gym.Env object.
  """
  # Check if the env is in x-magical.
  xmagical.register_envs()
  if env_name in xmagical.ALL_REGISTERED_ENVS:
    env = gym.make(env_name)
  else:
    raise ValueError(f"{env_name} is not a valid environment name.")

  if add_episode_monitor:
    env = wrappers.EpisodeMonitor(env)
  if action_repeat > 1:
    env = wrappers.ActionRepeat(env, action_repeat)
  env = RescaleAction(env, -1.0, 1.0)
  if save_dir is not None:
    env = wrappers.VideoRecorder(env, save_dir=save_dir)
  if frame_stack > 1:
    env = wrappers.FrameStack(env, frame_stack)

  # Seed.
  env.seed(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  return env


def wrap_learned_reward(env, config):
  """Wrap the environment with a learned reward wrapper.

  Args:
    env: A `gym.Env` to wrap with a `LearnedVisualRewardWrapper` wrapper.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    gym.Env object.
  """
  pretrained_path = config.reward_wrapper.pretrained_path
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_config, model = load_model_checkpoint(pretrained_path, device)

  kwargs = {
      "env": env,
      "model": model,
      "device": device,
      "res_hw": model_config.data_augmentation.image_size,
  }

  if config.reward_wrapper.type == "goal_classifier":
    env = wrappers.GoalClassifierLearnedVisualReward(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.DistanceToGoalLearnedVisualReward(**kwargs)

  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")

  return env


def make_buffer(
    env,
    device,
    config,
):
  """Replay buffer factory.

  Args:
    env: A `gym.Env`.
    device: A `torch.device` object.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    ReplayBuffer.
  """

  kwargs = {
      "obs_shape": env.observation_space.shape,
      "action_shape": env.action_space.shape,
      "capacity": config.replay_buffer_capacity,
      "device": device,
  }

  pretrained_path = config.reward_wrapper.pretrained_path
  if not pretrained_path:
    return replay_buffer.ReplayBuffer(**kwargs)

  model_config, model = load_model_checkpoint(pretrained_path, device)
  kwargs["model"] = model
  kwargs["res_hw"] = model_config.data_augmentation.image_size

  if config.reward_wrapper.type == "goal_classifier":
    buffer = replay_buffer.ReplayBufferGoalClassifier(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    buffer = replay_buffer.ReplayBufferDistanceToGoal(**kwargs)

  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")

  return buffer


# ========================================= #
# Misc. utils.
# ========================================= #


def plot_reward(rews):
  """Plot raw and cumulative rewards over an episode."""
  _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
  axes[0].plot(rews)
  axes[0].set_xlabel("Timestep")
  axes[0].set_ylabel("Reward")
  axes[1].plot(np.cumsum(rews))
  axes[1].set_xlabel("Timestep")
  axes[1].set_ylabel("Cumulative Reward")
  for ax in axes:
    ax.grid(visible=True, which="major", linestyle="-")
    ax.grid(visible=True, which="minor", linestyle="-", alpha=0.2)
  plt.minorticks_on()
  plt.show()
