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

"""Launch script for training RL policies with pretrained reward models."""

import collections
import os
import random
import typing

import gym
import numpy as np
import torch
import tqdm
import yaml
from absl import app, flags
from ml_collections import ConfigDict, config_flags
from torch.utils.tensorboard import SummaryWriter

from sac import agent, replay_buffer, video

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("embodiment", None, "The agent embodiment.")
flags.DEFINE_boolean("resume", False,
                     "Resume experiment from latest checkpoint.")

config_flags.DEFINE_config_file(
    "config",
    "configs/rl_default.py",
    "File path to the training hyperparameter configuration.",
)

flags.mark_flag_as_required("experiment_name")
flags.mark_flag_as_required("embodiment")


def evaluate(
    env: gym.Env,
    policy: agent.SAC,
    logger: SummaryWriter,
    step: int,
) -> None:
  """Evaluate the policy and dump rollout videos to disk."""
  policy.eval()
  eval_stats = collections.defaultdict(list)
  for episode in range(FLAGS.config.num_eval_episodes):
    observation = env.reset()
    while True:
      action = policy.act(observation, sample=False)
      observation, _, done, info = env.step(action)
      if done:
        for k, v in info["metrics"].items():
          eval_stats[k].append(v)
        break
  for k, v in eval_stats.items():
    logger.add_scalar(f"evaluation/{k}s", np.mean(v), step)


def start_or_resume(exp_dir: str, policy: agent.SAC) -> int:
  """Load a checkpoint if it exists, else start training from scratch."""
  model_dir = os.path.join(exp_dir, "weights")
  ckpts = []
  if os.path.exists(model_dir):
    ckpts = os.listdir(model_dir)
  if ckpts:
    last_ckpt = ckpts[-1]
    step = int(os.path.splitext(last_ckpt)[0].split("_")[-1])
    policy.load(os.path.join(model_dir, last_ckpt))
    return step
  return 0


def setup_experiment(exp_dir: str) -> None:
  """Setup the experiment."""
  exp_dir = os.path.join(FLAGS.config.save_dir, FLAGS.experiment_name)
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(FLAGS.config), fp)
  else:
    if not FLAGS.resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
      cfg = yaml.load(fp, Loader=yaml.FullLoader)
    FLAGS.config.update(cfg)
  if not os.path.exists(os.path.join(exp_dir, "weights")):
    os.makedirs(os.path.join(exp_dir, "weights"))


def main(_):
  exp_dir = os.path.join(FLAGS.config.save_dir, FLAGS.experiment_name)
  setup_experiment(exp_dir)
  device = torch.device(FLAGS.config.device)

  # Load env.
  env = None
  eval_env = None

  # Dynamically set observation and action space values.
  FLAGS.config.sac.obs_dim = env.observation_space.shape[0]
  FLAGS.config.sac.action_dim = env.action_space.shape[0]
  FLAGS.config.sac.action_range = [
      float(env.action_space.low.min()),
      float(env.action_space.high.max()),
  ]

  policy = agent.SAC(device, FLAGS.config.sac)

  # TODO(kevin): Load the learned replay buffer if we are using learned rewards.
  buffer = replay_buffer.ReplayBuffer(
      env.observation_space.shape,
      env.action_space.shape,
      FLAGS.config.replay_buffer_capacity,
      device,
  )

  logger = SummaryWriter(os.path.join(exp_dir, "tb"))

  try:
    start = start_or_resume(exp_dir, policy)
    done, info, observation = True, {"metrics": {}}, np.empty(())
    for i in tqdm.tqdm(
        range(start, FLAGS.config.num_train_steps), initial=start):
      if done:
        observation = env.reset()
        done = False
        for k, v in info["metrics"].items():
          logger.add_scalar(f"training/{k}", v, i)

      if i < FLAGS.config.num_seed_steps:
        action = env.action_space.sample()
      else:
        policy.eval()
        action = policy.act(observation, sample=True)
      next_observation, reward, done, info = env.step(action)

      if not done or "TimeLimit.truncated" in info:
        mask = 1.0
      else:
        mask = 0.0

      buffer.insert(observation, action, reward, next_observation, mask)
      observation = next_observation

      if i >= FLAGS.config.num_seed_steps:
        policy.train()
        train_info = policy.update(buffer, i)

        if (i + 1) % FLAGS.config.log_frequency == 0:
          for k, v in train_info.items():
            logger.add_scalar(k, v, i)

        logger.flush()

      if (i + 1) % FLAGS.config.eval_frequency == 0:
        evaluate(eval_env, policy, logger, i)

      if (i + 1) % FLAGS.config.checkpoint_frequency == 0:
        policy.save(os.path.join(exp_dir, "weights"), i)

  except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving before quitting.")

  finally:
    policy.save(os.path.join(exp_dir, "weights"), i)


if __name__ == "__main__":
  app.run(main)
