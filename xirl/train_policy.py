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

from typing import Dict

import collections
import os.path as osp
import gym
import numpy as np
import torch
from tqdm.auto import tqdm
from absl import flags
from absl import logging
from absl import app
from ml_collections import config_flags
from ml_collections import config_dict
from sac import agent
from sac import replay_buffer
from torchkit.checkpoint import CheckpointManager
from torchkit import experiment
from torchkit import Logger
import utils
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("env_name", None, "The environment name.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("resume", False, "Resume experiment from last checkpoint.")

config_flags.DEFINE_config_file(
    "config",
    "configs/rl_default.py",
    "File path to the training hyperparameter configuration.",
)


def evaluate(
    policy: agent.SAC,
    env: gym.Env,
    num_episodes: int,
) -> Dict[str, float]:
  """Evaluate the policy and dump rollout videos to disk."""
  policy.eval()
  stats = collections.defaultdict(list)
  for _ in range(num_episodes):
    observation, done = env.reset(), False
    while not done:
      action = policy.act(observation, sample=False)
      observation, _, done, info = env.step(action)
    for k, v in info["episode"].items():
      stats[k].append(v)
    if "eval_score" in info:
      stats["eval_score"].append(info["eval_score"])
  for k, v in stats.items():
    stats[k] = np.mean(v)
  return stats


@experiment.pdb_fallback
def main(_):
  config = FLAGS.config
  exp_dir = osp.join(
      config.save_dir,
      FLAGS.experiment_name,
      str(FLAGS.seed),
  )
  utils.setup_experiment(exp_dir, config, FLAGS.resume)

  # Setup compute device.
  if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
  else:
    logging.info("No GPU device found. Falling back to CPU.")
    device = torch.device("cpu")
  logging.info(f"Using device: {device}")

  # Set RNG seeds.
  if FLAGS.seed is not None:
    logging.info(f"RL experiment seed: {FLAGS.seed}")
    experiment.seed_rngs(FLAGS.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this RL experiment.")

  # Load env.
  env = utils.make_env(
      FLAGS.env_name,
      FLAGS.seed,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
  )
  eval_env = utils.make_env(
      FLAGS.env_name,
      FLAGS.seed + 42,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
      save_dir=osp.join(exp_dir, "video", "eval"),
  )

  # Dynamically set observation and action space values.
  config.sac.obs_dim = env.observation_space.shape[0]
  config.sac.action_dim = env.action_space.shape[0]
  config.sac.action_range = [
      float(env.action_space.low.min()),
      float(env.action_space.high.max()),
  ]

  # Resave the config since the dynamic values have been updated at this point
  # and make it immutable for safety :)
  utils.dump_config(exp_dir, config)
  config = config_dict.FrozenConfigDict(config)

  policy = agent.SAC(device, config.sac)

  # TODO(kevin): Load the learned replay buffer if we are using learned rewards.
  buffer = replay_buffer.ReplayBuffer(
      env.observation_space.shape,
      env.action_space.shape,
      config.replay_buffer_capacity,
      device,
  )

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      policy=policy,
      **policy.optim_dict(),
  )

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)

  try:
    start = checkpoint_manager.restore_or_initialize()
    observation, done = env.reset(), False
    for i in tqdm(range(start, config.num_train_steps), initial=start):
      if i < config.num_seed_steps:
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

      if done:
        observation, done = env.reset(), False
        for k, v in info["episode"].items():
          logger.log_scalar(v, info["total"]["timesteps"], k, "training")

      if i >= config.num_seed_steps:
        policy.train()
        train_info = policy.update(buffer, i)

        if (i + 1) % config.log_frequency == 0:
          for k, v in train_info.items():
            logger.log_scalar(v, info["total"]["timesteps"], k, "training")
          logger.flush()

      if (i + 1) % config.eval_frequency == 0:
        eval_stats = evaluate(policy, eval_env, config.num_eval_episodes)
        for k, v in eval_stats.items():
          logger.log_scalar(
              v,
              info["total"]["timesteps"],
              f"average_{k}s",
              "evaluation",
          )
        logger.flush()

      if (i + 1) % config.checkpoint_frequency == 0:
        checkpoint_manager.save(i)

  except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving before quitting.")

  finally:
    checkpoint_manager.save(i)  # pylint: disable=undefined-loop-variable
    logger.close()


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  flags.mark_flag_as_required("env_name")
  app.run(main)
