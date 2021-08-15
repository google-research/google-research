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
import functools
import math
import os
import random
import typing

import gym
import numpy as np
import torch
import tqdm
import xmagical
import yaml
from absl import app, flags
from ml_collections import ConfigDict, config_flags
from torch.utils.tensorboard import SummaryWriter

from sac import agent, replay_buffer, video, wrappers
from xirl import common

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("embodiment", None, "The agent embodiment.")
flags.DEFINE_boolean("resume", False, "Resume experiment from latest checkpoint.")

config_flags.DEFINE_config_file(
    "config",
    "configs/rl/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("experiment_name")
flags.mark_flag_as_required("embodiment")


def wrap_env(env: gym.Env, config: ConfigDict, device: torch.Device) -> gym.Env:
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


def make_env() -> gym.Env:
  xmagical.register_envs()
  embodiment_name = FLAGS.embodiment.capitalize()
  env = gym.make(f"SweepToTop-{embodiment_name}-State-Allo-TestLayout-v0")
  return env


def evaluate(
    env: gym.Env,
    policy: agent.SAC,
    video_recorder: video.VideoRecorder,
    logger: SummaryWriter,
    step: int,
) -> None:
  """Evaluate the policy and dump rollout videos to disk."""
  policy.eval()
  eval_stats = collections.defaultdict(list)
  for episode in range(FLAGS.config.num_eval_episodes):
    observation = env.reset()
    video_recorder.reset(enabled=(episode == 0))
    while True:
      action = policy.act(observation, sample=False)
      observation, _, done, info = env.step(action)
      video_recorder.record(env)
      if done:
        for k, v in info["metrics"].items():
          eval_stats[k].append(v)
        break
    video_recorder.save(f"{step}.mp4")
  for k, v in eval_stats.items():
    logger.add_scalar(f"evaluation/{k}s", np.mean(v), step)


def seed_rng(envs: typing.Sequence[gym.Env]) -> None:
  """Seed the RNGs across all modules."""
  for env in envs:
    env.seed(FLAGS.config.seed)
    env.action_space.seed(FLAGS.config.seed)
    env.observation_space.seed(FLAGS.config.seed)
  np.random.seed(FLAGS.config.seed)
  random.seed(FLAGS.config.seed)
  torch.manual_seed(FLAGS.config.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FLAGS.config.seed)
  torch.backends.cudnn.deterministic = FLAGS.config.cudnn_deterministic
  torch.backends.cudnn.benchmark = FLAGS.config.cudnn_benchmark


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
          "Experiment already exists. Run with --resume to continue."
      )
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
  env = wrap_env(make_env(), FLAGS.config, device)
  eval_env = wrap_env(make_env(), FLAGS.config, device)

  seed_rng([eval_env, env])

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

  video_recorder = video.VideoRecorder(
      exp_dir if FLAGS.config.save_video else None
  )

  logger = SummaryWriter(os.path.join(exp_dir, "tb"))

  try:
    start = start_or_resume(exp_dir, policy)
    done, info, observation = True, {"metrics": {}}, np.empty(())
    for i in tqdm.tqdm(range(start, FLAGS.config.num_train_steps), initial=start):
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
        evaluate(eval_env, policy, video_recorder, logger, i)

      if (i + 1) % FLAGS.config.checkpoint_frequency == 0:
        policy.save(os.path.join(exp_dir, "weights"), i)

  except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving before quitting.")

  finally:
    policy.save(os.path.join(exp_dir, "weights"), i)


if __name__ == "__main__":
    app.run(main)
