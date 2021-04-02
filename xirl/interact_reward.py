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

"""Teleop the agent and visualize the learned reward."""

from absl import app
from absl import flags
import gym
import matplotlib.pyplot as plt
from ml_collections.config_flags import config_flags
import torch
from wrappers import wrapper_from_config
import xmagical
from xmagical.utils import KeyboardEnvInteractor

FLAGS = flags.FLAGS

flags.DEFINE_string("embodiment", None, "The agent embodiment.")

config_flags.DEFINE_config_file(
    "config",
    "configs/rl/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("embodiment")


def make_env():
  xmagical.register_envs()
  embodiment_name = FLAGS.embodiment.capitalize()
  env = gym.make(f"SweepToTop-{embodiment_name}-State-Allo-TestLayout-v0")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = wrapper_from_config(FLAGS.config, env, device)
  return env


def main(_):
  env = make_env()
  viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])

  env.reset()
  obs = env.render("rgb_array")
  viewer.imshow(obs)

  i = [0]
  rews = []

  def step(action):
    obs, rew, done, info = env.step(action)
    rews.append(rew)
    if obs.ndim != 3:
      obs = env.render("rgb_array")
    if done and i[0] % 100 == 0:
      print(f"Done, score {info['eval_score']:.2f}/1.00")
    i[0] += 1
    return obs

  viewer.run_loop(step)

  # Plot the rewards over the episode.
  plt.plot(rews)
  plt.xlabel("Timestep")
  plt.ylabel("Reward")
  plt.show()


if __name__ == "__main__":
  app.run(main)
