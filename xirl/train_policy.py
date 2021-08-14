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

"""Launch script for training RL policies on x-MAGICAL."""

from absl import app
from absl import flags
import gym
from ml_collections.config_flags import config_flags
import torch
from torchsac.train import main as launch_rl
from wrappers import wrapper_from_config
import xmagical

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("embodiment", None, "The agent embodiment.")

config_flags.DEFINE_config_file(
    "config",
    "configs/rl/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("experiment_name")
flags.mark_flag_as_required("embodiment")


def make_env():
  xmagical.register_envs()
  embodiment_name = FLAGS.embodiment.capitalize()
  env = gym.make(f"SweepToTop-{embodiment_name}-State-Allo-TestLayout-v0")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = wrapper_from_config(FLAGS.config, env, device)
  return env


def main(_):
  launch_rl(FLAGS.config, make_env, FLAGS.experiment_name)


if __name__ == "__main__":
  app.run(main)
