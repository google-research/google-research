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

"""x-MAGICAL: Train a policy with a learned reward."""

import subprocess
from absl import app
from absl import flags
from absl import logging
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
CONFIG_PATH = "configs/xmagical/rl/learned_reward.py"

flags.DEFINE_integer("seeds", 1, "The number of seeds to run in parallel.")
flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train.")
flags.DEFINE_string("pretrained_path", None, "Path to pretraining experiment.")
flags.DEFINE_float("distance_scale", 1.0,
                   "Distance scale in reward computation.")
flags.DEFINE_enum("reward_type", "distance_to_goal",
                  ["distance_to_goal", "goal_classifier"],
                  "The type of reward function.")


def main(_):
  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]

  # Generate a unique experiment name.
  experiment_name = string_from_kwargs(
      env_name=env_name,
      reward="learned_reward",
      pretrained_path=FLAGS.pretrained_path,
      uid=unique_id(),
  )
  logging.info(f"Experiment name: {experiment_name}")

  # Execute each seed in parallel.
  procs = []
  for seed in range(FLAGS.seeds):
    procs.append(
        subprocess.Popen([  # pylint: disable=consider-using-with
            "python",
            "train_policy.py",
            "--experiment_name",
            experiment_name,
            "--env_name",
            f"{env_name}",
            "--config",
            f"{CONFIG_PATH}:{FLAGS.embodiment}",
            "--config.reward_wrapper.pretrained_path",
            f"{FLAGS.pretrained_path}",
            f"{FLAGS.reward_type}",
            "--config.reward_wrapper.type",
            f"{FLAGS.reward_type}",
            "--config.reward_wrapper.distance_scale",
            f"{FLAGS.distance_scale}",
            "--seed",
            f"{seed}",
        ]))

  # Wait for each seed to terminate.
  for p in procs:
    p.wait()


if __name__ == "__main__":
  flags.mark_flag_as_required("embodiment")
  flags.mark_flag_as_required("pretrained_path")
  app.run(main)
