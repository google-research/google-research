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

"""X-MAGICAL: Train a policy with a learned reward."""

import os
import subprocess
from absl import app
from absl import flags
from absl import logging
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
from configs.constants import EMBODIMENTS
from configs.constants import ALGORITHMS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from configs.constants import ALGO_TO_DISTANCE_SCALE_DICT
from utils import dict_from_experiment_name
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", None, "Path to pretraining experiment.")
flags.DEFINE_list("seeds", [0, 1], "List specifying the range of seeds to run.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")


def main(_):
  if not os.path.exists(FLAGS.pretrained_path):
    raise ValueError(f"{FLAGS.pretrained_path} does not exist.")

  # Parse some experiment args from the name of the pretrained_path. This is
  # hacky and I don't like it, but right now I'm trying to make it as easy as
  # possible to regenerate all the results we had in the paper with as little
  # manual input as possible.
  kwargs = dict_from_experiment_name(FLAGS.pretrained_path)
  embodiment = kwargs.pop("embodiment")
  algo = kwargs.pop("algo")
  assert embodiment in EMBODIMENTS
  assert algo in ALGORITHMS

  if algo == "goal_classifier":
    reward_type = "goal_classifier"
    distance_scale = 1.0  # This will be ignored.
  else:
    reward_type = "distance_to_goal"
    distance_scale = getattr(ALGO_TO_DISTANCE_SCALE_DICT[algo][embodiment],
                             kwargs["mode"])

  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[embodiment]

  # Generate a unique experiment name.
  experiment_name = string_from_kwargs(
      env_name=env_name,
      reward="learned",
      reward_type=reward_type,
      distance_scale=distance_scale,
      algo=algo,
      uid=unique_id(),
  )
  logging.info(f"Experiment name: {experiment_name}")

  # Execute each seed in parallel.
  procs = []
  for seed in range(*list(map(int, FLAGS.seeds))):
    procs.append(
        subprocess.Popen([  # pylint: disable=consider-using-with
            "python",
            "train_policy.py",
            "--experiment_name",
            experiment_name,
            "--env_name",
            f"{env_name}",
            "--config",
            f"configs/xmagical/rl/env_reward.py:{embodiment}",
            "--config.reward_wrapper.pretrained_path",
            f"{FLAGS.pretrained_path}",
            "--config.reward_wrapper.type",
            f"{reward_type}",
            "--config.reward_wrapper.distance_scale",
            f"{distance_scale}",
            "--seed",
            f"{seed}",
            "--device",
            f"{FLAGS.device}",
        ]))

  # Wait for each seed to terminate.
  for p in procs:
    p.wait()


if __name__ == "__main__":
  flags.mark_flag_as_required("pretrained_path")
  app.run(main)
