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
import yaml
from absl import app
from absl import flags
from absl import logging
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", None, "Path to pretraining experiment.")
flags.DEFINE_list("seeds", [0, 5], "List specifying the range of seeds to run.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")


def main(_):
  with open(os.path.join(FLAGS.pretrained_path, "metadata.yaml"), "r") as fp:
    kwargs = yaml.load(fp, Loader=yaml.FullLoader)

  if kwargs["algo"] == "goal_classifier":
    reward_type = "goal_classifier"
  else:
    reward_type = "distance_to_goal"

  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[kwargs["embodiment"]]

  # Generate a unique experiment name.
  experiment_name = string_from_kwargs(
      env_name=env_name,
      reward="learned",
      reward_type=reward_type,
      mode=kwargs["mode"],
      algo=kwargs["algo"],
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
            f"configs/xmagical/rl/env_reward.py:{kwargs['embodiment']}",
            "--config.reward_wrapper.pretrained_path",
            f"{FLAGS.pretrained_path}",
            "--config.reward_wrapper.type",
            f"{reward_type}",
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
