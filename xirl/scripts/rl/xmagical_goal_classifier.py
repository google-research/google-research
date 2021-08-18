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

"""x-MAGICAL: Train a policy with the a goal classifier visual reward wrapper."""

import random
import subprocess
from absl import app
from absl import flags

# The embodiment classes in the dataset.
EMBODIMENTS = ["longstick", "mediumstick", "shortstick", "gripper"]
# Mapping from embodiment to number of training steps.
EMBODIMENT_TO_TRAIN_STEPS = {
    "longstick": 75_000,
    "mediumstick": 250_000,
    "shortstick": 500_000,
    "gripper": 500_000,
}

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_seeds", 5, "The number of seeds to run.")
flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train.")
flags.mark_flag_as_required("embodiment")


def main(_):
  pass


if __name__ == "__main__":
  app.run(main)
