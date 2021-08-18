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

"""RLV pretraining."""

import os.path as osp
import random
import subprocess
from absl import app
from absl import flags

# We want to pretrain on the entire demonstrations.
MAX_DEMONSTRATIONS = -1


def main(_):
  # Generate a unique experiment name.
  experiment_name = "rlv_algo=xirl_maxdemos={}_uid={}".format(
      MAX_DEMONSTRATIONS, int(random.random() * 1e9))
  print(f"Experiment name: {experiment_name}")

  subprocess.call([
      "python",
      "pretrain.py",
      "--experiment_name",
      experiment_name,
      "--config",
      "experiments/rlv/pretrain/xirl.py",
      "--config.DATA.MAX_VIDS_PER_CLASS",
      f"{MAX_DEMONSTRATIONS}",
  ])

  subprocess.call([
      "python",
      "compute_goal_embedding.py",
      "--experiment_path",
      # Note: This assumes that the config.ROOT_DIR value has not been
      # changed to its default value of 'tmp/xirl'.
      osp.join("/tmp/xirl", experiment_name),
  ])


if __name__ == "__main__":
  app.run(main)
