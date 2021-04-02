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

"""Section V.B: Learning from Cross-Embodiment Demonstrations, Fig 6."""

import itertools
import os.path as osp
import random
import subprocess

# In the paper, this experiment was performed on the 'shortstick' embodiment,
# with the XIRL algorithm.
ALGORITHM = "xirl"
EMBODIMENT = "shortstick"
# The embodiment classes in the dataset.
EMBODIMENTS = ["gripper", "shortstick", "mediumstick", "longstick"]
# The total amount of allowed demonstrations.
MAX_DEMONSTRATIONS = 1_000


if __name__ == "__main__":
  # Train on all classes but the given embodiment.
  trainable_embs = tuple(set(EMBODIMENTS) - set([EMBODIMENT]))
  max_demos_per = MAX_DEMONSTRATIONS // len(trainable_embs)

  # Generate a unique experiment name.
  experiment_name = "exp3_trainon={}_maxdemosper={}_uid={}".format(
      repr(trainable_embs), max_demos_per, int(random.random() * 1e9))
  print(f"Experiment name: {experiment_name}")

  subprocess.call([
      "python",
      "pretrain.py",
      "--experiment_name",
      experiment_name,
      "--config",
      "configs/pretraining/tcc.py",
      "--config.DATA.PRETRAIN_ACTION_CLASS",
      f"{repr(trainable_embs)}",
      # For the downstream action classes, we'll load all embodiments to
      # monitor performance on the unseen embodiment as well.
      "--config.DATA.DOWNSTREAM_ACTION_CLASS",
      "()",
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

  double_embs = list(itertools.combinations(trainable_embs, 2))
  max_demos_per = MAX_DEMONSTRATIONS // 2
  for trainable_embs in double_embs:
    # Generate a unique experiment name.
    experiment_name = "exp3_trainon={}_maxdemosper={}_uid={}".format(
        repr(trainable_embs), max_demos_per, int(random.random() * 1e9))
    print(f"Experiment name: {experiment_name}")

    subprocess.call([
        "python",
        "pretrain.py",
        "--experiment_name",
        experiment_name,
        "--config",
        "configs/pretraining/tcc.py",
        "--config.DATA.PRETRAIN_ACTION_CLASS",
        f"{repr(trainable_embs)}",
        # For the downstream action classes, we'll load all embodiments to
        # monitor performance on the unseen embodiment as well.
        "--config.DATA.DOWNSTREAM_ACTION_CLASS",
        "()",
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
