# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""X-MAGICAL same-embodiment pretraining script."""

import os.path as osp
import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import ALGORITHMS
from configs.constants import EMBODIMENTS
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
import yaml

# pylint: disable=logging-fstring-interpolation

# Mapping from pretraining algorithm to config file.
ALGO_TO_CONFIG = {
    "xirl": "configs/xmagical/pretraining/tcc.py",
    "lifs": "configs/xmagical/pretraining/lifs.py",
    "tcn": "configs/xmagical/pretraining/tcn.py",
    "goal_classifier": "configs/xmagical/pretraining/classifier.py",
    "raw_imagenet": "configs/xmagical/pretraining/imagenet.py",
}
# We want to pretrain on the entire demonstrations.
MAX_DEMONSTRATIONS = -1
FLAGS = flags.FLAGS

flags.DEFINE_enum("algo", None, ALGORITHMS, "The pretraining algorithm to use.")
flags.DEFINE_enum(
    "embodiment", None, EMBODIMENTS,
    "Which embodiment to train. Will train all sequentially if not specified.")
flags.DEFINE_bool("unique_name", False,
                  "Whether to append a unique ID to the experiment name.")


def main(_):
  embodiments = EMBODIMENTS if FLAGS.embodiment is None else [FLAGS.embodiment]

  for embodiment in embodiments:
    # Generate a unique experiment name.
    kwargs = {
        "dataset": "xmagical",
        "mode": "same",
        "algo": FLAGS.algo,
        "embodiment": embodiment,
    }
    if FLAGS.unique_name:
      kwargs["uid"] = unique_id()
    experiment_name = string_from_kwargs(**kwargs)
    logging.info("Experiment name: %s", experiment_name)

    subprocess.run(
        [
            "python",
            "pretrain.py",
            "--experiment_name",
            experiment_name,
            "--raw_imagenet" if FLAGS.algo == "raw_imagenet" else "",
            "--config",
            f"{ALGO_TO_CONFIG[FLAGS.algo]}",
            "--config.data.pretrain_action_class",
            f"({repr(embodiment)},)",
            "--config.data.downstream_action_class",
            f"({repr(embodiment)},)",
            "--config.data.max_vids_per_class",
            f"{MAX_DEMONSTRATIONS}",
        ],
        check=True,
    )

    # Note: This assumes that the config.root_dir value has not been
    # changed to its default value of 'tmp/xirl/pretrain_runs/'.
    exp_path = osp.join("/tmp/xirl/pretrain_runs/", experiment_name)

    # The 'goal_classifier' baseline does not need to compute a goal embedding.
    if FLAGS.algo != "goal_classifier":
      subprocess.run(
          [
              "python",
              "compute_goal_embedding.py",
              "--experiment_path",
              exp_path,
          ],
          check=True,
      )

    # Dump experiment metadata as yaml file.
    with open(osp.join(exp_path, "metadata.yaml"), "wb") as fp:
      yaml.dump(kwargs, fp)


if __name__ == "__main__":
  flags.mark_flag_as_required("algo")
  app.run(main)
