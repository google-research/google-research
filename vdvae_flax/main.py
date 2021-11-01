# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited and the Google Research Authors.
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

"""Main file for running the example.

This file is intentionally kept short.
"""

from absl import app
from absl import flags

import jax
from ml_collections import config_flags
import tensorflow as tf

from vdvae_flax import experiment

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config", "workdir"])


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  exp = experiment.Experiment("train", FLAGS.config)
  exp.train_and_evaluate(FLAGS.workdir)


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
