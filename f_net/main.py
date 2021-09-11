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

"""Main file for pre-training or fine-tuning models."""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

from f_net import run_classifier
from f_net import run_pretraining
from f_net.configs.base import TrainingMode

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.", required=True)
flags.DEFINE_string(
    "vocab_filepath",
    None,
    "Absolute path to SentencePiece vocab model.",
    required=True)

FLAGS = flags.FLAGS


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  # Add a note so that we can tell which task is which JAX process.
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, process_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  train_mode = FLAGS.config.mode
  if train_mode == TrainingMode.PRETRAINING:
    train_lib = run_pretraining
  elif train_mode == TrainingMode.CLASSIFICATION:
    train_lib = run_classifier
  else:
    raise ValueError("Unknown training mode: %s" % train_mode)

  train_lib.train_and_evaluate(FLAGS.config, FLAGS.workdir,
                               FLAGS.vocab_filepath)


if __name__ == "__main__":
  app.run(main)
