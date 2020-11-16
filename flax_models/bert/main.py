# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Main file for pre-training or fine-tuning BERT."""

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

from flax_models.bert import run_classifier  # pylint: disable=fine-too-long CHANGEME
from flax_models.bert import run_pretraining  # pylint: disable=fine-too-long CHANGEME

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


  train_mode = FLAGS.config.mode
  train_lib = run_classifier
  if train_mode == "pretraining":
    train_lib = run_pretraining
  elif train_mode != "classification":
    logging.warning("Unknown mode: %s -- running classification", train_mode)

  train_lib.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  app.run(main)
