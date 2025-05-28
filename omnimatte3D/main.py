# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Main file for running the example."""

from absl import app
from absl import flags
from absl import logging

# Required import to setup work units when running through XManager.
from clu import platform
import flax
import jax
from ml_collections import config_flags
import tensorflow as tf

from omnimatte3D import eval_lib
from omnimatte3D import render_lib
from omnimatte3D import train_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_bool("is_train", None, "If true, run train else eval")
flags.DEFINE_bool(
    "is_render",
    None,
    "If true, run the rendering code, else run train or eval based on is_train",
)
flags.mark_flags_as_required(["config", "workdir"])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  # Currently used because orbax checkpointing produces broken checkpoints
  flax.config.update("flax_use_orbax_checkpointing", False)

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = (
        "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  if FLAGS.is_render:
    render_lib.evaluate(FLAGS.config, FLAGS.workdir)
    return
  if not FLAGS.is_train:
    eval_lib.evaluate(FLAGS.config, FLAGS.workdir)
    return

  # Training.
  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, "
      f"process_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
  )
  train_lib.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
