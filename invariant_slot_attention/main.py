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

"""Main file for running the model trainer."""

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from ml_collections import config_flags

import tensorflow as tf


from invariant_slot_attention.lib import trainer

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Config file.")
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("jax_backend_target", None, "JAX backend target to use.")
flags.mark_flags_as_required(["config", "workdir"])


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax.config.update("jax_xla_backend", "tpu_driver")
    jax.config.update("jax_backend_target", FLAGS.jax_backend_target)

  logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
  logging.info("JAX devices: %r", jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  platform.work_unit().set_task_status(
      f"host_id: {jax.host_id()}, host_count: {jax.host_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  trainer.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  app.run(main)
