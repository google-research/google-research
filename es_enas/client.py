# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Client file for ES_ENAS."""
import json
import time

from absl import app
from absl import flags
from absl import logging

import ml_collections
import numpy as np
import tensorflow as tf

import es_enas.config as config_util
import es_enas.es_enas_learner_grpc as es_enas_learner_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_servers", 1, "Num. of servers for perturb. eval")
flags.DEFINE_bool("run_on_borg", False, "Are the servers are running on borg.")
flags.DEFINE_string("current_time_string", "NA", "current time string")
ml_collections.config_flags.DEFINE_config_file("config", "path/to/config",
                                               "Configuration.")


def setup_logging(config):
  logging.info("LOGGING FOLDER: %s", config.folder_name)
  tf.gfile.MakeDirs(config.folder_name)
  with tf.gfile.Open(config.hparams_file, "w") as hparams_file:
    json.dump(config.json_hparams, hparams_file)


def setup_workers():
  """Sets up connection to servers.

  To be implemented by the user.

  Returns:
    workers: List of GRPC/Worker connections.
  """
  raise NotImplementedError(
      "User should implement this function based on their own distributed setup."
  )


def main(_):
  base_config = FLAGS.config
  config = config_util.generate_config(
      base_config, current_time_string=FLAGS.current_time_string)
  assert config.num_workers == FLAGS.num_servers

  config.setup_controller_fn()

  blackbox_object = config.blackbox_object_fn()
  init_current_input = blackbox_object.get_initial()
  init_best_input = []
  init_best_core_hyperparameters = []
  init_best_value = -float("inf")
  init_iteration = 0

  es_blackbox_optimizer = config.es_blackbox_optimizer_fn(
      blackbox_object.get_metaparams())

  setup_logging(config)

  logging.info("Sleeping to make sure workers are online.")
  time.sleep(60)
  workers = setup_workers()
  np.random.seed(config.seed)
  es_enas_learner_grpc.run_optimization(
      config,
      es_blackbox_optimizer,
      init_current_input,
      init_best_input,
      init_best_core_hyperparameters,
      init_best_value,
      init_iteration,
      workers=workers,
      log_bool=True)


if __name__ == "__main__":
  app.run(main)
