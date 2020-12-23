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

"""ES-MAML Client."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl import app
from absl import flags
from absl import logging

import grpc
import numpy as np

import tensorflow.compat.v1 as tf
from es_maml import config as config_util

from es_maml.first_order import first_order_maml_learner_grpc
from es_maml.first_order import first_order_pb2_grpc

from es_maml.zero_order import zero_order_maml_learner_grpc
from es_maml.zero_order import zero_order_pb2_grpc

tf.disable_v2_behavior()

flags.DEFINE_string("server_address", "127.0.0.1", "The address of the server.")
flags.DEFINE_string("current_time_string", "NA",
                    "Current time string for naming logging folders.")

FLAGS = flags.FLAGS


def main(unused_argv):
  base_config = config_util.get_config()
  config = config_util.generate_config(
      base_config, current_time_string=FLAGS.current_time_string)
  blackbox_object = config.blackbox_object_fn()
  init_current_input = blackbox_object.get_initial()
  init_best_input = []
  init_best_core_hyperparameters = []
  init_best_value = -float("inf")
  init_iteration = 0
  np.random.seed(0)
  # ------------------ OPTIMIZERS ----------------------------------------------
  num_servers = config.num_servers
  logging.info("Number of Servers: %d", num_servers)

  if not config.run_locally:
    servers = [
        "{}.{}".format(i, FLAGS.server_address) for i in range(num_servers)
    ]
  else:
    servers = ["127.0.0.1:{}".format(20000 + i) for i in range(num_servers)]
  logging.info("Running servers:")
  logging.info(servers)
  stubs = []
  for server in servers:
    channel = grpc.insecure_channel(server)
    grpc.channel_ready_future(channel).result()
    if config.algorithm == "zero_order":
      stubs.append(zero_order_pb2_grpc.EvaluationStub(channel))
    elif config.algorithm == "first_order":
      stubs.append(first_order_pb2_grpc.EvaluationStub(channel))

  tf.gfile.MakeDirs(config.global_logfoldername)
  logging.info("LOGGING FOLDER: %s", config.global_logfoldername)
  tf.gfile.MakeDirs(config.test_mamlpt_parallel_vals_folder)
  if config.log_states:
    tf.gfile.MakeDirs(config.states_folder)
  if config.recording:
    tf.gfile.MakeDirs(config.video_folder)
  with tf.gfile.Open(config.hparams_file, "w") as hparams_file:
    json.dump(config.json_hparams, hparams_file)
  # Runs main client's procedure responsible for optimization.

  if config.algorithm == "zero_order":
    es_blackbox_optimizer = config.es_blackbox_optimizer_fn(
        blackbox_object.get_metaparams())
    zero_order_maml_learner_grpc.run_blackbox(
        config,
        es_blackbox_optimizer,
        init_current_input,
        init_best_input,
        init_best_core_hyperparameters,
        init_best_value,
        init_iteration,
        stubs=stubs,
        log_bool=True)

  elif config.algorithm == "first_order":
    train_tasks = {
        "object": blackbox_object,
        "tasks": [config.make_task_fn(t) for t in range(config.train_set_size)],
        "ids": range(config.train_set_size)
    }

    test_tasks = {
        "object":
            blackbox_object,
        "tasks": [
            config.make_task_fn(t)
            for t in range(config.train_set_size, config.train_set_size +
                           config.test_set_size)
        ],
        "ids":
            range(config.train_set_size,
                  config.train_set_size + config.test_set_size)
    }
    first_order_maml_learner_grpc.run_blackbox(config, train_tasks, test_tasks,
                                               init_current_input, stubs)


if __name__ == "__main__":
  app.run(main)
