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

"""Server from classic Blackbox to drive ES-MAML."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from concurrent import futures
import time
from absl import app
from absl import flags
from absl import logging

import grpc
import numpy as np

import tensorflow.compat.v1 as tf
from es_maml import blackbox_maml_objects
from es_maml import config as config_util
from es_maml.first_order import first_order_pb2_grpc
from es_maml.zero_order import zero_order_pb2_grpc

tf.disable_v2_behavior()

# Server Setup
flags.DEFINE_integer("server_id", 0, "The id of the server.")
flags.DEFINE_integer("port", 20000, "Port number.")
flags.DEFINE_string("current_time_string", "NA",
                    "Current time string for naming logging folders.")

FLAGS = flags.FLAGS

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def main(unused_argv):
  base_config = config_util.get_config()
  config = config_util.generate_config(
      base_config, current_time_string=FLAGS.current_time_string)

  port = FLAGS.port
  if config.run_locally:
    port = 20000 + FLAGS.server_id
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

  blackbox_object = config.blackbox_object_fn()
  np.random.seed(FLAGS.server_id)

  if config.algorithm == "zero_order":
    if FLAGS.server_id < config.test_workers:
      worker_mode = "Test"
      task_ids = range(config.train_set_size,
                       config.train_set_size + config.test_set_size)

    else:
      worker_mode = "Train"
      task_ids = range(config.train_set_size)

    servicer = blackbox_maml_objects.GeneralMAMLBlackboxWorker(
        worker_id=FLAGS.server_id,
        blackbox_object=blackbox_object,
        task_ids=task_ids,
        task_batch_size=config.task_batch_size,
        worker_mode=worker_mode)
    zero_order_pb2_grpc.add_EvaluationServicer_to_server(servicer, server)

  elif config.algorithm == "first_order":
    tasks = [
        config.make_task_fn(s)
        for s in range(config.train_set_size + config.test_set_size)
    ]
    servicer = blackbox_maml_objects.GradientMAMLWorker(
        FLAGS.server_id, blackbox_object=blackbox_object, tasks=tasks)
    first_order_pb2_grpc.add_EvaluationServicer_to_server(servicer, server)

  server.add_insecure_port("[::]:{}".format(port))
  server.start()
  logging.info("Start server %d", FLAGS.server_id)

  # prevent the main thread from exiting
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == "__main__":
  app.run(main)
