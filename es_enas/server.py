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

"""Server file for ES_ENAS."""
import time

from absl import app
from absl import flags
from absl import logging

import ml_collections

from es_enas import workers
import es_enas.config as config_util

FLAGS = flags.FLAGS

# Server stuff
flags.DEFINE_integer("port", 20000, "Port Number.")
flags.DEFINE_integer("server_id", 0, "The id of the server.")
flags.DEFINE_bool("run_on_borg", False, "Are the servers running on borg.")
ml_collections.config_flags.DEFINE_config_file("config", "path/to/config",
                                               "Configuration.")

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Server(object):
  """Skeleton for Server API to accept requests from the aggregator."""

  def __init__(self, port):
    self.port = port

  def Bind(self, new_function_string, function):
    """Binds the objective function to the Server API."""
    raise NotImplementedError(
        "User should implement this function based on their own distributed setup."
    )

  def Start(self):
    """Starts the Server."""
    raise NotImplementedError(
        "User should implement this function based on their own distributed setup."
    )


def main(unused_argv):
  base_config = FLAGS.config
  config = config_util.generate_config(
      base_config, current_time_string=FLAGS.current_time_string)
  blackbox_object = config.blackbox_object_fn()

  servicer = workers.GeneralTopologyBlackboxWorker(
      FLAGS.server_id, blackbox_object=blackbox_object)

  server = Server(port=FLAGS.port)

  server.Bind("EvaluateBlackboxInput", servicer.EvaluateBlackboxInput)
  server.Start()
  logging.info("Start server %d", FLAGS.server_id)

  # prevent the main thread from exiting
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.Stop()


if __name__ == "__main__":
  app.run(main)
