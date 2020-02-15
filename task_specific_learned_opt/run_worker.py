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

"""Run the worker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

from absl import flags
from absl import logging
import setup_experiment
import tensorflow.compat.v1 as tf
import truncated_training

nest = tf.contrib.framework.nest

FLAGS = flags.FLAGS

flags.DEFINE_string("init_from_dir", None,
                    "Training directory to save summaries/checkpoints.")

flags.DEFINE_string("master", "local", "Tensorflow distributed target")

flags.DEFINE_integer("ps_tasks", 0, "number of parameter server tasks")

flags.DEFINE_integer("worker_tasks", 1, "number of parameter server tasks")

flags.DEFINE_integer("tf_seed", None, "tensorflow seed")


def main(_):
  train_dir = setup_experiment.setup_experiment("worker%d_logs" % FLAGS.task)

  run_fn = truncated_training.worker_run

  while True:
    logging.info("Starting a new graph reset iteration")
    g = tf.Graph()
    with g.as_default():
      np_global_step = 0

      tf.set_random_seed(FLAGS.tf_seed)
      logging.info("building graph... with state from %d", np_global_step)
      stime = time.time()
      graph_dict = truncated_training.build_graph(np_global_step=np_global_step)
      logging.info("done building graph... (took %f sec)", time.time() - stime)

      # perform a series of unrolls
      run_fn(train_dir, graph_dict)

      ## reset the graph
      if FLAGS.master:
        logging.info("running tf.session.Reset")
        config = tf.ConfigProto(
            device_filters=["/job:worker/replica:%d" % FLAGS.task])
        tf.Session.reset(FLAGS.master, config=config)
      if FLAGS.master == "local":
        logging.info("running tf.session.Reset")
        config = tf.ConfigProto()
        tf.Session.reset(FLAGS.master, config=config)


if __name__ == "__main__":
  tf.app.run(main)
