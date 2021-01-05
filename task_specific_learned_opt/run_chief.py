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

"""Run the chief worker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
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
  train_dir = setup_experiment.setup_experiment("chief_logs")
  graph_dict = truncated_training.build_graph()
  truncated_training.chief_run(train_dir, graph_dict)


if __name__ == "__main__":
  tf.app.run(main)
