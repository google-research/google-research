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

"""Setup the default boilerplate for experiments.

This includes parsing gin bindings, and some helpful logging.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import gin

flags.DEFINE_string("train_log_dir", None,
                    "Training directory to save summaries/checkpoints.")

flags.DEFINE_integer("task", 0, "Task of the evaluation job.")

flags.DEFINE_multi_string("config_file", None,
                          "List of paths to the config files.")

flags.DEFINE_multi_string("gin_bindings", None,
                          "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


def setup_experiment(log_name):  # pylint: disable=unused-argument
  """Setup experiment."""
  # hack to ensure that strings starting with @ or % are parsed as configurable
  if FLAGS.gin_bindings:
    for i, g in enumerate(FLAGS.gin_bindings):
      key, value = g.split("=")
      new_v = value.strip()
      if new_v[0:2] in ["\"@"]:
        new_v = new_v[1:-1]  # strip quotes
      FLAGS.gin_bindings[i] = key.strip() + "=" + new_v

  if FLAGS.config_file and FLAGS.config_file[0] == "/":
    config_file = [FLAGS.config_file]
  else:
    config_file = FLAGS.config_file

  gin.parse_config_files_and_bindings(
      config_file,
      FLAGS.gin_bindings,
      finalize_config=False,
      skip_unknown=True,
  )

  if FLAGS.train_log_dir:
    logging.info("Setup experiment! Training directory located: %s",
                 FLAGS.train_log_dir)

    return FLAGS.train_log_dir
  else:
    logging.info("Setup experiment! No training directory specified")
    return None
