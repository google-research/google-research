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

"""Main function."""

# pylint:disable=g-bad-import-order
# pylint:disable=unused-import
from absl import flags
from absl import logging
import tensorflow as tf
import gin
from absl import app
from smurf import smurf_flags
from smurf import smurf_trainer

FLAGS = flags.FLAGS


def main(unused_argv):
  if FLAGS.virtual_gpus > 1:
    smurf_trainer.set_virtual_gpus_to_at_least(FLAGS.virtual_gpus)

  if FLAGS.no_tf_function:
    tf.config.experimental_run_functions_eagerly(True)
    logging.info('TFFUNCTION DISABLED')

  logging.info('Parsing gin flags...')
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)

  smurf_trainer.train_eval()

if __name__ == '__main__':
  app.run(main)
