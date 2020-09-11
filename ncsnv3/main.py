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

"""Training and evaluation for NCSNv3."""

from . import ncsn_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("mode", "train", "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
  del argv
  tf.config.experimental.set_visible_devices([], "GPU")
  if FLAGS.mode == "train":
    ncsn_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    ncsn_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")

if __name__ == "__main__":
  app.run(main)
