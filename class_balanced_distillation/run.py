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

"""Class-Balanced Distillation for Long-Tailed Visual Recognition, BMVC 2021.

Paper: https://arxiv.org/abs/2104.05279

Example usage:
    $ python -m class_balanced_distillation.run --config $CONFIG --workdir $DIR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from class_balanced_distillation import train
from ml_collections import config_flags
import sonnet as snt
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(_):

  gpus = tf.config.list_physical_devices("GPU")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  devices = tf.config.experimental.list_logical_devices(device_type="GPU")
  devices = [d.name for d in devices]
  if len(devices) == 1:
    strategy = tf.distribute.OneDeviceStrategy(devices[0])
  else:
    strategy = snt.distribute.Replicator(devices)

  train.train_and_evaluate(FLAGS.config, FLAGS.workdir, strategy=strategy)


if __name__ == "__main__":
  app.run(main)
