# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Save BERT output layer weights for inspection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Model checkpoint directory.")


def main(_):
  checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  reader = tf.train.NewCheckpointReader(checkpoint)
  tensor = reader.get_tensor("output_weights")
  print(type(tensor))
  np.save(FLAGS.checkpoint_dir + "output_weights.npy", tensor)


if __name__ == "__main__":
  app.run(main)
