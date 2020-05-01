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

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Model checkpoint directory.")

flags.DEFINE_string("tensor_names", "output_weights,new_output_weights",
                    "Comma separated list of tensor names to save.")


def save_tensor(reader, name):
  tensor = reader.get_tensor(name)
  np.save(os.path.join(FLAGS.checkpoint_dir, name + ".npy"), tensor)


def main(_):
  checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  reader = tf.train.NewCheckpointReader(checkpoint)
  for name in FLAGS.tensor_names.split(","):
    save_tensor(reader, name)


if __name__ == "__main__":
  app.run(main)
