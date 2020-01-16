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

r"""Load notMNIST and covert it to numpy arrays.

Download notMNIST dataset from
http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz and unzip the file


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
from PIL import Image
import tensorflow as tf

flags.DEFINE_string('out_dir', '/tmp/image_data', 'Directory to save datasets.')
flags.DEFINE_string('raw_data_dir', '/tmp/notMNIST_small',
                    'Directory to raw data')

FLAGS = flags.FLAGS


def load_non_mnist(raw_data_dir):
  """load not_mnist raw data and save to numpy arrays."""
  max_count = 0
  for (root, _, files) in os.walk(raw_data_dir):
    for f in files:
      if f.endswith('.png'):
        max_count += 1
  print('Found %s files' % (max_count,))

  images_np = np.zeros((max_count, 28, 28), dtype=np.int32)
  labels_np = np.zeros((max_count,), dtype=np.int32)
  count = 0
  for (root, _, files) in os.walk(raw_data_dir):
    for f in files:
      if f.endswith('.png'):
        try:
          img = Image.open(os.path.join(root, f))
          images_np[count, :, :] = np.asarray(img)
          surround_folder = os.path.split(root)[-1]
          assert len(surround_folder) == 1
          labels_np[count] = ord(surround_folder) - ord('A')
          count += 1
        except OSError:
          pass
  print('All non-mnist loaded.')
  return images_np, labels_np


def main(unused_argv):
  images_np, labels_np = load_non_mnist(FLAGS.raw_data_dir)

  with tf.compat.v1.gfile.Open(
      os.path.join(FLAGS.out_dir, 'notmnist.npy'), 'wb') as f:
    np.save(f, np.expand_dims(images_np, axis=3))
    np.save(f, labels_np)
  print('Saved np arrays to %s' % FLAGS.out_dir)


if __name__ == '__main__':
  app.run(main)
