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

# Lint as: python3
"""Concate all of the images in a directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import imageio
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string('images_dir', None,
                    'Directory where summaries are located.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  images = []
  for file in sorted([int(x[:-4]) for x in tf.io.gfile.listdir(FLAGS.images_dir)
                      if x.endswith('.npz')]):
    with tf.gfile.Open(
        os.path.join(FLAGS.images_dir, '%d.npz' % file), 'rb') as f:
      image = np.load(f)
      images.append(image)
  image = np.concatenate(images, axis=0)
  with tf.gfile.Open(os.path.join(FLAGS.images_dir,
                                  'samples.png'), 'w') as out:
    imageio.imwrite(out, image, format='png')


if __name__ == '__main__':
  app.run(main)
