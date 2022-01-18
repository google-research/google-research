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

"""Creates fake cifar10 test data to be used in tf_cnn_benchmark tests.

Each image is a single color. There are 10 colors total, and each color appears
in the dataset 10 times, for a total of 100 images in the dataset. Each color
has a unique label. The complete dataset of 100 images is written to each of the
seven files data_batch_1 through data_batch_6 and test_batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import numpy as np

NAME_TO_RGB = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'gray': (128, 128, 128),
    'teal': (0, 128, 128)
}


COLORS = sorted(NAME_TO_RGB.keys())
NUM_COLORS = len(COLORS)
NUM_IMAGES_PER_COLOR = 10
NUM_BATCHES = NUM_COLORS * NUM_IMAGES_PER_COLOR
NUM_PIXELS_PER_IMAGE = 1024


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, required=True)
  args = parser.parse_args()

  data = np.empty((NUM_BATCHES, NUM_PIXELS_PER_IMAGE * 3), np.uint8)
  labels = []
  for i in range(NUM_BATCHES):
    color = COLORS[i % NUM_COLORS]
    red, green, blue = NAME_TO_RGB[color]
    data[i, 0:NUM_PIXELS_PER_IMAGE] = red
    data[i, NUM_PIXELS_PER_IMAGE:2 * NUM_PIXELS_PER_IMAGE] = green
    data[i, 2 * NUM_PIXELS_PER_IMAGE:] = blue
    labels.append(i % NUM_COLORS)
  d = {b'data': data, b'labels': labels}

  filenames = ['data_batch_%d' % i for i in range(1, 7)] + ['test_batch']
  for filename in filenames:
    with open(os.path.join(args.output_dir, filename), 'wb') as f:
      pickle.dump(d, f)


if __name__ == '__main__':
  main()
