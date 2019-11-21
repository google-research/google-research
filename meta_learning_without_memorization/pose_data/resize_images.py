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

# Lint as: python3
"""Resize images to 128 x 128.

Usage instructions:
    run the following:
    cp -r rotate/* rotate_resized/
    cd rotate_resized/
    python resize_images.py --data_dir=rotate_resized
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import multiprocessing
import os
from absl import app
from absl import flags
from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None,
                    'Root directory where images are stored.')


def resize(image_file):
  im = Image.open(image_file)
  im = im.resize((128, 128), resample=Image.LANCZOS)
  im = im.convert('L')
  im.save(image_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  all_images = glob.glob(os.path.join(FLAGS.data_dir, '*/*/*.png'))

  p = multiprocessing.Pool(10)
  p.map(resize, all_images)

if __name__ == '__main__':
  app.run(main)
