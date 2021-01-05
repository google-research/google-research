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

"""Evaluates a trained motion blur model on the real test dataset.

Learning to Synthesize Motion Blur
http://timothybrooks.com/tech/motion-blur
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from cvx2 import latest as cv2
import numpy as np
import skimage.measure
import tensorflow.compat.v1 as tf
from tensorflow.contrib import resampler as contrib_resampler

# Contrib is lazily loaded so this reference is needed to use the resampler op.
_RESAMPLER = contrib_resampler

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_dataset',
    None,
    'Location from which to load motion blur test dataset.')

flags.DEFINE_string(
    'model_output_dir',
    None,
    'Location from which to load output motion blurred images.')


def imread(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    raw_image = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image.astype(np.float32) / 255.0


def main(_):
  dir_names = tf.gfile.ListDirectory(FLAGS.test_dataset)
  dir_names = [os.path.join(FLAGS.test_dataset, name) for name in dir_names]

  psnr_sum = 0
  ssim_sum = 0
  for i, dir_name in enumerate(dir_names):
    label = imread(os.path.join(dir_name, 'blur.png'))
    output = imread(os.path.join(FLAGS.model_output_dir, '%02d_output.png' % i))

    psnr = skimage.measure.compare_psnr(label, output)
    ssim = skimage.measure.compare_ssim(label, output, multichannel=True)
    psnr_sum += psnr
    ssim_sum += ssim

    name = os.path.basename(dir_name)
    print('{}:\n  PSNR: {}\n  SSIM: {}'.format(name, psnr, ssim))

  print('Average PSNR: ' + str(psnr_sum / len(dir_names)))
  print('Average SSIM: ' + str(ssim_sum / len(dir_names)))


if __name__ == '__main__':
  app.run(main)
