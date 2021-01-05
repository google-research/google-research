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

"""Unprocessing evaluation on the Darmstadt Noise Dataset.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing

This file denoises images from the Darmstadt Noise Dataset using the
unprocessing neural networks. The full Darmstadt code and data should be
downloaded from https://noise.visinf.tu-darmstadt.de/downloads and this file
should replace the dnd_denoise.py file that is provided.

This file is modified from the original version by Tobias Plotz, TU Darmstadt
(tobias.ploetz@visinf.tu-darmstadt.de), and is part of the implementation as
described in the CVPR 2017 paper: Benchmarking Denoising Algorithms with Real
Photographs, Tobias Plotz and Stefan Roth. Modified by Tim Brooks of Google in
2019. The original license is below.

Copyright (c) 2017, Technische Universitat Darmstadt
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Any redistribution, use, or modification is done solely for non-commercial
purposes. Examples of non-commercial uses are teaching, academic research,
public demonstrations and personal experimentation.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import h5py
import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_ckpt',
    None,
    'Path to checkpoint of a trained unprocessing model. For example: '
    '/path/to/models/unprocessing_srgb_loss/model.ckpt-3516383')

flags.DEFINE_string(
    'data_dir',
    None,
    'Location from which to load input noisy images. This should correspond '
    'with the \'data\' directory downloaded as part of the Darmstadt Noise '
    'Dataset.')

flags.DEFINE_string(
    'output_dir',
    None,
    'Location at which to save output denoised images.')


def denoise_raw(denoiser, data_dir, output_dir):
  """Denoises all bounding boxes in all raw images from the DND dataset.

  The resulting denoised images are saved to disk.

  Args:
    denoiser: Function handle called as:
        denoised_img = denoiser(noisy_img, shot_noise, read_noise).
    data_dir: Folder where the DND dataset resides
    output_dir: Folder where denoised output should be written to

  Returns:
    None
  """
  # Loads image information and bounding boxes.
  info = h5py.File(os.path.join(data_dir, 'info.mat'), 'r')['info']
  bb = info['boundingboxes']

  # Denoise each image.
  for i in range(50):
    # Loads the noisy image.
    filename = os.path.join(data_dir, 'images_raw', '%04d.mat' % (i + 1))
    img = h5py.File(filename, 'r')
    noisy = np.float32(np.array(img['Inoisy']).T)

    # Loads raw Bayer color pattern.
    bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()

    # Denoises each bounding box in this image.
    boxes = np.array(info[bb[0][i]]).T
    for k in range(20):
      # Crops the image to this bounding box.
      idx = [
          int(boxes[k, 0] - 1),
          int(boxes[k, 2]),
          int(boxes[k, 1] - 1),
          int(boxes[k, 3])
      ]
      noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()

      # Flips the raw image to ensure RGGB Bayer color pattern.
      if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
      elif (bayer_pattern == [[2, 1], [3, 2]]):
        noisy_crop = np.fliplr(noisy_crop)
      elif (bayer_pattern == [[2, 3], [1, 2]]):
        noisy_crop = np.flipud(noisy_crop)
      else:
        print('Warning: assuming unknown Bayer pattern is RGGB.')

      # Loads shot and read noise factors.
      nlf_h5 = info[info['nlf'][0][i]]
      shot_noise = nlf_h5['a'][0][0]
      read_noise = nlf_h5['b'][0][0]

      # Extracts each Bayer image plane.
      denoised_crop = noisy_crop.copy()
      height, width = noisy_crop.shape
      channels = []
      for yy in range(2):
        for xx in range(2):
          noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
          channels.append(noisy_crop_c)
      channels = np.stack(channels, axis=-1)

      # Denoises this crop of the image.
      output = denoiser(channels, shot_noise, read_noise)

      # Copies denoised results to output denoised array.
      for yy in range(2):
        for xx in range(2):
          denoised_crop[yy:height:2, xx:width:2] = output[:, :, 2 * yy + xx]

      # Flips denoised image back to original Bayer color pattern.
      if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
      elif (bayer_pattern == [[2, 1], [3, 2]]):
        denoised_crop = np.fliplr(denoised_crop)
      elif (bayer_pattern == [[2, 3], [1, 2]]):
        denoised_crop = np.flipud(denoised_crop)

      # Saves denoised image crop.
      denoised_crop = np.clip(np.float32(denoised_crop), 0.0, 1.0)
      save_file = os.path.join(output_dir, '%04d_%02d.mat' % (i + 1, k + 1))
      sio.savemat(save_file, {'denoised_crop': denoised_crop})


def main(_):
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as sess:
      saver = tf.train.import_meta_graph(FLAGS.model_ckpt + '.meta')
      saver.restore(sess, FLAGS.model_ckpt)

      def denoiser(noisy_img, shot_noise, read_noise):
        """Unprocessing denoiser."""
        denoised_img_tensor = graph.get_tensor_by_name('denoised_img:0')
        noisy_img_tensor = graph.get_tensor_by_name('noisy_img:0')
        shot_noise_tensor = graph.get_tensor_by_name('stddev/shot_noise:0')
        read_noise_tensor = graph.get_tensor_by_name('stddev/read_noise:0')
        feed_dict = {
            noisy_img_tensor: noisy_img[np.newaxis, :, :, :],
            shot_noise_tensor: np.asarray([shot_noise]),
            read_noise_tensor: np.asarray([read_noise])
        }
        return sess.run(denoised_img_tensor, feed_dict=feed_dict)[0, :, :, :]

      denoise_raw(denoiser, FLAGS.data_dir, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
