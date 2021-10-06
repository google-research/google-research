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

"""Data loader for loading testing data."""

import collections
import os

import cv2 as cv

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile


def inference_data_loader(
    input_lr_dir,
    input_hr_dir=None,
    input_dir_len=-1,
):
  """Inference pipeline data loader."""
  filedir = input_lr_dir
  down_sp = False
  if (input_lr_dir is None) or (not gfile.exists(input_lr_dir)):
    if (input_hr_dir is None) or (not gfile.exists(input_hr_dir)):
      raise ValueError('Input directory not found')
    filedir = input_hr_dir
    down_sp = True

  image_list_lr_temp = gfile.listdir(filedir)
  image_list_lr_temp = [_ for _ in image_list_lr_temp if _.endswith('.png')]
  image_list_lr_temp = sorted(
      image_list_lr_temp
  )  # first sort according to abc, then sort according to 123
  image_list_lr_temp.sort(
      key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
  if input_dir_len > 0:
    image_list_lr_temp = image_list_lr_temp[:input_dir_len]

  image_list_lr = [os.path.join(filedir, _) for _ in image_list_lr_temp]

  # Read in and preprocess the images
  def preprocess_test(name):

    with tf.gfile.Open(name, 'rb') as fid:
      raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
      im = cv.imdecode(raw_im, cv.IMREAD_COLOR).astype(np.float32)[:, :, ::-1]

    if down_sp:
      icol_blur = cv.GaussianBlur(im, (0, 0), sigmaX=1.5)
      im = icol_blur[::4, ::4, ::]
    im = im / 255.0
    return im

  image_lr = [preprocess_test(_) for _ in image_list_lr]
  image_list_lr = image_list_lr[5:0:-1] + image_list_lr
  image_lr = image_lr[5:0:-1] + image_lr

  Data = collections.namedtuple('Data', 'paths_LR, inputs')
  return Data(paths_LR=image_list_lr, inputs=image_lr)
