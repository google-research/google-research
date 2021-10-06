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

"""Metrics for eval."""

import os

from absl import flags
import numpy as np
from skimage.measure import compare_ssim

import tensorflow.compat.v1.io.gfile as gfile

FLAGS = flags.FLAGS


def list_png_in_dir(dirpath):
  """List all directoties under dirpath."""
  filelist = gfile.listdir(dirpath)
  filelist = [_ for _ in filelist if _.endswith('.png')]
  filelist = [_ for _ in filelist if not _.startswith('IB')]
  filelist = sorted(filelist)
  filelist.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
  result = [os.path.join(dirpath, _) for _ in filelist if _.endswith('.png')]
  return result


def rgb_to_ycbcr(img, max_val=255):
  """color space transform, from https://github.com/yhjo09/VSR-DUF."""
  o = np.array([[16], [128], [128]])
  trans = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                    [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                    [0.439215686274510, -0.367788235294118,
                     -0.071427450980392]])

  if max_val == 1:
    o = o / 255.0

  t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
  t = np.dot(t, np.transpose(trans))
  t[:, 0] += o[0]
  t[:, 1] += o[1]
  t[:, 2] += o[2]
  ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

  return ycbcr


def to_uint8(x, vmin, vmax):
  # color space transform, originally from https://github.com/yhjo09/VSR-DUF
  x = x.astype('float32')
  x = (x - vmin) / (vmax - vmin) * 255  # 0~255
  return np.clip(np.round(x), 0, 255)


def psnr(img_true, img_pred, y_channel=True):
  """PSNR with color space transform, originally."""
  if y_channel:
    y_true = rgb_to_ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    y_pred = rgb_to_ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
  else:
    y_true = to_uint8(img_true, 0, 255)
    y_pred = to_uint8(img_pred, 0, 255)
  diff = y_true - y_pred
  rmse = np.sqrt(np.mean(np.power(diff, 2)))
  return 20 * np.log10(255. / rmse)


def ssim(img_true, img_pred, y_channel=True):
  """SSIM with color space transform."""
  if y_channel:
    y_true = rgb_to_ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    y_pred = rgb_to_ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
  else:
    y_true = to_uint8(img_true, 0, 255)
    y_pred = to_uint8(img_pred, 0, 255)
  return compare_ssim(
      y_true,
      y_pred,
      data_range=y_pred.max() - y_pred.min(),
      multichannel=not y_channel)


def crop_8x8(img):
  """Crop 8x8 of the input image."""
  ori_h = img.shape[0]
  ori_w = img.shape[1]

  h = (ori_h // 32) * 32
  w = (ori_w // 32) * 32

  while h > ori_h - 16:
    h = h - 32
  while w > ori_w - 16:
    w = w - 32

  y = (ori_h - h) // 2
  x = (ori_w - w) // 2
  crop_img = img[y:y + h, x:x + w]
  return crop_img, y, x
