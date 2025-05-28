# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Class definition for LHQ dataset."""
import cv2
import numpy as np
import torch
from torch.utils import data

scale_shift = [0.0, 0.8]  # scale and shift for normalized disparity value


class LHQTestDataset(data.Dataset):
  """LHQ Test Dataset."""

  def __init__(self, opt, data_list, phase):
    self.test_data = data_list
    self.opt = opt
    self.phase = phase

    self.use_dpt = opt.use_dpt

    if self.use_dpt:
      self.min_disp_t = 2.
    else:
      self.min_disp_t = 256.

    self.resize_w = opt.crop_size
    self.resize_h = opt.crop_size

  def scale_disp(self, disp_crop, disp_full):
    """Normalize cropped and original disparity to [0, 1]."""

    dmmin = np.percentile(disp_crop, 1)
    dmmax = np.percentile(disp_crop, 97)

    scaled_disp_crop = (disp_crop - dmmin) / (
        dmmax - dmmin + 1e-6) * scale_shift[1] - scale_shift[0]
    scaled_disp_crop = np.clip(scaled_disp_crop, 1e-2, 1.)

    scaled_disp = (disp_full - dmmin) / (dmmax - dmmin +
                                         1e-6) * scale_shift[1] - scale_shift[0]
    scaled_disp = np.clip(scaled_disp, 1e-2, 1.)

    return scaled_disp_crop, scaled_disp

  def read_rgbd(self, img_path):
    """Read required input data for running inference."""

    img = cv2.imread(img_path)
    img = np.float32(img[:, :, ::-1]) / 255.0
    disp_path = img_path.replace('images', 'midas-disp').replace('.png', '.npy')

    mono_disp = np.load(disp_path)

    mask_path = img_path.replace('images', 'sky_mask')

    labelmap = cv2.imread(mask_path)[:, :, 0]
    # 156 and 105 are labels for sky and clouds in DeepLabv2 trained on COCO.
    sky_mask = np.float32((labelmap == 156) | (labelmap == 105))

    x_offset = 0
    crop_size = (img.shape[0] - 256) // 2
    start_y = crop_size + x_offset
    start_x = crop_size

    img_crop = img[start_y:start_y + 256, start_x:start_x + 256, :]
    mono_disp_crop = mono_disp[start_y:start_y + 256, start_x:start_x + 256]

    scaled_disp_crop, scaled_disp = self.scale_disp(mono_disp_crop, mono_disp)

    img_crop = cv2.resize(
        img_crop, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
    scaled_disp_crop = cv2.resize(
        scaled_disp_crop, (self.resize_w, self.resize_h),
        interpolation=cv2.INTER_AREA)

    # Create intrinsics assuming 55 degree FoV and central principal point
    px, py = (img_crop.shape[1] - 1) / 2., (img_crop.shape[0] - 1) / 2.
    fov_in_degrees = 55.
    fx = fy = img_crop.shape[1] / (2. * np.tan(fov_in_degrees / 360. * np.pi))
    k_ref = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]],
                     dtype=np.float32)

    # full canvas
    canvas_size = (int(self.resize_w * img.shape[1] / 256),
                   int(self.resize_h * img.shape[0] / 256))
    img = cv2.resize(img, canvas_size, interpolation=cv2.INTER_AREA)
    scaled_disp = cv2.resize(
        scaled_disp, canvas_size, interpolation=cv2.INTER_AREA)
    sky_mask = cv2.resize(
        sky_mask, canvas_size, interpolation=cv2.INTER_NEAREST)

    k_full = k_ref.copy()
    k_full[0, 2] += (canvas_size[0] - self.resize_w) / 2.
    k_full[1, 2] += ((canvas_size[1] - self.resize_h) / 2. +
                     x_offset * img.shape[0] / 256.)

    real_rgbd_crop = np.concatenate([img_crop, scaled_disp_crop[Ellipsis, None]],
                                    axis=-1)
    real_rgbd_full = np.concatenate([img, scaled_disp[Ellipsis, None]], axis=-1)

    return {
        'real_rgbd_crop': real_rgbd_crop,
        'real_rgbd_full': real_rgbd_full,
        'sky_mask': sky_mask,
        'k_ref': k_ref,
        'k_full': k_full,
        'x_offset': x_offset
    }

  def __getitem__(self, index):
    """Loading LHQ data.

    Args:
      index: image index

    Returns:
      real_rgbd: central-cropped input RGBD image
      real_rgbd_full: outpainted input RGBD image
      sky_offset: disparity threshold to determine which region is sky
      x_offset: offset in x-axis while central crop the full image
      sky_ratio: percentage of sky content
      index: image index
      disp_gamma: whether input disparity is valid
      k_ref: intrinsic matrix of central-cropped image
      k_full: intrinsic matrix of outpainted full image
    """
    targets = {}

    gt_path_gan = self.test_data[index]

    ret_data = self.read_rgbd(gt_path_gan)

    real_rgbd_crop = ret_data['real_rgbd_crop']
    real_rgbd_full = ret_data['real_rgbd_full']
    sky_mask_full = ret_data['sky_mask']
    k_ref = ret_data['k_ref']
    k_full = ret_data['k_full']
    x_offset = ret_data['x_offset']

    disp_gamma = 1.

    real_disp_full = real_rgbd_full[Ellipsis, 3]

    sky_disp_candidate = real_disp_full[(real_disp_full < 0.08)
                                        & (sky_mask_full > 1e-3)]

    any_sky_pixel = sky_disp_candidate.shape[0] > 0
    if any_sky_pixel:
      sky_offset = np.percentile(sky_disp_candidate, 95)
      sky_ratio = np.mean(sky_disp_candidate.shape[0] /
                          (real_disp_full.shape[0] * real_disp_full.shape[1]))
    else:
      sky_offset = 0.
      sky_ratio = 0.

    targets['real_rgbd'] = torch.from_numpy(
        np.ascontiguousarray(real_rgbd_crop).transpose(2, 0,
                                                       1)).contiguous().float()
    targets['real_rgbd_full'] = torch.from_numpy(
        np.ascontiguousarray(real_rgbd_full).transpose(2, 0,
                                                       1)).contiguous().float()

    targets['sky_offset'] = sky_offset
    targets['x_offset'] = x_offset
    targets['sky_ratio'] = torch.from_numpy(
        np.ascontiguousarray(sky_ratio)).contiguous().float()

    targets['index'] = index
    targets['disp_gamma'] = disp_gamma
    targets['k_ref'] = k_ref
    targets['k_full'] = k_full

    return targets

  def __len__(self):
    return len(self.test_data)
