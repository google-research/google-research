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

"""Removes LHQ images unsuitable for flying."""
import os

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import midas

lhq = 'dataset/LHQ'  # source directory
# assumes source directory contains:
# 1) directory lhq_256 of LHQ images at size 256x256
# 2) directory dpt_depth of DPT depth result on lhq_256 (default setting)
# 2) directory dpt_seg of DPT segmentation result on lhq_256 (default setting)
lhq = os.path.abspath(lhq)  # get abspath for symlink
output_dir = 'dataset/lhq_processed'
os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'dpt_depth'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'dpt_depth-vis'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'dpt_sky'), exist_ok=True)
os.makedirs(os.path.join(lhq, 'dpt_sky_output'), exist_ok=True)

counter = 0
total = 90000

for c, i in enumerate(tqdm(range(total))):  # 90000))):
  path = os.path.join(lhq, 'dpt_depth', '%07d.pfm' % i)
  disp, scale = midas.read_pfm(path)

  # normalize the disparity
  max_disp = np.max(disp)
  min_disp = np.min(disp)
  disp_norm = disp / np.max(disp)

  seg = np.array(Image.open(os.path.join(lhq, 'dpt_seg', '%07d_seg.png' % i)))
  ### remove small contours
  # sky_mask is 0 in the sky region
  sky_mask = 1 - (seg == 3).astype(np.uint8)
  contours, hierarchy = cv.findContours(sky_mask * 255, 1, 2)
  processing_mask = np.ones_like(sky_mask)
  # processing mask is zero for small contours; one otherwise
  for j, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    if area < 250:
      cv.drawContours(processing_mask, contours, j, (0, 0, 0), cv.FILLED)
  # zero out small contours, and save the intermediate result
  sky_mask_processed = processing_mask * sky_mask
  np.savez(
      os.path.join(lhq, 'dpt_sky_output', '%07d.npz' % i),
      sky_mask=sky_mask_processed,
  )
  contours_processed, _ = cv.findContours(sky_mask_processed * 255, 1, 2)

  keep_img = True
  # too many contours --> trees
  if len(contours_processed) > 3:
    keep_img = False
  # check sky region --> too much non-sky region
  if np.mean(sky_mask_processed) > 0.9:
    keep_img = False
  # check upper part of scene
  h, w = sky_mask_processed.shape
  upper = sky_mask_processed[: h // 5, :]
  if np.mean(upper) > 0.4:
    # too much foreground in upper part of image
    keep_img = False
  # check lower part of scene
  lower = sky_mask_processed[-h // 4 :, :]
  if np.mean(lower) < 0.8:
    # too much sky in lower part of image
    keep_img = False
  # check not too many vertical edges
  if np.percentile(np.abs(disp_norm[:, 1:] - disp_norm[:, :-1]), 99) > 0.05:
    keep_img = False

  if keep_img:
    cmd = 'ln -s %s %s' % (
        os.path.join(lhq, 'dpt_depth', '%07d.pfm' % i),
        os.path.join(output_dir, 'dpt_depth'),
    )
    os.system(cmd)
    cmd = 'ln -s %s %s' % (
        os.path.join(lhq, 'lhq_256', '%07d.png' % i),
        os.path.join(output_dir, 'img'),
    )
    os.system(cmd)
    cmd = 'ln -s %s %s' % (
        os.path.join(lhq, 'dpt_sky_output', '%07d.npz' % i),
        os.path.join(output_dir, 'dpt_sky'),
    )
    os.system(cmd)
    counter += 1
    if c < 5000:
      disp_out = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
      im = Image.fromarray((disp_out * 255).astype(np.uint8))
      im.save(os.path.join(output_dir, 'dpt_depth-vis', '%07d.png' % i))

print('kept %d images %0.2f' % (counter, counter / total))
