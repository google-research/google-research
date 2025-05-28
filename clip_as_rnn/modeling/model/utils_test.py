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

"""This file contains the unit tests for the utils.py file."""

import numpy as np
from PIL import Image
import torch

# pylint: disable=g-bad-import-order
from modeling.model import utils


def test_scoremap2bbox():
  """Test the scoremap2bbox function."""
  scoremap = np.zeros((10, 10))
  scoremap[1:5, 1:5] = 1
  scoremap[5:9, 5:9] = 2
  scoremap[5:9, 1:5] = 3
  scoremap[1:5, 5:9] = 4
  bbox, len_bboxes = utils.scoremap2bbox(scoremap, 0.5)
  assert len_bboxes == 1
  assert bbox[0, 0] == 1
  assert bbox[0, 1] == 1
  assert bbox[0, 2] == 9
  assert bbox[0, 3] == 9


def test_mask2chw():
  """Test the mask2chw function."""
  mask = np.zeros((10, 10))
  mask[1:5, 1:5] = 1
  mask[5:9, 5:9] = 2
  mask[5:9, 1:5] = 3
  mask[1:5, 5:9] = 4
  mask = torch.tensor(mask)
  mask_center, mask_height, mask_width = utils.mask2chw(mask)
  assert len(mask_center) == 2
  assert mask_center[0] == 2
  assert mask_center[1] == 2
  assert mask_height == 4
  assert mask_width == 4


def test_unpad():
  """Test the unpad function."""
  image = np.zeros((10, 10, 1))
  image[1:5, 1:5] = 1
  image[5:9, 5:9] = 2
  image[5:9, 1:5] = 3
  image[1:5, 5:9] = 4
  unpad_image = utils.unpad(image, pad=(1, 1, 8, 8))
  assert len(unpad_image[0]) == 8, 'The width of the image is not 8.'
  assert len(unpad_image[1]) == 8, 'The height of the image is not 8.'
  unpad_image = utils.unpad(image, None)
  assert (unpad_image == image).sum() == 100


def test_apply_visual_prompts():
  """Test the apply_visual_prompts function."""
  image = np.ones((5, 5))
  mask = np.array([
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1.0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
  ])

  target = np.array([
      [1, 1, 255, 1, 1],
      [1, 255, 1, 255, 1],
      [255, 1, 1, 1, 255],
      [1, 255, 1, 255, 1],
      [1, 1, 255, 1, 1],
  ])
  mask[1:5, 1:5] = 1
  prompted_image = utils.apply_visual_prompts(
      image, mask, visual_prompt_type='circle', thickness=1
  )
  prompted_array = np.array(prompted_image)
  assert (prompted_array == target).sum() == 25


def test_reshape_transform():
  """Test the reshape_transform function."""
  image = torch.zeros((101, 10, 32))
  image = utils.reshape_transform(image, height=10, width=10)
  b, c, h, w = image.shape
  assert b == 10
  assert c == 32
  assert h == 10
  assert w == 10


def test_img_ms_and_flip():
  """Test the img_ms_and_flip function."""
  image = np.zeros((120, 150))
  image[1:5, 1:5] = 1
  image[5:9, 5:9] = 2
  image[5:9, 1:5] = 3
  image[1:5, 5:9] = 4
  image = Image.fromarray(image)
  image = utils.img_ms_and_flip(image, 120, 150, scales=[1.2], patch_size=16)
  image = image[0]
  h, w = image.shape[-2:]
  assert h == int(np.ceil(1.2 * 120 / 16) * 16)
  assert w == int(np.ceil(1.2 * 150 / 16) * 16)


if __name__ == '__main__':
  test_scoremap2bbox()
  test_mask2chw()
  test_unpad()
  test_apply_visual_prompts()
  test_reshape_transform()
  test_img_ms_and_flip()
