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

"""Utilities for processing image observations."""

import io
import numpy as np
from PIL import Image


def shape_img(img, num_channels=3):
  """Reshape and flip a (possibly batched) flattened square image."""
  flat_shape = img.shape[-1]
  hw = int(np.sqrt(flat_shape / num_channels))
  shaped_img = img.reshape(-1, hw, hw, num_channels)
  shaped_img = shaped_img[:, ::-1, ::-1]
  if len(img.shape) == 1:
    shaped_img = np.squeeze(shaped_img, axis=0)
  return shaped_img


def img_to_uint(img):
  """Convert a float array into uint8."""
  return (255 * img).astype(np.uint8)


def array_to_shaped_uint(img):
  """Convert a flat, float numpy array into a shaped uint image."""
  return img_to_uint(shape_img(img))


def compress_image(img_obs):
  """Convert numpy array to PNG."""
  pil_img = Image.fromarray(img_obs)
  img_buf = io.BytesIO()
  pil_img.save(img_buf, format='PNG')
  img_bytes = img_buf.getvalue()
  return img_bytes


def decompress_image(img_bytes):
  """Convert PNG image to numpy array."""
  img_buf = io.BytesIO(img_bytes)
  pil_img = Image.open(img_buf)
  img_obs = np.array(pil_img)
  return img_obs


def draw_random_crop(max_border):
  left_margin = np.random.randint(2 * max_border + 1)
  right_margin = 2 * max_border - left_margin
  top_margin = np.random.randint(2 * max_border + 1)
  bottom_margin = 2 * max_border - top_margin
  return left_margin, right_margin, top_margin, bottom_margin


def apply_crop(
    img, pixels_to_pad, left_margin, right_margin, top_margin, bottom_margin):
  p = pixels_to_pad
  padding = [[p, p], [p, p], [0, 0]]
  img = np.pad(img, padding)
  bottom = img.shape[0] - bottom_margin
  right = img.shape[1] - right_margin
  img = img[top_margin:bottom, left_margin:right]
  return img


def random_crop_image(img, pixels_to_pad=4):
  """Pad with black pixels and draw a random crop, maintaining image dimensions.

  Args:
    img: a numpy array with image to crop.
    pixels_to_pad: size of the black border (and maximum crop translation).

  Returns:
    img: the transformed image.
  """
  orig_shape = img.shape
  left_margin, right_margin, top_margin, bottom_margin = (
      draw_random_crop(pixels_to_pad))
  img = apply_crop(
      img, pixels_to_pad, left_margin, right_margin, top_margin, bottom_margin)
  assert img.shape == orig_shape
  return img


def random_crop_image_pair(img1, img2, pixels_to_pad=4):
  """Apply the same random crop to two images."""
  orig_shape = img1.shape
  left_margin, right_margin, top_margin, bottom_margin = (
      draw_random_crop(pixels_to_pad))
  img1 = apply_crop(
      img1, pixels_to_pad, left_margin, right_margin, top_margin, bottom_margin)
  img2 = apply_crop(
      img2, pixels_to_pad, left_margin, right_margin, top_margin, bottom_margin)
  assert img1.shape == orig_shape
  assert img2.shape == orig_shape
  return img1, img2
