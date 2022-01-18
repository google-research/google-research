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

"""This file contains the utilities for saving videos."""
# pylint: disable=g-inconsistent-quotes
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

from absl import logging
import numpy as np


try:
  import cv2
except ImportError as e:
  print(e)


font = cv2.FONT_HERSHEY_SIMPLEX
bottom_left_corner_text = (0, 275)
bottom_left_corner_text_2 = (0, 290)
font_scale = 0.5
line_type = 1
colors_values = {'white': (255, 255, 255), 'green': (44, 235, 169),
                 'red': (255, 75, 75)}


def pad_image(img, num_pixel=20):
  """Pad images with black pixels."""
  img_shape = img.shape
  padding = np.zeros((num_pixel, img_shape[1], img_shape[-1]))
  return np.concatenate([img, padding], axis=0)


def add_text(img, text, color='white', char_per_line=35):
  """Add text to image.

  Args:
    img: the original image
    text: the text to be added
    color: the color of the added text
    char_per_line: maximum characters allowed on each row

  Returns:
    image with text added
  """
  font_color = colors_values[color]
  text_len = len(text)
  num_lines = math.ceil(text_len/float(char_per_line))
  text_segments = []
  for i in range(num_lines):
    text_segments.append(text[i*char_per_line:(i+1)*char_per_line])
  for i, t in enumerate(text_segments):
    location = (0, 275 + i * 15)
    cv2.putText(img, t, location, font, font_scale, font_color, line_type)
  return img


def write_video(frames, path, fourcc, frame_size, fps=30.0, is_color=True):
  """Writes a sequence of frames to a video file at the given path."""
  vid = cv2.VideoWriter(path, fourcc, fps, frame_size, is_color)
  print('frame count', frames.shape)
  # write frames and release
  print(vid.isOpened())
  for frame in frames:
    vid.write(frame[Ellipsis, ::-1])  # convert RGB to BGR for OpenCV
  print(vid)
  vid.release()
  print(vid.isOpened())
  del vid


def save_video_local(frames, video_path, fps=30.0):
  """Saves a container of frames (B, W, H, C) to an mp4 file.

  Args:
    frames: A container of uint8 frames (B, W, H, C) in [0, 255]
    video_path: Path to save the video to, must end in .mp4
    fps: The frame rate of saved video
  """
  frames = np.array(frames)
  print('video path: {}'.format(video_path))
  _, width, height, _ = frames.shape
  # FFV1 is a lossless codec. Switch to "avc1" for H264 if needed
  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  saved = False
  if not saved:
    write_video(frames, video_path, fourcc, (width, height), fps=fps)




def save_video(frames, video_path, fps=30.0):
  saved = False
  if not saved:
    save_video_local(frames, video_path, fps)

########################################################################
############################ Json utils ################################
########################################################################


def floatify_json(content):
  """"Convert content of a dictionary to python primitives for saving json."""
  if isinstance(content, dict):
    for k, v in content.items():
      content[k] = floatify_json(v)
    return content
  elif isinstance(content, list):
    for i, v in enumerate(list(content)):
      content[i] = floatify_json(v)
    return content
  else:
    return float(content)


def save_json(content, path):
  floatify_json(content)
  with gfile.GFile(path, mode='w') as f:
    json.dump(content, f)
