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

"""Lightweighted utility library - also using lazy importing."""
import dataclasses
import os
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import tensorflow.compat.v2 as tf

from factors_of_influence.fids.fids_lazy_imports_lib import lazy_imports


def segmentation_set_background_label_to_zero(segmentation,
                                              old_background_label = 0
                                             ):
  """Sets new background (ignore) label in semantic segmentation map to zero.

  Args:
    segmentation: segmentation map
    old_background_label: current ignore (background) label

  Returns:
    updated segmentation map
  """
  segmentation = segmentation.astype(np.uint16)

  if old_background_label != 0:
    segmentation_ignore = (segmentation == old_background_label)
    segmentation[segmentation < old_background_label] += 1
    segmentation[segmentation_ignore] = 0

  return segmentation


def load_mat(full_path):
  """Load mat file from disk. Lazely using scipy.io."""
  with tf.io.gfile.GFile(full_path, 'rb') as f:
    return lazy_imports.scipy.io.loadmat(f)


def save_image(full_path, img_numpy):
  """Saves image to disk. Lazely using PIL Image."""
  img = lazy_imports.PIL_Image.fromarray(img_numpy)
  with tf.io.gfile.GFile(full_path, 'wb') as f:
    img.save(f)


def load_image(full_path):
  """Load image from disk. Lazely using PIL Image."""
  with tf.io.gfile.GFile(full_path, 'rb') as f:
    return np.array(lazy_imports.PIL_Image.open(f))


def load_png(full_path):
  """Load png. Lazely using PIL Image. Returns array with H x W x C."""
  assert full_path.endswith('.png')
  np_image = load_image(full_path)

  if len(np_image.shape) == 2:
    np_image = np_image[:, :, np.newaxis]

  return np_image


def load_image_cv2(full_path,
                   cv2_decode_flags = None,
                   cv2_convert_to_rgb = False):
  """Load image from dis. Lazely using OpenCV (v2)."""
  with tf.io.gfile.GFile(full_path, 'rb') as f:
    img = lazy_imports.cv2.imdecode(
        np.fromstring(f.read(), dtype=np.uint8), cv2_decode_flags)
  if cv2_convert_to_rgb and len(img.shape) == 3:
    img = lazy_imports.cv2.cvtColor(img, lazy_imports.cv2.COLOR_BGR2RGB)

  if len(img.shape) == 2:
    img = img[:, :, np.newaxis]

  return img


def load_image_cv2_any_color_any_depth(full_path):
  """Load image from disk with any color/any depth, using OpenCV (v2)."""
  decode_flags = lazy_imports.cv2.IMREAD_ANYCOLOR | lazy_imports.cv2.IMREAD_ANYDEPTH
  return load_image_cv2(full_path, cv2_decode_flags=decode_flags)


def resize_image_cv2(image,
                     desired_width,
                     desired_height,
                     method=None):
  """Resize image using OpenCV (v2) from lazy imports."""
  desired_shape = (desired_width, desired_height)
  method = method or lazy_imports.cv2.INTER_NEAREST
  return lazy_imports.cv2.resize(
      image, dsize=desired_shape, interpolation=method)


def load_text_to_list(full_path):
  with tf.io.gfile.GFile(full_path) as f:
    return [l.strip() for l in f]


@dataclasses.dataclass
class LabelColorDef:
  """Class to define label colors: RGB color, (desired) label id, and name."""
  name: str
  color: Tuple[int, int, int]
  id: int


@dataclasses.dataclass
class LabelMap:
  """Class to define a label mapping from original id to new id."""
  name: str
  id: int
  original_id: int


def convert_segmentation_map(
    segmentation_original,
    label_map):
  segmentation = np.zeros_like(segmentation_original, dtype=np.uint16)
  for label in label_map:
    segmentation[segmentation_original == label.original_id] = label.id
  return segmentation


def convert_segmentation_rgb_to_class_id(
    segmentation_rgb,
    label_list):
  """Loads and converts the color-coded annotations to class-id coded.

  Assumes that ignore label is defined in label_list.
  Pixels not assigned to any label will be assigned to class id 0.
  Args:
    segmentation_rgb: np.ndarray of segmentation or filename of segmentation
    label_list: list with possible labeldefinitions: a labelDef should have
      (label.name), label.id, label.color
  Returns:
    A segmentation image [W, H, 1], dtype=uint16.
  """

  if not isinstance(segmentation_rgb, np.ndarray):
    segmentation_rgb = load_image(segmentation_rgb)
  rgb_weights = np.asarray([1, 256, 256**2]).reshape((3, 1))
  segmentation_multiplied = np.matmul(segmentation_rgb, rgb_weights).squeeze()
  segmentation = np.zeros_like(segmentation_multiplied, dtype=np.uint16)
  for label in label_list:
    label_multiplied = np.dot(np.asarray(label.color), rgb_weights)
    segmentation[segmentation_multiplied == label_multiplied] = label.id

  return segmentation[:, :, np.newaxis]


class TSVFileExtractor:
  """Read Tab Separated Values file."""

  def __init__(self,
               filename,
               tab_separator = ' ',
               column = 0):
    self.filename = filename
    self.tab_separator = tab_separator
    self.column = column
    self.key_length = 0

  def tsv_file_exists(self):
    return tf.io.gfile.exists(self.filename)

  def set_filename_and_column(self, filename, column):
    self.filename = filename
    self.column = column

  def _get_key(self, data_path_without_ext):
    """Get key."""
    assert self.key_length >= 1
    return '/'.join(data_path_without_ext.split('/')[-self.key_length:])

  def _set_key_length(self, list_of_data_paths_without_ext):
    """Find length of key."""
    sample_key = list_of_data_paths_without_ext[0]
    max_splits = len(sample_key.split('/'))
    for key_length in range(1, max_splits + 1):
      all_keys = {}
      all_unique = True
      for data_path in list_of_data_paths_without_ext:
        key = '/'.join(data_path.split('/')[-key_length:])
        if key in all_keys:
          all_unique = False
          break
        all_keys[key] = True
      if all_unique:
        break
    print('KeyLength:', key_length)
    self.key_length = key_length

  def to_dict(self, root_dir = None):
    """Create dictionary which maps keys to filenames."""
    if root_dir is None:
      root_dir = ''
    with tf.io.gfile.GFile(self.filename, 'r') as f:
      data = [l.strip('\n').split(self.tab_separator) for l in f]
      if self.key_length == 0:
        data_paths_without_ext = [os.path.splitext(d[0])[0] for d in data]
        self._set_key_length(data_paths_without_ext)

      # Assumption 1. key is defined in column 0
      # Assumption 2. key is the last part of the filename pathA/pathB/key.ext
      #  which is unique, this could be set by _set_key_length.
      # Assumption 3. When key is defined multiple times, the last one is used.
      filedict = {}
      for l in data:
        path_without_ext = os.path.splitext(l[0])[0]
        key = self._get_key(path_without_ext)
        filedict[key] = os.path.join(root_dir, str(l[self.column]))

    return filedict
