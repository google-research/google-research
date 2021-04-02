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

"""File utilities."""

from glob import glob  # pylint: disable=g-importing-member
import os
import os.path as osp
from typing import Callable, List, Optional, Tuple, cast

import numpy as np
from PIL import Image


def get_subdirs(
    d,
    nonempty = False,
    sort = True,
    basename = False,
    sortfunc = None,  # pylint: disable=g-bare-generic
):
  """Return a list of subdirectories in a given directory.

  Args:
    d: The path to the directory.
    nonempty: Only return non-empty subdirs.
    sort: Whether to sort in lexicographical order.
    basename: Only return the tail of the subdir paths.
    sortfunc : An optional sorting Callable to override.

  Returns:
    The list of subdirectories.
  """
  subdirs = [
      cast(str, f.path) for f in os.scandir(d) if f.is_dir()
      if not f.name.startswith(".")
  ]
  if nonempty:
    subdirs = [f for f in subdirs if not is_folder_empty(f)]
  if sort:
    if sortfunc is None:
      subdirs.sort(key=lambda x: osp.basename(x))  # pylint: disable=unnecessary-lambda
    else:
      subdirs.sort(key=sortfunc)
  if basename:
    return [osp.basename(x) for x in subdirs]
  return subdirs


def get_files(
    d,
    pattern,
    sort = False,
    basename = False,
    sortfunc = None,  # pylint: disable=g-bare-generic
):
  """Return a list of files in a given directory.

  Args:
    d: The path to the directory.
    pattern: The wildcard to filter files with.
    sort: Whether to sort in lexicographical order.
    basename: Only return the tail of the subdir paths.
    sortfunc : An optional sorting Callable to override.

  Returns:
    The files in the directory.
  """
  files = glob(osp.join(d, pattern))
  files = [f for f in files if osp.isfile(f)]
  if sort:
    if sortfunc is None:
      files.sort(key=lambda x: osp.basename(x))  # pylint: disable=unnecessary-lambda
    else:
      files.sort(key=sortfunc)
  if basename:
    return [osp.basename(x) for x in files]
  return files


def is_folder_empty(d):
  """A folder is not empty if it contains >=1 non hidden files."""
  return len(glob(osp.join(d, "*"))) == 0  # pylint: disable=g-explicit-length-test


def load_image(
    filename,
    resize = None,
):
  """Loads an image as a numpy array.

  Args:
    filename: The name of the image file.
    resize: The height and width of the loaded image. Set to `None` to keep
      original image dims.

  Returns:
    A numpy uint8 array.
  """
  img = Image.open(filename)
  if resize is not None:
    # PIL expects a (width, height) tuple.
    img = img.resize((resize[1], resize[0]))
  return np.asarray(img)
