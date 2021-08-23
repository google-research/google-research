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

import os
from typing import List, Union

import numpy as np
import cv2
from pathlib import Path


def get_subdirs(
    d: str,
    nonempty: bool = False,
    basename: bool = False,
    sort_lexicographical: bool = False,
    sort_numerical: bool = False,
):
  """Return a list of subdirectories in a given directory.

  Args:
    d: The path to the directory.
    nonempty: Only return non-empty subdirs.
    basename: Only return the tail of the subdir paths.
    sort_lexicographical: Lexicographical sort.
    sort_numerical: Numerical sort.

  Returns:
    The list of subdirectories.
  """
  # Note: `iterdir()` does not yield special entries '.' and '..'.
  subdirs = [f for f in Path(d).iterdir() if f.is_dir()]
  if nonempty:
    # Eliminate empty directories.
    subdirs = [f for f in subdirs if not check_dir_empty(f)]
  if sort_lexicographical:
    subdirs = sorted(subdirs, key=lambda x: x.stem)
  if sort_numerical:
    subdirs = sorted(subdirs, key=lambda x: int(x.stem))
  if basename:
    # Only return the directory stem.
    subdirs = [f.stem for f in subdirs]
  return [str(f) for f in subdirs]


def get_files(
    d: str,
    pattern: str,
    sort_lexicographical: bool = False,
    sort_numerical: bool = False,
) -> List[Path]:
  """Return a list of files in a given directory.

  Args:
    d: The path to the directory.
    pattern: The wildcard to filter files with.
    sort_lexicographical: Lexicographical sort.
    sort_numerical: Numerical sort.
  """
  files = Path(d).glob(pattern)
  if sort_lexicographical:
    return sorted(files, key=lambda x: x.stem)
  if sort_numerical:
    return sorted(files, key=lambda x: int(x.stem))
  return list(files)


def check_dir_empty(path: Union[str, Path]) -> bool:
  """Return True if a directory is empty."""
  with os.scandir(path) as it:
    return not any(it)


def load_image(filename: Union[str, Path]) -> np.ndarray:
  """Loads an image as a numpy array.

  Args:
    filename: The name of the image file.

  Returns:
    A numpy uint8 array.
  """
  return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
