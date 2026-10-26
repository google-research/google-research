# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Filesystem helpers that work with or without TensorFlow installed.

Uses `tensorflow.io.gfile` when available (supporting remote filesystems such
as GCS), and otherwise falls back to the Python standard library for local
paths.
"""

from collections.abc import Sequence
import glob as _glob
import os
from typing import IO, Any

try:
  from tensorflow.io import gfile as _gfile  # pylint: disable=g-import-not-at-top
except ImportError:
  _gfile = None


def glob(pattern):
  """Returns a list of paths matching `pattern`."""
  if _gfile is not None:
    return _gfile.glob(pattern)
  return _glob.glob(pattern)


def exists(path):
  """Returns whether `path` exists."""
  if _gfile is not None:
    return _gfile.exists(path)
  return os.path.exists(path)


def open_file(path, mode = 'r'):
  """Opens `path` with the given `mode` and returns a file object."""
  if _gfile is not None:
    return _gfile.GFile(path, mode)
  return open(path, mode)


def makedirs(path):
  """Recursively creates `path`, succeeding if it already exists."""
  if _gfile is not None:
    _gfile.makedirs(path)
  else:
    os.makedirs(path, exist_ok=True)


def remove(path):
  """Deletes the file at `path`."""
  if _gfile is not None:
    _gfile.remove(path)
  else:
    os.remove(path)


def listdir(path):
  """Returns the names of the entries in directory `path`."""
  if _gfile is not None:
    return _gfile.listdir(path)
  return os.listdir(path)
