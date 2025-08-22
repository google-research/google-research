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

"""Common utils."""

import contextlib
import pickle as builtin_pickle
from typing import Any

from absl import logging
import jax
import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import


def report_memory():
  """Prints the size and number of live buffers tracked by the backend."""
  live_buffers = jax.live_arrays()
  total = 0
  for buf in live_buffers:
    total += buf.size
  logging.info(
      "num_buffers: %d | total_elements: %.2e", len(live_buffers), total)


def canonicalize_tensor_name(name):
  """Canonicalizes tensor names.

  For an op that produces only one output, we may be refer to its output tensor
  as either "op_name:0" or simply "op_name". This standardizes all internal
  names as "op_name:0" to simplify the logic.

  Args:
    name: Input name to canonicalize.

  Returns:
    The canonicalized input name.
  """
  if ":" not in name:
    return f"{name}:0"
  return name


@contextlib.contextmanager
def _maybe_open(file, mode):
  if isinstance(file, str):
    with gfile.GFile(file, mode=mode) as f:
      yield f
  else:
    yield file


def write_to_store(obj, path):
  """Util for writing arbitrary objects to store.

  Args:
    obj: arbitrary object to be saved.
    path: a path to save the object.
  """
  def dumps(obj, pickle=builtin_pickle):
    """Returns the bytes representation of a pickled object.

    Args:
      obj: Object to pickle.
      pickle: pickle library to use.
    """
    return pickle.dumps(obj)

  def dump(obj, file, pickle=builtin_pickle):
    """Pickles an object to a file.

    Args:
      obj: Object to pickle.
      file: Name of, or open file-handle to file to write pickled object to.
      pickle: pickle library to use.
    """
    # Because of latency involved in CNS sequential reads, it is way faster to
    # do f.write(dill.dumps(obj)) than dill.dump(f, obj).
    data = dumps(obj, pickle=pickle)
    with _maybe_open(file, mode="wb") as f:
      f.write(data)

  # In order to be robust to interruptions we first save checkpoint to the
  # temporal file and then move to actual path name.
  path_tmp = path + "-TEMPORARY"
  dump(obj, path_tmp)
  gfile.rename(path_tmp, path, overwrite=True)


def read_from_store(path):
  """Util for reading arbitrary objects from store.

  Args:
    path: path at which object is saved.

  Returns:
    Unpickled object.
  """
  def loads(bytes_object, pickle=builtin_pickle):
    """Returns an object unpickled to a bytes.

    Args:
      bytes_object: bytes to unpickle.
      pickle: pickle library to use.
    """
    try:
      return pickle.loads(bytes_object)
    except UnicodeDecodeError:
      logging.exception(
          "Could not read pickled data. "
          "This may be a old pickle written in Python 2, which is no longer "
          "supported. Please use the repickle script to update your pickles.")
      raise

  def load(file, pickle=builtin_pickle):
    """Returns an unpickled an object from a file.

    Args:
      file: Name of, or open file-handle to file containing pickled object.
      pickle: pickle library to use.
    """
    # Because of latency involved in CNS sequential reads, it is way faster to
    # do pickle.loads(f.read()) than pickle.load(f).
    with _maybe_open(file, mode="rb") as f:
      bytes_object = f.read()
    return loads(bytes_object, pickle=pickle)

  return load(path)
