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

# pylint: skip-file
"""Utility code for generating and saving image grids and checkpointing.

   The save_image code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import collections
import math
import re
from typing import Any, Dict, Optional, TypeVar

from absl import logging
import flax
import jax
import jax.numpy as jnp
from PIL import Image
import tensorflow as tf

T = TypeVar("T")


def load_state_dict(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state


class CheckpointInfo(
    collections.namedtuple("CheckpointInfo", ("prefix", "number"))):
  """Helper class to parse a TensorFlow checkpoint path."""

  CHECKPOINT_REGEX = r"^(?P<prefix>.*)-(?P<number>\d+)"

  @classmethod
  def initialize(cls, base_directory, checkpoint_name):
    """Creates a first CheckpointInfo (number=1)."""
    return cls(f"{base_directory}/{checkpoint_name}", 1)

  @classmethod
  def from_path(cls, checkpoint):
    """Parses a checkpoint.

    Args:
      checkpoint: A checkpoint prefix, as can be found in the
        `.latest_checkpoint` property of a `tf.train.CheckpointManager`.

    Returns:
      An instance of `CheckpointInfo` that represents `checkpoint`.
    """
    m = re.match(cls.CHECKPOINT_REGEX, checkpoint)
    if m is None:
      RuntimeError(f"Invalid checkpoint format: {checkpoint}")
    d = m.groupdict()  # pytype: disable=attribute-error
    return cls(d["prefix"], int(d["number"]))

  def increment(self):
    """Returns a new CheckpointInfo with `number` increased by one."""
    return CheckpointInfo(self.prefix, self.number + 1)

  def __str__(self):
    """Does the opposite of `.from_path()`."""
    return f"{self.prefix}-{self.number}"


class Checkpoint:
  """A utility class for storing and loading TF2/Flax checkpoints.


  Both the state of a `tf.data.Dataset` iterator and a `flax.struct.dataclass`
  are stored on disk in the following files:

  - {directory}/checkpoint
  - {directory}/ckpt-{number}.index
  - {directory}/ckpt-{number}.data@*
  - {directory}/ckpt-{number}.flax

  Where {number} starts at 1 is then incremented by 1 for every new checkpoint.
  The last file is the `flax.struct.dataclass`, serialized in Messagepack
  format. The other files are explained in more detail in the Tensorflow
  documentation:

  https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
  """

  def __init__(self,
               base_directory,
               tf_state = None,
               *,
               max_to_keep = None,
               checkpoint_name = "ckpt"):
    """Initializes a Checkpoint with a dictionary of TensorFlow Trackables.

    Args:
      base_directory: Directory under which the checkpoints will be stored. Use
        a different base_directory in every task.
      tf_state: A dictionary of TensorFlow `Trackable` to be serialized, for
        example a dataset iterator.
      max_to_keep: Number of checkpoints to keep in the directory. If there are
        more checkpoints than specified by this number, then the oldest
        checkpoints are removed.
      checkpoint_name: Prefix of the checkpoint files (before `-{number}`).
    """
    if tf_state is None:
      tf_state = dict()
    self.base_directory = base_directory
    self.max_to_keep = max_to_keep
    self.checkpoint_name = checkpoint_name
    self.tf_checkpoint = tf.train.Checkpoint(**tf_state)
    self.tf_checkpoint_manager = tf.train.CheckpointManager(
        self.tf_checkpoint,
        base_directory,
        max_to_keep=max_to_keep,
        checkpoint_name=checkpoint_name)

  def get_latest_checkpoint_to_restore_from(self):
    """Returns the latest checkpoint to restore from.

    In the current implementation, this method simply returns the attribute
    `latest_checkpoint`.

    Subclasses can override this method to provide an alternative checkpoint to
    restore from, for example for synchronization across multiple checkpoint
    directories.
    """
    return self.latest_checkpoint

  @property
  def latest_checkpoint(self):
    """Latest checkpoint, see `tf.train.CheckpointManager.latest_checkpoint`.

    Returns:
      A string to the latest checkpoint. Note that this string is path-like but
      it does not really describe a file, but rather a set of files that are
      constructed from this string, by appending different file extensions. The
      returned value is `None` if there is no previously stored checkpoint in
      `base_directory` specified to `__init__()`.
    """
    return self.tf_checkpoint_manager.latest_checkpoint

  @property
  def latest_checkpoint_flax(self):
    """Path of the latest serialized `state`.

    Returns:
      Path of the file containing the serialized Flax state. The returned value
      is `None` if there is no previously stored checkpoint in `base_directory`
      specified to `__init__()`.
    """
    if self.latest_checkpoint is None:
      return None
    return self._flax_path(self.latest_checkpoint)

  def _flax_path(self, checkpoint):
    return "{}.flax".format(checkpoint)

  def _next_checkpoint(self, checkpoint):
    if checkpoint is None:
      return str(
          CheckpointInfo.initialize(self.base_directory, self.checkpoint_name))
    return str(CheckpointInfo.from_path(checkpoint).increment())

  def save(self, state):
    """Saves a new checkpoints in the directory.

    Args:
      state: Flax checkpoint to be stored.

    Returns:
      The checkpoint identifier ({base_directory}/ckpt-{number}).
    """
    next_checkpoint = self._next_checkpoint(self.latest_checkpoint)
    flax_path = self._flax_path(next_checkpoint)
    if not tf.io.gfile.exists(self.base_directory):
      tf.io.gfile.makedirs(self.base_directory)
    with tf.io.gfile.GFile(flax_path, "wb") as f:
      f.write(flax.serialization.to_bytes(state))
    checkpoints = set(self.tf_checkpoint_manager.checkpoints)
    # Write Tensorflow data last. This way Tensorflow checkpoint generation
    # logic will make sure to only commit checkpoints if they complete
    # successfully. A previously written `flax_path` would then simply be
    # overwritten next time.
    self.tf_checkpoint_manager.save()
    for checkpoint in checkpoints.difference(
        self.tf_checkpoint_manager.checkpoints):
      tf.io.gfile.remove(self._flax_path(checkpoint))
    if next_checkpoint != self.latest_checkpoint:
      raise AssertionError(  # pylint: disable=g-doc-exception
          "Expected next_checkpoint to match latest_checkpoint: "
          f"{next_checkpoint} != {self.latest_checkpoint}")
    return self.latest_checkpoint

  def restore_or_initialize(self, state):
    """Restores from the latest checkpoint, or creates a first checkpoint.

    Args:
      state : A flax checkpoint to be stored or to serve as a template. If the
        checkoint is restored (and not initialized), then the fields of `state`
        must match the data previously stored.

    Returns:
      The restored `state` object. Note that all TensorFlow `Trackable`s in
      `tf_state` (see `__init__()`) are also updated.
    """
    latest_checkpoint = self.get_latest_checkpoint_to_restore_from()
    if not latest_checkpoint:
      logging.info("No previous checkpoint found.")
      # Only save one copy for host 0.
      if jax.host_id() == 0:
        self.save(state)
      return state
    self.tf_checkpoint.restore(latest_checkpoint)
    flax_path = self._flax_path(latest_checkpoint)
    with tf.io.gfile.GFile(flax_path, "rb") as f:
      state = flax.serialization.from_bytes(state, f.read())
    return state

  def restore(self, state):
    """Restores from the latest checkpoint.

    Similar to `restore_or_initialize()`, but raises a `FileNotFoundError` if
    there is no checkpoint.

    Args:
      state : A flax checkpoint to be stored or to serve as a template. If the
        checkoint is restored (and not initialized), then the fields of `state`
        must match the data previously stored.

    Returns:
      The restored `state` object. Note that all TensorFlow `Trackable`s in
      `tf_state` (see `__init__()`) are also updated.

    Raises:
      FileNotFoundError: If there is no checkpoint to restore.
    """
    latest_checkpoint = self.get_latest_checkpoint_to_restore_from()
    if not latest_checkpoint:
      raise FileNotFoundError(f"No checkpoint found at {self.base_directory}")
    return self.restore_or_initialize(state)


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, jnp.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
        type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height,
                     x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  im = Image.fromarray(ndarr.copy())
  im.save(fp, format=format)


def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict
