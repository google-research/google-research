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

"""Definitions and types representing videos in TensorFlow.

This provides a unified API for both training and evaluation videos, to be
ingested by neural methods.

To construct:
- training data: see training_data.py
- eval data: see video2/eval/datasets/store.py
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, NamedTuple, NewType, Tuple, Union, TypeVar

import tensorflow as tf


# A dataset where the elements are `TrainingVideo` instances.
TrainingVideoDataset = NewType("TrainingVideoDataset", tf.data.Dataset)


def normalize_for_rgb(raw_frame):
  """Normalize a uint8 [0...255] tensor to a float32 [0...1] tensor."""
  return tf.cast(raw_frame, tf.float32) / 255.0


def denormalize(float32_frame):
  """De-normalize to uint8 [0...255] tensor from a float32 [0...1] tensor."""
  if (not isinstance(float32_frame, tf.Tensor) or
      float32_frame.dtype != tf.float32):
    raise ValueError(f"Invalid input: {float32_frame}")
  return tf.image.convert_image_dtype(float32_frame, tf.uint8, saturate=True)


class Frame(NamedTuple):
  """Represent a batch of frames with metadata."""

  # Both have shape (B, H, W, C), B == 1 for eval.
  rgb: tf.Tensor

  @property
  def batch_size(self):
    """Return the batch size."""
    self.validate_shape_and_dtype()
    return self.rgb.shape[0]

  @property
  def spatial_shape(self):
    """Return (height, width)."""
    self.validate_shape_and_dtype()
    height, width = self.rgb.shape[1], self.rgb.shape[2]
    return height, width

  @property
  def num_pixels(self):
    """Return height*width."""
    height, width = self.spatial_shape
    return height*width

  def validate_shape_and_dtype(self):
    """Raise ValueError if we have invalid shapes."""
    if self.rgb.dtype != tf.float32:
      raise ValueError("Expected float32 rgb!")
    if len(self.rgb.shape) != 4:
      raise ValueError(f"Expected (B, H, W, C), got {self.rgb.shape}")
    _, _, _, channels = self.rgb.shape.as_list()
    if channels != 3:
      raise ValueError(f"Expected 3 rgb channels, got shape {self.rgb.shape}")

  def apply(self, fn):
    """Obtain a new Frame batch by applying `fn` on each element."""
    return Frame(fn(self.rgb))

  @classmethod
  def reduce(cls, frames,
             reduce_fn):
    """Obtain a new Frame batch by applying `reduce_fn` on each element."""
    rgb = reduce_fn([f.rgb for f in frames])
    return Frame(rgb)


class TrainingVideo(NamedTuple):
  """Represents a video to be used for training.

  In contrast to `EvalVideo`, here all data is available as a single tensor,
  and we thus know the number of frames. It remains to be seen whether this is
  useful, or whether we should unify the two NamedTuple's.
  """
  # Shape: (B, F, H, W, C), where F == number of frames.
  rgb: tf.Tensor

  @classmethod
  def from_frames(cls, frames):
    """Create an instance from frames."""
    rgbs = []
    for frame in frames:
      rgbs.append(frame.rgb)
    return cls(tf.stack(rgbs, axis=1))

  @classmethod
  def make_random(
      cls,
      batch_size = 2,
      num_frames = 5,
      dim = 64,
  ):
    """Create a random instance."""
    base_shape = (batch_size, num_frames, dim, dim)
    random_rgb = tf.random.stateless_normal((*base_shape, 3), seed=[1, 1])
    return cls(random_rgb)

  @classmethod
  def make(cls, rgb_uint8):
    """Create an instance from a uint8 rgb."""
    if rgb_uint8.dtype != tf.uint8:
      raise ValueError("Need uint8!")
    rgb = normalize_for_rgb(rgb_uint8)
    instance = cls(rgb)
    instance.validate_shape()
    return instance

  # TODO(mentzer): The following two methods are probably only needed because
  # of incorrect tests that do not use strategy.run. Should refactor the test
  # to see if we can remove them. Some context: if we do not use strategy.run,
  # self.rgb and self.flow are PerReplica tensors on TPU/multi GPU, and
  # we cannot do shape and dtype validation.
  @property
  def _first_rgb(self):
    if isinstance(self.rgb, tf.Tensor):
      return self.rgb
    else:  # PerReplica tensors here!
      return self.rgb.values[0]

  @property
  def num_frames(self):
    """Return the number of frames in the video."""
    return self._first_rgb.shape[1]

  @property
  def batch_size(self):
    """Return the batch size."""
    return self._first_rgb.shape[0]

  @property
  def spatial_shape(self):
    """Return (height, width)."""
    return self._first_rgb.shape[2], self._first_rgb.shape[3]

  def validate_shape(self):
    """Raise ValueError if we have invalid shapes."""
    if len(self._first_rgb.shape) != 5:
      raise ValueError(f"Invalid shape: {self._first_rgb.shape}")

  def get_frames(self):
    """Returns frames."""
    # We cannot validate shape on construction as that happens inside graph mode
    # as we construct from a tf.data.Dataset, so we validate here.
    self.validate_shape()
    frames = []
    for i in range(self.num_frames):
      rgb_i = self.rgb[:, i, Ellipsis]
      frames.append(Frame(rgb_i))
    return frames


class EvalVideo(NamedTuple):
  """Represents a video to be used for eval, via an iterator over frames."""

  video: Sequence[Frame]

  @classmethod
  def from_frames(cls, frames):
    return cls(frames)

  @classmethod
  def make_random(cls, num_frames = 5,
                  dim = 64):
    """Create a random instance."""
    if isinstance(dim, int):
      dim = (dim, dim)
    elif len(dim) != 2:
      raise ValueError(f"Dimension should be an int or a 2-tuple, not {dim}")
    dim = (1,) + tuple(dim)

    def video_iterator():
      for _ in range(num_frames):
        random_rgb = tf.random.stateless_normal(dim + (3,), seed=[1, 1])
        yield Frame(random_rgb)
    return cls(list(video_iterator()))

  def get_frames(self):
    """Returns frames, validates shape of first frame."""
    if not self.video:
      return []
    # We cannot validate shape on construction as that happens inside graph
    # mode as we construct from a tf.data.Dataset, so we validate here.
    self.video[0].validate_shape_and_dtype()
    return self.video

  @property
  def num_frames(self):
    """Return the number of frames in the video."""
    return len(self.video)


# Represents either of the above videos.
VideoT = TypeVar("VideoT", TrainingVideo, EvalVideo)
