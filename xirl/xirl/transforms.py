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

"""Transformations for video data."""

import enum
import warnings

import albumentations as alb
import numpy as np
import torch
from xirl.types import SequenceType


@enum.unique
class PretrainedMeans(enum.Enum):
  """Pretrained mean normalization values."""

  IMAGENET = (0.485, 0.456, 0.406)


@enum.unique
class PretrainedStds(enum.Enum):
  """Pretrained std deviation normalization values."""

  IMAGENET = (0.229, 0.224, 0.225)


class UnNormalize:
  """Unnormalize a batch of images that have been normalized.

  Speficially, re-multiply by the standard deviation and shift by the mean.
  """

  def __init__(
      self,
      mean,
      std,
  ):
    """Constructor.

    Args:
      mean: The color channel means.
      std: The color channel standard deviation.
    """
    if np.asarray(mean).shape:
      self.mean = torch.tensor(mean)[Ellipsis, :, None, None]
    if np.asarray(std).shape:
      self.std = torch.tensor(std)[Ellipsis, :, None, None]

  def __call__(self, tensor):
    return (tensor * self.std) + self.mean


def augment_video(
    frames,
    pipeline,  # pylint: disable=g-bare-generic
):
  """Apply the same augmentation pipeline to all frames in a video.

  Args:
    frames: A numpy array of shape (T, H, W, 3), where T is the number of frames
      in the video.
    pipeline (list): A list containing albumentation augmentations.

  Returns:
    The augmented frames of shape (T, H, W, 3).

  Raises:
    ValueError: If the input video doesn't have the correct shape.
  """
  if frames.ndim != 4:
    raise ValueError("Input video must be a 4D sequence of frames.")

  transform = alb.ReplayCompose(pipeline, p=1.0)

  # Apply a transformation to the first frame and record the parameters
  # that were sampled in a replay, then use the parameters stored in the
  # replay to apply an identical transform to the remaining frames in the
  # sequence.
  with warnings.catch_warnings():
    # This supresses albumentations' warning related to ReplayCompose.
    warnings.simplefilter("ignore")

    replay, frames_aug = None, []
    for frame in frames:
      if replay is None:
        aug = transform(image=frame)
        replay = aug.pop("replay")
      else:
        aug = transform.replay(replay, image=frame)
      frames_aug.append(aug["image"])

  return np.stack(frames_aug, axis=0)


class VideoAugmentor:
  """Data augmentation for videos.

  Augmentor consistently augments data across the time dimension (i.e. dim 0).
  In other words, the same transformation is applied to every single frame in
  a video sequence.

  Currently, only image frames, i.e. SequenceType.FRAMES in a video can be
  augmented.
  """

  MAP = {
      SequenceType.FRAMES: augment_video,
  }

  def __init__(
      self,
      params,  # pylint: disable=g-bare-generic
  ):
    """Constructor.

    Args:
      params:

    Raises:
      ValueError: If params contains an unsupported data augmentation.
    """
    for key in params.keys():
      if key not in SequenceType:
        raise ValueError(f"{key} is not a supported SequenceType.")
    self._params = params

  def __call__(
      self,
      data,
  ):
    """Iterate and transform the data values.

    Currently, data augmentation is only applied to video frames, i.e. the
    value of the data dict associated with the SequenceType.IMAGE key.

    Args:
      data: A dict mapping from sequence type to sequence value.

    Returns:
      A an augmented dict.
    """
    for key, transforms in self._params.items():
      data[key] = VideoAugmentor.MAP[key](data[key], transforms)
    return data
