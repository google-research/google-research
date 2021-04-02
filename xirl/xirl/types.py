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

"""Types shared across modules."""

import enum


@enum.unique
class SequenceType(enum.Enum):
  """Sequence data types we know how to preprocess.

  If you need to preprocess additional video data, you must add it here.
  """

  FRAMES = "frames"
  FRAME_IDXS = "frame_idxs"
  VIDEO_NAME = "video_name"
  VIDEO_LEN = "video_len"

  def __str__(self):  # pylint: disable=invalid-str-returned
    return self.value


@enum.unique
class ImageTransformationType(enum.Enum):
  """Transformations we know how to run on images.

  If you want to add additional augmentations, you must add them here.
  """

  RANDOM_RESIZED_CROP = "random_resized_crop"
  CENTER_CROP = "center_crop"
  GLOBAL_RESIZE = "global_resize"
  VERTICAL_FLIP = "vertical_flip"
  HORIZONTAL_FLIP = "horizontal_flip"
  COLOR_JITTER = "color_jitter"
  ROTATE = "rotate"
  DROPOUT = "dropout"
  NORMALIZE = "normalize"
  UNNORMALIZE = "unnormalize"

  def __str__(self):  # pylint: disable=invalid-str-returned
    return self.value
