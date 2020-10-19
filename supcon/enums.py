# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Enums used in contrastive learning code.

They're only here, rather than next to the code that uses them, so that they can
be used as hyperparameter values without pulling in heavy Tensorflow
dependencies to the hyperparameter code.
"""

import enum


@enum.unique
class ModelMode(enum.Enum):
  TRAIN = 1
  EVAL = 2
  INFERENCE = 3


@enum.unique
class AugmentationType(enum.Enum):
  """Valid augmentation types."""
  # SimCLR augmentation (Chen et al, https://arxiv.org/abs/2002.05709).
  SIMCLR = 's'
  # AutoAugment augmentation (Cubuk et al, https://arxiv.org/abs/1805.09501).
  AUTOAUGMENT = 'a'
  # RandAugment augmentation (Cubuk et al, https://arxiv.org/abs/1909.13719).
  RANDAUGMENT = 'r'
  # SimCLR combined with RandAugment.
  STACKED_RANDAUGMENT = 'sr'
  # No augmentation.
  IDENTITY = 'i'


@enum.unique
class LossContrastMode(enum.Enum):
  ALL_VIEWS = 'a'  # All views are contrasted against all other views.
  ONE_VIEW = 'o'  # Only one view is contrasted against all other views.


@enum.unique
class LossSummationLocation(enum.Enum):
  OUTSIDE = 'o'  # Summation location is outside of logarithm
  INSIDE = 'i'  # Summation location is inside of logarithm


@enum.unique
class LossDenominatorMode(enum.Enum):
  ALL = 'a'  # All negatives and all positives
  ONE_POSITIVE = 'o'  # All negatives and one positive
  ONLY_NEGATIVES = 'n'  # Only negatives


@enum.unique
class Optimizer(enum.Enum):
  RMSPROP = 'r'
  MOMENTUM = 'm'
  LARS = 'l'
  ADAM = 'a'
  NESTEROV = 'n'


@enum.unique
class EncoderArchitecture(enum.Enum):
  RESNET_V1 = 'r1'
  RESNEXT = 'rx'


@enum.unique
class DecayType(enum.Enum):
  COSINE = 'c'
  EXPONENTIAL = 'e'
  PIECEWISE_LINEAR = 'p'
  NO_DECAY = 'n'


@enum.unique
class EvalCropMethod(enum.Enum):
  """Methods of cropping eval images to the target dimensions."""
  # Resize so that min image dimension is IMAGE_SIZE + CROP_PADDING, then crop
  # the central IMAGE_SIZExIMAGE_SIZE square.
  RESIZE_THEN_CROP = 'rc'
  # Crop a central square of side length
  # natural_image_min_dim * IMAGE_SIZE/(IMAGE_SIZE+CROP_PADDING), then resize to
  # IMAGE_SIZExIMAGE_SIZE.
  CROP_THEN_RESIZE = 'cr'
  # Crop the central IMAGE_SIZE/(IMAGE_SIZE+CROP_PADDING) pixels along each
  # dimension, preserving the natural image aspect ratio, then resize to
  # IMAGE_SIZExIMAGE_SIZE, which distorts the image.
  CROP_THEN_DISTORT = 'cd'
  # Do nothing. Requires that the input image is already the desired size.
  IDENTITY = 'i'
