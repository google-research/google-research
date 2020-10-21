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

"""Defines constants used across module."""

import enum


@enum.unique
class Dataset(enum.Enum):
  """Look up for dataset names."""

  LBS = "librispeech"

  BSD = "birdsong_detection"

  MUSAN = "musan"

  AS = "audioset"

  TUT = "tut_2018"

  SPCV1 = "speech_commands_v1"

  SPCV2 = "speech_commands"

  NSYNTH_INST = "nsynth_instrument_family"

  VOXCELEB = "voxceleb"

  VOXFORGE = "voxforge"

  CREMA_D = "crema_d"


@enum.unique
class TrainingMode(enum.Enum):
  """Look up for model training modes."""

  SSL = "self_supervised"

  SUP = "supervised"

  RND = "random"

  DS = "downstream"


@enum.unique
class SimilarityMeasure(enum.Enum):
  """Look up for similarity measure in contrastive model."""

  DOT = "dot_product"

  BILINEAR = "bilinear_product"
