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

# Lint as: python3
"""Dataset configuration."""

import dataclasses
from typing import Optional, Tuple

from vatt.configs import base_config


@dataclasses.dataclass
class Dataset(base_config.Config):
  """The base configuration for building datasets."""

  name: Optional[str] = None
  split: str = ''
  batch_size: int = 8
  num_frames: int = 32
  frame_size: int = 224
  video_stride: int = 1
  audio_stride: int = 1
  crop_resize_style: str = 'VGG'  # either 'VGG' or 'Inception'
  min_area_ratio: float = 0.08  # valid if crop_resize_style = 'Inception'
  max_area_ratio: float = 1.0  # valid if crop_resize_style = 'Inception'
  min_aspect_ratio: float = 0.5  # valid if crop_resize_style = 'Inception'
  max_aspect_ratio: float = 2.0  # valid if crop_resize_style = 'Inception'
  scale_jitter: bool = True  # valid if crop_resize_style = 'VGG'
  min_resize: int = 224  # valid if crop_resize_style = 'VGG'
  color_augment: bool = True
  zero_centering_image: bool = True
  space_to_depth: bool = False
  text_tokenizer: str = 'WordTokenizer'  # 'BertTokenizer' or 'WordTokenizer'
  raw_audio: bool = True
  stft_length: float = 0.04267  # in ms - MMV: 0.04267
  stft_step: float = 0.02134  # in ms - MMV: 0.02134
  mel_bins: int = 80
  linearize_vision: bool = True

  @property
  def has_data(self):
    """Whether this dataset has any data associated with it."""
    return self.name is not None


@dataclasses.dataclass
class Pretrain(Dataset):
  """Pre-train dataset configuration."""

  name: str = 'howto100m+audioset'
  split: str = 'train'
  max_num_words: int = 16
  max_context_sentences: int = 4
  audio_noise: float = 0.01
  audio_mixup: bool = False
  mixup_alpha: int = 10
  mixup_beta: int = 2
  num_examples: int = -1


@dataclasses.dataclass
class Evaluation(Dataset):
  """Evauation dataset configuration."""

  name: Tuple[str, Ellipsis] = (
      'esc50',
      'hmdb51',
      'ucf101',
      'youcook2',
      'msrvtt',
      )
  split: str = 'test'
  video_stride: int = 2
  audio_stride: int = 1
  audio_mixup: bool = False
  mixup_alpha: int = 10
  mixup_beta: int = 1
  multi_crop: bool = False
  max_num_words: int = 16
  num_examples: int = 4096
  num_windows_test: int = 4
  num_augmentation: int = 1


@dataclasses.dataclass
class Finetune(Dataset):
  """Fine-tune dataset configuration."""

  name: str = 'kinetics400'
  num_frames: int = 32
  frame_size: int = 224
  video_stride: int = 2
  audio_stride: int = 2
  mixup: bool = False  # True when finetuning audio (audioset)
  mixup_alpha: float = 5
  color_augment: bool = True
  audio_noise: float = 0.0
  label_smoothing: float = 0.1
  num_windows_test: int = 4
  multi_crop: bool = True

