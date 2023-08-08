# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Define dataset specific auxiliary data.
"""
import enum
import json
from typing import List, Optional

import gin
import numpy as np
import tensorflow as tf


_OBJECT_MAP_PATH = './datasets/my_dataset_mapping.json'
_DEFAULT_EMBED_DIR = './embeddings/vit_b32/'
_BASE_INDICATOR_PATH = './datasets/my_dataset_base_indicator.json'


@gin.constants_from_enum
@enum.unique
class DatasetType(enum.Enum):
  """Dataset type for which dataset to use."""
  CUSTOM = 'custom'


DATASET_VOCABS = {DatasetType.CUSTOM: _OBJECT_MAP_PATH}
DATASET_NUM_CLASSES = {DatasetType.CUSTOM: 91}
DATASET_IS_BASE = {DatasetType.CUSTOM: _BASE_INDICATOR_PATH}


@gin.configurable
def load_dataset_base_indicator(
    dataset_type,
    pad_to_size = None):
  """Return the base category indicator of a dataset.

  This is used for ensembling scores for open-vocabulary detection.

  Args:
    dataset_type: The type of dataset.
    pad_to_size: Pad or trim the indicator to fixed length.

  Returns:
    class_names: A list of [0, 1] values indicating the base class (1) or novel
      class of a dataset. The length is num_classes + 1 (background).
  """
  data_path = DATASET_IS_BASE[dataset_type]
  with open(data_path) as fid:
    base_indicator = json.load(fid)

  if pad_to_size is None:
    return base_indicator

  if len(base_indicator) == pad_to_size:
    return base_indicator
  elif pad_to_size > len(base_indicator):
    return base_indicator + [0] * (pad_to_size - len(base_indicator))
  else:
    return base_indicator[:pad_to_size]


@gin.configurable
def load_dataset_vocab_embed(
    dataset_type,
    pad_to_size = None,
    vocab_embed_dir = _DEFAULT_EMBED_DIR):
  """Return the class embeddings of a dataset.

  Args:
    dataset_type: The type of dataset.
    pad_to_size: Pad the class names to fixed length.
    vocab_embed_dir: The directory to load the embeddings from.

  Returns:
    vocab_embeddings: A tensor of vocabulary embeddings with shape
      [padded_vocabulary, embedding size].
  """
  vocabulary_path = f'{vocab_embed_dir}/{dataset_type.value}_embed.npy'
  with open(vocabulary_path, 'rb') as fid:
    vocab_embeddings = np.load(fid)

  if pad_to_size and pad_to_size < vocab_embeddings.shape[0]:
    vocab_embeddings = vocab_embeddings[:pad_to_size]

  return tf.cast(tf.convert_to_tensor(vocab_embeddings), tf.float32)


@gin.configurable
def load_dataset_class_names(dataset_type,
                             pad_to_size = None):
  """Return the class names of a dataset.

  Args:
    dataset_type: the type of dataset.
    pad_to_size: Pad the class names to fixed length.

  Returns:
    class_names: A list of class names with background represented as '.' and
      optionally padded class names as 'empty'.
  """
  vocabulary_path = DATASET_VOCABS[dataset_type]
  with open(vocabulary_path) as fid:
    class_mapping = json.load(fid)
    # Create a list of class names with background class '.'.
    class_names = [
        '.' for _ in range(DATASET_NUM_CLASSES[dataset_type])]
    for k, v in class_mapping.items():
      k = int(k)
      if k == 0:
        raise ValueError(f'Class id must be > 0. Got {k}: {v}.')
      class_names[k] = v

  if pad_to_size and pad_to_size > len(class_names):
    num_to_pad = pad_to_size - len(class_names)
    class_names += num_to_pad * ['empty']

  return class_names

