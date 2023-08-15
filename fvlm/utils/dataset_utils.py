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

# TO use a custom dataset, users would need to add a new map path e.g.
# _MY_DATA_MAP_PATH = '/path/to/my_data_mapping.json'
_COCO_MAP_PATH = './datasets/coco_mapping.json'
_DEFAULT_EMBED_DIR = './embeddings/'

# To use a custom dataset, users would need to add a new DatasetType e.g.
# MYDATA = 'my_data'
@gin.constants_from_enum
@enum.unique
class DatasetType(enum.Enum):
  """Dataset type for which dataset to use."""
  COCO = 'coco'

# To use a custom dataset, users would need to update the dictionaries below
# with the new dataset type as key e.g. DatasetType.MYDATA and the corresponding
# map path and number of classes as values.
DATASET_VOCABS = {DatasetType.COCO: _COCO_MAP_PATH}
DATASET_NUM_CLASSES = {DatasetType.COCO: 91}


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

