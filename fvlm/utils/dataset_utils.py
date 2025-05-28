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

"""Define dataset specific auxiliary data.
"""
import json
from typing import List, Optional

import gin
import numpy as np
import tensorflow as tf


@gin.configurable
def load_dataset_vocab_embed(
    embedding_path = gin.REQUIRED,
    pad_to_size = None):
  """Return the class embeddings of a dataset.

  Args:
    embedding_path: The path to load vocabulary embeddings.
    pad_to_size: Pad the class names to fixed length.

  Returns:
    vocab_embeddings: A tensor of vocabulary embeddings with shape
      [padded_vocabulary, embedding size].
  """
  with tf.io.gfile.GFile(embedding_path, 'rb') as fid:
    vocab_embeddings = np.load(fid)

  if pad_to_size and pad_to_size < vocab_embeddings.shape[0]:
    vocab_embeddings = vocab_embeddings[:pad_to_size]

  return tf.cast(tf.convert_to_tensor(vocab_embeddings), tf.float32)


@gin.configurable
def load_dataset_class_names(vocabulary_path,
                             dataset_num_classes,
                             pad_to_size = None):
  """Return the class names of a dataset.

  Args:
    vocabulary_path: The path to load vocabulary e.g.
      './datasets/coco_mapping.json'
    dataset_num_classes: The number of categories in a dataset.
    pad_to_size: Pad the class names to fixed length.

  Returns:
    class_names: A list of class names with background represented as '.' and
      optionally padded class names as 'empty'.
  """
  with tf.io.gfile.GFile(vocabulary_path) as fid:
    class_mapping = json.load(fid)
    # Create a list of class names with background class '.'.
    class_names = [
        '.' for _ in range(dataset_num_classes)]
    for k, v in class_mapping.items():
      k = int(k)
      if k == 0:
        raise ValueError(f'Class id must be > 0. Got {k}: {v}.')
      class_names[k] = v

  if pad_to_size and pad_to_size > len(class_names):
    num_to_pad = pad_to_size - len(class_names)
    class_names += num_to_pad * ['empty']

  return class_names

