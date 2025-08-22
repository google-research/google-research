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

"""Data loading utilities for The Caltech-UCSD Birds-200-2011 dataset.

Uses similar train/test splits and preprocessing steps as in [1].

[1] Pang Wei Koh*, Thao Nguyen*, Yew Siang Tang*, Stephen Mussmann,
Emma Pierson, Been Kim, and Percy Liang. Concept Bottleneck Models, ICML 2020.
"""

import collections
import dataclasses
import functools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from interactive_cbms.datasets import preproc_util

# Zero indexed list of concepts in use
_CONCEPTS_IN_USE = (
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
    153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
    196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
    238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311
)
_CONCEPT_GROUPS_PATH = 'datasets/CUB_attributes.txt'


@dataclasses.dataclass(frozen=True, init=False)
class Config:
  n_concepts = 112
  n_classes = 200
  image_size = (299, 299, 3)
  data_dir = 'datasets/CUB_certainty_TFRecord'


def load_concept_groups():
  """Loads concept group information.

  Returns:
    concept_groups: A dictionary containing concept group names as keys and
      a list of concept indices as values.
  """
  with open(_CONCEPT_GROUPS_PATH) as f:
    # Each line in CUB_attributes.txt is of the format:
    #   <concept_idx> <concept_group_name>::<concept_name>
    concepts = f.read().strip().split('\n')

  concept_groups = collections.defaultdict(list)
  for concept in concepts:
    idx, c_name = concept.split()
    if int(idx) - 1 in _CONCEPTS_IN_USE:
      concept_groups[c_name.split('::')[0]].append(
          _CONCEPTS_IN_USE.index(int(idx) - 1))
  return concept_groups


def load_concept_costs(concept_groups,
                       seed):
  """Loads concept label acquisition costs for the CUB dataset.

  We assign random acquisition costs to concepts for this dataset.

  Args:
    concept_groups: A dictionary containing concept group names as keys and a
      list of concept indices as values, as returned by load_concept_groups()
    seed: Random seed used for cost assignment.

  Returns:
    concept_costs: A dictionary mapping concept group names to their respective
    label acquisition costs.
  """
  random_state = np.random.default_rng(seed=seed)
  concept_costs = {
      concept_group_name: random_state.integers(low=1, high=7)
      for concept_group_name in concept_groups
  }
  return concept_costs


def process_tfexample(
    example,
    train = True):
  """Parses and preprocesses CUB examples.

  Args:
    example: A single serialized tf.train.Example proto from the CUB dataset.
    train: Whether the example is from the training set. This flag is used to
      determine whether or not to perform image augmentation.

  Returns:
    A tuple containing the parsed image, concept_label, class_label, and
    concept_certainty
  """
  def deterministic_process_image(
      parsed_example):
    """Deterministic image transformations."""
    image = parsed_example['image']
    image = tf.cast(tf.io.parse_tensor(image, tf.uint8), tf.float32) / 255
    image = tf.ensure_shape(image, [None, None, 3])

    # Center-cropping training images here instead of taking random resized
    # crops like in [1]. The goal is to avoid cases where random resized crops
    # might crop out crucial information that is useful for concept/label
    # predictions. This change doesn't affect the reproducibility of the results
    # in [1].
    image = preproc_util.center_crop(
        image, height=Config.image_size[0], width=Config.image_size[1],
        crop_proportion=1)
    return image

  feature_description = {
      'image': tf.io.FixedLenFeature([], tf.string),
      # 'feature': tf.io.FixedLenFeature([2048], tf.float32),
      'img_path': tf.io.FixedLenFeature([], tf.string),
      'attribute_label': tf.io.FixedLenFeature([112], tf.int64),
      'attribute_certainty': tf.io.FixedLenFeature([312], tf.int64),
      'class_label': tf.io.FixedLenFeature([], tf.int64),
  }
  parsed_example = tf.io.parse_single_example(example, feature_description)

  if train:
    image = deterministic_process_image(parsed_example)
    image = tf.image.random_flip_left_right(image)
    image = preproc_util.color_jitter_rand(
        image, brightness=32 / 255, contrast=0, saturation=0.5, hue=0)
  else:
    image = deterministic_process_image(parsed_example)

  image = tf.clip_by_value(image, 0., 1.)
  image = image * 2 - 1
  parsed_example['image'] = image
  parsed_example['attribute_certainty'] = tf.gather(
      parsed_example['attribute_certainty'], _CONCEPTS_IN_USE, axis=0)

  return (parsed_example['image'], parsed_example['attribute_label'],
          parsed_example['class_label'], parsed_example['attribute_certainty'])


def load_dataset(
    batch_size = 64,
    merge_train_and_val = True
):
  """Loads the CUB dataset.

  Args:
    batch_size: Batch size.
    merge_train_and_val: Whether to merge the training and validation sets to
      create a bigger training set.

  Returns:
    The training, validation and test datasets.
  """

  if merge_train_and_val:
    ds_train = tf.data.TFRecordDataset(
        os.path.join(Config.data_dir, 'train_and_val.tfrecord'))
    ds_val = tf.data.TFRecordDataset(
        os.path.join(Config.data_dir, 'test.tfrecord'))
    ds_test = None
  else:
    ds_train = tf.data.TFRecordDataset(
        os.path.join(Config.data_dir, 'train.tfrecord'))
    ds_val = tf.data.TFRecordDataset(
        os.path.join(Config.data_dir, 'val.tfrecord'))
    ds_test = tf.data.TFRecordDataset(
        os.path.join(Config.data_dir, 'test.tfrecord'))
    ds_test = ds_test.map(functools.partial(process_tfexample, train=False))
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  ds_train = ds_train.map(functools.partial(process_tfexample, train=True))
  ds_train = ds_train.shuffle(1000)
  ds_train = ds_train.batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_val = ds_val.map(functools.partial(process_tfexample, train=False))
  ds_val = ds_val.batch(batch_size)
  ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_val, ds_test
