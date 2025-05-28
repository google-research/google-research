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

"""Data loading utilities for the OAI [1] dataset.

Uses the same preprocessing steps as in [2].

[1] Nevitt, M., Felson, D. T., and Lester, G. The Osteoarthritis Initiative.
Cohort study protocol, 2006.
[2] Pang Wei Koh*, Thao Nguyen*, Yew Siang Tang*, Stephen Mussmann,
Emma Pierson, Been Kim, and Percy Liang. Concept Bottleneck Models, ICML 2020.
"""

import collections
import dataclasses
import functools
from typing import Dict, List, Optional, Tuple

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from interactive_cbms.datasets import preproc_util

# Zero indexed list of concepts in use
_CONCEPTS_IN_USE = (0, 1, 3, 5, 6, 9, 10, 12, 14, 15)
_PATHOLOGIES = (
    'xrosfm',
    'xrscfm',
    'xrjsm',
    'xrostm',
    'xrsctm',
    'xrosfl',
    'xrscfl',
    'xrjsl',
    'xrostl',
    'xrsctl'
)


@dataclasses.dataclass(frozen=True, init=False)
class Config:
  n_concepts = 40
  n_classes = 4
  image_size = (512, 512, 3)
  data_dir = ('/namespace/health-research/ue/partner/encrypted/opensource-cxr/'
              'deid/etl/tfds/medical-concepts/')


def load_concept_groups():
  """Loads concept group information.

  Returns:
    concept_groups: A dictionary containing concept group names as keys and
      a list of concept indices as values.
  """
  concept_groups = collections.defaultdict(list)
  for concept_i in range(Config.n_concepts):
    concept_groups[_PATHOLOGIES[concept_i//4]].append(concept_i)
  return concept_groups


def load_concept_costs(concept_groups,
                       seed):
  """Loads concept label acquisition costs for the OAI dataset.

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
    train = True,):
  """Preprocesses OAI examples.

  The OAI dataset has annotations for 18 clinical concepts along with the
  Kellgren-Lawrence grade (KLG), a 5-level categorical variable (0 to 4), which
  constitutes the class label. Similar to [2], we preprocess the data by
    1) merging KLG = 0 and KLG = 1 classes into a single level and translating
       the other KLG levels downwards by 1, leading to a 4-level categorical
       target (0 to 3),
    2) using only 10 out of 18 available concepts,
    3) truncating the fractional grades present in the concept annotations by
       type-casting the labels to int64, and
    4) encoding the categorical concept annotations as one-hot vectors.

  Args:
    example: A dictionary containing a single OAI example.
    train: Whether the example is from the training set. This flag is used to
      determine whether or not to perform image augmentation.

  Returns:
    A tuple containing the parsed image, concept_label, class_label, and
    concept_certainty
  """

  def deterministic_process_image(example):
    """Deterministic image transformations."""
    image = example['image']
    image = tf.repeat(image, repeats=3, axis=2)
    image = tf.image.resize(image, Config.image_size[:2])
    image = tf.image.convert_image_dtype(image, tf.float32) / 255
    return image

  if train:
    image = deterministic_process_image(example)
    image = tf.image.random_flip_left_right(image)
    image = preproc_util.random_brightness(image, 0.1)
  else:
    image = deterministic_process_image(example)
  class_label = example['label']
  concept_labels = example['concepts']

  concept_labels = tf.gather(concept_labels, _CONCEPTS_IN_USE, axis=0)
  concept_labels = tf.cast(concept_labels, tf.int64)
  concept_labels = tf.clip_by_value(concept_labels, 0, 3)
  onehot_concept_labels = tf.equal(
      tf.range(4, dtype=tf.int64)[None, :],
      concept_labels[:, None])
  onehot_concept_labels = tf.reshape(onehot_concept_labels, [-1])
  onehot_concept_labels = tf.cast(onehot_concept_labels, tf.float32)

  class_label = tf.clip_by_value(class_label-1, 0, 3)
  concept_uncertainty = tf.zeros_like(onehot_concept_labels)

  return image, onehot_concept_labels, class_label, concept_uncertainty


def load_dataset(
    batch_size = 32,
    merge_train_and_val = True,
):
  """Loads the OAI dataset.

  Args:
    batch_size: Batch size.
    merge_train_and_val: Whether to merge the training and validation sets to
      create a bigger training set.

  Returns:
    The training, validation and test datasets.
  """

  if merge_train_and_val:
    ds_train, ds_val = tfds.load(
        'oai_knee_concept/per-knee-medium', split=['train+validation', 'test'],
        data_dir=Config.data_dir)
    ds_test = None
  else:
    ds_train, ds_val, ds_test = tfds.load(
        'oai_knee_concept/per-knee-medium',
        split=['train', 'validation', 'test'],
        data_dir=Config.data_dir)
    ds_test = ds_test.map(functools.partial(process_tfexample, train=False))
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  ds_train = ds_train.map(functools.partial(process_tfexample, train=True))
  ds_train = ds_train.shuffle(1000).batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_val = ds_val.map(functools.partial(process_tfexample, train=False))
  ds_val = ds_val.batch(batch_size)
  ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_val, ds_test
