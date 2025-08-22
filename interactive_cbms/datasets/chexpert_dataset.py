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

"""Data loading utilities for the CheXpert [1] dataset.

[1] Jeremy Irvin*, Pranav Rajpurkar*, Michael Ko, Yifan Yu,
Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball,
Katie Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi,
Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz,
Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. CheXpert: A Large Chest
Radiograph Dataset with Uncertainty Labels and Expert Comparison, AAAI 2019.
"""

import dataclasses
import functools
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from interactive_cbms.datasets import preproc_util

_PATHOLOGIES = (
    'enlarged_cardiom',
    'cardiomegaly',
    # The order of Lung Opacity and Lung Lesion is flipped in comparison to
    # Table 1 in [1].
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_0ther',
    'fracture',
    'support_devices'
)


@dataclasses.dataclass(frozen=True, init=False)
class Config:
  n_concepts = 13
  n_classes = 1
  image_size = (320, 320, 3)
  data_dir = ('/namespace/health-research/ue/partner/encrypted/opensource-cxr/'
              'deid/etl/tfds/medical-concepts/')


@dataclasses.dataclass(frozen=True, init=False)
class CostSpec:
  """Defines the acquisition costs for different concepts.

  "default" corresponds to acquisition costs for all concepts other than
  cardiomegaly, fracture and support_devices.
  """
  default = 10
  cardiomegaly = 3
  fracture = 1
  support_devices = 1

  @classmethod
  def get_cost(cls, concept_group_name):
    return getattr(cls, concept_group_name, cls.default)


def load_concept_groups():
  """Loads concept group information.

  Returns:
    concept_groups: A dictionary containing concept group names as keys and
      a list of concept indices as values.
  """
  concept_groups = {}
  for concept_i in range(Config.n_concepts):
    concept_groups[_PATHOLOGIES[concept_i]] = [concept_i]
  return concept_groups


def load_concept_costs(concept_groups,
                       **_):
  """Loads concept label acquisition costs for the CheXpert dataset.

  We assign acquisition costs to concepts based on a crude estimation of
  annotation difficulty and the degree of annotator expertise required for this
  dataset. We consider
    "Fracture" and "Support Devices" concepts as easy to annotate as even a
      non-radiologist can identify these,
    "Cardiomegaly" as having medium annotation difficulty as a non-radiologist
      could measure for this using callipers and some heuristics about the
      required ratio, and,
    all the remaining concepts as hard to annotate owing to a lot of
      within-label variation requiring an expert radiologist's opinion.
  The assigned concept costs quantify these qualitative annotation difficult
  estimates.

  Args:
    concept_groups: A dictionary containing concept group names as keys and a
      list of concept indices as values, as returned by load_concept_groups()

  Returns:
    concept_costs: A dictionary mapping concept group names to their respective
    label acquisition costs.
  """
  concept_costs = {}
  for concept_group_name in concept_groups:
    concept_costs[concept_group_name] = CostSpec.get_cost(concept_group_name)
  return concept_costs


def process_tfexample(
    example,
    train = True):
  """Preprocesses CheXpert examples.

  The CheXpert dataset has 14 concepts/pathologies annotated, one of which (the
  "No Finding" concept) is used as the final label. Each concept is annotated
  with integer labels representing
    positive (denoted by 1),
    negative or unmentioned (denoted by 0),
    uncertain (denoted by -1)
  cases. We convert the concept labels to binary by mapping the uncertain cases
  to 0. To represent annotation uncertainty (used in case
  --include_uncertain=False during intervention), we generate concept
  uncertainty labels by mapping uncertain cases to 1, and positive, negative and
  unmentioned cases to 0.

  Args:
    example: A dictionary containing a single CheXpert example.
    train: Whether the example is from the training set. This flag is used to
      determine whether or not to perform image augmentation.

  Returns:
    A tuple containing the parsed image, concept_label, class_label, and
    concept_certainty
  """
  def deterministic_process_image(example):
    """Deterministic image transformations."""
    image = example['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = preproc_util.center_crop(
        image, Config.image_size[0], Config.image_size[0], crop_proportion=1.0)
    return image

  if train:
    image = deterministic_process_image(example)
    image = tf.image.random_flip_left_right(image)
    image = preproc_util.random_brightness(image, 0.1)
  else:
    image = deterministic_process_image(example)

  concept_labels = example['concepts']
  class_label = example['label']
  concept_labels = tf.cast(concept_labels, tf.int64)

  relabel_table = tf.lookup.StaticHashTable(
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant([-1, 0, 1], dtype=tf.int64),
          values=tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)),
      default_value=0)
  uncertainty_table = tf.lookup.StaticHashTable(
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant([-1, 0, 1], dtype=tf.int64),
          values=tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)),
      default_value=0.0)

  concept_uncertainty = uncertainty_table.lookup(concept_labels)
  concept_labels = relabel_table.lookup(concept_labels)

  return image, concept_labels, class_label[None], concept_uncertainty


def load_dataset(
    batch_size = 32,
    merge_train_and_val = True
):
  """Loads the CheXpert dataset.

  Args:
    batch_size: Batch size.
    merge_train_and_val: Whether to merge the training and validation sets to
      create a bigger training set.

  Returns:
    The training, validation and test datasets.
  """

  if merge_train_and_val:
    ds_train, ds_val = tfds.load(
        'chexpert_concept/small', split=['train', 'validation'],
        data_dir=Config.data_dir)
    ds_test = None
  else:
    ds_train, ds_val, ds_test = tfds.load(
        'chexpert_concept/small',
        split=['train[:90%]', 'train[90%:]', 'validation'],
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
