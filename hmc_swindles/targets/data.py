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

# python3
"""Datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import pickle
from typing import Callable, NamedTuple, Optional, Text, Tuple

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

flags.DEFINE_string('german_credit_numeric_path', '/tmp/german.data-numeric',
                    'Path to the numeric German Credit dataset.')
flags.DEFINE_string('stan_item_response_theory_path', '/tmp/irt.data.pkl',
                    'Path to the Stan 1PL Item-Response Theory dataset.')

FLAGS = flags.FLAGS

__all__ = [
    'ClassificationDataset',
    'ItemResponseTheoryDataset',
    'german_credit_numeric',
    'stan_item_response_theory',
]


class ClassificationDataset(
    NamedTuple('ClassificationDataset', [
        ('train_features', np.ndarray),
        ('train_labels', np.ndarray),
        ('test_features', np.ndarray),
        ('test_labels', np.ndarray),
        ('code_name', Text),
        ('human_name', Text),
    ])):
  """Describes a classification dataset.

  Attributes:
    train_features: Floating-point `Tensor` with shape `[num_train_points,
      num_features]`. Training features.
    train_labels: Integer `Tensor` with shape `[num_train_points]`. Training
      labels.
    test_features: Floating-point `Tensor` with shape `[num_test_points,
      num_features]`. Testing features.
    test_labels: Integer `Tensor` with shape `[num_test_points]`. Testing
      labels.
    code_name: Code name of this dataset.
    human_name: Human readable name, suitable for a table in a paper.
  """


class ItemResponseTheoryDataset(
    NamedTuple('ItemResponseTheoryDataset', [
        ('train_student_ids', np.ndarray),
        ('train_question_ids', np.ndarray),
        ('train_correct', np.ndarray),
        ('test_student_ids', np.ndarray),
        ('test_question_ids', np.ndarray),
        ('test_correct', np.ndarray),
        ('code_name', Text),
        ('human_name', Text),
    ])):
  """Describes a regression dataset.

  `*_correct[i] == 1` means that student `*_student_ids[i]` answered question
  `*_question_ids[i]` correctly; `*_correct[i] == 0` means they didn't.

  Attributes:
    train_student_ids: Integer `Tensor` with shape `[num_train_points]`.
    train_question_ids: Integer `Tensor` with shape `[num_train_points]`.
    train_correct: Integer `Tensor` with shape `[num_train_points]`.
    test_student_ids: Integer `Tensor` with shape `[num_test_points]`.
    test_question_ids: Integer `Tensor` with shape `[num_test_points]`.
    test_correct: Integer `Tensor` with shape `[num_test_points]`.
    code_name: Code name of this dataset.
    human_name: Human readable name, suitable for a table in a paper.
  """


def _normalize_zero_mean_one_std(
    train, test):
  """Normalizes the data columnwise to have mean of 0 and std of 1.

  Assumes that the first axis indexes independent datapoints. The mean and
  standard deviation are estimated from the training set and are used
  for both train and test data, to avoid leaking information.

  Args:
    train: A floating point numpy array representing the training data.
    test: A floating point numpy array representing the test data.

  Returns:
    normalized_train: The normalized training data.
    normalized_test: The normalized test data.
  """
  train = np.asarray(train)
  test = np.asarray(test)
  train_mean = train.mean(0, keepdims=True)
  train_std = train.std(0, keepdims=True)
  return (train - train_mean) / train_std, (test - train_mean) / train_std


def german_credit_numeric(
    path = None,
    train_fraction = 1.,
    normalize_fn = _normalize_zero_mean_one_std,
    shuffle = True,
    shuffle_seed = 1337,
    check_hash = True,
):
  """The numeric German Credit dataset [1].

  This dataset contains 1000 data points with 24 features and 1 binary label.

  Args:
    path: Path to the dataset. If `None`, will read this value from the
      `german_credit_numeric_path` flag. The dataset must be a CSV file with
      1000 rows and 25 colums, each with a numeric value.
    train_fraction: What fraction of the data to put in the training set.
    normalize_fn: A callable to normalize the data. This should take the train
      and test datasets and return the normalized versions of them.
    shuffle: Whether to shuffle the dataset.
    shuffle_seed: Seed to use when shuffling.
    check_hash: Whether to check the loaded dataset against a known hash.

  Raises:
    ValueError: If the dataset has an unexpected hash.

  Returns:
    dataset: ClassificationDataset.

  #### References

  1. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
  """
  if path is None:
    path = FLAGS.german_credit_numeric_path

  with tf.io.gfile.GFile(path, 'rb') as f:
    data = np.genfromtxt(f)

  data = data.astype(np.float32)

  if check_hash:
    expected_digest = 'f905a324d38659765a0bc234238c1d77'
    actual_hash = hashlib.md5(data)
    actual_hash.update(str(data.shape).encode('ascii'))
    actual_digest = actual_hash.hexdigest()
    if expected_digest != actual_digest:
      raise ValueError('Hash of the dataset is wrong: {}'.format(actual_digest))

  if shuffle:
    np.random.RandomState(shuffle_seed).shuffle(data)

  features = data[:, :-1]
  labels = data[:, -1] - 1

  num_train = int(data.shape[0] * train_fraction)

  train_features = features[:num_train]
  test_features = features[num_train:]

  if normalize_fn is not None:
    train_features, test_features = normalize_fn(train_features, test_features)

  return ClassificationDataset(
      train_features=train_features,
      train_labels=labels[:num_train].astype(np.int32),
      test_features=test_features,
      test_labels=labels[num_train:].astype(np.int32),
      code_name='german_credit_numeric_tf{}'.format(train_fraction),
      human_name='German Credit (Numeric)',
  )


def stan_item_response_theory(
    path = None,
    train_fraction = 1.,
    shuffle = True,
    shuffle_seed = 1337,
    check_hash = True):
  """Synthetic item-response theory dataset from Stan example repo [1].

  This dataset is a simulation of 400 students each answering a subset of
  100 unique questions, with a total of 30105 questions answered.

  The dataset is split into train and test portions by randomly partitioning the
  student-question-response triplets. This has two consequences. First, the
  student and question ids are shared between test and train sets. Second, there
  is a possibility of some students or questions not being present in both sets.

  Args:
    path: Path to the dataset. If `None`, will read this value from the
      `stan_item_response_theory_path` flag. The dataset must be a binary
      pickle file containing a dict with keys `student_ids`, `question_ids`,
      and `correct`, which should map to integer numpy arrays each of length
      30105.
    train_fraction: What fraction of the data to put in the training set.
    shuffle: Whether to shuffle the dataset.
    shuffle_seed: Seed to use when shuffling.
    check_hash: Whether to check the loaded dataset against a known hash.

  Raises:
    ValueError: If the dataset has an unexpected hash.

  Returns:
    dataset: ItemResponseTheoryDataset.

  #### References

  1. https://github.com/stan-dev/example-models/blob/master/misc/irt/irt.data.R
  """
  if path is None:
    path = FLAGS.stan_item_response_theory_path

  data = pickle.load(tf.io.gfile.GFile(path, 'rb'))

  if check_hash:
    expected_digest = 'b5cf5a4ac7e1075a4bb04ae1194e4f2f'
    actual_hash = hashlib.md5(data['student_ids'])
    actual_hash.update(str(data['student_ids'].shape).encode('ascii'))
    actual_hash.update(data['question_ids'])
    actual_hash.update(str(data['question_ids'].shape).encode('ascii'))
    actual_hash.update(data['correct'])
    actual_hash.update(str(data['correct'].shape).encode('ascii'))
    actual_digest = actual_hash.hexdigest()
    if expected_digest != actual_digest:
      raise ValueError('Hash of the dataset is wrong: {}'.format(actual_digest))

  student_ids = data['student_ids']
  question_ids = data['question_ids']
  correct = data['correct']

  if shuffle:
    shuffle_idxs = np.arange(student_ids.shape[0])
    np.random.RandomState(shuffle_seed).shuffle(shuffle_idxs)
    student_ids = student_ids[shuffle_idxs]
    question_ids = question_ids[shuffle_idxs]
    correct = correct[shuffle_idxs]

  num_train = int(student_ids.shape[0] * train_fraction)

  return ItemResponseTheoryDataset(
      train_student_ids=student_ids[:num_train],
      train_question_ids=question_ids[:num_train],
      train_correct=correct[:num_train],
      test_student_ids=student_ids[num_train:],
      test_question_ids=question_ids[num_train:],
      test_correct=correct[num_train:],
      code_name='stan_irt_tf{}'.format(train_fraction),
      human_name='1PL Item-Response Theory',
  )
