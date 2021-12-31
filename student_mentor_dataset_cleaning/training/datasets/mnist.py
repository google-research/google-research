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

"""Class to load the MNIST dataset from Tensorflow Datasets."""

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

import student_mentor_dataset_cleaning.training.datasets as datasets


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`.

  Args:
    image: input image with uint8 values in [0, 255].
    label: the label to assign to the image.

  Returns:
    A tuple of:
      The image cast to float and normalized to [0.0, 1.0].
      The input label.
  """
  return tf.cast(image, tf.float32) / 255., label


def create_dataset(student_train_portion,
                   student_validation_portion,
                   mentor_portion,
                   noise_rate,
                   target_distribution_parameter,
                   shuffle_files=True):
  """Creates an MNIST data set from TFDS.

  The dataset will have three splits: student training, student validation and
  mentor training. The training split will be distorted by adding noise and
  resampling to change the class distribution to an exponential distribution.
  The data will be normalized to be within the [0.0, 1.0] range.

  Args:
    student_train_portion: The fraction of the dataset used for student
      training.
    student_validation_portion: The fraction of the dataset used for student
      validation.
    mentor_portion: The fraction of the dataset used for mentor training.
    noise_rate: Fraction of the data points whose labels will be randomized.
    target_distribution_parameter: Parameter controlling the steepness of the
      exponential distribution used for resamling the training split.
    shuffle_files: Whether to shuffle the input data.

  Returns:
    A list containing the three dataset splits as tf.data.Dataset.
  """

  logging.info('Start loading the data')

  portions = [
      0, student_train_portion,
      student_train_portion + student_validation_portion,
      student_train_portion + student_validation_portion + mentor_portion
  ]

  split = [
      tfds.core.ReadInstruction(
          'train', from_=portions[i - 1], to=portions[i], unit='%')
      for i in range(1, len(portions))
  ]

  dataset_splits = tfds.load(
      'mnist',
      split=split,
      shuffle_files=shuffle_files,
      as_supervised=True,
      with_info=False)

  dataset_splits = [
      split.map(
          normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      for split in dataset_splits
  ]

  logging.info('Adding noise to the data set')

  dataset_splits[0] = datasets.corrupt_dataset(
      dataset_splits[0],
      noise_rate=noise_rate,
      target_distribution_parameter=target_distribution_parameter)

  logging.info('Finished loading the data')

  return dataset_splits
