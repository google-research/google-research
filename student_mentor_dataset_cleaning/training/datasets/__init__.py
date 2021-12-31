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

"""Functions to load different datasets and process them."""

import math
import random

from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp


def add_noise(dataset, noise_rate):
  """Adds noise to the data set by flipping some labels randomly.

  Args:
    dataset: The input dataset.
    noise_rate: The probability that the label of an example will be randomized.

  Returns:
    The modified dataset. Weights of modified examples are set to 0.
  """

  num_labels = len(set([y.numpy() for _, y, _ in dataset]))

  def create_noise(x, y, w):
    shift = 0 if random.random() > noise_rate else int(random.random() *
                                                       num_labels)
    return (x, ((y + shift) % num_labels), (w if shift == 0 else 0.0))

  return dataset.map(create_noise)


def imbalanced_sample(dataset, target_distribution):
  """Oversamples training data to produce a given class distribution.

  Args:
    dataset: The input dataset to sample from.
    target_distribution: Mapping from class label to its probability of being
      drawn.

  Returns:
    The resampled dataset.
  """

  counts = dict()
  size = 0.0
  for _, y, _ in dataset:
    counts[y.numpy()] = counts.get(y.numpy(), 0.0) + 1.0
    size += 1.0
  classes = counts.keys()

  current_distribution = {
      class_id: counts[class_id] / size for class_id in classes
  }

  normalization_factor = 0.0
  for label in target_distribution:
    normalization_factor += target_distribution[label]
  for label in target_distribution:
    target_distribution[label] /= normalization_factor

  amplification_factor = {
      class_id: target_distribution[class_id] / current_distribution[class_id]
      for class_id in classes
  }
  normalization_factor = min(
      [v if v > 0 else float('inf') for v in amplification_factor.values()])
  amplification_factor = {
      k: (v / normalization_factor) for (k, v) in amplification_factor.items()
  }
  repeat_count = {
      class_id: math.floor(amplification_factor[class_id])
      for class_id in classes
  }
  residual_probability = {
      class_id: amplification_factor[class_id] - repeat_count[class_id]
      for class_id in classes
  }

  new_x_list = []
  new_y_list = []
  new_w_list = []
  for _, (x, y, w) in enumerate(dataset):
    repeat = repeat_count[y.numpy()]
    repeat += random.random() < residual_probability[y.numpy()]
    new_x_list += [x for _ in range(repeat)]
    new_y_list += [y for _ in range(repeat)]
    new_w_list += [w for _ in range(repeat)]
  new_x = tf.data.Dataset.from_tensor_slices(new_x_list)
  new_y = tf.data.Dataset.from_tensor_slices(new_y_list)
  new_w = tf.data.Dataset.from_tensor_slices(new_w_list)
  new_dataset = tf.data.Dataset.zip((new_x, new_y, new_w))

  return new_dataset


def imbalanced_sample_exponential(
    dataset,
    target_distribution_parameter):
  """Oversamples training data to produce a given exponential distribution.

  Args:
    dataset: The input dataset where each example is a 3-tuple of (data, label,
      weight).
    target_distribution_parameter: Parameter controlling the steepness of the
      exponential distribution for resampling. If set to 0, data is resampled
      uniformly.

  Returns:
    The modified dataset.
  """

  labels = set([y.numpy() for x, y, w in dataset])
  num_labels = len(labels)

  if 1 - math.exp(target_distribution_parameter) != 0:
    normalization_factor = (1 -
                            math.exp(num_labels * target_distribution_parameter)
                           ) / (1 - math.exp(target_distribution_parameter))
  else:
    normalization_factor = 1 / num_labels

  target_distribution = {
      label: math.exp(i * target_distribution_parameter) / normalization_factor
      for i, label in enumerate(labels)
  }

  return imbalanced_sample(dataset, target_distribution=target_distribution)


def dataset_split(dataset, first_split_ratio):
  """Splits a dataset into two parts.

  Args:
    dataset: The input dataset.
    first_split_ratio: The ratio of the sizes of the desired output splits.

  Returns:
    A tuple containing the two output datasets.
  """

  randomness = tf.data.Dataset.from_tensor_slices(
      tf.random.uniform(shape=(dataset.cardinality().numpy(),)))
  dataset = tf.data.Dataset.zip((dataset, randomness))
  first_split = dataset.filter(lambda data, r: r <= first_split_ratio).map(
      lambda data, r: data)
  second_split = dataset.filter(lambda data, r: r > first_split_ratio).map(
      lambda data, r: data)
  return first_split, second_split


def corrupt_dataset(dataset,
                    noise_rate,
                    target_distribution_parameter,
                    include_weights = False):
  """Corrupts a dataset by adding noise and unbalance.

  The dataset will be resampled to create a long-tailed class distribution.

  Args:
    dataset: The input dataset.
    noise_rate: Fraction of the labels that will be replaced by noise.
    target_distribution_parameter: Parameter controlling the steepness of the
      exponential distribution for resampling.
    include_weights: Whether weights will be returned for each example in the
      dataset.

  Returns:
    The modified dataset. If `include_weights` is True, each example has an
    addition third tuple element representing the weight.
  """

  dataset = dataset.map(lambda x, y: (x, y, 1.0))
  dataset = imbalanced_sample_exponential(
      dataset, target_distribution_parameter=target_distribution_parameter)
  dataset = add_noise(dataset, noise_rate=noise_rate)
  if include_weights:
    return dataset
  else:
    return dataset.map(lambda x, y, w: (x, y))
