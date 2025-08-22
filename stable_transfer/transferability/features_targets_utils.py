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

"""Utility functions for features and targets manipulation."""

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf


def pca_reduction(features, n_components=0.8, svd_solver='full'):
  """Apply PCA dimensionality reduction.

  Args:
    features: matrix of dimension [N, D], where N is the number of datapoints
      and D the feature dimensionality to reduce.
    n_components: if > 1 reduce the dimensionlity of the features to this value,
      if 0 < n_components < 1, select the number of components such that the
      percentage of variance explained is greater than (n_components * 100).
    svd_solver: SVD solver to use. As default we compute the exact full SVD.

  Returns:
    reduced_features: matrix [N, K] of features with reduced dimensionality K.

  """

  reduced_feature = PCA(
      n_components=n_components, svd_solver=svd_solver).fit_transform(features)
  return reduced_feature.astype(np.float32)


def shift_target_labels(target_labels):
  """Change target_labels values to be in the range [0, target classes number).

  Args:
    target_labels: ground truth target labels of dimension [N, 1].

  Returns:
    shifted_target_labels: target_labels in the range [0, target classes number)
  """

  target_labels = np.array(target_labels)
  dict_to_shift = {l: i for i, l in enumerate(set(target_labels))}
  shifted_target_labels = np.array([dict_to_shift[l] for l in target_labels])

  return shifted_target_labels.astype(np.int32)


def compute_class_frequencies(target_train_dataset):
  """Compute the classes frequencies of a semantic segmentation dataset."""
  class_counts = {}
  for _, labels in iter(target_train_dataset):
    labels = tf.reshape(labels, [-1])
    labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(labels, counts):
      if label not in class_counts:
        class_counts[int(label)] = count
      else:
        class_counts[int(label)] += count
  return class_counts


def get_sampling_indices(sampling_seed, class_counts, labels, num_samples):
  """Function to generates sampling indices of the given labels.

  Args:
    sampling_seed: seed used for the sampling process.
    class_counts: instances count of every class in the dataset. If None,
      uniform sampling is applied.
    labels: ground-truth labels from the target dataset.
    num_samples: number of labels to sample from the labels.

  Returns:
    indices: indices for sampling the labels.

  """
  np.random.seed(sampling_seed)
  num_samples = min(num_samples, labels.shape[0])  # To avoid sampling errors

  if not class_counts:  # Uniform sampling
    weights_labels = None
  else:  # Class balanced sampling
    weights_labels = [1/class_counts[int(x)] for x in labels]
    weights_labels = weights_labels / np.sum(weights_labels)

  indices = np.random.choice(
      np.arange(labels.shape[0]),
      num_samples,
      replace=False,
      p=weights_labels)
  return tf.cast(indices, tf.int32)


def sample_from_image(outputs, labels, sampling_seed, class_counts,
                      num_samples):
  """Function to sample images and labels for semantic segmentation.

  Args:
    outputs: either activations or predictions at the pixel level.
    labels: ground-truth labels from the target dataset.
    sampling_seed: seed used for the sampling process.
    class_counts: Instances count of every class in the dataset. If None, random
      sampling is applied.
    num_samples: number of labels to sample from the labels.

  Returns:
    sampled_outputs, sampled_labels

  """

  outputs = tf.reshape(outputs, [-1, outputs.shape[-1]])
  labels = tf.reshape(labels, [-1])

  # Remove background pixels located at label==0
  mask = tf.not_equal(labels, 0)
  labels = tf.gather_nd(labels, tf.where(mask))
  outputs = tf.gather_nd(outputs, tf.where(mask))
  if np.sum(mask) == 0:  # Just background pixels
    return  [], []

  if num_samples:  # Sample num_sample pixels per image.
    sampling_indices = get_sampling_indices(
        sampling_seed, class_counts, labels, num_samples)

    outputs = tf.gather(outputs, sampling_indices)
    labels = tf.gather(labels, sampling_indices)

  return outputs, labels
