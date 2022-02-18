# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Utility functions."""

import numpy as np
import tensorflow as tf


def maybe_one_hot(labels, depth):
  """Convert categorical labels to one-hot, if needed.

  Args:
    labels: A `Tensor` containing labels.
    depth: An integer specifying the depth of one-hot represention (number of
      classes).

  Returns:
    One-hot labels.
  """
  if len(labels.shape) > 1:
    return labels
  else:
    return tf.one_hot(labels, depth=depth)


def get_smoothed_labels(labels, preds, smoothing_weights):
  """Smoothen the labels."""
  smoothing_weights = tf.reshape(smoothing_weights, [-1, 1])
  return labels * smoothing_weights + preds * (1. - smoothing_weights)


def mixup(images,
          labels,
          num_classes,
          mixup_alpha,
          mixing_weights=None,
          mixing_probs=None):
  """Mixup with mixing weights and probabilities.

  Args:
    images: A `Tensor` containing batch of images.
    labels: A `Tensor` containing batch of labels.
    num_classes: Number of classes.
    mixup_alpha: Parameter of Beta distribution for sampling mixing ratio
      (applicable for regular mixup).
    mixing_weights: A `Tensor` of size [batch_size] specifying mixing weights.
    mixing_probs: A `Tensor` of size [batch_size] specifying probabilities for
      sampling images for imixing.

  Returns:
    Minibatch of mixed up images and labels.
  """

  images = images.numpy()
  labels = maybe_one_hot(labels, num_classes).numpy()
  num_examples = images.shape[0]
  mixing_ratios_im = np.random.beta(
      mixup_alpha, mixup_alpha, size=(num_examples, 1, 1, 1))
  mixing_ratios_lab = np.reshape(mixing_ratios_im, [num_examples, 1])
  if mixing_probs is None:
    mixing_indices = np.random.permutation(num_examples)
  else:
    mixing_probs = np.round(mixing_probs, 5)
    mixing_probs = mixing_probs / np.sum(mixing_probs)
    mixing_indices = np.random.choice(
        num_examples, size=num_examples, replace=True, p=mixing_probs)
  if mixing_weights is not None:
    mixing_ratios_im = mixing_weights / (
        mixing_weights + mixing_weights[mixing_indices])
    mixing_ratios_im = np.reshape(mixing_ratios_im, [-1, 1, 1, 1])
    # mix labels in same proportions
    mixing_ratios_lab = np.reshape(mixing_ratios_im, [num_examples, 1])
  images = (
      images * mixing_ratios_im + images[mixing_indices] *
      (1. - mixing_ratios_im))
  labels = (
      labels * mixing_ratios_lab + labels[mixing_indices] *
      (1. - mixing_ratios_lab))
  return images, labels
