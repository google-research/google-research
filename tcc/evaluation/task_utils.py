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

"""Common evalaution functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def regression_labels_for_class(labels, class_idx):
  # Assumes labels are ordered. Find the last occurrence of particular class.
  transition_frame = np.argwhere(labels == class_idx)[-1, 0]
  return (np.arange(float(len(labels))) - transition_frame) / len(labels)


def get_regression_labels(class_labels, num_classes):
  regression_labels = []
  for i in range(num_classes - 1):
    regression_labels.append(regression_labels_for_class(class_labels, i))
  return np.stack(regression_labels, axis=1)


def get_targets_from_labels(all_class_labels, num_classes):
  all_regression_labels = []
  for class_labels in all_class_labels:
    all_regression_labels.append(get_regression_labels(class_labels,
                                                       num_classes))
  return all_regression_labels


def unnormalize(preds):
  seq_len = len(preds)
  return np.mean([i - pred * seq_len for i, pred in enumerate(preds)])
