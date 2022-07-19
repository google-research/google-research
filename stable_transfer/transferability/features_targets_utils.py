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

"""Utility functions for features and targets manipulation."""

import numpy as np
from sklearn.decomposition import PCA


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
