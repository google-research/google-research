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

"""TODO(tsitsulin): add headers, tests, and improve style."""

import numpy as np


def line_gaussians(n_points,  # pylint: disable=missing-function-docstring
                   n_clusters = 2,
                   cluster_distance = 2,
                   noise_scale = 2):
  n_points = n_points // n_clusters * n_clusters
  points_per_cluster = n_points // n_clusters

  data_clean = np.vstack([
      np.random.normal(loc=cluster_distance * i, size=(points_per_cluster, 2))
      for i in range(n_clusters)
  ])

  data_clean -= data_clean.mean(axis=0)  # Make the data zero-mean.

  data_dirty = data_clean + np.random.normal(
      scale=noise_scale, size=data_clean.shape)  # Add random noise to the data.

  labels = np.zeros(n_points, dtype=np.int)
  for i in range(n_clusters):
    labels[points_per_cluster * i:points_per_cluster * (i + 1)] = i

  return data_clean, data_dirty, labels


def circular_gaussians(n_points, n_clusters=8):  # pylint: disable=missing-function-docstring
  avg_points_per_cluster = n_points / n_clusters
  sigma = 0.45 * avg_points_per_cluster
  cluster_sizes = np.random.normal(
      avg_points_per_cluster, sigma, size=n_clusters).astype(np.int)
  labels = np.hstack([
      cluster_idx * np.ones(cluster_size, dtype=np.int)
      for cluster_idx, cluster_size in enumerate(cluster_sizes)
  ])
  data_clean = np.vstack([
      np.random.normal(  # pylint: disable=g-complex-comprehension
          loc=(np.cos(cluster_idx / n_clusters * np.pi * 2),
               np.sin(cluster_idx / n_clusters * np.pi * 2)),
          scale=0.25,
          size=(cluster_size, 2))
      for cluster_idx, cluster_size in enumerate(cluster_sizes)
  ])
  data_dirty = data_clean + np.random.normal(scale=0.25, size=data_clean.shape)
  return data_clean, data_dirty, labels
