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


def shuffle_inbatch(features_mat, subgraph_sizes, shuffle_ratio = 1.0):  # pylint: disable=missing-function-docstring
  batch_size = features_mat.shape[0]
  assert batch_size == len(
      subgraph_sizes
  ), 'Length of subgraph size array is not equal to the inferred batch size.'
  matrix_content = np.vstack([
      features_mat[index, :, :][:subgraph_sizes[index]]  # Non-zero component.
      for index in range(batch_size)
  ])
  n_nodes = matrix_content.shape[0]  #  Total number of nodes sampled.
  shuffle_indices = np.random.choice(
      np.arange(n_nodes), int(n_nodes * shuffle_ratio))
  matrix_content_tmp = matrix_content[shuffle_indices]
  np.random.shuffle(matrix_content_tmp)
  matrix_content[shuffle_indices] = matrix_content_tmp
  features_corr = np.zeros(features_mat.shape)
  current_index = 0
  for index in range(batch_size):
    features_corr[index, :, :][:subgraph_sizes[index]] = matrix_content[
        current_index:current_index + subgraph_sizes[index]]
    current_index += subgraph_sizes[index]
  return features_corr
