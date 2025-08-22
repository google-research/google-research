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

"""A non-scalable InstantEmbedding implementation."""
import numpy as np
import scipy.sparse as sps


def instant_embedding(pprs, dimensionality):
  """Produces non-private InstantEmbeddings for given set of PPRs.

  Args:
    pprs: Input PPR matrix. Can be non-square (m x n).
    dimensionality: Dimensionality of the embedding vector.

  Returns:
    InstantEmbedding matrix of size (m x `dimensionality`).
  """
  n_input_nodes = pprs.shape[0]
  n_nodes = pprs.shape[1]
  ppr_trans = np.log(np.maximum(1, n_nodes * pprs))
  proj = sps.coo_matrix((
      np.random.choice([-1, 1], n_input_nodes),
      (np.arange(n_input_nodes), np.random.randint(0, dimensionality, n_nodes)),
  )).tocsr()
  return ppr_trans @ proj
