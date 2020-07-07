# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, g-multiple-import
import numpy as np
from numpy import dot
from numpy.linalg import norm
import scipy
from scipy.spatial.distance import pdist, squareform


def rbf(X, sigma=0.5):
  pairwise_dists = squareform(pdist(X, 'euclidean'))
  A = scipy.exp(-pairwise_dists**2 / (2. * sigma**2))
  return A


def cosine_similarity(X):
  d = []
  cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))
  for i in range(X.shape[0]):
    td = []
    for j in range(X.shape[0]):
      td.append(cos_sim(X[i], X[j]))
    d.append(td)
  A = np.array(d)
  return A
