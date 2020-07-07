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

# Refer to paper: k-DPPs: Fixed-Size Determinantal Point Processes [ICML 11]


def elem_sympoly(lmbda, k):
  N = len(lmbda)
  E = np.zeros((k + 1, N + 1))
  E[0, :] = 1
  for l in range(1, (k + 1)):
    for n in range(1, (N + 1)):
      E[l, n] = E[l, n - 1] + lmbda[n - 1] * E[l - 1, n - 1]
  return E


def sample_k_eigenvecs(lmbda, k):
  E = elem_sympoly(lmbda, k)
  i = len(lmbda)
  rem = k
  S = []
  while rem > 0:
    if i == rem:
      marg = 1
    else:
      marg = lmbda[i - 1] * E[rem - 1, i - 1] / E[rem, i]

    if np.random.rand() < marg:
      S.append(i - 1)
      rem -= 1
    i -= 1
  S = np.array(S)
  return S
