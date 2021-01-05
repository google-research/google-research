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

# pylint: disable=g-doc-return-or-yield,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement, g-multiple-import
import numpy as np
from numpy.linalg import eig
import scipy.linalg as la
from es_maml.util.dpp.kernels import cosine_similarity, rbf
from es_maml.util.dpp.utils import sample_k_eigenvecs

# Refer to paper: k-DPPs: Fixed-Size Determinantal Point Processes [ICML 11]


class DPP(object):

  def __init__(self, X=None, A=None):
    self.X = X
    if A:
      self.A = A

  def compute_kernel(self, kernel_type='cos-sim', kernel_func=None, **kwargs):
    if kernel_func is None:
      if kernel_type == 'cos-sim':
        self.A = cosine_similarity(self.X)
      elif kernel_type == 'rbf':
        self.A = rbf(self.X, **kwargs)
    else:
      self.A = kernel_func(self.X, **kwargs)

  def sample(self):

    if not hasattr(self, 'A'):
      self.compute_kernel(kernel_type='cos-sim')

    eigen_vals, eigen_vec = eig(self.A)
    eigen_vals = np.real(eigen_vals)
    eigen_vec = np.real(eigen_vec)
    eigen_vec = eigen_vec.T
    N = self.A.shape[0]

    probs = eigen_vals / (eigen_vals + 1)
    jidx = np.array(np.random.rand(N) <= probs)  # set j in paper

    V = eigen_vec[jidx]  # Set of vectors V in paper
    num_v = len(V)

    Y = []
    while num_v > 0:
      Pr = np.sum(V**2, 0) / np.sum(V**2)
      y_i = np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

      Y.append(y_i)
      V = V.T
      ri = np.argmax(np.abs(V[y_i]) > 0)
      V_r = V[:, ri]

      if num_v > 0:
        V = la.orth(V - np.outer(V_r, (V[y_i, :] / V_r[y_i])))
      V = V.T

      num_v -= 1

    Y.sort()

    out = np.array(Y)

    return out

  def sample_k(self, k=5):

    if not hasattr(self, 'A'):
      self.compute_kernel(kernel_type='cos-sim')

    eigen_vals, eigen_vec = eig(self.A)
    eigen_vals = np.real(eigen_vals)
    eigen_vec = np.real(eigen_vec)
    eigen_vec = eigen_vec.T
    N = self.A.shape[0]

    if k == -1:
      probs = eigen_vals / (eigen_vals + 1)
      jidx = np.array(np.random.rand(N) <= probs)  # set j in paper

    else:
      jidx = sample_k_eigenvecs(eigen_vals, k)

    V = eigen_vec[jidx]  # Set of vectors V in paper
    num_v = len(V)

    Y = []
    while num_v > 0:
      Pr = np.sum(V**2, 0) / np.sum(V**2)
      y_i = np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

      Y.append(y_i)

      V = V.T
      ri = np.argmax(np.abs(V[y_i]) > 0)

      V_r = V[:, ri]

      if num_v > 0:
        V = la.orth(V - np.outer(V_r, (V[y_i, :] / V_r[y_i])))

      V = V.T

      num_v -= 1

    Y.sort()
    out = np.array(Y)

    return out
