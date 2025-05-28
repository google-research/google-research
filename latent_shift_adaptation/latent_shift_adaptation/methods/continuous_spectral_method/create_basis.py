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

"""Construction of basis functions."""

import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.utils import gram_schmidt
import numpy as np
import sklearn.cluster
import sklearn.gaussian_process


def basis_from_cluster(data, k, l, select_pos=False):
  """Construct the set of basis functions from kernel function.

  We select the points via kmeans clustering

  Args:
    data: ndarry (num samples, num features)
    k: number of basis, must be smaller than the number of data, int
    l: length-scale of the kernel, float
    select_pos: only returns positive basis function if set True

  Returns:
    out_list: a list of basis functions, [EigenBase]
  """

  # select k centers from data
  kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(data)
  centers = np.sort(kmeans.cluster_centers_.squeeze())

  # generate the basis by kernel function
  base_list = []
  for center in centers:
    kernel = sklearn.gaussian_process.kernels.RBF(l)
    # set the weight to be 1.
    center = np.array(center).reshape(1, -1)
    base_list.append(cosde.base.EigenBase(kernel, center, np.array([1])))

  # run Gram-Schmidt procedure to orthogonalize the matrix
  ortho_list, _ = gram_schmidt(base_list)
  if select_pos:
    out_list = []
    for eb in ortho_list:
      if (eb.get_params()['weight'] >= 0).all():
        out_list.append(eb)
  else:
    out_list = ortho_list
  return out_list


def basis_from_centers(centers, l):
  """Construct the set of basis functions where each function is the RBF kernel of selected points.

  Args:
    centers: list of centers, ndarray
    l: length-scale of the kernel, float

  Returns:
    out_list: a list of basis functions, [EigenBase]
  """
  # centers = np.sort(np.unique(centers))

  # generate the basis by kernel function
  base_list = []
  for center in centers:
    kernel = sklearn.gaussian_process.kernels.RBF(l)
    # set the weight to be 1.
    new_eb = cosde.base.EigenBase(
        kernel, np.array(center).reshape(1, -1), np.array([1])
    )
    base_list.append(new_eb)

  return base_list


def basis_from_gram(data, k, l, select_pos=False):
  """Contrsut the set of basis functions from Gram matrix.

  The coefficeint is the i-th eigenvector of the Gram matrix

  Args:
    data: ndarry (num samples, num features)
    k: number of basis, must be smaller than the number of data, int
    l: length-scale of the kernel, float
    select_pos: only returns positive basis function if set True

  Returns:
    out_list: a list of basis functions, [EigenBase]
  """

  # construct the Gram matrix
  kernel = sklearn.gaussian_process.kernels.RBF(l)
  gram = kernel(data)
  # compute the eigenvectors
  _, vh = np.linalg.eigh(gram)
  select_vh = vh[:, ::-1][:, 0:k]
  # generate the basis by kernel function
  base_list = []
  weight_list = []
  if select_pos:
    for i in range(k):
      if (select_vh[:, i] >= 0.0).all():
        weight_list.append(select_vh[:, i])
  else:
    weight_list = [select_vh[:, i] for i in range(k)]

  for w in weight_list:
    base_list.append(cosde.base.EigenBase(kernel, data, w))
  # run Gram-Schmidt procedure to orthogonalize the matrix
  ortho_list, _ = gram_schmidt(base_list)
  if select_pos:
    out_list = []
    for eb in ortho_list:
      if (eb.get_params()['weight'] >= 0).all() > 0:
        out_list.append(eb)
  else:
    out_list = ortho_list
  return out_list
