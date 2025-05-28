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

"""Basic operations for the condtional density estimator."""
import copy
import numpy as np


def compute_h(datax, datay, base_listx, base_listy):
  """compute the h function in the paper.

  Args:
    datax: (n_samples, n_features)
    datay: (n_samples, n_features)
    base_listx: list of EigenBase objects
    base_listy: list of EigenBase objects

  Returns:
    h: 1darray, (len(base_listx)*len(base_listy),)
  """

  kernelx = base_listx[0].get_params()['kernel']
  for f in base_listx:
    assert kernelx == f.get_params()['kernel']
  kernely = base_listy[0].get_params()['kernel']
  for g in base_listy:
    assert kernely == g.get_params()['kernel']

  # embed data into bases
  # consturct h
  num_bases = len(base_listx) * len(base_listy)
  h = np.zeros(num_bases)

  for i, f in enumerate(base_listx):
    f_data = f.get_params()['data']
    f_gram = kernelx(datax, f_data)
    embed_x = np.einsum('ij,j->i', f_gram, f.get_params()['weight'],
                        optimize=True)

    for j, g in enumerate(base_listy):
      g_data = g.get_params()['data']
      g_gram = kernely(datay, g_data)
      embed_y = np.einsum('ij,j->i', g_gram, g.get_params()['weight'],
                          optimize=True)
      h[i*len(base_listy) + j] = np.mean(embed_x*embed_y)
  return h


def compute_pre_coeff(base_listy):
  """compute the coefficient matrix.

  Args:
    base_listy: list of EigenBase objects

  Returns:
    pre_coeff: ndarray, (len(base_listy), len(base_listy))
  """
  kernely = base_listy[0].get_params()['kernel']
  for g in base_listy:
    assert kernely == g.get_params()['kernel']

  l = kernely.get_params()['length_scale']
  new_kernely = copy.deepcopy(kernely)
  new_kernely.set_params(length_scale=l*np.sqrt(2))

  pre_coeff = np.zeros((len(base_listy), len(base_listy)))

  for i, f in enumerate(base_listy):
    for j, g in enumerate(base_listy):
      f_data = f.get_params()['data']
      g_data = g.get_params()['data']
      gram_y = new_kernely(f_data, g_data) * l * np.sqrt(np.pi)
      pre_coeff[i, j] = np.einsum('i,jk,k->', f.get_params()['weight'], gram_y,
                                  g.get_params()['weight'], optimize=True)
  return pre_coeff

