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

"""implementation of least-squares conditional density estimator (multivariate).

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""

import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.basic_operations import compute_pre_coeff
import numpy as np


def multi_lscde_base(datax, datay, base_listx, base_listy, lam):
  """Least-Squares Conditional Density Estimator p(y|x) x is multivariate.

  Args:
    datax: independent samples, ndarray (num data, num features)
    datay: dependent samples, ndarray (num data, num features)
    base_listx: list of list of basis functions of x, [EigenBasse]
    base_listy: list of basis functions of y, [EigenBasse]
    lam: regularization coefficient, float

  Returns:
    alpha: a vector of coefficient
  """

  assert len(base_listy) == len(base_listx)
  # embed data into bases
  # consturct h
  num_bases = len(base_listy)
  hy = np.zeros(len(base_listy))
  for i, f in enumerate(base_listy):
    kernely = f.get_params()['kernel']
    f_data = f.get_params()['data']
    f_gram = kernely(datay, f_data)
    f_weight = f.get_params()['weight']
    embed_y = np.einsum('ij,j->i', f_gram, f_weight, optimize=True)

    hy[i] = np.mean(embed_y)

  n = datax.shape[0]
  temp = np.ones((n, len(base_listx)))
  for i, sub_list in enumerate(base_listx):
    # iterate over each dimension of x
    for j, f in enumerate(sub_list):
      kernelx = f.get_params()['kernel']
      f_data = f.get_params()['data']
      f_gram = kernelx(datax[:, j].reshape(-1, 1), f_data)
      f_weight = f.get_params()['weight']
      embed_x = np.einsum('ij,j->i', f_gram, f_weight, optimize=True)
      temp[:, i] *= embed_x
  hx = np.mean(temp, axis=0)

  h = hx * hy

  # construct Phi

  pre_coeff = compute_pre_coeff(base_listy)
  embed_list = []
  for sub_listx in base_listx:
    embed_1 = np.ones(n)
    # iterate over each dimension of x
    for j, f in enumerate(sub_listx):
      kernelx = f.get_params()['kernel']
      f_data = f.get_params()['data']
      f_gram = kernelx(datax[:, j].reshape(-1, 1), f_data)
      embed_x_f = np.einsum(
          'ij,j->i', f_gram, f.get_params()['weight'], optimize=True
      )
      embed_1 *= embed_x_f
    embed_list.append(embed_1)

  embed_mat = np.array(embed_list)
  phix = np.einsum('ij,kj->ik', embed_mat, embed_mat) / n
  phi = phix * pre_coeff
  inv_phi = np.linalg.solve(phi + lam * np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0.0, i) for i in a])
  return tilde_a


class MultiCDEBase:
  """estimate p(y|x)."""

  def __init__(self, data_x, data_y, base_listx, base_listy, lam):
    # get the coefficient
    tilde_a = multi_lscde_base(data_x, data_y, base_listx, base_listy, lam)
    # consturct the base list

    # create a LSE objects
    base_list = []
    for f, g in zip(base_listx, base_listy):
      base_list.append(f + [g])

    self.cdensity_function = cosde.base.LSEigenBase(base_list, tilde_a)

  def get_density_function(self, new_x, normalize=True):
    """get the density function conditioned on new_x.

    Args:
      new_x: condition point, (,n_features)
      normalize: normalize the pdf to 1, Boolean

    Returns:
      pdf: LSEigenBase object
    """
    base_list = self.cdensity_function.get_params()['base_list']
    x_vec = np.ones(len(base_list))
    len_x = len(base_list[0])
    for i in range(len(base_list)):
      len_x = len(base_list[i])
      for j in range(len_x - 1):
        x_vec[i] *= base_list[i][j].eval(new_x[j].reshape(1, 1))

    coeff = self.cdensity_function.get_params()['coeff']
    new_coeff = coeff * x_vec

    new_base_list = []
    kernel_sum = 0.0
    for i in range(len(base_list)):
      new_base_list.append(base_list[i][len_x - 1])
      sum1 = np.sum(base_list[i][len_x - 1].get_params()['weight'])
      l_kernel = base_list[i][len_x - 1].get_params()['kernel']
      l = l_kernel.get_params()['length_scale']
      sum2 = np.sqrt(2 * np.pi) * l
      cons = sum1 * sum2
      kernel_sum += new_coeff[i] * cons

    if normalize:
      return cosde.base.LSEigenBase(new_base_list, new_coeff / kernel_sum)
    else:
      return cosde.base.LSEigenBase(new_base_list, new_coeff)

  def get_pdf(self, new_x, new_y):
    return self.cdensity_function.eval([new_x, new_y])
