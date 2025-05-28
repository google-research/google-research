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

"""implementation of least-squares conditional density estimator.

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""

import cosde
import latent_shift_adaptation.methods.continuous_spectral_method.basic_operations as basic
import numpy as np


def lscde_base(datax, datay, base_listx, base_listy, lam):
  """Least-Squares Conditional Density Estimator p(y|x).

  Args:
    datax: independent samples, ndarray (num data, num features)
    datay: dependent samples, ndarray (num data, num features)
    base_listx: list of basis functions of x, [EigenBasse]
    base_listy: list of basis functions of y, [EigenBasse]
    lam: regularization coefficient, float
  Returns:
    alpha: a vector of coefficient
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

  h = basic.compute_h(datax, datay, base_listx, base_listy)

  # construct Phi
  phi = np.zeros((num_bases, num_bases))
  pre_coeff = basic.compute_pre_coeff(base_listy)

  for i, f in enumerate(base_listx):
    f_data = f.get_params()['data']
    f_gram = kernelx(datax, f_data)
    embed_x_f = np.einsum('ij,j->i', f_gram, f.get_params()['weight'],
                          optimize=True)
    for j, g in enumerate(base_listx):
      g_data = g.get_params()['data']
      g_gram = kernelx(datax, g_data)
      embed_x_g = np.einsum('ij,j->i', g_gram, g.get_params()['weight'],
                            optimize=True)

      index = np.arange(len(base_listy))
      x_id = index + i*len(base_listy)
      y_id = index + j*len(base_listy)
      grid = np.meshgrid(x_id, y_id)
      phi[grid[0], grid[1]] = pre_coeff * np.mean(embed_x_f * embed_x_g)

  inv_phi = np.linalg.solve(phi + lam*np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0., i) for i in a])
  return tilde_a


class CDEBase:
  """estimate p(y|x)."""

  def __init__(self, data_x, data_y, base_listx, base_listy, lam):

    # get the coefficient
    tilde_a = lscde_base(data_x, data_y, base_listx, base_listy, lam)
    # consturct the base list

    # create a LSE objects
    base_list = []
    for f in base_listx:
      for g in base_listy:
        base_list.append([f, g])

    self.cdensity_function = cosde.base.LSEigenBase(base_list, tilde_a)

  def get_density_function(self, new_x, normalize=True):
    """get the density function conditioned on new_x.

    Args:
      new_x: condition point, (,n_features)
      normalize: normalize the pdf to 1, Boolean
    Returns:
      pdf: cosde.base.LSEigenBase object
    """
    base_list = self.cdensity_function.get_params()['base_list']
    x_vec = np.zeros(len(base_list))

    for i in range(len(base_list)):
      x_vec[i] = base_list[i][0].eval(new_x)

    coeff = self.cdensity_function.get_params()['coeff']
    new_coeff = coeff * x_vec

    new_base_list = []
    kernel_sum = 0.
    for i in range(len(base_list)):
      new_base_list.append(base_list[i][1])
      sum1 = np.sum(base_list[i][1].get_params()['weight'])
      l = base_list[i][1].get_params()['kernel'].get_params()['length_scale']
      sum2 = np.sqrt(2*np.pi) * l
      kernel_sum += new_coeff[i] * sum1 * sum2

    if normalize:
      return cosde.base.LSEigenBase(new_base_list, new_coeff / kernel_sum)
    else:
      return cosde.base.LSEigenBase(new_base_list, new_coeff)

  def get_pdf(self, new_x, new_y):
    return self.cdensity_function.eval([new_x, new_y])
