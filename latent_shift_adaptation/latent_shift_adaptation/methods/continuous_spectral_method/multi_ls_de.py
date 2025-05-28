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

"""implementation of least-squares density estimator (x is multivariate).

the density estimator is the variant of ls_cde,
where we marginalize the dependent variable.

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""

import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.basic_operations import compute_pre_coeff
import numpy as np


def multi_lsde_base(data_list, base_lists, lam):
  """Least-Squares Density Estimator.

  Args:
    data_list: list of data
    base_lists: list of list of basis functions, [[EigenBasse]]
    lam: regularization coefficient, float

  Returns:
    alpha: a vector of coefficient
  """

  # embed data into bases
  # consturct h
  num_bases = len(base_lists)

  h_list = []
  for datay, base_listy in zip(data_list, base_lists):
    hy = np.zeros(len(base_listy))
    for i, f in enumerate(base_listy):
      kernely = f.get_params()['kernel']
      f_data = f.get_params()['data']
      f_gram = kernely(datay, f_data)
      f_weight = f.get_params()['weight']
      embed_y = np.einsum('ij,j->i', f_gram, f_weight, optimize=True)

      hy[i] = np.mean(embed_y)
    h_list.append(hy)

  h = np.prod(np.array(h_list), axis=1)

  # construct Phi
  pre_coeff_list = []
  for mode in range(len(base_lists[0])):
    base_listy = [base_lists[i][mode] for i in range(len(base_lists))]
    pre_coeffy = compute_pre_coeff(base_listy)
    pre_coeff_list.append(pre_coeffy)

  phi = np.prod(np.array(pre_coeff_list), axis=0)

  inv_phi = np.linalg.solve(phi + lam * np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0, i) for i in a])
  return tilde_a


class MultiDEBase:
  """p(x,y)."""

  def __init__(self, data_list, base_lists, lam):
    # get the coefficient
    tilde_a = multi_lsde_base(data_list, base_lists, lam)
    # consturct the base list
    # normalization step
    sum_integral = 0
    # create a LSE objects

    for i, sub_list in enumerate(base_lists):
      coeff_x = np.ones(len(sub_list))

      for j, f in enumerate(sub_list):
        con = np.sqrt(2 * np.pi)
        ks1 = con * f.get_params()['kernel'].get_params()['length_scale']
        w_sum_f = np.sum(f.get_params()['weight'])
        coeff_x[j] = ks1 * w_sum_f

      sum_integral += np.prod(coeff_x) * tilde_a[i]

    self.density_function = cosde.base.LSEigenBase(
        base_lists, tilde_a / sum_integral
    )

  def get_pdf(self, new_x, new_y):
    return self.density_function.eval([new_x, new_y])
