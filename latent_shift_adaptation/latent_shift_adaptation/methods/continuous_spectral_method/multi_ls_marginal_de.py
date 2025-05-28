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

"""implementation of least-squares density estimator (multivariate rvs).

the marginal density estimator is the variant of ls_cde.

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""


import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.basic_operations import compute_pre_coeff
import numpy as np


def multi_lsmde_base(datax, base_listx, lam):
  """Least-Squares Conditional Density Estimator of multivariate.

  Args:
    datax: dependent samples, ndarray (num data, num features)
    base_listx: list of basis functions of x, [EigenBasse]
    lam: regularization coefficient, float
  Returns:
    alpha: a vector of coefficient
  """

  # embed data into bases
  # consturct h
  num_bases = len(base_listx)

  n = datax.shape[0]
  temp = np.ones((n, num_bases))
  for i, sub_list in enumerate(base_listx):
    for j, f in enumerate(sub_list):
      kernelx = f.get_params()['kernel']
      f_data = f.get_params()['data']
      f_gram = kernelx(datax[:, j].reshape(-1, 1), f_data)
      f_weight = f.get_params()['weight']
      embed_x = np.einsum('ij,j->i', f_gram, f_weight, optimize=True)
      temp[:, i] *= embed_x
  h = np.mean(temp, axis=0)

  # construct Phi
  phi = np.ones((num_bases, num_bases))
  for i  in range(len(base_listx[0])):
    sub_list = [base_listx[j][i] for j in range(num_bases)]
    phi *= compute_pre_coeff(sub_list)

  inv_phi = np.linalg.solve(phi+lam*np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0, i) for i in a])
  return tilde_a


class MultiMDEBase:
  """p(y)."""

  def __init__(self, data_x, base_listx, lam):
    # get the coefficient
    tilde_a = multi_lsmde_base(data_x, base_listx, lam)
    # consturct the base list

    # normalization step
    sum_integral = 0
    # create a LSE objects
    base_list = []
    for i, sub_list in enumerate(base_listx):
      base_list.append(sub_list)
      sub_weight = 1.0
      for f in sub_list:
        l = f.get_params()['kernel'].get_params()['length_scale']
        k_s = np.sqrt(2 * np.pi) * l
        sub_weight *= k_s * np.sum(f.get_params()['weight'])
      sum_integral += sub_weight * tilde_a[i]

    self.density_function = cosde.base.LSEigenBase(
        base_list, tilde_a / sum_integral
    )

  def get_pdf(self, new_x):
    return self.density_function.eval(new_x)
