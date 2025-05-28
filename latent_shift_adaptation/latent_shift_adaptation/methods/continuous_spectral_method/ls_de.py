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

"""Implementation of least-squares density estimator.

The density estimator is the variant of ls_cde,
where we marginalize the dependent variable.

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""


import cosde
import latent_shift_adaptation.methods.continuous_spectral_method.basic_operations as basic
import numpy as np


def lsde_base(datax, datay, base_listx, base_listy, lam):
  """Least-Squares Conditional Density Estimator.

  Args:
    datax: dependent samples, ndarray (num data, num features)
    datay: independent samples, ndarray (num data, num features)
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
  pre_coeffy = basic.compute_pre_coeff(base_listy)
  pre_coeffx = basic.compute_pre_coeff(base_listx)

  for i in range(len(base_listx)):
    for j in range(len(base_listx)):
      index = np.arange(len(base_listy))
      x_id = index + i * len(base_listy)
      y_id = index + j * len(base_listy)
      grid = np.meshgrid(x_id, y_id)
      phi[grid[0], grid[1]] = pre_coeffy * pre_coeffx[i, j]

  inv_phi = np.linalg.solve(phi + lam * np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0, i) for i in a])
  return tilde_a


class DEBase:
  """p(x,y)."""

  def __init__(self, data_x, data_y, base_listx, base_listy, lam):
    # get the coefficient
    tilde_a = lsde_base(data_x, data_y, base_listx, base_listy, lam)
    # consturct the base list
    # normalization step
    sum_integral = 0
    # create a LSE objects
    base_list = []
    for i, f in enumerate(base_listx):
      w_sum_f = np.sum(f.get_params()['weight'])
      for j, g in enumerate(base_listy):
        base_list.append([f, g])
        con = np.sqrt(2 * np.pi)
        ks1 = con * f.get_params()['kernel'].get_params()['length_scale']
        ks2 = con * g.get_params()['kernel'].get_params()['length_scale']

        k_s = ks1 * ks2
        w_sum_g = np.sum(g.get_params()['weight'])

        sum_integral += (
            w_sum_g * w_sum_f * k_s * tilde_a[i * len(base_listy) + j]
        )

    self.density_function = cosde.base.LSEigenBase(
        base_list, tilde_a / sum_integral
    )

  def get_pdf(self, new_x, new_y):
    return self.density_function.eval([new_x, new_y])
