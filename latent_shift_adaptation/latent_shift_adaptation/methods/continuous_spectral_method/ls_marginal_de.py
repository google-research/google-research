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

"""implementation of least-squares density estimator.

the marginal density estimator is the variant of ls_cde.

see: Sugiyama, Masashi, et al.
"Conditional density estimation via least-squares density ratio estimation."
JMLR Workshop and Conference Proceedings, 2010.
https://proceedings.mlr.press/v9/sugiyama10a.html
"""

import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.basic_operations import compute_pre_coeff
import numpy as np


def lsmde_base(datax, base_listx, lam):
  """Least-Squares Conditional Density Estimator.

  Args:
    datax: dependent samples, ndarray (num data, num features)
    base_listx: list of basis functions of x, [EigenBasse]
    lam: regularization coefficient, float
  Returns:
    alpha: a vector of coefficient
  """

  kernelx = base_listx[0].get_params()['kernel']
  for f in base_listx:
    assert kernelx == f.get_params()['kernel']

  # embed data into bases
  # consturct h
  num_bases = len(base_listx)

  h = np.zeros(num_bases)

  for i, f in enumerate(base_listx):
    f_data = f.get_params()['data']
    f_gram = kernelx(datax, f_data)
    f_weight = f.get_params()['weight']
    embed_x = np.einsum('ij,j->i', f_gram, f_weight, optimize=True)

    h[i] = np.mean(embed_x)

  # construct Phi

  phi = compute_pre_coeff(base_listx)

  inv_phi = np.linalg.solve(phi+lam*np.eye(num_bases), np.eye(num_bases))
  a = np.einsum('ij,j->i', inv_phi, h, optimize=True)

  tilde_a = np.array([max(0, i) for i in a])
  return tilde_a


class MDEBase:
  """p(y)."""

  def __init__(self, data_x, base_listx, lam):
    # get the coefficient
    tilde_a = lsmde_base(data_x, base_listx, lam)
    # consturct the base list

    # normalization step
    sum_integral = 0
    # create a LSE objects
    base_list = []
    for i, f in enumerate(base_listx):
      base_list.append(f)
      l = f.get_params()['kernel'].get_params()['length_scale']
      k_s = np.sqrt(2 * np.pi) * l
      sum_integral += k_s * np.sum(f.get_params()['weight']) * tilde_a[i]

    self.density_function = cosde.base.LSEigenBase(
        base_list, tilde_a / sum_integral
    )

  def get_pdf(self, new_x):
    return self.density_function.eval(new_x)
