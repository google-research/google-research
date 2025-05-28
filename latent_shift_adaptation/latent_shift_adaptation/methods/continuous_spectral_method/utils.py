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

"""utility functions."""

from cosde import base
from cosde import utils
import numpy as np
import scipy


def gram_schmidt(f_list, thre=1e-3):
  """contruct the Gram-Schmidt method for EigenBase object.

  restriction: each Eigenbase object has the same kernel function
  Args:
    f_list: a list of EigenBase objects
    thre: threshold, float

  Returns:
    out_list: a list of orthonormal EigenBase objects
    r_mat: cofficient matrix, ndarray
  """

  # check that each Eigebbase object has the same kernel function
  kernel = f_list[0].get_params()['kernel']

  for _ in f_list:
    assert f_list[0].get_params()['kernel'] == kernel

  # construct the inner product
  out_list = []
  r_mat = np.zeros((len(f_list), len(f_list)))
  for f in f_list:
    append_data = []
    append_weight = []
    for g in out_list:
      weight = g.get_params()['weight'] * (-utils.inner_product_base(f, g))
      data = g.get_params()['data']
      for w, d in zip(weight, data):
        # discard data if the coefficient is smaller than thre
        if np.abs(w) >= thre:
          append_data.append(d)
          append_weight.append(w)

    weight = f.get_params()['weight']
    data = f.get_params()['data']

    for w, d in zip(weight, data):
      if np.abs(w) >= thre:
        append_data.append(d)
        append_weight.append(w)

    new_weight = np.array(append_weight)
    new_data = np.array(append_data).reshape(len(append_data), -1)
    # construct new Eigenbase
    new_eb = base.EigenBase(kernel, new_data, new_weight)

    # normalize the Eigenbase object
    nor_weight = new_weight / utils.l2_norm_base(new_eb)
    new_eb.set_weight(nor_weight)
    out_list.append(new_eb)

  # contruct the upper coefficient matrix
  for i, f in enumerate(f_list):
    for j, g in enumerate(out_list):
      r_mat[j, i] = utils.inner_product_base(f, g)
      if np.abs(r_mat[j, i]) <= thre:
        r_mat[j, i] = 0.0
  return out_list, r_mat


def gram_schmidt_lse(f_list):
  """contruct the Gram-Schmidt method for LSEigenBase object.

  restriction: each Eigenbase object has the same kernel function

  Args:
    f_list: a list of LSEEigenBase objects

  Returns:
    out_list: a list of orthonormal LSEigenBase objects
    r_mat: cofficient matrix, ndarray
  """

  # check that each Eigebbase object has the same kernel function
  # construct the inner product
  out_list = []
  r_mat = np.zeros((len(f_list), len(f_list)))
  for f in f_list:
    append_basis = []
    append_coeff = []
    for g in out_list:
      coeff = g.get_params()['coeff'] * (-utils.inner_product(f, g))
      base_list = g.get_params()['base_list']
      for w, d in zip(coeff, base_list):
        append_basis.append(d)
        append_coeff.append(w)

    coeff = f.get_params()['coeff']
    base_list = f.get_params()['base_list']

    for w, d in zip(coeff, base_list):
      append_basis.append(d)
      append_coeff.append(w)

    new_coeff = np.array(append_coeff)
    # construct new LSEigenbase
    new_eb = base.LSEigenBase(append_basis, new_coeff)

    # normalize the Eigenbase object
    nor_coeff = new_coeff / utils.l2_norm(new_eb).squeeze()

    new_eb.set_coeff(nor_coeff)
    out_list.append(new_eb)

  # contruct the upper coefficient matrix
  for i, f in enumerate(f_list):
    for j, g in enumerate(out_list):
      r_mat[j, i] = utils.inner_product(f, g)

  return out_list, r_mat


def normalize_prob(obj_list):
  """normalize multiple LSEigenBase objects.

  Args:
    obj_list: LSeigenBase object

  Returns:
    sun_coeff: normalize coefficient
  """
  sum_coeff = 0
  for obj in obj_list:
    base_list = obj.get_params()['base_list']
    coeff = obj.get_params()['coeff']
    for i, f in enumerate(base_list):
      kernel_l = f.get_params()['kernel'].get_params()['length_scale']
      sum1 = np.sqrt(2 * np.pi) * kernel_l
      sum2 = coeff[i] * np.sum(f.get_params()['weight'])
      sum_coeff = sum_coeff + sum1 * sum2
  return sum_coeff


def compute_adaggerb_ortho(op_a, op_b, index):
  """compute inverse(A)B, when the coordinates are orthonormal.

  Args:
    op_a: LSEigenBase object
    op_b: LSEigenBase object
    index: dimension index to marginalize, int

  Returns:
    d_mat: weight matrix, ndarray
    x_coor: list of coordinates, [LSEigenBase]
    y_coor: list of coordinates, [LSEigenBase]
  """

  mu_a = op_a.get_params()['coeff']
  mu_b = op_b.get_params()['coeff']

  base_list_a = op_a.get_params()['base_list']
  base_list_b = op_b.get_params()['base_list']

  len_a = len(base_list_a)
  len_b = len(base_list_b)
  mode = len(base_list_a[0])
  d_mat = np.zeros((len_a, len_b))
  for i in range(len_a):
    for j in range(len_b):
      f_a = base_list_a[i][index - 1]
      f_b = base_list_b[i][index - 1]
      coef = mu_b[j] / (1e-8 + mu_a[i])
      d_mat[i, j] = utils.inner_product_base(f_a, f_b) * coef

  x_coor = []
  all_coor = np.arange(mode) + 1
  delete_id = np.where(all_coor == index)
  remain_coor = np.delete(all_coor, delete_id[0])

  for i in range(len_a):
    if mode == 2:
      x_coor.append(base_list_a[i][remain_coor[0] - 1])
    else:
      sub_coor = []
      for j in remain_coor:
        sub_coor.append(base_list_a[i][j - 1])
      x_coor.append(sub_coor)
  y_coor = []
  for i in range(len_b):
    if mode == 2:
      y_coor.append(base_list_b[i][remain_coor[0] - 1])
    else:
      sub_coor = []
      for j in remain_coor:
        sub_coor.append(base_list_b[i][j - 1])
      y_coor.append(sub_coor)
  return d_mat, x_coor, y_coor


def compute_adaggerb_dim2(op_a, op_b):
  """compute inverse(A)B, when the number of mode is 2.

  Args:
    op_a: LSEigenBase object
    op_b: LSEigenBase object

  Returns:
    d_mat: weight matrix, ndarray
    x_coor: list of coordinates, [LSEigenBase]
    y_coor: list of coordinates, [LSEigenBase]
  """

  mu_a = op_a.get_params()['coeff']
  mu_b = op_b.get_params()['coeff']

  base_list_a = op_a.get_params()['base_list']
  base_list_b = op_b.get_params()['base_list']

  len_a = len(base_list_a)
  len_b = len(base_list_b)
  mode = len(base_list_a[0])

  ortho_dict_a = {}

  # run Gram-Schmidt to orthogonalize the operator
  for i in range(mode):
    base_list = [base_list_a[j][i] for j in range(len_a)]
    ortho_list, r_mat = gram_schmidt(base_list)
    ortho_dict_a[i] = {'ortho_list': ortho_list, 'R': r_mat}
  dim_a = ortho_dict_a[0]['R'].shape[0]
  inv_1 = np.linalg.solve(ortho_dict_a[0]['R'], np.eye(dim_a))

  inv_2 = np.linalg.solve(ortho_dict_a[1]['R'], np.eye(dim_a))

  inv_a = np.einsum(
      'ji,j,jk->ik', inv_2, 1.0 / (1e-8 + mu_a), inv_1, optimize=True
  )

  ortho_dict_b = {}
  for i in range(mode):
    base_list = [base_list_b[j][i] for j in range(len_b)]
    ortho_list, r_mat = gram_schmidt(base_list)
    ortho_dict_b[i] = {'ortho_list': ortho_list, 'R': r_mat}
  b = np.einsum(
      'ij,j,kj->ik',
      ortho_dict_b[0]['R'],
      mu_b,
      ortho_dict_b[1]['R'],
      optimize=True,
  )
  d_mat = np.einsum('ij,jk->ik', inv_a, b, optimize=True)

  return d_mat, ortho_dict_a[1]['ortho_list'], ortho_dict_b[1]['ortho_list']


def compute_adaggerb_multi(op_a, op_b):
  """compute inverse(A)B, when the number of mode is greater than 2.

  Args:
    op_a: LSEigenBase object
    op_b: LSEigenBase object

  Returns:
    d_mat: weight matrix, ndarray
    x_coor: list of coordinates, [LSEigenBase]
    y_coor: list of coordinates, [LSEigenBase]
  """

  mu_a = op_a.get_params()['coeff']
  mu_b = op_b.get_params()['coeff']

  base_list_a = op_a.get_params()['base_list']
  base_list_b = op_b.get_params()['base_list']

  len_a = len(base_list_a)
  len_b = len(base_list_b)
  mode = len(base_list_a[0])

  ortho_dict_a = {}

  # run Gram-Schmidt to orthogonalize the operator
  for i in range(mode):
    base_list = [base_list_a[j][i] for j in range(len_a)]
    ortho_list, r_mat = gram_schmidt(base_list)
    ortho_dict_a[i] = {'ortho_list': ortho_list, 'R': r_mat}
  dim_a = ortho_dict_a[0]['R'].shape[0]

  inv_2 = np.linalg.solve(ortho_dict_a[mode - 1]['R'], np.eye(dim_a))

  inv_a = np.einsum(
      'ji,jk->ik', inv_2, np.diag(1.0 / (1e-8 + mu_a)), optimize=True
  )

  ortho_dict_b = {}
  for i in range(mode):
    base_list = [base_list_b[j][i] for j in range(len_b)]
    ortho_list, r_mat = gram_schmidt(base_list)
    ortho_dict_b[i] = {'ortho_list': ortho_list, 'R': r_mat}
  inv_list = []
  for i in range(mode - 1):
    inv = np.linalg.solve(ortho_dict_a[i]['R'], np.eye(dim_a))
    inv_list.append(np.einsum('ij,jk->ik', inv, ortho_dict_b[i]['R']))
  inv_mat = np.prod(np.array(inv_list), axis=0)
  b = np.einsum(
      'ij,kj->ik', np.diag(mu_b), ortho_dict_b[mode - 1]['R'], optimize=True
  )
  d_mat = np.einsum('ij,jm,mk->ik', inv_a, inv_mat, b, optimize=True)
  x_coor = ortho_dict_a[mode - 1]['ortho_list']
  y_coor = ortho_dict_b[mode - 1]['ortho_list']
  return d_mat, x_coor, y_coor


def least_squares(f_wu, f_wx, verbose=False, reuse_gram=False):
  """output a vector of probability distribution.

  Args:
    f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
    f_wx: a LSEigenBase object, f(W|X)
    verbose: boolean, default False
    reuse_gram: resue Gram matrix, default False

  Returns:
    f_ux: ndarray
  """

  k = len(f_wu)

  k_mat = np.zeros((k, k))
  for i in range(k):
    for j in range(k):
      k_mat[i, j] = utils.inner_product(f_wu[i], f_wu[j], reuse_gram)

  y = np.zeros(k)
  for i in range(k):
    y[i] = utils.inner_product(f_wu[i], f_wx, reuse_gram)

  f_ux, _ = scipy.optimize.nnls(k_mat + 1e-5 * np.eye(k_mat.shape[0]), y)
  results = scipy.optimize.lsq_linear(
      k_mat + 1e-5 * np.eye(k_mat.shape[0]), y, (0.0, 1), verbose=0
  )
  if results['status'] > 0:
    f_ux = results['x']
    if verbose:
      print(results)
    return f_ux

  return f_ux


def multi_least_squares(f_wu, f_wx0, f_wx1, reuse_gram=False):
  """solves constrained least-squares.

  min 0.5 x^T H x - c^Tx
  subject to Z_1x <= b_1, Z_2 x = b_2

  Args:
    f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
    f_wx0: a LSEigenBase object, f(W|X0), X0 is the first slice
    f_wx1: a LSEigenBase object, f(W|X1), X1 is the second slice
    reuse_gram: resue Gram matrix, default False

  Returns:
    res_cons['x']: estimated x
  """

  k = len(f_wu)
  tildef_wu, r_mat = gram_schmidt_lse(f_wu)
  a_mat = np.zeros((k * 2, k * 2)) + 1e-2 * np.eye(k * 2)
  a_mat[0:k, 0:k] = r_mat
  a_mat[k::, k::] = r_mat
  h_mat = np.einsum('ji,jk->ik', a_mat, a_mat)
  y0 = np.zeros(k)
  for i in range(k):
    y0[i] = utils.inner_product(tildef_wu[i], f_wx0, reuse_gram)

  y1 = np.zeros(k)
  for i in range(k):
    y1[i] = utils.inner_product(tildef_wu[i], f_wx1, reuse_gram)
  y = np.hstack((y0, y1))

  x0 = np.zeros(k * 2)
  c = np.dot(y, a_mat)
  b_1 = np.hstack((np.zeros(k * 2), np.ones(k * 2)))
  b_2 = np.ones(k)
  z_2 = np.eye(k)
  for _ in range(k - 1):
    z_2 = np.hstack((z_2, np.eye(k)))
  z_1 = -np.eye(2 * k)
  z_1 = np.vstack((z_1, np.eye(2 * k)))

  def loss(x, sign=1):
    return sign * (0.5 * np.dot(x.T, np.dot(h_mat, x)) + np.dot(-c, x))

  def jac(x, sign=1):
    return sign * (np.dot(x.T, h_mat) - c)

  cons1 = {
      'type': 'ineq',
      'fun': lambda x: b_1 - np.dot(z_1, x),
      'jac': lambda x: -z_1,
  }
  cons2 = {
      'type': 'eq',
      'fun': lambda x: b_2 - np.dot(z_2, x),
      'jac': lambda x: -z_2,
  }
  opt = {'disp': False}
  res_cons = scipy.optimize.minimize(
      loss, x0, jac=jac, constraints=[cons1, cons2], method='SLSQP', options=opt
  )

  return res_cons['x']


# the following function implements equation (18) in the appendix
def multi_least_squares_scale(f_wu, f_wx0, f_wx1, pu_x0, reuse_gram=False):
  """solves constrained least-squares, rescale the design matrix with pu_x0.

  min 0.5 x^T H x - c^Tx
  subject to Z_1x <= b_1, Z_2 x = b_2

  Args:
    f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
    f_wx0: a LSEigenBase object, f(W|X0), X0 is the first slice
    f_wx1: a LSEigenBase object, f(W|X1), X1 is the second slice
    pu_x0: p(U|X=x0), ndarray
    reuse_gram: resue Gram matrix, default False

  Returns:
    res_cons['x']: estimated x
  """

  k = len(f_wu)
  tildef_wu, r_mat = gram_schmidt_lse(f_wu)
  a_mat = np.zeros((k * 2, k * 2)) + 1e-2 * np.eye(k * 2)
  a_mat[0:k, 0:k] = r_mat * np.diag(pu_x0)
  a_mat[k::, k::] = r_mat * np.diag(pu_x0)
  h_mat = np.einsum('ji,jk->ik', a_mat, a_mat)
  y0 = np.zeros(k)
  for i in range(k):
    y0[i] = utils.inner_product(tildef_wu[i], f_wx0, reuse_gram)

  y1 = np.zeros(k)
  for i in range(k):
    y1[i] = utils.inner_product(tildef_wu[i], f_wx1, reuse_gram)
  y = np.hstack((y0, y1))

  x0 = np.zeros(k * 2)
  c = np.dot(y, a_mat)
  b_1 = np.hstack((np.zeros(k * 2), np.ones(k * 2)))
  b_2 = np.ones(k)
  z_2 = np.eye(k)
  for _ in range(k - 1):
    z_2 = np.hstack((z_2, np.eye(k)))
  z_1 = -np.eye(2 * k)
  z_1 = np.vstack((z_1, np.eye(2 * k)))

  def loss(x, sign=1):
    return sign * (0.5 * np.dot(x.T, np.dot(h_mat, x)) + np.dot(-c, x))

  def jac(x, sign=1):
    return sign * (np.dot(x.T, h_mat) - c)

  cons1 = {
      'type': 'ineq',
      'fun': lambda x: b_1 - np.dot(z_1, x),
      'jac': lambda x: -z_1,
  }
  cons2 = {
      'type': 'eq',
      'fun': lambda x: b_2 - np.dot(z_2, x),
      'jac': lambda x: -z_2,
  }
  opt = {'disp': False}
  res_cons = scipy.optimize.minimize(
      loss, x0, jac=jac, constraints=[cons1, cons2], method='SLSQP', options=opt
  )

  return res_cons['x']


def least_squares_gram_schmidt(f_wu, f_wx, reuse_gram=False):
  """estimate p(U|x) via least-squares with Gram-schmidt orthogonalization.

  Args:
    f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
    f_wx: a LSEigenBase object, f(W|X)
    reuse_gram: resue Gram matrix, default False

  Returns:
    f_ux: estimated values, ndarray
  """
  k = len(f_wu)

  tildef_wu, r_mat = gram_schmidt_lse(f_wu)
  y = np.zeros(k)
  for i in range(k):
    y[i] = utils.inner_product(tildef_wu[i], f_wx, reuse_gram)
  _, _ = scipy.optimize.nnls(r_mat + 1e-7 * np.eye(r_mat.shape[0]), y)

  results = scipy.optimize.lsq_linear(
      r_mat + 1e-3 * np.eye(r_mat.shape[0]), y, (0.0, 1), verbose=0
  )

  f_ux = results['x']

  return f_ux
