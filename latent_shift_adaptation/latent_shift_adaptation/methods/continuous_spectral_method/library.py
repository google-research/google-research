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

"""Implementation of evaluation functions.
"""

import cosde
from latent_shift_adaptation.methods.continuous_spectral_method.create_basis import basis_from_centers
import numpy as np
import pandas as pd
import scipy
from scipy import stats


def get_squeezed_df(data_dict):
  """Converts a dict of numpy arrays into a DataFrame.

  Args:
    data_dict: dict

  Returns:
    Dataframe
  """
  temp = {}
  for key, value in data_dict.items():
    squeezed_array = np.squeeze(value)
    if len(squeezed_array.shape) == 1:
      temp[key] = squeezed_array
    elif len(squeezed_array.shape) > 1:
      for i in range(value.shape[1]):
        temp[f'{key}_{i}'] = np.squeeze(value[:, i])
  return pd.DataFrame(temp)


def print_py_cu(data, ky, kc, ku):
  """print p(Y|C,U).

  Args:
    data: input data, ndarray
    ky: dimension of Y, int
    kc: dimension of C, int
    ku: dimension of U, int

  Returns:
    p(Y|C,U): ndarry
  """
  for i in range(ky):
    test_data = data[data['y'] == i]
    other_data = data[data['y'] != i]
    for j in range(kc):
      for k in range(ku):
        p = test_data[(test_data['c'] == j) & (test_data['u'] == k)].shape[0]
        q = other_data[(other_data['c'] == j) & (other_data['u'] == k)].shape[0]
        print('p(Y={}|C={}, U={})'.format(i, j, k), p / (p + q))


def print_py__multi_cu(data, ky, kc, ku):
  """print p(Y|C,U) when C is bivariate.

  Args:
    data: input data, ndarray
    ky: dimension of Y, int
    kc: dimension of C, int
    ku: dimension of U, int

  Returns:
    p(Y|C,U): ndarry
  """
  for i in range(ky):
    test_data = data[data['y'] == i]
    other_data = data[data['y'] != i]
    for j in range(kc):
      for u in range(kc):
        for v in range(kc):
          for k in range(ku):
            c1 = test_data['c_0'] == j
            c2 = test_data['c_1'] == u
            c3 = test_data['c_2'] == v
            c4 = test_data['u'] == k

            p = test_data[c1 & c2 & c3 & c4].shape[0]
            oc1 = other_data['c_0'] == j
            oc2 = other_data['c_1'] == u
            oc3 = other_data['c_2'] == v
            oc4 = other_data['u'] == k
            q = other_data[oc1 & oc2 & oc3 & oc4].shape[0]
            if p + q > 0:
              print(
                  'p(Y={}|C=({}{}{}),U={})'.format(i, j, u, v, k), p / (p + q)
              )


def get_py_cu_by_state(data, state_y, state_c, state_u):
  """print p(Y=state_y|C=state_c, U=state_u).

  Args:
    data: input data, ndarray
    state_y: the state of y, int
    state_c: the state of c, int
    state_u: the state of u, int

  Returns:
    p(y|c,u): float
  """
  test_data = data[data['y'] == state_y]
  other_data = data[data['y'] != state_y]
  p = test_data[
      (test_data['c'] == state_c) & (test_data['u'] == state_u)
  ].shape[0]
  q = other_data[
      (other_data['c'] == state_c) & (other_data['u'] == state_u)
  ].shape[0]
  return p / (p + q)


def true_p_u_x(x, true_pu, mean_list):
  """compute the true p(U|X) where X is a Gaussian of mixture of two.

  Args:
    x: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    mean_list: the mean of the Gaussian mixtures, list

  Returns:
    p(U|x): ndarray
  """
  p_x = true_pu[0] * stats.norm.pdf(x, mean_list[0], 1)
  p_x += true_pu[1] * stats.norm.pdf(x, mean_list[1], 1)

  pu0x = true_pu[0] * stats.norm.pdf(x, mean_list[0], 1)
  pu1x = true_pu[1] * stats.norm.pdf(x, mean_list[1], 1)
  return [pu0x / p_x, pu1x / p_x]


def true_p_w_x(w, x, true_pu, params):
  """compute the true p(W|X) where X is a Gaussian of mixture of two.

  Args:
    w: value of w, float
    x: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
  """
  p_ux = true_p_u_x(
      x, true_pu, params['mu_x_u_mat'].squeeze() * params['mu_x_u_coeff']
  )
  p_w_u0 = stats.norm.pdf(
      w, params['mu_w_u_mat'].squeeze()[0] * params['mu_w_u_coeff'], 1
  )
  p_w_u1 = stats.norm.pdf(
      w, params['mu_w_u_mat'].squeeze()[1] * params['mu_w_u_coeff'], 1
  )
  p_w_x = p_ux[0] * p_w_u0 + p_ux[1] * p_w_u1

  return p_w_x


def true_p_x(x, true_pu, mean_list):
  """compute the analytical form of p(y|X,u).

  Args:
    x: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    mean_list: the mean of the Gaussian mixtures, list

  Returns:
    p(x): ndarray
  """
  p_x = true_pu[0] * stats.norm.pdf(x, mean_list[0], 1)
  p_x += true_pu[1] * stats.norm.pdf(x, mean_list[1], 1)
  return p_x


def true_p_y_ux(x0, state_u, params):
  """compute the true p(Y|X=x0, U=state_u) where kc = 2.

  Args:
    x0: value of x, float
    state_u: state of U, int
    params: parameter configuration, dict

  Returns:
    p(Y|X=x0, U=state_u): ndarray
  """
  c0 = 0
  c1 = 1
  p_c1_xu = scipy.special.expit(
      params['mu_c_u_mat'][state_u].squeeze() * params['mu_c_u_coeff']
      + params['mu_c_x_mat'][state_u].squeeze() * x0 * params['mu_c_x_coeff']
  )
  p_c0_xu = 1 - p_c1_xu
  p_y_c0u = scipy.special.expit(
      params['mu_y_u_mat'][state_u].squeeze() * params['mu_y_u_coeff']
      + params['mu_y_c_mat'][state_u].squeeze() * c0 * params['mu_y_c_coeff']
  )

  p_y_c1u = scipy.special.expit(
      params['mu_y_u_mat'][state_u].squeeze() * params['mu_y_u_coeff']
      + params['mu_y_c_mat'][state_u].squeeze() * c1 * params['mu_y_c_coeff']
  )

  p_y_xu = p_c1_xu * p_y_c1u + p_c0_xu * p_y_c0u
  return p_y_xu


def true_p_yw_x(y0, w0, x0, true_pu, params):
  """compute p(Y=y0, W=w0|X=x0).

  Args:
    y0: value of Y, int
    w0: value of W, float
    x0: value of X, float
    true_pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(Y=y0, W=w0|X=x0): float
  """
  # compute p_w_u
  p_w0_u0 = stats.norm.pdf(
      w0, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[0], 1
  )
  p_w0_u1 = stats.norm.pdf(
      w0, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[1], 1
  )
  p_w0_u = np.array([p_w0_u0, p_w0_u1])

  # compute p_u_x
  p_u_x0 = true_p_u_x(
      x0, true_pu, params['mu_x_u_coeff'] * params['mu_x_u_mat'].squeeze()
  )

  # compute p_y_ux
  p_y1_u0x0 = true_p_y_ux(x0, 0, params)
  p_y1_u1x0 = true_p_y_ux(x0, 1, params)

  if y0 == 0:
    p_y_u0x0 = 1 - p_y1_u0x0
    p_y_u1x0 = 1 - p_y1_u1x0
  else:
    p_y_u0x0 = p_y1_u0x0
    p_y_u1x0 = p_y1_u1x0
  p_y_ux0 = np.array([p_y_u0x0, p_y_u1x0])
  return np.sum(p_y_ux0 * p_w0_u * p_u_x0)


def true_p_y_x(x, pu, params):
  """compute p(Y|X).

  Args:
    x: value of x, float
    pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(y|x): ndarray
  """
  p_u0_x, p_u1_x = true_p_u_x(
      x, pu, params['mu_x_u_mat'].squeeze() * params['mu_x_u_coeff']
  )
  p_y_xu0 = true_p_y_ux(x, 0, params)
  p_y_xu1 = true_p_y_ux(x, 1, params)
  return p_u0_x * p_y_xu0 + p_u1_x * p_y_xu1


def append_c(count):
  """append c.

  Args:
    count: int

  Returns:
    c_list: list
  """
  if count == 1:
    return [[0], [1]]
  else:
    app_list = append_c(count - 1)
    c_list = []
    for x in app_list:
      c_list.append([0] + x)
      c_list.append([1] + x)
    return c_list


def append_list(count, b_lists):
  """append list.

  Args:
    count: int
    b_lists: list

  Returns:
    c_list: list
  """
  len_b = len(b_lists)
  if count == len_b - 1:
    return [[x] for x in b_lists[len_b - 1]]
  else:
    app_list = append_list(count + 1, b_lists)
    c_list = []

    for f in b_lists[count]:
      for x in app_list:
        c_list.append([f] + x)
    return c_list


def multi_true_p_u_x(x, true_pu, mean_list):
  """compute p(U|X) where X is multivariate Gaussian.

  Args:
    x: values of x, ndarray
    true_pu: pmf of U (Bernoulli RVs), ndarray
    mean_list: mean of the Gaussian of X

  Returns:
    p(U|X=x): ndarray
  """
  dim_p = np.size(mean_list[0])
  p_x = true_pu[0] * stats.multivariate_normal(mean_list[0], np.eye(dim_p)).pdf(
      x
  )
  p_x += true_pu[1] * stats.multivariate_normal(
      mean_list[1], np.eye(dim_p)
  ).pdf(x)
  pu0x = true_pu[0] * stats.multivariate_normal(
      mean_list[0], np.eye(dim_p)
  ).pdf(x)
  pu1x = true_pu[1] * stats.multivariate_normal(
      mean_list[1], np.eye(dim_p)
  ).pdf(x)
  return [pu0x / p_x, pu1x / p_x]


def multi_true_p_w_x(w, x, true_pu, params):
  """compute p(W=w|X=x) where X is multivariate Gaussian.

  Args:
    w: values of w, ndarray
    x: values of x, ndarray
    true_pu: pmf of U (Bernoulli RVs), ndarray
    params: distribution config, dict

  Returns:
    p(W=w,X=x): float
  """
  p_ux = multi_true_p_u_x(
      x, true_pu, (params['mu_x_u_mat'].squeeze() * params['mu_x_u_coeff'])
  )
  p_w_x = p_ux[0] * stats.norm.pdf(
      w, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[0], 1
  )
  p_w_x += p_ux[1] * stats.norm.pdf(
      w, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[1], 1
  )
  return p_w_x


### compute the analytical form of p(y|X,u)
def multi_true_p_x(x, true_pu, mean_list):
  """compute p(X) where X is multivariate Gaussian.

  Args:
    x: values of x, float
    true_pu: pmf of U (Bernoulli RVs), ndarray
    mean_list: mean of the Gaussian of X

  Returns:
    p(X=x): float
  """
  dim_p = np.size(mean_list[0])
  p_x = true_pu[0] * stats.multivariate_normal(mean_list[0], np.eye(dim_p)).pdf(
      x
  )
  p_x += true_pu[1] * stats.multivariate_normal(
      mean_list[1], np.eye(dim_p)
  ).pdf(x)
  return p_x


def multi_true_p_y_ux(x0, state_u, params):
  """compute the probability of p(c|u,x).

  Args:
    x0: value of x, float
    state_u: value of U, int
    params: parameter configuration, dict

  Returns:
    p(Y|U=state_u, X=x0): output, ndarray
  """
  c_logits = x0.dot(params['mu_c_x_coeff'] * params['mu_c_x_mat'])[state_u, :]
  c_logits += params['mu_c_u_mat'][state_u].squeeze() * params['mu_c_u_coeff']
  c_logits = c_logits.reshape(-1, params['k_c'])

  c_prob_list = (
      np.array(
          [1 - scipy.special.expit(c_logits), scipy.special.expit(c_logits)]
      )
      .squeeze()
      .T
  )
  c_list = append_c(params['k_c'])
  out_p = 0
  for c in c_list:
    c_prob = np.prod(c_prob_list[np.arange(params['k_c']), c])
    y_logits = np.array(c).dot(params['mu_y_u_coeff'] * params['mu_y_c_mat'].T)[
        state_u
    ]
    y_logits += params['mu_y_u_mat'][state_u].squeeze() * params['mu_y_u_coeff']

    p_y_c = scipy.special.expit(y_logits)
    out_p += c_prob * p_y_c
  return out_p


def multi_true_p_yw_x(y0, w0, x0, true_pu, params):
  """compute p(Y=y0,W=w0|X=x0).

  Args:
    y0: value of y, int
    w0: value of w, float
    x0: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(Y=y0, W=w0|X=x0)
  """
  # compute p_w_u
  p_w0_u0 = stats.norm.pdf(
      w0, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[0], 1
  )
  p_w0_u1 = stats.norm.pdf(
      w0, params['mu_w_u_coeff'] * params['mu_w_u_mat'].squeeze()[1], 1
  )
  p_w0_u = np.array([p_w0_u0, p_w0_u1])

  # compute p_u_x
  p_u_x0 = multi_true_p_u_x(
      x0, true_pu, params['mu_x_u_coeff'] * params['mu_x_u_mat'].squeeze()
  )

  # compute p_y_ux
  p_y1_u0x0 = multi_true_p_y_ux(x0, 0, params)
  p_y1_u1x0 = multi_true_p_y_ux(x0, 1, params)

  if y0 == 0:
    p_y_u0x0 = 1 - p_y1_u0x0
    p_y_u1x0 = 1 - p_y1_u1x0
  else:
    p_y_u0x0 = p_y1_u0x0
    p_y_u1x0 = p_y1_u1x0
  p_y_ux0 = np.array([p_y_u0x0, p_y_u1x0])
  return np.sum(p_y_ux0 * p_w0_u * p_u_x0)


def multi_true_p_y_x(x, pu, params):
  """compute p(Y|X).

  Args:
    x: value of x, float
    pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(Y|X=x): ndarray, float
  """
  p_u0_x, p_u1_x = multi_true_p_u_x(
      x, pu, params['mu_x_u_mat'].squeeze() * params['mu_x_u_coeff']
  )
  p_y_xu0 = multi_true_p_y_ux(x, 0, params)
  p_y_xu1 = multi_true_p_y_ux(x, 1, params)
  return p_u0_x * p_y_xu0 + p_u1_x * p_y_xu1


def kernel_true_p_yw_x(y0, x0, true_pu, params):
  """compute p(Y=y0,W|x=x0).

  Args:
    y0: value of y, int
    x0: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(y0, W|x0): LSEigenBase
  """
  # compute p_w_u
  pw_u0 = basis_from_centers(params['mu_w_u_mat'].squeeze(), 1)[0]

  pw_u1 = basis_from_centers(params['mu_w_u_mat'].squeeze(), 1)[1]
  # compute p_u_x
  pw_u_list = [pw_u0, pw_u1]

  p_u_x0 = multi_true_p_u_x(
      x0, true_pu, params['mu_x_u_coeff'] * params['mu_x_u_mat'].squeeze()
  )

  # compute p_y_ux
  p_y1_u0x0 = multi_true_p_y_ux(x0, 0, params)
  p_y1_u1x0 = multi_true_p_y_ux(x0, 1, params)

  if y0 == 0:
    p_y_u0x0 = 1 - p_y1_u0x0
    p_y_u1x0 = 1 - p_y1_u1x0
  else:
    p_y_u0x0 = p_y1_u0x0
    p_y_u1x0 = p_y1_u1x0
  coeff = np.array([p_u_x0[0] * p_y_u0x0, p_u_x0[0] * p_y_u1x0]) / np.sqrt(
      2 * np.pi
  )
  return cosde.base.LSEigenBase(pw_u_list, coeff)


def kernel_true_pw_ux0(x0, true_pu, params):
  """compute p(W|U, X=x0).

  Args:
    x0: value of x, float
    true_pu: pmf of a Beroulli random variable, ndarray
    params: parameter configuration, dict

  Returns:
    p(W|U, X=x0): LSEigenBase
  """
  p_u_x0 = multi_true_p_u_x(
      x0, true_pu, params['mu_x_u_coeff'] * params['mu_x_u_mat'].squeeze()
  )
  pw_u = basis_from_centers(params['mu_w_u_mat'].squeeze(), 1)

  # compute p_u_x
  pw_u_list = [
      cosde.base.LSEigenBase(
          pw_u, np.array([p_u_x0[0] / np.sqrt(2 * np.pi), 0.0])
      ),
      cosde.base.LSEigenBase(
          pw_u, np.array([0.0, p_u_x0[1] / np.sqrt(2 * np.pi)])
      ),
  ]
  return pw_u_list
