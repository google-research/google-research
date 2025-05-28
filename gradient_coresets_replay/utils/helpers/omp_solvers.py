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

"""Orthogonal Matching Pursuit sovlvers."""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import lstsq
from scipy.linalg import solve
import torch


# Standard Algorithm, e.g. Tropp,
# "Greed is Good: Algorithmic Results for Sparse Approximation",
# IEEE Trans. Info. Theory, 2004.
def orthogonalmp(mat_a, b, tol=1e-4, nnz=None, positive=False):
  """approximately solves min_x |x|_0 s.t.

  Ax=b using Orthogonal Matching Pursuit

  Args:
    mat_a: design matrix of size (d, n)
    b: measurement vector of length d
    tol: solver tolerance
    nnz: maximum number of nonzero coefficients (if None set to n)
    positive: only allow positive nonzero coefficients

  Returns:
     vector of length n
  """

  mat_at = mat_a.T
  _, n = mat_a.shape
  if nnz is None:
    nnz = n
  x = np.zeros(n)
  resid = np.copy(b)
  normb = norm(b)
  indices = []
  x_i = []
  for _ in range(nnz):
    if norm(resid) / normb < tol:
      break
    projections = mat_at.dot(resid)
    if positive:
      index = np.argmax(projections)
    else:
      index = np.argmax(abs(projections))
    if index in indices:
      break
    indices.append(index)
    mat_ai = None
    if len(indices) == 1:
      mat_ai = mat_a[:, index]
      x_i = projections[index] / mat_ai.T.dot(mat_ai)
    else:
      mat_ai = np.vstack([mat_ai, mat_a[:, index]])
      x_i = solve(mat_ai.dot(mat_ai.T), mat_ai.dot(b), assume_a='sym')
      if positive:
        while min(x_i) < 0.0:
          argmin = np.argmin(x_i)
          indices = indices[:argmin] + indices[argmin + 1 :]
          mat_ai = np.vstack([mat_ai[:argmin], mat_ai[argmin + 1 :]])
          x_i = solve(mat_ai.dot(mat_ai.T), mat_ai.dot(b), assume_a='sym')
    resid = b - mat_ai.T.dot(x_i)

  for i, index in enumerate(indices):
    try:
      x[index] += x_i[i]
    except IndexError:
      x[index] += x_i
  return x


# Standard Algorithm, e.g. Tropp,
# "Greed is Good: Algorithmic Results for Sparse Approximation",
# IEEE Trans. Info. Theory, 2004.
def orthogonalmp_reg(mat_a, b, nnz=None, positive=False, lam=1):
  """approximately solves min_x |x|_0 s.t.

  Ax=b using Orthogonal Matching Pursuit

  Args:
    mat_a: design matrix of size (d, n)
    b: measurement vector of length d
    nnz: maximum number of nonzero coefficients (if None set to n)
    positive: only allow positive nonzero coefficients
    lam: regularization factor

  Returns:
     vector of length n
  """

  mat_at = mat_a.T
  _, n = mat_a.shape
  if nnz is None:
    nnz = n
  x = np.zeros(n)
  resid = np.copy(b)
  # normb = norm(b)
  indices = []
  x_i = []
  while len(indices) < nnz:
    # if resid.norm().item() / normb < tol:
    #     break
    projections = mat_at.dot(resid)
    if positive:
      index = np.argmax(projections)
    else:
      index = np.argmax(abs(projections))
    if index in indices:
      break
    indices.append(index)
    mat_ai = None
    if len(indices) == 1:
      mat_ai = mat_a[:, index]
      x_i = projections[index] / mat_ai.T.dot(mat_ai)
    else:
      mat_ai = np.vstack([mat_ai, mat_a[:, index]])
      x_i = lstsq(
          mat_ai.dot(mat_ai.T) + lam * np.identity(mat_ai.shape[0]),
          mat_ai.dot(b),
      )[0]
      # print(x_i.shape)
      if positive:
        while min(x_i) < 0.0:
          # print("Negative",b.shape,mat_ai.T.shape,x_i.shape)
          argmin = np.argmin(x_i)
          indices = indices[:argmin] + indices[argmin + 1 :]
          mat_ai = np.vstack([mat_ai[:argmin], mat_ai[argmin + 1 :]])
          x_i = lstsq(
              mat_ai.dot(mat_ai.T) + lam * np.identity(mat_ai.shape[0]),
              mat_ai.dot(b),
          )[0]
        # print(x_i)
    # print(b.shape,mat_ai.T.shape,x_i.shape)
    resid = b - mat_ai.T.dot(x_i)

  for i, index in enumerate(indices):
    try:
      x[index] += x_i[i]
    except IndexError:
      x[index] += x_i
  return x


# Standard Algorithm, e.g. Tropp,
# "Greed is Good: Algorithmic Results for Sparse Approximation",
# IEEE Trans. Info. Theory, 2004.
def orthogonalmp_reg_parallel(
    mat_a, b, nnz=None, positive=False, lam=1, device='cpu'
):
  """approximately solves min_x |x|_0 s.t.

  Ax=b using Orthogonal Matching Pursuit

  Args:
    mat_a: design matrix of size (d, n)
    b: measurement vector of length d
    nnz: maximum number of nonzero coefficients (if None set to n)
    positive: only allow positive nonzero coefficients
    lam: regularization factor
    device: device to run on

  Returns:
     vector of length n
  """
  mat_at = torch.transpose(mat_a, 0, 1)
  _, n = mat_a.shape
  if nnz is None:
    nnz = n
  x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
  resid = b.detach().clone()
  # normb = b.norm().item()
  indices = []

  argmin = torch.tensor([-1])
  x_i = torch.tensor([-1])
  while len(indices) < nnz:
    # if resid.norm().item() / normb < tol:
    #     break
    projections = torch.matmul(mat_at, resid)  # mat_at.dot(resid)
    # print("Projections",projections.shape)

    if positive:
      index = torch.argmax(projections)
    else:
      index = torch.argmax(torch.abs(projections))

    if index in indices:
      break

    indices.append(index)
    mat_ai = None
    if len(indices) == 1:
      mat_ai = mat_a[:, index]
      x_i = projections[index] / torch.dot(mat_ai, mat_ai).view(
          -1
      )  # mat_ai.T.dot(mat_ai)
      mat_ai = mat_a[:, index].view(1, -1)
    else:
      # print(indices)
      mat_ai = torch.cat(
          (mat_ai, mat_a[:, index].view(1, -1)), dim=0
      )  # np.vstack([mat_ai, mat_a[:,index]])
      temp = torch.matmul(
          mat_ai, torch.transpose(mat_ai, 0, 1)
      ) + lam * torch.eye(mat_ai.shape[0], device=device)
      x_i, _ = torch.lstsq(torch.matmul(mat_ai, b).view(-1, 1), temp)
      # print(x_i.shape)

      if positive:
        while min(x_i) < 0.0:
          argmin = torch.argmin(x_i)
          indices = indices[:argmin] + indices[argmin + 1 :]
          mat_ai = torch.cat(
              (mat_ai[:argmin], mat_ai[argmin + 1 :]), dim=0
          )  # np.vstack([mat_ai[:argmin], mat_ai[argmin+1:]])
          if argmin.item() == mat_ai.shape[0]:
            break
          # print(argmin.item(),mat_ai.shape[0],index.item())
          temp = torch.matmul(
              mat_ai, torch.transpose(mat_ai, 0, 1)
          ) + lam * torch.eye(mat_ai.shape[0], device=device)
          x_i, _ = torch.lstsq(torch.matmul(mat_ai, b).view(-1, 1), temp)

    if argmin.item() == mat_ai.shape[0]:
      break
    # print(b.shape,torch.transpose(mat_ai, 0, 1).shape,x_i.shape,\
    #  torch.matmul(torch.transpose(mat_ai, 0, 1), x_i).shape)
    resid = b - torch.matmul(torch.transpose(mat_ai, 0, 1), x_i).view(
        -1
    )  # mat_ai.T.dot(x_i)
    # print("REsID",resid.shape)

  x_i = x_i.view(-1)
  # print(x_i.shape)
  # print(len(indices))
  for i, index in enumerate(indices):
    # print(i,index,end="\t")
    try:
      x[index] += x_i[i]
    except IndexError:
      x[index] += x_i
  # print(x[indices])
  return x


# Standard Algorithm, e.g. Tropp,
# "Greed is Good: Algorithmic Results for Sparse Approximation",
# IEEE Trans. Info. Theory, 2004.
def matchingpursuit(
    mat_a, b, tol=1e-4, nnz=None, positive=False, orthogonal=False
):
  """approximately solves min_x |x|_0 s.t.

  Ax=b using Matching Pursuit

  Args:
    mat_a: design matrix of size (d, n)
    b: measurement vector of length d
    tol: solver tolerance
    nnz: maximum number of nonzero coefficients (if None set to n)
    positive: only allow positive nonzero coefficients
    orthogonal: use Orthogonal Matching Pursuit (OMP)

  Returns:
     vector of length n
  """

  if orthogonal:
    return orthogonalmp(mat_a, b, tol=tol, nnz=nnz, positive=positive)

  mat_at = mat_a.T
  _, n = mat_a.shape
  if nnz is None:
    nnz = n
  x = np.zeros(n)
  resid = np.copy(b)
  normb = norm(b)
  selected = np.zeros(n, dtype=np.bool)

  for _ in range(nnz):
    if norm(resid) / normb < tol:
      break
    projections = mat_at.dot(resid)
    projections[selected] = 0.0
    if positive:
      index = np.argmax(projections)
    else:
      index = np.argmax(abs(projections))
    atom = mat_at[index]
    coef = projections[index] / norm(mat_a[:, index])
    if positive and coef <= 0.0:
      break
    resid -= coef * atom
    x[index] = coef
    selected[index] = True
  return x


MP = matchingpursuit


def outer_product_cache(mat_x, limit=float('inf')):
  """cache method for computing and storing outer products.

  Args:
    mat_x: matrix of row vectors
    limit: stops storing outer products after cache contains this many elements

  Returns:
    function that computes outer product of row with itself given its index
  """

  cache = {}

  def outer_product(i):
    output = cache.get(i)
    if output is None:
      output = np.outer(mat_x[i], mat_x[i])
      if len(cache) < limit:
        cache[i] = output
    return output

  return outer_product


def binary_line_search(x, dx, f, nsplit=16):
  """computes update coefficient using binary line search.

  Args:
    x: current position
    dx: full step
    f: objective function
    nsplit: how many binary splits to perform when doing line search

  Returns:
    (coefficient, whether any coefficient was found to improve objective)
  """

  obj = f(x)
  alpha = 0.0
  failed = True
  increment = True
  while increment:
    alpha += 0.5
    for i in range(2, nsplit + 1):
      step = x + alpha * dx
      objstep = f(step)
      if objstep < obj:
        alpha += 2.0**-i
        obj = objstep
        failed = False
      else:
        alpha -= 2.0**-i
        increment = False
  return alpha, failed
