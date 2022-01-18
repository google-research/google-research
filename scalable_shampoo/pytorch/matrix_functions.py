# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

from __future__ import print_function
import torch


@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
  """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1)

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
  v = torch.rand(list(mat_g.shape)[0], device=mat_g.get_device()) * 2 - 1
  error = 1
  iters = 0
  singular_val = 0
  while error > error_tolerance and iters < num_iters:
    v = v / torch.norm(v)
    mat_v = torch.mv(mat_g, v)
    s_v = torch.dot(v, mat_v)
    error = torch.abs(s_v - singular_val)
    v = mat_v
    singular_val = s_v
    iters += 1
  return singular_val, v / torch.norm(v), iters


@torch.no_grad()
def MatPower(mat_m, p):
  """Computes mat_m^p, for p a positive integer.

  Args:
    mat_m: a square matrix
    p: a positive integer

  Returns:
    mat_m^p
  """
  if p in [1, 2, 4, 8, 16, 32]:
    p_done = 1
    res = mat_m
    while p_done < p:
      res = torch.matmul(res, res)
      p_done *= 2
    return res

  power = None
  while p > 0:
    if p % 2 == 1:
      power = torch.matmul(mat_m, power) if power is not None else mat_m
    p //= 2
    mat_m = torch.matmul(mat_m, mat_m)
  return power


@torch.no_grad()
def ComputePower(mat_g, p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
  """A method to compute G^{-1/p} using a coupled Newton iteration.

  See for example equation 3.2 on page 9 of:
  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: A square positive semidefinite matrix
    p: a positive integer
    iter_count: Stop iterating after this many rounds.
    error_tolerance: Threshold for stopping iteration
    ridge_epsilon: We add this times I to G, to make is positive definite.
                   For scaling, we multiply it by the largest eigenvalue of G.
  Returns:
    (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
  """
  shape = list(mat_g.shape)
  if len(shape) == 1:
    return torch.pow(mat_g + ridge_epsilon, -1/p)
  identity = torch.eye(shape[0], device=mat_g.get_device())
  if shape[0] == 1:
    return identity
  alpha = -1.0/p
  max_ev, _, _ = PowerIter(mat_g)
  ridge_epsilon *= max_ev
  mat_g += ridge_epsilon * identity
  z = (1 + p) / (2 * torch.norm(mat_g))
  # The best value for z is
  # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
  #            (c_max^{1+1/p} - c_min^{1+1/p})
  # where c_max and c_min are the largest and smallest singular values of
  # mat_g.
  # The above estimate assumes that c_max > c_min * 2^p
  # Can replace above line by the one below, but it is less accurate,
  # hence needs more iterations to converge.
  # z = (1 + p) / tf.trace(mat_g)
  # If we want the method to always converge, use z = 1 / norm(mat_g)
  # or z = 1 / tf.trace(mat_g), but these can result in many
  # extra iterations.

  mat_root = identity * torch.pow(z, 1.0/p)
  mat_m = mat_g * z
  error = torch.max(torch.abs(mat_m - identity))
  count = 0
  while error > error_tolerance and count < iter_count:
    tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
    new_mat_root = torch.matmul(mat_root, tmp_mat_m)
    mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
    new_error = torch.max(torch.abs(mat_m - identity))
    if new_error > error * 1.2:
      break
    mat_root = new_mat_root
    error = new_error
    count += 1
  return mat_root
