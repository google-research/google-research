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

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""W Matrix for Scenario/Method 7."""
import sys
import cvxopt
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def returnW(n, k, listofAis, lambdafactor):  # pylint: disable=invalid-name
  """Returns W matrix."""
  def yuvcoordmap(u, v):
    """Mapping yuv to index."""
    # u, v in 0,...,n-1
    # return u*n + v
    return int((v * (v + 1)) / 2 + u)

  def zbaruvcoordmap(u, v):
    """Mapping zbaruv to index."""
    return int((n * (n + 1)) / 2 + (v * (v + 1)) / 2 + u)

  def zprimeuvcoordmap(u, v):
    """Mapping zprimeuv to index."""
    return int(n * (n + 1) + (v * (v + 1)) / 2 + u)

  def Wijcoordmap(i, j):  # pylint: disable=invalid-name
    """Mapping Wij to index."""
    return int(3 * (n * (n + 1)) / 2 + (j * (j + 1)) / 2 + i)

  # G_0 is minus-indentiy for the first 3n^2 coords

  # G_0 = (-1)*np.eye(N = 3*n^2, M = k^2)

  # h_0 = np.zeros(3*n^2)

  G_list = []  # pylint: disable=invalid-name

  h_list = []  # pylint: disable=invalid-name

  G = []  # pylint: disable=invalid-name

  for v in range(n):
    for u in range(v + 1):
      G_uvfory = np.zeros((k, k))  # pylint: disable=invalid-name
      G_uvforzbar = np.zeros((k, k))  # pylint: disable=invalid-name
      G_uvforzprime = np.zeros((k, k))  # pylint: disable=invalid-name
      G.append(G_uvfory.flatten().tolist())
      G.append(G_uvforzbar.flatten().tolist())
      G.append(G_uvforzprime.flatten().tolist())

  for j in range(k):
    for i in range(j + 1):
      G_ij = np.zeros((k, k))  # pylint: disable=invalid-name
      G_ij[j, i] = -1.0
      G_ij[i, j] = -1.0
      G.append(G_ij.flatten().tolist())

  print("Full G")
  # print(G)

  # print(cvxopt.matrix(G))

  h = np.zeros((k, k))

  G_list.append(cvxopt.matrix(G))

  # print(G_list[0])

  # print("G_list")
  # print(G_list)

  h_list.append(cvxopt.matrix(h))

  # print("h_list")
  # print(h_list)

  c = np.zeros((int(3 * (n * (n + 1)) / 2 + (k * (k + 1)) / 2), 1))

  for v in range(n):
    for u in range(v + 1):
      c[yuvcoordmap(u, v), 0] = lambdafactor

  for i in range(k):
    c[Wijcoordmap(i, i), 0] = 1

  # print("c")
  # print(cvxopt.matrix(c))

  # G_0 and h_0 now

  # G_0 is minus-indentiy for the first 3n^2 coords

  # pylint: disable=invalid-name
  G_0 = cvxopt.matrix((-1) * np.eye(
      N=int(3 * (n * (n + 1)) / 2),
      M=int(3 * (n * (n + 1)) / 2 + (k * (k + 1)) / 2)))

  # G_0 = cvxopt.matrix((-1)*np.diag(c[:,0]))

  # G_0 = cvxopt.matrix((-1)*np.eye(N = 3*(n**2), M = 3*(n**2)+1))

  h_0 = cvxopt.matrix(np.zeros((int((3 * (n * (n + 1)) / 2)), 1)))

  print("G_0 = ")
  # print(G_0)
  # print(G_0.size)
  print("h_0 = ")
  # print(h_0)
  # print(h_0.size)

  # print("G_list = ")
  # print(G_list)
  # print("h_list = ")
  # print(h_list)

  # Construct A, b will be zero: encoding 2 constraints for each u, v

  b = np.zeros((n * (n + 1), 1))

  # pylint: disable=invalid-name
  # pylint: disable=redefined-outer-name
  A = np.zeros(((n * (n + 1)), int(3 * (n * (n + 1)) / 2 + (k * (k + 1)) / 2)))

  for v in range(n):
    for u in range(v + 1):
      # first constraint
      rownumber1 = yuvcoordmap(u, v)
      rownumber2 = int((n * (n + 1)) / 2) + rownumber1
      A[rownumber1, yuvcoordmap(u, v)] = 1
      A[rownumber2, yuvcoordmap(u, v)] = -1

      A[rownumber1, zbaruvcoordmap(u, v)] = -1
      A[rownumber2, zprimeuvcoordmap(u, v)] = 1
      for j in range(k):
        for i in range(j + 1):
          if i == j:
            A[rownumber1, Wijcoordmap(i, j)] = listofAis[i][u, v]
            A[rownumber2, Wijcoordmap(i, j)] = listofAis[i][u, v]
          else:
            A[rownumber1, Wijcoordmap(i, j)] = (listofAis[i][u, u]) * (
                listofAis[j][v, v]) + (listofAis[i][v, v]) * (
                    listofAis[j][u, u])
            A[rownumber2, Wijcoordmap(i, j)] = (listofAis[i][u, u]) * (
                listofAis[j][v, v]) + (listofAis[i][v, v]) * (
                    listofAis[j][u, u])

      if u == v:
        b[rownumber1] = 1
        b[rownumber2] = 1

  print("A = ")
  print(A)
  print(cvxopt.matrix(A))
  print(np.linalg.matrix_rank(A))
  # print("B")
  # print(cvxopt.matrix(b))

  sol = cvxopt.solvers.sdp(
      c=cvxopt.matrix(c),
      Gl=G_0,
      hl=h_0,
      Gs=G_list,
      hs=h_list,
      A=cvxopt.matrix(A),
      b=cvxopt.matrix(b))

  print(np.array(sol["x"]))
  print(sol["primal objective"])


list_ofAis = []

a = np.asarray([[0.2, 0.2, 0.2, 0.2, 0, 0]])

A = np.matmul(np.transpose(a), a)

print(A)

np.fill_diagonal(A, a)

print(A)

list_ofAis.append(A)

a = np.asarray([[0, 0, 0.6, 0.6, 0.6, 0.6]])

A = np.matmul(np.transpose(a), a)

print(A)

np.fill_diagonal(A, a)

print(A)

list_ofAis.append(A)

a = np.asarray([[0.2, 0.2, 0, 0, 0, 0]])

A = np.matmul(np.transpose(a), a)

print(A)

np.fill_diagonal(A, a)

print(A)

list_ofAis.append(A)

a = np.asarray([[0, 0, 0.2, 0.2, 0, 0]])

A = np.matmul(np.transpose(a), a)

print(A)

np.fill_diagonal(A, a)

print(A)

list_ofAis.append(A)

a = np.asarray([[0, 0, 0, 0, 0.2, 0.2]])

A = np.matmul(np.transpose(a), a)

print(A)

np.fill_diagonal(A, a)

print(A)

list_ofAis.append(A)

returnW(6, 5, list_ofAis, 100000)

# # For Method 7
# Optimal solution found.
# [[-3.66339546e-13]
#  [ 1.93243663e-09]
#  [-3.66348659e-13]
#  [-3.66254639e-13]
#  [-3.66256063e-13]
#  [-3.66265359e-13]
#  [-3.66259000e-13]
#  [-3.66253517e-13]
#  [ 6.73062626e-13]
#  [-3.66248103e-13]
#  [-1.00759591e-13]
#  [-1.00759591e-13]
#  [-3.62827814e-13]
#  [-3.62828805e-13]
#  [-3.66262203e-13]
#  [-1.00759590e-13]
#  [-1.00759591e-13]
#  [-3.62826923e-13]
#  [-3.62827261e-13]
#  [ 2.24692091e-10]
#  [-3.66243645e-13]
#  [-3.66282821e-13]
#  [ 3.86911031e-09]
#  [-3.66267355e-13]
#  [-3.63996986e-13]
#  [-3.64003347e-13]
#  [-3.65960071e-13]
#  [-3.63998695e-13]
#  [-3.63999432e-13]
#  [ 3.85067392e-12]
#  [-3.65986244e-13]
#  [-1.70109115e-12]
#  [-1.70109115e-12]
#  [-3.07918288e-13]
#  [-3.07923904e-13]
#  [-3.65501285e-13]
#  [-1.70109115e-12]
#  [-1.70109115e-12]
#  [-3.07922249e-13]
#  [-3.07921118e-13]
#  [ 4.53557640e-10]
#  [-3.65519412e-13]
#  [-3.66423213e-13]
#  [-4.23704086e-12]
#  [-3.66429557e-13]
#  [-3.68516266e-13]
#  [-3.68508516e-13]
#  [-3.66538157e-13]
#  [-3.68510214e-13]
#  [-3.68514955e-13]
#  [-2.50455642e-12]
#  [-3.66529288e-13]
#  [ 1.49957197e-12]
#  [ 1.49957197e-12]
#  [-4.17733819e-13]
#  [-4.17727406e-13]
#  [-3.67026075e-13]
#  [ 1.49957197e-12]
#  [ 1.49957197e-12]
#  [-4.17730873e-13]
#  [-4.17731631e-13]
#  [-4.17346423e-12]
#  [-3.67026513e-13]
#  [ 3.12538585e+00]
#  [-1.04183800e+00]
#  [ 5.95262691e-01]
#  [-3.12499997e+00]
#  [ 1.04172243e+00]
#  [ 3.12461414e+00]
#  [ 1.27994272e-04]
#  [-7.43949693e-01]
#  [-1.67151590e-04]
#  [ 2.23172011e+00]
#  [ 3.12549755e+00]
#  [-1.78574503e+00]
#  [-3.12515082e+00]
#  [ 2.23173641e+00]
#  [ 5.35710596e+00]]
# 14.434303980002149
