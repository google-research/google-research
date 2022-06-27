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

"""W Matrix for Scenario/Method 6."""
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

a = np.asarray([[0.4, 0.4, 0.8, 0.8, 0.8, 0.8]])

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

returnW(6, 4, list_ofAis, 100)

# For Method 6
# Optimal solution found.
# [[-9.39363932e-11]
#  [ 4.92100867e-02]
#  [-9.39363922e-11]
#  [-6.45927978e-11]
#  [-6.45927985e-11]
#  [-1.08694588e-10]
#  [-6.45927998e-11]
#  [-6.45927984e-11]
#  [ 1.79125869e-10]
#  [-1.08694583e-10]
#  [-6.45928060e-11]
#  [-6.45928060e-11]
#  [-1.03739874e-10]
#  [-1.03739866e-10]
#  [-1.08694509e-10]
#  [-6.45928070e-11]
#  [-6.45928053e-11]
#  [-1.03739867e-10]
#  [-1.03739868e-10]
#  [ 1.79125836e-10]
#  [-1.08694539e-10]
#  [-2.83255722e-10]
#  [ 9.84201746e-02]
#  [-2.83255704e-10]
#  [-3.89225701e-10]
#  [-3.89225698e-10]
#  [-1.44987895e-10]
#  [-3.89225700e-10]
#  [-3.89225700e-10]
#  [ 1.02892814e-09]
#  [-1.44987885e-10]
#  [-3.89225681e-10]
#  [-3.89225683e-10]
#  [-2.10235717e-10]
#  [-2.10235724e-10]
#  [-1.44987892e-10]
#  [-3.89225679e-10]
#  [-3.89225679e-10]
#  [-2.10235717e-10]
#  [-2.10235720e-10]
#  [ 1.02892805e-09]
#  [-1.44987951e-10]
#  [ 9.53828022e-11]
#  [-1.14296345e-09]
#  [ 9.53827726e-11]
#  [ 2.60040100e-10]
#  [ 2.60040095e-10]
#  [-7.24012486e-11]
#  [ 2.60040101e-10]
#  [ 2.60040100e-10]
#  [-6.70676412e-10]
#  [-7.24012647e-11]
#  [ 2.60040070e-10]
#  [ 2.60040072e-10]
#  [ 2.75597732e-12]
#  [ 2.75597709e-12]
#  [-7.24013340e-11]
#  [ 2.60040066e-10]
#  [ 2.60040063e-10]
#  [ 2.75597093e-12]
#  [ 2.75597533e-12]
#  [-6.70676393e-10]
#  [-7.24012375e-11]
#  [ 3.67647059e-01]
#  [-1.40782560e+00]
#  [ 5.39096636e+00]
#  [-1.47058823e+00]
#  [ 5.63130239e+00]
#  [ 5.88235294e+00]
#  [-1.47058823e+00]
#  [ 5.63130239e+00]
#  [ 5.88235291e+00]
#  [ 5.88235294e+00]]
# 22.44432784266947
