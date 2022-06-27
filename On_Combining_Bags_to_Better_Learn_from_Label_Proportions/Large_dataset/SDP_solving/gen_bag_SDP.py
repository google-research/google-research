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

"""Solving the SDP for generalized bags method."""
import pathlib
import pickle

import cvxpy as cp
import numpy as np

np.random.seed(1)

k = 5

n = 50

data_dir = (pathlib.Path(__file__).parent /
            "../Dataset_Preprocessing/Dataset/").resolve()

print(str(data_dir))
print()
file_to_read = open(str(data_dir) + "/corr_matrices_C14_bucket", "rb")
read_list_of_corr_matrices = pickle.load(file_to_read)
file_to_read.close()
print(read_list_of_corr_matrices)
stack_of_corr_matrices = np.stack(read_list_of_corr_matrices)

file_to_read = open(str(data_dir) + "/mean_vecs_C14_bucket", "rb")
read_list_of_mean_vecs = pickle.load(file_to_read)
file_to_read.close()
print(read_list_of_mean_vecs)
stack_of_mean_vecs = np.stack(read_list_of_mean_vecs)

W = cp.Variable((k, k), symmetric=True)

Y_upper = cp.Variable((n, n), symmetric=True)
Y_lower = cp.Variable((n, n), symmetric=True)

Lt = np.tril(np.ones((n, n))) + 100 * np.eye(n)

constraints = [W >> 0]
constraints += [Y_upper >= 0]
constraints += [Y_lower >= 0]

for u in range(n):
  for v in range(u, n):
    b_uv = 1 if u == v else 0
    diagonal = stack_of_corr_matrices[:, u, v]
    full_matrix = np.outer(stack_of_mean_vecs[:, v], stack_of_mean_vecs[:, u])
    np.fill_diagonal(full_matrix, diagonal)
    print(full_matrix)
    constraints += [cp.trace(full_matrix @ W) - b_uv >= (-1) * Y_lower[u, v]]
    constraints += [cp.trace(full_matrix @ W) - b_uv <= Y_upper[u, v]]

prob = cp.Problem(
    cp.Minimize(
        cp.trace(W) + 100 * cp.trace(Lt @ Y_upper) +
        100 * cp.trace(Lt @ Y_lower)), constraints)
prob.solve(solver=cp.CVXOPT, verbose=True)

print("The optimal value is", prob.value)
print("A normalized solution W is")
print(W.value / np.trace(W.value))

file_to_write = open(str(data_dir) + "/normalized_W_C14_bucket_cvxopt", "wb")

pickle.dump(W.value / np.trace(W.value), file_to_write)

file_to_write.close()
