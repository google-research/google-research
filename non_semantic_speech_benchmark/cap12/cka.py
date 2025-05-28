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

"""Utilities to compute CKA."""
# pylint:disable=invalid-name

import numpy as np


def linear_gram(X):
  return np.matmul(X, X.T)


def hsic(K, L):
  # X and Y are Nxd
  N = K.shape[0]
  ones_np = np.ones((N, 1))
  H = np.identity(N) - np.matmul(ones_np, ones_np.T) / N
  KH = np.matmul(K, H)
  LH = np.matmul(L, H)
  hsic_np = np.trace(np.matmul(KH, LH)) / (N-1)**2
  return hsic_np


def compute_cka(X, Y):
  K = linear_gram(X)
  L = linear_gram(Y)

  hsic_kl = hsic(K, L)
  hsic_kk = hsic(K, K)
  hsic_ll = hsic(L, L)
  cka = hsic_kl / (np.sqrt(hsic_kk) * np.sqrt(hsic_ll))
  return cka


def model_pair_cka(model1, model2, common_data_dict):
  """Compute CKA between models."""
  X = np.stack(common_data_dict[model1], axis=1)
  Y = np.stack(common_data_dict[model2], axis=1)

  n_layer_X = X.shape[0]
  n_layer_Y = Y.shape[0]

  cka = np.zeros((n_layer_X, n_layer_Y))
  for x in range(n_layer_X):
    print('X:', x, 'of', n_layer_X)
    for y in range(n_layer_Y):
      cka[x, y] = compute_cka(X[x], Y[y])

  return cka
