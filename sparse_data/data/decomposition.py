# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Provides functions for performing dimensionality reduction on features.

Implements truncated singular value decomposition (truncated SVD), linear
discriminant analysis (LDA) and random projection to features.
"""

from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn import random_projection as rand_proj


def truncated_svd(x_train, x_test, num_components=250, num_iter=10):
  """Reduces dimension of features using truncated singular value decomposition.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data
    num_components: number of components
    num_iter: number of iterations

  Returns:
    x_train: np.array
      array of transformed features of training data
    x_test: np.array
      array of transformed features of test data
  """
  dim_red = decomposition.TruncatedSVD(
      n_components=num_components, n_iter=num_iter)
  x_train_svd = dim_red.fit_transform(x_train)
  x_test_svd = dim_red.transform(x_test)

  return x_train_svd, x_test_svd


def lda(x_train, y_train, x_test):
  """Reduces dimension of features using linear discriminant analysis.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    y_train: np.array 1-D array of labels of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data

  Returns:
    x_train: np.array
      array of transformed features of training data
    x_test: np.array
      array of transformed features of test data
  """
  dim_red = discriminant_analysis.LinearDiscriminantAnalysis()
  x_train_lda = dim_red.fit_transform(x_train.toarray(), y_train)
  x_test_lda = dim_red.transform(x_test.toarray())

  return x_train_lda, x_test_lda


def random_projection(x_train, x_test):
  """Reduces dimension of features using random projection.

  Automatically determines number of components.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data

  Returns:
    x_train: np.array
      array of transformed features of training data
    x_test: np.array
      array of transformed features of test data
  """
  dim_red = rand_proj.GaussianRandomProjection()
  x_train_rp = dim_red.fit_transform(x_train)
  x_test_rp = dim_red.transform(x_test)

  return x_train_rp, x_test_rp
