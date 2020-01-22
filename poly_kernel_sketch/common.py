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

""" Common methods shared by MNIST and ImageNet experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import getpass

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


# mkdir -p in Python >2.5
def mkdir_p(path):
  try:
    os.makedirs(path, mode=0o755)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


# Returns path to postfix under user's Unix home directory.
def make_experiment_dir(postfix):
  home = os.path.expanduser('~')
  exp_dir = os.path.join(home, postfix)
  mkdir_p(exp_dir)
  return exp_dir


# appends .png to file name
def save_fig(folder, filename):
  if folder is None:
    return
  filename_out = os.path.join(folder, filename + '.png')
  print('saving {}'.format(filename_out))
  with open(filename_out, 'w') as out_file:
    plt.savefig(out_file)


# appends .txt to file name
def save_array(x, folder, filename, formatting):
  if folder is None:
    return
  filename_out = os.path.join(folder, filename + '.txt')
  print('saving {}'.format(filename_out))
  with open(filename_out, 'w') as out_file:
    np.savetxt(out_file, x, fmt=formatting)


def load_array(filename):
  with open(filename, 'r') as f:
    return np.loadtxt(f)


# count parameters for svd truncation
def count_parameters_list(k_values, nrows, ncols):
  new_list = []
  for k in k_values:
    new_k = count_parameters(k, nrows, ncols)
    new_list.append(new_k)
  return new_list


# number of parameters when nrows-by-ncols matrix is approximated
# with product of nrows-by-rank and rank-by-ncolds matrix.
def count_parameters(rank, nrows, ncols):
  return (nrows + ncols) * rank


# Return one random rademacher matrix
def fully_random_rademacher_matrix(nrows, ncols):
  plus_minus_one = np.array([-1, 1], dtype=np.float32)
  return np.random.choice(plus_minus_one, (nrows, ncols))


# Return a rank-1 Rademacher matrix
def rank1_rademacher(nrows, ncols):
  plus_minus_one = np.array([-1, 1], dtype=np.float32)
  column_vector = np.random.choice(plus_minus_one, (nrows, 1))
  row_vector = np.random.choice(plus_minus_one, (1, ncols))
  # Plain * is quicker than equivalent np.dot(column_vector, row_vector)
  return column_vector * row_vector

# Sketch matrix A
def sketch_matrix(A, sketch_type, k):
  tf.logging.info('sketch_matrix %s %d', sketch_type, k)
  h1 = A.shape[0]
  h2 = A.shape[1]
  # Numpy defaults to int64 or float64 (double precision).
  # Computing with float32 (single precision) is quicker.
  A_hat = np.zeros((h1, h2), dtype=np.float32)
  for i in range(0, k):
    tf.logging.log_every_n(tf.logging.INFO, 'sketch_matrix %s iter %d/%d', 1000,
                           sketch_type, i, k)
    # generate random matrix
    if sketch_type == 'arora':
      mat = fully_random_rademacher_matrix(h1, h2)
    elif sketch_type == 'our_sketch':
      mat = rank1_rademacher(h1, h2)
    else:
      print('wrong sketch_type variable')
      return -1
    # get coefficient
    coefficient = np.dot(np.ravel(A), np.ravel(mat))
    # add coefficient*matrix to A_hat
    A_hat += coefficient * mat
  tf.logging.info('Done sketch_matrix %s %d', sketch_type, k)
  return (1.0 / k) * A_hat


# Return truncated svd of A, where only the top k components are used.
# Adding --copt=-mavx --copt=-mavx2 --copt=-mfma compiler flags
# speeds up svd by almost 2x. However it makes sketching, which is dominant,
# a tiny bit slower and hence it's not worth it.
def truncated_svd(A, k):
  tf.logging.info('Computing SVD ...')
  u, s, v = np.linalg.svd(A, full_matrices=False)
  u_trunc = u[:, 0:k]
  s_trunc = s[0:k]
  v_trunc = v[0:k, :]
  A_hat = np.dot(u_trunc, np.dot(np.diag(s_trunc), v_trunc))
  tf.logging.info('Done computing SVD ...')
  return A_hat

# num_params is rank for SVD, number of coefficients for sketches.
def compress(A, compression_type, num_params):
  if compression_type == 'svd':
    A_hat = truncated_svd(A, num_params)
  elif compression_type == 'our_sketch' or compression_type == 'arora':
    A_hat = sketch_matrix(A, compression_type, num_params)
  else:
    print('Error: wrong compression type. Must be svd, our_sketch, or arora.')
  return A_hat


# return singular values of A sorted in descending order
def singular_values(A):
  u, s, v = np.linalg.svd(A)
  sing = sorted(s, reverse=True)
  return sing

def plot_and_save_singular_values(s, folder, fn, nrows, ncols):
  x = range(1, len(s) + 1)
  y = sorted(s, reverse=True)
  title = 'Singular values\ndim = (' + str(nrows) + 'x' + str(ncols) + ')'
  plt.plot(x, y)
  plt.title(title)
  plt.tight_layout()
  save_fig(folder, fn)
  save_array(np.array(s), folder, fn + '_vals', '%.18e')
