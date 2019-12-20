# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Computing extreme eigenvalues of implicitly-defined Hessians.

Uses power-iteration methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf
from extrapolation.influence import calculate_influence as ci
from extrapolation.utils import tensor_utils



def iterated_ev(matmul_fn, num_iter, m_size, tol=1e-6, print_freq=20):
  """Estimate largest eigenvalue of the matrix implicitly defined by matmul_fn.

  Args:
    matmul_fn (function): takes a vector, outputs deterministic
      matrix-vector product, for some square matrix M.
    num_iter (int): number of iterations in estimation.
    m_size (int): side length of M.
    tol (float, optional): stopping condition for estimation.
    print_freq (int, optional): how often to print updates.
  Returns:
    eigval (float): estimated largest eigenvalue of M.
    eigvec (vector): accompanying estimated eigenvector.
  """
  eigvec = tf.random.normal((1, m_size))
  curr_eigval = 0

  # By repeatedly taking powers H^i(eigvec), eigvec converges to the
  # largest eigenvector.
  for i in range(num_iter):
    eigvec = matmul_fn(eigvec)
    eigval = tf.math.reduce_euclidean_norm(eigvec)
    eigvec /= eigval
    if i % print_freq == 0: logging.info('Iteration %d: %.3lf', i, eigval)
    if tf.reduce_all(tf.abs(eigval - curr_eigval) < tol):
      logging.info('Breaking after %d iterations', i)
      break
    curr_eigval = eigval
  return eigval, eigvec


def iterated_ev_mean(matmul_fn, num_iter, m_size, tol=1e-6, print_freq=20,
                     burnin=100):
  """Robustly estimate largest eigenvalue of the matrix implicitly defined by matmul_fn.

  This version is fairly robust to stochasticity in matmul_fn.

  Args:
    matmul_fn (function): takes a vector, outputs
        matrix-vector product, for some square matrix M.
    num_iter (int): number of iterations in estimation.
    m_size (int): side length of M.
    tol (float, optional): stopping condition for estimation.
    print_freq (int, optional): how often to print updates.
    burnin (int, optional): how long to iterate before starting accumulation.
  Returns:
    eigval (float): estimated largest eigenvalue of M.
    eigvec (vector): accompanying estimated eigenvector.
  """
  # Initialize a random unit vector.
  eigvec = tf.random.normal((1, m_size))

  # Iterate some number of times to reach a steady state.
  for i in range(burnin):
    eigvec = matmul_fn(eigvec)
    eigval = tf.math.reduce_euclidean_norm(eigvec)
    eigvec /= eigval
    if i % print_freq == 0: logging.info('Iteration %d: %.3lf', i, eigval)
  logging.info('Burn-in finished, estimation beginning')
  # Now average consecutive powers of H^i(eigvec) -- this is an approximation to
  # averaging across many runs of this process.
  running_mean = 0.
  for i in range(num_iter):
    eigvec = matmul_fn(eigvec)
    eigval = tf.math.reduce_euclidean_norm(eigvec)
    eigvec /= eigval

    old_running_mean = running_mean
    running_mean += eigvec
    if i % print_freq == 0:
      logging.info('Iteration %d: %.3lf', i,
                   tf.math.reduce_euclidean_norm(matmul_fn(running_mean / i)))

    if tf.reduce_all(tf.abs(old_running_mean / running_mean - 1.) < tol):
      logging.info('Breaking after %d iterations', i)
      break
  eigvec = running_mean / num_iter
  return eigval, eigvec


def get_matmul_fn_ev(model, itr, loss_fn, grad_fn, map_grad_fn, n_samples=1):
  """Return a function which takes the HVP of model with some vector.

  Args:
    model (Classifier): a classification model.
    itr (Iterator): an iterator with data for Hessian estimation.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
      of a vector of losses.
    n_samples (int, optional): how many samples to take from itr to estimate
      the HVP.
  Returns:
     hessian_vector_product (function): a function taking the Hessian-vector
       product of model with some vector.
  """
  def hessian_vector_product(vec):
    """Takes the Hessian-vector product (HVP) of vec with model.

    Args:
      vec (vector): a (possibly batched) vector.
    Returns:
      flat_v_hvp: a (possibly batched) HVP of H(model) * vec.
    """
    weight_shaped_vector = tensor_utils.reshape_vector_as(model.weights, vec)

    v_hvp_total = 0.
    for _ in range(n_samples):
      v_hvp = ci.hvp(weight_shaped_vector, itr,
                     loss_fn, grad_fn, map_grad_fn)
      flat_v_hvp = tensor_utils.flat_concat(v_hvp)
      v_hvp_total += flat_v_hvp

    return v_hvp_total / float(n_samples)
  return hessian_vector_product


def estimate_scaling(eigvec, matmul_fn, num_samples):
  """Assuming eigvec is an eigenvector of matmul_fn, estimate its eigenvalue.

  Args:
    eigvec (vector): a vector.
    matmul_fn (function): a function defining a matrix-vector product.
    num_samples (int): how many samples to estimate the eigenvalue over.
  Returns:
    total_eigval (float): the estimated eigenvalue for eigvec.
  """
  total_eigval = 0.
  total_eigvec_hvp = 0.
  for _ in range(num_samples):
    eigvec_hvp = matmul_fn(eigvec)
    estimated_eigval = tf.math.reduce_euclidean_norm(eigvec_hvp)
    estimated_eigval = (estimated_eigval if tf.reduce_all(tf.reduce_sum(
        tf.multiply(eigvec, eigvec_hvp)) > 0.) else -estimated_eigval)

    total_eigval += estimated_eigval
    total_eigvec_hvp += eigvec_hvp
  total_eigval /= num_samples
  total_eigvec_hvp /= num_samples
  logging.info('Eigenvector estimation: difference in norms is %.3lf',
               tf.linalg.norm(eigvec - total_eigvec_hvp / total_eigval))
  logging.info('Eigenvector estimation: cosine is %.3lf',
               tensor_utils.cosine_similarity(eigvec,
                                              total_eigvec_hvp / total_eigval))
  return total_eigval


def estimate_largest_ev(model, ev_iters, itr, loss_fn, grad_fn, map_grad_fn,
                        tol=1e-6, n_samples=1, n_vectors=1, n_scaling=1,
                        burnin=100, print_freq=50):
  """Estimate the largest eigenvalue of model's Hessian.

  Args:
    model (Classifier): a classifier.
    ev_iters (int): number of iterations to run estimation for.
    itr (Iterator): iterator to sample data from for estimation.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
      of a vector of losses.
    tol (float, optional): stopping condition for estimation.
    n_samples (int, optional): how many times to sample from itr in each
      matmul_fn evaluation.
    n_vectors (int, optional): how many times to run the iterative eigenvector
      estimation process.
    n_scaling (int, optional): how many times to estimate HVP when estimating
      the scaling factor.
    burnin (int, optional): how long to iterate before starting accumulation.
    print_freq (int, optional): how often to print updates.

  Returns:
    eigval (float): an estimate of the largest eigenvalue of model.
  """
  matmul_fn_ev = get_matmul_fn_ev(model, itr, loss_fn, grad_fn, map_grad_fn,
                                  n_samples=n_samples)
  m_size = sum([tf.size(w) for w in model.weights])
  total_eigvec = 0.
  for _ in range(n_vectors):
    _, eigvec = iterated_ev_mean(matmul_fn_ev, ev_iters, m_size, tol=tol,
                                 print_freq=print_freq, burnin=burnin)
    total_eigvec += eigvec
  total_eigvec /= n_vectors
  return estimate_scaling(total_eigvec, matmul_fn_ev, n_scaling)


def estimate_smallest_ev(largest_ev, model, ev_iters, itr, loss_fn, grad_fn,
                         map_grad_fn, tol=1e-6, n_samples=1, n_vectors=1,
                         n_scaling=1, burnin=100, print_freq=50):
  """Estimate the smallest eigenvalue of model's Hessian.

  This uses the same machinery as estimate_largest_ev - estimating the smallest
  eigenvalue of model by finding the largest eigenvalue of an equivalent
  model, and then converting back.

  Args:
    largest_ev (float): the largest eigenvalue of model.
    model (Classifier): a classifier.
    ev_iters (int): number of iterations to run estimation for.
    itr (Iterator): iterator to sample data from for estimation.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
      of a vector of losses.
    tol (float, optional): stopping condition for estimation.
    n_samples (int, optional): how many times to sample from itr in each
      matmul_fn evaluation.
    n_vectors (int, optional): how many times to run the iterative eigenvector
      estimation process.
    n_scaling (int, optional): how many times to estimate HVP when estimating
      the scaling factor.
    burnin (int, optional): how long to iterate before starting accumulation.
    print_freq (int, optional): how often to print updates.

  Returns:
    eigval (float): an estimate of the smallest eigenvalue of model.
  """
  matmul_fn_ev = get_matmul_fn_ev(model, itr, loss_fn, grad_fn, map_grad_fn,
                                  n_samples=n_samples)
  def matmul_fn_smallest(vec):
    return largest_ev * vec - matmul_fn_ev(vec)
  m_size = sum([tf.size(w) for w in model.weights])
  total_eigvec = 0.
  for _ in range(n_vectors):
    _, eigvec = iterated_ev_mean(matmul_fn_smallest, ev_iters, m_size, tol=tol,
                                 print_freq=print_freq, burnin=burnin)
    total_eigvec += eigvec / n_vectors
  return estimate_scaling(total_eigvec, matmul_fn_ev, n_scaling)
