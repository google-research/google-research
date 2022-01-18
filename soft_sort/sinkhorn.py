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

# Lint as: python3
"""A Sinkhorn implementation for 1D Optimal Transport.

Sinkhorn algorithm was introduced in 1967 by R. Sinkhorn in the article
"Diagonal equivalence to matrices with prescribed row and column sums." in
The American Mathematical Monthly. It is an iterative algorithm that turns an
input matrix (here the kernel matrix corresponding to transportation costs) into
a matrix with prescribed (a, b) (row, colums) sum marginals by multiplying it on
the left an right by two diagonal matrices.
"""

from typing import Tuple
import gin
import tensorflow.compat.v2 as tf


def center(cost, f, g):
  if f.shape.rank == 2:
    return cost - f[:, :, tf.newaxis] - g[:, tf.newaxis, :]
  elif f.shape.rank == 3:
    return cost[:, :, :, tf.newaxis] - (
        f[:, :, tf.newaxis, :] + g[:, tf.newaxis, :, :])


def softmin(cost, f, g, eps, axis):
  return -eps * tf.reduce_logsumexp(-center(cost, f, g) / eps, axis=axis)


def error(cost, f, g, eps, b):
  b_target = tf.math.reduce_sum(transport(cost, f, g, eps), axis=1)
  return tf.reduce_max((tf.abs(b_target - b) / b)[:])


def transport(cost, f, g, eps):
  return tf.math.exp(-center(cost, f, g) / eps)


def cost_fn(x, y,
            power):
  """A transport cost in the form |x-y|^p and its derivative."""
  # Check if data is 1D.
  if x.shape.rank == 2 and y.shape.rank == 2:
    # If that is the case, it is convenient to use pairwise difference matrix.
    xy_difference = x[:, :, tf.newaxis] - y[:, tf.newaxis, :]
    if power == 1.0:
      cost = tf.math.abs(xy_difference)
      derivative = tf.math.sign(xy_difference)
    elif power == 2.0:
      cost = xy_difference**2.0
      derivative = 2.0 * xy_difference
    else:
      abs_diff = tf.math.abs(xy_difference)
      cost = abs_diff**power
      derivative = power * tf.math.sign(xy_difference) * abs_diff**(power - 1.0)
    return cost, derivative
  # Otherwise data is high dimensional, in form [batch,n,d]. L2 distance used.
  elif x.shape.rank == 3 and y.shape.rank == 3:
    x2 = tf.reduce_sum(x**2, axis=2)
    y2 = tf.reduce_sum(y**2, axis=2)
    cost = (x2[:, :, tf.newaxis] + y2[:, tf.newaxis, :] -
            tf.matmul(x, y, transpose_b=True))**(power / 2)
    derivative = None
    return cost, derivative


@gin.configurable
def sinkhorn_iterations(x,
                        y,
                        a,
                        b,
                        power = 2.0,
                        epsilon = 1e-3,
                        epsilon_0 = 1e-1,
                        epsilon_decay = 0.95,
                        threshold = 1e-2,
                        inner_num_iter = 5,
                        max_iterations = 2000):
  """Runs the Sinkhorn's algorithm from (x, a) to (y, b).

  Args:
   x: Tensor<float>[batch, n, d]: the input point clouds.
   y: Tensor<float>[batch, m, d]: the target point clouds.
   a: Tensor<float>[batch, n, q]: weights of each input point across batch. Note
     that q possible variants can be considered (for parallelism).
     Sums along axis 1 must match that of b to converge.
   b: Tensor<float>[batch, m, q]: weights of each input point across batch. As
     with a, q possible variants of weights can be considered.
   power: (float) the power of the distance for the cost function.
   epsilon: (float) the level of entropic regularization wanted.
   epsilon_0: (float) the initial level of entropic regularization.
   epsilon_decay: (float) a multiplicative factor applied at each iteration
     until reaching the epsilon value.
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
     Sinkhorn iterations.
   inner_num_iter: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead to avoid computational overhead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.

  Returns:
   A 5-tuple containing: the values of the conjugate variables f and g, the
   final value of the entropic parameter epsilon, the cost matrix and the number
   of iterations.
  """
  max_outer_iterations = max_iterations // inner_num_iter
  loga = tf.math.log(a)
  logb = tf.math.log(b)
  cost, d_cost = cost_fn(x, y, power)

  def body_fn(f, g, eps, num_iter):
    for _ in range(inner_num_iter):
      g = eps * logb + softmin(cost, f, g, eps, axis=1) + g
      f = eps * loga + softmin(cost, f, g, eps, axis=2) + f
      eps = tf.math.maximum(eps * epsilon_decay, epsilon)
    return [f, g, eps, num_iter + inner_num_iter]

  def cond_fn(f, g, eps, num_iter):
    return tf.math.reduce_all([
        tf.math.less(num_iter, max_iterations),
        tf.math.reduce_any([
            tf.math.greater(eps, epsilon),
            tf.math.greater(error(cost, f, g, eps, b), threshold)
        ])
    ])

  f, g, eps, iterations = tf.while_loop(
      cond_fn,
      body_fn, [
          tf.zeros_like(loga),
          tf.zeros_like(logb),
          tf.cast(epsilon_0, dtype=x.dtype),
          tf.constant(0, dtype=tf.int32)
      ],
      parallel_iterations=1,
      maximum_iterations=max_outer_iterations + 1)

  return f, g, eps, cost, d_cost, iterations


def transport_implicit_gradients(derivative_cost,
                                 transport_matrix, eps, b, d_p):
  """Application of the transpose of the Jacobians dP/dx and dP/db.

  This is applied to a perturbation of the size of the transport matrix.
  Required to back-propagate through Sinkhorn's output.

  Args:
   derivative_cost: the derivative of the cost function.
   transport_matrix: the obtained transport matrix tensor.
   eps: the value of the entropic regualarization parameter.
   b: the target weights.
   d_p: the perturbation of the transport matrix.

  Returns:
   A list of two tensor that correspond to the application of the transpose
   of dP/dx and dP/db on dP.
  """
  batch_size = tf.shape(b)[0]
  m = tf.shape(b)[1]
  invmargin1 = tf.math.reciprocal(tf.reduce_sum(transport_matrix, axis=2))
  m1 = invmargin1[:, 1:, tf.newaxis] * transport_matrix[:, 1:, :]
  m1 = tf.concat([tf.zeros([tf.shape(m1)[0], 1, tf.shape(m1)[2]]), m1], axis=1)

  invmargin2 = tf.math.reciprocal(tf.reduce_sum(transport_matrix, axis=1))
  m2 = invmargin2[:, :, tf.newaxis] * tf.transpose(transport_matrix, [0, 2, 1])
  eye_m = tf.eye(m, batch_shape=[batch_size])
  schur = eye_m - tf.linalg.matmul(m2, m1)

  def jac_b_p_transpose(d_p):
    """Transposed of the jacobian of the transport w.r.t the target weights."""
    d_p_p = d_p * transport_matrix
    u_f = tf.reduce_sum(d_p_p, axis=2) / eps
    u_g = tf.reduce_sum(d_p_p, axis=1) / eps

    m1_tranpose_u_f = tf.linalg.matvec(m1, u_f, transpose_a=True)
    to_invert = tf.concat(
        [m1_tranpose_u_f[:, :, tf.newaxis], u_g[:, :, tf.newaxis]], axis=2)
    inverses = tf.linalg.solve(tf.transpose(schur, [0, 2, 1]), to_invert)
    inv_m1_tranpose_u_f, inv_u_g = inverses[:, :, 0], inverses[:, :, 1]
    jac_2 = -inv_m1_tranpose_u_f + inv_u_g
    return eps * jac_2 / b

  def jac_x_p_transpose(d_p):
    """Transposed of the jacobian of the transport w.r.t the inputs."""
    d_p_p = d_p * transport_matrix
    c_x = -tf.reduce_sum(derivative_cost * d_p_p, axis=2) / eps
    u_f = tf.math.reduce_sum(d_p_p, axis=2) / eps
    u_g = tf.math.reduce_sum(d_p_p, axis=1) / eps
    m1_tranpose_u_f = tf.linalg.matvec(m1, u_f, transpose_a=True)
    to_invert = tf.concat(
        [m1_tranpose_u_f[:, :, tf.newaxis], u_g[:, :, tf.newaxis]], axis=2)
    inverses = tf.linalg.solve(tf.transpose(schur, [0, 2, 1]), to_invert)
    inv_m1_tranpose_u_f, inv_u_g = inverses[:, :, 0], inverses[:, :, 1]
    jac_1 = u_f + tf.linalg.matvec(
        m2, inv_m1_tranpose_u_f - inv_u_g, transpose_a=True)
    jac_2 = -inv_m1_tranpose_u_f + inv_u_g
    jac_1 = jac_1 * tf.reduce_sum(m1 * derivative_cost, axis=2)
    jac_2 = tf.linalg.matvec(
        tf.transpose(m2, [0, 2, 1]) * derivative_cost, jac_2)
    return c_x + jac_1 + jac_2

  return [jac_x_p_transpose(d_p), jac_b_p_transpose(d_p)]


def autodiff_sinkhorn(x, y, a, b,
                      **kwargs):
  """A Sinkhorn function that returns the transportation matrix.

  This function back-propagates through the computational graph defined by the
  Sinkhorn iterations.

  Args:
   x: [N, n, d] the input batch of points clouds
   y: [N, m, d] the target batch points clouds.
   a: [N, n, q] q probability weight vectors for the input point cloud. The sum
     of all elements of b along axis 1 must match that of a.
   b: [N, m, q] q probability weight vectors for the target point cloud. The sum
     of all elements of b along axis 1 must match that of a.
   **kwargs: additional parameters passed to the sinkhorn algorithm. See
     sinkhorn_iterations for more details.

  Returns:
   A tf.Tensor representing the optimal transport matrix and the regularized OT
   cost.
  """
  f, g, eps, cost, _, _ = sinkhorn_iterations(x, y, a, b, **kwargs)
  return transport(cost, f, g, eps)


def implicit_sinkhorn(x, y, a, b,
                      **kwargs):
  """A Sinkhorn function using the implicit function theorem.

  That is to say differentiating optimality confiditions to recover Jacobians.

  Args:
   x: the input batch of 1D points clouds
   y: the target batch 1D points clouds.
   a: the intput weight of each point in the input point cloud. The sum of all
     elements of b must match that of a to converge.
   b: the target weight of each point in the target point cloud. The sum of all
     elements of b must match that of a to converge.
   **kwargs: additional parameters passed to the sinkhorn algorithm. See
     sinkhorn_iterations for more details.

  Returns:
   A tf.Tensor representing the optimal transport matrix.
  """

  @tf.custom_gradient
  def _aux(x, b):
    """Auxiliary closure to compute custom gradient over x and b."""
    x = tf.stop_gradient(x)
    b = tf.stop_gradient(b)
    f, g, eps, cost, d_cost, _ = sinkhorn_iterations(x, y, a, b, **kwargs)
    # This centering is crucial to ensure Jacobian is invertible.
    # This centering is also assumed in the computation of the
    # transpose-Jacobians themselves.
    to_remove = f[:, 0]
    f = f - to_remove[:, tf.newaxis]
    g = g + to_remove[:, tf.newaxis]
    forward = transport(cost, f, g, eps)

    def grad(d_p):
      return transport_implicit_gradients(d_cost, forward, eps, b, d_p)

    return forward, grad

  return _aux(x, b)


@gin.configurable
def sinkhorn(x,
             y,
             a,
             b,
             implicit = True,
             **kwargs):
  """A Sinkhorn function that returns the transportation matrix.

  This function back-propagates through the computational graph defined by the
  Sinkhorn iterations.

  Args:
   x: the input batch of points clouds
   y: the target batch points clouds.
   a: the intput weight of each point in the input point cloud. The sum of all
     elements of b must match that of a to converge.
   b: the target weight of each point in the target point cloud. The sum of all
     elements of b must match that of a to converge.
   implicit: whether to run the autodiff version of the backprop or the implicit
     computation of the gradient. The implicit version is more efficient in
     terms of both speed and memory, but might be less stable numerically. It
     requires high-accuracy in the computation of the optimal transport itself.
   **kwargs: additional parameters passed to the sinkhorn algorithm. See
     sinkhorn_iterations for more details.

  Returns:
   A tf.Tensor representing the optimal transport matrix.
  """
  if implicit:
    if x.shape.rank == 2:
      return implicit_sinkhorn(x, y, a, b, **kwargs)
    else:
      raise ValueError('`Implicit` not yet implemented for multivariate data')
  return autodiff_sinkhorn(x, y, a, b, **kwargs)


def sinkhorn_divergence(x,
                        y,
                        a,
                        b,
                        only_x_varies = False,
                        **kwargs):
  """A simple implementation of the Sinkhorn divergence.

  This function back-propagates through the computational graph defined by the
  Sinkhorn iterations.

  Args:
   x: [N,n,d] the input batch of multivariate (dimension d) points clouds
   y: [N,m,d] the input batch of multivariate (dimension d) points clouds
   a: [N,n] probability weights per batch
   b: [N,n] probability weights per batch
   only_x_varies: <bool> if only x varies, that flag should be set to True,
     in order to avoid computing the divergence between y and itself.
   **kwargs: additional parameters passed to the sinkhorn algorithm. See
     sinkhorn_iterations for more details.

  Returns:
   A tf.Tensor representing the optimal transport matrix.
  """
  f_xy, g_xy = sinkhorn_iterations(x, y, a, b, **kwargs)[:2]
  f_xx, g_xx = sinkhorn_iterations(x, x, a, a, **kwargs)[:2]
  if only_x_varies:
    return tf.reduce_sum((f_xy - 0.5 * f_xx - 0.5 * g_xx) * a +
                         g_xy * b, axis=1)
  else:
    f_yy, g_yy = sinkhorn_iterations(y, y, b, b, **kwargs)[:2]
    return (tf.reduce_sum((f_xy - 0.5 * f_xx - 0.5 * g_xx) * a, axis=1) +
            tf.reduce_sum((g_xy - 0.5 * f_yy - 0.5 * g_yy) * b, axis=1))
