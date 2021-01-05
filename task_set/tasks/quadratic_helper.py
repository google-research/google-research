# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# python3
"""Utilities for tasks based off of quadratics."""
import collections
from typing import Optional
import sonnet as snt

from task_set.tasks import base
import tensorflow.compat.v1 as tf
from tensorflow.contrib import stateless as contrib_stateless


class ConstantDistribution(tf.distributions.Distribution):
  """A distribution whose samples are always a constant.

  This is equivalent to a a delta distribution centered at the specified value.
  """

  def __init__(self, value):
    self.value = value

  def sample(self, sample_shape=(), seed=None):
    return tf.ones(sample_shape, dtype=tf.float32) * self.value


class FixedEigenSpectrumMatrixDistribution(tf.distributions.Distribution):
  """A distribution over matrices with a fixed eigenvalue spectrum."""

  def __init__(self, spectrum):
    """Initializer of the distribution over matrices.

    Args:
      spectrum: tf.Tensor vector of length N that specifies the eigenvalue
        spectrum of the generated random matrix.
    """
    if len(spectrum.shape) != 1:
      raise ValueError("spectrum must be a vector!")

    self._spectrum = spectrum

  def sample(self, seed = None):
    """Sample a matrix with the given spectrum.

    Args:
      seed: if seed is set, use a constant random number generator to produce a
        sample, otherwise use built in tensorflow random numbers.

    Returns:
      The sampled matrix.
    """

    dims = self._spectrum.shape[0]
    if seed is not None:
      rand = contrib_stateless.stateless_random_uniform(
          shape=[dims, dims],
          dtype=tf.float32,
          # Arbitrary offset on seed to prevent overlap of random state.
          seed=[seed + 1233, seed + 341]) * 2 - 1
    else:
      rand = tf.random_uniform([dims, dims], -1., 1., dtype=tf.float32)
    q, r = tf.qr(rand, full_matrices=True)

    # Multiply by the sign of the diagonal to ensure a uniform distribution.
    q *= tf.sign(tf.matrix_diag_part(r))

    # qDq^T where D is a diagonal matrix containing the spectrum
    return tf.matmul(tf.matmul(q, tf.diag(self._spectrum)), q, transpose_b=True)


class QuadraticBasedTask(base.BaseTask):
  """A parametric task based on a sampled quadratic.

  See __init__ for the exact form of the loss and more information.
  """

  # disable invalid-names to use math based names (A, B, C).
  # pylint: disable=invalid-name
  def __init__(self,
               dims=10,
               seed=None,
               initial_dist=None,
               A_dist=None,
               A_noise_dist=None,
               B_dist=None,
               B_noise_dist=None,
               C_dist=None,
               C_noise_dist=None,
               grad_noise_dist=None,
               output_fn=None,
               weight_rescale=1.0):
    """Initializer for a sampled quadratic task.

    The loss for this task is described by:
      X = param * weight_rescale
      output_fn((AX-B)^2 + C)
    where param is initialized by:
      param = initial_dist.sample() / weight_rescale

    A, B, C are sampled once at task creation using either a random seed or
    the seed specified by `seed`. Each iteration
      {A, B, C}_noise_dist is sampled and added to the fixed values.

    Gradients are computed using backprop but have additional noise sampled from
    grad_noise_dist added to them.

    Args:
      dims: int Number of dims of base problem.
      seed: optional int Seed passed into sample function of the different
        distributions.
      initial_dist: tf.distributions.Distribution Distribution returning a
        tensor of the same size of dims. This is used as the initial value for
        the task.
      A_dist: tf.distributions.Distribution Distribution over the quadratic
        term. This should return a dims x dims matrix when sampled. This is
        sampled once at task construction.
      A_noise_dist: tf.distributions.Distribution Distribution over noise added
        to the quadratic term.
      B_dist: tf.distribution.Distribution Distribution over the linear term.
        This should be of size dims. This is sampled once at task construction.
      B_noise_dist: tf.distribution.Distribution Distribution over noise added
        to the linear term.
      C_dist: tf.distribution.Distribution Distribution over the scalar term.
        This should be a scalar. This is sampled once at task construction.
      C_noise_dist: tf.distribution.Distribution Distribution over noise added
        to the scalar term.
      grad_noise_dist: tf.distribution.Distribution Distribution over noise
        added to the gradient.
      output_fn: Callable[tf.Tensor, tf.Tensor] Callable applied just before
        returning the loss.
      weight_rescale: float Weight rescaling to change step size dynamics.
    """

    super(QuadraticBasedTask, self).__init__()
    if not A_noise_dist:
      A_noise_dist = ConstantDistribution(0.)

    if not B_noise_dist:
      B_noise_dist = ConstantDistribution(0.)

    if not C_noise_dist:
      C_noise_dist = ConstantDistribution(0.)

    if not grad_noise_dist:
      grad_noise_dist = ConstantDistribution(0.)

    self.A_noise_dist = A_noise_dist
    self.B_noise_dist = B_noise_dist
    self.C_noise_dist = C_noise_dist
    self.grad_noise_dist = grad_noise_dist

    self.output_fn = output_fn

    self.seed = seed
    self.dims = dims
    self.A_dist = A_dist

    self.weight_rescale = weight_rescale

    with self._enter_variable_scope():
      self.initial_dist = initial_dist

      init = initial_dist.sample(seed=seed + 1 if seed else None)
      self.weight = tf.get_variable(
          name="weight", initializer=init, trainable=True)

      A = A_dist.sample(seed=seed + 2 if seed else None)
      self.A = tf.get_variable("A", initializer=A, trainable=False)

      B = B_dist.sample(seed=seed + 3 if seed else None)
      self.B = tf.get_variable("B", initializer=B, trainable=False)

      C = C_dist.sample(seed=seed + 4 if seed else None)
      self.C = tf.get_variable("C", initializer=C, trainable=False)

  @snt.reuse_variables
  def call_split(self, params, split, batch=None, with_metrics=False):
    if batch is None:
      A_noise, B_noise, C_noise = self.get_batch()  # pylint: disable=unbalanced-tuple-unpacking
    else:
      A_noise, B_noise, C_noise = batch

    A = self.A + A_noise
    B = self.B + B_noise
    C = self.C + C_noise

    X = params["weight"] * self.weight_rescale

    X_mat = tf.reshape(X, [-1, 1])
    loss = tf.reduce_sum(tf.square(tf.squeeze(tf.matmul(A, X_mat), 1) - B)) + C

    if self.output_fn:
      loss = self.output_fn(loss)

    @tf.custom_gradient
    def noise_grad(loss, weight):

      def grad(dy):
        return dy, self.grad_noise_dist.sample(
            sample_shape=weight.shape.as_list())

      return loss, grad

    loss = noise_grad(loss, params["weight"])
    if with_metrics:
      return loss, {}
    else:
      return loss

  @snt.reuse_variables
  def current_params(self):
    return collections.OrderedDict([("weight", self.weight)])

  @snt.reuse_variables
  def initial_params(self):
    init = self.initial_dist.sample()
    init = init / self.weight_rescale
    i = collections.OrderedDict([("weight", init)])
    return i

  def with_weight(self, v):
    weight_dict = collections.OrderedDict([("weight", v)])
    return self(weight_dict)

  def get_batch(self, split=None):
    # pylint: disable=g-explicit-bool-comparison
    dists = [(self.A_noise_dist, self.A), (self.B_noise_dist, self.B),
             (self.C_noise_dist, self.C)]
    noises = []
    for noise_dist, dist in dists:
      if noise_dist.batch_shape == ():
        noise = noise_dist.sample(sample_shape=dist.shape.as_list())
      else:
        noise = noise_dist.sample()
      noises.append(noise)
    return tuple(noises)

  def get_variables(self):
    return [self.weight]


# pylint: enable=invalid-name
