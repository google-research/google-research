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

"""Construct different types of surrogate posteriors for VI."""
import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions
tfp_util = tfp.util

LinearGaussianVariables = collections.namedtuple('LinearGaussianVariables',
                                                 ['matrix', 'loc', 'scale'])


def make_flow_posterior(prior,
                        num_hidden_units,
                        invert=True,
                        num_flow_layers=2):
  """Make a MAF/IAF surrogate posterior.

  Args:
    prior: tfd.JointDistribution instance of the prior.
    num_hidden_units: int value. Specifies the number of hidden units.
    invert: Optional Boolean value. If `True`, produces inverse autoregressive
      flow. If `False`, produces a masked autoregressive flow.
      Default value: `True`.
    num_flow_layers: Optional int value. Specifies the number of layers.
  Returns:
    surrogate_posterior: A `tfd.TransformedDistribution` instance
      whose samples have shape and structure matching that of `prior`.
  """

  event_shape = prior.event_shape_tensor()
  event_space_bijector = prior.experimental_default_event_space_bijector()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = [
      tf.reduce_prod(s) for s in flat_event_shape]

  ndims = tf.reduce_sum(flat_event_size)
  dtype = tf.nest.flatten(prior.dtype)[0]

  make_swap = lambda: tfb.Permute(ps.range(ndims - 1, -1, -1))
  def make_maf():
    net = tfb.AutoregressiveNetwork(
        2,
        hidden_units=[num_hidden_units, num_hidden_units],
        activation=tf.tanh,
        dtype=dtype)

    maf = tfb.MaskedAutoregressiveFlow(
        bijector_fn=lambda x: tfb.Chain([tfb.Shift(net(x)[Ellipsis, 0]),  # pylint: disable=g-long-lambda
                                         tfb.Scale(log_scale=net(x)[Ellipsis, 1])]))
    if invert:
      maf = tfb.Invert(maf)
    # To track the variables
    maf._net = net  # pylint: disable=protected-access
    return maf

  dist = tfd.Sample(
      tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  bijectors = [
      event_space_bijector,
      tfb.Restructure(
          tf.nest.pack_sequence_as(event_shape, range(len(flat_event_shape)))),
      tfb.JointMap(tf.nest.map_structure(tfb.Reshape, flat_event_shape)),
      tfb.Split(flat_event_size),
      ]
  bijectors.append(make_maf())

  for _ in range(num_flow_layers - 1):
    bijectors.extend([make_swap(), make_maf()])

  return tfd.TransformedDistribution(dist, tfb.Chain(bijectors))


def make_mvn_posterior(prior):
  """Build a Multivariate Normal (MVN) posterior.

  Args:
    prior: tfd.JointDistribution instance of the prior.
  Returns:
    surrogate_posterior: A `tfd.TransformedDistribution` instance
    whose samples have shape and structure matching that of `prior`.
  """

  event_shape = prior.event_shape_tensor()
  event_space_bijector = prior.experimental_default_event_space_bijector()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = [
      tf.reduce_prod(s) for s in flat_event_shape]

  ndims = tf.reduce_sum(flat_event_size)

  dtype = tf.nest.flatten(prior.dtype)[0]

  base_dist = tfd.Sample(
      tfd.Normal(tf.zeros([], dtype), 1.), sample_shape=[ndims])
  op = make_trainable_linear_operator_tril(ndims)

  bijectors = [
      event_space_bijector,
      tfb.Restructure(
          tf.nest.pack_sequence_as(event_shape, range(len(flat_event_shape)))),
      tfb.JointMap(tf.nest.map_structure(tfb.Reshape, flat_event_shape)),
      tfb.Split(flat_event_size),
      tfb.Shift(tf.Variable(tf.zeros([ndims], dtype=dtype))),
      tfb.ScaleMatvecLinearOperator(op)]
  return tfd.TransformedDistribution(base_dist, tfb.Chain(bijectors))


def make_trainable_linear_operator_tril(
    dim,
    scale_initializer=1e-1,
    diag_bijector=None,
    diag_shift=1e-5,
    dtype=tf.float32):
  """Build a trainable lower triangular linop."""
  scale_tril_bijector = tfb.FillScaleTriL(
      diag_bijector, diag_shift=diag_shift)
  flat_initial_scale = tf.zeros((dim * (dim + 1) // 2,), dtype=dtype)
  initial_scale_tril = tfb.FillScaleTriL(
      diag_bijector=tfb.Identity(), diag_shift=scale_initializer)(
          flat_initial_scale)
  return tf.linalg.LinearOperatorLowerTriangular(
      tril=tfp_util.TransformedVariable(
          initial_scale_tril, bijector=scale_tril_bijector))


def build_autoregressive_surrogate_posterior(prior, make_conditional_dist_fn):
  """Build a chain-structured surrogate posterior.

  Args:
    prior: JointDistribution instance.
    make_conditional_dist_fn: callable with signature `dist, variables =
      make_conditional_dist_fn(event_shape, x, x_event_shape, variables=None)`
      that builds and returns a trainable distribution over unconstrained
      values, with the specific event shape, conditioned on an input `x`. If
      'variables' is not passed, the necessary variables should be created and
      returned. Passing the returned `variables` structure to future calls
      should replicate the same conditional distribution.
  Returns:
    surrogate_posterior: A `tfd.JointDistributionCoroutineAutoBatched` instance
    whose samples have shape and structure matching that of `prior`.
  """
  with tf.name_scope('build_autoregressive_surrogate_posterior'):

    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name
    trainable_variables = []

    def posterior_generator():
      prior_gen = prior._model_coroutine()  # pylint: disable=protected-access

      previous_value = None
      previous_event_ndims = 0
      previous_dist_was_global = True

      dist = next(prior_gen)

      i = 0
      try:
        while True:
          actual_dist = dist.distribution if isinstance(dist, Root) else dist
          event_shape = actual_dist.event_shape_tensor()

          # Keep global variables out of the chain.
          if previous_dist_was_global:
            previous_value = np.array(0., dtype=np.float32)
            previous_event_ndims = 0

          unconstrained_surrogate, dist_variables = make_conditional_dist_fn(
              y_event_shape=event_shape,
              x=previous_value,
              x_event_ndims=previous_event_ndims,
              variables=(trainable_variables[i]
                         if len(trainable_variables) > i else None))
          # If this is the first run, save the created variables to reuse later.
          if len(trainable_variables) <= i:
            trainable_variables.append(dist_variables)

          surrogate_dist = (
              actual_dist.experimental_default_event_space_bijector()(
                  unconstrained_surrogate))

          if previous_dist_was_global:
            value_out = yield Root(surrogate_dist)
          else:
            value_out = yield surrogate_dist

          previous_value = value_out
          previous_event_ndims = ps.rank_from_shape(event_shape)
          previous_dist_was_global = isinstance(dist, Root)

          dist = prior_gen.send(value_out)
          i += 1
      except StopIteration:
        pass

    surrogate_posterior = tfd.JointDistributionCoroutine(posterior_generator)

    # Build variables.
    _ = surrogate_posterior.sample()

    surrogate_posterior._also_track = trainable_variables  # pylint: disable=protected-access
    return surrogate_posterior


def make_conditional_linear_gaussian(y_event_shape,
                                     x,
                                     x_event_ndims,
                                     variables=None):
  """Build trainable distribution `p(y | x)` conditioned on an input Tensor `x`.

  The distribution is independent Gaussian with mean linearly transformed
  from `x`:
  `y ~ N(loc=matvec(matrix, x) + loc, scale_diag=scale)`

  Args:
    y_event_shape: int `Tensor` event shape.
    x: `Tensor` input to condition on.
    x_event_ndims: int number of dimensions in `x`'s `event_shape`.
    variables: Optional `LinearGaussianVariables` instance, or `None`.
      Default value: `None`.

  Returns:
    dist: Instance of `tfd.Distribution` representing the conditional
      distribution `p(y | x)`.
    variables: Instance of `LinearGaussianVariables` used to parameterize
      `dist`. If a `variables` arg was passed, it is returned unmodified;
      otherwise new variables are created.
  """
  x_shape = ps.shape(x)
  x_ndims = ps.rank_from_shape(x_shape)
  y_event_ndims = ps.rank_from_shape(y_event_shape)
  batch_shape, x_event_shape = (x_shape[:x_ndims - x_event_ndims],
                                x_shape[x_ndims - x_event_ndims:])

  x_event_size = ps.reduce_prod(x_event_shape)
  y_event_size = ps.reduce_prod(y_event_shape)

  x_flat_shape = ps.concat([batch_shape, [x_event_size]], axis=0)
  y_flat_shape = ps.concat([batch_shape, [y_event_size]], axis=0)
  y_full_shape = ps.concat([batch_shape, y_event_shape], axis=0)

  if variables is None:
    variables = LinearGaussianVariables(
        matrix=tf.Variable(
            tf.random.normal(
                ps.concat([batch_shape, [y_event_size, x_event_size]], axis=0),
                dtype=x.dtype),
            name='matrix'),
        loc=tf.Variable(
            tf.random.normal(y_flat_shape, dtype=x.dtype), name='loc'),
        scale=tfp_util.TransformedVariable(
            tf.ones(y_full_shape, dtype=x.dtype),
            bijector=tfb.Softplus(),
            name='scale'))

  flat_x = tf.reshape(x, x_flat_shape)
  dist = tfd.Normal(
      loc=tf.reshape(
          tf.linalg.matvec(variables.matrix, flat_x) + variables.loc,
          y_full_shape),
      scale=variables.scale)
  if y_event_ndims != 0:
    dist = tfd.Independent(dist, reinterpreted_batch_ndims=y_event_ndims)
  dist._also_track = variables  # pylint: disable=protected-access
  return dist, variables

