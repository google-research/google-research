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

# Lint as: python2, python3
"""Utilities for implementing UQ methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import sklearn.metrics
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
keras = tf.keras
tfd = tfp.distributions



def tfp_layer_with_scaled_kl(layer_builder, num_train_examples):
  def scaled_kl_fn(q, p, _):
    return tfd.kl_divergence(q, p) / num_train_examples

  return functools.partial(layer_builder, kernel_divergence_fn=scaled_kl_fn)


def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  """Posterior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  variable_layer = tfp.layers.VariableLayer(
      2 * n, dtype=dtype,
      initializer=tfp.layers.BlockwiseInitializer([
          keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
          keras.initializers.Constant(np.log(np.expm1(1e-5)))], sizes=[n, n]))

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                           reinterpreted_batch_ndims=1)
  distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])


def _make_prior_fn(kernel_size, bias_size=0, dtype=None):
  del dtype  # TODO(yovadia): Figure out what to do with this.
  loc = tf.zeros(kernel_size + bias_size)
  def distribution_fn(_):
    return tfd.Independent(tfd.Normal(loc=loc, scale=1),
                           reinterpreted_batch_ndims=1)
  return distribution_fn


def get_layer_builders(method, dropout_rate, num_train_examples):
  """Get method-appropriate functions for building and/or applying Keras layers.

  Args:
    method: UQ method (vanilla, ll_dropout, ll_svi, dropout, svi).
    dropout_rate: Rate for dropout layers. If zero, dropout_fn is a no-op.
    num_train_examples: Number of training examples. Used to scale KL loss.
  Returns:
    conv2d, dense_layer, dense_last, dropout_fn, dropout_fn_last
  """
  layers = keras.layers
  tfpl = tfp.layers

  dense_layer = dense_last = layers.Dense
  conv2d = keras.layers.Conv2D

  conv2d_variational = tfp_layer_with_scaled_kl(tfpl.Convolution2DFlipout,
                                                num_train_examples)
  # Only DenseVariational works in v2 / eager mode.
  # FMI: https://github.com/tensorflow/probability/issues/409
  if tf.executing_eagerly():
    def dense_variational(units, activation):
      return tfpl.DenseVariational(
          units,
          make_posterior_fn=_posterior_mean_field,
          make_prior_fn=_make_prior_fn,
          activation=activation,
          kl_weight=1./num_train_examples)
  else:
    dense_variational = tfp_layer_with_scaled_kl(tfpl.DenseFlipout,
                                                 num_train_examples)

  dropout_normal = lambda x: layers.Dropout(dropout_rate)(x, training=None)
  dropout_always = lambda x: layers.Dropout(dropout_rate)(x, training=True)

  if method == 'svi':
    dense_layer = dense_last = dense_variational
    conv2d = conv2d_variational
  elif method == 'll_svi':
    dense_last = dense_variational

  dropout_fn = dropout_fn_last = dropout_normal
  if dropout_rate == 0:
    dropout_fn = dropout_fn_last = lambda x: x
  elif method == 'dropout':
    dropout_fn = dropout_fn_last = dropout_always
  elif method == 'll_dropout':
    dropout_fn_last = dropout_always

  return conv2d, dense_layer, dense_last, dropout_fn, dropout_fn_last


def make_divergence_fn_for_empirical_bayes(std_prior_scale, examples_per_epoch):
  def divergence_fn(q, p, _):
    log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
    out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
    return out / examples_per_epoch
  return divergence_fn


def make_prior_fn_for_empirical_bayes(init_scale_mean=-1, init_scale_std=0.1):
  """Returns a prior function with stateful parameters for EB models."""
  def prior_fn(dtype, shape, name, _, add_variable_fn):
    """A prior for the variational layers."""
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=(1,),
        initializer=tf.compat.v1.initializers.random_normal(
            mean=init_scale_mean, stddev=init_scale_std),
        dtype=dtype,
        trainable=False)
    loc = add_variable_fn(
        name=name + '_loc',
        initializer=keras.initializers.Zeros(),
        shape=shape,
        dtype=dtype,
        trainable=True)
    scale = 1e-6 + tf.nn.softplus(untransformed_scale)
    dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
  return prior_fn


def np_inverse_softmax(x):
  """Invert a softmax operation over the last axis of a np.ndarray."""
  return np.log(x / x[Ellipsis, :1])


def np_soften_probabilities(probs, epsilon=1e-8):
  """Returns heavily weighted average of categorical distribution and uniform.

  Args:
    probs: Categorical probabilities of shape [num_samples, num_classes].
    epsilon: Small positive value for weighted average.
  Returns:
    epsilon * uniform + (1-epsilon) * probs
  """
  uniform = np.ones_like(probs) / probs.shape[-1]
  return epsilon * uniform + (1-epsilon) * probs


def _make_flatten_unflatten_fns_tf(batch_shape):
  """Returns functions to flatten and unflatten a batch shape."""
  batch_shape = tf.cast(batch_shape, dtype=tf.int32)
  batch_rank = batch_shape.shape[0]
  batch_ndims = tf.reduce_prod(batch_shape)

  @tf.function
  def flatten_fn(x):
    flat_shape = tf.concat([[batch_ndims], tf.shape(x)[batch_rank:]], axis=0)
    return tf.reshape(x, flat_shape)

  @tf.function
  def unflatten_fn(x):
    full_shape = tf.concat([batch_shape, tf.shape(x)[1:]], axis=0)
    return tf.reshape(x, full_shape)
  return flatten_fn, unflatten_fn


