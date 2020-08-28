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

"""Loss computation for nn."""
import functools
import math
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

from bnn_hmc import tree_utils


Batch = Tuple[onp.ndarray, onp.ndarray]
LossAcc = Tuple[jnp.ndarray, jnp.ndarray]
LossAccGrad = Tuple[jnp.ndarray, jnp.ndarray, hk.Params]
PriorFn = Callable[[hk.Params], jnp.array]
LikelihoodFn = Callable[[hk.Transformed, hk.Params, Batch], LossAcc]


def xent_likelihood(net, params,
                    batch):
  """Computes the negative log-likelihood."""
  _, y = batch
  logits = net.apply(params, None, batch)
  labels = jax.nn.one_hot(y, 10)
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))

  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
  return softmax_xent, accuracy


def make_gaussian_prior(weight_decay):
  """Returns the prior function given weight decay."""
  def prior(params):
    """Computes the Gaussian prior negative log-density."""
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    return (0.5 * tree_utils.tree_dot(params, params) * weight_decay +
            0.5 * n_params * jnp.log(weight_decay / (2 * math.pi)))

  return prior


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 2, 3])
def pmap_get_loss_acc_grad(net, params,
                           likelihood_fn,
                           prior_fn, dataset):
  """Computes loss value, accuracy and gradient via pmap."""

  loss_acc_val_grad = jax.value_and_grad(likelihood_fn, has_aux=True, argnums=1)
  (likelihood, acc), likelihood_grad = loss_acc_val_grad(net, params, dataset)
  prior, prior_grad = jax.value_and_grad(prior_fn)(params)

  acc = jax.lax.pmean(acc, axis_name='i')
  likelihood = jax.lax.psum(likelihood, axis_name='i')
  likelihood_grad = jax.lax.psum(likelihood_grad, axis_name='i')

  return likelihood + prior, acc, tree_utils.tree_add(likelihood_grad,
                                                      prior_grad)


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 2, 3])
def pmap_get_loss_and_acc(net, params,
                          likelihood_fn,
                          prior_fn, dataset):
  """Computes loss value and accuracy via pmap."""

  likelihood, acc = likelihood_fn(net, params, dataset)
  prior = prior_fn(params)

  acc = jax.lax.pmean(acc, axis_name='i')
  likelihood = jax.lax.psum(likelihood, axis_name='i')
  prior = jax.lax.pmean(prior, axis_name='i')

  return likelihood + prior, acc


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 3])
def pmap_get_softmax_preds(
    net,
    params,
    dataset,
    num_batches
):
  """Computes predictions via pmap."""

  batch_size = dataset[0].shape[0] // num_batches
  dataset = jax.tree_map(
      lambda x: x.reshape((num_batches, batch_size, *x.shape[1:])), dataset)

  def get_batch_preds(_, x):
    y = net.apply(params, None, x)
    preds = jax.nn.softmax(y)
    return None, preds

  _, preds = jax.lax.scan(get_batch_preds, None, dataset)

  return preds
