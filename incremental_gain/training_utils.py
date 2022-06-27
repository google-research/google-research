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

# pylint: disable=invalid-name
"""Training utils."""

import functools
import time

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from incremental_gain import jax_utils


def policy_net(x, hidden_width, output_width, activation):
  """Policy network."""
  assert len(x.shape) == 1  # (state_dim,)
  # TODO(stephentu): allow the depth to be customizable?
  mlp = hk.Sequential([
      hk.Linear(hidden_width),
      activation,
      hk.Linear(hidden_width),
      activation,
      hk.Linear(output_width),
  ])
  return mlp(x)


def make_policy_net(hidden_width, output_width, activation):
  return hk.without_apply_rng(
      hk.transform(
          functools.partial(
              policy_net,
              hidden_width=hidden_width,
              output_width=output_width,
              activation=activation)))


@functools.partial(jax.jit, static_argnums=(0, 1))
def mixed_policy_with_expert(policy_network, expert_policy,
                             policy_params, alpha, x):
  """Rollout the mixed policy."""
  logging.info("jit-ing mixed_policy_with_expert")
  n_policies = len(policy_params)
  ret = ((1 - alpha)**n_policies) * expert_policy(x)
  for k, params in enumerate(policy_params):
    this_ret = alpha * (
        (1 - alpha)**(n_policies - k - 1)) * policy_network.apply(params, x)
    ret += this_ret
  return ret


@functools.partial(jax.jit, static_argnums=(0, 1))
def dagger_policy_with_expert(policy_network, expert_policy,
                              policy_params, alpha, x):
  """Rollout the dagger policy."""
  logging.info("jit-ing dagger_policy_with_expert")
  n_policies = len(policy_params)
  expert_weight = (1 - alpha)**n_policies
  return expert_weight * expert_policy(x) + (
      1 - expert_weight) * policy_network.apply(policy_params[-1], x)


@functools.partial(jax.jit, static_argnums=(0,))
def final_policy(policy_network, policy_params, alpha, x):
  """Rollout the final policy."""
  logging.info("jit-ing final_policy")
  n_policies = len(policy_params)
  prefactor = alpha / (1 - ((1 - alpha)**n_policies))

  def weighted_policy(k, params):
    return prefactor * (
        (1 - alpha)**(n_policies - k - 1)) * policy_network.apply(params, x)

  ret = weighted_policy(0, policy_params[0])
  for k_minus_1, params in enumerate(policy_params[1:]):
    ret += weighted_policy(k_minus_1 + 1, params)
  return ret


@functools.partial(jax.jit, static_argnums=(0,))
def dagger_final_policy(policy_network, policy_params, alpha, x):
  del alpha
  return policy_network.apply(policy_params[-1], x)


@functools.partial(jax.jit, static_argnums=(0,))
def imitation_loss_fn(policy_network, policy_params, xs, us,
                      trust_region_params, trust_region_lam):
  """Standard imitation loss."""
  assert len(xs.shape) == 2  # [batch_size, state_dim]
  assert len(us.shape) == 2  # [batch_size, input_dim]
  assert xs.shape[0] == us.shape[0]

  def one_loss(x, u):
    return jax_utils.safe_norm(policy_network.apply(policy_params, x) - u, 1e-8)

  imitation_loss = jnp.sum(jax.vmap(one_loss, in_axes=(0, 0))(xs, us))
  trust_region_loss = trust_region_lam * jnp.sum(
      jnp.square(
          jax.flatten_util.ravel_pytree(policy_params)[0] -
          jax.flatten_util.ravel_pytree(trust_region_params)[0]))
  total_loss = imitation_loss + trust_region_loss
  return total_loss, (imitation_loss, trust_region_loss)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def imitation_loss_with_igs_constraint_fn(dynamics, igs_loss, policy_fn,
                                          policy_network, policy_params,
                                          xs, us,
                                          trust_region_params,
                                          trust_region_lam):
  """Imitation loss with IGS constraint."""
  total_loss, (imitation_loss, trust_region_loss) = imitation_loss_fn(
      policy_network, policy_params, xs, us,
      trust_region_params, trust_region_lam)

  def one_loss(x):
    y = jnp.zeros_like(x)
    fx = dynamics(x, policy_fn(policy_network, policy_params, x))
    fy = dynamics(y, policy_fn(policy_network, policy_params, y))
    return igs_loss(x, y, fx, fy)

  igs_constraint_loss = jnp.sum(jax.vmap(one_loss)(xs))
  total_loss += igs_constraint_loss
  return total_loss, (imitation_loss, trust_region_loss, igs_constraint_loss)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def do_batch(loss_fn, policy_network, opt_update,
             batch_xs, batch_us, policy_params,
             opt_state, trust_region_params, trust_region_lam):
  """Do one batch."""
  assert len(batch_xs.shape) == 2  # [batch_size, state_dim]
  assert len(batch_us.shape) == 2  # [batch_size, input_dim]
  assert batch_xs.shape[0] == batch_us.shape[0]

  # loss_fn has the signature:
  # loss_fn(policy_network, policy_params, xs, us,
  #         trust_region_params, trust_region_lam)
  #
  # we bind the policy_network argument and grad through policy_params

  grad_fn = jax.grad(
      functools.partial(loss_fn, policy_network), has_aux=True)
  grad, aux = grad_fn(policy_params, batch_xs, batch_us, trust_region_params,
                      trust_region_lam)
  updates, opt_state = opt_update(grad, opt_state, policy_params)
  policy_params = optax.apply_updates(policy_params, updates)
  return policy_params, opt_state, aux


def train_policy_network(policy_network,
                         opt_update,
                         train_xs,
                         train_expert_inputs,
                         policy_params,
                         opt_state,
                         trust_region_params,
                         trust_region_lam,
                         igs_constraint_args,
                         num_epochs,
                         batch_size,
                         holdout_ratio,
                         max_holdout_increases,
                         rng,
                         verbose=False):
  """Train a policy network from trajectories."""

  assert len(train_xs.shape) == 2  # [n, state_dim]
  assert len(train_expert_inputs.shape) == 2  # [n, input_dim]
  assert train_xs.shape[0] == train_expert_inputs.shape[0]
  assert igs_constraint_args is None or len(igs_constraint_args) == 3

  # copy data because we will be mutating it in place
  train_xs = np.array(train_xs)
  train_expert_inputs = np.array(train_expert_inputs)
  train_size = int(len(train_xs) * (1 - holdout_ratio))
  holdout_size = len(train_xs) - train_size
  assert train_size >= 1
  assert holdout_size >= 0

  if batch_size > train_size:
    batch_size = train_size
    n_batches = 1
  else:
    n_batches = train_size // batch_size

  # first, let us create our loss function which has signature
  # loss_fn(policy_network, policy_params, xs, us,
  #         trust_region_params, trust_region_lam)
  if igs_constraint_args is None:
    loss_fn = imitation_loss_fn
  else:
    dynamics, igs_loss, policy_fn = igs_constraint_args
    loss_fn = functools.partial(imitation_loss_with_igs_constraint_fn,
                                dynamics, igs_loss, policy_fn)

  # now let us create our batch_fn function
  batch_fn = functools.partial(do_batch, loss_fn, policy_network, opt_update)

  def holdout_loss_fn(policy_params, holdout_xs, holdout_expert_inputs,
                      trust_region_params, trust_region_lam):
    if holdout_xs.shape[0] == 0:
      return 0.0
    else:
      return loss_fn(policy_network, policy_params, holdout_xs,
                     holdout_expert_inputs, trust_region_params,
                     trust_region_lam)[0]

  if verbose:
    logging.info("train_xs.shape=%s, n_batches=%d, batch_size=%d",
                 train_xs.shape, n_batches, batch_size)
    logging.info("holdout_ratio=%f, train_size=%d, holdout_size=%d, "
                 "max_holdout_increases=%d",
                 holdout_ratio, train_size, holdout_size, max_holdout_increases)

  # shuffle data
  perm = rng.permutation(len(train_xs))
  train_xs = train_xs[perm]
  train_expert_inputs = train_expert_inputs[perm]

  # split data
  train_xs, holdout_xs = train_xs[:train_size], train_xs[train_size:]
  train_expert_inputs, holdout_expert_inputs = (
      train_expert_inputs[:train_size],
      train_expert_inputs[train_size:])

  prev_holdout_loss = None
  num_holdout_increases = 0
  for epoch in range(num_epochs):
    start_time = time.time()

    # shuffle data
    perm = rng.permutation(len(train_xs))
    train_xs = train_xs[perm]
    train_expert_inputs = train_expert_inputs[perm]

    assert len(train_xs) == train_size
    assert len(holdout_xs) == holdout_size
    assert len(train_expert_inputs) == train_size
    assert len(holdout_expert_inputs) == holdout_size

    epoch_losses = None
    for batch in range(n_batches):
      batch_xs = jax.device_put(train_xs[batch * batch_size:(batch + 1) *
                                         batch_size])
      batch_us = jax.device_put(
          train_expert_inputs[batch * batch_size:(batch + 1) * batch_size])
      policy_params, opt_state, loss_aux = batch_fn(
          batch_xs, batch_us, policy_params, opt_state, trust_region_params,
          trust_region_lam)
      if epoch_losses is None:
        epoch_losses = np.array(loss_aux)
      else:
        epoch_losses += np.array(loss_aux)
      del batch_xs
      del batch_us

    holdout_loss = holdout_loss_fn(policy_params, holdout_xs,
                                   holdout_expert_inputs, trust_region_params,
                                   trust_region_lam)

    if prev_holdout_loss is not None and prev_holdout_loss < holdout_loss:
      num_holdout_increases += 1

    if verbose:
      logging.info("epoch %d took %f seconds, avg_losses=%s, "
                   "avg_holdout_loss=%f, num_holdout_increases=%d",
                   epoch,
                   time.time() - start_time,
                   epoch_losses / train_size,
                   holdout_loss / holdout_size if holdout_size else 0.0,
                   num_holdout_increases)

    if num_holdout_increases >= max_holdout_increases:
      logging.info("num_holdout_increase >= %d, stopping training early.",
                   max_holdout_increases)
      break

    prev_holdout_loss = holdout_loss

  return policy_params, opt_state, epoch_losses
