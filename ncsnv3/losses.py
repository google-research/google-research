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

"""All functions related to loss computation and optimization.
"""

import flax
from flax.deprecated import nn
import jax
import jax.numpy as jnp
import jax.random as random


def get_optimizer(config):
  optimizer = None
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
        f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on config."""
  def optimize(state,
               grad,
               warmup=config.optim.warmup,
               grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
          sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
          lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  return optimize


def ncsn_loss(rng,
              state,
              batch,
              sigmas,
              continuous=False,
              train=True,
              optimize_fn=None,
              anneal_power=2.,
              loss_per_sigma=False,
              class_conditional=False,
              pmap_axis_name='batch'):
  """The objective function for NCSN.

  Does one step of training or evaluation.
  Store EMA statistics during training and use EMA for evaluation.
  Will be called by jax.pmap using `pmap_axis_name`.

  Args:
    rng: a jax random state.
    state: a pytree of training states, including the optimizer, lr, etc.
    batch: a pytree of data points.
    sigmas: a numpy arrary representing the array of noise levels.
    continuous: Use a continuous distribution of sigmas and sample from it.
    train: True if we will train the model. Otherwise just do the evaluation.
    optimize_fn: takes state and grad and performs one optimization step.
    anneal_power: balancing losses of different noise levels. Defaults to 2.
    loss_per_sigma: return the loss for each sigma separately.
    class_conditional: train a score-based model conditioned on class labels.
    pmap_axis_name: the axis_name used when calling this function with pmap.

  Returns:
    loss, new_state if not loss_per_sigma. Otherwise return loss, new_state,
    losses, and used_sigmas. Here used_sigmas are noise levels sampled in this
    mini-batch, and `losses` contains the loss value for each datapoint and
    noise level.
  """
  x = batch['image']
  rng1, rng2 = random.split(rng)
  if not continuous:
    labels = random.choice(rng1, len(sigmas), shape=(x.shape[0],))
    used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
  else:
    labels = random.uniform(
        rng1, (x.shape[0],),
        minval=jnp.log(sigmas[-1]),
        maxval=jnp.log(sigmas[0]))
    labels = jnp.exp(labels)
    used_sigmas = labels.reshape((x.shape[0], *([1] * len(x.shape[1:]))))

  if class_conditional:
    class_labels = batch['label']

  noise = random.normal(rng2, x.shape) * used_sigmas
  perturbed_data = noise + x

  run_rng, _ = random.split(rng2)
  @jax.jit
  def loss_fn(model):
    if train:
      with nn.stateful(state.model_state) as new_model_state:
        with nn.stochastic(run_rng):
          if not class_conditional:
            scores = model(perturbed_data, labels, train=train)
          else:
            scores = model(perturbed_data, labels, y=class_labels, train=train)
    else:
      with nn.stateful(state.model_state, mutable=False):
        with nn.stochastic(run_rng):
          if not class_conditional:
            scores = model(perturbed_data, labels, train=train)
          else:
            scores = model(perturbed_data, labels, y=class_labels, train=train)

      new_model_state = state.model_state

    scores = scores.reshape((scores.shape[0], -1))
    target = -1 / (used_sigmas ** 2) * noise
    target = target.reshape((target.shape[0], -1))
    losses = 1 / 2. * ((scores - target)**
                       2).sum(axis=-1) * used_sigmas.squeeze()**anneal_power
    loss = jnp.mean(losses)

    if loss_per_sigma:
      return loss, new_model_state, losses
    else:
      return loss, new_model_state

  if train:
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    if loss_per_sigma:
      (loss, new_model_state, losses), grad = grad_fn(state.optimizer.target)
    else:
      (loss, new_model_state), grad = grad_fn(state.optimizer.target)
    grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    new_optimizer = optimize_fn(state, grad)
    new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target.params)
    step = state.step + 1
    new_state = state.replace(  # pytype: disable=attribute-error
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema)
  else:
    model_ema = state.optimizer.target.replace(params=state.params_ema)
    if loss_per_sigma:
      loss, _, losses = loss_fn(model_ema)  # pytype: disable=bad-unpacking
    else:
      loss, *_ = loss_fn(model_ema)

    new_state = state

  loss = jax.lax.pmean(loss, axis_name=pmap_axis_name)
  if loss_per_sigma:
    return loss, new_state, losses, used_sigmas.squeeze()
  else:
    return loss, new_state


def ddpm_loss(rng,
              state,
              batch,
              ddpm_params,
              train=True,
              optimize_fn=None,
              pmap_axis_name='batch'):
  """The objective function for DDPM.

  Same as NCSN but with different noise perturbations. Mostly copied
  from https://github.com/hojonathanho/diffusion.

  Does one step of training or evaluation.
  Store EMA statistics during training and evaluate with EMA.
  Will be called by jax.pmap using `pmap_axis_name`.

  Args:
    rng: a jax random state.
    state: a pytree of training states, including the optimizer, lr, etc.
    batch: a pytree of data points.
    ddpm_params: a dictionary containing betas, alphas, and others.
    train: True if we will train the model. Otherwise just do the evaluation.
    optimize_fn: takes state and grad and performs one optimization step.
    pmap_axis_name: the axis_name used when calling this function with pmap.

  Returns:
    loss, new_state
  """

  x = batch['image']
  rng1, rng2 = random.split(rng)
  betas = jnp.asarray(ddpm_params['betas'], dtype=jnp.float32)
  sqrt_alphas_cumprod = jnp.asarray(
      ddpm_params['sqrt_alphas_cumprod'], dtype=jnp.float32)
  sqrt_1m_alphas_cumprod = jnp.asarray(
      ddpm_params['sqrt_1m_alphas_cumprod'], dtype=jnp.float32)
  T = random.choice(rng1, len(betas), shape=(x.shape[0],))  # pylint: disable=invalid-name

  noise = random.normal(rng2, x.shape)

  perturbed_data = sqrt_alphas_cumprod[T, None, None, None] * x + \
      sqrt_1m_alphas_cumprod[T, None, None, None] * noise

  run_rng, _ = random.split(rng2)

  @jax.jit
  def loss_fn(model):
    if train:
      with nn.stateful(state.model_state) as new_model_state:
        with nn.stochastic(run_rng):
          scores = model(perturbed_data, T, train=train)
    else:
      with nn.stateful(state.model_state, mutable=False):
        with nn.stochastic(run_rng):
          scores = model(perturbed_data, T, train=train)

      new_model_state = state.model_state

    scores = scores.reshape((scores.shape[0], -1))
    target = noise.reshape((noise.shape[0], -1))
    loss = jnp.mean((scores - target)**2)
    return loss, new_model_state

  if train:
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss, new_model_state), grad = grad_fn(state.optimizer.target)
    grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    ## WARNING: the gradient clip step differs slightly from the
    ## original DDPM implementation, and seem to be more reasonable.
    ## The impact of this difference on performance is negligible.
    new_optimizer = optimize_fn(state, grad)
    new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target.params)
    step = state.step + 1
    new_state = state.replace(  # pytype: disable=attribute-error
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema)
  else:
    model_ema = state.optimizer.target.replace(params=state.params_ema)
    loss, _ = loss_fn(model_ema)
    new_state = state

  loss = jax.lax.pmean(loss, axis_name=pmap_axis_name)
  return loss, new_state
