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

"""Custom optimizers."""
import pdb

import jax
import jax.numpy as jnp


# Functions for Adam
# ==================
def adam_reset_opt_params(params, init_opt_params):
  return {
      'lr': jax.tree_map(lambda x: jnp.array(init_opt_params['lr']), params),
      'b1': jax.tree_map(lambda x: jnp.array(init_opt_params['b1']), params),
      'b2': jax.tree_map(lambda x: jnp.array(init_opt_params['b2']), params),
      'eps': jax.tree_map(lambda x: jnp.array(init_opt_params['eps']), params),
      'wd': jax.tree_map(lambda x: jnp.array(init_opt_params['wd']), params),
      'm': jax.tree_map(lambda x: jnp.zeros(x.shape), params),
      'v': jax.tree_map(lambda x: jnp.zeros(x.shape), params),
      't': jnp.array(0)
  }


@jax.jit
def adam_optimizer_step(params, grads, opt_params):
  # AdamW weight decay
  t = opt_params['t']

  params_wd = jax.tree_multimap(lambda p, wd, lr: p * (1.0 - lr * wd), params,
                                opt_params['wd'], opt_params['lr'])
  opt_params['m'] = jax.tree_multimap(lambda b1, g, m: (1 - b1) * g + b1 * m,
                                      opt_params['b1'], grads, opt_params['m'])
  opt_params['v'] = jax.tree_multimap(lambda b2, g, v: (1 - b2) * g**2 + b2 * v,
                                      opt_params['b2'], grads, opt_params['v'])
  mhat = jax.tree_multimap(lambda b1, m: m / (1 - b1**(t + 1)),
                           opt_params['b1'], opt_params['m'])
  vhat = jax.tree_multimap(lambda b2, v: v / (1 - b2**(t + 1)),
                           opt_params['b2'], opt_params['v'])
  updated_params = jax.tree_multimap(
      lambda lr, eps, p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
      opt_params['lr'], opt_params['eps'], params, mhat, vhat)
  opt_params['t'] += 1
  return updated_params, opt_params


# Functions for SGDm
# ==================
def sgdm_reset_opt_params(params, init_opt_params):
  return {
      'lr': jax.tree_map(lambda x: jnp.array(init_opt_params['lr']), params),
      'mom': jax.tree_map(lambda x: jnp.array(init_opt_params['mom']), params),
      'wd': jax.tree_map(lambda x: jnp.array(init_opt_params['wd']), params),
      'buf': jax.tree_map(lambda x: jnp.zeros(x.shape), params)
  }


@jax.jit
def sgdm_optimizer_step(params, grads, opt_params):
  """This follows the PyTorch SGD + momentum implementation.

    From https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
  """
  nesterov = False

  # Weight decay
  d_p = jax.tree_multimap(lambda wd, g, p: g + wd * p, opt_params['wd'], grads,
                          params)
  # Momentum
  opt_params['buf'] = jax.tree_multimap(lambda mom, b, g: b * mom + g,
                                        opt_params['mom'], opt_params['buf'],
                                        d_p)
  # Nesterov
  if nesterov:
    d_p = jax.tree_multimap(lambda mom, g, b: g + mom * b, opt_params['mom'],
                            d_p, opt_params['buf'])
  else:
    d_p = opt_params['buf']

  steps = jax.tree_multimap(lambda lr, g: lr * g, opt_params['lr'], d_p)
  updated_params = jax.tree_multimap(lambda p, s: p - s, params, steps)
  return updated_params, opt_params


# Functions for vanilla SGD
# =========================
def sgd_reset_opt_params(params, init_opt_params):
  return {
      'lr': jax.tree_map(lambda x: jnp.array(init_opt_params['lr']), params),
      'wd': jax.tree_map(lambda x: jnp.array(init_opt_params['wd']), params)
  }


@jax.jit
def sgd_optimizer_step(params, grads, opt_params):
  # Weight decay
  d_p = jax.tree_multimap(lambda wd, g, p: g + wd * p, opt_params['wd'], grads,
                          params)
  updated_params = jax.tree_multimap(lambda lr, p, g: p - lr * g,
                                     opt_params['lr'], params, d_p)
  return updated_params, opt_params


def init_optimizer(name, **kwargs):
  if name == 'adam':
    return {
        'reset_opt_params': adam_reset_opt_params,
        'opt_step': adam_optimizer_step
    }
  elif name == 'sgdm':
    return {
        'reset_opt_params': sgdm_reset_opt_params,
        'opt_step': sgdm_optimizer_step
    }
  elif name == 'sgd':
    return {
        'reset_opt_params': sgd_reset_opt_params,
        'opt_step': sgd_optimizer_step
    }
