# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# An implementation of Mechanic - black box learning rate tuner from:
#
# Mechanic: A Learning Rate Tuner, https://arxiv.org/pdf/2306.00144.pdf
# Ashok Cutkosky, Aaron Defazio, Harsh Mehta
#
# Author: Harsh Mehta (harshm at google dot com)
#

"""Mechanic Implementation."""

import functools
import operator
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import optax


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def _vdot_safe(a, b):
  cvdot = _vdot(jnp.asarray(a), jnp.asarray(b))
  return cvdot


@jax.jit
def tree_vdot(tree_x, tree_y):
  """Compute the inner product <tree_x, tree_y>."""
  vdots = jax.tree_util.tree_map(_vdot_safe, tree_x, tree_y)
  return jax.tree_util.tree_reduce(operator.add, vdots)


@jax.jit
def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


@jax.jit
def tree_norm(tree):
  return jnp.sqrt(tree_sum(jax.tree.map(lambda x: jnp.sum(x**2), tree)))


class MechanicState(NamedTuple):
  """State of the `GradientTransformation` returned by `mechanize`."""
  base_optimizer_state: optax.OptState
  count: chex.Array  # shape=(), dtype=jnp.int32.
  r: optax.Updates
  m: optax.Updates
  v: optax.Updates
  s: optax.Updates
  x0: optax.Updates


def mechanize(
    base_optimizer,
    weight_decay = 1e-2,
    eps = 1e-10,
    s_init = 1e-8
):
  """Mechanic - black box learning rate tuner/optimizer.

  Accumulates updates returned by the base_optimizer and learns the scale of
  the updates (also know as learning rate or step size) to apply on a per
  iteration basis.

  Note that Mechanic does NOT eschew a need for a learning rate schedule,
  you are free to apply a learning rate schedule with base learning rate set to
  1.0 (or any other constant) and Mechanic will learn the right scale factor
  automatically.

  As of June, 2023, Mechanic is tested with SGD, Momentum, Adam and Lion as
  inner optimizers but we expect it to work with almost any first-order
  optimizer.

  References:
    [Cutkosky et al, 2023](https://arxiv.org/pdf/2306.00144.pdf)

  Args:
    base_optimizer: Base optimizer to compute updates from.
    weight_decay: A scalar weight decay rate.
    eps: epsilon for mechanic.
    s_init: initial scale factor. Default should work almost all the time.

  Returns:
    A `GradientTransformation` with init and update functions.
  """

  def init_fn(params):
    x0 = jax.tree_util.tree_map(lambda t: t.astype(jnp.float32), params)
    num_betas = 6
    r = jnp.zeros([num_betas,], jnp.float32)
    v = jnp.zeros([num_betas,], jnp.float32)
    m = jnp.zeros([num_betas,], jnp.float32)
    s = jnp.ones([num_betas,], jnp.float32) * s_init
    return MechanicState(
        base_optimizer_state=base_optimizer.init(params),
        count=jnp.zeros([], jnp.int32),
        r=r,
        m=m,
        v=v,
        s=s,
        x0=x0,
    )

  def update_fn(
      updates, state, params
  ):
    count_inc = optax.safe_int32_increment(state.count)
    new_neg_updates, base_optimizer_state = base_optimizer.update(
        updates, state.base_optimizer_state, params
    )
    # Since a lot of training loops unfreezes weights to replace it with
    # pre-trained weights, we want to make sure we start from actually used
    # weights instead of what they were initialized with.
    x0 = jax.lax.cond(state.count == 0, lambda: params, lambda: state.x0)

    # Add weight decay to raw gradients, note that this is othogonal to any
    # weight decay applied to inner_optimizer updates.
    s_sum = jnp.sum(state.s)
    grad_norm = tree_norm(updates)
    param_norm = tree_norm(params)

    def add_weight_decay(gi, pi):
      return gi + weight_decay * s_sum * grad_norm / (param_norm + eps) * pi

    updates = jax.tree_util.tree_map(
        add_weight_decay,
        updates,
        params,
    )

    # We use the memory efficient version of Mechanic where we re-compute
    # \Delta every iteration.
    delta_prev = jax.tree_util.tree_map(
        lambda xti, x0i: (x0i - xti) / (s_sum + eps),
        params,
        x0)

    # We actually want to add the updates, but since optax by default flips
    # signs when applying the learning rate, we substract instead.
    delta = jax.tree_util.tree_map(
        lambda si, ui: si - ui, delta_prev, new_neg_updates
    )

    # Now we are ready to run the actual Mechanic algorithm.
    h = tree_vdot(updates, delta_prev)
    betas = jnp.array([
        0.9,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
    ])

    m = jnp.maximum(betas * state.m, jnp.abs(h) + eps)
    v = (betas**2) * state.v + h**2
    r = betas * state.r + h * state.s
    rc = jnp.maximum(0.0, r)
    wealth = (s_init / jnp.size(betas)) * m + rc
    s = wealth / (jnp.sqrt(v) + eps)

    # Once we have the scale factor s, we produce new params with it.
    new_x0 = x0
    new_params = jax.tree_util.tree_map(
        lambda x0, deltai: x0 - jnp.sum(s) * deltai,
        new_x0,
        delta)
    new_neg_updates = jax.tree_util.tree_map(
        lambda np, op: np - op, new_params, params)

    return new_neg_updates, MechanicState(
        base_optimizer_state=base_optimizer_state,
        count=count_inc,
        r=r,
        m=m,
        v=v,
        s=s,
        x0=new_x0,
    )

  return optax.GradientTransformation(init_fn, update_fn)
