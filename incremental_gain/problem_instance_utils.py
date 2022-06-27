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

"""dynamics."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def make_dynamics_and_expert(key, state_dim, p, eta, activation):
  """Make dynamics and expert."""

  teacher_hidden_width = 32

  w_teacher_init = hk.initializers.RandomNormal(stddev=0.5)
  # b_teacher_init = hk.initializers.RandomNormal()

  # no bias, so that h(0) = 0
  def teacher_policy(state):
    mlp = hk.Sequential([
        hk.Linear(
            teacher_hidden_width, w_init=w_teacher_init, with_bias=False),
        activation,
        hk.Linear(
            teacher_hidden_width, w_init=w_teacher_init, with_bias=False),
        activation,
        hk.Linear(state_dim, w_init=w_teacher_init, with_bias=False),
    ])
    return mlp(state)

  teacher_policy_t = hk.without_apply_rng(hk.transform(teacher_policy))
  teacher_params = teacher_policy_t.init(key, jnp.zeros((state_dim,)))

  def h_disturbance(x):
    return teacher_policy_t.apply(teacher_params, x)

  assert np.allclose(h_disturbance(np.zeros((state_dim,))),
                     np.zeros((state_dim,)))

  def dynamics(x, u):
    f = x - eta * x * (jnp.abs(x) ** p) / (1 + (jnp.abs(x)**p))
    g = eta / (1 + (jnp.abs(x)**p)) * (h_disturbance(x) + u)
    return f + g

  def expert_policy(state):
    return -h_disturbance(state)

  return dynamics, expert_policy


def sample_initial_conditions(key, n_trajs, state_dim):
  return jax.random.normal(key, shape=(n_trajs, state_dim))


def rollout_policy(dynamics, policy, x0, horizon):

  def scan_fn(state, _):
    inp = policy(state)
    next_state = dynamics(state, inp)
    return next_state, (next_state, inp)

  _, (xs, us) = jax.lax.scan(scan_fn, x0, None, length=horizon)
  return jnp.concatenate([x0[None, :], xs], axis=0), us


def zero_policy(state):
  return jnp.zeros_like(state)
