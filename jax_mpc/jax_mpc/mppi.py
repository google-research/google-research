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

"""An MPPI based planner."""
import jax
import jax.numpy as jnp
from .mpc import MPC


class MPPI(MPC):
  """An MPPI based planner."""

  def __init__(self, n_iterations=5, n_steps=16, n_samples=16, temperature=0.01,
               damping=0.001, a_noise=0.1, scan=True,
               adaptive_covariance=False):
    self.n_iterations = n_iterations
    self.n_steps = n_steps
    self.n_samples = n_samples
    self.temperature = temperature
    self.damping = damping
    self.a_std = a_noise
    self.scan = scan  # whether to use jax.lax.scan instead of python loop
    self.adaptive_covariance = adaptive_covariance

  def init_state(self, a_shape, rng):
    # uses random as a hack to support vmap
    # we should find a non-hack approach to initializing the state
    dim_a = jnp.prod(a_shape)  # np.int32
    a_opt = 0.0*jax.random.uniform(rng, shape=(self.n_steps,
                                               dim_a))  # [n_steps, dim_a]
    # a_cov: [n_steps, dim_a, dim_a]
    if self.adaptive_covariance:
      # note: should probably store factorized cov,
      # e.g. cholesky, for faster sampling
      a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
    else:
      a_cov = None
    return (a_opt, a_cov)

  def update(self, mpc_state, env, env_state, rng, reward_fn=None,
             reward_params=None, reward_rng=None):
    # mpc_state: ([n_steps, dim_a], [n_steps, dim_a, dim_a])
    # env: {.step(s, a), .reward(s)}
    # env_state: [env_shape] np.float32
    # rng: rng key for mpc sampling
    # reward_fn: reward_fn(env, s, params, rng)
    # reward_params: params for reward function
    # reward_rng: rng key for reward function stochasticity, e.g. dropout
    dim_a = jnp.prod(env.a_shape)  # np.int32
    a_opt, a_cov = mpc_state
    a_opt = jnp.concatenate([a_opt[1:, :],
                             jnp.expand_dims(jnp.zeros((dim_a,)),
                                             axis=0)])  # [n_steps, dim_a]
    if self.adaptive_covariance:
      a_cov = jnp.concatenate([a_cov[1:, :],
                               jnp.expand_dims((self.a_std**2)*jnp.eye(dim_a),
                                               axis=0)])
    def iteration_step(input_, _):
      a_opt, a_cov, rng = input_
      rng_da, rng = jax.random.split(rng)
      if self.adaptive_covariance:
        da = jax.vmap(jax.random.multivariate_normal, (0, 0, 0, None), 1)(
            jax.random.split(rng_da, self.n_steps),  # [n_steps], rngs
            jnp.zeros((self.n_steps, dim_a)),  # [n_steps, dim_a] mean
            a_cov,  # [n_steps, dim_a, dim_a] covariance
            (self.n_samples,),
        )  # [n_samples, n_steps, dim_a]
      else:
        da = self.a_std*jax.random.normal(
            rng_da,
            shape=(self.n_samples, self.n_steps, dim_a)
        )  # [n_samples, n_steps, dim_a]
      # a: [n_samples, n_steps, dim_a]
      a = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
      r = jax.vmap(self.rollout, in_axes=(0, None, None, None, None, None))(
          a, env, env_state, reward_fn, reward_params, reward_rng
      )  # [n_samples, n_steps]
      R = jax.vmap(self.returns)(r)  # [n_samples, n_steps], pylint: disable=invalid-name
      w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
      da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
      a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
      if self.adaptive_covariance:
        a_cov = jax.vmap(jax.vmap(jnp.outer))(
            da, da
        )  # [n_samples, n_steps, dim_a, dim_a]
        a_cov = jax.vmap(jnp.average, (1, None, 1))(
            a_cov, 0, w
        )  # a_cov: [n_steps, dim_a, dim_a]
        # prevent loss of rank when one sample is heavily weighted
        a_cov = a_cov + jnp.eye(dim_a)*0.00001
      return (a_opt, a_cov, rng), None
    if not self.scan:
      for _ in range(self.n_iterations):
        (a_opt, a_cov, rng), _ = iteration_step((a_opt, a_cov, rng), None)
    else:
      (a_opt, a_cov, rng), _ = jax.lax.scan(
          iteration_step, (a_opt, a_cov, rng), None, length=self.n_iterations
      )
    return (a_opt, a_cov)

  def get_action(self, mpc_state, a_shape):
    a_opt, _ = mpc_state
    return jnp.reshape(a_opt[0, :], a_shape)

  def returns(self, r):
    # r: [n_steps]
    return jnp.dot(jnp.triu(jnp.ones((self.n_steps, self.n_steps))),
                   r)  # R: [n_steps]

  def weights(self, R):  # pylint: disable=invalid-name
    # R: [n_samples]
    # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
    # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
    R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
    w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
    w = w/jnp.sum(w)  # [n_samples] np.float32
    return w

  def rollout(self, actions, env, env_state, reward_fn=None,
              reward_params=None, reward_rng=None):
    # actions: [n_steps, dim_a]
    # env: {.step(s, a), .reward(s)}
    # env_state: np.float32
    # reward_fn: reward_fn(env, s, params, rng)
    # reward_params: params for reward function
    # reward_rng: rng key for reward function stochasticity, e.g. dropout
    # a: # a_0, ..., a_{n_steps}. [n_steps, dim_a]
    # s: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
    # r: # r_1, ..., r_{n_steps+1}. [n_steps]

    def rollout_step(env_state, a):
      a = jnp.reshape(a, env.a_shape)
      env_state = env.step(env_state, a)
      r = env.reward(env_state)
      return env_state, (env_state, r)
    if not self.scan:
      # python equivalent of lax.scan
      scan_output = []
      for t in range(self.n_steps):
        env_state, output = rollout_step(env_state, actions[t, :])
        scan_output.append(output)
      s, r = jax.tree_util.tree_multimap(lambda *x: jnp.stack(x), *scan_output)
    else:
      _, (s, r) = jax.lax.scan(rollout_step, env_state, actions)

    if reward_fn is not None:
      r = jax.vmap(reward_fn, (None, 0, None, None))(
          env, s, reward_params, reward_rng
      )  # [n_steps]

    return r


def unstack(a, axis=0):
  """The opposite of stack()."""
  shape = a.shape
  return [jnp.squeeze(b, axis=axis) for b in \
          jnp.split(a, shape[axis], axis=axis)]
jnp.unstack = unstack
