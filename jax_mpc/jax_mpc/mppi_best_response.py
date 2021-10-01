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

"""An MPPI based planner that plans a response to another agent.

A reward function for the other agent is supplied and this planner attempts to
choose the best action in response to the other agent's optimal policy under
the their supplied reward function.
"""
import jax
import jax.numpy as jnp
from .mpc import MPC


class MPPIBestResponse(MPC):
  """An MPPI based "best response" planner."""

  def __init__(self, n_iterations=5, n_steps=16, n_samples=16, temperature=0.01,
               damping=0.001, a_noise=0.1, scan=True):
    self.n_iterations = n_iterations
    self.n_steps = n_steps
    self.n_samples = n_samples
    self.temperature = temperature
    self.damping = damping
    self.a_std = a_noise
    self.scan = scan  # whether to use jax.lax.scan instead of python loop

  def init_state(self, a_shape, rng):
    # uses random as a hack to support vmap
    # we should find a non-hack approach to initializing the state
    dim_a = jnp.prod(a_shape)  # np.int32
    a_opt = 0.0*jax.random.uniform(rng, shape=(self.n_steps,
                                               dim_a))  # [n_steps, dim_a]
    return a_opt

  def update(self, mpc_state, env, env_state, rng, ampc, areward_fn=None,
             areward_params=None, areward_rng=None, ureward_fn=None,
             ureward_params=None, ureward_rng=None):
    # With this version of best response, there are two agents "a" and "u"
    # areward_x is the reward function for the primary agent "a", ureward_x is
    # the reward function for the other agent "u" that agent "a" would like to
    # respond to.
    #
    # mpc_state: [n_steps, dim_a]
    # env: {.step(s, a), .reward(s)}
    # env_state: [env_shape] np.float32
    # rng: rng key for mpc sampling
    # ampc: {.init_state(a_shape, rng),
    #        .update(mpc_state, env, s, rng, reward_fn=None,
    #                reward_params=None, reward_rng=None),
    #        .get_action(mpc_state, a_shape)}
    # areward_fn: reward_fn(env, s, params, rng)
    # areward_params: params for reward function
    # areward_rng: rng key for reward function stochasticity, e.g. dropout
    # ureward_fn: reward_fn(env, s, params, rng)
    # ureward_params: params for reward function
    # ureward_rng: rng key for reward function stochasticity, e.g. dropout
    dim_a = jnp.prod(env.a_shape)  # np.int32
    a_opt = mpc_state
    a_opt = jnp.concatenate([a_opt[1:, :],
                             jnp.expand_dims(jnp.zeros((dim_a,)),
                                             axis=0)])  # [n_steps, dim_a]
    def iteration_step(input_, _):
      a_opt, rng = input_
      rng_da, rng_rollout, rng = jax.random.split(rng, 3)
      # da: [n_samples, n_steps, dim_a]
      da = self.a_std*jax.random.normal(rng_da, shape=(self.n_samples,
                                                       self.n_steps,
                                                       dim_a))
      da = da.at[:, :, 0:2].set(0.0)
      da = da.at[:, :, 3].set(0.0)
      # a: [n_samples, n_steps, dim_a]
      a = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)

      a_new, r = jax.vmap(self.omniscient_rollout,
                          in_axes=(0, None, None, 0, None, None, None, None,
                                   None, None, None))\
          (a,
           env,
           env_state,
           jax.random.split(rng_rollout, self.n_samples),
           ampc,
           areward_fn,
           areward_params,
           areward_rng, ureward_fn,
           ureward_params,
           ureward_rng)  # [n_samples, n_steps, a_shape], [n_samples, n_steps]
      a = a_new.reshape(a.shape)  # [n_samples, n_steps, dim_a]

      R = jnp.sum(r, axis=-1)  # [n_samples], pylint: disable=invalid-name
      w = self.weights(R)  # [n_samples]

      a_opt = a[jnp.argmax(w), :, :]  # [n_steps, dim_a]
      return (a_opt, rng), None
    if not self.scan:
      for _ in range(self.n_iterations):
        (a_opt, rng), _ = iteration_step((a_opt, rng), None)
    else:
      (a_opt, rng), _ = jax.lax.scan(iteration_step, (a_opt, rng), None,
                                     length=self.n_iterations)
    return a_opt

  def get_action(self, mpc_state, a_shape):
    a_opt = mpc_state
    return jnp.reshape(a_opt[0, :], a_shape)

  def returns(self, r):
    # r: [n_steps]
    return jnp.dot(jnp.triu(jnp.ones((self.n_steps, self.n_steps))),
                   r)  # R: [n_steps]

  def weights(self, R):  # pylint: disable=invalid-name
    # R: [n_samples] np.float32
    # R_stdzd = ((R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
    # R_stdzd = R - jnp.max(R)
    R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
    w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
    w = w/jnp.sum(w)  # [n_samples] np.float32
    return w

  def omniscient_rollout(self, actions, env, env_state, rng, ampc,
                         areward_fn=None, areward_params=None,
                         areward_rng=None, ureward_fn=None, ureward_params=None,
                         ureward_rng=None):
    # actions: [n_steps, dim_a]
    # env: {.step(s, a), .reward(s)}
    # env_state: np.float32
    # rng
    # ampc: {.init_state(a_shape, rng),
    #        .update(mpc_state, env, s, rng, reward_fn=None,
    #                reward_params=None, reward_rng=None),
    #        .get_action(mpc_state)}
    # areward_fn: reward_fn(env, s, params, rng)
    # areward_params: params for reward function
    # areward_rng: rng key for reward function stochasticity, e.g. dropout
    # ureward_fn: reward_fn(env, s, params, rng)
    # ureward_params: params for reward function
    # ureward_rng: rng key for reward function stochasticity, e.g. dropout
    #
    # a: # a_0, ..., a_{n_steps}. [n_steps, dim_a]
    # s: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
    # r: # r_1, ..., r_{n_steps+1}. [n_steps]

    def rollout_step(carry, a):
      env_state, ampc_state = carry
      ampc_state = ampc.update(ampc_state, env, env_state, rng,
                               areward_fn, areward_params, areward_rng)
      aa = ampc.get_action(ampc_state, env.a_shape)  # [2,2] "agent, user"
      # aa = jnp.zeros(env.a_shape)
      au = jnp.reshape(a, env.a_shape)  # [2,2] "agent, user"
      a = jnp.stack([aa[0, :], au[1, :]])  # [2,2]
      env_state = env.step(env_state, a)
      r = env.reward(env_state)
      return (env_state, ampc_state), (a, env_state, r)
    ampc_state = ampc.init_state(env.a_shape, rng)
    if not self.scan:
      # python equivalent of lax.scan
      scan_output = []
      for t in range(self.n_steps):
        (env_state, ampc_state), output = rollout_step((env_state, ampc_state),
                                                       actions[t, :])
        scan_output.append(output)
      a, s, r = jax.tree_util.tree_multimap(lambda *x: jnp.stack(x),
                                            *scan_output)
    else:
      _, (a, s, r) = jax.lax.scan(rollout_step, (env_state, ampc_state),
                                  actions)

    if ureward_fn is not None:
      r = jax.vmap(ureward_fn, (None, 0, None, None))(
          env, s, ureward_params, ureward_rng
      )  # [n_steps]

    return a, r


def unstack(a, axis=0):
  """The opposite of stack()."""
  shape = a.shape
  return [jnp.squeeze(b, axis=axis) for b in \
          jnp.split(a, shape[axis], axis=axis)]
jnp.unstack = unstack
