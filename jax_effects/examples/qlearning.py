# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Q-learning examples."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument

from collections.abc import Sequence
from typing import Any, Callable

from absl import app
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import jax_effects

effect = jax_effects.effect
effectify = jax_effects.effectify
ParameterizedHandler = jax_effects.ParameterizedHandler

Aval = jax.core.AbstractValue
Unit = jax.core.ShapedArray(shape=(), dtype=jnp.bool_)

int_state = jnp.int32
int_action = jnp.int8
State = jax.Array
Action = int_action
Reward = jnp.float32
P = Any
Q = jax.Array


def q_learning_example():
  """Q-learning example."""

  width, height = (6, 4)
  action_list = [0, 1, 2, 3]  # U, D, L, R
  max_action = len(action_list)

  # Initial states and auxiliary functions.
  start_state = (0, 0)
  start_state = jnp.asarray(start_state, int_state)

  def is_goal(s: State) -> jax.Array:
    return jnp.logical_and(s[0] == width - 1, s[1] == 0)

  def on_cliff(s: State) -> jax.Array:
    x, y = s
    return jnp.logical_and(jnp.logical_and(y == 0, x > 0), x < width - 1)

  def update_state(s: State, a: Action) -> State:
    jmax = lambda a, b: jnp.max(jnp.asarray([a, b]))
    jmin = lambda a, b: jnp.min(jnp.asarray([a, b]))
    jax.debug.print('Updating state from {} with action {}', s, a)

    x, y = s
    l_update = lambda x, y: (jmax(0, x - 1), y)
    r_update = lambda x, y: (jmin(width - 1, x + 1), y)
    d_update = lambda x, y: (x, jmax(0, y - 1))
    u_update = lambda x, y: (x, jmin(height - 1, y + 1))

    new_state = lax.switch(
        a,
        [l_update, r_update, d_update, u_update],
        x,
        y,
    )
    jax.debug.print('new state: {}', jnp.asarray(new_state, int_state))

    return jnp.asarray(new_state, int_state)

  # Effect operations

  # Agent operations
  @effect(abstract_eval=lambda _: jax.core.ShapedArray((), int_action))
  def predict(state: State) -> Action:
    pass

  @effect(abstract_eval=lambda *_: Unit)
  def feedback(x: tuple[State, Action, Reward, State]) -> None:
    pass

  # Env operations
  @effect(abstract_eval=lambda _: [
      jax.core.ShapedArray((), Reward),
      jax.core.ShapedArray((2,), int_state),
  ])
  def observe(x: Action) -> Reward:
    pass

  @effect(abstract_eval=lambda _: jax.core.ShapedArray([], jnp.float_))
  def random_uniform(start, end) -> Reward:
    pass

  def random_uniform_handler(key: jax.Array, x: Any, k, lk):
    key, subkey = random.split(key)
    minval, maxval = x
    return k(key, random.uniform(subkey, minval=minval, maxval=maxval))

  def gen_random_action() -> Action:
    a = random_uniform(0, max_action)
    return jnp.asarray(jnp.floor(a), int_action)

  def find_max_action(s: State, q: Q) -> Action:
    rewards = jnp.asarray(
        list(map(lambda a: find_from_q(s, a, q), action_list))
    )
    return jnp.asarray(jnp.argmax(rewards), int_action)

  def find_max_reward(s: State, q: Q) -> Reward:
    rewards = jnp.asarray(
        list(map(lambda a: find_from_q(s, a, q), action_list))
    )
    return jnp.max(rewards)

  def find_from_q(s: State, a: Action, q: Q) -> Reward:
    return q[s[0], s[1], a]

  # Effect handlers
  def predict_handler(q: Q, state: State, k, lk):
    epsilon = random_uniform(0, 1)
    a = lax.select(
        jnp.less(epsilon, 0.1), gen_random_action(), find_max_action(state, q)
    )
    return k(q, a)

  def feedback_handler(q: Q, x: tuple[State, Action, Reward, State], k, lk):
    last_state, action, reward, state = x
    prev_reward = find_from_q(last_state, action, q)

    next_reward = find_max_reward(state, q)
    new_reward = prev_reward + 0.2 * (reward + 0.9 * next_reward - prev_reward)
    q_prime = q.at[last_state[0], last_state[1], action].set(new_reward)
    return k(q_prime, True)

  def observe_handler(s: State, a: Action, k, lk):
    s_prime = update_state(s, a)
    cond = on_cliff(s_prime)
    env_state = lax.select(cond, start_state, s_prime)
    reward = lax.select(cond, -100.0, -1.0)
    return k(env_state, reward, env_state)

  # Agent definition
  def agent(
      p: P, state: State, is_goal: Callable[[State], jax.Array]
  ) -> Action:
    def cond(arg: tuple[jax.Array, P, State]):
      _, _, state = arg
      return jnp.invert(is_goal(state))

    def body(arg: tuple[jax.Array, P, State]):
      key, p, state = arg
      with ParameterizedHandler(
          key, random_uniform=random_uniform_handler
      ) as random_handler:
        with ParameterizedHandler(
            p, predict=predict_handler, feedback=feedback_handler
        ) as agent_handler:
          with ParameterizedHandler(
              state, observe=observe_handler
          ) as env_handler:
            action = predict(state)
            reward, new_state = observe(action)
            feedback((state, action, reward, new_state))
            env_handler.result = new_state
          env_state, new_state = env_handler.result
          del env_state
          agent_handler.result = new_state
        random_handler.result = agent_handler.result
      new_key, (p, _) = random_handler.result
      return (new_key, p, new_state)

    key = random.PRNGKey(42)

    # @handle(
    #     key,
    #     random_uniform=random_uniform_handler,
    #     return_fun=lambda p, result: (p, *result),
    # )
    # @handle(
    #     p,
    #     predict=predict_handler,
    #     feedback=feedback_handler,
    # )
    # @handle(
    #     state,
    #     observe=observe_handler,
    #     return_fun=lambda _, result: result,
    # )
    # def body(arg: tuple[jax.Array, P, State]):
    #   key, p, state = arg
    #   del key, p
    #   action = predict(state)
    #   reward, new_state = observe(action)
    #   feedback((state, action, reward, new_state))
    #   return new_state

    _, p, state = lax.while_loop(cond, body, (key, p, state))
    return p, state

  def cliff_walking_agent(p: P, state: State) -> Action:
    return agent(p, state, is_goal)

  result = effectify(cliff_walking_agent, verbose=False)(
      jnp.zeros(shape=(width, height, max_action), dtype=Reward),
      jnp.asarray(([0, 0])),
  )
  return result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(q_learning_example())


if __name__ == '__main__':
  app.run(main)
