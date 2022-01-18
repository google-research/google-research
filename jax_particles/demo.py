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

"""Demonstrates usage of jax_particles."""
import time
import jax
import jax.numpy as jnp
from jax_particles import BaseEnvironment
from jax_particles import Entity
from jax_particles.renderer import ThreadedRenderer


class Environment(BaseEnvironment):
  """A simple demo environment. Typically you would create a module for this."""

  def __init__(self):
    super().__init__()

    # add agents
    agents = []
    agent = Entity()
    agent.name = "agent"
    agent.color = [64, 64, 64]
    agent.collideable = True
    agents.append(agent)

    # add collideable objects
    objects = []
    obj = Entity()
    obj.name = "object"
    obj.color = [64, 64, 204]
    obj.collideable = True
    objects.append(obj)

    # add landmarks
    landmarks = []
    landmark = Entity()
    landmark.name = "landmark"
    landmark.color = [64, 205, 64]
    landmark.collideable = False
    landmark.radius = 0.1
    landmarks.append(landmark)

    self.entities.extend(agents)
    self.entities.extend(objects)
    self.entities.extend(landmarks)

    self.a_shape = (1, 2)  # shape of actions, one agent here
    self.o_shape = {"vec": None}  # shape of observations, no observations here

    self._compile()

  def init_state(self, rng):
    """Returns a state (p,v,misc) tuple."""
    shape = (len(self.entities), 2)
    p = jax.random.uniform(rng, shape) * (self.max_p - self.min_p) + self.min_p
    v = jnp.zeros(shape)+0.0
    return (p, v, None)

  def obs(self, s):
    """Returns an observation dictionary."""
    o = {"vec": None}
    return o

  def reward(self, s):
    """Returns a joint reward: np.float32."""
    return 0.0


def main():
  """Create an environment and step through, taking user actions."""
  batch_size = 5  # 100000
  i_batch = 1  # 25843
  stop_every_step = False

  env = Environment()
  renderer = ThreadedRenderer()

  # compile a step function
  def step(s, a):
    s = env.step(s, a)
    o = env.obs(s)
    r = env.reward(s)
    return (s, o, r)
  step = jax.vmap(step)
  step = jax.jit(step)

  # compile an init_state function
  init_state = env.init_state
  init_state = jax.vmap(init_state)
  init_state = jax.jit(init_state)

  rng = jax.random.PRNGKey(1)
  s = init_state(jax.random.split(rng, batch_size))
  last_time = time.time()
  while True:
    # extract the state of the i'th environment
    s_i = [elem[i_batch, :] if elem is not None else None for elem in s]

    # render
    renderer.render(env, s_i)

    # get user action
    a = renderer.get_action()
    a = jnp.broadcast_to(jnp.array(a), (batch_size,)+env.a_shape)

    # do simulation step
    s, o, r = step(s, a)  # pylint: disable=unused-variable

    if stop_every_step:
      input("> ")
    else:
      while time.time() - last_time < env.dt:
        time.sleep(0.001)
      last_time = time.time()


if __name__ == "__main__":
  main()
