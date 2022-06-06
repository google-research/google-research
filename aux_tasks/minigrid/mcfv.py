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

"""Monte Carlo fisrt visit implementation."""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def mc_first_visit(env,
                   num_epochs=1,
                   num_episodes=100,
                   gamma_tasks=np.array([0.9, 0.99])):
  """Monte Carlo fisrt visit implementation."""
  for _ in tqdm(range(num_epochs)):
    values = dict(
        zip(
            gamma_tasks,
            [
                {
                    "counts": np.zeros((env.num_states,)),
                    "values": np.zeros((env.num_states,)),
                }
            ]
            * len(gamma_tasks),
        )
    )

    for _ in tqdm(range(num_episodes)):
      transitions = []

      # Collect an episodes worth of data
      terminal = False
      obs = env.reset()
      # We'll keep track of the first-visit so we can efficiently
      # calculate the first-visit MC return later.
      timestep = 0
      first_visits = {}
      while not terminal:
        state = obs["state"]
        if state not in first_visits:
          first_visits[state] = timestep

        action = env.action_space.sample()
        next_obs, reward, terminal, _ = env.step(action)
        transitions.append((state, reward))
        obs = next_obs
        timestep += 1

      returns = dict(zip(gamma_tasks, [0] * len(gamma_tasks)))
      # Calculate the first-visit MC return for each value of gamma
      for idx, (s, r) in enumerate(reversed(transitions)):
        for gamma in gamma_tasks:
          returns[gamma] = gamma * returns[gamma] + r
          if timestep - 1 - idx == first_visits[s]:
            values[gamma]["counts"][s] += 1
            values[gamma]["values"][s] += returns[gamma]

    # Construct the "aux. task matrix"
    V = np.zeros((  # pylint: disable=invalid-name
        env.num_states,
        len(gamma_tasks),
    ))
    for idx, (gamma, values_and_counts) in enumerate(values.items()):
      V[:, idx] = values_and_counts["values"] / values_and_counts["counts"]

      # Plot the value function for a sanity check
      grid = np.zeros((env.width, env.height))
      for index, v in enumerate(V[:, index]):
        pos = env.state_to_pos[id]
        grid[tuple(pos)] = v

      plt.matshow(grid.T)
      plt.title(f"Gamma={gamma}")
      plt.colorbar()

    return grid
