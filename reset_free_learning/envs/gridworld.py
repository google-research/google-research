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

# Lint as: python3
"""Continuous version of the frozen lake environment.

Everytime the agent hits the goal or the hole, the agent resets to the start
state with the reward.
"""
import contextlib
import io
import sys

from gym import spaces
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF",
        "FHFFHFHF", "FFFHFFFG"
    ],
}


def generate_random_map(size=8, p=0.8):
  """Generates a random valid map (one that has a path from start to goal).

    Args:
      size: size of each side of the grid
      p: probability that a tile is frozen

    Returns:
      The string representing grid.
  """
  valid = False

  # DFS to check that it's a valid path.
  def is_valid(res):
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
      r, c = frontier.pop()
      if (r, c) not in discovered:
        discovered.add((r, c))
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
          r_new = r + x
          c_new = c + y
          if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
            continue
          if res[r_new][c_new] == "G":
            return True
          if res[r_new][c_new] not in "#H":
            frontier.append((r_new, c_new))
    return False

  while not valid:
    p = min(1, p)
    res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
    res[0][0] = "S"
    res[-1][-1] = "G"
    valid = is_valid(res)
  return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
  """Winter is here.

  You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the
    lake. The water is mostly frozen, but there are a few holes where the ice
    has
    melted. If you step into one of those holes, you'll fall into the freezing
    water.
    At this time, there's an international frisbee shortage, so it's absolutely
    imperative that you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you
    intend. The surface is described using a grid like the following:
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
  """

  metadata = {"render.modes": ["human", "ansi"]}

  def __init__(self, desc=None, map_name="4x4", is_slippery=True):
    if desc is None and map_name is None:
      desc = generate_random_map()
    elif desc is None:
      desc = MAPS[map_name]
    self.desc = desc = np.asarray(desc, dtype="c")
    self.nrow, self.ncol = nrow, ncol = desc.shape
    self.reward_range = (0, 1)

    nA = 4
    nS = nrow * ncol
    isd = np.array(desc == b"S").astype("float64").ravel()
    isd /= isd.sum()

    P = {s: {a: [] for a in range(nA)} for s in range(nS)}

    def to_s(row, col):
      return row * ncol + col

    def inc(row, col, a):
      if a == LEFT:
        col = max(col - 1, 0)
      elif a == DOWN:
        row = min(row + 1, nrow - 1)
      elif a == RIGHT:
        col = min(col + 1, ncol - 1)
      elif a == UP:
        row = max(row - 1, 0)
      return (row, col)

    start_state = to_s(0, 0)
    for row in range(nrow):
      for col in range(ncol):
        s = to_s(row, col)
        letter = desc[row, col]
        for a in range(4):
          li = P[s][a]
          if letter in b"GH":
            li.append((1.0, start_state, 0.0, False))
          else:
            if is_slippery:
              for b in [(a - 1) % 4, a, (a + 1) % 4]:
                newrow, newcol = inc(row, col, b)
                newstate = to_s(newrow, newcol)
                newletter = desc[newrow, newcol]
                done = bytes(newletter) in b"GH"
                # if goal reward 1.0, if hole negative reward for the reset cost
                rew = 0.
                if newletter == b"G":
                  rew = 10.0
                elif newletter == b"H":
                  rew = -10.0
                li.append((1.0 / 3.0, newstate, rew, done))
            else:
              newrow, newcol = inc(row, col, a)
              newstate = to_s(newrow, newcol)
              newletter = desc[newrow, newcol]
              done = bytes(newletter) in b"GH"
              # if goal reward 1.0, if hole negative reward for the reset cost
              rew = 0.
              if newletter == b"G":
                rew = 10.0
              elif newletter == b"H":
                rew = -10.0
              li.append((1.0, newstate, rew, done))

    super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

  def render(self, mode="human"):
    outfile = io.StringIO() if mode == "ansi" else sys.stdout

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = self.desc.tolist()
    desc = [[c.decode("utf-8") for c in line] for line in desc]
    desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
    if self.lastaction is not None:
      outfile.write("  ({})\n".format(["Left", "Down", "Right",
                                       "Up"][self.lastaction]))
    else:
      outfile.write("\n")
    outfile.write("\n".join("".join(line) for line in desc) + "\n")

    if mode != "human":
      with contextlib.closing(outfile):
        return outfile.getvalue()


class FrozenLakeCont(FrozenLakeEnv):
  """Modified version of continuous FrozenLake which converts integer states to one hot encoding."""

  def __init__(self,
               map_name=None,
               is_slippery=False,
               continual=False,
               reset_reward=True):
    super().__init__(map_name=map_name, is_slippery=is_slippery)
    self.observation_space = spaces.Box(
        low=np.zeros(self.nS), high=np.ones(self.nS))
    self._continual = continual
    self._reset_reward = reset_reward

  def _s_to_one_hot(self, s):
    one_hot = np.zeros(self.nS)
    one_hot[s] = 1.
    return one_hot

  def step(self, a):
    (s, r, done, info) = super().step(a)
    one_hot = self._s_to_one_hot(s)

    # pretend fail states do not have fail reward
    if r == -10. and not self._reset_reward:
      r = 0.

    if self._continual:
      done = False
    return (one_hot, r, done, info)

  def reset(self):
    s = super().reset()
    one_hot = self._s_to_one_hot(s)
    return one_hot

  def get_failure_state_vector(self):
    failure_state_vector = np.zeros(self.nrow * self.ncol, dtype=np.float32)
    for row in range(self.nrow):
      for col in range(self.ncol):
        s = row * self.ncol + col
        letter = self.desc[row, col]
        if letter == b"H":
          failure_state_vector[s] = 1.
    return failure_state_vector
