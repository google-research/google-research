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

"""This file is adapted from https://github.com/likenneth/othello_world.

Othello is a strategy board game for two players (Black and White),
played on an 8 by 8 board.
The game traditionally begins with four discs placed in the middle of the
board as shown below. Black moves first.
W (27) B (28)
B (35) W (36)
"""

import copy
import random

import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)


class Color:
  PURPLE = "\033[95m"
  CYAN = "\033[96m"
  DARKCYAN = "\033[36m"
  BLUE = "\033[94m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  RED = "\033[91m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"
  END = "\033[0m"


def permit(s):
  s = s.lower()
  if len(s) != 2:
    return -1
  if s[0] not in rows or s[1] not in columns:
    return -1
  return rows.index(s[0]) * 8 + columns.index(s[1])


def permit_reverse(integer):
  r, c = integer // 8, integer % 8
  return "".join([rows[r], columns[c]])


start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

wanna_use = "othello_synthetic"


def get_ood_game(_):
  tbr = []
  ab = OthelloBoardState()
  possible_next_steps = ab.get_valid_moves()
  while possible_next_steps:
    next_step = random.choice(possible_next_steps)
    tbr.append(next_step)
    ab.update([
        next_step,
    ])
    possible_next_steps = ab.get_valid_moves()
  return tbr


# def get(ood_perc=0., data_root=None, wthor=False, ood_num=1000):
#     return Othello(ood_perc, data_root, wthor, ood_num)


class OthelloBoardState:
  """Othello board state class.

  1 is black, -1 is white.
  Attributes:
    board_size: length (or width) of the board
    initial_state: initial board state
    state: current board state
    age: a 2D array of the same shape as board indicating whether a move has
      happened in a particular location or not.
    next_hand_color: color of the next hand
    history: list of past moves
  """

  def __init__(self, board_size=8):
    self.board_size = board_size * board_size
    board = np.zeros((8, 8))
    board[3, 4] = 1
    board[3, 3] = -1
    board[4, 3] = 1
    board[4, 4] = -1
    self.initial_state = board
    self.state = self.initial_state
    self.age = np.zeros((8, 8))
    self.next_hand_color = 1
    self.history = []

  def get_occupied(
      self,
  ):
    board = self.state
    tbr = board.flatten() != 0
    return tbr.tolist()

  def get_state(
      self,
  ):
    board = self.state + 1  # white 0, blank 1, black 2
    tbr = board.flatten()
    return tbr.tolist()

  def get_age(
      self,
  ):
    return self.age.flatten().tolist()

  def get_next_hand_color(
      self,
  ):
    return (self.next_hand_color + 1) // 2

  def update(self, moves, prt=False):
    # takes a new move or new moves and update state
    if prt:
      self.__print__()
    for _, move in enumerate(moves):
      self.umpire(move)
      if prt:
        self.__print__()

  def umpire(self, move):
    """Function to handle conducting the Othello board game."""
    r, c = move // 8, move % 8
    assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
    color = self.next_hand_color
    tbf = []
    for direction in eights:
      buffer = []
      cur_r, cur_c = r, c
      while 1:
        cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
        if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
          break
        if self.state[cur_r, cur_c] == 0:
          break
        elif self.state[cur_r, cur_c] == color:
          tbf.extend(buffer)
          break
        else:
          buffer.append([cur_r, cur_c])

    if not tbf:  # means one hand is forfeited
      color *= -1
      self.next_hand_color *= -1
      for direction in eights:
        buffer = []
        cur_r, cur_c = r, c
        while 1:
          cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
          if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
            break
          if self.state[cur_r, cur_c] == 0:
            break
          elif self.state[cur_r, cur_c] == color:
            tbf.extend(buffer)
            break
          else:
            buffer.append([cur_r, cur_c])

    if not tbf:
      valids = self.get_valid_moves()
      if not valids:
        assert 0, "Both color cannot put piece, game should have ended!"
      else:
        assert 0, "Illegal move!"

    self.age += 1
    for ff in tbf:
      self.state[ff[0], ff[1]] *= -1
      self.age[ff[0], ff[1]] = 0
    self.state[r, c] = color
    self.age[r, c] = 0
    self.next_hand_color *= -1
    self.history.append(move)

  def __print__(self):
    print("-" * 20)
    print([permit_reverse(_) for _ in self.history])
    a = "abcdefgh"
    for k, row in enumerate(self.state.tolist()):
      tbp = []
      for ele in row:
        if ele == -1:
          tbp.append("O")
        elif ele == 0:
          tbp.append(" ")
        else:
          tbp.append("X")
      print(" ".join([a[k]] + tbp))
    tbp = [str(k) for k in range(1, 9)]
    print(" ".join([" "] + tbp))
    print("-" * 20)

  def plot_hm(self, ax, heatmap, pdmove, logit=False):
    """Plot a heatmap of model's valid moves."""
    padding = np.array([0.0, 0.0])
    trs = {-1: r"O", 0: " ", 1: r"X"}
    if len(heatmap) == 60:
      heatmap = [heatmap[:27], padding, heatmap[27:33], padding, heatmap[33:]]
      heatmap = np.concatenate(heatmap)
    assert len(heatmap) == 64
    heatmap = np.array(heatmap).reshape(8, 8)
    annot = [trs[_] for _ in self.state.flatten().tolist()]
    cloned = copy.deepcopy(self)
    cloned.update([
        pdmove,
    ])

    next_color = 1 - cloned.get_next_hand_color()
    annot[pdmove] = ("\\underline{" + (trs[next_color * 2 - 1]) + "}")[-13:]

    color = {-1: "white", 0: "grey", 1: "black"}
    ann_col = [color[_] for _ in self.state.flatten().tolist()]
    text_for_next_color = color[next_color * 2 - 1].capitalize()

    del cloned
    if logit:
      max_logit = np.max(np.abs(heatmap))
      sns.heatmap(
          data=heatmap,
          cbar=False,
          xticklabels=list(range(1, 9)),
          cmap=sns.color_palette("vlag", as_cmap=True),
          yticklabels=list("ABCDEFGH"),
          ax=ax,
          fmt="",
          square=True,
          linewidths=0.5,
          vmin=-max_logit,
          vmax=max_logit,
          center=0,
      )
    else:
      sns.heatmap(
          data=heatmap,
          cbar=False,
          xticklabels=list(range(1, 9)),
          cmap=sns.color_palette("vlag", as_cmap=True),
          yticklabels=list("ABCDEFGH"),
          ax=ax,
          fmt="",
          square=True,
          linewidths=0.5,
          vmin=-1,
          vmax=1,
          center=0,
      )
    ax.set_title(
        f"Prediction: {text_for_next_color} at "
        + permit_reverse(pdmove).upper()
    )
    ax.add_patch(
        mpatches.Rectangle(
            (pdmove % 8, pdmove // 8), 1, 1, fill=False, edgecolor="black", lw=2
        )
    )

    patch_list = []
    for loca, col in enumerate(ann_col):
      if col != "grey":
        patch_list.append(
            mcollections.PatchCollection(
                [
                    mpatches.Circle(
                        (loca % 8 + 0.5, loca // 8 + 0.5), 0.25, facecolor=col
                    )
                ],
                match_original=True,
            )
        )
    for i in patch_list:
      ax.add_collection(i)
    return ax

  def tentative_move(self, move):
    """Tentatively put a piece, do nothing to state.

    Args:
      move: a move

    Returns:
      Returns 0 if this is not a move at all: occupied or both player have
        to forfeit
      Return 1 if regular move
      Return 2 if forfeit happens but the opponent can drop piece at this place
    """
    r, c = move // 8, move % 8
    if self.state[r, c] != 0:
      return 0

    color = self.next_hand_color
    tbf = []
    for direction in eights:
      buffer = []
      cur_r, cur_c = r, c
      while 1:
        cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
        if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
          break
        if self.state[cur_r, cur_c] == 0:
          break
        elif self.state[cur_r, cur_c] == color:
          tbf.extend(buffer)
          break
        else:
          buffer.append([cur_r, cur_c])

    if tbf:
      return 1
    else:  # means one hand is forfeited
      color *= -1
      for direction in eights:
        buffer = []
        cur_r, cur_c = r, c
        while 1:
          cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
          if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
            break
          if self.state[cur_r, cur_c] == 0:
            break
          elif self.state[cur_r, cur_c] == color:
            tbf.extend(buffer)
            break
          else:
            buffer.append([cur_r, cur_c])

      if not tbf:
        return 0
      else:
        return 2

  def get_valid_moves(self):
    """Get a list of valid moves from the current board state."""
    regular_moves = []
    forfeit_moves = []
    for move in range(64):
      x = self.tentative_move(move)
      if x == 1:
        regular_moves.append(move)
      elif x == 2:
        forfeit_moves.append(move)
      else:
        pass
    if regular_moves:
      return regular_moves
    elif forfeit_moves:
      return forfeit_moves
    else:
      return []

  def get_gt(self, moves, func, prt=False):
    # takes a new move or new moves and update state
    container = []
    if prt:
      self.__print__()
    for _, move in enumerate(moves):
      self.umpire(move)
      container.append(getattr(self, func)())
      # to predict first y, we need already know the first x
      if prt:
        self.__print__()
    return container


if __name__ == "__main__":
  pass
