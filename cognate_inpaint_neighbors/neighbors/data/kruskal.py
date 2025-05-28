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

"""Pairwise string algorithm algorithm from Sankoff and Kruskal's book.

Source:
-------
  David Sankoff and Joseph Kruskal (1983): "Time Warps, String Edits, and
  Macromolecules: The Theory and Practice of Sequence Comparision",
  Addison-Wesley Publishing Company.
"""


def _default_ins(item):  # pylint: disable=unused-argument
  return 1


def _default_del(item):  # pylint: disable=unused-argument
  return 1


def _default_sub(item1, item2):
  if item1 == item2:
    return 0
  return 1


_INS = _default_ins
_DEL = _default_del
_SUB = _default_sub


class Cell:
  """Entry in a matrix."""

  def __init__(self, cost, bakptr, elt1, elt2):
    self.cost_ = cost
    self.bakptr_ = bakptr
    self.elt1_ = elt1
    self.elt2_ = elt2
    return

  def cost(self):
    return self.cost_

  def back_pointer(self):
    return self.bakptr_

  def pair(self):
    return (self.elt1_, self.elt2_)


class Matrix:
  """Dynamic programming matrix."""

  def __init__(self, list1, list2):
    list1 = [None] + list(list1)
    list2 = [None] + list(list2)
    self.max1_ = len(list1)
    self.max2_ = len(list2)
    self.data_ = {}
    self.data_[(0, 0)] = Cell(0, None, None, None)
    cum = 0
    for i in range(1, self.max1_):
      cum += _INS(list1[i])
      self.data_[(i, 0)] = Cell(cum, self.data_[(i-1, 0)], list1[i], None)
    cum = 0
    for i in range(1, self.max2_):
      cum += _DEL(list2[i])
      self.data_[(0, i)] = Cell(cum, self.data_[(0, i-1)], None, list2[i])
    for i in range(1, self.max1_):
      for j in range(1, self.max2_):
        l1el = list1[i]
        l2el = list2[j]
        c1 = self.data_[(i, j-1)].cost() + _INS(l1el)
        c2 = self.data_[(i-1, j)].cost() + _DEL(l2el)
        c3 = self.data_[(i-1, j-1)].cost() + _SUB(l1el, l2el)
        if c1 <= c2 and c1 <= c3:
          self.data_[(i, j)] = Cell(c1, self.data_[(i, j-1)], None, l2el)
        elif c2 <= c1 and c2 <= c3:
          self.data_[(i, j)] = Cell(c2, self.data_[(i-1, j)], l1el, None)
        else:
          self.data_[(i, j)] = Cell(c3, self.data_[(i-1, j-1)], l1el, l2el)
    c = self.data_[(self.max1_-1, self.max2_-1)]
    path = []
    while c:
      if c.pair() != (None, None):
        path.append(c.pair())
      c = c.back_pointer()
    path.reverse()
    self._path = path
    return

  def path(self):
    return self._path


def best_match(l1, l2):
  m = Matrix(l1, l2)
  cost = m.data_[(m.max1_-1, m.max2_-1)].cost_
  return cost, m.path()
