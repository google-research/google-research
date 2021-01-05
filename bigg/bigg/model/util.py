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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from bigg.common.consts import t_float
from bigg.common.pytorch_util import glorot_uniform, MLP


class AdjNode(object):
    def __init__(self, parent, row, col_range, depth):
        self.parent = parent
        self.depth = depth
        self.row = row
        self.col_range = col_range

        self.n_cols = self.col_range[1] - self.col_range[0]
        self.is_leaf = self.n_cols <= 1
        self.is_root = self.parent is None
        self.edge = None
        self.has_edge = False
        if self.is_leaf:
            self.edge = (self.row, self.col_range[0])

        self._state = None
        self._bits_rep = None
        self.lch = None
        self.rch = None

    @property
    def state(self):
        return self._state

    @property
    def bits_rep(self):
        if self._bits_rep is None and not self.is_leaf:
            bits = []
            if self.lch._bits_rep is not None:
                offset = self.rch.col_range[1] - self.rch.col_range[0]
                bits += [x + offset for x in self.lch._bits_rep]
            if self.rch._bits_rep is not None:
                bits += self.rch._bits_rep
            self._bits_rep = bits
        return self._bits_rep

    @bits_rep.setter
    def bits_rep(self, new_bits):
        self._bits_rep = new_bits

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def split(self):
        if (self.lch is not None and self.rch is not None) or self.is_leaf:
            return

        col_ranges = [(self.col_range[0], self.col_range[0] + self.n_cols // 2),
                        (self.col_range[0] + self.n_cols // 2, self.col_range[1])]
        self.lch = AdjNode(self, self.row, col_ranges[0], self.depth + 1)
        self.rch = AdjNode(self, self.row, col_ranges[1], self.depth + 1)


class ColAutomata(object):
    def __init__(self, supervised, indices=None):
        self.pos = 0
        self.indices = indices
        self.supervised = supervised
        if indices is None:
            self.indices = []

    @property
    def next_edge(self):
        if self.pos < len(self.indices):
            return self.indices[self.pos]
        else:
            return None

    @property
    def last_edge(self):
        if self.pos < len(self.indices):
            return self.indices[-1]
        else:
            return None

    def add_edge(self, col_idx):
        self.pos += 1
        if not self.supervised:
            self.indices.append(col_idx)

    def has_edge(self, range_start, range_end):
        for i in self.indices:
            if i >= range_start and i < range_end:
                return True
        return False


class AdjRow(object):
    def __init__(self, row, directed=False, self_loop=False, col_range=None):
        self.row = row
        assert not directed
        max_col = row
        if self_loop:
            max_col += 1
        if col_range is None:
            col_range = (0, max_col)
        self.root = AdjNode(None, row, col_range, 0)
