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

"""EdgeBank is a simple strong baseline for dynamic link prediction

it predicts the existence of edges based on their history of occurrence

Reference:
    - https://github.com/fpour/DGB/tree/main
"""

import warnings
import numpy as np


class EdgeBankPredictor(object):

  def __init__(
      self,
      src,
      dst,
      ts,
      memory_mode = 'unlimited',  # could be `unlimited` or `fixed_time_window`
      time_window_ratio = 0.15,
      pos_prob = 1.0,
  ):
    r"""intialize edgebank and specify the memory mode

    Parameters:
        src: source node id of the edges for initialization
        dst: destination node id of the edges for initialization
        ts: timestamp of the edges for initialization
        memory_mode: 'unlimited' or 'fixed_time_window'
        time_window_ratio: the ratio of the time window length to the total time
        length
        pos_prob: the probability of the link existence for the edges in memory
    """
    assert memory_mode in [
        'unlimited',
        'fixed_time_window',
    ], 'Invalide memory mode for EdgeBank!'
    self.memory_mode = memory_mode
    if self.memory_mode == 'fixed_time_window':
      self.time_window_ratio = time_window_ratio
      # determine the time window size based on ratio from the given src, dst, and ts for initialization
      duration = ts.max() - ts.min()
      self.prev_t = ts.min() + duration * (
          1 - time_window_ratio
      )  # the time windows starts from the last ratio% of time
      self.cur_t = ts.max()
      self.duration = self.cur_t - self.prev_t
    else:
      self.time_window_ratio = -1
      self.prev_t = -1
      self.cur_t = -1
      self.duration = -1

    self.memory = {}  # {(u,v):1}
    self.pos_prob = pos_prob
    self.update_memory(src, dst, ts)

  def update_memory(self, src, dst, ts):
    r"""generate the current and correct state of the memory with the observed edges so far

    note that historical edges may include training, validation, and already
    observed test edges
    Parameters:
        src: source node id of the edges
        dst: destination node id of the edges
        ts: timestamp of the edges
    """
    if self.memory_mode == 'unlimited':
      self._update_unlimited_memory(src, dst)  # ignores time
    elif self.memory_mode == 'fixed_time_window':
      self._update_time_window_memory(src, dst, ts)
    else:
      raise ValueError('Invalide memory mode!')

  @property
  def start_time(self):
    """return the start of time window for edgebank `fixed_time_window` only

    Returns:
        start of time window
    """
    if self.memory_mode == 'unlimited':
      warnings.warn(
          'start_time is not defined for unlimited memory mode, returns -1'
      )
    return self.prev_t

  @property
  def end_time(self):
    """return the end of time window for edgebank `fixed_time_window` only

    Returns:
        end of time window
    """
    if self.memory_mode == 'unlimited':
      warnings.warn(
          'end_time is not defined for unlimited memory mode, returns -1'
      )
    return self.cur_t

  def _update_unlimited_memory(
      self, update_src, update_dst
  ):
    r"""update self.memory with newly arrived src and dst

    Parameters:
        src: source node id of the edges
        dst: destination node id of the edges
    """
    for src, dst in zip(update_src, update_dst):
      if (src, dst) not in self.memory:
        self.memory[(src, dst)] = 1

  def _update_time_window_memory(
      self,
      update_src,
      update_dst,
      update_ts,
  ):
    r"""move the time window forward until end of dst timestamp here

    also need to remove earlier edges from memory which is not in the time
    window
    Parameters:
        update_src: source node id of the edges
        update_dst: destination node id of the edges
        update_ts: timestamp of the edges
    """

    # * initialize the memory if it is empty
    if len(self.memory) == 0:
      for src, dst, ts in zip(update_src, update_dst, update_ts):
        self.memory[(src, dst)] = ts
      return None

    # * update the memory if it is not empty
    if update_ts.max() > self.cur_t:
      self.cur_t = update_ts.max()
      self.prev_t = self.cur_t - self.duration

    # * add new edges to the time window
    for src, dst, ts in zip(update_src, update_dst, update_ts):
      self.memory[(src, dst)] = ts

  def predict_link(
      self, query_src, query_dst
  ):
    r"""predict the probability from query src,dst pair given the current memory,

    all edges not in memory will return 0.0 while all observed edges in memory
    will return self.pos_prob
    Parameters:
        query_src: source node id of the query edges
        query_dst: destination node id of the query edges
    Returns:
        pred: the prediction for all query edges
    """
    pred = np.zeros(len(query_src))
    idx = 0
    for src, dst in zip(query_src, query_dst):
      if (src, dst) in self.memory:
        if self.memory_mode == 'fixed_time_window':
          if self.memory[(src, dst)] >= self.prev_t:
            pred[idx] = self.pos_prob
        else:
          pred[idx] = self.pos_prob
      idx += 1

    return pred
