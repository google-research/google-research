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

"""Collaborative filtering model, predicts based on popular choice."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from hyperbolic.models.base import CFModel
import hyperbolic.utils.popchoice as popchoice


class PopChoice(CFModel):
  """model class for popular choice predictions."""

  def __init__(self, n_items, train):
    self.item_to_deg = popchoice.degree_of_item(train)
    self.sorted_by_deg = popchoice.sorted_by_degrees(self.item_to_deg)
    self.n_items = n_items

  def get_queries(self, input_tensor):
    pass

  def get_rhs(self, input_tensor):
    pass

  def get_candidates(self, input_tensor=None):
    pass

  def similarity_score(self, lhs, rhs, eval_mode):
    pass

  def call(self, input_tensor, eval_mode=False):
    pass

  def get_scores_targets(self, input_tensor):
    scores = popchoice.rank_all_by_pop(self.sorted_by_deg)
    targets = scores[input_tensor.numpy()[:, 1]].reshape(-1, 1)
    scores = np.broadcast_to(scores,
                             (tf.shape(input_tensor)[0], self.n_items)).copy()
    return scores, targets
