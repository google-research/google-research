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

"""Class for progressive sequential synthesizer."""

from __future__ import annotations

import enum
import random
import sys
import time
import traceback
from typing import List, Sequence, Tuple, Union, Optional

from absl import logging

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.model.concrete import Op
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphSpec
from abstract_nas.synthesis.enum_sequential import EnumerativeSequentialSynthesizer


def log_exc():
  exc_type, exc_value, exc_traceback = sys.exc_info()
  logging.info("".join(
      traceback.format_exception(exc_type, exc_value, exc_traceback)))


class ProgressiveSequentialSynthesizer(EnumerativeSequentialSynthesizer):
  """Synthesizer that selects ops by the progress measure.

  This synthesizer only works for sequential subgraphs (see sequential.py).

  There are three modes for the progressive synthesizer.
  - GREEDY selects the op which maximizes the progress at each step.
  - UNIFORM selects an op uniformly at random from the ops which make progress.
  - WEIGHTED selects an op weighted by the amount of progress it makes.

  For weighted, each op is selected with probability proportional to
    1. / (dist + self.eps)
  where dist is the distance of the resulting subgraph to the target properties.
  """

  class Mode(enum.Enum):
    GREEDY = 1
    UNIFORM = 2
    WEIGHTED = 3

  def __init__(self,
               subgraphs_and_props,
               generation,
               abstract = True,
               mode = Mode.GREEDY,
               max_len = -1,
               max_delta = -1,
               min_len = 0,
               min_delta = -1,
               eps = 1,
               exp = 1,
               p = 0.2,
               filter_progress = True):
    """Initializes a synthesizer.

    Args:
      subgraphs_and_props: A list of tuples providing subgraphs providing the
        enclosing contexts into which the synthesized subgraph will be embedded,
        and the properties for each instantiation to satisfy. Note that each
        subgraph will have the same graph, but differing constants, state, and
        inputs.
      generation: The generation of the new subgraph, the value of which must be
        incremented at least once every time a graph is selected to have a
        subgraph in it replaced. This is primarily to ensure that the
        synthesizer is easily able to generate unique names for any newly
        inserted nodes.
      abstract: Whether the properties of the synthesize subgraph need to to
        satisfy the properties abstractly or concretely (i.e., using actual
        values).
      mode: the progressive synthesizer mode.
      max_len: the maximum length of the subgraph to synthesize.
      max_delta: the maximum additional length over the number of ops in the
        original graph to synthesize.
      min_len: the minimum length of the subgraph to synthesize.
      min_delta: the minimum difference from the number of ops in the original
        graph to synthesize.
      eps: the minimum amount of probability to add for WEIGHTED mode.
      exp: the power to raise the weights to for WEIGHTED mode.
      p: the probability of continuing once the distance is 0.
      filter_progress: whether to only select ops that make positive progress.

    Raises:
      ValueError: If both max_delta and max_len are provided.
    """
    super().__init__(subgraphs_and_props, generation, abstract)
    if max_len >= 0 and max_delta >= 0:
      raise ValueError("Provided both max_len and max_delta.")
    if isinstance(mode, str):
      mode = self.Mode[mode.upper()]
    self.mode = mode
    self.filter_progress = filter_progress
    self.max_len = max_len
    self.max_delta = max_delta
    self.min_len = min_len
    self.min_delta = min_delta
    self.eps = eps
    self.exp = exp
    self.p = p

  def synthesize(self):
    """Synthesizes a subgraph satisfying all the properties."""
    subg_ops = []
    if self.max_delta > 0:
      max_len = self.num_ops + self.max_delta
    else:
      max_len = self.max_len
    if self.min_delta > 0:
      min_len = max(0, self.num_ops - self.min_delta)
    else:
      min_len = self.min_len
    last_dist = None
    syn_start = time.time()
    while True:
      if len(subg_ops) > min_len and last_dist == 0:
        # Only continue after properties are satisfied if:
        # - Greater than two ops away from the max length
        # - Unbounded max length
        if max_len > 0 and len(subg_ops) <= max_len - 2 or max_len == 0:
          if random.random() > self.p: break
        else:
          break

      last_len = len(subg_ops)
      if max_len > 0 and len(subg_ops) >= max_len:
        logging.warn("Maximum length %d exceeded.", max_len)
        break
      maybe_subg_and_dist = self.synthesize_one(subg_ops)
      assert len(subg_ops) == last_len
      if maybe_subg_and_dist is None:
        if last_dist != 0:
          logging.warn("Could not finish synthesis.")
          break
        else:
          break
      subg_ops, dist = maybe_subg_and_dist
      logging.info("new dist: %s", dist)
      for op in subg_ops:
        logging.info("  %s\n"
                     "    op_kwargs=%s\n"
                     "    input_kwargs=%s\n",
                     op.name, op.op_kwargs, op.input_kwargs)
      last_dist = dist
    subg_models = self.randomize_subgraph(subg_ops, self.kwarg_defaults)
    logging.info("Finished synthesis in %d sec.", time.time() - syn_start)
    return subg_models

  def synthesize_one(
      self,
      subg_ops,
      last_dist = None):
    """Appends a single op to subg_ops and computes the updated distance."""
    start = time.time()

    subgs_and_dists = self.append_one_op_distances(subg_ops)
    if self.filter_progress and last_dist is not None:
      subgs_and_dists = [
          (subg, dist) for subg, dist in subgs_and_dists if dist < last_dist
      ]
    if not subgs_and_dists:
      return
    dists = [dist for _, dist in subgs_and_dists]

    # Filter ops by progress.
    if (self.mode is self.Mode.GREEDY or
        self.max_len > 0 and len(subg_ops) >= self.max_len - 2):
      min_dist = min(dists)
      subgs_and_dists = [
          (subg, dist) for subg, dist in subgs_and_dists if dist == min_dist
      ]
      dists = [min_dist] * len(subgs_and_dists)
    elif (self.eps == 0 or self.mode == self.Mode.UNIFORM) and 0 in dists:
      subgs_and_dists = [
          (subg, dist) for subg, dist in subgs_and_dists if dist == 0
      ]
      dists = [0] * len(subgs_and_dists)

    # Compute weights according to mode and filter by type.
    if self.mode is self.Mode.WEIGHTED:
      weights = [1. / (dist + self.eps) ** self.exp for dist in dists]

      op_types_to_weights = {}
      subgs_and_dists_and_weights = []
      for (subg, dist), weight in zip(subgs_and_dists, weights):
        op = subg[-1]
        subgs_and_dists_and_weights.append((subg, dist, weight))
        cur_weight = op_types_to_weights.get(op.type, 0)
        op_types_to_weights[op.type] = max(cur_weight, weight)
      types, type_weights = zip(*op_types_to_weights.items())
      op_type = random.choices(types, weights=type_weights)[0]
      subgs_and_dists_and_weights = [
          (subg, weight, dist)
          for subg, weight, dist in subgs_and_dists_and_weights
          if subg[-1].type == op_type
      ]
    else:
      op_types = list(set([subg[-1].type for subg, _ in subgs_and_dists]))
      op_type = random.choice(op_types)
      subgs_and_dists_and_weights = [(subg, dist, 1)
                                     for subg, dist in subgs_and_dists
                                     if subg[-1].type == op_type]

    logging.info("Selected op_type %s (%d sec).", op_type.name,
                 time.time() - start)
    weights = [weight for _, _, weight in subgs_and_dists_and_weights]
    subg, dist, weight = random.choices(subgs_and_dists_and_weights,
                                        weights=weights)[0]

    return subg, dist

  def append_one_op_distances(
      self, subg_ops):
    """Enumerate ops and computes the distance after appending each to subg_ops.

    Note that this function needs to return the entire sequence of ops after
    appending (instead of just the op that is appended), as we may need to also
    mutate some previous ops in the process. In particular, if the subgraph is
    approaching the max length, we will try to adjust the output features to
    match the shape property.

    Args:
      subg_ops: a subgraph (as a list of ops).

    Returns:
      A list of tuples, where the first element is the input subgraph plus a
        single op, and the second element is the distance.
    """
    subgs_and_dists = []
    prefix = f"gen{self.generation}/"
    adjust_features = len(subg_ops) >= self.max_len - 1
    for op in self.op_enumerator(prefix, self.kwarg_defaults, full=False):
      logging.info("Trying %s", op.name)
      subg_ops.append(op)
      try:
        subg_spec = self.make_subgraph_spec(subg_ops, adjust_features)
        total_dist = self.get_subg_dist(subg_spec)
        logging.info("Dist: %f", total_dist)
        subg = [subg_node.op for subg_node in subg_spec]
        subgs_and_dists.append((subg, total_dist))
      except Exception:  # pylint: disable=broad-except
        log_exc()
      subg_ops.pop(-1)
    return subgs_and_dists

  def get_subg_dist(self, subg_spec):
    """Given a subgraph, returns its distance to the desired properties."""
    subg_models = self.make_subgraph_models(subg_spec)
    dist = {}
    for node in subg_spec:
      logging.info(node.op.name)
    for subg_model, (_, properties) in zip(subg_models,
                                           self.subgraphs_and_props):
      cur_dist = {}
      for prop in properties:
        new_prop = prop.infer(subg_model, abstract=self.abstract)
        cur_dist[type(prop)] = prop.distance_from(new_prop)
      for key in cur_dist:
        if key not in dist:
          dist[key] = cur_dist[key]
        else:
          assert dist[key] == cur_dist[key]
    logging.info(dist)
    total_dist = 0
    for _, d in dist.items():
      total_dist += d
    return total_dist
