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

"""Simple query engine implementations for routing problems."""

import itertools
import math
import random
from typing import Optional, Union
import numpy as np
from semantic_routing.benchmark import utils
from semantic_routing.benchmark.query_engines import query_engine

PREF_TYPE_WEIGHTS = (0.6, 0.2, 0.2, 0)


class POIBasedRoutingQueryEngine(query_engine.QueryEngine):
  """Simple query engine for routing tasks."""

  max_query_size: int = 3
  seed: int = 0
  query_either_or_weight: float = 0.3  # Weight penalty for "or" clause.
  max_poi_count: int = 4  # Maximum number of POI clauses.
  clause_poly_weight: float = 3

  def __init__(
      self,
      poi_specs,
      splits = None,
      seed = None,
      query_either_or_weight = None,
      max_poi_count = None,
  ):
    if splits is not None:
      self.splits = splits
    if seed is not None:
      self.seed = seed
    if query_either_or_weight is not None:
      self.query_either_or_weight = query_either_or_weight
    if max_poi_count is not None:
      self.max_poi_count = max_poi_count

    self.poi_specs = poi_specs
    self.poi_types = {}
    general_pois, specialized_pois = self.poi_specs
    for poi_info in general_pois + sum(specialized_pois.values(), []):
      self.poi_types[poi_info["poi_type_id"]] = poi_info[
          "poi_type_name"
      ].replace("_", " ")
    self.poi_options = list(self.poi_types.keys())

    self.rng = random.Random()
    self.rng.seed(self.seed)
    self.num_perms = [
        max(
            1,
            min(
                math.perm(len(self.poi_options), i),
                10 * len(self.poi_options) * (i**self.clause_poly_weight),
            ),
        )
        for i in range(self.max_poi_count + 1)
    ]
    self.clause_weight = [
        1
        / (
            self.num_perms[i]
            * (1 if i < 2 else 1 + self.query_either_or_weight)
        )
        for i in range(self.max_poi_count + 1)
    ]
    self._divide_dataset()

  def sample_query(
      self, split, rng = None
  ):
    rng = rng or self.rng

    # Sample a query.
    if split == 0:
      queries, weights = self.train_queries
    elif split == 1:
      queries, weights = self.val_queries
    elif split == 2:
      queries, weights = self.test_queries
    else:
      raise ValueError("No dataset split specified.")

    poi_query, query_text = rng.choices(queries, weights=weights)[0]
    return (
        {"pois": poi_query, "linear": ""},
        query_text,
    )

  def _create_query_text(self, query_data):
    if not query_data["pois"]:
      return "none"
    pois = [self.poi_types[p[0]] for p in query_data["pois"] if len(p) == 1]
    pois += [
        "either {} or {}".format(self.poi_types[p[0]], self.poi_types[p[1]])
        for p in query_data["pois"]
        if len(p) > 1
    ]
    return " and ".join(pois)

  def _divide_dataset(self):
    # Divide queries by training, validation and test split
    complex_query_weights_dict: dict[
        Union[tuple[()], tuple[tuple[query_engine.POIType, Ellipsis], Ellipsis]], float
    ] = {}
    simple_query_weights_dict: dict[
        Union[tuple[()], tuple[tuple[query_engine.POIType, Ellipsis], Ellipsis]], float
    ] = {}  # Queries with at most 1 clause.
    for i in range(0, min(self.max_poi_count + 1, len(self.poi_options) + 1)):
      possible_permutations = list(
          itertools.permutations(self.poi_options, r=i)
      )
      if i > 0:
        possible_permutations = self.rng.sample(
            possible_permutations, k=self.num_perms[i]
        )
      for pois in possible_permutations:
        query_data = tuple((p,) for p in pois)
        if i < 2:  # Simple queries.
          if query_data in simple_query_weights_dict:
            continue
          simple_query_weights_dict[query_data] = self.clause_weight[i]
          continue

        if query_data in complex_query_weights_dict:
          continue
        complex_query_weights_dict[query_data] = self.clause_weight[i]

        query_data = tuple((p,) for p in pois[:-2]) + (tuple(pois[-2:]),)
        if query_data in complex_query_weights_dict:
          continue
        complex_query_weights_dict[query_data] = (
            self.clause_weight[i] * self.query_either_or_weight
        )

    # Divide queries, ensuring simple queries go to training split.
    complex_query_weights = [
        (
            (
                query_data,
                self._create_query_text({"pois": query_data, "linear": ""}),
            ),
            weight,
        )
        for query_data, weight in complex_query_weights_dict.items()
    ]
    simple_query_weights = [
        (
            (
                query_data,
                self._create_query_text({"pois": query_data, "linear": ""}),
            ),
            weight,
        )
        for query_data, weight in simple_query_weights_dict.items()
    ]

    self.rng.shuffle(complex_query_weights)
    num_train = int(self.splits[0] * len(complex_query_weights))
    num_val = int(self.splits[1] * len(complex_query_weights))

    self.train_queries = (
        complex_query_weights[:num_train] + simple_query_weights
    )
    self.train_queries = [
        query_tuple for query_tuple, _ in self.train_queries
    ], [weight for _, weight in self.train_queries]
    self.val_queries = complex_query_weights[num_train : num_train + num_val]
    self.val_queries = [query_tuple for query_tuple, _ in self.val_queries], [
        weight for _, weight in self.val_queries
    ]
    self.test_queries = complex_query_weights[num_train + num_val :]
    self.test_queries = [query_tuple for query_tuple, _ in self.test_queries], [
        weight for _, weight in self.test_queries
    ]


class POIBasedTouringQueryEngine(POIBasedRoutingQueryEngine):
  """Simple query engine for tour planning tasks."""

  max_query_size: int = 3
  seed: int = 0
  binomial_denom: int = 10
  min_time_budget: int = 2
  mean_time_budget: int = 10
  touring_prop: float = 1

  def __init__(
      self,
      poi_specs,
      splits = None,
      seed = None,
  ):
    super().__init__(poi_specs, splits, seed)
    self.nprng = np.random.RandomState(seed=self.seed)

  def sample_query(
      self, split, rng = None
  ):
    rng = rng or self.rng

    # Sample a query.
    if split == 0:
      poi_queries, weights = self.train_poi_queries
    elif split == 1:
      poi_queries, weights = self.val_poi_queries
    elif split == 2:
      poi_queries, weights = self.test_poi_queries
    else:
      raise ValueError("No dataset split specified.")

    poi_query, query_text = rng.choices(poi_queries, weights=weights)[0]

    if rng.random() < self.touring_prop:
      time_budget = self.nprng.normal(
          self.mean_time_budget, self.mean_time_budget / 4
      )
      time_budget = int(time_budget)
      query_text += ". My time budget is {}".format(time_budget)
      return (
          {"pois": poi_query, "linear": "", "time_budget": time_budget},
          query_text,
      )
    else:
      pref_query = rng.choices(utils.PREF_TYPES, PREF_TYPE_WEIGHTS)[0]
      if pref_query:
        query_text += ". I " + pref_query + "."
      return (
          {"pois": poi_query, "linear": pref_query},
          query_text,
      )
