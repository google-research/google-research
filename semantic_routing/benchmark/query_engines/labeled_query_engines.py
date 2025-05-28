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

"""Query engine implementations for human labeled queries problems."""

import ast
import csv
import random
from typing import Optional
from semantic_routing.benchmark import config
from semantic_routing.benchmark import utils
from semantic_routing.benchmark.query_engines import query_engine


class HumanLabeledQueryEngine(query_engine.QueryEngine):
  """Simple query engine for routing tasks."""

  seed: int = 0
  touring_prop: float = 0.2

  def __init__(
      self,
      poi_specs,
      splits = None,
      seed = None,
  ):
    if splits is not None:
      self.splits = splits
    if seed is not None:
      self.seed = seed
    self.poi_specs = poi_specs
    self.rng = random.Random()
    self.rng.seed(self.seed)
    self._divide_dataset()
    self.poi_map = {}
    for poiinfo in self.poi_specs[0]:
      self.poi_map[poiinfo["poi_type_name"]] = poiinfo["poi_type_id"]
    for poiinfos in self.poi_specs[1].values():
      for v in poiinfos:
        self.poi_map[v["poi_type_name"]] = v["poi_type_id"]

  def sample_query(
      self, split, rng = None
  ):
    rng = rng or self.rng

    # Routing splits
    if split == 0:
      routing_queries = self.train_routing_queries
    elif split == 1:
      routing_queries = self.val_routing_queries
    elif split == 2:
      routing_queries = self.test_routing_queries
    else:
      raise ValueError("No dataset split specified.")
    # Touring splits
    if split == 0:
      touring_queries = self.train_touring_queries
    elif split == 1:
      touring_queries = self.val_touring_queries
    elif split == 2:
      touring_queries = self.test_touring_queries
    else:
      raise ValueError("No dataset split specified.")

    if rng.random() < self.touring_prop:
      tour_query = rng.choice(touring_queries)
      assert isinstance(tour_query["Query"], str)
      places = tour_query["Places"]
      return {
          "pois": tuple((self.poi_map[p],) for p in ast.literal_eval(places)),
          "linear": "",
          "time_budget": float(tour_query["Time"]),
      }, tour_query[
          "Query"
      ]  # type: ignore

    found = False
    pois = None
    routing_query = None
    while not found:
      routing_query = rng.choice(routing_queries)
      assert isinstance(routing_query["Query"], str)
      if routing_query["Places"] == "[[[]]]":
        pois = ()
      else:
        pois = tuple(
            tuple(self.poi_map[pp] for pp in p)
            for p in ast.literal_eval(routing_query["Places"])
        )
        # For location waypoint routing, change condition to:
        #   max(len(p) for p in pois) <= 1
        # For errand waypoint routing, change condition to:
        #   min([len(p) for p in pois]) >= 2
        found = True
      if found:
        break
    assert pois is not None
    assert routing_query is not None
    return {
        "pois": pois,
        "linear": routing_query["Highway Pref"],
    }, routing_query[
        "Query"
    ]  # type: ignore

  def _divide_dataset(self):
    # For routing
    self.train_routing_queries = []
    self.val_routing_queries = []
    self.test_routing_queries = []

    with open(config.ROUTING_DATASET_PATH, "r") as f:
      rows = list(csv.DictReader(f))
    idxs = self.rng.sample(list(range(len(rows))), len(rows))
    num_train = int(self.splits[0] * len(idxs))
    num_val = int(self.splits[1] * len(idxs))

    for idx in idxs[:num_train]:
      self.train_routing_queries.append(rows[idx])
    for idx in idxs[num_train : num_train + num_val]:
      self.val_routing_queries.append(rows[idx])
    for idx in idxs[num_train + num_val :]:
      self.test_routing_queries.append(rows[idx])

    # For touring
    self.train_touring_queries = []
    self.val_touring_queries = []
    self.test_touring_queries = []

    with open(config.TOURING_DATASET_PATH, "r") as f:
      rows = list(csv.DictReader(f))
    idxs = self.rng.sample(list(range(len(rows))), len(rows))
    num_train = int(self.splits[0] * len(idxs))
    num_val = int(self.splits[1] * len(idxs))

    for idx in idxs[:num_train]:
      self.train_touring_queries.append(rows[idx])
    for idx in idxs[num_train : num_train + num_val]:
      self.val_touring_queries.append(rows[idx])
    for idx in idxs[num_train + num_val :]:
      self.test_touring_queries.append(rows[idx])
