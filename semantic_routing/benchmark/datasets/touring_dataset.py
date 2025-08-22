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

"""Implements a dataset class for tour planning tasks."""

import itertools
from typing import Union
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.datasets import routing_dataset


class TouringDataset(routing_dataset.RoutingDataset):
  """Dataset class for tour planning tasks."""

  def sample_datapoint(
      self,
      labeled,
      split,
      prefix_len = -1,
      use_fresh = True,
  ):
    """Sample a next-edge prediction datapoint."""

    assert (
        prefix_len == -1 or prefix_len == 0 or prefix_len >= self.min_prefix_len
    )
    assert prefix_len <= self.max_segments - 1
    assert self.max_sample_attempts >= 1
    assert use_fresh or labeled

    if prefix_len == 0:
      prefix_len = self.min_prefix_len

    if (
        self._current_params != (labeled, split, self.min_prefix_len)
        or use_fresh
    ):  # Cache miss
      self._current_params = None
      end = None
      edgelist = None
      query_text = None
      query_data = None
      found = False
      parent = self
      path_info = None
      subset = None
      for _ in range(self.max_sample_attempts):
        query_data, query_text = self.query_engine.sample_query(split, self.rng)
        start = self.rng.choice(self.road_graph.central_edges)
        if "time_budget" in query_data:
          end = start
          for r in range(
              len(query_data["pois"]),
              int((len(query_data["pois"]) - 1) / 2),
              -1,
          ):  # r is the size of the subsets
            options = []
            for subset in itertools.combinations(query_data["pois"], r):
              this_data = query_data.copy()
              this_data["pois"] = subset
              this_info = self.road_graph.get_shortest_path_len(
                  start, end, this_data, return_path=True
              )
              if this_info is None:
                continue
              if this_info[0] > query_data["time_budget"] * 60:
                continue
              options.append((subset, this_info))
            if options:
              subset, path_info = min(options, key=lambda x: x[1][0])
              break
        else:
          end = self.road_graph.sample_noncentral_edge(split, self.rng)
          path_info = self.road_graph.get_shortest_path_len(
              start, end, query_data, return_path=True
          )
        if not path_info:
          continue
        assert isinstance(path_info, tuple)
        _, edgelist = path_info
        if len(edgelist) < self.min_prefix_len:
          continue
        if len(edgelist) <= self.min_segments:
          continue
        if prefix_len != -1 and len(edgelist) < prefix_len:
          continue
        if len(edgelist) > self.max_segments:
          continue
        if self.auto_simplify_datapoint:
          new_graph = self.road_graph.contract_graph([start, end])
          if "time_budget" in query_data:
            this_data = query_data.copy()
            this_data["pois"] = subset
            new_path_info = new_graph.get_shortest_path_len(
                start, end, this_data, return_path=True
            )
          else:
            new_path_info = new_graph.get_shortest_path_len(
                start, end, query_data, return_path=True
            )
          assert new_path_info is not None
          edgelist = new_path_info[1]
          parent = TouringDataset(
              tokenizer=self.tokenizer,
              graph=new_graph,
              engine=self.query_engine,
              poi_specs=self.poi_specs,
              seed=self.seed,
              receptive_field_size=self.receptive_field_size,
              poi_receptive_field_size=self.poi_receptive_field_size,
              auto_simplify_datapoint=False,
              max_segments=self.max_segments,
              cand_len=self.cand_len,
          )
        found = True
        break
      if not found:
        raise TimeoutError("")
      edgelist += (self.term_token,)
    else:  # Cache hit, re-use datapoint from last time
      assert self._current_datapoint is not None
      end = self._current_datapoint["end"]
      query_text = self._current_datapoint["query_text"]
      query_data = self._current_datapoint["query_data"]
      edgelist = self._current_datapoint["edgelist"]
      parent = self._current_datapoint["parent"]

    if not use_fresh:  # Cache this datapoint
      prefix_len = len(edgelist) - 1
      ground_truth = edgelist[prefix_len]  # Ground truth for CURRENT datapoint
      edgelist = edgelist[:prefix_len]  # Edge list for CURRENT datapoint
      if prefix_len >= 2:
        self._current_datapoint = dataset.DatapointType(
            end=end,
            query_text=query_text,
            query_data=query_data,
            edgelist=edgelist,
            ground_truth=ground_truth,
            candidates=(),
            parent=parent,
        )
        self._current_params = (labeled, split, self.min_prefix_len)
      else:
        self._current_datapoint = None
        self._current_params = None
    else:  # Do not cache this datapoint; pick a random position
      if prefix_len == -1:
        prefix_len = self.rng.randint(self.min_prefix_len, len(edgelist) - 1)
      if labeled:
        ground_truth = edgelist[prefix_len]
      else:
        ground_truth = None
      edgelist = edgelist[:prefix_len]
      self._current_datapoint = None
      self._current_params = None

    candidates = parent.get_candidates(edgelist)
    if edgelist[-1] == end:
      candidates += (dataset.TERM,)

    return dataset.DatapointType(
        end=end,
        query_text=query_text,
        query_data=query_data,
        edgelist=edgelist,
        ground_truth=ground_truth,
        candidates=candidates,
        parent=parent,
    )

  def evaluate_datapoint(
      self, datapoint
  ):
    if dataset.TERM in datapoint["edgelist"]:
      assert datapoint["edgelist"].count(dataset.TERM) == 1
      assert datapoint["edgelist"][-1] == dataset.TERM
      edgelist = datapoint["edgelist"][:-1]
    else:
      edgelist = datapoint["edgelist"]

    metrics = self.road_graph.route_metrics(
        datapoint["query_data"], datapoint["end"], edgelist
    )
    new_metrics = {
        k: metrics[k]
        for k in (
            "num_pois_achieved",
            "reaches_destination",
            "frac_pois_achieved",
            "travel_time",
        )
    }
    if "time_budget" in datapoint["query_data"]:
      new_metrics["task"] = "tour"
      new_metrics["tour_budget_met"] = metrics["budget_met"]
      new_metrics["mask"] = int(
          metrics["reaches_destination"] and metrics["budget_met"]
      )

      best_poi_count = 0
      for r in range(
          len(datapoint["query_data"]["pois"]), -1, -1
      ):  # r is the size of the subsets
        options = []
        for subset in itertools.combinations(
            datapoint["query_data"]["pois"], r
        ):
          this_data = datapoint["query_data"].copy()
          this_data["pois"] = subset
          this_info = datapoint["parent"].road_graph.get_shortest_path_len(
              datapoint["edgelist"][0],
              datapoint["end"],
              this_data,
              return_path=True,
          )
          if this_info is None:
            continue
          if this_info[0] > datapoint["query_data"]["time_budget"] * 60:
            continue
          options.append(this_info)
        if options:
          best_poi_count = r
          break
      new_metrics["tour_excess_frac_poi_missed"] = (
          best_poi_count - new_metrics["num_pois_achieved"]
      ) / len(datapoint["query_data"]["pois"])
      score_3 = 10000 * int(new_metrics["tour_excess_frac_poi_missed"])
      score_2 = 10000 * int(metrics["budget_met"]) + score_3
      score = 10000 * int(metrics["reaches_destination"]) + score_2
      new_metrics["score"] = score
      new_metrics["is_routing"] = 0
      new_metrics["tour_success_met"] = (
          new_metrics["tour_budget_met"]
          and new_metrics["tour_excess_frac_poi_missed"] <= 0.5
      )
    else:
      new_metrics["task"] = "routing"
      new_metrics["routing_excess_penalty"] = metrics["excess_penalty"]
      new_metrics["routing_excess_travel_time"] = metrics["excess_travel_time"]
      new_metrics["routing_poi_met"] = metrics["frac_pois_achieved"] > 0.999
      score = (
          10000 * int(metrics["reaches_destination"])
          + 10000 * int(new_metrics["routing_poi_met"])
          - metrics["excess_penalty"]
      )
      new_metrics["score"] = score
      new_metrics["mask"] = int(
          new_metrics["reaches_destination"] and new_metrics["routing_poi_met"]
      )
      new_metrics["is_routing"] = 1
    return new_metrics

  def fast_score(
      self, datapoint
  ):
    if dataset.TERM in datapoint["edgelist"]:
      assert datapoint["edgelist"].count(dataset.TERM) == 1
      assert datapoint["edgelist"][-1] == dataset.TERM
      edgelist = datapoint["edgelist"][:-1]
    else:
      edgelist = datapoint["edgelist"]

    fake_data = datapoint["query_data"].copy()
    if "time_budget" not in fake_data:
      fake_data["time_budget"] = 10000000
    metrics = self.road_graph.route_metrics(
        fake_data, datapoint["end"], edgelist
    )
    new_metrics = {
        k: metrics[k]
        for k in (
            "num_pois_achieved",
            "reaches_destination",
            "frac_pois_achieved",
            "travel_time",
        )
    }
    if "time_budget" in datapoint["query_data"]:
      metrics_5 = 10000 * int(metrics["reaches_destination"])
      metrics_4 = int(new_metrics["num_pois_achieved"])
      metrics_3 = metrics_4 - 0.001 * metrics["penalty"]
      metrics_2 = 10000 * int(metrics["budget_met"]) + metrics_3
      new_metrics["score"] = metrics_5 + metrics_2
    else:
      metrics_4 = 10000 * int(new_metrics["num_pois_achieved"]
                              ) - metrics["penalty"]
      new_metrics["score"] = (
          10000 * int(metrics["reaches_destination"]) + metrics_4
      )
    return new_metrics
