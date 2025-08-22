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

import math
import random
import sys
from geopy.distance import distance
import pandas as pd
from tqdm import tqdm

OSM_DIR = "*-PROCESSED DIRECTORY NAME"
NODES_FILE = "nodes.tsv"
OUTPUT_FILE_PREFIX = "medium_queries_random_"

FUNCTION = lambda x: math.pow(x, 1.0)
INVERSE = lambda x: math.pow(x, 1.0)
TILES = 100
MILE_LOWER_BOUND = 5
MILE_UPPER_BOUND = 20


def select_random_queries(n_queries):
  all_nodes = pd.read_csv(OSM_DIR + "/" + NODES_FILE, delimiter="\t")
  S = []
  while len(S) < n_queries:
    source = random.randrange(len(all_nodes))
    # distribution biased towards small distances
    ub = FUNCTION(len(all_nodes) - source - 1)
    lb = FUNCTION(source)
    func_diff = random.uniform(-lb, ub)
    if func_diff < 0:
      diff = -math.ceil(INVERSE(-func_diff))
      if diff < -source:
        raise Exception(
            "Diff is "
            + str(diff)
            + " but it should be no smaller than "
            + str(-source)
        )
    else:
      diff = math.ceil(INVERSE(func_diff))
      if diff > len(all_nodes) - source - 1:
        raise Exceptions(
            "Diff is "
            + str(diff)
            + " but it should be no greater than "
            + str(len(all_nodes) - source - 1)
        )
    target = source + diff

    source_lat = all_nodes["lat"][source]
    source_lng = all_nodes["lng"][source]
    target_lat = all_nodes["lat"][target]
    target_lng = all_nodes["lng"][target]
    d = distance((source_lat, source_lng), (target_lat, target_lng)).miles
    if d > MILE_LOWER_BOUND and d < MILE_UPPER_BOUND:
      S.append((source, target, d))

    if (len(S) % TILES) == 0:
      print(
          "Generating query " + str(len(S) + 1) + "/" + str(n_queries) + "..."
      )

  miles_values = sorted([m for (_, _, m) in S])

  for i in range(TILES + 1):
    index = max(min((i * len(S)) // TILES, len(S) - 1), 0)
    print(str(i) + "/" + str(TILES) + " are below " + str(miles_values[index]))

  with open(
      OSM_DIR + "/" + OUTPUT_FILE_PREFIX + str(n_queries) + ".tsv", "w"
  ) as f:
    print(
        "source_id\ttarget_id\tsource_lat\tsource_lng\ttarget_lat\ttarget_lng\tmiles",
        file=f,
    )
    for source, target, miles in S:
      print(
          str(source)
          + "\t"
          + str(target)
          + "\t"
          + str(all_nodes["lat"][source])
          + "\t"
          + str(all_nodes["lng"][source])
          + "\t"
          + str(all_nodes["lat"][target])
          + "\t"
          + str(all_nodes["lng"][target])
          + "\t"
          + str(miles),
          file=f,
      )


if __name__ == "__main__":
  num_queries = int(sys.argv[1])
  select_random_queries(num_queries)
