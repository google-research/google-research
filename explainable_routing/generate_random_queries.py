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

OSM_DIR = "*-PROCESSED DIRECTORY NAME/washington-processed"
# OSM_DIR = "*-PROCESSED DIRECTORY NAME"
NODES_FILE = "nodes.tsv"
OUTPUT_FILE_PREFIX = "seattle_medium_queries_random_"

TILES = 100
MILE_LOWER_BOUND = 5
MILE_UPPER_BOUND = 30
BOX_SOUTHWEST_CORNER = 47.45514245253568, -122.40503872251753
BOX_NORTHEAST_CORNER = 47.79667044267817, -122.10772121948678


def in_box(lat, lng):
  if BOX_SOUTHWEST_CORNER is None or BOX_NORTHEAST_CORNER is None:
    return True
  return (
      lat >= BOX_SOUTHWEST_CORNER[0]
      and lat <= BOX_NORTHEAST_CORNER[0]
      and lng >= BOX_SOUTHWEST_CORNER[1]
      and lng <= BOX_NORTHEAST_CORNER[1]
  )


def select_random_queries(n_queries):
  all_nodes = pd.read_csv(OSM_DIR + "/" + NODES_FILE, delimiter="\t")
  S = []
  while len(S) < n_queries:
    source = random.randrange(len(all_nodes))
    target = random.randrange(len(all_nodes))

    source_lat = all_nodes["lat"][source]
    source_lng = all_nodes["lng"][source]
    target_lat = all_nodes["lat"][target]
    target_lng = all_nodes["lng"][target]
    d = distance((source_lat, source_lng), (target_lat, target_lng)).miles
    if (
        d > MILE_LOWER_BOUND
        and d < MILE_UPPER_BOUND
        and in_box(source_lat, source_lng)
        and in_box(target_lat, target_lng)
    ):
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
