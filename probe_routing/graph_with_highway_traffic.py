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

import os
import random
import shutil
import pandas as pd
import s2cell
from tqdm import tqdm

OSM_DIR = "*-PROCESSED DIRECTORY NAME"
ARCS_FILE = "arcs.tsv"
NODES_FILE = "nodes.tsv"
CLUSTERS_FILE = "clusters.tsv"
TRAFFIC_SUBDIR_PREFIX = "t1_resampled_highway_traffic_"
MAX_TRAFFIC_DISTORTION = 2.0
S2_LEVEL = 8  # 27-38 km
MAX_PRIORITY = 1  # highways are 0, highway exits are 1


def get_traffic_subdir():
  return OSM_DIR + "/" + TRAFFIC_SUBDIR_PREFIX + str(MAX_TRAFFIC_DISTORTION)


def node_s2cells():
  nodes = pd.read_csv(OSM_DIR + "/" + NODES_FILE, delimiter="\t")
  node_cells = {}
  print("Computing node S2 cells...")
  for i in tqdm(nodes.index):
    node_cells[i] = s2cell.lat_lon_to_cell_id(
        nodes.at[i, "lat"], nodes.at[i, "lng"], S2_LEVEL
    )
  return node_cells


def get_clusters(arcs):
  node_cells = node_s2cells()
  cluster_id_options = {}
  cluster = pd.DataFrame(index=arcs.index.copy(), columns=["cluster_id"])
  print("Clustering highway arcs with the same S2 cell...")
  for i in tqdm(arcs.index):
    clusterable_priority = arcs.at[i, "road_type"] <= MAX_PRIORITY
    if clusterable_priority:
      src = arcs.at[i, "src"]
      if node_cells[src] not in cluster_id_options:
        cluster_id_options[node_cells[src]] = i
      cluster.at[i, "cluster_id"] = cluster_id_options[node_cells[src]]
    else:
      cluster.at[i, "cluster_id"] = i
  print("Num highway clusters: ", len(cluster_id_options))

  return cluster


def make_traffic():
  orig_arcs = pd.read_csv(OSM_DIR + "/" + ARCS_FILE, delimiter="\t")
  traffic_subdir = get_traffic_subdir()
  if os.path.exists(traffic_subdir):
    raise FileExistsError("Directory " + traffic_subdir + " already exists")
  os.makedirs(
      traffic_subdir
  )  # should throw error if already exists, as desired
  shutil.copy(OSM_DIR + "/" + NODES_FILE, traffic_subdir)

  cluster = get_clusters(orig_arcs)
  cluster.to_csv(traffic_subdir + "/" + CLUSTERS_FILE, sep="\t", index=False)

  noise = {}
  new_arcs = orig_arcs.copy()
  print("Num arcs: ", len(orig_arcs))
  print("Computing and writing traffic weights...")
  for i in tqdm(new_arcs.index):
    if cluster.at[i, "cluster_id"] not in noise:
      noise[cluster.at[i, "cluster_id"]] = random.uniform(
          1.0, MAX_TRAFFIC_DISTORTION
      )
    new_arcs.at[i, "duration_secs"] *= noise[cluster.at[i, "cluster_id"]]
  new_arcs.to_csv(traffic_subdir + "/" + ARCS_FILE, sep="\t", index=False)
  print("Num noises: ", len(noise))


make_traffic()
