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
from tqdm import tqdm

OSM_DIR = "*-PROCESSED DIRECTORY NAME"
ARCS_FILE = "arcs.tsv"
NODES_FILE = "nodes.tsv"
TRAFFIC_SUBDIR_PREFIX = "t0_traffic_"
MAX_TRAFFIC_DISTORTION = 128.0


def make_traffic():
  orig_arcs = pd.read_csv(OSM_DIR + "/" + ARCS_FILE, delimiter="\t")
  traffic_subdir = (
      OSM_DIR + "/" + TRAFFIC_SUBDIR_PREFIX + str(MAX_TRAFFIC_DISTORTION)
  )
  if os.path.exists(traffic_subdir):
    raise FileExistsError("Directory " + traffic_subdir + " already exists")
  os.makedirs(
      traffic_subdir
  )  # should throw error if already exists, as desired
  shutil.copy(OSM_DIR + "/" + NODES_FILE, traffic_subdir)
  new_arcs = orig_arcs.copy()
  print("Computing and writing traffic weights...")
  for i in tqdm(new_arcs.index):
    new_arcs.at[i, "duration_secs"] *= random.uniform(
        1.0, MAX_TRAFFIC_DISTORTION
    )
  new_arcs.to_csv(traffic_subdir + "/" + ARCS_FILE, sep="\t", index=False)


make_traffic()
