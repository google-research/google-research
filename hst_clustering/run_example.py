# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Run clustering code for paper.

Clustering code of "Scalable Differentially Private Clustering via
Hierarchically Separated Trees" KDD'22 https://arxiv.org/abs/2206.08646
"""

from collections.abc import Sequence

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from hst_clustering import io
# pylint: disable=g-bad-import-order
from hst_clustering import dynamic_program

MakeDirs = os.makedirs
ReadRawData = io.ReadRawData
LoadFilesIntoDataFrame = io.LoadFilesIntoDataFrame


# This file encodes an HST as follows:
# 1st column corresponds to a node id string
# 2nd column corresopnds to the left child node id
# 3rd column is the right child node id
# The remaining columns are doubles representing the centers of the node.
HST_DATA = flags.DEFINE_string(
    "hst_data", "", "File path where HST is stored as a CSV file"
)

# Tab separated file with the first column indicating the example number
# All other columns represent the coordinates of a point in R^d.
RAW_DATA = flags.DEFINE_string(
    "raw_data", "", "File path where raw data is stored"
)
OUTPUT_DIR = flags.DEFINE_string("output_dir", "", "Output directory")

K_PARAMS = flags.DEFINE_integer("k_params", 10, "Number of centers")


LLOYD_ITERS = 0


def main(argv):
  del argv
  raw_data = ReadRawData(RAW_DATA.value)
  data_frame, feature_columns = LoadFilesIntoDataFrame(HST_DATA.value)
  tree = io.DataFrameToTree(data_frame)
  tree.validate_tree(True)
  k = K_PARAMS.value
  tree.solve_dp(k)
  logging.info("Finished solving the tree for k=%d", k)
  score, centers = dynamic_program.eval_hst_solution(
      raw_data, data_frame, tree, feature_columns, LLOYD_ITERS
  )
  logging.info("Score %.3f", score)

  base_dir = OUTPUT_DIR.value
  MakeDirs(base_dir)
  results_output = base_dir + "/results.txt"
  centers_output = base_dir + "/centers.npy"
  with open(results_output, "w") as f:
    f.write(str(score))
  with open(centers_output, "wb") as f:
    np.save(f, centers)


if __name__ == "__main__":
  app.run(main)
