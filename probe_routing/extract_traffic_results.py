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

from collections import defaultdict
from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
import numpy as np


def parse_info(fname):
  with open(fname, "r") as file:
    lines = file.readlines()

  threshold_info = defaultdict(dict)
  current_threshold = None
  max_real_clusters_fraction = -1.0

  for line in lines:
    if "Results for threshold" in line:
      if current_threshold is not None:
        threshold_info[current_threshold][
            "max_real_clusters_fraction"
        ] = max_real_clusters_fraction
        max_real_clusters_fraction = -1.0
      current_threshold = float(
          line.split("Results for threshold")[1].replace(":", "")
      )
    if "real_clusters_fraction" in line:
      current_real_clusters_fraction = float(line.split(":")[-1])
      max_real_clusters_fraction = max(
          max_real_clusters_fraction, current_real_clusters_fraction
      )
    if "real_graph_ratio_90" in line:
      threshold_info[current_threshold]["real_graph_ratio_90"] = float(
          line.split(":")[-1]
      )
  threshold_info[current_threshold][
      "max_real_clusters_fraction"
  ] = max_real_clusters_fraction
  return threshold_info


def broken_axis_rcf_plot(x, y):
  for i in range(len(x)):
    if x[i] == 0:
      x[i] = 10 ** (-9)
  x = np.log10(x)

  # Create broken axes
  fig = plt.figure(figsize=(6, 4))
  bax = brokenaxes(xlims=((-9.5, -8.5), (-7.5, 0.5)))

  # Plot data
  bax.scatter(x, y, marker="o")

  # Labeling and title
  bax.set_xlabel(
      "Maximum fraction of clusters queried in full graph", labelpad=25
  )
  bax.set_ylabel("90th percentile approximation ratio", labelpad=35)
  # bax.set_title('Plot with Broken Axes')
  bax.axs[0].set_xticks([-9])
  bax.axs[0].set_xticklabels([0])

  bax.axs[1].set_xticks([-7, -6, -5, -4, -3, -2, -1, 0])
  bax.axs[1].set_xticklabels(
      ["1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", 0.1, 1]
  )
  bax.set_ylim([0.99, 1.11])

  plt.show()


def make_threshold_plot(fname):
  threshold_info = parse_info(fname)
  x, y = [], []
  for threshold, values in threshold_info.items():
    x.append(threshold)
    y.append(values["real_graph_ratio_90"])
  plt.xlabel("Threshold scalar")
  plt.ylabel("90th percentile path approx ratio")
  plt.scatter(x, y)
  print(list(zip(x, y)))
  plt.show()


def make_max_rcf_plot(fname):
  threshold_info = parse_info(fname)
  x, y = [], []
  for threshold, values in threshold_info.items():
    x.append(values["max_real_clusters_fraction"])
    y.append(values["real_graph_ratio_90"])
  print(list(zip(x, y)))
  broken_axis_rcf_plot(x, y)
