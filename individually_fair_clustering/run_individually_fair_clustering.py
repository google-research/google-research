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

"""Main file for the fair clustering algorithm."""

import datetime
import json
import random
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np

from individually_fair_clustering import fair_clustering_algorithms
from individually_fair_clustering import fair_clustering_utils

_INPUT = flags.DEFINE_string(
    "input",
    "",
    "Path to the input file. The input file must contain one line per point,"
    " with each line being a tab separated vector represented as the entries,"
    " each entry as a floating point number.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    "",
    "Path to the output file. The output file is a json encoded in text format"
    " containing the results of one run of the algorithm.",
)
_K = flags.DEFINE_integer("k", 5, "Number of centers for the clustering.")

# The following parameters affect how the fairness constraints are defined
# (for all algorithms).
# In order to allow the efficient definition of a the fairness distance bound
# for each point we do not do all pairs distance computations.
# Instead we use a sample of points. For each point p we define the threshold
# d(p) of the maximum distance that is allowed for a center near p to be the
# distance of the sample_rank_for_threshold-th point closest to be among
# sample_size_for_threshold sampled points times multiplier_for_threshold.
_SAMPLE_SIZE_FOR_THRESHOLD = flags.DEFINE_integer(
    "sample_size_for_threshold",
    500,
    "Size for the sample used to determine the fairness threshold.",
)
_SAMPLE_RANK_FOR_THRESHOLD = flags.DEFINE_integer(
    "sample_rank_for_threshold", 50, "Rank of the distance for the threshold."
)
_MULTIPLIER_FOR_THRESHOLD = flags.DEFINE_float(
    "multiplier_for_threshold", 1.0, "Multiplier for the distance threshold."
)

# The next parameters affect the runs of the slower algorithms used as
# baselines. For all algorithms except LSPP, Greedy, VanillaKMeans,
# if the size of the dataset is larger than large_scale_dataset_size, we sample
# sample_size_for_slow_algorithms points and run the algorithm on them to find
# a solution. Then, of course, this solution is evaluated on the whole dataset.
_SAMPLE_SIZE_FOR_SLOW_ALGORITHMS = flags.DEFINE_integer(
    "sample_size_for_slow_algorithms",
    4000,
    "Number of elements used in the input for the slow algorithms",
)
_LARGE_SCALE_DATASET_SIZE = flags.DEFINE_integer(
    "large_scale_dataset_size",
    10000,
    "Number of elements of a dataset that require using sampling for slow "
    "algorithms",
)

_ALGORITHM = flags.DEFINE_string(
    "algorithm",
    "LSPP",
    "name of the algorithm among: LSPP, IMCL20, Greedy, VanillaKMeans",
)


def main(argv):
  del argv
  assert _INPUT.value
  assert _OUTPUT.value
  assert _K.value > 0

  dataset = fair_clustering_utils.ReadData(_INPUT.value)

  logging.info("Computing the thresholds")
  dist_threshold_vec = fair_clustering_utils.ComputeDistanceThreshold(
      dataset,
      _SAMPLE_SIZE_FOR_THRESHOLD.value,
      _SAMPLE_RANK_FOR_THRESHOLD.value,
      _MULTIPLIER_FOR_THRESHOLD.value,
  )
  logging.info("[Done] computing the thresholds")

  # since ICML20 is not scalable in case we use a large scale dataset
  # the algorithm is restricted to using subset of the elements.
  # But the evaluation is done in the whole dataset.
  slow_algos = set(["ICML20"])

  if (_ALGORITHM.value in slow_algos) and dataset.shape[
      0
  ] >= _LARGE_SCALE_DATASET_SIZE.value:
    logging.info("Using a sample as the algorithm is not scalable")
    input_positions = random.sample(
        list(range(dataset.shape[0])), _SAMPLE_SIZE_FOR_SLOW_ALGORITHMS.value
    )
    input_positions.sort()
    dataset_input = np.array([dataset[i] for i in input_positions])
    dist_threshold_vec_input = np.array(
        [dist_threshold_vec[i] for i in input_positions]
    )
  else:  # Using the full dataset and threshold
    logging.info("Using the full dataset")
    dataset_input = dataset
    dist_threshold_vec_input = dist_threshold_vec

  logging.info(
      "Algorithm starts, running on dataset of size %d", dataset_input.shape[0]
  )
  start = datetime.datetime.now()
  if _ALGORITHM.value == "LSPP":
    centers = fair_clustering_algorithms.LocalSearchPlusPlus(
        dataset=dataset_input,
        k=_K.value,
        dist_threshold_vec=dist_threshold_vec_input,
        coeff_anchor=3.0,
        coeff_search=1.0,
        number_of_iterations=5000,
        use_lloyd=True,
    )
  elif _ALGORITHM.value == "ICML20":
    centers = fair_clustering_algorithms.LocalSearchICML2020(
        dataset=dataset_input,
        k=_K.value,
        dist_threshold_vec=dist_threshold_vec_input,
        coeff_anchor=3.0,
        coeff_search=1.0,
        epsilon=0.01,
        use_lloyd=False,
    )
  elif _ALGORITHM.value == "Greedy":
    centers = fair_clustering_algorithms.Greedy(
        dataset=dataset_input,
        k=_K.value,
        dist_threshold_vec=dist_threshold_vec_input,
        coeff_anchor=3.0,
    )
  elif _ALGORITHM.value == "VanillaKMeans":
    centers = fair_clustering_algorithms.VanillaKMeans(
        dataset=dataset_input, k=_K.value
    )
  else:
    raise RuntimeError("Algorithm not supported")
  end = datetime.datetime.now()
  duration = end - start
  logging.info("Algorithm completes.")

  # notice that in any case the evaluation is done on the whole dataset.
  results = {
      "k": _K.value,
      "time": duration.total_seconds(),
      "k-means-cost": fair_clustering_utils.KMeansCost(dataset, centers),
      "max-bound-ratio": fair_clustering_utils.MaxFairnessCost(
          dataset, centers, dist_threshold_vec
      ),
      "algorithm": _ALGORITHM.value,
  }
  logging.info(results)

  with open(_OUTPUT.value, "w") as outfile:
    json.dump(results, outfile)


if __name__ == "__main__":
  app.run(main)
