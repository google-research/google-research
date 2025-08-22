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

"""Quantitative analysis functions for experimental setup stability.

We assume a pandas dataframe as input containing the following columns:

Evaluation Measure: identifier of a specific evaluation measure.
Target Dataset: identifier of a specific target dataset.
Sources ID: identifier of a specific source pool.
Evaluation Measure Score: evaluation measure score of a transferability metric.
Transferability Metric: identifier of the transferability metric.

More generally we expect 3 experiment components (CompA, CompB, CompC)
in our case: (Evaluation Measure, Target Dataset, Sources ID)

Expected DataFrame structure:

|Transf. Metric| CompA | CompB | CompC |Eval. Measure Score|
------------------------------------------------------------
|     gbc      |   x   |   y   |   z   |        0.5        |
|     leep     |   x   |   y   |   z   |        0.2        |
|     logme    |   x   |   y   |   z   |        0.8        |
|     gbc      |   x   |   y   |   k   |        0.5        |
|     leep     |   x   |   y   |   k   |        0.8        |
|     logme    |   x   |   y   |   k   |        0.2        |
...

"""

import itertools

import numpy as np
from scipy import stats


def get_graph_stability_score(common_subset, varying):
  """Measure the ranking agreement across connected experiments."""
  outcomes = [outcome[1]['Evaluation Measure Score'].values
              for outcome in common_subset.groupby(varying)]
  connected_experiments = list(itertools.combinations(outcomes, 2))
  stability_score = np.nanmean([
      stats.kendalltau(outcome_pair[0], outcome_pair[1])[0]
      for outcome_pair in connected_experiments])
  return stability_score


def get_setup_stability_score(
    experiments_df,
    varying='Target Dataset',
    fixing=('Evaluation Measure', 'Sources ID')):
  """Get the setup stability score to the variation of a setup component."""
  ss_scores = []
  common_subsets = [cs[1] for cs in experiments_df.groupby(list(fixing))]
  for common_subset in common_subsets:
    ss_scores.append(get_graph_stability_score(common_subset, varying))
  return np.nanmean(ss_scores)
