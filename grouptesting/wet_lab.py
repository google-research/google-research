# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Simulates a wet lab."""

import csv
from typing import Union, Iterable

import gin
import jax
import numpy as np

from grouptesting import utils


@gin.configurable
class WetLab:
  """A simulator to produce patients conditions and tests.

  Attributes:
   num_patients: (int) the number of patients to be tested.
   diseased: Optional[np.ndarray<bool>[num_patients]]. If not none, we know the
    ground truth and store it here (for performance measures).
   freeze_diseased: (bool) if True, the diseased array will not be reset more
    than once.
  """

  def __init__(self,
               num_patients = 2,
               freeze_diseased = True,
               base_infection_rate = 0.05,
               specificity = 0.97,
               sensitivity = 0.85):
    """Initializes the WetLab.

    Args:
     num_patients: the number of patients to be considered.
     freeze_diseased: if True, the calling reset will not change the state
      of which patient is diseased and which is not. In a large scale
      simulations, consider using False in order to have different setups of
      diseased patients in each simulation.
     base_infection_rate: Can be either a single value for every one or a prior
      per patient as a list or array.
     specificity: the specificity of the PCR machine to be used. In case of an
      vector, it corresponds to the specificity for a given group size.
     sensitivity: the sensitivity of the PCR machine to be used. In case of an
      vector, it corresponds to the sensitivity for a given group size.
    """
    if num_patients < 2:
      raise ValueError('Number of patients should be larger than 2')

    self.num_patients = num_patients
    self.freeze_diseased = freeze_diseased
    self._base_infection_rate = np.atleast_1d(base_infection_rate)
    self._specificity = np.atleast_1d(specificity)
    self._sensitivity = np.atleast_1d(sensitivity)
    self.diseased = None

  def reset(self, rng):
    if not self.freeze_diseased or self.diseased is None:
      random_matrix = jax.random.uniform(rng, (self.num_patients,))
      self.diseased = random_matrix < self._base_infection_rate

  def group_tests_outputs(self, rng, groups):
    """Produces test outputs taking into account test errors."""
    n_groups = groups.shape[0]
    group_disease_indicator = np.dot(groups, self.diseased) > 0
    group_sizes = np.sum(groups, axis=1)
    specificity = utils.select_from_sizes(self._specificity, group_sizes)
    sensitivity = utils.select_from_sizes(self._sensitivity, group_sizes)
    draw_u = jax.random.uniform(rng, shape=(n_groups,))
    delta = sensitivity - specificity
    not_flip_proba = group_disease_indicator * delta + specificity
    test_flipped = draw_u > not_flip_proba
    return np.logical_xor(group_disease_indicator, test_flipped)


@gin.configurable
class WetLabCSV(WetLab):
  """Reads and write csv file with lab tests."""

  def __init__(self, filename = None):
    super().__init__()
    self.filename = filename
    self.group_results = []
    self.prior = []
    self.columns = []

  def reset(self, rng):
    del rng
    with open(self.filename) as fp:
      reader = csv.DictReader(fp)
      results_col = reader.fieldnames[-1]
      patients_cols = reader.fieldnames[1:-1]
      self.num_patients = len(patients_cols)
      self.columns = reader.fieldnames
      for i, row in enumerate(reader):
        if i == 0:
          self.prior = [row.get(p) for p in patients_cols]
        elif i > 1:
          self.group_results.append(row.get(results_col))

    self.prior = np.atleast_1d(self.prior, np.float)
    self.group_results = np.array(self.group_results, np.bool)

  def group_tests_outputs(self, rng, groups):
    del groups, rng
    return self.group_results

  def to_csv(self, groups, marginals):
    """Returns a csv containing the new groups to be tested."""
    output = self.filename + '.output.csv'

    def to_list(arr):
      result = ['']
      result.extend(list(np.array(arr).astype(str)))
      result.append('')
      return result

    with open(output, 'w') as fp:
      writer = csv.writer(fp)
      writer.writerow(self.columns)
      writer.writerow(to_list(self.prior))
      writer.writerow(to_list(marginals))
      for group in groups:
        writer.writerow(to_list(group))
