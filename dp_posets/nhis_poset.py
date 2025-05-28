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

"""Functions to generate the order matrix and ground set for k-way marginals."""

import numpy as np
from dp_posets import sensitivity_space_sampler
from dp_posets import utils


def get_nhis_ground_set_and_order(
    num_sections,
):
  """Returns (ground_set, order) for NHIS data poset.

  Args:
    num_sections: int representing the number of NHIS survey sections to
    include (between 1 and 3).
  """
  ground_set = list(range(15))
  order = np.asarray([
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
  ])
  if num_sections == 1:
    ground_set = ground_set[:4]
    order = order[:4, :4]
  elif num_sections == 2:
    ground_set = ground_set[:11]
    order = order[:11, :11]
  return ground_set, order


def compute_expected_norm_comparison(
    num_samples, num_sections
):
  """Compares poset_ball vs. unit l_inf ball expected squared l_{2} norm.

  Args:
    num_samples: int representing the number of samples from each ball.
    num_sections: int representing the number of NHIS survey sections to
    include (between 1 and 3).

  Returns:
    Ratio of the average squared l_{2} norm of the poset ball noise to the
    average squared l_{2} norm of the l_inf noise, from `num_samples` samples.
  """
  ground_set, order = get_nhis_ground_set_and_order(num_sections)
  poset_samples = []
  for _ in range(num_samples):
    poset_sample = sensitivity_space_sampler.sample_poset_ball(
        ground_set=ground_set, order=order
    )
    # Remove the root dimension from the poset sample
    poset_samples.append(poset_sample[1:])
  poset_expected_norm = utils.compute_average_squared_l2_norm(poset_samples)
  linf_expected_norm = utils.compute_linf_average_squared_l2_norm(
      len(ground_set)
  )
  return poset_expected_norm / linf_expected_norm
