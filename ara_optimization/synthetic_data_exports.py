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

"""Use export_synthetic_raw to generate synthetic Ads datasets for the paper."""

import argparse

import numpy as np

from ara_optimization import synthetic_dataset


def export_datasets(seed, skip_existing=True):
  """Generate 3 synthetic datasets with corresponding training and testing sets."""

  # Set the fixed random seed for reproducible results.
  np.random.seed(seed)
  synthetic_dataset.export_synthetic_raw(
      label='criteo-train',
      rate_of_size_1=0.50,
      rate_of_size_2=0.0675,
      average_conversion_per_impression=10,
      impression_side_dimensions=[20, 10, 8, 2],
      conversion_side_dimensions=[1],
      value_mean=130.05,
      value_mode=17.22,
      skip_existing=skip_existing)

  synthetic_dataset.export_synthetic_raw(
      label='criteo-test',
      rate_of_size_1=0.50,
      rate_of_size_2=0.0675,
      average_conversion_per_impression=10,
      impression_side_dimensions=[20, 10, 8, 2],
      conversion_side_dimensions=[1],
      value_mean=130.05,
      value_mode=17.22,
      skip_existing=skip_existing)

  # adtech re  mu=0.875 s=0.434
  synthetic_dataset.export_synthetic_raw(
      label='adtech_real_estate-train',
      rate_of_size_1=0.51,
      rate_of_size_2=0.25,
      average_conversion_per_impression=10,
      impression_side_dimensions=[2, 90],
      conversion_side_dimensions=[1],
      value_mean=2.6357769599728544,
      value_mode=1.9870358387102465,
      rescale_value=lambda x: 5000 * x,
      skip_existing=skip_existing)

  synthetic_dataset.export_synthetic_raw(
      label='adtech_real_estate-test',
      rate_of_size_1=0.51,
      rate_of_size_2=0.25,
      average_conversion_per_impression=10,
      impression_side_dimensions=[2, 90],
      conversion_side_dimensions=[1],
      value_mean=2.6357769599728544,
      value_mode=1.9870358387102465,
      rescale_value=lambda x: 5000 * x,
      skip_existing=skip_existing)

  # adtech travel mu=1.950 s=1.144
  synthetic_dataset.export_synthetic_raw(
      label='adtech_travel-train',
      rate_of_size_1=0.44045663923,
      rate_of_size_2=0.20,
      average_conversion_per_impression=10,
      impression_side_dimensions=[90, 2],
      conversion_side_dimensions=[1],
      value_mean=13.522676270135175,
      value_mode=1.8988795467748725,
      rescale_value=lambda x: 5000 * x,
      skip_existing=skip_existing)

  synthetic_dataset.export_synthetic_raw(
      label='adtech_travel-test',
      rate_of_size_1=0.44045663923,
      rate_of_size_2=0.20,
      average_conversion_per_impression=10,
      impression_side_dimensions=[90, 2],
      conversion_side_dimensions=[1],
      value_mean=13.522676270135175,
      value_mode=1.8988795467748725,
      rescale_value=lambda x: 5000 * x,
      skip_existing=skip_existing,
  )


if __name__ == '__main__':
  # Parse the command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  export_datasets(seed=args.seed, skip_existing=False)
