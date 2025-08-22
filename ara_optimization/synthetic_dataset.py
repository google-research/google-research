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

"""Defines export_synthetic_raw to generate synthetic Ads datasets."""

import csv
import itertools
import os
from typing import Callable, Union

import numpy as np


def _sample_discrete_powerlaw_approximately_d6(
    a, r, x_min = 1
):
  """Sample from a discrete power-law distribution.

  Implements approximate discrete power law per (D.6) in "Power-law
  distributions in empirical data" in https://arxiv.org/pdf/0706.1062.pdf
  A more accurate but computationally expensive method is also provided in the
  same paper.

  Args:
    a: Power-law exponent. The same as shape b param. The notation 'a' is used
      here to align with the paper's notations.
    r: A uniform random sample to transfrom into power-law one.
    x_min: Minimum value of the distribution.

  Returns:
    An array of samples from the discrete power-law distribution.
  """
  return np.floor((x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) + 0.5).astype(int)


def sample_discrete_power_law(
    b, x_min, x_max, n_samples
):
  """Generate samples from a discrete power-law distribution.

  Generates n_samples many samples in [x_min, x_max] range from a discrete
  power-law distribution, per the method outlined in (D.6) of the paper [1].

  Reference:
    [1] Clauset, Aaron, Cosma Rohilla Shalizi, and Mark EJ Newman. "Power-law
    distributions in empirical data." SIAM review 51.4 (2009): 661-703.

  Args:
    b: Power-law shape parameter.
    x_min: Minimum value of the distribution.
    x_max: Maximum value of the distribution.
    n_samples: Number of samples to generate.

  Returns:
    Array of samples from the target distribution.
  """
  if b <= 1:
    raise ValueError("Error: b must be greater than 1.")

  if x_min <= 0:
    raise ValueError("Error: x_min must be positive.")
  if x_max <= x_min:
    raise ValueError("Error: x_max must be larger than x_min.")

  # Goal is to generates in [x_min, x_max] range from a discrete
  # power-law distribution, per the method outlined in (D.6) of the paper [1].
  # However, that method only supports x_min and not x_max. To incorporate x_max
  # support, we utilize the following equation:
  # `floor( (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) + 0.5) <= x_max`
  # To solve for 'r' in this equation, follow these steps:
  # 1. Move the floor function to the right side of the inequality:
  # ```
  # (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) + 0.5 < floor(x_max +1)
  # ```
  # 2. Isolate the exponential term by subtracting 0.5 from both sides:
  #  ```
  #  (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) < floor(x_max +1) - 0.5
  #  ```
  # 3. Simplify the inequality by letting 'y = floor(x_max + 1) - 0.5':
  #  ```
  #  (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) < y
  #  ```
  # 4. Raise both sides of the inequality to the power of '-a + 1'.
  # Since '-a + 1' is negative (a is >1) inequality's direction changes:
  #  ```
  #  (1 - r) > (y / (x_min - 0.5)) ** (-a + 1)
  #  ```
  # 5. Since both sides are positive, we can take the reciprocal without
  # changing the inequality's direction:
  #  ```
  #  r < 1 - (y / (x_min - 0.5)) ** (-a + 1)
  #  ```

  # 6. Therefore, the solution for 'r' is:
  #  ```
  #  r < 1 - ( (floor(x_max +1) - 0.5) / (x_min - 0.5)) ** (-a + 1)
  #  ```

  # Reference:
  # [1] Clauset, Aaron, Cosma Rohilla Shalizi, and Mark EJ Newman. "Power-law
  # distributions in empirical data." SIAM review 51.4 (2009): 661-703.

  # Generate n_samples many samples from uniform distribution.
  r_max = 1 - ((np.floor(x_max+1) - 0.5) / (x_min - 0.5)) ** (-b + 1)
  r = np.random.uniform(0, r_max, n_samples)
  # Transform to discrete power law distribution.
  dpl = _sample_discrete_powerlaw_approximately_d6(b, r)
  return dpl


def _generate_slice_distribution_for_impressions(
    param_b, number_of_slices):
  """Generates a slice distribution for impressions.

  This function samples slice-level impression counts from a discrete power-law
  distribution and encodes the sample into a dictionary mapping slice sizes to
  their corresponding frequencies.

  Args:
    param_b: The power-law shape parameter.
    number_of_slices: The total number of slices to consider.
  Returns:
    a dictionary of slice-size:frequancy.

  **Interpretation of Returned Keys and Values:**
  The keys in the returned dictionary represent the slice sizes, e.g., key 2
  represent slices with exactly 2 impressions. The values in the dictionary
  represent the number of slices with that slice size. For instance, if the
  dictionary contains the key-value pair `{2: 100}`, this indicates that there
  are 100 slices that have exactly 2 impressions.
  """

  result_dist = {}
  num_impressions = sample_discrete_power_law(
      b=-param_b,  # paper use x^-b convension.
      x_min=1,
      x_max=number_of_slices,
      n_samples=number_of_slices,
  )

  # encode results into a dictionary
  unique_elements, counts = np.unique(num_impressions, return_counts=True)
  for element, count in zip(unique_elements, counts):
    result_dist[element + 1] = count
  return result_dist


def _get_b_param_from_slice_size_rates(
    rate_of_size_1, rate_of_size_2
):
  """Computes the power-law shape parameter 'b'.

  Consider the power-law function `power-law(x) = ax^b` where 'a' is a constant.
  The ratio of `power-law(2)` to `power-law(1)` can be expressed as:
    power-law(2)/power-law(1) = (a2^b) / (a1^b) = 2^b
  This implies that the exponent 'b' can be obtained by taking the logarithm of
  the ratio of power-law(2) and power-law(1):
    b = log2(power-law(2)/power-law(1))

  While the parameter 'b' could be directly provided by the user, this function
  utilizes the input parameters rate_of_size_1 and rate_of_size_2 to align with
  the experience in NoiseLab: https://developer.chrome.com/docs/privacy-sandbox/summary-reports/design-decisions/

  Args:
    rate_of_size_1: Rate of slices with one conversion.
    rate_of_size_2: Rate of slices with two conversions.

  Returns:
    The power-law shape parameter 'b'.
  """  # pylint: disable=line-too-long
  return np.log2(rate_of_size_2 / rate_of_size_1)


def get_conversion_distribution_uniformly(
    num_slices, total_conversions
):
  """Distributes conversions among slices uniformly.

  Args:
    num_slices: number of slices that conversions can be distributed.
    total_conversions: number of conversions to distribute.

  Returns:
    A NumPy array representing the number of conversions at each slice, where
      each element in the array corresponds to the number of conversions for the
      corresponding slice index.
  """

  conv_dist = np.zeros(num_slices, dtype=int)
  target = np.random.randint(0, num_slices, total_conversions)
  for i in target:
    conv_dist[i] += 1
  return conv_dist


def get_mu_sigma(value_mean, value_mode):
  """Calculates the mean (mu) and standard deviation (sigma) of a lognormal.

  **Description of the lognormal distribution PDF:**
  The probability density function of the lognormal distribution is given by::

    ```
    f(x) = (1 / (sigma * x * np.sqrt(2 * np.pi))) *
            np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))
    ```
  where:
    x is the random variable
    mu is the mean of the distribution
    sigma is the standard deviation of the distribution

  **Formulas for mean and mode in terms of mu and sigma:**

  The mean and mode of the lognormal distribution can be expressed in
  terms of the mu (μ) and sigma (σ) as:
  ```
  mode = exp(μ - σ^2)
  mean = exp(μ + σ^2 / 2)
  ```
  Thus mu (μ) and sigma (σ) can be expressed as:
  ```
  μ = (2*log(mean) + log(mode))/3
  σ = sqrt(μ - log(mode))
  ```
  Arguments:
    value_mean: The mean of the distribution.
    value_mode: The mode of the distribution.

  Returns:
    A tuple containing mu (μ) and sigma (σ) of the lognormal distribution.
  """
  value_mu = (2 * np.log(value_mean) + np.log(value_mode)) / 3
  value_sigma = np.sqrt(value_mu - np.log(value_mode))
  return value_mu, value_sigma


def generate_slice_distribution_with_conversions_raw(
    impression_dist,
    average_conversion_per_impression,
    impression_side_dimensions,
    conversion_side_dimensions,
    value_mean,
    value_mode,
):
  """Computes event-level attributed keys.

  Example:
  * Let an impression key [1, 3, 2] have 20 impressions.
  * Let the conversion side have a value of 2.
  * Attribution keys become [1, 3, 2, 0] and [1, 3, 2, 1], which would share
    conversions from the 20 impressions.
  * The 20 impressions generate a Poisson(lambda, 20) number of conversions.
    Overall, there will be num_conv=sum(Poisson(lambda, 20)) conversions, e.g.,
    194 conversions.
  * Distributing uniformly using get_conversion_distribution_uniformly(
      num_slices=2, total_conversions=num_conv) -> [98, 96].
  * Thus [1, 3, 2, 0] gets 98 conversions, and [1, 3, 2, 1] gets 96 conversions.

  Args:
    impression_dist: Impression distributions encoded as a dictionary mapping
      slice size to number of impressions.
    average_conversion_per_impression: Average number of conversions per
      impression.
    impression_side_dimensions: List of impression side dimensions, e.g., [16,
      8, 2].
    conversion_side_dimensions: List of conversion side dimensions, e.g., [3,2].
    value_mean: Average value in float.
    value_mode: Most common value in float.

  Returns:
    An array of event records, where each event is an array of [id, attribution
    key,count, value].
  """

  conversion_key_cardinality = np.prod(conversion_side_dimensions)
  value_mu, value_sigma = get_mu_sigma(value_mean, value_mode)
  print(
      f"\tLog-Normal Distribution parameters mu (μ) = {value_mu} and sigma (σ)"
      f" = {value_sigma}"
  )

  global_imp_id = 0
  ranges = []
  for key in impression_side_dimensions:
    ranges.append(range(key))
  all_keys = itertools.product(*ranges)

  dataset = []
  # Each impression key would make that many attributed keys
  for key_imp_cnt, freq in impression_dist.items():
    # Process each key
    for _ in range(freq):
      imp_key = next(all_keys)

      # Step 2: Generate conversions for each impression.
      # Current impression key `imp_key` has `key_imp_cnt` impressions.
      cur_contributions = np.random.poisson(
          lam=average_conversion_per_impression, size=key_imp_cnt
      )

      for cur_slice_conversion_cnt in cur_contributions:
        global_imp_id += 1
        conv_dist = get_conversion_distribution_uniformly(
            num_slices=conversion_key_cardinality,
            total_conversions=cur_slice_conversion_cnt,
        )

        # In this part, conversions are distributed to the slices.
        # The conversion distribution [98, 96] indicates that one slice
        # receives 98 conversions, while the other one receives 96 conversions.
        conv_ranges = []
        for key in conversion_side_dimensions:
          conv_ranges.append(range(key))
        all_conv_keys = itertools.product(*conv_ranges)

        for slice_size in conv_dist:
          # attribution key `imp_key+conv_idx` has slice_size many conversions.
          conv_key = next(all_conv_keys)

          # Step 3, generate value for each conversion.
          conv_value = np.random.lognormal(
              mean=value_mu, sigma=value_sigma, size=slice_size
          )
          for conv_id in range(slice_size):
            dataset.append([
                f"imp-{global_imp_id}",
                imp_key + conv_key,
                1,
                conv_value[conv_id],
            ])
  return dataset  # pytype: disable=bad-return-type


def generate_counts_and_values_dataset_raw(
    rate_of_size_1,
    rate_of_size_2,
    average_conversion_per_impression,
    impression_side_dimensions,
    conversion_side_dimensions,
    value_mean,
    value_mode,
):
  """Generates synthetic dataset for counts and values.

  Args:
    rate_of_size_1: The rate of slices with 1 conversion compared to all slices
      with non-zero conversions.
    rate_of_size_2: The rate of slices with 2 conversions compared to all slices
      with non-zero conversions.
    average_conversion_per_impression: The average number of conversions per
      impression.
    impression_side_dimensions: A list of impression side dimensions, e.g., [16,
      8, 2].
    conversion_side_dimensions: A list of conversion side dimensions, e.g., [3,
      2].
    value_mean: The average value.
    value_mode: The most common value.

  Returns:
    A synthetic datasets as an array of event records, where each event is an
    array of [id:str, attribution key:str,count:int, value:float].
  """
  param_b = _get_b_param_from_slice_size_rates(rate_of_size_1, rate_of_size_2)
  print(f"\tPower-Law Distribution parameter b = {param_b}")
  print(
      "\tPoisson Distribution parameter         λ ="
      f" {average_conversion_per_impression}"
  )

  # Step 1: Generate impression side distribution.
  num_impression_keys = np.prod(impression_side_dimensions)
  impression_dist = _generate_slice_distribution_for_impressions(
      param_b, num_impression_keys
  )

  dataset = generate_slice_distribution_with_conversions_raw(
      impression_dist,
      average_conversion_per_impression,
      impression_side_dimensions,
      conversion_side_dimensions,
      value_mean,
      value_mode,
  )
  return dataset


def export_synthetic_raw(
    label,
    rate_of_size_1,
    rate_of_size_2,
    average_conversion_per_impression,
    impression_side_dimensions,
    conversion_side_dimensions,
    value_mean,
    value_mode,
    rescale_value = lambda x: x,
    skip_existing = False,
):
  """Generates synthetic data and exports it to a CSV file.

  The synthetic data is generated based on the specified parameters and stored
  in a CSV file named `synthetic-counts_and_values-{label}-raw.csv` under
  the `./synthetic-datasets` folder.

  Args:
    label: (str) A string that becomes part of the output file name.
    rate_of_size_1: The rate of slices with 1 conversion compared to all
      slices with non-zero conversions.
    rate_of_size_2: The rate of slices with 2 conversions compared to
      all slices with non-zero conversions.
    average_conversion_per_impression: The average number of conversions
      per impression.
    impression_side_dimensions: (list[int]) A list of impression side
      dimensions, e.g., [16, 8, 2].
    conversion_side_dimensions: (list[int]) A list of conversion side
      dimensions, e.g., [3, 2].
    value_mean: The average value.
    value_mode: The most common value.
    rescale_value: callable) A function to post-process the generated values.
      The default function is the identity function.
    skip_existing: (bool) Whether to skip generation if the target output file
      already exists.
  """

  export_dir = "./synthetic-datasets"
  export_url = f"{export_dir}/synthetic-counts_and_values-{label}-raw.csv"

  if os.path.exists(export_url) and skip_existing:
    print(
        (
            f"skip_existing set to {skip_existing} and file"
            f" {export_url} already exists. Skipping export."
        )
    )
    return

  if not os.path.exists(export_dir):
    os.makedirs(export_dir)

  print("\nGenerating synthetic data with:")
  print(f"\tImpression side dimensions = {impression_side_dimensions}")
  print(f"\tConversion side dimensions = {conversion_side_dimensions}")
  res_counts_and_values = generate_counts_and_values_dataset_raw(
      rate_of_size_1=rate_of_size_1,
      rate_of_size_2=rate_of_size_2,
      average_conversion_per_impression=average_conversion_per_impression,
      impression_side_dimensions=impression_side_dimensions,
      conversion_side_dimensions=conversion_side_dimensions,
      value_mean=value_mean,
      value_mode=value_mode,
  )

  print(f"Saving synthetic data to {export_url}")
  # Create a CSV file
  with open(export_url, "w") as csvfile:
    writer = csv.writer(csvfile)
    # Add a header row
    writer.writerow(["Impression_ID", "Attributed_Key", "Count", "Value"])
    for dataset_row in res_counts_and_values:
      writer.writerow([
          dataset_row[0],
          dataset_row[1],
          dataset_row[2],
          rescale_value(dataset_row[3]),
      ])
