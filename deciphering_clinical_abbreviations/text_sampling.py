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

"""Helper functions for sampling text to create evaluation datasets."""

import collections
from typing import Optional
import pandas as pd


def sample_n_instances_per_contained_value(
    examples,
    contained_values_col_name,
    n_per_value,
    seed = None
):
  """Generates a sample containing at least n instances of each contained value.

  Examples are shuffled and then added iteratively to the sample if at least one
  contained value in the example has not already been sampled at least
  n_per_value times.

  Args:
    examples: a pd.DataFrame of examples to sample from. Must contain
      contained_values_col_name.
    contained_values_col_name: name of the column in examples indicating the
      values contained in each example. Each entry in the column should be an
      iterable over the present values.
    n_per_value: minimum number of instances of each value expected in the
      sample.
    seed: Optional seed for RNG determinism.

  Returns:
    A 2-tuple containing:
      - The sampled examples, as a pd.DataFrame
      - The total value instance counts for the sample, as a
        collections.Counter
  """
  value_counts = collections.Counter()

  sampled_snippets = []
  for example in examples.sample(
      frac=1, random_state=seed).itertuples(index=False):
    # itertuples() is generally much faster than iterrows(), but requires using
    # getattr() to access the values for the example.
    example_values = getattr(example, contained_values_col_name)
    if not example_values:
      continue
    min_count = min([value_counts[a] for a in example_values])
    if min_count < n_per_value:
      sampled_snippets.append(example)
      for value in example_values:
        value_counts[value] += 1

  return pd.DataFrame(sampled_snippets), value_counts
