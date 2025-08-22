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

"""Code for pre-computing and saving many geometric masks.

This code was adapted from Zerveas et al.:
https://github.com/gzerveas/mvts_transformer/tree/master
"""

import sys
import numpy as np

num_masks, mask_rate, lm = sys.argv[1:]

num_masks = int(num_masks)
mask_rate = float(mask_rate)
lm = int(lm)


def geom_noise_mask_single(seq_len, lm_geo, masking_ratio):
  """Sampling random geometric masks.

  Randomly create a boolean mask of length `L`, consisting of subsequences of
  average length lm, masking with 0s a `masking_ratio`. The length of masking
  subsequences and intervals follow a geometric distribution.This code was
  adapted from Zerveas et al.:
  https://github.com/gzerveas/mvts_transformer/tree/master

  Args:
    seq_len: length of mask and sequence to be masked
    lm_geo: average length of masking subsequences (streaks of 0s)
    masking_ratio: proportion of L to be masked

  Returns:
    (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of
    length L
  """
  # Probability of each masking sequence stopping (used for geometric dist.)
  p_m = 1 / lm_geo
  # Probability of each unmasked sequence stopping (used for geometric dist.)
  p_u = p_m * masking_ratio / (1 - masking_ratio)

  # Start in state 0 with masking_ratio probability
  state = int(
      np.random.rand() > masking_ratio
  )  # state 0 means masking, 1 means not masking
  keep_mask = [state]

  while len(keep_mask) <= seq_len:
    # for i in range(L):
    if state:
      keep_mask += [1] * np.random.geometric(p_u)
    else:
      keep_mask += [0] * np.random.geometric(p_m)

    state = 1 - state
  return np.array(keep_mask[:seq_len]).astype(bool)


def gen_regular_mask_zerveas_flat(
    n, mr, lm_geo=3, time_ranges=(0, 120), time_units=1
):
  """Sample n separate masks.

  Each row of the n returned masks will follow the geometric distribution
  specified by mr and lm_geo

  Args:
    n: Number of masks to sample.
    mr: Missingness rate.
    lm_geo: Average length of masked sequences.
    time_ranges: Expected times (we generate a mask across the interval)
    time_units: Chunks of time that are masked together

  Returns:
    An n binary masks representing whether the chunk of time should be masked.
  """
  floor_time_vals = np.arange(time_ranges[0], time_ranges[1], time_units)

  regular_masks = []
  for i in range(n):
    if i % 500000 == 0:
      print(i)
    regular_masks.append(
        geom_noise_mask_single(len(floor_time_vals), lm_geo, mr)
    )
  return np.array(regular_masks).astype(int)


# Compute masks and save them
reg = gen_regular_mask_zerveas_flat(num_masks, mask_rate, lm_geo=lm)

np.savetxt(
    "geometric_masks_cached/mask_{}_{}_geometric.txt".format(mask_rate, lm),
    reg,
    fmt="%i",
)
