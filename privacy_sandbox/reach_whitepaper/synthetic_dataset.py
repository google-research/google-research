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

"""Utililty functions for generatitng values from discrete power-law distribution."""

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
    r: A uniform random samples to transfrom into power-law one.
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
  However, that method only supports x_min and not x_max. To incorporate x_max
  support, we utilize the following equation:
  `floor( (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) + 0.5) <= x_max`
  To solve for 'r' in this equation, follow these steps:
  1. Move the floor function to the right side of the inequality:
  ```
  (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) + 0.5 < floor(x_max +1)
  ```
  2. Isolate the exponential term by subtracting 0.5 from both sides:
   ```
   (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) < floor(x_max +1) - 0.5
   ```
  3. Simplify the inequality by letting 'y = floor(x_max + 1) - 0.5':
   ```
   (x_min - 0.5) * (1 - r) ** (-1 / (a - 1)) < y
   ```
  4. Raise both sides of the inequality to the power of '-a + 1'.
  Since '-a + 1' is negative (a is >1) inequality's direction changes:
   ```
   (1 - r) > (y / (x_min - 0.5)) ** (-a + 1)
   ```
  5. Since both sides are positive, we can take the reciprocal without
  changing the inequality's direction:
   ```
   r < 1 - (y / (x_min - 0.5)) ** (-a + 1)
   ```
  6. Therefore, the solution for 'r' is:
   ```
   r < 1 - ( (floor(x_max +1) - 0.5) / (x_min - 0.5)) ** (-a + 1)
   ```

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

  # Generate n_samples many samples from uniform distribution.
  r_max = 1 - ((np.floor(x_max + 1) - 0.5) / (x_min - 0.5)) ** (-b + 1)
  r = np.random.uniform(0, r_max, n_samples)
  # Transform to discrete power law distribution.
  dpl = _sample_discrete_powerlaw_approximately_d6(b, r, x_min)
  return dpl
