# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Utility functions for aptamer simulations.
"""

import numpy


# numpy.random.RandomState uses uint32 seeds
RANDOM_SEED_MAX = 2 ** 32


def random_seed_stream(random_seed=None):
  """Yield an infinite stream of numbers for seeding random number generators.

  This method is not proven to be cryptographically secure, and only explores a
  small portion of the state space for NumPy's random number generator. Still,
  it's a useful shortcut for writing decoupled functions that rely on random
  state. See this thread for extensive discussion of its merits and the
  alternatives:
  https://mail.scipy.org/pipermail/numpy-discussion/2016-May/075487.html

  Example:

    >>> seed_gen = random_seed_stream(42)
    >>> next(seed_gen)
    1608637542

  Args:
    random_seed: optional integer used to seed this stream of random seeds.

  Yields:
    Integer seeds suitable for use in numpy.random.RandomState. Each seed is
    independent and psuedo-randomly generated from the `random_seed` argument.
  """
  rs = numpy.random.RandomState(random_seed)
  seed = rs.randint(RANDOM_SEED_MAX)

  while True:
    yield seed

    # Incrementing is better than generating new seeds with a call to randint,
    # because with random seeds collisions are likely after only around 2 ** 16
    # samples due to the birthday paradox.
    seed = (seed + 1) % RANDOM_SEED_MAX


def target_occupancy(target_affinity,
                     serum_affinity,
                     target_concentration,
                     serum_concentration):
  """Calculate target site occupancy in the presence of serum.

  Assumes that the amount of target and serum are very large (compared to the
  amount of aptamers), such that their concentration can be treated as fixed.

  TODO(mdimon): Validate this assumption.

  All argument should be provided with the same units.

  Args:
    target_affinity: number or ndarray-like giving affinity for the target site.
    serum_affinity: number or ndarray-like giving serum affinity.
    target_concentration: number or ndarray-like giving target concentration.
    serum_concentration: number or ndarray-like giving serum concentration.

  Returns:
    Number or ndarray-like giving the fraction of bound target sites.
  """
  # see Equation (7) from:
  # https://en.wikipedia.org/wiki/Competitive_inhibition#Derivation
  numerator = serum_affinity * target_concentration
  denominator = (target_affinity * serum_affinity
                 + serum_affinity * target_concentration
                 + target_affinity * serum_concentration)
  return numerator / denominator
