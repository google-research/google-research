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

# pylint: skip-file
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterator, Sequence, Tuple


def gen_primes():
  composites = defaultdict(list)
  n = 2
  while True:
    if n not in composites:
      yield n
      composites[n * n] = [n]
    else:
      for p in composites[n]:
        composites[n + p].append(p)
      del composites[n]
    n = n + 1


def integer_to_lattice_dimension(N, c = 1):
  logN = round(math.log2(N))
  return int(c * int(logN) // int(round(math.log2(logN))))


def first_n_primes(n):
  if n > len(first_n_primes._CACHE):
    prime_gen = gen_primes()
    first_n_primes._CACHE = tuple(int(next(prime_gen)) for _ in range(n))
  return first_n_primes._CACHE[:n]


first_n_primes._CACHE = ()


def exponent(x, p):
  e = 0
  while x % p == 0:
    e += 1
    x //= p
  return e


def iter_bits(val, width):
  """Iterate over the bits in a binary representation of `val`.

  This uses a big-endian convention where the most significant bit is yielded
  first.

  Args:
      val: The integer value. Its bitsize must fit within `width`
      width: The number of output bits.
  """
  if val.bit_length() > width:
    raise ValueError(f'{val} exceeds width {width}.')
  for b in f'{val:0{width}b}':
    yield int(b)


def is_smooth(x, smooth_bound):
  for p in first_n_primes(smooth_bound):
    while x % p == 0:
      x //= p
  return x == 1


def index_largest_prime(x):
  if x == 1:
    return 0
  i = 0
  for p in gen_primes():
    while x % p == 0:
      x //= p
    if x == 1:
      return i
    i += 1
  return 10**18  # some large number


@dataclass(frozen=True)
class SRSystem:
  sr_pairs: Tuple[int, Ellipsis]
  index_largest_prime: int


def pairs_to_sr_possiblities(
    pairs, N
):
  by_largest_prime = []
  for u, v in pairs:
    prime_index = max(
        index_largest_prime(x) for x in (u, v, abs(u - v * N))
    )  # index of smallest prime that makes (u, v) and SR pair.
    by_largest_prime.append((prime_index, u, v))
  by_largest_prime.sort()

  pairs = []
  for i, (prime_index, u, v) in enumerate(by_largest_prime):
    pairs.append((u, v))
    # if the number of pairs is larger than the index of the largest prime then
    # this prefix of pairs can build a valid system of equations.
    if i + 1 > prime_index:
      yield SRSystem(sr_pairs=pairs, index_largest_prime=prime_index)


if __name__ == '__main__':
  for sr in pairs_to_sr_possiblities([(6, 2), (5, 1), (6, 2)], 13):
    print(sr)
