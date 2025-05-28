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

"""CountSketch and BCountSketch with median and sign alignment estimators."""

import math
import time

from farmhash import FarmHash64
import numpy as np


class Sketch:
  """Common base for CountSketch and BCountSketch, which differ in their signs_and_buckets() method only."""

  # width : b
  # depth : d / b == ell
  # num_keys : input vector dimension
  def __init__(self, a_rnd_seed, a_width, a_depth, a_num_keys):
    self.rnd_seed = a_rnd_seed
    # avoid promotions to float64 in expressions in case it's a numpy int64
    self.width = int(a_width)
    self.depth = a_depth
    # Only used in sweep_depth, if > 0,
    # impacts estimates only, update ignores it and updates the whole sketch
    self.limit_depth = -1
    # flattened row-major for easier indexing later
    self.data = np.zeros(a_depth * a_width)
    self.num_keys = a_num_keys
    # Non-zero entries of the sketching matrix.
    # With this Python code it's much quicker to precompute these than
    # to do so on demand as a proper sketch would do.
    self.signs_and_buckets_cache = [
        self.signs_and_buckets(j) for j in range(self.num_keys)
    ]

  # returns (sign, pos) tuple of all buckets
  # that index j in an original vector likes
  # sign is array of +-1
  # pos in array of [0, depth*width)
  # If we write the sketch as A*v for sparse matrix A, then these are the
  # non-zeros in the jth row of A.
  # Consider converting everything to sparse matrix * vector if Python
  # loops are too slow.
  def signs_and_buckets(self, j):
    del j
    return []

  # Update with one entry of input vector v[idx] = val
  def update_one(self, idx, val):
    assert idx < self.num_keys
    signs, pos = self.signs_and_buckets_cache[idx]
    # print(idx, signs, pos, val)
    if pos.size != 0:
      self.data[pos] += signs * val

  # Updates with the whole vector
  def update(self, vector):
    assert len(vector) == self.num_keys
    for j in range(len(vector)):
      self.update_one(j, vector[j])

  # j index in original vector
  def median_estimate(self, j):
    """Returns the median estimate."""
    assert j < self.num_keys
    signs, pos = self.signs_and_buckets_cache[j]
    if self.limit_depth > 0:
      signs = signs[pos < self.limit_depth * self.width]
      pos = pos[pos < self.limit_depth * self.width]
    if pos.size == 0:
      return 0
    vals = signs * self.data[pos]
    # print(len(vals), vals)

    # Somehow np.median is pretty slow, np.mean is much much quicker.
    # E.g. np.median maskes heaviest_key_median 5x (!) slower than
    # heaviest_key_signs for width = 7, depth = 11.
    # Can show warnings if BCountSketch comes up with an empty list
    # return np.median(vals)
    # Our median implementation that is surprisingly much quicker.
    vals = np.sort(vals)
    mid = vals.size // 2
    if vals.size % 2 == 0:  # even, not empty
      return (vals[mid - 1] + vals[mid]) / 2.0
    else:  # odd
      return vals[mid]

  # Returns the index of heaviest entry in the original vector using the
  # median estimator
  def heaviest_key_median(self):
    estimated = [self.median_estimate(i) for i in range(self.num_keys)]
    return np.argmax(np.abs(estimated))

  # j index in original vector
  # Should be in [0.5, 1].
  # Close to 1 if item j is heavy, close to 0.5 if item j is light.
  def sign_alignment(self, j):
    """Returns the fraction of buckets that have the same sign as the majority of buckets that like item j."""
    assert j < self.num_keys
    signs, pos = self.signs_and_buckets_cache[j]
    if self.limit_depth > 0:
      signs = signs[pos < self.limit_depth * self.width]
      pos = pos[pos < self.limit_depth * self.width]
    if pos.size == 0:
      return 0
    estimates = signs * self.data[pos]
    # Section 3.3 Basic estimates, equation (10) in the paper
    num_positive = 0
    num_negative = 0
    for val in estimates:
      if val > 0:
        num_positive += 1
      elif val < 0:
        num_negative += 1
      else:  # val == 0
        num_negative += 0.5
        num_positive += 0.5
    # Paper divides by expected number of buckets that like item j,
    # (d/b) == self.width,
    # we divide by actual number of bucket that item j likes.
    # Can show warnings if BCountSketch comes up with an empty list
    return max(num_positive, num_negative) / len(estimates)

  # Returns the index of heaviest entry in the original vector with the highest
  # sign alignment
  def heaviest_key_signs(self):
    p = [self.sign_alignment(i) for i in range(self.num_keys)]
    return np.argmax(p)


class CountSketch(Sketch):
  """Classic CountSketch."""

  def signs_and_buckets(self, j):
    # i in [0,depth)
    # j index in original vector
    def get_sign(i):
      h_sign = FarmHash64(np.array([self.rnd_seed, 0, i, j]))
      sign = 2 * (h_sign % 2) - 1
      return sign

    def get_pos(i):
      h_pos = FarmHash64(np.array([self.rnd_seed, 1, i, j]))
      pos = i * self.width + h_pos % self.width  # concatenate sketch rows
      return pos

    signs = np.array([get_sign(i) for i in range(self.depth)])
    pos = np.array([get_pos(i) for i in range(self.depth)])
    return (signs, pos)


class BCountSketch(Sketch):
  """BucketCountSketch from the paper."""

  def signs_and_buckets(self, j):
    signs = []
    buckets = []
    # start = time.perf_counter()
    seed = FarmHash64(np.array([self.rnd_seed, j]))
    my_rng = np.random.default_rng(seed)
    for i in range(self.depth):
      # for pos in range(self.width):
      #  # Choose each position in a row with prob 1/width
      #  if my_rng.choice(self.width) == 0:
      #    signs.append(my_rng.choice([-1, 1]))
      #    buckets.append(i*self.width + pos)  # concatenate sketch rows
      #
      # Equivalent to the above, runs the loop body
      # once in expectation instead of self.width times.
      # Choose each position in a row with prob 1/width, hence gaps are
      # geometric
      pos = my_rng.geometric(1 / self.width) - 1
      while pos < self.width:
        signs.append(my_rng.choice([-1, 1]))
        buckets.append(i * self.width + pos)  # concatenate sketch rows
        pos += rng.geometric(1 / self.width)
    # print("BCS signs_and_buckets", time.perf_counter() - start)
    return (np.array(signs), np.array(buckets))


rnd_seed = 2022
rng = np.random.default_rng(rnd_seed)

width = 7
depth = 11

# width = 7
# depth = 101

# width = 7
# depth = 51


def sanity_check():
  """For debugging with couple values."""
  nkeys = 100
  u = rng.normal(loc=0.0, scale=1, size=nkeys)
  u[1] = 42
  u[2] = -60

  for sketch_class in [CountSketch, BCountSketch]:
    s = sketch_class(rnd_seed, width, depth, nkeys)

    s.update(u)
    print(sketch_class.__name__)
    # print(s.data)
    for i in [1, 2, 3]:
      print("index", i, "value", u[i], "estimated", s.median_estimate(i))
    print("heaviest index", np.argmax(np.abs(u)), "median estimator",
          s.heaviest_key_median(), "sign alignment estimator",
          s.heaviest_key_signs())
    # continue
    # w and u differ in one entry only.
    w = np.copy(u)
    w[9] = 10000
    s.update_one(9, w[9] - v[9])
    # Sign align finds a heavy key but not the heaviest.
    # Add print(p) above to see why.
    print("heaviest index", np.argmax(np.abs(w)), "median estimator",
          s.heaviest_key_median(), "sign alignment estimator",
          s.heaviest_key_signs())


num_trials = 100
# num_trials = 50
num_keys = 200
# Noise (light) items
v = rng.normal(loc=0.0, scale=1, size=num_keys)
# Success probabilities are somewhat higher with Rademacher across the board.
# Oddly it's a bit slower, perhaps due to ties in some sort or median (?)
# v = rng.choice([-1, 1], size=num_keys)
heaviest_noise = np.max(np.abs(v))
print("max(abs(v))", heaviest_noise)
# We'll set the heaviest value to these values,
# ceil is to make sure that this is indeed the heaviest
# It should actually go up to norm(v) ~ sqrt(num_keys) to satisfy
# heavy hitter definition.
heavy_values = np.arange(math.ceil(heaviest_noise), 10.1, 0.5)
# heavy_values = np.arange(100, 105, 0.5)

depth_values = np.append(np.arange(1, 5), np.arange(5, 30, 5))
width_values = np.arange(3, 21, 2)

# TODO(stamas): Undo some of the copy-paste (modify) for
# {heavy, depth, width} sweeps, at least for the run_* methods.


def sweep_heavy_trial(sketch, num_success):
  """Updates num_success table of shape (len(heavy_values), 2)."""
  # 1st column is for the median, 2nd is for the sign alignment estimator

  # start = time.perf_counter()
  sketch.update(v)
  # print("Sketch update time",time.perf_counter() - start)
  heavy_idx = 1  # idx 1 will be the heaviest
  prev_h = v[heavy_idx]
  for i in range(len(heavy_values)):
    h = heavy_values[i]
    # start = time.perf_counter()
    sketch.update_one(heavy_idx, h - prev_h)
    prev_h = h
    # t1 = time.perf_counter()
    heaviest_median = sketch.heaviest_key_median()
    # t2 = time.perf_counter()
    heaviest_signs = sketch.heaviest_key_signs()
    # end = time.perf_counter()
    # print("time: update_one", t1-start, "heaviest_median", t2-t1,
    # "heaviest_signs", end-t2)
    if heaviest_median == heavy_idx:
      num_success[i, 0] += 1
    if heaviest_signs == heavy_idx:
      num_success[i, 1] += 1


# Return a table of shape (len(heavy_values), 2) counting number of times
# the sketch correctly found the heaviest key
# 1st column is for the median, 2nd is for the sign alignment estimator
def sweep_heavy(sketch_class):
  print(sketch_class.__name__)
  num_success = np.zeros((len(heavy_values), 2))
  for t in range(num_trials):
    seed = rnd_seed + 47 * t
    start = time.perf_counter()
    sketch = sketch_class(seed, width, depth, num_keys)
    sweep_heavy_trial(sketch, num_success)
    print("Heavy trial #", t, "time", time.perf_counter() - start)
  return num_success


def run_sweep_heavy():
  """Writes a data file to plot containing the number of times the sketch-estimator combination successfully identified the heaviest key as we increase the magnitude of the heaviest item."""

  cs_num_success = sweep_heavy(CountSketch)
  bcs_num_success = sweep_heavy(BCountSketch)

  heavy_values_col = np.expand_dims(heavy_values, axis=-1)
  table = np.concatenate((heavy_values_col, cs_num_success, bcs_num_success),
                         axis=1)
  print(table)
  file_name = "sweep_heavy" + str(depth) + "X" + str(width) + "t" + str(
      num_trials) + "n" + str(num_keys) + ".tsv"
  np.savetxt(
      file_name,
      table,
      fmt="%g",
      header="heavy_value,CountSketchMedian,CountSketchSigns,BCountSketchMedian,BCountSketchSigns"
  )
  print("To plot the results run: python3 plot.py", file_name, num_trials)


def sweep_depth_trial(sketch, num_success):
  """Updates num_success table of shape (len(depth_values), 2)."""
  # 1st column is for the median, 2nd is for the sign alignment estimator
  # start = time.perf_counter()
  w = np.copy(v)
  heavy_idx = 1  # idx 1 will be the heaviest
  w[heavy_idx] = 16
  # print("Sketch update time",time.perf_counter() - start)
  sketch.update(w)
  for i in range(len(depth_values)):
    d = depth_values[i]
    sketch.limit_depth = d
    # sketch = sketch_class(seed + 20220409*(i+1), width, d, num_keys)
    # sketch.update(w)
    # start = time.perf_counter()
    heaviest_median = sketch.heaviest_key_median()
    # t1 = time.perf_counter()
    heaviest_signs = sketch.heaviest_key_signs()
    # end = time.perf_counter()
    # print("time: heaviest_median", t1-start, "heaviest_signs", end-t1)
    if heaviest_median == heavy_idx:
      num_success[i, 0] += 1
    if heaviest_signs == heavy_idx:
      num_success[i, 1] += 1


def sweep_depth(sketch_class):
  """Return a table of shape (len(depth_values), 2) counting number of times the sketch correctly found the heaviest key."""
  # 1st column is for the median, 2nd is for the sign alignment estimator
  print(sketch_class.__name__)
  num_success = np.zeros((len(depth_values), 2))
  max_depth = np.max(depth_values)
  print("Depth values", depth_values)
  for t in range(num_trials):
    seed = rnd_seed + 53 * t
    start = time.perf_counter()
    sketch = sketch_class(seed, width, max_depth, num_keys)
    sweep_depth_trial(sketch, num_success)
    # sweep_depth_trial(sketch_class, seed, num_success)
    print("Depth trial #", t, "time", time.perf_counter() - start)
  return num_success


def run_sweep_depth():
  """Writes a data file to plot containing the number of times the sketch-estimator combination successfully identified the heaviest key as we increase depth."""
  cs_num_success = sweep_depth(CountSketch)
  bcs_num_success = sweep_depth(BCountSketch)

  depth_values_col = np.expand_dims(depth_values, axis=-1)
  table = np.concatenate((depth_values_col, cs_num_success, bcs_num_success),
                         axis=1)
  print(table)
  file_name = "sweep_depth_w" + str(width) + "t" + str(num_trials) + "n" + str(
      num_keys) + ".tsv"
  np.savetxt(
      file_name,
      table,
      fmt="%g",
      header="depth,CountSketchMedian,CountSketchSigns,BCountSketchMedian,BCountSketchSigns"
  )
  print("To plot the results run: python3 plot.py", file_name, num_trials)


def sweep_width_trial(sketch_class, seed, num_success):
  """Updates num_success table of shape (len(width_values), 2)."""
  # 1st column is for the median, 2nd is for the sign alignment estimator.

  # start = time.perf_counter()
  w = np.copy(v)
  heavy_idx = 1  # idx 1 will be the heaviest
  w[heavy_idx] = 16
  for i in range(len(width_values)):
    wv = width_values[i]
    sketch = sketch_class(seed + 20010510 * (i + 1), wv, depth, num_keys)
    sketch.update(w)
    # start = time.perf_counter()
    heaviest_median = sketch.heaviest_key_median()
    # t1 = time.perf_counter()
    heaviest_signs = sketch.heaviest_key_signs()
    # end = time.perf_counter()
    # print("time: heaviest_median", t1-start, "heaviest_signs", end-t1)
    if heaviest_median == heavy_idx:
      num_success[i, 0] += 1
    if heaviest_signs == heavy_idx:
      num_success[i, 1] += 1


# Return a table of shape (len(width_values), 2) counting number of times
# the sketch correctly found the heaviest key
# 1st column is for the median, 2nd is for the sign alignment estimator
def sweep_width(sketch_class):
  print(sketch_class.__name__)
  num_success = np.zeros((len(width_values), 2))
  print("Width values", width_values)
  for t in range(num_trials):
    seed = rnd_seed + 59 * t
    start = time.perf_counter()
    sweep_width_trial(sketch_class, seed, num_success)
    print("Width trial #", t, "time", time.perf_counter() - start)
  return num_success


def run_sweep_width():
  """Writes a data file to plot containing the number of times the sketch-estimator combination successfully identified the heaviest key as we increase width."""
  cs_num_success = sweep_width(CountSketch)
  bcs_num_success = sweep_width(BCountSketch)

  width_values_col = np.expand_dims(width_values, axis=-1)
  table = np.concatenate((width_values_col, cs_num_success, bcs_num_success),
                         axis=1)
  print(table)
  file_name = "sweep_width_d" + str(depth) + "t" + str(num_trials) + "n" + str(
      num_keys) + ".tsv"
  np.savetxt(
      file_name,
      table,
      fmt="%g",
      header="width,CountSketchMedian,CountSketchSigns,BCountSketchMedian,BCountSketchSigns"
  )
  print("To plot the results run: python3 plot.py", file_name, num_trials)


# sanity_check()

print("depth", depth, "width", width, "num_trials", num_trials, "num_keys",
      num_keys)

run_sweep_heavy()
run_sweep_depth()
run_sweep_width()
