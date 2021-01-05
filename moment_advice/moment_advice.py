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

"""Experiments in estimating frequency moments with advice (ongoing research).

The main function currently uses four methods to estimate frequency moments:
1. Our method of sampling according to advice (read from file)
2. Sampling according to perfect advice (the advice is the exact frequencies
   of the elements in the dataset)
3. PPSWOR (ell_1 sampling)
4. ell_2 sampling

The following types of datasets are supported as input (same format is used
for the advice files):
graph - Each line in the file represents a directed edge ("u v" or "u v time"
    for temporal graphs), and we output estimates for the frequency moments
    of the in degrees and out degrees of nodes. We allow parallel edges, that
    is, if an edge between two nodes appears twice, it contributes 2 to the
    degrees.
net_traffic - Each line in the file represents a packet ("src_ip src_port dst_ip
    dst_port protocol") and we estimate the frequency moments for unordered IP
    pairs, that is, the frequency for each IP pair is the number of packets
    send between them (in any direction).
unweighted_elements - Each line in the file is a key and represents an element
    with that key and weight 1.
weighted - Each line in the file represents a weighted data element and is of
    the form "key weight".
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range

import hashlib
import math
import time
import random
import numpy as np
import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("train_path", None, "Path to a file containing the advice "
                    "data")
flags.DEFINE_string(
    "test_path", None, "Path to the dataset for which we "
    "estimate the frequency moment")
flags.DEFINE_integer("p", 3, "Moment to estimate", lower_bound=1)
flags.DEFINE_integer("k", 100, "Sample size", lower_bound=3)
flags.DEFINE_integer("num_iter", 1, "Number of iterations", lower_bound=1)
flags.DEFINE_enum("dataset_type", "graph",
                  ["graph", "net_traffic", "unweighted_elements", "weighted"],
                  "Type of dataset")

# Draws different default randomness for the hash function each run.
SEED = str(random.randint(1, 2**64 + 1)) + str(random.randint(1, 2**64 + 1))


def hash_exp(key, seed=SEED):
  """Hash function mapping each input into a value distributed as Exp(1).

  Args:
    key: The input to be hashed
    seed: A seed to the hash function (the randomness used by the function)

  Returns:
    The hash value
  """
  m = hashlib.sha256(("%s###%s" % (seed, key)).encode("ascii"))
  uniform = float(int(m.hexdigest(), 16) + 1) / 2**256
  return -1.0 * math.log(uniform, math.e)


class MomentEstimatorSketch(object):
  """Sketch for estimating frequency moments using advice.

  The sketch maintains a sample of at most k keys. For each key, we store
  its seed, which is computed using a hash function, and its frequency so far
  (the sum of weights of elements with that key).

  The sample always contains the k keys with lowest seed. For each key x,
  the sketch gets a prediction a_x of the total frequency of x from an advice
  object. The seed of x comes from the distribution Exp(a_x ** p) (when the
  sketch is used for estimating the p-th frequency moment).
  """

  def __init__(self, k, hash_func, p, advice_obj):
    """Initializes an empty sketch/sample of specified size.

    Args:
      k: Sample size
      hash_func: The randomness used for the sample (a hash function that maps
        each key into a supposedly independent exponential random variable with
        parameter 1)
      p: The moment estimated by the sketch
      advice_obj: An object that predicts the frequencies of elements (should
        support calls of the form advice_obj.predict(key))
    """
    # Maximum sample size
    self.k = k

    # The following hash function defines all the randomness used for
    # picking the sample
    self.hash_func = hash_func

    # A dictionary containing the sampled elements
    # The dictionary key is the key of the element
    # The value is a tuple (seed, count)
    self.elements = {}

    # The advice object
    self.advice_obj = advice_obj

    # The function of the frequencies that the sketch estimates
    # For now it's the p-th frequency moment, but in the future we may
    # support other functions (passed as a parameter)
    self.func_of_freq = lambda x: x**p

  def _remove_additional_elements(self):
    """Removes any elements in the sample beyond the k lowest elements.

    Used as part of an optimization that removes excessive elements only once
    the sample reached size greater than 2k (instead of k).
    """
    # Produces a list of keys in sample sorted by seed
    sorted_elements = sorted(self.elements.items(), key=lambda x: x[1][0])

    # Removes the keys with largest seed values (beyond the
    # first k keys)
    for i in range(self.k, len(sorted_elements)):
      del self.elements[sorted_elements[i][0]]

  def process(self, key, value):
    """Processes a weighted element by the sample.

    The predicted total frequency of the key in the stream will be obtained from
    advice_obj.

    Args:
      key: The key of the element
      value: The value (weight) of the element
    """
    if key in self.elements:
      seed, count = self.elements[key]
      self.elements[key] = (seed, count + value)
    else:
      pred = self.advice_obj.predict(key)
      # The seed hash(key) is drawn from the exponential distribution,
      # with parameter that is the predicted frequency raised to the p-th
      # power.
      seed = self.hash_func(key) / float(self.func_of_freq(pred))
      self.elements[key] = (seed, value)

      # Optimization: instead of removing excess elements from the sample
      # every time its size reaches k+1, we only remove elements after
      # the number of elements in the sample exceeds 2k.
      if len(self.elements) > 2 * self.k:
        self._remove_additional_elements()

  def estimate_moment(self):
    """Estimates the p-th frequency moment of the elements processed so far.

    p is passed as a parameter to the constructor.

    Returns:
      The estimate of the p-th frequency moment
    """
    # Due to the optimization, we may store more than k elements in the
    # sample. The following removes excessive elements if needed.
    if len(self.elements) > self.k:
      self._remove_additional_elements()

    # The inclusion threshold (highest seed of element in the sample) is
    # used to compute the inclusion probabilities for the other elements
    # in the sample.
    max_in_sample = max(self.elements.items(), key=lambda x: x[1][0])
    threshold = max_in_sample[1][0]

    # Computes and sums the inverse-probability estimator for all keys
    # in the sample.
    sum_estimator = 0.0
    for key, (seed, count) in self.elements.items():
      if key != max_in_sample[0]:
        weight = self.func_of_freq(self.advice_obj.predict(key))
        # Warns us if we may run into float precision issues.
        # TODO(ofirg): change this warning to something more robust than
        # a print (and maybe use other approximations of exp() that are
        # better for this case).
        if weight * threshold < 2.0**(-24):
          print("weight * threshold < 2^{-24}")
          print(weight * threshold)
        inc_pr = 1.0 - np.exp(-1.0 * weight * threshold)
        estimator = self.func_of_freq(count) / inc_pr
        sum_estimator += estimator

    return sum_estimator


class PpsworSketch(object):
  """A simple implementation of PPSWOR sampling for aggregated data.

  Used as a benchmark for estimating moments with advice.

  The sketch assumes input that consists of (key, value) pairs, which is
  aggregated (each key appears at most once, no guarantees on the output
  otherwise). The sketch supports sampling keys with weight that are any
  power of their value. The sample can then be used to estimate frequency
  moments.
  """

  def __init__(self, k, p, sample_p=1):
    """Initializes an empty sketch/sample of specified size.

    Args:
      k: Sample size
      p: The moment estimated by the sketch
      sample_p: The power of values used for the sampling weights, that is, the
        weight used for sampling an element (key, value) is going to be value **
        sample_p.
    """
    # Maximum sample size
    self.k = k

    # A dictionary containing the sampled elements
    # The dictionary key is the key of the element
    # The value is a tuple (seed, count)
    self.elements = {}

    # The function of the frequencies that the sketch estimates
    # For now it's the p-th frequency moment, but in the future we may
    # support other functions (passed as a parameter)
    self.func_of_freq = lambda x: x**p

    # The power of values used for the sampling weights
    self.sample_p = sample_p

  def process(self, key, value):
    """Processes a weighted element by the sample.

    Args:
      key: The key of the element
      value: The value (weight) of the element

    Raises:
      Exception: Raised when seeing a key that is already in the sample (since
        we assume the data is aggregated, i.e., each key appears only once)
    """
    if key in self.elements:
      raise Exception("This implementation works only for aggregated data")
    seed = np.random.exponential(1.0 / (value**self.sample_p))
    self.elements[key] = (seed, value)

    # Optimization: instead of removing excess elements from the sample
    # every time its size reaches k+1, we only remove elements after the
    # number of elements in the sample exceeds 2k.
    if len(self.elements) > 2 * self.k:
      self._remove_additional_elements()

  def _remove_additional_elements(self):
    """Removes any elements in the sample beyond the k lowest elements.

    Used as part of an optimization that removes excessive elements only once
    the sample reached size greater than 2k (instead of k).
    """
    # Produces a list of keys in sample sorted by seed
    sorted_elements = sorted(self.elements.items(), key=lambda x: x[1][0])

    # Removes the keys with largest seed values (beyond the first k keys)
    for i in range(self.k, len(sorted_elements)):
      del self.elements[sorted_elements[i][0]]

  def estimate_moment(self):
    """Estimates the p-th frequency moments of the elements processed so far.

    p is passed as a parameter to the constructor.

    Returns:
      The estimate of the p-th frequency moment
    """
    # Due to the optimization, we may store more than k elements in the
    # sample. The following removes excessive elements if needed.
    if len(self.elements) > self.k:
      self._remove_additional_elements()

    # The inclusion threshold (highest seed of element in the sample) is
    # used to compute the inclusion probabilities for the other elements
    # in the sample.
    max_in_sample = max(self.elements.items(), key=lambda x: x[1][0])
    threshold = max_in_sample[1][0]

    # Computes and sums the inverse-probability estimator for all keys
    # in the sample.
    sum_estimator = 0.0
    for key, (seed, count) in self.elements.items():
      if key != max_in_sample[0]:
        # Warns us if we may run into float precision issues.
        # TODO(ofirg): change this warning to something more robust than
        # a print (and maybe use other approximations of exp() that are
        # better for this case).
        if (count**self.sample_p) * threshold < 2.0**(-24):
          print("(count**self.sample_p) * threshold < 2^{-24}")
          print((count**self.sample_p) * threshold)
        inc_pr = 1.0 - np.exp(-1.0 * (count**self.sample_p) * threshold)
        estimator = self.func_of_freq(count) / inc_pr
        sum_estimator += estimator

    return sum_estimator


class LookupTableAdvice(object):
  """Implementation of a simple lookup table that can be used as advice.

  (The advice predicts element frequencies based on training data.)

  Supports adding noise to the frequency counts (when generating the advice
  from the actual input): For a noise parameter q, the count of each key
  will be multiplied by a random number between 1.0 and 1.0 + q.
  """

  def __init__(self):
    """Initializes an empty lookup table."""
    self.counts = {}

  def process(self, key, value):
    """Processes an element into the lookup table.

    Args:
      key: The key of the element
      value: The value (weight) of the element
    """
    if key not in self.counts:
      self.counts[key] = 0.0
    self.counts[key] += value

  def add_noise(self, noise):
    """Adds noise to the counts of the elements processed so far.

    Args:
      noise: The noise parameter
    """
    if noise > 0.0:
      for key in self.counts:
        self.counts[key] *= 1.0 + noise * np.random.random_sample()

  def predict(self, key):
    """Predicts the frequency of a key based on its count in the lookup table.

    If the key does not appear in the table, the predicted frequency is 1.

    Args:
      key: The key whose frequency is to be estimated
    """
    return self.counts.get(key, 1.0)

  def moment(self, p):
    """Computes the p-th frequency moment of the elements processed so far.

    Args:
      p: The moment to be computed
    """
    return sum([val**p for val in self.counts.values()])


# Functions that take a set of elements and process them through one of the
# the structures above.


def ppswor_estimate_moment(elements, k, p, sample_p=1):
  """Estimates the frequency moment of a given dataset using an ell_p sample.

  The functions uses a fixed-size PPSWOR sample according to weights that are
  the p-th power of the frequency.

  Args:
    elements: The dataset (an iterator over (key, value) tuples). Must be
      aggregated, that is, each key appears at most once (otherwise there are no
      guarantees on the output).
    k: The sample size to be used
    p: Which frequency moment to estimate
    sample_p: Which power of the frequency to use as the sampling weight

  Returns:
    The estimate of the p-th frequency moment
  """
  sk = PpsworSketch(k, p, sample_p)
  for key, value in elements:
    sk.process(key, value)
  return sk.estimate_moment()


def generate_advice(it, transforms=lambda x: (x, 1)):
  """Generates a lookup table advice object from a set of input lines.

  Args:
    it: An iterator over strings, where each string represents a data element
    transforms: A function that takes as input a string representing a data
      element, and outputs a (key, value) tuple. Can be a list of such functions
      in case each line represents elements in two or more datasets (for
      example, when each line is an edge (u,v) in a graph and we wish to
      estimate the moments of the in degrees and out degrees without going
      through the data twice).

  Returns:
    The advice object
  """
  if type(transforms) is not list:
    transforms = [transforms]
  advice = [LookupTableAdvice() for i in range(len(transforms))]
  for example in it:
    for i, tr in enumerate(transforms):
      key, value = tr(example)
      advice[i].process(key, value)
  return advice


def estimate_using_advice(it, transforms, k, p, hash_exp, advice_objs):
  """Estimates the frequency moment of a dataset using sampling with advice.

  Args:
    it: An iterator over strings, where each string represents a data element
    transforms: A function that takes as input a string representing a data
      element, and outputs a (key, value) tuple. Can be a list of such functions
      in case each line represents elements in two or more datasets (for
      example, when each line is an edge (u,v) in a graph and we wish to
      estimate the moments of the in degrees and out degrees without going
      through the data twice).
    k: The sample size to be used
    p: Which frequency moment to estimate
    hash_exp: The hash function to be used by the sampling sketch. The output
      should come from the distribution Exp(1). This function defines all the
      randomness that is used for sampling, so it should be seeded differently
      in order to get different outputs.
    advice_objs: The advice object used by the sample sketch. Should be a list
      if transforms is a list (a separate advice object for each dataset we
      process).

  Returns:
    The estimate of the p-th frequency moment
  """
  if type(transforms) is not list:
    transforms = [transforms]
  if type(advice_objs) is not list:
    advice_objs = [advice_objs]
  if len(transforms) != len(advice_objs):
    raise Exception("Input mismatch")
  #TODO(ofirg): should hash_exp be a list as well (different randomness)?
  sk = [MomentEstimatorSketch(k, hash_exp, p, advice) for advice in advice_objs]
  for example in it:
    for i, tr in enumerate(transforms):
      key, value = tr(example)
      sk[i].process(key, value)
  est = [x.estimate_moment() for x in sk]
  return est


# The following functions are used to generate synthetic datasets.


def generate_dataset_zipf(n, alpha=1.1):
  """Generates a dataset according to the Zipf distribution.

  Args:
    n: The size of the dataset
    alpha: The Zipf parameter

  Returns:
     A list of (key, value) tuples, where the key of each elements is drawn from
     the Zipf distribution
  """
  return [(np.random.zipf(alpha), 1) for _ in range(n)]


def generate_dataset_uniform(n, a, b):
  """Generates a dataset according to the uniform distribution.

  Args:
    n: The size of the dataset
    a: The low end of the interval from which keys are drawn
    b: The high end of the interval from which keys are drawn

  Returns:
    A list of (key, value) tuples, where the key of each element is drawn
    uniformly in [a,b)
  """
  return [(np.random.randint(a, b), 1) for _ in range(n)]


def uniform_but_one_dataset(n, p):
  """Generates a dataset according to the AMS lower bound.

  Args:
    n: The size of the dataset
    p: The frequency moment (used as a parameter in the lower bound)

  Returns:
    A list of (key, value) tuples, where all keys have weight one, except for
    one key which accounts for half of the p-th frequency moment
  """
  elements = []
  for i in range(n):
    elements.append((i, 1))
  elements.append((1, (n**(1.0 / p)) - 1))
  return elements


def uniform_but_one_dataset_no_weight(n, p):
  """Generates an unweighted dataset according to the AMS lower bound.

  Args:
    n: The size of the dataset
    p: The frequency moment (used as a parameter in the lower bound)

  Returns:
    A list of keys, where all keys appear once, except for one key which
    accounts for half of the p-th frequency moment
  """
  elements = []
  for i in range(n):
    elements.append(i)
  for i in range(int(n**(1.0 / p)) - 1):
    elements.append(1)
  return elements


# Input transforms:
# Functions that take as input a string representing an element (supposedly
# a line read from a file) and output a data element, that is, a (key, value)
# tuple.
# These transforms can be a list, see, for example, the list
# GRAPH_DEG_TRANSFORMS for parsing lists of graph edges.

# These functions take a directed graph edge ("u v" or "u v time" for temporal
# graphs) and return u and v.
# Used to estimate the moment of the in degrees and the out degrees.
GRAPH_DEG_TRANSFORMS = [
    lambda x: (int(x.split()[0]), 1), lambda x: (int(x.split()[1]), 1)
]


# Takes a list representing a network packet:
# "src_ip src_port dst_ip dst_port protocol"
# and returns a weight-one data element with key that is an unordered tuple of
# src_ip and dst_ip (the IPs will be appear in the same order in the output
# tuple, no matter which was src_ip and which was dst_ip).
def five_tuple_string_to_unordered_ip_pair(s):
  src_ip, src_port, dst_ip, dst_port, prot = s.split()
  return (tuple(sorted([src_ip, dst_ip])), 1)


NETWORK_DATA_TRANSFORM = [five_tuple_string_to_unordered_ip_pair]

# Takes a key as input and outputs an element (key, 1).
UNWEIGHTED_ELEMENTS_TRANSFORM = [lambda x: (x, 1)]

# Takes a string representing integer-weighted element ("key weight") and
# returns (key, weight).
WEIGHTED_ELEMENTS_TRANSFORM = [lambda x: (x.split()[0], int(x.split()[1]))]


def basic_experiments(k, p, num_iter, transforms, train_path, test_path):
  """Main function running a set of experiments.

  Compares estimates for the frequency moment computed using noisy advice (taken
  from some training dataset), perfect advice (the advice predicts the frequency
  exactly right), PPSWOR (ell_1 sampling), and exact ell_2 sampling.

  Args:
    k: The sample size to be used
    p: Which frequency moment to estimate
    num_iter: The number of iterations (how many estimates to compute using each
      method)
    transforms: A function that takes as input a string representing a data
      element, and outputs a (key, value) tuple. Can be a list of such functions
      in case each line represents elements in two or more datasets (for
      example, when each line is an edge (u,v) in a graph and we wish to
      estimate the moments of the in degrees and out degrees without going
      through the data twice).
    train_path: Path to a file containing the advice data
    test_path: Path to the dataset for which we estimate the frequency moment
  """
  start_time = time.time()
  print("Generating advice")
  with open(train_path, "r") as f:
    train_data = f.readlines()
  advice = generate_advice(train_data, transforms)
  print("Done: " + str(time.time() - start_time) + "\n")

  print("Generating real count for test data")
  with open(test_path, "r") as f:
    test_data = f.readlines()
  real_count = generate_advice(test_data, transforms)
  print("Done: " + str(time.time() - start_time))
  actual = [x.moment(p) for x in real_count]
  print("Actual moments:")
  print(actual)

  print("Estimating using noisy advice:")
  est_with_advice = []
  for i in range(num_iter):
    seed = str(random.randint(1, 2**64 + 1)) + str(random.randint(1, 2**64 + 1))
    this_hash_exp = lambda x: hash_exp(x, seed)
    est_with_advice.append(
        estimate_using_advice(test_data, transforms, k, p, this_hash_exp,
                              advice))
  print("Done: " + str(time.time() - start_time))
  print(est_with_advice)

  print("Estimating using perfect advice:")
  est_perfect_advice = []
  for i in range(num_iter):
    seed = str(random.randint(1, 2**64 + 1)) + str(random.randint(1, 2**64 + 1))
    this_hash_exp = lambda x: hash_exp(x, seed)
    est_perfect_advice.append(
        estimate_using_advice(test_data, transforms, k, p, this_hash_exp,
                              real_count))
  print("Done: " + str(time.time() - start_time))
  print(est_perfect_advice)

  print("Estimating using PPSWOR:")
  est_ppswor = []
  for i in range(num_iter):
    cur = []
    for x in real_count:
      cur.append(ppswor_estimate_moment(x.counts.items(), k, p))
    est_ppswor.append(cur)
  print("Done: " + str(time.time() - start_time))
  print(est_ppswor)

  print("Estimating using ell_2 PPSWOR:")
  est_ppswor_l2 = []
  for i in range(num_iter):
    cur = []
    for x in real_count:
      cur.append(ppswor_estimate_moment(x.counts.items(), k, p, 2))
    est_ppswor_l2.append(cur)
  print("Done: " + str(time.time() - start_time))
  print(est_ppswor_l2)


def main(args):
  if FLAGS.train_path is None:
    raise Exception("Missing train path")
  if FLAGS.test_path is None:
    raise Exception("Missing test path")

  print("Running %d iterations of k = %d" % (FLAGS.num_iter, FLAGS.k))
  print("Train path: " + FLAGS.train_path)
  print("Test path: " + FLAGS.test_path)
  print("Estimating moment %d" % FLAGS.p)

  if FLAGS.dataset_type == "graph":
    transforms = GRAPH_DEG_TRANSFORMS
    print("Dataset type: graph")
  elif FLAGS.dataset_type == "net_traffic":
    transforms = NETWORK_DATA_TRANSFORM
    print("Dataset type: network traffic")
  elif FLAGS.dataset_type == "unweighted_elements":
    transforms = UNWEIGHTED_ELEMENTS_TRANSFORM
    print("Dataset type: unweighted elements")
  elif FLAGS.dataset_type == "weighted":
    transforms = WEIGHTED_ELEMENTS_TRANSFORM
    print("Dataset type: weighted elements")

  basic_experiments(FLAGS.k, FLAGS.p, FLAGS.num_iter, transforms,
                    FLAGS.train_path, FLAGS.test_path)


if __name__ == "__main__":
  app.run(main)
