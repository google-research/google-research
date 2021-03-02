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

"""Sampling sketches with privacy guarantees.

This package was written for the experimental part of a research project on
adding differential privacy guarantees to sampling sketches. It contains an
implementation of threshold sampling, using either PPSWOR (probability
proportional to size and without replacement) or priority sampling as the
underlying sampling method.

The dataset consists of elements that are (key, weight) pairs. The
implementation assumes that the dataset is aggregated (each key appears once).
As a result, the weight of a data element is equal to the total frequency of
that key, so we can use the terms weight and frequency interchangeably.

For more information on PPSWOR and using bottom-k samples for estimating
statistics, see, for example, Section 2 in
E. Cohen and O. Geri, Sampling Sketches for Concave Sublinear Functions of
Frequencies, NeurIPS 2019
https://arxiv.org/abs/1907.02218
In comparison, here we assume that the sampling threshold is a parameter, and
in the linked paper the sampling threshold is k-th lowest score.

For more information on priority sampling, see
N. Duffield, C. Lund, and M. Thorup, Priority Sampling for Estimation of
Arbitrary Subset Sums, J. ACM 2007
https://nickduffield.net/download/papers/priority.pdf
"""

import abc
import math
import random


class SamplingMethod(abc.ABC):
  """Functions to compute score and inclusion probability for a sampling method.

  Threshold sampling works by computing a random score for each key, and keeping
  the keys with score below a given threshold (a parameter). This class includes
  (static) functions to compute the score of a key and the probability that a
  key is sampled. The functions satisfy the invariant
  Pr[sampling_score(weight) < threshold] == inclusion_prob(weight, threshold).
  """

  @staticmethod
  @abc.abstractmethod
  def sampling_score(weight):
    """Computes the score for a key with a given weight.

    Args:
      weight: The key weight

    Returns:
      The random score for the key
    """
    raise NotImplementedError(
        "sampling_score is abstract and needs to be implemented in a derived "
        "class")

  @staticmethod
  @abc.abstractmethod
  def inclusion_prob(weight, threshold):
    """Computes the probability that a key will be included in sample.

    Args:
      weight: The key weight
      threshold: The threshold used by the sample

    Returns:
      The inclusion probability
    """
    raise NotImplementedError(
        "inclusion_prob is abstract and needs to be implemented in a derived "
        "class")


class PpsworSamplingMethod(SamplingMethod):
  """Functions to compute score and inclusion probability for PPSWOR.

  For PPSWOR, the random score is drawn from Exp(weight).
  """

  @staticmethod
  def sampling_score(weight):
    if weight < 0.0:
      raise ValueError("The key weight %f should be nonnegative." % weight)
    if weight == 0.0:
      return math.inf
    return random.expovariate(1.0) / weight

  @staticmethod
  def inclusion_prob(weight, threshold):
    if weight < 0.0 or threshold < 0.0:
      raise ValueError(
          "The key weight %f and threshold %f should be nonnegative." %
          (weight, threshold))
    return 1.0 - math.exp(-1.0 * weight * threshold)


class PrioritySamplingMethod(SamplingMethod):
  """Functions to compute score and inclusion probability for priority sampling.

  For priority sampling, the score is computing by drawing a random number from
  U(0,1) and dividing by the weight of the key.
  """

  @staticmethod
  def sampling_score(weight):
    if weight < 0.0:
      raise ValueError("The key weight %f should be nonnegative." % weight)
    if weight == 0.0:
      return math.inf
    return random.uniform(0.0, 1.0) / weight

  @staticmethod
  def inclusion_prob(weight, threshold):
    if weight < 0.0 or threshold < 0.0:
      raise ValueError(
          "The key weight %f and threshold %f should be nonnegative." %
          (weight, threshold))
    if weight * threshold > 1.0:
      return 1.0
    return weight * threshold


class ThresholdSample(object):
  """Implementation of a threshold sampling sketch (without privacy guarantees).

  Threshold sampling works by computing a random score for each key, and keeping
  the keys with score below a given threshold (a parameter). The keys that are
  kept are the sample.

  The score is determined by the underlying sampling method, e.g., PPSWOR or
  priority sampling.

  This sketch only supports aggregated data: each key can only appear once in
  the dataset.
  """

  def __init__(self, threshold, sampling_method=PpsworSamplingMethod):
    """Initializes an empty sample.

    Args:
      threshold: The sampling threshold
      sampling_method: A class that provides functions to compute the score and
        inclusion probability according to the underlying sampling method (e.g.,
        PPSWOR, which is the default value)
    """
    self.threshold = threshold
    self.sampling_method = sampling_method

    # The following stores the sampled elements. It is a dict where the keys are
    # the keys in the samples, and the value for each key is (weight, score).
    self.elements = {}

  def process(self, key, weight):
    """Processes a data element into the sketch.

    Args:
      key: The key of the element. We assume the data is aggregated, so the key
        has to be unique to this element.
      weight: The weight of this element/key
    """
    if key in self.elements:
      raise ValueError("Only works for aggregated data: repeated key %s" % key)
    score = self.sampling_method.sampling_score(weight)
    if score < self.threshold:
      self.elements[key] = (weight, score)

  def estimate_full_statistics(self):
    """Estimates the full statistics/sum of the weights of the entire dataset.

    The estimate is computed by summing the inverse probability estimator for
    each one of the keys in the sample.

    Returns:
      An estimate for the sum of the weights of all keys.
    """
    sum_estimate = 0.0
    for weight, _ in self.elements.values():
      sum_estimate += weight / self.sampling_method.inclusion_prob(
          weight, self.threshold)
    return sum_estimate


# A default value for a parameter that trades off time and space.
# When computing values iteratively, we store some of the computed values to
# avoid recomputation. The store_every parameter controls how many values we
# store.
STORE_EVERY_DEFAULT = 1000


class PrivateThresholdSampleKeysOnly(object):
  """Threshold sampling with differential privacy (returns sampled keys only).

  This class implements threshold sampling, and then performs subsampling to
  satisfy the differential privacy constraints. The private sample only includes
  keys (and no information about their frequencies).

  The sketch only supports aggregated data: each key can only appear once in the
  dataset.
  """

  def __init__(self,
               threshold,
               eps,
               delta,
               sampling_method=PpsworSamplingMethod,
               store_every=STORE_EVERY_DEFAULT):
    """Initializes an empty sample.

    Args:
      threshold: The sampling threshold
      eps: The differential privacy parameter epsilon
      delta: The differential privacy parameter delta
      sampling_method: A class that provides functions to compute the score and
        inclusion probability according to the underlying non-private sampling
        method (e.g., PPSWOR, which is the default value).
      store_every: A parameter that trades off the use of space and time. When
        an element is processed into the sketch, we need to compute its
        inclusion probability iteratively, and this parameter controls how many
        such values we store (in order to not recompute). The number of values
        stored is (max weight seen so far / store_every) + 1.
    """
    self.threshold = threshold
    self.sampling_method = sampling_method
    self.eps = eps
    self.delta = delta
    self._store_every = store_every

    # Stores the computed inclusion probabilities: self._inclusion_prob[i] is
    # the inclusion probability of a key with frequency i * store_every.
    self._inclusion_prob = [0.0]

    # The set of keys that were sampled
    self.elements = set()

  @classmethod
  def from_non_private(cls,
                       sample,
                       eps,
                       delta,
                       store_every=STORE_EVERY_DEFAULT):
    """Creates a private sample from a given non-private threshold sample.

    The input sample is subsampled to satisfy the differential privacy
    constraints.

    Args:
      sample: The input non-private sample. Must be of type ThresholdSample.
      eps: The differential privacy parameter epsilon
      delta: The differential privacy parameter delta
      store_every: A parameter that trades off the use of space and time. When
        an element is processed into the sketch, we need to compute its
        inclusion probability iteratively, and this parameter controls how many
        such values we store (in order to not recompute). The number of values
        stored is (max weight seen so far / store_every) + 1.

    Returns:
      A private sample derived from the input non-private sample.
    """
    if not isinstance(sample, ThresholdSample):
      raise TypeError(
          "Tried to create a private sample from a non-sample object")
    s = cls(sample.threshold, eps, delta, sample.sampling_method, store_every)
    for key, (weight, _) in sample.elements.items():
      non_private_inclusion_prob = s.sampling_method.inclusion_prob(
          weight, s.threshold)
      private_inclusion_prob = s.compute_inclusion_prob(weight)
      # In the private sample, the inclusion probability of key should be
      # private_inclusion_prob.
      # The key was included in the non-private sample with probability
      # non_private_inclusion_prob, so we add it to the private sample with
      # probability private_inclusion_prob/non_private_inclusion_prob.
      if random.random() < (private_inclusion_prob /
                            non_private_inclusion_prob):
        s.elements.add(key)
    return s

  def compute_inclusion_prob(self, freq):
    """Computes the inclusion probability of a key in the private sample.

    The inclusion probability of a key in the private sample is determined by
    its frequency and the differential privacy parameters.

    The current implementation computes the maximum allowed inclusion
    probability by iterating from 1 to the frequency. To avoid recomputation, we
    store some of the computed inclusion probabilities for future calls to this
    function.

    Args:
      freq: The frequency of the key

    Returns:
      The inclusion probability of a key with the given frequency in the private
      sample.
    """
    if not isinstance(freq, int):
      raise TypeError("The frequency %f should be of type int" % freq)
    if freq < 0:
      raise ValueError("The frequency %d should be nonnegative" % freq)
    # Find the closest precomputed value to start iterating from
    start_arr_index = min(freq // self._store_every,
                          len(self._inclusion_prob) - 1)
    cur_prob = self._inclusion_prob[start_arr_index]
    cur_freq = start_arr_index * self._store_every
    # Invariant: before/after each iteration of the loop, cur_prob is the
    # inclusion probability for a key with frequency cur_freq.
    while cur_freq < freq:
      cur_freq += 1
      cur_prob = min(
          self.sampling_method.inclusion_prob(cur_freq, self.threshold),
          math.exp(self.eps) * cur_prob + self.delta,
          1.0 + math.exp(-1.0 * self.eps) * (cur_prob + self.delta - 1))
      if cur_freq == len(self._inclusion_prob) * self._store_every:
        self._inclusion_prob.append(cur_prob)
    return cur_prob

  def process(self, key, weight):
    """Processes a data element into the sketch.

    Args:
      key: The key of the element. We assume the data is aggregated, so the key
        has to be unique to this element.
      weight: The weight of this element/key
    """
    if key in self.elements:
      raise ValueError("Only works for aggregated data: repeated key %s" % key)
    inclusion_prob = self.compute_inclusion_prob(weight)
    if random.random() < inclusion_prob:
      self.elements.add(key)


class PrivateThresholdSampleWithFrequencies(object):
  """Threshold sampling with differential privacy (with frequencies).

  This class implements threshold sampling, and then performs subsampling to
  satisfy the differential privacy constraints. Together with each sampled key,
  the sketch reports a random value that is between 1 and the frequency of the
  key, taken with a distribution that:
  1. Ensures that the differential privacy constraints are satisfied.
  2. Has as much probability mass as possible on higher values (which are closer
  to the true frequency).
  The reported frequency values can be used to estimate statistics on the data.

  The sketch only supports aggregated data: each key can only appear once in the
  dataset.
  """

  def __init__(self,
               threshold,
               eps,
               delta,
               sampling_method=PpsworSamplingMethod,
               store_every=STORE_EVERY_DEFAULT):
    """Initializes an empty sample.

    Args:
      threshold: The sampling threshold
      eps: The differential privacy parameter epsilon
      delta: The differential privacy parameter delta
      sampling_method: A class that provides functions to compute the score and
        inclusion probability according to the underlying sampling method (e.g.,
        PPSWOR, which is the default value)
      store_every: A parameter that trades off the use of space and time. When
        an element is processed into the sketch, we need to compute the
        distribution of reported frequency iteratively, and this parameter
        controls how many such distributions we store (to avoid recomputation).
    """
    self.threshold = threshold
    self.sampling_method = sampling_method
    self.eps = eps
    self.delta = delta
    self._store_every = store_every

    # The stored distributions of reported frequencies
    # self._reported_weight_dist[i] is the probability distribution used for
    # keys with frequency i * store_every.
    # The key is not sampled with probability self._reported_weight_dist[i][0].
    # The key is sampled and reported with frequency j with probability
    # self._reported_weight_dist[i][j].
    self._reported_weight_dist = [[1.0]]

    # Stores the estimators used to estimate statistics (they are computed
    # iteratively together with the reported frequency distributions and are
    # stored to avoid recomputation).
    # We store all the estimators (and not just one every store_every). The
    # estimator used for a key with reported frequency i is self._estimators[i].
    # The reason is that the i-th estimator depends on all the previous
    # estimators (and not just the (i-1)-th estimator).
    self._estimators = [0.0]

    # The elements in the sample: for each key x, self.elements[x] is the
    # reported frequency (randomly chosen to satisfy the privacy constraints).
    self.elements = {}

  @classmethod
  def from_non_private(cls,
                       sample,
                       eps,
                       delta,
                       store_every=STORE_EVERY_DEFAULT):
    """Creates a private sample from a given non-private threshold sample.

    The input sample is subsampled to satisfy the differential privacy
    constraints.

    Args:
      sample: The input non-private sample
      eps: The differential privacy parameter epsilon
      delta: The differential privacy parameter delta
      store_every: A parameter that trades off the use of space and time. When
        an element is processed into the sketch, we need to compute the
        distribution of reported frequency iteratively, and this parameter
        controls how many such distributions we store (to avoid recomputation).

    Returns:
      A private sample derived from the input non-private sample.
    """
    if not isinstance(sample, ThresholdSample):
      raise TypeError(
          "Tried to create a private sample from a non-sample object")
    s = cls(sample.threshold, eps, delta, sample.sampling_method, store_every)
    for key, (weight, _) in sample.elements.items():
      # Determines whether the key should be included in the private sample.
      non_private_inclusion_prob = s.sampling_method.inclusion_prob(
          weight, s.threshold)
      weight_dist = s.compute_reported_frequency_dist(weight)
      private_inclusion_prob = 1.0 - weight_dist[0]
      if random.random() > (private_inclusion_prob /
                            non_private_inclusion_prob):
        continue
      # Randomly chooses the reported frequency for the key from the
      # distribution conditioned on the fact that the key is in the sample.
      x = random.random()
      reported_weight = 1
      while x >= weight_dist[reported_weight] / private_inclusion_prob:
        x -= weight_dist[reported_weight] / private_inclusion_prob
        reported_weight += 1
      s.elements[key] = reported_weight
    return s

  def compute_reported_frequency_dist(self, freq):
    """Computes the distribution of reported frequencies for a key.

    The distribution of reported frequencies of a key in the private sample is
    determined by its frequency and the differential privacy parameters.

    The current implementation computes the distribution by iterating from 1 to
    the frequency. To avoid recomputation, we store some of the computed values
    for future calls to this function. The computation of estimators (later used
    to compute statistics) requires iteratively computing the reported frequency
    distributions, so in order to avoid recomputation, this function also
    computes and stores the estimators (for all the possible reported
    frequencies so far).

    Args:
      freq: The frequency of the key

    Returns:
      A list of size freq+1. The first value is the probability that the key is
      not included in the private sample. The value at index j is the
      probability that the key is included with reported frequency j.
    """
    if not isinstance(freq, int):
      raise TypeError("The frequency %f should be of type int" % freq)
    if freq < 0:
      raise ValueError("The frequency %d should be nonnegative" % freq)
    # Find the closest precomputed value to start iterating from
    start_arr_index = min(freq // self._store_every,
                          len(self._reported_weight_dist) - 1)
    cur_dist = self._reported_weight_dist[start_arr_index].copy()
    cur_prob = 1.0 - cur_dist[0]
    cur_freq = start_arr_index * self._store_every
    # Invariant: at the beginning/end of each iteration, cur_dist is the
    # reported frequency distribution for a key with frequency cur_freq, and
    # cur_prob = 1.0 - cur_dist[0] (the inclusion probability for such key).
    while cur_freq < freq:
      # The pseudocode/details behind the computation below are in the writeup.
      cur_freq += 1
      cur_prob = min(
          self.sampling_method.inclusion_prob(cur_freq, self.threshold),
          math.exp(self.eps) * cur_prob + self.delta,
          1.0 + math.exp(-1.0 * self.eps) * (cur_prob + self.delta - 1))
      prev_dist = cur_dist
      cur_dist = [0.0] * (cur_freq + 1)
      cur_dist[0] = 1.0 - cur_prob
      prev_cumulative = prev_dist[0]
      cur_cumulative = cur_dist[0]
      for j in range(1, cur_freq):
        prev_cumulative += prev_dist[j]
        cur_dist[j] = max(
            0,
            math.exp(-1.0 * self.eps) * (prev_cumulative - self.delta) -
            cur_cumulative)
        cur_cumulative += cur_dist[j]
      remainder = cur_prob - (cur_cumulative - cur_dist[0])
      prev_cumulative = 0.0
      cur_cumulative = 0.0
      for j in range(cur_freq, 0, -1):
        if remainder <= 0.0:
          break
        max_prob_j = math.exp(
            self.eps) * prev_cumulative + self.delta - cur_cumulative
        amount_added = min(remainder, max_prob_j - cur_dist[j])
        cur_dist[j] += amount_added
        remainder -= amount_added
        prev_cumulative += prev_dist[j - 1]
        cur_cumulative += cur_dist[j]
      if cur_freq == len(self._reported_weight_dist) * self._store_every:
        self._reported_weight_dist.append(cur_dist.copy())
      # Computes the estimator for cur_freq if not already computed.
      if cur_freq == len(self._estimators):
        cur_est = cur_freq
        for j in range(1, cur_freq):
          cur_est -= self._estimators[j] * cur_dist[j]
        # The formula for cur_dist[cur_freq] guarantees that it is > 0.
        self._estimators.append(cur_est / cur_dist[cur_freq])
    return cur_dist

  def process(self, key, weight):
    """Processes a data element into the sketch.

    Args:
      key: The key of the element. We assume the data is aggregated, so the key
        has to be unique to this element.
      weight: The weight of this element/key
    """
    if key in self.elements:
      raise ValueError("Only works for aggregated data: repeated key %s" % key)
    weight_dist = self.compute_reported_frequency_dist(weight)
    x = random.random()
    if x < weight_dist[0]:
      return
    reported_weight = 1
    x -= weight_dist[0]
    while x >= weight_dist[reported_weight]:
      x -= weight_dist[reported_weight]
      reported_weight += 1
    self.elements[key] = reported_weight

  def estimate_full_statistics(self):
    """Estimates the full statistics/sum of the weights of the entire dataset.

    The estimate is computed by summing the estimator computed by
    compute_reported_frequency_dist for each one of the keys in the sample.

    Returns:
      An estimate for the sum of the weights of all keys.
    """
    sum_estimate = 0.0
    for reported_weight in self.elements.values():
      sum_estimate += self._estimators[reported_weight]
    return sum_estimate
