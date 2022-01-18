# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Tests for differentially private sampling.

Usage:
From google-research/
python -m private_sampling.private_sampling_test

"""

import itertools
import math
from absl.testing import absltest
from absl.testing import parameterized
from private_sampling import private_sampling

# A parameter that determines the failure probability of each randomized test:
# Each random test will fail with probability at most
# 1 / FAILURE_PROBABILITY_INVERSE.
FAILURE_PROBABILITY_INVERSE = 10000000.0


class ThresholdSampleTest(absltest.TestCase):
  """Tests for the (non-private) threshold sampling class."""

  def test_samples_high_weight_elements_ppswor(self):
    """Checks that an element with high weight is sampled when using PPSWOR.

    For the fixed threshold 1.0, an element with weight w is sampled with
    probability 1-exp(-w). Hence, this test uses an element with weight
    ln(10000000), so the test is supposed to fail (element not sampled) with
    probability 1/10000000.
    """
    s = private_sampling.ThresholdSample(1.0,
                                         private_sampling.PpsworSamplingMethod)
    s.process("a", math.log(FAILURE_PROBABILITY_INVERSE, math.e))
    self.assertCountEqual(["a"], s.elements.keys())

  def test_samples_high_weight_elements_priority(self):
    """Checks that high-weight elements are sampled (using priority sampling).

    For threshold t, an element with weight at least 1/t will always
    be sampled, so this test should always succeed.
    """
    s = private_sampling.ThresholdSample(
        0.5, private_sampling.PrioritySamplingMethod)
    s.process("a", 2.0)
    s.process("b", 3.0)
    self.assertCountEqual(["a", "b"], s.elements.keys())

  def test_samples_close_to_inclusion_probability_ppswor(self):
    """Confirms sampling close to the correct inclusion probability (PPSWOR).

    The test works as follows: We create an empty sample and process n (a large
    number) elements into it, such that each element is sampled with
    probability 0.5. Then, we check that between 0.49n and 0.51n elements were
    sampled. The number n needed to ensure that the test fails with probability
    at most 1/10000000 is computed using Chernoff bounds.
    """
    # The range we allow around 0.5n
    distance_from_half = 0.01
    # The number of elements we use (computed using Chernoff bounds)
    n = int((6.0 / (distance_from_half**2)) *
            math.log(2 * FAILURE_PROBABILITY_INVERSE, math.e) + 1)
    s = private_sampling.ThresholdSample(1.0,
                                         private_sampling.PpsworSamplingMethod)
    for i in range(n):
      s.process(i, math.log(2.0, math.e))
    self.assertGreaterEqual(len(s.elements), (0.5 - distance_from_half) * n)
    self.assertLessEqual(len(s.elements), (0.5 + distance_from_half) * n)

  def test_samples_close_to_inclusion_probability_priority(self):
    """Confirms sampling close to the correct inclusion probability (priority).

    The test works as follows: We create an empty sample and process n (a large
    number) elements into it, such that each element is sampled with
    probability 0.5. Then, we check that between 0.49n and 0.51n elements were
    sampled. The number n needed to ensure that the test fails with probability
    at most 1/10000000 is computed using Chernoff bounds.
    """
    # The range we allow around 0.5n
    distance_from_half = 0.01
    # The number of elements we use (computed using Chernoff bounds)
    n = int((6.0 / (distance_from_half**2)) *
            math.log(2 * FAILURE_PROBABILITY_INVERSE, math.e) + 1)
    s = private_sampling.ThresholdSample(
        0.5, private_sampling.PrioritySamplingMethod)
    for i in range(n):
      s.process(i, 1.0)
    self.assertGreaterEqual(len(s.elements), (0.5 - distance_from_half) * n)
    self.assertLessEqual(len(s.elements), (0.5 + distance_from_half) * n)

  def test_does_not_sample_twice_ppswor(self):
    """Checks that an exception is raised when processing the same key twice.

    The exception is raised when we process a key that is already in the sample
    (this event should not happen since we assume the data is aggregated).
    To implement that, we start with an element with high weight (and is thus
    sampled with high probability), and then try to add it again.
    As in test_samples_high_weight_elements_ppswor, the test fails with
    probability 1/10000000 (happens when the first element is not sampled).
    """
    with self.assertRaises(ValueError):
      s = private_sampling.ThresholdSample(
          1.0, private_sampling.PpsworSamplingMethod)
      s.process("a", math.log(FAILURE_PROBABILITY_INVERSE, math.e))
      s.process("a", 1)

  def test_does_not_sample_twice_priority(self):
    """Checks that an exception is raised when processing the same key twice.

    The exception is raised when we process a key that is already in the sample
    (this event should not happen since we assume the data is aggregated).
    To implement that, we start with an element with high weight (that is
    always sampled for priority sampling with this threshold), and then try to
    add it again.
    See test_samples_high_weight_elements_priority for why the first element
    is always sampled.
    """
    with self.assertRaises(ValueError):
      s = private_sampling.ThresholdSample(
          0.5, private_sampling.PrioritySamplingMethod)
      s.process("a", 2.0)
      s.process("a", 0.1)

  def test_does_not_sample_negligible_weight_ppswor(self):
    """Checks that a very low weight element is not sampled (with PPSWOR).

    For the fixed threshold 1.0, an element with weight w is sampled with
    probability 1-exp(-w). For this test to fail with probability 1/10000000,
    we add an element with weight ln(10000000/(10000000 - 1)) and check that the
    element was not sampled.
    """
    s = private_sampling.ThresholdSample(1.0,
                                         private_sampling.PpsworSamplingMethod)
    s.process(
        "a",
        math.log(
            FAILURE_PROBABILITY_INVERSE / (FAILURE_PROBABILITY_INVERSE - 1),
            math.e))
    self.assertEmpty(s.elements)

  def test_does_not_sample_negligible_weight_priority(self):
    """Checks that a very low weight element is not sampled (with priority).

    For the fixed threshold 1.0, an element with weight w is sampled with
    probability min{w,1}. For this test to fail with probability 1/10000000, we
    add an element with weight 1/10000000 and check that the element was not
    sampled.
    """
    s = private_sampling.ThresholdSample(
        1.0, private_sampling.PrioritySamplingMethod)
    s.process("a", 1.0 / FAILURE_PROBABILITY_INVERSE)
    self.assertEmpty(s.elements)

  def test_estimate_statistics_ppswor(self):
    """Checks the estimate for the full statistics (using PPSWOR).

    We check that the function that estimates the full statistics (sum of all
    weights) on a dataset that contains one element which is sampled with
    probability 1-1/10000000 (as in test_samples_high_weight_elements_ppswor).
    We compare the output of estimate_statistics with the estimate we should
    get when the element is sampled. Therefore, the test should fail with
    probability 1/10000000 (when the element is not sampled).
    """
    s = private_sampling.ThresholdSample(1.0,
                                         private_sampling.PpsworSamplingMethod)
    element_weight = math.log(FAILURE_PROBABILITY_INVERSE, math.e)
    s.process("a", element_weight)
    sampling_probability = (FAILURE_PROBABILITY_INVERSE -
                            1) / FAILURE_PROBABILITY_INVERSE
    self.assertEqual(s.estimate_statistics(),
                     element_weight / sampling_probability)

  def test_estimate_statistics_priority(self):
    """Checks the estimate for the full statistics (using priority sampling).

    We check the function that estimates the full statistics (sum of all
    weights) on a dataset where all the elements are sampled with probability
    1.0. As a result, the estimate for the statistics should be exactly
    accurate.

    As in test_samples_high_weight_elements_priority, the elements are sampled
    since for threshold t, an element with weight at least 1/t will always be
    sampled.
    """
    s = private_sampling.ThresholdSample(
        0.5, private_sampling.PrioritySamplingMethod)
    s.process("a", 2.0)
    s.process("b", 3.0)
    self.assertEqual(s.estimate_statistics(), 5.0)


class PrivateThresholdSampleTest(parameterized.TestCase):
  """Tests for the private threshold sampling classes."""

  @parameterized.parameters(
      itertools.product([
          private_sampling.PrivateThresholdSampleKeysOnly,
          private_sampling.PrivateThresholdSampleWithFrequencies
      ], [
          private_sampling.PpsworSamplingMethod,
          private_sampling.PrioritySamplingMethod
      ]))
  def test_low_delta_weight_one_not_sampled(self, sampling_class,
                                            sampling_method):
    """Checks that for very low delta, an element with weight 1 is not sampled.

    The motivation for that test is that the probability of including a key with
    weight 1 in a private sample can be at most delta (even if the threshold is
    high and without privacy the key is supposed to be included with high
    probability). This test fails with probability at most 1/10000000 (delta).

    Args:
      sampling_class: The private sampling class to be tested
      sampling_method: The underlying sampling method
    """
    s = sampling_class(
        threshold=100,
        eps=0.1,
        delta=1.0 / FAILURE_PROBABILITY_INVERSE,
        sampling_method=sampling_method)
    s.process(1, 1)
    self.assertEmpty(s.elements)

  @parameterized.parameters(
      itertools.product([
          private_sampling.PrivateThresholdSampleKeysOnly,
          private_sampling.PrivateThresholdSampleWithFrequencies
      ], [(private_sampling.PpsworSamplingMethod, math.log(2.0, math.e)),
          (private_sampling.PrioritySamplingMethod, 0.5)]))
  def test_high_delta_similar_to_threshold_dist(self, sampling_class,
                                                sampling_method_and_threshold):
    """Checks that for delta=1.0, private sampling is similar to non-private.

    This test is for PPSWOR and is similar to
    ThresholdSampleTest.test_samples_close_to_inclusion_probability_ppswor and
    ThresholdSampleTest.test_samples_close_to_inclusion_probability_priority.
    The motivation is that when delta is 1.0, privacy does not add constraints,
    so we can test the inclusion probability of elements in the same way we used
    for non-private sampling.

    Args:
      sampling_class: The private sampling class to be tested
      sampling_method_and_threshold: A tuple of the underlying sampling method
        and the threshold to be used
    """
    sampling_method, threshold = sampling_method_and_threshold
    # The range we allow around 0.5n
    distance_from_half = 0.01
    # The number of elements we use (computed using Chernoff bounds)
    n = int((6.0 / (distance_from_half**2)) *
            math.log(2 * FAILURE_PROBABILITY_INVERSE, math.e) + 1)
    s = sampling_class(
        threshold=threshold,
        eps=0.1,
        delta=1.0,
        sampling_method=sampling_method)
    for i in range(n):
      s.process(i, 1)
    self.assertGreaterEqual(len(s.elements), (0.5 - distance_from_half) * n)
    self.assertLessEqual(len(s.elements), (0.5 + distance_from_half) * n)

  @parameterized.parameters(
      itertools.product([
          private_sampling.PrivateThresholdSampleKeysOnly,
          private_sampling.PrivateThresholdSampleWithFrequencies
      ], [
          private_sampling.PpsworSamplingMethod,
          private_sampling.PrioritySamplingMethod
      ]))
  def test_high_delta_sample_stays_the_same(self, sampling_class,
                                            sampling_method):
    """Makes a non-private sample private, and checks it is the same (delta=1).

    This test checks the functions that create a private sample form an existing
    non-private threshold sample. When delta is 1.0, privacy does not add
    constraints, so the new private sample should contain the same elements as
    the non-private sample.

    Args:
      sampling_class: The private sampling class to be tested
      sampling_method: The underlying sampling method
    """
    s = private_sampling.ThresholdSample(0.5, sampling_method)
    for i in range(2000):
      s.process(i, 1)
    private_priority_sample = sampling_class.from_non_private(
        s, eps=0.1, delta=1.0)
    self.assertCountEqual(s.elements.keys(), private_priority_sample.elements)

  def test_valid_inclusion_probabilities(self):
    """Sanity checks on the inclusion probabilities in a private sample.

    This test contains various checks on the inclusion probabilities computed by
    the private sampling class that only returns keys:
    1. When delta is low (0.5**30), the inclusion probability of an element with
       frequency 1 is delta.
    2. When delta is 1.0, the inclusion probability is the same as in a
       non-private sample.
    3. Inclusion probabilities are between 0.0 and 1.0, and are nondecreasing in
       the frequency.
    """
    self.assertEqual(
        private_sampling.PrivateThresholdSampleKeysOnly(
            threshold=1, eps=0.1, delta=0.5**30).compute_inclusion_prob(1),
        0.5**30)
    self.assertEqual(
        private_sampling.PrivateThresholdSampleKeysOnly(
            threshold=0.5,
            eps=0.1,
            delta=1.0,
            sampling_method=private_sampling.PrioritySamplingMethod)
        .compute_inclusion_prob(1), 0.5)
    s = private_sampling.PrivateThresholdSampleKeysOnly(
        threshold=1, eps=0.1, delta=0.5**10)
    inclusion_prob = [s.compute_inclusion_prob(i) for i in range(0, 1000, 10)]
    for x in inclusion_prob:
      self.assertGreaterEqual(x, 0.0)
      self.assertLessEqual(x, 1.0)
    for i in range(len(inclusion_prob) - 1):
      self.assertGreaterEqual(inclusion_prob[i + 1], inclusion_prob[i])

  def test_valid_reported_frequency_distribution(self):
    """Checks that the distribution of reported frequencies is valid.

    Computes the distribution of frequencies that are reported (when computing
    a private sample) and checks that it is valid: all probabilities are between
    0 and 1, and they sum up to 1.
    """
    s = private_sampling.PrivateThresholdSampleWithFrequencies(
        threshold=0.5, eps=0.1, delta=0.5**20)
    freq_dists = [
        s.compute_reported_frequency_dist(i) for i in range(100, 1001, 100)
    ]
    for dist in freq_dists:
      self.assertAlmostEqual(sum(dist.values()), 1.0)
      for x in dist.values():
        self.assertGreaterEqual(x, 0.0)


if __name__ == "__main__":
  absltest.main()
