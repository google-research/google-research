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

"""Utilities for statistical significance testing."""

from absl import flags

import statsmodels.stats.api as sms

flags.DEFINE_string(
    "alternative_hypothesis", "two-sided",
    "Defines the alternative hypothesis. Let F(u) and G(u) be the cumulative "
    "distribution functions of the distributions underlying x and y, "
    "respectively. Then the following alternative hypotheses are available:\n"
    "`two-sided`: the distributions are not equal, i.e. F(u) ≠ G(u) for at "
    "least one u.\n"
    "`less`: the distribution underlying x is stochastically less than the "
    "distribution underlying y, i.e. F(u) > G(u) for all u."
    "`greater': the opposite of `less`.")

FLAGS = flags.FLAGS

# Variance option:
# If `pooled`, then the standard deviation of the samples is assumed to be the
# same. If `unequal`, then the variance of Welch $t$-test will be used, and the
# degrees of freedom are those of Satterthwaite.
_VARIANCE_OPTION = "unequal"


class ParameterStats(object):
  """A summary of a statistical parameter's value.

  Attributes:
    mean: A numeric value for value's mean.
    confidence_level: The probability that the value of a parameter falls within
      the range of values given by confidence_interval.
    confidence_interval: A tuple of two numeric values for the range in which
      the value lies with the specified confidence_level.
    t_test: Tuple consisting of t-statistic, p-value and degrees of freedom
      used in the test.
  """

  @classmethod
  def MeanDifference(cls, base_samples, test_samples, confidence_level=0.95):
    """Creates a ParameterStats for the difference in means between samples."""

    test_samples_stats = sms.DescrStatsW(test_samples)
    base_samples_stats = sms.DescrStatsW(base_samples)

    delta_mean = test_samples_stats.mean - base_samples_stats.mean
    if delta_mean < 0:
      # Improvement over the baseline.
      delta_mean_percent = delta_mean / base_samples_stats.mean
    else:
      delta_mean_percent = delta_mean / test_samples_stats.mean
    delta_mean_percent *= 100.0

    alpha = 1 - confidence_level
    cm = sms.CompareMeans(test_samples_stats, base_samples_stats)
    confidence_interval = cm.tconfint_diff(usevar=_VARIANCE_OPTION, alpha=alpha)
    t_test = cm.ttest_ind(usevar=_VARIANCE_OPTION)
    return cls(delta_mean, delta_mean_percent,
               1 - alpha, confidence_interval, t_test)

  def __init__(self, mean, mean_percent, confidence_level, confidence_interval,
               t_test):
    self.mean = mean
    self.mean_percent = mean_percent
    self.confidence_level = confidence_level
    self.confidence_interval = confidence_interval
    self.t_statistic = t_test[0]
    self.p_value = t_test[1]
    self.t_dof = t_test[2]

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __str__(self):
    return ("{:+.5f} ({:+.2f}%), {:g}% CI = [{:+.5f}, {:+.5f}] "
            "t-statistic: {:+.5f}, p-value: {:+.5f}, t-dof: {:+.5f}").format(
                self.mean, self.mean_percent, self.confidence_level * 100.0,
                self.confidence_interval[0],
        self.confidence_interval[1], self.t_statistic, self.p_value, self.t_dof)
