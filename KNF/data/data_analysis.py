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

"""Forecastability, trend and seasonality analysis."""


import numpy as np

from scipy.stats import entropy


def forecastabilty(ts):
  """Forecastability Measure.

  Args:
    ts: time series

  Returns:
    1 - the entropy of the fourier transformation of
          time series / entropy of white noise
  """
  ts = (ts - ts.min())/(ts.max()-ts.min())
  fourier_ts = np.fft.rfft(ts).real
  fourier_ts = (fourier_ts - fourier_ts.min()) / (
      fourier_ts.max() - fourier_ts.min())
  fourier_ts /= fourier_ts.sum()
  entropy_ts = entropy(fourier_ts)
  fore_ts = 1-entropy_ts/(np.log(len(ts)))
  if np.isnan(fore_ts):
    return 0
  return fore_ts


def forecastabilty_moving(ts, window, jump=1):
  """Calculates the forecastability of a moving window.

  Args:
    ts: time series
    window: length of slices
    jump: skipped step when taking subslices

  Returns:
    a list of forecastability measures for all slices.
  """

  # ts = Trend(ts).detrend()
  if len(ts) <= 25:
    return forecastabilty(ts)
  fore_lst = np.array([
      forecastabilty(ts[i - window:i])
      for i in np.arange(window, len(ts), jump)
  ])
  fore_lst = fore_lst[~np.isnan(fore_lst)]  # drop nan
  return fore_lst


class Trend():
  """Trend test."""

  def __init__(self, ts):
    self.ts = ts
    self.train_length = len(ts)
    self.a, self.b = self.find_trend(ts)

  def find_trend(self, insample_data):
    # fit a linear regression y=ax+b on the time series
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b

  def detrend(self):
    # remove trend
    return self.ts - (self.a * np.arange(0, len(self.ts), 1) + self.b)

  def inverse_input(self, insample_data):
    # add trend back to the input part of time series
    return insample_data + (self.a * np.arange(0, len(self.ts), 1) + self.b)

  def inverse_pred(self, outsample_data):
    # add trend back to the predictions
    return outsample_data + (
        self.a * np.arange(self.train_length,
                           self.train_length + len(outsample_data), 1) + self.b)


def seasonality_test(original_ts, ppy):
  """Seasonality test.

  Args:
    original_ts: time series
    ppy: periods per year/frequency

  Returns:
    boolean value: whether the TS is seasonal
  """

  s = acf(original_ts, 1)
  for i in range(2, ppy):
    s = s + (acf(original_ts, i)**2)

  limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

  return (abs(acf(original_ts, ppy))) > limit


def acf(ts, k):
  """Autocorrelation function.

  Args:
    ts: time series
    k: lag

  Returns:
    acf value
  """
  m = np.mean(ts)
  s1 = 0
  for i in range(k, len(ts)):
    s1 = s1 + ((ts[i] - m) * (ts[i - k] - m))

  s2 = 0
  for i in range(0, len(ts)):
    s2 = s2 + ((ts[i] - m)**2)

  return float(s1 / s2)
