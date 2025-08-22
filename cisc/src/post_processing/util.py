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

"""General post-processing utilities."""

import numpy as np


def softmax(nums, temp):
  if temp == 0:
    raise ValueError("Temperature cannot be 0.")
  nums = np.array(nums) / temp

  # Make numerics more stable (https://stackoverflow.com/q/42599498).
  # This shouldn't affect the result as sofmax(x) = sofmax(x + c).
  nums = nums - np.max(nums)

  e = np.exp(nums)
  s = e.sum()
  if s == 0:
    raise ValueError("Sum of exp(nums) is 0.")
  return e / s
