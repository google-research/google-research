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

"""Helper functions for various mechanisms."""


def binary_search(function, threshold, tolerance=1e-3):
  """Returns the minimum value such that function(value) <= threshold.

  Args:
    function: A real-valued function that is decreasing in its input.
    threshold: Float threshold for function.
    tolerance: Float accuracy for computed value. Note that this errs on the
      side of being conservative.
  """
  left_input = 0
  right_input = 1
  while function(right_input) > threshold:
    right_input = 2 * right_input
    left_input = right_input / 2
  while right_input -left_input > tolerance:
    mid_input = (left_input + right_input) / 2
    if function(mid_input) <= threshold:
      right_input = mid_input
    else:
      left_input = mid_input
  return right_input
