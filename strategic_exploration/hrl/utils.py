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

def mean_with_default(l, default_val):
  """Returns the mean of the list l.

  If l is empty, returns default_val instead

    Args: l (iterable[float | int]) default_val
  """
  if len(l) == 0:
    return default_val
  else:
    return float(sum(l)) / len(l)
