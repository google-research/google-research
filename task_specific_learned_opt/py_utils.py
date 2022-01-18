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

"""Python utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def eqzip(*args):
  """Zip but raises error if lengths don't match.

  Args:
    *args: list of lists or tuples

  Returns:
    list: the result of zip
  Raises:
    ValueError: when the lengths don't match
  """

  sizes = [len(x) for x in args]
  if not all([sizes[0] == x for x in sizes]):
    raise ValueError("Lists are of different sizes. \n %s" % str(sizes))
  return zip(*args)
