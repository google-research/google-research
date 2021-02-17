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

"""Classes representing a point."""


class Point(object):
  """Point that corresponds to a token edit operation.

  Attributes:
    point_index: The index of the next token in the sequence.
    added_phrase: A phrase that's inserted before the next token (can be empty).
  """

  def __init__(self, point_index, added_phrase=''):
    """Constructs a Point object .

    Args:
      point_index: The index the of the next token in the sequence.
      added_phrase: A phrase that's inserted before the next token.

    Raises:
      ValueError: If point_index is not an Integer.
    """

    self.added_phrase = added_phrase

    try:
      self.point_index = int(point_index)
    except ValueError:
      raise ValueError(
          'point_index should be an Integer, not {}'.format(point_index))

  def __str__(self):
    return '{}|{}'.format(self.point_index, self.added_phrase)
