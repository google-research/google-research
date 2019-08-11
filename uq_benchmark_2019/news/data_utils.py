# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Functions for loading dataset, fragment text into short pieces, mutating text with random words."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def fragment_into_short_sentence(text_list, label_list, fix_len, rate):
  """Randomly fragment sentences into short pieces.

  Args:
    text_list: a list of text
    label_list: a list of class labels corresponding to the text
    fix_len: the length of fragmented text
    rate: the sampling rate. the number of fragments = int(rate *
      original_text_length).

  Returns:
    A list of fragmented texts and the corresponding class labels.
  """
  n = len(text_list)
  text_frag_list = []
  label_frag_list = []

  for i in range(n):
    text_len = len(text_list[i])
    if text_len < fix_len:
      continue
    n_sample = int(rate * (text_len - fix_len + 1))
    if n_sample == 0:
      continue
    pos = np.random.choice((text_len - fix_len + 1), [n_sample])
    for j in pos:
      text_frag_list.append(text_list[i][j:(j + fix_len)])
      label_frag_list.append(label_list[i])
  return np.array(text_frag_list), np.array(label_frag_list)
