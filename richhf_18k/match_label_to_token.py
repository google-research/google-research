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

"""Library to match the misalignment label to the token."""

import re
from typing import List, Tuple


def match_misalignment_label_to_token(
    misalignment_label,
    prompt,
):
  """Matches the misalignment label to the token.

  Args:
    misalignment_label: The misalignment label from RichHF-18K dataset.
    prompt: The prompt from the Pick-a-pic dataset.

  Returns:
    A list of pairs of token and misalignment label.
  """
  delimiters = ',.?!":; '
  pattern = '|'.join(map(re.escape, delimiters))
  # Split by punctuation or space and remove empty tokens.
  tokens = re.split(pattern, prompt)
  tokens = [t for t in tokens if t]

  misalignment_label = misalignment_label.split(' ')
  misalignment_label = [int(l) for l in misalignment_label]
  assert len(tokens) == len(misalignment_label)
  return list(zip(tokens, misalignment_label))


if __name__ == '__main__':
  text = 'RichHF-18K: a dataset for rich human feedback on generative images.'
  label = '0 1 0 0 0 1 0 0 1 0'
  pairs = match_misalignment_label_to_token(label, text)
  print(pairs)
