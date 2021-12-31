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

"""Library for various ways of preprocessing the text data.
"""


def trigram_letters(text):
  """Implements the trigram letter idea from Penn & Choma.

  Penn, Gerald and Travis Choma. (2006). "Quantitative methods for
  classifying writing systems." Proceedings of the North American
  Chapter of the Association for Computational Linguistics, pages
  117--120.

  Args:
    text: a string of text.

  Returns:
    list of subword trigrams from text.
  """
  trigrams = []
  for word in text.split():
    for i in range(0, len(word), 3):
      trigram = word[i:i+3]
      if trigram:
        trigrams.append(trigram)
  return trigrams
