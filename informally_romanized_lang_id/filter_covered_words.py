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

# -*- coding: utf-8 -*-
r"""Filters pairs from input lexicon with only covered characters in baseline.
"""

import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('input_lexicon', '', 'Input lexicon file')

flags.DEFINE_string('baseline_lexicon', '', 'Baseline lexicon file')


def main(unused_argv):
  # Extracts native script characters from baseline lexicon.
  sys.stdout.reconfigure(encoding='utf-8')
  in_vocab_characters = set()
  baseline_lexicon = open(FLAGS.baseline_lexicon, 'r', encoding='utf-8')
  for line in baseline_lexicon:
    baseline_words = line.split()
    if len(baseline_words) > 1:
      baseline_chars = list(baseline_words[0])
      for baseline_char in baseline_chars:
        in_vocab_characters.add(baseline_char)
  with sys.stdout as fout:
    input_lexicon = open(FLAGS.input_lexicon, 'r', encoding='utf-8')
    for line in input_lexicon:
      input_words = line.split()
      assert len(input_words) == 2, (
          f'Unexpected line that did not have exactly two columns: {line}.')
      if len(input_words) == 2:
        input_chars = set(list(input_words[0]))
        source_has_no_ascii = not any([c.isascii() for c in input_chars])
        if input_words[1].isascii() and source_has_no_ascii:
          # Ignores pairs with ASCII in source or non-ASCII in target.
          oov_string = False
          for input_char in input_chars:
            if input_char not in in_vocab_characters:
              # Character not found in output, hence should be transliterated.
              oov_string = True
              break
          if oov_string:
            print(input_words[0],
                  input_words[1],
                  file=fout, sep='\t')


if __name__ == '__main__':
  app.run(main)
