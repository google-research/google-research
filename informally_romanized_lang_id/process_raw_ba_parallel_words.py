# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
r"""Convert json from bhasa-abhijnaanam to tsv.
"""

import glob
import json
import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('json_path', '', 'Input file')


# Fix messed up encoding.
def fix_enc(string):
  return bytes([ord(u) for u in string]).decode('utf8', errors='replace')


def main(unused_argv):
  sys.stdout.reconfigure(encoding='utf-8')
  parsed_json = []
  with sys.stdout as fout:
    for path in glob.glob(FLAGS.json_path):
      data_json = open(path, 'r', encoding='utf-8')
      parsed_json.extend([json.loads(line) for line in data_json.readlines()])
      for jline in parsed_json:
        for item in jline['data']:
          native_words = item['native sentence'].split()
          roman_words = item['romanized sentence'].split()
          if len(native_words) == len(roman_words):
            # Ignores strings with different number of native/romanized words.
            for i in range(0, len(native_words)):
              if native_words[i] != roman_words[i]:
                # Ignores identity pairs, i.e., words that were not romanized.
                if native_words[i][:-1] == roman_words[i][:-1]:
                  # Emits just last char if identical longest proper prefix.
                  native_out = native_words[i][-1]
                  roman_out = roman_words[i][-1]
                elif native_words[i][1:] == roman_words[i][1:]:
                  # Emits just first char if indentical longest proper suffix.
                  native_out = native_words[i][0]
                  roman_out = roman_words[i][0]
                else:
                  # Emits whole tokens.
                  native_out = native_words[i]
                  roman_out = roman_words[i]
                print(item['language'],
                      native_out,
                      roman_out,
                      file=fout, sep='\t')


if __name__ == '__main__':
  app.run(main)
