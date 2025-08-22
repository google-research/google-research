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

"""Ensemble multiple tsv files with the same shape by majority vote."""

import collections
from typing import Sequence

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_results_tsvs', None,
                    'comma-separated list of paths to input results files.')

flags.DEFINE_string(
    'output_results_tsv', None,
    'Location of the output file with decoding results. The '
    'file is in TSV format.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  file_list = FLAGS.input_results_tsvs.split(',')
  # Create a frame from the first file in the list.
  frame = []
  with open(file_list[0], 'r', encoding='utf8') as fin:
    for line in fin:
      row = line.strip('\n').split('\t')
      row = [[x] for x in row]
      frame.append(row)

  # Fill out the frame with the remaining files.
  for fname in file_list[1:]:
    with open(fname, 'r', encoding='utf8') as fin:
      for i, line in enumerate(fin):
        row = line.strip('\n').split('\t')
        for j, entry in enumerate(row):
          frame[i][j].append(entry)

  # Reduce each position to its most common entry.
  for i, row in enumerate(frame):
    for j, entry_list in enumerate(row):
      mc = collections.Counter(entry_list).most_common(1)[0][0]
      frame[i][j] = mc

  # Write the new combination file.
  with open(FLAGS.output_results_tsv, 'w', encoding='utf8') as fout:
    for row in frame:
      row = '\t'.join(row) + '\n'
      fout.write(row)


if __name__ == '__main__':
  app.run(main)
