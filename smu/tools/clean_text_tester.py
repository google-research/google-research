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
"""Compares the generated clean text output to samples."""

from dataclasses import dataclass
import difflib
import glob
import os
import re

from absl import app
from absl import flags
from absl import logging

from smu import smu_sqlite
from smu.parser import smu_writer_lib


from smu import dataset_pb2

flags.DEFINE_string('sqlite_standard', None, 'Standard SQLite file')
flags.DEFINE_string('sqlite_complete', None, 'Complete SQLite file')
flags.DEFINE_string('test_files_dir', None, 'Root directory to look for test files')
flags.DEFINE_string('output_dir', None, 'Root directory to write outputs and diffs to')

FLAGS = flags.FLAGS


@dataclass
class MatchResult:
  cnt_standard: int
  cnt_complete: int
  cnt_standard_error: int
  cnt_complete_error: int

  def add(this, other):
    return MatchResult(
      this.cnt_standard + other.cnt_standard,
      this.cnt_complete + other.cnt_complete,
      this.cnt_standard_error + other.cnt_standard_error,
      this.cnt_complete_error + other.cnt_complete_error)


SEPARATOR_LINE = "#==============================================================================="

def parse_expected_file(fn):
  out = []
  with open(fn, 'r') as f:
    pending = []
    for line in f:
      if not pending or not line.startswith(SEPARATOR_LINE):
        pending.append(line)
      else:
        out.append(pending)
        pending = [line]
    out.append(pending)
  return out


SAMPLES_REGEX = re.compile(r'(\d{6})\.(\d{3})')
def parse_samples_file(fn):
  with open(fn, 'r') as f:
    for line in f:
      line = line.strip()
      match = SAMPLES_REGEX.match(line)
      if not match:
        raise ValueError(f'In {fn} could not parse "{line}"')
      yield int(match.group(1)) * 1000 + int(match.group(2))


def process_one_samples_file(samples_fn, db_standard, db_complete):
  expected_standard = parse_expected_file(os.path.join(os.path.dirname(samples_fn),
                                                       'smu_db_standard.out'))
  expected_standard_idx = 0
  expected_complete = parse_expected_file(os.path.join(os.path.dirname(samples_fn),
                                                       'smu_db_complete.out'))
  expected_complete_idx = 0

  for mol_id in parse_samples_file(samples_fn):
    print(mol_id)

  return MatchResult(0, 0, 0, 0)


def process_one_expected(samples_fn, db, is_standard):
  logging.info(f'Processing {samples_fn} is_standard={is_standard}')
  writer = smu_writer_lib.CleanTextWriter()
  result = MatchResult(0, 0, 0, 0)

  file_keyword = 'standard' if is_standard else 'complete'
  expected = parse_expected_file(os.path.join(os.path.dirname(samples_fn),
                                              f'smu_db_{file_keyword}.out'))
  expected_idx = 0

  prefix_path = os.path.commonprefix([FLAGS.test_files_dir, samples_fn])
  suffix_path = os.path.dirname(samples_fn[len(prefix_path):])
  if suffix_path == '/':
    this_output_dir = FLAGS.output_dir
  else:
    this_output_dir = os.path.join(FLAGS.output_dir, suffix_path)
  os.makedirs(this_output_dir, exist_ok=True)
  with (open(os.path.join(this_output_dir, f'{file_keyword}.out'), 'w') as out_file,
        open(os.path.join(this_output_dir, f'{file_keyword}.diff'), 'w') as diff_file):
    for mol_id in parse_samples_file(samples_fn):
      try:
        mol = db.find_by_molecule_id(mol_id)
        actual = writer.process(mol)
      except KeyError:
        if is_standard:
          continue
        actual = f'Could not find mol_id {mol_id}\n'

      out_file.write(actual)

      diff_lines = difflib.unified_diff(expected[expected_idx],
                                        actual.splitlines(keepends=True),
                                        fromfile='expected',
                                        tofile='actual')
      diff = False
      for line in diff_lines:
        diff = True
        diff_file.write(line)
      if is_standard:
        result.cnt_standard += 1
        if diff:
          result.cnt_standard_error += 1
      else:
        result.cnt_complete += 1
        if diff:
          result.cnt_complete_error += 1

      expected_idx += 1

  return result


def main(unused_argv):
  samples_files = glob.glob(f'{FLAGS.test_files_dir}/**/samples.dat', recursive=True)
  if not samples_files:
    raise ValueError(f'No samples.dat foudnd in {FLAGS.test_files_dir}')
  logging.info(f'Found samples.dat files: {samples_files}')

  db_standard = smu_sqlite.SMUSQLite(FLAGS.sqlite_standard, 'r')
  db_complete = smu_sqlite.SMUSQLite(FLAGS.sqlite_complete, 'r')

  total_result = MatchResult(0, 0, 0, 0)
  for samples_fn in samples_files:
    result_standard = process_one_expected(samples_fn, db_standard, True)
    result_complete = process_one_expected(samples_fn, db_complete, False)
    logging.info(f'Results for {samples_fn}: {result_standard} {result_complete}')
    total_result = total_result.add(result_standard)
    total_result = total_result.add(result_complete)

  logging.info(f'Final result {total_result}')
  print(f'Processed {len(samples_files)}')
  print(f'Standard errors {total_result.cnt_standard_error} / {total_result.cnt_standard}')
  print(f'Complete errors {total_result.cnt_complete_error} / {total_result.cnt_complete}')




if __name__ == '__main__':
  app.run(main)
