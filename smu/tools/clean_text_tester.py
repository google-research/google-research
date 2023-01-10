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

import dataclasses
import difflib
import glob
import math
import os
import re

from absl import app
from absl import flags
from absl import logging
from smu import smu_sqlite
from smu.parser import smu_writer_lib

flags.DEFINE_string('sqlite_standard', None, 'Standard SQLite file')
flags.DEFINE_string('sqlite_complete', None, 'Complete SQLite file')
flags.DEFINE_string('test_files_dir', None,
                    'Root directory to look for test files')
flags.DEFINE_string('output_dir', None,
                    'Root directory to write outputs and diffs to')

FLAGS = flags.FLAGS


@dataclasses.dataclass
class MatchResult:
  """Match result.

  Attributes:
    cnt_standard:
    cnt_complete:
    cnt_standard_error:
    cnt_complete_error:
  """
  cnt_standard: int
  cnt_complete: int
  cnt_standard_error: int
  cnt_complete_error: int

  def add(self, other):
    """Adder.

    Args:
      other:

    Returns:

    """
    return MatchResult(
        self.cnt_standard + other.cnt_standard,
        self.cnt_complete + other.cnt_complete,
        self.cnt_standard_error + other.cnt_standard_error,
        self.cnt_complete_error + other.cnt_complete_error)


class SmuLineForDiff(str):
  """SmuLineForDiff.

  """

  def __hash__(self):
    return super().__hash__()

  def _tokenize(self):
    out = []
    parts = self.__str__().split()
    curr_idx = 0
    for part in parts:
      loc = self.__str__().index(part, curr_idx)
      out.append((loc, part))
      curr_idx = loc + len(part)
    return out

  def _num_sig(self, float_str):
    try:
      return len(float_str) - float_str.index('.') - 1
    except ValueError:
      return 0

  def __eq__(self, other):
    if not isinstance(other, SmuLineForDiff):
      return False

    if self.__str__() == other.__str__():
      return True

    if len(self.__str__()) != len(other.__str__()):
      return False

    tokens = self._tokenize()
    other_tokens = other._tokenize()

    if len(tokens) != len(other_tokens):
      return False

    for tok, other_tok in zip(tokens, other_tokens):
      try:
        val = float(tok[1])
        other_val = float(other_tok[1])

        # We have floating point values! First let's check for the -0 case
        if val == other_val:
          if val != 0.0 and tok[0] == other_tok[0]:
            continue
          # They aren't in the same position, let's check the -0 case
          new_tok_pos = tok[0]
          new_other_tok_pos = other_tok[0]
          if math.copysign(1.0, val) == -1.0:
            new_tok_pos += 1
          if math.copysign(1.0, other_val) == -1.0:
            new_other_tok_pos += 1
          if new_tok_pos != new_other_tok_pos:
            return False
        else:
          # Note that we are not going to handle the case where the adjustment
          # in the last digit causes the string tobe longer.
          if tok[0] != other_tok[0] or len(tok[1]) != len(other_tok[1]):
            return False

          num_sig = self._num_sig(tok[1])
          if num_sig == 0 or num_sig != self._num_sig(other_tok[1]):
            return False
          format_spec = '{:.' + str(num_sig) + 'f}'
          delta = 10**(-num_sig)
          if (format_spec.format(val + delta) == format_spec.format(other_val) or
              format_spec.format(val - delta) == format_spec.format(other_val)):
            continue

          return False

      except ValueError:
        # These are not floating point, just check equality.
        if tok != other_tok:
          return False

    return True

  def __ne__(self, other):
    return not self.__eq__(other)


SEPARATOR_LINE = '#' + '=' * 79


def parse_expected_file(fn):
  """Parse expected file.

  Args:
    fn:

  Returns:

  """
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
  """Parse samples file.

  Args:
    fn:

  Yields:

  Raises:
    <Any>:
  """
  with open(fn, 'r') as f:
    for line in f:
      line = line.strip()
      match = SAMPLES_REGEX.match(line)
      if not match:
        raise ValueError(f'In {fn} could not parse "{line}"')
      yield int(match.group(1)) * 1000 + int(match.group(2))


def process_one_expected(samples_fn, db, is_standard):
  """Process one.

  Args:
    samples_fn:
    db:
    is_standard:

  Returns:

  """
  logging.info('Processing %s is_standard=%s',
               samples_fn, 'True' if is_standard else 'False')
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
        mol = db.find_by_mol_id(mol_id)
        actual = writer.process(mol)
      except KeyError:
        if is_standard:
          continue
        actual = f'Could not find mol_id {mol_id}\n'

      out_file.write(actual)

      diff_lines = list(difflib.unified_diff(
          [SmuLineForDiff(s) for s in expected[expected_idx]],
          [SmuLineForDiff(s) for s in actual.splitlines(keepends=True)],
          fromfile='expected',
          tofile='actual'))
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
  logging.info('Found samples.dat files: %s', samples_files)

  db_standard = smu_sqlite.SMUSQLite(FLAGS.sqlite_standard, 'r')
  db_complete = smu_sqlite.SMUSQLite(FLAGS.sqlite_complete, 'r')

  total_result = MatchResult(0, 0, 0, 0)
  for samples_fn in samples_files:
    result_standard = process_one_expected(samples_fn, db_standard, True)
    result_complete = process_one_expected(samples_fn, db_complete, False)
    logging.info('Results for %s: %s %s', samples_fn, result_standard, result_complete)
    total_result = total_result.add(result_standard)
    total_result = total_result.add(result_complete)

  logging.info('Final result %s', total_result)
  print('Processed %d', len(samples_files))
  print('Standard errors %d / %d', total_result.cnt_standard_error, total_result.cnt_standard)
  print('Complete errors %d / %d', total_result.cnt_complete_error, total_result.cnt_complete)


if __name__ == '__main__':
  app.run(main)
