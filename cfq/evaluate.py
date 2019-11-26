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

# Lint as: python3
"""Given a list of questions, compare golden answers with inferred answers.

Writes accuracy (fraction of answers correct), and writes all correct and
incorrect output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from typing import List, Tuple, Text, Optional

from absl import app
from absl import flags
import dataclasses

FLAGS = flags.FLAGS

flags.DEFINE_string('questions_path', None, 'Path to the input questions.')
flags.DEFINE_string('golden_answers_path', None,
                    'Path to the expected (golden) answers.')
flags.DEFINE_string('inferred_answers_path', None,
                    'Path to the inferred answers.')
flags.DEFINE_string('output_path', None, 'Path to write evaluation results to')

flags.mark_flag_as_required('output_path')
flags.register_validator('questions_path', os.path.exists,
                         'Questions path not found.')
flags.register_validator('golden_answers_path', os.path.exists,
                         'Golden answers path not found.')
flags.register_validator('inferred_answers_path', os.path.exists,
                         'Inferred answers path not found.')


@dataclasses.dataclass
class AccuracyResult(object):
  total_lines: int
  matches: List[Tuple[Text, Text]]
  mismatches: List[Tuple[Text, Text, Text]]


def write_accuracy_result(result: Optional[AccuracyResult]) -> None:
  """Writes the accuracy results to a text file."""
  if not result:
    return
  accuracy = len(result.matches) / result.total_lines
  summary = f'Accuracy on {FLAGS.inferred_answers_path} is {accuracy}'
  with open(FLAGS.output_path, 'w') as f:
    f.write(f'{summary}\n')
    if result.mismatches:
      f.write('\n==========WRONG==========\n')
    for question, golden, inferred in result.mismatches:
      f.write(f'Q: {question}Gold: {golden}Inferred: {inferred}\n')
    if result.matches:
      f.write('\n==========CORRECT==========\n')
    for question, golden in result.matches:
      f.write(f'Q: {question}Gold/Inferred: {golden}\n')
  print(f'Evaluation result written to {FLAGS.output_path}\n')
  print(summary)


def get_accuracy_result() -> Optional[AccuracyResult]:
  """Collect accuracy results from input files."""
  questions = open(FLAGS.questions_path).readlines()
  golden_answers = open(FLAGS.golden_answers_path).readlines()
  inferred_answers = open(FLAGS.inferred_answers_path).readlines()

  result = AccuracyResult(total_lines=len(questions), matches=[], mismatches=[])
  if len(set((len(questions), len(golden_answers), len(inferred_answers)))) > 1:
    logging.fatal(f'Not writing accuracy results: Input files have different '
                  'lengths')
    logging.fatal(f'Questions: {len(questions)}, golden answers: '
                  '{len(golden_answers)}, inferred answers: '
                  '{len(inferred_answers)}')
    return None
  for question, golden, inferred in zip(questions, golden_answers,
                                        inferred_answers):
    if inferred == golden:
      result.matches.append((question, golden))
    else:
      result.mismatches.append((question, golden, inferred))
  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  write_accuracy_result(get_accuracy_result())


if __name__ == '__main__':
  app.run(main)
