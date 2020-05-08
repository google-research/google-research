# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Library for computing and writing accuracy results."""

import logging
from typing import List, Tuple, Optional

import dataclasses

from tensorflow.compat.v1.io import gfile


@dataclasses.dataclass
class AccuracyResult(object):
  total_lines: int
  matches: List[Tuple[str, str]]
  mismatches: List[Tuple[str, str, str]]
  inferred_answers_path: str

  def get_accuracy(self):
    return len(self.matches) / self.total_lines


def write_accuracy_result(result,
                          output_path,
                          print_output = False):
  """Writes the accuracy results to a text file."""
  if not result:
    return
  accuracy = result.get_accuracy()
  summary = f'Accuracy on {result.inferred_answers_path} is {accuracy}'
  with gfile.GFile(output_path, 'w') as f:
    f.write(f'{summary}\n')
    if result.mismatches:
      f.write('\n==========WRONG==========\n')
    for question, golden, inferred in result.mismatches:
      f.write(f'Q: {question}Gold: {golden}Inferred: {inferred}\n')
    if result.matches:
      f.write('\n==========CORRECT==========\n')
    for question, golden in result.matches:
      f.write(f'Q: {question}Gold/Inferred: {golden}\n')
  if print_output:
    print(f'Evaluation result written to {output_path}\n')
    print(summary)


def get_accuracy_result(
    questions_path, golden_answers_path,
    inferred_answers_path):
  """Collect accuracy results from input files."""
  questions = gfile.GFile(questions_path).readlines()
  golden_answers = gfile.GFile(golden_answers_path).readlines()
  inferred_answers = gfile.GFile(inferred_answers_path).readlines()

  result = AccuracyResult(
      total_lines=len(questions),
      matches=[],
      mismatches=[],
      inferred_answers_path=inferred_answers_path)
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
