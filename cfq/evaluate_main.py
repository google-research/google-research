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

"""Given a list of questions, compare golden answers with inferred answers.

Writes accuracy (fraction of answers correct), and writes all correct and
incorrect output.
"""
import os

from absl import app
from absl import flags

import evaluate as evaluator

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  accuracy_result = evaluator.get_accuracy_result(FLAGS.questions_path,
                                                  FLAGS.golden_answers_path,
                                                  FLAGS.inferred_answers_path)
  evaluator.write_accuracy_result(
      accuracy_result, FLAGS.output_path, print_output=True)


if __name__ == '__main__':
  app.run(main)
