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

r"""The main program for evaluating the document extraction results.

This utility can be used to evaluate multiple tasks on one dataset at the same
time depending on the number of extraction JSON files that exist in the given
`_EXTRACTION_PATH` by providing the micro- and macro-F1 scores of each task.
"""

from absl import app
from absl import flags
import pandas as pd

from vrdu import evaluate_utils


FLAGS = flags.FLAGS

_BASE_DIRPATH = flags.DEFINE_string(
    'base_dirpath', None, 'File path of dataset and splits.', short_name='b')
_EXTRACTION_PATH = flags.DEFINE_string(
    'extraction_path', None, 'Path of the extraction results, where the '
    'extraction outputs of JSON format can be found. Each JSON file corresponds'
    'to a task (split) so the file name is supposed to start with the split '
    'name and end with `-test_predictions.json`.', short_name='e')
_EVAL_OUTPUT_PATH = flags.DEFINE_string(
    'eval_output_path', None, 'Path to save the eval outputs. A single file'
    'will be written with all the results.', short_name='o')


def main(argv):

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load model extractions and the split files.
  ground_truth, experiments = evaluate_utils.load_experiments(
      _BASE_DIRPATH.value, _EXTRACTION_PATH.value)

  # Evaluate model extractions.
  evals = evaluate_utils.evaluate_experiments(ground_truth, experiments)

  # Save evaluation results to file.
  eval_df = (
      pd.DataFrame(evals)
      .groupby(['task', 'train_size'])[['metric-micro_f1', 'metric-macro_f1']]
      .mean()
  )
  eval_df.to_csv(open(_EVAL_OUTPUT_PATH.value, 'w'), sep='\t')


if __name__ == '__main__':
  app.run(main)
